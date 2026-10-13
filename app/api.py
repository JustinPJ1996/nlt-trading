"""The web front end's back door into the engine: a small JSON API, plus the built pages.

Run with:
    .venv/bin/python -m app.api            # http://127.0.0.1:8502

The TypeScript front end in `web/` is layout and nothing else, exactly like
`app/main.py` was: every decision still lives in `app/logic.py` and the engine.
In particular the readback -- the plain-English statement of what the system
understood, which is this product's entire safety story -- is generated here by
`nlt.translate.readback.describe` and sent as text. The browser displays it; it
never writes or rewords it.

Three rules this module keeps:

**Every route except logging in requires a login.** Same login ID and password
as the Streamlit dashboard (`app/auth.py`), and just as fail-closed: with either
missing, nobody gets in. The session is a signed, HttpOnly, SameSite=Strict
cookie, and every write must arrive as JSON, so another website cannot submit a
form here on the user's behalf.

**What is tested is what was confirmed.** `/api/translate` returns the spec with
its readback; `/api/backtest` takes that spec back and runs it as-is, never
re-reading the sentence (see `logic.backtest_spec`).

**A failure is a sentence, never a traceback.** Anything that raises becomes
`{"error": "..."}` a non-technical user can read.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from app import auth, logic, orders_view
from nlt.costs.slippage import ZERO as ZERO_SLIPPAGE
from nlt.costs.slippage import Slippage
from nlt.spec.models import StrategySpec
from nlt.store.db import Store
from nlt.translate.readback import describe_lines, format_inr

logger = logging.getLogger(__name__)

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
SECRET_FILE = Path.home() / ".config" / "nlt" / "web_session_secret"
SESSION_HOURS = 12

# Five wrong logins from one address locks that address out for fifteen minutes.
# The dashboard carries the kill switches and the Kite login form, so guessing
# has to be slow.
MAX_FAILURES = 5
LOCKOUT_SECONDS = 15 * 60

# A chart needs a few hundred points to look right; an intraday backtest can
# have tens of thousands of bars. Sending them all makes the page sluggish.
MAX_CHART_POINTS = 1200


# ------------------------------------------------------------------ plumbing


def _session_secret(path: Path = SECRET_FILE) -> str:
    """A random signing key, made once and kept outside the repo (mode 600).

    Kept in a file rather than made fresh at each start so a restart does not
    log everyone out; outside the repo because this repo is public.
    """
    try:
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_urlsafe(48)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(value)
    return value


class _Lockout:
    """Counts failed logins per address; thread-safe because routes run in a pool."""

    def __init__(self) -> None:
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> list[float]:
        return [t for t in self._failures.get(key, []) if now - t < LOCKOUT_SECONDS]

    def locked(self, key: str) -> bool:
        with self._lock:
            return len(self._recent(key, time.monotonic())) >= MAX_FAILURES

    def fail(self, key: str) -> None:
        with self._lock:
            now = time.monotonic()
            self._failures[key] = [*self._recent(key, now), now]

    def clear(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


class _ResultCache:
    """The last few backtests, so 'Save' records exactly the run that was shown."""

    def __init__(self, size: int = 20) -> None:
        self._items: OrderedDict[str, tuple[logic.PipelineResult, dict]] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def put(self, result: logic.PipelineResult, params: dict) -> str:
        key = uuid.uuid4().hex
        with self._lock:
            self._items[key] = (result, params)
            while len(self._items) > self._size:
                self._items.popitem(last=False)
        return key

    def get(self, key: str) -> tuple[logic.PipelineResult, dict] | None:
        with self._lock:
            return self._items.get(key)


def _client_key(request: Request) -> str:
    # Behind a Cloudflare tunnel every request arrives from 127.0.0.1; the real
    # address is in this header. The server only listens on 127.0.0.1, so the
    # header cannot be supplied by anyone who bypasses the tunnel.
    return request.headers.get("cf-connecting-ip") or (
        request.client.host if request.client else "unknown"
    )


def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


async def _json_body(request: Request) -> dict:
    """The request body as a dict; anything that is not a JSON object is refused.

    Requiring `application/json` is what stops a plain HTML form on another site
    from posting here: a browser will not send that content type cross-site
    without asking this server first, and this server never says yes.
    """
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise _BadRequest("Requests must be sent as JSON.")
    try:
        body = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise _BadRequest("That request was not valid JSON.") from None
    if not isinstance(body, dict):
        raise _BadRequest("That request was not a JSON object.")
    return body


class _BadRequest(Exception):
    pass


def _authed(handler):
    """Every route but login goes through this: no session, no answer."""

    async def wrapped(request: Request) -> Response:
        if not request.session.get("user"):
            return _error("Please log in.", 401)
        try:
            return await handler(request)
        except _BadRequest as exc:
            return _error(str(exc))
        except Exception as exc:  # noqa: BLE001 - the browser must never see a traceback
            logger.exception("API route %s failed", request.url.path)
            return _error(f"Something went wrong: {exc}", 500)

    return wrapped


def _store(request: Request) -> Store:
    return request.app.state.store


async def _run(fn, *args, **kwargs):
    """Runs blocking engine code off the event loop, so one slow backtest freezes nothing else."""
    from starlette.concurrency import run_in_threadpool

    return await run_in_threadpool(fn, *args, **kwargs)


# --------------------------------------------------------------------- login


async def login(request: Request) -> Response:
    try:
        body = await _json_body(request)
    except _BadRequest as exc:
        return _error(str(exc))
    key = _client_key(request)
    lockout: _Lockout = request.app.state.lockout
    if lockout.locked(key):
        return _error(
            "Too many wrong attempts. Wait 15 minutes, then try again.",
            429,
        )
    expected_id = auth.configured_login_id()
    expected_password = auth.configured_password()
    if expected_id is None or expected_password is None:
        return _error(
            "No login ID or password is set up on this computer, so nobody can log in. "
            "Ask Claude to set one.",
            503,
        )
    entered_id = str(body.get("login_id") or "")
    entered_password = str(body.get("password") or "")
    if not auth.login_matches(entered_id, entered_password, expected_id, expected_password):
        lockout.fail(key)
        return _error("Wrong login ID or password.", 401)
    lockout.clear(key)
    request.session.clear()
    request.session["user"] = expected_id
    request.session["at"] = int(time.time())
    return JSONResponse({"user": expected_id})


async def logout(request: Request) -> Response:
    request.session.clear()
    return JSONResponse({"ok": True})


async def me(request: Request) -> Response:
    user = request.session.get("user")
    if not user:
        return _error("Please log in.", 401)
    return JSONResponse({"user": user})


# ------------------------------------------------------------------ building


async def options(request: Request) -> Response:
    today = dt.date.today()
    return JSONResponse(
        {
            "examples": logic.EXAMPLE_STRATEGIES,
            "cost_models": list(logic.COST_MODEL_LABELS),
            "default_capital": logic.DEFAULT_CAPITAL,
            "default_start": (today - dt.timedelta(days=365 * 3)).isoformat(),
            "default_end": today.isoformat(),
        }
    )


def _spec_payload(
    spec: StrategySpec,
    sources: dict[str, str] | None = None,
    answers: dict[str, str] | None = None,
) -> dict:
    """The spec and its readback. `sources` (from the reader) adds a label per
    readback line -- you said / we worked out / default. Without them, as for a
    saved strategy, there are no labels rather than guessed ones."""
    return {
        "spec": spec.model_dump(mode="json"),
        "readback": logic.describe_spec(spec),
        "readback_labels": (
            [label for _h, _line, label in describe_lines(spec, sources, answers)]
            if sources
            else None
        ),
        # Sent back with form edits so the labels can be kept (see /api/adjust).
        "sources": sources or None,
        "name": spec.name,
        "symbol": spec.instrument.symbol,
        "timeframe": spec.instrument.timeframe,
    }


async def translate(request: Request) -> Response:
    body = await _json_body(request)
    text = str(body.get("text") or "").strip()
    if not text:
        raise _BadRequest("Describe a strategy first.")
    answers = body.get("answers") or {}
    if not isinstance(answers, dict):
        raise _BadRequest("Answers must be a list of field: answer pairs.")
    answers = {str(k): str(v) for k, v in answers.items() if str(v).strip()}
    tr = await _run(logic.translate_text, text, answers)
    return JSONResponse(
        {
            "understood": (
                _spec_payload(tr.spec, tr.sources, answers) if tr.spec is not None else None
            ),
            "questions": [
                {
                    "field": q.field,
                    "text": q.text,
                    "why": q.why,
                    "suggestion": q.suggestion,
                    "choices": [{"label": c.label, "value": c.value} for c in q.choices],
                    "answerable": q.answerable,
                }
                for q in tr.questions
                if q.field != "unparsed"
            ],
            "unparsed": list(tr.unparsed),
            "notes": list(tr.notes),
            "read_by_ai": tr.source != "rules",
            "answers": answers,
            "period": (
                logic.period_dates(
                    tr.period, dt.date.today(), data_end=logic.option_data_end(tr.spec)
                )
                if tr.period is not None
                else None
            ),
        }
    )


def _before_after(after: dict, before_run) -> list[dict] | None:
    """The same strategy with and without charges, side by side.

    None when there is nothing to compare: no second run, or no charges paid.
    """
    if before_run is None or not (after.get("total_charges") or 0):
        return None
    before = before_run.metrics
    rows = [
        ("Total return", "total_return_pct", logic.format_signed_pct),
        ("Biggest drop from a peak", "max_drawdown_pct", logic.format_pct),
        ("Trades that made money", "win_rate_pct", logic.format_pct),
        ("Number of trades", "total_trades", lambda v: str(int(v or 0))),
    ]
    out = [
        {"label": label, "before": fmt(before.get(key)), "after": fmt(after.get(key))}
        for label, key, fmt in rows
    ]
    out.append(
        {
            "label": "Charges paid",
            "before": format_inr(0),
            "after": format_inr(after.get("total_charges") or 0),
        }
    )
    return out


def _parse_spec(raw) -> StrategySpec:
    try:
        return StrategySpec.model_validate(raw)
    except ValidationError:
        raise _BadRequest(
            "That strategy could not be read back. Check it again from the start."
        ) from None


def _parse_slippage(raw) -> Slippage:
    """A run's slippage setting from a request body. Missing means zero.

    `raw` is `{"kind": "pct"|"rs", "value": number}`; anything else, or a
    value `Slippage` itself rejects (negative, or past the sane ceiling), is a
    400 with the plain-English reason, never a server error.
    """
    if raw in (None, ""):
        return ZERO_SLIPPAGE
    if not isinstance(raw, dict):
        raise _BadRequest("Slippage must be a kind and an amount.")
    kind = raw.get("kind")
    try:
        value = float(raw.get("value"))
    except (TypeError, ValueError):
        raise _BadRequest("The slippage amount must be a number.") from None
    try:
        return Slippage(kind=kind, value=value)
    except ValueError as exc:
        raise _BadRequest(str(exc)) from None


def _parse_date(raw, label: str) -> dt.date | None:
    if raw in (None, ""):
        return None
    try:
        return dt.date.fromisoformat(str(raw))
    except ValueError:
        raise _BadRequest(f"The {label} date is not a real date.") from None


def _downsample(frame: pd.DataFrame) -> pd.DataFrame:
    """At most MAX_CHART_POINTS rows: each bucket's last value, but its *worst* drawdown.

    Taking every n-th row would usually skip the deepest drawdown bar, and the
    chart would then disagree with the 'biggest drop' number printed under it.
    """
    if len(frame) <= MAX_CHART_POINTS:
        return frame
    bucket = (pd.Series(range(len(frame)), index=frame.index) * MAX_CHART_POINTS) // len(frame)
    grouped = frame.groupby(bucket.values)
    out = grouped.last()
    out["time"] = grouped["time"].last()
    for col in ("strategy_dd", "benchmark_dd"):
        out[col] = grouped[col].min()
    return out


def _chart(bt_equity: pd.Series, bench_equity: pd.Series) -> list[dict]:
    frame = pd.DataFrame({"strategy": bt_equity, "benchmark": bench_equity}).sort_index().ffill()
    frame = frame.dropna()
    frame["strategy_dd"] = (frame["strategy"] / frame["strategy"].cummax() - 1) * 100
    frame["benchmark_dd"] = (frame["benchmark"] / frame["benchmark"].cummax() - 1) * 100
    frame["time"] = [pd.Timestamp(t).isoformat() for t in frame.index]
    frame = _downsample(frame.reset_index(drop=True))
    return [
        {
            "t": row.time,
            "s": round(float(row.strategy), 2),
            "b": round(float(row.benchmark), 2),
            "sd": round(float(row.strategy_dd), 3),
            "bd": round(float(row.benchmark_dd), 3),
        }
        for row in frame.itertuples(index=False)
    ]


def _time_text(ts, timeframe: str) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("Asia/Kolkata")
    return t.strftime("%d %b %Y") if timeframe == "1d" else t.strftime("%d %b %Y, %H:%M")


def _result_payload(result: logic.PipelineResult, result_id: str, params: dict) -> dict:
    bt, bench, comp, verdict = result.backtest, result.benchmark, result.comparison, result.verdict
    spec = bt.spec
    m = bt.metrics
    tf = spec.instrument.timeframe
    spotted = set(result.weak_spots)
    notes = [n for n in dict.fromkeys([*result.data_notes, *bt.warnings]) if n not in spotted]
    return {
        "result_id": result_id,
        "tested": _spec_payload(result.spec),
        "params": params,
        "verdict": {
            "summary": verdict.summary,
            "passed": verdict.passed,
            "flags": [
                {"severity": f.severity, "headline": f.headline, "detail": f.detail}
                for f in logic.flags_by_severity(verdict.flags)
            ],
        },
        "headline": {
            "strategy_return": logic.format_signed_pct(comp.strategy_return_pct),
            "benchmark_return": logic.format_signed_pct(comp.benchmark_return_pct),
            "strategy_return_raw": comp.strategy_return_pct,
            "benchmark_return_raw": comp.benchmark_return_pct,
            "benchmark_name": bench.name,
            "first": _time_text(bt.equity.index[0], "1d") if len(bt.equity) else "",
            "last": _time_text(bt.equity.index[-1], "1d") if len(bt.equity) else "",
        },
        "metrics": [
            {
                "label": "Total return",
                "strategy": logic.format_signed_pct(m.get("total_return_pct")),
                "benchmark": logic.format_signed_pct(bench.total_return_pct),
            },
            {
                "label": "Biggest drop from a peak",
                "strategy": logic.format_pct(m.get("max_drawdown_pct")),
                "benchmark": logic.format_pct(bench.max_drawdown_pct),
            },
            {
                "label": "Trades that made money",
                "strategy": logic.format_pct(m.get("win_rate_pct")),
                "benchmark": "",
            },
            {
                "label": "Number of trades",
                "strategy": str(int(m.get("total_trades") or 0)),
                "benchmark": "1",
            },
            {
                "label": "Time with money in the market",
                "strategy": logic.format_pct(comp.time_in_market_pct),
                "benchmark": "100.0%",
            },
            {
                "label": "Charges paid",
                "strategy": logic.format_inr(m.get("total_charges") or 0),
                "benchmark": "",
            },
        ],
        "before_after_charges": _before_after(m, result.before_charges),
        "assumptions": list(result.assumptions),
        "weak_spots": list(result.weak_spots),
        "all_metrics": {
            k: (None if v is None or (isinstance(v, float) and pd.isna(v)) else v)
            for k, v in m.items()
        },
        "chart": _chart(bt.equity, bench.equity),
        "trades": [
            {
                "symbol": t.symbol or spec.instrument.symbol,
                "direction": "Buy" if t.direction == "long" else "Short",
                "entered": _time_text(t.entry_time, tf),
                "entry_price": format_inr(t.entry_price),
                "exited": _time_text(t.exit_time, tf),
                "exit_price": format_inr(t.exit_price),
                "reason": logic._EXIT_WORDS.get(t.exit_reason, t.exit_reason)
                + (" (at its last traded price)" if t.exit_untraded else ""),
                "why": t.contract_why or None,
                "pnl": round(float(t.net_pnl), 2),
                "pnl_text": ("+" if t.net_pnl >= 0 else "-") + format_inr(abs(t.net_pnl)),
            }
            for t in bt.trades
        ],
        "notes": notes,
        "window_notes": list(result.window_notes),
    }


async def backtest(request: Request) -> Response:
    body = await _json_body(request)
    spec = _parse_spec(body.get("spec"))
    try:
        capital = float(body.get("capital") or logic.DEFAULT_CAPITAL)
    except (TypeError, ValueError):
        raise _BadRequest("Starting money must be a number.") from None
    if not 1_000 <= capital <= 1_000_000_000:
        raise _BadRequest("Starting money must be between Rs 1,000 and Rs 100 crore.")
    cost_label = str(body.get("cost_model") or "")
    if cost_label not in logic.COST_MODEL_LABELS:
        raise _BadRequest("Pick which trading costs to charge.")
    start = _parse_date(body.get("start"), "start")
    end = _parse_date(body.get("end"), "end")
    if start and end and start >= end:
        raise _BadRequest("The start date must be before the end date.")
    slippage = _parse_slippage(body.get("slippage"))

    result = await _run(
        logic.backtest_spec,
        spec,
        capital=capital,
        cost_model_label=cost_label,
        start=start,
        end=end,
        slippage=slippage,
    )
    if not result.ok:
        return JSONResponse(
            {"error": result.error or "The backtest did not finish.", "notes": result.data_notes},
            status_code=422,
        )
    params = {
        "capital": capital,
        "capital_text": format_inr(capital),
        "cost_model": cost_label,
        # What was really charged: shares and futures use their own charges.
        "charges_applied": logic.applied_charges_label(spec, cost_label),
        "slippage": slippage.to_dict(),
        "slippage_text": slippage.describe(),
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
    }
    result_id = request.app.state.results.put(result, params)
    return JSONResponse(_result_payload(result, result_id, params))


async def save(request: Request) -> Response:
    body = await _json_body(request)
    cached = request.app.state.results.get(str(body.get("result_id") or ""))
    if cached is None:
        raise _BadRequest(
            "That backtest is no longer in memory (the app was restarted). Run it again, then save."
        )
    result, params = cached
    strategy_id = await _run(
        logic.save_strategy,
        _store(request),
        result.spec,
        result,
        capital=params["capital"],
        cost_model_label=params["cost_model"],
        slippage=params.get("slippage"),
    )
    return JSONResponse({"strategy_id": strategy_id, "name": result.spec.name})


async def library(request: Request) -> Response:
    """The ready-made strategy cards, recomputed at most once a day."""

    def cards():
        # One computation at a time: two first visitors of the day would
        # otherwise both run every backtest.
        with request.app.state.library_lock:
            return logic.library_cards(request.app.state.library_path, dt.date.today())

    return JSONResponse(await _run(cards))


async def trades_csv(request: Request) -> Response:
    cached = request.app.state.results.get(request.path_params["rid"])
    if cached is None:
        raise _BadRequest(
            "That backtest is no longer in memory (the app was restarted). Run it again."
        )
    result, _params = cached
    name = re.sub(r"[^A-Za-z0-9]+", "-", result.spec.name).strip("-").lower() or "trades"
    return Response(
        logic.trades_csv(result),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}-trades.csv"'},
    )


async def adjust(request: Request) -> Response:
    """The confirmed strategy with the form's boxes applied, and its new readback.

    Always applied to the strategy as first read, so the boxes can be changed
    any number of times without drifting. The readback comes from the result,
    never from the boxes, so it says what will really be tested.
    """
    body = await _json_body(request)
    spec = _parse_spec(body.get("spec"))
    edits = body.get("edits") or {}
    answers = body.get("answers") or {}
    sources = body.get("sources")
    if not isinstance(edits, dict) or not isinstance(answers, dict):
        raise _BadRequest("Those changes could not be read.")
    if sources is not None and not (
        isinstance(sources, dict)
        and all(v in {"said", "worked_out", "default"} for v in sources.values())
    ):
        raise _BadRequest("Those changes could not be read.")
    try:
        adjusted, new_sources, typed = logic.adjust_spec(spec, edits, sources)
    except logic.AdjustError as exc:
        raise _BadRequest(str(exc)) from None
    answers = {str(k): str(v) for k, v in answers.items()}
    return JSONResponse(_spec_payload(adjusted, new_sources, {**answers, **typed}))


# ---------------------------------------------------------------- strategies


def _run_summary(run: dict) -> dict:
    metrics = json.loads(run.get("metrics_json") or "{}")
    params = json.loads(run.get("params_json") or "{}")
    ret = metrics.get("total_return_pct")
    return {
        "id": run["id"],
        "kind": {"backtest": "Backtest", "paper": "Paper trading", "live": "Live"}.get(
            run["mode"], run["mode"]
        ),
        "status": {
            "running": "Running",
            "complete": "Finished",
            "failed": "Failed",
            "stopped": "Stopped",
        }.get(run["status"], run["status"]),
        "started": logic.when(run["started_at"]),
        "finished": logic.when(run.get("finished_at")),
        "return_text": logic.format_signed_pct(ret) if ret is not None else "",
        "capital_text": format_inr(params["capital"]) if params.get("capital") else "",
        "error": run.get("error") or "",
    }


_STATE_WORDS = {
    "draft": "Not tested",
    "backtested": "Backtested",
    "paper": "Paper trading",
    "live": "Live",
    "halted": "Halted",
}


def _strategies(store: Store) -> list[dict]:
    out = []
    for s in logic.list_strategies_view(store):
        out.append(
            {
                "id": s["id"],
                "name": s["name"],
                "version": s["version"],
                "state": _STATE_WORDS.get(s["state"], s["state"]),
                "proven": store.is_proven(s["id"]),
                "created": logic.when(s["created_at"]),
                "description": s["description"],
                "readback": s["readback"],
                "runs": [_run_summary(r) for r in logic.list_runs_view(store, s["id"])],
            }
        )
    return out


async def strategies(request: Request) -> Response:
    return JSONResponse({"strategies": await _run(_strategies, _store(request))})


async def strategy_spec(request: Request) -> Response:
    """A saved strategy, ready to test again -- the saved spec, not a re-read of its sentence."""
    store = _store(request)
    sid = int(request.path_params["sid"])
    if store.get_strategy(sid) is None:
        return _error("No such strategy.", 404)
    spec = await _run(store.load_spec, sid)
    return JSONResponse({"description": spec.description, "understood": _spec_payload(spec)})


# ------------------------------------------------------------- paper trading


def _paper_run_card(store: Store, run: dict) -> dict:
    spec = store.load_spec(run["strategy_id"])
    strategy = store.get_strategy(run["strategy_id"])
    params = json.loads(run["params_json"] or "{}")
    status = store.paper_status(run["id"]) or {}
    snap = json.loads(status.get("snapshot_json") or "{}")
    alert_only = store.is_alert_only(run["id"])
    # The label follows the Pause button at once, not the next minute's pass.
    paused_now = run["status"] == "running" and store.is_paper_run_paused(run["id"])
    kind, label = logic.paper_health_label(
        "paused" if paused_now else status.get("health", "waiting")
    )
    positions = []
    for p in [] if alert_only else (snap.get("open_positions") or []):
        # A futures quantity already names its contract ("1 lot of Gold futures").
        symbol = "" if spec.instrument.is_future else (p.get("symbol") or "")
        pnl = p.get("pnl_if_closed_now")
        positions.append(
            {
                "what": f"{logic.quantity_words(spec, p.get('quantity'))}{symbol}".strip(),
                "direction": "Bought" if p.get("direction") == "long" else "Sold short",
                "entry_price": format_inr(p["entry_price"]) if p.get("entry_price") else "",
                "last_price": format_inr(p["last_price"]) if p.get("last_price") else "",
                "pnl": pnl,
                "pnl_text": ("+" if pnl >= 0 else "-") + format_inr(abs(pnl))
                if pnl is not None
                else "",
            }
        )
    events = store.paper_events(run["id"])
    return {
        "run_id": run["id"],
        "name": strategy["name"],
        "version": strategy["version"],
        "readback": strategy["readback"],
        "paused": store.is_paper_run_paused(run["id"]),
        "health": kind,
        "health_label": label,
        "message": status.get("message", ""),
        "started": logic.when(params.get("started_at")),
        "alert_only": alert_only,
        "capital_text": "" if alert_only else format_inr(params.get("capital", 0)),
        "checked": logic.when(status.get("updated_at")),
        "daily": spec.instrument.timeframe == "1d",
        "snapshot": (
            None
            if alert_only or not snap
            else {
                "equity": format_inr(snap.get("equity", 0)),
                "realised": format_inr(snap.get("realised_pnl", 0)),
                "open": format_inr(snap.get("open_pnl_if_closed_now", 0)),
            }
        ),
        "positions": positions,
        "events": [
            {
                "when": logic.when(e["observed_at"]),
                "text": logic.describe_paper_event(e, spec, alert_only=alert_only),
                "drift": e["kind"] == "drift",
                "trade_key": e["event_key"] if e["kind"] in ("entry", "exit") else None,
            }
            for e in reversed(events)
        ],
    }


def _paper(store: Store) -> dict:
    kite_state, kite_line = logic.kite_status()
    running = store.active_paper_runs()
    running_ids = {r["strategy_id"] for r in running}
    startable = []
    for s in logic.list_strategies_view(store):
        if s["id"] in running_ids:
            continue
        startable.append(
            {
                "id": s["id"],
                "name": s["name"],
                "version": s["version"],
                "readback": s["readback"],
                "problem": logic.paper_problem(store.load_spec(s["id"])),
            }
        )
    return {
        "kill_switch": store.paper_kill_switch_engaged(),
        "job_installed": logic.paper_job_installed(),
        "kite": {"state": kite_state, "line": kite_line, "has_login": logic.has_kite_login()},
        "runs": [_paper_run_card(store, r) for r in running],
        "startable": startable,
        "cost_models": list(logic.COST_MODEL_LABELS),
        "default_capital": logic.DEFAULT_CAPITAL,
    }


async def paper(request: Request) -> Response:
    return JSONResponse(await _run(_paper, _store(request)))


async def paper_start(request: Request) -> Response:
    body = await _json_body(request)
    alert_only = bool(body.get("alert_only"))
    cost_label = str(body.get("cost_model") or "")
    if not alert_only and cost_label not in logic.COST_MODEL_LABELS:
        raise _BadRequest("Pick which trading costs to charge.")
    try:
        sid = int(body.get("strategy_id"))
        capital = float(body.get("capital") or logic.DEFAULT_CAPITAL)
    except (TypeError, ValueError):
        raise _BadRequest("Pick a strategy and an amount of money.") from None
    # An alert-only run never trades, so its capital figure never appears
    # anywhere -- but the engine still needs one to size a position it will
    # only ever describe, never place.
    if alert_only:
        capital = logic.DEFAULT_CAPITAL
        cost_label = cost_label or next(iter(logic.COST_MODEL_LABELS))
    elif capital < 1_000:
        raise _BadRequest("Starting money must be at least Rs 1,000.")
    slippage = _parse_slippage(body.get("slippage"))
    try:
        run_id = await _run(
            logic.start_paper,
            _store(request),
            sid,
            capital=capital,
            cost_model_label=cost_label,
            alert_only=alert_only,
            slippage=slippage,
        )
    except ValueError as exc:
        raise _BadRequest(str(exc)) from None
    return JSONResponse({"run_id": run_id})


def _active_run(store: Store, run_id: int) -> dict:
    if not any(r["id"] == run_id for r in store.active_paper_runs()):
        raise _BadRequest("That paper run is not running.")
    return store.get_run(run_id)


async def paper_check(request: Request) -> Response:
    store = _store(request)
    run_id = int(request.path_params["rid"])
    _active_run(store, run_id)
    message = await _run(logic.check_paper_now, store, run_id)
    return JSONResponse({"message": message})


async def paper_stop(request: Request) -> Response:
    await _json_body(request)
    store = _store(request)
    run_id = int(request.path_params["rid"])
    _active_run(store, run_id)
    await _run(logic.stop_paper, store, run_id)
    return JSONResponse({"ok": True})


async def paper_pause(request: Request) -> Response:
    await _json_body(request)
    store = _store(request)
    run_id = int(request.path_params["rid"])
    _active_run(store, run_id)
    try:
        await _run(logic.pause_paper, store, run_id)
    except ValueError as exc:
        raise _BadRequest(str(exc)) from None
    return JSONResponse({"run": await _run(_paper_run_card, store, store.get_run(run_id))})


async def paper_resume(request: Request) -> Response:
    await _json_body(request)
    store = _store(request)
    run_id = int(request.path_params["rid"])
    _active_run(store, run_id)
    try:
        await _run(logic.resume_paper, store, run_id)
    except ValueError as exc:
        raise _BadRequest(str(exc)) from None
    return JSONResponse({"run": await _run(_paper_run_card, store, store.get_run(run_id))})


# ------------------------------------------------------------- the running drawer


def _today_ist(bar_time: str | None, today: pd.Timestamp) -> bool:
    if not bar_time:
        return False
    ts = pd.Timestamp(bar_time)
    ts = ts.tz_localize("Asia/Kolkata") if ts.tzinfo is None else ts.tz_convert("Asia/Kolkata")
    return ts.normalize() == today


def _no_money() -> dict:
    """A waiting order has no money in play yet: show a dash, never "Rs 0",
    which would read as a result."""
    return {
        "closed_today": "-",
        "closed_today_value": 0.0,
        "open_now": "-",
        "open_now_value": 0.0,
        "open_now_as_of": "",
        "total": "-",
        "total_value": 0.0,
    }


def _running(store: Store, *, now: pd.Timestamp | None = None) -> dict:
    """Everything currently running or waiting, for the header drawer: every
    paper strategy run, every waiting one-off order, every open one-off
    position. The money maths is never redone here -- 'Closed today' is summed
    straight from `paper_event` exit rows (the same rows `_paper_run_card`
    reads), 'Open now' and 'Total' come from the same snapshot
    `_paper_run_card` shows, and the one-off figures come straight from
    `orders_view.open_positions`.

    `now` is a parameter (defaulting to the real clock, in IST) rather than
    read inline, so a test can pin the exact instant "today" is judged from --
    the whole point being caught here is a day boundary that only shows up for
    a few hours around midnight IST.
    """
    now = now or pd.Timestamp.now(tz="Asia/Kolkata")
    today = now.normalize()
    items = []
    totals = {"closed_today": 0.0, "open_now": 0.0, "total": 0.0}

    for run in store.active_paper_runs():
        strategy = store.get_strategy(run["strategy_id"])
        status = store.paper_status(run["id"]) or {}
        snap = json.loads(status.get("snapshot_json") or "{}")
        paused = store.is_paper_run_paused(run["id"])
        alert_only = store.is_alert_only(run["id"])
        # The label follows the Pause button at once, not the next minute's pass.
        health = "paused" if paused else status.get("health", "waiting")
        kind, label = logic.paper_health_label(health)
        if alert_only:
            # No paper money is ever shown for an alert-only run, and none of
            # it is counted into the drawer's totals either.
            item = {
                "kind": "strategy",
                "run_id": run["id"],
                "name": f"{strategy['name']} v{strategy['version']}",
                "health": kind,
                "health_label": label,
                "paused": paused,
                "alert_only": True,
                **_no_money(),
            }
            item["open_now"] = "Alerts only"
            item["total"] = "Alerts only"
            items.append(item)
            continue
        closed_today = sum(
            (json.loads(e["detail_json"] or "{}").get("net_pnl") or 0.0)
            for e in store.paper_events(run["id"])
            if e["kind"] == "exit" and _today_ist(e["bar_time"], today)
        )
        realised = snap.get("realised_pnl", 0.0)
        open_pnl = snap.get("open_pnl_if_closed_now", 0.0)
        total = realised + open_pnl
        totals["closed_today"] += closed_today
        totals["open_now"] += open_pnl
        totals["total"] += total
        items.append(
            {
                "kind": "strategy",
                "run_id": run["id"],
                "name": f"{strategy['name']} v{strategy['version']}",
                "health": kind,
                "health_label": label,
                "paused": paused,
                "alert_only": False,
                "closed_today": orders_view._signed(closed_today),
                "closed_today_value": closed_today,
                "open_now": orders_view._signed(open_pnl),
                "open_now_value": open_pnl,
                "open_now_as_of": logic.when(status.get("last_bar_time")),
                "total": orders_view._signed(total),
                "total_value": total,
            }
        )

    orders_page = orders_view.page(store)
    for o in orders_page["orders"]:
        if o["status"] != "waiting":
            continue
        items.append(
            {
                "kind": "one_off_order",
                "order_id": o["id"],
                "name": o["sentence"],
                "health": "info",
                "health_label": o["status_label"],
                "paused": False,
                **_no_money(),
            }
        )

    unpriced = 0
    for p in orders_page["positions"]:
        known = p.get("pnl") is not None
        pnl = p["pnl"] if known else 0.0
        if known:
            totals["open_now"] += pnl
            totals["total"] += pnl
        else:
            unpriced += 1
        items.append(
            {
                "kind": "one_off_position",
                "position_id": p["id"],
                "name": p["name"],
                "health": "success",
                "health_label": "Position open",
                "paused": False,
                "closed_today": "-",
                "closed_today_value": 0.0,
                "open_now": p["pnl_text"] if known else "price not known yet",
                "open_now_value": pnl,
                "open_now_as_of": p.get("last_at", ""),
                "total": p["pnl_text"] if known else "price not known yet",
                "total_value": pnl,
            }
        )

    # One-off orders that closed today are no longer listed above, but their
    # result belongs in today's total.
    for e in store.one_off_events():
        if e["kind"] == "exited" and _today_ist(e["bar_time"], today):
            net = json.loads(e["detail_json"] or "{}").get("net_pnl")
            if net is not None:
                totals["closed_today"] += net

    label = (
        "Paper money. Closed today includes one-off orders that closed today; "
        "Total covers what is listed above; open positions are before exit charges."
    )
    if unpriced:
        label += (
            f" {unpriced} open position{'s' if unpriced != 1 else ''} without a price yet "
            f"{'are' if unpriced != 1 else 'is'} left out."
        )
    return {
        "items": items,
        "totals": {
            "closed_today": orders_view._signed(totals["closed_today"]),
            "open_now": orders_view._signed(totals["open_now"]),
            "total": orders_view._signed(totals["total"]),
            "label": label,
        },
    }


async def running(request: Request) -> Response:
    return JSONResponse(await _run(_running, _store(request)))


def _paper_trade(store: Store, run_id: int, key: str) -> list[str]:
    run = store.get_run(run_id)
    if run is None:
        raise _BadRequest("No such paper run.")
    spec = store.load_spec(run["strategy_id"])
    events = [e for e in store.paper_events(run_id) if e["event_key"] == key]
    entry = next((e for e in events if e["kind"] == "entry"), None)
    if entry is None:
        raise _BadRequest("No such trade.")
    detail = json.loads(entry.get("detail_json") or "{}")
    signal_time = detail.get("signal_time")
    trade_events = [e for e in events if e["kind"] in ("entry", "exit")]
    if signal_time:
        trade_events += [
            e
            for e in store.paper_events(run_id)
            if e["kind"] == "signal"
            and e["symbol"] == entry["symbol"]
            and e["bar_time"] == signal_time
        ]
    open_pnl = None
    open_pnl_as_of = None
    if not any(e["kind"] == "exit" for e in trade_events):
        status = store.paper_status(run_id) or {}
        open_pnl_as_of = status.get("last_bar_time")
        snap = json.loads(status.get("snapshot_json") or "{}")
        for p in snap.get("open_positions") or []:
            if f"{p.get('symbol')}|{p.get('entry_time')}" == key:
                open_pnl = p.get("pnl_if_closed_now")
                break
    return logic.trade_timeline(
        trade_events, spec, open_pnl=open_pnl, open_pnl_as_of=open_pnl_as_of
    )


async def paper_trade(request: Request) -> Response:
    store = _store(request)
    run_id = int(request.path_params["rid"])
    key = request.path_params["key"]
    steps = await _run(_paper_trade, store, run_id, key)
    return JSONResponse({"steps": steps})


async def activity(request: Request) -> Response:
    store = _store(request)
    items = await _run(logic.activity_feed, store)
    source_id = request.query_params.get("source_id")
    if source_id not in (None, ""):
        try:
            sid = int(source_id)
        except ValueError:
            raise _BadRequest("source_id must be a number.") from None
        items = [it for it in items if it["source_id"] == sid]
    if request.query_params.get("skips_only") in ("1", "true", "True"):
        items = [it for it in items if it["skip"]]
    raw_limit = request.query_params.get("limit")
    try:
        limit = int(raw_limit) if raw_limit not in (None, "") else 200
    except ValueError:
        raise _BadRequest("limit must be a number.") from None
    return JSONResponse({"items": items[: max(0, limit)]})


async def kite_login_save(request: Request) -> Response:
    body = await _json_body(request)
    try:
        await _run(
            logic.save_kite_login,
            str(body.get("user_id") or ""),
            str(body.get("password") or ""),
            str(body.get("secret") or ""),
        )
    except ValueError as exc:
        raise _BadRequest(f"Could not save that login: {exc}") from None
    return JSONResponse({"ok": True})


async def kite_login_forget(request: Request) -> Response:
    await _json_body(request)
    await _run(logic.forget_kite_login)
    return JSONResponse({"ok": True})


async def kite_token_save(request: Request) -> Response:
    body = await _json_body(request)
    token = str(body.get("token") or "").strip()
    if not token:
        raise _BadRequest("Paste a token first.")
    try:
        await _run(logic.save_kite_token, token)
    except ValueError as exc:
        raise _BadRequest(f"Could not save that token: {exc}") from None
    return JSONResponse({"ok": True})


# ------------------------------------------------------------- one-off orders


async def orders(request: Request) -> Response:
    return JSONResponse(await _run(orders_view.page, _store(request)))


async def orders_read(request: Request) -> Response:
    body = await _json_body(request)
    text = str(body.get("text") or "").strip()
    if not text:
        raise _BadRequest("Type an order first.")
    answers = body.get("answers") or {}
    if not isinstance(answers, dict):
        raise _BadRequest("Answers must be a list of field: answer pairs.")
    answers = {str(k): str(v) for k, v in answers.items() if str(v).strip()}
    return JSONResponse(await _run(orders_view.read, _store(request), text, answers))


async def orders_create(request: Request) -> Response:
    body = await _json_body(request)
    slippage = _parse_slippage(body.get("slippage"))
    try:
        order_id = await _run(
            orders_view.create,
            _store(request),
            str(body.get("text") or ""),
            {str(k): str(v) for k, v in (body.get("answers") or {}).items() if str(v).strip()}
            if isinstance(body.get("answers"), dict)
            else {},
            body.get("order"),
            body.get("positions"),
            slippage=slippage,
        )
    except orders_view.OrderError as exc:
        raise _BadRequest(str(exc)) from None
    return JSONResponse({"order_id": order_id})


async def orders_cancel(request: Request) -> Response:
    await _json_body(request)
    try:
        await _run(orders_view.cancel, _store(request), int(request.path_params["oid"]))
    except orders_view.OrderError as exc:
        raise _BadRequest(str(exc)) from None
    return JSONResponse({"ok": True})


async def orders_close(request: Request) -> Response:
    await _json_body(request)
    try:
        order_id = await _run(
            orders_view.close_now, _store(request), int(request.path_params["pid"])
        )
    except orders_view.OrderError as exc:
        raise _BadRequest(str(exc)) from None
    return JSONResponse({"order_id": order_id})


async def orders_check(request: Request) -> Response:
    await _json_body(request)
    return JSONResponse({"message": await _run(orders_view.check_now, _store(request))})


def _order_timeline(store: Store, order_id: int) -> list[str]:
    if not any(row["id"] == order_id for row in store.one_off_orders()):
        raise _BadRequest("No such order.")
    from nlt.orders import runner as orders_runner
    from nlt.orders.models import NowTrigger

    book = orders_runner.current_book(store)
    pressed = frozenset(o.id for o in book.orders if isinstance(o.order.trigger, NowTrigger))
    events = [e for e in store.one_off_events() if e["order_id"] == order_id]
    return orders_view.order_timeline(events, order_id, pressed)


async def orders_timeline(request: Request) -> Response:
    store = _store(request)
    order_id = int(request.path_params["oid"])
    steps = await _run(_order_timeline, store, order_id)
    return JSONResponse({"steps": steps})


# -------------------------------------------------------------- notifications


def _notifications(store: Store) -> dict:
    return {
        "unread_count": store.unread_notification_count(),
        "items": [
            {
                "id": n["id"],
                "when": logic.when(n["created_at"]),
                "source": n["source"],
                "source_id": n["source_id"],
                "text": n["text"],
                "read": n["read_at"] is not None,
            }
            for n in store.notifications()
        ],
    }


async def notifications(request: Request) -> Response:
    return JSONResponse(await _run(_notifications, _store(request)))


async def notifications_read(request: Request) -> Response:
    await _json_body(request)
    store = _store(request)
    await _run(store.mark_notification_read, int(request.path_params["nid"]))
    return JSONResponse(await _run(_notifications, store))


async def notifications_read_all(request: Request) -> Response:
    await _json_body(request)
    store = _store(request)
    await _run(store.mark_all_notifications_read)
    return JSONResponse(await _run(_notifications, store))


# -------------------------------------------------------------------- safety


def _safety(store: Store) -> dict:
    freshness = []
    for symbol in ("NIFTY", "BANKNIFTY"):
        days, err = logic.safe_call(lambda s=symbol: logic.staleness_days(s), on_error="")
        if err or days is None:
            freshness.append({"symbol": symbol, "level": "info", "text": "No data saved yet."})
        elif days > 3:
            freshness.append(
                {
                    "symbol": symbol,
                    "level": "critical",
                    "text": f"{days} trading days old. Refresh before trusting a new backtest.",
                }
            )
        else:
            freshness.append(
                {
                    "symbol": symbol,
                    "level": "good",
                    "text": f"Up to date ({days} trading day{'s' if days != 1 else ''} old).",
                }
            )
    trail = []
    for row in store.audit_trail(limit=50):
        detail = json.loads(row.get("detail_json") or "{}")
        trail.append(
            {
                "when": logic.when(row["at"]),
                "event": row["event"].replace("_", " "),
                "detail": ", ".join(f"{k}: {v}" for k, v in detail.items()),
            }
        )
    return {
        "live": store.kill_switch_engaged(),
        "paper": store.paper_kill_switch_engaged(),
        "freshness": freshness,
        "audit": trail,
    }


async def safety(request: Request) -> Response:
    return JSONResponse(await _run(_safety, _store(request)))


async def kill_switch(request: Request) -> Response:
    body = await _json_body(request)
    store = _store(request)
    which, action = request.path_params["which"], request.path_params["action"]
    actions = {
        ("live", "engage"): lambda: store.engage_kill_switch("engaged from web dashboard"),
        ("live", "release"): lambda: store.release_kill_switch("released from web dashboard"),
        ("paper", "engage"): lambda: store.engage_paper_kill_switch("engaged from web dashboard"),
        ("paper", "release"): lambda: store.release_paper_kill_switch(
            "released from web dashboard"
        ),
    }
    if (which, action) not in actions:
        return _error("No such switch.", 404)
    if action == "engage" and body.get("confirm") is not True:
        raise _BadRequest("Tick the box to confirm first.")
    await _run(actions[(which, action)])
    return JSONResponse(await _run(_safety, store))


# --------------------------------------------------------------------- pages


async def spa(request: Request) -> Response:
    """Any path that is not the API or a built file gets the app shell."""
    index = WEB_DIST / "index.html"
    if not index.exists():
        return PlainTextResponse(
            "The web front end has not been built yet. Run: make web", status_code=503
        )
    return FileResponse(index, headers={"Cache-Control": "no-store"})


async def api_not_found(request: Request) -> Response:
    return _error("No such page.", 404)


class _SecurityHeaders:
    """Headers that stop this page being framed by, or leaking to, another site."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers += [
                    (b"x-frame-options", b"DENY"),
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (
                        b"content-security-policy",
                        b"default-src 'self'; img-src 'self' data:; style-src 'self' "
                        b"'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'",
                    ),
                ]
                if scope["path"].startswith("/api/"):
                    headers.append((b"cache-control", b"no-store"))
                message["headers"] = headers
            await send(message)

        return await self.app(scope, receive, send_with_headers)


def create_app(store: Store | None = None, secret: str | None = None) -> Starlette:
    def r(path, fn, methods=("GET",)):
        return Route(path, _authed(fn), methods=list(methods))

    routes = [
        Route("/api/login", login, methods=["POST"]),
        Route("/api/logout", logout, methods=["POST"]),
        Route("/api/me", me),
        r("/api/options", options),
        r("/api/translate", translate, ["POST"]),
        r("/api/backtest", backtest, ["POST"]),
        r("/api/adjust", adjust, ["POST"]),
        r("/api/results/{rid}/trades.csv", trades_csv),
        r("/api/library", library),
        r("/api/strategies", strategies),
        r("/api/strategies", save, ["POST"]),
        r("/api/strategies/{sid:int}/spec", strategy_spec),
        r("/api/paper", paper),
        r("/api/paper/start", paper_start, ["POST"]),
        r("/api/paper/{rid:int}/check", paper_check, ["POST"]),
        r("/api/paper/{rid:int}/stop", paper_stop, ["POST"]),
        r("/api/paper/{rid:int}/pause", paper_pause, ["POST"]),
        r("/api/paper/{rid:int}/resume", paper_resume, ["POST"]),
        r("/api/paper/{rid:int}/trade/{key}", paper_trade),
        r("/api/running", running),
        r("/api/activity", activity),
        r("/api/orders", orders),
        r("/api/orders/read", orders_read, ["POST"]),
        r("/api/orders/create", orders_create, ["POST"]),
        r("/api/orders/check", orders_check, ["POST"]),
        r("/api/orders/{oid:int}/cancel", orders_cancel, ["POST"]),
        r("/api/orders/{oid:int}/timeline", orders_timeline),
        r("/api/orders/positions/{pid:int}/close", orders_close, ["POST"]),
        r("/api/kite/login", kite_login_save, ["POST"]),
        r("/api/kite/forget", kite_login_forget, ["POST"]),
        r("/api/kite/token", kite_token_save, ["POST"]),
        r("/api/safety", safety),
        r("/api/safety/{which}/{action}", kill_switch, ["POST"]),
        r("/api/notifications", notifications),
        r("/api/notifications/{nid:int}/read", notifications_read, ["POST"]),
        r("/api/notifications/read-all", notifications_read_all, ["POST"]),
        r("/api/{rest:path}", api_not_found, ["GET", "POST"]),
    ]
    if (WEB_DIST / "assets").is_dir():
        routes.append(Mount("/assets", StaticFiles(directory=WEB_DIST / "assets")))
    routes.append(Route("/{rest:path}", spa))

    app = Starlette(
        routes=routes,
        middleware=[
            Middleware(_SecurityHeaders),
            Middleware(
                SessionMiddleware,
                secret_key=secret or _session_secret(),
                session_cookie="nlt_session",
                max_age=SESSION_HOURS * 3600,
                same_site="strict",
                https_only=True,
            ),
        ],
    )
    app.state.store = store or Store()
    app.state.lockout = _Lockout()
    app.state.results = _ResultCache()
    # Kept beside the price cache, outside the repo (data/ is not committed).
    from nlt.data.source import CACHE_DIR

    app.state.library_path = CACHE_DIR.parent / "library.json"
    app.state.library_lock = threading.Lock()
    return app


def main() -> None:  # pragma: no cover - a launcher
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("NLT_WEB_PORT", "8502"))
    # 127.0.0.1 only: the outside world reaches this through a tunnel or not at all.
    uvicorn.run(create_app(), host="127.0.0.1", port=port, proxy_headers=False)


if __name__ == "__main__":  # pragma: no cover
    main()
