"""Everything the dashboard does, minus the widgets.

Streamlit widgets are awkward to unit test -- they need a running script
context -- so every decision the app makes (parse text, run a backtest, judge
the result, format a number for a screen) lives here as a plain function.
`app/main.py` (and the page modules it drives) should do nothing but call
these functions and hand the results to `st.*` calls; if a test wants to
check *behaviour* rather than *layout*, it imports from here.

The other governing rule, inherited from `nlt.translate.rules`: nothing in
this module may turn an ambiguous or broken result into a silent best guess.
A backtest that cannot run becomes a `PipelineResult.error` string a
non-technical user can read, never a raised traceback on their screen.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from nlt.costs.charges import (
    ChargeModel,
    NseEquityDeliveryCharges,
    NseFuturesCharges,
    NseOptionsCharges,
    ZeroCharges,
)
from nlt.costs.charges import charge_fn as _adapt_charge_fn
from nlt.data.yahoo import YahooSource
from nlt.engine.backtest import BacktestResult, run_backtest
from nlt.engine.basket import BasketResult, run_basket_backtest
from nlt.engine.loader import load_for_spec
from nlt.spec.models import StrategySpec
from nlt.store.db import Store
from nlt.translate.llm import translate
from nlt.translate.readback import describe, format_inr
from nlt.translate.rules import Question, TranslationResult

try:  # pragma: no cover -- exercised once nlt.report exists
    from nlt.report.benchmark import Benchmark, Comparison, buy_and_hold, compare
    from nlt.report.verdict import Flag, Verdict, assess

    REPORT_SOURCE = "nlt.report"
except ImportError:  # nlt.report is being built by another agent concurrently
    from app.report_fallback import (
        Benchmark,
        Comparison,
        Flag,
        Verdict,
        assess,
        buy_and_hold,
        compare,
    )

    REPORT_SOURCE = "app.report_fallback"

logger = logging.getLogger(__name__)

__all__ = [
    "COST_MODEL_LABELS",
    "DEFAULT_CAPITAL",
    "EXAMPLE_STRATEGIES",
    "PipelineResult",
    "Benchmark",
    "Comparison",
    "Flag",
    "Verdict",
    "answers_from_questions",
    "charge_model_for_label",
    "charge_model_for_spec",
    "color_for_flag",
    "color_for_pnl",
    "describe_spec",
    "flags_by_severity",
    "format_inr",
    "format_pct",
    "format_signed_pct",
    "list_runs_view",
    "list_strategies_view",
    "load_bars",
    "run_pipeline",
    "safe_call",
    "save_strategy",
    "translate_text",
]

# --------------------------------------------------------------------- misc

DEFAULT_CAPITAL = 100_000.0

# Every example is verified (in tests/test_app.py) to parse into a spec with
# no follow-up questions -- a "try this" button that itself needs clarifying
# would be a bad first impression.
EXAMPLE_STRATEGIES = [
    "buy nifty when rsi cracks 30, target 2%, stop loss 1%",
    "buy nifty when 20 sma crosses above 50 sma, target 3%, stop loss 1.5%",
    "short banknifty on death cross, target 2%, stop 1%",
    "buy nifty when close is above 200 ema, target 5%, trailing stop 2%",
]

COST_MODEL_LABELS: dict[str, ChargeModel | None] = {
    "Index options (NIFTY/BANKNIFTY weekly)": NseOptionsCharges(),
    "Index futures": NseFuturesCharges(),
    "Equity delivery": NseEquityDeliveryCharges(),
    "No costs (for comparison only)": ZeroCharges(),
}


def charge_model_for_label(label: str) -> ChargeModel:
    """Looks up a cost model by its dropdown label, defaulting to options costs."""
    return COST_MODEL_LABELS.get(label, NseOptionsCharges())


def charge_model_for_spec(spec: StrategySpec, label: str) -> tuple[ChargeModel, str | None]:
    """The cost model to actually use, and a note if it overrode the dropdown.

    The dropdown is a real choice for an index strategy -- "buy NIFTY" is not
    directly tradable, so whether you would express it as options or futures
    changes the costs and only the user knows which they meant.

    It is not a real choice for shares. A strategy that buys RELIANCE pays
    equity delivery charges, and the dropdown defaults to *options* costs, so
    the untouched default silently applied option STT to a share trade. The two
    differ enough to move a marginal strategy from profitable to not, which is
    exactly the kind of wrong answer this project treats as serious.

    So: shares are reconciled to equity charges and the override is said out
    loud rather than done quietly. "No costs" is never overridden -- it is a
    deliberate comparison baseline, and silently adding charges to it would
    destroy the one thing it is for.
    """
    chosen = charge_model_for_label(label)

    if isinstance(chosen, ZeroCharges):
        return chosen, None

    if spec.instrument.trade_as == "stock" and not isinstance(chosen, NseEquityDeliveryCharges):
        return NseEquityDeliveryCharges(), (
            f"Costs: this strategy buys and sells shares, so equity delivery charges "
            f"were applied instead of '{label}'. Share trades are not taxed the way "
            f"index options are, and using the option figures would have overstated "
            f"what this strategy costs to run."
        )

    return chosen, None


# ---------------------------------------------------------------- formatting


def format_pct(value: float | None, *, decimals: int = 1) -> str:
    """`12.345` -> `'12.3%'`. `None` reads as 'n/a', never as a crash or 'None%'."""
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}%"


def format_signed_pct(value: float | None, *, decimals: int = 1) -> str:
    """Like `format_pct`, but always shows the sign: `+2.0%` / `-1.0%`."""
    if value is None:
        return "n/a"
    return f"{value:+.{decimals}f}%"


def color_for_pnl(value: float) -> str:
    """The delta-good / delta-bad ink from the palette -- never a raw guess.

    Positive is the success-text green, negative the status-critical red,
    zero the muted/secondary ink. These are *text* colors (a P&L number in a
    table), which is the one place the palette allows a status hue on text.
    """
    if value > 0:
        return "#006300"
    if value < 0:
        return "#d03b3b"
    return "#52514e"


_SEVERITY_COLORS = {
    "critical": "#d03b3b",
    "warning": "#fab219",
    "info": "#52514e",
}
_SEVERITY_ICONS = {"critical": "🔴", "warning": "⚠️", "info": "ℹ️"}


def color_for_flag(severity: str) -> str:
    """The fixed status hex for a verdict flag's severity, never eyeballed."""
    return _SEVERITY_COLORS.get(severity, _SEVERITY_COLORS["info"])


def icon_for_flag(severity: str) -> str:
    return _SEVERITY_ICONS.get(severity, _SEVERITY_ICONS["info"])


def flags_by_severity(flags: list[Flag]) -> list[Flag]:
    """Most severe first -- critical, then warning, then info."""
    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(flags, key=lambda f: order.get(f.severity, 3))


def describe_spec(spec: StrategySpec) -> str:
    return describe(spec)


# ---------------------------------------------------------------- translate


def translate_text(text: str, answers: dict[str, str] | None = None) -> TranslationResult:
    """Wraps `nlt.translate.llm.translate` so a bug in a reader cannot crash the page.

    The rules read the sentence first; the AI reader is only asked when they
    could not understand it. Both are written to never raise on bad input -- unparseable text comes
    back as `unparsed` fragments and `Question`s -- but this dashboard is the
    only thing standing between that guarantee and a stack trace on a
    non-technical user's screen, so it is wrapped anyway.
    """
    try:
        return translate(text, answers=answers or {})
    except Exception:  # pragma: no cover -- defensive; translate() is designed not to raise
        logger.exception("translate_text: translate() raised on %r", text)
        return TranslationResult(
            spec=None,
            questions=[
                Question(
                    text="I couldn't read that description. Try rephrasing it more simply.",
                    why="Something in the parser broke; the detail is in the app log.",
                    suggestion=None,
                    field="description",
                )
            ],
            source="rules",
        )


def answers_from_questions(questions: list[Question], values: dict[str, str]) -> dict[str, str]:
    """Keeps only non-empty answers, keyed by `Question.field`, for re-parsing."""
    return {q.field: values[q.field] for q in questions if values.get(q.field)}


# --------------------------------------------------------------------- data


def load_bars(
    symbol: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Daily OHLCV bars for `symbol`, from the local Yahoo cache (or a fresh download).

    Plain function, no `st.cache_data` here -- `app/main.py` wraps this with
    the Streamlit cache decorator so this module stays importable and testable
    without a Streamlit runtime.
    """
    return YahooSource().bars(symbol, start=start, end=end)


def staleness_days(symbol: str) -> int | None:
    return YahooSource().staleness_days(symbol)


# ------------------------------------------------------------------ pipeline


@dataclass
class PipelineResult:
    """The full text-to-verdict path, or as much of it as completed.

    `error` is set (and everything after `translation` is `None`) whenever a
    step downstream of a valid spec failed -- a data problem, an engine bug,
    anything unexpected. `translation.spec is None` (no `error`) means the
    text itself was not yet understood, which is a normal, expected outcome,
    not a failure.
    """

    translation: TranslationResult
    spec: StrategySpec | None = None
    bars: pd.DataFrame | None = None
    backtest: BacktestResult | None = None
    benchmark: Benchmark | None = None
    comparison: Comparison | None = None
    verdict: Verdict | None = None
    # Populated only for a basket run. `backtest` still carries the metrics and
    # trades so every downstream consumer (charts, verdict, tables) works on one
    # shape regardless of whether one symbol or a hundred were traded.
    basket: BasketResult | None = None
    data_notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.backtest is not None


def _as_backtest_result(basket: BasketResult, spec: StrategySpec) -> BacktestResult:
    """Present a basket run in the shape the rest of the UI already understands.

    Charts, the metrics table, the trade list and the verdict were all written
    against `BacktestResult`. A basket carries the same information -- one
    portfolio equity curve, one set of metrics, one list of trades -- so it is
    adapted rather than special-cased everywhere downstream. The alternative is
    a second rendering path that drifts from the first.
    """
    return BacktestResult(
        spec=spec,
        trades=basket.trades,
        equity=basket.equity,
        metrics=basket.metrics,
        features=pd.DataFrame(index=basket.equity.index),
        warnings=list(basket.warnings),
    )


def _equal_weight_index(bars_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """An equal-weight basket of the same names, to benchmark a basket against.

    Comparing a hundred-stock strategy to NIFTY would conflate two questions:
    whether the timing rules add anything, and whether those hundred stocks beat
    the index. This isolates the first -- "would picking your moments have beaten
    simply owning all of them?" -- which is what the user is actually asking.

    Each symbol is normalised to its own first close so no single high-priced
    stock dominates, then averaged across whichever symbols have a bar that day.
    """
    normalised = {}
    for symbol, df in bars_by_symbol.items():
        if df.empty:
            continue
        base = df["close"].iloc[0]
        if base > 0:
            normalised[symbol] = df["close"] / base

    if not normalised:
        raise ValueError("no usable price history in the basket")

    combined = pd.DataFrame(normalised).sort_index()
    level = 100.0 * combined.mean(axis=1, skipna=True)

    # An OHLCV frame is what the benchmark and charts expect. The synthetic
    # index has no meaningful intraday range, so open/high/low all equal close;
    # `buy_and_hold` only reads the first open and the last close.
    return pd.DataFrame(
        {"open": level, "high": level, "low": level, "close": level, "volume": 0.0},
        index=level.index,
    ).dropna()


def run_pipeline(
    text: str,
    *,
    capital: float = DEFAULT_CAPITAL,
    cost_model_label: str = next(iter(COST_MODEL_LABELS)),
    symbol: str | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    answers: dict[str, str] | None = None,
) -> PipelineResult:
    """Plain English in, verdict out -- the function `run_backtest_flow` in the brief.

    Every downstream step (data load, engine, benchmark, verdict) is wrapped
    so a real bug never reaches the UI as a traceback; it comes back as
    `PipelineResult.error` instead.
    """
    translation = translate_text(text, answers)
    if translation.spec is None:
        return PipelineResult(translation=translation)

    spec = translation.spec
    try:
        model, cost_note = charge_model_for_spec(spec, cost_model_label)
        cf = _adapt_charge_fn(model)

        # `load_for_spec` knows how to turn a symbol into bars whether it names
        # an index, a single stock or a whole universe, and hands back every
        # data-quality note it collected on the way -- artefact cleaning, load
        # failures, survivorship. Those notes reach the user; they are the
        # reason a basket that quietly lost fifteen constituents is visible.
        # An explicit `symbol` override replaces the one the spec carries. It has
        # to be pushed into the spec rather than passed alongside it, because
        # `load_for_spec` decides index-vs-stock-vs-universe from the spec --
        # honouring the override anywhere else would load one symbol and then
        # backtest against another's rules.
        load_spec = spec
        if symbol and symbol != spec.instrument.symbol:
            load_spec = spec.model_copy(
                update={"instrument": spec.instrument.model_copy(update={"symbol": symbol})}
            )

        bars_by_symbol, data_notes = load_for_spec(load_spec, start=start, end=end)
        if cost_note:
            data_notes = [cost_note, *data_notes]
        if not bars_by_symbol or all(df.empty for df in bars_by_symbol.values()):
            return PipelineResult(
                translation=translation,
                spec=spec,
                data_notes=data_notes,
                error="There is no price history for that symbol and date range to test against.",
            )

        if len(bars_by_symbol) > 1:
            basket = run_basket_backtest(load_spec, bars_by_symbol, capital=capital, charge_fn=cf)
            backtest = _as_backtest_result(basket, load_spec)
            # The benchmark for a basket is an equal-weight buy-and-hold of the
            # same names, which is the honest comparison: "would picking your
            # moments have beaten simply owning all of them?"
            bars = _equal_weight_index(bars_by_symbol)
            benchmark_name = f"holding all {len(bars_by_symbol)} of these stocks equally"
            data_notes = data_notes + basket.warnings
        else:
            bars = next(iter(bars_by_symbol.values()))
            benchmark_name = f"holding {load_spec.instrument.symbol}"
            basket = None
            backtest = run_backtest(load_spec, bars, capital=capital, charge_fn=cf)
            data_notes = data_notes + list(backtest.warnings)

        benchmark = buy_and_hold(bars, capital, charge_fn=cf, name=benchmark_name)
        comparison = compare(backtest, benchmark, bars)
        verdict = assess(backtest, comparison)

        return PipelineResult(
            translation=translation,
            spec=spec,
            bars=bars,
            backtest=backtest,
            benchmark=benchmark,
            comparison=comparison,
            verdict=verdict,
            basket=basket,
            data_notes=data_notes,
        )
    except Exception as exc:  # noqa: BLE001 - deliberately broad; this is the UI's last line of defence
        logger.exception("run_pipeline failed for spec %r", getattr(spec, "name", None))
        return PipelineResult(
            translation=translation,
            spec=spec,
            error=f"The backtest could not be run: {exc}",
        )


def safe_call(fn: Callable[[], object], *, on_error: str) -> tuple[object | None, str | None]:
    """Runs `fn`, turning any exception into a plain-English message instead of a crash.

    Returns `(result, None)` on success or `(None, message)` on failure. Every
    place the dashboard touches the store, the data source, or the engine
    outside `run_pipeline` (re-runs, kill switch, etc.) should go through this
    so a raw traceback never reaches the browser.
    """
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001
        logger.error("safe_call failed: %s\n%s", exc, traceback.format_exc())
        return None, f"{on_error}: {exc}"


# ----------------------------------------------------------------- storage


def save_strategy(
    store: Store,
    spec: StrategySpec,
    result: PipelineResult | None = None,
    *,
    capital: float | None = None,
    cost_model_label: str | None = None,
) -> int:
    """Saves the strategy and, when given, the backtest that was just shown.

    Recording the backtest as a completed run is what lets the strategy earn
    "Proven" later: the proving gate wants both a backtest and a paper run.
    Before this, nothing ever recorded a backtest, so no strategy could pass.
    """
    readback = describe_spec(spec)
    strategy_id = store.save_strategy(spec, readback)
    if result is not None and result.ok:
        run_id = store.start_run(
            strategy_id,
            "backtest",
            {"capital": capital, "cost_model_label": cost_model_label},
        )
        store.save_trades(run_id, result.backtest.trades, spec.instrument.symbol)
        store.finish_run(run_id, result.backtest.metrics)
        if store.get_strategy(strategy_id)["state"] == "draft":
            store.set_state(strategy_id, "backtested")
    return strategy_id


def list_strategies_view(store: Store) -> list[dict]:
    """Empty database -> empty list, never an exception."""
    return store.list_strategies()


def list_runs_view(store: Store, strategy_id: int | None = None) -> list[dict]:
    return store.list_runs(strategy_id=strategy_id)


# ------------------------------------------------------------ paper trading


def kite_status() -> tuple[str, str]:
    """("connected" | "expired" | "not_connected" | "unreachable", plain-English line)."""
    from nlt.data import kite

    if not kite.is_connected():
        return "not_connected", "Kite is not connected. Paste today's token below."
    try:
        user_id = kite.check_connection()
    except kite.KiteTokenExpired:
        return "expired", "The saved Kite token has expired. Paste today's token below."
    except kite.KiteError as exc:
        return "unreachable", f"Could not reach Kite just now: {exc}"
    return "connected", f"Connected to Kite as {user_id}."


def save_kite_token(token: str) -> None:
    from nlt.data import kite

    kite.save_token(token)


def paper_job_installed() -> bool:
    """Is the every-minute background job in this machine's crontab?"""
    import subprocess

    try:
        out = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    return "# nlt-paper" in out.stdout


def paper_problem(spec: StrategySpec) -> str | None:
    """Why this strategy cannot be paper traded, in plain English, or None."""
    if spec.instrument.trade_as == "option":
        return "Option strategies can't be paper traded yet -- there is no option price feed."
    if spec.instrument.is_universe and spec.instrument.timeframe != "1d":
        return "Stock baskets can only run on daily candles for now."
    return None


def start_paper(store: Store, strategy_id: int, *, capital: float, cost_model_label: str) -> int:
    spec = store.load_spec(strategy_id)
    problem = paper_problem(spec)
    if problem:
        raise ValueError(problem)
    if any(r["strategy_id"] == strategy_id for r in store.active_paper_runs()):
        raise ValueError("This strategy is already being paper traded.")
    started_at = pd.Timestamp.now(tz="Asia/Kolkata")
    run_id = store.start_run(
        strategy_id,
        "paper",
        {
            "capital": capital,
            "cost_model_label": cost_model_label,
            "started_at": started_at.isoformat(),
        },
    )
    store.set_state(strategy_id, "paper")
    store.set_paper_status(run_id, "waiting", "Started. Waiting for the next finished candle.")
    return run_id


def check_paper_now(store: Store, run_id: int) -> str:
    """One pass for one run, right now, regardless of the schedule."""
    from nlt.data.kite import KiteSource
    from nlt.paper import runner

    run = store.get_run(run_id)
    spec = store.load_spec(run["strategy_id"])
    params = json.loads(run["params_json"] or "{}")
    model, _ = charge_model_for_spec(spec, params.get("cost_model_label", ""))
    source = KiteSource()
    result = runner.step(
        store,
        run,
        fetch=lambda sym, tf, a, b: source.bars(sym, tf, a, b),
        charge_fn=_adapt_charge_fn(model),
    )
    return result.message


def stop_paper(store: Store, run_id: int) -> None:
    """Ends a paper run. A completed paper run is what the proving gate counts."""
    status = store.paper_status(run_id) or {}
    events = store.paper_events(run_id)
    metrics = json.loads(status.get("snapshot_json") or "{}")
    metrics["drift_events"] = sum(e["kind"] == "drift" for e in events)
    metrics["stopped_at"] = pd.Timestamp.now(tz="Asia/Kolkata").isoformat()
    store.finish_run(run_id, metrics, status="complete")


_HEALTH_LABELS = {
    "ok": ("success", "Running"),
    "waiting": ("info", "Waiting for the next candle"),
    "paused": ("warning", "Paused"),
    "token_expired": ("error", "Kite token expired"),
    "not_connected": ("error", "Kite not connected"),
    "feed_down": ("error", "Price feed problem"),
    "error": ("error", "Error"),
}


def paper_health_label(health: str) -> tuple[str, str]:
    """(streamlit message kind, short label) for a run's health."""
    return _HEALTH_LABELS.get(health, ("info", health))


def when(iso: str | None) -> str:
    if not iso:
        return ""
    return pd.Timestamp(iso).tz_convert("Asia/Kolkata").strftime("%d %b %H:%M")


def describe_paper_event(event: dict) -> str:
    """One line of the activity log, written from the recorded event alone."""
    detail = json.loads(event.get("detail_json") or "{}")
    kind = event["kind"]
    qty = event.get("quantity")
    qty_text = f"{qty:g} " if qty else ""
    if kind == "signal":
        return (
            f"Signal: the entry rule was met on the {when(event['bar_time'])} candle. "
            "It fills at the next candle's open."
        )
    if kind == "entry":
        verb = "Bought" if event["direction"] == "long" else "Sold short"
        return (
            f"{verb} {qty_text}{event['symbol']} at {format_inr(event['price'])} "
            f"(the open of the {when(event['bar_time'])} candle)."
        )
    if kind == "exit":
        reason = _EXIT_WORDS.get(detail.get("reason"), detail.get("reason", "exit"))
        pnl = detail.get("net_pnl")
        pnl_text = (
            f" Result: {'+' if pnl >= 0 else '-'}{format_inr(abs(pnl))}." if pnl is not None else ""
        )
        verb = "Sold" if event["direction"] == "long" else "Bought back"
        return (
            f"{verb} {qty_text}{event['symbol']} at {format_inr(event['price'])} "
            f"on the {when(event['bar_time'])} candle -- {reason}.{pnl_text}"
        )
    if kind == "drift":
        return (
            f"Kite's prices changed after this was recorded ({detail.get('event')} "
            f"{detail.get('key')}): recorded {detail.get('recorded')}, now {detail.get('now')}. "
            "The original record stands."
        )
    return detail.get("note", kind)


_EXIT_WORDS = {
    "stop": "stop loss hit",
    "trailing_stop": "trailing stop hit",
    "target": "target reached",
    "condition": "exit rule met",
    "max_bars": "time limit reached",
    "square_off": "end-of-day square-off",
}
