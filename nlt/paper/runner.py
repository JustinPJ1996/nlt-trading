"""Paper trading: run a saved strategy forward against the market as it happens.

The design rests on one idea: **the paper trader is the backtest engine, re-run
on every pass, plus a record that cannot be rewritten.**

On each pass the runner fetches candles up to now, drops any candle that has
not finished, and runs the very same engine a backtest uses -- with
`trade_from` set so nothing before the moment the user pressed Start can open a
position. Every entry and exit the engine produces is then written to the
append-only `paper_event` table the first time it is seen, stamped with the
wall-clock time it was seen.

Why this shape rather than a separate live state machine:

  * Paper and backtest cannot disagree about the rules -- closed candles only,
    fills at the next candle's open, stop assumed first -- because they are
    the same code. A second implementation would be a second place for those
    rules to drift apart, which is exactly what paper trading exists to catch.
  * The record is honest. Re-deriving from scratch every pass means a revised
    candle would change the engine's answer; the original row stands and the
    disagreement is written as a separate 'drift' event, never a correction.
  * A crash, a restart or a missed minute costs nothing: the next pass
    re-derives everything, and only events not yet recorded are written.

The runner never places an order. There is no code path from here to a broker.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from nlt.data import kite
from nlt.data.session import NSE_EQUITY, Session, session_date
from nlt.engine.backtest import Trade, run_backtest
from nlt.engine.basket import run_basket_backtest
from nlt.engine.conditions import evaluate
from nlt.spec.models import StrategySpec
from nlt.store.db import Store

IST = "Asia/Kolkata"

# When a candle is finished, measured from its own timestamp. A daily candle is
# stamped at midnight and finishes when its market closes -- 15:30 for NSE,
# 23:55 for MCX (see `close_offset`).
_CLOSE_OFFSET = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(minutes=60),
}

# History fetched before the start, so indicators are warmed up on the first
# live candle. Two years covers a 200-day average with room to spare; a month
# of intraday candles covers a 200-candle average on anything but 1-minute.
_WARMUP = {"1d": dt.timedelta(days=730)}
_INTRADAY_WARMUP = dt.timedelta(days=30)

Fetch = Callable[[str, str, dt.date, dt.date], pd.DataFrame]
"""(symbol, timeframe, start date, end date) -> candles. `KiteSource().bars` in real use.

A futures strategy is fetched under its data key ("CRUDEOIL FUT"), never its
bare symbol, which for NIFTY would be the index."""

ListedExpiries = Callable[[str], list[dt.date]]
"""symbol -> expiries the exchange has published. `kite.listed_expiries` in real use."""


def _session_of(spec: StrategySpec) -> Session:
    if spec.instrument.is_future:
        from nlt.data.futures import contract

        return contract(spec.instrument.symbol).session
    return NSE_EQUITY


def _close_time(session: Session) -> pd.Timedelta:
    return pd.Timedelta(hours=session.close_time.hour, minutes=session.close_time.minute)


def close_offset(timeframe: str, session: Session = NSE_EQUITY) -> pd.Timedelta:
    if timeframe == "1d":
        return _close_time(session)
    try:
        return _CLOSE_OFFSET[timeframe]
    except KeyError:
        raise ValueError(f"paper trading has no candle length for {timeframe!r}") from None


def closed_bars(
    bars: pd.DataFrame, timeframe: str, now: pd.Timestamp, session: Session = NSE_EQUITY
) -> pd.DataFrame:
    """Only candles that have finished by `now`. A forming candle is never traded on."""
    if bars.empty:
        return bars
    return bars[bars.index + close_offset(timeframe, session) <= now]


def trade_from_for(
    started_at: pd.Timestamp, timeframe: str, session: Session = NSE_EQUITY
) -> pd.Timestamp:
    """The first candle allowed to signal: the first one that finishes after Start."""
    return started_at - close_offset(timeframe, session) + pd.Timedelta(microseconds=1)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST)


def _iso(ts) -> str:
    return pd.Timestamp(ts).isoformat()


@dataclass
class StepResult:
    health: str
    message: str
    new_events: list[str] = field(default_factory=list)


def _symbols(spec: StrategySpec) -> list[str]:
    if spec.instrument.is_universe:
        from nlt.data.universe import resolve_symbols

        return list(resolve_symbols(spec.instrument.symbol))
    return [spec.instrument.symbol]


def _session_in_progress(
    bars: pd.DataFrame, now: pd.Timestamp, session: Session = NSE_EQUITY
) -> bool:
    """Is the newest candle's trading day still going on at `now`?"""
    last_day = bars.index[-1].tz_convert(IST).normalize()
    return now.tz_convert(IST) < last_day + _close_time(session)


def _current_contract_from(
    spec: StrategySpec, bars: pd.DataFrame, listed: list[dt.date]
) -> pd.Timestamp | None:
    """For intraday futures: the first candle of the contract now being traded.

    Kite has intraday candles only for live contracts, so after each expiry the
    series fetched is the NEW contract's -- its whole history, including the
    weeks it was the second month, at prices that were never the ones traded.
    Re-running the engine on that history could find trades in the past that
    nobody was ever in. So nothing before the current contract became the
    nearest month may open a position, and nothing recorded before then is
    re-judged against prices from a different contract.
    """
    if not spec.instrument.is_future or spec.instrument.timeframe == "1d" or bars.empty:
        return None
    from nlt.data.futures import contract, expiry_sessions

    session = contract(spec.instrument.symbol).session
    today = session_date(bars.index[-1], session)
    past = [d for d in expiry_sessions(spec.instrument.symbol, bars.index, listed) if d < today]
    if not past:
        return None
    after = [ts for ts in bars.index if session_date(ts, session) > max(past)]
    return after[0] if after else None


def _run_engine(spec, bars_by_symbol, *, capital, charge_fn, trade_from, now, listed=()):
    """The same engine a backtest uses. Returns (trades, single-symbol result or None)."""
    if spec.instrument.is_universe:
        result = run_basket_backtest(
            spec, bars_by_symbol, capital=capital, charge_fn=charge_fn, trade_from=trade_from
        )
        return result.trades, None
    (bars,) = bars_by_symbol.values()
    # The forming candle was already removed against the runner's own clock;
    # the engine's wall-clock check is turned off so a replayed past day is
    # judged by the time being replayed, not by today's.
    result = run_backtest(
        spec,
        bars,
        capital=capital,
        charge_fn=charge_fn,
        drop_partial_last_bar=False,
        trade_from=trade_from,
        final_session_in_progress=spec.instrument.timeframe != "1d"
        and _session_in_progress(bars, now, _session_of(spec)),
        listed_expiries=listed,
    )
    return result.trades, result


def _key(t: Trade, fallback_symbol: str) -> str:
    return f"{t.symbol or fallback_symbol}|{_iso(t.entry_time)}"


def step(
    store: Store,
    run: dict,
    *,
    fetch: Fetch,
    charge_fn,
    now: pd.Timestamp | None = None,
    listed_expiries: ListedExpiries | None = None,
) -> StepResult:
    """One pass of one paper run. Safe to repeat: only unseen events are recorded."""
    now = now or _now()
    run_id = run["id"]
    params = json.loads(run["params_json"] or "{}")

    if store.paper_kill_switch_engaged():
        store.set_paper_status(run_id, "paused", "Paused: the paper kill switch is on.")
        return StepResult("paused", "paper kill switch engaged")

    spec = store.load_spec(run["strategy_id"])
    timeframe = spec.instrument.timeframe
    session = _session_of(spec)
    started_at = pd.Timestamp(params["started_at"]).tz_convert(IST)
    trade_from = trade_from_for(started_at, timeframe, session)
    listed: list[dt.date] = []
    if spec.instrument.is_future and listed_expiries is not None:
        try:
            listed = list(listed_expiries(spec.instrument.symbol))
        except kite.KiteError:
            listed = []
    warmup = _WARMUP.get(timeframe, _INTRADAY_WARMUP)
    fetch_start = (started_at - warmup).date()

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    for symbol in _symbols(spec):
        if spec.instrument.is_future:
            from nlt.data.futures import data_key

            source_key = data_key(symbol)
        else:
            source_key = symbol
        try:
            raw = fetch(source_key, timeframe, fetch_start, now.date())
        except kite.KiteTokenExpired:
            msg = "Kite token has expired. Paste today's token to resume."
            store.set_paper_status(run_id, "token_expired", msg)
            return StepResult("token_expired", msg)
        except kite.KiteNotConnected:
            msg = "Kite is not connected. Save your Kite login to start."
            store.set_paper_status(run_id, "not_connected", msg)
            return StepResult("not_connected", msg)
        except kite.KiteLoginFailed as exc:
            msg = f"Could not log in to Kite. {exc}"
            store.set_paper_status(run_id, "login_failed", msg)
            return StepResult("login_failed", msg)
        except kite.KiteError as exc:
            if len(_symbols(spec)) == 1:
                msg = f"Could not get prices from Kite: {exc}"
                store.set_paper_status(run_id, "feed_down", msg)
                return StepResult("feed_down", msg)
            failures.append(symbol)
            continue
        bars = closed_bars(raw, timeframe, now, session)
        if not bars.empty:
            bars_by_symbol[symbol] = bars

    if not bars_by_symbol:
        msg = "No finished candles yet."
        store.set_paper_status(run_id, "waiting", msg)
        return StepResult("waiting", msg)

    contract_from = None
    if len(bars_by_symbol) == 1:
        contract_from = _current_contract_from(spec, next(iter(bars_by_symbol.values())), listed)
    if contract_from is not None:
        trade_from = max(trade_from, contract_from)

    trades, single = _run_engine(
        spec,
        bars_by_symbol,
        capital=float(params.get("capital", 100_000.0)),
        charge_fn=charge_fn,
        trade_from=trade_from,
        now=now,
        listed=listed,
    )
    fallback_symbol = spec.instrument.symbol
    observed = _iso(now)
    new: list[str] = []

    recorded = {(e["kind"], e["event_key"]): e for e in store.paper_events(run_id)}
    if contract_from is not None:
        # Recorded on an earlier contract's prices: final, and not comparable
        # with the current contract's candles. See `_current_contract_from`.
        recorded = {
            k: e
            for k, e in recorded.items()
            if e["kind"] not in ("entry", "exit")
            or pd.Timestamp(e["event_key"].split("|", 1)[1]) >= contract_from
        }
    derived_entries: set[str] = set()
    derived_exits: set[str] = set()

    for t in trades:
        key = _key(t, fallback_symbol)
        derived_entries.add(key)
        symbol = t.symbol or fallback_symbol
        entry_fields = {
            "price": round(t.entry_price, 4),
            "quantity": t.quantity,
            "direction": t.direction,
        }
        prior = recorded.get(("entry", key))
        if prior is None:
            if store.record_paper_event(
                run_id,
                "entry",
                key,
                observed_at=observed,
                symbol=symbol,
                bar_time=_iso(t.entry_time),
                direction=t.direction,
                quantity=t.quantity,
                price=t.entry_price,
                detail={"reason": t.entry_reason},
            ):
                new.append(f"entry {key}")
        else:
            was = {
                "price": round(prior["price"], 4),
                "quantity": prior["quantity"],
                "direction": prior["direction"],
            }
            if was != entry_fields:
                _drift(store, run_id, observed, symbol, "entry", key, was, entry_fields, new)

        if t.exit_reason == "end_of_data":
            if ("exit", key) in recorded:
                _drift(
                    store,
                    run_id,
                    observed,
                    symbol,
                    "exit",
                    key,
                    {"exit_time": recorded[("exit", key)]["bar_time"]},
                    {"exit_time": None},
                    new,
                )
            continue

        derived_exits.add(key)
        exit_fields = {
            "exit_time": _iso(t.exit_time),
            "price": round(t.exit_price, 4),
            "reason": t.exit_reason,
        }
        prior = recorded.get(("exit", key))
        if prior is None:
            if store.record_paper_event(
                run_id,
                "exit",
                key,
                observed_at=observed,
                symbol=symbol,
                bar_time=_iso(t.exit_time),
                direction=t.direction,
                quantity=t.quantity,
                price=t.exit_price,
                detail={
                    "reason": t.exit_reason,
                    "net_pnl": t.net_pnl,
                    "charges": t.charges,
                    "entry_time": _iso(t.entry_time),
                    "entry_price": t.entry_price,
                },
            ):
                store.save_trades(run_id, [t], symbol)
                new.append(f"exit {key}")
        else:
            was = {
                "exit_time": prior["bar_time"],
                "price": round(prior["price"], 4),
                "reason": json.loads(prior["detail_json"]).get("reason"),
            }
            if was != exit_fields:
                _drift(store, run_id, observed, symbol, "exit", key, was, exit_fields, new)

    # Something recorded earlier that today's data no longer produces.
    for (kind, key), prior in recorded.items():
        gone = (kind == "entry" and key not in derived_entries) or (
            kind == "exit" and key not in derived_exits and key not in derived_entries
        )
        if gone:
            _drift(
                store,
                run_id,
                observed,
                prior["symbol"],
                kind,
                key,
                {"recorded": True},
                {"recorded": False},
                new,
            )

    last_bar = max(df.index[-1] for df in bars_by_symbol.values())
    open_trades = [t for t in trades if t.exit_reason == "end_of_data"]

    if single is not None and len(open_trades) < spec.risk.max_concurrent_positions:
        entry_signal = evaluate(spec.entry, single.features)
        bar_ts = single.features.index[-1]
        if bar_ts >= trade_from and bool(entry_signal.iloc[-1]):
            if store.record_paper_event(
                run_id,
                "signal",
                f"{fallback_symbol}|{_iso(bar_ts)}",
                observed_at=observed,
                symbol=fallback_symbol,
                bar_time=_iso(bar_ts),
                price=float(single.features["close"].iloc[-1])
                if "close" in single.features
                else None,
                detail={"note": "entry condition met; fills at the next candle's open"},
            ):
                new.append(f"signal {_iso(bar_ts)}")

    closed = [t for t in trades if t.exit_reason != "end_of_data"]
    capital = float(params.get("capital", 100_000.0))
    realised = sum(t.net_pnl for t in closed)
    open_pnl = sum(t.net_pnl for t in open_trades)
    snapshot = {
        "capital": capital,
        "realised_pnl": realised,
        "open_pnl_if_closed_now": open_pnl,
        "equity": capital + realised + open_pnl,
        "closed_trades": len(closed),
        "open_positions": [
            {
                "symbol": t.symbol or fallback_symbol,
                "direction": t.direction,
                "quantity": t.quantity,
                "entry_time": _iso(t.entry_time),
                "entry_price": t.entry_price,
                "last_price": t.exit_price,
                "pnl_if_closed_now": t.net_pnl,
            }
            for t in open_trades
        ],
        "skipped_symbols": failures,
    }
    message = f"Up to date to the candle at {pd.Timestamp(last_bar):%d %b %H:%M}."
    if failures:
        message += f" Could not load {len(failures)} symbol(s)."
    store.set_paper_status(run_id, "ok", message, _iso(last_bar), snapshot)
    return StepResult("ok", message, new)


def _drift(store, run_id, observed, symbol, kind, key, was, now_fields, new) -> None:
    fingerprint = json.dumps([was, now_fields], sort_keys=True, default=str)
    if store.record_paper_event(
        run_id,
        "drift",
        f"{kind}|{key}|{fingerprint}",
        observed_at=observed,
        symbol=symbol,
        detail={"event": kind, "key": key, "recorded": was, "now": now_fields},
    ):
        new.append(f"drift {kind} {key}")


# ------------------------------------------------------------------ scheduling


def is_due(run: dict, status: dict | None, spec: StrategySpec, now: pd.Timestamp) -> bool:
    """Whether a pass could find anything new, judged from the clock alone.

    Saves fetching a whole basket every minute when nothing can have changed:
    outside market hours, or for a daily strategy whose latest candle is
    already recorded. Holidays are not known in advance -- on one, a due pass
    simply finds no new candle.

    A daily candle is looked for whenever the market is shut and the last
    finished day's candle is not yet recorded -- not only in the minutes after
    the close. MCX closes at 23:55, so "after the close, same day" would leave
    a five-minute window, and Friday's candle unseen until Monday night.
    """
    now = now.tz_convert(IST)
    session = _session_of(spec)
    today = now.normalize()
    open_ = today + pd.Timedelta(hours=session.open_time.hour, minutes=session.open_time.minute)
    close = today + _close_time(session)
    timeframe = spec.instrument.timeframe

    if timeframe == "1d":
        if now.weekday() < 5 and open_ <= now < close:
            return False  # today's candle is still forming
        finished = today if now.weekday() < 5 and now >= close else _previous_weekday(today)
        last = status.get("last_bar_time") if status else None
        if last and pd.Timestamp(last).tz_convert(IST).normalize() >= finished:
            return False
        # Re-check at most every 15 minutes until the candle appears.
        if status and status.get("updated_at"):
            updated = pd.Timestamp(status["updated_at"])
            if updated.tzinfo is None:
                updated = updated.tz_localize("UTC")
            if now - updated.tz_convert(IST) < pd.Timedelta(minutes=15):
                return False
        return True

    if now.weekday() >= 5:
        return False
    return open_ < now <= close + pd.Timedelta(minutes=5)


def _previous_weekday(day: pd.Timestamp) -> pd.Timestamp:
    day -= pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day
