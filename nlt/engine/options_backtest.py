"""The backtest loop for buying one call or one put (Phase 2, batch 1).

The signal is judged on the **index's** candles, exactly as `run_backtest`
judges any other strategy; the trade happens in an **option contract** chosen
at the moment of the signal. Everything `run_backtest` refuses to do, this
refuses too -- it is a separate loop rather than a branch inside that one only
because every price after the signal comes from a different series (the
chosen contract's), and threading that through the stock/futures loop would
put the rules every existing result depends on at risk.

The rules, each one Justin's or the engine's own (see `HANDOFF.md`):

  * **The contract is chosen at the signal, from the signal candle's close.**
    `OptionExecution.resolve` picks the listed strike nearest the index's close
    on that closed candle -- nothing later is visible to it.
  * **Fill at the next candle's open, on the option**, plus the run's
    slippage. If the chosen option did not trade in that candle, the signal is
    **skipped** (`option_not_traded`) -- never filled on another strike, never
    filled at a price that did not trade.
  * **Stop and target are percentages of the option's price at the signal**:
    the last traded price of that contract up to the end of the signal
    candle. The levels belong to the setup, the same rule as every other
    strategy, applied to the premium. If the option had not traded at all that
    day up to the signal, there is no price to measure from, and the signal
    is skipped (`no_option_price`).
  * **Stop first** when one candle could hit both; **a gap past the stop
    fills at the open** -- both unchanged.
  * **A candle in which the option did not trade cannot fill an exit.** A
    stop, target or condition exit waits for the next candle with a real
    trade, where the gap rule above then applies.
  * **Square-off fills at the 15:15 price itself** -- the opening price of the
    square-off candle (Justin, 2026-10-13), not its close. If the option did
    not trade in that candle, the **last traded price** is used and the trade
    is flagged `exit_untraded` (Justin's choice over the worse of last trade
    and bid).
  * **Never held to expiry.** A position still open on its contract's expiry
    day is closed at the square-off time that day, priced the same way. So an
    option is never exercised or settled, and the STT charged on an exercised
    in-the-money option can never arise (`settled_at_expiry` is always False).
  * **A listed expiry that fell on a market holiday really ended the session
    before** (NIFTY/BANKNIFTY 30 Mar 2023 and 29 Jun 2023 are listed at their
    nominal dates). Every expiry rule here -- the expiry-day close, no entry
    after the last close, never held past expiry -- uses that real last
    trading day (`last_day_fn`, the rule `nlt.options.iv` uses).
  * **No entry after the contract's last close time.** A signal that would
    fill in a contract at or after its expiry-day square-off is skipped.
  * Lot size is **today's** lot (`nlt.data.instruments`), not the data's.

Only long (bought) single options reach here: the spec model refuses a sold
option, and a multi-leg trade cannot be written as a spec yet.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nlt.costs.slippage import ZERO, Slippage
from nlt.data.session import (
    NSE_EQUITY,
    is_last_bar_of_session,
    is_partial_last_bar,
    session_date,
)
from nlt.engine.backtest import (
    _CANDLE_LENGTH,
    BARS_PER_YEAR,
    FNO_MINIMUM_CAPITAL,
    NO_CHARGES_NOTE,
    BacktestResult,
    Decision,
    Trade,
    _assert_equity_floor,
    _assert_no_overnight_carry,
    _bars_per_session_safe,
    _default_lot_size,
    _describe,
    _in_blocked_window,
    _size_position,
    _warn_on_cross_session_lookback,
    _zero_charges,
    past_entry_cutoff,
)
from nlt.engine.conditions import evaluate
from nlt.engine.features import build_features
from nlt.engine.metrics import compute_metrics
from nlt.options.contracts import Contract, format_strike
from nlt.options.execution import OptionExecution
from nlt.options.iv import _last_trading_day
from nlt.spec.models import StrategySpec

_DEFAULT_INTRADAY_BARS_PER_YEAR = 252 * 75


@dataclass
class _OptionPosition:
    contract: Contract
    contract_why: str
    entry_time: pd.Timestamp
    entry_price: float
    raw_entry_price: float
    quantity: int
    basis: float  # the option's price at the signal; stop/target measure from it
    stop_level: float | None
    target_level: float | None
    trailing_pct: float | None
    trail_best: float
    signal_time: pd.Timestamp
    last_price: float  # last traded price seen, for valuing it on untraded candles
    bars_held: int = 0
    pending_condition_exit: bool = False


def _trades_words(n: int | None) -> str:
    return "one trade" if n == 1 else f"{n} trades"


@dataclass
class _Counts:
    not_traded: int = 0
    no_price: int = 0
    no_contract: dict[str, int] = field(default_factory=dict)
    capital: int = 0
    concurrency: int = 0
    boundary: int = 0
    daily_limit: int = 0
    expiry_entry: int = 0
    expiry_exit: int = 0
    untraded_exit: int = 0
    stop_beat_target: int = 0
    risk_fallback: int = 0
    capped: int = 0


def run_option_backtest(
    spec: StrategySpec,
    bars: pd.DataFrame,
    execution: OptionExecution,
    *,
    capital: float = 100_000.0,
    lot_size: int | None = None,
    charge_fn=None,
    slippage: Slippage = ZERO,
    drop_partial_last_bar: bool = True,
    trade_from: pd.Timestamp | None = None,
    blocked_windows: list[tuple[dt.datetime, dt.datetime]] = (),
    final_session_in_progress: bool = False,
    fno_close: dt.time | None = None,
) -> BacktestResult:
    """Run a bought-option `spec`: signals on the index `bars`, trades via `execution`.

    `bars` are the underlying index's candles at the spec's timeframe;
    `execution` supplies the chosen contracts' own prices. The other
    arguments mean what they mean for `run_backtest`.

    `fno_close` is paper trading's F&O close (15:40 since 3 Aug 2026), ten
    minutes after the index's last candle. With it (Justin, 2026-10-15):

      * a position still open after the index's last candle of the day has its
        option's own prices watched until then -- stop, trailing stop, target,
        and an exit rule met on that last candle fills at the next option
        price, not the next morning (`_after_close`);
      * a position still open on its expiry day is held to that close and
        valued at its last traded price before it, instead of being closed at
        the square-off time as a backtest does.

    Backtests leave it None: the 2022-2025 dataset closes at 15:30.
    """
    if not spec.instrument.is_option:
        raise ValueError("run_option_backtest only runs option strategies")
    if spec.direction != "long":
        raise ValueError("only bought options can be backtested yet")
    if spec.exit.stop_atr_mult is not None:
        # An ATR stop is measured in index points; an option's price does not
        # move point for point with the index, so the level would mean nothing.
        raise ValueError("an ATR-based stop cannot be applied to an option's price")

    warnings: list[str] = []
    if lot_size is None:
        lot_size = _default_lot_size(spec)
    if charge_fn is None:
        charge_fn = _zero_charges
        warnings.append(NO_CHARGES_NOTE)
    warnings.append(
        f"Sized at today's lot of {lot_size} units for {spec.instrument.symbol}, used for "
        "every year tested even where the exchange's lot was different then."
    )
    if capital < FNO_MINIMUM_CAPITAL:
        warnings.append(
            f"OPTIONS ON A SMALL ACCOUNT: this was tested with Rs {capital:,.0f}. Options are "
            f"not really tradeable below about Rs {FNO_MINIMUM_CAPITAL:,.0f} – one lot is "
            f"{lot_size} units and indivisible, so a smaller account must either skip most "
            "trades or bet far too much of itself on a single contract. Treat these results "
            "as theoretical."
        )

    timeframe = spec.instrument.timeframe
    is_intraday_tf = timeframe != "1d"
    is_intraday_spec = is_intraday_tf and spec.schedule.intraday
    session = NSE_EQUITY  # NIFTY, BANKNIFTY and SENSEX all trade 09:15-15:30
    square_off = spec.schedule.square_off

    if drop_partial_last_bar and is_intraday_tf and len(bars) > 0:
        try:
            partial = is_partial_last_bar(bars, session, timeframe)
        except ValueError:
            partial = False
        if partial:
            bars = bars.iloc[:-1]

    last_day = last_day_fn(bars)
    features = build_features(spec, bars)
    entry_signal = evaluate(spec.entry, features)
    exit_signal = (
        evaluate(spec.exit.condition, features) if spec.exit.condition is not None else None
    )

    session_bar_count = _bars_per_session_safe(session, timeframe)
    if is_intraday_tf and spec.indicators and session_bar_count:
        _warn_on_cross_session_lookback(spec, session_bar_count, warnings)

    if is_intraday_tf and len(bars) > 0:
        last_bar_of_session = is_last_bar_of_session(bars.index, session).to_numpy()
        if final_session_in_progress:
            # Paper trading mid-day: today is not over, so none of today's
            # candles is its last (see `run_backtest`, which has the same rule).
            today = session_date(bars.index[-1], session)
            last_bar_of_session = last_bar_of_session.copy()
            k = len(bars) - 1
            while k >= 0 and session_date(bars.index[k], session) == today:
                last_bar_of_session[k] = False
                k -= 1
    else:
        last_bar_of_session = np.zeros(len(bars), dtype=bool)

    entry_reason = _describe(spec.entry)
    max_concurrent = spec.risk.max_concurrent_positions
    counts = _Counts()

    n = len(bars)
    equity = np.empty(n, dtype="float64")
    trades: list[Trade] = []
    decisions: list[Decision] = []
    open_positions: list[_OptionPosition] = []
    pending: tuple[pd.Timestamp, Contract, str] | None = None
    realized_pnl = 0.0

    def close_position(pos, ts, reason, raw_price, ambiguous=False, untraded=False):
        nonlocal realized_pnl
        exit_price = slippage.apply(raw_price, "sell")
        charges = charge_fn(pos.entry_price, pos.quantity, "buy") + charge_fn(
            exit_price, pos.quantity, "sell"
        )
        gross = (exit_price - pos.entry_price) * pos.quantity
        net = gross - charges
        realized_pnl += net
        if is_intraday_spec:
            _assert_no_overnight_carry(pos.entry_time, ts, session)
        if ts.date() > last_day(pos.contract.expiry):
            raise RuntimeError(
                f"an option position in {pos.contract.tradingsymbol} was held past its "
                f"expiry ({last_day(pos.contract.expiry)}) to {ts}"
            )
        trades.append(
            Trade(
                entry_time=pos.entry_time,
                entry_price=pos.entry_price,
                exit_time=ts,
                exit_price=exit_price,
                direction="long",
                quantity=pos.quantity,
                gross_pnl=gross,
                charges=charges,
                net_pnl=net,
                exit_reason=reason,
                bars_held=pos.bars_held + 1,
                entry_reason=entry_reason,
                symbol=pos.contract.tradingsymbol,
                signal_time=pos.signal_time,
                stop_level=pos.stop_level,
                target_level=pos.target_level,
                raw_entry_price=pos.raw_entry_price,
                raw_exit_price=raw_price,
                both_hit=bool(ambiguous),
                contract_why=pos.contract_why,
                exit_untraded=untraded,
            )
        )

    daily_limit = spec.schedule.max_entries_per_day
    entries_by_day: dict = {}

    for i in range(n):
        ts = bars.index[i]

        # --- 1. fill the entry queued by the previous candle's signal ---------
        if pending is not None:
            signal_ts, contract, why = pending
            pending = None
            label = _contract_label(contract)
            option_bar = execution.bar(contract, ts)
            basis = execution.price_at_signal(contract, signal_ts)
            if len(open_positions) >= max_concurrent:
                counts.concurrency += 1
                decisions.append(
                    Decision(signal_ts, contract.tradingsymbol, "signal_skipped", "concurrency", {})
                )
            elif daily_limit is not None and entries_by_day.get(ts.date(), 0) >= daily_limit:
                counts.daily_limit += 1
                decisions.append(
                    Decision(signal_ts, contract.tradingsymbol, "signal_skipped", "daily_limit", {})
                )
            elif option_bar is None or not bool(option_bar["traded"]):
                counts.not_traded += 1
                decisions.append(
                    Decision(
                        signal_ts,
                        contract.tradingsymbol,
                        "signal_skipped",
                        "option_not_traded",
                        {
                            "why": why,
                            "message": f"{label} did not trade in the {ts:%d %b %Y %H:%M} "
                            "candle, so this signal was skipped – never another strike.",
                        },
                    )
                )
            elif basis is None:
                counts.no_price += 1
                decisions.append(
                    Decision(
                        signal_ts,
                        contract.tradingsymbol,
                        "signal_skipped",
                        "no_option_price",
                        {
                            "why": why,
                            "message": f"{label} had not traded that day by the signal, so "
                            "there was no price to measure the stop and target from; the "
                            "signal was skipped.",
                        },
                    )
                )
            else:
                pos, used_fallback, capped = _open(
                    spec,
                    contract,
                    why,
                    ts,
                    option_bar,
                    basis,
                    signal_ts,
                    slippage,
                    lot_size,
                    capital + realized_pnl,
                    capital
                    + realized_pnl
                    - sum(p.entry_price * p.quantity for p in open_positions),
                )
                if pos is None:
                    counts.capital += 1
                    reason = "position_cap" if spec.risk.max_position_pct is not None else "capital"
                    decisions.append(
                        Decision(signal_ts, contract.tradingsymbol, "signal_skipped", reason, {})
                    )
                else:
                    open_positions.append(pos)
                    entries_by_day[ts.date()] = entries_by_day.get(ts.date(), 0) + 1
                    counts.risk_fallback += int(used_fallback)
                    counts.capped += int(capped)

        # --- 2. exits ---------------------------------------------------------
        still_open: list[_OptionPosition] = []
        for pos in open_positions:
            outcome = _check_exit(
                pos,
                execution,
                ts,
                spec,
                is_intraday_tf,
                is_intraday_spec,
                bool(last_bar_of_session[i]),
                square_off,
                last_day,
                fno_close,
                _CANDLE_LENGTH.get(timeframe),
            )
            if outcome is None:
                pos.bars_held += 1
                still_open.append(pos)
                continue
            reason, raw_price, ambiguous, untraded = outcome
            if reason == "expiry":
                counts.expiry_exit += 1
            if ambiguous:
                counts.stop_beat_target += 1
            if untraded:
                counts.untraded_exit += 1
            close_position(pos, ts, reason, raw_price, ambiguous, untraded)
        open_positions = still_open

        # --- 3. mark to market at each option's last traded price ------------
        unrealized = sum((p.last_price - p.entry_price) * p.quantity for p in open_positions)
        equity[i] = capital + realized_pnl + unrealized
        _assert_equity_floor(equity[i], ts, "long")

        # --- 4. this closed candle's signal queues a fill for the next one ----
        entry_true = bool(entry_signal.iloc[i])
        trade_from_ok = trade_from is None or ts >= trade_from
        if entry_true and trade_from_ok:
            skip = None
            if len(open_positions) >= max_concurrent:
                skip = "concurrency_at_signal"
            elif ts.weekday() not in spec.schedule.weekdays:
                skip = "weekday"
            elif _in_blocked_window(ts, blocked_windows):
                skip = "paused"
            elif past_entry_cutoff(spec, ts, timeframe):
                skip = "no_entry_after"
            elif daily_limit is not None and entries_by_day.get(ts.date(), 0) >= daily_limit:
                counts.daily_limit += 1
                skip = "daily_limit"
            elif is_intraday_spec and bool(last_bar_of_session[i]):
                counts.boundary += 1
                skip = "boundary"
            elif i + 1 >= n:
                # Nothing left in this run to fill on; paper trading's next pass
                # will have the candle this promises.
                decisions.append(Decision(ts, spec.instrument.symbol, "signal_taken", "queued", {}))
            if skip is not None:
                decisions.append(Decision(ts, spec.instrument.symbol, "signal_skipped", skip, {}))
            elif i + 1 < n:
                resolution = execution.resolve(ts.to_pydatetime(), float(bars["close"].iloc[i]))
                contract = resolution.contract
                fill_ts = bars.index[i + 1]
                if contract is None:
                    code = resolution.reason_code or "no_contract"
                    counts.no_contract[code] = counts.no_contract.get(code, 0) + 1
                    decisions.append(
                        Decision(
                            ts,
                            spec.instrument.symbol,
                            "signal_skipped",
                            code,
                            {"why": resolution.why, "message": resolution.why},
                        )
                    )
                elif _past_last_close(contract, fill_ts, timeframe, square_off, last_day):
                    counts.expiry_entry += 1
                    decisions.append(
                        Decision(
                            ts,
                            contract.tradingsymbol,
                            "signal_skipped",
                            "expiry",
                            {
                                "why": resolution.why,
                                "message": f"{_contract_label(contract)} expires at "
                                f"{square_off:%H:%M} that day, before this signal could be "
                                "filled, so it was skipped.",
                            },
                        )
                    )
                else:
                    decisions.append(
                        Decision(
                            ts,
                            contract.tradingsymbol,
                            "signal_taken",
                            "queued",
                            {"why": resolution.why, "contract": contract.tradingsymbol},
                        )
                    )
                    pending = (ts, contract, resolution.why)

        if exit_signal is not None and bool(exit_signal.iloc[i]):
            for pos in open_positions:
                pos.pending_condition_exit = True

        # --- 5. after the index's last candle, the options still trade --------
        if fno_close is not None and is_intraday_tf and bool(last_bar_of_session[i]):
            still_open = []
            for pos in open_positions:
                outcome = _after_close(pos, execution, ts, fno_close, last_day)
                if outcome is None:
                    still_open.append(pos)
                    continue
                reason, at, raw_price, ambiguous, untraded = outcome
                if reason == "expiry":
                    counts.expiry_exit += 1
                if ambiguous:
                    counts.stop_beat_target += 1
                if untraded:
                    counts.untraded_exit += 1
                close_position(pos, at, reason, raw_price, ambiguous, untraded)
            open_positions = still_open
            unrealized = sum((p.last_price - p.entry_price) * p.quantity for p in open_positions)
            equity[i] = capital + realized_pnl + unrealized

    # --- 5. anything still open at the end of the data ----------------------
    if open_positions:
        ts = bars.index[-1]
        for pos in open_positions:
            close_position(pos, ts, "end_of_data", pos.last_price)
        equity[-1] = capital + realized_pnl
        _assert_equity_floor(equity[-1], ts, "long")

    warnings.extend(_count_warnings(counts, square_off, spec))

    equity_series = pd.Series(equity, index=bars.index, name="equity")
    bars_per_year = BARS_PER_YEAR.get(timeframe, _DEFAULT_INTRADAY_BARS_PER_YEAR)
    metrics = compute_metrics(trades, equity_series, capital, bars_per_year)
    return BacktestResult(
        spec=spec,
        trades=trades,
        equity=equity_series,
        metrics=metrics,
        features=features,
        warnings=warnings,
        decisions=decisions,
    )


def _contract_label(contract: Contract) -> str:
    return f"{contract.underlying} {format_strike(contract.strike)} {contract.option_type}"


def last_day_fn(bars: pd.DataFrame):
    """expiry -> the session it really expired on, from the index's own sessions.

    A listed expiry that was a market holiday is the session before it
    (`nlt.options.iv._last_trading_day`); one past the last session in `bars`
    is taken as listed.
    """
    sessions = sorted(set(bars.index.date))
    cache: dict[dt.date, dt.date] = {}

    def last_day(expiry: dt.date) -> dt.date:
        if expiry not in cache:
            cache[expiry] = _last_trading_day(expiry, sessions)
        return cache[expiry]

    return last_day


def _past_last_close(
    contract: Contract,
    fill_ts: pd.Timestamp,
    timeframe: str,
    square_off: dt.time,
    last_day=lambda expiry: expiry,
) -> bool:
    """Would a fill at `fill_ts` be at or after the contract's final close time?

    On daily candles the fill is the day's open, before the expiry-day
    square-off, so only a day after expiry is too late.
    """
    day = fill_ts.date()
    expiry = last_day(contract.expiry)
    if day > expiry:
        return True
    if timeframe == "1d":
        return False
    return day == expiry and fill_ts.time() >= square_off


def _open(
    spec: StrategySpec,
    contract: Contract,
    why: str,
    ts: pd.Timestamp,
    option_bar: pd.Series,
    basis: float,
    signal_ts: pd.Timestamp,
    slippage: Slippage,
    lot_size: int,
    capital_total: float,
    capital_spare: float,
) -> tuple[_OptionPosition | None, bool, bool]:
    raw = float(option_bar["open"])
    entry_price = slippage.apply(raw, "buy")

    stop_level = None
    stop_distance = None
    if spec.exit.stop_pct is not None:
        stop_distance = basis * spec.exit.stop_pct / 100.0
        stop_level = entry_price - stop_distance
    target_level = None
    if spec.exit.target_pct is not None:
        target_level = entry_price + basis * spec.exit.target_pct / 100.0
    trailing_pct = spec.exit.trailing_stop_pct
    sizing_distance = stop_distance
    if sizing_distance is None and trailing_pct is not None:
        sizing_distance = entry_price * trailing_pct / 100.0

    quantity, used_fallback = _size_position(
        spec, entry_price, capital_total, sizing_distance, lot_size
    )
    # A bought option is paid for in full: the premium is the whole cost.
    lot_cost = entry_price * lot_size
    affordable = max(math.floor(capital_spare / lot_cost), 0) * lot_size if lot_cost > 0 else 0
    capped = False
    if quantity > affordable:
        quantity = affordable
        capped = True
    if quantity <= 0:
        return None, used_fallback, False

    return (
        _OptionPosition(
            contract=contract,
            contract_why=why,
            entry_time=ts,
            entry_price=entry_price,
            raw_entry_price=raw,
            quantity=quantity,
            basis=basis,
            stop_level=stop_level,
            target_level=target_level,
            trailing_pct=trailing_pct,
            trail_best=entry_price,
            signal_time=signal_ts,
            last_price=float(option_bar["close"]),
        ),
        used_fallback,
        capped,
    )


def _check_exit(
    pos: _OptionPosition,
    execution: OptionExecution,
    ts: pd.Timestamp,
    spec: StrategySpec,
    is_intraday_tf: bool,
    is_intraday_spec: bool,
    is_last_bar_of_session: bool,
    square_off: dt.time,
    last_day=lambda expiry: expiry,
    fno_close: dt.time | None = None,
    candle: pd.Timedelta | None = None,
) -> tuple[str, float, bool, bool] | None:
    """(reason, raw price before slippage, stop-and-target-both-hit, untraded) or None.

    Precedence as in `run_backtest`: square-off (and, for options, the
    expiry-day close, which is a square-off forced by the contract), stop,
    trailing stop, target, condition, max bars held.
    """
    bar = execution.bar(pos.contract, ts)
    traded = bar is not None and bool(bar["traded"])
    if traded:
        pos.last_price = float(bar["close"])

    on_expiry_day = ts.date() == last_day(pos.contract.expiry)
    clock_reached = is_intraday_tf and reaches_clock(ts, candle, square_off)
    forced = None
    if is_intraday_spec and (clock_reached or is_last_bar_of_session):
        forced = "square_off"
    elif on_expiry_day and fno_close is not None:
        # Paper: held to the close, valued at the last trade before it. A
        # daily candle already spans the whole day; on intraday candles the
        # stop and target are watched below and then by `_after_close`.
        if not is_intraday_tf:
            price, untraded = expiry_close_price(execution, pos.contract, ts.date(), fno_close)
            return "expiry", (pos.last_price if price is None else price), False, untraded
    elif on_expiry_day and (not is_intraday_tf or clock_reached or is_last_bar_of_session):
        forced = "expiry"
    if forced is not None:
        if not is_intraday_tf:
            # A daily candle has no 15:15 in it: read the minute prices.
            priced = square_off_price(execution, pos.contract, ts, square_off)
            if priced is not None:
                return forced, priced[0], False, priced[1]
            return forced, pos.last_price, False, True
        # The clock may fall inside this candle (15:25 in the 15:15 candle):
        # the stop and target still apply to the minutes before it.
        before = minutes_before_clock(execution, pos.contract, ts, square_off)
        if before is not None:
            pos.last_price = float(before["close"])
            hit = _price_exit(pos, before)
            if hit is not None:  # an exit rule pending fills at the candle's open
                return hit[0], hit[1], hit[2], False
        priced = square_off_price(execution, pos.contract, ts, square_off)
        if priced is not None:
            return forced, priced[0], False, priced[1]
        return forced, pos.last_price, False, True

    if not traded:
        return None  # nothing traded in this candle, so no exit could fill

    hit = _price_exit(pos, bar)
    if hit is not None:
        reason, price, ambiguous = hit
        return reason, price, ambiguous, False
    if spec.exit.max_bars_held is not None and pos.bars_held + 1 >= spec.exit.max_bars_held:
        return "max_bars", float(bar["close"]), False, False
    return None


def expiry_close_price(
    execution: OptionExecution, contract: Contract, day: dt.date, close: dt.time
) -> tuple[float | None, bool]:
    """(last traded price before `close` on expiry `day`, untraded that day).

    Untraded is True when the last trade was on an earlier day; None when the
    contract never traded at all."""
    boundary = pd.Timestamp(dt.datetime.combine(day, close), tz="Asia/Kolkata")
    last = execution.last_trade(contract, boundary.to_pydatetime())
    if last is None:
        return None, True
    price, at = last
    return price, pd.Timestamp(at).date() != day


def _price_exit(pos: _OptionPosition, bar: pd.Series) -> tuple[str, float, bool] | None:
    """(reason, raw price, stop-and-target-both-hit) on one traded option
    candle: stop first, then trailing stop, target, and an exit rule decided
    on the previous candle -- `_check_exit`'s order."""
    o, h, low = float(bar["open"]), float(bar["high"]), float(bar["low"])
    target_hit = pos.target_level is not None and h >= pos.target_level
    if pos.stop_level is not None and low <= pos.stop_level:
        return "stop", (o if o < pos.stop_level else pos.stop_level), target_hit
    if pos.trailing_pct is not None:
        pos.trail_best = max(pos.trail_best, h)
        trail_level = pos.trail_best * (1.0 - pos.trailing_pct / 100.0)
        if low <= trail_level:
            return "trailing_stop", (o if o < trail_level else trail_level), target_hit
    if target_hit:
        return "target", (o if o > pos.target_level else pos.target_level), False
    if pos.pending_condition_exit:
        return "condition", o, False
    return None


def _after_close(
    pos: _OptionPosition,
    execution: OptionExecution,
    ts: pd.Timestamp,
    fno_close: dt.time,
    last_day,
) -> tuple[str, pd.Timestamp, float, bool, bool] | None:
    """(reason, time, raw price, both hit, untraded) for a position still open
    after the index's last candle `ts`, from its option's candles up to the F&O
    close; None if it stays open overnight.

    The index has no candles after 15:29, so nothing else would look at them.
    On expiry day, a position that survives them is valued at its last trade.
    """
    day = ts.date()
    for at in execution.bar_times(pos.contract, ts, fno_close):
        bar = execution.bar(pos.contract, at)
        if bar is None or not bool(bar["traded"]):
            continue
        pos.last_price = float(bar["close"])
        hit = _price_exit(pos, bar)
        if hit is not None:
            reason, price, ambiguous = hit
            return reason, at, price, ambiguous, False
    if day == last_day(pos.contract.expiry):
        price, untraded = expiry_close_price(execution, pos.contract, day, fno_close)
        last = execution.last_trade(
            pos.contract,
            pd.Timestamp(dt.datetime.combine(day, fno_close), tz="Asia/Kolkata").to_pydatetime(),
        )
        at = pd.Timestamp(last[1]) if last and not untraded else ts
        return "expiry", at, (pos.last_price if price is None else price), False, untraded
    return None


def _day_end(day: dt.date) -> dt.datetime:
    return pd.Timestamp(
        dt.datetime.combine(day, dt.time(23, 59)), tz="Asia/Kolkata"
    ).to_pydatetime()


def reaches_clock(ts: pd.Timestamp, candle: pd.Timedelta | None, clock: dt.time) -> bool:
    """Does the candle starting at `ts` reach `clock` -- contain it, or start after it?"""
    at = ts.normalize() + pd.Timedelta(hours=clock.hour, minutes=clock.minute)
    return ts >= at if candle is None else ts + candle > at


def minutes_before_clock(
    execution: OptionExecution, contract: Contract, ts: pd.Timestamp, clock: dt.time
) -> pd.Series | None:
    """The contract's traded minutes from `ts` to the end of the `clock` minute,
    as one candle (open, high, low, close), or None if it did not trade then."""
    day = ts.date()
    at = ts.normalize() + pd.Timedelta(hours=clock.hour, minutes=clock.minute)
    minutes = execution.source.contract_bars(
        contract.underlying, contract.tradingsymbol, "1m", day, day
    )
    if minutes is None or len(minutes) == 0:
        return None
    traded = minutes[minutes["traded"]] if "traded" in minutes else minutes
    # Up to and including the clock's own minute: the square-off is at its end.
    inside = traded[(traded.index >= ts) & (traded.index < at + pd.Timedelta(minutes=1))]
    if inside.empty:
        return None
    return pd.Series(
        {
            "open": float(inside["open"].iloc[0]),
            "high": float(inside["high"].max()),
            "low": float(inside["low"].min()),
            "close": float(inside["close"].iloc[-1]),
            "traded": True,
        }
    )


def square_off_price(
    execution: OptionExecution,
    contract: Contract,
    ts: pd.Timestamp,
    clock: dt.time,
) -> tuple[float, bool] | None:
    """(price, untraded) at the end of the `clock` minute -- the price at
    15:15:59 for a 15:15 square-off (Justin, 2026-10-15): the close of that
    1-minute candle, whatever the strategy's candle size.

    If nothing traded in that minute, the last trade before it that day,
    flagged untraded; None if it had not traded at all that day.
    """
    at = ts.normalize() + pd.Timedelta(hours=clock.hour, minutes=clock.minute)
    day = ts.date()
    minutes = execution.source.contract_bars(
        contract.underlying, contract.tradingsymbol, "1m", day, day
    )
    if minutes is None or len(minutes) == 0:
        return None
    traded = minutes[minutes["traded"]] if "traded" in minutes else minutes
    upto = traded[(traded.index.date == day) & (traded.index < at + pd.Timedelta(minutes=1))]
    if upto.empty:
        return None
    return float(upto["close"].iloc[-1]), upto.index[-1] != at


def _price_at_clock(
    execution: OptionExecution, contract: Contract, day: dt.date, clock: dt.time
) -> float | None:
    """The first traded 1-minute open at or after `clock` on `day`, if any."""
    minutes = execution.source.contract_bars(
        contract.underlying, contract.tradingsymbol, "1m", day, day
    )
    if minutes is None or len(minutes) == 0:
        return None
    traded = minutes[minutes["traded"]] if "traded" in minutes else minutes
    at_or_after = traded[traded.index.time >= clock]
    at_or_after = at_or_after[at_or_after.index.date == day]
    if at_or_after.empty:
        return None
    return float(at_or_after["open"].iloc[0])


def _count_warnings(counts: _Counts, square_off: dt.time, spec: StrategySpec) -> list[str]:
    out: list[str] = []
    if counts.not_traded:
        out.append(
            f"{counts.not_traded} signal(s) skipped: the chosen option did not trade in the "
            "candle we would have bought in. Another strike is never used instead."
        )
    if counts.no_price:
        out.append(
            f"{counts.no_price} signal(s) skipped: the chosen option had not traded that day "
            "by the signal, so there was no price to measure the stop and target from."
        )
    for code, k in sorted(counts.no_contract.items()):
        if code == "no_weekly":
            out.append(
                f"{k} signal(s) skipped: {spec.instrument.symbol} had no weekly options on "
                "those days, so no contract could be chosen."
            )
        else:
            out.append(f"{k} signal(s) skipped: no suitable option contract was listed.")
    if counts.capital:
        pct = spec.risk.max_position_pct
        why = (
            f"one position may not exceed {pct:g}% of the account, and a single lot cost more"
            if pct is not None
            else "not enough spare money for even one lot"
        )
        out.append(f"{counts.capital} signal(s) skipped: {why}.")
    if counts.concurrency:
        out.append(
            f"{counts.concurrency} signal(s) skipped: the most positions allowed at once were "
            "already open."
        )
    if counts.daily_limit:
        out.append(
            f"{counts.daily_limit} signal(s) skipped: this strategy opens at most "
            f"{_trades_words(spec.schedule.max_entries_per_day)} a day."
        )
    if counts.boundary:
        out.append(
            f"{counts.boundary} signal(s) skipped: they fired on the day's last candle, and "
            "this strategy squares off the same day."
        )
    if counts.expiry_entry:
        out.append(
            f"{counts.expiry_entry} signal(s) skipped: the chosen option expired before the "
            "trade could be placed."
        )
    if counts.expiry_exit:
        out.append(
            f"{counts.expiry_exit} position(s) closed at {square_off:%H:%M} on their expiry "
            "day. An option is never held to settlement here."
        )
    if counts.untraded_exit:
        out.append(
            f"{counts.untraded_exit} position(s) closed at the option's last traded price: it "
            "did not trade at the moment of closing, so no fresher price exists."
        )
    if counts.stop_beat_target:
        out.append(
            f"{counts.stop_beat_target} trade(s) could have hit both the stop loss and the "
            "target within one candle; the stop was assumed to come first, the cautious choice."
        )
    if counts.risk_fallback:
        out.append(
            f"{counts.risk_fallback} trade(s) had no stop-loss distance to size the risk from, "
            "so one lot was bought."
        )
    if counts.capped:
        out.append(f"{counts.capped} trade(s) sized down to what the account could pay for.")
    return out


__all__ = ["run_option_backtest"]
