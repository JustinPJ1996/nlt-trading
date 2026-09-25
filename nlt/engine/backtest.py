"""The backtest loop.

Everything else in `nlt.engine` is vectorised; this file is deliberately not.
Position state -- what is open, how large, where its stop has ratcheted to --
is path-dependent, so it has to be walked bar by bar. What stays disciplined
here is *when* information is allowed to affect a fill:

  * A signal is a property of a *closed* bar. It can only ever fill on the bar
    after it was observed, at that bar's open -- never on the bar that produced
    it, and never at that bar's own close (which is where every look-ahead bug
    in a hand-rolled backtester actually hides).
  * Stop/target/trailing levels are known the instant a position opens, so they
    are allowed to trigger intrabar, starting on the very bar the position
    fills, using that bar's high/low.
  * OHLC bars cannot tell you whether the high or the low came first. When a
    stop and a target are both inside one bar's range, this engine always
    assumes the stop filled first. It is the pessimistic assumption, which is
    the only one that does not make a backtest look better than the strategy
    actually is.

Two more invariants that broke in an earlier version of this file, in a way
that produced a silently-wrong -113% "total return" on a real strategy:

  * A position is never opened for more than the account can fund. `lot_size`
    used to default to 75 (an *options* lot) even for a cash-index backtest,
    so "1 lot" of NIFTY meant 75x the index on Rs 1 lakh of capital -- 4-17x
    leverage, reported as if it were a real result. The default is now derived
    from `spec.instrument.trade_as`, and every fill is additionally capped to
    what the account's spare capital can actually buy (see `_open_position`).
  * Equity in a long-only backtest can never go negative: the worst case is
    losing the entire position, not more than it. `run_backtest` asserts this
    after every bar instead of trusting the arithmetic to get it right.

Intraday timeframes (anything but "1d") add a session clock daily bars do not
have, and with it a set of failure modes that lose real money in a way a bad
Sharpe ratio does not: a position quietly held overnight, an entry filled
across a 14-hour gap because the signal happened to land on a session's last
bar, a signal acted on before its own candle finished printing. `is_intraday_tf`
(`spec.instrument.timeframe != "1d"`) gates every one of the following, so
daily-bar behaviour -- and every test that predates it -- is provably
unaffected:

  * A session's last bar never schedules an entry fill for "the next bar":
    that next bar is the following session's open, not 15/30/60 minutes away,
    and this engine refuses to fill across that gap (see the "skip the entry
    outright" note by the `can_enter` block for why, over filling it anyway).
  * `square_off` forces every open position closed, ahead of every other exit
    reason -- including on a session that ends before the square-off clock
    time is ever reached (a short/muhurat day), via `is_last_bar_of_session`
    rather than a hardcoded clock.
  * `_assert_no_overnight_carry` makes "no intraday position survives a
    session boundary" a hard invariant, not a hope: it raises rather than
    warn, because a warning would let a silently-wrong "intraday" result reach
    someone who explicitly asked never to hold overnight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nlt.data.session import (
    NSE_EQUITY,
    Session,
    bars_per_session,
    is_last_bar_of_session,
    is_partial_last_bar,
    session_date,
)
from nlt.engine.conditions import evaluate
from nlt.engine.features import build_features
from nlt.engine.metrics import compute_metrics
from nlt.spec.models import (
    All,
    Any_,
    Compare,
    Condition,
    Const,
    IsTrue,
    Not,
    Operand,
    PercentChange,
    Ref,
    StrategySpec,
)

BARS_PER_YEAR = {"1d": 252}
_DEFAULT_INTRADAY_BARS_PER_YEAR = 252 * 75  # rough fallback; only 1d is exercised today


def _zero_charges(price: float, quantity: int, side: str) -> float:
    return 0.0


def _default_lot_size(spec: StrategySpec) -> int:
    """The tradeable unit when the caller does not specify one.

    75 is NIFTY's *options* lot size -- a contract term, not a property of the
    index. A cash-index backtest that inherited it as a default would buy 75
    units of NIFTY per "1 lot", which on a modest account is several times
    leverage the caller never asked for (this is exactly how the engine used
    to report a -113% total return on a strategy that could not lose more than
    its stop loss). An index backtest trades in units of 1; only an options
    backtest inherits the options convention.
    """
    return 75 if spec.instrument.trade_as == "option" else 1


def _session_for_spec(spec: StrategySpec) -> Session:
    """The trading-hours calendar `spec.instrument` runs on.

    `Instrument.symbol` today only permits NIFTY and BANKNIFTY, both NSE
    equity & F&O names, so this always resolves to `NSE_EQUITY`. `nlt.data.session`
    also defines `MCX` (a session that runs to 23:30, well past NSE's close), but
    there is no commodity symbol in the spec model yet to map onto it. Everything
    below is written against `Session` generically -- once `Instrument.symbol`
    grows an MCX name, this function is the only place that needs to change.
    """
    return NSE_EQUITY


def _bars_per_session_safe(session: Session, timeframe: str) -> int | None:
    """`bars_per_session`, or None if the timeframe has no known session length.

    `nlt.data.session` only knows bar lengths for a subset of the timeframes
    `Instrument.timeframe` allows (notably: not "3m"). Rather than let that
    raise and crash the whole backtest, the checks that depend on it (partial
    last-bar dropping, the lookback/session warning) are skipped for a
    timeframe it does not recognise, with a warning explaining why.
    """
    try:
        return bars_per_session(session, timeframe)
    except ValueError:
        return None


@dataclass(frozen=True)
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    direction: str
    quantity: int
    gross_pnl: float
    charges: float
    net_pnl: float
    exit_reason: str
    bars_held: int
    entry_reason: str
    symbol: str = ""


@dataclass
class BacktestResult:
    spec: StrategySpec
    trades: list[Trade]
    equity: pd.Series
    metrics: dict
    features: pd.DataFrame
    warnings: list[str]


@dataclass
class _OpenPosition:
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    quantity: int
    stop_level: float | None       # fixed stop (from stop_pct / stop_atr_mult), set once at entry
    target_level: float | None
    trailing_pct: float | None
    trail_best: float              # best price seen since entry; only moves favourably
    bars_held: int = 0
    pending_condition_exit: bool = False


def run_backtest(
    spec: StrategySpec,
    bars: pd.DataFrame,
    *,
    capital: float = 100_000.0,
    lot_size: int | None = None,
    charge_fn=None,
    slippage_pct: float = 0.02,
    drop_partial_last_bar: bool = True,
) -> BacktestResult:
    """Run `spec` bar by bar over `bars`.

    `lot_size` is optional and should almost always be left unset: it is
    derived from `spec.instrument.trade_as` (see `_default_lot_size`) so that
    an index backtest trades in units of 1 and only an options backtest
    inherits the 75-share options convention. Pass it explicitly only to
    model a specific contract's lot size (e.g. BANKNIFTY's, which differs
    from NIFTY's).

    `drop_partial_last_bar`, when the timeframe is intraday, drops the final
    bar of `bars` if it is still forming (its close time is in the future
    relative to now) rather than let the engine signal on it. The data layer
    (`nlt.data.session.is_partial_last_bar`) is expected to have already
    filtered this out upstream; this is defence in depth specifically because
    the engine is what places orders, so it should not trust that every caller
    remembered the upstream guard.
    """
    warnings: list[str] = []
    if lot_size is None:
        lot_size = _default_lot_size(spec)
    if charge_fn is None:
        charge_fn = _zero_charges
        warnings.append("no charge_fn supplied; results exclude transaction costs")

    is_intraday_tf = spec.instrument.timeframe != "1d"
    session = _session_for_spec(spec)
    session_bar_count = _bars_per_session_safe(session, spec.instrument.timeframe)

    # --- drop a still-forming final bar before anything downstream sees it --
    # (daily bars have no intrabar clock to be "partial" about, and 1d is not
    # a key `nlt.data.session` recognises, so this only ever applies intraday.)
    if drop_partial_last_bar and is_intraday_tf and len(bars) > 0:
        try:
            partial = is_partial_last_bar(bars, session, spec.instrument.timeframe)
        except ValueError:
            partial = False
        if partial:
            warnings.append(
                f"dropped the final bar ({bars.index[-1]}) as still-forming: its close "
                "time has not arrived yet, so signalling on it would be trading a candle "
                "that has not finished printing"
            )
            bars = bars.iloc[:-1]

    features = build_features(spec, bars)
    entry_signal = evaluate(spec.entry, features)
    exit_signal = (
        evaluate(spec.exit.condition, features) if spec.exit.condition is not None else None
    )

    if not is_intraday_tf and spec.schedule.intraday:
        warnings.append(
            "schedule time-of-day rules (no_entry_after/square_off) skipped: "
            "instrument timeframe is 1d, where bar timestamps carry no intraday clock"
        )

    # --- session bookkeeping, used only for intraday timeframes below -------
    # `last_bar_of_session` backs two independent checks: forcing square-off on
    # a short/early-close day that never reaches the square-off clock time (2),
    # and refusing to schedule an entry fill across a session boundary (1).
    if is_intraday_tf and len(bars) > 0:
        last_bar_of_session = is_last_bar_of_session(bars.index, session).to_numpy()
    else:
        last_bar_of_session = np.zeros(len(bars), dtype=bool)

    if is_intraday_tf and spec.indicators and session_bar_count:
        _warn_on_cross_session_lookback(spec, session_bar_count, warnings)
    elif is_intraday_tf and spec.indicators and session_bar_count is None:
        warnings.append(
            f"cannot determine bars-per-session for timeframe {spec.instrument.timeframe!r}; "
            "the cross-session indicator-lookback warning was skipped for it"
        )

    max_concurrent = spec.risk.max_concurrent_positions
    entry_reason = _describe(spec.entry)

    n = len(bars)
    equity = np.empty(n, dtype="float64")
    trades: list[Trade] = []
    open_positions: list[_OpenPosition] = []
    pending_entry = False
    realized_pnl = 0.0
    stop_beat_target_count = 0
    risk_fallback_count = 0
    dropped_for_concurrency = 0
    affordability_capped_count = 0
    affordability_skipped_count = 0
    boundary_entry_skipped_count = 0

    for i in range(n):
        ts = bars.index[i]
        bar = bars.iloc[i]

        # --- 1. fill an entry scheduled from the previous bar's signal -----
        if pending_entry:
            pending_entry = False
            if len(open_positions) < max_concurrent:
                capital_total = capital + realized_pnl
                # Capital already tied up in other open positions is not spare
                # capital: without netting it out, two concurrent positions
                # could each be sized as if the whole account were free,
                # re-creating the leverage bug on max_concurrent_positions > 1.
                committed = sum(p.entry_price * p.quantity for p in open_positions)
                capital_spare = capital_total - committed
                pos, used_fallback, capped = _open_position(
                    spec, features, i, ts, bar, slippage_pct, lot_size,
                    capital_total, capital_spare,
                )
                if pos is not None:
                    open_positions.append(pos)
                    if used_fallback:
                        risk_fallback_count += 1
                    if capped:
                        affordability_capped_count += 1
                else:
                    affordability_skipped_count += 1
            else:
                dropped_for_concurrency += 1

        # --- 2. check exits, in the mandated precedence order --------------
        still_open: list[_OpenPosition] = []
        for pos in open_positions:
            outcome = _check_exit(pos, bar, ts, spec, is_intraday_tf, bool(last_bar_of_session[i]))
            if outcome is None:
                pos.bars_held += 1
                still_open.append(pos)
                continue

            reason, raw_price, ambiguous = outcome
            if ambiguous:
                stop_beat_target_count += 1
            exit_side = "sell" if pos.direction == "long" else "buy"
            exit_price = _apply_slippage(raw_price, exit_side, slippage_pct)
            entry_side = "buy" if pos.direction == "long" else "sell"
            charges = charge_fn(pos.entry_price, pos.quantity, entry_side) + charge_fn(
                exit_price, pos.quantity, exit_side
            )
            gross = _gross_pnl(pos, exit_price)
            net = gross - charges
            realized_pnl += net
            if is_intraday_tf and spec.schedule.intraday:
                _assert_no_overnight_carry(pos.entry_time, ts, session)
            trades.append(
                Trade(
                    entry_time=pos.entry_time,
                    entry_price=pos.entry_price,
                    exit_time=ts,
                    exit_price=exit_price,
                    direction=pos.direction,
                    quantity=pos.quantity,
                    gross_pnl=gross,
                    charges=charges,
                    net_pnl=net,
                    exit_reason=reason,
                    bars_held=pos.bars_held + 1,
                    entry_reason=entry_reason,
                    symbol=spec.instrument.symbol,
                )
            )
        open_positions = still_open

        # --- 3. mark-to-market equity for this bar --------------------------
        unrealized = sum(_gross_pnl(pos, bar.close) for pos in open_positions)
        equity[i] = capital + realized_pnl + unrealized
        _assert_equity_floor(equity[i], ts, spec.direction)

        # --- 4. schedule fills for the NEXT bar, from this closed bar's signal
        #
        # `no_entry_after` is a strict "at or after" cutoff -- a signal on the
        # bar timestamped exactly at the cutoff must NOT open a position. Using
        # `<=` here (the engine's original condition) let a 15:00 signal open a
        # trade on a 15:00 cutoff, which is the "at" half of "at or after" being
        # silently ignored.
        is_intraday_spec = is_intraday_tf and spec.schedule.intraday
        can_enter = (
            len(open_positions) < max_concurrent
            and bool(entry_signal.iloc[i])
            and ts.weekday() in spec.schedule.weekdays
            and (not is_intraday_spec or ts.time() < spec.schedule.no_entry_after)
        )
        # A signal on the last bar of a session has no valid intraday fill: the
        # only bar left to fill it on is tomorrow morning's open, 14+ hours away
        # across a gap the user never asked to hold through. Decision: SKIP the
        # entry outright rather than fill it anyway with a warning. Filling it
        # would silently convert an intraday strategy into an overnight one on
        # exactly the bars where that matters most (the close), which is a
        # worse failure than a missed trade -- a missed trade costs the
        # strategy an entry it never gets to try; a filled-anyway entry costs
        # the user an unintended overnight position with unbounded gap risk,
        # on an account sized for intraday margin. Counted and surfaced below.
        if can_enter and is_intraday_spec and bool(last_bar_of_session[i]):
            boundary_entry_skipped_count += 1
        elif can_enter and i + 1 < n:
            pending_entry = True

        if exit_signal is not None:
            for pos in open_positions:
                if bool(exit_signal.iloc[i]):
                    pos.pending_condition_exit = True

    # --- 5. anything still open at the end of the data is forced closed ----
    if open_positions:
        ts = bars.index[-1]
        bar = bars.iloc[-1]
        for pos in open_positions:
            exit_side = "sell" if pos.direction == "long" else "buy"
            exit_price = _apply_slippage(bar.close, exit_side, slippage_pct)
            entry_side = "buy" if pos.direction == "long" else "sell"
            charges = charge_fn(pos.entry_price, pos.quantity, entry_side) + charge_fn(
                exit_price, pos.quantity, exit_side
            )
            gross = _gross_pnl(pos, exit_price)
            net = gross - charges
            realized_pnl += net
            if is_intraday_tf and spec.schedule.intraday:
                # In practice this should never fire: `last_bar_of_session` is
                # true on the very last row of `bars` by construction (nothing
                # follows it), so an intraday position still open here would
                # already have been square-off'd inside the loop above on this
                # same bar. Checked anyway -- an invariant that is "obviously"
                # unreachable is exactly the kind that a future change to the
                # exit-precedence order could quietly break.
                _assert_no_overnight_carry(pos.entry_time, ts, session)
            trades.append(
                Trade(
                    entry_time=pos.entry_time,
                    entry_price=pos.entry_price,
                    exit_time=ts,
                    exit_price=exit_price,
                    direction=pos.direction,
                    quantity=pos.quantity,
                    gross_pnl=gross,
                    charges=charges,
                    net_pnl=net,
                    exit_reason="end_of_data",
                    bars_held=pos.bars_held + 1,
                    entry_reason=entry_reason,
                    symbol=spec.instrument.symbol,
                )
            )
        equity[-1] = capital + realized_pnl
        _assert_equity_floor(equity[-1], ts, spec.direction)

    if stop_beat_target_count:
        warnings.append(
            f"{stop_beat_target_count} trade(s) had both stop and target reachable in the "
            "same bar; the stop was assumed to have filled first (pessimistic tie-break)"
        )
    if risk_fallback_count:
        warnings.append(
            f"{risk_fallback_count} trade(s) used risk_based sizing with no stop distance "
            "available; sized as 1 lot instead"
        )
    if dropped_for_concurrency:
        warnings.append(
            f"{dropped_for_concurrency} entry signal(s) skipped: max_concurrent_positions "
            f"({max_concurrent}) already reached"
        )
    if affordability_capped_count:
        warnings.append(
            f"{affordability_capped_count} entr{'y' if affordability_capped_count == 1 else 'ies'} "
            "sized down: the requested position value exceeded spare capital, so it was reduced "
            "to what the account could actually fund"
        )
    if affordability_skipped_count:
        warnings.append(
            f"{affordability_skipped_count} entry signal(s) skipped: not enough spare capital "
            "to fund even a single lot"
        )
    if boundary_entry_skipped_count:
        warnings.append(
            f"{boundary_entry_skipped_count} entry signal(s) skipped: they fired on the last "
            "bar of a trading session, and this intraday strategy would otherwise have filled "
            "them at the next session's open -- a gap of 14+ hours the strategy never asked "
            "to hold through"
        )

    equity_series = pd.Series(equity, index=bars.index, name="equity")
    bars_per_year = BARS_PER_YEAR.get(spec.instrument.timeframe, _DEFAULT_INTRADAY_BARS_PER_YEAR)
    metrics = compute_metrics(trades, equity_series, capital, bars_per_year)

    return BacktestResult(
        spec=spec,
        trades=trades,
        equity=equity_series,
        metrics=metrics,
        features=features,
        warnings=warnings,
    )


# --------------------------------------------------------------- fills & pnl


def _apply_slippage(price: float, side: str, slippage_pct: float) -> float:
    """A buy always fills a little worse (higher); a sell always fills a little worse (lower).

    This is applied to every fill -- entries and exits alike -- so a round trip
    always costs slippage twice, the same as it would in a live account.
    """
    factor = 1.0 + slippage_pct / 100.0 if side == "buy" else 1.0 - slippage_pct / 100.0
    return price * factor


def _gross_pnl(pos: _OpenPosition, exit_price: float) -> float:
    if pos.direction == "long":
        return (exit_price - pos.entry_price) * pos.quantity
    return (pos.entry_price - exit_price) * pos.quantity


def _assert_equity_floor(equity_value: float, ts: pd.Timestamp, direction: str) -> None:
    """A long-only account can lose at most what it put in, never more.

    Every long position is now capped at open time to what spare capital can
    fund (see `_open_position`), so if this ever fires it means that cap was
    bypassed -- an accounting bug, not a bad but valid backtest result. We
    raise rather than clamp-and-warn: silently clamping would hide exactly the
    class of bug (buying more than the account can afford) that this file
    exists to prevent, and a caller can always catch and inspect. Short
    positions have theoretically unbounded loss, so the invariant is only
    enforced for `direction == "long"`.
    """
    if direction == "long" and equity_value < -1e-6:
        raise RuntimeError(
            f"equity went negative ({equity_value:.2f}) at {ts} in a long-only backtest; "
            "this means a position was opened larger than the account could fund, which "
            "should be impossible after the affordability check in _open_position"
        )


def _assert_no_overnight_carry(
    entry_time: pd.Timestamp, exit_time: pd.Timestamp, session: Session
) -> None:
    """An intraday position must never wake up still holding a session boundary.

    This is the "woke up still holding it" failure mode made an invariant: for
    `schedule.intraday` specs on an intraday timeframe, entry and exit must
    belong to the same trading session date. If this ever fires, it means the
    square-off / last-bar-of-session logic above failed to force the exit in
    time -- an engine bug, not a valid (if unlucky) backtest outcome, so this
    raises rather than warns. A warning would let a silently-wrong "intraday"
    result reach a user who explicitly asked never to carry a position
    overnight.
    """
    entry_date = session_date(entry_time, session)
    exit_date = session_date(exit_time, session)
    if entry_date != exit_date:
        raise RuntimeError(
            f"intraday position opened {entry_time} (session {entry_date}) exited "
            f"{exit_time} (session {exit_date}) -- an intraday strategy must never "
            "carry a position across a session boundary; this indicates the "
            "square-off / last-bar-of-session forcing logic failed to fire"
        )


def _warn_on_cross_session_lookback(
    spec: StrategySpec, session_bar_count: int, warnings: list[str]
) -> None:
    """Informational warning: does the longest indicator lookback span multiple sessions?

    Users reason in calendar days ("a 200-day average"); indicators are computed
    in bars, with no awareness that an overnight or weekend gap sits between two
    consecutive intraday bars (see the module docstring on indicators not being
    modifiable from here -- this cannot change how they compute, only warn about
    the gap between how the user and the engine each think about "200 periods").
    `IndicatorSpec` params are keyed by convention (see `models.py`'s own
    `_params_are_valid`): a lookback-shaped parameter has "length" in its name,
    or is one of `fast`/`slow`/`signal`/`period`. This mirrors that convention
    rather than re-deriving it, since it is already the model's own definition
    of "a lookback".
    """
    longest = 0
    for ind in spec.indicators:
        for key, value in ind.resolved_params().items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if "length" in key or key in {"fast", "slow", "signal", "period"}:
                longest = max(longest, int(value))

    if longest <= 0:
        return
    sessions_spanned = math.ceil(longest / session_bar_count)
    if sessions_spanned > 1:
        warnings.append(
            f"the longest indicator lookback ({longest} bars) spans roughly "
            f"{sessions_spanned} trading session(s) at {session_bar_count} bars/session on "
            f"{spec.instrument.timeframe} bars -- it will smooth straight across the "
            "overnight/weekend gap as though those bars were consecutive trading minutes"
        )


def _open_position(
    spec: StrategySpec,
    features: pd.DataFrame,
    i: int,
    ts: pd.Timestamp,
    bar: pd.Series,
    slippage_pct: float,
    lot_size: int,
    capital_total: float,
    capital_spare: float,
) -> tuple[_OpenPosition | None, bool, bool]:
    """Returns (position or None, used_risk_fallback, size_was_capped).

    `capital_total` is capital + realized pnl, the basis `risk_based` sizing
    divides into. `capital_spare` nets out what is already tied up in other
    open positions -- it is what the *affordability* check below funds the
    new position out of, so that concurrent positions cannot each be sized as
    though the whole account were free.

    A position is never opened for more than `capital_spare` can fund: this is
    the fix for the leverage bug where a "1 lot" fixed_lots position on a
    small account could cost several times the account's capital. If not even
    one lot is affordable, no position is opened at all (returns None).
    """
    direction = spec.direction
    entry_side = "buy" if direction == "long" else "sell"
    entry_price = _apply_slippage(bar.open, entry_side, slippage_pct)

    stop_distance = _fixed_stop_distance(spec, features, i, entry_price)
    stop_level = None
    if stop_distance is not None:
        stop_level = (
            entry_price - stop_distance if direction == "long" else entry_price + stop_distance
        )

    target_level = None
    if spec.exit.target_pct is not None:
        delta = signal_close(features, i, entry_price) * spec.exit.target_pct / 100.0
        target_level = entry_price + delta if direction == "long" else entry_price - delta

    trailing_pct = spec.exit.trailing_stop_pct

    # Risk-based sizing needs *some* stop distance to divide by; if there is
    # only a trailing stop, its initial (pre-ratchet) distance stands in.
    sizing_distance = stop_distance
    if sizing_distance is None and trailing_pct is not None:
        sizing_distance = entry_price * trailing_pct / 100.0

    quantity, used_fallback = _size_position(
        spec, entry_price, capital_total, sizing_distance, lot_size
    )

    # --- affordability: never open a position the account cannot fund ------
    capped = False
    affordable_lots = math.floor(capital_spare / entry_price / lot_size) if entry_price > 0 else 0
    affordable_quantity = max(affordable_lots, 0) * lot_size
    if quantity > affordable_quantity:
        quantity = affordable_quantity
        capped = True
    if quantity <= 0:
        return None, used_fallback, False

    pos = _OpenPosition(
        direction=direction,
        entry_time=ts,
        entry_price=entry_price,
        quantity=quantity,
        stop_level=stop_level,
        target_level=target_level,
        trailing_pct=trailing_pct,
        trail_best=entry_price,
    )
    return pos, used_fallback, capped


def _fixed_stop_distance(
    spec: StrategySpec, features: pd.DataFrame, i: int, entry_price: float
) -> float | None:
    """The hard-stop distance in price points, fixed at entry for the trade's life.

    Everything here is read from bar `i-1`, the signal bar -- the last bar that had
    completed when the decision was made. Bar `i` is the bar we are filling on, and
    none of it has happened yet at the instant of the fill.

    * `stop_pct` is measured from the signal bar's close. This is a deliberate
      product choice: the levels belong to the setup the user described, so
      "RSI cracks 30, 1% stop" means 1% from the price where RSI cracked 30.
      The consequence is that an overnight gap moves the fill away from that
      base, so the realised risk can differ from the stated percentage -- the
      engine measures this and warns when it is material.
    * ATR likewise comes from bar `i-1`. Bar `i`'s ATR incorporates its own high,
      low and close, so using it would be reading the future.

    Reading bar `i`'s close here -- which the engine originally did -- is invisible
    to the prefix-invariance test, because that test only varies *future* bars and
    bar `i` is present in both runs. Misuse of the current bar is a separate
    failure mode with its own tests.

    The prefix-invariance test cannot catch either mistake: it compares runs that
    differ in *future* bars, and both values are equally available in a truncated
    run. Misusing the current bar is a separate failure mode, tested directly.

    `stop_pct` and `stop_atr_mult` may both be set; the tighter of the two wins,
    since that is the one that would actually be hit first. A trailing-only
    strategy has no fixed distance here -- its stop is computed bar by bar.
    """
    candidates: list[float] = []
    exit_rules = spec.exit
    basis = signal_close(features, i, entry_price)

    if exit_rules.stop_pct is not None:
        candidates.append(basis * exit_rules.stop_pct / 100.0)
    if exit_rules.stop_atr_mult is not None and exit_rules.atr_id is not None and i >= 1:
        atr_val = features[exit_rules.atr_id].iloc[i - 1]
        if pd.notna(atr_val):
            candidates.append(exit_rules.stop_atr_mult * float(atr_val))
    return min(candidates) if candidates else None


def signal_close(features: pd.DataFrame, i: int, fallback: float) -> float:
    """The close of bar `i-1`, the bar whose close produced the signal.

    Falls back to the fill price on the first bar, where there is no prior bar.
    Stop and target both measure from this so that "1% stop, 2% target" are two
    percentages of the same number.
    """
    if i < 1:
        return fallback
    value = features["close"].iloc[i - 1]
    return float(value) if pd.notna(value) else fallback


def _size_position(
    spec: StrategySpec,
    entry_price: float,
    capital_now: float,
    stop_distance: float | None,
    lot_size: int,
) -> tuple[int, bool]:
    """Quantity in units (already a multiple of `lot_size`), and whether a fallback was used."""
    sizing = spec.sizing
    used_fallback = False

    if sizing.mode == "fixed_lots":
        quantity = sizing.lots * lot_size
    elif sizing.mode == "fixed_value":
        lots = math.floor(sizing.value / entry_price / lot_size)
        quantity = max(lots, 1) * lot_size
    else:  # risk_based
        if stop_distance is None or stop_distance <= 0:
            quantity = lot_size
            used_fallback = True
        else:
            risk_amount = sizing.risk_pct / 100.0 * capital_now
            units = risk_amount / stop_distance
            lots = math.floor(units / lot_size)
            quantity = max(lots, 1) * lot_size

    max_quantity = spec.risk.max_lots * lot_size
    return min(quantity, max_quantity), used_fallback


# ------------------------------------------------------------------- exits


def _check_exit(
    pos: _OpenPosition,
    bar: pd.Series,
    ts: pd.Timestamp,
    spec: StrategySpec,
    is_intraday_tf: bool,
    is_last_bar_of_session: bool,
) -> tuple[str, float, bool] | None:
    """Returns (reason, raw fill price before slippage, ambiguous_stop_vs_target) or None.

    Checked in the precedence the brief mandates: square-off, stop, trailing
    stop, target, condition, max_bars_held. The first one that fires wins --
    a bar that hits both a stop and a target only ever reports the stop.
    """
    long = pos.direction == "long"

    if is_intraday_tf and spec.schedule.intraday and (
        ts.time() >= spec.schedule.square_off or is_last_bar_of_session
    ):
        # `is_last_bar_of_session` is the fallback for a short/early-close day
        # that ends before the clock ever reaches `square_off` (a muhurat
        # session, a declared early close). Without it, a position on such a
        # day would sail straight past the session's actual last bar still
        # open, and get force-closed only by the end-of-data handler using
        # that day's close -- which happens to look like a square-off in
        # effect, but does not carry the "square_off" reason and, worse, would
        # not fire at all if more bars for a *later* session follow in the
        # same `bars` frame. Deriving this from the data (via
        # `is_last_bar_of_session`) rather than a hardcoded early-close
        # calendar is the same trade-off `nlt.data.session` documents for
        # itself: it cannot tell a short session from live data that just
        # hasn't finished, and forcing the exit either way is the safe
        # direction to be wrong in.
        return "square_off", bar.close, False

    target_hit = pos.target_level is not None and (
        bar.high >= pos.target_level if long else bar.low <= pos.target_level
    )

    if pos.stop_level is not None:
        stop_hit = bar.low <= pos.stop_level if long else bar.high >= pos.stop_level
        if stop_hit:
            # A stop does not protect you through a gap. If the bar OPENED beyond
            # the stop, the level was never tradeable -- the first price you could
            # actually have got out at is the open, which may be far worse.
            #
            # Filling at the stop level regardless is the single most flattering
            # lie a backtester can tell: it makes every stop-loss strategy look
            # like its worst case is bounded by the stop, when overnight gaps are
            # exactly when large losses happen.
            gapped_through = (
                bar.open < pos.stop_level if long else bar.open > pos.stop_level
            )
            fill = bar.open if gapped_through else pos.stop_level
            return "stop", fill, target_hit

    if pos.trailing_pct is not None:
        # The best price ratchets using *this* bar's extreme before the trail
        # level is tested against this same bar's opposite extreme. That is a
        # known simplification (see module docstring on intrabar ordering):
        # a bar that both makes a new high and reverses hard can trigger the
        # trail on the bar that set it. `trail_best` itself never moves
        # backwards, which is the property the caller actually relies on.
        pos.trail_best = max(pos.trail_best, bar.high) if long else min(pos.trail_best, bar.low)
        trail_level = (
            pos.trail_best * (1.0 - pos.trailing_pct / 100.0)
            if long
            else pos.trail_best * (1.0 + pos.trailing_pct / 100.0)
        )
        trail_hit = bar.low <= trail_level if long else bar.high >= trail_level
        if trail_hit:
            return "trailing_stop", trail_level, target_hit

    if target_hit:
        # Mirror of the stop case: a gap past the target fills at the open, which
        # here is in our favour. Modelling only the unfavourable gap would bias
        # results the other way.
        gapped_past = (
            bar.open > pos.target_level if long else bar.open < pos.target_level
        )
        return "target", (bar.open if gapped_past else pos.target_level), False

    if pos.pending_condition_exit:
        return "condition", bar.open, False

    if spec.exit.max_bars_held is not None and pos.bars_held + 1 >= spec.exit.max_bars_held:
        return "max_bars", bar.close, False

    return None


# -------------------------------------------------------------- description


def _describe(condition: Condition) -> str:
    """Best-effort plain-English rendering of a condition tree, for `Trade.entry_reason`."""
    if isinstance(condition, Compare):
        left = _describe_operand(condition.left)
        right = _describe_operand(condition.right)
        verbs = {
            "lt": "<", "lte": "<=", "gt": ">", "gte": ">=", "eq": "==",
            "crosses_above": "crossed above", "crosses_below": "crossed below",
        }
        return f"{left} {verbs[condition.op]} {right}"
    if isinstance(condition, IsTrue):
        return f"{_describe_operand(condition.ref)} is true"
    if isinstance(condition, PercentChange):
        verbs = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}
        ref = _describe_operand(condition.ref)
        return (
            f"{ref} changed {verbs[condition.op]} {condition.value}% "
            f"over {condition.lookback} bar(s)"
        )
    if isinstance(condition, All):
        return " and ".join(_describe(c) for c in condition.conditions)
    if isinstance(condition, Any_):
        return " or ".join(f"({_describe(c)})" for c in condition.conditions)
    if isinstance(condition, Not):
        return f"not ({_describe(condition.condition)})"
    raise TypeError(f"unknown condition type {type(condition).__name__}")


def _describe_operand(operand: Operand | Ref) -> str:
    if isinstance(operand, Const):
        return str(operand.value)
    text = operand.key()
    if operand.bars_ago:
        text += f"[{operand.bars_ago} bars ago]"
    return text
