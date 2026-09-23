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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

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
) -> BacktestResult:
    """Run `spec` bar by bar over `bars`.

    `lot_size` is optional and should almost always be left unset: it is
    derived from `spec.instrument.trade_as` (see `_default_lot_size`) so that
    an index backtest trades in units of 1 and only an options backtest
    inherits the 75-share options convention. Pass it explicitly only to
    model a specific contract's lot size (e.g. BANKNIFTY's, which differs
    from NIFTY's).
    """
    warnings: list[str] = []
    if lot_size is None:
        lot_size = _default_lot_size(spec)
    if charge_fn is None:
        charge_fn = _zero_charges
        warnings.append("no charge_fn supplied; results exclude transaction costs")

    features = build_features(spec, bars)
    entry_signal = evaluate(spec.entry, features)
    exit_signal = (
        evaluate(spec.exit.condition, features) if spec.exit.condition is not None else None
    )

    is_intraday_tf = spec.instrument.timeframe != "1d"
    if not is_intraday_tf and spec.schedule.intraday:
        warnings.append(
            "schedule time-of-day rules (no_entry_after/square_off) skipped: "
            "instrument timeframe is 1d, where bar timestamps carry no intraday clock"
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
            outcome = _check_exit(pos, bar, ts, spec, is_intraday_tf)
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
                )
            )
        open_positions = still_open

        # --- 3. mark-to-market equity for this bar --------------------------
        unrealized = sum(_gross_pnl(pos, bar.close) for pos in open_positions)
        equity[i] = capital + realized_pnl + unrealized
        _assert_equity_floor(equity[i], ts, spec.direction)

        # --- 4. schedule fills for the NEXT bar, from this closed bar's signal
        can_enter = (
            len(open_positions) < max_concurrent
            and bool(entry_signal.iloc[i])
            and ts.weekday() in spec.schedule.weekdays
            and (not is_intraday_tf or ts.time() <= spec.schedule.no_entry_after)
        )
        if can_enter and i + 1 < n:
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

    stop_distance = _fixed_stop_distance(spec, features, i)
    stop_level = None
    if stop_distance is not None:
        stop_level = (
            entry_price - stop_distance if direction == "long" else entry_price + stop_distance
        )

    target_level = None
    if spec.exit.target_pct is not None:
        delta = entry_price * spec.exit.target_pct / 100.0
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


def _fixed_stop_distance(spec: StrategySpec, features: pd.DataFrame, i: int) -> float | None:
    """The hard-stop distance in price points, fixed at entry for the trade's life.

    `stop_pct` and `stop_atr_mult` may both be set; the tighter of the two wins,
    since that is the one that would actually be hit first. A trailing-only
    strategy has no fixed distance here -- its stop is computed bar by bar.
    """
    candidates: list[float] = []
    exit_rules = spec.exit
    if exit_rules.stop_pct is not None:
        candidates.append(features["close"].iloc[i] * exit_rules.stop_pct / 100.0)
    if exit_rules.stop_atr_mult is not None and exit_rules.atr_id is not None:
        atr_val = features[exit_rules.atr_id].iloc[i]
        if pd.notna(atr_val):
            candidates.append(exit_rules.stop_atr_mult * float(atr_val))
    return min(candidates) if candidates else None


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
) -> tuple[str, float, bool] | None:
    """Returns (reason, raw fill price before slippage, ambiguous_stop_vs_target) or None.

    Checked in the precedence the brief mandates: square-off, stop, trailing
    stop, target, condition, max_bars_held. The first one that fires wins --
    a bar that hits both a stop and a target only ever reports the stop.
    """
    long = pos.direction == "long"

    if is_intraday_tf and spec.schedule.intraday and ts.time() >= spec.schedule.square_off:
        return "square_off", bar.close, False

    target_hit = pos.target_level is not None and (
        bar.high >= pos.target_level if long else bar.low <= pos.target_level
    )

    if pos.stop_level is not None:
        stop_hit = bar.low <= pos.stop_level if long else bar.high >= pos.stop_level
        if stop_hit:
            return "stop", pos.stop_level, target_hit

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
        return "target", pos.target_level, False

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
