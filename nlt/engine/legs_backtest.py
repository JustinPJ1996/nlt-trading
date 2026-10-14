"""The backtest loop for sold options and multi-leg option positions (Phase 2, batch 2).

A straddle, a strangle, a spread, a condor, or a single sold option: one
signal on the **index** opens every leg together, each in its own contract.
Batch 1's single bought option keeps its own loop (`options_backtest`); this
one is separate for the same reason that one is separate from `run_backtest`
-- the rules every existing result depends on are not threaded through code
written for a different shape of trade.

The rules, each one Justin's or the engine's own (see `HANDOFF.md`):

  * **Signals on closed index candles; every leg fills at the next candle's
    open**, on its own option, plus the run's slippage (a sell receives a
    little less, a buy pays a little more -- `Slippage.apply`). The contracts
    are chosen at the signal from the signal candle's index close
    (`OptionExecution.resolve_leg`: strikes counted from ATM, or the listed
    strike nearest spot + points, refused -- never clamped -- when nothing is
    listed within one strike gap).
  * **All legs or none.** If any leg's option did not trade in the fill
    candle, or had no price yet that day to measure its stop from, the whole
    entry is skipped and the reason names the leg. A strike is never
    substituted and a position is never opened with only some of its legs.
  * **Each leg's stop and target are percentages of that leg's price at the
    signal** (its last trade up to the end of the signal candle), applied
    from its fill exactly as batch 1 does for a bought option: the distance
    is `basis x pct`, the level is the fill plus or minus it. A SOLD leg's
    stop is ABOVE its fill (checked on the candle's high) and its target
    below (on the low); a bought leg's the other way round.
  * **Stop first** when one candle could hit both; **a gap past a stop fills
    at the open** (the worse price); a gap past a target at the open too, so
    the rule is not one-sided.
  * **A candle in which an option did not trade cannot fill that leg's own
    stop or target** -- it waits for a real trade.
  * **One leg stopped, the others stay open.** "25% stop on each leg" means
    exactly that; only a combined exit, a condition exit, the square-off or
    expiry closes every leg at once.
  * **Combined exits** (`combined_target_pct`, `combined_stop_pct`) compare the
    position's profit or loss -- legs already closed plus legs still open,
    before charges -- with its net premium at the signal (the credit for a
    selling position, the debit for a buying one, each leg's price at the
    signal times its quantity). OHLC cannot show *when* inside a candle each
    leg made its high or low, so the check is deliberately cautious:

      - the **stop** is judged with every open leg at its worst price of the
        candle at once (a sold leg at its high, a bought leg at its low) --
        legs that may never have been at their worst together are assumed to
        have been;
      - the **target** is judged only on the candle's closing prices -- a
        profit that existed only for a moment inside the candle is not
        counted;
      - when both are met on one candle, the stop wins.

    The decision is made on the closed candle and **filled at the next
    candle's open on every leg**, like an exit rule. The level itself cannot
    be the fill: there is no way to know what each leg was trading at the
    moment the combined figure crossed it, and inventing one per-leg price
    set is exactly the "price that never traded" this engine refuses. A leg
    that does not trade in that candle is closed at its last traded price
    and flagged `exit_untraded` (Justin's square-off rule).
  * **Re-entry** after a leg's own stop, up to `reentry_max` times:
    `same_strike_at_cost` sells (or buys) the SAME contract again once a later
    closed candle's close is back at or better than the leg's first fill;
    `fresh_strike_now` picks the strike again from the index close of the
    candle the stop happened in and fills at the next candle's open. Either
    way the re-entered leg's stop is measured afresh from its new signal
    price, no re-entry is made at or after `no_entry_after` or into the
    square-off, the money it needs must fit (else it is skipped and says so),
    and a fill candle in which the option did not trade is skipped (the
    same-strike kind keeps waiting; the fresh kind was "now" or not at all).
  * **Square-off at the 15:15 candle's open**, last traded price + flag when
    an option did not trade then -- batch 1's rule. **A position held
    overnight is closed on its nearest leg's expiry day at the square-off
    time, every leg together** -- never settled. Closing all legs then, not
    just the expiring one, is the simple and honest choice: a calendar
    spread whose near leg has gone is a different, unhedged position the user
    never asked to hold.
  * **Sizing.** A position with a sold leg is held on MARGIN: `margin_fn` is
    asked for one set (each leg's lots x today's lot size) on the fill day,
    and as many whole sets are opened as fit within `risk.max_position_pct`
    of the account (20% for selling unless the spec says otherwise) AND
    within the spare money (capital plus realized profit, minus what open
    positions tie up). A bought leg's premium is paid in cash on top of the
    margin: `MarginBreakdown.total` is taken to EXCLUDE premium paid for
    bought options (the fixture `test_bought_premium_is_added_on_top_of_margin`
    pins that reading). A buying-only position pays its premium, capped at
    10% as in batch 1. `sizing.lots` is the number of sets wanted.
  * **Margin is checked at entry only.** The exchange can ask for more as
    prices move; this engine does not model that, and the report says so.
  * A sold option can lose more than the account. There is no equity floor
    to assert for such a position; instead the result carries a "LOSS
    LARGER THAN THE ACCOUNT" warning when it happened.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nlt.costs.slippage import ZERO, Slippage
from nlt.data.session import (
    NSE_EQUITY,
    is_last_bar_of_session,
    is_partial_last_bar,
)
from nlt.engine.backtest import (
    BARS_PER_YEAR,
    FNO_MINIMUM_CAPITAL,
    NO_CHARGES_NOTE,
    BacktestResult,
    Decision,
    LegFill,
    Trade,
    _assert_equity_floor,
    _assert_no_overnight_carry,
    _bars_per_session_safe,
    _default_lot_size,
    _describe,
    _in_blocked_window,
    _warn_on_cross_session_lookback,
    _zero_charges,
)
from nlt.engine.conditions import EXPIRY_DAY_COLUMN, evaluate, iv_percentile_column
from nlt.engine.features import build_features
from nlt.engine.metrics import compute_metrics
from nlt.engine.options_backtest import _day_end, _past_last_close, _price_at_clock, last_day_fn
from nlt.options.contracts import Contract, format_strike
from nlt.options.execution import OptionExecution
from nlt.options.expiry import pick_expiry
from nlt.spec.models import (
    All,
    Any_,
    IvPercentile,
    Not,
    OnExpiryDay,
    OptionLeg,
    StrategySpec,
)
from nlt.translate.readback import format_inr

_DEFAULT_INTRADAY_BARS_PER_YEAR = 252 * 75

# The concentration rails when the spec does not set one: selling is held on
# margin and capped at 20% of the account (Justin, 2026-10-14); buying pays
# its premium and keeps batch 1's 10%.
SELL_POSITION_PCT = 20.0
BUY_POSITION_PCT = 10.0

_TYPE_WORDS = {"CE": "call", "PE": "put"}


@dataclass(frozen=True)
class MarginQuery:
    """One leg of a margin question, shaped like `nlt.options.margin.MarginLeg`.

    The engine does not import the margin module (a caller adapts this to it),
    so a test can answer margin questions with a fixed number.
    `quantity` is in units: positive bought, negative sold.
    """

    option_type: str
    strike: float
    expiry: dt.date
    quantity: int


# (underlying, day, legs, index_prev_close) -> an object with `.total` (rupees),
# `.source` (where the figure came from) and `.note` (str | None), as
# `nlt.options.margin.MarginBreakdown` has.
MarginFn = Callable[[str, dt.date, list[MarginQuery], float], object]


class MarginNotAvailable(Exception):
    """Raised by a `margin_fn` that has no honest figure for that day (no
    exchange file covers it). The signal is skipped with the message as its
    reason -- never sized on a guessed margin."""


@dataclass
class _Leg:
    index: int
    spec: OptionLeg
    side: str  # "buy" or "sell" -- the opening side
    quantity: int
    contract: Contract
    why: str
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    raw_entry_price: float
    basis: float
    stop_level: float | None
    target_level: float | None
    last_price: float
    first_fill: float  # the first fill ever, which "back at cost" re-entry means
    is_open: bool = True
    reentries: int = 0
    waiting: bool = False  # stopped, and a re-entry is still allowed
    stopped_at: int = -1  # candle index of the last stop
    # A queued re-entry: (signal time, contract, why, basis) to fill next candle.
    pending: tuple | None = None

    def sign(self) -> int:
        """+1 when the leg profits as its price rises (bought), -1 when sold."""
        return 1 if self.side == "buy" else -1

    def unrealized(self, price: float) -> float:
        return self.sign() * (price - self.entry_price) * self.quantity


@dataclass
class _Position:
    legs: list[_Leg]
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    sets: int
    tied_up: float  # margin plus bought premium, held against spare money
    margin_total: float
    credit_basis: float  # rupees: sold legs' signal prices x qty minus bought legs'
    entry_net: float  # net premium per unit-set at the fills (credit +, debit -)
    expiry: dt.date  # the nearest leg's expiry
    fills: list[LegFill] = field(default_factory=list)
    realized_gross: float = 0.0
    realized_net: float = 0.0
    pending_exit: str | None = None  # "combined_target", "combined_stop", "condition"
    pending_both: bool = False
    bars_held: int = 0
    closed: bool = False


@dataclass
class _Counts:
    leg_not_traded: int = 0
    no_price: int = 0
    no_contract: dict[str, int] = field(default_factory=dict)
    margin: int = 0
    margin_unavailable: dict[str, int] = field(default_factory=dict)
    concurrency: int = 0
    boundary: int = 0
    expiry_entry: int = 0
    expiry_exit: int = 0
    untraded_exit: int = 0
    stop_beat_target: int = 0
    combined_both: int = 0
    reentries: int = 0
    reentry_skipped: int = 0
    capped: int = 0
    worst_equity: float = math.inf
    worst_equity_at: pd.Timestamp | None = None
    margin_sources: dict[str, int] = field(default_factory=dict)
    margin_notes: dict[str, int] = field(default_factory=dict)


def run_legs_backtest(
    spec: StrategySpec,
    bars: pd.DataFrame,
    execution: OptionExecution,
    margin_fn: MarginFn,
    *,
    capital: float = 2_500_000.0,
    lot_size: int | None = None,
    charge_fn=None,
    slippage: Slippage = ZERO,
    drop_partial_last_bar: bool = True,
    trade_from: pd.Timestamp | None = None,
    blocked_windows: list[tuple[dt.datetime, dt.datetime]] = (),
    iv_fn: Callable[[str, pd.DataFrame, int], pd.Series] | None = None,
) -> BacktestResult:
    """Run a multi-leg (or sold) option `spec`: signals on the index `bars`,
    trades via `execution`, margin from `margin_fn`.

    `iv_fn(underlying, index_bars, lookback_days)` supplies the IV percentile
    for an `IvPercentile` condition; it defaults to `nlt.options.iv.iv_feature`.
    The other arguments mean what they mean for `run_option_backtest`.
    """
    legs_spec = spec.instrument.legs
    if not legs_spec:
        raise ValueError("run_legs_backtest runs specs with option legs only")
    underlying = spec.instrument.symbol

    warnings: list[str] = []
    if lot_size is None:
        lot_size = _default_lot_size(spec)
    if charge_fn is None:
        charge_fn = _zero_charges
        warnings.append(NO_CHARGES_NOTE)
    warnings.append(
        f"Sized at today's lot of {lot_size} units for {underlying}, used for every year "
        "tested even where the exchange's lot was different then."
    )
    if capital < FNO_MINIMUM_CAPITAL:
        warnings.append(
            f"OPTIONS ON A SMALL ACCOUNT: this was tested with Rs {capital:,.0f}. Options are "
            f"not really tradeable below about Rs {FNO_MINIMUM_CAPITAL:,.0f} -- one lot is "
            f"{lot_size} units and indivisible. Treat these results as theoretical."
        )
    has_sell = any(leg.side == "sell" for leg in legs_spec)

    timeframe = spec.instrument.timeframe
    is_intraday_tf = timeframe != "1d"
    is_intraday_spec = is_intraday_tf and spec.schedule.intraday
    session = NSE_EQUITY
    square_off = spec.schedule.square_off

    if drop_partial_last_bar and is_intraday_tf and len(bars) > 0:
        try:
            partial = is_partial_last_bar(bars, session, timeframe)
        except ValueError:
            partial = False
        if partial:
            bars = bars.iloc[:-1]

    # A listed expiry that was a market holiday really ended the session before
    # (NIFTY/BANKNIFTY 30 Mar and 29 Jun 2023); every expiry rule uses that day.
    last_day = last_day_fn(bars)
    features = build_features(spec, bars)
    conditions = [spec.entry] + ([spec.exit.condition] if spec.exit.condition else [])
    used = [c for cond in conditions for c in _walk(cond)]
    if any(isinstance(c, OnExpiryDay) for c in used):
        features[EXPIRY_DAY_COLUMN] = _expiry_day_column(
            bars, execution, legs_spec, underlying, last_day
        )
    for lookback in sorted({c.lookback_days for c in used if isinstance(c, IvPercentile)}):
        if iv_fn is None:
            from nlt.options.iv import iv_feature as iv_fn
        column = iv_fn(underlying, bars, lookback)
        features[iv_percentile_column(lookback)] = column.reindex(bars.index)
        known = column.dropna()
        if known.empty:
            warnings.append(
                f"IV percentile ({lookback} days) could not be worked out for any day tested, "
                "so the IV condition never let a trade through."
            )
        else:
            warnings.append(
                f"IV percentile ({lookback} days) is first known on "
                f"{known.index[0]:%d %b %Y}: it needs {lookback} earlier sessions of the option's "
                "own implied volatility to rank against, so no trade could pass the IV "
                "condition before then."
            )

    entry_signal = evaluate(spec.entry, features)
    exit_signal = (
        evaluate(spec.exit.condition, features) if spec.exit.condition is not None else None
    )

    session_bar_count = _bars_per_session_safe(session, timeframe)
    if is_intraday_tf and spec.indicators and session_bar_count:
        _warn_on_cross_session_lookback(spec, session_bar_count, warnings)

    if is_intraday_tf and len(bars) > 0:
        last_bar_of_session = is_last_bar_of_session(bars.index, session).to_numpy()
    else:
        last_bar_of_session = np.zeros(len(bars), dtype=bool)

    # The previous session's closing index level, which a SPAN margin needs.
    closes = bars["close"].groupby(bars.index.date).last() if len(bars) else pd.Series()
    prev_close: dict[dt.date, float] = {}
    for k in range(1, len(closes)):
        prev_close[closes.index[k]] = float(closes.iloc[k - 1])

    entry_reason = _describe(spec.entry)
    max_concurrent = spec.risk.max_concurrent_positions
    counts = _Counts()

    n = len(bars)
    equity = np.empty(n, dtype="float64")
    trades: list[Trade] = []
    decisions: list[Decision] = []
    positions: list[_Position] = []
    pending: tuple | None = None  # (signal_ts, [(leg_spec, contract, why)])
    realized = 0.0

    def tied_up_now(exclude: _Position | None = None) -> float:
        return sum(p.tied_up for p in positions if p is not exclude and not p.closed)

    def skip(signal_ts, code, message, symbol=underlying, **extra):
        decisions.append(
            Decision(signal_ts, symbol, "signal_skipped", code, {"message": message, **extra})
        )

    def close_leg(pos: _Position, leg: _Leg, ts, reason, raw, *, both=False, untraded=False):
        nonlocal realized
        exit_side = "buy" if leg.side == "sell" else "sell"
        exit_price = slippage.apply(raw, exit_side)
        gross = leg.sign() * (exit_price - leg.entry_price) * leg.quantity
        charges = charge_fn(leg.entry_price, leg.quantity, leg.side) + charge_fn(
            exit_price, leg.quantity, exit_side
        )
        if is_intraday_spec:
            _assert_no_overnight_carry(leg.entry_time, ts, session)
        if ts.date() > last_day(leg.contract.expiry):
            raise RuntimeError(
                f"an option leg in {leg.contract.tradingsymbol} was held past its expiry "
                f"({last_day(leg.contract.expiry)}) to {ts}"
            )
        pos.fills.append(
            LegFill(
                leg=leg.index,
                symbol=leg.contract.tradingsymbol,
                label=_label(leg.contract),
                side=leg.side,
                quantity=leg.quantity,
                entry_time=leg.entry_time,
                entry_price=leg.entry_price,
                raw_entry_price=leg.raw_entry_price,
                basis=leg.basis,
                exit_time=ts,
                exit_price=exit_price,
                raw_exit_price=raw,
                exit_reason=reason,
                gross_pnl=gross,
                charges=charges,
                net_pnl=gross - charges,
                stop_level=leg.stop_level,
                target_level=leg.target_level,
                both_hit=both,
                exit_untraded=untraded,
                why=leg.why,
                reentry=leg.reentries,
            )
        )
        leg.is_open = False
        pos.realized_gross += gross
        pos.realized_net += gross - charges
        realized += gross - charges
        if untraded:
            counts.untraded_exit += 1
        if both:
            counts.stop_beat_target += 1

    def finalize(pos: _Position) -> None:
        pos.closed = True
        fills = sorted(pos.fills, key=lambda f: (f.entry_time, f.leg))
        last = max(fills, key=lambda f: (f.exit_time, f.leg))
        gross = sum(f.gross_pnl for f in fills)
        charges = sum(f.charges for f in fills)
        # Net premium per unit-set to close, from each leg's final exit.
        exit_net = 0.0
        final: dict[int, LegFill] = {}
        for f in fills:
            final[f.leg] = f
        for f in final.values():
            lots = legs_spec[f.leg].lots
            exit_net += (lots if f.side == "sell" else -lots) * f.exit_price
        credit = pos.entry_net >= 0
        trades.append(
            Trade(
                entry_time=pos.entry_time,
                entry_price=abs(pos.entry_net),
                exit_time=last.exit_time,
                exit_price=exit_net if credit else -exit_net,
                direction="short" if credit else "long",
                quantity=pos.sets * lot_size,
                gross_pnl=gross,
                charges=charges,
                net_pnl=gross - charges,
                exit_reason=last.exit_reason,
                bars_held=pos.bars_held,
                entry_reason=entry_reason,
                symbol=" + ".join(dict.fromkeys(f.symbol for f in fills)),
                signal_time=pos.signal_time,
                both_hit=any(f.both_hit for f in fills) or pos.pending_both,
                contract_why=" ".join(dict.fromkeys(f.why for f in fills)),
                exit_untraded=any(f.exit_untraded for f in fills),
                legs=tuple(fills),
            )
        )

    def margin_needed(
        legs: list[tuple[OptionLeg, Contract, float]], sets: int, day: dt.date, spot: float
    ) -> tuple[float, float, object | None]:
        """(money tied up, margin part, the MarginBreakdown) for `sets` sets.

        `legs` are (leg, contract, price per unit) -- the fill price for a
        bought leg, whose premium is paid in cash on top of the margin.
        """
        bought = sum(
            price * leg.lots * lot_size * sets for leg, _c, price in legs if leg.side == "buy"
        )
        if not any(leg.side == "sell" for leg, _c, _p in legs):
            return bought, 0.0, None
        query = [
            MarginQuery(
                c.option_type,
                float(c.strike),
                c.expiry,
                (1 if leg.side == "buy" else -1) * leg.lots * lot_size * sets,
            )
            for leg, c, _p in legs
        ]
        breakdown = margin_fn(underlying, day, query, prev_close.get(day, spot))
        return float(breakdown.total) + bought, float(breakdown.total), breakdown

    def record_margin(breakdown) -> None:
        if breakdown is None:
            return
        source = getattr(breakdown, "source", None)
        if source:
            counts.margin_sources[source] = counts.margin_sources.get(source, 0) + 1
        note = getattr(breakdown, "note", None)
        if note:
            counts.margin_notes[note] = counts.margin_notes.get(note, 0) + 1

    def try_open(i: int, ts: pd.Timestamp, signal_ts: pd.Timestamp, chosen) -> None:
        if len(positions) >= max_concurrent:
            counts.concurrency += 1
            skip(signal_ts, "concurrency", "The most positions allowed at once were already open.")
            return
        fills = []
        for leg, contract, why in chosen:
            bar = execution.bar(contract, ts)
            if bar is None or not bool(bar["traded"]):
                counts.leg_not_traded += 1
                skip(
                    signal_ts,
                    "leg_not_traded",
                    f"The {format_strike(contract.strike)} {_TYPE_WORDS[contract.option_type]} "
                    f"did not trade at {ts:%H:%M} on {ts:%d %b %Y}, so the whole position was "
                    "skipped -- no leg is ever opened without the others, and another strike "
                    "is never used.",
                    contract.tradingsymbol,
                    why=why,
                )
                return
            basis = execution.price_at_signal(contract, signal_ts)
            if basis is None:
                counts.no_price += 1
                skip(
                    signal_ts,
                    "no_option_price",
                    f"The {format_strike(contract.strike)} {_TYPE_WORDS[contract.option_type]} "
                    "had not traded that day by the signal, so there was no price to measure "
                    "its stop from; the whole position was skipped.",
                    contract.tradingsymbol,
                    why=why,
                )
                return
            raw = float(bar["open"])
            fills.append((leg, contract, why, raw, slippage.apply(raw, leg.side), basis, bar))

        day = ts.date()
        spot = float(bars["close"].iloc[i - 1])
        priced = [(leg, c, price) for leg, c, _w, _r, price, _b, _bar in fills]
        pct = spec.risk.max_position_pct
        if pct is None:
            pct = SELL_POSITION_PCT if has_sell else BUY_POSITION_PCT
        account = capital + realized
        budget = pct / 100.0 * account
        spare = account - tied_up_now()
        limit = min(budget, spare)
        wanted = spec.sizing.lots if spec.sizing.mode == "fixed_lots" else 1
        if spec.risk.max_lots is not None:
            wanted = min(wanted, spec.risk.max_lots // max(leg.lots for leg in legs_spec))
        if wanted <= 0:
            counts.margin += 1
            skip(
                signal_ts,
                "position_cap",
                f"The most lots allowed per position ({spec.risk.max_lots}) is fewer than one "
                "set of this position needs, so it was skipped.",
            )
            return
        try:
            one_set, _m, breakdown = margin_needed(priced, 1, day, spot)
            sets = min(wanted, math.floor(limit / one_set)) if one_set > 0 else wanted
            need, margin_total = one_set, _m
            while sets > 0:
                need, margin_total, breakdown = margin_needed(priced, sets, day, spot)
                if need <= limit + 1e-9:
                    break
                sets -= 1
        except MarginNotAvailable as exc:
            message = str(exc)
            counts.margin_unavailable[message] = counts.margin_unavailable.get(message, 0) + 1
            skip(
                signal_ts,
                "margin_unavailable",
                f"No exchange margin figure exists for this day, so the position was skipped "
                f"rather than sized on a guess: {message}",
            )
            return
        if sets <= 0:
            counts.margin += 1
            what = "margin" if has_sell else "premium"
            if budget <= spare:
                why_not = f"{pct:g}% of the account is {format_inr(budget)}"
            else:
                why_not = f"the spare money was {format_inr(max(spare, 0.0))}"
            skip(
                signal_ts,
                "margin" if has_sell else "position_cap",
                f"One set needs {format_inr(one_set)} {what}; {why_not}, so the position was "
                "skipped.",
            )
            return
        if sets < wanted:
            counts.capped += 1
        record_margin(breakdown)

        legs: list[_Leg] = []
        for leg, contract, why, raw, price, basis, bar in fills:
            stop = target = None
            sign = 1 if leg.side == "buy" else -1
            if leg.stop_pct is not None:
                stop = price - sign * basis * leg.stop_pct / 100.0
            if leg.target_pct is not None:
                target = price + sign * basis * leg.target_pct / 100.0
            legs.append(
                _Leg(
                    index=legs_spec.index(leg),
                    spec=leg,
                    side=leg.side,
                    quantity=leg.lots * lot_size * sets,
                    contract=contract,
                    why=why,
                    signal_time=signal_ts,
                    entry_time=ts,
                    entry_price=price,
                    raw_entry_price=raw,
                    basis=basis,
                    stop_level=stop,
                    target_level=target,
                    last_price=float(bar["close"]),
                    first_fill=price,
                )
            )
        credit_basis = sum(-leg.sign() * leg.basis * leg.quantity for leg in legs)
        entry_net = sum(-leg.sign() * leg.spec.lots * leg.entry_price for leg in legs)
        positions.append(
            _Position(
                legs=legs,
                signal_time=signal_ts,
                entry_time=ts,
                sets=sets,
                tied_up=need,
                margin_total=margin_total,
                credit_basis=credit_basis,
                entry_net=entry_net,
                expiry=min(last_day(leg.contract.expiry) for leg in legs),
            )
        )

    def fill_reentry(pos: _Position, leg: _Leg, i: int, ts: pd.Timestamp) -> None:
        signal_ts, contract, why, basis = leg.pending
        leg.pending = None
        kind = leg.spec.reentry
        bar = execution.bar(contract, ts)
        if bar is None or not bool(bar["traded"]):
            counts.reentry_skipped += 1
            skip(
                signal_ts,
                "reentry_not_traded",
                f"Re-entering the {_label(contract)}: it did not trade at {ts:%H:%M} on "
                f"{ts:%d %b %Y}, so the re-entry was skipped"
                + (" and it waits for the next chance." if kind == "same_strike_at_cost" else "."),
                contract.tradingsymbol,
            )
            if kind != "same_strike_at_cost":
                leg.waiting = False
            return
        raw = float(bar["open"])
        price = slippage.apply(raw, leg.side)
        open_after = [
            (x.spec, x.contract, x.entry_price) for x in pos.legs if x.is_open and x is not leg
        ] + [(leg.spec, contract, price)]
        try:
            need, _m, breakdown = margin_needed(
                open_after, pos.sets, ts.date(), float(bars["close"].iloc[i - 1])
            )
        except MarginNotAvailable as exc:
            counts.reentry_skipped += 1
            leg.waiting = False
            skip(
                signal_ts,
                "reentry_margin",
                f"Re-entering the {_label(contract)}: no exchange margin figure exists for "
                f"this day ({exc}), so the re-entry was skipped.",
                contract.tradingsymbol,
            )
            return
        pct = spec.risk.max_position_pct
        if pct is None:
            pct = SELL_POSITION_PCT if has_sell else BUY_POSITION_PCT
        account = capital + realized
        extra = max(need - pos.tied_up, 0.0)
        if need > pct / 100.0 * account + 1e-9 or extra > account - tied_up_now() + 1e-9:
            counts.reentry_skipped += 1
            leg.waiting = False
            skip(
                signal_ts,
                "reentry_margin",
                f"Re-entering the {_label(contract)} would have needed "
                f"{format_inr(need)} for the position, more than the account allowed, so "
                "the re-entry was skipped.",
                contract.tradingsymbol,
            )
            return
        record_margin(breakdown)
        pos.tied_up = max(pos.tied_up, need)
        sign = leg.sign()
        leg.contract = contract
        leg.why = why
        leg.signal_time = signal_ts
        leg.entry_time = ts
        leg.entry_price = price
        leg.raw_entry_price = raw
        leg.basis = basis
        leg.stop_level = (
            price - sign * basis * leg.spec.stop_pct / 100.0 if leg.spec.stop_pct else None
        )
        leg.target_level = (
            price + sign * basis * leg.spec.target_pct / 100.0 if leg.spec.target_pct else None
        )
        leg.last_price = float(bar["close"])
        leg.is_open = True
        leg.reentries += 1
        leg.waiting = False  # open again; the limit is checked at its next stop
        counts.reentries += 1
        decisions.append(
            Decision(
                signal_ts,
                contract.tradingsymbol,
                "signal_taken",
                "reentry",
                {"why": why, "message": f"Re-entered the {_label(contract)} at {ts:%H:%M}."},
            )
        )

    def leg_exit(leg: _Leg, bar) -> tuple[str, float, bool] | None:
        """A leg's own stop or target on a traded candle: (reason, raw fill, both)."""
        o, h, low = float(bar["open"]), float(bar["high"]), float(bar["low"])
        if leg.side == "sell":
            target_hit = leg.target_level is not None and low <= leg.target_level
            if leg.stop_level is not None and h >= leg.stop_level:
                return "stop", (o if o > leg.stop_level else leg.stop_level), target_hit
            if target_hit:
                return "target", (o if o < leg.target_level else leg.target_level), False
            return None
        target_hit = leg.target_level is not None and h >= leg.target_level
        if leg.stop_level is not None and low <= leg.stop_level:
            return "stop", (o if o < leg.stop_level else leg.stop_level), target_hit
        if target_hit:
            return "target", (o if o > leg.target_level else leg.target_level), False
        return None

    def forced_price(leg: _Leg, ts, bar, clock_reached) -> tuple[float, bool]:
        """(raw price, untraded) for closing a leg at a square-off or expiry close."""
        if not is_intraday_tf:
            price = _price_at_clock(execution, leg.contract, ts.date(), square_off)
            if price is not None:
                return price, False
            last = execution.last_trade(leg.contract, _day_end(ts.date()))
            return (last[0] if last else leg.last_price), True
        if bar is not None and bool(bar["traded"]):
            return float(bar["open"] if clock_reached else bar["close"]), False
        last = execution.last_trade(leg.contract, ts.to_pydatetime())
        return (last[0] if last else leg.last_price), True

    def open_price(leg: _Leg, ts, bar) -> tuple[float, bool]:
        """(raw price, untraded) for closing a leg at this candle's open."""
        if bar is not None and bool(bar["traded"]):
            return float(bar["open"]), False
        last = execution.last_trade(leg.contract, ts.to_pydatetime())
        return (last[0] if last else leg.last_price), True

    def reentry_allowed_at(i: int, ts: pd.Timestamp) -> bool:
        if i + 1 >= n:
            return False
        if is_intraday_spec and ts.time() >= spec.schedule.no_entry_after:
            return False
        if is_intraday_spec and bool(last_bar_of_session[i]):
            return False
        return not (is_intraday_tf and ts.time() >= square_off)

    for i in range(n):
        ts = bars.index[i]

        # --- 1. fill the entry queued by the previous candle's signal --------
        if pending is not None:
            signal_ts, chosen = pending
            pending = None
            try_open(i, ts, signal_ts, chosen)

        # --- 2. each open position, on this candle -----------------------------
        for pos in positions:
            leg_bars = {}
            for leg in pos.legs:
                if leg.is_open:
                    bar = execution.bar(leg.contract, ts)
                    leg_bars[id(leg)] = bar
                    if bar is not None and bool(bar["traded"]):
                        leg.last_price = float(bar["close"])

            on_expiry_day = ts.date() == pos.expiry
            clock_reached = is_intraday_tf and ts.time() >= square_off
            forced = None
            if is_intraday_spec and (clock_reached or bool(last_bar_of_session[i])):
                forced = "square_off"
            elif on_expiry_day and (
                not is_intraday_tf or clock_reached or bool(last_bar_of_session[i])
            ):
                forced = "expiry"

            if pos.pending_exit is not None:
                # Decided on the previous closed candle; every open leg fills at
                # this candle's open (its last trade, flagged, if it did not trade).
                # That open is never later than a square-off on the same candle.
                reason = pos.pending_exit
                for leg in pos.legs:
                    if leg.is_open:
                        raw, untraded = open_price(leg, ts, leg_bars.get(id(leg)))
                        close_leg(pos, leg, ts, reason, raw, untraded=untraded)
                    leg.waiting = False
                    leg.pending = None
                pos.bars_held += 1
                finalize(pos)
                continue

            if forced is not None:
                if forced == "expiry":
                    counts.expiry_exit += 1
                for leg in pos.legs:
                    if leg.is_open:
                        raw, untraded = forced_price(leg, ts, leg_bars.get(id(leg)), clock_reached)
                        close_leg(pos, leg, ts, forced, raw, untraded=untraded)
                    leg.waiting = False
                    leg.pending = None
                pos.bars_held += 1
                finalize(pos)
                continue

            # Re-entries queued by the previous candle fill at this open.
            for leg in pos.legs:
                if leg.pending is not None:
                    fill_reentry(pos, leg, i, ts)
                    if leg.is_open:
                        bar = execution.bar(leg.contract, ts)
                        leg_bars[id(leg)] = bar

            for leg in pos.legs:
                if not leg.is_open:
                    continue
                bar = leg_bars.get(id(leg))
                if bar is None or not bool(bar["traded"]):
                    continue  # no trade, so this leg's own exit cannot fill here
                hit = leg_exit(leg, bar)
                if hit is None:
                    continue
                reason, raw, both = hit
                close_leg(pos, leg, ts, reason, raw, both=both)
                if reason == "stop" and leg.reentries < leg.spec.reentry_max:
                    leg.waiting = True
                    leg.stopped_at = i
                else:
                    leg.waiting = False

            pos.bars_held += 1
            if (
                spec.exit.max_bars_held is not None
                and pos.bars_held >= spec.exit.max_bars_held
                and any(leg.is_open for leg in pos.legs)
            ):
                for leg in pos.legs:
                    if leg.is_open:
                        bar = leg_bars.get(id(leg))
                        if bar is not None and bool(bar["traded"]):
                            close_leg(pos, leg, ts, "max_bars", float(bar["close"]))
                        else:
                            close_leg(pos, leg, ts, "max_bars", leg.last_price, untraded=True)
                    leg.waiting = False
                    leg.pending = None

            if not any(leg.is_open or leg.waiting for leg in pos.legs):
                finalize(pos)
        positions = [p for p in positions if not p.closed]

        # --- 3. mark to market at each leg's last traded price ----------------
        unrealized = sum(
            leg.unrealized(leg.last_price) for p in positions for leg in p.legs if leg.is_open
        )
        equity[i] = capital + realized + unrealized
        if equity[i] < counts.worst_equity:
            counts.worst_equity = float(equity[i])
            counts.worst_equity_at = ts
        if not has_sell:
            _assert_equity_floor(equity[i], ts, "long")

        # --- 4. this closed candle: combined exits, exit rule, re-entries -----
        exit_true = exit_signal is not None and bool(exit_signal.iloc[i])
        for pos in positions:
            if pos.pending_exit is not None:
                continue
            decided = _combined_decision(pos, spec, execution, ts)
            if decided is not None:
                pos.pending_exit, both = decided
                pos.pending_both = pos.pending_both or both
                counts.combined_both += int(both)
            elif exit_true and any(leg.is_open for leg in pos.legs):
                pos.pending_exit = "condition"
            if pos.pending_exit is not None:
                continue
            allowed = reentry_allowed_at(i, ts)
            for leg in pos.legs:
                if not leg.waiting or leg.is_open or leg.pending is not None:
                    continue
                if not allowed:
                    leg.waiting = False
                    continue
                queued = _queue_reentry(
                    leg, i, ts, bars, execution, timeframe, square_off, skip, counts, last_day
                )
                if queued is not None:
                    leg.pending = queued
            if not any(leg.is_open or leg.waiting for leg in pos.legs):
                finalize(pos)
        positions = [p for p in positions if not p.closed]

        # --- 5. this closed candle's entry signal queues a fill ---------------
        if bool(entry_signal.iloc[i]) and (trade_from is None or ts >= trade_from):
            reason = None
            if len(positions) >= max_concurrent:
                reason = "concurrency_at_signal"
            elif ts.weekday() not in spec.schedule.weekdays:
                reason = "weekday"
            elif _in_blocked_window(ts, blocked_windows):
                reason = "paused"
            elif is_intraday_spec and ts.time() >= spec.schedule.no_entry_after:
                reason = "no_entry_after"
            elif is_intraday_spec and bool(last_bar_of_session[i]):
                counts.boundary += 1
                reason = "boundary"
            elif i + 1 >= n:
                decisions.append(Decision(ts, underlying, "signal_taken", "queued", {}))
            if reason is not None:
                decisions.append(Decision(ts, underlying, "signal_skipped", reason, {}))
            elif i + 1 < n:
                pending = _choose_legs(
                    ts,
                    float(bars["close"].iloc[i]),
                    bars.index[i + 1],
                    legs_spec,
                    execution,
                    timeframe,
                    square_off,
                    counts,
                    decisions,
                    underlying,
                    last_day,
                )

    # --- 6. anything still open at the end of the data -----------------------
    if positions:
        ts = bars.index[-1]
        for pos in positions:
            for leg in pos.legs:
                if leg.is_open:
                    close_leg(pos, leg, ts, "end_of_data", leg.last_price)
            if pos.fills:
                finalize(pos)
        equity[-1] = capital + realized

    warnings.extend(_count_warnings(counts, square_off, spec, capital))

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


# ------------------------------------------------------------------ helpers


def _label(contract: Contract) -> str:
    return f"{contract.underlying} {format_strike(contract.strike)} {contract.option_type}"


def _walk(condition):
    yield condition
    if isinstance(condition, (All, Any_)):
        for c in condition.conditions:
            yield from _walk(c)
    elif isinstance(condition, Not):
        yield from _walk(condition.condition)


def _expiry_day_column(
    bars: pd.DataFrame,
    execution: OptionExecution,
    legs: list[OptionLeg],
    underlying: str,
    last_day,
) -> pd.Series:
    """True on every candle of a day on which one of the legs' expiries falls --
    the day it really expired, the session before a listed holiday.

    Which expiry a leg means on a day ("nearest weekly", "monthly") is read
    from the contracts listed THAT day, as when choosing a contract -- never a
    weekday rule.
    """
    kinds = sorted({leg.expiry for leg in legs})
    by_day: dict[dt.date, bool] = {}
    for day in sorted(set(bars.index.date)):
        listing = execution.listing(day)
        expiries = list(listing["expiry"]) if not listing.empty else []
        picked = [pick_expiry(expiries, day, kind, underlying)[0] for kind in kinds]
        by_day[day] = any(e is not None and last_day(e) == day for e in picked)
    return pd.Series([by_day[d] for d in bars.index.date], index=bars.index)


def _choose_legs(
    signal_ts,
    spot_close,
    fill_ts,
    legs_spec,
    execution,
    timeframe,
    square_off,
    counts,
    decisions,
    underlying,
    last_day,
):
    """Every leg's contract at the signal, or None (recorded) if any cannot be chosen."""
    chosen = []
    for leg in legs_spec:
        resolution = execution.resolve_leg(signal_ts.to_pydatetime(), spot_close, leg)
        contract = resolution.contract
        if contract is None:
            code = resolution.reason_code or "no_contract"
            counts.no_contract[code] = counts.no_contract.get(code, 0) + 1
            decisions.append(
                Decision(
                    signal_ts,
                    underlying,
                    "signal_skipped",
                    code,
                    {"why": resolution.why, "message": resolution.why},
                )
            )
            return None
        if _past_last_close(contract, fill_ts, timeframe, square_off, last_day):
            counts.expiry_entry += 1
            decisions.append(
                Decision(
                    signal_ts,
                    contract.tradingsymbol,
                    "signal_skipped",
                    "expiry",
                    {
                        "why": resolution.why,
                        "message": f"{_label(contract)} expires at {square_off:%H:%M} that "
                        "day, before this signal could be filled, so it was skipped.",
                    },
                )
            )
            return None
        chosen.append((leg, contract, resolution.why))
    decisions.append(
        Decision(
            signal_ts,
            underlying,
            "signal_taken",
            "queued",
            {
                "why": " ".join(w for _l, _c, w in chosen),
                "contract": " + ".join(c.tradingsymbol for _l, c, _w in chosen),
            },
        )
    )
    return signal_ts, chosen


def _queue_reentry(leg, i, ts, bars, execution, timeframe, square_off, skip, counts, last_day):
    """(signal time, contract, why, basis) if this closed candle re-enters `leg`."""
    if i + 1 >= len(bars):
        return None
    fill_ts = bars.index[i + 1]
    if leg.spec.reentry == "same_strike_at_cost":
        if i <= leg.stopped_at:
            return None  # "a later candle": not the one the stop happened in
        bar = execution.bar(leg.contract, ts)
        if bar is None or not bool(bar["traded"]):
            return None
        close = float(bar["close"])
        back = close <= leg.first_fill if leg.side == "sell" else close >= leg.first_fill
        if not back or _past_last_close(leg.contract, fill_ts, timeframe, square_off, last_day):
            return None
        return ts, leg.contract, leg.why, close

    # fresh_strike_now: from the index close of the candle the stop happened in.
    resolution = execution.resolve_leg(ts.to_pydatetime(), float(bars["close"].iloc[i]), leg.spec)
    leg.waiting = False  # "now" or not at all, unless the fill succeeds
    contract = resolution.contract
    if contract is None or _past_last_close(contract, fill_ts, timeframe, square_off, last_day):
        counts.reentry_skipped += 1
        skip(
            ts,
            "reentry_no_contract",
            resolution.why
            if contract is None
            else f"{_label(contract)} was expiring; the re-entry was skipped.",
        )
        return None
    basis = execution.price_at_signal(contract, ts)
    if basis is None:
        counts.reentry_skipped += 1
        skip(
            ts,
            "reentry_no_price",
            f"{_label(contract)} had not traded that day by {ts:%H:%M}, so there was no price "
            "to measure a stop from; the re-entry was skipped.",
            contract.tradingsymbol,
        )
        return None
    leg.waiting = True  # until the fill decides
    return ts, contract, resolution.why, basis


def _combined_decision(pos, spec, execution, ts) -> tuple[str, bool] | None:
    """("combined_stop" | "combined_target", both met) on this closed candle, or None.

    See the module docstring: the stop with every open leg at its worst price
    of the candle together, the target on closing prices only, stop first.
    """
    target_pct = spec.exit.combined_target_pct
    stop_pct = spec.exit.combined_stop_pct
    base = abs(pos.credit_basis)
    if (target_pct is None and stop_pct is None) or base <= 0:
        return None
    worst = close = pos.realized_gross
    for leg in pos.legs:
        if not leg.is_open:
            continue
        bar = execution.bar(leg.contract, ts)
        if bar is not None and bool(bar["traded"]):
            adverse = float(bar["high"] if leg.side == "sell" else bar["low"])
            last = float(bar["close"])
        else:
            adverse = last = leg.last_price
        worst += leg.unrealized(adverse)
        close += leg.unrealized(last)
    stop_hit = stop_pct is not None and worst <= -stop_pct / 100.0 * base
    target_hit = target_pct is not None and close >= target_pct / 100.0 * base
    if stop_hit:
        return "combined_stop", target_hit
    if target_hit:
        return "combined_target", False
    return None


def _count_warnings(counts: _Counts, square_off: dt.time, spec: StrategySpec, capital: float):
    out: list[str] = []
    if counts.margin_sources:
        sources = sorted(counts.margin_sources)
        shown = sources[0] if len(sources) == 1 else f"{sources[0]} ... {sources[-1]}"
        out.append(
            "MARGIN AT ENTRY: each selling position was held on the margin worked out for the "
            f"day it opened ({shown}). It was checked at entry only; the exchange may ask for "
            "more as prices move, and a broker would then demand it or close the position."
        )
    for note, k in sorted(counts.margin_notes.items()):
        out.append(f"Margin ({k} position(s)): {note}")
    if counts.worst_equity < 0:
        out.append(
            "LOSS LARGER THAN THE ACCOUNT: at "
            f"{counts.worst_equity_at:%d %b %Y %H:%M} this account was worth "
            f"{format_inr(counts.worst_equity)} -- a sold option lost more than the "
            f"{format_inr(capital)} started with. A broker would have closed it long before."
        )
    if counts.leg_not_traded:
        out.append(
            f"{counts.leg_not_traded} signal(s) skipped: at least one of the chosen options did "
            "not trade in the candle we would have traded in. Every leg must trade, or none "
            "is opened; another strike is never used instead."
        )
    if counts.no_price:
        out.append(
            f"{counts.no_price} signal(s) skipped: one of the chosen options had not traded that "
            "day by the signal, so there was no price to measure its stop from."
        )
    for code, k in sorted(counts.no_contract.items()):
        if code == "no_weekly":
            out.append(
                f"{k} signal(s) skipped: {spec.instrument.symbol} had no weekly options on "
                "those days, so no contract could be chosen."
            )
        elif code == "points_out_of_range":
            out.append(
                f"{k} signal(s) skipped: no strike was listed near the number of points from "
                "the index that the strategy asks for."
            )
        else:
            out.append(f"{k} signal(s) skipped: no suitable option contract was listed.")
    for message, k in sorted(counts.margin_unavailable.items()):
        out.append(
            f"{k} signal(s) skipped: no exchange margin figure exists for the day -- {message}"
        )
    if counts.margin:
        out.append(
            f"{counts.margin} signal(s) skipped: one set of the position needed more money "
            "than the account's limit for one position, or than the money left spare."
        )
    if counts.capped:
        out.append(f"{counts.capped} position(s) opened with fewer sets than asked, to fit.")
    if counts.concurrency:
        out.append(
            f"{counts.concurrency} signal(s) skipped: the most positions allowed at once were "
            "already open."
        )
    if counts.boundary:
        out.append(
            f"{counts.boundary} signal(s) skipped: they fired on the day's last candle, and "
            "this strategy squares off the same day."
        )
    if counts.expiry_entry:
        out.append(
            f"{counts.expiry_entry} signal(s) skipped: a chosen option expired before the "
            "trade could be placed."
        )
    if counts.expiry_exit:
        out.append(
            f"{counts.expiry_exit} position(s) closed at {square_off:%H:%M} on their nearest "
            "leg's expiry day, every leg together. An option is never held to settlement here."
        )
    if counts.untraded_exit:
        out.append(
            f"{counts.untraded_exit} leg exit(s) used the option's last traded price: it did "
            "not trade at the moment of closing, so no fresher price exists."
        )
    if counts.stop_beat_target:
        out.append(
            f"{counts.stop_beat_target} leg(s) could have hit both the stop loss and the target "
            "within one candle; the stop was assumed to come first, the cautious choice."
        )
    if counts.combined_both:
        out.append(
            f"{counts.combined_both} position(s) met both the combined stop and the combined "
            "target on one candle; the stop was assumed to come first."
        )
    if counts.reentries:
        out.append(f"{counts.reentries} leg(s) re-entered after their stop loss.")
    if counts.reentry_skipped:
        out.append(
            f"{counts.reentry_skipped} re-entry attempt(s) skipped (the option did not trade, "
            "had no price, or the money did not fit)."
        )
    return out


__all__ = ["MarginNotAvailable", "MarginQuery", "run_legs_backtest"]
