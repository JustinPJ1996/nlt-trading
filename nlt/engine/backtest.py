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

import datetime as dt
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nlt.costs.slippage import ZERO, Slippage
from nlt.data.instruments import lot_size as exchange_lot_size
from nlt.data.instruments import lot_size_note
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
    AtTime,
    Compare,
    Condition,
    Const,
    IsTrue,
    IvPercentile,
    Not,
    OnExpiryDay,
    Operand,
    PercentChange,
    Ref,
    Schedule,
    StrategySpec,
)

_CANDLE_LENGTH = {
    "1m": pd.Timedelta(minutes=1),
    "3m": pd.Timedelta(minutes=3),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
}


def _plural_trades(n: int | None) -> str:
    return "one trade" if n == 1 else f"{n} trades"


def past_entry_cutoff(spec: StrategySpec, ts: pd.Timestamp, timeframe: str) -> bool:
    """Would a signal on the candle starting at `ts` fill after the entry cut-off?

    "No new entries after 15:00" is about when the trade would open: at the
    next candle's open, when this one ends. Judged on the candle's START (as it
    was until 2026-10-15) the rule let a 1-hour candle starting 14:15 open a
    trade at 15:15, after a 15:00 cut-off; on candles that line up with the
    cut-off the two readings agree. Daily candles carry no clock: never cut.
    """
    if not spec.schedule.cutoff_applies or timeframe not in _CANDLE_LENGTH:
        return False
    cutoff = spec.schedule.no_entry_after
    limit = ts.normalize() + pd.Timedelta(hours=cutoff.hour, minutes=cutoff.minute)
    return ts + _CANDLE_LENGTH[timeframe] > limit


BARS_PER_YEAR = {"1d": 252}
_DEFAULT_INTRADAY_BARS_PER_YEAR = 252 * 75  # rough fallback; only 1d is exercised today


# Said when a run was given no charges to deduct; the verdict looks for it.
NO_CHARGES_NOTE = "No charges were deducted from these results."


def _zero_charges(price: float, quantity: int, side: str) -> float:
    return 0.0


def _default_lot_size(spec: StrategySpec) -> int:
    """The tradeable unit when the caller does not specify one.

    A lot is an *options* contract term, not a property of the index. A
    cash-index backtest that inherited one would buy a whole lot of NIFTY per
    "1 lot", which on a modest account is several times the leverage the caller
    asked for -- exactly how the engine once reported a -113% total return on a
    strategy that could not lose more than its stop loss. An index or stock
    backtest trades in units of 1; only options inherit the lot convention.

    The size itself comes from `nlt.data.instruments`, which holds the CURRENT
    exchange lot for every symbol and explains why the current one is used even
    for years when a different one applied.
    """
    return exchange_lot_size(spec.instrument.symbol, spec.instrument.trade_as)


def _margin_rate(spec: StrategySpec) -> float:
    """The share of a position's value it ties up: the exchange margin for futures.

    Justin's call (2026-10-12): futures are held on the margin the exchange sets,
    which is not borrowing, so a NIFTY lot ties up about 11% of its value, not
    all of it. Everything else is paid for in full. Profit and loss are always on
    the full value -- only what counts against the account's limits changes.
    """
    if spec.instrument.is_future:
        from nlt.data.futures import contract

        return contract(spec.instrument.symbol).margin_rate
    return 1.0


def _session_for_spec(spec: StrategySpec) -> Session:
    """The trading-hours calendar `spec.instrument` runs on.

    `Instrument.symbol` today only permits NIFTY and BANKNIFTY, both NSE
    equity & F&O names, so this always resolves to `NSE_EQUITY`. `nlt.data.session`
    also defines `MCX` (a session that runs to 23:30, well past NSE's close), but
    there is no commodity symbol in the spec model yet to map onto it. Everything
    below is written against `Session` generically -- once `Instrument.symbol`
    grows an MCX name, this function is the only place that needs to change.

    It has: a futures contract carries its own session, so an MCX strategy runs
    on MCX's hours (to 23:30, or 23:55 in winter).
    """
    if spec.instrument.is_future:
        from nlt.data.futures import contract

        return contract(spec.instrument.symbol).session
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
    # The closed candle whose signal caused this entry -- one bar before
    # `entry_time`, which is the bar the fill happened on. None only for a
    # Trade built by hand (e.g. in a test) that never went through the engine.
    signal_time: pd.Timestamp | None = None
    # Stop/target as set at entry (a trailing stop ratchets afterwards; this
    # is always the original fixed level, or None if the rule has none).
    stop_level: float | None = None
    target_level: float | None = None
    # The price before slippage was applied, for each side of the trade.
    raw_entry_price: float | None = None
    raw_exit_price: float | None = None
    # True when the exit candle touched both the stop and the target -- OHLC
    # cannot show which came first, so the stop is always assumed to have hit
    # first (see the module docstring); this flag says that assumption was
    # actually exercised on this trade, not merely available.
    both_hit: bool = False
    # Options only (`nlt.engine.options_backtest`): why this contract was
    # chosen, in words, and whether the exit had to use the option's last
    # traded price because it did not trade at the moment of closing.
    contract_why: str = ""
    exit_untraded: bool = False
    # Multi-leg and sold options only (`nlt.engine.legs_backtest`): one Trade is
    # one whole position, so win rate and trade counts count positions; each
    # option bought or sold inside it is a `LegFill` here. Empty for everything
    # else.
    legs: tuple[LegFill, ...] = ()


@dataclass(frozen=True)
class LegFill:
    """One option inside a multi-leg position, from its fill to its exit.

    A leg that was stopped and re-entered appears twice, `reentry` counting
    which time it is (0 = the first fill). `side` is the opening side: a
    "sell" leg is sold first and bought back at the exit. Prices are what was
    paid or received after slippage; `raw_*` before it. `basis` is the
    option's price at the signal, which the leg's stop and target measure from.
    """

    leg: int
    symbol: str
    label: str
    side: str
    quantity: int
    entry_time: pd.Timestamp
    entry_price: float
    raw_entry_price: float
    basis: float
    exit_time: pd.Timestamp
    exit_price: float
    raw_exit_price: float
    exit_reason: str
    gross_pnl: float
    charges: float
    net_pnl: float
    stop_level: float | None = None
    target_level: float | None = None
    both_hit: bool = False
    exit_untraded: bool = False
    why: str = ""
    reentry: int = 0


@dataclass(frozen=True)
class Decision:
    """One thing the engine decided on one bar, whether or not it became a trade.

    `kind` is "signal_taken" (the entry rule was true and the engine queued a
    fill for the next bar's open) or "signal_skipped" (the entry rule was true
    but the engine did not act on it -- `reason_code` says why). Nothing is
    recorded for a bar where the entry rule was simply false: a "decision" is
    the engine choosing between taking and not taking a signal, not a log of
    every bar.
    """

    time: pd.Timestamp
    symbol: str
    kind: str
    reason_code: str
    detail: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    spec: StrategySpec
    trades: list[Trade]
    equity: pd.Series
    metrics: dict
    features: pd.DataFrame
    warnings: list[str]
    decisions: list[Decision] = field(default_factory=list)


@dataclass
class _OpenPosition:
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    quantity: int
    stop_level: float | None  # fixed stop (from stop_pct / stop_atr_mult), set once at entry
    target_level: float | None
    trailing_pct: float | None
    trail_best: float  # best price seen since entry; only moves favourably
    bars_held: int = 0
    pending_condition_exit: bool = False
    # The closed bar whose signal led to this entry; None only when a position
    # is opened without having gone through the scheduling step below (never
    # happens today, but keeps the field honest about when it can be absent).
    signal_time: pd.Timestamp | None = None
    raw_entry_price: float | None = None  # bar.open, before slippage


# Below this, index options stop being a sensible instrument rather than merely
# an aggressive one. An index option lot is indivisible, so a small account cannot take
# a proportionate position -- it either cannot afford one lot at all, or one lot
# is most of the account. Later this becomes a setting; for now it is a warning
# nobody can miss.
FNO_MINIMUM_CAPITAL = 1_000_000.0


def run_backtest(
    spec: StrategySpec,
    bars: pd.DataFrame,
    *,
    capital: float = 100_000.0,
    lot_size: int | None = None,
    charge_fn=None,
    slippage: Slippage = ZERO,
    drop_partial_last_bar: bool = True,
    trade_from: pd.Timestamp | None = None,
    final_session_in_progress: bool = False,
    listed_expiries: Iterable[dt.date] = (),
    blocked_windows: list[tuple[dt.datetime, dt.datetime]] = (),
    minute_bars: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run `spec` bar by bar over `bars`.

    `minute_bars` are the same instrument's 1-minute candles. With them, an
    intraday square-off sells at the close of the square-off minute -- the
    price at 15:15:59 for 15:15 (Justin, 2026-10-15) -- whatever the candle
    size, and the stop and target still apply to the minutes before it. A
    day without them squares off as before: the close of the first candle
    starting at or after the time, and the result says how many days that was.

    `blocked_windows` pauses new entries during the given `[start, end)` wall-
    clock windows: a signal whose bar falls inside one is skipped (reason
    "paused"), exactly like any other skip. It does not touch exits -- a stop,
    a target, square-off and expiry all still fire normally for a position
    already open, because pausing the strategy is a decision about new risk,
    not about abandoning risk already taken on. Default empty means identical
    behaviour to not having the feature at all.

    `lot_size` is optional and should almost always be left unset: it is
    derived from `spec.instrument.trade_as` (see `_default_lot_size`) so that
    an index backtest trades in units of 1 and only an options backtest
    inherits the index-option lot convention. Pass it explicitly only to
    model a specific contract's lot size (e.g. BANKNIFTY's, which differs
    from NIFTY's).

    `drop_partial_last_bar`, when the timeframe is intraday, drops the final
    bar of `bars` if it is still forming (its close time is in the future
    relative to now) rather than let the engine signal on it. The data layer
    (`nlt.data.session.is_partial_last_bar`) is expected to have already
    filtered this out upstream; this is defence in depth specifically because
    the engine is what places orders, so it should not trust that every caller
    remembered the upstream guard.

    `trade_from`, when set, is the first bar allowed to *signal* an entry.
    Earlier bars still feed every indicator -- a 200-day average needs its 200
    days -- but cannot open a position. This is how paper trading starts flat
    at the moment the user pressed start, instead of inheriting a position the
    history would have been holding.

    `final_session_in_progress` says the last session in `bars` has not ended
    yet -- true for a paper run mid-day, never for a backtest. Without it, the
    engine infers "last candle of the day" from the data, sees the newest
    candle so far, and squares off at 13:35 a position the real day would have
    held until its stop, its target or the 15:15 square-off.

    For futures, every open position is closed at the close of its contract's
    expiry day, and charged for that exit -- Justin's choice (2026-10-07) over
    rolling into the next month. `listed_expiries` are the dates the exchange
    has published for live contracts (Kite's instrument list); they override
    the computed rule for their month. See `nlt.data.futures`.
    """
    warnings: list[str] = []
    if lot_size is None:
        lot_size = _default_lot_size(spec)
    if charge_fn is None:
        charge_fn = _zero_charges
        warnings.append(NO_CHARGES_NOTE)

    note = lot_size_note(spec.instrument.symbol, spec.instrument.trade_as)
    if note:
        warnings.append(note)

    if spec.instrument.trade_as == "option" and capital < FNO_MINIMUM_CAPITAL:
        # Not a preference -- arithmetic. An index option lot is tens of units
        # (NIFTY is 65 as of the NSE circular of 28 Nov 2025, and the exchange
        # revises it), and it
        # cannot be divided, so a small account either cannot buy one at all or
        # has to put an irresponsible share of itself into a single contract that
        # can lose its entire premium in a session. Below this figure the honest
        # answer is that options are not a suitable instrument yet, whatever the
        # backtest says.
        warnings.append(
            f"OPTIONS ON A SMALL ACCOUNT: this was tested with "
            f"Rs {capital:,.0f}. Options are not really tradeable below about "
            f"Rs {FNO_MINIMUM_CAPITAL:,.0f} – one NIFTY lot is "
            f"{exchange_lot_size('NIFTY', 'option')} units and "
            "indivisible, so a smaller account must either skip most trades or "
            "bet far too much of itself on a single contract. Treat these "
            "results as theoretical."
        )

    if spec.instrument.is_future and len(bars) > 0:
        _warn_about_margin(spec, bars, capital, lot_size, warnings)
        _warn_if_one_lot_is_unaffordable(spec, bars, capital, lot_size, warnings)

    is_intraday_tf = spec.instrument.timeframe != "1d"
    session = _session_for_spec(spec)
    minutes_by_day: dict[dt.date, pd.DataFrame] = {}
    if minute_bars is not None and len(minute_bars) and is_intraday_tf and spec.schedule.intraday:
        minutes_by_day = {d: g for d, g in minute_bars.groupby(minute_bars.index.date)}
    square_off_fallback_days: set[dt.date] = set()
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
                f"The last candle ({bars.index[-1]:%d %b %Y %H:%M}) was still forming, so it "
                "was left out: acting on it would mean trading on a candle that had not "
                "finished."
            )
            bars = bars.iloc[:-1]

    features = build_features(spec, bars)
    entry_signal = evaluate(spec.entry, features)
    exit_signal = (
        evaluate(spec.exit.condition, features) if spec.exit.condition is not None else None
    )

    # Only when the user set times: the default schedule rides along on every
    # daily spec and a note about it on every daily run was noise.
    if not is_intraday_tf and spec.schedule.intraday and spec.schedule != Schedule():
        warnings.append(
            f"The square-off time ({spec.schedule.square_off:%H:%M}) was not used: on daily "
            "candles there is no time of day to act on, so positions are held overnight."
        )

    # --- session bookkeeping, used only for intraday timeframes below -------
    # `last_bar_of_session` backs two independent checks: forcing square-off on
    # a short/early-close day that never reaches the square-off clock time (2),
    # and refusing to schedule an entry fill across a session boundary (1).
    if is_intraday_tf and len(bars) > 0:
        last_bar_of_session = is_last_bar_of_session(bars.index, session).to_numpy()
        if final_session_in_progress:
            # The day is not over, so none of today's candles is its last.
            # Walk back from the newest candle only as far as today's first.
            today = session_date(bars.index[-1], session)
            last_bar_of_session = last_bar_of_session.copy()
            k = len(bars) - 1
            while k >= 0 and session_date(bars.index[k], session) == today:
                last_bar_of_session[k] = False
                k -= 1
    else:
        last_bar_of_session = np.zeros(len(bars), dtype=bool)

    # The bar on which a futures contract stops trading: the expiry day's
    # candle on daily bars, its session's last candle on intraday ones (never
    # a candle of today's session while it is still going on).
    expiry_close = np.zeros(len(bars), dtype=bool)
    if spec.instrument.is_future and len(bars) > 0:
        from nlt.data.futures import expiry_close_mask

        expiry_close = expiry_close_mask(
            spec.instrument.symbol,
            bars.index,
            spec.instrument.timeframe,
            listed_expiries,
            last_bar_of_session,
        )

    if is_intraday_tf and spec.indicators and session_bar_count:
        _warn_on_cross_session_lookback(spec, session_bar_count, warnings)
    elif is_intraday_tf and spec.indicators and session_bar_count is None:
        warnings.append(
            f"For {spec.instrument.timeframe} candles we could not check whether the "
            "indicators reach back across market closes."
        )

    max_concurrent = spec.risk.max_concurrent_positions
    margin_rate = _margin_rate(spec)
    entry_reason = _describe(spec.entry)

    symbol = spec.instrument.symbol
    n = len(bars)
    equity = np.empty(n, dtype="float64")
    trades: list[Trade] = []
    decisions: list[Decision] = []
    open_positions: list[_OpenPosition] = []
    pending_entry = False
    pending_entry_ts: pd.Timestamp | None = None
    realized_pnl = 0.0
    stop_beat_target_count = 0
    risk_fallback_count = 0
    dropped_for_concurrency = 0
    affordability_capped_count = 0
    affordability_skipped_count = 0
    boundary_entry_skipped_count = 0
    expiry_entry_skipped_count = 0
    expiry_exit_count = 0
    daily_limit = spec.schedule.max_entries_per_day
    entries_by_day: dict = {}
    daily_limit_skipped_count = 0
    level_skipped_count = 0
    entry_side = "buy" if spec.direction == "long" else "sell"

    for i in range(n):
        ts = bars.index[i]
        bar = bars.iloc[i]

        # --- 1. fill an entry scheduled from the previous bar's signal -----
        if pending_entry:
            pending_entry = False
            signal_ts = pending_entry_ts
            pending_entry_ts = None
            level_problem = level_stop_problem(
                spec, features, i, _apply_slippage(bar.open, entry_side, slippage)
            )
            if daily_limit is not None and entries_by_day.get(ts.date(), 0) >= daily_limit:
                daily_limit_skipped_count += 1
                decisions.append(Decision(signal_ts, symbol, "signal_skipped", "daily_limit", {}))
            elif level_problem is not None:
                level_skipped_count += 1
                decisions.append(
                    Decision(
                        signal_ts, symbol, "signal_skipped", "stop_level", {"why": level_problem}
                    )
                )
            elif len(open_positions) < max_concurrent:
                capital_total = capital + realized_pnl
                # Capital already tied up in other open positions is not spare
                # capital: without netting it out, two concurrent positions
                # could each be sized as if the whole account were free,
                # re-creating the leverage bug on max_concurrent_positions > 1.
                committed = margin_rate * sum(p.entry_price * p.quantity for p in open_positions)
                capital_spare = capital_total - committed
                pos, used_fallback, capped = _open_position(
                    spec,
                    features,
                    i,
                    ts,
                    bar,
                    slippage,
                    lot_size,
                    capital_total,
                    capital_spare,
                    signal_time=signal_ts,
                )
                if pos is not None:
                    open_positions.append(pos)
                    entries_by_day[ts.date()] = entries_by_day.get(ts.date(), 0) + 1
                    if used_fallback:
                        risk_fallback_count += 1
                    if capped:
                        affordability_capped_count += 1
                else:
                    affordability_skipped_count += 1
                    pct = spec.risk.max_position_pct
                    reason_code = "position_cap" if pct is not None else "capital"
                    decisions.append(Decision(signal_ts, symbol, "signal_skipped", reason_code, {}))
            else:
                dropped_for_concurrency += 1
                decisions.append(Decision(signal_ts, symbol, "signal_skipped", "concurrency", {}))

        # --- 2. check exits, in the mandated precedence order --------------
        still_open: list[_OpenPosition] = []
        for pos in open_positions:
            outcome = None
            if is_intraday_tf and spec.schedule.intraday and minutes_by_day:
                outcome = _minute_square_off(
                    pos,
                    ts,
                    spec,
                    bool(last_bar_of_session[i]),
                    minutes_by_day.get(ts.date()),
                    square_off_fallback_days,
                )
            if outcome is None:
                outcome = _check_exit(
                    pos,
                    bar,
                    ts,
                    spec,
                    is_intraday_tf,
                    bool(last_bar_of_session[i]),
                    bool(expiry_close[i]),
                )
            if outcome is None:
                pos.bars_held += 1
                still_open.append(pos)
                continue

            reason, raw_price, ambiguous = outcome
            if reason == "expiry":
                expiry_exit_count += 1
            if ambiguous:
                stop_beat_target_count += 1
            exit_side = "sell" if pos.direction == "long" else "buy"
            exit_price = _apply_slippage(raw_price, exit_side, slippage)
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
                    signal_time=pos.signal_time,
                    stop_level=pos.stop_level,
                    target_level=pos.target_level,
                    raw_entry_price=pos.raw_entry_price,
                    raw_exit_price=raw_price,
                    both_hit=bool(ambiguous),
                )
            )
        open_positions = still_open
        if open_positions and expiry_close[i]:
            # Unreachable while `_check_exit` closes every position on this
            # candle. Checked anyway: a position that outlived its contract would
            # be valued on the next contract's price, booking the jump between
            # the two as profit or loss nobody made.
            raise RuntimeError(f"a futures position was held past its contract's expiry at {ts}")

        # --- 3. mark-to-market equity for this bar --------------------------
        unrealized = sum(_gross_pnl(pos, bar.close) for pos in open_positions)
        equity[i] = capital + realized_pnl + unrealized
        _assert_equity_floor(equity[i], ts, spec.direction)

        # --- 4. schedule fills for the NEXT bar, from this closed bar's signal
        #
        # `no_entry_after`: no trade may open after it (`past_entry_cutoff`).
        is_intraday_spec = is_intraday_tf and spec.schedule.intraday
        entry_true = bool(entry_signal.iloc[i])
        concurrency_ok = len(open_positions) < max_concurrent
        weekday_ok = ts.weekday() in spec.schedule.weekdays
        paused = _in_blocked_window(ts, blocked_windows)
        no_entry_after_ok = not past_entry_cutoff(spec, ts, spec.instrument.timeframe)
        trade_from_ok = trade_from is None or ts >= trade_from
        daily_ok = daily_limit is None or entries_by_day.get(ts.date(), 0) < daily_limit
        can_enter = (
            concurrency_ok
            and entry_true
            and weekday_ok
            and not paused
            and no_entry_after_ok
            and daily_ok
            and trade_from_ok
        )
        # A decision is only meaningful once the signal could actually have
        # mattered: a bar before `trade_from` is history the engine is only
        # warming indicators on, never a choice it made (see `trade_from`'s
        # own docstring above).
        emit_decision = entry_true and trade_from_ok
        if emit_decision:
            if not concurrency_ok:
                decisions.append(
                    Decision(ts, symbol, "signal_skipped", "concurrency_at_signal", {})
                )
            elif not weekday_ok:
                decisions.append(Decision(ts, symbol, "signal_skipped", "weekday", {}))
            elif paused:
                decisions.append(Decision(ts, symbol, "signal_skipped", "paused", {}))
            elif not no_entry_after_ok:
                decisions.append(Decision(ts, symbol, "signal_skipped", "no_entry_after", {}))
            elif not daily_ok:
                daily_limit_skipped_count += 1
                decisions.append(Decision(ts, symbol, "signal_skipped", "daily_limit", {}))
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
        # A signal on the last candle of an expiring contract would fill in the
        # NEXT contract, at a different price level, with its stop and target
        # measured from the old contract's close (the levels belong to the
        # signal bar). The gap between two contracts is not a price move, so the
        # entry is skipped rather than filled on levels from another contract.
        if can_enter and bool(expiry_close[i]):
            expiry_entry_skipped_count += 1
            if emit_decision:
                decisions.append(Decision(ts, symbol, "signal_skipped", "expiry", {}))
        elif can_enter and is_intraday_spec and bool(last_bar_of_session[i]):
            boundary_entry_skipped_count += 1
            if emit_decision:
                decisions.append(Decision(ts, symbol, "signal_skipped", "boundary", {}))
        elif can_enter:
            # Emitted even on the very last bar of `bars`, where there is no
            # further bar in THIS run to fill on (`i + 1 >= n`, handled below):
            # paper trading calls this engine again next pass with one more
            # candle appended, and that candle is the fill this decision
            # promises. A real backtest simply has nothing to show for a
            # "queued" entry past the end of its data, which is fine --
            # nobody reads a finished backtest's decisions as a live feed.
            if emit_decision:
                decisions.append(Decision(ts, symbol, "signal_taken", "queued", {}))
            if i + 1 < n:
                pending_entry = True
                pending_entry_ts = ts

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
            exit_price = _apply_slippage(bar.close, exit_side, slippage)
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
                    signal_time=pos.signal_time,
                    stop_level=pos.stop_level,
                    target_level=pos.target_level,
                    raw_entry_price=pos.raw_entry_price,
                    raw_exit_price=bar.close,
                    both_hit=False,
                )
            )
        equity[-1] = capital + realized_pnl
        _assert_equity_floor(equity[-1], ts, spec.direction)

    if stop_beat_target_count:
        warnings.append(
            f"{stop_beat_target_count} trade(s) could have hit both the stop loss and the "
            "target within one candle; the stop was assumed to come first, the cautious choice."
        )
    if risk_fallback_count:
        warnings.append(
            f"{risk_fallback_count} trade(s) had no stop-loss distance to size the risk from, "
            "so the smallest size (one lot, or one share) was bought instead."
        )
    if dropped_for_concurrency:
        warnings.append(
            f"{dropped_for_concurrency} signal(s) to open a trade skipped: {max_concurrent} "
            "position(s) were already open, the most allowed at once."
        )
    if affordability_capped_count:
        warnings.append(
            f"{affordability_capped_count} trade(s) sized down: the full size needed more "
            "spare money than the account had, so it was cut to what the account could fund."
        )
    if affordability_skipped_count:
        # Distinguish "you are out of money" from "the concentration rail stopped
        # you". They look identical from the trade list and mean opposite things:
        # the first says trade smaller, the second says the rail is set tighter
        # than one indivisible lot, and staring at an untouched balance while
        # being told there is "not enough capital" would be baffling.
        pct = spec.risk.max_position_pct
        if pct is not None:
            cost = "lot's margin is" if spec.instrument.is_future else "lot costs"
            reason = (
                f"one position may not exceed {pct:g}% of the account, and a single "
                f"{cost} more than that. Either raise that limit or trade a "
                "cheaper contract"
            )
        else:
            reason = "not enough spare capital to fund even a single lot"
        warnings.append(
            f"{affordability_skipped_count} signal(s) to open a trade skipped: {reason}"
        )
    if expiry_exit_count:
        warnings.append(
            f"{expiry_exit_count} position(s) closed at the close of their contract's expiry "
            "day and charged for that exit. A position is never carried into the next month's "
            "contract; the strategy opens a new one only when its entry rule fires again."
        )
    if expiry_entry_skipped_count:
        warnings.append(
            f"{expiry_entry_skipped_count} entry signal(s) skipped: they fired on the last candle "
            "of an expiring contract, so the trade would have been placed in the next contract "
            "at a different price, with its stop and target measured from the old one"
        )
    if level_skipped_count:
        warnings.append(
            f"{level_skipped_count} entry signal(s) skipped: the stop level was not on the "
            "safe side of the price when the trade would have opened (or not known yet)."
        )
    if daily_limit_skipped_count:
        warnings.append(
            f"{daily_limit_skipped_count} entry signal(s) skipped: this strategy opens at most "
            f"{_plural_trades(daily_limit)} a day."
        )
    if boundary_entry_skipped_count:
        warnings.append(
            f"{boundary_entry_skipped_count} entry signal(s) skipped: they fired on the last "
            "bar of a trading session, and this intraday strategy would otherwise have filled "
            "them at the next session's open – a gap of 14+ hours the strategy never asked "
            "to hold through"
        )

    if square_off_fallback_days:
        days = len(square_off_fallback_days)
        warnings.append(
            f"On {days} day(s) there were no 1-minute prices, so the square-off there sold at "
            f"the close of the first candle starting at or after "
            f"{spec.schedule.square_off:%H:%M}, not at that minute's own close"
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
        decisions=decisions,
    )


# --------------------------------------------------------------- fills & pnl


def _in_blocked_window(
    ts: pd.Timestamp, blocked_windows: Iterable[tuple[dt.datetime, dt.datetime]]
) -> bool:
    """Is `ts` inside one of `blocked_windows`'s `[start, end)` ranges?

    `ts` comes from `bars.index`, which is timezone-aware; a window compared
    against it must be too (a naive datetime compared to an aware one raises),
    so callers pass windows already in `bars.index`'s timezone.
    """
    for start, end in blocked_windows:
        if start <= ts < end:
            return True
    return False


def _apply_slippage(price: float, side: str, slippage: Slippage) -> float:
    """A buy always fills a little worse (higher); a sell always fills a little worse (lower).

    This is applied to every fill -- entries and exits alike -- so a round trip
    always costs slippage twice, the same as it would in a live account. See
    `nlt.costs.slippage.Slippage.apply` for the actual arithmetic (percentage
    or rupee, and the floor on a sell price).
    """
    return slippage.apply(price, side)


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
            f"{exit_time} (session {exit_date}) – an intraday strategy must never "
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
        if ind.type == "daily_average":
            continue  # counted in days on purpose; it does not cross gaps candle by candle
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
            f"The slowest indicator looks back {longest} candles: about {sessions_spanned} "
            f"trading days at {session_bar_count} candles a day on "
            f"{spec.instrument.timeframe} candles. It runs straight across each overnight "
            "and weekend gap as if the candles either side were back to back."
        )


def _warn_about_margin(
    spec: StrategySpec, bars: pd.DataFrame, capital: float, lot_size: int, warnings: list[str]
) -> None:
    """Say what margin means for this result, and when a stop can outrun it.

    A futures position ties up only its margin but gains and loses on its full
    value, so every 1% the market moves is several percent of the money set
    aside. The user is told the figure we used and how old it is, and is warned
    if the stop is wide enough to lose more than the whole margin.
    """
    from nlt.data.futures import MARGIN_AS_OF, MARGIN_SOURCE, contract

    c = contract(spec.instrument.symbol)
    price = float(bars["close"].iloc[-1])
    margin = price * lot_size * c.margin_rate
    per_move = 1.0 / c.margin_rate
    warnings.append(
        f"FUTURES ON MARGIN: one lot of {c.title} ties up about {_inr(margin)} of margin "
        f"({c.margin_pct:g}% of the lot's value, {MARGIN_SOURCE} on "
        f"{MARGIN_AS_OF:%d %b %Y}; that figure is used for every year tested). Profit and "
        f"loss are on the whole lot, so a 1% move in the price is about {per_move:.0f}% of "
        "the margin set aside."
    )
    stop = spec.exit.stop_pct
    if stop is not None and stop >= c.margin_pct:
        warnings.append(
            f"STOP WIDER THAN THE MARGIN: a {stop:g}% stop on {c.title} can lose more than "
            f"the whole {c.margin_pct:g}% margin before it is hit. The broker would ask for "
            "more money, or close the position, before then."
        )


def _warn_if_one_lot_is_unaffordable(
    spec: StrategySpec, bars: pd.DataFrame, capital: float, lot_size: int, warnings: list[str]
) -> None:
    """Say up front when the account cannot hold a single lot, and what would.

    Futures are indivisible; one lot ties up its exchange margin (Gold: about
    Rs 14 lakh). Without this, a small account produces a backtest with no
    trades and a terse "skipped" count at the bottom -- which reads like a
    strategy that never signalled rather than an account that could not play.
    """
    from nlt.data.futures import contract

    c = contract(spec.instrument.symbol)
    price = float(bars["close"].iloc[-1])
    lot_margin = price * lot_size * c.margin_rate
    pct = spec.risk.max_position_pct
    budget = capital * (pct / 100.0 if pct is not None else 1.0)
    if lot_margin <= budget:
        return
    limit = (
        f"at most {pct:g}% of the account ({_inr(budget)} of {_inr(capital)}) may go into one "
        "position"
        if pct is not None
        else f"the account holds {_inr(capital)}"
    )
    needed = lot_margin / (pct / 100.0) if pct is not None else lot_margin
    warnings.append(
        f"ACCOUNT TOO SMALL FOR ONE LOT: one lot of {c.title} ({c.lot_text}) needs about "
        f"{_inr(lot_margin)} of margin at the latest price, and {limit}. No trade can be "
        f"taken at that size. Test with at least {_inr(needed)}, or a smaller contract."
    )


def _inr(value: float) -> str:
    from nlt.translate.readback import format_inr

    return format_inr(value)


def _open_position(
    spec: StrategySpec,
    features: pd.DataFrame,
    i: int,
    ts: pd.Timestamp,
    bar: pd.Series,
    slippage: Slippage,
    lot_size: int,
    capital_total: float,
    capital_spare: float,
    signal_time: pd.Timestamp | None = None,
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
    entry_price = _apply_slippage(bar.open, entry_side, slippage)

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
    lot_cost = entry_price * lot_size * _margin_rate(spec)
    affordable_lots = math.floor(capital_spare / lot_cost) if lot_cost > 0 else 0
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
        signal_time=signal_time,
        raw_entry_price=bar.open,
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
    level = level_stop(spec, features, i)
    if level is not None:
        # The stop sits AT the level: the distance is from the fill.
        candidates.append(abs(entry_price - level))
    if exit_rules.stop_atr_mult is not None and exit_rules.atr_id is not None and i >= 1:
        atr_val = features[exit_rules.atr_id].iloc[i - 1]
        if pd.notna(atr_val):
            candidates.append(exit_rules.stop_atr_mult * float(atr_val))
    return min(candidates) if candidates else None


def level_stop(spec: StrategySpec, features: pd.DataFrame, i: int) -> float | None:
    """`exit.stop_level`'s value on the signal candle (bar `i-1`), or None.

    Read once, at the signal, and kept for the trade's life: "stop at the low
    of the opening range" is the low as it stood when the trade was decided.
    """
    ref = spec.exit.stop_level
    if ref is None or i < 1:
        return None
    value = features[ref.key()].iloc[i - 1]
    return float(value) if pd.notna(value) else None


def level_stop_problem(
    spec: StrategySpec, features: pd.DataFrame, i: int, entry_price: float
) -> str | None:
    """Why a level stop cannot protect this trade, or None.

    A long's stop must sit below both the signal close and the fill (a short's
    above). A level the price has already passed -- or one not known yet --
    would stop the trade out on its first candle or never; the signal is
    skipped and said, never given another stop.
    """
    if spec.exit.stop_level is None:
        return None
    level = level_stop(spec, features, i)
    if level is None:
        return "the stop level was not known yet on the signal candle"
    basis = signal_close(features, i, entry_price)
    below = spec.direction == "long"
    if (
        (level >= basis or level >= entry_price)
        if below
        else (level <= basis or level <= entry_price)
    ):
        side = "below" if below else "above"
        return f"the stop level ({level:,.2f}) was not {side} the price when the trade would open"
    return None


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

    # The concentration rail, applied before the lot cap so both bind.
    # Rounded DOWN to a whole lot: one lot over the limit is still over it, and
    # a rail that rounds up in the account's disfavour is not doing its job.
    pct = spec.risk.max_position_pct
    if pct is not None and entry_price > 0:
        budget = pct / 100.0 * capital_now
        affordable_lots = math.floor(budget / (entry_price * lot_size * _margin_rate(spec)))
        quantity = min(quantity, max(affordable_lots, 0) * lot_size)

    # No lot cap is the normal case for stocks; capital and max_loss_per_trade
    # are what bound a stock position, and both scale with the share price.
    if spec.risk.max_lots is None:
        return quantity, used_fallback

    max_quantity = spec.risk.max_lots * lot_size
    return min(quantity, max_quantity), used_fallback


# ------------------------------------------------------------------- exits


def _minute_square_off(
    pos: _OpenPosition,
    ts: pd.Timestamp,
    spec: StrategySpec,
    is_last_bar_of_session: bool,
    minutes: pd.DataFrame | None,
    fallback_days: set[dt.date],
) -> tuple[str, float, bool] | None:
    """The square-off from 1-minute candles, on the candle that reaches it.

    The minutes from this candle's start to the end of the square-off minute
    are checked as one candle for the stop, target and the rest (the usual
    order, stop first); if none fires, the position sells at the close of the
    square-off minute -- the price at 15:15:59. None when this candle does not
    reach the square-off, or there are no minutes for the day (`_check_exit`
    then squares off the old way, and the day is counted).
    """
    clock = spec.schedule.square_off
    at = ts.normalize() + pd.Timedelta(hours=clock.hour, minutes=clock.minute)
    candle = _CANDLE_LENGTH.get(spec.instrument.timeframe)
    if not (ts + candle > at or is_last_bar_of_session):
        return None
    upto = None if minutes is None else minutes[minutes.index < at + pd.Timedelta(minutes=1)]
    if upto is None or upto.empty:
        fallback_days.add(ts.date())
        return None
    inside = upto[upto.index >= ts]
    if len(inside):
        before = pd.Series(
            {
                "open": float(inside["open"].iloc[0]),
                "high": float(inside["high"].max()),
                "low": float(inside["low"].min()),
                "close": float(inside["close"].iloc[-1]),
            }
        )
        hit = _check_exit(pos, before, ts, spec, False, False)
        if hit is not None:
            return hit
    return "square_off", float(upto["close"].iloc[-1]), False


def _check_exit(
    pos: _OpenPosition,
    bar: pd.Series,
    ts: pd.Timestamp,
    spec: StrategySpec,
    is_intraday_tf: bool,
    is_last_bar_of_session: bool,
    is_expiry_close: bool = False,
) -> tuple[str, float, bool] | None:
    """Returns (reason, raw fill price before slippage, ambiguous_stop_vs_target) or None.

    Checked in the precedence the brief mandates: square-off, stop, trailing
    stop, target, condition, max_bars_held. The first one that fires wins --
    a bar that hits both a stop and a target only ever reports the stop.
    """
    long = pos.direction == "long"

    if (
        is_intraday_tf
        and spec.schedule.intraday
        and (ts.time() >= spec.schedule.square_off or is_last_bar_of_session)
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
            gapped_through = bar.open < pos.stop_level if long else bar.open > pos.stop_level
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
        gapped_past = bar.open > pos.target_level if long else bar.open < pos.target_level
        return "target", (bar.open if gapped_past else pos.target_level), False

    if pos.pending_condition_exit:
        return "condition", bar.open, False

    if spec.exit.max_bars_held is not None and pos.bars_held + 1 >= spec.exit.max_bars_held:
        return "max_bars", bar.close, False

    if is_expiry_close:
        # Last: every other exit can happen during the day, this one only at
        # the close. The next candle belongs to a different contract.
        return "expiry", bar.close, False

    return None


# -------------------------------------------------------------- description


def _describe(condition: Condition) -> str:
    """Best-effort plain-English rendering of a condition tree, for `Trade.entry_reason`."""
    if isinstance(condition, Compare):
        left = _describe_operand(condition.left)
        right = _describe_operand(condition.right)
        verbs = {
            "lt": "<",
            "lte": "<=",
            "gt": ">",
            "gte": ">=",
            "eq": "==",
            "crosses_above": "crossed above",
            "crosses_below": "crossed below",
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
    if isinstance(condition, AtTime):
        return f"the clock reached {condition.time:%H:%M}"
    if isinstance(condition, OnExpiryDay):
        return "expiry day"
    if isinstance(condition, IvPercentile):
        verbs = {"lt": "<", "lte": "<=", "gt": ">", "gte": ">="}
        return f"IV percentile {verbs[condition.op]} {condition.value:g}"
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
