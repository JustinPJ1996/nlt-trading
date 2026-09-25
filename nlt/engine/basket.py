"""Running one strategy across a basket of symbols with one shared capital pool.

`run_backtest` (`nlt/engine/backtest.py`) walks one symbol's bars, one bar at a
time, deciding fills and exits against a single account. A "buy Nifty 100
stocks when RSI drops below 40" strategy is not fifty independent runs of that
loop summed together -- capital is shared, so two symbols signalling on the
same day are competing for the same rupees, and `max_concurrent_positions` is
a limit on the whole portfolio, not on each name separately. Summing fifty
single-symbol backtests would silently assume infinite capital, which is
exactly the leverage bug `_open_position`'s affordability cap exists to rule
out in the single-symbol case -- doing the sum instead would just move that
bug up one level.

This module does NOT reimplement the fill or exit rules. Every entry fill goes
through `_open_position` and every exit through `_check_exit`, the same two
functions `run_backtest` calls -- see the imports below. What is different
here is the *loop* around them: instead of one pending-entry flag and one list
of open positions, there is one per symbol, all reading from and writing to a
single shared cash/position-count balance sheet. Reusing the single-symbol
helpers rather than re-deriving "does this bar hit the stop" or "can this
fill be afforded" is what keeps this file from drifting from
`run_backtest`'s behaviour -- see `test_basket.py`'s single-symbol-equivalence
test, which asserts the two produce byte-identical trades when there is only
one symbol in the basket.

Symbols are aligned to the UNION of their own trading days, never forward-filled
(see `_union_index` and the per-symbol `.index` lookups throughout the loop): a
day a symbol has no bar for -- a halt, a late listing, a delisting -- is a day
that symbol contributes nothing, not a day it silently repeats its last price.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nlt.engine.backtest import (
    _DEFAULT_INTRADAY_BARS_PER_YEAR,
    BARS_PER_YEAR,
    Trade,
    _apply_slippage,
    _assert_equity_floor,
    _check_exit,
    _default_lot_size,
    _describe,
    _gross_pnl,
    _open_position,
    _OpenPosition,
    _zero_charges,
)
from nlt.engine.conditions import evaluate
from nlt.engine.features import build_features
from nlt.engine.metrics import compute_metrics
from nlt.spec.models import StrategySpec


@dataclass
class BasketResult:
    spec: StrategySpec
    trades: list[Trade]
    equity: pd.Series
    metrics: dict
    per_symbol: dict[str, dict]
    warnings: list[str]
    skipped: dict[str, str]


@dataclass
class _SymbolTrack:
    """Everything about one symbol that stays local to it: its own bars,
    its own precomputed signals (vectorised once, exactly as `run_backtest`
    does), and its own open positions. Cash, drawdown and the concurrency
    limit live one level up, in `run_basket_backtest` itself -- see the module
    docstring on why that split is the whole point.
    """

    symbol: str
    bars: pd.DataFrame
    features: pd.DataFrame
    entry_signal: pd.Series
    exit_signal: pd.Series | None
    entry_reason: str
    open_positions: list[_OpenPosition] = field(default_factory=list)
    pending_entry: bool = False
    trades: list[Trade] = field(default_factory=list)


def run_basket_backtest(
    spec: StrategySpec,
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    capital: float = 100_000.0,
    charge_fn=None,
    slippage_pct: float = 0.02,
) -> BasketResult:
    """Run `spec` across every symbol in `bars_by_symbol`, sharing one account.

    `spec.risk.max_concurrent_positions` bounds the whole portfolio: if it is
    2 and six symbols signal on the same day, at most two of those signals can
    ever be filled, no matter how much capital is free. Which two is decided
    by `spec.instrument.selection` (see `_contention_order` below) and is
    always deterministic -- the same inputs produce the same trades every
    time, which `test_basket.py` checks directly by running twice and
    comparing trade lists.
    """

    # The basket loop does not carry the intraday machinery the single-symbol
    # engine has: no square-off at the session close, and no overnight-carry
    # invariant. Running an intraday spec here would silently hold positions
    # through the night on a strategy whose author wrote "square off at 3:15",
    # and the single-symbol engine treats exactly that as a hard error.
    #
    # Refusing is the only consistent answer. Every stock basket today is on
    # daily bars, so this costs nothing and prevents a whole class of wrong
    # results that would look completely ordinary.
    if spec.instrument.timeframe != "1d":
        raise NotImplementedError(
            f"basket backtests are daily-only for now; this strategy asks for "
            f"{spec.instrument.timeframe} bars. The intraday session rules "
            "(square-off, no overnight carry) are implemented for single "
            "instruments but not yet across a basket, and running without them "
            "would hold positions overnight on an intraday strategy."
        )

    warnings: list[str] = []
    skipped: dict[str, str] = {}
    if charge_fn is None:
        charge_fn = _zero_charges
        warnings.append("no charge_fn supplied; results exclude transaction costs")

    lot_size = _default_lot_size(spec)
    is_intraday_tf = spec.instrument.timeframe != "1d"
    max_concurrent = spec.risk.max_concurrent_positions
    entry_reason = _describe(spec.entry)

    if spec.instrument.is_universe:
        _add_survivorship_warning(spec, bars_by_symbol, warnings)

    tracks: dict[str, _SymbolTrack] = {}
    for symbol, bars in bars_by_symbol.items():
        if bars is None or bars.empty:
            skipped[symbol] = "no bars available"
            continue
        try:
            features = build_features(spec, bars)
            entry_signal = evaluate(spec.entry, features)
            exit_signal = (
                evaluate(spec.exit.condition, features)
                if spec.exit.condition is not None
                else None
            )
        except Exception as exc:  # a bad symbol must not abort the whole basket
            skipped[symbol] = f"failed to build signals: {exc}"
            continue
        tracks[symbol] = _SymbolTrack(
            symbol=symbol,
            bars=bars,
            features=features,
            entry_signal=entry_signal,
            exit_signal=exit_signal,
            entry_reason=entry_reason,
        )

    bars_per_year = BARS_PER_YEAR.get(spec.instrument.timeframe, _DEFAULT_INTRADAY_BARS_PER_YEAR)

    if not tracks:
        empty_equity = pd.Series(dtype="float64", name="equity")
        metrics = compute_metrics([], empty_equity, capital, bars_per_year)
        return BasketResult(
            spec=spec,
            trades=[],
            equity=empty_equity,
            metrics=metrics,
            per_symbol={},
            warnings=warnings,
            skipped=skipped,
        )

    ordered_symbols = _contention_order(spec, tracks)
    last_bar_ts = {s: t.bars.index[-1] for s, t in tracks.items()}
    all_days = _union_index(tracks)

    all_trades: list[Trade] = []
    equity = np.empty(len(all_days), dtype="float64")
    realized_pnl = 0.0

    dropped_for_slots = 0
    dropped_for_capital = 0
    risk_fallback_count = 0
    capped_count = 0

    for i, ts in enumerate(all_days):
        # --- 1. fill entries scheduled from the previous bar's signal, in ---
        # contention order, against capital netted for what is already open
        # (the exact `capital_spare` idea `_open_position` documents for the
        # single-symbol case -- extended here to sum across every symbol's
        # open positions, not just one symbol's).
        total_open = sum(len(t.open_positions) for t in tracks.values())
        committed = sum(
            p.entry_price * p.quantity for t in tracks.values() for p in t.open_positions
        )
        capital_total = capital + realized_pnl
        capital_spare = capital_total - committed

        candidates = [
            s for s in ordered_symbols if tracks[s].pending_entry and ts in tracks[s].bars.index
        ]
        if spec.instrument.selection == "ranked":
            candidates = sorted(
                candidates,
                key=lambda s: (
                    -_rank_key(tracks[s], tracks[s].bars.index.get_loc(ts)),
                    s,
                ),
            )
        for s in candidates:
            track = tracks[s]
            track.pending_entry = False
            if total_open >= max_concurrent:
                dropped_for_slots += 1
                continue
            j = track.bars.index.get_loc(ts)
            bar = track.bars.iloc[j]
            pos, used_fallback, capped = _open_position(
                spec, track.features, j, ts, bar, slippage_pct, lot_size,
                capital_total, capital_spare,
            )
            if pos is None:
                dropped_for_capital += 1
                continue
            if used_fallback:
                risk_fallback_count += 1
            if capped:
                capped_count += 1
            track.open_positions.append(pos)
            total_open += 1
            committed += pos.entry_price * pos.quantity
            capital_spare = capital_total - committed

        # --- 2. check exits, per symbol, in the same precedence order -------
        # `run_backtest` uses (see `_check_exit`). A symbol on the last bar of
        # its OWN data (delisted, or simply the end of the download) is force
        # closed here too, mirroring `run_backtest`'s end-of-data handling --
        # done inline, per symbol, rather than after the whole basket's loop,
        # so a symbol that stops trading early does not silently vanish from
        # equity for the days afterward.
        for s in ordered_symbols:
            track = tracks[s]
            if ts not in track.bars.index:
                continue
            j = track.bars.index.get_loc(ts)
            bar = track.bars.iloc[j]
            is_own_last_bar = ts == last_bar_ts[s]

            still_open: list[_OpenPosition] = []
            for pos in track.open_positions:
                outcome = _check_exit(pos, bar, ts, spec, is_intraday_tf, False)
                if outcome is None and not is_own_last_bar:
                    pos.bars_held += 1
                    still_open.append(pos)
                    continue
                if outcome is None and is_own_last_bar:
                    # Mirrors `run_backtest` exactly: a position that survives
                    # every ordinary exit check even on the data's last bar
                    # still takes the same "still open, bump bars_held" branch
                    # for that bar (see the loop above), and is *then* force
                    # closed by a separate block that adds one more for the
                    # closing bar itself. The same bar is counted twice by
                    # that convention -- once for "held through" it, once for
                    # "exited on" it -- and this replicates it rather than
                    # quietly fixing what looks like an off-by-one, since a
                    # single-symbol basket must produce byte-identical trades.
                    pos.bars_held += 1
                    outcome = ("end_of_data", bar.close, False)
                reason, raw_price, _ambiguous = outcome
                trade = _close_position(
                    pos, s, ts, reason, raw_price, slippage_pct, charge_fn, track.entry_reason
                )
                realized_pnl += trade.net_pnl
                track.trades.append(trade)
                all_trades.append(trade)
            track.open_positions = still_open

        # --- 3. mark-to-market equity for this bar, portfolio-wide ----------
        unrealized = 0.0
        for s in ordered_symbols:
            track = tracks[s]
            if ts not in track.bars.index or not track.open_positions:
                continue
            close_px = track.bars.loc[ts, "close"]
            unrealized += sum(_gross_pnl(pos, close_px) for pos in track.open_positions)
        equity[i] = capital + realized_pnl + unrealized
        _assert_equity_floor(equity[i], ts, spec.direction)

        # --- 4. schedule next entries + mark pending condition exits --------
        # The portfolio-wide slot count used here is taken AFTER this bar's
        # exits (mirroring `run_backtest`'s scheduling gate, which checks
        # `open_positions` post-exit) so a slot freed by an exit this same bar
        # is available to a fresh signal on this same bar.
        total_open_after_exits = sum(len(t.open_positions) for t in tracks.values())
        for s in ordered_symbols:
            track = tracks[s]
            if ts not in track.bars.index:
                continue
            j = track.bars.index.get_loc(ts)
            weekday_ok = ts.weekday() in spec.schedule.weekdays
            sig = bool(track.entry_signal.iloc[j])
            has_next_bar = j + 1 < len(track.bars)
            if sig and weekday_ok and has_next_bar:
                if total_open_after_exits < max_concurrent:
                    track.pending_entry = True
                else:
                    # The portfolio is already fully invested, so this signal
                    # never even gets a chance to compete at fill time -- it
                    # would otherwise vanish from every count below (fill-time
                    # `dropped_for_slots` only sees signals that made it to
                    # `pending_entry`). Counted here so a strategy that fires
                    # on 40 names while 2 slots are held does not look, from
                    # the warnings alone, like it only ever saw 2 signals.
                    dropped_for_slots += 1
            if track.exit_signal is not None and bool(track.exit_signal.iloc[j]):
                for pos in track.open_positions:
                    pos.pending_condition_exit = True

    if dropped_for_slots:
        warnings.append(
            f"{dropped_for_slots} entry signal(s) dropped: max_concurrent_positions "
            f"({max_concurrent}) already reached across the basket"
        )
    if dropped_for_capital:
        warnings.append(
            f"{dropped_for_capital} entry signal(s) dropped: not enough spare capital "
            "to fund even a single lot at the time they fired"
        )
    if risk_fallback_count:
        warnings.append(
            f"{risk_fallback_count} trade(s) used risk_based sizing with no stop distance "
            "available; sized as 1 lot instead"
        )
    if capped_count:
        warnings.append(
            f"{capped_count} entr{'y' if capped_count == 1 else 'ies'} sized down: the "
            "requested position value exceeded spare capital, so it was reduced to what "
            "the account could actually fund"
        )

    equity_series = pd.Series(equity, index=all_days, name="equity")
    metrics = compute_metrics(all_trades, equity_series, capital, bars_per_year)
    per_symbol = {s: _per_symbol_summary(t.trades) for s, t in tracks.items()}

    return BasketResult(
        spec=spec,
        trades=all_trades,
        equity=equity_series,
        metrics=metrics,
        per_symbol=per_symbol,
        warnings=warnings,
        skipped=skipped,
    )


def _close_position(
    pos: _OpenPosition,
    symbol: str,
    ts: pd.Timestamp,
    reason: str,
    raw_price: float,
    slippage_pct: float,
    charge_fn,
    entry_reason: str,
) -> Trade:
    """Turn a `_check_exit` outcome into a `Trade`, exactly as `run_backtest` does.

    Kept as one small function (rather than inlined twice) since the basket
    loop needs it both for an ordinary exit and for the end-of-data force
    close, and it must do precisely what `run_backtest`'s two Trade-building
    blocks do -- same slippage direction, same charge_fn call shape, same
    pnl arithmetic (`_gross_pnl`, imported, not re-derived).
    """
    exit_side = "sell" if pos.direction == "long" else "buy"
    exit_price = _apply_slippage(raw_price, exit_side, slippage_pct)
    entry_side = "buy" if pos.direction == "long" else "sell"
    charges = charge_fn(pos.entry_price, pos.quantity, entry_side) + charge_fn(
        exit_price, pos.quantity, exit_side
    )
    gross = _gross_pnl(pos, exit_price)
    net = gross - charges
    return Trade(
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
        symbol=symbol,
    )


def _union_index(tracks: dict[str, _SymbolTrack]) -> pd.DatetimeIndex:
    """The union of every symbol's own trading days, sorted, deduplicated.

    Nothing is forward-filled onto this index: it exists only to give the
    outer loop a single clock to iterate, and every per-symbol lookup below
    checks `ts in track.bars.index` before touching that symbol's bars, so a
    day a symbol has no row for is a day that symbol is simply skipped.
    """
    days: set[pd.Timestamp] = set()
    for track in tracks.values():
        days.update(track.bars.index)
    return pd.DatetimeIndex(sorted(days))


def _contention_order(spec: StrategySpec, tracks: dict[str, _SymbolTrack]) -> list[str]:
    """A fixed, deterministic order to resolve contention when more symbols
    signal on a bar than there is room (capital or `max_concurrent_positions`)
    to take them all.

    `selection == "first"`: the universe's own membership order if the spec
    names a universe (alphabetical, per `nlt.data.universe`), or otherwise the
    order `bars_by_symbol` was given in. Arbitrary, but stated and stable.

    `selection == "ranked"`: at fill time (see the `candidates` sort inside
    the main loop), symbols are instead ranked by the *signal bar's* absolute
    percentage move -- `abs(close[j-1] / close[j-2] - 1)` for signal bar
    `j-1`, descending -- computed fresh for the actual day being filled, not
    fixed for the whole backtest. It is a generic, always-available proxy for
    "how strong was the move that produced this signal" that does not depend
    on which indicator the strategy happens to use (RSI, a moving average, a
    raw price cross all have wildly different scales, so ranking on the
    condition's own indicator value would need a bespoke rule per indicator
    type). It is a stated, debatable choice, not a neutral one -- see the
    report's self-critique. Ties (including "no prior bar to measure") are
    broken by symbol name so two runs of the same basket never disagree.

    This function only supplies the *iteration order* used elsewhere (exit
    checks, equity mark-to-market, schedule flagging), where order does not
    affect the result -- and, for `selection == "first"`, the contention
    order itself. For `selection == "ranked"` the contention order is instead
    recomputed per day in the main loop, since ranking is a property of that
    day's move, not a fixed property of the symbol.
    """
    if spec.instrument.is_universe:
        from nlt.data.universe import get_universe

        try:
            universe_order = list(get_universe(spec.instrument.symbol).symbols)
            return [s for s in universe_order if s in tracks] + sorted(
                s for s in tracks if s not in universe_order
            )
        except Exception:
            pass
    return list(tracks)


def _rank_key(track: _SymbolTrack, j: int) -> float:
    """`abs` percentage move of the signal bar (`j - 1`) versus the bar before
    it, used to rank `selection == "ranked"` contention for the fill scheduled
    at bar `j`. Falls back to 0.0 (lowest rank) when there is no such history,
    which the symbol-name tiebreak in `_contention_order`'s caller then
    resolves deterministically.
    """
    closes = track.bars["close"]
    if j < 2:
        return 0.0
    prior = float(closes.iloc[j - 1])
    prior_prior = float(closes.iloc[j - 2])
    if prior_prior == 0:
        return 0.0
    return abs(prior / prior_prior - 1.0)


def _per_symbol_summary(trades: list[Trade]) -> dict:
    if not trades:
        return {"trades": 0, "net_pnl": 0.0, "win_rate_pct": 0.0}
    net_pnl = sum(t.net_pnl for t in trades)
    wins = sum(1 for t in trades if t.net_pnl > 0)
    return {
        "trades": len(trades),
        "net_pnl": float(net_pnl),
        "win_rate_pct": 100.0 * wins / len(trades),
    }


def _add_survivorship_warning(
    spec: StrategySpec, bars_by_symbol: dict[str, pd.DataFrame], warnings: list[str]
) -> None:
    """A universe backtest only ever sees today's survivors (see `nlt.data.universe`'s
    module docstring); this makes that impossible to miss in `BasketResult.warnings`
    rather than something only visible to a caller who thought to check.
    """
    from nlt.data.universe import get_universe, survivorship_warning

    try:
        universe = get_universe(spec.instrument.symbol)
    except Exception:
        return
    starts = [df.index[0].date() for df in bars_by_symbol.values() if df is not None and len(df)]
    if not starts:
        return
    msg = survivorship_warning(universe, min(starts))
    if msg:
        warnings.append(msg)
