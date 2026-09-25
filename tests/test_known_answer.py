"""The known-answer backtest.

Every other test checks a component against its own definition. This one checks
the assembled engine against an answer worked out entirely outside it: the dates
on which a 20/50 SMA crossover occurs on real NIFTY history, computed in four
lines of pandas with no reference to any engine code.

If the engine's trade dates do not line up with those crossovers, something
between the signal and the fill is wrong -- and that is a class of bug the
component tests cannot see, because each part can be individually correct while
the assembly is not.

The plan called for cross-checking against TradingView. Without an account this
recomputes the crossover independently instead, which catches the same failure:
an engine that signals on the wrong bar, fills on the wrong bar, or silently
drops signals.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nlt.engine.backtest import run_backtest
from nlt.spec.models import StrategySpec


def _crossover_dates(close: pd.Series, fast: int = 20, slow: int = 50) -> list[pd.Timestamp]:
    """Bars where the fast average crosses above the slow one.

    Written longhand and deliberately not using anything from `nlt.indicators`,
    so a bug shared by the indicator library and the engine cannot hide here.
    """
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    crossed = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    return list(close.index[crossed.fillna(False)])


def _crossover_spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "name": "20/50 crossover",
            "description": "Buy NIFTY when the 20 SMA crosses above the 50 SMA",
            "instrument": {"symbol": "NIFTY", "trade_as": "index"},
            "indicators": [
                {"id": "sma20", "type": "sma", "params": {"length": 20}},
                {"id": "sma50", "type": "sma", "params": {"length": 50}},
            ],
            "entry": {
                "kind": "compare", "op": "crosses_above",
                "left": {"kind": "ref", "name": "sma20"},
                "right": {"kind": "ref", "name": "sma50"},
            },
            # A single long hold per signal, so each crossover maps to one trade
            # and the comparison is one-to-one.
            "exit": {"max_bars_held": 10},
            "sizing": {"mode": "fixed_lots", "lots": 1},
            "risk": {"max_concurrent_positions": 1, "max_position_pct": 100.0},
        }
    )


def test_engine_trades_match_independently_computed_crossovers(nifty_bars):
    """Every trade must begin on the bar after a real 20/50 crossover."""
    bars = nifty_bars
    expected = set(_crossover_dates(bars.close))
    assert len(expected) > 5, "too few crossovers in the data to be a real test"

    result = run_backtest(
        _crossover_spec(), bars, capital=10_000_000.0, lot_size=1, slippage_pct=0.0
    )
    assert result.trades, "the engine found no trades where crossovers plainly exist"

    positions = {ts: i for i, ts in enumerate(bars.index)}
    for trade in result.trades:
        entry_i = positions[trade.entry_time]
        signal_bar = bars.index[entry_i - 1]
        assert signal_bar in expected, (
            f"trade entered {trade.entry_time.date()}, but the bar before it "
            f"({signal_bar.date()}) is not a 20/50 crossover"
        )


def test_engine_finds_every_crossover_it_could_act_on(nifty_bars):
    """Signals must not be silently dropped.

    Only crossovers that fire while flat can be acted on, since the spec holds
    one position at a time -- so the check is that every *actionable* crossover
    produced a trade, not merely that the trades that exist are valid.
    """
    bars = nifty_bars
    crossovers = _crossover_dates(bars.close)
    result = run_backtest(
        _crossover_spec(), bars, capital=10_000_000.0, lot_size=1, slippage_pct=0.0
    )

    positions = {ts: i for i, ts in enumerate(bars.index)}
    held = set()
    for trade in result.trades:
        held.update(range(positions[trade.entry_time], positions[trade.exit_time] + 1))

    entered_after = {positions[t.entry_time] - 1 for t in result.trades}
    missed = [
        ts for ts in crossovers
        if positions[ts] not in held
        and positions[ts] not in entered_after
        and positions[ts] < len(bars) - 2
    ]
    assert not missed, (
        f"{len(missed)} crossover(s) produced no trade while flat, first at "
        f"{missed[0].date() if missed else None}"
    )


def test_entry_price_is_the_next_bars_open(nifty_bars):
    """The fill rule, checked against the raw data rather than the engine's own
    bookkeeping."""
    bars = nifty_bars
    result = run_backtest(
        _crossover_spec(), bars, capital=10_000_000.0, lot_size=1, slippage_pct=0.0
    )

    for trade in result.trades[:20]:
        assert trade.entry_price == pytest.approx(bars.loc[trade.entry_time, "open"]), (
            f"entered at {trade.entry_price} but that bar opened at "
            f"{bars.loc[trade.entry_time, 'open']}"
        )
