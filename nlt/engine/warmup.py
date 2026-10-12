"""Indicator warm-up: history loaded before the start date so indicators are ready on it.

A 200-day average means nothing until 200 days of prices exist. Loading prices
from the user's start date therefore spent the first 200 days of their test
warming up, and no trade could happen in them -- silently. Justin's call
(2026-10-12): load the extra history first, so trading can begin on the date
they chose, and say how much was used.

How much: three times the longest lookback. A simple average is exact after
one; recursive ones (EMA, RSI, ATR) start from a seed and settle over roughly
three, which is what keeps them close to TradingView's, computed over all
history. History before `DATA_FLOOR` is not trusted (see `nlt.data.source`),
so the padding stops there, and the report says when that left an indicator
not ready on the start date.

Nothing here changes what the engine sees: it still judges closed bars only.
Bars before the start can only *feed indicators*; `trade_from` stops them
opening a position, and the result is cut back to the start so returns,
drawdown and time in the market cover only the period the user chose.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math

import pandas as pd

from nlt.data.source import DATA_FLOOR
from nlt.engine.backtest import _DEFAULT_INTRADAY_BARS_PER_YEAR, BARS_PER_YEAR, BacktestResult
from nlt.engine.metrics import compute_metrics
from nlt.spec.models import StrategySpec

SETTLE_FACTOR = 3

# Rough bars per NSE session, to turn a bar count into calendar days.
_BARS_PER_SESSION = {"1m": 375, "3m": 125, "5m": 75, "15m": 25, "30m": 13, "1h": 7, "1d": 1}


def longest_lookback(spec: StrategySpec) -> int:
    """The most bars any one indicator needs before its first value.

    Lookback-shaped parameters are the model's own convention (see
    `_warn_on_cross_session_lookback`): "length" in the name, or fast/slow/
    signal/period. They are added up per indicator because they stack: MACD's
    signal line is averaged over values that already needed the slow length.
    """
    longest = 0
    per_session = _BARS_PER_SESSION.get(spec.instrument.timeframe, 1)
    for ind in spec.indicators:
        if ind.type == "daily_average":
            # Its length is in days, whatever the candles: that many sessions.
            length = int(ind.resolved_params().get("length", 50))
            longest = max(longest, length * per_session)
            continue
        total = 0
        for key, value in ind.resolved_params().items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if "length" in key or key in {"fast", "slow", "signal", "period"}:
                total += int(value)
        longest = max(longest, total)
    return longest


def warmup_bars(spec: StrategySpec) -> int:
    return SETTLE_FACTOR * longest_lookback(spec)


def padded_start(spec: StrategySpec, start: dt.date) -> dt.date:
    """Where to start loading so `warmup_bars` exist before `start`."""
    bars = warmup_bars(spec)
    if bars == 0:
        return start
    per_session = _BARS_PER_SESSION.get(spec.instrument.timeframe, 1)
    sessions = math.ceil(bars / per_session)
    # 7 calendar days hold 5 sessions; ten more days cover holidays.
    days = math.ceil(sessions * 7 / 5) + 10
    return max(start - dt.timedelta(days=days), DATA_FLOOR)


def first_bar_on_or_after(index: pd.DatetimeIndex, start: dt.date) -> pd.Timestamp | None:
    on_or_after = index[index.normalize().date >= start] if len(index) else index
    return on_or_after[0] if len(on_or_after) else None


def trim(result: BacktestResult, start_bar: pd.Timestamp, capital: float) -> BacktestResult:
    """`result` cut back to `start_bar`, with its numbers recomputed on that span.

    No trade can exist before `start_bar` (the run used `trade_from`), so the
    equity before it is just the untouched capital; dropping it changes no rupee,
    only which days count towards returns, drawdown and time in the market.
    """
    equity = result.equity.loc[start_bar:]
    timeframe = result.spec.instrument.timeframe
    bars_per_year = BARS_PER_YEAR.get(timeframe, _DEFAULT_INTRADAY_BARS_PER_YEAR)
    metrics = compute_metrics(result.trades, equity, capital, bars_per_year)
    return dataclasses.replace(result, equity=equity, metrics=metrics)


def warmup_note(spec: StrategySpec, bars: pd.DataFrame, start_bar: pd.Timestamp) -> str | None:
    """What the report says about warm-up, or None when no indicator needs one."""
    need = longest_lookback(spec)
    if need == 0:
        return None
    before = int((bars.index < start_bar).sum())
    unit = "day" if spec.instrument.timeframe == "1d" else "candle"
    if before >= need:
        first = bars.index[0]
        return (
            f"Indicators were warmed up on {before:,} {unit}s of history before your start "
            f"date (from {first:%d %b %Y}), so trading could begin on the first day."
        )
    ready = bars.index[min(need - 1, len(bars) - 1)]
    return (
        f"Only {before:,} {unit}s of history exist before your start date (prices here start "
        f"on {DATA_FLOOR:%d %b %Y}), and the slowest indicator needs {need:,}. It was not ready "
        f"until {ready:%d %b %Y}, so no trade could happen before then."
    )
