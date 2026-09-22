"""Price levels -- pivots, CPR, prior period highs/lows, Fibonacci retracements.

Every level here is computed from *completed* periods and then shifted forward,
so a level available on a given bar was genuinely knowable before that bar opened.
"""

from __future__ import annotations

import pandas as pd


def _prev_period_hlc(
    high: pd.Series, low: pd.Series, close: pd.Series, period: str
) -> pd.DataFrame:
    """Previous period's high/low/close, aligned onto each bar of the next one."""
    rule = {"day": "D", "week": "W-FRI", "month": "ME"}[period]

    agg = pd.DataFrame(
        {
            "high": high.resample(rule).max(),
            "low": low.resample(rule).min(),
            "close": close.resample(rule).last(),
        }
    ).dropna()

    # shift(1) is what makes this safe: bars in period N see period N-1's values.
    prev = agg.shift(1)
    return prev.reindex(high.index, method="ffill")


def pivots(
    high: pd.Series, low: pd.Series, close: pd.Series, period: str = "day"
) -> pd.DataFrame:
    """Classic floor-trader pivots with three support and resistance levels."""
    prev = _prev_period_hlc(high, low, close, period)
    p = (prev.high + prev.low + prev.close) / 3.0
    span = prev.high - prev.low

    return pd.DataFrame(
        {
            "pivot": p,
            "r1": 2.0 * p - prev.low,
            "s1": 2.0 * p - prev.high,
            "r2": p + span,
            "s2": p - span,
            "r3": prev.high + 2.0 * (p - prev.low),
            "s3": prev.low - 2.0 * (p - prev.high),
        },
        index=high.index,
    )


def central_pivot_range(
    high: pd.Series, low: pd.Series, close: pd.Series, period: str = "day"
) -> pd.DataFrame:
    """CPR -- pivot, bottom and top central pivot, plus the range width.

    A narrow CPR is read as a breakout setup, a wide one as range-bound. Varsity
    covers this in its final chapter.
    """
    prev = _prev_period_hlc(high, low, close, period)
    pivot = (prev.high + prev.low + prev.close) / 3.0
    bc = (prev.high + prev.low) / 2.0
    tc = 2.0 * pivot - bc

    # BC and TC are defined by which is larger, not by their formulas.
    top = pd.concat([bc, tc], axis=1).max(axis=1)
    bottom = pd.concat([bc, tc], axis=1).min(axis=1)

    return pd.DataFrame(
        {
            "pivot": pivot,
            "tc": top,
            "bc": bottom,
            "width_pct": 100.0 * (top - bottom) / pivot,
        },
        index=high.index,
    )


def prior_period(
    high: pd.Series, low: pd.Series, close: pd.Series, period: str = "day"
) -> pd.DataFrame:
    """Previous day's/week's high, low and close as named levels."""
    prev = _prev_period_hlc(high, low, close, period)
    return prev.rename(columns={"high": "high", "low": "low", "close": "close"})


def rolling_extremes(high: pd.Series, low: pd.Series, length: int = 20) -> pd.DataFrame:
    """Rolling highest high / lowest low -- the practical form of support & resistance.

    Uses `shift(1)` so the level excludes the current bar; otherwise "price broke
    the 20-day high" would be trivially true on the bar that set it.
    """
    return pd.DataFrame(
        {
            "highest": high.rolling(length, min_periods=length).max().shift(1),
            "lowest": low.rolling(length, min_periods=length).min().shift(1),
        },
        index=high.index,
    )


def fibonacci_retracement(
    high: pd.Series, low: pd.Series, length: int = 60
) -> pd.DataFrame:
    """Fib retracement levels across the last `length` bars' swing.

    Levels are drawn from the swing low up to the swing high, so `level_0` is the
    high and `level_100` the low -- matching how a chart tool draws an uptrend.
    """
    swing_high = high.rolling(length, min_periods=length).max().shift(1)
    swing_low = low.rolling(length, min_periods=length).min().shift(1)
    span = swing_high - swing_low

    out = {"swing_high": swing_high, "swing_low": swing_low}
    for ratio in (0.236, 0.382, 0.5, 0.618, 0.786):
        out[f"level_{int(ratio * 1000)}"] = swing_high - ratio * span
    return pd.DataFrame(out, index=high.index)
