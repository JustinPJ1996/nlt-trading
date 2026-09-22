"""Volatility and channel indicators -- ATR, Bollinger, Keltner, Supertrend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nlt.indicators.smoothing import ema, rma, sma, stdev, true_range


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range -- Pine `ta.atr`, which is rma(tr, length)."""
    return rma(true_range(high, low, close), length)


def bollinger(close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands, plus %B and bandwidth.

    %B places price within the bands (0 = lower, 1 = upper), which is what
    "price broke the lower band" should actually test against.
    """
    basis = sma(close, length)
    dev = mult * stdev(close, length)
    upper, lower = basis + dev, basis - dev
    span = upper - lower

    return pd.DataFrame(
        {
            "middle": basis,
            "upper": upper,
            "lower": lower,
            "percent_b": (close - lower) / span.replace(0.0, np.nan),
            "bandwidth": 100.0 * span / basis.replace(0.0, np.nan),
        },
        index=close.index,
    )


def keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
    mult: float = 2.0,
    atr_length: int = 10,
) -> pd.DataFrame:
    """Keltner Channels -- EMA basis with ATR-scaled bands."""
    basis = ema(close, length)
    band = mult * atr(high, low, close, atr_length)
    return pd.DataFrame(
        {"middle": basis, "upper": basis + band, "lower": basis - band},
        index=close.index,
    )


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 10,
    mult: float = 3.0,
) -> pd.DataFrame:
    """Supertrend line and direction (+1 uptrend, -1 downtrend).

    The band-ratchet logic matters: bands only tighten toward price while the
    trend holds, and reset on a flip. Recomputing bands from scratch each bar --
    the common shortcut -- produces far more flips than any chart shows.
    """
    hl2 = (high + low) / 2.0
    band = mult * atr(high, low, close, length)

    upper_basic = (hl2 + band).to_numpy(dtype="float64")
    lower_basic = (hl2 - band).to_numpy(dtype="float64")
    closes = close.to_numpy(dtype="float64")

    n = len(closes)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    trend = np.full(n, np.nan)

    start = int(np.argmax(~np.isnan(upper_basic))) if not np.isnan(upper_basic).all() else n
    if start >= n:
        return pd.DataFrame(
            {"supertrend": trend, "direction": direction}, index=close.index
        )

    upper[start], lower[start] = upper_basic[start], lower_basic[start]
    direction[start] = 1.0
    trend[start] = lower[start]

    for i in range(start + 1, n):
        upper[i] = (
            min(upper_basic[i], upper[i - 1])
            if closes[i - 1] <= upper[i - 1]
            else upper_basic[i]
        )
        lower[i] = (
            max(lower_basic[i], lower[i - 1])
            if closes[i - 1] >= lower[i - 1]
            else lower_basic[i]
        )

        if closes[i] > upper[i - 1]:
            direction[i] = 1.0
        elif closes[i] < lower[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]

        trend[i] = lower[i] if direction[i] > 0 else upper[i]

    return pd.DataFrame(
        {"supertrend": trend, "direction": direction}, index=close.index
    )


def historical_volatility(close: pd.Series, length: int = 20, periods: int = 252) -> pd.Series:
    """Annualised volatility of log returns, in percent."""
    log_ret = np.log(close / close.shift(1))
    return 100.0 * stdev(log_ret, length) * np.sqrt(periods)
