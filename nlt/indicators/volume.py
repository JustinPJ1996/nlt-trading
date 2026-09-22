"""Volume indicators -- OBV, CMF, VWAP, Chaikin oscillator.

All of these are meaningless on a series with no volume. Index feeds frequently
report zero volume, so the registry blocks them rather than let a strategy build
on a flat line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nlt.indicators.smoothing import ema, sma


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume -- running total of volume signed by the close's direction."""
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def _money_flow_multiplier(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    span = (high - low).replace(0.0, np.nan)
    return (((close - low) - (high - close)) / span).fillna(0.0)


def accumulation_distribution(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Accumulation/Distribution line."""
    return (_money_flow_multiplier(high, low, close) * volume).cumsum()


def cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 20,
) -> pd.Series:
    """Chaikin Money Flow."""
    mfv = _money_flow_multiplier(high, low, close) * volume
    vol_sum = volume.rolling(length, min_periods=length).sum()
    return mfv.rolling(length, min_periods=length).sum() / vol_sum.replace(0.0, np.nan)


def chaikin_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    fast: int = 3,
    slow: int = 10,
) -> pd.Series:
    """Chaikin Oscillator -- EMA spread of the A/D line."""
    ad = accumulation_distribution(high, low, close, volume)
    return ema(ad, fast) - ema(ad, slow)


def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Session-anchored VWAP -- resets each trading day.

    On daily bars every session is one bar, so this collapses to the typical
    price. It only carries information intraday.
    """
    tp = (high + low + close) / 3.0
    session = close.index.tz_convert("Asia/Kolkata").date

    grouped_pv = (tp * volume).groupby(session).cumsum()
    grouped_v = volume.groupby(session).cumsum()
    return grouped_pv / grouped_v.replace(0.0, np.nan)


def volume_ratio(volume: pd.Series, length: int = 20) -> pd.Series:
    """Volume as a multiple of its own average -- "volume above average"."""
    avg = sma(volume, length)
    return volume / avg.replace(0.0, np.nan)
