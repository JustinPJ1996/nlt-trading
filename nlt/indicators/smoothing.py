"""Moving-average primitives, matching TradingView's Pine `ta.*` semantics.

The seeding rules here are the whole point. TradingView seeds EMA and RMA with a
simple average of the first `length` values, then recurses. pandas' `ewm` seeds
from the first observation instead, which leaves a visible offset for hundreds of
bars on short series -- the kind of discrepancy that makes a user's backtest
disagree with the chart they were looking at when they wrote the strategy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple moving average -- Pine `ta.sma`."""
    _check(length)
    return series.rolling(length, min_periods=length).mean()


def _seeded_recursive(series: pd.Series, length: int, alpha: float) -> pd.Series:
    """Shared body of EMA/RMA: SMA seed at bar `length-1`, then recurse."""
    values = series.to_numpy(dtype="float64")
    out = np.full(values.shape, np.nan)

    if len(values) < length:
        return pd.Series(out, index=series.index)

    seed_slice = values[:length]
    if np.isnan(seed_slice).any():
        # Leading NaNs (e.g. an indicator of an indicator) -- start from the
        # first window that is fully populated.
        first_valid = series.first_valid_index()
        if first_valid is None:
            return pd.Series(out, index=series.index)
        start = series.index.get_loc(first_valid)
        if start + length > len(values):
            return pd.Series(out, index=series.index)
    else:
        start = 0

    seed_end = start + length - 1
    out[seed_end] = values[start : seed_end + 1].mean()
    for i in range(seed_end + 1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]

    return pd.Series(out, index=series.index)


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential moving average -- Pine `ta.ema`, alpha = 2/(length+1)."""
    _check(length)
    return _seeded_recursive(series, length, 2.0 / (length + 1.0))


def rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's smoothing -- Pine `ta.rma`, alpha = 1/length.

    Used by RSI, ATR and ADX. Substituting a plain EMA here is the single most
    common way a homegrown RSI ends up disagreeing with every chart package.
    """
    _check(length)
    return _seeded_recursive(series, length, 1.0 / length)


def wma(series: pd.Series, length: int) -> pd.Series:
    """Linearly weighted moving average -- Pine `ta.wma`."""
    _check(length)
    weights = np.arange(1, length + 1, dtype="float64")
    weights /= weights.sum()
    return series.rolling(length, min_periods=length).apply(
        lambda w: float(np.dot(w, weights)), raw=True
    )


def hma(series: pd.Series, length: int) -> pd.Series:
    """Hull moving average -- wma(2*wma(n/2) - wma(n), sqrt(n))."""
    _check(length)
    half = max(1, int(length / 2))
    root = max(1, int(np.sqrt(length)))
    return wma(2.0 * wma(series, half) - wma(series, length), root)


def stdev(series: pd.Series, length: int) -> pd.Series:
    """Rolling standard deviation -- Pine `ta.stdev` uses the population form."""
    _check(length)
    return series.rolling(length, min_periods=length).std(ddof=0)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """max(h-l, |h-prev_close|, |l-prev_close|); first bar falls back to h-l."""
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _check(length: int) -> None:
    if not isinstance(length, (int, np.integer)) or length < 1:
        raise ValueError(f"length must be a positive integer, got {length!r}")
