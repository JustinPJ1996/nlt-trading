"""Momentum oscillators -- RSI, MACD, Stochastic, CCI, Williams %R, ROC, MFI."""

from __future__ import annotations

import pandas as pd

from nlt.indicators.smoothing import ema, rma, sma


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index -- Pine `ta.rsi`, Wilder-smoothed.

    Returns 100 where there are no losses in the window (Pine's division by zero
    resolves that way), rather than NaN.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)

    rs = avg_gain / avg_loss
    out = 100.0 - (100.0 / (1.0 + rs))
    out[avg_loss == 0] = 100.0
    out[(avg_gain == 0) & (avg_loss == 0)] = 50.0
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD line, signal line and histogram -- Pine `ta.macd`."""
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame(
        {"macd": line, "signal": sig, "histogram": line - sig}, index=close.index
    )


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 14,
    smooth_k: int = 1,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """Stochastic %K / %D. TradingView's built-in defaults are 14, 1, 3."""
    highest = high.rolling(length, min_periods=length).max()
    lowest = low.rolling(length, min_periods=length).min()
    span = highest - lowest

    raw_k = 100.0 * (close - lowest) / span
    # A perfectly flat window has no range; Pine yields 50 rather than blowing up.
    raw_k[span == 0] = 50.0

    k = sma(raw_k, smooth_k) if smooth_k > 1 else raw_k
    return pd.DataFrame({"k": k, "d": sma(k, smooth_d)}, index=close.index)


def stoch_rsi(
    close: pd.Series,
    rsi_length: int = 14,
    stoch_length: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3,
) -> pd.DataFrame:
    """Stochastic RSI -- the stochastic formula applied to the RSI series."""
    r = rsi(close, rsi_length)
    highest = r.rolling(stoch_length, min_periods=stoch_length).max()
    lowest = r.rolling(stoch_length, min_periods=stoch_length).min()
    span = highest - lowest

    raw = 100.0 * (r - lowest) / span
    raw[span == 0] = 50.0

    k = sma(raw, smooth_k)
    return pd.DataFrame({"k": k, "d": sma(k, smooth_d)}, index=close.index)


def cci(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20
) -> pd.Series:
    """Commodity Channel Index -- uses mean absolute deviation, not stdev."""
    tp = (high + low + close) / 3.0
    ma = sma(tp, length)
    mad = tp.rolling(length, min_periods=length).apply(
        lambda w: pd.Series(w).sub(pd.Series(w).mean()).abs().mean(), raw=True
    )
    out = (tp - ma) / (0.015 * mad)
    out[mad == 0] = 0.0
    return out


def williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.Series:
    """Williams %R -- ranges from -100 (weakest) to 0 (strongest)."""
    highest = high.rolling(length, min_periods=length).max()
    lowest = low.rolling(length, min_periods=length).min()
    span = highest - lowest
    out = -100.0 * (highest - close) / span
    out[span == 0] = -50.0
    return out


def roc(close: pd.Series, length: int = 9) -> pd.Series:
    """Rate of change, as a percentage -- Pine `ta.roc`."""
    return 100.0 * (close - close.shift(length)) / close.shift(length)


def momentum(close: pd.Series, length: int = 10) -> pd.Series:
    """Raw price momentum -- close minus close `length` bars ago."""
    return close - close.shift(length)


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    length: int = 14,
) -> pd.Series:
    """Money Flow Index -- a volume-weighted RSI. Needs real volume.

    Indices report zero volume from several vendors, which would make this
    constant; the registry refuses to build it when volume is absent.
    """
    tp = (high + low + close) / 3.0
    raw_flow = tp * volume
    up = tp > tp.shift(1)

    pos = raw_flow.where(up, 0.0).rolling(length, min_periods=length).sum()
    neg = raw_flow.where(~up, 0.0).rolling(length, min_periods=length).sum()

    out = 100.0 - (100.0 / (1.0 + pos / neg))
    out[neg == 0] = 100.0
    out[(pos == 0) & (neg == 0)] = 50.0
    return out
