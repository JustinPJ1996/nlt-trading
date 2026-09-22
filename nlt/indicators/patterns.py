"""Candlestick patterns from the Varsity technical analysis module.

Each returns a boolean Series: True on the bar that *completes* the pattern.

Two deliberate choices. First, "small" and "long" are defined relative to recent
average range rather than by fixed point values, so the same rule works on NIFTY
at 8,000 and at 25,000. Second, patterns that Varsity defines as requiring a prior
trend (hammer, engulfing, stars) actually check for one -- a hammer in the middle
of a sideways drift is not a hammer, and treating it as one is how pattern
strategies quietly become noise.
"""

from __future__ import annotations

import pandas as pd

from nlt.indicators.smoothing import sma

# Fractions of the candle's total range.
_DOJI_BODY = 0.05
_MARUBOZU_BODY = 0.95
_SMALL_BODY = 0.30
_LONG_SHADOW = 2.0


def _parts(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, lo, c = df.open, df.high, df.low, df.close
    body = (c - o).abs()
    rng = (h - lo).replace(0.0, pd.NA).astype("float64")
    return {
        "open": o,
        "high": h,
        "low": lo,
        "close": c,
        "body": body,
        "range": rng,
        "body_frac": body / rng,
        "upper_shadow": h - pd.concat([o, c], axis=1).max(axis=1),
        "lower_shadow": pd.concat([o, c], axis=1).min(axis=1) - lo,
        "bullish": c > o,
        "bearish": c < o,
    }


def _downtrend(close: pd.Series, length: int = 10) -> pd.Series:
    """Price below its own short average and falling -- a prior decline."""
    ma = sma(close, length)
    return (close < ma) & (ma.diff() < 0)


def _uptrend(close: pd.Series, length: int = 10) -> pd.Series:
    ma = sma(close, length)
    return (close > ma) & (ma.diff() > 0)


def doji(df: pd.DataFrame) -> pd.Series:
    """Open and close nearly equal -- indecision."""
    p = _parts(df)
    return (p["body_frac"] < _DOJI_BODY).fillna(False)


def marubozu_bullish(df: pd.DataFrame) -> pd.Series:
    """A full-bodied up candle with no meaningful shadows."""
    p = _parts(df)
    return ((p["body_frac"] > _MARUBOZU_BODY) & p["bullish"]).fillna(False)


def marubozu_bearish(df: pd.DataFrame) -> pd.Series:
    p = _parts(df)
    return ((p["body_frac"] > _MARUBOZU_BODY) & p["bearish"]).fillna(False)


def spinning_top(df: pd.DataFrame) -> pd.Series:
    """Small body with shadows on both sides."""
    p = _parts(df)
    return (
        (p["body_frac"] < _SMALL_BODY)
        & (p["upper_shadow"] > p["body"])
        & (p["lower_shadow"] > p["body"])
    ).fillna(False)


def hammer(df: pd.DataFrame) -> pd.Series:
    """Small body at the top, long lower shadow, after a decline -- bullish."""
    p = _parts(df)
    return (
        (p["body_frac"] < _SMALL_BODY)
        & (p["lower_shadow"] > _LONG_SHADOW * p["body"])
        & (p["upper_shadow"] < p["body"])
        & _downtrend(df.close)
    ).fillna(False)


def hanging_man(df: pd.DataFrame) -> pd.Series:
    """The same shape as a hammer, but after an advance -- bearish."""
    p = _parts(df)
    return (
        (p["body_frac"] < _SMALL_BODY)
        & (p["lower_shadow"] > _LONG_SHADOW * p["body"])
        & (p["upper_shadow"] < p["body"])
        & _uptrend(df.close)
    ).fillna(False)


def shooting_star(df: pd.DataFrame) -> pd.Series:
    """Small body at the bottom, long upper shadow, after an advance."""
    p = _parts(df)
    return (
        (p["body_frac"] < _SMALL_BODY)
        & (p["upper_shadow"] > _LONG_SHADOW * p["body"])
        & (p["lower_shadow"] < p["body"])
        & _uptrend(df.close)
    ).fillna(False)


def engulfing_bullish(df: pd.DataFrame) -> pd.Series:
    """An up candle whose body swallows the prior down candle's body."""
    p = _parts(df)
    return (
        p["bullish"]
        & p["bearish"].shift(1).fillna(False)
        & (df.close >= df.open.shift(1))
        & (df.open <= df.close.shift(1))
        & (p["body"] > p["body"].shift(1))
        & _downtrend(df.close)
    ).fillna(False)


def engulfing_bearish(df: pd.DataFrame) -> pd.Series:
    p = _parts(df)
    return (
        p["bearish"]
        & p["bullish"].shift(1).fillna(False)
        & (df.open >= df.close.shift(1))
        & (df.close <= df.open.shift(1))
        & (p["body"] > p["body"].shift(1))
        & _uptrend(df.close)
    ).fillna(False)


def harami_bullish(df: pd.DataFrame) -> pd.Series:
    """A small up candle contained within the prior large down candle."""
    p = _parts(df)
    return (
        p["bullish"]
        & p["bearish"].shift(1).fillna(False)
        & (df.close <= df.open.shift(1))
        & (df.open >= df.close.shift(1))
        & (p["body"] < p["body"].shift(1))
        & _downtrend(df.close)
    ).fillna(False)


def harami_bearish(df: pd.DataFrame) -> pd.Series:
    p = _parts(df)
    return (
        p["bearish"]
        & p["bullish"].shift(1).fillna(False)
        & (df.open <= df.close.shift(1))
        & (df.close >= df.open.shift(1))
        & (p["body"] < p["body"].shift(1))
        & _uptrend(df.close)
    ).fillna(False)


def morning_star(df: pd.DataFrame) -> pd.Series:
    """Three bars: a long decline, a small indecisive bar, then a strong recovery."""
    p = _parts(df)
    avg_body = sma(p["body"], 14)

    first_long_down = p["bearish"].shift(2) & (p["body"].shift(2) > avg_body.shift(2))
    middle_small = p["body_frac"].shift(1) < _SMALL_BODY
    last_strong_up = p["bullish"] & (df.close > (df.open.shift(2) + df.close.shift(2)) / 2.0)

    return (first_long_down & middle_small & last_strong_up).fillna(False)


def evening_star(df: pd.DataFrame) -> pd.Series:
    """The bearish mirror of the morning star."""
    p = _parts(df)
    avg_body = sma(p["body"], 14)

    first_long_up = p["bullish"].shift(2) & (p["body"].shift(2) > avg_body.shift(2))
    middle_small = p["body_frac"].shift(1) < _SMALL_BODY
    last_strong_down = p["bearish"] & (df.close < (df.open.shift(2) + df.close.shift(2)) / 2.0)

    return (first_long_up & middle_small & last_strong_down).fillna(False)


PATTERNS = {
    "doji": doji,
    "marubozu_bullish": marubozu_bullish,
    "marubozu_bearish": marubozu_bearish,
    "spinning_top": spinning_top,
    "hammer": hammer,
    "hanging_man": hanging_man,
    "shooting_star": shooting_star,
    "engulfing_bullish": engulfing_bullish,
    "engulfing_bearish": engulfing_bearish,
    "harami_bullish": harami_bullish,
    "harami_bearish": harami_bearish,
    "morning_star": morning_star,
    "evening_star": evening_star,
}
