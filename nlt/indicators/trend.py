"""Trend indicators -- ADX/DI, Parabolic SAR, Ichimoku."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nlt.indicators.smoothing import rma, true_range


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.DataFrame:
    """ADX with +DI and -DI -- Wilder's original, as Pine implements it.

    ADX measures trend *strength* regardless of direction; +DI/-DI carry the
    direction. "Only trade when ADX > 25" is the usual Varsity framing.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    atr_ = rma(true_range(high, low, close), length)
    plus_di = 100.0 * rma(plus_dm, length) / atr_
    minus_di = 100.0 * rma(minus_dm, length) / atr_

    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan)

    return pd.DataFrame(
        {"adx": rma(dx, length), "plus_di": plus_di, "minus_di": minus_di},
        index=high.index,
    )


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    start: float = 0.02,
    increment: float = 0.02,
    maximum: float = 0.2,
) -> pd.DataFrame:
    """Parabolic SAR with its trend direction (+1 long, -1 short)."""
    highs = high.to_numpy(dtype="float64")
    lows = low.to_numpy(dtype="float64")
    n = len(highs)

    sar = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    if n < 2:
        return pd.DataFrame({"sar": sar, "direction": direction}, index=high.index)

    # Seed from the first two bars' direction.
    long = highs[1] >= highs[0]
    af = start
    ep = highs[1] if long else lows[1]
    sar[1] = lows[0] if long else highs[0]
    direction[1] = 1.0 if long else -1.0

    for i in range(2, n):
        prev = sar[i - 1]
        cur = prev + af * (ep - prev)

        if long:
            # SAR may never move above the last two lows.
            cur = min(cur, lows[i - 1], lows[i - 2])
            if lows[i] < cur:
                long, cur, ep, af = False, ep, lows[i], start
            elif highs[i] > ep:
                ep, af = highs[i], min(af + increment, maximum)
        else:
            cur = max(cur, highs[i - 1], highs[i - 2])
            if highs[i] > cur:
                long, cur, ep, af = True, ep, highs[i], start
            elif lows[i] < ep:
                ep, af = lows[i], min(af + increment, maximum)

        sar[i] = cur
        direction[i] = 1.0 if long else -1.0

    return pd.DataFrame({"sar": sar, "direction": direction}, index=high.index)


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    conversion: int = 9,
    base: int = 26,
    span_b: int = 52,
) -> pd.DataFrame:
    """Ichimoku Cloud.

    Leading spans are shifted forward by `base` bars, as on a chart. That shift
    is future-facing by construction, so the engine only ever reads the cloud
    values that are already visible at the current bar.
    """

    def donchian_mid(length: int) -> pd.Series:
        return (
            high.rolling(length, min_periods=length).max()
            + low.rolling(length, min_periods=length).min()
        ) / 2.0

    tenkan = donchian_mid(conversion)
    kijun = donchian_mid(base)

    return pd.DataFrame(
        {
            "conversion": tenkan,
            "base": kijun,
            "span_a": ((tenkan + kijun) / 2.0).shift(base),
            "span_b": donchian_mid(span_b).shift(base),
            "lagging": close.shift(-base),
        },
        index=close.index,
    )
