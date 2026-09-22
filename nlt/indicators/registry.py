"""The indicator vocabulary.

This registry is the contract between three parties: it tells the translator what
Claude is allowed to emit, tells the engine how to compute each one, and supplies
the plain-English names the readback shows the user.

Adding an indicator means adding one entry here -- nothing else changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from nlt.indicators import levels, momentum, patterns, trend, volatility
from nlt.indicators import volume as vol
from nlt.indicators.smoothing import ema, hma, sma, wma


@dataclass(frozen=True)
class IndicatorDef:
    """One entry in the vocabulary."""

    name: str
    fn: Callable[..., pd.Series | pd.DataFrame]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    params: dict[str, object] = field(default_factory=dict)
    needs_volume: bool = False
    label: str = ""
    description: str = ""

    @property
    def is_multi(self) -> bool:
        return len(self.outputs) > 1


def _price_fn(fn):
    """Wrap a close-only moving average so it accepts a configurable source."""

    def inner(df: pd.DataFrame, length: int, source: str = "close"):
        return fn(df[source], length)

    return inner


REGISTRY: dict[str, IndicatorDef] = {}


def _add(d: IndicatorDef) -> None:
    REGISTRY[d.name] = d


# ---------------------------------------------------------------- moving averages
for _name, _fn, _label in [
    ("sma", sma, "Simple Moving Average"),
    ("ema", ema, "Exponential Moving Average"),
    ("wma", wma, "Weighted Moving Average"),
    ("hma", hma, "Hull Moving Average"),
]:
    _add(
        IndicatorDef(
            name=_name,
            fn=_price_fn(_fn),
            inputs=("close",),
            outputs=(_name,),
            params={"length": 20, "source": "close"},
            label=_label,
            description=f"{_label} of price over `length` bars.",
        )
    )

# ------------------------------------------------------------------- momentum
_add(
    IndicatorDef(
        name="rsi",
        fn=lambda df, length=14: momentum.rsi(df.close, length),
        inputs=("close",),
        outputs=("rsi",),
        params={"length": 14},
        label="RSI",
        description="Relative Strength Index, 0-100. Below 30 is oversold, above 70 overbought.",
    )
)
_add(
    IndicatorDef(
        name="macd",
        fn=lambda df, fast=12, slow=26, signal=9: momentum.macd(df.close, fast, slow, signal),
        inputs=("close",),
        outputs=("macd", "signal", "histogram"),
        params={"fast": 12, "slow": 26, "signal": 9},
        label="MACD",
        description="Moving Average Convergence Divergence. Outputs: macd, signal, histogram.",
    )
)
_add(
    IndicatorDef(
        name="stochastic",
        fn=lambda df, length=14, smooth_k=1, smooth_d=3: momentum.stochastic(
            df.high, df.low, df.close, length, smooth_k, smooth_d
        ),
        inputs=("high", "low", "close"),
        outputs=("k", "d"),
        params={"length": 14, "smooth_k": 1, "smooth_d": 3},
        label="Stochastic",
        description="Stochastic oscillator, 0-100. Outputs: k, d.",
    )
)
_add(
    IndicatorDef(
        name="stoch_rsi",
        fn=lambda df, rsi_length=14, stoch_length=14, smooth_k=3, smooth_d=3: momentum.stoch_rsi(
            df.close, rsi_length, stoch_length, smooth_k, smooth_d
        ),
        inputs=("close",),
        outputs=("k", "d"),
        params={"rsi_length": 14, "stoch_length": 14, "smooth_k": 3, "smooth_d": 3},
        label="Stochastic RSI",
        description="Stochastic applied to RSI, 0-100. Outputs: k, d.",
    )
)
_add(
    IndicatorDef(
        name="cci",
        fn=lambda df, length=20: momentum.cci(df.high, df.low, df.close, length),
        inputs=("high", "low", "close"),
        outputs=("cci",),
        params={"length": 20},
        label="CCI",
        description="Commodity Channel Index. Typically oscillates between -100 and +100.",
    )
)
_add(
    IndicatorDef(
        name="williams_r",
        fn=lambda df, length=14: momentum.williams_r(df.high, df.low, df.close, length),
        inputs=("high", "low", "close"),
        outputs=("williams_r",),
        params={"length": 14},
        label="Williams %R",
        description="Williams %R, -100 to 0. Below -80 oversold, above -20 overbought.",
    )
)
_add(
    IndicatorDef(
        name="roc",
        fn=lambda df, length=9: momentum.roc(df.close, length),
        inputs=("close",),
        outputs=("roc",),
        params={"length": 9},
        label="Rate of Change",
        description="Percent change in price over `length` bars.",
    )
)
_add(
    IndicatorDef(
        name="momentum",
        fn=lambda df, length=10: momentum.momentum(df.close, length),
        inputs=("close",),
        outputs=("momentum",),
        params={"length": 10},
        label="Momentum",
        description="Price minus the price `length` bars ago, in points.",
    )
)
_add(
    IndicatorDef(
        name="mfi",
        fn=lambda df, length=14: momentum.mfi(df.high, df.low, df.close, df.volume, length),
        inputs=("high", "low", "close", "volume"),
        outputs=("mfi",),
        params={"length": 14},
        needs_volume=True,
        label="Money Flow Index",
        description="Volume-weighted RSI, 0-100.",
    )
)

# ------------------------------------------------------------------ volatility
_add(
    IndicatorDef(
        name="atr",
        fn=lambda df, length=14: volatility.atr(df.high, df.low, df.close, length),
        inputs=("high", "low", "close"),
        outputs=("atr",),
        params={"length": 14},
        label="ATR",
        description="Average True Range in points -- a measure of typical bar movement.",
    )
)
_add(
    IndicatorDef(
        name="bollinger",
        fn=lambda df, length=20, mult=2.0: volatility.bollinger(df.close, length, mult),
        inputs=("close",),
        outputs=("middle", "upper", "lower", "percent_b", "bandwidth"),
        params={"length": 20, "mult": 2.0},
        label="Bollinger Bands",
        description="Outputs: middle, upper, lower, percent_b (0=lower band, 1=upper), bandwidth.",
    )
)
_add(
    IndicatorDef(
        name="keltner",
        fn=lambda df, length=20, mult=2.0, atr_length=10: volatility.keltner(
            df.high, df.low, df.close, length, mult, atr_length
        ),
        inputs=("high", "low", "close"),
        outputs=("middle", "upper", "lower"),
        params={"length": 20, "mult": 2.0, "atr_length": 10},
        label="Keltner Channels",
        description="ATR-based channel. Outputs: middle, upper, lower.",
    )
)
_add(
    IndicatorDef(
        name="supertrend",
        fn=lambda df, length=10, mult=3.0: volatility.supertrend(
            df.high, df.low, df.close, length, mult
        ),
        inputs=("high", "low", "close"),
        outputs=("supertrend", "direction"),
        params={"length": 10, "mult": 3.0},
        label="Supertrend",
        description="Trend-following stop line. direction is +1 in an uptrend, -1 in a downtrend.",
    )
)
_add(
    IndicatorDef(
        name="historical_volatility",
        fn=lambda df, length=20: volatility.historical_volatility(df.close, length),
        inputs=("close",),
        outputs=("historical_volatility",),
        params={"length": 20},
        label="Historical Volatility",
        description="Annualised volatility of returns, in percent.",
    )
)

# ----------------------------------------------------------------------- trend
_add(
    IndicatorDef(
        name="adx",
        fn=lambda df, length=14: trend.adx(df.high, df.low, df.close, length),
        inputs=("high", "low", "close"),
        outputs=("adx", "plus_di", "minus_di"),
        params={"length": 14},
        label="ADX",
        description="Trend strength. Above 25 means a strong trend. Outputs: adx, plus_di, minus_di.",
    )
)
_add(
    IndicatorDef(
        name="parabolic_sar",
        fn=lambda df, start=0.02, increment=0.02, maximum=0.2: trend.parabolic_sar(
            df.high, df.low, start, increment, maximum
        ),
        inputs=("high", "low"),
        outputs=("sar", "direction"),
        params={"start": 0.02, "increment": 0.02, "maximum": 0.2},
        label="Parabolic SAR",
        description="Trailing stop-and-reverse level. Outputs: sar, direction (+1/-1).",
    )
)
_add(
    IndicatorDef(
        name="ichimoku",
        fn=lambda df, conversion=9, base=26, span_b=52: trend.ichimoku(
            df.high, df.low, df.close, conversion, base, span_b
        ),
        inputs=("high", "low", "close"),
        outputs=("conversion", "base", "span_a", "span_b", "lagging"),
        params={"conversion": 9, "base": 26, "span_b": 52},
        label="Ichimoku Cloud",
        description="Outputs: conversion, base, span_a, span_b, lagging.",
    )
)

# ---------------------------------------------------------------------- volume
_add(
    IndicatorDef(
        name="obv",
        fn=lambda df: vol.obv(df.close, df.volume),
        inputs=("close", "volume"),
        outputs=("obv",),
        needs_volume=True,
        label="On Balance Volume",
        description="Cumulative volume signed by price direction.",
    )
)
_add(
    IndicatorDef(
        name="cmf",
        fn=lambda df, length=20: vol.cmf(df.high, df.low, df.close, df.volume, length),
        inputs=("high", "low", "close", "volume"),
        outputs=("cmf",),
        params={"length": 20},
        needs_volume=True,
        label="Chaikin Money Flow",
        description="Buying vs selling pressure, roughly -1 to +1.",
    )
)
_add(
    IndicatorDef(
        name="vwap",
        fn=lambda df: vol.vwap(df.high, df.low, df.close, df.volume),
        inputs=("high", "low", "close", "volume"),
        outputs=("vwap",),
        needs_volume=True,
        label="VWAP",
        description="Volume-weighted average price, reset each session. Intraday only.",
    )
)
_add(
    IndicatorDef(
        name="volume_ratio",
        fn=lambda df, length=20: vol.volume_ratio(df.volume, length),
        inputs=("volume",),
        outputs=("volume_ratio",),
        params={"length": 20},
        needs_volume=True,
        label="Volume vs Average",
        description="Volume as a multiple of its average. 1.5 means 50% above average.",
    )
)

# ---------------------------------------------------------------------- levels
_add(
    IndicatorDef(
        name="pivots",
        fn=lambda df, period="day": levels.pivots(df.high, df.low, df.close, period),
        inputs=("high", "low", "close"),
        outputs=("pivot", "r1", "s1", "r2", "s2", "r3", "s3"),
        params={"period": "day"},
        label="Pivot Points",
        description="Floor-trader pivots. Outputs: pivot, r1-r3 (resistance), s1-s3 (support).",
    )
)
_add(
    IndicatorDef(
        name="cpr",
        fn=lambda df, period="day": levels.central_pivot_range(
            df.high, df.low, df.close, period
        ),
        inputs=("high", "low", "close"),
        outputs=("pivot", "tc", "bc", "width_pct"),
        params={"period": "day"},
        label="Central Pivot Range",
        description="Outputs: pivot, tc (top), bc (bottom), width_pct. Narrow width suggests a breakout.",
    )
)
_add(
    IndicatorDef(
        name="prior_period",
        fn=lambda df, period="day": levels.prior_period(df.high, df.low, df.close, period),
        inputs=("high", "low", "close"),
        outputs=("high", "low", "close"),
        params={"period": "day"},
        label="Previous Period Levels",
        description="Previous day's/week's high, low and close. Outputs: high, low, close.",
    )
)
_add(
    IndicatorDef(
        name="rolling_extremes",
        fn=lambda df, length=20: levels.rolling_extremes(df.high, df.low, length),
        inputs=("high", "low"),
        outputs=("highest", "lowest"),
        params={"length": 20},
        label="Recent High/Low",
        description="Highest high and lowest low of the last `length` bars, excluding the current bar.",
    )
)
_add(
    IndicatorDef(
        name="fibonacci",
        fn=lambda df, length=60: levels.fibonacci_retracement(df.high, df.low, length),
        inputs=("high", "low"),
        outputs=(
            "swing_high",
            "swing_low",
            "level_236",
            "level_382",
            "level_500",
            "level_618",
            "level_786",
        ),
        params={"length": 60},
        label="Fibonacci Retracement",
        description="Retracement levels over the last `length` bars' swing.",
    )
)

# -------------------------------------------------------------------- patterns
for _pname, _pfn in patterns.PATTERNS.items():
    _add(
        IndicatorDef(
            name=_pname,
            fn=(lambda f: lambda df: f(df))(_pfn),
            inputs=("open", "high", "low", "close"),
            outputs=(_pname,),
            label=_pname.replace("_", " ").title(),
            description=f"True on the bar completing a {_pname.replace('_', ' ')} candlestick pattern.",
        )
    )


def get(name: str) -> IndicatorDef:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown indicator {name!r}. Available: {', '.join(sorted(REGISTRY))}"
        ) from None


def vocabulary() -> str:
    """A compact listing of the whole vocabulary, for the translator prompt."""
    lines = []
    for name in sorted(REGISTRY):
        d = REGISTRY[name]
        params = ", ".join(f"{k}={v!r}" for k, v in d.params.items()) or "no parameters"
        outs = ", ".join(d.outputs)
        lines.append(f"- {name}({params}) -> [{outs}] : {d.description}")
    return "\n".join(lines)
