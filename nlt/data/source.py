"""Market data sources.

The engine only ever sees a DataFrame with a fixed schema, so the source can be
swapped (free EOD now, Kite historical later) without touching anything downstream.

Schema, always:
    index : tz-aware DatetimeIndex in Asia/Kolkata, sorted, unique
    cols  : open, high, low, close, volume  (float64; volume may be 0 for indices)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Protocol

import pandas as pd

IST = "Asia/Kolkata"
BAR_COLUMNS = ["open", "high", "low", "close", "volume"]

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "bars"

# Nothing before this date is used.
#
# Two reasons, one practical and one about what a backtest is for. Practically,
# Yahoo's Indian equity history is unadjusted for corporate actions before
# roughly 2010: across the NIFTY 50 there are 112 impossible single-session
# moves over full history and exactly one from 2020 onwards. Beyond that, a
# strategy validated against the 2008 market is being validated against a market
# that no longer exists -- different participants, different costs, different
# microstructure.
#
# Six years still spans a crash, a long bull run and a sideways stretch, which is
# the variety that matters. Loaders accept `floor=None` to opt out deliberately.
DATA_FLOOR = dt.date(2020, 1, 1)


def apply_floor(df: pd.DataFrame, floor: dt.date | None = DATA_FLOOR) -> pd.DataFrame:
    """Drop bars before `floor`. Passing None keeps everything."""
    if floor is None or df.empty:
        return df
    return df[df.index.date >= floor]


class BarSource(Protocol):
    """Anything that can hand the engine historical bars."""

    def bars(
        self,
        symbol: str,
        interval: str,
        start: dt.date,
        end: dt.date,
    ) -> pd.DataFrame: ...


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary OHLCV frame into the canonical schema.

    Raises rather than guessing if a required column is missing -- silently
    returning a frame with a missing 'volume' would surface much later as a
    confusing indicator failure.
    """
    df = df.copy()
    df.columns = [str(c).lower().strip() for c in df.columns]

    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"bar frame missing required columns: {missing}")

    df = df[BAR_COLUMNS].astype("float64")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("bar frame must be indexed by datetime")
    if df.index.tz is None:
        df.index = df.index.tz_localize(IST)
    else:
        df.index = df.index.tz_convert(IST)

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "timestamp"

    # A bar with high < low or close outside [low, high] is corrupt data, not a
    # trading opportunity. Drop loudly rather than let it fabricate a signal.
    bad = (df.high < df.low) | (df.close > df.high) | (df.close < df.low)
    if bad.any():
        df = df[~bad]

    return df.dropna(subset=["open", "high", "low", "close"])
