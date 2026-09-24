"""Single entry point for the app: get bars for any timeframe, daily or intraday."""

from __future__ import annotations

import pandas as pd

_DAILY_TIMEFRAMES = {"1d", "1w"}


def load_bars(symbol: str, timeframe: str) -> tuple[pd.DataFrame, list[str]]:
    """Fetch bars for `timeframe`, routing to the daily or intraday source.

    Returns the bar frame plus human-readable notes about anything dropped
    along the way (out-of-hours bars, incomplete sessions) so a caller can
    surface them instead of silently trading on cleaned-up data.
    """
    if timeframe in _DAILY_TIMEFRAMES:
        from nlt.data.yahoo import YahooSource

        return YahooSource().bars(symbol, timeframe), []

    from nlt.data.intraday import SUPPORTED, IntradaySource

    if timeframe not in SUPPORTED:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; expected one of "
            f"{sorted(_DAILY_TIMEFRAMES | set(SUPPORTED))}"
        )

    source = IntradaySource()
    bars = source.bars(symbol, timeframe)
    return bars, list(source.last_notes)
