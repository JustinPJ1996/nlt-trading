"""Free NIFTY history via Yahoo, used until Kite Connect is subscribed.

Daily bars only, and good enough to build and validate the engine. Kite replaces
this for intraday and for anything that will trade real money.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from nlt.data.source import CACHE_DIR, DATA_FLOOR, apply_floor, normalise

# Yahoo tickers for the instruments we care about in v1.
SYMBOL_MAP = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
}

_INTERVAL_MAP = {"1d": "1d", "1w": "1wk"}


class YahooSource:
    """Downloads daily bars and caches them to parquet under data/bars/."""

    name = "yahoo"

    def __init__(self, cache_dir: Path = CACHE_DIR, max_age_days: int = 1):
        self.cache_dir = cache_dir
        self.max_age_days = max_age_days
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / f"{self.name}_{symbol}_{interval}.parquet"


    def _load_cached(self, path: Path, symbol: str, interval: str) -> pd.DataFrame:
        """Return cached bars, re-downloading when they have gone stale.

        The cache used to be permanent: written once, returned forever. That is
        harmless for a backtest over two decades and dangerous everywhere else --
        a paper or live runner would act on last week's prices and never say so.

        Staleness is measured in trading days rather than calendar days, so a
        weekend or a market holiday does not trigger a pointless re-download.
        """
        if not path.exists():
            df = self._download(symbol, interval)
            df.to_parquet(path)
            return df

        df = pd.read_parquet(path)
        if df.empty:
            df = self._download(symbol, interval)
            df.to_parquet(path)
            return df

        if self._trading_days_behind(df.index[-1].date()) <= self.max_age_days:
            return df

        try:
            fresh = self._download(symbol, interval)
        except Exception:
            # An offline box should still be able to backtest. Return what we
            # have and let `staleness_days` tell the caller how old it is.
            return df
        fresh.to_parquet(path)
        return fresh

    @staticmethod
    def _trading_days_behind(last: dt.date, today: dt.date | None = None) -> int:
        """Weekdays between `last` and today. Ignores exchange holidays."""
        today = today or dt.date.today()
        if last >= today:
            return 0
        days = pd.bdate_range(last, today)
        return max(len(days) - 1, 0)

    def staleness_days(self, symbol: str, interval: str = "1d") -> int | None:
        """Trading days between the newest cached bar and today, or None if uncached."""
        path = self._cache_path(symbol, interval)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if df.empty:
            return None
        return self._trading_days_behind(df.index[-1].date())

    def bars(
        self,
        symbol: str,
        interval: str = "1d",
        start: dt.date | None = None,
        end: dt.date | None = None,
        max_age_days: int = 1,
        floor: dt.date | None = DATA_FLOOR,
    ) -> pd.DataFrame:
        if interval not in _INTERVAL_MAP:
            raise ValueError(
                f"YahooSource supports {sorted(_INTERVAL_MAP)}, not {interval!r}. "
                "Intraday needs Kite."
            )

        self.max_age_days = max_age_days
        path = self._cache_path(symbol, interval)
        df = self._load_cached(path, symbol, interval)

        df = apply_floor(df, floor)
        if start is not None:
            df = df[df.index.date >= start]
        if end is not None:
            df = df[df.index.date <= end]
        return df

    def _download(self, symbol: str, interval: str) -> pd.DataFrame:
        import yfinance as yf

        ticker = SYMBOL_MAP.get(symbol.upper(), symbol)
        raw = yf.download(
            ticker,
            interval=_INTERVAL_MAP[interval],
            period="max",
            auto_adjust=False,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"no data returned for {symbol} ({ticker})")

        # yfinance returns a MultiIndex column frame when given a ticker list,
        # and sometimes for a single ticker too. Flatten to the price field.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        return normalise(raw)


def refresh(symbol: str = "NIFTY", interval: str = "1d") -> pd.DataFrame:
    """Force a re-download, discarding the cache."""
    src = YahooSource()
    path = src._cache_path(symbol, interval)
    path.unlink(missing_ok=True)
    return src.bars(symbol, interval)
