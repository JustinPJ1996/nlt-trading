"""Daily bars for many NSE stocks at once, via Yahoo, cached like `YahooSource`.

`YahooSource` (nlt/data/yahoo.py) knows two symbols: NIFTY and BANKNIFTY. The
stock strategies in the sample set ("Buy Nifty 100 stocks above the 200 DMA
when RSI drops below 40...") need dozens to hundreds of NSE equities, loaded
together and screened as a basket. This module is that: the same cache and
staleness pattern as `YahooSource`, but batched, and with three additions a
single-symbol source does not need --

- `bars_many` must not let one bad symbol (delisted, renamed, mistyped) abort
  the rest of the basket. Failures are collected, not raised.
- `panel` aligns many symbols onto one trading-day index for screening. A
  halted or newly-listed stock leaves real gaps; those gaps stay NaN. Forward-
  filling them would manufacture a price -- and therefore a signal -- for a
  day the stock did not trade, which is exactly the kind of bug that looks
  fine in a backtest and is wrong in a way nobody notices.
- `coverage` reports what was actually loaded per symbol, so a basket that is
  quietly missing 20 names because of a rename or an API hiccup is visible
  before it produces a backtest result that looks normal but is not.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from nlt.data.quality import clean as quality_clean
from nlt.data.source import CACHE_DIR, DATA_FLOOR, apply_floor, normalise

_INTERVAL_MAP = {"1d": "1d", "1w": "1wk"}


def _ticker(symbol: str) -> str:
    """NSE symbol -> Yahoo ticker. RELIANCE -> RELIANCE.NS."""
    symbol = symbol.strip().upper()
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


class StockSource:
    """Downloads and caches daily bars for NSE equities, one parquet per symbol."""

    name = "stock"

    def __init__(
        self, cache_dir: Path = CACHE_DIR, max_age_days: int = 1, clean_artefacts: bool = True
    ):
        self.cache_dir = cache_dir
        self.max_age_days = max_age_days
        # Yahoo's Indian equity history is unadjusted for corporate actions
        # before roughly 2010: 21 of the 50 NIFTY 50 constituents carry at least
        # one fake crash, BAJFINANCE's being -99.1%. Left in, those bars produce
        # confident, profitable, entirely fictional trades. Cleaning is on by
        # default; a caller studying the raw feed can turn it off deliberately.
        self.clean_artefacts = clean_artefacts
        self._quality_notes: dict[str, list[str]] = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Symbol -> reason it could not be loaded, from the most recent
        # `bars_many`/`panel` call. Never raised past that call: a basket
        # strategy should still run on the symbols that did load.
        self._failures: dict[str, str] = {}

    def _cache_path(self, symbol: str, interval: str) -> Path:
        # "stock_" keeps this out of YahooSource's "yahoo_*" and
        # IntradaySource's "intraday_*" namespaces in the shared data/bars/ dir.
        return self.cache_dir / f"{self.name}_{symbol.upper()}_{interval}.parquet"

    @staticmethod
    def _trading_days_behind(last: dt.date, today: dt.date | None = None) -> int:
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

    def _load_cached(self, symbol: str, interval: str) -> pd.DataFrame:
        """Same freshness contract as `YahooSource._load_cached`: re-download
        when stale, but fall back to whatever is cached if that download
        fails, so an offline run still has something to backtest against."""
        path = self._cache_path(symbol, interval)

        if not path.exists():
            df = self._download_one(symbol, interval)
            df.to_parquet(path)
            return df

        df = pd.read_parquet(path)
        if df.empty:
            df = self._download_one(symbol, interval)
            df.to_parquet(path)
            return df

        if self._trading_days_behind(df.index[-1].date()) <= self.max_age_days:
            return df

        try:
            fresh = self._download_one(symbol, interval)
        except Exception:
            return df
        fresh.to_parquet(path)
        return fresh

    def _download_one(self, symbol: str, interval: str) -> pd.DataFrame:
        """Single-symbol download, used for cache misses and refreshes.

        Bulk `bars_many` calls use `_download_many` instead; this stays
        separate (rather than being a loop of one) so a single symbol can be
        refreshed cheaply without re-downloading everything else.
        """
        import yfinance as yf

        raw = yf.download(
            _ticker(symbol),
            interval=_INTERVAL_MAP[interval],
            period="max",
            auto_adjust=False,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"no data returned for {symbol} ({_ticker(symbol)})")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return normalise(raw)

    def _download_many(self, symbols: Sequence[str], interval: str) -> dict[str, pd.DataFrame]:
        """Batch download via yfinance's multi-ticker support.

        One HTTP round trip for the whole basket instead of one per symbol --
        the difference between loading NIFTY 500 in seconds and hammering
        Yahoo with 500 sequential requests. Symbols yfinance could not fill
        (bad ticker, delisted, empty) are simply absent from the result; the
        caller records that as a failure rather than raising.
        """
        import yfinance as yf

        tickers = [_ticker(s) for s in symbols]
        raw = yf.download(
            tickers,
            interval=_INTERVAL_MAP[interval],
            period="max",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
        )

        out: dict[str, pd.DataFrame] = {}
        for symbol, ticker in zip(symbols, tickers, strict=True):
            try:
                if len(tickers) == 1:
                    frame = raw
                else:
                    frame = raw[ticker]
                frame = frame.dropna(how="all")
                if frame.empty:
                    raise RuntimeError(f"no data returned for {symbol} ({ticker})")
                out[symbol] = normalise(frame)
            except Exception as exc:
                self._failures[symbol] = str(exc)
        return out

    def bars(
        self,
        symbol: str,
        interval: str = "1d",
        start: dt.date | None = None,
        end: dt.date | None = None,
        floor: dt.date | None = DATA_FLOOR,
    ) -> pd.DataFrame:
        """One stock's daily bars, canonical schema, parquet-cached."""
        if interval not in _INTERVAL_MAP:
            raise ValueError(f"StockSource supports {sorted(_INTERVAL_MAP)}, not {interval!r}.")

        symbol = symbol.strip().upper()
        df = self._load_cached(symbol, interval)

        if self.clean_artefacts and interval == "1d":
            df, notes = quality_clean(df, symbol)
            if notes:
                self._quality_notes[symbol] = notes

        # The project-wide floor applies unless the caller asks for something
        # later. Asking for something EARLIER is honoured -- opting out is a
        # deliberate act, not an accident.
        df = apply_floor(df, floor)
        if start is not None:
            df = df[df.index.date >= start]
        if end is not None:
            df = df[df.index.date <= end]
        return df

    def bars_many(
        self,
        symbols: Sequence[str],
        interval: str = "1d",
        start: dt.date | None = None,
        end: dt.date | None = None,
        floor: dt.date | None = DATA_FLOOR,
    ) -> dict[str, pd.DataFrame]:
        """Many stocks at once. A symbol that fails is recorded and skipped,
        never allowed to abort the rest of the basket.

        The floor and the artefact cleaning apply here exactly as they do in
        `bars`. They have to: the basket path is the one stock strategies use,
        so a floor that only guarded the single-symbol path would protect the
        case nobody runs.
        """
        if interval not in _INTERVAL_MAP:
            raise ValueError(f"StockSource supports {sorted(_INTERVAL_MAP)}, not {interval!r}.")

        self._failures = {}
        symbols = [s.strip().upper() for s in symbols]

        cached: dict[str, pd.DataFrame] = {}
        to_fetch: list[str] = []
        for symbol in symbols:
            path = self._cache_path(symbol, interval)
            if path.exists():
                df = pd.read_parquet(path)
                if not df.empty and self._trading_days_behind(df.index[-1].date()) <= (
                    self.max_age_days
                ):
                    cached[symbol] = df
                    continue
            to_fetch.append(symbol)

        if to_fetch:
            try:
                fresh = self._download_many(to_fetch, interval)
            except Exception as exc:
                # The whole batch call failed (e.g. offline). Fall back to
                # whatever is cached for each symbol individually rather than
                # losing the entire basket to one network error.
                fresh = {}
                for symbol in to_fetch:
                    path = self._cache_path(symbol, interval)
                    if path.exists():
                        df = pd.read_parquet(path)
                        if not df.empty:
                            fresh[symbol] = df
                            continue
                    self._failures[symbol] = str(exc)
            for symbol, df in fresh.items():
                self._cache_path(symbol, interval).parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(self._cache_path(symbol, interval))
            cached.update(fresh)

        result: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            if symbol not in cached:
                continue
            df = cached[symbol]
            if self.clean_artefacts and interval == "1d":
                df, notes = quality_clean(df, symbol)
                if notes:
                    self._quality_notes[symbol] = notes
            df = apply_floor(df, floor)
            if start is not None:
                df = df[df.index.date >= start]
            if end is not None:
                df = df[df.index.date <= end]
            result[symbol] = df
        return result

    def panel(
        self,
        symbols: Sequence[str],
        field: str = "close",
        start: dt.date | None = None,
        end: dt.date | None = None,
        floor: dt.date | None = DATA_FLOOR,
    ) -> pd.DataFrame:
        """One field across many symbols: rows = timestamps, columns = symbols.

        Columns follow the order `symbols` was given in, not load order or
        alphabetical -- callers screening a fixed universe expect the basket
        to come back in the order they specified it. Rows are the union of
        every symbol's trading days; a day a symbol did not trade (holiday
        specific to that listing, halt, not yet listed) is NaN, not filled.
        """
        if field not in ("open", "high", "low", "close", "volume"):
            raise ValueError(f"unknown field {field!r}")

        frames = self.bars_many(symbols, start=start, end=end, floor=floor)
        columns = {s: df[field] for s, df in frames.items()}
        panel = pd.DataFrame(columns)
        # Reindex to the requested order; missing symbols become all-NaN
        # columns rather than being silently dropped, so a caller iterating
        # `panel.columns` still sees the basket they asked for.
        wanted = [s.strip().upper() for s in symbols]
        panel = panel.reindex(columns=wanted)
        return panel.sort_index()

    def quality_notes(self) -> dict[str, list[str]]:
        """History dropped as corporate-action artefacts, by symbol.

        A basket that quietly lost fifteen years of one constituent's history
        gives an answer that looks entirely normal, so the caller has to be able
        to find out and say so.
        """
        return {k: list(v) for k, v in self._quality_notes.items()}

    def failures(self) -> dict[str, str]:
        """Symbols that could not be loaded on the most recent bulk call, and why."""
        return dict(self._failures)

    def coverage(self, symbols: Sequence[str]) -> pd.DataFrame:
        """Per symbol: first bar, last bar, bar count, % of trading days present.

        "% of trading days present" is against the union of trading days seen
        across the whole basket (what `panel` would align to), not against a
        theoretical exchange calendar -- so a symbol that is fully populated
        relative to its peers reads as 100% even if the basket itself starts
        late for a newly listed name.
        """
        frames = self.bars_many(symbols)
        all_days = sorted({d for df in frames.values() for d in df.index})
        total = len(all_days)

        rows = []
        for symbol in symbols:
            symbol = symbol.strip().upper()
            df = frames.get(symbol)
            if df is None or df.empty:
                rows.append(
                    {
                        "symbol": symbol,
                        "first_bar": pd.NaT,
                        "last_bar": pd.NaT,
                        "bar_count": 0,
                        "pct_of_days": 0.0,
                    }
                )
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "first_bar": df.index[0],
                    "last_bar": df.index[-1],
                    "bar_count": len(df),
                    "pct_of_days": round(100.0 * len(df) / total, 2) if total else 0.0,
                }
            )
        return pd.DataFrame(rows).set_index("symbol")
