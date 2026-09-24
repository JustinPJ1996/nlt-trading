"""Intraday bars for NSE instruments, via Yahoo, cached like `YahooSource`.

Same shape as `nlt/data/yahoo.py` -- a thin wrapper around `yfinance` with a
parquet cache -- but everything about intraday data is a matter of degree more
demanding than daily:

- Yahoo only keeps a short backward window per interval (`SUPPORTED`), and it
  is much shorter than the multi-decade daily history. Ask for more than that
  and you get either silently truncated data or a confusing empty frame from
  yfinance; this module raises instead, naming the real limit.
- A daily cache goes stale over trading *days*. An intraday cache goes stale
  over trading *minutes* during the session, and does not go stale at all
  outside it -- nothing new is coming until the next session opens, no matter
  how many wall-clock hours have piled up overnight or over a weekend.
- Every load is passed through the session calendar (`nlt.data.session`):
  bars outside trading hours are dropped, and a session with fewer bars than
  a full session should have -- almost always today's session still in
  progress -- is dropped too, so a live-feed refresh never hands the caller a
  day that has not finished happening yet.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from nlt.data.session import (
    NSE_EQUITY,
    Session,
    bars_per_session,
    drop_incomplete_sessions,
    filter_to_session,
    is_session_time,
    session_date,
)
from nlt.data.source import CACHE_DIR, normalise
from nlt.data.yahoo import SYMBOL_MAP

# interval -> maximum number of days of history Yahoo will actually serve.
# (Verified empirically against ^NSEI; yfinance does not expose this itself.)
SUPPORTED = {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730}


class IntradaySource:
    """Downloads intraday bars and caches them to parquet under data/bars/."""

    name = "intraday"

    def __init__(self, cache_dir: Path = CACHE_DIR, session: Session = NSE_EQUITY):
        self.cache_dir = cache_dir
        self.session = session
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Notes from the most recent `bars()` call -- what got dropped and why.
        # `bars()` itself returns a bare DataFrame to match the `BarSource`
        # shape; callers that want the notes (see `nlt.data.load_bars`) read
        # this straight after calling it.
        self.last_notes: list[str] = []

    def _cache_path(self, symbol: str, interval: str) -> Path:
        # "intraday_" prefix keeps this out of YahooSource's "yahoo_*" namespace
        # even though both live in the same data/bars/ directory.
        return self.cache_dir / f"{self.name}_{symbol}_{interval}.parquet"

    def bars(
        self,
        symbol: str,
        interval: str = "15m",
        start: dt.date | None = None,
        end: dt.date | None = None,
        max_age_minutes: int = 15,
    ) -> pd.DataFrame:
        if interval not in SUPPORTED:
            raise ValueError(
                f"IntradaySource supports {sorted(SUPPORTED)}, not {interval!r}."
            )

        max_days = SUPPORTED[interval]
        if start is not None:
            back = (dt.date.today() - start).days
            if back > max_days:
                raise ValueError(
                    f"{interval} is only available for the last {max_days} days; "
                    f"requested start {start} is {back} days back."
                )

        path = self._cache_path(symbol, interval)
        df, notes = self._load_cached(path, symbol, interval, max_age_minutes)
        self.last_notes = notes

        if start is not None:
            df = df[df.index.date >= start]
        if end is not None:
            df = df[df.index.date <= end]
        return df

    def coverage(self, symbol: str, interval: str) -> dict:
        """What we actually have cached: extent, session count, gaps.

        `sessions_with_missing_bars` lists every trading date in the cache
        whose bar count falls short of a full session -- almost always the
        most recent (still in progress) session, but a real data gap would
        show up here too, indistinguishably (see `nlt.data.session`).
        """
        path = self._cache_path(symbol, interval)
        if not path.exists():
            return {"cached": False, "symbol": symbol, "interval": interval}

        df = pd.read_parquet(path)
        if df.empty:
            return {
                "cached": True,
                "symbol": symbol,
                "interval": interval,
                "total_bars": 0,
                "sessions": 0,
                "first_bar": None,
                "last_bar": None,
                "sessions_with_missing_bars": [],
            }

        expected = bars_per_session(self.session, interval)
        dates = pd.Series([session_date(ts, self.session) for ts in df.index])
        counts = dates.value_counts()
        missing = sorted(str(d) for d, c in counts.items() if c < expected)

        return {
            "cached": True,
            "symbol": symbol,
            "interval": interval,
            "total_bars": int(len(df)),
            "sessions": int(counts.shape[0]),
            "bars_per_session_expected": expected,
            "first_bar": df.index[0],
            "last_bar": df.index[-1],
            "sessions_with_missing_bars": missing,
        }

    # -- internals ----------------------------------------------------------

    def _load_cached(
        self, path: Path, symbol: str, interval: str, max_age_minutes: int
    ) -> tuple[pd.DataFrame, list[str]]:
        notes: list[str] = []
        cached = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        now = pd.Timestamp.now(tz=self.session.tz)

        if not self._is_stale(cached, now, max_age_minutes, interval):
            return self._clean(cached, interval, notes)

        try:
            fresh = self._download(symbol, interval)
        except Exception:
            # An offline box should still be able to backtest on what it has.
            if cached.empty:
                raise
            return self._clean(cached, interval, notes)

        merged = self._merge(cached, fresh)
        merged.to_parquet(path)
        return self._clean(merged, interval, notes)

    def _is_stale(
        self, df: pd.DataFrame, now: pd.Timestamp, max_age_minutes: int, interval: str
    ) -> bool:
        """Minute-granularity staleness, aware of whether the market is open.

        During the session, a cache older than `max_age_minutes` is stale --
        a 15-minute cache checked at minute 20 needs a refresh. Outside the
        session, elapsed wall-clock time means nothing (no new bar is coming
        until the next open), so staleness there is instead "does the cache
        already fully cover the most recently completed session" -- checked
        two ways: is the cache missing that session's date entirely, and (just
        as important) does the cache have *some* bars for that date but fewer
        than a full session, which happens whenever the last refresh landed
        mid-session and nothing has topped it up since. Either case triggers
        one refresh; neither re-downloads purely because it is now Sunday and
        the cache is two days old.

        This uses `session.weekdays` to find the last completed session and,
        like the rest of this module, cannot see exchange holidays it has no
        data for -- see `nlt.data.session` for that trade-off.
        """
        if df.empty:
            return True

        last_ts = df.index[-1]
        if last_ts.tz is None:
            last_ts = last_ts.tz_localize(self.session.tz)
        else:
            last_ts = last_ts.tz_convert(self.session.tz)

        if is_session_time(self.session, now):
            age_minutes = (now - last_ts).total_seconds() / 60.0
            return age_minutes > max_age_minutes

        last_session = session_date(last_ts, self.session)
        latest_closed = self._latest_closed_session(now)
        if last_session < latest_closed:
            return True
        if last_session == latest_closed:
            expected = bars_per_session(self.session, interval)
            bar_count = int((pd.Series(df.index).map(
                lambda ts: session_date(ts, self.session)
            ) == last_session).sum())
            return bar_count < expected
        return False

    def _latest_closed_session(self, now: pd.Timestamp) -> dt.date:
        """Most recent trading date whose session has fully closed by `now`."""
        if now.weekday() in self.session.weekdays and now.time() >= self.session.close_time:
            return now.date()
        cur = now
        while True:
            cur = cur - pd.Timedelta(days=1)
            if cur.weekday() in self.session.weekdays:
                return cur.date()

    def _merge(self, cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
        """Union of cached and fresh bars: unique index, sorted, newest wins."""
        if cached.empty:
            return fresh
        if fresh.empty:
            return cached
        combined = pd.concat([cached, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined

    def _clean(
        self, df: pd.DataFrame, interval: str, notes: list[str]
    ) -> tuple[pd.DataFrame, list[str]]:
        if df.empty:
            return df, notes
        df = normalise(df)
        df, session_notes = filter_to_session(df, self.session)
        notes.extend(session_notes)
        df, incomplete_notes = drop_incomplete_sessions(df, self.session, interval)
        notes.extend(incomplete_notes)
        return df, notes

    def _download(self, symbol: str, interval: str) -> pd.DataFrame:
        import yfinance as yf

        ticker = SYMBOL_MAP.get(symbol.upper(), symbol)
        raw = yf.download(
            ticker,
            interval=interval,
            period=f"{SUPPORTED[interval]}d",
            auto_adjust=False,
            progress=False,
        )
        if raw.empty:
            raise RuntimeError(f"no data returned for {symbol} ({ticker}) at {interval}")

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        return normalise(raw)
