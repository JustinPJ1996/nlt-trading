"""IntradaySource tests -- cache round-trip, staleness, merging, all offline.

Only `test_real_15m_nifty_has_a_full_session` touches the network, and it is
marked `network` and skipped by default (see tests/conftest.py).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from nlt.data.intraday import SUPPORTED, IntradaySource
from nlt.data.session import NSE_EQUITY

IST = "Asia/Kolkata"


@pytest.fixture
def source(tmp_path) -> IntradaySource:
    return IntradaySource(cache_dir=tmp_path)


def _session_bars(date: dt.date, start: str, end: str, freq_minutes: int = 15) -> pd.DataFrame:
    idx = pd.date_range(
        pd.Timestamp(f"{date} {start}", tz=IST),
        pd.Timestamp(f"{date} {end}", tz=IST),
        freq=f"{freq_minutes}min",
    )
    n = len(idx)
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1000.0] * n,
        },
        index=idx,
    )


def _full_day(date: dt.date) -> pd.DataFrame:
    return _session_bars(date, "09:15", "15:15")


def _prior_trading_day(date: dt.date, days_back: int) -> dt.date:
    """`date` minus `days_back` trading (Mon-Fri) days -- keeps fixtures off weekends."""
    d = date
    while days_back:
        d -= dt.timedelta(days=1)
        if d.weekday() < 5:
            days_back -= 1
    return d


# ---------------------------------------------------------------------------
# 1. cache round-trip, no network
# ---------------------------------------------------------------------------


def test_cache_round_trip_no_download_when_fresh(source, monkeypatch):
    today = dt.date.today()
    path = source._cache_path("NIFTY", "15m")
    _full_day(today).to_parquet(path)

    now = pd.Timestamp(f"{today} 15:20", tz=IST)  # inside session, just after last bar

    def explode(*a, **k):
        raise AssertionError("re-downloaded a cache that was still fresh")

    monkeypatch.setattr(source, "_download", explode)
    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: now))

    result = source.bars("NIFTY", "15m", max_age_minutes=15)
    assert len(result) == 25


# ---------------------------------------------------------------------------
# 2. refresh appends without duplicating
# ---------------------------------------------------------------------------


def test_refresh_appends_without_duplicating(source, monkeypatch):
    today = dt.date.today()
    path = source._cache_path("NIFTY", "15m")
    seed = _full_day(today).iloc[:10]  # first 10 bars only, stale on purpose
    seed.to_parquet(path)

    # "fresh" overlaps the last 3 cached bars and adds new ones beyond them.
    fresh = _full_day(today).iloc[7:]

    monkeypatch.setattr(source, "_download", lambda symbol, interval: fresh)
    now = pd.Timestamp(f"{today} 20:00", tz=IST)  # well after close -> stale vs full session
    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: now))

    result = source.bars("NIFTY", "15m", max_age_minutes=15)

    assert result.index.is_unique
    assert result.index.is_monotonic_increasing
    assert len(result) == 25  # union of [0:10) and [7:25) == full day
    # overlapping bars keep the fresh values (open at index 7 should be 107.0)
    overlap_ts = _full_day(today).index[7]
    assert result.loc[overlap_ts, "open"] == 107.0


# ---------------------------------------------------------------------------
# 3. staleness during market hours vs outside them
# ---------------------------------------------------------------------------


def test_stale_during_market_hours_when_older_than_max_age(source, monkeypatch):
    today = dt.date.today()
    path = source._cache_path("NIFTY", "15m")
    _session_bars(today, "09:15", "09:45").to_parquet(path)  # last bar 09:45

    now = pd.Timestamp(f"{today} 10:15", tz=IST)  # 30 min after last bar, in session
    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: now))

    calls = []

    def fake_download(symbol, interval):
        calls.append(1)
        return _session_bars(today, "09:15", "10:15")

    monkeypatch.setattr(source, "_download", fake_download)
    source.bars("NIFTY", "15m", max_age_minutes=15)
    assert calls, "cache older than max_age_minutes during market hours must refresh"


def test_not_stale_outside_market_hours_even_if_old(source, monkeypatch):
    # Cache already covers today's full session; asking well after close
    # (or on a weekend) must not trigger a re-download.
    today = dt.date.today()
    path = source._cache_path("NIFTY", "15m")
    _full_day(today).to_parquet(path)

    now = pd.Timestamp(f"{today} 22:00", tz=IST)  # long after close, same day
    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: now))

    def explode(*a, **k):
        raise AssertionError("re-downloaded outside market hours with a cache already current")

    monkeypatch.setattr(source, "_download", explode)
    result = source.bars("NIFTY", "15m", max_age_minutes=15)
    assert len(result) == 25


# ---------------------------------------------------------------------------
# 4. interval beyond its available window raises, naming the limit
# ---------------------------------------------------------------------------


def test_requesting_1m_beyond_7_days_raises_naming_the_limit(source):
    start = dt.date.today() - dt.timedelta(days=30)
    with pytest.raises(ValueError, match="7 days"):
        source.bars("NIFTY", "1m", start=start)


# ---------------------------------------------------------------------------
# 5. download failure falls back to cache
# ---------------------------------------------------------------------------


def test_download_failure_falls_back_to_cache(source, monkeypatch):
    stale_day = _prior_trading_day(dt.date.today(), 5)
    path = source._cache_path("NIFTY", "15m")
    _full_day(stale_day).to_parquet(path)

    def offline(*a, **k):
        raise ConnectionError("no network")

    monkeypatch.setattr(source, "_download", offline)
    result = source.bars("NIFTY", "15m", max_age_minutes=15)
    assert len(result) == 25


def test_download_failure_with_no_cache_raises(source, monkeypatch):
    def offline(*a, **k):
        raise ConnectionError("no network")

    monkeypatch.setattr(source, "_download", offline)
    with pytest.raises(ConnectionError):
        source.bars("NIFTY", "15m")


# ---------------------------------------------------------------------------
# 6. coverage() reports sessions with missing bars
# ---------------------------------------------------------------------------


def test_coverage_reports_missing_bars_and_uncached_symbol(source):
    assert source.coverage("NIFTY", "15m") == {
        "cached": False,
        "symbol": "NIFTY",
        "interval": "15m",
    }

    full_day = _prior_trading_day(dt.date.today(), 2)
    short_day = _prior_trading_day(dt.date.today(), 1)
    combined = pd.concat(
        [_full_day(full_day), _session_bars(short_day, "09:15", "10:45")]
    )
    combined.to_parquet(source._cache_path("NIFTY", "15m"))

    cov = source.coverage("NIFTY", "15m")
    assert cov["cached"] is True
    assert cov["sessions"] == 2
    assert cov["bars_per_session_expected"] == 25
    assert str(short_day) in cov["sessions_with_missing_bars"]
    assert str(full_day) not in cov["sessions_with_missing_bars"]


# ---------------------------------------------------------------------------
# 7. session filtering applied on load
# ---------------------------------------------------------------------------


def test_out_of_hours_cached_bar_does_not_survive_load(source, monkeypatch):
    today = dt.date.today()
    full = _full_day(today)
    out_of_hours = _session_bars(today, "16:00", "16:00")  # single bar, after close
    combined = pd.concat([full, out_of_hours])
    combined.to_parquet(source._cache_path("NIFTY", "15m"))

    now = pd.Timestamp(f"{today} 22:00", tz=IST)
    monkeypatch.setattr(pd.Timestamp, "now", staticmethod(lambda tz=None: now))
    monkeypatch.setattr(
        source, "_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download"))
    )

    result = source.bars("NIFTY", "15m")
    assert pd.Timestamp(f"{today} 16:00", tz=IST) not in result.index
    assert len(result) == 25
    assert "16:00" in " ".join(source.last_notes) or any(
        "outside" in n for n in source.last_notes
    )


# ---------------------------------------------------------------------------
# SUPPORTED sanity
# ---------------------------------------------------------------------------


def test_supported_matches_documented_windows():
    assert SUPPORTED == {"1m": 7, "5m": 60, "15m": 60, "30m": 60, "1h": 730}


def test_unsupported_interval_rejected(source):
    with pytest.raises(ValueError):
        source.bars("NIFTY", "3m")


# ---------------------------------------------------------------------------
# 8. real network test, skipped by default
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_real_15m_nifty_has_a_full_session():
    from nlt.data.session import session_date

    src = IntradaySource()
    df = src.bars("NIFTY", "15m")
    assert not df.empty

    by_day = pd.Series([session_date(ts, NSE_EQUITY) for ts in df.index]).value_counts()
    assert by_day.max() == 25
