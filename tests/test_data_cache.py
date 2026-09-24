"""Cache freshness.

The cache was originally write-once: downloaded on first use and returned
unchanged forever after. Over a nineteen-year backtest that is invisible. On a
paper or live runner it is silent failure -- the system acts on stale prices and
reports nothing unusual, because as far as it knows the data is simply what the
data is.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from nlt.data.yahoo import YahooSource


@pytest.fixture
def source(tmp_path) -> YahooSource:
    return YahooSource(cache_dir=tmp_path)


def _frame(last: dt.date, rows: int = 5) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp(last), periods=rows, tz="Asia/Kolkata")
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
        index=idx,
    )


def test_trading_days_ignores_weekends():
    """A Friday bar looked at on Monday is one trading day old, not three."""
    friday, monday = dt.date(2026, 9, 18), dt.date(2026, 9, 21)
    assert YahooSource._trading_days_behind(friday, monday) == 1


def test_trading_days_behind_is_zero_for_today():
    today = dt.date(2026, 9, 24)
    assert YahooSource._trading_days_behind(today, today) == 0


def test_same_day_cache_is_reused_without_downloading(source, monkeypatch):
    path = source._cache_path("NIFTY", "1d")
    _frame(dt.date.today()).to_parquet(path)

    def explode(*a, **k):
        raise AssertionError("re-downloaded a cache that was still current")

    monkeypatch.setattr(source, "_download", explode)
    assert len(source.bars("NIFTY")) == 5


def test_stale_cache_triggers_a_refresh(source, monkeypatch):
    path = source._cache_path("NIFTY", "1d")
    _frame(dt.date.today() - dt.timedelta(days=30)).to_parquet(path)

    calls = []

    def fake_download(symbol, interval):
        calls.append(symbol)
        return _frame(dt.date.today(), rows=9)

    monkeypatch.setattr(source, "_download", fake_download)
    result = source.bars("NIFTY")

    assert calls == ["NIFTY"], "a month-old cache must be refreshed"
    assert len(result) == 9


def test_download_failure_falls_back_to_stale_data(source, monkeypatch):
    """An offline machine should still backtest, not crash."""
    path = source._cache_path("NIFTY", "1d")
    _frame(dt.date.today() - dt.timedelta(days=30)).to_parquet(path)

    def offline(*a, **k):
        raise ConnectionError("no network")

    monkeypatch.setattr(source, "_download", offline)
    assert len(source.bars("NIFTY")) == 5


def test_staleness_is_reportable(source):
    """Callers must be able to ask how old the data is before trusting it."""
    assert source.staleness_days("NIFTY") is None

    _frame(dt.date.today() - dt.timedelta(days=14)).to_parquet(
        source._cache_path("NIFTY", "1d")
    )
    assert source.staleness_days("NIFTY") >= 9
