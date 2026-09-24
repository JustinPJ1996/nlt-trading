"""Multi-symbol stock loading: cache, batching, failure collection, alignment.

Mirrors the monkeypatching style of tests/test_data_cache.py -- `_download_one`
and `_download_many` are the seams, same as `YahooSource._download`, so no
test here touches the network except the one marked `network`.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from nlt.data.stocks import StockSource


@pytest.fixture
def source(tmp_path) -> StockSource:
    return StockSource(cache_dir=tmp_path)


def _frame(last: dt.date, rows: int = 5, start_price: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range(end=pd.Timestamp(last), periods=rows, tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": start_price,
            "high": start_price + 1,
            "low": start_price - 1,
            "close": start_price + 0.5,
            "volume": 1000.0,
        },
        index=idx,
    )


# --------------------------------------------------------------- single-symbol cache


def test_same_day_cache_is_reused_without_downloading(source, monkeypatch):
    path = source._cache_path("RELIANCE", "1d")
    _frame(dt.date.today()).to_parquet(path)

    def explode(*a, **k):
        raise AssertionError("re-downloaded a cache that was still current")

    monkeypatch.setattr(source, "_download_one", explode)
    assert len(source.bars("RELIANCE")) == 5


def test_stale_cache_triggers_a_refresh(source, monkeypatch):
    path = source._cache_path("RELIANCE", "1d")
    _frame(dt.date.today() - dt.timedelta(days=30)).to_parquet(path)

    monkeypatch.setattr(
        source, "_download_one", lambda symbol, interval: _frame(dt.date.today(), rows=9)
    )
    result = source.bars("RELIANCE")
    assert len(result) == 9


def test_offline_fallback_returns_cached_data(source, monkeypatch):
    """An offline machine should still backtest, not crash."""
    path = source._cache_path("RELIANCE", "1d")
    _frame(dt.date.today() - dt.timedelta(days=30)).to_parquet(path)

    def offline(*a, **k):
        raise ConnectionError("no network")

    monkeypatch.setattr(source, "_download_one", offline)
    assert len(source.bars("RELIANCE")) == 5


def test_cache_filenames_do_not_collide_with_other_sources(source):
    path = source._cache_path("RELIANCE", "1d")
    assert path.name == "stock_RELIANCE_1d.parquet"
    assert "yahoo" not in path.name
    assert "intraday" not in path.name


# --------------------------------------------------------------- bars_many failures


def test_bars_many_one_failing_symbol_does_not_abort_the_rest(source, monkeypatch):
    def fake_download_many(symbols, interval):
        out = {}
        for symbol in symbols:
            if symbol == "BADSYM":
                source._failures[symbol] = "no data returned for BADSYM (BADSYM.NS)"
                continue
            out[symbol] = _frame(dt.date.today(), rows=5)
        return out

    monkeypatch.setattr(source, "_download_many", fake_download_many)
    result = source.bars_many(["RELIANCE", "BADSYM", "TCS"])

    assert set(result) == {"RELIANCE", "TCS"}
    assert "BADSYM" not in result
    failures = source.failures()
    assert "BADSYM" in failures
    assert "BADSYM" in failures["BADSYM"] or failures["BADSYM"]


def test_failures_reset_between_calls(source, monkeypatch):
    def fake_download_many(symbols, interval):
        source._failures["BADSYM"] = "boom"
        return {}

    monkeypatch.setattr(source, "_download_many", fake_download_many)
    source.bars_many(["BADSYM"])
    assert source.failures() == {"BADSYM": "boom"}

    def fake_ok(symbols, interval):
        return {s: _frame(dt.date.today()) for s in symbols}

    monkeypatch.setattr(source, "_download_many", fake_ok)
    source.bars_many(["RELIANCE"])
    assert source.failures() == {}


# --------------------------------------------------------------- panel alignment


def test_panel_aligns_symbols_with_differing_date_ranges(source, monkeypatch):
    # RELIANCE trades Mon-Fri this week; TCS is missing Wednesday (a halt).
    full_days = pd.bdate_range("2026-01-05", periods=5, tz="Asia/Kolkata")
    tcs_days = full_days.delete(2)  # drop Wednesday

    reliance = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
        index=full_days,
    )
    tcs = pd.DataFrame(
        {"open": 200.0, "high": 201.0, "low": 199.0, "close": 200.5, "volume": 2000.0},
        index=tcs_days,
    )

    monkeypatch.setattr(
        source, "_download_many", lambda symbols, interval: {"RELIANCE": reliance, "TCS": tcs}
    )

    panel = source.panel(["RELIANCE", "TCS"])

    wednesday = full_days[2]
    assert pd.isna(panel.loc[wednesday, "TCS"])
    assert panel.loc[wednesday, "RELIANCE"] == 100.5
    # No forward fill: the gap is NaN, not Tuesday's value carried forward.
    assert panel["TCS"].isna().sum() == 1


def test_panel_column_order_matches_requested_order(source, monkeypatch):
    days = pd.bdate_range("2026-01-05", periods=3, tz="Asia/Kolkata")
    frame = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=days
    )
    monkeypatch.setattr(
        source,
        "_download_many",
        lambda symbols, interval: {s: frame for s in symbols},
    )

    panel = source.panel(["TCS", "RELIANCE", "INFY"])
    assert list(panel.columns) == ["TCS", "RELIANCE", "INFY"]


def test_panel_missing_symbol_becomes_all_nan_column_not_dropped(source, monkeypatch):
    days = pd.bdate_range("2026-01-05", periods=3, tz="Asia/Kolkata")
    frame = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=days
    )

    def fake_download_many(symbols, interval):
        source._failures["GHOST"] = "delisted"
        return {"RELIANCE": frame}

    monkeypatch.setattr(source, "_download_many", fake_download_many)
    panel = source.panel(["RELIANCE", "GHOST"])

    assert list(panel.columns) == ["RELIANCE", "GHOST"]
    assert panel["GHOST"].isna().all()


# --------------------------------------------------------------- coverage


def test_coverage_reports_missing_days(source, monkeypatch):
    full_days = pd.bdate_range("2026-01-05", periods=10, tz="Asia/Kolkata")
    gappy_days = full_days.delete([2, 5])

    def fake_download_many(symbols, interval):
        out = {
            "RELIANCE": pd.DataFrame(
                {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
                index=full_days,
            ),
            "TCS": pd.DataFrame(
                {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
                index=gappy_days,
            ),
        }
        return out

    monkeypatch.setattr(source, "_download_many", fake_download_many)
    coverage = source.coverage(["RELIANCE", "TCS"])

    assert coverage.loc["RELIANCE", "bar_count"] == 10
    assert coverage.loc["TCS", "bar_count"] == 8
    assert coverage.loc["TCS", "pct_of_days"] < coverage.loc["RELIANCE", "pct_of_days"]


def test_coverage_symbol_with_no_data_reports_zero(source, monkeypatch):
    def fake_download_many(symbols, interval):
        source._failures["GHOST"] = "delisted"
        return {}

    monkeypatch.setattr(source, "_download_many", fake_download_many)
    coverage = source.coverage(["GHOST"])
    assert coverage.loc["GHOST", "bar_count"] == 0
    assert coverage.loc["GHOST", "pct_of_days"] == 0.0


# --------------------------------------------------------------- schema


def test_canonical_schema_is_enforced(source, monkeypatch):
    path = source._cache_path("RELIANCE", "1d")
    _frame(dt.date.today()).to_parquet(path)

    df = source.bars("RELIANCE")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    for col in df.columns:
        assert df[col].dtype == "float64"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "Asia/Kolkata"


# --------------------------------------------------------------- network


@pytest.mark.network
def test_real_download_of_three_nse_stocks():
    src = StockSource()
    result = src.bars_many(["RELIANCE", "TCS", "HDFCBANK"])

    assert set(result) == {"RELIANCE", "TCS", "HDFCBANK"}
    for symbol, df in result.items():
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert str(df.index.tz) == "Asia/Kolkata"
        assert len(df) > 1000  # years of daily bars
    assert src.failures() == {}
