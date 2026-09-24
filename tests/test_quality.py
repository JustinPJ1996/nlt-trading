"""Tests for corporate-action artefact detection.

The failure being guarded against is not a crash but a plausible lie: an
unadjusted 1:10 bonus looks exactly like a 90% collapse followed by a full
recovery, which a dip-buying strategy reports as its best trade ever.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nlt.data.quality import Artefact, clean, detect_artefacts, usable_from


def _bars(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(closes), freq="D", tz="Asia/Kolkata")
    c = pd.Series(closes, index=idx, dtype="float64")
    return pd.DataFrame(
        {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6},
        index=idx,
    )


def test_ordinary_volatility_is_not_an_artefact():
    """A brutal but real week must survive. False positives cost real history."""
    bars = _bars([100, 92, 85, 78, 88, 95, 100])  # -8%, -7.6%, -8.2% days
    assert detect_artefacts(bars) == []
    assert usable_from(bars) is None


def test_a_two_for_one_split_is_detected():
    bars = _bars([100, 101, 50, 51, 52])
    found = detect_artefacts(bars)
    assert len(found) == 1
    assert found[0].change_pct == pytest.approx(-50.5, abs=0.5)
    assert found[0].ratio == pytest.approx(2.0, abs=0.1)


def test_a_ten_for_one_bonus_is_detected():
    bars = _bars([1000, 1010, 100, 102, 104])
    assert len(detect_artefacts(bars)) == 1
    assert detect_artefacts(bars)[0].ratio == pytest.approx(10.0, abs=0.5)


def test_an_upward_artefact_is_detected():
    """A reverse split scales prices up, and is just as wrong."""
    bars = _bars([100, 101, 500, 505, 510])
    assert len(detect_artefacts(bars)) == 1
    assert detect_artefacts(bars)[0].change_pct > 0


def test_clean_drops_everything_up_to_the_last_artefact():
    bars = _bars([1000, 1010, 100, 102, 50, 51, 52, 53])  # two artefacts
    cleaned, notes = clean(bars, "TESTCO")

    assert len(cleaned) == 3, "only bars after the SECOND artefact are usable"
    assert cleaned.close.tolist() == [51.0, 52.0, 53.0]
    assert notes and "TESTCO" in notes[0]


def test_clean_never_drops_silently():
    bars = _bars([1000, 1010, 100, 102, 104])
    _, notes = clean(bars, "TESTCO")
    assert len(notes) == 1
    assert "dropped" in notes[0]
    assert "2020-01-03" in notes[0] or "corporate action" in notes[0]


def test_clean_is_a_no_op_on_healthy_data():
    bars = _bars([100, 102, 99, 101, 103])
    cleaned, notes = clean(bars, "TESTCO")
    assert cleaned.equals(bars)
    assert notes == []


def test_empty_and_single_bar_inputs_are_safe():
    assert detect_artefacts(_bars([])) == []
    assert detect_artefacts(_bars([100.0])) == []
    cleaned, notes = clean(_bars([]))
    assert cleaned.empty and notes == []


def test_artefact_renders_readably():
    a = Artefact(pd.Timestamp("2005-07-29", tz="Asia/Kolkata"), -99.1, 109.0)
    text = str(a)
    assert "2005-07-29" in text and "-99.1%" in text


def test_threshold_is_respected():
    """A 30% drop is real news; a 40% drop is not, at the default threshold."""
    assert detect_artefacts(_bars([100, 70, 71])) == []
    assert len(detect_artefacts(_bars([100, 58, 59]))) == 1


def test_detects_the_real_bajfinance_artefact():
    """The concrete case this module exists for."""
    pytest.importorskip("pyarrow")
    from nlt.data.source import CACHE_DIR
    from nlt.data.stocks import StockSource

    matches = list(CACHE_DIR.glob("*BAJFINANCE*.parquet"))
    if not matches:
        pytest.skip("BAJFINANCE not cached; run the stock fetch first")

    bars = StockSource(clean_artefacts=False).bars("BAJFINANCE", floor=None)
    found = detect_artefacts(bars)
    assert found, "the 2005-07-29 -99.1% artefact must be detected"

    cleaned, notes = clean(bars, "BAJFINANCE")
    assert len(cleaned) < len(bars)
    assert detect_artefacts(cleaned) == [], "cleaned history must be artefact-free"
    # And enough history must survive to be worth backtesting.
    assert len(cleaned) > 3000


# ---------------------------------------------------------------------------
# Wiring: cleaning must be on by default, and opting out must be deliberate
# ---------------------------------------------------------------------------


def test_stock_source_cleans_by_default(tmp_path, monkeypatch):
    from nlt.data.stocks import StockSource

    dirty = _bars([1000, 1010, 100, 102, 104])
    src = StockSource(cache_dir=tmp_path)
    monkeypatch.setattr(src, "_load_cached", lambda symbol, interval: dirty)

    # [1000, 1010, 100, 102, 104]: the artefact is the drop to 100, so the first
    # trustworthy bar is the one after it -- leaving 102 and 104.
    result = src.bars("TESTCO")
    assert len(result) == 2, "artefact history must be dropped without being asked"
    assert "TESTCO" in src.quality_notes()


def test_stock_source_opt_out_returns_raw_history(tmp_path, monkeypatch):
    """Studying the raw feed must remain possible, but only on purpose."""
    from nlt.data.stocks import StockSource

    dirty = _bars([1000, 1010, 100, 102, 104])
    src = StockSource(cache_dir=tmp_path, clean_artefacts=False)
    monkeypatch.setattr(src, "_load_cached", lambda symbol, interval: dirty)

    assert len(src.bars("TESTCO")) == 5
    assert src.quality_notes() == {}


def test_clean_history_contains_no_remaining_artefacts(tmp_path, monkeypatch):
    """The output of the default path must itself be clean -- the property that
    actually matters, rather than merely that something was dropped."""
    from nlt.data.stocks import StockSource

    dirty = _bars([1000, 1010, 100, 102, 50, 51, 52, 53, 54])
    src = StockSource(cache_dir=tmp_path)
    monkeypatch.setattr(src, "_load_cached", lambda symbol, interval: dirty)

    assert detect_artefacts(src.bars("TESTCO")) == []


# ---------------------------------------------------------------------------
# The 2020 data floor
#
# Across the NIFTY 50 there are 112 impossible single-session moves over full
# history and exactly one from 2020 onwards. The floor is what makes stock
# backtests trustworthy; the artefact guard above is what catches the one that
# still gets through (a 2025 demerger).
# ---------------------------------------------------------------------------


def test_index_bars_start_at_the_floor():
    from nlt.data.source import DATA_FLOOR
    from nlt.data.yahoo import YahooSource

    bars = YahooSource().bars("NIFTY")
    assert bars.index[0].date() >= DATA_FLOOR


def test_floor_can_be_opted_out_of_deliberately():
    from nlt.data.source import DATA_FLOOR
    from nlt.data.yahoo import YahooSource

    full = YahooSource().bars("NIFTY", floor=None)
    assert full.index[0].date() < DATA_FLOOR, "opting out must return real earlier history"
    assert len(full) > len(YahooSource().bars("NIFTY"))


def test_apply_floor_is_inclusive_of_the_floor_date():
    import datetime as dt

    from nlt.data.source import apply_floor

    idx = pd.date_range("2019-12-28", periods=10, freq="D", tz="Asia/Kolkata")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
                      index=idx)
    out = apply_floor(df, dt.date(2020, 1, 1))
    assert out.index[0].date() == dt.date(2020, 1, 1)
    assert len(out) == 6


def test_apply_floor_with_none_is_a_no_op():
    from nlt.data.source import apply_floor

    df = _bars([100, 101, 102])
    assert apply_floor(df, None).equals(df)


def test_nifty50_has_essentially_no_artefacts_after_the_floor():
    """The measurement that justifies the floor, kept as a regression test."""
    from nlt.data.source import CACHE_DIR
    from nlt.data.stocks import StockSource
    from nlt.data.universe import get_universe

    if len(list(CACHE_DIR.glob("stock_*.parquet"))) < 40:
        pytest.skip("NIFTY 50 stock cache not populated")

    # Floor applied, artefact cleaning off: this measures what the floor alone
    # achieves, which is the claim being regression-tested.
    src = StockSource(clean_artefacts=False)
    frames = src.bars_many(get_universe("NIFTY 50").symbols)

    total = sum(len(detect_artefacts(df)) for df in frames.values() if not df.empty)
    assert total <= 3, (
        f"{total} corporate-action artefacts survive the 2020 floor; the floor was "
        "chosen because there was only one"
    )
