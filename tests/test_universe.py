"""NSE index membership: name normalisation, snapshot contents, survivorship."""

from __future__ import annotations

import datetime as dt

import pytest

from nlt.data.universe import (
    Universe,
    get_universe,
    list_universes,
    resolve_symbols,
    survivorship_warning,
)


@pytest.fixture(autouse=True)
def no_live_fetch(monkeypatch):
    """Force every `get_universe` call in this file onto the bundled snapshot.

    Without this, `get_universe` would try NSE first (per its contract) and
    tests would pass or fail depending on whether this box happens to have
    network access at the moment -- exactly the flakiness the task forbids.
    Tests that specifically want to exercise the live path do so separately
    and are marked `network`.
    """

    def fail(name):
        raise RuntimeError("no network in tests")

    monkeypatch.setattr("nlt.data.universe.refresh_from_nse", fail)


# --------------------------------------------------------------- spelling


@pytest.mark.parametrize(
    "spelling",
    [
        "NIFTY 50",
        "nifty50",
        "NIFTY_50",
        "Nifty 50",
        "nifty-50",
        " NIFTY   50 ",
        "nifty_50",
    ],
)
def test_get_universe_accepts_loose_spelling(spelling):
    universe = get_universe(spelling)
    assert universe.name == "NIFTY 50"
    assert universe == get_universe("NIFTY 50")


# --------------------------------------------------------------- membership counts


def test_nifty50_has_50_symbols():
    assert len(get_universe("NIFTY 50").symbols) == 50


def test_nifty100_has_100_symbols():
    assert len(get_universe("NIFTY 100").symbols) == 100


def test_nifty500_has_roughly_500_symbols():
    n = len(get_universe("NIFTY 500").symbols)
    assert 490 <= n <= 510


def test_nifty50_is_subset_of_nifty100():
    n50 = set(get_universe("NIFTY 50").symbols)
    n100 = set(get_universe("NIFTY 100").symbols)
    assert n50 <= n100


def test_nifty100_is_subset_of_nifty500():
    n100 = set(get_universe("NIFTY 100").symbols)
    n500 = set(get_universe("NIFTY 500").symbols)
    assert n100 <= n500


# --------------------------------------------------------------- symbol shape


def test_symbols_look_like_nse_symbols():
    for symbol in get_universe("NIFTY 50").symbols:
        assert symbol == symbol.upper()
        assert not symbol.endswith(".NS")
        assert " " not in symbol
        assert symbol == symbol.strip()


# --------------------------------------------------------------- resolve_symbols


def test_resolve_symbols_from_universe_name():
    symbols = resolve_symbols("NIFTY 50")
    assert len(symbols) == 50
    assert "RELIANCE" in symbols


def test_resolve_symbols_from_bare_symbol():
    assert resolve_symbols("reliance") == ("RELIANCE",)


def test_resolve_symbols_from_list():
    assert resolve_symbols(["tcs", "infy"]) == ("TCS", "INFY")


# --------------------------------------------------------------- errors


def test_unknown_universe_raises_with_available_list():
    with pytest.raises(ValueError, match="NIFTY 50"):
        get_universe("NIFTY 9000")


# --------------------------------------------------------------- survivorship


def test_survivorship_warning_fires_for_an_old_backtest():
    universe = Universe(
        name="NIFTY 50",
        symbols=("RELIANCE",),
        as_of=dt.date(2026, 1, 1),
        source="bundled snapshot",
    )
    warning = survivorship_warning(universe, dt.date(2020, 1, 1))
    assert warning is not None
    assert "survivorship" in warning.lower() or "surviv" in warning.lower()
    assert "2026-01-01" in warning
    assert "2020-01-01" in warning


def test_survivorship_warning_silent_for_a_recent_backtest():
    universe = Universe(
        name="NIFTY 50",
        symbols=("RELIANCE",),
        as_of=dt.date(2026, 1, 1),
        source="bundled snapshot",
    )
    assert survivorship_warning(universe, dt.date(2025, 11, 1)) is None


# --------------------------------------------------------------- snapshot file


def test_bundled_snapshot_parses_and_has_capture_date():
    import json

    from nlt.data.universe import _SNAPSHOT_PATH

    with open(_SNAPSHOT_PATH) as f:
        data = json.load(f)

    assert dt.date.fromisoformat(data["captured"])
    assert "NIFTY 50" in data["universes"]
    assert len(data["universes"]["NIFTY 50"]) == 50


def test_list_universes_matches_snapshot():
    assert set(list_universes()) == {"NIFTY 50", "NIFTY 100", "NIFTY 500"}
