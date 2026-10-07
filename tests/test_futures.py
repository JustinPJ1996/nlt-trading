"""Futures contracts: lot sizes and expiry dates, held to answer keys.

An expiry date that is wrong by a day either closes a position early, or holds
it across the switch to the next contract -- where the price jumps by the gap
between the two, and the engine would book that jump as a real profit or loss.

The answer key (`fixtures/futures_expiry_key.json`) was read off Kite on
2026-10-07, independently of the rules: the expiries Kite listed that day, and
every day since 2020 on which the open interest of Kite's continuous series
jumped at least threefold -- the first day of the next contract.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from nlt.data import futures as F

KEY = json.loads((Path(__file__).parent / "fixtures" / "futures_expiry_key.json").read_text())
SYMBOLS = sorted(F.FUTURES)

# Indian holidays that move an expiry Kite listed on 2026-10-07. Each exchange
# keeps its own: NSE shuts for Guru Nanak Jayanti, MCX trades through it.
# Weekends are handled by the rule.
_LISTED_HOLIDAYS = {
    "NFO": {dt.date(2026, 11, 24)},
    "MCX": {dt.date(2027, 1, 26)},
}


def _traded_days(symbol: str) -> list[dt.date]:
    k = KEY[symbol]
    closed = {dt.date.fromisoformat(d) for d in k["closed_weekdays"]}
    days = [d for d in pd.bdate_range(k["first"], k["last"]).date if d not in closed]
    days += [dt.date.fromisoformat(d) for d in k["weekend_sessions"]]
    return sorted(days)


def test_the_answer_key_covers_every_supported_contract():
    assert set(SYMBOLS) <= set(KEY)


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_rules_reproduce_every_expiry_kite_listed(symbol):
    for listed in KEY[symbol]["listed_on_2026_10_07"]:
        expiry = dt.date.fromisoformat(listed)
        days = [
            d
            for d in pd.bdate_range(
                expiry - dt.timedelta(days=20), expiry + dt.timedelta(days=5)
            ).date
            if d not in _LISTED_HOLIDAYS[F.contract(symbol).exchange]
        ]
        found = [
            d
            for d in F.expiry_dates(symbol, days)
            if (d.year, d.month) == (expiry.year, expiry.month)
        ]
        assert found == [expiry], f"{symbol}: Kite lists {expiry}, the rule says {found}"


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_every_contract_switch_since_2020_comes_the_day_after_an_expiry(symbol):
    days = _traded_days(symbol)
    expiries = set(F.expiry_dates(symbol, days))
    previous = dict(zip(days[1:], days))
    artifacts = KEY.get("_artifacts", {}).get(symbol, {})
    missed = [
        s
        for s in KEY[symbol]["switches"]
        if s not in artifacts and previous[dt.date.fromisoformat(s)] not in expiries
    ]
    assert not missed, (
        f"{symbol}: the next contract started on {missed} with no expiry the day before"
    )


@pytest.mark.parametrize("symbol", ["GOLD", "GOLDM", "SILVER", "SILVERM"])
def test_every_bullion_expiry_is_followed_by_a_switch(symbol):
    """Bullion open interest collapses before expiry, so every switch is visible.

    The converse of the test above: no expiry the data does not confirm.
    Energy and index futures roll more gradually, so not every one of their
    switches clears the threefold bar, and this check would be noise there.
    """
    days = _traded_days(symbol)
    following = dict(zip(days, days[1:]))
    switches = set(KEY[symbol]["switches"])
    unconfirmed = [
        e
        for e in F.expiry_dates(symbol, days)
        if e in following and str(following[e]) not in switches
    ]
    assert not unconfirmed, f"{symbol}: no switch after the expiries {unconfirmed}"


def test_an_expiry_on_a_closed_day_moves_back_to_the_last_day_the_market_traded():
    # Crude's December 2026 contract: the rule names Friday 18 Dec. If that day
    # had been a holiday, the expiry is Thursday, never the Monday after.
    days = [
        d for d in pd.bdate_range("2026-12-01", "2026-12-31").date if d != dt.date(2026, 12, 18)
    ]
    assert dt.date(2026, 12, 17) in F.expiry_dates("CRUDEOIL", days)


def test_a_published_expiry_replaces_the_rule_for_its_month():
    # The morning of 25 Jan 2027, the price data cannot know 26 Jan is a holiday.
    days = list(pd.bdate_range("2027-01-01", "2027-01-25").date)
    assert dt.date(2027, 1, 25) not in F.expiry_dates("NATURALGAS", days)
    assert dt.date(2027, 1, 25) in F.expiry_dates("NATURALGAS", days, listed=[dt.date(2027, 1, 25)])


# Source: Zerodha Varsity and broker contract pages, checked 2026-10-07; MCX's
# own site refuses automated requests. Kite's instrument list says 1 for every
# MCX contract, which is the order unit, not the size -- see nlt/data/futures.py.
@pytest.mark.parametrize(
    "symbol, multiplier",
    [
        ("CRUDEOIL", 100),  # 100 barrels, quoted per barrel
        ("CRUDEOILM", 10),
        ("NATURALGAS", 1250),  # 1,250 mmBtu, quoted per mmBtu
        ("NATGASMINI", 250),
        ("GOLD", 100),  # 1 kg, quoted per 10 g
        ("GOLDM", 10),  # 100 g, quoted per 10 g
        ("SILVER", 30),  # 30 kg, quoted per kg
        ("SILVERM", 5),
        ("NIFTY", 65),  # NSE circular, January 2026 series
        ("BANKNIFTY", 30),
    ],
)
def test_one_lot_is_the_real_contract_size(symbol, multiplier):
    assert F.contract(symbol).multiplier == multiplier
