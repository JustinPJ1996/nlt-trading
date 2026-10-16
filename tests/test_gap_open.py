"""A gap at the open (Tradl #26), 2026-10-16: today's open against the previous
session's close, and "still below the open". `levels.session_open`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nlt.indicators import levels
from nlt.translate.readback import describe
from nlt.translate.rules import parse
from tests.test_conditions_a6 import _session_bars


def test_the_gap_is_todays_open_against_yesterdays_last_close() -> None:
    bars = _session_bars(days=3, minutes=15)
    out = levels.session_open(bars.open, bars.close)
    days = sorted(set(bars.index.date))
    d1_last = bars.close[bars.index.date == days[0]].iloc[-1]
    d2 = bars[bars.index.date == days[1]]
    assert (out.loc[d2.index, "open"] == d2.open.iloc[0]).all()
    assert out.loc[d2.index, "gap_pct"].iloc[5] == pytest.approx(
        100 * (d2.open.iloc[0] / d1_last - 1)
    )
    assert out.loc[bars.index.date == days[0], "gap_pct"].isna().all()


@pytest.mark.parametrize("cut", [30, 40, 60])
def test_nothing_later_in_the_day_changes_an_earlier_candle(cut) -> None:
    bars = _session_bars(days=3, minutes=15)
    full = levels.session_open(bars.open, bars.close)
    part = levels.session_open(bars.open.iloc[:cut], bars.close.iloc[:cut])
    pd.testing.assert_frame_equal(full.iloc[:cut], part)


@pytest.mark.parametrize(
    ("words", "line"),
    [
        (
            "the day opens more than 0.5% below the previous close",
            "the day opened more than 0.5% below the previous day's close",
        ),
        (
            "the day opens 0.5% below the previous close",
            "the day opened 0.5% or more below the previous day's close",
        ),
        (
            "it gaps up by more than 1%",
            "the day opened more than 1% above the previous day's close",
        ),
        ("it gaps down 1%", "the day opened 1% or more below the previous day's close"),
        ("the price is still below the open", "the price is below today's open"),
        ("price is above today's open", "the price is above today's open"),
    ],
)
def test_the_words_read_and_read_back(words, line) -> None:
    spec = parse(f"buy nifty on 5 minute candles when {words}, stop 1%, target 1%").spec
    assert spec is not None
    assert f"  When {line}" in describe(spec).splitlines()


def test_the_open_alone_is_not_assumed_to_be_the_days() -> None:
    """ "Close above the open" may mean the candle's own open; not guessed."""
    assert parse("buy nifty on 5 minute candles when close is above the open, stop 1%").spec is None


def test_gaps_up_is_not_read_as_a_target() -> None:
    spec = parse("buy reliance on 15 minute candles when it gaps up 1%, stop 2%").spec
    assert spec is not None and spec.exit.target_pct is None


def test_a_gap_compared_the_wrong_way_is_not_called_a_gap() -> None:
    """ "gap below +0.3%" is not a gap down: worded as the plain comparison."""
    from nlt.spec.models import Compare, Const, IndicatorSpec, Ref
    from nlt.translate.readback import _describe_condition

    inds = {"g": IndicatorSpec(id="g", type="session_open")}
    cond = Compare(op="lt", left=Ref(name="g", output="gap_pct"), right=Const(value=0.3))
    assert "opened" not in _describe_condition(cond, inds, "5m")
