"""Bought straddles and strangles (2026-10-16), and the any-strike cover rule.

Until then "buy the ATM straddle" was refused ("coming later"). The engine
already handled positions of bought options; only the reader refused them. A
bought position needs no stop -- the most it can lose is what was paid -- and
"each leg" can only mean each bought option.
"""

from __future__ import annotations

import pytest

from nlt.translate.readback import describe
from nlt.translate.rules import parse

STRADDLE = "Buy the NIFTY weekly ATM straddle at 09:20. Square off at 15:15."


def _spec(text: str, **answers):
    result = parse(text, answers=answers)
    assert result.spec is not None, result.questions
    return result.spec


def test_a_bought_straddle_reads_as_two_bought_at_the_money_options() -> None:
    spec = _spec(STRADDLE)
    legs = spec.instrument.legs
    assert [(leg.side, leg.option_type, leg.strike_offset) for leg in legs] == [
        ("buy", "CE", 0),
        ("buy", "PE", 0),
    ]
    assert all(leg.stop_pct is None for leg in legs)
    text = describe(spec)
    assert (
        "the most it can lose is what was paid for them, plus charges, so no stop loss is needed"
        in text
    )
    assert "Selling an option" not in text


def test_a_bought_strangle_takes_a_stop_and_target_on_each_option() -> None:
    spec = _spec(
        "Buy the NIFTY weekly strangle 2 strikes OTM at 09:20, target 50% on each leg, "
        "stop 30% on each leg. Square off at 15:15."
    )
    assert [
        (leg.side, leg.strike_offset, leg.stop_pct, leg.target_pct) for leg in spec.instrument.legs
    ] == [
        ("buy", 2, 30.0, 50.0),
        ("buy", 2, 30.0, 50.0),
    ]
    assert "so no stop loss is needed" not in describe(spec)


def test_a_plain_stop_on_a_bought_straddle_asks_each_or_total_paid() -> None:
    text = "Buy the NIFTY weekly ATM straddle at 09:20, target 40%, stop 25%. Square off at 15:15."
    asked = parse(text)
    assert asked.spec is None
    q = asked.questions[0]
    assert q.field == "exit.stop_basis" and "total premium paid" in q.text
    asked = parse(text, answers={"exit.stop_basis": "combined"})
    assert asked.questions[0].field == "exit.target_basis"
    spec = _spec(text, **{"exit.stop_basis": "combined", "exit.target_basis": "combined"})
    assert (spec.exit.combined_stop_pct, spec.exit.combined_target_pct) == (25.0, 40.0)
    assert "40% of the premium paid at the signal prices" in describe(spec)
    spec = _spec(text, **{"exit.stop_basis": "each_leg", "exit.target_basis": "each_leg"})
    assert [(leg.stop_pct, leg.target_pct) for leg in spec.instrument.legs] == [(25.0, 40.0)] * 2
    assert spec.exit.combined_stop_pct is None


def test_of_the_premium_paid_is_on_the_whole_position() -> None:
    spec = _spec(
        "Buy the NIFTY weekly ATM straddle at 09:20, exit at 40% of the premium paid, stop loss "
        "25% of the premium paid. Square off at 15:15."
    )
    assert (spec.exit.combined_stop_pct, spec.exit.combined_target_pct) == (25.0, 40.0)


def test_of_the_premium_paid_on_a_position_that_sells_is_asked() -> None:
    result = parse(
        "Sell the NIFTY weekly ATM straddle at 09:20, stop loss 25% of the premium paid. "
        "Square off at 15:15."
    )
    assert result.spec is None
    assert "only buys options" in result.questions[0].text


def test_a_bought_option_stop_of_100_percent_or_more_is_asked() -> None:
    result = parse(
        "Buy the NIFTY weekly ATM straddle at 09:20, stop 100% on each leg. Square off at 15:15."
    )
    assert result.spec is None


@pytest.mark.parametrize("shape", ["iron condor", "iron butterfly"])
def test_a_bought_iron_shape_is_not_read_backwards(shape: str) -> None:
    result = parse(f"Buy the NIFTY weekly {shape} at 09:20. Square off at 15:15.")
    assert result.spec is None


# ----------------------------------------------------- the any-strike cover


def test_a_bull_call_spread_needs_no_stop() -> None:
    """Buy the ATM call, sell the one 200 points above: it can lose at most
    what it paid. Asked for a stop until 2026-10-16."""
    spec = _spec(
        "At 09:30, buy the NIFTY weekly ATM call and sell the weekly call 200 points above. "
        "Square off at 15:15."
    )
    assert [(leg.side, leg.stop_pct) for leg in spec.instrument.legs] == [
        ("buy", None),
        ("sell", None),
    ]
    assert "The bought options cap the worst loss, so no stop loss is needed" in describe(spec)


def test_a_naked_sell_still_needs_a_stop() -> None:
    result = parse("Sell the NIFTY weekly ATM straddle at 09:20. Square off at 15:15.")
    assert result.spec is None
    assert result.questions[0].field == "exit.stop_pct"


# ------------------------------------------------- calendars and ratios


def test_a_calendar_reads_one_expiry_per_option_and_still_needs_a_stop() -> None:
    text = (
        "At 09:30, sell this week's NIFTY ATM call and buy next week's ATM call, 30% stop on "
        "each sold option. Hold overnight."
    )
    spec = _spec(text)
    assert [(leg.side, leg.expiry, leg.stop_pct) for leg in spec.instrument.legs] == [
        ("sell", "nearest_weekly", 30.0),
        ("buy", "next_weekly", None),
    ]
    assert "every option is closed on the expiry day of whichever expires first" in describe(spec)
    asked = parse(text.replace(", 30% stop on each sold option", ""))
    assert asked.spec is None and asked.questions[0].field == "exit.stop_pct"


def test_two_expiries_without_saying_which_option_is_which_is_asked() -> None:
    result = parse(
        "Sell the NIFTY ATM call and buy the ATM call, weekly and monthly, 30% stop on each "
        "sold option at 09:30. Square off at 15:15."
    )
    assert result.spec is None
    assert "say the expiry of each one" in result.questions[0].text


@pytest.mark.parametrize(
    ("counts", "lots", "sets"),
    [(("1", "2"), [1, 2], 1), (("2", "4"), [1, 2], 2), (("2", "2"), [1, 1], 2)],
)
def test_counts_become_a_ratio_and_a_size(counts, lots, sets) -> None:
    a, b = counts
    spec = _spec(
        f"At 09:30, buy {a} NIFTY weekly ATM calls and sell {b} weekly calls 2 strikes OTM with "
        "a 30% stop on each sold option. Square off at 15:15."
    )
    assert [leg.lots for leg in spec.instrument.legs] == lots
    assert spec.sizing.lots == sets


def test_a_ratio_with_a_size_said_twice_is_asked() -> None:
    result = parse(
        "At 09:30, buy 1 NIFTY weekly ATM call and sell 2 weekly calls 2 strikes OTM with a 30% "
        "stop on each sold option, 3 lots. Square off at 15:15."
    )
    assert result.spec is None and "size twice" in result.questions[0].text


def test_a_count_on_only_some_options_is_asked() -> None:
    result = parse(
        "At 09:30, buy the NIFTY weekly ATM call and sell 2 weekly calls 2 strikes OTM with a "
        "30% stop on each sold option. Square off at 15:15."
    )
    assert result.spec is None and "how many of each" in result.questions[0].text


def test_a_ratio_leaves_its_extra_sold_lots_needing_a_stop() -> None:
    result = parse(
        "At 09:30, buy 1 NIFTY weekly ATM call and sell 2 weekly calls 2 strikes OTM. "
        "Square off at 15:15."
    )
    assert result.spec is None and result.questions[0].field == "exit.stop_pct"


@pytest.mark.parametrize(
    ("named", "legs", "reads"),
    [
        (
            "a debit spread",
            "buy the NIFTY weekly ATM call and sell the weekly call 2 strikes OTM",
            True,
        ),
        # The words do not match the legs: a credit spread called "debit".
        (
            "a debit spread",
            "sell the NIFTY weekly ATM call and buy the weekly call 2 strikes OTM",
            False,
        ),
        (
            "a calendar spread",
            "buy the NIFTY weekly ATM call and sell the weekly call 2 strikes OTM",
            False,
        ),
        (
            "a ratio spread",
            "buy the NIFTY weekly ATM call and sell the weekly call 2 strikes OTM",
            False,
        ),
        (
            "as a hedge",
            "buy the NIFTY weekly ATM call and sell the weekly call 2 strikes OTM",
            False,
        ),
    ],
)
def test_a_named_shape_must_be_the_shape_the_legs_make(named, legs, reads) -> None:
    result = parse(f"At 09:30 {legs}, {named}, 30% stop on each sold option. Square off at 15:15.")
    assert (result.spec is not None) == reads, result.questions


def test_an_expiry_word_outside_the_options_is_not_dropped() -> None:
    """Both options say weekly, and "monthly" appears too: which is meant is asked."""
    result = parse(
        "At 09:30, sell the NIFTY weekly ATM call and buy the weekly call 2 strikes OTM, the "
        "monthly one, 30% stop on each sold option. Square off at 15:15."
    )
    assert result.spec is None
