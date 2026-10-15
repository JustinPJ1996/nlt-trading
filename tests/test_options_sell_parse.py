"""Phase 2 batch 2: reading sold and multi-leg option positions, and reading them back.

Justin's rules (2026-10-13/14), each tested here:

* A sold option with no bought one protecting it needs a stop loss -- asked,
  with buttons, never invented. A position whose every sold option is covered
  by a bought one needs none, and the readback says why.
* "Stop at N% of the premium" on more than one sold option is asked: each
  option's own price, or the total premium collected.
* "Re-enter if stopped out" without saying how is asked, with two buttons.
* "At 09:20" opens the position at the 09:20 price; with no candle size said,
  1-minute candles, labelled "We worked out".
* "Hold it to expiry" still closes at the square-off time on expiry day.
* The AI reader never sees a leg position.

Every readback line below was read against its sentence by hand.
"""

from __future__ import annotations

import datetime as dt

import pytest

from nlt.spec.models import All, AtTime, IvPercentile, OnExpiryDay
from nlt.translate import llm
from nlt.translate.readback import DEFAULT, SAID, WORKED_OUT, describe_lines
from nlt.translate.rules import parse

STRADDLE_920 = (
    "Sell the NIFTY weekly ATM straddle at 09:20 with a 25% stop on each leg. Square off at "
    "15:15. Backtest the last 1 year."
)
BULL_PUT = (
    "At 09:30, sell the NIFTY weekly put 200 points below spot and buy the put 400 points "
    "below. Exit at 50% of the credit received or at 15:15. Backtest the last 1 year."
)
PUT_TO_EXPIRY = (
    "Every Monday at 09:30, sell the NIFTY weekly put 300 points below spot and hold it to "
    "expiry. Backtest the last 1 year."
)
REENTRY = (
    "Sell the NIFTY weekly ATM straddle at 09:20 with a 25% stop on each leg, and re-enter a "
    "leg once if it gets stopped out. Square off at 15:15. Backtest the last 1 year."
)
CONDOR = (
    "At 09:30, sell a NIFTY weekly iron condor with short strikes 200 points out and wings 100 "
    "points beyond them. Exit at 50% of the credit received or at 15:15. Backtest the last 1 "
    "year."
)
BUTTERFLY = (
    "At 09:20, sell the NIFTY weekly ATM call and put and buy wings 200 points on either side. "
    "Take profit at 40% of the net credit. Square off at 15:15. Backtest the last 1 year."
)
HIGH_IV = (
    "Sell the NIFTY weekly ATM straddle at 09:20 only on days when IV percentile is above 70. "
    "Stop at 30% of the premium collected. Square off at 15:15. Backtest the last 1 year."
)
SENSEX = (
    "Sell the SENSEX weekly ATM straddle at 09:20 with a 30% stop on each leg. Square off at "
    "15:15. Backtest the last 6 months."
)
USER_21 = (
    "Sell the at-the-money straddle at 9:35 on expiry day. Stop at 25% of premium, square-off "
    "at 3:10 PM"
)


def _spec(text, **kw):
    result = parse(text, **kw)
    assert result.spec is not None, result.questions and result.questions[0].text
    return result


def _asks(text, field, **kw):
    result = parse(text, **kw)
    assert result.spec is None, "expected a question, got a spec"
    assert result.questions[0].field == field, result.questions[0].text
    return result.questions[0]


def _lines(result, answers=None) -> list[tuple[str, str]]:
    return [
        (line, label) for _h, line, label in describe_lines(result.spec, result.sources, answers)
    ]


def _legs(spec) -> list[tuple]:
    return [
        (leg.side, leg.option_type, leg.strike_offset, leg.strike_points, leg.stop_pct)
        for leg in spec.instrument.legs
    ]


# ------------------------------------------------------- the acceptance prompts


def test_short_straddle_at_0920_reads_back_line_by_line() -> None:
    result = _spec(STRADDLE_920)
    spec = result.spec
    assert _legs(spec) == [("sell", "CE", 0, None, 25.0), ("sell", "PE", 0, None, 25.0)]
    assert spec.entry == AtTime(time=dt.time(9, 20))
    assert spec.direction == "long"
    assert spec.exit.stop_pct is None and spec.exit.exits_on_legs
    assert result.period.last == 1 and result.period.unit == "years"
    assert _lines(result) == [
        ("NIFTY options", SAID),
        ("Sell the nearest-weekly call, at-the-money", SAID),
        ("Sell the nearest-weekly put, at-the-money", SAID),
        ("1-minute candles", WORKED_OUT),
        ("At 09:20 each day", SAID),
        (
            "Opens at the 09:20 price; the strikes are picked from the index's close on the "
            "candle ending 09:20",
            WORKED_OUT,
        ),
        (
            "Stop loss on each sold option: closed if its price rises above where we sold it "
            "by 25% of its price at the signal",
            SAID,
        ),
        (
            "Selling an option without a bought one to protect it: a sudden jump can blow "
            "straight past the stop",
            WORKED_OUT,
        ),
        ("No new entries after 15:00 (40 minutes before the 15:40 close)", DEFAULT),
        ("Square off at 15:15 if still open", SAID),
        (
            "If still open on expiry day, closed at 15:15 (its price at 15:15:59, never held "
            "to settlement)",
            WORKED_OUT,
        ),
        (
            "If any of the chosen options didn't trade in the candle we'd open in, that "
            "signal is skipped -- never a different strike",
            WORKED_OUT,
        ),
        ("1 lot of each option per trade (quantity 65 each, today's lot of 65)", DEFAULT),
        ("At most 1 position open at a time", DEFAULT),
        ("Never set aside more than 20% of the account as margin for one position", DEFAULT),
    ]


def test_bull_put_spread_is_hedged_and_needs_no_stop() -> None:
    result = _spec(BULL_PUT)
    spec = result.spec
    assert _legs(spec) == [("sell", "PE", 0, -200, None), ("buy", "PE", 0, -400, None)]
    assert spec.exit.combined_target_pct == 50
    assert spec.schedule.square_off == dt.time(15, 15) and spec.schedule.intraday
    assert spec.entry == AtTime(time=dt.time(9, 30))
    lines = dict(_lines(result))
    assert lines["Sell the nearest-weekly put, strike nearest 200 points below the index"] == SAID
    assert lines["Buy the nearest-weekly put, strike nearest 400 points below the index"] == SAID
    assert (
        lines[
            "Take profit when the position's profit reaches 50% of the credit at the signal prices "
            "(judged on candle closes; closed at the next candle's opening prices)"
        ]
        == SAID
    )
    assert lines["The bought options cap the worst loss, so no stop loss is needed"] == WORKED_OUT
    assert not any("without a bought one" in line for line in lines)
    assert lines["Square off at 15:15 if still open"] == SAID


def test_iron_condor_wings_sit_beyond_the_short_strikes() -> None:
    result = _spec(CONDOR)
    assert _legs(result.spec) == [
        ("sell", "CE", 0, 200, None),
        ("sell", "PE", 0, -200, None),
        ("buy", "CE", 0, 300, None),
        ("buy", "PE", 0, -300, None),
    ]
    assert result.spec.exit.combined_target_pct == 50
    lines = dict(_lines(result))
    # 300 is our sum of the 200 and the 100 the user typed: worked out.
    assert lines["Buy the nearest-weekly call, strike nearest 300 points above the index"] == (
        WORKED_OUT
    )
    assert "The bought options cap the worst loss, so no stop loss is needed" in lines


def test_iron_butterfly_writes_the_atm_shorts_in_points_so_the_hedge_is_seen() -> None:
    result = _spec(BUTTERFLY)
    assert _legs(result.spec) == [
        ("sell", "CE", 0, 0, None),
        ("sell", "PE", 0, 0, None),
        ("buy", "CE", 0, 200, None),
        ("buy", "PE", 0, -200, None),
    ]
    assert result.spec.exit.combined_target_pct == 40
    lines = dict(_lines(result))
    assert "Sell the nearest-weekly call, at-the-money (the strike nearest the index)" in lines
    assert (
        "Take profit when the position's profit reaches 40% of the credit at the signal prices "
        "(judged on candle closes; closed at the next candle's opening prices)"
    ) in lines
    assert "The bought options cap the worst loss, so no stop loss is needed" in lines


def test_named_iron_butterfly_reads_the_same_legs() -> None:
    named = _spec(
        "At 09:20, sell the NIFTY weekly iron butterfly with wings 200 points on either side. "
        "Take profit at 40% of the net credit. Square off at 15:15."
    )
    assert _legs(named.spec) == _legs(_spec(BUTTERFLY).spec)


def test_sensex_straddle() -> None:
    result = _spec(SENSEX)
    assert result.spec.instrument.symbol == "SENSEX"
    assert _legs(result.spec) == [("sell", "CE", 0, None, 30.0), ("sell", "PE", 0, None, 30.0)]
    assert result.period.last == 6 and result.period.unit == "months"
    assert "1 lot of each option per trade (quantity 20 each, today's lot of 20)" in dict(
        _lines(result)
    )


# ------------------------------------------------------------ the stop loss


def test_weekly_put_to_expiry_has_no_stop_so_it_asks_with_buttons() -> None:
    q = _asks(PUT_TO_EXPIRY, "exit.stop_pct")
    assert [c.value for c in q.choices] == ["20", "30", "50"]
    assert q.answerable
    assert "each sold option" in q.text


def test_answering_the_stop_question_reads_the_put_to_expiry() -> None:
    answers = {"exit.stop_pct": "30"}
    result = _spec(PUT_TO_EXPIRY, answers=answers)
    spec = result.spec
    assert _legs(spec) == [("sell", "PE", 0, -300, 30.0)]
    assert spec.schedule.weekdays == [0]
    assert not spec.schedule.intraday
    # 15 minutes before the 15:40 F&O close (Justin's default, 2026-10-15).
    assert spec.schedule.square_off == dt.time(15, 25)
    lines = dict(_lines(result, answers))
    assert lines["At 09:30 every Monday"] == SAID
    assert (
        lines[
            "Stop loss: the sold option is closed if its price rises above where we sold it by "
            "30% of its price at the signal"
        ]
        == SAID
    )
    assert (
        "Otherwise held until expiry day, then closed at 15:25 that day (never left to settle)"
        in lines
    )
    assert not any(line.startswith("Square off at") for line in lines)


def test_a_straddle_without_a_stop_asks() -> None:
    _asks("Sell the NIFTY weekly ATM straddle at 09:20. Square off at 15:15.", "exit.stop_pct")


def test_a_strangle_without_a_stop_asks() -> None:
    _asks(
        "Sell the NIFTY weekly strangle 200 points out at 09:20. Square off at 15:15.",
        "exit.stop_pct",
    )


def test_premium_stop_on_a_straddle_asks_each_or_total() -> None:
    q = _asks(HIGH_IV, "exit.stop_basis")
    assert [(c.label, c.value) for c in q.choices] == [
        ("On each sold option's own price", "each_leg"),
        ("On the total premium collected", "combined"),
    ]


def test_answering_each_leg_puts_the_stop_on_each_sold_option() -> None:
    result = _spec(HIGH_IV, answers={"exit.stop_basis": "each_leg"})
    assert _legs(result.spec) == [("sell", "CE", 0, None, 30.0), ("sell", "PE", 0, None, 30.0)]
    assert result.spec.exit.combined_stop_pct is None


def test_answering_total_puts_the_stop_on_the_whole_position() -> None:
    answers = {"exit.stop_basis": "combined"}
    result = _spec(HIGH_IV, answers=answers)
    assert _legs(result.spec) == [("sell", "CE", 0, None, None), ("sell", "PE", 0, None, None)]
    assert result.spec.exit.combined_stop_pct == 30
    lines = dict(_lines(result, answers))
    assert (
        lines[
            "Stop loss when the position's loss reaches 30% of the credit at the signal prices "
            "(judged with each option at its worst price of the candle; closed at the next "
            "candle's opening prices)"
        ]
        == SAID
    )


def test_words_that_say_total_need_no_question() -> None:
    result = _spec(
        "Sell the NIFTY weekly ATM straddle at 09:20. Stop at 30% of the total premium "
        "collected. Square off at 15:15."
    )
    assert result.spec.exit.combined_stop_pct == 30


def test_a_plain_stop_on_a_straddle_asks_too() -> None:
    _asks(
        "Sell the NIFTY weekly ATM straddle at 09:20, stop 30%. Square off at 15:15.",
        "exit.stop_basis",
    )


def test_a_plain_stop_on_one_sold_option_is_its_price_worked_out() -> None:
    result = _spec("Sell the NIFTY weekly ATM put at 09:20, stop 30%. Square off at 15:15.")
    assert _legs(result.spec) == [("sell", "PE", 0, None, 30.0)]
    assert result.sources["exit.stop"] == WORKED_OUT


def test_a_premium_stop_on_one_sold_option_is_said() -> None:
    result = _spec(
        "Sell the NIFTY weekly ATM put at 09:20, stop 30% of the premium. Square off at 15:15."
    )
    assert result.sources["exit.stop"] == SAID


def test_each_leg_on_a_position_with_bought_options_asks() -> None:
    _asks(
        "At 09:30, sell the NIFTY weekly put 200 points below spot and buy the put 400 points "
        "below, with a 30% stop on each leg. Square off at 15:15.",
        "exit.rule",
    )


def test_each_sold_option_on_a_spread_is_read() -> None:
    result = _spec(
        "At 09:30, sell the NIFTY weekly put 200 points below spot and buy the put 400 points "
        "below, with a 30% stop on each sold option. Square off at 15:15."
    )
    assert _legs(result.spec) == [("sell", "PE", 0, -200, 30.0), ("buy", "PE", 0, -400, None)]


# ------------------------------------------------------------------ re-entry


def test_reentry_without_how_asks_with_two_buttons() -> None:
    q = _asks(REENTRY, "legs.reentry")
    assert [(c.label, c.value) for c in q.choices] == [
        (
            "Re-sell the same option when its price comes back to where we first sold it",
            "same_strike_at_cost",
        ),
        ("Sell a fresh at-the-money option at the next candle", "fresh_strike_now"),
    ]


@pytest.mark.parametrize("kind", ["same_strike_at_cost", "fresh_strike_now"])
def test_answering_reentry_reads_once(kind) -> None:
    answers = {"legs.reentry": kind}
    result = _spec(REENTRY, answers=answers)
    for leg in result.spec.instrument.legs:
        assert (leg.reentry, leg.reentry_max, leg.stop_pct) == (kind, 1, 25.0)
    how = {
        "same_strike_at_cost": "re-sell the same option when its price comes back to where we "
        "first sold it",
        "fresh_strike_now": "sell a fresh at-the-money option at the next candle",
    }[kind]
    line = f"After a sold option's stop loss closes it, {how} -- at most once"
    assert dict(_lines(result, answers))[line] == SAID


def test_reentry_at_cost_is_read_from_the_words() -> None:
    result = _spec(REENTRY.replace("re-enter a leg once", "re-enter a leg once at cost"))
    assert {leg.reentry for leg in result.spec.instrument.legs} == {"same_strike_at_cost"}


def test_reentry_at_the_new_atm_is_read_from_the_words() -> None:
    result = _spec(
        REENTRY.replace(
            "if it gets stopped out", "if it gets stopped out, immediately at the new ATM"
        )
    )
    assert {leg.reentry for leg in result.spec.instrument.legs} == {"fresh_strike_now"}


def test_reentry_without_a_count_asks_how_many() -> None:
    q = _asks(
        REENTRY.replace("re-enter a leg once", "re-enter a leg"),
        "legs.reentry_max",
        answers={"legs.reentry": "same_strike_at_cost"},
    )
    assert [c.value for c in q.choices] == ["1", "2"]


def test_twice_is_two() -> None:
    result = _spec(
        REENTRY.replace("re-enter a leg once", "re-enter a leg twice"),
        answers={"legs.reentry": "same_strike_at_cost"},
    )
    assert {leg.reentry_max for leg in result.spec.instrument.legs} == {2}


# ----------------------------------------------- IV, expiry day, the 70's #21


def test_high_iv_straddle_reads_the_iv_percentile() -> None:
    answers = {"exit.stop_basis": "each_leg"}
    result = _spec(HIGH_IV, answers=answers)
    assert result.spec.entry == All(
        conditions=[AtTime(time=dt.time(9, 20)), IvPercentile(op="gt", value=70)]
    )
    lines = dict(_lines(result, answers))
    assert lines["At 09:20 each day, only when the IV percentile is above 70"] == SAID
    assert (
        lines[
            "IV percentile above 70 (implied volatility of the at-the-money options at the "
            "previous day's close, ranked against the past year)"
        ]
        == WORKED_OUT
    )


def test_user_sentence_21_defaults_nifty_and_asks_each_or_total() -> None:
    _asks(USER_21, "exit.stop_basis")


def test_user_sentence_21_after_the_answer() -> None:
    answers = {"exit.stop_basis": "each_leg"}
    result = _spec(USER_21, answers=answers)
    spec = result.spec
    assert spec.instrument.symbol == "NIFTY"
    assert {leg.expiry for leg in spec.instrument.legs} == {"nearest_weekly"}
    assert _legs(spec) == [("sell", "CE", 0, None, 25.0), ("sell", "PE", 0, None, 25.0)]
    assert spec.entry == All(conditions=[AtTime(time=dt.time(9, 35)), OnExpiryDay()])
    assert spec.schedule.square_off == dt.time(15, 10) and spec.schedule.intraday
    lines = dict(_lines(result, answers))
    assert lines["NIFTY options"] == DEFAULT
    # The expiry was inferred from "on expiry day", not said.
    assert lines["Sell the nearest-weekly call, at-the-money"] == WORKED_OUT
    assert lines["At 09:35 on expiry day"] == SAID
    assert "Square off at 15:10 if still open" in lines


def test_banknifty_on_expiry_day_asks_which_expiry() -> None:
    _asks(
        "Sell the BANKNIFTY ATM straddle at 09:35 on expiry day with a 25% stop on each leg.",
        "instrument.expiry",
    )


# -------------------------------------------------- candles and entry times


def test_0920_on_15_minute_candles_asks() -> None:
    q = _asks(
        STRADDLE_920.replace("at 09:20", "at 09:20 on 15 minute candles"), "instrument.timeframe"
    )
    assert "No 15-minute candle closes at 09:20" in q.text


def test_0920_on_5_minute_candles_is_said() -> None:
    result = _spec(STRADDLE_920.replace("at 09:20", "at 09:20 on 5 minute candles"))
    assert result.spec.instrument.timeframe == "5m"
    assert result.sources["timeframe"] == SAID


def test_an_indicator_entry_without_a_candle_size_asks() -> None:
    _asks(
        "Sell the NIFTY weekly ATM straddle when RSI crosses above 60 with a 25% stop on each "
        "leg. Square off at 15:15.",
        "instrument.timeframe",
    )


def test_no_time_and_no_condition_asks_when_to_open() -> None:
    _asks(
        "Sell the NIFTY weekly ATM straddle with a 25% stop on each leg. Square off at 15:15.",
        "entry.condition",
    )


def test_hold_to_expiry_and_square_off_together_asks() -> None:
    _asks(
        "Sell the NIFTY weekly ATM straddle at 09:20 with a 25% stop on each leg and hold it to "
        "expiry. Square off at 15:15.",
        "schedule.intraday",
    )


# ------------------------------------------------------------ strike words


def test_strangle_points_out_are_either_side() -> None:
    result = _spec(
        "Sell the NIFTY weekly strangle 200 points out at 09:20 with a 25% stop on each leg."
    )
    assert _legs(result.spec) == [("sell", "CE", 0, 200, 25.0), ("sell", "PE", 0, -200, 25.0)]


def test_strangle_strikes_otm() -> None:
    result = _spec(
        "Sell the NIFTY weekly 2 strikes OTM strangle at 09:20 with a 25% stop on each leg."
    )
    assert _legs(result.spec) == [("sell", "CE", 2, None, 25.0), ("sell", "PE", 2, None, 25.0)]


def test_mixed_strike_units_are_refused() -> None:
    _asks(
        "At 09:30, sell the NIFTY weekly put 2 strikes OTM and buy the put 400 points below. "
        "Exit at 50% of the credit received or at 15:15.",
        "instrument.legs",
    )


def test_a_fixed_strike_stays_refused() -> None:
    _asks(
        "Sell the 25000 Nifty Call, hedge it with the 25100. Close at 50% of the max profit",
        "instrument.strike_offset",
    )


def test_a_word_inside_a_leg_is_never_dropped() -> None:
    result = parse("Sell the ITC weekly ATM call at 09:20 with a 25% stop on each leg.")
    assert result.spec is None


def test_a_straddle_with_no_strike_asks() -> None:
    _asks("Sell the NIFTY weekly straddle at 09:20 with a 25% stop on each leg.", "instrument.legs")


# --------------------------------------------------------------- risk, AI


def test_a_position_that_sells_gets_the_20_percent_margin_cap() -> None:
    risk = _spec(STRADDLE_920).spec.risk
    assert risk.max_position_pct == 20
    assert risk.max_lots is None


def test_a_bought_option_keeps_the_10_percent_cap() -> None:
    spec = _spec(
        "Buy NIFTY weekly ATM call on 15 minute bars when RSI crosses above 60. Stop 30%"
    ).spec
    assert spec.risk.max_position_pct == 10
    assert spec.risk.max_lots == 2


def test_the_ai_reader_is_never_asked_about_legs() -> None:
    calls = []

    def transport(system, user):
        calls.append(user)
        raise AssertionError("the model must not be asked")

    text = "Sell the NIFTY weekly ATM straddle at 09:20 with a 25% stop on each leg, blah."
    result = llm.translate(text, transport=transport)
    assert calls == []
    assert result.spec is None and result.source == "rules"
    assert result.questions[0].field == "unparsed"


def test_targets_and_bought_stops_read_back_as_falls_from_the_fill() -> None:
    """Every leg level is said as the engine sets it: from our fill, by a % of the signal price."""
    from nlt.spec.models import StrategySpec
    from nlt.translate.readback import _leg_percent_lines

    spec = StrategySpec.model_validate(
        {
            "name": "x",
            "description": "x",
            "instrument": {
                "symbol": "NIFTY",
                "trade_as": "option",
                "timeframe": "1m",
                "legs": [
                    {"side": "sell", "option_type": "PE", "strike_offset": 1, "target_pct": 40},
                    {"side": "buy", "option_type": "PE", "strike_offset": 3, "stop_pct": 50},
                ],
            },
            "entry": {"kind": "at_time", "time": "09:20"},
            "exit": {"exits_on_legs": True},
        }
    )
    legs = spec.instrument.legs
    stop = _leg_percent_lines(legs, "stop_pct")
    target = _leg_percent_lines(legs, "target_pct")
    assert len(stop) == 1 and stop[0].endswith(
        "closed if its price falls below where we bought it by 50% of its price at the signal"
    )
    assert target == [
        "Take profit: the sold option is closed if its price falls below where we sold it by "
        "40% of its price at the signal"
    ]
