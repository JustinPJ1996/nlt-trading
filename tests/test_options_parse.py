"""Phase 2 batch 1: buying one call or one put.

Everything else an options sentence might describe -- selling, two or more
legs, a fixed strike, a points offset -- must still come back as a question,
never a spec: the 0-wrong property (`test_rules.py`'s 70 sentences,
`test_template_prompts.py`'s 43) is exactly as important for options as for
anything else this parser reads.

The loader-safety tests at the bottom are inherited from the file this
replaces (`test_options_refused.py`): `nlt/options/` and the engine side of
options are a different phase, built by someone else, and `load_for_spec`
must keep refusing an option spec with real data not yet wired in until that
lands. They are kept here, unchanged in spirit, because they are the other
half of the same safety property -- the parser stops a bad strategy being
built, the loader stops one however built from being fed the wrong series.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nlt.engine.loader import load_for_spec
from nlt.spec.models import Instrument, StrategySpec
from nlt.translate.options_words import BANKNIFTY_WEEKLY_LAST_EXPIRY
from nlt.translate.readback import SAID, WORKED_OUT, describe, describe_lines
from nlt.translate.rules import parse

# A fully-specified single-leg buy, reused by several tests below.
BUY_CALL = (
    "Buy NIFTY weekly ATM call on 15 minute bars when RSI crosses above 60. "
    "Stop 30%, target 60% on the premium."
)


def _spec(text, **kw):
    result = parse(text, **kw)
    assert result.spec is not None, result.questions and result.questions[0].text
    return result


def _refused(text, **kw):
    result = parse(text, **kw)
    assert result.spec is None, "expected a question, got a spec"
    assert result.questions, "a refusal with nothing to say is not a refusal"
    return result


# ------------------------------------------------------------- single-leg buys


def test_a_fully_specified_call_buy_reads() -> None:
    result = _spec(BUY_CALL)
    i = result.spec.instrument
    assert (i.symbol, i.trade_as, i.option_type, i.strike_offset, i.expiry) == (
        "NIFTY",
        "option",
        "CE",
        0,
        "nearest_weekly",
    )
    assert result.spec.direction == "long"
    assert result.spec.exit.stop_pct == 30.0
    assert result.spec.exit.target_pct == 60.0


def test_a_put_buy_reads() -> None:
    result = _spec("Buy NIFTY weekly ATM put on 15 minute bars when RSI crosses below 30. Stop 1%")
    assert result.spec.instrument.option_type == "PE"


@pytest.mark.parametrize(
    "words,offset",
    [
        ("2 strikes OTM", 2),
        ("2 strikes out of the money", 2),
        ("3 strikes ITM", -3),
        ("3 strikes in the money", -3),
        ("OTM", 1),
        ("ITM", -1),
        ("ATM", 0),
        ("at-the-money", 0),
    ],
)
def test_strike_offsets_are_read(words, offset) -> None:
    result = _spec(
        f"Buy NIFTY {words} weekly call on 15 minute bars when RSI crosses above 60. Stop 1%"
    )
    assert result.spec.instrument.strike_offset == offset


@pytest.mark.parametrize(
    "words,expiry",
    [
        ("weekly", "nearest_weekly"),
        ("this week's", "nearest_weekly"),
        ("next week", "next_weekly"),
        ("next weekly", "next_weekly"),
        ("monthly", "monthly"),
        ("this month's", "monthly"),
    ],
)
def test_expiries_are_read(words, expiry) -> None:
    result = _spec(
        f"Buy NIFTY ATM {words} call on 15 minute bars when RSI crosses above 60. Stop 1%"
    )
    assert result.spec.instrument.expiry == expiry


def test_sensex_is_tradeable_as_an_option() -> None:
    result = _spec(
        "Buy SENSEX weekly ATM call on 15 minute bars when RSI crosses above 60. Stop 1%"
    )
    assert result.spec.instrument.symbol == "SENSEX"


def test_banknifty_is_tradeable_as_an_option() -> None:
    result = _spec(
        "Buy BANKNIFTY weekly ATM put on 15 minute bars when RSI crosses below 30. Stop 1%"
    )
    assert result.spec.instrument.symbol == "BANKNIFTY"


def test_an_unnamed_underlying_silently_means_nifty() -> None:
    """Consistent with the rest of the app (HANDOFF A1): no instrument named
    means NIFTY, and the readback must label that line "Default", not "You said"."""
    result = _spec("Buy weekly ATM call on 15 minute bars when RSI crosses above 60. Stop 1%")
    assert result.spec.instrument.symbol == "NIFTY"
    labels = {line: label for _h, line, label in describe_lines(result.spec, result.sources)}
    instrument_line = next(line for line in labels if line.startswith("NIFTY,"))
    assert labels[instrument_line] != SAID


# ------------------------------------------------------- overnight and expiry


def test_overnight_holding_is_allowed_on_an_option() -> None:
    """Unlike the old rule, an option need not square off the same day -- only
    expiry forces an exit, not the end of every session."""
    result = _spec(
        "Buy NIFTY weekly ATM call on 5 minute bars holding overnight when RSI "
        "crosses above 60. Stop 1%"
    )
    assert result.spec.schedule.intraday is False


def test_the_readback_promises_the_expiry_day_close_not_settlement() -> None:
    text = describe(_spec(BUY_CALL).spec)
    assert "closed at 15:15" in text
    assert "opening price of that candle" in text
    assert "never held to settlement" in text


def test_the_readback_promises_the_skip_if_no_trade_rule() -> None:
    text = describe(_spec(BUY_CALL).spec)
    assert "skipped" in text and "different strike" in text


def test_banknifty_weekly_gets_the_discontinued_note() -> None:
    result = _spec(
        "Buy BANKNIFTY weekly ATM call on 15 minute bars when RSI crosses above 60. Stop 1%"
    )
    text = describe(result.spec)
    assert f"{BANKNIFTY_WEEKLY_LAST_EXPIRY:%d %b %Y}" in text


def test_banknifty_monthly_gets_no_discontinued_note() -> None:
    result = _spec(
        "Buy BANKNIFTY monthly ATM call on 15 minute bars when RSI crosses above 60. Stop 1%"
    )
    assert "stopped trading" not in describe(result.spec)


def test_nifty_weekly_gets_no_discontinued_note() -> None:
    result = _spec(BUY_CALL)
    assert "stopped trading" not in describe(result.spec)


# ---------------------------------------------------------------------- sizing


def test_sizing_defaults_to_one_lot_of_todays_size() -> None:
    result = _spec(BUY_CALL)
    assert result.spec.sizing.mode == "fixed_lots"
    assert result.spec.sizing.lots == 1
    text = describe(result.spec)
    assert "1 lot of 65, today's lot" in text


# --------------------------------------------------------- the premium labels


def test_stop_and_target_say_premium_and_are_said_when_the_user_wrote_it() -> None:
    result = _spec(BUY_CALL)  # says "...on the premium"
    labels = {line: label for _h, line, label in describe_lines(result.spec, result.sources)}
    assert labels["Take profit at +60% of the option's premium"] == SAID
    assert labels["Stop loss at -30% of the option's premium"] == SAID


def test_stop_and_target_are_worked_out_when_the_user_did_not_say_premium() -> None:
    result = _spec(
        "Buy NIFTY weekly ATM call on 15 minute bars when RSI crosses above 60. "
        "Stop 30%, target 60%."
    )
    labels = {line: label for _h, line, label in describe_lines(result.spec, result.sources)}
    assert labels["Take profit at +60% of the option's premium"] == WORKED_OUT
    assert labels["Stop loss at -30% of the option's premium"] == WORKED_OUT


def test_option_price_wording_also_counts_as_said() -> None:
    result = _spec(
        "Buy NIFTY weekly ATM call on 15 minute bars when RSI crosses above 60. "
        "Stop 30% of the option's price."
    )
    labels = {line: label for _h, line, label in describe_lines(result.spec, result.sources)}
    assert labels["Stop loss at -30% of the option's premium"] == SAID


# ------------------------------------------------------------ missing pieces


def test_missing_candle_size_asks_a_question() -> None:
    result = _refused("Buy NIFTY weekly ATM call when RSI crosses above 60. Stop 1%")
    assert result.questions[0].field == "instrument.timeframe"


def test_answering_the_candle_size_question_resumes_the_parse() -> None:
    result = _spec(
        "Buy NIFTY weekly ATM call when RSI crosses above 60. Stop 1%",
        answers={"instrument.timeframe": "15m"},
    )
    assert result.spec.instrument.timeframe == "15m"


def test_missing_strike_asks_a_question() -> None:
    result = _refused("Buy NIFTY weekly call on 15 minute bars when RSI crosses above 60. Stop 1%")
    assert result.questions[0].field == "instrument.strike_offset"


def test_answering_the_strike_question_resumes_the_parse() -> None:
    result = _spec(
        "Buy NIFTY weekly call on 15 minute bars when RSI crosses above 60. Stop 1%",
        answers={"instrument.strike_offset": "1"},
    )
    assert result.spec.instrument.strike_offset == 1


def test_missing_expiry_asks_a_question() -> None:
    result = _refused("Buy NIFTY ATM call on 15 minute bars when RSI crosses above 60. Stop 1%")
    assert result.questions[0].field == "instrument.expiry"


def test_answering_the_expiry_question_resumes_the_parse() -> None:
    result = _spec(
        "Buy NIFTY ATM call on 15 minute bars when RSI crosses above 60. Stop 1%",
        answers={"instrument.expiry": "monthly"},
    )
    assert result.spec.instrument.expiry == "monthly"


def test_missing_call_or_put_asks_a_question() -> None:
    result = _refused(
        "Buy NIFTY weekly ATM option on 15 minute bars when RSI crosses above 60. Stop 1%"
    )
    assert result.questions[0].field == "instrument.option_type"


def test_answering_the_call_or_put_question_resumes_the_parse() -> None:
    result = _spec(
        "Buy NIFTY weekly ATM option on 15 minute bars when RSI crosses above 60. Stop 1%",
        answers={"instrument.option_type": "PE"},
    )
    assert result.spec.instrument.option_type == "PE"


def test_a_stop_loss_is_still_required_on_an_option() -> None:
    """Fully specified in every other way, an option strategy with no stop must
    still ask -- the stop-loss rule is never relaxed for any instrument."""
    result = _refused("Buy NIFTY weekly ATM call on 15 minute bars when RSI crosses above 60.")
    assert result.questions[0].field == "exit.stop_pct"


# -------------------------------------------------------- refused: selling


@pytest.mark.parametrize(
    "text",
    [
        "Sell the NIFTY weekly ATM call, bank 50% of the premium, SL 2%",
        "Short the NIFTY weekly ATM put when RSI crosses below 30, stop 30%",
        "Write a NIFTY weekly ATM call when RSI crosses above 60, stop 30%",
    ],
)
def test_selling_an_option_is_refused(text) -> None:
    result = _refused(text)
    assert result.questions[0].field == "instrument.trade_as"
    assert "next update" in result.questions[0].text


@pytest.mark.parametrize(
    "text",
    [
        "Sell the NIFTY weekly ATM straddle at 09:20, stop 30% on each leg",
        "Buy a NIFTY strangle when IV spikes, stop 30%",
        "Sell a NIFTY iron condor at 09:20, stop 30%",
        "Sell a NIFTY butterfly spread at 09:20, stop 30%",
        "Sell the NIFTY weekly ATM call and put at 09:20, stop 30%",
        "Sell a NIFTY credit spread at 09:20, stop 30%",
        "Hedge my NIFTY call with a put, stop 30%",
    ],
)
def test_multi_leg_options_are_refused(text) -> None:
    result = _refused(text)
    assert result.questions[0].field == "instrument.trade_as"
    assert "next update" in result.questions[0].text


def test_the_70s_two_option_sell_sentences_stay_refused() -> None:
    """The same two sentences HANDOFF.md names (#8, #21) -- still refused,
    this time for the real reason instead of "options aren't supported at all"."""
    for text in [
        "Sell the 25000 Nifty Call, hedge it with the 25100. Close at 50% of the max profit",
        "Sell the at-the-money straddle at 9:35 on expiry day. Stop at 25% of premium, square-off at 3:10 PM",
    ]:
        result = _refused(text)
        assert result.questions[0].field == "instrument.trade_as"


# ---------------------------------------------- refused: fixed strike / points


@pytest.mark.parametrize(
    "text",
    [
        "Buy the NIFTY 25000 call on 15 minute bars when RSI crosses above 60, stop 30%",
        "Buy the NIFTY weekly 25000 call when RSI crosses above 60, stop 30%",
        "Buy the NIFTY weekly put 200 points below spot when RSI crosses below 30, stop 30%",
    ],
)
def test_fixed_strike_or_points_offset_is_refused(text) -> None:
    result = _refused(text)
    assert result.questions[0].field == "instrument.strike_offset"
    assert "coming later" in result.questions[0].text


# -------------------------------------------------------- refused: underlying


def test_a_stock_option_is_refused_by_name() -> None:
    result = _refused("Buy RELIANCE call when RSI crosses above 60. Stop 30%")
    assert result.questions[0].field == "instrument.symbol"
    assert "RELIANCE" in result.questions[0].text


def test_sensex_without_an_option_is_refused() -> None:
    result = _refused("Buy SENSEX when RSI crosses above 60. Stop 1%, target 2%")
    assert result.questions[0].field == "instrument.symbol"
    assert "option" in result.questions[0].text.lower()


def test_sensex_futures_still_gets_the_ordinary_futures_refusal() -> None:
    """Unrelated to options: SENSEX futures is just another unsupported
    contract, and must read the same way every other one does."""
    result = _refused("buy sensex futures when rsi cracks 30, target 2%, stop loss 1%")
    assert "Futures work on" in result.questions[0].text


# ------------------------------------------------------------- the model spec


def test_options_long_only_at_the_model_level() -> None:
    """Defence in depth: even built directly, a sold option is rejected."""
    with pytest.raises(ValidationError, match="selling"):
        StrategySpec.model_validate(
            {
                "name": "x",
                "description": "x",
                "instrument": {
                    "symbol": "NIFTY",
                    "trade_as": "option",
                    "option_type": "CE",
                },
                "direction": "short",
                "entry": {
                    "kind": "compare",
                    "op": "gt",
                    "left": {"kind": "ref", "name": "close"},
                    "right": {"kind": "const", "value": 0},
                },
                "exit": {"stop_pct": 10.0},
            }
        )


def test_option_type_auto_is_still_legal_for_non_product_uses() -> None:
    """`trade_as="option"` with the field's own default is used all over
    `test_engine.py` to exercise lot/margin mechanics generically, on symbols
    that are not real option underlyings -- that must keep working."""
    Instrument(symbol="TCS", trade_as="option")  # option_type defaults to "auto"


def test_option_type_has_no_value_besides_auto_ce_pe() -> None:
    """The field's own `Literal` is the first line of defence -- nothing but
    "auto", "CE" or "PE" can reach the model at all."""
    with pytest.raises(ValidationError):
        Instrument(symbol="NIFTY", trade_as="option", option_type="XY")


def test_a_tradeable_option_must_be_on_a_supported_underlying() -> None:
    with pytest.raises(ValidationError, match="not supported"):
        Instrument(symbol="TCS", trade_as="option", option_type="CE")


# ----------------------------------------------------------- 0-wrong, options


def test_the_43_options_prompts_that_are_not_single_leg_buys_never_become_a_spec() -> None:
    import json
    from pathlib import Path

    templates = json.loads(
        (Path(__file__).parent / "fixtures" / "template_prompts.json").read_text()
    )
    for case in templates:
        if case["klass"] == "options-later":
            result = parse(case["text"])
            assert result.spec is None, case["name"]


# -------------------------------------------------------- the loader refuses
#
# Inherited from `test_options_refused.py`: a different phase builds the
# contract-picking and premium data this needs, and until it lands,
# `load_for_spec` must keep refusing an option spec rather than quietly
# backtesting the index's own price under an option's name.


def _loader_test_spec():
    spec = parse("buy nifty when rsi cracks 30, target 2%, stop loss 1%").spec
    assert spec is not None
    return spec.model_copy(update={"instrument": Instrument(symbol="NIFTY", trade_as="option")})


def test_load_for_spec_refuses_an_option_spec() -> None:
    with pytest.raises(NotImplementedError) as exc:
        load_for_spec(_loader_test_spec())
    message = str(exc.value).lower()
    assert "option" in message
    assert "data" in message


def test_the_loader_refusal_message_reaches_the_user_intact() -> None:
    with pytest.raises(NotImplementedError) as exc:
        load_for_spec(_loader_test_spec())
    message = str(exc.value)
    assert "Traceback" not in message
    assert len(message) > 40
    for jargon in ("NotImplementedError", "trade_as", "load_for_spec", "None"):
        assert jargon not in message


def test_an_index_spec_still_loads() -> None:
    spec = parse("buy nifty when rsi cracks 30, target 2%, stop loss 1%").spec
    assert spec is not None
    bars_by_symbol, _notes = load_for_spec(spec)
    assert bars_by_symbol
