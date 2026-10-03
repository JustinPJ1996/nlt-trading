"""The AI reader, tested against a scripted stand-in for the model.

No test here reaches a real model -- `conftest._no_live_ai` blocks that for the
whole suite. Each test hands `translate`/`parse_llm` a transport that returns a
fixed reply, which is how the checks are proven: most of these replies are
deliberately wrong in one specific way, and the test asserts the code caught it.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from nlt.translate import llm
from nlt.translate.readback import describe
from nlt.translate.rules import parse

SENTENCE = (
    "Go long on TCS the moment its 9-day EMA climbs over its 50-day EMA, bank 4% profit, SL 2%"
)

GOOD = {
    "kind": "strategy",
    "symbol": "TCS",
    "direction": "long",
    "indicators": [
        {"id": "ema9", "type": "ema", "params": {"length": 9}},
        {"id": "ema50", "type": "ema", "params": {"length": 50}},
    ],
    "entry": {
        "kind": "compare",
        "op": "crosses_above",
        "left": {"kind": "ref", "name": "ema9"},
        "right": {"kind": "ref", "name": "ema50"},
    },
    "exit": {"target_pct": 4, "stop_pct": 2},
    "evidence": [
        {"field": "direction", "quote": "Go long"},
        {"field": "instrument", "quote": "TCS"},
        {"field": "entry", "quote": "the moment its 9-day EMA climbs over its 50-day EMA"},
        {"field": "exit.target", "quote": "bank 4% profit"},
        {"field": "exit.stop", "quote": "SL 2%"},
    ],
}


class Scripted:
    """A fake model: returns `reply` and records every call."""

    def __init__(self, reply):
        self.reply = reply if isinstance(reply, str) else json.dumps(reply)
        self.calls: list[str] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append(user)
        return self.reply


def _variant(**changes) -> dict:
    d = copy.deepcopy(GOOD)
    d.update(changes)
    return d


def _fields(result) -> set[str]:
    return {q.field for q in result.questions}


def _rejected_numbers(result, *values: str) -> bool:
    """True when the result is the "numbers not in your words" refusal naming `values`."""
    if result is None or result.spec is not None or len(result.questions) != 1:
        return False
    text = result.questions[0].text
    return "couldn't match every number" in text and all(v in text for v in values)


# ---------------------------------------------------------------- happy path


def test_the_sentence_is_one_the_rules_cannot_read() -> None:
    """Otherwise every test below would be exercising the rules, not the AI path."""
    assert parse(SENTENCE).spec is None
    assert llm.should_ask_llm(parse(SENTENCE))


def test_a_checked_reply_becomes_a_spec() -> None:
    fake = Scripted(GOOD)
    result = llm.translate(SENTENCE, transport=fake)

    assert fake.calls == [SENTENCE]
    assert result.source == "llm"
    assert result.spec is not None
    assert result.spec.instrument.symbol == "TCS"
    assert result.spec.instrument.trade_as == "stock"
    assert result.spec.exit.stop_pct == 2
    assert result.spec.exit.target_pct == 4
    assert result.spec.description == SENTENCE
    assert llm.AI_NOTE in result.notes


def test_defaults_come_from_the_rules_not_the_model() -> None:
    """Risk limits and sizing must be identical whichever reader built the recipe."""
    spec = llm.translate(SENTENCE, transport=Scripted(GOOD)).spec
    rules_spec = parse("Buy TCS when RSI crosses below 30, target 4%, stop loss 2%").spec
    assert spec.risk == rules_spec.risk
    assert spec.sizing == rules_spec.sizing


def test_the_readback_is_still_generated_mechanically() -> None:
    spec = llm.translate(SENTENCE, transport=Scripted(GOOD)).spec
    text = describe(spec)
    assert "TCS" in text
    assert "Stop loss at -2%" in text
    assert "Take profit at +4%" in text


def test_naming_the_only_output_reads_the_same_as_the_rules() -> None:
    draft = copy.deepcopy(GOOD)
    draft["entry"]["left"]["output"] = "ema"
    spec = llm.translate(SENTENCE, transport=Scripted(draft)).spec
    assert spec == llm.translate(SENTENCE, transport=Scripted(GOOD)).spec
    assert "ema ema" not in describe(spec).lower()


def test_a_target_stated_as_a_gain_is_accepted() -> None:
    sentence = SENTENCE.replace("bank 4% profit", "cover at 4% gain")
    draft = copy.deepcopy(GOOD)
    draft["evidence"][3] = {"field": "exit.target", "quote": "cover at 4% gain"}
    spec = llm.parse_llm(sentence, transport=Scripted(draft)).spec
    assert spec is not None
    assert spec.exit.target_pct == 4


# ------------------------------------------------------------ rules go first


@pytest.mark.parametrize(
    "sentence",
    [
        "Buy RELIANCE at RSI 30 with 5 percent stop loss and 10 percent target",
        "Instantly buy NIFTY 100 stocks when ema8 crosses above ema21 and adx14 > 20. "
        "Sell at 3% profit. Stop loss 1.5%",
    ],
)
def test_the_model_is_never_asked_about_a_sentence_the_rules_read(sentence) -> None:
    fake = Scripted(GOOD)
    result = llm.translate(sentence, transport=fake)
    assert fake.calls == []
    assert result.source == "rules"
    assert result.spec is not None


@pytest.mark.parametrize(
    "sentence",
    [
        # options: refused on purpose until there is option price data
        "Sell the 25000 Nifty Call, hedge it with the 25100. Close at 50% of the max profit",
        # an unsupported candle size
        "Buy ITC when RSI drops below 30 and sell above 70. Use 2-minute candles. stop 1%",
        # "NIFTY 50" -- the index or the basket?
        "How is NIFTY50 trending right now and what should I do?",
        # VWAP on a basket that only has daily data
        "Instantly buy NIFTY 50 stocks when price > vwap and rsi < 30. Sell at 2% profit. "
        "Stop loss 1%.",
        # understood everything, only needs a stop loss
        "Buy TCS when RSI drops below 30",
    ],
)
def test_the_model_is_never_asked_to_overrule_a_deliberate_refusal(sentence) -> None:
    fake = Scripted(GOOD)
    result = llm.translate(sentence, transport=fake)
    assert fake.calls == []
    assert result.source == "rules"
    assert result.spec is None


def test_the_ai_reader_applies_the_options_refusal_itself() -> None:
    """Even called directly, past the rules-first gate, it never reads an option trade."""
    fake = Scripted(GOOD)
    result = llm.parse_llm("Go long on the NIFTY 25000 call, bank 4% profit, SL 2%", transport=fake)
    assert fake.calls == []
    assert _fields(result) == {"instrument.trade_as"}


def test_futures_are_refused_without_asking_the_model() -> None:
    fake = Scripted(GOOD)
    result = llm.parse_llm(
        "Buy TCS futures when its 9-day EMA climbs over 50, SL 2%", transport=fake
    )
    assert fake.calls == []
    assert result.spec is None
    assert "futures" in result.questions[0].text


# -------------------------------------------------- falling back to the rules


def test_no_key_means_no_call_and_the_rules_answer_stands() -> None:
    """The suite runs with the AI disabled, exactly like a box with no key."""
    assert not llm.is_configured()
    assert llm.parse_llm(SENTENCE) is None
    result = llm.translate(SENTENCE)
    assert result.source == "rules"
    assert result.spec is None


@pytest.mark.parametrize("reply", ["not json at all", "{broken", "[1, 2]", ""])
def test_garbage_from_the_model_falls_back_to_the_rules(reply) -> None:
    result = llm.translate(SENTENCE, transport=Scripted(reply))
    assert result.source == "rules"
    assert result.spec is None


def test_a_transport_failure_falls_back_to_the_rules() -> None:
    def broken(system, user):
        raise ConnectionError("offline")

    result = llm.translate(SENTENCE, transport=broken)
    assert result.source == "rules"


def test_an_invented_indicator_falls_back_to_the_rules() -> None:
    bad = _variant(indicators=[{"id": "ema9", "type": "magic_line", "params": {}}])
    assert llm.parse_llm(SENTENCE, transport=Scripted(bad)) is None


# ------------------------------------------------------------ fabrication


def test_a_quote_the_user_never_wrote_discards_the_whole_reply() -> None:
    bad = copy.deepcopy(GOOD)
    bad["evidence"][-1] = {"field": "exit.stop", "quote": "stop loss 2%"}  # user wrote "SL 2%"
    assert llm.parse_llm(SENTENCE, transport=Scripted(bad)) is None


def test_a_dropped_clause_comes_back_as_a_question() -> None:
    sentence = SENTENCE + " and only if P/E is under 18"
    result = llm.parse_llm(sentence, transport=Scripted(GOOD))
    assert result.spec is None
    assert "unparsed" in _fields(result)
    assert "18" in result.questions[0].text


# -------------------------------------------------------------- the stop loss


def test_no_stop_in_the_sentence_means_a_question_not_a_default() -> None:
    sentence = "Go long on TCS the moment its 9-day EMA climbs over its 50-day EMA, bank 4% profit"
    draft = _variant(
        exit={"target_pct": 4, "stop_pct": 2},  # the model invented the stop
        evidence=GOOD["evidence"][:-1],
    )
    result = llm.parse_llm(sentence, transport=Scripted(draft))
    assert _rejected_numbers(result, "2")


def test_a_model_that_leaves_the_stop_out_gets_the_stop_question() -> None:
    sentence = "Go long on TCS the moment its 9-day EMA climbs over its 50-day EMA, bank 4% profit"
    draft = _variant(exit={"target_pct": 4}, evidence=GOOD["evidence"][:-1])
    result = llm.parse_llm(sentence, transport=Scripted(draft))
    assert result.spec is None
    assert _fields(result) == {"exit.stop_pct"}

    answered = llm.parse_llm(sentence, answers={"exit.stop_pct": "1.5"}, transport=Scripted(draft))
    assert answered.spec is not None
    assert answered.spec.exit.stop_pct == 1.5


def test_a_stop_and_target_swapped_by_the_model_is_caught() -> None:
    """Both numbers are in the sentence -- only the quote tying each to its word catches it."""
    swapped = _variant(exit={"target_pct": 2, "stop_pct": 4})
    result = llm.parse_llm(SENTENCE, transport=Scripted(swapped))
    assert _rejected_numbers(result, "2", "4")


def test_a_stop_quote_without_a_stop_word_is_caught() -> None:
    bad = copy.deepcopy(GOOD)
    bad["evidence"][-2] = {"field": "exit.stop", "quote": "bank 4% profit"}
    bad["evidence"][-1] = {"field": "exit.target", "quote": "SL 2%"}
    bad["exit"] = {"target_pct": 2, "stop_pct": 4}
    result = llm.parse_llm(SENTENCE, transport=Scripted(bad))
    assert _rejected_numbers(result, "2", "4")


# ------------------------------------------------------------------ numbers


def test_an_indicator_length_the_user_never_said_is_caught() -> None:
    bad = copy.deepcopy(GOOD)
    bad["indicators"][1]["params"]["length"] = 21
    result = llm.parse_llm(SENTENCE, transport=Scripted(bad))
    assert _rejected_numbers(result, "21")


def test_a_level_the_user_never_said_is_caught() -> None:
    """ "Oversold" is not a number. Turning it into 30 is a guess."""
    sentence = "Go long on TCS the moment its RSI turns oversold, bank 4% profit, SL 2%"
    draft = _variant(
        indicators=[{"id": "rsi14", "type": "rsi", "params": {"length": 14}}],
        entry={
            "kind": "compare",
            "op": "crosses_below",
            "left": {"kind": "ref", "name": "rsi14"},
            "right": {"kind": "const", "value": 30},
        },
        evidence=[
            {"field": "direction", "quote": "Go long"},
            {"field": "instrument", "quote": "TCS"},
            {"field": "entry", "quote": "the moment its RSI turns oversold"},
            {"field": "exit.target", "quote": "bank 4% profit"},
            {"field": "exit.stop", "quote": "SL 2%"},
        ],
    )
    result = llm.parse_llm(sentence, transport=Scripted(draft))
    assert _rejected_numbers(result, "30")


def test_a_number_borrowed_from_another_clause_is_caught() -> None:
    """30 is in the sentence -- but as the profit target, not as an RSI level."""
    sentence = "Go long on TCS the moment its RSI turns oversold, bank 30% profit, SL 2%"
    draft = _variant(
        indicators=[{"id": "rsi14", "type": "rsi", "params": {"length": 14}}],
        entry={
            "kind": "compare",
            "op": "crosses_below",
            "left": {"kind": "ref", "name": "rsi14"},
            "right": {"kind": "const", "value": 30},
        },
        exit={"target_pct": 30, "stop_pct": 2},
        evidence=[
            {"field": "direction", "quote": "Go long"},
            {"field": "instrument", "quote": "TCS"},
            {"field": "entry", "quote": "the moment its RSI turns oversold"},
            {"field": "exit.target", "quote": "bank 30% profit"},
            {"field": "exit.stop", "quote": "SL 2%"},
        ],
    )
    result = llm.parse_llm(sentence, transport=Scripted(draft))
    assert _rejected_numbers(result, "30")


# ----------------------------------------------------- instrument, direction


def test_the_model_cannot_overrule_the_rules_on_the_instrument() -> None:
    sentence = "Go long on NIFTY the moment its 9-day EMA climbs over its 50-day EMA, bank 4% profit, SL 2%"
    bad = copy.deepcopy(GOOD)
    bad["evidence"][1] = {"field": "instrument", "quote": "NIFTY"}
    result = llm.parse_llm(sentence, transport=Scripted(bad))  # still says TCS
    assert result.spec is None
    assert _fields(result) == {"instrument.symbol"}


def test_the_model_cannot_invent_an_instrument() -> None:
    sentence = "Go long the moment the 9-day EMA climbs over the 50-day EMA, bank 4% profit, SL 2%"
    draft = _variant(
        evidence=[
            {"field": "direction", "quote": "Go long"},
            {"field": "entry", "quote": "the moment the 9-day EMA climbs over the 50-day EMA"},
            {"field": "exit.target", "quote": "bank 4% profit"},
            {"field": "exit.stop", "quote": "SL 2%"},
        ]
    )
    result = llm.parse_llm(sentence, transport=Scripted(draft))
    assert result.spec is None
    assert _fields(result) == {"instrument.symbol"}


def test_the_model_cannot_offer_no_instrument_at_all() -> None:
    sentence = "Go long the moment the 9-day EMA climbs over the 50-day EMA, bank 4% profit, SL 2%"
    draft = _variant(
        symbol=None,
        evidence=[
            {"field": "direction", "quote": "Go long"},
            {"field": "entry", "quote": "the moment the 9-day EMA climbs over the 50-day EMA"},
            {"field": "exit.target", "quote": "bank 4% profit"},
            {"field": "exit.stop", "quote": "SL 2%"},
        ],
    )
    result = llm.parse_llm(sentence, transport=Scripted(draft))
    assert result.spec is None
    assert result.questions[0].text == "Which instrument should this strategy trade?"


@pytest.mark.parametrize("invented", ["NIFTY", "BANKNIFTY", "NIFTY 100"])
def test_the_model_cannot_invent_an_index_or_basket(invented) -> None:
    sentence = "Go long the moment the 9-day EMA climbs over the 50-day EMA, bank 4% profit, SL 2%"
    draft = _variant(
        symbol=invented,
        evidence=[
            {"field": "direction", "quote": "Go long"},
            {"field": "entry", "quote": "the moment the 9-day EMA climbs over the 50-day EMA"},
            {"field": "exit.target", "quote": "bank 4% profit"},
            {"field": "exit.stop", "quote": "SL 2%"},
        ],
    )
    result = llm.parse_llm(sentence, transport=Scripted(draft))
    assert result.spec is None
    assert _fields(result) == {"instrument.symbol"}


def test_the_model_cannot_overrule_the_rules_on_direction() -> None:
    result = llm.parse_llm(SENTENCE, transport=Scripted(_variant(direction="short")))
    assert result.spec is None
    assert _fields(result) == {"direction"}


def test_vwap_on_daily_bars_is_refused_on_the_ai_path_too() -> None:
    sentence = "Go long on TCS the moment it climbs over its VWAP, bank 4% profit, SL 2%"
    draft = _variant(
        indicators=[{"id": "vwap", "type": "vwap", "params": {}}],
        entry={
            "kind": "compare",
            "op": "crosses_above",
            "left": {"kind": "ref", "name": "close"},
            "right": {"kind": "ref", "name": "vwap"},
        },
        evidence=[
            {"field": "direction", "quote": "Go long"},
            {"field": "instrument", "quote": "TCS"},
            {"field": "entry", "quote": "the moment it climbs over its VWAP"},
            {"field": "exit.target", "quote": "bank 4% profit"},
            {"field": "exit.stop", "quote": "SL 2%"},
        ],
    )
    result = llm.parse_llm(sentence, transport=Scripted(draft))
    assert result.spec is None
    assert _fields(result) == {"instrument.timeframe"}


# ------------------------------------------------------------------ refusals


def test_a_refusal_uses_our_words_not_the_models() -> None:
    reply = {
        "kind": "refuse",
        "refuse_category": "fundamental",
        "unsupported_quote": "P/E is under 18",
        "reason": "the model's own opinion",  # not a field; the whole reply is rejected
    }
    assert llm.parse_llm(SENTENCE + " and P/E is under 18", transport=Scripted(reply)) is None

    del reply["reason"]
    result = llm.parse_llm(SENTENCE + " and P/E is under 18", transport=Scripted(reply))
    assert result.spec is None
    assert result.questions[0].text.startswith(llm._REFUSALS["fundamental"][0])
    assert '"P/E is under 18"' in result.questions[0].text


def test_a_refusal_never_echoes_words_the_user_did_not_type() -> None:
    reply = {"kind": "refuse", "refuse_category": "event", "unsupported_quote": "buy crypto now"}
    result = llm.parse_llm(SENTENCE, transport=Scripted(reply))
    assert "crypto" not in result.questions[0].text


# ------------------------------------------- the 70 real sentences, replayed

# Written by `scripts/eval_llm.py` from a live run. Replayed here, offline,
# through exactly the same checks -- so a later change that would let one of
# those real replies through wrongly fails `make check`, not a user.
_REPLIES_PATH = Path(__file__).parent / "fixtures" / "llm_replies.json"
_NEVER_A_STRATEGY = {"question", "robo", "fundamental", "event"}


def _replayed_results():
    if not _REPLIES_PATH.exists():
        pytest.skip("no recorded AI replies yet -- run scripts/eval_llm.py with a key")
    replies = json.loads(_REPLIES_PATH.read_text())["replies"]
    cases = json.loads((Path(__file__).parent / "fixtures" / "user_strategies.json").read_text())
    missing = []

    def replay(system, user):
        if user not in replies:
            missing.append(user)
            raise KeyError(user)
        return replies[user]

    results = [(case, llm.translate(case["text"], transport=replay)) for case in cases]
    assert not missing, f"sentences now sent to the model with no recorded reply: {missing}"
    return results


def test_no_wrong_parses_among_the_70_user_strategies_on_the_ai_path() -> None:
    """The same promise the rules keep, kept by the AI reader too: a question,
    an investment request, a fundamentals screen or an event trade never comes
    back as a tradeable recipe."""
    wrong = [
        (case["n"], case["klass"], case["text"])
        for case, result in _replayed_results()
        if result.spec is not None and case["klass"] in _NEVER_A_STRATEGY
    ]
    assert not wrong, f"the AI path produced a spec for out-of-scope sentences: {wrong}"


def test_every_ai_recipe_among_the_70_states_a_stop_the_user_gave() -> None:
    for case, result in _replayed_results():
        if result.spec is not None:
            assert result.spec.exit.has_stop, case["text"]
