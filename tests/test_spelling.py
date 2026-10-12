"""Typos (A5): everyday words fixed and said; names of what is traded only asked.

The danger is a "fix" that changes meaning. Each guard below exists because a
test caught one: "bank 4% profit" read as "band", "goldm" as "gold", "ROCE"
as the "roc" indicator.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlt.translate import llm, spelling
from nlt.translate.rules import parse

FIXTURES = Path(__file__).parent / "fixtures"
GOOD = "buy reliance when rsi crosses below 30, target 2%, stop loss 1%"


def _same_strategy(a, b) -> bool:
    keep = {"description"}
    return a.model_dump(exclude=keep) == b.model_dump(exclude=keep)


def test_misspelt_everyday_words_are_fixed_and_said() -> None:
    typed = "buy reliance when rsi crosess below 30, targt 2%, stop loss 1%"
    result = parse(typed)
    assert result.spec is not None, result.questions
    assert _same_strategy(result.spec, parse(GOOD).spec)
    assert "Read 'crosess' as 'crosses'." in result.notes
    assert "Read 'targt' as 'target'." in result.notes
    assert result.spec.description == typed  # the user's words are kept as typed


@pytest.mark.parametrize(
    ("typed", "fixed"),
    [("stcoks", "stocks"), ("oversld", "oversold"), ("bolinger", "bollinger"), ("taget", "target")],
)
def test_common_slips(typed, fixed) -> None:
    _, fixes = spelling.fix_words(typed)
    assert fixes == [(typed, fixed)]


@pytest.mark.parametrize("word", ["bank", "book", "goldm", "crudeoilm", "ROCE", "Sbin"])
def test_real_words_names_and_capitals_are_never_changed(word) -> None:
    text, fixes = spelling.fix_words(word)
    assert (text, fixes) == (word, [])


def test_a_word_equally_close_to_two_words_is_left_alone() -> None:
    """'closs' is one slip from close, cross and loss: no guess."""
    assert spelling.fix_words("closs") == ("closs", [])


def test_a_word_in_capitals_is_left_alone() -> None:
    """Capitals mean a name or an acronym; "TARGT" might be one we do not know."""
    assert spelling.fix_words("TARGT") == ("TARGT", [])
    assert spelling.fix_words("targt")[1] == [("targt", "target")]


def test_nothing_is_fixed_into_an_abbreviation() -> None:
    assert spelling.fix_words("rsix")[1] == []


def test_a_misspelt_stock_is_asked_never_fixed() -> None:
    typed = "buy relaince when rsi crosses below 30, target 2%, stop loss 1%"
    result = parse(typed)
    assert result.spec is None
    q = result.questions[0]
    assert (q.field, q.text) == (spelling.SPELLING_FIELD, "Did you mean RELIANCE?")
    assert [c.value for c in q.choices] == ["RELIANCE"]
    confirmed = parse(typed, answers={spelling.SPELLING_FIELD: "RELIANCE"})
    assert confirmed.spec is not None
    assert _same_strategy(confirmed.spec, parse(GOOD).spec)
    assert confirmed.spec.description == typed


def test_an_answer_naming_another_stock_asks_again() -> None:
    typed = "buy relaince when rsi crosses below 30, target 2%, stop loss 1%"
    result = parse(typed, answers={spelling.SPELLING_FIELD: "TCS"})
    assert result.spec is None
    assert result.questions[0].field == spelling.SPELLING_FIELD


def test_a_misspelt_index_is_asked_too() -> None:
    q = parse("buy nifyt when rsi cracks 30, target 2%, stop 1%").questions[0]
    assert q.text == "Did you mean NIFTY?"


def test_the_ai_is_never_asked_to_overrule_the_spelling_question() -> None:
    result = parse("buy relaince when rsi crosses below 30, target 2%, stop loss 1%")
    assert not llm.should_ask_llm(result)


def test_the_ai_reads_the_fixed_words_and_keeps_the_typed_ones() -> None:
    seen: list[str] = []

    def transport(system, user):
        seen.append(user)
        raise ConnectionError("no model in tests")

    typed = "go long on TCS when the 9-day EMA climbs over the 50-day EMA, targt 4%, SL 2%"
    llm.parse_llm(typed, transport=transport)
    assert seen == [typed.replace("targt", "target")]


@pytest.mark.parametrize("fixture", ["user_strategies.json", "template_prompts.json"])
def test_no_real_sentence_is_changed_by_the_typo_fixer(fixture) -> None:
    """The 70 and the 43 are written correctly; a fix to any of them is a false alarm."""
    for case in json.loads((FIXTURES / fixture).read_text()):
        text, notes, question = spelling.prepare(case["text"], {})
        assert (text, notes, question) == (case["text"], [], None), case["n"]
