"""Tradl's 43 suggested prompts, a second fixture kept apart from the 70.

The 70 real sentences prove the reader is honest with real users. These are
well-written prompts from a competitor (26 backtests, 17 screens), and they
measure coverage: how many complete strategies we can actually read.

The rule is the same as for the 70 -- **0 wrong** -- and stricter in one way.
Each complete strategy carries the facts a human read off the sentence
(symbol, bar size, stop, target, square-off). A recipe that runs with any of
those different is wrong, however plausible it looks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlt.translate import llm
from nlt.translate.rules import parse

FIXTURES = Path(__file__).parent / "fixtures"
TEMPLATES = json.loads((FIXTURES / "template_prompts.json").read_text())
_REPLIES_PATH = FIXTURES / "llm_template_replies.json"

# Never a strategy: a screen is a list of stocks, a prompt without a
# stop must get the stop-loss question (#28: a naked sold put), and a
# "needs-answer" prompt asks a question with buttons first (#24 and #36 once a
# day or every time, #33 how to re-enter, #37 whether the stop is on each
# option or the total).
_NEVER_A_STRATEGY = {"screen", "options-later", "no-stop", "unsupported-stop", "needs-answer"}

# Pinned so coverage cannot drift silently. Raise it deliberately, after reading
# each new readback against its sentence by hand.
# 19, 20, 34, 35, 40 joined with selling and multi-leg options (Phase 2 batch
# 2, 2026-10-14): read by the rules alone, each readback checked by hand.
_EXPECTED_RULES_SPECS: list[int] = [19, 20, 34, 35, 40]
# Grew from 0 when the test period, "hourly" and "hold overnight" were read
# (2026-10-12). Each readback was checked against its sentence by hand.
# 29 joined when the daily average on an intraday chart was built (A6), its
# readback checked by hand: it says "the previous day's close" out loud.
# 31, 36, 41 joined with options (Phase 2 batch 1): each is a single-leg
# option buy the rules reader cannot read on its own ("On X bars, buy..." puts
# the verb after the instrument clause), but the AI path reads the indicator
# condition and `options_words` resolves the option fields; each readback
# checked by hand against its sentence.
# The leg positions above come from the rules, so the AI path returns them too
# (the model is never asked about a leg position).
# 36 left (2026-10-16): "buy when a candle closes above the first-15-minute
# high" now asks "Once a day" or "Every time it crosses" first -- it had been
# read as "whenever the price is above", buying again straight after a stop.
# `test_36_asks_how_often_then_reads_each_answer` covers both answers.
# 26 joined (2026-10-16): the gap at the open and "at 09:30" read; its readback
# checked by hand and its entry pinned by `test_26_reads_the_gap_the_time_and_the_open`.
_EXPECTED_AI_SPECS = [19, 20, 22, 23, 26, 29, 30, 31, 34, 35, 38, 39, 40, 41, 42]


def _facts(spec) -> dict:
    if spec.instrument.legs:
        # Every leg is a fact: which side, call or put, which strike rule,
        # which expiry, its own stop/target and re-entry. A position with
        # one leg different is a different position.
        from nlt.spec.models import All, AtTime

        conds = spec.entry.conditions if isinstance(spec.entry, All) else [spec.entry]
        at = next((c for c in conds if isinstance(c, AtTime)), None)
        return {
            "symbol": spec.instrument.symbol,
            "timeframe": spec.instrument.timeframe,
            "intraday": spec.schedule.intraday,
            # `strike` (a named strike, 2026-10-16) is a fact only when set.
            "legs": [
                leg.model_dump(exclude={"strike"} if leg.strike is None else set())
                for leg in spec.instrument.legs
            ],
            "combined_target_pct": spec.exit.combined_target_pct,
            "combined_stop_pct": spec.exit.combined_stop_pct,
            "entry_time": f"{at.time:%H:%M}" if at else None,
        }
    facts = {
        "symbol": spec.instrument.symbol,
        "timeframe": spec.instrument.timeframe,
        "stop_pct": spec.exit.stop_pct,
        "target_pct": spec.exit.target_pct,
        "intraday": spec.schedule.intraday,
    }
    if spec.instrument.trade_as == "option":
        # A different strike, expiry or side is just as wrong as a different
        # stop -- it is a different contract, bought or sold at a different
        # price, however similar the rest of the recipe looks.
        facts["option_type"] = spec.instrument.option_type
        facts["strike_offset"] = spec.instrument.strike_offset
        facts["expiry"] = spec.instrument.expiry
    return facts


def _wrong(case, result) -> str | None:
    """Why this result is wrong for this prompt, or None if it is not."""
    if result.spec is None:
        return None
    if case["klass"] in _NEVER_A_STRATEGY:
        return f"a {case['klass']} prompt became a strategy"
    if result.spec.direction != "long":
        return "every template strategy buys"
    got = _facts(result.spec)
    if got != case["expect"]:
        return f"read as {got}, the sentence says {case['expect']}"
    return None


def test_every_template_is_labelled() -> None:
    assert len(TEMPLATES) == 43
    for case in TEMPLATES:
        assert case["klass"] in _NEVER_A_STRATEGY | {"strategy"}
        assert ("expect" in case) == (case["klass"] == "strategy"), case["name"]


def test_no_wrong_reads_among_the_template_prompts() -> None:
    wrong = [(c["n"], c["name"], _wrong(c, parse(c["text"]))) for c in TEMPLATES]
    wrong = [w for w in wrong if w[2]]
    assert not wrong, wrong


def test_pinned_template_coverage() -> None:
    produced = [c["n"] for c in TEMPLATES if parse(c["text"]).spec is not None]
    assert produced == _EXPECTED_RULES_SPECS, (
        f"template prompts that now produce a strategy: {produced}. If this grew, "
        "check each readback by hand and update _EXPECTED_RULES_SPECS deliberately."
    )


def test_a_template_without_a_stop_never_runs() -> None:
    for case in TEMPLATES:
        if case["klass"] == "no-stop":
            result = parse(case["text"])
            assert result.spec is None, case["name"]


# --------------------------------------------------------------- the AI path


def _replayed():
    if not _REPLIES_PATH.exists():
        pytest.skip("no recorded AI replies -- run scripts/eval_llm.py --templates")
    replies = json.loads(_REPLIES_PATH.read_text())["replies"]
    missing = []

    def replay(system, user):
        if user not in replies:
            missing.append(user)
            raise KeyError(user)
        return replies[user]

    results = [(c, llm.translate(c["text"], transport=replay)) for c in TEMPLATES]
    assert not missing, f"prompts now sent to the model with no recorded reply: {missing}"
    return results


def test_no_wrong_reads_among_the_template_prompts_on_the_ai_path() -> None:
    wrong = [(c["n"], c["name"], _wrong(c, r)) for c, r in _replayed()]
    wrong = [w for w in wrong if w[2]]
    assert not wrong, wrong


def test_pinned_template_coverage_on_the_ai_path() -> None:
    produced = [c["n"] for c, r in _replayed() if r.spec is not None]
    assert produced == _EXPECTED_AI_SPECS, (
        f"template prompts the AI path now reads: {produced}. Check each readback by hand."
    )


def test_ai_path_template_periods_become_the_test_period() -> None:
    for case, result in _replayed():
        if result.spec is not None:
            assert result.period is not None, case["name"]
            assert result.period.words in case["text"].lower(), case["name"]


def test_ai_path_labels_for_a_template() -> None:
    """INFY, hourly, hold overnight, stop 2%: all typed. The Bollinger 20/2 is ours."""
    from nlt.translate.readback import SAID, WORKED_OUT, describe_lines

    case, result = next((c, r) for c, r in _replayed() if c["name"] == "Bollinger Band Bounce")
    labels = {line: label for _h, line, label in describe_lines(result.spec, result.sources)}
    assert labels["INFY (NSE)"] == SAID
    assert labels["Hourly candles"] == SAID
    assert labels["Stop loss at -2%"] == SAID
    assert labels["Hold overnight: no square-off before the close"] == SAID
    entry = next(line for line in labels if line.startswith("When "))
    assert labels[entry] == WORKED_OUT


@pytest.mark.parametrize(("pick", "limit"), [("once", 1), ("every", None)])
def test_36_asks_how_often_then_reads_each_answer(pick, limit) -> None:
    from nlt.spec.models import Compare, Ref

    replies = json.loads(_REPLIES_PATH.read_text())["replies"]
    case = next(c for c in TEMPLATES if c["n"] == 36)
    asked = llm.translate(case["text"], transport=lambda _s, u: replies[u])
    assert [q.field for q in asked.questions] == ["schedule.daily_limit"]
    assert [c.value for c in asked.questions[0].choices] == ["once", "every"]

    answered = llm.translate(
        case["text"], answers={"schedule.daily_limit": pick}, transport=lambda _s, u: replies[u]
    )
    assert answered.spec is not None, answered.questions
    assert _wrong(case, answered) is None
    assert answered.spec.schedule.max_entries_per_day == limit
    entry = answered.spec.entry
    assert isinstance(entry, Compare) and entry.op == "crosses_above"
    assert isinstance(entry.left, Ref) and isinstance(entry.right, Ref)
    assert entry.left.name == "close" and entry.right.output == "high"


def test_26_reads_the_gap_the_time_and_the_open() -> None:
    import datetime as dt

    from nlt.spec.models import All, AtTime, Compare, Const, Ref

    replies = json.loads(_REPLIES_PATH.read_text())["replies"]
    case = next(c for c in TEMPLATES if c["n"] == 26)
    spec = llm.translate(case["text"], transport=lambda _s, u: replies[u]).spec
    assert spec is not None
    assert isinstance(spec.entry, All)
    parts = list(spec.entry.conditions)
    assert AtTime(time=dt.time(9, 30)) in parts
    kinds = {i.id: i.type for i in spec.indicators}
    gaps = [c for c in parts if isinstance(c, Compare) and isinstance(c.right, Const)]
    (gap,) = gaps
    assert isinstance(gap.left, Ref) and isinstance(gap.right, Const)
    assert (kinds[gap.left.name], gap.left.output, gap.op, gap.right.value) == (
        "session_open",
        "gap_pct",
        "lt",
        -0.5,
    )
    opens = [c for c in parts if isinstance(c, Compare) and isinstance(c.right, Ref)]
    (below,) = opens
    assert isinstance(below.left, Ref) and isinstance(below.right, Ref)
    assert (below.left.name, below.op, kinds[below.right.name], below.right.output) == (
        "close",
        "lt",
        "session_open",
        "open",
    )
