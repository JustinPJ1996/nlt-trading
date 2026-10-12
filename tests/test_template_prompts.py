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

# Never a strategy: a screen is a list of stocks, options wait for Phase 2, a
# prompt without a stop must get the stop-loss question, and a stop at "the
# opening range low" is not something the engine can hold yet.
_NEVER_A_STRATEGY = {"screen", "options", "no-stop", "unsupported-stop"}

# Pinned so coverage cannot drift silently. Raise it deliberately.
_EXPECTED_RULES_SPECS: list[int] = []


def _facts(spec) -> dict:
    return {
        "symbol": spec.instrument.symbol,
        "timeframe": spec.instrument.timeframe,
        "stop_pct": spec.exit.stop_pct,
        "target_pct": spec.exit.target_pct,
        "intraday": spec.schedule.intraday,
    }


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
