#!/usr/bin/env python
"""Run the 70 real user sentences through the AI reader and record what it said.

Needs an OpenRouter key (see `nlt/translate/llm.py`). Costs roughly Rs 1-2 per
sentence the rules could not read -- about 65 of the 70.

    .venv/bin/python scripts/eval_llm.py

Every raw model reply is saved to `tests/fixtures/llm_replies.json`, keyed by
sentence. The test suite replays those replies through the same checks, so
"0 wrong answers" is defended on every `make check` without the network --
and a change to the checks that would have let a bad reply through fails there.

Prints each sentence's outcome, and the full readback for every recipe the AI
produced: each one needs a human to read it against the sentence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from nlt.translate import llm
from nlt.translate.readback import describe

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
NEVER_A_STRATEGY = {"question", "robo", "fundamental", "event"}


def main() -> int:
    if not llm.is_configured():
        print(f"No OpenRouter key. Put it in {llm.KEY_PATH} (one line, nothing else).")
        return 1

    cases = json.loads((FIXTURES / "user_strategies.json").read_text())
    replies: dict[str, str] = {}

    def recording(system: str, user: str) -> str:
        reply = llm.openrouter_transport(system, user)
        replies[user] = reply
        return reply

    produced, wrong, asked = [], [], 0
    for case in cases:
        result = llm.translate(case["text"], transport=recording)
        asked += case["text"] in replies
        outcome = "RECIPE" if result.spec else "refused"
        print(f"#{case['n']:<3} [{case['klass']}] {result.source:5} {outcome}: {case['text'][:70]}")
        if result.spec is None:
            print(f"       -> {result.questions[0].text[:150] if result.questions else '(none)'}")
            continue
        produced.append(case["n"])
        if case["klass"] in NEVER_A_STRATEGY:
            wrong.append(case["n"])
        print("\n" + "\n".join("       " + line for line in describe(result.spec).splitlines()))
        print()

    out = FIXTURES / "llm_replies.json"
    out.write_text(
        json.dumps({"model": llm.model_name(), "replies": replies}, indent=2, sort_keys=True) + "\n"
    )
    print(f"\nAsked the model about {asked} of {len(cases)} sentences; replies saved to {out}.")
    print(f"Recipes: {len(produced)} {produced}")
    print(f"Recipes for sentences that are never a strategy: {len(wrong)} {wrong}")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
