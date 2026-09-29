"""Options are refused, loudly, until there is option price data to test against.

This whole file is temporary scaffolding and should be deleted in the phase that
lands real contract resolution. It exists because of a live bug, not a missing
feature:

    Input:    "buy nifty call when rsi cracks 30, target 2%, stop loss 1%"
    Readback: "NIFTY, at-the-money call option, nearest weekly expiry"
              "Quantity 65 per trade (1 lot of 65)"
    Reality:  `load_for_spec` never read `trade_as`, so it returned NIFTY *spot*
              bars. The engine bought 65 units of the index at ~24,000 each --
              about Rs 15.6 lakh on a Rs 1 lakh account -- and "+2% target" meant
              2% of the index rather than 2% of the premium.

Nothing crashed. A number came back, formatted like every other number the
dashboard produces. That is the readback stating something untrue, which
HANDOFF.md names as the most serious bug class here, and it is why these tests
assert on *refusal* rather than on any result.

The user-facing half (the parser) and the safety half (the loader) are tested
separately on purpose. The parser stops the strategy being built through the
product; the loader stops any option spec, however constructed, being fed the
wrong series. Either alone would leave a route through.
"""

from __future__ import annotations

import pytest

from nlt.engine.loader import load_for_spec
from nlt.spec.models import Instrument
from nlt.translate.rules import parse

OPTION_SENTENCES = [
    "buy nifty call when rsi cracks 30, target 2%, stop loss 1%",
    "buy a nifty put when rsi crosses above 70, target 2%, stop loss 1%",
    "buy nifty ce when rsi cracks 30, target 2%, stop loss 1%",
    "buy nifty pe when rsi crosses above 70, target 2%, stop loss 1%",
    "buy nifty options when rsi cracks 30, target 2%, stop loss 1%",
]


# ------------------------------------------------------- the parser refuses


@pytest.mark.parametrize("sentence", OPTION_SENTENCES)
def test_an_option_sentence_produces_no_spec(sentence: str) -> None:
    """No spec means no backtest, which is the entire point."""
    result = parse(sentence)
    assert result.spec is None, f"{sentence!r} built a spec; it must be refused"


@pytest.mark.parametrize("sentence", OPTION_SENTENCES)
def test_the_refusal_explains_itself_in_plain_english(sentence: str) -> None:
    """A refusal a non-technical user cannot act on is barely better than a wrong number."""
    result = parse(sentence)
    assert result.questions, f"{sentence!r} was refused with no explanation"

    said = " ".join(q.text + " " + q.why for q in result.questions).lower()
    assert "option" in said, "the refusal must name what it is refusing"
    # It has to leave the user somewhere to go, not just say no.
    assert "index" in said or "nifty" in said, (
        "the refusal should point at something that does work"
    )


def test_the_same_strategy_on_the_index_still_works() -> None:
    """The refusal must be specific to options, not collateral damage.

    If "call" in a sentence started refusing index strategies too, the 70-sentence
    corpus would still pass -- it only asserts nothing parses *wrongly* -- and the
    product would quietly lose the thing it does do.
    """
    result = parse("buy nifty when rsi cracks 30, target 2%, stop loss 1%")
    assert result.spec is not None, "the plain index strategy must still parse"
    assert result.spec.instrument.trade_as == "index"


# -------------------------------------------------------- the loader refuses


def _option_spec():
    """A valid option spec, built directly -- the route the parser no longer offers."""
    spec = parse("buy nifty when rsi cracks 30, target 2%, stop loss 1%").spec
    assert spec is not None
    return spec.model_copy(update={"instrument": Instrument(symbol="NIFTY", trade_as="option")})


def test_load_for_spec_refuses_an_option_spec() -> None:
    """The gate that matters: no option spec may be handed a price series.

    Constructing an option spec is still legal -- the readback describes them and
    the engine's sizing tests use them. What must not happen is one reaching
    `load_for_spec` and being quietly given the index's own bars.
    """
    with pytest.raises(NotImplementedError) as exc:
        load_for_spec(_option_spec())

    message = str(exc.value).lower()
    assert "option" in message
    assert "data" in message, "the message must say why, not just that it failed"


def test_the_refusal_message_reaches_the_user_intact() -> None:
    """`run_pipeline` renders the exception text straight into the error shown
    on screen, so the message is user-facing copy and not a developer note."""
    with pytest.raises(NotImplementedError) as exc:
        load_for_spec(_option_spec())

    message = str(exc.value)
    assert "Traceback" not in message
    assert len(message) > 40, "a one-word failure tells the user nothing"
    # No jargon that would mean nothing to someone who does not code.
    for jargon in ("NotImplementedError", "trade_as", "load_for_spec", "None"):
        assert jargon not in message, f"{jargon!r} is not plain English"


def test_an_index_spec_still_loads() -> None:
    """The gate must not have closed the door on everything."""
    spec = parse("buy nifty when rsi cracks 30, target 2%, stop loss 1%").spec
    assert spec is not None
    bars_by_symbol, _notes = load_for_spec(spec)
    assert bars_by_symbol, "the index path must still return bars"
