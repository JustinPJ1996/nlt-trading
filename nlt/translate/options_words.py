"""Option-specific vocabulary: type, strike, expiry, and the sell/multi-leg refusal.

Phase 2 batch 1 supports exactly one shape: **buying one call or one put**.
Everything else an options sentence might describe -- selling, two or more
legs, a fixed strike price, a points offset from spot -- is refused by name
here, before the sentence ever reaches the AI reader, so that reader never has
to re-learn this boundary on its own (`rules._extract_instrument` calls
straight into this module and returns whatever question it raises).

Kept deliberately separate from `rules.py`: this is the one place that knows
what "ATM", "2 strikes OTM" and "monthly expiry" mean, and it is testable on
its own. `Question`/`Choice` live in `rules.py`; this module imports them
lazily inside functions (never at module scope) so the two modules can import
each other without a circular-import error -- `rules.py` imports this module
at the top of the file.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

# NSE/BSE underlyings this platform can test as options. A stock option is not
# supported yet -- "RELIANCE call" is refused by name rather than silently
# traded as a NIFTY option, which would be a wrong answer dressed up as a
# strategy.
OPTION_UNDERLYINGS = ("NIFTY", "BANKNIFTY", "SENSEX")

# NSE discontinued BANKNIFTY weekly options; the last weekly expiry traded was
# 13 Nov 2024. A single named constant so this can be corrected in one place
# if the exact date turns out to be wrong.
BANKNIFTY_WEEKLY_LAST_EXPIRY = dt.date(2024, 11, 13)


# --------------------------------------------------------------------- words

# Unambiguous on their own: no sentence uses these words to mean anything else.
# "credit spread"/"debit spread" are kept as whole phrases rather than folded
# into the weak list below -- a spread of *something* is generic, but a credit
# or debit spread specifically is options jargon with no other meaning.
_UNAMBIGUOUS_MULTI_LEG = re.compile(
    r"\b(straddles?|strangles?|condors?|butterfl(?:y|ies)|credit\s+spreads?|debit\s+spreads?)\b"
)
# "call and put" / "put and call" bought or sold together is two legs, however
# it is phrased.
_CALL_AND_PUT = re.compile(r"\b(?:calls?\s+and\s+puts?|puts?\s+and\s+calls?)\b")
# Ambiguous alone ("hedge my portfolio" names no option); only counted as
# multi-leg when an option word also appears in the same sentence.
_WEAK_MULTI_LEG = re.compile(r"\b(spreads?|credits?|hedg(?:e|es|ed|ing)?)\b")

# Marks a sentence as being about options at all. "ce"/"pe" are deliberately
# *not* bare two-letter words here on their own -- see `_CALL_WORD`/`_PUT_WORD`
# below, which anchor them the same way so "P/E" (a fundamental, not an
# option) never matches.
_OPTION_TRIGGER = re.compile(r"\b(calls?|puts?|ce|pe|options?)\b")

# The AI-path backstop in `llm.py`: broader than `_OPTION_TRIGGER` so an
# unambiguous multi-leg word alone also trips it, since by the time that code
# runs `read()` should already have turned any of these into either an option
# reading or a question.
OPTION_TRIGGER_FOR_LLM = re.compile(
    r"\b(calls?|puts?|ce|pe|options?|straddles?|strangles?|condors?|butterfl(?:y|ies))\b"
)

_CALL_WORD = re.compile(r"\b(calls?|ce)\b")
_PUT_WORD = re.compile(r"\b(puts?|pe)\b")

# Only the *leading* verb says who initiates the trade -- the same reasoning
# `rules._extract_direction` uses for buy/sell. "write" is an options-only verb
# ("write a call") that the general direction table has no reason to know.
_SELL_LEAD = re.compile(
    r"^\s*(?:please\s+)?(?:i(?:'d| would)? (?:want to|like to)\s+)?"
    r"(sell|short sell|short|go short|write)\b"
)

_ATM_WORD = re.compile(r"\bat[\s-]?the[\s-]?money\b|\batm\b")
_OTM_WITH_COUNT = re.compile(r"\b(?P<n>\d+)\s*strikes?\s*(?:out\s*of\s*the\s*money|otm)\b")
_ITM_WITH_COUNT = re.compile(r"\b(?P<n>\d+)\s*strikes?\s*(?:in\s*the\s*money|itm)\b")
_BARE_OTM = re.compile(r"\botm\b")
_BARE_ITM = re.compile(r"\bitm\b")

# A fixed strike price ("25000 call", "strike 48000") or a points offset from
# spot ("200 points below spot") -- both need a listed strike resolved by
# price, which this platform cannot do yet. 3-6 digits keeps this well clear
# of the small counts used in "2 strikes OTM".
_FIXED_STRIKE = re.compile(
    r"\b(?P<strike>\d{3,6})\s*(?:call|put|ce|pe|strike)\b|\bstrike\s*(?:of\s*)?(?P<strike2>\d{3,6})\b"
)
_POINTS_OFFSET = re.compile(
    r"\b\d+(?:\.\d+)?\s*points?\s*(?:below|above|away\s*from|out|beyond|from\s*spot)\b"
)

_EXPIRY_NEXT_WEEKLY = re.compile(r"\bnext\s*week(?:'?s)?\s*(?:expiry)?\b|\bnext\s*weekly\b")
_EXPIRY_WEEKLY = re.compile(
    r"\bweekly\b|\bthis\s*week'?s?\b|\bnearest\s*weekly\b|\bnearest\s*expiry\b"
)
_EXPIRY_MONTHLY = re.compile(r"\bmonthly\b|\bthis\s*month'?s?\b")

# "Stop 30%, target 60% **on the premium**" -- the word that tells us the
# percentages are of the option's own price, not the index. It need not sit
# next to either number: one mention covers the whole exit clause, because
# that is how people actually write it.
_PREMIUM_WORD = re.compile(
    r"\b(?:on|of)\s+(?:the\s+)?(?:option'?s?\s+price|option\s+price|premiums?(?:\s+collected)?)\b"
)


def _blank(text: str, pattern: re.Pattern) -> str:
    return pattern.sub(lambda m: " " * len(m.group()), text)


# -------------------------------------------------------------------- result


@dataclass
class OptionReading:
    is_option: bool = False
    question: object | None = None  # a rules.Question, set when `is_option` and nothing else is
    kwargs: dict = field(default_factory=dict)  # option_type, strike_offset, expiry
    text: str = ""


def _question(text: str, why: str, field_name: str, *, choices=None, answerable: bool = False):
    from nlt.translate.rules import Question

    return Question(
        text=text,
        why=why,
        suggestion=None,
        field=field_name,
        choices=choices or [],
        answerable=answerable,
    )


def _multi_leg_question():
    return _question(
        "Selling options and trades with more than one option (straddles, spreads, condors) "
        "are coming in the next update. Today you can test buying one call or one put, e.g. "
        "'On NIFTY 15-minute bars, buy the weekly ATM call when RSI crosses above 60. "
        "Stop 30%, target 60% on the premium.'",
        why=(
            "Selling or combining options needs margin and risk rules this platform does not "
            "model yet -- testing it as a single bought option would be a different trade "
            "with a different, misleadingly confident-looking result."
        ),
        field_name="instrument.trade_as",
    )


def _fixed_strike_question():
    return _question(
        "Trading one exact strike price, or a strike picked by points away from spot, is "
        "coming later. Today you can ask for the at-the-money option, or a number of strikes "
        "out of or into the money, e.g. '2 strikes OTM'.",
        why=(
            "Resolving a price level or a points offset into the strike that was actually "
            "listed on that day needs the option chain, which this platform does not yet "
            "look up."
        ),
        field_name="instrument.strike_offset",
    )


def unsupported_underlying_question(symbol: str | None):
    name = symbol or "that stock"
    return _question(
        f"I can only test options on NIFTY, BANKNIFTY or SENSEX right now -- not {name}. "
        "Try the same idea on one of those.",
        why="Stock options are not wired in yet; trading a different underlying would test a different strategy.",
        field_name="instrument.symbol",
    )


def sensex_non_option_question():
    return _question(
        "SENSEX can only be tested as an option right now, e.g. 'buy the SENSEX weekly ATM "
        "call when...'. The index itself and SENSEX futures aren't supported yet.",
        why="Backtesting SENSEX itself or its futures would be a different instrument from the one asked for.",
        field_name="instrument.symbol",
    )


def _expiry_question():
    from nlt.translate.rules import Choice

    return _question(
        "Which expiry should this trade -- the nearest weekly, next week's, or the monthly?",
        why="Each expiry is a different contract with a different price; guessing one would test a trade you didn't describe.",
        field_name="instrument.expiry",
        choices=[
            Choice("Nearest weekly", "nearest_weekly"),
            Choice("Next weekly", "next_weekly"),
            Choice("Monthly", "monthly"),
        ],
    )


def _strike_question():
    from nlt.translate.rules import Choice

    return _question(
        "Which strike should this trade -- at-the-money, or a number of strikes out of or "
        "into the money?",
        why="A strike changes what is actually bought; guessing one would test a trade you didn't describe.",
        field_name="instrument.strike_offset",
        choices=[
            Choice("At-the-money", "0"),
            Choice("1 strike OTM", "1"),
            Choice("1 strike ITM", "-1"),
        ],
        answerable=True,
    )


def _type_question():
    from nlt.translate.rules import Choice

    return _question(
        "Should this buy a call or a put?",
        why="Guessing between a call and a put would test the opposite side of the market from the one meant.",
        field_name="instrument.option_type",
        choices=[Choice("Call", "CE"), Choice("Put", "PE")],
    )


def _candle_size_question():
    from nlt.translate.rules import Choice

    return _question(
        "What candle size should this option strategy use?",
        why="An option's premium moves very differently depending on how often the signal is checked; guessing a candle size would test a different strategy.",
        field_name="instrument.timeframe",
        choices=[
            Choice("5-minute", "5m"),
            Choice("15-minute", "15m"),
            Choice("Daily", "1d"),
        ],
        answerable=True,
    )


def read(text: str, answers: dict[str, str] | None = None) -> OptionReading:
    """Reads everything option-specific out of `text`.

    `text` is expected to already be lower-cased, with the underlying's own
    words (NIFTY/BANKNIFTY/SENSEX) blanked out by the caller -- this module
    only deals in option words, never the instrument name.

    `answers` is the user's picks from a previous question (e.g. pressing
    "Call" on the call-or-put button); each is tried before the matching
    question is raised again, the same way `rules.parse` resolves "exit.stop_pct".
    """
    answers = answers or {}
    reading = OptionReading()

    is_option_sentence = bool(
        _UNAMBIGUOUS_MULTI_LEG.search(text)
        or _CALL_AND_PUT.search(text)
        or _OPTION_TRIGGER.search(text)
        or (_WEAK_MULTI_LEG.search(text) and _OPTION_TRIGGER.search(text))
    )
    if not is_option_sentence:
        return reading

    reading.is_option = True

    is_multi_leg = bool(
        _UNAMBIGUOUS_MULTI_LEG.search(text)
        or _CALL_AND_PUT.search(text)
        or (_WEAK_MULTI_LEG.search(text) and _OPTION_TRIGGER.search(text))
    )
    is_sell = bool(_SELL_LEAD.match(text)) and bool(_OPTION_TRIGGER.search(text))
    if is_multi_leg or is_sell:
        reading.question = _multi_leg_question()
        return reading

    # A fixed strike or a points offset -- refused by name before strike/expiry
    # extraction even runs, since neither can be turned into a strike offset.
    if _POINTS_OFFSET.search(text) or (
        _FIXED_STRIKE.search(text)
        and not (_ATM_WORD.search(text) or _BARE_OTM.search(text) or _BARE_ITM.search(text))
    ):
        reading.question = _fixed_strike_question()
        return reading

    out_text = text

    has_call = bool(_CALL_WORD.search(text))
    has_put = bool(_PUT_WORD.search(text))
    if has_call and has_put:
        # Should already have been caught by `_CALL_AND_PUT`/multi-leg above;
        # kept as a defensive backstop against a phrasing those miss.
        reading.question = _multi_leg_question()
        return reading
    elif has_call:
        reading.kwargs["option_type"] = "CE"
        out_text = _blank(out_text, _CALL_WORD)
    elif has_put:
        reading.kwargs["option_type"] = "PE"
        out_text = _blank(out_text, _PUT_WORD)
    else:
        picked = answers.get("instrument.option_type", "").strip().upper()
        if picked in ("CE", "PE"):
            reading.kwargs["option_type"] = picked
            # Blank only the specific "option"/"options" word that made this
            # ambiguous, not every occurrence -- a later, unrelated "option's
            # price" (see `consume_premium_words`) must survive this.
            m = _OPTION_TRIGGER.search(out_text)
            if m:
                out_text = out_text[: m.start()] + " " * (m.end() - m.start()) + out_text[m.end() :]
        else:
            reading.question = _type_question()
            return reading

    m = _OTM_WITH_COUNT.search(out_text)
    if m:
        reading.kwargs["strike_offset"] = int(m.group("n"))
        out_text = out_text[: m.start()] + " " * (m.end() - m.start()) + out_text[m.end() :]
    else:
        m = _ITM_WITH_COUNT.search(out_text)
        if m:
            reading.kwargs["strike_offset"] = -int(m.group("n"))
            out_text = out_text[: m.start()] + " " * (m.end() - m.start()) + out_text[m.end() :]
        elif _ATM_WORD.search(out_text):
            reading.kwargs["strike_offset"] = 0
            out_text = _blank(out_text, _ATM_WORD)
        elif _BARE_OTM.search(out_text):
            reading.kwargs["strike_offset"] = 1
            out_text = _blank(out_text, _BARE_OTM)
        elif _BARE_ITM.search(out_text):
            reading.kwargs["strike_offset"] = -1
            out_text = _blank(out_text, _BARE_ITM)
        else:
            picked = answers.get("instrument.strike_offset", "").strip()
            try:
                offset = int(float(picked))
            except ValueError:
                offset = None
            if offset is not None and -10 <= offset <= 10:
                reading.kwargs["strike_offset"] = offset
            else:
                reading.question = _strike_question()
                return reading

    if _EXPIRY_NEXT_WEEKLY.search(out_text):
        reading.kwargs["expiry"] = "next_weekly"
        out_text = _blank(out_text, _EXPIRY_NEXT_WEEKLY)
    elif _EXPIRY_WEEKLY.search(out_text):
        reading.kwargs["expiry"] = "nearest_weekly"
        out_text = _blank(out_text, _EXPIRY_WEEKLY)
    elif _EXPIRY_MONTHLY.search(out_text):
        reading.kwargs["expiry"] = "monthly"
        out_text = _blank(out_text, _EXPIRY_MONTHLY)
    else:
        picked = answers.get("instrument.expiry", "").strip()
        if picked in ("nearest_weekly", "next_weekly", "monthly"):
            reading.kwargs["expiry"] = picked
        else:
            reading.question = _expiry_question()
            return reading

    reading.text = out_text
    reading.kwargs["trade_as"] = "option"
    return reading


def premium_mentioned(text: str) -> bool:
    """Whether the sentence says its stop/target is a percentage of the premium."""
    return bool(_PREMIUM_WORD.search(text))


def consume_premium_words(text: str) -> tuple[str, bool]:
    """Strips "on/of the premium"/"option's price" wording, saying whether it was there.

    Called once the stop/target percentages have already been pulled out of
    the sentence, since it is common for one mention to cover both ("stop 30%,
    target 60% on the premium") rather than sitting next to either number.
    """
    m = _PREMIUM_WORD.search(text)
    if not m:
        return text, False
    text = _blank(text, _PREMIUM_WORD)
    return text, True
