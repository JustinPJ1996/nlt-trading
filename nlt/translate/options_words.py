"""Option-specific vocabulary: type, strike, expiry, legs, and what is refused.

Two shapes are read:

* **Buying one call or one put** (Phase 2 batch 1) -- `read` returns
  `option_type` / `strike_offset` / `expiry` for `Instrument`.
* **Selling, and positions of up to four options** (batch 2) -- a sold call or
  put, straddles, strangles, iron condors, iron butterflies, and positions
  written leg by leg ("sell the weekly put 200 points below spot and buy the
  put 400 points below"). `read` returns `legs`; `read_position_exits` and
  `read_position_timing` read the words only such a position has (a stop on
  each leg, "50% of the credit received", re-entry, "at 09:20", "on expiry
  day", "IV percentile above 70", "hold it to expiry").

Everything else -- a fixed strike price, a debit or calendar spread, buying
two options together, a ratio -- is refused by name here, before the sentence
ever reaches the AI reader. The AI reader never sees a leg position at all:
legs, their strikes and their exits come only from this module.

Kept deliberately separate from `rules.py`: this is the one place that knows
what "ATM", "2 strikes OTM", "200 points out" and "monthly expiry" mean, and it
is testable on its own. `Question`/`Choice` live in `rules.py`; this module
imports them lazily inside functions (never at module scope) so the two
modules can import each other without a circular-import error -- `rules.py`
imports this module at the top of the file.
"""

from __future__ import annotations

import datetime as dt
import math
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
    kwargs: dict = field(default_factory=dict)  # option_type, strike_offset, expiry -- or legs
    text: str = ""
    # A leg position whose expiry we inferred ("on expiry day" -> the nearest
    # weekly) rather than read; the readback labels it "We worked out".
    expiry_worked_out: bool = False


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
        "I can't read that option position yet. What I can read: buying one call or "
        "put; selling one call or put; a straddle or strangle; an iron condor or iron "
        "butterfly; or a position written leg by leg, e.g. 'At 09:30, sell the NIFTY weekly "
        "put 200 points below spot and buy the put 400 points below. Exit at 50% of the "
        "credit received or at 15:15.' Bought straddles and strangles, debit spreads, "
        "calendars ('sell this week's ATM call and buy next week's') and ratios ('buy 1 ... "
        "and sell 2 ...') read when each option is written out like that.",
        why=(
            "Testing a different combination of options from the one you meant would give a "
            "different, misleadingly confident-looking result."
        ),
        field_name="instrument.trade_as",
    )


def _fixed_strike_question():
    return _question(
        "That strike can't be read. Say the at-the-money option, a number of strikes out of "
        "or into the money ('2 strikes OTM'), or a full index strike ('the 24000 call'). "
        "A bought option can't yet be picked by points away from spot; options you sell can "
        "('sell the put 200 points below spot').",
        why="Picking a strike from a number we are not sure of would test a different trade.",
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


class _Ask(Exception):
    """Raised inside the leg reader to stop and return a question."""

    def __init__(self, question):
        super().__init__(getattr(question, "text", ""))
        self.question = question


def _plain_question(text: str, why: str, field_name: str = "instrument.legs"):
    return _question(text, why=why, field_name=field_name)


# ---------------------------------------------------------------- leg clauses
#
# One leg clause is a verb, an optional article, an optional expiry word, an
# optional strike phrase, what is traded, and an optional strike phrase after
# it: "sell the weekly ATM straddle", "buy the put 400 points below". Every
# piece must sit side by side -- a word in between that is not in this
# vocabulary breaks the match, and that word then reaches the "I didn't
# understand" question instead of being dropped.

_VERB = r"(?P<verb>sell|short|write|buy)"
_ART = r"(?:(?:the|a|an)\s+)?"
_EXPW = (
    r"(?:nearest\s+weekly|next\s+weekly|next\s+week'?s?|this\s+week'?s?|weekly|monthly|"
    r"this\s+month'?s?)"
)
_ATM_P = r"(?:atm|at[\s-]?the[\s-]?money)"
_STRIKES_N = r"\d+\s*strikes?\s*(?:otm|itm|out\s+of\s+the\s+money|in\s+the\s+money)"
_POINTS_P = (
    r"\d+\s*points?\s*(?:below|above|out\s+of\s+the\s+money|out|otm|away)"
    r"(?:\s+(?:from\s+)?(?:the\s+)?(?:spot|index))?"
)
# A strike the user named: "sell the 25000 call" (paper trading only).
_FIXED_P = r"\d{4,6}"
_STRIKE_P = rf"(?:{_ATM_P}|{_STRIKES_N}|{_POINTS_P}|{_FIXED_P})"
_WHAT = (
    r"(?P<what>calls?\s+and\s+puts?|puts?\s+and\s+calls?|straddles?|strangles?|"
    r"iron\s+condors?|iron\s+butterfl(?:y|ies)|iron\s+fly|calls?|puts?|ce|pe)"
)
# "buy 1 ... and sell 2 ...": how many of each, for a ratio (2026-10-16).
_COUNT = r"(?:(?P<count>\d{1,2})\s+(?:lots?\s+(?:of\s+)?)?)?"
_LEG_CLAUSE = re.compile(
    rf"\b{_VERB}\s+{_COUNT}{_ART}(?:(?P<exp1>{_EXPW})\s+)?(?:(?P<pre>{_STRIKE_P})\s+)?"
    rf"(?:(?P<exp2>{_EXPW})\s+)?{_WHAT}\b(?:\s+options?\b)?(?:\s+(?P<post>{_STRIKE_P}))?"
)
_EITHER_SIDE = r"(?:on\s+either\s+side|on\s+each\s+side|either\s+side|each\s+side|away|out)"
_CONDOR_DETAIL = re.compile(
    r"\s+with\s+(?:the\s+)?short\s+strikes?\s+(?P<n>\d+)\s*points?\s*(?:out|away|otm)"
    r"\s+and\s+(?:the\s+)?wings?\s+(?P<m>\d+)\s*points?\s*(?:beyond|further\s+out|further|past)"
    r"(?:\s+(?:them|those))?\b"
)
_FLY_DETAIL = re.compile(rf"\s+with\s+(?:the\s+)?wings?\s+(?P<w>\d+)\s*points?\s*{_EITHER_SIDE}\b")
_WINGS_CLAUSE = re.compile(
    rf"\bbuy\s+(?:the\s+)?wings?\s+(?P<w>\d+)\s*points?\s*"
    rf"(?:{_EITHER_SIDE}|beyond(?:\s+(?:them|those))?)\b"
)
_CHAIN_GAP = re.compile(r"[\s,]*(?:(?:and|then)\b[\s,]*)?")
_EXPIRY_DAY_WORDS = re.compile(r"\bexpiry\s+days?\b|\bday\s+of\s+expiry\b")


def _strike_from_phrase(phrase: str, option_type: str) -> dict:
    """{"strike_offset": n} or {"strike_points": n} for one leg of `option_type`."""
    phrase = phrase.strip()
    if re.fullmatch(_ATM_P, phrase):
        return {"strike_offset": 0}
    if re.fullmatch(_FIXED_P, phrase):
        return {"strike": int(phrase)}
    m = re.fullmatch(r"(?P<n>\d+)\s*strikes?\s*(?P<way>.+)", phrase)
    if m:
        n = int(m.group("n"))
        itm = m.group("way").startswith(("itm", "in"))
        if not 0 < n <= 10:
            raise _Ask(
                _plain_question(
                    f"{n} strikes away is outside what can be tested (1 to 10).",
                    "A strike that far out is refused rather than moved closer.",
                )
            )
        return {"strike_offset": -n if itm else n}
    m = re.match(r"(?P<n>\d+)\s*points?\s*(?P<way>\w+)", phrase)
    n = int(m.group("n"))
    way = m.group("way")
    if way == "below":
        return {"strike_points": -n}
    if way == "above":
        return {"strike_points": n}
    # "out" / "otm" / "away" / "out of the money": away from the money for the leg's type.
    return {"strike_points": n if option_type == "CE" else -n}


def _is_atm(phrase: str | None) -> bool:
    return phrase is not None and bool(re.fullmatch(_ATM_P, phrase.strip()))


def _buy_only_question():
    return _question(
        "A bought iron condor or iron butterfly isn't something this platform can test "
        "yet. You can test the sold one, a bought straddle or strangle, or write each "
        "option out ('buy the ATM call and sell the call 200 points above').",
        why="Testing it as something else would be a different trade from the one described.",
        field_name="instrument.trade_as",
    )


def _is_out_phrase(strike: str | None) -> bool:
    """ "2 strikes OTM" or "200 points out/away" -- out of the money on both sides."""
    if strike is None:
        return False
    strike = strike.strip()
    if re.fullmatch(_STRIKES_N, strike):
        return not re.search(r"\bitm\b|\bin\s+the\b", strike)
    return "point" in strike and not re.search(r"\b(?:below|above)\b", strike)


def _expiry_kind(word: str | None) -> str | None:
    if not word:
        return None
    for pattern, kind in (
        (_EXPIRY_NEXT_WEEKLY, "next_weekly"),
        (_EXPIRY_WEEKLY, "nearest_weekly"),
        (_EXPIRY_MONTHLY, "monthly"),
    ):
        if pattern.search(word):
            return kind
    return None


def _legs_from_clause(m: re.Match, text: str) -> tuple[list[dict], int]:
    """The legs one clause describes, and where the clause (with its details) ends.

    Each leg also carries its clause's own expiry word and count, if it named
    them (`_expiry`, `_count`), for calendars and ratios; `_read_legs` takes
    them off again."""
    legs, end = _legs_from_clause_shape(m, text)
    expiry = _expiry_kind(m.group("exp1") or m.group("exp2"))
    count = int(m.group("count")) if m.group("count") else None
    return [{**leg, "_expiry": expiry, "_count": count} for leg in legs], end


def _legs_from_clause_shape(m: re.Match, text: str) -> tuple[list[dict], int]:
    side = "buy" if m.group("verb") == "buy" else "sell"
    what = re.sub(r"\s+", " ", m.group("what"))
    pre, post = m.group("pre"), m.group("post")
    if pre and post:
        raise _Ask(
            _plain_question(
                f'"{m.group().strip()}" names the strike twice. Say it once.',
                "Two strike rules for one option would be a guess at which one you meant.",
            )
        )
    strike = pre or post
    end = m.end()

    # A bought straddle or strangle reads (2026-10-16); a "bought iron condor"
    # is a different position from the one built below, so it is not read.
    if what.startswith("iron") and side == "buy":
        raise _Ask(_buy_only_question())

    if what.startswith("iron condor"):
        detail = _CONDOR_DETAIL.match(text, end)
        if strike or detail is None:
            raise _Ask(
                _plain_question(
                    "How far out should the iron condor's options be? Say it like 'short "
                    "strikes 200 points out and wings 100 points beyond them'.",
                    "An iron condor's result depends entirely on where its four strikes sit; "
                    "guessing them would test a trade you didn't describe.",
                )
            )
        n, wing = int(detail.group("n")), int(detail.group("m"))
        legs = [
            {"side": "sell", "option_type": "CE", "strike_points": n},
            {"side": "sell", "option_type": "PE", "strike_points": -n},
            {"side": "buy", "option_type": "CE", "strike_points": n + wing},
            {"side": "buy", "option_type": "PE", "strike_points": -(n + wing)},
        ]
        return legs, detail.end()

    if what.startswith(("iron butterfl", "iron fly")):
        detail = _FLY_DETAIL.match(text, end)
        if (strike and not _is_atm(strike)) or detail is None:
            raise _Ask(
                _plain_question(
                    "How far out should the iron butterfly's bought wings be? Say it like "
                    "'with wings 200 points on either side'.",
                    "Where the wings sit decides the worst loss; guessing it would test a "
                    "trade you didn't describe.",
                )
            )
        w = int(detail.group("w"))
        legs = [
            {"side": "sell", "option_type": "CE", "strike_points": 0},
            {"side": "sell", "option_type": "PE", "strike_points": 0},
            {"side": "buy", "option_type": "CE", "strike_points": w},
            {"side": "buy", "option_type": "PE", "strike_points": -w},
        ]
        return legs, detail.end()

    if what.startswith("straddle"):
        if not _is_atm(strike):
            raise _Ask(
                _plain_question(
                    "Which strike should the straddle use? Say e.g. 'the ATM straddle'.",
                    "A straddle's call and put share one strike; guessing it would test a "
                    "trade you didn't describe.",
                )
            )
        return [
            {"side": side, "option_type": "CE", "strike_offset": 0},
            {"side": side, "option_type": "PE", "strike_offset": 0},
        ], end

    if what.startswith("strangle"):
        if not _is_out_phrase(strike):
            raise _Ask(
                _plain_question(
                    "How far out should the strangle's call and put be? Say e.g. '2 strikes "
                    "OTM' or '200 points out'.",
                    "A strangle's strikes decide what it earns and risks; guessing them would "
                    "test a trade you didn't describe.",
                )
            )
        return [
            {"side": side, "option_type": "CE", **_strike_from_phrase(strike, "CE")},
            {"side": side, "option_type": "PE", **_strike_from_phrase(strike, "PE")},
        ], end

    types = ["CE", "PE"] if " and " in what else (["CE"] if what[0] == "c" else ["PE"])
    if strike is None:
        # Asked about only once this turns out to be a position that sells:
        # a single bought option without a strike gets batch 1's own question.
        return [{"side": side, "option_type": t, "_no_strike": True} for t in types], end
    if len(types) == 2 and strike and re.search(r"\b(?:below|above)\b", strike):
        raise _Ask(
            _plain_question(
                f'"{m.group().strip()}" puts the call and the put on the same side of the '
                "index. Write each one out, e.g. 'sell the call 200 points above and the put "
                "200 points below'.",
                "Guessing which side each belongs on would test a different position.",
            )
        )
    return [{"side": side, "option_type": t, **_strike_from_phrase(strike, t)} for t in types], end


def _wings(m: re.Match, legs: list[dict]) -> list[dict]:
    """ "buy wings 200 points on either side" -- of the sold call and the sold put."""
    sold_ce = [x for x in legs if x["side"] == "sell" and x["option_type"] == "CE"]
    sold_pe = [x for x in legs if x["side"] == "sell" and x["option_type"] == "PE"]
    ask = _Ask(
        _plain_question(
            "Wings 'on either side' need exactly one sold call and one sold put, each picked "
            "at-the-money or by points from the index. Write each bought option out instead, "
            "e.g. 'buy the call 200 points above'.",
            "Guessing which sold option a wing protects could test a different position.",
        )
    )
    if len(sold_ce) != 1 or len(sold_pe) != 1 or any("_no_strike" in x for x in legs):
        raise ask

    def points(leg: dict) -> int:
        if leg.get("strike_points") is not None:
            return leg["strike_points"]
        if leg.get("strike_offset", 0) == 0:
            return 0
        raise ask

    w = int(m.group("w"))
    return [
        {"side": "buy", "option_type": "CE", "strike_points": points(sold_ce[0]) + w},
        {"side": "buy", "option_type": "PE", "strike_points": points(sold_pe[0]) - w},
    ]


def _read_chain(text: str) -> tuple[list[dict], list[tuple[int, int]]] | None:
    """The legs of the first run of leg clauses joined by "and" / "," / "then".

    None when the sentence has no leg clause at all.
    """
    m: re.Match | None = _LEG_CLAUSE.search(text)
    if m is None:
        return None
    legs: list[dict] = []
    spans: list[tuple[int, int]] = []
    while m is not None:
        if m.re is _WINGS_CLAUSE:
            legs.extend(_wings(m, legs))
            end = m.end()
        else:
            new, end = _legs_from_clause(m, text)
            legs.extend(new)
        spans.append((m.start(), end))
        nxt = _CHAIN_GAP.match(text, end).end()
        m = _WINGS_CLAUSE.match(text, nxt) or _LEG_CLAUSE.match(text, nxt)
        if m is not None:
            spans.append((end, nxt))  # the "and" joining two clauses
    return legs, spans


def _expiries_in(text: str) -> tuple[str, set[str]]:
    """Every expiry word in the sentence, blanked, and the kinds they name."""
    kinds: set[str] = set()
    for pattern, kind in (
        (_EXPIRY_NEXT_WEEKLY, "next_weekly"),
        (_EXPIRY_WEEKLY, "nearest_weekly"),
        (_EXPIRY_MONTHLY, "monthly"),
    ):
        if pattern.search(text):
            kinds.add(kind)
            text = _blank(text, pattern)
    return text, kinds


def _normalise_units(legs: list[dict]) -> list[dict]:
    """One unit for every strike: points as soon as any leg uses points.

    `is_defined_risk` treats mixed units as unprotected (it cannot know which
    strike is further out), so an ATM leg -- the strike nearest the index --
    is written as 0 points when the rest of the position is in points. Any
    other mix is refused.
    """
    if not any(leg.get("strike_points") is not None for leg in legs):
        return legs
    out = []
    for leg in legs:
        if leg.get("strike_points") is None:
            if leg.get("strike_offset", 0) != 0:
                raise _Ask(
                    _plain_question(
                        "This mixes strikes counted from at-the-money ('2 strikes OTM') with "
                        "strikes in points from the index ('200 points out'). Use one or the "
                        "other for every option.",
                        "Which strike is further out cannot be known for sure across the two, "
                        "and that decides whether the position is protected.",
                    )
                )
            leg = {**{k: v for k, v in leg.items() if k != "strike_offset"}, "strike_points": 0}
        out.append(leg)
    return out


_NAMED_UNREAD_SHAPES = re.compile(r"\b(?:spreads?|hedg\w*|ratio|calendar|debit)\b")


def _read_legs(
    text: str, answers: dict[str, str], symbol: str | None, reading: OptionReading
) -> bool:
    """Fills `reading` for a selling / multi-leg sentence. False if this is not one."""
    from nlt.spec.models import OptionLeg

    multi_word = bool(
        _UNAMBIGUOUS_MULTI_LEG.search(text)
        or _CALL_AND_PUT.search(text)
        or (_WEAK_MULTI_LEG.search(text) and _OPTION_TRIGGER.search(text))
    )
    chain = _read_chain(text)
    if chain is None:
        if multi_word or _SELL_LEAD.match(text):
            raise _Ask(_multi_leg_question())
        return False
    legs, spans = chain
    if not any(leg["side"] == "sell" for leg in legs):
        if len(legs) == 1 and not multi_word:
            return False  # one bought option: batch 1's reader
        if len(legs) == 1:
            raise _Ask(_multi_leg_question())
        # Two or more bought options (a bought straddle or strangle, a call and
        # a put): read as a position. The most it can lose is what was paid.
    if any("_no_strike" in leg for leg in legs):
        raise _Ask(
            _plain_question(
                "Which strike should each option use? Say e.g. 'the ATM call', 'the call 2 "
                "strikes OTM' or 'the put 200 points below spot'.",
                "A strike changes what is actually traded; guessing one would test a trade "
                "you didn't describe.",
            )
        )
    if len(legs) > 4:
        raise _Ask(
            _plain_question(
                "At most four options can be tested in one position.",
                "Dropping some of the options would test a different position.",
            )
        )
    clause_expiries = [leg.pop("_expiry", None) for leg in legs]
    counts = [leg.pop("_count", None) for leg in legs]
    lots, sets = _ratio(counts)
    # Expiry words are read across the whole sentence, inside the clauses too
    # ("sell the weekly put ... and buy the put ..."): one expiry for all,
    # unless each option names its own (a calendar). Blanking keeps every
    # offset, so the clause spans stay valid.
    text, kinds = _expiries_in(text)
    for a, b in spans:
        text = text[:a] + " " * (b - a) + text[b:]
    legs = _normalise_units(legs)

    per_leg: list[str] | None = None
    if len(kinds) > 1:
        # Every option names its own expiry, and no other expiry word is left.
        if set(clause_expiries) != kinds:
            raise _Ask(
                _plain_question(
                    "This names more than one expiry. For options with different expiries, "
                    "say the expiry of each one, e.g. 'sell this week's ATM call and buy next "
                    "week's ATM call'.",
                    "Guessing which option belongs to which expiry would test a different trade.",
                )
            )
        per_leg = clause_expiries
    text = _check_named_shapes(text, legs, per_leg, lots)
    if per_leg is not None:
        expiry = None
    elif kinds:
        expiry = kinds.pop()
    else:
        picked = answers.get("instrument.expiry", "").strip()
        if picked in ("nearest_weekly", "next_weekly", "monthly"):
            expiry = picked
        elif _EXPIRY_DAY_WORDS.search(text) and symbol != "BANKNIFTY":
            # "Sell the straddle on expiry day" trades the contract expiring
            # that day: the nearest expiry, which for NIFTY and SENSEX is
            # always a weekly. Labelled "We worked out". BANKNIFTY has had no
            # weeklies since Nov 2024, so there it is asked.
            expiry = "nearest_weekly"
            reading.expiry_worked_out = True
        else:
            raise _Ask(_expiry_question())

    built = []
    for k, leg in enumerate(legs):
        try:
            built.append(
                OptionLeg(
                    expiry=per_leg[k] if per_leg is not None else expiry,
                    lots=lots[k],
                    **leg,
                )
            )
        except ValueError as exc:
            raise _Ask(
                _plain_question(
                    "One of those options can't be tested: a strike or size is outside what "
                    "can be tested.",
                    "A strike or size outside what can be tested is refused, never moved.",
                )
            ) from exc
    if len(set(built)) != len(built):
        raise _Ask(
            _plain_question(
                "The same option appears twice in that position. Write it once.",
                "Guessing whether you meant twice the size or a different option would test a "
                "different trade.",
            )
        )
    reading.kwargs = {"trade_as": "option", "legs": built}
    if sets is not None:
        # "sell 2 lots of the straddle": the size, said inside the clause.
        reading.kwargs["_sets"] = sets
    reading.text = text
    return True


def _ratio(counts: list[int | None]) -> tuple[list[int], int | None]:
    """Each option's lots per set, and how many sets, from "buy 1 ... sell 2".

    No counts: one of each, size from elsewhere. The same count on every
    option ("sell 2 lots of the straddle"): that many sets of one each.
    Different counts: a ratio, written in its smallest whole numbers -- buy 2
    and sell 4 is two sets of one-to-two. Some counted and some not is asked.
    """
    if all(c is None for c in counts):
        return [1] * len(counts), None
    if any(c is None for c in counts):
        raise _Ask(
            _plain_question(
                "Say how many of each option, e.g. 'buy 1 ATM call and sell 2 calls 200 "
                "points above'.",
                "With a count on only some of the options, the rest would be a guess.",
            )
        )
    if any(c == 0 for c in counts):
        raise _Ask(
            _plain_question(
                "Zero of an option isn't a position. Say how many of each.",
                "An option bought or sold zero times would be dropped, testing a different trade.",
            )
        )
    g = math.gcd(*counts)
    return [c // g for c in counts], g


def _check_named_shapes(text: str, legs: list[dict], per_leg, lots: list[int]) -> str:
    """A shape named outside the legs read must be the shape those legs make.

    "Calendar" needs two expiries, "ratio" unequal counts, "spread" a bought
    and a sold option, "debit" a position that pays more than it collects.
    Anything else named -- "hedge" among them -- may say more than the legs
    read, so it is asked about, never assumed to match."""
    sides = {leg["side"] for leg in legs}
    for m in list(_NAMED_UNREAD_SHAPES.finditer(text)):
        word = m.group()
        fits = (
            (word == "calendar" and per_leg is not None)
            or (word == "ratio" and len(set(lots)) > 1)
            or (word.startswith("spread") and sides == {"buy", "sell"})
            or (word == "debit" and "buy" in sides and not _credit_shape_dicts(legs))
        )
        if not fits:
            raise _Ask(_multi_leg_question())
        # Said, and true of the legs read: nothing left to read in it.
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
    return text


def _credit_shape_dicts(legs: list[dict]) -> bool:
    from nlt.spec.models import OptionLeg

    return _credit_shape([OptionLeg(**leg) for leg in legs])


def read(
    text: str, answers: dict[str, str] | None = None, symbol: str | None = None
) -> OptionReading:
    """Reads everything option-specific out of `text`.

    `text` is expected to already be lower-cased, with the underlying's own
    words (NIFTY/BANKNIFTY/SENSEX) blanked out by the caller -- this module
    only deals in option words, never the instrument name. `symbol` is the
    underlying the caller found, if any: BANKNIFTY changes what an unnamed
    expiry can mean.

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

    # A strike named as a price ("the 25000 call") reads since 2026-10-16
    # (paper trading only). A number too small to be an index strike, or one
    # beside another strike rule, is asked about.
    fixed = _FIXED_STRIKE.search(text)
    if fixed and len(fixed.group("strike") or fixed.group("strike2")) < 4:
        reading.question = _fixed_strike_question()
        return reading

    try:
        if _read_legs(text, answers, symbol, reading):
            return reading
    except _Ask as ask:
        reading.question = ask.question
        return reading
    return _read_single_buy(text, answers, reading)


def _read_single_buy(text: str, answers: dict[str, str], reading: OptionReading) -> OptionReading:
    """Batch 1: buying one call or one put."""
    is_multi_leg = bool(
        _UNAMBIGUOUS_MULTI_LEG.search(text)
        or _CALL_AND_PUT.search(text)
        or (_WEAK_MULTI_LEG.search(text) and _OPTION_TRIGGER.search(text))
    )
    is_sell = bool(_SELL_LEAD.match(text)) and bool(_OPTION_TRIGGER.search(text))
    if is_multi_leg or is_sell:
        reading.question = _multi_leg_question()
        return reading

    # A points offset from spot is read for positions that sell, but the
    # single bought option's engine counts listed strikes only.
    if _POINTS_OFFSET.search(text):
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

    # Read on the words as typed: the call/put word after the number has
    # already been blanked from `out_text`, but every offset is kept.
    fixed = _FIXED_STRIKE.search(text)
    if fixed:
        if _ATM_WORD.search(out_text) or _BARE_OTM.search(out_text) or _BARE_ITM.search(out_text):
            reading.question = _plain_question(
                "That names the strike twice -- as a price and as at/out of/in the money. "
                "Say it once.",
                "Two strike rules for one option would be a guess at which one you meant.",
            )
            return reading
        reading.kwargs["strike"] = int(fixed.group("strike") or fixed.group("strike2"))
        out_text = (
            out_text[: fixed.start()]
            + " " * (fixed.end() - fixed.start())
            + out_text[fixed.end() :]
        )
    m = None if fixed else _OTM_WITH_COUNT.search(out_text)
    if fixed:
        pass
    elif m:
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


# --------------------------------------------------- a position's own exits
#
# Only for a position read as legs (`read` returned `legs`). Each phrase below
# says *what* a percentage is measured on; where the words do not say -- "stop
# at 30% of the premium collected" on a straddle could be each option's own
# price or the total -- the reader asks, with buttons, never picks.

_STOPW = r"(?:stop(?:[\s-]*loss)?|sl)"
_PCT = r"(?P<v>\d+(?:\.\d+)?)\s*%"
_EACH = (
    r"(?:(?:on|for|per)\s+(?:each|every|both|all)\s+(?:of\s+the\s+)?(?P<sold>sold\s+)?"
    r"(?:legs?|sides?|options?)\b|\bper\s+leg\b)"
)
_PER_LEG_STOP = [
    re.compile(rf"(?:\bwith\s+)?(?:\ban?\s+)?{_PCT}\s*{_STOPW}\s*{_EACH}"),
    re.compile(
        rf"\b{_STOPW}\s*(?:of|at)?\s*{_PCT}\s*(?:(?:of|on)\s+(?:the\s+)?premium\s+)?{_EACH}"
    ),
    re.compile(
        rf"\b{_STOPW}\s*(?:of|at)?\s*{_PCT}\s*of\s+each\s+(?P<sold>sold\s+)?(?:leg|option)'?s?"
        r"\s*(?:own\s+)?(?:premium|price)\b"
    ),
]
_COMBINED_STOP = [
    re.compile(
        rf"\b{_STOPW}\s*(?:of|at)?\s*{_PCT}\s*of\s+(?:the\s+)?(?:total|combined|net|whole)\s+"
        r"(?:premiums?|credit)(?:\s+(?:collected|received))?\b"
    ),
    re.compile(rf"\b{_STOPW}\s*(?:of|at)?\s*{_PCT}\s*of\s+(?:the\s+)?credit(?:\s+received)?\b"),
]
# A position of bought options only: "N% of the premium paid" is on the total.
_PAID = r"(?:the\s+)?(?:total\s+|combined\s+|net\s+)?(?:premiums?\s+paid|debit(?:\s+paid)?)\b"
_PAID_STOP = re.compile(rf"\b{_STOPW}\s*(?:of|at)?\s*{_PCT}\s*(?:of|on)\s+{_PAID}")
_PAID_TARGET = re.compile(
    r"\b(?:exit|take\s+profit|book\s+profit|close(?:\s+the\s+position)?|target)\s*(?:at)?\s*"
    rf"{_PCT}\s*(?:profit\s+)?(?:of|on)\s+{_PAID}"
)
# On more than one sold option these do not say each-or-total.
_PREMIUM_STOP = [
    re.compile(
        rf"\b{_STOPW}\s*(?:of|at)?\s*{_PCT}\s*(?:of|on)\s+(?:the\s+)?premiums?"
        r"(?:\s+(?:collected|received))?\b"
    ),
    re.compile(
        rf"{_PCT}\s*{_STOPW}\s*(?:of|on)\s+(?:the\s+)?premiums?(?:\s+(?:collected|received))?\b"
    ),
]
_PER_LEG_TARGET = [
    re.compile(rf"\b(?:target|take\s+profit)\s*(?:of|at)?\s*{_PCT}\s*{_EACH}"),
    re.compile(rf"{_PCT}\s*(?:target|take\s+profit)\s*{_EACH}"),
]
_TIME_TAIL = r"(?:\s*,)?\s+or\s+(?:at\s+)?(?P<h>\d{1,2}):(?P<m>\d{2})(?:\s*(?P<ampm>am|pm)\b)?"
_COMBINED_TARGET = re.compile(
    r"\b(?:exit|take\s+profit|book\s+profit|close(?:\s+the\s+position)?|target)\s*"
    r"(?:at|when\s+(?:the\s+position\s+)?(?:has\s+)?(?:made|kept|captured|reache[sd]|reach))?\s*"
    rf"{_PCT}\s*of\s+(?:the\s+)?(?:credit(?:\s+received)?|net\s+credit|total\s+credit|"
    rf"max(?:imum)?\s+profit)\b(?:{_TIME_TAIL})?"
)
_REENTRY_KIND = (
    r"(?:at\s+(?:cost|the\s+same\s+price|the\s+original\s+price|the\s+entry\s+price)"
    r"|(?:immediately\s+)?at\s+the\s+(?:new|fresh|current)\s+atm(?:\s+strike)?"
    r"|(?:immediately\s+)?(?:with|at)\s+a\s+(?:new|fresh)\s+atm(?:\s+strike)?)"
)
_REENTRY_COUNT = r"once|twice|[1-5]\s*times?"
_REENTRY = re.compile(
    r"(?:,\s*)?(?:\band\s+)?\bre[\s-]?enter\s+(?:(?:a|the|each|that|any)\s+)?"
    r"(?:(?:sold\s+)?(?:legs?|sides?|options?)\s+)?"
    rf"(?:(?P<count1>{_REENTRY_COUNT})\s+)?(?:(?P<kind1>{_REENTRY_KIND})\s+)?"
    rf"(?:(?P<count2>{_REENTRY_COUNT})\s+)?"
    r"(?:if|when|after)\s+(?:(?:it|that\s+leg|the\s+leg|that\s+option)\s+)?"
    r"(?:(?:gets|is|has\s+been)\s+)?(?:stopped\s+out|hits?\s+(?:its|the)\s+stop(?:[\s-]*loss)?)"
    rf"(?:\s*,?\s*(?P<kind2>{_REENTRY_KIND}))?"
)
_SOLD_STOP_CHOICES = ["20", "30", "50"]


def _choices(pairs):
    from nlt.translate.rules import Choice

    return [Choice(label, value) for label, value in pairs]


def sold_stop_question():
    """The stop-loss question for a sold option with nothing protecting it."""
    return _question(
        "What stop loss should each sold option have? It closes that option when its price "
        "rises by this much above its price when the position opens.",
        why=(
            "A sold option with no bought one protecting it can lose far more than it "
            "collected -- we never assume a stop loss for you."
        ),
        field_name="exit.stop_pct",
        choices=_choices([(f"{v}%", v) for v in _SOLD_STOP_CHOICES]),
        answerable=True,
    )


def _paid_target_question(v: float):
    return _question(
        f"Should the {v:g}% target be measured on each option's own price, or on the total "
        "premium paid?",
        why=(
            "They close at different times: one option can be up "
            f"{v:g}% while the two together are still at a loss."
        ),
        field_name="exit.target_basis",
        choices=_choices(
            [
                ("On each option's own price", "each_leg"),
                ("On the total premium paid", "combined"),
            ]
        ),
    )


def _stop_basis_question(v: float, *, paid: bool = False):
    if paid:
        return _question(
            f"Should the {v:g}% stop be measured on each option's own price, or on the "
            "total premium paid?",
            why=(
                "They close at different times: one option can fall "
                f"{v:g}% while the two together are still in profit."
            ),
            field_name="exit.stop_basis",
            choices=_choices(
                [
                    ("On each option's own price", "each_leg"),
                    ("On the total premium paid", "combined"),
                ]
            ),
        )
    return _question(
        f"Should the {v:g}% stop be measured on each sold option's own price, or on the "
        "total premium collected?",
        why=(
            "They close the position at different times: one option can rise "
            f"{v:g}% while the position as a whole is still in profit."
        ),
        field_name="exit.stop_basis",
        choices=_choices(
            [
                ("On each sold option's own price", "each_leg"),
                ("On the total premium collected", "combined"),
            ]
        ),
    )


def _all_atm(legs) -> bool:
    return all(
        (leg.strike_points == 0) if leg.strike_points is not None else leg.strike_offset == 0
        for leg in legs
    )


def _reentry_question(legs):
    fresh = (
        "Sell a fresh at-the-money option at the next candle"
        if _all_atm(legs)
        else "Sell a fresh option at the next candle, its strike picked again the same way"
    )
    return _question(
        "After a sold option's stop loss closes it, how should it be re-entered?",
        why=(
            "Re-selling the same option later and selling a fresh one straight away are "
            "different trades with different results; guessing would test one you didn't mean."
        ),
        field_name="legs.reentry",
        choices=_choices(
            [
                (
                    "Re-sell the same option when its price comes back to where we first sold it",
                    "same_strike_at_cost",
                ),
                (fresh, "fresh_strike_now"),
            ]
        ),
    )


def _reentry_count_question():
    return _question(
        "How many times may each sold option be re-entered after its stop loss?",
        why="Each re-entry is another trade; guessing how many would test a different strategy.",
        field_name="legs.reentry_max",
        choices=_choices([("Once", "1"), ("Twice", "2")]),
    )


def _credit_shape(legs) -> bool:
    """Every bought option is paid for by a sold one: same type and expiry, same
    lots, further out -- so the position takes in more than it pays out."""
    from nlt.spec.models import _further_out

    sells = [leg for leg in legs if leg.side == "sell"]
    for b in (leg for leg in legs if leg.side == "buy"):
        match = next(
            (
                s
                for s in sells
                if s.option_type == b.option_type
                and s.expiry == b.expiry
                and s.lots == b.lots
                and _further_out(b, s)
            ),
            None,
        )
        if match is None:
            return False
        sells.remove(match)
    return True


def _clock(h: str, m: str, ampm: str | None) -> dt.time | None:
    hour, minute = int(h), int(m)
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return dt.time(hour, minute)


@dataclass
class PositionExits:
    text: str
    legs: list  # OptionLeg, with their stops / targets / re-entry set
    combined_target_pct: float | None = None
    combined_stop_pct: float | None = None
    square_off: dt.time | None = None  # "... or at 15:15"
    stop_source: str | None = None  # readback label for the stop, when one was read
    target_source: str | None = None
    reentry_said: bool = False
    question: object | None = None


def _consume_first(text: str, patterns) -> tuple[str, re.Match | None]:
    for pattern in patterns:
        m = pattern.search(text)
        if m:
            return text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :], m
    return text, None


def read_position_exits(text: str, legs: list, answers: dict[str, str]) -> PositionExits:
    """Stops, targets and re-entry for a leg position, each measured on what the words say.

    The plain stock-style patterns ("stop 30%", "target 50%") are read here
    too, so nothing reaches `ExitRules`' own stop/target -- which a leg
    position must not use.
    """
    from nlt.translate.readback import SAID, WORKED_OUT
    from nlt.translate.rules import _extract_percent_exits

    try:
        return _read_position_exits(text, legs, answers, SAID, WORKED_OUT, _extract_percent_exits)
    except _Ask as ask:
        return PositionExits(text=text, legs=legs, question=ask.question)


def _read_position_exits(text, legs, answers, said, worked_out, percent_exits):
    out = PositionExits(text=text, legs=legs)
    has_bought = any(leg.side == "buy" for leg in legs)
    # Bought options only (a bought straddle or strangle): "each leg" can only
    # mean each bought option, and the whole is the premium paid.
    bought_only = not any(leg.side == "sell" for leg in legs)
    mixed = has_bought and not bought_only
    single = len(legs) == 1
    credit = _credit_shape(legs)

    def ask(message: str):
        raise _Ask(
            _plain_question(
                message,
                "Guessing what a percentage is measured on would test a different trade.",
                "exit.rule",
            )
        )

    stops: list[tuple[str, float, str]] = []  # (basis, value, label)
    text, m = _consume_first(text, _PER_LEG_STOP)
    if m:
        if mixed and not m.group("sold"):
            ask(
                "Should the stop on each leg apply to the sold options only? Say 'a "
                f"{m.group('v')}% stop on each sold option' if so."
            )
        stops.append(("each_leg", float(m.group("v")), said))
    text, m = _consume_first(text, _COMBINED_STOP)
    if m:
        stops.append(("combined", float(m.group("v")), said))
    text, m = _consume_first(text, [_PAID_STOP])
    if m:
        if not bought_only:
            ask("'Of the premium paid' fits a position that only buys options; this one sells too.")
        stops.append(("combined", float(m.group("v")), said))
    text, m = _consume_first(text, _PREMIUM_STOP)
    if m:
        stops.append(("ambiguous", float(m.group("v")), said))

    text, m = _consume_first(text, [_COMBINED_TARGET])
    combined_target = None
    if m:
        combined_target = float(m.group("v"))
        if m.group("h"):
            out.square_off = _clock(m.group("h"), m.group("m"), m.group("ampm"))
            if out.square_off is None:
                ask(f'"{m.group().strip()}" has a time that does not exist.')
    text, m = _consume_first(text, [_PAID_TARGET])
    if m:
        if not bought_only:
            ask("'Of the premium paid' fits a position that only buys options; this one sells too.")
        combined_target = float(m.group("v"))
    text, m = _consume_first(text, _PER_LEG_TARGET)
    leg_target = float(m.group("v")) if m else None
    if m and mixed and not m.group("sold"):
        ask(
            "Should the target on each leg apply to the sold options only? Say 'a "
            f"{m.group('v')}% target on each sold option' if so."
        )

    text, plain = percent_exits(text)
    text, premium_said = consume_premium_words(text)
    if "stop_pct" in plain:
        label = said if premium_said else worked_out
        stops.append(("ambiguous", float(plain["stop_pct"]), label))
    plain_target = plain.get("target_pct")

    if len({(basis, v) for basis, v, _ in stops}) > 1:
        ask("That gives more than one stop loss. Say just one, e.g. 'a 25% stop on each leg'.")

    leg_stop = None
    if stops:
        basis, value, label = stops[0]
        if basis == "ambiguous":
            if single:
                basis = "each_leg"
            else:
                picked = answers.get("exit.stop_basis", "").strip()
                if picked not in ("each_leg", "combined"):
                    raise _Ask(_stop_basis_question(value, paid=bought_only))
                basis, label = picked, said
        if basis == "combined":
            if not (credit or bought_only):
                ask(
                    "A stop on the total premium collected needs a position that collects "
                    "a premium; this one may pay more than it collects."
                )
            out.combined_stop_pct = value
        else:
            leg_stop = value
        out.stop_source = label

    if plain_target is not None and bought_only and not single:
        picked = answers.get("exit.target_basis", "").strip()
        if picked not in ("each_leg", "combined"):
            raise _Ask(_paid_target_question(plain_target))
        if picked == "combined":
            combined_target = plain_target
        else:
            leg_target = plain_target
        plain_target = None
        out.target_source = said
    if plain_target is not None:
        if not single:
            ask(
                f"Is the {plain_target:g}% target on the total credit received ('exit at "
                f"{plain_target:g}% of the credit received') or on each option ('a "
                f"{plain_target:g}% target on each sold option')?"
            )
        leg_target = plain_target
        out.target_source = said if premium_said else worked_out
    elif leg_target is not None:
        out.target_source = said
    if combined_target is not None:
        if not (credit or bought_only):
            ask(
                "'Of the credit received' needs a position that collects a premium; this one "
                "may pay more than it collects."
            )
        out.combined_target_pct = combined_target
        out.target_source = said

    new_legs = []
    for leg in legs:
        update = {}
        if bought_only:
            if leg_stop is not None:
                if leg_stop >= 100:
                    ask("A bought option's price can fall at most 100%.")
                update["stop_pct"] = leg_stop
            if leg_target is not None:
                update["target_pct"] = leg_target
        elif leg.side == "sell":
            if leg_stop is not None:
                update["stop_pct"] = leg_stop
            if leg_target is not None:
                if leg_target > 100:
                    ask("A sold option's price can fall at most 100%.")
                update["target_pct"] = leg_target
        new_legs.append(leg.model_copy(update=update) if update else leg)
    legs = new_legs

    m = _REENTRY.search(text)
    if m:
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
        stopped = [leg for leg in legs if leg.side == "sell" and leg.stop_pct is not None]
        if not stopped:
            ask(
                "Re-entering needs a stop loss on each sold option to say when it was "
                "stopped out, e.g. 'a 25% stop on each leg'."
            )
        words = m.group("kind1") or m.group("kind2")
        if words:
            kind = (
                "same_strike_at_cost"
                if re.search(r"cost|same|original|entry", words)
                else "fresh_strike_now"
            )
            if kind == "fresh_strike_now" and not _all_atm(stopped):
                ask(
                    "'A fresh ATM' only fits options sold at-the-money. Say 'sell a fresh "
                    "one at the next candle' or 're-enter at cost'."
                )
        else:
            kind = answers.get("legs.reentry", "").strip()
            if kind not in ("same_strike_at_cost", "fresh_strike_now"):
                raise _Ask(_reentry_question(stopped))
        count_words = m.group("count1") or m.group("count2")
        if count_words:
            count = {"once": 1, "twice": 2}.get(count_words) or int(count_words[0])
        else:
            picked = answers.get("legs.reentry_max", "").strip()
            if picked not in ("1", "2"):
                raise _Ask(_reentry_count_question())
            count = int(picked)
        legs = [
            leg.model_copy(update={"reentry": kind, "reentry_max": count})
            if leg in stopped
            else leg
            for leg in legs
        ]
        out.reentry_said = True

    out.text = text
    out.legs = legs
    return out


# --------------------------------------------------- when a position opens

_AT_TIME = re.compile(r"\b(?:at|from)\s+(?P<h>\d{1,2}):(?P<m>\d{2})(?:\s*(?P<ampm>am|pm)\b)?")
_ON_EXPIRY_DAY = re.compile(
    r"(?:\bonly\s+)?\bon\s+(?:the\s+)?(?:expiry\s+days?|days?\s+of\s+expiry)\b"
)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday")
_WEEKDAY = re.compile(
    r"(?:\bonly\s+)?\b(?:every|each|on)\s+(?P<d>monday|tuesday|wednesday|thursday|friday)s?\b"
)
_IV_PERCENTILE = re.compile(
    r"(?:\bonly\s+)?(?:\bon\s+days\s+)?(?:\b(?:when|if|where)\s+)?(?:\bthe\s+)?"
    r"\b(?:iv\s*percentile|ivp)\s+(?:is\s+)?"
    r"(?P<op>above|over|greater\s+than|more\s+than|higher\s+than|>|below|under|less\s+than|"
    r"lower\s+than|<)\s*(?P<v>\d+(?:\.\d+)?)\b"
)
_HOLD_TO_EXPIRY = re.compile(
    r"(?:\band\s+)?\b(?:hold|holding|keep|carry)\s+(?:(?:it|them|the\s+position|the\s+trade|"
    r"positions?)\s+)?(?:to|till|until|into|through)\s+(?:the\s+)?expiry(?:\s+day)?\b"
)


@dataclass
class PositionTiming:
    text: str
    conditions: list = field(default_factory=list)  # AtTime / OnExpiryDay / IvPercentile
    weekdays: list[int] | None = None
    hold_to_expiry: bool = False
    question: object | None = None


def read_position_timing(text: str) -> PositionTiming:
    """ "At 09:20", "every Monday", "on expiry day", "IV percentile above 70", "hold to expiry".

    Run after the square-off and exit phrases have been taken out, so the
    only clock time left is the entry's.
    """
    from nlt.spec.models import AtTime, IvPercentile, OnExpiryDay

    out = PositionTiming(text=text)

    def take(pattern):
        m = pattern.search(out.text)
        if m:
            out.text = out.text[: m.start()] + " " * (m.end() - m.start()) + out.text[m.end() :]
        return m

    def refuse(message: str) -> PositionTiming:
        out.question = _plain_question(
            message,
            "Guessing when to open the position would test a different trade.",
            "entry.condition",
        )
        return out

    if take(_HOLD_TO_EXPIRY):
        out.hold_to_expiry = True

    times = list(_AT_TIME.finditer(out.text))
    if len(times) > 1:
        return refuse("That gives more than one time to open the position. Say just one.")
    if times:
        m = take(_AT_TIME)
        clock = _clock(m.group("h"), m.group("m"), m.group("ampm"))
        try:
            if clock is None:
                raise ValueError
            out.conditions.append(AtTime(time=clock))
        except ValueError:
            return refuse(
                f'"{m.group().strip()}" is not a time inside the 09:15-15:30 market session.'
            )

    days = list(_WEEKDAY.finditer(out.text))
    if len(days) > 1:
        return refuse("Only one weekday can be read for now, e.g. 'every Monday'.")
    if days:
        m = take(_WEEKDAY)
        out.weekdays = [_WEEKDAYS.index(m.group("d"))]

    if take(_ON_EXPIRY_DAY):
        out.conditions.append(OnExpiryDay())

    m = take(_IV_PERCENTILE)
    if m:
        value = float(m.group("v"))
        if value > 100:
            return refuse("An IV percentile runs from 0 to 100.")
        op = "gt" if re.match(r"above|over|greater|more|higher|>", m.group("op")) else "lt"
        out.conditions.append(IvPercentile(op=op, value=value))
    elif re.search(r"\b(?:iv|ivp|implied\s+volatility)\b", out.text):
        return refuse(
            "IV can be read as 'IV percentile above 70' (or below). Other IV rules are not "
            "read yet."
        )
    return out
