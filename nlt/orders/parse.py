"""English -> a one-off conditional order. A phrase table, no AI.

The same rule as the strategy reader: **refuse rather than guess.** Every part
of the sentence must be understood, or the person gets a question. A wrong
order that reads plausibly is the worst outcome this module can produce; a
question costs them ten seconds.

The sentence has two halves -- a condition and an action -- in either order:

    If NIFTY goes below 22000, buy 1 lot NIFTY futures
    Buy 1 lot futures if NIFTY has had 4 red weekly candles in a row
    If P&L falls below -50000, exit positions

Anything in brackets must restate or add to the condition -- "(4 consecutive
red candles)" -- and is checked against it, never ignored.

Ambiguities this module asks about instead of resolving:

  * "P&L below Rs 50,000" -- a loss of 50,000, or a profit shrinking under it?
  * "going down for 4 weeks" -- four red candles, or four lower closes?
  * "crosses / hits / reaches 22000" -- from above or below?
  * "buy 1 lot NIFTY" -- NIFTY futures, or an option? Only futures work here.
  * "candles" with no length -- daily or weekly?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from nlt.orders.models import (
    AlertAction,
    ConditionalOrder,
    ExitAction,
    PnlTrigger,
    PriceTrigger,
    StreakTrigger,
    TradeAction,
    Watched,
)
from nlt.translate.rules import _COMMODITIES, _UNSUPPORTED_COMMODITY, Question


@dataclass
class OrderReading:
    order: ConditionalOrder | None
    questions: list[Question] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ numbers

_NUM = r"(?:rs\.?\s*|inr\s*)?(?P<{name}>\d+(?:\.\d+)?)\s*(?P<{name}_unit>k|lakhs?|lacs?)?(?![\w.])"
_WORD_NUMBERS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_COUNT = r"(?P<{name}>\d{{1,2}}|" + "|".join(_WORD_NUMBERS) + ")"


def _num(name: str) -> str:
    return _NUM.format(name=name)


def _count(name: str) -> str:
    return _COUNT.format(name=name)


def _amount(m: re.Match, name: str) -> float:
    value = float(m.group(name))
    unit = m.group(f"{name}_unit")
    if unit == "k":
        value *= 1_000
    elif unit:
        value *= 100_000
    return value


def _to_count(raw: str) -> int:
    return _WORD_NUMBERS[raw] if raw in _WORD_NUMBERS else int(raw)


# ------------------------------------------------------------- clean-up

_SPACES = re.compile(r"\s+")


def _normalise(text: str) -> str:
    t = text.lower().replace("₹", " rs ").replace("’", "'").replace("−", "-")
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # 22,000 -> 22000
    t = t.replace("/-", "")
    t = re.sub(r"\b(?:rs|inr)\b\.?", "rs", t)  # "Rs." is not the end of a sentence
    t = re.sub(r"\brs\s*-\s*(?=\d)", "-rs ", t)  # "Rs -50000" -> "-rs 50000"
    return _SPACES.sub(" ", t).strip()


def _tidy(clause: str) -> str:
    """Drop joining punctuation and words left at the edges of a clause."""
    clause = _SPACES.sub(" ", clause).strip(" ,.;:!")
    clause = re.sub(r"^(?:and|then|,)\s+", "", clause)
    clause = re.sub(r"\s+(?:and|then)$", "", clause)
    return clause.strip(" ,.;:!")


# ------------------------------------------------------------- refusals

_REPEATING = re.compile(
    r"\bwhenever\b|\bevery\s+(?:time|day|week|month|session)\b|\beach\s+time\b|\brepeatedly\b"
)
_OPTIONS = re.compile(
    r"\b(?:calls?|puts?|ce|pe|strikes?|straddles?|strangles?|options?|premiums?|atm|otm|itm)\b"
)
_VIX = re.compile(r"\bvix\b")


def _refusal(text: str, why: str, field_name: str = "unparsed") -> Question:
    return Question(text=text, why=why, suggestion=None, field=field_name)


def _not_understood(part: str) -> Question:
    return _refusal(
        f'I didn\'t understand this part: "{part}". Could you say it another way?',
        "Silently ignoring words you typed could mean the order does something you did not intend.",
    )


_SUPPORTED_TEXT = (
    "NIFTY and BANKNIFTY (the index, or its futures), and the MCX futures this app "
    "trades: Crude Oil, Natural Gas, Gold, Silver and their minis"
)

# --------------------------------------------------------- stop & target

_STOP = re.compile(
    r"(?:with\s+(?:a\s+)?)?\b(?:stop\s*-?\s*loss|stoploss|sl|stop)\s*(?:of|at|:|=)?\s*"
    + r"(?P<v>\d+(?:\.\d+)?)\s*(?P<pct>%|percent|pct)?"
)
_TARGET = re.compile(
    r"(?:with\s+(?:a\s+)?)?\b(?:target|take\s*profit|tp)\s*(?:of|at|:|=)?\s*"
    + r"(?P<v>\d+(?:\.\d+)?)\s*(?P<pct>%|percent|pct)?"
)
_NO_STOP = re.compile(
    r"\b(?:with\s*out|with\s+no|no)\s+(?:a\s+)?(?:stop\s*-?\s*loss|stoploss|sl)\b"
)


def _pull_percent(
    text: str, pattern: re.Pattern, what: str, questions: list[Question]
) -> tuple[str, float | None]:
    found = list(pattern.finditer(text))
    if not found:
        return text, None
    if len(found) > 1:
        questions.append(
            _refusal(
                f"I found more than one {what} in that sentence. Which one did you mean?",
                f"Two different {what}s cannot both apply.",
                field_name=f"trade.{what.replace(' ', '_')}",
            )
        )
        return pattern.sub(" , ", text), None
    m = found[0]
    if not m.group("pct"):
        questions.append(
            _refusal(
                f'Is the {what} "{m.group(0).strip()}" a percentage? If so, please add the '
                f"% sign. A {what} in points or rupees can't be set yet -- only in percent.",
                f"A {what} in points or rupees reads very differently from one in percent, "
                "and guessing wrong would put it in the wrong place.",
                field_name=f"trade.{what.replace(' ', '_')}",
            )
        )
        return pattern.sub(" , ", text), None
    return pattern.sub(" , ", text), float(m.group("v"))


# ------------------------------------------------------------- instruments

_INDEX_NAMES = (
    ("BANKNIFTY", r"bank\s*nifty|nifty\s*bank"),
    ("NIFTY", r"nifty(?:\s*-?\s*50)?"),
)
_FUT = r"(?:futures?|fut|fno|f&o)"
_SPOT = r"(?:spot|index|cash)"
_PRICE_WORDS = r"(?:\s+(?:price|level|value|ltp|spot\s+price|futures?\s+price))?"


def _commodity(phrase: str) -> str | None:
    for symbol, pattern in _COMMODITIES:
        if pattern.fullmatch(phrase):
            return symbol
    return None


def _watched(phrase: str) -> Watched | Question | None:
    """The thing a condition watches. None if this phrase does not name one at all."""
    phrase = re.sub(r"^(?:the|mcx)\s+", "", phrase.strip())
    phrase = re.sub(r"^(?:the|mcx)\s+", "", phrase)
    phrase = re.sub(_PRICE_WORDS + "$", "", phrase).strip()
    for symbol, names in _INDEX_NAMES:
        m = re.fullmatch(rf"(?:{names})(?:\s+(?P<q>{_SPOT}|{_FUT}))?", phrase)
        if m:
            q = m.group("q") or ""
            kind = "future" if re.fullmatch(_FUT, q) else "index"
            return Watched(symbol=symbol, kind=kind)
    m = re.fullmatch(rf"(?P<name>.+?)(?:\s+{_FUT})?", phrase)
    if m:
        symbol = _commodity(m.group("name"))
        if symbol:
            return Watched(symbol=symbol, kind="future")
    if _UNSUPPORTED_COMMODITY.search(phrase) or re.search(
        r"\b(?:sensex|fin\s*nifty|midcp\s*nifty|nifty\s*next|bankex)\b", phrase
    ):
        return _refusal(
            f'I can\'t watch "{phrase}" yet. One-off orders work on {_SUPPORTED_TEXT}.',
            "Each market needs its own prices, lot size and expiry rules, and guessing "
            "them would get the order wrong.",
            field_name="trigger.instrument",
        )
    return None


def _traded(
    phrase: str, watched: Watched | None, answers: dict, trigger_failed: bool = False
) -> str | Question | None:
    """The futures contract an order buys or sells."""
    phrase = re.sub(r"^(?:the|mcx)\s+", "", phrase.strip())
    phrase = re.sub(
        r"\s+(?:at\s+(?:the\s+)?market(?:\s+price)?|immediately|right away)$", "", phrase
    )
    phrase = re.sub(r"\s+contracts?$", "", phrase).strip()
    if re.fullmatch(_FUT, phrase):
        # "Buy 1 lot futures if NIFTY ..." -- the futures of what is being watched.
        if watched is not None:
            return watched.symbol
        if trigger_failed:
            return ""  # the condition's own question comes first
        return _refusal(
            "Futures of which market?",
            "The sentence says 'futures' without naming the market.",
            field_name="trade.symbol",
        )
    for symbol, names in _INDEX_NAMES:
        m = re.fullmatch(rf"(?:{names})(?:\s+(?P<q>{_FUT}|{_SPOT}))?", phrase)
        if m:
            q = m.group("q") or ""
            if re.fullmatch(_FUT, q):
                return symbol
            if re.fullmatch(_SPOT, q):
                return _refusal(
                    f"The {symbol} index itself cannot be bought -- only its futures or "
                    f"options. Did you mean {symbol} futures?",
                    "An index is a number, not something you can own.",
                    field_name="trade.symbol",
                )
            answer = answers.get("trade.symbol", "")
            if re.search(r"\bfut", answer.lower()):
                return symbol
            return Question(
                text=f'Did you mean {symbol} futures? A "lot" of {symbol} could also be an '
                "option, and one-off orders only trade futures for now.",
                why="A futures lot and an option lot can differ in cost by a hundred times.",
                suggestion=f"{symbol} futures",
                field="trade.symbol",
            )
    m = re.fullmatch(rf"(?P<name>.+?)(?:\s+{_FUT})?", phrase)
    if m:
        symbol = _commodity(m.group("name"))
        if symbol:
            return symbol
    if _UNSUPPORTED_COMMODITY.search(phrase):
        return _refusal(
            f'I can\'t trade "{phrase}" yet. One-off orders trade the futures of {_SUPPORTED_TEXT}.',
            "Each contract has its own lot size and expiry, and guessing them would get "
            "the order wrong.",
            field_name="trade.symbol",
        )
    return None


# ------------------------------------------------------------------ actions

_SIDE = r"(?P<side>buy|sell|short|go\s+long(?:\s+on)?|go\s+short(?:\s+on)?)"
_LOTS = r"(?P<lots>\d{1,3}|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+lots?"
_EXIT = re.compile(
    r"(?:exit|close|square\s*-?\s*off|squareoff|sell\s+off|get\s+out\s+of)"
    r"(?:\s+(?:all|all\s+of|everything))?"
    r"(?:\s+(?:my|the))?(?:\s+open)?(?:\s+(?:positions?|trades?|everything))?"
)
# "tell me", "alert me", "notify me", "let me know" -- an AlertAction, never a
# trade or an exit. Matched as a whole clause: anything else in the clause is
# not understood, so "tell me and buy 1 lot" is refused rather than guessed at.
_ALERT = re.compile(
    r"(?:please\s+)?(?:tell\s+me|alert\s+me|notify\s+me|let\s+me\s+know)"
    r"(?:\s+about\s+(?:it|this))?"
)


def _side(raw: str) -> str:
    return "buy" if raw.startswith(("buy", "go long")) else "sell"


def _action(
    clause: str,
    watched: Watched | None,
    answers: dict,
    questions: list[Question],
    trigger_failed: bool = False,
) -> TradeAction | ExitAction | AlertAction | None:
    clause = _tidy(clause)
    if _ALERT.fullmatch(clause):
        return AlertAction()
    if _EXIT.fullmatch(clause):
        return ExitAction()

    m = re.fullmatch(rf"{_SIDE}\s+{_LOTS}\s+(?:of\s+)?(?P<inst>.+)", clause) or re.fullmatch(
        rf"{_SIDE}\s+(?P<inst>.+?)\s*,?\s+{_LOTS}", clause
    )
    lots: int | None = None
    if m:
        lots = _to_count(m.group("lots"))
    else:
        m = re.fullmatch(rf"{_SIDE}\s+(?P<inst>.+)", clause)
        if not m:
            questions.append(_not_understood(clause))
            return None
        answer = answers.get("trade.lots", "").strip().lower()
        a = re.fullmatch(r"(\d{1,3})(?:\s+lots?)?", answer)
        if a:
            lots = int(a.group(1))
        else:
            questions.append(
                Question(
                    text="How many lots?",
                    why="An order needs a size, and we never pick one for you.",
                    suggestion="1",
                    field="trade.lots",
                )
            )
    symbol = _traded(m.group("inst"), watched, answers, trigger_failed)
    if isinstance(symbol, Question):
        questions.append(symbol)
        return None
    if symbol == "":
        return None
    if symbol is None:
        questions.append(
            _refusal(
                f'I can\'t trade "{m.group("inst")}". One-off orders trade the futures of '
                f"{_SUPPORTED_TEXT}.",
                "Each contract has its own lot size and expiry, and guessing them would "
                "get the order wrong.",
                field_name="trade.symbol",
            )
        )
        return None
    if lots is None:
        return None
    try:
        return TradeAction(side=_side(m.group("side")), symbol=symbol, lots=lots)
    except ValidationError:
        questions.append(
            _refusal(
                f"{lots} lots is more than a one-off order here allows (up to 100).",
                "A typo in the size is the costliest kind.",
                field_name="trade.lots",
            )
        )
        return None


# ----------------------------------------------------------------- triggers

_VERB = (
    r"(?:(?:has|have|had)\s+)?(?:(?:gone|goes|go|going|falls?|fallen|falling|fell|drops?|dropped|"
    r"dropping|trades?|traded|trading|moves?|moved|moving|comes?|came|coming|is|was|slips?|"
    r"slipped|rises?|rose|risen|rising|climbs?|climbed|jumps?|jumped|gets|got|closes|closed)\s+)?"
    r"(?:down\s+|up\s+)?"
)
_BELOW = r"below|under|beneath|less\s+than|lower\s+than"
_ABOVE = r"above|over|higher\s+than|more\s+than"
_PRICE = re.compile(rf"(?P<w>.+?)\s+{_VERB}(?P<dir>{_BELOW}|{_ABOVE})\s+{_num('lvl')}")
_PRICE_NO_DIRECTION = re.compile(
    rf"(?P<w>.+?)\s+(?:(?:has\s+)?(?:crosses|crossed|cross|breaks|broke|hits|hit|touches|touched|"
    rf"reaches|reached|goes\s+past|crosses\s+past|is\s+at|at|beyond|past)\s+)+{_num('lvl')}"
)


def _price(clause: str, answers: dict, questions: list[Question]) -> PriceTrigger | None:
    m = _PRICE.fullmatch(clause)
    if m:
        direction = "below" if re.fullmatch(_BELOW, m.group("dir")) else "above"
    else:
        m = _PRICE_NO_DIRECTION.fullmatch(clause)
        if not m:
            return None
        answer = answers.get("price.direction", "").lower()
        if re.search(rf"\b(?:{_BELOW}|down|fall|drop)", answer):
            direction = "below"
        elif re.search(rf"\b(?:{_ABOVE}|up|rise|climb)", answer):
            direction = "above"
        else:
            questions.append(
                Question(
                    text=f"Should this go through when the price goes below "
                    f"{m.group('lvl')}, or above it?",
                    why='"Crosses" or "hits" could mean either way, and the two are opposite '
                    "orders.",
                    suggestion="below",
                    field="price.direction",
                )
            )
            return None
    watched = _watched(m.group("w"))
    if isinstance(watched, Question):
        questions.append(watched)
        return None
    if watched is None:
        questions.append(_not_understood(m.group("w")))
        return None
    return PriceTrigger(watch=watched, direction=direction, level=_amount(m, "lvl"))


# A run of candles: the count, what each one did, and how long each candle is.
_CONSEC = r"(?:\s+(?:consecutive|straight|successive|continuous))?"
_IN_A_ROW = r"(?:\s+(?:in\s+a\s+row|straight|running|consecutively))?"
_COLOUR = r"(?P<colour>red|green|bearish|bullish|negative|positive|down|up)"
_UNIT = (
    r"(?P<unit>weekly\s+candles?|daily\s+candles?|weekly\s+bars?|daily\s+bars?|weeks?|days?|"
    r"sessions?|trading\s+days?|candles?|bars?)"
)
_RUN = rf"{_count('n')}{_CONSEC}\s+(?:{_COLOUR}\s+)?{_UNIT}{_IN_A_ROW}"
_MOVE = (
    r"(?P<move>going\s+down|gone\s+down|falling|fallen|declining|down|red|lower|"
    r"going\s+up|gone\s+up|rising|risen|up|green|higher|closing\s+lower|closed\s+lower|"
    r"closing\s+higher|closed\s+higher)"
)
_HAS = (
    r"(?:\s+(?:has\s+been|had\s+been|has\s+had|has|had|have|was|is|been|formed|forms|made|"
    r"makes|printed|prints|shows|showed|closed|closes|given|gave|seen|saw))*"
)
_STREAK_FORMS = (
    # "nifty has been going down for 4 consecutive weeks"
    re.compile(rf"(?P<w>.+?){_HAS}\s+{_MOVE}\s+(?:for\s+)?(?:the\s+(?:last|past)\s+)?{_RUN}"),
    # "nifty has had 4 consecutive red weekly candles"
    re.compile(rf"(?P<w>.+?){_HAS}\s+(?:the\s+(?:last|past)\s+)?{_RUN}"),
    # "4 red weekly candles in a row on nifty"
    re.compile(rf"(?:there\s+(?:are|have\s+been)\s+)?{_RUN}\s+(?:on|in|for)\s+(?P<w>.+)"),
)
_FRAGMENT = re.compile(
    rf"(?:i\.?e\.?\s+|that\s+is\s+|meaning\s+)?(?:the\s+(?:last|past)\s+)?{_RUN}"
)

_DOWN_WORDS = {"red", "bearish", "negative", "down"}
_UP_WORDS = {"green", "bullish", "positive", "up"}


@dataclass
class _Run:
    count: int
    colour: str | None  # "red" / "green", if a candle colour was named
    move: str | None  # "down" / "up" / "lower" / "higher", if a direction was named
    candle: str | None  # "day" / "week", if a length was named


def _run_from(m: re.Match) -> _Run:
    colour = None
    if m.groupdict().get("colour"):
        c = m.group("colour")
        # "4 down weeks" names a direction, not a candle colour.
        if c in ("red", "bearish", "negative"):
            colour = "red"
        elif c in ("green", "bullish", "positive"):
            colour = "green"
    move = None
    raw_move = (m.groupdict().get("move") or "").replace("  ", " ")
    if raw_move in ("red",):
        colour = colour or "red"
    elif raw_move in ("green",):
        colour = colour or "green"
    elif "lower" in raw_move:
        move = "lower"
    elif "higher" in raw_move:
        move = "higher"
    elif raw_move:
        move = "down" if any(w in raw_move for w in ("down", "fall", "declin", "red")) else "up"
    if m.groupdict().get("colour") in ("down", "up"):
        move = move or m.group("colour")
    unit = m.group("unit")
    candle = None
    if unit.startswith("week"):
        candle = "week"
    elif unit.startswith(("da", "session", "trading")):
        candle = "day"
    return _Run(_to_count(m.group("n")), colour, move, candle)


def _streak(
    clause: str, extras: list[str], answers: dict, questions: list[Question]
) -> StreakTrigger | None:
    m = None
    for form in _STREAK_FORMS:
        m = form.fullmatch(clause)
        if m:
            break
    if not m:
        return None
    run = _run_from(m)
    watched = _watched(m.group("w"))
    if isinstance(watched, Question):
        questions.append(watched)
        return None
    if watched is None:
        questions.append(_not_understood(m.group("w")))
        return None

    # Bracketed restatements must agree with the sentence, and may fill a gap.
    for extra in list(extras):
        f = _FRAGMENT.fullmatch(extra)
        if not f:
            continue
        extras.remove(extra)
        more = _run_from(f)
        clash = (
            more.count != run.count
            or (more.colour and run.colour and more.colour != run.colour)
            or (more.candle and run.candle and more.candle != run.candle)
            or (more.colour == "red" and run.move in ("up", "higher"))
            or (more.colour == "green" and run.move in ("down", "lower"))
        )
        if clash:
            questions.append(
                _refusal(
                    f'"{extra}" doesn\'t match the rest of the sentence. Which did you mean?',
                    "Two descriptions of the same condition that disagree cannot both be followed.",
                    field_name="trigger.streak",
                )
            )
            return None
        run = _Run(
            run.count, run.colour or more.colour, run.move or more.move, run.candle or more.candle
        )

    candle = run.candle
    if candle is None:
        answer = answers.get("streak.candle", "").lower()
        if "week" in answer:
            candle = "week"
        elif "day" in answer or "daily" in answer:
            candle = "day"
        else:
            questions.append(
                Question(
                    text="Daily candles, or weekly candles?",
                    why="Four red daily candles and four red weekly candles are very "
                    "different things.",
                    suggestion="weekly",
                    field="streak.candle",
                )
            )
            return None

    if run.colour:
        rule = run.colour
    elif run.move == "lower":
        rule = "lower_close"
    elif run.move == "higher":
        rule = "higher_close"
    elif run.move in ("down", "up"):
        answer = answers.get("streak.rule", "").lower()
        down = run.move == "down"
        if re.search(r"\b(?:red|green|candle|open)", answer):
            rule = "red" if down else "green"
        elif re.search(r"\b(?:lower|higher|previous|before|last)", answer):
            rule = "lower_close" if down else "higher_close"
        else:
            unit = "week" if candle == "week" else "day"
            questions.append(
                Question(
                    text=(
                        f'What does "going {run.move}" mean here: each {unit} closed '
                        f"{'below' if down else 'above'} its own open (a "
                        f"{'red' if down else 'green'} candle), or each {unit} closed "
                        f"{'lower' if down else 'higher'} than the {unit} before?"
                    ),
                    why=(
                        f"The two are different. A {unit} can open with a gap "
                        f"{'down' if down else 'up'} and still close "
                        f"{'above' if down else 'below'} its open."
                    ),
                    suggestion=f"{'red' if down else 'green'} candles",
                    field="streak.rule",
                )
            )
            return None
    else:
        questions.append(_not_understood(clause))
        return None
    return StreakTrigger(watch=watched, candle=candle, count=run.count, rule=rule)


_PNL_WORD = (
    r"(?:(?:my|the|overall|total|combined|net|portfolio)\s+)*"
    r"(?:p\s*&\s*l|p\s*n\s*l|pnl|mtm|m2m|profit\s*(?:and|&|/)\s*loss)"
    r"(?:\s+(?:of|on)\s+(?:my\s+|the\s+)?(?:open\s+)?(?:positions?|trades?))?"
)
_PNL_BELOW = re.compile(
    rf"{_PNL_WORD}\s+{_VERB}(?:{_BELOW})\s+(?P<neg>-\s*|minus\s+|negative\s+)?{_num('amt')}"
)
_LOSS = re.compile(
    rf"(?:{_PNL_WORD}\s+(?:is\s+)?)?(?:(?:my|the|a|total|overall|combined)\s+)*loss(?:es)?\s+"
    rf"(?:(?:of|is|goes|crosses|crossed|exceeds|exceeded|hits|reaches|reached|more\s+than|"
    rf"over|above|beyond|bigger\s+than|greater\s+than|at\s+least)\s+)*{_num('amt')}"
)
_LOSE = re.compile(
    rf"(?:i\s+)?(?:lose|lost|am\s+down|am\s+losing|'m\s+down)\s+(?:more\s+than\s+|over\s+)?"
    rf"{_num('amt')}"
)


def _pnl(clause: str, answers: dict, questions: list[Question]) -> PnlTrigger | None:
    for pattern in (_LOSS, _LOSE):
        m = pattern.fullmatch(clause)
        if m:
            return PnlTrigger(threshold=-_amount(m, "amt"))
    m = _PNL_BELOW.fullmatch(clause)
    if not m:
        return None
    amount = _amount(m, "amt")
    if m.group("neg"):
        return PnlTrigger(threshold=-amount)
    answer = answers.get("pnl.sign", "").lower()
    if re.search(r"\b(?:loss|losing|lose|minus|negative|-)", answer):
        return PnlTrigger(threshold=-amount)
    if re.search(r"\b(?:profit|gain|plus|positive)", answer):
        return PnlTrigger(threshold=amount)
    questions.append(
        Question(
            text=(
                f"Do you mean a loss of Rs {amount:,.0f} (P&L at minus {amount:,.0f}), "
                f"or your profit falling under Rs {amount:,.0f}?"
            ),
            why=(
                "The two are opposite. Read the wrong way, the positions would be closed "
                "while still in profit, or kept open well past the loss you meant."
            ),
            suggestion="a loss",
            field="pnl.sign",
        )
    )
    return None


def _trigger(clause: str, extras: list[str], answers: dict, questions: list[Question]):
    clause = _tidy(clause)
    before = len(questions)
    for attempt in (
        lambda: _pnl(clause, answers, questions),
        lambda: _streak(clause, extras, answers, questions),
        lambda: _price(clause, answers, questions),
    ):
        found = attempt()
        if found is not None or len(questions) > before:
            return found
    questions.append(
        _refusal(
            f'I didn\'t understand the condition "{clause}". One-off orders can wait for: a '
            "price going above or below a level, a run of red or green daily or weekly "
            "candles, or your paper positions' P&L falling to a limit.",
            "Guessing at a condition would mean the order goes through at a moment you "
            "did not choose.",
            field_name="trigger",
        )
    )
    return None


# ---------------------------------------------------------------- the reader

_LEAD = r"(?:if|when|once|as\s+soon\s+as|in\s+case)"
_IF_FIRST = re.compile(rf"{_LEAD}\s+(?P<trig>.+?)\s*(?:,|\bthen\b)\s*(?P<act>.+)")
# "if crude goes above 6500 buy 1 lot" -- no comma; the order starts at its verb.
_IF_FIRST_NO_COMMA = re.compile(
    rf"{_LEAD}\s+(?P<trig>.+?)\s+(?P<act>(?:buy|sell|short|go\s+long|go\s+short|exit|close|"
    rf"square\s*-?\s*off|squareoff|get\s+out|tell\s+me|alert\s+me|notify\s+me|let\s+me\s+know)\b.*)"
)
_IF_LATER = re.compile(rf"(?P<act>.+?)\s*,?\s+{_LEAD}\s+(?P<trig>.+)")
_BRACKETS = re.compile(r"\(([^()]*)\)")


def parse_order(text: str, answers: dict[str, str] | None = None) -> OrderReading:
    """Read one sentence into a one-off order, or say exactly what is missing."""
    answers = {k: str(v) for k, v in (answers or {}).items()}
    questions: list[Question] = []
    t = _normalise(text)
    if not t:
        return OrderReading(None, [_refusal("Type an order to read.", "There is nothing here.")])

    if _REPEATING.search(t):
        return OrderReading(
            None,
            [
                _refusal(
                    "This page sets up orders that go through once and are then finished. "
                    "For a rule that trades again and again, build a strategy on the Build "
                    "page, where it can be tested on past prices first.",
                    '"Whenever" and "every time" describe a strategy, not a one-off order.',
                    field_name="kind",
                )
            ],
        )
    if _OPTIONS.search(t):
        return OrderReading(
            None,
            [
                _refusal(
                    "One-off orders can't trade options yet -- only futures.",
                    "Options need a strike and an expiry chosen, and their prices modelled; "
                    "that part of the app is not built.",
                    field_name="trade.symbol",
                )
            ],
        )
    if _VIX.search(t):
        return OrderReading(
            None,
            [
                _refusal(
                    "I can't watch India VIX yet. One-off orders can watch "
                    + _SUPPORTED_TEXT
                    + ".",
                    "The app has no VIX prices to check the condition against.",
                    field_name="trigger.instrument",
                )
            ],
        )

    no_stop_said = bool(_NO_STOP.search(t))
    t = _NO_STOP.sub(" , ", t)
    t, stop = _pull_percent(t, _STOP, "stop loss", questions)
    t, target = _pull_percent(t, _TARGET, "target", questions)
    extras = [_tidy(x) for x in _BRACKETS.findall(t)]
    t = _BRACKETS.sub(" , ", t)
    t = re.sub(r"(?:\s*[,.;]\s*)+", ", ", t).strip(" ,")

    m = _IF_FIRST.fullmatch(t) or _IF_FIRST_NO_COMMA.fullmatch(t) or _IF_LATER.fullmatch(t)
    if not m:
        questions.append(
            _refusal(
                "I couldn't find both a condition and an order. Try the shape "
                '"If <something happens>, <buy/sell/exit>", for example '
                '"If NIFTY goes below 22000, buy 1 lot NIFTY futures".',
                "A one-off order needs to know what to wait for and what to do.",
                field_name="kind",
            )
        )
        return OrderReading(None, questions)

    trigger = _trigger(m.group("trig"), extras, answers, questions)
    watched = getattr(trigger, "watch", None)
    action = _action(m.group("act"), watched, answers, questions, trigger is None)
    for extra in extras:
        if extra:
            questions.append(_not_understood(extra))

    if stop is None and "trade.stop_loss" in answers and answers["trade.stop_loss"].strip():
        a = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*%?\s*", answers["trade.stop_loss"])
        if a:
            stop = float(a.group(1))
        else:
            questions.append(
                _refusal(
                    "Please give the stop loss as a percentage, for example 1%.",
                    "A stop loss in points or rupees reads very differently.",
                    field_name="trade.stop_loss",
                )
            )

    if isinstance(action, (ExitAction, AlertAction)) and (stop is not None or target is not None):
        questions.append(
            _refusal(
                "A stop loss or target doesn't apply here: this order doesn't open a "
                "position. Remove it, or put it on an order that buys or sells.",
                "There is nothing it could protect -- nothing is bought or sold.",
                field_name="trade.stop_loss",
            )
        )
    if questions or trigger is None or action is None:
        return OrderReading(None, questions)

    if isinstance(action, TradeAction):
        try:
            action = action.model_copy(update={"stop_pct": stop, "target_pct": target})
            action = TradeAction.model_validate(action.model_dump())
        except ValidationError:
            return OrderReading(
                None,
                [
                    _refusal(
                        "That stop loss or target is outside what this app allows "
                        "(stop up to 50%, target up to 500%).",
                        "A number that size is more likely a typo than a plan.",
                        field_name="trade.stop_loss",
                    )
                ],
            )
    try:
        order = ConditionalOrder(trigger=trigger, action=action)
    except ValidationError:
        return OrderReading(
            None,
            [
                _refusal(
                    "A P&L limit can only close positions, not open new ones.",
                    "Opening a position because others are losing is not something this "
                    "app will do on a guess.",
                    field_name="kind",
                )
            ],
        )

    warnings: list[str] = []
    if isinstance(action, TradeAction) and action.stop_pct is None:
        warnings.append(
            "No stop loss"
            + (" (as you asked)" if no_stop_said else "")
            + ". This position stays open, however far the price moves against it, until "
            "you close it or the contract expires."
        )
    return OrderReading(order, [], warnings)
