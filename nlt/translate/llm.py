"""The AI reader: a second pass for sentences the phrase table did not understand.

`rules.parse` is still the first and preferred reader -- free, deterministic,
and never wrong on the 70 real user sentences. It is also narrow: it builds a
recipe for 3 of them. This module asks a language model (via OpenRouter) to try
the sentences the rules could not *read*, and then refuses to trust a word of
what comes back until code has checked it.

What the model is allowed to decide is deliberately small. It proposes the
entry and exit rules and the indicators they use, and nothing else:

  * The instrument must agree with what `rules._extract_instrument` finds in
    the same sentence. If the rules found none, the model's choice must be a
    known index, basket or NIFTY 500 ticker that the sentence actually names.
  * The timeframe and backtest window are taken from the rules' own extractors;
    the model does not get a say.
  * Risk limits, default sizing and the schedule come from the same defaults
    the rules use, so a recipe reads and runs identically whichever reader
    built it.

And the model has to show its working. Every part of the recipe cites the
user's exact words (`evidence`), and code checks, mechanically:

  * every quote really appears in the sentence;
  * every number in the recipe appears in the sentence -- the exit numbers in
    the very quote that justifies them, next to a word like "stop" or "target";
  * every word the user typed is covered by some quote, or is filler. A clause
    the model quietly dropped ("...and P/E under 18") comes back as a question.

A reply that fails the first check -- a quote the user never wrote -- is thrown
away entirely: a model that fabricates evidence is not trusted on anything
else in that reply either. Any failure to reach the model at all (no key, no
network, garbage back) also falls back to the rules' own answer. This module
never raises into the page and never guesses.

The refusal messages are written here, not by the model. The model only
chooses which one applies.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import Field, ValidationError

from nlt.indicators.registry import REGISTRY, vocabulary
from nlt.spec.models import (
    INDICES,
    Base,
    Condition,
    ExitRules,
    IndicatorSpec,
    Instrument,
    Schedule,
    Sizing,
    StrategySpec,
)
from nlt.translate import options_words, readback, rules
from nlt.translate import spelling as rules_spelling
from nlt.translate.rules import Question, TranslationResult

logger = logging.getLogger(__name__)

# One setting to change the model. Checked against OpenRouter's price list when
# chosen: roughly Rs 1-2 per sentence read.
DEFAULT_MODEL = "anthropic/claude-sonnet-5.5"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Outside the repository on purpose, so the key can never be committed.
KEY_PATH = Path.home() / ".config" / "nlt" / "openrouter_key"

# Set by the test suite so no test can ever spend money or depend on the network.
DISABLE_ENV = "NLT_LLM_DISABLED"

Transport = Callable[[str, str], str]
"""(system prompt, user message) -> the model's raw reply text."""

AI_NOTE = (
    "This strategy was read by an AI model, not by the fixed phrase list. "
    "Check every line below against what you meant before continuing."
)

# ------------------------------------------------------------------ gating

# The rules' questions that mean "I could not read these words" -- the only
# situation the model is allowed to help with.
_NOT_UNDERSTOOD = {"unparsed", "entry.condition"}
# Questions that are fine to repeat alongside those, because they follow from
# not having understood the sentence rather than from a policy decision.
# "Once a day or every time?" is asked again on the AI path (`rules.range_repeat`).
_FOLLOW_ON = {"exit.stop_pct", "exit.rule", "schedule.daily_limit"}


def should_ask_llm(result: TranslationResult) -> bool:
    """True only when the rules failed to *understand*, never when they refused on purpose.

    A deliberate refusal -- options, an unsupported candle size, VWAP on daily
    bars, "NIFTY 50" the index or the basket, an ambiguous "it" -- is a
    decision, and a second reader must not be allowed to overrule it. Nor is
    the model asked when the rules understood everything and only need the
    user to supply a stop loss.
    """
    if result.spec is not None:
        return False
    fields = {q.field for q in result.questions}
    return bool(fields & _NOT_UNDERSTOOD) and fields <= (_NOT_UNDERSTOOD | _FOLLOW_ON)


def translate(
    description: str,
    *,
    answers: dict[str, str] | None = None,
    transport: Transport | None = None,
) -> TranslationResult:
    """Rules first; the model only when the rules could not read the sentence."""
    answers = answers or {}
    result = rules.parse(description, answers=answers)
    if not should_ask_llm(result):
        return result
    ai = parse_llm(description, answers=answers, transport=transport)
    return result if ai is None else ai


# ------------------------------------------------------------------ the draft

EvidenceField = Literal[
    "instrument",
    "direction",
    "entry",
    "exit.stop",
    "exit.trailing_stop",
    "exit.atr_stop",
    "exit.target",
    "exit.condition",
    "exit.max_bars_held",
    "sizing",
    "square_off",
]

RefuseCategory = Literal[
    "question",
    "robo",
    "fundamental",
    "event",
    "options",
    "unsupported_instrument",
    "vague",
    "unsupported_rule",
]


class Evidence(Base):
    field: EvidenceField
    quote: str = Field(min_length=1, max_length=120)


class ExitDraft(Base):
    """`ExitRules` without its "must have an exit" check, so a missing stop can
    become a question instead of a validation failure."""

    target_pct: float | None = Field(default=None, gt=0, le=100)
    stop_pct: float | None = Field(default=None, gt=0, le=100)
    trailing_stop_pct: float | None = Field(default=None, gt=0, le=100)
    stop_atr_mult: float | None = Field(default=None, gt=0, le=20)
    atr_id: str | None = None
    condition: Condition | None = None
    max_bars_held: int | None = Field(default=None, ge=1, le=5000)


class Draft(Base):
    kind: Literal["strategy", "refuse"]
    refuse_category: RefuseCategory | None = None
    unsupported_quote: str | None = None
    symbol: str | None = None
    direction: Literal["long", "short"] = "long"
    indicators: list[IndicatorSpec] = Field(default_factory=list, max_length=20)
    entry: Condition | None = None
    exit: ExitDraft = Field(default_factory=ExitDraft)
    sizing: Sizing | None = None
    square_off: str | None = None
    evidence: list[Evidence] = Field(default_factory=list, max_length=40)


# ------------------------------------------------------------------ refusals

_SUPPORTED = (
    "NIFTY, BANKNIFTY, single NSE stocks, the NIFTY 50 / 100 / 500 stock baskets, "
    "NIFTY and BANKNIFTY futures, and MCX futures on crude oil, natural gas, gold and "
    "silver (and their mini contracts)"
)

_REFUSALS: dict[str, tuple[str, str]] = {
    "question": (
        "That reads as a question or a request for analysis, not a trading rule. "
        "I can build and test a strategy with a clear entry and exit – for example "
        "'Buy TCS when RSI crosses below 30, target 4%, stop loss 2%'.",
        "Answering market questions or giving opinions is not something this tool does; "
        "turning a question into trades would be a guess.",
    ),
    "robo": (
        "That sounds like a request to invest or manage money for you. This tool does not "
        "pick investments – it tests exact buy and sell rules that you describe.",
        "Choosing what to buy for you would be advice, and there is no rule here to test.",
    ),
    "fundamental": (
        "That depends on company fundamentals (such as P/E, ROE, earnings, debt or promoter "
        "holding). This platform only has price and volume history, so it cannot test that yet.",
        "Leaving the fundamental part out would test a different strategy from the one "
        "you described.",
    ),
    "event": (
        "That depends on news or corporate events (dividends, board meetings, deals, "
        "announcements). There is no event history here yet, so it cannot be tested.",
        "Leaving the event part out would test a different strategy from the one you described.",
    ),
    "options": (
        "I can't read that option position. Options I can test: buying one call or put "
        "('On NIFTY 15-minute bars, buy the weekly ATM call when RSI crosses above 60. Stop "
        "30%, target 60% on the premium.'), or selling, written like 'Sell the NIFTY weekly "
        "ATM straddle at 09:20 with a 25% stop on each leg. Square off at 15:15.'",
        "Testing a different combination of options from the one you meant would give a "
        "different, misleadingly confident-looking result.",
    ),
    "unsupported_instrument": (
        "That trades something this platform can't test yet (for example stock futures, "
        f"other commodities, VIX or other markets). What it can test: {_SUPPORTED}.",
        "Substituting a different instrument would test a trade you never described.",
    ),
    "vague": (
        "That's too loose to turn into exact rules. Say exactly when to buy, when to sell, "
        "and what stop loss to use.",
        "Filling in the details for you would be a guess, and this tool never guesses "
        "with your money.",
    ),
    "unsupported_rule": (
        "Part of that uses something this platform can't calculate yet.",
        "Leaving that part out would test a different strategy from the one you described.",
    ),
}


def _refusal(category: str, quote: str | None = None) -> TranslationResult:
    text, why = _REFUSALS.get(category, _REFUSALS["vague"])
    if quote:
        text = f'{text} The part I can\'t handle: "{quote}".'
    return TranslationResult(
        spec=None,
        questions=[Question(text=text, why=why, suggestion=None, field="description")],
        source="llm",
    )


def _question(text: str, why: str, field: str, *, notes=None, unparsed=None) -> TranslationResult:
    return TranslationResult(
        spec=None,
        questions=[Question(text=text, why=why, suggestion=None, field=field)],
        notes=list(notes or []),
        unparsed=list(unparsed or []),
        source="llm",
    )


def _stop_question(notes: list[str]) -> TranslationResult:
    return TranslationResult(
        spec=None,
        questions=[
            Question(
                text="What stop loss should this strategy use?",
                why=(
                    "A strategy with no stop loss can lose far more than intended before "
                    "anything closes the position – we never assume one for you."
                ),
                suggestion="A common starting point is 1%.",
                field="exit.stop_pct",
                choices=rules.STOP_CHOICES,
                answerable=True,
            )
        ],
        notes=notes,
        source="llm",
    )


# ------------------------------------------------------------------ the prompt

_EXAMPLE_STRATEGY = {
    "kind": "strategy",
    "symbol": "NIFTY 100",
    "direction": "long",
    "indicators": [
        {"id": "ema8", "type": "ema", "params": {"length": 8}},
        {"id": "ema21", "type": "ema", "params": {"length": 21}},
        {"id": "adx14", "type": "adx", "params": {"length": 14}},
    ],
    "entry": {
        "kind": "all",
        "conditions": [
            {
                "kind": "compare",
                "op": "crosses_above",
                "left": {"kind": "ref", "name": "ema8"},
                "right": {"kind": "ref", "name": "ema21"},
            },
            {
                "kind": "compare",
                "op": "gt",
                "left": {"kind": "ref", "name": "adx14", "output": "adx"},
                "right": {"kind": "const", "value": 20},
            },
        ],
    },
    "exit": {"target_pct": 3, "stop_pct": 1.5},
    "evidence": [
        {"field": "direction", "quote": "buy"},
        {"field": "instrument", "quote": "NIFTY 100 stocks"},
        {"field": "entry", "quote": "when ema8 crosses above ema21"},
        {"field": "entry", "quote": "and adx14 > 20"},
        {"field": "exit.target", "quote": "Sell at 3% profit"},
        {"field": "exit.stop", "quote": "Stop loss 1.5%"},
    ],
}

_EXAMPLE_REFUSAL = {
    "kind": "refuse",
    "refuse_category": "fundamental",
    "unsupported_quote": "whose P/E ratio is less than 30",
}


def system_prompt() -> str:
    return f"""You translate one trading-strategy sentence from an Indian retail trader into a
strict JSON recipe for a backtesting engine, or refuse. Reply with ONE JSON object and nothing else.

Your output is checked by code. Anything you cannot express exactly must be refused, never
approximated. Leaving out part of the sentence is the worst mistake you can make.

## Refuse (kind = "refuse") when the sentence is:
- "question": a question, request for analysis/opinion/tips/screening ("find", "which", "is X good")
- "robo": asks to invest/manage money for them with no explicit rules
- "fundamental": uses P/E, ROE, EPS, revenue, debt, promoter holding or any fundamental data
- "event": depends on news, dividends, board meetings, deals, tariffs, announcements
- "options": SELLS (writes) an option, or uses more than one option leg (straddle, strangle,
  condor, butterfly, spread, "call and put", a credit, a hedge). Buying a single call or put is
  NOT refused here -- by the time you see a sentence like that, the instrument has already been
  read as an option by other code, so do set "symbol" as usual (NIFTY/BANKNIFTY/SENSEX, whichever
  the sentence names, defaulting to NIFTY if it names none) with its own evidence quote exactly as
  you would for a non-option sentence; just never try to say which strike, expiry or side (call vs
  put) it is -- leave that entirely to the other code, which does not read anything you say about it.
- "unsupported_instrument": stock futures, commodities other than those listed below, VIX,
  currencies, anything not listed below
- "vague": no precise entry rule ("buy dips", "sell rallies")
- "unsupported_rule": needs something not in the indicator list or not expressible in the schema
  (e.g. "while price holds X"). Set unsupported_quote to the
  exact words you could not express.
For refusals, fill only kind, refuse_category and (optionally) unsupported_quote.

## Instruments (field "symbol")
"NIFTY" (the index), "BANKNIFTY", "NIFTY 50" / "NIFTY 100" / "NIFTY 500" (baskets of stocks,
only when the sentence says stocks of that index), or a single NSE ticker in capitals such as
"RELIANCE", "TCS", "HDFCBANK". "SENSEX" is valid ONLY when the sentence is buying a SENSEX
option -- the SENSEX index itself and SENSEX futures are not supported. Futures: "NIFTY" or
"BANKNIFTY" when the sentence says futures,
and the MCX contracts "CRUDEOIL", "CRUDEOILM" (crude oil mini), "NATURALGAS", "NATGASMINI"
(natural gas mini), "GOLD", "GOLDM" (gold mini), "SILVER", "SILVERM" (silver mini). Quote the
words that name the contract, including "futures", as evidence for "instrument". Never pick an
instrument the sentence does not name. For an option (buying a single call or put), quote the
whole phrase that names it -- e.g. "weekly ATM call", "monthly put 1 strike OTM" -- as evidence
for "instrument" too; which option it actually is (strike, expiry, side) is decided by other
code, not by you, but every one of those words still needs a quote or it will look unread.

## Indicators (the ONLY ones available)
{vocabulary()}
Declare each in "indicators" as {{"id": "<lowercase id>", "type": "<name>", "params": {{...}}}}.
Only set params the sentence states; leave the rest at their defaults. Multi-output indicators
need "output" on every ref (e.g. {{"kind":"ref","name":"macd1","output":"signal"}}). Price
fields usable as refs without declaring: open, high, low, close, volume.

## Conditions (for "entry" and "exit.condition")
- {{"kind":"compare","op":"lt|lte|gt|gte|eq|crosses_above|crosses_below","left":OPERAND,"right":OPERAND}}
  OPERAND is {{"kind":"ref","name":"<id or price field>","output":null,"bars_ago":0}} or
  {{"kind":"const","value":<number>}}.
- {{"kind":"is_true","ref":REF}} for candlestick patterns.
- {{"kind":"percent_change","ref":REF,"lookback":<bars>,"op":"lt|lte|gt|gte","value":<percent>}}
  ("falls 2% in one session" -> ref close, lookback 1, op lte, value -2).
- {{"kind":"at_time","time":"HH:MM"}} for an entry at a time of day ("buy at 09:30"), 24-hour.
- {{"kind":"all","conditions":[...]}}, {{"kind":"any","conditions":[...]}}, {{"kind":"not","condition":...}}
"drops below"/"crosses below"/"cracks" = crosses_below. "is below"/"<" = lt. Keep that distinction.
"between A and B" = all of gte A and lte B.
A gap at the open uses session_open: "the day opens more than 0.5% below the previous close" =
compare session_open output gap_pct lt -0.5 ("0.5% below" with no "more than" = lte -0.5; above =
gt/gte +0.5). "Still below the open" / "below today's open" = compare close lt session_open
output open.

## Exits ("exit" object, all optional)
target_pct, stop_pct, trailing_stop_pct, stop_atr_mult (+ atr_id naming a declared atr),
condition, max_bars_held. Only include what the sentence states. NEVER invent a stop loss: if the
sentence has none, leave the stop fields out entirely. A stop at a price level ("stop at the low of
the opening range", "stop at the previous day's low", "stop at the lowest low of the last 10
candles") is read by other code: quote those words as evidence for "exit.stop" and leave every
stop field empty -- do not turn it into a condition. A later "sell when ..." is an exit
condition, not a short entry.

## Other fields
direction: "long" or "short" (from the leading verb). sizing: only if stated, one of
{{"mode":"fixed_lots","lots":N}}, {{"mode":"fixed_value","value":RUPEES}},
{{"mode":"risk_based","risk_pct":P}}. square_off: "HH:MM" 24-hour, only if stated.
Do NOT output timeframe or backtest dates -- those are handled separately.

## Evidence (required for kind = "strategy")
A list of {{"field": ..., "quote": ...}} where quote is copied EXACTLY from the sentence
(same words, same spelling). field is one of: instrument, direction, entry, exit.stop,
exit.trailing_stop, exit.atr_stop, exit.target, exit.condition, exit.max_bars_held, sizing,
square_off. Every exit value needs its own quote containing its number. Every number anywhere
in the recipe must appear in the sentence. Together the quotes must cover every meaningful word
of the sentence apart from candle size / backtest dates -- if some words don't fit any field,
refuse with "unsupported_rule" instead.

## Example
Sentence: "Instantly buy NIFTY 100 stocks when ema8 crosses above ema21 and adx14 > 20. Sell at 3% profit. Stop loss 1.5%"
{json.dumps(_EXAMPLE_STRATEGY)}

Sentence: "Instantly buy NIFTY 50 stocks whose P/E ratio is less than 30 and sell at 5% profit. Stop loss is 2%."
{json.dumps(_EXAMPLE_REFUSAL)}
"""


# ------------------------------------------------------------------ transport


def _api_key() -> str | None:
    if os.environ.get(DISABLE_ENV):
        return None
    env = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env:
        return env
    try:
        key = KEY_PATH.read_text().strip()
    except OSError:
        return None
    return key or None


def is_configured() -> bool:
    return _api_key() is not None


def model_name() -> str:
    return os.environ.get("NLT_LLM_MODEL", DEFAULT_MODEL)


_REPLY_CACHE: dict[tuple[str, str], str] = {}


def openrouter_transport(system: str, user: str) -> str:
    """Calls OpenRouter. Raises on any failure; `parse_llm` turns that into a fallback.

    Replies are cached per sentence for the life of the process, so answering
    a follow-up question (the stop loss, say) re-reads the same draft instead
    of paying for, and possibly getting, a different one.
    """
    import requests

    key = _api_key()
    if key is None:
        raise RuntimeError("no OpenRouter key configured")
    model = model_name()
    cache_key = (model, user)
    if cache_key in _REPLY_CACHE:
        return _REPLY_CACHE[cache_key]

    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0,
            "max_tokens": 2000,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    _REPLY_CACHE[cache_key] = content
    return content


def _extract_json(reply: str) -> dict | None:
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


# ------------------------------------------------------------------ checks

_SPELLED = {
    "zero": 0,
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
    "eleven": 11,
    "twelve": 12,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}

_STOP_WORDS = re.compile(r"stop|\bsl\b")
_TRAIL_WORDS = re.compile(r"trail")
_ATR_WORDS = re.compile(r"\batr")
_TARGET_WORDS = re.compile(r"target|profit|\btp\b|take|\bgain")
_DIRECTION_WORDS = {
    "long": re.compile(r"\bbuy|\blong\b|\benter\b"),
    "short": re.compile(r"\bshort|\bsell\b"),
}


def _numbers_in(text: str) -> set[float]:
    found: set[float] = set()
    for m in re.finditer(r"\d+(?:,\d+)*(?:\.\d+)?", text):
        found.add(float(m.group().replace(",", "")))
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _SPELLED:
            found.add(float(_SPELLED[word]))
    return found


def _has_number(value: float, numbers: set[float]) -> bool:
    return any(abs(abs(value) - n) < 1e-9 for n in numbers)


def _quote_pattern(quote: str) -> re.Pattern:
    words = quote.lower().split()
    return re.compile(r"\s+".join(re.escape(w) for w in words))


def _condition_numbers(cond) -> list[float]:
    """Every number a condition tree states, except the structural 0/1 defaults."""
    out: list[float] = []
    if cond is None:
        return out
    kind = cond.kind
    if kind == "compare":
        for side in (cond.left, cond.right):
            if side.kind == "const":
                out.append(side.value)
            elif side.bars_ago > 1:
                out.append(float(side.bars_ago))
    elif kind == "at_time":
        out.extend([float(cond.time.hour), float(cond.time.minute)])
    elif kind == "percent_change":
        out.append(cond.value)
        if cond.lookback > 1:
            out.append(float(cond.lookback))
    elif kind in ("all", "any"):
        for c in cond.conditions:
            out.extend(_condition_numbers(c))
    elif kind == "not":
        out.extend(_condition_numbers(cond.condition))
    return out


def _indicator_numbers(indicators: list[IndicatorSpec]) -> list[float]:
    """Parameters that differ from the registry default must come from the sentence."""
    out: list[float] = []
    for ind in indicators:
        defaults = REGISTRY[ind.type].params
        for key, value in ind.params.items():
            if isinstance(value, str) or value == defaults.get(key):
                continue
            out.append(float(value))
    return out


def _length_said_in_days(indicator: IndicatorSpec, lowered: str) -> bool:
    """A moving average whose length the user gave in days: "50-day", "50 day",
    "50 DMA". Read from the sentence, not the model's reply."""
    if indicator.type not in ("sma", "ema", "wma", "hma"):
        return False
    length = indicator.resolved_params().get("length")
    if not isinstance(length, (int, float)):
        return False
    n = int(length)
    return re.search(rf"(?<!\d){n}\s*-?\s*(?:days?\b|dma\b)", lowered) is not None


def _without_level_words(draft: Draft, lowered: str, span: tuple[int, int]) -> Draft:
    """The draft with whatever the model made of the level-stop words removed.

    Tradl #24's recorded reply turned "stop at the low of the opening range"
    into an exit condition ("low < the range low", filled at the next open):
    not the stop the user asked for. Any exit condition or stop quoted only
    from those words is dropped; the rules' level stop replaces it.
    """
    a, b = span

    def inside(quote: str) -> bool:
        m = _quote_pattern(quote).search(lowered)
        return m is not None and m.start() >= a and m.end() <= b

    ex = draft.exit
    update: dict = {}
    for field, key in (("exit.condition", "condition"), ("exit.stop", "stop_pct")):
        quotes = [ev.quote for ev in draft.evidence if ev.field == field]
        if quotes and all(inside(q) for q in quotes) and getattr(ex, key) is not None:
            update[key] = None
    if not update:
        return draft
    evidence = [
        ev
        for ev in draft.evidence
        if not (ev.field in ("exit.condition", "exit.stop") and inside(ev.quote))
    ]
    return draft.model_copy(update={"exit": ex.model_copy(update=update), "evidence": evidence})


def _drop_redundant_outputs(raw: dict) -> None:
    """`rsi14.rsi` -> `rsi14`, the shape the rules write, so the readback reads the same.

    Naming the only output of a single-output indicator means nothing different
    to the engine, but the readback prints it -- "RSI(14) rsi crosses below 30".
    """
    types = {i.get("id"): i.get("type") for i in raw.get("indicators") or [] if isinstance(i, dict)}

    def walk(node):
        if isinstance(node, dict):
            if node.get("kind") == "ref" and node.get("output") is not None:
                d = REGISTRY.get(types.get(node.get("name")))
                if d is not None and d.outputs == (node["output"],):
                    node["output"] = None
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(raw.get("entry"))
    walk(raw.get("exit"))


def _resolve_symbol(
    draft_symbol: str | None, rules_kwargs: dict, lowered: str
) -> tuple[str | None, str | None]:
    """Returns (symbol, problem). The rules' reading wins; the model may only fill a gap."""
    rules_symbol = rules_kwargs.get("symbol")
    cleaned = " ".join((draft_symbol or "").upper().split()) or None

    if rules_symbol is not None:
        if cleaned != rules_symbol:
            return None, (
                f"I read the instrument two different ways ({rules_symbol} and "
                f"{cleaned or 'nothing'}). Which one do you mean?"
            )
        return rules_symbol, None

    if cleaned is None:
        return None, "Which instrument should this strategy trade?"

    # The rules recognise every spelling of NIFTY, BANKNIFTY and the baskets, so
    # if they found none the model can only be offering a stock -- and it must
    # be a real one that the sentence spells out.
    squashed = re.sub(r"[^a-z0-9&]", "", lowered)
    if cleaned not in rules._known_tickers() or cleaned.lower() not in squashed:
        return None, (
            f"I'm not sure which stock you mean (I read it as {cleaned}). "
            "Please write its NSE symbol, e.g. RELIANCE or HDFCBANK."
        )
    return cleaned, None


def parse_llm(
    description: str,
    *,
    answers: dict[str, str] | None = None,
    transport: Transport | None = None,
) -> TranslationResult | None:
    """The model's reading, checked. `None` means "no usable AI answer" -- use the rules'."""
    answers = answers or {}
    typed = description.strip()
    if not typed:
        return None
    # Read the sentence with the same typo fixes the rules made; the recipe
    # keeps the words as typed.
    original, spelling_notes, spelling_question = rules_spelling.prepare(typed, answers)
    if spelling_question is not None:
        return TranslationResult(spec=None, questions=[spelling_question], source="rules")
    lowered = original.lower()

    # Deterministic policy first, on the raw sentence. These are the rules'
    # own refusals; the model is never consulted past them.
    _, rules_kwargs, instrument_question = rules._extract_instrument(lowered, answers)
    if instrument_question is not None:
        return TranslationResult(spec=None, questions=[instrument_question], source="rules")
    # `_extract_instrument` gives no kwargs exactly when it gives a question.
    rules_kwargs = cast(dict, rules_kwargs)
    # The rules refuse futures on anything they have no contract for; this is
    # the backstop if a futures word ever got past them without a contract.
    if rules_kwargs.get("trade_as") != "future" and re.search(r"\bfutures?\b|\bfut\b", lowered):
        return _refusal("unsupported_instrument")
    # Same backstop for options: the rules reader (`options_words.read`) always
    # either tags a sentence as `trade_as="option"` or raises a question for
    # it. If an option word is still here and neither happened, something in
    # that reader missed it -- refuse rather than let the model invent a
    # strike, expiry or side the rules never checked.
    if rules_kwargs.get("trade_as") != "option" and options_words.OPTION_TRIGGER_FOR_LLM.search(
        lowered
    ):
        return _refusal("options")
    # A position of option legs (a sold option, a straddle, a condor) is read
    # by the rules alone: its legs, strikes, stops, targets, re-entry and entry
    # time all come from `options_words`. The model is never asked, so it can
    # never invent a leg; whatever the rules said (usually "I didn't
    # understand this part") stands.
    if rules_kwargs.get("legs"):
        return None

    masked, timeframe, timeframe_refusal = rules._extract_timeframe(lowered)
    if timeframe_refusal:
        return _question(
            timeframe_refusal,
            "Silently rounding to the nearest bar size we do support would run a "
            "different strategy than the one described.",
            "instrument.timeframe",
        )
    masked, backtest_note, period = rules._extract_backtest_window(masked)
    masked, overnight = rules._extract_overnight(masked)
    # "Stop at the low of the opening range": read by the rules' own words, as
    # the timeframe is; the model's reading of those words is set aside below.
    masked, level_stop, level_span = rules.read_level_stop(masked)

    if transport is None:
        if not is_configured():
            return None
        transport = openrouter_transport
    try:
        reply = transport(system_prompt(), original)
    except Exception:
        logger.exception("parse_llm: the model could not be reached")
        return None

    raw = _extract_json(reply)
    if raw is None:
        logger.warning("parse_llm: reply was not JSON: %r", reply[:500])
        return None
    _drop_redundant_outputs(raw)
    try:
        draft = Draft.model_validate(raw)
    except ValidationError as exc:
        logger.warning("parse_llm: reply did not fit the draft shape: %s", exc)
        return None

    if draft.kind == "refuse":
        quote = draft.unsupported_quote
        if quote and not _quote_pattern(quote).search(lowered):
            quote = None  # only ever echo the user's own words back
        return _refusal(draft.refuse_category or "vague", quote)

    # ---- every quote must be the user's own words, or the reply is discarded
    covered = [False] * len(masked)
    for ev in draft.evidence:
        m = _quote_pattern(ev.quote).search(lowered)
        if m is None:
            logger.warning("parse_llm: fabricated quote %r for %r", ev.quote, original)
            return None
        for i in range(m.start(), m.end()):
            covered[i] = True

    notes = [AI_NOTE, *spelling_notes]
    if backtest_note:
        notes.append(backtest_note)

    # ---- every meaningful word must be accounted for
    unparsed = []
    for m in re.finditer(r"[a-z0-9]+(?:[.'][a-z0-9]+)*", masked):
        if all(covered[m.start() : m.end()]):
            continue
        word = m.group()
        if word not in rules._FILLER:
            unparsed.append(word)
    if unparsed:
        return _question(
            f'I didn\'t understand this part: "{" ".join(unparsed)}". Did you mean something '
            "specific?",
            "Silently ignoring words you typed could mean the strategy misses a rule you intended.",
            "unparsed",
            notes=notes,
            unparsed=unparsed,
        )

    # ---- instrument: the rules' reading wins
    symbol, problem = _resolve_symbol(draft.symbol, rules_kwargs, lowered)
    if problem:
        return _question(
            problem,
            "Testing the wrong instrument gives a result that looks just as believable.",
            "instrument.symbol",
            notes=notes,
        )

    # ---- direction: the rules' reading of the leading verb wins
    _, rules_direction, explicit = rules._extract_direction(lowered)
    direction_quotes = " ".join(
        ev.quote.lower() for ev in draft.evidence if ev.field == "direction"
    )
    if explicit and draft.direction != rules_direction:
        problem = "I read this as both a buy and a sell. Which direction should it trade?"
    elif not explicit and not _DIRECTION_WORDS[draft.direction].search(direction_quotes):
        problem = "Should this strategy buy or sell short?"
    else:
        problem = None
    if problem:
        return _question(
            problem,
            "Trading the wrong direction turns every winning trade into a losing one.",
            "direction",
            notes=notes,
        )

    if draft.entry is None:
        return _question(
            "What should trigger this strategy to enter a trade?",
            "Without an entry rule there is nothing for the strategy to act on.",
            "entry.condition",
            notes=notes,
        )
    from nlt.spec.models import AtTime
    from nlt.translate.readback import _conditions_in

    if (timeframe or "1d") == "1d" and any(
        isinstance(c, AtTime) for c in _conditions_in(draft.entry)
    ):
        return _question(
            "A time of day needs minute or hourly candles. Say e.g. 'on 5-minute candles'.",
            "A daily candle has no clock: the time would be ignored, testing something else.",
            "instrument.timeframe",
            notes=notes,
        )

    # ---- a stop at a price level: ours, not the model's
    stop_level = None
    if level_stop is not None:
        if rules_kwargs.get("trade_as") == "option":
            return _question(
                rules.LEVEL_STOP_ON_OPTION,
                "Mixing the two would close the option at a point you never chose.",
                "exit.rule",
                notes=notes,
            )
        draft = _without_level_words(
            draft,
            lowered,
            cast(tuple[int, int], level_span),  # set with level_stop
        )
        level_ind = rules._Indicators()
        entry_dicts = [
            {"id": i.id, "type": i.type, "params": dict(i.params)} for i in draft.indicators
        ]
        stop_level, level_question = rules._resolve_level_stop(
            level_stop, entry_dicts, timeframe or "1d", level_ind
        )
        if level_question is not None:
            return TranslationResult(
                spec=None, questions=[level_question], notes=notes, source="llm"
            )
        known = {i.id for i in draft.indicators}
        extra = [IndicatorSpec(**d) for d in level_ind.as_list() if d["id"] not in known]
        draft = draft.model_copy(update={"indicators": [*draft.indicators, *extra]})

    # ---- numbers: nothing the user did not say
    by_field: dict[str, list[str]] = {}
    for ev in draft.evidence:
        by_field.setdefault(ev.field, []).append(ev.quote.lower())
    sentence_numbers = _numbers_in(lowered)

    def quoted(field: str, value: float, keyword: re.Pattern | None = None) -> bool:
        for q in by_field.get(field, []):
            if _has_number(value, _numbers_in(q)) and (keyword is None or keyword.search(q)):
                return True
        return False

    stray = []
    for value in _condition_numbers(draft.entry):
        if value != 0 and not quoted("entry", value):
            stray.append(value)
    for value in _condition_numbers(draft.exit.condition):
        if value != 0 and not quoted("exit.condition", value):
            stray.append(value)
    for value in _indicator_numbers(draft.indicators):
        if not _has_number(value, sentence_numbers):
            stray.append(value)

    ex = draft.exit
    exit_checks = [
        (ex.stop_pct, "exit.stop", _STOP_WORDS),
        (ex.trailing_stop_pct, "exit.trailing_stop", _TRAIL_WORDS),
        (ex.stop_atr_mult, "exit.atr_stop", _ATR_WORDS),
        (ex.target_pct, "exit.target", _TARGET_WORDS),
        (ex.max_bars_held, "exit.max_bars_held", None),
    ]
    for value, field, keyword in exit_checks:
        if value is not None and not quoted(field, float(value), keyword):
            stray.append(value)

    sizing = None
    if draft.sizing is not None:
        s = draft.sizing
        size_value = {"fixed_lots": s.lots, "fixed_value": s.value, "risk_based": s.risk_pct}[
            s.mode
        ]
        if size_value is None or not quoted("sizing", float(size_value)):
            stray.append(size_value)
        sizing = s

    square_off = None
    if draft.square_off is not None:
        try:
            square_off = dt.time.fromisoformat(draft.square_off)
        except ValueError:
            return None
        spellings = {
            f"{square_off.hour}:{square_off.minute:02d}",
            f"{(square_off.hour - 1) % 12 + 1}:{square_off.minute:02d}",
        }
        if not any(s in q for q in by_field.get("square_off", []) for s in spellings):
            stray.append(draft.square_off)

    if overnight and square_off is not None:
        picked = rules.overnight_answer(answers)
        if picked is None:
            return TranslationResult(
                spec=None,
                questions=[rules._overnight_conflict_question()],
                notes=notes,
                source="llm",
            )
        if picked == "square_off":
            overnight = False  # "overnight" needs nothing: holding overnight wins below

    if stray:
        logger.warning("parse_llm: numbers not in the sentence %r: %r", original, stray)
        return _question(
            "I couldn't match every number in my reading to your words "
            f"({', '.join(str(v) for v in stray)}). Please restate the strategy with each "
            "number spelled out next to what it is for.",
            "A number you never typed must never end up deciding a trade.",
            "unparsed",
            notes=notes,
        )

    # ---- the stop loss: stated by the user, or asked for
    stop_pct = ex.stop_pct
    has_stop = bool(stop_pct or ex.trailing_stop_pct or ex.stop_atr_mult or stop_level)
    if not has_stop:
        stop_pct = rules._float_answer(answers, "exit.stop_pct")
        if not stop_pct:
            return _stop_question(notes)

    instrument_kwargs: dict[str, Any] = {
        "symbol": symbol,
        "trade_as": rules_kwargs.get("trade_as") or ("index" if symbol in INDICES else "stock"),
    }
    if timeframe:
        instrument_kwargs["timeframe"] = timeframe
    if instrument_kwargs["trade_as"] == "option":
        # Option fields never come from the model -- only from the same rules
        # extractor that already resolved them before the model was ever asked
        # (`_extract_instrument` raises a question of its own for anything it
        # could not resolve, which `should_ask_llm` never lets this path see).
        for key in ("option_type", "strike_offset", "expiry"):
            if key in rules_kwargs:
                instrument_kwargs[key] = rules_kwargs[key]

    indicator_dicts = [
        {
            "id": i.id,
            "type": i.type,
            "params": dict(i.params),
            # "50-day SMA" on 15-minute candles is not 50 candles; the rules'
            # refusal below asks which was meant, on this path too.
            **({"in_days": True} if _length_said_in_days(i, lowered) else {}),
        }
        for i in draft.indicators
    ]
    vwap_question = rules._vwap_refusal(indicator_dicts, instrument_kwargs)
    if vwap_question is not None:
        return TranslationResult(spec=None, questions=[vwap_question], notes=notes, source="llm")

    entry, daily_limit, repeat_question, repeat_said = rules.range_repeat(
        draft.entry, draft.indicators, answers
    )
    if repeat_question is not None:
        return TranslationResult(spec=None, questions=[repeat_question], notes=notes, source="llm")

    exit_kwargs = ex.model_dump(exclude_none=True)
    exit_kwargs["stop_pct"] = stop_pct
    if stop_level is not None:
        exit_kwargs["stop_level"] = stop_level
    exit_kwargs = {k: v for k, v in exit_kwargs.items() if v is not None}
    schedule_kwargs = rules._schedule_kwargs(
        instrument_kwargs, square_off, None, overnight=overnight
    )
    if daily_limit is not None:
        schedule_kwargs["max_entries_per_day"] = daily_limit

    try:
        spec = StrategySpec(
            name=rules._name_from(original),
            description=typed,
            instrument=Instrument(**instrument_kwargs),
            indicators=draft.indicators,
            direction=draft.direction,
            entry=cast(Condition, entry),  # `draft.entry` was checked non-None above
            exit=ExitRules(**exit_kwargs),
            sizing=sizing or rules._default_sizing(instrument_kwargs),
            risk=rules._default_risk(instrument_kwargs),
            schedule=Schedule(**schedule_kwargs),
        )
    except ValidationError as exc:
        return _question(
            f"That description doesn't quite make a valid strategy yet: {exc.errors()[0]['msg']}",
            "The platform double-checks every strategy before it can trade; this one failed "
            "that check.",
            "spec",
            notes=notes,
        )

    spec = spec.model_copy(update={"name": rules.strategy_name(spec)})
    sources = rules.recipe_sources(
        spec,
        symbol_said=bool(rules_kwargs.get("symbol") or by_field.get("instrument")),
        timeframe_said=bool(timeframe),
        sizing_said=draft.sizing is not None,
        schedule_said=square_off is not None or overnight,
        premium_said=options_words.premium_mentioned(lowered),
    )
    if repeat_said:
        sources["schedule.daily_limit"] = readback.SAID
    return TranslationResult(
        spec=spec, questions=[], notes=notes, source="llm", period=period, sources=sources
    )
