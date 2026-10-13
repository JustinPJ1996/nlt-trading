"""The deterministic English-to-strategy parser.

There is no LLM in this path, on purpose. Every phrase this module understands is
listed explicitly below, in `CONDITION_PATTERNS`, `EXIT_PATTERNS` and friends --
that table IS the product's vocabulary today, and it is meant to be read, not just
executed. Extending what the platform understands means adding a row, not
reworking a monolithic regex or a wall of if/elif.

The governing rule, repeated because it is the one mistake this module must never
make: a strategy that reaches `StrategySpec` will eventually place real orders.
Anything this parser is not sure about must come back as a `Question`, never as a
best guess baked silently into the spec. In particular:

  * A missing stop loss is never defaulted in. We ask.
  * An indicator that is not in the registry is refused by name, never swapped
    for something "close enough".
  * Text we did not recognise is reported in `unparsed`, not dropped on the floor.

The crossing-vs-state distinction gets a whole section of its own below because it
is the highest-stakes piece of vocabulary here: "RSI cracks 30" (fires once, on
the bar the level is breached) and "RSI is below 30" (true on every bar it holds)
produce wildly different trade counts from the same number. Users say "cracks"
and "breaks" and mean the former; conflating the two turns one intended trade
into dozens of silent re-entries.

Two more judgment calls recur often enough to spell out here rather than only in
a code comment where they are made:

  * "at RSI 30" names a level with no up/down verb attached. We read it as a
    crossing (see `_build_at_rsi_level`) because "buy at RSI 30" is swing-trade
    shorthand for "the moment RSI gets there", not "for as long as it happens to
    sit there" -- and we pick the direction of the crossing from the trade's own
    direction (long -> falling to the level, short -> rising to it), the same
    trick `_build_pattern_needs_direction` already uses for "harami"/"marubozu".
  * "it" in a second clause ("...sell when it crosses 70") refers back to
    whichever indicator the first clause named. If the first clause named more
    than one, "it" is ambiguous and we ask rather than pick one -- see
    `_find_antecedent`.
"""

from __future__ import annotations

import datetime as dt
import functools
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import ValidationError

from nlt.indicators.patterns import PATTERNS
from nlt.indicators.registry import REGISTRY
from nlt.spec.models import (
    INDICES,
    All,
    Any_,
    Compare,
    Condition,
    Const,
    ExitRules,
    Instrument,
    IsTrue,
    PercentChange,
    Ref,
    RiskLimits,
    Schedule,
    Sizing,
    StrategySpec,
)

# --------------------------------------------------------------------- results


@dataclass
class Choice:
    """One tap-to-answer button: what it says, and the answer it sends."""

    label: str
    value: str


@dataclass
class Question:
    """Something we need the user to answer before we can trade their money on it.

    `choices` are buttons; nothing is ever pre-selected. `answerable` says the
    reader will take a typed answer for `field`. A question with neither can
    only be answered by changing the description, and the screen says so
    instead of showing a box whose answer would be ignored.
    """

    text: str
    why: str
    suggestion: str | None
    field: str
    choices: list[Choice] = field(default_factory=list)
    answerable: bool = False


@dataclass(frozen=True)
class BacktestPeriod:
    """A test period in the user's own terms: the last N units, or one calendar year.

    Relative, not dated: the screen turns it into dates on the day it runs, so
    the reader gives the same answer every day.
    """

    words: str
    last: int | None = None
    unit: str | None = None  # "days" | "weeks" | "months" | "years"
    year: int | None = None


# Tap-to-answer buttons. Common round numbers, offered side by side; none is
# chosen for the user, and "type your own" stays beside them.
STOP_CHOICES = [Choice("0.5%", "0.5"), Choice("1%", "1"), Choice("2%", "2")]
TARGET_CHOICES = [Choice("+2%", "2"), Choice("+3%", "3"), Choice("+5%", "5")]
OVERNIGHT_CHOICES = [
    Choice("Hold overnight", "overnight"),
    Choice("Square off the same day", "square_off"),
]


@dataclass
class TranslationResult:
    spec: StrategySpec | None
    questions: list[Question] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)
    source: str = "rules"
    # Where each part of the recipe came from, for the readback's labels:
    # part name -> readback.SAID / WORKED_OUT / DEFAULT. Only set with a spec.
    sources: dict[str, str] = field(default_factory=dict)
    # The test period the user asked for ("backtest the last 2 years"), kept
    # off the spec on purpose -- it belongs to a run, not to the strategy.
    period: BacktestPeriod | None = None


@dataclass
class _Indicators:
    """Accumulates indicator declarations, deduping by (type, params)."""

    by_key: dict[tuple[str, tuple], str] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    specs: dict[str, dict] = field(default_factory=dict)

    def declare(self, ind_id: str, ind_type: str, params: dict) -> str:
        key = (ind_type, tuple(sorted(params.items())))
        if key in self.by_key:
            return self.by_key[key]
        # id collision with a different indicator -- disambiguate deterministically.
        candidate = ind_id
        n = 2
        while candidate in self.specs:
            candidate = f"{ind_id}_{n}"
            n += 1
        self.by_key[key] = candidate
        self.order.append(candidate)
        self.specs[candidate] = {"type": ind_type, "params": params}
        return candidate

    def as_list(self) -> list[dict]:
        return [{"id": i, **self.specs[i]} for i in self.order]


# Filler words stripped out before deciding what counts as "unparsed" input.
_FILLER = {
    "please",
    "i",
    "want",
    "to",
    "a",
    "an",
    "the",
    "on",
    "in",
    "at",
    "with",
    "and",
    "or",
    "when",
    "if",
    "only",
    "but",
    "of",
    "for",
    "my",
    "strategy",
    "then",
    "so",
    "it",
    "is",
    "s",
    "this",
    "that",
    "would",
    "like",
    "d",
    "me",
    "build",
    "make",
    "create",
    "trade",
    "trading",
    "while",
    "as",
    "long",
    "still",
    "open",
    "just",
    "now",
    "up",
    "not",
    "candle",
    "candles",
    "pattern",
    "forms",
    "instantly",
    "use",
    "using",
}


def _clean(word: str) -> str:
    return word.strip(".,!?;:'\"()")


# ------------------------------------------------------------------ instrument
#
# `symbol` can name an index, a single NSE stock, or a universe (a basket of
# stocks screened and traded independently -- see the Instrument docstring in
# nlt/spec/models.py). Telling those apart is the highest-risk decision this
# module makes: an index and the universe of the same name are two completely
# different things to backtest, and both look like perfectly ordinary English.
#
# The rule, in order:
#   1. "NIFTY 100" / "NIFTY 500" are never index names in this platform (the
#      only tradeable indices are NIFTY and BANKNIFTY) -- so a mention of them
#      is unambiguous and always means the 100/500-stock universe, whether or
#      not the word "stocks" is nearby. Fixture #31-#33 rely on exactly this:
#      a bare "nifty 100" at the tail of a sentence, no "stocks" in sight.
#   2. "NIFTY 50" is genuinely ambiguous: "50" is both a stock count and the
#      literal name traders use for the index itself ("the Nifty 50 closed
#      up today" means the index). We only read it as the 50-stock universe
#      when a qualifier word -- "stocks", "shares", or "stocks in" -- says so.
#      Otherwise we ask, rather than guess which one the user meant; guessing
#      wrong here silently backtests fifty independent instruments instead of
#      one index, or vice versa, and both results look entirely plausible.
#   3. Bare "NIFTY" (no number) is the index, unchanged from before.
#
# A single bare stock ticker ("TCS", "RELIANCE") is recognised only against
# the bundled NIFTY 500 constituent list -- see `_known_tickers` -- never
# invented from an arbitrary English word that happens to be shaped like one.

_BANKNIFTY = re.compile(r"\bbank\s*nifty\b")
_NIFTY_UNIVERSE_NUM = re.compile(r"\bnifty\s*[\s\-_]?\s*(?P<num>50|100|500)\b")
_NIFTY_BARE = re.compile(r"\bnifty\b")
# SENSEX has no tradeable index or futures product here -- only options on it
# (see `nlt.translate.options_words.OPTION_UNDERLYINGS`) -- so finding the bare
# word here only stakes a claim on the symbol; whether that claim is allowed
# depends on whether the sentence turns out to be an option one.
_SENSEX = re.compile(r"\bsensex\b")

# "stocks"/"shares" immediately after the number ("nifty 100 stocks"), or
# "stocks in"/"shares in" immediately before it ("stocks in nifty 500").
_UNIVERSE_QUALIFIER_AFTER = re.compile(r"\A\s*(?:stocks?|shares?)\b")
_UNIVERSE_QUALIFIER_BEFORE = re.compile(r"(?:stocks?|shares?)\s+in\s*\Z")


@functools.lru_cache(maxsize=1)
def _known_tickers() -> frozenset[str]:
    """NSE symbols we will accept as a bare single-stock ticker.

    Sourced from the bundled NIFTY 500 constituent snapshot rather than a live
    NSE fetch or `get_universe`, deliberately: this parser has no network
    dependency anywhere else, and it must not gain one just to recognise a
    ticker -- that would make "does this sentence parse" depend on whether
    the box happens to be online, which breaks the determinism this whole
    module promises. NIFTY 500 covers the large majority of names anyone is
    likely to type; a symbol not in it is not invented, it falls through to
    `unparsed` like any other word we do not recognise.
    """
    from nlt.data.universe import _load_snapshot

    return frozenset(_load_snapshot()["universes"]["NIFTY 500"])


def _universe_qualifier_span(text: str, m: re.Match) -> tuple[int, int]:
    """Extend a matched 'nifty NNN' span to swallow an adjacent stocks/shares
    qualifier, so it disappears from the text instead of surfacing as
    unparsed leftover once the number itself has been understood."""
    start, end = m.start(), m.end()
    after = _UNIVERSE_QUALIFIER_AFTER.match(text[end:])
    if after:
        end += after.end()
    before = _UNIVERSE_QUALIFIER_BEFORE.search(text[:start])
    if before:
        start = before.start()
    return start, end


def _has_universe_qualifier(text: str, m: re.Match) -> bool:
    return bool(
        _UNIVERSE_QUALIFIER_AFTER.match(text[m.end() :])
        or _UNIVERSE_QUALIFIER_BEFORE.search(text[: m.start()])
    )


def _find_stock_ticker(text: str) -> tuple[str, int, int] | None:
    """A single bare word right after the leading buy/sell verb, if -- and
    only if -- it is a known NSE symbol (see `_known_tickers`).

    Returns (SYMBOL, start, end) with offsets into `text`, or None.

    Anchoring on the direction verb keeps ordinary English out: "buy" is
    almost always followed by what is being bought, so "buy TCS when..."
    offers up "tcs" as a candidate, while a random word elsewhere in the
    sentence never gets the chance. Multi-word names such as "HDFC Bank" do
    not match (the pattern is exactly one word), which is a deliberate
    refusal: guessing "HDFCBANK" from "HDFC Bank" is a guess this module will
    not make.
    """
    lead = _DIRECTION_LEAD.match(text)
    if not lead:
        return None
    word_m = re.match(r"\s+(?P<word>[a-z][a-z0-9&]{1,19})\b", text[lead.end() :])
    if not word_m:
        return None
    word = word_m.group("word")
    if word.upper() not in _known_tickers():
        return None
    start = lead.end() + word_m.start("word")
    end = lead.end() + word_m.end("word")
    return word.upper(), start, end


# ------------------------------------------------------------------- futures
#
# An MCX commodity can only be traded as a future, so naming one is enough --
# "buy crude when..." means Crude Oil futures, and the readback says so in
# those words. An index needs the word: "NIFTY futures" is the future, bare
# "NIFTY" stays the index. The mini contracts are matched before their
# full-size names, because one Gold lot is ten Gold Mini lots and confusing the
# two would size every trade tenfold.

_FUTURES_WORD = re.compile(r"\b(?:futures?|fut)\b")
_MCX_WORD = re.compile(r"\bmcx\b")
_COMMODITIES = (
    (
        "CRUDEOILM",
        re.compile(r"\bcrude\s*oil\s*m(?:ini)?\b|\bcrude\s*mini\b|\bmini\s*crude(?:\s*oil)?\b"),
    ),
    ("CRUDEOIL", re.compile(r"\bcrude\s*oil\b|\bcrude\b")),
    (
        "NATGASMINI",
        re.compile(r"\bnat(?:ural)?\s*gas\s*mini\b|\bmini\s*nat(?:ural)?\s*gas\b"),
    ),
    ("NATURALGAS", re.compile(r"\bnat(?:ural)?\s*gas\b")),
    ("GOLDM", re.compile(r"\bgold\s*m(?:ini)?\b|\bmini\s*gold\b")),
    ("GOLD", re.compile(r"\bgold\b")),
    ("SILVERM", re.compile(r"\bsilver\s*m(?:ini)?\b|\bmini\s*silver\b")),
    ("SILVER", re.compile(r"\bsilver\b")),
)
# Contracts that exist on MCX but not here. Named so they are refused, never
# read as the nearest supported name ("silver micro" is not Silver).
_UNSUPPORTED_COMMODITY = re.compile(
    r"\b(?:copper|zinc|alumin(?:i)?um|nickel|cotton|kapas|mentha\s*oil|cardamom|"
    r"steel\s*rebar|bulldex|metldex|leadmini|goldpetal|goldguinea|goldten|silvermic|silver100)\b"
    r"|\blead\s+(?:mini|futures?)\b|\bgold\s+(?:petal|guinea|ten)\b|\bsilver\s+(?:micro|mic|100)\b"
)


def _supported_futures_text() -> str:
    from nlt.data.futures import COMMODITIES, contract

    return "NIFTY, BANKNIFTY, and on MCX " + ", ".join(contract(c).title for c in COMMODITIES)


def _futures_question(what: str) -> Question:
    return Question(
        text=(
            f"I can't test {what} yet. Futures work on {_supported_futures_text()}. "
            "Try one of those, e.g. 'buy crude oil futures when RSI crosses below 30, "
            "target 2%, stop 1%'."
        ),
        why=(
            "Every futures contract has its own lot size and expiry dates. Guessing them "
            "for a contract this platform does not know would get every profit and loss "
            "figure wrong."
        ),
        suggestion=None,
        field="instrument.symbol",
    )


def _blank(text: str, pattern: re.Pattern) -> str:
    return pattern.sub(lambda m: " " * len(m.group()), text)


def _extract_instrument(
    text: str,
    answers: dict[str, str] | None = None,
) -> tuple[str, dict | None, Question | None]:
    """Returns (text, instrument kwargs, ambiguity question).

    `kwargs` is `None` exactly when `question` is set -- an ambiguous "NIFTY
    50" with no qualifier, the one case this function refuses to guess at.
    `answers` carries the user's picks from a previous options question
    (call/put, strike, expiry) back in.
    """
    kwargs: dict = {}

    unsupported = _UNSUPPORTED_COMMODITY.search(text)
    if unsupported:
        return text, None, _futures_question(f"'{unsupported.group()}'")
    commodity = next(((sym, pat) for sym, pat in _COMMODITIES if pat.search(text)), None)

    if _BANKNIFTY.search(text):
        kwargs["symbol"] = "BANKNIFTY"
        text = _BANKNIFTY.sub(" ", text)
    else:
        num_match = _NIFTY_UNIVERSE_NUM.search(text)
        if num_match:
            num = num_match.group("num")
            qualified = _has_universe_qualifier(text, num_match)
            if num == "50" and not qualified:
                question = Question(
                    text=(
                        "Does 'NIFTY 50' mean the index itself, or all 50 stocks in it? "
                        "Say 'NIFTY 50 stocks' for the basket of stocks, or just 'NIFTY' "
                        "for the index."
                    ),
                    why=(
                        "NIFTY 50 is both the name of the index and the name of the "
                        "50-stock basket it tracks. Backtesting the wrong one produces a "
                        "completely different result that would look just as plausible."
                    ),
                    suggestion=None,
                    field="instrument.symbol",
                )
                return text, None, question
            start, end = _universe_qualifier_span(text, num_match)
            kwargs["symbol"] = f"NIFTY {num}"
            kwargs["trade_as"] = "stock"
            text = text[:start] + " " * (end - start) + text[end:]
        elif _NIFTY_BARE.search(text):
            kwargs["symbol"] = "NIFTY"
            text = _NIFTY_BARE.sub(" ", text)
        elif _SENSEX.search(text):
            kwargs["symbol"] = "SENSEX"
            text = _SENSEX.sub(" ", text)

    # Options: buying one call or one put. Everything else an options sentence
    # might describe -- selling, two or more legs, a fixed strike, a points
    # offset -- is refused by `options_words.read` before it ever reaches here.
    from nlt.translate import options_words

    opt = options_words.read(text, answers)
    if opt.is_option:
        # The underlying is checked before anything else an option question
        # might ask -- there is no point asking "which strike?" of a sentence
        # we are about to refuse anyway for naming a stock, and defaulting it
        # to NIFTY in the meantime would silently drop the stock the user
        # actually named.
        symbol = kwargs.get("symbol")
        if symbol is not None and symbol not in options_words.OPTION_UNDERLYINGS:
            return text, None, options_words.unsupported_underlying_question(symbol)
        if symbol is None:
            stock = _find_stock_ticker(text)
            if stock is not None:
                return text, None, options_words.unsupported_underlying_question(stock[0])

        if opt.question is not None:
            return text, None, opt.question

        kwargs.update(opt.kwargs)
        if symbol is not None:
            kwargs["symbol"] = symbol
        # Else: no underlying named at all. Leave "symbol" out of kwargs
        # entirely -- `Instrument`'s own field default ("NIFTY") then applies
        # exactly the way it does for every other instrument, which is also
        # what makes `symbol_said` false and the readback say "Default"
        # rather than "You said".
        return opt.text, kwargs, None

    if kwargs.get("symbol") == "SENSEX":
        # SENSEX was named but this never turned out to be an option sentence
        # -- the index itself and SENSEX futures are not supported. A futures
        # word gets the same "Futures work on ..." refusal every other
        # unsupported contract gets, so the message stays one voice.
        if _FUTURES_WORD.search(text):
            return text, None, _futures_question("SENSEX futures")
        return text, None, options_words.sensex_non_option_question()

    futures_word = _FUTURES_WORD.search(text)
    if commodity is not None:
        symbol, pattern = commodity
        if "symbol" in kwargs:
            return (
                text,
                None,
                Question(
                    text=(
                        f"This names both {kwargs['symbol']} and an MCX commodity. Which one "
                        "should the strategy trade? (Signals on one instrument that trade "
                        "another aren't supported yet.)"
                    ),
                    why="Testing the wrong instrument gives a result that looks just as believable.",
                    suggestion=None,
                    field="instrument.symbol",
                ),
            )
        kwargs = {"symbol": symbol, "trade_as": "future"}
        for p in (pattern, _FUTURES_WORD, _MCX_WORD):
            text = _blank(text, p)
        return text, kwargs, None

    if futures_word:
        if kwargs.get("symbol") in INDICES:
            kwargs["trade_as"] = "future"
            return _blank(text, _FUTURES_WORD), kwargs, None
        return text, None, _futures_question("futures on that")

    if "symbol" not in kwargs:
        found = _find_stock_ticker(text)
        if found is not None:
            symbol, start, end = found
            kwargs["symbol"] = symbol
            kwargs["trade_as"] = "stock"
            text = text[:start] + " " * (end - start) + text[end:]

    return text, kwargs, None


# -------------------------------------------------------------------- direction

# Only the *leading* verb sets direction. Later occurrences of "sell" almost
# always describe an exit ("sell when RSI tops 70"), not a short entry -- so we
# deliberately do not scan the whole string for direction words. A handful of
# throat-clearing openers ("i want to", "please") are allowed before it, since
# "I want to buy NIFTY when..." is a completely ordinary way to open a request
# and treating it as unrecognised would be a false refusal on the most common
# possible phrasing.
_LEAD_FILLER = r"(?:please\s+)?(?:instantly\s+)?(?:i(?:'d| would)? (?:want to|like to)\s+)?"
_DIRECTION_LEAD = re.compile(
    rf"^\s*{_LEAD_FILLER}(buy|go long|long|sell short|short sell|short|go short|sell)\b"
)


def _extract_direction(text: str) -> tuple[str, str, bool]:
    """Returns (text, direction, explicit). `explicit` distinguishes "no verb
    found, defaulted to long" from "the user actually typed 'buy'" -- a named
    pattern like 'death cross' is allowed to set direction only in the former
    case, so it never overrides something the user said outright.
    """
    m = _DIRECTION_LEAD.match(text)
    if not m:
        return text, "long", False
    word = m.group(1)
    direction = "short" if "short" in word or word == "sell" else "long"
    text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
    return text, direction, True


# ---------------------------------------------------------- indicator ids


def _len_suffix(prefix: str, length: int | None, default: int) -> tuple[str, int]:
    n = length if length is not None else default
    return f"{prefix}{n}", n


_MA_TYPE_WORD = {
    "ma": "sma",
    "dma": "sma",  # "DMA" = daily moving average, the common name for a plain SMA
    "moving average": "sma",
    "sma": "sma",
    "ema": "ema",
    "wma": "wma",
    "hma": "hma",
}


def _num(s: str | None) -> float | None:
    return float(s) if s is not None else None


# -------------------------------------------------------- atomic condition table
#
# Each entry matches one self-contained clause ("rsi cracks 30", "20 ma crosses
# above 50 ma", ...) and builds a Condition plus whatever indicators it needs.
# `example` is the phrase a human typed to reach this row -- read it as the
# product's current vocabulary, not as documentation of the regex.


@dataclass
class PatternRule:
    name: str
    regex: re.Pattern[str]
    builder: Callable
    example: str


def _rsi(ind: _Indicators, length: int | None) -> Ref:
    ind_id, n = _len_suffix("rsi", length, 14)
    ind.declare(ind_id, "rsi", {"length": n})
    return Ref(name=ind_id, output=None)


def _build_rsi_cross(op: str):
    def build(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
        length = int(m.group("len")) if m.groupdict().get("len") else None
        ref = _rsi(ind, length)
        return Compare(op=op, left=ref, right=Const(value=float(m.group("level"))))

    return build


def _ma_ref(ind: _Indicators, kind: str, length: int) -> Ref:
    normalized = _MA_TYPE_WORD.get(kind.lower().strip(), "sma")
    ind_id, n = _len_suffix(normalized, length, 20)
    ind.declare(ind_id, normalized, {"length": n})
    return Ref(name=ind_id)


def _build_ma_cross(op: str):
    def build(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
        t1 = m.group("t1") or "ma"
        t2 = m.group("t2") or t1
        left = _ma_ref(ind, t1, int(m.group("len1")))
        right = _ma_ref(ind, t2, int(m.group("len2")))
        return Compare(op=op, left=left, right=right)

    return build


def _build_price_vs_ma(op: str):
    def build(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
        ref = _ma_ref(ind, m.group("t"), int(m.group("len")))
        return Compare(op=op, left=Ref(name="close"), right=ref)

    return build


def _build_golden_cross(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    left = _ma_ref(ind, "sma", 50)
    right = _ma_ref(ind, "sma", 200)
    return Compare(op="crosses_above", left=left, right=right)


def _build_death_cross(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    left = _ma_ref(ind, "sma", 50)
    right = _ma_ref(ind, "sma", 200)
    notes.append("'death cross' was read as the 50-day average crossing below the 200-day, short.")
    return Compare(op="crosses_below", left=left, right=right)


def _macd_ref(ind: _Indicators, output: str) -> Ref:
    ind.declare("macd", "macd", {})
    return Ref(name="macd", output=output)


def _build_macd_cross_signal(op: str):
    def build(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
        return Compare(op=op, left=_macd_ref(ind, "macd"), right=_macd_ref(ind, "signal"))

    return build


def _build_macd_positive(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    return Compare(op="crosses_above", left=_macd_ref(ind, "macd"), right=Const(value=0))


def _build_macd_negative(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    return Compare(op="crosses_below", left=_macd_ref(ind, "macd"), right=Const(value=0))


def _build_macd_crossover(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    notes.append("'MACD crossover' with no direction stated was read as crossing above signal.")
    return Compare(op="crosses_above", left=_macd_ref(ind, "macd"), right=_macd_ref(ind, "signal"))


def _adx_ref(ind: _Indicators, length: int | None) -> Ref:
    ind_id, n = _len_suffix("adx", length, 14)
    ind.declare(ind_id, "adx", {"length": n})
    return Ref(name=ind_id, output="adx")


def _build_adx_gt(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    length = int(m.group("len")) if m.groupdict().get("len") else None
    return Compare(op="gt", left=_adx_ref(ind, length), right=Const(value=float(m.group("level"))))


def _build_trend_strong(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    notes.append("'the trend is strong' was read as ADX(14) above 25.")
    return Compare(op="gt", left=_adx_ref(ind, None), right=Const(value=25))


def _build_supertrend_bullish(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    ind.declare("supertrend10", "supertrend", {})
    return Compare(
        op="crosses_above", left=Ref(name="supertrend10", output="direction"), right=Const(value=0)
    )


def _build_supertrend_bearish(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    ind.declare("supertrend10", "supertrend", {})
    return Compare(
        op="crosses_below", left=Ref(name="supertrend10", output="direction"), right=Const(value=0)
    )


def _bb_ref(ind: _Indicators, length: int | None, output: str) -> Ref:
    ind_id, n = _len_suffix("bb", length, 20)
    ind.declare(ind_id, "bollinger", {"length": n})
    return Ref(name=ind_id, output=output)


def _build_touch_lower_band(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    return Compare(op="lte", left=Ref(name="close"), right=_bb_ref(ind, None, "lower"))


def _build_touch_upper_band(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    return Compare(op="gte", left=Ref(name="close"), right=_bb_ref(ind, None, "upper"))


def _build_bb_squeeze(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    notes.append(
        "'bollinger squeeze' was read as bandwidth below 0.1 -- an arbitrary but "
        "conservative threshold; adjust it if this fires too often or too rarely."
    )
    return Compare(op="lt", left=_bb_ref(ind, None, "bandwidth"), right=Const(value=0.1))


def _osc_builder(kind: str, ind_type: str, output: str, default_len: int):
    def _ref(ind: _Indicators, length: int | None) -> Ref:
        ind_id, n = _len_suffix(kind, length, default_len)
        ind.declare(ind_id, ind_type, {"length": n})
        return Ref(name=ind_id, output=output)

    def build(op: str):
        def inner(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
            length = int(m.group("len")) if m.groupdict().get("len") else None
            value = float(m.group("level"))
            return Compare(op=op, left=_ref(ind, length), right=Const(value=value))

        return inner

    return build


_stoch_builder = _osc_builder("stoch", "stochastic", "k", 14)
_cci_builder = _osc_builder("cci", "cci", "cci", 20)
_wr_builder = _osc_builder("wr", "williams_r", "williams_r", 14)


def _prior_ref(ind: _Indicators, period: str, output: str) -> Ref:
    ind_id = "prior_week" if period == "week" else "prior_day"
    ind.declare(ind_id, "prior_period", {"period": period})
    return Ref(name=ind_id, output=output)


def _build_prior_level(op: str, output: str, period: str):
    def build(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
        return Compare(op=op, left=Ref(name="close"), right=_prior_ref(ind, period, output))

    return build


def _extremes_ref(ind: _Indicators, length: int, output: str) -> Ref:
    ind_id = f"extremes{length}"
    ind.declare(ind_id, "rolling_extremes", {"length": length})
    return Ref(name=ind_id, output=output)


def _build_breakout_high(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    length = int(m.group("len"))
    return Compare(
        op="crosses_above", left=Ref(name="close"), right=_extremes_ref(ind, length, "highest")
    )


def _build_breakout_low(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    length = int(m.group("len"))
    return Compare(
        op="crosses_below", left=Ref(name="close"), right=_extremes_ref(ind, length, "lowest")
    )


_PATTERN_ALIASES = {
    "bullish engulfing": "engulfing_bullish",
    "engulfing bullish": "engulfing_bullish",
    "bearish engulfing": "engulfing_bearish",
    "engulfing bearish": "engulfing_bearish",
    "hammer": "hammer",
    "hanging man": "hanging_man",
    "doji": "doji",
    "morning star": "morning_star",
    "evening star": "evening_star",
    "shooting star": "shooting_star",
    "bullish harami": "harami_bullish",
    "bearish harami": "harami_bearish",
    "bullish marubozu": "marubozu_bullish",
    "bearish marubozu": "marubozu_bearish",
    "spinning top": "spinning_top",
}


def _build_pattern(pattern_id: str):
    def build(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
        ind.declare(pattern_id, pattern_id, {})
        return IsTrue(ref=Ref(name=pattern_id))

    return build


def _build_pattern_needs_direction(base: str, default_note: str):
    """'harami'/'marubozu' with no bullish/bearish qualifier -- pick from direction."""

    def build(
        m: re.Match, ind: _Indicators, notes: list[str], direction: str = "long"
    ) -> Condition:
        variant = f"{base}_{'bullish' if direction == 'long' else 'bearish'}"
        notes.append(default_note.format(variant=variant))
        ind.declare(variant, variant, {})
        return IsTrue(ref=Ref(name=variant))

    return build


def _build_pct_change(op: str, sign: float):
    def build(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
        lookback = int(m.group("bars")) if m.groupdict().get("bars") else 1
        value = sign * float(m.group("pct"))
        return PercentChange(ref=Ref(name="close"), lookback=lookback, op=op, value=value)

    return build


# `len(word)` groups below use a shared vocabulary of "crossing" verbs vs "state"
# verbs. See the module docstring: this split is the single highest-stakes rule
# in the file, so it gets named constants rather than being buried in regex soup.
_CROSS_DOWN_WORDS = (
    r"cracks|crosses below|crosses under|falls below|falls under|breaks below|breaks under"
    r"|dips under|dips below|drops below|drops under|goes below|goes under|slips below"
    r"|moves below|sinks below"
)
_CROSS_UP_WORDS = (
    r"crosses above|crosses over|breaks above|breaks out above|goes above|goes over"
    r"|rises above|rises over|climbs above|cracks above|moves above|pops above"
)
# Users often name the subject before the verb ("price closes above ..."). Without
# this the subject is left over and surfaces as a spurious "I did not understand".
_OPT_PRICE = r"(?:(?:the\s*)?(?:price|close|nifty|banknifty|it)\s*)?"

_STATE_DOWN_WORDS = r"is below|is under|while below|as long as it'?s under|below|under|<"
_STATE_UP_WORDS = r"is above|is over|while above|above|over|>"

_LEN = r"(?:\s*\(?\s*(?P<len>\d{1,3})\s*\)?)?"

_AT_RSI_LEVEL = re.compile(r"\bat rsi\s*(?P<level>\d+(?:\.\d+)?)\b")


def _build_at_rsi_level(
    m: re.Match, ind: _Indicators, notes: list[str], direction: str = "long"
) -> Condition:
    """ "at RSI 30" names a level with no up/down verb -- see the module docstring
    for why we read it as a crossing whose direction follows the trade direction,
    rather than as a state true on every bar.

    Deliberately does not share `_LEN` with the other RSI patterns: with nothing
    but whitespace between an optional inline length and the mandatory level
    ("at rsi14 30"), a greedy-then-backtracking `_LEN` can eat part of the level
    itself (matching "at rsi 30" as length="3", level="0"). Keeping this pattern
    to the plain "at RSI <level>" phrasing users actually type avoids that trap
    outright rather than fixing it with a more delicate regex.
    """
    ref = _rsi(ind, None)
    op = "crosses_below" if direction == "long" else "crosses_above"
    verb = "dropping to" if direction == "long" else "rising to"
    notes.append(
        f"'at RSI {m.group('level')}' was read as RSI {verb} that level, based on "
        f"the {direction} direction of the trade -- say 'crosses below'/'crosses "
        "above' explicitly if that is not what you meant."
    )
    return Compare(op=op, left=ref, right=Const(value=float(m.group("level"))))


def _try_at_rsi_level(
    clause: str, ind: _Indicators, notes: list[str], direction: str
) -> tuple[Condition, str] | None:
    m = _AT_RSI_LEVEL.search(clause)
    if not m:
        return None
    cond = _build_at_rsi_level(m, ind, notes, direction=direction)
    residual = clause[: m.start()] + " " + clause[m.end() :]
    return cond, residual


_RSI_BETWEEN = re.compile(
    rf"rsi{_LEN}\s*is\s*between\s*(?P<lo>\d+(?:\.\d+)?)\s*(?:-|to|and)\s*(?P<hi>\d+(?:\.\d+)?)"
)


def _build_rsi_between(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    length = int(m.group("len")) if m.groupdict().get("len") else None
    ref = _rsi(ind, length)
    lo, hi = float(m.group("lo")), float(m.group("hi"))
    return All(
        conditions=[
            Compare(op="gte", left=ref, right=Const(value=lo)),
            Compare(op="lte", left=ref, right=Const(value=hi)),
        ]
    )


# ------------------------------------------------------ indicator-vs-indicator
#
# "ema8 > ema21", "ema8 crosses above ema21", "close > vwap" -- both sides name
# a value rather than one side being a fixed number. `Compare` already allows
# this (see `nlt/spec/models.py`); what was missing was a way to *recognise* it
# in English. `_OPERAND_TOKEN` is deliberately narrow: only the handful of
# things a user is likely to put on either side of a bare comparison verb, in
# the inline-suffix spelling ("ema8", not "8 ema" -- that format already has
# its own patterns above). Widening this alternation is exactly the kind of
# change that risks swallowing something it should not, so anything not listed
# here still falls through to `unparsed` rather than being guessed at.
_OPERAND_TOKEN = r"(?:close|price|vwap|rsi\d{0,3}|adx\d{0,3}|(?:ema|sma|wma|hma)\d{1,4})"


def _operand_ref(token: str, ind: _Indicators) -> Ref:
    if token in ("close", "price"):
        return Ref(name="close")
    if token == "vwap":
        ind.declare("vwap", "vwap", {})
        return Ref(name="vwap")
    m = re.fullmatch(r"rsi(\d{0,3})", token)
    if m:
        length = int(m.group(1)) if m.group(1) else None
        return _rsi(ind, length)
    m = re.fullmatch(r"adx(\d{0,3})", token)
    if m:
        length = int(m.group(1)) if m.group(1) else None
        return _adx_ref(ind, length)
    m = re.fullmatch(r"(ema|sma|wma|hma)(\d{1,4})", token)
    if m:
        return _ma_ref(ind, m.group(1), int(m.group(2)))
    raise AssertionError(f"unhandled operand token {token!r}")  # pragma: no cover


def _build_generic_compare(op: str):
    def build(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
        left = _operand_ref(m.group("left"), ind)
        right = _operand_ref(m.group("right"), ind)
        return Compare(op=op, left=left, right=right)

    return build


def _stoch_kd_ref(ind: _Indicators, output: str) -> Ref:
    ind.declare("stoch14", "stochastic", {"length": 14})
    return Ref(name="stoch14", output=output)


def _build_stoch_k_cross_d(op: str):
    def build(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
        return Compare(op=op, left=_stoch_kd_ref(ind, "k"), right=_stoch_kd_ref(ind, "d"))

    return build


def _build_stoch_k_level(op: str):
    def build(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
        return Compare(op=op, left=_stoch_kd_ref(ind, "k"), right=Const(value=float(m["level"])))

    return build


def _build_extreme_level(op: str, output: str):
    """ "close above the highest high of the last 20 days" -- a Donchian channel edge."""

    def build(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
        length = int(m.group("len"))
        return Compare(op=op, left=Ref(name="close"), right=_extremes_ref(ind, length, output))

    return build


def _build_opening_range(op: str, output: str):
    """ "breaks above the high of the first 15 minutes" -- only once those minutes end."""

    def build(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
        minutes = int(m.group("min") or m.group("min2"))
        ind_id = f"orange{minutes}"
        ind.declare(ind_id, "opening_range", {"minutes": minutes})
        return Compare(op=op, left=Ref(name="close"), right=Ref(name=ind_id, output=output))

    return build


def _build_daily_close_vs_average(op: str):
    """ "the daily close is above the 50-day SMA" on an intraday chart.

    Read as the previous day's close against its average, because today's
    daily close does not exist yet; the readback says "previous day" out loud.
    """

    def build(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
        length = int(m.group("len"))
        kind = "ema" if m.group("kind").startswith(("ema", "exp")) else "sma"
        ind_id = f"daily_{kind}{length}"
        ind.declare(ind_id, "daily_average", {"length": length, "kind": kind})
        return Compare(
            op=op, left=Ref(name=ind_id, output="close"), right=Ref(name=ind_id, output="average")
        )

    return build


_BARS_WORD = r"(?:days?|bars?|candles?|sessions?)"
_RANGE_HIGH_OR_LOW = (
    r"(?:the\s*)?(?:{hl}\s*of\s*the\s*first\s*(?P<min>\d{{1,3}})\s*(?:minutes?|mins?)"
    r"|first\s*(?P<min2>\d{{1,3}})\s*-?\s*min(?:ute)?s?\s*{hl})"
)
_DAILY_AVG_KIND = (
    r"(?P<kind>sma|ema|simple\s*moving\s*average|exponential\s*moving\s*average"
    r"|moving\s*average|average)"
)
_DAILY_CLOSE = r"(?:the\s*)?(?:daily|previous\s*day'?s?|yesterday'?s?)\s*close\s*(?:is\s*)?"

_A6_PATTERNS: list[PatternRule] = [
    PatternRule(
        "stoch_k_cross_up_d",
        re.compile(rf"(?:stochastic\s*)?%\s*k\s*(?:{_CROSS_UP_WORDS})\s*(?:the\s*)?%\s*d"),
        _build_stoch_k_cross_d("crosses_above"),
        "%K crosses above %D",
    ),
    PatternRule(
        "stoch_k_cross_down_d",
        re.compile(rf"(?:stochastic\s*)?%\s*k\s*(?:{_CROSS_DOWN_WORDS})\s*(?:the\s*)?%\s*d"),
        _build_stoch_k_cross_d("crosses_below"),
        "%K crosses below %D",
    ),
    PatternRule(
        "stoch_k_up_level",
        re.compile(rf"(?:stochastic\s*)?%\s*k\s*(?:{_CROSS_UP_WORDS})\s*(?P<level>\d+(?:\.\d+)?)"),
        _build_stoch_k_level("crosses_above"),
        "%K goes above 80",
    ),
    PatternRule(
        "stoch_k_down_level",
        re.compile(
            rf"(?:stochastic\s*)?%\s*k\s*(?:{_CROSS_DOWN_WORDS})\s*(?P<level>\d+(?:\.\d+)?)"
        ),
        _build_stoch_k_level("crosses_below"),
        "%K falls below 20",
    ),
    PatternRule(
        "above_highest_high",
        re.compile(
            rf"{_OPT_PRICE}(?:is\s*)?(?:above|over)\s*(?:the\s*)?highest\s*high\s*"
            rf"(?:of|in|over)\s*(?:the\s*)?(?:last|past|previous)\s*(?P<len>\d{{1,3}})\s*{_BARS_WORD}"
        ),
        _build_extreme_level("gt", "highest"),
        "close is above the highest high of the last 20 days",
    ),
    PatternRule(
        "below_lowest_low",
        re.compile(
            rf"{_OPT_PRICE}(?:{_CROSS_DOWN_WORDS})\s*(?:the\s*)?lowest\s*low\s*"
            rf"(?:of|in|over)\s*(?:the\s*)?(?:last|past|previous)\s*(?P<len>\d{{1,3}})\s*{_BARS_WORD}"
        ),
        _build_extreme_level("crosses_below", "lowest"),
        "close falls below the lowest low of the last 10 days",
    ),
    PatternRule(
        "is_below_lowest_low",
        re.compile(
            rf"{_OPT_PRICE}(?:is\s*)?(?:below|under)\s*(?:the\s*)?lowest\s*low\s*"
            rf"(?:of|in|over)\s*(?:the\s*)?(?:last|past|previous)\s*(?P<len>\d{{1,3}})\s*{_BARS_WORD}"
        ),
        _build_extreme_level("lt", "lowest"),
        "close is below the lowest low of the last 10 days",
    ),
    PatternRule(
        "breaks_opening_range_high",
        re.compile(
            rf"{_OPT_PRICE}(?:{_CROSS_UP_WORDS}|closes\s*above)\s*"
            + _RANGE_HIGH_OR_LOW.format(hl="high")
        ),
        _build_opening_range("crosses_above", "high"),
        "breaks above the high of the first 15 minutes",
    ),
    PatternRule(
        "breaks_opening_range_low",
        re.compile(
            rf"{_OPT_PRICE}(?:{_CROSS_DOWN_WORDS}|closes\s*below)\s*"
            + _RANGE_HIGH_OR_LOW.format(hl="low")
        ),
        _build_opening_range("crosses_below", "low"),
        "breaks below the low of the first 15 minutes",
    ),
    PatternRule(
        "daily_close_above_average",
        re.compile(
            rf"{_DAILY_CLOSE}(?:above|over)\s*(?:the\s*|its\s*)?(?P<len>\d{{1,3}})\s*-?\s*day\s*{_DAILY_AVG_KIND}"
        ),
        _build_daily_close_vs_average("gt"),
        "the daily close is above the 50-day SMA",
    ),
    PatternRule(
        "daily_close_below_average",
        re.compile(
            rf"{_DAILY_CLOSE}(?:below|under)\s*(?:the\s*|its\s*)?(?P<len>\d{{1,3}})\s*-?\s*day\s*{_DAILY_AVG_KIND}"
        ),
        _build_daily_close_vs_average("lt"),
        "the daily close is below the 50-day SMA",
    ),
]


CONDITION_PATTERNS: list[PatternRule] = [
    *_A6_PATTERNS,
    PatternRule(
        "rsi_crosses_below",
        re.compile(rf"rsi{_LEN}\s*(?:{_CROSS_DOWN_WORDS})\s*(?P<level>\d+(?:\.\d+)?)"),
        _build_rsi_cross("crosses_below"),
        "RSI cracks 30",
    ),
    PatternRule(
        "rsi_crosses_above",
        re.compile(rf"rsi{_LEN}\s*(?:{_CROSS_UP_WORDS})\s*(?P<level>\d+(?:\.\d+)?)"),
        _build_rsi_cross("crosses_above"),
        "RSI crosses above 70",
    ),
    PatternRule(
        "rsi_lt",
        re.compile(rf"rsi{_LEN}\s*(?:{_STATE_DOWN_WORDS})\s*(?P<level>\d+(?:\.\d+)?)"),
        _build_rsi_cross("lt"),
        "RSI is below 30",
    ),
    PatternRule(
        "rsi_gt",
        re.compile(rf"rsi{_LEN}\s*(?:{_STATE_UP_WORDS})\s*(?P<level>\d+(?:\.\d+)?)"),
        _build_rsi_cross("gt"),
        "RSI is above 70",
    ),
    PatternRule(
        "golden_cross",
        re.compile(r"golden cross"),
        _build_golden_cross,
        "golden cross",
    ),
    PatternRule(
        "death_cross",
        re.compile(r"death cross"),
        _build_death_cross,
        "death cross",
    ),
    PatternRule(
        "ma_crosses_above",
        re.compile(
            r"(?P<len1>\d{1,3})\s*(?:day\s*)?(?P<t1>ema|sma|wma|hma|dma|ma|moving average)\s*"
            r"crosses(?:\s*above)?\s*(?:the\s*)?"
            r"(?P<len2>\d{1,3})\s*(?:day\s*)?(?P<t2>ema|sma|wma|hma|dma|ma|moving average)?"
        ),
        _build_ma_cross("crosses_above"),
        "20 MA crosses above 50 MA",
    ),
    PatternRule(
        "ma_crosses_below",
        re.compile(
            r"(?P<len1>\d{1,3})\s*(?:day\s*)?(?P<t1>ema|sma|wma|hma|dma|ma|moving average)\s*"
            r"crosses below\s*(?:the\s*)?"
            r"(?P<len2>\d{1,3})\s*(?:day\s*)?(?P<t2>ema|sma|wma|hma|dma|ma|moving average)?"
        ),
        _build_ma_cross("crosses_below"),
        "20 MA crosses below 50 MA",
    ),
    PatternRule(
        "ma_above",
        re.compile(
            r"(?P<len1>\d{1,3})\s*(?:day\s*)?(?P<t1>ema|sma|wma|hma|dma|ma|moving average)\s*above\s*"
            r"(?:the\s*)?(?P<len2>\d{1,3})\s*(?:day\s*)?(?P<t2>ema|sma|wma|hma|dma|ma|moving average)?"
        ),
        _build_ma_cross("gt"),
        "20 EMA above 200 EMA",
    ),
    PatternRule(
        "price_above_ma",
        re.compile(
            # "price"/"close" is optional: "Nifty 100 stocks above the 200 DMA"
            # has no explicit subject, but with the instrument already
            # resolved (the universe or a bare ticker), "above the N-day
            # average" can only sensibly mean each stock's own price.
            r"(?:(?:price|close)\s*(?:is\s*)?)?above\s*(?:the\s*)?"
            r"(?P<len>\d{1,3})\s*(?:day\s*)?(?P<t>ema|sma|wma|hma|dma|ma|moving average)"
        ),
        _build_price_vs_ma("gt"),
        "price above the 200 day moving average",
    ),
    PatternRule(
        "price_below_ma",
        re.compile(
            r"(?:(?:price|close)\s*(?:is\s*)?)?below\s*(?:the\s*)?"
            r"(?P<len>\d{1,3})\s*(?:day\s*)?(?P<t>ema|sma|wma|hma|dma|ma|moving average)"
        ),
        _build_price_vs_ma("lt"),
        "close below the 50 EMA",
    ),
    PatternRule(
        "macd_crosses_above_signal",
        re.compile(r"macd\s*crosses\s*above\s*(?:the\s*)?signal"),
        _build_macd_cross_signal("crosses_above"),
        "MACD crosses above signal",
    ),
    PatternRule(
        "macd_crosses_below_signal",
        re.compile(r"macd\s*crosses\s*below\s*(?:the\s*)?signal"),
        _build_macd_cross_signal("crosses_below"),
        "MACD crosses below signal",
    ),
    PatternRule(
        "macd_turns_positive",
        re.compile(r"macd\s*turns\s*positive"),
        _build_macd_positive,
        "MACD turns positive",
    ),
    PatternRule(
        "macd_turns_negative",
        re.compile(r"macd\s*turns\s*negative"),
        _build_macd_negative,
        "MACD turns negative",
    ),
    PatternRule(
        "macd_crossover",
        re.compile(r"macd\s*crossover"),
        _build_macd_crossover,
        "MACD crossover",
    ),
    PatternRule(
        "adx_above",
        re.compile(
            rf"adx{_LEN}\s*(?:{_STATE_UP_WORDS}|{_CROSS_UP_WORDS})\s*(?P<level>\d+(?:\.\d+)?)"
        ),
        _build_adx_gt,
        "ADX above 25",
    ),
    PatternRule(
        "trend_is_strong",
        re.compile(r"(?:the\s*)?trend\s*is\s*strong"),
        _build_trend_strong,
        "when the trend is strong",
    ),
    PatternRule(
        "supertrend_bullish",
        re.compile(r"supertrend\s*(?:turns\s*)?(?:green|bullish|positive)"),
        _build_supertrend_bullish,
        "supertrend turns green",
    ),
    PatternRule(
        "supertrend_bearish",
        re.compile(r"supertrend\s*(?:turns\s*)?(?:red|bearish|negative)"),
        _build_supertrend_bearish,
        "supertrend turns red",
    ),
    PatternRule(
        "bb_touch_lower",
        re.compile(r"(?:price\s*)?touches?\s*(?:the\s*)?lower\s*bollinger\s*band"),
        _build_touch_lower_band,
        "price touches the lower bollinger band",
    ),
    PatternRule(
        "bb_close_above_upper",
        re.compile(r"closes?\s*above\s*(?:the\s*)?upper\s*(?:bollinger\s*)?band"),
        _build_touch_upper_band,
        "closes above the upper band",
    ),
    PatternRule(
        "bb_squeeze",
        re.compile(r"bollinger\s*squeeze"),
        _build_bb_squeeze,
        "bollinger squeeze",
    ),
    PatternRule(
        "stoch_below",
        re.compile(
            rf"stochastic{_LEN}\s*(?:{_STATE_DOWN_WORDS}|{_CROSS_DOWN_WORDS})\s*(?P<level>\d+(?:\.\d+)?)"
        ),
        _stoch_builder("lt"),
        "stochastic below 20",
    ),
    PatternRule(
        "stoch_above",
        re.compile(
            rf"stochastic{_LEN}\s*(?:{_STATE_UP_WORDS}|{_CROSS_UP_WORDS})\s*(?P<level>\d+(?:\.\d+)?)"
        ),
        _stoch_builder("gt"),
        "stochastic above 80",
    ),
    PatternRule(
        "cci_below",
        re.compile(
            rf"cci{_LEN}\s*(?:{_STATE_DOWN_WORDS}|{_CROSS_DOWN_WORDS})\s*(?P<level>-?\d+(?:\.\d+)?)"
        ),
        _cci_builder("lt"),
        "CCI below -100",
    ),
    PatternRule(
        "cci_above",
        re.compile(
            rf"cci{_LEN}\s*(?:{_STATE_UP_WORDS}|{_CROSS_UP_WORDS})\s*(?P<level>-?\d+(?:\.\d+)?)"
        ),
        _cci_builder("gt"),
        "CCI above 100",
    ),
    PatternRule(
        "williams_below",
        re.compile(
            rf"williams\s*%?r{_LEN}\s*(?:{_STATE_DOWN_WORDS}|{_CROSS_DOWN_WORDS})\s*(?P<level>-?\d+(?:\.\d+)?)"
        ),
        _wr_builder("lt"),
        "Williams %R below -80",
    ),
    PatternRule(
        "williams_above",
        re.compile(
            rf"williams\s*%?r{_LEN}\s*(?:{_STATE_UP_WORDS}|{_CROSS_UP_WORDS})\s*(?P<level>-?\d+(?:\.\d+)?)"
        ),
        _wr_builder("gt"),
        "Williams %R above -20",
    ),
    PatternRule(
        "breaks_prev_day_high",
        re.compile(
            _OPT_PRICE + r"(?:breaks|crosses above)\s*(?:the\s*)?(?:previous|prior|yesterday'?s?)"
            r"\s*(?:day'?s?\s*)?high"
        ),
        _build_prior_level("crosses_above", "high", "day"),
        "breaks yesterday's high",
    ),
    PatternRule(
        "below_prev_day_low",
        re.compile(
            _OPT_PRICE
            + r"(?:closes?\s*)?(?:below|under)\s*(?:the\s*)?(?:previous|prior|yesterday'?s?)"
            r"\s*(?:day'?s?\s*)?low"
        ),
        _build_prior_level("lt", "low", "day"),
        "below the previous day's low",
    ),
    PatternRule(
        "closes_above_prev_day_high",
        re.compile(
            _OPT_PRICE + r"closes?\s*above\s*(?:the\s*)?(?:previous|prior|yesterday'?s?)"
            r"\s*(?:day'?s?\s*)?high"
        ),
        _build_prior_level("gt", "high", "day"),
        "closes above the previous day's high",
    ),
    PatternRule(
        "below_last_week_low",
        re.compile(r"(?:below|under)\s*(?:last|the\s*previous)\s*week'?s?\s*low"),
        _build_prior_level("lt", "low", "week"),
        "below last week's low",
    ),
    PatternRule(
        "breakout_high",
        re.compile(r"(?:breaks?\s*out\s*to\s*a|new)\s*(?P<len>\d{1,3})\s*day\s*high"),
        _build_breakout_high,
        "breaks out to a 50 day high",
    ),
    PatternRule(
        "bare_high",
        re.compile(r"(?P<len>\d{1,3})\s*day\s*high"),
        _build_breakout_high,
        "20 day high",
    ),
    PatternRule(
        "breakout_low",
        re.compile(r"(?:breaks?\s*out\s*to\s*a|new)\s*(?P<len>\d{1,3})\s*day\s*low"),
        _build_breakout_low,
        "breaks out to a 50 day low",
    ),
    PatternRule(
        "rsi_between",
        _RSI_BETWEEN,
        _build_rsi_between,
        "RSI is between 40-60",
    ),
    PatternRule(
        "generic_crosses_below",
        re.compile(
            rf"(?P<left>{_OPERAND_TOKEN})\s*(?:{_CROSS_DOWN_WORDS})\s*(?P<right>{_OPERAND_TOKEN})"
        ),
        _build_generic_compare("crosses_below"),
        "ema8 crosses below ema21",
    ),
    PatternRule(
        "generic_crosses_above",
        re.compile(
            rf"(?P<left>{_OPERAND_TOKEN})\s*(?:{_CROSS_UP_WORDS})\s*(?P<right>{_OPERAND_TOKEN})"
        ),
        _build_generic_compare("crosses_above"),
        "ema8 crosses above ema21",
    ),
    PatternRule(
        "generic_lt",
        re.compile(
            rf"(?P<left>{_OPERAND_TOKEN})\s*(?:{_STATE_DOWN_WORDS})\s*(?P<right>{_OPERAND_TOKEN})"
        ),
        _build_generic_compare("lt"),
        "close below vwap",
    ),
    PatternRule(
        "generic_gt",
        re.compile(
            rf"(?P<left>{_OPERAND_TOKEN})\s*(?:{_STATE_UP_WORDS})\s*(?P<right>{_OPERAND_TOKEN})"
        ),
        _build_generic_compare("gt"),
        "close above vwap",
    ),
    PatternRule(
        "pct_falls",
        re.compile(
            r"falls?\s*(?P<pct>\d+(?:\.\d+)?)\s*%(?:\s*in\s*(?P<bars>\d+)\s*(?:days?|bars?))?"
        ),
        _build_pct_change("lte", -1.0),
        "NIFTY falls 1%",
    ),
    PatternRule(
        "pct_drops",
        re.compile(
            r"drops?\s*(?P<pct>\d+(?:\.\d+)?)\s*%(?:\s*in\s*(?P<bars>\d+)\s*(?:days?|bars?))?"
        ),
        _build_pct_change("lte", -1.0),
        "drops 2% in 3 days",
    ),
    PatternRule(
        "pct_rises",
        re.compile(
            r"(?:rises?|gains?)\s*(?P<pct>\d+(?:\.\d+)?)\s*%(?:\s*in\s*(?P<bars>\d+)\s*(?:days?|bars?))?"
        ),
        _build_pct_change("gte", 1.0),
        "gains 2% in 3 days",
    ),
]

assert set(_PATTERN_ALIASES.values()) <= set(PATTERNS), (
    "a candlestick alias points at a pattern id the registry doesn't have"
)

for _phrase, _pattern_id in _PATTERN_ALIASES.items():
    CONDITION_PATTERNS.append(
        PatternRule(
            f"pattern_{_pattern_id}",
            re.compile(rf"\b{re.escape(_phrase)}\b"),
            _build_pattern(_pattern_id),
            _phrase,
        )
    )

# "harami"/"marubozu" alone have no bullish/bearish variant in the registry --
# pick one from the trade direction and say so, rather than silently guessing.
_DIRECTIONAL_PATTERNS = {
    "harami": _build_pattern_needs_direction(
        "harami", "'harami' with no bullish/bearish stated was read as {variant}."
    ),
    "marubozu": _build_pattern_needs_direction(
        "marubozu", "'marubozu' with no bullish/bearish stated was read as {variant}."
    ),
}


def _try_directional_patterns(clause: str, ind: _Indicators, notes: list[str], direction: str):
    for word, builder in _DIRECTIONAL_PATTERNS.items():
        if re.search(rf"\b{word}\b", clause):
            return builder(re.search(rf"\b{word}\b", clause), ind, notes, direction=direction)
    return None


# Terms that sound like an indicator but are not, and are not close enough to
# anything in the registry to guess at -- so we refuse by name instead.
_UNSUPPORTED_TERMS = ["elliott wave", "gann", "ichimoku cloud count", "fibonacci extension count"]

_ALTERNATIVES_HINT = ", ".join(
    f"{REGISTRY[n].label} ({n})" for n in ("rsi", "macd", "adx", "bollinger", "supertrend")
)


def _find_unsupported_indicator(clause: str) -> str | None:
    for term in _UNSUPPORTED_TERMS:
        if term in clause:
            return term
    return None


def _parse_atomic_clause(
    clause: str, direction: str
) -> tuple[Condition | None, list[dict], list[str], str | None, str]:
    """Parse one clause with no top-level 'and'/'or' in it.

    Returns (condition, indicator_dicts, notes, refusal_message, residual_text).
    `refusal_message` set means: stop everything, this clause named something we
    will not guess at. `residual_text` is whatever part of the clause the match
    did NOT cover -- e.g. "please buy when rsi cracks 30 too" leaves "please" and
    "too" as residual even though the middle matched, so they still surface as
    unparsed instead of silently vanishing along with the part we understood.
    """
    clause = clause.strip()
    if not clause:
        return None, [], [], None, ""

    unsupported = _find_unsupported_indicator(clause)
    if unsupported:
        return (
            None,
            [],
            [],
            (
                f"'{unsupported}' is not an indicator this platform supports. "
                f"Close alternatives already available: {_ALTERNATIVES_HINT}."
            ),
            "",
        )

    ind = _Indicators()
    notes: list[str] = []
    for rule in CONDITION_PATTERNS:
        m = rule.regex.search(clause)
        if m:
            cond = rule.builder(m, ind, notes)
            residual = clause[: m.start()] + " " + clause[m.end() :]
            return cond, ind.as_list(), notes, None, residual

    at_rsi = _try_at_rsi_level(clause, ind, notes, direction)
    if at_rsi is not None:
        cond, residual = at_rsi
        return cond, ind.as_list(), notes, None, residual

    directional = _try_directional_patterns(clause, ind, notes, direction)
    if directional is not None:
        for word in _DIRECTIONAL_PATTERNS:
            m = re.search(rf"\b{word}\b", clause)
            if m:
                residual = clause[: m.start()] + " " + clause[m.end() :]
                return directional, ind.as_list(), notes, None, residual

    return None, [], [], None, clause


def _split_top_level(text: str, seps: list[str]) -> list[str]:
    pattern = re.compile(r"\s+(?:" + "|".join(re.escape(s) for s in seps) + r")\s+")
    return [p for p in pattern.split(text) if p.strip()]


def _parse_condition_text(
    text: str, direction: str
) -> tuple[Condition | None, list[dict], list[str], list[str], str | None]:
    """Parse a chunk of text that may join several clauses with and/or.

    Only one level of nesting is supported (an "or" of "and"-groups). Anything
    more elaborate than that is rare enough in practice that surfacing it as
    unparsed and asking the user to simplify beats guessing at precedence.
    """
    all_indicators: list[dict] = []
    all_notes: list[str] = []
    leftover: list[str] = []

    or_groups = _split_top_level(text, ["or"])
    built_groups: list[Condition] = []
    for group in or_groups:
        # "when" joins two clauses exactly like "and" does once the leading
        # "when" that introduces the whole condition has already been consumed
        # as filler -- "Nifty 100 stocks above the 200 DMA when RSI drops
        # below 40" is two independent conditions glued by "when", not one.
        and_clauses = _split_top_level(group, ["and", "but only if", "but", "when"])
        built_clauses: list[Condition] = []
        for clause in and_clauses:
            cond, inds, notes, refusal, residual = _parse_atomic_clause(clause, direction)
            if refusal:
                return None, [], [], [], refusal
            if cond is None:
                leftover.append(clause.strip())
                continue
            built_clauses.append(cond)
            all_indicators.extend(inds)
            all_notes.extend(notes)
            if residual.strip():
                leftover.append(residual.strip())
        if len(built_clauses) == 1:
            built_groups.append(built_clauses[0])
        elif len(built_clauses) > 1:
            built_groups.append(All(conditions=built_clauses))

    if not built_groups:
        return None, all_indicators, all_notes, leftover, None
    if len(built_groups) == 1:
        return built_groups[0], all_indicators, all_notes, leftover, None
    return Any_(conditions=built_groups), all_indicators, all_notes, leftover, None


# ------------------------------------------------------------------------ exits

_EXIT_TARGET_OR_STOP = re.compile(
    r"exit at \+?(?P<target>\d+(?:\.\d+)?)\s*%\s*or\s*-(?P<stop>\d+(?:\.\d+)?)\s*%"
)
_TARGET_PATTERNS = [
    re.compile(r"(?:take profit|target|book profit)(?:\s*at)?\s*\+?(?P<v>\d+(?:\.\d+)?)\s*%"),
    re.compile(r"exit at \+?(?P<v>\d+(?:\.\d+)?)\s*%"),
    re.compile(r"sell at\s*(?P<v>\d+(?:\.\d+)?)\s*%\s*gain"),
    re.compile(r"(?:^|\s)\+(?P<v>\d+(?:\.\d+)?)\s*%"),
    re.compile(r"(?:up|gain of)\s*(?P<v>\d+(?:\.\d+)?)\s*%"),
    # value-first, "profit" rather than "gain": "sell at 2% profit", "Sell at
    # 3% profit" -- the exact wording fixture #58/#59/#33 use. Kept as its own
    # pattern rather than widening "sell at N% gain" above, so "gain" and
    # "profit" stay two independently readable rows instead of one regex that
    # tries to cover both and is harder to see the vocabulary of at a glance.
    re.compile(r"sell at\s*(?P<v>\d+(?:\.\d+)?)\s*%\s*profit"),
    re.compile(r"book\s*(?P<v>\d+(?:\.\d+)?)\s*%\s*profit"),
    re.compile(r"price\s*reaches\s*(?P<v>\d+(?:\.\d+)?)\s*%\s*profit"),
    # value-first order: "6% take profit" rather than "take profit 6%".
    re.compile(r"(?P<v>\d+(?:\.\d+)?)\s*%\s*(?:take profit|target|book profit)"),
]
_STOP_PATTERNS = [
    re.compile(r"stop\s*loss(?:\s*of)?\s*(?P<v>\d+(?:\.\d+)?)\s*%"),
    re.compile(r"\bsl\s*(?:of)?\s*(?P<v>\d+(?:\.\d+)?)\s*%"),
    re.compile(r"stop at\s*(?P<v>\d+(?:\.\d+)?)\s*%"),
    re.compile(r"stop\s*loss\s*at\s*(?P<v>\d+(?:\.\d+)?)\s*%"),
    re.compile(r"with a\s*(?P<v>\d+(?:\.\d+)?)\s*%\s*stop"),
    re.compile(r"\bstop\s*(?P<v>\d+(?:\.\d+)?)\s*%"),
    re.compile(r"(?:^|\s)-(?P<v>\d+(?:\.\d+)?)\s*%"),
    # value-first order: "2% stop loss" rather than "stop loss 2%".
    re.compile(r"(?P<v>\d+(?:\.\d+)?)\s*%\s*stop(?:\s*loss)?\b"),
]
_ATR_STOP_PATTERN = re.compile(
    r"stop(?:\s*loss)?\s*(?:at|of)?\s*(?P<mult>\d+(?:\.\d+)?)\s*x?\s*(?:times\s*)?atr"
)
_TRAILING_PATTERN = re.compile(
    r"trailing\s*stop\s*(?:of)?\s*(?P<v>\d+(?:\.\d+)?)\s*%|trail(?:ing)?\s*by\s*(?P<v2>\d+(?:\.\d+)?)\s*%"
)
_MAX_BARS_PATTERN = re.compile(
    r"(?:hold|max)\s*(?:for\s*)?(?:a\s*)?(?:max\s*(?:of\s*)?)?(?P<v>\d+)\s*days?"
)
_SQUARE_OFF_PATTERN = re.compile(r"square\s*off\s*at\s*(?P<h>\d{1,2}):(?P<m>\d{2})")
_EOD_PATTERN = re.compile(r"end of (?:the )?day|intraday only|eod")


def _consume(text: str, pattern: re.Pattern[str]) -> tuple[str, re.Match | None]:
    m = pattern.search(text)
    if not m:
        return text, None
    text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
    return text, m


def _extract_percent_exits(text: str) -> tuple[str, dict]:
    out: dict = {}
    m = _EXIT_TARGET_OR_STOP.search(text)
    if m:
        out["target_pct"] = float(m.group("target"))
        out["stop_pct"] = float(m.group("stop"))
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
        return text, out

    for pat in _STOP_PATTERNS:
        text, m = _consume(text, pat)
        if m and "stop_pct" not in out:
            out["stop_pct"] = float(m.group("v"))
            break
    for pat in _TARGET_PATTERNS:
        text, m = _consume(text, pat)
        if m and "target_pct" not in out:
            out["target_pct"] = float(m.group("v"))
            break
    return text, out


def _extract_exit_rest(text: str, ind: _Indicators) -> tuple[str, dict, list[str]]:
    out: dict = {}
    notes: list[str] = []

    text, m = _consume(text, _ATR_STOP_PATTERN)
    if m:
        out["stop_atr_mult"] = float(m.group("mult"))
        atr_id, n = _len_suffix("atr", None, 14)
        ind.declare(atr_id, "atr", {"length": n})
        out["atr_id"] = atr_id

    text, m = _consume(text, _TRAILING_PATTERN)
    if m:
        out["trailing_stop_pct"] = float(m.group("v") or m.group("v2"))

    text, m = _consume(text, _MAX_BARS_PATTERN)
    if m:
        out["max_bars_held"] = int(m.group("v"))

    text, m = _consume(text, _SQUARE_OFF_PATTERN)
    square_off = None
    if m:
        import datetime as dt

        square_off = dt.time(int(m.group("h")), int(m.group("m")))

    text, m = _consume(text, _EOD_PATTERN)
    intraday = bool(m) or square_off is not None

    return text, {**out, "_square_off": square_off, "_intraday": intraday}, notes


_EXIT_CONDITION_MARKERS = re.compile(r"\b(?:exit|sell|cover|square off)\s*when\b")

# ------------------------------------------------------------------- anaphora
#
# "Buy X when RSI drops below 30, sell when it crosses 70" -- "it" names
# whichever indicator the entry clause just talked about. Each entry below is
# (family label, pattern matching that family with an optional inline length),
# restricted to the indicators that ever appear as a bare "<name> <verb> <level>"
# condition -- the only shape "it <verb> <level>" could possibly be standing in
# for. Two distinct families mentioned in the entry make "it" genuinely
# ambiguous, so we ask instead of picking the first one we saw.
_ANTECEDENT_FAMILIES: list[tuple[str, str, re.Pattern[str]]] = [
    ("RSI", "rsi", re.compile(r"\brsi(?P<len>\d{1,3})?\b")),
    ("ADX", "adx", re.compile(r"\badx(?P<len>\d{1,3})?\b")),
    ("stochastic", "stochastic", re.compile(r"\bstochastic(?P<len>\d{1,3})?\b")),
    ("CCI", "cci", re.compile(r"\bcci(?P<len>\d{1,3})?\b")),
    ("Williams %R", "williamsr", re.compile(r"\bwilliams\s*%?r(?P<len>\d{1,3})?\b")),
]


def _find_antecedent(
    entry_text: str,
) -> tuple[str | None, str | None, list[tuple[str, str]]]:
    """Returns (substitution token, ambiguity description).

    Exactly one of the two is set: a substitution token (e.g. "rsi14", to
    splice in for a bare "it") when the entry names exactly one recognised
    family, or a human-readable list of the families found when it names more
    than one -- the caller turns that into a refusal rather than a guess.
    """
    tokens: list[str] = []
    labels_seen: list[str] = []
    for label, token_prefix, pattern in _ANTECEDENT_FAMILIES:
        m = pattern.search(entry_text)
        if m:
            labels_seen.append(label)
            length = m.group("len")
            tokens.append(f"{token_prefix}{length}" if length else token_prefix)
    candidates = list(zip(labels_seen, tokens, strict=True))
    if len(labels_seen) > 1:
        return None, " or the ".join(labels_seen), candidates
    if len(labels_seen) == 1:
        return tokens[0], None, candidates
    return None, None, candidates


def _extract_exit_condition(
    text: str, direction: str, chosen: str | None = None
) -> tuple[str, Condition | None, list[dict], list[str], Question | None]:
    """The second (or later) 'when ...' clause in the text is an exit condition.

    Returns (text, condition, indicators, notes, ambiguous_antecedent). The last
    value set means: a bare "it" in the exit clause could refer to more than one
    indicator named in the entry -- stop and ask, the same discipline an
    unsupported indicator gets elsewhere in this file.
    """
    whens = list(re.finditer(r"\bwhen\b", text))
    if len(whens) < 2:
        return text, None, [], [], None

    start = whens[1].end()
    rest = text[start:]
    end_marker = re.search(r",|$", rest)
    end = end_marker.start() if end_marker else len(rest)
    clause_text = rest[:end]

    if re.search(r"\bit\b", clause_text):
        entry_text = text[: whens[1].start()]
        token, ambiguous, candidates = _find_antecedent(entry_text)
        if ambiguous and chosen in {t for _label, t in candidates}:
            token, ambiguous = chosen, None
        if ambiguous:
            return (
                text,
                None,
                [],
                [],
                Question(
                    text=(
                        f"'it' in \"...{clause_text.strip()}\" could mean either the "
                        f"{ambiguous} mentioned in the entry. Which one?"
                    ),
                    why=(
                        "Guessing which indicator 'it' refers to could exit the trade on a "
                        "condition you never actually described."
                    ),
                    suggestion=None,
                    field="exit.condition",
                    choices=[Choice(label, t) for label, t in candidates],
                ),
            )
        if token:
            clause_text = re.sub(r"\bit\b", token, clause_text)

    cond, inds, notes, _leftover, refusal = _parse_condition_text(clause_text, direction)
    if refusal or cond is None:
        return text, None, [], [], None

    # Also blank a leading "exit"/"sell"/"cover"/"square off" marker word right
    # before this "when", so it doesn't show up as unparsed leftover once the
    # clause after it has been understood.
    blank_start = whens[1].start()
    marker = re.search(r"(?:exit|sell|cover|square off)\s*$", text[:blank_start])
    if marker:
        blank_start = marker.start()
    blank_end = start + end
    text = text[:blank_start] + " " * (blank_end - blank_start) + text[blank_end:]
    return text, cond, inds, notes, None


# --------------------------------------------------------------------- timeframe
#
# The engine only knows how to run bars of a handful of fixed sizes (see
# `Instrument.timeframe`). A user who types "2 minute candles" or "4 hour
# chart" has asked for something we cannot run -- rounding that silently to
# the nearest supported size would run a different strategy than the one they
# described, so we refuse by name instead, the same policy as an unsupported
# indicator gets elsewhere in this file.

_MINUTE_TIMEFRAME = re.compile(
    r"(?P<n>\d{1,3})\s*-?\s*min(?:ute)?s?\s*(?:candles?|chart|bars?|timeframe)"
)
_HOUR_TIMEFRAME = re.compile(r"(?P<n>\d{1,2})\s*-?\s*hours?\s*(?:candles?|chart|bars?|timeframe)")
_HOURLY_TIMEFRAME = re.compile(r"\bhourly\b(?:\s*(?:candles?|charts?|bars?|timeframe))?")
_DAY_TIMEFRAME = re.compile(r"(?P<n>\d{1,2})\s*-?\s*days?\s*timeframe")
_DAILY_TIMEFRAME = re.compile(r"\bdaily\b(?:\s*(?:candles?|charts?|bars?|timeframe))?")
_INTRADAY_TIMEFRAME = re.compile(r"\bintraday\b")

_SUPPORTED_MINUTES = {"1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m", "60": "1h"}


def _extract_timeframe(text: str) -> tuple[str, str | None, str | None]:
    """Returns (text, timeframe literal or None, refusal message or None).

    `timeframe` is `None` when nothing was said (caller keeps the model's
    "1d" default) and also for a bare "intraday" -- see below. A refusal
    means: stop, do not guess a nearby size.
    """
    m = _MINUTE_TIMEFRAME.search(text)
    if m:
        n = m.group("n")
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
        if n not in _SUPPORTED_MINUTES:
            supported = ", ".join(f"{v}" for v in sorted(_SUPPORTED_MINUTES, key=int))
            return (
                text,
                None,
                (
                    f"{n}-minute candles are not a timeframe this platform runs. "
                    f"Supported minute candles: {supported}."
                ),
            )
        return text, _SUPPORTED_MINUTES[n], None

    m = _HOUR_TIMEFRAME.search(text)
    if m:
        n = m.group("n")
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
        if n != "1":
            return (
                text,
                None,
                (
                    f"{n}-hour candles are not a timeframe this platform runs. "
                    "Only 1-hour candles are available above minute bars."
                ),
            )
        return text, "1h", None

    m = _HOURLY_TIMEFRAME.search(text)
    if m:
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
        return text, "1h", None

    m = _DAY_TIMEFRAME.search(text)
    if m:
        n = m.group("n")
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
        if n != "1":
            return (
                text,
                None,
                (
                    f"{n}-day candles are not a timeframe this platform runs; "
                    "only 1-day (daily) bars are."
                ),
            )
        return text, "1d", None

    m = _DAILY_TIMEFRAME.search(text)
    if m:
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
        return text, "1d", None

    m = _INTRADAY_TIMEFRAME.search(text)
    if m:
        # "Intraday" says the strategy squares off same-day, not which bar
        # size to use -- 5-minute and 15-minute intraday strategies are both
        # completely ordinary. Picking one would be exactly the invented
        # guess this module refuses to make elsewhere, so we only consume the
        # word (it must not become a bogus "I didn't understand" question)
        # and leave the timeframe unset; a specific size stated elsewhere in
        # the same sentence already won above, since those patterns run first.
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
        return text, None, None

    return text, None, None


# ---------------------------------------------------------------- backtest window
#
# "Backtest this for the last 3 months", "backtest for 2024", "full year 2024" --
# a date range to run the backtest over, not anything about what the strategy
# itself does. `StrategySpec` has no field for a date range (that is a property
# of a single run, not of the strategy), and inventing one here would be out of
# scope for a parser whose whole job is staying inside the spec it is given.
# Previously this just sat in `unparsed` and blocked the parse outright; now it
# is recognised, stripped so it stops blocking, and named in a note so nothing
# the user typed is silently dropped on the floor.

_PERIOD_UNIT = r"(?P<unit>days?|weeks?|months?|years?)"
_BACKTEST_WINDOW_PATTERNS = [
    re.compile(
        r"backtest\s*(?:it\s*|this\s*)?(?:for\s*|on\s*|over\s*)?(?:the\s*)?(?:last|past)\s*"
        r"(?P<n>\d+)\s*" + _PERIOD_UNIT + r"\b"
    ),
    re.compile(r"backtest\s*(?:it\s*|this\s*)?(?:for|on|in)\s*(?P<year>\d{4})\b"),
    re.compile(r"\bfor\s*(?:the\s*)?last\s*(?P<n>\d+)\s*" + _PERIOD_UNIT + r"\b"),
    re.compile(r"\bfull\s*year\s*(?P<year>\d{4})\b"),
    # "For last year" could mean the last calendar year or the last twelve
    # months; it is understood but sets no dates.
    re.compile(r"\bfor\s*last\s*year\b"),
]


def _extract_backtest_window(text: str) -> tuple[str, str | None, BacktestPeriod | None]:
    """Returns (text, note or None, the period or None).

    Exactly one clear period ("the last 2 years", "2024") comes back as a
    `BacktestPeriod` for the run screen to turn into dates. Anything else that
    was recognised is only named in a note, so it is not silently dropped.
    """
    found: list[str] = []
    periods: list[BacktestPeriod] = []
    for pat in _BACKTEST_WINDOW_PATTERNS:
        m = pat.search(text)
        if m:
            words = text[m.start() : m.end()].strip()
            found.append(words)
            groups = m.groupdict()
            if groups.get("n"):
                unit = groups["unit"] if groups["unit"].endswith("s") else groups["unit"] + "s"
                periods.append(BacktestPeriod(words=words, last=int(groups["n"]), unit=unit))
            elif groups.get("year"):
                periods.append(BacktestPeriod(words=words, year=int(groups["year"])))
            text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
    if not found:
        return text, None, None
    if len(found) == 1 and len(periods) == 1 and periods[0].last != 0:
        return text, None, periods[0]
    return (
        text,
        (
            "The description mentioned a backtest date range (" + "; ".join(found) + ") -- "
            "it was not clear enough to set the dates from, so set the test period "
            "yourself when you run it."
        ),
        None,
    )


# ---------------------------------------------------------------- holding overnight
#
# On intraday bars a strategy squares off before the close unless told
# otherwise. "Hold overnight" is the user telling us otherwise. It is only a
# fact about the schedule: on daily bars every position is held overnight
# anyway, and the words are simply understood.

_OVERNIGHT_PATTERN = re.compile(
    r"\b(?:hold|holding|carry|keep)\s*(?:it\s*|them\s*|the\s*position\s*|positions?\s*|"
    r"trades?\s*)?(?:overnight|forward)\b"
)


def _extract_overnight(text: str) -> tuple[str, bool]:
    text, m = _consume(text, _OVERNIGHT_PATTERN)
    return text, m is not None


def _overnight_conflict_question() -> Question:
    return Question(
        text=(
            "You asked to hold overnight and also to square off the same day. "
            "Which one should it do?"
        ),
        why=(
            "Holding overnight and squaring off before the close are opposite rules; "
            "picking one for you could test a strategy you never described."
        ),
        suggestion=None,
        field="schedule.intraday",
        choices=OVERNIGHT_CHOICES,
    )


def overnight_answer(answers: dict[str, str]) -> str | None:
    """The user's pick on the overnight-or-square-off question, if they made one."""
    value = answers.get("schedule.intraday", "").strip()
    return value if value in {c.value for c in OVERNIGHT_CHOICES} else None


# ----------------------------------------------------------------------- sizing

_LOTS_PATTERN = re.compile(r"(?P<v>\d+)\s*lots?\b")
_RISK_PCT_PATTERN = re.compile(r"risk\s*(?P<v>\d+(?:\.\d+)?)\s*%\s*per\s*trade")
_VALUE_PATTERN = re.compile(r"with\s*(?:rs\.?|₹|rupees)?\s*(?P<v>\d{4,9})\b")


# Position size when the user did not say. "1 lot" is the right default for an
# options strategy, where a lot is a real contract term. It is meaningless for a
# stock, where one lot is one share -- so "Buy RELIANCE when RSI cracks 30" on a
# Rs 1,00,000 account bought Rs 640 of stock and reported a return of -0.03%,
# which is not a test of the strategy so much as a test of nothing at all.
#
# Risking a fixed slice of capital is the standard answer and works at any share
# price. It needs a stop distance to divide by, which is always available here
# because a strategy without a stop loss is refused before it reaches this point.
DEFAULT_RISK_PCT_PER_TRADE = 1.0


def _default_sizing(instrument_kwargs: dict) -> Sizing:
    if instrument_kwargs.get("trade_as") == "option":
        return Sizing(mode="fixed_lots", lots=1)
    return Sizing(mode="risk_based", risk_pct=DEFAULT_RISK_PCT_PER_TRADE)


def _extract_sizing(text: str, instrument_kwargs: dict | None = None) -> tuple[str, Sizing]:
    text, m = _consume(text, _RISK_PCT_PATTERN)
    if m:
        return text, Sizing(mode="risk_based", risk_pct=float(m.group("v")))

    text, m = _consume(text, _LOTS_PATTERN)
    if m:
        return text, Sizing(mode="fixed_lots", lots=int(m.group("v")))

    text, m = _consume(text, _VALUE_PATTERN)
    if m:
        return text, Sizing(mode="fixed_value", value=float(m.group("v")))

    return text, _default_sizing(instrument_kwargs or {})


def _sizing_said(text: str) -> bool:
    """Whether `_extract_sizing` would read a size the user stated."""
    return any(p.search(text) for p in (_RISK_PCT_PATTERN, _LOTS_PATTERN, _VALUE_PATTERN))


# --------------------------------------------------------------------- unparsed


def _leftover_fragments(text: str) -> list[str]:
    fragments = []
    for chunk in re.split(r"[,.;]", text):
        words = [_clean(w) for w in chunk.split()]
        words = [w for w in words if w and w.lower() not in _FILLER]
        if words:
            fragments.append(" ".join(words))
    return fragments


# ------------------------------------------------------------------------- main


def _indicator_dicts_to_specs(dicts: list[dict]):
    from nlt.spec.models import IndicatorSpec

    return [IndicatorSpec(id=d["id"], type=d["type"], params=d["params"]) for d in dicts]


def _merge_indicators(*groups: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for group in groups:
        for d in group:
            if d["id"] in merged and merged[d["id"]] != {"type": d["type"], "params": d["params"]}:
                # Same id, different indicator -- extremely unlikely given our
                # deterministic ids, but if it happens, keep the first and let
                # spec validation catch anything that actually conflicts.
                continue
            merged[d["id"]] = {"type": d["type"], "params": d["params"]}
    return [{"id": k, **v} for k, v in merged.items()]


# ------------------------------------------------------------------------- vwap
#
# VWAP resets every trading session. On an intraday bar that means something --
# "above VWAP" tracks where price sits relative to the session's volume-weighted
# average so far. On a daily bar, a whole session IS one bar, so VWAP collapses
# to that single bar's own typical price: "close above VWAP" would silently
# become "close above roughly its own high/low/close average", which is not a
# question anyone asking for VWAP meant to ask. Rather than run that, we refuse
# by name -- the same policy an unsupported indicator or an unsupported
# timeframe already gets elsewhere in this file.
#
# Baskets (`Instrument.symbol` universes, and single stocks -- see
# `nlt/engine/basket.py`) are daily-only in this platform today regardless of
# what the user asked for, so a VWAP basket strategy cannot be backtested at
# all yet; that is a stronger and permanent refusal, not just "pick a smaller
# bar size".


def _is_universe_symbol(symbol: str) -> bool:
    from nlt.data.universe import is_universe

    return is_universe(symbol)


_TIMEFRAME_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}


def _intraday_only_refusal(indicators: list[dict], instrument_kwargs: dict) -> Question | None:
    """The opening range and a daily average on an intraday chart, where they cannot work.

    On daily candles a whole session is one candle: there is no "first 15
    minutes" inside it, and a daily average is just an ordinary average. And
    the opening range must be a whole number of candles -- the first hourly
    candle runs past 15 minutes, so its high is not the 15-minute high.
    """
    timeframe = instrument_kwargs.get("timeframe", "1d")
    why = "Running it anyway would test something other than what was described."
    for d in indicators:
        if d["type"] == "opening_range":
            minutes = int((d.get("params") or {}).get("minutes", 15))
            if timeframe == "1d":
                text = (
                    f"The first {minutes} minutes only exist on intraday candles; on daily "
                    "candles a whole session is a single candle. Say e.g. '5-minute candles'."
                )
            elif minutes % _TIMEFRAME_MINUTES[timeframe]:
                text = (
                    f"The first {minutes} minutes do not line up with {timeframe} candles: the "
                    f"first candle already runs past them. Use candles of {minutes} minutes or less "
                    f"that divide it evenly, e.g. 5-minute candles."
                )
            else:
                continue
            return Question(text=text, why=why, suggestion=None, field="instrument.timeframe")
        if d["type"] == "daily_average" and timeframe == "1d":
            return Question(
                text=(
                    "A daily average on a daily chart is just an ordinary moving average. Say "
                    "e.g. 'close above the 50-day SMA' instead."
                ),
                why=why,
                suggestion=None,
                field="instrument.timeframe",
            )
    return None


def _vwap_refusal(indicators: list[dict], instrument_kwargs: dict) -> Question | None:
    intraday_only = _intraday_only_refusal(indicators, instrument_kwargs)
    if intraday_only is not None:
        return intraday_only
    if not any(d["type"] == "vwap" for d in indicators):
        return None

    symbol = instrument_kwargs.get("symbol", "NIFTY")
    trade_as = instrument_kwargs.get("trade_as", "index")
    timeframe = instrument_kwargs.get("timeframe", "1d")
    is_basket = trade_as == "stock" and _is_universe_symbol(symbol)

    vwap_explainer = (
        "VWAP only means something on intraday candles -- on a daily bar, a whole "
        "trading session is a single bar, so VWAP collapses to that bar's own "
        "typical price, not what 'above VWAP' usually means."
    )

    if is_basket:
        return Question(
            text=(
                f"{vwap_explainer} On top of that, {symbol} resolves to a basket of "
                "stocks, and every stock in this platform only has daily price data "
                "available today -- so a VWAP strategy on this basket cannot be "
                "backtested yet, on any timeframe."
            ),
            why=(
                "Running this on daily bars would silently answer a different "
                "question than the one asked, and there is no intraday stock data "
                "yet that would let it run the way it was actually described."
            ),
            suggestion="Try VWAP on a single stock or NIFTY/BANKNIFTY on an intraday timeframe instead.",
            field="instrument.timeframe",
        )

    if timeframe == "1d":
        return Question(
            text=(
                f"{vwap_explainer} Pick an intraday timeframe (1m, 3m, 5m, 15m or 30m) "
                "for this condition to do anything meaningful."
            ),
            why="Running this on daily bars would silently answer a question you didn't ask.",
            suggestion="Say e.g. 'use 5-minute candles'.",
            field="instrument.timeframe",
        )

    return None


# A basket strategy holding one position at a time is not what anyone means by
# "buy Nifty 100 stocks". With the single-position default, that strategy
# generated 2,300 signals and took 242 of them -- the other 2,058 were dropped
# for want of a slot, and the backtest silently described a completely different
# strategy from the one described. Baskets therefore get a default that lets the
# basket actually be a basket, and the choice is stated in the readback.
BASKET_DEFAULT_CONCURRENT = 5

# Ceiling on how much of the account a single position may occupy, whatever the
# sizing arithmetic produces. Without it, risk-based sizing with a tight stop
# puts nearly the whole account into one trade: the rupee risk is still small,
# but there is no room for a second position and an overnight gap can jump
# straight past the stop that made the size look safe.
STOCK_MAX_POSITION_PCT = 20.0
FNO_MAX_POSITION_PCT = 10.0


# MCX trades until 23:30 (23:55 in winter); NSE's 15:00 / 15:15 defaults would
# square off a commodity position before its busiest hours. Fifteen minutes
# before the earlier close, as 15:15 is before NSE's.
_MCX_NO_ENTRY_AFTER = dt.time(23, 0)
_MCX_SQUARE_OFF = dt.time(23, 15)


def _schedule_kwargs(
    instrument_kwargs: dict, square_off, intraday, *, overnight: bool = False
) -> dict:
    """Schedule fields for the spec: what the user said, plus the market's defaults."""
    from nlt.data.futures import is_commodity

    out: dict = {}
    if square_off:
        out["square_off"] = square_off
    if intraday:
        out["intraday"] = True
    if overnight:
        out["intraday"] = False
    if instrument_kwargs.get("trade_as") == "future" and is_commodity(
        instrument_kwargs.get("symbol", "")
    ):
        out.setdefault("square_off", _MCX_SQUARE_OFF)
        cutoff = (
            dt.datetime.combine(dt.date.today(), out["square_off"]) - dt.timedelta(minutes=15)
        ).time()
        out["no_entry_after"] = min(_MCX_NO_ENTRY_AFTER, cutoff)
    return out


def _default_risk(instrument_kwargs: dict) -> RiskLimits:
    """Risk limits, with a basket-appropriate concurrency default."""
    from nlt.data.universe import is_universe

    symbol = instrument_kwargs.get("symbol", "")
    # A lot cap belongs to instruments that trade in lots. Applying one to a
    # stock caps the position at that many SHARES, which is never what anyone
    # means -- the account balance and the per-trade loss limit already bound a
    # stock position, and unlike a lot count they scale with the share price.
    lot_cap = 2 if instrument_kwargs.get("trade_as") == "option" else None

    # How much of the account one position may occupy. Options get the tighter
    # limit: an option can lose its whole premium in a session. Futures get the
    # share limit, Justin's call (2026-10-07), measured against the lot's
    # exchange margin rather than its value (his call, 2026-10-12: margin is
    # set by the exchange and is not borrowing).
    position_pct = (
        FNO_MAX_POSITION_PCT
        if instrument_kwargs.get("trade_as") == "option"
        else STOCK_MAX_POSITION_PCT
    )

    if symbol and is_universe(symbol):
        return RiskLimits(
            max_concurrent_positions=BASKET_DEFAULT_CONCURRENT,
            max_lots=lot_cap,
            max_position_pct=position_pct,
        )
    return RiskLimits(max_lots=lot_cap, max_position_pct=position_pct)


def parse(description: str, *, answers: dict[str, str] | None = None) -> TranslationResult:
    answers = answers or {}
    original = description.strip()
    notes: list[str] = []
    questions: list[Question] = []

    if not original:
        return TranslationResult(
            spec=None,
            questions=[
                Question(
                    text="What would you like the strategy to do?",
                    why="An empty description has nothing for us to trade on.",
                    suggestion=None,
                    field="description",
                )
            ],
            source="rules",
        )

    # Typos: everyday words fixed and said; a misspelt stock name only asked.
    # The spec keeps `original`, the user's words as typed.
    from nlt.translate import spelling

    read, spelling_notes, spelling_question = spelling.prepare(original, answers)
    if spelling_question is not None:
        return TranslationResult(spec=None, questions=[spelling_question], source="rules")
    notes.extend(spelling_notes)
    text = read.lower()
    # "5 percent" -> "5%" and "close_price" -> "close" up front so every pattern
    # below only has to know the "%"/"close" spelling, not every synonym a user
    # might type for it.
    text = re.sub(r"(\d+(?:\.\d+)?)\s*percent\b", r"\1%", text)
    text = re.sub(r"\bclose_price\b", "close", text)

    text, instrument_kwargs, instrument_question = _extract_instrument(text, answers)
    symbol_said = bool(instrument_kwargs and "symbol" in instrument_kwargs)
    if instrument_question is not None:
        return TranslationResult(
            spec=None, questions=[instrument_question], notes=notes, source="rules"
        )
    text, direction, direction_explicit = _extract_direction(text)
    if not direction_explicit and re.search(r"death cross", text):
        direction = "short"

    # Exits before the "when" split, so exit-condition clauses like "exit when
    # RSI tops 70" don't get mistaken for a second entry condition. Trailing/ATR
    # stops are pulled out before the plain percent patterns, because a bare
    # "stop N%" pattern would otherwise eat the "stop 1%" inside "trailing stop
    # 1%" and leave "trailing" stranded as unparsed.
    exit_ind = _Indicators()
    text, rest_exits, exit_notes = _extract_exit_rest(text, exit_ind)
    notes.extend(exit_notes)

    # Timeframe after the EOD/intraday-only exit markers (so "intraday only"
    # is consumed there first) but before anything else, so a stray "day" or
    # "hour" doesn't get mistaken for something else downstream.
    text, timeframe, timeframe_refusal = _extract_timeframe(text)
    if timeframe_refusal:
        return TranslationResult(
            spec=None,
            questions=[
                Question(
                    text=timeframe_refusal,
                    why=(
                        "Silently rounding to the nearest bar size we do support would run a "
                        "different strategy than the one described."
                    ),
                    suggestion=None,
                    field="instrument.timeframe",
                )
            ],
            notes=notes,
            source="rules",
        )
    if timeframe:
        instrument_kwargs["timeframe"] = timeframe
    elif instrument_kwargs.get("trade_as") == "option":
        # An option's premium behaves very differently depending on how often
        # the signal is checked, unlike an index or a stock where "no candle
        # size stated" can default to daily without changing what the strategy
        # means. So, for options only, this is asked rather than defaulted.
        picked = answers.get("instrument.timeframe", "").strip()
        valid_timeframes = {"1m", "3m", "5m", "15m", "30m", "1h", "1d"}
        if picked in valid_timeframes:
            timeframe = picked
            instrument_kwargs["timeframe"] = picked
        else:
            from nlt.translate import options_words

            return TranslationResult(
                spec=None,
                questions=[options_words._candle_size_question()],
                notes=notes,
                source="rules",
            )

    text, backtest_note, period = _extract_backtest_window(text)
    if backtest_note:
        notes.append(backtest_note)
    text, overnight = _extract_overnight(text)
    if overnight and rest_exits.get("_intraday"):
        picked = overnight_answer(answers)
        if picked is None:
            return TranslationResult(
                spec=None, questions=[_overnight_conflict_question()], notes=notes, source="rules"
            )
        if picked == "square_off":
            overnight = False  # "overnight" needs nothing: holding overnight wins below

    text, percent_exits = _extract_percent_exits(text)
    premium_said = False
    if instrument_kwargs.get("trade_as") == "option":
        from nlt.translate import options_words

        text, premium_said = options_words.consume_premium_words(text)
    text, exit_condition, exit_cond_inds, exit_cond_notes, exit_ambiguous = _extract_exit_condition(
        text, direction, answers.get("exit.condition")
    )
    if exit_ambiguous:
        return TranslationResult(spec=None, questions=[exit_ambiguous], notes=notes, source="rules")
    notes.extend(exit_cond_notes)

    sizing_said = _sizing_said(text)
    text, sizing = _extract_sizing(text, instrument_kwargs)

    entry_condition, entry_inds, entry_notes, entry_leftover, entry_refusal = _parse_condition_text(
        text, direction
    )
    notes.extend(entry_notes)

    if entry_refusal:
        questions.append(
            Question(
                text=f"I don't recognise part of that: {entry_refusal}",
                why="Guessing at an unsupported indicator could trade on rules you never asked for.",
                suggestion=None,
                field="entry.condition",
            )
        )
        return TranslationResult(spec=None, questions=questions, notes=notes, source="rules")

    unparsed: list[str] = []
    for frag in entry_leftover:
        unparsed.extend(_leftover_fragments(frag))

    if entry_condition is None:
        questions.append(
            Question(
                text="What should trigger this strategy to enter a trade?",
                why="Without an entry rule there is nothing for the strategy to act on.",
                suggestion=None,
                field="entry.condition",
            )
        )

    all_indicators = _merge_indicators(entry_inds, exit_ind.as_list(), exit_cond_inds)

    vwap_question = _vwap_refusal(all_indicators, instrument_kwargs)
    if vwap_question is not None:
        return TranslationResult(spec=None, questions=[vwap_question], notes=notes, source="rules")

    stop_pct = percent_exits.get("stop_pct") or _float_answer(answers, "exit.stop_pct")
    target_pct = (
        percent_exits.get("target_pct")
        or _float_answer(answers, "exit.target_pct")
        or _float_answer(answers, "exit.rule")
    )
    stop_atr_mult = rest_exits.get("stop_atr_mult")
    trailing = rest_exits.get("trailing_stop_pct")
    max_bars_held = rest_exits.get("max_bars_held")

    has_any_stop = bool(stop_pct or stop_atr_mult or trailing)
    if not has_any_stop:
        stop_pct = _float_answer(answers, "exit.stop_pct")
        if not stop_pct:
            questions.append(
                Question(
                    text="What stop loss should this strategy use?",
                    why=(
                        "A strategy with no stop loss can lose far more than intended before "
                        "anything closes the position -- we never assume one for you."
                    ),
                    suggestion="A common starting point is 1%.",
                    field="exit.stop_pct",
                    choices=STOP_CHOICES,
                    answerable=True,
                )
            )

    has_any_exit = bool(
        target_pct or stop_pct or trailing or stop_atr_mult or exit_condition or max_bars_held
    )
    if not has_any_exit:
        questions.append(
            Question(
                text="How should this strategy get out of a trade?",
                why=(
                    "Without an exit, a position would be held forever. A stop loss alone is "
                    "enough; a profit target here is optional."
                ),
                suggestion="A target and a stop loss, e.g. +2% / -1%.",
                field="exit.rule",
                choices=TARGET_CHOICES,
                answerable=True,
            )
        )

    if unparsed:
        questions.append(
            Question(
                text=f'I didn\'t understand this part: "{"; ".join(unparsed)}". Did you mean something specific?',
                why="Silently ignoring words you typed could mean the strategy misses a rule you intended.",
                suggestion=None,
                field="unparsed",
            )
        )

    if questions or entry_condition is None:
        return TranslationResult(
            spec=None, questions=questions, notes=notes, unparsed=unparsed, source="rules"
        )

    exit_kwargs: dict = {}
    if target_pct:
        exit_kwargs["target_pct"] = target_pct
    if stop_pct:
        exit_kwargs["stop_pct"] = stop_pct
    if trailing:
        exit_kwargs["trailing_stop_pct"] = trailing
    if stop_atr_mult:
        exit_kwargs["stop_atr_mult"] = stop_atr_mult
        exit_kwargs["atr_id"] = rest_exits["atr_id"]
    if exit_condition is not None:
        exit_kwargs["condition"] = exit_condition
    if max_bars_held:
        exit_kwargs["max_bars_held"] = max_bars_held

    schedule_kwargs = _schedule_kwargs(
        instrument_kwargs,
        rest_exits.get("_square_off"),
        rest_exits.get("_intraday"),
        overnight=overnight,
    )

    name = _name_from(original)

    try:
        instrument = Instrument(**instrument_kwargs)
        spec = StrategySpec(
            name=name,
            description=original,
            instrument=instrument,
            indicators=_indicator_dicts_to_specs(all_indicators),
            direction=direction,
            entry=entry_condition,
            exit=ExitRules(**exit_kwargs),
            sizing=sizing,
            risk=_default_risk(instrument_kwargs),
            schedule=Schedule(**schedule_kwargs),
        )
    except ValidationError as exc:
        questions.append(
            Question(
                text=f"That description doesn't quite make a valid strategy yet: {exc.errors()[0]['msg']}",
                why="The platform double-checks every strategy before it can trade; this one failed that check.",
                suggestion=None,
                field="spec",
            )
        )
        return TranslationResult(
            spec=None, questions=questions, notes=notes, unparsed=unparsed, source="rules"
        )

    spec = spec.model_copy(update={"name": strategy_name(spec)})
    sources = recipe_sources(
        spec,
        symbol_said=symbol_said,
        timeframe_said=bool(timeframe),
        sizing_said=sizing_said,
        schedule_said=bool(
            rest_exits.get("_square_off") or rest_exits.get("_intraday") or overnight
        ),
        premium_said=premium_said,
    )
    return TranslationResult(
        spec=spec,
        questions=[],
        notes=notes,
        unparsed=unparsed,
        source="rules",
        period=period,
        sources=sources,
    )


def recipe_sources(
    spec: StrategySpec,
    *,
    symbol_said: bool,
    timeframe_said: bool,
    sizing_said: bool,
    schedule_said: bool,
    premium_said: bool = False,
) -> dict[str, str]:
    """Where each readback part came from, for both readers.

    Entry and exit rules are always the user's: neither reader invents one
    (a stop given as an answer to our question is still theirs). Everything
    else is theirs only when the reader saw them say it.
    """
    from nlt.data.futures import is_commodity
    from nlt.translate.readback import DEFAULT, SAID, WORKED_OUT

    out = {"entry": SAID}
    e = spec.exit
    for part, value in [
        ("exit.target", e.target_pct),
        ("exit.stop", e.stop_pct),
        ("exit.trailing_stop", e.trailing_stop_pct),
        ("exit.atr_stop", e.stop_atr_mult),
        ("exit.condition", e.condition),
        ("exit.max_bars_held", e.max_bars_held),
    ]:
        if value is not None:
            out[part] = SAID
    out["instrument"] = SAID if symbol_said else DEFAULT
    out["timeframe"] = SAID if timeframe_said else DEFAULT
    out["sizing"] = SAID if sizing_said else DEFAULT
    if schedule_said:
        out["schedule"] = SAID
    elif spec.instrument.is_future and is_commodity(spec.instrument.symbol):
        out["schedule"] = WORKED_OUT  # from the market's own closing time
    else:
        out["schedule"] = DEFAULT

    if spec.instrument.is_option and not premium_said:
        # "Stop 30%" on an option is read as 30% of the option's own price --
        # an assumption, not something the sentence said, unless it actually
        # named the premium. Claiming "You said" for a reading we supplied
        # would be the readback lying about its own evidence.
        for part in ("exit.stop", "exit.target"):
            if part in out:
                out[part] = WORKED_OUT
    return out


def _float_answer(answers: dict[str, str], field_name: str) -> float | None:
    """A typed answer as a number: "1", "1.5" or "1%" -- and nothing looser.

    The stop-loss question's own hint says "A common starting point is 1%", so
    "1%" is the answer people type. It used to fail `float()` and the question
    came straight back with no explanation. Anything else that is not plainly
    a number still returns None, so the question is asked again rather than
    an answer being guessed at.
    """
    if field_name in answers:
        raw = answers[field_name].strip().removesuffix("%").strip()
        try:
            return float(raw)
        except ValueError:
            return None
    return None


def _name_from(description: str) -> str:
    """A placeholder until the spec exists; `strategy_name` replaces it."""
    words = re.findall(r"[A-Za-z]+", description)
    name = " ".join(w.capitalize() for w in words[:6]) or "Untitled Strategy"
    return name[:80]


_SHORT_LABELS = {"sma": "SMA", "ema": "EMA", "wma": "WMA", "hma": "HMA"}


def strategy_name(spec: StrategySpec) -> str:
    """A name made from what the strategy is: "Buy INFY · Bollinger Bands".

    Built from the spec, not the first words typed -- those gave names like
    "On Infy Hourly Bars Buy When".
    """
    from nlt.indicators.registry import get as get_indicator

    used = _refs_in(spec.entry)
    families: list[str] = []
    for ind in spec.indicators:
        if ind.id in used:
            label = _SHORT_LABELS.get(ind.type) or get_indicator(ind.type).label
            if label not in families:
                families.append(label)
    verb = "Buy" if spec.direction == "long" else "Short"
    what = " + ".join(families) if families else "price rule"
    return f"{verb} {spec.instrument.symbol} · {what}"[:80]


def _refs_in(condition) -> set[str]:
    """Every indicator id a condition reads."""
    from nlt.spec.models import All, Any_, Compare, IsTrue, Not, PercentChange, Ref

    if isinstance(condition, Compare):
        return {o.name for o in (condition.left, condition.right) if isinstance(o, Ref)}
    if isinstance(condition, (IsTrue, PercentChange)):
        return {condition.ref.name}
    if isinstance(condition, (All, Any_)):
        return set().union(*(_refs_in(c) for c in condition.conditions))
    if isinstance(condition, Not):
        return _refs_in(condition.condition)
    return set()
