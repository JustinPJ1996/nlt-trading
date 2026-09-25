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

import functools
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import ValidationError

from nlt.indicators.patterns import PATTERNS
from nlt.indicators.registry import REGISTRY
from nlt.spec.models import (
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
class Question:
    """Something we need the user to answer before we can trade their money on it."""

    text: str
    why: str
    suggestion: str | None
    field: str


@dataclass
class TranslationResult:
    spec: StrategySpec | None
    questions: list[Question] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unparsed: list[str] = field(default_factory=list)
    source: str = "rules"


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
    "please", "i", "want", "to", "a", "an", "the", "on", "in", "at", "with",
    "and", "or", "when", "if", "only", "but", "of", "for", "my", "strategy",
    "then", "so", "it", "is", "s", "this", "that", "would", "like", "d",
    "me", "build", "make", "create", "trade", "trading",
    "while", "as", "long", "still", "open", "just", "now", "up", "not",
    "candle", "candles", "pattern", "forms", "instantly", "use", "using",
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


def _extract_instrument(
    text: str,
) -> tuple[str, dict | None, Question | None]:
    """Returns (text, instrument kwargs, ambiguity question).

    `kwargs` is `None` exactly when `question` is set -- an ambiguous "NIFTY
    50" with no qualifier, the one case this function refuses to guess at.
    """
    kwargs: dict = {}

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

    option_word = re.search(r"\b(call|put|ce|pe|option|options)\b", text)
    if option_word:
        kwargs["trade_as"] = "option"
        if re.search(r"\b(call|ce)\b", text):
            kwargs["option_type"] = "CE"
        elif re.search(r"\b(put|pe)\b", text):
            kwargs["option_type"] = "PE"
        text = re.sub(r"\b(call|put|ce|pe|options?)\b", " ", text)

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
_LEAD_FILLER = (
    r"(?:please\s+)?(?:instantly\s+)?(?:i(?:'d| would)? (?:want to|like to)\s+)?"
)
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
    return Compare(op="crosses_above", left=Ref(name="close"), right=_extremes_ref(ind, length, "highest"))


def _build_breakout_low(m: re.Match, ind: _Indicators, notes: list[str]) -> Condition:
    length = int(m.group("len"))
    return Compare(op="crosses_below", left=Ref(name="close"), right=_extremes_ref(ind, length, "lowest"))


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

    def build(m: re.Match, ind: _Indicators, notes: list[str], direction: str = "long") -> Condition:
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
    """"at RSI 30" names a level with no up/down verb -- see the module docstring
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


CONDITION_PATTERNS: list[PatternRule] = [
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
        re.compile(rf"adx{_LEN}\s*(?:{_STATE_UP_WORDS}|{_CROSS_UP_WORDS})\s*(?P<level>\d+(?:\.\d+)?)"),
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
        re.compile(rf"stochastic{_LEN}\s*(?:{_STATE_DOWN_WORDS}|{_CROSS_DOWN_WORDS})\s*(?P<level>\d+(?:\.\d+)?)"),
        _stoch_builder("lt"),
        "stochastic below 20",
    ),
    PatternRule(
        "stoch_above",
        re.compile(rf"stochastic{_LEN}\s*(?:{_STATE_UP_WORDS}|{_CROSS_UP_WORDS})\s*(?P<level>\d+(?:\.\d+)?)"),
        _stoch_builder("gt"),
        "stochastic above 80",
    ),
    PatternRule(
        "cci_below",
        re.compile(rf"cci{_LEN}\s*(?:{_STATE_DOWN_WORDS}|{_CROSS_DOWN_WORDS})\s*(?P<level>-?\d+(?:\.\d+)?)"),
        _cci_builder("lt"),
        "CCI below -100",
    ),
    PatternRule(
        "cci_above",
        re.compile(rf"cci{_LEN}\s*(?:{_STATE_UP_WORDS}|{_CROSS_UP_WORDS})\s*(?P<level>-?\d+(?:\.\d+)?)"),
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
            _OPT_PRICE + r"(?:closes?\s*)?(?:below|under)\s*(?:the\s*)?(?:previous|prior|yesterday'?s?)"
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
        re.compile(rf"(?P<left>{_OPERAND_TOKEN})\s*(?:{_CROSS_DOWN_WORDS})\s*(?P<right>{_OPERAND_TOKEN})"),
        _build_generic_compare("crosses_below"),
        "ema8 crosses below ema21",
    ),
    PatternRule(
        "generic_crosses_above",
        re.compile(rf"(?P<left>{_OPERAND_TOKEN})\s*(?:{_CROSS_UP_WORDS})\s*(?P<right>{_OPERAND_TOKEN})"),
        _build_generic_compare("crosses_above"),
        "ema8 crosses above ema21",
    ),
    PatternRule(
        "generic_lt",
        re.compile(rf"(?P<left>{_OPERAND_TOKEN})\s*(?:{_STATE_DOWN_WORDS})\s*(?P<right>{_OPERAND_TOKEN})"),
        _build_generic_compare("lt"),
        "close below vwap",
    ),
    PatternRule(
        "generic_gt",
        re.compile(rf"(?P<left>{_OPERAND_TOKEN})\s*(?:{_STATE_UP_WORDS})\s*(?P<right>{_OPERAND_TOKEN})"),
        _build_generic_compare("gt"),
        "close above vwap",
    ),
    PatternRule(
        "pct_falls",
        re.compile(r"falls?\s*(?P<pct>\d+(?:\.\d+)?)\s*%(?:\s*in\s*(?P<bars>\d+)\s*(?:days?|bars?))?"),
        _build_pct_change("lte", -1.0),
        "NIFTY falls 1%",
    ),
    PatternRule(
        "pct_drops",
        re.compile(r"drops?\s*(?P<pct>\d+(?:\.\d+)?)\s*%(?:\s*in\s*(?P<bars>\d+)\s*(?:days?|bars?))?"),
        _build_pct_change("lte", -1.0),
        "drops 2% in 3 days",
    ),
    PatternRule(
        "pct_rises",
        re.compile(r"(?:rises?|gains?)\s*(?P<pct>\d+(?:\.\d+)?)\s*%(?:\s*in\s*(?P<bars>\d+)\s*(?:days?|bars?))?"),
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
_TRAILING_PATTERN = re.compile(r"trailing\s*stop\s*(?:of)?\s*(?P<v>\d+(?:\.\d+)?)\s*%|trail(?:ing)?\s*by\s*(?P<v2>\d+(?:\.\d+)?)\s*%")
_MAX_BARS_PATTERN = re.compile(r"(?:hold|max)\s*(?:for\s*)?(?:a\s*)?(?:max\s*(?:of\s*)?)?(?P<v>\d+)\s*days?")
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


def _extract_exit_rest(
    text: str, ind: _Indicators
) -> tuple[str, dict, list[str]]:
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


def _find_antecedent(entry_text: str) -> tuple[str | None, str | None]:
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
    if len(labels_seen) > 1:
        return None, " and ".join(labels_seen)
    if len(labels_seen) == 1:
        return tokens[0], None
    return None, None


def _extract_exit_condition(
    text: str, direction: str
) -> tuple[str, Condition | None, list[dict], list[str], str | None]:
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
        token, ambiguous = _find_antecedent(entry_text)
        if ambiguous:
            return (
                text,
                None,
                [],
                [],
                (
                    f"'it' in \"...{clause_text.strip()}\" could mean either the {ambiguous} "
                    "mentioned in the entry -- say which one, e.g. 'sell when RSI crosses 70'."
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

_MINUTE_TIMEFRAME = re.compile(r"(?P<n>\d{1,3})\s*-?\s*min(?:ute)?s?\s*(?:candles?|chart|bars?|timeframe)")
_HOUR_TIMEFRAME = re.compile(r"(?P<n>\d{1,2})\s*-?\s*hours?\s*(?:candles?|chart|bars?|timeframe)")
_DAY_TIMEFRAME = re.compile(r"(?P<n>\d{1,2})\s*-?\s*days?\s*timeframe")
_DAILY_TIMEFRAME = re.compile(r"\bdaily\b")
_INTRADAY_TIMEFRAME = re.compile(r"\bintraday\b")

_SUPPORTED_MINUTES = {"1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m"}


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
            return text, None, (
                f"{n}-minute candles are not a timeframe this platform runs. "
                f"Supported minute candles: {supported}."
            )
        return text, _SUPPORTED_MINUTES[n], None

    m = _HOUR_TIMEFRAME.search(text)
    if m:
        n = m.group("n")
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
        if n != "1":
            return text, None, (
                f"{n}-hour candles are not a timeframe this platform runs. "
                "Only 1-hour candles are available above minute bars."
            )
        return text, "1h", None

    m = _DAY_TIMEFRAME.search(text)
    if m:
        n = m.group("n")
        text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
        if n != "1":
            return text, None, (
                f"{n}-day candles are not a timeframe this platform runs; "
                "only 1-day (daily) bars are."
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

_BACKTEST_WINDOW_PATTERNS = [
    re.compile(
        r"backtest\s*(?:this\s*)?for\s*(?:the\s*)?last\s*\d+\s*"
        r"(?:days?|weeks?|months?|years?)\b"
    ),
    re.compile(r"backtest\s*(?:for|on)\s*\d{4}\b"),
    re.compile(r"\bfor\s*(?:the\s*)?last\s*\d+\s*(?:days?|weeks?|months?|years?)\b"),
    re.compile(r"\bfull\s*year\s*\d{4}\b"),
    re.compile(r"\bfor\s*last\s*year\b"),
]


def _extract_backtest_window(text: str) -> tuple[str, str | None]:
    found: list[str] = []
    for pat in _BACKTEST_WINDOW_PATTERNS:
        m = pat.search(text)
        if m:
            found.append(text[m.start() : m.end()].strip())
            text = text[: m.start()] + " " * (m.end() - m.start()) + text[m.end() :]
    if not found:
        return text, None
    return text, (
        "The description mentioned a backtest date range (" + "; ".join(found) + ") -- "
        "this platform has no field for that on the strategy itself, so it was left out; "
        "set the backtest window separately when you actually run it."
    )


# ----------------------------------------------------------------------- sizing

_LOTS_PATTERN = re.compile(r"(?P<v>\d+)\s*lots?\b")
_RISK_PCT_PATTERN = re.compile(r"risk\s*(?P<v>\d+(?:\.\d+)?)\s*%\s*per\s*trade")
_VALUE_PATTERN = re.compile(r"with\s*(?:rs\.?|₹|rupees)?\s*(?P<v>\d{4,9})\b")


def _extract_sizing(text: str) -> tuple[str, Sizing]:
    text, m = _consume(text, _RISK_PCT_PATTERN)
    if m:
        return text, Sizing(mode="risk_based", risk_pct=float(m.group("v")))

    text, m = _consume(text, _LOTS_PATTERN)
    if m:
        return text, Sizing(mode="fixed_lots", lots=int(m.group("v")))

    text, m = _consume(text, _VALUE_PATTERN)
    if m:
        return text, Sizing(mode="fixed_value", value=float(m.group("v")))

    return text, Sizing()


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


def _vwap_refusal(indicators: list[dict], instrument_kwargs: dict) -> Question | None:
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

    text = original.lower()
    # "5 percent" -> "5%" and "close_price" -> "close" up front so every pattern
    # below only has to know the "%"/"close" spelling, not every synonym a user
    # might type for it.
    text = re.sub(r"(\d+(?:\.\d+)?)\s*percent\b", r"\1%", text)
    text = re.sub(r"\bclose_price\b", "close", text)

    text, instrument_kwargs, instrument_question = _extract_instrument(text)
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

    text, backtest_note = _extract_backtest_window(text)
    if backtest_note:
        notes.append(backtest_note)

    text, percent_exits = _extract_percent_exits(text)
    text, exit_condition, exit_cond_inds, exit_cond_notes, exit_ambiguous = _extract_exit_condition(
        text, direction
    )
    if exit_ambiguous:
        return TranslationResult(
            spec=None,
            questions=[
                Question(
                    text=exit_ambiguous,
                    why=(
                        "Guessing which indicator 'it' refers to could exit the trade on a "
                        "condition you never actually described."
                    ),
                    suggestion=None,
                    field="exit.condition",
                )
            ],
            notes=notes,
            source="rules",
        )
    notes.extend(exit_cond_notes)

    text, sizing = _extract_sizing(text)

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
    target_pct = percent_exits.get("target_pct") or _float_answer(answers, "exit.target_pct")
    stop_atr_mult = rest_exits.get("stop_atr_mult")
    trailing = rest_exits.get("trailing_stop_pct")
    max_bars_held = rest_exits.get("max_bars_held")

    has_any_stop = bool(stop_pct or stop_atr_mult or trailing)
    if not has_any_stop:
        if "exit.stop_pct" in answers:
            try:
                stop_pct = float(answers["exit.stop_pct"])
            except ValueError:
                stop_pct = None
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
                )
            )

    has_any_exit = bool(
        target_pct or stop_pct or trailing or stop_atr_mult or exit_condition or max_bars_held
    )
    if not has_any_exit:
        questions.append(
            Question(
                text="How should this strategy get out of a trade?",
                why="Without an exit, a position would be held forever.",
                suggestion="A target and a stop loss, e.g. +2% / -1%.",
                field="exit.rule",
            )
        )

    if unparsed:
        questions.append(
            Question(
                text=f"I didn't understand this part: \"{'; '.join(unparsed)}\". Did you mean something specific?",
                why="Silently ignoring words you typed could mean the strategy misses a rule you intended.",
                suggestion=None,
                field="unparsed",
            )
        )

    if questions or entry_condition is None:
        return TranslationResult(spec=None, questions=questions, notes=notes, unparsed=unparsed, source="rules")

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

    schedule_kwargs: dict = {}
    if rest_exits.get("_square_off"):
        schedule_kwargs["square_off"] = rest_exits["_square_off"]
    if rest_exits.get("_intraday"):
        schedule_kwargs["intraday"] = True

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
            risk=RiskLimits(),
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
        return TranslationResult(spec=None, questions=questions, notes=notes, unparsed=unparsed, source="rules")

    return TranslationResult(spec=spec, questions=[], notes=notes, unparsed=unparsed, source="rules")


def _float_answer(answers: dict[str, str], field_name: str) -> float | None:
    if field_name in answers:
        try:
            return float(answers[field_name])
        except ValueError:
            return None
    return None


def _name_from(description: str) -> str:
    words = re.findall(r"[A-Za-z]+", description)
    name = " ".join(w.capitalize() for w in words[:6]) or "Untitled Strategy"
    return name[:80]
