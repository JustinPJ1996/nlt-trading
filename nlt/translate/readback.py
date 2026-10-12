"""Turn a `StrategySpec` back into plain English.

This is the entire safety story for a non-technical user: they cannot audit
JSON, but they can read "Buy NIFTY when RSI(14) crosses below 30" and
say "no, that's not what I meant." So `describe` has exactly one job -- say
what the spec *says*, mechanically and completely, with nothing invented and
nothing left silent. No LLM, no paraphrasing, no field left out: a field this
function forgets to render is a field the user can never catch being wrong.

Deliberately dumb by design. Cleverness here (inferring intent, summarising,
"the strategy roughly does X") would make the readback a second guesser
instead of a faithful mirror -- exactly the failure mode `rules.py` exists to
avoid on the way in.
"""

from __future__ import annotations

import re

from nlt.indicators.registry import get as get_indicator
from nlt.spec.models import (
    All,
    Any_,
    Compare,
    Condition,
    Const,
    IsTrue,
    Not,
    PercentChange,
    Ref,
    StrategySpec,
)

_TIMEFRAME_WORDS = {
    "1m": "1-minute",
    "3m": "3-minute",
    "5m": "5-minute",
    "15m": "15-minute",
    "30m": "30-minute",
    "1h": "hourly",
    "1d": "daily",
}

_MA_TYPES = {"sma", "ema", "wma", "hma"}

_PRICE_WORDS = {
    "close": "the price",
    "open": "the day's open",
    "high": "the high",
    "low": "the low",
    "volume": "volume",
}

_OP_WORDS = {
    "lt": "is below",
    "lte": "is at or below",
    "gt": "is above",
    "gte": "is at or above",
    "eq": "equals",
    "crosses_above": "crosses above",
    "crosses_below": "crosses below",
}


def format_inr(value: float) -> str:
    """Rs 1,00,000 -- Indian digit grouping (last three digits, then pairs)."""
    n = round(value)
    sign = "-" if n < 0 else ""
    digits = str(abs(n))
    if len(digits) <= 3:
        grouped = digits
    else:
        last3, rest = digits[-3:], digits[:-3]
        parts: list[str] = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    return f"{sign}Rs {grouped}"


def _num(v: float) -> str:
    """1.0 -> '1', 1.5 -> '1.5' -- no trailing zeroes to read out loud."""
    return f"{v:g}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _bar_word(timeframe: str) -> str:
    return "day" if timeframe == "1d" else "bar"


def _indicator_desc(ind_id: str, indicators: dict[str, dict], timeframe: str) -> tuple[str, object]:
    """Returns (label, IndicatorDef) for a declared indicator id."""
    spec = indicators[ind_id]
    d = get_indicator(spec.type)
    resolved = spec.resolved_params()
    if spec.type in _MA_TYPES and "length" in resolved:
        unit = _bar_word(timeframe)
        return f"{int(resolved['length'])}-{unit} {d.label}", d
    numeric = {k: v for k, v in resolved.items() if isinstance(v, (int, float))}
    if not numeric:
        return d.label, d
    if list(numeric) == ["length"]:
        return f"{d.label}({int(numeric['length'])})", d
    return f"{d.label}({', '.join(str(v) for v in numeric.values())})", d


_OUTPUT_WORDS = {
    "signal": "signal line",
    "histogram": "histogram",
    "upper": "upper band",
    "lower": "lower band",
    "middle": "middle band",
    "bandwidth": "bandwidth",
    "percent_b": "%B",
    "direction": "direction",
    "adx": "reading",
    "plus_di": "+DI",
    "minus_di": "-DI",
    "k": "%K",
    "d": "%D",
    "highest": "highest high",
    "lowest": "lowest low",
    "high": "high",
    "low": "low",
    "close": "close",
}


def _ref_desc(ref: Ref, indicators: dict[str, dict], timeframe: str) -> str:
    if ref.name in _PRICE_WORDS:
        text = _PRICE_WORDS[ref.name]
    else:
        label, _d = _indicator_desc(ref.name, indicators, timeframe)
        if ref.output is not None:
            out_word = _OUTPUT_WORDS.get(ref.output, ref.output)
            text = f"{label} {out_word}"
        else:
            text = label
    if ref.bars_ago:
        text += f" {_plural(ref.bars_ago, _bar_word(timeframe))} ago"
    return text


def _const_desc(c: Const) -> str:
    return _num(c.value)


def _operand_desc(op, indicators: dict[str, dict], timeframe: str) -> str:
    if isinstance(op, Ref):
        return _ref_desc(op, indicators, timeframe)
    return _const_desc(op)


def _describe_condition(cond: Condition, indicators: dict[str, dict], timeframe: str) -> str:
    if isinstance(cond, Compare):
        left = _operand_desc(cond.left, indicators, timeframe)
        right = _operand_desc(cond.right, indicators, timeframe)
        return f"{left} {_OP_WORDS[cond.op]} {right}"

    if isinstance(cond, IsTrue):
        label, _d = _indicator_desc(cond.ref.name, indicators, timeframe)
        return f"a {label} candle forms"

    if isinstance(cond, PercentChange):
        ref = _ref_desc(cond.ref, indicators, timeframe)
        bars = _plural(cond.lookback, _bar_word(timeframe))
        magnitude = _num(abs(cond.value))
        # "lte -2" includes a fall of exactly 2%; "lt -2" does not. The wording
        # has to say which, or the readback promises a trade the engine skips.
        if cond.op in ("lt", "lte") and cond.value < 0:
            amount = f"{magnitude}% or more" if cond.op == "lte" else f"more than {magnitude}%"
            return f"{ref} falls {amount} over {bars}"
        if cond.op in ("gt", "gte") and cond.value > 0:
            amount = f"{magnitude}% or more" if cond.op == "gte" else f"more than {magnitude}%"
            return f"{ref} rises {amount} over {bars}"
        return f"{ref}'s change over {bars} {_OP_WORDS[cond.op]} {_num(cond.value)}%"

    if isinstance(cond, All):
        parts = [_wrap(c, indicators, timeframe) for c in cond.conditions]
        return " and ".join(parts)

    if isinstance(cond, Any_):
        parts = [_wrap(c, indicators, timeframe) for c in cond.conditions]
        return " or ".join(parts)

    if isinstance(cond, Not):
        return f"NOT ({_describe_condition(cond.condition, indicators, timeframe)})"

    raise TypeError(f"unhandled condition node: {cond!r}")  # pragma: no cover


def _wrap(cond: Condition, indicators: dict[str, dict], timeframe: str) -> str:
    """Parenthesise nested All/Any_/Not so a mixed and/or tree reads unambiguously."""
    text = _describe_condition(cond, indicators, timeframe)
    if isinstance(cond, (All, Any_, Not)):
        return f"({text})"
    return text


def _universe_size(symbol: str) -> int:
    """How many stocks a universe name actually contains, from the same
    bundled snapshot the parser and the spec validator use -- never a live
    NSE fetch, so this never makes the readback depend on network access or
    on-the-day membership drift.
    """
    from nlt.data.universe import _load_snapshot

    return len(_load_snapshot()["universes"][symbol])


def _instrument_desc(spec: StrategySpec) -> str:
    if spec.instrument.trade_as == "index":
        return f"{spec.instrument.symbol} index"

    if spec.instrument.is_future:
        from nlt.data.futures import contract

        c = contract(spec.instrument.symbol)
        return f"{c.title} futures ({c.market})"

    if spec.instrument.trade_as == "stock" and spec.instrument.is_universe:
        # The whole point of this line: a user who typed "nifty 100 stocks"
        # may not have registered that they are about to backtest a hundred
        # independent instruments at once, sharing one pool of capital. Saying
        # so plainly is the only thing standing between them and finding out
        # the hard way.
        n = _universe_size(spec.instrument.symbol)
        return f"All {n} stocks in {spec.instrument.symbol}"

    if spec.instrument.trade_as == "stock":
        return f"{spec.instrument.symbol} (NSE)"

    if spec.instrument.option_type == "auto":
        side = "call" if spec.direction == "long" else "put"
    else:
        side = "call" if spec.instrument.option_type == "CE" else "put"

    offset = spec.instrument.strike_offset
    if offset == 0:
        strike = "at-the-money"
    else:
        direction_word = "out of the money" if offset > 0 else "in the money"
        strike = f"{_plural(abs(offset), 'strike')} {direction_word}"

    expiry_word = {
        "nearest_weekly": "nearest weekly expiry",
        "next_weekly": "next weekly expiry",
        "monthly": "monthly expiry",
    }[spec.instrument.expiry]

    return f"{spec.instrument.symbol}, {strike} {side} option, {expiry_word}"


def _contract_desc(spec: StrategySpec) -> str:
    """Which futures contract, and what one lot physically is.

    Say what one lot physically is. "1 lot of GOLD" means nothing until you
    know it is a kilogram priced per 10 grams -- and the readback is where a
    wrong contract size would be caught by a person.
    """
    from nlt.data.futures import contract

    return f"Nearest-month contract. {contract(spec.instrument.symbol).describe_lot}"


def _timeframe_desc(spec: StrategySpec) -> str:
    """Its own line, so a bar size we assumed can be labelled as assumed."""
    word = _TIMEFRAME_WORDS.get(spec.instrument.timeframe, spec.instrument.timeframe)
    return f"{word[0].upper()}{word[1:]} candles"


def _exit_lines(
    spec: StrategySpec, indicators: dict[str, dict], timeframe: str
) -> list[tuple[str, str]]:
    """(line, the part of the strategy it states) for each exit rule."""
    lines: list[tuple[str, str]] = []
    e = spec.exit
    if e.target_pct:
        lines.append((f"Take profit at +{_num(e.target_pct)}%", "exit.target"))
    if e.stop_pct:
        lines.append((f"Stop loss at -{_num(e.stop_pct)}%", "exit.stop"))
    if e.trailing_stop_pct:
        lines.append((f"Trailing stop of {_num(e.trailing_stop_pct)}%", "exit.trailing_stop"))
    if e.stop_atr_mult:
        atr_label, _d = _indicator_desc(e.atr_id, indicators, timeframe)
        lines.append((f"Stop loss at {_num(e.stop_atr_mult)}x {atr_label}", "exit.atr_stop"))
    if e.condition is not None:
        condition = _describe_condition(e.condition, indicators, timeframe)
        lines.append((f"Exit when {condition}", "exit.condition"))
    if e.max_bars_held:
        bars = _plural(e.max_bars_held, _bar_word(timeframe))
        lines.append((f"Exit after {bars} if still open", "exit.max_bars_held"))
    # Only promise the square-off when it will actually happen. The engine skips
    # every time-of-day rule on daily bars, because a daily bar's timestamp
    # carries no intraday clock -- so saying "square off at 15:15" on a daily
    # strategy describes behaviour that will not occur. The readback's whole
    # value is that it says what the spec really does, so a line that is merely
    # plausible is worse than no line.
    if spec.schedule.intraday and timeframe != "1d":
        square_off = spec.schedule.square_off.strftime("%H:%M")
        lines.append((f"Square off at {square_off} if still open", "schedule"))
    elif timeframe != "1d":
        # Said out loud because on minute or hourly candles squaring off is the
        # usual habit; a position carried overnight can gap straight past its stop.
        lines.append(("Hold overnight: no square-off before the close", "schedule"))
    if spec.instrument.is_future:
        lines.append(("Close at the end of the contract's expiry day if still open", "expiry"))
    return lines


# Options and futures are dealt in lots; stocks are not. "1 lot of TCS" is not a
# thing anyone says, and a reader who owns shares would rightly wonder how many
# shares a lot is. Quantity is the one word that is correct everywhere -- it is
# also what a contract note and a broker's order window both show -- so the
# readback speaks in quantity and mentions the lot only where a lot exists.
def _lot_size_for(spec: StrategySpec) -> int:
    """Units per lot, from the same table the engine uses.

    Duplicating the number here would let the readback quietly describe a
    different position size from the one traded, which is the exact failure the
    readback exists to prevent.
    """
    from nlt.data.instruments import lot_size

    return lot_size(spec.instrument.symbol, spec.instrument.trade_as)


def _sizing_desc(spec: StrategySpec) -> str:
    s = spec.sizing
    if s.mode == "fixed_lots" and spec.instrument.is_future:
        # Futures are ordered in lots, and "quantity 100" of gold would be 100
        # lots of 10 grams to anyone reading it. Say lots, and what they are.
        from nlt.data.futures import contract

        c = contract(spec.instrument.symbol)
        each = " each" if s.lots != 1 else ""
        return f"{_plural(s.lots, 'lot')} per trade ({c.lot_text}{each})"
    if s.mode == "fixed_lots":
        lot_size = _lot_size_for(spec)
        quantity = s.lots * lot_size
        if lot_size > 1:
            # Say both: the quantity is what gets ordered, the lot count is what
            # the user typed and what an F&O trader thinks in.
            return f"Quantity {quantity:,} per trade ({_plural(s.lots, 'lot')} of {lot_size})"
        return f"Quantity {quantity:,} per trade"
    if s.mode == "fixed_value":
        return f"About {format_inr(s.value)} per trade"
    return f"Risk {_num(s.risk_pct)}% of capital per trade"


def _risk_lines(spec: StrategySpec) -> list[str]:
    r = spec.risk
    # `max_daily_loss` and `max_loss_per_trade` are deliberately not shown: no
    # engine applies them yet, and a readback promising a limit nothing enforces
    # is the readback lying. Justin's call (2026-10-12): they come back as real
    # limits with live trading (the hard daily limits, D2).
    lines = [f"At most {_plural(r.max_concurrent_positions, 'position')} open at a time"]

    # Only mention a quantity cap where one exists. Stocks have none, because a
    # fixed count cannot bound risk across a market priced from Rs 50 to
    # Rs 20,000 a share -- the rupee limits above do that instead, and stating
    # a cap that is not enforced would be the readback lying again.
    if r.max_position_pct is not None and spec.instrument.is_future:
        # A futures lot ties up its exchange margin, not its value (see
        # `nlt.engine.backtest._margin_rate`); say what the limit measures.
        lines.append(
            f"Never set aside more than {_num(r.max_position_pct)}% of the account "
            "as margin for one position"
        )
    elif r.max_position_pct is not None:
        lines.append(
            f"Never put more than {_num(r.max_position_pct)}% of the account into one position"
        )
    if r.max_lots is not None and spec.instrument.is_future:
        lines.append(f"Never hold more than {_plural(r.max_lots, 'lot')}")
    elif r.max_lots is not None:
        lot_size = _lot_size_for(spec)
        max_quantity = r.max_lots * lot_size
        lines.append(
            f"Never hold a quantity above {max_quantity:,} "
            f"({_plural(r.max_lots, 'lot')} of {lot_size})"
            if lot_size > 1
            else f"Never hold a quantity above {max_quantity:,}"
        )
    return lines


# Where a readback line came from. Shown beside each line so a user can tell
# their own words from what we filled in.
SAID = "said"  # the user's words, or their answer to a question
WORKED_OUT = "worked_out"  # our reasoning from what they said
DEFAULT = "default"  # filled in because they did not say

# Parts no sentence can set: always ours.
_FIXED_SOURCES = {"contract": WORKED_OUT, "expiry": WORKED_OUT, "risk": DEFAULT}

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(text.replace(",", ""))}


def _sections(spec: StrategySpec) -> list[tuple[str, list[tuple[str, str]]]]:
    """(heading, [(line, part)]) -- the one place the readback's lines are made."""
    indicators = {i.id: i for i in spec.indicators}
    timeframe = spec.instrument.timeframe
    entry_verb = "BUYS" if spec.direction == "long" else "SHORTS"
    exit_verb = "SELLS" if spec.direction == "long" else "COVERS"
    entry = _describe_condition(spec.entry, indicators, timeframe)
    trades = [(_instrument_desc(spec), "instrument")]
    if spec.instrument.is_future:
        trades.append((_contract_desc(spec), "contract"))
    trades.append((_timeframe_desc(spec), "timeframe"))
    return [
        ("WHAT IT TRADES", trades),
        (f"WHEN IT {entry_verb}", [(f"When {entry}", "entry")]),
        (f"WHEN IT {exit_verb}", _exit_lines(spec, indicators, timeframe)),
        ("HOW MUCH", [(_sizing_desc(spec), "sizing")]),
        ("SAFETY LIMITS", [(line, "risk") for line in _risk_lines(spec)]),
    ]


def describe(spec: StrategySpec) -> str:
    """A plain-English readback of `spec`, generated mechanically from its fields.

    Every optional field that is set produces a line. Nothing here interprets
    intent -- it reports what the spec contains, so a wrong or surprising line
    means the spec itself needs to change, not the wording.
    """
    lines = [spec.name]
    for heading, body in _sections(spec):
        lines.append("")
        lines.append(heading)
        lines.extend(f"  {line}" for line, _part in body)
    return "\n".join(lines)


def describe_lines(
    spec: StrategySpec,
    sources: dict[str, str] | None,
    answers: dict[str, str] | None = None,
) -> list[tuple[str, str, str | None]]:
    """(heading, line, label) for every body line of `describe(spec)`, in order.

    The label says where the line came from: `SAID`, `WORKED_OUT` or `DEFAULT`.
    "Said" is a claim about the user's own words, so it is only ever given when
    the reader recorded it; a part with no record is labelled `DEFAULT`, never
    `SAID`. And a "said" line showing any number the user never typed -- the
    14 in "RSI(14)" when they wrote "RSI", the 65 in "1 lot of 65" -- is
    `WORKED_OUT`: we filled that number in. With no record at all
    (`sources is None`, e.g. a strategy saved before labels existed) every
    label is None rather than a guess.
    """
    typed = _numbers(" ".join([spec.description, *(answers or {}).values()]))
    out = []
    for heading, body in _sections(spec):
        for line, part in body:
            if sources is None:
                label = None
            elif part in _FIXED_SOURCES:
                label = _FIXED_SOURCES[part]
            else:
                label = sources.get(part, DEFAULT)
                if label == SAID and not _numbers(line) <= typed:
                    label = WORKED_OUT
            out.append((heading, line, label))
    return out
