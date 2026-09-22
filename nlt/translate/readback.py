"""Turn a `StrategySpec` back into plain English.

This is the entire safety story for a non-technical user: they cannot audit
JSON, but they can read "Buy 1 lot of NIFTY when RSI(14) crosses below 30" and
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
        if cond.op in ("lt", "lte") and cond.value < 0:
            return f"{ref} falls more than {magnitude}% over {bars}"
        if cond.op in ("gt", "gte") and cond.value > 0:
            return f"{ref} rises more than {magnitude}% over {bars}"
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


def _instrument_desc(spec: StrategySpec) -> str:
    tf_word = _TIMEFRAME_WORDS.get(spec.instrument.timeframe, spec.instrument.timeframe)
    if spec.instrument.trade_as == "index":
        return f"{spec.instrument.symbol}, on {tf_word} bars"

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

    return f"{spec.instrument.symbol}, {strike} {side} option, {expiry_word}, on {tf_word} bars"


def _exit_lines(spec: StrategySpec, indicators: dict[str, dict], timeframe: str) -> list[str]:
    lines: list[str] = []
    e = spec.exit
    if e.target_pct:
        lines.append(f"Take profit at +{_num(e.target_pct)}%")
    if e.stop_pct:
        lines.append(f"Stop loss at -{_num(e.stop_pct)}%")
    if e.trailing_stop_pct:
        lines.append(f"Trailing stop of {_num(e.trailing_stop_pct)}%")
    if e.stop_atr_mult:
        atr_label, _d = _indicator_desc(e.atr_id, indicators, timeframe)
        lines.append(f"Stop loss at {_num(e.stop_atr_mult)}x {atr_label}")
    if e.condition is not None:
        lines.append(f"Exit when {_describe_condition(e.condition, indicators, timeframe)}")
    if e.max_bars_held:
        lines.append(f"Exit after {_plural(e.max_bars_held, _bar_word(timeframe))} if still open")
    if spec.schedule.intraday:
        lines.append(f"Square off at {spec.schedule.square_off.strftime('%H:%M')} if still open")
    return lines


def _sizing_desc(spec: StrategySpec) -> str:
    s = spec.sizing
    if s.mode == "fixed_lots":
        return f"{_plural(s.lots, 'lot')} per trade"
    if s.mode == "fixed_value":
        return f"About {format_inr(s.value)} per trade"
    return f"Risk {_num(s.risk_pct)}% of capital per trade"


def _risk_lines(spec: StrategySpec) -> list[str]:
    r = spec.risk
    return [
        f"Stop trading for the day after losing {format_inr(r.max_daily_loss)}",
        f"Never risk more than {format_inr(r.max_loss_per_trade)} on one trade",
        f"At most {_plural(r.max_concurrent_positions, 'position')} open at a time",
        f"Never hold more than {_plural(r.max_lots, 'lot')} at once",
    ]


def describe(spec: StrategySpec) -> str:
    """A plain-English readback of `spec`, generated mechanically from its fields.

    Every optional field that is set produces a line. Nothing here interprets
    intent -- it reports what the spec contains, so a wrong or surprising line
    means the spec itself needs to change, not the wording.
    """
    indicators = {i.id: i for i in spec.indicators}
    timeframe = spec.instrument.timeframe
    entry_verb = "BUYS" if spec.direction == "long" else "SHORTS"
    exit_verb = "SELLS" if spec.direction == "long" else "COVERS"

    lines = [spec.name, ""]
    lines.append("WHAT IT TRADES")
    lines.append(f"  {_instrument_desc(spec)}")
    lines.append("")
    lines.append(f"WHEN IT {entry_verb}")
    lines.append(f"  When {_describe_condition(spec.entry, indicators, timeframe)}")
    lines.append("")
    lines.append(f"WHEN IT {exit_verb}")
    for line in _exit_lines(spec, indicators, timeframe):
        lines.append(f"  {line}")
    lines.append("")
    lines.append("HOW MUCH")
    lines.append(f"  {_sizing_desc(spec)}")
    lines.append("")
    lines.append("SAFETY LIMITS")
    for line in _risk_lines(spec):
        lines.append(f"  {line}")

    return "\n".join(lines)
