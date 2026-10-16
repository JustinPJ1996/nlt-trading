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
    AtTime,
    Compare,
    Condition,
    Const,
    IsTrue,
    IvPercentile,
    Not,
    OnExpiryDay,
    OptionLeg,
    PercentChange,
    Ref,
    StrategySpec,
    is_defined_risk,
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


# Every square-off is priced at the end of its own minute -- 15:15 means the
# price at 15:15:59 (Justin, 2026-10-15) -- whatever the candle size.
def _expiry_price_words(square_off) -> str:
    return f"its price at {square_off}:59"


def _cutoff_lines(spec, timeframe: str) -> list[tuple[str, str]]:
    """The entry cut-off, said out loud -- it was a hidden rule until the
    settings screen (2026-10-15). Daily candles carry no clock, so none."""
    if timeframe == "1d" or not spec.schedule.cutoff_applies:
        return []
    from nlt.spec.clock import market_close, minutes_before_close

    i = spec.instrument
    close = market_close(i.trade_as, i.symbol)
    minutes = minutes_before_close(i.trade_as, i.symbol, spec.schedule.no_entry_after)
    return [
        (
            f"No new entries after {spec.schedule.no_entry_after:%H:%M} "
            f"({minutes} minutes before the {close:%H:%M} close)",
            "schedule.cutoff",
        )
    ]


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
    # Raw output names that would otherwise be printed as code (2026-10-16:
    # "MACD(12, 26, 9) macd crosses above ..." was on screen).
    "macd": "line",
    "supertrend": "line",
    "sar": "level",
    "conversion": "conversion line",
    "base": "base line",
    "span_a": "leading span A",
    "span_b": "leading span B",
    "lagging": "lagging span",
}

# Levels worked out from the previous period: "the previous day's R1".
_PERIOD_WORDS = {"day": "day", "week": "week", "month": "month"}
_PIVOT_WORDS = {
    "pivot": "pivot point",
    "r1": "R1 resistance",
    "r2": "R2 resistance",
    "r3": "R3 resistance",
    "s1": "S1 support",
    "s2": "S2 support",
    "s3": "S3 support",
}
_CPR_WORDS = {
    "pivot": "central pivot",
    "tc": "CPR top",
    "bc": "CPR bottom",
    "width_pct": "CPR width (%)",
}
_FIB_WORDS = {
    "level_236": "23.6%",
    "level_382": "38.2%",
    "level_500": "50%",
    "level_618": "61.8%",
    "level_786": "78.6%",
}


def _special_ref_desc(ref: Ref, indicators: dict[str, dict], timeframe: str) -> str | None:
    """Indicators whose plain meaning is not "label(params) output".

    The daily average says "previous day" out loud because that is what it
    reads: today's daily close is not known until the market shuts.
    """
    spec = indicators.get(ref.name)
    if spec is None:
        return None
    params = spec.resolved_params()
    if spec.type == "rolling_extremes" and ref.output in ("highest", "lowest"):
        # Excludes the current candle (see `levels.rolling_extremes`), hence "last".
        words = "highest high" if ref.output == "highest" else "lowest low"
        return f"the {words} of the last {_plural(int(params['length']), _bar_word(timeframe))}"
    if spec.type == "opening_range" and ref.output in ("high", "low"):
        return f"the {ref.output} of the first {int(params['minutes'])} minutes"
    period = _PERIOD_WORDS.get(str(params.get("period", "day")), "period")
    if spec.type == "prior_period" and ref.output in ("high", "low", "close"):
        return f"the previous {period}'s {ref.output}"
    if spec.type == "pivots" and ref.output in _PIVOT_WORDS:
        return f"the {_PIVOT_WORDS[ref.output]} from the previous {period}"
    if spec.type == "cpr" and ref.output in _CPR_WORDS:
        return f"the {_CPR_WORDS[ref.output]} from the previous {period}"
    if spec.type == "fibonacci":
        bars = _plural(int(params["length"]), _bar_word(timeframe))
        if ref.output in _FIB_WORDS:
            return f"the {_FIB_WORDS[ref.output]} Fibonacci retracement of the last {bars}"
        if ref.output in ("swing_high", "swing_low"):
            words = "highest high" if ref.output == "swing_high" else "lowest low"
            return f"the {words} of the last {bars}"
    if spec.type == "daily_average":
        if ref.output == "close":
            return "the previous day's close"
        if ref.output == "average":
            kind = "exponential average" if params.get("kind") == "ema" else "average"
            return f"the {int(params['length'])}-day {kind} of daily closes, up to the previous day"
    return None


def _ref_desc(ref: Ref, indicators: dict[str, dict], timeframe: str) -> str:
    special = _special_ref_desc(ref, indicators, timeframe)
    if special is not None:
        text = special
    elif ref.name in _PRICE_WORDS:
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


def _same_indicator(left, right) -> bool:
    """Both sides are outputs of one declared indicator, worded plainly
    (`_special_ref_desc` phrases are whole sentences of their own)."""
    return (
        isinstance(left, Ref)
        and isinstance(right, Ref)
        and left.name == right.name
        and left.output is not None
        and right.output is not None
        and left.name not in _PRICE_WORDS
    )


_TREND_WORDS = {
    "crosses_above": "turns to an uptrend",
    "crosses_below": "turns to a downtrend",
    "gt": "is in an uptrend",
    "gte": "is in an uptrend",
    "lt": "is in a downtrend",
    "lte": "is in a downtrend",
}


def _trend_turn_desc(cond: Compare, indicators: dict[str, dict], timeframe: str) -> str | None:
    """Supertrend / Parabolic SAR "direction" is +1 up, -1 down (never 0), so
    "direction crosses above 0" is "turns to an uptrend"."""
    left, right = cond.left, cond.right
    if not (
        isinstance(left, Ref)
        and left.output == "direction"
        and not left.bars_ago
        and isinstance(right, Const)
        and right.value == 0
        and cond.op in _TREND_WORDS
    ):
        return None
    spec = indicators.get(left.name)
    if spec is None or spec.type not in ("supertrend", "parabolic_sar"):
        return None
    label, _d = _indicator_desc(left.name, indicators, timeframe)
    return f"{label} {_TREND_WORDS[cond.op]}"


def _describe_condition(cond: Condition, indicators: dict[str, dict], timeframe: str) -> str:
    if isinstance(cond, Compare):
        trend = _trend_turn_desc(cond, indicators, timeframe)
        if trend is not None:
            return trend
        left = _operand_desc(cond.left, indicators, timeframe)
        right = _operand_desc(cond.right, indicators, timeframe)
        if (
            _same_indicator(cond.left, cond.right)
            and _special_ref_desc(cond.right, indicators, timeframe) is None
        ):
            # "MACD(12, 26, 9) line crosses above its signal line", not the
            # indicator's name and settings said twice.
            out_word = _OUTPUT_WORDS.get(cond.right.output, cond.right.output)
            right = f"its {out_word}"
            if cond.right.bars_ago:
                right += f" {_plural(cond.right.bars_ago, _bar_word(timeframe))} ago"
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

    if isinstance(cond, AtTime):
        return f"the time is {cond.time:%H:%M}"

    if isinstance(cond, OnExpiryDay):
        return "it is expiry day"

    if isinstance(cond, IvPercentile):
        return f"the IV percentile {_OP_WORDS[cond.op]} {_num(cond.value)}"

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
    # On an option, a percentage stop/target is read against the option's own
    # price (the premium), never the index -- a 30% move in NIFTY and a 30%
    # move in a premium are nothing alike, so the line has to say which one
    # this is measuring.
    pct_suffix = " of the option's premium" if spec.instrument.is_option else ""
    if e.target_pct:
        lines.append((f"Take profit at +{_num(e.target_pct)}%{pct_suffix}", "exit.target"))
    if e.stop_pct:
        lines.append((f"Stop loss at -{_num(e.stop_pct)}%{pct_suffix}", "exit.stop"))
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
    lines.extend(_cutoff_lines(spec, timeframe))
    if spec.schedule.intraday and timeframe != "1d":
        square_off = spec.schedule.square_off.strftime("%H:%M")
        lines.append((f"Square off at {square_off} if still open", "schedule"))
    elif timeframe != "1d":
        # Said out loud because on minute or hourly candles squaring off is the
        # usual habit; a position carried overnight can gap straight past its stop.
        lines.append(("Hold overnight: no square-off before the close", "schedule"))
    if spec.instrument.is_future:
        lines.append(("Close at the end of the contract's expiry day if still open", "expiry"))
    if spec.instrument.is_option:
        square_off = spec.schedule.square_off.strftime("%H:%M")
        # Overnight holding is allowed on an option (unlike a future, which is
        # closed at that day's *close*), but never into settlement: the
        # exchange guarantees an exit, at the square-off candle's opening
        # price -- not its close -- so the wording has to say "opening price"
        # for this not to overstate what actually gets paid.
        lines.append(
            (
                f"If still open on expiry day, closed at {square_off} "
                f"({_expiry_price_words(square_off)}, never held to settlement)",
                "option.expiry_close",
            )
        )
        lines.append(
            (
                "If the chosen option didn't trade in the candle we'd buy in, "
                "that signal is skipped -- never a different strike",
                "option.skip_if_no_trade",
            )
        )
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
    if spec.instrument.legs and s.mode == "fixed_lots":
        lot_size = _lot_size_for(spec)
        if all(leg.lots == 1 for leg in spec.instrument.legs):
            return (
                f"{_plural(s.lots, 'lot')} of each option per trade (quantity "
                f"{s.lots * lot_size:,} each, today's lot of {lot_size})"
            )
        return (
            f"{_plural(s.lots, 'set')} of the options above per trade, each option's lots as "
            f"listed (today's lot of {lot_size})"
        )
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
            # the user typed and what an F&O trader thinks in. For an option,
            # also say the lot used is *today's* -- the exchange has changed it
            # before, and every past year in the test is sized by it regardless.
            today_note = ", today's lot" if spec.instrument.is_option else ""
            return f"Quantity {quantity:,} per trade ({_plural(s.lots, 'lot')} of {lot_size}{today_note})"
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
    sells = any(leg.side == "sell" for leg in spec.instrument.legs)
    if r.max_position_pct is not None and (spec.instrument.is_future or sells):
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
_FIXED_SOURCES = {
    "contract": WORKED_OUT,
    "expiry": WORKED_OUT,
    "risk": DEFAULT,
    "option.expiry_close": WORKED_OUT,
    "option.skip_if_no_trade": WORKED_OUT,
    "option.banknifty_weekly_note": WORKED_OUT,
    "option.entry_fill": WORKED_OUT,
    "option.iv_method": WORKED_OUT,
    "option.hedge": WORKED_OUT,
    "option.naked": WORKED_OUT,
}

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(text.replace(",", ""))}


_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_EVERY_WEEKDAY = [0, 1, 2, 3, 4]


def _weekdays_words(weekdays: list[int]) -> str | None:
    """ "every Monday" / "on Mondays and Thursdays", or None for every weekday."""
    if weekdays == _EVERY_WEEKDAY:
        return None
    names = [_WEEKDAY_NAMES[d] for d in weekdays]
    if len(names) == 1:
        return f"every {names[0]}"
    plural = [f"{n}s" for n in names]
    return "on " + ", ".join(plural[:-1]) + f" and {plural[-1]}"


def _entry_line(spec: StrategySpec, indicators: dict, timeframe: str) -> str:
    """ "At 09:20 each day", "At 09:35 on expiry day", or "When <condition>".

    A time of day leads the line because that is how it was said; every other
    part of the rule follows "only when".
    """
    conditions = spec.entry.conditions if isinstance(spec.entry, All) else [spec.entry]
    at = next((c for c in conditions if isinstance(c, AtTime)), None)
    days = _weekdays_words(spec.schedule.weekdays)
    if at is None:
        line = f"When {_describe_condition(spec.entry, indicators, timeframe)}"
        return f"{line}, only {days}" if days else line
    rest = [c for c in conditions if c is not at]
    expiry_day = any(isinstance(c, OnExpiryDay) for c in rest)
    others = [c for c in rest if not isinstance(c, OnExpiryDay)]
    parts = [f"At {at.time:%H:%M}"]
    if expiry_day:
        parts.append("on expiry day")
    if days:
        parts.append(days)
    if not expiry_day and not days:
        parts.append("each day")
    line = " ".join(parts)
    if others:
        line += ", only when " + " and ".join(_wrap(c, indicators, timeframe) for c in others)
    return line


def _conditions_in(condition) -> list:
    if isinstance(condition, (All, Any_)):
        return [x for c in condition.conditions for x in _conditions_in(c)]
    if isinstance(condition, Not):
        return _conditions_in(condition.condition)
    return [condition]


_EXPIRY_ADJ = {
    "nearest_weekly": "nearest-weekly",
    "next_weekly": "next-weekly",
    "monthly": "monthly",
}
_TYPE_WORD = {"CE": "call", "PE": "put"}


def _leg_strike(leg: OptionLeg) -> str:
    if leg.strike_points is not None:
        if leg.strike_points == 0:
            return "at-the-money (the strike nearest the index)"
        where = "above" if leg.strike_points > 0 else "below"
        return f"strike nearest {abs(leg.strike_points)} points {where} the index"
    if leg.strike_offset == 0:
        return "at-the-money"
    way = "out of the money" if leg.strike_offset > 0 else "in the money"
    return f"{_plural(abs(leg.strike_offset), 'strike')} {way}"


def _leg_desc(leg: OptionLeg) -> str:
    """ "Sell the nearest-weekly put, strike nearest 200 points below the index"."""
    side = "Sell" if leg.side == "sell" else "Buy"
    lots = f"{_plural(leg.lots, 'lot')} of " if leg.lots != 1 else ""
    return f"{side} {lots}the {_EXPIRY_ADJ[leg.expiry]} {_TYPE_WORD[leg.option_type]}, {_leg_strike(leg)}"


def _leg_short(leg: OptionLeg) -> str:
    """ "the sold call (at-the-money)" -- enough to tell legs apart in an exit line."""
    side = "sold" if leg.side == "sell" else "bought"
    return f"the {side} {_TYPE_WORD[leg.option_type]} ({_leg_strike(leg)})"


def _times(n: int) -> str:
    return {1: "once", 2: "twice"}.get(n, f"{n} times")


def _leg_move(v: str, side: str, rises: bool) -> str:
    """How far a leg's price must move, said the way the engine measures it.

    The distance is a percentage of the option's price at the signal, counted
    from the price we actually got (the engine's rule for every strategy: the
    levels belong to the setup, see HANDOFF.md). "Rises 25% above its price at
    the signal" would be a different, untrue level whenever the fill differs
    from the signal price.
    """
    where = "where we sold it" if side == "sell" else "where we bought it"
    direction = "rises above" if rises else "falls below"
    return f"its price {direction} {where} by {v}% of its price at the signal"


def _leg_percent_lines(legs: list[OptionLeg], attr: str) -> list[str]:
    """One line for a stop (or target) shared by every sold option, else one per leg."""
    with_value = [leg for leg in legs if getattr(leg, attr) is not None]
    if not with_value:
        return []
    sold = [leg for leg in legs if leg.side == "sell"]
    values = {getattr(leg, attr) for leg in with_value}
    is_stop = attr == "stop_pct"
    title = "Stop loss" if is_stop else "Take profit"
    if with_value == sold and len(values) == 1:
        move = _leg_move(_num(values.pop()), "sell", rises=is_stop)
        if len(sold) == 1:
            return [f"{title}: the sold option is closed if {move}"]
        return [f"{title} on each sold option: closed if {move}"]
    lines = []
    for leg in with_value:
        # A sold option loses when its price rises; a bought one when it falls.
        rises = (leg.side == "sell") == is_stop
        move = _leg_move(_num(getattr(leg, attr)), leg.side, rises=rises)
        lines.append(f"{title} on {_leg_short(leg)}: closed if {move}")
    return lines


def _reentry_lines(legs: list[OptionLeg]) -> list[str]:
    lines = []
    groups: dict[tuple[str, int], list[OptionLeg]] = {}
    for leg in legs:
        if leg.reentry != "none":
            groups.setdefault((leg.reentry, leg.reentry_max), []).append(leg)
    for (kind, count), group in groups.items():
        if kind == "same_strike_at_cost":
            how = "re-sell the same option when its price comes back to where we first sold it"
        elif all(_leg_strike(leg).startswith("at-the-money") for leg in group):
            how = "sell a fresh at-the-money option at the next candle"
        else:
            how = "sell a fresh option at the next candle, its strike picked again the same way"
        if len(group) == len([leg for leg in legs if leg.side == "sell"]):
            who = "a sold option's"
        else:
            who = " or ".join(f"{_leg_short(leg)}'s" for leg in group)
        lines.append(f"After {who} stop loss closes it, {how} -- at most {_times(count)}")
    return lines


def _legs_exit_lines(
    spec: StrategySpec, indicators: dict[str, dict], timeframe: str
) -> list[tuple[str, str]]:
    from nlt.translate.options_words import _credit_shape

    legs = spec.instrument.legs
    e = spec.exit
    lines: list[tuple[str, str]] = []
    credit = _credit_shape(legs)
    lines += [(line, "exit.stop") for line in _leg_percent_lines(legs, "stop_pct")]
    # Said exactly as `nlt.engine.legs_backtest._combined_decision` judges it:
    # against the premium at the SIGNAL prices (not the fills), the stop at
    # every option's worst price of the candle, the target on closes, and
    # either one filled at the next candle's open.
    whole = "the credit at the signal prices" if credit else "the premium paid at the signal prices"
    if e.combined_stop_pct:
        lines.append(
            (
                f"Stop loss when the position's loss reaches {_num(e.combined_stop_pct)}% of "
                f"{whole} (judged with each option at its worst price of the candle; closed at "
                "the next candle's opening prices)",
                "exit.combined_stop",
            )
        )
    lines += [(line, "exit.target") for line in _leg_percent_lines(legs, "target_pct")]
    if e.combined_target_pct:
        lines.append(
            (
                f"Take profit when the position's profit reaches {_num(e.combined_target_pct)}% "
                f"of {whole} (judged on candle closes; closed at the next candle's opening "
                "prices)",
                "exit.combined_target",
            )
        )
    lines += [(line, "legs.reentry") for line in _reentry_lines(legs)]
    if e.condition is not None:
        condition = _describe_condition(e.condition, indicators, timeframe)
        lines.append((f"Exit when {condition}", "exit.condition"))
    if e.max_bars_held:
        bars = _plural(e.max_bars_held, _bar_word(timeframe))
        lines.append((f"Exit after {bars} if still open", "exit.max_bars_held"))

    if any(leg.side == "sell" for leg in legs):
        if is_defined_risk(legs):
            lines.append(
                ("The bought options cap the worst loss, so no stop loss is needed", "option.hedge")
            )
        else:
            lines.append(
                (
                    "Selling an option without a bought one to protect it: a sudden jump can "
                    "blow straight past the stop",
                    "option.naked",
                )
            )

    square_off = spec.schedule.square_off.strftime("%H:%M")
    lines.extend(_cutoff_lines(spec, timeframe))
    if spec.schedule.intraday and timeframe != "1d":
        lines.append((f"Square off at {square_off} if still open", "schedule"))
        lines.append(
            (
                f"If still open on expiry day, closed at {square_off} "
                f"({_expiry_price_words(square_off)}, never held to settlement)",
                "option.expiry_close",
            )
        )
    else:
        held = f"held until expiry day, then closed at {square_off} that day (never left to settle)"
        other_exits = any(part.startswith("exit.") for _line, part in lines)
        lines.append(
            (f"Otherwise {held}" if other_exits else held[0].upper() + held[1:], "schedule")
        )
    lines.append(
        (
            "If any of the chosen options didn't trade in the candle we'd open in, that signal "
            "is skipped -- never a different strike",
            "option.skip_if_no_trade",
        )
    )
    return lines


def _legs_sections(spec: StrategySpec) -> list[tuple[str, list[tuple[str, str]]]]:
    indicators = {i.id: i for i in spec.indicators}
    timeframe = spec.instrument.timeframe
    legs = spec.instrument.legs
    trades = [(f"{spec.instrument.symbol} options", "instrument")]
    trades += [(_leg_desc(leg), "option.legs") for leg in legs]
    if spec.instrument.symbol == "BANKNIFTY" and any(leg.expiry != "monthly" for leg in legs):
        from nlt.translate.options_words import BANKNIFTY_WEEKLY_LAST_EXPIRY

        trades.append(
            (
                "BANKNIFTY weekly options stopped trading after "
                f"{BANKNIFTY_WEEKLY_LAST_EXPIRY:%d %b %Y} -- a weekly expiry won't find a real "
                "contract after that date.",
                "option.banknifty_weekly_note",
            )
        )
    trades.append((_timeframe_desc(spec), "timeframe"))

    entry = [(_entry_line(spec, indicators, timeframe), "entry"), *_repeat_lines(spec, timeframe)]
    parts = _conditions_in(spec.entry)
    at = next((c for c in parts if isinstance(c, AtTime)), None)
    if at is not None:
        t = f"{at.time:%H:%M}"
        entry.append(
            (
                f"Opens at the {t} price; the strikes are picked from the index's close on the "
                f"candle ending {t}",
                "option.entry_fill",
            )
        )
    for iv in (c for c in parts if isinstance(c, IvPercentile)):
        past = (
            "the past year" if iv.lookback_days == 252 else f"the past {iv.lookback_days} sessions"
        )
        word = {"gt": "above", "gte": "at or above", "lt": "below", "lte": "at or below"}[iv.op]
        entry.append(
            (
                f"IV percentile {word} {_num(iv.value)} (implied volatility of the at-the-money "
                f"options at the previous day's close, ranked against {past})",
                "option.iv_method",
            )
        )
    return [
        ("WHAT IT TRADES", trades),
        ("WHEN IT OPENS THE POSITION", entry),
        ("WHEN IT CLOSES THE POSITION", _legs_exit_lines(spec, indicators, timeframe)),
        ("HOW MUCH", [(_sizing_desc(spec), "sizing")]),
        ("SAFETY LIMITS", [(line, "risk") for line in _risk_lines(spec)]),
    ]


def _repeat_lines(spec: StrategySpec, timeframe: str) -> list[tuple[str, str]]:
    """How often it may trade in a day. "Closes above the first-15-minute
    high" can mean once, or every time; the engine does what the spec says, so
    the readback says which (Justin, 2026-10-16)."""
    limit = spec.schedule.max_entries_per_day
    if limit is not None:
        trades = "one trade" if limit == 1 else f"{limit} trades"
        return [
            (
                f"At most {trades} a day: no new trade after that until the next day, "
                "even after a stop loss",
                "schedule.daily_limit",
            )
        ]
    if timeframe == "1d" or not _crosses_a_session_level(spec):
        return []
    return [
        (
            "Can trade again the same day each time the price crosses that level again",
            "schedule.daily_limit",
        )
    ]


def _crosses_a_session_level(spec: StrategySpec) -> bool:
    kinds = {i.id: i.type for i in spec.indicators}
    for c in _conditions_in(spec.entry):
        if isinstance(c, Compare) and c.op in ("crosses_above", "crosses_below"):
            for side in (c.left, c.right):
                if isinstance(side, Ref) and kinds.get(side.name) == "opening_range":
                    return True
    return False


def _sections(spec: StrategySpec) -> list[tuple[str, list[tuple[str, str]]]]:
    """(heading, [(line, part)]) -- the one place the readback's lines are made."""
    if spec.instrument.legs:
        return _legs_sections(spec)
    indicators = {i.id: i for i in spec.indicators}
    timeframe = spec.instrument.timeframe
    entry_verb = "BUYS" if spec.direction == "long" else "SHORTS"
    exit_verb = "SELLS" if spec.direction == "long" else "COVERS"
    trades = [(_instrument_desc(spec), "instrument")]
    if spec.instrument.is_future:
        trades.append((_contract_desc(spec), "contract"))
    if (
        spec.instrument.is_option
        and spec.instrument.symbol == "BANKNIFTY"
        and spec.instrument.expiry != "monthly"
    ):
        from nlt.translate.options_words import BANKNIFTY_WEEKLY_LAST_EXPIRY

        trades.append(
            (
                "BANKNIFTY weekly options stopped trading after "
                f"{BANKNIFTY_WEEKLY_LAST_EXPIRY:%d %b %Y} -- a weekly expiry won't find a real "
                "contract after that date.",
                "option.banknifty_weekly_note",
            )
        )
    trades.append((_timeframe_desc(spec), "timeframe"))
    return [
        ("WHAT IT TRADES", trades),
        (
            f"WHEN IT {entry_verb}",
            [(_entry_line(spec, indicators, timeframe), "entry"), *_repeat_lines(spec, timeframe)],
        ),
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
