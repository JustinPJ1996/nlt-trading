"""Opening-range breakouts: "once a day" or "every time it crosses".

Tradl #36, "buy when a candle closes above the first-15-minute high", was read
as "whenever the price is above" -- so after a stop it bought again at once.
Since 2026-10-16 the reader asks which is meant (Justin's call), the answer
becomes `Schedule.max_entries_per_day`, and every engine obeys it.

Also here: the breakout on the very first candle after the range. The range
does not exist on the candle before, so a plain "previous close <= previous
level" test could never fire there -- and that candle is the one most
breakouts happen on.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from nlt.engine.backtest import run_backtest
from nlt.spec.models import Compare, Ref, Schedule
from nlt.translate import rules
from nlt.translate.readback import describe

IST = "Asia/Kolkata"
DAY1, DAY2 = dt.date(2026, 3, 2), dt.date(2026, 3, 3)
SENTENCE = (
    "buy reliance on 5-minute candles when the close breaks above the high of the first "
    "15 minutes, target 1.2%, stop 0.5%"
)


def _day(day: dt.date, closes: dict[int, tuple[float, float, float, float]]) -> list[dict]:
    """75 five-minute candles from 09:15, flat at 1000 (high 1005 / low 995 for
    the first three, the opening range), with `closes[k] = (o, h, l, c)`."""
    rows = []
    start = pd.Timestamp(dt.datetime.combine(day, dt.time(9, 15)), tz=IST)
    for k in range(75):
        o, h, low, c = closes.get(
            k, (1000.0, 1005.0 if k < 3 else 1001.0, 995.0 if k < 3 else 999.0, 1000.0)
        )
        rows.append(
            {
                "timestamp": start + pd.Timedelta(minutes=5 * k),
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "volume": 1.0,
            }
        )
    return rows


# Candle 3 (09:30) is the first after the range and closes above its 1005 high.
# Candle 5 (09:40) trades down to 1000: the 0.5% stop (1010 * 0.995 = 1004.95)
# is hit. The price sits below the high, then candle 15 (10:30) crosses it again.
BREAKOUT_DAY = {
    3: (1006.0, 1011.0, 1006.0, 1010.0),
    4: (1010.0, 1010.0, 1008.0, 1009.0),
    5: (1008.0, 1008.0, 1000.0, 1001.0),
    15: (1001.0, 1011.0, 1001.0, 1010.0),
}


def _bars(days=(DAY1,)) -> pd.DataFrame:
    rows = [r for d in days for r in _day(d, BREAKOUT_DAY)]
    return pd.DataFrame(rows).set_index("timestamp")


def _spec(pick: str):
    result = rules.parse(SENTENCE, answers={"schedule.daily_limit": pick})
    assert result.spec is not None, result.questions
    return result.spec


def _run(spec, bars):
    return run_backtest(spec, bars, capital=2_500_000, drop_partial_last_bar=False)


# ------------------------------------------------------------------ the reader


def test_a_range_breakout_asks_once_or_every_time() -> None:
    result = rules.parse(SENTENCE)
    assert result.spec is None
    assert [q.field for q in result.questions] == ["schedule.daily_limit"]
    assert [(c.label, c.value) for c in result.questions[0].choices] == [
        ("Once a day", "once"),
        ("Every time it crosses", "every"),
    ]


def test_a_strategy_not_on_the_opening_range_is_not_asked() -> None:
    result = rules.parse("buy reliance on 5-minute candles when rsi crosses below 30, stop 1%")
    assert result.spec is not None
    assert result.spec.schedule.max_entries_per_day is None


@pytest.mark.parametrize(("pick", "limit"), [("once", 1), ("every", None)])
def test_each_answer_sets_the_limit_and_the_readback_says_it(pick, limit) -> None:
    spec = _spec(pick)
    assert spec.schedule.max_entries_per_day == limit
    text = describe(spec)
    if limit == 1:
        assert "At most one trade a day" in text
        assert "each time the price crosses" not in text
    else:
        assert "Can trade again the same day each time the price crosses that level again" in text
        assert "At most" not in text.split("SAFETY LIMITS")[0]


def test_a_state_read_becomes_the_crossing_the_answer_confirms() -> None:
    """The AI path read #36 as "close > range high" (a state). Once the user
    has answered, it is a crossing either way -- with the range kept on the right."""
    entry = Compare(op="lt", left=Ref(name="or15", output="high"), right=Ref(name="close"))
    indicators = [{"id": "or15", "type": "opening_range", "params": {"minutes": 15}}]
    crossed, limit, question, said = rules.range_repeat(
        entry, indicators, {"schedule.daily_limit": "once"}
    )
    assert question is None and said and limit == 1
    assert crossed == Compare(
        op="crosses_above", left=Ref(name="close"), right=Ref(name="or15", output="high")
    )


# ------------------------------------------------------------------ the engine


def test_a_breakout_on_the_first_candle_after_the_range_trades() -> None:
    res = _run(_spec("every"), _bars())
    assert res.trades
    first = res.trades[0]
    assert first.signal_time.time() == dt.time(9, 30)
    assert first.entry_time.time() == dt.time(9, 35)


def test_every_time_trades_again_after_the_stop() -> None:
    res = _run(_spec("every"), _bars())
    assert [t.entry_time.time() for t in res.trades] == [dt.time(9, 35), dt.time(10, 35)]
    assert res.trades[0].exit_reason == "stop"


def test_once_a_day_waits_for_the_next_day() -> None:
    res = _run(_spec("once"), _bars((DAY1, DAY2)))
    assert [(t.entry_time.date(), t.entry_time.time()) for t in res.trades] == [
        (DAY1, dt.time(9, 35)),
        (DAY2, dt.time(9, 35)),
    ]
    skipped = [d for d in res.decisions if d.reason_code == "daily_limit"]
    assert [d.time for d in skipped] == [
        pd.Timestamp(dt.datetime.combine(DAY1, dt.time(10, 30)), tz=IST),
        pd.Timestamp(dt.datetime.combine(DAY2, dt.time(10, 30)), tz=IST),
    ]
    assert any("at most one trade a day" in w for w in res.warnings)
    # Turned away when the signal appears, so the paper feed never shows a
    # "queued for the next candle" it then takes back.
    queued = [d.time.time() for d in res.decisions if d.kind == "signal_taken"]
    assert dt.time(10, 30) not in queued


def test_the_limit_counts_trades_opened_not_signals() -> None:
    """Two allowed a day: both crossings trade, a third would not."""
    spec = _spec("every")
    spec = spec.model_copy(
        update={"schedule": Schedule(**{**spec.schedule.model_dump(), "max_entries_per_day": 2})}
    )
    res = _run(spec, _bars())
    assert len(res.trades) == 2
    assert not [d for d in res.decisions if d.reason_code == "daily_limit"]


# --------------------------------------------------------- the option engines


def test_a_bought_option_opens_once_a_day() -> None:
    from tests import test_options_engine as oe

    # Two signals on D1 (10:00 and 12:00); the first trade is stopped in between.
    second = oe._ts(oe.D1, 12, 0)
    bars = oe._index_bars([oe.D1], oe.SIGNAL)
    bars.loc[second, ["open", "high", "low", "close"]] = 21510.0
    source = oe._chain(
        [oe.D1],
        {oe.ATM_CE: {oe.FILL: (100, 100, 100, 100), oe._ts(oe.D1, 11, 0): (60, 60, 60, 60)}},
    )
    every = oe._run(oe._spec(), bars, source)
    assert len(every.trades) == 2
    spec = oe._spec()
    spec = spec.model_copy(
        update={"schedule": spec.schedule.model_copy(update={"max_entries_per_day": 1})}
    )
    once = oe._run(spec, bars, source)
    assert len(once.trades) == 1
    assert [d.reason_code for d in once.decisions if d.kind == "signal_skipped"] == ["daily_limit"]
    assert any("at most one trade a day" in w for w in once.warnings)


def test_a_multi_leg_position_opens_once_a_day() -> None:
    from nlt.spec.models import Any_, AtTime
    from tests import test_legs_engine as le

    entry = Any_(conditions=[AtTime(time=dt.time(9, 20)), AtTime(time=dt.time(11, 0))])
    every = le._run(
        le._spec(entry=entry, max_concurrent=2),
        le._source(),
        days=(le.D1,),
        index=le._index(days=(le.D1,)),
    )
    assert len(every.trades) == 2
    spec = le._spec(entry=entry, max_concurrent=2)
    spec = spec.model_copy(
        update={"schedule": spec.schedule.model_copy(update={"max_entries_per_day": 1})}
    )
    once = le._run(spec, le._source(), days=(le.D1,), index=le._index(days=(le.D1,)))
    assert len(once.trades) == 1
    assert "daily_limit" in [d.reason_code for d in once.decisions]
    assert any("at most one position a day" in w for w in once.warnings)


def test_the_activity_feed_says_why_a_signal_was_skipped() -> None:
    from app.logic import describe_decision

    line = describe_decision(
        {"kind": "signal_skipped", "reason_code": "daily_limit", "symbol": "RELIANCE"}
    )
    assert "already opened as many trades as it may in one day" in line
