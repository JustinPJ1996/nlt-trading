"""A stop at a price level (2026-10-16): "stop at the low of the opening range".

Tradl #24's opening-range breakout needed it. The level's value on the signal
candle is the stop for the trade's life; a level not on the safe side of the
price skips the signal, never gets another stop.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import pytest

from nlt.engine.backtest import run_backtest
from nlt.spec.models import ExitRules, Ref
from nlt.translate import llm
from nlt.translate.readback import describe
from nlt.translate.rules import parse
from tests.test_once_a_day import BREAKOUT_DAY, DAY1, IST, _day

SENTENCE = (
    "buy reliance on 5-minute candles when the close breaks above the high of the first 15 "
    "minutes, stop at the low of the opening range, target 1.2%"
)


def _spec(sentence=SENTENCE):
    result = parse(sentence, answers={"schedule.daily_limit": "once"})
    assert result.spec is not None, result.questions
    return result.spec


def _bars(day_rows):
    return pd.DataFrame(day_rows).set_index("timestamp")


def _run(spec, bars):
    return run_backtest(spec, bars, capital=2_500_000, drop_partial_last_bar=False)


# ------------------------------------------------------------------ reading


def test_the_opening_range_low_is_the_stop() -> None:
    spec = _spec()
    assert spec.exit.stop_level == Ref(name="orange15", output="low")
    assert spec.exit.stop_pct is None
    assert (
        "Stop loss at the low of the first 15 minutes (its value when the buy signal appears, "
        "kept for the whole trade)"
    ) in describe(spec)


@pytest.mark.parametrize(
    ("words", "kind"),
    [
        ("stop at the previous day's low", ("prior_period", "low")),
        ("stop at the lowest low of the last 10 candles", ("rolling_extremes", "lowest")),
        ("stop loss below the low of the first 15 minutes", ("opening_range", "low")),
    ],
)
def test_each_level_reads(words, kind) -> None:
    spec = _spec(f"buy reliance on 5-minute candles when rsi crosses below 30, {words}, target 1%")
    types = {i.id: i.type for i in spec.indicators}
    assert (types[spec.exit.stop_level.name], spec.exit.stop_level.output) == kind


@pytest.mark.parametrize(
    "sentence",
    [
        # Which opening range? The entry names none.
        "buy reliance on 5-minute candles when rsi crosses below 30, stop at the low of the opening range, target 1%",
        # Days are not candles.
        "buy reliance on 5-minute candles when rsi crosses below 30, stop at the lowest low of the last 10 days, target 1%",
        # On daily candles "the previous day" is two different days.
        "buy reliance when rsi crosses below 30, stop at the previous day's low, target 3%",
        # A short's stop is a high.
        "sell reliance on 5-minute candles when rsi crosses above 70, stop at the previous day's low, target 1%",
        # An option's stop is on its own price.
        "buy the nifty weekly atm call on 5-minute candles when the close breaks above the high of the first 15 minutes, stop at the low of the opening range, target 60% on the premium",
    ],
)
def test_a_level_stop_that_cannot_be_meant_is_asked(sentence) -> None:
    assert parse(sentence, answers={"schedule.daily_limit": "once"}).spec is None


# ------------------------------------------------------------------ the engine


def test_the_stop_fills_at_the_level() -> None:
    """Range low 995 (candles 0-2). Signal 09:30 (close 1010), filled 09:35 at
    1010. Candle 5 trades down to 1000 -- above 995, no stop. Then candle 20
    trades to 990: stopped at 995, not at a percentage."""
    rows = _day(DAY1, {**BREAKOUT_DAY, 20: (1001.0, 1001.0, 990.0, 991.0)})
    res = _run(_spec(), _bars(rows))
    (t,) = res.trades
    assert t.entry_price == 1010.0
    assert t.stop_level == 995.0
    assert (t.exit_reason, t.exit_price) == ("stop", 995.0)
    assert t.exit_time == pd.Timestamp(dt.datetime.combine(DAY1, dt.time(10, 55)), tz=IST)


def test_a_gap_through_the_level_fills_at_the_open() -> None:
    rows = _day(DAY1, {**BREAKOUT_DAY, 20: (980.0, 981.0, 975.0, 978.0)})
    (t,) = _run(_spec(), _bars(rows)).trades
    assert (t.exit_reason, t.exit_price) == ("stop", 980.0)


def test_sizing_risks_the_real_distance_to_the_level() -> None:
    """Risk 1% of Rs 25 lakh = Rs 25,000 over 1010 - 995 = 15 a share:
    1,666 shares -- capped by the 20% position limit (Rs 5 lakh / 1010 = 495)."""
    (t,) = _run(_spec(), _bars(_day(DAY1, BREAKOUT_DAY))).trades
    assert t.quantity == 495


def test_a_level_on_the_wrong_side_skips_the_signal() -> None:
    """The fill opens at 990, already below the 995 stop: no stop could
    protect it, so it is skipped and said -- not opened with another stop."""
    rows = _day(DAY1, {**BREAKOUT_DAY, 4: (990.0, 1009.0, 989.0, 1008.0)})
    res = _run(_spec(), _bars(rows))
    # The 09:30 signal is skipped; the later 10:30 crossing (its level safely
    # below) is a trade of its own -- "once a day" counts trades opened.
    assert [t.entry_time.time() for t in res.trades] == [dt.time(10, 35)]
    (d,) = [d for d in res.decisions if d.reason_code == "stop_level"]
    assert "not below the price" in d.detail["why"]
    assert any("stop level was not on the safe side" in w for w in res.warnings)


def test_the_level_is_read_on_the_signal_candle_never_the_fill_candle() -> None:
    """Signal candle 4 trades down to 980. The lowest low of the 3 candles
    before it is 999 (candles 1-3); the fill candle 5 would already "see"
    980 in its own lookback. Reading it there is reading the future."""
    spec = _spec(
        "buy reliance on 5-minute candles when rsi crosses below 30, stop at the lowest low "
        "of the last 3 candles, target 50%"
    )
    from nlt.engine.backtest import level_stop
    from nlt.engine.features import build_features

    rows = _day(
        DAY1,
        {k: (1000.0, 1001.0, 999.0, 1000.0) for k in range(1, 4)}
        | {4: (1000.0, 1001.0, 980.0, 1000.0)},
    )
    features = build_features(spec, _bars(rows))
    assert level_stop(spec, features, 5) == 999.0


def test_two_stops_say_the_nearer_one_applies() -> None:
    spec = _spec(SENTENCE + ", stop 0.5%")
    assert spec.exit.stop_pct == 0.5 and spec.exit.stop_level is not None
    assert "the one nearer the price applies" in describe(spec)


# -------------------------------------------------------- Tradl #24, AI path


def test_tradl_24_reads_with_the_range_low_as_its_stop() -> None:
    """Its recorded AI reply turned the stop into an exit condition ("low <
    range low", filled next open). The rules' level stop replaces it."""
    replies = json.loads(
        (Path(__file__).parent / "fixtures/llm_template_replies.json").read_text()
    )["replies"]
    case = next(
        c
        for c in json.loads((Path(__file__).parent / "fixtures/template_prompts.json").read_text())
        if c["n"] == 24
    )
    result = llm.translate(
        case["text"], answers={"schedule.daily_limit": "once"}, transport=lambda _s, u: replies[u]
    )
    spec = result.spec
    assert spec is not None, result.questions
    assert spec.exit.condition is None
    assert (spec.exit.stop_level.output, spec.exit.target_pct) == ("low", 1.2)
    assert {i.type for i in spec.indicators} == {"opening_range"}
    assert (spec.instrument.symbol, spec.instrument.timeframe) == ("HDFCBANK", "5m")
    assert spec.schedule.square_off == dt.time(15, 15)


def test_exit_rules_count_a_level_as_a_stop() -> None:
    assert ExitRules(stop_level=Ref(name="x", output="low")).has_stop


def test_a_short_level_on_the_wrong_side_skips_the_signal_too() -> None:
    """Range high 1005. Candle 3 closes 990 (a break below the 995 low); the
    fill candle opens at 1010, already above the 1005 stop: skipped."""
    spec = _spec(
        "sell reliance on 5-minute candles when the close breaks below the low of the first 15 "
        "minutes, stop at the high of the opening range, target 1.2%"
    )
    rows = _day(DAY1, {3: (994.0, 994.0, 989.0, 990.0), 4: (1010.0, 1011.0, 1009.0, 1010.0)})
    res = _run(spec, _bars(rows))
    assert [d.reason_code for d in res.decisions if d.kind == "signal_skipped"] == ["stop_level"]
    assert not [t for t in res.trades if t.entry_time.time() == dt.time(9, 35)]
