"""More conditions (A6): the opening range, a daily average on an intraday chart,
and the reader's wording for Stochastic %K/%D and Donchian-style highs and lows.

The two new indicators each have a look-ahead trap, and those are what most of
this file is about: the 15-minute high is not known until 15 minutes have
passed, and today's daily close is not known until the market shuts.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nlt.engine import warmup
from nlt.engine.backtest import run_backtest
from nlt.indicators import levels
from nlt.translate import rules
from nlt.translate.readback import describe

IST = "Asia/Kolkata"


def _session_bars(days: int = 3, minutes: int = 5, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stamps = []
    for d in pd.bdate_range("2026-03-02", periods=days):
        open_ = pd.Timestamp(d.date()).tz_localize(IST) + pd.Timedelta(hours=9, minutes=15)
        stamps.extend(open_ + pd.Timedelta(minutes=minutes * k) for k in range(375 // minutes))
    idx = pd.DatetimeIndex(stamps)
    close = 1000 + np.cumsum(rng.normal(0, 1, len(idx)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + rng.uniform(0, 2, len(idx)),
            "low": close - rng.uniform(0, 2, len(idx)),
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )


# ----------------------------------------------------------------- opening range


def test_the_opening_range_is_unknown_until_its_minutes_have_passed() -> None:
    bars = _session_bars(minutes=5)
    rng = levels.opening_range(bars.high, bars.low, 15)
    day = bars.loc["2026-03-03"]
    inside = day.index[day.index.time < dt.time(9, 30)]
    assert rng.loc[inside].isna().all().all()
    after = day.index[day.index.time >= dt.time(9, 30)]
    first_three = day.loc[inside]
    assert (rng.loc[after, "high"] == first_three.high.max()).all()
    assert (rng.loc[after, "low"] == first_three.low.min()).all()


def test_each_day_has_its_own_opening_range() -> None:
    bars = _session_bars(minutes=5)
    rng = levels.opening_range(bars.high, bars.low, 15)
    highs = {d: rng.loc[str(d)].high.dropna().unique() for d in ["2026-03-02", "2026-03-03"]}
    assert all(len(v) == 1 for v in highs.values())
    assert highs["2026-03-02"][0] != highs["2026-03-03"][0]


@pytest.mark.parametrize("cut", [2, 3, 40, 80])
def test_opening_range_never_changes_when_later_candles_are_removed(cut) -> None:
    bars = _session_bars(minutes=5)
    full = levels.opening_range(bars.high, bars.low, 15)
    part = levels.opening_range(bars.high.iloc[:cut], bars.low.iloc[:cut], 15)
    pd.testing.assert_frame_equal(full.iloc[:cut], part)


def test_no_trade_is_entered_before_the_range_is_known() -> None:
    bars = _session_bars(days=5, minutes=5)
    spec = rules.parse(
        "buy reliance on 5-minute candles when the close breaks above the high of the first "
        "15 minutes, target 1%, stop 0.5%",
        answers={"schedule.daily_limit": "every"},
    ).spec
    assert spec is not None
    result = run_backtest(spec, bars, capital=2_500_000, drop_partial_last_bar=False)
    assert result.trades
    # The earliest possible signal is the 09:30 candle, filled at the next open.
    assert all(t.entry_time.time() >= dt.time(9, 35) for t in result.trades)


# ------------------------------------------------- daily average, intraday chart


def test_a_candle_sees_only_the_previous_days_close_and_average() -> None:
    bars = _session_bars(days=6, minutes=15)
    daily_closes = bars.close.groupby(bars.index.date).last()  # ty: ignore[unresolved-attribute]  # pandas stub lacks Index.date
    avg = levels.daily_average(bars.close, length=3)
    friday = bars.loc["2026-03-06"].index[0]
    assert avg.loc[friday, "close"] == pytest.approx(daily_closes.iloc[3])  # Thursday's
    assert avg.loc[friday, "average"] == pytest.approx(daily_closes.iloc[1:4].mean())


@pytest.mark.parametrize("cut", [30, 50, 60, 100])
def test_daily_average_never_changes_when_later_candles_are_removed(cut) -> None:
    """Cutting a day in half must not change that day's values: they are yesterday's."""
    bars = _session_bars(days=6, minutes=15)
    full = levels.daily_average(bars.close, length=3)
    part = levels.daily_average(bars.close.iloc[:cut], length=3)
    pd.testing.assert_frame_equal(full.iloc[:cut], part)


def test_warm_up_for_a_daily_average_counts_days_not_candles() -> None:
    spec = rules.parse(
        "buy axisbank on 15-minute candles when the daily close is above the 50-day sma, stop 1%"
    ).spec
    assert spec is not None
    assert warmup.longest_lookback(spec) == 50 * 25  # 50 sessions of 25 candles


# ------------------------------------------------------------ where they cannot run


def test_the_opening_range_is_refused_on_daily_candles() -> None:
    q = rules._intraday_only_refusal(
        [{"type": "opening_range", "params": {"minutes": 15}}], {"timeframe": "1d"}
    )
    assert q is not None and "intraday candles" in q.text


def test_an_opening_range_that_does_not_fit_the_candles_is_refused() -> None:
    result = rules.parse(
        "buy reliance on hourly candles when the close breaks above the high of the first 15 "
        "minutes, stop 1%"
    )
    assert result.spec is None
    assert "do not line up" in result.questions[0].text


def test_30_minutes_on_15_minute_candles_is_fine() -> None:
    assert (
        rules._intraday_only_refusal(
            [{"type": "opening_range", "params": {"minutes": 30}}], {"timeframe": "15m"}
        )
        is None
    )


def test_a_daily_average_on_a_daily_chart_is_refused() -> None:
    q = rules._intraday_only_refusal(
        [{"type": "daily_average", "params": {"length": 50}}], {"timeframe": "1d"}
    )
    assert q is not None and "ordinary moving average" in q.text


# ------------------------------------------------------------------ the wording


@pytest.mark.parametrize(
    ("sentence", "line"),
    [
        (
            "buy bhartiartl when %K crosses above %D, exit when %K goes above 80, stop 0.8%",
            "When Stochastic(14, 1, 3) %K crosses above its %D",
        ),
        (
            "buy maruti when the close is above the highest high of the last 20 days, stop 2%",
            "When the price is above the highest high of the last 20 days",
        ),
        (
            "buy maruti when rsi cracks 30, sell when the close falls below the lowest low of "
            "the last 10 days, stop 2%",
            "Exit when the price crosses below the lowest low of the last 10 days",
        ),
        (
            "buy hdfcbank on 5-minute candles when the close breaks above the high of the first "
            "15 minutes, target 1.2%, stop 0.5%",
            "When the price crosses above the high of the first 15 minutes",
        ),
        (
            "buy axisbank on 15-minute candles when the daily close is above the 50-day sma, "
            "stop 1%",
            "When the previous day's close is above the 50-day average of daily closes, up to "
            "the previous day",
        ),
    ],
)
def test_the_new_wordings_read_and_read_back(sentence, line) -> None:
    result = rules.parse(sentence, answers={"schedule.daily_limit": "every"})
    assert result.spec is not None, result.questions
    assert f"  {line}" in describe(result.spec).splitlines()
