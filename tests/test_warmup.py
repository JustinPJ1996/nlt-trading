"""Indicator warm-up: history before the start date feeds indicators, never trades.

What must hold:
- an indicator is ready on the start date when history allows;
- nothing is bought before the start date, and the reported numbers cover only
  the chosen period;
- when history runs out (prices start in 2020) the report says when trading
  could really begin, instead of silently losing those days.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from app import logic
from nlt.data.source import DATA_FLOOR
from nlt.engine import warmup
from nlt.spec.models import IndicatorSpec

EMA200 = "Buy NIFTY when close is above 200 EMA. Target 5%, stop loss 2%."


def _spec(text: str):
    spec = logic.translate_text(text).spec
    assert spec is not None
    return spec


def _run(text: str, start: dt.date, end: dt.date = dt.date(2024, 12, 31)):
    return logic.backtest_spec(_spec(text), start=start, end=end)


def test_longest_lookback_adds_up_stacked_lengths() -> None:
    spec = _spec(EMA200)
    assert warmup.longest_lookback(spec) == 200
    macd = spec.model_copy(update={"indicators": [IndicatorSpec(id="m", type="macd", params={})]})
    assert warmup.longest_lookback(macd) == 12 + 26 + 9


def test_padding_reaches_back_three_lookbacks_but_never_before_2020() -> None:
    spec = _spec(EMA200)
    start = dt.date(2024, 1, 1)
    padded = warmup.padded_start(spec, start)
    assert (start - padded).days >= 600 * 7 / 5
    assert warmup.padded_start(spec, dt.date(2020, 6, 1)) == DATA_FLOOR


def test_no_indicator_means_no_padding() -> None:
    spec = _spec(EMA200).model_copy(update={"indicators": []})
    assert warmup.padded_start(spec, dt.date(2024, 1, 1)) == dt.date(2024, 1, 1)


def test_the_indicator_is_ready_on_the_start_date() -> None:
    result = _run(EMA200, dt.date(2024, 1, 1))
    assert result.error is None, result.error
    first = result.backtest.equity.index[0]
    assert first.date() >= dt.date(2024, 1, 1)
    ema_column = next(c for c in result.backtest.features.columns if c.startswith("ema"))
    assert pd.notna(result.backtest.features.loc[first, ema_column])


def test_nothing_is_bought_before_the_start_date() -> None:
    result = _run(EMA200, dt.date(2024, 1, 1))
    assert result.backtest.trades
    assert all(t.entry_time.date() >= dt.date(2024, 1, 1) for t in result.backtest.trades)


def test_the_numbers_cover_only_the_chosen_period() -> None:
    result = _run(EMA200, dt.date(2024, 1, 1))
    assert result.bars.index[0].date() >= dt.date(2024, 1, 1)
    assert result.bars.index[0] == result.backtest.equity.index[0]
    assert result.before_charges.equity.index[0] == result.backtest.equity.index[0]
    capital = logic.DEFAULT_CAPITAL
    expected = 100.0 * (result.backtest.equity.iloc[-1] - capital) / capital
    assert result.backtest.metrics["total_return_pct"] == pytest.approx(expected)


def test_the_report_says_how_much_warm_up_was_used() -> None:
    result = _run(EMA200, dt.date(2024, 1, 1))
    assert any("warmed up on" in n for n in result.data_notes)


def test_running_out_of_history_is_said_with_the_real_first_day() -> None:
    result = _run(EMA200, dt.date(2020, 3, 2), end=dt.date(2021, 6, 30))
    note = next(n for n in result.data_notes if "history exist before your start date" in n)
    assert "not ready" in note
    ready = pd.Timestamp(note.split("until ")[1].split(",")[0])
    assert all(t.entry_time >= ready.tz_localize(t.entry_time.tz) for t in result.backtest.trades)


def test_warm_up_lets_a_long_average_trade_from_day_one() -> None:
    """The point of it: before, a 200-day average lost its first 200 days."""
    spec = _spec(EMA200)
    start, end = dt.date(2024, 1, 1), dt.date(2024, 12, 31)
    warmed = logic.backtest_spec(spec, start=start, end=end)
    first_trade = warmed.backtest.trades[0].entry_time.date()
    assert first_trade < start + dt.timedelta(days=200)


def test_time_in_market_counts_only_the_chosen_period() -> None:
    """Warm-up days are not days out of the market: they are not in the test."""
    from nlt.engine.backtest import BARS_PER_YEAR
    from nlt.engine.metrics import compute_metrics

    result = _run(EMA200, dt.date(2024, 1, 1))
    bt = result.backtest
    expected = compute_metrics(bt.trades, bt.equity, logic.DEFAULT_CAPITAL, BARS_PER_YEAR["1d"])
    assert bt.metrics["exposure_pct"] == pytest.approx(expected["exposure_pct"])
    assert bt.metrics["cagr_pct"] == pytest.approx(expected["cagr_pct"])
