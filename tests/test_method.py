"""The report's "How this was tested": what it assumed, and its weak spots.

Each line claims something about the engine, so each claim that can be checked
is checked here against the engine itself -- the readback's rule, applied to
the report.
"""

from __future__ import annotations

import datetime as dt
import inspect

import pytest

from app import logic
from nlt.engine.backtest import run_backtest
from nlt.engine.basket import run_basket_backtest
from nlt.report import method

WINDOW = {"start": dt.date(2023, 1, 1), "end": dt.date(2024, 12, 31)}
NIFTY = "buy nifty when rsi cracks 30, target 2%, stop loss 1%"
STOCK = "buy reliance when rsi cracks 30, target 5%, stop loss 2%"
BASKET = "buy nifty 50 stocks when rsi cracks 30, target 3%, stop loss 1.5%"


def _run(text: str, **kw):
    result = logic.backtest_spec(logic.translate_text(text).spec, **{**WINDOW, **kw})
    assert result.error is None, result.error
    return result


def test_the_slippage_stated_is_the_one_every_engine_uses() -> None:
    basket_default = inspect.signature(run_basket_backtest).parameters["slippage_pct"].default
    single_default = inspect.signature(run_backtest).parameters["slippage_pct"].default
    assert method.SLIPPAGE_PCT == single_default == basket_default == 0.02
    assert any("0.02% worse" in a for a in _run(NIFTY).assumptions)


def test_same_strategy_same_prices_same_result() -> None:
    """The report promises this, so it is tested: two runs, identical answers."""
    a, b = _run(NIFTY), _run(NIFTY)
    assert method.SAME_RESULT in a.assumptions
    assert [(t.entry_time, t.exit_time, t.quantity, t.net_pnl) for t in a.backtest.trades] == [
        (t.entry_time, t.exit_time, t.quantity, t.net_pnl) for t in b.backtest.trades
    ]
    assert a.backtest.equity.equals(b.backtest.equity)


def test_the_assumptions_name_the_period_capital_and_charges_used() -> None:
    result = _run(STOCK)
    text = " ".join(result.assumptions)
    first = result.backtest.equity.index[0]
    assert f"{first:%d %b %Y}" in text
    assert "Rs 25,00,000" in text
    assert "NSE share delivery charges" in text  # what a share trade is really charged
    assert "risk 1% of capital per trade" in text


def test_the_stop_basis_is_stated_only_when_there_is_a_percent_stop() -> None:
    assert any("close of the candle that gave" in a for a in _run(NIFTY).assumptions)
    trailing = _run("buy tcs when close is above 200 ema, trailing stop 2%")
    assert not any("close of the candle that gave" in a for a in trailing.assumptions)


def test_a_share_strategy_is_told_about_dividends_and_the_depository_fee() -> None:
    spots = " ".join(_run(STOCK).weak_spots)
    assert "Dividends are not counted" in spots
    assert "depository fee" in spots


def test_an_index_strategy_has_no_share_only_weak_spots() -> None:
    spots = " ".join(_run(NIFTY).weak_spots)
    assert "Dividends are not counted" in spots
    assert "depository fee" not in spots


def test_a_basket_says_once_that_it_only_sees_survivors_from_the_chosen_start() -> None:
    result = _run(BASKET)
    survivors = [s for s in result.weak_spots if "only sees survivors" in s]
    assert len(survivors) == 1
    first = result.backtest.equity.index[0].date().isoformat()
    assert first in survivors[0]  # the test's first day, not the warm-up's
    assert not any("only sees survivors" in n for n in result.data_notes)


def test_futures_weak_spots_carry_the_margin_note() -> None:
    from nlt.translate.rules import parse

    spec = parse("buy crude oil futures when rsi cracks 30, target 2%, stop 1%").spec
    spots, notes = method.weak_spots(spec, ["FUTURES ON MARGIN: one lot...", "Other note"])
    assert "FUTURES ON MARGIN: one lot..." in spots
    assert notes == ["Other note"]
    assert not any("Dividends" in s for s in spots)


def test_same_day_share_trades_are_said_to_be_charged_as_delivery() -> None:
    from nlt.translate.rules import parse

    intraday = parse("buy reliance when rsi cracks 30 on 15 minute candles, stop 1%").spec
    spots, _ = method.weak_spots(intraday, [])
    assert any("charged as delivery" in s for s in spots)
    held = parse("buy reliance when rsi cracks 30 on 15 minute candles, stop 1%, hold overnight")
    spots, _ = method.weak_spots(held.spec, [])
    assert not any("charged as delivery" in s for s in spots)


@pytest.mark.parametrize("text", [NIFTY, STOCK])
def test_every_report_ends_with_past_is_not_a_promise(text) -> None:
    assert _run(text).weak_spots[-1].startswith("Past prices are not a promise")


def test_the_loader_dates_its_survivor_note_from_the_test_not_the_warm_up() -> None:
    from nlt.engine.loader import load_for_spec

    spec = logic.translate_text(BASKET).spec
    _, notes = load_for_spec(
        spec, start=dt.date(2022, 6, 1), end=dt.date(2022, 6, 10), tested_from=dt.date(2023, 1, 1)
    )
    survivor = next(n for n in notes if "only sees survivors" in n)
    assert "2023-01-01" in survivor
