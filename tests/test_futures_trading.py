"""Futures, end to end: the engine, the benchmark, paper trading, the readback, the parser.

The failure that matters most: a position valued across the switch from one
contract to the next. The two contracts trade at different prices, so the jump
between them would be booked as profit or loss nobody made -- and it would look
exactly like a real result. Close behind it: a lot size off by the 100x between
what Kite says (1) and what one crude lot is (100 barrels).

Nothing here touches Kite or the network.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from app import logic
from nlt.costs.charges import McxFuturesCharges, NseFuturesCharges, ZeroCharges
from nlt.data import kite
from nlt.engine.backtest import run_backtest
from nlt.paper import runner
from nlt.report.benchmark import hold_futures
from nlt.spec.models import (
    Compare,
    Const,
    ExitRules,
    Instrument,
    Ref,
    RiskLimits,
    Schedule,
    StrategySpec,
)
from nlt.store.db import Store
from nlt.translate.readback import describe
from nlt.translate.rules import parse

IST = "Asia/Kolkata"


def _zero(price, qty, side):
    return 0.0


def _spec(
    symbol="CRUDEOIL", timeframe="1d", entry=None, indicators=(), **exit_kwargs
) -> StrategySpec:
    exit_kwargs = exit_kwargs or {"stop_pct": 50.0, "target_pct": 50.0}
    return StrategySpec(
        name="test",
        description="test",
        instrument=Instrument(symbol=symbol, trade_as="future", timeframe=timeframe),
        indicators=list(indicators),
        entry=entry or Compare(op="gt", left=Ref(name="close"), right=Const(value=0)),
        exit=ExitRules(**exit_kwargs),
        risk=RiskLimits(max_position_pct=None),
        schedule=Schedule(intraday=False),
    )


def _daily(start: str, end: str, price) -> pd.DataFrame:
    idx = pd.bdate_range(start, end, tz=IST)
    p = np.asarray(price(idx) if callable(price) else np.full(len(idx), float(price)))
    return pd.DataFrame(
        {"open": p, "high": p * 1.001, "low": p * 0.999, "close": p, "volume": 1000.0}, index=idx
    )


# Crude's October and November 2026 contracts expire on 19 Oct and 19 Nov.
OCT, NOV = dt.date(2026, 10, 19), dt.date(2026, 11, 19)


def _contango(idx: pd.DatetimeIndex) -> np.ndarray:
    """Flat within each contract; each new contract 50% above the last."""
    level = np.ones(len(idx)) * 100.0
    level[idx.date > OCT] = 150.0
    level[idx.date > NOV] = 225.0
    return level


# ---------------------------------------------------------------------- engine


def test_every_position_is_closed_at_the_close_of_its_expiry_day():
    bars = _daily("2026-10-01", "2026-12-10", _contango)
    result = run_backtest(_spec(), bars, capital=10_000_000, charge_fn=_zero, slippage_pct=0)
    expiry_exits = [t for t in result.trades if t.exit_reason == "expiry"]
    assert [t.exit_time.date() for t in expiry_exits] == [OCT, NOV]
    for t in expiry_exits:
        assert t.exit_price == bars.loc[t.exit_time, "close"]


def test_the_jump_between_two_contracts_is_never_booked():
    # A long held through each switch would show +50% twice. Every price within
    # a contract is flat, so the honest profit is nil.
    bars = _daily("2026-10-01", "2026-12-10", _contango)
    result = run_backtest(_spec(), bars, capital=10_000_000, charge_fn=_zero, slippage_pct=0)
    assert result.trades
    assert sum(t.gross_pnl for t in result.trades) == pytest.approx(0.0, abs=1e-6)
    for t in result.trades:
        assert not (t.entry_time.date() <= OCT < t.exit_time.date())
        assert not (t.entry_time.date() <= NOV < t.exit_time.date())


def test_a_signal_on_an_expiry_day_does_not_open_a_trade_in_the_next_contract():
    bars = _daily("2026-10-12", "2026-10-30", 100.0)
    signal_only_on_expiry = Compare(op="gt", left=Ref(name="volume"), right=Const(value=1500))
    bars.loc[bars.index.date == OCT, "volume"] = 2000.0
    result = run_backtest(
        _spec(entry=signal_only_on_expiry), bars, capital=10_000_000, charge_fn=_zero
    )
    assert result.trades == []
    assert any("expiring contract" in w for w in result.warnings)


def test_profit_and_loss_is_in_whole_lots_of_the_real_contract_size():
    # Gold is quoted per 10 grams and one lot is a kilogram: a Rs 100 rise in
    # the quote is Rs 10,000 on one lot -- not Rs 100, which Kite's "lot size 1"
    # would give.
    idx = pd.bdate_range("2026-10-12", "2026-10-16", tz=IST)
    bars = pd.DataFrame(
        {
            "open": [100_000.0, 100_000, 100_100, 100_100, 100_100],
            "high": [100_000.0, 100_000, 100_100, 100_100, 100_100],
            "low": [100_000.0, 100_000, 100_100, 100_100, 100_100],
            "close": [100_000.0, 100_000, 100_100, 100_100, 100_100],
            "volume": 1000.0,
        },
        index=idx,
    )
    capital = 15_000_000  # one 1 kg lot at Rs 1 crore fits; two do not
    result = run_backtest(_spec("GOLD"), bars, capital=capital, charge_fn=_zero, slippage_pct=0)
    first = result.trades[0]
    assert first.quantity == 100  # one lot: 100 x 10 grams
    assert first.gross_pnl == pytest.approx(100 * 100)


def test_an_account_too_small_for_one_lot_is_told_so_and_what_would_do():
    bars = _daily("2026-10-01", "2026-10-15", 8_700.0)
    spec = _spec().model_copy(update={"risk": RiskLimits(max_position_pct=20)})
    result = run_backtest(spec, bars, capital=100_000, charge_fn=_zero)
    assert result.trades == []
    warning = next(w for w in result.warnings if w.startswith("ACCOUNT TOO SMALL"))
    assert "Crude Oil (100 barrels)" in warning
    assert "Rs 8,70,000" in warning  # one lot
    assert "Rs 43,50,000" in warning  # five times that, under the 20% limit


def test_an_evening_mcx_position_is_held_until_the_evening_square_off():
    # MCX's busiest hours are after NSE has shut: a 20:00 entry is held to 23:15.
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-10-06", tz=IST)
            + pd.Timedelta(hours=9)
            + pd.Timedelta(minutes=15 * k)
            for k in range(58)
        ]
    )
    p = np.full(len(idx), 5000.0)
    bars = pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1.0}, index=idx)
    entry = Compare(op="gt", left=Ref(name="volume"), right=Const(value=1.5))
    bars.loc[idx[44], "volume"] = 2.0  # the 20:00 candle
    spec = _spec("CRUDEOILM", "15m", entry=entry).model_copy(
        update={"schedule": Schedule(no_entry_after=dt.time(23), square_off=dt.time(23, 15))}
    )
    result = run_backtest(
        spec, bars, capital=1_000_000, charge_fn=_zero, drop_partial_last_bar=False
    )
    (trade,) = result.trades
    assert trade.entry_time.time() == dt.time(20, 15)
    assert trade.exit_time.time() == dt.time(23, 15)
    assert trade.exit_reason == "square_off"


def test_mcx_strategies_count_sessions_in_mcx_hours():
    # The "your indicator spans several sessions" warning divides by bars per
    # session: 60 fifteen-minute candles from 09:00 to 23:55, not NSE's 25.
    idx = pd.DatetimeIndex(
        [_at("2026-10-06 09:00") + pd.Timedelta(minutes=15 * k) for k in range(58)]
    )
    p = np.full(len(idx), 5000.0)
    bars = pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1.0}, index=idx)
    spec = _spec(
        "CRUDEOILM",
        "15m",
        entry=Compare(op="gt", left=Ref(name="close"), right=Ref(name="sma100")),
        indicators=[{"id": "sma100", "type": "sma", "params": {"length": 100}}],
    )
    result = run_backtest(
        spec, bars, capital=1_000_000, charge_fn=_zero, drop_partial_last_bar=False
    )
    assert any("at 60 bars/session" in w for w in result.warnings)


def test_a_position_that_outlived_its_contract_stops_the_backtest(monkeypatch):
    """The second line of defence: if the expiry exit ever failed to fire, the
    engine must refuse to carry on, not value the position on the next
    contract's price."""
    from nlt.engine import backtest

    real = backtest._check_exit

    def forgets_expiry(*args, **kwargs):
        outcome = real(*args, **kwargs)
        return None if outcome is not None and outcome[0] == "expiry" else outcome

    monkeypatch.setattr(backtest, "_check_exit", forgets_expiry)
    bars = _daily("2026-10-01", "2026-12-10", _contango)
    with pytest.raises(RuntimeError, match="held past its contract's expiry"):
        run_backtest(_spec(), bars, capital=10_000_000, charge_fn=_zero)


# ------------------------------------------------------------------- benchmark


def test_holding_futures_never_counts_the_jump_between_contracts():
    bars = _daily("2026-10-01", "2026-12-10", _contango)
    expiry = np.isin(bars.index.date, [OCT, NOV])
    bench = hold_futures(
        bars, 10_000_000, multiplier=100, expiry_close=expiry, name="holding", charge_fn=None
    )
    assert bench.total_return_pct == pytest.approx(0.0, abs=1e-9)


def test_holding_futures_is_charged_at_every_expiry():
    bars = _daily("2026-10-01", "2026-12-10", 100.0)
    expiry = np.isin(bars.index.date, [OCT, NOV])
    charged = hold_futures(
        bars, 10_000_000, multiplier=100, expiry_close=expiry, name="h", charge_fn=lambda *a: 10.0
    )
    # Three holdings (to Oct, to Nov, to the end), each bought and sold once.
    assert charged.equity.iloc[-1] == pytest.approx(10_000_000 - 6 * 10.0)


# ----------------------------------------------------------------------- costs


@pytest.mark.parametrize(
    ("symbol", "model"), [("CRUDEOIL", McxFuturesCharges), ("NIFTY", NseFuturesCharges)]
)
def test_futures_pay_their_own_exchanges_charges_whatever_the_dropdown_says(symbol, model):
    chosen, note = logic.charge_model_for_spec(
        _spec(symbol), "Index options (NIFTY/BANKNIFTY weekly)"
    )
    assert isinstance(chosen, model)
    assert note is not None


def test_no_costs_stays_no_costs_for_futures():
    chosen, note = logic.charge_model_for_spec(_spec(), "No costs (for comparison only)")
    assert isinstance(chosen, ZeroCharges) and note is None


# -------------------------------------------------------------- paper trading


def _at(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=IST)


@pytest.mark.parametrize(
    ("symbol", "timeframe", "now", "status", "due"),
    [
        ("CRUDEOIL", "5m", "2026-10-05 21:00", None, True),  # MCX open in the evening
        ("NIFTY", "5m", "2026-10-05 21:00", None, False),  # NSE is not
        ("CRUDEOIL", "5m", "2026-10-05 08:50", None, False),  # before MCX opens at 9
        ("CRUDEOIL", "1d", "2026-10-05 16:00", None, False),  # today's candle still forming
        ("CRUDEOIL", "1d", "2026-10-05 23:58", None, True),  # finished
        # Friday's candle not yet seen on Saturday morning: look for it.
        (
            "CRUDEOIL",
            "1d",
            "2026-10-10 10:00",
            {"last_bar_time": "2026-10-08T00:00:00+05:30"},
            True,
        ),
        # ...and once it is recorded, the weekend is quiet.
        (
            "CRUDEOIL",
            "1d",
            "2026-10-10 10:00",
            {"last_bar_time": "2026-10-09T00:00:00+05:30"},
            False,
        ),
        # Monday 02:00, Friday's candle recorded: nothing new until Monday's close.
        (
            "CRUDEOIL",
            "1d",
            "2026-10-12 02:00",
            {"last_bar_time": "2026-10-09T00:00:00+05:30"},
            False,
        ),
    ],
)
def test_paper_trading_follows_each_markets_hours(symbol, timeframe, now, status, due):
    assert runner.is_due({}, status, _spec(symbol, timeframe), _at(now)) is due


def test_an_mcx_daily_candle_is_not_finished_until_mcx_closes():
    bars = _daily("2026-10-05", "2026-10-06", 100.0)
    from nlt.data.futures import contract

    mcx = contract("CRUDEOIL").session
    assert len(runner.closed_bars(bars, "1d", _at("2026-10-06 20:00"), mcx)) == 1
    assert len(runner.closed_bars(bars, "1d", _at("2026-10-06 23:56"), mcx)) == 2


def _intraday(days: list[str], level: float) -> pd.DataFrame:
    idx = pd.DatetimeIndex(
        [
            _at(d) + pd.Timedelta(hours=9) + pd.Timedelta(minutes=15 * k)
            for d in days
            for k in range(58)
        ]
    )
    rng = np.random.default_rng(int(level))
    close = level * np.exp(np.cumsum(rng.normal(0, 0.002, len(idx))))
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1.0},
        index=idx,
    )


def test_after_an_expiry_paper_trading_does_not_rejudge_the_old_contract(tmp_path):
    """Kite only has intraday candles for live contracts. The day after an
    expiry, the series fetched is the new contract's history -- different
    prices for days already traded. Nothing recorded before the switch may be
    called drift, and no trade may appear in the past that was never live."""
    store = Store(tmp_path / "s.db")
    spec = _spec(
        "CRUDEOILM",
        "15m",
        entry=Compare(op="crosses_above", left=Ref(name="close"), right=Ref(name="ema5")),
        indicators=[{"id": "ema5", "type": "ema", "params": {"length": 5}}],
        stop_pct=0.3,
        target_pct=0.3,
    )
    sid = store.save_strategy(spec, describe(spec))
    run_id = store.start_run(
        sid, "paper", {"capital": 2_000_000, "started_at": "2026-10-14T09:00:00+05:30"}
    )
    run = store.get_run(run_id)
    days_old = ["2026-10-14", "2026-10-15", "2026-10-16", "2026-10-19"]
    old_contract = _intraday(days_old, 5000.0)
    new_contract = _intraday([*days_old, "2026-10-20"], 5100.0)  # a different contract's prices

    runner.step(
        store, run, fetch=lambda *a: old_contract, charge_fn=_zero, now=_at("2026-10-19 23:59")
    )
    before = store.paper_events(run_id)
    assert any(e["kind"] == "entry" for e in before)

    runner.step(
        store, run, fetch=lambda *a: new_contract, charge_fn=_zero, now=_at("2026-10-20 23:59")
    )
    after = store.paper_events(run_id)
    added = after[len(before) :]
    assert not [e for e in added if e["kind"] == "drift"]
    switch = _at("2026-10-20 09:00")
    assert all(pd.Timestamp(e["bar_time"]) >= switch for e in added if e["bar_time"])


def test_paper_trading_asks_for_the_futures_series_not_the_index(tmp_path):
    asked = []

    def fetch(symbol, *a):
        asked.append(symbol)
        return pd.DataFrame()

    store = Store(tmp_path / "s.db")
    spec = _spec("NIFTY")
    sid = store.save_strategy(spec, describe(spec))
    run_id = store.start_run(sid, "paper", {"started_at": "2026-10-14T09:00:00+05:30"})
    runner.step(
        store, store.get_run(run_id), fetch=fetch, charge_fn=_zero, now=_at("2026-10-15 16:00")
    )
    assert asked == ["NIFTY FUT"]


# ------------------------------------------------------------- the readback


def test_the_readback_says_what_one_lot_is_and_that_expiry_closes_it():
    text = describe(parse("buy gold mini when rsi cracks 30, target 2%, stop loss 1%").spec)
    assert "Gold Mini futures (MCX)" in text
    assert "1 lot = 100 grams, price quoted per 10 grams" in text
    assert "Close at the end of the contract's expiry day if still open" in text


def test_the_readback_names_the_exchange_as_people_know_it():
    # Kite's code for NSE's derivatives segment is "NFO"; nobody else calls it that.
    text = describe(parse("short nifty futures on death cross, target 2%, stop 1%").spec)
    assert "NIFTY futures (NSE)" in text


def test_the_activity_log_counts_futures_in_lots():
    spec = _spec("GOLD")
    event = {
        "kind": "entry",
        "quantity": 200,
        "symbol": "GOLD",
        "price": 100_000.0,
        "bar_time": "2026-10-07T00:00:00+05:30",
        "direction": "long",
        "detail_json": json.dumps({}),
    }
    line = logic.describe_paper_event(event, spec)
    assert line.startswith("Bought 2 lots of Gold futures (1 kg each)")


# ---------------------------------------------------------------- the parser


@pytest.mark.parametrize(
    ("words", "symbol"),
    [
        ("crude oil futures", "CRUDEOIL"),
        ("crude", "CRUDEOIL"),
        ("crude oil mini", "CRUDEOILM"),
        ("mini crude", "CRUDEOILM"),
        ("natural gas", "NATURALGAS"),
        ("natural gas mini", "NATGASMINI"),
        ("gold", "GOLD"),
        ("gold mini", "GOLDM"),
        ("goldm", "GOLDM"),
        ("mcx silver", "SILVER"),
        ("silver mini", "SILVERM"),
        ("nifty futures", "NIFTY"),
        ("banknifty futures", "BANKNIFTY"),
    ],
)
def test_contract_names_are_read_to_the_right_contract(words, symbol):
    spec = parse(f"buy {words} when rsi cracks 30, target 2%, stop loss 1%").spec
    assert spec is not None
    assert (spec.instrument.symbol, spec.instrument.trade_as) == (symbol, "future")


def test_bare_nifty_is_still_the_index():
    spec = parse("buy nifty when rsi cracks 30, target 2%, stop loss 1%").spec
    assert spec.instrument.trade_as == "index"


@pytest.mark.parametrize(
    "words", ["silver micro", "gold petal", "copper", "tcs futures", "sensex futures"]
)
def test_contracts_this_platform_does_not_have_are_refused_not_substituted(words):
    result = parse(f"buy {words} when rsi cracks 30, target 2%, stop loss 1%")
    assert result.spec is None
    assert "Futures work on" in result.questions[0].text


def test_an_mcx_intraday_strategy_squares_off_in_the_evening_not_at_nse_hours():
    spec = parse(
        "buy crude oil on 15 minute candles when rsi crosses below 30, target 1%, stop 0.5%"
    ).spec
    assert spec.schedule.square_off == dt.time(23, 15)
    assert spec.schedule.no_entry_after == dt.time(23, 0)


def test_futures_get_the_share_position_limit():
    spec = parse("buy crude oil when rsi cracks 30, target 2%, stop loss 1%").spec
    assert spec.risk.max_position_pct == 20.0


def test_the_ai_reader_keeps_a_futures_contract_a_future():
    """When the rules name the contract and the model fills in the rest, the
    recipe must trade the future -- not the index, and not a share called GOLD."""
    from nlt.translate import llm

    sentence = (
        "Go long on gold mini futures the moment its 9-day EMA climbs over its 50-day EMA, "
        "bank 4% profit, SL 2%"
    )
    reply = {
        "kind": "strategy",
        "symbol": "GOLDM",
        "direction": "long",
        "indicators": [
            {"id": "ema9", "type": "ema", "params": {"length": 9}},
            {"id": "ema50", "type": "ema", "params": {"length": 50}},
        ],
        "entry": {
            "kind": "compare",
            "op": "crosses_above",
            "left": {"kind": "ref", "name": "ema9"},
            "right": {"kind": "ref", "name": "ema50"},
        },
        "exit": {"target_pct": 4, "stop_pct": 2},
        "evidence": [
            {"field": "direction", "quote": "Go long"},
            {"field": "instrument", "quote": "gold mini futures"},
            {"field": "entry", "quote": "the moment its 9-day EMA climbs over its 50-day EMA"},
            {"field": "exit.target", "quote": "bank 4% profit"},
            {"field": "exit.stop", "quote": "SL 2%"},
        ],
    }
    assert parse(sentence).spec is None  # the rules alone cannot read it
    result = llm.parse_llm(sentence, transport=lambda system, user: json.dumps(reply))
    assert result.spec is not None, result.questions
    assert (result.spec.instrument.symbol, result.spec.instrument.trade_as) == ("GOLDM", "future")
    assert result.spec.schedule.square_off == dt.time(23, 15)


# ------------------------------------------------------------------- Kite


def _instruments_with_futures() -> pd.DataFrame:
    today = dt.date.today()
    rows = [
        (256265, "NIFTY 50", "INDICES", "EQ", "NSE", None, "NIFTY 50"),
        (111, "CRUDEOIL26XFUT", "MCX-FUT", "FUT", "MCX", today, "CRUDEOIL"),
        (222, "CRUDEOIL26YFUT", "MCX-FUT", "FUT", "MCX", today + dt.timedelta(days=30), "CRUDEOIL"),
        (333, "NIFTY26XFUT", "NFO-FUT", "FUT", "NFO", today + dt.timedelta(days=5), "NIFTY"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "instrument_token",
            "tradingsymbol",
            "segment",
            "instrument_type",
            "exchange",
            "expiry",
            "name",
        ],
    )


class _Http:
    def __init__(self):
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {})))

        class R:
            status_code = 200

            def json(self):
                return {
                    "status": "success",
                    "data": {"candles": [["2026-10-06T00:00:00+0530", 1, 1, 1, 1, 1]]},
                }

        return R()


@pytest.fixture
def kite_with_futures(monkeypatch):
    monkeypatch.setattr(kite, "_instruments", lambda *a, **k: _instruments_with_futures())
    monkeypatch.setattr(kite, "_REQUEST_GAP_SECONDS", 0)
    kite.save_token("FAKE+token/for==tests")


def test_on_its_expiry_day_a_contract_is_still_the_near_month(kite_with_futures):
    assert int(kite.near_month("CRUDEOIL", dt.date.today())["instrument_token"]) == 111


def test_daily_futures_come_from_kites_continuous_series(kite_with_futures):
    http = _Http()
    end = dt.date.today()
    kite.KiteSource(http).bars("CRUDEOIL FUT", "1d", end - dt.timedelta(days=10), end)
    url, params = http.calls[0]
    assert "/historical/111/day" in url
    assert params.get("continuous") == 1


def test_intraday_futures_say_how_little_history_there_is(kite_with_futures):
    http = _Http()
    end = dt.date.today()
    source = kite.KiteSource(http)
    source.bars("NIFTY FUT", "15m", end - dt.timedelta(days=10), end)
    url, params = http.calls[0]
    assert "/historical/333/15minute" in url and "continuous" not in params
    assert any(
        "Kite keeps no intraday history for expired contracts" in n for n in source.last_notes
    )


def test_the_index_is_still_the_index(kite_with_futures):
    http = _Http()
    end = dt.date.today()
    kite.KiteSource(http).bars("NIFTY", "1d", end - dt.timedelta(days=10), end)
    assert "/historical/256265/day" in http.calls[0][0]
