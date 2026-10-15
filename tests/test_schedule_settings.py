"""The settings screen (Justin, 2026-10-15): algo type, entry cut-off, square-off.

Settings live on the strategy and count back from each market's own close:
stocks and the index 15:30, NSE/BSE futures and options 15:40, MCX 23:30 (its
earlier close). Defaults: no new entries in the last 30 minutes, square off 15
minutes before the close.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from test_options_engine import ATM_CE, D1, _chain, _index_bars, _run, _spec, _ts
from test_options_paper_rules import _minute, _plus

from app import logic
from nlt.engine.backtest import past_entry_cutoff
from nlt.spec import clock
from nlt.translate.readback import describe
from nlt.translate.rules import parse

IST = "Asia/Kolkata"

STOCK = "buy nifty when rsi crosses below 40 on 15 minute candles, target 3%, stop 2%"
OPTION = (
    "Buy NIFTY weekly ATM call on 15 minute bars when RSI crosses above 55. "
    "Stop 20%, target 40% on the premium."
)
DAILY = "buy nifty when rsi crosses below 40, target 3%, stop 2%"


def _parsed(text):
    spec = parse(text).spec
    assert spec is not None, text
    return spec


# ----------------------------------------------------------- each market's close


@pytest.mark.parametrize(
    ("trade_as", "symbol", "close"),
    [
        ("index", "NIFTY", dt.time(15, 30)),
        ("stock", "RELIANCE", dt.time(15, 30)),
        ("option", "NIFTY", dt.time(15, 40)),
        ("future", "NIFTY", dt.time(15, 40)),
        ("future", "CRUDEOIL", dt.time(23, 30)),
    ],
)
def test_minutes_count_back_from_each_markets_own_close(trade_as, symbol, close):
    assert clock.market_close(trade_as, symbol) == close
    at = clock.before_close(trade_as, symbol, 15)
    assert clock.minutes_before_close(trade_as, symbol, at) == 15


def test_defaults_are_30_and_15_minutes_before_the_close():
    stock = _parsed(STOCK).schedule
    assert (stock.no_entry_after, stock.square_off) == (dt.time(15, 0), dt.time(15, 15))
    option = _parsed(OPTION).schedule
    assert (option.no_entry_after, option.square_off) == (dt.time(15, 10), dt.time(15, 25))


def test_mcx_says_which_close_it_counts_from():
    spec = _parsed("buy crude oil futures when rsi crosses below 30 on 15 minute candles, stop 1%")
    assert spec.schedule.square_off == dt.time(23, 15)
    settings = logic.schedule_settings(spec)
    assert settings["square_off_minutes"] == 15
    assert "11:55 PM in winter" in settings["close_note"]


def test_the_screen_shows_a_strategys_settings_in_minutes():
    s = logic.schedule_settings(_parsed(OPTION))
    assert s["algo_type"] == "intraday"
    assert (s["entry_cutoff"], s["entry_cutoff_minutes"], s["square_off_minutes"]) == (
        True,
        30,
        15,
    )
    assert s["close"] == "15:40"
    daily = logic.schedule_settings(_parsed(DAILY))
    assert daily["algo_type"] == "multi_day" and not daily["can_be_intraday"]


# ------------------------------------------------------------ the entry cut-off


def _at(hhmm):
    return pd.Timestamp(f"2026-10-15 {hhmm}", tz=IST)


def test_the_cut_off_is_about_when_the_trade_would_open():
    spec = _parsed(STOCK)  # 15:00 cut-off
    assert not past_entry_cutoff(spec, _at("14:45"), "15m")  # fills 15:00
    assert past_entry_cutoff(spec, _at("15:00"), "15m")  # fills 15:15
    # A 1-hour candle from 14:15 would fill at 15:15, after 15:00.
    assert past_entry_cutoff(spec, _at("14:15"), "1h")
    assert not past_entry_cutoff(spec, _at("13:15"), "1h")
    assert not past_entry_cutoff(spec, _at("15:15"), "1d")


def test_a_cut_off_switched_off_never_stops_an_entry():
    spec = _parsed(STOCK)
    off = spec.model_copy(
        update={"schedule": spec.schedule.model_copy(update={"entry_cutoff": False})}
    )
    assert not past_entry_cutoff(off, _at("15:00"), "15m")
    assert "No new entries" not in describe(off)


def test_a_multi_day_strategy_has_no_cut_off_unless_ticked():
    spec = _parsed(STOCK)
    multi = spec.model_copy(
        update={"schedule": spec.schedule.model_copy(update={"intraday": False})}
    )
    assert not past_entry_cutoff(multi, _at("15:00"), "15m")
    ticked = multi.model_copy(
        update={"schedule": multi.schedule.model_copy(update={"entry_cutoff": True})}
    )
    assert past_entry_cutoff(ticked, _at("15:00"), "15m")


def test_the_readback_says_the_cut_off_it_used_to_hide():
    text = describe(_parsed(STOCK))
    assert "No new entries after 15:00 (30 minutes before the 15:30 close)" in text


# ------------------------------------------------------------------ the boxes


def _adjust(text, **edits):
    spec, _sources, _typed = logic.adjust_spec(_parsed(text), edits, None)
    return spec


def test_square_off_minutes_set_the_clock_for_that_market():
    assert _adjust(STOCK, square_off_minutes="10").schedule.square_off == dt.time(15, 20)
    assert _adjust(OPTION, square_off_minutes="10").schedule.square_off == dt.time(15, 30)


@pytest.mark.parametrize(
    ("edits", "says"),
    [
        ({"square_off_minutes": "9"}, "between 10 and 30"),
        ({"square_off_minutes": "31"}, "between 10 and 30"),
        ({"entry_cutoff_minutes": "401"}, "between 10 and 400"),
        ({"entry_cutoff_minutes": "400"}, "before the market opens"),
        ({"entry_cutoff_minutes": "10", "square_off_minutes": "20"}, "at or before the square-off"),
        ({"algo_type": "weekly"}, "intraday or multi-day"),
        ({"square_off_minutes": "ten"}, "whole number"),
    ],
)
def test_impossible_settings_are_refused_in_plain_words(edits, says):
    with pytest.raises(logic.AdjustError, match=says):
        _adjust(STOCK, **edits)


def test_daily_candles_cannot_be_intraday():
    with pytest.raises(logic.AdjustError, match="only be multi-day"):
        _adjust(DAILY, algo_type="intraday")


def test_switching_to_multi_day_holds_overnight_and_says_so():
    spec = _adjust(STOCK, algo_type="multi_day")
    assert not spec.schedule.intraday
    text = describe(spec)
    assert "Hold overnight" in text and "Square off at" not in text


def test_a_cut_off_switched_off_on_the_screen_leaves_the_readback():
    spec = _adjust(STOCK, entry_cutoff=False)
    assert spec.schedule.entry_cutoff is False
    assert "No new entries" not in describe(spec)


def test_a_square_off_between_candles_is_still_said_as_the_time_itself():
    """Every square-off sells at the close of its own minute (2026-10-15), so
    15:10 on 15-minute candles needs no note about candles."""
    spec = _adjust(STOCK, square_off_minutes="20")
    assert "Square off at 15:10 if still open\n" in describe(spec) + "\n"


def _square_off_at(clock_time, minutes):
    source = _plus(_chain([D1], {}), minutes)
    spec = _spec(stop=30.0, target=100.0)
    spec = spec.model_copy(
        update={"schedule": spec.schedule.model_copy(update={"square_off": clock_time})}
    )
    signal = _ts(D1, 10, 0)
    return _run(spec, _index_bars([D1], signal), source).trades[0]


def test_an_option_squares_off_at_the_minute_chosen_inside_a_candle():
    """15:25 falls inside the 15:15-15:30 candle: the first trade from 15:25."""
    t = _square_off_at(dt.time(15, 25), [_minute(ATM_CE, _ts(D1, 15, 25), 91.0)])
    assert (t.exit_reason, t.exit_price) == ("square_off", 91.0)


def test_the_stop_still_counts_in_the_minutes_before_the_square_off():
    """Bought at 100, stop 70. At 15:20 -- before a 15:25 square-off, in the
    same candle -- it trades at 60: stopped at 70, not squared off. The
    minutes before the clock are one candle that opened at 100, so the stop
    fills at its level, as on any candle."""
    t = _square_off_at(
        dt.time(15, 25),
        [_minute(ATM_CE, _ts(D1, 15, 20), 60.0), _minute(ATM_CE, _ts(D1, 15, 25), 91.0)],
    )
    assert (t.exit_reason, t.exit_price) == ("stop", 70.0)


def test_an_option_square_off_after_the_index_closes_uses_the_option_minutes():
    """15:30 -- 10 minutes before the 15:40 F&O close, after the index's last
    candle: the option's own 15:30 minute, its price at 15:30:59."""
    t = _square_off_at(dt.time(15, 30), [_minute(ATM_CE, _ts(D1, 15, 30), 88.0)])
    assert (t.exit_reason, t.exit_price) == ("square_off", 88.0)


# ------------------------------------------------------------- the API


@pytest.fixture
def client(tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    from app.api import create_app
    from nlt.store.db import Store

    monkeypatch.setenv("NLT_DASHBOARD_LOGIN_ID", "t")
    monkeypatch.setenv("NLT_DASHBOARD_PASSWORD", "pass-for-tests-123")
    store = Store(tmp_path / "s.sqlite")
    c = TestClient(create_app(store), base_url="https://testserver")
    assert c.post("/api/login", json={"login_id": "t", "password": "pass-for-tests-123"}).is_success
    return c, store


def test_changing_settings_on_the_paper_form_saves_a_new_version(client):
    c, store = client
    spec = _parsed(STOCK)
    sid = store.save_strategy(spec, describe(spec))
    r = c.post(f"/api/strategies/{sid}/settings", json={"edits": {"square_off_minutes": 10}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["new_version"] and body["id"] != sid and body["version"] == 2
    assert "Square off at 15:20" in body["readback"]
    assert body["settings"]["square_off_minutes"] == 10
    # The version that was tested keeps its own settings.
    assert store.load_spec(sid).schedule.square_off == dt.time(15, 15)


def test_impossible_settings_are_refused_by_the_api(client):
    c, store = client
    spec = _parsed(DAILY)
    sid = store.save_strategy(spec, describe(spec))
    r = c.post(f"/api/strategies/{sid}/settings", json={"edits": {"algo_type": "intraday"}})
    assert r.status_code == 400 and "multi-day" in r.text


def test_an_option_square_off_inside_an_earlier_candle_is_not_late():
    """15:10 (30 minutes before the 15:40 close) is inside the 15:00 candle,
    not the day's last: the 15:10 trade, not the 15:15 candle's open."""
    t = _square_off_at(dt.time(15, 10), [_minute(ATM_CE, _ts(D1, 15, 10), 91.0)])
    assert (t.exit_reason, t.exit_price) == ("square_off", 91.0)


def test_a_sold_legs_stop_still_counts_in_the_minutes_before_the_square_off():
    """Straddle on 5-minute candles, square-off 15:27 (inside the 15:25 candle).
    The call (sold 102, stop 127) trades at 130 at 15:26: stopped at 127."""
    from test_legs_engine import CE as LEG_CE
    from test_legs_engine import OPEN_PRICES, PE_FLAT, _legs
    from test_legs_engine import _run as _run_legs
    from test_legs_engine import _source as _legs_source
    from test_legs_engine import _spec as _legs_spec

    src = _plus(
        _legs_source(OPEN_PRICES, PE_FLAT),
        [_minute(LEG_CE, _ts(D1, 15, 26), 130.0), _minute(LEG_CE, _ts(D1, 15, 27), 128.0)],
    )
    spec = _legs_spec()
    spec = spec.model_copy(
        update={"schedule": spec.schedule.model_copy(update={"square_off": dt.time(15, 27)})}
    )
    t = _run_legs(spec, src, days=(D1,)).trades[0]
    (ce,) = _legs(t, LEG_CE)
    assert (ce.exit_reason, ce.exit_price) == ("stop", 127.0)


# ------------------------------------- stocks, index, futures: the minute's close


def _stock_day(
    square_off=dt.time(15, 15), minute_overrides=None, with_minutes=True, minutes_day=None
):
    """One NSE day of 15-minute NIFTY candles, flat at 100, an entry at 10:00,
    and 1-minute candles under them (also flat unless overridden)."""
    from nlt.engine.backtest import run_backtest
    from nlt.spec.models import Compare, Const, Ref

    day = "2026-10-15"
    idx15 = pd.date_range(f"{day} 09:15", periods=25, freq="15min", tz=IST)
    bars = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 0.0}, index=idx15
    )
    bars.loc[pd.Timestamp(f"{day} 09:45", tz=IST), "close"] = 101.0  # the entry signal
    idx1 = pd.date_range(f"{minutes_day or day} 09:15", periods=375, freq="1min", tz=IST)
    minutes = pd.DataFrame(
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 0.0}, index=idx1
    )
    for hhmm, row in (minute_overrides or {}).items():
        minutes.loc[pd.Timestamp(f"{day} {hhmm}", tz=IST), ["open", "high", "low", "close"]] = row
    spec = _parsed(STOCK).model_copy(
        update={
            "entry": Compare(op="gt", left=Ref(name="close"), right=Const(value=100.5)),
            "indicators": [],
            "schedule": _parsed(STOCK).schedule.model_copy(update={"square_off": square_off}),
        }
    )
    return run_backtest(
        spec,
        bars,
        capital=1_000_000.0,
        drop_partial_last_bar=False,
        minute_bars=minutes if with_minutes else None,
    )


def test_a_stock_squares_off_at_the_close_of_the_1515_minute():
    """Justin: "Square off at 15:15 should take the close of the 15:15
    1-minute candle -- the price at 15:15:59"."""
    r = _stock_day(minute_overrides={"15:15": (100.0, 100.0, 100.0, 103.0)})
    (t,) = r.trades
    assert (t.exit_reason, t.exit_price) == ("square_off", 103.0)


def test_a_stock_square_off_between_candles_uses_that_minute():
    r = _stock_day(
        square_off=dt.time(15, 10), minute_overrides={"15:10": (100.0, 100.0, 100.0, 104.0)}
    )
    (t,) = r.trades
    assert (t.exit_reason, t.exit_price) == ("square_off", 104.0)


def test_a_stock_stop_in_the_minutes_before_the_square_off_still_counts():
    """Bought at 100 with a 2% stop (98.0, from the 101 signal close: 98.98).
    At 15:05 -- in the candle the 15:10 square-off falls in -- it trades at 97."""
    r = _stock_day(
        square_off=dt.time(15, 10), minute_overrides={"15:05": (100.0, 100.0, 97.0, 97.0)}
    )
    (t,) = r.trades
    assert t.exit_reason == "stop"


def test_without_minutes_the_old_rule_holds_and_nothing_is_counted():
    r = _stock_day(minute_overrides={"15:15": (100.0, 100.0, 100.0, 103.0)}, with_minutes=False)
    (t,) = r.trades
    assert (t.exit_reason, t.exit_price) == ("square_off", 100.0)


def test_a_day_without_minutes_is_counted_in_the_result():
    """Minutes loaded, but none for the trade's day: the old rule that day,
    and the result says so."""
    r = _stock_day(minutes_day="2026-10-14")
    (t,) = r.trades
    assert (t.exit_reason, t.exit_price) == ("square_off", 100.0)
    assert any("On 1 day(s) there were no 1-minute prices" in w for w in r.warnings)
    assert not any("no 1-minute prices" in w for w in _stock_day().warnings)


def test_paper_trading_reads_the_minutes_and_sells_at_the_1515_close(tmp_path):
    from nlt.engine.backtest import run_backtest  # noqa: F401
    from nlt.paper import runner
    from nlt.spec.models import Compare, Const, Ref
    from nlt.store.db import Store

    day = "2026-10-15"
    idx15 = pd.date_range(f"{day} 09:15", periods=25, freq="15min", tz=IST)
    bars = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 0.0}, index=idx15
    )
    bars.loc[pd.Timestamp(f"{day} 09:45", tz=IST), "close"] = 101.0
    idx1 = pd.date_range(f"{day} 09:15", periods=375, freq="1min", tz=IST)
    minutes = pd.DataFrame(
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "volume": 0.0}, index=idx1
    )
    minutes.loc[pd.Timestamp(f"{day} 15:15", tz=IST), "close"] = 103.0
    spec = _parsed(STOCK).model_copy(
        update={
            "entry": Compare(op="gt", left=Ref(name="close"), right=Const(value=100.5)),
            "indicators": [],
        }
    )
    store = Store(tmp_path / "p.sqlite")
    sid = store.save_strategy(spec, describe(spec))
    rid = store.start_run(
        sid,
        "paper",
        {
            "capital": 1_000_000.0,
            "started_at": f"{day}T09:14:00+05:30",
            "slippage": {"kind": "pct", "value": 0},
        },
    )
    asked = []

    def fetch(symbol, timeframe, start, end):
        asked.append((symbol, timeframe, start))
        return minutes if timeframe == "1m" else bars

    runner.step(
        store,
        store.get_run(rid),
        fetch=fetch,
        charge_fn=lambda p, q, s: 0.0,
        now=pd.Timestamp(f"{day} 15:31", tz=IST),
    )
    assert ("NIFTY", "1m", dt.date(2026, 10, 15)) in asked
    (exit_,) = [e for e in store.paper_events(rid) if e["kind"] == "exit"]
    assert exit_["price"] == pytest.approx(103.0)


def test_a_stop_after_the_square_off_minute_never_counts():
    """Squared off at 15:10 (91); the 60 at 15:12 came after it, in the same
    15:00 candle, and must not be read as the stop."""
    t = _square_off_at(
        dt.time(15, 10),
        [_minute(ATM_CE, _ts(D1, 15, 10), 91.0), _minute(ATM_CE, _ts(D1, 15, 12), 60.0)],
    )
    assert (t.exit_reason, t.exit_price, t.exit_time) == ("square_off", 91.0, _ts(D1, 15, 0))
