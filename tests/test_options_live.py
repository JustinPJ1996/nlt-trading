"""Live option data for paper trading: Kite's list, India VIX, and the gates.

No network: Kite's instrument list and candles are made up here.
"""

from __future__ import annotations

import datetime as dt
import os
import time

import numpy as np
import pandas as pd
import pytest

from app.logic import paper_notes, paper_problem
from nlt.data import kite
from nlt.options.live import vix_iv_fn
from nlt.paper import runner
from nlt.spec.models import IvPercentile
from nlt.translate.rules import parse

IST = "Asia/Kolkata"


def _row(token, exchange, tradingsymbol, segment, name="", **kw):
    return {
        "instrument_token": token,
        "exchange": exchange,
        "tradingsymbol": tradingsymbol,
        "segment": segment,
        "name": name,
        "instrument_type": kw.get("instrument_type", ""),
        "expiry": kw.get("expiry", ""),
        "strike": kw.get("strike", 0.0),
        "lot_size": kw.get("lot_size", 1),
    }


INSTRUMENTS = pd.DataFrame(
    [
        _row(256265, "NSE", "NIFTY 50", "INDICES", "NIFTY 50"),
        _row(265, "BSE", "SENSEX", "INDICES", "SENSEX"),
        _row(137902852, "BSE", "SENSEXBEES", "BSE", "SENSEX"),
        _row(264969, "NSE", "INDIA VIX", "INDICES", "INDIA VIX"),
    ]
)


@pytest.mark.parametrize(("symbol", "token"), [("SENSEX", 265), ("INDIA VIX", 264969)])
def test_sensex_and_india_vix_are_read_from_their_own_index(symbol, token):
    assert kite.instrument_token(symbol, INSTRUMENTS) == token


def _option(day: dt.date, expiry: dt.date):
    return _row(
        1,
        "NFO",
        f"NIFTY{expiry:%y%m%d}23000CE",
        "NFO-OPT",
        "NIFTY",
        instrument_type="CE",
        expiry=expiry.isoformat(),
        strike=23000.0,
        lot_size=65,
    )


def test_an_option_list_still_carrying_yesterdays_expiry_is_not_todays(monkeypatch, tmp_path):
    """Kite publishes the list early each morning; a copy that still lists an
    expired option is yesterday's, and contracts must never be picked from it."""
    cache = tmp_path / "kite_instruments.csv"
    monkeypatch.setattr(kite, "_INSTRUMENTS_CACHE", cache)
    today = dt.date(2026, 10, 16)
    now = pd.Timestamp("2026-10-16 09:20", tz=IST)
    stale = pd.DataFrame([_option(today, dt.date(2026, 10, 15))])
    fresh = pd.DataFrame([_option(today, dt.date(2026, 10, 20))])
    downloads = []

    def download(max_age_hours=20.0):
        downloads.append(max_age_hours)
        stale.to_csv(cache, index=False)
        stamp = now.timestamp()
        os.utime(cache, (stamp, stamp))

    monkeypatch.setattr(kite, "_instruments", download)
    with pytest.raises(kite.KiteUnavailable, match="not published today's list"):
        kite.todays_instruments(now)
    assert downloads  # it did try for a new one

    fresh.to_csv(cache, index=False)
    stamp = now.timestamp()
    os.utime(cache, (stamp, stamp))
    downloads.clear()
    got = kite.todays_instruments(now)
    assert not downloads
    assert list(pd.to_datetime(got["expiry"]).dt.date) == [dt.date(2026, 10, 20)]


def test_yesterdays_download_is_not_todays_list(monkeypatch, tmp_path):
    cache = tmp_path / "kite_instruments.csv"
    monkeypatch.setattr(kite, "_INSTRUMENTS_CACHE", cache)
    pd.DataFrame([_option(dt.date(2026, 10, 16), dt.date(2026, 10, 20))]).to_csv(cache, index=False)
    yesterday = time.time() - 86_400 * 3
    os.utime(cache, (yesterday, yesterday))
    called = []
    monkeypatch.setattr(kite, "_instruments", lambda max_age_hours=20.0: called.append(1))
    with pytest.raises(kite.KiteUnavailable):
        kite.todays_instruments(pd.Timestamp.now(tz=IST))
    assert called


# ---------------------------------------------------------------- India VIX


class _VixKite:
    def __init__(self, closes: pd.Series):
        self.closes = closes

    def index_bars(self, symbol, timeframe, start, end):
        assert (symbol, timeframe) == ("INDIA VIX", "1d")
        s = self.closes[(self.closes.index.date >= start) & (self.closes.index.date <= end)]
        return pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 0.0})


def test_vix_percentile_uses_yesterdays_close_ranked_against_the_days_before():
    days = pd.bdate_range("2026-01-01", periods=40, tz=IST)
    closes = pd.Series(np.arange(10.0, 50.0), index=days)
    closes.iloc[-2] = 5.0  # yesterday: the lowest of all
    closes.iloc[-1] = 99.0  # today's candle, still forming: must not count
    now = days[-1] + pd.Timedelta(hours=11)
    bars = pd.DataFrame(
        {"close": 1.0},
        index=pd.DatetimeIndex([days[-2] + pd.Timedelta(hours=10), now - pd.Timedelta(hours=1)]),
    )
    iv = vix_iv_fn(_VixKite(closes), now)("NIFTY", bars, 20)
    # Today's value: yesterday's 5.0 against the 20 sessions to yesterday -> 0.
    assert iv.iloc[-1] == 0.0
    # Yesterday's value: the day before (48.0, the highest) -> 95 (19 of 20 below).
    assert iv.iloc[0] == pytest.approx(95.0)


def test_vix_never_stands_in_for_banknifty():
    with pytest.raises(ValueError):
        vix_iv_fn(_VixKite(pd.Series(dtype=float)), pd.Timestamp.now(tz=IST))(
            "BANKNIFTY", pd.DataFrame({"close": []}), 20
        )


# -------------------------------------------------------------------- gates


def _iv_spec(symbol: str):
    text = f"Sell the {symbol} weekly ATM straddle at 09:20 with a 30% stop on each leg."
    spec = parse(text).spec
    assert spec is not None
    return spec.model_copy(update={"entry": IvPercentile(op="gt", value=70, lookback_days=252)})


def test_iv_strategies_paper_trade_on_nifty_only():
    assert paper_problem(_iv_spec("NIFTY")) is None
    assert "India VIX" in " ".join(paper_notes(_iv_spec("NIFTY")))
    for symbol in ("BANKNIFTY", "SENSEX"):
        problem = paper_problem(_iv_spec(symbol))
        assert problem and "NIFTY" in problem


def test_options_are_no_longer_refused_and_say_how_paper_differs():
    spec = parse(
        "Buy NIFTY weekly ATM call on 15 minute bars when RSI crosses above 55. "
        "Stop 20%, target 40% on the premium."
    ).spec
    assert paper_problem(spec) is None
    # Squared off at 15:15 every day, expiry day included: nothing differs.
    assert not any("3:40 PM" in n for n in paper_notes(spec))
    overnight = spec.model_copy(
        update={"schedule": spec.schedule.model_copy(update={"intraday": False})}
    )
    assert any("3:40 PM" in n for n in paper_notes(overnight))


def test_option_runs_keep_checking_until_the_fno_close():
    """F&O trades to 15:40 since 3 Aug 2026: an option run's day is not over --
    nor an expiring option's last price known -- at the equity close."""
    option = parse(
        "Buy NIFTY weekly ATM call on 15 minute bars when RSI crosses above 55. "
        "Stop 20%, target 40% on the premium."
    ).spec
    stock = parse("buy nifty when rsi crosses below 40, target 3%, stop 2%").spec
    stock = stock.model_copy(
        update={"instrument": stock.instrument.model_copy(update={"timeframe": "15m"})}
    )
    at = pd.Timestamp("2026-10-15 15:44", tz=IST)
    run = {"id": 1}
    assert runner.is_due(run, None, option, at)
    assert not runner.is_due(run, None, stock, at)


def test_an_alert_for_a_sold_position_never_says_buy():
    from nlt.engine.backtest import Decision

    spec = parse(
        "Sell the NIFTY weekly ATM straddle at 09:20 with a 25% stop on each leg. "
        "Square off at 15:15."
    ).spec
    at = pd.Timestamp("2026-10-15 09:19", tz=IST)
    opened = runner._signal_alert_text(
        spec,
        "S",
        Decision(at, "NIFTY", "signal_taken", "queued", {"contract": "A + B"}),
        "NIFTY",
        at,
    )
    assert "would open its position (A + B)" in opened
    assert "buy" not in opened.replace("Nothing is bought", "")
    again = runner._signal_alert_text(
        spec, "S", Decision(at, "A", "signal_taken", "reentry", {}), "A", at
    )
    assert "would re-enter A" in again
    assert "09:20" not in again  # not the entry rule


def test_a_position_keeps_its_key_when_a_leg_re_enters_at_a_new_strike():
    """A fresh-strike re-entry adds a contract to the position's symbol. The
    paper record must still see the same position, not a new one (which would
    be written as a second entry, and the first as gone)."""
    from types import SimpleNamespace

    from nlt.engine.backtest import Trade

    at = pd.Timestamp("2026-10-15 09:20", tz=IST)
    common = dict(
        entry_time=at,
        entry_price=200.0,
        exit_time=at,
        exit_price=0.0,
        direction="short",
        quantity=65,
        gross_pnl=0.0,
        charges=0.0,
        net_pnl=0.0,
        exit_reason="end_of_data",
        bars_held=1,
        entry_reason="x",
    )
    leg = (SimpleNamespace(symbol="A"),)
    before = Trade(symbol="A + B", legs=leg, **common)
    after = Trade(symbol="A + B + C", legs=leg, **common)
    assert runner._key(before, "NIFTY") == runner._key(after, "NIFTY")
