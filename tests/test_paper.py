"""Paper trading, replayed.

The central property: replaying past candles one at a time through the paper
runner, as if each were arriving live, must record exactly the trades a
backtest finds on the same candles -- no more, no fewer, same prices -- and
must never record anything before the candle it depends on had finished.

No test here touches Kite or the network: `fetch` is a function returning
made-up candles, and `conftest._no_live_kite` hides the real token.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nlt.data import kite
from nlt.engine.backtest import run_backtest
from nlt.paper import runner
from nlt.spec.models import Instrument, StrategySpec
from nlt.store.db import Store
from nlt.translate.readback import describe
from nlt.translate.rules import parse

IST = "Asia/Kolkata"
CAPITAL = 1_000_000.0  # one NIFTY unit must fit under the 20%-per-position limit


def _zero(price, qty, side):
    return 0.0


def _daily_bars(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 20000 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.005, n)))
    idx = pd.bdate_range("2024-01-01", periods=n, tz=IST)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 0.0}, index=idx
    )


def _intraday_bars(sessions: int = 7, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2024-03-04", periods=sessions, tz=IST)
    idx = pd.DatetimeIndex(
        [
            d + pd.Timedelta(hours=9, minutes=15) + pd.Timedelta(minutes=5 * k)
            for d in days
            for k in range(75)
        ]
    )
    n = len(idx)
    close = 22000 * np.exp(np.cumsum(rng.normal(0, 0.0015, n)))
    open_ = close * (1 + rng.normal(0, 0.0005, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.0008, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.0008, n)))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 0.0}, index=idx
    )


DAILY_SPEC = parse("buy nifty when rsi crosses below 40, target 3%, stop 2%").spec


def _intraday_spec() -> StrategySpec:
    spec = parse("buy nifty when rsi crosses below 40, target 0.4%, stop 0.3%").spec
    return spec.model_copy(
        update={"instrument": Instrument(symbol="NIFTY", trade_as="index", timeframe="5m")}
    )


def _start(store: Store, spec: StrategySpec, started_at: pd.Timestamp) -> dict:
    sid = store.save_strategy(spec, describe(spec))
    run_id = store.start_run(
        sid, "paper", {"capital": CAPITAL, "started_at": started_at.isoformat()}
    )
    return store.get_run(run_id)


def _replay(store, run, bars, timeframe, *, from_index: int, extra=None):
    """Feed candles one at a time: each pass happens one minute after a candle finishes."""
    fetch = (lambda s, tf, a, b: bars) if extra is None else extra
    offset = runner.close_offset(timeframe)
    for ts in bars.index[from_index:]:
        runner.step(
            store, run, fetch=fetch, charge_fn=_zero, now=ts + offset + pd.Timedelta(minutes=1)
        )


def _events(store, run_id, kind):
    return [e for e in store.paper_events(run_id) if e["kind"] == kind]


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "paper.sqlite")


@dataclass
class Replay:
    store: Store
    run: dict
    spec: StrategySpec
    bars: pd.DataFrame
    timeframe: str
    started_at: pd.Timestamp

    def copy(self, tmp_path: Path) -> Replay:
        """A private copy of the database, for a test that adds more passes."""
        for suffix in ("", "-wal", "-shm"):
            src = Path(str(self.store.path) + suffix)
            if src.exists():
                shutil.copy(src, tmp_path / f"copy.sqlite{suffix}")
        return Replay(
            Store(tmp_path / "copy.sqlite"),
            self.run,
            self.spec,
            self.bars,
            self.timeframe,
            self.started_at,
        )


def _replayed(tmp_path_factory, name, spec, bars, timeframe, start_at) -> Replay:
    store = Store(tmp_path_factory.mktemp(name) / "paper.sqlite")
    started_at = bars.index[start_at] - pd.Timedelta(minutes=1)
    run = _start(store, spec, started_at)
    _replay(store, run, bars, timeframe, from_index=start_at)
    return Replay(store, run, spec, bars, timeframe, started_at)


# Each replay is run once and shared: every pass runs the whole engine, so
# replaying hundreds of candles per test would make the suite crawl.
@pytest.fixture(scope="module")
def daily(tmp_path_factory) -> Replay:
    return _replayed(tmp_path_factory, "daily", DAILY_SPEC, _daily_bars(), "1d", 300)


@pytest.fixture(scope="module")
def intraday(tmp_path_factory) -> Replay:
    return _replayed(tmp_path_factory, "intraday", _intraday_spec(), _intraday_bars(), "5m", 150)


# ------------------------------------------------------- the central property


@pytest.mark.parametrize("which", ["daily", "intraday"])
def test_replayed_paper_trades_match_the_backtest_exactly(request, which):
    r: Replay = request.getfixturevalue(which)
    expected = run_backtest(
        r.spec,
        r.bars,
        capital=CAPITAL,
        charge_fn=_zero,
        drop_partial_last_bar=False,
        trade_from=runner.trade_from_for(r.started_at, r.timeframe),
    ).trades
    closed = [t for t in expected if t.exit_reason != "end_of_data"]
    assert len(closed) >= 3, "the scenario must actually trade for this test to mean anything"

    entries = _events(r.store, r.run["id"], "entry")
    exits = _events(r.store, r.run["id"], "exit")
    assert [(e["bar_time"], round(e["price"], 6), e["quantity"]) for e in entries] == [
        (t.entry_time.isoformat(), round(t.entry_price, 6), t.quantity) for t in expected
    ]
    assert [(e["bar_time"], round(e["price"], 6)) for e in exits] == [
        (t.exit_time.isoformat(), round(t.exit_price, 6)) for t in closed
    ]
    assert _events(r.store, r.run["id"], "drift") == []
    assert len(r.store.trades(r.run["id"])) == len(closed)


def test_intraday_positions_are_held_through_the_day_not_squared_off_each_pass(intraday):
    """Mid-session, the newest candle is not the day's last. Treating it as such
    squared off every position on the next pass -- caught by the replay above."""
    exits = _events(intraday.store, intraday.run["id"], "exit")
    reasons = {json.loads(e["detail_json"])["reason"] for e in exits}
    assert reasons - {"square_off"}, reasons


def test_nothing_is_recorded_before_the_candle_it_depends_on_had_finished(intraday):
    offset = runner.close_offset("5m")
    events = _events(intraday.store, intraday.run["id"], "entry") + _events(
        intraday.store, intraday.run["id"], "exit"
    )
    assert events
    for e in events:
        assert pd.Timestamp(e["bar_time"]) + offset <= pd.Timestamp(e["observed_at"]), e


def test_paper_starts_flat_at_the_moment_start_was_pressed(store):
    bars = _daily_bars().iloc[:400]
    # Press Start in the middle of a trade the history would be holding.
    unrestricted = run_backtest(DAILY_SPEC, bars, capital=CAPITAL, charge_fn=_zero)
    held = next(
        t for t in unrestricted.trades if t.bars_held >= 3 and t.entry_time > bars.index[250]
    )
    k = bars.index.get_loc(held.entry_time) + 1
    started_at = bars.index[k] + pd.Timedelta(hours=11)  # mid-session, position open
    run = _start(store, DAILY_SPEC, started_at)
    _replay(store, run, bars, "1d", from_index=k)

    entries = _events(store, run["id"], "entry")
    assert entries
    assert held.entry_time.isoformat() not in {e["bar_time"] for e in entries}
    # The earliest possible signal is day k's own close; it fills at day k+1's open.
    assert min(pd.Timestamp(e["bar_time"]) for e in entries) >= bars.index[k + 1]


def test_a_forming_candle_is_never_traded_on(store):
    bars = _intraday_bars().iloc[:401]
    spec = _intraday_spec()
    run = _start(store, spec, bars.index[200] - pd.Timedelta(minutes=1))
    _replay(store, run, bars.iloc[:400], "5m", from_index=380)
    before = store.paper_events(run["id"])

    # A candle that started two minutes ago, with an absurd crash in it. A
    # runner that looked at it would see the stop hit, or a fresh RSI signal.
    forming = bars.iloc[[400]].copy()
    forming[["open", "high", "low", "close"]] = [100.0, 100.0, 1.0, 1.0]
    with_forming = pd.concat([bars.iloc[:400], forming])
    runner.step(
        store,
        run,
        fetch=lambda s, tf, a, b: with_forming,
        charge_fn=_zero,
        now=bars.index[400] + pd.Timedelta(minutes=2),
    )
    assert store.paper_events(run["id"]) == before
    assert before, "the run must have something recorded for this test to mean anything"


def test_repeating_a_pass_records_nothing_new(daily, tmp_path):
    r = daily.copy(tmp_path)
    before = r.store.paper_events(r.run["id"])
    now = r.bars.index[-1] + runner.close_offset("1d") + pd.Timedelta(minutes=5)
    result = runner.step(r.store, r.run, fetch=lambda *a: r.bars, charge_fn=_zero, now=now)
    assert result.new_events == []
    assert r.store.paper_events(r.run["id"]) == before


# -------------------------------------------------------- the honest record


def test_revised_history_is_recorded_as_drift_and_the_original_stands(daily, tmp_path):
    r = daily.copy(tmp_path)
    first_entry = _events(r.store, r.run["id"], "entry")[0]

    revised = r.bars.copy()
    fill_ts = pd.Timestamp(first_entry["bar_time"])
    revised.loc[fill_ts, "open"] = revised.loc[fill_ts, "open"] * 1.01
    revised.loc[fill_ts, "high"] = max(revised.loc[fill_ts, "high"], revised.loc[fill_ts, "open"])
    now = r.bars.index[-1] + runner.close_offset("1d") + pd.Timedelta(minutes=5)
    runner.step(r.store, r.run, fetch=lambda *a: revised, charge_fn=_zero, now=now)

    assert _events(r.store, r.run["id"], "entry")[0] == first_entry
    entry_drift = [
        json.loads(d["detail_json"])
        for d in _events(r.store, r.run["id"], "drift")
        if json.loads(d["detail_json"])["event"] == "entry"
        and json.loads(d["detail_json"])["key"] == first_entry["event_key"]
    ]
    # (Later trades may drift too -- a different fill changes the account
    # balance, and with it how many units the next trade could afford.)
    assert len(entry_drift) == 1
    assert entry_drift[0]["recorded"]["price"] == round(first_entry["price"], 4)
    assert entry_drift[0]["now"]["price"] != entry_drift[0]["recorded"]["price"]


def test_the_record_cannot_be_edited_or_deleted(daily, tmp_path):
    r = daily.copy(tmp_path)
    assert r.store.paper_events(r.run["id"])
    conn = sqlite3.connect(r.store.path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE paper_event SET price = 1")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM paper_event")
    conn.close()


# --------------------------------------------------- feed problems, switches


@pytest.mark.parametrize(
    ("error", "health"),
    [
        (kite.KiteTokenExpired("x"), "token_expired"),
        (kite.KiteNotConnected("x"), "not_connected"),
        (kite.KiteLoginFailed("x"), "login_failed"),
        (kite.KiteUnavailable("x"), "feed_down"),
    ],
)
def test_a_feed_problem_pauses_and_says_so_without_guessing(store, error, health):
    bars = _daily_bars()
    run = _start(store, DAILY_SPEC, bars.index[300] - pd.Timedelta(minutes=1))

    def broken(*a):
        raise error

    now = bars.index[310] + runner.close_offset("1d")
    result = runner.step(store, run, fetch=broken, charge_fn=_zero, now=now)
    assert result.health == health
    assert store.paper_status(run["id"])["health"] == health
    assert store.paper_events(run["id"]) == []


def test_the_paper_kill_switch_pauses_paper_runs(store):
    bars = _daily_bars()
    run = _start(store, DAILY_SPEC, bars.index[300] - pd.Timedelta(minutes=1))
    store.engage_paper_kill_switch("test")
    _replay(store, run, bars.iloc[:320], "1d", from_index=300)
    assert store.paper_events(run["id"]) == []
    assert store.paper_status(run["id"])["health"] == "paused"


def test_the_live_kill_switch_does_not_stop_paper(store):
    bars = _daily_bars()
    run = _start(store, DAILY_SPEC, bars.index[300] - pd.Timedelta(minutes=1))
    store.engage_kill_switch("test")
    _replay(store, run, bars.iloc[:340], "1d", from_index=300)
    assert _events(store, run["id"], "entry")


def test_the_two_kill_switches_are_independent(store):
    store.engage_kill_switch("live")
    assert not store.paper_kill_switch_engaged()
    store.release_kill_switch("live")
    store.engage_paper_kill_switch("paper")
    assert not store.kill_switch_engaged()
    assert store.paper_kill_switch_engaged()


# ------------------------------------------------------------- the snapshot


def test_the_snapshot_adds_up(daily):
    snap = json.loads(daily.store.paper_status(daily.run["id"])["snapshot_json"])
    realised = sum(t["net_pnl"] for t in daily.store.trades(daily.run["id"]))
    assert snap["realised_pnl"] == pytest.approx(realised)
    assert snap["equity"] == pytest.approx(
        snap["capital"] + snap["realised_pnl"] + snap["open_pnl_if_closed_now"]
    )


# ------------------------------------------------------------- scheduling


def _at(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=IST)


@pytest.mark.parametrize(
    ("timeframe", "now", "status", "due"),
    [
        ("5m", "2026-10-05 10:01", None, True),  # Monday, market open
        ("5m", "2026-10-05 08:30", None, False),  # before the open
        ("5m", "2026-10-05 16:00", None, False),  # after the close
        ("5m", "2026-10-03 11:00", None, False),  # Saturday
        ("1d", "2026-10-05 12:00", None, False),  # today's daily candle not finished
        ("1d", "2026-10-05 15:35", None, True),  # finished, not yet recorded
        (
            "1d",
            "2026-10-05 15:50",
            {
                "last_bar_time": "2026-10-05T00:00:00+05:30",
                "updated_at": "2026-10-05T10:15:00+00:00",
            },
            False,
        ),  # already have today's candle
        (
            "1d",
            "2026-10-05 15:50",
            {
                "last_bar_time": "2026-10-02T00:00:00+05:30",
                "updated_at": "2026-10-05T10:15:00+00:00",
            },
            False,
        ),  # checked 5 minutes ago, wait a bit
        (
            "1d",
            "2026-10-05 16:00",
            {
                "last_bar_time": "2026-10-02T00:00:00+05:30",
                "updated_at": "2026-10-05T10:15:00+00:00",
            },
            True,
        ),
    ],
)
def test_is_due(timeframe, now, status, due):
    spec = DAILY_SPEC.model_copy(
        update={"instrument": Instrument(symbol="NIFTY", trade_as="index", timeframe=timeframe)}
    )
    assert runner.is_due({}, status, spec, _at(now)) is due


# ------------------------------------------------------- the dashboard's logic


def _saved_with_backtest(store) -> int:
    from app import logic
    from nlt.translate.rules import TranslationResult

    bt = run_backtest(DAILY_SPEC, _daily_bars(), capital=CAPITAL, charge_fn=_zero)
    result = logic.PipelineResult(
        translation=TranslationResult(spec=DAILY_SPEC), spec=DAILY_SPEC, backtest=bt
    )
    return logic.save_strategy(
        store, DAILY_SPEC, result, capital=CAPITAL, cost_model_label="Index futures"
    )


def test_saving_records_the_backtest_and_paper_then_earns_proven(store):
    """Nothing used to record a backtest, so no strategy could ever be Proven."""
    from app import logic

    sid = _saved_with_backtest(store)
    assert [r["mode"] for r in store.list_runs(sid)] == ["backtest"]
    assert store.get_strategy(sid)["state"] == "backtested"
    assert not store.is_proven(sid)

    run_id = logic.start_paper(store, sid, capital=CAPITAL, cost_model_label="Index futures")
    assert store.get_strategy(sid)["state"] == "paper"
    assert not store.is_proven(sid)  # a paper run still going does not count
    logic.stop_paper(store, run_id)
    assert store.is_proven(sid)


def test_a_strategy_cannot_be_paper_traded_twice_at_once(store):
    from app import logic

    sid = _saved_with_backtest(store)
    logic.start_paper(store, sid, capital=CAPITAL, cost_model_label="Index futures")
    with pytest.raises(ValueError, match="already"):
        logic.start_paper(store, sid, capital=CAPITAL, cost_model_label="Index futures")


@pytest.mark.parametrize(
    ("instrument", "refused"),
    [
        (Instrument(symbol="NIFTY", trade_as="index", timeframe="5m"), False),
        (Instrument(symbol="NIFTY 50", trade_as="stock", timeframe="1d"), False),
        (Instrument(symbol="NIFTY 50", trade_as="stock", timeframe="5m"), True),
    ],
)
def test_what_can_be_paper_traded(instrument, refused):
    from app import logic

    spec = DAILY_SPEC.model_copy(update={"instrument": instrument})
    assert (logic.paper_problem(spec) is not None) is refused


def _event(**kw):
    base = {
        "observed_at": "2026-10-05T04:51:00+00:00",
        "symbol": "NIFTY",
        "bar_time": "2026-10-05T10:15:00+05:30",
        "direction": "long",
        "quantity": 2.0,
        "price": 24500.5,
        "detail_json": "{}",
    }
    return {**base, **kw}


def test_the_activity_log_says_what_happened_in_plain_english():
    from app import logic

    entry = logic.describe_paper_event(_event(kind="entry"))
    assert entry.startswith("Bought 2 NIFTY at Rs 24,500")
    assert "05 Oct 10:15" in entry

    stop = logic.describe_paper_event(
        _event(kind="exit", detail_json=json.dumps({"reason": "stop", "net_pnl": -812.4}))
    )
    assert "stop loss hit" in stop and "-Rs 812" in stop

    target = logic.describe_paper_event(
        _event(kind="exit", detail_json=json.dumps({"reason": "target", "net_pnl": 950.0}))
    )
    assert "target reached" in target and "+Rs 950" in target

    short = logic.describe_paper_event(_event(kind="entry", direction="short"))
    assert short.startswith("Sold short")

    drift = logic.describe_paper_event(
        _event(
            kind="drift",
            detail_json=json.dumps(
                {"event": "entry", "key": "k", "recorded": {"price": 1}, "now": {"price": 2}}
            ),
        )
    )
    assert "original record stands" in drift
