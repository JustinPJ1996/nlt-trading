"""Paper trading options (Phase 2 batch 3), replayed on real past prices.

The central property is the one `test_paper.py` holds stocks to: feed past
candles to the paper runner one at a time, as if each were arriving live, and
it must record exactly the trades a backtest finds on the same candles.

Here the "live" option prices come from a pretend Kite (`_PastKite`) built from
the historical F&O dataset, which behaves like the real one where it matters:

  * it lists only the options live "today", with tokens that mean nothing later;
  * it forgets a contract's prices once the contract has expired;
  * it returns the minute that is still forming, which must never be used.

No test here touches Kite or the network.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from nlt.costs.slippage import Slippage
from nlt.engine.legs_backtest import run_legs_backtest
from nlt.engine.loader import load_option_bars
from nlt.engine.options_backtest import run_option_backtest
from nlt.options.data import ParquetSource
from nlt.options.execution import OptionExecution
from nlt.options.live import LiveOptionSource, _shape
from nlt.paper import runner
from nlt.spec.models import (
    AtTime,
    ExitRules,
    Instrument,
    OptionLeg,
    RiskLimits,
    Schedule,
    Sizing,
    StrategySpec,
)
from nlt.store.db import Store
from nlt.translate.readback import describe
from nlt.translate.rules import parse

IST = "Asia/Kolkata"
CAPITAL = 2_500_000.0
_REAL_1M = Path(__file__).resolve().parents[1] / "data/fno/1m/underlying=NIFTY"
needs_data = pytest.mark.skipif(
    not _REAL_1M.exists(), reason="the F&O 1-minute data is not on this machine"
)


def _zero(price, qty, side):
    return 0.0


class _PastKite:
    """A pretend Kite whose clock is `now`, answering from the dataset."""

    def __init__(self) -> None:
        self.real = ParquetSource()
        self.now: pd.Timestamp | None = None
        self._tokens: dict[int, tuple[str, str, dt.date]] = {}
        self._minutes: dict[str, pd.DataFrame] = {}
        self.requests = 0

    def instruments(self, now):
        frames = []
        for underlying in ("NIFTY",):
            rows = self.real.listing(underlying, now.date()).copy()
            # Tokens are only meaningful today: a fresh numbering each day.
            rows["instrument_token"] = [
                hash((r, now.date())) % 10**9 for r in rows["tradingsymbol"]
            ]
            for token, sym, exp in zip(
                rows["instrument_token"], rows["tradingsymbol"], rows["expiry"], strict=True
            ):
                self._tokens[int(token)] = (underlying, sym, exp)
            frames.append(rows.assign(name=underlying, lot_size=65))
        return pd.concat(frames, ignore_index=True)

    def minute_candles(self, token, start, end):
        self.requests += 1
        underlying, sym, expiry = self._tokens[int(token)]
        if expiry < self.now.date():
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        if sym not in self._minutes:
            raw = self.real.contract_bars(
                underlying, sym, "1m", expiry - dt.timedelta(days=40), expiry
            )
            raw = raw.copy()
            # Kite marks no "bad" rows; give an untraded or bad minute no volume.
            raw.loc[~raw["traded"], "volume"] = 0
            self._minutes[sym] = raw[["open", "high", "low", "close", "volume"]]
        rows = self._minutes[sym]
        days = rows.index.date
        rows = rows[(days >= start) & (days <= end)]
        # Kite hands back the minute still forming, too.
        return rows[rows.index <= self.now]

    def index_bars(self, symbol, timeframe, start, end):
        return self.real.underlying_bars(symbol, timeframe, start, end)


@dataclass
class OptionReplay:
    store: Store
    run: dict
    spec: StrategySpec
    bars: pd.DataFrame
    trade_from: pd.Timestamp
    kite: _PastKite
    root: Path


def _replay(tmp_path, spec, first_day, last_day, *, evenings=True, margin_fn=None, until=None):
    kite = _PastKite()
    root = tmp_path / "fno_live"
    store = Store(tmp_path / "paper.sqlite")
    start = pd.Timestamp(dt.datetime.combine(first_day, dt.time(9, 14)), tz=IST)
    all_bars = load_option_bars(spec, first_day - dt.timedelta(days=30), last_day, ParquetSource())
    sid = store.save_strategy(spec, describe(spec))
    run_id = store.start_run(
        sid,
        "paper",
        {"capital": CAPITAL, "started_at": start.isoformat(), "slippage": Slippage().to_dict()},
    )
    run = store.get_run(run_id)

    def source(now):
        kite.now = now
        return LiveOptionSource(kite, now, root=root)

    feed = runner.OptionFeed(source=source, margin_fn=margin_fn)
    offset = runner.close_offset(spec.instrument.timeframe)
    times = [ts + offset + pd.Timedelta(minutes=1) for ts in all_bars.loc[start:].index]
    if evenings:
        days = sorted({t.date() for t in times})
        times += [pd.Timestamp(dt.datetime.combine(d, dt.time(15, 46)), tz=IST) for d in days]
        times.sort()
    if until is not None:
        times = [t for t in times if t <= until]
    for now in times:
        runner.step(
            store,
            run,
            fetch=lambda s, tf, a, b: all_bars,
            charge_fn=_zero,
            now=now,
            option_feed=feed,
        )
    trade_from = runner.trade_from_for(start, spec.instrument.timeframe, runner.NSE_FNO)
    return OptionReplay(store, run, spec, all_bars, trade_from, kite, root)


def _paper_closed(r: OptionReplay) -> list[tuple]:
    events = r.store.paper_events(r.run["id"])
    entries = {e["event_key"]: e for e in events if e["kind"] == "entry"}
    out = []
    for e in events:
        if e["kind"] != "exit":
            continue
        entry = entries[e["event_key"]]
        out.append(
            (
                pd.Timestamp(entry["bar_time"]),
                round(entry["price"], 4),
                pd.Timestamp(e["bar_time"]),
                round(e["price"], 4),
                eval_reason(e),
            )
        )
    return sorted(out)


def eval_reason(event) -> str:
    import json

    return json.loads(event["detail_json"])["reason"]


def _backtest_closed(trades) -> list[tuple]:
    return sorted(
        (
            pd.Timestamp(t.entry_time),
            round(t.entry_price, 4),
            pd.Timestamp(t.exit_time),
            round(t.exit_price, 4),
            t.exit_reason,
        )
        for t in trades
        if t.exit_reason != "end_of_data"
    )


BOUGHT = parse(
    "Buy NIFTY weekly ATM call on 15 minute bars when RSI crosses above 55. "
    "Stop 20%, target 40% on the premium."
)


@dataclass(frozen=True)
class _Margin:
    total: float
    source: str = "test"
    note: str | None = None


def _fixed_margin(underlying, day, legs, prev_close):
    return _Margin(sum(-leg.quantity for leg in legs if leg.quantity < 0) * 1000.0)


STRADDLE_SPEC = StrategySpec(
    name="09:30 straddle",
    description="t",
    instrument=Instrument(
        symbol="NIFTY",
        timeframe="5m",
        trade_as="option",
        legs=[
            OptionLeg(side="sell", option_type="CE", stop_pct=25),
            OptionLeg(side="sell", option_type="PE", stop_pct=25),
        ],
    ),
    entry=AtTime(time=dt.time(9, 30)),
    exit=ExitRules(exits_on_legs=True),
    sizing=Sizing(mode="fixed_lots", lots=1),
    risk=RiskLimits(max_position_pct=20),
    schedule=Schedule(intraday=True),
)


@pytest.fixture(scope="module")
def bought(tmp_path_factory) -> OptionReplay:
    spec = BOUGHT.spec
    assert spec is not None, BOUGHT
    return _replay(
        tmp_path_factory.mktemp("bought"), spec, dt.date(2025, 1, 6), dt.date(2025, 1, 10)
    )


@pytest.fixture(scope="module")
def straddle(tmp_path_factory) -> OptionReplay:
    return _replay(
        tmp_path_factory.mktemp("straddle"),
        STRADDLE_SPEC,
        dt.date(2025, 1, 6),
        dt.date(2025, 1, 7),
        margin_fn=_fixed_margin,
    )


@needs_data
def test_replayed_bought_option_paper_trades_match_the_backtest(bought):
    r = bought
    execution = OptionExecution(
        ParquetSource(),
        "NIFTY",
        r.spec.instrument.timeframe,
        r.spec.instrument.option_type,
        r.spec.instrument.strike_offset,
        r.spec.instrument.expiry,
    )
    backtest = run_option_backtest(
        r.spec,
        r.bars,
        execution,
        capital=CAPITAL,
        charge_fn=_zero,
        drop_partial_last_bar=False,
        trade_from=r.trade_from,
    )
    expected = _backtest_closed(backtest.trades)
    assert len(expected) >= 2, "the week must have trades for this to mean anything"
    assert _paper_closed(r) == expected
    # Each signal was announced on the candle it happened, as for stocks.
    assert [e for e in r.store.paper_events(r.run["id"]) if e["kind"] == "signal"]
    # Nothing ever drifted: no pass saw a different answer from an earlier one.
    assert not [e for e in r.store.paper_events(r.run["id"]) if e["kind"] == "drift"]


@needs_data
def test_replayed_straddle_paper_trades_match_the_backtest(straddle):
    r = straddle
    execution = OptionExecution(ParquetSource(), "NIFTY", "5m")
    backtest = run_legs_backtest(
        r.spec,
        r.bars,
        execution,
        _fixed_margin,
        capital=CAPITAL,
        charge_fn=_zero,
        drop_partial_last_bar=False,
        trade_from=r.trade_from,
    )
    expected = _backtest_closed(backtest.trades)
    assert len(expected) == 2  # one straddle a day, Monday and Tuesday
    assert _paper_closed(r) == expected
    assert not [e for e in r.store.paper_events(r.run["id"]) if e["kind"] == "drift"]


@needs_data
def test_an_option_position_is_held_through_the_day_not_squared_off_each_pass(bought):
    """The bug the stock engine once had: mid-session, the newest candle was
    taken for the day's last, and every position squared off on each pass."""
    for e in bought.store.paper_events(bought.run["id"]):
        if e["kind"] == "exit" and eval_reason(e) == "square_off":
            assert pd.Timestamp(e["bar_time"]).time() >= dt.time(15, 15)


@needs_data
def test_an_expired_contracts_prices_are_kept_after_kite_forgets_them(bought):
    """9 Jan 2025 was a weekly expiry. Kite would no longer have that week's
    contracts on the 10th; the replay on the 10th still re-derived every trade
    (no drift), which it could only do from the saved copies."""
    saved = list((bought.root / "bars" / "NIFTY").glob("NIFTY25109*/2025-01-09.parquet"))
    assert saved, "the expiring contracts' last day must have been saved"
    status = bought.store.paper_status(bought.run["id"])
    assert "not saved" not in status["message"]


@needs_data
def test_a_lost_expiry_day_is_said_plainly_never_guessed(tmp_path):
    """No pass after the close on an expiry day: that day's prices are gone
    from Kite the next morning, and the run must say so."""
    spec = BOUGHT.spec
    r = _replay(tmp_path, spec, dt.date(2025, 1, 8), dt.date(2025, 1, 10), evenings=False)
    trades_on_expiring = [
        e
        for e in r.store.paper_events(r.run["id"])
        if e["kind"] == "entry" and e["symbol"].startswith("NIFTY25109")
    ]
    if not trades_on_expiring:
        pytest.skip("no trade in the expiring weekly in this window")
    status = r.store.paper_status(r.run["id"])
    assert "were not saved before Kite dropped them" in status["message"]


@needs_data
def test_the_forming_minute_is_never_used(tmp_path):
    kite = _PastKite()
    now = pd.Timestamp("2025-01-08 10:20:30", tz=IST)
    kite.now = now
    source = LiveOptionSource(kite, now, root=tmp_path)
    listing = source.listing("NIFTY", now.date())
    sym = listing[listing["expiry"] == dt.date(2025, 1, 9)]["tradingsymbol"].iloc[0]
    bars = source.contract_bars("NIFTY", sym, "1m", now.date(), now.date())
    assert bars.index.max() <= pd.Timestamp("2025-01-08 10:19", tz=IST)
    # ... and an unfinished day is not saved, so a later pass sees all of it.
    assert not list(tmp_path.glob("bars/**/2025-01-08.parquet"))


def test_kite_minutes_with_no_volume_are_not_trades():
    idx = pd.date_range("2026-10-15 09:15", periods=3, freq="min", tz=IST)
    raw = pd.DataFrame(
        {
            "open": [10.0, 10.0, 11.0],
            "high": [10.5, 10.0, 11.5],
            "low": [9.5, 10.0, 0.0],
            "close": [10.0, 10.0, 11.0],
            "volume": [100.0, 0.0, 50.0],
        },
        index=idx,
    )
    shaped = _shape(raw)
    # Traded; carried forward with no volume; a zero low (impossible price).
    assert shaped["traded"].tolist() == [True, False, False]


@needs_data
def test_paper_freezes_an_expiring_option_at_its_last_traded_price(tmp_path):
    """Justin (2026-10-15): in paper, an option still open on expiry day is held
    to the close and valued at its last trade -- worked out here straight from
    the raw minute rows, not through the engine."""
    spec = BOUGHT.spec.model_copy(
        update={
            "schedule": BOUGHT.spec.schedule.model_copy(update={"intraday": False}),
            "exit": BOUGHT.spec.exit.model_copy(
                update={"stop_pct": None, "target_pct": None, "max_bars_held": 1000}
            ),
            # A put: the call had decayed to 0.05 by 15:15, where both rules agree.
            "instrument": BOUGHT.spec.instrument.model_copy(update={"option_type": "PE"}),
        }
    )
    r = _replay(tmp_path, spec, dt.date(2025, 1, 8), dt.date(2025, 1, 9))
    expired = [
        e
        for e in r.store.paper_events(r.run["id"])
        if e["kind"] == "exit" and eval_reason(e) == "expiry"
    ]
    assert expired, "a position must have been held into the 9 Jan expiry"
    for e in expired:
        raw = ParquetSource().contract_bars(
            "NIFTY", e["symbol"], "1m", dt.date(2025, 1, 9), dt.date(2025, 1, 9)
        )
        traded = raw[raw["traded"]]
        last = traded["close"].iloc[-1]
        assert e["price"] == pytest.approx(last)
        # ... which is not what the backtest's 15:15 rule would have given.
        at_1515 = traded[traded.index.time >= dt.time(15, 15)]["open"].iloc[0]
        assert at_1515 != pytest.approx(last)


@needs_data
def test_an_open_straddle_shows_its_profit_so_far_on_its_timeline(tmp_path):
    """Mid-morning the straddle is open; the trade's timeline must find it in
    the latest snapshot by the record's own key (the index, not its contracts)."""
    import json

    from app.api import _paper_trade

    r = _replay(
        tmp_path,
        STRADDLE_SPEC,
        dt.date(2025, 1, 6),
        dt.date(2025, 1, 6),
        margin_fn=_fixed_margin,
        until=pd.Timestamp("2025-01-06 11:01", tz=IST),
    )
    (entry,) = [e for e in r.store.paper_events(r.run["id"]) if e["kind"] == "entry"]
    snap = json.loads(r.store.paper_status(r.run["id"])["snapshot_json"])
    (pos,) = snap["open_positions"]
    assert pos["key"] == entry["event_key"]
    steps = _paper_trade(r.store, r.run["id"], entry["event_key"])
    assert any("Open profit or loss" in step for step in steps), steps
