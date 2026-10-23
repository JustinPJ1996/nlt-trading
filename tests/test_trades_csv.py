"""The trades download for multi-leg option positions: one row per position,
then one row per option inside it, built from a real legs backtest."""

from __future__ import annotations

import csv
import io
from types import SimpleNamespace

import pytest

from app import logic
from tests.test_legs_engine import (
    CE,
    D1,
    OPEN_PRICES,
    PE,
    PE_FLAT,
    _run,
    _source,
    _spec,
    _ts,
    _with,
    buy,
    sell,
)


def _export(spec, source):
    bt = _run(spec, source, days=(D1,))
    assert bt.trades
    text = logic.trades_csv(SimpleNamespace(backtest=SimpleNamespace(spec=spec, trades=bt.trades)))  # ty: ignore[invalid-argument-type]  # duck-typed stand-in for PipelineResult
    return bt, list(csv.DictReader(io.StringIO(text)))


def _stopped_ce_source():
    return _source(
        _with(OPEN_PRICES, {CE: {_ts(D1, 10, 0): (110.0, 127.0, 108.0, 120.0)}}), PE_FLAT
    )


def test_leg_rows_sit_under_their_position_and_gross_adds_up_to_the_paisa():
    bt, rows = _export(_spec(), _stopped_ce_source())
    assert [r["row_type"] for r in rows] == ["position", "leg", "leg"]
    assert {r["position"] for r in rows} == {"1"}
    pos, legs = rows[0], rows[1:]
    assert pos["leg"] == "" and pos["side"] == ""
    assert {r["symbol"] for r in legs} == {CE, PE}
    assert sum(float(r["leg_profit_before_charges"]) for r in legs) == pytest.approx(
        float(pos["profit_before_charges"]), abs=0.005
    )
    assert sum(float(r["leg_charges"]) for r in legs) == pytest.approx(
        float(pos["charges"]), abs=0.005
    )
    # Leg rows leave the position money columns blank so a column sum counts once.
    assert all(r["profit_after_charges"] == "" for r in legs)
    ce = next(r for r in legs if r["symbol"] == CE)
    assert ce["why_it_closed"] == "stop loss hit"
    assert ce["exited"].endswith("10:00")


def test_each_leg_row_shows_its_true_side_even_when_the_position_reads_buy():
    legs = [buy("CE"), sell("PE", stop_pct=25)]
    bt, rows = _export(_spec(legs), _source(OPEN_PRICES, PE_FLAT))
    pos, *leg_rows = rows
    by_symbol = {r["symbol"]: r["side"] for r in leg_rows}
    assert by_symbol == {CE: "buy", PE: "sell"}
    # The position row keeps today's wording (a net debit reads "buy").
    assert pos["direction"] == ("buy" if bt.trades[0].direction == "long" else "short")


def test_a_reentered_leg_is_flagged():
    legs = [
        sell("CE", stop_pct=25, reentry="same_strike_at_cost", reentry_max=1),
        sell("PE", stop_pct=25),
    ]
    src = _source(
        _with(
            OPEN_PRICES,
            {
                CE: {
                    _ts(D1, 10, 0): (110.0, 130.0, 100.0, 100.0),
                    _ts(D1, 10, 30): (101.0, 101.0, 101.0, 101.0),
                    _ts(D1, 10, 35): (100.0, 100.0, 100.0, 100.0),
                    _ts(D1, 11, 0): (120.0, 126.0, 120.0, 125.0),
                }
            },
        ),
        {**PE_FLAT, CE: (110.0, 110.0, 110.0, 110.0)},
    )
    _bt, rows = _export(_spec(legs), src)
    ce_rows = [r for r in rows if r["row_type"] == "leg" and r["symbol"] == CE]
    assert [r["re_entry"] for r in ce_rows] == ["", "re-entry 1"]
    assert sum(1 for r in rows if r["row_type"] == "position") == 1
    pos = rows[0]
    assert sum(float(r["leg_profit_before_charges"]) for r in rows[1:]) == pytest.approx(
        float(pos["profit_before_charges"]), abs=0.005
    )


def test_every_leg_of_the_position_is_listed():
    bt, rows = _export(_spec(), _stopped_ce_source())
    assert len([r for r in rows if r["row_type"] == "leg"]) == len(bt.trades[0].legs) == 2
