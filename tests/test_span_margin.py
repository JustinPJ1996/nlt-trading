"""SPAN margin for sold and multi-leg index options (`nlt.options.margin`).

The answer key is Zerodha's public F&O calculator. On 14 Oct 2026 23 positions
were entered into it (`scripts/calibrate_span_zerodha.py`) and its figures
recorded in `tests/fixtures/zerodha_span_calibration.json`, alongside a cut-down
copy of that morning's NSE and BSE start-of-day files. The first test replays
every one of them offline.

The rest pin down each rule by date on small hand-made files, so each can be
broken on purpose and seen to fail.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from nlt.options.margin import (
    FixedMargin,
    MarginLeg,
    MarginUnavailable,
    SpanMargin,
)
from nlt.options.span_store import (
    KIND_CALL,
    KIND_PUT,
    CalendarSpread,
    IndexParams,
    SpanDay,
    load_day,
    parse_span_xml,
    save_day,
)

FIXTURES = Path(__file__).parent / "fixtures"
CALIBRATION = json.loads((FIXTURES / "zerodha_span_calibration.json").read_text())


# ------------------------------------------------------------ the answer key


def _legs(position: dict) -> list[MarginLeg]:
    return [
        MarginLeg(
            leg["option_type"],
            leg["strike"],
            dt.date.fromisoformat(leg["expiry"]),
            leg["quantity"],
        )
        for leg in position["legs"]
    ]


def _close(underlying: str) -> float:
    for file in CALIBRATION["files"].values():
        if underlying in file["index_close"]:
            return file["index_close"][underlying]
    raise KeyError(underlying)


def _zerodha_index_price(underlying: str) -> float:
    """The index level Zerodha's calculator multiplies the exposure rate by.

    It is NOT the previous close: on 14 Oct 2026 every NIFTY short leg was
    charged 2% (or 3%) of 23,997.35 per unit, BANKNIFTY of 55,703.92 and
    SENSEX of 84,559.65 -- while the exchanges' own files (and the market)
    had NIFTY at 22,721.95, BANKNIFTY 55,481.05, SENSEX 73,033.78. Zerodha's
    which-rate decision (deep out of the money or not) does use the real close
    -- a NIFTY 25000 call, 10.03% above 22,721.95 but only 4.2% above 23,997,
    got 3%. So we follow NSE's rule (previous close) for both and derive
    Zerodha's stale level from its plain single-leg short, then check every
    other position against it.
    """
    single = {
        "NIFTY": "NIFTY short ATM call, weekly",
        "BANKNIFTY": "BANKNIFTY short ATM put, monthly",
        "SENSEX": "SENSEX short ATM call, next week",
    }[underlying]
    pos = next(p for p in CALIBRATION["positions"] if p["name"] == single)
    units = -pos["legs"][0]["quantity"]
    return pos["zerodha"]["exposure"] / (0.02 * units)


@pytest.mark.parametrize("position", CALIBRATION["positions"], ids=lambda p: p["name"])
def test_matches_zerodha_on_the_calibration_day(position):
    margin = SpanMargin(root=FIXTURES / "span")
    day = dt.date.fromisoformat(CALIBRATION["captured_on"])
    und = position["underlying"]
    close = _close(und)
    ours = margin.margin(und, day, _legs(position), index_prev_close=close)
    z = position["zerodha"]

    # SPAN: to the rupee (Zerodha's own rounding leaves at most ~Rs 0.41).
    assert ours.span == pytest.approx(z["span"], abs=1.0)
    # Exposure: same rates on the same legs, at Zerodha's index level.
    assert ours.exposure * _zerodha_index_price(und) / close == pytest.approx(
        z["exposure"], abs=1.0
    )
    assert ours.total == pytest.approx(ours.span + ours.exposure)
    assert ours.note is None
    assert "start of day" in ours.source and "14 Oct 2026" in ours.source


def test_the_calibration_covers_what_it_claims():
    """The fixture exercises the cases the brief asked for, not just singles."""
    names = [p["name"] for p in CALIBRATION["positions"]]
    assert len(names) >= 20
    for word in (
        "straddle",
        "strangle",
        "bull put",
        "bear call",
        "iron condor",
        "iron butterfly",
        "long call only",
        "2 lots",
        "calendar",
        ">10% OTM",
    ):
        assert any(word in n for n in names), word
    assert {p["underlying"] for p in CALIBRATION["positions"]} == {"NIFTY", "BANKNIFTY", "SENSEX"}


def _calibrated(name: str):
    pos = next(p for p in CALIBRATION["positions"] if p["name"] == name)
    margin = SpanMargin(root=FIXTURES / "span")
    day = dt.date.fromisoformat(CALIBRATION["captured_on"])
    return margin.margin(
        pos["underlying"], day, _legs(pos), index_prev_close=_close(pos["underlying"])
    )


def test_a_hedged_spread_is_far_cheaper_than_the_naked_short():
    naked = _calibrated("NIFTY short ATM put, 2 lots, weekly").span / 2
    spread = _calibrated("NIFTY bull put spread, weekly").span
    assert spread < naked / 5
    condor = _calibrated("NIFTY iron condor, weekly")
    strangle = _calibrated("NIFTY short strangle, weekly")
    assert condor.span < strangle.span / 4


def test_bought_legs_pay_no_exposure_but_hedged_short_legs_still_do():
    long_only = _calibrated("NIFTY long call only, weekly")
    assert long_only.span == 0 and long_only.exposure == 0 and long_only.total == 0
    # Zerodha charges exposure on every sold leg of an iron condor, hedged or not:
    # two sold legs, two charges; the two bought wings add nothing.
    condor = _calibrated("NIFTY iron condor, weekly")
    single = _calibrated("NIFTY short ATM call, weekly")
    assert condor.exposure == pytest.approx(2 * single.exposure)


def test_the_calendar_spread_charge_is_what_zerodha_charges():
    # Sell this week's call, buy the monthly: the scenarios almost cancel and
    # nearly all of the SPAN figure is the exchange's calendar-spread charge.
    cal = _calibrated("NIFTY calendar: sell weekly call, buy monthly call")
    assert 13_000 < cal.span < 14_000


# ------------------------------------------------- hand-made exchange files

NIFTY_CLOSE = 20_000.0
# A long call's loss per unit in each of the 16 scenarios (loss > 0).
CALL_RISK = np.linspace(-200.0, 300.0, 16)


def _day(
    day: dt.date,
    contracts: list[tuple[int, dt.date, float, float]],
    *,
    spreads: list[CalendarSpread] | None = None,
    som_rate: float = 0.0,
    exchange: str = "NSE",
    underlying: str = "NIFTY",
    name: str | None = None,
) -> SpanDay:
    """contracts: (kind, expiry, strike, price); every one gets CALL_RISK and delta 0.5."""
    n = len(contracts)
    return SpanDay(
        exchange=exchange,
        file_date=day,
        created=f"{day:%Y%m%d}0000",
        is_settlement=False,
        source_name=name or f"test.{day:%Y%m%d}",
        params={underlying: IndexParams(underlying, NIFTY_CLOSE, 1860.0, som_rate, spreads or [])},
        underlying=np.array([underlying] * n, dtype="U9"),
        kind=np.array([c[0] for c in contracts], dtype=np.int8),
        expiry=np.array([int(c[1].strftime("%Y%m%d")) for c in contracts], dtype=np.int32),
        strike=np.array([c[2] for c in contracts], dtype=np.float64),
        price=np.array([c[3] for c in contracts], dtype=np.float64),
        delta=np.full(n, 0.5),
        risk=np.tile(CALL_RISK, (n, 1)),
    )


def _store(root: Path, day: SpanDay, tag: str = "i1") -> None:
    ex = day.exchange
    save_day(day, root / ex / f"{day.file_date.year}" / f"{day.file_date:%Y%m%d}-{tag}.npz")


def _sell(strike, expiry, units=65, kind="CE"):
    return MarginLeg(kind, strike, expiry, -units)


# ------------------------------------------------------------------ exposure


@pytest.mark.parametrize(
    "day, expiry, extra",
    [
        # The extra 2% starts on 20 Nov 2024 and only on the leg's own expiry day.
        (dt.date(2024, 11, 14), dt.date(2024, 11, 14), 0.0),  # an expiry day before the rule
        (dt.date(2024, 11, 19), dt.date(2024, 11, 19), 0.0),
        (dt.date(2024, 11, 20), dt.date(2024, 11, 20), 0.02),  # the rule's first day
        (dt.date(2024, 11, 21), dt.date(2024, 11, 21), 0.02),  # first NIFTY expiry under it
        (dt.date(2025, 6, 26), dt.date(2025, 6, 26), 0.02),
        (dt.date(2024, 11, 20), dt.date(2024, 11, 21), 0.0),  # rule live, not expiry day
        (dt.date(2025, 6, 25), dt.date(2025, 6, 26), 0.0),
    ],
)
def test_expiry_day_extra_exposure_by_date(tmp_path, day, expiry, extra):
    _store(tmp_path, _day(day, [(KIND_CALL, expiry, 20_000, 100.0)]))
    got = SpanMargin(tmp_path).margin(
        "NIFTY", day, [_sell(20_000, expiry)], index_prev_close=NIFTY_CLOSE
    )
    notional = NIFTY_CLOSE * 65
    assert got.expiry_day_extra == pytest.approx(extra * notional)
    assert got.exposure == pytest.approx((0.02 + extra) * notional)


def test_expiry_day_extra_applies_on_top_of_the_deep_otm_rate(tmp_path):
    # NSE Clearing 125/2024's own example: deep out of the money on expiry day = 3% + 2%.
    day = dt.date(2025, 6, 26)
    _store(tmp_path, _day(day, [(KIND_CALL, day, 22_100, 1.0), (KIND_PUT, day, 17_900, 1.0)]))
    m = SpanMargin(tmp_path)
    for leg in (_sell(22_100, day), _sell(17_900, day, kind="PE")):
        got = m.margin("NIFTY", day, [leg], index_prev_close=NIFTY_CLOSE)
        assert got.exposure == pytest.approx(0.05 * NIFTY_CLOSE * 65)


@pytest.mark.parametrize(
    "strike, kind, rate",
    [
        (22_000, KIND_CALL, 0.02),  # exactly 10% out: not "more than 10%"
        (22_100, KIND_CALL, 0.03),
        (17_900, KIND_PUT, 0.03),
        (18_000, KIND_PUT, 0.02),
        (17_900, KIND_CALL, 0.02),  # deep IN the money is not deep out of it
        (22_100, KIND_PUT, 0.02),
    ],
)
def test_deep_out_of_the_money_exposure_is_3_percent(tmp_path, strike, kind, rate):
    day, expiry = dt.date(2025, 6, 20), dt.date(2025, 6, 26)
    _store(tmp_path, _day(day, [(kind, expiry, strike, 1.0)]))
    leg = _sell(strike, expiry, kind="CE" if kind == KIND_CALL else "PE")
    got = SpanMargin(tmp_path).margin("NIFTY", day, [leg], index_prev_close=NIFTY_CLOSE)
    assert got.exposure == pytest.approx(rate * NIFTY_CLOSE * 65)


@pytest.mark.parametrize(
    "expiry, rate",
    [(dt.date(2026, 3, 20), 0.02), (dt.date(2026, 3, 21), 0.05), (dt.date(2026, 6, 25), 0.05)],
)
def test_more_than_nine_months_to_expiry_is_5_percent(tmp_path, expiry, rate):
    day = dt.date(2025, 6, 20)
    _store(tmp_path, _day(day, [(KIND_CALL, expiry, 20_000, 900.0)]))
    got = SpanMargin(tmp_path).margin(
        "NIFTY", day, [_sell(20_000, expiry)], index_prev_close=NIFTY_CLOSE
    )
    assert got.exposure == pytest.approx(rate * NIFTY_CLOSE * 65)


def test_exposure_is_charged_on_sold_legs_only(tmp_path):
    day, expiry = dt.date(2025, 6, 20), dt.date(2025, 6, 26)
    _store(
        tmp_path, _day(day, [(KIND_CALL, expiry, 20_000, 100.0), (KIND_CALL, expiry, 20_500, 40.0)])
    )
    m = SpanMargin(tmp_path)
    bought = m.margin(
        "NIFTY", day, [MarginLeg("CE", 20_500, expiry, 130)], index_prev_close=NIFTY_CLOSE
    )
    assert bought.exposure == 0
    spread = m.margin(
        "NIFTY",
        day,
        [_sell(20_000, expiry), MarginLeg("CE", 20_500, expiry, 65)],
        index_prev_close=NIFTY_CLOSE,
    )
    assert spread.exposure == pytest.approx(0.02 * NIFTY_CLOSE * 65)


# ---------------------------------------------------------------------- SPAN


def test_span_is_worst_scenario_plus_sold_premium_and_a_bought_leg_alone_is_zero(tmp_path):
    day, expiry = dt.date(2025, 6, 20), dt.date(2025, 6, 26)
    # The second contract is priced at its own worst loss (300), as a real
    # bought option is: it cannot lose more than was paid for it.
    _store(
        tmp_path,
        _day(day, [(KIND_CALL, expiry, 20_000, 100.0), (KIND_CALL, expiry, 20_500, 300.0)]),
    )
    m = SpanMargin(tmp_path)
    sold = m.margin("NIFTY", day, [_sell(20_000, expiry)], index_prev_close=NIFTY_CLOSE)
    # Sold: its worst scenario is CALL_RISK's biggest gain for the buyer (200)
    # plus the premium it owes (100), per unit.
    assert sold.span == pytest.approx((200 + 100) * 65)
    bought = m.margin(
        "NIFTY", day, [MarginLeg("CE", 20_500, expiry, 65)], index_prev_close=NIFTY_CLOSE
    )
    assert bought.span == 0 and bought.total == 0


def test_a_bought_leg_worth_more_than_its_worst_loss_never_makes_margin_negative(tmp_path):
    # A deep in-the-money option can be priced above its worst scenario loss;
    # the surplus is not a refund.
    day, expiry = dt.date(2025, 6, 20), dt.date(2025, 6, 26)
    _store(tmp_path, _day(day, [(KIND_CALL, expiry, 19_000, 1_000.0)]))
    got = SpanMargin(tmp_path).margin(
        "NIFTY", day, [MarginLeg("CE", 19_000, expiry, 65)], index_prev_close=NIFTY_CLOSE
    )
    assert got.span == 0 and got.total == 0


def test_two_sold_expiries_are_not_a_calendar_spread(tmp_path):
    # The calendar charge needs one expiry net bought and the other net sold.
    day, near, far = dt.date(2025, 6, 20), dt.date(2025, 6, 26), dt.date(2025, 7, 3)
    _calendar_files(tmp_path, day, near, far)
    got = SpanMargin(tmp_path).margin(
        "NIFTY", day, [_sell(20_000, near), _sell(20_000, far)], index_prev_close=NIFTY_CLOSE
    )
    assert got.span == pytest.approx(130 * 200 + 130 * 100)


def test_short_option_minimum_applies_when_the_file_sets_one(tmp_path):
    day, expiry = dt.date(2025, 6, 20), dt.date(2025, 6, 26)
    _store(tmp_path, _day(day, [(KIND_CALL, expiry, 20_000, 100.0)], som_rate=1_000.0))
    got = SpanMargin(tmp_path).margin(
        "NIFTY", day, [_sell(20_000, expiry)], index_prev_close=NIFTY_CLOSE
    )
    assert got.span == pytest.approx(1_000 * 65 + 100 * 65)


def _calendar_files(root: Path, day: dt.date, near: dt.date, far: dt.date) -> None:
    spreads = [CalendarSpread(1, "F", 400.0, int(f"{near:%Y%m%d}"), int(f"{far:%Y%m%d}"))]
    _store(
        root,
        _day(
            day,
            [(KIND_CALL, near, 20_000, 100.0), (KIND_CALL, far, 20_000, 100.0)],
            spreads=spreads,
        ),
    )


def _calendar_legs(near, far):
    return [_sell(20_000, near), MarginLeg("CE", 20_000, far, 65)]


@pytest.mark.parametrize(
    "day, near, far, separate",
    [
        # Before 10 Feb 2025 an expiry-day calendar still offsets.
        (dt.date(2025, 2, 6), dt.date(2025, 2, 6), dt.date(2025, 2, 13), False),
        # From 10 Feb 2025, on its expiry day the expiring leg stands alone.
        (dt.date(2025, 2, 13), dt.date(2025, 2, 13), dt.date(2025, 2, 20), True),
        (dt.date(2025, 6, 26), dt.date(2025, 6, 26), dt.date(2025, 7, 3), True),
        # ...but only on its expiry day.
        (dt.date(2025, 2, 12), dt.date(2025, 2, 13), dt.date(2025, 2, 20), False),
        # The last day before and the first day of the rule.
        (dt.date(2025, 2, 7), dt.date(2025, 2, 7), dt.date(2025, 2, 13), False),
        (dt.date(2025, 2, 10), dt.date(2025, 2, 10), dt.date(2025, 2, 13), True),
    ],
)
def test_expiry_day_calendar_rule_from_10_feb_2025(tmp_path, day, near, far, separate):
    _calendar_files(tmp_path, day, near, far)
    got = SpanMargin(tmp_path).margin(
        "NIFTY", day, _calendar_legs(near, far), index_prev_close=NIFTY_CLOSE
    )
    if separate:
        # Sold leg alone: 65 x (200 + 100); bought leg alone: max(0, 65 x (300 - 100)).
        assert got.span == pytest.approx(65 * 300 + 65 * 200)
    else:
        # Scenarios cancel and premiums cancel: only the calendar charge,
        # 65 units x delta 0.5 x Rs 400.
        assert got.span == pytest.approx(65 * 0.5 * 400)


# -------------------------------------------------------- which file, fallbacks


def test_uses_that_days_start_of_day_file(tmp_path):
    expiry = dt.date(2025, 6, 26)
    for d, price in ((dt.date(2025, 6, 19), 50.0), (dt.date(2025, 6, 20), 100.0)):
        _store(tmp_path, _day(d, [(KIND_CALL, expiry, 20_000, price)]))
    got = SpanMargin(tmp_path).margin(
        "NIFTY", dt.date(2025, 6, 20), [_sell(20_000, expiry)], index_prev_close=NIFTY_CLOSE
    )
    assert got.span == pytest.approx((200 + 100) * 65)
    assert got.source == "NSE's margin file for 20 Jun 2025 (start of day)"
    assert got.note is None


def test_a_missing_day_uses_the_previous_days_end_of_day_file_and_says_so(tmp_path):
    expiry = dt.date(2025, 6, 26)
    day = _day(
        dt.date(2025, 6, 17),
        [(KIND_CALL, expiry, 20_000, 70.0)],
        exchange="BSE",
        underlying="SENSEX",
        name="BSERISK20250617-00",
    )
    _store(tmp_path, day, tag="00")
    eod = _day(
        dt.date(2025, 6, 18),
        [(KIND_CALL, expiry, 20_000, 80.0)],
        exchange="BSE",
        underlying="SENSEX",
        name="BSERISK20250618-FINAL",
    )
    _store(tmp_path, eod, tag="FINAL")
    m = SpanMargin(tmp_path)
    leg = MarginLeg("CE", 20_000, expiry, -20)
    got = m.margin("SENSEX", dt.date(2025, 6, 19), [leg], index_prev_close=NIFTY_CLOSE)
    assert got.span == pytest.approx((200 + 80) * 20)
    assert got.source == "BSE's margin file for 18 Jun 2025 (end of day)"
    assert "did not publish a start-of-day margin file for 19 Jun 2025" in got.note
    assert "end-of-day file for 18 Jun 2025" in got.note


def test_a_missing_day_with_no_end_of_day_file_uses_the_last_start_of_day_file(tmp_path):
    expiry = dt.date(2025, 6, 26)
    _store(tmp_path, _day(dt.date(2025, 6, 13), [(KIND_CALL, expiry, 20_000, 70.0)]))
    got = SpanMargin(tmp_path).margin(
        "NIFTY", dt.date(2025, 6, 16), [_sell(20_000, expiry)], index_prev_close=NIFTY_CLOSE
    )
    assert got.span == pytest.approx((200 + 70) * 65)
    assert got.note == (
        "NSE did not publish a start-of-day margin file for 16 Jun 2025, "
        "so we used the one for 13 Jun 2025."
    )


def test_a_contract_listed_during_the_day_comes_from_the_next_file_and_says_so(tmp_path):
    day, expiry, new = dt.date(2025, 6, 20), dt.date(2025, 6, 26), dt.date(2025, 7, 3)
    _store(tmp_path, _day(day, [(KIND_CALL, expiry, 20_000, 100.0)]))
    _store(
        tmp_path,
        _day(
            dt.date(2025, 6, 23),
            [(KIND_CALL, expiry, 20_000, 90.0), (KIND_CALL, new, 20_000, 120.0)],
        ),
    )
    got = SpanMargin(tmp_path).margin(
        "NIFTY", day, [_sell(20_000, expiry), _sell(20_000, new)], index_prev_close=NIFTY_CLOSE
    )
    # Both legs from the later file -- scenarios from two files cannot be added.
    assert got.span == pytest.approx(130 * 200 + 65 * 90 + 65 * 120)
    assert got.source == "NSE's margin file for 23 Jun 2025 (start of day)"
    assert "NIFTY 3 Jul 2025 20000 CE was not yet listed" in got.note
    assert "file for 23 Jun 2025, the first one that has it" in got.note


def test_a_contract_in_no_file_is_refused(tmp_path):
    day = dt.date(2025, 6, 20)
    _store(tmp_path, _day(day, [(KIND_CALL, dt.date(2025, 6, 26), 20_000, 100.0)]))
    with pytest.raises(MarginUnavailable, match="20100 CE is not in NSE's margin file"):
        SpanMargin(tmp_path).margin(
            "NIFTY", day, [_sell(20_100, dt.date(2025, 6, 26))], index_prev_close=NIFTY_CLOSE
        )


@pytest.mark.parametrize(
    "underlying, day",
    [
        ("NIFTY", dt.date(2022, 11, 30)),
        ("NIFTY", dt.date(2026, 1, 1)),
        ("SENSEX", dt.date(2023, 7, 19)),  # NIFTY's window starts earlier; SENSEX's does not
    ],
)
def test_outside_the_stored_window_is_refused(tmp_path, underlying, day):
    # Even with a file the day before, a day outside the window is not guessed.
    prior = day - dt.timedelta(days=1)
    ex = "BSE" if underlying == "SENSEX" else "NSE"
    _store(
        tmp_path,
        _day(
            prior,
            [(KIND_CALL, day + dt.timedelta(days=7), 20_000, 1.0)],
            exchange=ex,
            underlying=underlying,
        ),
        tag="00" if ex == "BSE" else "i1",
    )
    leg = _sell(20_000, day + dt.timedelta(days=7))
    with pytest.raises(MarginUnavailable, match="margin files from"):
        SpanMargin(tmp_path).margin(underlying, day, [leg], index_prev_close=NIFTY_CLOSE)


def test_a_long_gap_inside_the_window_is_refused(tmp_path):
    with pytest.raises(MarginUnavailable, match="No NSE margin file for 20 Jun 2025"):
        SpanMargin(tmp_path).margin(
            "NIFTY",
            dt.date(2025, 6, 20),
            [_sell(20_000, dt.date(2025, 6, 26))],
            index_prev_close=NIFTY_CLOSE,
        )


def test_fixed_margin_for_tests():
    leg = _sell(20_000, dt.date(2025, 6, 26))
    assert (
        FixedMargin(50_000).margin("NIFTY", dt.date(2025, 6, 20), [leg], index_prev_close=1).total
        == 50_000
    )
    per_unit = FixedMargin(lambda u, d, legs, close: 10 * sum(-x.quantity for x in legs))
    assert per_unit.margin("NIFTY", dt.date(2025, 6, 20), [leg], index_prev_close=1).total == 650


# ------------------------------------------------------------ reading files

_XML = """<?xml version="1.0"?>
<spanFile><fileFormat>4.00</fileFormat><created>202503102130</created>
<pointInTime><date>20250311</date><isSetl>0</isSetl>
<clearingOrg><ec>NSCCL</ec><exchange><exch>NSE</exch>
<phyPf><pfId>1</pfId><pfCode>NIFTY</pfCode>
<phy><cId>1</cId><pe>00000000</pe><p>22460.30</p>
<scanRate><r>1</r><priceScan>2088.81</priceScan><volScan>0.04</volScan></scanRate></phy></phyPf>
<phyPf><pfId>9</pfId><pfCode>RELIANCE</pfCode>
<phy><cId>1</cId><pe>00000000</pe><p>1250.00</p>
<scanRate><r>1</r><priceScan>170.00</priceScan><volScan>0.1</volScan></scanRate></phy></phyPf>
<futPf><pfId>2</pfId><pfCode>NIFTY</pfCode>
<fut><cId>1</cId><pe>20250327</pe><p>22510.00</p>
<ra><r>1</r>{fut}<d>1.00000</d></ra></fut></futPf>
<oopPf><pfId>3</pfId><pfCode>NIFTY</pfCode>
<series><pe>20250313</pe>
<opt><cId>1</cId><o>C</o><k>22500.00</k><p>151.35</p><d>0.48</d>
<ra><r>1</r>{opt}<d>0.4712</d></ra></opt>
<opt><cId>2</cId><o>P</o><k>22000.00</k><p>40.05</p><d>-0.2</d>
<ra><r>1</r>{opt}<d>-0.2101</d></ra></opt>
</series></oopPf>
<oopPf><pfId>10</pfId><pfCode>RELIANCE</pfCode>
<series><pe>20250327</pe>
<opt><cId>1</cId><o>C</o><k>1300.00</k><p>12.00</p><d>0.3</d>
<ra><r>1</r>{opt}<d>0.3</d></ra></opt></series></oopPf>
</exchange>
<ccDef><cc>NIFTY</cc><somMeth>GROSS</somMeth>
<somTiers><tier><tn>0</tn><rate><r>1</r><val>0</val></rate></tier></somTiers>
<dSpread><spread>2</spread><chargeMeth>F</chargeMeth><rate><r>1</r><val>375.000000</val></rate>
<pLeg><cc>NIFTY</cc><pe>20250313</pe><rs>A</rs><i>1.000000</i></pLeg>
<pLeg><cc>NIFTY</cc><pe>20250327</pe><rs>B</rs><i>1.000000</i></pLeg></dSpread>
<dSpread><spread>1</spread><chargeMeth>F</chargeMeth><rate><r>1</r><val>375.000000</val></rate>
<pLeg><cc>NIFTY</cc><pe>20250313</pe><rs>B</rs><i>1.000000</i></pLeg>
<pLeg><cc>NIFTY</cc><pe>20250320</pe><rs>A</rs><i>1.000000</i></pLeg></dSpread>
</ccDef>
<ccDef><cc>BANKNIFTY</cc><somMeth>GROSS</somMeth></ccDef>
</clearingOrg></pointInTime></spanFile>
"""


def _arr(values):
    return "".join(f"<a>{v:.2f}</a>" for v in values)


def test_reads_an_exchange_file_and_keeps_only_our_indices(tmp_path):
    opt = [round(x, 2) for x in np.linspace(-150.25, 151.35, 16)]
    fut = [round(x, 2) for x in np.linspace(-2088.81, 2088.81, 16)]
    path = tmp_path / "t.spn"
    path.write_text(_XML.format(opt=_arr(opt), fut=_arr(fut)))
    day = parse_span_xml(path, "NSE", "nsccl.20250311.i1")

    assert day.file_date == dt.date(2025, 3, 11) and day.created == "202503102130"
    assert not day.is_settlement
    assert set(day.underlying) == {"NIFTY"}  # RELIANCE dropped
    assert len(day.kind) == 3
    call = int(np.flatnonzero(day.kind == KIND_CALL)[0])
    assert day.strike[call] == 22_500 and day.price[call] == 151.35
    assert day.expiry[call] == 20250313 and day.delta[call] == pytest.approx(0.4712)
    assert day.risk[call].tolist() == opt
    nifty = day.params["NIFTY"]
    assert nifty.spot == 22_460.30 and nifty.price_scan == 2088.81
    # Spreads in priority order, sides A/B as the file gives them.
    assert [s.priority for s in nifty.spreads] == [1, 2]
    assert (nifty.spreads[0].leg_a, nifty.spreads[0].leg_b) == (20250320, 20250313)

    # The compact copy keeps every number exactly.
    save_day(day, tmp_path / "c.npz")
    back = load_day(tmp_path / "c.npz")
    assert back.risk.tolist() == day.risk.tolist()
    assert back.price.tolist() == day.price.tolist()
    assert back.strike.tolist() == day.strike.tolist()
    assert back.delta.tolist() == pytest.approx(day.delta.tolist())
    assert back.params["NIFTY"].spreads == nifty.spreads
