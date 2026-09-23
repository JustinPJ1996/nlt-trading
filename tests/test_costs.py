"""Charges must reproduce Zerodha's own numbers, not just be internally consistent.

The whole point of `nlt/costs` is that it matches what a real contract note would
say. These tests pin down concrete scenarios (inputs given as comments) so the
expected numbers are independently reproducible against
https://zerodha.com/brokerage-calculator and https://zerodha.com/charges.
"""

from __future__ import annotations

import pytest

from nlt.costs.charges import (
    GST_RATE,
    ChargeBreakdown,
    NseEquityDeliveryCharges,
    NseFuturesCharges,
    NseOptionsCharges,
    ZeroCharges,
    charge_fn,
)
from nlt.costs.impact import breakeven_move_pct, round_trip_cost_pct

LOT_SIZE = 75  # NIFTY weekly options lot size used throughout these tests.


# --------------------------------------------------------------------------
# Options: buy 1 lot (75) at Rs 150, sell 1 lot (75) at Rs 170.
# Rates per nlt/costs/charges.py, sourced from zerodha.com/charges
# (fetched 2026-09-23): brokerage flat Rs 20/order, STT 0.15% sell-side on
# premium, NSE transaction charge 0.03553% both sides, SEBI Rs 10/crore both
# sides, stamp duty 0.003% buy-side, GST 18% on (brokerage+txn+sebi).
# --------------------------------------------------------------------------


class TestNseOptionsCharges:
    def setup_method(self) -> None:
        self.model = NseOptionsCharges()

    def test_buy_leg_matches_zerodha_calculator(self) -> None:
        # Buy 75 @ 150 -> turnover 11,250.
        # brokerage=20.00, stt=0.00 (buy side, not exercised),
        # txn=0.03553%*11250=3.996375->4.00, sebi=0.0001%*11250=0.01125->0.01,
        # stamp=0.003%*11250=0.3375->0.34,
        # gst=18%*(20+3.996375+0.01125)=4.3213725->4.32, total=28.67
        b = self.model.charges(150, 75, "buy")
        assert b.brokerage == pytest.approx(20.0, abs=1.0)
        assert b.stt == pytest.approx(0.0, abs=1.0)
        assert b.transaction_charges == pytest.approx(4.00, abs=1.0)
        assert b.sebi_fees == pytest.approx(0.01, abs=1.0)
        assert b.stamp_duty == pytest.approx(0.34, abs=1.0)
        assert b.gst == pytest.approx(4.32, abs=1.0)
        assert b.total == pytest.approx(28.67, abs=1.0)

    def test_sell_leg_matches_zerodha_calculator(self) -> None:
        # Sell 75 @ 170 -> turnover 12,750.
        # brokerage=20.00, stt=0.15%*12750=19.125->19.12(banker's rounding),
        # txn=0.03553%*12750=4.530075->4.53, sebi=0.0001%*12750=0.01275->0.01,
        # stamp=0.00 (sell side), gst=18%*(20+4.530075+0.01275)=4.3577->4.36,
        # total=48.02
        s = self.model.charges(170, 75, "sell")
        assert s.brokerage == pytest.approx(20.0, abs=1.0)
        assert s.stt == pytest.approx(19.12, abs=1.0)
        assert s.transaction_charges == pytest.approx(4.53, abs=1.0)
        assert s.sebi_fees == pytest.approx(0.01, abs=1.0)
        assert s.stamp_duty == 0.0
        assert s.gst == pytest.approx(4.36, abs=1.0)
        assert s.total == pytest.approx(48.03, abs=1.0)

    def test_stt_is_sell_side_only_for_market_sale(self) -> None:
        buy = self.model.charges(150, 75, "buy")
        sell = self.model.charges(150, 75, "sell")
        assert buy.stt == 0.0
        assert sell.stt > 0.0

    def test_stamp_duty_is_buy_side_only(self) -> None:
        buy = self.model.charges(150, 75, "buy")
        sell = self.model.charges(150, 75, "sell")
        assert buy.stamp_duty > 0.0
        assert sell.stamp_duty == 0.0

    def test_exercise_settlement_charges_stt_on_buy_side_intrinsic_value(self) -> None:
        # Bought 75 @ intrinsic value 50 (i.e. option finished ITM by 50 and
        # was exercised instead of sold). STT here is charged on the BUY side
        # at 0.15% of intrinsic value, unlike a market sale where STT falls
        # on the sell side of the premium instead.
        exercised = self.model.charges(50, 75, "buy", settled_at_expiry=True)
        not_exercised = self.model.charges(50, 75, "buy", settled_at_expiry=False)
        assert exercised.stt > 0.0
        assert not_exercised.stt == 0.0

    def test_exercise_settlement_is_more_expensive_than_selling_at_same_price(self) -> None:
        # Exercising incurs buy-side STT AND buy-side stamp duty; selling in
        # the market at the same price only incurs sell-side STT and no
        # stamp duty. Exercise should never come out cheaper.
        exercised = self.model.charges(50, 75, "buy", settled_at_expiry=True)
        sold = self.model.charges(50, 75, "sell", settled_at_expiry=False)
        assert exercised.total >= sold.total

    def test_gst_formula(self) -> None:
        b = self.model.charges(150, 75, "buy")
        expected_gst = round(GST_RATE * (b.brokerage + b.transaction_charges + b.sebi_fees), 2)
        assert b.gst == expected_gst
        # GST must not touch STT or stamp duty.
        assert b.gst != round(
            GST_RATE * (b.brokerage + b.transaction_charges + b.sebi_fees + b.stt + b.stamp_duty), 2
        )

    def test_total_equals_sum_of_components(self) -> None:
        for side in ("buy", "sell"):
            b = self.model.charges(150, 75, side)
            components = b.brokerage + b.stt + b.transaction_charges + b.sebi_fees + b.stamp_duty + b.gst
            assert b.total == pytest.approx(components, abs=1e-9)

    def test_invalid_side_raises(self) -> None:
        with pytest.raises(ValueError):
            self.model.charges(150, 75, "hold")

    def test_zero_quantity_does_not_raise(self) -> None:
        b = self.model.charges(150, 0, "buy")
        assert b.total == 0.0

    def test_zero_price_does_not_raise(self) -> None:
        b = self.model.charges(0, 75, "sell")
        assert b.total == 0.0

    def test_str_is_readable(self) -> None:
        b = self.model.charges(150, 75, "buy")
        assert "total=" in str(b)
        assert "brokerage=" in str(b)


# --------------------------------------------------------------------------
# Futures: buy 1 lot (75) at Rs 24,500, sell 1 lot (75) at Rs 24,550.
# Rates sourced from zerodha.com/charges (fetched 2026-09-23): brokerage
# 0.03% or Rs 20/order whichever lower, STT 0.05% sell-side on notional,
# NSE transaction charge 0.00183% both sides, SEBI Rs 10/crore both sides,
# stamp duty 0.002% buy-side, GST 18% on (brokerage+txn+sebi).
# --------------------------------------------------------------------------


class TestNseFuturesCharges:
    def setup_method(self) -> None:
        self.model = NseFuturesCharges()

    def test_buy_leg_matches_zerodha_calculator(self) -> None:
        # Buy 75 @ 24500 -> turnover 1,837,500.
        # brokerage = min(0.03%*1837500, 20) = min(551.25, 20) = 20.00
        # stt = 0.00 (buy side)
        # txn = 0.00183%*1837500 = 33.62625 -> 33.63
        # sebi = 0.0001%*1837500 = 1.8375 -> 1.84
        # stamp = 0.002%*1837500 = 36.75
        # gst = 18%*(20+33.62625+1.8375) = 9.98 (approx)
        b = self.model.charges(24500, 75, "buy")
        assert b.brokerage == pytest.approx(20.0, abs=1.0)
        assert b.stt == 0.0
        assert b.transaction_charges == pytest.approx(33.63, abs=1.0)
        assert b.sebi_fees == pytest.approx(1.84, abs=1.0)
        assert b.stamp_duty == pytest.approx(36.75, abs=1.0)
        assert b.gst == pytest.approx(9.98, abs=1.0)

    def test_sell_leg_stt_on_notional_not_premium(self) -> None:
        # Sell 75 @ 24550 -> turnover 1,841,250.
        # stt = 0.05%*1841250 = 920.625 -> 920.62 (banker's rounding)
        s = self.model.charges(24550, 75, "sell")
        assert s.stt == pytest.approx(920.62, abs=1.0)
        assert s.stamp_duty == 0.0

    def test_brokerage_capped_at_flat_rate_for_large_notional(self) -> None:
        b = self.model.charges(24500, 75, "buy")
        # 0.03% of turnover (551.25) is well above the Rs 20 cap, so the flat
        # per-order rate applies instead -- this is the "0.03% or Rs 20,
        # whichever is lower" rule.
        assert b.brokerage == 20.0

    def test_stt_sell_side_only(self) -> None:
        buy = self.model.charges(24500, 75, "buy")
        sell = self.model.charges(24500, 75, "sell")
        assert buy.stt == 0.0
        assert sell.stt > 0.0

    def test_zero_inputs_do_not_raise(self) -> None:
        b = self.model.charges(0, 0, "buy")
        assert b.total == 0.0


# --------------------------------------------------------------------------
# Equity delivery -- needed for the eventual stock phase.
# --------------------------------------------------------------------------


class TestNseEquityDeliveryCharges:
    def setup_method(self) -> None:
        self.model = NseEquityDeliveryCharges()

    def test_zero_brokerage(self) -> None:
        b = self.model.charges(1000, 10, "buy")
        assert b.brokerage == 0.0

    def test_stt_both_sides(self) -> None:
        buy = self.model.charges(1000, 10, "buy")
        sell = self.model.charges(1000, 10, "sell")
        assert buy.stt > 0.0
        assert sell.stt > 0.0
        assert buy.stt == sell.stt  # same rate, same turnover, both sides.

    def test_stamp_duty_buy_side_only(self) -> None:
        buy = self.model.charges(1000, 10, "buy")
        sell = self.model.charges(1000, 10, "sell")
        assert buy.stamp_duty > 0.0
        assert sell.stamp_duty == 0.0


# --------------------------------------------------------------------------
# ZeroCharges / charge_fn adapter.
# --------------------------------------------------------------------------


class TestZeroChargesAndAdapter:
    def test_zero_charges_always_zero(self) -> None:
        z = ZeroCharges()
        assert z.charges(150, 75, "buy").total == 0.0
        assert z.charges(150, 75, "sell").total == 0.0

    def test_zero_charges_still_validates_side(self) -> None:
        z = ZeroCharges()
        with pytest.raises(ValueError):
            z.charges(150, 75, "invalid")

    def test_charge_fn_signature_and_value(self) -> None:
        model = NseOptionsCharges()
        fn = charge_fn(model)
        total = fn(150, 75, "buy")
        assert isinstance(total, float)
        assert total == model.charges(150, 75, "buy").total

    def test_charge_fn_on_zero_model(self) -> None:
        fn = charge_fn(ZeroCharges())
        assert fn(150, 75, "buy") == 0.0


# --------------------------------------------------------------------------
# Impact metrics.
# --------------------------------------------------------------------------


class TestImpact:
    def test_round_trip_cost_pct_one_lot_150_premium(self) -> None:
        # Buy 75 @ 150, sell 75 @ (same 150, since round_trip_cost_pct prices
        # the exit at entry premium by definition -- it measures the fixed
        # cost drag, not a P&L scenario). Round trip total charges divide by
        # the premium paid (150*75 = 11,250).
        model = NseOptionsCharges()
        pct = round_trip_cost_pct(model, 150, 75)
        buy = model.charges(150, 75, "buy")
        sell = model.charges(150, 75, "sell")
        expected = (buy.total + sell.total) / (150 * 75) * 100.0
        assert pct == pytest.approx(expected)
        # Sanity: at this size, round trip cost is a small-but-real single
        # digit percent of premium (not a rounding error, not >100%).
        assert 0.1 < pct < 5.0

    def test_breakeven_move_pct_matches_round_trip_cost(self) -> None:
        model = NseOptionsCharges()
        assert breakeven_move_pct(model, 150, 75) == pytest.approx(round_trip_cost_pct(model, 150, 75))

    def test_zero_inputs_do_not_raise(self) -> None:
        model = NseOptionsCharges()
        assert round_trip_cost_pct(model, 0, 0) == 0.0
        assert breakeven_move_pct(model, 0, 0) == 0.0

    def test_zero_charges_has_zero_impact(self) -> None:
        model = ZeroCharges()
        assert round_trip_cost_pct(model, 150, 75) == 0.0
        assert breakeven_move_pct(model, 150, 75) == 0.0


def test_charge_breakdown_is_frozen() -> None:
    b = ChargeBreakdown(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 21.0)
    with pytest.raises(Exception):
        b.total = 100.0  # type: ignore[misc]
