"""The Slippage run setting: validation, the buy/sell arithmetic, and the
plain-English text a non-technical user reads. Every rule here was checked by
breaking the code it covers and watching the matching test fail.
"""

from __future__ import annotations

import pytest

from nlt.costs.slippage import LEGACY, ZERO, Slippage


def test_zero_is_the_default_and_is_zero():
    s = Slippage()
    assert s.kind == "pct"
    assert s.value == 0.0
    assert s.is_zero
    assert s == ZERO


def test_legacy_is_the_old_hidden_default():
    assert LEGACY.kind == "pct"
    assert LEGACY.value == 0.02
    assert not LEGACY.is_zero


def test_negative_value_is_refused_in_plain_english():
    with pytest.raises(ValueError, match="negative"):
        Slippage("pct", -0.1)


def test_percentage_over_5_is_refused():
    with pytest.raises(ValueError, match="5"):
        Slippage("pct", 5.01)
    Slippage("pct", 5.0)  # the boundary itself is fine


def test_rupee_amount_over_100_is_refused():
    with pytest.raises(ValueError, match="100"):
        Slippage("rs", 100.01)
    Slippage("rs", 100.0)


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="percentage or a rupee"):
        Slippage("bps", 1)


def test_pct_buy_costs_more_and_sell_receives_less():
    s = Slippage("pct", 1.0)
    assert s.apply(100.0, "buy") == pytest.approx(101.0)
    assert s.apply(100.0, "sell") == pytest.approx(99.0)


def test_rs_buy_adds_and_sell_subtracts():
    s = Slippage("rs", 2.5)
    assert s.apply(100.0, "buy") == pytest.approx(102.5)
    assert s.apply(100.0, "sell") == pytest.approx(97.5)


def test_zero_slippage_changes_nothing():
    assert ZERO.apply(123.45, "buy") == pytest.approx(123.45)
    assert ZERO.apply(123.45, "sell") == pytest.approx(123.45)


def test_a_sell_can_never_go_below_one_tick():
    # Rs slippage larger than the price itself would otherwise go negative.
    s = Slippage("rs", 100.0)
    assert s.apply(0.5, "sell") == pytest.approx(0.05)
    # A buy has no floor -- it only ever gets more expensive.
    assert s.apply(0.5, "buy") == pytest.approx(100.5)


def test_an_aggressive_percentage_also_respects_the_floor():
    s = Slippage("pct", 5.0)
    assert s.apply(0.01, "sell") == pytest.approx(0.05)


def test_apply_rejects_an_unknown_side():
    with pytest.raises(ValueError, match="side"):
        ZERO.apply(100.0, "short")


def test_describe_at_zero_is_the_honest_warning():
    assert ZERO.describe() == "No slippage assumed, so real fills would likely be worse."


def test_describe_states_the_kind_and_amount():
    assert Slippage("pct", 0.5).describe() == "0.5% slippage on every order."
    assert Slippage("rs", 1.0).describe() == "Rs 1.00 per unit slippage on every order."


def test_to_dict_and_from_dict_round_trip():
    s = Slippage("rs", 3.25)
    assert Slippage.from_dict(s.to_dict()) == s


def test_from_dict_of_none_is_zero_not_legacy():
    # The whole point: a missing stored value must never be silently treated
    # as the pre-2026-10-13 legacy rate. The caller decides that explicitly.
    assert Slippage.from_dict(None) == ZERO
    assert Slippage.from_dict(None) != LEGACY
