"""Formula correctness.

Each test recomputes the indicator from its textbook definition, written out
longhand and independently of the implementation. Where the "obvious" shortcut
gives a different answer (Wilder smoothing vs EMA, population vs sample stdev),
there is an explicit test that the shortcut is *not* what we do -- otherwise a
future simplification silently desynchronises us from every charting package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nlt.indicators import levels, momentum, trend, volatility
from nlt.indicators.smoothing import ema, rma, sma, stdev, true_range, wma


# --------------------------------------------------------------- primitives

def test_sma_known_values():
    s = pd.Series([1.0, 2, 3, 4, 5, 6])
    assert sma(s, 3).tolist()[2:] == [2.0, 3.0, 4.0, 5.0]
    assert sma(s, 3).isna().sum() == 2


def test_wma_weights_are_linear():
    s = pd.Series([1.0, 2, 3])
    # (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6
    assert wma(s, 3).iloc[-1] == pytest.approx(14 / 6)


def test_ema_is_sma_seeded():
    s = pd.Series(np.arange(1, 21, dtype="float64"))
    out = ema(s, 5)
    assert out.iloc[4] == pytest.approx(3.0)  # mean(1..5)
    alpha = 2 / 6
    assert out.iloc[5] == pytest.approx(alpha * 6 + (1 - alpha) * 3.0)


def test_rma_uses_wilder_alpha_not_ema_alpha():
    s = pd.Series(np.arange(1, 21, dtype="float64"))
    r = rma(s, 5)
    assert r.iloc[5] == pytest.approx((1 / 5) * 6 + (4 / 5) * 3.0)
    # And is demonstrably not the EMA, which would use alpha = 2/6.
    assert r.iloc[5] != pytest.approx(ema(s, 5).iloc[5])


def test_stdev_is_population_not_sample(synthetic_bars):
    got = stdev(synthetic_bars.close, 20)
    window = synthetic_bars.close.iloc[:20].to_numpy()
    assert got.iloc[19] == pytest.approx(np.std(window, ddof=0))
    assert got.iloc[19] != pytest.approx(np.std(window, ddof=1))


def test_true_range_uses_previous_close(synthetic_bars):
    b = synthetic_bars
    tr = true_range(b.high, b.low, b.close)
    i = 50
    expected = max(
        b.high.iloc[i] - b.low.iloc[i],
        abs(b.high.iloc[i] - b.close.iloc[i - 1]),
        abs(b.low.iloc[i] - b.close.iloc[i - 1]),
    )
    assert tr.iloc[i] == pytest.approx(expected)


# --------------------------------------------------------------------- RSI

def _wilder_rsi_longhand(close: pd.Series, length: int) -> pd.Series:
    """Wilder's RSI written out step by step, independent of our implementation."""
    values = close.to_numpy(dtype="float64")
    gains, losses = [], []
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    out = [np.nan] * len(values)
    avg_gain = float(np.mean(gains[:length]))
    avg_loss = float(np.mean(losses[:length]))
    out[length] = 100.0 - 100.0 / (1 + avg_gain / avg_loss) if avg_loss else 100.0

    for i in range(length + 1, len(values)):
        avg_gain = (avg_gain * (length - 1) + gains[i - 1]) / length
        avg_loss = (avg_loss * (length - 1) + losses[i - 1]) / length
        out[i] = 100.0 - 100.0 / (1 + avg_gain / avg_loss) if avg_loss else 100.0

    return pd.Series(out, index=close.index)


def test_rsi_matches_wilder_longhand(synthetic_bars):
    got = momentum.rsi(synthetic_bars.close, 14)
    want = _wilder_rsi_longhand(synthetic_bars.close, 14)
    pd.testing.assert_series_equal(
        got.dropna(), want.dropna(), check_names=False, rtol=1e-9
    )


def test_rsi_is_not_ema_smoothed(synthetic_bars):
    """Guards against 'simplifying' rma to a pandas ewm span -- the classic bug."""
    close = synthetic_bars.close
    delta = close.diff()
    ema_rsi = 100 - 100 / (
        1
        + delta.clip(lower=0).ewm(span=14, adjust=False).mean()
        / (-delta).clip(lower=0).ewm(span=14, adjust=False).mean()
    )
    diff = (momentum.rsi(close, 14) - ema_rsi).abs().max()
    assert diff > 1.0, "RSI appears to be EMA-smoothed; it must use Wilder's RMA"


def test_rsi_bounds_and_extremes():
    r = momentum.rsi(pd.Series(np.arange(1.0, 60.0)), 14)  # monotonically rising
    assert r.dropna().max() == pytest.approx(100.0)
    assert (r.dropna() >= 0).all() and (r.dropna() <= 100).all()

    falling = momentum.rsi(pd.Series(np.arange(60.0, 1.0, -1.0)), 14)
    assert falling.dropna().min() == pytest.approx(0.0)


def test_rsi_flat_series_is_fifty():
    assert momentum.rsi(pd.Series([100.0] * 40), 14).dropna().eq(50.0).all()


# ------------------------------------------------------------------ others

def test_atr_is_rma_of_true_range(synthetic_bars):
    b = synthetic_bars
    pd.testing.assert_series_equal(
        volatility.atr(b.high, b.low, b.close, 14),
        rma(true_range(b.high, b.low, b.close), 14),
        check_names=False,
    )


def test_bollinger_bands_and_percent_b(synthetic_bars):
    bb = volatility.bollinger(synthetic_bars.close, 20, 2.0)
    i = 100
    basis = sma(synthetic_bars.close, 20).iloc[i]
    dev = 2.0 * stdev(synthetic_bars.close, 20).iloc[i]

    assert bb.middle.iloc[i] == pytest.approx(basis)
    assert bb.upper.iloc[i] == pytest.approx(basis + dev)
    assert bb.lower.iloc[i] == pytest.approx(basis - dev)

    # %B is 0 at the lower band and 1 at the upper.
    close_i = synthetic_bars.close.iloc[i]
    assert bb.percent_b.iloc[i] == pytest.approx(
        (close_i - bb.lower.iloc[i]) / (bb.upper.iloc[i] - bb.lower.iloc[i])
    )


def test_macd_components(synthetic_bars):
    m = momentum.macd(synthetic_bars.close, 12, 26, 9)
    close = synthetic_bars.close
    pd.testing.assert_series_equal(
        m.macd, ema(close, 12) - ema(close, 26), check_names=False
    )
    pd.testing.assert_series_equal(m.histogram, m.macd - m.signal, check_names=False)


def test_stochastic_k_definition(synthetic_bars):
    b = synthetic_bars
    st = momentum.stochastic(b.high, b.low, b.close, 14, smooth_k=1, smooth_d=3)
    i = 100
    hh = b.high.iloc[i - 13 : i + 1].max()
    ll = b.low.iloc[i - 13 : i + 1].min()
    assert st.k.iloc[i] == pytest.approx(100 * (b.close.iloc[i] - ll) / (hh - ll))


def test_williams_r_is_stochastic_k_minus_100(synthetic_bars):
    """A known identity -- %R = %K - 100 for the same length."""
    b = synthetic_bars
    k = momentum.stochastic(b.high, b.low, b.close, 14, smooth_k=1).k
    r = momentum.williams_r(b.high, b.low, b.close, 14)
    pd.testing.assert_series_equal(r, k - 100, check_names=False)


def test_cci_uses_mean_absolute_deviation(synthetic_bars):
    b = synthetic_bars
    c = momentum.cci(b.high, b.low, b.close, 20)
    tp = (b.high + b.low + b.close) / 3
    i = 100
    window = tp.iloc[i - 19 : i + 1]
    mad = (window - window.mean()).abs().mean()
    assert c.iloc[i] == pytest.approx((tp.iloc[i] - window.mean()) / (0.015 * mad))


def test_adx_di_are_percentages(synthetic_bars):
    b = synthetic_bars
    a = trend.adx(b.high, b.low, b.close, 14).dropna()
    assert (a.adx.between(0, 100)).all()
    assert (a.plus_di >= 0).all() and (a.minus_di >= 0).all()


def test_supertrend_direction_flips_are_rare(synthetic_bars):
    """A band-ratchet bug shows up as direction flipping on a large share of bars."""
    b = synthetic_bars
    st = volatility.supertrend(b.high, b.low, b.close, 10, 3.0).dropna()
    flip_rate = (st.direction.diff() != 0).mean()
    assert flip_rate < 0.10, f"supertrend flips on {flip_rate:.1%} of bars -- bands not ratcheting"


def test_supertrend_line_sits_on_correct_side(synthetic_bars):
    b = synthetic_bars
    st = volatility.supertrend(b.high, b.low, b.close, 10, 3.0)
    joined = pd.concat([st, b.close], axis=1).dropna()
    up, down = joined[joined.direction > 0], joined[joined.direction < 0]
    # In an uptrend the line is a support below price; in a downtrend, above it.
    assert (up.supertrend <= up.close).mean() > 0.95
    assert (down.supertrend >= down.close).mean() > 0.95


def test_rolling_extremes_exclude_current_bar(synthetic_bars):
    b = synthetic_bars
    ext = levels.rolling_extremes(b.high, b.low, 20)
    i = 100
    assert ext.highest.iloc[i] == pytest.approx(b.high.iloc[i - 20 : i].max())


def test_prior_day_levels_come_from_a_completed_day(nifty_bars):
    """On daily bars, 'previous day high' must be the preceding row's high."""
    pp = levels.prior_period(nifty_bars.high, nifty_bars.low, nifty_bars.close, "day")
    i = 500
    assert pp.high.iloc[i] == pytest.approx(nifty_bars.high.iloc[i - 1])
    assert pp.close.iloc[i] == pytest.approx(nifty_bars.close.iloc[i - 1])


def test_cpr_tc_is_above_bc(nifty_bars):
    c = levels.central_pivot_range(
        nifty_bars.high, nifty_bars.low, nifty_bars.close
    ).dropna()
    assert (c.tc >= c.bc).all()
    assert (c.width_pct >= 0).all()
