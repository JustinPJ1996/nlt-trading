"""The most important test in the project.

An indicator that peeks at future bars makes a backtest look brilliant and lose
money live. The check is simple and total: compute an indicator on the first k
bars, compute it on all bars, and require the first k values to agree. If a
future bar can change a past value, this fails.

It runs over every indicator in the registry, so a newly added one is covered
the moment it is registered.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nlt.indicators.registry import REGISTRY

# Ichimoku's lagging span is drawn into the past on a chart by definition -- it
# is the one output that is knowingly future-shifted. The engine never reads it.
KNOWN_FUTURE_OUTPUTS = {("ichimoku", "lagging")}

CUT = 400


def _as_frame(out) -> pd.DataFrame:
    return out.to_frame("value") if isinstance(out, pd.Series) else out


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_prefix_invariance(name, synthetic_bars):
    d = REGISTRY[name]

    full = _as_frame(d.fn(synthetic_bars, **d.params))
    prefix = _as_frame(d.fn(synthetic_bars.iloc[:CUT], **d.params))

    for col in prefix.columns:
        if (name, col) in KNOWN_FUTURE_OUTPUTS:
            continue

        a = full[col].iloc[:CUT]
        b = prefix[col]

        # Both must agree on which values exist at all.
        assert a.isna().to_numpy().tolist() == b.isna().to_numpy().tolist(), (
            f"{name}.{col}: NaN pattern changes when future bars are appended -- "
            "a past value became computable only because of later data"
        )

        if a.dtype == bool or b.dtype == bool:
            assert a.equals(b), f"{name}.{col}: boolean output changed with future data"
        else:
            pd.testing.assert_series_equal(
                a, b, check_names=False, rtol=1e-9, atol=1e-9,
                obj=f"{name}.{col} leaks future data",
            )


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_prefix_invariance_on_real_data(name, nifty_bars):
    """Same check against real NIFTY bars, which have gaps, holidays and zero-volume days."""
    d = REGISTRY[name]
    bars = nifty_bars.copy()
    bars.loc[bars.volume == 0, "volume"] = 1_000_000.0
    cut = len(bars) - 200

    full = _as_frame(d.fn(bars, **d.params))
    prefix = _as_frame(d.fn(bars.iloc[:cut], **d.params))

    for col in prefix.columns:
        if (name, col) in KNOWN_FUTURE_OUTPUTS:
            continue
        a, b = full[col].iloc[:cut], prefix[col]
        if a.dtype == bool or b.dtype == bool:
            assert a.equals(b), f"{name}.{col} changed with future data"
        else:
            pd.testing.assert_series_equal(
                a, b, check_names=False, rtol=1e-9, atol=1e-9,
                obj=f"{name}.{col} leaks future data",
            )
