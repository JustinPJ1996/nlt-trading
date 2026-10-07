import os

import numpy as np
import pandas as pd
import pytest


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.network tests unless NLT_RUN_NETWORK_TESTS=1.

    These hit live market data and are not something CI or a routine local
    run should depend on being online for.
    """
    if os.environ.get("NLT_RUN_NETWORK_TESTS") == "1":
        return
    skip_network = pytest.mark.skip(reason="network test; set NLT_RUN_NETWORK_TESTS=1 to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture(autouse=True)
def _no_live_ai(monkeypatch):
    """No test may reach the AI model: it costs money, needs the network, and
    would make results depend on whatever the model happens to say today.
    Tests of the AI path pass a scripted `transport` instead."""
    monkeypatch.setenv("NLT_LLM_DISABLED", "1")


@pytest.fixture(autouse=True)
def _no_live_kite(monkeypatch, tmp_path):
    """No test may use the real Kite token on this machine: it is a key to a
    real trading account, and test results must not depend on the market.
    Tests that need a token save a fake one into this temporary path."""
    monkeypatch.setattr("nlt.data.kite.TOKEN_PATH", tmp_path / "kite_enctoken")
    # Nor the real saved login: a test must never log in to a real account.
    monkeypatch.setattr("nlt.data.kite_login.CREDENTIALS_PATH", tmp_path / "kite_login.json")
    monkeypatch.setattr("nlt.data.kite_login.STATE_PATH", tmp_path / "kite_login_state.json")


@pytest.fixture(scope="session")
def synthetic_bars() -> pd.DataFrame:
    """A deterministic OHLCV series with trend, chop and a crash.

    Random-walk data alone never triggers pattern or reversal logic, so this
    deliberately includes a sharp decline and a recovery.
    """
    rng = np.random.default_rng(20260922)
    n = 600

    drift = np.concatenate(
        [
            np.linspace(0.0, 0.6, 200),  # uptrend
            np.full(200, 0.6),  # sideways
            np.linspace(0.6, -0.9, 100),  # decline
            np.linspace(-0.9, 0.3, 100),  # recovery
        ]
    )
    noise = rng.normal(0, 0.8, n)
    close = 20000 + np.cumsum(drift + noise) * 12

    spread = np.abs(rng.normal(0, 1, n)) * 40 + 10
    open_ = close - rng.normal(0, 1, n) * 20
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(80_000, 400_000, n).astype("float64")

    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="D", tz="Asia/Kolkata")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# Every test built on real NIFTY history is cut off here.
#
# The cache grows: `bars()` refreshes it whenever it goes stale, so without a
# cut-off these fixtures return a different frame depending on the day you run
# them. `test_daily_nifty_rsi_regression_pinned` pins an exact trade count, and
# on 2026-09-29 it broke -- not because anything regressed, but because the
# market had opened and added a bar, turning 56 trades into 57. A test whose
# result depends on the date is not pinning anything.
#
# Moving this date forward is a deliberate act: it will move the pinned numbers
# in `test_daily_nifty_rsi_regression_pinned`, and those numbers exist to prove
# that daily-bar runs take none of the intraday session code paths. Re-pin them
# in the same commit, and only after checking the change is the new data and
# not a real regression.
NIFTY_FIXTURE_CUTOFF = "2026-09-25"


@pytest.fixture(scope="session")
def nifty_bars() -> pd.DataFrame:
    """Real NIFTY daily bars up to `NIFTY_FIXTURE_CUTOFF`, so the frame is
    identical on every run. Skipped if the cache has not been populated."""
    from nlt.data.source import CACHE_DIR

    path = CACHE_DIR / "yahoo_NIFTY_1d.parquet"
    if not path.exists():
        pytest.skip("NIFTY cache not populated; run scripts/fetch_data.py")
    bars = pd.read_parquet(path)
    return bars.loc[: f"{NIFTY_FIXTURE_CUTOFF} 23:59:59+05:30"]
