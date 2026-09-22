import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def synthetic_bars() -> pd.DataFrame:
    """A deterministic OHLCV series with trend, chop and a crash.

    Random-walk data alone never triggers pattern or reversal logic, so this
    deliberately includes a sharp decline and a recovery.
    """
    rng = np.random.default_rng(20260922)
    n = 600

    drift = np.concatenate([
        np.linspace(0.0, 0.6, 200),      # uptrend
        np.full(200, 0.6),               # sideways
        np.linspace(0.6, -0.9, 100),     # decline
        np.linspace(-0.9, 0.3, 100),     # recovery
    ])
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


@pytest.fixture(scope="session")
def nifty_bars() -> pd.DataFrame:
    """Real NIFTY daily bars, skipped if the cache has not been populated."""
    from nlt.data.source import CACHE_DIR

    path = CACHE_DIR / "yahoo_NIFTY_1d.parquet"
    if not path.exists():
        pytest.skip("NIFTY cache not populated; run scripts/fetch_data.py")
    return pd.read_parquet(path)
