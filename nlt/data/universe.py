"""NSE index membership: which symbols make up NIFTY 50 / 100 / 500.

The 19 "stock" strategies in the sample set ("Buy Nifty 100 stocks above the
200 DMA...") name an index, not a symbol list. Something has to turn "NIFTY
100" into 100 concrete NSE symbols before the engine can screen anything.

SURVIVORSHIP BIAS -- read this before backtesting a universe.
================================================================
The membership lists here are a *current* snapshot (see `Universe.as_of`).
Index constituents change: companies get added, dropped, delisted, merged,
renamed. Running a five-year backtest over today's NIFTY 100 only tests
companies that survived and stayed in the index for those five years -- every
name that was kicked out along the way (bankruptcy, delisting, demotion to a
smaller index) is invisible. That systematically flatters the results,
because the failures are silently excluded from the sample.

We do not have point-in-time constituent history, so this module cannot fix
the bias -- it can only make it impossible to ignore. `survivorship_warning`
below returns a plain-English warning whenever a backtest start date is far
enough before the snapshot date that this matters, and callers (the engine,
reports) should surface it rather than swallow it.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_SNAPSHOT_PATH = Path(__file__).resolve().parent / "universe_snapshot.json"

# How many days of "backtest starts before the snapshot" before we warn. A
# few weeks of drift is normal (nobody re-captures the day of every backtest);
# a few years is the survivorship problem described above.
_WARNING_THRESHOLD_DAYS = 365


@dataclass(frozen=True)
class Universe:
    name: str
    symbols: tuple[str, ...]
    as_of: dt.date
    source: str  # "bundled snapshot" | "nseindia.com"


def _canonical_name(raw: str) -> str:
    """Normalise 'nifty50', 'NIFTY_50', 'Nifty 100' etc. to 'NIFTY 50'.

    Users type index names in every format imaginable; the sample strategy
    sentences alone contain "Nifty 100", "NIFTY 50" and "nifty 100" for the
    same three indices. Strip everything but letters and digits, then insert
    the canonical single space between the name and the number.
    """
    stripped = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    match = re.match(r"^(NIFTY)(\d+)$", stripped)
    if not match:
        return raw.strip().upper()
    return f"{match.group(1)} {match.group(2)}"


def _load_snapshot() -> dict:
    with open(_SNAPSHOT_PATH) as f:
        return json.load(f)


def list_universes() -> list[str]:
    """Names of the universes we can resolve, in a stable, readable order."""
    return list(_load_snapshot()["universes"])


def get_universe(name: str) -> Universe:
    """Look up a universe by name, accepting loose spelling.

    Tries a live NSE fetch first (so the caller gets today's membership when
    the network is available) and falls back to the bundled snapshot
    otherwise. `Universe.source` records which one actually supplied the
    data, since that matters for interpreting `as_of`.
    """
    canonical = _canonical_name(name)
    snapshot = _load_snapshot()
    universes = snapshot["universes"]

    if canonical not in universes:
        available = ", ".join(sorted(universes))
        raise ValueError(f"unknown universe {name!r}; available universes: {available}")

    try:
        return refresh_from_nse(canonical)
    except Exception:
        symbols = tuple(sorted(universes[canonical]))
        as_of = dt.date.fromisoformat(snapshot["captured"])
        return Universe(name=canonical, symbols=symbols, as_of=as_of, source="bundled snapshot")


# NSE's constituent CSVs, keyed by canonical universe name.
_NSE_URLS = {
    "NIFTY 50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTY 100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "NIFTY 500": "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
}

_NSE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def refresh_from_nse(name: str) -> Universe:
    """Best-effort live fetch of current constituents straight from NSE.

    NSE blocks requests that don't look like a browser, hence the user-agent.
    This is inherently flaky (NSE changes its anti-scraping rules without
    notice), so every caller of this module goes through `get_universe`,
    which catches failures here and falls back to the bundled snapshot rather
    than letting a backtest depend on NSE being reachable right now.
    """
    import csv
    import io
    import urllib.request

    canonical = _canonical_name(name)
    url = _NSE_URLS.get(canonical)
    if url is None:
        raise ValueError(f"no NSE source URL for {name!r}")

    request = urllib.request.Request(url, headers={"User-Agent": _NSE_USER_AGENT})
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(body))
    symbols = tuple(sorted(row["Symbol"].strip().upper() for row in reader if row.get("Symbol")))
    if not symbols:
        raise RuntimeError(f"NSE returned no constituents for {name!r}")

    return Universe(name=canonical, symbols=symbols, as_of=dt.date.today(), source="nseindia.com")


def resolve_symbols(name_or_symbols: str | Sequence[str]) -> tuple[str, ...]:
    """Turn a universe name, a bare symbol, or a list of symbols into symbols.

    "NIFTY 100" -> 100 symbols. "RELIANCE" -> ("RELIANCE",). ["TCS", "INFY"]
    -> ("TCS", "INFY"), preserving the caller's order (a universe returns its
    symbols sorted, but an explicit list is presumably ordered on purpose).
    """
    if isinstance(name_or_symbols, str):
        canonical = _canonical_name(name_or_symbols)
        if canonical in list_universes():
            return get_universe(canonical).symbols
        return (name_or_symbols.strip().upper(),)

    return tuple(s.strip().upper() for s in name_or_symbols)


def survivorship_warning(universe: Universe, start_date: dt.date) -> str | None:
    """A plain-English warning when a backtest start predates the snapshot
    by enough that survivorship bias is likely to matter.

    Returns None for a backtest that starts recently relative to `as_of`;
    returns a message (never raises) otherwise, so a caller can log or
    display it without a backtest failing outright.
    """
    age_days = (universe.as_of - start_date).days
    if age_days <= _WARNING_THRESHOLD_DAYS:
        return None

    years = age_days / 365.25
    return (
        f"{universe.name} membership is a snapshot as of {universe.as_of.isoformat()} "
        f"({universe.source}), but this backtest starts {years:.1f} years earlier on "
        f"{start_date.isoformat()}. Companies that left the index in between (delisted, "
        "acquired, demoted) are missing from this universe, so the backtest only sees "
        "survivors and its results are likely optimistic. Point-in-time constituent "
        "history would be needed to fix this; treat these results as an upper bound."
    )
