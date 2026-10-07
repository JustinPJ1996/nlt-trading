"""Candles from Zerodha Kite, read through a logged-in Kite web session.

This is NOT the official Kite Connect API. It reuses the `enctoken` that the
Kite website sets in the browser after login -- the workaround Justin chose on
2026-10-03 for a prototype to show colleagues, knowing the trade-offs:

  * It goes against Zerodha's terms for the web session, and can stop working
    without notice. The official route is the Rs 500/month Kite Connect plan;
    swapping to it means replacing `_get`'s host and auth header, nothing more.
  * The token is a key to a REAL trading account. It could place real orders.

So this module is built around one rule: **it can only read prices.** `_get` is
the only function that talks to Zerodha, it only sends GET, and it refuses any
path that is not on `_ALLOWED_PATHS` -- historical candles, and the profile
lookup used to tell whether the token still works. There is no code here for
orders, positions, funds or anything else, and `tests/test_kite.py` fails if
any appears.

The token lives in `~/.config/nlt/kite_enctoken`, outside the repository (which
is public), readable only by this user. It is written by `save_token` and never
printed, logged, or shown on screen again.

If a Kite login has been saved, `nlt/data/kite_login.py` fetches a fresh token
by itself whenever there is none or Kite rejects the old one. Logging in has to
POST, so it lives in that module, not here: this one still only reads.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import time
from pathlib import Path

import pandas as pd

from nlt.data.session import NSE_EQUITY, filter_to_session
from nlt.data.source import CACHE_DIR, normalise

KITE_HOST = "https://kite.zerodha.com"
INSTRUMENTS_URL = "https://api.kite.trade/instruments"

TOKEN_PATH = Path.home() / ".config" / "nlt" / "kite_enctoken"

# Read-only, and nothing else. Anything not matching is refused before a byte
# leaves this machine.
_ALLOWED_PATHS = (
    re.compile(
        r"^/oms/instruments/historical/\d+/"
        r"(minute|3minute|5minute|15minute|30minute|60minute|day)$"
    ),
    re.compile(r"^/oms/user/profile$"),
)

KITE_INTERVALS = {
    "1m": "minute",
    "3m": "3minute",
    "5m": "5minute",
    "15m": "15minute",
    "30m": "30minute",
    "1h": "60minute",
    "1d": "day",
}

# Kite's own limit on how many days one request may span, per interval.
_MAX_DAYS_PER_REQUEST = {
    "minute": 60,
    "3minute": 100,
    "5minute": 100,
    "15minute": 200,
    "30minute": 200,
    "60minute": 400,
    "day": 2000,
}

# Kite allows about 3 historical requests a second.
_REQUEST_GAP_SECONDS = 0.35

_INDEX_NAMES = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK"}


class KiteError(RuntimeError):
    """Base for every Kite problem, so callers can catch one thing."""


class KiteNotConnected(KiteError):
    """No token has been saved."""


class KiteTokenExpired(KiteError):
    """Kite rejected the token, and logging in again did not help or was not possible."""


class KiteUnavailable(KiteError):
    """Kite could not be reached, or answered with something unusable."""


class KiteLoginFailed(KiteError):
    """A saved Kite login was tried and did not work. The message says what to do."""


# ------------------------------------------------------------------ the token


def save_token(token: str) -> None:
    """Store the token where only this user can read it. Never echoes it."""
    token = token.strip()
    if token.lower().startswith("enctoken "):
        token = token[len("enctoken ") :].strip()
    if not token or any(c.isspace() for c in token):
        raise ValueError("that doesn't look like a Kite token -- copy just the enctoken value")
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(TOKEN_PATH.parent, 0o700)
    fd = os.open(TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(token + "\n")


def load_token() -> str | None:
    try:
        token = TOKEN_PATH.read_text().strip()
    except OSError:
        return None
    return token or None


def _can_log_in() -> bool:
    from nlt.data import kite_login

    return kite_login.has_credentials()


def _log_in() -> str:
    from nlt.data import kite_login

    kite_login.login()
    token = load_token()
    if token is None:
        raise KiteLoginFailed("Kite login finished but no token was saved")
    return token


def is_connected() -> bool:
    return load_token() is not None or _can_log_in()


# ------------------------------------------------------------------ transport


def _get(path: str, params: dict | None = None, *, session=None) -> dict:
    """The only function that talks to Zerodha. GET, allow-listed paths only.

    With a saved Kite login, a missing or rejected token is replaced by logging
    in -- once per call, so a token Kite keeps rejecting cannot cause a loop.
    """
    if not any(p.fullmatch(path) for p in _ALLOWED_PATHS):
        raise PermissionError(f"refusing to call Kite path {path!r}: this client only reads prices")
    token = load_token()
    logged_in = False
    if token is None:
        if not _can_log_in():
            raise KiteNotConnected("no Kite token saved")
        token, logged_in = _log_in(), True
    try:
        return _send(path, params, token, session)
    except KiteTokenExpired:
        if logged_in or not _can_log_in():
            raise
    return _send(path, params, _log_in(), session)


def _send(path: str, params: dict | None, token: str, session) -> dict:
    import requests

    http = session or requests
    try:
        resp = http.get(
            KITE_HOST + path,
            params=params or {},
            headers={"Authorization": f"enctoken {token}", "X-Kite-Version": "3"},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise KiteUnavailable(f"could not reach Kite: {exc.__class__.__name__}") from None

    if resp.status_code in (401, 403):
        raise KiteTokenExpired("Kite rejected the token -- it has expired or was logged out")
    try:
        body = resp.json()
    except ValueError:
        raise KiteUnavailable(
            f"Kite answered {resp.status_code} with something unreadable"
        ) from None
    if body.get("error_type") == "TokenException":
        raise KiteTokenExpired("Kite rejected the token -- it has expired or was logged out")
    if resp.status_code != 200 or body.get("status") != "success":
        raise KiteUnavailable(f"Kite answered {resp.status_code}: {body.get('message', '')[:200]}")
    return body.get("data") or {}


def check_connection() -> str:
    """Returns the Kite user id the token belongs to. Raises a `KiteError` if unusable."""
    data = _get("/oms/user/profile")
    return str(data.get("user_id", ""))


# ---------------------------------------------------------------- instruments

_INSTRUMENTS_CACHE = CACHE_DIR / "kite_instruments.csv"


def _instruments(max_age_hours: float = 20.0) -> pd.DataFrame:
    """Kite's public instrument list (no login needed), cached for a day."""
    fresh = (
        _INSTRUMENTS_CACHE.exists()
        and time.time() - _INSTRUMENTS_CACHE.stat().st_mtime < max_age_hours * 3600
    )
    if not fresh:
        import requests

        try:
            resp = requests.get(INSTRUMENTS_URL, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            if _INSTRUMENTS_CACHE.exists():
                return pd.read_csv(_INSTRUMENTS_CACHE)
            raise KiteUnavailable(f"could not download Kite's instrument list: {exc}") from None
        _INSTRUMENTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _INSTRUMENTS_CACHE.write_text(resp.text)
    return pd.read_csv(_INSTRUMENTS_CACHE)


def futures_contracts(symbol: str, instruments: pd.DataFrame | None = None) -> pd.DataFrame:
    """Kite's live futures contracts on `symbol`, nearest expiry first."""
    from nlt.data.futures import contract

    c = contract(symbol)
    df = _instruments() if instruments is None else instruments
    rows = df[
        (df["exchange"] == c.exchange) & (df["instrument_type"] == "FUT") & (df["name"] == c.symbol)
    ].copy()
    if rows.empty:
        raise KiteUnavailable(f"Kite lists no {c.exchange} futures contracts for {c.symbol}")
    rows["expiry"] = pd.to_datetime(rows["expiry"]).dt.date
    return rows.sort_values("expiry").reset_index(drop=True)


def listed_expiries(symbol: str, instruments: pd.DataFrame | None = None) -> list[dt.date]:
    """Expiry dates the exchange has published for `symbol`'s live contracts."""
    return list(futures_contracts(symbol, instruments)["expiry"])


def near_month(symbol: str, on: dt.date, instruments: pd.DataFrame | None = None) -> pd.Series:
    """The contract a near-month trader holds on `on`: the first not yet expired.

    On its expiry day a contract is still the near month -- it trades until the
    close, and that is when a position in it is closed.
    """
    rows = futures_contracts(symbol, instruments)
    live = rows[rows["expiry"] >= on]
    if live.empty:
        raise KiteUnavailable(f"Kite lists no {symbol} futures contract expiring on or after {on}")
    return live.iloc[0]


def instrument_token(symbol: str, instruments: pd.DataFrame | None = None) -> int:
    """NSE instrument token for an index ("NIFTY", "BANKNIFTY") or an NSE share."""
    df = _instruments() if instruments is None else instruments
    nse = df[df["exchange"] == "NSE"]
    if symbol in _INDEX_NAMES:
        rows = nse[(nse["segment"] == "INDICES") & (nse["tradingsymbol"] == _INDEX_NAMES[symbol])]
    else:
        rows = nse[(nse["instrument_type"] == "EQ") & (nse["tradingsymbol"] == symbol)]
        rows = rows[rows["segment"] == "NSE"]
    if len(rows) != 1:
        raise KiteUnavailable(f"Kite has no single NSE instrument for {symbol!r}")
    return int(rows["instrument_token"].iloc[0])


# --------------------------------------------------------------------- candles


def _parse_candles(candles: list) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        [row[:6] for row in candles],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    # Kite stamps candles with an explicit +05:30 offset. Parsed as such, they
    # are already tz-aware -- never naive, which `normalise` would assume is IST
    # without being able to check.
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        raise KiteUnavailable("Kite returned candles without a timezone; refusing to guess")
    return df.set_index("timestamp")


class KiteSource:
    """`BarSource` over Kite historical candles. Daily or intraday, from the live session."""

    def __init__(self, http=None) -> None:
        self._http = http
        self.last_notes: list[str] = []

    def bars(
        self,
        symbol: str,
        interval: str = "1d",
        start: dt.date | None = None,
        end: dt.date | None = None,
    ) -> pd.DataFrame:
        if interval not in KITE_INTERVALS:
            raise ValueError(
                f"Kite has no {interval!r} candles; supported: {sorted(KITE_INTERVALS)}"
            )
        kite_interval = KITE_INTERVALS[interval]
        end = end or dt.date.today()
        start = start or end - dt.timedelta(days=_MAX_DAYS_PER_REQUEST[kite_interval])
        notes: list[str] = []
        session = NSE_EQUITY
        extra: dict = {}

        from nlt.data import futures

        future = futures.from_data_key(symbol)
        if future is None:
            token = instrument_token(symbol)
        else:
            session = futures.contract(future).session
            current = near_month(future, dt.date.today())
            token = int(current["instrument_token"])
            if interval == "1d":
                # Kite's continuous series: each day's candle from the contract
                # that was nearest to expiry that day, asked for through today's.
                # Contracts are joined end to end without adjustment, so the
                # jump between two contracts is in the data; the engine closes
                # every position at expiry so that jump is never traded.
                extra = {"continuous": 1}
            else:
                # Kite keeps no intraday candles for expired contracts, so an
                # intraday futures test can only cover the live contract's life.
                notes.append(
                    f"Intraday candles for {future} futures come from the current contract "
                    f"({current['tradingsymbol']}, expiring {current['expiry']:%d %b %Y}) only. "
                    "Kite keeps no intraday history for expired contracts, so the test "
                    "cannot reach further back than this contract has existed. Before the "
                    "previous contract expired, this one was not yet the nearest month and "
                    "traded more thinly than the test assumes."
                )

        frames = []
        step = dt.timedelta(days=_MAX_DAYS_PER_REQUEST[kite_interval] - 1)
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + step, end)
            if frames:
                time.sleep(_REQUEST_GAP_SECONDS)
            data = _get(
                f"/oms/instruments/historical/{token}/{kite_interval}",
                {
                    "from": f"{chunk_start} 00:00:00",
                    "to": f"{chunk_end} 23:59:59",
                    "oi": 0,
                    **extra,
                },
                session=self._http,
            )
            frames.append(_parse_candles(data.get("candles") or []))
            chunk_start = chunk_end + dt.timedelta(days=1)

        frames = [f for f in frames if not f.empty]
        if not frames:
            self.last_notes = [*notes, f"Kite returned no {interval} candles for {symbol}"]
            return normalise(
                pd.DataFrame(
                    columns=["open", "high", "low", "close", "volume"],
                    index=pd.DatetimeIndex([], tz="Asia/Kolkata"),
                )
            )
        df = normalise(pd.concat(frames))
        self.last_notes = notes
        if interval != "1d":
            df, dropped = filter_to_session(df, session)
            self.last_notes.extend(dropped)
        return df
