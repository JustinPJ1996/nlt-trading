"""The Kite reader: read-only by construction, and careful with the token.

`conftest._no_live_kite` points the token file at an empty temporary path for
every test, so nothing here can use the real token on this machine. HTTP is a
fake object; nothing reaches Zerodha.
"""

from __future__ import annotations

import ast
import datetime as dt
import stat
from pathlib import Path

import pandas as pd
import pytest

from nlt.data import kite

KITE_SOURCE = Path(kite.__file__)
REPO = Path(__file__).resolve().parents[1]

INSTRUMENTS = pd.DataFrame(
    [
        (256265, "NIFTY 50", "INDICES", "EQ", "NSE"),
        (260105, "NIFTY BANK", "INDICES", "EQ", "NSE"),
        (738561, "RELIANCE", "NSE", "EQ", "NSE"),
        (128083204, "RELIANCE", "BSE", "EQ", "BSE"),
    ],
    columns=["instrument_token", "tradingsymbol", "segment", "instrument_type", "exchange"],
)


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeHttp:
    """Records every request; answers from a queue (the last answer repeats)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def _candles(*rows):
    return FakeResponse(200, {"status": "success", "data": {"candles": list(rows)}})


@pytest.fixture
def token():
    kite.save_token("FAKE+token/for==tests")
    return "FAKE+token/for==tests"


@pytest.fixture(autouse=True)
def _instruments(monkeypatch):
    monkeypatch.setattr(kite, "_instruments", lambda *a, **k: INSTRUMENTS)
    monkeypatch.setattr(kite, "_REQUEST_GAP_SECONDS", 0)


# ------------------------------------------------------------- read-only only


@pytest.mark.parametrize(
    "path",
    [
        "/oms/orders/regular",
        "/oms/orders",
        "/oms/portfolio/positions",
        "/oms/user/margins",
        "/oms/instruments/historical/256265/day/../../orders",
        "/oms/instruments/historical/256265/day?x=1",
    ],
)
def test_any_path_but_prices_and_profile_is_refused_before_sending(token, path):
    http = FakeHttp(_candles())
    with pytest.raises(PermissionError, match="only reads prices"):
        kite._get(path, session=http)
    assert http.calls == []


def test_the_kite_module_contains_no_way_to_send_anything_but_get() -> None:
    """Read the source itself: no POST/PUT/DELETE/PATCH call, no order paths."""
    tree = ast.parse(KITE_SOURCE.read_text())
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not calls & {"post", "put", "delete", "patch", "request"}, calls
    strings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    code_strings = [s for s in strings if s.startswith("/") or "oms/" in s]
    for s in code_strings:
        assert not any(w in s for w in ("order", "portfolio", "margin", "funds", "gtt")), s


def test_no_other_module_talks_to_zerodha() -> None:
    """`nlt/data/kite.py` is the only door. Everything else goes through it."""
    offenders = []
    for path in [*REPO.glob("nlt/**/*.py"), *REPO.glob("app/**/*.py"), *REPO.glob("scripts/*.py")]:
        if path == KITE_SOURCE:
            continue
        text = path.read_text()
        if "kite.zerodha.com" in text or "enctoken" in text.lower().replace("kite_enctoken", ""):
            offenders.append(str(path.relative_to(REPO)))
    assert offenders == []


# ----------------------------------------------------------------- the token


def test_the_token_lives_outside_the_repository() -> None:
    real = Path.home() / ".config" / "nlt" / "kite_enctoken"
    assert REPO not in real.parents


def test_a_saved_token_is_readable_only_by_its_owner(token) -> None:
    mode = stat.S_IMODE(kite.TOKEN_PATH.stat().st_mode)
    assert mode == 0o600
    assert kite.load_token() == token


@pytest.mark.parametrize(
    ("pasted", "stored"),
    [("  abc+def==  \n", "abc+def=="), ("enctoken abc+def==", "abc+def==")],
)
def test_a_pasted_token_is_cleaned_up(pasted, stored) -> None:
    kite.save_token(pasted)
    assert kite.load_token() == stored


@pytest.mark.parametrize("bad", ["", "   ", "two words"])
def test_something_that_is_not_a_token_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        kite.save_token(bad)
    assert kite.load_token() is None


def test_the_token_is_sent_only_in_the_header_never_in_the_address(token) -> None:
    """Addresses end up in logs and error messages; headers do not."""
    http = FakeHttp(_candles())
    kite._get("/oms/instruments/historical/256265/day", {"from": "x"}, session=http)
    call = http.calls[0]
    assert call["headers"]["Authorization"] == f"enctoken {token}"
    assert token not in call["url"]
    assert token not in str(call["params"])


def test_without_a_token_nothing_is_sent() -> None:
    http = FakeHttp(_candles())
    with pytest.raises(kite.KiteNotConnected):
        kite._get("/oms/instruments/historical/256265/day", session=http)
    assert http.calls == []


# ------------------------------------------------------------------ failures


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (
            FakeResponse(403, {"status": "error", "error_type": "TokenException"}),
            "KiteTokenExpired",
        ),
        (
            FakeResponse(200, {"status": "error", "error_type": "TokenException"}),
            "KiteTokenExpired",
        ),
        (FakeResponse(403, ValueError("an html login page")), "KiteTokenExpired"),
        (FakeResponse(500, {"status": "error", "message": "boom"}), "KiteUnavailable"),
        (FakeResponse(502, ValueError("not json")), "KiteUnavailable"),
    ],
)
def test_kite_failures_become_named_errors(token, response, error) -> None:
    with pytest.raises(getattr(kite, error)):
        kite._get("/oms/instruments/historical/256265/day", session=FakeHttp(response))


# -------------------------------------------------------------------- candles


def test_candles_are_read_in_indian_time_and_never_guessed(token) -> None:
    http = FakeHttp(
        _candles(
            ["2026-09-01T09:15:00+0530", 100, 101, 99, 100.5, 10],
            ["2026-09-01T09:20:00+0530", 100.5, 102, 100, 101.5, 12],
        )
    )
    bars = kite.KiteSource(http).bars("RELIANCE", "5m", dt.date(2026, 9, 1), dt.date(2026, 9, 1))
    assert str(bars.index.tz) == "Asia/Kolkata"
    assert bars.index[0] == pd.Timestamp("2026-09-01 09:15", tz="Asia/Kolkata")
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]


def test_candles_without_a_timezone_are_refused(token) -> None:
    http = FakeHttp(_candles(["2026-09-01T09:15:00", 100, 101, 99, 100.5, 10]))
    with pytest.raises(kite.KiteUnavailable, match="timezone"):
        kite.KiteSource(http).bars("NIFTY", "5m", dt.date(2026, 9, 1), dt.date(2026, 9, 1))


def test_long_ranges_are_split_into_requests_kite_accepts(token) -> None:
    http = FakeHttp(_candles(["2026-09-01T09:15:00+0530", 100, 101, 99, 100.5, 10]))
    kite.KiteSource(http).bars("NIFTY", "1m", dt.date(2026, 1, 1), dt.date(2026, 6, 30))
    spans = [
        (
            dt.date.fromisoformat(c["params"]["from"][:10]),
            dt.date.fromisoformat(c["params"]["to"][:10]),
        )
        for c in http.calls
    ]
    assert len(spans) >= 3
    assert all((b - a).days < 60 for a, b in spans)
    assert spans[0][0] == dt.date(2026, 1, 1) and spans[-1][1] == dt.date(2026, 6, 30)
    assert all(nxt[0] == prev[1] + dt.timedelta(days=1) for prev, nxt in zip(spans, spans[1:]))


@pytest.mark.parametrize(
    ("symbol", "expected"), [("NIFTY", 256265), ("BANKNIFTY", 260105), ("RELIANCE", 738561)]
)
def test_symbols_map_to_their_nse_instrument(symbol, expected) -> None:
    assert kite.instrument_token(symbol, INSTRUMENTS) == expected


def test_an_unknown_symbol_is_refused_not_guessed() -> None:
    with pytest.raises(kite.KiteUnavailable):
        kite.instrument_token("RELIANC", INSTRUMENTS)
