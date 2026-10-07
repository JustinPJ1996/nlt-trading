"""Logging in to Kite by itself: careful with the login, and never hammering Zerodha.

`conftest._no_live_kite` points the saved login and token at empty temporary
paths, so nothing here can use the real login on this machine. HTTP is a fake
object; nothing reaches Zerodha.
"""

from __future__ import annotations

import ast
import stat
from pathlib import Path

import pytest

from nlt.data import kite, kite_login

LOGIN_SOURCE = Path(kite_login.__file__)
REPO = Path(__file__).resolve().parents[1]

# RFC 6238's own test secret ("12345678901234567890"), in base32.
RFC_SECRET = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"
NOW = 1_800_000_000.0


class FakeResponse:
    def __init__(self, status, body, cookies=None):
        self.status_code = status
        self._body = body
        self.cookies = cookies or {}

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


STEP1_OK = FakeResponse(200, {"status": "success", "data": {"request_id": "req-1"}})
STEP2_OK = FakeResponse(200, {"status": "success", "data": {}}, {"enctoken": "fresh+token=="})
BAD_PASSWORD = FakeResponse(400, {"status": "error", "message": "Invalid password"})
BAD_CODE = FakeResponse(400, {"status": "error", "message": "Invalid TOTP"})


class FakeSession:
    """Answers the two login POSTs in order; records everything sent."""

    def __init__(self, *posts, error=None):
        self.posts = list(posts)
        self.error = error
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(("GET", url, None))
        if self.error:
            raise self.error
        return FakeResponse(200, {})

    def post(self, url, data=None, **kw):
        self.calls.append(("POST", url, data))
        return self.posts.pop(0)


@pytest.fixture
def saved():
    kite_login.save_credentials("ab1234", "hunter2", RFC_SECRET)


# ------------------------------------------------------- authenticator codes


@pytest.mark.parametrize(
    ("at", "code"),
    [(59, "287082"), (1111111109, "081804"), (1234567890, "005924"), (2000000000, "279037")],
)
def test_codes_match_the_published_standard(at, code) -> None:
    assert kite_login.totp(RFC_SECRET, at) == code


def test_an_authenticator_link_is_accepted_as_well_as_the_bare_secret() -> None:
    link = f"otpauth://totp/Kite:AB1234?secret={RFC_SECRET}&issuer=Kite"
    kite_login.save_credentials("AB1234", "pw", link)
    assert kite_login.load_credentials()["totp_secret"] == RFC_SECRET


@pytest.mark.parametrize("bad", ["123456", "not a secret!", "", "otpauth://totp/x?issuer=Kite"])
def test_something_that_is_not_an_authenticator_secret_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        kite_login.save_credentials("AB1234", "pw", bad)
    assert not kite_login.has_credentials()


# ---------------------------------------------------------- the login details


def test_the_login_lives_outside_the_repository() -> None:
    real = Path.home() / ".config" / "nlt" / "kite_login.json"
    assert REPO not in real.parents


def test_a_saved_login_is_readable_only_by_its_owner(saved) -> None:
    assert stat.S_IMODE(kite_login.CREDENTIALS_PATH.stat().st_mode) == 0o600
    assert kite_login.load_credentials()["user_id"] == "AB1234"


def test_forgetting_the_login_removes_it(saved) -> None:
    kite_login.forget_credentials()
    assert not kite_login.has_credentials()
    assert not kite.is_connected()


def test_a_saved_login_counts_as_connected(saved) -> None:
    assert kite.load_token() is None
    assert kite.is_connected()


# -------------------------------------------------------------- logging in


def test_a_good_login_saves_the_fresh_token(saved) -> None:
    http = FakeSession(STEP1_OK, STEP2_OK)
    kite_login.login(http, now=NOW)
    assert kite.load_token() == "fresh+token=="
    _, (_, url1, data1), (_, url2, data2) = http.calls
    assert url1 == kite_login.LOGIN_URL and data1 == {"user_id": "AB1234", "password": "hunter2"}
    assert url2 == kite_login.TWOFA_URL
    assert data2["request_id"] == "req-1"
    assert data2["twofa_value"] == kite_login.totp(RFC_SECRET, NOW)


def test_a_wrong_password_stops_all_further_attempts(saved) -> None:
    with pytest.raises(kite.KiteLoginFailed, match="stopped"):
        kite_login.login(FakeSession(BAD_PASSWORD), now=NOW)
    later = FakeSession(STEP1_OK, STEP2_OK)
    with pytest.raises(kite.KiteLoginFailed, match="stopped"):
        kite_login.login(later, now=NOW + 24 * 3600)
    assert later.calls == []
    assert kite.load_token() is None


def test_saving_the_login_again_lifts_the_stop(saved) -> None:
    with pytest.raises(kite.KiteLoginFailed):
        kite_login.login(FakeSession(BAD_PASSWORD), now=NOW)
    kite_login.save_credentials("AB1234", "the-right-one", RFC_SECRET)
    kite_login.login(FakeSession(STEP1_OK, STEP2_OK), now=NOW + 1)
    assert kite.load_token() == "fresh+token=="


def test_a_rejected_code_waits_then_tries_once_more_then_stops(saved) -> None:
    with pytest.raises(kite.KiteLoginFailed, match="15 minutes"):
        kite_login.login(FakeSession(STEP1_OK, BAD_CODE), now=NOW)
    too_soon = FakeSession(STEP1_OK, STEP2_OK)
    with pytest.raises(kite.KiteLoginFailed):
        kite_login.login(too_soon, now=NOW + 60)
    assert too_soon.calls == []
    with pytest.raises(kite.KiteLoginFailed, match="stopped"):
        kite_login.login(FakeSession(STEP1_OK, BAD_CODE), now=NOW + 16 * 60)
    never = FakeSession(STEP1_OK, STEP2_OK)
    with pytest.raises(kite.KiteLoginFailed):
        kite_login.login(never, now=NOW + 24 * 3600)
    assert never.calls == []


def test_a_network_failure_waits_but_does_not_stop(saved) -> None:
    import requests

    with pytest.raises(kite.KiteLoginFailed, match="15 minutes"):
        kite_login.login(FakeSession(error=requests.ConnectionError()), now=NOW)
    with pytest.raises(kite.KiteLoginFailed):
        kite_login.login(FakeSession(STEP1_OK, STEP2_OK), now=NOW + 60)
    kite_login.login(FakeSession(STEP1_OK, STEP2_OK), now=NOW + 16 * 60)
    assert kite.load_token() == "fresh+token=="


def test_without_a_saved_login_nothing_is_sent() -> None:
    http = FakeSession(STEP1_OK, STEP2_OK)
    with pytest.raises(kite.KiteNotConnected):
        kite_login.login(http, now=NOW)
    assert http.calls == []


def test_the_password_never_appears_in_an_error(saved) -> None:
    with pytest.raises(kite.KiteLoginFailed) as caught:
        kite_login.login(FakeSession(BAD_PASSWORD), now=NOW)
    assert "hunter2" not in str(caught.value)
    assert RFC_SECRET not in str(caught.value)


# ------------------------------------------- the price reader logs in by itself


class FakeHttp:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.tokens = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.tokens.append(headers["Authorization"])
        return self.responses.pop(0)


PROFILE = FakeResponse(200, {"status": "success", "data": {"user_id": "AB1234"}})
EXPIRED = FakeResponse(403, {"status": "error", "error_type": "TokenException"})


@pytest.fixture
def fake_login(monkeypatch):
    logins = []

    def login():
        logins.append(1)
        kite.save_token(f"token-{len(logins)}")

    monkeypatch.setattr(kite_login, "login", login)
    return logins


def test_an_expired_token_is_replaced_by_logging_in(saved, fake_login) -> None:
    kite.save_token("stale")
    http = FakeHttp(EXPIRED, PROFILE)
    assert kite._get("/oms/user/profile", session=http)["user_id"] == "AB1234"
    assert http.tokens == ["enctoken stale", "enctoken token-1"]
    assert fake_login == [1]


def test_no_token_yet_means_log_in_first(saved, fake_login) -> None:
    http = FakeHttp(PROFILE)
    kite._get("/oms/user/profile", session=http)
    assert http.tokens == ["enctoken token-1"]


def test_a_token_kite_keeps_rejecting_does_not_cause_a_loop(saved, fake_login) -> None:
    kite.save_token("stale")
    http = FakeHttp(EXPIRED, EXPIRED, PROFILE)
    with pytest.raises(kite.KiteTokenExpired):
        kite._get("/oms/user/profile", session=http)
    assert fake_login == [1]
    assert len(http.tokens) == 2


def test_a_fresh_token_kite_rejects_is_not_followed_by_a_second_login(saved, fake_login) -> None:
    with pytest.raises(kite.KiteTokenExpired):
        kite._get("/oms/user/profile", session=FakeHttp(EXPIRED, PROFILE))
    assert fake_login == [1]


def test_without_a_saved_login_an_expired_token_is_just_reported(fake_login) -> None:
    kite.save_token("stale")
    with pytest.raises(kite.KiteTokenExpired):
        kite._get("/oms/user/profile", session=FakeHttp(EXPIRED))
    assert fake_login == []


def test_a_disallowed_path_is_refused_before_any_login(saved, fake_login) -> None:
    with pytest.raises(PermissionError):
        kite._get("/oms/orders/regular", session=FakeHttp(PROFILE))
    assert fake_login == []


# ------------------------------------------------------- it only logs in


def test_the_login_module_sends_only_to_the_login_addresses() -> None:
    """Read the source: every Zerodha address in it is one of the three login ones,
    and it contains nothing that could place or change an order."""
    tree = ast.parse(LOGIN_SOURCE.read_text())
    strings = [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    addresses = {s for s in strings if "zerodha.com" in s or "kite.trade" in s}
    assert addresses == {
        "https://kite.zerodha.com",
        "https://kite.zerodha.com/api/login",
        "https://kite.zerodha.com/api/twofa",
    }
    for s in strings:
        if s.startswith("/") or "oms/" in s:
            raise AssertionError(f"unexpected path in kite_login: {s!r}")
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert not calls & {"put", "delete", "patch", "request"}, calls
