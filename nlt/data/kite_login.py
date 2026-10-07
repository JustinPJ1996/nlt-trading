"""Logging in to Kite by itself, so nobody has to paste a token every morning.

Adapted from the login half of the script Balajee shared on 2026-10-03
(`~/kite_api/kite_api.py`, kept outside the repository). It replays the Kite
website's own login -- user id and password, then the 6-digit authenticator
code -- and keeps the `enctoken` cookie that comes back, which `nlt/data/kite.py`
then uses exactly as it used a pasted token.

Justin chose this after being told the trade-offs: like the pasted token it is
against Zerodha's terms and can break without notice, and the saved password plus
authenticator secret are together the full keys to a REAL trading account --
more than a token, which at least expires.

So:

  * The login details live in `~/.config/nlt/kite_login.json`, outside the
    repository (which is public), readable only by this user. They are never
    printed, logged, or shown on screen again.
  * This module only logs in. It sends to the two login addresses and nothing
    else; `tests/test_kite_login.py` reads the source and fails otherwise.
  * It never hammers Zerodha. Repeated wrong passwords or codes can lock the
    account, and the paper job runs every minute. A rejected password stops all
    further attempts until the login is saved again; a rejected code is retried
    once, after a pause; a network failure is retried after a pause.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from nlt.data import kite

BASE_URL = "https://kite.zerodha.com"
LOGIN_URL = "https://kite.zerodha.com/api/login"
TWOFA_URL = "https://kite.zerodha.com/api/twofa"

CREDENTIALS_PATH = Path.home() / ".config" / "nlt" / "kite_login.json"
STATE_PATH = Path.home() / ".config" / "nlt" / "kite_login_state.json"

# After a failed attempt, wait this long before the next one.
RETRY_AFTER_SECONDS = 15 * 60
# A rejected authenticator code can be bad timing; this many in a row cannot.
MAX_CODE_REJECTIONS = 2

_BROWSER_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
}


# ------------------------------------------------------- authenticator codes


def _secret_from(value: str) -> str:
    """The base32 secret, from either the bare secret or an `otpauth://` link."""
    value = value.strip()
    if value.lower().startswith("otpauth://"):
        found = parse_qs(urlparse(value).query).get("secret")
        if not found or not found[0]:
            raise ValueError("that authenticator link has no secret in it")
        value = found[0]
    return value.replace(" ", "").upper()


def totp(secret: str, at: float | None = None, digits: int = 6) -> str:
    """The code an authenticator app shows for `secret` at time `at` (RFC 6238)."""
    padded = secret + "=" * (-len(secret) % 8)
    key = base64.b32decode(padded, casefold=True)
    counter = int((time.time() if at is None else at) // 30)
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    number = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(number % 10**digits).zfill(digits)


# ---------------------------------------------------------- the login details


def _write_private(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(payload, fh)


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def save_credentials(user_id: str, password: str, authenticator_secret: str) -> None:
    """Store the login where only this user can read it. Never echoes any of it.

    Saving also clears any earlier 'stop trying' decision, since new details are
    exactly what that decision was waiting for.
    """
    user_id = user_id.strip().upper()
    if not user_id or any(c.isspace() for c in user_id):
        raise ValueError("the Kite user ID looks wrong -- it is short, like AB1234")
    if not password:
        raise ValueError("the password is empty")
    secret = _secret_from(authenticator_secret)
    try:
        totp(secret)
    except (ValueError, TypeError):
        raise ValueError(
            "that doesn't look like an authenticator secret -- it is a long run of "
            "letters A-Z and digits 2-7, not the 6-digit code"
        ) from None
    if len(secret) < 16:
        raise ValueError(
            "that is too short for an authenticator secret -- it is not the 6-digit code"
        )
    _write_private(
        CREDENTIALS_PATH,
        {"user_id": user_id, "password": password, "totp_secret": secret},
    )
    STATE_PATH.unlink(missing_ok=True)


def load_credentials() -> dict | None:
    data = _read(CREDENTIALS_PATH)
    if not data or not all(data.get(k) for k in ("user_id", "password", "totp_secret")):
        return None
    return data


def has_credentials() -> bool:
    return load_credentials() is not None


def forget_credentials() -> None:
    CREDENTIALS_PATH.unlink(missing_ok=True)
    STATE_PATH.unlink(missing_ok=True)


# -------------------------------------------------------------- logging in


def _fail(state: dict, message: str, *, stop: bool, now: float) -> kite.KiteLoginFailed:
    state.update(last_failure=now, message=message, stopped=stop)
    _write_private(STATE_PATH, state)
    if stop:
        return kite.KiteLoginFailed(
            f"{message} Automatic login has stopped so Zerodha does not lock the "
            "account. Check the details and save your Kite login again."
        )
    return kite.KiteLoginFailed(f"{message} Will try again in 15 minutes.")


def _json(resp) -> dict:
    try:
        body = resp.json()
    except ValueError:
        body = None
    return body if isinstance(body, dict) else {}


def login(http=None, now: float | None = None) -> None:
    """Log in to Kite and save the fresh token. Raises a `KiteError` if it cannot."""
    creds = load_credentials()
    if creds is None:
        raise kite.KiteNotConnected("no Kite login saved")
    now = time.time() if now is None else now

    state = _read(STATE_PATH) or {}
    if state.get("stopped"):
        raise kite.KiteLoginFailed(
            f"{state.get('message', 'Kite refused the login.')} Automatic login has "
            "stopped. Check the details and save your Kite login again."
        )
    if state.get("last_failure") and now - state["last_failure"] < RETRY_AFTER_SECONDS:
        wait = int((RETRY_AFTER_SECONDS - (now - state["last_failure"])) // 60) + 1
        raise kite.KiteLoginFailed(
            f"{state.get('message', 'The last Kite login failed.')} "
            f"Trying again in about {wait} minutes."
        )

    import requests

    session = http or requests.Session()
    if http is None:
        session.headers.update(_BROWSER_HEADERS)
    try:
        session.get(BASE_URL, timeout=20)
        step1 = session.post(
            LOGIN_URL,
            data={"user_id": creds["user_id"], "password": creds["password"]},
            timeout=20,
        )
        body1 = _json(step1)
        request_id = (body1.get("data") or {}).get("request_id")
        if body1.get("status") != "success" or not request_id:
            reason = body1.get("message") or f"answer {step1.status_code}"
            raise _fail(
                state, f"Kite refused the user ID or password ({reason}).", stop=True, now=now
            )

        step2 = session.post(
            TWOFA_URL,
            data={
                "user_id": creds["user_id"],
                "request_id": request_id,
                "twofa_value": totp(creds["totp_secret"], now),
                "twofa_type": "totp",
                "skip_session": "",
            },
            timeout=20,
        )
        body2 = _json(step2)
        if body2.get("status") != "success":
            reason = body2.get("message") or f"answer {step2.status_code}"
            rejections = int(state.get("code_rejections", 0)) + 1
            state["code_rejections"] = rejections
            raise _fail(
                state,
                f"Kite refused the authenticator code ({reason}).",
                stop=rejections >= MAX_CODE_REJECTIONS,
                now=now,
            )
        token = step2.cookies.get("enctoken")
        if not token:
            raise _fail(state, "Kite accepted the login but sent no token.", stop=False, now=now)
    except requests.RequestException as exc:
        raise _fail(
            state,
            f"Could not reach Kite to log in ({exc.__class__.__name__}).",
            stop=False,
            now=now,
        ) from None

    kite.save_token(token)
    STATE_PATH.unlink(missing_ok=True)
