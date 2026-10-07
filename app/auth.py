"""A password gate for the dashboard, for showing it over a public tunnel.

A `cloudflared` quick tunnel URL is public and unauthenticated: anyone who has
it, or guesses it, reaches whatever is behind it. The Safety page carries the
kill switch -- the control that stops everything trading -- so the dashboard
must not go onto the open internet without something in front of it.

Two deliberate choices:

**It fails closed.** With no password configured, `require_password` refuses to
render the dashboard at all rather than waving everyone through. The dangerous
failure here is an unprotected dashboard nobody noticed was unprotected, so a
missing password has to be loud and blocking, never silent and open.

**The password is never in the repo.** It is read from the environment, or from
`~/.nlt-dashboard-password` outside the repository tree. `JustinPJ1996/nlt-trading`
is a *public* GitHub repo -- a password committed to it is a password published,
and `tests/test_repo_hygiene.py` exists because this project has already been
bitten once by assuming what is and is not tracked.

This is a shared password for a demo, not a user system: everyone who has it is
the same anonymous viewer, and anyone shown it can pass it on. That is an
acceptable trade for showing work to colleagues for an afternoon. It is not
acceptable once a live broker session sits behind this screen -- at that point
this file should be replaced by real per-person sign-in, not extended.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

import streamlit as st

ENV_VAR = "NLT_DASHBOARD_PASSWORD"
PASSWORD_FILE = Path.home() / ".nlt-dashboard-password"

# A login ID as well as the password (Justin's choice, 2026-10-07), so a leaked
# password alone does not open the dashboard. Configured the same way, and
# just as fail-closed: no login ID set means nobody gets in.
LOGIN_ID_ENV_VAR = "NLT_DASHBOARD_LOGIN_ID"
LOGIN_ID_FILE = Path.home() / ".nlt-dashboard-login-id"

_STATE_KEY = "_auth_ok"


def configured_password(env: dict[str, str] | None = None, path: Path | None = None) -> str | None:
    """The expected password, or None if none is configured.

    The environment wins over the file so a one-off run can override without
    editing anything. Whitespace is stripped from both because a trailing
    newline in a file written by `echo` is invisible and would otherwise make
    every correct password fail.
    """
    env = os.environ if env is None else env
    path = PASSWORD_FILE if path is None else path

    raw = (env.get(ENV_VAR) or "").strip()
    if raw:
        return raw

    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        # Missing, unreadable, a directory -- all mean "not configured", which
        # the caller must treat as "refuse", never as "allow".
        return None

    return raw or None


def configured_login_id(env: dict[str, str] | None = None, path: Path | None = None) -> str | None:
    """The expected login ID, or None if none is configured. Same rules as the password."""
    env = os.environ if env is None else env
    raw = (env.get(LOGIN_ID_ENV_VAR) or "").strip()
    if raw:
        return raw
    try:
        raw = (LOGIN_ID_FILE if path is None else path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return raw or None


def login_matches(
    entered_id: str, entered_password: str, expected_id: str, expected_password: str
) -> bool:
    """Both must match. The ID ignores case and spaces at the ends; the password does not.

    Both are always compared, so a wrong ID and a wrong password take the same
    time and a guesser cannot tell which one was right.
    """
    if not expected_id:
        return False
    id_ok = hmac.compare_digest(
        entered_id.strip().lower().encode("utf-8"), expected_id.strip().lower().encode("utf-8")
    )
    password_ok = password_matches(entered_password, expected_password)
    return id_ok and password_ok


def password_matches(entered: str, expected: str) -> bool:
    """Constant-time comparison, so a wrong guess leaks nothing by timing."""
    if not expected:
        # Defensive: an empty expected password must never match anything,
        # including an empty entry. Without this, a blank config would open
        # the dashboard to anyone who submitted an empty form.
        return False
    return hmac.compare_digest(entered.encode("utf-8"), expected.encode("utf-8"))


def require_password() -> None:
    """Block the script until the viewer has entered the password.

    Call this as the very first thing in `main()`, before any page renders.
    On failure it calls `st.stop()`, which ends this script run -- nothing
    below it executes, so no page content is ever sent to an unauthenticated
    browser.
    """
    if st.session_state.get(_STATE_KEY):
        return

    expected = configured_password()
    expected_id = configured_login_id()

    if expected is None or expected_id is None:
        st.error(
            "This dashboard has no login ID or password set, so it will not start.\n\n"
            f"Set `{LOGIN_ID_ENV_VAR}` and `{ENV_VAR}`, or write them to "
            f"`{LOGIN_ID_FILE}` and `{PASSWORD_FILE}`, then reload."
        )
        st.stop()

    st.title("Trading Strategy Builder")
    st.caption("Enter your login ID and password to continue.")

    with st.form("login"):
        entered_id = st.text_input("Login ID")
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Open dashboard")

    if submitted:
        if login_matches(entered_id, entered, expected_id, expected):
            st.session_state[_STATE_KEY] = True
            st.rerun()
        else:
            st.error("Wrong login ID or password.")

    st.stop()
