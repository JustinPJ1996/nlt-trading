"""The screens, driven in a real browser.

`test_api.py` proves the server sends the engine's own readback. This proves
the *screen* then shows it: every line, in order, with its words unchanged,
and that changing a box on the Build page shows the readback of the changed
strategy rather than leaving the old one up. A restyle that dropped a line or
kept a stale readback on screen would be the most serious kind of bug here,
and no server-side test can see it.

The front end is built into a temporary folder for this run -- never into
web/dist, which the live site serves.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from app import api
from nlt.store.db import Store

playwright_api = pytest.importorskip("playwright.sync_api")

LOGIN_ID = "tester"
PASSWORD = "correct horse"
RSI = "buy nifty when rsi cracks 30, target 2%, stop loss 1%"
WEB = Path(__file__).resolve().parent.parent / "web"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    if not shutil.which("pnpm"):
        pytest.skip("pnpm is not installed, so the screens cannot be built")
    out = tmp_path_factory.mktemp("web-dist")
    subprocess.run(
        [
            "pnpm",
            "exec",
            "vite",
            "build",
            "--outDir",
            str(out),
            "--emptyOutDir",
            "--logLevel",
            "error",
        ],
        cwd=WEB,
        check=True,
    )
    return out


@pytest.fixture
def site(built, monkeypatch, tmp_path):
    monkeypatch.setenv(api.auth.LOGIN_ID_ENV_VAR, LOGIN_ID)
    monkeypatch.setenv(api.auth.ENV_VAR, PASSWORD)
    monkeypatch.setattr(api, "WEB_DIST", built)
    app = api.create_app(Store(tmp_path / "web.sqlite"), secret="test-secret")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        assert time.monotonic() < deadline, "the test server did not start"
        time.sleep(0.05)
    # "localhost" rather than 127.0.0.1: the browser keeps the Secure session
    # cookie over plain http only for localhost.
    yield f"http://localhost:{port}"
    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def page(site):
    with playwright_api.sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page()
        # The ready-made strategy cards run real backtests; they are not what
        # this test is about, so the browser is told there are none.
        pg.route("**/api/library", lambda r: r.fulfill(status=503, json={"error": "Not in tests."}))
        pg.goto(site + "/")
        pg.get_by_label("Login ID").fill(LOGIN_ID)
        pg.get_by_label("Password").fill(PASSWORD)
        pg.keyboard.press("Enter")
        pg.locator("textarea").first.wait_for()
        yield pg
        browser.close()


def _screen_lines(pg) -> list[str]:
    """The readback as the screen shows it: title, headings and lines, in
    order, without the "You said" labels (the server's own words for where a
    line came from, laid out beside it)."""
    return pg.locator('[data-testid="readback"]').evaluate(
        """el => [...el.querySelectorAll('.readback-title, .readback-heading, .readback-line')]
                 .map(p => [...p.childNodes]
                   .filter(n => n.nodeType === Node.TEXT_NODE)
                   .map(n => n.textContent).join('').trim())"""
    )


def _server_lines(pg, site: str, text: str) -> list[str]:
    body = pg.request.post(site + "/api/translate", data={"text": text, "answers": {}}).json()
    readback = body["understood"]["readback"]
    return [line.strip() for line in readback.split("\n") if line.strip()]


def test_build_page_shows_the_engines_readback_word_for_word(page, site) -> None:
    page.locator("textarea").first.fill(RSI)
    page.get_by_role("button", name="Check my strategy").click()
    page.locator('[data-testid="readback"]').wait_for()

    expected = _server_lines(page, site, RSI)
    assert len(expected) > 5, "guard: the readback should have several lines"
    assert _screen_lines(page) == expected


def test_changing_the_stop_box_shows_the_new_readback(page, site) -> None:
    page.locator("textarea").first.fill(RSI)
    page.get_by_role("button", name="Check my strategy").click()
    readback = page.locator('[data-testid="readback"]')
    readback.wait_for()
    assert "1.5%" not in readback.inner_text()

    page.get_by_label("Stop loss (%)").fill("1.5")

    # The readback on screen must become the engine's readback of the edited
    # strategy -- not stay as it was, and not be patched up in the browser.
    page.wait_for_function(
        "() => document.querySelector('[data-testid=readback]').innerText.includes('1.5%')",
        timeout=10_000,
    )
    edited = page.request.post(
        site + "/api/adjust",
        data={
            "spec": page.evaluate(
                "() => JSON.parse(sessionStorage.getItem('sd-build')).translation.understood.spec"
            ),
            "edits": {"stop_pct": "1.5"},
        },
    ).json()
    expected = [line.strip() for line in edited["readback"].split("\n") if line.strip()]
    assert _screen_lines(page) == expected
