"""The Safety page's log, in words.

The database keeps each event's raw details -- a strategy's fingerprint, ids
-- for auditing. The screen must say what happened in plain English, so every
event the store can write needs its own wording, and none may fall through to
the raw "name: ..., hash: ..." dump that used to be shown.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from app import api
from nlt.store.db import Store

DB_SOURCE = Path(__file__).resolve().parent.parent / "nlt" / "store" / "db.py"
EVENTS = sorted(set(re.findall(r'_audit\(\s*conn,\s*"([a-z_]+)"', DB_SOURCE.read_text())))


def test_guard_the_event_list_was_found() -> None:
    assert len(EVENTS) >= 10, EVENTS


@pytest.mark.parametrize("event", EVENTS)
def test_every_event_the_store_writes_has_its_own_words(event) -> None:
    detail = {
        "name": "Buy NIFTY · RSI",
        "version": 2,
        "hash": "0b09f7b1f70951bd",
        "reason": "test",
        "mode": "paper",
        "status": "finished",
        "state": "backtested",
        "error": "no prices",
        "order_id": 7,
        "allow_unproven": True,
    }
    title, words = api._audit_words(event, detail)
    # The fallback lists raw "key: value" pairs; a worded event never does.
    assert not re.search(r"\b(?:" + "|".join(detail) + r"): ", words), words
    assert "0b09f7b1f70951bd" not in words
    assert "_" not in title + words


def test_the_safety_page_shows_words_not_raw_details(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(api.auth.LOGIN_ID_ENV_VAR, "tester")
    monkeypatch.setenv(api.auth.ENV_VAR, "pw")
    store = Store(tmp_path / "s.sqlite")
    from app import logic

    spec = logic.translate_text("buy nifty when rsi cracks 30, target 2%, stop loss 1%").spec
    store.save_strategy(spec, logic.describe_spec(spec))
    store.engage_paper_kill_switch("checking the page")
    client = TestClient(api.create_app(store, secret="s"), base_url="https://testserver")
    client.post("/api/login", json={"login_id": "tester", "password": "pw"})
    audit = client.get("/api/safety").json()["audit"]
    shown = " ".join(a["event"] + " " + a["detail"] for a in audit)
    assert "Strategy saved" in shown and f"“{spec.name}”" in shown
    assert "Reason given: checking the page." in shown
    assert "hash" not in shown
