"""Ready-made strategies (B4): cards with their own tested numbers, refreshed daily.

A card must never look better than its report: the numbers are a real run, the
holding return sits beside the strategy's, and serious warnings show on it.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from app import api, logic
from tests.conftest import NIFTY_FIXTURE_CUTOFF

PINNED = dt.date.fromisoformat(NIFTY_FIXTURE_CUTOFF)


@pytest.mark.parametrize("entry", logic.LIBRARY, ids=lambda e: e["title"])
def test_every_ready_made_strategy_reads_cleanly(entry) -> None:
    t = logic.translate_text(entry["sentence"])
    assert t.spec is not None, t.questions
    assert not t.unparsed
    assert t.period is not None
    assert t.spec.exit.has_stop


@pytest.mark.parametrize("entry", logic.LIBRARY, ids=lambda e: e["title"])
def test_every_card_is_a_real_run_with_trades(entry) -> None:
    card = logic.library_card(entry, PINNED)
    assert "error" not in card, card.get("error")
    assert card["trades"] >= 1
    assert card["holding_return"].endswith("%")
    assert isinstance(card["serious_warning"], bool)
    assert card["last"] == f"{dt.date(2026, 9, 25):%d %b %Y}"


def test_a_cards_numbers_are_the_reports_numbers() -> None:
    entry = logic.LIBRARY[0]
    card = logic.library_card(entry, PINNED)
    t = logic.translate_text(entry["sentence"])
    dates = logic.period_dates(t.period, PINNED)
    run = logic.backtest_spec(
        t.spec, start=dt.date.fromisoformat(dates["start"]), end=PINNED, translation=t
    )
    assert card["return"] == logic.format_signed_pct(run.comparison.strategy_return_pct)
    assert card["holding_return"] == logic.format_signed_pct(run.comparison.benchmark_return_pct)
    assert card["serious_warning"] == any(f.severity == "critical" for f in run.verdict.flags)


def _counting():
    calls: list[tuple[str, dt.date]] = []

    def compute(entry, tested_to):
        calls.append((entry["title"], tested_to))
        return {"title": entry["title"], "sentence": entry["sentence"], "trades": 1}

    return calls, compute


def test_cards_are_computed_once_a_day_and_tested_to_yesterday(tmp_path) -> None:
    path = tmp_path / "library.json"
    calls, compute = _counting()
    today = dt.date(2026, 10, 13)
    first = logic.library_cards(path, today, compute)
    assert first["tested_to"] == "2026-10-12"
    assert len(calls) == len(logic.LIBRARY)
    assert {d for _t, d in calls} == {dt.date(2026, 10, 12)}
    logic.library_cards(path, today, compute)
    assert len(calls) == len(logic.LIBRARY)  # same day: from disk
    logic.library_cards(path, today + dt.timedelta(days=1), compute)
    assert len(calls) == 2 * len(logic.LIBRARY)  # next day: fresh


def test_a_damaged_cache_is_recomputed_not_trusted(tmp_path) -> None:
    path = tmp_path / "library.json"
    path.write_text("{not json")
    calls, compute = _counting()
    out = logic.library_cards(path, dt.date(2026, 10, 13), compute)
    assert len(calls) == len(logic.LIBRARY)
    assert json.loads(path.read_text()) == out


def test_the_library_needs_a_login_and_serves_the_cards(client, monkeypatch, tmp_path) -> None:
    assert client.get("/api/library").status_code == 401
    _calls, compute = _counting()
    monkeypatch.setattr(logic, "library_card", compute)
    client.app.state.library_path = tmp_path / "library.json"
    client.post("/api/login", json={"login_id": "tester", "password": "correct horse"})
    body = client.get("/api/library").json()
    assert [c["title"] for c in body["cards"]] == [e["title"] for e in logic.LIBRARY]
    assert body["capital_text"] == "Rs 25,00,000"


@pytest.fixture
def client(monkeypatch, tmp_path):
    from starlette.testclient import TestClient

    from nlt.store.db import Store

    monkeypatch.setenv(api.auth.LOGIN_ID_ENV_VAR, "tester")
    monkeypatch.setenv(api.auth.ENV_VAR, "correct horse")
    app = api.create_app(Store(tmp_path / "web.sqlite"), secret="test-secret")
    return TestClient(app, base_url="https://testserver")


@pytest.mark.parametrize("entry", logic.LIBRARY, ids=lambda e: e["title"])
def test_every_card_has_enough_trades_to_judge(entry) -> None:
    """The cards were swapped on 2026-10-16 because five of six had too few
    trades to mean anything (12-22). Chosen by count, never by result."""
    card = logic.library_card(entry, PINNED)
    assert card["trades"] >= 30, card
