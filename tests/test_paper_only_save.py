"""A strategy with a strike the user named can only be paper traded.

It cannot be backtested, so it is saved without a backtest; every other
strategy must still be backtested before it can be saved.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from app import api, logic
from nlt.spec.models import (
    Compare,
    Const,
    ExitRules,
    Instrument,
    Ref,
    RiskLimits,
    Schedule,
    Sizing,
    StrategySpec,
)
from nlt.store.db import Store

LOGIN_ID = "tester"
PASSWORD = "correct horse"
NAMED = (
    "Buy the NIFTY weekly 24000 call when RSI crosses above 60 on 15 minute candles. "
    "Stop 30%, target 60% on the premium. Square off at 15:15."
)
ORDINARY = "buy nifty when rsi cracks 30, target 2%, stop loss 1%"


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("NLT_LLM_DISABLED", "1")
    monkeypatch.setenv(api.auth.LOGIN_ID_ENV_VAR, LOGIN_ID)
    monkeypatch.setenv(api.auth.ENV_VAR, PASSWORD)


@pytest.fixture
def client(tmp_path) -> TestClient:
    app = api.create_app(Store(tmp_path / "web.sqlite"), secret="test-secret")
    return TestClient(app, base_url="https://testserver")


@pytest.fixture
def authed(client) -> TestClient:
    r = client.post("/api/login", json={"login_id": LOGIN_ID, "password": PASSWORD})
    assert r.status_code == 200
    return client


def _named_spec() -> StrategySpec:
    spec = StrategySpec(
        name="named strike",
        description="named strike",
        instrument=Instrument(
            symbol="NIFTY",
            timeframe="15m",
            trade_as="option",
            option_type="CE",
            strike_offset=0,
            expiry="nearest_weekly",
        ),
        entry=Compare(op="gt", left=Ref(name="close"), right=Const(value=21500.0)),
        exit=ExitRules(stop_pct=30.0, target_pct=60.0),
        sizing=Sizing(mode="fixed_lots", lots=1),
        risk=RiskLimits(max_lots=2, max_position_pct=10.0),
        schedule=Schedule(intraday=True),
    )
    return spec.model_copy(
        update={"instrument": spec.instrument.model_copy(update={"strike": 21600})}
    )


def test_a_named_strike_is_saved_without_a_backtest_and_can_start_for_paper(authed) -> None:
    spec = _named_spec()
    assert logic.backtest_problem(spec) is not None
    r = authed.post("/api/strategies/untested", json={"spec": spec.model_dump(mode="json")})
    assert r.status_code == 200, r.text
    saved = r.json()
    listed = authed.get("/api/strategies").json()["strategies"]
    assert [s["id"] for s in listed] == [saved["strategy_id"]]
    assert listed[0]["runs"] == []
    startable = authed.get("/api/paper").json()["startable"]
    assert [(x["id"], x["problem"]) for x in startable] == [(saved["strategy_id"], None)]


def test_an_ordinary_strategy_must_still_be_backtested_first(authed) -> None:
    spec = logic.translate_text(ORDINARY).spec
    assert spec is not None
    r = authed.post("/api/strategies/untested", json={"spec": spec.model_dump(mode="json")})
    assert r.status_code == 400
    assert "backtest" in r.json()["error"].lower()
    assert authed.get("/api/strategies").json()["strategies"] == []


def test_saving_untested_needs_a_login(client) -> None:
    spec = _named_spec()
    r = client.post("/api/strategies/untested", json={"spec": spec.model_dump(mode="json")})
    assert r.status_code == 401


def test_translate_says_a_named_strike_cannot_be_backtested(authed) -> None:
    u = authed.post("/api/translate", json={"text": NAMED}).json()["understood"]
    assert u is not None
    assert "can't be backtested" in u["backtest_problem"]
    assert "24,000" in u["backtest_problem"]


def test_translate_has_no_backtest_problem_for_an_ordinary_strategy(authed) -> None:
    u = authed.post("/api/translate", json={"text": ORDINARY}).json()["understood"]
    assert u["backtest_problem"] is None
