"""Tests for `app/api.py` -- the web front end's door into the engine.

What matters here is not layout (the TypeScript screens are checked by driving
them in a browser) but the three promises the API makes:

1. Nothing answers without a login, and a missing login configuration keeps
   everybody out rather than letting everybody in.
2. What is backtested is the spec the user confirmed, not a fresh reading of
   their sentence -- the AI reader can answer differently twice.
3. The readback the browser shows is the engine's own text.

Most tests below are "refuses" tests, for the same reason as `test_auth.py`:
the bug worth catching is a gate that falls open.
"""

from __future__ import annotations

import pandas as pd
import pytest
from starlette.routing import Route
from starlette.testclient import TestClient

from app import api, logic
from nlt.spec.models import StrategySpec
from nlt.store.db import Store
from nlt.translate.readback import describe

LOGIN_ID = "tester"
PASSWORD = "correct horse"
RSI = "buy nifty when rsi cracks 30, target 2%, stop loss 1%"
SMA = "buy nifty when 20 sma crosses above 50 sma, target 3%, stop loss 1.5%"
# Pinned dates: the cached NIFTY history grows every day, and a test anchored
# on today would change its answer as it does.
WINDOW = {"start": "2021-01-01", "end": "2024-12-31"}


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv(api.auth.LOGIN_ID_ENV_VAR, LOGIN_ID)
    monkeypatch.setenv(api.auth.ENV_VAR, PASSWORD)


@pytest.fixture
def client(configured, tmp_path) -> TestClient:
    app = api.create_app(Store(tmp_path / "web.sqlite"), secret="test-secret")
    # https: the session cookie is marked Secure, as it must be behind a tunnel.
    return TestClient(app, base_url="https://testserver")


@pytest.fixture
def authed(client) -> TestClient:
    r = client.post("/api/login", json={"login_id": LOGIN_ID, "password": PASSWORD})
    assert r.status_code == 200
    return client


def _spec_json(text: str) -> dict:
    return logic.translate_text(text).spec.model_dump(mode="json")


# --------------------------------------------------------------------- login


def _api_routes(app) -> list[tuple[str, str]]:
    out = []
    for route in app.routes:
        if isinstance(route, Route) and route.path.startswith("/api/"):
            path = route.path.replace("{sid:int}", "1").replace("{rid:int}", "1")
            path = path.replace("{which}", "paper").replace("{action}", "engage")
            path = path.replace("{rest:path}", "anything")
            for method in route.methods - {"HEAD"}:
                out.append((method, path))
    return out


def test_every_api_route_but_login_refuses_without_a_session(client) -> None:
    routes = _api_routes(client.app)
    # Guard the guard: if route discovery broke, this test would pass vacuously.
    assert len(routes) >= 18
    open_routes = {("POST", "/api/login"), ("POST", "/api/logout")}
    for method, path in routes:
        if (method, path) in open_routes:
            continue
        r = client.request(method, path, json={"confirm": True})
        assert r.status_code == 401, f"{method} {path} answered {r.status_code} without a login"
        assert "log in" in r.json()["error"].lower()


def test_logged_out_cannot_flip_a_kill_switch(client) -> None:
    client.post("/api/safety/paper/engage", json={"confirm": True})
    assert not client.app.state.store.paper_kill_switch_engaged()


def test_right_login_opens_and_logout_closes(client) -> None:
    assert client.get("/api/me").status_code == 401
    r = client.post("/api/login", json={"login_id": " TESTER ", "password": PASSWORD})
    assert r.status_code == 200
    assert client.get("/api/me").status_code == 200
    client.post("/api/logout", json={})
    assert client.get("/api/me").status_code == 401


@pytest.mark.parametrize(
    "login_id, password",
    [(LOGIN_ID, "wrong"), ("someone", PASSWORD), ("", ""), (LOGIN_ID, "")],
)
def test_wrong_login_is_refused(client, login_id, password) -> None:
    r = client.post("/api/login", json={"login_id": login_id, "password": password})
    assert r.status_code == 401
    assert client.get("/api/options").status_code == 401


def test_no_login_configured_means_nobody_gets_in(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(api.auth.LOGIN_ID_ENV_VAR, raising=False)
    monkeypatch.delenv(api.auth.ENV_VAR, raising=False)
    monkeypatch.setattr(api.auth, "LOGIN_ID_FILE", tmp_path / "absent-id")
    monkeypatch.setattr(api.auth, "PASSWORD_FILE", tmp_path / "absent-pw")
    c = TestClient(api.create_app(Store(tmp_path / "s.sqlite"), secret="x"), base_url="https://t")
    for login_id, password in [("", ""), ("anyone", "anything")]:
        r = c.post("/api/login", json={"login_id": login_id, "password": password})
        assert r.status_code == 503
    assert c.get("/api/options").status_code == 401


def test_repeated_wrong_logins_lock_out_even_the_right_password(client) -> None:
    for _ in range(api.MAX_FAILURES):
        client.post("/api/login", json={"login_id": LOGIN_ID, "password": "guess"})
    r = client.post("/api/login", json={"login_id": LOGIN_ID, "password": PASSWORD})
    assert r.status_code == 429
    assert client.get("/api/me").status_code == 401


def test_one_wrong_login_does_not_lock_out(client) -> None:
    client.post("/api/login", json={"login_id": LOGIN_ID, "password": "typo"})
    r = client.post("/api/login", json={"login_id": LOGIN_ID, "password": PASSWORD})
    assert r.status_code == 200


def test_session_cookie_is_httponly_strict_and_secure(client) -> None:
    r = client.post("/api/login", json={"login_id": LOGIN_ID, "password": PASSWORD})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    assert "secure" in cookie


def test_a_forged_session_cookie_is_refused(configured, tmp_path) -> None:
    """A cookie signed with a different secret must not open anything."""
    store = Store(tmp_path / "f.sqlite")
    other = TestClient(api.create_app(store, secret="attacker"), base_url="https://t")
    other.post("/api/login", json={"login_id": LOGIN_ID, "password": PASSWORD})
    real = TestClient(api.create_app(store, secret="real"), base_url="https://t")
    real.cookies = other.cookies
    assert real.get("/api/me").status_code == 401


def test_writes_must_be_json(authed) -> None:
    """A plain form post from another website must not be able to act."""
    r = authed.post(
        "/api/safety/paper/engage",
        content="confirm=true",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 400
    assert not authed.app.state.store.paper_kill_switch_engaged()


# ------------------------------------------------------------ the readback


def test_translate_returns_the_engines_own_readback(authed) -> None:
    body = authed.post("/api/translate", json={"text": RSI}).json()
    spec = StrategySpec.model_validate(body["understood"]["spec"])
    assert body["understood"]["readback"] == describe(spec)
    assert body["understood"]["readback"] == describe(logic.translate_text(RSI).spec)


def test_a_missing_stop_loss_is_a_question_with_nothing_filled_in(authed) -> None:
    body = authed.post("/api/translate", json={"text": "buy nifty when rsi cracks 30"}).json()
    assert body["understood"] is None
    assert "exit.stop_pct" in [q["field"] for q in body["questions"]]
    assert body["answers"] == {}


def test_answering_with_a_percent_sign_is_understood(authed) -> None:
    body = authed.post(
        "/api/translate",
        json={"text": "buy nifty when rsi cracks 30", "answers": {"exit.stop_pct": "1%"}},
    ).json()
    assert body["understood"] is not None
    assert body["understood"]["spec"]["exit"]["stop_pct"] == 1.0


# ---------------------------------------------------- what runs is what was confirmed


def test_backtest_runs_the_confirmed_spec_and_never_rereads_the_sentence(
    authed, monkeypatch
) -> None:
    """The readback the user said yes to is the strategy that gets tested.

    The confirmed spec is the RSI strategy, but its `description` is made to
    say something else entirely. If the backtest re-read the sentence it would
    test the SMA strategy; it must test the RSI one, and report doing so.
    """
    confirmed = _spec_json(RSI)
    confirmed["description"] = SMA

    def no_rereading(*_a, **_k):
        raise AssertionError("the backtest re-read the sentence")

    monkeypatch.setattr(logic, "translate_text", no_rereading)
    r = authed.post(
        "/api/backtest",
        json={"spec": confirmed, "capital": 100000, "cost_model": "Index futures", **WINDOW},
    )
    assert r.status_code == 200, r.json()
    body = r.json()
    tested = StrategySpec.model_validate(body["tested"]["spec"])
    assert tested == StrategySpec.model_validate(confirmed)
    assert body["tested"]["readback"] == describe(tested)
    assert "RSI" in body["tested"]["readback"]


def test_backtest_result_matches_the_engine(authed) -> None:
    spec = StrategySpec.model_validate(_spec_json(RSI))
    body = authed.post(
        "/api/backtest",
        json={"spec": _spec_json(RSI), "capital": 100000, "cost_model": "Index futures", **WINDOW},
    ).json()
    direct = logic.backtest_spec(
        spec,
        capital=100000,
        cost_model_label="Index futures",
        start=pd.Timestamp(WINDOW["start"]).date(),
        end=pd.Timestamp(WINDOW["end"]).date(),
    )
    assert body["verdict"]["summary"] == direct.verdict.summary
    assert body["verdict"]["passed"] == direct.verdict.passed
    assert len(body["trades"]) == len(direct.backtest.trades)
    assert body["headline"]["strategy_return"] == logic.format_signed_pct(
        direct.comparison.strategy_return_pct
    )
    assert [f["headline"] for f in body["verdict"]["flags"]] == [
        f.headline for f in logic.flags_by_severity(direct.verdict.flags)
    ]


@pytest.mark.parametrize(
    "change, words",
    [
        ({"spec": {"nonsense": True}}, "could not be read back"),
        ({"capital": 10}, "between"),
        ({"cost_model": "Free"}, "costs"),
        ({"start": "2024-01-01", "end": "2023-01-01"}, "before"),
        ({"start": "not a date"}, "not a real date"),
    ],
)
def test_bad_backtest_requests_are_refused_in_words(authed, change, words) -> None:
    body = {"spec": _spec_json(RSI), "capital": 100000, "cost_model": "Index futures", **WINDOW}
    r = authed.post("/api/backtest", json={**body, **change})
    assert r.status_code == 400
    assert words in r.json()["error"]


def test_saving_records_the_backtest_that_was_shown(authed) -> None:
    body = authed.post(
        "/api/backtest",
        json={"spec": _spec_json(RSI), "capital": 250000, "cost_model": "Index futures", **WINDOW},
    ).json()
    saved = authed.post("/api/strategies", json={"result_id": body["result_id"]}).json()
    listed = authed.get("/api/strategies").json()["strategies"]
    assert [s["id"] for s in listed] == [saved["strategy_id"]]
    (run,) = listed[0]["runs"]
    assert run["kind"] == "Backtest" and run["status"] == "Finished"
    assert run["capital_text"] == "Rs 2,50,000"
    assert run["return_text"] == body["headline"]["strategy_return"]


def test_saving_an_unknown_result_is_refused(authed) -> None:
    r = authed.post("/api/strategies", json={"result_id": "nope"})
    assert r.status_code == 400
    assert authed.get("/api/strategies").json()["strategies"] == []


# -------------------------------------------------------------- kill switches


def test_engaging_a_kill_switch_needs_explicit_confirmation(authed) -> None:
    store = authed.app.state.store
    for body in [{}, {"confirm": "yes"}, {"confirm": 1}]:
        assert authed.post("/api/safety/live/engage", json=body).status_code == 400
    assert not store.kill_switch_engaged()
    assert authed.post("/api/safety/live/engage", json={"confirm": True}).json()["live"] is True
    assert store.kill_switch_engaged()
    assert not store.paper_kill_switch_engaged()


def test_releasing_a_kill_switch_works(authed) -> None:
    store = authed.app.state.store
    authed.post("/api/safety/paper/engage", json={"confirm": True})
    assert store.paper_kill_switch_engaged()
    assert authed.post("/api/safety/paper/release", json={}).json()["paper"] is False
    assert not store.paper_kill_switch_engaged()


# ---------------------------------------------------------------- the chart


def test_chart_downsampling_keeps_the_deepest_drawdown() -> None:
    """The chart must not disagree with the 'biggest drop' printed beside it.

    One sharp dip in a long series is exactly the bar an every-n-th sample skips.
    """
    idx = pd.date_range("2020-01-01", periods=10_000, freq="h", tz="Asia/Kolkata")
    strategy = pd.Series(100_000.0, index=idx)
    strategy.iloc[5_003] = 60_000.0  # a one-bar 40% drop
    bench = pd.Series(range(10_000), index=idx, dtype=float) + 100_000
    points = api._chart(strategy, bench)
    assert len(points) <= api.MAX_CHART_POINTS
    assert min(p["sd"] for p in points) == pytest.approx(-40.0)
    assert points[-1]["t"] == idx[-1].isoformat()
    assert points[-1]["s"] == pytest.approx(100_000.0)


def test_short_charts_are_sent_whole() -> None:
    idx = pd.date_range("2024-01-01", periods=50, freq="D", tz="Asia/Kolkata")
    s = pd.Series(range(100, 150), index=idx, dtype=float)
    assert len(api._chart(s, s)) == 50
