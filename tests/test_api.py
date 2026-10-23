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

from typing import cast

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
    spec = logic.translate_text(text).spec
    assert spec is not None
    return spec.model_dump(mode="json")


# --------------------------------------------------------------------- login


def _api_routes(app) -> list[tuple[str, str]]:
    out = []
    for route in app.routes:
        if isinstance(route, Route) and route.path.startswith("/api/"):
            path = route.path.replace("{sid:int}", "1").replace("{rid:int}", "1")
            path = path.replace("{oid:int}", "1").replace("{pid:int}", "1")
            path = path.replace("{nid:int}", "1")
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
    assert body["understood"]["readback"] == describe(
        cast(StrategySpec, logic.translate_text(RSI).spec)
    )


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
        start=cast(pd.Timestamp, pd.Timestamp(WINDOW["start"])).date(),
        end=cast(pd.Timestamp, pd.Timestamp(WINDOW["end"])).date(),
    )
    assert direct.verdict is not None
    assert direct.backtest is not None
    assert direct.comparison is not None
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
        ({"slippage": {"kind": "pct", "value": -1}}, "negative"),
        ({"slippage": {"kind": "pct", "value": 50}}, "5"),
        ({"slippage": {"kind": "rs", "value": 1000}}, "100"),
        ({"slippage": {"kind": "bps", "value": 1}}, "percentage or a rupee"),
        ({"slippage": {"kind": "pct", "value": "not a number"}}, "must be a number"),
        ({"slippage": "not a dict"}, "kind and an amount"),
    ],
)
def test_bad_backtest_requests_are_refused_in_words(authed, change, words) -> None:
    body = {"spec": _spec_json(RSI), "capital": 100000, "cost_model": "Index futures", **WINDOW}
    r = authed.post("/api/backtest", json={**body, **change})
    assert r.status_code == 400
    assert words in r.json()["error"]


def test_backtest_defaults_to_zero_slippage_and_says_so(authed) -> None:
    body = authed.post(
        "/api/backtest",
        json={"spec": _spec_json(RSI), "capital": 100000, "cost_model": "Index futures", **WINDOW},
    ).json()
    assert body["params"]["slippage"] == {"kind": "pct", "value": 0.0}
    assert "No slippage assumed" in body["params"]["slippage_text"]
    assert any("No slippage assumed" in a for a in body["assumptions"])


def test_a_chosen_slippage_is_accepted_and_stated(authed) -> None:
    body = authed.post(
        "/api/backtest",
        json={
            "spec": _spec_json(RSI),
            "capital": 100000,
            "cost_model": "Index futures",
            "slippage": {"kind": "rs", "value": 1.5},
            **WINDOW,
        },
    ).json()
    assert body["params"]["slippage"] == {"kind": "rs", "value": 1.5}
    assert "Rs 1.50 per unit slippage on every order" in body["params"]["slippage_text"]


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


def test_changing_slippage_does_not_create_a_new_strategy_version(authed) -> None:
    """Slippage is a run setting, not part of the spec -- re-running the same
    strategy with a different slippage and saving again must land on the same
    strategy id and version, never a new one, and must not touch its state.
    """
    zero_body = authed.post(
        "/api/backtest",
        json={"spec": _spec_json(RSI), "capital": 250000, "cost_model": "Index futures", **WINDOW},
    ).json()
    first = authed.post("/api/strategies", json={"result_id": zero_body["result_id"]}).json()
    before = authed.get("/api/strategies").json()["strategies"]
    assert len(before) == 1 and before[0]["version"] == 1

    other_body = authed.post(
        "/api/backtest",
        json={
            "spec": _spec_json(RSI),
            "capital": 250000,
            "cost_model": "Index futures",
            "slippage": {"kind": "pct", "value": 0.3},
            **WINDOW,
        },
    ).json()
    second = authed.post("/api/strategies", json={"result_id": other_body["result_id"]}).json()
    assert second["strategy_id"] == first["strategy_id"]
    after = authed.get("/api/strategies").json()["strategies"]
    assert len(after) == 1 and after[0]["version"] == 1


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


# ------------------------------------------------------------ one-off orders

ORDER = "If Nifty spot goes below 22000, buy 1 lot Nifty Futures"


@pytest.fixture
def no_prices(monkeypatch):
    def unavailable(name):
        raise api.orders_view.runner.kite.KiteNotConnected("no token in tests")

    monkeypatch.setattr(api.orders_view, "kite_daily", unavailable)


def _read(authed, text=ORDER, answers=None) -> dict:
    r = authed.post("/api/orders/read", json={"text": text, "answers": answers or {}})
    assert r.status_code == 200, r.text
    return r.json()


def test_order_read_back_is_the_engines_text(authed, no_prices) -> None:
    from nlt.orders.parse import parse_order
    from nlt.orders.readback import describe as describe_order

    body = _read(authed)
    parsed_order = parse_order(ORDER).order
    assert parsed_order is not None
    assert body["readback"] == describe_order(parsed_order)
    assert body["offer_stop"] is True
    assert any("couldn't check today's price" in n for n in body["notes"])


def test_order_read_with_a_price_says_how_far_away_it_is(authed, monkeypatch) -> None:
    frame = pd.DataFrame(
        {"open": [21900.0], "high": [21950.0], "low": [21800.0], "close": [21850.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-10-08", tz="Asia/Kolkata")]),
    )
    monkeypatch.setattr(api.orders_view, "kite_daily", lambda name: frame)
    body = _read(authed)
    assert any("already below 22,000" in w for w in body["warnings"])


def test_saving_needs_the_order_that_was_read_back(authed, no_prices) -> None:
    body = _read(authed)
    tampered = {**body["order"], "action": {**body["order"]["action"], "lots": 9}}
    r = authed.post("/api/orders/create", json={"text": ORDER, "order": tampered})
    assert r.status_code == 400
    assert "no longer matches" in r.json()["error"]
    r = authed.post("/api/orders/create", json={"text": ORDER, "order": body["order"]})
    assert r.status_code == 200
    page = authed.get("/api/orders").json()
    (saved,) = page["orders"]
    assert saved["status"] == "waiting" and saved["can_cancel"] is True
    assert saved["readback"] == body["readback"]


def test_answers_are_part_of_what_is_saved(authed, no_prices) -> None:
    body = _read(authed, answers={"trade.stop_loss": "1"})
    assert body["order"]["action"]["stop_pct"] == 1.0
    r = authed.post("/api/orders/create", json={"text": ORDER, "order": body["order"]})
    assert r.status_code == 400  # without the answer, the sentence means no stop loss
    r = authed.post(
        "/api/orders/create",
        json={"text": ORDER, "answers": {"trade.stop_loss": "1"}, "order": body["order"]},
    )
    assert r.status_code == 200


def test_cancel_once_only(authed, no_prices) -> None:
    body = _read(authed)
    oid = authed.post("/api/orders/create", json={"text": ORDER, "order": body["order"]}).json()[
        "order_id"
    ]
    assert authed.post(f"/api/orders/{oid}/cancel", json={}).status_code == 200
    assert authed.post(f"/api/orders/{oid}/cancel", json={}).status_code == 400
    assert authed.get("/api/orders").json()["orders"][0]["status"] == "cancelled"


def test_an_exit_order_needs_open_positions_chosen(authed, no_prices) -> None:
    text = "If my loss exceeds 50000, exit positions"
    body = _read(authed, text)
    assert body["needs_positions"] is True and body["positions"] == []
    for positions in ([], [7], None):
        r = authed.post(
            "/api/orders/create",
            json={"text": text, "order": body["order"], "positions": positions},
        )
        assert r.status_code == 400
    assert authed.post("/api/orders/positions/7/close", json={}).status_code == 400


def test_positions_flow_from_fill_to_exit_order_to_close_now(authed, no_prices) -> None:
    from nlt.orders import runner
    from nlt.orders.engine import Contract

    store = authed.app.state.store
    body = _read(authed)
    # Non-zero slippage, deliberately: this test exists to check a position's
    # fill price moves away from the raw candle price, which the new zero
    # default would make untrue.
    authed.post(
        "/api/orders/create",
        json={
            "text": ORDER,
            "order": body["order"],
            "slippage": {"kind": "pct", "value": 0.02},
        },
    )
    created = cast(
        pd.Timestamp,
        pd.Timestamp(store.one_off_orders()[0]["created_at"]).tz_convert("Asia/Kolkata"),
    )
    start = created.ceil("min")
    idx = pd.date_range(start, periods=30, freq="1min")
    index = pd.DataFrame({"open": 21900.0, "high": 21900.0, "low": 21900.0, "close": 21900.0}, idx)
    fut = pd.DataFrame({"open": 22000.0, "high": 22000.0, "low": 22000.0, "close": 22000.0}, idx)
    oct_ = Contract(
        "NIFTY", "NIFTY26OCTFUT", cast(pd.Timestamp, start + pd.Timedelta(days=20)).date(), 1
    )
    feed = runner.Feed(
        index_minutes=lambda s, a, b: index,
        contract_minutes=lambda c, a, b: fut,
        daily=lambda n, a, b: index,
        contracts=lambda s: [oct_],
    )
    runner.step(
        store,
        feed,
        now=cast(pd.Timestamp, start + pd.Timedelta(minutes=10)),
        charges=lambda s: lambda *a: 0.0,
    )

    page = authed.get("/api/orders").json()
    (pos,) = page["positions"]
    assert pos["what"] == "Bought 1 lot of NIFTY26OCTFUT"
    assert pos["pnl_text"].startswith("-")  # slippage: bought a little above 22,000
    assert page["orders"][0]["status"] == "open"

    text = "If my loss exceeds 50000, exit positions"
    reading = _read(authed, text)
    assert [p["id"] for p in reading["positions"]] == [pos["id"]]
    r = authed.post(
        "/api/orders/create",
        json={"text": text, "order": reading["order"], "positions": [pos["id"]]},
    )
    assert r.status_code == 200
    saved = authed.get("/api/orders").json()["orders"][0]
    assert any(pos["name"] in line for line in saved["readback"])

    assert authed.post(f"/api/orders/positions/{pos['id']}/close", json={}).status_code == 200
    again = authed.post(f"/api/orders/positions/{pos['id']}/close", json={})
    assert again.status_code == 400 and "already under way" in again.json()["error"]


# ------------------------------------------------------------- the test period


def test_a_test_period_in_the_sentence_comes_back_as_dates(authed) -> None:
    body = authed.post("/api/translate", json={"text": RSI + ", backtest the last 2 years"}).json()
    assert body["understood"] is not None
    period = body["period"]
    assert period["words"] == "backtest the last 2 years"
    start = cast(pd.Timestamp, pd.Timestamp(period["start"]))
    end = cast(pd.Timestamp, pd.Timestamp(period["end"]))
    assert end.date() == pd.Timestamp.today().date()
    assert end - pd.DateOffset(years=2) == start


def test_no_test_period_leaves_the_dates_alone(authed) -> None:
    assert authed.post("/api/translate", json={"text": RSI}).json()["period"] is None


def test_period_dates_never_start_before_trusted_history() -> None:
    import datetime as dt

    from nlt.data.source import DATA_FLOOR
    from nlt.translate.rules import BacktestPeriod

    today = dt.date(2026, 10, 12)
    out = logic.period_dates(
        BacktestPeriod(words="the last 10 years", last=10, unit="years"), today
    )
    assert out is not None
    assert out["start"] == DATA_FLOOR.isoformat()
    assert out["note"] and "starts on" in out["note"]
    year = logic.period_dates(BacktestPeriod(words="2024", year=2024), today)
    assert year is not None
    assert (year["start"], year["end"], year["note"]) == ("2024-01-01", "2024-12-31", None)
    this_year = logic.period_dates(BacktestPeriod(words="2026", year=2026), today)
    assert this_year is not None
    assert this_year["end"] == "2026-10-12"
    assert logic.period_dates(BacktestPeriod(words="2027", year=2027), today) is None
    months = logic.period_dates(BacktestPeriod(words="6 months", last=6, unit="months"), today)
    assert months is not None
    assert months["start"] == "2026-04-12"


# ------------------------------------------------------------ readback labels


def test_translate_labels_every_readback_line(authed) -> None:
    body = authed.post("/api/translate", json={"text": RSI}).json()["understood"]
    body_lines = [r for r in body["readback"].splitlines() if r.startswith("  ")]
    assert len(body["readback_labels"]) == len(body_lines)
    assert set(body["readback_labels"]) <= {"said", "worked_out", "default"}


def test_a_saved_strategy_has_no_guessed_labels(authed) -> None:
    from nlt.translate.readback import describe_lines

    spec = logic.translate_text(RSI).spec
    assert spec is not None
    assert api._spec_payload(spec)["readback_labels"] is None
    assert [label for *_x, label in describe_lines(spec, None)] == [None] * len(
        describe_lines(spec, None)
    )


def test_questions_carry_their_buttons_and_nothing_is_picked(authed) -> None:
    body = authed.post("/api/translate", json={"text": "buy nifty when rsi cracks 30"}).json()
    stop = next(q for q in body["questions"] if q["field"] == "exit.stop_pct")
    assert [c["label"] for c in stop["choices"]] == ["0.5%", "1%", "2%"]
    assert stop["answerable"] is True
    assert body["answers"] == {}


# ------------------------------------------------------- before and after charges


def _pinned_run(spec, cost_model="Index futures"):
    return logic.backtest_spec(
        spec,
        capital=2_500_000,
        cost_model_label=cost_model,
        start=cast(pd.Timestamp, pd.Timestamp(WINDOW["start"])).date(),
        end=cast(pd.Timestamp, pd.Timestamp(WINDOW["end"])).date(),
    )


def test_before_charges_is_a_second_run_with_no_charges() -> None:
    from nlt.spec.models import Sizing

    # A fixed size, so both runs buy exactly the same quantities and the gap
    # between them is the charges and nothing else.
    spec = logic.translate_text(RSI).spec
    assert spec is not None
    spec = spec.model_copy(update={"sizing": Sizing(mode="fixed_lots", lots=10)})
    result = _pinned_run(spec)
    after, before = result.backtest, result.before_charges
    assert [t.quantity for t in before.trades] == [t.quantity for t in after.trades]
    assert sum(t.charges for t in before.trades) == 0
    charges = after.metrics["total_charges"]
    assert charges > 0
    gap = before.equity.iloc[-1] - after.equity.iloc[-1]
    assert gap == pytest.approx(charges)


def test_before_charges_never_looks_worse_than_after() -> None:
    result = _pinned_run(logic.translate_text(SMA).spec)
    assert (
        result.before_charges.metrics["total_return_pct"]
        >= result.backtest.metrics["total_return_pct"]
    )


def test_the_report_shows_before_and_after_charges(authed) -> None:
    body = authed.post(
        "/api/backtest",
        json={"spec": _spec_json(RSI), "capital": 2500000, "cost_model": "Index futures", **WINDOW},
    ).json()
    rows = {r["label"]: r for r in body["before_after_charges"]}
    assert set(rows) == {
        "Total return",
        "Biggest drop from a peak",
        "Trades that made money",
        "Number of trades",
        "Charges paid",
    }
    assert rows["Charges paid"]["before"] == "Rs 0"
    assert rows["Charges paid"]["after"] != "Rs 0"
    assert rows["Total return"]["after"] == body["metrics"][0]["strategy"]
    direct = _pinned_run(StrategySpec.model_validate(_spec_json(RSI)))
    assert rows["Total return"]["before"] == logic.format_signed_pct(
        direct.before_charges.metrics["total_return_pct"]
    )
    # Compare the raw figures, not their rounded-to-0.1% display strings: at
    # zero slippage (the new default) this strategy's before/after numbers
    # happen to round to the same "+0.6%" even though they really differ --
    # exactly the trap of pinning a formatted string instead of the number
    # behind it.
    assert direct.before_charges.metrics["total_return_pct"] != pytest.approx(
        direct.backtest.metrics["total_return_pct"]
    )


def test_no_charges_means_nothing_to_compare(authed) -> None:
    body = authed.post(
        "/api/backtest",
        json={
            "spec": _spec_json(RSI),
            "capital": 2500000,
            "cost_model": "No costs (for comparison only)",
            **WINDOW,
        },
    ).json()
    assert body["before_after_charges"] is None


def test_a_basket_is_also_run_again_with_no_charges() -> None:
    spec = logic.translate_text(
        "Buy NIFTY 100 stocks when ema8 crosses above ema21, target 3%, stop loss 1.5%"
    ).spec
    assert spec is not None
    result = logic.backtest_spec(
        spec,
        capital=2_500_000,
        start=cast(pd.Timestamp, pd.Timestamp("2024-01-01")).date(),
        end=cast(pd.Timestamp, pd.Timestamp("2024-06-30")).date(),
    )
    assert result.error is None, result.error
    assert result.backtest is not None
    assert result.before_charges is not None
    assert result.backtest.metrics["total_charges"] > 0
    assert sum(t.charges for t in result.before_charges.trades) == 0


def test_the_report_carries_how_it_was_tested_and_says_nothing_twice(authed) -> None:
    # A basket: its engine and its loader both warn about survivors.
    basket = _spec_json("buy nifty 50 stocks when rsi cracks 30, target 3%, stop loss 1.5%")
    body = authed.post(
        "/api/backtest",
        json={"spec": basket, "capital": 2500000, "cost_model": "Index futures", **WINDOW},
    ).json()
    assert any("only sees survivors" in w for w in body["weak_spots"])
    assert body["assumptions"] and body["weak_spots"]
    assert not set(body["notes"]) & set(body["weak_spots"])


# ------------------------------------------------------------------ CSV


def test_trades_download_as_csv_that_adds_up(authed) -> None:
    import csv
    import io

    body = authed.post(
        "/api/backtest",
        json={"spec": _spec_json(RSI), "capital": 2500000, "cost_model": "Index futures", **WINDOW},
    ).json()
    r = authed.get(f"/api/results/{body['result_id']}/trades.csv")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert len(rows) == len(body["trades"])
    assert rows[0]["symbol"] == "NIFTY"
    net = sum(float(row["profit_after_charges"]) for row in rows)
    assert net == pytest.approx(sum(t["pnl"] for t in body["trades"]), abs=0.05)
    for row in rows:
        gross, charges = float(row["profit_before_charges"]), float(row["charges"])
        assert float(row["profit_after_charges"]) == pytest.approx(gross - charges, abs=0.02)


def test_trades_csv_needs_a_login(client) -> None:
    assert client.get("/api/results/anything/trades.csv").status_code == 401


def test_an_unknown_result_has_no_csv(authed) -> None:
    assert authed.get("/api/results/nope/trades.csv").status_code == 400


# ------------------------------------------------- change the numbers before testing


def _understood(authed, text):
    return authed.post("/api/translate", json={"text": text}).json()["understood"]


def _adjust(authed, understood, edits, answers=None):
    return authed.post(
        "/api/adjust",
        json={
            "spec": understood["spec"],
            "sources": understood["sources"],
            "answers": answers or {},
            "edits": edits,
        },
    )


def _labels(body) -> dict[str, str]:
    lines = [r.strip() for r in body["readback"].splitlines() if r.startswith("  ")]
    return dict(zip(lines, body["readback_labels"], strict=True))


def test_moving_the_stop_redraws_the_readback_from_the_result(authed) -> None:
    u = _understood(authed, RSI)
    body = _adjust(authed, u, {"stop_pct": "1.5"}).json()
    spec = StrategySpec.model_validate(body["spec"])
    assert spec.exit.stop_pct == 1.5
    assert body["readback"] == describe(spec)
    assert _labels(body)["Stop loss at -1.5%"] == "said"  # typed in the box, not the sentence


def test_a_stop_loss_can_be_moved_but_never_removed(authed) -> None:
    r = _adjust(authed, _understood(authed, RSI), {"stop_pct": ""})
    assert r.status_code == 400
    assert "always needs one" in r.json()["error"]


@pytest.mark.parametrize("bad", ["0", "-1", "abc", "60"])
def test_a_box_that_cannot_run_says_why(authed, bad) -> None:
    r = _adjust(authed, _understood(authed, RSI), {"stop_pct": bad})
    assert r.status_code == 400


def test_the_target_can_be_removed_or_added(authed) -> None:
    removed = _adjust(authed, _understood(authed, RSI), {"target_pct": ""}).json()
    assert StrategySpec.model_validate(removed["spec"]).exit.target_pct is None
    assert "Take profit" not in removed["readback"]
    no_target = _understood(authed, "buy nifty when rsi cracks 30, stop loss 1%")
    added = _adjust(authed, no_target, {"target_pct": "3"}).json()
    assert "Take profit at +3%" in added["readback"]


def test_changing_the_candles_shows_what_that_brings(authed) -> None:
    body = _adjust(authed, _understood(authed, RSI), {"timeframe": "15m"}).json()
    assert "15-minute candles" in body["readback"]
    assert "Square off at 15:15 if still open" in body["readback"]
    assert _labels(body)["15-minute candles"] == "said"


def test_vwap_cannot_be_moved_to_daily_candles(authed) -> None:
    u = _understood(authed, "buy tcs when close > vwap, stop 1%, target 2%, use 15-minute candles")
    r = _adjust(authed, u, {"timeframe": "1d"})
    assert r.status_code == 400
    assert "VWAP" in r.json()["error"]


def test_a_basket_stays_on_daily_candles(authed) -> None:
    u = _understood(authed, "buy nifty 50 stocks when rsi cracks 30, target 3%, stop loss 1.5%")
    assert _adjust(authed, u, {"timeframe": "15m"}).status_code == 400


def test_the_form_edits_the_strategy_it_does_not_rewrite_it(authed) -> None:
    """A trailing-stop strategy has no fixed stop box; sending one adds nothing."""
    u = _understood(authed, "buy tcs when close is above 200 ema, target 5%, trailing stop 2%")
    body = _adjust(authed, u, {"stop_pct": "1"}).json()
    spec = StrategySpec.model_validate(body["spec"])
    assert spec.exit.stop_pct is None
    assert spec.exit.trailing_stop_pct == 2.0


def test_edits_always_start_from_the_strategy_as_first_read(authed) -> None:
    u = _understood(authed, RSI)
    _adjust(authed, u, {"stop_pct": "3"})
    body = _adjust(authed, u, {"target_pct": "4"}).json()
    spec = StrategySpec.model_validate(body["spec"])
    assert (spec.exit.stop_pct, spec.exit.target_pct) == (1.0, 4.0)


def test_made_up_labels_are_refused(authed) -> None:
    u = _understood(authed, RSI)
    r = authed.post(
        "/api/adjust",
        json={"spec": u["spec"], "sources": {"exit.stop": "certain"}, "edits": {"stop_pct": "2"}},
    )
    assert r.status_code == 400


def test_the_results_line_names_the_charges_really_applied(authed) -> None:
    stock = _spec_json("buy reliance when rsi cracks 30, target 5%, stop loss 2%")
    body = authed.post(
        "/api/backtest",
        json={"spec": stock, "capital": 2500000, "cost_model": "Index futures", **WINDOW},
    ).json()
    assert body["params"]["cost_model"] == "Index futures"
    assert body["params"]["charges_applied"] == "Equity delivery"


def test_daily_trades_download_without_a_made_up_time(authed) -> None:
    body = authed.post(
        "/api/backtest",
        json={"spec": _spec_json(RSI), "capital": 2500000, "cost_model": "Index futures", **WINDOW},
    ).json()
    text = authed.get(f"/api/results/{body['result_id']}/trades.csv").text
    assert "00:00" not in text


# ----------------------------------------------------------- pause / resume


def _seed_paper_run(store: Store, *, realised: float = 500.0, open_pnl: float = 200.0) -> int:
    """A minimal paper run with a status snapshot, as if one pass had already
    happened -- enough to drive the pause/resume routes and `/api/running`
    without needing a real Kite feed."""
    spec = logic.translate_text(RSI).spec
    assert spec is not None
    sid = store.save_strategy(spec, describe(spec))
    started_at = pd.Timestamp.now(tz="Asia/Kolkata") - pd.Timedelta(days=1)
    run_id = store.start_run(
        sid, "paper", {"capital": 100_000.0, "started_at": started_at.isoformat()}
    )
    store.set_paper_status(
        run_id,
        "ok",
        "Up to date.",
        pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        {
            "capital": 100_000.0,
            "realised_pnl": realised,
            "open_pnl_if_closed_now": open_pnl,
            "equity": 100_000.0 + realised + open_pnl,
            "closed_trades": 1,
            "open_positions": [],
            "skipped_symbols": [],
        },
    )
    return run_id


def test_pause_then_resume_round_trip(authed) -> None:
    run_id = _seed_paper_run(authed.app.state.store)
    r = authed.post(f"/api/paper/{run_id}/pause", json={})
    assert r.status_code == 200
    assert r.json()["run"]["paused"] is True

    again = authed.post(f"/api/paper/{run_id}/pause", json={})
    assert again.status_code == 400

    r2 = authed.post(f"/api/paper/{run_id}/resume", json={})
    assert r2.status_code == 200
    assert r2.json()["run"]["paused"] is False

    again2 = authed.post(f"/api/paper/{run_id}/resume", json={})
    assert again2.status_code == 400


def test_pause_and_resume_require_json(authed) -> None:
    run_id = _seed_paper_run(authed.app.state.store)
    r = authed.post(f"/api/paper/{run_id}/pause")
    assert r.status_code == 400
    r2 = authed.post(f"/api/paper/{run_id}/resume")
    assert r2.status_code == 400


def test_pause_refuses_a_run_that_is_not_running(authed) -> None:
    run_id = _seed_paper_run(authed.app.state.store)
    authed.app.state.store.finish_run(run_id, {}, status="complete")
    r = authed.post(f"/api/paper/{run_id}/pause", json={})
    assert r.status_code == 400


# --------------------------------------------------------------- /api/running


def test_running_lists_a_strategy_run_with_its_pause_flag(authed) -> None:
    run_id = _seed_paper_run(authed.app.state.store)
    body = authed.get("/api/running").json()
    item = next(i for i in body["items"] if i.get("run_id") == run_id)
    assert item["kind"] == "strategy"
    assert item["paused"] is False

    authed.post(f"/api/paper/{run_id}/pause", json={})
    body2 = authed.get("/api/running").json()
    item2 = next(i for i in body2["items"] if i.get("run_id") == run_id)
    assert item2["paused"] is True
    # Straight after pressing Pause, before any pass has run, the tag must not
    # still say the run is going.
    assert item2["health_label"] == logic.paper_health_label("paused")[1]
    assert item2["health_label"] != item["health_label"]


def test_running_money_figures_match_the_paper_run_cards_own_numbers(authed) -> None:
    store = authed.app.state.store
    run_id = _seed_paper_run(store, realised=500.0, open_pnl=200.0)
    card = api._paper_run_card(store, store.get_run(run_id))
    body = authed.get("/api/running").json()
    item = next(i for i in body["items"] if i.get("run_id") == run_id)

    # Same number as the card; the drawer adds the sign.
    assert item["open_now"] == "+" + card["snapshot"]["open"]
    assert item["open_now_value"] == 200.0
    assert item["total_value"] == 700.0
    assert item["closed_today_value"] == 0.0


def test_running_requires_a_login(client) -> None:
    r = client.get("/api/running")
    assert r.status_code == 401


def test_closed_today_is_judged_in_ist_not_utc(authed) -> None:
    """01:00 IST on a given day is still the previous calendar day in UTC. An
    exit recorded at that time must count as "today" when "now" is that same
    morning in IST -- the only calendar the rest of this product (and its
    user) uses. `now` is pinned so the test does not depend on when it runs.
    """
    store = authed.app.state.store
    run_id = _seed_paper_run(store)
    pinned_now = cast(pd.Timestamp, pd.Timestamp("2024-06-10T01:30:00", tz="Asia/Kolkata"))
    bar_time = pd.Timestamp("2024-06-10T01:00:00", tz="Asia/Kolkata")
    store.record_paper_event(
        run_id,
        "exit",
        "TEST|edge-of-day",
        observed_at=bar_time.isoformat(),
        symbol="NIFTY",
        bar_time=bar_time.isoformat(),
        detail={"net_pnl": 321.0},
    )
    body = api._running(store, now=pinned_now)
    item = next(i for i in body["items"] if i.get("run_id") == run_id)
    assert item["closed_today_value"] == pytest.approx(321.0)


def test_today_ist_judges_the_ist_calendar_day_not_utcs() -> None:
    """01:00 IST on 10 June is 19:30 UTC on 9 June -- a different calendar date.
    `_today_ist` must use the IST date, the only calendar this product's user
    (and the rest of the paper-trading feed) uses."""
    bar_time = "2024-06-10T01:00:00+05:30"
    today_ist = cast(pd.Timestamp, pd.Timestamp("2024-06-10T00:00:00", tz="Asia/Kolkata"))
    assert api._today_ist(bar_time, today_ist) is True
    assert api._today_ist(bar_time, cast(pd.Timestamp, today_ist + pd.Timedelta(days=1))) is False


def test_running_never_shows_an_unknown_price_as_rs_0(authed, monkeypatch) -> None:
    """A position with no price yet is not a flat result. It shows as unknown,
    is left out of the totals, and the totals line says so."""
    store = authed.app.state.store
    page = {
        "orders": [],
        "positions": [
            {"id": 1, "name": "1 lot NIFTY FUT", "pnl": None, "pnl_text": "", "last_at": ""},
            {
                "id": 2,
                "name": "1 lot NIFTY FUT",
                "pnl": 250.0,
                "pnl_text": "+Rs 250",
                "last_at": "",
            },
        ],
    }
    monkeypatch.setattr(api.orders_view, "page", lambda _store: page)
    body = api._running(store)
    unpriced = next(i for i in body["items"] if i.get("position_id") == 1)
    assert unpriced["open_now"] == "price not known yet"
    assert "Rs 0" not in unpriced["open_now"] + unpriced["total"]
    assert body["totals"]["open_now"] == "+Rs 250"
    assert "1 open position without a price yet is left out" in body["totals"]["label"]


def test_running_closed_today_includes_one_off_orders_closed_today(authed, monkeypatch) -> None:
    store = authed.app.state.store
    now = cast(pd.Timestamp, pd.Timestamp("2024-06-10T14:00:00", tz="Asia/Kolkata"))
    exits = [
        {
            "kind": "exited",
            "bar_time": "2024-06-10T10:00:00+05:30",
            "detail_json": '{"net_pnl": 400.0}',
        },
        {
            "kind": "exited",
            "bar_time": "2024-06-09T10:00:00+05:30",
            "detail_json": '{"net_pnl": 999.0}',
        },
    ]
    monkeypatch.setattr(api.orders_view, "page", lambda _store: {"orders": [], "positions": []})
    monkeypatch.setattr(store, "one_off_events", lambda: exits)
    body = api._running(store, now=now)
    assert body["totals"]["closed_today"] == "+Rs 400"
    assert "one-off orders that closed today" in body["totals"]["label"]


def test_the_paper_page_card_says_paused_straight_after_pause(authed) -> None:
    store = authed.app.state.store
    run_id = _seed_paper_run(store)
    authed.post(f"/api/paper/{run_id}/pause", json={})
    card = api._paper_run_card(store, store.get_run(run_id))
    assert card["paused"] is True
    assert logic.paper_health_label("paused")[1] in str(card)
