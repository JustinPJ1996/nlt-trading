"""Odd input gets a plain refusal, never a crash.

Each of these was found by the type checker (ty) when it was adopted: a value
the code assumed was always there could, on some real input, be missing or
not a number at all.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from starlette.testclient import TestClient

from app import api, logic
from nlt.options.data import traded_mask
from nlt.store.db import Store

LOGIN_ID = "tester"
PASSWORD = "correct horse"
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def authed(monkeypatch, tmp_path) -> TestClient:
    monkeypatch.setenv(api.auth.LOGIN_ID_ENV_VAR, LOGIN_ID)
    monkeypatch.setenv(api.auth.ENV_VAR, PASSWORD)
    app = api.create_app(Store(tmp_path / "web.sqlite"), secret="test-secret")
    client = TestClient(app, base_url="https://testserver")
    r = client.post("/api/login", json={"login_id": LOGIN_ID, "password": PASSWORD})
    assert r.status_code == 200
    return client


def _post_raw(client: TestClient, path: str, text: str):
    return client.post(path, content=text, headers={"content-type": "application/json"})


@pytest.mark.parametrize("constant", ["Infinity", "-Infinity", "NaN"])
def test_a_number_that_is_not_a_number_is_refused_not_a_server_error(authed, constant) -> None:
    # Python's JSON reader accepts these; real JSON does not. Infinity as a
    # strategy id used to be a 500 "Something went wrong".
    cost = next(iter(logic.COST_MODEL_LABELS))
    r = _post_raw(
        authed, "/api/paper/start", f'{{"strategy_id": {constant}, "cost_model": "{cost}"}}'
    )
    assert r.status_code == 400
    assert "not valid JSON" in r.text


def test_an_ordinary_request_still_reads_after_that_change(authed) -> None:
    # The same endpoint with honest JSON gets past the reader to its own
    # check -- so the refusal above is the constant, not the reader breaking.
    cost = next(iter(logic.COST_MODEL_LABELS))
    r = _post_raw(authed, "/api/paper/start", f'{{"cost_model": "{cost}"}}')
    assert r.status_code == 400
    assert "Pick a strategy" in r.text


def test_traded_mask_without_a_bad_ohlc_column_is_volume_alone() -> None:
    df = pd.DataFrame({"volume": [0, 5, 10]})
    with warnings.catch_warnings():
        # `~False` (what this used to compute) warns today and is an error in
        # a later Python; the mask must not depend on it.
        warnings.simplefilter("error")
        mask = traded_mask(df)
    assert mask.tolist() == [False, True, True]
    assert mask.dtype == bool


def test_check_vix_with_no_option_prices_says_so(monkeypatch, capsys) -> None:
    build_iv = _script("build_iv")
    monkeypatch.setattr(build_iv.option_data, "coverage", lambda *_a: None)
    build_iv.check_vix()
    assert "no option prices on disk" in capsys.readouterr().out


def test_span_check_with_no_legs_says_so() -> None:
    calibrate = _script("calibrate_span_zerodha")
    with pytest.raises(SystemExit, match="no legs"):
        calibrate.zerodha(MagicMock(), "NFO", [])
