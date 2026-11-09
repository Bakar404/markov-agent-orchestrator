"""The bridge's weight-space-programming half, against the real API.

``arena_bridge.wsp`` needs only httpx and the backend's stdlib-only editor module, so it loads
here without Agent Framework; the Copilot call itself is replaced by a stub editor.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.orchestration.wsp import Patch
from app.orchestration.wsp.calibrate import CALIBRATION_SEED_BAND
from app.orchestration.wsp.editor import example_path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bridge"))
from arena_bridge import wsp as bridge_wsp

CUSTOM_GOAL = "Lean on the verifier before terminating, and keep coalitions small."


@pytest.fixture(autouse=True)
def no_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("WSP_CHECKPOINT", str(tmp_path / "absent.json"))


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    import app.db as db_module

    db_path = tmp_path_factory.mktemp("bridge") / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}", connect_args={"check_same_thread": False}
    )
    db_module.engine = engine
    db_module.SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )
    db_module.Base.metadata.create_all(engine)
    with TestClient(create_app()) as test_client:
        yield test_client


class _Arena:
    """The two Arena calls program_patch makes, routed into the TestClient."""

    def __init__(self, client: TestClient) -> None:
        self.client = client
        self.brief_params: dict = {}

    def wsp_brief(self, **params):
        self.brief_params = params
        response = self.client.get("/api/wsp/brief", params=params)
        assert response.status_code == 200, response.text
        return response.json()


class _StubEditor:
    def __init__(self, model: str) -> None:
        self.model = model

    async def apropose(self, rendered: str, goal: str) -> Patch:
        recorded = json.loads(example_path("narrow_coalitions").read_text(encoding="utf-8"))
        return Patch.from_dict({**recorded, "goal": goal})


def test_the_bridge_band_mirrors_the_backend():
    assert bridge_wsp.CALIBRATION_SEED_BAND == CALIBRATION_SEED_BAND


@pytest.mark.parametrize(
    "seeds", [[101, CALIBRATION_SEED_BAND.start], [True], [-1], [2**31], [1.5]]
)
def test_bad_seeds_are_refused_before_any_editor_call(seeds):
    with pytest.raises(ValueError):
        bridge_wsp.check_evaluation_seeds(seeds)


def test_custom_goal_patch_reaches_the_server_with_goal_and_provenance(client, monkeypatch):
    monkeypatch.setattr(
        bridge_wsp, "_editor_module", lambda: SimpleNamespace(CopilotEditor=_StubEditor)
    )
    arena = _Arena(client)
    wsp = asyncio.run(
        bridge_wsp.program_patch(
            arena, model="stub", goal=CUSTOM_GOAL, max_steps=8, belief_dim=4,
            budget=250_000.0, latency_budget_ms=1_800_000.0,
        )
    )
    # The brief was calibrated under the arm's own budget and latency budget.
    assert arena.brief_params["budget_usd"] == 250_000.0
    assert arena.brief_params["latency_budget_ms"] == 1_800_000.0

    run = {"task": "pick a cache", "strategy": "wsp_markov_game", "mode": "live",
           "belief_dim": 4, "max_steps": 8, "budget_usd": 250_000.0,
           "latency_budget_ms": 1_800_000.0, "cost_unit": "tokens", "experiment": "bridge"}
    fields = bridge_wsp.programmed_run_fields(wsp)
    assert set(fields) == {"wsp_patch", "wsp_goal", "wsp_provenance"}

    # What the driver used to send: no wsp_goal. The arena replays the patch against the
    # strategy's default goal and refuses it, on every seed.
    no_goal = {k: v for k, v in fields.items() if k != "wsp_goal"}
    patch_only = client.post("/api/runs", json={**run, "seed": 101, **no_goal})
    assert patch_only.status_code == 400
    assert "goal" in patch_only.json()["detail"]

    for seed in (101, 102):
        created = client.post("/api/runs", json={**run, "seed": seed, **fields})
        assert created.status_code == 201, created.text
        artifact = client.get(f"/api/runs/{created.json()['id']}/wsp").json()
        assert artifact["goal"] == CUSTOM_GOAL
        assert artifact["base"]["patch_provenance"] == fields["wsp_provenance"]
        assert artifact["calibration"]["overrides"]["budget_usd"] == 250_000.0


def test_no_programming_adds_no_fields():
    assert bridge_wsp.programmed_run_fields(None) == {}
