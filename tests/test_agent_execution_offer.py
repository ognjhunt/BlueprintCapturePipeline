"""The execution offer a website capture publishes for self-serve robot-team runs.

Pins the contract the WebApp's self-serve preparation and this repo's
``agent_run_executor`` both rely on: the offer names the capture root inside the
partition, exactly one scenario, and an episode count equal to the episode
specs the executor will read. Anything less publishes nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.agent_execution_offer import (
    OFFER_SCHEMA_VERSION,
    build_agent_execution_offer,
    publish_agent_execution_offer,
)


def _capture(tmp_path: Path, *, episodes: list[dict] | None, website: bool = True, count: int | None = None) -> Path:
    root = tmp_path / "partition" / "scenes" / "site-req1" / "captures" / "walkthrough-req1"
    root.mkdir(parents=True)
    (root / "capture_descriptor.json").write_text(json.dumps({
        "scene_id": "site-req1",
        "capture_id": "walkthrough-req1",
        "site_submission_id": "req1",
        "metadata": {"capture_entry_source": "browser_self_capture" if website else "iphone"},
    }))
    if episodes is not None:
        specs = root / "pipeline" / "simulation_automation" / "episode_specs.json"
        specs.parent.mkdir(parents=True)
        specs.write_text(json.dumps({
            "episode_count": len(episodes) if count is None else count,
            "episodes": episodes,
        }))
    return root


def _episodes(n: int, scenario: str = "capture_observed") -> list[dict]:
    return [{"episode_id": f"ep-{i}", "scenario_id": scenario} for i in range(n)]


def test_offer_names_the_partition_root_one_scenario_and_the_spec_count(tmp_path: Path) -> None:
    root = _capture(tmp_path, episodes=_episodes(50))
    offer, reason = build_agent_execution_offer(root)
    assert reason is None
    specs_bytes = (root / "pipeline" / "simulation_automation" / "episode_specs.json").read_bytes()
    assert offer == {
        "schema_version": OFFER_SCHEMA_VERSION,
        "scene_id": "site-req1",
        "capture_id": "walkthrough-req1",
        "capture_root": str(root.resolve()),
        "scenario_id": "capture_observed",
        "episode_count": 50,
        "episode_specs_sha256": "sha256:" + hashlib.sha256(specs_bytes).hexdigest(),
    }


@pytest.mark.parametrize(
    ("episodes", "count", "reason"),
    [
        (None, None, "episode_specs_missing"),
        ([], None, "episode_specs_count_invalid"),
        (_episodes(3), 5, "episode_specs_count_invalid"),
        (_episodes(2) + _episodes(2, "reset_drift"), None, "episode_specs_scenario_not_single"),
    ],
)
def test_no_offer_when_the_executor_would_refuse(tmp_path: Path, episodes, count, reason) -> None:
    root = _capture(tmp_path, episodes=episodes, count=count)
    assert build_agent_execution_offer(root) == (None, reason)


def test_publishes_over_the_signed_transport_for_a_website_capture(tmp_path: Path) -> None:
    root = _capture(tmp_path, episodes=_episodes(50))
    calls: list[dict] = []

    def transport(**kwargs):
        calls.append(kwargs)
        return {"ok": True}

    result = publish_agent_execution_offer(root, transport=transport)
    assert result["status"] == "published"
    assert calls == [{
        "capture_id": "walkthrough-req1",
        "operation": "agent-execution-offer",
        "payload": {"request_id": "req1", "scene_id": "site-req1", "offer": result["offer"]},
    }]
    recorded = json.loads((root / "pipeline" / "agent_execution_offer_publication.json").read_text())
    assert recorded["status"] == "published"


def test_skips_non_website_captures_and_never_raises_on_transport_failure(tmp_path: Path) -> None:
    app_capture = _capture(tmp_path / "a", episodes=_episodes(50), website=False)
    assert publish_agent_execution_offer(app_capture, transport=lambda **_: pytest.fail("must not publish"))["status"] == "skipped"

    website = _capture(tmp_path / "b", episodes=_episodes(50))

    def failing(**_kwargs):
        raise ValueError("website_control_agent-execution-offer_http_503:unavailable")

    result = publish_agent_execution_offer(website, transport=failing)
    assert result["status"] == "failed"
    assert "http_503" in result["reason"]


def test_the_signed_transport_accepts_the_offer_operation() -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline" / "website_task_context.py").read_text()
    assert '"agent-execution-offer"' in source
