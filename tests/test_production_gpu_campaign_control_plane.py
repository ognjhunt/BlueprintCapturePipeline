import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from blueprint_pipeline.production_gpu_campaign_control_plane import (
    CampaignControlPlaneError,
    ProductionGpuCampaignControlPlane,
    REQUIRED_ARTIFACTS,
    build_campaign_spec,
    create_production_gpu_campaign_app,
)


def _spec(**overrides):
    return build_campaign_spec(
        campaign_id="campaign-1",
        source_sha="a" * 40,
        release_candidate_fingerprint="sha256:" + "b" * 64,
        worker_image_ref="registry/worker@sha256:" + "c" * 64,
        scenario_id="g1-kitchen-microwave",
        task_id="open-microwave",
        policy_revision="groot@revision",
        model_asset_revisions={"kitchen": "asset@revision"},
        **overrides,
    )


def _control(tmp_path):
    return ProductionGpuCampaignControlPlane(tmp_path / "control.sqlite", tmp_path / "artifacts")


def _complete_artifacts(control, attempt):
    for artifact in REQUIRED_ARTIFACTS:
        payload = (artifact + "-evidence").encode()
        digest = hashlib.sha256(payload).hexdigest()
        control.begin_artifact(
            "campaign-1",
            attempt,
            artifact,
            relative_path=f"{artifact}.bin",
            total_size=len(payload),
            expected_sha256=digest,
        )
        control.append_artifact_chunk(
            "campaign-1", attempt, artifact, offset=0, data=payload, chunk_sha256=digest
        )


def _pass_attempt(control, attempt):
    control.transition_attempt("campaign-1", attempt, "running")
    control.transition_attempt("campaign-1", attempt, "collecting")
    _complete_artifacts(control, attempt)
    control.transition_attempt(
        "campaign-1", attempt, "validating", simulator_steps=10, policy_actions=10
    )
    return control.transition_attempt(
        "campaign-1",
        attempt,
        "passed",
        terminal_reason="declared_completion",
        semantic_task_success=True,
    )


def test_campaign_spec_is_dynamic_640x480_and_exact_release():
    spec = _spec()
    assert spec["status"] == "valid"
    assert spec["runtime"]["fixed_frame_count"] is None
    assert spec["runtime"]["dynamic_episode_termination"] is True
    assert (spec["runtime"]["review_width"], spec["runtime"]["review_height"]) == (640, 480)
    assert [row["seed"] for row in spec["attempts"]] == [1000, 1001, 1002, 1003]


def test_smoke_failure_prevents_every_full_episode(tmp_path):
    control = _control(tmp_path)
    control.create_campaign(_spec())
    control.transition_attempt("campaign-1", "smoke", "running")
    snapshot = control.transition_attempt(
        "campaign-1", "smoke", "failed", terminal_reason="policy_unavailable"
    )
    assert snapshot["state"] == "smoke_blocked"
    with pytest.raises(CampaignControlPlaneError, match="full_episode_blocked"):
        control.transition_attempt("campaign-1", "episode-1", "running")


def test_resumable_artifact_rejects_wrong_offset_and_finishes_by_hash(tmp_path):
    control = _control(tmp_path)
    control.create_campaign(_spec())
    payload = b"abcdefgh"
    control.begin_artifact(
        "campaign-1",
        "smoke",
        "worker_log",
        relative_path="logs/worker.log",
        total_size=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    first = payload[:4]
    result = control.append_artifact_chunk(
        "campaign-1",
        "smoke",
        "worker_log",
        offset=0,
        data=first,
        chunk_sha256=hashlib.sha256(first).hexdigest(),
    )
    assert result["state"] == "receiving"
    with pytest.raises(CampaignControlPlaneError, match="offset_conflict"):
        control.append_artifact_chunk(
            "campaign-1",
            "smoke",
            "worker_log",
            offset=0,
            data=b"x",
            chunk_sha256=hashlib.sha256(b"x").hexdigest(),
        )
    second = payload[4:]
    result = control.append_artifact_chunk(
        "campaign-1",
        "smoke",
        "worker_log",
        offset=4,
        data=second,
        chunk_sha256=hashlib.sha256(second).hexdigest(),
    )
    assert result["state"] == "complete"
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()


def test_complete_campaign_has_four_terminal_attempts_and_honest_status(tmp_path):
    control = _control(tmp_path)
    control.create_campaign(_spec())
    _pass_attempt(control, "smoke")
    for attempt in ("episode-1", "episode-2", "episode-3"):
        snapshot = _pass_attempt(control, attempt)
    assert snapshot["state"] == "completed"
    assert snapshot["terminal"] is True
    status = control.customer_status("campaign-1")
    assert status["status"] == "completed"
    assert status["attempts_completed"] == 4
    assert status["semantic_task_success"] is True
    assert status["estimated_completion_time"] is None


def test_customer_path_has_no_provider_dependency():
    import blueprint_pipeline.production_gpu_campaign_control_plane as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "gpu_render_providers" not in source
    assert "runpod_provider_adapter" not in source
    assert "boto3" not in source


def test_private_api_exposes_honest_status_and_requires_auth(tmp_path):
    app = create_production_gpu_campaign_app(
        database_path=tmp_path / "control.sqlite",
        artifact_root=tmp_path / "artifacts",
        auth_token="t" * 32,
    )
    client = TestClient(app)
    assert client.post("/v1/campaigns", json=_spec()).status_code == 401
    headers = {"Authorization": "Bearer " + "t" * 32}
    assert client.post("/v1/campaigns", json=_spec(), headers=headers).status_code == 200
    status = client.get("/v1/campaigns/campaign-1/status", headers=headers)
    assert status.status_code == 200
    assert status.json()["status"] == "accepted"
    assert status.json()["provider_internal_state_exposed"] is False


def test_queue_slo_fails_campaign_without_inventing_completion_eta(tmp_path):
    now = [100.0]
    control = ProductionGpuCampaignControlPlane(
        tmp_path / "control.sqlite", tmp_path / "artifacts", clock=lambda: now[0]
    )
    control.create_campaign(_spec(queue_timeout_seconds=30))
    now[0] = 130.0
    status = control.customer_status("campaign-1")
    assert status["status"] == "failed"
    assert status["estimated_completion_time"] is None
    snapshot = control.snapshot("campaign-1")
    assert snapshot["blocker"] == "queue_timeout"
    assert all(row["state"] == "cancelled" for row in snapshot["attempts"])


def test_campaign_rejects_tampered_spec_after_digest(tmp_path):
    control = _control(tmp_path)
    spec = _spec()
    spec["worker_image_ref"] = "registry/worker@sha256:" + "d" * 64
    with pytest.raises(CampaignControlPlaneError, match="spec_digest_mismatch"):
        control.create_campaign(spec)
