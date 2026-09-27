"""A reviewed endpoint must be probed, scored, and closed in one scene run."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from blueprint_pipeline.core.security_controls import BoundedHttpResponse
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_team_supervised_episode import (
    FILENAME,
    run_g1_team_supervised_episode,
)
from tests.test_native_g1_shared_scene_episode import _Bridge
from tests.test_native_g1_team_runtime_session import _action
from tests.test_native_g1_team_policy_approval import _approval
from tests.test_native_g1_team_scored_scene_episode import _inputs
from tests.test_native_task_episode_environment import _RigidNativeReadback
from tests.test_team_policy_delivery_profile import OWNER, _profile


def _run(tmp_path: Path, monkeypatch, *, reject_scene_inference: bool = False):
    from blueprint_pipeline import adp_task_scoring
    from blueprint_pipeline import native_g1_joint_episode_environment as g1_environment
    from blueprint_pipeline import native_task_arena_readback

    setup, _, scene = _inputs(tmp_path, monkeypatch)
    profile = _profile(setup, {
        "mode": "authenticated_endpoint",
        "endpoint_url": "https://policy.example.org/v1/action",
        "auth_secret_ref": "secretref:team/policy",
        "timeout_ms": 5000,
    })
    requests = []

    def fetcher(url, **options):
        request = json.loads(options["data"])
        requests.append(request)
        assert options["headers"]["Authorization"] == "Bearer private-token"
        response = {
            "protocol": request["protocol"],
            "profile_digest": request["profile_digest"],
            "request_id": request["request_id"],
        }
        response.update({"ok": True} if request["kind"] == "reset" else {
            "action_chunk": [_action()]
        })
        return BoundedHttpResponse(
            body=json.dumps(response).encode(),
            status=503 if reject_scene_inference and len(requests) == 4 else 200,
            content_type="application/json", final_url=url,
        )

    monkeypatch.setattr(g1_environment, "NativeG1JointEpisodeEnvironment", lambda **kwargs: scene)
    monkeypatch.setattr(
        native_task_arena_readback,
        "NativeRigidTaskArenaReadback",
        lambda built: _RigidNativeReadback(
            finger_separation_m=0.08,
            grasp_frame_position_world_m=[1.1, 2.1, 0.9],
            destination_scene_forbidden_contact_peak_force_n=0.0,
        ),
    )
    monkeypatch.setattr(
        adp_task_scoring,
        "score_task_episode_from_spec",
        lambda *, task_spec, samples: {
            "status": "scored", "outcome": "failure", "samples": len(samples),
        },
    )
    output = tmp_path / "run"
    binding = {
        "mode": "authenticated_endpoint",
        "profile_digest": profile["profile_digest"],
        "approved_origin": "https://policy.example.org",
        "resolved_secret_ref": "secretref:team/policy",
    }
    approval = _approval(setup, profile, binding)
    approval["expires_at_epoch"] = time.time() + 3600
    approval["approval_digest"] = canonical_digest(approval, digest_field="approval_digest")
    result = run_g1_team_supervised_episode(
        built=type("Built", (), {"plan": scene.plan})(),
        profile=profile,
        trusted_setup=setup,
        authenticated_owner=OWNER,
        operator_approval=approval,
        sonic_bridge=_Bridge(),
        objective_id="task_success",
        max_steps=2,
        output_dir=output,
        to_tensor=lambda value: value,
        make_action_tensor=lambda value, **kwargs: value,
        credential="private-token",
        fetcher=fetcher,
    )
    return result, output, requests


def test_supervised_team_episode_scores_and_closes_same_endpoint(tmp_path: Path, monkeypatch) -> None:
    result, output, requests = _run(tmp_path, monkeypatch)
    assert result["status"] == "completed_development_only"
    assert result["policy_query_count"] == 2
    assert result["synthetic_conformance_digest"] is not None
    assert result["operator_approval_digest"] is not None
    assert result["scored_episode_result_digest"] is not None
    assert result["runtime_teardown_digest"] is not None
    assert result["provider_teardown_verified"] is False
    assert result["official_billing_reconciled"] is False
    assert [request["kind"] for request in requests] == [
        "reset", "infer", "reset", "infer", "infer",
    ]
    assert list((output / "episode").rglob("*.mp4"))
    sealed = json.loads((output / FILENAME).read_text())
    assert sealed == result
    assert sealed["result_digest"] == canonical_digest(sealed, digest_field="result_digest")
    assert "private-token" not in "".join(path.read_text() for path in output.rglob("*.json"))


def test_supervised_team_episode_retains_failure_and_closes_runtime(tmp_path: Path, monkeypatch) -> None:
    result, output, requests = _run(tmp_path, monkeypatch, reject_scene_inference=True)
    assert result["status"] == "blocked"
    assert result["blocker_type"] == "ValueError"
    assert result["policy_query_count"] == 0
    assert result["runtime_teardown_digest"] is not None
    assert result["provider_teardown_verified"] is False
    assert [request["kind"] for request in requests] == ["reset", "infer", "reset", "infer"]
    assert "private-token" not in (output / FILENAME).read_text()


def test_supervised_team_episode_requires_site_exchange_approval_before_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    from blueprint_pipeline import native_g1_team_supervised_episode as supervised

    setup, _, scene = _inputs(tmp_path, monkeypatch)
    profile = _profile(setup, {
        "mode": "authenticated_endpoint",
        "endpoint_url": "https://policy.example.org/v1/action",
        "auth_secret_ref": "secretref:team/policy", "timeout_ms": 5000,
    })
    binding = {
        "mode": "authenticated_endpoint", "profile_digest": profile["profile_digest"],
        "approved_origin": "https://policy.example.org",
        "resolved_secret_ref": "secretref:team/policy",
    }
    approval = _approval(setup, profile, binding)
    approval["site_observation_exchange_authorized"] = False
    approval["approval_digest"] = canonical_digest(approval, digest_field="approval_digest")
    monkeypatch.setattr(
        supervised, "open_g1_team_runtime_session",
        lambda **kwargs: pytest.fail("runtime opened before site approval"),
    )
    output = tmp_path / "denied"
    with pytest.raises(ValueError, match="approval_binding_invalid"):
        run_g1_team_supervised_episode(
            built=type("Built", (), {"plan": scene.plan})(),
            profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
            operator_approval=approval, sonic_bridge=_Bridge(),
            objective_id="task_success", max_steps=2, output_dir=output,
            to_tensor=lambda value: value,
            make_action_tensor=lambda value, **kwargs: value,
        )
    assert not output.exists()
