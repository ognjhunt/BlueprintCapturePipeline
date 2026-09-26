"""The three delivery modes share one qualified client and cleanup contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.core.security_controls import BoundedHttpResponse
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_g1_team_runtime_session import (
    SESSION_SCHEMA,
    open_g1_team_runtime_session,
)
from tests.test_native_g1_team_scored_scene_episode import _inputs, _run
from tests.test_team_policy_delivery_profile import OWNER, _profile, _setup


def _action() -> list[float]:
    action = [0.0] * 40
    action[3:9] = [1, 0, 0, 1, 0, 0]
    return action


def test_endpoint_probes_then_reuses_same_client_without_persisting_secret(tmp_path: Path) -> None:
    setup = _setup()
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
            body=json.dumps(response).encode(), status=200,
            content_type="application/json", final_url=url,
        )

    output = tmp_path / "endpoint"
    with open_g1_team_runtime_session(
        profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
        approved_binding={
            "mode": "authenticated_endpoint",
            "profile_digest": profile["profile_digest"],
            "approved_origin": "https://policy.example.org",
            "resolved_secret_ref": "secretref:team/policy",
        },
        output_dir=output, credential="private-token", fetcher=fetcher,
    ) as session:
        assert session.conformance["synthetic_policy_query_count"] == 1
        assert session.client.profile_digest == profile["profile_digest"]
        session.client.reset(seed=19)
        assert session.client.infer_chunk(
            front_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            observation_state=[0.0] * 64, task="pick the book",
        ) == [_action()]
    assert [request["kind"] for request in requests] == ["reset", "infer", "reset", "infer"]
    receipt = json.loads((output / (SESSION_SCHEMA + ".json")).read_text())
    assert receipt["status"] == "closed"
    assert receipt["child_teardown_required"] is False
    assert receipt["linked_scored_episode_result_digest"] is None
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    assert "private-token" not in "".join(path.read_text() for path in output.iterdir())


def test_qualified_endpoint_client_runs_same_scored_g1_scene(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import adp_task_scoring

    setup, _, scene = _inputs(tmp_path, monkeypatch)
    profile = _profile(setup, {
        "mode": "authenticated_endpoint",
        "endpoint_url": "https://policy.example.org/v1/action",
        "auth_secret_ref": "secretref:team/policy", "timeout_ms": 5000,
    })
    requests = []

    def fetcher(url, **options):
        request = json.loads(options["data"])
        requests.append(request)
        response = {
            "protocol": request["protocol"],
            "profile_digest": request["profile_digest"],
            "request_id": request["request_id"],
        }
        response.update({"ok": True} if request["kind"] == "reset" else {
            "action_chunk": [_action()]
        })
        return BoundedHttpResponse(
            body=json.dumps(response).encode(), status=200,
            content_type="application/json", final_url=url,
        )

    monkeypatch.setattr(
        adp_task_scoring, "score_task_episode_from_spec",
        lambda *, task_spec, samples: {
            "status": "scored", "outcome": "failure", "samples": len(samples),
        },
    )
    with open_g1_team_runtime_session(
        profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
        approved_binding={
            "mode": "authenticated_endpoint", "profile_digest": profile["profile_digest"],
            "approved_origin": "https://policy.example.org",
            "resolved_secret_ref": "secretref:team/policy",
        },
        output_dir=tmp_path / "runtime", credential="private-token", fetcher=fetcher,
    ) as session:
        result = _run(
            setup, profile, scene, session.client, tmp_path, monkeypatch,
            objective_id="task_success",
        )
        session.link_scored_episode(result)
    assert result["policy_query_count"] == 2
    assert result["score"]["status"] == "scored"
    assert [request["kind"] for request in requests] == [
        "reset", "infer", "reset", "infer", "infer",
    ]
    assert result["policy_runtime_identity_verified"] is False
    receipt = json.loads((tmp_path / "runtime" / (SESSION_SCHEMA + ".json")).read_text())
    assert receipt["linked_scored_episode_result_digest"] == result["result_digest"]
    assert receipt["linked_episode_media_verified_by_session"] is False


def test_runtime_session_rejects_a_mismatched_episode_link(tmp_path: Path, monkeypatch) -> None:
    setup, profile, _ = _inputs(tmp_path, monkeypatch)
    delivery = profile["delivery"]
    close_count = []

    class Lease:
        client = object()

        def close(self):
            close_count.append(1)
            return {"status": "container_removed", "receipt_digest": "sha256:" + "c" * 64}

    def launch(**kwargs):
        kwargs["output_dir"].mkdir()
        return Lease(), {
            "status": "synthetic_wire_compatible", "receipt_digest": "sha256:" + "d" * 64,
            "source_setup_digest": setup["setup_digest"],
        }

    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_runtime_session.launch_g1_team_container_synthetic_probe",
        launch,
    )
    with open_g1_team_runtime_session(
        profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
        approved_binding={
            "mode": "container", "profile_digest": profile["profile_digest"],
            "image_ref": delivery["image_ref"], "gpu_device": None,
        },
        output_dir=tmp_path / "runtime",
    ) as session:
        malformed = {
            "schema_version": "native_g1_team_scored_scene_episode.v1",
            "status": "development_only_scored_episode",
            "profile_digest": "sha256:" + "0" * 64,
            "candidate_id": "team_policy_" + "0" * 64,
            "source_setup_digest": setup["setup_digest"],
            "delivery_mode": "container", "policy_query_count": 1,
        }
        malformed["result_digest"] = canonical_digest(malformed, digest_field="result_digest")
        with pytest.raises(ValueError, match="episode_link_invalid"):
            session.link_scored_episode(malformed)
    assert close_count == [1]
    receipt = json.loads((tmp_path / "runtime" / (SESSION_SCHEMA + ".json")).read_text())
    assert receipt["linked_scored_episode_result_digest"] is None


@pytest.mark.parametrize("mode", ["container", "noncontainer_artifact"])
def test_process_modes_close_exact_lease(tmp_path: Path, monkeypatch, mode: str) -> None:
    setup = _setup()
    delivery = (
        {"mode": "container", "image_ref": "registry.example.org/team/g1@sha256:" + "a" * 64,
         "protocol": "jsonl_observation_action_v1"}
        if mode == "container" else
        {"mode": "noncontainer_artifact", "artifact_uri": "https://files.example.org/policy.tar.gz",
         "artifact_sha256": "sha256:" + "b" * 64, "entrypoint": "policy/run.py",
         "protocol": "jsonl_observation_action_v1"}
    )
    profile = _profile(setup, delivery)
    binding = {"mode": mode, "profile_digest": profile["profile_digest"]}
    if mode == "container":
        binding.update(image_ref=delivery["image_ref"], gpu_device=None)
    else:
        binding.update(
            artifact_sha256=delivery["artifact_sha256"],
            staged_artifact_path=str(tmp_path / "staged.tar.gz"),
        )
    close_count = []

    class Lease:
        client = object()

        def close(self):
            close_count.append(1)
            return {
                "status": "container_removed" if mode == "container" else "process_exited",
                "receipt_digest": "sha256:" + "c" * 64,
            }

    def launch(**kwargs):
        assert kwargs["operator_approved_profile_digest"] == profile["profile_digest"]
        kwargs["output_dir"].mkdir()
        return Lease(), {
            "status": "synthetic_wire_compatible",
            "receipt_digest": "sha256:" + "d" * 64,
        }

    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_runtime_session.launch_g1_team_"
        + ("container" if mode == "container" else "artifact") + "_synthetic_probe",
        launch,
    )
    output = tmp_path / "runtime"
    with open_g1_team_runtime_session(
        profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
        approved_binding=binding, output_dir=output,
    ) as session:
        assert session.client is Lease.client
    assert close_count == [1]
    assert session.close()["status"] == "closed"
    assert close_count == [1]
    receipt = json.loads((output / (SESSION_SCHEMA + ".json")).read_text())
    assert receipt["child_teardown_required"] is True
    assert receipt["child_teardown_digest"] == "sha256:" + "c" * 64
    assert receipt["provider_teardown_verified"] is False


def test_unapproved_binding_never_calls_endpoint(tmp_path: Path) -> None:
    setup = _setup()
    profile = _profile(setup, {
        "mode": "authenticated_endpoint",
        "endpoint_url": "https://policy.example.org/v1/action",
        "auth_secret_ref": "secretref:team/policy", "timeout_ms": 5000,
    })
    with pytest.raises(ValueError, match="binding_invalid"):
        open_g1_team_runtime_session(
            profile=profile, trusted_setup=setup, authenticated_owner=OWNER,
            approved_binding={
                "mode": "authenticated_endpoint",
                "profile_digest": "sha256:" + "0" * 64,
                "approved_origin": "https://policy.example.org",
                "resolved_secret_ref": "secretref:team/policy",
            },
            output_dir=tmp_path / "runtime", credential="private-token",
            fetcher=lambda *_args, **_kwargs: pytest.fail("endpoint was called"),
        )
    assert not (tmp_path / "runtime").exists()
