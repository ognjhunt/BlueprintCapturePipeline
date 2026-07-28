from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from blueprint_pipeline.policy_ranking_roboarena_evaluator_openai import (
    _score_one,
    GATE_ENV,
    evaluator_contract,
    run_evaluator_inventory,
    supersede_transport_inventory_v3,
)
from blueprint_pipeline.policy_ranking_roboarena_calibration import canonical_sha256


def test_contract_freezes_mini_snapshot_and_label_blind_payload() -> None:
    contract = evaluator_contract()
    assert contract["model"] == "gpt-5-mini-2025-08-07"
    assert contract["frame_count"] == 32
    assert contract["policy_identity_in_provider_payload"] is False
    assert contract["benchmark_outcomes_in_provider_payload"] is False
    assert contract["physical_ground_truth_pixels_in_provider_payload"] is False
    assert contract["idempotency_header"] == "request_id"
    amendment = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/experiments/policy_ranking_roboarena_full_stack_calibration_20260728"
            / "evaluator_transport_amendment_v3.json"
        ).read_text(encoding="utf-8")
    )
    assert amendment["new_evaluator_digest"] == contract["evaluator_digest"]
    assert amendment["prompt_changed"] is False


def test_score_sends_only_registered_metadata_and_32_audited_images(tmp_path: Path) -> None:
    frames = []
    for index in range(32):
        path = tmp_path / f"{index}.jpg"
        path.write_bytes(b"audited-generated-image-" + str(index).encode())
        import hashlib

        frames.append(
            {
                "sample_position": index,
                "source_frame_index": index,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    class Responses:
        def create(self, **kwargs):
            assert kwargs["extra_headers"] == {"Idempotency-Key": "r1"}
            content = kwargs["input"][0]["content"]
            text = content[0]["text"]
            assert "secret-policy-name" not in text
            assert len([row for row in content if row["type"] == "input_image"]) == 32
            return types.SimpleNamespace(
                id="resp_test",
                status="completed",
                usage=types.SimpleNamespace(
                    input_tokens=1000,
                    output_tokens=500,
                    input_tokens_details=types.SimpleNamespace(cached_tokens=0),
                ),
                output_text=json.dumps(
                    {
                        "task_specific_rubric": ["start", "contact", "complete"],
                        "progress_score_0_to_5": 2,
                        "success_probability": 0.2,
                        "stable_success_confirmed": False,
                        "stable_success_frame_positions": [],
                        "success_evidence": [],
                        "failure_evidence": ["not complete"],
                        "artifact_flags": ["none"],
                        "temporal_consistency": 0.8,
                        "action_following_confidence": 0.5,
                        "uncertainty": 0.3,
                        "abstain": False,
                        "abstention_factors": [],
                    }
                ),
            )

    result = _score_one(
        types.SimpleNamespace(responses=Responses()),
        {
            "request_id": "r1",
            "source_request_id": "s1",
            "session_id": "session",
            "policy_id_internal_only": "secret-policy-name",
            "task_instruction": "put the cup near the plate",
            "frames": frames,
            "short_episode_source": False,
            "unique_sampled_frame_count": 32,
            "repeated_sample_count": 0,
            "evaluator_digest": evaluator_contract()["evaluator_digest"],
            "deterministic_collapse_flags": [],
            "deterministic_safety_abstention_recommended": False,
        },
    )
    assert result["policy_identity_sent_to_provider"] is False
    assert result["structured_response"]["progress_score_0_to_5"] == 2
    assert result["usage"]["estimated_cost_usd"] > 0


def test_transport_v3_supersession_rebinds_idempotent_request_before_provider(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"audited")
    import hashlib

    prior = {
        "schema_version": "policy_ranking_roboarena_evaluator_inventory.v2",
        "status": "ready",
        "evaluator": {"evaluator_digest": "old-digest"},
        "request_count": 1,
        "requests": [
            {
                "request_id": "old-request-id",
                "source_request_id": "source-id",
                "cropped_output_sha256": "crop-digest",
                "task_instruction": "move object",
                "frames": [
                    {
                        "path": str(frame),
                        "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    }
                ],
            }
        ],
        "blockers": [],
        "provider_called": False,
        "data_uploaded": False,
        "outcome_labels_accessed": False,
    }
    prior["inventory_sha256"] = canonical_sha256(prior)
    amended = supersede_transport_inventory_v3(prior, expected_request_count=1)
    assert amended["status"] == "ready"
    assert amended["schema_version"].endswith(".v3")
    assert amended["evaluator"]["idempotency_header"] == "request_id"
    assert amended["requests"][0]["request_id"] != "old-request-id"
    assert amended["transport_amendment"]["provider_called"] is False

    prior["provider_called"] = True
    prior["inventory_sha256"] = canonical_sha256(
        {key: value for key, value in prior.items() if key != "inventory_sha256"}
    )
    blocked = supersede_transport_inventory_v3(prior, expected_request_count=1)
    assert blocked["status"] == "blocked"
    assert "prior_inventory_provider_already_called" in blocked["blockers"]


def test_live_runner_exact_cap_and_digest_valid_resume_are_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    frames = []
    import hashlib

    for index in range(32):
        path = tmp_path / f"frame-{index}.jpg"
        path.write_bytes(f"generated-{index}".encode())
        frames.append(
            {
                "sample_position": index,
                "source_frame_index": index,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    contract = evaluator_contract()
    request = {
        "request_id": "request-idempotent",
        "source_request_id": "source-id",
        "session_id": "session-id",
        "policy_id_internal_only": "policy-id",
        "task_instruction": "move the object",
        "frames": frames,
        "short_episode_source": False,
        "unique_sampled_frame_count": 32,
        "repeated_sample_count": 0,
        "evaluator_digest": contract["evaluator_digest"],
        "deterministic_collapse_flags": [],
        "deterministic_safety_abstention_recommended": False,
    }
    inventory = {
        "schema_version": "policy_ranking_roboarena_evaluator_inventory.v3",
        "status": "ready",
        "evaluator": contract,
        "request_count": 1,
        "requests": [request],
        "provider_called": False,
        "data_uploaded": False,
        "outcome_labels_accessed": False,
    }
    inventory["inventory_sha256"] = canonical_sha256(inventory)
    key = tmp_path / "key"
    key.write_text("secure-test-key", encoding="utf-8")
    key.chmod(0o600)
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(
            {
                "schema_version": "blueprint_openai_key_rotation_attestation.v1",
                "recorded_at": "2026-07-28T00:00:00Z",
                "previous_chat_exposed_key_revoked": True,
                "replacement_key_created_after_revocation": True,
                "attested_by": "user",
            }
        ),
        encoding="utf-8",
    )
    attestation.chmod(0o600)

    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(
                id="resp-live-test",
                status="completed",
                usage=types.SimpleNamespace(
                    input_tokens=100,
                    output_tokens=50,
                    input_tokens_details=types.SimpleNamespace(cached_tokens=0),
                ),
                output_text=json.dumps(
                    {
                        "task_specific_rubric": ["start", "motion", "finish"],
                        "progress_score_0_to_5": 3,
                        "success_probability": 0.4,
                        "stable_success_confirmed": False,
                        "stable_success_frame_positions": [],
                        "success_evidence": [],
                        "failure_evidence": ["not complete"],
                        "artifact_flags": ["none"],
                        "temporal_consistency": 0.8,
                        "action_following_confidence": 0.8,
                        "uncertainty": 0.2,
                        "abstain": False,
                        "abstention_factors": [],
                    }
                ),
            )

    class OpenAI:
        def __init__(self, *, api_key):
            assert api_key == "secure-test-key"
            self.responses = Responses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))
    monkeypatch.setenv(GATE_ENV, "true")
    output = tmp_path / "run"
    kwargs = {
        "output_root": output,
        "api_key_file": key,
        "rotation_attestation_file": attestation,
        "max_requests": 1,
        "max_cost_usd": 0.05,
        "max_workers": 1,
        "source_commit": "a" * 40,
    }
    first = run_evaluator_inventory(inventory, **kwargs)
    assert first["status"] == "completed"
    assert len(calls) == 1
    assert calls[0]["extra_headers"] == {"Idempotency-Key": "request-idempotent"}
    second = run_evaluator_inventory(inventory, **kwargs)
    assert second["status"] == "completed"
    assert len(calls) == 1

    result_path = output / "requests/request-idempotent/result.json"
    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    persisted["structured_response"]["progress_score_0_to_5"] = 0
    result_path.write_text(json.dumps(persisted), encoding="utf-8")
    third = run_evaluator_inventory(inventory, **kwargs)
    assert third["status"] == "blocked"
    assert third["blockers"] == ["persisted_result_invalid:request-idempotent"]
    assert len(calls) == 1
