from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_controls_visual_supervisor import (
    AUTHORIZATION_SCHEMA_VERSION,
    COST_SCOPE_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    ControlsVisualSupervisorError,
    run_controls_visual_supervisor,
)


def _sha(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _json_artifact(root: Path, name: str, value: dict[str, object]) -> dict[str, object]:
    path = root / name
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return _artifact(path)


def _request(tmp_path: Path, *, maximum_cost: float = 0.08) -> tuple[Path, Path]:
    renders = []
    for index in range(8):
        path = tmp_path / f"render-{index}.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([index]))
        renders.append({"camera_id": f"camera-{index}", "artifact": _artifact(path)})

    inputs: dict[str, object] = {}
    for name in (
        "configured_usd_manifest",
        "task_manifest",
        "robot_manifest",
        "camera_manifest",
        "native_construction_readback",
        "collision_result",
        "reachability_result",
    ):
        inputs[name] = _json_artifact(
            tmp_path,
            f"{name}.json",
            {"schema_version": f"{name}.v1", "status": "observed"},
        )
    inputs["zero_action_trace"] = _json_artifact(
        tmp_path,
        "zero.json",
        {
            "schema_version": "native_task_arena_control_result.v1",
            "control_selection": "zero_action_negative",
            "candidate_policy_queried": False,
            "status": "completed",
        },
    )
    inputs["scripted_positive_trace"] = _json_artifact(
        tmp_path,
        "positive.json",
        {
            "schema_version": "native_task_arena_control_result.v1",
            "control_selection": "deterministic_scripted_positive",
            "candidate_policy_queried": False,
            "status": "blocked",
        },
    )
    run_id = "configured-controls-scene-841007"
    cost = {
        "schema_version": COST_SCOPE_SCHEMA_VERSION,
        "status": "reserved_before_vlm_call",
        "run_id": run_id,
        "attempt": 1,
        "provider": "openai",
        "model": "gpt-5.4",
        "exclusive_scope": True,
        "zero_cost_baseline_confirmed": True,
        "maximum_cost_usd": maximum_cost,
        "cost_scope_digest": "",
    }
    cost["cost_scope_digest"] = canonical_digest(cost, digest_field="cost_scope_digest")
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "status": "authorized",
        "program_id": "arm-decision-proof-v1",
        "run_id": run_id,
        "provider": "openai",
        "model": "gpt-5.4",
        "private_derived_renders_disclosure_authorized": True,
        "configured_manifests_disclosure_authorized": True,
        "native_readback_and_deterministic_traces_disclosure_authorized": True,
        "raw_capture_or_splat_disclosure_authorized": False,
        "provider_training_authorized": False,
        "revision_proposal_only": True,
        "vlm_may_grade_controls": False,
        "issued_by_agent": False,
        "authorized_by": "controls-owner",
        "authorization_digest": "",
    }
    authorization["authorization_digest"] = canonical_digest(
        authorization, digest_field="authorization_digest"
    )
    request: dict[str, object] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "run_id": run_id,
        "attempt": 1,
        "max_attempts": 3,
        "renders": renders,
        "inputs": inputs,
        "revision_scope": {
            "allowed_kinds": ["camera", "base", "contact", "approach", "task"],
            "allowed_target_ids": {
                "camera": ["camera-0"],
                "base": ["franka-base"],
                "contact": ["contact-open"],
                "approach": ["approach"],
                "task": ["planar-relocation"],
            },
        },
        "vlm_authorization": authorization,
        "cost_scope": cost,
        "prior_attempt_receipts": [],
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    credential = tmp_path / "openai-key"
    credential.write_text("opaque-test-credential", encoding="utf-8")
    credential.chmod(0o600)
    return request_path, credential


def _proposal() -> dict[str, object]:
    return {
        "provider": "openai",
        "model": "gpt-5.4",
        "response_store": False,
        "tracing_disabled": True,
        "provider_request_id": "request-123",
        "reported_cost_usd": 0.02,
        "response": {
            "disposition": "propose_bounded_revision",
            "diagnoses": ["The external camera may hide the approach clearance."],
            "revisions": [
                {
                    "kind": "camera",
                    "target_id": "camera-0",
                    "parameters": {
                        "translation_delta_m": [0.01, 0.0, 0.0],
                        "rotation_delta_rad": [0.0, 0.02, 0.0],
                    },
                    "rationale": "Move the camera slightly while preserving calibration review.",
                }
            ],
            "uncertainty": "A deterministic rerun must test this hypothesis.",
        },
    }


def test_proposes_only_bounded_revision_and_never_grades_or_applies(tmp_path: Path) -> None:
    request, credential = _request(tmp_path)
    observed: dict[str, object] = {}

    def invoke(prompt, frames, secret, idempotency_key):
        observed.update(
            prompt=prompt,
            frame_count=len(frames),
            secret=secret,
            idempotency_key=idempotency_key,
        )
        return _proposal()

    result = run_controls_visual_supervisor(
        request_path=request,
        output_root=tmp_path / "output",
        credential_file=credential,
        invoker=invoke,
    )

    assert result["status"] == "bounded_revision_proposed"
    assert result["controls_qualified"] is False
    assert result["vlm_may_grade_success"] is False
    assert result["revision_applied"] is False
    assert result["requires_deterministic_rerun"] is True
    assert result["scripted_controls_authoritative"] is True
    assert observed["frame_count"] == 8
    assert observed["secret"] == "opaque-test-credential"
    assert observed["idempotency_key"] == result["source_request_digest"]
    retained = json.dumps(result, sort_keys=True)
    assert "opaque-test-credential" not in retained
    assert str(credential) not in retained
    assert str(tmp_path) not in retained


def test_exact_replay_is_idempotent_without_second_provider_call(tmp_path: Path) -> None:
    request, credential = _request(tmp_path)
    calls = 0

    def invoke(*_args):
        nonlocal calls
        calls += 1
        return _proposal()

    first = run_controls_visual_supervisor(
        request_path=request,
        output_root=tmp_path / "output",
        credential_file=credential,
        invoker=invoke,
    )
    second = run_controls_visual_supervisor(
        request_path=request,
        output_root=tmp_path / "output",
        credential_file=credential,
        invoker=invoke,
    )

    assert first == second
    assert calls == 1


def test_replay_rehashes_inputs_and_refuses_rehashed_authority_escalation(
    tmp_path: Path,
) -> None:
    request, credential = _request(tmp_path)
    output = tmp_path / "output"
    result = run_controls_visual_supervisor(
        request_path=request,
        output_root=output,
        credential_file=credential,
        invoker=lambda *_args: _proposal(),
    )
    result_path = output / "controls_visual_supervisor_attempt_1.v1.json"
    result["controls_qualified"] = True
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ControlsVisualSupervisorError, match="existing_result_conflict"):
        run_controls_visual_supervisor(
            request_path=request,
            output_root=output,
            credential_file=credential,
            invoker=lambda *_args: pytest.fail("provider must not be called"),
        )

    result["controls_qualified"] = False
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    (tmp_path / "render-0.png").write_bytes(b"changed")

    with pytest.raises(ControlsVisualSupervisorError, match="render_invalid"):
        run_controls_visual_supervisor(
            request_path=request,
            output_root=output,
            credential_file=credential,
            invoker=lambda *_args: pytest.fail("provider must not be called"),
        )


def test_tampered_render_fails_before_provider_call(tmp_path: Path) -> None:
    request, credential = _request(tmp_path)
    (tmp_path / "render-3.png").write_bytes(b"changed")

    with pytest.raises(ControlsVisualSupervisorError, match="render_invalid"):
        run_controls_visual_supervisor(
            request_path=request,
            output_root=tmp_path / "output",
            credential_file=credential,
            invoker=lambda *_args: pytest.fail("provider must not be called"),
        )


def test_requires_exactly_eight_digest_bound_renders(tmp_path: Path) -> None:
    request_path, credential = _request(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["renders"].pop()
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ControlsVisualSupervisorError, match="exactly_8_renders"):
        run_controls_visual_supervisor(
            request_path=request_path,
            output_root=tmp_path / "output",
            credential_file=credential,
            invoker=lambda *_args: pytest.fail("provider must not be called"),
        )


def test_secret_like_authoritative_input_fails_before_disclosure(tmp_path: Path) -> None:
    request_path, credential = _request(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    unsafe = tmp_path / "robot_manifest.json"
    unsafe.write_text(json.dumps({"api_key": "must-not-leave"}), encoding="utf-8")
    request["inputs"]["robot_manifest"] = _artifact(unsafe)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ControlsVisualSupervisorError, match="secret_unsafe"):
        run_controls_visual_supervisor(
            request_path=request_path,
            output_root=tmp_path / "output",
            credential_file=credential,
            invoker=lambda *_args: pytest.fail("provider must not be called"),
        )


def test_credentials_must_be_regular_0600_secret_file(tmp_path: Path) -> None:
    request, credential = _request(tmp_path)
    credential.chmod(0o644)

    with pytest.raises(ControlsVisualSupervisorError, match="credential_invalid"):
        run_controls_visual_supervisor(
            request_path=request,
            output_root=tmp_path / "output",
            credential_file=credential,
            invoker=lambda *_args: pytest.fail("provider must not be called"),
        )


def test_out_of_bounds_revision_abstains_and_retains_no_revision(tmp_path: Path) -> None:
    request, credential = _request(tmp_path)
    response = _proposal()
    response["response"]["revisions"][0]["parameters"]["translation_delta_m"] = [0.051, 0, 0]

    result = run_controls_visual_supervisor(
        request_path=request,
        output_root=tmp_path / "output",
        credential_file=credential,
        invoker=lambda *_args: response,
    )

    assert result["status"] == "abstained_fail_closed"
    assert result["bounded_revisions"] == []
    assert result["diagnosis"] is None
    assert result["controls_qualified"] is False
    assert result["blockers"] == ["vlm_diagnostic_abstained:ControlsVisualSupervisorError"]


def test_model_cannot_add_a_self_grade_field(tmp_path: Path) -> None:
    request, credential = _request(tmp_path)
    response = _proposal()
    response["response"]["controls_qualified"] = True

    result = run_controls_visual_supervisor(
        request_path=request,
        output_root=tmp_path / "output",
        credential_file=credential,
        invoker=lambda *_args: response,
    )

    assert result["status"] == "abstained_fail_closed"
    assert result["diagnosis"] is None
    assert result["controls_qualified"] is False


def test_cost_scope_must_be_digest_bound_and_under_hard_cap(tmp_path: Path) -> None:
    request_path, credential = _request(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["cost_scope"]["maximum_cost_usd"] = 0.26
    request["cost_scope"]["cost_scope_digest"] = canonical_digest(
        request["cost_scope"], digest_field="cost_scope_digest"
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ControlsVisualSupervisorError, match="cost_scope_invalid"):
        run_controls_visual_supervisor(
            request_path=request_path,
            output_root=tmp_path / "output",
            credential_file=credential,
            invoker=lambda *_args: pytest.fail("provider must not be called"),
        )


def test_control_pair_must_be_deterministic_and_policy_free(tmp_path: Path) -> None:
    request_path, credential = _request(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    trace = tmp_path / "positive.json"
    value = json.loads(trace.read_text(encoding="utf-8"))
    value["candidate_policy_queried"] = True
    trace.write_text(json.dumps(value), encoding="utf-8")
    request["inputs"]["scripted_positive_trace"] = _artifact(trace)
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ControlsVisualSupervisorError, match="control_trace_invalid"):
        run_controls_visual_supervisor(
            request_path=request_path,
            output_root=tmp_path / "output",
            credential_file=credential,
            invoker=lambda *_args: pytest.fail("provider must not be called"),
        )


def test_attempts_are_hard_capped_and_require_an_exact_prior_chain(tmp_path: Path) -> None:
    request_path, credential = _request(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["max_attempts"] = 4
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ControlsVisualSupervisorError, match="request_invalid"):
        run_controls_visual_supervisor(
            request_path=request_path,
            output_root=tmp_path / "output",
            credential_file=credential,
            invoker=lambda *_args: pytest.fail("provider must not be called"),
        )

    request["max_attempts"] = 3
    request["attempt"] = 2
    request["cost_scope"]["attempt"] = 2
    request["cost_scope"]["cost_scope_digest"] = canonical_digest(
        request["cost_scope"], digest_field="cost_scope_digest"
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ControlsVisualSupervisorError, match="attempt_chain_invalid"):
        run_controls_visual_supervisor(
            request_path=request_path,
            output_root=tmp_path / "output",
            credential_file=credential,
            invoker=lambda *_args: pytest.fail("provider must not be called"),
        )


def test_agent_cannot_authorize_its_own_visual_revision_scope(tmp_path: Path) -> None:
    request_path, credential = _request(tmp_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["vlm_authorization"]["issued_by_agent"] = True
    request["vlm_authorization"]["authorization_digest"] = canonical_digest(
        request["vlm_authorization"], digest_field="authorization_digest"
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(ControlsVisualSupervisorError, match="authorization_invalid"):
        run_controls_visual_supervisor(
            request_path=request_path,
            output_root=tmp_path / "output",
            credential_file=credential,
            invoker=lambda *_args: pytest.fail("provider must not be called"),
        )
