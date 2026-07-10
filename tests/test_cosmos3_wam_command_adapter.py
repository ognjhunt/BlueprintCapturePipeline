from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline import cosmos3_wam_command_adapter as adapter
from blueprint_pipeline import wam_backend_strategy as backend_strategy
from blueprint_pipeline.sc3_fidelity_contracts import (
    SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
)


ADAPTER_ENV_VARS = (
    "BLUEPRINT_COSMOS3_WAM_SOURCE_ROOT",
    "BLUEPRINT_COSMOS3_NANO_SOURCE_ROOT",
    "BLUEPRINT_COSMOS3_SOURCE_ROOT",
    "BLUEPRINT_COSMOS3_WAM_CHECKPOINT",
    "BLUEPRINT_COSMOS3_NANO_CHECKPOINT",
    "BLUEPRINT_WAM_MODEL_CHECKPOINT",
    "BLUEPRINT_ALLOW_LOCAL_WAM_MODEL",
    "BLUEPRINT_COSMOS3_WAM_ENTRYPOINT",
    "BLUEPRINT_WAM_ROLLOUT_INPUT",
    "BLUEPRINT_WAM_ROLLOUT_OUTPUT",
    "BLUEPRINT_WAM_MODEL_CANDIDATE",
)


def _clear_adapter_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ADAPTER_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _signed_attestation(payload: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    signed_payload_sha256 = hashlib.sha256(message).hexdigest()
    report = tmp_path / "executor-verification.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "sc3_signature_verification_report.v1",
                "algorithm": "Ed25519",
                "verification_status": "verified",
                "public_key_sha256": public_key_sha256,
                "signed_payload_sha256": signed_payload_sha256,
                "signer_key_id": "test-signer",
                "verifier_id": "test-verifier",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "algorithm": "Ed25519",
        "signature_verified": True,
        "signer_key_id": "test-signer",
        "verifier_id": "test-verifier",
        "public_key_base64": base64.b64encode(public_key).decode("ascii"),
        "public_key_sha256": public_key_sha256,
        "signature_base64": base64.b64encode(private_key.sign(message)).decode("ascii"),
        "signed_payload_sha256": signed_payload_sha256,
        "verification_report_artifact": {
            "path": str(report),
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        },
    }


def _source_root(tmp_path: Path) -> Path:
    source_root = tmp_path / "cosmos3-source"
    (source_root / "examples").mkdir(parents=True)
    (source_root / "examples" / "action_conditioned.py").write_text("# cosmos3\n", encoding="utf-8")
    (source_root / "pyproject.toml").write_text(
        '[project]\nname = "cosmos3-nano-oss"\n', encoding="utf-8"
    )
    return source_root


def _checkpoint(tmp_path: Path, *, base_model: str | None) -> Path:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "weights.bin").write_bytes(b"weights")
    if base_model is not None:
        _write_json(checkpoint / "config.json", {"base_model": base_model})
    return checkpoint


def _output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    output_path = tmp_path / "out" / "wam_provider_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output_path))
    return output_path


def test_adapter_blocks_and_writes_typed_payload_when_unconfigured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_adapter_env(monkeypatch)
    output_path = _output_path(tmp_path, monkeypatch)

    payload = adapter.run([])

    assert payload["schema_version"] == "cosmos3_wam_command_adapter.v1"
    assert payload["status"] == "blocked"
    assert "blocked_missing_cosmos3_source_root" in payload["blockers"]
    assert "blocked_missing_cosmos3_checkpoint" in payload["blockers"]
    assert payload["learned_wam_model_ran"] is False
    assert payload["fresh_model_run_claimed"] is False
    assert payload["evaluation_substrate"] == "cosmos3_wam"
    assert payload["raw_credentials_written_to_artifacts"] is False
    assert payload["run_gates"]["auto_run_allowed_without_gate"] is False
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["status"] == "blocked"


def test_adapter_fails_closed_on_wrong_family_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_adapter_env(monkeypatch)
    _output_path(tmp_path, monkeypatch)
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    source_root = _source_root(tmp_path)
    checkpoint = _checkpoint(tmp_path, base_model="Cosmos-Predict2.5-2B")

    def _must_not_run(**kwargs: Any) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("wrong-family checkpoint must never launch the model")

    monkeypatch.setattr(adapter, "_run_cosmos3", _must_not_run)
    monkeypatch.setattr(adapter, "_run_import_probe", _must_not_run)

    payload = adapter.run(["--source-root", str(source_root), "--checkpoint", str(checkpoint)])

    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["blocked_wrong_model_family_checkpoint_for_cosmos3_wam"]
    assert payload["learned_wam_model_ran"] is False
    probe = payload["checkpoint_identity_probe"]
    assert probe["wrong_model_family_detected"] is True
    assert "Cosmos-Predict2.5-2B" in probe["declared_identity_values"]


def test_adapter_fails_closed_on_unverified_checkpoint_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_adapter_env(monkeypatch)
    _output_path(tmp_path, monkeypatch)
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    source_root = _source_root(tmp_path)
    checkpoint = _checkpoint(tmp_path, base_model=None)

    payload = adapter.run(["--source-root", str(source_root), "--checkpoint", str(checkpoint)])

    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["blocked_cosmos3_checkpoint_identity_unverified"]
    assert payload["learned_wam_model_ran"] is False


def test_adapter_respects_local_model_run_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_adapter_env(monkeypatch)
    _output_path(tmp_path, monkeypatch)
    source_root = _source_root(tmp_path)
    checkpoint = _checkpoint(tmp_path, base_model="nvidia/Cosmos3-Nano")
    monkeypatch.setattr(
        adapter,
        "_run_import_probe",
        lambda **kwargs: {"status": "completed", "blockers": []},
    )

    def _must_not_run(**kwargs: Any) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("model must never run without the explicit gate")

    monkeypatch.setattr(adapter, "_run_cosmos3", _must_not_run)

    payload = adapter.run(["--source-root", str(source_root), "--checkpoint", str(checkpoint)])

    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["blocked_BLUEPRINT_ALLOW_LOCAL_WAM_MODEL_not_enabled"]
    assert payload["run_gates"]["local_model_gate_enabled"] is False
    assert payload["learned_wam_model_ran"] is False


def test_adapter_emits_trusted_schema_with_verified_identity_and_declared_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_adapter_env(monkeypatch)
    output_path = _output_path(tmp_path, monkeypatch)
    monkeypatch.setenv("BLUEPRINT_ALLOW_LOCAL_WAM_MODEL", "true")
    source_root = _source_root(tmp_path)
    checkpoint = _checkpoint(tmp_path, base_model="Cosmos3-Nano")
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(rollout_input, {"task_prompts": []})
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))

    save_root = tmp_path / "generated"
    save_root.mkdir()
    package_manifest = {
        "schema_version": "blueprint_cosmos_rollout_input_package.v1",
        "save_root": str(save_root),
        "inference_params_path": str(tmp_path / "params.json"),
        "source_review_video_path": str(tmp_path / "review.mp4"),
        "source_camera": "robot_pov",
        "scenario_eval_run_id": "run-1",
        "task_id": "task-1",
        "spawn_id": "spawn-1",
    }
    monkeypatch.setattr(
        adapter,
        "_materialize_cosmos_input_package",
        lambda **kwargs: dict(package_manifest),
    )
    monkeypatch.setattr(
        adapter,
        "_run_import_probe",
        lambda **kwargs: {"status": "completed", "blockers": []},
    )

    def _fake_run_cosmos3(**kwargs: Any) -> dict[str, Any]:
        (save_root / "rollout_0001.mp4").write_bytes(b"generated mp4")
        return {
            "schema_version": "cosmos3_subprocess_result.v1",
            "status": "completed",
            "returncode": 0,
            "blockers": [],
        }

    monkeypatch.setattr(adapter, "_run_cosmos3", _fake_run_cosmos3)
    monkeypatch.setattr(
        adapter,
        "validate_generated_mp4_for_review",
        lambda path: {"status": "completed", "blockers": []},
    )

    payload = adapter.run(["--source-root", str(source_root), "--checkpoint", str(checkpoint)])

    assert payload["schema_version"] == "cosmos3_wam_command_adapter.v1"
    assert payload["status"] == "completed"
    assert payload["base_model"] == "Cosmos3-Nano"
    assert payload["learned_wam_model_ran"] is True
    assert payload["fresh_model_run_claimed"] is True
    assert payload["rollouts"][0]["base_model"] == "Cosmos3-Nano"
    assert payload["rollouts"][0]["success_label_source"] == ("generated_video_requires_review")
    truth = payload["truth_boundary"]
    assert truth["checkpoint_identity_verified_as_cosmos3_nano"] is True
    assert truth["sc3_recipe_metadata_is_declared_config_not_proof"] is True
    assert truth["generated_world_rank_fidelity_result_proven"] is False
    recipe = payload["sc3_recipe_declared_config"]
    assert recipe["training_mixture"] == {
        "forward_dynamics": 0.8,
        "cross_view": 0.1,
        "inverse_dynamics": 0.1,
    }
    assert recipe["horizon_decoupling"] == {
        "predict_horizon_frames": 24,
        "execute_horizon_frames": 16,
    }
    assert recipe["claim_boundary"]["recipe_metadata_is_operator_declared_config"] is True
    assert recipe["claim_boundary"]["recipe_metadata_is_execution_or_training_proof"] is False
    assert payload["raw_credentials_written_to_artifacts"] is False
    assert payload["secret_hashes_written_to_artifacts"] is False
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "completed"

    from blueprint_pipeline.oscar_cosmos_wam_evaluator import (
        TRUSTED_WAM_MODEL_PAYLOAD_SCHEMAS,
    )

    assert "cosmos3_wam_command_adapter.v1" in TRUSTED_WAM_MODEL_PAYLOAD_SCHEMAS


def test_rollout_payload_never_claims_learned_run_without_verified_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_root = tmp_path / "generated"
    save_root.mkdir()
    (save_root / "rollout_0001.mp4").write_bytes(b"generated mp4")
    monkeypatch.setattr(
        adapter,
        "validate_generated_mp4_for_review",
        lambda path: {"status": "completed", "blockers": []},
    )

    payload = adapter._rollout_payload(
        package_manifest={"save_root": str(save_root)},
        checkpoint=tmp_path / "checkpoint",
        source_root=tmp_path,
        subprocess_detail={"status": "completed", "blockers": []},
        model="cosmos3/nano/action-cond",
        checkpoint_identity={"checkpoint_identity_verified": False},
        source_identity={"source_identity_verified": True},
    )

    assert payload["status"] == "completed"
    assert payload["fresh_model_command_executed_this_invocation"] is True
    assert payload["learned_wam_model_ran"] is False
    assert payload["fresh_model_run_claimed"] is False


def test_checkpoint_identity_probe_recognizes_cosmos3_and_wrong_families(
    tmp_path: Path,
) -> None:
    verified = _checkpoint(tmp_path / "good", base_model="nvidia/Cosmos3-Nano")
    probe = adapter.checkpoint_identity_probe(verified)
    assert probe["checkpoint_identity_verified"] is True
    assert probe["sc3_trained_checkpoint_proven"] is False
    assert probe["sc3_checkpoint_attestation_validation"]["status"] == "blocked"
    assert probe["wrong_model_family_detected"] is False

    wrong = _checkpoint(tmp_path / "bad", base_model="Cosmos3-Super")
    probe = adapter.checkpoint_identity_probe(wrong)
    assert probe["checkpoint_identity_verified"] is False
    assert probe["wrong_model_family_detected"] is True

    oscar = _checkpoint(tmp_path / "oscar", base_model="OSCAR-2B")
    probe = adapter.checkpoint_identity_probe(oscar)
    assert probe["wrong_model_family_detected"] is True

    weights_file = tmp_path / "flat" / "weights.bin"
    weights_file.parent.mkdir(parents=True)
    weights_file.write_bytes(b"weights")
    _write_json(weights_file.parent / "config.json", {"model_name": "Cosmos3-Nano"})
    probe = adapter.checkpoint_identity_probe(weights_file)
    assert probe["checkpoint_identity_verified"] is True


def test_cosmos3_strategy_state_is_derived_from_machine_checked_preconditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_adapter_env(monkeypatch)
    monkeypatch.delenv("BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND", raising=False)
    monkeypatch.delenv("BLUEPRINT_COSMOS3_CALIBRATION_ANCHORS_PATH", raising=False)

    row = backend_strategy.get_wam_backend_strategy("cosmos3_wam")
    assert row["aspirational"] is True
    assert row["preferred_candidate_state"] == "aspirational"
    preconditions = row["preconditions"]
    assert preconditions["state_derived_from_machine_checks"] is True
    assert preconditions["state_asserted_manually"] is False
    assert preconditions["checks"]["adapter_module_present"]["passed"] is True
    assert preconditions["checks"]["consistency_scorer_available"]["passed"] is False
    assert preconditions["checks"]["calibration_anchors_present"]["passed"] is False

    manifest = backend_strategy.build_wam_backend_strategy_manifest(generated_at="now")
    assert manifest["cosmos3_wam_aspirational"] is True
    assert manifest["cosmos3_wam_preferred_candidate_state"] == "aspirational"
    assert (
        manifest["claim_boundary"][
            "cosmos3_preferred_candidate_state_is_machine_derived_not_asserted"
        ]
        is True
    )
    by_id = {r["backend_id"]: r for r in manifest["backend_strategies"]}
    assert by_id["cosmos3_wam"]["aspirational"] is True

    anchors_path = tmp_path / "accepted_anchors.json"
    _write_json(
        anchors_path,
        {
            "schema_version": "accepted_real_world_anchor.v1",
            "anchors": [
                {
                    "scenario_eval_run_id": "run-1",
                    "policy_id": "policy-1",
                    "task_id": "task-1",
                    "scenario_variation_instance_id": "var-1",
                    "actual_success": True,
                }
            ],
        },
    )
    monkeypatch.setenv("BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND", "/opt/cosmos3-adapter")
    derived = backend_strategy.evaluate_cosmos3_wam_preconditions(
        calibration_anchors_path=anchors_path,
        consistency_scorer_available=True,
    )
    assert derived["preconditions_met"] is True
    assert derived["aspirational"] is False
    assert derived["preferred_candidate_state"] == "preferred_configured_candidate"


def test_sc3_horizon_trace_validates_runtime_emitted_25_24_16_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    action_vector = [0.0] * 7
    action_sha256 = hashlib.sha256(
        json.dumps(action_vector, separators=(",", ":")).encode()
    ).hexdigest()
    actions = [
        {
            "action_id": f"action-{index:02d}",
            "action_vector_7d": action_vector,
            "action_sha256": action_sha256,
        }
        for index in range(25)
    ]
    predictions = []
    for index, action in enumerate(actions[:24]):
        prediction_id = f"prediction-{index:02d}"
        runtime_result_id = f"prediction-runtime-{index:02d}"
        evidence = tmp_path / f"prediction-evidence-{index:02d}.json"
        _write_json(
            evidence,
            {
                "schema_version": "sc3_world_model_prediction_evidence.v1",
                "status": "completed",
                "runtime_session_id": "runtime-session-1",
                "runtime_result_id": runtime_result_id,
                "prediction_id": prediction_id,
                "action_id": action["action_id"],
                "action_sha256": action["action_sha256"],
                "world_model_checkpoint_sha256": "d" * 64,
            },
        )
        predictions.append(
            {
                **action,
                "prediction_result_schema_version": ("sc3_world_model_prediction_result.v1"),
                "prediction_id": prediction_id,
                "prediction_runtime_result_id": runtime_result_id,
                "prediction_evidence_artifact": {
                    "path": str(evidence),
                    "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                },
                "prediction_index": index,
                "prediction_status": "completed",
            }
        )
    executed = []
    for index, action in enumerate(actions[:16]):
        runtime_result_id = f"controller-runtime-{index:02d}"
        timestamp = 2.0 + index / 20.0
        evidence = tmp_path / f"controller-evidence-{index:02d}.json"
        _write_json(
            evidence,
            {
                "schema_version": "sc3_controller_execution_evidence.v1",
                "status": "completed",
                "runtime_session_id": "runtime-session-1",
                "runtime_result_id": runtime_result_id,
                "action_id": action["action_id"],
                "action_sha256": action["action_sha256"],
                "controller_id": "controller-1",
                "controller_sha256": "c" * 64,
                "execution_timestamp_sec": timestamp,
            },
        )
        executed.append(
            {
                **action,
                "execution_result_schema_version": ("sc3_controller_execution_result.v1"),
                "controller_runtime_result_id": runtime_result_id,
                "controller_evidence_artifact": {
                    "path": str(evidence),
                    "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
                },
                "execution_status": "executed",
                "execution_timestamp_sec": timestamp,
            }
        )
    runtime_trace = {
        "trace_producer_id": "blueprint_sc3_receding_horizon_executor",
        "runtime_session_id": "runtime-session-1",
        "runtime_executor_id": "executor-1",
        "runtime_executor_code_sha256": "b" * 64,
        "controller_id": "controller-1",
        "controller_sha256": "c" * 64,
        "world_model_checkpoint_sha256": "d" * 64,
        "runtime_execution_proven": True,
        "world_model_prediction_proven": True,
        "receding_horizon_controller_proven": True,
        "proposed_actions": actions,
        "world_model_predictions": predictions,
        "retained_actions": [
            {**action, "retention_status": "retained_for_execution"} for action in actions[:16]
        ],
        "executed_actions": executed,
        "discarded_predictions": [
            {
                **action,
                "retention_status": "discarded_not_executed",
                "executed": False,
            }
            for action in actions[16:24]
        ],
        "control_rate_hz": 20.0,
        "chunk_start_timestamp_sec": 2.0,
        "requery_timestamp_sec": 2.8,
    }
    trace_artifact = tmp_path / "executor-trace.json"
    trace_artifact.write_text(
        json.dumps(
            {"schema_version": "sc3_horizon_executor_trace.v1", **runtime_trace},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    runtime_trace["executor_trace_artifact"] = {
        "path": str(trace_artifact),
        "sha256": hashlib.sha256(trace_artifact.read_bytes()).hexdigest(),
    }
    signed_fields = (
        "trace_producer_id",
        "runtime_session_id",
        "runtime_executor_id",
        "runtime_executor_code_sha256",
        "controller_id",
        "controller_sha256",
        "world_model_checkpoint_sha256",
        "runtime_execution_proven",
        "world_model_prediction_proven",
        "receding_horizon_controller_proven",
        "proposed_actions",
        "world_model_predictions",
        "retained_actions",
        "executed_actions",
        "discarded_predictions",
        "control_rate_hz",
        "chunk_start_timestamp_sec",
        "requery_timestamp_sec",
    )
    runtime_trace["executor_attestation"] = _signed_attestation(
        {field: runtime_trace.get(field) for field in signed_fields},
        tmp_path,
    )
    monkeypatch.setenv(
        SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        runtime_trace["executor_attestation"]["public_key_sha256"],
    )
    package = {"sc3_horizon_execution_trace": runtime_trace}

    trace = adapter.build_sc3_horizon_execution_trace(package)

    assert trace["status"] == "validated"
    assert len(trace["proposed_actions"]) == 25
    assert len(trace["world_model_predictions"]) == 24
    assert len(trace["retained_actions"]) == 16
    assert len(trace["executed_actions"]) == 16
    assert len(trace["discarded_predictions"]) == 8
    assert trace["requery_timestamp_sec"] == 2.8


def test_sc3_receding_horizon_executor_produces_runtime_25_24_16_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_key = Ed25519PrivateKey.generate()
    private_key_path = tmp_path / "executor-private.pem"
    private_key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv(
        SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        hashlib.sha256(public_key).hexdigest(),
    )
    prediction_calls: list[int] = []
    execution_calls: list[int] = []
    callback_evidence_dir = tmp_path / "callback-evidence"
    callback_evidence_dir.mkdir()

    def propose_actions() -> list[dict[str, Any]]:
        rows = []
        for index in range(25):
            vector = [round(index * 0.01 + axis * 0.001, 6) for axis in range(7)]
            rows.append(
                {
                    "action_id": f"action-{index:02d}",
                    "action_vector_7d": vector,
                    "action_sha256": hashlib.sha256(
                        json.dumps(vector, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
            )
        return rows

    def predict(action: Mapping[str, Any], index: int) -> dict[str, Any]:
        prediction_calls.append(index)
        prediction_id = f"prediction-{index:02d}"
        runtime_result_id = f"wam-output-{index:02d}"
        evidence = callback_evidence_dir / f"prediction-{index:02d}.json"
        evidence.write_text(
            json.dumps(
                {
                    "schema_version": "sc3_world_model_prediction_evidence.v1",
                    "status": "completed",
                    "runtime_session_id": "runtime-session",
                    "runtime_result_id": runtime_result_id,
                    "prediction_id": prediction_id,
                    "action_id": action["action_id"],
                    "action_sha256": action["action_sha256"],
                    "world_model_checkpoint_sha256": "d" * 64,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "schema_version": "sc3_world_model_prediction_result.v1",
            "status": "completed",
            "prediction_id": prediction_id,
            "runtime_result_id": runtime_result_id,
            "action_id": action["action_id"],
            "action_sha256": action["action_sha256"],
            "world_model_checkpoint_sha256": "d" * 64,
            "evidence_artifact": {
                "path": str(evidence),
                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            },
        }

    def execute(action: Mapping[str, Any], index: int, timestamp: float) -> dict[str, Any]:
        execution_calls.append(index)
        runtime_result_id = f"controller-{index:02d}"
        evidence = callback_evidence_dir / f"controller-{index:02d}.json"
        evidence.write_text(
            json.dumps(
                {
                    "schema_version": "sc3_controller_execution_evidence.v1",
                    "status": "completed",
                    "runtime_session_id": "runtime-session",
                    "runtime_result_id": runtime_result_id,
                    "action_id": action["action_id"],
                    "action_sha256": action["action_sha256"],
                    "controller_id": "controller",
                    "controller_sha256": "c" * 64,
                    "execution_timestamp_sec": timestamp,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "schema_version": "sc3_controller_execution_result.v1",
            "status": "completed",
            "runtime_result_id": runtime_result_id,
            "action_id": action["action_id"],
            "action_sha256": action["action_sha256"],
            "controller_id": "controller",
            "controller_sha256": "c" * 64,
            "execution_timestamp_sec": timestamp,
            "evidence_artifact": {
                "path": str(evidence),
                "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            },
        }

    trace = adapter.execute_sc3_receding_horizon_chunk(
        propose_actions=propose_actions,
        world_model_predict=predict,
        controller_execute=execute,
        output_dir=tmp_path / "runtime-trace",
        runtime_session_id="runtime-session",
        runtime_executor_id="runtime-executor",
        runtime_executor_code_sha256="b" * 64,
        controller_id="controller",
        controller_sha256="c" * 64,
        world_model_checkpoint_sha256="d" * 64,
        control_rate_hz=20.0,
        chunk_start_timestamp_sec=4.0,
        signing_private_key_file=private_key_path,
    )

    assert trace["status"] == "validated"
    assert prediction_calls == list(range(24))
    assert execution_calls == list(range(16))
    assert len(trace["discarded_predictions"]) == 8
    assert trace["requery_timestamp_sec"] == 4.8

    completed_execution_call_count = len(execution_calls)
    with pytest.raises(ValueError, match="sc3_executor_prediction_result_invalid:0"):
        adapter.execute_sc3_receding_horizon_chunk(
            propose_actions=propose_actions,
            world_model_predict=lambda action, index: {},
            controller_execute=execute,
            output_dir=tmp_path / "runtime-trace-empty-prediction",
            runtime_session_id="runtime-session",
            runtime_executor_id="runtime-executor",
            runtime_executor_code_sha256="b" * 64,
            controller_id="controller",
            controller_sha256="c" * 64,
            world_model_checkpoint_sha256="d" * 64,
            control_rate_hz=20.0,
            chunk_start_timestamp_sec=4.0,
            signing_private_key_file=private_key_path,
        )
    assert len(execution_calls) == completed_execution_call_count

    failed_controller_calls: list[int] = []

    def failed_controller(action, index, timestamp):
        failed_controller_calls.append(index)
        return {"status": "failed"}

    with pytest.raises(ValueError, match="sc3_executor_controller_result_invalid:0"):
        adapter.execute_sc3_receding_horizon_chunk(
            propose_actions=propose_actions,
            world_model_predict=predict,
            controller_execute=failed_controller,
            output_dir=tmp_path / "runtime-trace-failed-controller",
            runtime_session_id="runtime-session",
            runtime_executor_id="runtime-executor",
            runtime_executor_code_sha256="b" * 64,
            controller_id="controller",
            controller_sha256="c" * 64,
            world_model_checkpoint_sha256="d" * 64,
            control_rate_hz=20.0,
            chunk_start_timestamp_sec=4.0,
            signing_private_key_file=private_key_path,
        )
    assert failed_controller_calls == [0]
    assert not (
        tmp_path / "runtime-trace-failed-controller" / "sc3_horizon_executor_trace.json"
    ).exists()


def test_sc3_horizon_trace_blocks_metadata_only_action_chunk() -> None:
    trace = adapter.build_sc3_horizon_execution_trace(
        {
            "action_records": [{"action_id": "a0"}] * 16,
            "control_rate_hz": 20.0,
            "chunk_start_timestamp_sec": 0.0,
        }
    )

    assert trace["status"] == "blocked"
    assert "horizon_runtime_execution_trace_missing" in trace["blockers"]
    assert "horizon_proposed_count_must_equal_25" in trace["blockers"]
