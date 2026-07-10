from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from PIL import Image
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from blueprint_pipeline import sc3_fidelity_contracts as fidelity_contracts
from blueprint_pipeline.sc3_fidelity_contracts import (
    SC3_CHECKPOINT_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    SC3_ANCHOR_OUTCOME_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    SC3_ANCHOR_PREDICTION_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    SC3_MULTIVIEW_CHECKER_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    SC3_OOD_AXES,
    SC3_OOD_EVIDENCE_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    validate_anchor_artifacts,
    validate_benchmark_cards,
    validate_checkpoint_attestation,
    validate_external_study,
    validate_horizon_execution_trace,
    validate_ood_registry,
    validate_synchronized_multiview,
)


IDENTITY = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
CHECKER_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x02" * 32)
ANCHOR_PREDICTION_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x0a" * 32)
ANCHOR_OUTCOME_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x0b" * 32)


@pytest.fixture(autouse=True)
def _trusted_multiview_checker(monkeypatch: pytest.MonkeyPatch) -> None:
    public_key = CHECKER_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv(
        SC3_MULTIVIEW_CHECKER_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        hashlib.sha256(public_key).hexdigest(),
    )
    for env_name, key in (
        (
            SC3_ANCHOR_PREDICTION_TRUSTED_PUBLIC_KEY_SHA256_ENV,
            ANCHOR_PREDICTION_PRIVATE_KEY,
        ),
        (
            SC3_ANCHOR_OUTCOME_TRUSTED_PUBLIC_KEY_SHA256_ENV,
            ANCHOR_OUTCOME_PRIVATE_KEY,
        ),
    ):
        authority_key = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        monkeypatch.setenv(env_name, hashlib.sha256(authority_key).hexdigest())


def _signed_attestation(
    payload: dict,
    tmp_path: Path,
    stem: str,
    *,
    private_key: Ed25519PrivateKey | None = None,
) -> dict:
    private_key = private_key or Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    signed_payload_sha256 = hashlib.sha256(message).hexdigest()
    report = tmp_path / f"{stem}-signature-verification.json"
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


def _write_correlated_frame(path: Path, *, group_index: int, camera_index: int) -> None:
    image = Image.new("RGB", (32, 24))
    pixels = image.load()
    for y in range(24):
        for x in range(32):
            structure = (x * 7 + y * 11 + (x * y) % 29 + group_index * 3) % 256
            watermark = camera_index * ((x * 3 + y * 5) % 23)
            value = (structure + watermark) % 256
            pixels[x, y] = (value, (value + camera_index * 9) % 256, value)
    image.save(path)


def _multiview(tmp_path: Path) -> dict:
    expected_camera_ids = [f"cam-{index}" for index in range(3)]
    joint_artifact = tmp_path / "joint-generation.json"
    frame_groups = []
    frame_group_input_sha256: list[str] = []
    for group_index in range(3):
        frames = []
        for camera_index in range(3):
            path = tmp_path / f"camera-{group_index}-{camera_index}.png"
            _write_correlated_frame(
                path,
                group_index=group_index,
                camera_index=camera_index,
            )
            frames.append(
                {
                    "camera_id": f"cam-{camera_index}",
                    "timestamp_sec": 1.0 + group_index * 0.05 + camera_index * 0.001,
                    "simultaneous_frame_index": 10 + group_index,
                    "image_path": str(path),
                    "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "intrinsics": {
                        "fx": 24,
                        "fy": 24,
                        "cx": 16,
                        "cy": 12,
                        "width": 32,
                        "height": 24,
                    },
                    "world_from_camera": [
                        [1.0, 0.0, 0.0, camera_index * 0.1],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                }
            )
        frame_group_id = f"g{group_index}"
        input_sha256 = hashlib.sha256(
            json.dumps(
                {"frame_group_id": frame_group_id, "frames": frames},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        frame_group_input_sha256.append(input_sha256)

        def check_artifact(check_type: str, result: dict) -> dict:
            path = tmp_path / f"{frame_group_id}-{check_type}.json"
            evidence_payload = {
                "schema_version": "sc3_multiview_check_evidence.v1",
                "check_type": check_type,
                "status": "passed",
                "checker_id": "test-multiview-checker",
                "checker_code_sha256": "a" * 64,
                "input_manifest_sha256": input_sha256,
                "result": result,
            }
            evidence_payload["checker_attestation"] = _signed_attestation(
                evidence_payload,
                tmp_path,
                f"{frame_group_id}-{check_type}",
                private_key=CHECKER_PRIVATE_KEY,
            )
            path.write_text(
                json.dumps(evidence_payload, sort_keys=True),
                encoding="utf-8",
            )
            return {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        shared = {
            "status": "passed",
            "checker_id": "test-multiview-checker",
            "checker_code_sha256": "a" * 64,
            "input_manifest_sha256": input_sha256,
        }
        frame_groups.append(
            {
                "frame_group_id": frame_group_id,
                "frames": frames,
                "correspondence_check": {
                    **shared,
                    "reprojection_error_px": 0.4,
                    "threshold_px": 1.0,
                    "matched_point_count": 8,
                    "evidence_artifact": check_artifact(
                        "correspondence_check",
                        {
                            "reprojection_error_px": 0.4,
                            "threshold_px": 1.0,
                            "matched_point_count": 8,
                        },
                    ),
                },
                "occlusion_reentry_check": {
                    **shared,
                    "visible_before_occlusion": True,
                    "occlusion_observed": True,
                    "reentry_correspondence_verified": True,
                    "evidence_artifact": check_artifact(
                        "occlusion_reentry_check",
                        {
                            "visible_before_occlusion": True,
                            "occlusion_observed": True,
                            "reentry_correspondence_verified": True,
                        },
                    ),
                },
                "camera_assignment_check": {
                    **shared,
                    "verified_camera_ids": expected_camera_ids,
                    "evidence_artifact": check_artifact(
                        "camera_assignment_check",
                        {"verified_camera_ids": expected_camera_ids},
                    ),
                },
            }
        )
    joint_artifact.write_text(
        json.dumps(
            {
                "schema_version": "sc3_joint_multiview_generation.v1",
                "joint_generation_proven": True,
                "expected_camera_ids": expected_camera_ids,
                "frame_group_input_sha256": frame_group_input_sha256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "expected_camera_ids": expected_camera_ids,
        "joint_generation_proven": True,
        "joint_generation_artifact": {
            "path": str(joint_artifact),
            "sha256": hashlib.sha256(joint_artifact.read_bytes()).hexdigest(),
        },
        "frame_groups": frame_groups,
    }


def _horizon(tmp_path: Path) -> dict:
    actions = []
    for index in range(25):
        vector = [round(index * 0.001 + dimension * 0.01, 6) for dimension in range(7)]
        actions.append(
            {
                "action_id": f"a-{index:02d}",
                "action_vector_7d": vector,
                "action_sha256": hashlib.sha256(
                    json.dumps(vector, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
    predictions = []
    for index, action in enumerate(actions[:24]):
        prediction_id = f"prediction-{index:02d}"
        runtime_result_id = f"prediction-runtime-{index:02d}"
        evidence = tmp_path / f"horizon-prediction-{index:02d}.json"
        evidence.write_text(
            json.dumps(
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
                sort_keys=True,
            ),
            encoding="utf-8",
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
        timestamp = 10.0 + index / 20.0
        evidence = tmp_path / f"horizon-controller-{index:02d}.json"
        evidence.write_text(
            json.dumps(
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
                sort_keys=True,
            ),
            encoding="utf-8",
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
    trace = {
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
        "chunk_start_timestamp_sec": 10.0,
        "requery_timestamp_sec": 10.8,
    }
    artifact = tmp_path / "horizon-executor-trace.json"
    artifact.write_text(
        json.dumps(
            {"schema_version": "sc3_horizon_executor_trace.v1", **trace},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    trace["executor_trace_artifact"] = {
        "path": str(artifact),
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
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
    trace["executor_attestation"] = _signed_attestation(
        {field: trace.get(field) for field in signed_fields},
        tmp_path,
        "horizon",
    )
    return trace


def _checkpoint(tmp_path: Path) -> dict:
    artifacts = {}
    for name in (
        "checkpoint",
        "training_dataset",
        "training_split",
        "training_objective",
        "trainer_code",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode())
        artifacts[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    probes = []
    for mode in ("forward_dynamics", "inverse_dynamics", "cross_view"):
        input_path = tmp_path / f"{mode}-input.json"
        output_path = tmp_path / f"{mode}-output.json"
        probe_id = f"golden-{mode}"
        input_path.write_text(
            json.dumps(
                {
                    "schema_version": "sc3_golden_probe_input.v1",
                    "mode": mode,
                    "probe_id": probe_id,
                    "input_values": [0.0, 0.25, 0.5],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
        output_key = {
            "forward_dynamics": "predicted_next_state",
            "inverse_dynamics": "predicted_action_7d",
            "cross_view": "predicted_cross_view_embedding",
        }[mode]
        output_values = [0.1] * (7 if mode == "inverse_dynamics" else 3)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": "sc3_golden_probe_output.v1",
                    "mode": mode,
                    "probe_id": probe_id,
                    "input_sha256": input_sha256,
                    "status": "completed",
                    output_key: output_values,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        probes.append(
            {
                "mode": mode,
                "probe_id": probe_id,
                "status": "passed",
                "input_sha256": input_sha256,
                "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "input_artifact": {
                    "path": str(input_path),
                    "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                },
                "output_artifact": {
                    "path": str(output_path),
                    "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                },
            }
        )
    attestation = {
        "status": "attested",
        "base_checkpoint_only": False,
        "checkpoint_sha256": artifacts["checkpoint"]["sha256"],
        "training_dataset_sha256": artifacts["training_dataset"]["sha256"],
        "training_split_sha256": artifacts["training_split"]["sha256"],
        "training_objective_sha256": artifacts["training_objective"]["sha256"],
        "trainer_code_sha256": artifacts["trainer_code"]["sha256"],
        "checkpoint_artifact": artifacts["checkpoint"],
        "training_dataset_manifest_artifact": artifacts["training_dataset"],
        "training_split_manifest_artifact": artifacts["training_split"],
        "training_objective_artifact": artifacts["training_objective"],
        "trainer_code_artifact": artifacts["trainer_code"],
        "trained_modes": ["forward_dynamics", "inverse_dynamics", "cross_view"],
        "golden_functional_probes": probes,
    }
    attestation["attestation_signature"] = _signed_attestation(
        attestation,
        tmp_path,
        "checkpoint",
    )
    return attestation


def test_multiview_rejects_duplicate_unsynchronized_and_swapped_cameras(
    tmp_path: Path,
) -> None:
    assert validate_synchronized_multiview(_multiview(tmp_path))["status"] == "validated"
    invalid = _multiview(tmp_path)
    frames = invalid["frame_groups"][0]["frames"]
    frames[1]["camera_id"] = frames[0]["camera_id"]
    frames[1]["image_sha256"] = frames[0]["image_sha256"]
    frames[2]["timestamp_sec"] = 2.0
    invalid["frame_groups"][0]["camera_assignment_check"] = {"status": "failed"}

    validation = validate_synchronized_multiview(invalid)

    assert validation["status"] == "blocked"
    assert any("camera_ids_missing_or_duplicate" in item for item in validation["blockers"])
    assert any("duplicate_camera_content" in item for item in validation["blockers"])
    assert any("timestamp_skew_exceeded" in item for item in validation["blockers"])
    assert any("camera_assignment_check_failed" in item for item in validation["blockers"])

    unrelated = _multiview(tmp_path)
    unrelated_frame = unrelated["frame_groups"][0]["frames"][2]
    unrelated_path = Path(unrelated_frame["image_path"])
    random_image = Image.new("RGB", (32, 24))
    random_pixels = random_image.load()
    for y in range(24):
        for x in range(32):
            random_pixels[x, y] = (
                (x * 97 + y * 13) % 256,
                (x * 31 + y * 71) % 256,
                (x * 53 + y * 43) % 256,
            )
    random_image.save(unrelated_path)
    unrelated_frame["image_sha256"] = hashlib.sha256(unrelated_path.read_bytes()).hexdigest()
    unrelated_validation = validate_synchronized_multiview(unrelated)
    assert any(
        "cross_view_visual_structure_inconsistent" in item
        for item in unrelated_validation["blockers"]
    )


def test_horizon_enforces_25_24_16_and_exact_requery_time(
    tmp_path: Path,
    monkeypatch,
) -> None:
    valid = _horizon(tmp_path)
    monkeypatch.setenv(
        SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        valid["executor_attestation"]["public_key_sha256"],
    )
    assert validate_horizon_execution_trace(valid)["status"] == "validated"
    invalid = _horizon(tmp_path)
    monkeypatch.setenv(
        SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        invalid["executor_attestation"]["public_key_sha256"],
    )
    invalid["executed_actions"] = invalid["executed_actions"][:-1]
    invalid["requery_timestamp_sec"] = 10.75

    validation = validate_horizon_execution_trace(invalid)

    assert validation["status"] == "blocked"
    assert "horizon_executed_count_must_equal_16" in validation["blockers"]
    assert "horizon_requery_timestamp_mismatch" in validation["blockers"]
    assert (
        "horizon_executor_attestation_cryptographic_verification_failed" in validation["blockers"]
    )


def test_horizon_rejects_metadata_sliced_as_runtime_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    invalid = _horizon(tmp_path)
    monkeypatch.setenv(
        SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        invalid["executor_attestation"]["public_key_sha256"],
    )
    invalid["runtime_execution_proven"] = False
    invalid["world_model_predictions"] = invalid["proposed_actions"][:24]
    invalid["executed_actions"] = invalid["proposed_actions"][:16]

    validation = validate_horizon_execution_trace(invalid)

    assert validation["status"] == "blocked"
    assert "horizon_runtime_execution_not_proven" in validation["blockers"]
    assert any("horizon_prediction_not_completed" in row for row in validation["blockers"])
    assert any("horizon_execution_status_invalid" in row for row in validation["blockers"])


def test_base_checkpoint_or_missing_training_mode_cannot_impersonate_sc3(
    tmp_path: Path,
    monkeypatch,
) -> None:
    valid = _checkpoint(tmp_path)
    monkeypatch.setenv(
        SC3_CHECKPOINT_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        valid["attestation_signature"]["public_key_sha256"],
    )
    assert validate_checkpoint_attestation(valid)["status"] == "validated"
    monkeypatch.setenv(
        SC3_CHECKPOINT_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        "f" * 64,
    )
    untrusted = validate_checkpoint_attestation(valid)
    assert untrusted["status"] == "blocked"
    assert "sc3_checkpoint_attestation_signature_public_key_not_authorized" in untrusted["blockers"]
    invalid = _checkpoint(tmp_path)
    monkeypatch.setenv(
        SC3_CHECKPOINT_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        invalid["attestation_signature"]["public_key_sha256"],
    )
    invalid["base_checkpoint_only"] = True
    invalid["trained_modes"] = ["forward_dynamics"]

    validation = validate_checkpoint_attestation(invalid)

    assert validation["status"] == "blocked"
    assert "base_checkpoint_without_sc3_finetuning" in validation["blockers"]
    assert "sc3_checkpoint_training_modes_incomplete" in validation["blockers"]
    assert (
        "sc3_checkpoint_attestation_signature_cryptographic_verification_failed"
        in validation["blockers"]
    )


def _write_json_artifact(path: Path, payload: dict) -> dict[str, str]:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _refresh_ood_registry_sha256(registry: dict) -> None:
    registry.pop("registry_sha256", None)
    registry["registry_sha256"] = hashlib.sha256(
        json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _ood_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    duplicate_checkpoint: bool = False,
    conflicting_family_axis: str | None = None,
    heterogeneous_axis: str | None = None,
) -> tuple[dict, str]:
    private_key = Ed25519PrivateKey.from_private_bytes(b"\x03" * 32)
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    monkeypatch.setenv(
        SC3_OOD_EVIDENCE_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        public_key_sha256,
    )
    decision_thresholds = {
        "min_pearson_success_rate_correlation": 0.8,
        "max_mean_maximum_rank_violation": 0.2,
        "max_mean_absolute_success_rate_error": 0.15,
        "max_abstention_rate": 0.3,
        "registered_uncertainty_method": (fidelity_contracts.SC3_OOD_UNCERTAINTY_METHOD),
    }
    decision_thresholds_sha256 = fidelity_contracts._canonical_sha256(decision_thresholds)
    policy_specs = [
        {
            "policy_id": f"policy-{index:02d}",
            "policy_checkpoint_sha256": hashlib.sha256(
                f"policy-checkpoint-{index}".encode("utf-8")
            ).hexdigest(),
            "policy_family_id": f"policy-family-{index:02d}",
        }
        for index in range(3)
    ]
    if duplicate_checkpoint:
        policy_specs[2]["policy_checkpoint_sha256"] = policy_specs[1]["policy_checkpoint_sha256"]
    axis_results = []
    for axis in sorted(SC3_OOD_AXES):
        train_group_ids = [f"{axis}-train-group"]
        train_source_ids = [f"{axis}-train-source"]
        heldout_group_count = 3 if axis == heterogeneous_axis else 1
        heldout_group_ids = [
            f"{axis}-heldout-group-{index:02d}" for index in range(heldout_group_count)
        ]
        heldout_source_ids = [
            f"{axis}-heldout-source-{index:02d}" for index in range(heldout_group_count)
        ]
        source_ref = _write_json_artifact(
            tmp_path / f"ood-{axis}-source-manifest.json",
            {
                "schema_version": "sc3_ood_axis_source_manifest.v2",
                "axis": axis,
                "train_source_ids": train_source_ids,
                "heldout_source_ids": heldout_source_ids,
                "registered_policies": policy_specs,
                "registered_uncertainty_method": (fidelity_contracts.SC3_OOD_UNCERTAINTY_METHOD),
            },
        )
        train_ref = _write_json_artifact(
            tmp_path / f"ood-{axis}-train-split.json",
            {
                "schema_version": "sc3_ood_axis_split.v1",
                "axis": axis,
                "split": "train",
                "group_ids": train_group_ids,
                "source_ids": train_source_ids,
                "source_manifest_sha256": source_ref["sha256"],
            },
        )
        heldout_ref = _write_json_artifact(
            tmp_path / f"ood-{axis}-heldout-split.json",
            {
                "schema_version": "sc3_ood_axis_split.v1",
                "axis": axis,
                "split": "heldout",
                "group_ids": heldout_group_ids,
                "source_ids": heldout_source_ids,
                "source_manifest_sha256": source_ref["sha256"],
            },
        )
        raw_rows = []
        success_limits = (5, 10, 15)
        reverse_success_limits = tuple(reversed(success_limits))
        for group_index, (heldout_group_id, heldout_source_id) in enumerate(
            zip(heldout_group_ids, heldout_source_ids)
        ):
            condition_id = f"{axis}-condition-{group_index:02d}"
            for policy_index, (policy_spec, predicted_limit) in enumerate(
                zip(policy_specs, success_limits)
            ):
                actual_limit = (
                    reverse_success_limits[policy_index]
                    if axis == heterogeneous_axis and group_index == 2
                    else predicted_limit
                )
                for seed in range(20):
                    policy_family_id = policy_spec["policy_family_id"]
                    if (
                        axis == conflicting_family_axis
                        and group_index == 0
                        and policy_index == 0
                        and seed == 0
                    ):
                        policy_family_id = "conflicting-policy-family"
                    evidence_payload = {
                        "schema_version": "sc3_ood_replicate_evidence.v2",
                        "axis": axis,
                        "policy_id": policy_spec["policy_id"],
                        "policy_checkpoint_sha256": policy_spec["policy_checkpoint_sha256"],
                        "policy_family_id": policy_family_id,
                        "condition_id": condition_id,
                        "heldout_group_id": heldout_group_id,
                        "source_id": heldout_source_id,
                        "replicate_id": (
                            f"{axis}-{group_index:02d}-{policy_spec['policy_id']}-{seed:02d}"
                        ),
                        "replicate_seed": seed,
                        "predicted_success": seed < predicted_limit,
                        "actual_success": seed < actual_limit,
                        "abstained": False,
                        "train_split_sha256": train_ref["sha256"],
                        "heldout_split_sha256": heldout_ref["sha256"],
                        "source_manifest_sha256": source_ref["sha256"],
                        "decision_thresholds_sha256": decision_thresholds_sha256,
                    }
                    evidence_stem = (
                        f"ood-{axis}-{group_index:02d}-{policy_spec['policy_id']}-{seed:02d}"
                    )
                    evidence_ref = _write_json_artifact(
                        tmp_path / f"{evidence_stem}-evidence.json",
                        evidence_payload,
                    )
                    raw_rows.append(
                        {
                            **{
                                field: evidence_payload[field]
                                for field in (fidelity_contracts.OOD_REPLICATE_BINDING_FIELDS)
                            },
                            "evidence_artifact": evidence_ref,
                            "evidence_attestation": _signed_attestation(
                                evidence_payload,
                                tmp_path,
                                evidence_stem,
                                private_key=private_key,
                            ),
                        }
                    )
        raw_ref = _write_json_artifact(
            tmp_path / f"ood-{axis}-raw-replicates.json",
            {
                "schema_version": "sc3_ood_axis_raw_replicates.v3",
                "axis": axis,
                "train_split_sha256": train_ref["sha256"],
                "heldout_split_sha256": heldout_ref["sha256"],
                "source_manifest_sha256": source_ref["sha256"],
                "decision_thresholds_sha256": decision_thresholds_sha256,
                "heldout_group_ids": heldout_group_ids,
                "heldout_source_ids": heldout_source_ids,
                "registered_policies": policy_specs,
                "registered_uncertainty_method": (fidelity_contracts.SC3_OOD_UNCERTAINTY_METHOD),
                "rows": raw_rows,
            },
        )
        result = {
            "axis": axis,
            "train_group_ids": train_group_ids,
            "heldout_group_ids": heldout_group_ids,
            "train_source_ids": train_source_ids,
            "heldout_source_ids": heldout_source_ids,
            "train_split_sha256": train_ref["sha256"],
            "heldout_split_sha256": heldout_ref["sha256"],
            "source_manifest_sha256": source_ref["sha256"],
            "decision_thresholds_sha256": decision_thresholds_sha256,
            "registered_policies": policy_specs,
            "registered_uncertainty_method": (fidelity_contracts.SC3_OOD_UNCERTAINTY_METHOD),
            "train_split_artifact": train_ref,
            "heldout_split_artifact": heldout_ref,
            "source_manifest_artifact": source_ref,
            "raw_rows_artifact": raw_ref,
        }
        recomputed, blockers = fidelity_contracts._recomputed_ood_axis_metrics(
            axis=axis,
            result=result,
        )
        if not duplicate_checkpoint and axis != conflicting_family_axis:
            assert blockers == []
        result.update(recomputed)
        pearson_ci = recomputed.get("pearson_95_ci") or []
        mmrv_ci = recomputed.get("mmrv_95_ci") or []
        error_ci = recomputed.get("error_95_ci") or []
        abstention_ci = recomputed.get("abstention_95_ci") or []
        result["thresholds_passed"] = bool(
            recomputed.get("pearson_success_rate_correlation")
            >= decision_thresholds["min_pearson_success_rate_correlation"]
            and len(pearson_ci) == 2
            and pearson_ci[0] >= decision_thresholds["min_pearson_success_rate_correlation"]
            and recomputed.get("mean_maximum_rank_violation")
            <= decision_thresholds["max_mean_maximum_rank_violation"]
            and len(mmrv_ci) == 2
            and mmrv_ci[1] <= decision_thresholds["max_mean_maximum_rank_violation"]
            and recomputed.get("mean_absolute_success_rate_error")
            <= decision_thresholds["max_mean_absolute_success_rate_error"]
            and len(error_ci) == 2
            and error_ci[1] <= decision_thresholds["max_mean_absolute_success_rate_error"]
            and recomputed.get("abstention_rate") <= decision_thresholds["max_abstention_rate"]
            and len(abstention_ci) == 2
            and abstention_ci[1] <= decision_thresholds["max_abstention_rate"]
        )
        axis_results.append(result)
    registry = {
        "frozen_axes": sorted(SC3_OOD_AXES),
        "decision_thresholds": decision_thresholds,
        "leave_one_group_results": axis_results,
    }
    _refresh_ood_registry_sha256(registry)
    return registry, public_key_sha256


def test_frozen_ood_requires_signed_matched_leave_one_group_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, _ = _ood_registry(tmp_path, monkeypatch)

    validation = validate_ood_registry(registry)

    assert validation["status"] == "validated"
    assert validation["pooled_ood_headline_allowed"] is True
    assert validation["minimum_replicates_per_policy_condition"] == 20
    assert validation["minimum_policy_count"] == 3
    for result in validation["recomputed_results_by_axis"].values():
        assert result["sample_count"] == 60
        assert result["minimum_matched_seed_count"] == 20
        assert result["distinct_policy_checkpoint_count"] == 3
        assert result["pearson_success_rate_correlation"] == 1.0
        assert result["bootstrap_method"] == (fidelity_contracts.SC3_OOD_UNCERTAINTY_METHOD)
        assert result["bootstrap_cluster_levels"] == [
            "heldout_group_id",
            "condition_id",
            "replicate_seed",
        ]
        assert result["abstention_interval_method"] == (
            fidelity_contracts.SC3_OOD_UNCERTAINTY_METHOD
        )


def test_frozen_ood_rejects_untrusted_overlap_small_n_and_supplied_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, trusted_key_sha256 = _ood_registry(tmp_path, monkeypatch)

    untrusted = json.loads(json.dumps(registry))
    monkeypatch.setenv(
        SC3_OOD_EVIDENCE_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        "f" * 64,
    )
    untrusted_validation = validate_ood_registry(untrusted)
    assert any(
        "ood_axis_replicate_evidence_attestation" in blocker
        and "public_key_not_authorized" in blocker
        for blocker in untrusted_validation["blockers"]
    )
    monkeypatch.setenv(
        SC3_OOD_EVIDENCE_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        trusted_key_sha256,
    )

    overlap = json.loads(json.dumps(registry))
    first_overlap = overlap["leave_one_group_results"][0]
    first_overlap["train_group_ids"].append(first_overlap["heldout_group_ids"][0])
    first_overlap["train_source_ids"].append(first_overlap["heldout_source_ids"][0])
    _refresh_ood_registry_sha256(overlap)
    overlap_validation = validate_ood_registry(overlap)
    assert any(
        "train_heldout_group_overlap" in blocker for blocker in overlap_validation["blockers"]
    )
    assert any(
        "train_heldout_source_overlap" in blocker for blocker in overlap_validation["blockers"]
    )

    mismatched = json.loads(json.dumps(registry))
    mismatched["leave_one_group_results"][0]["pearson_95_ci"] = [0.0, 0.1]
    mismatched["leave_one_group_results"][0]["pearson_success_rate_correlation"] = 0.0
    _refresh_ood_registry_sha256(mismatched)
    mismatch_validation = validate_ood_registry(mismatched)
    assert any(
        "declared_interval_does_not_match_raw_rows" in blocker
        for blocker in mismatch_validation["blockers"]
    )
    assert any(
        "declared_metric_does_not_match_raw_rows" in blocker
        for blocker in mismatch_validation["blockers"]
    )

    changed_thresholds = json.loads(json.dumps(registry))
    changed_thresholds["decision_thresholds"]["min_pearson_success_rate_correlation"] = 0.0
    _refresh_ood_registry_sha256(changed_thresholds)
    changed_threshold_validation = validate_ood_registry(changed_thresholds)
    assert any(
        "decision_thresholds_digest_mismatch" in blocker
        for blocker in changed_threshold_validation["blockers"]
    )

    first = registry["leave_one_group_results"][0]
    raw_path = Path(first["raw_rows_artifact"]["path"])
    raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
    original_rows = list(raw_payload["rows"])
    raw_payload["rows"] = raw_payload["rows"][:-1]
    first["raw_rows_artifact"] = _write_json_artifact(raw_path, raw_payload)
    _refresh_ood_registry_sha256(registry)
    small_n_validation = validate_ood_registry(registry)
    assert any(
        "policy_condition_replicates_lt_20" in blocker for blocker in small_n_validation["blockers"]
    )

    raw_payload["rows"] = [row for row in original_rows if row["policy_id"] != "policy-02"]
    first["raw_rows_artifact"] = _write_json_artifact(raw_path, raw_payload)
    _refresh_ood_registry_sha256(registry)
    policy_count_validation = validate_ood_registry(registry)
    assert any(
        "ood_axis_policy_count_lt_3" in blocker for blocker in policy_count_validation["blockers"]
    )


def test_frozen_ood_rejects_checkpoint_alias_and_policy_family_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate_root = tmp_path / "duplicate-checkpoint"
    duplicate_root.mkdir()
    duplicate_registry, _ = _ood_registry(
        duplicate_root,
        monkeypatch,
        duplicate_checkpoint=True,
    )

    duplicate_validation = validate_ood_registry(duplicate_registry)

    assert duplicate_validation["status"] == "blocked"
    assert any(
        "distinct_policy_checkpoint_count_lt_3" in blocker
        for blocker in duplicate_validation["blockers"]
    )
    assert any(
        "policy_checkpoint_alias_detected" in blocker
        for blocker in duplicate_validation["blockers"]
    )

    conflict_axis = sorted(SC3_OOD_AXES)[0]
    conflict_root = tmp_path / "family-drift"
    conflict_root.mkdir()
    conflict_registry, _ = _ood_registry(
        conflict_root,
        monkeypatch,
        conflicting_family_axis=conflict_axis,
    )

    conflict_validation = validate_ood_registry(conflict_registry)

    assert conflict_validation["status"] == "blocked"
    assert any(
        f"ood_axis_policy_identity_binding_invalid:{conflict_axis}" in blocker
        for blocker in conflict_validation["blockers"]
    )


def test_frozen_ood_uses_hierarchical_ci_bounds_for_axis_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    heterogeneous_axis = sorted(SC3_OOD_AXES)[0]
    registry, _ = _ood_registry(
        tmp_path,
        monkeypatch,
        heterogeneous_axis=heterogeneous_axis,
    )

    validation = validate_ood_registry(registry)
    axis_result = validation["recomputed_results_by_axis"][heterogeneous_axis]

    assert axis_result["pearson_success_rate_correlation"] >= 0.8
    assert axis_result["pearson_95_ci"][0] < 0.8
    assert axis_result["bootstrap_method"] == (fidelity_contracts.SC3_OOD_UNCERTAINTY_METHOD)
    assert validation["status"] == "blocked"
    assert validation["pooled_ood_headline_allowed"] is False
    assert f"ood_axis_threshold_failed:{heterogeneous_axis}" in validation["blockers"]


def _anchor_row(
    tmp_path: Path,
    *,
    policy_id: str = "p",
    checkpoint_id: str = "c",
    replicate_seed: int = 1,
    registered_split: str = "test",
    task_id: str = "pick-object",
    condition_source_id: str = "source-1",
) -> dict:
    stem = f"{policy_id}-{replicate_seed}"
    split_manifest = tmp_path / "anchor-split-manifest.json"
    split_manifest.write_text(
        json.dumps(
            {
                "schema_version": "sc3_anchor_split_manifest.v1",
                "status": "frozen",
                "split_manifest_id": "anchor-test-split-v1",
                "registered_split": registered_split,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    split_manifest_sha256 = hashlib.sha256(split_manifest.read_bytes()).hexdigest()
    checkpoint = tmp_path / f"checkpoint-{policy_id}.bin"
    checkpoint.write_bytes(f"checkpoint:{checkpoint_id}".encode())
    policy_checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    prediction = tmp_path / f"prediction-{stem}.json"
    outcome = tmp_path / f"outcome-{stem}.json"
    join = {
        "policy_id": policy_id,
        "checkpoint_id": checkpoint_id,
        "policy_checkpoint_sha256": policy_checkpoint_sha256,
        "criterion_id": "lift",
        "registered_split": registered_split,
        "split_manifest_id": "anchor-test-split-v1",
        "split_manifest_sha256": split_manifest_sha256,
        "task_family": "pick",
        "task_id": task_id,
        "scenario_eval_run_id": f"run-{stem}",
        "scenario_variation_instance_id": "variation-1",
        "condition_id": "condition-1",
        "condition_source_id": condition_source_id,
        "replicate_id": f"replicate-{stem}",
        "replicate_seed": replicate_seed,
    }
    prediction_payload = {
        "schema_version": "sc3_anchor_prediction.v1",
        **join,
        "predicted_success": True,
    }
    prediction_payload["authority_attestation"] = _signed_attestation(
        prediction_payload,
        tmp_path,
        f"anchor-prediction-{stem}",
        private_key=ANCHOR_PREDICTION_PRIVATE_KEY,
    )
    prediction.write_text(
        json.dumps(prediction_payload, sort_keys=True),
        encoding="utf-8",
    )
    outcome_payload = {
        "schema_version": "sc3_anchor_outcome.v1",
        **join,
        "actual_success": False,
    }
    outcome_payload["authority_attestation"] = _signed_attestation(
        outcome_payload,
        tmp_path,
        f"anchor-outcome-{stem}",
        private_key=ANCHOR_OUTCOME_PRIVATE_KEY,
    )
    outcome.write_text(
        json.dumps(outcome_payload, sort_keys=True),
        encoding="utf-8",
    )
    return {
        **join,
        "predicted_success": True,
        "actual_success": False,
        "split_manifest_artifact": {
            "path": str(split_manifest),
            "sha256": split_manifest_sha256,
        },
        "policy_checkpoint_artifact": {
            "path": str(checkpoint),
            "sha256": policy_checkpoint_sha256,
        },
        "prediction_artifact": {
            "path": str(prediction),
            "sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
        },
        "outcome_artifact": {
            "path": str(outcome),
            "sha256": hashlib.sha256(outcome.read_bytes()).hexdigest(),
        },
    }


def test_anchor_rows_require_actual_hash_verified_files(tmp_path: Path) -> None:
    row = _anchor_row(tmp_path)
    assert validate_anchor_artifacts([row])["status"] == "validated"
    reused = {
        **row,
        "policy_id": "p-other",
        "scenario_eval_run_id": "run-other",
    }
    reuse_validation = validate_anchor_artifacts([row, reused])
    assert reuse_validation["status"] == "blocked"
    assert any("artifact_path_reused" in item for item in reuse_validation["blockers"])
    row["prediction_artifact"]["sha256"] = "0" * 64
    assert validate_anchor_artifacts([row])["status"] == "blocked"


def test_anchor_rows_reject_wrong_or_mixed_splits_and_untrusted_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_root = tmp_path / "train"
    train_root.mkdir()
    train_row = _anchor_row(train_root, registered_split="train")
    train_validation = validate_anchor_artifacts([train_row])
    assert train_validation["status"] == "blocked"
    assert "accepted_anchor_registered_split_not_evaluation:0" in train_validation["blockers"]

    test_root = tmp_path / "test"
    locked_root = tmp_path / "locked"
    test_root.mkdir()
    locked_root.mkdir()
    test_row = _anchor_row(test_root, policy_id="p-test", checkpoint_id="c-test")
    locked_row = _anchor_row(
        locked_root,
        policy_id="p-locked",
        checkpoint_id="c-locked",
        registered_split="locked_test",
    )
    mixed_validation = validate_anchor_artifacts([test_row, locked_row])
    assert mixed_validation["status"] == "blocked"
    assert "accepted_anchor_registered_split_mixed" in mixed_validation["blockers"]

    untrusted_root = tmp_path / "untrusted"
    untrusted_root.mkdir()
    untrusted_row = _anchor_row(untrusted_root)
    monkeypatch.setenv(
        SC3_ANCHOR_PREDICTION_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        "0" * 64,
    )
    untrusted_validation = validate_anchor_artifacts([untrusted_row])
    assert untrusted_validation["status"] == "blocked"
    assert any(
        "accepted_anchor_prediction_authority:0_public_key_not_authorized" in blocker
        for blocker in untrusted_validation["blockers"]
    )


def test_anchor_rows_require_distinct_prediction_and_outcome_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = _anchor_row(tmp_path)
    prediction_public_key = ANCHOR_PREDICTION_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    shared_fingerprint = hashlib.sha256(prediction_public_key).hexdigest()
    monkeypatch.setenv(
        SC3_ANCHOR_PREDICTION_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        shared_fingerprint,
    )
    monkeypatch.setenv(
        SC3_ANCHOR_OUTCOME_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        shared_fingerprint,
    )
    outcome_path = Path(row["outcome_artifact"]["path"])
    outcome_payload = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome_payload.pop("authority_attestation")
    outcome_payload["authority_attestation"] = _signed_attestation(
        outcome_payload,
        tmp_path,
        "anchor-outcome-shared-authority",
        private_key=ANCHOR_PREDICTION_PRIVATE_KEY,
    )
    outcome_path.write_text(
        json.dumps(outcome_payload, sort_keys=True),
        encoding="utf-8",
    )
    row["outcome_artifact"]["sha256"] = hashlib.sha256(outcome_path.read_bytes()).hexdigest()

    validation = validate_anchor_artifacts([row])

    assert validation["status"] == "blocked"
    assert "accepted_anchor_prediction_outcome_authorities_not_separated" in validation["blockers"]


def test_anchor_decision_grade_rejects_checkpoint_aliases_and_condition_drift(
    tmp_path: Path,
) -> None:
    alias_rows = []
    for index in range(3):
        root = tmp_path / f"alias-{index}"
        root.mkdir()
        alias_rows.append(
            _anchor_row(
                root,
                policy_id=f"policy-{index}",
                checkpoint_id="shared-checkpoint",
                replicate_seed=index,
            )
        )
    alias_validation = validate_anchor_artifacts(alias_rows)
    assert alias_validation["decision_grade_status"] != "decision_grade"
    assert (
        "accepted_anchor_checkpoint_reused_across_policies"
        in alias_validation["decision_grade_blockers"]
    )

    drift_rows = []
    for index, task_id in enumerate(("pick-object", "place-object", "pick-object")):
        root = tmp_path / f"drift-{index}"
        root.mkdir()
        drift_rows.append(
            _anchor_row(
                root,
                policy_id=f"drift-policy-{index}",
                checkpoint_id=f"drift-checkpoint-{index}",
                replicate_seed=index,
                task_id=task_id,
            )
        )
    drift_validation = validate_anchor_artifacts(drift_rows)
    assert drift_validation["decision_grade_status"] != "decision_grade"
    assert (
        "accepted_anchor_condition_descriptor_mismatch:condition-1"
        in drift_validation["decision_grade_blockers"]
    )


def test_benchmark_names_and_external_study_remain_separate() -> None:
    cards = {
        "sc3_eval": {
            "benchmark_family": "sc3_eval",
            "model_id": "sc3-ft",
            "protocol_id": "sc3-v3",
            "label_unit": "criterion",
            "sample_unit": "checkpoint_criterion",
            "metric_names": [
                "pearson_success_rate_correlation",
                "spearman_rank_correlation",
                "mean_maximum_rank_violation",
            ],
        },
        "oscar": {
            "benchmark_family": "oscar",
            "model_id": "oscar",
            "protocol_id": "oscar-v1",
            "label_unit": "episode",
            "sample_unit": "rollout",
            "metric_names": ["success_rate_difference_pp", "mae"],
        },
    }
    assert validate_benchmark_cards(cards)["status"] == "validated"
    cards["oscar"]["metric_names"] = ["sisr_delta"]
    assert validate_benchmark_cards(cards)["status"] == "blocked"

    cards["oscar"]["metric_names"] = [
        "success_rate_difference_pp",
        "pearson_success_rate_correlation",
    ]
    assert validate_benchmark_cards(cards)["status"] == "blocked"

    cards["oscar"]["metric_names"] = ["success_rate_difference_pp"]
    cards["oscar"]["model_id"] = cards["sc3_eval"]["model_id"]
    cards["oscar"]["protocol_id"] = cards["sc3_eval"]["protocol_id"]
    assert validate_benchmark_cards(cards)["status"] == "blocked"

    missing_study = validate_external_study({})
    assert missing_study["status"] == "external_proof_required"
    assert missing_study["external_manual_proof"] is True
    assert "external_sc3_study_requires_independent_manual_acceptance" in missing_study["blockers"]

    malformed_counts = validate_external_study(
        {
            "independent_policy_checkpoint_count": "seven",
            "accepted_anchor_count": "many",
        }
    )
    assert malformed_counts["status"] == "external_proof_required"
    assert "external_sc3_study_policy_checkpoint_count_lt_7" in malformed_counts["blockers"]
    assert "external_sc3_study_has_no_accepted_anchors" in malformed_counts["blockers"]
