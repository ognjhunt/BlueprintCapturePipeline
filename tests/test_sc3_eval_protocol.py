from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from PIL import Image
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from blueprint_pipeline.sc3_eval_protocol import (
    SC3_EVAL_PROTOCOL_SCHEMA_VERSION,
    build_sc3_eval_protocol_artifact,
)
from blueprint_pipeline.sc3_fidelity_contracts import (
    SC3_ANCHOR_OUTCOME_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    SC3_ANCHOR_PREDICTION_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    SC3_MULTIVIEW_CHECKER_TRUSTED_PUBLIC_KEY_SHA256_ENV,
)


CHECKER_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x03" * 32)
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


def _checker_attestation(payload: dict, tmp_path: Path, stem: str) -> dict:
    public_key = CHECKER_PRIVATE_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    payload_sha256 = hashlib.sha256(message).hexdigest()
    report = tmp_path / f"{stem}-checker-signature-report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "sc3_signature_verification_report.v1",
                "algorithm": "Ed25519",
                "verification_status": "verified",
                "public_key_sha256": public_key_sha256,
                "signed_payload_sha256": payload_sha256,
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
        "signature_base64": base64.b64encode(CHECKER_PRIVATE_KEY.sign(message)).decode("ascii"),
        "signed_payload_sha256": payload_sha256,
        "verification_report_artifact": {
            "path": str(report),
            "sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
        },
    }


def _anchor_attestation(
    payload: dict,
    tmp_path: Path,
    stem: str,
    private_key: Ed25519PrivateKey,
) -> dict:
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    message = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    payload_sha256 = hashlib.sha256(message).hexdigest()
    report = tmp_path / f"{stem}-authority-signature-report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "sc3_signature_verification_report.v1",
                "algorithm": "Ed25519",
                "verification_status": "verified",
                "public_key_sha256": public_key_sha256,
                "signed_payload_sha256": payload_sha256,
                "signer_key_id": stem,
                "verifier_id": "test-verifier",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return {
        "algorithm": "Ed25519",
        "signature_verified": True,
        "signer_key_id": stem,
        "verifier_id": "test-verifier",
        "public_key_base64": base64.b64encode(public_key).decode("ascii"),
        "public_key_sha256": public_key_sha256,
        "signature_base64": base64.b64encode(private_key.sign(message)).decode("ascii"),
        "signed_payload_sha256": payload_sha256,
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


def _robot_pov_manifest(tmp_path: Path, camera_count: int = 3) -> dict[str, object]:
    camera_ids = [f"cam_{index}" for index in range(camera_count)]
    joint_artifact = tmp_path / f"joint-{camera_count}.json"
    frame_groups = []
    frame_group_input_sha256: list[str] = []
    for group_index in range(3):
        frames = []
        for camera_index in range(camera_count):
            image_path = tmp_path / (f"camera-{camera_count}-{group_index}-{camera_index}.png")
            _write_correlated_frame(
                image_path,
                group_index=group_index,
                camera_index=camera_index,
            )
            frames.append(
                {
                    "camera_id": f"cam_{camera_index}",
                    "timestamp_sec": 1.0 + group_index * 0.05 + camera_index * 0.001,
                    "simultaneous_frame_index": 1 + group_index,
                    "image_path": str(image_path),
                    "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
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
        group_id = f"g{group_index}"
        input_sha256 = hashlib.sha256(
            json.dumps(
                {"frame_group_id": group_id, "frames": frames},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        frame_group_input_sha256.append(input_sha256)

        def evidence(check_type: str, result: dict) -> dict:
            path = tmp_path / f"{camera_count}-{group_id}-{check_type}.json"
            evidence_payload = {
                "schema_version": "sc3_multiview_check_evidence.v1",
                "check_type": check_type,
                "status": "passed",
                "checker_id": "test-checker",
                "checker_code_sha256": "a" * 64,
                "input_manifest_sha256": input_sha256,
                "result": result,
            }
            evidence_payload["checker_attestation"] = _checker_attestation(
                evidence_payload,
                tmp_path,
                f"{camera_count}-{group_id}-{check_type}",
            )
            path.write_text(
                json.dumps(evidence_payload, sort_keys=True),
                encoding="utf-8",
            )
            return {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        common = {
            "status": "passed",
            "checker_id": "test-checker",
            "checker_code_sha256": "a" * 64,
            "input_manifest_sha256": input_sha256,
        }
        frame_groups.append(
            {
                "frame_group_id": group_id,
                "frames": frames,
                "correspondence_check": {
                    **common,
                    "reprojection_error_px": 0.1,
                    "threshold_px": 1.0,
                    "matched_point_count": 8,
                    "evidence_artifact": evidence(
                        "correspondence_check",
                        {
                            "reprojection_error_px": 0.1,
                            "threshold_px": 1.0,
                            "matched_point_count": 8,
                        },
                    ),
                },
                "occlusion_reentry_check": {
                    **common,
                    "visible_before_occlusion": True,
                    "occlusion_observed": True,
                    "reentry_correspondence_verified": True,
                    "evidence_artifact": evidence(
                        "occlusion_reentry_check",
                        {
                            "visible_before_occlusion": True,
                            "occlusion_observed": True,
                            "reentry_correspondence_verified": True,
                        },
                    ),
                },
                "camera_assignment_check": {
                    **common,
                    "verified_camera_ids": camera_ids,
                    "evidence_artifact": evidence(
                        "camera_assignment_check",
                        {"verified_camera_ids": camera_ids},
                    ),
                },
            }
        )
    joint_artifact.write_text(
        json.dumps(
            {
                "schema_version": "sc3_joint_multiview_generation.v1",
                "joint_generation_proven": True,
                "expected_camera_ids": camera_ids,
                "frame_group_input_sha256": frame_group_input_sha256,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    camera_profile_registry = {
        "active_robot_profile_id": "customer_bot",
        "profiles": [
            {
                "robot_profile_id": "customer_bot",
                "cameras": [{"camera_id": f"cam_{index}"} for index in range(camera_count)],
            }
        ],
    }
    profile_artifact = tmp_path / f"camera-profile-{camera_count}.json"
    profile_artifact.write_text(
        json.dumps(
            {
                "schema_version": "sc3_robot_camera_profile_evidence.v1",
                "camera_profile_registry": camera_profile_registry,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source_frames = [frame for group in frame_groups for frame in group["frames"]][:2]
    observation_artifacts = []
    for observation_index, frame in enumerate(source_frames):
        observation_path = tmp_path / (f"observation-{camera_count}-{observation_index}.json")
        observation_path.write_text(
            json.dumps(
                {
                    "schema_version": "sc3_initial_observation_evidence.v1",
                    "observation_index": observation_index,
                    "camera_id": frame["camera_id"],
                    "image_artifact": {
                        "path": frame["image_path"],
                        "sha256": frame["image_sha256"],
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        observation_artifacts.append(
            {
                "path": str(observation_path),
                "sha256": hashlib.sha256(observation_path.read_bytes()).hexdigest(),
            }
        )
    return {
        "status": "completed",
        "observation_count": len(observation_artifacts),
        "observation_artifacts": observation_artifacts,
        "camera_profile_registry": camera_profile_registry,
        "robot_camera_profile_artifact": {
            "path": str(profile_artifact),
            "sha256": hashlib.sha256(profile_artifact.read_bytes()).hexdigest(),
        },
        "robot_camera_profile_launch_readiness": {
            "status": "smoke_only_owner_calibration_required"
        },
        "synchronized_multiview": {
            "expected_camera_ids": camera_ids,
            "joint_generation_proven": True,
            "joint_generation_artifact": {
                "path": str(joint_artifact),
                "sha256": hashlib.sha256(joint_artifact.read_bytes()).hexdigest(),
            },
            "frame_groups": frame_groups,
        },
    }


def test_sc3_protocol_defines_required_data_and_blocks_correlation_without_anchors(
    tmp_path: Path,
) -> None:
    artifact = build_sc3_eval_protocol_artifact(
        generated_at="now",
        job_request={"robot_profile": {"robot_profile_id": "customer_bot"}},
        policy_package_manifest={
            "selected_modalities": ["policy_api_endpoint"],
            "modalities": {
                "policy_api_endpoint": {
                    "selected": True,
                    "status": "launch_ready_review_required",
                    "interface_contract": {"observation_schema": {"schema_id": "x"}},
                }
            },
        },
        policy_execution_manifest={
            "status": "blocked",
            "policy_execution_trace_path": "policy_execution_trace.json",
            "modality_results": {
                "policy_api_endpoint": {
                    "status": "blocked_policy_execution_gate",
                    "execution_performed": False,
                    "robot_policy_execution_proven": False,
                }
            },
        },
        robot_pov_observation_manifest=_robot_pov_manifest(tmp_path),
    )

    assert artifact["schema_version"] == SC3_EVAL_PROTOCOL_SCHEMA_VERSION
    assert artifact["source_facts"]["paper_id"] == "arXiv:2606.18610v3"
    assert artifact["data_requirements"]["synchronized_multi_view_cameras"]["status"] == "ready"
    assert artifact["data_requirements"]["accepted_anchor_joins"]["status"] == (
        "correlation_not_measured"
    )
    assert {
        "policy_checkpoint_sha256",
        "split_manifest_id",
        "split_manifest_sha256",
        "condition_source_id",
    } <= set(artifact["data_requirements"]["accepted_anchor_joins"]["join_keys"])
    assert artifact["metrics"]["pearson_success_rate_correlation"]["status"] == (
        "correlation_not_measured"
    )
    assert (
        artifact["claim_boundary"]["ninety_percent_or_better_blueprint_accuracy_claim_allowed"]
        is False
    )
    assert (
        artifact["policy_adapter_pack_contracts"][0]["launch_reviewable_without_execution"] is True
    )


def test_sc3_protocol_requires_real_booleans_for_policy_execution_claims(
    tmp_path: Path,
) -> None:
    artifact = build_sc3_eval_protocol_artifact(
        generated_at="now",
        job_request={},
        policy_package_manifest={
            "selected_modalities": ["policy_api_endpoint", "docker_container"],
            "modalities": {
                "policy_api_endpoint": {
                    "selected": True,
                    "status": "launch_ready_review_required",
                },
                "docker_container": {
                    "selected": "true",
                    "status": "launch_ready_review_required",
                },
            },
        },
        policy_execution_manifest={
            "status": "completed",
            "robot_team_policy_execution_proven": "true",
            "modality_results": {
                "policy_api_endpoint": {
                    "status": "completed",
                    "execution_performed": "true",
                    "robot_policy_execution_proven": "true",
                },
                "docker_container": {
                    "status": "completed",
                    "execution_performed": True,
                    "robot_policy_execution_proven": True,
                },
            },
        },
        robot_pov_observation_manifest=_robot_pov_manifest(tmp_path),
    )

    assert [row["modality"] for row in artifact["policy_adapter_pack_contracts"]] == [
        "policy_api_endpoint"
    ]
    contract = artifact["policy_adapter_pack_contracts"][0]
    assert contract["execution_performed"] is False
    assert contract["robot_team_policy_execution_proven"] is False
    assert contract["launch_reviewable_without_execution"] is True
    assert (
        artifact["data_requirements"]["policy_requery_trace"]["robot_team_policy_execution_proven"]
        is False
    )


def test_sc3_protocol_fails_closed_on_missing_multiview_and_preserves_ranking_status(
    tmp_path: Path,
) -> None:
    artifact = build_sc3_eval_protocol_artifact(
        generated_at="now",
        job_request={},
        policy_package_manifest={
            "selected_modalities": ["recorded_action_trace"],
            "modalities": {
                "recorded_action_trace": {
                    "selected": True,
                    "status": "launch_ready_review_required",
                }
            },
        },
        policy_execution_manifest={
            "status": "completed",
            "modality_results": {
                "recorded_action_trace": {
                    "status": "completed_reference_replay",
                    "execution_performed": False,
                    "robot_policy_execution_proven": False,
                }
            },
        },
        robot_pov_observation_manifest=_robot_pov_manifest(tmp_path, camera_count=1),
        policy_ranking_scorecard={
            "status": "blocked_inconclusive_ranking",
            "comparison_blockers": ["policy_coverage_missing_required_scenario_eval_run_ids"],
        },
    )

    assert artifact["status"] == "blocked_runtime_inputs_missing_or_invalid"
    assert artifact["data_requirements"]["synchronized_multi_view_cameras"]["status"] == "blocked"
    assert artifact["data_requirements"]["generated_rollout_frames"]["status"] == ("blocked")
    assert artifact["ranking_interpretation"]["status"] == "blocked_inconclusive_ranking"
    assert artifact["ranking_interpretation"]["missing_symmetric_coverage_status"] == (
        "blocked_inconclusive_ranking"
    )


def test_sc3_protocol_recomputes_metrics_only_from_hash_verified_anchor_rows(
    tmp_path: Path,
) -> None:
    rows = []
    split_manifest = tmp_path / "anchor-split-manifest.json"
    split_manifest.write_text(
        json.dumps(
            {
                "schema_version": "sc3_anchor_split_manifest.v1",
                "status": "frozen",
                "split_manifest_id": "protocol-anchor-test-v1",
                "registered_split": "locked_test",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    split_manifest_sha256 = hashlib.sha256(split_manifest.read_bytes()).hexdigest()
    for policy_index, success_count in enumerate((20, 10, 0)):
        checkpoint = tmp_path / f"checkpoint-{policy_index}.bin"
        checkpoint.write_bytes(f"checkpoint-{policy_index}".encode())
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        for replicate_seed in range(20):
            predicted = replicate_seed < success_count
            actual = predicted
            prediction = tmp_path / (f"prediction-{policy_index}-{replicate_seed}.json")
            outcome = tmp_path / f"outcome-{policy_index}-{replicate_seed}.json"
            row_id = f"{policy_index}-{replicate_seed}"
            join = {
                "policy_id": f"p{policy_index}",
                "checkpoint_id": f"c{policy_index}",
                "policy_checkpoint_sha256": checkpoint_sha256,
                "criterion_id": "lift",
                "registered_split": "locked_test",
                "split_manifest_id": "protocol-anchor-test-v1",
                "split_manifest_sha256": split_manifest_sha256,
                "task_family": "pick",
                "task_id": "pick-object",
                "scenario_eval_run_id": f"run-{row_id}",
                "scenario_variation_instance_id": "variation-matched",
                "condition_id": "matched-condition",
                "condition_source_id": "source-trajectory-matched",
                "replicate_id": f"replicate-{row_id}",
                "replicate_seed": replicate_seed,
            }
            prediction_payload = {
                "schema_version": "sc3_anchor_prediction.v1",
                **join,
                "predicted_success": predicted,
            }
            prediction_payload["authority_attestation"] = _anchor_attestation(
                prediction_payload,
                tmp_path,
                f"prediction-{row_id}",
                ANCHOR_PREDICTION_PRIVATE_KEY,
            )
            prediction.write_text(
                json.dumps(prediction_payload, sort_keys=True),
                encoding="utf-8",
            )
            outcome_payload = {
                "schema_version": "sc3_anchor_outcome.v1",
                **join,
                "actual_success": actual,
            }
            outcome_payload["authority_attestation"] = _anchor_attestation(
                outcome_payload,
                tmp_path,
                f"outcome-{row_id}",
                ANCHOR_OUTCOME_PRIVATE_KEY,
            )
            outcome.write_text(
                json.dumps(outcome_payload, sort_keys=True),
                encoding="utf-8",
            )
            rows.append(
                {
                    **join,
                    "predicted_success": predicted,
                    "actual_success": actual,
                    "split_manifest_artifact": {
                        "path": str(split_manifest),
                        "sha256": split_manifest_sha256,
                    },
                    "policy_checkpoint_artifact": {
                        "path": str(checkpoint),
                        "sha256": checkpoint_sha256,
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
            )
    insufficient = build_sc3_eval_protocol_artifact(
        generated_at="now",
        job_request={},
        policy_package_manifest={"selected_modalities": []},
        policy_execution_manifest={},
        robot_pov_observation_manifest=_robot_pov_manifest(tmp_path),
        sim_vs_real_calibration_report={"accepted_anchor_rows": rows[:2]},
    )
    assert insufficient["metrics"]["pearson_success_rate_correlation"]["status"] == (
        "inconclusive_insufficient_n"
    )
    artifact = build_sc3_eval_protocol_artifact(
        generated_at="now",
        job_request={},
        policy_package_manifest={"selected_modalities": ["high_level_skill_trace"]},
        policy_execution_manifest={"status": "completed", "modality_results": {}},
        robot_pov_observation_manifest=_robot_pov_manifest(tmp_path),
        sim_vs_real_calibration_report={
            "accepted_anchor_rows": rows,
        },
    )

    assert artifact["metrics"]["pearson_success_rate_correlation"]["status"] == "measured"
    assert artifact["metrics"]["pearson_success_rate_correlation"]["value"] == 1.0
    assert artifact["metrics"]["spearman_rank_correlation"]["value"] == 1.0
    assert artifact["metrics"]["srcc"]["value"] == 1.0
    assert artifact["metrics"]["mean_maximum_rank_violation"]["value"] == 0.0
    assert artifact["metrics"]["calibration_error"]["value"] == 0.0


def test_sc3_protocol_ignores_caller_eligibility_and_blocks_nonfinite_metrics(
    tmp_path: Path,
) -> None:
    artifact = build_sc3_eval_protocol_artifact(
        generated_at="now",
        job_request={},
        policy_package_manifest={"selected_modalities": []},
        policy_execution_manifest={},
        robot_pov_observation_manifest=_robot_pov_manifest(tmp_path),
        sim_vs_real_calibration_report={
            "pearson_success_rate_correlation": float("nan"),
            "rank_fidelity_claim_eligibility": {
                "status": "eligible",
                "public_rank_fidelity_claim_eligible": True,
                "metrics": {"joint_rank_fidelity": {"eligible": True}},
            },
        },
    )

    assert artifact["public_rank_fidelity_claim_eligible"] is False
    assert (
        artifact["rank_fidelity_claim_eligibility"]["caller_supplied_eligibility_ignored"] is True
    )
    assert (
        "supplied_pearson_success_rate_correlation_missing_or_nonfinite"
        in artifact["supplied_metric_mismatches"]
    )
    assert artifact["data_requirements"]["generated_rollout_frames"]["status"] == ("blocked")
    assert artifact["readiness"]["runtime_ready"] is False


def test_sc3_protocol_requires_hash_bound_generated_rollout_content(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "generated-frame.png"
    _write_correlated_frame(frame, group_index=5, camera_index=1)
    rollout = tmp_path / "generated-rollout.json"
    rollout.write_text(
        json.dumps(
            {
                "schema_version": "sc3_generated_rollout_evidence.v1",
                "status": "completed",
                "rollout_id": "rollout-1",
                "world_model_checkpoint_sha256": "a" * 64,
                "generated_frame_artifacts": [
                    {
                        "path": str(frame),
                        "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    boundary = {
        "generated_rollout_artifact": {
            "path": str(rollout),
            "sha256": hashlib.sha256(rollout.read_bytes()).hexdigest(),
        }
    }
    artifact = build_sc3_eval_protocol_artifact(
        generated_at="now",
        job_request={},
        policy_package_manifest={"selected_modalities": []},
        policy_execution_manifest={},
        robot_pov_observation_manifest=_robot_pov_manifest(tmp_path),
        wam_eval_claim_boundary=boundary,
    )
    assert artifact["data_requirements"]["generated_rollout_frames"]["status"] == ("ready")

    rollout.write_text("{}", encoding="utf-8")
    tampered = build_sc3_eval_protocol_artifact(
        generated_at="now",
        job_request={},
        policy_package_manifest={"selected_modalities": []},
        policy_execution_manifest={},
        robot_pov_observation_manifest=_robot_pov_manifest(tmp_path),
        wam_eval_claim_boundary=boundary,
    )
    assert tampered["data_requirements"]["generated_rollout_frames"]["status"] == ("blocked")
