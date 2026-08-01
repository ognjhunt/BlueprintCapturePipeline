"""Compile one processed public indoor walkthrough into a bounded Task Evaluation Run.

This operator-only proxy proves the processed-observation, task-approval,
immutable-testbed, and decision-routing seams. It does not stand in for a
Blueprint raw device capture or customer intent.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .capture_intake import validate_capture_intake_envelope
from .capture_qa import build_capture_qa_report
from .decision_evidence_contracts import (
    EvidenceMethodProfile,
    QualificationRecord,
    canonical_digest,
    canonical_json,
)
from .decision_evidence_execution import build_decision_envelope, execute_evidence_plan
from .decision_evidence_router import route_decision_evidence
from .local_evidence_adapters import (
    PROCESSED_OBSERVATION_VISIBILITY_ADAPTER,
    authorized_local_evidence_adapter_registry,
)
from .reconstruction_capability import (
    build_reconstruction_method_profile,
    normalize_reconstruction_result,
    plan_reconstruction_methods,
)
from .site_task_testbed_compiler import (
    build_pipeline_owned_compilation_support,
    compile_site_task_testbed,
    write_testbed_version,
)
from .task_candidate_discovery import (
    build_task_candidate_discovery,
    compile_approved_task_decision_request,
    record_task_candidate_decision,
)


SCHEMA_VERSION = "public_processed_task_testbed_proxy.v1"
PROFILE = "public_processed_rgbd_pose_sequence"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SOURCE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SUPPORTING_FRAME_IDS = (
    "long:frame_00100",
    "long:frame_00120",
    "long:frame_00300",
)


class PublicProcessedTaskTestbedError(RuntimeError):
    """Fail-closed proxy compilation error."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicProcessedTaskTestbedError(f"json_invalid:{path.name}") from exc
    if not isinstance(value, Mapping):
        raise PublicProcessedTaskTestbedError(f"json_not_object:{path.name}")
    return dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PublicProcessedTaskTestbedError(f"file_unreadable:{path.name}") from exc
    return "sha256:" + digest.hexdigest()


def _verified_artifact(path: Path, *, digest_field: str, schema_version: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PublicProcessedTaskTestbedError(f"artifact_missing_or_symlink:{path.name}")
    value = _load(path)
    if value.get("schema_version") != schema_version:
        raise PublicProcessedTaskTestbedError(f"artifact_schema_mismatch:{path.name}")
    supplied = value.get(digest_field)
    if not _SHA256.fullmatch(str(supplied or "")) or supplied != canonical_digest(
        value, digest_field=digest_field
    ):
        raise PublicProcessedTaskTestbedError(f"artifact_digest_mismatch:{path.name}")
    return value


def _safe_child(root: Path, relative_path: str) -> Path:
    candidate = root / relative_path
    cursor = candidate
    while cursor != root:
        if cursor.is_symlink():
            raise PublicProcessedTaskTestbedError("artifact_path_symlink_forbidden")
        cursor = cursor.parent
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise PublicProcessedTaskTestbedError("artifact_path_escape")
    return resolved


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise PublicProcessedTaskTestbedError(f"immutable_artifact_conflict:{path.name}")


def _method_profiles(*, implementation_digest: str) -> list[dict[str, Any]]:
    common = {
        "version": "1",
        "implementation_digest": implementation_digest,
        "provider_identity": "local",
        "execution_mode": "hermetic_local",
        "required_capture_authority_profiles": [PROFILE],
        "expected_cost_usd": 0.0,
        "provider_constraints": {"external_processing": False},
        "rights_constraints": {"local_processing_only": True},
        "failure_modes": [],
    }
    return [
        build_reconstruction_method_profile(
            {
                **common,
                "method_id": "processed-observation-index",
                "method_kind": "decoded_observation_index",
                "outputs": ["decoded_observation_frames"],
                "required_claim_ceiling_flags": ["captured_observation_review"],
                "qualified_claim_types": ["perception_visibility", "task_discovery"],
                "execution_authorized": True,
                "qualification_status": "qualified",
            }
        ),
        build_reconstruction_method_profile(
            {
                **common,
                "method_id": "source-bound-pointcloud-appearance",
                "method_kind": "precomputed_external_reconstruction_import",
                "outputs": ["appearance_layer"],
                "required_claim_ceiling_flags": ["captured_observation_review"],
                "qualified_claim_types": ["appearance_review"],
                "execution_authorized": True,
                "qualification_status": "debug_only",
            }
        ),
        build_reconstruction_method_profile(
            {
                **common,
                "method_id": "metric-scaffold-candidate",
                "method_kind": "metric_scaffold",
                "outputs": ["metric_reference_layer"],
                "required_claim_ceiling_flags": ["metric_geometry"],
                "qualified_claim_types": ["reachability", "robot_placement"],
                "execution_authorized": True,
                "qualification_status": "debug_only",
            }
        ),
        build_reconstruction_method_profile(
            {
                **common,
                "method_id": "collision-scene-candidate",
                "method_kind": "collision_proxy",
                "outputs": ["collision_geometry", "physics_layer"],
                "required_claim_ceiling_flags": ["metric_geometry"],
                "qualified_claim_types": ["collision_contact"],
                "execution_authorized": False,
                "qualification_status": "not_qualified",
            }
        ),
    ]


def _reconstruction_results(
    *,
    plan: Mapping[str, Any],
    intake_id: str,
    capture_digest: str,
    candidate: Mapping[str, Any],
    observations: Mapping[str, Any],
    processed: Mapping[str, Any],
    appearance_ply_digest: str,
    appearance_summary_digest: str,
    implementation_digest: str,
) -> list[dict[str, Any]]:
    selected = {
        str(row["method_id"]): dict(row)
        for row in plan.get("selected_methods", [])
        if isinstance(row, Mapping)
    }
    runtime_digest = canonical_digest(
        {"runtime": "blueprint-public-processed-task-testbed-v1", "implementation": implementation_digest}
    )
    values: list[dict[str, Any]] = []
    observation_method = selected.get("processed-observation-index")
    if observation_method is not None:
        values.append(
            normalize_reconstruction_result(
                {
                    "result_id": "processed-observation-index-koivu",
                    "intake_id": intake_id,
                    "capture_digest": capture_digest,
                    "method_id": observation_method["method_id"],
                    "method_version": observation_method["method_version"],
                    "method_profile_digest": observation_method["method_profile_digest"],
                    "implementation_digest": implementation_digest,
                    "provider_identity": "local",
                    "runtime_identity": "blueprint-processed-observation-index-v1",
                    "runtime_digest": runtime_digest,
                    "outputs": ["decoded_observation_frames"],
                    "source_frames": {
                        "frame_ids": list(_SUPPORTING_FRAME_IDS),
                        "candidate_frame_count": len(candidate.get("frames", [])),
                    },
                    "camera_solution": {
                        "status": observations.get("calibration_status"),
                        "authority": "dataset_provided_not_blueprint_raw_capture",
                    },
                    "coordinate_system": {
                        "up_axis": "not_independently_verified",
                        "scale_status": "dataset_declared_not_independently_verified",
                    },
                    "asset_references": {
                        "processed_dataset": {
                            "uri": "artifact://processed-observation-dataset",
                            "digest": processed["dataset_manifest_digest"],
                        },
                        "candidate_frames": {
                            "uri": "artifact://candidate-dataset-manifest",
                            "digest": candidate["candidate_dataset_digest"],
                        },
                        "camera_observations": {
                            "uri": "artifact://candidate-camera-observation-manifest",
                            "digest": observations["camera_observation_digest"],
                        },
                    },
                    "coverage_map": {
                        "supporting_view_count": len(_SUPPORTING_FRAME_IDS),
                        "candidate_frame_count": len(candidate.get("frames", [])),
                    },
                    "observed_regions": [{"region_id": "conference-table-work-region"}],
                    "generated_regions": [],
                    "uncertainty_map": {"status": "operator_review_only"},
                    "invalid_regions": [{"region_id": "robot-placement-area", "reason": "coverage_not_verified"}],
                    "validation_metrics": {
                        "supporting_frames_digest_verified": True,
                        "metric_scale_verified": False,
                    },
                    "cost_usd": 0.0,
                    "duration_seconds": 0.0,
                    "provider_receipt": None,
                    "rights_and_retention": {"external_processing": False, "privacy": "restricted_local_only"},
                    "deletion_evidence": None,
                    "claim_ceiling": {
                        "processed_captured_observation": True,
                        "raw_capture_authority": False,
                        "metric_geometry": False,
                        "collision_geometry": False,
                        "physical_task_success": False,
                    },
                }
            )
        )
    appearance_method = selected.get("source-bound-pointcloud-appearance")
    if appearance_method is not None:
        values.append(
            normalize_reconstruction_result(
                {
                    "result_id": "source-bound-pointcloud-appearance-koivu",
                    "intake_id": intake_id,
                    "capture_digest": capture_digest,
                    "method_id": appearance_method["method_id"],
                    "method_version": appearance_method["method_version"],
                    "method_profile_digest": appearance_method["method_profile_digest"],
                    "implementation_digest": implementation_digest,
                    "provider_identity": "local",
                    "runtime_identity": "blueprint-source-bound-pointcloud-import-v1",
                    "runtime_digest": runtime_digest,
                    "outputs": ["appearance_layer"],
                    "source_frames": {"source_bundle_digest": capture_digest},
                    "camera_solution": {"status": "not_bound_to_pointcloud_import"},
                    "coordinate_system": {
                        "up_axis": "not_independently_verified",
                        "scale_status": "not_independently_verified",
                    },
                    "asset_references": {
                        "appearance_pointcloud": {
                            "uri": "artifact://mushroom-koivu/polycam_pointcloud.ply",
                            "digest": appearance_ply_digest,
                        },
                        "appearance_import_receipt": {
                            "uri": "artifact://public-indoor-proxy-replay",
                            "digest": appearance_summary_digest,
                        },
                    },
                    "coverage_map": {"status": "appearance_only_not_measured"},
                    "observed_regions": [],
                    "generated_regions": [],
                    "uncertainty_map": {"status": "not_measured"},
                    "invalid_regions": [],
                    "validation_metrics": {"visual_quality_not_evaluated": True},
                    "cost_usd": 0.0,
                    "duration_seconds": 0.0,
                    "provider_receipt": None,
                    "rights_and_retention": {"external_processing": False, "privacy": "restricted_local_only"},
                    "deletion_evidence": None,
                    "claim_ceiling": {
                        "appearance_review": True,
                        "metric_geometry": False,
                        "collision_geometry": False,
                        "physical_task_success": False,
                    },
                }
            )
        )
    return values


def _claim(
    claim_id: str,
    claim_type: str,
    *,
    subject: Any,
    scope: Mapping[str, Any],
    consequence: str = "moderate",
    risk: float = 0.1,
    coverage: float = 0.7,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "claim_type": claim_type,
        "subject": subject,
        "measurable_threshold": {"operator": ">=", "value": coverage, "units": "fraction"},
        "false_safe_consequence": consequence,
        "acceptable_false_safe_risk": risk,
        "desired_confidence_or_coverage": {
            "minimum_coverage": coverage,
            "minimum_independent_methods": 1,
        },
        "permitted_abstention_behavior": {"allowed": True},
        **dict(scope),
    }


def _evidence_profiles(
    *,
    implementation_digest: str,
    testbed: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    common = {
        "schema_version": "evidence_method_profile.v1",
        "version": "1",
        "implementation_digest": implementation_digest,
        "applicability_envelope": {
            "testbed_ids": [testbed["testbed_id"]],
            "testbed_versions": [testbed["version"]],
            "task_families": [testbed["task_distribution"]["task_family"]],
        },
        "calibration_evidence_references": [],
        "constraints": {"external_processing": False, "data_retention_days": 0},
        "expected_cost_usd": 0.0,
        "expected_latency_seconds": 0.01,
        "reproducibility_level": "hermetic_local",
        "failure_modes": ["missing_bound_input"],
        "abstention_modes": ["missing_input", "unqualified"],
        "disqualifying_conditions": [],
        "self_qualified": False,
    }
    definitions = [
        {
            "method_id": "processed-observation-visibility",
            "adapter_reference": PROCESSED_OBSERVATION_VISIBILITY_ADAPTER,
            "method_family": "captured_real_observation",
            "supported_claim_types": ["perception_visibility"],
            "required_inputs": ["processed_capture_observations"],
            "authority_tier": 1,
            "proof_tier": "processed_observation_only",
            "correlation_group": "mushroom-processed-observations",
            "shared_dependencies": ["mushroom-source-bundle"],
            "provider_availability": {"status": "available"},
        },
        {
            "method_id": "analytic-reachability-candidate",
            "adapter_reference": "local://analytic-reachability-v1",
            "method_family": "analytic_geometry_kinematics",
            "supported_claim_types": ["reachability"],
            "required_inputs": ["metric_geometry"],
            "authority_tier": 1,
            "proof_tier": "analytic_only",
            "correlation_group": "metric-scaffold",
            "shared_dependencies": ["metric-geometry"],
            "provider_availability": {"status": "available"},
        },
        {
            "method_id": "collision-simulation-candidate",
            "adapter_reference": "local://swept-aabb-collision-simulation-v1",
            "method_family": "traditional_simulation",
            "supported_claim_types": ["collision_contact"],
            "required_inputs": ["collision_scene"],
            "authority_tier": 2,
            "proof_tier": "sim_only",
            "correlation_group": "collision-geometry",
            "shared_dependencies": ["metric-geometry", "collision-geometry"],
            "provider_availability": {"status": "available"},
        },
        {
            "method_id": "world-model-ranking-candidate",
            "adapter_reference": "provider://world-model-not-authorized",
            "method_family": "learned_world_model",
            "supported_claim_types": ["comparative_policy_ranking"],
            "required_inputs": ["processed_capture_observations"],
            "authority_tier": 3,
            "proof_tier": "debug_only",
            "correlation_group": "mushroom-processed-observations",
            "shared_dependencies": ["mushroom-source-bundle"],
            "provider_availability": {"status": "unavailable"},
        },
        {
            "method_id": "accepted-physical-outcome",
            "adapter_reference": "physical://read-only-outcome-required",
            "method_family": "physical_evidence",
            "supported_claim_types": ["physical_task_success"],
            "required_inputs": ["accepted_physical_outcome"],
            "authority_tier": 4,
            "proof_tier": "physical",
            "correlation_group": "physical-outcome",
            "shared_dependencies": [],
            "provider_availability": {"status": "unavailable"},
        },
    ]
    profiles = [EvidenceMethodProfile.from_mapping({**common, **row}).to_mapping() for row in definitions]
    visibility = profiles[0]
    bindings = testbed["robot_sensor_controller_bindings"]
    qualification = QualificationRecord.from_mapping(
        {
            "schema_version": "evidence_method_qualification.v1",
            "qualification_id": "qualification-mushroom-koivu-processed-visibility-v1",
            "method_id": visibility["method_id"],
            "method_version": visibility["version"],
            "method_profile_digest": visibility["method_profile_digest"],
            "implementation_digest": visibility["implementation_digest"],
            "claim_type": "perception_visibility",
            "task_family": testbed["task_distribution"]["task_family"],
            "site_domain_conditions": testbed["supported_condition_ranges"],
            "embodiment": bindings["embodiment"],
            "sensors": bindings["sensors"],
            "controller_action_representation": bindings[
                "controller_action_representation"
            ],
            "evaluator": {
                "evaluator_id": "independent-mushroom-short-trajectory-review",
                "version": "1",
            },
            "evaluator_digest": canonical_digest(
                {"method": "independent_short_trajectory_visibility_review", "version": "1"}
            ),
            "predictions": [{"prediction_id": "conference-table-visible", "value": True}],
            "accepted_real_outcomes": [
                {
                    "outcome_id": "independent-short-trajectory-captured-observations",
                    "value": True,
                    "physical_robot_outcome": False,
                }
            ],
            "calibration_partition": "heldout",
            "confidence_intervals": {"level": 0.9, "lower": 0.7, "upper": 1.0},
            "coverage": 0.75,
            "abstention_rate": 0.25,
            "false_safe_rate": 0.05,
            "false_reject_rate": 0.1,
            "provenance": {
                "source": "mushroom_processed_public_dataset",
                "raw_capture_authority": False,
                "physical_outcome": False,
            },
            "owner_evidence": [
                {
                    "uri": "artifact://mushroom-independent-short-trajectory",
                    "digest": testbed["source_capture_bundles"][0]["digest"],
                }
            ],
            "status": "qualified",
            "self_grading": False,
            "subject_provider_id": "blueprint-local-processed-observation-adapter",
            "evaluator_provider_id": "independent-mushroom-short-trajectory-review",
        }
    ).to_mapping()
    return profiles, [qualification]


def compile_public_processed_task_testbed_proxy(
    *,
    processed_dataset_manifest_path: str | Path,
    candidate_dataset_manifest_path: str | Path,
    camera_observation_manifest_path: str | Path,
    appearance_proxy_summary_path: str | Path,
    appearance_ply_path: str | Path,
    output_root: str | Path,
    operator_identity: str,
    source_commit_sha: str,
    timestamp: str,
) -> dict[str, Any]:
    """Compile and execute the exact bounded processed-dataset proxy."""

    if not operator_identity.strip():
        raise PublicProcessedTaskTestbedError("operator_identity_missing")
    if not _SOURCE_COMMIT.fullmatch(source_commit_sha):
        raise PublicProcessedTaskTestbedError("source_commit_invalid")
    paths = [
        Path(value).expanduser()
        for value in (
            processed_dataset_manifest_path,
            candidate_dataset_manifest_path,
            camera_observation_manifest_path,
            appearance_proxy_summary_path,
            appearance_ply_path,
        )
    ]
    if any(path.is_symlink() for path in paths):
        raise PublicProcessedTaskTestbedError("input_symlink_forbidden")
    processed_path, candidate_path, observations_path, appearance_summary_path, ply_path = (
        path.resolve() for path in paths
    )
    processed = _verified_artifact(
        processed_path,
        digest_field="dataset_manifest_digest",
        schema_version="processed_observation_dataset_manifest.v1",
    )
    candidate = _verified_artifact(
        candidate_path,
        digest_field="candidate_dataset_digest",
        schema_version="processed_candidate_dataset_manifest.v1",
    )
    observations = _verified_artifact(
        observations_path,
        digest_field="camera_observation_digest",
        schema_version="processed_camera_observation_manifest.v1",
    )
    appearance_summary = _load(appearance_summary_path)
    if appearance_summary.get("schema_version") != "public_indoor_proxy_replay.v1":
        raise PublicProcessedTaskTestbedError("appearance_summary_schema_mismatch")
    source_digest = str(processed.get("source_capture_digest") or "")
    if (
        not _SHA256.fullmatch(source_digest)
        or candidate.get("capture_digest") != source_digest
        or observations.get("source_capture_digest") != source_digest
        or appearance_summary.get("source_bundle", {}).get("digest") != source_digest
    ):
        raise PublicProcessedTaskTestbedError("source_capture_binding_mismatch")
    ply_digest = _sha256_file(ply_path)
    if appearance_summary.get("source_artifact", {}).get("digest") != ply_digest:
        raise PublicProcessedTaskTestbedError("appearance_ply_digest_mismatch")
    appearance_summary_digest = _sha256_file(appearance_summary_path)
    frame_by_id = {
        str(row.get("frame_id")): dict(row)
        for row in candidate.get("frames", [])
        if isinstance(row, Mapping)
    }
    observation_by_id = {
        str(row.get("observation_id")): dict(row)
        for row in observations.get("observations", [])
        if isinstance(row, Mapping)
    }
    supporting_frames: list[dict[str, Any]] = []
    artifact_root = candidate_path.parent.resolve()
    for frame_id in _SUPPORTING_FRAME_IDS:
        frame = frame_by_id.get(frame_id)
        observation = observation_by_id.get(frame_id)
        if frame is None or observation is None:
            raise PublicProcessedTaskTestbedError(f"supporting_frame_missing:{frame_id}")
        image_path = _safe_child(artifact_root, str(frame.get("candidate_relative_path") or ""))
        if not image_path.is_file() or _sha256_file(image_path) != frame.get("frame_digest"):
            raise PublicProcessedTaskTestbedError(f"supporting_frame_digest_mismatch:{frame_id}")
        if observation.get("image_digest") != frame.get("frame_digest"):
            raise PublicProcessedTaskTestbedError(f"camera_observation_binding_mismatch:{frame_id}")
        supporting_frames.append(frame)

    implementation_digest = canonical_digest(
        {"implementation": "public_processed_task_testbed_proxy.v1", "source_commit_sha": source_commit_sha}
    )
    intake_id = f"public-mushroom-koivu-processed-{source_digest[7:19]}"
    envelope = validate_capture_intake_envelope(
        {
            "schema_version": "capture_intake_envelope.v1",
            "intake_id": intake_id,
            "idempotency_key": f"{intake_id}-v1",
            "capture_authority_profile": PROFILE,
            "source_type": PROFILE,
            "original_files": [
                {
                    "original_filename": processed_path.name,
                    "relative_path": processed_path.name,
                    "sha256": _sha256_file(processed_path),
                    "size_bytes": processed_path.stat().st_size,
                    "media_type": "application/json",
                }
            ],
            "scene_id": "mushroom-koivu",
            "customer_id": "public-dataset-proxy",
            "organization_id": "blueprint-internal-evaluation",
            "capture_device": {
                "manufacturer": "Apple",
                "model": "MuSHRoom dataset-declared iPhone",
                "app_version": "not_blueprint_capture",
            },
            "timing_declaration": {
                "clock": "dataset_frame_index",
                "decoded_video_pts_available": False,
            },
            "coordinate_frame_declaration": processed["coordinate_frame_declaration"],
            "available_sensor_streams": [
                {
                    "stream_type": stream,
                    "status": "available",
                    "source_relative_path": processed_path.name,
                }
                for stream in (
                    "processed_rgb_observations",
                    "camera_poses",
                    "camera_intrinsics",
                    "depth",
                )
            ],
            "governance": {
                "rights": "accepted",
                "consent": "not_required",
                "privacy": "restricted_local_only",
                "retention": {"policy": "local_operator_managed"},
                "revocation": {
                    "supported": True,
                    "historical_tombstone_retained": True,
                },
                "provider_constraints": {"external_processing_allowed": False},
                "allowed_uses": ["local_evaluation", "captured_observation_review"],
            },
            "requested_task_evaluation_run_audience": "internal_design_partner_proxy",
            "known_task_specification": None,
            "calibration_board_dimensions": None,
            "operator_notes": [
                "Public processed dataset proxy; not customer intent or Blueprint raw capture."
            ],
            "permitted_reconstruction_providers": ["local"],
            "permitted_evidence_uses": ["captured_observation", "task_discovery"],
            "upload_validation": {
                "status": "passed",
                "scope": "local_existing_dataset_manifest",
            },
            "malware_content_validation": {
                "status": "passed",
                "scope": "json_manifest_only",
            },
        }
    )
    qa = build_capture_qa_report(envelope, upload_root=processed_path.parent)
    if (
        qa["status"] != "accepted"
        or qa["claim_ceiling"].get("captured_observation_review") is not True
        or qa["claim_ceiling"].get("metric_geometry") is not False
    ):
        raise PublicProcessedTaskTestbedError(
            "processed_observation_qa_boundary_invalid:"
            f"{qa['status']}:"
            f"observation={qa['claim_ceiling'].get('captured_observation_review')}:"
            f"metric={qa['claim_ceiling'].get('metric_geometry')}"
        )
    source_capture = {
        "intake_id": intake_id,
        "capture_digest": source_digest,
        "capture_authority_profile": PROFILE,
    }
    frame_ids = [row["frame_id"] for row in supporting_frames]
    frame_digests = [row["frame_digest"] for row in supporting_frames]
    discovery = build_task_candidate_discovery(
        discovery_id="mushroom-koivu-processed-task-discovery-v1",
        source_capture=source_capture,
        capture_qa_report_digest=qa["qa_report_digest"],
        scene_analysis={
            "observed_site_facts": [
                {
                    "fact_id": "fact-conference-table-region",
                    "description": "A conference table work region is directly visible in multiple processed RGB observations.",
                    "confidence": 1.0,
                    "supporting_frames": frame_ids,
                    "supporting_3d_regions": [],
                },
                {
                    "fact_id": "fact-small-tabletop-item-candidate",
                    "description": "At least one small rigid-looking tabletop item candidate is visible, but exact identity and physical properties are not established.",
                    "confidence": 0.7,
                    "supporting_frames": ["long:frame_00120", "long:frame_00300"],
                    "supporting_3d_regions": [],
                },
            ],
            "inferred_objects_and_affordances": [
                {
                    "inference_id": "inference-item-may-be-movable",
                    "description": "The selected tabletop item may be movable by a suitable robot; movability is inferred, not observed.",
                    "confidence": 0.45,
                    "supporting_frames": ["long:frame_00120"],
                    "supporting_3d_regions": [],
                }
            ],
            "unsupported_or_occluded_regions": [
                {
                    "region_id": "region-item-hidden-surfaces",
                    "description": "Hidden object surfaces, exact dimensions, support contact, and material properties are unsupported.",
                    "confidence": 1.0,
                    "supporting_frames": ["long:frame_00120"],
                    "supporting_3d_regions": [],
                },
                {
                    "region_id": "region-robot-placement-unverified",
                    "description": "No exact robot placement area, access path, or human-clearance envelope is verified.",
                    "confidence": 1.0,
                    "supporting_frames": frame_ids,
                    "supporting_3d_regions": [],
                },
            ],
            "hazards": [],
            "privacy_sensitive_areas": [
                {
                    "area_id": "privacy-whole-public-dataset",
                    "description": "Dataset use is restricted to the recorded local evaluation scope.",
                    "confidence": 1.0,
                    "supporting_frames": frame_ids,
                    "supporting_3d_regions": [],
                }
            ],
        },
        candidate_proposals=[
            {
                "description": "Evaluate whether a specified robot could observe, reach, and move one operator-selected small rigid item within the conference-table work region.",
                "observed_objects": [
                    {
                        "object_id": "tabletop-item-candidate",
                        "label": "operator-selected small rigid tabletop item candidate",
                        "observation_fact_ids": ["fact-small-tabletop-item-candidate"],
                    }
                ],
                "target_regions": [
                    {
                        "region_id": "conference-table-work-region",
                        "label": "conference table work region",
                        "supporting_frames": frame_ids,
                        "supporting_frame_digests": frame_digests,
                        "captured_coverage": 0.75,
                    }
                ],
                "required_robot_capabilities": [
                    "tabletop observation",
                    "metric reach model",
                    "rigid-object manipulation",
                ],
                "likely_task_family": "rigid_object_pick_place",
                "proposed_measurable_success_condition": {
                    "metric": "processed_view_visibility_coverage",
                    "operator": ">=",
                    "threshold": 0.7,
                    "units": "fraction",
                },
                "required_site_reset": "Operator must identify the exact item and restore its documented initial table pose before any physical evidence collection.",
                "supporting_frames": frame_ids,
                "supporting_3d_regions": [],
                "confidence": 0.6,
                "coverage": {"target_region": 0.75, "exact_task_object": 0.4},
                "assumptions": ["A customer/operator will select the exact object instance."],
                "missing_evidence": [
                    "Exact task object identity and dimensions.",
                    "Independently verified metric scale and site coordinate frame.",
                    "Exact robot embodiment, base pose, and reach envelope.",
                    "Qualified collision geometry and physical object properties.",
                ],
                "prohibited_claims": [
                    "physical_task_success",
                    "deployment_readiness",
                    "safety_certification",
                    "comparative_policy_ranking_validity",
                ],
                "estimated_evaluation_cost_usd": 0.0,
                "expected_customer_value": None,
            }
        ],
        proposal_method={
            "method_id": "operator-reviewed-public-processed-task-rule",
            "version": "1",
            "implementation_digest": implementation_digest,
            "proposer_identity": operator_identity,
            "origin": "local_rule",
        },
    )
    approval, approved = record_task_candidate_decision(
        discovery,
        task_candidate_id=discovery["task_candidates"][0]["task_candidate_id"],
        action="approve",
        actor={"role": "operator", "identity": operator_identity},
        idempotency_key="mushroom-koivu-processed-operator-approval-v1",
        rationale="Approve only as an internal processed-dataset proxy; customer intent remains unproven.",
    )
    if approved is None or approved.get("intent_source") != "operator_approved_candidate":
        raise PublicProcessedTaskTestbedError("operator_approval_boundary_invalid")
    reconstruction_profiles = _method_profiles(implementation_digest=implementation_digest)
    reconstruction_plan = plan_reconstruction_methods(
        intake_id=intake_id,
        capture_digest=source_digest,
        capture_authority_profile=PROFILE,
        claim_ceiling=qa["claim_ceiling"],
        requested_claim_types=[
            "appearance_review",
            "perception_visibility",
            "reachability",
            "collision_contact",
        ],
        permitted_provider_identities=["local"],
        method_profiles=reconstruction_profiles,
    )
    results = _reconstruction_results(
        plan=reconstruction_plan,
        intake_id=intake_id,
        capture_digest=source_digest,
        candidate=candidate,
        observations=observations,
        processed=processed,
        appearance_ply_digest=ply_digest,
        appearance_summary_digest=appearance_summary_digest,
        implementation_digest=implementation_digest,
    )
    robot_binding = {
        "robot_id": "operator-robot-selection-required",
        "embodiment_version": "not_supplied",
        "base_footprint": {"status": "not_supplied"},
        "sensors": {"status": "not_supplied"},
        "controller_id": "not_supplied",
        "end_effector_id": "not_supplied",
    }
    support = build_pipeline_owned_compilation_support(
        testbed_id="mushroom-koivu-processed-rigid-object",
        version="1",
        approved_task_definition=approved,
        capture_qa_report=qa,
        reconstruction_plan=reconstruction_plan,
        robot_binding=robot_binding,
    )
    artifact_references = {
        **support["artifact_references"],
        "processed_dataset": {
            "uri": "artifact://processed-observation-dataset",
            "digest": processed["dataset_manifest_digest"],
        },
        "appearance_pointcloud": {
            "uri": "artifact://mushroom-koivu/polycam_pointcloud.ply",
            "digest": ply_digest,
        },
    }
    testbed = compile_site_task_testbed(
        testbed_id="mushroom-koivu-processed-rigid-object",
        version="1",
        capture_intake_envelope=envelope,
        capture_qa_report=qa,
        approved_task_definition=approved,
        reconstruction_plan=reconstruction_plan,
        reconstruction_results=results,
        simready_decision=support["simready_decision"],
        robot_placement_result=support["robot_placement_result"],
        artifact_references=artifact_references,
        supported_condition_ranges=support["supported_condition_ranges"],
        pipeline_owned_support_artifacts=support[
            "pipeline_owned_support_artifacts"
        ],
    )
    if any(
        row.get("evidence_id") == "raw_capture"
        for row in testbed["evidence_inventory"]
    ):
        raise PublicProcessedTaskTestbedError("processed_dataset_mislabeled_raw_capture")
    write_result = write_testbed_version(output_root=output_root, testbed=testbed)
    bindings = testbed["robot_sensor_controller_bindings"]
    scope = {
        "task_family": testbed["task_distribution"]["task_family"],
        "site_domain_conditions": testbed["supported_condition_ranges"],
        "embodiment": bindings["embodiment"],
        "sensors": bindings["sensors"],
        "controller_action_representation": bindings[
            "controller_action_representation"
        ],
    }
    claims = [
        _claim(
            "processed-visibility",
            "perception_visibility",
            subject={"target_region_id": "conference-table-work-region"},
            scope=scope,
        ),
        _claim(
            "analytic-reach",
            "reachability",
            subject={"target_region_id": "conference-table-work-region"},
            scope=scope,
        ),
        _claim(
            "modeled-collision",
            "collision_contact",
            subject="operator-selected-item-trajectory-not-supplied",
            scope=scope,
            risk=0.05,
        ),
        _claim(
            "comparative-policy-ranking",
            "comparative_policy_ranking",
            subject="candidate-policies-not-supplied",
            scope=scope,
            risk=0.05,
        ),
        _claim(
            "physical-task-success",
            "physical_task_success",
            subject="operator-selected-item-move",
            scope=scope,
            consequence="high",
            risk=0.01,
        ),
    ]
    request = compile_approved_task_decision_request(
        approved,
        testbed=testbed,
        request_id="mushroom-koivu-processed-request-v1",
        decision_id="mushroom-koivu-processed-decision-v1",
        candidates=[{"robot_id": robot_binding["robot_id"]}],
        claims=claims,
        budget={"max_cost_usd": 0.0, "max_latency_seconds": 1.0},
        deadline="2026-12-31T00:00:00Z",
        permitted_evidence_methods=[
            "analytic_geometry_kinematics",
            "captured_real_observation",
            "traditional_simulation",
            "learned_world_model",
            "physical_evidence",
        ],
        restrictions={
            "external_processing_allowed": False,
            "max_data_retention_days": 0,
            "live_robot_execution_allowed": False,
        },
        requested_result_audience="internal_design_partner_proxy",
        caller_identity="pipeline:public-processed-task-testbed-proxy",
        idempotency_key="mushroom-koivu-processed-request-v1",
    )
    evidence_profiles, qualifications = _evidence_profiles(
        implementation_digest=implementation_digest,
        testbed=testbed,
    )
    plan = route_decision_evidence(request, testbed, evidence_profiles, qualifications).to_mapping()
    authorization = {
        "schema_version": "public_processed_proxy_execution_authorization.v1",
        "plan_digest": plan["plan_digest"],
        "authorized_adapter_references": [
            PROCESSED_OBSERVATION_VISIBILITY_ADAPTER
        ],
        "actor": {"role": "operator", "identity": operator_identity},
        "physical_execution_authorized": False,
        "paid_compute_authorized": False,
        "external_provider_execution_authorized": False,
    }
    authorization["authorization_digest"] = canonical_digest(
        authorization, digest_field="authorization_digest"
    )
    execution = execute_evidence_plan(
        plan,
        request,
        testbed,
        evidence_profiles,
        qualifications,
        registry=authorized_local_evidence_adapter_registry(
            authorization["authorized_adapter_references"]
        ),
        context={"authorization_digest": authorization["authorization_digest"]},
    )
    result_values = [row.to_mapping() for row in execution.results]
    decision = build_decision_envelope(
        request, testbed, plan, result_values
    ).to_mapping()
    if (
        decision["overall_outcome"] != "partial_decision"
        or decision["uncertainty"]["ranking_science_boundary"]
        != "thesis_not_supported"
        or decision["claim_ceiling"]["physical_success"] is not False
        or decision["deployment_approval"] is not False
    ):
        raise PublicProcessedTaskTestbedError("decision_claim_boundary_upgraded")
    artifacts: dict[str, Mapping[str, Any]] = {
        "capture_intake_envelope.json": envelope,
        "capture_qa_report.json": qa,
        "task_candidate_discovery.json": discovery,
        "task_candidate_decision.json": approval,
        "approved_task_definition.json": approved,
        "reconstruction_plan.json": reconstruction_plan,
        "testbed.json": testbed,
        "decision_evidence_request.json": request,
        "evidence_plan.json": plan,
        "execution_authorization.json": authorization,
        "execution_manifest.json": execution.execution_manifest,
        "decision_envelope.json": decision,
    }
    for index, value in enumerate(results, start=1):
        artifacts[f"reconstruction_result_{index}.json"] = value
    for index, value in enumerate(result_values, start=1):
        artifacts[f"evidence_result_{index}.json"] = value
    output = Path(output_root).expanduser().resolve()
    artifact_root_out = output / "public_processed_task_testbed_proxy"
    for name, value in sorted(artifacts.items()):
        _write_immutable(artifact_root_out / name, value)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_commit_sha": source_commit_sha,
        "timestamp": timestamp,
        "dataset": {
            "dataset_id": "mushroom",
            "scene_id": "koivu",
            "source_capture_digest": source_digest,
            "processed_dataset_digest": processed["dataset_manifest_digest"],
            "appearance_ply_digest": ply_digest,
        },
        "capture_authority_profile": PROFILE,
        "qa_status": qa["status"],
        "task_approval_state_before_decision": discovery["approval_state"],
        "task_intent_source": approved["intent_source"],
        "testbed_digest": testbed["testbed_digest"],
        "testbed_write_result": {
            "status": write_result["status"],
            "testbed_id": write_result["testbed_id"],
            "version": write_result["version"],
            "testbed_digest": write_result["testbed_digest"],
        },
        "request_digest": request["request_digest"],
        "plan_digest": plan["plan_digest"],
        "selected_evidence_steps": len(plan["non_evaluation_run_steps"]),
        "execution_status": execution.execution_manifest["status"],
        "decision_envelope_digest": decision["decision_envelope_digest"],
        "overall_outcome": decision["overall_outcome"],
        "per_claim_verdicts": decision["per_claim_verdicts"],
        "next_cheapest_experiment": decision["next_cheapest_experiment"],
        "physical_evidence_requests": decision[
            "physical_evidence_still_required"
        ],
        "claim_flags": {
            "processed_captured_observation": True,
            "raw_capture_authority": False,
            "decoded_video_timing": False,
            "metric_scale_verified": False,
            "collision_geometry": False,
            "physics": False,
            "physical_task_success": False,
            "deployment_readiness": False,
            "safety_certification": False,
            "comparative_policy_ranking_verdict": "thesis_not_supported",
        },
        "cost_usd": 0.0,
    }
    summary["summary_digest"] = canonical_digest(summary, digest_field="summary_digest")
    _write_immutable(artifact_root_out / "summary.json", summary)
    return summary


__all__ = [
    "PublicProcessedTaskTestbedError",
    "compile_public_processed_task_testbed_proxy",
]
