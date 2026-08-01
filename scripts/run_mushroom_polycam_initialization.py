#!/usr/bin/env python3
"""Bind the MuSHRoom Polycam point cloud into the frozen candidate COLMAP frame.

Local-only runner: strict Polycam PLY import, candidate-frames-only similarity
alignment between the published Polycam and COLMAP trajectories, fail-closed
gates, and a derived point-seeded candidate COLMAP dataset beside the existing
pose-only baseline.  A gate failure is preserved as a typed abstention record;
it never deletes or weakens the pose-only export.  No network, provider, or
hidden-held-out access occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.external_pointcloud_initialization import (
    ExternalPointcloudInitializationError,
    REQUEST_SCHEMA_VERSION as INITIALIZATION_REQUEST_SCHEMA_VERSION,
    compile_external_pointcloud_initialization,
)
from blueprint_pipeline.external_reconstruction_import import (
    ExternalReconstructionImportError,
    build_external_reconstruction_import_request,
    import_external_reconstruction,
)
from blueprint_pipeline.mushroom_processed_proxy import build_mushroom_colmap_export_request
from blueprint_pipeline.reconstruction_colmap_dataset import (
    ColmapTrainingDatasetError,
    bind_colmap_initialization_points,
    export_colmap_training_dataset,
)

RUN_SCHEMA_VERSION = "mushroom_polycam_initialization_run.v1"
ABSTENTION_SCHEMA_VERSION = "mushroom_polycam_initialization_abstention.v1"
POINTCLOUD_RELATIVE_PATH = "long_capture/polycam_pointcloud.ply"
SOURCE_TRAJECTORY_RELATIVE_PATH = "long_capture/transformations.json"
TARGET_TRAJECTORY_RELATIVE_PATH = "long_capture/transformations_colmap.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(record) + "\n").encode("utf-8")
    if path.exists() and path.read_bytes() != payload:
        raise SystemExit(f"refusing to overwrite differing record: {path}")
    path.write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-root", required=True, type=Path)
    parser.add_argument("--scene-root", required=True, type=Path)
    parser.add_argument("--maximum-points", type=int, default=100000)
    parser.add_argument("--maximum-rms-residual", type=float, default=0.05)
    parser.add_argument("--maximum-max-residual", type=float, default=0.20)
    parser.add_argument("--minimum-in-bounds-ratio", type=float, default=0.95)
    parser.add_argument("--bounds-inflation-factor", type=float, default=3.0)
    parser.add_argument("--minimum-bounds-margin", type=float, default=1.0)
    arguments = parser.parse_args()

    proxy_root = arguments.proxy_root.resolve()
    scene_root = arguments.scene_root.resolve()
    report = _load_json(proxy_root / "mushroom_processed_proxy.json")
    if report.get("mushroom_processed_proxy_digest") != canonical_digest(
        report, digest_field="mushroom_processed_proxy_digest"
    ):
        raise SystemExit("proxy report digest mismatch; refusing to proceed")
    camera = _load_json(proxy_root / "camera_observation_manifest.json")
    candidate = _load_json(proxy_root / "candidate_dataset_manifest.json")
    if camera.get("camera_observation_digest") != report["camera_observation_digest"] or (
        camera.get("camera_observation_digest")
        != canonical_digest(camera, digest_field="camera_observation_digest")
    ):
        raise SystemExit("camera observation manifest digest mismatch")
    if candidate.get("candidate_dataset_digest") != canonical_digest(
        candidate, digest_field="candidate_dataset_digest"
    ):
        raise SystemExit("candidate dataset manifest digest mismatch")

    dataset_digest = canonical_digest(
        {
            "candidate": candidate["candidate_dataset_digest"],
            "hidden": report["hidden_evaluator_digest"],
        }
    )
    export_request = build_mushroom_colmap_export_request(
        source_capture_digest=report["source_capture_digest"],
        source_commit_sha=report["source_commit_sha"],
        dataset_digest=dataset_digest,
        split_digest=report["frozen_split_digest"],
        camera_observation_manifest=camera,
        candidate_dataset_manifest=candidate,
        authority_used=report["authority_used"],
        timestamp=report["timestamp"],
        configuration_digest=report["deterministic_configuration_digest"],
    )
    export_request["colmap_training_dataset_export_request_digest"] = canonical_digest(
        export_request, digest_field="colmap_training_dataset_export_request_digest"
    )
    recorded_request_digest = report["colmap_training_dataset_export_result"][
        "parent_artifact_or_event"
    ]["request_digest"]
    if export_request["colmap_training_dataset_export_request_digest"] != recorded_request_digest:
        raise SystemExit(
            "rebuilt COLMAP export request does not match the recorded request digest; "
            "compiler drift must be resolved before binding points"
        )

    pointcloud_path = scene_root / POINTCLOUD_RELATIVE_PATH
    pointcloud_digest = _sha256_file(pointcloud_path)
    declaration = {
        "provider_identity": "polycam",
        "product_tier": "dataset_published_export",
        "terms_version": "mushroom_cc_by_4.0_zenodo_10230733",
        "provider_scan_or_job_identity": "mushroom-koivu-iphone-polycam",
        "export_created_at": "2023-11-28T00:00:00Z",
        "export_performed_by": "mushroom_dataset_publishers",
        "source_capture_identity": report["source_capture_identity"],
        "source_capture_digest": report["source_capture_digest"],
        "ownership_or_license_confirmed": True,
        "commercial_use_status": "permitted",
        "intended_uses": ["reconstruction_initialization_geometry"],
        "consent_status": "not_required",
        "privacy_status": "cleared",
        "confidentiality_terms_status": "public_cc_by_dataset",
        "retention_status": "public_dataset_archival",
        "deletion_status": "not_requested_public_dataset",
        "model_training_terms_status": "cc_by_4.0_attribution_required",
        "competitive_use_status": "cc_by_4.0_permitted",
        "resale_status": "cc_by_4.0_attribution_required",
        "benchmarking_status": "cc_by_4.0_permitted",
        "user_managed_provider_processing_attested": True,
        "blueprint_remote_upload_performed": False,
    }
    declaration["declaration_digest"] = canonical_digest(
        declaration, digest_field="declaration_digest"
    )
    import_request = build_external_reconstruction_import_request(
        {
            "stable_run_identity": report["stable_run_identity"],
            "source_capture_identity": report["source_capture_identity"],
            "source_capture_digest": report["source_capture_digest"],
            "original_file_references": [
                {"artifact_id": "polycam_pointcloud.ply", "digest": pointcloud_digest}
            ],
            "producing_method": "mushroom_polycam_initialization_runner",
            "implementation_version": "1",
            "source_commit_sha": report["source_commit_sha"],
            "deterministic_configuration_digest": report["deterministic_configuration_digest"],
            "input_digests": [
                {"artifact_id": "polycam_pointcloud.ply", "digest": pointcloud_digest}
            ],
            "output_digests": [],
            "train_heldout_split_digest": report["frozen_split_digest"],
            "camera_calibration_binding": {
                "camera_observation_digest": report["camera_observation_digest"]
            },
            "coordinate_frame_declaration": {
                "declaration": "polycam_export_frame",
                "handedness": "not_independently_declared",
                "gravity_alignment": "not_independently_validated",
            },
            "units": "meters",
            "provider_runtime_identity": {"provider": "local", "source_provider": "polycam"},
            "cost_usd": 0.0,
            "duration_seconds": 0.0,
            "authority_used": dict(report["authority_used"]),
            "warnings": [],
            "blockers": [],
            "parent_artifact_or_event": {"digest": report["mushroom_processed_proxy_digest"]},
            "timestamp": report["timestamp"],
            "provider_identity": "polycam",
            "import_lane": "local_external_import",
            "asset_bindings": [
                {
                    "asset_id": "polycam-pointcloud",
                    "relative_path": POINTCLOUD_RELATIVE_PATH,
                    "digest": pointcloud_digest,
                }
            ],
            "provenance_rights_declaration": declaration,
            "remote_calls_authorized": False,
            "remote_calls_performed": False,
            "proof_effect": "external_import_request_only",
            "claim_ceiling": "none",
        }
    )
    import_output_root = proxy_root / "external_imports"
    import_receipt = import_external_reconstruction(
        source_artifact=import_request,
        artifact_root=scene_root,
        output_root=import_output_root,
    )

    candidate_ids = sorted(frame["frame_id"] for frame in candidate["frames"])
    initialization_request = {
        "schema_version": INITIALIZATION_REQUEST_SCHEMA_VERSION,
        "stable_run_identity": report["stable_run_identity"],
        "source_capture_digest": report["source_capture_digest"],
        "frozen_split_digest": report["frozen_split_digest"],
        "camera_observation_digest": report["camera_observation_digest"],
        "source_commit_sha": report["source_commit_sha"],
        "candidate_observation_ids": candidate_ids,
        "pointcloud_asset_id": "polycam-pointcloud",
        "source_trajectory_relative_path": SOURCE_TRAJECTORY_RELATIVE_PATH,
        "target_trajectory_relative_path": TARGET_TRAJECTORY_RELATIVE_PATH,
        "source_trajectory_digest": _sha256_file(scene_root / SOURCE_TRAJECTORY_RELATIVE_PATH),
        "target_trajectory_digest": _sha256_file(scene_root / TARGET_TRAJECTORY_RELATIVE_PATH),
        "alignment_thresholds": {
            "maximum_rms_residual": arguments.maximum_rms_residual,
            "maximum_max_residual": arguments.maximum_max_residual,
            "minimum_in_bounds_ratio": arguments.minimum_in_bounds_ratio,
            "bounds_inflation_factor": arguments.bounds_inflation_factor,
            "minimum_bounds_margin": arguments.minimum_bounds_margin,
        },
        "thresholds_frozen_before_alignment": True,
        "hidden_heldout_access_requested": False,
        "maximum_points": arguments.maximum_points,
        "units": "publisher_pose_units_not_independently_validated",
        "metric_scale_status": "not_independently_validated",
        "coordinate_frame_declaration": {
            "declaration": "mushroom_published_camera_to_world",
            "handedness": "not_independently_declared",
            "gravity_alignment": "not_independently_validated",
        },
        "authority_used": dict(report["authority_used"]),
        "timestamp": report["timestamp"],
    }
    initialization_request["external_pointcloud_initialization_request_digest"] = canonical_digest(
        initialization_request,
        digest_field="external_pointcloud_initialization_request_digest",
    )

    try:
        initialization_result = compile_external_pointcloud_initialization(
            source_artifact=initialization_request,
            import_receipt=import_receipt,
            import_output_root=import_output_root,
            source_trajectory_root=scene_root,
            output_root=proxy_root,
        )
        bound_request = bind_colmap_initialization_points(
            source_artifact=export_request,
            initialization_result=initialization_result,
        )
        export_result = export_colmap_training_dataset(
            source_artifact=bound_request,
            artifact_root=proxy_root,
            output_root=proxy_root / "trainer_input",
            initialization_artifact_root=proxy_root,
        )
    except (
        ExternalPointcloudInitializationError,
        ColmapTrainingDatasetError,
        ExternalReconstructionImportError,
    ) as exc:
        abstention = {
            "schema_version": ABSTENTION_SCHEMA_VERSION,
            "status": "initialization_binding_abstained",
            "failure_codes": list(getattr(exc, "codes", []) or [str(exc)]),
            "source_capture_digest": report["source_capture_digest"],
            "frozen_split_digest": report["frozen_split_digest"],
            "external_import_receipt_digest": import_receipt["external_import_receipt_digest"],
            "initialization_request_digest": initialization_request[
                "external_pointcloud_initialization_request_digest"
            ],
            "pose_only_dataset_preserved": True,
            "recorded_request_digest": recorded_request_digest,
            "proof_effect": "initialization_binding_abstention_only",
            "claim_ceiling": "reconstruction_training_request",
            "timestamp": report["timestamp"],
        }
        abstention["abstention_digest"] = canonical_digest(
            abstention, digest_field="abstention_digest"
        )
        _write_record(
            proxy_root / "initialization" / "mushroom_polycam_initialization_abstention.v1.json",
            abstention,
        )
        print(json.dumps(abstention, indent=2, sort_keys=True))
        print("ABSTAINED: point binding gates failed; pose-only dataset remains valid.")
        return 2

    run_record = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "point_seeded_candidate_dataset_exported",
        "source_capture_digest": report["source_capture_digest"],
        "frozen_split_digest": report["frozen_split_digest"],
        "external_import_receipt_digest": import_receipt["external_import_receipt_digest"],
        "initialization_request_digest": initialization_request[
            "external_pointcloud_initialization_request_digest"
        ],
        "initialization_result_digest": initialization_result[
            "external_pointcloud_initialization_result_digest"
        ],
        "alignment": initialization_result["alignment"],
        "pose_only_request_digest": recorded_request_digest,
        "point_seeded_request_digest": bound_request[
            "colmap_training_dataset_export_request_digest"
        ],
        "point_seeded_dataset_digest": export_result["colmap_training_dataset_digest"],
        "point_seeded_dataset_relative_path": "trainer_input/" + export_result["relative_path"],
        "initialization_point_count": export_result["initialization_point_count"],
        "image_count": export_result["image_count"],
        "hidden_heldout_pixels_included": False,
        "proof_effect": "trainer_input_materialization_only",
        "claim_ceiling": "reconstruction_training_request",
        "timestamp": report["timestamp"],
    }
    run_record["run_digest"] = canonical_digest(run_record, digest_field="run_digest")
    _write_record(
        proxy_root / "initialization" / "mushroom_polycam_initialization_run.v1.json",
        run_record,
    )
    print(json.dumps(run_record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
