#!/usr/bin/env python3
"""Freeze the Postshot P0/P1/P2 execution packet for the MuSHRoom bakeoff.

Local-only: binds the exact candidate datasets (pose-corrected point-seeded
COLMAP text and candidate images), frozen per-arm training profiles, the
Windows GPU worker requirements, secret-handling rules, and the still-required
external inputs.  No provider, license, cloud, or network action happens here.

Windows provider decision (recorded, not executed): Vast is rejected for this
lane because Postshot requires Windows and Vast provides Linux container
capacity; the recommended lane is an AWS EC2 G6/G5-class Windows instance (or
Azure NVadsA10 equivalent) admitted through the canonical paid-resource seam
with a bounded IAM role.  That choice still requires operator-supplied
restricted credentials and quota, and a live terms/price recheck.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint_pipeline.decision_evidence_contracts import (  # noqa: E402
    canonical_digest,
    canonical_json,
)

PACKET_SCHEMA_VERSION = "postshot_execution_packet.v1"

COMMON_WORKER_REQUIREMENTS = {
    "operating_system": "windows_server_2022_or_windows_11",
    "gpu": "nvidia_compute_capable_24gb_class_preferred_12gb_minimum",
    "candidate_instances": [
        "aws_ec2_g6.xlarge_windows",
        "aws_ec2_g5.xlarge_windows",
        "azure_nv_a10_v5_windows",
    ],
    "provider_admission": "canonical_paid_resource_seam_with_bounded_iam_role_only",
    "vast_rejected_reason": "postshot_requires_windows_vast_is_linux_container_capacity",
    "gpu_ttl_minutes": 360,
    "gpu_max_retries": 1,
    "gpu_max_concurrent": 1,
    "concurrency_note": "one_postshot_studio_license_one_concurrent_worker",
    "windows_gpu_spend_cap_usd": 90.0,
    "postshot_license_spend_cap_usd": 85.0,
    "license_status": "studio_purchased_2026-08-01_renews_2026-09-01_count_1",
    "credentials_path": "~/.blueprint-secrets/postshot.env",
    "secret_rules": [
        "never_print_log_commit_or_upload_credential_values",
        "cli_login_arguments_must_be_redacted_from_all_process_receipts",
        "prefer_persisting_authenticated_state_in_private_sealed_image",
        "verify_single_concurrent_activation_before_worker_churn",
    ],
    "reverify_before_live": [
        "postshot_current_release_and_cli_flags",
        "eula_and_activation_behavior_on_ephemeral_vms",
        "export_flag_is_export_splat_not_deprecated_variant",
        "instance_pricing_and_windows_license_cost",
        "driver_and_gpu_compatibility",
    ],
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-root", required=True, type=Path)
    parser.add_argument("--teleport-packet", required=True, type=Path)
    arguments = parser.parse_args()
    proxy_root = arguments.proxy_root.resolve()

    report = json.loads((proxy_root / "mushroom_processed_proxy.json").read_text(encoding="utf-8"))
    if report.get("mushroom_processed_proxy_digest") != canonical_digest(
        report, digest_field="mushroom_processed_proxy_digest"
    ):
        raise SystemExit("proxy report digest mismatch")
    pose_only = report["colmap_training_dataset_export_result"]
    run_record = json.loads(
        (
            proxy_root / "initialization" / "mushroom_polycam_initialization_run.v1.json"
        ).read_text(encoding="utf-8")
    )
    if run_record.get("run_digest") != canonical_digest(run_record, digest_field="run_digest"):
        raise SystemExit("initialization run record digest mismatch")
    point_seeded_root = proxy_root / run_record["point_seeded_dataset_relative_path"]
    point_seeded_result = json.loads(
        (point_seeded_root / "colmap_training_dataset_export_result.json").read_text(
            encoding="utf-8"
        )
    )
    teleport_packet = json.loads(arguments.teleport_packet.read_text(encoding="utf-8"))
    if teleport_packet.get("teleport_t1_upload_packet_digest") != canonical_digest(
        teleport_packet, digest_field="teleport_t1_upload_packet_digest"
    ):
        raise SystemExit("teleport packet digest mismatch")

    arms = [
        {
            "arm_id": "P1_postshot_imported_poses_and_points_splat3",
            "input_dataset": {
                "kind": "colmap_text_dataset_point_seeded",
                "relative_path": run_record["point_seeded_dataset_relative_path"],
                "colmap_training_dataset_digest": run_record["point_seeded_dataset_digest"],
                "image_count": run_record["image_count"],
                "initialization_point_count": run_record["initialization_point_count"],
            },
            "training_profile": "splat3",
            "pose_estimation_by_provider": False,
            "expected_behavior": "postshot_skips_tracking_and_trains_from_imported_cameras",
        },
        {
            "arm_id": "P2_postshot_imported_poses_and_points_mcmc",
            "input_dataset": {
                "kind": "colmap_text_dataset_point_seeded",
                "relative_path": run_record["point_seeded_dataset_relative_path"],
                "colmap_training_dataset_digest": run_record["point_seeded_dataset_digest"],
                "image_count": run_record["image_count"],
                "initialization_point_count": run_record["initialization_point_count"],
            },
            "training_profile": "mcmc",
            "pose_estimation_by_provider": False,
            "expected_behavior": "identical_input_to_P1_profile_change_only",
        },
        {
            "arm_id": "P0_postshot_self_tracked_rgb_only",
            "conditional": True,
            "condition": "run_only_if_it_resolves_a_measured_uncertainty_and_budget_allows",
            "input_dataset": {
                "kind": "candidate_images_only",
                "upload_zip_digest": teleport_packet["upload_zip"]["digest"],
                "image_count": teleport_packet["upload_zip"]["image_count"],
            },
            "training_profile": "splat3",
            "pose_estimation_by_provider": True,
            "expected_behavior": "postshot_estimates_its_own_cameras_for_pipeline_attribution",
        },
    ]
    shared_training_intent = {
        "input_resolution": "native_738x994_no_downsampling",
        "training_steps": "postshot_profile_default_recorded_at_execution",
        "splat_budget": "profile_default_recorded_at_execution_within_vram",
        "precommitted_toggles": [
            "no_masking",
            "no_manual_cropping",
            "exposure_compensation_only_if_documented_and_recorded",
        ],
        "required_outputs": [
            "psht_project",
            "exported_splat_ply_or_spz",
            "camera_poses_export_if_supported",
            "training_log",
            "execution_receipt_with_versions_and_durations",
        ],
    }

    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "status": "packet_ready_execution_not_performed",
        "source_capture_digest": report["source_capture_digest"],
        "frozen_split_digest": report["frozen_split_digest"],
        "pose_only_dataset_digest": pose_only["colmap_training_dataset_digest"],
        "pose_only_dataset_relative_path": "trainer_input/" + pose_only["relative_path"],
        "point_seeded_export_result_digest": point_seeded_result[
            "colmap_training_dataset_export_result_digest"
        ],
        "arms": arms,
        "shared_training_intent": shared_training_intent,
        "worker_requirements": COMMON_WORKER_REQUIREMENTS,
        "hidden_images_included": False,
        "provider_sees_hidden_views": False,
        "required_external_inputs": [
            "restricted_windows_cloud_credentials_aws_or_azure",
            "windows_gpu_quota_confirmation",
            "typed_provider_admission_record",
            "live_postshot_cli_and_eula_reverification",
        ],
        "post_run_obligations": [
            "retrieve_psht_ply_spz_cameras_logs_before_teardown",
            "hash_all_retrieved_outputs",
            "terminate_windows_instance_and_verify_provider_zero",
            "deactivate_or_release_postshot_license_session",
            "reconcile_spend_against_caps",
        ],
        "proof_effect": "local_execution_packet_preparation_only",
        "claim_ceiling": "none",
        "timestamp": report["timestamp"],
    }
    packet["postshot_execution_packet_digest"] = canonical_digest(
        packet, digest_field="postshot_execution_packet_digest"
    )
    output_dir = proxy_root / "provider_packets" / "postshot"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "postshot_execution_packet.v1.json"
    payload = (canonical_json(packet) + "\n").encode("utf-8")
    if manifest_path.exists() and manifest_path.read_bytes() != payload:
        raise SystemExit("refusing to overwrite differing packet manifest")
    manifest_path.write_bytes(payload)
    print(
        json.dumps(
            {
                "status": packet["status"],
                "packet_manifest": str(manifest_path),
                "packet_digest": packet["postshot_execution_packet_digest"],
                "point_seeded_dataset": run_record["point_seeded_dataset_relative_path"],
                "arms": [arm["arm_id"] for arm in arms],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
