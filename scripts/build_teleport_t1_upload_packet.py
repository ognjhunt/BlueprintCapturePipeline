#!/usr/bin/env python3
"""Build the immutable Teleport T1 candidate-only upload packet, locally.

Produces a deterministic ZIP containing exactly the 265 candidate images (no
hidden frames, no metadata sidecars), a digest-bound packet manifest with the
frozen T1 provider configuration, the still-required external inputs, and the
claim boundary.  This script performs no network access and no upload; the
packet is admission input for a later explicitly authorized Teleport run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint_pipeline.decision_evidence_contracts import (  # noqa: E402
    canonical_digest,
    canonical_json,
)

PACKET_SCHEMA_VERSION = "teleport_t1_upload_packet.v1"
EXPECTED_IMAGE_COUNT = 265
# Frozen T1 arm configuration per the bakeoff design.  Values marked
# reverify_before_live must be checked against live Teleport docs/plan at
# execution time; a mismatch requires a new frozen packet, not an in-flight
# edit.
FROZEN_T1_CONFIGURATION = {
    "arm_id": "T1_teleport_managed_maximum_quality",
    "provider": "teleport",
    "upload_form": "bulk-images_zip_of_candidate_rgb_images_only",
    "training_parameters": {
        "modelv3": {
            "spherical_harmonics_degrees": 3,
            "splat_count": 10_000_000,
            "training_resolution": 3200,
            "level_of_detail": False,
        }
    },
    "reverify_before_live": [
        "current_api_endpoints_and_plan_tier",
        "custom_parameter_availability_on_paid_plan",
        "output_ply_and_camera_metadata_format",
        "deletion_endpoint_behavior",
        "terms_and_data_handling_version",
    ],
    "exact_operator_spend_cap_required_before_upload": True,
    "preapproved_provider_quote_required_before_upload": True,
    "provider_delete_success_response": "204_no_content",
    "durable_provider_deletion_receipt_available": False,
    "provider_sees_hidden_views": False,
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
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args()
    proxy_root = arguments.proxy_root.resolve()
    output_dir = (arguments.output_dir or proxy_root / "provider_packets" / "teleport_t1").resolve()
    terms_path = REPO_ROOT / "docs/evidence/teleport_provider_terms_review_2026-08-03.json"
    terms_review = json.loads(terms_path.read_text(encoding="utf-8"))
    if terms_review.get("teleport_provider_terms_review_digest") != canonical_digest(
        terms_review, digest_field="teleport_provider_terms_review_digest"
    ):
        raise SystemExit("Teleport terms review digest mismatch")

    report = json.loads((proxy_root / "mushroom_processed_proxy.json").read_text(encoding="utf-8"))
    if report.get("mushroom_processed_proxy_digest") != canonical_digest(
        report, digest_field="mushroom_processed_proxy_digest"
    ):
        raise SystemExit("proxy report digest mismatch")
    if "pose_image_consistency_inconclusive" in (report.get("blockers") or []):
        raise SystemExit("pose consistency inconclusive; refusing to build provider packet")
    if report.get("license") != "CC-BY-4.0":
        raise SystemExit("frozen MuSHRoom proxy must retain the reviewed CC-BY-4.0 license")
    candidate = json.loads(
        (proxy_root / "candidate_dataset_manifest.json").read_text(encoding="utf-8")
    )
    if candidate.get("candidate_dataset_digest") != canonical_digest(
        candidate, digest_field="candidate_dataset_digest"
    ):
        raise SystemExit("candidate manifest digest mismatch")
    frames = sorted(candidate["frames"], key=lambda row: row["frame_id"])
    rows = []
    for frame in frames:
        relative = str(frame["candidate_relative_path"])
        if "evaluator_hidden" in relative or "held_out" in relative:
            raise SystemExit(f"hidden path leaked into candidate manifest: {relative}")
        path = proxy_root / relative
        digest = _sha256_file(path)
        if digest != frame["frame_digest"]:
            raise SystemExit(f"candidate image digest mismatch: {relative}")
        upload_name = f"{frame['frame_id']}{Path(relative).suffix.lower()}"
        rows.append((upload_name, path, digest, frame["frame_id"]))
    names = [name for name, _, _, _ in rows]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate upload names")
    if len(rows) != EXPECTED_IMAGE_COUNT:
        raise SystemExit(
            f"frozen Teleport T1 split requires {EXPECTED_IMAGE_COUNT} candidate images; "
            f"observed {len(rows)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "teleport_t1_candidate_images.zip"
    temporary_fd, temporary_name = tempfile.mkstemp(prefix=".teleport-t1-", suffix=".zip", dir=output_dir)
    os.close(temporary_fd)
    temporary_zip = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_STORED) as archive:
            for upload_name, path, _digest, _frame_id in rows:
                info = zipfile.ZipInfo(upload_name, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = (0o644 & 0xFFFF) << 16
                with path.open("rb") as source, archive.open(info, "w") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
        if zip_path.exists():
            if _sha256_file(zip_path) != _sha256_file(temporary_zip):
                raise SystemExit("refusing to overwrite differing Teleport upload ZIP")
        else:
            os.replace(temporary_zip, zip_path)
    finally:
        temporary_zip.unlink(missing_ok=True)
    zip_digest = _sha256_file(zip_path)

    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "status": "ready_to_upload_authorization_required",
        "source_capture_digest": report["source_capture_digest"],
        "source_proxy_digest": report["mushroom_processed_proxy_digest"],
        "dataset_class": "rights_cleared_public_dataset",
        "source_license": report["license"],
        "customer_or_confidential_data_included": False,
        "frozen_split_digest": report["frozen_split_digest"],
        "candidate_dataset_digest": candidate["candidate_dataset_digest"],
        "deterministic_configuration_digest": report["deterministic_configuration_digest"],
        "upload_zip": {
            "relative_path": zip_path.name,
            "digest": zip_digest,
            "size_bytes": zip_path.stat().st_size,
            "image_count": len(rows),
            "compression": "stored_deterministic",
        },
        "upload_name_to_observation_id": {name: frame_id for name, _, _, frame_id in rows},
        "frozen_provider_configuration": FROZEN_T1_CONFIGURATION,
        "teleport_provider_terms_review_digest": terms_review[
            "teleport_provider_terms_review_digest"
        ],
        "teleport_provider_terms_review_path": terms_path.relative_to(REPO_ROOT).as_posix(),
        "hidden_images_included": False,
        "hidden_filenames_included": False,
        "authority_recorded": {
            "public_mushroom_provider_upload_authorized": False,
            "teleport_paid_service_authorized": False,
            "teleport_service_spend_cap_usd": None,
            "authorization_issued_by_agent": False,
        },
        "provider_mutations_performed": False,
        "required_external_inputs": [
            "teleport_api_credentials",
            "fresh_human_legal_review_acceptance",
            "exact_preapproved_provider_quote",
            "exact_operator_spend_cap",
            "TELEPORT_PUBLIC_DATA_UPLOAD_AUTHORIZED=true",
            "typed_provider_admission_record",
            "typed_operator_authorization_receipt",
        ],
        "post_run_obligations": [
            "retrieve_ply_and_camera_metadata_before_any_deletion",
            "hash_all_retrieved_outputs",
            "request_provider_deletion_and_record_204_only_evidence",
            "verify_deletion_after_job",
            "record_provider_zero_and_final_billing_limitations",
            "reconcile_final_billing_statement_against_cap_when_available",
        ],
        "proof_effect": "local_upload_packet_preparation_only",
        "claim_ceiling": "none",
        "timestamp": report["timestamp"],
    }
    packet["teleport_t1_upload_packet_digest"] = canonical_digest(
        packet, digest_field="teleport_t1_upload_packet_digest"
    )
    manifest_path = output_dir / "teleport_t1_upload_packet.v1.json"
    payload = (canonical_json(packet) + "\n").encode("utf-8")
    if manifest_path.exists() and manifest_path.read_bytes() != payload:
        raise SystemExit("refusing to overwrite differing packet manifest")
    manifest_path.write_bytes(payload)
    print(
        json.dumps(
            {
                "status": packet["status"],
                "zip": str(zip_path),
                "zip_digest": zip_digest,
                "image_count": len(rows),
                "packet_manifest": str(manifest_path),
                "packet_digest": packet["teleport_t1_upload_packet_digest"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
