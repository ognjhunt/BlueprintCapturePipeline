#!/usr/bin/env python3
"""Record explicit owner acceptance of one preserved, AI-rejected appearance result."""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile
from blueprint_pipeline.artifixer_completed_training_reuse import _read, _sha, PREFIX
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.task_evaluation_scene_configuration_appearance_review import (
    HUMAN_APPROVAL_SCHEMA,
    HUMAN_ACCEPTED_STATUS,
    validate_human_approval,
)
from blueprint_pipeline.task_evaluation_scene_configuration_artifixer_warm_checkpoint import (
    SCHEMA_VERSION,
)


def record_acceptance(
    *, source_launch_root, accepted_by, statement, approval_reference, known_artifacts, output_root
):
    source = Path(source_launch_root)
    zero = _read(source / "post_teardown_provider_zero_receipt.json")
    launch = _read(source / "launch_receipt.json")
    if (
        launch.get("status") not in {"blocked", "completed"}
        or zero.get("launch_id") != source.name
        or zero.get("provider_zero_verified") is not True
        or zero.get("continuing_spend_from_this_run") is not False
        or zero.get("provider_zero_receipt_digest")
        != canonical_digest(zero, digest_field="provider_zero_receipt_digest")
    ):
        raise ValueError("human_acceptance_source_not_closed")
    job = source / "allocator/scene-configuration-job"
    with zipfile.ZipFile(job / "api_pretraining_capsule.zip") as archive:
        stage = json.loads(archive.read("output/stage_production_input.v1.json"))
    if stage["configuration"]["human_authority"]["accepted_by"] != accepted_by:
        raise ValueError("human_acceptance_owner_mismatch")
    archive_path = job / "vast_provider_run/vast_provider_runtime_output.zip"
    with zipfile.ZipFile(archive_path) as archive:
        prefix = "stages/stage-1/producer/released_artifixer_runtime/artifixer_candidate_round_0/"
        checkpoint = json.loads(archive.read(PREFIX + SCHEMA_VERSION + ".json"))
        review_input = json.loads(
            archive.read(prefix + "task_evaluation_artifixer3d_dual_target_review_input.v1.json")
        )
        execution = json.loads(
            archive.read(
                prefix
                + "independent_visual_review/task_evaluation_artifixer_ai_visual_review_execution.v1.json"
            )
        )
        runtime = json.loads(archive.read(PREFIX + "runtime/runtime_result.json"))
    frames = [
        {"camera_id": r["camera_id"], "frame_sha256": r["sha256"]}
        for r in runtime["tasks"][0]["artifixer3d_review_frames"]
    ]
    if (
        checkpoint.get("checkpoint_digest")
        != canonical_digest(checkpoint, digest_field="checkpoint_digest")
        or review_input.get("receipt_digest")
        != canonical_digest(review_input, digest_field="receipt_digest")
        or runtime.get("artifixer3d_distillation_executed") is not True
    ):
        raise ValueError("human_acceptance_completed_result_invalid")
    approval = {
        "schema_version": HUMAN_APPROVAL_SCHEMA,
        "status": HUMAN_ACCEPTED_STATUS,
        "scope": "appearance_only_continuation",
        "accepted_by": accepted_by,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "statement": statement,
        "approval_reference": approval_reference,
        "known_artifacts": list(known_artifacts),
        "source_launch_id": source.name,
        "source_provider_output_sha256": _sha(archive_path),
        "source_checkpoint_digest": checkpoint["checkpoint_digest"],
        "source_post_training_binding_digest": checkpoint["binding_digest"],
        "source_review_input_digest": review_input["receipt_digest"],
        "input_digest": execution["input_digest"],
        "frames": frames,
        "source_ai_review_execution": execution,
    }
    approval["approval_digest"] = canonical_digest(approval, digest_field="approval_digest")
    validate_human_approval(approval, expected_owner=accepted_by)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "human_appearance_approval.json"
    if path.exists():
        retained = validate_human_approval(_read(path), expected_owner=accepted_by)
        ignored = {"recorded_at", "approval_digest"}
        if {k: v for k, v in retained.items() if k not in ignored} != {
            k: v for k, v in approval.items() if k not in ignored
        }:
            raise ValueError("human_acceptance_existing_approval_changed")
        return {
            "status": "already_recorded",
            "path": str(path),
            "approval_digest": retained["approval_digest"],
        }
    with path.open("x") as stream:
        stream.write(canonical_json(approval) + "\n")
    path.chmod(0o640)
    return {
        "status": "recorded",
        "path": str(path),
        "approval_digest": approval["approval_digest"],
        "frame_count": len(frames),
        "ai_decision_preserved": "rejected",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "source-launch-root",
        "accepted-by",
        "statement",
        "approval-reference",
        "output-root",
    ]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--known-artifact", action="append", required=True)
    a = p.parse_args()
    print(
        json.dumps(
            record_acceptance(
                source_launch_root=a.source_launch_root,
                accepted_by=a.accepted_by,
                statement=a.statement,
                approval_reference=a.approval_reference,
                known_artifacts=a.known_artifact,
                output_root=a.output_root,
            )
        )
    )


if __name__ == "__main__":
    main()
