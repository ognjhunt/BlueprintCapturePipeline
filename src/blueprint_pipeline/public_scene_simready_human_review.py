"""Bind an explicit human visual decision to exact SimReady review artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import utc_now_iso
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "adp009b_simready_human_visual_review.v1"


class SimReadyHumanReviewError(ValueError):
    """The decision or artifact identity is not safe to bind."""


def _read(path: Path, *, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SimReadyHumanReviewError(error) from exc
    if not isinstance(value, dict):
        raise SimReadyHumanReviewError(error)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _under(path: Path, root: Path, *, error: str) -> Path:
    path = path.expanduser().resolve()
    root = root.expanduser().resolve()
    if path != root and root not in path.parents:
        raise SimReadyHumanReviewError(error)
    return path


def _verified_receipt(
    path: Path, *, expected_digest: object, error: str
) -> dict[str, Any]:
    value = _read(path, error=error)
    supplied = value.get("receipt_digest")
    if supplied != canonical_digest(value, digest_field="receipt_digest"):
        raise SimReadyHumanReviewError(f"{error}_digest_invalid")
    if supplied != expected_digest:
        raise SimReadyHumanReviewError(f"{error}_identity_mismatch")
    return value


def materialize_human_review(
    *,
    request_path: str | Path,
    repo_root: str | Path,
    evidence_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    evidence = Path(evidence_root).expanduser().resolve()
    request_file = _under(Path(request_path), repo, error="human_review_request_outside_repo")
    output = _under(Path(output_path), repo, error="human_review_output_outside_repo")
    request = _read(request_file, error="human_review_request_invalid")
    if request.get("schema_version") != "adp009b_simready_human_visual_review_request.v1":
        raise SimReadyHumanReviewError("human_review_request_schema_invalid")
    if {"admitted", "qualified", "dynamic_contact_proven"}.intersection(request):
        raise SimReadyHumanReviewError("human_review_cannot_assert_technical_admission")
    if request.get("decision") != "approve_for_native_validation":
        raise SimReadyHumanReviewError("human_review_explicit_approval_required")
    if request.get("reviewer_role") != "project_owner":
        raise SimReadyHumanReviewError("human_review_project_owner_required")
    approval_statement = str(request.get("approval_statement") or "").strip()
    if not approval_statement:
        raise SimReadyHumanReviewError("human_review_approval_statement_missing")
    artifacts = request.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise SimReadyHumanReviewError("human_review_artifacts_missing")
    replacement_path = _under(
        repo / str(artifacts.get("replacement_receipt_path")),
        repo,
        error="replacement_receipt_outside_repo",
    )
    replacement = _verified_receipt(
        replacement_path,
        expected_digest=artifacts.get("replacement_receipt_digest"),
        error="replacement_receipt",
    )
    visual_path = _under(
        evidence / str(artifacts.get("visual_review_relative_path")),
        evidence,
        error="visual_review_outside_evidence",
    )
    visual = _verified_receipt(
        visual_path,
        expected_digest=artifacts.get("visual_review_receipt_digest"),
        error="visual_review_receipt",
    )
    match_path = _under(
        evidence / str(artifacts.get("match_review_relative_path")),
        evidence,
        error="match_review_outside_evidence",
    )
    match = _verified_receipt(
        match_path,
        expected_digest=artifacts.get("match_review_receipt_digest"),
        error="match_review_receipt",
    )
    aggregate = match.get("aggregate")
    if (
        replacement.get("status") != "composed_static_candidate"
        or visual.get("status") != "rendered_visual_review_candidate"
        or match.get("status") != "diagnosed_match_candidate"
        or not isinstance(aggregate, Mapping)
        or aggregate.get("projected_scale_and_pose_gate_passed") is not True
        or aggregate.get("colour_appearance_gate_passed") is not True
        or visual.get("replacement_receipt_digest") != replacement.get("receipt_digest")
        or match.get("visual_review_receipt_digest") != visual.get("receipt_digest")
    ):
        raise SimReadyHumanReviewError("human_review_artifact_chain_not_acceptable")
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "human_accepted_for_native_validation",
        "decision": request["decision"],
        "reviewer_role": request["reviewer_role"],
        "approval_statement": approval_statement,
        "approval_scope": "visual_identity_scale_pose_sufficient_to_continue_native_validation",
        "artifact_chain": {
            "replacement_receipt_digest": replacement["receipt_digest"],
            "visual_review_receipt_digest": visual["receipt_digest"],
            "match_review_receipt_digest": match["receipt_digest"],
            "match_review_file_sha256": _sha256(match_path),
            "camera_count": aggregate["camera_count"],
            "median_silhouette_iou": aggregate["median_silhouette_iou"],
            "median_delta_e76": aggregate["median_delta_e76"],
        },
        "technical_admission": False,
        "dynamic_contact_proven": False,
        "native_ovrtx_proven": False,
        "blockers": [
            "native_ovrtx_material_render_missing",
            "native_ovphysx_drop_contact_settle_missing",
            "nvidia_agent_v2_static_validation_missing",
        ],
        "claim_ceiling": "human_visual_acceptance_of_exact_static_candidate",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = materialize_human_review(
        request_path=args.request,
        repo_root=args.repo_root,
        evidence_root=args.evidence_root,
        output_path=args.output,
    )
    print(json.dumps({"status": receipt["status"], "receipt_digest": receipt["receipt_digest"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
