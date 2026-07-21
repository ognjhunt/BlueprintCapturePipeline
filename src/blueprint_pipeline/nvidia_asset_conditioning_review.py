"""Proposal-only evidence for deferred NVIDIA asset-conditioning workflows.

This contract covers CAD-to-SimReady, Content Agents, and SimReady Blender
without making any of them a capture reconstruction path or an authoritative
source of material, mass, friction, semantic, or collision truth.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_json_any, sha256_file, utc_now_iso, write_json
from .external_tool_runtime import canonical_sha256
from .local_capture import resolve_local_capture_context
from .nvidia_siggraph_policy import (
    evaluate_component_activation,
    validate_post_conference_source_review_file,
)


SCHEMA_VERSION = "nvidia_asset_conditioning_review.v1"
COMPONENTS = {"cad_to_simready_skill", "content_agents", "simready_blender"}
CAD_STAGES = (
    "import",
    "minimum_usd_validation",
    "material_proposal",
    "physics_proposal",
    "conformance",
    "report",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _within(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def build_asset_conditioning_review(
    *,
    capture_root: str | Path,
    component: str,
    buyer_need_id: str,
    original_asset_path: str | Path,
    candidate_output_paths: Sequence[str | Path],
    component_version: str,
    source_revision: str,
    license_id: str,
    license_compatible: bool,
    output_path: str | Path,
    staged_evidence: Mapping[str, str | Path] | None = None,
    human_approval_path: str | Path | None = None,
    critical_path: bool = False,
    physical_metadata_treated_as_proposal: bool = True,
    as_of_date: str | None = None,
    post_conference_source_review_path: str | Path | None = None,
) -> dict[str, Any]:
    if component not in COMPONENTS:
        raise ValueError(f"unsupported deferred NVIDIA asset component: {component}")
    context = resolve_local_capture_context(capture_root)
    pipeline_root = context.pipeline_root.resolve()
    raw_root = context.capture_root.resolve() / "raw"
    original = Path(original_asset_path).resolve()
    outputs = [Path(path).resolve() for path in candidate_output_paths]
    blockers: list[str] = []
    buyer_asset = bool(
        original.is_file() and _within(original, pipeline_root) and not _within(original, raw_root)
    )
    if not buyer_asset:
        blockers.append("asset_conditioning_requires_pipeline_staged_buyer_asset")
    if not buyer_need_id:
        blockers.append("asset_conditioning_buyer_need_id_missing")
    if not component_version or not source_revision:
        blockers.append("asset_conditioning_component_identity_not_pinned")
    if not license_id or not license_compatible:
        blockers.append("asset_conditioning_license_not_verified_compatible")
    original_before = sha256_file(original) if original.is_file() else None
    output_rows: list[dict[str, Any]] = []
    for index, path in enumerate(outputs):
        valid = path.is_file() and _within(path, pipeline_root) and not _within(path, raw_root)
        if not valid:
            blockers.append(f"asset_conditioning_candidate_missing_or_outside_pipeline:{index}")
        output_rows.append(
            {
                "path": str(path),
                "sha256": sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else 0,
                "pipeline_derived_support_asset": valid,
            }
        )
    if not output_rows:
        blockers.append("asset_conditioning_candidate_outputs_missing")
    stage_rows: list[dict[str, Any]] = []
    raw_stages = dict(staged_evidence or {})
    if component == "cad_to_simready_skill":
        for stage in CAD_STAGES:
            path = Path(raw_stages.get(stage, "")).resolve()
            valid = path.is_file() and _within(path, pipeline_root)
            if not valid:
                blockers.append(f"cad_to_simready_stage_evidence_missing:{stage}")
            stage_rows.append(
                {
                    "stage": stage,
                    "path": str(path),
                    "sha256": sha256_file(path) if path.is_file() else None,
                    "valid": valid,
                }
            )
    approval_path = Path(human_approval_path).resolve() if human_approval_path else None
    approval = (
        _mapping(read_json_any(approval_path)) if approval_path and approval_path.is_file() else {}
    )
    approval_ok = bool(
        approval.get("schema_version") == "nvidia_asset_conditioning_human_approval.v1"
        and approval.get("status") == "approved"
        and _string(approval.get("approval_id"))
        and _string(approval.get("reviewer_id"))
        and approval.get("buyer_need_id") == buyer_need_id
        and approval.get("component") == component
    )
    if component in {"content_agents", "simready_blender"} and not approval_ok:
        blockers.append("asset_conditioning_human_approval_missing_or_invalid")
    effective_date = as_of_date or date.today().isoformat()
    activation_evidence: dict[str, Any] = {"explicit_opt_in": True}
    if post_conference_source_review_path is not None:
        activation_evidence["post_conference_source_review"] = (
            validate_post_conference_source_review_file(
                post_conference_source_review_path,
                as_of_date=effective_date,
            )
        )
    if component == "cad_to_simready_skill":
        activation_evidence.update(
            {
                "buyer_supplied_asset": buyer_asset,
                "staged_validation_evidence": bool(stage_rows)
                and all(row["valid"] for row in stage_rows),
                "capture_reconstruction_primary_path": False,
            }
        )
    elif component == "content_agents":
        activation_evidence.update(
            {
                "specific_buyer_asset_conditioning_need": bool(buyer_need_id),
                "immutable_before_after_evidence": bool(original_before and output_rows),
                "human_approval": approval_ok,
                "privacy_safe_inputs_only": buyer_asset,
                "physical_metadata_treated_as_proposal": physical_metadata_treated_as_proposal,
            }
        )
    else:
        activation_evidence["critical_path"] = critical_path
    activation = evaluate_component_activation(
        component,
        evidence=activation_evidence,
        as_of_date=effective_date,
    )
    blockers.extend(activation["blockers"])
    original_after = sha256_file(original) if original.is_file() else None
    if original_before != original_after:
        blockers.append("asset_conditioning_original_asset_modified")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "accepted_advisory_proposal" if not blockers else "blocked",
        "component": component,
        "buyer_need_id": buyer_need_id,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "component_identity": {
            "version": component_version,
            "source_revision": source_revision,
            "license_id": license_id,
            "license_compatible": license_compatible,
        },
        "original_asset": {
            "path": str(original),
            "sha256_before": original_before,
            "sha256_after": original_after,
            "modified": original_before != original_after,
            "buyer_supplied_pipeline_staged": buyer_asset,
        },
        "candidate_outputs": output_rows,
        "staged_evidence": stage_rows,
        "human_approval": {
            "path": str(approval_path) if approval_path else None,
            "sha256": sha256_file(approval_path)
            if approval_path and approval_path.is_file()
            else None,
            "verified": approval_ok,
            "approval_id": approval.get("approval_id"),
        },
        "activation_evaluation": activation,
        "blockers": list(dict.fromkeys(blockers)),
        "claim_boundary": {
            "proposal_only": True,
            "capture_reconstruction_primary_path": False,
            "raw_capture_modified": False,
            "material_mass_friction_semantics_or_colliders_authoritative": False,
            "headless_pipeline_or_simulator_proof": False,
            "critical_path_allowed": False,
            "human_review_does_not_prove_physical_correctness": True,
        },
    }
    payload["review_fingerprint"] = canonical_sha256(payload)
    write_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a deferred NVIDIA asset proposal review")
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--component", choices=sorted(COMPONENTS), required=True)
    parser.add_argument("--buyer-need-id", required=True)
    parser.add_argument("--original-asset", required=True)
    parser.add_argument("--candidate-output", action="append", required=True)
    parser.add_argument("--component-version", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--license-compatible", action="store_true")
    parser.add_argument("--staged-evidence-json", default=None)
    parser.add_argument("--human-approval", default=None)
    parser.add_argument("--critical-path", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--post-conference-source-review", default=None)
    args = parser.parse_args(argv)
    stages = (
        _mapping(read_json_any(Path(args.staged_evidence_json)))
        if args.staged_evidence_json
        else {}
    )
    result = build_asset_conditioning_review(
        capture_root=args.capture_root,
        component=args.component,
        buyer_need_id=args.buyer_need_id,
        original_asset_path=args.original_asset,
        candidate_output_paths=args.candidate_output,
        component_version=args.component_version,
        source_revision=args.source_revision,
        license_id=args.license_id,
        license_compatible=args.license_compatible,
        output_path=args.output,
        staged_evidence=stages,
        human_approval_path=args.human_approval,
        critical_path=args.critical_path,
        as_of_date=args.as_of_date,
        post_conference_source_review_path=args.post_conference_source_review,
    )
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "accepted_advisory_proposal" else 2


if __name__ == "__main__":
    raise SystemExit(main())
