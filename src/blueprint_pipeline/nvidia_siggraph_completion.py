"""Requirement-to-evidence matrix for the SIGGRAPH 2026 implementation memo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .common import read_json_any, sha256_file, utc_now_iso, write_json
from .external_tool_runtime import canonical_sha256
from .nvidia_siggraph_policy import validate_post_conference_source_review


SCHEMA_VERSION = "nvidia_siggraph_2026_completion_matrix.v1"


REQUIREMENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "phase0.contract_schemas",
        "paths": (
            "docs/schemas/external_simready_validation_request.schema.json",
            "docs/schemas/external_simready_validation_result.schema.json",
            "docs/schemas/external_simready_validation_claim_boundary.schema.json",
        ),
    },
    {
        "id": "phase0.fixture_corpus",
        "paths": (
            "tests/fixtures/nvidia_siggraph_2026/simready_fixture_corpus.json",
            "tests/fixtures/nvidia_siggraph_2026/simready_valid.usda",
            "tests/fixtures/nvidia_siggraph_2026/simready_missing_default_prim.usda",
        ),
    },
    {
        "id": "phase0.local_baseline",
        "paths": ("src/blueprint_pipeline/external_simready_validation.py",),
    },
    {
        "id": "phase0.post_conference_refresh",
        "paths": (
            "docs/schemas/nvidia_siggraph_post_conference_source_review.schema.json",
            "docs/nvidia_siggraph_post_conference_source_review.template.json",
        ),
    },
    {
        "id": "phase0.rule_promotion_calibration",
        "paths": (
            "src/blueprint_pipeline/simready_rule_calibration.py",
            "docs/schemas/simready_rule_calibration.schema.json",
            "tests/test_simready_rule_calibration.py",
        ),
    },
    {
        "id": "phase1.isolated_simready_worker",
        "paths": (
            "src/blueprint_pipeline/external_simready_validation.py",
            "scripts/run_simready_validator_worker.py",
            "scripts/setup_simready_validator_env.sh",
            "tests/test_external_simready_validation.py",
        ),
    },
    {
        "id": "phase1.immutable_conditioning_proposals",
        "paths": (
            "src/blueprint_pipeline/nvidia_asset_conditioning_review.py",
            "docs/schemas/nvidia_asset_conditioning_review.schema.json",
            "tests/test_nvidia_asset_conditioning_review.py",
        ),
    },
    {
        "id": "phase2.ovrtx_preflight",
        "paths": (
            "src/blueprint_pipeline/omniverse_library_preflight.py",
            "scripts/run_ovrtx_preflight_worker.py",
            "docs/schemas/omniverse_preflight_result.schema.json",
        ),
    },
    {
        "id": "phase2.ovphysx_preflight",
        "paths": (
            "src/blueprint_pipeline/omniverse_library_preflight.py",
            "scripts/run_ovphysx_preflight_worker.py",
            "docs/schemas/omniverse_preflight_result.schema.json",
        ),
    },
    {
        "id": "phase2.same_scene_benchmark",
        "paths": (
            "docs/schemas/omniverse_preflight_benchmark_suite.schema.json",
            "tests/test_omniverse_library_preflight.py",
        ),
    },
    {
        "id": "phase2.paid_resource_closeout",
        "paths": (
            "src/blueprint_pipeline/nvidia_experiment_resource.py",
            "docs/schemas/nvidia_experiment_resource_context.schema.json",
            "docs/schemas/nvidia_experiment_resource_closeout.schema.json",
            "tests/test_nvidia_experiment_resource.py",
        ),
    },
    {
        "id": "phase3.distinct_edge_runtime",
        "paths": (
            "src/blueprint_pipeline/cosmos3_edge_experiment.py",
            "scripts/run_cosmos3_edge_worker.py",
            "scripts/setup_cosmos3_edge_env.sh",
            "tests/test_cosmos3_edge_experiment.py",
        ),
    },
    {
        "id": "phase3.edge_qualification",
        "paths": (
            "src/blueprint_pipeline/cosmos3_edge_qualification.py",
            "docs/schemas/cosmos3_edge_qualification.schema.json",
            "tests/test_cosmos3_edge_qualification.py",
        ),
    },
    {
        "id": "existing.gsplat_conformance_only",
        "paths": (
            "src/blueprint_pipeline/gsplat_conformance.py",
            "tests/test_gsplat_conformance.py",
        ),
    },
    {
        "id": "existing.artifixer_heldout_only",
        "paths": (
            "src/blueprint_pipeline/artifixer_heldout_evaluation.py",
            "tests/test_artifixer_heldout_evaluation.py",
        ),
    },
    {
        "id": "governance.component_registry_and_stop_rules",
        "paths": (
            "src/blueprint_pipeline/nvidia_siggraph_policy.py",
            "tests/test_nvidia_siggraph_policy.py",
        ),
    },
    {
        "id": "governance.advisory_surface",
        "paths": (
            "src/blueprint_pipeline/simulation_automation.py",
            "tests/test_simulation_automation.py",
        ),
    },
    {
        "id": "documentation.runbook",
        "paths": (
            "docs/NVIDIA_SIGGRAPH_2026_IMPLEMENTATION_RUNBOOK.md",
            "docs/NVIDIA_SIGGRAPH_2026_STACK_IMPACT_2026-07-21.md",
        ),
    },
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _portable_evidence_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return f"external:{path.name}"


def build_completion_matrix(
    *,
    repository_root: str | Path,
    output_path: str | Path,
    verification_receipt_path: str | Path | None = None,
    post_conference_source_review_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for requirement in REQUIREMENTS:
        evidence = []
        missing = []
        for relative in requirement["paths"]:
            path = root / relative
            if path.is_file():
                evidence.append({"path": relative, "sha256": sha256_file(path)})
            else:
                missing.append(relative)
        if missing:
            blockers.append(f"implementation_evidence_missing:{requirement['id']}")
        rows.append(
            {
                "requirement_id": requirement["id"],
                "implementation_status": "implemented" if not missing else "missing",
                "evidence": evidence,
                "missing_paths": missing,
            }
        )

    verification: dict[str, Any] = {}
    if verification_receipt_path is not None:
        verification_path = Path(verification_receipt_path).resolve()
        verification = _mapping(read_json_any(verification_path))
        if verification.get("schema_version") != "nvidia_siggraph_2026_verification_receipt.v1":
            blockers.append("verification_receipt_schema_invalid")
        if verification.get("status") != "passed" or verification.get("exit_code") != 0:
            blockers.append("verification_receipt_not_passed")
        verification = {
            **verification,
            "path": _portable_evidence_path(verification_path, root),
            "sha256": sha256_file(verification_path) if verification_path.is_file() else None,
        }
    else:
        blockers.append("verification_receipt_missing")

    source_review: dict[str, Any] = {}
    source_review_status = "pending_until_2026-07-24_or_later"
    if post_conference_source_review_path is not None:
        source_path = Path(post_conference_source_review_path).resolve()
        source_review = _mapping(read_json_any(source_path))
        validation = validate_post_conference_source_review(source_review, as_of_date="2026-07-24")
        review_blockers = list(validation["blockers"])
        if review_blockers:
            source_review_status = "blocked"
            blockers.extend(f"post_conference_source_review:{value}" for value in review_blockers)
        else:
            source_review_status = "completed"
        source_review = {
            **source_review,
            "path": _portable_evidence_path(source_path, root),
            "sha256": sha256_file(source_path) if source_path.is_file() else None,
        }

    implementation_complete = not any(
        value.startswith(("implementation_evidence_missing", "verification_receipt"))
        for value in blockers
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "memo_path": "docs/NVIDIA_SIGGRAPH_2026_STACK_IMPACT_2026-07-21.md",
        "implementation_complete": implementation_complete,
        "status": (
            "implementation_complete_external_qualification_pending"
            if implementation_complete and source_review_status != "completed"
            else "complete_with_source_review"
            if implementation_complete
            else "implementation_incomplete"
        ),
        "requirements": rows,
        "verification_receipt": verification or None,
        "post_conference_source_review": {
            "status": source_review_status,
            "evidence": source_review or None,
        },
        "external_qualification_state": {
            "official_simready_execution": "unproven",
            "linux_rtx_ovrtx_execution": "unproven",
            "official_ovphysx_execution": "unproven",
            "same_scene_isaac_comparison": "unproven",
            "cosmos3_edge_checkpoint_execution": "unproven",
            "cosmos3_edge_rank_fidelity": "unproven",
            "provider_allocation_or_spend": "not_performed_by_implementation",
        },
        "blockers": list(dict.fromkeys(blockers)),
        "claim_boundary": {
            "repository_implementation_completeness_only": True,
            "external_runtime_qualification_proven": False,
            "semantic_or_ranking_success_proven": False,
            "provider_teardown_proven_without_attempt": False,
            "production_promotion_allowed": False,
        },
    }
    payload["matrix_fingerprint"] = canonical_sha256(payload)
    write_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build SIGGRAPH 2026 implementation matrix")
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verification-receipt")
    parser.add_argument("--post-conference-source-review")
    args = parser.parse_args(argv)
    result = build_completion_matrix(
        repository_root=args.repository_root,
        output_path=args.output,
        verification_receipt_path=args.verification_receipt,
        post_conference_source_review_path=args.post_conference_source_review,
    )
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["implementation_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
