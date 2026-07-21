#!/usr/bin/env python3
"""Refresh worktree artifact digests in the fail-closed SC3 quality ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json"

LAUNCH_SCOPE_POLICY = {
    "profile_id": "sim_policy_comparison_with_live_buyer_delivery.v1",
    "enabled_scope_labels": ["BASE", "SIM", "SC3", "LIVE"],
    "enabled_paid_gap_ids": ["EVID-10", "EVID-11"],
    "disabled_unmarketed_features": [
        "ptdp",
        "payments",
        "payouts",
        "unsupported_devices",
        "physical_robot",
    ],
    "conditional_nonblocking_gap_ids": ["SC3-22", "EVID-01"],
    "correlation_claim_mode": "correlation_not_measured",
}

EVIDENCE_OVERRIDES: dict[str, list[tuple[str, str]]] = {
    "SC3-02-AC-01": [
        ("src/blueprint_pipeline/closed_loop_consistency_scoring.py", "implementation"),
        ("src/blueprint_pipeline/generated_episode_authority.py", "implementation"),
        ("src/blueprint_pipeline/oscar_isaac_closed_loop_eval.py", "implementation"),
        ("tests/test_oscar_isaac_closed_loop_eval.py", "test_contract"),
    ],
    "SC3-16-AC-01": [
        ("src/blueprint_pipeline/evaluator_evidence_profiles.py", "implementation"),
        ("src/blueprint_pipeline/policy_evaluation_contracts.py", "implementation"),
        ("src/blueprint_pipeline/decision_grade_ranking.py", "implementation"),
        ("tests/test_policy_evaluation_contracts.py", "test_contract"),
    ],
    "SC3-22-AC-01": [
        ("docs/external_anchor_candidate_registry_2026-07-20.json", "candidate_registry"),
        ("src/blueprint_pipeline/external_study_protocols.py", "implementation"),
        ("src/blueprint_pipeline/sc3_fidelity_contracts.py", "implementation"),
        ("tests/test_external_anchor_candidate_registry.py", "test_contract"),
        ("tests/test_sc3_fidelity_contracts.py", "test_contract"),
    ],
    "EVID-01-AC-01": [
        ("docs/external_anchor_candidate_registry_2026-07-20.json", "candidate_registry"),
        ("src/blueprint_pipeline/decision_grade_ranking.py", "implementation"),
        ("tests/test_external_anchor_candidate_registry.py", "test_contract"),
        ("tests/test_policy_evaluation_contracts.py", "test_contract"),
    ],
}

EVIDENCE_EXTENSIONS: dict[str, list[tuple[str, str]]] = {
    "DATA-03-AC-01": [
        ("src/blueprint_pipeline/site_reference_database.py", "implementation"),
        ("tests/test_site_reference_database_contract.py", "test_contract"),
    ],
    "DATA-04-AC-01": [
        ("src/blueprint_pipeline/site_reference_database.py", "implementation"),
        ("tests/test_site_reference_database_contract.py", "test_contract"),
    ],
    "DATA-13-AC-01": [
        ("src/blueprint_pipeline/site_reference_database.py", "implementation"),
        ("tests/test_site_reference_database_contract.py", "test_contract"),
    ],
    "SC3-08-AC-01": [
        ("src/blueprint_pipeline/evaluator_evidence_profiles.py", "implementation"),
        ("src/blueprint_pipeline/oscar_action_control_contracts.py", "implementation"),
        ("src/blueprint_pipeline/policy_evaluation_contracts.py", "implementation"),
        ("tests/test_oscar_action_control_contracts.py", "test_contract"),
        ("tests/test_policy_evaluation_contracts.py", "test_contract"),
    ],
    "SC3-12-AC-01": [
        ("src/blueprint_pipeline/evaluator_evidence_profiles.py", "implementation"),
        ("src/blueprint_pipeline/decision_grade_ranking.py", "implementation"),
        ("tests/test_policy_evaluation_contracts.py", "test_contract"),
    ],
    "SC3-20-AC-01": [
        ("src/blueprint_pipeline/decision_grade_ranking.py", "implementation"),
        ("tests/test_policy_evaluation_contracts.py", "test_contract"),
    ],
    "SC3-21-AC-01": [
        ("src/blueprint_pipeline/evaluator_evidence_profiles.py", "implementation"),
    ],
    "REL-14-AC-01": [
        ("src/blueprint_pipeline/evaluator_evidence_profiles.py", "implementation"),
    ],
    "SC3-10-AC-01": [],
    "SC3-11-AC-01": [],
    "EVID-11-AC-01": [
        ("src/blueprint_pipeline/oscar_runtime_asset_contract.py", "implementation"),
        ("docs/external_anchor_candidate_registry_2026-07-20.json", "candidate_registry"),
        ("tests/test_oscar_runtime_asset_contract.py", "test_contract"),
    ],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _text_sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _mapping_sha256(ledger: dict[str, Any]) -> str:
    mapping = []
    for gap in ledger.get("gaps", []):
        for criterion in gap.get("criteria", []):
            command = criterion.get("command_result", {})
            mapping.append(
                {
                    "criterion_id": criterion.get("criterion_id"),
                    "acceptance_text_sha256": criterion.get("acceptance_text_sha256"),
                    "evidence_artifacts": [
                        {
                            field: artifact.get(field)
                            for field in (
                                "artifact_id",
                                "path",
                                "role",
                                "supports_remediation",
                                "supports_closure",
                            )
                        }
                        for artifact in criterion.get("evidence_artifacts", [])
                    ],
                    "command": {
                        field: command.get(field) for field in ("applicable", "command")
                    },
                }
            )
    return _text_sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )


def _launch_scope(gap_id: str, scopes: list[str]) -> dict[str, Any]:
    enabled_labels = set(LAUNCH_SCOPE_POLICY["enabled_scope_labels"])
    basis = [f"enabled_scope:{scope}" for scope in scopes if scope in enabled_labels]
    if gap_id in set(LAUNCH_SCOPE_POLICY["enabled_paid_gap_ids"]):
        basis.append("enabled_feature:buyer_delivery_and_rights")
    if gap_id in set(LAUNCH_SCOPE_POLICY["conditional_nonblocking_gap_ids"]):
        return {
            "scoped": True,
            "blocking": False,
            "basis": basis,
            "nonblocking_reason": "external_correlation_claim_not_enabled",
        }
    if basis:
        return {"scoped": True, "blocking": True, "basis": basis, "nonblocking_reason": None}
    reason = {
        "EVID-14": "physical_robot_claim_not_enabled",
        "EVID-09": "payments_and_payouts_not_enabled",
        "EVID-12": "unsupported_device_lanes_not_marketed",
    }.get(gap_id, "ptdp_or_paid_feature_not_enabled")
    return {"scoped": False, "blocking": False, "basis": [], "nonblocking_reason": reason}


def _artifact(criterion: dict[str, Any], path: str, role: str, index: int) -> dict[str, Any]:
    return {
        "artifact_id": f"{criterion['criterion_id']}-EV-{index:02d}",
        "path": path,
        "sha256": _sha256(ROOT / path),
        "role": role,
        "authoritative": True,
        "authority": "repository_worktree_digest",
        "supports_remediation": True,
        "supports_closure": False,
        "generated_at": criterion["generated_at"],
        "freshness_evaluated_at": criterion["freshness"]["evaluated_at"],
        "fresh_until": criterion["freshness"]["fresh_until"],
        "freshness_status": "current_unbound",
        "commit": None,
        "release_id": None,
    }


def _baseline_criteria() -> dict[str, dict[str, Any]]:
    completed = subprocess.run(
        ["git", "show", "HEAD:docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    ledger = json.loads(completed.stdout)
    return {
        str(criterion.get("criterion_id") or ""): criterion
        for gap in ledger.get("gaps", [])
        for criterion in gap.get("criteria", [])
    }


def _replace_or_extend_evidence(
    criterion: dict[str, Any], baseline: dict[str, Any] | None
) -> int:
    criterion_id = str(criterion.get("criterion_id") or "")
    rows = EVIDENCE_OVERRIDES.get(criterion_id)
    if rows is None:
        if baseline is not None:
            criterion["evidence_artifacts"] = [
                dict(row) for row in baseline.get("evidence_artifacts", [])
            ]
            criterion["command_result"] = dict(baseline.get("command_result", {}))
        if criterion_id not in EVIDENCE_EXTENSIONS:
            criterion["acceptance_check"]["evidence_artifact_ids"] = [
                row["artifact_id"] for row in criterion["evidence_artifacts"]
            ]
            return 0
        existing = [
            (str(row.get("path") or ""), str(row.get("role") or ""))
            for row in criterion.get("evidence_artifacts", [])
        ]
        rows = existing + [
            row for row in EVIDENCE_EXTENSIONS.get(criterion_id, []) if row not in existing
        ]
    before = [(row.get("path"), row.get("role")) for row in criterion.get("evidence_artifacts", [])]
    prior_by_path = {
        str(row.get("path") or ""): dict(row)
        for row in criterion.get("evidence_artifacts", [])
    }
    artifacts = []
    for index, (path, role) in enumerate(rows, start=1):
        artifact = prior_by_path.get(path) or _artifact(criterion, path, role, index)
        artifact["artifact_id"] = f"{criterion_id}-EV-{index:02d}"
        artifacts.append(artifact)
    criterion["evidence_artifacts"] = artifacts
    test_paths = sorted(path for path, role in rows if role == "test_contract")
    command = criterion["command_result"]
    command["applicable"] = bool(test_paths)
    marker = " -m ''" if "tests/test_oscar_isaac_closed_loop_eval.py" in test_paths else ""
    command["command"] = (
        f"python -m pytest -q{marker} " + " ".join(test_paths) if test_paths else None
    )
    if not test_paths:
        command["status"] = "not_applicable"
    elif command.get("status") == "not_applicable":
        command["status"] = "not_recorded"
    criterion["acceptance_check"]["evidence_artifact_ids"] = [
        row["artifact_id"] for row in criterion["evidence_artifacts"]
    ]
    return int(before != rows)


def rebind(ledger: dict[str, Any]) -> int:
    changed = 0
    baseline_criteria = _baseline_criteria()
    ledger["schema_version"] = "blueprint.public_launch_sc3_quality_gap_ledger.v3"
    ledger["launch_scope_policy"] = LAUNCH_SCOPE_POLICY
    for gap in ledger.get("gaps", []):
        launch_scope = _launch_scope(str(gap.get("id") or ""), list(gap.get("scopes") or []))
        gap["launch_scope"] = launch_scope
        for criterion in gap.get("criteria", []):
            criterion["launch_scope"] = launch_scope
            changed += _replace_or_extend_evidence(
                criterion, baseline_criteria.get(str(criterion.get("criterion_id") or ""))
            )
            for artifact in criterion.get("evidence_artifacts", []):
                relative = Path(str(artifact.get("path") or ""))
                candidate = (ROOT / relative).resolve()
                try:
                    candidate.relative_to(ROOT)
                except ValueError as exc:
                    raise ValueError(f"artifact_path_outside_repository:{relative}") from exc
                if not candidate.is_file():
                    raise FileNotFoundError(f"artifact_missing:{relative}")
                digest = _sha256(candidate)
                if artifact.get("sha256") != digest:
                    artifact["sha256"] = digest
                    changed += 1
            remediating = any(
                artifact.get("supports_remediation") is True
                for artifact in criterion.get("evidence_artifacts", [])
            )
            criterion["derived_status"] = "partial" if remediating else "open"
            criterion["freshness"]["status"] = (
                "current_unbound" if remediating else "missing_closure_evidence"
            )
        gap["derived_status"] = "partial" if any(
            criterion["derived_status"] == "partial" for criterion in gap["criteria"]
        ) else "open"
        gap["status"] = gap["derived_status"]
    ledger["evidence_mapping_sha256"] = _mapping_sha256(ledger)
    gap_status_counts = Counter(gap["derived_status"] for gap in ledger["gaps"])
    ledger["status_counts"] = {
        "open": gap_status_counts["open"],
        "partial": gap_status_counts["partial"],
        "closed": gap_status_counts["closed"],
        "reopened": gap_status_counts["reopened"],
        "total": len(ledger["gaps"]),
    }
    ledger["criteria_counts"] = dict(ledger["status_counts"])
    scoped = [gap for gap in ledger["gaps"] if gap["launch_scope"]["scoped"]]
    blocking = [gap for gap in scoped if gap["launch_scope"]["blocking"]]
    status_counts = Counter(gap["derived_status"] for gap in blocking)
    ledger["launch_scope_counts"] = {
        "scoped": len(scoped),
        "blocking": len(blocking),
        "nonblocking": len(ledger["gaps"]) - len(blocking),
        "blocking_status_counts": {
            "open": status_counts["open"],
            "partial": status_counts["partial"],
            "closed": status_counts["closed"],
            "reopened": status_counts["reopened"],
        },
    }
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    ledger_path = args.ledger.expanduser().resolve()
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    changed = rebind(ledger)
    ledger_path.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ledger": str(ledger_path), "digests_rebound": changed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
