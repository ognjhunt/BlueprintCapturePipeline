"""Calibrate advisory SimReady findings against frozen expert-reviewed cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import read_json_any, sha256_file, utc_now_iso, write_json
from .external_tool_runtime import PUBLIC_CLAIM_UPGRADE_KEY, canonical_sha256


SCHEMA_VERSION = "simready_rule_calibration.v1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _within(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except OSError:
        return False


def build_simready_rule_calibration(
    *,
    manifest_path: str | Path,
    evidence_root: str | Path,
    output_path: str | Path,
    minimum_cases_per_rule: int = 2,
    authorize_rule_ids: Sequence[str] = (),
    human_promotion_approval_id: str | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    evidence_dir = Path(evidence_root).resolve()
    manifest = _mapping(read_json_any(manifest_file))
    blockers: list[str] = []
    if manifest.get("schema_version") != "simready_rule_calibration_manifest.v1":
        blockers.append("simready_calibration_manifest_schema_invalid")
    if manifest.get("frozen") is not True:
        blockers.append("simready_calibration_manifest_not_frozen")
    if minimum_cases_per_rule < 2:
        blockers.append("simready_calibration_requires_at_least_two_cases_per_rule")
    raw_cases = manifest.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raw_cases = []
        blockers.append("simready_calibration_cases_missing")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_valid_cases = 0
    expected_invalid_cases = 0
    identity_fingerprints: set[str] = set()
    for index, value in enumerate(raw_cases):
        case = _mapping(value)
        case_id = _string(case.get("case_id"))
        if not case_id or case_id in seen:
            blockers.append(f"simready_calibration_case_id_missing_or_duplicate:{index}")
        seen.add(case_id)
        result_path = Path(_string(case.get("result_path")))
        result_path = (
            result_path if result_path.is_absolute() else manifest_file.parent / result_path
        )
        if not result_path.is_file() or not _within(result_path, evidence_dir):
            blockers.append(
                f"simready_calibration_result_missing_or_outside_evidence_root:{case_id or index}"
            )
            result: dict[str, Any] = {}
        else:
            result = _mapping(read_json_any(result_path))
        if result.get("schema_version") != "external_simready_validation_result.v1":
            blockers.append(f"simready_calibration_result_schema_invalid:{case_id or index}")
        if _mapping(result.get("repeatability")).get("stable_normalized_results") is not True:
            blockers.append(f"simready_calibration_result_not_repeatable:{case_id or index}")
        review = _mapping(case.get("expert_review"))
        review_ok = bool(
            review.get("status") == "approved"
            and _string(review.get("reviewer_id"))
            and _string(review.get("reviewed_at"))
        )
        if not review_ok:
            blockers.append(f"simready_calibration_expert_review_missing:{case_id or index}")
        expected = {
            _string(item) for item in review.get("expected_error_rule_ids", []) if _string(item)
        }
        observed = {
            _string(item.get("rule_id"))
            for item in result.get("normalized_findings", [])
            if isinstance(item, Mapping)
            and _string(item.get("severity")).lower() == "error"
            and _string(item.get("rule_id"))
        }
        expected_status = _string(case.get("expected_validation_status"))
        if expected_status == "passed_advisory":
            expected_valid_cases += 1
        elif expected_status == "validation_failed":
            expected_invalid_cases += 1
        else:
            blockers.append(f"simready_calibration_expected_status_invalid:{case_id or index}")
        if result.get("status") != expected_status:
            blockers.append(f"simready_calibration_case_status_mismatch:{case_id or index}")
        identity = {
            "requested_identity": result.get("requested_identity"),
            "reported_identity": result.get("reported_identity"),
        }
        identity_fingerprints.add(canonical_sha256(identity))
        rows.append(
            {
                "case_id": case_id or f"case_{index}",
                "result_path": str(result_path.resolve()),
                "result_sha256": sha256_file(result_path) if result_path.is_file() else None,
                "expected_validation_status": expected_status or None,
                "observed_validation_status": result.get("status"),
                "expected_error_rule_ids": sorted(expected),
                "observed_error_rule_ids": sorted(observed),
                "expert_review_approved": review_ok,
            }
        )
    if expected_valid_cases == 0 or expected_invalid_cases == 0:
        blockers.append("simready_calibration_requires_valid_and_invalid_cases")
    if len(identity_fingerprints) > 1:
        blockers.append("simready_calibration_validator_or_profile_identity_mismatch")
    all_rules = sorted(
        {
            rule
            for row in rows
            for rule in (*row["expected_error_rule_ids"], *row["observed_error_rule_ids"])
        }
    )
    rule_rows: list[dict[str, Any]] = []
    eligible: set[str] = set()
    for rule_id in all_rules:
        tp = fp = fn = tn = 0
        for row in rows:
            expected = rule_id in row["expected_error_rule_ids"]
            observed = rule_id in row["observed_error_rule_ids"]
            if expected and observed:
                tp += 1
            elif observed:
                fp += 1
            elif expected:
                fn += 1
            else:
                tn += 1
        eligible_for_promotion = bool(
            len(rows) >= minimum_cases_per_rule
            and fp == 0
            and fn == 0
            and all(row["expert_review_approved"] for row in rows)
            and not blockers
        )
        if eligible_for_promotion:
            eligible.add(rule_id)
        rule_rows.append(
            {
                "rule_id": rule_id,
                "case_count": len(rows),
                "true_positive_count": tp,
                "false_positive_count": fp,
                "false_negative_count": fn,
                "true_negative_count": tn,
                "eligible_for_blocking_promotion": eligible_for_promotion,
            }
        )
    requested = {_string(value) for value in authorize_rule_ids if _string(value)}
    if requested and not human_promotion_approval_id:
        blockers.append("simready_rule_promotion_human_approval_id_required")
    unauthorized = sorted(requested - eligible)
    if unauthorized:
        blockers.extend(
            f"simready_rule_not_calibrated_for_promotion:{rule}" for rule in unauthorized
        )
    authorized = sorted(requested & eligible) if not blockers else []
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if not blockers else "blocked",
        "manifest_path": str(manifest_file),
        "manifest_sha256": sha256_file(manifest_file) if manifest_file.is_file() else None,
        "validator_profile_identity_consistent": len(identity_fingerprints) == 1,
        "case_count": len(rows),
        "expected_valid_case_count": expected_valid_cases,
        "expected_invalid_case_count": expected_invalid_cases,
        "cases": rows,
        "rules": rule_rows,
        "eligible_rule_ids": sorted(eligible),
        "authorized_blocking_rule_ids": authorized,
        "human_promotion_approval_id": human_promotion_approval_id,
        "blockers": list(dict.fromkeys(blockers)),
        "claim_boundary": {
            "uncalibrated_rules_advisory_only": True,
            "authorized_rules_may_block_cpu_pre_gpu_gate": bool(authorized),
            "validator_pass_is_simulator_or_task_proof": False,
            PUBLIC_CLAIM_UPGRADE_KEY: False,
        },
    }
    payload["calibration_fingerprint"] = canonical_sha256(payload)
    write_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate SimReady rules against expert labels")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-cases-per-rule", type=int, default=2)
    parser.add_argument("--authorize-rule", action="append", default=[])
    parser.add_argument("--human-promotion-approval-id", default=None)
    args = parser.parse_args(argv)
    result = build_simready_rule_calibration(
        manifest_path=args.manifest,
        evidence_root=args.evidence_root,
        output_path=args.output,
        minimum_cases_per_rule=args.minimum_cases_per_rule,
        authorize_rule_ids=args.authorize_rule,
        human_promotion_approval_id=args.human_promotion_approval_id,
    )
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
