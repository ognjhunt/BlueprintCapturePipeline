from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs" / "PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md"
LEDGER = ROOT / "docs" / "public_launch_sc3_quality_gap_ledger_2026-07-09.json"
STATUS = ROOT / "docs" / "PUBLIC_LAUNCH_SC3_REMEDIATION_STATUS_2026-07-09.md"
P2_04_EVIDENCE_PATH = "docs/specs/launch-audit-2026-07-02/README.md"
ID_PATTERN = re.compile(r"^(?:REL|DATA|SC3|RUN|P2|EVID)-\d{2}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
# 2026-07-28: DATA-06-AC-01-EV-01 implementation evidence followed the module
# rename src/blueprint_pipeline/qualification.py -> site_package_orchestrator.py
# (same orchestration spine; blueprint_pipeline.qualification stays a
# deprecated alias pinned by tests/test_qualification_alias_contract.py).
APPROVED_CRITERION_EVIDENCE_MAP_SHA256 = (
    "sha256:5edb2016c1c26f094901b6a1abd87d1e181b462a23040e74a04c69a1b0f42ad7"
)
ALLOWED_STATUSES = {"open", "partial", "closed", "reopened"}
ALLOWED_SCOPES = {"BASE", "SIM", "PTDP", "SC3", "PAID", "LIVE", "PHYSICAL"}
CONTROL_ARTIFACTS = {
    "docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json",
    "docs/PUBLIC_LAUNCH_SC3_REMEDIATION_STATUS_2026-07-09.md",
    "tests/test_quality_gap_ledger.py",
}
EVIDENCE_DIRECTORY_ROOTS = {
    ".github",
    "deploy",
    "docs",
    "ops",
    "scripts",
    "src",
    "tests",
}
EVIDENCE_ROOT_FILES = {
    "Dockerfile",
    "LICENSE",
    "MANIFEST.in",
    "SECURITY.md",
    "docker-compose.yml",
    "pyproject.toml",
    "requirements-geometry.txt",
    "requirements.txt",
    "uv.lock",
}
RECORDED_COMMAND_AUTHORITIES: frozenset[str] = frozenset()
CLAIM_BOUNDARY_TRUE_FIELDS = {
    "code_change_is_not_external_proof",
    "missing_or_stale_evidence_cannot_close_a_gap",
    "sim_only_scope_does_not_require_physical_robot_proof",
    "focused_local_evidence_is_not_full_suite_green",
    "external_evidence_rows_remain_open",
    "physical_evidence_is_nonblocking_for_sim_only",
    "ledger_is_not_its_own_closure_evidence",
    "status_document_is_not_closure_evidence",
    "p2_04_requires_independent_evidence",
    "command_result_without_artifact_binding_is_not_release_proof",
    "untracked_worktree_evidence_cannot_support_status",
    "closure_is_disabled_without_external_attestation_verifier",
    "recorded_commands_are_disabled_without_external_attestation_verifier",
}
CLOSURE_AUTHORITY_POLICY = {
    "enabled": False,
    "verification_mode": "external_signed_release_attestation_required",
    "accepted_authorities": [],
    "reason": (
        "No cryptographically verified commit/release attestation is bound in this worktree ledger."
    ),
}
COMMAND_ATTESTATION_POLICY = {
    "enabled": False,
    "verification_mode": "external_signed_command_attestation_required",
    "accepted_authorities": [],
    "reason": (
        "No cryptographically verified command attestation is bound in this worktree ledger."
    ),
}
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


def _expected_launch_scope(gap_id: str, scopes: list[str]) -> dict[str, Any]:
    enabled_labels = set(LAUNCH_SCOPE_POLICY["enabled_scope_labels"])
    basis = [f"enabled_scope:{scope}" for scope in scopes if scope in enabled_labels]
    if gap_id in set(LAUNCH_SCOPE_POLICY["enabled_paid_gap_ids"]):
        basis.append("enabled_feature:buyer_delivery_and_rights")
    scoped = bool(basis)
    if gap_id in set(LAUNCH_SCOPE_POLICY["conditional_nonblocking_gap_ids"]):
        return {
            "scoped": True,
            "blocking": False,
            "basis": basis,
            "nonblocking_reason": "external_correlation_claim_not_enabled",
        }
    if scoped:
        return {
            "scoped": True,
            "blocking": True,
            "basis": basis,
            "nonblocking_reason": None,
        }
    if gap_id == "EVID-14":
        reason = "physical_robot_claim_not_enabled"
    elif gap_id == "EVID-09":
        reason = "payments_and_payouts_not_enabled"
    elif gap_id == "EVID-12":
        reason = "unsupported_device_lanes_not_marketed"
    else:
        reason = "ptdp_or_paid_feature_not_enabled"
    return {
        "scoped": False,
        "blocking": False,
        "basis": [],
        "nonblocking_reason": reason,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _text_sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _criterion_evidence_mapping_sha256(ledger: Mapping[str, Any]) -> str:
    mapping = []
    for gap in ledger.get("gaps", []):
        if not isinstance(gap, Mapping):
            continue
        for criterion in gap.get("criteria", []):
            if not isinstance(criterion, Mapping):
                continue
            artifacts = criterion.get("evidence_artifacts", [])
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
                        for artifact in artifacts
                        if isinstance(artifact, Mapping)
                    ],
                    "command": {field: command.get(field) for field in ("applicable", "command")}
                    if isinstance(command, Mapping)
                    else None,
                }
            )
    canonical = json.dumps(
        mapping,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return _text_sha256(canonical)


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None and parsed.utcoffset() is not None else None


def _git_head() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip().lower()
    return value if completed.returncode == 0 and COMMIT_PATTERN.fullmatch(value) else None


def _git_tracked_paths() -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return {item.decode("utf-8") for item in completed.stdout.split(b"\0") if item}


def _safe_repository_file(path_text: str) -> Path | None:
    relative = PurePosixPath(path_text)
    if (
        not path_text
        or relative.is_absolute()
        or str(relative) != path_text
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        return None
    if len(relative.parts) == 1:
        if path_text not in EVIDENCE_ROOT_FILES:
            return None
    elif relative.parts[0] not in EVIDENCE_DIRECTORY_ROOTS:
        return None
    path = ROOT / path_text
    cursor = ROOT
    if any((cursor := cursor / part).is_symlink() for part in relative.parts):
        return None
    if not path.is_file():
        return None
    try:
        path.resolve().relative_to(ROOT.resolve())
    except (OSError, ValueError):
        return None
    return path


def _safe_command_output_file(path_text: str) -> Path | None:
    relative = PurePosixPath(path_text)
    if (
        not path_text
        or relative.is_absolute()
        or str(relative) != path_text
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.parts[:2] != ("output", "release-evidence")
    ):
        return None
    path = ROOT / path_text
    cursor = ROOT
    if any((cursor := cursor / part).is_symlink() for part in relative.parts):
        return None
    if not path.is_file():
        return None
    try:
        path.resolve().relative_to((ROOT / "output" / "release-evidence").resolve())
    except (OSError, ValueError):
        return None
    return path


def _load_ledger() -> dict[str, Any]:
    value = json.loads(LEDGER.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _audit_rows() -> list[dict[str, Any]]:
    lines = AUDIT.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        heading = re.match(r"^### ((?:REL|DATA|SC3|RUN)-\d{2}) — ", line)
        if heading:
            end = next(
                (
                    candidate
                    for candidate in range(index + 1, len(lines))
                    if lines[candidate].startswith(("## ", "### "))
                ),
                len(lines),
            )
            section = lines[index + 1 : end]
            scopes_line = next(item for item in section if item.startswith("**Scopes:**"))
            criterion_line = next(item for item in section if item.startswith("**Exit criteria:**"))
            rows.append(
                {
                    "id": heading.group(1),
                    "scopes": re.findall(r"`([^`]+)`", scopes_line),
                    "acceptance_text": criterion_line.removeprefix("**Exit criteria:** "),
                    "source_line": index + 1,
                    "source_kind": "detailed_exit_criteria",
                }
            )
            continue
        if not line.startswith("| **"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        first = re.match(
            r"^\*\*((?:REL|DATA|SC3|RUN|P2|EVID)-\d{2})\*\* (.+)$",
            cells[0],
        )
        if first is None:
            continue
        rows.append(
            {
                "id": first.group(1),
                "scopes": re.findall(r"`([^`]+)`", first.group(2)),
                "acceptance_text": cells[-1],
                "source_line": index + 1,
                "source_kind": "table_acceptance_criterion",
            }
        )
    return sorted(rows, key=lambda row: int(row["source_line"]))


def _command_result_errors(
    value: object,
    *,
    prefix: str,
    as_of: datetime,
    expected_commit: str | None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return [f"{prefix}:command_result_not_object"]
    expected_fields = {
        "applicable",
        "command",
        "status",
        "exit_code",
        "generated_at",
        "output_artifact",
        "output_artifact_sha256",
        "authority",
        "commit",
        "release_id",
        "summary",
    }
    if set(value) != expected_fields:
        errors.append(f"{prefix}:command_result_fields_invalid")
    applicable = value.get("applicable")
    status = value.get("status")
    if type(applicable) is not bool:
        errors.append(f"{prefix}:command_result_applicable_invalid")
    if status not in {"not_applicable", "not_recorded", "passed", "failed", "blocked"}:
        errors.append(f"{prefix}:command_result_status_invalid")
    if applicable is True:
        if not isinstance(value.get("command"), str) or not value["command"].strip():
            errors.append(f"{prefix}:command_missing")
        if status == "not_applicable":
            errors.append(f"{prefix}:applicable_command_marked_not_applicable")
    elif applicable is False:
        if status != "not_applicable":
            errors.append(f"{prefix}:nonapplicable_command_status_invalid")
        for field in (
            "command",
            "exit_code",
            "generated_at",
            "output_artifact",
            "output_artifact_sha256",
            "authority",
            "commit",
            "release_id",
            "summary",
        ):
            if value.get(field) is not None:
                errors.append(f"{prefix}:nonapplicable_command_field_not_null:{field}")
    if status == "not_recorded":
        for field in (
            "exit_code",
            "generated_at",
            "output_artifact",
            "output_artifact_sha256",
            "authority",
            "commit",
            "release_id",
            "summary",
        ):
            if value.get(field) is not None:
                errors.append(f"{prefix}:unrecorded_command_field_not_null:{field}")
    if status in {"passed", "failed", "blocked"}:
        errors.append(f"{prefix}:recorded_command_attestation_disabled")
        if type(value.get("exit_code")) is not int:
            errors.append(f"{prefix}:command_exit_code_invalid")
        if status == "passed" and value.get("exit_code") != 0:
            errors.append(f"{prefix}:passed_command_exit_code_nonzero")
        generated_at = _parse_timestamp(value.get("generated_at"))
        if generated_at is None:
            errors.append(f"{prefix}:command_generated_at_invalid")
        elif generated_at > as_of:
            errors.append(f"{prefix}:command_generated_at_future")
        output_path_text = str(value.get("output_artifact") or "")
        output_path = _safe_command_output_file(output_path_text)
        if output_path is None:
            errors.append(f"{prefix}:command_output_artifact_missing_or_unsafe")
        elif value.get("output_artifact_sha256") != _sha256(output_path):
            errors.append(f"{prefix}:command_output_artifact_digest_mismatch")
        if value.get("authority") not in RECORDED_COMMAND_AUTHORITIES:
            errors.append(f"{prefix}:command_authority_invalid")
        if value.get("commit") != expected_commit or expected_commit is None:
            errors.append(f"{prefix}:command_commit_binding_invalid")
        if not isinstance(value.get("release_id"), str) or not value["release_id"].strip():
            errors.append(f"{prefix}:command_release_binding_invalid")
        if not isinstance(value.get("summary"), str) or not value["summary"].strip():
            errors.append(f"{prefix}:command_summary_missing")
    return errors


def _artifact_freshness_is_valid(
    artifact: Mapping[str, Any], *, status: str, as_of: datetime
) -> bool:
    generated_at = _parse_timestamp(artifact.get("generated_at"))
    evaluated_at = _parse_timestamp(artifact.get("freshness_evaluated_at"))
    fresh_until = _parse_timestamp(artifact.get("fresh_until"))
    return bool(
        artifact.get("freshness_status") == status
        and generated_at is not None
        and evaluated_at is not None
        and fresh_until is not None
        and generated_at <= evaluated_at <= as_of < fresh_until
    )


def _artifact_is_valid_remediation(
    artifact: Mapping[str, Any],
    *,
    control_artifacts: set[str],
    tracked_paths: set[str],
    as_of: datetime,
) -> bool:
    path_text = str(artifact.get("path") or "")
    path = _safe_repository_file(path_text)
    expected_authority = (
        "repository_worktree_digest"
        if path_text in tracked_paths
        else "repository_untracked_worktree_digest"
    )
    return bool(
        artifact.get("authoritative") is True
        and artifact.get("supports_remediation") is True
        and path_text in tracked_paths
        and path_text not in control_artifacts
        and path is not None
        and artifact.get("authority") == expected_authority
        and artifact.get("sha256") == _sha256(path)
        and _artifact_freshness_is_valid(artifact, status="current_unbound", as_of=as_of)
    )


def _artifact_is_valid_p2_04_evidence(artifact: Mapping[str, Any]) -> bool:
    path = _safe_repository_file(str(artifact.get("path") or ""))
    if path is None:
        return False
    banner = path.read_text(encoding="utf-8")[:1200]
    return bool(
        artifact.get("path") == P2_04_EVIDENCE_PATH
        and artifact.get("role") == "policy_or_runbook"
        and artifact.get("supports_remediation") is True
        and artifact.get("supports_closure") is False
        and "SUPERSEDED FOR CURRENT LAUNCH STATUS" in banner
        and "public_launch_sc3_quality_gap_ledger_2026-07-09.json" in banner
    )


def _derive_criterion_status(
    criterion: Mapping[str, Any],
    *,
    control_artifacts: set[str],
    tracked_paths: set[str],
    as_of: datetime,
    git_head: str | None,
    closure_policy: Mapping[str, Any],
) -> str:
    artifacts = criterion.get("evidence_artifacts")
    artifact_rows = (
        [item for item in artifacts if isinstance(item, Mapping)]
        if isinstance(artifacts, list)
        else []
    )
    binding = criterion.get("binding") if isinstance(criterion.get("binding"), Mapping) else {}
    command = (
        criterion.get("command_result")
        if isinstance(criterion.get("command_result"), Mapping)
        else {}
    )
    acceptance = (
        criterion.get("acceptance_check")
        if isinstance(criterion.get("acceptance_check"), Mapping)
        else {}
    )
    criterion_freshness = (
        criterion.get("freshness") if isinstance(criterion.get("freshness"), Mapping) else {}
    )
    criterion_evaluated_at = _parse_timestamp(criterion_freshness.get("evaluated_at"))
    criterion_fresh_until = _parse_timestamp(criterion_freshness.get("fresh_until"))
    criterion_generated_at = _parse_timestamp(criterion.get("generated_at"))
    fresh = bool(
        criterion_freshness.get("status") == "current_bound"
        and criterion_generated_at is not None
        and criterion_evaluated_at is not None
        and criterion_fresh_until is not None
        and criterion_generated_at <= criterion_evaluated_at <= as_of < criterion_fresh_until
    )
    release_path = _safe_command_output_file(str(binding.get("release_artifact_path") or ""))
    accepted_closure_authorities = set(closure_policy.get("accepted_authorities") or [])
    binding_complete = bool(
        closure_policy.get("enabled") is True
        and binding.get("status") == "bound"
        and git_head is not None
        and binding.get("commit") == git_head
        and str(binding.get("release_id") or "").strip()
        and release_path is not None
        and binding.get("release_artifact_sha256") == _sha256(release_path)
        and binding.get("release_authority") in accepted_closure_authorities
    )
    command_ok = bool(
        not _command_result_errors(
            command,
            prefix="derivation",
            as_of=as_of,
            expected_commit=git_head,
        )
        and (
            command.get("applicable") is False
            or (command.get("status") == "passed" and command.get("exit_code") == 0)
        )
    )
    closure_artifact = any(
        artifact.get("authoritative") is True
        and artifact.get("supports_closure") is True
        and artifact.get("role") == "closure_attestation"
        and artifact.get("authority") in accepted_closure_authorities
        and artifact.get("commit") == binding.get("commit")
        and artifact.get("release_id") == binding.get("release_id")
        and (artifact_path := _safe_repository_file(str(artifact.get("path") or ""))) is not None
        and artifact.get("sha256") == _sha256(artifact_path)
        and _artifact_freshness_is_valid(artifact, status="current_bound", as_of=as_of)
        for artifact in artifact_rows
    )
    closure = bool(
        closure_policy.get("enabled") is True
        and acceptance.get("status") == "passed"
        and binding_complete
        and command_ok
        and closure_artifact
        and fresh
        and criterion.get("remaining_work") == []
    )
    if criterion.get("prior_status") == "closed" and not closure:
        return "reopened"
    if closure:
        return "closed"
    if any(
        _artifact_is_valid_remediation(
            artifact,
            control_artifacts=control_artifacts,
            tracked_paths=tracked_paths,
            as_of=as_of,
        )
        and (
            criterion.get("criterion_id") != "P2-04-AC-01"
            or _artifact_is_valid_p2_04_evidence(artifact)
        )
        for artifact in artifact_rows
    ):
        return "partial"
    return "open"


def _derive_gap_status(
    criteria: list[Mapping[str, Any]],
    *,
    control_artifacts: set[str],
    tracked_paths: set[str],
    as_of: datetime,
    git_head: str | None,
    closure_policy: Mapping[str, Any],
) -> str:
    statuses = [
        _derive_criterion_status(
            criterion,
            control_artifacts=control_artifacts,
            tracked_paths=tracked_paths,
            as_of=as_of,
            git_head=git_head,
            closure_policy=closure_policy,
        )
        for criterion in criteria
    ]
    if "reopened" in statuses:
        return "reopened"
    if statuses and all(status == "closed" for status in statuses):
        return "closed"
    if any(status in {"partial", "closed"} for status in statuses):
        return "partial"
    return "open"


def _validate_ledger(ledger: Mapping[str, Any], *, as_of: datetime | None = None) -> list[str]:
    errors: list[str] = []
    validation_time = as_of or datetime.now(timezone.utc)
    if validation_time.tzinfo is None or validation_time.utcoffset() is None:
        return ["validation_time_invalid"]
    validation_time = validation_time.astimezone(timezone.utc)
    tracked_paths = _git_tracked_paths()
    git_head = _git_head()
    expected_top_fields = {
        "schema_version",
        "ledger_id",
        "generated_at",
        "updated_at",
        "source_audit",
        "source_audit_sha256",
        "evidence_mapping_sha256",
        "authoritative_for_current_status",
        "status_definitions",
        "status_derivation",
        "status_counts",
        "criteria_counts",
        "freshness_policy",
        "closure_authority_policy",
        "command_attestation_policy",
        "launch_scope_policy",
        "launch_scope_counts",
        "claim_boundary",
        "control_artifacts",
        "supersedes",
        "supersession_sets",
        "validation_command_result",
        "gaps",
        "remediation_status_document",
    }
    if set(ledger) != expected_top_fields:
        errors.append("ledger_fields_invalid")
    if ledger.get("schema_version") != "blueprint.public_launch_sc3_quality_gap_ledger.v3":
        errors.append("ledger_schema_invalid")
    if ledger.get("ledger_id") != "public-launch-sc3-quality-2026-07-09":
        errors.append("ledger_id_invalid")
    ledger_generated_at = _parse_timestamp(ledger.get("generated_at"))
    ledger_updated_at = _parse_timestamp(ledger.get("updated_at"))
    if ledger_generated_at is None:
        errors.append("ledger_generated_at_invalid")
    if ledger_updated_at is None:
        errors.append("ledger_updated_at_invalid")
    if (
        ledger_generated_at is not None
        and ledger_updated_at is not None
        and ledger_updated_at < ledger_generated_at
    ):
        errors.append("ledger_updated_before_generated")
    if ledger_generated_at is not None and ledger_generated_at > validation_time:
        errors.append("ledger_generated_at_future")
    if ledger_updated_at is not None and ledger_updated_at > validation_time:
        errors.append("ledger_updated_at_future")
    if ledger.get("source_audit") != "docs/PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md":
        errors.append("source_audit_path_invalid")
    if ledger.get("source_audit_sha256") != _sha256(AUDIT):
        errors.append("source_audit_digest_mismatch")
    computed_mapping_digest = _criterion_evidence_mapping_sha256(ledger)
    if (
        ledger.get("evidence_mapping_sha256") != computed_mapping_digest
        or computed_mapping_digest != APPROVED_CRITERION_EVIDENCE_MAP_SHA256
    ):
        errors.append("criterion_evidence_mapping_digest_mismatch")
    if (
        ledger.get("remediation_status_document")
        != ("docs/PUBLIC_LAUNCH_SC3_REMEDIATION_STATUS_2026-07-09.md")
        or not STATUS.is_file()
    ):
        errors.append("remediation_status_document_invalid")
    if ledger.get("authoritative_for_current_status") is not True:
        errors.append("ledger_not_authoritative")
    status_definitions = ledger.get("status_definitions")
    if not isinstance(status_definitions, Mapping) or set(status_definitions) != ALLOWED_STATUSES:
        errors.append("status_definitions_invalid")
    derivation = ledger.get("status_derivation")
    if not isinstance(derivation, Mapping) or set(derivation) != {
        "rule_id",
        "open",
        "partial",
        "closed",
        "reopened",
        "gap_aggregation",
    }:
        errors.append("status_derivation_invalid")
    elif derivation.get("rule_id") != "blueprint.criterion_status_derivation.v1":
        errors.append("status_derivation_rule_invalid")
    freshness_policy = ledger.get("freshness_policy")
    if not isinstance(freshness_policy, Mapping) or set(freshness_policy) != {
        "evaluated_at",
        "fresh_until",
        "worktree_evidence_is_never_closure_without_commit_binding",
        "expired_or_digest_mismatched_evidence_cannot_support_status",
    }:
        errors.append("freshness_policy_invalid")
    else:
        evaluated_at = _parse_timestamp(freshness_policy.get("evaluated_at"))
        fresh_until = _parse_timestamp(freshness_policy.get("fresh_until"))
        if (
            evaluated_at is None
            or fresh_until is None
            or ledger_generated_at is None
            or not ledger_generated_at <= evaluated_at < fresh_until
        ):
            errors.append("freshness_policy_timestamps_invalid")
        elif evaluated_at > validation_time:
            errors.append("freshness_policy_evaluated_in_future")
        elif validation_time >= fresh_until:
            errors.append("freshness_policy_expired")
        if (
            freshness_policy.get("worktree_evidence_is_never_closure_without_commit_binding")
            is not True
            or freshness_policy.get("expired_or_digest_mismatched_evidence_cannot_support_status")
            is not True
        ):
            errors.append("freshness_policy_boundary_invalid")
    closure_policy = ledger.get("closure_authority_policy")
    if closure_policy != CLOSURE_AUTHORITY_POLICY:
        errors.append("closure_authority_policy_not_fail_closed")
    if not isinstance(closure_policy, Mapping):
        closure_policy = CLOSURE_AUTHORITY_POLICY
    if ledger.get("command_attestation_policy") != COMMAND_ATTESTATION_POLICY:
        errors.append("command_attestation_policy_not_fail_closed")
    if ledger.get("launch_scope_policy") != LAUNCH_SCOPE_POLICY:
        errors.append("launch_scope_policy_invalid")
    claim_boundary = ledger.get("claim_boundary")
    if not isinstance(claim_boundary, Mapping) or set(claim_boundary) != CLAIM_BOUNDARY_TRUE_FIELDS:
        errors.append("claim_boundary_invalid")
    else:
        for name in CLAIM_BOUNDARY_TRUE_FIELDS:
            if claim_boundary.get(name) is not True:
                errors.append(f"claim_boundary_missing:{name}")

    control_artifacts = ledger.get("control_artifacts")
    if not isinstance(control_artifacts, list) or set(control_artifacts) != CONTROL_ARTIFACTS:
        errors.append("control_artifacts_invalid")
        control_set = CONTROL_ARTIFACTS
    else:
        control_set = set(control_artifacts)

    supersedes = ledger.get("supersedes")
    supersession_ids: set[str] = set()
    supersession_paths_by_id: dict[str, str] = {}
    if not isinstance(supersedes, list) or not supersedes:
        errors.append("supersedes_invalid")
        supersedes = []
    for item in supersedes:
        if not isinstance(item, Mapping):
            errors.append("supersession_not_object")
            continue
        if set(item) != {
            "supersession_id",
            "path",
            "status",
            "current_status_source",
            "sha256",
            "banner_checked_at",
        }:
            errors.append("supersession_fields_invalid")
        supersession_id = str(item.get("supersession_id") or "")
        if not supersession_id or supersession_id in supersession_ids:
            errors.append("supersession_id_invalid_or_duplicate")
        supersession_ids.add(supersession_id)
        path_text = str(item.get("path") or "")
        supersession_paths_by_id[supersession_id] = path_text
        path = _safe_repository_file(path_text)
        if path is None or item.get("sha256") != _sha256(path):
            errors.append(f"supersession_digest_invalid:{path_text}")
        if item.get("status") != "historical":
            errors.append(f"supersession_status_invalid:{path_text}")
        if item.get("current_status_source") != (
            "docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json"
        ):
            errors.append(f"supersession_current_source_invalid:{path_text}")
        banner_checked_at = _parse_timestamp(item.get("banner_checked_at"))
        if banner_checked_at is None or banner_checked_at > validation_time:
            errors.append(f"supersession_banner_timestamp_invalid:{path_text}")

    supersession_sets = ledger.get("supersession_sets")
    resolved_sets: set[str] = set()
    supersession_paths_by_set: dict[str, set[str]] = {}
    if not isinstance(supersession_sets, list) or not supersession_sets:
        errors.append("supersession_sets_invalid")
        supersession_sets = []
    for item in supersession_sets:
        if not isinstance(item, Mapping) or set(item) != {"set_id", "supersession_ids"}:
            errors.append("supersession_set_fields_invalid")
            continue
        set_id = str(item.get("set_id") or "")
        refs = item.get("supersession_ids")
        if not set_id or set_id in resolved_sets:
            errors.append("supersession_set_id_invalid_or_duplicate")
        resolved_sets.add(set_id)
        if not isinstance(refs, list) or set(refs) != supersession_ids:
            errors.append(f"supersession_set_refs_invalid:{set_id}")
        else:
            supersession_paths_by_set[set_id] = {
                supersession_paths_by_id[ref] for ref in refs if ref in supersession_paths_by_id
            }

    errors.extend(
        _command_result_errors(
            ledger.get("validation_command_result"),
            prefix="ledger_validation",
            as_of=validation_time,
            expected_commit=git_head,
        )
    )
    validation_command = ledger.get("validation_command_result")
    command_generated_at = (
        _parse_timestamp(validation_command.get("generated_at"))
        if isinstance(validation_command, Mapping)
        else None
    )
    if (
        ledger_updated_at is not None
        and command_generated_at is not None
        and command_generated_at > ledger_updated_at
    ):
        errors.append("ledger_validation_command_newer_than_ledger")
    audit_rows = _audit_rows()
    audit_by_id = {row["id"]: row for row in audit_rows}
    gaps = ledger.get("gaps")
    if not isinstance(gaps, list):
        return [*errors, "gaps_not_list"]
    ledger_ids = [str(gap.get("id") or "") for gap in gaps if isinstance(gap, Mapping)]
    if ledger_ids != [row["id"] for row in audit_rows]:
        errors.append("gap_ids_or_order_mismatch")
    if len(ledger_ids) != len(set(ledger_ids)):
        errors.append("gap_ids_duplicate")

    criterion_statuses: list[str] = []
    gap_statuses: list[str] = []
    criterion_ids: set[str] = set()
    for raw_gap in gaps:
        if not isinstance(raw_gap, Mapping):
            errors.append("gap_not_object")
            continue
        gap_id = str(raw_gap.get("id") or "")
        prefix = gap_id or "missing-gap-id"
        if ID_PATTERN.fullmatch(gap_id) is None:
            errors.append(f"{prefix}:gap_id_invalid")
        if raw_gap.get("status") not in ALLOWED_STATUSES:
            errors.append(f"{prefix}:gap_status_invalid")
        if raw_gap.get("derived_status") not in ALLOWED_STATUSES:
            errors.append(f"{prefix}:gap_derived_status_invalid")
        if set(raw_gap) != {
            "id",
            "status",
            "derived_status",
            "scopes",
            "nonblocking_for_scopes",
            "launch_scope",
            "commit",
            "release_id",
            "criteria",
            "remaining_work",
        }:
            errors.append(f"{prefix}:gap_fields_invalid")
        audit_row = audit_by_id.get(gap_id, {})
        if raw_gap.get("scopes") != audit_row.get("scopes"):
            errors.append(f"{prefix}:gap_scopes_mismatch")
        if (
            not isinstance(raw_gap.get("scopes"), list)
            or not set(raw_gap.get("scopes") or []) <= ALLOWED_SCOPES
        ):
            errors.append(f"{prefix}:gap_scopes_invalid")
        if (
            not isinstance(raw_gap.get("nonblocking_for_scopes"), list)
            or not set(raw_gap.get("nonblocking_for_scopes") or []) <= ALLOWED_SCOPES
        ):
            errors.append(f"{prefix}:gap_nonblocking_scopes_invalid")
        if set(raw_gap.get("scopes") or []) & set(raw_gap.get("nonblocking_for_scopes") or []):
            errors.append(f"{prefix}:blocking_and_nonblocking_scopes_overlap")
        expected_launch_scope = _expected_launch_scope(gap_id, list(raw_gap.get("scopes") or []))
        if raw_gap.get("launch_scope") != expected_launch_scope:
            errors.append(f"{prefix}:launch_scope_mismatch")
        commit = raw_gap.get("commit")
        release_id = raw_gap.get("release_id")
        if commit is not None and COMMIT_PATTERN.fullmatch(str(commit)) is None:
            errors.append(f"{prefix}:gap_commit_invalid")
        if commit is not None and commit != git_head:
            errors.append(f"{prefix}:gap_commit_not_current_head")
        if commit is not None and closure_policy.get("enabled") is not True:
            errors.append(f"{prefix}:gap_binding_forbidden_without_closure_authority")
        if commit is None and release_id is not None:
            errors.append(f"{prefix}:release_without_commit")
        if commit is not None and (not isinstance(release_id, str) or not release_id.strip()):
            errors.append(f"{prefix}:commit_without_release")
        criteria_value = raw_gap.get("criteria")
        if not isinstance(criteria_value, list) or not criteria_value:
            errors.append(f"{prefix}:criteria_invalid")
            continue
        criteria = [item for item in criteria_value if isinstance(item, Mapping)]
        if len(criteria) != len(criteria_value):
            errors.append(f"{prefix}:criterion_not_object")
        for criterion in criteria:
            criterion_prefix = str(criterion.get("criterion_id") or prefix)
            expected_criterion_fields = {
                "criterion_id",
                "source",
                "acceptance_text",
                "acceptance_text_sha256",
                "derived_status",
                "prior_status",
                "scopes",
                "nonblocking_for_scopes",
                "launch_scope",
                "generated_at",
                "freshness",
                "binding",
                "evidence_artifacts",
                "command_result",
                "acceptance_check",
                "supersession_refs",
                "remaining_work",
            }
            if set(criterion) != expected_criterion_fields:
                errors.append(f"{criterion_prefix}:criterion_fields_invalid")
            if criterion.get("derived_status") not in ALLOWED_STATUSES:
                errors.append(f"{criterion_prefix}:derived_status_invalid")
            if criterion.get("prior_status") not in {None, *ALLOWED_STATUSES}:
                errors.append(f"{criterion_prefix}:prior_status_invalid")
            if criterion_prefix != f"{gap_id}-AC-01" or criterion_prefix in criterion_ids:
                errors.append(f"{criterion_prefix}:criterion_id_invalid_or_duplicate")
            criterion_ids.add(criterion_prefix)
            if criterion.get("scopes") != raw_gap.get("scopes"):
                errors.append(f"{criterion_prefix}:criterion_scopes_mismatch")
            if criterion.get("nonblocking_for_scopes") != raw_gap.get("nonblocking_for_scopes"):
                errors.append(f"{criterion_prefix}:criterion_nonblocking_scopes_mismatch")
            if criterion.get("launch_scope") != raw_gap.get("launch_scope"):
                errors.append(f"{criterion_prefix}:criterion_launch_scope_mismatch")
            source = criterion.get("source")
            if not isinstance(source, Mapping) or set(source) != {
                "audit_path",
                "line",
                "kind",
            }:
                errors.append(f"{criterion_prefix}:criterion_source_invalid")
            else:
                if source.get("audit_path") != ledger.get("source_audit"):
                    errors.append(f"{criterion_prefix}:criterion_source_audit_mismatch")
                if source.get("line") != audit_row.get("source_line"):
                    errors.append(f"{criterion_prefix}:criterion_source_line_mismatch")
                if source.get("kind") != audit_row.get("source_kind"):
                    errors.append(f"{criterion_prefix}:criterion_source_kind_mismatch")
            acceptance_text = str(criterion.get("acceptance_text") or "")
            if acceptance_text != audit_row.get("acceptance_text"):
                errors.append(f"{criterion_prefix}:acceptance_text_mismatch")
            if criterion.get("acceptance_text_sha256") != _text_sha256(acceptance_text):
                errors.append(f"{criterion_prefix}:acceptance_text_digest_mismatch")
            criterion_generated_at = _parse_timestamp(criterion.get("generated_at"))
            if criterion_generated_at is None:
                errors.append(f"{criterion_prefix}:generated_at_invalid")
            elif criterion_generated_at > validation_time:
                errors.append(f"{criterion_prefix}:generated_at_future")
            if criterion.get("generated_at") != ledger.get("generated_at"):
                errors.append(f"{criterion_prefix}:generated_at_not_ledger_bound")

            freshness = criterion.get("freshness")
            if not isinstance(freshness, Mapping) or set(freshness) != {
                "evaluated_at",
                "fresh_until",
                "status",
            }:
                errors.append(f"{criterion_prefix}:freshness_invalid")
            else:
                evaluated_at = _parse_timestamp(freshness.get("evaluated_at"))
                fresh_until = _parse_timestamp(freshness.get("fresh_until"))
                if (
                    criterion_generated_at is None
                    or evaluated_at is None
                    or fresh_until is None
                    or not criterion_generated_at <= evaluated_at <= validation_time < fresh_until
                ):
                    errors.append(f"{criterion_prefix}:freshness_timestamps_invalid")
                if freshness.get("status") not in {
                    "current_unbound",
                    "current_bound",
                    "missing_closure_evidence",
                    "stale",
                }:
                    errors.append(f"{criterion_prefix}:freshness_status_invalid")
                if isinstance(freshness_policy, Mapping) and (
                    freshness.get("evaluated_at") != freshness_policy.get("evaluated_at")
                    or freshness.get("fresh_until") != freshness_policy.get("fresh_until")
                ):
                    errors.append(f"{criterion_prefix}:freshness_policy_mismatch")

            binding = criterion.get("binding")
            if not isinstance(binding, Mapping) or set(binding) != {
                "repository",
                "commit",
                "release_id",
                "release_artifact_path",
                "release_artifact_sha256",
                "release_authority",
                "status",
            }:
                errors.append(f"{criterion_prefix}:binding_invalid")
            else:
                if binding.get("repository") != "BlueprintCapturePipeline":
                    errors.append(f"{criterion_prefix}:binding_repository_invalid")
                if binding.get("commit") != raw_gap.get("commit"):
                    errors.append(f"{criterion_prefix}:binding_commit_mismatch")
                if binding.get("release_id") != raw_gap.get("release_id"):
                    errors.append(f"{criterion_prefix}:binding_release_mismatch")
                if binding.get("commit") is None:
                    if binding.get("status") != "unbound" or any(
                        binding.get(field) is not None
                        for field in (
                            "release_id",
                            "release_artifact_path",
                            "release_artifact_sha256",
                            "release_authority",
                        )
                    ):
                        errors.append(f"{criterion_prefix}:unbound_binding_invalid")
                else:
                    if binding.get("status") != "bound":
                        errors.append(f"{criterion_prefix}:bound_binding_status_invalid")
                    release_path_text = str(binding.get("release_artifact_path") or "")
                    release_path = _safe_command_output_file(release_path_text)
                    if release_path is None:
                        errors.append(f"{criterion_prefix}:release_artifact_missing_or_unsafe")
                    elif binding.get("release_artifact_sha256") != _sha256(release_path):
                        errors.append(f"{criterion_prefix}:release_artifact_digest_mismatch")
                    accepted_authorities = set(closure_policy.get("accepted_authorities") or [])
                    if binding.get("release_authority") not in accepted_authorities:
                        errors.append(f"{criterion_prefix}:release_authority_invalid")
                    if closure_policy.get("enabled") is not True:
                        errors.append(f"{criterion_prefix}:bound_binding_without_closure_authority")

            artifacts = criterion.get("evidence_artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                errors.append(f"{criterion_prefix}:evidence_artifacts_invalid")
                artifacts = []
            artifact_ids: set[str] = set()
            artifact_paths: set[str] = set()
            for artifact_index, artifact in enumerate(artifacts, start=1):
                if not isinstance(artifact, Mapping):
                    errors.append(f"{criterion_prefix}:evidence_artifact_not_object")
                    continue
                if set(artifact) != {
                    "artifact_id",
                    "path",
                    "sha256",
                    "role",
                    "authoritative",
                    "authority",
                    "supports_remediation",
                    "supports_closure",
                    "generated_at",
                    "freshness_evaluated_at",
                    "fresh_until",
                    "freshness_status",
                    "commit",
                    "release_id",
                }:
                    errors.append(f"{criterion_prefix}:evidence_artifact_fields_invalid")
                artifact_id = str(artifact.get("artifact_id") or "")
                if (
                    artifact_id != f"{criterion_prefix}-EV-{artifact_index:02d}"
                    or artifact_id in artifact_ids
                ):
                    errors.append(f"{criterion_prefix}:artifact_id_invalid_or_duplicate")
                artifact_ids.add(artifact_id)
                path_text = str(artifact.get("path") or "")
                if path_text in artifact_paths:
                    errors.append(f"{criterion_prefix}:artifact_path_duplicate:{path_text}")
                artifact_paths.add(path_text)
                path = _safe_repository_file(path_text)
                if path is None:
                    errors.append(f"{criterion_prefix}:artifact_missing_or_unsafe:{path_text}")
                elif artifact.get("sha256") != _sha256(path):
                    errors.append(f"{criterion_prefix}:artifact_digest_mismatch:{path_text}")
                expected_authority = (
                    "repository_worktree_digest"
                    if path_text in tracked_paths
                    else "repository_untracked_worktree_digest"
                )
                if artifact.get("authoritative") is not True:
                    errors.append(f"{criterion_prefix}:artifact_authority_invalid:{path_text}")
                if artifact.get("authority") != expected_authority:
                    errors.append(
                        f"{criterion_prefix}:artifact_tracking_authority_mismatch:{path_text}"
                    )
                if path_text not in tracked_paths:
                    errors.append(f"{criterion_prefix}:artifact_not_git_tracked:{path_text}")
                if artifact.get("supports_closure") is True:
                    accepted_authorities = set(closure_policy.get("accepted_authorities") or [])
                    if (
                        closure_policy.get("enabled") is not True
                        or artifact.get("role") != "closure_attestation"
                        or artifact.get("authority") not in accepted_authorities
                        or artifact.get("freshness_status") != "current_bound"
                    ):
                        errors.append(
                            f"{criterion_prefix}:closure_artifact_not_trusted:{path_text}"
                        )
                if (
                    type(artifact.get("supports_remediation")) is not bool
                    or type(artifact.get("supports_closure")) is not bool
                ):
                    errors.append(f"{criterion_prefix}:artifact_support_flags_invalid:{path_text}")
                if DIGEST_PATTERN.fullmatch(str(artifact.get("sha256") or "")) is None:
                    errors.append(f"{criterion_prefix}:artifact_digest_format_invalid:{path_text}")
                if artifact.get("supports_remediation") is True and path_text in control_set:
                    errors.append(f"{criterion_prefix}:circular_remediation_evidence:{path_text}")
                if artifact.get("supports_remediation") is True and path_text not in tracked_paths:
                    errors.append(
                        f"{criterion_prefix}:untracked_artifact_cannot_support_remediation:{path_text}"
                    )
                artifact_generated_at = _parse_timestamp(artifact.get("generated_at"))
                artifact_evaluated_at = _parse_timestamp(artifact.get("freshness_evaluated_at"))
                artifact_fresh_until = _parse_timestamp(artifact.get("fresh_until"))
                if artifact_generated_at is None:
                    errors.append(
                        f"{criterion_prefix}:artifact_timestamp_invalid:{path_text}:generated_at"
                    )
                if artifact_evaluated_at is None:
                    errors.append(
                        f"{criterion_prefix}:artifact_timestamp_invalid:{path_text}:freshness_evaluated_at"
                    )
                if artifact_fresh_until is None:
                    errors.append(
                        f"{criterion_prefix}:artifact_timestamp_invalid:{path_text}:fresh_until"
                    )
                if artifact.get("generated_at") != ledger.get("generated_at"):
                    errors.append(f"{criterion_prefix}:artifact_generated_at_mismatch:{path_text}")
                if isinstance(freshness_policy, Mapping) and (
                    artifact.get("freshness_evaluated_at") != freshness_policy.get("evaluated_at")
                    or artifact.get("fresh_until") != freshness_policy.get("fresh_until")
                ):
                    errors.append(
                        f"{criterion_prefix}:artifact_freshness_policy_mismatch:{path_text}"
                    )
                if (
                    artifact_generated_at is None
                    or artifact_evaluated_at is None
                    or artifact_fresh_until is None
                    or not artifact_generated_at
                    <= artifact_evaluated_at
                    <= validation_time
                    < artifact_fresh_until
                ):
                    errors.append(
                        f"{criterion_prefix}:artifact_freshness_window_invalid:{path_text}"
                    )
                if artifact.get("freshness_status") not in {
                    "current_unbound",
                    "current_bound",
                    "definition_only",
                    "stale",
                }:
                    errors.append(
                        f"{criterion_prefix}:artifact_freshness_status_invalid:{path_text}"
                    )
                if artifact.get("commit") != raw_gap.get("commit"):
                    errors.append(f"{criterion_prefix}:artifact_commit_mismatch:{path_text}")
                if artifact.get("release_id") != raw_gap.get("release_id"):
                    errors.append(f"{criterion_prefix}:artifact_release_mismatch:{path_text}")

            inferred_test_paths = sorted(
                str(artifact.get("path"))
                for artifact in artifacts
                if isinstance(artifact, Mapping)
                and PurePosixPath(str(artifact.get("path") or "")).parts[:1] == ("tests",)
                and PurePosixPath(str(artifact.get("path") or "")).name.startswith("test_")
                and PurePosixPath(str(artifact.get("path") or "")).suffix == ".py"
            )
            declared_test_paths = sorted(
                str(artifact.get("path"))
                for artifact in artifacts
                if isinstance(artifact, Mapping) and artifact.get("role") == "test_contract"
            )
            if inferred_test_paths != declared_test_paths:
                errors.append(f"{criterion_prefix}:test_contract_role_mismatch")
            test_paths = inferred_test_paths
            if criterion_prefix == "P2-04-AC-01":
                test_paths = ["tests/test_quality_gap_ledger.py"]
            expected_applicable = bool(test_paths)
            command_result = criterion.get("command_result")
            if isinstance(command_result, Mapping):
                marker_override = (
                    " -m ''"
                    if "tests/test_oscar_isaac_closed_loop_eval.py" in test_paths
                    else ""
                )
                expected_command = (
                    "python -m pytest -q"
                    + marker_override
                    + " "
                    + " ".join(test_paths)
                    if expected_applicable
                    else None
                )
                if command_result.get("applicable") is not expected_applicable:
                    errors.append(f"{criterion_prefix}:command_applicability_mismatch")
                if command_result.get("command") != expected_command:
                    errors.append(f"{criterion_prefix}:command_target_mismatch")
            errors.extend(
                _command_result_errors(
                    criterion.get("command_result"),
                    prefix=criterion_prefix,
                    as_of=validation_time,
                    expected_commit=git_head,
                )
            )
            acceptance = criterion.get("acceptance_check")
            if not isinstance(acceptance, Mapping) or set(acceptance) != {
                "status",
                "evaluated_at",
                "evidence_artifact_ids",
                "blockers",
            }:
                errors.append(f"{criterion_prefix}:acceptance_check_invalid")
            else:
                if acceptance.get("status") not in {"not_proven", "passed", "failed", "blocked"}:
                    errors.append(f"{criterion_prefix}:acceptance_check_status_invalid")
                acceptance_evaluated_at = _parse_timestamp(acceptance.get("evaluated_at"))
                if (
                    acceptance_evaluated_at is None
                    or criterion_generated_at is None
                    or not criterion_generated_at <= acceptance_evaluated_at <= validation_time
                ):
                    errors.append(f"{criterion_prefix}:acceptance_check_timestamp_invalid")
                if acceptance.get("evidence_artifact_ids") != [
                    str(artifact.get("artifact_id") or "")
                    for artifact in artifacts
                    if isinstance(artifact, Mapping)
                ]:
                    errors.append(f"{criterion_prefix}:acceptance_evidence_refs_invalid")
                blockers = acceptance.get("blockers")
                if not isinstance(blockers, list) or not all(
                    isinstance(item, str) and item.strip() for item in blockers
                ):
                    errors.append(f"{criterion_prefix}:acceptance_blockers_invalid")
                if acceptance.get("status") != "passed" and not acceptance.get("blockers"):
                    errors.append(f"{criterion_prefix}:acceptance_blockers_missing")
                if acceptance.get("status") == "passed" and acceptance.get("blockers") != []:
                    errors.append(f"{criterion_prefix}:passed_acceptance_has_blockers")
            refs = criterion.get("supersession_refs")
            if not isinstance(refs, list) or not refs or not set(refs) <= resolved_sets:
                errors.append(f"{criterion_prefix}:supersession_refs_invalid")
            if criterion_prefix == "P2-04-AC-01":
                ref_values = refs if isinstance(refs, list) else []
                p2_evidence = [
                    artifact
                    for artifact in artifacts
                    if isinstance(artifact, Mapping)
                    and artifact.get("supports_remediation") is True
                ]
                linked_paths = {
                    path
                    for ref in ref_values
                    for path in supersession_paths_by_set.get(str(ref), set())
                }
                if (
                    len(p2_evidence) != 1
                    or not _artifact_is_valid_p2_04_evidence(p2_evidence[0])
                    or P2_04_EVIDENCE_PATH not in linked_paths
                ):
                    errors.append(f"{criterion_prefix}:independent_evidence_invalid")
            remaining_work = criterion.get("remaining_work")
            if not isinstance(remaining_work, list) or not all(
                isinstance(item, str) and item.strip() for item in remaining_work
            ):
                errors.append(f"{criterion_prefix}:remaining_work_invalid")
            if criterion.get("derived_status") != "closed" and not remaining_work:
                errors.append(f"{criterion_prefix}:remaining_work_missing")
            if criterion.get("derived_status") == "closed" and remaining_work != []:
                errors.append(f"{criterion_prefix}:closed_has_remaining_work")

            derived = _derive_criterion_status(
                criterion,
                control_artifacts=control_set,
                tracked_paths=tracked_paths,
                as_of=validation_time,
                git_head=git_head,
                closure_policy=closure_policy,
            )
            criterion_statuses.append(derived)
            if criterion.get("derived_status") != derived:
                errors.append(f"{criterion_prefix}:derived_status_mismatch:{derived}")
            expected_freshness_status = {
                "open": "missing_closure_evidence",
                "partial": "current_unbound",
                "closed": "current_bound",
                "reopened": "stale",
            }[derived]
            if isinstance(freshness, Mapping) and (
                freshness.get("status") != expected_freshness_status
            ):
                errors.append(f"{criterion_prefix}:freshness_status_mismatch:{derived}")

        derived_gap = _derive_gap_status(
            criteria,
            control_artifacts=control_set,
            tracked_paths=tracked_paths,
            as_of=validation_time,
            git_head=git_head,
            closure_policy=closure_policy,
        )
        gap_statuses.append(derived_gap)
        if raw_gap.get("derived_status") != derived_gap or raw_gap.get("status") != derived_gap:
            errors.append(f"{prefix}:gap_derived_status_mismatch:{derived_gap}")
        if raw_gap.get("remaining_work") != [
            item for criterion in criteria for item in criterion.get("remaining_work", [])
        ]:
            errors.append(f"{prefix}:gap_remaining_work_mismatch")
        if raw_gap.get("status") != "closed" and not raw_gap.get("remaining_work"):
            errors.append(f"{prefix}:gap_remaining_work_missing")

    gap_counts = Counter(gap_statuses)
    criterion_counts = Counter(criterion_statuses)
    expected_gap_counts = {
        "open": gap_counts["open"],
        "partial": gap_counts["partial"],
        "closed": gap_counts["closed"],
        "reopened": gap_counts["reopened"],
        "total": len(gap_statuses),
    }
    expected_criterion_counts = {
        "open": criterion_counts["open"],
        "partial": criterion_counts["partial"],
        "closed": criterion_counts["closed"],
        "reopened": criterion_counts["reopened"],
        "total": len(criterion_statuses),
    }
    if ledger.get("status_counts") != expected_gap_counts:
        errors.append("status_counts_mismatch")
    if ledger.get("criteria_counts") != expected_criterion_counts:
        errors.append("criteria_counts_mismatch")
    launch_scoped = [
        gap for gap in gaps if isinstance(gap, Mapping) and gap.get("launch_scope", {}).get("scoped")
    ]
    launch_blocking = [
        gap
        for gap in launch_scoped
        if isinstance(gap.get("launch_scope"), Mapping)
        and gap["launch_scope"].get("blocking") is True
    ]
    launch_status_counts = Counter(str(gap.get("derived_status")) for gap in launch_blocking)
    expected_launch_counts = {
        "scoped": len(launch_scoped),
        "blocking": len(launch_blocking),
        "nonblocking": len(gaps) - len(launch_blocking),
        "blocking_status_counts": {
            "open": launch_status_counts["open"],
            "partial": launch_status_counts["partial"],
            "closed": launch_status_counts["closed"],
            "reopened": launch_status_counts["reopened"],
        },
    }
    if ledger.get("launch_scope_counts") != expected_launch_counts:
        errors.append("launch_scope_counts_mismatch")
    return sorted(set(errors))


def test_current_gap_ledger_maps_all_107_acceptance_criteria_and_derives_status() -> None:
    ledger = _load_ledger()
    audit_rows = _audit_rows()

    assert len(audit_rows) == 107
    assert len({row["id"] for row in audit_rows}) == 107
    assert _validate_ledger(ledger) == []
    assert ledger["evidence_mapping_sha256"] == (APPROVED_CRITERION_EVIDENCE_MAP_SHA256)
    assert ledger["status_counts"] == {
        "open": 13,
        "partial": 94,
        "closed": 0,
        "reopened": 0,
        "total": 107,
    }
    assert ledger["criteria_counts"] == ledger["status_counts"]
    assert all(gap["commit"] is None and gap["release_id"] is None for gap in ledger["gaps"])


def test_digest_binding_and_non_circular_status_derivation_fail_closed() -> None:
    ledger = _load_ledger()
    rel_01 = next(gap for gap in ledger["gaps"] if gap["id"] == "REL-01")
    rel_01["criteria"][0]["evidence_artifacts"][0]["sha256"] = f"sha256:{'0' * 64}"
    errors = _validate_ledger(ledger)
    assert any("REL-01-AC-01:artifact_digest_mismatch" in item for item in errors)

    ledger = _load_ledger()
    p2_04 = next(gap for gap in ledger["gaps"] if gap["id"] == "P2-04")
    artifact = p2_04["criteria"][0]["evidence_artifacts"][0]
    artifact["path"] = "docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json"
    artifact["sha256"] = _sha256(LEDGER)
    errors = _validate_ledger(ledger)
    assert any("P2-04-AC-01:circular_remediation_evidence" in item for item in errors)
    assert any("P2-04-AC-01:derived_status_mismatch:open" in item for item in errors)


def test_false_closure_stale_evidence_and_malformed_command_result_are_rejected() -> None:
    ledger = _load_ledger()
    rel_03 = next(gap for gap in ledger["gaps"] if gap["id"] == "REL-03")
    rel_03["status"] = "closed"
    rel_03["derived_status"] = "closed"
    rel_03["criteria"][0]["derived_status"] = "closed"
    assert any(
        "REL-03:gap_derived_status_mismatch:partial" in item for item in _validate_ledger(ledger)
    )

    ledger = _load_ledger()
    rel_04 = next(gap for gap in ledger["gaps"] if gap["id"] == "REL-04")
    for artifact in rel_04["criteria"][0]["evidence_artifacts"]:
        artifact["freshness_status"] = "stale"
    assert any(
        "REL-04-AC-01:derived_status_mismatch:open" in item for item in _validate_ledger(ledger)
    )

    ledger = _load_ledger()
    p2_04 = next(gap for gap in ledger["gaps"] if gap["id"] == "P2-04")
    command_result = p2_04["criteria"][0]["command_result"]
    command_result.update(
        {
            "status": "passed",
            "exit_code": 0,
            "generated_at": "2030-07-10T12:45:00+00:00",
            "output_artifact": "docs/specs/launch-audit-2026-07-02/README.md",
            "output_artifact_sha256": _sha256(
                ROOT / "docs/specs/launch-audit-2026-07-02/README.md"
            ),
            "authority": "self_asserted_summary",
            "commit": f"{'0' * 40}",
            "release_id": "forged-release",
            "summary": "fabricated passing result",
        }
    )
    errors = _validate_ledger(ledger)
    assert "P2-04-AC-01:command_generated_at_future" in errors
    assert "P2-04-AC-01:recorded_command_attestation_disabled" in errors
    assert "P2-04-AC-01:command_output_artifact_missing_or_unsafe" in errors
    assert "P2-04-AC-01:command_authority_invalid" in errors
    assert "P2-04-AC-01:command_commit_binding_invalid" in errors


def test_expired_consistent_freshness_windows_fail_closed() -> None:
    ledger = _load_ledger()
    fresh_until = _parse_timestamp(ledger["freshness_policy"]["fresh_until"])
    assert fresh_until is not None

    errors = _validate_ledger(ledger, as_of=fresh_until + timedelta(seconds=1))

    assert "freshness_policy_expired" in errors
    assert "REL-01-AC-01:derived_status_mismatch:open" in errors
    assert "REL-01:gap_derived_status_mismatch:open" in errors


def test_forged_commit_release_and_closure_policy_cannot_close_p2_04() -> None:
    ledger = _load_ledger()
    ledger["closure_authority_policy"].update(
        {
            "enabled": True,
            "accepted_authorities": ["repository_worktree_digest"],
        }
    )
    p2_04 = next(gap for gap in ledger["gaps"] if gap["id"] == "P2-04")
    criterion = p2_04["criteria"][0]
    artifact = criterion["evidence_artifacts"][0]
    forged_commit = f"{'0' * 40}"
    forged_release = "forged-release"
    release_path = artifact["path"]

    p2_04.update(
        {
            "status": "closed",
            "derived_status": "closed",
            "commit": forged_commit,
            "release_id": forged_release,
            "remaining_work": [],
        }
    )
    criterion.update(
        {
            "derived_status": "closed",
            "freshness": {
                **criterion["freshness"],
                "status": "current_bound",
            },
            "binding": {
                "repository": "BlueprintCapturePipeline",
                "commit": forged_commit,
                "release_id": forged_release,
                "release_artifact_path": release_path,
                "release_artifact_sha256": artifact["sha256"],
                "release_authority": "repository_worktree_digest",
                "status": "bound",
            },
            "acceptance_check": {
                **criterion["acceptance_check"],
                "status": "passed",
                "blockers": [],
            },
            "remaining_work": [],
        }
    )
    artifact.update(
        {
            "role": "closure_attestation",
            "supports_closure": True,
            "freshness_status": "current_bound",
            "commit": forged_commit,
            "release_id": forged_release,
        }
    )

    errors = _validate_ledger(ledger)

    assert "closure_authority_policy_not_fail_closed" in errors
    assert "P2-04:gap_commit_not_current_head" in errors
    assert "P2-04-AC-01:release_artifact_missing_or_unsafe" in errors
    assert "P2-04-AC-01:derived_status_mismatch:open" in errors
    assert "P2-04:gap_derived_status_mismatch:open" in errors


def test_symlinked_or_out_of_policy_local_evidence_cannot_derive_partial() -> None:
    ledger = _load_ledger()
    p2_04 = next(gap for gap in ledger["gaps"] if gap["id"] == "P2-04")
    artifact = p2_04["criteria"][0]["evidence_artifacts"][0]
    unsafe_path = ROOT / ".venv/bin/python"

    artifact["path"] = ".venv/bin/python"
    artifact["sha256"] = _sha256(unsafe_path) if unsafe_path.is_file() else f"sha256:{'0' * 64}"
    artifact["authority"] = "repository_untracked_worktree_digest"

    assert _safe_repository_file(artifact["path"]) is None
    errors = _validate_ledger(ledger)
    assert "P2-04-AC-01:artifact_missing_or_unsafe:.venv/bin/python" in errors
    assert "P2-04-AC-01:derived_status_mismatch:open" in errors


def test_unrelated_tracked_file_cannot_impersonate_p2_04_banner_evidence() -> None:
    ledger = _load_ledger()
    p2_04 = next(gap for gap in ledger["gaps"] if gap["id"] == "P2-04")
    artifact = p2_04["criteria"][0]["evidence_artifacts"][0]

    artifact["path"] = "LICENSE"
    artifact["sha256"] = _sha256(ROOT / "LICENSE")
    artifact["authority"] = "repository_worktree_digest"

    assert artifact["path"] in _git_tracked_paths()
    errors = _validate_ledger(ledger)
    assert "P2-04-AC-01:independent_evidence_invalid" in errors
    assert "P2-04-AC-01:derived_status_mismatch:open" in errors


def test_p2_04_has_independent_evidence_and_no_exemption() -> None:
    ledger = _load_ledger()
    p2_04 = next(gap for gap in ledger["gaps"] if gap["id"] == "P2-04")
    criterion = p2_04["criteria"][0]
    independent = [
        artifact
        for artifact in criterion["evidence_artifacts"]
        if artifact["supports_remediation"] and artifact["path"] not in CONTROL_ARTIFACTS
    ]

    assert p2_04["status"] == "partial"
    assert independent
    assert [artifact["path"] for artifact in independent] == [P2_04_EVIDENCE_PATH]
    assert _artifact_is_valid_p2_04_evidence(independent[0])
    assert all((ROOT / artifact["path"]).is_file() for artifact in independent)
    assert criterion["command_result"]["command"] == (
        "python -m pytest -q tests/test_quality_gap_ledger.py"
    )
    assert criterion["command_result"]["status"] == "not_recorded"
    assert all(artifact["path"] in _git_tracked_paths() for artifact in independent)
    assert criterion["remaining_work"]
    assert ledger["claim_boundary"]["p2_04_requires_independent_evidence"] is True
    assert ledger["claim_boundary"]["ledger_is_not_its_own_closure_evidence"] is True


def test_external_manual_and_physical_rows_remain_honestly_unclosed() -> None:
    ledger = _load_ledger()
    gaps = {gap["id"]: gap for gap in ledger["gaps"]}

    partial_external = {"SC3-22", "EVID-01", "EVID-11"}
    assert all(gaps[gap_id]["status"] == "partial" for gap_id in partial_external)
    open_external = {
        "REL-02",
        *(f"EVID-{index:02d}" for index in range(2, 11)),
        *(f"EVID-{index:02d}" for index in range(12, 15)),
    }
    assert all(gaps[gap_id]["status"] == "open" for gap_id in open_external)
    assert gaps["REL-02"]["status"] == "open"
    for gap_id in sorted(partial_external | open_external):
        criterion = gaps[gap_id]["criteria"][0]
        assert criterion["acceptance_check"]["status"] == "not_proven"
        assert criterion["remaining_work"]
        assert criterion["derived_status"] != "closed"
    for gap_id in open_external:
        assert not any(
            artifact["supports_remediation"]
            for artifact in gaps[gap_id]["criteria"][0]["evidence_artifacts"]
        )
    assert gaps["SC3-22"]["launch_scope"]["blocking"] is False
    assert gaps["EVID-01"]["launch_scope"]["blocking"] is False
    assert gaps["EVID-14"]["scopes"] == ["PHYSICAL"]
    assert gaps["EVID-14"]["nonblocking_for_scopes"] == ["SIM"]
    assert ledger["claim_boundary"]["physical_evidence_is_nonblocking_for_sim_only"] is True


def test_remediation_status_matches_criterion_ledger_without_claiming_full_green() -> None:
    ledger = _load_ledger()
    text = STATUS.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert ledger["remediation_status_document"] == (
        "docs/PUBLIC_LAUNCH_SC3_REMEDIATION_STATUS_2026-07-09.md"
    )
    assert "107 authored acceptance criteria" in normalized
    assert "94 rows and criteria are `partial`" in normalized
    assert "13 rows and criteria remain `open`" in normalized
    assert "No row or criterion is `closed`" in normalized
    assert "all 107 `commit` and `release_id` bindings remain `null`" in normalized
    assert "266 criterion-evidence records covering 164" in normalized
    assert "252 are remediation records" in normalized
    assert "14 are definition-only" in normalized
    assert "all 94 are `not_recorded`" in normalized
    assert "the other 13 are `not_applicable`" in normalized
    assert "Recorded command attestations are disabled in v3" in normalized
    assert "Closure derivation is disabled in this v3 snapshot" in normalized
    assert "99 criteria are launch-scoped" in normalized
    assert "97 are launch-blocking" in normalized
    assert "P2-04 has no self-evidence exemption" in normalized
    p2_04 = next(gap for gap in ledger["gaps"] if gap["id"] == "P2-04")
    p2_digest = p2_04["criteria"][0]["evidence_artifacts"][0]["sha256"]
    assert p2_digest.removeprefix("sha256:") in text
    assert "full-suite rerun is still pending" in text
    assert "nonblocking for the\n  evaluator-bounded sim-only scope" in text


def test_every_superseded_audit_is_digest_bound_and_bannered() -> None:
    ledger = _load_ledger()
    superseded = ledger["supersedes"]
    paths = [item["path"] for item in superseded]

    assert len(paths) == len(set(paths))
    for item in superseded:
        path = ROOT / item["path"]
        text = path.read_text(encoding="utf-8")
        assert "SUPERSEDED FOR CURRENT LAUNCH STATUS" in text[:1200], item["path"]
        assert "public_launch_sc3_quality_gap_ledger_2026-07-09.json" in text[:1200]
        assert item["sha256"] == _sha256(path)


def test_mutation_fixture_does_not_modify_authoritative_ledger() -> None:
    original = _load_ledger()
    mutated = copy.deepcopy(original)
    mutated["gaps"][0]["criteria"][0]["scopes"] = ["LIVE"]

    assert any(
        "REL-01-AC-01:criterion_scopes_mismatch" in item for item in _validate_ledger(mutated)
    )
    assert _load_ledger() == original
