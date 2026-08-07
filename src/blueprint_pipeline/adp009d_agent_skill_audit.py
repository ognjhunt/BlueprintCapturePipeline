"""Fail-closed contract for the ADP-009D official agent-skill audit.

The checked-in receipt records what was observed from primary publisher sources
at a specific time.  It is not a live package resolver and it does not turn a
publisher branch head into a qualified runtime.  Runtime manifests must bind
the compatible commits selected by the receipt separately.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "adp009d_agent_skill_audit.v1"
PROGRAM_ID = "arm-decision-proof-v1"

EXPECTED_REPOSITORIES = {
    "isaac_sim_omniverse": {
        "repository_url": "https://github.com/isaac-sim/IsaacSim",
        "revision": "987015050efebfd0cd5d3736ae47fffe5adee308",
        "tree": "7530065964b0bfbb3e6c102ffa20456870079d7e",
    },
    "isaac_lab": {
        "repository_url": "https://github.com/isaac-sim/IsaacLab",
        "revision": "3ea6f7bbf6c7d515aa1f8e8c54bfdfffda2d4857",
        "tree": "e188a36c7538ed84e7e0918f143f82890cbe98cb",
    },
    "isaac_lab_arena": {
        "repository_url": "https://github.com/isaac-sim/IsaacLab-Arena",
        "revision": "8b4a3a47fc53de23e8205089d71109a2e2348acd",
        "tree": "03f31f3dd56c56d00f24dbfb09711ec0ab345de8",
    },
    "nvidia_skills": {
        "repository_url": "https://github.com/NVIDIA/skills",
        "revision": "358510a74b59120bc3d4bd183ac92734018b38f3",
        "tree": "b9c86063c7aeb40f83dd4fd8f84051ab96813eb4",
    },
    "ovrtx": {
        "repository_url": "https://github.com/NVIDIA-Omniverse/ovrtx",
        "revision": "4b9a5fe6f8becf6c5ff031e167cd4201054a96ce",
        "tree": "9bde58c9194250f226d4d24b37924fca62c9ac22",
    },
    "physx": {
        "repository_url": "https://github.com/NVIDIA-Omniverse/PhysX",
        "revision": "7845321d31fa3619917ebe127ab5e08e73de0bdb",
        "tree": "c2e1ed19bb3e2648af75bf68437e91f8e0faaca0",
    },
}

REQUIRED_GUIDANCE = {
    "isaaclab-building-environments",
    "isaaclab-planning-manipulation-tasks",
    "isaaclab-randomizing-with-events",
    "isaaclab-using-sensors-actuators",
    "isaaclab-selecting-backends",
    "isaaclab-using-presets",
    "ovrtx-loading-usd",
    "ovrtx-renderer-creation",
    "ovrtx-render-settings",
    "ovrtx-camera-outputs-rt2",
    "ovrtx-reading-render-output",
    "ovrtx-stepping-and-rendering",
    "ovrtx-warmup",
    "ovrtx-semantic-labels",
    "nvidia-omniverse-cad-to-simready",
    "nvidia-omniverse-usd-performance-tuning",
    "nvidia-omniverse-realtime-viewer",
}

APPLICATION_STATUSES = {
    "followed_in_architecture",
    "followed_in_implementation",
    "pending_implementation",
    "not_applicable",
    "gated_until_measured",
}

_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class AgentSkillAuditError(ValueError):
    """Stable fail-closed validation errors for the audit receipt."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__("; ".join(self.errors))


def _rows(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _nonempty_strings(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(item, str) and bool(item.strip()) for item in value
    )


def validate_agent_skill_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an audit receipt and return a detached JSON-compatible copy."""

    try:
        normalized = json.loads(json.dumps(audit))
    except (TypeError, ValueError) as exc:
        raise AgentSkillAuditError(["audit_not_json_serializable"]) from exc
    if not isinstance(normalized, dict):
        raise AgentSkillAuditError(["audit_not_mapping"])

    errors: list[str] = []
    if normalized.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    if normalized.get("program_id") != PROGRAM_ID:
        errors.append("program_id_invalid")
    if normalized.get("milestone") != "ADP-009D_public_scene_franka_policy_rehearsal":
        errors.append("milestone_invalid")
    if normalized.get("audit_status") not in {
        "official_source_audit_complete_implementation_pending",
        "official_source_audit_complete_implementation_bound",
    }:
        errors.append("audit_status_invalid")

    audit_window = normalized.get("audit_window")
    if not isinstance(audit_window, Mapping):
        errors.append("audit_window_missing")
    else:
        for field in ("started_at", "completed_at"):
            if not _RFC3339_UTC.fullmatch(str(audit_window.get(field) or "")):
                errors.append(f"audit_window_{field}_invalid")

    base = normalized.get("implementation_base")
    if not isinstance(base, Mapping):
        errors.append("implementation_base_missing")
    else:
        if not _SHA.fullmatch(str(base.get("commit") or "")):
            errors.append("implementation_base_commit_invalid")
        if not _SHA.fullmatch(str(base.get("tree") or "")):
            errors.append("implementation_base_tree_invalid")
        if base.get("provisional_until_pr_374_merge") is not False:
            errors.append("implementation_base_not_final")

    repository_rows = _rows(normalized.get("repositories"))
    repositories = {
        str(row.get("source_id") or ""): row for row in repository_rows
    }
    if len(repositories) != len(repository_rows):
        errors.append("repository_source_id_duplicate_or_missing")
    if set(repositories) != set(EXPECTED_REPOSITORIES):
        errors.append("repository_set_invalid")
    for source_id, expected in EXPECTED_REPOSITORIES.items():
        row = repositories.get(source_id)
        if row is None:
            continue
        for field, expected_value in expected.items():
            if row.get(field) != expected_value:
                errors.append(f"repository_{source_id}_{field}_invalid")
        if row.get("ref_verified_at_retrieval") is not True:
            errors.append(f"repository_{source_id}_ref_not_verified")
        if not _RFC3339_UTC.fullmatch(str(row.get("retrieved_at") or "")):
            errors.append(f"repository_{source_id}_retrieved_at_invalid")
        if not _nonempty_strings(row.get("official_source_urls")):
            errors.append(f"repository_{source_id}_official_source_urls_missing")
        if not isinstance(row.get("license"), Mapping):
            errors.append(f"repository_{source_id}_license_missing")
        if not isinstance(row.get("version_compatibility"), Mapping):
            errors.append(f"repository_{source_id}_version_compatibility_missing")
        path_rows = _rows(row.get("path_identities"))
        if not path_rows:
            errors.append(f"repository_{source_id}_path_identities_missing")
        for path_index, path_row in enumerate(path_rows):
            if not str(path_row.get("path") or "").strip():
                errors.append(f"repository_{source_id}_path_{path_index}_missing")
            if not _SHA.fullmatch(str(path_row.get("blob") or "")):
                errors.append(f"repository_{source_id}_path_{path_index}_blob_invalid")

    physx = repositories.get("physx")
    if physx is not None:
        absence = physx.get("skill_path_audit")
        if not isinstance(absence, Mapping):
            errors.append("physx_skill_path_audit_missing")
        else:
            if absence.get("skill_tree_match_count") != 0:
                errors.append("physx_skill_tree_match_count_not_zero")
            if absence.get("ovphysx_skill_path") is not None:
                errors.append("physx_ovphysx_skill_path_invented")
            if absence.get("decision") != "do_not_invent_ovphysx_skill_path":
                errors.append("physx_skill_absence_decision_invalid")

    guidance_rows = _rows(normalized.get("guidance_application"))
    guidance = {str(row.get("guidance_id") or ""): row for row in guidance_rows}
    if len(guidance) != len(guidance_rows):
        errors.append("guidance_id_duplicate_or_missing")
    if set(guidance) != REQUIRED_GUIDANCE:
        errors.append("guidance_set_invalid")
    for guidance_id, row in guidance.items():
        status = row.get("application_status")
        if status not in APPLICATION_STATUSES:
            errors.append(f"guidance_{guidance_id}_status_invalid")
        if not _nonempty_strings(row.get("decisions")):
            errors.append(f"guidance_{guidance_id}_decisions_missing")
        actually_followed = row.get("actually_followed")
        if not isinstance(actually_followed, bool):
            errors.append(f"guidance_{guidance_id}_actual_flag_invalid")
        if actually_followed and status not in {
            "followed_in_architecture",
            "followed_in_implementation",
        }:
            errors.append(f"guidance_{guidance_id}_actual_status_mismatch")
        if status == "followed_in_implementation" and not _nonempty_strings(
            row.get("implementation_evidence")
        ):
            errors.append(f"guidance_{guidance_id}_implementation_evidence_missing")

    runtime = normalized.get("runtime_compatibility_decision")
    if not isinstance(runtime, Mapping):
        errors.append("runtime_compatibility_decision_missing")
    else:
        if runtime.get("isaac_lab_runtime_commit") != (
            "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
        ):
            errors.append("runtime_isaac_lab_commit_invalid")
        if runtime.get("isaac_lab_arena_commit") != EXPECTED_REPOSITORIES[
            "isaac_lab_arena"
        ]["revision"]:
            errors.append("runtime_arena_commit_invalid")
        if runtime.get("isaac_sim_version_family") != "6.0.x":
            errors.append("runtime_isaac_sim_version_family_invalid")
        if runtime.get("isaac_sim_container") != (
            "nvcr.io/nvidia/isaac-sim:6.0.0-dev2@"
            "sha256:c3e7bef5b2bfdb9972807c34195206078372bf8c6cff79716be130a3fe3e9ce9"
        ):
            errors.append("runtime_isaac_sim_container_invalid")
        if runtime.get("latest_isaac_lab_develop_runtime_admitted") is not False:
            errors.append("runtime_incompatible_develop_not_rejected")
        if runtime.get("arena_claim_ceiling") != "alpha_internal_rehearsal_only":
            errors.append("runtime_arena_claim_ceiling_invalid")
        if runtime.get("ovrtx_selected") is not True:
            errors.append("runtime_ovrtx_selection_invalid")

    if not _nonempty_strings(normalized.get("claim_ceiling")):
        errors.append("claim_ceiling_missing")
    if normalized.get("audit_digest") != canonical_digest(
        normalized, digest_field="audit_digest"
    ):
        errors.append("audit_digest_mismatch")

    if errors:
        raise AgentSkillAuditError(errors)
    return normalized


def load_agent_skill_audit(path: str | Path) -> dict[str, Any]:
    """Load and validate a checked-in audit receipt."""

    audit_path = Path(path).expanduser().resolve()
    try:
        value = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentSkillAuditError(["audit_file_unreadable_or_invalid_json"]) from exc
    if not isinstance(value, Mapping):
        raise AgentSkillAuditError(["audit_not_mapping"])
    return validate_agent_skill_audit(value)
