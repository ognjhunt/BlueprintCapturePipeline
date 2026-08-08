"""Fail-closed selected-scene and articulated-task freeze for ADP v1.

The freeze is the boundary between outcome-blind public-scene discovery and
construction.  It binds exact rights/source/evidence bytes, one scene, one
task joint, locked non-task joints, deterministic scoring parameters, seeds,
and exactly two candidates.  It authorizes construction only; later native
gates own scenario materialization and evaluation admission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adp_task_scoring import (
    TASK_KIND_ARTICULATED_OPEN_CLOSE,
    validate_articulated_task_spec,
)
from .articulated_workspace_clearance import validate_sage_mesh_sweep
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "second_scene_scene_task_freeze.v1"
PROGRAM_ID = "arm-decision-proof-v1"
SOURCE_ROLES = {"interiorgs_splat", "interiorgs_labels", "interiorgs_structure", "sage_collision"}
EVIDENCE_ROLES = {
    "room_survey",
    "room_survey_render",
    "target_closeup_render",
    "target_collision_identity",
    "member_seam_observation",
    "sage_sweep_obstacle_inventory",
    "exact_sage_mesh_sweep",
}
FROZEN_CANDIDATES = ["pi05_droid", "groot_n17_droid"]


class SecondSceneFreezeError(ValueError):
    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _resolve_record(
    record: Mapping[str, Any], *, root: Path, role: str, verify_files: bool
) -> Path | None:
    relative = str(record.get("relative_path") or "")
    path = Path(relative)
    if not relative or path.is_absolute() or ".." in path.parts:
        raise SecondSceneFreezeError([f"freeze_{role}_path_invalid"])
    expected_sha = str(record.get("sha256") or "")
    expected_size = record.get("size_bytes")
    if not expected_sha.startswith("sha256:") or len(expected_sha) != 71 or (
        isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0
    ):
        raise SecondSceneFreezeError([f"freeze_{role}_record_invalid"])
    if not verify_files:
        return None
    resolved = (root / path).resolve()
    if root != resolved and root not in resolved.parents:
        raise SecondSceneFreezeError([f"freeze_{role}_path_escape"])
    if not resolved.is_file() or resolved.is_symlink():
        raise SecondSceneFreezeError([f"freeze_{role}_file_missing"])
    errors: list[str] = []
    if resolved.stat().st_size != expected_size:
        errors.append(f"freeze_{role}_size_mismatch")
    if _sha256(resolved) != expected_sha:
        errors.append(f"freeze_{role}_digest_mismatch")
    if errors:
        raise SecondSceneFreezeError(errors)
    return resolved


def validate_second_scene_freeze(
    value: Mapping[str, Any],
    *,
    repo_root: str | Path,
    data_root: str | Path,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate one retained freeze and, optionally, all referenced bytes."""

    payload = json.loads(json.dumps(value))
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("freeze_schema_invalid")
    if payload.get("program_id") != PROGRAM_ID:
        errors.append("freeze_program_invalid")
    if payload.get("status") != "frozen_for_construction_before_learned_outcomes":
        errors.append("freeze_status_invalid")
    scene = payload.get("scene")
    if not isinstance(scene, Mapping) or str(scene.get("publisher_scene_id") or "") in {"", "840313"}:
        errors.append("freeze_scene_identity_invalid")
    if payload.get("learned_policy_outcomes_consulted") is not False:
        errors.append("freeze_policy_outcome_leakage")
    if payload.get("new_inpainting_outcomes_consulted") is not False:
        errors.append("freeze_inpainting_outcome_leakage")
    if payload.get("scenario_materialization_authorized") is not False:
        errors.append("freeze_premature_scenario_authority")
    if payload.get("candidate_ids") != FROZEN_CANDIDATES:
        errors.append("freeze_candidate_pair_invalid")
    task = payload.get("task_spec")
    if not isinstance(task, Mapping):
        errors.append("freeze_task_spec_invalid")
    else:
        try:
            normalized_task = validate_articulated_task_spec(task)
        except ValueError as exc:
            errors.append(f"freeze_task_spec_invalid:{exc}")
        else:
            if task.get("task_kind") != TASK_KIND_ARTICULATED_OPEN_CLOSE:
                errors.append("freeze_task_kind_invalid")
            if set(normalized_task["joint_reset_positions_rad"]) != {
                "refrigerator_upper_door_hinge",
                "refrigerator_lower_door_hinge",
            }:
                errors.append("freeze_joint_set_invalid")
    seeds = payload.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        errors.append("freeze_seeds_invalid")
    rights = payload.get("rights")
    if not isinstance(rights, Mapping) or (
        rights.get("declared_use_scope") != "noncommercial_internal_research"
        or rights.get("raw_dataset_redistribution_allowed") is not False
        or rights.get("external_provider_upload_authorized") is not False
        or rights.get("commercial_use_allowed") is not False
    ):
        errors.append("freeze_rights_invalid")
    source_rows = payload.get("source_artifacts")
    if not isinstance(source_rows, list) or {str(row.get("role")) for row in source_rows if isinstance(row, Mapping)} != SOURCE_ROLES:
        errors.append("freeze_source_roles_invalid")
    evidence_rows = payload.get("evidence_artifacts")
    if not isinstance(evidence_rows, list) or {str(row.get("role")) for row in evidence_rows if isinstance(row, Mapping)} != EVIDENCE_ROLES:
        errors.append("freeze_evidence_roles_invalid")
    if payload.get("freeze_digest") != canonical_digest(payload, digest_field="freeze_digest"):
        errors.append("freeze_digest_invalid")
    if errors:
        raise SecondSceneFreezeError(errors)

    repo = Path(repo_root).expanduser().resolve()
    data = Path(data_root).expanduser().resolve()
    for row in source_rows:
        _resolve_record(row, root=data, role=str(row["role"]), verify_files=verify_files)
    loaded_evidence: dict[str, tuple[Mapping[str, Any], Path]] = {}
    for row in evidence_rows:
        role = str(row["role"])
        path = _resolve_record(row, root=data, role=role, verify_files=verify_files)
        if path is not None:
            try:
                parsed = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SecondSceneFreezeError([f"freeze_{role}_json_invalid"]) from exc
            if not isinstance(parsed, Mapping):
                raise SecondSceneFreezeError([f"freeze_{role}_json_invalid"])
            loaded_evidence[role] = (parsed, path)
    rights_record = payload.get("rights_authority_record")
    if not isinstance(rights_record, Mapping):
        raise SecondSceneFreezeError(["freeze_rights_authority_record_invalid"])
    rights_path = _resolve_record(
        rights_record, root=repo, role="rights_authority", verify_files=verify_files
    )
    if rights_path is not None:
        retained_rights = json.loads(rights_path.read_text(encoding="utf-8"))
        if (
            retained_rights.get("revision") != payload["rights"]["interiorgs_revision"]
            or retained_rights.get("reviewer_status") != "approved_for_declared_use"
            or retained_rights.get("raw_dataset_redistribution_allowed") is not False
            or retained_rights.get("commercial_use_allowed") is not False
        ):
            raise SecondSceneFreezeError(["freeze_rights_authority_mismatch"])
    if verify_files:
        sweep = loaded_evidence["exact_sage_mesh_sweep"][0]
        try:
            validate_sage_mesh_sweep(sweep)
        except ValueError as exc:
            raise SecondSceneFreezeError([f"freeze_exact_sweep_invalid:{exc}"]) from exc
        if sweep.get("status") != "exact_sage_mesh_clearance_candidate_only":
            raise SecondSceneFreezeError(["freeze_exact_sweep_not_clear"])
        seam = loaded_evidence["member_seam_observation"][0]
        if (
            seam.get("schema_version") != "articulated_horizontal_seam_observation.v1"
            or seam.get("receipt_digest") != canonical_digest(seam, digest_field="receipt_digest")
        ):
            raise SecondSceneFreezeError(["freeze_member_seam_invalid"])
        identity = loaded_evidence["target_collision_identity"][0]
        if (
            identity.get("receipt_digest") != canonical_digest(identity, digest_field="receipt_digest")
            or identity.get("whole_object_collision_identity_passed") is not True
            or len(identity.get("whole_object_matches") or []) != 1
        ):
            raise SecondSceneFreezeError(["freeze_collision_identity_invalid"])
    return payload


__all__ = [
    "EVIDENCE_ROLES",
    "FROZEN_CANDIDATES",
    "PROGRAM_ID",
    "SCHEMA_VERSION",
    "SOURCE_ROLES",
    "SecondSceneFreezeError",
    "validate_second_scene_freeze",
]
