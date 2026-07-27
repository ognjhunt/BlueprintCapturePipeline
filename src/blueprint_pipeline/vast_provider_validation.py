"""Final evidence validation shared by Vast adapter execution paths."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import write_json


SCHEMA_VERSION = "vast_final_validation.v1"
REQUIRED_PHASES = (
    "vast_docs_checked",
    "vast_secret_file_verified",
    "vast_offer_search_started",
    "vast_offer_selected",
    "vast_instance_create_requested",
    "vast_instance_started_or_blocked",
    "vast_heartbeat_started",
    "vast_heartbeat_completed_or_blocked",
    "vast_gpu_sanity_started",
    "vast_gpu_sanity_completed_or_blocked",
    "vast_isaac_smoke_started",
    "vast_isaac_smoke_completed_or_blocked",
    "vast_blueprint_bundle_started",
    "vast_blueprint_bundle_completed_or_blocked",
    "vast_artifacts_exported",
    "vast_instance_teardown_started",
    "vast_instance_teardown_completed",
)
REQUIRED_ARTIFACTS = (
    "vast_runtime_discovery.json",
    "vast_provider_plan.json",
    "vast_offer_selection_manifest.json",
    "vast_budget_ledger.json",
    "vast_runtime_phase_log.jsonl",
    "vast_startup_probe_manifest.json",
    "vast_gpu_sanity_report.json",
    "vast_isaac_smoke_result.json",
    "vast_provider_command_result.json",
    "vast_video_smoke_result.json",
    "vast_teardown_manifest.json",
)


def _truth_boundaries() -> dict[str, Any]:
    return {
        "isaac_sim_does_not_make_spz_or_3dgs_physical": True,
        "direct_splat_collision_proven": False,
        "collider_source_if_used": "metadata_derived_collider_proxy_required",
        "splat_visuals_if_used": "splat_rendered_visual_evidence_synchronized_with_isaac_state",
        "real_wam_vla_runtime_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "generated_world_policy_evaluation_scope_proven": False,
        "dexterous_hand_policy_proven": False,
        "controller_grade_execution_proven": False,
        "official_policy_execution_proven": False,
        "rank_fidelity_result_proven": False,
        "public_claim_upgrade_allowed": False,
    }


def final_validation(
    *,
    job_dir: Path,
    generated_at: str,
    instance_ids: Sequence[int],
    continuing_spend: bool,
    estimated_cost_usd: float,
    hard_cap_usd: float,
    authorized_retention: bool = False,
) -> dict[str, Any]:
    missing = [name for name in REQUIRED_ARTIFACTS if not (job_dir / name).exists()]
    json_errors: list[str] = []
    for path in job_dir.glob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - every invalid artifact must be listed
            json_errors.append(f"{path.name}:{type(exc).__name__}")
    phase_rows: list[dict[str, Any]] = []
    phase_path = job_dir / "vast_runtime_phase_log.jsonl"
    if phase_path.exists():
        for line_number, line in enumerate(phase_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except Exception as exc:  # noqa: BLE001 - every invalid row must be listed
                json_errors.append(f"{phase_path.name}:{line_number}:{type(exc).__name__}")
                continue
            if isinstance(parsed, Mapping):
                phase_rows.append(dict(parsed))
    observed_phases = {str(row.get("phase")) for row in phase_rows}
    missing_phases = [phase for phase in REQUIRED_PHASES if phase not in observed_phases]
    blockers: list[str] = []
    if missing:
        blockers.append("missing_required_vast_artifacts")
    if json_errors:
        blockers.append("json_parse_errors")
    if missing_phases:
        blockers.append("missing_required_vast_runtime_phases")
    if continuing_spend and not authorized_retention:
        blockers.append("continuing_vast_spend_detected")
    if estimated_cost_usd > hard_cap_usd:
        blockers.append("vast_estimated_spend_exceeded_hard_cap")
    video_smoke: Mapping[str, Any] = {}
    video_path = job_dir / "vast_video_smoke_result.json"
    if video_path.is_file():
        try:
            parsed_video = json.loads(video_path.read_text(encoding="utf-8"))
            video_smoke = parsed_video if isinstance(parsed_video, Mapping) else {}
        except Exception:  # noqa: BLE001 - already reported through json_errors
            pass
    validation = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed" if not blockers else "blocked",
        "job_dir": str(job_dir),
        "required_artifacts": list(REQUIRED_ARTIFACTS),
        "missing_required_artifacts": missing,
        "json_parse_errors": json_errors,
        "required_phases": list(REQUIRED_PHASES),
        "missing_required_phases": missing_phases,
        "vast_instance_ids": list(instance_ids),
        "estimated_cost_usd": estimated_cost_usd,
        "spend_hard_cap_usd": hard_cap_usd,
        "continuing_spend_from_this_run": continuing_spend,
        "all_vast_instances_destroyed_by_adapter": not continuing_spend,
        "authorized_watchdog_owned_retention": authorized_retention,
        "video_smoke_proven": video_smoke.get("video_smoke_proven") is True,
        "video_smoke_status": video_smoke.get("status"),
        "raw_secret_values_recorded": False,
        "blockers": blockers,
        **_truth_boundaries(),
    }
    write_json(job_dir / "vast_final_validation.json", validation)
    return validation


__all__ = ["REQUIRED_ARTIFACTS", "REQUIRED_PHASES", "SCHEMA_VERSION", "final_validation"]
