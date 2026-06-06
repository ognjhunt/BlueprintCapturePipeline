"""Final capture-to-GPU-handoff readiness manifest builder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .common import PipelineError, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context
from .provider_preview_qa import validate_provider_preview_packet


PRODUCTION_HANDOFF_READINESS_SCHEMA_VERSION = "production_handoff_readiness_manifest.v1"
OWNER_GPU_BLOCKER = "owner_gpu_simulator_execution_not_run"

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "capture_to_worldlabs_to_gpu_handoff_readiness_summary",
    "live_provider_calls_performed_by_this_validator": False,
    "remote_asset_downloads_performed_by_this_validator": False,
    "owner_gpu_simulator_execution_proven": False,
    "simulator_execution_proven": False,
    "robot_readiness_proven": False,
    "public_claim_upgrade_allowed": False,
}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def _string_list(value: Any) -> List[str]:
    if value is None:
        values: Iterable[Any] = []
    elif isinstance(value, str):
        values = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]
    out: List[str] = []
    seen: set[str] = set()
    for item in values:
        text = _string(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _append_unique(target: List[str], values: Iterable[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _artifact(path: Path) -> Dict[str, Any]:
    return {"path": str(path), "exists": path.is_file()}


def _status_blocker(status: str | None, *, expected: Sequence[str], blocker: str) -> List[str]:
    return [] if _string(status) in set(expected) else [blocker]


def build_production_handoff_readiness(
    *,
    capture_root: str | Path,
    mode: str = "production",
    run_provider_preview_qa: bool = True,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    automation_dir = pipeline_dir / "simulation_automation"
    generated_at = utc_now_iso()
    normalized_mode = _string(mode).lower() or "production"

    provider_qa: Dict[str, Any]
    if run_provider_preview_qa:
        provider_qa = validate_provider_preview_packet(
            capture_root=context.capture_root,
            mode=normalized_mode,
            require_webapp_sync=normalized_mode == "production",
        )
    else:
        provider_qa = _read_optional_mapping(pipeline_dir / "provider_preview_qa_manifest.json")

    worldlabs_request = _read_optional_mapping(pipeline_dir / "worldlabs_request_manifest.json")
    operation_manifest = _read_optional_mapping(pipeline_dir / "worldlabs_operation_manifest.json")
    world_manifest = _read_optional_mapping(pipeline_dir / "worldlabs_world_manifest.json")
    materialization_manifest = _read_optional_mapping(
        pipeline_dir / "worldlabs_assets" / "materialized_assets_manifest.json"
    )
    export_manifest = _read_optional_mapping(pipeline_dir / "worldlabs_export_manifest.json")
    marble_asset_manifest = _read_optional_mapping(
        pipeline_dir / "marble_sim_assets" / "marble_asset_manifest.json"
    )
    marble_bridge = _read_optional_mapping(
        pipeline_dir / "marble_sim_assets" / "marble_simready_bridge.json"
    )
    scene_inventory = _read_optional_mapping(automation_dir / "scene_asset_inventory.json")
    scene_preflight = _read_optional_mapping(automation_dir / "scene_asset_preflight.json")
    cpu_preflight = _read_optional_mapping(automation_dir / "cpu_preflight_manifest.json")
    gpu_handoff = _read_optional_mapping(automation_dir / "gpu_handoff_packet.json")
    gpu_proof_schema = _read_optional_mapping(automation_dir / "gpu_owner_system_proof_schema.json")
    owner_gpu_blocked = _read_optional_mapping(
        automation_dir / "owner_gpu_simulator_execution_blocked_manifest.json"
    )

    blockers: List[str] = []
    warnings: List[str] = []
    if not provider_qa:
        blockers.append("provider_preview_qa_missing")
    elif provider_qa.get("status") != "passed":
        _append_unique(blockers, _string_list(provider_qa.get("blockers")) or ["provider_preview_qa_blocked"])
    webapp_sync_projection = _mapping(provider_qa.get("webapp_sync_projection"))
    redaction_proof = _mapping(provider_qa.get("redaction_proof"))
    raw_path_policy = _mapping(provider_qa.get("raw_path_policy"))
    worldlabs_input_lineage = _mapping(provider_qa.get("worldlabs_input_lineage"))
    privacy_safe_worldlabs_input = bool(
        redaction_proof.get("privacy_completed")
        and raw_path_policy.get("privacy_safe_input")
        and worldlabs_input_lineage.get("audit_matches_request")
        and (
            worldlabs_input_lineage.get("source_is_final_walkthrough")
            or worldlabs_input_lineage.get("derivative_of_final_walkthrough")
        )
    )

    if not worldlabs_request:
        blockers.append("worldlabs_request_manifest_missing")
    elif not bool(worldlabs_request.get("privacy_safe_input")):
        blockers.append("worldlabs_request_not_privacy_safe")

    if not operation_manifest:
        blockers.append("worldlabs_operation_manifest_missing")
    if not world_manifest:
        blockers.append("worldlabs_world_manifest_missing")
    elif not _string(world_manifest.get("world_id") or world_manifest.get("id")):
        blockers.append("worldlabs_world_id_missing")

    if not materialization_manifest:
        blockers.append("worldlabs_asset_materialization_manifest_missing")
    else:
        _append_unique(
            blockers,
            _status_blocker(
                _string(materialization_manifest.get("status")),
                expected=("complete", "partial"),
                blocker="worldlabs_asset_materialization_not_complete",
            ),
        )
        if int(materialization_manifest.get("download_count") or 0) <= 0:
            blockers.append("worldlabs_materialized_asset_download_missing")
    if not export_manifest:
        blockers.append("worldlabs_export_manifest_missing")

    if not marble_asset_manifest:
        blockers.append("marble_asset_manifest_missing")
    if not marble_bridge:
        blockers.append("marble_simready_bridge_missing")

    if not scene_inventory:
        blockers.append("scene_asset_inventory_missing")
    elif int(scene_inventory.get("asset_count") or 0) <= 0:
        blockers.append("scene_asset_inventory_empty")
    if not scene_preflight:
        blockers.append("scene_asset_preflight_missing")

    if not cpu_preflight:
        blockers.append("cpu_preflight_manifest_missing")
    elif not bool(cpu_preflight.get("ready_for_owner_gpu_preflight")):
        blockers.append("cpu_preflight_not_ready_for_owner_gpu")

    if not gpu_handoff:
        blockers.append("gpu_handoff_packet_missing")
    else:
        if gpu_handoff.get("status") != "ready_for_owner_gpu_preflight_handoff":
            blockers.append("gpu_handoff_packet_not_ready")
        if bool(gpu_handoff.get("robot_readiness_proven")):
            blockers.append("gpu_handoff_illegally_marks_robot_readiness")
        handoff_blockers = _string_list(gpu_handoff.get("blockers"))
        if handoff_blockers != [OWNER_GPU_BLOCKER]:
            for blocker in handoff_blockers:
                if blocker != OWNER_GPU_BLOCKER:
                    blockers.append(blocker)
            if OWNER_GPU_BLOCKER not in handoff_blockers:
                blockers.append("gpu_handoff_missing_owner_gpu_blocker")
        else:
            warnings.append(OWNER_GPU_BLOCKER)

    if not gpu_proof_schema:
        blockers.append("gpu_owner_system_proof_schema_missing")
    if not owner_gpu_blocked:
        blockers.append("owner_gpu_blocked_manifest_missing")
    elif owner_gpu_blocked.get("blocker_id") != OWNER_GPU_BLOCKER:
        blockers.append("owner_gpu_blocked_manifest_wrong_blocker")

    unique_blockers: List[str] = []
    _append_unique(unique_blockers, blockers)
    non_owner_blockers = [item for item in unique_blockers if item != OWNER_GPU_BLOCKER]
    status = (
        "ready_except_owner_gpu_simulator_execution"
        if not non_owner_blockers
        else "blocked_before_owner_gpu_handoff"
    )
    remaining_unproven_steps = [OWNER_GPU_BLOCKER] if status.startswith("ready_") else unique_blockers

    manifest = {
        "schema_version": PRODUCTION_HANDOFF_READINESS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "mode": normalized_mode,
        "status": status,
        "owner_gpu_simulator_execution_is_only_unproven_step": status.startswith("ready_"),
        "remaining_unproven_steps": remaining_unproven_steps,
        "blockers": unique_blockers,
        "warnings": warnings,
        "proof_summary": {
            "privacy_safe_worldlabs_input": privacy_safe_worldlabs_input,
            "webapp_sync_succeeded": bool(webapp_sync_projection.get("sync_succeeded")),
            "webapp_upstream_links_verified": bool(
                webapp_sync_projection.get("upstream_links_verified")
            ),
            "worldlabs_generation_manifested": bool(operation_manifest and world_manifest),
            "worldlabs_assets_materialized": bool(materialization_manifest and export_manifest),
            "marble_sim_asset_handoff_manifested": bool(marble_asset_manifest and marble_bridge),
            "scene_asset_preflight_manifested": bool(scene_inventory and scene_preflight),
            "cpu_preflight_ready_for_owner_gpu": bool(
                cpu_preflight.get("ready_for_owner_gpu_preflight")
            ),
            "gpu_handoff_packet_ready": gpu_handoff.get("status")
            == "ready_for_owner_gpu_preflight_handoff",
            "owner_gpu_simulator_execution_proven": False,
            "robot_readiness_proven": False,
        },
        "artifacts": {
            "provider_preview_qa": _artifact(pipeline_dir / "provider_preview_qa_manifest.json"),
            "webapp_sync_result": _artifact(pipeline_dir / "webapp_sync_result.json"),
            "worldlabs_request_manifest": _artifact(pipeline_dir / "worldlabs_request_manifest.json"),
            "worldlabs_operation_manifest": _artifact(pipeline_dir / "worldlabs_operation_manifest.json"),
            "worldlabs_world_manifest": _artifact(pipeline_dir / "worldlabs_world_manifest.json"),
            "worldlabs_asset_materialization": _artifact(
                pipeline_dir / "worldlabs_assets" / "materialized_assets_manifest.json"
            ),
            "worldlabs_export_manifest": _artifact(pipeline_dir / "worldlabs_export_manifest.json"),
            "marble_asset_manifest": _artifact(
                pipeline_dir / "marble_sim_assets" / "marble_asset_manifest.json"
            ),
            "marble_simready_bridge": _artifact(
                pipeline_dir / "marble_sim_assets" / "marble_simready_bridge.json"
            ),
            "scene_asset_inventory": _artifact(automation_dir / "scene_asset_inventory.json"),
            "scene_asset_preflight": _artifact(automation_dir / "scene_asset_preflight.json"),
            "cpu_preflight_manifest": _artifact(automation_dir / "cpu_preflight_manifest.json"),
            "gpu_handoff_packet": _artifact(automation_dir / "gpu_handoff_packet.json"),
            "gpu_owner_system_proof_schema": _artifact(
                automation_dir / "gpu_owner_system_proof_schema.json"
            ),
            "owner_gpu_blocked_manifest": _artifact(
                automation_dir / "owner_gpu_simulator_execution_blocked_manifest.json"
            ),
        },
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(pipeline_dir / "production_handoff_readiness_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build final production handoff readiness manifest")
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--mode", choices=("production", "advisory"), default="production")
    parser.add_argument("--skip-provider-preview-qa", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_production_handoff_readiness(
            capture_root=args.capture_root,
            mode=args.mode,
            run_provider_preview_qa=not args.skip_provider_preview_qa,
        )
    except (PipelineError, ValueError) as exc:
        print(f"[production-handoff-readiness] status=error reason={exc}")
        return 2
    print(
        "[production-handoff-readiness] manifest="
        f"{Path(args.capture_root) / 'pipeline' / 'production_handoff_readiness_manifest.json'}"
    )
    print(f"[production-handoff-readiness] status={result['status']}")
    if result["status"] != "ready_except_owner_gpu_simulator_execution":
        print(f"[production-handoff-readiness] blockers={','.join(result.get('blockers') or [])}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
