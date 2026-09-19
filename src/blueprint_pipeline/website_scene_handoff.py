"""Join prepared website captures to collected reconstruction assets, without spend.

ADP-030/040, day 28: retain the task placement and authoring inputs as soon as
the world finishes. A held provider or missing execution authority must not
discard completed preparation or label the scene simulator-ready.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .local_reconstruction_adapters import _sha256_file
from .website_task_preparation import compile_website_scene_preparation
from .website_scene_runtime_inputs import prepare_website_runtime_inputs


def prepare_website_scene_handoff(*, descriptor: Mapping[str, Any], clean_plate: Mapping[str, Any],
                                 provider_run: Mapping[str, Any], capture_root: Path,
                                 now: float) -> dict[str, Any]:
    root = capture_root / "pipeline" / "website_scene_preparation"
    root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": "website_scene_handoff.v1", "status": "awaiting_reconstruction",
        "blockers": [], "claim_ceiling": "development_only", "simulator_ready": False,
        "provider_mutation_performed": False,
    }
    metadata = descriptor.get("metadata") or {}
    result["capture_id"] = descriptor["capture_id"]
    result["scene_id"] = descriptor["scene_id"]
    try:
        if clean_plate.get("privacy_verified") is not True or clean_plate.get("status") not in {"noop", "objects_removed"}:
            raise ValueError("website_scene_preparation_pending")
        if provider_run.get("status") not in {"ready", "completed"}:
            raise ValueError("website_reconstruction_pending")
        path = Path((provider_run.get("worldlabs_asset_materialization") or {}).get("manifest_path") or root / "missing")
        if not path.resolve().is_relative_to(capture_root.resolve() / "pipeline") or not path.is_file():
            raise ValueError("website_reconstruction_assets_pending")
        assets = json.loads(path.read_text())
        if assets.get("world_id") != provider_run.get("world_id"):
            raise ValueError("website_reconstruction_world_mismatch")
        rows = assets.get("downloads") or []
        collision = next((row for row in rows if row.get("kind") == "collider_mesh_glb"), None)
        splat = next((row for row in rows if row.get("kind") in {"splat_ply", "splat_spz"}), None)
        if collision is None or splat is None:
            raise ValueError("website_reconstruction_assets_pending")
        for row in (collision, splat):
            artifact = Path(row["local_path"])
            if (not artifact.resolve().is_relative_to(capture_root.resolve() / "pipeline")
                    or not artifact.is_file() or _sha256_file(artifact) != "sha256:" + row["sha256"]):
                raise ValueError("website_reconstruction_asset_changed")
        context = metadata.get("site_task_context") or {}
        if context.get("capture_id") != descriptor["capture_id"] or context.get("scene_id") != descriptor["scene_id"]:
            raise ValueError("website_scene_task_identity_mismatch")
        removal_path = Path(clean_plate["removal_manifest_path"])
        if not removal_path.resolve().is_relative_to(capture_root.resolve() / "pipeline"):
            raise ValueError("website_scene_removal_manifest_outside_capture")
        base = {
            "splat_path": splat["local_path"], "splat_digest": "sha256:" + splat["sha256"],
            "splat_binding_id": "website-splat-" + splat["sha256"][:32],
            "collision_mesh_path": collision["local_path"], "collision_mesh_digest": "sha256:" + collision["sha256"],
            "collision_binding_id": "website-collider-" + collision["sha256"][:32],
            "up_axis": "Y", "meters_per_unit": None, "provider": "world_labs",
            "operation_id": provider_run.get("provider_run_id"), "world_id": assets["world_id"],
        }
        # GLB is Y-up. Source-to-collider registration supplies the estimated
        # metric scale; splat decoding/frame qualification remains downstream.
        write_json(root / "base_scene.json", base)
        authority = metadata.get("website_scene_execution_authority") or {}
        if not all(key in authority for key in ("max_total_spend_usd", "max_paid_attempts", "expires_at_epoch")):
            # Compile geometry even while execution is held. Zero authority
            # cannot validate as a paid intake request.
            authority = {"max_total_spend_usd": 0, "max_paid_attempts": 0, "expires_at_epoch": now}
        preparation = compile_website_scene_preparation(
            task_context=context, task_masks=clean_plate["task_masks"],
            removal_manifest=json.loads(removal_path.read_text()), source_geometry=clean_plate["source_geometry"],
            base_scene=base, output_root=root, spend=authority, now=now,
        )
        result.update(status=preparation["status"], blockers=preparation["blockers"],
                      preparation_path=str(root / "preparation.json"), preparation_digest=preparation["digest"],
                      thumbnail=preparation["thumbnail"], base_scene_path=str(root / "base_scene.json"))
        # A missing execution authority must not stop this CPU-only geometry
        # conversion. The object stays in its separate authoring lane.
        try:
            runtime_inputs = prepare_website_runtime_inputs(preparation=preparation, base_scene=base,
                source_geometry=clean_plate["source_geometry"], task_masks=clean_plate["task_masks"],
                output_root=root / "native")
            result["runtime_inputs"] = {"status": runtime_inputs["status"], "digest": runtime_inputs["digest"],
                                        "path": str(root / "native" / "runtime_inputs.json")}
            from .website_native_background import prepare_collision_stage, prepare_appearance_stage
            collision_stage = prepare_collision_stage(root / "native" / "runtime_inputs.json")
            write_json(root / "native" / "collision_stage_inputs.json", collision_stage)
            result["runtime_inputs"]["collision_stage_inputs_path"] = str(root / "native" / "collision_stage_inputs.json")
            if runtime_inputs["appearance"]["status"] == "native_appearance_authored":
                appearance_stage = prepare_appearance_stage(root / "native" / "runtime_inputs.json")
                write_json(root / "native" / "appearance_stage_inputs.json", appearance_stage)
                result["runtime_inputs"]["appearance_stage_inputs_path"] = str(root / "native" / "appearance_stage_inputs.json")
        except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
            result["runtime_inputs"] = {"status": "awaiting_inputs", "blockers": [str(exc)]}
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result.update(status="awaiting_inputs", blockers=[str(exc)])
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(root / "handoff.json", result)
    return result
