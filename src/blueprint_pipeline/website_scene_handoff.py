"""Join prepared website captures to collected reconstruction assets.

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
        if clean_plate.get("source_geometry") is None:
            from .website_scene_geometry import run_website_scene_geometry
            from .website_task_masks import bind_task_masks_to_geometry
            video = Path(clean_plate.get("input_video_path") or "")
            if not video.is_file() or not video.resolve().is_relative_to(capture_root.resolve()):
                raise ValueError("website_source_video_outside_capture")
            result.pop("provider_mutation_performed", None)
            result["geometry_controller_invoked"] = True
            geometry = run_website_scene_geometry(source_video=video,
                output_root=Path(clean_plate["stage_manifest_path"]).parent / "source_geometry",
                capture_id=context["capture_id"], task_context=context)
            masks = bind_task_masks_to_geometry(task_masks=clean_plate["task_masks"], source_geometry=geometry)
            write_json(root / "task_masks.geometry.json", masks)
            clean_plate = {**clean_plate, "source_geometry": geometry, "task_masks": masks}
        removal_path = Path(clean_plate["removal_manifest_path"])
        if not removal_path.resolve().is_relative_to(capture_root.resolve() / "pipeline"):
            raise ValueError("website_scene_removal_manifest_outside_capture")
        # World Labs exports are OpenCV Y-down, including their GLB meshes:
        # https://docs.worldlabs.ai/marble/export/specs . The world manifest
        # declares an estimated metric factor and ground plane (raw units x
        # factor = metres; ground at y = offset after scaling), and the world
        # is generated from the prepared views in submission order, so the
        # first view's camera anchors registration. None of it is measured.
        semantics: dict[str, Any] = {}
        world_path = Path(assets.get("source_world_manifest") or root / "missing")
        if world_path.resolve().is_relative_to(capture_root.resolve() / "pipeline") and world_path.is_file():
            from .marble_sim_assets import _semantics_metadata
            world = json.loads(world_path.read_text())
            if world.get("world_id") == assets["world_id"]:
                semantics = _semantics_metadata(world)
        prepared = ((clean_plate.get("prepared_views") or (metadata.get("clean_plate") or {}).get("prepared_views")
                     or {}).get("frames") or [])
        declared = semantics.get("metric_scale_factor")
        base = {
            "splat_path": splat["local_path"], "splat_digest": "sha256:" + splat["sha256"],
            "splat_binding_id": "website-splat-" + splat["sha256"][:32],
            "collision_mesh_path": collision["local_path"], "collision_mesh_digest": "sha256:" + collision["sha256"],
            "collision_binding_id": "website-collider-" + collision["sha256"][:32],
            "up_axis": "-Y", "meters_per_unit": declared, "provider": "world_labs",
            "scale_authority": "provider_declared_estimate" if declared else "registration_estimate",
            "ground_plane_offset_m": semantics.get("ground_plane_offset") if declared else None,
            "anchor": ({"kind": "first_input_view_camera", "frame_id": prepared[0]["frame_id"]}
                       if declared and prepared else None),
            "operation_id": provider_run.get("provider_run_id"), "world_id": assets["world_id"],
        }
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
            from .website_native_background import prepare_collision_stage, prepare_appearance_stage, prepare_construction_stages
            collision_stage = prepare_collision_stage(root / "native" / "runtime_inputs.json")
            write_json(root / "native" / "collision_stage_inputs.json", collision_stage)
            result["runtime_inputs"]["collision_stage_inputs_path"] = str(root / "native" / "collision_stage_inputs.json")
            if runtime_inputs["appearance"]["status"] == "native_appearance_authored":
                appearance_stage = prepare_appearance_stage(root / "native" / "runtime_inputs.json")
                write_json(root / "native" / "appearance_stage_inputs.json", appearance_stage)
                result["runtime_inputs"]["appearance_stage_inputs_path"] = str(root / "native" / "appearance_stage_inputs.json")
                construction = prepare_construction_stages(runtime_inputs_path=root / "native" / "runtime_inputs.json",
                                                            preparation_path=root / "preparation.json")
                if preparation["status"] == "intake_ready":
                    from .website_native_background import construction_rights_admission
                    from .website_object_observations import _record
                    rights = construction_rights_admission(preparation=preparation, task_context=context, now=now)
                    rights_path = root / "native" / "rights_admission.json"
                    write_json(rights_path, rights)
                    construction["references"].append({"contract_path": "scene.rights.admission", **_record(rights_path)})
                write_json(root / "native" / "construction_inputs.json", construction)
                result["runtime_inputs"]["construction_inputs_path"] = str(root / "native" / "construction_inputs.json")
                if preparation["status"] == "intake_ready":
                    from .website_scene_dispatch import binding_root, register_website_preparation
                    context_path = root / "task_context.json"
                    write_json(context_path, context)
                    result["source_registration"] = register_website_preparation(
                        preparation_path=root / "preparation.json", runtime_inputs_path=root / "native/runtime_inputs.json",
                        task_context_path=context_path, root=binding_root(), now=now)
                    if authority.get("schema_version") == "website_scene_sponsorship.v1":
                        from .website_task_context import enqueue_website_prepared_scene
                        result["website_intake_outbox"] = enqueue_website_prepared_scene(
                            task_context=context, request=preparation["intake_request"])
        except (ValueError, KeyError, TypeError, OSError, ImportError) as exc:
            # Preserve finished CPU outputs when a later handoff is held.
            result.setdefault("runtime_inputs", {}).update(status="awaiting_inputs", blockers=[str(exc)])
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result.update(status="awaiting_inputs", blockers=[str(exc)])
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(root / "handoff.json", result)
    return result
