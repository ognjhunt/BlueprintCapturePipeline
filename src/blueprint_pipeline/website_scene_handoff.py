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
    handoff_path = root / "handoff.json"
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
        context = metadata.get("site_task_context") or {}
        if context.get("capture_id") != descriptor["capture_id"] or context.get("scene_id") != descriptor["scene_id"]:
            raise ValueError("website_scene_task_identity_mismatch")
        from .website_development_test import enabled
        failed_fixture = (provider_run.get("status") == "failed"
            and provider_run.get("operation_terminal_status") == "failed"
            and enabled(context["context_digest"]))
        if failed_fixture:
            operation_path = Path(provider_run.get("worldlabs_operation_manifest_uri") or root / "missing")
            if (not operation_path.resolve().is_relative_to(capture_root.resolve() / "pipeline")
                    or not operation_path.is_file()):
                raise ValueError("website_reconstruction_failure_evidence_missing")
            operation = json.loads(operation_path.read_text())
            if (operation.get("done") is not True or not operation.get("error")
                    or operation.get("operation_id") != provider_run.get("provider_run_id")):
                raise ValueError("website_reconstruction_failure_evidence_invalid")
            result["captured_scene_reconstruction"] = {"status": "failed", "operation_id": operation["operation_id"],
                "operation_digest": canonical_digest(operation), "claim_ceiling": "development_only"}
            assets = None
        else:
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
        from .website_worldlabs import settle_website_reconstruction
        settlement = settle_website_reconstruction(provider_run=provider_run, capture_root=capture_root,
                                                    task_context=context)
        if settlement is not None:
            result["reconstruction_settlement"] = settlement
        if clean_plate.get("source_geometry") is None:
            from .website_scene_geometry import run_website_scene_geometry
            from .website_task_masks import bind_task_masks_to_geometry, complete_retained_static_task_masks
            video = Path(clean_plate.get("input_video_path") or "")
            if not video.is_file() or not video.resolve().is_relative_to(capture_root.resolve()):
                raise ValueError("website_source_video_outside_capture")
            result.pop("provider_mutation_performed", None)
            if (clean_plate.get("task_masks") or {}).get("deferred_target_ids"):
                result["visual_reconstruction_ready"] = True
                result["deferred_masks_controller_invoked"] = True
                plan_path = Path(clean_plate["removal_plan_path"])
                if not plan_path.resolve().is_relative_to(capture_root.resolve() / "pipeline"):
                    raise ValueError("website_scene_removal_plan_outside_capture")
                masks = complete_retained_static_task_masks(plan=json.loads(plan_path.read_text()),
                    source_geometry=clean_plate["source_frames"], task_masks=clean_plate["task_masks"],
                    output_root=plan_path.parent / "task_masks",
                    view_plan_root=plan_path.parent / "mask_view_plan")
                clean_plate = {**clean_plate, "task_masks": masks}
            result["geometry_controller_invoked"] = True
            geometry = run_website_scene_geometry(source_video=video,
                output_root=Path(clean_plate["stage_manifest_path"]).parent / "source_geometry",
                capture_id=context["capture_id"], task_context=context, task_masks=clean_plate["task_masks"])
            masks = bind_task_masks_to_geometry(task_masks=clean_plate["task_masks"], source_geometry=geometry)
            write_json(root / "task_masks.geometry.json", masks)
            clean_plate = {**clean_plate, "source_geometry": geometry, "task_masks": masks}
        removal_path = Path(clean_plate["removal_manifest_path"])
        if not removal_path.resolve().is_relative_to(capture_root.resolve() / "pipeline"):
            raise ValueError("website_scene_removal_manifest_outside_capture")
        removal = json.loads(removal_path.read_text())
        # Each articulated task target needs views of its whole assembly from
        # the full track, not only the geometry frames. Retained receipts and
        # decoded frames make a restart free.
        from .website_assembly_coverage import attach_assembly_coverage
        source_video = Path(clean_plate.get("input_video_path") or "")
        masks = attach_assembly_coverage(task_masks=clean_plate["task_masks"],
            source_geometry=clean_plate["source_geometry"], removal_manifest=removal, task_context=context,
            source_video=(source_video if source_video.is_file()
                          and source_video.resolve().is_relative_to(capture_root.resolve()) else None),
            output_root=root / "assembly_coverage")
        if masks != clean_plate["task_masks"]:
            write_json(root / "task_masks.coverage.json", masks)
            clean_plate = {**clean_plate, "task_masks": masks}
        # World Labs exports are OpenCV Y-down, including their GLB meshes:
        # https://docs.worldlabs.ai/marble/export/specs . The world manifest
        # declares an estimated metric factor and ground plane (raw units x
        # factor = metres; ground at y = offset after scaling), and the world
        # is generated from the prepared views in submission order, so the
        # first view's camera anchors registration. None of it is measured.
        if failed_fixture:
            import trimesh
            seed_path = root / "development_fixture_seed.glb"
            if not seed_path.is_file():
                seed_path.write_bytes(trimesh.creation.box(extents=(1.0, 1.0, 0.1)).export(file_type="glb"))
            seed_digest = _sha256_file(seed_path)
            base = {"mode": "development_fixture_seed", "provider": "blueprint_authored_development_seed",
                "reconstruction_blocker": "website_reconstruction_failed", "collision_mesh_path": str(seed_path),
                "collision_mesh_digest": seed_digest, "collision_binding_id": "website-development-seed-" + seed_digest[7:39],
                "up_axis": "Z", "meters_per_unit": 1.0, "scale_authority": "model_estimated_object_frame",
                "operation_id": provider_run.get("provider_run_id"), "physical_registration_proven": False}
        else:
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
            removal_manifest=removal, source_geometry=clean_plate["source_geometry"],
            base_scene=base, output_root=root, spend=authority, now=now,
        )
        result.update(status=preparation["status"], blockers=preparation["blockers"],
                      preparation_path=str(root / "preparation.json"), preparation_digest=preparation["digest"],
                      thumbnail=preparation["thumbnail"], base_scene_path=str(root / "base_scene.json"))
        # A missing execution authority must not stop this CPU-only geometry
        # conversion. The object stays in its separate authoring lane.
        try:
            from .website_development_test import enabled, prepare_development_test
            development = enabled(context["context_digest"])
            if development:
                root = root / "development_test"
                preparation, runtime_inputs = prepare_development_test(preparation=preparation,
                    source_geometry=clean_plate["source_geometry"], task_masks=clean_plate["task_masks"],
                    output_root=root)
                result["development_test"] = {**preparation["development_test"],
                    "preparation_path": str(root / "preparation.json")}
                native_root = root
            else:
                native_root = root / "native"
                runtime_inputs = prepare_website_runtime_inputs(preparation=preparation, base_scene=base,
                    source_geometry=clean_plate["source_geometry"], task_masks=clean_plate["task_masks"],
                    output_root=native_root)
            result["runtime_inputs"] = {"status": runtime_inputs["status"], "digest": runtime_inputs["digest"],
                                        "path": str(native_root / "runtime_inputs.json")}
            from .website_native_background import prepare_collision_stage, prepare_appearance_stage, prepare_construction_stages
            collision_stage = prepare_collision_stage(native_root / "runtime_inputs.json")
            write_json(native_root / "collision_stage_inputs.json", collision_stage)
            result["runtime_inputs"]["collision_stage_inputs_path"] = str(native_root / "collision_stage_inputs.json")
            if runtime_inputs["appearance"]["status"] == "native_appearance_authored":
                appearance_stage = prepare_appearance_stage(native_root / "runtime_inputs.json")
                write_json(native_root / "appearance_stage_inputs.json", appearance_stage)
                result["runtime_inputs"]["appearance_stage_inputs_path"] = str(native_root / "appearance_stage_inputs.json")
                construction = prepare_construction_stages(runtime_inputs_path=native_root / "runtime_inputs.json",
                                                            preparation_path=root / "preparation.json")
                if preparation["status"] == "intake_ready":
                    from .website_native_background import construction_rights_admission
                    from .website_object_observations import _record
                    rights = construction_rights_admission(preparation=preparation, task_context=context, now=now)
                    rights_path = native_root / "rights_admission.json"
                    write_json(rights_path, rights)
                    construction["references"].append({"contract_path": "scene.rights.admission", **_record(rights_path)})
                write_json(native_root / "construction_inputs.json", construction)
                result["runtime_inputs"]["construction_inputs_path"] = str(native_root / "construction_inputs.json")
                if preparation["status"] == "intake_ready":
                    from .website_scene_dispatch import binding_root, register_website_preparation
                    context_path = root / "task_context.json"
                    write_json(context_path, context)
                    result["source_registration"] = register_website_preparation(
                        preparation_path=root / "preparation.json", runtime_inputs_path=native_root / "runtime_inputs.json",
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
    write_json(handoff_path, result)
    return result
