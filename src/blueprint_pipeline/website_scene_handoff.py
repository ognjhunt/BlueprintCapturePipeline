"""Join prepared website captures to the scene their Atlas job reconstructed.

ADP-030/040, day 28: retain the task placement and authoring inputs as soon as
the Atlas job finishes. A held job or missing execution authority must not
discard completed preparation or label the scene simulator-ready.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .website_task_preparation import (
    compile_website_scene_preparation, native_task_preparation_gap, native_task_subject_profile,
)
from .website_scene_runtime_inputs import prepare_website_runtime_inputs


def execute_prepared_task_family(*, preparation_path: Path, runtime: Any, policy: Any,
                                output_dir: Path, episode_id: str,
                                maximum_steps: int, control_frequency_hz: float) -> dict[str, Any]:
    """Consume the compiled family route using an already admitted native runtime.

    The operator transfers runtime ownership through its existing execution
    lane. Every preflight refusal closes it; the episode executor owns closure
    once called. This function buys nothing and cannot admit a provider.
    Runtime readback must bind the exact compiled scene, not only its engine.
    """
    executor_owns_runtime = False
    try:
        preparation = json.loads(preparation_path.read_text())
        if (preparation.get("digest") != canonical_digest(preparation, digest_field="digest")
                or preparation.get("status") != "task_evaluation_ready" or preparation.get("blockers") != []):
            raise ValueError("website_task_family_preparation_not_ready")
        inputs = preparation["task_evaluation_inputs"]
        from .task_evaluation_task_family_native_adapter import (
            execute_task_family_episode, validate_task_family_native_runtime_contract,
        )
        native = validate_task_family_native_runtime_contract(inputs["native_runtime_contract"])
        from .task_family_preparation import family_source_blockers
        if family_source_blockers(native=native, source_bindings=inputs["source_bindings"], source_scene=inputs["source_scene"]):
            raise ValueError("website_task_family_source_binding_incomplete")
        if (getattr(runtime, "native_runtime_contract_digest", None) != native["contract_digest"]
                or native["task_spec"]["task_success_contract"] != inputs["task_success_contract"]
                or inputs["scope"]["task_context_digest"] != preparation["binding"]["task_context_digest"]):
            raise ValueError("website_task_family_runtime_scene_mismatch")
        from .local_reconstruction_adapters import _sha256_file
        source = inputs["source_scene"]
        for key in ("splat", "collision_mesh"):
            if _sha256_file(Path(source[key + "_path"])) != source[key + "_digest"]:
                raise ValueError("website_task_family_source_scene_changed")
        executor_owns_runtime = True
        return execute_task_family_episode(runtime=runtime, policy=policy, contract=inputs["task_success_contract"],
            native_runtime_contract=native,
            maximum_steps=maximum_steps, control_frequency_hz=control_frequency_hz,
            output_dir=output_dir, episode_id=episode_id, seed=native["seed"])
    finally:
        if not executor_owns_runtime:
            runtime.close()


def prepare_website_scene_handoff(*, descriptor: Mapping[str, Any], clean_plate: Mapping[str, Any],
                                 capture_root: Path, now: float) -> dict[str, Any]:
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

    def retain_gap(gap: Mapping[str, Any]) -> dict[str, Any]:
        result.update(status="needs_input", blockers=[gap["code"]], native_task_preparation_gap=dict(gap))
        result["digest"] = canonical_digest(result, digest_field="digest")
        write_json(handoff_path, result)
        return result

    try:
        if clean_plate.get("privacy_verified") is not True or clean_plate.get("status") not in {"noop", "objects_removed"}:
            raise ValueError("website_scene_preparation_pending")
        context = metadata.get("site_task_context") or {}
        if context.get("capture_id") != descriptor["capture_id"] or context.get("scene_id") != descriptor["scene_id"]:
            raise ValueError("website_scene_task_identity_mismatch")
        # The capture's Atlas job is its one paid reconstruction lane: its
        # object-removed background is the scene and its source pose the
        # geometry. Nothing here buys anything.
        from .website_atlas_scene_source import atlas_scene_source
        atlas = atlas_scene_source(capture_root=capture_root, output_root=root / "atlas")
        if atlas["status"] != "ready":
            result.update(status="awaiting_reconstruction" if atlas["status"] == "pending" else "needs_input",
                blockers=[atlas["reason"]], geometry_backend=None, provider_mutation_performed=False,
                visual_reconstruction_ready=False,
                atlas_scene_source={key: atlas.get(key) for key in ("status", "reason", "stage", "job_reason")},
                native_geometry_requirements={
                    "source_aligned_depth_and_camera_poses": "required_for_mask_lifting_and_scene_registration",
                    "measured_scale": "unknown", "atlas_required": True})
            result["digest"] = canonical_digest(result, digest_field="digest")
            write_json(handoff_path, result)
            return result
        base = dict(atlas["base_scene"])
        write_json(root / "base_scene.json", base)
        result.update(base_scene_path=str(root / "base_scene.json"), visual_reconstruction_ready=True)
        removal_path = Path(clean_plate["removal_manifest_path"])
        if not removal_path.resolve().is_relative_to(capture_root.resolve() / "pipeline"):
            raise ValueError("website_scene_removal_manifest_outside_capture")
        removal = json.loads(removal_path.read_text())
        # Capture truth applies to held families too. Reading this retained
        # manifest performs no mask, authoring or provider work.
        if (atlas.get("subject_origin") == "synthetic_design") != ("synthetic_subject" in removal):
            raise ValueError("website_subject_origin_conflict")
        gap = native_task_preparation_gap(context)
        if gap is not None:
            # Retain the completed reconstruction and original task. Missing
            # adapter/scoring support cannot turn a different job into pick/place.
            result.update(geometry_backend=atlas["source_geometry"]["geometry_backend"],
                          source_geometry_digest=atlas["source_geometry"]["digest"])
            return retain_gap(gap)
        from .website_task_masks import bind_task_masks_to_geometry, complete_retained_static_task_masks
        from .task_family_scoring import canonical_task_family
        family = canonical_task_family(context.get("task_family"))
        result.pop("provider_mutation_performed", None)
        if family is None and (clean_plate.get("task_masks") or {}).get("deferred_target_ids"):
            result["deferred_masks_controller_invoked"] = True
            plan_path = Path(clean_plate["removal_plan_path"])
            if not plan_path.resolve().is_relative_to(capture_root.resolve() / "pipeline"):
                raise ValueError("website_scene_removal_plan_outside_capture")
            masks = complete_retained_static_task_masks(plan=json.loads(plan_path.read_text()),
                source_geometry=clean_plate["source_frames"], task_masks=clean_plate["task_masks"],
                output_root=plan_path.parent / "task_masks",
                view_plan_root=plan_path.parent / "mask_view_plan")
            clean_plate = {**clean_plate, "task_masks": masks}
        geometry = atlas["source_geometry"]
        result.update(geometry_backend=geometry["geometry_backend"],
                      atlas_scene_source={key: atlas[key] for key in (
                          "status", "source_pose_digest", "background_pose_digest", "alignment")})
        masks = bind_task_masks_to_geometry(task_masks=clean_plate["task_masks"], source_geometry=geometry)
        write_json(root / "task_masks.geometry.json", masks)
        clean_plate = {**clean_plate, "source_geometry": geometry, "task_masks": masks}
        if family is not None:
            # Family tasks compile all retained targets and their native
            # measurements. The single rigid-subject authoring path cannot
            # represent fluid, residue, cloth or an ordered set of operations.
            preparation = compile_website_scene_preparation(task_context=context, task_masks=masks,
                removal_manifest=removal, source_geometry=geometry, base_scene=base, output_root=root,
                spend=metadata.get("website_scene_execution_authority") or {}, now=now,
                evaluation_inputs=metadata.get("native_task_evaluation"))
            result.update(status=preparation["status"], blockers=preparation["blockers"],
                job_disposition="accepted", task_evaluation_inputs=preparation["task_evaluation_inputs"],
                preparation_path=str(root / "preparation.json"), preparation_digest=preparation["digest"],
                source_geometry_digest=geometry["digest"], provider_mutation_performed=False)
            result["digest"] = canonical_digest(result, digest_field="digest")
            write_json(handoff_path, result)
            return result
        targets = {row["target_id"]: row for row in masks.get("targets", [])}
        subjects = [row for row in removal.get("entries", []) if row.get("task_effect") == "manipulated"]
        synthetic = removal.get("synthetic_subject")
        inferred = None
        if synthetic is not None:
            if (not isinstance(synthetic, Mapping) or subjects
                    or synthetic.get("start_support_target_id") not in targets):
                raise ValueError("website_synthetic_subject_binding_invalid")
            _, inferred = native_task_subject_profile({}, targets[synthetic["start_support_target_id"]],
                                                      synthetic_subject=synthetic)
        elif len(subjects) == 1 and subjects[0].get("target_id") in targets:
            _, inferred = native_task_subject_profile(subjects[0], targets[subjects[0]["target_id"]])
        if inferred is not None:
            gap = native_task_preparation_gap(context, inferred_profile_id=inferred)
            if gap is not None:
                # Catch a contradictory profile before coverage/spec callbacks.
                # Earlier mask work may have called a provider, so do not reset
                # provider_mutation_performed to false on this later path.
                result["source_geometry_digest"] = geometry["digest"]
                return retain_gap(gap)
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
        # Published size and mass of an identified object reach compile before
        # the builder. An unknown identity or closed agent gate records not_run
        # and spends nothing; retained receipts make restarts free.
        from .website_object_spec_research import attach_object_specs
        masks = attach_object_specs(task_masks=masks, removal_manifest=removal, task_context=context,
                                    output_root=root / "object_spec")
        if masks != clean_plate["task_masks"]:
            write_json(root / "task_masks.coverage.json", masks)
            clean_plate = {**clean_plate, "task_masks": masks}
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
        if isinstance(preparation.get("native_task_preparation_gap"), Mapping):
            # Later source-bound evidence can also reveal a mismatch. A held
            # compiler output has no native intake and must not enter runtime preparation.
            return retain_gap(preparation["native_task_preparation_gap"])
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
                if (preparation["status"] == "intake_ready"
                        and preparation["intake_request"]["task"]["strategy"] == "pick_and_place"):
                    # Abstain before the owner is asked to pay when the arm cannot carry the task.
                    from .website_rigid_feasibility import rigid_task_verdict
                    verdict = rigid_task_verdict(preparation=preparation, collision_path=Path(runtime_inputs["collision"]["path"]))
                    write_json(native_root / "capability_verdict.json", verdict)
                    result["capability_verdict"] = {"status": verdict["status"], "blockers": verdict["blockers"],
                        "path": str(native_root / "capability_verdict.json"), "digest": verdict["verdict_digest"]}
                    if verdict["status"] != "clear":
                        result.update(status="needs_input", blockers=list(verdict["blockers"]))
                if preparation["status"] == "intake_ready" and result.get("capability_verdict", {}).get("status") != "abstained":
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
