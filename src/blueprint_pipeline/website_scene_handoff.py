"""Join prepared website captures to the scene their Atlas job reconstructed.

ADP-030/040, day 28: retain the task placement and authoring inputs as soon as
the Atlas job finishes. A held job or missing execution authority must not
discard completed preparation or label the scene simulator-ready.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core.handoff_lease import (
    HandoffLeaseError, guarded_handoff_write_json as write_json, guarded_handoff_write_track_json as write_track_json,
    handoff_effect, handoff_lease_checkpoint,
)
from .decision_evidence_contracts import canonical_digest
from .website_task_preparation import (
    GEOMETRY_ONLY_BLOCKER, compile_website_scene_preparation, native_task_preparation_gap,
    native_task_subject_gap, native_task_subject_profile, subject_rigidity,
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
        # ``source_targets`` carries every target's full-video source track: read and hashed compact.
        from .website_track_mask_codec import canonical_track_digest, load_track_json
        preparation = load_track_json(preparation_path)
        if (preparation.get("digest") != canonical_track_digest(preparation, digest_field="digest")
                or preparation.get("status") not in {"task_evaluation_ready", "intake_ready"} or preparation.get("blockers") != []):
            raise ValueError("website_task_family_preparation_not_ready")
        inputs = preparation["task_evaluation_inputs"]
        from .task_evaluation_task_family_native_adapter import (
            execute_task_family_episode, validate_task_family_native_runtime_contract,
        )
        native = validate_task_family_native_runtime_contract(inputs["native_runtime_contract"])
        from .task_family_native_contracts import family_source_blockers
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


def _robot_fit_table(*, preparation: Mapping[str, Any], verdict: Mapping[str, Any], collision_path: Path,
                     native_root: Path) -> dict[str, Any]:
    """Which catalog robot models fit the job by spec sheet and, for the arms that have one, their own
    kinematic chain (``website_robot_fit``), beside the verdict.

    Informational: it protects no privacy, rights or money, so it fails open. A failure is recorded
    and the handoff goes on with its gating unchanged; only a lost handoff lease, which stops every
    write, stops it here too.
    """
    try:
        from .robot_arm_chains import load_robot_arm_chains
        from .robot_spec_catalog import load_robot_spec_catalog
        from .website_robot_fit import FILENAME, robot_fit_table
        table = robot_fit_table(preparation=preparation, collision_path=collision_path, verdict=verdict,
                                catalog=load_robot_spec_catalog(), arm_chains=load_robot_arm_chains())
        write_json(native_root / FILENAME, table)
        return {"path": str(native_root / FILENAME), "digest": table["table_digest"], "counts": table["counts"]}
    except HandoffLeaseError:
        raise
    except Exception as exc:  # noqa: BLE001 - only privacy, rights and money stop a job; this is recorded
        return {"status": "failed", "error": type(exc).__name__}


def _rigid_verdict_and_fit(*, preparation: Mapping[str, Any], collision_path: Path,
                           native_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The capability verdict (written beside the native inputs) and the robot fit table, CPU only.

    One read scope holds both: a USDA text collider (one normalized before 2026-11-03) is read once, into
    a binary copy every search of both opens, where each search parsed it again at about 3.6 GB
    (``usd_binary_read_cache``). A binary collider is opened as it is.
    """
    from .usd_binary_read_cache import read_scope
    from .website_rigid_feasibility import rigid_task_verdict

    with read_scope():
        verdict = rigid_task_verdict(preparation=preparation, collision_path=collision_path)
        write_json(native_root / "capability_verdict.json", verdict)
        return verdict, _robot_fit_table(preparation=preparation, verdict=verdict, native_root=native_root,
                                         collision_path=collision_path)


def geometry_only_fit_check(*, preparation: Mapping[str, Any], base_scene: Mapping[str, Any],
                            native_root: Path) -> dict[str, Any]:
    """Where a geometry-only job ends: the capability verdict and the robot fit table, CPU only.

    The scene keeps its captured subject (``website_atlas_pipeline.ROUTE_ENV``), its box cut
    from the collider at compile (``website_subject_excision``), so the job is never
    registered or enqueued: nothing paid follows. Only a rigid pick-and-place preparation
    held for nothing but this check, with no uncut subject, is assessed, as the full route
    assesses only an intake-ready one; anything else is recorded as not run.
    """
    if list(preparation.get("blockers") or []) != [GEOMETRY_ONLY_BLOCKER]:
        return {"status": "not_run", "reason": "preparation_blocked"}
    if preparation["intake_request"]["task"]["strategy"] != "pick_and_place":
        return {"status": "not_run", "reason": "task_strategy_not_modelled"}
    if (preparation["subject"].get("geometry_origin") != "synthetic_design"
            and (preparation.get("subject_excision") or {}).get("status") != "excised"):
        # Every robot check would meet the subject itself.
        return {"status": "not_run", "reason": "subject_not_excised"}
    from .local_reconstruction_adapters import _sha256_file
    from .website_scene_runtime_inputs import normalize_runtime_collision
    if (_sha256_file(Path(base_scene["collision_mesh_path"])) != base_scene["collision_mesh_digest"]
            or base_scene["collision_mesh_digest"] != preparation["binding"]["collision_mesh_digest"]):
        raise ValueError("website_native_background_changed")
    collision_root, normalized = normalize_runtime_collision(preparation=preparation, base_scene=base_scene,
                                                             output_root=native_root)
    collision = collision_root / normalized["output"]["relative_path"]
    verdict, fit = _rigid_verdict_and_fit(preparation=preparation, collision_path=collision, native_root=native_root)
    return {"status": "completed",
            "collision": {"path": str(collision), "digest": normalized["output"]["sha256"]},
            "capability_verdict": {"status": verdict["status"], "blockers": verdict["blockers"],
                                   **({"risks": verdict["risks"]} if "risks" in verdict else {}),
                                   "path": str(native_root / "capability_verdict.json"),
                                   "digest": verdict["verdict_digest"]},
            "robot_fit_table": fit}


def name_anthropic_for_site(rights: Mapping[str, Any], capture_root: Path | None) -> bool:
    """Before a rights admission that lets frames and task evidence go to Anthropic is written,
    name Anthropic in the site's own withdrawal ledger (the folder the withdrawal planner reads)."""
    if "anthropic_provider_terms_reference" not in rights:
        return False
    from .website_capture_withdrawal import _site
    from .website_capture_withdrawal_provider_copies import record_provider_admission
    if capture_root is None:
        raise ValueError("website_anthropic_admission_site_unknown")
    record_provider_admission(_site(Path(capture_root)), "anthropic")
    return True


def artisan_provider_choice_for(*, authority: Mapping[str, Any], root: Path, capture_root: Path,
                                geometry_only: bool = False) -> dict[str, Any] | None:
    """The host's once-per-grant choice of ARTISAN's text provider, or ``None`` (today's path).

    Only on a host set to prefer Anthropic, for a both-providers grant of an ARTISAN job; every
    other job runs today's path untouched, writing nothing. When Anthropic is chosen, the site's
    withdrawal ledger names it before the choice is returned, so no site text or frame reaches
    Anthropic before that entry exists.
    """
    import os

    from .artisan_provider_choice import PREFER_ANTHROPIC_ENV, both_providers_granted, retained_or_new_choice
    from .task_object_authoring_backend import ARTISAN_BACKEND, default_backend
    if (os.environ.get(PREFER_ANTHROPIC_ENV) != "1" or geometry_only or not both_providers_granted(authority)
            or (authority.get("replacement_authoring_backend") or default_backend()) != ARTISAN_BACKEND):
        return None
    choice = retained_or_new_choice(root=root, grant=authority)
    if choice["provider"] == "anthropic":
        from .website_capture_withdrawal import _site
        from .website_capture_withdrawal_provider_copies import record_provider_admission
        record_provider_admission(_site(Path(capture_root)), "anthropic")
    return choice


def prepare_rigid_native_inputs(*, result: dict[str, Any], preparation: Mapping[str, Any], preparation_path: Path,
                                native_root: Path, task_context: Mapping[str, Any], now: float,
                                runtime_inputs: Mapping[str, Any] | None = None,
                                base_scene: Mapping[str, Any] | None = None,
                                source_geometry: Mapping[str, Any] | None = None,
                                task_masks: Mapping[str, Any] | None = None,
                                capture_root: Path | None = None) -> bool:
    """The rigid native tail of a compiled preparation, CPU only: it buys nothing.

    Runtime inputs (unless the caller already made them), the collision and
    appearance stages, the construction inputs, the rights admission and the
    capability verdict, with the informational robot fit table beside it. The
    handoff and a renewal re-issue
    (``website_preparation_renewal``) share it, so the inputs bound to a
    preparation's digest cannot drift between them. Records into ``result``
    and returns whether the preparation may be registered: intake ready and
    appearance authored. The capability verdict and the support record are
    recorded and never hold the job.
    """
    if runtime_inputs is None:
        runtime_inputs = prepare_website_runtime_inputs(preparation=preparation, base_scene=base_scene,
            source_geometry=source_geometry, task_masks=task_masks, output_root=native_root)
    result["runtime_inputs"] = {"status": runtime_inputs["status"], "digest": runtime_inputs["digest"],
                                "path": str(native_root / "runtime_inputs.json")}
    pad = (runtime_inputs.get("collision") or {}).get("support_pad")
    support_hold = None
    if isinstance(pad, Mapping):
        from .website_support_pad import hold_reason
        support_hold = hold_reason(pad)
        result["task_support_pad"] = {key: pad[key] for key in ("status", "reason", "scene_revision_collider")
                                      if key in pad}
        result["task_support_pad"]["hold_reason"] = support_hold
    from .website_native_background import prepare_collision_stage, prepare_appearance_stage, prepare_construction_stages
    collision_stage = prepare_collision_stage(native_root / "runtime_inputs.json")
    write_json(native_root / "collision_stage_inputs.json", collision_stage)
    result["runtime_inputs"]["collision_stage_inputs_path"] = str(native_root / "collision_stage_inputs.json")
    if runtime_inputs["appearance"]["status"] != "native_appearance_authored":
        return False
    appearance_stage = prepare_appearance_stage(native_root / "runtime_inputs.json")
    write_json(native_root / "appearance_stage_inputs.json", appearance_stage)
    result["runtime_inputs"]["appearance_stage_inputs_path"] = str(native_root / "appearance_stage_inputs.json")
    construction = prepare_construction_stages(runtime_inputs_path=native_root / "runtime_inputs.json",
                                                preparation_path=preparation_path)
    if preparation["status"] == "intake_ready":
        from .website_native_background import construction_rights_admission
        from .website_object_observations import _record
        rights = construction_rights_admission(preparation=preparation, task_context=task_context, now=now)
        name_anthropic_for_site(rights, capture_root)
        rights_path = native_root / "rights_admission.json"
        write_json(rights_path, rights)
        construction["references"].append({"contract_path": "scene.rights.admission", **_record(rights_path)})
    write_json(native_root / "construction_inputs.json", construction)
    result["runtime_inputs"]["construction_inputs_path"] = str(native_root / "construction_inputs.json")
    if preparation["status"] == "intake_ready":
        from .usd_binary_read_cache import read_scope
        collision = Path(runtime_inputs["collision"]["path"])
        verdict = None
        # One read scope for every search below: a USDA text collider is read once (``usd_binary_read_cache``).
        with read_scope():
            if preparation["intake_request"]["task"]["strategy"] == "pick_and_place":
                # Abstain before the owner is asked to pay when no enabled embodiment can carry the
                # task; one the gate cannot model is unassessed and never refused.
                verdict, fit = _rigid_verdict_and_fit(preparation=preparation, collision_path=collision,
                                                      native_root=native_root)
                result["capability_verdict"] = {"status": verdict["status"], "blockers": verdict["blockers"],
                    **({"risks": verdict["risks"]} if "risks" in verdict else {}),
                    "path": str(native_root / "capability_verdict.json"), "digest": verdict["verdict_digest"]}
                result["robot_fit_table"] = fit
                # An abstained verdict (no enabled arm, no clear stance found, or the object beyond reach
                # from every stance it found) is a judgement about the outcome: recorded, never a stop. It
                # travels as ``scene_estimate:capability_unassessed`` (``atlas_scene_estimates``).
        # A support on which the object cannot rest everywhere it is counted as placed is recorded, never a
        # stop (owner, 2026-11-16): the package says so (``website_company_simulation``), and the paid
        # robot-free probe still refuses it (``rigid_support_pad.half_support_hold``).
    return preparation["status"] == "intake_ready"


#: Where a run keeps the records a base scene from an earlier Atlas result named (``retain_superseded_records``).
SUPERSEDED_DIRECTORY = "superseded_atlas_results"
#: The workspace and its directories whose records each run rewrites in place; what they name is content-addressed.
_RECORD_DIRECTORIES = ("", "native", "fit_check", "development_test")


def retain_superseded_records(*, capture_root: Path, root: Path, atlas_result_digest: Any,
                              levelling_digest: Any = None) -> Path | None:
    """Copy the records a base scene from another Atlas result named, before this run replaces them.

    The levelled scene, the navigation scene half and every file they name sit in content-addressed
    directories, and a new Atlas result writes new ones beside them. The records naming them (the handoff,
    the base scene, the compiled specs, the native inputs) sit at fixed paths each run rewrites. When the
    base scene this run replaces came from another result (the job's Atlas result was published again,
    ``website_held_handoff_replay``), each is copied once, as it is, under
    ``superseded_atlas_results/<that result>/``, so evidence made on the old scene still verifies against
    the records it named. Nothing is deleted. Returns that directory, or None when nothing was superseded.
    """
    import os
    import re
    import shutil

    try:
        previous = json.loads((root / "base_scene.json").read_text())
    except (OSError, ValueError):
        return None
    built = previous.get("atlas_result_digest") if isinstance(previous, dict) else None
    levelled = (previous.get("levelling") or {}).get("digest") if isinstance(previous, dict) else None
    if not isinstance(built, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", built):
        return None
    preparation = capture_root / "pipeline" / "website_scene_preparation"
    if built != atlas_result_digest:
        archive = preparation / SUPERSEDED_DIRECTORY / built[7:23]
    elif (levelling_digest is not None and isinstance(levelled, str)
          and re.fullmatch(r"sha256:[0-9a-f]{64}", levelled) and levelled != levelling_digest):
        # The same Atlas result levelled again by a newer levelling revision (``superseded_levelling``).
        archive = preparation / SUPERSEDED_DIRECTORY / f"{built[7:23]}-levelling-{levelled[7:23]}"
    else:
        return None
    records = [preparation / "handoff.json", *sorted(path for name in _RECORD_DIRECTORIES
                                                     for path in (root / name).glob("*"))]
    for path in dict.fromkeys(records):
        target = archive / path.relative_to(preparation)
        if path.is_symlink() or not path.is_file() or target.exists():
            continue
        # A whole copy or none: a crash leaves only the staging name, and the next run copies again.
        staging = target.with_name(target.name + ".partial")
        with handoff_effect("website_scene_handoff_superseded_records"):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, staging)
            os.replace(staging, target)
    return archive


#: Owner direction 2026-11-16: any job is accepted; what is missing is estimated and flagged, and the job keeps
#: going. A job the robot does by moving itself (navigation), and one Blueprint has no task profile, no installed
#: runtime or no one-object model for, gets the room package: the room as reconstructed, the job as the site wrote
#: it and, for navigation, the goal region with an assumed rule. The company builder reads it from the handoff.
ROOM_PACKAGE = "room_package_written"
ROOM_RECORD = "room_package_inputs.v1.json"
#: The navigation rule's two assumed numbers: how near the goal centre the robot's base must stop, and for how long.
GOAL_RADIUS_M, GOAL_DWELL_SECONDS = 0.5, 2.0
#: The handoff gaps that mean "no task profile": ``website_task_preparation.profile_preparation_gap`` and an operation
#: reading naming no family (``website_operation_coverage``).
ROOM_GAPS = frozenset({"website_native_task_profile_preparation_required", "website_task_operation_coverage_unresolved"})
#: The other "no model for this job" findings that ship the room package, each flagged on its record by its code.
MISMATCH = "website_task_profile_mechanism_mismatch"
RUNTIME_GAP = "website_task_family_runtime_preparation_required"


def _sim_frame(*, base_scene: Mapping[str, Any], source_geometry: Mapping[str, Any],
               focus: Mapping[str, Any] | None) -> tuple[float, Any, Any]:
    """Metres per runtime unit, runtime-to-simulator (Z up) and the source-to-runtime registration (or None).

    As compile frames a scene (``compile_website_scene_preparation``): the base scene's declared scale, else the
    registration's. Registration runs only when a goal must be placed or no scale is declared; a goal that does not
    register is left out (None), a scene with neither scale is refused.
    """
    import numpy as np

    from .website_task_preparation import _UP_INDEX, register_source_to_runtime
    mpu = base_scene.get("meters_per_unit")
    matrix = None
    if focus is not None or mpu is None:
        anchor = base_scene.get("anchor")
        anchor = ({**anchor, "meters_per_unit": mpu, "up_axis": base_scene["up_axis"],
                   "ground_plane_offset_m": base_scene.get("ground_plane_offset_m")}
                  if isinstance(anchor, Mapping) and mpu is not None else None)
        try:
            registration = register_source_to_runtime(
                source_geometry=source_geometry, collision_mesh_path=Path(base_scene["collision_mesh_path"]),
                anchor=anchor, focus_bounds=focus, up_axis=base_scene["up_axis"])
        except ValueError:
            if mpu is None:
                raise
        else:
            matrix = np.asarray(registration["source_to_runtime"])
            mpu = 1.0 / registration["scale"] if mpu is None else mpu
    up_sign = -1 if base_scene["up_axis"] == "-Y" else 1
    runtime_to_sim = np.eye(4)
    runtime_to_sim[:3, :3] = float(mpu) * (np.array([[1, 0, 0], [0, 0, -up_sign], [0, up_sign, 0]])
                                           if _UP_INDEX[base_scene["up_axis"]] == 1 else np.eye(3))
    return float(mpu), runtime_to_sim, matrix


def write_room_package(*, task_context: Mapping[str, Any], base_scene: Mapping[str, Any],
                       source_geometry: Mapping[str, Any], removal: Mapping[str, Any], output_root: Path,
                       task_profile: str | None, task_masks: Mapping[str, Any] | None = None,
                       objects_in_scan: bool = False, flags: Sequence[str] = ()) -> dict[str, Any]:
    """The room package's half (``scene-package/``: the room's collision and look, the plan) and its sealed record.

    CPU only, buys nothing. A navigation job (``task_masks`` given) whose reading names exactly one goal mask gets
    a goal region: that mask's visible bounds lifted onto the scene by the registration compile uses, its footprint
    centred at its lowest point (the floor it stands on), and the rule "stop within ``GOAL_RADIUS_M`` of it for
    ``GOAL_DWELL_SECONDS``", both numbers assumed. Any other job ships no rule of Blueprint's: the record says so,
    and the site's words stay on the job's page, the one place they are shown redacted. The job's objects are counted, never placed: taken out of the scan by the
    clean plate, or part of it on a geometry-only route.
    """
    import shutil

    import numpy as np

    from .local_reconstruction_adapters import _sha256_file
    from .scene_package_stage import placements
    from .team_scene_robot_absent import DIRECTORY, PLAN_FILENAME, PLAN_SCHEMA, PROBE_FIELD, PROBE_NOT_QUEUED
    from .website_native_appearance import prepare_native_appearance
    from .website_scene_runtime_inputs import normalize_runtime_collision
    from .website_task_preparation import _runtime_bounds
    for key in ("splat", "collision_mesh"):
        if _sha256_file(Path(base_scene[key + "_path"])) != base_scene[key + "_digest"]:
            raise ValueError("website_base_scene_changed")
    goals = [row for row in (task_masks or {}).get("targets") or []
             if row.get("target_role") == "destination" and isinstance(row.get("estimated_visible_bounds"), Mapping)]
    focus = goals[0]["estimated_visible_bounds"] if task_masks is not None and len(goals) == 1 else None
    mpu, runtime_to_sim, matrix = _sim_frame(base_scene=base_scene, source_geometry=source_geometry, focus=focus)
    room = {"binding": {key + "_digest": base_scene[key + "_digest"] for key in ("splat", "collision_mesh")},
            "coordinate_frame": {"declared_meters_per_unit": mpu, "declared_up_axis": base_scene["up_axis"],
                                 "physical_scale_measured": False, "runtime_to_simulator": runtime_to_sim.tolist()}}
    room["digest"] = canonical_digest(room, digest_field="digest")
    collision_root, normalized = normalize_runtime_collision(preparation=room, base_scene=base_scene,
                                                             output_root=output_root)
    appearance = prepare_native_appearance(preparation=room, base_scene=base_scene, output_root=output_root / "appearance")
    goal = None
    if focus is not None and matrix is not None:
        low, high = (np.asarray(corner) @ runtime_to_sim[:3, :3].T for corner in _runtime_bounds(focus, matrix))
        low, high = np.minimum(low, high), np.maximum(low, high)
        goal = {"basis": "goal_mask_bounds_registered_footprint_centre_at_its_lowest_point",
                "centre_world_m": [float((low[0] + high[0]) / 2), float((low[1] + high[1]) / 2), float(low[2])],
                "radius_m": GOAL_RADIUS_M, "dwell_seconds": GOAL_DWELL_SECONDS, "radius_and_dwell": "assumed"}
    elif task_masks is not None:
        goal = {"basis": "goal_not_placed:" + ("no_single_goal_mask" if focus is None else "goal_not_registered")}
    packet = output_root / "half" / DIRECTORY
    if packet.exists():
        shutil.rmtree(packet)
    rows = []
    for role, source, visible in (("scene_collision", collision_root / normalized["output"]["relative_path"], False),
                                  ("scene_appearance", Path(appearance["artifact"]["path"]), True)):
        target = packet / "assets" / f"{role}{source.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"name": role, "semantic_role": role, "usd_path": f"assets/{target.name}",
                     "sha256": _sha256_file(target), "size_bytes": target.stat().st_size, "visible": visible,
                     "pose_world": {"position_world_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}})
    ruled = goal is not None and "centre_world_m" in goal
    task_kind = "goal_region_navigation" if task_masks is not None else "not_defined_by_blueprint"
    criteria = ({"surface_target": {"surface_position_world_m": goal["centre_world_m"], "radius_m": GOAL_RADIUS_M},
                 "settling": {"stable_seconds": GOAL_DWELL_SECONDS}} if ruled else None)
    plan = {"schema_version": PLAN_SCHEMA, "scene_id": task_context.get("scene_id"),
            "task_id": "website-" + task_context["context_digest"][7:27], "task_kind": task_kind,
            "task_spec": {"task_kind": task_kind, **({"task_success_contract": {"criteria": criteria}} if ruled else {})},
            "asset_directory": "assets", "objects": rows,
            "appearance_frame_alignment": {"representation": "particlefield_3d_gaussian_splat"},
            "robot": None, "cameras": [], "stage_5": None, "stage_4": None, "plan_digest": ""}
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    placements(plan)
    write_json(packet / PLAN_FILENAME, plan)
    removed = [row for row in removal.get("entries") or [] if row.get("task_effect") == "manipulated"]
    record = {"schema_version": "blueprint.room_package_inputs.v1",
              "task_context_digest": task_context["context_digest"], "task_kind": task_kind,
              "task_profile": task_profile or "none", "success_rule_defined": ruled, "success_rule_assumed": ruled,
              **({"goal_region": goal} if goal is not None else {}),
              "job_objects": {"count": len(removed), "in_package": "part_of_room_scan" if objects_in_scan
                              else "taken_out_of_room_scan"} if removed else None,
              **({"flags": list(flags)} if flags else {}), PROBE_FIELD: PROBE_NOT_QUEUED,
              "robot_included": False, "simulator_run": False, "plan_digest": plan["plan_digest"],
              "inputs_digest": ""}
    record["inputs_digest"] = canonical_digest(record, digest_field="inputs_digest")
    write_json(output_root / "half" / ROOM_RECORD, record)
    return record


def prepare_website_scene_handoff(*, descriptor: Mapping[str, Any], clean_plate: Mapping[str, Any],
                                 capture_root: Path, now: float,
                                 allow_operation_coverage_review: bool = False,
                                 allow_static_target_recovery: bool = False) -> dict[str, Any]:
    handoff_lease_checkpoint("website_scene_handoff_start")
    from .website_preparation_revision import workspace, bind_preparation
    root = workspace(capture_root)
    root.mkdir(parents=True, exist_ok=True)
    handoff_path = capture_root / "pipeline" / "website_scene_preparation" / "handoff.json"
    result: dict[str, Any] = {
        "schema_version": "website_scene_handoff.v1", "status": "awaiting_reconstruction",
        "blockers": [], "claim_ceiling": "development_only", "simulator_ready": False,
        "provider_mutation_performed": False,
    }
    metadata = descriptor.get("metadata") or {}
    operation_review_requested = False
    result["capture_id"] = descriptor["capture_id"]
    result["scene_id"] = descriptor["scene_id"]

    def retain_gap(gap: Mapping[str, Any]) -> dict[str, Any]:
        result.update(status="needs_input", blockers=[gap["code"]], native_task_preparation_gap=dict(gap))
        result["digest"] = canonical_digest(result, digest_field="digest")
        write_json(handoff_path, result)
        return result

    def room_package(profile: str | None, masks: Mapping[str, Any] | None = None,
                     flags: Sequence[str] = ()) -> dict[str, Any]:
        # Not held: the room package goes to the company builder (``write_room_package``).
        record = write_room_package(task_context=context, base_scene=base, source_geometry=atlas["source_geometry"],
                                    removal=removal, output_root=root / "room_package", task_profile=profile,
                                    task_masks=masks, objects_in_scan=geometry_only, flags=flags)
        result.update(status=ROOM_PACKAGE, blockers=[], job_disposition="accepted",
                      geometry_backend=atlas["source_geometry"]["geometry_backend"],
                      source_geometry_digest=atlas["source_geometry"]["digest"],
                      room_package={"path": str(root / "room_package" / "half" / ROOM_RECORD),
                                    **{key: record[key] for key in ("inputs_digest", "task_kind", "task_profile",
                                                                    "success_rule_defined", "success_rule_assumed")}})
        result["digest"] = canonical_digest(result, digest_field="digest")
        write_json(handoff_path, result)
        return result

    def room_for(gap: Mapping[str, Any]) -> dict[str, Any]:
        # A gap that says Blueprint has no model for the job, by code: the job's declared profile, flagged.
        declared = gap.get("task_family") if gap["code"] == MISMATCH else gap.get("inferred_profile_id")
        return room_package(declared, flags=[gap["code"]])

    try:
        if clean_plate.get("privacy_verified") is not True or clean_plate.get("status") not in {"noop", "objects_removed"}:
            raise ValueError("website_scene_preparation_pending")
        context = metadata.get("site_task_context") or {}
        if context.get("capture_id") != descriptor["capture_id"] or context.get("scene_id") != descriptor["scene_id"]:
            raise ValueError("website_scene_task_identity_mismatch")
        # The capture's Atlas job is its one paid reconstruction lane: its
        # object-removed background is the scene and its source pose the
        # geometry. Reopening it never reconstructs or purchases geometry.
        from .website_atlas_scene_source import atlas_scene_source
        atlas = atlas_scene_source(capture_root=capture_root, output_root=root / "atlas")
        handoff_lease_checkpoint("website_scene_handoff_after_source")
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
        # A base scene the job's earlier Atlas result built is replaced: its records are kept first.
        retain_superseded_records(capture_root=capture_root, root=root,
                                  atlas_result_digest=base.get("atlas_result_digest"),
                                  levelling_digest=(base.get("levelling") or {}).get("digest"))
        write_json(root / "base_scene.json", base)
        result.update(base_scene_path=str(root / "base_scene.json"), visual_reconstruction_ready=True)
        # A geometry-only job keeps its subject in the scene: it ends at the robot fit
        # check (geometry_only_fit_check) and is never registered or enqueued.
        geometry_only = atlas.get("atlas_route") == "geometry_only"
        if geometry_only:
            result["atlas_route"] = "geometry_only"
        removal_path = Path(clean_plate["removal_manifest_path"])
        if not removal_path.resolve().is_relative_to(capture_root.resolve() / "pipeline"):
            raise ValueError("website_scene_removal_manifest_outside_capture")
        removal = json.loads(removal_path.read_text())
        # Capture truth applies to held families too. Reading this retained
        # manifest performs no mask, authoring or provider work.
        if (atlas.get("subject_origin") == "synthetic_design") != ("synthetic_subject" in removal):
            raise ValueError("website_subject_origin_conflict")
        from .website_task_semantics import load_task_semantics
        source_video_digest = (atlas["source_geometry"].get("binding") or {}).get("source_video_digest")
        semantics = (load_task_semantics(job_dir=capture_root / "pipeline" / "atlas", task_context=context,
                                        source_video_digest=source_video_digest)
                     if context.get("task_family") == "general_task" and source_video_digest else None)
        if semantics is not None and allow_operation_coverage_review:
            from .website_operation_coverage import clarification_request
            if clarification_request(semantics, task_context=context, source_video_digest=source_video_digest) is not None:
                # A completed saved Atlas job need not re-enter its controller.
                # Clarify its operation reading through the same capped source
                # allowance; no scene/pose/selection or owner fact is rewritten.
                from .website_operation_coverage_review import clarify_operation_coverage
                operation_review_requested = True
                result.pop("provider_mutation_performed", None)
                handoff_lease_checkpoint("website_scene_handoff_operation_review")
                clarified = clarify_operation_coverage(job_dir=capture_root / "pipeline" / "atlas",
                    task_context=context, source_video_digest=source_video_digest)
                semantics = clarified["task_semantics"]
        semantics_options = {"task_semantics": semantics, "source_video_digest": source_video_digest}
        gap = native_task_preparation_gap(context, **semantics_options)
        if gap is not None and gap["code"] in ROOM_GAPS:
            from .website_operation_coverage import clarification_request
            if semantics is None or clarification_request(semantics, task_context=context,
                                                          source_video_digest=source_video_digest) is None:
                # No task profile, nor one a clarification could still find: never turned into pick/place,
                # the job ships as the room package.
                return room_package(None)
        if gap is not None:
            # Retain the completed reconstruction and original task. Missing
            # adapter/scoring support cannot turn a different job into pick/place.
            result.update(geometry_backend=atlas["source_geometry"]["geometry_backend"],
                          source_geometry_digest=atlas["source_geometry"]["digest"])
            return retain_gap(gap)
        from .task_profile_contracts import task_profile_for
        from .website_task_semantics import effective_task_profile
        resolved = task_profile_for(effective_task_profile(task_context=context, **semantics_options))
        navigation = resolved is not None and not resolved.requires_manipulation
        from .website_task_masks import bind_task_masks_to_geometry, complete_retained_static_task_masks
        from .task_family_scoring import canonical_task_family
        from .website_task_semantics import effective_task_profile
        family = canonical_task_family(effective_task_profile(task_context=context, **semantics_options))
        result.pop("provider_mutation_performed", None)
        if family is None and (clean_plate.get("task_masks") or {}).get("deferred_target_ids"):
            result["deferred_masks_controller_invoked"] = True
            plan_path = Path(clean_plate["removal_plan_path"])
            if not plan_path.resolve().is_relative_to(capture_root.resolve() / "pipeline"):
                raise ValueError("website_scene_removal_plan_outside_capture")
            recovery_options = {}
            video = clean_plate.get("input_video_path")
            if allow_static_target_recovery and isinstance(video, str):
                source_video = Path(video)
                if (not source_video.is_symlink()
                        and source_video.resolve().is_relative_to(capture_root.resolve())):
                    recovery_options = {"task_context": context, "source_video": source_video,
                                        "recovery_geometry": atlas["source_geometry"]}
            handoff_lease_checkpoint("website_scene_handoff_static_masks")
            # Kept surfaces' full-video source tracks stay compact (``RunLengths``) through
            # binding, coverage, compile and the native inputs; each writes and hashes them
            # with the codec, to the bytes and digests the plain lists give.
            masks = complete_retained_static_task_masks(plan=json.loads(plan_path.read_text()),
                source_geometry=clean_plate["source_frames"], task_masks=clean_plate["task_masks"],
                output_root=plan_path.parent / "task_masks",
                view_plan_root=plan_path.parent / "mask_view_plan", compact_runs=True, **recovery_options)
            clean_plate = {**clean_plate, "task_masks": masks}
        geometry = atlas["source_geometry"]
        result.update(geometry_backend=geometry["geometry_backend"],
                      atlas_scene_source={key: atlas[key] for key in (
                          "status", "source_pose_digest", "background_pose_digest", "alignment")})
        masks = bind_task_masks_to_geometry(task_masks=clean_plate["task_masks"], source_geometry=geometry)
        write_track_json(root / "task_masks.geometry.json", masks)
        clean_plate = {**clean_plate, "source_geometry": geometry, "task_masks": masks}
        from .website_target_background_worker import queue_correction
        correction = queue_correction(capture_root=capture_root, task_context=context, task_masks=masks, now=now)
        if correction['status'] != 'ready':
            result.update(status='awaiting_reconstruction' if correction['status'] == 'pending' else 'needs_input',
                blockers=[correction['reason']], visual_reconstruction_ready=False,
                atlas_scene_source={'status': correction['status'], 'reason': correction['reason'],
                                    'stage': 'target_correction'})
            result['digest'] = canonical_digest(result, digest_field='digest')
            write_json(handoff_path, result)
            return result
        if navigation:
            # The robot moves itself: the room, and the goal region lifted from the reading's goal mask.
            return room_package(resolved.profile_id, masks)
        if family is not None:
            from .website_early_task_hold import task_family_runtime_gap
            if task_family_runtime_gap(context, **semantics_options) is not None:
                # No runtime for the family is installed here: its route cannot run, so the room package instead.
                return room_package(family, flags=[RUNTIME_GAP])
        if family is not None and geometry_only:
            # The fit check models one rigid subject; a family task is recorded, never assessed.
            result.update(status="needs_input", blockers=[GEOMETRY_ONLY_BLOCKER],
                          fit_check={"status": "not_run", "reason": "task_family_not_modelled"})
            result["digest"] = canonical_digest(result, digest_field="digest")
            write_json(handoff_path, result)
            return result
        if family is not None:
            # Family tasks compile all retained targets and their native
            # measurements. The single rigid-subject authoring path cannot
            # represent fluid, residue, cloth or an ordered set of operations.
            preparation = compile_website_scene_preparation(task_context=context, task_masks=masks,
                removal_manifest=removal, source_geometry=geometry, base_scene=base, output_root=root,
                spend=metadata.get("website_scene_execution_authority") or {}, now=now,
                evaluation_inputs=metadata.get("native_task_evaluation"),
                **({"task_semantics": semantics} if semantics is not None else {}))
            preparation = bind_preparation(preparation, output_root=root, capture_root=capture_root,
                authority=metadata.get("website_scene_execution_authority") or {})
            result.update(status=preparation["status"], blockers=preparation["blockers"],
                job_disposition="accepted", task_evaluation_inputs=preparation["task_evaluation_inputs"],
                preparation_path=str(root / "preparation.json"), preparation_digest=preparation["digest"],
                source_geometry_digest=geometry["digest"],
                dispatch_blockers=preparation.get("dispatch_blockers", []))
            if not operation_review_requested:
                result["provider_mutation_performed"] = False
            if preparation["status"] == "intake_ready":
                from .website_scene_dispatch import binding_root, register_website_preparation
                native_path = root / "native_runtime_contract.json"
                write_json(native_path, preparation["task_evaluation_inputs"]["native_runtime_contract"])
                context_path = root / "task_context.json"
                write_json(context_path, context)
                result["source_registration"] = register_website_preparation(
                    preparation_path=root / "preparation.json", runtime_inputs_path=native_path,
                    task_context_path=context_path, root=binding_root(), now=now)
                if (metadata.get("website_scene_execution_authority") or {}).get("schema_version") == "website_scene_sponsorship.v1":
                    from .website_task_context import enqueue_website_prepared_scene
                    result["website_intake_outbox"] = enqueue_website_prepared_scene(
                        task_context=context, request=preparation["intake_request"])
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
            gap = native_task_preparation_gap(context, inferred_profile_id=inferred, **semantics_options)
            if gap is not None:
                # The reading and the object disagree (the only gap left here): fitted to neither, the room
                # package, flagged, before coverage or spec callbacks buy anything.
                return room_for(gap)
        if synthetic is None:
            # Several task objects, or a soft one, under a one-rigid-object or one-opening
            # profile: held here, before coverage, spec research and authoring buy anything for it.
            gap = native_task_subject_gap(context, subjects=subjects, targets=targets, **semantics_options)
            if gap is not None:
                # Not narrowed to one rigid object: the room package, the objects counted, no rule of Blueprint's.
                return room_for(gap)
            # Rigid or unknown goes on; the reading, its source and an unknown's flag stay on the record.
            rigidity = subject_rigidity(context, subjects=subjects, targets=targets, **semantics_options)
            if rigidity is not None:
                result["task_object_rigidity"] = rigidity
                if "flag" in rigidity:
                    result["flags"] = [rigidity["flag"]]
        # Each articulated task target needs views of its whole assembly from
        # the full track, not only the geometry frames. Retained receipts and
        # decoded frames make a restart free.
        from .website_assembly_coverage import attach_assembly_coverage
        source_video = Path(clean_plate.get("input_video_path") or "")
        handoff_lease_checkpoint("website_scene_handoff_assembly_coverage")
        masks = attach_assembly_coverage(task_masks=clean_plate["task_masks"],
            source_geometry=clean_plate["source_geometry"], removal_manifest=removal, task_context=context,
            source_video=(source_video if source_video.is_file()
                          and source_video.resolve().is_relative_to(capture_root.resolve()) else None),
            output_root=root / "assembly_coverage")
        # Published size and mass of an identified object reach compile before
        # the builder. An unknown identity or closed agent gate records not_run
        # and spends nothing; retained receipts make restarts free.
        from .website_object_spec_research import attach_object_specs
        handoff_lease_checkpoint("website_scene_handoff_object_specs")
        masks = attach_object_specs(task_masks=masks, removal_manifest=removal, task_context=context,
                                    output_root=root / "object_spec")
        if masks != clean_plate["task_masks"]:
            write_track_json(root / "task_masks.coverage.json", masks)
            clean_plate = {**clean_plate, "task_masks": masks}
        authority = metadata.get("website_scene_execution_authority") or {}
        if not all(key in authority for key in ("max_total_spend_usd", "max_paid_attempts", "expires_at_epoch")):
            # Compile geometry even while execution is held. Zero authority
            # cannot validate as a paid intake request.
            authority = {"max_total_spend_usd": 0, "max_paid_attempts": 0, "expires_at_epoch": now}
        def choose():  # called by compile only once it found no gap: a room package never chooses
            return artisan_provider_choice_for(authority=authority, root=root, capture_root=capture_root,
                                               geometry_only=geometry_only)
        preparation = compile_website_scene_preparation(
            task_context=context, task_masks=clean_plate["task_masks"],
            removal_manifest=removal, source_geometry=clean_plate["source_geometry"],
            base_scene=base, output_root=root, spend=authority, now=now,
            artisan_provider_choice=choose,
            **({"task_semantics": semantics} if semantics is not None else {}),
            **({"atlas_route": "geometry_only"} if geometry_only else {}),
        )
        preparation = bind_preparation(preparation, output_root=root, capture_root=capture_root, authority=authority)
        if ((preparation.get("collider_conditioning") or {}).get("status") == "conditioned"
                or (preparation.get("subject_excision") or {}).get("status") == "excised"):
            # Compile sealed a conditioned or cut collider into the base scene; later stages read that one.
            base = json.loads((root / "base_scene.json").read_text())
        result.update(status=preparation["status"], blockers=preparation["blockers"],
                      preparation_path=str(root / "preparation.json"), preparation_digest=preparation["digest"],
                      thumbnail=preparation["thumbnail"], base_scene_path=str(root / "base_scene.json"))
        if isinstance(preparation.get("native_task_preparation_gap"), Mapping):
            # Later source-bound evidence can also reveal a mismatch: no native intake, so no runtime preparation.
            # Every gap compile writes says only that Blueprint has no model for the job: the room package, flagged.
            return room_for(preparation["native_task_preparation_gap"])
        if geometry_only:
            result["fit_check"] = geometry_only_fit_check(preparation=preparation, base_scene=base,
                                                          native_root=root / "fit_check")
            result["digest"] = canonical_digest(result, digest_field="digest")
            write_json(handoff_path, result)
            return result
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
                runtime_inputs = None
            if prepare_rigid_native_inputs(result=result, preparation=preparation,
                                           preparation_path=root / "preparation.json", native_root=native_root,
                                           task_context=context, now=now, runtime_inputs=runtime_inputs,
                                           base_scene=base, source_geometry=clean_plate["source_geometry"],
                                           task_masks=clean_plate["task_masks"], capture_root=capture_root):
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
        from .website_recapture_required import CODES as RECAPTURE_CODES, FIELD as RECAPTURE_FIELD, record
        stopped_context = metadata.get("site_task_context") or {}
        if (isinstance(exc, ValueError) and str(exc) in RECAPTURE_CODES
                and isinstance(stopped_context.get("context_digest"), str)):
            # Only a new video clears it: the stage ends as a hold that asks the site for one, never retried.
            result.update(status="needs_input", blockers=[str(exc)], **{RECAPTURE_FIELD: record(
                str(exc), task_context_digest=stopped_context["context_digest"])})
        else:
            result.update(status="awaiting_inputs", blockers=[str(exc)])
    result["digest"] = canonical_digest(result, digest_field="digest")
    write_json(handoff_path, result)
    return result
