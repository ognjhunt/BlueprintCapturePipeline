"""A website rigid task's support sealed flat at its success plane, where the room's collider is first produced (CPU).

Beta journey step: scene preparation, before anything is paid for. A website
scene's room collider is first written in simulator metres by
``website_scene_runtime_inputs`` (the normalized Atlas collider); stage 2
copies those bytes, stage 6 assembles them into the configured scene bundle,
and every reader after (publication, the team half, the robot-free probe) reads
them. capture-447ae4f4 showed what a scanned support costs there (#3349): its
tray, seated on the bumpy table, tilted 0.044 rad and settled 2.8 mm low at the
start, and 1.2 mm under the sealed success window at the goal, so the paid
probe was unscorable. #3349 planarizes the support in each team half's derived
copy of the sealed collider (``team_scene_packet``). This module applies the
same model one step earlier, so the scene revision seals a collider the task
can be scored on, and checks it on the CPU before any stage spends.

Scope: a rigid pick-and-place task whose start and goal lie on the one
registered support (``intake_request.task.support``), its destination an
existing surface it is placed ``on``. Nothing else changes and no record is
written: an articulated part, a container (``inside``), a destination on
another surface, a staged destination asset and an authored development
surface keep the collider exactly as normalized. Their team halves keep their
own typed reasons (``rigid_support_destination_on_another_surface``,
``rigid_support_destination_is_a_staged_asset``). Walls, floors the task does
not touch and other furniture stay as scanned: only collider points inside the
support's plan bounds and its own scan band move (``rigid_support_pad``).

The model is ``rigid_support_pad``'s (``support_region_planarized_to_success_plane``),
reused, not copied, with these website inputs (all estimates, never
measurements):

- the plane is the surface target native submission seals into the success
  document (``website_surface_target_admission.sealed_surface_target``): the
  same function, the same inputs. A preparation whose destination submission
  would move first is left to the team half, by name;
- the region holds every replacement static qualification admits, at any yaw:
  around the start, the square of the target disc's radius (the radius holds
  the largest admitted replacement's footprint: ``website_target_radius``),
  and the target disc itself. So a team half's own region, read with the
  qualified replacement, lies inside it, and its pad finds nothing to move;
- the success window and the start are the ones the native bind derives
  (``bind_native_surface_target``) from the subject's estimated bounds, with
  the scorer's reset tolerances. The CPU rest check seats the estimated box on
  the highest point under its footprint, as a team half seats the replacement,
  and asks that it rest there within the reset tolerance and, at the goal,
  inside the window.

A check that fails holds the preparation by its typed reason
(``website_scene_handoff.prepare_rigid_native_inputs``): nothing is registered,
enqueued or paid for. Everything else (an unreadable collider, a region off
the support, a refused surface target) keeps the normalized collider, records
why, and the job goes on; the team half models the support again either way.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core.handoff_lease import HandoffLeaseError, guarded_handoff_write_json as write_json, handoff_effect
from .decision_evidence_contracts import canonical_digest

SCHEMA = "website_rigid_support_pad.v1"
DIRECTORY = "collision-support-pad"
COLLIDER_NAME, RECEIPT_NAME = "collision.usdc", "support_pad.json"
APPLIED_AT = "website_runtime_inputs"


class _NotApplied(ValueError):
    """The task is a rigid support task, but its support is not modelled here: the reason is recorded."""


def rigid_support_task(preparation: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The intake task when it starts and ends on its one registered support, else None.

    None for an articulated part, a container, a destination on another
    surface or a staged destination asset, an authored development surface,
    and a preparation without a task to model.
    """

    request = preparation.get("intake_request")
    if not isinstance(request, Mapping) or preparation.get("development_test") is not None:
        return None
    task = request.get("task")
    if not isinstance(task, Mapping) or task.get("strategy") != "pick_and_place":
        return None
    destination, support, subject = task.get("destination"), task.get("support"), preparation.get("subject")
    if not (isinstance(destination, Mapping) and isinstance(support, Mapping) and isinstance(subject, Mapping)):
        return None
    if (destination.get("relation") != "on" or destination.get("mode") != "existing_support_surface"
            or destination.get("support") is not None or "interior_bounds_world_m" in destination
            or "position_world_m" not in destination):
        return None
    if not all(isinstance(support.get(key), (list, tuple)) and len(support[key]) == 3
               for key in ("aabb_min_xyz", "aabb_max_xyz")):
        return None  # no support was registered under the subject (``support_surface_not_found_under_subject``)
    return task


def task_specs(preparation: Mapping[str, Any], task: Mapping[str, Any]) -> tuple[dict, dict, dict]:
    """The scored task's support fields as the native bind will seal them, the region's task, and the support.

    Simulator frame, Z up. The first holds the subject's estimated bounds about
    its centre (the scoring frame), the start, the sealed surface target, the
    goal and its success window, and the scorer's reset tolerances; the second
    the same with the footprint of any admitted replacement at any yaw.
    """

    from .task_evaluation_rigid_relocation_native_adapter import (
        RESET_ORIENTATION_TOLERANCE_RAD,
        RESET_TRANSLATION_TOLERANCE_M,
        _estimated_support_height_tolerance,
    )
    from .task_evaluation_surface_target import bind_native_surface_target
    from .website_surface_target_admission import (
        SUPPORT_PRIM_PATH,
        sealed_surface_target,
        submission_relocates_destination,
    )

    subject, support, destination = preparation["subject"], task["support"], task["destination"]
    lower = [float(value) for value in subject["aabb_min_xyz"]]
    upper = [float(value) for value in subject["aabb_max_xyz"]]
    if submission_relocates_destination(preparation, destination, subject_min=lower, subject_max=upper):
        # Its region is placed at submission, from the preparation's own collider: the team half models it.
        raise _NotApplied("rigid_support_pad_destination_moved_at_submission")
    target = sealed_surface_target(preparation, destination=destination, support=support, subject_min=lower,
                                   subject_max=upper, development=False)
    centre = [(a + b) / 2.0 for a, b in zip(lower, upper, strict=True)]
    bounds = {"minimum": [lower[axis] - centre[axis] for axis in range(3)],
              "maximum": [upper[axis] - centre[axis] for axis in range(3)]}
    support_plane = {"sage_prim_path": SUPPORT_PRIM_PATH, "top_z_m": float(support["aabb_max_xyz"][2]),
                     "bounds_min_xyz_m": [float(value) for value in support["aabb_min_xyz"]],
                     "bounds_max_xyz_m": [float(value) for value in support["aabb_max_xyz"]]}
    success = task["success"]
    bound = bind_native_surface_target(
        task_spec={"interaction_affordance": {"asset_root_from_scoring_frame": {
                       "position_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}},
                   "destination_position_tolerance_m": float(success["maximum_final_planar_target_error_m"]),
                   "control_frequency_hz": float(success["control_frequency_hz"])},
        target=target, static={"observed_structure": {"collision_bounds_body_frame_m": bounds}},
        support=support_plane, height_tolerance_m=_estimated_support_height_tolerance({
            "dimension_authority": "estimated",
            "placement_uncertainty_m": preparation["coordinate_frame"].get("placement_uncertainty_m")}))
    spec = {key: bound[key] for key in ("subject_collision_bounds_scoring_frame_m", "target_position_world_m",
                                        "surface_target", "support_height_interval_m",
                                        "destination_position_bounds_world_m")}
    spec.update(start_pose_world=[*centre, 0.0, 0.0, 0.0, 1.0],
                reset_translation_tolerance_m=RESET_TRANSLATION_TOLERANCE_M,
                reset_orientation_tolerance_rad=RESET_ORIENTATION_TOLERANCE_RAD)
    radius = float(target["radius_m"])
    scoring = spec["subject_collision_bounds_scoring_frame_m"]
    region = {**spec, "subject_collision_bounds_scoring_frame_m": {
        "minimum": [-radius, -radius, scoring["minimum"][2]], "maximum": [radius, radius, scoring["maximum"][2]]}}
    return spec, region, support_plane


def _file(path: Path) -> dict[str, Any]:
    from .local_reconstruction_adapters import _sha256_file

    return {"path": str(path), "digest": _sha256_file(path), "size_bytes": path.stat().st_size}


def seal_support_pad(*, preparation: Mapping[str, Any], collision: Path, output_root: Path) -> dict[str, Any] | None:
    """The support modelled on the runtime collider: the record, with the planarized collider when one was written.

    None when the task does not start and end on one registered support (the
    collider stays exactly as normalized, and nothing is recorded). Otherwise
    the record has ``status``: ``consistent`` or ``held`` (with ``reasons`` and
    ``hold_reason``, the typed preparation hold), or ``not_applied`` with the
    ``reason`` the normalized collider was kept. ``derived_collider`` names the
    planarized collider (``path``, ``sha256``, ``size_bytes``), or is None when
    the collider was flat at the plane already. A retry with the same inputs
    reuses the receipt under ``output_root/collision-support-pad``.
    """

    task = rigid_support_task(preparation)
    if task is None:
        return None
    source = _file(Path(collision))
    record: dict[str, Any] = {"schema_version": SCHEMA, "applied_at": APPLIED_AT,
                              "source_collider": {"digest": source["digest"], "size_bytes": source["size_bytes"]}}
    try:
        spec, region, support_plane = task_specs(preparation, task)
    except HandoffLeaseError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        # A surface target admission refuses by its own code (preparation recorded it as advisory too).
        return {**record, "status": "not_applied", "reason": str(exc)[:200]}
    from .rigid_support_pad import SCHEMA as PAD_SCHEMA, planarize_support

    binding = {"schema_version": SCHEMA, "pad_schema_version": PAD_SCHEMA, "source_collider": record["source_collider"],
               "task_spec": spec, "region_spec": region, "support_plane": support_plane}
    root = Path(output_root) / DIRECTORY / canonical_digest(binding)[7:31]
    receipt_path = root / RECEIPT_NAME
    prior = _reused(receipt_path, binding)
    if prior is not None:
        return prior
    staging = root / f".staging-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        from .usd_binary_read_cache import read_scope

        # One binary copy of a text collider for every read below (a normalization from before 2026-11-03).
        with read_scope():
            modelled, derived = planarize_support(collision, staging / COLLIDER_NAME, task_spec=spec,
                                                  support_plane=support_plane, region_spec=region)
    except HandoffLeaseError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:  # noqa: BLE001 - geometry is never a hard stop: keep the normalized collider, say why
        shutil.rmtree(staging, ignore_errors=True)
        return {**record, "status": "not_applied", "reason": f"rigid_support_pad_unmodelled:{type(exc).__name__}:"
                                                             f"{str(exc)[:160]}"}
    modelled.pop("sealed_revision_changed", None)
    value = {**record, **modelled, "schema_version": SCHEMA, "pad_schema_version": modelled["schema_version"],
             "success_document_changed": False, "registered_support_changed": False,
             "scene_revision_collider": "planarized" if derived is not None else "as_normalized",
             "estimate_basis": {
                 "plane": "surface_target_native_submission_seals",
                 "start_and_goal": "subject_estimated_bounds_seated_on_highest_contact",
                 "region": "any_admitted_replacement_at_any_yaw_and_the_target_disc"},
             "support_plane": support_plane}
    if derived is not None:
        output = root / COLLIDER_NAME
        with handoff_effect("shared_artifact_promotion"):
            os.replace(derived, output)
        value["derived_collider"] = {**value["derived_collider"], "path": str(output)}
    shutil.rmtree(staging, ignore_errors=True)
    receipt = {"schema_version": SCHEMA, "binding": binding, "record": value}
    receipt["digest"] = canonical_digest(receipt, digest_field="digest")
    write_json(receipt_path, receipt)
    return value


def _reused(receipt_path: Path, binding: Mapping[str, Any]) -> dict[str, Any] | None:
    """A retry's record: the receipt's, when it binds the same inputs and its collider is still those bytes."""

    if not receipt_path.is_file():
        return None
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("binding") != json.loads(json.dumps(binding))
                or receipt.get("digest") != canonical_digest(receipt, digest_field="digest")):
            return None
        value = receipt["record"]
        derived = value.get("derived_collider")
        if derived is not None:
            path = Path(derived["path"])
            if (path.parent != receipt_path.parent or not path.is_file()
                    or _file(path)["digest"] != derived["sha256"]):
                return None
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return None


def hold_reason(record: Mapping[str, Any] | None) -> str | None:
    """The typed reason a preparation's support record holds it before anything is paid for, or None."""

    from .rigid_support_pad import hold_reason as reason

    return reason(record)


__all__ = ["APPLIED_AT", "DIRECTORY", "SCHEMA", "hold_reason", "rigid_support_task", "seal_support_pad", "task_specs"]
