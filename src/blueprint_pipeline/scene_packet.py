"""Compile a website job's scene packet: the site's simulation with the job's object in it, and its success rule.

A published configured scene revision becomes one directory (the "half"): ``native-task-packet/``
(plan, assets, receipts, from the shared ``native_task_arena_packet``) and
``team_scene_inputs.v1.json``, the sealed record the scripted check's reader validates. The task
comes from the revision's documents through the shared adapters (rigid pick and place or push,
seated and padded on the collider the physics runs on; or opening and closing a jointed part).
The success contract (confirmed by the site, Blueprint's standard thresholds marked as assumed,
derived, or the default) is always in the plan, so the download can ship it. The packet carries
one idle DROID Franka at the first clear stance. Nothing here spends money.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .usd_binary_read_cache import binary_copy, read_scoped

# The record and directory names the kept readers open (team_scene_packet's, unchanged).
SCHEMA = "blueprint.team_scene_inputs.v1"
CLAIM_CEILING = "diagnostic_policy_execution"
RECORD_FILENAME = "team_scene_inputs.v1.json"
SCENE_DIRECTORY = "scene"
PACKET_DIRECTORY = "native-task-packet"
SUPPORT_PAD_DIRECTORY = "support-pad"
PLAN_FILENAME = "native_task_arena_scene_plan.v1.json"
RECEIPT_FILENAME = "native_task_arena_packet_receipt.v1.json"
# A pedestal for the arm's base: its half extent on the floor, metres (team_scene_packet).
STATION_FOOTPRINT_HALF_EXTENT_M = (0.15, 0.15)
# The scripted check's reader requires this record (team_scene_packet.PENDING_CONTROLS): controls are not run
# before a check, and a contract that requires them per cell runs under this typed diagnostic omission.
PENDING_CONTROLS = {"mode": "nonblocking_diagnostic_pending", "typed_gap": "controls_pending_at_submission",
                    "policy_execution_blocked": False}
RIGID_STRATEGIES = ("pick_and_place", "planar_push")
ARTICULATED_STRATEGY = "articulated_open_close"
DEFAULT_CONTRACT_AUTHOR = "blueprint:manipulation_strategy_defaults.v1"
# The prefix the site-confirmation exchange recognizes an owner-contract refusal by.
OWNER_CONTRACT_INVALID = "team_scene_owner_contract_invalid:"


class ScenePacketError(ValueError):
    """The scene cannot become a packet; the message is the typed reason."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _read(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ScenePacketError(code) from exc
    if not isinstance(value, dict):
        raise ScenePacketError(code)
    return value


def check_droid_profile(path: Path, digest: str) -> dict[str, Any]:
    """The operator's DROID profile: bytes bound to the registry's digest, cameras external, wrist and overview."""

    path = Path(path)
    if path.is_symlink() or not path.is_file() or _sha256(path) != digest:
        raise ScenePacketError("team_scene_droid_profile_binding_invalid")
    profile = _read(path, "team_scene_droid_profile_invalid")
    roles = {str(row.get("role")): row for row in profile.get("cameras") or [] if isinstance(row, Mapping)}
    if (profile.get("schema_version") != "native_task_arena_packet_request.v1"
            or profile.get("request_digest") != canonical_digest(profile, digest_field="request_digest")
            or set(roles) != {"external", "wrist", "overview"} or roles["wrist"].get("pose_frame") != "robot_body"
            or not isinstance(roles["external"].get("intrinsics"), Mapping)
            or not isinstance(profile.get("robot_joint_reset_positions_rad"), Mapping)):
        raise ScenePacketError("team_scene_droid_profile_invalid")
    return profile


def check_runtime_source(path: Path) -> dict[str, Any]:
    """The runtime-source receipt, recorded by path, size and digest. Hard stop: redistribution must be permitted."""

    receipt = _read(Path(path), "team_scene_runtime_source_invalid")
    if (receipt.get("schema_version") != "native_task_runtime_source_packet.v1"
            or not str(receipt.get("packet_sha256") or "").startswith("sha256:")
            or int(receipt.get("packet_size_bytes") or 0) <= 0):
        raise ScenePacketError("team_scene_runtime_source_invalid")
    if receipt.get("redistribution_permitted") is not True:
        # A rights stop: the code carries the word, so no timed retry re-drives it (job_retry_policy).
        raise ScenePacketError("team_scene_runtime_source_redistribution_rights_not_granted")
    return _file_record(Path(path))


def check_bundle(bundle: Path, revision: Mapping[str, Any]) -> Path:
    """The configured bundle, whose size and digest must be the ones the revision publishes."""

    bundle = Path(bundle)
    declared = revision.get("configured_scene_bundle") or {}
    if (bundle.is_symlink() or not bundle.is_file() or bundle.stat().st_size != declared.get("size_bytes")
            or _sha256(bundle) != declared.get("digest")):
        raise ScenePacketError("team_scene_configured_bundle_binding_invalid")
    return bundle


def fetch_sources(*, run_root: str | Path, output_root: str | Path,
                  fetcher: Callable[[Mapping[str, Any]], bytes] | None = None
                  ) -> tuple[dict[str, Any], dict[str, Path]]:
    """The source launch's published revision, and its documents fetched and digest-checked under ``output_root``."""

    from .robot_free_readiness_retention import _revision
    from .task_evaluation_configured_controls_source_evidence import _validate_source
    from .task_evaluation_scene_rigid_robot_free_readiness import fetch_revision_documents

    result, _receipt, _zero = _validate_source(Path(run_root))
    revision = _revision(result["configured_scene_revision_reference"],
                         revision_digest=result["configured_scene_revision_digest"], fetcher=fetcher)
    documents = fetch_revision_documents(revision=revision, output_root=Path(output_root), fetcher=fetcher)
    return revision, documents


def _with_scoring_bounds(spec: Mapping[str, Any], bounds: Mapping[str, Any] | None) -> dict[str, Any]:
    """``spec`` with the object's qualified collision bounds in its scoring frame, as a surface target binds them."""

    value = dict(spec)
    if value.get("subject_collision_bounds_scoring_frame_m") is None and isinstance(bounds, Mapping):
        transform = (value.get("interaction_affordance") or {}).get("asset_root_from_scoring_frame") or {}
        offset = [float(item) for item in transform.get("position_m") or []]
        if len(offset) == 3 and [float(item) for item in transform.get("orientation_xyzw") or []] == [0, 0, 0, 1]:
            value["subject_collision_bounds_scoring_frame_m"] = {
                key: [float(bounds[key][axis]) - offset[axis] for axis in range(3)] for key in ("minimum", "maximum")}
    return value


def _collider_reading(spec: Mapping[str, Any], documents: Mapping[str, Any], bounds: Mapping[str, Any],
                      collision: Path, *, one_support: bool) -> list[Any]:
    """The collider's triangles under the object's footprint at start and goal, or over the pad's whole region."""

    from .rigid_support_pad import MAXIMUM_EDGE_REACH_M, RigidSupportPadError, pad_region
    from .rigid_support_seating import collider_triangles

    observed = documents["static_qualification"].get("observed_structure") or {}
    center = [float(value) for value in observed["center_of_mass_m"]]
    corners = [(float(x) - center[0], float(y) - center[1])
               for x in (bounds["minimum"][0], bounds["maximum"][0]) for y in (bounds["minimum"][1], bounds["maximum"][1])]
    reach = max(math.hypot(x, y) for x, y in corners)
    points: list[Any] = [spec["start_pose_world"][:2], spec["target_position_world_m"][:2]]
    if one_support:
        try:
            points = pad_region(_with_scoring_bounds(spec, bounds), support_bounds=documents["support_plane"]).tolist()
            reach = MAXIMUM_EDGE_REACH_M * math.sqrt(2.0)
        except RigidSupportPadError:
            pass
    return collider_triangles(collision, points_xy=points, reach_m=reach,
                              height_z=float(documents["support_plane"]["top_z_m"]))


def _seated_rigid_adapter(*, revision: Mapping[str, Any], references: Mapping[str, Any], collision: Path,
                          staged_destination: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """The rigid task adapted on the support the physics runs on: (adapter, seating record, support pad record).

    The registered support top is the support's highest face, not where the object rests
    (capture-447: 14.7 mm too high, beyond the scorer's reset tolerance). So the collider is
    read under the object's qualified footprint; a task on one support first has that support
    planarized to its success plane (``rigid_support_pad``). The seated task replaces the
    revision's only when it moves the object by more than ``SEATING_AGREEMENT_M``. The pad
    record keeps, under ``plan``, what ``_stage_support_pad`` writes.
    """

    from .rigid_support_pad import plan_support_pad
    from .rigid_support_seating import SEATING_AGREEMENT_M, triangle_support_surface
    from .task_evaluation_rigid_relocation_native_adapter import adapt_rigid_relocation_task_template

    adapter = adapt_rigid_relocation_task_template(configured_revision=revision, materialized_references=references)
    spec = adapter["native_task_definition"]["task_spec"]
    documents = adapter["source_documents"]["documents"]
    bounds = (documents["static_qualification"].get("observed_structure") or {}).get("collision_bounds_body_frame_m")
    other_surface = staged_destination or documents["definition"].get("destination_support_plane") is not None
    if not isinstance(bounds, Mapping):
        return adapter, {"status": "registered_support_top", "reason": "qualified_collision_bounds_missing"}, {
            "status": "unchecked", "reason": "rigid_support_qualified_collision_bounds_missing"}
    try:
        triangles = _collider_reading(spec, documents, bounds, collision, one_support=not other_surface)
    except Exception as exc:  # noqa: BLE001 - an unread collider keeps the registered top, by name
        reason = f"{type(exc).__name__}:{str(exc)[:160]}"
        return adapter, {"status": "registered_support_top", "reason": reason}, {
            "status": "unchecked", "reason": "rigid_support_collider_unread", "detail": reason}

    pad: dict[str, Any] | None = None
    if other_surface:
        support: dict[str, Any] = {"status": "not_applicable", "reason": (
            "rigid_support_destination_is_a_staged_asset" if staged_destination
            else "rigid_support_destination_on_another_surface")}
    else:
        try:
            pad = plan_support_pad(_with_scoring_bounds(spec, bounds), support_plane=documents["support_plane"],
                                   triangles=triangles)
            support = dict(pad["record"])
        except (KeyError, TypeError, ValueError) as exc:  # RigidSupportPadError names its own reason
            support = {"status": "not_applicable", "reason": str(exc)[:200]}
    planar = pad is not None and pad["record"]["plane"]["consistent"] is True
    physics = pad["planarized_triangles"] if planar else triangles

    seated = adapt_rigid_relocation_task_template(configured_revision=revision, materialized_references=references,
                                                  support_surface=triangle_support_surface(physics))
    definition = seated["native_task_definition"]
    alignment = definition["task_spec"]["interaction_affordance"]["support_alignment"]
    goal = alignment.get("goal_support") or {}
    reading: dict[str, Any] = {"status": "measured_on_scene_collider", "start": alignment.get("start_support"),
                               "goal": alignment.get("goal_support"),
                               **({"support": "planarized_to_success_plane"} if planar else {})}
    if goal.get("within_success_window") is False:
        # Only a corrected success document, confirmed by the site, can change the sealed window.
        reading["flag"] = "goal_rest_outside_sealed_success_window"
    rest = goal.get("rest_scoring_z_m")
    registered_z = float(adapter["native_task_definition"]["task_object_pose_world"]["position_world_m"][2])
    moved = max(abs(float(definition["task_object_pose_world"]["position_world_m"][2]) - registered_z),
                abs(float(rest) - float(spec["target_position_world_m"][2])) if rest is not None else 0.0)
    if pad is not None:
        support["plan"] = {"selection": pad["selection"], "band": pad["record"]["scan_band_z_m"],
                           "plane_z_m": pad["record"]["plane_z_m"], "planarize": planar,
                           "triangles": physics, "scan_triangles": triangles, "collision_bounds_body_frame_m": bounds}
        if not planar:
            support.update(status="held", reasons=[pad["record"]["plane"]["reason"]],
                           requires=pad["record"]["plane"].get("requires"))
    if moved <= SEATING_AGREEMENT_M:
        return adapter, {**reading, "applied": False}, support
    return seated, {**reading, "applied": True}, support


def _stage_support_pad(support: dict[str, Any], *, task_spec: Mapping[str, Any], sealed_collision: Path,
                       root: Path) -> Path | None:
    """Complete the pad record in place (rest check, status, hold reason); write the derived collider when needed.

    Returns the derived collider, read back where the object rests, or None when the physics
    runs on the sealed bytes. A pad that cannot be written, or reads back differently, refuses.
    """

    from .rigid_support_pad import hold_reason, read_back, support_rest_check, write_planarized_collider

    plan = support.pop("plan", None)
    if plan is None:
        return None
    spec = _with_scoring_bounds(task_spec, plan["collision_bounds_body_frame_m"])
    check = support_rest_check(plan["triangles"], spec)
    derived = None
    if plan["planarize"] and support.get("read_points_moved", 0) > 0:
        derived = root / SUPPORT_PAD_DIRECTORY / "collision.usdc"
        try:
            written = write_planarized_collider(sealed_collision, derived, selection=plan["selection"],
                                                band=plan["band"], plane_z=plan["plane_z_m"])
            readback = read_back(derived, spec, record=support)
        except Exception as exc:  # noqa: BLE001 - a support that cannot be written is named, never guessed at
            raise ScenePacketError(f"team_scene_support_pad_unwritten:{type(exc).__name__}:{str(exc)[:120]}") from exc
        if readback["status"] != check["status"] or readback["reasons"] != check["reasons"]:
            raise ScenePacketError("team_scene_support_pad_readback_disagrees")
        support["derived_collider"] = {
            "relative_path": derived.relative_to(root).as_posix(), "sha256": _sha256(derived),
            "size_bytes": derived.stat().st_size, "derived_from_sha256": _sha256(sealed_collision),
            "edit": "point_heights_only", **written, "rest_check_read_back": True}
    elif plan["planarize"]:
        # Already flat at the success plane: the physics runs on the sealed bytes.
        support["derived_collider"] = None
        support["collider_already_planar"] = True
    if plan["planarize"]:
        # What the scan as captured would have done: the reason the pad exists.
        scan = support_rest_check(plan["scan_triangles"], spec, seated_on_highest_contact=True)
        support["scan_rest_check"] = {key: scan.get(key) for key in ("status", "reasons", "start", "goal")}
    support["rest_check"] = check
    if support.get("status") != "held":
        support["status"] = check["status"]
        if check["status"] == "held":
            support["reasons"] = list(check["reasons"])
    elif check["status"] == "held":
        support["reasons"] = [*support.get("reasons", []), *check["reasons"]]
    support["hold_reason"] = hold_reason(support)
    return derived


def _rigid_task(*, revision: Mapping[str, Any], references: Mapping[str, Any], documents: Mapping[str, Path],
                assets: Mapping[str, Path], collision: Path, root: Path, profile: Mapping[str, Any],
                destination_qualification: Path | None, owner_scope: Mapping[str, Any] | None) -> dict[str, Any]:
    """The rigid task as the plan carries it, with where the station must look and stand."""

    from . import task_evaluation_native_arena_episode_compiler as compiler
    from .task_evaluation_rigid_relocation_native_adapter import (
        SOURCE_OBJECT_CONTRACT_PATH,
        SUPPORT_PLANE_CONTRACT_PATH,
    )

    destination_declared = revision["task_template"].get("destination") is not None
    adapter, seating, support = _seated_rigid_adapter(revision=revision, references=references, collision=collision,
                                                      staged_destination=destination_declared)
    definition = adapter["native_task_definition"]
    source_documents = adapter["source_documents"]["documents"]
    task_spec = compiler._runtime_subject_task_spec(definition["task_spec"])
    task_spec["success_criteria"] = adapter["native_success_criteria"].get("criteria")
    destination_asset = None
    if destination_declared:
        from .rigid_destination_task_binding import bind_destination_task_spec, stage_revision_destination

        try:
            destination_asset = stage_revision_destination(
                revision=revision, documents=documents, task_spec=task_spec, collision_path=assets["collision"],
                output_root=root, require_placement=True, placement_qualification_path=destination_qualification,
                expected_execution_binding=profile.get("destination_execution_binding"))
            if destination_asset is None:
                raise ValueError("destination_not_staged")
            task_spec = bind_destination_task_spec(
                task_spec, destination_asset, source_support=source_documents["support_plane"],
                support_plane_digest=_sha256(Path(documents[SUPPORT_PLANE_CONTRACT_PATH])))
        except (ValueError, KeyError, TypeError) as exc:
            raise ScenePacketError("team_scene_destination_qualification_invalid:" + str(exc)) from exc
    task_spec, confirmation = owner_confirmed(task_spec, revision=revision, scope=owner_scope)

    start = [float(v) for v in task_spec["start_pose_world"][:3]]
    target = [float(v) for v in task_spec["target_position_world_m"]]
    # The base stands on the surface the object was measured resting on, when measured (capture-447).
    support_top = float(source_documents["support_plane"]["top_z_m"])
    measured = ((definition["task_spec"].get("interaction_affordance") or {}).get("support_alignment")
                or {}).get("start_support") or {}
    measured_top = measured.get("collider_top_z_m")
    if not isinstance(measured_top, bool) and isinstance(measured_top, (int, float)) and math.isfinite(measured_top):
        support_top = float(measured_top)
    subject = _read(documents[SOURCE_OBJECT_CONTRACT_PATH], "team_scene_subject_bounds_invalid")
    bounds = (subject.get("aabb_min_xyz_m"), subject.get("aabb_max_xyz_m"))
    center = start
    destination_support = source_documents["definition"].get("destination_support_plane")
    if destination_support is not None:
        # The object ends on another surface: the mount is level with the lower one, facing the whole region.
        try:
            support_top = min(support_top, float(destination_support["top_z_m"]))
            half = [(float(b) - float(a)) / 2.0 for a, b in zip(*bounds, strict=True)]
            bounds = ([min(float(bounds[0][i]), target[i] - half[i]) for i in range(3)],
                      [max(float(bounds[1][i]), target[i] + half[i]) for i in range(3)])
        except (KeyError, TypeError, ValueError) as exc:
            raise ScenePacketError("team_scene_destination_support_invalid") from exc
        center = [(a + b) / 2.0 for a, b in zip(*bounds, strict=True)]
    pose = definition["task_object_pose_world"]
    row = {**compiler.evidence_asset_row(assets["replacement"], root=root, semantic_role="task_object",
                                         pose_world=pose),
           "asset_id": task_spec["subject_asset_id"], "source_asset_id": task_spec["source_subject_identity"],
           "object_type": "RIGID", "reset_state": {"root_pose_world": pose, "joint_positions": {}}}
    return {"kind": "rigid_pick_place", "adapter": adapter, "definition": definition, "task_spec": task_spec,
            "confirmation": confirmation, "row": row, "state_binding": definition.get("task_state_binding"),
            "seating": seating, "support": support, "destination_asset": destination_asset,
            "start": start, "target": target, "support_top": support_top, "bounds": bounds, "center": center,
            "whole_region": destination_support is not None, "motion": None}


def _task_motion(task_spec: Mapping[str, Any], *, definition: Mapping[str, Any], asset: Path,
                 pose: Mapping[str, Any]) -> dict[str, Any]:
    """The target joint from reset to its success threshold, and whether the part holds where released.

    Copied from ``task_evaluation_diagnostic_native_arena_compiler._task_motion`` and
    ``_released_task_motion`` (that module leaves with the early trial).
    """

    from .adp_task_scoring import articulated_release_required
    from .scene_physics_preflight import ScenePhysicsPreflightError, joint_holds_released_position
    from .task_evaluation_native_arena_episode_compiler import _derive_runtime_joint_overlay

    try:
        threshold = task_spec["executable_opening_threshold"]
        joint_id, reset = str(threshold["target_joint_id"]), float(threshold["reset_position"])
        goal = float(threshold["success_interval"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("diagnostic_native_compiler_task_motion_unknown") from exc
    motion: dict[str, Any] = {"joint_id": joint_id, "task_range": [reset, goal],
                              "reset_joint_positions": {joint_id: reset}}
    if "release_required" in task_spec:
        motion["release_required"] = articulated_release_required(task_spec)
    with tempfile.TemporaryDirectory(prefix="blueprint-released-motion-") as scratch:
        runtime, _ = _derive_runtime_joint_overlay(task_spec=task_spec, task_definition=definition,
                                                   replacement_path=asset, output_root=Path(scratch))
        try:
            motion["holds_released_position"] = joint_holds_released_position(runtime, joint_id, asset_pose_world=pose)
        except ScenePhysicsPreflightError as exc:
            raise ValueError(f"diagnostic_native_compiler_task_motion_unmeasurable:{exc}") from exc
    return motion


def _articulated_task(*, revision: Mapping[str, Any], references: Mapping[str, Any], replacement: Path, root: Path,
                      owner_scope: Mapping[str, Any] | None) -> dict[str, Any]:
    """The open/close task as the website episode compiler builds it, with the part's sweep the station avoids."""

    import numpy as np

    from . import task_evaluation_native_arena_episode_compiler as compiler
    from .native_articulated_construction_inputs import materialize_articulated_construction_inputs
    from .scene_physics_preflight import _pose_matrix, asset_envelope, load_asset_model, task_motion_surface
    from .task_evaluation_articulated_open_close_native_adapter import adapt_articulated_open_close_task_template
    from .task_evaluation_configured_controls_deferred_inputs import articulated_placement_plan_from_adapter
    from .task_evaluation_rigid_relocation_native_adapter import (
        STATIC_QUALIFICATION_CONTRACT_PATH,
        SUPPORT_PLANE_CONTRACT_PATH,
    )
    from .task_evaluation_robot_placement_trajectory import placement_trajectory_from_native_plan

    adapter = adapt_articulated_open_close_task_template(configured_revision=revision,
                                                         materialized_references=references)
    definition = adapter["native_task_definition"]
    task_spec = compiler._runtime_subject_task_spec(definition["task_spec"])
    task_spec["success_criteria"] = adapter["native_success_criteria"].get("criteria")
    # The site's confirmation applies before construction, so every binding derived from the task holds it.
    task_spec, confirmation = owner_confirmed(task_spec, revision=revision, scope=owner_scope)
    sources = adapter["source_documents"]["documents"]
    digest, size = compiler._sha256_and_size(replacement)
    trajectory = placement_trajectory_from_native_plan(articulated_placement_plan_from_adapter(
        adapted=adapter, asset=replacement, asset_reference={"digest": digest, "size_bytes": size},
        qualified_static_digest=revision["replacement"]["static_qualification"]["digest"],
        configured_revision_digest=revision["revision_digest"]))
    contact = next(phase for phase in trajectory["phases"] if phase["phase_id"] == "handle_contact")
    runtime, adaptation = compiler._derive_runtime_joint_overlay(
        task_spec=task_spec, task_definition=definition, replacement_path=replacement, output_root=root)
    construction = materialize_articulated_construction_inputs(
        task_spec=task_spec, definition=definition, asset=runtime, asset_digest=compiler._sha256_and_size(runtime)[0],
        link_prim_paths=sources[STATIC_QUALIFICATION_CONTRACT_PATH]["link_prim_paths"])
    pose = definition["task_object_pose_world"]
    motion = _task_motion(construction["task_spec"], definition=definition, asset=replacement, pose=pose)

    def sweep(task_range: list[float] | None) -> Any:
        return task_motion_surface(asset_usd=runtime, asset_pose_world=pose, joint_id=motion["joint_id"],
                                   task_range=task_range, reset_joint_positions=motion["reset_joint_positions"])

    whole = sweep(None)
    joints = dict(definition.get("task_object_reset_joint_positions") or {})
    if not joints:
        raise ScenePacketError("team_scene_articulated_reset_joints_missing")
    # The fixture's own envelope at reset, in the world: what the station must stand clear of.
    low, high = asset_envelope(load_asset_model(runtime), joints)
    corners = np.array([[x, y, z, 1.0] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    world = (corners @ _pose_matrix(pose).T)[:, :3]
    row = {**compiler.evidence_asset_row(runtime, root=root, semantic_role="task_object", pose_world=pose),
           "asset_id": task_spec["subject_asset_id"], "source_asset_id": task_spec["source_subject_identity"],
           "object_type": "ARTICULATION", "reset_state": {"root_pose_world": pose, "joint_positions": joints},
           **({"articulation_adaptation": adaptation} if adaptation is not None else {})}
    start = [float(v) for v in contact["position_world_m"]]
    return {"kind": ARTICULATED_STRATEGY, "adapter": adapter, "definition": definition,
            "task_spec": construction["task_spec"], "confirmation": confirmation, "row": row,
            "state_binding": construction["task_state_binding"],
            "kinematic_path": construction["articulated_kinematic_path_receipt"],
            "seating": None, "support": None, "destination_asset": None,
            "start": start, "target": [float(v) for v in trajectory["phases"][-1]["position_world_m"]],
            "support_top": float(sources[SUPPORT_PLANE_CONTRACT_PATH]["top_z_m"]),
            "bounds": (world.min(axis=0).tolist(), world.max(axis=0).tolist()), "center": start,
            "whole_region": False,
            "motion": {"robot_id": "franka_panda", "path_points": sweep(motion["task_range"]),
                       "fall_points": np.zeros((0, 3)) if motion["holds_released_position"] else whole,
                       "limit_points": whole}}


def owner_confirmed(task_spec: Mapping[str, Any], *, revision: Mapping[str, Any],
                    scope: Mapping[str, Any] | None) -> tuple[dict[str, Any], str | None]:
    """The task with the site's confirmation of its ``proposal_only`` thresholds, and that confirmation's digest.

    Only the site's sealed record for this exact intent, revision, success document,
    thresholds and proposal counts (``task_evaluation_owner_success_confirmation``).
    """

    if scope is None:
        return dict(task_spec), None
    from .task_evaluation_owner_success_confirmation import confirmed_owner_authority

    authority = confirmed_owner_authority(task_spec=task_spec, revision=revision, intent=scope["intent"],
                                          directory=scope["directory"])
    if authority is None:
        return dict(task_spec), None
    return {**task_spec, "configured_owner_authority": authority}, authority["confirmation_digest"]


def success_contract(task_spec: Mapping[str, Any], *, task_kind: str, scene_id: str, task_id: str,
                     revision: Mapping[str, Any], confirmed: bool, assume: bool,
                     owner_scope: Mapping[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The job's sealed success contract, and the assumption record when Blueprint's thresholds were assumed.

    In order: Blueprint's standard thresholds, marked as assumed, for a ``proposal_only`` task the
    site has not confirmed (website jobs only, ``assume``); else the owner contract the task's
    authority supports (rigid: derived from the configured criteria; articulated: the owner's frozen
    criteria, only when the task requires an owner contract); else the strategy's compatibility default.
    """

    from . import website_owner_success_assumption as assumption
    from .adp_articulated_task_success_contract import seal_task_success_contract, validate_task_success_contract
    from .adp_task_scoring import TaskNeutralScoringError

    owner, assumed = None, None
    articulated = task_kind == ARTICULATED_STRATEGY
    assumable = assume and not confirmed and assumption.assumable(task_spec)
    if articulated:
        required = (task_spec.get("configured_success_criteria") or {}).get("owner_success_contract_required") is True
        try:
            if required and assumable:
                owner = assumption.assumed_contract(task_spec, task_kind=task_kind, site_id=scene_id, task_id=task_id)
            elif required:
                from .task_evaluation_scene_control_omission import confirmed_articulated_contract

                owner = confirmed_articulated_contract(task_spec=task_spec, site_id=scene_id, task_id=task_id)
        except ValueError as exc:
            raise ScenePacketError(OWNER_CONTRACT_INVALID + str(exc)) from exc
    else:
        try:
            if assumable:
                owner = assumption.assumed_contract(task_spec, task_kind=task_kind, site_id=scene_id, task_id=task_id)
            else:
                from .task_evaluation_rigid_owner_contract import _derive_configured_owner_success_contract

                owner = _derive_configured_owner_success_contract(task_spec, site_id=scene_id, task_id=task_id,
                                                                 team_namespace=revision.get("team_namespace"))
        except TaskNeutralScoringError as exc:
            raise ScenePacketError(OWNER_CONTRACT_INVALID + ",".join(exc.errors)) from exc
    if owner is not None and assumable:
        assumed = assumption.assumption_record(task_spec=task_spec, revision=revision, scope=owner_scope,
                                               contract=owner)
    try:
        contract = owner or seal_task_success_contract(
            task_kind=task_kind, task_spec=task_spec, site_id=scene_id, task_id=task_id,
            author_source="compatibility_default", author_id=DEFAULT_CONTRACT_AUTHOR, confirmation_status="confirmed")
        contract = validate_task_success_contract(contract, task_kind=task_kind)
    except TaskNeutralScoringError as exc:
        raise ScenePacketError("team_scene_success_contract_invalid:" + ",".join(exc.errors)) from exc
    return contract, assumed


def _station(task: Mapping[str, Any], *, collision: Path, physics_collision: Path) -> dict[str, Any]:
    """The first clear floor stance facing the task, on a pedestal level with its support, out of a part's sweep."""

    from .scene_physics_preflight import collision_kept_loaded
    from .team_embodiment import TeamEmbodimentError
    from .team_embodiment_stance import PEDESTAL_SEARCH, ground_stance

    rigid = task["kind"] != ARTICULATED_STRATEGY
    with collision_kept_loaded(collision):
        try:
            return ground_stance(
                collision_path=collision, collision_sha256=_sha256(physics_collision), offset=[0.0, 0.0, 0.0],
                subject_center=list(task["center"]), subject_minimum=[float(v) for v in task["bounds"][0]],
                subject_maximum=[float(v) for v in task["bounds"][1]],
                footprint_half_extent_m=list(STATION_FOOTPRINT_HALF_EXTENT_M), root_world_z_m=task["support_top"],
                task_motion=task["motion"], **PEDESTAL_SEARCH,
                **({"task_points_world_m": [task["start"], task["target"]]} if rigid else {}), ranked=0)
        except TeamEmbodimentError as exc:
            raise ScenePacketError(",".join(exc.blockers) or "team_scene_station_inputs_invalid") from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise ScenePacketError("team_scene_station_inputs_invalid") from exc


def _world_camera(role: str, *, position: Sequence[float], target: Sequence[float],
                  intrinsics: Mapping[str, Any]) -> dict[str, Any]:
    """A world-fixed OpenCV camera looking from ``position`` at ``target``.

    Copied from ``task_evaluation_diagnostic_native_arena_compiler._world_camera`` / ``_look_at``.
    """

    def unit(vector: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(float(value) ** 2 for value in vector))
        if not math.isfinite(norm) or norm <= 1.0e-9:
            raise ScenePacketError("diagnostic_native_compiler_camera_aim_invalid")
        return [float(value) / norm for value in vector]

    def cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    forward = unit([target[i] - position[i] for i in range(3)])
    right = unit(cross(forward, [0.0, 0.0, 1.0]))
    down = unit(cross(forward, right))
    matrix = [right[0], down[0], forward[0], float(position[0]), right[1], down[1], forward[1], float(position[1]),
              right[2], down[2], forward[2], float(position[2]), 0.0, 0.0, 0.0, 1.0]
    return {"role": role, "policy_input": role == "external", "scoring_input": False, "pose_frame": "world",
            "parent_prim_path": "{ENV_REGEX_NS}", "optical_convention": "opencv",
            "frame_from_camera_matrix": matrix, "intrinsics": json.loads(json.dumps(intrinsics))}


def _cameras(profile: Mapping[str, Any], *, base: Sequence[float], start: Sequence[float], target: Sequence[float],
             whole_region: bool) -> list[dict[str, Any]]:
    """The DROID cameras: external above the base, the wrist as profiled, and an overview beside the task."""

    roles = {str(row["role"]): dict(row) for row in profile["cameras"]}
    intrinsics = roles["external"]["intrinsics"]
    direction = [target[0] - start[0], target[1] - start[1]]
    norm = math.hypot(*direction) or 1.0
    lateral = [-direction[1] / norm, direction[0] / norm]
    midpoint = [(start[i] + target[i]) / 2.0 for i in range(3)]
    overview = _world_camera("overview", position=[midpoint[0] + 0.9 * lateral[0], midpoint[1] + 0.9 * lateral[1],
                                                   max(base[2], start[2]) + 1.45], target=midpoint,
                             intrinsics=intrinsics)
    overview["policy_input"] = False
    external = _world_camera("external", position=[base[0], base[1], base[2] + 1.35],
                             target=midpoint if whole_region else start, intrinsics=intrinsics)
    return [external, json.loads(json.dumps(roles["wrist"])), overview]


@read_scoped
def compile_half(*, revision: Mapping[str, Any], documents: Mapping[str, Path], configured_scene_bundle_path: str | Path,
                 droid_profile_path: str | Path, droid_profile_digest: str, runtime_source_receipt_path: str | Path,
                 output_root: str | Path, owner_scope: Mapping[str, Any] | None = None,
                 assume_unconfirmed_thresholds: bool = True, keep_assumed_thresholds: bool = False,
                 destination_placement_qualification_path: str | Path | None = None,
                 native_appearance_materializer: Callable[..., Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Compile the revision into ``output_root`` and seal its ``team_scene_inputs.v1.json``.

    ``owner_scope`` (the request's own intent and its directory) is where the site's confirmation
    of ``proposal_only`` thresholds may be found. ``assume_unconfirmed_thresholds`` scores an
    unconfirmed website job on them at once, marked as assumed. ``keep_assumed_thresholds`` (a
    half of this revision already assumed them) keeps that contract whatever was confirmed since,
    so no digest changes.
    """

    from . import task_evaluation_configured_controls_deferred_inputs as deferred
    from . import task_evaluation_native_arena_episode_compiler as compiler
    from .native_task_arena_packet import materialize_native_task_arena_packet
    from .task_evaluation_rigid_relocation_native_adapter import DEFINITION_CONTRACT_PATH

    root = Path(output_root).resolve()
    if root.exists():
        raise ScenePacketError("team_scene_output_exists")
    destination = revision["task_template"].get("destination")
    if (destination is not None and destination_placement_qualification_path is None
            and "placement_qualification" not in destination):
        raise ScenePacketError("team_scene_destination_placement_qualification_required")
    strategy = _read(documents[DEFINITION_CONTRACT_PATH], "team_scene_task_invalid").get("strategy")
    if strategy != ARTICULATED_STRATEGY and strategy not in RIGID_STRATEGIES:
        raise ScenePacketError("team_scene_task_kind_unsupported")
    bundle = check_bundle(Path(configured_scene_bundle_path), revision)
    profile = check_droid_profile(Path(droid_profile_path), droid_profile_digest)
    runtime_source = check_runtime_source(Path(runtime_source_receipt_path))
    root.mkdir(parents=True, mode=0o750)

    references = deferred._materialized_references(revision=revision, documents=documents)
    scene_id, task_id = str(revision["scene_identity"]["id"]), str(revision["task_template"]["identity"]["id"])
    assets = compiler._extract_configured_assets(bundle, output_root=root)
    # Every collider reader opens this one binary copy of a text room (usd_binary_read_cache).
    collision = binary_copy(assets["collision"], root / ".collider-read-cache")
    physics_collision = assets["collision"]
    scope = None if keep_assumed_thresholds else owner_scope
    if strategy == ARTICULATED_STRATEGY:
        try:
            task = _articulated_task(revision=revision, references=references, replacement=assets["replacement"],
                                     root=root, owner_scope=scope)
        except (KeyError, StopIteration, TypeError, ValueError) as exc:
            raise ScenePacketError("team_scene_articulated_task_invalid:" + str(exc)) from exc
    else:
        task = _rigid_task(revision=revision, references=references, documents=documents, assets=assets,
                           collision=collision, root=root, profile=profile, owner_scope=scope,
                           destination_qualification=(Path(destination_placement_qualification_path)
                                                      if destination_placement_qualification_path else None))

    # The success rule goes into the plan, whichever branch made it (the old stage left a default or an
    # articulated contract only in the record, so the download refused the job).
    task_spec = task["task_spec"]
    contract, assumed = success_contract(task_spec, task_kind=task["kind"], scene_id=scene_id, task_id=task_id,
                                         revision=revision, confirmed=task["confirmation"] is not None,
                                         assume=assume_unconfirmed_thresholds, owner_scope=owner_scope)
    task_spec.update(task_success_contract=contract, task_success_contract_digest=contract["contract_digest"])

    if task["support"] is not None:
        derived = _stage_support_pad(task["support"], task_spec=task_spec, sealed_collision=assets["collision"],
                                     root=root)
        if derived is not None:
            physics_collision = collision = derived
    appearance = dict((native_appearance_materializer or compiler._materialize_native_particlefield_appearance)(
        source_path=assets["appearance"], output_root=root / "native-appearance"))
    if task["kind"] != ARTICULATED_STRATEGY:
        # The room's particles are pressed out of the placed object, so the simulation does not show it twice.
        appearance = compiler.apply_task_volume_clearance(appearance, task_spec=task_spec, output_root=root)
    rows = [compiler.evidence_asset_row(Path(appearance["path"]), root=root, semantic_role="scene_appearance"),
            compiler.evidence_asset_row(physics_collision, root=root, semantic_role="scene_collision"), task["row"]]
    staged = task["destination_asset"]
    if staged is not None:
        rows.append({**compiler.evidence_asset_row(staged["path"], root=root, semantic_role="task_support"),
                     "asset_id": staged["asset_id"], "source_asset_id": staged["source_asset_id"],
                     "object_type": "RIGID", "pose_world": staged["pose_world"],
                     "reset_state": {"root_pose_world": staged["pose_world"], "joint_positions": {}}})

    station = _station(task, collision=collision, physics_collision=physics_collision)
    base = {"position_world_m": station["position_world_m"], "orientation_xyzw": station["orientation_xyzw"]}
    execution = task["adapter"]["native_episode_execution"]
    request: dict[str, Any] = {
        "schema_version": "native_task_arena_packet_request.v1", "scene_id": scene_id, "task_id": task_id,
        "task_spec": task_spec, "task_joint_bindings": task["definition"].get("task_joint_bindings") or [],
        "task_state_binding": task["state_binding"],
        **({"articulated_kinematic_path_receipt": task["kinematic_path"]} if "kinematic_path" in task else {}),
        "assets": rows, "robot_base_pose_world": base,
        "robot_joint_reset_positions_rad": dict(profile["robot_joint_reset_positions_rad"]),
        "cameras": _cameras(profile, base=station["position_world_m"], start=task["start"], target=task["target"],
                            whole_region=task["whole_region"]),
        "scenario": execution.get("scenario"), "physics_frequency_hz": execution.get("physics_frequency_hz"),
        "configured_task_template_adapter": task["adapter"],
        "appearance_variant": compiler.packet_appearance_variant(appearance), "request_digest": ""}
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    packet = root / PACKET_DIRECTORY
    materialize_native_task_arena_packet(request=request, evidence_root=root, output_dir=packet,
                                         link_sources_within=root)
    path = seal_record(root, packet=packet, revision=revision, contract=contract, runtime_source=runtime_source,
                       seating=task["seating"], support=task["support"], assumed=assumed)
    return {"path": str(path), "scene_id": scene_id, "task_id": task_id, "robot_station": station,
            "inputs_digest": json.loads(path.read_text(encoding="utf-8"))["inputs_digest"],
            **({"owner_success_confirmation": task["confirmation"]} if task["confirmation"] else {}),
            **({"owner_success_assumption": assumed} if assumed is not None else {})}


def seal_record(root: Path, *, packet: Path, revision: Mapping[str, Any], contract: Mapping[str, Any],
                runtime_source: Mapping[str, Any], seating: Mapping[str, Any] | None,
                support: Mapping[str, Any] | None, assumed: Mapping[str, Any] | None) -> Path:
    """Write ``team_scene_inputs.v1.json`` (mode 0444) with only what a kept reader reads or validates."""

    from .website_owner_success_assumption import FIELD as ASSUMPTION_FIELD

    receipt = _file_record(packet / RECEIPT_FILENAME)
    value: dict[str, Any] = {
        "schema_version": SCHEMA, "claim_ceiling": CLAIM_CEILING,
        "scene_revision_digest": revision["revision_digest"], "task_success_contract": dict(contract),
        "base_native_packet": receipt, "construction_result": receipt, "runtime_source": dict(runtime_source),
        "control_diagnostic": dict(PENDING_CONTROLS),
        **({"task_support_seating": dict(seating)} if seating is not None else {}),
        **({"task_support_pad": dict(support)} if support is not None else {}),
        **({ASSUMPTION_FIELD: dict(assumed)} if assumed is not None else {}),
        "inputs_digest": ""}
    value["inputs_digest"] = canonical_digest(value, digest_field="inputs_digest")
    path = root / RECORD_FILENAME
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o444)
    return path


def read_half(path: str | Path) -> dict[str, Any]:
    """The sealed record at ``path``, once its seal and its packet and runtime-source files still check, or raise."""

    value = _read(Path(path), "team_scene_inputs_invalid")
    if (value.get("schema_version") != SCHEMA or value.get("claim_ceiling") != CLAIM_CEILING
            or value.get("inputs_digest") != canonical_digest(value, digest_field="inputs_digest")
            or value.get("construction_result") != value.get("base_native_packet")):
        raise ScenePacketError("team_scene_inputs_invalid")
    for field in ("base_native_packet", "runtime_source"):
        record = value.get(field) or {}
        if not Path(str(record.get("path") or "")).is_file() or _file_record(Path(record["path"])) != record:
            raise ScenePacketError(f"team_scene_inputs_{field}_changed")
    return value


def half_plan(path: str | Path) -> dict[str, Any]:
    """The scene plan of the half whose record is at ``path``."""

    value = read_half(path)
    return _read(Path(value["base_native_packet"]["path"]).parent / PLAN_FILENAME, "team_scene_packet_invalid")


def record_half(compiled: Mapping[str, Any], *, run_root: str | Path, capture_root: str | Path, scenario_id: str,
                scene_revision_digest: str) -> dict[str, Any]:
    """Point the capture's derived registry row at the half, then queue the scripted check.

    The row (written by rename) is the only pointer to the half; the check is queued only once it is written.
    The launch's run root is where the check's receipt is retained, so it is the check's root.
    """

    from .team_candidate_quick10_registry import write_derived_profile

    descriptor = _read(Path(capture_root) / "capture_descriptor.json", "team_scene_capture_descriptor_invalid")
    written = write_derived_profile({
        "capture_root": str(Path(capture_root).resolve()), "scene_id": str(descriptor.get("scene_id") or ""),
        "task_id": compiled["task_id"], "scenario_id": scenario_id,
        "robot_free_probe_root": str(Path(run_root).resolve()),
        "scene_revision_digest": scene_revision_digest, "team_scene_inputs_path": compiled["path"]})
    probe: dict[str, Any] = {"status": "profile_not_written", "queued": False}
    if written.get("status") == "written":
        from .rigid_robot_free_probe_runner import enqueue_rigid_robot_free_probe

        probe = enqueue_rigid_robot_free_probe(capture_root)
    return {**compiled, "registry": written, "robot_free_probe": probe}


def prepare(*, run_root: str | Path, capture_root: str | Path, scenario_id: str, output_root: str | Path,
            fetcher: Callable[[Mapping[str, Any]], bytes] | None = None, **compile_inputs: Any) -> dict[str, Any]:
    """Fetch the launch's revision under ``output_root``, compile ``output_root/scene``, record the row."""

    root = Path(output_root)
    revision, documents = fetch_sources(run_root=run_root, output_root=root, fetcher=fetcher)
    compiled = compile_half(revision=revision, documents=documents, output_root=root / SCENE_DIRECTORY,
                            **compile_inputs)
    return record_half(compiled, run_root=run_root, capture_root=capture_root, scenario_id=scenario_id,
                       scene_revision_digest=revision["revision_digest"])

