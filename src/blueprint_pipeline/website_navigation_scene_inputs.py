"""A website navigation job's scene half for the G1 lane: a base packet with the G1 at the compiled start.

Part C of docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md. The handoff writes two records for a job whose
robot walks to a goal in a scene kept as filmed: ``navigation_task_spec.json`` (A1,
``website_navigation_packet``) and ``navigation_scene_assets.json`` (A2,
``website_navigation_scene_assets``). This module turns them into what the G1 lane reads for a
rigid team scene (``team_scene_packet``, ``website_task_family_scene_inputs``):

- a base native arena packet (``materialize_native_task_arena_packet``) of kind
  ``goal_region_navigation``: the scene's collision and appearance at identity pose, the A1 task
  spec plus its success contract (``native_g1_navigation_goal.navigation_task_success_contract``),
  and the pinned G1 (``build_pinned_g1_arena_robot_configuration``) standing at the compiled start:
  its x, y and yaw from ``start_pose_world``, its pelvis the G1 profile's height above the compiled
  floor (the rule ``team_embodiment_stance.ground_stance`` applies for the G1 preparation). The head
  camera is the G1 preparation's; the G1 contract also requires one world overview camera, placed
  as the team scene places its overview;
- the scene's robot-free probe receipt (``navigation_robot_free_probe``), written beside the packet
  under the name the probe gate reads (``robot_free_probe_gate.RECEIPT_FILENAME``): the navigation
  scorer run on synthetic root traces built from the plan's own task spec, bound to the plan digest,
  the task spec, its success contract and the scorer. It proves only that the scorer tells those
  traces apart, never anything about the scene, a simulator or a robot;
- a sealed record, ``blueprint.website_navigation_scene_inputs.v1``, holding the packet's files and
  digests, the probe receipt's file record, digest and status, the bindings to both input records,
  ``scene_revision_digest`` (the base plan digest) and every disclosed assumption.

The current-source identity the screening lane binds (``native_g1_beta_authority.SOURCE_FIELDS``)
names a scene intent. A scene kept as filmed gets one from the website's scene intake with a navigation
task (``website_navigation_intake``); its intake receipt, written beside this record once the intent is
staged, binds the record's source. The record itself stays as sealed (``source`` None, its ``source_gap``
naming what the receipt supplies). ``screening_source`` returns the receipt's source and refuses with
``website_navigation_scene_inputs_source_identity_unbound`` while this record has no receipt. Nothing here
spends, calls a provider, runs a simulator or claims a result; everything stays ``development_only``.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_navigation_goal import (
    NAVIGATION_TASK_KIND,
    navigation_task_success_contract,
    validate_navigation_task_success_contract,
)
from .robot_free_probe_gate import RECEIPT_FILENAME as PROBE_RECEIPT_FILENAME
from .usd_binary_read_cache import read_scoped

SCHEMA = "blueprint.website_navigation_scene_inputs.v1"
DIGEST_FIELD = "scene_inputs_digest"
RECORD_FILENAME = "website_navigation_scene_inputs.v1.json"
#: Where the handoff keeps the record: ``<capture workspace>/navigation_scene_inputs/<inputs key>/``,
#: under the site directory, so the site's withdrawal deletes it with the job's other derived files
#: (``website_capture_withdrawal_deletion``); a derived row naming a record elsewhere fails closed there.
OUTPUT_DIRECTORY = "navigation_scene_inputs"
PACKET_DIRECTORY = "native-task-packet"
INPUTS_DIRECTORY = ".inputs"
CLAIM_CEILING = "development_only"
#: The task-spec fields this module adds to the A1 spec; its own seal covers the spec without them.
CONTRACT_FIELDS = ("task_success_contract", "task_success_contract_digest")
#: The G1 preparation's physics rate (``beta_policy_screening_g1_prepare``).
PHYSICS_FREQUENCY_HZ = 200.0
#: Why ``screening_source`` names no source: this record has no intake receipt (``website_navigation_intake``).
SOURCE_UNBOUND = "website_navigation_scene_inputs_source_identity_unbound"
#: The G1 runtime source is the G1 registration's own record (``beta_policy_screening_g1``), not the scene's.
RUNTIME_SOURCE_GAP = "website_navigation_scene_inputs_runtime_source_from_g1_registration"
_PACKET_FILES = (("packet_request", "native_task_arena_packet_request.v1.json"),
                 ("runtime_contract", "native_task_runtime_contract.v1.json"),
                 ("arena_scene_plan", "native_task_arena_scene_plan.v1.json"),
                 ("packet_receipt", "native_task_arena_packet_receipt.v1.json"))


class WebsiteNavigationSceneInputsError(ValueError):
    """A typed reason a navigation job's scene inputs cannot be written or read."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise WebsiteNavigationSceneInputsError(code)


def _file_record(path: Path) -> dict[str, Any]:
    from .beta_policy_screening_files import _record

    return _record(path)


def _finite(values: Any, length: int) -> list[float] | None:
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    return numbers if len(numbers) == length and all(math.isfinite(value) for value in numbers) else None


def _task_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    """The A1 navigation task spec, its seal intact; the start's yaw-only quaternion checked."""
    from .website_navigation_packet import TASK_SPEC_SCHEMA

    _require(isinstance(value, Mapping) and value.get("task_kind") == NAVIGATION_TASK_KIND,
             "website_navigation_scene_inputs_task_kind_invalid")
    spec = json.loads(json.dumps(value, allow_nan=False))
    _require(spec.get("schema_version") == TASK_SPEC_SCHEMA and spec.get("claim_ceiling") == CLAIM_CEILING
             and not set(CONTRACT_FIELDS) & set(spec)
             and spec.get("task_spec_digest") == canonical_digest(spec, digest_field="task_spec_digest"),
             "website_navigation_scene_inputs_task_spec_invalid")
    start = _finite(spec.get("start_pose_world"), 7)
    _require(start is not None and start[3] == 0.0 and start[4] == 0.0
             and abs(math.hypot(start[5], start[6]) - 1.0) <= 1e-9,
             "website_navigation_scene_inputs_start_pose_invalid")
    return spec


def _scene_assets(value: Mapping[str, Any]) -> dict[str, Any]:
    """The A2 scene files record, its seal intact, each file still the bytes it names."""
    from .website_navigation_scene_assets import SCHEMA as ASSETS_SCHEMA

    _require(isinstance(value, Mapping), "website_navigation_scene_inputs_scene_assets_invalid")
    assets = json.loads(json.dumps(value, allow_nan=False))
    _require(assets.get("schema_version") == ASSETS_SCHEMA and assets.get("claim_ceiling") == CLAIM_CEILING
             and assets.get("scene_edited") is False and assets.get("paid_construction_performed") is False
             and assets.get("digest") == canonical_digest(assets, digest_field="digest")
             and all(isinstance(assets.get(role), Mapping) for role in ("appearance", "collision")),
             "website_navigation_scene_inputs_scene_assets_invalid")
    for role in ("appearance", "collision"):
        row = assets[role]
        path = Path(str(row.get("path") or ""))
        observed = _file_record(path) if path.is_absolute() and not path.is_symlink() and path.is_file() else {}
        _require(observed.get("sha256") == row.get("sha256") and observed.get("size_bytes") == row.get("size_bytes"),
                 "website_navigation_scene_inputs_scene_asset_changed")
    return assets


def _stage(source: Path, destination: Path, *, sha256: str, size_bytes: int) -> None:
    """A verified copy under the packet's evidence root (the G1 preparation's exclusive publish)."""
    from .beta_policy_screening_g1_prepare import _copy

    try:
        _copy(source, destination, sha256=sha256, size_bytes=size_bytes)
    except (OSError, ValueError) as exc:
        raise WebsiteNavigationSceneInputsError("website_navigation_scene_inputs_source_changed") from exc


def _standing_pose(start_pose_world: Sequence[float]) -> tuple[dict[str, Any], float, float]:
    """The G1's root at the compiled start: x, y and yaw as compiled, the pelvis on the compiled floor.

    The compiled start's z is the floor the compiler inferred from the levelled collider
    (``task_spec_compiler``); the G1 stands its profile's pelvis height above it, as the G1
    preparation's stance search places it (``ground_stance(root_height_m=pelvis_height_m)``).
    """
    from .scene_placement.robot_profile import UNITREE_G1_PROFILE

    x, y, floor, qx, qy, qz, qw = (float(value) for value in start_pose_world)
    height = float(UNITREE_G1_PROFILE.pelvis_height_m)
    pose = {"position_world_m": [x, y, floor + height], "orientation_xyzw": [qx, qy, qz, qw]}
    yaw = 2.0 * math.atan2(qz, qw)
    return pose, math.atan2(math.sin(yaw), math.cos(yaw)), height


def _overview_camera(*, root: Sequence[float], start: Sequence[float], target: Sequence[float],
                     intrinsics: Mapping[str, Any]) -> dict[str, Any]:
    """The team scene's overview rule (``team_scene_packet._cameras``) over the start-to-goal route."""
    from .task_evaluation_diagnostic_native_arena_compiler import _world_camera

    direction = [float(target[0]) - float(start[0]), float(target[1]) - float(start[1])]
    norm = math.hypot(*direction) or 1.0
    lateral = [-direction[1] / norm, direction[0] / norm]
    midpoint = [(float(start[i]) + float(target[i])) / 2.0 for i in range(3)]
    camera = _world_camera("overview", position=[midpoint[0] + 0.9 * lateral[0], midpoint[1] + 0.9 * lateral[1],
                                                 max(float(root[2]), float(start[2])) + 1.45],
                           target=midpoint, intrinsics=intrinsics)
    camera["policy_input"] = False
    return camera


def _nominal_scenario() -> dict[str, Any]:
    """The scene's nominal cell (``team_candidate_quick10_registry.NOMINAL_CELL``), no perturbation."""
    from .native_task_arena_packet import EVALUATION_INSTANCE_SCHEMA_VERSION
    from .team_candidate_quick10_registry import NOMINAL_CELL

    context = {"schema_version": EVALUATION_INSTANCE_SCHEMA_VERSION, "cell_id": NOMINAL_CELL["cell_id"],
               "seed": NOMINAL_CELL["seed"], "family": NOMINAL_CELL["family"], "partition": "development",
               "policy_neutral": True, "caller_asserted_success": False,
               "learned_policy_outcomes_consulted": False, "factor_records": [], "resolved_parameters": {},
               "scenario_scope": "native_frozen_reset_seeds"}
    context["instance_digest"] = canonical_digest(context, digest_field="instance_digest")
    return {"context_kind": "evaluation_cell", "cell_id": context["cell_id"], "seed": context["seed"],
            "instance_digest": context["instance_digest"], "context_document": context}


def _assumptions(*, root_height_m: float) -> list[str]:
    return [
        f"g1_pelvis_{root_height_m:.2f}m_above_compiled_floor_profile_height_not_measured_from_this_usd",
        "g1_stance_is_the_compiled_start_no_clear_footprint_search_run_here",
        "g1_spawn_clearance_not_measured_here",
        "head_camera_is_the_existing_g1_development_mount_simulation_assumption",
        "overview_camera_added_because_the_g1_contract_requires_one_review_only_not_visibility_checked",
        "overview_camera_pose_is_the_team_scene_rule_0.9m_beside_the_route_midpoint_1.45m_above_the_root",
        "overview_camera_intrinsics_are_the_head_cameras",
        "task_hand_right_is_a_packet_field_only_a_navigation_task_grasps_nothing",
        f"physics_frequency_{PHYSICS_FREQUENCY_HZ:g}hz_as_the_g1_preparation",
        "nominal_scenario_cell_only_no_perturbation",
        "appearance_is_the_renderer_unqualified_website_particlefield_no_appearance_variant_recorded",
        "scene_assets_and_task_spec_bound_by_the_handoff_that_wrote_both_not_by_a_shared_digest",
        "source_identity_is_bound_by_the_jobs_website_intake_receipt_not_by_this_record",
    ]


def _request(*, task: Mapping[str, Any], contract: Mapping[str, Any], assets: Mapping[str, Any],
             robot: Mapping[str, Any], scene_rows: Sequence[Mapping[str, Any]],
             assumptions: Sequence[str]) -> dict[str, Any]:
    from .beta_policy_screening_g1_prepare import HEAD_CAMERA
    from .native_task_arena_packet import REQUEST_SCHEMA_VERSION

    goal = task["g1_navigation_goal"]["center_world_m"]
    overview = _overview_camera(root=robot["base_pose_world"]["position_world_m"], start=task["start_pose_world"][:3],
                                target=goal, intrinsics=HEAD_CAMERA["intrinsics"])
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION, "scene_id": task["site_id"], "task_id": task["task_id"],
        "task_spec": {**task, "task_success_contract": dict(contract),
                      "task_success_contract_digest": contract["contract_digest"]},
        "task_joint_bindings": [], "task_state_binding": None,
        "assets": [dict(row) for row in scene_rows],
        "robot_configuration": dict(robot), "robot_base_pose_world": robot["base_pose_world"],
        "robot_joint_reset_positions_rad": robot["joint_reset_positions_rad"],
        "cameras": [json.loads(json.dumps(HEAD_CAMERA)), overview],
        "scenario": _nominal_scenario(), "physics_frequency_hz": PHYSICS_FREQUENCY_HZ,
        "navigation_scene_derivation": {
            "schema_version": "website_navigation_scene_derivation.v1", "claim_ceiling": CLAIM_CEILING,
            "navigation_task_spec_digest": task["task_spec_digest"], "scene_assets_digest": assets["digest"],
            "start_pose_world": list(task["start_pose_world"]), "disclosed_assumptions": list(assumptions)},
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


@read_scoped
def write_navigation_scene_inputs(*, navigation_task_spec: Mapping[str, Any], scene_assets: Mapping[str, Any],
                                  robot_record: Mapping[str, Any], output_root: str | Path,
                                  capture_root: str | Path) -> dict[str, Any]:
    """Compile the base G1 packet under ``output_root/<inputs key>/`` and seal its scene-inputs record.

    ``robot_record`` is the pinned G1 USD's file record (``path``, ``sha256``, ``size_bytes``), as the
    G1 preparation takes it. Returns ``{"path", "record"}``. A second call with the same inputs
    re-verifies and returns the same record without rewriting anything; a record its reader refuses
    (unreadable, changed, or written under an earlier contract) is kept aside, as a killed compile's
    packet is, and written again. The robot-free probe receipt is written beside the packet whether it
    passed or not; the probe gate decides what it admits.

    Privacy: ``capture_root`` is the job's website capture (``scenes/<site_id>/captures/<capture>``,
    the task's own site) and ``output_root`` must lie inside it, so the site's withdrawal deletes
    everything written here (``website_capture_withdrawal_deletion``); any other root is refused. A
    capture withdrawn on this host (``website_atlas_preview_store.capture_withdrawn``, the check the
    rigid team scene compile makes before it compiles) writes nothing, checked again before the probe
    receipt and the record are published.
    """
    from . import beta_policy_screening_g1_prepare as prepare
    from . import native_g1_arena_robot_configuration as g1_robot
    from .beta_policy_screening_files import _bound_path
    from .native_task_arena_packet import materialize_native_task_arena_packet
    from .native_task_robot_contact_topology import _asset_rigid_body_paths
    from .navigation_robot_free_probe import score_navigation_robot_free_probe
    from .team_scene_packet import EXECUTION_AUTHORITY, PENDING_CONTROLS

    task = _task_spec(navigation_task_spec)
    try:
        contract = navigation_task_success_contract(task)
    except ValueError as exc:
        raise WebsiteNavigationSceneInputsError("website_navigation_scene_inputs_task_spec_invalid") from exc
    assets = _scene_assets(scene_assets)
    expected_robot = ("sha256:" + g1_robot.G1_USD_SHA256, g1_robot.G1_USD_SIZE_BYTES)
    try:
        robot_source = _bound_path(robot_record)
    except (AttributeError, OSError, ValueError) as exc:
        raise WebsiteNavigationSceneInputsError("website_navigation_scene_inputs_robot_asset_mismatch") from exc
    _require((robot_record.get("sha256"), robot_record.get("size_bytes")) == expected_robot,
             "website_navigation_scene_inputs_robot_asset_mismatch")
    bound = {"schema_version": SCHEMA, "navigation_task_spec_digest": task["task_spec_digest"],
             "scene_assets_digest": assets["digest"], "robot_usd_sha256": expected_robot[0]}
    key = canonical_digest(bound)[7:]
    capture = _capture_root(capture_root, site_id=task["site_id"])
    _require(".." not in Path(output_root).parts, "website_navigation_scene_inputs_output_root_outside_capture")
    root = Path(output_root).absolute() / key
    _require(not any(part.is_symlink() for part in (root, *root.parents)),
             "website_navigation_scene_inputs_symlink_refused")
    _require(root.is_relative_to(capture), "website_navigation_scene_inputs_output_root_outside_capture")
    _require_not_withdrawn(capture)
    path, packet, evidence = root / RECORD_FILENAME, root / PACKET_DIRECTORY, root / INPUTS_DIRECTORY
    probe_path = root / PROBE_RECEIPT_FILENAME
    with ExitStack() as held:
        try:
            held.enter_context(prepare._namespace_lock(root.parent / ("." + key + ".lock")))
        except ValueError as exc:
            raise WebsiteNavigationSceneInputsError("website_navigation_scene_inputs_in_progress") from exc
        if path.exists() or path.is_symlink():
            try:
                record = read_navigation_scene_inputs(path)
            except WebsiteNavigationSceneInputsError:
                record = None  # Unreadable, changed or an earlier contract's: set aside below, compiled again.
            if record is not None:
                _require(all(record.get(name) == bound[name] for name in ("navigation_task_spec_digest",
                                                                           "scene_assets_digest"))
                         and record["robot"]["usd_sha256"] == bound["robot_usd_sha256"],
                         "website_navigation_scene_inputs_existing_conflict")
                return {"path": str(path), "record": record}
        for incomplete in (path, packet, probe_path):
            if incomplete.exists() or incomplete.is_symlink():
                # A killed compile, or a record its reader refuses: keep its bytes aside (inside the
                # capture, so a withdrawal deletes them too) and compile the same path again.
                try:
                    prepare._preserve_incomplete(incomplete)
                except (OSError, ValueError) as exc:
                    raise WebsiteNavigationSceneInputsError("website_navigation_scene_inputs_symlink_refused") from exc
        rows = []
        for role, semantic in (("collision", "scene_collision"), ("appearance", "scene_appearance")):
            row = assets[role]
            source = Path(row["path"])
            filename = semantic + source.suffix
            _stage(source, evidence / "scene" / filename, sha256=row["sha256"], size_bytes=row["size_bytes"])
            rows.append({"semantic_role": semantic, "filename": filename,
                         "source": {"root": "evidence", "relative_path": "scene/" + filename,
                                    "size_bytes": row["size_bytes"], "sha256": row["sha256"]},
                         "pose_world": {"position_world_m": [0.0, 0.0, 0.0],
                                        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}})
        robot_asset = evidence / "robot" / robot_source.name
        _stage(robot_source, robot_asset, sha256=expected_robot[0], size_bytes=expected_robot[1])
        pose, yaw, height = _standing_pose(task["start_pose_world"])
        try:
            robot = g1_robot.build_pinned_g1_arena_robot_configuration(
                asset=robot_asset, evidence_root=evidence, base_pose_world=pose, task_hand=prepare.TASK_HAND)
        except ValueError as exc:
            raise WebsiteNavigationSceneInputsError("website_navigation_scene_inputs_robot_invalid:" + str(exc)) from exc
        _require(prepare.HEAD_CAMERA["parent_prim_path"] in _asset_rigid_body_paths(robot_asset),
                 "website_navigation_scene_inputs_head_body_missing")
        assumptions = _assumptions(root_height_m=height)
        request = _request(task=task, contract=contract, assets=assets, robot=robot, scene_rows=rows,
                           assumptions=assumptions)
        try:
            receipt = materialize_native_task_arena_packet(request=request, evidence_root=evidence, output_dir=packet,
                                                           link_sources_within=evidence)
        except ValueError as exc:
            raise WebsiteNavigationSceneInputsError(
                "website_navigation_scene_inputs_packet_invalid:" + str(exc)[:300]) from exc
        plan = json.loads((packet / "native_task_arena_scene_plan.v1.json").read_text(encoding="utf-8"))
        files = {role: _file_record(packet / name) for role, name in _PACKET_FILES}
        try:
            probe = score_navigation_robot_free_probe(plan=plan)
        except ValueError as exc:
            raise WebsiteNavigationSceneInputsError(
                "website_navigation_scene_inputs_robot_free_probe_invalid:" + str(exc)[:200]) from exc
        _require_not_withdrawn(capture)
        _write_once(probe_path, probe)
        value = {
            "schema_version": SCHEMA, "claim_ceiling": CLAIM_CEILING, "task_kind": NAVIGATION_TASK_KIND,
            "scene_id": task["site_id"], "task_id": task["task_id"],
            "scene_revision_digest": plan["plan_digest"], "plan_digest": plan["plan_digest"],
            "configuration_digest": canonical_digest({"scene_revision_digest": plan["plan_digest"],
                                                      "packet": files["packet_receipt"]["sha256"],
                                                      "robot_base_pose_world": pose}),
            "task_success_contract": contract, "task_success_contract_digest": contract["contract_digest"],
            "runtime_contract_digest": receipt["runtime_contract_digest"],
            "packet_request_digest": request["request_digest"], "packet_receipt_digest": receipt["receipt_digest"],
            "base_native_packet": files["packet_receipt"], "construction_result": files["packet_receipt"],
            "packet_root": str(packet.resolve()), "packet_files": files,
            "navigation_task_spec_digest": task["task_spec_digest"],
            "compiled_task_spec_digest": task["compiled_task_spec_digest"],
            "scene_assets_digest": assets["digest"],
            "scene_assets": {"frame_digest": assets.get("frame_digest"),
                             "appearance_sha256": assets["appearance"]["sha256"],
                             "collision_sha256": assets["collision"]["sha256"]},
            "robot": {"robot_id": "unitree_g1", "usd_sha256": robot["usd_sha256"], "base_pose_world": pose,
                      "start_pose_world": list(task["start_pose_world"]), "start_yaw_rad": yaw,
                      "root_height_above_floor_m": height, "camera_roles": ["head", "overview"]},
            "runtime_source": None, "runtime_source_gap": RUNTIME_SOURCE_GAP,
            "source": None,
            "source_gap": {"code": SOURCE_UNBOUND, "missing": ["capture_id", "scene_intent_id", "scene_intent_digest"],
                           "known": {"task_id": task["task_id"], "scene_revision_digest": plan["plan_digest"],
                                     "task_context_digest": task["navigation_goal_basis"]["task_context_digest"]}},
            "execution_authority": dict(EXECUTION_AUTHORITY), "control_diagnostic": dict(PENDING_CONTROLS),
            "robot_free_probe": {**_file_record(probe_path), "schema_version": probe["schema_version"],
                                 "receipt_digest": probe["receipt_digest"], "status": probe["status"]},
            "disclosed_assumptions": assumptions,
            "native_application_claimed": False, "policy_episode_claimed": False,
            "provider_mutation_performed": False, "paid_authority_granted": False,
            DIGEST_FIELD: "",
        }
        value[DIGEST_FIELD] = canonical_digest(value, digest_field=DIGEST_FIELD)
        _require_not_withdrawn(capture)
        _write_once(path, value)
        return {"path": str(path), "record": read_navigation_scene_inputs(path)}


def _capture_root(value: str | Path, *, site_id: str) -> Path:
    """The job's website capture directory, ``scenes/<site_id>/captures/<capture>``, the task's own site."""
    capture = Path(value).absolute()
    _require(".." not in Path(value).parts and capture.parent.name == "captures"
             and capture.parent.parent.name == site_id and site_id.startswith("site-")
             and not any(part.is_symlink() for part in (capture, *capture.parents)) and capture.is_dir(),
             "website_navigation_scene_inputs_capture_root_invalid")
    return capture


def _require_not_withdrawn(capture: Path) -> None:
    from .website_atlas_preview_store import capture_withdrawn

    _require(not capture_withdrawn(capture), "website_navigation_scene_inputs_capture_withdrawn")


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name("." + path.name + "." + os.urandom(8).hex())
    try:
        temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
                             encoding="utf-8")
        temporary.chmod(0o444)
        os.link(temporary, path)  # Exclusive publish, never overwrite.
    finally:
        temporary.unlink(missing_ok=True)


def _strip_contract(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in task_spec.items() if key not in CONTRACT_FIELDS}


def read_navigation_scene_inputs(path: str | Path) -> dict[str, Any]:
    """Reopen the record: its seal, every packet file's bytes, and each binding, or raise by name."""
    from .native_task_arena_bundle import verify_native_task_arena_packet

    record_path = Path(path)
    try:
        _require(not record_path.is_symlink(), "website_navigation_scene_inputs_unreadable")
        value = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WebsiteNavigationSceneInputsError("website_navigation_scene_inputs_unreadable") from exc
    _require(isinstance(value, dict) and value.get("schema_version") == SCHEMA
             and value.get("claim_ceiling") == CLAIM_CEILING
             and value.get(DIGEST_FIELD) == canonical_digest(value, digest_field=DIGEST_FIELD),
             "website_navigation_scene_inputs_seal_invalid")
    _require(value.get("task_kind") == NAVIGATION_TASK_KIND, "website_navigation_scene_inputs_task_kind_invalid")
    try:
        packet = Path(value["packet_root"])
        root, receipt, _ = verify_native_task_arena_packet(packet)
        files = value["packet_files"]
        changed = (root != packet or set(files) != {role for role, _ in _PACKET_FILES}
                   or any(files[role] != _file_record(root / name) for role, name in _PACKET_FILES))
        robot_binding = receipt["robot_asset_binding"]
        changed = changed or (_file_record(root / robot_binding["staged_relative_path"])["sha256"]
                              != robot_binding["staged_sha256"])
    except (KeyError, TypeError, OSError, ValueError) as exc:
        raise WebsiteNavigationSceneInputsError("website_navigation_scene_inputs_packet_changed") from exc
    _require(not changed, "website_navigation_scene_inputs_packet_changed")
    try:
        plan = json.loads((root / "native_task_arena_scene_plan.v1.json").read_text(encoding="utf-8"))
        task = plan["task_spec"]
        contract = validate_navigation_task_success_contract(value["task_success_contract"], task_spec=task)
        sources = {row["semantic_role"]: row["source"]["sha256"] for row in receipt["source_bindings"]}
        robot = value["robot"]
        bound = (
            value["base_native_packet"] == value["construction_result"] == files["packet_receipt"]
            and plan["plan_digest"] == canonical_digest(plan, digest_field="plan_digest")
            and value["scene_revision_digest"] == value["plan_digest"] == plan["plan_digest"]
            == receipt["arena_scene_plan_digest"]
            and value["runtime_contract_digest"] == receipt["runtime_contract_digest"]
            == plan["runtime_contract_digest"]
            and value["packet_receipt_digest"] == receipt["receipt_digest"]
            and value["packet_request_digest"] == receipt["request_digest"]
            and plan["task_kind"] == NAVIGATION_TASK_KIND and plan["robot"]["robot_id"] == "unitree_g1"
            and plan["robot"]["base_pose_world"] == robot["base_pose_world"]
            and plan["robot"]["usd_sha256"] == robot["usd_sha256"]
            and task["task_success_contract"] == contract
            and value["task_success_contract_digest"] == task["task_success_contract_digest"]
            == contract["contract_digest"]
            and canonical_digest(_strip_contract(task), digest_field="task_spec_digest")
            == task["task_spec_digest"] == value["navigation_task_spec_digest"]
            and task["compiled_task_spec_digest"] == value["compiled_task_spec_digest"]
            and task["start_pose_world"] == robot["start_pose_world"]
            and sources == {"scene_collision": value["scene_assets"]["collision_sha256"],
                            "scene_appearance": value["scene_assets"]["appearance_sha256"]}
            and value["source"] is None and value["source_gap"]["code"] == SOURCE_UNBOUND
        )
    except (KeyError, TypeError, OSError, ValueError) as exc:
        raise WebsiteNavigationSceneInputsError("website_navigation_scene_inputs_binding_changed") from exc
    _require(bound, "website_navigation_scene_inputs_binding_changed")
    _verify_probe(value, packet_root=root, plan=plan)
    return json.loads(json.dumps(value, allow_nan=False))


def _verify_probe(value: Mapping[str, Any], *, packet_root: Path, plan: Mapping[str, Any]) -> None:
    """The probe receipt beside the packet: its bytes as recorded, its seal, and this plan's binding."""
    from .navigation_robot_free_probe import validate_navigation_robot_free_probe_receipt

    code = "website_navigation_scene_inputs_robot_free_probe_changed"
    try:
        recorded = value["robot_free_probe"]
        path = Path(recorded["path"])
        _require(path == packet_root.parent / PROBE_RECEIPT_FILENAME
                 and not any(part.is_symlink() for part in (path, *path.parents)) and path.is_file(), code)
        _require(_file_record(path) == {key: recorded[key] for key in ("path", "size_bytes", "sha256")}, code)
        receipt = validate_navigation_robot_free_probe_receipt(json.loads(path.read_text(encoding="utf-8")),
                                                               plan=plan)
        _require(all(receipt[key] == recorded[key] for key in ("schema_version", "receipt_digest", "status")), code)
    except WebsiteNavigationSceneInputsError:
        raise
    except (KeyError, TypeError, OSError, UnicodeError, ValueError) as exc:
        raise WebsiteNavigationSceneInputsError(code) from exc


def derived_profile_row(written: Mapping[str, Any], *, capture_root: str | Path) -> dict[str, Any]:
    """The team registry's derived row for a written record (``write_derived_profile``): scene facts only.

    ``written`` is :func:`write_navigation_scene_inputs`'s result. The row names the record, the
    directory its robot-free probe receipt sits in, its revision and its nominal scenario cell, as the
    packet's own plan states it; never approvers, rights, rates or a salt.
    """
    record = written["record"]
    plan = json.loads((Path(record["packet_root"]) / "native_task_arena_scene_plan.v1.json").read_text(encoding="utf-8"))
    return {"capture_root": str(Path(capture_root).resolve()), "scene_id": record["scene_id"],
            "task_id": record["task_id"], "scenario_id": plan["scenario"]["cell_id"],
            "robot_free_probe_root": str(Path(record["robot_free_probe"]["path"]).parent),
            "scene_revision_digest": record["scene_revision_digest"],
            "navigation_scene_inputs_path": str(written["path"])}


def screening_source(record: Mapping[str, Any]) -> dict[str, Any]:
    """The current-source identity the screening lane binds (``SOURCE_FIELDS``), from this job's intake receipt.

    ``record`` is the record its reader has just verified. A scene kept as filmed gets its scene intent
    from the website's navigation intake (``prepared-scene``); the Pipeline's intake receipt beside the
    record (``website_navigation_intake.bind_navigation_intake_receipt``) binds the intent's id and digest,
    the capture, the task, the task context and this record's scene revision. Without a receipt (no
    intake yet, or one the website refused, such as an unconfirmed brief) this refuses by name.
    """
    from .website_navigation_intake import read_navigation_intake_receipt

    try:
        path = Path(record["packet_root"]).parent / RECORD_FILENAME
    except (KeyError, TypeError) as exc:
        raise WebsiteNavigationSceneInputsError(SOURCE_UNBOUND) from exc
    # A receipt that no longer reads or binds this record refuses by its own name.
    receipt = read_navigation_intake_receipt(path, record)
    if receipt is None:
        raise WebsiteNavigationSceneInputsError(SOURCE_UNBOUND)
    return dict(receipt["source"])


__all__ = ["CLAIM_CEILING", "OUTPUT_DIRECTORY", "RECORD_FILENAME", "SCHEMA", "SOURCE_UNBOUND",
           "WebsiteNavigationSceneInputsError", "derived_profile_row",
           "read_navigation_scene_inputs", "screening_source", "write_navigation_scene_inputs"]
