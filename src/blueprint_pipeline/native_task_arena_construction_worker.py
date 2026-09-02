"""Execute one sealed, task-neutral native Franka construction canary.

The worker consumes only ``native_task_packet`` and its provider manifest.  It
does not know a scene id, object class, task coordinate, or candidate outcome.
It verifies the complete dependency matrix before scene construction, applies
the exact Arena plan, measures reset/contact/camera state, and drives the
Franka finger midpoint through the contact-clear phase plan using the same 8-D
absolute action seam later used by controls and learned policies.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence

from blueprint_pipeline.native_franka_action_math import (
    is_unauthored_identity_quaternion_xyzw,
)
from blueprint_pipeline.native_franka_grasp_geometry import (
    measure_live_robotiq_grasp_geometry,
)
from blueprint_pipeline.native_task_arena_dependency_contract import (
    NATIVE_TASK_ARENA_DEPENDENCY_IMPORTS as DEPENDENCY_IMPORTS,
)
from blueprint_pipeline.native_task_arena_feedback_bootstrap_runtime import (
    feedback_bootstrap_result,
    verified_construction_phase_plan_path,
)
from blueprint_pipeline.native_task_curobo_path_execution import (
    advance_solver_waypoint,
    solver_command_target,
    solver_path_result_fields,
    validated_solver_joint_sequence,
)
from blueprint_pipeline.native_task_nurec_render_setup import (
    prepare_site_appearance_renderer as _prepare_site_appearance_renderer,
)
from blueprint_pipeline.native_task_servo_command_limits import (
    servo_command_limits as _servo_command_limits,
)
from blueprint_pipeline.rigid_frame_transforms import (
    quaternion_conjugate_xyzw,
    rotate_vector_xyzw,
)
from pathlib import Path
from typing import Any


RESULT_SCHEMA_VERSION = "native_task_arena_construction_result.v1"
RESULT_FILENAME = "native_task_arena_construction_result.v1.json"
CAMERA_THRESHOLDS = {
    "external": {"minimum_pixels": 200, "minimum_pixel_fraction": 0.003},
    "wrist": {"minimum_pixels": 120, "minimum_pixel_fraction": 0.002},
    "overview": {"minimum_pixels": 200, "minimum_pixel_fraction": 0.003},
}

# Whether this runtime can render the captured-site appearance volume at all.
#
# The Arena bundle is pinned separately to Isaac Sim 6.0.1. The sealed site is
# a plain NuRec volume (no authored ``info:spg:sourceAsset``), so it does not
# execute an SPG/PPISP graph. Live 6.0.1 evidence showed the plain volume remained
# void without the image's ``omni.rtx.spg`` renderer component loaded at Kit launch.
# Loading it is the remaining explicit hypothesis; only the pixel gate below may
# say whether that made the site render. The preflight keeps the claims separate:
# it verifies the asset classification and refuses unless the launch-time renderer extension,
# plain-volume settings, and renderer hints read back as qualified, while the packet gate
# verifies the authored ``OmniNuRecFieldAsset`` type-name signal. NVIDIA's own
# 6.0.1 NuRec utilities define and detect that raw type name without requiring
# a concrete ``Usd.SchemaRegistry`` entry. The camera gate remains the independent
# content-level check: an available renderer cannot vouch that this exact
# captured site contributed pixels.
SITE_APPEARANCE_RENDER_EXPECTED = True


def _announce(phase: str, status: str = "started") -> None:
    print(
        f"BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena:{phase}:{status}",
        flush=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any], *, field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _quaternion_angle_xyzw(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        raise RuntimeError("native_task_construction_quaternion_invalid")
    qa = [float(value) for value in a]
    qb = [float(value) for value in b]
    if not all(math.isfinite(value) for value in [*qa, *qb]):
        raise RuntimeError("native_task_construction_quaternion_invalid")
    norm_a = math.sqrt(sum(value * value for value in qa))
    norm_b = math.sqrt(sum(value * value for value in qb))
    if norm_a <= 0.0 or norm_b <= 0.0:
        raise RuntimeError("native_task_construction_quaternion_invalid")
    dot = abs(
        sum(left * right for left, right in zip(qa, qb, strict=True))
        / (norm_a * norm_b)
    )
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _pose_arrival_readback(
    *,
    position_world_m: Sequence[float],
    target_position_world_m: Sequence[float],
    orientation_world_xyzw: Sequence[float],
    target_orientation_world_xyzw: Sequence[float],
    position_tolerance_m: float,
    orientation_tolerance_rad: float | None,
) -> dict[str, Any]:
    position_error = math.dist(position_world_m, target_position_world_m)
    orientation_error = _quaternion_angle_xyzw(
        orientation_world_xyzw, target_orientation_world_xyzw
    )
    reached = position_error <= float(position_tolerance_m) and (
        orientation_tolerance_rad is None
        or orientation_error <= float(orientation_tolerance_rad)
    )
    return {
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "reached": reached,
    }


def _terminal_grasp_frame_arrival_readback(
    *,
    grasp_pose_world: Sequence[float],
    body_pose_world: Sequence[float],
    target_position_world_m: Sequence[float],
    target_orientation_world_xyzw: Sequence[float],
    position_tolerance_m: float,
    orientation_tolerance_rad: float | None,
) -> dict[str, Any]:
    """Judge arrival in the commanded TCP frame while retaining the body pose."""

    arrival = _pose_arrival_readback(
        position_world_m=grasp_pose_world[:3],
        target_position_world_m=target_position_world_m,
        orientation_world_xyzw=grasp_pose_world[3:7],
        target_orientation_world_xyzw=target_orientation_world_xyzw,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad,
    )
    return {
        **arrival,
        "terminal_grasp_frame_orientation_world_xyzw": list(
            grasp_pose_world[3:7]
        ),
        "terminal_body_orientation_world_xyzw": list(body_pose_world[3:7]),
    }


def _retain_task_path_samples(*, task_kind: str, task_spec: Mapping[str, Any]) -> bool:
    return task_kind == "rigid_pick_place" or (
        task_kind == "articulated_open_close"
        and task_spec.get("schema_version") == "adp_task_spec.v2"
    )


def _evaluate_task_construction_gates(
    *,
    phase_plan: Mapping[str, Any],
    phase_results: Sequence[Mapping[str, Any]],
    reset_replay: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    from blueprint_pipeline.native_task_construction_plan import (
        evaluate_graph_articulated_construction_gates,
        evaluate_rigid_construction_gates,
    )

    schema = phase_plan.get("schema_version")
    if schema == "native_rigid_construction_phase_plan.v1":
        return (
            "rigid_construction_gates",
            evaluate_rigid_construction_gates(
                phase_plan=phase_plan,
                phase_results=phase_results,
                reset_replay=reset_replay,
            ),
        )
    if schema == "native_articulated_graph_construction_phase_plan.v1":
        return (
            "articulated_graph_construction_gates",
            evaluate_graph_articulated_construction_gates(
                phase_plan=phase_plan,
                phase_results=phase_results,
                reset_replay=reset_replay,
            ),
        )
    return None


def _jsonable(value: Any) -> Any:
    if hasattr(value, "detach"):  # torch tensor
        value = value.detach().cpu()
    if not hasattr(value, "tolist") and hasattr(value, "numpy"):
        # warp arrays have neither `detach` nor `tolist`, so without this they
        # fell through unconverted and every downstream use failed far away
        # from here -- r12 died on `_jsonable(robot.data.root_pose_w)[0]` with
        # "Item indexing is not supported on wp.array objects", after a clean
        # environment build. Isaac Lab's physics views return warp arrays, so
        # this is the normal type here, not an exotic one.
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def preflight_native_dependency_matrix(*, robot_id: str) -> dict[str, Any]:
    """Probe all worker imports and media tools in one retained receipt."""

    imports = []
    blockers = []
    try:
        from blueprint_pipeline.native_task_arena_import_scope import (
            install_scoped_arena_embodiment,
        )

        embodiment_scope = install_scoped_arena_embodiment(robot_id)
    except Exception as exc:  # noqa: BLE001 - exact scope failure is evidence
        embodiment_scope = {
            "schema_version": "native_task_arena_embodiment_scope.v1",
            "robot_id": str(robot_id),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        blockers.append(f"native_task_arena_embodiment_scope_failed:{robot_id}")
    for name in DEPENDENCY_IMPORTS:
        try:
            module = importlib.import_module(name)
            imports.append(
                {
                    "module": name,
                    "available": True,
                    "version": str(getattr(module, "__version__", "unreported")),
                }
            )
        except Exception as exc:  # noqa: BLE001 - exact missing matrix is evidence
            imports.append(
                {
                    "module": name,
                    "available": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            blockers.append(f"native_task_dependency_missing:{name}")
    tools = []
    for executable in ("ffmpeg", "ffprobe"):
        try:
            completed = subprocess.run(
                [executable, "-version"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            tools.append(
                {
                    "executable": executable,
                    "available": False,
                    "returncode": None,
                    "version_line": "",
                }
            )
            blockers.append(f"native_task_dependency_missing:{executable}")
            continue
        tools.append(
            {
                "executable": executable,
                "available": completed.returncode == 0,
                "returncode": completed.returncode,
                "version_line": (
                    (completed.stdout or completed.stderr).splitlines() or [""]
                )[0],
            }
        )
        if completed.returncode != 0:
            blockers.append(f"native_task_dependency_missing:{executable}")
    return {
        "schema_version": "native_task_dependency_matrix.v1",
        "embodiment_scope": embodiment_scope,
        "imports": imports,
        "tools": tools,
        "all_required_available": not blockers,
        "blockers": sorted(set(blockers)),
    }


def _persist(output: Path, result: dict[str, Any]) -> None:
    # Normalise before digesting. This runs from a `finally`, and
    # `_canonical_digest` refuses values json cannot encode -- a stray warp
    # array or Path would raise *inside* the handler, replace the real
    # exception and leave a paid run with no receipt at all. Passing
    # `default=str` to the write alone is not enough, because the digest is
    # computed first. Normalising both also makes the digest describe exactly
    # the bytes on disk.
    normalised = json.loads(json.dumps(result, default=str))
    normalised["result_digest"] = _canonical_digest(
        normalised, field="result_digest"
    )
    result["result_digest"] = normalised["result_digest"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(normalised, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_and_verify_manifest(
    runtime: Path, *, expected_execution_mode: str = "construction_canary"
) -> dict[str, Any]:
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    path = runtime / "adp_arena_provider_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "native_task_arena_provider_bundle.v1"
        or manifest.get("execution_mode") != expected_execution_mode
        or manifest.get("input_digest")
        != canonical_digest(manifest, digest_field="input_digest")
    ):
        raise RuntimeError("native_task_construction_manifest_invalid")
    return manifest


def _body_pose_world(
    robot: Any, *, body_name: str, torch: Any
) -> list[float]:
    names = list(robot.data.body_names)
    if body_name not in names:
        raise RuntimeError(f"native_task_gripper_body_missing:{body_name}")
    pose = torch.as_tensor(robot.data.body_pose_w)[0, names.index(body_name), :7]
    result = [float(value) for value in pose]
    norm = math.sqrt(sum(value * value for value in result[3:7]))
    if len(result) != 7 or not all(math.isfinite(value) for value in result) or norm <= 0.0:
        raise RuntimeError(f"native_task_gripper_body_pose_invalid:{body_name}")
    result[3:7] = [value / norm for value in result[3:7]]
    return result


def _pad_centers_from_finger_body_offsets(
    *, robot: Any, offsets_body_m: Mapping[str, Sequence[float]], torch: Any
) -> dict[str, list[float]]:
    centers: dict[str, list[float]] = {}
    for side in ("left", "right"):
        pose = _body_pose_world(
            robot, body_name=f"{side}_inner_finger", torch=torch
        )
        offset = rotate_vector_xyzw(pose[3:7], offsets_body_m[side])
        centers[side] = [pose[axis] + offset[axis] for axis in range(3)]
    return centers


def _pad_offsets_from_relative_geometry(
    geometry: Mapping[str, Any],
) -> dict[str, list[float]] | None:
    """Prefer collider-to-finger offsets measured in one coherent USD frame."""

    selected = geometry.get("selected_pad_colliders")
    if not isinstance(selected, Mapping):
        return None
    offsets: dict[str, list[float]] = {}
    for side in ("left", "right"):
        row = selected.get(side)
        if not isinstance(row, Mapping):
            return None
        raw = row.get("center_inner_finger_body_m")
        if raw is None:
            return None
        try:
            offset = [float(value) for value in raw]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"native_task_gripper_pad_relative_offset_invalid:{side}"
            ) from exc
        if len(offset) != 3 or not all(math.isfinite(value) for value in offset):
            raise RuntimeError(
                f"native_task_gripper_pad_relative_offset_invalid:{side}"
            )
        offsets[side] = offset
    return offsets


def _physical_pad_binding(
    *, robot: Any, torch: Any
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    body_name = next(
        (
            name
            for name in ("panda_hand", "base_link")
            if name in list(robot.data.body_names)
        ),
        None,
    )
    if body_name is None:
        raise RuntimeError("native_task_gripper_controlled_body_missing")
    body_pose = _body_pose_world(robot, body_name=body_name, torch=torch)
    geometry = measure_live_robotiq_grasp_geometry(
        stage=stage,
        controlled_body_position_world_m=body_pose[:3],
        controlled_body_quaternion_world_xyzw=body_pose[3:7],
    )
    relative_offsets = _pad_offsets_from_relative_geometry(geometry)
    if relative_offsets is not None:
        return geometry, relative_offsets
    offsets: dict[str, list[float]] = {}
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    for side in ("left", "right"):
        pad_path = geometry["selected_pad_colliders"][side]["prim_path"]
        finger_prim = stage.GetPrimAtPath(pad_path)
        while finger_prim and finger_prim.GetName() != f"{side}_inner_finger":
            finger_prim = finger_prim.GetParent()
        if not finger_prim:
            raise RuntimeError(
                f"native_task_gripper_pad_finger_ancestor_missing:{side}"
            )
        matrix = xforms.GetLocalToWorldTransform(finger_prim)
        translation = matrix.ExtractTranslation()
        quaternion = matrix.ExtractRotationQuat()
        imaginary = quaternion.GetImaginary()
        finger = [
            *[float(translation[axis]) for axis in range(3)],
            *[float(imaginary[axis]) for axis in range(3)],
            float(quaternion.GetReal()),
        ]
        world_offset = [
            float(geometry["pad_centers_world_m"][side][axis]) - finger[axis]
            for axis in range(3)
        ]
        offsets[side] = rotate_vector_xyzw(
            quaternion_conjugate_xyzw(finger[3:7]), world_offset
        )
    return geometry, offsets


def _requested_arm_reset(
    *, plan: Mapping[str, Any], servo_binding: Mapping[str, Any]
) -> list[float]:
    resets = plan["robot"]["joint_reset_positions_rad"]
    return [float(resets[name]) for name in servo_binding["arm_joint_names"]]


def _task_joint_reset_passed(
    *, absolute_errors_rad: Mapping[str, float], task_spec: Mapping[str, Any]
) -> bool:
    """Apply a joint tolerance only when the task scorer declares joint resets."""

    if not absolute_errors_rad:
        return True
    try:
        tolerance = float(task_spec["reset_tolerance_rad"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("native_task_joint_reset_tolerance_missing") from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise RuntimeError("native_task_joint_reset_tolerance_invalid")
    return max(float(value) for value in absolute_errors_rad.values()) <= tolerance


def _phase_target_orientation(
    phase: Mapping[str, Any], *, reset_body_orientation_xyzw: Sequence[float]
) -> list[float]:
    """Bind the grasp-frame phase orientation, treating identity as unspecified.

    ``native_articulated_control_plan._pose_phase`` already declares the intent:
    a phase with no orientation binds "the controlled body's measured reset
    orientation. No appliance-facing axis is guessed here."  That fallback was
    reachable only when the key was absent, so an identity placeholder -- which
    carries exactly the same meaning -- was executed as a real target instead.

    The argument retains its historical name for compatibility, but callers
    pass the measured reset *grasp-frame* orientation. Passing the controlled
    body orientation here became wrong once the TCP transform was measured:
    the returned quaternion is consumed as a grasp-frame target. These phases
    are open-gripper clearance and reachability probes, so the measured reset
    grasp orientation is a legitimate binding. It is not a contact grasp: the
    contact replay still refuses an unauthored orientation.
    """

    orientation = phase.get("orientation_world_xyzw")
    if is_unauthored_identity_quaternion_xyzw(orientation):
        return [float(value) for value in reset_body_orientation_xyzw]
    return [float(value) for value in orientation]


# What the pinned Arena embodiment actually applies to the 7 arm joints, read
# from the provisioned runtime source rather than inferred
# (IsaacLab-Arena/isaaclab_arena/embodiments/droid/droid.py, DroidSceneCfg):
#
#   panda_shoulder  panda_joint[1-4]  stiffness 400.0  damping 80.0
#                                     effort_limit 87.0  velocity_limit 2.175
#   panda_forearm   panda_joint[5-7]  stiffness 400.0  damping 80.0
#                                     effort_limit 12.0  velocity_limit 2.61
#   spawn.rigid_props.disable_gravity = True
#
# Those arm gains are already Isaac Lab's FRANKA_PANDA_HIGH_PD_CFG values
# (isaaclab_assets/robots/franka.py sets shoulder and forearm to 400.0/80.0 and
# disable_gravity True), so "adopt the stiffer upstream PD config" is a no-op
# here and the soft-gain hypothesis does not survive the pinned source.  For
# implicit actuators effort_limit and effort_limit_sim are synchronized
# (actuator_base.py: "For implicit actuators, the effort_limit and
# effort_limit_sim are the same"), so Arena naming the non-_sim field is also
# not a defect.  Two consequences worth measuring rather than assuming:
#
#   saturation error  = effort_limit / stiffness = 0.218 rad shoulder,
#                       0.030 rad forearm -- the forearm is already at maximum
#                       torque for any error past 0.03 rad
#   terminal speed    = (stiffness / damping) * lead = 5 * lead rad/s, capped by
#                       velocity_limit
#
# Isaac Lab has also renamed several joint-limit buffers across releases.  Probe
# the known names rather than pinning one, and retain which name resolved so the
# receipt says what was actually read instead of implying a value we guessed.
ACTUATOR_READBACK_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("joint_stiffness", ("joint_stiffness",)),
    ("joint_damping", ("joint_damping",)),
    (
        "joint_effort_limit_n_m",
        ("joint_effort_limits_sim", "joint_effort_limits", "joint_effort_limit"),
    ),
    (
        "joint_velocity_limit_rad_s",
        (
            "joint_velocity_limits_sim",
            "joint_velocity_limits",
            "joint_velocity_limit",
        ),
    ),
    ("joint_friction", ("joint_friction_coeff", "joint_friction")),
    ("applied_torque_n_m", ("applied_torque",)),
    ("computed_torque_n_m", ("computed_torque",)),
)


def _arm_slice(value: Any, *, joint_ids: Sequence[int]) -> list[float]:
    """Return the first environment's value for the bound arm joints."""

    row = _jsonable(value)
    if isinstance(row, list) and row and isinstance(row[0], list):
        row = row[0]
    if not isinstance(row, list):
        raise TypeError("actuator_readback_not_indexable")
    return [float(row[index]) for index in joint_ids]


def read_native_arm_actuator_readback(
    robot: Any, *, joint_ids: Sequence[int]
) -> dict[str, Any]:
    """Retain the arm actuator configuration that bounds achievable torque.

    A position-controlled joint can only develop ``stiffness * position_error``
    of restoring torque, and the command clamp bounds that error, so the gains,
    the effort limit, and the torque actually applied at a stall are the
    measurements that decide whether a phase failed because it was commanded too
    conservatively or because the actuator could never deliver the load.  None of
    that is currently recorded anywhere in a run receipt.

    This is diagnostic evidence, not a gate: a missing or renamed buffer is
    retained as an explicit unavailability rather than failing a paid run.
    """

    readback: dict[str, Any] = {
        "schema_version": "native_task_arena_arm_actuator_readback.v1",
        "arm_joint_ids": [int(index) for index in joint_ids],
    }
    data = getattr(robot, "data", None)
    for field, candidates in ACTUATOR_READBACK_FIELDS:
        resolved: str | None = None
        for candidate in candidates:
            if data is not None and getattr(data, candidate, None) is not None:
                resolved = candidate
                break
        if resolved is None:
            readback[field] = {
                "available": False,
                "reason": "attribute_absent",
                "probed_attributes": list(candidates),
            }
            continue
        try:
            readback[field] = _arm_slice(
                getattr(data, resolved), joint_ids=joint_ids
            )
            readback[f"{field}_source_attribute"] = resolved
        except Exception as exc:  # noqa: BLE001 - unreadable buffer is evidence
            readback[field] = {
                "available": False,
                "reason": f"{type(exc).__name__}:{exc}",
                "probed_attributes": list(candidates),
            }

    # Which joints an actuator group actually covers settles whether the
    # "Not all actuators are configured" warning is benign (passive gripper
    # linkage) or is silently leaving arm joints undriven.
    groups: list[dict[str, Any]] = []
    actuators = getattr(robot, "actuators", None)
    if isinstance(actuators, Mapping):
        for name, actuator in actuators.items():
            group: dict[str, Any] = {"name": str(name)}
            group["class"] = type(actuator).__name__
            for attribute in ("joint_names", "joint_indices"):
                try:
                    group[attribute] = _jsonable(getattr(actuator, attribute))
                except Exception as exc:  # noqa: BLE001
                    group[attribute] = f"unavailable:{type(exc).__name__}:{exc}"
            groups.append(group)
    readback["actuator_groups"] = groups
    try:
        joint_names = [str(name) for name in robot.joint_names]
    except Exception:  # noqa: BLE001
        joint_names = []
    covered: set[str] = set()
    for group in groups:
        names = group.get("joint_names")
        if isinstance(names, list):
            covered.update(str(name) for name in names)
    readback["joint_names"] = joint_names
    readback["unactuated_joint_names"] = (
        sorted(set(joint_names) - covered) if joint_names and covered else []
    )
    readback["arm_joint_names_without_actuator_group"] = (
        sorted(
            {
                joint_names[index]
                for index in joint_ids
                if 0 <= index < len(joint_names)
            }
            - covered
        )
        if joint_names and covered
        else []
    )
    for label, source in (
        ("disable_gravity", ("cfg", "spawn", "rigid_props", "disable_gravity")),
        ("is_fixed_base", ("is_fixed_base",)),
    ):
        cursor: Any = robot
        for attribute in source:
            cursor = getattr(cursor, attribute, None)
            if cursor is None:
                break
        readback[label] = (
            cursor if isinstance(cursor, bool) else {"available": False}
        )
    return readback


def _arm_buffer(
    robot: Any, attribute: str, *, joint_ids: Sequence[int]
) -> list[float] | None:
    """Return one arm-joint buffer, or ``None`` when it cannot be read."""

    data = getattr(robot, "data", None)
    value = getattr(data, attribute, None) if data is not None else None
    if value is None:
        return None
    try:
        return _arm_slice(value, joint_ids=joint_ids)
    except Exception:  # noqa: BLE001 - absence is retained, never fatal
        return None


def _applied_arm_torque(
    robot: Any, *, joint_ids: Sequence[int]
) -> list[float] | None:
    """Return the arm joints' applied torque, or ``None`` when unreadable."""

    return _arm_buffer(robot, "applied_torque", joint_ids=joint_ids)


def _commanded_arm_joint_target(
    robot: Any, *, joint_ids: Sequence[int]
) -> list[float] | None:
    """Return the position target the actuators actually received.

    The worker emits an absolute joint-position action, but everything between
    that action and the drive - the embodiment's action term, any scale or
    offset, any clip - is Arena's, not ours.  Retaining the realized target next
    to the action we sent is what distinguishes "our command was reshaped on the
    way in" from "the command arrived and the joint could not follow it".
    """

    return _arm_buffer(robot, "joint_pos_target", joint_ids=joint_ids)


def _initial_contact_blocked(
    *, task_kind: str, sample: Mapping[str, Any], collision_threshold_n: float
) -> bool:
    channels = [
        float(sample["task_robot_contact_peak_force_n"]),
        float(sample["robot_scene_contact_peak_force_n"]),
    ]
    if task_kind == "rigid_pick_place":
        channels.append(
            float(sample["robot_task_forbidden_collision_peak_force_n"])
        )
    channels.append(
        float(
            sample[
                "task_scene_contact_peak_force_n"
                if task_kind == "articulated_open_close"
                else "task_scene_collision_peak_force_n"
            ]
        )
    )
    return max(channels) >= float(collision_threshold_n)


def _gripper_convention_probe(*, env: Any, robot: Any, seed: int, torch: Any) -> dict[str, Any]:
    separations: dict[str, float] = {}
    pad_centers: dict[str, dict[str, list[float]]] = {}
    pad_midpoint_controlled_body: dict[str, list[float]] = {}
    pad_offsets: dict[str, list[float]] | None = None
    geometry: dict[str, Any] | None = None
    for command in (0.0, 1.0):
        env.reset(seed=seed)
        for _ in range(30):
            current = torch.as_tensor(robot.data.joint_pos)[0, :7]
            action = torch.tensor(
                [[*[float(value) for value in current], command]],
                device=env.unwrapped.device,
                dtype=torch.float32,
            )
            env.step(action)
        if pad_offsets is None:
            geometry, pad_offsets = _physical_pad_binding(
                robot=robot, torch=torch
            )
        centers = _pad_centers_from_finger_body_offsets(
            robot=robot, offsets_body_m=pad_offsets, torch=torch
        )
        pad_centers[str(command)] = centers
        separations[str(command)] = math.dist(
            centers["left"], centers["right"]
        )
        controlled_body_name = next(
            (
                name
                for name in ("panda_hand", "base_link")
                if name in list(robot.data.body_names)
            ),
            None,
        )
        if controlled_body_name is None:
            raise RuntimeError("native_task_gripper_controlled_body_missing")
        controlled_body_pose = _body_pose_world(
            robot, body_name=controlled_body_name, torch=torch
        )
        endpoint_midpoint = [
            (centers["left"][axis] + centers["right"][axis]) / 2.0
            for axis in range(3)
        ]
        pad_midpoint_controlled_body[str(command)] = rotate_vector_xyzw(
            quaternion_conjugate_xyzw(controlled_body_pose[3:7]),
            [
                endpoint_midpoint[axis] - controlled_body_pose[axis]
                for axis in range(3)
            ],
        )
    travel = abs(separations["0.0"] - separations["1.0"])
    midpoint = {
        command: [
            (centers["left"][axis] + centers["right"][axis]) / 2.0
            for axis in range(3)
        ]
        for command, centers in pad_centers.items()
    }
    midpoint_travel = math.dist(midpoint["0.0"], midpoint["1.0"])
    evidence = {
        "separation_measurement": "distal_collision_pad_center_distance",
        "pad_centers_world_m": pad_centers,
        "pad_center_offsets_in_finger_body_m": pad_offsets,
        "pad_midpoint_world_m": midpoint,
        "pad_midpoint_controlled_body_m": pad_midpoint_controlled_body,
        "pad_midpoint_travel_m": midpoint_travel,
        "selected_pad_colliders": (
            None if geometry is None else geometry["selected_pad_colliders"]
        ),
    }
    if travel < 1.0e-3:
        return {
            "status": "ambiguous",
            "finger_separation_m": separations,
            "separation_travel_m": travel,
            **evidence,
            "blockers": ["native_task_gripper_convention_travel_below_floor"],
        }
    closed = 1.0 if separations["1.0"] < separations["0.0"] else 0.0
    return {
        "status": "measured",
        "finger_separation_m": separations,
        "separation_travel_m": travel,
        **evidence,
        "closed_command": closed,
        "open_command": 1.0 - closed,
        "blockers": [],
    }


def _camera_snapshot(
    *,
    env: Any,
    camera_scene_names: Mapping[str, str],
    output_root: Path,
    snapshot_id: str,
    framing_expectations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    from blueprint_pipeline.native_task_camera_observability import (
        measure_native_task_camera_observability,
    )
    from blueprint_pipeline.native_task_frame_display_encoding import (
        display_encode_hdr,
    )

    rows = []
    diagnostics: dict[str, Any] = {
        "schema_version": "native_task_camera_snapshot_diagnostics.v1",
        "snapshot_id": snapshot_id,
        "cameras": [],
    }

    def _explicit_array(value: Any) -> Any:
        # Isaac Lab 3.x camera outputs are ProxyArray instances. Their public
        # contract requires explicit ``.torch`` access; relying on implicit
        # indexing can preserve the flattened Warp storage shape instead of
        # the logical image shape.
        tensor = getattr(value, "torch", value)
        return np.asarray(_jsonable(tensor))

    for role, scene_name in camera_scene_names.items():
        camera = env.unwrapped.scene[scene_name]
        outputs = camera.data.output
        rgb_raw = _explicit_array(outputs["rgb"])
        rgb_array = rgb_raw[0] if rgb_raw.ndim == 4 and rgb_raw.shape[0] == 1 else rgb_raw
        if rgb_array.shape[-1] == 4:
            rgb_array = rgb_array[..., :3]
        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)
        rgb_source = "isaac_ldr_annotator"
        hdr_raw = None
        hdr_array = None
        if "rgb_hdr" in outputs:
            hdr_raw = _explicit_array(outputs["rgb_hdr"])
            hdr_array = (
                hdr_raw[0]
                if hdr_raw.ndim == 4 and hdr_raw.shape[0] == 1
                else hdr_raw
            )
            hdr_array = np.asarray(hdr_array, dtype=np.float32)[..., :3]
            # Prefer our own display encoding of the linear buffer over the
            # annotator's per-channel clip: the retained frames from this
            # lane's attempt 001 carried a 17% over-white tail that clipped
            # to white blobs with chromatic fringes. The retained PNG is the
            # frame the camera gates measure and the one a human reviews.
            rgb_array = display_encode_hdr(hdr_array)
            rgb_source = "rgb_hdr_display_encoded"
        semantic_raw = _explicit_array(outputs["semantic_segmentation"])
        semantic = np.squeeze(semantic_raw)
        expected_hw = tuple(int(value) for value in rgb_array.shape[:2])
        if semantic.shape != expected_hw and semantic.size == expected_hw[0] * expected_hw[1]:
            semantic = semantic.reshape(expected_hw)
        info = _jsonable((camera.data.info or {}).get("semantic_segmentation") or {})
        labels = info.get("idToLabels") or {}
        thresholds = CAMERA_THRESHOLDS[role]
        # Persist before measuring: a refusal (an unreadable frame) aborts the
        # run from inside the measurement, and the frame that caused it is the
        # one thing the next engineer needs.
        frame_dir = output_root / "construction_frames" / role
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"{snapshot_id}.png"
        Image.fromarray(rgb_array, mode="RGB").save(
            frame_path, format="PNG", compress_level=9
        )
        hdr_record = None
        if hdr_array is not None:
            hdr_path = frame_dir / f"{snapshot_id}.rgb_hdr.npy"
            np.save(hdr_path, hdr_array, allow_pickle=False)
            hdr_record = {
                "path": str(hdr_path.relative_to(output_root)),
                "sha256": _sha256(hdr_path),
                "shape": list(hdr_array.shape),
                "minimum": float(hdr_array.min()),
                "maximum": float(hdr_array.max()),
                "mean": float(hdr_array.mean()),
                "std": float(hdr_array.std()),
                "finite_fraction": float(np.isfinite(hdr_array).mean()),
            }
        diagnostics["cameras"].append(
            {
                "role": role,
                "scene_name": scene_name,
                "rgb_source": rgb_source,
                "rgb_raw_shape": list(rgb_raw.shape),
                "rgb_image_shape": list(rgb_array.shape),
                "rgb_hdr_raw_shape": (
                    list(hdr_raw.shape) if hdr_raw is not None else None
                ),
                "rgb_hdr_image_shape": (
                    list(hdr_array.shape) if hdr_array is not None else None
                ),
                "semantic_raw_shape": list(semantic_raw.shape),
                "semantic_image_shape": list(semantic.shape),
                "semantic_dtype": str(semantic.dtype),
                "semantic_label_count": len(labels),
            }
        )
        (output_root / "native_task_camera_snapshot_diagnostics.v1.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # `rgb_array` is the exact array hashed into `rgb_png.sha256` below, so
        # the verdict is bound to the retained frame rather than to a second
        # read of the sensor.
        observability = measure_native_task_camera_observability(
            semantic_ids=semantic,
            id_to_labels=labels,
            rgb=rgb_array,
            site_appearance_render_expected=SITE_APPEARANCE_RENDER_EXPECTED,
            target_label="task_object",
            minimum_pixels=thresholds["minimum_pixels"],
            minimum_pixel_fraction=thresholds["minimum_pixel_fraction"],
            # Sealed at plan time for static world-frame cameras: scales the
            # configured minimums down to what this scene's geometry can
            # project, never up.  Robot-parented cameras have no row here and
            # keep the configured constants.
            framing_expectation=(
                (framing_expectations or {}).get(role) or None
            ),
        )
        rows.append(
            {
                "role": role,
                "scene_name": scene_name,
                "snapshot_id": snapshot_id,
                "rgb_png": {
                    "path": str(frame_path.relative_to(output_root)),
                    "sha256": _sha256(frame_path),
                },
                "rgb_source": rgb_source,
                "rgb_min": int(rgb_array.min()),
                "rgb_max": int(rgb_array.max()),
                "rgb_mean": float(rgb_array.mean()),
                "rgb_hdr": hdr_record,
                "intrinsic_matrix": _jsonable(camera.data.intrinsic_matrices)[0],
                "position_world_m": _jsonable(camera.data.pos_w)[0],
                "quaternion_world_opengl_xyzw": _jsonable(
                    camera.data.quat_w_opengl
                )[0],
                "observability": observability,
                "semantic_id_to_labels": labels,
                "raw_shapes": diagnostics["cameras"][-1],
                "native_sensor_timestamp": _jsonable(
                    getattr(camera.data, "frame", None)
                ),
            }
        )
    return {"snapshot_id": snapshot_id, "cameras": rows}


def expected_articulation_prim_paths(plan: Mapping[str, Any]) -> list[str]:
    """Concrete env-0 prim paths for every articulation the plan declares.

    The plan carries `{ENV_REGEX_NS}` templates; the tensor views want a real
    path. Deriving them here means the device report does not depend on a stage
    traversal, which is exactly what was unavailable when it was needed.
    """

    env_ns = "/World/envs/env_0"
    paths = [f"{env_ns}/Robot"]
    for entry in plan.get("objects", []) or []:
        if str(entry.get("object_type", "")).upper() != "ARTICULATION":
            continue
        prim_path = str(entry.get("prim_path", "")).replace("{ENV_REGEX_NS}", env_ns)
        if prim_path and prim_path not in paths:
            paths.append(prim_path)
    return paths


# Ordered stage accessors, most-certain first. `isaacsim.core.utils` is NOT
# present in this runtime -- r11 spent $0.056 collecting nothing because the
# diagnostic imported it and gave up. Its appearance in a shipped isaaclab
# source file proved only that upstream references it, not that this image
# ships it. Both entries below are used by isaaclab's own runtime code paths,
# and a single missing module can no longer blind the whole diagnostic.
_STAGE_ACCESSORS: tuple[tuple[str, str], ...] = (
    ("omni.usd", "omni.usd.get_context().get_stage()"),
    ("isaaclab.sim.utils.stage", "isaaclab.sim.utils.stage.get_current_stage()"),
    ("isaacsim.core.utils.stage", "isaacsim.core.utils.stage.get_current_stage()"),
)


def _current_stage() -> tuple[Any, dict[str, Any]]:
    """Return the live USD stage and a note of how it was reached."""

    attempts: dict[str, Any] = {}
    for module_name, _description in _STAGE_ACCESSORS:
        try:
            if module_name == "omni.usd":
                import omni.usd

                stage = omni.usd.get_context().get_stage()
            elif module_name == "isaaclab.sim.utils.stage":
                from isaaclab.sim.utils import stage as stage_utils

                stage = stage_utils.get_current_stage()
            else:
                import isaacsim.core.utils.stage as stage_utils

                stage = stage_utils.get_current_stage()
        except Exception as exc:  # noqa: BLE001
            attempts[module_name] = f"{type(exc).__name__}:{exc}"[:160]
            continue
        if stage is None:
            attempts[module_name] = "returned_none"
            continue
        attempts[module_name] = "ok"
        return stage, {"stage_source": module_name, "stage_attempts": attempts}
    return None, {"stage_source": None, "stage_attempts": attempts}


def physics_scene_device_evidence(
    articulation_prim_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report what PhysX was actually configured with.

    Articulation initialisation happens inside the environment build, so a
    failure there leaves no built object to interrogate. What does survive is
    the PhysX manager (the traceback runs through it, so it is certainly
    importable) and the USD stage carrying the `physxScene:*` attributes Isaac
    Lab authored. That is the only place the GPU-dynamics decision is
    observable: PhysX resolves an unsupported scene in C++ without logging it,
    and the first visible symptom is a CPU-backed tensor view.

    Nothing here may raise, and no single import may make the whole report
    empty -- each fact is collected independently.
    """

    evidence: dict[str, Any] = {}
    # The PhysX manager first: it needs no stage, and it holds the device the
    # whole decision hangs on.
    try:
        from isaaclab_physx.physics.physx_manager import PhysxManager

        evidence["physics_manager_device"] = str(PhysxManager.get_device())
        evidence["articulation_view_devices"] = _articulation_view_devices(
            PhysxManager, articulation_prim_paths
        )
    except Exception as exc:  # noqa: BLE001
        evidence["physics_manager_unavailable"] = f"{type(exc).__name__}:{exc}"[:200]
    try:
        from isaaclab.sim import SimulationContext

        instance = SimulationContext.instance()
        evidence["simulation_context_device"] = str(getattr(instance, "device", None))
        for setting in (
            "/physics/suppressReadback",
            "/physics/cudaDevice",
            "/physics/physxDispatcher",
        ):
            try:
                evidence.setdefault("settings", {})[setting] = str(
                    instance.get_setting(setting)
                )
            except Exception:  # noqa: BLE001
                evidence.setdefault("settings", {})[setting] = "unreadable"
    except Exception as exc:  # noqa: BLE001
        evidence["simulation_context_unavailable"] = f"{type(exc).__name__}:{exc}"[:200]

    stage, stage_note = _current_stage()
    evidence.update(stage_note)
    if stage is None:
        return evidence
    scenes: dict[str, Any] = {}
    try:
        for prim in stage.Traverse():
            # GetTypeName alone identifies the scene; an IsA(SchemaBase) guard
            # would only add a way for this to raise.
            if prim.GetTypeName() != "PhysicsScene":
                continue
            attributes: dict[str, Any] = {}
            for attribute in prim.GetAttributes():
                name = attribute.GetName()
                if not name.startswith(("physxScene:", "physics:")):
                    continue
                try:
                    attributes[name] = str(attribute.Get())
                except Exception:  # noqa: BLE001
                    attributes[name] = "unreadable"
            scenes[str(prim.GetPath())] = attributes
    except Exception as exc:  # noqa: BLE001
        evidence["traverse_failed"] = f"{type(exc).__name__}:{exc}"[:200]
    evidence["physics_scenes"] = scenes
    return evidence


def _articulation_view_devices(
    physx_manager: Any, prim_paths: Sequence[str] | None
) -> dict[str, Any]:
    """Report the backing device of every articulation view.

    This is the question a device-mismatch traceback cannot answer. If every
    articulation is CPU-backed the scene never got GPU dynamics at all, and the
    cause is scene-level. If exactly one is CPU-backed the cause is that asset.
    Those two answers point at completely different fixes.

    Paths come from the scene plan rather than a stage traversal, so this still
    reports when no stage accessor resolves.
    """

    view = getattr(physx_manager, "_view", None)
    if view is None:
        return {"unavailable": "simulation_view_is_none"}
    candidates = list(prim_paths or ())
    if not candidates:
        return {"unavailable": "no_articulation_prim_paths_supplied"}
    rows: dict[str, Any] = {}
    for path in candidates:
        try:
            articulation = view.create_articulation_view(path)
            velocities = articulation.get_dof_velocities()
            rows[path] = {
                "device": str(getattr(velocities, "device", None)),
                "backend_present": getattr(articulation, "_backend", None) is not None,
            }
        except Exception as exc:  # noqa: BLE001
            rows[path] = {"unavailable": f"{type(exc).__name__}:{exc}"[:200]}
    return rows


def _articulation_device_binding(
    built: Any, *, expected_device: str
) -> dict[str, Any]:
    """Report the device backing each articulation's joint-state arrays.

    Isaac Lab raises the mismatch from inside a Warp kernel launch, naming the
    kernel argument, so the message cannot say which asset is on the wrong
    device. This reads each articulation directly and says so.
    """

    rows: dict[str, Any] = {
        "expected_device": expected_device,
        "articulations": {},
        "non_articulation_assets_skipped": [],
    }
    try:
        scene = built.env.unwrapped.scene
    except Exception as exc:  # the scene may not be reachable on some failures
        rows["unavailable"] = f"{type(exc).__name__}:{exc}"[:200]
        return rows
    all_asset_names = set(built.scene_asset_names.values()) | {"robot"}
    plan = getattr(built, "plan", None)
    if isinstance(plan, Mapping):
        required_names = {"robot"} | {
            str(row.get("name") or row.get("semantic_role"))
            for row in plan.get("objects") or []
            if isinstance(row, Mapping) and row.get("object_type") == "ARTICULATION"
        }
        rows["non_articulation_assets_skipped"] = sorted(
            all_asset_names - required_names
        )
    else:
        required_names = all_asset_names
    rows["required_articulation_names"] = sorted(required_names)
    for name in sorted(required_names):
        entry: dict[str, Any] = {}
        try:
            asset = scene[name]
            data = asset.data
            for field in ("joint_pos", "joint_vel"):
                value = getattr(data, field, None)
                entry[field] = str(getattr(value, "device", None))
            entry["num_joints"] = len(getattr(asset, "joint_names", []) or [])
            entry["num_actuators"] = len(getattr(asset, "actuators", {}) or {})
            entry["data_device"] = str(getattr(data, "device", None))
            entry["on_expected_device"] = entry.get("joint_vel") == expected_device
        except Exception as exc:
            entry["unavailable"] = f"{type(exc).__name__}:{exc}"[:200]
        rows["articulations"][name] = entry
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    runtime = Path(__file__).resolve().parent
    output_root = Path(
        os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR")
        or runtime.parent / "runtime_output"
    ).resolve()
    output = output_root / RESULT_FILENAME
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [],
        "phase_reached": "start",
        "native_isaac_executed": False,
        "construction_gate_qualified": False,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
        "simulator_execution_is_not_physical_truth": True,
    }
    simulation_app = None
    try:
        _announce("packet_verification")
        manifest = _load_and_verify_manifest(runtime)
        result["manifest_input_digest"] = manifest["input_digest"]
        result["implementation_commit"] = manifest["implementation_commit"]
        packet = runtime / "native_task_packet"
        receipt_path = packet / "native_task_arena_packet_receipt.v1.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("receipt_digest") != manifest.get("packet_receipt_digest"):
            raise RuntimeError("native_task_construction_packet_binding_mismatch")
        plan_path = packet / "native_task_arena_scene_plan.v1.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("plan_digest") != manifest.get("arena_scene_plan_digest"):
            raise RuntimeError("native_task_construction_plan_binding_mismatch")
        result["packet_receipt_digest"] = receipt["receipt_digest"]
        result["scene_plan_digest"] = plan["plan_digest"]
        result["scenario"] = plan["scenario"]
        from blueprint_pipeline.native_task_construction_plan import (
            materialize_native_task_construction_phase_plan,
        )

        frozen_phase_path = verified_construction_phase_plan_path(runtime, manifest)
        frozen_phase_plan = json.loads(
            frozen_phase_path.read_text(encoding="utf-8")
        )
        recomputed_phase_plan = materialize_native_task_construction_phase_plan(plan)
        if frozen_phase_plan != recomputed_phase_plan:
            raise RuntimeError("native_task_construction_phase_plan_binding_mismatch")
        phase_plan = frozen_phase_plan
        result["construction_phase_plan"] = phase_plan
        bootstrap = feedback_bootstrap_result(
            runtime=runtime, manifest=manifest, packet=packet
        )
        if bootstrap is not None:
            result.update(bootstrap)
            _announce("feedback_bootstrap", "completed")
            return 1
        result["phase_reached"] = "packet_verified"
        _announce("packet_verification", "completed")

        _announce("simulation_app")
        from blueprint_pipeline.native_task_isaaclab_launch import (
            NATIVE_TASK_ARENA_DEVICE,
            launch_native_task_isaaclab,
        )
        from blueprint_pipeline.native_task_nurec_render_setup import (
            appearance_render_path_from_plan,
        )

        simulation_app, launch_receipt = launch_native_task_isaaclab(
            output_root / "native_task_runtime_source_provisioning.v1.json",
            device=NATIVE_TASK_ARENA_DEVICE,
            appearance_render_path=appearance_render_path_from_plan(plan),
        )
        result["isaaclab_launch"] = launch_receipt
        _announce("simulation_app", "completed")
        _announce("dependency_matrix")
        dependency_matrix = preflight_native_dependency_matrix(
            robot_id=str(plan["robot"]["robot_id"])
        )
        result["dependency_matrix"] = dependency_matrix
        if not dependency_matrix["all_required_available"]:
            result["blockers"].extend(dependency_matrix["blockers"])
            raise RuntimeError("native_task_construction_dependency_preflight_failed")
        result["phase_reached"] = "dependencies_qualified"
        _announce("dependency_matrix", "completed")

        import torch

        from blueprint_pipeline.native_franka_pose_servo import (
            NativeFrankaDifferentialIkServo,
            PINK_GLOBAL_REFERENCE_SEEDS,
        )
        from blueprint_pipeline.native_task_arena_readback import (
            NativeArticulatedTaskArenaReadback,
            NativeRigidTaskArenaReadback,
            read_native_task_arena_object_reset_state,
            read_native_task_arena_scenario_parameters,
        )
        from blueprint_pipeline.native_task_arena_device_readback import (
            read_native_task_arena_device_binding,
        )
        from blueprint_pipeline.native_task_arena_preconstruction import (
            prepare_native_task_arena_preconstruction,
        )
        from blueprint_pipeline.native_task_arena_runtime import (
            build_native_task_arena_environment,
        )

        _announce("preconstruction_device_binding")
        preconstruction = prepare_native_task_arena_preconstruction(
            expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        result["preconstruction_device_binding"] = preconstruction
        if not preconstruction["passed"]:
            result["blockers"].extend(preconstruction["blockers"])
            raise RuntimeError("native_task_arena_preconstruction_failed")
        _announce("preconstruction_device_binding", "completed")

        _announce("environment_build")
        try:
            built = build_native_task_arena_environment(
                plan,
                device=NATIVE_TASK_ARENA_DEVICE,
                bundle_root=packet,
                preconstruction_receipt=preconstruction,
            )
        except Exception as exc:
            # Articulation views are created here, inside sim.reset(). Attempts
            # r6-r10 all died in this call with a cuda/cpu mismatch naming a
            # Warp kernel argument rather than the asset or the setting that
            # demoted the scene. Record what PhysX was configured with before
            # re-raising, so the next failure is diagnosable from the receipt.
            result["environment_build_failure"] = {
                "error": f"{type(exc).__name__}:{exc}"[:400],
                "traceback": traceback.format_exc()[-4000:],
                "physics_scene_device_evidence": physics_scene_device_evidence(
                    expected_articulation_prim_paths(plan)
                ),
            }
            raise
        device_readback = read_native_task_arena_device_binding(
            built, expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        result["device_readback"] = device_readback
        if not device_readback["passed"]:
            result["blockers"].extend(device_readback["blockers"])
            raise RuntimeError("native_task_arena_device_binding_failed")
        env = built.env
        seed = int(plan["scenario"]["seed"])
        # env.reset() computes observations, which touches every articulation's
        # joint state. Attempts r6-r9 all died inside it with a cuda/cpu array
        # mismatch that names a kernel argument and not the asset, so record
        # which articulation is backed by which device before re-raising: one
        # run of certainty instead of another round of hypotheses.
        result["articulation_device_binding"] = _articulation_device_binding(
            built, expected_device=str(preconstruction["expected_device"])
        )
        try:
            env.reset(seed=seed)
        except Exception as exc:
            result["reset_failure"] = {
                "error": f"{type(exc).__name__}:{exc}"[:400],
                "traceback": traceback.format_exc()[-4000:],
                "articulation_device_binding": result["articulation_device_binding"],
            }
            raise
        _announce("site_appearance_renderer")
        result["site_appearance_renderer"] = _prepare_site_appearance_renderer(
            simulation_app=simulation_app,
            plan=plan,
            progress_callback=lambda row: _announce(
                f"nurec_warmup_round_{row['round']}", "completed"
            ),
        )
        if not result["site_appearance_renderer"]["passed"]:
            result["blockers"].extend(
                result["site_appearance_renderer"]["blockers"]
            )
            raise RuntimeError("native_task_arena_nurec_setup_failed")
        _announce("site_appearance_renderer", "completed")
        scene = env.unwrapped.scene
        robot = scene["robot"]
        task_object = scene[built.scene_asset_names["task_object"]]
        task_kind = str(plan["task_kind"])
        readback = (
            NativeArticulatedTaskArenaReadback(built)
            if task_kind == "articulated_open_close"
            else NativeRigidTaskArenaReadback(built)
        )
        result["native_isaac_executed"] = True
        result["phase_reached"] = "environment_built"
        _announce("environment_build", "completed")

        initial_sample = readback.read_task_sample()
        scenario_parameter_readback = read_native_task_arena_scenario_parameters(built)
        result["scenario_parameter_readback"] = scenario_parameter_readback
        if not scenario_parameter_readback["passed"]:
            result["blockers"].append(
                "native_task_scenario_parameter_readback_mismatch"
            )
        result["initial_readback"] = {
            "robot_root_pose_world": _jsonable(robot.data.root_pose_w)[0],
            "robot_joint_names": list(robot.joint_names),
            "robot_joint_positions_rad": _jsonable(robot.data.joint_pos)[0],
            "robot_body_names": list(robot.data.body_names),
            "task_joint_names": list(getattr(task_object, "joint_names", ()) or ()),
            "task_sample": initial_sample,
            "scene_asset_names": dict(built.scene_asset_names),
            "contact_sensor_names": dict(built.contact_sensor_names),
            "camera_scene_names": dict(built.camera_scene_names),
        }
        initial_native = initial_sample.get("native_readback") or initial_sample
        collision_threshold = float(
            (
                plan["articulation"]["state_thresholds"]
                if task_kind == "articulated_open_close"
                else phase_plan["thresholds"]
            )["collision_failure_minimum_force_n"]
        )
        if _initial_contact_blocked(
            task_kind=task_kind,
            sample=initial_native,
            collision_threshold_n=collision_threshold,
        ):
            result["blockers"].append("native_task_initial_penetration_or_contact")

        _announce("gripper_convention")
        gripper = _gripper_convention_probe(
            env=env, robot=robot, seed=seed, torch=torch
        )
        result["gripper_convention"] = gripper
        result["blockers"].extend(gripper["blockers"])
        if gripper["status"] != "measured":
            raise RuntimeError("native_task_construction_gripper_convention_unresolved")
        env.reset(seed=seed)
        result["phase_reached"] = "gripper_convention_measured"
        _announce("gripper_convention", "completed")

        servo = NativeFrankaDifferentialIkServo(
            env=env, robot=robot, gripper_convention=gripper
        )
        if task_kind == "articulated_open_close":
            # Subsequent task-path and reset samples must share the servo's
            # measured physical-pad grasp frame.  The pre-servo initial sample
            # remains explicitly labelled as the body-origin fallback.
            readback = NativeArticulatedTaskArenaReadback(
                built,
                grasp_frame_pose_callback=servo.current_grasp_frame_pose_world,
            )
        else:
            # The Robotiq inner-finger body origins nearly coincide throughout
            # travel.  Release evidence must use the physical fingertip pad
            # centers sealed by the native convention probe, not those origins.
            readback = NativeRigidTaskArenaReadback(
                built,
                gripper_pad_readback_callback=servo.current_gripper_pad_readback,
            )
        result["franka_pose_binding"] = servo.binding
        result["arm_actuator_readback"] = read_native_arm_actuator_readback(
            robot, joint_ids=servo.binding["arm_joint_ids"]
        )
        # Three sealed artifacts disagree about which frame the controlled body
        # is in, and none of them can settle it, because they are the things
        # that disagree.  The reset pose can: the direction between the two
        # finger bodies is the jaw axis and the direction from the body origin
        # to their midpoint is the direction the tool extends, both with no
        # convention assumed.  Both buffers are already read on the control
        # path and the direction thrown away, so this retains a measurement the
        # run was making anyway.  It is taken here, at the sealed reset, before
        # any phase has moved the arm.
        result["gripper_frame_axis_readback"] = (
            servo.current_gripper_frame_axis_readback()
        )
        reset_grasp_pose = servo.current_grasp_frame_pose_world()
        snapshots = []
        for _ in range(8):
            current = servo.read_arm_joint_positions()
            env.step(
                torch.tensor(
                    [[*current, float(gripper["open_command"])]],
                    device=env.unwrapped.device,
                    dtype=torch.float32,
                )
            )
        camera_framing_expectations = (
            plan.get("task_object_observability") or {}
        ).get("cameras") or {}
        snapshots.append(
            _camera_snapshot(
                env=env,
                camera_scene_names=built.camera_scene_names,
                output_root=output_root,
                snapshot_id="reset",
                framing_expectations=camera_framing_expectations,
            )
        )

        phase_results = []
        total_steps = 0
        # Reserve the settle window out of the episode budget: the qualifying
        # controls episode replays every qualified phase at its exact step
        # count and then appends this settle window inside the same
        # ``maximum_action_steps`` cap, so a construction that consumed the
        # full cap would qualify here and be refused there
        # (``native_rigid_control_action_budget_exceeded``) after the paid run.
        # Phase plans sealed by the current materializers carry the reserved
        # budget; legacy plans without it keep their historical cap.
        max_total_steps = min(
            int(
                phase_plan["execution_parameters"].get(
                    "maximum_construction_total_steps"
                )
                or plan["cadence"]["maximum_action_steps"]
            ),
            int(plan["cadence"]["maximum_action_steps"]),
        )
        execution_parameters = phase_plan["execution_parameters"]
        arrival_tolerance = float(execution_parameters["arrival_tolerance_m"])
        default_orientation_tolerance = execution_parameters.get(
            "arrival_orientation_tolerance_rad"
        )
        stable_samples = int(execution_parameters["stable_samples"])
        maximum_steps_per_phase = int(
            execution_parameters["maximum_steps_per_phase"]
        )
        servo_command_limits = _servo_command_limits(execution_parameters)
        result["servo_command_limits"] = dict(servo_command_limits)
        global_ik_solutions: dict[str, list[float]] = {}
        affordance = phase_plan.get("interaction_affordance") or {}
        front_entry_multistart = bool(
            float(affordance.get("contact_outward_standoff_m", 0.0)) > 0.0
            and affordance.get("grasp_swept_volume_receipt_digest")
        )
        if front_entry_multistart:
            _announce("pink_global_ik_preflight")
            reference_joints = servo.read_arm_joint_positions()
            preflight_phases = []
            for phase in phase_plan["phases"]:
                _announce(f"pink_global_ik_{phase['phase_id']}")
                target_orientation = _phase_target_orientation(
                    phase,
                    reset_body_orientation_xyzw=reset_grasp_pose[3:7],
                )
                solved = servo.solve_grasp_target_multistart(
                    target_position_world_m=phase["position_world_m"],
                    target_grasp_frame_quaternion_world_xyzw=(
                        target_orientation
                    ),
                    preferred_seeds=[
                        reference_joints,
                        *PINK_GLOBAL_REFERENCE_SEEDS,
                    ],
                    reference_joint_positions_rad=reference_joints,
                )
                selected = solved.get("selected")
                preflight_phases.append(
                    {
                        "phase_id": phase["phase_id"],
                        **solved,
                    }
                )
                if isinstance(selected, Mapping):
                    reference_joints = [
                        float(value)
                        for value in selected["joint_positions_rad"]
                    ]
                    global_ik_solutions[str(phase["phase_id"])] = list(
                        reference_joints
                    )
                    _announce(
                        f"pink_global_ik_{phase['phase_id']}",
                        "completed",
                    )
                else:
                    _announce(
                        f"pink_global_ik_{phase['phase_id']}",
                        "blocked",
                    )
            result["pink_global_ik_preflight"] = {
                "schema_version": "native_task_pink_global_ik_preflight.v1",
                "status": (
                    "all_phases_solved"
                    if len(global_ik_solutions) == len(phase_plan["phases"])
                    else "partial"
                ),
                "phase_count": len(phase_plan["phases"]),
                "solved_phase_count": len(global_ik_solutions),
                "phases": preflight_phases,
                "provider_mutation_performed": False,
                "physics_steps_performed": 0,
                "claim_boundary": (
                    "off_sim_multistart_pose_ik_only;native_execution_remains_"
                    "the_collision_contact_dynamics_and_arrival_authority"
                ),
            }
            _announce(
                "pink_global_ik_preflight",
                (
                    "completed"
                    if len(global_ik_solutions) == len(phase_plan["phases"])
                    else "blocked"
                ),
            )
        for phase in phase_plan["phases"]:
            _announce(f"phase_{phase['phase_id']}")
            servo.reset_command_state()
            stable = 0
            diagnostics = []
            start_position = servo.current_grasp_frame_position_world()
            start_body_pose = servo.current_body_pose_world()
            target_orientation = _phase_target_orientation(
                phase, reset_body_orientation_xyzw=reset_grasp_pose[3:7]
            )
            orientation_tolerance = phase.get(
                "arrival_orientation_tolerance_rad",
                default_orientation_tolerance,
            )
            gripper_command = float(
                gripper[
                    "closed_command"
                    if phase.get("gripper_state") == "closed"
                    else "open_command"
                ]
            )
            task_samples = []
            solver_sequence = validated_solver_joint_sequence(
                phase, arm_joint_names=servo.binding["arm_joint_names"]
            )
            solver_waypoint_index = 0
            while (
                total_steps < max_total_steps
                and len(diagnostics) < maximum_steps_per_phase
            ):
                global_target = solver_command_target(
                    solver_sequence,
                    waypoint_index=solver_waypoint_index,
                    fallback=global_ik_solutions.get(str(phase["phase_id"])),
                )
                if global_target is not None:
                    action, diagnostic = servo.action_for_joint_target(
                        target_joint_positions_rad=global_target,
                        gripper_command=gripper_command,
                        max_joint_delta_rad=servo_command_limits[
                            "max_joint_delta_rad"
                        ],
                        max_joint_setpoint_lead_rad=servo_command_limits[
                            "max_joint_setpoint_lead_rad"
                        ],
                        velocity_feedforward_scale=servo_command_limits[
                            "velocity_feedforward_scale"
                        ],
                    )
                    diagnostic["pink_global_ik_phase_solution_used"] = not bool(
                        solver_sequence
                    )
                    diagnostic["curobo_solver_path_waypoint_used"] = bool(
                        solver_sequence
                    )
                else:
                    action, diagnostic = servo.action_for_grasp_target(
                        target_position_world_m=phase["position_world_m"],
                        target_grasp_frame_quaternion_world_xyzw=(
                            target_orientation
                        ),
                        gripper_command=gripper_command,
                        max_joint_delta_rad=servo_command_limits[
                            "max_joint_delta_rad"
                        ],
                        max_joint_setpoint_lead_rad=servo_command_limits[
                            "max_joint_setpoint_lead_rad"
                        ],
                        velocity_feedforward_scale=servo_command_limits[
                            "velocity_feedforward_scale"
                        ],
                    )
                    diagnostic["pink_global_ik_phase_solution_used"] = False
                    diagnostic["curobo_solver_path_waypoint_used"] = False
                env.step(
                    torch.tensor(
                        [action],
                        device=env.unwrapped.device,
                        dtype=torch.float32,
                    )
                )
                total_steps += 1
                achieved_grasp_pose = servo.current_grasp_frame_pose_world()
                achieved = achieved_grasp_pose[:3]
                error = math.dist(achieved, phase["position_world_m"])
                arrival = _pose_arrival_readback(
                    position_world_m=achieved,
                    target_position_world_m=phase["position_world_m"],
                    orientation_world_xyzw=achieved_grasp_pose[3:7],
                    target_orientation_world_xyzw=target_orientation,
                    position_tolerance_m=arrival_tolerance,
                    orientation_tolerance_rad=(
                        None
                        if orientation_tolerance is None
                        else float(orientation_tolerance)
                    ),
                )
                orientation_error = arrival["orientation_error_rad"]
                solver_waypoint_index = advance_solver_waypoint(
                    solver_sequence,
                    waypoint_index=solver_waypoint_index,
                    measured_joint_positions_rad=servo.read_arm_joint_positions(),
                    tolerance_rad=servo_command_limits["max_joint_delta_rad"],
                    diagnostic=diagnostic,
                )
                stable = (
                    stable + 1
                    if solver_waypoint_index >= len(solver_sequence)
                    and arrival["reached"]
                    else 0
                )
                diagnostic["step_index"] = total_steps
                diagnostic["position_error_m"] = error
                diagnostic["orientation_error_rad"] = orientation_error
                # Torque at the step is what separates "commanded too little"
                # from "actuator could not deliver the load".  Retained per step
                # so a stalled phase carries its own attribution.
                diagnostic["applied_torque_n_m"] = _applied_arm_torque(
                    robot, joint_ids=servo.binding["arm_joint_ids"]
                )
                diagnostic["realized_joint_position_target_rad"] = (
                    _commanded_arm_joint_target(
                        robot, joint_ids=servo.binding["arm_joint_ids"]
                    )
                )
                diagnostics.append(diagnostic)
                if _retain_task_path_samples(
                    task_kind=task_kind, task_spec=plan["task_spec"]
                ):
                    task_samples.append(readback.read_task_sample())
                required_stable = (
                    int(phase_plan["settle_window_samples"])
                    if phase.get("phase_id") == "settle_observe"
                    else stable_samples
                )
                if stable >= required_stable:
                    break
            terminal_grasp_pose = servo.current_grasp_frame_pose_world()
            terminal = terminal_grasp_pose[:3]
            terminal_body_pose = servo.current_body_pose_world()
            terminal_error = math.dist(terminal, phase["position_world_m"])
            terminal_arrival = _terminal_grasp_frame_arrival_readback(
                grasp_pose_world=terminal_grasp_pose,
                body_pose_world=terminal_body_pose,
                target_position_world_m=phase["position_world_m"],
                target_orientation_world_xyzw=target_orientation,
                position_tolerance_m=arrival_tolerance,
                orientation_tolerance_rad=(
                    None
                    if orientation_tolerance is None
                    else float(orientation_tolerance)
                ),
            )
            terminal_orientation_error = terminal_arrival["orientation_error_rad"]
            sample = readback.read_task_sample()
            row = {
                "phase_id": phase["phase_id"],
                "target_position_world_m": phase["position_world_m"],
                "start_position_world_m": start_position,
                "start_body_orientation_world_xyzw": start_body_pose[3:7],
                "terminal_position_world_m": terminal,
                "terminal_position_error_m": terminal_error,
                "target_orientation_world_xyzw": target_orientation,
                "terminal_grasp_frame_orientation_world_xyzw": (
                    terminal_arrival[
                        "terminal_grasp_frame_orientation_world_xyzw"
                    ]
                ),
                "terminal_body_orientation_world_xyzw": terminal_arrival[
                    "terminal_body_orientation_world_xyzw"
                ],
                "terminal_orientation_error_rad": terminal_orientation_error,
                "arrival_orientation_tolerance_rad": orientation_tolerance,
                "arrival_tolerance_m": arrival_tolerance,
                "target_reached": (
                    terminal_arrival["reached"]
                    and stable >= required_stable
                ),
                "gripper_state": phase.get("gripper_state", "open"),
                "gripper_command": gripper_command,
                "gate_ids": list(phase.get("gate_ids") or []),
                "steps": len(diagnostics),
                **solver_path_result_fields(
                    solver_sequence, waypoint_index=solver_waypoint_index
                ),
                "diagnostics": diagnostics[:4] + diagnostics[-2:],
                "task_sample": sample,
                "task_samples": task_samples,
            }
            phase_results.append(row)
            snapshots.append(
                _camera_snapshot(
                    env=env,
                    camera_scene_names=built.camera_scene_names,
                    output_root=output_root,
                    snapshot_id=phase["phase_id"],
                    framing_expectations=camera_framing_expectations,
                )
            )
            _announce(
                f"phase_{phase['phase_id']}",
                "completed" if row["target_reached"] else "blocked",
            )
        result["phase_results"] = phase_results
        result["total_action_steps"] = total_steps
        # All attempted and budget-skipped phase rows are now sealed. If a
        # downstream evaluator refuses a missing path sample, attribute that
        # refusal to phase execution instead of the much earlier gripper probe.
        result["phase_reached"] = "phase_execution_complete"
        failed_phases = [
            row["phase_id"] for row in phase_results if not row["target_reached"]
        ]
        result["blockers"].extend(
            f"native_task_phase_ik_unreached:{phase_id}"
            for phase_id in failed_phases
        )

        camera_gates = {}
        for role in built.camera_scene_names:
            observations = [
                next(row for row in snapshot["cameras"] if row["role"] == role)
                for snapshot in snapshots
            ]
            # Rank a qualifying observation above a larger non-qualifying one.
            # `passed` is now the conjunction of semantic framing and rendered
            # radiance, so ranking on pixel_count alone could report a black
            # frame as `best_observability` beside a passing gate -- the same
            # shape of claim this gate exists to stop.
            best = max(
                observations,
                key=lambda row: (
                    bool(row["observability"]["passed"]),
                    row["observability"]["pixel_count"],
                ),
            )
            best_site = (
                best["observability"]["render_evidence"].get("site_region") or {}
            )
            camera_gates[role] = {
                "passed": any(
                    row["observability"]["passed"] for row in observations
                ),
                "best_snapshot_id": best["snapshot_id"],
                "best_observability": best["observability"],
                # Surfaced at the top of the gate, not buried in the frame
                # statistics: `passed` deliberately does not assert the captured
                # site rendered while this image ships no NuRec renderer, so the
                # receipt has to say so where a reader cannot miss it.
                "claim": best["observability"]["claim"],
                "site_appearance_claimed": best["observability"][
                    "site_appearance_claimed"
                ],
                "site_void_pixel_fraction": best_site.get("void_pixel_fraction"),
                "notices": sorted(
                    {
                        notice
                        for row in observations
                        for notice in row["observability"]["notices"]
                    }
                ),
            }
            if not camera_gates[role]["passed"]:
                result["blockers"].append(
                    f"native_task_camera_observability_failed:{role}"
                )
        result["camera_snapshots"] = snapshots
        result["camera_gates"] = camera_gates

        _announce("reset_replay")
        env.reset(seed=seed)
        reset_sample = readback.read_task_sample()
        reset_arm = servo.read_arm_joint_positions()
        requested_reset = _requested_arm_reset(
            plan=plan, servo_binding=servo.binding
        )
        reset_errors = [
            abs(actual - expected)
            for actual, expected in zip(reset_arm, requested_reset, strict=True)
        ]
        task_joint_resets = dict(
            phase_plan.get("joint_reset_positions")
            if phase_plan.get("schema_version")
            == "native_articulated_graph_construction_phase_plan.v1"
            else plan["task_spec"].get("joint_reset_positions_rad", {})
        )
        reset_joint_positions = reset_sample.get("joint_positions")
        if reset_joint_positions is None:
            reset_joint_positions = reset_sample.get("joint_positions_rad", {})
        task_reset_errors = {
            joint_id: abs(
                float(reset_joint_positions[joint_id])
                - float(expected)
            )
            for joint_id, expected in task_joint_resets.items()
        }
        object_reset_readback = read_native_task_arena_object_reset_state(built)
        if phase_plan.get("schema_version") == (
            "native_articulated_graph_construction_phase_plan.v1"
        ):
            task_joint_reset_passed = all(
                error
                <= float(phase_plan["joint_reset_tolerances"][joint_id])
                for joint_id, error in task_reset_errors.items()
            )
        else:
            task_joint_reset_passed = _task_joint_reset_passed(
                absolute_errors_rad=task_reset_errors,
                task_spec=plan["task_spec"],
            )
        reset_passed = (
            max(reset_errors, default=0.0) <= 1.0e-4
            and task_joint_reset_passed
            and object_reset_readback["passed"]
        )
        result["reset_replay"] = {
            "passed": reset_passed,
            "robot_joint_absolute_errors_rad": reset_errors,
            "task_joint_absolute_errors_rad": task_reset_errors,
            "object_reset_readback": object_reset_readback,
            "task_sample": reset_sample,
        }
        if not reset_passed:
            result["blockers"].append("native_task_reset_replay_mismatch")
        if not object_reset_readback["passed"]:
            result["blockers"].append(
                "native_task_object_reset_replay_mismatch"
            )
        task_gate_evaluation = _evaluate_task_construction_gates(
            phase_plan=phase_plan,
            phase_results=phase_results,
            reset_replay=result["reset_replay"],
        )
        if task_gate_evaluation is not None:
            gate_key, gate_result = task_gate_evaluation
            result[gate_key] = gate_result
            result["blockers"].extend(gate_result["blockers"])
        _announce("reset_replay", "completed" if reset_passed else "blocked")

        result["blockers"] = sorted(set(result["blockers"]))
        result["construction_gate_qualified"] = not result["blockers"]
        result["status"] = (
            "completed" if result["construction_gate_qualified"] else "blocked"
        )
        result["phase_reached"] = "construction_gate_complete"
        _announce(
            "construction_gate",
            "completed" if result["construction_gate_qualified"] else "blocked",
        )
    except BaseException as exc:  # noqa: BLE001 - one paid launch retains every failure
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "phase": result["phase_reached"],
            "traceback": traceback.format_exc(),
        }
        result["blockers"].append(
            f"native_task_construction_failed_at_{result['phase_reached']}:"
            f"{type(exc).__name__}:{exc}"
        )
        result["blockers"] = sorted(set(result["blockers"]))
        result["status"] = "blocked"
        _announce(str(result["phase_reached"]), "blocked")
    finally:
        result["completed_at_unix_ns"] = time.time_ns()
        _persist(output, result)
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:  # noqa: BLE001
                pass
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
