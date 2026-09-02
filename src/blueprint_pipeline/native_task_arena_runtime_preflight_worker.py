"""Paid GPU preflight for the exact native Arena runtime, without task motion."""

from __future__ import annotations

import json
import math
import os
import traceback
import zipfile
from pathlib import Path
from typing import Any

from blueprint_pipeline.native_task_nurec_render_setup import (
    setup_and_warm_native_nurec_renderer as _official_nurec_render_setup_and_warmup,
)


RESULT_FILENAME = "native_task_arena_runtime_preflight.v1.json"
RESULT_SCHEMA_VERSION = "native_task_arena_runtime_preflight.v1"


def _bind_measured_gripper_servo(
    *,
    env: Any,
    robot: Any,
    seed: int,
    torch: Any,
    gripper_probe: Any,
    servo_factory: Any,
) -> tuple[dict[str, Any], Any | None]:
    """Mirror the construction/policy gripper binding before live pad reads."""

    gripper = gripper_probe(env=env, robot=robot, seed=seed, torch=torch)
    if gripper.get("status") != "measured":
        return gripper, None
    env.reset(seed=seed)
    return gripper, servo_factory(
        env=env,
        robot=robot,
        gripper_convention=gripper,
    )


def _prepolicy_visual_gate_from_snapshot(
    *, snapshot: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    """Measure the exact retained reset PNGs before any candidate can load."""

    import numpy as np
    from PIL import Image

    from blueprint_pipeline.native_task_camera_observability import (
        measure_native_task_prepolicy_visual_frames,
    )

    frames: dict[str, Any] = {}
    root = output_root.resolve()
    for row in snapshot.get("cameras") or []:
        role = str(row.get("role") or "")
        relative = str((row.get("rgb_png") or {}).get("path") or "")
        frame_path = (root / relative).resolve()
        if not relative or not frame_path.is_relative_to(root):
            raise RuntimeError(
                f"native_task_arena_preflight_camera_path_invalid:{role}"
            )
        with Image.open(frame_path) as image:
            frames[role] = np.asarray(image.convert("RGB"))
    return measure_native_task_prepolicy_visual_frames(frames)


def _robot_reset_task_space_readback(
    *,
    plan: dict[str, Any],
    gripper_frame_axis_readback: dict[str, Any],
    object_reset_readback: dict[str, Any],
) -> dict[str, Any]:
    """Reject roots that look correct while the live fingers are unusably placed."""

    from blueprint_pipeline.rigid_frame_transforms import rotate_vector_xyzw

    try:
        base_pose = plan["robot"]["base_pose_world"]
        base = [float(value) for value in base_pose["position_world_m"]]
        base_quaternion = [
            float(value) for value in base_pose["orientation_xyzw"]
        ]
        midpoint = [
            float(value)
            for value in gripper_frame_axis_readback["measured"][
                "finger_midpoint_world_m"
            ]
        ]
        contact = [
            float(value)
            for value in object_reset_readback["task_link_frame_equivalence"][
                "observed_contact_position_world_m"
            ]
        ]
        if not all(len(row) == 3 for row in (base, midpoint, contact)):
            raise ValueError("vector length")
        forward = rotate_vector_xyzw(base_quaternion, [1.0, 0.0, 0.0])
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "schema_version": "native_task_arena_robot_reset_task_space.v1",
            "passed": False,
            "blockers": [
                "native_task_arena_robot_reset_task_space_readback_invalid"
            ],
            "error_type": type(exc).__name__,
        }

    base_to_midpoint = [midpoint[index] - base[index] for index in range(3)]
    base_to_contact = [contact[index] - base[index] for index in range(3)]
    finger_forward_m = sum(
        base_to_midpoint[index] * forward[index] for index in range(3)
    )
    contact_forward_m = sum(
        base_to_contact[index] * forward[index] for index in range(3)
    )
    finger_lateral_m = math.sqrt(
        sum(
            (
                base_to_midpoint[index] - finger_forward_m * forward[index]
            )
            ** 2
            for index in range(2)
        )
    )
    reach_m = math.dist(base, midpoint)
    clearance_m = midpoint[2] - base[2]
    contact_distance_m = math.dist(midpoint, contact)
    contact_to_base_horizontal = [
        base[0] - contact[0],
        base[1] - contact[1],
        0.0,
    ]
    contact_to_base_norm = math.sqrt(
        sum(value * value for value in contact_to_base_horizontal)
    )
    if not math.isfinite(contact_to_base_norm) or contact_to_base_norm <= 1.0e-9:
        return {
            "schema_version": "native_task_arena_robot_reset_task_space.v1",
            "passed": False,
            "blockers": [
                "native_task_arena_robot_reset_task_space_readback_invalid"
            ],
            "error_type": "DegenerateContactBearing",
        }
    approach = [
        contact[index]
        + 0.12 * contact_to_base_horizontal[index] / contact_to_base_norm
        for index in range(3)
    ]
    approach_distance_m = math.dist(midpoint, approach)
    checks = {
        "finger_midpoint_above_floor": clearance_m >= 0.20,
        "finger_midpoint_in_front_of_base": finger_forward_m >= 0.15,
        "finger_midpoint_within_franka_reach": 0.20 <= reach_m <= 0.85,
        "task_contact_in_front_of_base": contact_forward_m >= 0.40,
        "finger_midpoint_laterally_task_relevant": finger_lateral_m <= 0.45,
        "finger_midpoint_near_approach_region": approach_distance_m <= 0.75,
    }
    blockers = [
        f"native_task_arena_robot_reset_{name}_failed"
        for name, passed in checks.items()
        if not passed
    ]
    return {
        "schema_version": "native_task_arena_robot_reset_task_space.v1",
        "robot_forward_axis_source": "robot_base_local_positive_x",
        "base_position_world_m": base,
        "base_orientation_world_xyzw": base_quaternion,
        "robot_forward_unit_world": forward,
        "finger_midpoint_world_m": midpoint,
        "observed_contact_position_world_m": contact,
        "approach_standoff_position_world_m": approach,
        "approach_standoff_m": 0.12,
        "finger_height_above_base_m": clearance_m,
        "finger_forward_projection_m": finger_forward_m,
        "task_contact_forward_projection_m": contact_forward_m,
        "finger_lateral_offset_m": finger_lateral_m,
        "base_to_finger_distance_m": reach_m,
        "finger_to_contact_distance_m": contact_distance_m,
        "finger_to_approach_standoff_distance_m": approach_distance_m,
        "checks": checks,
        "blockers": blockers,
        "passed": not blockers,
    }


def _gripper_pad_geometry_axis_readback(
    *, stage: Any, body_axis_readback: dict[str, Any]
) -> dict[str, Any]:
    """Measure the TCP from the distal collision pad on each inner finger."""

    from blueprint_pipeline.native_franka_grasp_geometry import (
        NativeFrankaGraspGeometryError,
        measure_live_robotiq_grasp_geometry,
    )
    from blueprint_pipeline.native_franka_pose_servo import (
        NativeFrankaPoseServoError,
        gripper_frame_axis_readback,
    )

    measured = body_axis_readback["measured"]
    try:
        geometry = measure_live_robotiq_grasp_geometry(
            stage=stage,
            controlled_body_position_world_m=measured[
                "controlled_body_position_world_m"
            ],
            controlled_body_quaternion_world_xyzw=measured[
                "controlled_body_quaternion_world_xyzw"
            ],
        )
        axis = gripper_frame_axis_readback(
            controlled_body_name=str(body_axis_readback["controlled_body_name"]),
            body_position_world_m=measured["controlled_body_position_world_m"],
            body_quaternion_world_xyzw=measured[
                "controlled_body_quaternion_world_xyzw"
            ],
            finger_positions_world_m={
                "left_inner_finger": geometry["pad_centers_world_m"]["left"],
                "right_inner_finger": geometry["pad_centers_world_m"]["right"],
            },
        )
    except (
        KeyError,
        TypeError,
        NativeFrankaGraspGeometryError,
        NativeFrankaPoseServoError,
    ) as exc:
        return {
            "schema_version": "native_task_arena_gripper_pad_geometry.v1",
            "blockers": [
                "native_task_arena_gripper_pad_geometry_axis_readback_invalid"
            ],
            "error_type": type(exc).__name__,
            "passed": False,
        }
    separation = float(axis["measured"]["finger_separation_m"])
    tool_offset = float(axis["measured"]["body_origin_to_finger_midpoint_m"])
    orthogonality = abs(float(axis["derived"]["jaw_approach_orthogonality_dot"]))
    checks = {
        "pad_separation_physical": 0.01 <= separation <= 0.12,
        "body_to_tcp_offset_physical": 0.05 <= tool_offset <= 0.30,
        "jaw_and_approach_orthogonal": orthogonality <= 0.25,
    }
    blockers = [
        f"native_task_arena_gripper_{name}_failed"
        for name, passed in checks.items()
        if not passed
    ]
    return {
        "schema_version": "native_task_arena_gripper_pad_geometry.v1",
        "measurement_authority": geometry["measurement_authority"],
        "matched_collision_candidates": geometry[
            "matched_collision_candidates"
        ],
        "selected_pad_colliders": geometry["selected_pad_colliders"],
        "pad_bounds_world_m": geometry["pad_bounds_world_m"],
        "pad_centers_world_m": geometry["pad_centers_world_m"],
        "axis_readback": axis,
        "checks": checks,
        "blockers": blockers,
        "passed": not blockers,
    }


def _particlefield_stage_readback(stage: Any) -> dict[str, Any]:
    """Measure whether one conforming ParticleField actually composed live."""

    from pxr import UsdGeom

    rows = []
    blockers: list[str] = []
    for prim in stage.Traverse():
        if str(prim.GetTypeName()) != "ParticleField3DGaussianSplat":
            continue

        def _count(name: str) -> int:
            value = prim.GetAttribute(name).Get()
            return len(value) if value is not None else 0

        degree = prim.GetAttribute(
            "radiance:sphericalHarmonicsDegree"
        ).Get()
        sh_attr = prim.GetAttribute(
            "radiance:sphericalHarmonicsCoefficients"
        )
        sh_primvar = UsdGeom.Primvar(sh_attr)
        position_count = _count("positions")
        expected_element_size = (
            (int(degree) + 1) ** 2
            if isinstance(degree, int)
            and not isinstance(degree, bool)
            and 0 <= int(degree) <= 3
            else None
        )
        raw_extent = prim.GetAttribute("extent").Get()
        extent = (
            [[float(component) for component in vector] for vector in raw_extent]
            if raw_extent is not None
            else None
        )
        material_targets = [
            str(path)
            for path in prim.GetRelationship("material:binding").GetTargets()
        ]
        material = (
            stage.GetPrimAtPath(material_targets[0])
            if len(material_targets) == 1
            else None
        )
        shader = (
            stage.GetPrimAtPath(f"{material_targets[0]}/Shader")
            if material
            else None
        )
        source_asset = (
            shader.GetAttribute("info:mdl:sourceAsset").Get()
            if shader
            else None
        )
        row = {
            "prim_path": str(prim.GetPath()),
            "active": bool(prim.IsActive()),
            "defined": bool(prim.IsDefined()),
            "loaded": bool(prim.IsLoaded()),
            "position_count": position_count,
            "scale_count": _count("scales"),
            "orientation_count": _count("orientations"),
            "opacity_count": _count("opacities"),
            "sh_degree": degree,
            "sh_coefficient_count": _count(
                "radiance:sphericalHarmonicsCoefficients"
            ),
            "sh_element_size": sh_primvar.GetElementSize(),
            "sh_interpolation": str(sh_primvar.GetInterpolation()),
            "expected_sh_element_size": expected_element_size,
            "extent": extent,
            "material_binding_targets": material_targets,
            "material_prim_valid": bool(material),
            "material_shader_valid": bool(shader),
            "material_shader_source_asset": (
                source_asset.path if source_asset is not None else None
            ),
            "material_shader_sub_identifier": (
                shader.GetAttribute(
                    "info:mdl:sourceAsset:subIdentifier"
                ).Get()
                if shader
                else None
            ),
        }
        upstream_native_material = (
            not material_targets
            and not row["material_prim_valid"]
            and not row["material_shader_valid"]
        )
        legacy_emissive_material = (
            len(material_targets) == 1
            and row["material_prim_valid"]
            and row["material_shader_valid"]
            and row["material_shader_source_asset"] == "ParticleFieldEmissive.mdl"
            and row["material_shader_sub_identifier"] == "ParticleFieldEmissive"
        )
        row["material_contract"] = (
            "upstream_native_unbound"
            if upstream_native_material
            else "legacy_particlefield_emissive"
            if legacy_emissive_material
            else "invalid"
        )
        row["passed"] = bool(
            row["active"]
            and row["defined"]
            and row["loaded"]
            and "/scene_appearance/" in row["prim_path"]
            and position_count > 0
            and row["scale_count"] == position_count
            and row["orientation_count"] == position_count
            and row["opacity_count"] == position_count
            and expected_element_size is not None
            and row["sh_element_size"] == expected_element_size
            and row["sh_interpolation"] == "vertex"
            and row["sh_coefficient_count"]
            == position_count * expected_element_size
            and isinstance(row["extent"], list)
            and len(row["extent"]) == 2
            and (upstream_native_material or legacy_emissive_material)
        )
        rows.append(row)
    if len(rows) != 1:
        blockers.append("native_task_arena_particlefield_prim_not_exact")
    if any(not row["passed"] for row in rows):
        blockers.append("native_task_arena_particlefield_composition_invalid")
    return {
        "schema_version": "native_task_arena_particlefield_stage_readback.v1",
        "particlefield_prim_count": len(rows),
        "particlefields": rows,
        "blockers": blockers,
        "passed": not blockers,
    }


def _plain_nurec_volume_contract(packet: Path, plan: dict[str, Any]) -> dict[str, Any]:
    appearance_rows = [
        row
        for row in plan.get("objects") or []
        if isinstance(row, dict) and row.get("semantic_role") == "scene_appearance"
    ]
    blockers: list[str] = []
    relative = ""
    if len(appearance_rows) == 1:
        relative = str(appearance_rows[0].get("usd_path") or "")
    parts = Path(relative).parts
    if (
        len(appearance_rows) != 1
        or not relative
        or Path(relative).is_absolute()
        or ".." in parts
    ):
        blockers.append("native_task_arena_nurec_appearance_path_invalid")
    asset = (packet / relative).resolve() if relative else packet
    if asset != packet and packet not in asset.parents:
        blockers.append("native_task_arena_nurec_appearance_path_escape")
    alignment = plan.get("appearance_frame_alignment") or {}
    particlefield = (
        alignment.get("status") == "aligned"
        and alignment.get("representation")
        == "particlefield_3d_gaussian_splat"
        and alignment.get("measurement_authority")
        == "particlefield_position_quantiles"
        and relative.lower().endswith((".usd", ".usda", ".usdc"))
    )
    if particlefield and not blockers:
        return {
            "render_path": "particlefield_3d_gaussian_splat",
            "asset_relative_path": relative,
            "nurec_volume_signals_present": False,
            "particlefield_alignment_receipt_present": True,
            "spg_source_asset_authored": False,
            "spg_graph_execution_required": False,
            "renderer_extension_activation_expected": True,
            "passed": True,
            "blockers": [],
        }
    text = b""
    if not blockers:
        try:
            with zipfile.ZipFile(asset) as archive:
                text = b"\n".join(
                    archive.read(info)
                    for info in archive.infolist()
                    if info.filename.lower().endswith((".usd", ".usda"))
                    and info.file_size <= 2_000_000
                )
        except (OSError, zipfile.BadZipFile, KeyError):
            blockers.append("native_task_arena_nurec_appearance_unreadable")
    volume = (
        b"omni:nurec:isNuRecVolume" in text
        and b"OmniNuRecFieldAsset" in text
    )
    spg = b"info:spg:sourceAsset" in text
    if not volume:
        blockers.append("native_task_arena_plain_nurec_volume_signals_missing")
    if spg:
        blockers.append("native_task_arena_spg_asset_requires_separate_launch_path")
    return {
        "render_path": "plain_nurec_volume" if volume and not spg else "unsupported",
        "asset_relative_path": relative,
        "nurec_volume_signals_present": volume,
        "spg_source_asset_authored": spg,
        "spg_graph_execution_required": spg,
        "renderer_extension_activation_expected": True,
        "passed": not blockers,
        "blockers": sorted(set(blockers)),
    }


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _announce(phase: str, status: str = "started") -> None:
    print(
        f"BLUEPRINT_WAM_RUNTIME_PHASE:native_task_arena_preflight:{phase}:{status}",
        flush=True,
    )


def main() -> int:
    runtime = Path(__file__).resolve().parent
    output_root = Path(
        os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR", runtime / "runtime_output")
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "preflight_only": True,
        "task_motion_executed": False,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
        "phase_reached": "not_started",
        "blockers": [],
    }
    simulation_app = None
    env = None
    try:
        from blueprint_pipeline.native_task_arena_construction_worker import (
            _articulation_device_binding,
            _camera_snapshot,
            _gripper_convention_probe,
            _load_and_verify_manifest,
            preflight_native_dependency_matrix,
        )

        _announce("packet_verification")
        manifest = _load_and_verify_manifest(
            runtime, expected_execution_mode="runtime_preflight"
        )
        packet = runtime / "native_task_packet"
        plan = json.loads(
            (packet / "native_task_arena_scene_plan.v1.json").read_text(
                encoding="utf-8"
            )
        )
        result["implementation_commit"] = manifest["implementation_commit"]
        result["manifest_input_digest"] = manifest["input_digest"]
        result["packet_receipt_digest"] = manifest["packet_receipt_digest"]
        result["scene_plan_digest"] = plan["plan_digest"]
        result["appearance_render_path"] = _plain_nurec_volume_contract(
            packet, plan
        )
        if not result["appearance_render_path"]["passed"]:
            result["blockers"].extend(
                result["appearance_render_path"]["blockers"]
            )
            raise RuntimeError("native_task_arena_nurec_render_path_failed")
        result["phase_reached"] = "packet_verified"
        _announce("packet_verification", "completed")

        _announce("simulation_app")
        from blueprint_pipeline.native_task_isaaclab_launch import (
            NATIVE_TASK_ARENA_DEVICE,
            launch_native_task_isaaclab,
        )

        simulation_app, launch = launch_native_task_isaaclab(
            output_root / "native_task_runtime_source_provisioning.v1.json",
            device=NATIVE_TASK_ARENA_DEVICE,
            appearance_render_path=result["appearance_render_path"]["render_path"],
        )
        result["isaaclab_launch"] = launch
        result["phase_reached"] = "simulation_app_qualified"
        _announce("simulation_app", "completed")

        _announce("dependency_matrix")
        dependencies = preflight_native_dependency_matrix(
            robot_id=str(plan["robot"]["robot_id"])
        )
        result["dependency_matrix"] = dependencies
        if not dependencies["all_required_available"]:
            result["blockers"].extend(dependencies["blockers"])
            raise RuntimeError("native_task_arena_preflight_dependencies_failed")
        _announce("dependency_matrix", "completed")

        from blueprint_pipeline.native_task_arena_preconstruction import (
            prepare_native_task_arena_preconstruction,
        )
        from blueprint_pipeline.native_task_arena_runtime import (
            build_native_task_arena_environment,
        )
        from blueprint_pipeline.native_task_arena_device_readback import (
            read_native_task_arena_device_binding,
        )
        from blueprint_pipeline.native_task_arena_readback import (
            read_native_task_arena_object_reset_state,
        )

        _announce("environment_build")
        preconstruction = prepare_native_task_arena_preconstruction(
            expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        result["preconstruction_device_binding"] = preconstruction
        if not preconstruction["passed"]:
            result["blockers"].extend(preconstruction["blockers"])
            raise RuntimeError("native_task_arena_preflight_preconstruction_failed")
        built = build_native_task_arena_environment(
            plan,
            device=NATIVE_TASK_ARENA_DEVICE,
            bundle_root=packet,
            preconstruction_receipt=preconstruction,
        )
        device = read_native_task_arena_device_binding(
            built, expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        result["device_readback"] = device
        if not device["passed"]:
            result["blockers"].extend(device["blockers"])
            raise RuntimeError("native_task_arena_preflight_device_binding_failed")
        env = built.env
        seed = int(plan["scenario"]["seed"])
        env.reset(seed=seed)
        result["object_reset_readback"] = (
            read_native_task_arena_object_reset_state(built)
        )
        if not result["object_reset_readback"]["passed"]:
            result["blockers"].append(
                "native_task_arena_preflight_object_reset_not_equivalent"
            )
            raise RuntimeError(
                "native_task_arena_preflight_object_reset_failed"
            )
        render_path = result["appearance_render_path"]["render_path"]
        if render_path == "particlefield_3d_gaussian_splat":
            import omni.usd

            result["particlefield_stage_readback"] = (
                _particlefield_stage_readback(
                    omni.usd.get_context().get_stage()
                )
            )
            if not result["particlefield_stage_readback"]["passed"]:
                result["blockers"].extend(
                    result["particlefield_stage_readback"]["blockers"]
                )
                raise RuntimeError(
                    "native_task_arena_preflight_particlefield_composition_failed"
                )
        if render_path in {
            "particlefield_3d_gaussian_splat",
            "plain_nurec_volume",
        }:
            import omni.usd

            result["official_nurec_render_setup"] = (
                _official_nurec_render_setup_and_warmup(
                    simulation_app,
                    omni.usd.get_context().get_stage(),
                    progress_callback=lambda row: _announce(
                        f"nurec_warmup_round_{row['round']}", "completed"
                    ),
                    require_display_referred_particlefield=(
                        render_path == "particlefield_3d_gaussian_splat"
                    ),
                )
            )
            if not result["official_nurec_render_setup"]["passed"]:
                result["blockers"].extend(
                    result["official_nurec_render_setup"]["blockers"]
                )
                raise RuntimeError(
                    "native_task_arena_preflight_nurec_setup_failed"
                )
        result["articulation_device_binding"] = _articulation_device_binding(
            built, expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        articulation_rows = result["articulation_device_binding"].get(
            "articulations"
        ) or {}
        if not articulation_rows or not all(
            row.get("on_expected_device") is True and "unavailable" not in row
            for row in articulation_rows.values()
        ):
            result["blockers"].append(
                "native_task_arena_preflight_articulation_device_mismatch"
            )
            raise RuntimeError("native_task_arena_preflight_articulation_device_failed")
        result["phase_reached"] = "environment_built"
        _announce("environment_build", "completed")

        import torch

        from blueprint_pipeline.native_franka_pose_servo import (
            NativeFrankaDifferentialIkServo,
        )

        robot = env.unwrapped.scene["robot"]
        _announce("gripper_convention")
        gripper, servo = _bind_measured_gripper_servo(
            env=env,
            robot=robot,
            seed=seed,
            torch=torch,
            gripper_probe=_gripper_convention_probe,
            servo_factory=NativeFrankaDifferentialIkServo,
        )
        result["gripper_convention"] = gripper
        result["blockers"].extend(gripper.get("blockers") or [])
        if servo is None:
            raise RuntimeError(
                "native_task_arena_preflight_gripper_convention_unresolved"
            )
        result["phase_reached"] = "gripper_convention_measured"
        _announce("gripper_convention", "completed")
        result["gripper_body_origin_axis_readback"] = (
            servo.current_gripper_frame_axis_readback()
        )
        import omni.usd

        result["gripper_pad_geometry_readback"] = (
            _gripper_pad_geometry_axis_readback(
                stage=omni.usd.get_context().get_stage(),
                body_axis_readback=result["gripper_body_origin_axis_readback"],
            )
        )
        if not result["gripper_pad_geometry_readback"]["passed"]:
            result["blockers"].extend(
                result["gripper_pad_geometry_readback"]["blockers"]
            )
            raise RuntimeError(
                "native_task_arena_preflight_gripper_pad_geometry_failed"
            )
        result["gripper_frame_axis_readback"] = result[
            "gripper_pad_geometry_readback"
        ]["axis_readback"]
        result["robot_reset_task_space_readback"] = (
            _robot_reset_task_space_readback(
                plan=plan,
                gripper_frame_axis_readback=result["gripper_frame_axis_readback"],
                object_reset_readback=result["object_reset_readback"],
            )
        )
        if not result["robot_reset_task_space_readback"]["passed"]:
            result["blockers"].extend(
                result["robot_reset_task_space_readback"]["blockers"]
            )
            raise RuntimeError(
                "native_task_arena_preflight_robot_reset_task_space_failed"
            )
        for _ in range(8):
            current = servo.read_arm_joint_positions()
            env.step(
                torch.tensor(
                    [[*current, 0.0]],
                    device=env.unwrapped.device,
                    dtype=torch.float32,
                )
            )
        result["camera_snapshot"] = _camera_snapshot(
            env=env,
            camera_scene_names=built.camera_scene_names,
            output_root=output_root,
            snapshot_id="runtime_preflight",
        )
        camera_rows = result["camera_snapshot"]["cameras"]
        if not camera_rows or not all(
            row["observability"]["passed"] for row in camera_rows
        ):
            result["blockers"].append(
                "native_task_arena_preflight_camera_observability_failed"
            )
            raise RuntimeError("native_task_arena_preflight_camera_failed")
        result["prepolicy_visual_gate"] = _prepolicy_visual_gate_from_snapshot(
            snapshot=result["camera_snapshot"],
            output_root=output_root,
        )
        if not result["prepolicy_visual_gate"]["passed"]:
            result["blockers"].extend(
                result["prepolicy_visual_gate"]["blockers"]
            )
            raise RuntimeError(
                "native_task_arena_preflight_prepolicy_visual_gate_failed"
            )
        result["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "environment_device": str(env.unwrapped.device),
        }
        result["phase_reached"] = "runtime_preflight_completed"
        result["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - retained as typed preflight evidence
        launch_errors = getattr(exc, "errors", None)
        launch_diagnostics = getattr(exc, "diagnostics", None)
        if launch_errors is not None:
            result["isaaclab_launch_error"] = {
                "errors": list(launch_errors),
                "diagnostics": dict(launch_diagnostics or {}),
            }
        result["blockers"].append(
            f"native_task_arena_runtime_preflight_failed_at_{result['phase_reached']}:"
            f"{type(exc).__name__}:{exc}"[:500]
        )
        result["traceback"] = traceback.format_exc()[-6000:]
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:  # noqa: BLE001
                result["blockers"].append(
                    f"native_task_arena_preflight_env_close_failed:{type(exc).__name__}"
                )
                result["status"] = "blocked"
        result["blockers"] = sorted(set(result["blockers"]))
        (output_root / RESULT_FILENAME).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "BLUEPRINT_NATIVE_TASK_ARENA_RUNTIME_PREFLIGHT_"
            + ("OK" if result["status"] == "completed" else "BLOCKED"),
            flush=True,
        )
        # Persist before closing. Isaac Sim's close path may terminate the
        # process instead of returning, which previously erased a successful
        # preflight receipt while the shell still observed exit code 0.
        if simulation_app is not None:
            try:
                simulation_app.close()
            except Exception:  # noqa: BLE001 - result is already durable
                pass
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
