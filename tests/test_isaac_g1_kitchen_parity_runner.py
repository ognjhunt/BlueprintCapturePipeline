"""Hermetic tests for the GPU runner's non-Isaac helpers (importing the runner must NOT pull
in isaacsim — the Isaac-API calls are lazily imported inside the GPU-only functions)."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest

from blueprint_pipeline.scene_placement import SceneObject


pytestmark = [pytest.mark.slow, pytest.mark.integration]

_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_isaac_g1_kitchen_parity_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("parity_runner", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # would raise if it imported isaacsim at module load
    return mod


M = _load()


def test_runner_imports_without_isaacsim() -> None:
    assert hasattr(M, "run_scenarios") and hasattr(M, "parse_scenarios")


def test_manipulation_reach_fraction_progresses_for_robot_pov() -> None:
    values = [
        M.manipulation_reach_fraction_for_frame(
            step / 7.0,
            manipulation_cam=True,
            frame_count=8,
        )
        for step in range(8)
    ]

    assert values[0] == pytest.approx(M.MANIPULATION_POV_REACH_RAMP_START_FRACTION)
    assert values[-1] == pytest.approx(1.0)
    assert values == sorted(values)
    assert len({round(value, 6) for value in values}) == 8
    assert M.manipulation_reach_fraction_for_frame(
        0.0,
        manipulation_cam=True,
        frame_count=1,
    ) == pytest.approx(1.0)


def test_manipulation_runner_does_not_hold_robot_pov_reach_constant() -> None:
    source = _RUNNER.read_text()

    assert "1.0 if manipulation_cam else alpha" not in source
    assert "manipulation_reach_fraction" in source


def test_resolve_existing_kitchen_usd_accepts_root_zip_layout(tmp_path: Path) -> None:
    kitchen_root = tmp_path / "bundle" / "kitchen"
    kitchen_root.mkdir(parents=True)
    root_usd = kitchen_root / "KitchenRoom.usd"
    root_usd.write_text("#usda root", encoding="utf-8")
    requested = kitchen_root / "Collected_KitchenRoom" / "KitchenRoom.usd"

    resolved, detail = M._resolve_existing_kitchen_usd(str(requested))

    assert resolved == str(root_usd)
    assert detail["requested_exists"] is False
    assert detail["resolved_from_existing_candidate"] is True
    assert str(root_usd) in detail["existing_candidate_paths"]


def test_resolve_existing_kitchen_usd_keeps_collected_layout_when_present(tmp_path: Path) -> None:
    kitchen_root = tmp_path / "bundle" / "kitchen"
    collected_usd = kitchen_root / "Collected_KitchenRoom" / "KitchenRoom.usd"
    collected_usd.parent.mkdir(parents=True)
    collected_usd.write_text("#usda collected", encoding="utf-8")

    resolved, detail = M._resolve_existing_kitchen_usd(str(collected_usd))

    assert resolved == str(collected_usd)
    assert detail["requested_exists"] is True
    assert detail["resolved_from_existing_candidate"] is False


def test_runner_writes_result_before_simulation_app_close() -> None:
    source = _RUNNER.read_text()

    preclose_marker = '"isaac_g1_kitchen_parity_result.json").write_text'
    close_marker = "sim.close()"
    assert preclose_marker in source
    assert source.index(preclose_marker) < source.index(close_marker)


def test_g1_visual_asset_candidates_try_exact_then_visual_siblings() -> None:
    candidates = M._g1_visual_asset_candidates("Isaac/Robots/Unitree/G1/g1.usda")

    assert candidates == [
        "Isaac/Robots/Unitree/G1/g1.usd",
        "Unitree/G1/g1.usd",
        "Unitree/G1/g1.usda",
        "Isaac/Robots/Unitree/G1/g1.usda",
    ]
    assert len(candidates) == len(set(candidates))

    absolute = M._g1_visual_asset_candidates("/Isaac/Robots/Unitree/G1/g1.usda")
    assert absolute == [
        "/Isaac/Robots/Unitree/G1/g1.usd",
        "/Unitree/G1/g1.usd",
        "/Unitree/G1/g1.usda",
        "/Isaac/Robots/Unitree/G1/g1.usda",
    ]

    exact_visual = M._g1_visual_asset_candidates("Unitree/G1/g1.usd")
    assert exact_visual == ["Unitree/G1/g1.usd"]


def test_robot_visual_geometry_missing_requires_renderable_gprim() -> None:
    assert M._robot_visual_geometry_missing(None) is True
    assert M._robot_visual_geometry_missing({"gprim_count": 0, "blockers": []}) is True
    assert M._robot_visual_geometry_missing({
        "gprim_count": 4,
        "blockers": ["robot_gprims_unmaterialized"],
    }) is False
    assert M._robot_visual_geometry_missing({
        "gprim_count": 4,
        "blockers": [M.ROBOT_VISUAL_MESH_MISSING_BLOCKER],
    }) is True


def test_bind_g1_visual_fallback_preserves_articulation_candidate_when_visual_missing(
    monkeypatch,
) -> None:
    class _Prim:
        def __init__(self, stage):
            self._stage = stage

        def IsValid(self) -> bool:
            return bool(self._stage.bound)

    class _Stage:
        def __init__(self) -> None:
            self.bound: list[str] = []
            self.removed = 0

        def GetPrimAtPath(self, _path):
            return _Prim(self)

        def RemovePrim(self, _path) -> None:
            self.removed += 1

    stage = _Stage()
    monkeypatch.setattr(M, "_g1_visual_asset_candidates", lambda _value: ["physics.usd", "missing.usd"])
    monkeypatch.setattr(M, "_resolve_asset_uri", lambda value: f"resolved:{value}")

    def fake_bind(fake_stage, resolved, prim_path="/World/G1"):
        fake_stage.bound.append(resolved)
        has_physics = resolved.endswith("physics.usd")
        return {
            "prim_path": prim_path,
            "controllable_articulation_detected": has_physics,
            "collision_enabled_verified": has_physics,
            "articulation_root_api_prim_count": 1 if has_physics else 0,
            "collision_api_prim_count": 1 if has_physics else 0,
            "resolved_g1_usd": resolved,
        }

    def fake_diag(fake_stage, _prim_path):
        resolved = fake_stage.bound[-1]
        return {
            "status": "FAIL",
            "blockers": [M.ROBOT_VISUAL_MESH_MISSING_BLOCKER],
            "gprim_count": 0,
            "mesh_count": 0,
            "resolved_seen_by_test": resolved,
        }

    monkeypatch.setattr(M, "_bind_g1", fake_bind)
    monkeypatch.setattr(M, "_robot_render_visibility_diagnostics", fake_diag)

    binding = M._bind_g1_with_visual_fallback(stage, "requested.usd")

    assert binding["candidate_g1_usd"] == "physics.usd"
    assert binding["resolved_g1_usd"] == "resolved:physics.usd"
    assert binding["visual_binding_status"] == "blocked_missing_renderable_robot_geometry"
    assert binding["selected_nonvisual_candidate_reason"].startswith("preserved_articulation")
    assert stage.bound == [
        "resolved:physics.usd",
        "resolved:missing.usd",
        "resolved:physics.usd",
    ]


def test_open_stage_waits_for_async_context_stage(monkeypatch) -> None:
    stage = object()
    state = {"updates": 0, "opened": None}

    class _Context:
        def open_stage(self, path):
            state["opened"] = path
            return True

        def get_stage(self):
            return stage if state["updates"] >= 2 else None

    class _App:
        def update(self):
            state["updates"] += 1

    omni_mod = types.ModuleType("omni")
    usd_mod = types.ModuleType("omni.usd")
    usd_mod.get_context = lambda: _Context()
    kit_mod = types.ModuleType("omni.kit")
    app_mod = types.ModuleType("omni.kit.app")
    app_mod.get_app = lambda: _App()
    omni_mod.usd = usd_mod
    omni_mod.kit = kit_mod
    kit_mod.app = app_mod
    monkeypatch.setitem(sys.modules, "omni", omni_mod)
    monkeypatch.setitem(sys.modules, "omni.usd", usd_mod)
    monkeypatch.setitem(sys.modules, "omni.kit", kit_mod)
    monkeypatch.setitem(sys.modules, "omni.kit.app", app_mod)

    assert M._open_stage("/workspace/kitchen.usd", timeout_s=1.0) is stage
    assert state["opened"] == "/workspace/kitchen.usd"
    assert state["updates"] == 2


def test_manipulation_cam_is_egocentric_vs_follow_chase() -> None:
    root, yaw = (1.75, 1.25, 0.79), 0.0  # robot at the sink, facing +x
    me, mt = M.manipulation_cam_pose(root, yaw)
    fe, ft = M.follow_cam_pose(root, yaw)
    # manipulation eye: head height + slightly FORWARD of the root (egocentric)
    assert me[2] > root[2] + 0.4
    assert me[0] >= root[0]
    # follow eye: BEHIND the root (the chase shot that gave OSCAR a room-scale navigation view)
    assert fe[0] < root[0]
    # manipulation looks DOWN-forward at the workspace (target ahead of root, below eye, counter level)
    assert mt[0] > root[0]
    assert mt[2] < me[2]
    assert mt[2] < 1.0


def test_manipulation_cam_fixed_look_at_pins_faucet_regardless_of_yaw() -> None:
    # robot standing in front of the sink; faucet world point is known
    faucet = (2.28, 1.33, 0.9)
    e1, t1 = M.manipulation_cam_pose((2.28, 0.73, 0.79), math.pi / 2, look_at=faucet)
    # target stays anchored near the faucet while blending toward the active arm corridor.
    assert math.dist(t1, faucet) < 0.25
    assert e1[2] > 1.0
    # a wrong/noisy final yaw must NOT fall back to a yaw-relative navigation target.
    _, t2 = M.manipulation_cam_pose((2.28, 0.73, 0.79), -math.pi / 2, look_at=faucet)
    assert math.dist(t2, faucet) < 0.25
    # without look_at it falls back to the yaw-relative target (forward of the robot) — a robot
    # standing elsewhere/facing elsewhere then frames its own front, NOT the faucet
    _, t3 = M.manipulation_cam_pose((1.0, 0.0, 0.79), 0.0)  # at [1,0] facing +x
    assert t3 != faucet and t3[0] > 1.0  # forward (+x) of the root, not the sink


def test_surface_affordance_point_projects_target_to_nearest_face() -> None:
    stance_plan = {
        "task_target_xyz": [-1.979559, 0.655166, 1.025963],
        "task_target_bounds": {
            "bbox_min_xyz": [-2.521971, 0.13306, -0.000508],
            "bbox_max_xyz": [-1.437147, 1.177273, 2.052435],
        },
    }

    point = M._surface_affordance_point_for_stance(
        stance_plan,
        root_pose=(-1.035327, 0.658475, 0.84),
    )

    assert point == pytest.approx((-1.437147, 0.655166, 1.025963))


def test_manipulation_cam_with_look_at_frames_active_arm_corridor() -> None:
    root = (-1.035327, 0.658475, 0.84)
    yaw = -3.138089
    look_at = (-1.437147, 0.655166, 1.025963)

    eye, target = M.manipulation_cam_pose(root, yaw, look_at=look_at, reach_arm="right")

    assert target[0] > look_at[0]  # target is blended toward the active shoulder, not inside target
    assert target[1] > look_at[1]
    assert target[2] > look_at[2]
    assert eye[0] < root[0]       # head-mounted seed: slightly forward toward the fridge
    assert abs(eye[1] - root[1]) < 0.02
    assert eye[2] > look_at[2]

    left_eye, left_target = M.manipulation_cam_pose(root, yaw, look_at=look_at, reach_arm="left")
    assert abs(left_eye[1] - root[1]) < 0.02
    assert left_target[1] < look_at[1]


def test_manipulation_camera_target_blends_affordance_with_visible_arm_context() -> None:
    affordance = (-1.44, 0.66, 1.03)
    arm_points = {
        "elbow": (-1.05, 0.55, 1.10),
        "wrist": (-1.22, 0.58, 0.98),
        "hand": (-1.34, 0.62, 0.96),
    }

    target = M._manipulation_camera_target_with_arm_context(affordance, arm_points)

    assert target[0] > affordance[0]
    assert target[1] < affordance[1]
    assert abs(target[2] - affordance[2]) < 0.08
    assert math.dist(target, affordance) < 0.25


def test_robot_head_lens_eye_offsets_link_origin_out_of_head_mesh() -> None:
    eye, meta = M._robot_head_lens_eye_from_mount((-1.0, 0.6, 1.35), math.pi)

    assert eye[0] < -1.0
    assert eye[1] == pytest.approx(0.6)
    assert eye[2] > 1.35
    assert meta["raw_mount_eye_xyz"] == [-1.0, 0.6, 1.35]
    assert meta["lens_height_correction_applied"] is False

    low_eye, low_meta = M._robot_head_lens_eye_from_mount(
        (-0.86, 0.65, 0.84),
        math.pi,
        root_pose=(-1.04, 0.65, 0.84),
        arm_points={
            "shoulder": (-0.87, 0.65, 1.13),
            "wrist": (-0.98, 0.65, 0.94),
            "hand": (-1.15, 0.65, 0.95),
        },
    )
    assert -1.02 < low_eye[0] < -0.90
    assert 1.25 < low_eye[2] < 1.32
    assert low_meta["lens_height_correction_applied"] is True
    assert low_meta["min_head_lens_z"] > 1.2
    assert 0.08 <= low_meta["shoulder_to_lens_z_m"] <= 0.18
    # The fallback lens stays behind the forearm; otherwise wrist/elbow links project behind the
    # head camera and only gripper tips can appear at the bottom of the POV frame.
    assert low_eye[0] > -0.98

    bounded, bounded_meta = M._robot_head_lens_eye_from_mount(
        (-0.86, 0.65, 0.84),
        math.pi,
        root_pose=(-1.04, 0.65, 0.84),
        arm_points={"shoulder": (-0.87, 0.65, 1.13)},
        head_bounds={
            "source_prim_path": "/World/G1/torso_link/head_link",
            "bbox_min_xyz": [-0.95, 0.55, 1.2],
            "bbox_max_xyz": [-0.75, 0.75, 1.4],
            "center_xyz": [-0.85, 0.65, 1.3],
            "size_xyz": [0.2, 0.2, 0.2],
        },
    )
    assert 1.30 < bounded[2] < 1.34
    assert bounded_meta["head_lens_z_source"] == "head_bounds_center_above_shoulders"

    authored, authored_meta = M._robot_head_lens_eye_from_mount(
        (-1.0, 0.6, 1.35),
        math.pi,
        root_pose=(-1.04, 0.65, 0.84),
        arm_points={"shoulder": (-0.87, 0.65, 1.13)},
        authored_camera=True,
    )
    assert authored == (-1.0, 0.6, 1.35)
    assert authored_meta["lens_offset_xyz_robot_frame"] == [0.0, 0.0, 0.0]
    assert authored_meta["lens_height_correction_applied"] is False


def test_robot_mounted_manipulation_camera_metadata_is_replayable() -> None:
    source = _RUNNER.read_text()

    assert '"camera_eye_xyz": [round(float(v), 6) for v in eye]' in source
    assert '"camera_target_xyz": [round(float(v), 6) for v in target]' in source
    assert '"camera_vfov_deg": round(float(vfov_deg), 6)' in source
    assert '"viewport_size_px": [int(width), int(height)]' in source


def test_local_dry_render_uses_same_min_manipulation_fov_as_gpu_runner() -> None:
    source = _RUNNER.read_text()

    render_local_preview = source[source.index("def render_local_preview("):]
    render_local_preview = render_local_preview[: render_local_preview.index("def main(")]

    assert "MANIPULATION_POV_MIN_VFOV_DEG" in render_local_preview
    assert "max(float(camera_vfov_deg), 90.0)" not in render_local_preview


def test_real_g1_visual_meshes_preserve_authored_materials_by_default() -> None:
    source = _RUNNER.read_text()

    assert "bool(robot_review_material_override)" in source
    assert 'os.getenv("PARITY_ROBOT_REVIEW_MATERIAL_OVERRIDE", "") == "1"' in source
    assert "override_robot_material = bool(robot_visual_missing or force_review_material)" in source
    assert "override_authored_materials=override_robot_material" in source
    assert '"authored_robot_materials_preserved"] = not bool(override_robot_material)' in source


def test_manipulation_seed_arm_target_is_forward_ready_not_affordance_contact() -> None:
    shoulder = (0.0, 0.0, 1.2)
    low_handle = (0.45, 0.0, 0.85)
    high_handle = (0.45, 0.0, 1.45)

    seed = M._manipulation_seed_arm_target_for_shoulder(
        shoulder,
        low_handle,
        forward_yaw=0.0,
    )
    assert seed[0] > shoulder[0]
    assert seed[1] == pytest.approx(shoulder[1])
    assert seed[2] > low_handle[2]
    assert seed[2] < shoulder[2]

    high_seed = M._manipulation_seed_arm_target_for_shoulder(
        shoulder,
        high_handle,
        forward_yaw=0.0,
    )
    assert high_seed[0] > shoulder[0]
    assert high_seed[1] == pytest.approx(shoulder[1])
    assert high_seed[2] > shoulder[2]
    assert high_seed[2] <= shoulder[2] + M.MANIPULATION_HIGH_REACH_MAX_SEED_Z_ABOVE_SHOULDER_M
    assert high_seed != high_handle
    assert high_seed[2] > seed[2]

    side_handle = (0.0, 0.45, 0.95)
    fallback_seed = M._manipulation_seed_arm_target_for_shoulder(shoulder, side_handle)
    assert fallback_seed[0] > shoulder[0]
    assert fallback_seed[1] == pytest.approx(shoulder[1])
    assert fallback_seed != side_handle


def test_manipulation_reach_target_blends_to_affordance_only_at_endpoint() -> None:
    shoulder = (0.0, 0.0, 1.2)
    side_handle = (0.0, 0.45, 1.0)

    seed = M._manipulation_arm_target_for_reach_fraction(
        shoulder,
        side_handle,
        M.MANIPULATION_ENDPOINT_AFFORDANCE_AIM_START_FRACTION,
        forward_yaw=0.0,
    )
    endpoint = M._manipulation_arm_target_for_reach_fraction(
        shoulder,
        side_handle,
        1.0,
        forward_yaw=0.0,
    )

    assert seed[0] > shoulder[0]
    assert seed[1] == pytest.approx(shoulder[1])
    assert seed != side_handle
    assert endpoint == pytest.approx(side_handle)


def test_manipulation_arm_link_name_filter_is_side_and_arm_specific() -> None:
    assert M._is_manipulation_arm_link_name("right_shoulder_pitch_link", "right")
    assert M._is_manipulation_arm_link_name("right_wrist_roll_link", "right")
    assert M._is_manipulation_arm_link_name("left_gripper_finger_link", "left")
    assert not M._is_manipulation_arm_link_name("right_hip_pitch_link", "right")
    assert not M._is_manipulation_arm_link_name("left_wrist_roll_link", "right")
    assert not M._is_manipulation_arm_link_name("bright_panel_link", "right")


def test_manipulation_camera_target_selection_rejects_downward_pitch_workaround() -> None:
    eye = (-1.14832, 0.655171, 1.2802)
    affordance = (-1.98, 0.66, 1.03)
    # This mirrors the bad seed class: low wrist/hand points can be made visible only by aiming the
    # robot-head camera steeply downward. The selector should keep the frame head-forward and let the
    # geometry gate fail on the arm seed instead of choosing a downward workaround.
    arm_points = {
        "shoulder": (-0.868313, 0.655166, 1.13178),
        "elbow": (-0.884094, 0.655166, 0.945243),
        "wrist": (-0.984094, 0.655166, 0.935237),
        "hand": (-1.148447, 0.65439, 0.948114),
    }

    _target, meta = M._select_manipulation_camera_target_for_visible_arm(
        affordance,
        arm_points,
        eye,
        M._manipulation_camera_target_with_arm_context(affordance, arm_points),
        vfov_deg=M.MANIPULATION_POV_MIN_VFOV_DEG,
        width=1280,
        height=960,
        arm="both",
        arm_points_by_arm={"left": arm_points, "right": arm_points},
    )

    chosen = next(
        c for c in meta["camera_target_candidates"]
        if c["candidate"] == meta["selected_camera_target"]
    )
    assert chosen["selection_allowed"] is True
    assert chosen["pitch_down_deg"] <= M.MANIPULATION_POV_MAX_CAMERA_PITCH_DOWN_DEG
    assert "manipulation_pov_camera_pitched_down_too_far" not in chosen["blockers"]


def test_manipulation_camera_target_selection_accepts_hand_wrist_seed_without_elbow() -> None:
    eye = (-0.938489, 0.655171, 1.2802)
    affordance = (-1.437147, 0.655166, 1.025963)
    arm_points_by_arm = {
        "left": {
            "elbow": (-0.958272, 0.543407, 0.961516),
            "hand": (-1.225504, 0.61788, 1.085062),
            "shoulder": (-0.868313, 0.554951, 1.13178),
            "wrist": (-1.05134, 0.570512, 0.988109),
        },
        "right": {
            "elbow": (-0.988705, 0.755822, 0.981045),
            "hand": (-1.176726, 0.701394, 1.093561),
            "shoulder": (-0.868313, 0.755381, 1.13178),
            "wrist": (-1.074752, 0.730238, 1.026266),
        },
    }
    arm_points = M._average_arm_link_points(arm_points_by_arm)

    target, meta = M._select_manipulation_camera_target_for_visible_arm(
        affordance,
        arm_points,
        eye,
        M._manipulation_camera_target_with_arm_context(affordance, arm_points),
        vfov_deg=90.0,
        width=1280,
        height=960,
        arm="both",
        arm_points_by_arm=arm_points_by_arm,
    )
    geom = M._manipulation_pov_geometry(
        arm_points=arm_points,
        arm_points_by_arm=arm_points_by_arm,
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=M.MANIPULATION_POV_MIN_VFOV_DEG,
        width=1280,
        height=960,
        arm="both",
    )

    assert meta["selected_camera_target"].startswith("head_forward_pitch_limited_")
    assert geom["status"] == "PASS"
    assert geom["seed_arm_visibility"]["status"] == "PASS"
    assert "manipulation_pov_arm_chain_not_in_frame" not in geom["blockers"]
    assert geom["reach_feasibility"]["status"] == "PASS"
    assert "manipulation_pov_left_arm_seed_failed" not in geom["blockers"]
    assert "manipulation_pov_affordance_outside_g1_reach_envelope" not in geom["blockers"]
    assert geom["camera_pitch_down_deg"] <= M.MANIPULATION_POV_HEAD_FORWARD_PITCH_DOWN_DEG
    assert geom["arm_roles_in_frame_by_arm"]["left"] == ["hand", "wrist"]
    assert geom["arm_roles_in_frame_by_arm"]["right"] == ["hand", "wrist"]
    assert geom["arm_roles_usefully_in_frame_by_arm"]["left"] == ["hand"]
    assert geom["arm_roles_usefully_in_frame_by_arm"]["right"] == ["hand", "wrist"]
    assert geom["seed_arm_visibility"]["by_arm"]["left"]["arm_chain_roles_in_frame"] == ["hand", "wrist"]
    assert geom["seed_arm_visibility"]["by_arm"]["right"]["arm_chain_roles_in_frame"] == ["hand", "wrist"]


def test_render_step_watchdog_timeout_result_is_fail_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PARITY_RENDER_STEP_WATCHDOG_SECONDS", raising=False)
    assert M._render_step_watchdog_seconds() == M.DEFAULT_RENDER_STEP_WATCHDOG_SECONDS
    monkeypatch.setenv("PARITY_RENDER_STEP_WATCHDOG_SECONDS", "12.5")
    assert M._render_step_watchdog_seconds() == pytest.approx(12.5)
    monkeypatch.setenv("PARITY_RENDER_STEP_WATCHDOG_SECONDS", "not-a-number")
    assert M._render_step_watchdog_seconds() == M.DEFAULT_RENDER_STEP_WATCHDOG_SECONDS

    result_path = tmp_path / "isaac_g1_kitchen_parity_result.json"
    M._write_render_step_timeout_result(
        result_path,
        label="scenario:warmup:0",
        seconds=12.5,
        scenario_id="scenario",
    )

    payload = json.loads(result_path.read_text())
    assert payload["status"] == "blocked"
    assert payload["blockers"] == ["render_step_timeout"]
    assert payload["render_step_timeout"]["label"] == "scenario:warmup:0"
    assert payload["render_step_timeout"]["scenario_id"] == "scenario"
    assert payload["rendered_by_isaac_rtx"] is True


def test_render_quality_config_defaults_seed_frames_to_realtime_and_keeps_explicit_pathtraced(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PARITY_RENDER_QUALITY_MODE", raising=False)
    monkeypatch.delenv("PARITY_PATH_TRACING_SAMPLES_PER_PIXEL", raising=False)

    cfg = M._render_quality_config(
        render_subframes=32,
        manipulation_cam=True,
        verify_cam=True,
    )

    assert cfg["use_pathtraced"] is False
    assert cfg["samples_per_pixel"] == 0
    assert cfg["optix_denoiser_requested"] is False
    assert cfg["firefly_filter_requested"] is False

    realtime = M._render_quality_config(
        render_subframes=32,
        manipulation_cam=True,
        verify_cam=True,
        mode="realtime",
    )
    assert realtime["use_pathtraced"] is False

    single_plain = M._render_quality_config(
        render_subframes=1,
        manipulation_cam=False,
        verify_cam=False,
    )
    assert single_plain["use_pathtraced"] is False

    forced = M._render_quality_config(
        render_subframes=1,
        manipulation_cam=False,
        verify_cam=False,
        mode="pathtraced",
    )
    assert forced["use_pathtraced"] is True
    # Explicit path tracing still uses the audit-proven floor. It is opt-in because source seed
    # cleanliness beats physically richer GI/reflections for policy/WAM observation frames.
    assert forced["samples_per_pixel"] == M.DEFAULT_PATH_TRACING_MIN_SAMPLES_PER_PIXEL
    assert forced["samples_per_pixel"] == 384
    assert forced["optix_denoiser_requested"] is True
    assert forced["firefly_filter_requested"] is True
    # Path-traced steps stay at 1 subframe: the sample lever is the explicit
    # /rtx/pathtracing/spp per-frame budget set by _apply_render_quality_settings
    # (8 subframes measurably did NOT reduce noise while spp was starved, and only
    # multiply cost once spp is correct — metallic re-test 2026-07-02).
    assert M._effective_render_rt_subframes(32, forced) == 1
    assert M._effective_render_rt_subframes(32, cfg) == 32
    assert M._effective_render_rt_subframes(32, realtime) == 32


def test_default_software_denoise_does_not_upgrade_random_noise_to_source_qa(
    monkeypatch,
    tmp_path,
) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    from PIL import Image  # type: ignore
    from blueprint_pipeline.wam_generated_video_review import (
        assess_source_policy_observation_visual_qa,
    )

    monkeypatch.delenv("PARITY_SOFTWARE_DENOISE_MODE", raising=False)
    rng = np.random.default_rng(17)
    noisy = Image.fromarray(
        rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8),
        mode="RGB",
    )
    denoised = M._software_denoise_image(noisy)
    frame = tmp_path / "denoised.png"
    denoised.save(frame)

    qa = assess_source_policy_observation_visual_qa(
        frame,
        generated_at="now",
        target_object_id="Sink054_handle",
        task_id="sink_faucet",
        visual_profile="review_quality",
        review_quality_required=True,
    )

    assert qa["status"] == "failed_visual_quality_gate"
    assert "source_policy_observation_speckled_or_noisy_for_review_quality" in qa["blockers"]
    assert qa["metrics"]["edge_density"] < 0.45


def test_path_traced_rt_subframes_env_override_is_bounded(monkeypatch) -> None:
    quality = {"use_pathtraced": True}

    monkeypatch.setenv("PARITY_PATH_TRACED_RT_SUBFRAMES", "4")
    assert M._effective_render_rt_subframes(32, quality) == 4

    monkeypatch.setenv("PARITY_PATH_TRACED_RT_SUBFRAMES", "99")
    assert M._effective_render_rt_subframes(32, quality) == 8

    monkeypatch.setenv("PARITY_PATH_TRACED_RT_SUBFRAMES", "not-a-number")
    assert M._effective_render_rt_subframes(32, quality) == 1


def test_replicator_step_waits_for_accumulated_subframes_before_saving() -> None:
    source = _RUNNER.read_text()

    helper = source.index("def _replicator_step_with_watchdog(")
    step = source.index("rep.orchestrator.step(rt_subframes=subframes)", helper)
    wait = source.index("rep.orchestrator.wait_until_complete()", step)
    save = source.index("def _save_rgb(", wait)

    assert step < wait < save


def test_task_visual_qc_splits_verify_and_pov_rubrics(monkeypatch, tmp_path) -> None:
    qc_mod = __import__("blueprint_pipeline.render_visual_qc", fromlist=["dummy"])
    calls: dict[str, list[str]] = {}

    def fake_placement(frames, target, *, task_description="", sample_n=4, generate=None):
        calls["placement"] = [Path(p).name for p in frames]
        return {
            "schema_version": "robot_placement_visual_qc.v1",
            "status": "passed",
            "target": target,
            "task_description": task_description,
            "frames_reviewed": len(frames),
            "blockers": [],
            "per_frame": [],
        }

    def fake_pov(frames, target, *, task_description="", sample_n=4, generate=None):
        calls["pov"] = [Path(p).name for p in frames]
        return {
            "schema_version": "manipulation_pov_visual_qc.v1",
            "status": "passed",
            "target": target,
            "task_description": task_description,
            "frames_reviewed": len(frames),
            "blockers": [],
            "per_frame": [],
        }

    monkeypatch.setattr(qc_mod, "qc_robot_placement_frames", fake_placement)
    monkeypatch.setattr(qc_mod, "qc_manipulation_pov_frames", fake_pov)
    verify = tmp_path / "verify_0000.png"
    pov = tmp_path / "robot_pov_0000.png"

    report = M._run_task_visual_qc(
        [verify],
        [pov],
        target_label="refrigerator",
        task_description="open the refrigerator",
    )

    assert report["status"] == "passed"
    assert calls == {"placement": ["verify_0000.png"], "pov": ["robot_pov_0000.png"]}
    assert report["placement"]["schema_version"] == "robot_placement_visual_qc.v1"
    assert report["manipulation_pov"]["schema_version"] == "manipulation_pov_visual_qc.v1"


def test_task_visual_qc_blocks_when_manipulation_pov_has_no_frames(monkeypatch, tmp_path) -> None:
    qc_mod = __import__("blueprint_pipeline.render_visual_qc", fromlist=["dummy"])

    def fake_placement(frames, target, *, task_description="", sample_n=4, generate=None):
        return {
            "schema_version": "robot_placement_visual_qc.v1",
            "status": "passed",
            "target": target,
            "task_description": task_description,
            "frames_reviewed": len(frames),
            "blockers": [],
            "per_frame": [],
        }

    monkeypatch.setattr(qc_mod, "qc_robot_placement_frames", fake_placement)
    verify = tmp_path / "verify_0000.png"

    report = M._run_task_visual_qc(
        [verify],
        [],
        target_label="refrigerator",
        task_description="open the refrigerator",
    )

    assert report["status"] == "blocked"
    assert "manipulation_pov_visual_qc_no_frames" in report["blockers"]
    assert report["manipulation_pov"]["status"] == "blocked"


def test_arm_reach_skeleton_moves_hand_toward_faucet_and_into_view() -> None:
    # a minimal right-arm chain at rest: arm hangs forward-low at the body (y ~ 0), hand at y=0.25
    rest = [
        ("torso_link", (2.28, 0.73, 1.1)),
        ("right_shoulder_link", (2.18, 0.73, 1.10)),
        ("right_elbow_link", (2.18, 0.78, 0.95)),
        ("right_wrist_link", (2.18, 0.83, 0.85)),
        ("right_hand_palm_link", (2.18, 0.88, 0.80)),
    ]
    faucet = (2.28, 1.33, 0.95)
    rest_hand = rest[-1][1]
    # at reach_frac=0 nothing moves
    assert M.compute_arm_reach_skeleton(rest, faucet, 0.0) == rest
    # at full reach the hand is much closer to the faucet than at rest
    full = dict(M.compute_arm_reach_skeleton(rest, faucet, 1.0, forward_yaw=math.pi / 2))

    def d(a, b):
        return math.dist(a, b)

    assert d(full["right_hand_palm_link"], faucet) < d(rest_hand, faucet)
    # the hand advances toward the faucet in +y (into the camera's forward view)
    assert full["right_hand_palm_link"][1] > rest_hand[1]
    # the arm never overstretches beyond its rest length from the shoulder
    sh = full["right_shoulder_link"]
    arm_len = d(rest[1][1], rest_hand)
    assert d(full["right_hand_palm_link"], sh) <= arm_len + 1e-6
    # non-arm links (torso) are untouched
    assert full["torso_link"] == (2.28, 0.73, 1.1)
    # the reach is monotonic: half-reach hand sits between rest and full
    half = dict(M.compute_arm_reach_skeleton(rest, faucet, 0.5, forward_yaw=math.pi / 2))
    assert rest_hand[1] < half["right_hand_palm_link"][1] < full["right_hand_palm_link"][1]


def test_arm_reach_skeleton_endpoint_aims_at_affordance_not_just_forward_seed() -> None:
    rest = [
        ("torso_link", (0.0, 0.0, 1.1)),
        ("right_shoulder_link", (0.0, 0.0, 1.0)),
        ("right_elbow_link", (0.15, 0.0, 1.0)),
        ("right_wrist_link", (0.30, 0.0, 1.0)),
        ("right_hand_palm_link", (0.45, 0.0, 1.0)),
    ]
    side_handle = (0.0, 0.45, 1.0)

    seed_phase = dict(
        M.compute_arm_reach_skeleton(
            rest,
            side_handle,
            M.MANIPULATION_ENDPOINT_AFFORDANCE_AIM_START_FRACTION,
            forward_yaw=0.0,
        )
    )
    endpoint = dict(M.compute_arm_reach_skeleton(rest, side_handle, 1.0, forward_yaw=0.0))

    assert seed_phase["right_hand_palm_link"][0] > 0.0
    assert seed_phase["right_hand_palm_link"][1] < 0.05
    assert endpoint["right_hand_palm_link"][1] > 0.40
    assert math.dist(endpoint["right_hand_palm_link"], side_handle) < 0.02


def test_manipulation_ready_arm_pose_defaults_to_both_arms() -> None:
    deltas = M.manipulation_ready_arm_joint_deltas()
    assert deltas["left_shoulder_pitch_joint"] < 0
    assert deltas["right_shoulder_pitch_joint"] < 0
    assert deltas["left_elbow_joint"] < 0
    assert deltas["right_elbow_joint"] < 0
    assert "left_wrist_pitch_joint" in deltas
    assert "right_wrist_pitch_joint" in deltas

    right_only = M.manipulation_ready_arm_joint_deltas("right")
    assert "right_shoulder_pitch_joint" in right_only
    assert "left_shoulder_pitch_joint" not in right_only

    with pytest.raises(ValueError):
        M.manipulation_ready_arm_joint_deltas("center")


def test_manipulation_pov_render_and_validation_use_same_arm_selection() -> None:
    source = _RUNNER.read_text()
    assert "requested_reach_arm = _normalize_reach_arm_selection(manipulation_reach_arm)" in source
    assert "_resolve_manipulation_reach_arm_selection(" in source
    assert "pov_reach_arm = resolved_reach_arm" in source
    assert "rendered_reach_arm = pov_reach_arm" in source
    assert "rendered_reach_arm = resolved_reach_arm" in source
    assert "arm=rendered_reach_arm" in source
    assert "manipulation_reach_arm=rendered_reach_arm" in source
    assert "forward_yaw=decision.yaw" in source
    assert "arm=pov_reach_arm" in source
    assert 'arm_points_by_arm=cam_meta.get("arm_link_points_by_arm_xyz") or {}' in source
    assert "reach_arm = args.manipulation_reach_arm" in source
    assert 'default="auto", choices=["auto", "right", "left", "both"]' in source
    assert 'if args.manipulation_reach_arm != "both" else "right"' not in source


def test_auto_manipulation_reach_arm_resolves_from_final_stance_and_affordance() -> None:
    stale_candidate_plan = {
        "status": "accepted",
        "selected_candidate_index": 0,
        "candidates": [
            {
                "reachability_estimate": {
                    "status": "PASS",
                    "best_reach_arm": "right",
                    "passing_arms": ["right"],
                    "by_arm": {
                        "right": {
                            "status": "PASS",
                            "seed_effector_to_affordance_m": 0.20,
                            "shoulder_to_affordance_m": 0.45,
                        },
                        "left": {
                            "status": "FAIL",
                            "seed_effector_to_affordance_m": 0.40,
                            "shoulder_to_affordance_m": 0.65,
                        },
                    },
                }
            }
        ],
    }
    # This mirrors the microwave run: the accepted stance plus final resolved
    # affordance makes the left arm the better single-arm review pose even if an
    # earlier coarse candidate record preferred the right arm.
    resolved = M._resolve_manipulation_reach_arm_selection(
        "auto",
        stance_plan=stale_candidate_plan,
        affordance=(-1.591312, 1.471274, 1.241574),
        root_pose=(-1.313504, 1.518814, 0.84),
        yaw=-2.972108,
    )

    assert resolved == "left"
    assert (
        M._resolve_manipulation_reach_arm_selection("both", stance_plan=stale_candidate_plan)
        == "both"
    )
    assert (
        M._resolve_manipulation_reach_arm_selection("right", stance_plan=stale_candidate_plan)
        == "right"
    )


def test_apply_joint_deltas_updates_only_available_joint_targets() -> None:
    targets = [0.0, 0.0, 0.0]
    default = [1.0, 2.0, 3.0]
    applied = M._apply_joint_deltas(
        targets,
        default,
        {"right_shoulder_pitch_joint": 0, "right_elbow_joint": 2},
        {
            "right_shoulder_pitch_joint": -0.85,
            "missing_joint": 99.0,
            "right_elbow_joint": -0.23,
        },
    )
    assert applied == ["right_shoulder_pitch_joint", "right_elbow_joint"]
    assert targets == pytest.approx([0.15, 0.0, 2.77])


def test_apply_named_joint_targets_overrides_available_absolute_targets() -> None:
    targets = [1.0, 2.0, 3.0]
    applied = M._apply_named_joint_targets(
        targets,
        {
            "right_shoulder_pitch_joint": 0,
            "right_elbow_joint": 2,
        },
        {
            "right_shoulder_pitch_joint": -0.4,
            "missing_joint": 99.0,
            "right_elbow_joint": "0.35",
            "bad_joint_value": object(),
        },
    )

    assert applied == ["right_shoulder_pitch_joint", "right_elbow_joint"]
    assert targets == pytest.approx([-0.4, 2.0, 0.35])


def test_action_record_persists_policy_joint_targets() -> None:
    decision = M.policy_mod.StepDecision(
        root_pose=(0.0, 0.0, 0.79),
        yaw=0.0,
        desired_root_position=(0.0, 0.0, 0.79),
        route_segment_index=0,
        policy_action="learned_policy_action",
        collision_probe_candidate_count=0,
        rejected_collision_probe_count=0,
        joint_targets={"right_elbow_joint": 0.35},
    )

    record = M.policy_mod.action_record(
        decision=decision,
        step=0,
        sim_time_s=0.0,
        target=(0.0, 0.0, 0.79),
    )

    assert record["joint_targets"] == {"right_elbow_joint": 0.35}
    assert record["joint_target_count"] == 1


def test_groot_policy_command_infer_builds_payload_and_returns_action(tmp_path: Path) -> None:
    frame = tmp_path / "robot_pov.png"
    frame.write_bytes(b"not-a-real-png-for-this-subprocess-test")
    fake_command = tmp_path / "fake_groot_policy_command.py"
    fake_command.write_text(
        "\n".join(
            [
                "import json, sys",
                "payload = json.loads(sys.stdin.read())",
                "obs = payload['observation']",
                "assert obs['visual_observation']['camera_frame_path']",
                "assert obs['unitree_g1_sonic_state']['projected_gravity'] == [0.0, 0.0, -1.0]",
                "print(json.dumps({'status':'completed','action':{"
                "'hand_targets':{'left_hand_joints':[0.11,0.22]},"
                "'policy_action':'unit-test-policy-action'}}))",
            ]
        ),
        encoding="utf-8",
    )
    infer = M._make_groot_policy_command_infer(
        command=f"{sys.executable} {fake_command}",
        scenario={"scenario_id": "sink", "task_instruction": "turn on the faucet"},
        call_dir=tmp_path / "calls",
        timeout_seconds=5,
    )

    action = infer({"step": 3, "camera_rgb": str(frame)})

    assert action["hand_targets"]["left_hand_joints"] == [0.11, 0.22]
    call = json.loads((tmp_path / "calls" / "groot_policy_call_0003.json").read_text())
    assert call["payload"]["observation"]["task_prompt"] == "turn on the faucet"
    assert call["payload"]["observation"]["step"] == 3
    assert call["command_value_redacted"] == "<configured>"


def test_arm_reach_skeleton_can_pose_both_arms_for_seed_phase() -> None:
    rest = [
        ("left_shoulder_link", (0.0, -0.2, 1.1)),
        ("left_hand_palm_link", (0.0, -0.45, 0.8)),
        ("right_shoulder_link", (0.0, 0.2, 1.1)),
        ("right_hand_palm_link", (0.0, 0.45, 0.8)),
    ]
    target = (0.5, 0.0, 0.95)
    seed_phase = dict(
        M.compute_arm_reach_skeleton(
            rest,
            target,
            M.MANIPULATION_ENDPOINT_AFFORDANCE_AIM_START_FRACTION,
            arm="both",
            forward_yaw=0.0,
        )
    )
    assert seed_phase["left_hand_palm_link"][0] > rest[1][1][0]
    assert seed_phase["right_hand_palm_link"][0] > rest[3][1][0]
    assert seed_phase["left_hand_palm_link"][1] < 0.0
    assert seed_phase["right_hand_palm_link"][1] > 0.0
    assert (
        seed_phase["right_hand_palm_link"][1] - seed_phase["left_hand_palm_link"][1]
        > 0.30
    )
    assert seed_phase["left_hand_palm_link"] != rest[1][1]
    assert seed_phase["right_hand_palm_link"] != rest[3][1]


def test_skeleton_world_for_frame_falls_back_when_articulation_has_no_links() -> None:
    offsets = [
        ("torso_link", (0.0, 0.0, 0.3)),
        ("right_hand_palm_link", (0.2, 0.1, 0.0)),
    ]
    skel = M.skeleton_world_for_frame(
        art_ctx={"art": object(), "link_names": []},
        rest_offsets=offsets,
        root_pose=(1.0, 2.0, 0.7),
        yaw=0.0,
    )
    assert skel == [
        ("torso_link", (1.0, 2.0, 1.0)),
        ("right_hand_palm_link", (1.2, 2.1, 0.7)),
    ]


def test_camera_aperture_widens_fov_vs_default_telephoto() -> None:
    # the POV camera was rendering at USD's ~17deg-vertical default (focal 50 / vap 15.29),
    # which zooms into the dark basin; we widen it to the projection FOV
    focal, hap, vap = M.camera_aperture_for_fov(50.0, 1280, 960)
    got_vfov = 2 * math.degrees(math.atan(vap / (2 * focal)))
    assert got_vfov == pytest.approx(50.0, abs=1e-6)          # vertical FOV matches the request
    assert hap / vap == pytest.approx(1280 / 960, abs=1e-6)   # horizontal aperture matches aspect
    # default telephoto would be ~17deg vertical -> the new FOV is much wider (less zoomed)
    default_vfov = 2 * math.degrees(math.atan(15.2908 / (2 * 50.0)))
    assert got_vfov > default_vfov + 25


def test_camera_intrinsics_from_usd_aperture_matches_pinhole() -> None:
    width, height, vfov = 1280, 960, 50.0
    focal, hap, vap = M.camera_aperture_for_fov(vfov, width, height)

    intr = M._camera_intrinsics_from_usd_aperture(focal, hap, vap, width, height)

    got_vfov = 2 * math.degrees(math.atan(0.5 * height / intr["fy"]))
    assert intr["available"] is True
    assert got_vfov == pytest.approx(vfov, abs=1e-6)
    assert intr["cx"] == pytest.approx(width / 2.0)
    assert intr["cy"] == pytest.approx(height / 2.0)
    assert intr["fx"] == pytest.approx(float(width) * float(focal) / float(hap))
    assert intr["fy"] == pytest.approx(float(height) * float(focal) / float(vap))
    assert intr["projection_method"] == "isaac_usd_camera_pinhole_from_focal_aperture"


def test_camera_contract_emission_wired_in_render_loop() -> None:
    source = _RUNNER.read_text()
    frame_save_start = source.index("over_ok = _save_rgb(")
    frame_save_end = source.index("cap += 1", frame_save_start)
    frame_save_region = source[frame_save_start:frame_save_end]

    assert "camera_contract.jsonl" in source
    assert "def _isaac_camera_contract" in source
    assert "def _append_camera_contract_row(" in source
    assert "_isaac_camera_contract(" in source
    assert "_isaac_camera_contract(stage, pov_cam, width, height)" in frame_save_region
    assert "pov_frame_path" in frame_save_region


def test_depth_pass_source_wiring_is_present() -> None:
    source = _RUNNER.read_text()

    assert "distance_to_image_plane" in source
    assert "def _save_depth(" in source
    assert "--depth-pass" in source
    assert "depth_render_pass" in source


def test_make_render_product_attaches_depth_to_same_render_product(monkeypatch) -> None:
    render_product = object()
    requested: list[object] = []
    render_products: list[tuple[str, tuple[int, int]]] = []

    class _Annotator:
        def __init__(self, name: str) -> None:
            self.name = name
            self.attached: list[list[object]] = []

        def attach(self, products) -> None:
            self.attached.append(list(products))

    def render_product_fn(camera_path, resolution):
        render_products.append((camera_path, tuple(resolution)))
        return render_product

    def get_annotator(name):
        annot = _Annotator(name)
        requested.append(annot)
        return annot

    fake_core = types.ModuleType("omni.replicator.core")
    fake_core.create = types.SimpleNamespace(render_product=render_product_fn)
    fake_core.AnnotatorRegistry = types.SimpleNamespace(get_annotator=get_annotator)
    fake_replicator = types.ModuleType("omni.replicator")
    fake_replicator.core = fake_core
    fake_omni = types.ModuleType("omni")
    fake_omni.replicator = fake_replicator
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    monkeypatch.setitem(sys.modules, "omni.replicator", fake_replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", fake_core)

    rgb, depth = M._make_render_product("/World/Cameras/pov", 64, 48, with_depth=True)

    assert render_products[-1] == ("/World/Cameras/pov", (64, 48))
    assert [annot.name for annot in requested] == ["rgb", "distance_to_image_plane"]
    assert rgb.attached == [[render_product]]
    assert depth.attached == [[render_product]]

    requested.clear()
    single = M._make_render_product("/World/Cameras/overview", 80, 60)

    assert single.name == "rgb"
    assert [annot.name for annot in requested] == ["rgb"]


def test_make_render_product_attaches_segmentation_annotators(monkeypatch) -> None:
    render_product = object()
    calls: list[tuple[str, object]] = []

    class _Annotator:
        def __init__(self, name: str, init_params=None) -> None:
            self.name = name
            self.init_params = init_params
            self.attached: list[list[object]] = []

        def attach(self, products) -> None:
            self.attached.append(list(products))

    def get_annotator(name, init_params=None):
        calls.append((name, init_params))
        return _Annotator(name, init_params)

    fake_core = types.ModuleType("omni.replicator.core")
    fake_core.create = types.SimpleNamespace(
        render_product=lambda _camera_path, _resolution: render_product
    )
    fake_core.AnnotatorRegistry = types.SimpleNamespace(get_annotator=get_annotator)
    fake_replicator = types.ModuleType("omni.replicator")
    fake_replicator.core = fake_core
    fake_omni = types.ModuleType("omni")
    fake_omni.replicator = fake_replicator
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    monkeypatch.setitem(sys.modules, "omni.replicator", fake_replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", fake_core)

    annots = M._make_render_product("/World/Cameras/pov", 64, 48, with_segmentation=True)

    assert [name for name, _params in calls] == [
        "rgb",
        "instance_segmentation",
        "semantic_segmentation",
    ]
    assert calls[1][1] == {"colorize": True}
    assert calls[2][1] == {"colorize": True}
    assert annots["rgb"].attached == [[render_product]]
    assert annots["instance"].attached == [[render_product]]
    assert annots["semantic"].attached == [[render_product]]

    calls.clear()
    single = M._make_render_product("/World/Cameras/overview", 80, 60)

    assert single.name == "rgb"
    assert calls == [("rgb", None)]


def test_dry_render_uses_nominal_skeleton_for_local_pov_geometry() -> None:
    source = _RUNNER.read_text()

    assert "nominal_g1_dry_render_skeleton" in source
    assert "_arm_link_points_by_arm_from_skeleton(" in source
    assert "manipulation_pov_geometry_unavailable_without_g1_link_geometry" not in source[
        source.index("def render_local_preview("):
    ]


def test_save_depth_writes_lossless_npy_and_preview(tmp_path: Path) -> None:
    import numpy as np

    arr = np.array([[0.0, 1.25], [2.5, np.inf]], dtype=np.float32)

    class _Annotator:
        def __init__(self, data) -> None:
            self._data = data

        def get_data(self):
            return self._data

    preview_path = tmp_path / "depth.png"
    raw_path = tmp_path / "depth.npy"

    assert M._save_depth(_Annotator(arr), preview_path, npy_path=raw_path) is True
    assert preview_path.exists()
    np.testing.assert_array_equal(np.load(raw_path), arr)

    none_path = tmp_path / "none.png"
    assert M._save_depth(_Annotator(None), none_path) is False
    assert not none_path.exists()
    assert not none_path.with_suffix(".npy").exists()

    empty_path = tmp_path / "empty.png"
    assert M._save_depth(_Annotator(np.array([], dtype=np.float32)), empty_path) is False
    assert not empty_path.exists()
    assert not empty_path.with_suffix(".npy").exists()


def test_save_segmentation_writes_masks_and_id_labels(tmp_path: Path) -> None:
    import numpy as np

    pytest.importorskip("PIL.Image")
    mask = np.array(
        [
            [[255, 0, 0, 255], [0, 255, 0, 255]],
            [[0, 0, 255, 255], [255, 255, 0, 255]],
        ],
        dtype=np.uint8,
    )

    class _Annotator:
        def __init__(self, data) -> None:
            self._data = data

        def get_data(self):
            return self._data

    result = M._save_segmentation(
        {
            "instance": _Annotator(
                {"data": mask, "info": {"idToLabels": {"1": {"class": "fridge"}}}}
            ),
            "semantic": _Annotator({"data": mask}),
        },
        instance_png=tmp_path / "seg" / "instance.png",
        semantic_png=tmp_path / "seg" / "semantic.png",
        id_label_json=tmp_path / "seg" / "id_labels.json",
    )

    assert result["instance_saved"] is True
    assert result["semantic_saved"] is True
    assert result["blockers"] == []
    assert Path(result["instance_png"]).is_file()
    assert Path(result["semantic_png"]).is_file()
    assert json.loads(Path(result["id_label_json"]).read_text(encoding="utf-8")) == {
        "1": {"class": "fridge"}
    }


def test_author_scene_semantic_labels_skips_robot_and_debug_prims(monkeypatch) -> None:
    labeled: list[tuple[str, str, str]] = []

    class _Prim:
        def __init__(self, path: str, name: str) -> None:
            self._path = path
            self._name = name

        def GetPath(self):
            return self._path

        def GetName(self):
            return self._name

        def IsA(self, _schema) -> bool:
            return True

    class _Stage:
        def Traverse(self):
            return [
                _Prim("/World/G1/body", "body"),
                _Prim("/World/PlacementDebug/box", "box"),
                _Prim("/World/Kitchen/Counter_12", "Counter_12"),
            ]

    def add_prim_semantics(prim, semantic_label=None, type_label=None):
        labeled.append((str(prim.GetPath()), str(semantic_label), str(type_label)))

    fake_semantics = types.ModuleType("semantics")
    fake_schema_editor = types.ModuleType("semantics.schema_editor")
    fake_schema_editor.add_prim_semantics = add_prim_semantics
    fake_semantics.schema_editor = fake_schema_editor
    monkeypatch.setitem(sys.modules, "semantics", fake_semantics)
    monkeypatch.setitem(sys.modules, "semantics.schema_editor", fake_schema_editor)

    summary = M._author_scene_semantic_labels(
        _Stage(),
        robot_prim_path="/World/G1",
        keep_substrings=("room", "floor"),
    )

    assert summary["labeled_prim_count"] == 1
    assert labeled == [("/World/Kitchen/Counter_12", "counter", "class")]
    assert summary["sample_labels"][0]["semantic_label"] == "counter"


def test_build_result_segmentation_pass_is_isaac_only_and_gated() -> None:
    base = {
        "scenarios": [{"scenario_id": "s0"}],
        "outcomes": [{"task_success": True}],
        "policy_id": "p",
        "kitchen_usd": "k.usd",
        "g1_usd": "g.usd",
        "blockers": [],
    }

    no_seg = M.build_result(**base)
    proven = M.build_result(
        **base,
        segmentation_summary={
            "labeled_prim_count": 2,
            "instance_mask_frames": 1,
            "semantic_mask_frames": 1,
        },
    )
    unproven = M.build_result(
        **base,
        segmentation_summary={
            "labeled_prim_count": 2,
            "instance_mask_frames": 0,
            "semantic_mask_frames": 1,
        },
    )

    assert "segmentation_pass" not in no_seg
    assert proven["segmentation_pass"]["simulator_backend"] == "isaac_replicator"
    assert proven["segmentation_pass"]["native_segmentation_proven"] is True
    assert unproven["segmentation_pass"]["native_segmentation_proven"] is False


def test_segmentation_pass_field_is_not_a_mujoco_marker() -> None:
    source = _RUNNER.read_text()
    start = source.index('"segmentation_pass"')
    segment = source[start : source.index("return result", start)]

    assert '"simulator_backend": "isaac_replicator"' in segment
    assert "replicator_instance_semantic_annotator" in segment
    assert '"simulator_backend": "mujoco"' not in segment


def test_arm_reach_rotation_swings_rest_bone_toward_target() -> None:
    # shoulder at origin; rest upper arm hangs down (-z); faucet is forward (+y), level
    shoulder = (0.0, 0.0, 0.0)
    rest_elbow = (0.0, 0.0, -0.3)          # arm hanging straight down
    target = (0.0, 0.6, 0.0)                # reach forward (+y)
    axis, angle = M.arm_reach_rotation(shoulder, rest_elbow, target, 1.0)
    # rest dir = -z, want dir = +y -> 90 deg; axis perpendicular to both (= +/-x)
    assert angle == pytest.approx(math.pi / 2, abs=1e-6)
    assert abs(axis[0]) == pytest.approx(1.0, abs=1e-6) and abs(axis[1]) < 1e-9 and abs(axis[2]) < 1e-9
    # reach_frac scales the swing
    _, half = M.arm_reach_rotation(shoulder, rest_elbow, target, 0.5)
    assert half == pytest.approx(math.pi / 4, abs=1e-6)
    _, zero = M.arm_reach_rotation(shoulder, rest_elbow, target, 0.0)
    assert zero == pytest.approx(0.0, abs=1e-9)
    assert math.isclose(math.sqrt(sum(c * c for c in axis)), 1.0, abs_tol=1e-9)  # unit axis


def test_parse_scenarios_normalizes_to_pelvis_height_route() -> None:
    req = {"scenarios": [
        {"scenario_id": "s1", "spawn_position_xyz": [-4.25, -3.35, 0.05],
         "target_position_xyz": [1.75, 1.25, 0.05], "description": "to sink",
         "target_object_id": "sink", "affordance_object_ids": ["handle", "lever"]},
        {"id": "s2", "route_points": [[0, 0, 0.1], [1, 1, 0.1], [2, 2, 0.1]]},
        {
            "scenario_id": "task_only",
            "description": "open the service door",
            "target_object_ids": ["door_handle", "door"],
            "affordance_object_ids": ["door_handle"],
        },
        {"scenario_id": "bad"},  # no start/target -> skipped
    ]}
    sc = M.parse_scenarios(req)
    assert [s["scenario_id"] for s in sc] == ["s1", "s2", "task_only"]
    # navigation route lifted to pelvis height
    assert all(p[2] == M.ROBOT_PELVIS_HEIGHT_M for p in sc[0]["route_points"])
    assert sc[0]["start"][2] == M.ROBOT_PELVIS_HEIGHT_M
    assert len(sc[1]["route_points"]) == 3
    assert sc[0]["raw_target_position_xyz"] == [1.75, 1.25, 0.05]
    assert sc[0]["target_object_id"] == "sink"
    assert sc[0]["affordance_object_ids"] == ["handle", "lever"]
    assert sc[2]["task_target_deferred"] is True
    assert sc[2]["route_points"] == []
    assert sc[2]["instruction"] == "open the service door"
    assert sc[2]["target_object_ids"] == ["door_handle", "door"]
    assert sc[2]["affordance_object_ids"] == ["door_handle"]


def test_deferred_task_route_materializes_from_dynamic_stance() -> None:
    scenario = {
        "scenario_id": "task_only",
        "instruction": "open the refrigerator",
        "route_points": [],
        "task_target_deferred": True,
    }
    M._materialize_deferred_task_route(
        scenario,
        stance_plan={"task_target_xyz": [1.5, 2.5, 1.1]},
        root_pose=(1.0, 2.0, 0.84),
        look_at=(1.5, 2.5, 1.1),
    )

    assert scenario["task_target_deferred"] is False
    assert scenario["deferred_task_resolution"] == "materialized_from_task_stance_plan"
    assert scenario["start"] == [1.0, 2.0, 0.84]
    assert scenario["target"] == [1.5, 2.5, M.ROBOT_PELVIS_HEIGHT_M]
    assert scenario["route_points"] == [scenario["start"], scenario["target"]]
    assert scenario["raw_target_position_xyz"] == [1.5, 2.5, 1.1]


def test_task_stance_planner_uses_target_as_thing_to_face_not_pelvis() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [1.0],
        "floor_z_hint": 0.05,
    }
    plan = M.plan_task_stance(scenario=scenario, probe_collision=lambda pose, yaw: 0)
    assert plan["status"] == "accepted"
    assert plan["accepted_pose"][:2] == [-1.0, 0.0]
    assert plan["accepted_pose"][2] == pytest.approx(M.ROBOT_PELVIS_HEIGHT_M + 0.05)
    assert plan["accepted_pose"][:2] != plan["task_target_xyz"][:2]
    assert plan["accepted_yaw"] == pytest.approx(0.0)
    assert plan["candidates"][0]["scene_collision_contact_count"] == 0


def test_task_stance_planner_samples_around_target_until_collision_free() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [1.0],
    }

    def probe(pose, yaw):
        return 1 if pose[0] < -0.99 and abs(pose[1]) < 1e-6 else 0

    plan = M.plan_task_stance(scenario=scenario, probe_collision=probe)
    assert plan["status"] == "accepted"
    assert plan["selected_candidate_index"] == 1
    assert plan["candidates"][0]["scene_collision_contact_count"] == 1
    assert plan["candidates"][1]["scene_collision_contact_count"] == 0
    assert math.dist(plan["accepted_pose"][:2], plan["task_target_xyz"][:2]) == pytest.approx(1.0)


def test_task_stance_planner_prefers_reachable_affordance_pose() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "task_affordance_xyz": [0.0, -0.1, 1.0],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [0.4],
        "floor_z_hint": 0.05,
    }

    plan = M.plan_task_stance(scenario=scenario, probe_collision=lambda pose, yaw: 0)

    assert plan["status"] == "accepted"
    assert plan["reachability_selection_enabled"] is True
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["reachability_estimate"]["status"] == "PASS"
    assert chosen["reachability_estimate"]["best_reach_arm"] in {"left", "right"}
    chosen_best_shoulder = chosen["reachability_estimate"]["best_reach_arm_estimate"][
        "shoulder_to_affordance_m"
    ]
    all_best_shoulders = [
        c["reachability_estimate"]["best_reach_arm_estimate"]["shoulder_to_affordance_m"]
        for c in plan["candidates"]
    ]
    assert chosen_best_shoulder == min(all_best_shoulders)
    assert plan["accepted_pose"][1] < 0.0
    assert plan["task_affordance_xyz"] == [0.0, -0.1, 1.0]
    assert plan["stance_focus_source"] == "task_affordance_xyz"
    assert plan["stance_focus_xyz"] == [0.0, -0.1, 1.0]
    assert plan["accepted_pose"][1] == pytest.approx(scenario["task_affordance_xyz"][1])


def test_task_stance_planner_samples_offset_handle_not_fixture_centroid() -> None:
    scenario = {
        "task_target_position_xyz": [2.277888, 1.333059, 0.848527],
        "task_affordance_xyz": [2.489866, 1.069795, 0.886474],
        "robot_start_position_xyz": [1.609779, 0.947326, 0.84],
        "target_object_bbox_min_xyz": [2.066149, 0.683999, 0.047727],
        "target_object_bbox_max_xyz": [2.489866, 1.982118, 1.649326],
        "stance_distance_candidates_m": [0.30, 0.32, 0.38],
        "floor_z_hint": 0.05,
    }

    plan = M.plan_task_stance(scenario=scenario, probe_collision=lambda pose, yaw: 0)

    assert plan["status"] == "accepted"
    assert plan["stance_focus_source"] == "task_affordance_xyz"
    assert plan["stance_focus_xyz"] == [2.489866, 1.069795, 0.886474]
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["stance_focus_source"] == "task_affordance_xyz"
    assert math.dist(plan["accepted_pose"][:2], scenario["task_affordance_xyz"][:2]) < math.dist(
        [1.609779, 0.947326],
        scenario["task_affordance_xyz"][:2],
    )
    assert chosen["reachability_estimate"]["nearest_seed_effector_to_affordance_m"] < 0.45


def test_task_stance_reach_uses_effector_neighborhood_not_tiny_shoulder_cutoff() -> None:
    reach = M._task_stance_reachability_estimate(
        (1.769021, 1.069795, 0.84),
        0.0,
        (2.489866, 1.069795, 0.886474),
    )

    assert reach is not None
    assert reach["status"] == "PASS"
    assert reach["required_max_shoulder_to_affordance_m"] == pytest.approx(
        M.G1_APPROX_ARM_SPAN_M + M.MANIPULATION_SEED_MAX_EFFECTOR_TO_AFFORDANCE_M
    )
    assert reach["nearest_seed_effector_to_affordance_m"] == pytest.approx(0.3572)
    assert "manipulation_pov_affordance_outside_g1_reach_envelope" not in reach["blockers"]
    assert M._seed_reach_blockers(
        shoulder_to_affordance_m=0.7946,
        effector_to_affordance_m=0.3834,
        shoulder_margin_m=M.MANIPULATION_RENDERED_SEED_SHOULDER_MARGIN_M,
        effector_margin_m=M.MANIPULATION_RENDERED_SEED_EFFECTOR_MARGIN_M,
    ) == []
    assert M._seed_reach_blockers(
        shoulder_to_affordance_m=0.7946,
        effector_to_affordance_m=0.3834,
        shoulder_margin_m=M.MANIPULATION_RENDERED_SEED_SHOULDER_MARGIN_M,
        effector_margin_m=0.0,
    ) == ["manipulation_pov_effector_too_far_from_affordance"]


def test_task_stance_recomputes_reach_after_placement_root_correction() -> None:
    scenario = {
        "task_target_position_xyz": [2.277888, 1.333059, 0.848527],
        "task_affordance_xyz": [2.489866, 1.069795, 0.886474],
        "target_object_bbox_min_xyz": [2.009021, 0.89582, 0.555082],
        "target_object_bbox_max_xyz": [2.546755, 1.770299, 1.141971],
        "stance_distance_candidates_m": [0.24],
        "floor_z_hint": 0.05,
    }

    def probe(_pose, yaw):
        return 0 if abs(float(yaw)) < 1e-6 else 1

    def placement_validator(_pose, _yaw, _record):
        return {
            "status": "accepted",
            "blockers": [],
            "place_root_diagnostics": {
                "corrected_root_translation_xyz": [1.600195, 1.06979, 0.84],
            },
        }

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=probe,
        placement_validator=placement_validator,
    )

    assert plan["status"] == "blocked"
    assert plan["blockers"] == ["no_reach_seed_task_stance_candidate"]
    candidate = plan["candidates"][0]
    assert candidate["pre_placement_reachability_estimate"]["status"] == "PASS"
    assert candidate["reachability_estimate"]["status"] == "FAIL"
    assert candidate["reachability_estimate"]["pose_source"] == "placement_corrected_root_translation_xyz"
    assert candidate["placement_corrected_root_pose"] == [1.600195, 1.06979, 0.84]
    assert "manipulation_pov_effector_too_far_from_affordance" in candidate[
        "reachability_estimate"
    ]["blockers"]


def test_handleless_top_cabinet_derives_scoped_lower_front_edge_affordance() -> None:
    target_resolution = {
        "status": "resolved",
        "selected": {
            "target_object_id": "topcabinet",
            "prim_path": "/root/Kitchen_TopCabinet_01",
            "path_depth": 2,
            "center_xyz": [0.63, 2.26, 1.94],
            "bbox_min_xyz": [-1.29, 2.06, 1.50],
            "bbox_max_xyz": [2.56, 2.47, 2.38],
        },
    }
    blocked_affordance = {
        "status": "blocked",
        "blockers": ["affordance_not_scoped_to_target_fixture"],
    }

    derived = M._derive_handleless_upper_cabinet_affordance_resolution(
        target_resolution=target_resolution,
        affordance_resolution=blocked_affordance,
        scenario={"task_id": "top_cabinet"},
    )

    assert derived is not None
    assert derived["status"] == "resolved"
    assert derived["source"] == "usd_target_bounds_derived_affordance"
    selected = derived["selected"]
    assert selected["derived_affordance"] is True
    assert selected["target_object_id"] == "derived_upper_cabinet_lower_front_edge"
    assert selected["center_xyz"][0] == pytest.approx(0.63)
    assert selected["center_xyz"][1] == pytest.approx(2.06)
    assert selected["center_xyz"][2] == pytest.approx(1.6056)
    assert "not a detected handle" in selected["claim_boundary"].lower()


def test_robot_proxy_clean_label_is_excluded_from_scene_obstacles() -> None:
    class Obj:
        id = "g"
        label = "g"

    assert M._is_robot_scene_object(Obj()) is True


def test_task_stance_planner_uses_calibrated_g1_shoulder_geometry_for_close_appliance() -> None:
    scenario = {
        "task_target_position_xyz": [0.558827, 2.161934, 0.809859],
        "task_affordance_xyz": [0.566285, 1.922063, 0.807764],
        "robot_start_position_xyz": [0.558827, 1.505526, 0.84],
        "target_object_bbox_min_xyz": [0.140191, 1.885526, 0.787217],
        "target_object_bbox_max_xyz": [0.977464, 2.438342, 0.832501],
        "stance_distance_candidates_m": [0.30, 0.38],
        "floor_z_hint": 0.05,
    }

    plan = M.plan_task_stance(scenario=scenario, probe_collision=lambda pose, yaw: 0)

    assert plan["status"] == "accepted"
    assert plan["reachability_selection_enabled"] is True
    selected = plan["candidates"][plan["selected_candidate_index"]]
    farther = next(
        c for c in plan["candidates"]
        if c["standoff_from_target_surface_m"] == pytest.approx(0.38)
        and c["angle_offset_deg"] == 0
    )
    assert selected["standoff_from_target_surface_m"] == pytest.approx(0.30)
    assert selected["reachability_estimate"]["status"] == "PASS"
    assert farther["reachability_estimate"]["status"] == "PASS"


def test_task_stance_planner_blocks_when_affordance_never_reachable() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "task_affordance_xyz": [0.0, -0.1, 2.2],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "target_object_bbox_min_xyz": [-0.1, -0.1, 0.8],
        "target_object_bbox_max_xyz": [0.1, 0.1, 1.0],
        "stance_distance_candidates_m": [0.30, 0.38],
        "floor_z_hint": 0.05,
    }

    plan = M.plan_task_stance(scenario=scenario, probe_collision=lambda pose, yaw: 0)

    assert plan["status"] == "blocked"
    assert plan["blockers"] == ["no_reach_seed_task_stance_candidate"]
    assert plan["reachability_selection_enabled"] is True
    assert plan["reachability_rejected_candidate_count"] > 0
    assert all(
        candidate["reachability_estimate"]["status"] == "FAIL"
        for candidate in plan["candidates"]
    )


def test_task_stance_planner_offsets_from_target_footprint_surface() -> None:
    scenario = {
        "task_target_position_xyz": [2.0, 2.0, 0.9],
        "robot_start_position_xyz": [2.0, 0.0, 0.05],
        "target_object_bbox_min_xyz": [1.5, 1.65, 0.75],
        "target_object_bbox_max_xyz": [2.5, 2.35, 1.15],
        "stance_distance_candidates_m": [0.85],
        "floor_z_hint": 0.05,
    }

    plan = M.plan_task_stance(scenario=scenario, probe_collision=lambda pose, yaw: 0)

    assert plan["status"] == "accepted"
    # The robot stands on the approach-side aisle. Without the footprint-surface offset, this
    # would be y=1.15 (0.85m from the center); with the sink/counter half-depth included, it moves
    # to y=0.80, clear of the target footprint.
    assert plan["accepted_pose"][:2] == [2.0, 0.8]
    assert plan["accepted_yaw"] == pytest.approx(math.pi / 2)
    first = plan["candidates"][0]
    assert first["standoff_from_target_surface_m"] == pytest.approx(0.85)
    assert first["target_surface_offset_m"] == pytest.approx(0.35)
    assert first["distance_to_target_m"] == pytest.approx(1.2)
    assert plan["task_target_bounds"]["bbox_min_xyz"] == [1.5, 1.65, 0.75]


def test_open_articulated_target_uses_close_surface_standoff_from_bounds() -> None:
    scenario = {
        "instruction": "open the refrigerator",
        "target_object_id": "refrigerator",
        "target_object_label": "refrigerator door",
        "target_object_position_xyz": [-1.979559, 0.655166, 1.025963],
        "robot_start_position_xyz": [-0.6, 0.66, 0.05],
        "target_object_bbox_min_xyz": [-2.521971, 0.13306, -0.000508],
        "target_object_bbox_max_xyz": [-1.437147, 1.177273, 2.052435],
        "floor_z_hint": 0.05,
    }

    distances = M.task_stance_distance_candidates(scenario)
    assert distances[0] == pytest.approx(0.35)
    assert M._validation_standoff_range_for_scenario(scenario) == pytest.approx(
        M.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M
    )

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=lambda _pose, _yaw, record: (
            {"status": "accepted", "blockers": []}
            if record["angle_offset_deg"] == 0
            and record["standoff_from_target_surface_m"] == pytest.approx(0.35)
            else {"status": "blocked", "blockers": ["synthetic_reach_profile_reject"]}
        ),
    )

    assert plan["status"] == "accepted"
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["angle_offset_deg"] == 0
    assert chosen["standoff_from_target_surface_m"] == pytest.approx(0.35)
    assert plan["accepted_pose"][0] == pytest.approx(-1.085326, abs=0.002)
    assert plan["accepted_pose"][1] == pytest.approx(0.658299, abs=0.002)
    assert abs(abs(plan["accepted_yaw"]) - math.pi) < 0.01


def test_non_articulated_target_keeps_default_standoff_profile() -> None:
    scenario = {
        "instruction": "stand near the counter",
        "target_object_id": "counter",
        "target_object_label": "counter",
    }

    assert M.task_stance_distance_candidates(scenario)[0] == pytest.approx(
        M.TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M
    )
    assert M._validation_standoff_range_for_scenario(scenario) == pytest.approx(
        M.TASK_STANCE_DEFAULT_VALIDATION_STANDOFF_RANGE_M
    )


def test_faucet_and_sink_targets_use_close_reach_profile() -> None:
    scenario = {
        "instruction": "turn on the faucet",
        "target_object_id": "sink",
        "target_object_label": "sink",
    }

    assert M.task_stance_distance_candidates(scenario)[0] < M.TASK_STANCE_DEFAULT_SURFACE_STANDOFF_M
    assert M._validation_standoff_range_for_scenario(scenario) == pytest.approx(
        M.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M
    )


def test_task_stance_default_distances_include_counter_clearance_band() -> None:
    scenario = {
        "task_target_position_xyz": [2.277888, 1.333059, 0.848527],
        "robot_start_position_xyz": [2.35, 0.1, 0.05],
        "target_object_bbox_min_xyz": [2.009021, 0.89582, 0.555082],
        "target_object_bbox_max_xyz": [2.546755, 1.770299, 1.141971],
        "floor_z_hint": 0.05,
    }

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=lambda _pose, _yaw, record: (
            {"status": "accepted", "blockers": []}
            if record["angle_offset_deg"] == 0
            and abs(float(record["standoff_from_target_surface_m"]) - 1.4025) < 1e-9
            else {"status": "blocked", "blockers": ["synthetic_clearance_or_reach_failure"]}
        ),
    )

    assert plan["status"] == "accepted"
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["angle_offset_deg"] == 0
    assert chosen["standoff_from_target_surface_m"] == pytest.approx(1.4025)
    assert plan["accepted_pose"][0] == pytest.approx(2.386169, abs=1e-6)
    assert plan["accepted_pose"][1] == pytest.approx(-0.518468, abs=1e-6)


def test_task_stance_planner_tries_near_angled_aisle_before_farther_straight_back() -> None:
    scenario = {
        "task_target_position_xyz": [2.277888, 1.333059, 0.848527],
        "robot_start_position_xyz": [2.35, 0.1, 0.05],
        "target_object_bbox_min_xyz": [2.009021, 0.89582, 0.555082],
        "target_object_bbox_max_xyz": [2.546755, 1.770299, 1.141971],
        "floor_z_hint": 0.05,
    }

    def validator(_pose, _yaw, record):
        # The straight-back closest candidate clips. The planner should try a nearby angled stance
        # at the same standoff before walking farther backward along the wall-side ray.
        if (
            record["angle_offset_deg"] == 0
            and abs(float(record["standoff_from_target_surface_m"]) - 0.85) < 1e-9
        ):
            return {"status": "blocked", "blockers": ["synthetic_wall_side_clearance_failure"]}
        return {"status": "accepted", "blockers": []}

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=validator,
    )

    assert plan["status"] == "accepted"
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["standoff_from_target_surface_m"] == pytest.approx(0.85)
    assert chosen["angle_offset_deg"] == -15
    assert plan["accepted_pose"][0] == pytest.approx(2.009, abs=0.002)
    assert plan["accepted_pose"][1] == pytest.approx(0.028, abs=0.002)


def test_task_stance_planner_prefers_approach_ray_over_backside_candidate() -> None:
    scenario = {
        "task_target_position_xyz": [2.277888, 1.333059, 0.848527],
        "robot_start_position_xyz": [2.35, 0.1, 0.05],
        "target_object_bbox_min_xyz": [2.009021, 0.89582, 0.555082],
        "target_object_bbox_max_xyz": [2.546755, 1.770299, 1.141971],
        "floor_z_hint": 0.05,
    }

    def validator(_pose, _yaw, record):
        # The closer 180-degree candidate is on the backside of the target. A farther point on the
        # approach ray is the room-side stance and must win once both validate geometrically.
        accepts = (
            record["angle_offset_deg"] == 180
            and abs(float(record["standoff_from_target_surface_m"]) - 1.0625) < 1e-9
        ) or (
            record["angle_offset_deg"] == 0
            and abs(float(record["standoff_from_target_surface_m"]) - 1.4025) < 1e-9
        )
        return (
            {"status": "accepted", "blockers": []}
            if accepts
            else {"status": "blocked", "blockers": ["synthetic_geometry_reject"]}
        )

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=validator,
    )

    assert plan["status"] == "accepted"
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["angle_offset_deg"] == 0
    assert chosen["standoff_from_target_surface_m"] == pytest.approx(1.4025)
    assert plan["accepted_pose"][0] == pytest.approx(2.386169, abs=1e-6)
    assert plan["accepted_pose"][1] == pytest.approx(-0.518468, abs=1e-6)
    assert plan["accepted_candidate_count"] == 2


def test_task_stance_planner_without_approach_hint_prefers_nearest_validated_face() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 1.0],
        "target_object_bbox_min_xyz": [-0.5, -0.5, 0.0],
        "target_object_bbox_max_xyz": [0.5, 0.5, 2.0],
        "floor_z_hint": 0.05,
        "stance_distance_candidates_m": [0.4, 1.13],
    }

    def validator(_pose, _yaw, record):
        accepts = (
            record["angle_offset_deg"] == 180
            and abs(float(record["standoff_from_target_surface_m"]) - 0.4) < 1e-9
        ) or (
            record["angle_offset_deg"] == 90
            and abs(float(record["standoff_from_target_surface_m"]) - 1.13) < 1e-9
        )
        return (
            {"status": "accepted", "blockers": []}
            if accepts
            else {"status": "blocked", "blockers": ["synthetic_geometry_reject"]}
        )

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=validator,
    )

    assert plan["status"] == "accepted"
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["angle_offset_deg"] == 180
    assert chosen["standoff_from_target_surface_m"] == pytest.approx(0.4)
    assert chosen["approach_bias_enabled"] is False
    assert plan["accepted_candidate_count"] == 2


def test_xy_rect_overlap_and_gap_reports_overlap_and_clearance() -> None:
    overlapping = M._xy_rect_overlap_and_gap(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.25, 0.0],
        [1.5, 0.75, 1.0],
    )
    assert overlapping["overlaps_xy"] is True
    assert overlapping["overlap_area_xy_m2"] == pytest.approx(0.25)
    assert overlapping["gap_m"] == pytest.approx(0.0)

    separated = M._xy_rect_overlap_and_gap(
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.3, 1.4, 0.0],
        [2.0, 2.0, 1.0],
    )
    assert separated["overlaps_xy"] is False
    assert separated["overlap_area_xy_m2"] == pytest.approx(0.0)
    assert separated["gap_m"] == pytest.approx(0.5)


def test_task_stance_planner_rejects_candidate_failed_by_placement_validation() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "target_object_bbox_min_xyz": [-0.25, -0.25, 0.6],
        "target_object_bbox_max_xyz": [0.25, 0.25, 1.2],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [1.0],
    }

    def validator(_pose, _yaw, record):
        if record["angle_offset_deg"] == 0:
            return {"status": "blocked", "blockers": ["placed_robot_bbox_overlaps_target_bbox"]}
        return {"status": "accepted", "blockers": []}

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=validator,
    )

    assert plan["status"] == "accepted"
    assert plan["selected_candidate_index"] == 1
    assert plan["candidates"][0]["placement_validation"]["blockers"] == [
        "placed_robot_bbox_overlaps_target_bbox"
    ]
    assert plan["placement_validation"]["status"] == "accepted"


def test_task_stance_planner_fails_when_all_collision_free_candidates_fail_validation() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [1.0],
    }

    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=lambda pose, yaw, record: {
            "status": "blocked",
            "blockers": ["placed_robot_bbox_center_far_from_root_pose"],
        },
    )

    assert plan["status"] == "blocked"
    assert plan["blockers"] == ["no_validated_task_stance_candidate"]
    assert plan["placement_validation_rejected_candidate_count"] == len(plan["candidates"])
    assert all("placement_validation" in candidate for candidate in plan["candidates"])


def test_stage_placement_validator_blocks_actual_robot_bbox_overlap(monkeypatch) -> None:
    placed = []
    monkeypatch.setattr(
        M,
        "_place_root",
        lambda stage, prim_path, pose, yaw: placed.append((stage, prim_path, pose, yaw)),
    )
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [1.1, -0.2, 0.0],
            "bbox_max_xyz": [1.4, 0.2, 1.7],
            "center_xyz": [1.25, 0.0, 0.85],
            "size_xyz": [0.3, 0.4, 1.7],
        },
    )
    validator = M._placement_validator_for_stage(
        object(),
        "/World/G1",
        ((1.0, -0.5, 0.5), (1.5, 0.5, 1.5)),
    )

    result = validator((0.0, 0.0, 0.84), 0.0, {"standoff_from_target_surface_m": 0.85})

    assert placed
    assert result["status"] == "blocked"
    assert "placed_robot_bbox_overlaps_target_bbox" in result["blockers"]
    assert "placed_robot_bbox_center_far_from_root_pose" in result["blockers"]


def test_stage_placement_validator_accepts_clear_actual_robot_bbox(monkeypatch) -> None:
    monkeypatch.setattr(M, "_place_root", lambda _stage, _prim_path, _pose, _yaw: None)
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [-0.3, -0.3, 0.0],
            "bbox_max_xyz": [0.3, 0.3, 1.7],
            "center_xyz": [0.0, 0.0, 0.85],
            "size_xyz": [0.6, 0.6, 1.7],
        },
    )
    validator = M._placement_validator_for_stage(
        object(),
        "/World/G1",
        ((1.0, -0.5, 0.5), (1.5, 0.5, 1.5)),
    )

    result = validator((0.0, 0.0, 0.84), 0.0, {"standoff_from_target_surface_m": 0.85})

    assert result["status"] == "accepted"
    assert result["blockers"] == []
    assert result["target_bbox_relation"]["gap_m"] == pytest.approx(0.7)
    assert result["required_target_gap_m"] == pytest.approx(0.35)


def test_stage_placement_validator_accepts_close_reach_gap_when_task_range_allows(monkeypatch) -> None:
    pose = (-1.037152, 0.65847, 0.84)
    target = SceneObject(
        id="refrigerator",
        label="refrigerator door",
        bbox_min=(-2.521971, 0.13306, -0.000508),
        bbox_max=(-1.437147, 1.177273, 2.052435),
        centroid=(-1.979559, 0.655166, 1.025963),
    )
    monkeypatch.setattr(M, "_place_root", lambda _stage, _prim_path, _pose, _yaw: None)
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [pose[0] - 0.28, pose[1] - 0.28, 0.22],
            "bbox_max_xyz": [pose[0] + 0.28, pose[1] + 0.28, 1.46],
            "center_xyz": [pose[0], pose[1], pose[2]],
            "size_xyz": [0.56, 0.56, 1.24],
        },
    )
    validator = M._placement_validator_for_stage(
        object(),
        "/World/G1",
        (target.bbox_min, target.bbox_max),
        target_object=target,
        scene_objects=[],
        floor_z=0.05,
        standoff_range=M.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M,
    )

    result = validator(pose, -3.138083, {"standoff_from_target_surface_m": 0.4})

    assert result["status"] == "accepted"
    assert result["blockers"] == []
    assert result["target_bbox_relation"]["gap_m"] == pytest.approx(0.12, abs=0.002)
    assert result["deterministic_geometry"]["standoff_m"] == pytest.approx(0.279, abs=0.002)
    assert result["validation_standoff_range_m"] == pytest.approx(
        list(M.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M)
    )


def test_stage_placement_validator_allows_visual_bbox_overlap_when_floor_footprint_clear(
    monkeypatch,
) -> None:
    pose = (0.75, 0.0, 0.79)
    target = SceneObject(
        id="sink",
        label="sink counter",
        bbox_min=(1.0, -0.2, 0.8),
        bbox_max=(1.4, 0.2, 1.2),
        centroid=(1.2, 0.0, 1.0),
    )
    monkeypatch.setattr(M, "_place_root", lambda _stage, _prim_path, _pose, _yaw: None)
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [0.45, -0.15, 0.0],
            "bbox_max_xyz": [1.05, 0.15, 1.5],
            "center_xyz": [0.75, 0.0, 0.75],
            "size_xyz": [0.6, 0.3, 1.5],
        },
    )
    validator = M._placement_validator_for_stage(
        object(),
        "/World/G1",
        (target.bbox_min, target.bbox_max),
        target_object=target,
        scene_objects=[target],
        floor_z=0.0,
        standoff_range=M.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M,
    )

    result = validator(
        pose,
        0.0,
        {"standoff_from_target_surface_m": 0.13, "scene_collision_contact_count": 0},
    )

    assert result["status"] == "accepted"
    assert result["blockers"] == []
    assert result["target_bbox_relation"]["overlaps_xy"] is True
    assert result["target_bbox_relation"]["hard_blocker"] is False


def test_stage_placement_validator_suppresses_broad_aabb_clip_only_after_zero_physx_contact(
    monkeypatch,
) -> None:
    pose = (-1.037152, 0.65847, 0.84)
    target = SceneObject(
        id="articulated_target",
        label="openable appliance door",
        bbox_min=(-2.521971, 0.13306, -0.000508),
        bbox_max=(-1.437147, 1.177273, 2.052435),
        centroid=(-1.979559, 0.655166, 1.025963),
    )
    broad_false_positive = SceneObject(
        id="broad_asset_leaf",
        label="asset_leaf",
        bbox_min=(-1.30, -0.15, 0.0),
        bbox_max=(2.57, 2.46, 0.84),
        centroid=(0.635, 1.155, 0.42),
        source="usd_leaf",
    )
    monkeypatch.setattr(M, "_place_root", lambda _stage, _prim_path, _pose, _yaw: None)
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [pose[0] - 0.28, pose[1] - 0.28, 0.22],
            "bbox_max_xyz": [pose[0] + 0.28, pose[1] + 0.28, 1.46],
            "center_xyz": [pose[0], pose[1], pose[2]],
            "size_xyz": [0.56, 0.56, 1.24],
        },
    )
    validator = M._placement_validator_for_stage(
        object(),
        "/World/G1",
        (target.bbox_min, target.bbox_max),
        target_object=target,
        scene_objects=[broad_false_positive],
        floor_z=0.05,
        standoff_range=M.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M,
    )

    accepted = validator(pose, -3.138083, {"standoff_from_target_surface_m": 0.4, "scene_collision_contact_count": 0})
    blocked = validator(pose, -3.138083, {"standoff_from_target_surface_m": 0.4, "scene_collision_contact_count": 1})

    assert accepted["status"] == "accepted"
    assert accepted["deterministic_geometry"]["ok"] is True
    assert accepted["deterministic_geometry_raw"]["ok"] is False
    assert accepted["deterministic_geometry_adjustments"]["suppressed_broad_aabb_clips"][0]["object_id"] == "broad_asset_leaf"
    assert blocked["status"] == "blocked"
    assert "placement_geometry_invalid" in blocked["blockers"]


def test_placement_manifest_preserves_broad_aabb_adjustment(monkeypatch) -> None:
    pose = (-1.037152, 0.65847, 0.84)
    stance_plan = {
        "floor_z_hint": 0.05,
        "accepted_pose": list(pose),
        "accepted_yaw": -3.138083,
        "selected_candidate_index": 0,
        "candidates": [
            {
                "standoff_from_target_surface_m": 0.4,
                "scene_collision_contact_count": 0,
            }
        ],
        "task_target_xyz": [-1.979559, 0.655166, 1.025963],
        "task_target_bounds": {
            "bbox_min_xyz": [-2.521971, 0.13306, -0.000508],
            "bbox_max_xyz": [-1.437147, 1.177273, 2.052435],
        },
        "target_resolution": {
            "selected": {
                "target_object_id": "articulated_target",
                "target_object_label": "openable appliance door",
            }
        },
        "placement_validation": {
            "validation_standoff_range_m": list(M.TASK_STANCE_CLOSE_REACH_GAP_RANGE_M)
        },
    }
    broad_false_positive = SceneObject(
        id="broad_asset_leaf",
        label="asset_leaf",
        bbox_min=(-1.30, -0.15, 0.0),
        bbox_max=(2.57, 2.46, 0.84),
        centroid=(0.635, 1.155, 0.42),
        source="usd_leaf",
    )
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [pose[0] - 0.28, pose[1] - 0.28, 0.22],
            "bbox_max_xyz": [pose[0] + 0.28, pose[1] + 0.28, 1.46],
            "center_xyz": [pose[0], pose[1], pose[2]],
            "size_xyz": [0.56, 0.56, 1.24],
        },
    )

    manifest = M._build_placement_validation_manifest(
        stage=object(),
        robot_prim_path="/World/G1",
        stance_plan=stance_plan,
        accepted_pose=pose,
        accepted_yaw=-3.138083,
        root_diagnostics={"status": "corrected"},
        scene_objects=[broad_false_positive],
        scenario_id="open_articulated_target",
        visual_qc={"status": "passed"},
    )

    assert manifest["status"] == "PASS"
    assert manifest["blockers"] == []
    assert manifest["intended_geometry"]["ok"] is True
    assert manifest["intended_geometry"]["raw_geometry"]["ok"] is False
    assert manifest["intended_geometry"]["adjustments"]["suppressed_broad_aabb_clips"][0]["object_id"] == "broad_asset_leaf"


def test_place_root_corrects_measured_world_footprint_offset(monkeypatch) -> None:
    placements = []
    current = {"pose": (0.0, 0.0, 0.0)}
    local_offset = (0.35, -0.2)

    def fake_set_root(_stage, _prim_path, pose, _yaw):
        current["pose"] = tuple(float(v) for v in pose)
        placements.append(current["pose"])

    def fake_bbox(_stage, _prim_path):
        pose = current["pose"]
        center = (pose[0] + local_offset[0], pose[1] + local_offset[1])
        return {
            "bbox_min_xyz": [center[0] - 0.25, center[1] - 0.25, 0.0],
            "bbox_max_xyz": [center[0] + 0.25, center[1] + 0.25, 1.6],
            "center_xyz": [center[0], center[1], 0.8],
            "size_xyz": [0.5, 0.5, 1.6],
        }

    monkeypatch.setattr(M, "_set_root_xform", fake_set_root)
    monkeypatch.setattr(M, "_world_bbox_for_prim", fake_bbox)
    monkeypatch.setattr(M, "_root_transform_diagnostics", lambda _stage, _prim_path: {"root": "diag"})

    diag = M._place_root(object(), "/World/G1", (2.0, 0.8, 0.84), 1.57)

    assert diag["status"] == "corrected"
    assert diag["correction_applied"] is True
    assert diag["measured_offset_xy_m"] == [0.35, -0.2]
    assert diag["final_footprint_center_xy"] == [2.0, 0.8]
    assert diag["final_xy_error_m"] == pytest.approx(0.0)
    assert placements == [(2.0, 0.8, 0.84), (1.65, 1.0, 0.84)]


def test_placement_validation_manifest_passes_with_actual_bbox_center_match(monkeypatch) -> None:
    sp = importlib.import_module("blueprint_pipeline.scene_placement")
    target = sp.SceneObject(
        id="sink",
        label="sink",
        bbox_min=(1.5, 1.65, 0.75),
        bbox_max=(2.5, 2.35, 1.15),
        centroid=(2.0, 2.0, 0.95),
        source="test",
    )
    stance_plan = {
        "status": "accepted",
        "accepted_pose": [2.0, 0.8, 0.84],
        "accepted_yaw": math.pi / 2,
        "floor_z_hint": 0.05,
        "task_target_xyz": [2.0, 2.0, 0.95],
        "task_target_bounds": {
            "bbox_min_xyz": [1.5, 1.65, 0.75],
            "bbox_max_xyz": [2.5, 2.35, 1.15],
        },
    }
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [1.72, 0.52, 0.0],
            "bbox_max_xyz": [2.28, 1.08, 1.6],
            "center_xyz": [2.0, 0.8, 0.8],
            "size_xyz": [0.56, 0.56, 1.6],
        },
    )

    manifest = M._build_placement_validation_manifest(
        stage=object(),
        robot_prim_path="/World/G1",
        stance_plan=stance_plan,
        accepted_pose=(2.0, 0.8, 0.84),
        accepted_yaw=math.pi / 2,
        root_diagnostics={"status": "placed"},
        scene_objects=[target],
        scenario_id="sink_stance",
        visual_qc={"status": "passed", "blockers": []},
        topdown_frame="/tmp/placement_topdown_0000.png",
    )

    assert manifest["status"] == "PASS"
    assert manifest["blockers"] == []
    assert manifest["ground_truth_placement"]["xy_error_m"] == pytest.approx(0.0)
    assert manifest["intended_geometry"]["ok"] is True
    assert manifest["topdown_debug_frame"].endswith("placement_topdown_0000.png")


def test_placement_validation_treats_unparsed_visual_qc_as_nonblocking_evidence(monkeypatch) -> None:
    stance_plan = {
        "status": "accepted",
        "accepted_pose": [2.0, 0.8, 0.84],
        "accepted_yaw": math.pi / 2,
        "floor_z_hint": 0.05,
        "task_target_xyz": [2.0, 2.0, 0.95],
        "task_target_bounds": {
            "bbox_min_xyz": [1.5, 1.65, 0.75],
            "bbox_max_xyz": [2.5, 2.35, 1.15],
        },
    }
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [1.72, 0.52, 0.0],
            "bbox_max_xyz": [2.28, 1.08, 1.6],
            "center_xyz": [2.0, 0.8, 0.8],
            "size_xyz": [0.56, 0.56, 1.6],
        },
    )

    manifest = M._build_placement_validation_manifest(
        stage=object(),
        robot_prim_path="/World/G1",
        stance_plan=stance_plan,
        accepted_pose=(2.0, 0.8, 0.84),
        accepted_yaw=math.pi / 2,
        root_diagnostics={"status": "placed"},
        scene_objects=[],
        scenario_id="sink_stance",
        visual_qc={
            "status": "blocked",
            "blockers": ["placement_visual_qc_unparsed"],
            "per_frame": [
                {
                    "parsed": False,
                    "passed": False,
                    "error": "ClientError('429 RESOURCE_EXHAUSTED')",
                }
            ],
        },
    )

    assert manifest["status"] == "PASS"
    assert manifest["blockers"] == []
    assert manifest["visual_qc"]["status"] == "blocked"


def test_placement_validation_fails_on_parsed_visual_qc_failure(monkeypatch) -> None:
    stance_plan = {
        "status": "accepted",
        "accepted_pose": [2.0, 0.8, 0.84],
        "accepted_yaw": math.pi / 2,
        "floor_z_hint": 0.05,
        "task_target_xyz": [2.0, 2.0, 0.95],
        "task_target_bounds": {
            "bbox_min_xyz": [1.5, 1.65, 0.75],
            "bbox_max_xyz": [2.5, 2.35, 1.15],
        },
    }
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [1.72, 0.52, 0.0],
            "bbox_max_xyz": [2.28, 1.08, 1.6],
            "center_xyz": [2.0, 0.8, 0.8],
            "size_xyz": [0.56, 0.56, 1.6],
        },
    )

    manifest = M._build_placement_validation_manifest(
        stage=object(),
        robot_prim_path="/World/G1",
        stance_plan=stance_plan,
        accepted_pose=(2.0, 0.8, 0.84),
        accepted_yaw=math.pi / 2,
        root_diagnostics={"status": "placed"},
        scene_objects=[],
        scenario_id="sink_stance",
        visual_qc={
            "status": "blocked",
            "blockers": ["placement_visual_qc_failed"],
            "per_frame": [{"parsed": True, "passed": False, "reason": "robot is clipping"}],
        },
    )

    assert manifest["status"] == "FAIL"
    assert "placement_visual_qc_failed" in manifest["blockers"]


def test_placement_validation_manifest_fails_on_actual_bbox_center_mismatch(monkeypatch) -> None:
    stance_plan = {
        "status": "accepted",
        "accepted_pose": [2.0, 0.8, 0.84],
        "accepted_yaw": math.pi / 2,
        "floor_z_hint": 0.05,
        "task_target_xyz": [2.0, 2.0, 0.95],
        "task_target_bounds": {
            "bbox_min_xyz": [1.5, 1.65, 0.75],
            "bbox_max_xyz": [2.5, 2.35, 1.15],
        },
    }
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [1.72, 1.32, 0.0],
            "bbox_max_xyz": [2.28, 1.88, 1.6],
            "center_xyz": [2.0, 1.6, 0.8],
            "size_xyz": [0.56, 0.56, 1.6],
        },
    )
    monkeypatch.setattr(M, "_root_transform_diagnostics", lambda _stage, _prim_path: {"root": "diag"})

    manifest = M._build_placement_validation_manifest(
        stage=object(),
        robot_prim_path="/World/G1",
        stance_plan=stance_plan,
        accepted_pose=(2.0, 0.8, 0.84),
        accepted_yaw=math.pi / 2,
        root_diagnostics={"status": "placed"},
        scene_objects=[],
        scenario_id="sink_stance",
        visual_qc={"status": "passed", "blockers": []},
    )

    assert manifest["status"] == "FAIL"
    assert "placement_ground_truth_center_mismatch" in manifest["blockers"]
    gt = manifest["ground_truth_placement"]
    assert gt["robot_prim_path"] == "/World/G1"
    assert gt["accepted_pose_xyz"] == [2.0, 0.8, 0.84]
    assert gt["actual_world_aabb"]["center_xyz"] == [2.0, 1.6, 0.8]
    assert gt["actual_footprint_center_xyz"] == [2.0, 1.6, 0.8]
    assert gt["computed_xyz_offset_m"] == [0.0, 0.8, -0.04]
    assert gt["xy_error_m"] > 0.1
    assert gt["xform_diagnostics"] == {"root": "diag"}


def test_dynamic_scene_target_bounds_thread_into_task_stance(monkeypatch) -> None:
    def fake_resolve(_stage, _scenario):
        return {
            "status": "resolved",
            "source": "scene_placement_task_label",
            "selected": {
                "target_object_id": "sink",
                "target_object_label": "sink",
                "center_xyz": [2.0, 2.0, 0.95],
                "size_xyz": [1.0, 0.7, 0.4],
                "bbox_min_xyz": [1.5, 1.65, 0.75],
                "bbox_max_xyz": [2.5, 2.35, 1.15],
            },
        }

    monkeypatch.setattr(M, "_resolve_task_target_from_stage", fake_resolve)
    scenario = {
        "instruction": "Stand at the kitchen sink and turn on the faucet.",
        "raw_spawn_position_xyz": [2.0, 0.0, 0.05],
        "floor_z_hint": 0.05,
    }

    plan = M._plan_task_stance_for_stage(
        stage=object(),
        scenario=scenario,
        manipulation_look_at=None,
        probe=lambda pose, yaw: 0,
        no_collision_probe=False,
    )

    assert plan["status"] == "accepted"
    assert plan["accepted_pose"][1] == pytest.approx(1.4)
    assert plan["target_resolution"]["selected"]["target_object_id"] == "sink"
    assert plan["task_target_bounds"]["bbox_max_xyz"] == [2.5, 2.35, 1.15]
    assert plan["candidates"][0]["standoff_from_target_surface_m"] == pytest.approx(
        0.25
    )


def test_stage_affordance_resolution_threads_into_stance_reach_selection(monkeypatch) -> None:
    def fake_resolve(_stage, scenario, allow_scene_placement_fallback=True):
        if scenario.get("target_object_ids") == ["spout"]:
            return {
                "status": "resolved",
                "source": "usd_prim_bounds",
                "selected": {
                    "target_object_id": "spout",
                    "target_object_priority": 0,
                    "prim_path": "/root/CoffeeMachine006/CoffeeMachine006_Spout",
                    "center_xyz": [0.0, -0.1, 1.0],
                    "size_xyz": [0.1, 0.1, 0.1],
                    "bbox_min_xyz": [-0.05, -0.15, 0.95],
                    "bbox_max_xyz": [0.05, -0.05, 1.05],
                },
                "matches_considered": [
                    {
                        "target_object_id": "spout",
                        "target_object_priority": 0,
                        "prim_path": "/root/CoffeeMachine006/CoffeeMachine006_Spout",
                        "center_xyz": [0.0, -0.1, 1.0],
                        "bbox_min_xyz": [-0.05, -0.15, 0.95],
                        "bbox_max_xyz": [0.05, -0.05, 1.05],
                    },
                    {
                        "target_object_id": "spout",
                        "target_object_priority": 0,
                        "prim_path": "/root/Sink054_01/Sink054_spout",
                        "center_xyz": [0.0, -0.1, 1.0],
                        "bbox_min_xyz": [-0.05, -0.15, 0.95],
                        "bbox_max_xyz": [0.05, -0.05, 1.05],
                    },
                ],
            }
        return {
            "status": "resolved",
            "source": "usd_prim_bounds",
            "selected": {
                "target_object_id": "sink",
                "target_object_priority": 0,
                "prim_path": "/root/Sink054_01",
                "center_xyz": [0.0, 0.0, 0.9],
                "size_xyz": [0.2, 0.2, 0.2],
                "bbox_min_xyz": [-0.1, -0.1, 0.8],
                "bbox_max_xyz": [0.1, 0.1, 1.0],
            },
        }

    monkeypatch.setattr(M, "_resolve_task_target_from_stage", fake_resolve)
    scenario = {
        "instruction": "Stand at the kitchen sink and turn on the faucet.",
        "raw_spawn_position_xyz": [-3.0, 0.0, 0.05],
        "floor_z_hint": 0.05,
        "stance_distance_candidates_m": [0.4],
        "target_object_ids": ["sink"],
        "affordance_object_ids": ["spout"],
    }

    plan = M._plan_task_stance_for_stage(
        stage=object(),
        scenario=scenario,
        manipulation_look_at=None,
        probe=lambda pose, yaw: 0,
        no_collision_probe=False,
    )

    assert plan["status"] == "accepted"
    assert plan["affordance_resolution"]["selected"]["prim_path"].endswith("Sink054_spout")
    assert plan["affordance_resolution"]["scope_filter"]["status"] == "scoped_to_target_fixture"
    assert plan["reachability_selection_enabled"] is True
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["reachability_estimate"]["target_affordance_xyz"] == [0.0, -0.1, 1.0]


def test_stage_affordance_resolution_with_explicit_coarse_target_still_drives_stance(
    monkeypatch,
) -> None:
    calls: list[list[str]] = []

    def fake_resolve(_stage, scenario, allow_scene_placement_fallback=True):
        target_ids = list(scenario.get("target_object_ids") or [])
        calls.append(target_ids)
        if target_ids == ["handle"]:
            return {
                "status": "resolved",
                "source": "usd_prim_bounds",
                "selected": {
                    "target_object_id": "handle",
                    "target_object_priority": 0,
                    "prim_path": "/root/Sink054_01/Sink054_handle",
                    "center_xyz": [0.0, -0.1, 1.0],
                    "bbox_min_xyz": [-0.05, -0.15, 0.95],
                    "bbox_max_xyz": [0.05, -0.05, 1.05],
                },
                "matches_considered": [
                    {
                        "target_object_id": "handle",
                        "target_object_priority": 0,
                        "prim_path": "/root/Sink054_01/Sink054_handle",
                        "center_xyz": [0.0, -0.1, 1.0],
                        "bbox_min_xyz": [-0.05, -0.15, 0.95],
                        "bbox_max_xyz": [0.05, -0.05, 1.05],
                    }
                ],
            }
        return {
            "status": "resolved",
            "source": "usd_prim_bounds",
            "selected": {
                "target_object_id": "sink",
                "target_object_priority": 0,
                "prim_path": "/root/Sink054_01",
                "center_xyz": [0.0, 0.0, 0.9],
                "bbox_min_xyz": [-0.1, -0.1, 0.8],
                "bbox_max_xyz": [0.1, 0.1, 1.0],
            },
        }

    monkeypatch.setattr(M, "_resolve_task_target_from_stage", fake_resolve)
    scenario = {
        "instruction": "Stand at the kitchen sink and turn on the faucet.",
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "target_object_bbox_min_xyz": [-0.1, -0.1, 0.8],
        "target_object_bbox_max_xyz": [0.1, 0.1, 1.0],
        "raw_spawn_position_xyz": [-3.0, 0.0, 0.05],
        "floor_z_hint": 0.05,
        "stance_distance_candidates_m": [0.4],
        "target_object_ids": ["sink"],
        "affordance_object_ids": ["handle"],
    }

    plan = M._plan_task_stance_for_stage(
        stage=object(),
        scenario=scenario,
        manipulation_look_at=None,
        probe=lambda pose, yaw: 0,
        no_collision_probe=False,
    )

    assert calls == [["sink"], ["handle"]]
    assert plan["status"] == "accepted"
    assert plan["task_target_xyz"] == [0.0, 0.0, 0.9]
    assert plan["task_affordance_xyz"] == [0.0, -0.1, 1.0]
    assert plan["stance_focus_source"] == "task_affordance_xyz"
    assert plan["affordance_focus_source"] == "usd_affordance_object_alias"
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["stance_focus_xyz"] == [0.0, -0.1, 1.0]
    assert chosen["reachability_estimate"]["status"] == "PASS"


def test_scoped_affordance_resolution_preserves_explicit_control_priority() -> None:
    target_resolution = {
        "status": "resolved",
        "source": "usd_prim_bounds",
        "selected": {
            "target_object_id": "sink",
            "target_object_priority": 0,
            "prim_path": "/root/Sink054_01",
            "center_xyz": [2.277888, 1.333059, 0.848527],
            "bbox_min_xyz": [2.009021, 0.89582, 0.555082],
            "bbox_max_xyz": [2.546755, 1.770299, 1.141971],
        },
    }
    affordance_resolution = {
        "status": "resolved",
        "source": "usd_prim_bounds",
        "selected": {
            "target_object_id": "handle",
            "target_object_priority": 0,
            "prim_path": "/root/Sink054_01/Sink054_handle",
            "center_xyz": [2.489866, 1.069795, 0.886474],
            "bbox_min_xyz": [2.466351, 1.050669, 0.82909],
            "bbox_max_xyz": [2.513381, 1.088922, 0.943858],
        },
        "matches_considered": [
            {
                "target_object_id": "spout",
                "target_object_priority": 5,
                "prim_path": "/root/Sink054_01/Sink054_spout",
                "center_xyz": [2.409514, 1.145855, 1.017716],
                "bbox_min_xyz": [2.31906, 1.136575, 0.893461],
                "bbox_max_xyz": [2.499968, 1.155135, 1.141971],
            },
            {
                "target_object_id": "handle",
                "target_object_priority": 0,
                "prim_path": "/root/Sink054_01/Sink054_handle",
                "center_xyz": [2.489866, 1.069795, 0.886474],
                "bbox_min_xyz": [2.466351, 1.050669, 0.82909],
                "bbox_max_xyz": [2.513381, 1.088922, 0.943858],
            },
        ],
    }

    scoped = M._scope_affordance_resolution_to_target(
        affordance_resolution,
        target_resolution,
    )

    assert scoped["selected"]["target_object_id"] == "handle"
    assert scoped["selected"]["prim_path"].endswith("Sink054_handle")
    assert scoped["matches_considered"][0]["target_object_id"] == "handle"


def test_stage_affordance_resolution_rejects_unscoped_global_handle(monkeypatch) -> None:
    def fake_resolve(_stage, scenario, allow_scene_placement_fallback=True):
        if scenario.get("target_object_ids") == ["handle"]:
            return {
                "status": "resolved",
                "source": "usd_prim_bounds",
                "selected": {
                    "target_object_id": "handle",
                    "target_object_priority": 0,
                    "prim_path": "/root/CoffeeMachine006/CoffeeMachine006_Handle",
                    "center_xyz": [0.0, 1.0, 0.9],
                    "bbox_min_xyz": [-0.05, 0.95, 0.85],
                    "bbox_max_xyz": [0.05, 1.05, 0.95],
                },
                "matches_considered": [
                    {
                        "target_object_id": "handle",
                        "target_object_priority": 0,
                        "prim_path": "/root/CoffeeMachine006/CoffeeMachine006_Handle",
                        "center_xyz": [0.0, 1.0, 0.9],
                        "bbox_min_xyz": [-0.05, 0.95, 0.85],
                        "bbox_max_xyz": [0.05, 1.05, 0.95],
                    }
                ],
            }
        return {
            "status": "resolved",
            "source": "usd_prim_bounds",
            "selected": {
                "target_object_id": "topcabinet",
                "target_object_priority": 0,
                "prim_path": "/root/Kitchen_TopCabinet_01",
                "center_xyz": [0.0, 1.0, 1.9],
                "size_xyz": [1.0, 0.4, 0.8],
                "bbox_min_xyz": [-0.5, 0.8, 1.5],
                "bbox_max_xyz": [0.5, 1.2, 2.3],
            },
        }

    monkeypatch.setattr(M, "_resolve_task_target_from_stage", fake_resolve)

    plan = M._plan_task_stance_for_stage(
        stage=object(),
        scenario={
            "instruction": "Stand at the upper kitchen cabinet and reach for the cabinet handle.",
            "raw_spawn_position_xyz": [0.0, 0.0, 0.05],
            "floor_z_hint": 0.05,
            "target_object_ids": ["topcabinet", "cabinet"],
            "affordance_object_ids": ["handle"],
            "stance_distance_candidates_m": [0.4],
        },
        manipulation_look_at=None,
        probe=lambda pose, yaw: 0,
        no_collision_probe=False,
    )

    assert plan["status"] == "blocked"
    assert plan["blockers"] == ["task_affordance_resolution_failed"]
    assert plan["affordance_resolution"]["status"] == "blocked"
    assert "affordance_not_scoped_to_target_fixture" in plan["affordance_resolution"]["blockers"]
    assert "task_affordance_xyz" not in plan
    assert "affordance_focus_source" not in plan


def test_surface_affordance_point_prefers_fine_affordance() -> None:
    plan = {
        "task_target_xyz": [2.0, 1.0, 0.9],
        "task_target_bounds": {
            "bbox_min_xyz": [1.5, 0.5, 0.7],
            "bbox_max_xyz": [2.5, 1.5, 1.2],
        },
        "task_affordance_xyz": [2.4, 0.8, 1.05],
    }

    point = M._surface_affordance_point_for_stance(plan, [1.0, 1.0, 0.84])

    assert point == (2.4, 0.8, 1.05)


def test_task_stance_planner_fails_closed_when_all_candidates_collide() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [0.8, 1.0],
    }
    plan = M.plan_task_stance(scenario=scenario, probe_collision=lambda pose, yaw: 2)
    assert plan["status"] == "blocked"
    assert plan["blockers"] == ["no_collision_free_task_stance_candidate"]
    assert all(c["scene_collision_contact_count"] == 2 for c in plan["candidates"])


def test_sink_target_standoff_does_not_use_broad_cabinet_fixture() -> None:
    sink = SceneObject(
        id="sink",
        label="sink",
        bbox_min=(2.0, 0.9, 0.5),
        bbox_max=(2.6, 1.8, 1.1),
        centroid=(2.3, 1.35, 0.8),
    )
    cabinet = SceneObject(
        id="kitchen_cabinet_1",
        label="kitchen_cabinet",
        bbox_min=(-1.3, -0.15, 0.0),
        bbox_max=(2.6, 2.45, 0.85),
        centroid=(0.65, 1.15, 0.4),
    )
    faucet = SceneObject(
        id="faucet",
        label="faucet",
        bbox_min=(2.25, 1.25, 0.9),
        bbox_max=(2.35, 1.35, 1.1),
        centroid=(2.3, 1.3, 1.0),
    )

    assert M._find_standoff_fixtures([cabinet, sink], sink) == []
    assert M._find_standoff_fixtures([cabinet, sink], faucet) == [cabinet, sink]


def test_placement_shell_obstacles_include_walls_not_floor_or_lights(monkeypatch) -> None:
    class FakeRangeBox:
        def __init__(self, bmin, bmax):
            self._min = bmin
            self._max = bmax

        def IsEmpty(self):
            return False

        def GetMin(self):
            return self._min

        def GetMax(self):
            return self._max

        def GetSize(self):
            return [self._max[i] - self._min[i] for i in range(3)]

    class FakeBound:
        def __init__(self, box):
            self._box = box

        def ComputeAlignedBox(self):
            return self._box

    class FakePrim:
        def __init__(self, path, name, bmin, bmax):
            self._path = path
            self._name = name
            self.box = FakeRangeBox(bmin, bmax)

        def GetPath(self):
            return self._path

        def GetName(self):
            return self._name

    class FakeStage:
        def Traverse(self):
            return [
                FakePrim("/World/Kitchen_Wall001", "Kitchen_Wall001", [2.8, -1, 0], [2.9, 2, 2.5]),
                FakePrim("/World/Kitchen_Wall_Group", "Kitchen_Wall_Group", [-2, -2, 0], [4, 4, 2.5]),
                FakePrim("/World/Kitchen_Cabinet_Door001", "Kitchen_Cabinet_Door001", [1, 1, 0], [1.05, 1.5, 0.8]),
                FakePrim("/World/Kitchen_Floor", "Kitchen_Floor", [-2, -2, 0], [4, 4, 0.05]),
                FakePrim("/World/RectLight_01", "RectLight_01", [0, 0, 2], [1, 1, 2.1]),
            ]

    class FakeBBoxCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def ComputeWorldBound(self, prim):
            return FakeBound(prim.box)

    fake_usd = types.SimpleNamespace(TimeCode=types.SimpleNamespace(Default=lambda: "default"))
    fake_usd_geom = types.SimpleNamespace(
        Tokens=types.SimpleNamespace(default_="default", render="render", proxy="proxy"),
        BBoxCache=FakeBBoxCache,
    )
    fake_pxr = types.SimpleNamespace(Usd=fake_usd, UsdGeom=fake_usd_geom)
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    monkeypatch.setitem(sys.modules, "pxr.Usd", fake_usd)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", fake_usd_geom)

    obstacles = M._placement_shell_obstacles_for_stage(FakeStage())

    assert [obj.id for obj in obstacles] == ["world_kitchen_wall001"]
    assert obstacles[0].source == "usd_shell"
    assert obstacles[0].bbox_min == (2.8, -1.0, 0.0)


def test_placement_shell_obstacles_synthesize_broad_wall_mesh_edges(monkeypatch) -> None:
    class FakeRangeBox:
        def __init__(self, bmin, bmax):
            self._min = bmin
            self._max = bmax

        def IsEmpty(self):
            return False

        def GetMin(self):
            return self._min

        def GetMax(self):
            return self._max

        def GetSize(self):
            return [self._max[i] - self._min[i] for i in range(3)]

    class FakeBound:
        def __init__(self, box):
            self._box = box

        def ComputeAlignedBox(self):
            return self._box

    class FakePrim:
        def __init__(self, path, name, bmin, bmax, is_mesh=False):
            self._path = path
            self._name = name
            self.box = FakeRangeBox(bmin, bmax)
            self._is_mesh = is_mesh

        def GetPath(self):
            return self._path

        def GetName(self):
            return self._name

        def IsA(self, type_marker):
            return self._is_mesh and type_marker == "MeshType"

    class FakeStage:
        def Traverse(self):
            return [
                FakePrim(
                    "/root/Kitchen_Wall001",
                    "Kitchen_Wall001",
                    [-2.4, -1.6, 0.0],
                    [2.76, 2.61, 2.7],
                    is_mesh=True,
                )
            ]

    class FakeBBoxCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def ComputeWorldBound(self, prim):
            return FakeBound(prim.box)

    fake_usd = types.SimpleNamespace(TimeCode=types.SimpleNamespace(Default=lambda: "default"))
    fake_usd_geom = types.SimpleNamespace(
        Tokens=types.SimpleNamespace(default_="default", render="render", proxy="proxy"),
        BBoxCache=FakeBBoxCache,
        Mesh="MeshType",
    )
    fake_pxr = types.SimpleNamespace(Usd=fake_usd, UsdGeom=fake_usd_geom)
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    monkeypatch.setitem(sys.modules, "pxr.Usd", fake_usd)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", fake_usd_geom)

    obstacles = M._placement_shell_obstacles_for_stage(FakeStage())

    assert [obj.id for obj in obstacles] == [
        "root_kitchen_wall001_xmin",
        "root_kitchen_wall001_xmax",
        "root_kitchen_wall001_ymin",
        "root_kitchen_wall001_ymax",
    ]
    xmax = obstacles[1]
    assert xmax.bbox_min[0] == pytest.approx(2.72)
    assert xmax.bbox_max[0] == pytest.approx(2.8)
    assert xmax.source == "usd_shell"


def test_usd_task_target_resolver_prefers_object_root_over_descendant(monkeypatch) -> None:
    class FakeBox:
        def __init__(self, center, size):
            self._center = center
            self._size = size

        def IsEmpty(self):
            return False

        def GetCenter(self):
            return self._center

        def GetSize(self):
            return self._size

    class FakeRangeBox:
        def __init__(self, bmin, bmax):
            self._min = bmin
            self._max = bmax

        def IsEmpty(self):
            return False

        def GetMin(self):
            return self._min

        def GetMax(self):
            return self._max

        def GetSize(self):
            return [self._max[i] - self._min[i] for i in range(3)]

    class FakeBound:
        def __init__(self, box):
            self._box = box

        def ComputeAlignedBox(self):
            return self._box

    class FakePrim:
        def __init__(self, path, name, center, size, *, range_box=False):
            self._path = path
            self._name = name
            if range_box:
                self.box = FakeRangeBox(
                    [center[i] - size[i] / 2 for i in range(3)],
                    [center[i] + size[i] / 2 for i in range(3)],
                )
            else:
                self.box = FakeBox(center, size)

        def GetPath(self):
            return self._path

        def GetName(self):
            return self._name

    root = FakePrim(
        "/World/Sink054", "Sink054", [2.0, 0.5, 0.9], [1.2, 0.8, 1.0],
        range_box=True,
    )
    child = FakePrim("/World/Sink054/tiny_mesh", "tiny_mesh", [9.0, 9.0, 9.0], [0.1, 0.1, 0.1])

    class FakeStage:
        def Traverse(self):
            return [child, root]

    class FakeBBoxCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def ComputeWorldBound(self, prim):
            return FakeBound(prim.box)

    fake_usd = types.SimpleNamespace(TimeCode=types.SimpleNamespace(Default=lambda: "default"))
    fake_usd_geom = types.SimpleNamespace(
        Tokens=types.SimpleNamespace(default_="default", render="render", proxy="proxy"),
        BBoxCache=FakeBBoxCache,
    )
    fake_pxr = types.SimpleNamespace(Usd=fake_usd, UsdGeom=fake_usd_geom)
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    monkeypatch.setitem(sys.modules, "pxr.Usd", fake_usd)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", fake_usd_geom)

    result = M._resolve_task_target_from_stage(FakeStage(), {"target_object_id": "Sink054"})

    assert result["status"] == "resolved"
    assert result["selected"]["prim_path"] == "/World/Sink054"
    assert result["selected"]["center_xyz"] == [2.0, 0.5, 0.9]
    assert result["selected"]["match_kind"] == "exact_prim_name_or_path_segment"


def test_usd_task_target_resolver_prefers_ordered_affordance_alias(monkeypatch) -> None:
    class FakeRangeBox:
        def __init__(self, bmin, bmax):
            self._min = bmin
            self._max = bmax

        def IsEmpty(self):
            return False

        def GetMin(self):
            return self._min

        def GetMax(self):
            return self._max

        def GetSize(self):
            return [self._max[i] - self._min[i] for i in range(3)]

    class FakeBound:
        def __init__(self, box):
            self._box = box

        def ComputeAlignedBox(self):
            return self._box

    class FakePrim:
        def __init__(self, path, name, bmin, bmax):
            self._path = path
            self._name = name
            self.box = FakeRangeBox(bmin, bmax)

        def GetPath(self):
            return self._path

        def GetName(self):
            return self._name

    sink = FakePrim("/World/Sink054", "Sink054", [1.5, 1.65, 0.75], [2.5, 2.35, 1.15])
    faucet = FakePrim(
        "/World/Sink054/FaucetLever",
        "FaucetLever",
        [2.08, 1.92, 1.05],
        [2.22, 2.08, 1.18],
    )

    class FakeStage:
        def Traverse(self):
            return [sink, faucet]

    class FakeBBoxCache:
        def __init__(self, *_args, **_kwargs):
            pass

        def ComputeWorldBound(self, prim):
            return FakeBound(prim.box)

    fake_usd = types.SimpleNamespace(TimeCode=types.SimpleNamespace(Default=lambda: "default"))
    fake_usd_geom = types.SimpleNamespace(
        Tokens=types.SimpleNamespace(default_="default", render="render", proxy="proxy"),
        BBoxCache=FakeBBoxCache,
    )
    fake_pxr = types.SimpleNamespace(Usd=fake_usd, UsdGeom=fake_usd_geom)
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    monkeypatch.setitem(sys.modules, "pxr.Usd", fake_usd)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", fake_usd_geom)

    result = M._resolve_task_target_from_stage(
        FakeStage(),
        {"target_object_ids": ["faucet", "lever", "sink"]},
    )

    assert result["status"] == "resolved"
    assert result["selected"]["prim_path"] == "/World/Sink054/FaucetLever"
    assert result["selected"]["target_object_id"] == "faucet"
    assert result["selected"]["target_object_priority"] == 0


def test_assemble_collision_summary_counts() -> None:
    actions = [
        {"scene_collision_contact_count": 0, "policy_action": "accepted_direct_collision_checked_motion"},
        {"scene_collision_contact_count": 0, "policy_action": "redirected_by_collision_probe"},
    ]
    summ = M.assemble_collision_summary(actions=actions, rejected_probe_total=3, response_event_total=1)
    assert summ["robot_scene_contact_event_count"] == 0
    assert summ["rejected_scene_collision_probe_count"] == 3
    assert summ["near_miss_event_count"] == 3
    assert summ["collision_response_event_count"] == 1


def test_mp4_command_is_web_playable_ffmpeg() -> None:
    cmd = M.mp4_command("frames/overview_*.png", 24, "overview.mp4")
    assert cmd[0] == "ffmpeg"
    assert "yuv420p" in cmd  # web-playable pixel format
    assert "libx264" in cmd
    assert cmd[-1] == "overview.mp4"


def test_yaw_to_quat_is_wxyz_about_z() -> None:
    w, x, y, z = M.yaw_to_quat(math.pi / 2)
    assert x == 0.0 and y == 0.0
    assert w == pytest.approx(math.cos(math.pi / 4))
    assert z == pytest.approx(math.sin(math.pi / 4))


def test_build_result_aggregates_and_labels_truthfully() -> None:
    scs = [{"scenario_id": "a"}, {"scenario_id": "b"}]
    outs = [{"task_success": True}, {"task_success": False}]
    res = M.build_result(scenarios=scs, outcomes=outs, policy_id="blueprint_default_walk_to_target_smoke_policy",
                         kitchen_usd="k.usd", g1_usd="g1.usd", blockers=[])
    assert res["status"] == "completed"
    assert res["scenarios_passed"] == 1 and res["scenarios_executed"] == 2
    assert res["review_grade_scenarios_passed"] == 0
    assert res["rendered_by_isaac_rtx"] is True
    assert "not dynamic locomotion" in res["proof_boundary"].lower()
    assert res["scenarios"][0]["scenario_id"] == "a" and res["scenarios"][0]["task_success"] is True
    assert res["scenarios"][0]["review_task_success"] is False
    assert "review_camera_evidence_missing" in res["scenarios"][0]["review_task_success_evidence"]["blockers"]


def test_build_result_counts_review_grade_success_only_with_visible_action_evidence() -> None:
    res = M.build_result(
        scenarios=[{"scenario_id": "visible_reach"}],
        outcomes=[
            {
                "task_success": True,
                "review_camera_evidence": {
                    "robot_pov_camera_mode": "robot_mounted_manipulation",
                    "visible_embodied_robot_action_evidence": True,
                },
                "robot_visual_geometry": {"status": "PASS", "blockers": []},
                "manipulation_pov_geometry": {"status": "PASS", "blockers": []},
            }
        ],
        policy_id="blueprint_default_walk_to_target_smoke_policy",
        kitchen_usd="k.usd",
        g1_usd="g1.usd",
        blockers=[],
    )

    assert res["scenarios_passed"] == 1
    assert res["review_grade_scenarios_passed"] == 1
    assert res["scenarios"][0]["review_task_success"] is True
    assert res["scenarios"][0]["review_task_success_evidence"]["blockers"] == []


def test_visible_reach_success_contract_passes_only_with_required_evidence() -> None:
    outcome = {
        "task_success": False,
        "task_status": "failed_task_criteria",
        "failure_mode_ids": ["failure_target_not_reached"],
        "episode_termination": {
            "enabled": True,
            "status": "PASS",
            "terminal_reason": "success_held",
            "blockers": [],
        },
    }

    contract = M._apply_visible_reach_to_affordance_success_contract(
        outcome,
        placement_validation={"status": "PASS", "blockers": []},
        pov_geometry={
            "status": "PASS",
            "blockers": [],
            "reach_feasibility": {"status": "PASS", "blockers": []},
            "effector_distance_to_affordance_m": {"hand": 0.08},
        },
        robot_visual_ready=True,
        temporal_conditioning={"status": "PASS", "blockers": []},
    )

    assert contract["status"] == "PASS"
    assert outcome["task_success"] is True
    assert outcome["task_status"] == "passed"
    assert outcome["failure_mode_ids"] == []
    assert "faucet state change" in contract["claim_boundary"]


def test_visible_reach_success_contract_requires_dynamic_success_held() -> None:
    outcome = {
        "task_success": False,
        "episode_termination": {
            "enabled": False,
            "status": "FAIL",
            "terminal_reason": "dynamic_episode_disabled",
            "blockers": ["dynamic_episode_termination_required"],
        },
    }

    contract = M._apply_visible_reach_to_affordance_success_contract(
        outcome,
        placement_validation={"status": "PASS", "blockers": []},
        pov_geometry={
            "status": "PASS",
            "blockers": [],
            "frames": [
                {
                    "status": "PASS",
                    "reach_feasibility": {"status": "PASS", "blockers": []},
                    "effector_distance_to_affordance_m": {"hand": 0.07},
                }
            ],
        },
        robot_visual_ready=True,
        temporal_conditioning={"status": "PASS", "blockers": []},
    )

    assert contract["status"] == "FAIL"
    assert contract["required_evidence"]["episode_termination_passed"] is False
    assert outcome["task_success"] is False
    assert "visible_reach_episode_not_successfully_terminated" in outcome["failure_mode_ids"]
    assert "dynamic_episode_termination_required" in outcome["failure_mode_ids"]
    assert "dynamic_episode_terminal_reason:dynamic_episode_disabled" in outcome["failure_mode_ids"]


def test_visible_reach_success_contract_fails_when_dynamic_report_missing() -> None:
    outcome = {"task_success": False}

    contract = M._apply_visible_reach_to_affordance_success_contract(
        outcome,
        placement_validation={"status": "PASS", "blockers": []},
        pov_geometry={
            "status": "PASS",
            "blockers": [],
            "frames": [
                {
                    "status": "PASS",
                    "reach_feasibility": {"status": "PASS", "blockers": []},
                    "effector_distance_to_affordance_m": {"hand": 0.07},
                }
            ],
        },
        robot_visual_ready=True,
        temporal_conditioning={"status": "PASS", "blockers": []},
    )

    assert contract["status"] == "FAIL"
    assert outcome["task_success"] is False
    assert "dynamic_episode_termination_missing" in outcome["failure_mode_ids"]


def test_visible_reach_success_contract_fails_without_pov_geometry() -> None:
    outcome = {"task_success": False}

    contract = M._apply_visible_reach_to_affordance_success_contract(
        outcome,
        placement_validation={"status": "PASS", "blockers": []},
        pov_geometry={"status": "FAIL", "blockers": ["arm_not_in_frame"]},
        robot_visual_ready=True,
        temporal_conditioning={"status": "PASS", "blockers": []},
    )

    assert contract["status"] == "FAIL"
    assert outcome["task_success"] is False
    assert "visible_reach_pov_geometry_not_passed" in outcome["failure_mode_ids"]
    assert "arm_not_in_frame" in outcome["failure_mode_ids"]


def test_visible_reach_success_contract_fails_when_final_frame_reach_is_infeasible() -> None:
    outcome = {"task_success": False}

    contract = M._apply_visible_reach_to_affordance_success_contract(
        outcome,
        placement_validation={"status": "PASS", "blockers": []},
        pov_geometry={
            "status": "PASS",
            "blockers": [],
            "frames": [
                {
                    "status": "PASS",
                    "reach_feasibility": {
                        "status": "FAIL",
                        "blockers": [
                            "manipulation_pov_affordance_outside_g1_reach_envelope",
                            "manipulation_pov_effector_too_far_from_affordance",
                        ],
                        "passing_arms": [],
                    },
                }
            ],
        },
        robot_visual_ready=True,
        temporal_conditioning={"status": "PASS", "blockers": []},
    )

    assert contract["status"] == "FAIL"
    assert outcome["task_success"] is False
    assert "visible_reach_reach_feasibility_not_passed" in outcome["failure_mode_ids"]
    assert "visible_reach_final_frame_reach_feasibility_not_passed" in outcome["failure_mode_ids"]
    assert "manipulation_pov_effector_too_far_from_affordance" in outcome["failure_mode_ids"]
    assert contract["reach_feasibility_evidence"]["final_frame_reach_feasibility_passed"] is False


def test_visible_reach_success_contract_fails_when_final_frame_effector_is_not_close_enough() -> None:
    outcome = {"task_success": False}

    contract = M._apply_visible_reach_to_affordance_success_contract(
        outcome,
        placement_validation={"status": "PASS", "blockers": []},
        pov_geometry={
            "status": "PASS",
            "blockers": [],
            "frames": [
                {
                    "status": "PASS",
                    "reach_feasibility": {"status": "PASS", "blockers": []},
                    "effector_distance_to_affordance_m_by_arm": {
                        "left": {"hand": 0.2195, "wrist": 0.3973},
                        "right": {"hand": 0.2303, "wrist": 0.3485},
                    },
                }
            ],
        },
        robot_visual_ready=True,
        temporal_conditioning={"status": "PASS", "blockers": []},
    )

    assert contract["status"] == "FAIL"
    assert outcome["task_success"] is False
    assert "visible_reach_reach_feasibility_not_passed" in outcome["failure_mode_ids"]
    assert "visible_reach_final_frame_effector_not_close_enough" in outcome["failure_mode_ids"]
    evidence = contract["reach_feasibility_evidence"]
    assert evidence["final_frame_reach_feasibility_passed"] is True
    assert evidence["final_frame_effector_close_enough"] is False
    assert evidence["final_frame_nearest_effector_to_affordance_m"] == 0.2195
    assert evidence["max_final_effector_to_affordance_m"] == 0.08


def test_visible_reach_success_contract_fails_when_final_frame_is_only_a_near_miss() -> None:
    outcome = {"task_success": False}

    contract = M._apply_visible_reach_to_affordance_success_contract(
        outcome,
        placement_validation={"status": "PASS", "blockers": []},
        pov_geometry={
            "status": "PASS",
            "blockers": [],
            "frames": [
                {
                    "status": "PASS",
                    "reach_feasibility": {"status": "PASS", "blockers": []},
                    "effector_distance_to_affordance_m": {"hand": 0.1095},
                }
            ],
        },
        robot_visual_ready=True,
        temporal_conditioning={"status": "PASS", "blockers": []},
    )

    assert contract["status"] == "FAIL"
    assert outcome["task_success"] is False
    assert "visible_reach_final_frame_effector_not_close_enough" in outcome["failure_mode_ids"]
    evidence = contract["reach_feasibility_evidence"]
    assert evidence["final_frame_nearest_effector_to_affordance_m"] == 0.1095
    assert evidence["max_final_effector_to_affordance_m"] == 0.08


def test_visible_reach_success_contract_can_use_derived_fingertip_proxy() -> None:
    outcome = {
        "task_success": False,
        "episode_termination": {
            "enabled": True,
            "status": "PASS",
            "terminal_reason": "success_held",
            "blockers": [],
        },
    }

    contract = M._apply_visible_reach_to_affordance_success_contract(
        outcome,
        placement_validation={"status": "PASS", "blockers": []},
        pov_geometry={
            "status": "PASS",
            "blockers": [],
            "frames": [
                {
                    "status": "PASS",
                    "reach_feasibility": {"status": "PASS", "blockers": []},
                    "effector_distance_to_affordance_m": {
                        "hand": 0.1095,
                        "fingertip_proxy": 0.0704,
                    },
                }
            ],
        },
        robot_visual_ready=True,
        temporal_conditioning={"status": "PASS", "blockers": []},
    )

    assert contract["status"] == "PASS"
    assert outcome["task_success"] is True
    evidence = contract["reach_feasibility_evidence"]
    assert evidence["final_frame_nearest_effector_to_affordance_m"] == 0.0704


def test_derived_fingertip_proxy_extends_from_measured_wrist_hand_axis() -> None:
    target = (-1.591312, 1.471274, 1.241574)
    arm_points = {
        "wrist": (-1.3527, 1.4331, 1.063),
        "hand": (-1.485, 1.4643, 1.2163),
    }

    proxy = M._derived_fingertip_proxy_point(arm_points)

    assert proxy is not None
    assert math.dist(proxy, target) == pytest.approx(0.0704, abs=0.001)


def test_visible_reach_dynamic_episode_terminal_success_requires_hold() -> None:
    state = M._initial_dynamic_episode_state()
    frame = {
        "status": "PASS",
        "reach_feasibility": {"status": "PASS", "blockers": []},
        "effector_distance_to_affordance_m": {"hand": 0.07},
    }

    state = M._update_visible_reach_dynamic_episode_state(
        state,
        frame,
        captured_frames=1,
        min_frames=1,
        max_frames=10,
        success_hold_frames=2,
        no_progress_patience_frames=5,
    )
    assert state["terminal_reason"] is None

    state = M._update_visible_reach_dynamic_episode_state(
        state,
        frame,
        captured_frames=2,
        min_frames=1,
        max_frames=10,
        success_hold_frames=2,
        no_progress_patience_frames=5,
    )

    assert state["terminal_reason"] == "success_held"
    report = M._dynamic_episode_termination_report(
        enabled=True,
        task_success_contract="visible_reach_to_affordance",
        requested_steps=2,
        min_steps=1,
        max_steps=10,
        frames_captured=2,
        state=state,
    )
    assert report["status"] == "PASS"
    assert report["terminal_reason"] == "success_held"


def test_visible_reach_dynamic_episode_terminal_no_progress_after_min_steps() -> None:
    state = M._initial_dynamic_episode_state()
    frame = {
        "status": "PASS",
        "reach_feasibility": {"status": "PASS", "blockers": []},
        "effector_distance_to_affordance_m": {"hand": 0.1095},
    }

    for captured in range(1, 5):
        state = M._update_visible_reach_dynamic_episode_state(
            state,
            frame,
            captured_frames=captured,
            min_frames=2,
            max_frames=20,
            success_hold_frames=2,
            no_progress_patience_frames=3,
        )

    assert state["terminal_reason"] == "no_progress"
    assert state["best_effector_to_affordance_m"] == 0.1095
    assert state["terminal_sample"]["terminal_success"] is False
    assert "visible_reach_terminal_frame_effector_not_close_enough" in (
        state["terminal_sample"]["blockers"]
    )


def test_visible_reach_dynamic_episode_terminal_max_steps_when_cap_hit() -> None:
    state = M._initial_dynamic_episode_state()
    frame = {
        "status": "PASS",
        "reach_feasibility": {"status": "PASS", "blockers": []},
        "effector_distance_to_affordance_m": {"hand": 0.1095},
    }

    for captured in range(1, 4):
        state = M._update_visible_reach_dynamic_episode_state(
            state,
            frame,
            captured_frames=captured,
            min_frames=1,
            max_frames=3,
            success_hold_frames=2,
            no_progress_patience_frames=99,
        )

    assert state["terminal_reason"] == "max_steps"
    report = M._dynamic_episode_termination_report(
        enabled=True,
        task_success_contract="visible_reach_to_affordance",
        requested_steps=2,
        min_steps=1,
        max_steps=3,
        frames_captured=3,
        state=state,
    )
    assert report["status"] == "FAIL"
    assert report["terminal_reason"] == "max_steps"
    assert report["blockers"] == ["dynamic_episode_terminal_reason:max_steps"]


def test_visible_reach_dynamic_episode_terminal_wall_clock_cap() -> None:
    state = M._initial_dynamic_episode_state()

    report = M._dynamic_episode_termination_report(
        enabled=True,
        task_success_contract="visible_reach_to_affordance",
        requested_steps=8,
        min_steps=8,
        max_steps=12,
        frames_captured=9,
        state=state,
        wall_clock_truncated=True,
    )

    assert report["status"] == "FAIL"
    assert report["terminal_reason"] == "wall_clock_cap"
    assert report["blockers"] == ["dynamic_episode_terminal_reason:wall_clock_cap"]


def test_visible_reach_success_contract_fails_when_dynamic_episode_no_progress() -> None:
    outcome = {
        "task_success": False,
        "episode_termination": {
            "enabled": True,
            "status": "FAIL",
            "terminal_reason": "no_progress",
            "blockers": ["dynamic_episode_terminal_reason:no_progress"],
        },
    }

    contract = M._apply_visible_reach_to_affordance_success_contract(
        outcome,
        placement_validation={"status": "PASS", "blockers": []},
        pov_geometry={
            "status": "PASS",
            "blockers": [],
            "frames": [
                {
                    "status": "PASS",
                    "reach_feasibility": {"status": "PASS", "blockers": []},
                    "effector_distance_to_affordance_m": {"hand": 0.07},
                }
            ],
        },
        robot_visual_ready=True,
        temporal_conditioning={"status": "PASS", "blockers": []},
    )

    assert contract["status"] == "FAIL"
    assert outcome["task_success"] is False
    assert "visible_reach_episode_not_successfully_terminated" in outcome["failure_mode_ids"]
    assert "dynamic_episode_terminal_reason:no_progress" in outcome["failure_mode_ids"]


def test_visible_reach_review_grade_rejects_failed_reach_contract() -> None:
    res = M.build_result(
        scenarios=[{"scenario_id": "faucet_reach"}],
        outcomes=[
            {
                "task_success": True,
                "task_success_contract": "visible_reach_to_affordance",
                "visible_reach_to_affordance_success": {
                    "status": "FAIL",
                    "blockers": ["visible_reach_reach_feasibility_not_passed"],
                },
                "review_camera_evidence": {
                    "robot_pov_camera_mode": "robot_mounted_manipulation",
                    "visible_embodied_robot_action_evidence": True,
                },
                "robot_visual_geometry": {"status": "PASS", "blockers": []},
                "manipulation_pov_geometry": {"status": "PASS", "blockers": []},
            }
        ],
        policy_id="blueprint_default_walk_to_target_smoke_policy",
        kitchen_usd="k.usd",
        g1_usd="g1.usd",
        blockers=[],
    )

    blockers = res["scenarios"][0]["review_task_success_evidence"]["blockers"]
    assert res["review_grade_scenarios_passed"] == 0
    assert res["scenarios"][0]["review_task_success"] is False
    assert "visible_reach_success_contract_not_passed" in blockers
    assert "visible_reach_reach_feasibility_not_passed" in blockers


def test_build_result_rejects_root_follow_camera_as_review_grade_success() -> None:
    res = M.build_result(
        scenarios=[{"scenario_id": "camera_motion_only"}],
        outcomes=[
            {
                "task_success": True,
                "review_camera_evidence": {
                    "robot_pov_camera_mode": "root_follow",
                    "visible_embodied_robot_action_evidence": False,
                },
            }
        ],
        policy_id="blueprint_default_walk_to_target_smoke_policy",
        kitchen_usd="k.usd",
        g1_usd="g1.usd",
        blockers=[],
    )

    blockers = res["scenarios"][0]["review_task_success_evidence"]["blockers"]
    assert res["scenarios_passed"] == 1
    assert res["review_grade_scenarios_passed"] == 0
    assert res["scenarios"][0]["review_task_success"] is False
    assert "robot_pov_is_root_follow_camera_not_head_pov" in blockers
    assert "visible_embodied_robot_action_not_proven" in blockers


def test_runner_labels_legacy_robot_pov_as_root_follow_when_not_manipulation() -> None:
    source = _RUNNER.read_text()

    assert 'pov_camera_mode = "robot_mounted_manipulation" if manipulation_cam else "root_follow"' in source
    assert "camera_mode=pov_camera_mode" in source
    assert '"true_robot_head_pov": bool(manipulation_cam)' in source


def test_runner_cli_dynamic_episode_termination_defaults_to_contract_driven() -> None:
    default_args = M.build_arg_parser().parse_args(["--out-dir", "/tmp/review"])
    assert default_args.dynamic_episode_termination is None
    assert default_args.dynamic_episode_check_every == 1
    assert default_args.capture_every == 1
    assert default_args.no_placement_topdown_capture is False

    enabled_args = M.build_arg_parser().parse_args([
        "--out-dir",
        "/tmp/review",
        "--dynamic-episode-termination",
        "--dynamic-episode-check-every",
        "2",
        "--capture-every",
        "4",
        "--no-placement-topdown-capture",
    ])
    assert enabled_args.dynamic_episode_termination is True
    assert enabled_args.dynamic_episode_check_every == 2
    assert enabled_args.capture_every == 4
    assert enabled_args.no_placement_topdown_capture is True

    disabled_args = M.build_arg_parser().parse_args([
        "--out-dir",
        "/tmp/review",
        "--no-dynamic-episode-termination",
    ])
    assert disabled_args.dynamic_episode_termination is False


def test_parse_scenarios_preserves_explicit_task_success_contract() -> None:
    parsed = M.parse_scenarios({
        "scenarios": [
            {
                "scenario_id": "reach_handle",
                "description": "Reach toward the faucet handle.",
                "task_success_contract": "visible_reach_to_affordance",
                "task_id": "sink_faucet_reach",
                "target_object_ids": ["sink", "basin"],
                "affordance_object_ids": ["faucet", "handle"],
            }
        ]
    })

    assert parsed[0]["task_target_deferred"] is True
    assert parsed[0]["task_success_contract"] == "visible_reach_to_affordance"
    assert parsed[0]["task_id"] == "sink_faucet_reach"


def test_parse_scenarios_preserves_explicit_task_affordance_and_bounds() -> None:
    parsed = M.parse_scenarios({
        "scenarios": [
            {
                "scenario_id": "microwave_reach",
                "description": "Reach toward the microwave door handle.",
                "task_target_position_xyz": [-1.721881, 1.512216, 1.242495],
                "task_affordance_xyz": [-1.591312, 1.471274, 1.241574],
                "target_object_bbox_min_xyz": [-1.864127, 1.205936, 1.056935],
                "target_object_bbox_max_xyz": [-1.579635, 1.818496, 1.428055],
            }
        ]
    })

    scenario = parsed[0]
    assert scenario["task_target_deferred"] is True
    assert scenario["task_target_position_xyz"] == [-1.721881, 1.512216, 1.242495]
    assert scenario["task_affordance_xyz"] == [-1.591312, 1.471274, 1.241574]
    assert scenario["target_object_bbox_min_xyz"] == [-1.864127, 1.205936, 1.056935]
    assert scenario["target_object_bbox_max_xyz"] == [-1.579635, 1.818496, 1.428055]


def test_root_displacement_metrics_reports_drop_and_missing_pose() -> None:
    metrics = M._root_displacement_metrics(
        {"available": True, "position_xyz": [0.0, 0.0, 1.0]},
        {"available": True, "position_xyz": [0.3, 0.4, 0.7]},
    )
    missing = M._root_displacement_metrics(
        {"available": False},
        {"available": True, "position_xyz": [0.0, 0.0, 0.7]},
    )

    assert metrics["available"] is True
    assert metrics["root_displacement_m"] == pytest.approx(math.sqrt(0.34), abs=1e-6)
    assert metrics["root_vertical_drop_m"] == pytest.approx(0.3)
    assert missing["available"] is False
    assert missing["root_displacement_m"] == 0.0
    assert missing["root_vertical_drop_m"] == 0.0


def test_dynamic_standing_contact_report_emits_integration_fields(monkeypatch) -> None:
    poses = iter(
        [
            {"available": True, "position_xyz": [0.0, 0.0, 1.0]},
            {"available": True, "position_xyz": [0.0, 0.0, 0.6]},
        ]
    )
    monkeypatch.setattr(M, "_safe_usd_root_world_pose", lambda *_args, **_kwargs: next(poses))
    monkeypatch.setattr(M, "_enable_contact_reports", lambda *_args, **_kwargs: {"status": "ok"})
    monkeypatch.setattr(M, "_contact_report_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(M, "_sim_step", lambda *_args, **_kwargs: None)

    report = M._settle_dynamic_standing_contacts(
        stage=object(),
        art_ctx={"ctx": object(), "art": None, "gravity_z": -9.81},
        robot_prim_path="/World/G1",
        root_pose=(0.0, 0.0, 1.0),
        yaw=0.0,
        phase=0.0,
        moving=False,
        settle_steps=1,
        scenario_id="scenario-a",
    )

    assert report["gravity_on"] is True
    assert report["physics_integrated"] is True
    assert report["root_vertical_drop_m"] == pytest.approx(0.4)
    assert report["dynamic_settle_verdict"] == "fell"
    assert report["tensor_view_used"] is False
    assert report["root_pose_teleport_during_physics_settle"] is False
    assert report["schema_version"] == "isaac_g1_physics_articulation_standing_contact_report.v1"


def test_physics_contact_summary_aggregates_integration_verdicts() -> None:
    reports = [
        {
            "status": "completed",
            "contact_event_count": 1,
            "support_contact_event_count": 1,
            "root_pose_teleport_during_physics_settle": False,
            "physics_integrated": True,
            "gravity_on": True,
            "root_vertical_drop_m": 0.02,
            "dynamic_settle_verdict": "stable",
        },
        {
            "status": "completed",
            "contact_event_count": 1,
            "support_contact_event_count": 1,
            "root_pose_teleport_during_physics_settle": False,
            "physics_integrated": True,
            "gravity_on": True,
            "root_vertical_drop_m": 0.03,
            "dynamic_settle_verdict": "stable",
        },
    ]

    summary = M.summarize_physics_articulation_contact_reports(reports)

    assert summary["any_physics_integrated"] is True
    assert summary["gravity_on_all"] is True
    assert summary["max_root_vertical_drop_m"] == pytest.approx(0.03)
    assert summary["verdict_counts"]["stable"] == 2


def test_pd_leg_joint_efforts_matches_mujoco_law() -> None:
    tau_static = M._pd_leg_joint_efforts([0.1], [0.0], [0.0], kp=100.0, kd=2.0)
    tau_damped = M._pd_leg_joint_efforts([0.1], [0.0], [1.0], kp=100.0, kd=2.0)

    assert tau_static.tolist() == pytest.approx([10.0])
    assert tau_damped.tolist() == pytest.approx([8.0])
    assert str(tau_static.dtype) == "float32"


def test_author_target_contact_material_scopes_authoring_to_target(monkeypatch) -> None:
    target_path = "/World/Kitchen/Sink054"
    target_mutations: set[str] = set()
    material_values: dict[str, float] = {}
    bind_purposes: list[str] = []

    class _Attr:
        def __init__(self, name: str) -> None:
            self.name = name

        def Set(self, value):
            material_values[self.name] = value

    class _Prim:
        def __init__(self, path: str) -> None:
            self.path = path

        def IsValid(self) -> bool:
            return True

        def GetPath(self):
            return self.path

    class _Stage:
        def __init__(self) -> None:
            self.prim = _Prim(target_path)
            self.material_paths: list[str] = []

        def GetPrimAtPath(self, path: str):
            return self.prim if path == target_path else None

    class _MassAPI:
        def __init__(self, prim) -> None:
            self.prim = prim

        @staticmethod
        def Apply(prim):
            target_mutations.add(str(prim.GetPath()))
            return _MassAPI(prim)

        def CreateMassAttr(self, _value=None):
            return _Attr("mass")

        def CreateDensityAttr(self, _value=None):
            return _Attr("density")

    class _MaterialAPI:
        @staticmethod
        def Apply(_prim):
            return _MaterialAPI()

        def CreateStaticFrictionAttr(self, _value=None):
            return _Attr("static_friction")

        def CreateDynamicFrictionAttr(self, _value=None):
            return _Attr("dynamic_friction")

        def CreateRestitutionAttr(self, _value=None):
            return _Attr("restitution")

    class _MeshCollisionAPI:
        def __init__(self, prim) -> None:
            self.prim = prim

        @staticmethod
        def Apply(prim):
            target_mutations.add(str(prim.GetPath()))
            return _MeshCollisionAPI(prim)

        def CreateApproximationAttr(self):
            return _Attr("approximation")

    class _MaterialObject:
        def __init__(self, prim) -> None:
            self.prim = prim

        def GetPrim(self):
            return self.prim

    class _Material:
        @staticmethod
        def Define(stage, path: str):
            stage.material_paths.append(path)
            return _MaterialObject(_Prim(path))

    class _BindingAPI:
        def __init__(self, prim) -> None:
            self.prim = prim

        @staticmethod
        def Apply(prim):
            target_mutations.add(str(prim.GetPath()))
            return _BindingAPI(prim)

        def Bind(self, _material, materialPurpose=None, purpose=None):
            bind_purposes.append(str(materialPurpose or purpose))

    fake_usd_physics = types.SimpleNamespace(
        MassAPI=_MassAPI,
        MaterialAPI=_MaterialAPI,
        MeshCollisionAPI=_MeshCollisionAPI,
        Tokens=types.SimpleNamespace(convexDecomposition="convexDecomposition"),
    )
    fake_usd_shade = types.SimpleNamespace(
        Material=_Material,
        MaterialBindingAPI=_BindingAPI,
        Tokens=types.SimpleNamespace(physics="physics"),
    )
    fake_pxr = types.SimpleNamespace(UsdPhysics=fake_usd_physics, UsdShade=fake_usd_shade)
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    monkeypatch.setitem(sys.modules, "pxr.UsdPhysics", fake_usd_physics)
    monkeypatch.setitem(sys.modules, "pxr.UsdShade", fake_usd_shade)

    diag = M._author_target_contact_material(
        _Stage(),
        target_path,
        friction=0.7,
        restitution=0.1,
        mass=3.0,
        density=12.0,
    )

    assert diag["status"] == "authored"
    assert diag["target_prim_path"] == target_path
    assert set(diag["mutated_prim_paths"]) == {target_path}
    assert target_mutations == {target_path}
    assert material_values["mass"] == pytest.approx(3.0)
    assert material_values["density"] == pytest.approx(12.0)
    assert material_values["static_friction"] == pytest.approx(0.7)
    assert material_values["dynamic_friction"] == pytest.approx(0.7)
    assert material_values["restitution"] == pytest.approx(0.1)
    assert material_values["approximation"] == "convexDecomposition"
    assert bind_purposes == ["physics"]
    assert diag["bind_purpose"] == "physics"


def test_effort_drive_args_and_result_defaults() -> None:
    defaults = M.build_arg_parser().parse_args(["--out-dir", "out"])
    flagged = M.build_arg_parser().parse_args([
        "--out-dir",
        "out",
        "--effort-drive",
        "--author-target-contact-material",
    ])

    assert defaults.effort_drive is False
    assert defaults.author_target_contact_material is False
    assert flagged.effort_drive is True
    assert flagged.author_target_contact_material is True
    assert M.run_scenarios.__kwdefaults__["effort_drive"] is False
    assert M.run_scenarios.__kwdefaults__["torque_drive"] is False
    assert M.run_scenarios.__kwdefaults__["author_target_contact_material"] is False

    res = M.build_result(
        scenarios=[{"scenario_id": "s0"}],
        outcomes=[{"task_success": True}],
        policy_id="p",
        kitchen_usd="k.usd",
        g1_usd="g.usd",
        blockers=[],
    )
    assert res["actuator_output_mode"] == "position_target"
    assert "authored_target_contact_material" not in res


def test_effort_drive_fails_closed_without_tensor_view(monkeypatch) -> None:
    poses = iter(
        [
            {"available": True, "position_xyz": [0.0, 0.0, 1.0]},
            {"available": True, "position_xyz": [0.0, 0.0, 1.0]},
        ]
    )
    monkeypatch.setattr(M, "_safe_usd_root_world_pose", lambda *_args, **_kwargs: next(poses))
    monkeypatch.setattr(M, "_enable_contact_reports", lambda *_args, **_kwargs: {"status": "ok"})
    monkeypatch.setattr(M, "_contact_report_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(M, "_sim_step", lambda *_args, **_kwargs: None)

    report = M._settle_dynamic_standing_contacts(
        stage=object(),
        art_ctx={"ctx": object(), "art": None, "gravity_z": -9.81},
        robot_prim_path="/World/G1",
        root_pose=(0.0, 0.0, 1.0),
        yaw=0.0,
        phase=0.0,
        moving=False,
        settle_steps=1,
        scenario_id="scenario-effort-fallback",
        effort_drive=True,
    )

    assert report["status"] == "completed"
    assert report["joint_command_mode"] == "usd_physx_articulation_default_drives_no_tensor_view"
    assert report["actuator_output_mode"] == "position_target_fallback"
    assert "effort_drive_requested_without_tensor_view" in report["effort_drive_blockers"]


def test_effort_drive_uses_joint_efforts_with_live_tensor_view(monkeypatch) -> None:
    class _Art:
        def __init__(self) -> None:
            self.efforts: list[list[float]] = []

        def get_joint_positions(self):
            return [0.0]

        def get_joint_velocities(self):
            return [0.0]

    art = _Art()
    poses = iter(
        [
            {"available": True, "position_xyz": [0.0, 0.0, 1.0]},
            {"available": True, "position_xyz": [0.0, 0.0, 1.0]},
        ]
    )
    monkeypatch.setattr(M, "_safe_articulation_world_pose", lambda *_args, **_kwargs: next(poses))
    monkeypatch.setattr(M, "_enable_contact_reports", lambda *_args, **_kwargs: {"status": "ok"})
    monkeypatch.setattr(M, "_contact_report_records", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(M, "_sim_step", lambda *_args, **_kwargs: None)

    def fake_apply(fake_art, efforts):
        fake_art.efforts.append([float(v) for v in efforts])
        return "articulation_action_joint_efforts"

    monkeypatch.setattr(M, "_apply_articulation_joint_efforts", fake_apply)

    report = M._settle_dynamic_standing_contacts(
        stage=object(),
        art_ctx={
            "ctx": object(),
            "art": art,
            "gravity_z": -9.81,
            "default": [0.0],
            "dof_index": {},
        },
        robot_prim_path="/World/G1",
        root_pose=(0.0, 0.0, 1.0),
        yaw=0.0,
        phase=0.0,
        moving=False,
        settle_steps=1,
        scenario_id="scenario-effort",
        effort_drive=True,
        effort_kp=100.0,
        effort_kd=2.0,
    )

    assert report["joint_command_mode"] == "articulation_action_joint_efforts"
    assert report["actuator_output_mode"] == "effort"
    assert report["effort_drive_blockers"] == []
    assert art.efforts


def test_build_result_adds_gravity_integration_boundary_only_when_integrated() -> None:
    integrated_report = {
        "scenario_id": "sink_stand",
        "status": "completed",
        "contact_event_count": 2,
        "support_contact_event_count": 1,
        "root_pose_teleport_during_physics_settle": False,
        "physics_integrated": True,
        "gravity_on": True,
        "root_vertical_drop_m": 0.02,
        "dynamic_settle_verdict": "stable",
    }
    unintegrated_report = {**integrated_report, "physics_integrated": False}
    base = {
        "scenarios": [{"scenario_id": "sink_stand"}],
        "outcomes": [{"task_success": True}],
        "policy_id": "blueprint_default_walk_to_target_smoke_policy",
        "kitchen_usd": "k.usd",
        "g1_usd": "g1.usd",
        "blockers": [],
    }

    integrated = M.build_result(
        **base,
        physics_articulation_contact_reports=[integrated_report],
    )
    unintegrated = M.build_result(
        **base,
        physics_articulation_contact_reports=[unintegrated_report],
    )

    assert "integrated under gravity" in integrated["proof_boundary"]
    assert "max vertical drop 0.020 m" in integrated["proof_boundary"]
    assert "integrated under gravity" not in unintegrated["proof_boundary"]


def test_build_result_keeps_dynamic_standing_contacts_bounded() -> None:
    res = M.build_result(
        scenarios=[{"scenario_id": "sink_stand"}],
        outcomes=[{"task_success": True}],
        policy_id="blueprint_default_walk_to_target_smoke_policy",
        kitchen_usd="k.usd",
        g1_usd="g1.usd",
        blockers=[],
        physics_articulation_contact_reports=[
            {
                "scenario_id": "sink_stand",
                "status": "completed",
                "contact_event_count": 2,
                "support_contact_event_count": 1,
                "root_pose_teleport_during_physics_settle": False,
            }
        ],
    )
    summary = res["physics_articulation_standing_contact_summary"]
    assert summary["completed_scenario_count"] == 1
    assert summary["all_have_support_contact_evidence"] is True
    assert summary["root_pose_teleport_during_physics_settle"] is False
    assert "standing/contact settle" in res["proof_boundary"]
    assert "not full dynamic locomotion" in res["proof_boundary"].lower()
    assert "not deployment readiness" in res["proof_boundary"].lower()


def test_summarize_physics_articulation_contact_reports_fails_closed_on_missing_support() -> None:
    summary = M.summarize_physics_articulation_contact_reports([
        {
            "status": "completed",
            "contact_event_count": 1,
            "support_contact_event_count": 0,
            "root_pose_teleport_during_physics_settle": False,
        }
    ])
    assert summary["all_completed"] is True
    assert summary["all_have_support_contact_evidence"] is False
    assert "not prove full dynamic locomotion" in summary["claim_boundary"]


def test_build_result_does_not_claim_support_contact_when_none_observed() -> None:
    res = M.build_result(
        scenarios=[{"scenario_id": "floor_stand"}],
        outcomes=[{"task_success": False}],
        policy_id="blueprint_default_walk_to_target_smoke_policy",
        kitchen_usd="k.usd",
        g1_usd="g1.usd",
        blockers=[],
        physics_articulation_contact_reports=[
            {
                "scenario_id": "floor_stand",
                "status": "completed",
                "contact_event_count": 0,
                "support_contact_event_count": 0,
                "root_pose_teleport_during_physics_settle": False,
            }
        ],
    )
    summary = res["physics_articulation_standing_contact_summary"]
    assert summary["all_completed"] is True
    assert summary["all_have_support_contact_evidence"] is False
    assert "support-contact events were not observed" in res["proof_boundary"]
    assert "does not prove support contact" in res["proof_boundary"]


def test_build_result_blocks_on_blockers() -> None:
    res = M.build_result(scenarios=[], outcomes=[], policy_id="p", kitchen_usd="k", g1_usd=None,
                         blockers=["official_isaac_unitree_g1_articulation_api_unverified"])
    assert res["status"] == "blocked"


def test_build_result_blocks_zero_outcomes_with_explicit_reason() -> None:
    res = M.build_result(
        scenarios=[{"scenario_id": "sink_faucet"}],
        outcomes=[],
        policy_id="blueprint_default_walk_to_target_smoke_policy",
        kitchen_usd="k.usd",
        g1_usd="g1.usd",
        blockers=[],
    )

    assert res["status"] == "blocked"
    assert res["scenarios_executed"] == 0
    assert "scenario_execution_returned_no_outcomes" in res["blockers"]


def _rotate_by_quat(q, v):
    # rotate vector v by quaternion q=(w,x,y,z)
    w, x, y, z = q
    # q * (0,v) * q^-1, expanded
    tx, ty, tz = v
    # vector part of q
    ux, uy, uz = x, y, z
    # t = 2 * cross(u, v)
    cx, cy, cz = (uy * tz - uz * ty, uz * tx - ux * tz, ux * ty - uy * tx)
    cx, cy, cz = 2 * cx, 2 * cy, 2 * cz
    # v + w*t + cross(u, t)
    c2 = (uy * cz - uz * cy, uz * cx - ux * cz, ux * cy - uy * cx)
    return (tx + w * cx + c2[0], ty + w * cy + c2[1], tz + w * cz + c2[2])


def test_look_at_quat_points_camera_minus_z_at_target() -> None:
    eye, target = (5.0, 0.0, 1.0), (0.0, 0.0, 1.0)
    q = M.look_at_quat(eye, target)
    # USD camera views along local -Z; rotated -Z should point from eye toward target (-X here)
    view = _rotate_by_quat(q, (0.0, 0.0, -1.0))
    expected = M._norm((target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]))
    assert view[0] == pytest.approx(expected[0], abs=1e-6)
    assert view[1] == pytest.approx(expected[1], abs=1e-6)
    assert view[2] == pytest.approx(expected[2], abs=1e-6)


def test_scene_framing_center_and_radius() -> None:
    scs = [{"route_points": [[0, 0, 0.79], [2, 0, 0.79]]},
           {"route_points": [[0, 2, 0.79], [2, 2, 0.79]]}]
    center, radius = M.scene_framing(scs)
    assert center[0] == pytest.approx(1.0) and center[1] == pytest.approx(1.0)
    assert radius >= 1.0


def test_project_point_to_pixel() -> None:
    eye, target, up = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)  # looking +X
    r = M.project_point_to_pixel((5.0, 0.0, 0.0), eye, target, up, 60.0, 640, 480)
    assert r is not None
    u, v, z = r
    assert abs(u - 320) < 1e-3 and abs(v - 240) < 1e-3 and abs(z - 5.0) < 1e-6  # on-axis -> center
    assert M.project_point_to_pixel((-5.0, 0.0, 0.0), eye, target, up, 60.0, 640, 480) is None  # behind
    up_pt = M.project_point_to_pixel((5.0, 0.0, 1.0), eye, target, up, 60.0, 640, 480)
    assert up_pt is not None and up_pt[1] < 240  # +Z world -> above image center (smaller v)
    # far off-axis -> out of frame -> None
    assert M.project_point_to_pixel((0.05, 50.0, 0.0), eye, target, up, 60.0, 640, 480) is None


def test_manipulation_pov_geometry_requires_seed_arm_chain_and_effector_in_frame() -> None:
    eye, target = (0.0, 0.0, 1.0), (1.0, 0.0, 1.0)
    affordance = (1.0, 0.0, 1.0)
    visible = M._manipulation_pov_geometry(
        arm_points={
            "shoulder": (0.55, -0.05, 1.02),
            "elbow": (0.66, -0.05, 1.02),
            "wrist": (0.78, -0.03, 1.01),
            "hand": (0.86, -0.02, 1.0),
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=68.0,
        width=640,
        height=480,
        arm="right",
    )
    assert visible["status"] == "PASS"
    assert {"elbow", "wrist", "hand"}.issubset(set(visible["arm_roles_in_frame"]))
    assert visible["effector_distance_is_metadata_only"] is False
    assert visible["reach_feasibility"]["status"] == "PASS"
    assert visible["effector_distance_to_affordance_m"]["hand"] > 0.1

    too_far_affordance = M._manipulation_pov_geometry(
        arm_points={
            "shoulder": (0.55, -0.05, 1.02),
            "elbow": (0.66, -0.05, 1.02),
            "wrist": (0.78, -0.03, 1.01),
            "hand": (0.86, -0.02, 1.0),
        },
        affordance=(1.0, 0.55, 1.0),
        eye=eye,
        target=target,
        vfov_deg=95.0,
        width=640,
        height=480,
        arm="right",
    )
    assert too_far_affordance["status"] == "FAIL"
    assert too_far_affordance["target_in_frame"] is True
    assert too_far_affordance["arm_extension"]["status"] == "PASS"
    assert too_far_affordance["arm_extension"]["horizontal_extension_ratio"] > 0.35
    assert too_far_affordance["effector_distance_is_metadata_only"] is False
    assert "alignment_to_affordance_direction" not in too_far_affordance["arm_extension"]
    assert too_far_affordance["reach_feasibility"]["status"] == "FAIL"
    assert too_far_affordance["reach_feasibility"]["required_for_seed_geometry"] is True
    assert "manipulation_pov_effector_too_far_from_affordance" in too_far_affordance[
        "reach_feasibility"
    ]["blockers"]
    assert "manipulation_pov_effector_too_far_from_affordance" in too_far_affordance[
        "blockers"
    ]

    both_visible = M._manipulation_pov_geometry(
        arm_points={},
        arm_points_by_arm={
            "left": {
                "shoulder": (0.55, 0.18, 1.02),
                "elbow": (0.66, 0.16, 1.02),
                "wrist": (0.78, 0.14, 1.01),
                "hand": (0.86, 0.12, 1.0),
            },
            "right": {
                "shoulder": (0.55, -0.18, 1.02),
                "elbow": (0.66, -0.16, 1.02),
                "wrist": (0.78, -0.14, 1.01),
                "hand": (0.86, -0.12, 1.0),
            },
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=68.0,
        width=640,
        height=480,
        arm="both",
    )
    assert both_visible["status"] == "PASS"
    assert both_visible["required_arms"] == ["left", "right"]
    assert set(both_visible["per_arm_geometry"]) == {"left", "right"}
    assert both_visible["arm_extension"]["status"] == "PASS"
    assert both_visible["reach_feasibility"]["status"] == "PASS"
    assert both_visible["reach_feasibility"]["passing_arms"] == ["left", "right"]
    assert both_visible["two_arm_coordination"]["status"] == "PASS"

    both_converged = M._manipulation_pov_geometry(
        arm_points={},
        arm_points_by_arm={
            "left": {
                "shoulder": (0.55, 0.18, 1.02),
                "elbow": (0.66, 0.08, 1.02),
                "wrist": (0.80, 0.04, 1.01),
                "hand": (0.90, 0.02, 1.0),
            },
            "right": {
                "shoulder": (0.55, -0.18, 1.02),
                "elbow": (0.66, -0.08, 1.02),
                "wrist": (0.80, -0.04, 1.01),
                "hand": (0.90, -0.02, 1.0),
            },
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=68.0,
        width=640,
        height=480,
        arm="both",
    )
    assert both_converged["status"] == "FAIL"
    assert both_converged["two_arm_coordination"]["status"] == "FAIL"
    assert (
        "manipulation_pov_both_arms_converge_at_single_affordance"
        in both_converged["blockers"]
    )

    both_crossed = M._manipulation_pov_geometry(
        arm_points={},
        arm_points_by_arm={
            "left": {
                "shoulder": (0.55, 0.18, 1.02),
                "elbow": (0.66, 0.08, 1.02),
                "wrist": (0.78, -0.10, 1.01),
                "hand": (0.86, -0.16, 1.0),
            },
            "right": {
                "shoulder": (0.55, -0.18, 1.02),
                "elbow": (0.66, -0.08, 1.02),
                "wrist": (0.78, 0.10, 1.01),
                "hand": (0.86, 0.16, 1.0),
            },
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=68.0,
        width=640,
        height=480,
        arm="both",
    )
    assert both_crossed["status"] == "FAIL"
    assert both_crossed["two_arm_coordination"]["status"] == "FAIL"
    assert "manipulation_pov_both_arms_cross_midline" in both_crossed["blockers"]

    one_arm_reachable = M._manipulation_pov_geometry(
        arm_points={},
        arm_points_by_arm={
            "left": {
                "shoulder": (0.55, 0.45, 1.02),
                "elbow": (0.66, 0.45, 1.02),
                "wrist": (0.78, 0.43, 1.01),
                "hand": (0.86, 0.42, 1.0),
            },
            "right": {
                "shoulder": (0.55, -0.05, 1.02),
                "elbow": (0.66, -0.05, 1.02),
                "wrist": (0.78, -0.03, 1.01),
                "hand": (0.86, -0.02, 1.0),
            },
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=95.0,
        width=640,
        height=480,
        arm="both",
    )
    assert one_arm_reachable["seed_arm_visibility"]["status"] == "PASS"
    assert one_arm_reachable["arm_extension"]["status"] == "PASS"
    assert one_arm_reachable["reach_feasibility"]["status"] == "PASS"
    assert one_arm_reachable["reach_feasibility"]["passing_arms"] == ["right"]
    assert "manipulation_pov_affordance_outside_g1_reach_envelope" not in one_arm_reachable["blockers"]

    right_only_for_both = M._manipulation_pov_geometry(
        arm_points={},
        arm_points_by_arm={
            "right": {
                "shoulder": (0.55, -0.05, 1.02),
                "elbow": (0.66, -0.05, 1.02),
                "wrist": (0.78, -0.03, 1.01),
                "hand": (0.86, -0.02, 1.0),
            },
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=68.0,
        width=640,
        height=480,
        arm="both",
    )
    assert right_only_for_both["status"] == "FAIL"
    assert "manipulation_pov_left_arm_seed_failed" in right_only_for_both["blockers"]

    hand_only = M._manipulation_pov_geometry(
        arm_points={
            "hand": (0.82, -0.02, 1.0),
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=68.0,
        width=640,
        height=480,
        arm="right",
    )
    assert hand_only["status"] == "FAIL"
    assert "manipulation_pov_arm_chain_not_in_frame" in hand_only["blockers"]

    cropped = M._manipulation_pov_geometry(
        arm_points={
            "elbow": (-0.4, -0.05, 1.02),
            "wrist": (-0.3, -0.03, 1.01),
            "hand": (-0.2, -0.02, 1.0),
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=68.0,
        width=640,
        height=480,
        arm="right",
    )
    assert cropped["status"] == "FAIL"
    assert "manipulation_pov_arm_not_in_frame" in cropped["blockers"]

    visible_but_hanging = M._manipulation_pov_geometry(
        arm_points={
            "shoulder": (0.3, -0.05, 1.2),
            "elbow": (0.3, -0.05, 1.0),
            "wrist": (0.3, -0.03, 0.82),
            "hand": (0.3, -0.02, 0.7),
        },
        affordance=affordance,
        eye=eye,
        target=target,
        vfov_deg=68.0,
        width=640,
        height=480,
        arm="right",
    )
    assert visible_but_hanging["status"] == "FAIL"
    assert "manipulation_pov_arm_not_extended_forward" in visible_but_hanging["blockers"]
    assert "manipulation_pov_effector_too_far_from_affordance" in visible_but_hanging[
        "reach_feasibility"
    ]["blockers"]
    assert "manipulation_pov_effector_too_far_from_affordance" in visible_but_hanging[
        "blockers"
    ]


def test_pov_seed_frame_quality_rejects_black_edge_occlusion(tmp_path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw  # type: ignore

    clean = tmp_path / "clean.png"
    clean_img = Image.new("RGB", (640, 480), (185, 185, 185))
    ImageDraw.Draw(clean_img).rectangle((260, 120, 380, 330), fill=(8, 8, 8))
    clean_img.save(clean)
    clean_report = M._pov_seed_frame_quality(clean)
    assert clean_report["status"] == "PASS"

    occluded = tmp_path / "occluded.png"
    occ_img = Image.new("RGB", (640, 480), (185, 185, 185))
    ImageDraw.Draw(occ_img).rectangle((520, 0, 640, 480), fill=(0, 0, 0))
    occ_img.save(occluded)
    occ_report = M._pov_seed_frame_quality(occluded)
    assert occ_report["status"] == "FAIL"
    assert "manipulation_pov_edge_self_occlusion" in occ_report["blockers"]


def test_follow_cam_is_behind_and_above_robot() -> None:
    eye, target = M.follow_cam_pose((0.0, 0.0, 0.79), 0.0)  # facing +X
    assert eye[0] < 0.0           # behind the robot along -X
    assert eye[2] > 0.79          # above the root
    assert target[0] > 0.0        # looking ahead toward +X


def test_verify_cam_pose_is_behind_robot_for_visual_placement_qc() -> None:
    root = (-1.037147, 0.655166, 0.84)
    yaw = math.pi
    look_at = (-1.437147, 0.655166, 1.025963)
    eye, target = M.verify_cam_pose(root, yaw, look_at=look_at)
    fx, fy = math.cos(yaw), math.sin(yaw)
    px, py = -fy, fx
    from_root = (eye[0] - root[0], eye[1] - root[1])
    behind_m = -(from_root[0] * fx + from_root[1] * fy)
    side_m = abs(from_root[0] * px + from_root[1] * py)

    assert behind_m > side_m
    assert side_m > 0.2
    assert eye[2] > root[2] + 0.8
    assert target[2] > root[2]


def _fake_scene_index(monkeypatch, objects, *, obstacle_boxes=None):
    """Patch scene_placement's USD index to enumerate a preset object list (no pxr/GPU)."""
    sp = importlib.import_module("blueprint_pipeline.scene_placement")

    class _FakeIndex:
        def __init__(self, **kw):  # accepts stage=... / usd_path=...
            pass

        def objects(self):
            return list(objects)

        def obstacle_boxes(self):
            return list(objects if obstacle_boxes is None else obstacle_boxes)

    monkeypatch.setattr(sp, "UsdSceneSpatialIndex", _FakeIndex)
    return sp


def test_resolve_task_target_via_scene_placement_maps_task_to_object(monkeypatch) -> None:
    # No object id, no coords — just a natural-language task. The runner enumerates the scene via
    # scene_placement and maps "turn on the faucet" onto the faucet object's center. No hardcoding.
    sp = importlib.import_module("blueprint_pipeline.scene_placement")
    SceneObject = sp.SceneObject
    objs = [
        SceneObject(id="faucet_1", label="faucet", bbox_min=(2.4, 1.0, 0.9),
                    bbox_max=(2.6, 1.3, 1.1), centroid=(2.5, 1.15, 1.0), source="usd"),
        SceneObject(id="stove_1", label="stove", bbox_min=(0.0, 0.0, 0.0),
                    bbox_max=(0.6, 0.6, 0.9), centroid=(0.3, 0.3, 0.45), source="usd"),
    ]
    _fake_scene_index(monkeypatch, objs)

    res = M._resolve_task_target_via_scene_placement(
        stage=object(), scenario={"description": "Stand at the sink and turn on the faucet."}
    )
    assert res is not None and res["status"] == "resolved"
    assert res["source"] == "scene_placement_task_label"
    assert res["selected"]["target_object_id"] == "faucet_1"      # task -> faucet, not stove
    assert res["selected"]["center_xyz"] == [2.5, 1.15, 1.0]      # dynamic center from the scene


def test_resolve_task_target_via_scene_placement_blocks_on_no_match(monkeypatch) -> None:
    sp = importlib.import_module("blueprint_pipeline.scene_placement")
    SceneObject = sp.SceneObject
    objs = [SceneObject(id="rug_1", label="rug", bbox_min=(0, 0, 0),
                        bbox_max=(1, 1, 0.1), centroid=(0.5, 0.5, 0.05), source="usd")]
    _fake_scene_index(monkeypatch, objs)
    res = M._resolve_task_target_via_scene_placement(
        stage=object(), scenario={"task": "turn on the faucet"}
    )
    assert res is not None and res["status"] == "blocked"
    assert "scene_placement_no_task_match" in res["blockers"]


def test_resolve_task_target_via_scene_placement_returns_none_without_task() -> None:
    # No task description at all -> nothing to resolve; defer to the id-driven path (None).
    assert M._resolve_task_target_via_scene_placement(stage=object(), scenario={}) is None


def test_scene_placement_stand_plan_stands_on_clear_floor_when_probe_blocks_wall() -> None:
    # When the probe DOES see the counter/wall (blocks y >= 1.0), it stands on the open floor in front.
    tr = {"status": "resolved", "source": "scene_placement_task_label",
          "selected": {"target_object_id": "sink", "target_object_label": "sink",
                       "center_xyz": [2.28, 1.33, 1.0], "size_xyz": [0.4, 0.4, 0.3]}}
    plan = M._scene_placement_stand_plan(tr, lambda p, y: 0 if p[1] < 1.0 else 1, floor_z=0.05)
    assert plan["status"] == "accepted" and plan["accepted_pose"][1] < 1.0


def test_scene_placement_stand_plan_none_without_geometry() -> None:
    # A resolution lacking center/size -> None, so the caller falls back to plan_task_stance.
    assert M._scene_placement_stand_plan({"selected": {"center_xyz": [1, 2, 3]}}, lambda p, y: 0) is None
    assert M._scene_placement_stand_plan(None, lambda p, y: 0) is None


def test_topdown_debug_overlay_is_added_only_after_verify_and_pov_are_saved() -> None:
    source = _RUNNER.read_text()
    capture_start = source.index("debug_root_path = (")
    normal_render = source.index("_replicator_step_with_watchdog(", capture_start)
    pov_save = source.index("pov_ok = _save_rgb(pov_annot, pov_frame_path", capture_start)
    verify_save = source.index("verify_ok = _save_rgb(", capture_start)
    overlay_update = source.index("_update_topdown_debug_scene(", capture_start)
    topdown_save = source.index("placement_topdown_frame_path =", overlay_update)
    overlay_remove = source.index("stage.RemovePrim(debug_root_path)", topdown_save)

    assert normal_render < pov_save < overlay_update
    assert normal_render < verify_save < overlay_update
    assert overlay_update < topdown_save < overlay_remove


def test_first_frame_warmup_runs_after_pov_camera_placement() -> None:
    source = _RUNNER.read_text()
    capture_start = source.index("if should_capture_step:")
    pov_place = source.index("_place_camera(stage, pov_cam, eye, tgt)", capture_start)
    warmup_log = source.index("first-frame warmup", pov_place)
    warmup_step = source.index('label=f"{sid}:frame:{cap}:warmup:{wi}"', warmup_log)
    frame_step = source.index('label=f"{sid}:frame:{cap}:rt_subframes:{capture_rt_subframes}"', warmup_step)
    pov_save = source.index("pov_ok = _save_rgb(pov_annot, pov_frame_path", frame_step)

    assert 'label=f"{sid}:warmup:{wi}"' not in source
    assert pov_place < warmup_log < warmup_step < frame_step < pov_save


def test_auto_render_settle_is_only_for_repeated_or_review_render(monkeypatch) -> None:
    monkeypatch.delenv("PARITY_AUTO_RENDER_SETTLE_SECONDS", raising=False)

    assert M._auto_render_settle_seconds(
        configured_settle_seconds=0,
        no_collision_probe=True,
        manipulation_cam=True,
        verify_cam=False,
        manipulation_stand=False,
        warmup_frames=6,
        render_subframes=4,
    ) == 0
    assert M._auto_render_settle_seconds(
        configured_settle_seconds=12,
        no_collision_probe=False,
        manipulation_cam=False,
        verify_cam=False,
        manipulation_stand=False,
        warmup_frames=1,
        render_subframes=1,
    ) == 12
    assert M._auto_render_settle_seconds(
        configured_settle_seconds=0,
        no_collision_probe=False,
        manipulation_cam=False,
        verify_cam=False,
        manipulation_stand=False,
        warmup_frames=1,
        render_subframes=1,
    ) == 0
    assert M._auto_render_settle_seconds(
        configured_settle_seconds=0,
        no_collision_probe=False,
        manipulation_cam=True,
        verify_cam=False,
        manipulation_stand=False,
        warmup_frames=1,
        render_subframes=1,
    ) == 60

    monkeypatch.setenv("PARITY_AUTO_RENDER_SETTLE_SECONDS", "7.5")
    assert M._auto_render_settle_seconds(
        configured_settle_seconds=0,
        no_collision_probe=False,
        manipulation_cam=False,
        verify_cam=True,
        manipulation_stand=False,
        warmup_frames=1,
        render_subframes=1,
    ) == 7.5


def test_placement_obstacles_use_fine_boxes_not_grouped_cabinet_slab(monkeypatch) -> None:
    sp = importlib.import_module("blueprint_pipeline.scene_placement")
    SceneObject = sp.SceneObject
    sink = SceneObject(
        id="sink",
        label="sink",
        bbox_min=(2.009, 0.896, 0.555),
        bbox_max=(2.547, 1.770, 1.142),
        centroid=(2.278, 1.333, 0.849),
        source="usd",
    )
    broad_cabinet = SceneObject(
        id="kitchen_cabinet_1",
        label="kitchen_cabinet",
        bbox_min=(-1.300, -0.148, -0.014),
        bbox_max=(2.568, 2.459, 0.836),
        centroid=(0.634, 1.156, 0.411),
        source="usd",
    )
    fine_cabinet_back_run = SceneObject(
        id="kitchen_cabinet_leaf_back",
        label="kitchen_cabinet",
        bbox_min=(-1.300, 1.850, -0.014),
        bbox_max=(2.568, 2.459, 0.836),
        centroid=(0.634, 2.155, 0.411),
        source="usd_leaf",
    )
    fine_cabinet_left_run = SceneObject(
        id="kitchen_cabinet_leaf_left",
        label="kitchen_cabinet",
        bbox_min=(-1.300, -0.148, -0.014),
        bbox_max=(-0.700, 2.459, 0.836),
        centroid=(-1.000, 1.156, 0.411),
        source="usd_leaf",
    )
    dishwasher = SceneObject(
        id="dishwasher",
        label="dishwasher",
        bbox_min=(1.936, 0.267, 0.073),
        bbox_max=(2.552, 0.884, 0.751),
        centroid=(2.244, 0.575, 0.412),
        source="usd_leaf",
    )
    flower_on_counter = SceneObject(
        id="kitchen_flowers",
        label="kitchen_flowers",
        bbox_min=(1.806, -0.406, 0.794),
        bbox_max=(2.666, 0.460, 1.451),
        centroid=(2.236, 0.027, 1.122),
        source="usd_leaf",
    )
    _fake_scene_index(
        monkeypatch,
        [sink, broad_cabinet],
        obstacle_boxes=[sink, fine_cabinet_back_run, fine_cabinet_left_run, dishwasher, flower_on_counter],
    )
    monkeypatch.setattr(M, "_placement_shell_obstacles_for_stage", lambda _stage: [])
    monkeypatch.setattr(M, "_place_root", lambda _stage, _prim_path, _pose, _yaw: {"status": "placed"})
    pose = (2.366319, -0.179048, 0.84)
    monkeypatch.setattr(
        M,
        "_world_bbox_for_prim",
        lambda _stage, _prim_path: {
            "bbox_min_xyz": [pose[0] - 0.28, pose[1] - 0.28, 0.22],
            "bbox_max_xyz": [pose[0] + 0.28, pose[1] + 0.28, 1.46],
            "center_xyz": [pose[0], pose[1], pose[2]],
            "size_xyz": [0.56, 0.56, 1.24],
        },
    )

    obstacles = M._placement_obstacles_for_stage(object())
    assert {obj.id for obj in obstacles} == {
        "sink",
        "kitchen_cabinet_leaf_back",
        "kitchen_cabinet_leaf_left",
        "dishwasher",
        "kitchen_flowers",
    }
    assert broad_cabinet.id not in {obj.id for obj in obstacles}

    validator = M._placement_validator_for_stage(
        object(),
        "/World/G1",
        (sink.bbox_min, sink.bbox_max),
        target_object=sink,
        scene_objects=obstacles,
        floor_z=0.05,
    )
    result = validator(pose, 1.629212, {"standoff_from_target_surface_m": 1.0625})

    assert result["status"] == "accepted"
    assert result["blockers"] == []
    assert result["deterministic_geometry"]["ok"] is True


def test_scenario_render_quality_sets_explicit_pathtracing_spp(tmp_path, monkeypatch) -> None:
    """Explicit path-traced scenario renders must set /rtx/pathtracing/spp AND totalSpp.

    Realtime RTX is the default seed-frame path, but the explicit path-traced/audit path still
    needs the same sample budget hardening. Relying on rep.settings.set_render_pathtraced alone
    left per-frame samples starved -> grainy specular robots (hf 1.77 vs 0.58, 2026-07-02).
    """
    import types as _types

    monkeypatch.setenv("PARITY_RENDER_QUALITY_MODE", "pathtraced")
    set_calls: list = []

    class _Settings:
        def set(self, path, value):
            set_calls.append((path, value))

    fake_carb = _types.ModuleType("carb")
    fake_carb.settings = _types.SimpleNamespace(get_settings=lambda: _Settings())
    monkeypatch.setitem(sys.modules, "carb", fake_carb)

    class _RepSettings:
        def set_render_pathtraced(self, samples_per_pixel):
            set_calls.append(("rep.set_render_pathtraced", samples_per_pixel))

    fake_rep = _types.SimpleNamespace(settings=_RepSettings())

    diag = M._apply_render_quality_settings(
        fake_rep,
        render_subframes=16,
        manipulation_cam=True,
        verify_cam=True,
        out_dir=tmp_path,
    )
    assert diag["use_pathtraced"] is True
    spp = diag["samples_per_pixel"]
    assert ("/rtx/pathtracing/spp", spp) in set_calls
    assert ("/rtx/pathtracing/totalSpp", spp) in set_calls


def test_capture_settle_steps_match_audit_recipe(monkeypatch) -> None:
    """Scenario captures must settle like the audit path: 3 extra replicator steps
    after a pose change in path-traced mode (the audit's empirically clean recipe;
    a single step per capture rendered hf 1.77-2.70 vs the audit's 0.58)."""
    monkeypatch.delenv("PARITY_CAPTURE_SETTLE_FRAMES", raising=False)
    assert M._capture_settle_steps({"use_pathtraced": True}) == 3
    assert M._capture_settle_steps({"use_pathtraced": False}) == 0
    assert M._capture_settle_steps(None) == 0
    monkeypatch.setenv("PARITY_CAPTURE_SETTLE_FRAMES", "5")
    assert M._capture_settle_steps({"use_pathtraced": True}) == 5
    monkeypatch.setenv("PARITY_CAPTURE_SETTLE_FRAMES", "99")
    assert M._capture_settle_steps({"use_pathtraced": True}) == 8
    monkeypatch.setenv("PARITY_CAPTURE_SETTLE_FRAMES", "junk")
    assert M._capture_settle_steps({"use_pathtraced": True}) == 3


def test_scenario_frame_loop_settles_before_capture() -> None:
    source = _RUNNER.read_text()
    assert ":frame:{cap}:settle:" in source, (
        "scenario frame loop must run audit-style settle steps before each capture"
    )


def test_software_denoise_applies_to_pathtraced_review_saves_unless_disabled(monkeypatch) -> None:
    """Non-white review-material path-traced frames can still save speckled; deterministic
    saved-frame denoise is allowed, and source QA remains the pass/fail authority."""
    monkeypatch.delenv("PARITY_SOFTWARE_DENOISE_PATH_TRACED", raising=False)
    assert M._effective_software_denoise(True, {"use_pathtraced": True}) is True
    assert M._effective_software_denoise(True, {"use_pathtraced": False}) is True
    assert M._effective_software_denoise(False, {"use_pathtraced": False}) is False
    assert M._effective_software_denoise(True, None) is True
    monkeypatch.setenv("PARITY_SOFTWARE_DENOISE_PATH_TRACED", "raw")
    assert M._effective_software_denoise(True, {"use_pathtraced": True}) is False


# --------------------------------------------------------------------------
# Feedback-driven stance-configuration search (bounded scene-configuration agent)
# --------------------------------------------------------------------------


def test_plan_task_stance_honors_custom_angle_offsets() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [1.0],
        "floor_z_hint": 0.05,
    }
    plan = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        angle_offsets_deg=(7.5,),
    )
    assert plan["status"] == "accepted"
    assert [c["angle_offset_deg"] for c in plan["candidates"]] == [7.5]


def test_staging_anchor_candidates_prefer_far_collision_free_diverse_angles() -> None:
    plan = {
        "candidates": [
            {"pose": [1.0, 0.0, 0.79], "standoff_from_target_surface_m": 0.4,
             "angle_offset_deg": 0, "scene_collision_contact_count": 2},
            {"pose": [2.0, 0.0, 0.79], "standoff_from_target_surface_m": 1.2,
             "angle_offset_deg": 0, "scene_collision_contact_count": 0},
            {"pose": [2.0, 0.3, 0.79], "standoff_from_target_surface_m": 1.2,
             "angle_offset_deg": 15, "scene_collision_contact_count": 0},
            {"pose": [1.4, -1.4, 0.79], "standoff_from_target_surface_m": 1.2,
             "angle_offset_deg": -45, "scene_collision_contact_count": 0},
        ]
    }
    anchors = M._staging_anchor_candidates_from_plan(plan)
    assert anchors
    assert all(a["source"] == "collision_free_sweep_candidate" for a in anchors)
    angles = [a["angle_offset_deg"] for a in anchors]
    assert 0 in angles and -45 in angles and 15 not in angles  # <30 deg from 0 -> skipped
    assert all(a["standoff_from_target_surface_m"] == pytest.approx(1.2) for a in anchors)


def test_adaptive_stance_search_recovers_reach_blocked_ladder() -> None:
    # Every ladder distance leaves the affordance beyond the seed reach envelope;
    # the agent must descend below the ladder using the measured shortfall and get
    # a pose accepted by the SAME sweep gates (no thresholds touched).
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "task_affordance_xyz": [0.0, 0.0, 1.0],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [0.9, 1.1],
        "floor_z_hint": 0.05,
    }
    initial = M.plan_task_stance(scenario=scenario, probe_collision=lambda pose, yaw: 0)
    assert initial["status"] == "blocked"
    assert initial["blockers"] == ["no_reach_seed_task_stance_candidate"]

    plan = M._adaptive_task_stance_search(
        stance_scenario=scenario,
        manipulation_look_at=None,
        probe=lambda pose, yaw: 0,
        placement_validator=None,
        initial_plan=initial,
    )
    assert plan is not None
    assert plan["status"] == "accepted"
    assert plan["stance_search_status"] == "accepted"
    search = plan["stance_search"]
    assert search["schema_version"] == "stance_configuration_search.v1"
    assert search["status"] == "accepted"
    assert search["round_count"] >= 2
    assert search["rounds"][0]["strategy_id"] == "initial_structured_sweep"
    assert search["rounds"][-1]["strategy_id"] in set(search["registered_strategy_ids"])
    # The accepted pose is closer than any original ladder standoff.
    chosen = plan["candidates"][plan["selected_candidate_index"]]
    assert chosen["standoff_from_target_surface_m"] < 0.9
    assert chosen["reachability_estimate"]["status"] == "PASS"
    assert plan["reach_seed_gate"]["status"] == "PASS"
    # The search only re-parameterized the sweep; acceptance evidence is sweep-shaped.
    assert "accepted_pose" in plan and "accepted_yaw" in plan


def test_adaptive_stance_search_reports_proved_infeasibility() -> None:
    # Close poses collide, far poses cannot reach: contradictory measured bounds in
    # every direction and via every staging anchor -> proved infeasibility, with the
    # original blocked plan and blockers preserved (fail-closed).
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "task_affordance_xyz": [0.0, 0.0, 1.0],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [0.9, 1.2],
        "floor_z_hint": 0.05,
    }

    def probe(pose, yaw):
        return 1 if math.dist(pose[:2], [0.0, 0.0]) < 1.05 else 0

    initial = M.plan_task_stance(scenario=scenario, probe_collision=probe)
    assert initial["status"] == "blocked"

    plan = M._adaptive_task_stance_search(
        stance_scenario=scenario,
        manipulation_look_at=None,
        probe=probe,
        placement_validator=None,
        initial_plan=initial,
    )
    assert plan is not None
    assert plan["status"] == "blocked"
    assert plan["blockers"] == initial["blockers"]
    assert plan["stance_search_status"] in {"infeasible", "budget_exhausted", "search_stalled"}
    search = plan["stance_search"]
    if plan["stance_search_status"] == "infeasible":
        proof = search["infeasibility_proof"]
        assert proof["per_direction"]
        assert all(d["window_is_empty"] for d in proof["per_direction"])
    assert "accepted_plan" not in search


def test_adaptive_stance_search_requires_resolved_affordance() -> None:
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [0.4],
        "floor_z_hint": 0.05,
    }
    blocked = {
        "status": "blocked",
        "blockers": ["no_collision_free_task_stance_candidate"],
        "candidates": [],
    }
    assert (
        M._adaptive_task_stance_search(
            stance_scenario=scenario,
            manipulation_look_at=None,
            probe=lambda pose, yaw: 0,
            placement_validator=None,
            initial_plan=blocked,
        )
        is None
    )


def test_adaptive_stance_search_ignores_non_geometric_blockers() -> None:
    blocked = {
        "status": "blocked",
        "blockers": ["missing_task_stance_target"],
        "task_affordance_xyz": [0.0, 0.0, 1.0],
        "candidates": [],
    }
    assert (
        M._adaptive_task_stance_search(
            stance_scenario={},
            manipulation_look_at=None,
            probe=lambda pose, yaw: 0,
            placement_validator=None,
            initial_plan=blocked,
        )
        is None
    )


def test_adaptive_stance_search_still_runs_placement_validator_on_new_poses() -> None:
    # The agent's re-sweeps must pass through the same placement validator; a
    # validator that rejects everything keeps the plan blocked no matter what the
    # search proposes.
    scenario = {
        "task_target_position_xyz": [0.0, 0.0, 0.9],
        "task_affordance_xyz": [0.0, 0.0, 1.0],
        "robot_start_position_xyz": [-3.0, 0.0, 0.05],
        "stance_distance_candidates_m": [0.9, 1.1],
        "floor_z_hint": 0.05,
    }
    validated_poses = []

    def rejecting_validator(pose, yaw, record):
        validated_poses.append(tuple(pose))
        return {"status": "blocked", "blockers": ["placed_robot_target_gap_below_threshold"]}

    initial = M.plan_task_stance(
        scenario=scenario,
        probe_collision=lambda pose, yaw: 0,
        placement_validator=rejecting_validator,
    )
    assert initial["status"] == "blocked"
    before = len(validated_poses)

    plan = M._adaptive_task_stance_search(
        stance_scenario=scenario,
        manipulation_look_at=None,
        probe=lambda pose, yaw: 0,
        placement_validator=rejecting_validator,
        initial_plan=initial,
    )
    assert plan is not None
    assert plan["status"] == "blocked"
    assert len(validated_poses) > before, "re-swept poses must hit the same validator"
    assert plan["stance_search_status"] in {"infeasible", "budget_exhausted", "search_stalled"}


# --------------------------------------------------------------------------
# Strict graded episode trace-consistency gate (fail-closed direction only)
# --------------------------------------------------------------------------


def test_trace_consistency_gate_blocks_success_claim_without_scored_trace() -> None:
    blockers = M.episode_trace_consistency_gate_blockers(
        {"task_success": True},
        {"status": "blocked", "consistency_score": None, "passed": False},
    )
    assert blockers == ["episode_success_claim_without_scored_trace_consistency"]


def test_trace_consistency_gate_blocks_success_claim_below_min_score() -> None:
    blockers = M.episode_trace_consistency_gate_blockers(
        {"task_success": True},
        {"status": "scored", "consistency_score": 0.41, "passed": False},
    )
    assert blockers == ["episode_success_claim_below_min_trace_consistency_score"]


def test_trace_consistency_gate_never_blocks_or_upgrades_failed_episode() -> None:
    # Failed episodes carry the graded score as evidence but are not re-gated,
    # and a high score must not upgrade the failure.
    for trace in (
        {"status": "scored", "consistency_score": 1.0, "passed": True},
        {"status": "blocked", "consistency_score": None, "passed": False},
    ):
        assert (
            M.episode_trace_consistency_gate_blockers({"task_success": False}, trace)
            == []
        )


def test_trace_consistency_gate_accepts_scored_passing_success() -> None:
    assert (
        M.episode_trace_consistency_gate_blockers(
            {"task_success": True},
            {"status": "scored", "consistency_score": 0.93, "passed": True},
        )
        == []
    )


def test_runner_attaches_graded_trace_consistency_to_episode_outcome() -> None:
    # The scenario loop itself needs Isaac; assert the wiring at source level the
    # same way other GPU-loop contracts are pinned in this file.
    source = _RUNNER.read_text()
    assert "compute_episode_trace_consistency(actions=actions)" in source
    assert 'outcome["episode_trace_consistency"] = trace_consistency' in source
    gate_call = source.index(
        "trace_gate_blockers = episode_trace_consistency_gate_blockers("
    )
    # The gate must run AFTER the success contract can promote task_success and
    # BEFORE the outcome is recorded, and it must demote the scenario outcome
    # itself — otherwise scenarios_passed still counts the episode as passed.
    contract_call = source.index(
        "contract_result = _apply_visible_reach_to_affordance_success_contract("
    )
    append_call = source.index("outcomes.append(outcome)")
    assert contract_call < gate_call < append_call
    demotion = source.index('outcome["task_success"] = False', gate_call)
    assert gate_call < demotion < append_call
