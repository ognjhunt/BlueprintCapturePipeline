from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import gear_sonic_joint_order_contract as contract
from blueprint_pipeline import gear_sonic_official_zmq_executor as executor
from blueprint_pipeline import isaac_runtime_task_backend as isaac_backend

FIXTURE_MODEL = (
    Path(__file__).parent / "fixtures" / "gear_sonic_g1_min" / "g1_29dof_with_hand_min.xml"
)


def _request(action: dict) -> dict:
    return {
        "step_index": 3,
        "action": action,
        "source_action_sha256": hashlib.sha256(
            json.dumps(action, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _controller_state(motion_token, left_hand=None, right_hand=None, **overrides) -> dict:
    state = {
        "token_state": motion_token,
        "body_q_target": [0.1] * 29,
        "body_q_measured": [0.0] * 29,
        "last_left_hand_action": list(left_hand) if left_hand is not None else [0.0] * 7,
        "last_right_hand_action": list(right_hand) if right_hand is not None else [0.0] * 7,
        "base_quat_measured": [1.0, 0.0, 0.0, 0.0],
        "ros_timestamp": 123,
    }
    state.update(overrides)
    return state


def _echoing_transport(calls=None):
    def transport(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return _controller_state(
            kwargs["motion_token"],
            left_hand=kwargs["left_hand"],
            right_hand=kwargs["right_hand"],
        )

    return transport


def _fake_fk(**kwargs):
    names = list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)
    positions = list(kwargs["body_positions"]) + list(kwargs["left_hand"]) + list(
        kwargs["right_hand"]
    )
    applied = [
        {"joint_name": name, "protocol_index": index, "model_qpos_address": index + 7}
        for index, name in enumerate(names)
    ]
    return names, positions, [
        {"name": "right_wrist_yaw_link", "x": 0.1, "y": 0.2, "z": 1.0}
    ], applied


def _install_model(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "wbc"
    model = root / "gear_sonic_deploy" / "g1" / "g1_29dof_with_hand.xml"
    model.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE_MODEL, model)
    return root, model


def _live_projection_context(
    tmp_path: Path,
    *,
    pelvis_quaternion_wxyz: list[float] | None = None,
    camera_world_xyz_m: list[float] | None = None,
    camera_xmat_row_major: list[list[float]] | None = None,
    fx: float = 100.0,
    fy: float = 100.0,
) -> dict:
    frame = tmp_path / "initial_policy_frame.png"
    frame.write_bytes(b"live-isaac-frame")
    return {
        "schema_version": executor.CAMERA_PROJECTION_SCHEMA_VERSION,
        "status": executor.CAMERA_PROJECTION_LIVE_STATUS,
        "attempt_id": "attempt-1",
        "launch_nonce": "nonce-1",
        "simulator_session_id": "isaac-session-1",
        "stage_id": "stage-1",
        "source_frame_artifact": {
            "path": str(frame),
            "sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
            "width": 640,
            "height": 480,
        },
        "camera_contract": {
            "available": True,
            "projection_token": "perspective",
            "intrinsics": {
                "available": True,
                "fx": fx,
                "fy": fy,
                "cx": 320.0,
                "cy": 240.0,
                "image_width": 640,
                "image_height": 480,
            },
            "camera_world_xyz_m": camera_world_xyz_m or [0.0, 0.0, 0.0],
            "camera_xmat_row_major": camera_xmat_row_major
            or [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "clipping_range_m": [0.05, 50.0],
            "resolution": [640, 480],
        },
        "live_isaac_pelvis_world_pose": {
            "prim_path": "/World/G1/pelvis",
            "position_xyz": [0.0, 0.0, 0.0],
            "quaternion_wxyz": pelvis_quaternion_wxyz or [1.0, 0.0, 0.0, 0.0],
        },
        "standing_cross_simulator_registration": {
            "status": "pending_official_mujoco_named_link_residual_verification",
            "surrogate": False,
        },
        "coordinate_transform": executor.CAMERA_PROJECTION_TRANSFORM,
    }


def _execute(request, **kwargs):
    return executor.execute(
        request,
        controller_revision_resolver=lambda _root: contract.PINNED_WBC_SOURCE_REVISION,
        **kwargs,
    )


def test_pinned_controller_revision_accepts_runtime_marker_without_git(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wbc"
    root.mkdir()
    (root / ".blueprint-source-revision").write_text(
        contract.PINNED_WBC_SOURCE_REVISION + "\n", encoding="utf-8"
    )
    assert executor._pinned_controller_revision(root) == (
        contract.PINNED_WBC_SOURCE_REVISION
    )


def test_pinned_controller_revision_rejects_wrong_runtime_marker(
    tmp_path: Path,
) -> None:
    root = tmp_path / "wbc"
    root.mkdir()
    (root / ".blueprint-source-revision").write_text("wrong\n", encoding="utf-8")
    with pytest.raises(
        RuntimeError, match="official_gear_sonic_controller_revision_mismatch"
    ):
        executor._pinned_controller_revision(root)


def test_pinned_controller_revision_scopes_safe_directory_to_exact_checkout(
    tmp_path: Path, monkeypatch
) -> None:
    root = (tmp_path / "wbc").resolve()
    root.mkdir()
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=0,
            stdout=contract.PINNED_WBC_SOURCE_REVISION + "\n",
        )

    monkeypatch.setattr(executor.subprocess, "run", fake_run)

    assert executor._pinned_controller_revision(root) == (
        contract.PINNED_WBC_SOURCE_REVISION
    )
    assert calls == [
        (
            [
                "git",
                "-c",
                f"safe.directory={root}",
                "rev-parse",
                "HEAD",
            ],
            {
                "cwd": root,
                "capture_output": True,
                "text": True,
                "check": False,
                "timeout": 10,
            },
        )
    ]


def test_pinned_controller_revision_reads_legacy_detached_head_without_git(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "wbc"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text(
        contract.PINNED_WBC_SOURCE_REVISION + "\n", encoding="ascii"
    )

    def unexpected_git(*_args, **_kwargs):
        raise AssertionError("detached HEAD provenance must not invoke Git")

    monkeypatch.setattr(executor.subprocess, "run", unexpected_git)
    assert executor._pinned_controller_revision(root) == (
        contract.PINNED_WBC_SOURCE_REVISION
    )


def test_pinned_controller_revision_rejects_wrong_legacy_detached_head(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "wbc"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "HEAD").write_text("0" * 40 + "\n", encoding="ascii")

    def unexpected_git(*_args, **_kwargs):
        raise AssertionError("a mismatched detached HEAD must fail before Git")

    monkeypatch.setattr(executor.subprocess, "run", unexpected_git)
    with pytest.raises(
        RuntimeError, match="official_gear_sonic_controller_revision_mismatch"
    ):
        executor._pinned_controller_revision(root)


def test_legacy_detached_head_rejects_symbolic_or_symlink_metadata(
    tmp_path: Path,
) -> None:
    symbolic = tmp_path / "symbolic"
    (symbolic / ".git").mkdir(parents=True)
    (symbolic / ".git" / "HEAD").write_text(
        "ref: refs/heads/main\n", encoding="ascii"
    )
    assert executor._legacy_detached_head_revision(symbolic) == ""

    target = tmp_path / "head-target"
    target.write_text(contract.PINNED_WBC_SOURCE_REVISION, encoding="ascii")
    linked = tmp_path / "linked"
    (linked / ".git").mkdir(parents=True)
    (linked / ".git" / "HEAD").symlink_to(target)
    assert executor._legacy_detached_head_revision(linked) == ""


def test_fk_landmarks_project_through_bound_live_isaac_camera(tmp_path: Path) -> None:
    frame = tmp_path / "initial_policy_frame.png"
    frame.write_bytes(b"live-isaac-frame")
    frame_sha256 = hashlib.sha256(frame.read_bytes()).hexdigest()
    landmarks = [
        {
            "name": "right_wrist_yaw_link",
            "model_root_relative_xyz": [0.0, 0.0, -2.0],
        },
        {
            "name": "right_hand_palm_link",
            "model_root_relative_xyz": [0.2, -0.1, -2.0],
        },
    ]
    context = {
        "schema_version": executor.CAMERA_PROJECTION_SCHEMA_VERSION,
        "status": executor.CAMERA_PROJECTION_LIVE_STATUS,
        "attempt_id": "attempt-1",
        "launch_nonce": "nonce-1",
        "simulator_session_id": "isaac-session-1",
        "stage_id": "stage-1",
        "source_frame_artifact": {
            "path": str(frame),
            "sha256": frame_sha256,
            "width": 640,
            "height": 480,
        },
        "camera_contract": {
            "available": True,
            "projection_token": "perspective",
            "intrinsics": {
                "available": True,
                "fx": 100.0,
                "fy": 100.0,
                "cx": 320.0,
                "cy": 240.0,
                "image_width": 640,
                "image_height": 480,
            },
            "camera_world_xyz_m": [0.0, 0.0, 0.0],
            "camera_xmat_row_major": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "clipping_range_m": [0.05, 50.0],
            "resolution": [640, 480],
        },
        "live_isaac_pelvis_world_pose": {
            "prim_path": "/World/G1/pelvis",
            "position_xyz": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "standing_cross_simulator_registration": {
            "status": "pending_official_mujoco_named_link_residual_verification",
            "surrogate": False,
        },
        "coordinate_transform": executor.CAMERA_PROJECTION_TRANSFORM,
    }

    projected, context_sha256 = executor._project_fk_landmarks(landmarks, context)

    assert context_sha256 == executor._canonical(context)
    assert projected[0]["image_projection"] == {
        "available": True,
        "u_px": 320.0,
        "v_px": 240.0,
        "depth_m": 2.0,
        "projection_context_sha256": context_sha256,
        "source_frame_sha256": frame_sha256,
    }
    assert projected[1]["image_projection"]["u_px"] == 330.0
    assert projected[1]["image_projection"]["v_px"] == 245.0


def test_projection_uses_full_live_pelvis_quaternion_not_yaw_only(
    tmp_path: Path,
) -> None:
    half_sqrt = 2.0**-0.5
    context = _live_projection_context(
        tmp_path,
        pelvis_quaternion_wxyz=[half_sqrt, -half_sqrt, 0.0, 0.0],
    )
    projected, _ = executor._project_fk_landmarks(
        [
            {
                "name": "left_wrist_yaw_link",
                "model_root_relative_xyz": [0.0, 2.0, 0.0],
            }
        ],
        context,
    )
    assert projected[0]["world_xyz"] == pytest.approx([0.0, 0.0, -2.0])
    assert projected[0]["image_projection"]["u_px"] == 320.0
    assert projected[0]["image_projection"]["v_px"] == 240.0


def test_projection_rejects_reflection_stale_context_and_frame_hash_mismatch(
    tmp_path: Path,
) -> None:
    reflected = _live_projection_context(
        tmp_path,
        camera_xmat_row_major=[
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    landmark = [{"name": "wrist", "model_root_relative_xyz": [0.0, 0.0, -2.0]}]
    with pytest.raises(ValueError, match="not_proper_rotation"):
        executor._project_fk_landmarks(landmark, reflected)

    stale = _live_projection_context(tmp_path)
    stale["status"] = "bundled_static_seed"
    with pytest.raises(ValueError, match="not_live_session_capture"):
        executor._project_fk_landmarks(landmark, stale)

    mismatched = _live_projection_context(tmp_path)
    Path(mismatched["source_frame_artifact"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source_frame_missing_or_hash_mismatch"):
        executor._project_fk_landmarks(landmark, mismatched)


def test_projection_retains_explicit_out_of_view_evidence_without_blocking_fk(
    tmp_path: Path,
) -> None:
    context = _live_projection_context(tmp_path)

    projected, context_sha256 = executor._project_fk_landmarks(
        [
            {
                "name": "standing_wrist_outside_robot_pov",
                "model_root_relative_xyz": [100.0, 0.0, -2.0],
            }
        ],
        context,
    )

    projection = projected[0]["image_projection"]
    assert projection["available"] is False
    assert projection["unavailable_reason"] == "outside_live_camera_viewport"
    assert projection["u_px"] > 640.0
    assert projection["v_px"] == pytest.approx(240.0)
    assert projection["projection_context_sha256"] == context_sha256
    assert projection["source_frame_sha256"] == context["source_frame_artifact"][
        "sha256"
    ]


def test_projection_honors_live_camera_near_and_far_clipping_range(
    tmp_path: Path,
) -> None:
    context = _live_projection_context(tmp_path)
    context["camera_contract"]["clipping_range_m"] = [0.05, 2.0]

    projected, _ = executor._project_fk_landmarks(
        [
            {"name": "near", "model_root_relative_xyz": [0.0, 0.0, -0.01]},
            {"name": "visible", "model_root_relative_xyz": [0.0, 0.0, -1.0]},
            {"name": "far", "model_root_relative_xyz": [0.0, 0.0, -3.0]},
        ],
        context,
    )

    assert projected[0]["image_projection"]["available"] is False
    assert projected[0]["image_projection"]["unavailable_reason"] == (
        "outside_live_camera_depth_range"
    )
    assert projected[0]["image_projection"]["depth_m"] == 0.01
    assert projected[1]["image_projection"]["available"] is True
    assert projected[1]["image_projection"]["depth_m"] == 1.0
    assert projected[2]["image_projection"]["available"] is False
    assert projected[2]["image_projection"]["unavailable_reason"] == (
        "outside_live_camera_depth_range"
    )
    assert projected[2]["image_projection"]["depth_m"] == 3.0


@pytest.mark.parametrize(
    "clipping_range",
    [None, [0.0, 50.0], [0.05, 0.05], [0.05, float("nan")]],
)
def test_projection_rejects_invalid_live_camera_clipping_range(
    tmp_path: Path,
    clipping_range,
) -> None:
    context = _live_projection_context(tmp_path)
    if clipping_range is None:
        context["camera_contract"].pop("clipping_range_m")
    else:
        context["camera_contract"]["clipping_range_m"] = clipping_range

    with pytest.raises(
        ValueError,
        match="official_gear_sonic_camera_clipping_range_invalid",
    ):
        executor._project_fk_landmarks(
            [{"name": "wrist", "model_root_relative_xyz": [0.0, 0.0, -1.0]}],
            context,
        )


def test_july_10_golden_camera_rows_reproduce_recorded_target_projection(
    tmp_path: Path,
) -> None:
    context = _live_projection_context(
        tmp_path,
        camera_world_xyz_m=[-1.102809, 1.471279, 1.2802],
        camera_xmat_row_major=[
            [-0.1126371, 0.99363619, 0.00000002],
            [-0.4041482, -0.04581365, 0.91354548],
            [0.90773185, 0.1028991, 0.4067366],
        ],
        fx=168.0498118992199,
        fy=168.0498118992199,
    )
    projected, _ = executor._project_fk_landmarks(
        [
            {
                "name": "microwave_target_golden_vector",
                "model_root_relative_xyz": [-1.591312, 1.471274, 1.241574],
            }
        ],
        context,
    )
    projection = projected[0]["image_projection"]
    assert projection["u_px"] == pytest.approx(340.1373, abs=1e-4)
    assert projection["v_px"] == pytest.approx(180.6548, abs=1e-4)
    assert projection["depth_m"] == pytest.approx(0.459141, abs=1e-6)


def test_named_link_registration_residual_gate_passes_and_fails_closed(
    tmp_path: Path,
) -> None:
    context = _live_projection_context(tmp_path)
    names = [
        "left_shoulder_pitch_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ]
    model_points = {
        name: [0.1 * index, (-1.0) ** index * 0.2, 0.5 + 0.03 * index]
        for index, name in enumerate(names)
    }
    context["standing_cross_simulator_registration"].update(
        {
            "required_landmark_names": names,
            "isaac_named_link_world_poses": [
                {"landmark_id": name, "world_position_xyz": list(point)}
                for name, point in model_points.items()
            ],
            "maximum_residual_tolerance_m": 0.001,
        }
    )
    standing = [
        {"name": name, "model_root_relative_xyz": list(point)}
        for name, point in model_points.items()
    ]
    evidence = executor._verify_standing_cross_simulator_registration(
        context=context,
        standing_landmarks=standing,
    )
    assert evidence["status"] == "passed"
    assert evidence["maximum_residual_m"] == pytest.approx(0.0)

    context["standing_cross_simulator_registration"][
        "isaac_named_link_world_poses"
    ][0]["world_position_xyz"][0] += 0.01
    with pytest.raises(RuntimeError, match="registration_residual_exceeded"):
        executor._verify_standing_cross_simulator_registration(
            context=context,
            standing_landmarks=standing,
        )


@pytest.fixture()
def wbc_env(tmp_path, monkeypatch):
    root, model = _install_model(tmp_path)
    monkeypatch.setenv(executor.ROOT_ENV, str(root))
    monkeypatch.setenv(executor.MODEL_ENV, str(model))
    return root, model


def test_executor_sends_78d_sonic_action_to_official_protocol_and_uses_fk(
    wbc_env,
) -> None:
    _, model = wbc_env
    action = {"sonic_action_chunk": [float(index) / 100 for index in range(78)]}
    calls = []
    transport = _echoing_transport(calls)

    def fk_solver(**kwargs):
        assert kwargs["model_path"] == model
        assert kwargs["body_positions"] == [0.1] * 29
        return _fake_fk(**kwargs)

    result = _execute(_request(action), transport=transport, fk_solver=fk_solver)

    assert len(calls) == 1
    assert len(calls[0]["motion_token"]) == 64
    assert calls[0]["left_hand"] == action["sonic_action_chunk"][64:71]
    assert calls[0]["right_hand"] == action["sonic_action_chunk"][71:78]
    assert result["status"] == "completed"
    assert result["proprioceptive_state"]["official_controller_protocol"] == 4
    assert result["joint_order_schema_version"] == contract.JOINT_ORDER_SCHEMA_VERSION
    assert result["mapping_digest"] == contract.PROTOCOL_V4_MAPPING_DIGEST
    assert result["controller_revision"] == contract.PINNED_WBC_SOURCE_REVISION
    assert result["robot_model_sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    assert [row["joint_name"] for row in result["applied_dof_mapping"]] == list(
        contract.PROTOCOL_V4_FULL_JOINT_ORDER
    )
    assert result["joint_names"] == list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)


def test_executor_streams_explicit_horizon_once_and_returns_every_fk_frame(
    wbc_env,
) -> None:
    frames = [
        [float(frame_index + 1)] * 64
        + [float(frame_index) / 10] * 7
        + [-float(frame_index) / 10] * 7
        for frame_index in range(3)
    ]
    sequence = {
        "schema_version": "gear_sonic_controller_action_sequence.v1",
        "execution_mode": "bounded_model_horizon_prefix",
        "execution_frame_count": 3,
        "source_horizon_frame_count": 40,
        "frame_dimension": 78,
        "control_hz": 50.0,
        "sample_period_seconds": 0.02,
        "execution_duration_seconds": 0.06,
        "frames": frames,
        "frames_sha256": executor._canonical(frames),
        "source_frames_sha256": "f" * 64,
    }
    action = {
        "sonic_action_chunk": frames[0],
        "controller_action": sequence,
    }
    calls = []

    def horizon_transport(**kwargs):
        calls.append(kwargs)
        return [
            _controller_state(
                frame[:64],
                left_hand=frame[64:71],
                right_hand=frame[71:78],
                body_q_target=[0.1 * (index + 1)] * 29,
                ros_timestamp=100 + index,
            )
            for index, frame in enumerate(kwargs["action_frames"])
        ]

    result = _execute(
        _request(action), transport=horizon_transport, fk_solver=_fake_fk
    )

    assert len(calls) == 1
    assert calls[0]["action_frames"] == frames
    assert calls[0]["control_hz"] == 50.0
    rows = result["controller_fk_sequence"]
    assert len(rows) == 3
    assert [row["horizon_frame_index"] for row in rows] == [0, 1, 2]
    assert calls[0]["frame_index"] == 81
    assert [row["controller_frame_index"] for row in rows] == [81, 82, 83]
    assert [row["source_action_frame_sha256"] for row in rows] == [
        executor._canonical(frame) for frame in frames
    ]
    assert rows[-1]["joint_positions"][:29] == pytest.approx([0.3] * 29)
    assert result["joint_positions"] == rows[-1]["joint_positions"]
    assert result["landmarks"] == rows[-1]["landmarks"]
    assert result["controller_fk_sequence_sha256"] == executor._canonical(rows)
    execution = result["execution_contract"]
    assert execution["controller_session_count"] == 1
    assert execution["execution_frame_count"] == 3
    assert execution["source_horizon_frame_count"] == 40
    assert execution["control_hz"] == 50.0
    assert execution["input_action_frames_sha256"] == executor._canonical(frames)


def test_explicit_controller_frame_ranges_are_monotonic_across_outer_steps() -> None:
    assert [
        contract.controller_frame_sequence_start(
            outer_step_index=1,
            source_horizon_frame_count=40,
            explicit_horizon=True,
        )
        + index
        for index in range(40)
    ] == list(range(1, 41))
    assert [
        contract.controller_frame_sequence_start(
            outer_step_index=2,
            source_horizon_frame_count=40,
            explicit_horizon=True,
        )
        + index
        for index in range(40)
    ] == list(range(41, 81))
    assert contract.controller_frame_sequence_start(
        outer_step_index=-1,
        source_horizon_frame_count=40,
        explicit_horizon=True,
    ) == -1
    assert contract.controller_frame_sequence_start(
        outer_step_index=7,
        source_horizon_frame_count=1,
        explicit_horizon=False,
    ) == 7


def test_executor_rejects_mutated_explicit_horizon_before_transport(wbc_env) -> None:
    frames = [[0.0] * 78, [1.0] * 78]
    action = {
        "sonic_action_chunk": frames[0],
        "controller_action": {
            "schema_version": "gear_sonic_controller_action_sequence.v1",
            "execution_mode": "bounded_model_horizon_prefix",
            "execution_frame_count": 2,
            "source_horizon_frame_count": 2,
            "frame_dimension": 78,
            "control_hz": 50.0,
            "sample_period_seconds": 0.02,
            "execution_duration_seconds": 0.04,
            "frames": frames,
            "frames_sha256": "0" * 64,
            "source_frames_sha256": "f" * 64,
        },
    }
    with pytest.raises(ValueError, match="sequence_sha256_mismatch"):
        _execute(_request(action), transport=_echoing_transport(), fk_solver=_fake_fk)


def test_executor_rejects_shape_only_or_nonfinite_action(wbc_env) -> None:
    with pytest.raises(ValueError, match="dimension_or_value_invalid"):
        executor.execute(_request({"action_chunk": [0.0] * 77}))
    with pytest.raises(ValueError, match="dimension_or_value_invalid"):
        executor.execute(_request({"action_chunk": [0.0] * 77 + [float("nan")]}))


def test_executor_accepts_official_positional_controller_state(wbc_env) -> None:
    action = {"sonic_action_chunk": [0.0] * 78}

    def transport(**kwargs):
        return _controller_state(kwargs["motion_token"])

    result = _execute(_request(action), transport=transport, fk_solver=_fake_fk)
    assert result["mapping_source"] == "pinned_wbc_mujoco_order"


def test_executor_rejects_unpinned_installed_controller_revision(wbc_env) -> None:
    action = {"sonic_action_chunk": [0.0] * 78}

    def transport(**kwargs):
        return _controller_state(kwargs["motion_token"])

    with pytest.raises(ValueError, match="controller_revision_mismatch"):
        executor.execute(
            _request(action),
            transport=transport,
            fk_solver=_fake_fk,
            controller_revision_resolver=lambda _root: "unreviewed-revision",
        )


def test_sealed_revision_marker_replaces_runtime_git_history(wbc_env) -> None:
    root, _ = wbc_env
    marker = root / executor.SEALED_REVISION_FILE
    marker.write_text(contract.PINNED_WBC_SOURCE_REVISION + "\n", encoding="utf-8")
    assert executor._pinned_controller_revision(root) == contract.PINNED_WBC_SOURCE_REVISION
    marker.write_text("0" * 40 + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="controller_revision_mismatch"):
        executor._pinned_controller_revision(root)


def test_executor_validates_live_isaac_articulation_joints(wbc_env) -> None:
    action = {"sonic_action_chunk": [0.0] * 78}

    def transport(**kwargs):
        return _controller_state(kwargs["motion_token"])

    live = list(reversed(contract.PROTOCOL_V4_FULL_JOINT_ORDER))
    result = _execute(
        _request(action),
        transport=transport,
        fk_solver=_fake_fk,
        isaac_joint_names=live,
    )
    assert len(result["isaac_dof_mapping"]) == 43
    for row in result["isaac_dof_mapping"]:
        assert live[row["articulation_dof_index"]] == row["joint_name"]

    with pytest.raises(ValueError, match="isaac_articulation_joint_names_missing"):
        _execute(
            _request(action),
            transport=transport,
            fk_solver=_fake_fk,
            isaac_joint_names=live[:-1],
        )


def _mujoco_or_skip():
    return pytest.importorskip("mujoco", reason="mujoco_not_installed_in_venv")


def test_real_fk_maps_values_by_joint_name_not_position() -> None:
    _mujoco_or_skip()
    body = [0.0] * 29
    body[contract.PROTOCOL_V4_BODY_JOINT_NAMES.index("right_wrist_yaw_joint")] = 0.5
    names, positions, _, applied = executor._official_mujoco_fk(
        model_path=FIXTURE_MODEL,
        body_positions=body,
        left_hand=[0.0] * 7,
        right_hand=[0.0] * 7,
    )
    assert names == list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)
    assert len(positions) == 43
    by_name = {row["joint_name"]: row for row in applied}
    assert by_name["right_wrist_yaw_joint"]["applied_value"] == 0.5
    assert all(
        row["applied_value"] == 0.0
        for row in applied
        if row["joint_name"] != "right_wrist_yaw_joint"
    )
    # qpos addresses must be unique: no positional collapsing.
    addresses = [row["model_qpos_address"] for row in applied]
    assert len(set(addresses)) == 43


def test_registration_uses_only_real_common_mujoco_body_names() -> None:
    _mujoco_or_skip()
    _, _, landmarks, _ = executor._official_mujoco_fk(
        model_path=FIXTURE_MODEL,
        body_positions=[0.0] * 29,
        left_hand=[0.0] * 7,
        right_hand=[0.0] * 7,
    )
    names = {row["name"] for row in landmarks}
    required = {
        "left_shoulder_pitch_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_pitch_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    }
    assert required <= names
    assert set(isaac_backend.CONTROLLER_FK_REGISTRATION_LANDMARK_NAMES) == required


def _landmarks_by_name(landmarks) -> dict[str, tuple[float, float, float]]:
    return {row["name"]: (row["x"], row["y"], row["z"]) for row in landmarks}


def test_real_fk_two_asymmetric_perturbations_move_distinct_joints() -> None:
    _mujoco_or_skip()
    neutral = [0.0] * 29
    left = list(neutral)
    left[contract.PROTOCOL_V4_BODY_JOINT_NAMES.index("left_elbow_joint")] = 0.6
    right = list(neutral)
    right[contract.PROTOCOL_V4_BODY_JOINT_NAMES.index("right_elbow_joint")] = 0.6
    hands = {"left_hand": [0.0] * 7, "right_hand": [0.0] * 7}
    _, _, base_marks, _ = executor._official_mujoco_fk(
        model_path=FIXTURE_MODEL, body_positions=neutral, **hands
    )
    _, _, left_marks, _ = executor._official_mujoco_fk(
        model_path=FIXTURE_MODEL, body_positions=left, **hands
    )
    _, _, right_marks, _ = executor._official_mujoco_fk(
        model_path=FIXTURE_MODEL, body_positions=right, **hands
    )
    base = _landmarks_by_name(base_marks)
    after_left = _landmarks_by_name(left_marks)
    after_right = _landmarks_by_name(right_marks)
    assert after_left != after_right
    # The left perturbation moves the left wrist and leaves the right wrist alone.
    assert after_left["left_wrist_yaw_link"] != base["left_wrist_yaw_link"]
    assert after_left["right_wrist_yaw_link"] == base["right_wrist_yaw_link"]
    assert after_right["right_wrist_yaw_link"] != base["right_wrist_yaw_link"]
    assert after_right["left_wrist_yaw_link"] == base["left_wrist_yaw_link"]


def test_real_fk_rejects_model_with_wrong_joint_set(tmp_path) -> None:
    _mujoco_or_skip()
    mutated = tmp_path / "mutated.xml"
    mutated.write_text(
        FIXTURE_MODEL.read_text(encoding="utf-8").replace(
            "left_elbow_joint", "left_elbow_mystery_joint"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mujoco_model_joint_names_unknown"):
        executor._official_mujoco_fk(
            model_path=mutated,
            body_positions=[0.0] * 29,
            left_hand=[0.0] * 7,
            right_hand=[0.0] * 7,
        )


def test_execute_end_to_end_with_real_fk_carries_sha_and_digest(wbc_env) -> None:
    _mujoco_or_skip()
    action = {"sonic_action_chunk": [0.01] * 78}
    request = _request(action)

    result = _execute(request, transport=_echoing_transport())
    assert result["source_action_sha256"] == request["source_action_sha256"]
    assert result["mapping_digest"] == contract.PROTOCOL_V4_MAPPING_DIGEST
    assert result["joint_names"] == list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)
    assert len(result["joint_positions"]) == 43
    assert result["landmarks"]


def test_executor_rejects_stale_hand_echo_before_fk(wbc_env) -> None:
    action = {"sonic_action_chunk": [float(index) / 100 for index in range(78)]}

    def stale_left(**kwargs):
        return _controller_state(
            kwargs["motion_token"],
            last_left_hand_action=[9.9] * 7,
            last_right_hand_action=list(kwargs["right_hand"]),
        )

    with pytest.raises(
        RuntimeError, match="official_gear_sonic_controller_hand_echo_mismatch:left"
    ):
        _execute(_request(action), transport=stale_left, fk_solver=_fake_fk)

    def stale_right(**kwargs):
        return _controller_state(
            kwargs["motion_token"],
            last_left_hand_action=list(kwargs["left_hand"]),
            last_right_hand_action=[9.9] * 7,
        )

    with pytest.raises(
        RuntimeError, match="official_gear_sonic_controller_hand_echo_mismatch:right"
    ):
        _execute(_request(action), transport=stale_right, fk_solver=_fake_fk)


def test_executor_accepts_matching_hand_echo(wbc_env) -> None:
    action = {"sonic_action_chunk": [float(index) / 100 for index in range(78)]}

    def matching(**kwargs):
        return _controller_state(
            kwargs["motion_token"],
            last_left_hand_action=list(kwargs["left_hand"]),
            last_right_hand_action=list(kwargs["right_hand"]),
        )

    result = _execute(_request(action), transport=matching, fk_solver=_fake_fk)
    assert result["status"] == "completed"
    assert result["joint_positions"][29:36] == action["sonic_action_chunk"][64:71]
    assert result["joint_positions"][36:43] == action["sonic_action_chunk"][71:78]
