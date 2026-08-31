from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_construction_worker import (
    DEPENDENCY_IMPORTS,
    _initial_contact_blocked,
    _load_and_verify_manifest,
    _pad_centers_from_finger_body_offsets,
    _pad_offsets_from_relative_geometry,
    _pose_arrival_readback,
    _prepare_site_appearance_renderer,
    _requested_arm_reset,
    _retain_task_path_samples,
    _task_joint_reset_passed,
    _terminal_grasp_frame_arrival_readback,
)
from blueprint_pipeline.native_task_arena_feedback_bootstrap_runtime import (
    verified_construction_phase_plan_path,
    verified_terminal_feedback_adoption_path,
)
from blueprint_pipeline.native_task_arena_import_scope import ROBOT_EMBODIMENT_MODULES
from blueprint_pipeline.native_task_runtime_source_provision import TOP_LEVEL_PACKAGES


def test_construction_warms_plain_nurec_before_camera_evidence() -> None:
    class App:
        updates = 0

        def update(self):
            self.updates += 1

    app = App()
    result = _prepare_site_appearance_renderer(
        simulation_app=app,
        plan={
            "appearance_frame_alignment": {"representation": "nurec_volume"}
        },
        stage=object(),
        setup_for_rendering_factory=lambda _stage: (True, True, False, []),
        warmup_steps=40,
    )

    assert result["passed"] is True
    assert result["representation"] == "nurec_volume"
    assert result["stage_classified_nurec"] is True
    assert result["app_update_count"] == 45
    assert app.updates == 45


def test_construction_skips_nurec_setup_for_other_appearance_formats() -> None:
    result = _prepare_site_appearance_renderer(
        simulation_app=object(),
        plan={
            "appearance_frame_alignment": {"representation": "mesh_texture"}
        },
    )

    assert result == {
        "schema_version": "native_task_arena_nurec_warmup.v1",
        "status": "not_required",
        "representation": "mesh_texture",
        "passed": True,
        "blockers": [],
    }


def test_construction_refuses_nurec_stage_that_renderer_cannot_classify() -> None:
    result = _prepare_site_appearance_renderer(
        simulation_app=object(),
        plan={
            "appearance_frame_alignment": {"representation": "nurec_volume"}
        },
        stage=object(),
        setup_for_rendering_factory=lambda _stage: (True, False, False, []),
    )

    assert result["passed"] is False
    assert result["representation"] == "nurec_volume"
    assert result["blockers"] == [
        "native_task_arena_nurec_official_setup_not_qualified"
    ]


def test_physical_pad_centers_follow_finger_bodies_not_their_origins() -> None:
    import numpy as np
    import pytest

    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_names=["left_inner_finger", "right_inner_finger"],
            body_pose_w=np.asarray(
                [
                    [
                        [0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 1.0],
                        [0.0, -0.04, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                ]
            ),
        )
    )

    centers = _pad_centers_from_finger_body_offsets(
        robot=robot,
        offsets_body_m={"left": [0.0, 0.01, 0.0], "right": [0.0, -0.01, 0.0]},
        torch=SimpleNamespace(as_tensor=np.asarray),
    )

    assert centers["left"] == pytest.approx([0.0, 0.05, 0.0])
    assert centers["right"] == pytest.approx([0.0, -0.05, 0.0])


def test_pad_offsets_prefer_coherent_collider_to_finger_frame() -> None:
    offsets = _pad_offsets_from_relative_geometry(
        {
            "selected_pad_colliders": {
                "left": {"center_inner_finger_body_m": [0.13, 0.052, 0.0]},
                "right": {"center_inner_finger_body_m": [0.13, -0.052, 0.0]},
            }
        }
    )

    assert offsets == {
        "left": [0.13, 0.052, 0.0],
        "right": [0.13, -0.052, 0.0],
    }


def test_worker_source_contains_no_scene_or_task_object_identity() -> None:
    source = Path(
        __import__(
            "blueprint_pipeline.native_task_arena_construction_worker",
            fromlist=["x"],
        ).__file__
    ).read_text(encoding="utf-8")

    for forbidden in ("840313", "840796", "refrigerator", "approved_can"):
        assert forbidden not in source


def test_dependency_matrix_is_declared_as_one_preflight() -> None:
    assert ROBOT_EMBODIMENT_MODULES == {
        "franka_panda": "isaaclab_arena.embodiments.droid.droid"
    }
    assert {
        "torch",
        "pxr.UsdVol",
        "gymnasium",
        "lazy_loader",
        "cloudpickle",
        "farama_notifications",
        "packaging",
        "prettytable",
        "typing_extensions",
        "wcwidth",
        "h5py",
        "yaml",
        "toml",
        "antlr4",
        "omegaconf",
        "hydra",
        "hydra.core",
        "msgpack",
        "zmq",
        "rsl_rl",
        "rsl_rl.runners",
        "tensordict",
        "importlib_metadata",
        "orjson",
        "pyvers",
        "git",
        "gitdb",
        "smmap",
        "lightwheel_sdk",
        "lightwheel_sdk.loader",
        "requests",
        "charset_normalizer",
        "idna",
        "urllib3",
        "certifi",
        "tqdm",
        "termcolor",
        "click",
        "isaaclab_contrib",
        "isaaclab_newton",
        "isaaclab.controllers",
        "isaaclab_assets",
        "isaaclab_tasks",
        "isaaclab_teleop",
        "isaaclab_arena.environments.arena_env_builder",
        "isaacsim.robot_motion.experimental.motion_generation",
        "isaacsim.robot_motion.pink",
        "pinocchio",
        "pink",
        "qpsolvers",
        "osqp",
    }.issubset(DEPENDENCY_IMPORTS)
    assert set(TOP_LEVEL_PACKAGES).issubset(DEPENDENCY_IMPORTS)
    assert DEPENDENCY_IMPORTS.index("isaaclab_contrib") < DEPENDENCY_IMPORTS.index(
        "isaaclab_arena.environments.arena_env_builder"
    )
    assert DEPENDENCY_IMPORTS.index("isaaclab_newton") < DEPENDENCY_IMPORTS.index(
        "isaaclab_arena.environments.arena_env_builder"
    )


def test_manifest_binding_rejects_tamper_before_isaac(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "native_task_arena_provider_bundle.v1",
        "execution_mode": "construction_canary",
        "implementation_commit": "a" * 40,
        "input_digest": "",
    }
    manifest["input_digest"] = canonical_digest(
        manifest, digest_field="input_digest"
    )
    (tmp_path / "adp_arena_provider_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert _load_and_verify_manifest(tmp_path)["input_digest"].startswith("sha256:")

    manifest["execution_mode"] = "policy"
    (tmp_path / "adp_arena_provider_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    try:
        _load_and_verify_manifest(tmp_path)
    except RuntimeError as exc:
        assert str(exc) == "native_task_construction_manifest_invalid"
    else:  # pragma: no cover - explicit fail message is clearer than pytest magic
        raise AssertionError("tampered execution mode was accepted")


def test_reset_readback_uses_semantic_joint_order_not_json_key_order() -> None:
    resets = {
        "finger_joint": 9.0,
        **{f"panda_joint{index}": float(index) for index in range(7, 0, -1)},
    }

    result = _requested_arm_reset(
        plan={"robot": {"joint_reset_positions_rad": resets}},
        servo_binding={
            "arm_joint_names": [f"panda_joint{index}" for index in range(1, 8)]
        },
    )

    assert result == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]


def test_worker_does_not_require_articulated_joint_tolerance_for_rigid_task() -> None:
    assert _task_joint_reset_passed(
        absolute_errors_rad={},
        task_spec={"task_kind": "rigid_pick_place"},
    ) is True


def test_worker_reverifies_frozen_construction_plan_before_native_startup(
    tmp_path: Path,
) -> None:
    runtime_inputs = tmp_path / "runtime_inputs"
    runtime_inputs.mkdir()
    plan = runtime_inputs / "native_task_construction_phase_plan.v1.json"
    plan.write_text('{"plan_digest":"sha256:fixture"}\n', encoding="utf-8")
    manifest = {
        "bound_runtime_inputs": [
            {
                "relative_path": (
                    "runtime_inputs/native_task_construction_phase_plan.v1.json"
                ),
                "size_bytes": plan.stat().st_size,
                "sha256": "sha256:"
                + hashlib.sha256(plan.read_bytes()).hexdigest(),
            }
        ]
    }

    assert verified_construction_phase_plan_path(tmp_path, manifest) == plan
    plan.write_text("tampered\n", encoding="utf-8")
    try:
        verified_construction_phase_plan_path(tmp_path, manifest)
    except RuntimeError as exc:
        assert str(exc) == "native_task_construction_phase_plan_identity_mismatch"
    else:
        raise AssertionError("tampered construction phase plan was accepted")


def test_worker_reverifies_optional_terminal_feedback_bootstrap_input(
    tmp_path: Path,
) -> None:
    runtime_inputs = tmp_path / "runtime_inputs"
    runtime_inputs.mkdir()
    plan = runtime_inputs / "native_task_construction_phase_plan.v1.json"
    adoption = (
        runtime_inputs / "native_construction_terminal_feedback_adoption.v1.json"
    )
    plan.write_text("{}\n", encoding="utf-8")
    adoption.write_text('{"checkpoint_digest":"sha256:fixture"}\n', encoding="utf-8")

    def row(path: Path) -> dict:
        return {
            "relative_path": "runtime_inputs/" + path.name,
            "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    manifest = {"bound_runtime_inputs": [row(plan), row(adoption)]}
    assert verified_construction_phase_plan_path(tmp_path, manifest) == plan
    assert verified_terminal_feedback_adoption_path(tmp_path, manifest) == adoption
    adoption.write_text("tampered\n", encoding="utf-8")
    try:
        verified_terminal_feedback_adoption_path(tmp_path, manifest)
    except RuntimeError as exc:
        assert str(exc) == "native_task_terminal_feedback_adoption_invalid"
    else:
        raise AssertionError("tampered terminal feedback adoption was accepted")


def test_rigid_initial_support_force_is_not_misclassified_as_penetration() -> None:
    sample = {
        "task_robot_contact_peak_force_n": 0.0,
        "task_support_contact_peak_force_n": 20.0,
        "task_scene_collision_peak_force_n": 0.0,
        "robot_scene_contact_peak_force_n": 0.0,
        "robot_task_forbidden_collision_peak_force_n": 0.0,
    }

    assert _initial_contact_blocked(
        task_kind="rigid_pick_place",
        sample=sample,
        collision_threshold_n=1.0,
    ) is False
    sample["task_scene_collision_peak_force_n"] = 2.0
    assert _initial_contact_blocked(
        task_kind="rigid_pick_place",
        sample=sample,
        collision_threshold_n=1.0,
    ) is True
    assert _task_joint_reset_passed(
        absolute_errors_rad={"hinge": 5.0e-5},
        task_spec={"reset_tolerance_rad": 1.0e-4},
    ) is True


def test_worker_arrival_requires_orientation_as_well_as_position() -> None:
    result = _pose_arrival_readback(
        position_world_m=[1.0, 2.0, 3.0],
        target_position_world_m=[1.0, 2.0, 3.0],
        orientation_world_xyzw=[1.0, 0.0, 0.0, 0.0],
        target_orientation_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        position_tolerance_m=0.01,
        orientation_tolerance_rad=0.05,
    )

    assert result["position_error_m"] == 0.0
    assert result["orientation_error_rad"] > 0.05
    assert result["reached"] is False


def test_terminal_arrival_uses_commanded_tcp_orientation_not_body_orientation() -> None:
    result = _terminal_grasp_frame_arrival_readback(
        grasp_pose_world=[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        body_pose_world=[0.9, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        target_position_world_m=[1.0, 2.0, 3.0],
        target_orientation_world_xyzw=[0.0, 0.0, 0.0, 1.0],
        position_tolerance_m=0.01,
        orientation_tolerance_rad=0.05,
    )

    assert result["reached"] is True
    assert result["orientation_error_rad"] == 0.0
    assert result["terminal_grasp_frame_orientation_world_xyzw"] == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert result["terminal_body_orientation_world_xyzw"] == [
        1.0,
        0.0,
        0.0,
        0.0,
    ]


def test_worker_retains_path_samples_for_general_graph_and_rigid_tasks() -> None:
    assert _retain_task_path_samples(
        task_kind="articulated_open_close",
        task_spec={"schema_version": "adp_task_spec.v2"},
    )
    assert _retain_task_path_samples(
        task_kind="rigid_pick_place",
        task_spec={"schema_version": "adp_task_spec.v2"},
    )
    assert not _retain_task_path_samples(
        task_kind="articulated_open_close",
        task_spec={"schema_version": "adp_task_spec.v1"},
    )


def test_articulation_device_binding_names_the_asset_on_the_wrong_device() -> None:
    """A cuda/cpu mismatch must name the asset, not just a kernel argument.

    Isaac Lab raises this from inside a Warp kernel launch:

        Error launching kernel 'get_joint_acc_from_joint_vel', device='cuda:0',
        but input array for argument 'joint_vel' is on device=cpu

    That message cannot say which articulation is wrong, and attempts r6-r9
    each spent ~$0.065 failing on it. This reads every articulation directly
    so one run identifies the asset.
    """

    from types import SimpleNamespace

    from blueprint_pipeline.native_task_arena_construction_worker import (
        _articulation_device_binding,
    )

    def _asset(joint_device: str, joints: int, actuators: int) -> SimpleNamespace:
        return SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=SimpleNamespace(device=joint_device),
                joint_vel=SimpleNamespace(device=joint_device),
                device="cuda:0",
            ),
            joint_names=[f"j{i}" for i in range(joints)],
            actuators={f"a{i}": None for i in range(actuators)},
        )

    scene = {
        "robot": _asset("cuda:0", 13, 8),
        "task_object": _asset("cpu", 5, 0),
    }
    built = SimpleNamespace(
        env=SimpleNamespace(unwrapped=SimpleNamespace(scene=scene)),
        scene_asset_names={"task_object": "task_object"},
    )

    binding = _articulation_device_binding(built, expected_device="cuda:0")

    assert binding["articulations"]["robot"]["on_expected_device"] is True
    offender = binding["articulations"]["task_object"]
    assert offender["on_expected_device"] is False
    assert offender["joint_vel"] == "cpu"
    # the numbers that distinguish the assets are recorded too
    assert offender["num_joints"] == 5
    assert offender["num_actuators"] == 0


def test_device_binding_survives_an_unreachable_scene() -> None:
    """Diagnostics must never replace the real failure with their own."""

    from types import SimpleNamespace

    from blueprint_pipeline.native_task_arena_construction_worker import (
        _articulation_device_binding,
    )

    class _Boom:
        @property
        def scene(self):
            raise RuntimeError("scene not ready")

    built = SimpleNamespace(
        env=SimpleNamespace(unwrapped=_Boom()), scene_asset_names={}
    )

    binding = _articulation_device_binding(built, expected_device="cuda:0")

    assert "unavailable" in binding
    assert binding["expected_device"] == "cuda:0"


def test_articulation_device_binding_skips_plan_declared_base_assets() -> None:
    from types import SimpleNamespace

    from blueprint_pipeline.native_task_arena_construction_worker import (
        _articulation_device_binding,
    )

    articulation = SimpleNamespace(
        data=SimpleNamespace(
            joint_pos=SimpleNamespace(device="cuda:0"),
            joint_vel=SimpleNamespace(device="cuda:0"),
            device="cuda:0",
        ),
        joint_names=["joint"],
        actuators={},
    )
    base = SimpleNamespace()
    scene = {
        "robot": articulation,
        "task_object": articulation,
        "scene_appearance": base,
        "scene_collision": base,
    }
    built = SimpleNamespace(
        env=SimpleNamespace(unwrapped=SimpleNamespace(scene=scene)),
        scene_asset_names={name: name for name in scene if name != "robot"},
        plan={
            "objects": [
                {"name": "task_object", "object_type": "ARTICULATION"},
                {"name": "scene_appearance", "object_type": "BASE"},
                {"name": "scene_collision", "object_type": "BASE"},
            ]
        },
    )

    binding = _articulation_device_binding(built, expected_device="cuda:0")

    assert sorted(binding["articulations"]) == ["robot", "task_object"]
    assert binding["required_articulation_names"] == ["robot", "task_object"]
    assert binding["non_articulation_assets_skipped"] == [
        "scene_appearance",
        "scene_collision",
    ]


def test_articulation_view_devices_separate_scene_wide_from_per_asset() -> None:
    """The traceback cannot say whether the whole scene lost GPU dynamics.

    If every articulation view is CPU-backed the physics scene never got GPU
    dynamics and the fix is scene-level. If exactly one is CPU-backed the fix
    belongs to that asset. Those are different repairs and nothing else in a
    failed run tells them apart. Paths come from the scene plan, so this still
    reports when no stage accessor resolves -- which is exactly what happened
    on attempt r11, where a single missing module returned no evidence at all.
    """

    from types import SimpleNamespace

    from blueprint_pipeline.native_task_arena_construction_worker import (
        _articulation_view_devices,
    )

    devices = {
        "/World/envs/env_0/Robot": "cuda:0",
        "/World/envs/env_0/task_object": "cpu",
    }

    def create_articulation_view(path):
        return SimpleNamespace(
            get_dof_velocities=lambda: SimpleNamespace(device=devices[path]),
            _backend=object(),
        )

    manager = SimpleNamespace(
        _view=SimpleNamespace(create_articulation_view=create_articulation_view)
    )

    rows = _articulation_view_devices(manager, list(devices))

    assert rows["/World/envs/env_0/Robot"]["device"] == "cuda:0"
    assert rows["/World/envs/env_0/task_object"]["device"] == "cpu"
    assert rows["/World/envs/env_0/Robot"]["backend_present"] is True


def test_articulation_view_devices_report_a_missing_simulation_view() -> None:
    """No view at all is a different diagnosis from a CPU-backed view."""

    from types import SimpleNamespace

    from blueprint_pipeline.native_task_arena_construction_worker import (
        _articulation_view_devices,
    )

    rows = _articulation_view_devices(SimpleNamespace(_view=None), ["/World/x"])

    assert rows == {"unavailable": "simulation_view_is_none"}


def test_expected_articulation_prim_paths_come_from_the_plan() -> None:
    """Paths must not require a stage traversal to obtain."""

    from blueprint_pipeline.native_task_arena_construction_worker import (
        expected_articulation_prim_paths,
    )

    plan = {
        "objects": [
            {"object_type": "ARTICULATION", "prim_path": "{ENV_REGEX_NS}/task_object"},
            {"object_type": "BASE", "prim_path": "{ENV_REGEX_NS}/scene_collision"},
        ]
    }

    paths = expected_articulation_prim_paths(plan)

    assert paths == ["/World/envs/env_0/Robot", "/World/envs/env_0/task_object"]


def test_every_evidence_fact_fails_independently() -> None:
    """No single missing module may empty the whole report.

    r11 spent $0.056 and returned nothing because one unavailable import
    (`isaacsim.core.utils`, which this runtime does not ship) aborted the
    entire collection. Each fact is now gathered separately, and every stage
    accessor attempt is recorded by name.
    """

    from blueprint_pipeline.native_task_arena_construction_worker import (
        _STAGE_ACCESSORS,
        physics_scene_device_evidence,
    )

    evidence = physics_scene_device_evidence(["/World/envs/env_0/Robot"])

    # Isaac is absent here, so every fact should report its own reason
    assert "physics_manager_unavailable" in evidence
    assert "simulation_context_unavailable" in evidence
    assert evidence["stage_source"] is None
    # and every accessor must have been tried, by name
    assert set(evidence["stage_attempts"]) == {name for name, _ in _STAGE_ACCESSORS}
    # the accessor that r11 relied on alone is no longer first
    assert _STAGE_ACCESSORS[0][0] == "omni.usd"


def test_warp_arrays_are_serialised_not_returned_raw() -> None:
    """Isaac Lab's physics views return warp arrays, not torch tensors.

    A wp.array has neither `detach` nor `tolist`, so it used to fall straight
    through `_jsonable` unconverted. Nothing failed at that point -- it failed
    far away, at the first use of the result. r12 reached a clean environment
    build (every articulation on cuda:0, device readback passed) and then died
    on `_jsonable(robot.data.root_pose_w)[0]` with "Item indexing is not
    supported on wp.array objects".
    """

    from blueprint_pipeline.native_task_arena_construction_worker import _jsonable

    class _WarpArray:
        """A wp.array as far as _jsonable can tell: numpy() and nothing else."""

        def __init__(self, rows):
            self._rows = rows

        def numpy(self):
            import numpy as np

            return np.array(self._rows)

        def __getitem__(self, index):  # pragma: no cover - must never be hit
            raise RuntimeError("Item indexing is not supported on wp.array objects")

    value = _jsonable(_WarpArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    assert value == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    # the point of the fix: the caller can index the result
    assert value[0] == [1.0, 2.0, 3.0]


def test_torch_tensors_still_take_the_detach_path() -> None:
    """The warp branch must not shadow the tensor branch."""

    from blueprint_pipeline.native_task_arena_construction_worker import _jsonable

    class _Tensor:
        def __init__(self, rows):
            self._rows = rows
            self.detached = False

        def detach(self):
            self.detached = True
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self._rows

    tensor = _Tensor([[7.0, 8.0]])
    assert _jsonable(tensor) == [[7.0, 8.0]]
    assert tensor.detached is True


def test_persist_survives_values_json_cannot_encode() -> None:
    """A receipt that cannot be written destroys the diagnosis of a paid run.

    `_persist` runs from a `finally` and digests *before* it writes, so
    `default=str` on the write alone did not stop a stray warp array from
    raising inside the handler and leaving a paid run with no receipt.
    """

    import json
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from blueprint_pipeline.native_task_arena_construction_worker import (
        _canonical_digest,
        _persist,
    )

    class _Unencodable:
        def __repr__(self) -> str:
            return "<warp array>"

    with TemporaryDirectory() as directory:
        target = Path(directory) / "native_task_arena_construction_result.v1.json"
        _persist(target, {"status": "blocked", "stray": _Unencodable()})

        written = json.loads(target.read_text(encoding="utf-8"))

    assert written["status"] == "blocked"
    assert written["stray"] == "<warp array>"
    assert written["result_digest"] == _canonical_digest(
        written, field="result_digest"
    )


# --- the construction lane's own camera path must read pixels --------------


class _FakeCameraData:
    """Only what `_camera_snapshot` touches, in the shapes Isaac Lab returns."""

    def __init__(self, *, rgb, semantic, labels) -> None:
        import numpy as np

        self.output = {"rgb": rgb[None, ...], "semantic_segmentation": semantic[None, ...]}
        self.info = {"semantic_segmentation": {"idToLabels": labels}}
        self.intrinsic_matrices = np.eye(3, dtype=np.float32)[None, ...]
        self.pos_w = np.zeros((1, 3), dtype=np.float32)
        self.quat_w_opengl = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.frame = 0


class _FakeEnv:
    def __init__(self, cameras) -> None:
        self.unwrapped = type(
            "_Unwrapped", (), {"scene": type("_Scene", (), {
                "__getitem__": lambda _self, name: cameras[name],
            })()}
        )()


def _snapshot_one_camera(*, rgb, semantic, labels, output_root):
    from blueprint_pipeline.native_task_arena_construction_worker import (
        _camera_snapshot,
    )

    camera = type("_Camera", (), {})()
    camera.data = _FakeCameraData(rgb=rgb, semantic=semantic, labels=labels)
    return _camera_snapshot(
        env=_FakeEnv({"external_cam": camera}),
        camera_scene_names={"external": "external_cam"},
        output_root=output_root,
        snapshot_id="reset",
    )


def test_construction_camera_snapshot_fails_a_black_frame(tmp_path) -> None:
    """The exact r13..r23 condition, driven through the lane's real call site.

    `_camera_snapshot` is what both construction snapshot points call (the
    post-reset one and the per-phase one), and `camera_gates` -- which
    `native_task_control_plan` and `native_articulated_control_plan` require to
    be `passed: True` -- is derived from nothing else.
    """

    import numpy as np

    semantic = np.full((64, 64), 7, dtype=np.int32)
    black = np.zeros((64, 64, 3), dtype=np.uint8)

    snapshot = _snapshot_one_camera(
        rgb=black,
        semantic=semantic,
        labels={"7": {"class": "task_object"}},
        output_root=tmp_path,
    )
    observability = snapshot["cameras"][0]["observability"]

    assert observability["semantic_passed"] is True
    assert observability["passed"] is False
    assert observability["rgb_or_model_label_used"] is True
    assert "native_task_camera_rgb_frame_void" in observability["blockers"]


def test_construction_camera_snapshot_restores_flat_proxy_semantics(tmp_path) -> None:
    import numpy as np

    semantic = np.zeros((32, 48), dtype=np.int32)
    semantic[8:24, 12:36] = 7
    frame = np.full((32, 48, 3), 96, dtype=np.uint8)
    camera = type("_Camera", (), {})()
    camera.data = _FakeCameraData(
        rgb=frame,
        semantic=semantic,
        labels={"7": {"class": "task_object"}},
    )
    camera.data.output["semantic_segmentation"] = semantic.reshape(1, -1, 1)

    from blueprint_pipeline.native_task_arena_construction_worker import (
        _camera_snapshot,
    )

    snapshot = _camera_snapshot(
        env=_FakeEnv({"external_cam": camera}),
        camera_scene_names={"external": "external_cam"},
        output_root=tmp_path,
        snapshot_id="flat_proxy",
    )

    row = snapshot["cameras"][0]
    assert row["raw_shapes"]["semantic_raw_shape"] == [1, 1536, 1]
    assert row["raw_shapes"]["semantic_image_shape"] == [32, 48]
    assert row["observability"]["semantic_passed"] is True
    assert row["observability"]["pixel_count"] == 384
    diagnostics = json.loads(
        (tmp_path / "native_task_camera_snapshot_diagnostics.v1.json").read_text()
    )
    assert diagnostics["cameras"][0]["semantic_image_shape"] == [32, 48]


def test_construction_camera_snapshot_refuses_when_the_claimed_site_is_void(
    tmp_path,
) -> None:
    """A NuRec-capable image makes site appearance a required pixel claim."""

    import numpy as np

    from blueprint_pipeline.native_task_arena_construction_worker import (
        SITE_APPEARANCE_RENDER_EXPECTED,
    )

    assert SITE_APPEARANCE_RENDER_EXPECTED is True

    generator = np.random.default_rng(20260819)
    semantic = np.zeros((64, 64), dtype=np.int32)
    semantic[24:40, 24:40] = 7
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[22:42, 22:42] = generator.integers(60, 200, size=(20, 20, 3), dtype=np.uint8)

    snapshot = _snapshot_one_camera(
        rgb=frame,
        semantic=semantic,
        labels={"7": {"class": "task_object"}},
        output_root=tmp_path,
    )
    observability = snapshot["cameras"][0]["observability"]
    assert observability["passed"] is False
    assert observability["site_appearance_claimed"] is False
    assert (
        "native_task_camera_rgb_site_void_fraction_above_ceiling"
        in observability["blockers"]
    )


def test_construction_camera_frame_is_retained_before_the_gate_can_refuse(
    tmp_path,
) -> None:
    """A refusal aborts the run from inside the measurement.

    If the frame were written after the verdict, the one artifact that explains
    the refusal would never reach the receipt.
    """

    import numpy as np
    import pytest

    from blueprint_pipeline.native_task_camera_observability import (
        NativeTaskCameraObservabilityError,
    )

    with pytest.raises(NativeTaskCameraObservabilityError):
        _snapshot_one_camera(
            rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            semantic=np.full((64, 64), 7, dtype=np.int32),
            labels={"7": {"class": "task_object"}},
            output_root=tmp_path,
        )

    assert (tmp_path / "construction_frames" / "external" / "reset.png").is_file()


def test_front_entry_construction_uses_off_sim_multistart_then_native_replay() -> None:
    import inspect

    from blueprint_pipeline.native_task_arena_construction_worker import main

    source = inspect.getsource(main)
    assert "solve_grasp_target_multistart" in source
    assert "action_for_joint_target" in source
    assert '"physics_steps_performed": 0' in source
    assert "native_execution_remains_" in source
    assert "reset_grasp_pose = servo.current_grasp_frame_pose_world()" in source
    assert "reset_body_pose = servo.current_body_pose_world()" not in source
