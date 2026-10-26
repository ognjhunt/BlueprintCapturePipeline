"""A native policy episode renders its cameras only on steps whose frames it uses.

Run 57749451 (pi05 cell 00, 2026-10-26) spent 227 s of a 441 s episode inside
``env.step()``.  The pinned Isaac Lab renders lazily: reading a camera after
physics advanced is what pulls the RTX ``app.update()``.  Arena's camera
observation terms read every camera after every step, so all three cameras
rendered at the full 15 Hz control rate while the episode uses frames only at
policy queries, review samples, and terminal and readiness observations.

This fake executes the pinned Isaac Lab bytes that decide when a render happens
(``ManagerBasedRLEnv.step``, ``ManagerBasedEnv.reset``,
``SimulationContext.step/render/update_visualizers``, the recording hook, and
``ensure_isaac_rtx_render_update``) and models only the warp-kernel camera
bookkeeping that is pinned beside them.  Every render writes the physics step
it saw into the pixels it produces, so a stale frame would be visible in the
retained PNGs themselves.
"""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.adp009d_droid_observation import DROID_EXTERIOR_VIEW_1
from blueprint_pipeline.adp009d_isaac_episode_adapter import (
    DEFAULT_CAMERA_SCENE_NAMES,
    IsaacEpisodeAdapter,
    IsaacEpisodeAdapterError,
)
from blueprint_pipeline.adp009d_policy_episode import (
    EVALUATION_REVIEW_FRAME_STRIDE_STEPS,
    run_policy_episode,
)
from blueprint_pipeline.adp009d_task_scoring import CAN_START_POSITION_M
from blueprint_pipeline.native_task_arena_runtime import (
    ARENA_CAMERA_OBSERVATION_GROUP,
    drop_per_step_camera_observations,
)
from blueprint_pipeline.native_task_runtime_source_packet import (
    ARENA_COMMIT,
    ISAACLAB_COMMIT,
)
from tests.test_adp009d_isaac_episode_adapter import _Robot
from tests.test_adp009d_policy_episode import (
    _DESTINATION,
    _MEASURED,
    _LifecyclePolicy,
    _integrity,
    _observation_integrity_authority,
)

torch = pytest.importorskip("torch")

_FIXTURES = Path(__file__).parent / "fixtures"
_RENDER_PATH = json.loads(
    (_FIXTURES / "isaaclab_render" / "pinned_render_path.json").read_text(
        encoding="utf-8"
    )
)
_RESET_PATH = json.loads(
    (_FIXTURES / "isaaclab_reset" / "manager_reset_method.json").read_text(
        encoding="utf-8"
    )
)
_CONTROL_HZ = 15
_DECIMATION = 8
_PHYSICS_DT = 1.0 / (_CONTROL_HZ * _DECIMATION)
_CAMERA_BASE = {"external_camera": 40, "wrist_camera": 80, "external_camera_2": 120}
_EPISODE_ID = "render-cadence"


def _compile(source: str, filename: str, namespace: dict[str, Any], name: str) -> Any:
    exec(compile("from __future__ import annotations\n" + source, filename, "exec"), namespace)
    return namespace[name]


def _pinned(key: str, namespace: dict[str, Any]) -> Any:
    """Compile one pinned Isaac Lab function into ``namespace`` and return it."""

    method = _RENDER_PATH["methods"][key]
    return _compile(
        method["source"],
        method["source_path"],
        namespace,
        method["qualified_name"].rsplit(".", 1)[-1],
    )


def _rendered_image(camera_name: str, moment: dict[str, Any]) -> np.ndarray:
    """What the RTX annotator holds after one render, stamped with its moment."""

    yy, xx = np.indices((24, 32))
    base = _CAMERA_BASE[camera_name]
    image = np.stack(
        [
            base + (xx % 17),
            base + (yy % 13),
            base + ((xx + yy) % 19),
            np.full_like(xx, 255),
        ],
        axis=-1,
    ).astype(np.uint8)
    physics_step = int(moment["physics_step"])
    env_step = int(moment["env_step"])
    image[0, 0, :3] = (physics_step % 256, physics_step // 256, moment["reset_generation"])
    image[0, 1, :3] = (env_step % 256, env_step // 256, 255)
    return image


def _decoded_moment(rgb: np.ndarray) -> dict[str, Any]:
    pixel = np.asarray(rgb)
    return {
        "physics_step": int(pixel[0, 0, 0]) + 256 * int(pixel[0, 0, 1]),
        "reset_generation": int(pixel[0, 0, 2]),
        "env_step": int(pixel[0, 1, 0]) + 256 * int(pixel[0, 1, 1]),
        "rendered": int(pixel[0, 1, 2]) == 255,
    }


class _Camera:
    """Pinned ``Camera``/``SensorBase`` lazy-buffer bookkeeping, minus warp.

    ``update`` and ``reset`` mirror ``update_timestamp_kernel`` and
    ``reset_envs_kernel``.  Reading ``data`` refreshes an outdated buffer
    through ``ensure_isaac_rtx_render_update`` exactly as
    ``Camera._update_buffers_impl`` -> ``RenderContext.render_into_camera`` ->
    ``IsaacRtxRenderer.render`` does, then copies the annotator output.  The
    wrist camera carries the production native attachment, whose
    ``get_world_poses`` writes the live mount pose and calls ``sim.render()``
    during ``Camera._update_poses``, before the camera's RTX update.
    """

    def __init__(self, world: _PinnedIsaacWorld, name: str) -> None:
        self._world = world
        self.name = name
        self._writes_live_mount_pose = name == DEFAULT_CAMERA_SCENE_NAMES["wrist"]
        self.cfg = SimpleNamespace(
            update_period=0.0, spawn=SimpleNamespace(clipping_range=(0.01, 10.0))
        )
        self.frame = np.zeros(1, dtype=np.int64)
        self._is_outdated = True
        self._timestamp = 0.0
        self._timestamp_last_update = 0.0
        self._data = SimpleNamespace(
            output={},
            intrinsic_matrices=np.array(
                [[[20.0, 0.0, 16.0], [0.0, 20.0, 12.0], [0.0, 0.0, 1.0]]]
            ),
            pos_w=np.array([[0.0, 0.0, 1.0]]),
            quat_w_opengl=np.array([[0.0, 0.0, 0.0, 1.0]]),
        )

    @property
    def data(self) -> SimpleNamespace:
        self._update_outdated_buffers()
        return self._data

    def update(self, dt: float, force_recompute: bool = False) -> None:
        self._timestamp += dt
        if self._timestamp - self._timestamp_last_update + 1e-6 >= self.cfg.update_period:
            self._is_outdated = True
        if force_recompute:
            self._update_outdated_buffers()

    def reset(self) -> None:
        self._timestamp = 0.0
        self._timestamp_last_update = 0.0
        self._is_outdated = True
        self.frame[:] = 0

    def _update_outdated_buffers(self) -> None:
        if not self._is_outdated:
            return
        self.frame += 1
        if self._writes_live_mount_pose:
            # NativeAttachedCameraView.get_world_poses: pose write, then render.
            self._world.sim.render()
        self._world.rtx["ensure_isaac_rtx_render_update"]()
        image = self._world.render_products[self.name]
        self._data.output = {"rgb": image[np.newaxis].copy()}
        self._timestamp_last_update = self._timestamp
        self._is_outdated = False
        self._world.buffer_refreshes.append(
            {**self._world.moment(), "camera": self.name, "rendered": _decoded_moment(image)}
        )


class _PhysicsManager:
    def __init__(self, world: _PinnedIsaacWorld) -> None:
        self._world = world

    def wait_for_playing(self) -> None:
        pass

    def step(self) -> None:
        self._world.robot.data.joint_pos[0, :7] = self._world.joint_targets

    def forward(self) -> None:
        pass

    def pre_render(self) -> None:
        pass

    def after_visualizers_render(self) -> None:
        pass


class _SimulationContext:
    """The ``SimulationContext`` state the pinned render path reads."""

    # Headless with RTX camera sensors: ``is_rendering`` is true and no Kit
    # visualizer or headless video recorder is attached -- the production
    # ``AppLauncher(headless=True, enable_cameras=True)`` configuration.
    is_rendering = True

    def __init__(self, world: _PinnedIsaacWorld) -> None:
        self.physics_manager = _PhysicsManager(world)
        self._visualizers: list[Any] = []
        self._physics_step_count = 0
        self._render_generation = 0
        self._settings = {
            "/isaaclab/render/rtx_sensors": True,
            "/isaaclab/video/enabled": False,
        }

    @property
    def render_generation(self) -> int:
        return self._render_generation

    @property
    def visualizers(self) -> list[Any]:
        return list(self._visualizers)

    def get_setting(self, name: str) -> Any:
        return self._settings.get(name)

    def set_setting(self, name: str, value: Any) -> None:
        self._settings[name] = value

    def get_rendering_dt(self) -> float:
        return _PHYSICS_DT * _DECIMATION

    def forward(self) -> None:
        self.physics_manager.forward()

    def _should_forward_before_visualizer_update(self) -> bool:
        return False


class _ObservationManager:
    """Compute every configured group; a ``None`` group is skipped, as pinned."""

    def __init__(self, world: _PinnedIsaacWorld, observations: Any) -> None:
        self._world = world
        self._observations = observations

    def compute(self, update_history: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for group_name, group in vars(self._observations).items():
            if group is None:
                continue
            result[group_name] = {
                term_name: self._world.read_observation_term(term)
                for term_name, term in group.terms.items()
            }
        return result


def _arena_observations() -> SimpleNamespace:
    """Arena's composed groups: state terms plus one image term per camera.

    Terms follow ``DroidCameraCfg`` field order, as Arena composes them.
    """

    return SimpleNamespace(
        policy=SimpleNamespace(terms={"joint_pos": ("robot", "joint_pos")}),
        **{
            ARENA_CAMERA_OBSERVATION_GROUP: SimpleNamespace(
                terms={
                    f"{name}_rgb": ("camera", name)
                    for name in ("external_camera", "external_camera_2", "wrist_camera")
                }
            )
        },
    )


class _PinnedIsaacWorld:
    """One single-env Arena environment whose render decisions are pinned bytes."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        drop_camera_observations: bool,
        num_rerenders_on_reset: int = 1,
    ) -> None:
        self.renders: list[dict[str, Any]] = []
        self.buffer_refreshes: list[dict[str, Any]] = []
        self.env_steps = 0
        self.reset_generation = 0
        self.inside_env_step = False
        self.physics_step_at: dict[tuple[int, int], int] = {}
        self.robot = _Robot()
        self._reset_joint_pos = self.robot.data.joint_pos.copy()
        self.joint_targets = self._reset_joint_pos[0, :7].copy()
        self.can = SimpleNamespace(
            data=SimpleNamespace(
                root_pose_w=np.array([[*CAN_START_POSITION_M, 0.0, 0.0, 0.0, 1.0]])
            )
        )
        self.cameras = {
            name: _Camera(self, name) for name in DEFAULT_CAMERA_SCENE_NAMES.values()
        }
        unrendered = np.zeros((24, 32, 4), dtype=np.uint8)
        self.render_products = {name: unrendered for name in self.cameras}
        self.sim = _SimulationContext(self)
        self._install_pinned_render_path(monkeypatch)
        cfg = SimpleNamespace(
            decimation=_DECIMATION,
            sim=SimpleNamespace(dt=_PHYSICS_DT, render_interval=_DECIMATION),
            num_rerenders_on_reset=num_rerenders_on_reset,
            wait_for_textures=False,
            observations=_arena_observations(),
        )
        self.camera_observations_dropped = (
            drop_per_step_camera_observations(cfg)
            if drop_camera_observations
            else False
        )
        self.unwrapped = self._arena_env(cfg)
        self.env = _GymEnv(self)

    def _install_pinned_render_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        world = self
        kit_app = SimpleNamespace(update=self.kit_app_update)
        omni = types.ModuleType("omni")
        omni.kit = types.ModuleType("omni.kit")
        omni.kit.app = types.ModuleType("omni.kit.app")
        omni.kit.app.get_app = lambda: kit_app
        monkeypatch.setitem(sys.modules, "omni", omni)
        monkeypatch.setitem(sys.modules, "omni.kit", omni.kit)
        monkeypatch.setitem(sys.modules, "omni.kit.app", omni.kit.app)

        def streaming_never_stalls() -> None:
            raise AssertionError("stage streaming is idle in this rehearsal")

        self.rtx: dict[str, Any] = {
            "logger": logging.getLogger("isaac_rtx_renderer_utils"),
            "sim_utils": SimpleNamespace(
                SimulationContext=SimpleNamespace(instance=lambda: world.sim)
            ),
            "_last_render_update_key": (0, -1, -1),
            "_get_stage_streaming_busy": lambda: False,
            "_wait_for_streaming_complete": streaming_never_stalls,
        }
        _pinned("isaac_rtx_renderer_utils.ensure_isaac_rtx_render_update", self.rtx)
        _pinned(
            "isaac_rtx_renderer_utils.pump_kit_app_for_headless_video_render_if_needed",
            self.rtx,
        )
        renderers = types.ModuleType("isaaclab_physx.renderers")
        utils = types.ModuleType("isaaclab_physx.renderers.isaac_rtx_renderer_utils")
        utils.pump_kit_app_for_headless_video_render_if_needed = self.rtx[
            "pump_kit_app_for_headless_video_render_if_needed"
        ]
        renderers.isaac_rtx_renderer_utils = utils
        physx = types.ModuleType("isaaclab_physx")
        physx.renderers = renderers
        monkeypatch.setitem(sys.modules, "isaaclab_physx", physx)
        monkeypatch.setitem(sys.modules, "isaaclab_physx.renderers", renderers)
        monkeypatch.setitem(
            sys.modules, "isaaclab_physx.renderers.isaac_rtx_renderer_utils", utils
        )

        hooks: dict[str, Any] = {}
        _pinned("recording_hooks._recording_followup_isaac_sim", hooks)
        _pinned("recording_hooks.run_recording_hooks_after_visualizers", hooks)
        context = {
            "logger": logging.getLogger("simulation_context"),
            "run_recording_hooks_after_visualizers": hooks[
                "run_recording_hooks_after_visualizers"
            ],
        }
        for key in ("step", "render", "update_visualizers"):
            method = _pinned(f"simulation_context.{key}", dict(context))
            setattr(self.sim, key, MethodType(method, self.sim))

    def _arena_env(self, cfg: SimpleNamespace) -> SimpleNamespace:
        world = self
        env = SimpleNamespace(
            cfg=cfg,
            sim=self.sim,
            scene=_Scene(self),
            device="cpu",
            num_envs=1,
            render_enabled=True,
            has_rtx_sensors=True,
            _physics_handles_decimation=False,
            _sim_step_counter=0,
            physics_dt=_PHYSICS_DT,
            step_dt=_PHYSICS_DT * _DECIMATION,
            episode_length_buf=torch.zeros(1, dtype=torch.long),
            common_step_counter=0,
            extras={},
            obs_buf={},
            action_manager=_ActionManager(self),
            recorder_manager=SimpleNamespace(
                active_terms=[],
                record_pre_step=lambda: None,
                record_post_physics_decimation_step=lambda: None,
                record_post_step=lambda: None,
                record_pre_reset=lambda env_ids: None,
                record_post_reset=lambda env_ids: None,
            ),
            termination_manager=SimpleNamespace(
                compute=lambda: torch.zeros(1, dtype=torch.bool),
                terminated=torch.zeros(1, dtype=torch.bool),
                time_outs=torch.zeros(1, dtype=torch.bool),
            ),
            reward_manager=SimpleNamespace(compute=lambda dt: torch.zeros(1)),
            command_manager=SimpleNamespace(compute=lambda dt: None),
            event_manager=SimpleNamespace(available_modes=[]),
            observation_manager=_ObservationManager(self, cfg.observations),
            seed=lambda seed: seed,
            _reset_idx=lambda env_ids: world.reset_state(),
        )
        env.step = MethodType(_pinned("manager_based_rl_env.step", {"torch": torch}), env)
        env.reset = MethodType(
            _compile(
                _RESET_PATH["reset_method"],
                _RESET_PATH["source_path"],
                {"torch": torch},
                "reset",
            ),
            env,
        )
        return env

    def moment(self) -> dict[str, Any]:
        return {
            "reset_generation": self.reset_generation,
            "env_step": self.env_steps,
            "physics_step": self.sim._physics_step_count,
            "inside_env_step": self.inside_env_step,
        }

    def kit_app_update(self) -> None:
        """One RTX frame: every render product sees the current physics state."""

        moment = self.moment()
        self.renders.append(moment)
        self.render_products = {
            name: _rendered_image(name, moment) for name in self.cameras
        }

    def read_observation_term(self, term: tuple[str, str]) -> Any:
        kind, name = term
        if kind == "camera":
            # ``mdp.image``: sensor.data.output[data_type], cloned.
            return self.cameras[name].data.output["rgb"].copy()
        return self.robot.data.joint_pos.copy()

    def reset_state(self) -> None:
        self.reset_generation += 1
        self.env_steps = 0
        self.robot.data.joint_pos[...] = self._reset_joint_pos
        self.joint_targets = self._reset_joint_pos[0, :7].copy()
        for camera in self.cameras.values():
            camera.reset()
        self.physics_step_at[(self.reset_generation, 0)] = self.sim._physics_step_count


class _Scene(dict):
    """``InteractiveScene`` with ``lazy_sensor_update=True`` (Arena's default)."""

    def __init__(self, world: _PinnedIsaacWorld) -> None:
        super().__init__(world.cameras)
        self._world = world

    def write_data_to_sim(self) -> None:
        pass

    def update(self, dt: float) -> None:
        for camera in self._world.cameras.values():
            camera.update(dt, force_recompute=False)


class _ActionManager:
    total_action_dim = 8

    def __init__(self, world: _PinnedIsaacWorld) -> None:
        self._world = world
        self._action: np.ndarray | None = None

    def process_action(self, action: Any) -> None:
        self._action = action.detach().cpu().numpy()[0]

    def apply_action(self) -> None:
        self._world.joint_targets = self._action[:7].copy()


class _GymEnv:
    """The ``gym.make`` wrapper the adapter drives; Isaac Lab code is ``unwrapped``."""

    def __init__(self, world: _PinnedIsaacWorld) -> None:
        self._world = world
        self.unwrapped = world.unwrapped

    def step(self, action: Any) -> Any:
        # Anything rendered during this call shows the step it executes.
        self._world.env_steps += 1
        self._world.inside_env_step = True
        try:
            result = self.unwrapped.step(action)
        finally:
            self._world.inside_env_step = False
        self._world.physics_step_at[
            (self._world.reset_generation, self._world.env_steps)
        ] = self.unwrapped.sim._physics_step_count
        return result

    def reset(self, seed: int | None = None, options: Any = None) -> Any:
        return self.unwrapped.reset(seed=seed, options=options)


def _adapter(world: _PinnedIsaacWorld) -> IsaacEpisodeAdapter:
    return IsaacEpisodeAdapter(
        env=world.env,
        robot=world.robot,
        rigid_task_object=world.can,
        action_dim=8,
        reset_seed=20260806,
        to_torch=lambda value: value,
        gripper_closed_width_m=0.0,
        gripper_open_width_m=0.06,
        simulation_step_seconds=1.0 / _CONTROL_HZ,
    )


def _run_episode(
    world: _PinnedIsaacWorld,
    output_dir: Path,
    *,
    max_policy_queries: int = 4,
    settle_window_samples: int = 10,
) -> dict[str, Any]:
    return run_policy_episode(
        environment=_adapter(world),
        policy=_LifecyclePolicy(),
        candidate_id="pi05_droid",
        destination_position_world_m=_DESTINATION,
        prompt="pick up the can and place it on the counter",
        gripper=_MEASURED,
        max_policy_queries=max_policy_queries,
        settle_window_samples=settle_window_samples,
        media_output_dir=output_dir,
        episode_id=_EPISODE_ID,
        require_complete_multicamera_media=True,
        require_prestart_readiness=True,
        observation_integrity=_integrity(_observation_integrity_authority()),
    )


def _frame_reading_steps(
    *, max_policy_queries: int, open_loop_horizon: int, settle_window_samples: int
) -> set[int]:
    """Control steps at which the episode loop reads camera frames."""

    action_steps = max_policy_queries * open_loop_horizon
    last_step = action_steps + settle_window_samples
    queries = {index * open_loop_horizon for index in range(max_policy_queries)}
    review = set(
        range(
            EVALUATION_REVIEW_FRAME_STRIDE_STEPS,
            last_step + 1,
            EVALUATION_REVIEW_FRAME_STRIDE_STEPS,
        )
    )
    return queries | review | {last_step}


def _retained_frames(output_dir: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    manifest = json.loads(
        (output_dir / "media" / _EPISODE_ID / "multicamera_frame_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    observations = [
        *manifest["policy_input_observations"],
        *manifest["review_observations"],
        manifest["terminal_observation"],
    ]
    return [
        (view, _decoded_moment(np.asarray(Image.open(output_dir / view["relative_path"]))))
        for observation in observations
        for view in observation["views"].values()
    ]


def test_episode_renders_only_on_frame_reading_steps_and_never_inside_env_step(
    monkeypatch, tmp_path
) -> None:
    world = _PinnedIsaacWorld(monkeypatch, drop_camera_observations=True)
    assert world.camera_observations_dropped is True

    receipt = _run_episode(world, tmp_path)

    # Non-consuming steps issue no render: nothing rendered inside env.step().
    assert [row for row in world.renders if row["inside_env_step"]] == []
    # The prestart rehearsal reads its reset frame and its one probe step; the
    # episode (third reset) reads at queries, review samples, and the end.
    episode_generation = 3
    expected = {(1, 0), (1, 1)} | {
        (episode_generation, step)
        for step in _frame_reading_steps(
            max_policy_queries=4, open_loop_horizon=8, settle_window_samples=10
        )
    }
    assert receipt["environment_steps"] == 42
    rendered = [(row["reset_generation"], row["env_step"]) for row in world.renders]
    assert sorted(rendered) == sorted(expected)  # exactly one render per step
    read = {(row["reset_generation"], row["env_step"]) for row in world.buffer_refreshes}
    assert read == expected
    # Every frame read follows a render issued at that same step, and its
    # pixels carry that step's physics state rather than an earlier one.
    for row in world.buffer_refreshes:
        assert row["rendered"] == {
            "physics_step": row["physics_step"],
            "reset_generation": row["reset_generation"],
            "env_step": row["env_step"],
            "rendered": True,
        }
    cadence = receipt["performance_diagnostics"]["camera_render_cadence"]
    assert cadence["renders_issued"] == len(
        [row for row in rendered if row[0] == episode_generation]
    )
    assert cadence["render_seconds"] >= 0.0


def test_retained_frames_are_rendered_at_their_own_control_step(
    monkeypatch, tmp_path
) -> None:
    world = _PinnedIsaacWorld(monkeypatch, drop_camera_observations=True)

    receipt = _run_episode(world, tmp_path)

    frames = _retained_frames(tmp_path)
    assert len(frames) == 3 * (4 + 5 + 1)  # three cameras; queries, reviews, end
    for view, decoded in frames:
        env_step = round(view["simulation_time_s"] * _CONTROL_HZ)
        assert decoded == {
            "physics_step": world.physics_step_at[(3, env_step)],
            "reset_generation": 3,
            "env_step": env_step,
            "rendered": True,
        }
        render = view["synchronization"]["render"]
        assert render["control_step_index"] == env_step
        assert render["source"] == "adapter_render_before_first_frame_read"
    # The native frame counter advances once per rendered step, so the
    # evidence index's strictly increasing freshness check still holds.
    frame_indices = [
        row["cameras"]["external"]["frame_index"] for row in receipt["sensor_freshness"]
    ]
    assert frame_indices == list(range(1, len(frame_indices) + 1))
    receipt_path = tmp_path / f"{_EPISODE_ID}.v4.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    from blueprint_pipeline.adp_episode_evidence_index import _episode_row

    assert _episode_row(tmp_path, receipt_path)["episode_id"] == _EPISODE_ID


def test_arena_camera_observation_terms_render_every_control_step(
    monkeypatch, tmp_path
) -> None:
    """The regression this gate removes, reproduced with the same pinned bytes."""

    gated_dir, legacy_dir = tmp_path / "gated", tmp_path / "legacy"
    gated = _PinnedIsaacWorld(monkeypatch, drop_camera_observations=True)
    _run_episode(gated, gated_dir)
    legacy = _PinnedIsaacWorld(monkeypatch, drop_camera_observations=False)
    receipt = _run_episode(legacy, legacy_dir)

    inside_step = [row for row in legacy.renders if row["inside_env_step"]]
    steps = {(row["reset_generation"], row["env_step"]) for row in inside_step}
    # The prestart probe step and every episode step rendered, twice each: the
    # external camera's term pulled one update, then the wrist mount write
    # forced a second.  Each of the three resets did the same.
    assert len(steps) == 1 + receipt["environment_steps"]
    assert len(inside_step) == 2 * len(steps)
    assert len(legacy.renders) == 2 * (len(steps) + 3)
    # The gate renders once per frame-reading step, wrist first.
    assert len(gated.renders) == 9
    # Gating removes renders nobody read; the frames that were read are the
    # same bytes in both runs.
    gated_frames = [view for view, _ in _retained_frames(gated_dir)]
    legacy_frames = [view for view, _ in _retained_frames(legacy_dir)]
    assert [view["raw_rgb_sha256"] for view in gated_frames] == [
        view["raw_rgb_sha256"] for view in legacy_frames
    ]


def test_read_after_reset_renders_even_without_an_isaac_reset_rerender(
    monkeypatch,
) -> None:
    """A reset moves no physics step, so only the render generation separates
    the pre- and post-reset frames in the RTX update key."""

    world = _PinnedIsaacWorld(
        monkeypatch, drop_camera_observations=True, num_rerenders_on_reset=0
    )
    adapter = _adapter(world)
    adapter.reset()
    adapter.step([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    first = adapter.read_policy_inputs()[DROID_EXTERIOR_VIEW_1]
    assert _decoded_moment(first)["reset_generation"] == 1

    adapter.reset()
    fresh = adapter.read_evaluation_camera_inputs()

    assert all(
        _decoded_moment(image)["reset_generation"] == 2 for image in fresh.values()
    )
    assert world.renders[-1]["reset_generation"] == 2


def test_read_after_reset_is_stale_without_the_adapter_render(monkeypatch) -> None:
    """Without ``sim.render()`` before the read, the pinned dedupe key matches
    the pre-reset render and the annotator serves the old pose."""

    world = _PinnedIsaacWorld(
        monkeypatch, drop_camera_observations=True, num_rerenders_on_reset=0
    )
    adapter = _adapter(world)
    adapter.reset()
    adapter.step([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    adapter.read_policy_inputs()
    adapter.reset()
    world.sim.render = lambda *args, **kwargs: None

    stale = adapter.read_evaluation_camera_inputs()

    assert {_decoded_moment(image)["reset_generation"] for image in stale.values()} == {1}


def test_physics_only_reads_never_render(monkeypatch) -> None:
    world = _PinnedIsaacWorld(monkeypatch, drop_camera_observations=True)
    adapter = _adapter(world)
    adapter.reset()
    for _ in range(3):
        adapter.step([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        adapter.read_arm_joint_positions()
        adapter.read_object_sample()
        adapter.joint_limits()

    assert world.renders == []
    assert world.buffer_refreshes == []


def test_frame_read_without_a_render_seam_is_refused() -> None:
    world_env = SimpleNamespace(unwrapped=SimpleNamespace(scene={}, device="cpu"))
    adapter = IsaacEpisodeAdapter(
        env=world_env,
        robot=_Robot(),
        rigid_task_object=SimpleNamespace(),
        action_dim=8,
        reset_seed=1,
        to_torch=lambda value: value,
        gripper_closed_width_m=0.0,
        gripper_open_width_m=0.06,
        simulation_step_seconds=1.0 / _CONTROL_HZ,
    )

    for read in (
        adapter.read_policy_inputs,
        adapter.read_evaluation_camera_inputs,
        adapter.read_control_observation_metadata,
    ):
        with pytest.raises(
            IsaacEpisodeAdapterError, match="isaac_episode_frame_render_seam_missing"
        ):
            read()


def test_pinned_render_path_matches_the_runtime_isaaclab_and_arena_pins() -> None:
    """The modelled camera bookkeeping is checked against the bytes it imitates.

    Bumping either runtime pin fails here until the render path is re-read.
    """

    assert _RENDER_PATH["isaaclab_commit"] == ISAACLAB_COMMIT
    arena = _RENDER_PATH["arena"]
    assert arena["commit"] == ARENA_COMMIT
    # The group the runtime drops is the one Arena composes from the cameras.
    assert f'("{ARENA_CAMERA_OBSERVATION_GROUP}", AutoCameraObsCfg' in arena["source"]
    assert "func=mdp.image" in arena["source"]
    methods = {key: row["source"] for key, row in _RENDER_PATH["methods"].items()}
    assert "self._update_outdated_buffers()" in methods["camera.data"]
    assert "frame_op=1" in methods["camera._update_buffers_impl"]
    assert "render_into_camera" in methods["camera._update_buffers_impl"]
    rtx_render = methods["isaac_rtx_renderer.render"]
    assert rtx_render.index("ensure_isaac_rtx_render_update()") < rtx_render.index(
        "annotator.get_data()"
    )
    assert ">= update_period" in methods["sensor_kernels.update_timestamp_kernel"]
    assert "is_outdated[env] = True" in methods["sensor_kernels.reset_envs_kernel"]
    assert "force_recompute or self._is_visualizing" in methods["sensor_base.update"]
    # A read refreshes only outdated buffers, and only once per physics step
    # and render generation does the renderer pump Kit.
    ensure = methods["isaac_rtx_renderer_utils.ensure_isaac_rtx_render_update"]
    assert "key = (id(sim), sim._physics_step_count, render_generation)" in ensure
    assert "omni.kit.app.get_app().update()" in ensure
