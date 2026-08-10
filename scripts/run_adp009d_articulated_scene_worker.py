#!/usr/bin/env python3
"""Compose an articulated task scene with a robot in it and run the control pair.

The hinge probe answered whether the door opens. This answers whether an arm
can be put in front of it: the twin spawned as an articulation, the scene
collision and appearance behind it, a Droid-configured Franka at the resolved
base, and the same task-neutral control pair the rigid lane uses.

Almost nothing here is new logic. The episode adapter already accepts a task
sample callback, the scorer already handles the articulated task kind, and
`run_task_neutral_controls` already runs a negative and a positive from a frozen
plan. What was missing was a composition that puts an articulated object and a
robot in one scene, which is what this is.

Every step is guarded by name. This file cannot be run off a GPU, so the only
debugging channel it has is the retained result, and a launch that comes back
saying something failed without saying which step is a launch spent for nothing.
The spawn type is checked rather than assumed for the same reason: a
refrigerator spawned rigid has frozen joints, raises nothing at all, and reads
as an impossible task rather than a misconfigured one.

The result is persisted before the simulator is closed. Isaac's close can end
the process outright, and a run that dies after the app opens would otherwise
take its own diagnosis with it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

# The bundle does not keep the runtime module and the modules it imports in
# one directory: this file is installed as provider_runtime/<runtime>.py while
# extra natives land in provider_runtime/native/. Both are added before the
# import so a flat bundle resolves without depending on the working directory,
# and the repository src tree stays the last resort it is.
for _candidate in (
    Path(__file__).resolve().parent / "native",
    Path(__file__).resolve().parent,
):
    import sys as _sys

    if _candidate.is_dir() and str(_candidate) not in _sys.path:
        _sys.path.insert(0, str(_candidate))

try:  # flat provider bundle
    from runtime_asset_resolution import (
        RuntimeAssetResolutionError,
        resolve_runtime_asset,
    )
    from gripper_convention_probe import measure_gripper_convention
    from articulated_scene_observations import build_scene_observations
except ModuleNotFoundError:  # repository checkout
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from blueprint_pipeline.runtime_asset_resolution import (
        RuntimeAssetResolutionError,
        resolve_runtime_asset,
    )
    from blueprint_pipeline.gripper_convention_probe import (
        measure_gripper_convention,
    )
    from blueprint_pipeline.articulated_scene_observations import (
        build_scene_observations,
    )


RESULT_SCHEMA_VERSION = "adp009d_articulated_scene_result.v1"
# The Arena lane reads a runner before it will launch it, and these are the
# facts it insists a runner carry. None are arbitrary: the filename its
# collector looks for, the two revisions whose provenance the receipt must
# record, and whether a learned policy was consulted.
RESULT_FILENAME = "adp009d_native_microcheck.json"
ARENA_REVISION = "8b4a3a47fc53de23e8205089d71109a2e2348acd"
ISAAC_LAB_REVISION = "e57379c634b42db5a0fe9f754341be6e2a7c7c43"
# Scripted controls only. This composition never consults a learned policy, and
# saying otherwise would overclaim on the one axis the program cares most about.
CANDIDATE_POLICY_QUERIED = False
# What the policies actually consume. Rendering above this is the dominant
# cost in an episode and is discarded downstream.
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 180
ARENA_ENVIRONMENT_NAME = "Blueprint-ADP009D-Articulated-Scene-v0"


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


def _persist(path: Path, value: dict[str, Any]) -> None:
    value["result_digest"] = _canonical_digest(value, field="result_digest")
    value["_canonical_digest"] = value["result_digest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _phase(result: dict[str, Any], name: str) -> None:
    """Record a phase and announce it, in that order.

    The transport watches stdout for progress and kills a run that goes quiet.
    Writing phases only into the result file made a slow Arena boot
    indistinguishable from a hang: the heartbeat fired, the container log came
    back empty, and the launch bought an ambiguity instead of an answer.
    Flushed, because progress buffered past the timeout is progress nobody saw.
    """

    result["phase_reached"] = name
    print(f"BLUEPRINT_WAM_RUNTIME_PHASE:adp009d_scene:{name}:reached", flush=True)


def _finalize(*, output: Path, result: dict[str, Any], simulation_app: Any) -> None:
    """Write first, close second - close can end the process."""

    _persist(output, result)
    try:
        simulation_app.close()
    except Exception:  # noqa: BLE001
        pass


def _resolve_paths(argv: Sequence[str] | None) -> dict[str, Any]:
    """Work out the spec and output whether or not anyone passed arguments.

    The Arena entrypoint calls its runner bare and configures it through the
    environment; the rigid worker was written for that. Requiring --spec and
    --output meant the entrypoint reached this worker, argparse rejected the
    empty command line, and the run died having already downloaded the bundle
    and booted Isaac. Explicit arguments still win, because local debugging
    passes them.
    """

    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default=None)
    parser.add_argument("--output", default=None)
    # The provisioner's contract: it installs Arena, then calls the runtime
    # module with these. Unknown app flags (--headless, --device, ...) belong
    # to Isaac Lab's launcher and must not make argparse reject the command.
    parser.add_argument("--runtime-dir", dest="runtime_dir", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default=None)
    arguments, _unknown = parser.parse_known_args(
        list(argv) if argv is not None else None
    )

    if arguments.spec:
        spec = Path(arguments.spec).expanduser().resolve()
    else:
        # Beside the runner, where the bundle stages a worker's payload.
        candidates = [
            *(
                [Path(arguments.runtime_dir) / "native"
                 / "adp009d_articulated_scene_spec.json"]
                if arguments.runtime_dir
                else []
            ),
            Path("native") / "adp009d_articulated_scene_spec.json",
            Path(__file__).resolve().parent / "native"
            / "adp009d_articulated_scene_spec.json",
        ]
        spec = next((c.resolve() for c in candidates if c.is_file()), None)

    if arguments.output:
        output = Path(arguments.output).expanduser().resolve()
    else:
        base = (
            arguments.output_dir
            or os.environ.get("BLUEPRINT_ADP009D_OUTPUT_DIR")
            or "."
        )
        output = (Path(base).expanduser() / RESULT_FILENAME).resolve()
    return {"spec": spec, "output": output}


def main(argv: Sequence[str] | None = None) -> int:
    resolved = _resolve_paths(argv)
    output = resolved["output"]
    if resolved["spec"] is None:
        _persist(
            output,
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "status": "blocked",
                "blockers": ["articulated_scene_spec_not_found_beside_runner"],
                "native_isaac_executed": False,
                "candidate_policy_queried": CANDIDATE_POLICY_QUERIED,
                "provider_zero_required_after_return": True,
                "probe_results": [],
            },
        )
        return 1
    spec_path = resolved["spec"]
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [],
        "native_isaac_executed": False,
        "scene_composed": False,
        "controls_qualified": False,
        "physical_success_established": False,
        "provider_zero_required_after_return": True,
        "candidate_policy_queried": CANDIDATE_POLICY_QUERIED,
        "official_sources": {
            "isaac_lab_arena_revision": ARENA_REVISION,
            "isaac_lab_revision": ISAAC_LAB_REVISION,
        },
        "phase_reached": "start",
        "probe_results": [],
    }

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        result["spec_sha256"] = _sha256(spec_path)
    except (OSError, ValueError):
        result["blockers"].append("articulated_scene_spec_unreadable")
        _persist(output, result)
        return 1

    try:
        # Isaac Lab's own launcher, as the proven rigid runtime uses. It wires
        # the extension paths that make isaaclab and isaaclab_arena importable;
        # raw SimulationApp starts Kit and leaves those packages invisible,
        # which is what "No module named 'isaaclab'" meant on a host where the
        # worker had otherwise started cleanly.
        from isaaclab.app import AppLauncher  # type: ignore

        launcher_parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(launcher_parser)
        # Parsed from an empty list, so the worker's own command line never
        # reaches these - both flags must be set explicitly. Configuring a
        # camera is not the same as enabling rendering: rt15 configured all
        # three, built the embodiment, and Isaac then refused with "A camera
        # was spawned without the --enable_cameras flag".
        launcher_args = launcher_parser.parse_args([])
        launcher_args.headless = True
        launcher_args.enable_cameras = True
        app_launcher = AppLauncher(launcher_args)
        simulation_app = app_launcher.app
    except Exception as exc:  # noqa: BLE001
        result["blockers"].append(
            f"articulated_scene_app_launcher_failed:{type(exc).__name__}:{exc}"
        )
        _persist(output, result)
        return 1

    try:
        from isaaclab_arena.assets.object import Object  # type: ignore
        from isaaclab_arena.assets.object_base import ObjectType  # type: ignore
        from isaaclab_arena.embodiments.droid.droid import (  # type: ignore
            DroidAbsoluteJointPositionEmbodiment,
        )
        from isaaclab_arena.environments.isaaclab_arena_environment import (  # type: ignore
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.environments.arena_env_builder import (  # type: ignore
            ArenaEnvBuilder,
        )
        from isaaclab_arena.scene.scene import Scene  # type: ignore
        from isaaclab_arena.tasks.no_task import NoTask  # type: ignore
        from isaaclab_arena.utils.pose import Pose  # type: ignore
        from isaaclab_arena.assets.asset_registry import AssetRegistry  # type: ignore

        asset_registry = AssetRegistry()

        _phase(result, "arena_imported")

        root = spec_path.parent
        composition = spec.get("composition") or {}
        binding = (composition.get("task_sample_binding") or {})
        joint_ids = [str(v) for v in (binding.get("joint_ids") or [])]
        if not joint_ids:
            raise RuntimeError("articulated_scene_joint_binding_empty")

        # ObjectType is resolved by name so a renamed or missing member reports
        # what the runtime actually offers instead of raising AttributeError.
        available = {member.name for member in ObjectType}
        assets = []
        task_object = None
        for row in composition.get("objects") or []:
            kind = str(row.get("object_type") or "")
            if kind not in available:
                raise RuntimeError(
                    f"articulated_scene_object_type_unknown:{kind}:{sorted(available)}"
                )
            # Asset bindings rename files into the bundle, so the authoring
            # name in the spec is often not the name on the provider. Resolve
            # by declared name, then by the bundle's binding names, and on a
            # miss report the whole layout - a bare missing filename costs one
            # launch to learn the name and another to use it.
            try:
                resolution = resolve_runtime_asset(
                    runtime_dir=root,
                    declared_filename=str(row.get("usd_filename") or ""),
                    aliases=[str(value) for value in (row.get("usd_filename_aliases") or ())],
                    role=str(row.get("name") or "asset"),
                )
            except RuntimeAssetResolutionError as exc:
                raise RuntimeError(
                    f"articulated_scene_asset_missing:{';'.join(exc.errors)}"
                ) from exc
            asset_path = Path(resolution["resolved_path"])
            result.setdefault("asset_resolution", []).append(
                {
                    "role": resolution["role"],
                    "declared_filename": resolution["declared_filename"],
                    "matched_on": resolution["matched_on"],
                    "resolved_path": resolution["resolved_path"],
                }
            )
            obj = Object(
                name=str(row.get("name")),
                object_type=getattr(ObjectType, kind),
                usd_path=str(asset_path),
                initial_pose=Pose(
                    position_xyz=tuple(
                        float(v) for v in (row.get("initial_position_world_m") or (0, 0, 0))
                    )
                ),
                spawn_cfg_addon={
                    "visible": bool(row.get("visible", True)),
                    # Arena's object cfgs do not set this; the DROID robot's
                    # own spawn does. Without it a contact sensor on this prim
                    # is attached to geometry that never reports, so every
                    # contact reads false and the task looks untouched.
                    "activate_contact_sensors": True,
                },
            )
            assets.append(obj)
            if row.get("semantic_role") == "task_object":
                task_object = obj
                result["task_object_spawn_type"] = kind
        if task_object is None:
            raise RuntimeError("articulated_scene_no_task_object_in_composition")
        # A refrigerator spawned rigid has frozen joints and raises nothing.
        if result.get("task_object_spawn_type") != "ARTICULATION":
            raise RuntimeError(
                "articulated_scene_task_object_not_articulated:"
                f"{result.get('task_object_spawn_type')}"
            )
        _phase(result, "assets_resolved")

        base = spec.get("robot_base") or {}
        embodiment = DroidAbsoluteJointPositionEmbodiment(
            enable_cameras=True,
            initial_pose=Pose(
                position_xyz=tuple(
                    float(v) for v in (base.get("position_xyz") or (0.0, 0.0, 0.0))
                ),
                rotation_xyzw=tuple(
                    float(v) for v in (base.get("rotation_xyzw") or (0, 0, 0, 1))
                ),
            ),
            initial_joint_pose=[float(v) for v in (base.get("reset_joints") or [])] or None,
        )
        # The adapter binds three cameras by name and refuses without them.
        # Arena supplies the configs on the embodiment; what it does not supply
        # is the resolution and data types this lane needs, and a camera left
        # at Arena's defaults does not carry the pose metadata the policy-input
        # contract requires. Mirrors the rigid lane rather than inventing a
        # second convention.
        for camera_name in ("external_camera", "wrist_camera", "external_camera_2"):
            camera_cfg = getattr(
                getattr(embodiment, "camera_config", None), camera_name, None
            )
            if camera_cfg is None:
                raise RuntimeError(
                    f"articulated_scene_required_camera_config_missing:{camera_name}"
                )
            camera_cfg.data_types = ["rgb", "semantic_segmentation"]
            if camera_name != "external_camera_2":
                camera_cfg.data_types.insert(1, "distance_to_camera")
            camera_cfg.colorize_semantic_segmentation = False
            camera_cfg.update_period = 0.0
            # Arena leaves this false, which freezes camera.data.pos_w at
            # initialisation even while the parented view moves.
            camera_cfg.update_latest_camera_pose = True
            camera_cfg.width = CAMERA_WIDTH
            camera_cfg.height = CAMERA_HEIGHT
        result["camera_resolution"] = [CAMERA_WIDTH, CAMERA_HEIGHT]
        _phase(result, "cameras_configured")

        _phase(result, "embodiment_configured")

        # Arena ships a DomeLight asset (name "light", /World/Light, its own
        # spawner cfg). The first version of this hand-rolled an Object
        # subclass with type() to inject spawner_cfg, which is a guess at what
        # a spawner needs; the registry is what Arena's own scenes use.
        assets.append(asset_registry.get_asset_by_name("light")())

        def configure(cfg):
            from isaaclab_physx.physics import PhysxCfg  # type: ignore
            from isaaclab.sensors import ContactSensorCfg  # type: ignore

            # The scorer needs contact, and contact needs sensors declared
            # before construction - there is no way to attach one to a built
            # scene. Three separate sensors rather than one, because "the
            # gripper is holding the handle" and "the elbow hit the cabinet"
            # are opposite verdicts read off the same forces.
            # Capital R. The scene *key* is "robot" but the prim is
            # "{ENV_REGEX_NS}/Robot" - DroidSceneCfg declares both and they do
            # not match. A ContactSensor whose pattern matches nothing does not
            # say so: Isaac Lab indexes _parent_prims[0] and raises a bare
            # IndexError from inside the physics callback, which is what rt16
            # and rt17 returned.
            cfg.scene.robot_contact_sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                history_length=1,
                track_air_time=False,
            )
            cfg.scene.task_object_contact_sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/task_object/.*",
                history_length=1,
                track_air_time=False,
            )
            cfg.scene.scene_collision_contact_sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/scene_collision/.*",
                history_length=1,
                track_air_time=False,
            )

            cfg.sim.dt = 1.0 / 120.0
            cfg.seed = int(spec.get("seed") or 20260810)
            cfg.sim.render_interval = 8
            cfg.decimation = 8
            cfg.episode_length_s = float(spec.get("episode_length_s") or 8.0)
            cfg.sim.physics = PhysxCfg(
                solver_type=1,
                enable_enhanced_determinism=True,
                gpu_max_rigid_contact_count=2**23,
                gpu_max_rigid_patch_count=2**15,
            )
            return cfg

        # Keywords mirror the rigid lane exactly. The first version of this
        # used embodiments=[...] and env_config_modifier=..., which are not
        # Arena's API - they were what I assumed, and the stub accepted them
        # because I had written the stub from the same assumption. A fake built
        # from a guess validates the guess. A parity test now pins these
        # against the rigid runtime's own call site.
        arena_env = IsaacLabArenaEnvironment(
            name=ARENA_ENVIRONMENT_NAME,
            scene=Scene(assets=assets),
            embodiment=embodiment,
            task=NoTask(),
            env_cfg_callback=configure,
        )
        builder_args = argparse.Namespace(
            num_envs=1,
            env_spacing=2.0,
            solve_relations=False,
            placement_seed=int(spec.get("seed") or 20260810),
            mimic=False,
            device="cuda:0",
            disable_fabric=False,
            presets=None,
        )
        builder = ArenaEnvBuilder(arena_env, builder_args)
        env, _env_cfg = builder.make_registered_and_return_cfg(render_mode="rgb_array")
        result["scene_composed"] = True
        _phase(result, "environment_built")

        scene = env.unwrapped.scene
        try:
            live = scene[task_object.name]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"articulated_scene_task_object_not_in_scene:{task_object.name}"
            ) from exc
        dof_names = list(getattr(live.data, "joint_names", []) or [])
        result["observed_dof_names"] = dof_names
        missing = [name for name in joint_ids if name not in dof_names]
        if missing:
            # Naming both sides is what turns this into a one-line fix rather
            # than another launch spent guessing at the runtime's naming.
            raise RuntimeError(
                f"articulated_scene_joints_absent:{missing}:observed={dof_names}"
            )
        _phase(result, "articulation_bound")

        import torch  # type: ignore

        indices = {name: dof_names.index(name) for name in joint_ids}

        def _read_joint_state(joint_id: str):
            positions = live.data.joint_pos[0]
            velocities = live.data.joint_vel[0]
            index = indices[joint_id]
            return (float(positions[index]), float(velocities[index]))

        sys_path = str(root)
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        from articulated_task_sample import (  # type: ignore
            build_articulated_task_sample,
        )
        from adp009d_isaac_episode_adapter import IsaacEpisodeAdapter  # type: ignore
        from adp009d_control_episode import run_task_neutral_controls  # type: ignore

        # The step index must be the adapter's, not a second counter kept
        # beside it: the control episode compares the sample's step to its own
        # and a private tally never advances, so every step after the first
        # fails task_control_sample_step_mismatch. Late-bound because the
        # callback is built before the adapter that owns the count.
        adapter_holder: dict[str, Any] = {}

        def _sensor_forces(sensor_name: str):
            """Net contact forces, or None so the predicate refuses loudly."""

            try:
                return scene[sensor_name].data.net_forces_w
            except (KeyError, AttributeError, TypeError):
                return None

        def _body_position(articulation, body_name: str):
            names = list(getattr(articulation.data, "body_names", []) or [])
            if body_name not in names:
                return None
            return [
                float(value)
                for value in articulation.data.body_pose_w[0, names.index(body_name), :3]
            ]

        task_spec_row = spec.get("task_spec") or {}
        # Top level, NOT inside task_spec: the control plan is digest-bound to
        # the task spec, so adding a scene-composition field there invalidates
        # a sealed plan. The contract caught this immediately, which is what
        # the binding is for.
        support_link_body = str(spec.get("support_link_body") or "")
        if not support_link_body:
            raise RuntimeError("articulated_scene_support_link_body_missing")
        authored_base = [
            float(value)
            for value in (task_object.initial_pose.position_xyz or (0.0, 0.0, 0.0))
        ]
        handle_position = [
            float(value)
            for value in (spec.get("handle_position_world_m") or authored_base)
        ]
        # Built after the gripper probe, which is where body_names and the
        # finger indices come from; the readers below close over `robot` and
        # `scene`, so only the index arguments need to wait.
        observation_holder: dict[str, Any] = {}

        def _task_sample():
            bound = adapter_holder.get("adapter")
            readers = observation_holder.get("readers")
            if readers is None:
                raise RuntimeError("articulated_scene_observation_readers_unbound")
            return build_articulated_task_sample(
                joint_ids=joint_ids,
                read_joint_state=_read_joint_state,
                joint_hard_limits_rad=task_spec_row.get("joint_hard_limits_rad") or {},
                joint_limit_tolerance_rad=float(
                    task_spec_row.get("reset_tolerance_rad") or 0.005
                ),
                step_index=int(bound.control_step_index) if bound is not None else 0,
                **readers,
            )

        robot = scene[embodiment.name] if hasattr(embodiment, "name") else scene["robot"]

        def _to_torch(value):
            return value.detach() if hasattr(value, "detach") else torch.as_tensor(value)

        # Which command closes the fingers is a property of Arena's action
        # convention, not of DROID's, and an inverted one turns every grasp
        # into a release - a policy that reaches the handle and drops it.
        # Measure it, and take the two widths the adapter needs from the same
        # measurement instead of hardcoding them.
        action_dim = int(env.unwrapped.action_manager.total_action_dim)
        body_names = list(getattr(robot.data, "body_names", []) or [])
        finger_indices = [
            body_names.index(name)
            for name in ("left_inner_finger", "right_inner_finger")
            if name in body_names
        ]

        def _apply_gripper_command(command: float) -> None:
            env.reset(seed=int(spec.get("seed") or 20260810))
            probe_action = torch.zeros((1, action_dim))
            probe_action[:, :7] = _to_torch(robot.data.joint_pos)[:, :7]
            probe_action[:, 7] = float(command)
            for _ in range(30):
                env.step(probe_action)

        def _read_finger_separation() -> float:
            poses = _to_torch(robot.data.body_pose_w)[0, finger_indices, :3]
            return float(torch.linalg.vector_norm(poses[0] - poses[1]))

        gripper = measure_gripper_convention(
            apply_command=_apply_gripper_command,
            read_finger_separation_m=_read_finger_separation,
            body_names=body_names,
        )
        result["gripper_convention_probe"] = gripper
        _phase(result, "gripper_convention_measured")

        observation_holder["readers"] = build_scene_observations(
            read_task_contact_forces=lambda: _sensor_forces(
                "task_object_contact_sensor"
            ),
            read_robot_contact_forces=lambda: _sensor_forces("robot_contact_sensor"),
            read_scene_contact_forces=lambda: _sensor_forces(
                "scene_collision_contact_sensor"
            ),
            # Named by the spec, never defaulted to a literal. A guessed body
            # name that happens not to exist refuses (which is fine), but one
            # that happens to exist and is the wrong link reads a containment
            # verdict off whatever that link did.
            read_task_object_base_position_m=lambda: _body_position(
                live, support_link_body
            ),
            authored_task_object_base_position_m=authored_base,
            read_end_effector_position_m=lambda: _body_position(robot, "panda_hand"),
            read_handle_position_m=lambda: handle_position,
            finger_body_indices=finger_indices,
            non_finger_body_indices=[
                index
                for index in range(len(body_names))
                if index not in set(finger_indices)
            ],
        )

        _phase(result, "observations_bound")

        adapter = IsaacEpisodeAdapter(
            env=env,
            robot=robot,
            task_sample_callback=_task_sample,
            action_dim=action_dim,
            reset_seed=int(spec.get("seed") or 20260810),
            to_torch=_to_torch,
            gripper_closed_width_m=float(gripper["gripper_closed_width_m"]),
            gripper_open_width_m=float(gripper["gripper_open_width_m"]),
            simulation_step_seconds=1.0 / 120.0 * 8,
        )
        adapter_holder["adapter"] = adapter
        _phase(result, "adapter_wired")

        pair = run_task_neutral_controls(
            environment=adapter,
            task_spec=spec["task_spec"],
            control_plan=spec["control_plan"],
            gripper_open_command=float(spec.get("gripper_open_command") or 1.0),
            output_dir=output.parent / "controls",
        )
        result["control_pair"] = pair
        result["controls_qualified"] = bool(
            pair.get("cell_admitted_for_policy_execution")
        )
        result["probe_results"] = [
            {"name": row["control_id"], "passed": bool(row["control_passed"])}
            for row in (pair.get("controls") or [])
        ]
        result["native_isaac_executed"] = True
        result["status"] = (
            "completed" if result["controls_qualified"] else "completed_with_failures"
        )
        _phase(result, "controls_complete")
        result["claim_boundary"] = {
            "robot_present_and_grasping": True,
            "policy_queried": False,
            "success_is_scored_by_the_task_neutral_scorer": True,
        }
    except BaseException as exc:  # noqa: BLE001 - one launch, one retained result
        import traceback

        # The type and message alone are not always enough to locate a fault:
        # rt16 returned "IndexError: list index out of range" from inside
        # Arena, which names neither the file nor the frame. Retained in the
        # result and echoed to stdout so it also lands in the container log.
        formatted = traceback.format_exc()
        result["failure_traceback"] = formatted.splitlines()[-40:]
        print(formatted, flush=True)
        result["blockers"].append(
            f"articulated_scene_failed_at_{result['phase_reached']}:"
            f"{type(exc).__name__}:{exc}"
        )

    _finalize(output=output, result=result, simulation_app=simulation_app)
    # The collector looks for the lane's own filename; writing only to --output
    # leaves a completed run looking like one that produced nothing.
    if output.name != RESULT_FILENAME:
        _persist(output.with_name(RESULT_FILENAME), dict(result))
    return 0 if result.get("native_isaac_executed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
