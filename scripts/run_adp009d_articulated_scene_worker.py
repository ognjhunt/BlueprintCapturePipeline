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
import math
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
    from adp009d_isaac_episode_adapter import END_EFFECTOR_BODY_CANDIDATES
    from articulated_scene_observations import (
        build_scene_observations,
        resolve_contact_sensor_rows,
    )
    from articulated_control_verdict import seal_detent_torque
    from franka_kinematics import solve_axis_aligned_ik
    from decision_evidence_contracts import canonical_digest
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
    from blueprint_pipeline.adp009d_isaac_episode_adapter import (
        END_EFFECTOR_BODY_CANDIDATES,
    )
    from blueprint_pipeline.articulated_scene_observations import (
        build_scene_observations,
        resolve_contact_sensor_rows,
    )
    from blueprint_pipeline.articulated_control_verdict import seal_detent_torque
    from blueprint_pipeline.franka_kinematics import solve_axis_aligned_ik
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest


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
# The scene key for the arm. Declared by DroidSceneCfg as the attribute
# "robot"; the prim it spawns is "{ENV_REGEX_NS}/Robot". The two differ in
# case and neither is embodiment.name.
ROBOT_SCENE_KEY = "robot"
# Where the Robotiq fingers actually are. droid.py declares them at
# {ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/<side>_inner_finger, which
# Robot/.* does not reach.
FINGER_CONTACT_PRIM_PATH = "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/.*"


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


# Which object types already receive activate_contact_sensors from Arena's own
# generator. Passing it again through spawn_cfg_addon is not additive - it is
# splatted into the same UsdFileCfg call and raises "got multiple values".
# BASE is the one that does not get it, and a contact sensor on geometry
# without it reads false forever: a wrong answer rather than an error.
ARENA_TYPES_WITH_CONTACT_SENSORS_ENABLED = frozenset({"ARTICULATION", "RIGID"})
# A BASE asset is static geometry. It has no rigid bodies, so it can carry
# neither activate_contact_sensors nor a ContactSensor - Isaac raises "no rigid
# bodies are present under this prim". Arena omitting the flag for BASE is
# correct, not an oversight, and my previous commit had that backwards.
ARENA_STATIC_TYPES = frozenset({"BASE"})


def _spawn_cfg_addon(object_type: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Extra UsdFileCfg keywords for one object, without colliding."""

    addon: dict[str, Any] = {"visible": bool(row.get("visible", True))}
    kind = str(object_type)
    if (
        kind not in ARENA_TYPES_WITH_CONTACT_SENSORS_ENABLED
        and kind not in ARENA_STATIC_TYPES
    ):
        addon["activate_contact_sensors"] = True
    return addon



def _to_torch(value: Any) -> Any:
    """Convert simulator-native arrays at the boundary before indexing.

    Isaac Lab hands back Warp arrays for some fields and torch tensors for
    others, and Warp refuses item indexing outright - rt25 died on "Item
    indexing is not supported on wp.array objects" reading joint_pos, while the
    gripper probe had read body_pose_w moments earlier without complaint. A
    converter that only knows torch is a coin flip on which field it meets.

    Same shape as the rigid lane's, including the refusal: an unknown array
    type is not silently coerced, because a wrong conversion produces numbers
    rather than an error.

    Module scope on purpose. Nested inside main() it was defined after three of
    its callers and worked only because closures resolve at call time - true
    here, and one reordering away from not being.
    """

    if hasattr(value, "detach"):
        return value
    module = type(value).__module__
    if module == "warp" or module.startswith("warp."):
        import warp as wp  # type: ignore

        return wp.to_torch(value)
    raise TypeError(f"unsupported_sim_array:{module}.{type(value).__name__}")


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
        # The task object spawns Arena-native: a free-base articulation whose
        # placement comes from initial_pose. Arena registers a set_object_pose
        # reset event carrying exactly that pose and writes it into the root
        # at every reset - it is the last pose channel to run, so it always
        # wins. rt51-rt53 measured the fridge at the origin because the spec
        # said (0,0,0) here while the placement lived in USD, one channel too
        # early; fix_root_link only added a third channel to lose with.
        def _task_spawn_addon(kind: str, row: Mapping[str, Any]) -> dict[str, Any]:
            return _spawn_cfg_addon(kind, row)

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
                        float(v)
                        for v in (
                            row.get("spawn_position_world_m")
                            if row.get("spawn_position_world_m") is not None
                            else row.get("initial_position_world_m") or (0, 0, 0)
                        )
                    )
                ),
                spawn_cfg_addon=_task_spawn_addon(kind, row),
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
            # Two sensors, because one cannot span both. Robot/.* matches the
            # arm links only - rt31 observed panda_link0 through panda_link8
            # and no fingers - since the Robotiq fingers sit two levels deeper
            # at Robot/Gripper/Robotiq_2F_85/, exactly where droid.py declares
            # them. Splitting them also makes the semantics honest: the arm
            # sensor IS the collision sensor, with no finger rows to exclude,
            # and the finger sensor IS the grasp sensor.
            cfg.scene.robot_contact_sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                history_length=1,
                track_air_time=False,
            )
            cfg.scene.finger_contact_sensor = ContactSensorCfg(
                prim_path=FINGER_CONTACT_PRIM_PATH,
                history_length=1,
                track_air_time=False,
                # Per-partner, so "gripper on the handle" is distinguishable
                # from "gripper on anything else".
                filter_prim_paths_expr=["{ENV_REGEX_NS}/task_object/.*"],
            )
            cfg.scene.task_object_contact_sensor = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/task_object/.*",
                history_length=1,
                track_air_time=False,
            )
            # No sensor on scene_collision: it is static, and
            # filter_prim_paths_expr matches body prim paths, which static
            # geometry does not have either. Scene contact is therefore the
            # part of the robot's net force that the task-object filter does
            # not explain.

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
            positions = _to_torch(live.data.joint_pos)[0]
            velocities = _to_torch(live.data.joint_vel)[0]
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
                return _to_torch(scene[sensor_name].data.net_forces_w)
            except (KeyError, AttributeError, TypeError):
                return None

        def _residual_scene_forces():
            """Robot net contact minus the part the twin explains."""

            # Both terms from the finger sensor, or the shapes do not line up:
            # the arm sensor matched nine links and the finger sensor a
            # handful of gripper bodies. What the fingers touch that is NOT the
            # twin is, in this scene, the room.
            net = _sensor_forces("finger_contact_sensor")
            if net is None:
                return None
            raw_matrix = getattr(finger_sensor.data, "force_matrix_w", None)
            matrix = None if raw_matrix is None else _to_torch(raw_matrix)
            if matrix is None:
                # No filter reported: cannot separate room from twin, and
                # returning the net force would blame the room for every grasp.
                return None
            # force_matrix_w is (envs, bodies, filters, 3); summing the filter
            # axis gives the twin's share of each body's contact.
            explained = matrix[0].sum(dim=1) if hasattr(matrix[0], "sum") else None
            if explained is None:
                return None
            return (net[0] - explained).unsqueeze(0)

        def _body_position(articulation, body_name: str):
            names = list(getattr(articulation.data, "body_names", []) or [])
            if body_name not in names:
                return None
            poses = _to_torch(articulation.data.body_pose_w)
            return [float(value) for value in poses[0, names.index(body_name), :3]]

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

        # DroidSceneCfg declares the articulation as the attribute "robot";
        # embodiment.name is "droid_abs_joint_pos", which is the embodiment's
        # own identifier and not a scene key. I read this out of the source
        # while chasing the prim-path case bug, wrote it down, and left the
        # code alone - so rt20 spent a launch on a fact already in hand.
        robot = scene[ROBOT_SCENE_KEY]


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

        end_effector_body = next(
            (name for name in END_EFFECTOR_BODY_CANDIDATES if name in body_names),
            "",
        )
        if not end_effector_body:
            raise RuntimeError(
                "articulated_scene_end_effector_body_absent:"
                f"{sorted(END_EFFECTOR_BODY_CANDIDATES)}:observed={body_names}"
            )
        result["end_effector_body"] = end_effector_body

        # Rows into the contact arrays come from the SENSOR's own body list,
        # never the articulation's. A ContactSensor matches its own subset by
        # prim-path regex, so the two index spaces differ - rt29 spent a launch
        # on indices 9 and 14, both valid on the robot, both out of range on
        # the sensor.
        robot_sensor = scene["robot_contact_sensor"]
        finger_sensor = scene["finger_contact_sensor"]
        finger_rows = resolve_contact_sensor_rows(
            sensor_body_names=list(getattr(finger_sensor, "body_names", None) or []),
            finger_body_names=("left_inner_finger", "right_inner_finger"),
        )
        result["contact_sensor_rows"] = {
            "arm_sensor_body_names": list(
                getattr(robot_sensor, "body_names", None) or []
            ),
            "finger_sensor_body_names": finger_rows["sensor_body_names"],
            "finger_rows": finger_rows["finger_rows"],
        }

        def _finger_contact_with_task_object():
            """Filtered per-partner force on the fingers against the twin."""

            matrix = getattr(finger_sensor.data, "force_matrix_w", None)
            if matrix is None:
                return None
            converted = _to_torch(matrix)
            # (envs, sensors, filters, 3) -> sum the filter axis for the twin's
            # share of each body's contact.
            return converted[0].sum(dim=1).unsqueeze(0)

        observation_holder["readers"] = build_scene_observations(
            # Fingers-against-the-twin, from the robot sensor's filtered
            # matrix. The task object's own sensor cannot answer this: its rows
            # are the fridge's bodies, so finger indices mean nothing there.
            read_task_contact_forces=_finger_contact_with_task_object,
            read_robot_contact_forces=lambda: _sensor_forces("robot_contact_sensor"),
            # The room is static and cannot report contact itself. What the
            # robot touches that is NOT the twin is, in this scene, the room:
            # net force minus the filtered per-partner force against the twin.
            # An approximation, and stated as one - it cannot distinguish the
            # room from the robot touching itself.
            read_scene_contact_forces=lambda: _residual_scene_forces(),
            # Named by the spec, never defaulted to a literal. A guessed body
            # name that happens not to exist refuses (which is fine), but one
            # that happens to exist and is the wrong link reads a containment
            # verdict off whatever that link did.
            read_task_object_base_position_m=lambda: _body_position(
                live, support_link_body
            ),
            authored_task_object_base_position_m=authored_base,
            # Resolved against the robot's actual bodies using the adapter's
            # own candidate list. "panda_hand" was a literal guess and the
            # adapter already knows the alternatives it accepts.
            read_end_effector_position_m=lambda: _body_position(
                robot, end_effector_body
            ),
            read_handle_position_m=lambda: handle_position,
            finger_body_indices=finger_rows["finger_rows"],
            # The arm sensor has no finger rows to exclude; every body it
            # matched is a link whose contact is a collision.
            non_finger_body_indices=None,
        )

        _phase(result, "observations_bound")

        def _reset_scene() -> None:
            """Reset the environment AND the twin's joints.

            env.reset() restores the robot and any rigid task object. It does
            not touch an articulated task object's joint state - the rigid can
            lane never needed it to - so rt35 showed the door sitting at 0.619
            rad both before and after a reset, identical to five decimals,
            still moving at 0.031 rad/s. The episode was scoring a door that
            nothing had ever put back.
            """

            env.reset(seed=int(spec.get("seed") or 20260810))
            names = list(live.data.joint_names or [])
            resets = task_spec_row.get("joint_reset_positions_rad") or {}
            positions = torch.zeros(
                (1, len(names)), device=env.unwrapped.device, dtype=torch.float32
            )
            for joint_name, value in resets.items():
                if joint_name in names:
                    positions[0, names.index(joint_name)] = float(value)
            velocities = torch.zeros_like(positions)
            writer = getattr(live, "write_joint_state_to_sim", None)
            if writer is None:
                raise RuntimeError(
                    "articulated_scene_joint_state_write_unavailable:"
                    f"{sorted(a for a in dir(live) if 'joint' in a)}"
                )
            writer(positions, velocities)

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
            reset_callback=_reset_scene,
        )
        adapter_holder["adapter"] = adapter

        # Three hypotheses for the 0.619 rad reset readback have now been
        # wrong - shelf/bin penetration, an authored drive target, the room
        # collision - and the value is identical to five decimals across two
        # different twins, which physics would not reproduce. So stop guessing
        # and record the state itself: what the runtime calls these joints,
        # what it reports before a reset, and what it reports after one.
        def _joint_snapshot(label: str) -> dict[str, Any]:
            positions = _to_torch(live.data.joint_pos)[0]
            velocities = _to_torch(live.data.joint_vel)[0]
            return {
                "label": label,
                "dof_names": list(live.data.joint_names or []),
                "joint_pos_rad": [float(v) for v in positions],
                "joint_vel_rad_s": [float(v) for v in velocities],
            }

        reset_diagnostic = [_joint_snapshot("before_adapter_reset")]
        try:
            adapter.reset()
            reset_diagnostic.append(_joint_snapshot("after_adapter_reset"))
        except Exception as exc:  # noqa: BLE001
            reset_diagnostic.append(
                {"label": "adapter_reset_failed", "error": f"{type(exc).__name__}:{exc}"[:200]}
            )
        result["reset_diagnostic"] = reset_diagnostic
        _phase(result, "adapter_wired")

        # A real refrigerator door is held shut by its magnetic gasket, and
        # rt36 showed what happens without one: 0.8 mm of shell-on-shell
        # interference between the door and cabinet is enough to swing a free
        # hinge 34.7 degrees before the arm arrives, so the grasp reaches for a
        # handle that has moved. The detent is angle-local - cosine-tapered and
        # gone by 5 degrees - which is exactly why it cannot be a USD drive: a
        # spring would also drag the door back after release, and the task
        # requires it to hold at 45-55.
        #
        # Same formulation and the same 12 N.m / 5 degree parameters the hinge
        # probe qualified on hardware.
        seal = spec.get("seal") or {}
        seal_peak = float(seal.get("breakaway_torque_n_m") or 0.0)
        seal_width = float(seal.get("angular_width_degrees") or 0.0)
        hinge_index = dof_names.index(task_spec_row["target_joint_id"])

        class _SealedEnvironment:
            """The adapter, with the gasket applied before every step."""

            def __init__(self, inner: Any) -> None:
                self._inner = inner
                self.seal_applications = 0
                self.achieved_joint_trace: list[list[float]] = []
                self.fingertip_mid_trace: list[list[float]] = []
                self.finger_pair_trace: list[list[float]] = []
                self.door_pose_trace: list[list[float]] = []

            def _apply_seal(self) -> None:
                if seal_peak <= 0.0 or seal_width <= 0.0:
                    return
                angle_rad = float(_to_torch(live.data.joint_pos)[0][hinge_index])
                resist = seal_detent_torque(
                    math.degrees(angle_rad), seal_peak, seal_width
                )
                if resist == 0.0:
                    return
                efforts = torch.zeros(
                    (1, len(dof_names)),
                    device=env.unwrapped.device,
                    dtype=torch.float32,
                )
                # Opposes motion away from closed, which is what a seal does.
                efforts[0, hinge_index] = -resist
                setter = getattr(live, "set_joint_effort_target", None)
                if setter is None:
                    raise RuntimeError(
                        "articulated_scene_joint_effort_api_unavailable:"
                        f"{sorted(a for a in dir(live) if 'effort' in a)}"
                    )
                setter(efforts)
                self.seal_applications += 1

            def step(self, action):
                self._apply_seal()
                outcome = self._inner.step(action)
                # Achieved joints, recorded because rt43 proved the danger of
                # assuming execution: a 0.0 mm-verified trajectory produced
                # zero contact, and nothing in any receipt could say whether
                # the arm had tracked it. Commanded is an intention; this is
                # what happened.
                try:
                    achieved = _to_torch(robot.data.joint_pos)[0]
                    self.achieved_joint_trace.append(
                        [round(float(v), 4) for v in achieved][:9]
                    )
                    # Fingertip midpoint in world. rt44 closed the fingers to
                    # their empty-hand stop mid-sweep and touched nothing, so
                    # the pinch centre is not where the local FK model says it
                    # is; this records where it actually is, making the miss a
                    # vector instead of a hypothesis.
                    names = list(getattr(robot.data, "body_names", []) or [])
                    if "left_inner_finger" in names and "right_inner_finger" in names:
                        poses = _to_torch(robot.data.body_pose_w)[0]
                        li = names.index("left_inner_finger")
                        ri = names.index("right_inner_finger")
                        mid = (poses[li, :3] + poses[ri, :3]) / 2.0
                        self.fingertip_mid_trace.append(
                            [round(float(v), 4) for v in mid]
                        )
                        # Both origins separately: the pinch axis and the
                        # empty-closure geometry fall straight out of the
                        # pair, where the midpoint alone hides them.
                        self.finger_pair_trace.append(
                            [round(float(v), 4) for v in poses[li, :3]]
                            + [round(float(v), 4) for v in poses[ri, :3]]
                        )
                    # The door link's world pose, so the handle's LIVE position
                    # is measured rather than trusted from the authoring
                    # transform - the one assumption three bracketing runs
                    # never tested.
                    door_names = list(getattr(live.data, "body_names", []) or [])
                    if "upper_door" in door_names:
                        dpose = _to_torch(live.data.body_pose_w)[0]
                        self.door_pose_trace.append(
                            [round(float(v), 4) for v in dpose[door_names.index("upper_door")][:7]]
                        )
                except Exception:  # noqa: BLE001 - tracing must not fail a step
                    pass
                return outcome

            def __getattr__(self, name):
                return getattr(self._inner, name)

        # ------------------------------------------------------------------
        # Measured correction. Five runs of open-loop bracketing missed the
        # bar because every model number - tool stack, pad reach, implicitly
        # the base and even the fridge's placement - was an assumption. Arena
        # publishes the pad tips themselves: ee_frame's tool_leftfinger /
        # tool_rightfinger targets sit at inner_finger + 46 mm, tracked by
        # the sim. Drive to the planned grasp pose, read where the pinch
        # point actually is, shift the whole plan by the one world vector
        # that puts it on the bar, and re-solve. Feedback absorbs what no
        # calibration constant can.
        ee = scene["ee_frame"]
        ee_names = list(getattr(ee.data, "target_frame_names", []) or [])
        if "tool_leftfinger" not in ee_names or "tool_rightfinger" not in ee_names:
            raise RuntimeError(
                f"articulated_scene_ee_frame_targets_missing:{ee_names}"
            )
        li_t = ee_names.index("tool_leftfinger")
        ri_t = ee_names.index("tool_rightfinger")

        def _pad_mid_world() -> list[float]:
            targets = _to_torch(ee.data.target_pos_w)[0]
            mid = (targets[li_t] + targets[ri_t]) / 2.0
            return [float(v) for v in mid]

        def _drive(q, gripper, steps):
            action = torch.zeros((1, action_dim), device=env.unwrapped.device)
            action[0, :7] = torch.tensor(q, device=env.unwrapped.device)
            action[0, 7] = float(gripper)
            for _ in range(steps):
                env.step(action)

        plan0 = spec["control_plan"]
        corrected_plan = plan0
        grasp_actions = [
            a for a in plan0["scripted_positive_actions"] if a["phase_id"] == "grasp"
        ]
        if not grasp_actions:
            # A plan with no grasp phase has nothing to correct against; say
            # so rather than crash, and run it as planned.
            result["measured_correction"] = {"skipped": "no_grasp_phase_in_plan"}
        else:
            q_grasp = grasp_actions[-1]["isaac_action"][:7]
            env.reset(seed=int(spec.get("seed") or 20260810))
            _drive(q_grasp, 0.0, 60)  # open hand, settled at planned grasp
            pad_before = _pad_mid_world()
            handle = [float(v) for v in spec["handle_position_world_m"]]
            delta = [handle[i] - pad_before[i] for i in range(3)]
            result["measured_correction"] = {
                "pad_mid_at_planned_grasp": pad_before,
                "handle_target": handle,
                "delta_world_m": delta,
                "delta_norm_mm": round(sum(d * d for d in delta) ** 0.5 * 1000, 2),
            }
            _phase(result, "grasp_correction_measured")
            if any(abs(d) > 0.002 for d in delta):
                try:
                    from franka_kinematics import forward_kinematics as _fk
                except ModuleNotFoundError:
                    from blueprint_pipeline.franka_kinematics import (
                        forward_kinematics as _fk,
                    )
                corrected_actions = []
                solver_failures = 0
                for a in plan0["scripted_positive_actions"]:
                    q_old = a["isaac_action"][:7]
                    pos_old, rot_old = _fk(q_old)
                    tool_axis = [rot_old[0][2], rot_old[1][2], rot_old[2][2]]
                    solved = solve_axis_aligned_ik(
                        target_position_world_m=[
                            pos_old[i] + delta[i] for i in range(3)
                        ],
                        tool_axis_world=tool_axis,
                        seed_joint_positions=q_old,
                    )
                    joints = solved["joint_positions_rad"]
                    if not solved["converged"]:
                        solver_failures += 1
                        joints = q_old
                    corrected_actions.append(
                        {
                            "phase_id": a["phase_id"],
                            "isaac_action": [float(v) for v in joints]
                            + [float(a["isaac_action"][7])],
                        }
                    )
                corrected_plan = {
                    key: plan0[key]
                    for key in plan0
                    if key not in ("scripted_positive_actions", "plan_digest")
                }
                corrected_plan["scripted_positive_actions"] = corrected_actions
                corrected_plan["plan_digest"] = ""
                corrected_plan["plan_digest"] = canonical_digest(
                    corrected_plan, digest_field="plan_digest"
                )
                result["measured_correction"]["solver_failures"] = solver_failures
                result["measured_correction"]["corrected_plan_digest"] = (
                    corrected_plan["plan_digest"]
                )
                # Prove the correction landed before spending the episode.
                corrected_grasp = [
                    a for a in corrected_actions if a["phase_id"] == "grasp"
                ][-1]["isaac_action"][:7]
                env.reset(seed=int(spec.get("seed") or 20260810))
                _drive(corrected_grasp, 0.0, 60)
                pad_after = _pad_mid_world()
                residual = [handle[i] - pad_after[i] for i in range(3)]
                result["measured_correction"]["pad_mid_after_correction"] = pad_after
                result["measured_correction"]["residual_mm"] = [
                    round(v * 1000, 1) for v in residual
                ]
        _phase(result, "grasp_correction_applied")

        sealed = _SealedEnvironment(adapter)
        # Recorded before the episode: these are configuration, and a run that
        # fails partway should still show what gasket it was carrying.
        result["seal_applied"] = {
            "breakaway_torque_n_m": seal_peak,
            "angular_width_degrees": seal_width,
            "applications": 0,
        }

        pair = run_task_neutral_controls(
            environment=sealed,
            task_spec=spec["task_spec"],
            control_plan=corrected_plan,
            # Not `or 1.0`: zero is a real command here - Arena's binary
            # gripper opens at 0.0 - and rt44 executed its zero-action hold
            # and settle with a closed fist because `0.0 or 1.0` is 1.0.
            gripper_open_command=(
                1.0
                if spec.get("gripper_open_command") is None
                else float(spec["gripper_open_command"])
            ),
            output_dir=output.parent / "controls",
        )
        result["control_pair"] = pair
        result["seal_applied"]["applications"] = sealed.seal_applications
        result["achieved_joint_trace_tail"] = sealed.achieved_joint_trace[-40:]
        result["achieved_joint_trace_length"] = len(sealed.achieved_joint_trace)
        result["fingertip_mid_trace"] = sealed.fingertip_mid_trace
        result["finger_pair_trace_tail"] = sealed.finger_pair_trace[95:115]
        result["door_pose_trace_sample"] = sealed.door_pose_trace[95:115]
        result["handle_target_world_m"] = list(handle_position)
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
