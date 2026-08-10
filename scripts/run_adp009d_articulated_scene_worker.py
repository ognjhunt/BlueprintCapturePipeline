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
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    if arguments.spec:
        spec = Path(arguments.spec).expanduser().resolve()
    else:
        # Beside the runner, where the bundle stages a worker's payload.
        candidates = [
            Path("native") / "adp009d_articulated_scene_spec.json",
            Path(__file__).resolve().parent / "native"
            / "adp009d_articulated_scene_spec.json",
        ]
        spec = next((c.resolve() for c in candidates if c.is_file()), None)

    if arguments.output:
        output = Path(arguments.output).expanduser().resolve()
    else:
        base = os.environ.get("BLUEPRINT_ADP009D_OUTPUT_DIR") or "."
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
        launcher_args = launcher_parser.parse_args([])
        launcher_args.headless = True
        app_launcher = AppLauncher(launcher_args)
        simulation_app = app_launcher.app
    except Exception as exc:  # noqa: BLE001
        result["blockers"].append(
            f"articulated_scene_app_launcher_failed:{type(exc).__name__}:{exc}"
        )
        _persist(output, result)
        return 1

    try:
        import isaaclab.sim as sim_utils  # type: ignore
        from isaaclab_arena.assets.object import Object  # type: ignore
        from isaaclab_arena.assets.object_base import ObjectType  # type: ignore
        from isaaclab_arena.embodiments.droid.droid import (  # type: ignore
            DroidAbsoluteJointPositionEmbodiment,
        )
        from isaaclab_arena.environments.isaaclab_arena_environment import (  # type: ignore
            IsaacLabArenaEnvironment,
        )
        from isaaclab_arena.scene.scene import Scene  # type: ignore
        from isaaclab_arena.tasks.no_task import NoTask  # type: ignore
        from isaaclab_arena.utils.pose import Pose  # type: ignore

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
            asset_path = root / str(row.get("usd_filename") or "")
            if not asset_path.is_file():
                raise RuntimeError(
                    f"articulated_scene_asset_missing:{row.get('name')}:{asset_path.name}"
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
                spawn_cfg_addon={"visible": bool(row.get("visible", True))},
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
        _phase(result, "embodiment_configured")

        assets.append(
            type(
                "SpawnerObject",
                (Object,),
                {
                    "__init__": lambda self, **kw: (
                        setattr(self, "spawner_cfg", kw.pop("spawner_cfg")),
                        Object.__init__(self, object_type=ObjectType.SPAWNER, **kw),
                    )[1]
                },
            )(
                name="light",
                prim_path="/World/Light",
                spawner_cfg=sim_utils.DomeLightCfg(
                    color=(0.75, 0.75, 0.75), intensity=1500.0
                ),
            )
        )

        def configure(cfg):
            from isaaclab_physx.physics import PhysxCfg  # type: ignore

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

        arena_env = IsaacLabArenaEnvironment(
            scene=Scene(assets=assets),
            embodiments=[embodiment],
            task=NoTask(),
            env_config_modifier=configure,
        )
        env = arena_env.get_env()
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

        step_counter = {"value": 0}

        def _task_sample():
            return build_articulated_task_sample(
                joint_ids=joint_ids,
                read_joint_state=_read_joint_state,
                step_index=step_counter["value"],
            )

        robot = scene[embodiment.name] if hasattr(embodiment, "name") else scene["robot"]
        adapter = IsaacEpisodeAdapter(
            env=env,
            robot=robot,
            task_sample_callback=_task_sample,
            action_dim=int(env.unwrapped.action_manager.total_action_dim),
            reset_seed=int(spec.get("seed") or 20260810),
            to_torch=lambda value: value.detach() if hasattr(value, "detach") else torch.as_tensor(value),
        )
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
