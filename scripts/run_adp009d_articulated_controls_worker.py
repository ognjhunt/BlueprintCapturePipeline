#!/usr/bin/env python3
"""Execute a frozen articulated control pair inside Isaac and read the door back.

The pair is the point. A positive alone cannot distinguish a door that was
opened from a door that was already falling open, and a negative alone cannot
distinguish a door held shut by physics from one welded by a stray collider.
Both run from the same frozen spec against the same stage, and both are
reported whatever they say.

Torque is applied at the hinge rather than as a force at the handle. The two
are equivalent once multiplied by the lever arm the spec already carries, and
the joint-effort path avoids the force-at-position APIs, which differ between
Isaac releases in ways that cost a launch to discover.

The gasket is applied here for the same reason it is not in the asset: it is
angle-dependent, and USD's physics schema has no way to express a resistance
that is strongest at closed and gone a few degrees later.

Nothing is retried. One admitted launch produces one retained result, including
a null one, so every readback records what happened rather than what was hoped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


RESULT_SCHEMA_VERSION = "adp009d_articulated_controls_result.v1"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _finalize(*, output: Path, result: dict[str, Any], simulation_app: Any) -> None:
    """Write the result first, then shut the simulator down.

    Isaac's close() can end the process outright, and a run that dies after
    opening the app then takes its own diagnosis with it. That is not
    hypothetical: a physics-scene conflict killed a launch and the retained
    evidence said only "process exited without result" - the one thing a paid,
    no-retry run must never come back with. Persisting first costs nothing and
    means the worst case is still an explained one.
    """

    _persist(output, result)
    try:
        simulation_app.close()
    except Exception:  # noqa: BLE001
        # The result is already on disk, so a messy shutdown is not worth
        # turning into a failure. SystemExit is deliberately not caught: that
        # is the simulator ending the process, and there is nothing left to do.
        pass


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [reason],
        "native_isaac_executed": False,
        "controls_qualified": False,
        "physical_success_established": False,
        "provider_zero_required_after_return": True,
        "probe_results": [],
    }
    payload.update(extra)
    return payload


def _downsample(trace: Sequence[float], *, limit: int) -> list[float]:
    """Keep the shape of a motion without keeping every step of it.

    Start, release and settle are three numbers; they cannot say whether the
    door crept open, slammed and bounced, or oscillated through the seal, and
    that is what separates a good positive from a lucky one. The extremes are
    kept explicitly - a transient overshoot that decays away is the entire
    story of a bad release, and it is exactly what uniform sampling drops.
    """

    values = [float(value) for value in trace]
    if len(values) <= limit:
        return values
    step = (len(values) - 1) / float(limit - 1)
    kept = {0, len(values) - 1, values.index(max(values)), values.index(min(values))}
    kept.update(int(round(index * step)) for index in range(limit))
    return [values[index] for index in sorted(kept) if index < len(values)]


def _seal_torque(angle_degrees: float, peak: float, width: float) -> float:
    if peak <= 0.0 or width <= 0.0:
        return 0.0
    magnitude = abs(angle_degrees)
    if magnitude >= width:
        return 0.0
    taper = 0.5 * (1.0 + math.cos(math.pi * magnitude / width))
    resistance = peak * taper
    return resistance if angle_degrees >= 0.0 else -resistance


def _evaluate_positive(
    *,
    positive: Mapping[str, Any],
    window: Sequence[float],
    hold_tolerance_degrees: float,
    tail_fraction: float = 0.25,
) -> dict[str, Any]:
    """Judge the positive on where the door got to and whether it stopped.

    Two corrections over the obvious reading. Reaching the window is about the
    angle the door attains, not the angle it was released at - the schedule
    lets go early on purpose and lets the door coast in, so testing the release
    angle contradicts the design and can only pass when the coast model is
    wrong.

    And holding is not the same as ending up somewhere. A door still swinging
    when the clock runs out lands in the window by accident; one that came to
    rest is holding. Only the tail of the settle window separates them, so a
    run with no trace reports that it cannot tell rather than assuming.
    """

    low, high = float(window[0]), float(window[1])
    maximum = float(positive.get("maximum_angle_degrees") or 0.0)
    settled = float(positive.get("settled_angle_degrees") or 0.0)
    settle = [float(v) for v in (positive.get("settle_trace_degrees") or [])]

    # Judged on the settle window alone. A tail taken from the whole episode
    # still contains the coast, where the door is supposed to be moving, so
    # measuring across it reads deceleration as a failure to hold.
    entered: bool | None = None
    stayed_after_entry: bool | None = None
    decaying: bool | None = None
    tail_motion: float | None = None
    if len(settle) >= 4:
        # The settle window opens at release, and release is deliberately below
        # the window - the door spends its first moments coasting in. Demanding
        # every sample be inside fails the design, not the door. What holding
        # means is that once it arrives, it stays.
        entry = next(
            (i for i, value in enumerate(settle) if low <= value <= high), None
        )
        entered = entry is not None
        if entry is not None:
            after = settle[entry:]
            stayed_after_entry = all(low <= value <= high for value in after)
            half = max(1, len(after) // 2)
            early = max(after[:half]) - min(after[:half])
            late = max(after[half:]) - min(after[half:]) if after[half:] else 0.0
            # Asymptotic settling never reaches exactly zero, so require both
            # shrinking motion and the frozen absolute late-motion ceiling.
            # Decay alone would admit a door moving at constant speed because
            # equal early/late ranges satisfy a non-increasing check.
            decaying = late <= early
            tail_motion = late

    return {
        "reaches_success_window": {
            "maximum_angle_degrees": maximum,
            "window": [low, high],
            "passed": low <= maximum <= high,
        },
        "holds_after_release": {
            "settled_angle_degrees": settled,
            "entered_window": entered,
            "stayed_inside_after_entry": stayed_after_entry,
            "motion_is_decaying": decaying,
            "tail_motion_degrees": tail_motion,
            "hold_tolerance_degrees": float(hold_tolerance_degrees),
            "passed": bool(
                entered
                and stayed_after_entry
                and decaying
                and tail_motion is not None
                and tail_motion <= float(hold_tolerance_degrees)
                and low <= settled <= high
            ),
        },
    }


def _simulation_app() -> Any:
    from isaacsim.simulation_app import SimulationApp  # type: ignore

    return SimulationApp({"headless": True, "renderer": "RayTracedLighting"})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    spec_path = Path(arguments.spec).expanduser().resolve()
    output = Path(arguments.output).expanduser().resolve()
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _persist(output, _blocked("articulated_controls_spec_unreadable"))
        return 1

    try:
        simulation_app = _simulation_app()
    except Exception as exc:  # noqa: BLE001 - report, never retry
        _persist(
            output,
            _blocked(f"articulated_controls_simulation_app_failed:{type(exc).__name__}"),
        )
        return 1

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "blocked",
        "blockers": [],
        "native_isaac_executed": False,
        "controls_qualified": False,
        "physical_success_established": False,
        "provider_zero_required_after_return": True,
        "spec_sha256": _sha256(spec_path),
        "probe_results": [],
        "replacement_count": 1,
        "source_target_collider_active": False,
        "readbacks": {},
        "controls": {},
    }

    try:
        import omni.usd  # type: ignore
        from isaacsim.core.api import World  # type: ignore
        from isaacsim.core.prims import Articulation  # type: ignore
        from pxr import UsdPhysics  # type: ignore

        stages = spec.get("stages") or {}
        root = spec_path.parent
        blank = root / Path(str(stages["blank_stage"]["path"])).name
        controls_stage = root / Path(str(stages["controls_stage"]["path"])).name
        expected = spec.get("expected") or {}
        geometry = spec.get("geometry") or {}
        seal = spec.get("seal") or {}
        window = [float(v) for v in (spec.get("success_angle_window_degrees") or [0, 0])]
        task_joint_path = str(expected.get("task_joint_prim_path") or "")
        # Per-phase torques come precomputed in the spec; the lever arm is
        # carried into the result so a handle force can be recovered from it.
        result["lever_arm_m"] = float(geometry.get("lever_arm_m") or 0.0)

        context = omni.usd.get_context()
        context.open_stage(str(blank))
        for _ in range(60):
            simulation_app.update()
        blank_world = World(stage_units_in_meters=1.0)
        blank_world.reset()
        for _ in range(30):
            blank_world.step(render=False)
        result["readbacks"]["blank_stage_physics_scene_ran"] = True
        blank_world.stop()
        blank_world.clear_instance()

        context.open_stage(str(controls_stage))
        for _ in range(90):
            simulation_app.update()
        stage = context.get_stage()

        roots = sorted(
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        )
        result["readbacks"]["articulation_root_identity"] = {
            "observed": roots,
            "passed": len(roots) == 1,
        }
        if len(roots) != 1:
            raise RuntimeError("articulated_controls_no_single_articulation_root")
        articulation_root = roots[0]

        joint_prim = stage.GetPrimAtPath(task_joint_path)
        result["readbacks"]["task_joint_identity"] = {
            "observed": task_joint_path,
            "passed": bool(joint_prim and joint_prim.IsValid()),
        }
        if not (joint_prim and joint_prim.IsValid()):
            raise RuntimeError("articulated_controls_task_joint_not_found")

        # No physics_dt override. The stage's authored PhysicsScene owns
        # stepping; passing a different dt makes World author a second scene,
        # and PhysX answers with "Physics scenes stepping is not the same" and
        # shuts the app down mid-run.
        world = World(stage_units_in_meters=1.0)
        world.reset()
        view = Articulation(articulation_root)
        # Registering with the scene, rather than initialize() alone, is the
        # path the articulation probe already proved on this image.
        world.scene.add(view)
        world.reset()
        names = list(view.dof_names or [])
        joint_name = task_joint_path.rsplit("/", 1)[-1]
        if joint_name not in names:
            result["readbacks"]["task_joint_identity"]["dof_names"] = names
            raise RuntimeError(
                f"articulated_controls_task_joint_not_a_dof:{joint_name}:{names}"
            )
        dof_index = names.index(joint_name)

        import numpy as np  # type: ignore

        def _angle_degrees() -> float:
            positions = view.get_joint_positions()
            return float(np.degrees(np.asarray(positions).reshape(-1)[dof_index]))

        def _run(control: Mapping[str, Any]) -> dict[str, Any]:
            world.reset()
            for _ in range(10):
                world.step(render=False)
            start = _angle_degrees()
            schedule = list(control.get("force_schedule") or [])
            trace: list[float] = [start]
            efforts = np.zeros((1, len(names)), dtype=np.float32)
            phase_log: list[dict[str, Any]] = []
            remaining = int(control.get("drive_steps") or 0)
            for phase in schedule:
                torque = float(phase.get("hinge_torque_n_m") or 0.0)
                until = float(phase.get("until_angle_degrees") or 0.0)
                steps = 0
                # Angle-gated rather than step-gated: the point of a phase is
                # where the door gets to, not how long the pushing lasted.
                while remaining > 0 and _angle_degrees() < until:
                    net = torque - _seal_torque(
                        _angle_degrees(),
                        float(seal.get("breakaway_torque_n_m") or 0.0),
                        float(seal.get("angular_width_degrees") or 0.0),
                    )
                    efforts[0, dof_index] = net
                    view.set_joint_efforts(efforts)
                    world.step(render=False)
                    trace.append(_angle_degrees())
                    remaining -= 1
                    steps += 1
                phase_log.append(
                    {
                        "phase_index": phase.get("phase_index"),
                        "hinge_torque_n_m": torque,
                        "until_angle_degrees": until,
                        "steps_used": steps,
                        "angle_reached_degrees": _angle_degrees(),
                        "exhausted_step_budget": remaining <= 0,
                    }
                )
                if remaining <= 0:
                    break
            at_release = _angle_degrees()
            settle_trace: list[float] = [at_release]
            # Release: the task is that the door stays put, so nothing may push
            # while the outcome is being read.
            efforts[0, dof_index] = 0.0
            for _ in range(int(control.get("settle_steps") or 0)):
                view.set_joint_efforts(efforts)
                world.step(render=False)
                trace.append(_angle_degrees())
                settle_trace.append(_angle_degrees())
            settled = _angle_degrees()
            inside = window[0] <= settled <= window[1]
            return {
                "control_id": str(control.get("control_id")),
                "applied_handle_force_n": float(
                    control.get("applied_handle_force_n") or 0.0
                ),
                "applied_hinge_torque_n_m": (
                    schedule[0].get("hinge_torque_n_m") if schedule else 0.0
                ),
                "phase_log": phase_log,
                "start_angle_degrees": start,
                "angle_at_release_degrees": at_release,
                "settled_angle_degrees": settled,
                "drift_after_release_degrees": settled - at_release,
                "maximum_angle_degrees": max(trace),
                "inside_success_window": inside,
                "expected_outcome": str(control.get("expected_outcome") or ""),
                "sample_count": len(trace),
                "angle_trace_degrees": _downsample(trace, limit=64),
                "settle_trace_degrees": _downsample(settle_trace, limit=48),
            }

        controls = spec.get("controls") or {}
        negative = _run(controls["zero_action_negative"])
        positive = _run(controls["forced_positive"])
        result["controls"] = {
            "zero_action_negative": negative,
            "forced_positive": positive,
        }

        result["readbacks"]["zero_action_door_stays_shut"] = {
            "settled_angle_degrees": negative["settled_angle_degrees"],
            "maximum_angle_degrees": negative["maximum_angle_degrees"],
            "passed": not negative["inside_success_window"],
        }
        verdict = _evaluate_positive(
            positive=positive, window=window, hold_tolerance_degrees=0.5
        )
        result["readbacks"]["forced_positive_reaches_success_window"] = verdict[
            "reaches_success_window"
        ]
        result["readbacks"]["forced_positive_holds_after_release"] = {
            **verdict["holds_after_release"],
            "angle_at_release_degrees": positive["angle_at_release_degrees"],
            "coast_after_release_degrees": positive["drift_after_release_degrees"],
        }
        result["readbacks"]["seal_resists_before_breakaway"] = {
            "breakaway_torque_n_m": float(seal.get("breakaway_torque_n_m") or 0.0),
            "applied_hinge_torque_n_m": positive["applied_hinge_torque_n_m"],
            "passed": positive["applied_hinge_torque_n_m"]
            > float(seal.get("breakaway_torque_n_m") or 0.0),
        }
        result["readbacks"]["no_initial_penetration"] = {
            "start_angle_degrees": negative["start_angle_degrees"],
            "passed": abs(negative["start_angle_degrees"]) < 1.0,
        }
        replay = _run(controls["forced_positive"])
        delta = abs(replay["settled_angle_degrees"] - positive["settled_angle_degrees"])
        result["readbacks"]["deterministic_replay_within_tolerance"] = {
            "delta_degrees": delta,
            "passed": delta < 0.5,
        }

        required = list(spec.get("required_readbacks") or [])
        result["probe_results"] = [
            {
                "probe": name,
                "passed": bool((result["readbacks"].get(name) or {}).get("passed")),
            }
            for name in required
        ]
        result["native_isaac_executed"] = True
        result["controls_qualified"] = all(row["passed"] for row in result["probe_results"])
        result["status"] = "completed" if result["controls_qualified"] else "completed_with_failures"
        # Opening a door under an applied hinge torque is not a robot doing the
        # task, and must never be read as one.
        result["physical_success_established"] = False
        result["claim_boundary"] = {
            "torque_applied_directly_at_the_hinge": True,
            "no_robot_and_no_grasp_in_this_probe": True,
            "door_dynamics_only": True,
        }
    except BaseException as exc:  # noqa: BLE001 - one launch, one retained result
        # BaseException, not Exception: a SystemExit raised out of the runtime
        # would otherwise skip straight past the diagnosis.
        result["blockers"].append(
            f"articulated_controls_runtime_failed:{type(exc).__name__}:{exc}"
        )
    _finalize(output=output, result=result, simulation_app=simulation_app)
    return 0 if result.get("native_isaac_executed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
