"""Real MuJoCo development-execution worker for measurement benchmarks.

This is the first engine-in-the-loop worker behind the fail-closed execution
boundary in ``measurement_adapter_execution``: it receives one digest-bound
execution request (``--request``), runs actual MuJoCo rigid-body physics for a
deterministic development case, and writes one bounded, digest-bound worker
result (``--output``).

Boundaries, in line with the measurement-routing research:

- development split only (the execution-request contract already rejects
  qualification cases for local runners);
- CPU physics, no rendering, no network, no provider spend, no secrets;
- the engine version pin is explicit: ``solver_settings.engine_version_policy``
  must be ``require_target`` (block on any mismatch with the descriptor's
  pinned target) or ``record_actual_development_only`` (run, but record the
  observed version and the mismatch honestly) — there is no silent default;
- the rollout is executed twice and must reproduce bit-identically, otherwise
  the worker fails closed with ``nondeterministic_execution``;
- a completed result is development evidence for R3/R4 progress. It is never
  a qualification, a route authorization, or physical-success evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from .measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_worker_result,
    validate_measurement_adapter_execution_request,
)


MUJOCO_WORKER_ID = "blueprint-mujoco-development-worker"
MUJOCO_WORKER_VERSION = "1"

ENGINE_VERSION_POLICIES = frozenset(
    {"require_target", "record_actual_development_only"}
)

SUPPORTED_SCENES = frozenset({"box_settle"})

_BOUNDS = {
    "timestep": (1e-4, 0.02),
    "steps": (10, 50_000),
    "drop_height_m": (0.0, 2.0),
    "box_half_extent_m": (0.005, 0.5),
    "friction": (0.0, 2.0),
}


class MujocoWorkerError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _number_in_bounds(value: Any, bounds_key: str) -> float:
    low, high = _BOUNDS[bounds_key]
    if isinstance(value, bool):
        raise MujocoWorkerError(f"operating_point_out_of_bounds:{bounds_key}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MujocoWorkerError(f"operating_point_out_of_bounds:{bounds_key}") from exc
    if not low <= number <= high:
        raise MujocoWorkerError(f"operating_point_out_of_bounds:{bounds_key}")
    return number


def _box_settle_mjcf(
    *, timestep: float, drop_height: float, half_extent: float, friction: float
) -> str:
    return f"""
<mujoco model="blueprint-box-settle">
  <option timestep="{timestep}" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" friction="{friction} 0.005 0.0001"/>
    <body name="box" pos="0 0 {drop_height}">
      <freejoint/>
      <geom name="box" type="box" size="{half_extent} {half_extent} {half_extent}"
            mass="0.25" friction="{friction} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def _rollout(mujoco_module: Any, model: Any, steps: int) -> dict[str, Any]:
    data = mujoco_module.MjData(model)
    trajectory = hashlib.sha256()
    first_contact_step: int | None = None
    max_penetration = 0.0
    for step in range(steps):
        mujoco_module.mj_step(model, data)
        trajectory.update(data.qpos.tobytes())
        if data.ncon > 0 and first_contact_step is None:
            first_contact_step = step
        for index in range(data.ncon):
            distance = float(data.contact[index].dist)
            if distance < 0.0:
                max_penetration = max(max_penetration, -distance)
    return {
        "trajectory_digest": "sha256:" + trajectory.hexdigest(),
        "settled_height_m": float(data.qpos[2]),
        "first_contact_step": first_contact_step,
        "max_penetration_m": max_penetration,
        "qpos_dtype": str(data.qpos.dtype),
    }


def execute_mujoco_benchmark_case(request_value: Mapping[str, Any]) -> dict[str, Any]:
    """Run real MuJoCo physics for one validated development execution request.

    Always returns a valid ``measurement_adapter_worker_result.v1`` mapping
    (``completed``, ``blocked``, or ``failed``); typed codes replace silent
    fallbacks throughout.
    """

    request = validate_measurement_adapter_execution_request(request_value)
    runtime = dict(request["runtime_configuration"])
    settings = dict(runtime.get("solver_settings") or {})
    target_version = str(runtime.get("target_engine_version") or "")
    seed = runtime.get("seed")

    def _result(
        status: str,
        *,
        metrics: Mapping[str, Any] | None = None,
        observations: Mapping[str, Any] | None = None,
        failure_codes: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        base_observations = {
            "engine_version": "unavailable",
            "backend_id": runtime.get("backend_id"),
            "precision": runtime.get("precision"),
            "seed": seed,
            "worker_id": MUJOCO_WORKER_ID,
            "worker_version": MUJOCO_WORKER_VERSION,
            "target_engine_version": target_version,
        }
        base_observations.update(dict(observations or {}))
        return build_measurement_adapter_worker_result(
            request,
            status=status,
            observed_metrics=dict(metrics or {}),
            unsafe_condition_predicted=None,
            runtime_observations=base_observations,
            failure_codes=failure_codes,
        )

    try:
        import mujoco  # noqa: PLC0415 - engine import is the worker's purpose
    except ImportError:
        return _result("blocked", failure_codes=("mujoco_not_installed",))

    engine_version = str(mujoco.__version__)
    version_matches = engine_version == target_version
    policy = str(settings.get("engine_version_policy") or "")
    if policy not in ENGINE_VERSION_POLICIES:
        return _result(
            "blocked",
            observations={"engine_version": engine_version},
            failure_codes=("engine_version_policy_missing_or_invalid",),
        )
    if policy == "require_target" and not version_matches:
        return _result(
            "blocked",
            observations={
                "engine_version": engine_version,
                "engine_version_matches_target": False,
                "engine_version_policy": policy,
            },
            failure_codes=(
                f"engine_version_mismatch:installed-{engine_version}"
                f":target-{target_version}",
            ),
        )

    operating_point = dict(request["case_manifest"].get("operating_point") or {})
    scene = str(operating_point.get("scene") or "")
    if scene not in SUPPORTED_SCENES:
        return _result(
            "blocked",
            observations={"engine_version": engine_version},
            failure_codes=(f"scene_unknown:{scene or 'missing'}",),
        )
    try:
        timestep = _number_in_bounds(settings.get("timestep"), "timestep")
        steps_raw = settings.get("steps")
        if isinstance(steps_raw, bool) or not isinstance(steps_raw, int):
            raise MujocoWorkerError("solver_settings_invalid:steps")
        low, high = _BOUNDS["steps"]
        if not low <= steps_raw <= high:
            raise MujocoWorkerError("solver_settings_invalid:steps")
        drop_height = _number_in_bounds(
            operating_point.get("drop_height_m"), "drop_height_m"
        )
        half_extent = _number_in_bounds(
            operating_point.get("box_half_extent_m"), "box_half_extent_m"
        )
        friction = _number_in_bounds(operating_point.get("friction"), "friction")
    except MujocoWorkerError as exc:
        return _result(
            "blocked",
            observations={"engine_version": engine_version},
            failure_codes=exc.codes,
        )

    mjcf = _box_settle_mjcf(
        timestep=timestep,
        drop_height=drop_height,
        half_extent=half_extent,
        friction=friction,
    )
    model_digest = "sha256:" + hashlib.sha256(mjcf.encode("utf-8")).hexdigest()
    try:
        model = mujoco.MjModel.from_xml_string(mjcf)
        first = _rollout(mujoco, model, steps_raw)
        second = _rollout(mujoco, model, steps_raw)
    except Exception as exc:  # noqa: BLE001 - typed forwarding, never a crash
        return _result(
            "failed",
            observations={"engine_version": engine_version},
            failure_codes=(f"mujoco_execution_error:{type(exc).__name__}",),
        )

    deterministic = first["trajectory_digest"] == second["trajectory_digest"]
    observations = {
        "engine_version": engine_version,
        "engine_version_matches_target": version_matches,
        "engine_version_policy": policy,
        "deterministic_rerun_match": deterministic,
        "model_digest": model_digest,
        "trajectory_digest": first["trajectory_digest"],
        "qpos_dtype": first["qpos_dtype"],
        "steps": steps_raw,
        "timestep": timestep,
        "scene": scene,
        "seed_unused_passive_rollout": True,
        "headless": True,
        "rendering_used": False,
        "network_used": False,
    }
    if not deterministic:
        return _result(
            "failed",
            observations=observations,
            failure_codes=("nondeterministic_execution",),
        )
    metrics = {
        "final_object_pose_error": first["settled_height_m"],
        "penetration": first["max_penetration_m"],
        "contact_sequence": float(
            first["first_contact_step"] if first["first_contact_step"] is not None else -1
        ),
    }
    allowed = set(request["case_manifest"]["requested_metric_ids"])
    return _result(
        "completed",
        metrics={key: value for key, value in metrics.items() if key in allowed},
        observations=observations,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    try:
        request = json.loads(Path(arguments.request).read_text(encoding="utf-8"))
        result = execute_mujoco_benchmark_case(request)
    except (OSError, json.JSONDecodeError, MeasurementAdapterExecutionError) as exc:
        print(f"mujoco_worker_request_invalid:{type(exc).__name__}", file=sys.stderr)
        return 2
    Path(arguments.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ENGINE_VERSION_POLICIES", "MUJOCO_WORKER_ID", "MUJOCO_WORKER_VERSION",
    "MujocoWorkerError", "SUPPORTED_SCENES", "execute_mujoco_benchmark_case",
    "main",
]
