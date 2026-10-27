"""Record one prepared rigid scene's robot-free probe on a rented GPU; query no policy, score nothing.

Beta journey step: L5 team policy evaluation. Owner approval 2026-10-26: under
each capture's website grant, build the scene and run the no-robot check
automatically. This module is the provider runner of a ``native_task_arena``
bundle in execution mode ``rigid_robot_free_probe``
(``rigid_robot_free_probe_bundle``). On the GPU host it:

1. verifies the sealed manifest and the one bound input, which carries the
   control plane's probe request (``team_robot_free_probe.probe_request``,
   bound to the task specification teams are scored with) and the scene's own
   packet plan resolved for its nominal cell;
2. launches Isaac Lab and builds the scene exactly as a team session does, with
   the packet's robot standing idle at its reset pose (nothing commands it);
3. runs ``team_embodiment_runtime.record_arena_robot_free_probe`` once; and
4. writes the raw trace and a ``status: completed`` result wrapper before Isaac
   closes (Isaac may end the process from ``close``).

No policy is loaded or queried and nothing is scored here: the control plane
checks the trace against the request it planned, then scores and retains it
(``rigid_robot_free_probe_runner``). A recording failure is evidence carried in
the trace; a scene that cannot be launched or built ends ``blocked``.

Imports stay absolute and lazy: the worker runs as ``adp_arena_provider_runner``
beside the shipped package, and the control plane imports only its contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blueprint_pipeline.decision_evidence_contracts import canonical_digest

EXECUTION_MODE = "rigid_robot_free_probe"
RESULT_FILENAME = "rigid_robot_free_probe_result.v1.json"
RESULT_SCHEMA_VERSION = "rigid_robot_free_probe_result.v1"
INPUTS_FILENAME = "rigid_robot_free_probe_inputs.v1.json"
INPUTS_SCHEMA = "blueprint.rigid_robot_free_probe_inputs.v1"
# The control plane retains the trace under this name beside the source launch
# (``robot_free_readiness_retention.TRACE_FILENAME``); a test pins them together.
TRACE_FILENAME = "rigid_robot_free_probe_trace.v1.json"
# ``team_robot_free_probe`` constants, literal so the GPU closure never imports the
# control plane's offer publisher; a test pins them together.
REQUEST_SCHEMA = "blueprint.team_robot_free_probe_request.v1"
SCENE_CONTEXT = "native_arena_scene_robot_idle_at_reset"
MANIFEST_SCHEMA = "native_task_arena_provider_bundle.v1"
PLAN_FILENAME = "native_task_arena_scene_plan.v1.json"


class RigidRobotFreeProbeWorkerError(ValueError):
    """The sealed inputs do not bind one probe; nothing was simulated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RigidRobotFreeProbeWorkerError("rigid_robot_free_probe_record_invalid")
    return value


def validate_probe_inputs(value: Any) -> dict[str, Any]:
    """The bound probe input, or raise: one sealed request, for one revision, with its nominal plan."""

    if not isinstance(value, Mapping) or value.get("schema_version") != INPUTS_SCHEMA or value.get(
            "inputs_digest") != canonical_digest(dict(value), digest_field="inputs_digest"):
        raise RigidRobotFreeProbeWorkerError("rigid_robot_free_probe_inputs_invalid")
    request = value.get("probe_request")
    binding = request.get("binding") if isinstance(request, Mapping) else None
    if (not isinstance(binding, Mapping) or request.get("schema_version") != REQUEST_SCHEMA
            or not isinstance(request.get("task_spec"), Mapping) or not isinstance(request.get("plan"), Mapping)
            or request.get("request_digest") != canonical_digest(dict(request), digest_field="request_digest")
            or binding.get("task_spec_digest") != canonical_digest(dict(request["task_spec"]))
            or binding.get("scene_context") != SCENE_CONTEXT or binding.get("scene_collisions_modeled") is not True
            or binding.get("configured_scene_revision_digest") != value.get("scene_revision_digest")):
        raise RigidRobotFreeProbeWorkerError("rigid_robot_free_probe_request_invalid")
    plan = value.get("scene_plan")
    scenario = plan.get("scenario") if isinstance(plan, Mapping) else None
    if (not isinstance(plan, Mapping) or plan.get("task_kind") != "rigid_pick_place"
            or not isinstance(scenario, Mapping) or type(scenario.get("seed")) is not int
            or not isinstance(plan.get("task_spec"), Mapping)):
        raise RigidRobotFreeProbeWorkerError("rigid_robot_free_probe_scene_plan_invalid")
    return dict(value)


def verified_runtime_inputs(runtime_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The sealed manifest and the probe input it binds, reverified on this host."""

    manifest = _read(runtime_dir / "adp_arena_provider_manifest.json")
    if (manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("status") != "ready"
            or manifest.get("execution_mode") != EXECUTION_MODE
            or manifest.get("expected_output_filename") != RESULT_FILENAME
            or manifest.get("policy_candidate_id") is not None or manifest.get("candidate_policy_queried") is not False
            or manifest.get("input_digest") != canonical_digest(manifest, digest_field="input_digest")):
        raise RigidRobotFreeProbeWorkerError("rigid_robot_free_probe_manifest_invalid")
    relative = f"runtime_inputs/{INPUTS_FILENAME}"
    rows = [row for row in manifest.get("bound_runtime_inputs") or [] if isinstance(row, Mapping)]
    path = runtime_dir / relative
    if ([row.get("relative_path") for row in rows] != [relative] or path.is_symlink() or not path.is_file()
            or _sha256(path) != rows[0].get("sha256")):
        raise RigidRobotFreeProbeWorkerError("rigid_robot_free_probe_inputs_binding_invalid")
    plan_rows = [row for row in manifest.get("packet_files") or []
                 if isinstance(row, Mapping) and row.get("relative_path") == PLAN_FILENAME]
    base_path = runtime_dir / "native_task_packet" / PLAN_FILENAME
    if len(plan_rows) != 1 or not base_path.is_file() or _sha256(base_path) != plan_rows[0].get("sha256"):
        raise RigidRobotFreeProbeWorkerError("rigid_robot_free_probe_packet_plan_invalid")
    inputs = validate_probe_inputs(_read(path))
    if (inputs.get("base_plan_digest") != _read(base_path).get("plan_digest")
            or inputs.get("packet_receipt_digest") != manifest.get("packet_receipt_digest")):
        raise RigidRobotFreeProbeWorkerError("rigid_robot_free_probe_packet_binding_invalid")
    return manifest, inputs


@dataclass(frozen=True)
class ProbeRuntime:
    """Every simulator seam the probe touches; tests bind fakes with Isaac's semantics."""

    device: str
    resolve_render_path: Callable[[Mapping[str, Any]], str]
    launch_isaac: Callable[..., tuple[Any, Mapping[str, Any]]]
    prepare_preconstruction: Callable[..., Mapping[str, Any]]
    build_environment: Callable[..., Any]
    read_device_binding: Callable[..., Mapping[str, Any]]
    make_rigid_task_readback: Callable[[Any], Any]
    record_probe: Callable[..., Mapping[str, Any]]


def isaac_probe_runtime() -> ProbeRuntime:
    """Bind the production Isaac Lab and native arena seams (imported only on the GPU host)."""

    from blueprint_pipeline.native_task_arena_device_readback import read_native_task_arena_device_binding
    from blueprint_pipeline.native_task_arena_preconstruction import prepare_native_task_arena_preconstruction
    from blueprint_pipeline.native_task_arena_readback import NativeRigidTaskArenaReadback
    from blueprint_pipeline.native_task_arena_runtime import build_native_task_arena_environment
    from blueprint_pipeline.native_task_isaaclab_launch import NATIVE_TASK_ARENA_DEVICE, launch_native_task_isaaclab
    from blueprint_pipeline.native_task_nurec_render_setup import appearance_render_path_from_plan
    from blueprint_pipeline.team_embodiment_runtime import record_arena_robot_free_probe

    return ProbeRuntime(
        device=NATIVE_TASK_ARENA_DEVICE, resolve_render_path=appearance_render_path_from_plan,
        launch_isaac=launch_native_task_isaaclab, prepare_preconstruction=prepare_native_task_arena_preconstruction,
        build_environment=build_native_task_arena_environment, read_device_binding=read_native_task_arena_device_binding,
        make_rigid_task_readback=NativeRigidTaskArenaReadback, record_probe=record_arena_robot_free_probe)


def _announce(phase: str, status: str = "started") -> None:
    print(f"BLUEPRINT_WAM_RUNTIME_PHASE:rigid_robot_free_probe:{phase}:{status}", flush=True)


def _seal(output_root: Path, result: dict[str, Any]) -> dict[str, Any]:
    result["blockers"] = sorted(set(str(item) for item in result["blockers"] if str(item)))
    result["result_digest"] = ""
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (output_root / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
    return result


def run_rigid_robot_free_probe(*, runtime_dir: Path, output_root: Path,
                               runtime: ProbeRuntime | None = None) -> tuple[dict[str, Any], Any]:
    """Record the probe once; return the sealed result and the simulation app for the caller to close."""

    output_root.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION, "status": "blocked", "execution_mode": EXECUTION_MODE,
        "phase_reached": "not_started", "probe_request_digest": None, "inputs_digest": None,
        "candidate_policy_queried": False, "candidate_outcomes_accessed": False, "policy_loaded": False,
        "robot_commanded": False, "scored_here": False, "native_isaac_executed": False,
        "provider_zero_required_after_return": True, "blockers": [],
    }
    simulation_app = None
    built = None
    try:
        _announce("inputs")
        manifest, inputs = verified_runtime_inputs(runtime_dir)
        request, plan = inputs["probe_request"], inputs["scene_plan"]
        result.update(implementation_commit=manifest.get("implementation_commit"),
                      manifest_input_digest=manifest["input_digest"], inputs_digest=inputs["inputs_digest"],
                      probe_request_digest=request["request_digest"], binding=dict(request["binding"]),
                      phase_reached="inputs_verified")
        _announce("inputs", "completed")
        bound = runtime if runtime is not None else isaac_probe_runtime()
        render_path = bound.resolve_render_path(plan)
        _announce("simulation_app")
        simulation_app, launch = bound.launch_isaac(
            output_root / "native_task_runtime_source_provisioning.v1.json", device=bound.device,
            appearance_render_path=render_path)
        result.update(isaaclab_launch=dict(launch), native_isaac_executed=True, phase_reached="simulation_app_launched")
        _announce("simulation_app", "completed")
        _announce("environment_build")
        preconstruction = dict(bound.prepare_preconstruction(expected_device=bound.device))
        result["preconstruction_device_binding"] = preconstruction
        if preconstruction.get("passed") is not True:
            result["blockers"].extend(preconstruction.get("blockers") or [])
            raise RuntimeError("rigid_robot_free_probe_preconstruction_failed")
        built = bound.build_environment(plan, device=bound.device, bundle_root=runtime_dir / "native_task_packet",
                                        preconstruction_receipt=preconstruction)
        device = dict(bound.read_device_binding(built, expected_device=bound.device))
        result["device_readback"] = device
        if device.get("passed") is not True:
            result["blockers"].extend(device.get("blockers") or [])
            raise RuntimeError("rigid_robot_free_probe_device_binding_failed")
        result["phase_reached"] = "environment_built"
        _announce("environment_build", "completed")
        _announce("probe")
        # The robot stays at its reset pose; only the task object is placed, carried and released.
        trace = dict(bound.record_probe(built=built, task_readback=bound.make_rigid_task_readback(built),
                                        plan=plan, request=request))
        (output_root / TRACE_FILENAME).write_text(json.dumps(trace, sort_keys=True) + "\n", encoding="utf-8")
        result.update(status="completed", phase_reached="probe_recorded", trace=trace,
                      trace_digest=trace.get("trace_digest"), trace_status=trace.get("status"))
        _announce("probe", "completed")
    except Exception as exc:  # noqa: BLE001 - every failure is retained as typed evidence
        launch_errors = getattr(exc, "errors", None)
        if launch_errors is not None:
            result["isaaclab_launch_error"] = {"errors": list(launch_errors),
                                               "diagnostics": dict(getattr(exc, "diagnostics", None) or {})}
        result["blockers"].append(f"rigid_robot_free_probe_failed_at_{result['phase_reached']}:"
                                  f"{type(exc).__name__}:{exc}"[:500])
        result["traceback"] = traceback.format_exc()[-6000:]
    finally:
        close = getattr(getattr(built, "env", None), "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - the recorded trace stays valid evidence
                result["environment_close_error"] = type(exc).__name__
    return _seal(output_root, result), simulation_app


def main() -> int:
    runtime_dir = Path(__file__).resolve().parent
    output_root = Path(os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR", runtime_dir / "runtime_output")).resolve()
    result, simulation_app = run_rigid_robot_free_probe(runtime_dir=runtime_dir, output_root=output_root)
    print("BLUEPRINT_RIGID_ROBOT_FREE_PROBE_" + ("OK" if result["status"] == "completed" else "BLOCKED"), flush=True)
    if simulation_app is not None:
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001 - the result is already durable
            pass
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "EXECUTION_MODE",
    "INPUTS_FILENAME",
    "INPUTS_SCHEMA",
    "ProbeRuntime",
    "RESULT_FILENAME",
    "RESULT_SCHEMA_VERSION",
    "RigidRobotFreeProbeWorkerError",
    "TRACE_FILENAME",
    "isaac_probe_runtime",
    "main",
    "run_rigid_robot_free_probe",
    "validate_probe_inputs",
    "verified_runtime_inputs",
]
