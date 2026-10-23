"""Measure an unchanged native cell runner across two explicitly selected runtimes.

This is diagnostic instrumentation, not a new evaluator or allocator. Results
remain incomplete until both real GPU runs and their media/control evidence are
available. Newton and standalone renderers cannot enter this comparison.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable, Mapping

from .decision_evidence_contracts import canonical_digest
from .native_task_isaaclab_ga import IMAGE as GA_IMAGE, COMMIT as GA_COMMIT, ARENA_COMMIT as GA_ARENA_COMMIT
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE as BASE_IMAGE
from .native_task_runtime_source_packet import ISAACLAB_COMMIT as BASE_COMMIT, ARENA_COMMIT as BASE_ARENA_COMMIT

SCHEMA = "native_task_runtime_measurement.v1"
FIXED_FIELDS = ("scene_plan", "task", "assets", "controls", "policy_inputs", "scoring", "cells")


def validate_runtime_identity(variant: str, identity: Mapping[str, Any]) -> None:
    if variant not in {"baseline", "isaaclab_3_ga"}:
        raise ValueError("runtime_comparison_variant_invalid")
    if identity.get("physics_backend") != "isaacsim_physx" or identity.get("renderer") != "isaac_rtx":
        raise ValueError("runtime_comparison_backend_change_requires_separate_experiment")
    expected = ((GA_IMAGE, GA_COMMIT, GA_ARENA_COMMIT) if variant == "isaaclab_3_ga"
                else (BASE_IMAGE, BASE_COMMIT, BASE_ARENA_COMMIT))
    observed = tuple(identity.get(k) for k in ("container_image", "isaaclab_commit", "arena_commit"))
    if observed != expected:
        raise ValueError("runtime_comparison_" + variant + "_identity_invalid")


def comparison_binding(**invariants: Any) -> dict[str, Any]:
    if set(invariants) != set(FIXED_FIELDS) or any(value is None for value in invariants.values()):
        raise ValueError("runtime_comparison_invariants_incomplete")
    rows = {key: canonical_digest({"value": value}) for key, value in invariants.items()}
    return {"digests": rows, "binding_digest": canonical_digest(rows)}


def retained_workcell_binding(*, inputs: Mapping[str, Any], plan: Mapping[str, Any],
                              specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    policies = {}
    for candidate, spec in specs.items():
        # Only simulator release identity can differ. Keep model bytes, prompt,
        # limits, observation transforms and action adapters part of the freeze.
        policies[candidate] = {key: value for key, value in spec.items()
                               if key not in {"runtime_identity", "runtime_identity_digest", "execution_spec_digest"}}
        identity = spec.get("runtime_identity") or {}
        policies[candidate]["runtime_identity"] = {key: value for key, value in identity.items()
                                                   if key not in {"source_commit", "container_image"}}
    return comparison_binding(scene_plan=plan["plan_digest"], task=plan["task_spec"],
        assets={"objects": plan["objects"], "robot": plan["robot"]},
        controls=(inputs["task_success_contract"].get("criteria") or {}).get("controls", {}),
        policy_inputs=policies, scoring=inputs["task_success_contract"], cells=inputs["cells"])


def construction_comparison_binding(plan: Mapping[str, Any], phase_plan: Mapping[str, Any]) -> dict[str, Any]:
    """The precursor replay freezes the existing controller, cameras and gates."""
    return comparison_binding(scene_plan=plan["plan_digest"], task=plan["task_spec"],
        assets={"objects": plan["objects"], "robot": plan["robot"]}, controls=phase_plan,
        policy_inputs={"cameras": plan["cameras"], "policy_queried": False},
        scoring={"task": plan["task_spec"], "phase_plan": phase_plan}, cells=plan["scenario"])


def validate_construction_comparison_request(value: Mapping[str, Any], *, plan: Mapping[str, Any],
                                             phase_plan: Mapping[str, Any]) -> dict[str, Any]:
    if (value.get("schema_version") != "native_task_runtime_comparison_request.v1"
            or value.get("scope") != "retained_construction_replay"
            or value.get("request_digest") != canonical_digest(value, digest_field="request_digest")
            or value.get("binding") != construction_comparison_binding(plan, phase_plan)):
        raise ValueError("runtime_comparison_construction_input_mismatch")
    validate_runtime_identity(str(value.get("variant")), value.get("runtime_identity") or {})
    return dict(value)


def validate_comparison_request(value: Mapping[str, Any], *, inputs: Mapping[str, Any],
                                plan: Mapping[str, Any], specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if (value.get("schema_version") != "native_task_runtime_comparison_request.v1"
            or value.get("request_digest") != canonical_digest(value, digest_field="request_digest")
            or value.get("binding") != retained_workcell_binding(inputs=inputs, plan=plan, specs=specs)):
        raise ValueError("runtime_comparison_request_fixed_input_mismatch")
    validate_runtime_identity(str(value.get("variant")), value.get("runtime_identity") or {})
    return dict(value)


def gpu_observation() -> dict[str, Any]:
    """Read the process's real memory allocation, including renderer allocations."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory", "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True, timeout=10)
    rows = []
    for line in result.stdout.splitlines():
        fields = [s.strip() for s in line.split(",")]
        if len(fields) == 3 and fields[0] == str(os.getpid()):
            value = float(fields[2])
            if not math.isfinite(value) or value < 0:
                raise ValueError("gpu_memory_measurement_invalid")
            row = {"gpu_uuid": fields[1], "used_mib": value}
            # RTX can report the same process twice for one GPU. Identical
            # observations are one sample; conflicting rows remain ambiguous.
            if row not in rows:
                rows.append(row)
    if len(rows) != 1:
        raise ValueError("runtime_comparison_process_gpu_not_unique")
    return rows[0]


def _sync() -> None:
    import torch
    torch.cuda.synchronize()


def _summary(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"count": len(values), "total_seconds": sum(values),
            "median_seconds": statistics.median(values) if values else None,
            "p95_seconds": ordered[math.ceil(len(ordered) * .95) - 1] if ordered else None}


class RuntimeMeasurement:
    """Wrap CellRuntime without changing observations, actions, scoring or controls."""

    def __init__(self, *, output_path: Path, variant: str, runtime_identity: Mapping[str, Any],
                 binding: Mapping[str, Any], clock: Callable[[], float] = time.perf_counter,
                 synchronize: Callable[[], None] = _sync,
                 gpu_reader: Callable[[], Mapping[str, Any]] = gpu_observation):
        validate_runtime_identity(variant, runtime_identity)
        if (set(binding.get("digests", {})) != set(FIXED_FIELDS)
                or binding.get("binding_digest") != canonical_digest(binding["digests"])):
            raise ValueError("runtime_comparison_binding_invalid")
        if output_path.exists():
            raise ValueError("runtime_comparison_output_exists")
        self.output_path = output_path
        self.clock, self.synchronize, self.gpu_reader = clock, synchronize, gpu_reader
        self.timings: dict[str, list[float]] = {name: [] for name in ("startup", "build", "step", "reset")}
        self.value: dict[str, Any] = {
            "schema_version": SCHEMA, "variant": variant, "runtime_identity": dict(runtime_identity),
            "binding": dict(binding), "status": "running", "gpu_samples": [], "camera_gates": [],
            "resets": [], "episodes": [], "errors": [], "physical_evidence": False,
            "measurement_scope": "one_retained_workcell_runtime_diagnostic",
            "timing_method": "perf_counter_with_cuda_sync_at_step_boundaries",
            "memory_method": "sampled_process_nvidia_smi_includes_renderer_lower_bound_peak",
            "standalone_renderer_tested": False, "newton_tested": False,
        }

    def persist(self, status: str | None = None) -> None:
        if status:
            self.value["status"] = status
        self.value["timings"] = {name: _summary(times) for name, times in self.timings.items()}
        self.value["peak_observed_gpu_memory_mib"] = max(
            (row["used_mib"] for row in self.value["gpu_samples"]), default=None)
        self.value["receipt_digest"] = canonical_digest(self.value, digest_field="receipt_digest")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.output_path)

    def sample_memory(self, phase: str) -> None:
        try:
            self.value["gpu_samples"].append({"phase": phase, **self.gpu_reader()})
        except Exception as exc:
            self.value["errors"].append({"phase": phase, "kind": type(exc).__name__, "detail": str(exc)})

    def timed(self, name: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if name in {"step", "reset"}:
            self.synchronize()
        started = self.clock()
        result = function(*args, **kwargs)
        if name in {"step", "reset"}:
            self.synchronize()
        self.timings[name].append(self.clock() - started)
        return result

    def instrument_environment(self, built: Any) -> Any:
        self.sample_memory("build")
        original_step, original_reset = built.env.step, built.env.reset

        def step(*args: Any, **kwargs: Any) -> Any:
            result = self.timed("step", original_step, *args, **kwargs)
            if len(self.timings["step"]) % 50 == 0:
                self.sample_memory("step")
            return result

        def reset(*args: Any, **kwargs: Any) -> Any:
            result = self.timed("reset", original_reset, *args, **kwargs)
            positions = built.env.unwrapped.scene["robot"].data.joint_pos
            values = getattr(positions, "torch", positions).detach().cpu().tolist()
            self.value["resets"].append({"seed": kwargs.get("seed"), "robot_joint_positions": values})
            self.sample_memory("reset")
            return result

        built.env.step, built.env.reset = step, reset
        self.persist()
        return built

    def wrap(self, runtime: Any) -> Any:
        measurement = self

        def launch(*args: Any, **kwargs: Any) -> Any:
            app, receipt = self.timed("startup", runtime.launch_isaac, *args, **kwargs)
            self.value["launch_receipt"] = dict(receipt)
            self.sample_memory("startup")
            self.persist()

            class App:
                def __getattr__(self, name: str) -> Any:
                    return getattr(app, name)

                def close(self, *args: Any, **kwargs: Any) -> Any:
                    measurement.sample_memory("before_close")
                    measurement.persist("measured" if not measurement.value["errors"] else "incomplete")
                    # Kit can end the interpreter here. Evidence must precede it.
                    return app.close(*args, **kwargs)

            return App(), receipt

        def build(*args: Any, **kwargs: Any) -> Any:
            built = self.timed("build", runtime.build_environment, *args, **kwargs)
            return self.instrument_environment(built)

        def camera(*args: Any, **kwargs: Any) -> Any:
            result = runtime.prepolicy_camera_gate(*args, **kwargs)
            self.value["camera_gates"].append(dict(result))
            self.persist()
            return result

        def episode(*args: Any, **kwargs: Any) -> Any:
            result = runtime.run_policy_episode(*args, **kwargs)
            self.value["episodes"].append({key: result.get(key) for key in (
                "episode_id", "candidate_id", "receipt_digest", "score", "contact_force_evidence",
                "candidate_exact_policy_input_manifest_digest", "visual_evidence", "media_artifacts",
                "performance_diagnostics", "scoring_authorized")})
            self.sample_memory("episode")
            self.persist()
            return result

        return replace(runtime, launch_isaac=launch, build_environment=build,
                       prepolicy_camera_gate=camera if runtime.prepolicy_camera_gate else None,
                       run_policy_episode=episode)


def compare_runtime_measurements(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    blockers = []
    if baseline.get("variant") != "baseline" or candidate.get("variant") != "isaaclab_3_ga":
        blockers.append("runtime_comparison_variants_invalid")
    for name, value in (("baseline", baseline), ("candidate", candidate)):
        if (value.get("schema_version") != SCHEMA or value.get("status") != "measured"
                or value.get("receipt_digest") != canonical_digest(value, digest_field="receipt_digest")):
            blockers.append(name + "_measurement_incomplete")
        for field in ("camera_gates", "resets", "episodes", "gpu_samples"):
            if not value.get(field):
                blockers.append(name + "_" + field + "_missing")
    if baseline.get("binding") != candidate.get("binding"):
        blockers.append("fixed_workcell_inputs_changed")
    gpu_sets = [{row["gpu_uuid"] for row in value.get("gpu_samples", [])} for value in (baseline, candidate)]
    if len(gpu_sets[0]) != 1 or gpu_sets[0] != gpu_sets[1]:
        blockers.append("same_gpu_required")
    metrics = {}
    for name in ("startup", "build", "step", "reset"):
        a = baseline.get("timings", {}).get(name, {}).get("median_seconds")
        b = candidate.get("timings", {}).get(name, {}).get("median_seconds")
        if a is None or b is None or a <= 0 or b < 0:
            blockers.append(name + "_measurement_missing")
        else:
            metrics[name] = {"baseline_seconds": a, "candidate_seconds": b,
                             "candidate_change_percent": 100 * (b / a - 1)}
    report = {"schema_version": "native_task_runtime_comparison.v1",
              "status": "blocked" if blockers else "measured_pending_correctness_review",
              "baseline_receipt_digest": baseline.get("receipt_digest"),
              "candidate_receipt_digest": candidate.get("receipt_digest"),
              "memory": {"baseline_peak_observed_mib": baseline.get("peak_observed_gpu_memory_mib"),
                         "candidate_peak_observed_mib": candidate.get("peak_observed_gpu_memory_mib")},
              "metrics": metrics, "blockers": blockers,
              "runtime_promotion_authorized": False, "improvement_qualified": False,
              "correctness_review_required": ["camera_pixels_and_calibration", "reset_state",
                                             "contact_forces", "episode_scores", "complete_policy_media"]}
    report["receipt_digest"] = canonical_digest(report, digest_field="receipt_digest")
    return report


