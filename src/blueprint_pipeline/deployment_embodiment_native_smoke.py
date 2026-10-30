"""Bounded gravity-off native controller diagnostic; no task or policy evidence."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest
from .deployment_embodiment_asset_import import verify_prepared_asset_directory
from .deployment_embodiment_preparation_contract import read, sealed, write_once
from .deployment_embodiment_runtime_configuration import verify_runtime_configuration
from .native_franka_pose_servo import contract_xyzw_to_native_xyzw
from .team_embodiment import camera_names, gripper_channels, validate_embodiment_manifest, verify_model_file

SCHEMA = "embodiment_native_controller_smoke.v1"
FILENAME = SCHEMA + ".json"
PHYSICS_DT = 1 / 120
MAX_TICKS = 600
MAX_WALL_SECONDS = 90.0
RESET_TOLERANCE = .001


def robot_configuration(manifest: Mapping[str, Any], model_path: Path) -> dict:
    """The existing Team factory's robot-only inputs, with no invented scene/task."""
    channels, joints, gains = gripper_channels(manifest), manifest["joints"], manifest["actuator"]
    return {"robot_id": "team_embodiment", "usd_path": str(model_path),
        "base_mode": "fixed_at_station",
        "base_pose_world": {"position_world_m": [0., 0., 0.],
                            "orientation_xyzw": contract_xyzw_to_native_xyzw([0., 0., 0., 1.])},
        "joint_names": [j["name"] for j in joints],
        "joint_controls": {j["name"]: j["control"] for j in joints},
        "joint_gains": {j["name"]: {key: float(j.get(key, gains[key])) for key in ("stiffness", "damping")}
                        for j in joints},
        "actuator": dict(gains), "gripper_joints": [j for c in channels for j in c["joints"]],
        "joint_reset_positions_rad": dict(manifest["reset_joint_positions"]),
        "camera_names": camera_names(manifest), "enabled_self_collisions": False}


class NativeArticulation:
    """Use the pinned Beta2 APIs and the same implicit drives as Team policies."""
    def __init__(self, robot_row: Mapping[str, Any]):
        import torch
        from isaaclab.assets import Articulation
        from isaaclab.sim import SimulationCfg, SimulationContext
        from isaaclab_physx.physics import PhysxCfg
        from isaaclab_arena.utils.pose import Pose
        from .team_embodiment_runtime import build_team_embodiment

        self.torch, self.robot, self.sim = torch, None, None
        try:
            if SimulationContext.instance() is not None:
                SimulationContext.clear_instance()
            self.sim = SimulationContext(SimulationCfg(
                dt=PHYSICS_DT, device="cuda:0", gravity=(0., 0., 0.),
                create_stage_in_memory=True, physics=PhysxCfg(solver_type=1)))
            embodiment = build_team_embodiment(robot_row, enable_cameras=False, pose_class=Pose)
            cfg = embodiment.scene_config.robot.replace(prim_path="/World/Robot")
            cfg.init_state = cfg.init_state.replace(
                pos=tuple(robot_row["base_pose_world"]["position_world_m"]),
                rot=tuple(contract_xyzw_to_native_xyzw(robot_row["base_pose_world"]["orientation_xyzw"])))
            cfg.spawn.activate_contact_sensors = False
            self.robot = Articulation(cfg)
            self.sim.reset()
            self.robot.update(PHYSICS_DT)
            self.joint_names = list(self.robot.joint_names)
            self.runtime_readback = {"articulation_class": type(self.robot).__module__ + "." + type(self.robot).__name__,
                                    "device": str(self.robot.device), "physics_dt_seconds": self.sim.get_physics_dt()}
        except Exception:
            self.close()
            raise

    def _tensor(self, positions):
        return self.torch.tensor([[positions[name] for name in self.joint_names]],
                                 dtype=self.torch.float32, device="cuda:0")

    def reset(self, positions):
        value = self._tensor(positions)
        self.robot.reset()
        self.robot.write_joint_state_to_sim_index(position=value, velocity=self.torch.zeros_like(value))
        self.robot.set_joint_velocity_target_index(target=self.torch.zeros_like(value))
        self.robot.set_joint_position_target_index(target=value)

    def step(self, targets):
        self.robot.set_joint_position_target_index(target=self._tensor(targets))
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.robot.update(PHYSICS_DT)

    def positions(self):
        from .team_embodiment_runtime import _array
        return dict(zip(self.joint_names, map(float, _array(self.robot.data.joint_pos)[0]), strict=True))

    def close(self):
        if self.sim is not None:
            try:
                self.sim.stop()
            finally:
                self.sim.clear_instance()
                self.sim = None


def run_native_controller_smoke(binding: Mapping[str, Any], model_path: Path, output: Path, *,
                                configuration_dir: Path,
                                backend_factory=NativeArticulation, now=time.monotonic,
                                wall_budget_seconds: float = MAX_WALL_SECONDS) -> dict:
    """Record measured reset/drive response, retaining imports even when diagnostics fail."""
    start, backend, ticks = now(), None, 0
    result: dict[str, Any] = {"schema_version": SCHEMA, "status": "failed", "probes": [],
        "claim_ceiling": "native_controller_diagnostic_only", "candidate_policy_queried": False,
        "task_executed": False, "physical_success_proven": False, "collision_behavior_qualified": False,
        "device": "cuda:0", "gravity_m_s2": [0., 0., 0.], "self_collisions_enabled": False,
        "cameras_enabled": False, "physics_dt_seconds": PHYSICS_DT, "maximum_ticks": MAX_TICKS,
        "maximum_wall_seconds": min(MAX_WALL_SECONDS, max(0., wall_budget_seconds)),
        "quaternion_convention": "xyzw", "reset_tolerance_SI": RESET_TOLERANCE,
        "minimum_command_response_fraction": .1, "inactive_gripper_drift_fraction": .02,
        "reasons": [], "binding_digest": canonical_digest(binding)}
    deadline = start + result["maximum_wall_seconds"]

    def check():
        if now() >= deadline:
            raise TimeoutError("native_controller_smoke_wall_budget_exhausted")
        if ticks >= MAX_TICKS:
            raise TimeoutError("native_controller_smoke_tick_budget_exhausted")

    def advance(targets, count, *, reference=None, peaks=None):
        nonlocal ticks
        for _ in range(count):
            check()
            backend.step(targets)
            ticks += 1
            measured = backend.positions()
            if set(measured) != set(reset) or not all(math.isfinite(value) for value in measured.values()):
                raise ValueError("native_controller_joint_readback_invalid")
            result["last_joint_readback"] = measured
            if any(not lo - 1e-4 <= measured[name] <= hi + 1e-4 for name, (lo, hi) in bounds.items()):
                raise ValueError("native_controller_measured_joint_limit_exceeded")
            if peaks is not None:
                for name in reset:
                    delta = abs(measured[name] - reference[name])
                    peaks[name] = max(peaks.get(name, 0.), delta)
        return measured

    try:
        check()
        manifest = validate_embodiment_manifest(binding["robot_embodiment"])
        if manifest["base"]["mode"] != "fixed" or any(j["control"] != "position" for j in manifest["joints"]):
            raise ValueError("native_controller_smoke_fixed_position_robot_required")
        prepared = verify_prepared_asset_directory(model_path.parent)
        configuration = verify_runtime_configuration(configuration_dir)
        expected_manifest = validate_embodiment_manifest({**configuration["manifest_fields"],
            **binding["runtime_settings"], "model": manifest["model"]})
        model = verify_model_file(manifest, model_path)
        if (model != {key: binding["model_record"][key] for key in model}
                or manifest != expected_manifest
                or configuration["configuration_digest"] != binding["configuration_digest"]
                or configuration["source_bundle_digest"] != prepared["source_bundle_digest"]
                or prepared["import_digest"] != binding["model_record"]["source_import_provenance"]["import_digest"]):
            raise ValueError("native_controller_source_model_binding_changed")
        result.update(embodiment_id=manifest["embodiment_id"], embodiment_digest=manifest["embodiment_digest"],
                      model_sha256=model["sha256"], import_digest=prepared["import_digest"],
                      source_bundle_digest=prepared["source_bundle_digest"], configuration_digest=binding["configuration_digest"])
        check()
        backend = backend_factory(robot_configuration(manifest, model_path))
        result["runtime_readback"] = dict(backend.runtime_readback)
        reset = {name: float(value) for name, value in manifest["reset_joint_positions"].items()}
        bounds = {j["name"]: (j["lower"], j["upper"]) for j in manifest["joints"]}
        bounds.update({j["name"]: tuple(sorted((j["open"], j["closed"])))
                       for c in gripper_channels(manifest) for j in c["joints"]})
        result["measured_joint_bounds_SI"] = {name: list(span) for name, span in bounds.items()}
        result["maximum_command_response_fraction"] = 1.5
        result["inactive_joint_drift_limits_SI"] = {name: max(1e-4, min(.002, (hi - lo) * .02))
                                                    for name, (lo, hi) in bounds.items()}
        result["native_joint_names"] = list(backend.joint_names)
        if len(backend.joint_names) != len(set(backend.joint_names)) or set(backend.joint_names) != set(reset):
            raise ValueError("native_controller_joint_names_mismatch")
        check()
        backend.reset(reset)
        baseline = advance(reset, 20)
        errors = {name: abs(baseline[name] - value) for name, value in reset.items()}
        result["reset"] = {"targets": reset, "observed": baseline, "absolute_errors": errors,
                           "passed": max(errors.values(), default=0.) <= RESET_TOLERANCE}
        if not result["reset"]["passed"]:
            result["reasons"].append("native_controller_reset_error")
        joints = {joint["name"]: joint for joint in manifest["joints"]}
        channels = gripper_channels(manifest)
        groups = manifest.get("arms") or [{"name": "arm", "joint_names": list(joints)}]
        probes = []
        for group in groups:
            name = group["joint_names"][0]
            joint = joints[name]
            delta = min(.025, (joint["upper"] - joint["lower"]) * .02)
            direction = 1 if reset[name] + delta <= joint["upper"] else -1
            probes.append(("arm", group["name"], {name: reset[name] + direction * delta}, []))
        for channel in channels:
            targets = {}
            for joint in channel["joints"]:
                fraction = .25 if abs(reset[joint["name"]] - joint["open"]) <= abs(reset[joint["name"]] - joint["closed"]) else .75
                targets[joint["name"]] = joint["open"] + fraction * (joint["closed"] - joint["open"])
            inactive = [j for c in channels if c["name"] != channel["name"] for j in c["joints"]]
            probes.append(("gripper", channel["name"], targets, inactive))
        for kind, name, changes, inactive in probes:
            check()
            backend.reset(reset)
            before = advance(reset, 10)
            target = {**reset, **changes}
            peaks = {}
            after = advance(target, 80, reference=before, peaks=peaks)
            fractions = {joint: (after[joint] - before[joint]) / (value - before[joint])
                         for joint, value in changes.items() if abs(value - before[joint]) > 1e-9}
            drift = {joint: peaks[joint] for joint in reset if joint not in changes}
            peak_fractions = {joint: peaks[joint] / abs(value - before[joint])
                              for joint, value in changes.items() if abs(value - before[joint]) > 1e-9}
            passed = bool(fractions) and all(.1 <= value <= 1.5 for value in fractions.values()) and all(
                value <= 1.5 for value in peak_fractions.values()) and all(
                value <= result["inactive_joint_drift_limits_SI"][joint] for joint, value in drift.items())
            result["probes"].append({"kind": kind, "name": name, "status": "passed" if passed else "failed",
                "targets": changes, "before": before, "after": after, "response_fractions": fractions,
                "peak_response_fractions": peak_fractions, "inactive_drift_sampling": "maximum_per_tick",
                "inactive_joint_drift": drift,
                "inactive_gripper_drift": {j["name"]: drift[j["name"]] for j in inactive}})
            if not passed:
                result["reasons"].append("native_controller_probe_failed:" + name)
        check()
        backend.reset(reset)
        final = advance(reset, 20)
        result["final_reset"] = {"observed": final,
            "passed": max(abs(final[name] - value) for name, value in reset.items()) <= RESET_TOLERANCE}
        if not result["final_reset"]["passed"]:
            result["reasons"].append("native_controller_final_reset_error")
        result["status"] = "passed" if not result["reasons"] else "failed"
    except Exception as exc:  # Diagnostic failure cannot discard a verified import.
        result["reasons"].append(type(exc).__name__ + ":" + str(exc)[:240])
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                result["status"] = "failed"
                result["reasons"].append("native_controller_cleanup_failed:" + type(exc).__name__)
    result.update(physics_ticks=ticks, elapsed_wall_seconds=max(0., now() - start))
    result = sealed(result, "smoke_digest")
    write_once(output / FILENAME, result)
    return result


def run_bounded_smoke_process(body: Path, *, wall_seconds: float, run=subprocess.run) -> dict:
    """Hard process timeout includes SDK startup and cleanup; imported bytes already exist."""
    seconds = min(MAX_WALL_SECONDS, max(0., wall_seconds))
    result = {"schema_version": "embodiment_native_controller_process.v1", "status": "failed",
              "maximum_wall_seconds": seconds, "timeout_includes_sdk_startup": True,
              "candidate_policy_queried": False, "task_executed": False, "physical_success_proven": False,
              "reasons": [], "smoke_digest": None}
    if seconds < 1:
        result["reasons"].append("native_controller_process_budget_exhausted")
    else:
        log = body / "native-controller-smoke.log"
        try:
            with log.open("wb") as stream:
                process = run([sys.executable, "-m", __name__, "--body", str(body), "--wall-seconds", str(seconds)],
                              stdout=stream, stderr=subprocess.STDOUT, timeout=seconds, check=False)
            if process.returncode != 0:
                result["reasons"].append("native_controller_process_failed:" + str(process.returncode))
            else:
                native = read(body / FILENAME)
                if (native["smoke_digest"] != canonical_digest(native, digest_field="smoke_digest")
                        or native["binding_digest"] != canonical_digest(read(body / "binding-inputs.json"))
                        or native["candidate_policy_queried"] is not False or native["task_executed"] is not False):
                    raise ValueError("native_controller_process_result_invalid")
                result.update(status=native["status"], smoke_digest=native["smoke_digest"], reasons=native["reasons"])
        except subprocess.TimeoutExpired:
            result["reasons"].append("native_controller_process_timeout")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result["reasons"].append("native_controller_process_result_unavailable:" + type(exc).__name__)
        if log.is_file() and log.stat().st_size > 1024**2:
            with log.open("rb") as stream:
                stream.seek(-1024**2, os.SEEK_END)
                tail = stream.read()
            log.write_bytes(tail)
    result = sealed(result, "process_digest")
    write_once(body / "native-controller-process.v1.json", result)
    return result


def main(argv=None):
    from .deployment_embodiment_isaac_launch import launch_embodiment_isaaclab
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    start = time.monotonic()
    body = args.body.resolve()
    binding = read(body / "binding-inputs.json")
    app, _launch = launch_embodiment_isaaclab(Path(os.environ["BLUEPRINT_EMBODIMENT_PROVISIONING_RECEIPT"]),
        output=body / "controller-launch")
    try:
        run_native_controller_smoke(binding, body / "prepared" / binding["model_record"]["source_import_provenance"]["model"]["path"],
            body, configuration_dir=body / "configuration",
            wall_budget_seconds=max(0., args.wall_seconds - (time.monotonic() - start)))
    finally:
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
