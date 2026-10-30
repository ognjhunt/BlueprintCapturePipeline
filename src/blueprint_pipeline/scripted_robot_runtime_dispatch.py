"""Resolve a sealed scripted profile while preserving the original controller inventory."""
from __future__ import annotations


def runtime_for_policy(policy):
    from . import scripted_robot_kuka_robotiq_runtime as kuka
    if policy.get("policy_id") == kuka.PROFILE_ID:
        return kuka
    from . import scripted_robot_yaskawa_robotiq_runtime as yaskawa
    if policy.get("policy_id") == yaskawa.PROFILE_ID:
        return yaskawa
    from . import scripted_robot_baseline_runtime as original
    if policy.get("policy_id") == original.PROFILE_ID:
        return original
    from . import scripted_robot_robotiq_runtime as robotiq
    if policy.get("policy_id") == robotiq.PROFILE_ID:
        return robotiq
    from . import scripted_robot_industrial_robotiq_runtime as industrial
    if policy.get("policy_id") == industrial.PROFILE_ID:
        return industrial
    raise ValueError("scripted_baseline_profile_unsupported")


def runtime_for_manifest(manifest):
    from . import scripted_robot_kuka_robotiq_runtime as kuka
    if manifest.get("embodiment_id") in kuka.BODIES:
        return kuka
    from . import scripted_robot_yaskawa_robotiq_runtime as yaskawa
    if manifest.get("embodiment_id") in yaskawa.BODIES:
        return yaskawa
    from . import scripted_robot_industrial_robotiq_runtime as industrial
    if manifest.get("embodiment_id") in industrial.BODIES:
        return industrial
    from . import scripted_robot_robotiq_runtime as robotiq
    if manifest.get("embodiment_id") in robotiq.BODIES:
        return robotiq
    from . import scripted_robot_baseline_runtime as original
    return original


def build_policy_spec(manifest):
    return runtime_for_manifest(manifest).build_policy_spec(manifest)


def client_for_spec(spec):
    runtime = runtime_for_policy(spec["policy_spec"])
    runtime.validate_execution_spec(spec)
    from .scripted_robot_baseline import ScriptedRobotBaseline
    cls = getattr(runtime, "ScriptedRobotiqBaseline", ScriptedRobotBaseline)
    return cls(plan=spec["scripted_task_plan"], policy_id=spec["candidate_id"],
               policy_spec=spec["policy_spec"], manifest=spec["team_embodiment"]["manifest"])
