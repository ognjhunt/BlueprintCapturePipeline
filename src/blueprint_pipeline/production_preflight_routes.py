"""Maintained access boundaries for a scoped, shadow beta rigid preflight.

Static imports discover candidates; only this explicit route contract confirms
required access. This is plumbing coverage, never runtime qualification.
"""

from __future__ import annotations

from collections.abc import Mapping

ROOT = "/var/lib/blueprint/pipeline-control-plane"
BETA_RIGID_ENTRIES = {
    "blueprint-pipeline-intake.service": "blueprint_pipeline.live_pipeline_intake_service",
    "blueprint-pubsub-handoff-listener.service": "blueprint_pipeline.task_evaluation_scene_retirement_supervisor",
    "blueprint-task-evaluation-scene-progression.service": "blueprint_pipeline.task_evaluation_scene_retirement_supervisor",
    "blueprint-capture-reconstruction-dispatcher.service": "blueprint_pipeline.capture_reconstruction_launch_dispatcher",
    "blueprint-team-scene-packet.service": "blueprint_pipeline.team_scene_packet_worker",
    "blueprint-rigid-robot-free-probe-runner.service": "blueprint_pipeline.rigid_robot_free_probe_runner",
    "blueprint-beta-policy-screening.service": "blueprint_pipeline.beta_policy_screening_worker",
}
BETA_RIGID_UNITS = tuple(BETA_RIGID_ENTRIES)
_WRAPPED_WORKERS = {
    "blueprint-pubsub-handoff-listener.service": "blueprint_pipeline.pubsub_handoff_listener",
    "blueprint-task-evaluation-scene-progression.service": "blueprint_pipeline.task_evaluation_scene_progression",
}
_PATHS = {
    "blueprint-pubsub-handoff-listener.service": (
        ("BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT", "/var/lib/blueprint/pubsub-handoffs", "write"),
    ),
    "blueprint-task-evaluation-scene-progression.service": (
        ("", "/etc/blueprint/task-evaluation-scene-progression.json", "read"),
    ),
    "blueprint-capture-reconstruction-dispatcher.service": (
        ("BLUEPRINT_CAPTURE_RECONSTRUCTION_QUEUE_ROOT", f"{ROOT}/capture-reconstruction-queue", "write"),
        ("BLUEPRINT_CAPTURE_RECONSTRUCTION_STATE_ROOT", f"{ROOT}/capture-reconstruction-runs", "write"),
        ("BLUEPRINT_CAPTURE_RECONSTRUCTION_DERIVED_ROOT", f"{ROOT}/capture-reconstruction-derived", "write"),
        ("BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT", "/var/lib/blueprint/pubsub-handoffs", "read"),
    ),
    "blueprint-team-scene-packet.service": (
        ("BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT", f"{ROOT}/team-scene-packets", "write"),
    ),
    "blueprint-rigid-robot-free-probe-runner.service": (
        ("BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE", f"{ROOT}/rigid-robot-free-probes", "write"),
    ),
    "blueprint-beta-policy-screening.service": (
        ("BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT", f"{ROOT}/beta-policy-screening", "write"),
        ("BLUEPRINT_TEAM_SCENE_PACKET_QUEUE_ROOT", f"{ROOT}/team-scene-packets", "read"),
    ),
}


def access_contract(unit: str, entry: str, environment: Mapping[str, str], *, worker: str | None = None) -> tuple[dict, list[dict]]:
    expected = BETA_RIGID_ENTRIES.get(unit)
    if expected is None or expected != entry:
        return {"status": "unknown", "route": "beta_rigid", "reason": "entrypoint_unrecognized"}, []
    if unit in _WRAPPED_WORKERS and worker != _WRAPPED_WORKERS[unit]:
        return {"status": "unknown", "route": "beta_rigid", "reason": "wrapped_worker_unrecognized"}, []
    rows = []
    for name, default, access in _PATHS.get(unit, ()):
        value = environment.get(name) or default
        rows.append({"path": value, "access": access, "env_name": name,
                     "basis": "maintained_beta_rigid_boundary_v1"})
    return {"status": "selected_route_contract", "route": "beta_rigid",
            "boundary_version": 1, "required_path_count": len(rows),
            "scope": "maintained_path_requirements_only",
            "proof_boundary": "plumbing_only_not_runtime_qualification"}, rows
