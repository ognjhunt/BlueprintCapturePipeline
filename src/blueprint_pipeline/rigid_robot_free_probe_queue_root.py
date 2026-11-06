"""Where the rigid robot-free probe queue keeps a capture's entries and attempts, for readers that must not launch.

Split out of ``rigid_robot_free_probe_runner``, which reaches the paid allocator, so withdrawal
deletion can find a withdrawn capture's probe entries and attempts without importing it. The runner
re-exports every name it owns here, so its callers are unchanged.

A capture's entry is ``<queue>/pending/<capture key>.json`` until a pass retires it to
``<queue>/done/<capture key>.json``; each probe request's attempts sit under
``<queue>/jobs/<capture key>-<request digest prefix>/attempts/<NN>``. The capture key is
``team_scene_capture_root.capture_key`` of the capture's resolved root. With a work root, an
attempt's sealed bundle and allocator job directory are under
``<work root>/jobs/<job>/attempts/<NN>``, and the attempt's record names that directory.
"""

from __future__ import annotations

QUEUE_ENV = "BLUEPRINT_RIGID_ROBOT_FREE_PROBE_QUEUE"
DEFAULT_QUEUE = "/var/lib/blueprint/pipeline-control-plane/rigid-robot-free-probes"
ENTRY_DIRECTORIES = ("pending", "done")
JOBS_DIRECTORY = "jobs"
ATTEMPTS_DIRECTORY = "attempts"
# Held by the runner while it advances a job (the allocator it starts included).
JOB_LOCK_FILENAME = "runner.lock"
# The bulk volume a new attempt stages its bundle and allocator job directory under; the unit sets
# deploy/systemd/blueprint-rigid-robot-free-probe-runner.service's value.
WORK_ENV = "BLUEPRINT_RIGID_ROBOT_FREE_PROBE_WORK_ROOT"
DEFAULT_WORK_ROOT = "/mnt/blueprint-work/rigid-robot-free-probe-work"
WORK_DIR_FIELD = "work_dir"
# The runner's records in each attempt directory of the queue.
ATTEMPT_FILENAME = "rigid_robot_free_probe_attempt.json"
INTENT_FILENAME = "allocator_launch_intent.json"
PROCESS_FILENAME = "allocator_process.json"
SPAWN_FAILED_FILENAME = "allocator_spawn_failed.json"
RESULT_FILENAME = "allocator_result.json"
# Named by the money and worker modules, which keep their own definitions (pinned equal to these by
# tests/test_website_capture_withdrawal_team_scene.py): the settlement record the runner writes once
# beside an attempt's records (rigid_robot_free_probe_settlement), the website-grant reservation in
# the attempt's allocator job directory (rigid_robot_free_probe_lane), and the sealed probe input,
# the half's resolved scene plan (rigid_robot_free_probe_worker).
SETTLEMENT_FILENAME = "rigid_robot_free_probe_spend_settlement.json"
SETTLEMENT_SCHEMA = "blueprint.rigid_robot_free_probe_spend_settlement.v1"
FUNDING_FILENAME = "website_preparation_spend_admission.json"
INPUTS_FILENAME = "rigid_robot_free_probe_inputs.v1.json"

__all__ = ["ATTEMPTS_DIRECTORY", "ATTEMPT_FILENAME", "DEFAULT_QUEUE", "DEFAULT_WORK_ROOT", "ENTRY_DIRECTORIES",
           "FUNDING_FILENAME", "INPUTS_FILENAME", "INTENT_FILENAME", "JOBS_DIRECTORY", "JOB_LOCK_FILENAME",
           "PROCESS_FILENAME", "QUEUE_ENV", "RESULT_FILENAME", "SETTLEMENT_FILENAME", "SETTLEMENT_SCHEMA",
           "SPAWN_FAILED_FILENAME", "WORK_DIR_FIELD", "WORK_ENV"]
