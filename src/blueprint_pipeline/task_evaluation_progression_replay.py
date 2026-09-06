"""Production look-ahead admission using saved parent/child and consumer contracts."""
from __future__ import annotations

import os
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_launch_preparation_queue import write_launch_preparation_record_exclusive


def replay_progression_admission(*, result_path, queue_root, replay_root,
                                child_queue_root=None, input_root=None, approved_roots=None):
    """Retain a digest-bound no-allocation report before activation can advance.

    Child replay uses the real worker admission and retained-output validators;
    it never reruns a model or paid stage. The parent executes its real worker
    against a scratch queue, stopping at the explicit rendering boundary.
    Finally both actual downstream consumers reopen the original parent result.
    """
    from . import task_evaluation_stage_replay as replay
    from .task_evaluation_sam31_preparation_execution import _validated_job, _retained_result
    from .task_evaluation_scene_configuration_sam31_preparation_driver import CHILD_QUEUE_ENV
    result_path, queue_root, replay_root = map(Path, (result_path, queue_root, replay_root))
    result = replay._read(result_path)
    located = replay.locate_parent(queue_root, result["preparation_id"])
    envelope = replay._read(located.envelope_path)
    child_queue = Path(child_queue_root or os.environ.get(CHILD_QUEUE_ENV, str(replay.DEFAULT_QUEUE_ROOT)))
    inputs = Path(input_root or os.environ.get("BLUEPRINT_TASK_EVALUATION_LAUNCH_PREPARATION_INPUT_ROOT", str(replay.DEFAULT_INPUT_ROOT)))
    roots = tuple(Path(p) for p in (approved_roots or replay.DEFAULT_APPROVED_ROOTS))
    report = {"schema_version": "task_evaluation_progression_replay.v1",
              "source_commit": result.get("source_commit"),
              "preparation_result_digest": result.get("result_digest"),
              "preparation_result_sha256": replay._sha(result_path),
              "parent_envelope_sha256": replay._sha(located.envelope_path),
              "child_admission": [], "provider_mutation_performed": False,
              "paid_execution_requested": False, "blockers": []}
    report["consumer_code_sha256"] = {
        name: replay._sha(Path(__file__).with_name(name + ".py")) for name in (
            "task_evaluation_progression_replay", "task_evaluation_stage_replay",
            "task_evaluation_sam31_preparation_execution",
            "task_evaluation_scene_configuration_sam31_preparation_driver",
            "task_evaluation_scene_configuration_activation_automation",
            "task_evaluation_configured_controls_continuation_provisioning")}
    jobs = []
    for state in replay.JOB_STATES:
        for path in sorted((child_queue / state).glob("*.json")):
            job = replay._read(path)
            if job.get("parent_request_digest") == envelope["request_digest"]:
                jobs.append((path, job))
    for path, job in jobs:
        row = {"child_id": job["child_id"], "job_sha256": replay._sha(path)}
        try:
            actual_inputs = replay.discover_input_root(job) or inputs
            _validated_job(job, parent_queue=queue_root, input_root=actual_inputs,
                           source_commit=result["source_commit"], approved_roots=roots)
            saved_path = child_queue / "results" / path.name
            saved = _retained_result(saved_path, job, roots)
            if saved["status"] != "completed":
                raise ValueError("lookahead_child_not_completed")
            row.update(status="accepted", result_sha256=replay._sha(saved_path))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status="refused", blocker=str(exc)[:700])
            report["blockers"].append("child:" + job["child_id"] + ":" + str(exc)[:500])
        report["child_admission"].append(row)
    if jobs:
        try:
            parent = replay.replay_parent(parent_queue_root=queue_root,
                preparation_id=result["preparation_id"], child_queue_root=child_queue,
                input_root=inputs, replay_root=replay_root,
                allowed_uri_prefixes=replay.envelope_uri_prefixes(envelope))
            report["parent_replay"] = parent
            if not parent.get("sam31_ready") and parent.get("status") != "queued_for_production_scene_configuration":
                report["blockers"].append("parent_replay_refused")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["blockers"].append("parent_replay:" + str(exc)[:500])
    else:
        report["parent_replay"] = {"status": "no_sam31_children", "sam31_ready": False}
    report["next_consumer_admission"] = replay.replay_next_consumers(result_path=result_path, queue_root=queue_root)
    for row in report["next_consumer_admission"]:
        if row["status"] != "accepted":
            report["blockers"].append(row["consumer"] + ":" + row.get("blocker", "refused"))
    report["status"] = "accepted" if not report["blockers"] else "blocked"
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    replay_root.mkdir(parents=True, exist_ok=True)
    output = replay_root / (report["report_digest"].removeprefix("sha256:") + ".json")
    try:
        write_launch_preparation_record_exclusive(output, report)
    except FileExistsError:
        if replay._read(output) != report:
            raise ValueError("lookahead_immutable_report_conflict")
    return {**report, "report_path": str(output)}
