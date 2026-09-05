"""Execute closed, immutable SAM preparation phases outside the no-spend parent."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .fail_closed_blocker_explainer import annotate_blocker
from .decision_evidence_contracts import canonical_digest
from .public_scene_host_input_intake import _verified_checkout_head
from .task_evaluation_launch_preparation_contract import (
    launch_preparation_request_digest, validate_launch_preparation_request,
)
from .task_evaluation_launch_preparation_queue import (
    QUEUE_STATES,
)
from .task_evaluation_release_reference_lock import release_reference_lock
from .task_evaluation_sam31_preparation_queue import (
    SAM31_EXECUTION_ROOT, WAITING_STATE, load_progress, stage_resume_signal, verify_evidence_reference,
)
from .task_evaluation_scene_construction_recipe import validate_scene_construction_recipe
from .task_evaluation_sam31_phase_queue import (
    JOB_SCHEMA, PHASES, STATES as STATES,
    Sam31PhaseExecutionError,
    _ensure, _read, _ref, _require, _write,
    enqueue_sam31_phase as enqueue_sam31_phase,
)

RESULT_SCHEMA = "task_evaluation_sam31_preparation_execution_result.v1"
PROGRESS_SCHEMA = "task_evaluation_sam31_preparation_execution_progress.v1"
DEFAULT_QUEUE = Path("/var/lib/blueprint/pipeline-control-plane/sam31-preparation-executions")
DEFAULT_PARENT_QUEUE = Path("/var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-preparations")
DEFAULT_INPUT_ROOT = Path("/var/lib/blueprint/task-evaluation-inputs/prepared-references")
DEFAULT_EXECUTION_ROOT = SAM31_EXECUTION_ROOT
DEFAULT_APPROVED_ROOTS = (Path("/var/lib/blueprint/task-evaluation-inputs"),
                          Path("/var/lib/blueprint/pipeline-control-plane"))


def _collect_preparation_references(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect typed immutable references without importing the hot launch worker."""

    references: list[dict[str, Any]] = []

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            if set(node) == {"uri", "digest", "size_bytes"}:
                references.append(
                    {
                        "contract_path": ".".join(path),
                        "uri": str(node["uri"]),
                        "digest": str(node["digest"]),
                        "size_bytes": int(node["size_bytes"]),
                    }
                )
                return
            for key, child in node.items():
                visit(child, (*path, str(key)))
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for index, child in enumerate(node):
                visit(child, (*path, str(index)))

    visit(value, ())
    identities: dict[str, tuple[str, int]] = {}
    for reference in references:
        identity = (reference["digest"], reference["size_bytes"])
        prior = identities.setdefault(reference["uri"], identity)
        _require(prior == identity, "reference_uri_identity_conflict")
    return references


def _file_record(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "sha256": "sha256:" + digest.hexdigest(), "size_bytes": size}


def _parent(job: dict, root: Path) -> tuple[dict, str, Path]:
    digest = job.get("parent_request_digest")
    identifier = job.get("parent_preparation_id")
    _require(isinstance(identifier, str) and identifier and "/" not in identifier
             and isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None,
             "parent_identity_invalid")
    filename = f"{identifier}-{digest.removeprefix('sha256:')}.json"
    matches = [(state, root / state / filename) for state in QUEUE_STATES if (root / state / filename).exists()]
    _require(len(matches) == 1, "parent_identity_ambiguous")
    state, path = matches[0]
    envelope = _read(path)
    request = validate_launch_preparation_request(envelope["request"])
    _require(envelope.get("envelope_digest") == canonical_digest(envelope, digest_field="envelope_digest")
             and envelope.get("request_digest") == digest
             and launch_preparation_request_digest(request) == digest
             and request["preparation_id"] == identifier, "parent_envelope_invalid")
    return request, state, path


def _validated_job(job: dict, *, parent_queue: Path, input_root: Path,
                   source_commit: str, approved_roots: Sequence[Path]) -> tuple[dict, dict]:
    _require(set(job) == {"schema_version", "child_id", "parent_preparation_id",
             "parent_request_digest", "plan_digest", "phase", "inputs_digest",
             "expected_source_commit", "plan_ref", "inputs", "job_digest"}
             and job.get("schema_version") == JOB_SCHEMA and job.get("phase") in PHASES
             and job.get("job_digest") == canonical_digest(job, digest_field="job_digest"),
             "job_contract_invalid")
    identities = {name: {key: value[key] for key in ("sha256", "size_bytes")}
                  for name, value in job["inputs"].items()}
    key = {name: job[name] for name in ("parent_request_digest", "plan_digest", "phase", "inputs_digest")}
    _require(job["inputs_digest"] == canonical_digest(identities)
             and job["child_id"] == "sam31-" + canonical_digest(key).removeprefix("sha256:"),
             "job_identity_invalid")
    request, _, _ = _parent(job, parent_queue)
    _require(request["expected_production_commit"] == source_commit == job["expected_source_commit"],
             "source_commit_mismatch")
    plan_ref = _ref(job["plan_ref"])
    plan_path = verify_evidence_reference(plan_ref, approved_roots)
    _require(plan_ref["sha256"] == job["plan_digest"], "plan_identity_mismatch")
    bound = [ref for ref in _collect_preparation_references(request)
             if ref["contract_path"].startswith("runtime.mounts.")
             and ref["digest"] == plan_ref["sha256"] and ref["size_bytes"] == plan_ref["size_bytes"]]
    _require(bool(bound), "plan_not_bound_to_parent")
    cas = input_root / "content-addressed" / "sha256"
    def read_cas(ref):
        path = cas / ref["digest"].removeprefix("sha256:")
        verify_evidence_reference({"path": str(path), "sha256": ref["digest"],
                                  "size_bytes": ref["size_bytes"]}, (input_root,))
        return _read(path)
    recipe = validate_scene_construction_recipe(read_cas(request["construction"]["recipe"]))
    stage_one = read_cas(recipe["stage_sequence"][0]["configuration"])
    declared = stage_one.get("sam31_preparation_plan", {})
    _require(stage_one.get("sam31_review_kind") == "ai"
             and declared.get("digest") == plan_ref["sha256"]
             and declared.get("size_bytes") == plan_ref["size_bytes"]
             and any(ref["uri"] == declared.get("uri") for ref in bound), "plan_stage_binding_invalid")
    for value in job["inputs"].values():
        verify_evidence_reference(_ref(value), approved_roots)
    return request, _read(plan_path)


def _retained_result(path: Path, job: dict, roots: Sequence[Path]) -> dict:
    value = _read(path)
    _require(value.get("schema_version") == RESULT_SCHEMA and value.get("job_digest") == job["job_digest"]
             and value.get("child_id") == job["child_id"] and value.get("status") in {"completed", "failed"}
             and value.get("result_digest") == canonical_digest(value, digest_field="result_digest"),
             "retained_result_invalid")
    _require(isinstance(value.get("artifacts"), dict), "retained_artifacts_invalid")
    for ref in value["artifacts"].values():
        verify_evidence_reference(ref, roots)
    return value


def _wake_parent(root: Path, job: dict, parent_queue: Path, roots: Sequence[Path]) -> bool:
    result_path = root / "results" / f"{job['child_id']}.json"
    _retained_result(result_path, job, roots)
    completed_marker = root / "wake-completed" / f"{job['child_id']}.json"
    if completed_marker.exists():
        _require(_read(completed_marker).get("job_digest") == job["job_digest"], "wake_identity_conflict")
        (root / "wake-pending" / completed_marker.name).unlink(missing_ok=True)
        return False
    _, state, parent_path = _parent(job, parent_queue)
    marker = root / "wake-pending" / f"{job['child_id']}.json"
    if state in {"blocked", "completed", "materialized"}:
        _write(root / "wake-completed" / marker.name, {"status": "parent_terminal", "job_digest": job["job_digest"]})
        marker.unlink(missing_ok=True)
        return True
    progress = load_progress(parent_queue, parent_path.name, job["parent_request_digest"])
    if state != WAITING_STATE or progress is None or progress.get("status") != "waiting_for_child":
        return False
    signal = stage_resume_signal(
        queue_root=parent_queue, preparation_id=job["parent_preparation_id"],
        request_digest=job["parent_request_digest"], progress_digest=progress["progress_digest"],
        source_commit=job["expected_source_commit"], kind="child_result",
        evidence_ref=_file_record(result_path), approved_roots=roots,
    )
    _write(root / "wake-completed" / marker.name, {"status": "signaled", "job_digest": job["job_digest"],
                                                "signal_digest": signal["signal_digest"]})
    marker.unlink(missing_ok=True)
    return True


def process_sam31_phase_queue(
    *, queue_root: str | Path = DEFAULT_QUEUE, parent_queue_root: str | Path = DEFAULT_PARENT_QUEUE,
    preparation_input_root: str | Path = DEFAULT_INPUT_ROOT,
    execution_root: str | Path = DEFAULT_EXECUTION_ROOT, source_commit: str | None = None,
    max_messages: int = 1, phase_executor: Callable[[dict], dict] | None = None,
    approved_roots: Sequence[Path] = DEFAULT_APPROVED_ROOTS,
) -> dict:
    _require(type(max_messages) is int and 1 <= max_messages <= 16, "message_bound_invalid")
    observed_commit = _verified_checkout_head()
    _require(source_commit is None or source_commit == observed_commit, "execution_commit_mismatch")
    root, parent_queue, input_root = Path(queue_root), Path(parent_queue_root), Path(preparation_input_root)
    _ensure(root)
    completed = []
    with release_reference_lock(root / "worker-lock", exclusive=True):
        candidates = [path for state in ("processing", "pending", "waiting_external")
                      for path in sorted((root / state).glob("*.json"))][:max_messages]
        for source in candidates:
            claimed = root / "processing" / source.name
            if source != claimed:
                _require(not claimed.exists(), "claim_conflict")
                os.replace(source, claimed)
            job: dict = {}
            result_path = root / "results" / claimed.name
            try:
                job = _read(claimed)
                request, plan = _validated_job(job, parent_queue=parent_queue, input_root=input_root,
                                              source_commit=observed_commit, approved_roots=approved_roots)
                if result_path.exists():
                    result = _retained_result(result_path, job, approved_roots)
                else:
                    output = Path(execution_root) / job["parent_request_digest"].removeprefix("sha256:") / job["child_id"]
                    _require(not any(p.is_symlink() for p in (output, *output.parents))
                             and any(output.resolve().is_relative_to(p.resolve()) for p in approved_roots),
                             "execution_path_invalid")
                    output.mkdir(parents=True, exist_ok=True)
                    started = root / "started" / claimed.name
                    resume_only = started.exists()
                    _write(started, {"job_digest": job["job_digest"], "child_id": job["child_id"]})
                    progress_files = sorted((root / "progress" / job["child_id"]).glob("*.json"))
                    previous = _read(progress_files[-1]) if progress_files else None
                    if previous:
                        _require(previous.get("progress_digest") == canonical_digest(previous, digest_field="progress_digest")
                                 and previous.get("job_digest") == job["job_digest"], "progress_invalid")
                        for ref in previous.get("executor_result", {}).get("artifacts", {}).values():
                            verify_evidence_reference(_ref(ref), approved_roots)
                    if phase_executor is None:
                        from .task_evaluation_sam31_preparation_stages import execute_stage
                        executor = execute_stage
                    else:
                        executor = phase_executor
                    outcome = executor({**job, "request": request, "plan": plan,
                                        "queue_root": str(root), "output_root": str(output),
                                        "preparation_input_root": str(input_root),
                                        "resume_only": resume_only, "previous_progress": previous})
                    _require(isinstance(outcome, dict) and outcome.get("status") in {
                        "completed", "waiting_for_external_result", "failed"}, "executor_result_invalid")
                    artifacts = outcome.get("artifacts", {})
                    _require(isinstance(artifacts, dict) and all(
                        isinstance(name, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
                        for name in artifacts), "executor_artifacts_invalid")
                    for artifact in artifacts.values():
                        verify_evidence_reference(_ref(artifact), approved_roots)
                    if outcome["status"] == "waiting_for_external_result":
                        progress = {"schema_version": PROGRESS_SCHEMA, "job_digest": job["job_digest"],
                                    "status": outcome["status"], "executor_result": outcome,
                                    "sequence": len(progress_files) + 1}
                        progress["progress_digest"] = canonical_digest(progress, digest_field="progress_digest")
                        _write(root / "progress" / job["child_id"] / f"{progress['sequence']:06d}.json", progress)
                        os.replace(claimed, root / "waiting_external" / claimed.name)
                        completed.append({"child_id": job["child_id"], "status": outcome["status"]})
                        continue
                    _require(outcome["status"] != "completed" or bool(artifacts), "completed_artifacts_missing")
                    result = {"schema_version": RESULT_SCHEMA, "child_id": job["child_id"],
                              "job_digest": job["job_digest"], "parent_request_digest": job["parent_request_digest"],
                              "plan_digest": job["plan_digest"], "phase": job["phase"],
                              "source_commit": observed_commit, "status": outcome["status"],
                              "artifacts": artifacts, "executor_result": outcome,
                              **({"blocker": ";".join(str(b) for b in (outcome.get("blockers") or ["sam31_stage_failed"]))[:700]}
                                 if outcome["status"] != "completed" else {})}
                    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
                    _write(result_path, result)
            except Exception as exc:
                result = {"schema_version": RESULT_SCHEMA, "child_id": job.get("child_id", claimed.stem),
                          "job_digest": job.get("job_digest"), "parent_request_digest": job.get("parent_request_digest"),
                          "plan_digest": job.get("plan_digest"), "phase": job.get("phase"),
                          "source_commit": observed_commit, "status": "failed", "artifacts": {},
                          "blocker": annotate_blocker(
                              str(exc) if isinstance(exc, (Sam31PhaseExecutionError, ValueError))
                              else f"sam31_phase_execution_failed:{type(exc).__name__}", exc)}
                result["result_digest"] = canonical_digest(result, digest_field="result_digest")
                if result_path.exists():
                    _write(root / "results" / f"{claimed.stem}.conflict-{result['result_digest'][7:]}.json", result)
                else:
                    _write(result_path, result)
            target = "completed" if result["status"] == "completed" else "failed"
            os.replace(claimed, root / target / claimed.name)
            if not (root / "wake-completed" / claimed.name).exists():
                _write(root / "wake-pending" / claimed.name, {"job_digest": job.get("job_digest")})
            completed.append({"child_id": job.get("child_id"), "status": result["status"],
                              "result_path": str(result_path)})
        wakes = []
        for marker in sorted((root / "wake-pending").glob("*.json"))[:max_messages]:
            jobs = [root / state / marker.name for state in ("completed", "failed")
                    if (root / state / marker.name).exists()]
            if len(jobs) != 1:
                continue
            try:
                job = _read(jobs[0])
                _validated_job(job, parent_queue=parent_queue, input_root=input_root,
                               source_commit=observed_commit, approved_roots=approved_roots)
                if _wake_parent(root, job, parent_queue, approved_roots):
                    wakes.append(job["child_id"])
            except (OSError, ValueError):
                continue
    return {"schema_version": "task_evaluation_sam31_preparation_execution_queue_run.v1",
            "status": "processed" if completed or wakes else "idle", "results": completed,
            "parent_wakeups": wakes, "generic_commands_accepted": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", default=str(DEFAULT_QUEUE))
    parser.add_argument("--parent-queue-root", default=str(DEFAULT_PARENT_QUEUE))
    parser.add_argument("--preparation-input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--execution-root", default=str(DEFAULT_EXECUTION_ROOT))
    parser.add_argument("--source-commit")
    parser.add_argument("--max-messages", type=int, default=1)
    args = parser.parse_args()
    result = process_sam31_phase_queue(
        queue_root=args.queue_root, parent_queue_root=args.parent_queue_root,
        preparation_input_root=args.preparation_input_root, execution_root=args.execution_root,
        source_commit=args.source_commit, max_messages=args.max_messages,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
