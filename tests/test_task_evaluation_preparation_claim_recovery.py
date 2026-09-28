"""An interrupted ``processing/`` claim is terminalized only after ownership is proven.

Production shape (2026-09-12, scene 840938): the controller tick holding the
claim was SIGTERMed by a restart; the queue's only work sat in ``processing/``,
every later tick reported ``processed_count: 0``, and the scene progression
refused a release successor with ``previous_release_attempt_not_terminal``.
"""
from __future__ import annotations

import hashlib
import json
import os
import pwd
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_launch_preparation_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_launch_preparation_queue import (
    launch_preparation_status,
    stage_launch_preparation_request,
)
from blueprint_pipeline.task_evaluation_preparation_claim_recovery import (
    INTERRUPTED_CLAIM_BLOCKER,
    RESULT_SCHEMA_VERSION,
    WORKER_MODULES,
    PreparationClaimRecoveryError,
    live_worker_processes,
    main,
    recover_interrupted_preparation_claim,
)
from blueprint_pipeline.task_evaluation_scene_progression import _queue
from tests.test_task_evaluation_launch_preparation_contract import request

SERVICE_ACCOUNT = pwd.getpwuid(os.geteuid()).pw_name
RECOVERING = "b" * 40


def _interrupted_claim(tmp_path: Path) -> tuple[dict, Path, str, Path]:
    """Stage a request and simulate a worker that claimed it and then died."""
    value = request()
    queue = tmp_path / "state" / "owned-preparations"
    intake = stage_launch_preparation_request(value=value, queue_root=queue, submitted_by="blueprint-webapp")
    name = Path(intake["queue_path"]).name
    os.replace(queue / "pending" / name, queue / "processing" / name)
    child_queue = tmp_path / "state" / "sam31-preparation-executions"
    for state in ("pending", "processing", "started", "waiting_external", "wake-pending", "completed", "failed"):
        (child_queue / state).mkdir(parents=True)
    return value, queue, name, child_queue


def _after(queue: Path, name: str, seconds: float = 5.0) -> list[dict]:
    started = (queue / "processing" / name).stat().st_ctime + seconds
    return [{"pid": 4242, "module": WORKER_MODULES[0], "started_at_epoch": started}]


def _before(queue: Path, name: str, seconds: float = 5.0) -> list[dict]:
    started = (queue / "processing" / name).stat().st_ctime - seconds
    return [{"pid": 4071318, "module": WORKER_MODULES[0], "started_at_epoch": started}]


def test_result_schema_matches_the_worker_contract() -> None:
    assert RESULT_SCHEMA_VERSION == worker.RESULT_SCHEMA_VERSION


def test_orphaned_claim_becomes_a_blocked_result_the_progression_accepts(tmp_path) -> None:
    value, queue, name, child_queue = _interrupted_claim(tmp_path)
    original = (queue / "processing" / name).read_bytes()
    receipt = recover_interrupted_preparation_claim(
        queue_root=queue, envelope_name=name, recovering_source_commit=RECOVERING,
        child_queue_root=child_queue, service_account=SERVICE_ACCOUNT, previous_owner_pid=2**22 - 1,
        apply=True, live_workers=lambda: _after(queue, name), now=1_789_221_000.0)
    assert receipt["status"] == "recovered" and receipt["applied"] is True
    assert receipt["provider_mutation_performed"] is False and receipt["paid_execution_requested"] is False
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    # Envelope bytes are preserved exactly; only its state changed.
    assert not (queue / "processing" / name).exists()
    assert (queue / "blocked" / name).read_bytes() == original
    assert receipt["result"]["claim_recovery"]["envelope_sha256"] == "sha256:" + hashlib.sha256(original).hexdigest()
    # The sealed result is the worker's own document shape, bound to the claim's release.
    sealed = json.loads((queue / "results" / name).read_text())
    assert sealed == receipt["result"]
    assert sealed["schema_version"] == worker.RESULT_SCHEMA_VERSION
    assert sealed["status"] == "blocked" and sealed["blockers"] == [INTERRUPTED_CLAIM_BLOCKER]
    assert sealed["source_commit"] == value["expected_production_commit"]
    assert sealed["claim_recovery"]["recovered_by_source_commit"] == RECOVERING
    assert sealed["claim_recovery"]["request_re_executed"] is False
    assert sealed["result_digest"] == canonical_digest(sealed, digest_field="result_digest")
    # The consumers that were stuck now observe a terminal preparation.
    envelope = json.loads((queue / "blocked" / name).read_text())
    observed = _queue(envelope["request"], queue)
    assert observed["status"] == "blocked"
    assert observed["result"]["blockers"] == [INTERRUPTED_CLAIM_BLOCKER]
    status = launch_preparation_status(preparation_id=value["preparation_id"], queue_root=queue)
    assert status["status"] == "blocked" and status["blockers"] == [INTERRUPTED_CLAIM_BLOCKER]
    # A later worker pass has nothing to claim and never re-executes the request.
    run = worker.process_launch_preparation_queue(
        queue_root=queue, input_root=tmp_path / "inputs", allowed_uri_prefixes=["s3://blueprint-production-inputs/"],
        service_account=SERVICE_ACCOUNT, source_commit=RECOVERING)
    assert run["status"] == "idle" and run["processed_count"] == 0
    assert (queue / "blocked" / name).read_bytes() == original


def test_dry_run_proves_ownership_without_mutating_the_queue(tmp_path) -> None:
    value, queue, name, child_queue = _interrupted_claim(tmp_path)
    receipt = recover_interrupted_preparation_claim(
        queue_root=queue, envelope_name=name, recovering_source_commit=RECOVERING,
        child_queue_root=child_queue, live_workers=lambda: _after(queue, name))
    assert receipt["status"] == "recoverable" and receipt["applied"] is False
    assert (queue / "processing" / name).exists()
    assert not (queue / "results" / name).exists() and not (queue / "blocked" / name).exists()


def test_refuses_while_a_worker_that_predates_the_claim_is_alive(tmp_path) -> None:
    value, queue, name, child_queue = _interrupted_claim(tmp_path)
    with pytest.raises(PreparationClaimRecoveryError, match="^claim_owner_may_be_alive:4071318$"):
        recover_interrupted_preparation_claim(
            queue_root=queue, envelope_name=name, recovering_source_commit=RECOVERING,
            child_queue_root=child_queue, apply=True, live_workers=lambda: _before(queue, name))
    assert (queue / "processing" / name).exists() and not (queue / "results" / name).exists()


def test_refuses_when_the_named_previous_owner_is_still_alive(tmp_path) -> None:
    value, queue, name, child_queue = _interrupted_claim(tmp_path)
    with pytest.raises(PreparationClaimRecoveryError, match=f"^previous_owner_alive:{os.getpid()}$"):
        recover_interrupted_preparation_claim(
            queue_root=queue, envelope_name=name, recovering_source_commit=RECOVERING,
            previous_owner_pid=os.getpid(), apply=True, live_workers=lambda: [])
    assert (queue / "processing" / name).exists()


def test_refuses_when_a_live_child_job_references_the_request(tmp_path) -> None:
    value, queue, name, child_queue = _interrupted_claim(tmp_path)
    digest = json.loads((queue / "processing" / name).read_text())["request_digest"]
    (child_queue / "waiting_external" / "sam31-abc.json").write_text(
        json.dumps({"parent_request_digest": digest, "phase": "deleted_layer_projection"}))
    with pytest.raises(PreparationClaimRecoveryError,
                       match="^claim_child_live:waiting_external/sam31-abc.json$"):
        recover_interrupted_preparation_claim(
            queue_root=queue, envelope_name=name, recovering_source_commit=RECOVERING,
            child_queue_root=child_queue, apply=True, live_workers=lambda: [])
    assert (queue / "processing" / name).exists()
    # A terminal child is history, not a live owner.
    os.replace(child_queue / "waiting_external" / "sam31-abc.json", child_queue / "completed" / "sam31-abc.json")
    receipt = recover_interrupted_preparation_claim(
        queue_root=queue, envelope_name=name, recovering_source_commit=RECOVERING,
        child_queue_root=child_queue, live_workers=lambda: [])
    assert receipt["status"] == "recoverable"


def test_refuses_entries_that_are_not_an_unfinished_processing_claim(tmp_path) -> None:
    value, queue, name, child_queue = _interrupted_claim(tmp_path)
    os.replace(queue / "processing" / name, queue / "pending" / name)
    with pytest.raises(PreparationClaimRecoveryError, match="^claim_not_in_processing$"):
        recover_interrupted_preparation_claim(
            queue_root=queue, envelope_name=name, recovering_source_commit=RECOVERING, live_workers=lambda: [])
    os.replace(queue / "pending" / name, queue / "processing" / name)
    (queue / "results").mkdir(exist_ok=True)
    (queue / "results" / name).write_text("{}")
    with pytest.raises(PreparationClaimRecoveryError, match="^claim_result_already_exists$"):
        recover_interrupted_preparation_claim(
            queue_root=queue, envelope_name=name, recovering_source_commit=RECOVERING, live_workers=lambda: [])
    (queue / "results" / name).unlink()
    tampered = json.loads((queue / "processing" / name).read_text())
    tampered["request"]["run_id"] = "another-run"
    (queue / "processing" / name).unlink()  # sealed 0440; a tamper replaces the inode
    (queue / "processing" / name).write_text(json.dumps(tampered))
    with pytest.raises(PreparationClaimRecoveryError, match="^claim_envelope_invalid$"):
        recover_interrupted_preparation_claim(
            queue_root=queue, envelope_name=name, recovering_source_commit=RECOVERING, live_workers=lambda: [])


def _fake_proc(root: Path, boot_epoch: int, processes: dict[int, tuple[list[str], int]]) -> None:
    root.mkdir(exist_ok=True)
    (root / "stat").write_text(f"cpu  1 2 3 4\nbtime {boot_epoch}\nprocesses 10\n")
    for pid, (arguments, start_ticks) in processes.items():
        (root / str(pid)).mkdir()
        (root / str(pid) / "cmdline").write_bytes(b"\0".join(a.encode() for a in arguments) + b"\0")
        rest = " ".join(["S", "1"] + ["0"] * 17 + [str(start_ticks)] + ["0"] * 20)
        (root / str(pid) / "stat").write_text(f"{pid} (py thon) {rest}\n")


def test_live_worker_processes_reads_start_times_from_the_process_table(tmp_path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    _fake_proc(proc, 1_000_000, {
        4126512: (["/opt/python", "-m", "blueprint_pipeline.task_evaluation_scene_progression", "--config", "x"], 500),
        4126600: (["/bin/bash", "-lc", "exec env PYTHONPATH=src python -m "
                   "blueprint_pipeline.task_evaluation_launch_preparation_worker --queue-root q"], 700),
        4099670: (["/opt/python", "-m", "blueprint_pipeline.agent_execution.production"], 100),
        4099363: (["/opt/python", "-m", "blueprint_pipeline.live_pipeline_intake_service"], 100),
    })
    hz = os.sysconf("SC_CLK_TCK")
    assert live_worker_processes(proc) == [
        {"pid": 4126512, "module": WORKER_MODULES[0], "started_at_epoch": 1_000_000 + 500 / hz},
        {"pid": 4126600, "module": WORKER_MODULES[2], "started_at_epoch": 1_000_000 + 700 / hz},
    ]
    with pytest.raises(PreparationClaimRecoveryError, match="^process_table_unavailable$"):
        live_worker_processes(tmp_path / "missing-proc")


def test_cli_refuses_closed_and_applies_only_with_the_flag(tmp_path, capsys) -> None:
    value, queue, name, child_queue = _interrupted_claim(tmp_path)
    proc = tmp_path / "proc"
    proc.mkdir()
    claimed_at = (queue / "processing" / name).stat().st_ctime
    hz = os.sysconf("SC_CLK_TCK")
    boot = int(claimed_at) - 10_000
    _fake_proc(proc, boot, {
        7: (["python", "-m", "blueprint_pipeline.task_evaluation_scene_progression"], 100),  # predates the claim
    })
    common = ["--queue-root", str(queue), "--envelope-name", name, "--child-queue-root", str(child_queue),
              "--service-account", SERVICE_ACCOUNT, "--recovering-source-commit", RECOVERING,
              "--proc-root", str(proc)]
    assert main(common + ["--apply"]) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["status"] == "refused" and refused["blockers"] == ["claim_owner_may_be_alive:7"]
    assert (queue / "processing" / name).exists()
    # The only live worker started after the claim: dry-run reports, --apply seals.
    _fake_proc(tmp_path / "proc2", boot, {
        8: (["python", "-m", "blueprint_pipeline.task_evaluation_scene_progression"], int((claimed_at - boot + 30) * hz)),
    })
    common[-1] = str(tmp_path / "proc2")
    assert main(common) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "recoverable"
    assert (queue / "processing" / name).exists()
    receipt_out = tmp_path / "receipt.json"
    assert main(common + ["--apply", "--receipt-out", str(receipt_out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "recovered" and json.loads(receipt_out.read_text()) == printed
    assert (queue / "blocked" / name).exists() and (queue / "results" / name).exists()
    assert main(common + ["--apply"]) == 2
    assert json.loads(capsys.readouterr().out)["blockers"] == ["claim_not_in_processing"]
