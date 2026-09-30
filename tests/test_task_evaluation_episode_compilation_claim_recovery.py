# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_episode_compilation_claim_recovery.py
#   src/blueprint_pipeline/task_evaluation_episode_compilation_remote.py
#   src/blueprint_pipeline/task_evaluation_episode_compilation_worker.py
"""Plan 14 task 4.7: an interrupted episode-compilation claim is recovered, never stranded in processing/."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_episode_compilation_claim_recovery as recovery
from blueprint_pipeline import task_evaluation_episode_compilation_remote as remote
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.remote_cpu_job_contract import job_id_for
from blueprint_pipeline.remote_cpu_job_records import write_remote_cpu_record
from blueprint_pipeline.task_evaluation_episode_compilation_worker import compile_claimed_envelope
from blueprint_pipeline.task_evaluation_launch_preparation_queue import write_launch_preparation_record_exclusive
from tests.remote_cpu_allocator_fakes import remote_cpu_config
from tests.remote_episode_compilation_support import HOST_RECORD, Crash, Host, stage_compile, tree_snapshot

COMMIT = "a" * 40


def _run(host: Host, compiler, *, now: float, mode: str = "host") -> dict:
    return remote.run_no_spend_unit(
        queue_root=host.queue, input_root=host.inputs, output_root=host.outputs, source_commit=COMMIT,
        max_messages=8, jobs_root=host.jobs, environ={remote.EXECUTION_ENV: mode}, episode_compiler=compiler,
        filesystem_root=host.fs, cache_root=host.cache, config=remote_cpu_config(), host_environment=HOST_RECORD,
        now=lambda: now)


def _killed(**kwargs) -> dict:
    """The compiling process dies mid-compile (a deploy restart, SIGTERM, OOM): nothing catches it."""

    (kwargs["output_root"] / "partial.bin").write_bytes(b"half")
    raise Crash("worker_process_terminated")


def _stand_ins(monkeypatch):
    from tests.remote_cpu_worker_stages import install_compile_stand_ins

    return install_compile_stand_ins(monkeypatch.setattr)


def test_orphaned_claim_is_requeued_with_partial_output_set_aside(tmp_path: Path, monkeypatch) -> None:
    host = Host(tmp_path)
    envelope, name = stage_compile(host)
    queued = (host.queue / "pending" / name).read_bytes()
    with pytest.raises(Crash):
        _run(host, _killed, now=1_000.0)
    output = host.outputs / envelope["compilation_id"]
    # The dead run left a bare claim and a partial output that a later compile's exclusive mkdir would trip on.
    assert (host.queue / "processing" / name).is_file() and (output / "partial.bin").is_file()

    run = _run(host, _stand_ins(monkeypatch), now=2_000.0)
    aside = host.outputs / f".{envelope['compilation_id']}.interrupted-2000"
    assert run["recovered_claims"] == [
        {"name": name, "action": "requeued", "interruptions": 1, "set_aside": aside.name}]
    # Set aside, never deleted; the requeued row compiled in the same run, from its own unchanged bytes.
    assert (aside / "partial.bin").read_bytes() == b"half"
    assert (host.queue / "completed" / name).read_bytes() == queued
    assert json.loads((host.queue / "results" / name).read_bytes())["status"] == "compiled_for_production_launch"
    assert output.is_dir() and not (output / "partial.bin").exists()
    record = json.loads(recovery.record_path(host.jobs, name).read_bytes())
    assert record["interruptions"] == 1 and record["history"][0]["set_aside"] == aside.name
    assert record["record_digest"] == canonical_digest(record, digest_field="record_digest")
    # Recovery lives beside the hand-offs, outside the queue: rows stay in the four states.
    assert not recovery.record_path(host.jobs, name).is_relative_to(host.queue)
    # A run with nothing to recover is today's host run, byte for byte: no recovery key appears.
    assert "recovered_claims" not in _run(host, _stand_ins(monkeypatch), now=3_000.0)


def test_third_interruption_blocks_with_a_typed_blocker(tmp_path: Path) -> None:
    host = Host(tmp_path)
    envelope, name = stage_compile(host)
    compilation_id = envelope["compilation_id"]
    for now in (1_000.0, 2_000.0, 3_000.0):  # the first claim, then two requeued claims, each killed
        with pytest.raises(Crash):
            _run(host, _killed, now=now)
    run = _run(host, _killed, now=4_000.0)  # nothing compiles on the third interruption

    assert run["recovered_claims"] == [{"name": name, "action": "blocked", "interruptions": 3,
                                        "set_aside": f".{compilation_id}.interrupted-4000"}]
    assert (host.queue / "blocked" / name).is_file() and not (host.queue / "pending" / name).exists()
    result = json.loads((host.queue / "results" / name).read_bytes())
    assert result["blockers"] == [recovery.INTERRUPTED_CLAIM_BLOCKER]
    assert recovery.INTERRUPTED_CLAIM_BLOCKER == "episode_compilation_claim_interrupted:worker_process_terminated"
    assert result["status"] == "blocked" and result["schema_version"] == "task_evaluation_episode_compilation_result.v1"
    assert result["result_digest"] == canonical_digest(result, digest_field="result_digest")
    assert (result["compilation_id"], result["run_id"], result["team_namespace"]) == (
        compilation_id, envelope["run_id"], envelope["team_namespace"])
    assert result["provider_mutation_performed"] is False and result["paid_execution_requested"] is False
    assert result["claim_recovery"]["interruptions"] == 3 and result["claim_recovery"]["requeued"] == 2
    # Every interrupted attempt's partial output is still there to inspect; none sits at the output path.
    assert sorted(path.name for path in host.outputs.glob(f".{compilation_id}.interrupted-*")) == [
        f".{compilation_id}.interrupted-{epoch}" for epoch in (2000, 3000, 4000)]
    assert not (host.outputs / compilation_id).exists()
    # A sealed row is terminal: the next run leaves it alone.
    assert "recovered_claims" not in _run(host, _killed, now=5_000.0)


def test_rows_with_handoff_fallback_or_lease_records_are_never_recovered(tmp_path: Path) -> None:
    host = Host(tmp_path)
    host.record_worker_environment()
    rows = {}
    for label in ("handoff", "shadow", "fallback", "lease"):
        envelope, name = stage_compile(host, label=label)
        claimed = host.claim(name)
        (host.outputs / envelope["compilation_id"]).mkdir()
        rows[label] = name
        if label == "lease":
            write_remote_cpu_record(host.jobs / "leases" / f"{job_id_for(remote.STAGE, name)}.json",
                                    {"job_id": job_id_for(remote.STAGE, name), "state": "running"})
            continue
        plan = remote.plan_remote_compilation(
            claimed, inputs=host.inputs, outputs=host.outputs, source_commit=COMMIT, config=remote_cpu_config(),
            jobs_root=host.jobs, filesystem_root=host.fs, cache_root=host.cache, host_environment=HOST_RECORD,
            require_shadow_gate=False)
        assert isinstance(plan, remote.RemotePlan)
        if label == "fallback":
            remote.write_fallback(host.jobs, plan.queue_row, reason="remote_cpu_receipt_missing", attempts=2, now=1.0)
        else:
            remote.write_handoff(host.jobs, plan, mode="authoritative" if label == "handoff" else "shadow", now=1.0)
    before = [tree_snapshot(root) for root in (host.queue, host.outputs, host.jobs)]

    assert recovery.recover_interrupted_claims(host.queue, jobs_root=host.jobs, output_root=host.outputs,
                                               source_commit=COMMIT, now=9_000.0) == []
    assert [tree_snapshot(root) for root in (host.queue, host.outputs, host.jobs)] == before
    assert all((host.queue / "processing" / name).is_file() for name in rows.values())
    assert not any(recovery.record_path(host.jobs, name).exists() for name in rows.values())


def test_a_bare_placeholder_is_dropped_and_a_recorded_result_is_finished_not_recompiled(
        tmp_path: Path, monkeypatch) -> None:
    host = Host(tmp_path)
    compiler = _stand_ins(monkeypatch)
    envelope, done = stage_compile(host, label="done")
    claimed = host.claim(done)
    _, result = compile_claimed_envelope(
        claimed, source_name=done, inputs=host.inputs.resolve(), outputs=host.outputs.resolve(), source_commit=COMMIT,
        episode_compiler=compiler, disk_reservation_root=None, storage_pins_root=None)
    write_launch_preparation_record_exclusive(host.queue / "results" / done, result)  # died before the row moved
    output = tree_snapshot(host.outputs / envelope["compilation_id"])
    _, waiting = stage_compile(host, label="waiting")
    (host.queue / "processing" / waiting).touch(mode=0o600)  # died between the claim's placeholder and its replace

    run = _run(host, compiler, now=2_000.0)
    assert run["recovered_claims"] == [
        {"name": done, "action": "finished", "interruptions": 0, "set_aside": None},
        {"name": waiting, "action": "placeholder_removed", "interruptions": 0, "set_aside": None}]
    assert (host.queue / "completed" / done).is_file()
    assert json.loads((host.queue / "results" / done).read_bytes()) == result
    assert tree_snapshot(host.outputs / envelope["compilation_id"]) == output
    assert (host.queue / "completed" / waiting).is_file()  # claimed and compiled normally once unblocked
    assert not any(recovery.record_path(host.jobs, name).exists() for name in (done, waiting))


def test_an_interrupted_fallback_compile_sets_its_partial_output_aside_and_counts(tmp_path: Path, monkeypatch) -> None:
    host = Host(tmp_path)
    host.record_worker_environment()
    envelope, name = stage_compile(host)
    claimed = host.claim(name)
    plan = remote.plan_remote_compilation(
        claimed, inputs=host.inputs, outputs=host.outputs, source_commit=COMMIT, config=remote_cpu_config(),
        jobs_root=host.jobs, filesystem_root=host.fs, cache_root=host.cache, host_environment=HOST_RECORD,
        require_shadow_gate=False)
    remote.write_fallback(host.jobs, plan.queue_row, reason="remote_cpu_receipt_missing", attempts=2, now=1.0)
    with pytest.raises(Crash):
        _run(host, _killed, now=1_000.0)
    # Still handed back, so never requeued: the next run compiles it again, from a clean output path.
    assert remote.marker_path(host.jobs, "fallback", name).is_file()
    run = _run(host, _stand_ins(monkeypatch), now=2_000.0)
    assert run["fallback_results"][0]["status"] == "compiled_for_production_launch"
    assert (host.queue / "completed" / name).is_file()
    aside = host.outputs / f".{envelope['compilation_id']}.interrupted-2000"
    assert (aside / "partial.bin").read_bytes() == b"half"
    assert json.loads(recovery.record_path(host.jobs, name).read_bytes())["interruptions"] == 1


def test_a_handed_back_row_whose_result_was_written_is_finished_not_recompiled(tmp_path: Path, monkeypatch) -> None:
    """Review I1: the fallback compile wrote the result, then died before the row moved.  The next run moves
    the row as its result says; it never sets the finished output aside, counts it, or compiles it again."""

    host = Host(tmp_path)
    host.record_worker_environment()
    envelope, name = stage_compile(host)
    claimed = host.claim(name)
    plan = remote.plan_remote_compilation(
        claimed, inputs=host.inputs, outputs=host.outputs, source_commit=COMMIT, config=remote_cpu_config(),
        jobs_root=host.jobs, filesystem_root=host.fs, cache_root=host.cache, host_environment=HOST_RECORD,
        require_shadow_gate=False)
    remote.write_fallback(host.jobs, plan.queue_row, reason="remote_cpu_receipt_missing", attempts=2, now=1.0)
    _, result = compile_claimed_envelope(
        claimed, source_name=name, inputs=host.inputs.resolve(), outputs=host.outputs.resolve(), source_commit=COMMIT,
        episode_compiler=_stand_ins(monkeypatch), disk_reservation_root=None, storage_pins_root=None)
    write_launch_preparation_record_exclusive(host.queue / "results" / name, result)  # then the process died
    output = tree_snapshot(host.outputs / envelope["compilation_id"])

    run = _run(host, _killed, now=2_000.0)  # a compile would crash the run: none may start
    assert run["fallback_results"][0]["status"] == "compiled_for_production_launch"
    assert (host.queue / "completed" / name).is_file() and not (host.queue / "processing" / name).exists()
    assert json.loads((host.queue / "results" / name).read_bytes()) == result
    assert tree_snapshot(host.outputs / envelope["compilation_id"]) == output
    assert not list(host.outputs.glob(f".{envelope['compilation_id']}.interrupted-*"))
    assert not remote.marker_path(host.jobs, "fallback", name).exists()
    assert not recovery.record_path(host.jobs, name).exists()


def test_a_second_no_spend_run_never_recovers_the_first_runs_live_claim(tmp_path: Path, monkeypatch) -> None:
    """Review minor: recovery trusts that one run holds the queue.  A run takes the queue's lock, and another
    run started meanwhile (a manual one, say) skips with a note rather than recover the live claim."""

    import fcntl
    import os

    host = Host(tmp_path)
    _, name = stage_compile(host)
    host.claim(name)  # the running instance's claim, mid-compile
    held = os.open(host.queue, os.O_RDONLY)
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        run = _run(host, _killed, now=2_000.0)
    finally:
        os.close(held)
    assert (run["status"], run["reason"]) == ("skipped", "episode_compilation_no_spend_run_in_progress")
    assert (host.queue / "processing" / name).is_file() and not recovery.record_path(host.jobs, name).exists()
    # Once that run is gone, its claim is an orphan like any other.
    run = _run(host, _stand_ins(monkeypatch), now=3_000.0)
    assert run["recovered_claims"][0]["action"] == "requeued" and (host.queue / "completed" / name).is_file()


def test_a_compile_never_removes_an_output_directory_it_did_not_create(tmp_path: Path, monkeypatch) -> None:
    """Review minor (pre-existing, moved verbatim in 4.1): when the compile's exclusive ``mkdir`` finds the
    output already there, its failure path removed that directory.  It now blocks and leaves it untouched."""

    host = Host(tmp_path)
    envelope, name = stage_compile(host)
    claimed = host.claim(name)
    existing = host.outputs / envelope["compilation_id"]
    existing.mkdir()
    (existing / "someone-elses.bin").write_bytes(b"keep")
    state, result = compile_claimed_envelope(
        claimed, source_name=name, inputs=host.inputs.resolve(), outputs=host.outputs.resolve(), source_commit=COMMIT,
        episode_compiler=_stand_ins(monkeypatch), disk_reservation_root=None, storage_pins_root=None)
    assert state == "blocked" and result["blockers"] == ["episode_compilation_failed:OSError:errno_17"]
    assert (existing / "someone-elses.bin").read_bytes() == b"keep"
