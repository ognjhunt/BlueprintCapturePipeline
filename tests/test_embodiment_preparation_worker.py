"""Batch scheduling keeps each robot's evidence within one funded deadline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from blueprint_pipeline import deployment_embodiment_assets as assets
from blueprint_pipeline import deployment_embodiment_preparation_worker as worker
from blueprint_pipeline.deployment_embodiment_preparation_contract import (
    INPUT_FILENAME, INPUT_SCHEMA, REQUEST_SCHEMA, RESULT_FILENAME,
    sealed, validate_preparation_result,
)


IDS = ["i2rt_yam_bimanual", "franka_research_3_bimanual", "agilex_piper_bimanual"]


@pytest.fixture
def batch(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = tmp_path / "output"
    clock = SimpleNamespace(monotonic=0.0, wall=1000.0)
    source_checks = []
    source_digests = {identity: "sha256:" + hashlib.sha256(identity.encode()).hexdigest() for identity in IDS}

    def verify_source(path):
        # Public source retention is independently tested. These tests exercise
        # the real batch input/request seals without fetching robot sources.
        source_checks.append(path.name)
        return {"bundle_digest": source_digests[path.name]}

    monkeypatch.setattr(assets, "verify_source_bundle", verify_source)
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH", "10000")

    def inputs(*, ids=IDS, unavailable=(), expires=20000.0):
        request = sealed({
            "schema_version": REQUEST_SCHEMA, "authorization_id": "worker-test",
            "release_commit": "1" * 40, "embodiment_ids": list(ids), "max_spend_usd": 4.0,
            "max_hourly_rate_usd": .8, "ttl_seconds": 14400,
            "expires_at_epoch": expires, "authorization_reference": "test://worker-batch",
        }, "request_digest")
        value = sealed({"schema_version": INPUT_SCHEMA, "request": request,
            "sources": [{"embodiment_id": identity, "status": "source_unavailable", "reason": "public_fetch_failed"}
                        if identity in unavailable else
                        {"embodiment_id": identity, "status": "source_ready",
                         "bundle_digest": source_digests[identity], "recipe_digest": source_digests[identity],
                         "source_bytes": 32}
                        for identity in ids]}, "inputs_digest")
        (runtime / INPUT_FILENAME).write_text(json.dumps(value))
        return value

    def execute(run):
        result = worker.run_batch(runtime, output, run=run,
                                  now=lambda: clock.monotonic, wall=lambda: clock.wall)
        source = json.loads((runtime / INPUT_FILENAME).read_text())
        assert validate_preparation_result(
            result, expected_inputs_digest=source["inputs_digest"],
            expected_ids=source["request"]["embodiment_ids"], output_root=output) == result
        assert json.loads((output / RESULT_FILENAME).read_text()) == result
        assert result["claim_ceiling"] == "robot_asset_import_only"
        assert result["candidate_policy_queried"] is False
        assert result["task_executed"] is False
        assert result["provider_teardown_verified"] is False
        return result

    return SimpleNamespace(runtime=runtime, output=output, clock=clock, source_checks=source_checks,
                           inputs=inputs, execute=execute)


def imported(command, *, size=32):
    """Retained files from a simulated importer subprocess, not an Isaac run."""
    identity = command[command.index("--embodiment") + 1]
    body = Path(command[command.index("--output") + 1])
    (body / "prepared").mkdir(parents=True)
    (body / "prepared" / "robot.usd").write_bytes(b"x" * size)
    (body / "body-result.json").write_text(json.dumps({
        "embodiment_id": identity, "status": "imported", "native_import_executed": True,
        "task_executed": False, "policy_executed": False,
    }))
    return SimpleNamespace(returncode=0)


def test_process_failure_and_timeout_keep_later_robot_imports_running(batch):
    batch.inputs()
    attempted = []

    def run(command, **options):
        identity = command[-1]
        attempted.append(identity)
        options["stdout"].write(b"bounded importer diagnostic\n")
        if identity == IDS[0]:
            return SimpleNamespace(returncode=23)
        if identity == IDS[1]:
            raise subprocess.TimeoutExpired(command, options["timeout"])
        return imported(command)

    result = batch.execute(run)
    assert attempted == IDS
    assert [row["status"] for row in result["results"]] == ["import_failed", "import_failed", "imported"]
    assert result["results"][0]["exit_code"] == 23
    assert result["results"][1]["reason"] == "isaac_import_timeout"
    assert (batch.output / "bodies" / IDS[2] / "prepared" / "robot.usd").is_file()


def test_unavailable_source_has_its_own_reason_and_does_not_launch_or_stop_others(batch):
    batch.inputs(unavailable=[IDS[1]])
    attempted = []

    def run(command, **_options):
        attempted.append(command[-1])
        return imported(command)

    result = batch.execute(run)
    assert attempted == [IDS[0], IDS[2]]
    assert batch.source_checks == attempted
    assert result["results"][1] == {
        "embodiment_id": IDS[1], "status": "not_imported", "reason": "source_unavailable",
        "source_reason": "public_fetch_failed",
    }


@pytest.mark.parametrize("deadline_kind", ["provider", "authorization"])
def test_remaining_absolute_deadline_includes_elapsed_work_and_shutdown_margin(batch, monkeypatch, deadline_kind):
    batch.inputs(expires=1700 if deadline_kind == "authorization" else 20000)
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH",
                       "1700" if deadline_kind == "provider" else "10000")
    calls = []

    def run(command, **options):
        calls.append((command[-1], options["timeout"]))
        batch.clock.monotonic += 50
        return imported(command)

    result = batch.execute(run)
    assert len(calls) == 1
    # The original request allows four hours, but only 100 seconds remain
    # after the existing absolute deadline's upload/teardown reservation.
    assert 0 < calls[0][1] <= 100
    assert [row["reason"] for row in result["results"][1:]] == [
        "batch_time_budget_exhausted", "batch_time_budget_exhausted"]


def test_expired_provider_window_records_every_skipped_robot_without_subprocess(batch, monkeypatch):
    batch.inputs()
    monkeypatch.setenv("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH", "999")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an expired allocation cannot start an importer")

    result = batch.execute(forbidden)
    assert [row["status"] for row in result["results"]] == ["not_imported"] * len(IDS)
    assert all(row["reason"] == "batch_time_budget_exhausted" for row in result["results"])


def test_missing_provider_deadline_cannot_silently_start_a_fresh_ttl(batch, monkeypatch):
    batch.inputs()
    monkeypatch.delenv("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a missing allocation deadline cannot start an importer")

    with pytest.raises(KeyError, match="BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH"):
        batch.execute(forbidden)
    assert not (batch.output / RESULT_FILENAME).exists()


def test_output_limit_skips_oversize_robot_but_keeps_smaller_later_robot(batch, monkeypatch):
    batch.inputs()
    # Preserve the production log/receipt reservation while shrinking the model
    # capacity; no large fixture or real importer is needed to test scheduling.
    monkeypatch.setattr(worker, "MAX_OUTPUT_BYTES", 64 * 1024**2 + 2048)
    sizes = dict(zip(IDS, [1024, 2048, 16], strict=True))

    def run(command, **_options):
        return imported(command, size=sizes[command[-1]])

    result = batch.execute(run)
    assert [row["status"] for row in result["results"]] == ["imported", "not_retained", "imported"]
    assert result["results"][1]["reason"] == "batch_output_budget_exhausted"
    assert not (batch.output / "bodies" / IDS[1]).exists()
    assert (batch.output / "bodies" / IDS[0]).is_dir()
    assert (batch.output / "bodies" / IDS[2]).is_dir()
    assert sum(row["size_bytes"] for row in result["files"]) <= 2048


def test_failed_process_logs_keep_a_bounded_tail_and_no_success_claim(batch):
    batch.inputs(ids=IDS[:1])

    def run(_command, **options):
        options["stdout"].write(b"prefix" + b"x" * 1024**2 + b" final error")
        return SimpleNamespace(returncode=1)

    result = batch.execute(run)
    diagnostic = (batch.output / (IDS[0] + ".log")).read_bytes()
    assert len(diagnostic) <= 1024**2
    assert diagnostic.endswith(b" final error")
    assert result["status"] == "completed"
    assert result["results"][0]["status"] == "import_failed"
    assert all(row["path"].endswith(".log") for row in result["files"])


@pytest.mark.parametrize("receipt_fault", ["missing", "invalid_json"])
def test_zero_exit_without_readable_import_receipt_is_individual_failure(batch, receipt_fault):
    batch.inputs(ids=IDS[:2])

    def run(command, **_options):
        if command[-1] == IDS[0]:
            if receipt_fault == "invalid_json":
                body = Path(command[command.index("--output") + 1])
                body.mkdir(parents=True)
                (body / "body-result.json").write_text("incomplete result {")
            return SimpleNamespace(returncode=0)
        return imported(command)

    result = batch.execute(run)
    assert [row["status"] for row in result["results"]] == ["import_failed", "imported"]
    assert not (batch.output / "bodies" / IDS[0]).exists()
    assert (batch.output / "bodies" / IDS[1] / "prepared" / "robot.usd").is_file()
