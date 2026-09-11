"""A verified failed replay becomes one bounded engineering handoff."""

import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from blueprint_pipeline.agent_execution import stage_recovery
from blueprint_pipeline.agent_execution.contracts import digest
from blueprint_pipeline.agent_execution.engineering import (
    EngineeringPolicy,
    queue_engineering_handoff,
    flush_engineering_handoffs,
    verify_engineering_candidate,
)
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.agent_execution.production import ProductionAgentService
from tests.test_agent_stage_recovery import prepare_fixture
from tests.test_agent_production_service import write


def setup(tmp_path, monkeypatch, *, status="job_refused"):
    service, record, runtime, _ = prepare_fixture(tmp_path, owner_client_id="blueprint-webapp")
    monkeypatch.setattr(stage_recovery, "require_offline_isolation", lambda _: None)

    def execute(argv, **kwargs):
        stage_recovery.write_json(
            Path(argv[argv.index("--json-out") + 1]),
            {
                "status": status,
                "phase": "source_selections",
                "blocker": "sam31_phase_file_reference_invalid",
            },
        )
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(stage_recovery.subprocess, "run", execute)
    stage_recovery.process_one(service)
    runtime.operations.execute(
        record.task,
        turn_id="fixture_turn",
        call_id="fixture_call",
        name="replay_retained_stage",
        arguments={"replay_id": "failed_boundary"},
    )
    service.journal.set_state(
        record.task.task_id,
        "completed",
        result={
            "result_digest": digest({"diagnosis": 1}),
            "output": {
                "disposition": "investigate",
                "summary": "Untrusted model prose is not an engineering instruction.",
            },
        },
    )
    policy = EngineeringPolicy(
        schema_version="blueprint_agent_engineering_policy.v1",
        enabled=True,
        policy_id="fixture-policy",
        repository="ognjhunt/BlueprintCapturePipeline",
        run_id_prefixes=("recovery_",),
        allowed_paths=("src/blueprint_pipeline/task_evaluation_sam31_preparation_stages.py",),
        required_test_paths=("tests/test_task_evaluation_stage_replay.py",),
        maximum_handoffs=1,
        maximum_changed_files=1,
        maximum_patch_bytes=8000,
        worker_budget_reference="existing-worker-budget",
        maximum_worker_timeout_seconds=600,
        accepted_by="fixture-operator",
        expires_at=int(time.time()) + 1200,
    )
    policy_path = tmp_path / "engineering-policy.json"
    write(policy_path, policy.model_dump(mode="json"))
    token = tmp_path / "sync-key"
    token.write_text("fixture-sync-token")
    token.chmod(0o600)
    config = json.loads(service.config_path.read_text())
    config.update(
        engineering_policy_file=str(policy_path),
        webapp_admission_url="https://example.com/api/internal/pipeline/agent-execution/admissions",
        webapp_sync_token_file=str(token),
    )
    write(service.config_path, config)
    service = ProductionAgentService(service.config_path, source_commit=record.task.source_commit)
    return service, service.record(record.task.task_id), policy_path


def test_verified_replay_queues_once_and_lost_ack_reuses_exact_packet(tmp_path, monkeypatch):
    service, record, _ = setup(tmp_path, monkeypatch)
    packet = queue_engineering_handoff(service, record)
    assert packet["child_id"] == record.stage_replays[0].child_id
    assert packet["independent_review_required"] and not packet["paid_resubmission_authorized"]
    assert "Untrusted model prose" not in json.dumps(packet)
    assert queue_engineering_handoff(service, record) == packet
    calls = []

    def post(value, **kwargs):
        calls.append(value)
        assert kwargs["endpoint"].endswith("/agent-execution/engineering")
        if len(calls) == 1:
            raise OSError("response lost after persistence")
        return {
            "schema_version": "blueprint_agent_engineering_admission_receipt.v1",
            "handoff_id": value["handoff_id"],
            "handoff_digest": value["handoff_digest"],
            "stored": True,
        }

    assert flush_engineering_handoffs(service, post=post)["status"] == "delivery_pending"
    restarted = ProductionAgentService(service.config_path, source_commit=record.task.source_commit)
    assert queue_engineering_handoff(restarted, record) == packet
    assert flush_engineering_handoffs(restarted, post=post)["stored"]
    assert flush_engineering_handoffs(restarted, post=post) is None
    assert calls[0] == calls[1]


def test_capacity_refusal_does_not_authorize_a_code_repair(tmp_path, monkeypatch):
    service, record, _ = setup(tmp_path, monkeypatch, status="admission_refused")
    assert queue_engineering_handoff(service, record) is None


def test_revocation_prevents_queued_handoff_delivery(tmp_path, monkeypatch):
    service, record, path = setup(tmp_path, monkeypatch)
    queue_engineering_handoff(service, record)
    value = json.loads(path.read_text())
    value["enabled"] = False
    write(path, value)

    def post(*args, **kwargs):
        pytest.fail("revoked engineering packet was transmitted")

    assert flush_engineering_handoffs(service, post=post) is None
    assert (
        service.status(record.task.task_id, "blueprint-webapp")["engineering_policy"]["enabled"]
        is False
    )


def test_new_diagnostic_is_not_blocked_by_optional_engineering_policy(tmp_path, monkeypatch):
    service, record, path = setup(tmp_path, monkeypatch)
    # An unavailable optional engineering policy cannot prevent cleanup or
    # unrelated task reconciliation in the production worker.
    path.write_text("{}")
    path.chmod(0o600)
    service.autostart()
    assert service.journal.task(record.task.task_id)["cleanup_state"] == "pending"


def test_candidate_scope_reads_real_git_objects_and_protects_baseline_tests(tmp_path):
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.name", "Fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("remote", "add", "origin", "https://github.com/ognjhunt/BlueprintCapturePipeline.git")
    (root / "src").mkdir()
    (root / "tests").mkdir()
    source = root / "src/handler.py"
    test = root / "tests/test_guard.py"
    source.write_text("value = 1\n")
    test.write_text("assert True\n")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    policy = EngineeringPolicy(
        schema_version="blueprint_agent_engineering_policy.v1",
        enabled=True,
        policy_id="fixture",
        repository="ognjhunt/BlueprintCapturePipeline",
        run_id_prefixes=("scene-",),
        allowed_paths=("src/handler.py", "tests/test_guard.py"),
        required_test_paths=("tests/test_guard.py",),
        maximum_handoffs=1,
        maximum_changed_files=2,
        maximum_patch_bytes=8000,
        worker_budget_reference="existing",
        maximum_worker_timeout_seconds=300,
        accepted_by="fixture",
        expires_at=int(time.time()) + 600,
    )
    packet = {
        "source_commit": base,
        "policy": policy.model_dump(mode="json"),
        "policy_digest": policy.policy_digest,
        "handoff_id": "repair-" + "a" * 64,
    }
    packet["handoff_digest"] = cross_runtime_canonical_digest(packet)
    source.write_text("value = 2\n")
    git("add", ".")
    git("commit", "-qm", "candidate")
    candidate = git("rev-parse", "HEAD")
    result = verify_engineering_candidate(packet, repository_root=root, candidate_commit=candidate)
    assert result["changed_paths"] == ["src/handler.py"]
    assert not result["tests_passed_inferred"] and not result["production_promotion_authorized"]
    test.write_text("assert False\n")
    git("add", ".")
    git("commit", "-qm", "changed mandatory test")
    with pytest.raises(Exception, match="required_baseline_test_changed"):
        verify_engineering_candidate(
            packet, repository_root=root, candidate_commit=git("rev-parse", "HEAD")
        )
