"""Tiny archive proofs for stopped-launch extracted-cache reclamation."""

from datetime import datetime, timedelta, timezone
import json
import os
import zipfile

import pytest

from blueprint_pipeline import task_evaluation_stopped_pretraining_retention as retention
from blueprint_pipeline.control_plane_workspace_lock import workspace_lock
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.launch_profile_immutable_inputs import immutable_input_digest
from tests.test_task_evaluation_launch_dispatcher import _profile, _zero_guard


def save(path, value, field):
    value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value))
    return value


@pytest.fixture
def stopped(tmp_path, monkeypatch):
    launch_base = tmp_path / "runs"
    launch_base.mkdir()
    logical = tmp_path / "semantic-pretraining"
    logical.mkdir()
    monkeypatch.setattr(retention, "LAUNCH_ROOT", launch_base)
    monkeypatch.setattr(retention, "LOGICAL_ROOT", logical)
    launch = launch_base / "stopped-launch"
    launch.mkdir()
    archive = tmp_path / "provider.zip"
    with zipfile.ZipFile(archive, "w") as z:
        info = zipfile.ZipInfo("provider_runtime/source.py")
        info.external_attr = 0o444 << 16
        z.writestr(info, b"# exact source\n")
    archive_hash = retention._file_record(archive, blocker="test")["sha256"]
    bundle_path = tmp_path / "bundle-receipt.json"
    bundle = save(
        bundle_path,
        {
            "schema_version": "task_evaluation_scene_configuration_provider_bundle.v1",
            "bundle_path": str(archive),
            "bundle_sha256": archive_hash,
            "run_id": "source-run",
            "source_commit": "a" * 40,
        },
        "receipt_digest",
    )
    authority_path = tmp_path / "authority.json"
    authority = save(
        authority_path,
        {
            "schema_version": "task_evaluation_scene_configuration_paid_authority.v1",
            "provider": "vast",
            "maximum_paid_attempts": 1,
            "maximum_provider_allocations": 1,
            "maximum_automatic_retries": 0,
            "automatic_paid_retry_authorized": False,
            "retry_cap": 0,
            **{k: bundle[k] for k in ("bundle_sha256", "run_id", "source_commit")},
        },
        "authority_digest",
    )
    profile = _profile(tmp_path)
    profile["allocator"]["argv"] += [
        "--scene-configuration-bundle-receipt",
        str(bundle_path),
        "--scene-configuration-attempt-authority",
        str(authority_path),
    ]
    profile["immutable_inputs"] = [
        {"name": name, "path": str(path), "digest": immutable_input_digest(path)}
        for name, path in [
            ("source_bundle_manifest", bundle_path),
            ("evaluation_run_spec", bundle_path),
            ("scene_configuration_attempt_authority", authority_path),
        ]
    ]
    save(launch / "launch_profile.json", profile, "profile_digest")
    request = save(
        launch / "launch_request.json",
        {
            "schema_version": "task_evaluation_launch_request.v1",
            "launch_id": launch.name,
            "launch_profile_digest": profile["profile_digest"],
        },
        "request_digest",
    )
    now = datetime.now(timezone.utc)
    started = save(
        launch / "launch_started.json",
        {
            "schema_version": "task_evaluation_launch_started.v1",
            "launch_id": launch.name,
            "request_digest": request["request_digest"],
            "automatic_retry_authorized": False,
            "process_id": 99999999,
            "started_at": (now - timedelta(minutes=2)).isoformat(),
        },
        "started_digest",
    )
    reconciliations = launch / "reconciliations"
    reconciliations.mkdir()
    guard = reconciliations / "zero.json"
    guard.write_text(json.dumps(_zero_guard(generated_at=now)))
    save(
        launch / "orphan_recovery_receipt.json",
        {
            "schema_version": "task_evaluation_launch_orphan_recovery.v1",
            "launch_id": launch.name,
            "launch_profile_digest": profile["profile_digest"],
            "request_digest": request["request_digest"],
            "started_digest": started["started_digest"],
            "observed_at": now.isoformat(),
            "status": "provider_zero_confirmed",
            "provider_zero_confirmed": True,
            "recovery_basis": "stopped_dispatcher_and_fresh_provider_zero",
            "automatic_retry_performed": False,
            "allocator_invoked": False,
            "blockers": [],
            "dispatcher_state": {"ActiveState": "failed", "MainPID": "0", "ControlGroup": ""},
            "guard_report_path": str(guard),
            "guard_report_sha256": retention._file_record(guard, blocker="test")["sha256"],
            "required_providers": ["vast"],
        },
        "recovery_digest",
    )
    key = canonical_digest({"bundle": archive_hash, "authority": authority["authority_digest"]})[7:]
    workspace = logical / key
    (workspace / "bundle/provider_runtime").mkdir(parents=True)
    source = workspace / "bundle/provider_runtime/source.py"
    source.write_bytes(b"# exact source\n")
    source.chmod(0o444)
    (workspace / "output").mkdir()
    (workspace / "output/receipt.json").write_text("retained output")
    (workspace / "secret-evidence.json").write_text("never remove this sibling")
    with workspace_lock(workspace):
        pass
    return launch, workspace, archive, tmp_path / "retention-plan.json"


def test_dry_run_then_apply_removes_only_identical_extraction(stopped):
    launch, workspace, archive, plan = stopped
    before = archive.read_bytes()
    dry = retention.retain_stopped_bundle(launch_root=launch, plan_out=plan)
    assert (workspace / "bundle/provider_runtime/source.py").exists()
    assert dry["removable_bundle"]["archive_byte_identity_proven"] is True
    result = retention.retain_stopped_bundle(
        plan_path=plan, apply=True, acknowledgement=retention.ACK
    )
    assert result["status"] == "applied" and result["historical_spend_settled"] is False
    assert not (workspace / "bundle").exists()
    assert archive.read_bytes() == before
    assert (workspace / "output/receipt.json").read_text() == "retained output"
    assert (workspace / "secret-evidence.json").read_text() == "never remove this sibling"
    assert plan.with_name(plan.name + ".applied.json").is_file()


@pytest.mark.parametrize(
    "change",
    [
        "bytes",
        "mode",
        "extra_file",
        "archive",
        "live_lock",
        "live_pid",
        "path",
        "recovery",
        "new_scope",
    ],
)
def test_changes_and_live_writers_refuse_without_deleting(stopped, change):
    launch, workspace, archive, plan = stopped
    retention.retain_stopped_bundle(launch_root=launch, plan_out=plan)
    source = workspace / "bundle/provider_runtime/source.py"
    if change == "bytes":
        source.chmod(0o644)
        source.write_text("different")
    elif change == "mode":
        source.chmod(0o777)
    elif change == "extra_file":
        (workspace / "bundle/unbound").write_text("unexpected")
    elif change == "archive":
        archive.write_bytes(b"different zip")
    elif change == "path":
        original = workspace / "bundle"
        original.rename(workspace / "original")
        original.symlink_to(workspace / "original", target_is_directory=True)
    elif change == "recovery":
        p = launch / "orphan_recovery_receipt.json"
        v = json.loads(p.read_text())
        v["provider_zero_confirmed"] = False
        save(p, v, "recovery_digest")
    elif change == "new_scope":
        p = launch / "launch_profile.json"
        v = json.loads(p.read_text())
        v["allocator"]["retry_cap"] = 1
        save(p, v, "profile_digest")
    elif change == "live_pid":
        p = launch / "launch_started.json"
        v = json.loads(p.read_text())
        v["process_id"] = os.getpid()
        save(p, v, "started_digest")
        p = launch / "orphan_recovery_receipt.json"
        r = json.loads(p.read_text())
        r["started_digest"] = v["started_digest"]
        save(p, r, "recovery_digest")
    if change == "live_lock":
        with workspace_lock(workspace):
            with pytest.raises(ValueError, match="workspace_writer"):
                retention.retain_stopped_bundle(
                    plan_path=plan, apply=True, acknowledgement=retention.ACK
                )
    else:
        with pytest.raises(ValueError):
            retention.retain_stopped_bundle(
                plan_path=plan, apply=True, acknowledgement=retention.ACK
            )
    assert (workspace / "bundle").exists() and archive.exists()
    assert (workspace / "output/receipt.json").read_text() == "retained output"
    assert not plan.with_name(plan.name + ".applied.json").exists()
