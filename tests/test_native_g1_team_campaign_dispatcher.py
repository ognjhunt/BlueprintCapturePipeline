"""The G1 queue worker uses the canonical allocator once per accepted intent."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline.native_g1_team_campaign_dispatcher import (
    _refresh_paid_admission,
    dispatch_one_g1_team_campaign,
)
from tests.test_native_g1_team_campaign_preparation import COMMIT, _accepted


def _ready(tmp_path, monkeypatch):
    registry, _, intent, _ = _accepted(tmp_path, monkeypatch)
    def bundle(**kwargs):
        kwargs["job_dir"].mkdir()
        (kwargs["job_dir"] / "native_g1_provider_bundle.v1.json").write_text("{}")
        return {"bundle_sha256": "sha256:" + "1" * 64}
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_preparation.build_g1_provider_bundle", bundle,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_preparation.load_verified_g1_provider_bundle",
        lambda *_args, **_kwargs: {"bundle_sha256": "sha256:" + "1" * 64},
    )
    return dict(
        queue_root=intent.parent.parent, registry_path=registry,
        work_root=tmp_path / "work", implementation_commit=COMMIT,
        admission_refresher=lambda _run_root: [],
    )


def _write_adapter(command, status):
    output = command[command.index("--adapter-output") + 1]
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(status, stream)


def test_dry_then_paid_dispatch_uses_exact_once_bound_and_terminal_proof(tmp_path, monkeypatch):
    args = _ready(tmp_path, monkeypatch)
    commands = []
    def runner(command, log_path):
        commands.append(command)
        assert log_path.parent.name == "run"
        assert "--adp-max-spend-usd" in command
        assert command[command.index("--adp-max-spend-usd") + 1] == "12"
        assert command[command.index("--adp-hard-ttl-seconds") + 1] == "14400"
        if "--execute" in command:
            _write_adapter(command, {
                "status": "completed", "continuing_spend_from_this_run": False,
                "g1_output_verification": {
                    "status": "verified_development_only", "episodes": [{}, {}, {}, {}],
                },
            })
        else:
            _write_adapter(command, {"status": "dry_run_ready"})
        return 0
    dry = dispatch_one_g1_team_campaign(**args, allocator_runner=runner)
    assert dry["status"] == "dry_run_ready"
    assert "--execute" not in commands[0]
    terminal = dispatch_one_g1_team_campaign(**args, execute=True, allocator_runner=runner)
    assert terminal["status"] == "controller_completed_pending_billing_and_private_delivery"
    assert terminal["four_episodes_verified"] is True
    assert terminal["run_teardown_confirmed_by_adapter"] is True
    assert terminal["official_billing_reconciled"] is False
    assert terminal["private_review_delivered"] is False
    assert "--execute" in commands[1]
    assert len(commands) == 2
    assert dispatch_one_g1_team_campaign(**args, execute=True, allocator_runner=runner)["status"] == "no_pending_intent"
    assert len(commands) == 2


def test_dry_run_blocker_never_writes_paid_start_and_can_retry(tmp_path, monkeypatch):
    args = _ready(tmp_path, monkeypatch)
    calls = []
    def runner(command, log_path):
        calls.append(log_path)
        _write_adapter(command, {"status": "blocked", "blockers": ["disk_headroom"]})
        return 2
    for _ in range(2):
        result = dispatch_one_g1_team_campaign(**args, execute=True, allocator_runner=runner)
        assert result["status"] == "blocked_before_provider"
        assert result["provider_mutation_performed"] is False
    assert calls[0] != calls[1]
    assert not list(args["work_root"].glob("g1-*/execution_started.json"))


def test_spend_lock_refresh_follows_dry_run_and_blocks_before_one_use_start(tmp_path, monkeypatch):
    args = _ready(tmp_path, monkeypatch)
    events = []
    def runner(command, _log_path):
        events.append("paid" if "--execute" in command else "dry")
        _write_adapter(command, {"status": "dry_run_ready"})
        return 0
    def stale(_run_root):
        events.append("refresh_stale")
        return ["spend_admission_lock_stale"]
    args["admission_refresher"] = stale
    blocked = dispatch_one_g1_team_campaign(**args, execute=True, allocator_runner=runner)
    assert blocked["status"] == "blocked_before_provider"
    assert blocked["blockers"] == ["spend_admission_lock_stale"]
    assert events == ["dry", "refresh_stale"]
    assert not list(args["work_root"].glob("g1-*/execution_started.json"))
    args["admission_refresher"] = lambda _run_root: events.append("refresh_open") or []
    completed = dispatch_one_g1_team_campaign(**args, execute=True, allocator_runner=runner)
    assert events == ["dry", "refresh_stale", "refresh_open", "paid"]
    assert completed["status"] == "blocked_after_allocator_attempt"
    assert len(list(args["work_root"].glob("g1-*/execution_started.json"))) == 1


def test_paid_admission_refresh_uses_canonical_guard_and_three_minute_lock(tmp_path, monkeypatch):
    module = "blueprint_pipeline.native_g1_team_campaign_dispatcher"
    values = {
        "BLUEPRINT_GPU_SPEND_GUARD_MAX_BOOT_SECONDS": "480",
        "BLUEPRINT_GPU_SPEND_GUARD_MAX_BOOTED_ORPHAN_SECONDS": "14400",
        "BLUEPRINT_GPU_FLEET_MAX_LIVE_INSTANCES": "10",
        "BLUEPRINT_GPU_FLEET_MAX_BURN_USD_PER_HOUR": "10.0",
        "BLUEPRINT_GPU_SPEND_LEDGER": str(tmp_path / "gpu_spend_guard" / "spend_ledger.json"),
        "BLUEPRINT_GPU_FLEET_MAX_DAILY_SPEND_USD": "100.0",
        "BLUEPRINT_GPU_FLEET_MAX_TOTAL_SPEND_USD": "5000.0",
        "BLUEPRINT_GPU_BILLING_EXPORT": str(tmp_path / "gpu_spend_guard" / "billing.json"),
        "BLUEPRINT_PAID_SPEND_ADMISSION_LOCK_PATH": str(tmp_path / "gpu_spend_guard" / "lock.json"),
        "BLUEPRINT_GPU_SPEND_GUARD_REPORT": str(tmp_path / "gpu_spend_guard" / "latest.json"),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    captured = []
    def run(command, **kwargs):
        captured.append((command, kwargs))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(f"{module}.subprocess.run", run)
    monkeypatch.setattr(f"{module}._load_spend_admission_lock", lambda _path: {"status": "open"})
    def validate(lock, **kwargs):
        assert lock["status"] == "open"
        assert kwargs["max_age_seconds"] == 180
        assert kwargs["required_provider"] == "vast"
        return []
    monkeypatch.setattr(f"{module}.validate_spend_admission_lock", validate)
    run_root = tmp_path / "run"
    run_root.mkdir()
    assert _refresh_paid_admission(run_root) == []
    command, kwargs = captured[0]
    assert "--reap" in command
    assert "--require-billing-reconciliation" in command
    assert command[command.index("--output-root") + 1] == str(tmp_path)
    assert command[command.index("--admission-lock-report") + 1] == values["BLUEPRINT_PAID_SPEND_ADMISSION_LOCK_PATH"]
    assert kwargs["timeout"] == 300


def test_interrupted_paid_invocation_is_never_automatically_relaunched(tmp_path, monkeypatch):
    args = _ready(tmp_path, monkeypatch)
    calls = []
    def runner(command, _log_path):
        calls.append(command)
        if "--execute" in command:
            raise RuntimeError("simulated caller interruption")
        _write_adapter(command, {"status": "dry_run_ready"})
        return 0
    with pytest.raises(RuntimeError, match="caller interruption"):
        dispatch_one_g1_team_campaign(**args, execute=True, allocator_runner=runner)
    assert list(args["work_root"].glob("g1-*/execution_started.json"))
    assert dispatch_one_g1_team_campaign(**args, execute=True, allocator_runner=runner)["status"] == "awaiting_exact_attempt_reconciliation"
    assert len(calls) == 2


def test_expired_authority_is_sealed_without_provider_or_bundle_work(tmp_path, monkeypatch):
    args = _ready(tmp_path, monkeypatch)
    intent_path = next(args["queue_root"].glob("g1-*/intent.json"))
    intent = json.loads(intent_path.read_text())
    expiry = intent["request"]["authorization"]["expires_at_epoch"]
    monkeypatch.setattr(
        "blueprint_pipeline.native_g1_team_campaign_dispatcher.time.time",
        lambda: expiry + 1,
    )
    calls = []
    def runner(command, log_path):
        calls.append((command, log_path))
        raise AssertionError("allocator must not run")
    expired = dispatch_one_g1_team_campaign(**args, execute=True, allocator_runner=runner)
    assert expired["status"] == "authorization_expired_before_provider"
    assert expired["provider_mutation_performed"] is False
    assert not calls
    assert not list(args["work_root"].glob("g1-*/bundle"))
    assert dispatch_one_g1_team_campaign(**args, execute=True, allocator_runner=runner)["status"] == "no_pending_intent"


def test_prior_release_dry_bundle_is_preserved_and_superseded_without_paid_start(tmp_path, monkeypatch):
    args = _ready(tmp_path, monkeypatch)
    commands = []
    def runner(command, log_path):
        commands.append(command)
        _write_adapter(command, {"status": "dry_run_ready"})
        return 0
    dry = dispatch_one_g1_team_campaign(**args, allocator_runner=runner)
    assert dry["status"] == "dry_run_ready"
    old_bundle = next(args["work_root"].glob("g1-*/bundle/native_g1_provider_bundle.v1.json"))
    old_bytes = old_bundle.read_bytes()
    args["implementation_commit"] = "b" * 40
    superseded = dispatch_one_g1_team_campaign(**args, execute=True, allocator_runner=runner)
    assert superseded["status"] == "superseded_before_provider_by_release"
    assert superseded["provider_mutation_performed"] is False
    assert superseded["prepared_implementation_commit"] == COMMIT
    assert superseded["implementation_commit"] == "b" * 40
    assert old_bundle.read_bytes() == old_bytes
    assert len(commands) == 1
    assert not list(args["work_root"].glob("g1-*/execution_started.json"))
    assert dispatch_one_g1_team_campaign(**args, execute=True, allocator_runner=runner)["status"] == "no_pending_intent"
