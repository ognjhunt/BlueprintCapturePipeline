"""The G1 queue worker uses the canonical allocator once per accepted intent."""

from __future__ import annotations

import json

import pytest

from blueprint_pipeline.native_g1_team_campaign_dispatcher import dispatch_one_g1_team_campaign
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
