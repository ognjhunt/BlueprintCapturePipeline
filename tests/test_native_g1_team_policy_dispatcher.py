"""The accepted selected policy reaches a durable, one-attempt queue bridge."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from blueprint_pipeline import native_g1_team_policy_dispatcher as dispatcher
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest as digest
from tests.test_native_g1_team_policy_preparation import _selected


def _queue(tmp_path, monkeypatch):
    selected, authority = _selected(tmp_path, monkeypatch)
    intent = authority["intent"]
    approval_root = tmp_path / "operator-approvals"
    approval_root.mkdir()
    approval_path = approval_root / (intent["intent_id"] + ".json")
    approval_path.write_text(json.dumps(authority["operator_approval"]))
    events = []
    class Reservation:
        def __enter__(self):
            events.append("reserve")
            return self
        def __exit__(self, *args):
            events.append("release")
    class Health:
        def check(self):
            pass
    def reservation(*args, **kwargs):
        assert args == ("launch_preparation",)
        return Reservation()
    monkeypatch.setattr(dispatcher, "reserve_control_plane_disk", reservation)
    monkeypatch.setattr(dispatcher, "keep_reservation_live", lambda reservation: nullcontext(Health()))
    def prepare(**kwargs):
        assert kwargs["authority_arguments"]["approval_path"] == approval_path
        directory = kwargs["work_root"] / intent["intent_id"]
        directory.mkdir(exist_ok=True)
        prepared = {
            "schema_version": "native_g1_team_policy_preparation.v1",
            "status": "bundle_prepared_not_executed", "intent_id": intent["intent_id"],
            "intent_digest": intent["intent_digest"], "implementation_commit": selected["implementation_commit"],
            "bundle_receipt_path": str(directory / "bundle/receipt.json"),
            "authorization": intent["request"]["authorization"],
            "execution_packet_digest": "sha256:" + "a" * 64,
        }
        prepared["preparation_digest"] = digest(prepared, digest_field="preparation_digest")
        return prepared
    monkeypatch.setattr(dispatcher, "prepare_g1_team_policy", prepare)
    return {
        "queue_root": selected["authority_arguments"]["intent_path"].parent.parent,
        "registry_path": selected["authority_arguments"]["registry_path"],
        "approval_root": approval_root, "trusted_clients": {"blueprint-webapp"},
        "work_root": selected["work_root"], "sonic_asset_dir": selected["sonic_asset_dir"],
        "credential_registry_path": tmp_path / "credential-registry.json",
        "implementation_commit": selected["implementation_commit"],
        "admission_refresher": lambda _: [],
    }, intent, approval_path, events


def _adapter(command, value):
    Path(command[command.index("--adapter-output") + 1]).write_text(json.dumps(value))


def test_selected_probe_replays_dry_and_paid_once_without_delivery_claim(tmp_path, monkeypatch):
    args, intent, _, events = _queue(tmp_path, monkeypatch)
    commands = []
    def runner(command, log):
        commands.append(command)
        assert command[command.index("--probe-kind") + 1] == "native-g1-team-policy"
        assert command[command.index("--adp-max-spend-usd") + 1] == "12"
        assert "--g1-team-credential-registry" in command
        assert log.parent.name == "run"
        if "--execute" in command:
            _adapter(command, {"status": "completed", "continuing_spend_from_this_run": False,
                               "g1_team_output_verification": {"status": "verified_development_only", "policy_query_count": 1}})
        else:
            _adapter(command, {"status": "dry_run_ready", "provider_mutations_performed": 0})
        return 0
    assert dispatcher.dispatch_one_g1_team_policy(**args, allocator_runner=runner)["status"] == "dry_run_ready"
    result = dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=runner)
    assert ["--execute" in command for command in commands] == [False, False, True]
    assert result["status"] == "controller_completed_pending_billing_and_private_delivery"
    assert result["official_billing_reconciled"] is False
    assert result["private_review_delivered"] is False
    assert result["public_redistribution_authorized"] is False
    assert (args["work_root"] / intent["intent_id"] / "execution_started.json").is_file()
    assert dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=runner)["status"] == "no_pending_intent"
    assert len(commands) == 3
    assert events.count("reserve") == events.count("release") == 2


def test_dry_refusal_is_retryable_and_never_consumes_paid_start(tmp_path, monkeypatch):
    args, _, _, _ = _queue(tmp_path, monkeypatch)
    calls = []
    def runner(command, log):
        calls.append(log)
        _adapter(command, {"status": "blocked", "blockers": ["collection_capacity_insufficient"], "provider_mutations_performed": 0})
        return 2
    for _ in range(2):
        result = dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=runner)
        assert result["status"] == "blocked_before_provider"
        assert result["provider_mutation_performed"] is False
    assert len(set(calls)) == 2
    assert not list(args["work_root"].glob("*/execution_started.json"))


def test_missing_approval_never_prepares_or_invokes_allocator(tmp_path, monkeypatch):
    args, intent, approval, _ = _queue(tmp_path, monkeypatch)
    approval.unlink()
    monkeypatch.setattr(dispatcher, "prepare_g1_team_policy", lambda **kwargs: pytest.fail("unapproved preparation"))
    result = dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=lambda *_: pytest.fail("unapproved allocator"))
    assert result["status"] == "awaiting_operator_approval"
    assert result["pending_approval_intent_ids"] == [intent["intent_id"]]


def test_revocation_after_fresh_spend_guard_prevents_paid_start(tmp_path, monkeypatch):
    args, _, approval, _ = _queue(tmp_path, monkeypatch)
    def refresh(_):
        approval.unlink()
        return []
    def runner(command, _):
        assert "--execute" not in command
        _adapter(command, {"status": "dry_run_ready", "provider_mutations_performed": 0})
        return 0
    args["admission_refresher"] = refresh
    result = dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=runner)
    assert result["status"] == "blocked_before_provider"
    assert not list(args["work_root"].glob("*/execution_started.json"))


def test_interrupted_paid_command_requires_reconciliation_and_never_restarts(tmp_path, monkeypatch):
    args, _, _, _ = _queue(tmp_path, monkeypatch)
    commands = []
    def runner(command, _):
        commands.append(command)
        if "--execute" in command:
            raise RuntimeError("hermetic caller interruption")
        _adapter(command, {"status": "dry_run_ready", "provider_mutations_performed": 0})
        return 0
    with pytest.raises(RuntimeError):
        dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=runner)
    result = dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=runner)
    assert result["status"] == "awaiting_exact_attempt_reconciliation"
    assert len(commands) == 2


def test_absent_paid_result_is_unproven_not_terminal_or_relaunchable(tmp_path, monkeypatch):
    args, _, _, _ = _queue(tmp_path, monkeypatch)
    commands = []
    def runner(command, _):
        commands.append(command)
        if "--execute" not in command:
            _adapter(command, {"status": "dry_run_ready", "provider_mutations_performed": 0})
        return 0
    result = dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=runner)
    assert result["provider_mutation_unproven"] is True
    assert result["global_provider_zero_verified"] is False
    assert not list(args["work_root"].glob("*/dispatch_final.json"))
    assert dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=runner)["status"] == "awaiting_exact_attempt_reconciliation"
    assert len(commands) == 2


def test_expired_intent_is_terminal_before_preparation_or_allocator(tmp_path, monkeypatch):
    args, intent, _, _ = _queue(tmp_path, monkeypatch)
    monkeypatch.setattr(dispatcher.time, "time", lambda: intent["request"]["authorization"]["expires_at_epoch"])
    monkeypatch.setattr(dispatcher, "prepare_g1_team_policy", lambda **kwargs: pytest.fail("expired bundle preparation"))
    result = dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=lambda *_: pytest.fail("expired allocator"))
    assert result["status"] == "authorization_expired_before_provider"
    assert result["provider_mutation_performed"] is False
    assert dispatcher.dispatch_one_g1_team_policy(**args)["status"] == "no_pending_intent"


def test_preparation_capacity_refusal_never_reaches_builder_or_allocator(tmp_path, monkeypatch):
    args, _, _, _ = _queue(tmp_path, monkeypatch)
    def refuse(*args, **kwargs):
        raise dispatcher.ControlPlaneDiskBudgetError("hermetic insufficient headroom")
    monkeypatch.setattr(dispatcher, "reserve_control_plane_disk", refuse)
    monkeypatch.setattr(dispatcher, "prepare_g1_team_policy", lambda **kwargs: pytest.fail("unreserved builder"))
    result = dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=lambda *_: pytest.fail("unreserved allocator"))
    assert result["status"] == "blocked_before_provider"
    assert not list(args["work_root"].glob("*/execution_started.json"))


def test_held_intent_does_not_wait_or_call_allocator(tmp_path, monkeypatch):
    import fcntl
    import os
    args, intent, _, _ = _queue(tmp_path, monkeypatch)
    locks = args["work_root"] / ".intent-locks"
    locks.mkdir(parents=True)
    descriptor = os.open(locks / (intent["intent_id"] + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=lambda *_: pytest.fail("held intent allocated twice"))
        assert result["status"] == "other_intent_active"
        assert result["held_intent_ids"] == [intent["intent_id"]]
    finally:
        os.close(descriptor)


def test_prior_release_preparation_is_preserved_and_terminal_without_builder(tmp_path, monkeypatch):
    args, intent, _, _ = _queue(tmp_path, monkeypatch)
    directory = args["work_root"] / intent["intent_id"]
    directory.mkdir(parents=True)
    old = {"intent_digest": intent["intent_digest"], "implementation_commit": "b" * 40}
    old["preparation_digest"] = digest(old, digest_field="preparation_digest")
    path = directory / "preparation.json"
    path.write_text(json.dumps(old))
    original = path.read_bytes()
    monkeypatch.setattr(dispatcher, "prepare_g1_team_policy", lambda **kwargs: pytest.fail("stale bundle replaced"))
    result = dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=lambda *_: pytest.fail("stale allocation"))
    assert result["status"] == "superseded_before_provider_by_release"
    assert result["provider_mutation_performed"] is False
    assert path.read_bytes() == original
    assert dispatcher.dispatch_one_g1_team_policy(**args)["status"] == "no_pending_intent"


def test_malformed_authorization_has_typed_refusal(tmp_path, monkeypatch):
    args, intent, _, _ = _queue(tmp_path, monkeypatch)
    intent["request"]["authorization"] = ["not an authority object"]
    intent["intent_digest"] = digest(intent, digest_field="intent_digest")
    path = args["queue_root"] / intent["intent_id"] / "intent.json"
    path.chmod(0o600)
    path.write_text(json.dumps(intent))
    with pytest.raises(ValueError, match="dispatch_intent_invalid"):
        dispatcher.dispatch_one_g1_team_policy(**args, execute=True, allocator_runner=lambda *_: pytest.fail("malformed allocator"))


def test_held_first_intent_does_not_starve_another_ready_choice(tmp_path, monkeypatch):
    import fcntl
    import os
    args, intent, _, _ = _queue(tmp_path, monkeypatch)
    held = json.loads(json.dumps(intent))
    held["intent_id"] = "g1-team-policy-" + "0" * 64
    held["intent_digest"] = digest(held, digest_field="intent_digest")
    path = args["queue_root"] / held["intent_id"] / "intent.json"
    path.parent.mkdir()
    path.write_text(json.dumps(held))
    locks = args["work_root"] / ".intent-locks"
    locks.mkdir(parents=True)
    descriptor = os.open(locks / (held["intent_id"] + ".lock"), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def runner(command, _):
            _adapter(command, {"status": "dry_run_ready", "provider_mutations_performed": 0})
            return 0
        result = dispatcher.dispatch_one_g1_team_policy(**args, allocator_runner=runner)
        assert result["status"] == "dry_run_ready"
        assert result["intent_id"] == intent["intent_id"]
    finally:
        os.close(descriptor)
