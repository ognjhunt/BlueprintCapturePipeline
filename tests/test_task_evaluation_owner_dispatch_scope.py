"""Persistent owner mode cannot consume legacy queue rows or authorize a stripped setup."""
import json
from pathlib import Path
import pytest

from blueprint_pipeline import task_evaluation_owner_dispatch_scope as scope
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_launch_scope_keeps_legacy_and_unknown_rows_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv(scope.SCOPE_ENV, "persistent_owner_only")
    pending, profiles = tmp_path / "queue/pending", tmp_path / "profiles"
    legacy = write(pending / "legacy.json", {"launch_profile_id": "old", "launch_profile_digest": "sha256:" + "a" * 64})
    write(profiles / "old.json", {"profile_digest": "sha256:" + "a" * 64})
    owned = write(pending / "owned.json", {"launch_profile_id": "new", "launch_profile_digest": "sha256:" + "b" * 64})
    write(profiles / "new.json", {"profile_digest": "sha256:" + "b" * 64,
        "scene_intent_digest": "sha256:" + "c" * 64, "scene_attempt_id": "attempt1", "scene_attempt_binding": {}})
    unknown = write(pending / "unknown.json", {"launch_profile_id": "missing"})
    before = {p: p.read_bytes() for p in (legacy, owned, unknown)}
    assert scope.owner_launch_sources([legacy, owned, unknown], profiles) == [owned]
    assert all(p.read_bytes() == content for p, content in before.items())
    with pytest.raises(ValueError, match="persistent_owner_binding_required"):
        scope.require_owner_dispatch_scope({})


def test_real_launch_queue_does_not_claim_legacy_in_owner_mode(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_launch_dispatcher import process_launch_queue
    monkeypatch.setenv(scope.SCOPE_ENV, "persistent_owner_only")
    source = write(tmp_path / "queue/pending/legacy.json", {"launch_profile_id": "old"})
    before = source.read_bytes()
    result = process_launch_queue(queue_root=tmp_path / "queue", profile_dir=tmp_path / "profiles",
        state_root=tmp_path / "runs", execute=True, execute_launch_id="legacy",
        allocator_runner=lambda _: pytest.fail("legacy request allocated"))
    assert result["processed_count"] == 0 and source.read_bytes() == before


def test_policy_scope_does_not_claim_legacy_or_corrupt_owned_envelope(tmp_path, monkeypatch):
    from blueprint_pipeline.task_evaluation_policy_canary_dispatcher import process_policy_canary_dispatch_queue
    monkeypatch.setenv(scope.SCOPE_ENV, "persistent_owner_only")
    queue = tmp_path / "queue"
    for state in ("pending", "processing", "completed", "blocked"):
        (queue / state).mkdir(parents=True)
    legacy = write(queue / "pending/old.json", {})
    corrupt = write(queue / "pending/corrupt.json", {"scene_intent_digest": "sha256:" + "c" * 64})
    process_policy_canary_dispatch_queue(dispatch_queue_root=queue, execution_setup_root=tmp_path / "setups",
        dispatch_root=tmp_path / "runs", implementation_commit="d" * 40, execute=True,
        provider_zero_collector=lambda **_: pytest.fail("unexpected provider call"))
    assert legacy.is_file() and corrupt.is_file()
    assert list((queue / "blocked").iterdir()) == []
    envelope = {"scene_intent_digest": "sha256:" + "c" * 64}
    envelope["envelope_digest"] = canonical_digest(envelope, digest_field="envelope_digest")
    owned = write(queue / "pending/owned.json", envelope)
    assert scope.owner_policy_sources([legacy, corrupt, owned]) == [owned]


def test_invalid_scope_refuses_instead_of_enabling_legacy(monkeypatch):
    monkeypatch.setenv(scope.SCOPE_ENV, "typo")
    with pytest.raises(ValueError, match="dispatch_owner_scope_invalid"):
        scope.owner_launch_sources([], Path("/unused"))
