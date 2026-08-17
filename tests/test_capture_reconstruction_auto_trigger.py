"""Auto-trigger tests: a staged capture enqueues its own reconstruction.

These bind the dispatcher to the bytes a real iPhone bundle actually stages, so
the trigger cannot pass on a synthetic shape the device never produces.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.capture_reconstruction_launch_dispatcher import (
    CaptureReconstructionLaunchError,
    compute_capture_digest,
    enqueue_capture_reconstruction,
    resolve_capture_identity,
    validate_launch_request,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _sha(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _policy_file(root: Path) -> dict:
    policy = {
        "schema_version": "site_task_reconstruction_policy.v1",
        "policy_id": "policy-warehouse-a",
        "site_id": "site-warehouse-a",
        "task_id": "task-shelf-restock",
        "selector": "profile_bound_quality_filter",
        "parameters": {
            "allowed_tracking_states": ["normal"],
            "require_pose_assisted_eligible": True,
            "exclude_relocalization_events": True,
        },
        "rights_authority": {
            "rights_profile": "operator_authorized_commercial",
            "rights_evidence_digest": _sha("rights"),
        },
        "arms": ["postshot-primary"],
        "max_spend_usd": 10.0,
        "hard_ttl_seconds": 5400,
        "authority_id": "authority-founder-20260817",
    }
    policy["policy_digest"] = canonical_digest(policy, digest_field="policy_digest")
    root.mkdir(parents=True, exist_ok=True)
    (root / "policy-warehouse-a.json").write_text(json.dumps(policy), encoding="utf-8")
    return policy


def _capture_root(tmp_path: Path, *, site_id: str = "site-warehouse-a",
                  task_id: str | None = "task-shelf-restock") -> Path:
    """Materialize the subset of a staged iPhone bundle the trigger reads."""
    capture_root = tmp_path / "captures" / "capture-live-001"
    raw = capture_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    (raw / "video.mov").write_bytes(b"not-a-real-encode-but-real-bytes")
    (raw / "manifest.json").write_text(
        json.dumps({"capture_schema_version": "3.2.0"}), encoding="utf-8"
    )
    (raw / "capture_upload_complete.json").write_text(
        json.dumps({"capture_upload_completed_at": "2026-08-17T00:00:00Z"}),
        encoding="utf-8",
    )
    manifest = {"manifest_digest": _sha("candidate-manifest")}
    (raw / "downstream_candidate_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    context = {"site_id": site_id}
    if task_id is not None:
        context["task_id"] = task_id
    (raw / "capture_context.json").write_text(json.dumps(context), encoding="utf-8")

    # hashes.json is written last on device, covering every other raw file.
    artifacts = {}
    for path in sorted(raw.rglob("*")):
        if path.is_file() and path.name != "hashes.json":
            artifacts[str(path.relative_to(raw))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    (raw / "hashes.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )
    return capture_root


def _payload() -> dict:
    return {
        "bucket": "blueprint-captures",
        "scene_id": "scene-live-001",
        "capture_id": "capture-live-001",
        "raw_prefix_uri": "gs://blueprint-captures/scenes/scene-live-001/captures/capture-live-001/raw",
    }


# --------------------------------------------------------------------------
# Capture digest binds every raw byte
# --------------------------------------------------------------------------


def test_capture_digest_is_derived_from_the_validated_hash_manifest(tmp_path: Path) -> None:
    raw = _capture_root(tmp_path) / "raw"
    digest = compute_capture_digest(raw)
    assert digest.startswith("sha256:")
    assert compute_capture_digest(raw) == digest


def test_capture_digest_changes_when_any_raw_byte_changes(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)
    raw = capture_root / "raw"
    before = compute_capture_digest(raw)
    (raw / "video.mov").write_bytes(b"tampered")
    hashes = json.loads((raw / "hashes.json").read_text(encoding="utf-8"))
    hashes["artifacts"]["video.mov"] = hashlib.sha256(b"tampered").hexdigest()
    (raw / "hashes.json").write_text(json.dumps(hashes), encoding="utf-8")
    assert compute_capture_digest(raw) != before


def test_capture_digest_refuses_a_hash_manifest_that_disagrees_with_bytes(
    tmp_path: Path,
) -> None:
    """Output tamper: bytes changed, hash manifest left stale."""
    raw = _capture_root(tmp_path) / "raw"
    (raw / "video.mov").write_bytes(b"tampered-without-updating-hashes")
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        compute_capture_digest(raw)
    assert "hash_mismatch" in str(excinfo.value)


def test_capture_digest_refuses_an_uncovered_raw_file(tmp_path: Path) -> None:
    """Smuggled file: present on disk, absent from the hash manifest."""
    raw = _capture_root(tmp_path) / "raw"
    (raw / "extra_unlisted_payload.bin").write_bytes(b"smuggled")
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        compute_capture_digest(raw)
    assert "hash_coverage_missing" in str(excinfo.value)


def test_capture_digest_refuses_a_missing_hash_manifest(tmp_path: Path) -> None:
    raw = _capture_root(tmp_path) / "raw"
    (raw / "hashes.json").unlink()
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        compute_capture_digest(raw)
    assert "hash_manifest" in str(excinfo.value)


# --------------------------------------------------------------------------
# Identity resolution never guesses
# --------------------------------------------------------------------------


def test_identity_resolves_site_and_task_from_capture_context(tmp_path: Path) -> None:
    identity = resolve_capture_identity(
        capture_root=_capture_root(tmp_path), payload=_payload()
    )
    assert identity["site_id"] == "site-warehouse-a"
    assert identity["task_id"] == "task-shelf-restock"
    assert identity["candidate_manifest_digest"] == _sha("candidate-manifest")


def test_identity_reports_a_missing_task_rather_than_refusing(tmp_path: Path) -> None:
    """A one-off personal walk has no task; identity states that, it does not judge."""
    identity = resolve_capture_identity(
        capture_root=_capture_root(tmp_path, task_id=None), payload=_payload()
    )
    assert identity["task_id"] == ""
    assert identity["capture_id"] == "capture-live-001"


def test_an_untasked_capture_still_abstains_without_authority(tmp_path: Path) -> None:
    """The gate moved to policy resolution; it did not disappear."""
    _policy_file(tmp_path / "policies")
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        enqueue_capture_reconstruction(
            capture_root=_capture_root(tmp_path, task_id=None),
            payload=_payload(),
            policy_root=tmp_path / "policies",
            queue_root=tmp_path / "queue",
            source_commit_sha="a" * 40,
            requested_at="2026-08-17T00:00:00Z",
        )
    assert "site_task_reconstruction_policy_absent" in str(excinfo.value)
    assert not (tmp_path / "queue" / "pending").exists()


def test_identity_abstains_without_the_upload_complete_marker(tmp_path: Path) -> None:
    """An interrupted upload must not look like a finished capture."""
    capture_root = _capture_root(tmp_path)
    (capture_root / "raw" / "capture_upload_complete.json").unlink()
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        resolve_capture_identity(capture_root=capture_root, payload=_payload())
    assert "capture_upload_complete" in str(excinfo.value)


# --------------------------------------------------------------------------
# End-to-end trigger
# --------------------------------------------------------------------------


def test_staged_capture_enqueues_exactly_one_launch(tmp_path: Path) -> None:
    _policy_file(tmp_path / "policies")
    capture_root = _capture_root(tmp_path)
    kwargs = dict(
        capture_root=capture_root,
        payload=_payload(),
        policy_root=tmp_path / "policies",
        queue_root=tmp_path / "queue",
        source_commit_sha="a" * 40,
        requested_at="2026-08-17T00:00:00Z",
    )
    first = enqueue_capture_reconstruction(**kwargs)
    second = enqueue_capture_reconstruction(**kwargs)

    assert first["status"] == "queued"
    assert second["already_exists"] is True
    assert len(list((tmp_path / "queue" / "pending").glob("*.json"))) == 1
    assert first["provider_mutation_performed"] is False

    queued = json.loads(Path(first["queue_path"]).read_text(encoding="utf-8"))
    assert validate_launch_request(queued) == []
    assert queued["capture_id"] == "capture-live-001"
    assert queued["max_spend_usd"] == 10.0


def test_unregistered_site_abstains_without_enqueueing(tmp_path: Path) -> None:
    _policy_file(tmp_path / "policies")
    capture_root = _capture_root(tmp_path, site_id="site-never-registered")
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        enqueue_capture_reconstruction(
            capture_root=capture_root,
            payload=_payload(),
            policy_root=tmp_path / "policies",
            queue_root=tmp_path / "queue",
            source_commit_sha="a" * 40,
            requested_at="2026-08-17T00:00:00Z",
        )
    assert "site_task_reconstruction_policy_absent" in str(excinfo.value)
    assert not (tmp_path / "queue" / "pending").exists()


def _handoff() -> object:
    from blueprint_pipeline.pubsub_handoff_listener import HandoffMessage

    return HandoffMessage(
        bucket="blueprint-captures",
        scene_id="scene-live-001",
        capture_id="capture-live-001",
        raw_prefix_uri="gs://blueprint-captures/scenes/scene-live-001/captures/capture-live-001/raw",
        pipeline_handoff_uri=None,
    )


def test_listener_enqueues_automatically_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trigger must fire from the listener, not only from a direct call."""
    from blueprint_pipeline import pubsub_handoff_listener as listener

    _policy_file(tmp_path / "policies")
    capture_root = _capture_root(tmp_path)
    monkeypatch.setenv(
        listener.RECONSTRUCTION_POLICY_ROOT_ENV, str(tmp_path / "policies")
    )
    monkeypatch.setenv(
        listener.RECONSTRUCTION_QUEUE_ROOT_ENV, str(tmp_path / "queue")
    )

    # Two deliveries of the same message, seconds apart, as Pub/Sub really does.
    first = listener._enqueue_capture_reconstruction_if_configured(
        handoff=_handoff(), capture_root=capture_root
    )
    second = listener._enqueue_capture_reconstruction_if_configured(
        handoff=_handoff(), capture_root=capture_root
    )

    assert first["enqueued"] is True, first
    assert second["enqueued"] is False, second
    assert second["already_exists"] is True
    assert len(list((tmp_path / "queue" / "pending").glob("*.json"))) == 1


def test_listener_is_inert_when_reconstruction_is_not_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import pubsub_handoff_listener as listener

    monkeypatch.delenv(listener.RECONSTRUCTION_POLICY_ROOT_ENV, raising=False)
    monkeypatch.delenv(listener.RECONSTRUCTION_QUEUE_ROOT_ENV, raising=False)
    result = listener._enqueue_capture_reconstruction_if_configured(
        handoff=_handoff(), capture_root=_capture_root(tmp_path)
    )
    assert result == {"status": "not_configured", "enqueued": False}


def test_listener_abstention_does_not_raise_and_dead_letter_the_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unregistered site is a handled outcome, not a delivery failure."""
    from blueprint_pipeline import pubsub_handoff_listener as listener

    _policy_file(tmp_path / "policies")
    monkeypatch.setenv(
        listener.RECONSTRUCTION_POLICY_ROOT_ENV, str(tmp_path / "policies")
    )
    monkeypatch.setenv(
        listener.RECONSTRUCTION_QUEUE_ROOT_ENV, str(tmp_path / "queue")
    )
    result = listener._enqueue_capture_reconstruction_if_configured(
        handoff=_handoff(),
        capture_root=_capture_root(tmp_path, site_id="site-never-registered"),
    )
    assert result["status"] == "abstained"
    assert result["enqueued"] is False
    assert any("policy_absent" in blocker for blocker in result["blockers"])


def _bind_capture_policy(root: Path, capture_id: str, **overrides) -> dict:
    from blueprint_pipeline.capture_reconstruction_launch_dispatcher import (
        build_capture_scoped_policy,
    )

    kwargs = dict(
        capture_id=capture_id,
        site_id="site-my-living-room",
        task_id="task-one-off-walk",
        selector="profile_bound_quality_filter",
        parameters={
            "allowed_tracking_states": ["normal"],
            "require_pose_assisted_eligible": True,
            "exclude_relocalization_events": True,
        },
        rights_profile="operator_authorized_personal",
        rights_evidence_digest=_sha("rights"),
        arms=["postshot-primary"],
        max_spend_usd=50.0,
        hard_ttl_seconds=5400,
        authority_id="authority-founder-20260817",
    )
    kwargs.update(overrides)
    policy = build_capture_scoped_policy(**kwargs)
    root.mkdir(parents=True, exist_ok=True)
    (root / f"capture-{capture_id}.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    return policy


def test_an_untasked_capture_runs_once_an_operator_binds_it(tmp_path: Path) -> None:
    """The living-room case: no site, no task, no job — governed by name."""
    capture_root = _capture_root(tmp_path, site_id="", task_id=None)
    _bind_capture_policy(tmp_path / "policies", "capture-live-001")

    receipt = enqueue_capture_reconstruction(
        capture_root=capture_root,
        payload=_payload(),
        policy_root=tmp_path / "policies",
        queue_root=tmp_path / "queue",
        source_commit_sha="a" * 40,
        requested_at="2026-08-17T00:00:00Z",
    )
    assert receipt["status"] == "queued"
    queued = json.loads(Path(receipt["queue_path"]).read_text(encoding="utf-8"))
    assert validate_launch_request(queued) == []
    assert queued["max_spend_usd"] == 50.0


def test_a_capture_scoped_policy_governs_only_its_own_capture(tmp_path: Path) -> None:
    """Binding one capture must not silently authorize the next one."""
    _bind_capture_policy(tmp_path / "policies", "capture-live-001")
    other = _capture_root(tmp_path / "second", site_id="", task_id=None)
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        enqueue_capture_reconstruction(
            capture_root=other,
            payload={**_payload(), "capture_id": "capture-live-002"},
            policy_root=tmp_path / "policies",
            queue_root=tmp_path / "queue",
            source_commit_sha="a" * 40,
            requested_at="2026-08-17T00:00:00Z",
        )
    assert "site_task_reconstruction_policy_absent" in str(excinfo.value)


def test_a_capture_scoped_policy_does_not_leak_to_the_same_site_and_task(
    tmp_path: Path,
) -> None:
    """Sharing a site and task is not the same as being the bound capture."""
    _bind_capture_policy(
        tmp_path / "policies",
        "capture-live-001",
        site_id="site-warehouse-a",
        task_id="task-shelf-restock",
    )
    other = _capture_root(tmp_path / "second")  # same site/task, different capture
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        enqueue_capture_reconstruction(
            capture_root=other,
            payload={**_payload(), "capture_id": "capture-live-002"},
            policy_root=tmp_path / "policies",
            queue_root=tmp_path / "queue",
            source_commit_sha="a" * 40,
            requested_at="2026-08-17T00:00:00Z",
        )
    assert "site_task_reconstruction_policy_absent" in str(excinfo.value)


def test_tampered_capture_never_reaches_the_queue(tmp_path: Path) -> None:
    _policy_file(tmp_path / "policies")
    capture_root = _capture_root(tmp_path)
    (capture_root / "raw" / "video.mov").write_bytes(b"tampered-after-hashing")
    with pytest.raises(CaptureReconstructionLaunchError):
        enqueue_capture_reconstruction(
            capture_root=capture_root,
            payload=_payload(),
            policy_root=tmp_path / "policies",
            queue_root=tmp_path / "queue",
            source_commit_sha="a" * 40,
            requested_at="2026-08-17T00:00:00Z",
        )
    assert not (tmp_path / "queue" / "pending").exists()
