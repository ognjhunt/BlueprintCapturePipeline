"""Full-chain test: staged iPhone bundle -> queue -> real COLMAP dataset.

This deliberately reuses the canonical V3.2 fixture the pipeline's own
end-to-end test uses, so the dispatcher is proven against the same bytes the
real preparation consumes rather than a shape invented here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.capture_reconstruction_checkpoints import (
    read_checkpoints,
)
from blueprint_pipeline.capture_reconstruction_launch_dispatcher import (
    CaptureReconstructionLaunchError,
    dispatch_launch_request,
    enqueue_capture_reconstruction,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

from .test_canonical_3dgs_pipeline import (  # noqa: F401 - shared canonical fixture
    _canonical_v32_fixture,
    _stub_v32_media,
)

SITE = "site-warehouse-a"
TASK = "task-shelf-restock"


def _sha(seed: str) -> str:
    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _policy(root: Path, ordinals: list[int], manifest_digest: str) -> dict:
    """Register the site/task authority a human would have registered once."""
    policy = {
        "schema_version": "site_task_reconstruction_policy.v1",
        "policy_id": "policy-warehouse-a",
        "site_id": SITE,
        "task_id": TASK,
        "selector": "explicit_encoded_frame_ordinals",
        "parameters": {"encoded_frame_ordinals": ordinals},
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
    (root / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return policy


def _stage_as_uploaded_capture(tmp_path: Path) -> tuple[Path, list[float], str]:
    """Lay the canonical fixture out the way Storage->Pub/Sub actually stages it.

    The device uploads under ``<capture>/raw/``; preparation is handed that raw
    root directly.  Building the layout here keeps the dispatcher honest about
    the shape it will really see.
    """
    capture_root = tmp_path / "captures" / "capture-live-001"
    raw = capture_root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    times = _canonical_v32_fixture(raw)

    (raw / "capture_upload_complete.json").write_text(
        json.dumps({"capture_upload_completed_at": "2026-08-17T00:00:00Z"}),
        encoding="utf-8",
    )
    (raw / "capture_context.json").write_text(
        json.dumps({"site_id": SITE, "task_id": TASK}), encoding="utf-8"
    )

    # hashes.json is written last on device and must cover every other raw file.
    artifacts = {}
    for path in sorted(raw.rglob("*")):
        if path.is_file() and path.name != "hashes.json":
            artifacts[str(path.relative_to(raw)).replace("\\", "/")] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    (raw / "hashes.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )

    manifest = json.loads(
        (raw / "downstream_candidate_manifest.json").read_text(encoding="utf-8")
    )
    return capture_root, times, str(manifest["manifest_digest"])


def _payload() -> dict:
    return {
        "bucket": "blueprint-captures",
        "scene_id": "scene-live-001",
        "capture_id": "capture-live-001",
        "raw_prefix_uri": "gs://blueprint-captures/scenes/scene-live-001/captures/capture-live-001/raw",
    }


def test_uploaded_capture_reaches_a_real_candidate_only_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root, times, manifest_digest = _stage_as_uploaded_capture(tmp_path)
    _stub_v32_media(monkeypatch, times)
    _policy(tmp_path / "policies", [0, 2, 5, 7], manifest_digest)

    receipt = enqueue_capture_reconstruction(
        capture_root=capture_root,
        payload=_payload(),
        policy_root=tmp_path / "policies",
        queue_root=tmp_path / "queue",
        source_commit_sha="a" * 40,
        requested_at="2026-08-17T00:00:00Z",
    )
    assert receipt["status"] == "queued"

    result = dispatch_launch_request(
        queue_path=receipt["queue_path"],
        state_root=tmp_path / "checkpoints",
        derived_root=tmp_path / "derived",
        capture_store_root=capture_root / "raw",
        execute=False,
    )

    assert result["status"] == "prepared_awaiting_paid_authority"
    assert result["preparation_status"] == "training_dataset_ready"
    assert result["dataset_image_count"] >= 3
    assert result["provider_mutation_performed"] is False

    ledger = read_checkpoints(
        state_root=tmp_path / "checkpoints", capture_digest=receipt["capture_digest"]
    )
    assert ledger["recorded_stages"] == [
        "upload_received",
        "intake_validated",
        "queued",
    ]
    assert ledger["paid_stages_completed"] == []


def test_evaluator_hidden_pixels_never_enter_the_trainer_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root, times, manifest_digest = _stage_as_uploaded_capture(tmp_path)
    _stub_v32_media(monkeypatch, times)
    _policy(tmp_path / "policies", [0, 2, 5, 7], manifest_digest)
    receipt = enqueue_capture_reconstruction(
        capture_root=capture_root,
        payload=_payload(),
        policy_root=tmp_path / "policies",
        queue_root=tmp_path / "queue",
        source_commit_sha="a" * 40,
        requested_at="2026-08-17T00:00:00Z",
    )
    dispatch_launch_request(
        queue_path=receipt["queue_path"],
        state_root=tmp_path / "checkpoints",
        derived_root=tmp_path / "derived",
        capture_store_root=capture_root / "raw",
        execute=False,
    )
    ledger = read_checkpoints(
        state_root=tmp_path / "checkpoints", capture_digest=receipt["capture_digest"]
    )
    intake = next(
        row for row in ledger["checkpoints"] if row["stage"] == "intake_validated"
    )
    assert intake["evidence"]["hidden_heldout_pixels_included"] is False


def test_dispatch_resumes_without_redoing_prepared_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root, times, manifest_digest = _stage_as_uploaded_capture(tmp_path)
    _stub_v32_media(monkeypatch, times)
    _policy(tmp_path / "policies", [0, 2, 5, 7], manifest_digest)
    receipt = enqueue_capture_reconstruction(
        capture_root=capture_root,
        payload=_payload(),
        policy_root=tmp_path / "policies",
        queue_root=tmp_path / "queue",
        source_commit_sha="a" * 40,
        requested_at="2026-08-17T00:00:00Z",
    )
    common = dict(
        queue_path=receipt["queue_path"],
        state_root=tmp_path / "checkpoints",
        derived_root=tmp_path / "derived",
        capture_store_root=capture_root / "raw",
        execute=False,
    )
    first = dispatch_launch_request(**common)
    second = dispatch_launch_request(**common)

    assert first["already_prepared"] is False
    assert second["already_prepared"] is True
    ledger = read_checkpoints(
        state_root=tmp_path / "checkpoints", capture_digest=receipt["capture_digest"]
    )
    assert ledger["recorded_stages"].count("intake_validated") == 1


def test_paid_dispatch_refuses_when_no_allocator_is_wired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root, times, manifest_digest = _stage_as_uploaded_capture(tmp_path)
    _stub_v32_media(monkeypatch, times)
    _policy(tmp_path / "policies", [0, 2, 5, 7], manifest_digest)
    receipt = enqueue_capture_reconstruction(
        capture_root=capture_root,
        payload=_payload(),
        policy_root=tmp_path / "policies",
        queue_root=tmp_path / "queue",
        source_commit_sha="a" * 40,
        requested_at="2026-08-17T00:00:00Z",
    )
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        dispatch_launch_request(
            queue_path=receipt["queue_path"],
            state_root=tmp_path / "checkpoints",
            derived_root=tmp_path / "derived",
            capture_store_root=capture_root / "raw",
            execute=True,
        )
    assert "requires_allocator" in str(excinfo.value)
    ledger = read_checkpoints(
        state_root=tmp_path / "checkpoints", capture_digest=receipt["capture_digest"]
    )
    assert ledger["paid_stages_completed"] == []


def _mutate_candidate_manifest(capture_root, mutate) -> str:
    """Rewrite the candidate manifest and re-seal hashes.json around it.

    Re-sealing matters: otherwise the hash guard rejects the capture first and
    the test never reaches the contract it means to exercise.
    """
    raw = capture_root / "raw"
    path = raw / "downstream_candidate_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")

    artifacts = {}
    for item in sorted(raw.rglob("*")):
        if item.is_file() and item.name != "hashes.json":
            artifacts[str(item.relative_to(raw)).replace("\\", "/")] = hashlib.sha256(
                item.read_bytes()
            ).hexdigest()
    (raw / "hashes.json").write_text(
        json.dumps({"artifacts": artifacts}), encoding="utf-8"
    )
    return str(manifest.get("manifest_digest") or "")


def _enqueue_and_dispatch(tmp_path, capture_root, manifest_digest):
    _policy(tmp_path / "policies", [0, 2, 5, 7], manifest_digest)
    receipt = enqueue_capture_reconstruction(
        capture_root=capture_root,
        payload=_payload(),
        policy_root=tmp_path / "policies",
        queue_root=tmp_path / "queue",
        source_commit_sha="a" * 40,
        requested_at="2026-08-17T00:00:00Z",
    )
    return dispatch_launch_request(
        queue_path=receipt["queue_path"],
        state_root=tmp_path / "checkpoints",
        derived_root=tmp_path / "derived",
        capture_store_root=capture_root / "raw",
        execute=False,
    )


def test_pose_drift_in_the_candidate_manifest_abstains(tmp_path, monkeypatch) -> None:
    """Hostile: a pose matrix that is no longer a pose."""
    capture_root, times, _ = _stage_as_uploaded_capture(tmp_path)
    _stub_v32_media(monkeypatch, times)

    def break_pose(manifest):
        manifest["candidates"][0]["T_world_camera"] = [[1.0, 0.0], [0.0, 1.0]]

    digest = _mutate_candidate_manifest(capture_root, break_pose)
    result = _enqueue_and_dispatch(tmp_path, capture_root, digest)
    assert result["status"] == "abstained"
    assert result["blockers"]
    assert result["provider_mutation_performed"] is False


def test_intrinsics_drift_in_the_candidate_manifest_abstains(tmp_path, monkeypatch) -> None:
    """Hostile: intrinsics that cannot describe a camera."""
    capture_root, times, _ = _stage_as_uploaded_capture(tmp_path)
    _stub_v32_media(monkeypatch, times)

    def break_intrinsics(manifest):
        intrinsics = manifest["candidates"][0]["camera_intrinsics"]
        # A negative focal length is not a camera.
        intrinsics["fx"] = -50
        intrinsics["fy"] = -50
        intrinsics["matrix_column_major"] = [-50, 0, 0, 0, -50, 0, 32, 24, 1]

    digest = _mutate_candidate_manifest(capture_root, break_intrinsics)
    result = _enqueue_and_dispatch(tmp_path, capture_root, digest)
    assert result["status"] == "abstained"
    assert result["provider_mutation_performed"] is False


def test_a_stale_app_build_schema_abstains(tmp_path, monkeypatch) -> None:
    """Hostile: a bundle from an older app whose contract is not V3.2."""
    capture_root, times, _ = _stage_as_uploaded_capture(tmp_path)
    _stub_v32_media(monkeypatch, times)

    def stale_schema(manifest):
        manifest["schema_version"] = "downstream_candidate_manifest.v0"

    digest = _mutate_candidate_manifest(capture_root, stale_schema)
    result = _enqueue_and_dispatch(tmp_path, capture_root, digest)
    assert result["status"] == "abstained"
    assert result["provider_mutation_performed"] is False


def _allocator(calls: list, **result):
    def allocate(**kwargs):
        calls.append(kwargs)
        return {
            "status": "execute_ready",
            "admission_digest": _sha("admission"),
            "provider_mutations_performed": 1,
            **result,
        }

    return allocate


def test_execute_allocates_against_the_policy_ceiling_without_asking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production shape: the registered policy is the standing authority."""
    capture_root, times, manifest_digest = _stage_as_uploaded_capture(tmp_path)
    _stub_v32_media(monkeypatch, times)
    _policy(tmp_path / "policies", [0, 2, 5, 7], manifest_digest)
    receipt = enqueue_capture_reconstruction(
        capture_root=capture_root,
        payload=_payload(),
        policy_root=tmp_path / "policies",
        queue_root=tmp_path / "queue",
        source_commit_sha="a" * 40,
        requested_at="2026-08-17T00:00:00Z",
    )
    calls: list = []
    result = dispatch_launch_request(
        queue_path=receipt["queue_path"],
        state_root=tmp_path / "checkpoints",
        derived_root=tmp_path / "derived",
        capture_store_root=capture_root / "raw",
        execute=True,
        allocator=_allocator(calls),
    )

    assert result["status"] == "worker_allocated"
    assert result["provider_mutation_performed"] is True
    assert len(calls) == 1
    # The ceiling travels with the allocation; it is enforced per run.
    assert calls[0]["max_spend_usd"] == 10.0
    assert calls[0]["retry_cap"] == 0
    assert calls[0]["hard_ttl_seconds"] == 5400


def test_a_resumed_dispatch_cannot_allocate_a_second_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_root, times, manifest_digest = _stage_as_uploaded_capture(tmp_path)
    _stub_v32_media(monkeypatch, times)
    _policy(tmp_path / "policies", [0, 2, 5, 7], manifest_digest)
    receipt = enqueue_capture_reconstruction(
        capture_root=capture_root,
        payload=_payload(),
        policy_root=tmp_path / "policies",
        queue_root=tmp_path / "queue",
        source_commit_sha="a" * 40,
        requested_at="2026-08-17T00:00:00Z",
    )
    calls: list = []
    common = dict(
        queue_path=receipt["queue_path"],
        state_root=tmp_path / "checkpoints",
        derived_root=tmp_path / "derived",
        capture_store_root=capture_root / "raw",
        execute=True,
        allocator=_allocator(calls),
    )
    dispatch_launch_request(**common)
    with pytest.raises(Exception) as excinfo:
        dispatch_launch_request(**common)
    assert "paid_stage_already_completed" in str(excinfo.value)
    assert len(calls) == 1


def test_a_failed_allocation_is_still_recorded_as_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An allocator that mutated a provider and then failed still cost money."""
    capture_root, times, manifest_digest = _stage_as_uploaded_capture(tmp_path)
    _stub_v32_media(monkeypatch, times)
    _policy(tmp_path / "policies", [0, 2, 5, 7], manifest_digest)
    receipt = enqueue_capture_reconstruction(
        capture_root=capture_root,
        payload=_payload(),
        policy_root=tmp_path / "policies",
        queue_root=tmp_path / "queue",
        source_commit_sha="a" * 40,
        requested_at="2026-08-17T00:00:00Z",
    )
    calls: list = []
    result = dispatch_launch_request(
        queue_path=receipt["queue_path"],
        state_root=tmp_path / "checkpoints",
        derived_root=tmp_path / "derived",
        capture_store_root=capture_root / "raw",
        execute=True,
        allocator=_allocator(
            calls, status="blocked", blockers=["gpu_capacity_unavailable"]
        ),
    )
    assert result["status"] == "allocation_blocked"
    assert result["blockers"] == ["gpu_capacity_unavailable"]
    ledger = read_checkpoints(
        state_root=tmp_path / "checkpoints", capture_digest=receipt["capture_digest"]
    )
    assert "worker_allocated" in ledger["recorded_stages"]


def test_a_capture_whose_frames_were_removed_abstains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hostile: policy names ordinals the capture does not contain."""
    capture_root, times, manifest_digest = _stage_as_uploaded_capture(tmp_path)
    _stub_v32_media(monkeypatch, times)
    _policy(tmp_path / "policies", [0, 2, 5, 999], manifest_digest)
    receipt = enqueue_capture_reconstruction(
        capture_root=capture_root,
        payload=_payload(),
        policy_root=tmp_path / "policies",
        queue_root=tmp_path / "queue",
        source_commit_sha="a" * 40,
        requested_at="2026-08-17T00:00:00Z",
    )
    result = dispatch_launch_request(
        queue_path=receipt["queue_path"],
        state_root=tmp_path / "checkpoints",
        derived_root=tmp_path / "derived",
        capture_store_root=capture_root / "raw",
        execute=False,
    )

    # Abstention, not a crash: the smallest missing input is named and no
    # intake_validated checkpoint is written, so a later retry can still run.
    assert result["status"] == "abstained"
    assert result["stage"] == "intake_validated"
    assert result["blockers"]
    assert result["provider_mutation_performed"] is False
    ledger = read_checkpoints(
        state_root=tmp_path / "checkpoints", capture_digest=receipt["capture_digest"]
    )
    assert "intake_validated" not in ledger["recorded_stages"]
