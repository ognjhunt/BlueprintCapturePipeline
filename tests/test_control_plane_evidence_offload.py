"""Sealed run evidence moves to the artifact store behind a digest-bound pointer, and comes back."""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from blueprint_pipeline.control_plane_evidence_offload import (
    ABANDONED_TERMINAL_RECEIPT,
    EXECUTE_ACK,
    POINTER_SUFFIX,
    ControlPlaneEvidenceOffloadError,
    apply_evidence_offload,
    build_evidence_offload_manifest,
    restore_offloaded_evidence,
)
from tests.test_task_evaluation_configured_scene_object_store import (
    _ContentAddressedClient,
)


BUCKET = "blueprint-production-inputs"


def _unclassified(*_args, **_kwargs) -> None:
    return None


def _run(root: Path, name: str, *, receipt: str | None, age: float, now: float) -> Path:
    directory = root / name
    (directory / "episodes").mkdir(parents=True)
    (directory / "episodes" / "frame.bin").write_bytes(os.urandom(2048))
    (directory / "status_events.jsonl").write_text('{"stage":"done"}\n', encoding="utf-8")
    if receipt:
        (directory / receipt).write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    stamp = now - age
    for path in [directory, *directory.rglob("*")]:
        os.utime(path, (stamp, stamp))
    return directory


def test_manifest_lists_only_sealed_runs_past_the_hot_window(tmp_path: Path) -> None:
    root = tmp_path / "launch-runs"
    root.mkdir()
    now = 1_000_000.0
    sealed = _run(root, "run-sealed", receipt="dispatch_receipt.json", age=20 * 86400, now=now)
    _run(root, "run-hot", receipt="launch_receipt.json", age=2 * 86400, now=now)
    _run(root, "run-open", receipt=None, age=30 * 86400, now=now)
    done = _run(root, "run-done", receipt="dispatch_receipt.json", age=30 * 86400, now=now)
    (root / f"run-done{POINTER_SUFFIX}").write_text("{}", encoding="utf-8")
    (root / "stray.txt").write_text("x", encoding="utf-8")

    manifest = build_evidence_offload_manifest(
        evidence_roots=[root], hot_window_seconds=14 * 86400, now=lambda: now, classifier=_unclassified
    )

    assert [row["name"] for row in manifest["candidates"]] == [sealed.name]
    assert manifest["candidates"][0]["terminal_receipt"] == "dispatch_receipt.json"
    assert manifest["candidates"][0]["file_count"] == 3
    assert manifest["retained_counts"] == {
        "active_or_unsealed": 1,
        "hot": 1,
        "already_offloaded": 1,
        "unsafe": 1,
    }
    assert manifest["evidence_hot_roots_scanned"] is False
    assert done.is_dir()
    # The real classifier refuses anything that is not sealed run evidence.
    with pytest.raises(ValueError, match="control_plane_evidence_offload_root_class:unclassified"):
        build_evidence_offload_manifest(evidence_roots=[root], now=lambda: now)
    with pytest.raises(ValueError, match="control_plane_evidence_offload_root_class:evidence_hot"):
        build_evidence_offload_manifest(
            evidence_roots=["/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard"],
            now=lambda: now,
        )


def test_local_write_during_archive_publication_prevents_eviction(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    directory = _run(root, "run-1", receipt="dispatch_receipt.json", age=100, now=1000)
    manifest = build_evidence_offload_manifest(evidence_roots=[root], hot_window_seconds=0,
        now=lambda: 1000, classifier=_unclassified)
    client = _ContentAddressedClient()
    def publisher(**kwargs):
        result = store.publish_configured_scene_artifact(**kwargs, client=client, bucket=BUCKET)
        (directory / "episodes" / "frame.bin").write_bytes(b"new evidence")
        return result
    result = apply_evidence_offload(manifest, ack=EXECUTE_ACK, publisher=publisher)
    assert result["offloaded_count"] == 0
    assert result["skipped"] == [{"name": "run-1", "reason": "candidate_changed_during_archive"}]
    assert (directory / "episodes" / "frame.bin").read_bytes() == b"new evidence"
    assert not (root / ("run-1" + POINTER_SUFFIX)).exists()


def test_offload_publishes_verifies_points_then_removes_and_restore_round_trips(
    tmp_path: Path,
) -> None:
    root = tmp_path / "launch-runs"
    root.mkdir()
    now = 2_000_000.0
    directory = _run(root, "run-1", receipt="dispatch_receipt.json", age=30 * 86400, now=now)
    original = {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }
    client = _ContentAddressedClient()
    publisher = functools.partial(
        store.publish_configured_scene_artifact, client=client, bucket=BUCKET
    )
    manifest = build_evidence_offload_manifest(
        evidence_roots=[root], hot_window_seconds=14 * 86400, now=lambda: now, classifier=_unclassified
    )

    with pytest.raises(
        ControlPlaneEvidenceOffloadError,
        match="control_plane_evidence_offload_apply_not_authorized",
    ):
        apply_evidence_offload(manifest, ack="wrong", publisher=publisher, now=lambda: now)
    receipt = apply_evidence_offload(manifest, ack=EXECUTE_ACK, publisher=publisher, now=lambda: now)

    assert receipt["offloaded_count"] == 1 and receipt["skipped"] == []
    assert receipt["evidence_deleted"] is False
    assert not directory.exists()
    pointer_path = root / f"run-1{POINTER_SUFFIX}"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert pointer["status"] == "offloaded"
    assert pointer["directory"] == "run-1"
    assert pointer["terminal_receipt"] == "dispatch_receipt.json"
    assert pointer["uri"].endswith(f"/artifacts/control-plane-evidence/sha256/{pointer['digest'].removeprefix('sha256:')}/{Path(receipt['offloaded'][0]['uri']).name}")
    assert {row["relative_path"] for row in pointer["members"]} == set(original)
    for row in pointer["members"]:
        assert row["sha256"] == "sha256:" + hashlib.sha256(original[row["relative_path"]]).hexdigest()
    assert client.upload_count == 1
    stored = client.objects[(BUCKET, pointer["uri"].split(f"s3://{BUCKET}/", 1)[1])]
    assert hashlib.sha256(stored).hexdigest() == pointer["digest"].removeprefix("sha256:")
    assert not list(root.glob(".run-1.offload-*"))

    def materializer(*, reference, destination, maximum_size_bytes):
        payload = client.objects[(BUCKET, reference["uri"].split(f"s3://{BUCKET}/", 1)[1])]
        assert len(payload) <= maximum_size_bytes
        Path(destination).write_bytes(payload)
        return {"status": "materialized"}

    restored = restore_offloaded_evidence(
        pointer_path=pointer_path, destination=tmp_path / "restored" / "run-1", materializer=materializer
    )
    assert restored["status"] == "restored"
    assert {
        path.relative_to(tmp_path / "restored" / "run-1").as_posix(): path.read_bytes()
        for path in (tmp_path / "restored" / "run-1").rglob("*")
        if path.is_file()
    } == original

    # A second tick sees the pointer and does not offload the run again.
    again = build_evidence_offload_manifest(
        evidence_roots=[root], hot_window_seconds=14 * 86400, now=lambda: now, classifier=_unclassified
    )
    assert again["candidate_count"] == 0
    assert pointer_path.is_file() and not directory.exists()
    # A directory that reappears beside its pointer is never offloaded twice.
    directory.mkdir()
    (directory / "dispatch_receipt.json").write_text("{}", encoding="utf-8")
    twice = build_evidence_offload_manifest(
        evidence_roots=[root], hot_window_seconds=0, now=lambda: now, classifier=_unclassified
    )
    assert twice["candidate_count"] == 0 and twice["retained_counts"]["already_offloaded"] == 1


def test_offload_keeps_the_directory_when_publication_or_verification_fails(
    tmp_path: Path,
) -> None:
    root = tmp_path / "canaries"
    root.mkdir()
    now = 3_000_000.0
    directory = _run(root, "run-2", receipt="dispatch_receipt.json", age=30 * 86400, now=now)
    manifest = build_evidence_offload_manifest(
        evidence_roots=[root], hot_window_seconds=0, now=lambda: now, classifier=_unclassified
    )

    def failing_publisher(**_kwargs):
        raise store.TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_publication_failed"
        )

    failed = apply_evidence_offload(manifest, ack=EXECUTE_ACK, publisher=failing_publisher, now=lambda: now)
    assert failed["offloaded_count"] == 0
    assert failed["skipped"] == [
        {"name": "run-2", "reason": "offload_failed:TaskEvaluationConfiguredSceneObjectStoreError"}
    ]
    assert directory.is_dir() and not (root / f"run-2{POINTER_SUFFIX}").exists()
    assert not list(root.glob(".run-2.offload-*"))

    def lying_publisher(*, path, artifact_kind):
        return {
            "uri": "s3://elsewhere/x",
            "digest": "sha256:" + "0" * 64,
            "size_bytes": Path(path).stat().st_size,
            "full_byte_service_account_readback_passed": True,
        }

    mismatch = apply_evidence_offload(manifest, ack=EXECUTE_ACK, publisher=lying_publisher, now=lambda: now)
    assert mismatch["skipped"] == [
        {"name": "run-2", "reason": "offload_failed:ControlPlaneEvidenceOffloadError"}
    ]
    assert directory.is_dir()

    # A candidate whose seal changed after the dry run is skipped, not offloaded.
    (directory / "dispatch_receipt.json").unlink()
    changed = apply_evidence_offload(
        manifest,
        ack=EXECUTE_ACK,
        publisher=functools.partial(
            store.publish_configured_scene_artifact, client=_ContentAddressedClient(), bucket=BUCKET
        ),
        now=lambda: now,
    )
    assert changed["skipped"] == [{"name": "run-2", "reason": "candidate_changed"}]
    assert directory.is_dir()


def test_unsealed_directory_idle_past_the_abandonment_window_is_sealed_as_abandoned(
    tmp_path: Path,
) -> None:
    """Without a terminal receipt a run directory was retained forever as
    "active".  Twenty-three such directories sat on the production host from
    workers that were superseded or torn down.  Idle past the window they are
    archived like any sealed run; nothing is deleted and restore is unchanged."""
    root = tmp_path / "policy-canaries"
    root.mkdir()
    now = 3_000_000.0
    abandoned = _run(root, "run-abandoned", receipt=None, age=5 * 86400, now=now)
    _run(root, "run-active", receipt=None, age=3600, now=now)

    without = build_evidence_offload_manifest(
        evidence_roots=[root], hot_window_seconds=86400, now=lambda: now, classifier=_unclassified
    )
    assert without["candidates"] == [] and without["retained_counts"]["active_or_unsealed"] == 2
    manifest = build_evidence_offload_manifest(
        evidence_roots=[root],
        hot_window_seconds=86400,
        abandoned_after_seconds=3 * 86400,
        now=lambda: now,
        classifier=_unclassified,
    )
    assert [row["name"] for row in manifest["candidates"]] == ["run-abandoned"]
    assert manifest["candidates"][0]["terminal_receipt"] == ABANDONED_TERMINAL_RECEIPT
    assert manifest["retained_counts"]["active_or_unsealed"] == 1

    client = _ContentAddressedClient()
    publisher = functools.partial(store.publish_configured_scene_artifact, client=client, bucket=BUCKET)
    receipt = apply_evidence_offload(manifest, ack=EXECUTE_ACK, publisher=publisher, now=lambda: now)
    assert receipt["offloaded_count"] == 1 and not abandoned.exists()
    pointer = json.loads((root / f"run-abandoned{POINTER_SUFFIX}").read_text(encoding="utf-8"))
    assert pointer["terminal_receipt"] == ABANDONED_TERMINAL_RECEIPT
    assert (root / "run-active").exists()


def test_an_abandoned_candidate_touched_after_the_dry_run_is_kept(tmp_path: Path) -> None:
    root = tmp_path / "policy-canaries"
    root.mkdir()
    now = 3_000_000.0
    directory = _run(root, "run-abandoned", receipt=None, age=5 * 86400, now=now)
    manifest = build_evidence_offload_manifest(
        evidence_roots=[root],
        hot_window_seconds=86400,
        abandoned_after_seconds=3 * 86400,
        now=lambda: now,
        classifier=_unclassified,
    )
    (directory / "status_events.jsonl").write_text('{"stage":"resumed"}\n', encoding="utf-8")
    client = _ContentAddressedClient()
    publisher = functools.partial(store.publish_configured_scene_artifact, client=client, bucket=BUCKET)

    receipt = apply_evidence_offload(manifest, ack=EXECUTE_ACK, publisher=publisher, now=lambda: now)

    assert receipt["offloaded_count"] == 0
    assert receipt["skipped"] == [{"name": "run-abandoned", "reason": "candidate_changed"}]
    assert directory.exists() and client.upload_count == 0
    with pytest.raises(ControlPlaneEvidenceOffloadError, match="input_invalid"):
        build_evidence_offload_manifest(
            evidence_roots=[root],
            hot_window_seconds=86400,
            abandoned_after_seconds=-1,
            now=lambda: now,
            classifier=_unclassified,
        )
