from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_storage_gc as gc_module
from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from blueprint_pipeline.control_plane_storage_gc import (
    ControlPlaneStorageGCError,
    DERIVED_ACK,
    EXECUTE_ACK,
    RUN_ACK,
    apply_derived_directory_manifest,
    apply_gc_manifest,
    build_derived_directory_manifest,
    build_gc_manifest,
    main as gc_main,
    run_storage_gc,
)
from blueprint_pipeline.control_plane_storage_pins import write_storage_pin
from tests.test_task_evaluation_configured_scene_object_store import _ContentAddressedClient


def _blob(root, payload: bytes):
    digest = hashlib.sha256(payload).hexdigest()
    path = root / digest
    path.write_bytes(payload)
    os.utime(path, (10, 10))
    return path


def test_gc_only_selects_old_unreferenced_verified_blobs(tmp_path) -> None:
    root = tmp_path / "sha256"
    root.mkdir()
    unreferenced = _blob(root, b"unreferenced")
    linked = _blob(root, b"linked")
    os.link(linked, tmp_path / "projection")
    young = _blob(root, b"young")
    os.utime(young, (95, 95))
    corrupt = root / ("f" * 64)
    corrupt.write_bytes(b"wrong digest")
    os.utime(corrupt, (10, 10))

    manifest = build_gc_manifest(
        content_store_roots=[root],
        minimum_age_seconds=20,
        now=lambda: 100,
    )

    assert manifest["candidate_count"] == 1
    assert manifest["candidate_bytes"] == len(b"unreferenced")
    assert manifest["candidates"][0]["digest"].endswith(unreferenced.name)
    assert manifest["retained_counts"] == {
        "linked": 1,
        "young": 1,
        "unsafe_or_unverified": 1,
    }
    assert manifest["evidence_roots_scanned"] is False


def test_gc_apply_requires_ack_and_rechecks_link_count(tmp_path) -> None:
    root = tmp_path / "sha256"
    root.mkdir()
    candidate = _blob(root, b"candidate")
    manifest = build_gc_manifest(
        content_store_roots=[root], minimum_age_seconds=0, now=lambda: 100
    )
    with pytest.raises(
        ControlPlaneStorageGCError,
        match="control_plane_storage_gc_apply_not_authorized",
    ):
        apply_gc_manifest(manifest, ack="wrong")

    os.link(candidate, tmp_path / "late-projection")
    changed = apply_gc_manifest(manifest, ack=EXECUTE_ACK)
    assert changed["removed_count"] == 0
    assert changed["skipped"] == [
        {"digest": "sha256:" + candidate.name, "reason": "candidate_changed"}
    ]
    assert candidate.exists()


def test_gc_apply_removes_only_manifest_candidates(tmp_path) -> None:
    root = tmp_path / "sha256"
    root.mkdir()
    candidate = _blob(root, b"candidate")
    manifest = build_gc_manifest(
        content_store_roots=[root], minimum_age_seconds=0, now=lambda: 100
    )

    result = apply_gc_manifest(manifest, ack=EXECUTE_ACK)

    assert result["removed_count"] == 1
    assert result["removed_bytes"] == len(b"candidate")
    assert result["evidence_removed"] is False
    assert not candidate.exists()


def test_gc_rejects_non_sha256_or_symlink_roots(tmp_path) -> None:
    unsafe = tmp_path / "evidence"
    unsafe.mkdir()
    with pytest.raises(
        ControlPlaneStorageGCError,
        match="control_plane_storage_gc_root_unsafe",
    ):
        build_gc_manifest(content_store_roots=[unsafe])

    safe = tmp_path / "safe" / "sha256"
    safe.mkdir(parents=True)
    alias = tmp_path / "sha256"
    alias.symlink_to(safe, target_is_directory=True)
    with pytest.raises(
        ControlPlaneStorageGCError,
        match="control_plane_storage_gc_root_unsafe",
    ):
        build_gc_manifest(content_store_roots=[alias])



def _noclass(*_args, **_kwargs) -> None:
    return None


def _derived(root: Path, name: str, *, age: float, now: float) -> Path:
    directory = root / name
    directory.mkdir()
    payload = directory / "x.bin"
    payload.write_bytes(b"data" * 10)
    stamp = now - age
    os.utime(payload, (stamp, stamp))
    os.utime(directory, (stamp, stamp))
    return directory


def test_derived_directories_retire_only_when_unpinned_unqueued_and_idle(tmp_path) -> None:
    root = tmp_path / "prepared-references"
    (root / "content-addressed" / "sha256").mkdir(parents=True)
    now = 10_000_000.0
    idle = _derived(root, "prep-idle", age=10 * 86400, now=now)
    pinned = _derived(root, "prep-pinned", age=10 * 86400, now=now)
    queued = _derived(root, "prep-queued", age=10 * 86400, now=now)
    young = _derived(root, "prep-young", age=86400, now=now)
    pins = tmp_path / "pins"
    write_storage_pin(
        pins_root=pins, kind="preparation", owner_id="prep-pinned", paths=[pinned], now=lambda: now
    )
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    (queue / "pending" / "message.json").write_text(
        json.dumps({"request": {"preparation_id": "prep-queued"}}), encoding="utf-8"
    )

    manifest = build_derived_directory_manifest(
        derived_roots=[root],
        pins_root=pins,
        queue_roots=[queue],
        minimum_age_seconds=7 * 86400,
        now=lambda: now,
        classifier=_noclass,
    )

    assert [row["name"] for row in manifest["candidates"]] == ["prep-idle"]
    assert manifest["retained_counts"] == {
        "pinned": 1,
        "queue_referenced": 1,
        "young": 1,
        "unsafe": 0,
    }
    assert manifest["evidence_roots_scanned"] is False
    with pytest.raises(ControlPlaneStorageGCError, match="apply_not_authorized"):
        apply_derived_directory_manifest(
            manifest, ack="wrong", pins_root=pins, queue_roots=[queue], classifier=_noclass
        )
    receipt = apply_derived_directory_manifest(
        manifest,
        ack=DERIVED_ACK,
        pins_root=pins,
        queue_roots=[queue],
        now=lambda: now,
        classifier=_noclass,
    )
    assert receipt["removed"] == [{"name": "prep-idle", "size_bytes": 40}]
    assert receipt["evidence_removed"] is False
    assert not idle.exists()
    for kept in (pinned, queued, young, root / "content-addressed"):
        assert kept.exists()
    # The production classifier refuses roots that are not cache class.
    with pytest.raises(ValueError, match="control_plane_storage_gc_derived_root_class:unclassified"):
        build_derived_directory_manifest(
            derived_roots=[root], pins_root=pins, queue_roots=[queue], now=lambda: now
        )
    with pytest.raises(ValueError, match="control_plane_storage_gc_derived_root_class:evidence_hot"):
        build_derived_directory_manifest(
            derived_roots=["/var/lib/blueprint/pipeline-control-plane/gpu_spend_guard"],
            pins_root=pins,
            queue_roots=[],
            now=lambda: now,
        )


def test_apply_skips_a_directory_pinned_or_queued_after_the_dry_run(tmp_path) -> None:
    root = tmp_path / "compiled-episodes"
    root.mkdir()
    now = 11_000_000.0
    late_pinned = _derived(root, "comp-late-pinned", age=10 * 86400, now=now)
    late_queued = _derived(root, "comp-late-queued", age=10 * 86400, now=now)
    pins = tmp_path / "pins"
    queue = tmp_path / "queue"
    (queue / "processing").mkdir(parents=True)
    manifest = build_derived_directory_manifest(
        derived_roots=[root], pins_root=pins, queue_roots=[queue], minimum_age_seconds=0,
        now=lambda: now, classifier=_noclass,
    )
    assert manifest["candidate_count"] == 2

    write_storage_pin(
        pins_root=pins, kind="compilation", owner_id="comp-late-pinned", paths=[late_pinned],
        now=lambda: now,
    )
    (queue / "processing" / "late.json").write_text(
        json.dumps({"compilation_id": "comp-late-queued"}), encoding="utf-8"
    )
    receipt = apply_derived_directory_manifest(
        manifest, ack=DERIVED_ACK, pins_root=pins, queue_roots=[queue], now=lambda: now,
        classifier=_noclass,
    )
    assert receipt["removed"] == []
    assert sorted(row["name"] for row in receipt["skipped"]) == [
        "comp-late-pinned",
        "comp-late-queued",
    ]
    assert late_pinned.exists() and late_queued.exists()


def test_run_retires_directories_before_reaping_the_blobs_they_linked(tmp_path) -> None:
    root = tmp_path / "prepared-references"
    cas = root / "content-addressed" / "sha256"
    cas.mkdir(parents=True)
    payload = b"layer-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    blob = cas / digest
    blob.write_bytes(payload)
    prep = root / "prep-1"
    prep.mkdir()
    os.link(blob, prep / digest)
    now = 20_000_000.0
    old = now - 10 * 86400
    for path in (blob, prep / digest, prep):
        os.utime(path, (old, old))
    pins = tmp_path / "pins"
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    common = dict(
        content_store_roots=[cas],
        derived_roots=[root],
        queue_roots=[queue],
        pins_root=pins,
        now=lambda: now,
        classifier=_noclass,
    )

    dry = run_storage_gc(**common)
    assert dry["status"] == "dry_run"
    assert dry["derived_directories"]["candidate_count"] == 1
    assert dry["content_store"]["retained_counts"]["linked"] == 1
    assert "evidence_offload" not in dry
    assert blob.exists() and prep.exists()

    with pytest.raises(ControlPlaneStorageGCError, match="apply_not_authorized"):
        run_storage_gc(**common, apply=True, ack="wrong")
    applied = run_storage_gc(**common, apply=True, ack=RUN_ACK)
    assert applied["status"] == "applied"
    assert applied["derived_directories"]["removed_count"] == 1
    assert applied["content_store"]["removed_count"] == 1
    assert not prep.exists() and not blob.exists()
    assert applied["skipped_roots"] == []

    partial = run_storage_gc(
        content_store_roots=[tmp_path / "absent" / "sha256"],
        derived_roots=[tmp_path / "absent-derived"],
        queue_roots=[queue],
        pins_root=pins,
        now=lambda: now,
        classifier=_noclass,
    )
    assert set(partial["skipped_roots"]) == {
        str(tmp_path / "absent" / "sha256"),
        str(tmp_path / "absent-derived"),
    }


def test_run_offloads_sealed_evidence_only_when_enabled(tmp_path) -> None:
    evidence = tmp_path / "launch-runs"
    run = evidence / "run-1"
    run.mkdir(parents=True)
    (run / "dispatch_receipt.json").write_text("{}", encoding="utf-8")
    (run / "log.txt").write_text("x", encoding="utf-8")
    now = 30_000_000.0
    old = now - 30 * 86400
    for path in (run / "dispatch_receipt.json", run / "log.txt", run):
        os.utime(path, (old, old))
    pins = tmp_path / "pins"
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    common = dict(
        content_store_roots=[],
        derived_roots=[],
        queue_roots=[queue],
        pins_root=pins,
        evidence_roots=[evidence],
        now=lambda: now,
        classifier=_noclass,
    )

    disabled = run_storage_gc(**common, apply=True, ack=RUN_ACK, offload_enabled=False)
    assert disabled["evidence_offload"]["status"] == "dry_run"
    assert disabled["evidence_offload"]["candidate_count"] == 1
    assert disabled["evidence_offload_enabled"] is False
    assert run.is_dir()

    client = _ContentAddressedClient()
    enabled = run_storage_gc(
        **common,
        apply=True,
        ack=RUN_ACK,
        offload_enabled=True,
        publisher=functools.partial(
            store.publish_configured_scene_artifact,
            client=client,
            bucket="blueprint-production-inputs",
        ),
    )
    assert enabled["evidence_offload"]["offloaded_count"] == 1
    assert not run.exists()
    assert (evidence / "run-1.offloaded.v1.json").is_file()
    assert client.upload_count == 1


def test_run_cli_reads_roots_from_the_unit_environment(tmp_path, monkeypatch, capsys) -> None:
    root = tmp_path / "prepared-references"
    cas = root / "content-addressed" / "sha256"
    cas.mkdir(parents=True)
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_CONTENT_STORE_ROOTS", str(cas))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_DERIVED_ROOTS", str(root))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS", str(tmp_path / "queue"))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_EVIDENCE_ROOTS", "")
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT", str(tmp_path / "pins"))
    monkeypatch.setattr(gc_module, "require_storage_class", _noclass)
    report = tmp_path / "storage-gc" / "latest.json"

    assert gc_main(["run", "--report-out", str(report)]) == 0

    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["schema_version"] == "control_plane_storage_gc_run.v1"
    assert written["status"] == "dry_run"
    assert json.loads(capsys.readouterr().out)["report_digest"] == written["report_digest"]
    monkeypatch.delenv("BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT")
    with pytest.raises(ControlPlaneStorageGCError, match="pins_root_missing"):
        gc_main(["run"])
