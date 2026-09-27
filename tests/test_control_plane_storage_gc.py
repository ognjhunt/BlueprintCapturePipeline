from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import stat
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import control_plane_storage_gc as gc_module
from blueprint_pipeline import website_scene_workspace_retention as retention_module
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


@pytest.fixture(autouse=True)
def isolated_disk_ledger(tmp_path, monkeypatch):
    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT",
                       str(tmp_path / "disk-reservations"))
    monkeypatch.setattr("blueprint_pipeline.pubsub_handoff_disk_admission.reserve_control_plane_disk",
                        functools.partial(reserve_control_plane_disk,
                            disk_usage=lambda _: SimpleNamespace(total=100 * 1024**3, free=80 * 1024**3)))
    monkeypatch.setattr("blueprint_pipeline.control_plane_evidence_offload.reserve_control_plane_disk",
                        functools.partial(reserve_control_plane_disk,
                            disk_usage=lambda _: SimpleNamespace(total=100 * 1024**3, free=80 * 1024**3)))
    # An empty process table; active_reference answers from the same sweep.
    monkeypatch.setattr("blueprint_pipeline.completed_replay_cache_retention.process_reference", lambda _, **kwargs: None)
    monkeypatch.setattr("blueprint_pipeline.control_plane_evidence_offload.DEFAULT_RESERVATION_ROOT",
                        tmp_path / "disk-reservations")
    monkeypatch.setattr("blueprint_pipeline.website_scene_workspace_retention.reserve_control_plane_disk",
                        functools.partial(reserve_control_plane_disk,
                            disk_usage=lambda _: SimpleNamespace(total=100 * 1024**3, free=80 * 1024**3)))
    monkeypatch.setattr("blueprint_pipeline.website_scene_workspace_retention.DEFAULT_RESERVATION_ROOT",
                        tmp_path / "disk-reservations")


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
    assert changed["candidate_count"] == manifest["candidate_count"]
    assert changed["candidate_bytes"] == manifest["candidate_bytes"]
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


def test_derived_manifest_names_each_retention_reason_with_bytes(tmp_path, monkeypatch) -> None:
    """2026-09-27: the derived phase counted what it kept, but no count said how many bytes it
    held, and a pinned count did not say what kind of pin held it."""

    root = tmp_path / "launch-activations"
    root.mkdir()
    now, day = 12_000_000.0, 86400

    def derived(name: str, size: int, *, age: float = 10 * day) -> Path:
        directory = root / name
        directory.mkdir()
        (directory / "set.bin").write_bytes(b"d" * size)
        for path in (directory / "set.bin", directory):
            os.utime(path, (now - age, now - age))
        return directory

    derived("act-idle", 100)
    by_activation = derived("act-pinned", 200)
    by_two_kinds = derived("act-pinned-twice", 300)
    derived("act-queued", 400)
    derived("act-young", 500, age=3600)
    (root / "act-link").symlink_to(root / "act-idle")
    (root / "stray.bin").write_bytes(b"s" * 9)
    pins = tmp_path / "pins"
    for kind, owner, path in (("activation", "act-pinned", by_activation),
                              ("activation", "act-pinned-twice", by_two_kinds),
                              ("preparation", "prep-1", by_two_kinds)):
        write_storage_pin(pins_root=pins, kind=kind, owner_id=owner, paths=[path], now=lambda: now)
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    (queue / "pending" / "row.json").write_text(json.dumps({"activation_id": "act-queued"}), encoding="utf-8")
    walked: list[str] = []
    real_census = gc_module._tree_census
    monkeypatch.setattr(gc_module, "_tree_census", lambda path: walked.append(path.name) or real_census(path))

    manifest = build_derived_directory_manifest(
        derived_roots=[root], pins_root=pins, queue_roots=[queue], minimum_age_seconds=day,
        now=lambda: now, classifier=_noclass,
    )

    assert manifest["retained_by_reason"] == {
        "pinned": {"count": 2, "bytes": 500, "by_kind": {
            "activation": {"count": 1, "bytes": 200},
            "activation+preparation": {"count": 1, "bytes": 300},
        }},
        "queue_referenced": {"count": 1, "bytes": 400},
        "young": {"count": 1, "bytes": 500},
        "unsafe": {"count": 2, "bytes": os.lstat(root / "act-link").st_size + 9},
    }
    assert manifest["retained_counts"] == {"pinned": 2, "queue_referenced": 1, "young": 1, "unsafe": 2}
    assert [row["name"] for row in manifest["candidates"]] == ["act-idle"]
    assert manifest["candidate_bytes"] == 100
    # Each directory is walked once; a link or stray file never is.
    assert sorted(walked) == ["act-idle", "act-pinned", "act-pinned-twice", "act-queued", "act-young"]
    assert manifest["walked_file_count"] == 5
    assert isinstance(manifest["walk_seconds"], float) and manifest["walk_seconds"] >= 0


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


def test_plan_only_derived_root_is_never_evicted_by_apply(tmp_path) -> None:
    root = tmp_path / "sam31-preparations"
    root.mkdir()
    candidate = _derived(root, "finished-preparation", age=7200, now=20_000_000)
    common = dict(
        content_store_roots=[], derived_roots=[], plan_only_derived_roots=[root],
        queue_roots=[], pins_root=tmp_path / "pins", now=lambda: 20_000_000,
        derived_minimum_age_seconds=3600, classifier=_noclass,
    )
    report = run_storage_gc(**common, apply=True, ack=RUN_ACK)
    assert report["planned_derived_directories"]["status"] == "dry_run"
    assert report["planned_derived_directories"]["candidate_count"] == 1
    assert candidate.exists()
    with pytest.raises(ControlPlaneStorageGCError, match="plan_only_root_in_apply_roots"):
        run_storage_gc(**{**common, "derived_roots": [root]}, apply=True, ack=RUN_ACK)
    assert candidate.exists()


def test_run_offloads_sealed_evidence_only_when_enabled(tmp_path, monkeypatch) -> None:
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

    monkeypatch.setattr("blueprint_pipeline.completed_replay_cache_retention.process_reference",
                        lambda _, **kwargs: "referenced")
    active = run_storage_gc(**common, apply=True, ack=RUN_ACK, offload_enabled=True)
    assert active["evidence_offload"]["offloaded_count"] == 0
    assert run.is_dir()
    monkeypatch.setattr("blueprint_pipeline.completed_replay_cache_retention.process_reference", lambda _, **kwargs: None)
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


def test_applied_receipts_carry_retained_reasons(tmp_path) -> None:
    """2026-09-27: the applied tick's receipts dropped the manifests' retained counts, so a
    tick that removed nothing reported nothing about what it kept."""

    now, day = 40_000_000.0, 86400

    def tree(path: Path, size: int, *, age: float, receipt: bool = False) -> Path:
        path.mkdir(parents=True)
        (path / "payload.bin").write_bytes(b"p" * size)
        if receipt:
            (path / "dispatch_receipt.json").write_text("{}", encoding="utf-8")
        for item in (*path.iterdir(), path):
            os.utime(item, (now - age, now - age))
        return path

    derived = tmp_path / "prepared-references"
    tree(derived / "prep-idle", 100, age=10 * day)
    pinned = tree(derived / "prep-pinned", 200, age=10 * day)
    tree(derived / "prep-young", 300, age=60)
    evidence = tmp_path / "launch-runs"
    tree(evidence / "run-cold", 1000, age=30 * day, receipt=True)
    tree(evidence / "run-hot", 2000, age=day, receipt=True)
    tree(evidence / "run-queued", 3000, age=30 * day, receipt=True)
    pins = tmp_path / "pins"
    write_storage_pin(pins_root=pins, kind="preparation", owner_id="prep-pinned", paths=[pinned], now=lambda: now)
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    (queue / "pending" / "row.json").write_text(json.dumps({"run": "run-queued"}), encoding="utf-8")
    common = dict(content_store_roots=[], derived_roots=[derived], queue_roots=[queue], pins_root=pins,
                  evidence_roots=[evidence], hot_window_seconds=2 * day, derived_minimum_age_seconds=day,
                  now=lambda: now, classifier=_noclass)

    planned = run_storage_gc(**common)
    applied = run_storage_gc(**common, apply=True, ack=RUN_ACK, offload_enabled=True, publisher=functools.partial(
        store.publish_configured_scene_artifact, client=_ContentAddressedClient(), bucket="blueprint-production-inputs"))

    derived_receipt, evidence_receipt = applied["derived_directories"], applied["evidence_offload"]
    assert (derived_receipt["status"], derived_receipt["removed_count"]) == ("applied", 1)
    assert derived_receipt["retained_by_reason"] == planned["derived_directories"]["retained_by_reason"] == {
        "pinned": {"count": 1, "bytes": 200, "by_kind": {"preparation": {"count": 1, "bytes": 200}}},
        "young": {"count": 1, "bytes": 300},
    }
    assert (derived_receipt["candidate_count"], derived_receipt["candidate_bytes"]) == (1, 100)
    assert (evidence_receipt["status"], evidence_receipt["offloaded_count"]) == ("applied", 1)
    assert evidence_receipt["retained_by_reason"] == planned["evidence_offload"]["retained_by_reason"] == {
        "hot": {"count": 1, "bytes": 2002},
        "protected_queue": {"count": 1, "bytes": 3002},
    }
    assert (evidence_receipt["candidate_count"], evidence_receipt["candidate_bytes"]) == (1, 1002)
    assert derived_receipt["walked_file_count"] == planned["derived_directories"]["walked_file_count"] == 3
    assert evidence_receipt["walked_file_count"] == planned["evidence_offload"]["walked_file_count"] == 6
    assert all(isinstance(receipt["walk_seconds"], float) for receipt in (derived_receipt, evidence_receipt))
    # The receipts stay digest-bound with the new fields inside.
    for receipt in (derived_receipt, evidence_receipt):
        assert receipt["result_digest"] == gc_module.canonical_digest(receipt, digest_field="result_digest")


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
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))
    from operator_door.config import DoorConfig
    from operator_door.fsview import FileView

    door = FileView(DoorConfig(read_roots=(str(tmp_path),), hidden_paths=()))
    contents, _ = door.read_range(str(report))
    assert json.loads(contents)["report_digest"] == written["report_digest"]
    monkeypatch.delenv("BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT")
    with pytest.raises(ControlPlaneStorageGCError, match="pins_root_missing"):
        gc_main(["run"])


RUNNING_COMMIT = "a" * 40
STALE_COMMIT = "b" * 40


def test_latest_gc_report_is_readable_by_the_operator_door_after_each_tick(tmp_path) -> None:
    report_dir = tmp_path / "storage-gc"
    report_dir.mkdir(mode=0o700)
    path = report_dir / "latest.json"

    for status in ("dry_run", "applied"):
        gc_module._write_report(path, {"status": status})
        assert report_dir.stat().st_mode & 0o777 == 0o755
        assert path.stat().st_mode & 0o777 == 0o644
        assert json.loads(path.read_text(encoding="utf-8")) == {"status": status}


def test_report_reclaims_a_directory_owned_by_the_previous_service_user(tmp_path, monkeypatch) -> None:
    path = tmp_path / "storage-gc" / "latest.json"
    path.parent.mkdir(mode=0o700)
    original_fchmod = os.fchmod
    calls: list[tuple[int, int]] = []

    def previous_owner_blocks_chmod(fd: int, mode: int) -> None:
        if not calls:
            raise PermissionError("old service user owns report directory")
        original_fchmod(fd, mode)

    def change_owner(fd: int, uid: int, gid: int) -> None:
        calls.append((uid, gid))

    monkeypatch.setattr(os, "fchmod", previous_owner_blocks_chmod)
    monkeypatch.setattr(os, "fchown", change_owner)
    gc_module._write_report(path, {"status": "dry_run"})

    assert calls == [(os.geteuid(), -1)]
    assert path.parent.stat().st_mode & 0o777 == 0o755
    assert path.stat().st_mode & 0o777 == 0o644


def test_report_publication_stays_bound_to_checked_directory(tmp_path, monkeypatch) -> None:
    report_dir = tmp_path / "storage-gc"
    report_dir.mkdir()
    moved_dir = tmp_path / "moved-storage-gc"
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    other_report = other_dir / "latest.json"
    other_report.write_text('and keep this report', encoding="utf-8")
    original_fchmod = os.fchmod
    retargeted = False

    def retarget_after_directory_check(fd: int, mode: int) -> None:
        nonlocal retargeted
        original_fchmod(fd, mode)
        if stat.S_ISDIR(os.fstat(fd).st_mode) and not retargeted:
            report_dir.rename(moved_dir)
            report_dir.symlink_to(other_dir, target_is_directory=True)
            retargeted = True

    monkeypatch.setattr(os, "fchmod", retarget_after_directory_check)
    with pytest.raises(ControlPlaneStorageGCError, match="storage_gc_report_directory_retargeted"):
        gc_module._write_report(report_dir / "latest.json", {"status": "dry_run"})

    assert retargeted
    assert other_report.read_text(encoding="utf-8") == 'and keep this report'
    assert json.loads((moved_dir / "latest.json").read_text(encoding="utf-8")) == {
        "status": "dry_run"
    }


def _queue_row(root: Path, state: str, name: str, *, commit: str | None) -> Path:
    directory = root / state
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict = {"schema_version": "row.v1", "derived_directory": f"derived-{name}"}
    if commit is not None:
        payload["expected_production_commit"] = commit
    path = directory / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_pending_rows_bound_to_another_release_are_stranded_with_receipts(tmp_path) -> None:
    """Every worker honours only same-release rows, so a pending row bound to a
    superseded release never progresses, yet it pinned that release's trees
    and caches as a live reference: 34 such rows protected 33 dead releases."""
    queue = tmp_path / "queue"
    stale = _queue_row(queue, "pending", "stale", commit=STALE_COMMIT)
    same = _queue_row(queue, "pending", "same", commit=RUNNING_COMMIT)
    unbound = _queue_row(queue, "pending", "unbound", commit=None)
    busy = _queue_row(queue, "processing", "busy", commit=STALE_COMMIT)

    manifest = gc_module.build_stranded_queue_manifest(
        queue_roots=[queue], running_commit=RUNNING_COMMIT, now=lambda: 1_000.0, classifier=_noclass
    )

    assert [row["name"] for row in manifest["candidates"]] == ["stale.json"]
    assert manifest["retained_counts"] == {"same_release": 1, "unbound": 1, "unsafe": 0}
    with pytest.raises(ControlPlaneStorageGCError, match="stranded_apply_not_authorized"):
        gc_module.apply_stranded_queue_manifest(manifest, ack="wrong")
    receipt = gc_module.apply_stranded_queue_manifest(
        manifest, ack=gc_module.STRANDED_ACK, now=lambda: 1_001.0
    )
    assert receipt["stranded_count"] == 1 and receipt["skipped"] == []
    assert receipt["evidence_deleted"] is False
    assert not stale.exists() and same.exists() and unbound.exists() and busy.exists()
    moved = queue / "stranded" / "stale.json"
    assert json.loads(moved.read_text(encoding="utf-8"))["expected_production_commit"] == STALE_COMMIT
    row_receipt = json.loads((queue / "stranded" / "stale.json.stranded.v1.json").read_text(encoding="utf-8"))
    assert row_receipt["bound_commit"] == STALE_COMMIT
    assert row_receipt["running_commit"] == RUNNING_COMMIT
    assert row_receipt["previous_state"] == "pending"
    # A stranded row no longer counts as a live queue reference.
    text = gc_module._queue_reference_text([queue])
    assert "derived-stale" not in text and "derived-same" in text


def test_stranding_skips_a_row_rewritten_after_the_dry_run(tmp_path) -> None:
    queue = tmp_path / "queue"
    stale = _queue_row(queue, "pending", "stale", commit=STALE_COMMIT)
    manifest = gc_module.build_stranded_queue_manifest(
        queue_roots=[queue], running_commit=RUNNING_COMMIT, now=lambda: 1_000.0, classifier=_noclass
    )
    stale.write_text(json.dumps({"expected_production_commit": STALE_COMMIT, "retry": 2}), encoding="utf-8")

    receipt = gc_module.apply_stranded_queue_manifest(manifest, ack=gc_module.STRANDED_ACK)

    assert receipt["stranded_count"] == 0
    assert receipt["skipped"] == [{"name": "stale.json", "reason": "candidate_changed"}]
    assert stale.exists()
    with pytest.raises(ControlPlaneStorageGCError, match="running_commit_invalid"):
        gc_module.build_stranded_queue_manifest(queue_roots=[queue], running_commit="", classifier=_noclass)


def test_policy_dispatcher_keeps_ownership_of_old_release_delivery(tmp_path):
    queue = tmp_path / "queue"
    path = _queue_row(queue, "pending", "delivery", commit=STALE_COMMIT)
    row = json.loads(path.read_text())
    row["schema_version"] = "task_evaluation_policy_canary_dispatch_envelope.v1"
    path.write_text(json.dumps(row))
    manifest = gc_module.build_stranded_queue_manifest(
        queue_roots=[queue], running_commit=RUNNING_COMMIT, classifier=_noclass)
    assert manifest["candidate_count"] == 0
    gc_module.apply_stranded_queue_manifest(manifest, ack=gc_module.STRANDED_ACK)
    assert path.exists()


def _scratch(root: Path, name: str, *, age: float, now: float, directory: bool = True) -> Path:
    path = root / name
    if directory:
        (path / "sub").mkdir(parents=True)
        (path / "sub" / "probe.bin").write_bytes(b"x" * 64)
    else:
        root.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"y" * 32)
    stamp = now - age
    for item in [path, *(path.rglob("*") if directory else [])]:
        os.utime(item, (stamp, stamp))
    return path


def test_scratch_is_reaped_by_idle_age_alone_and_touched_trees_survive(tmp_path) -> None:
    root = tmp_path / "engineering"
    now = 5_000_000.0
    idle = _scratch(root, "old-probe", age=4 * 86400, now=now)
    recent = _scratch(root, "fresh-probe", age=3600, now=now)
    idle_file = _scratch(root, "old.log", age=4 * 86400, now=now, directory=False)

    manifest = gc_module.build_scratch_manifest(
        scratch_roots=[root], minimum_age_seconds=3 * 86400, now=lambda: now, classifier=_noclass
    )

    assert sorted(row["name"] for row in manifest["candidates"]) == ["old-probe", "old.log"]
    assert manifest["retained_counts"] == {"recent": 1, "unsafe": 0}
    (idle / "sub" / "probe.bin").write_bytes(b"still in use")
    with pytest.raises(ControlPlaneStorageGCError, match="scratch_apply_not_authorized"):
        gc_module.apply_scratch_manifest(manifest, ack="wrong")
    receipt = gc_module.apply_scratch_manifest(manifest, ack=gc_module.SCRATCH_ACK, now=lambda: now)
    assert receipt["candidate_count"] == manifest["candidate_count"]
    assert receipt["candidate_bytes"] == manifest["candidate_bytes"]
    assert [row["name"] for row in receipt["removed"]] == ["old.log"]
    assert receipt["skipped"] == [{"name": "old-probe", "reason": "candidate_changed"}]
    assert idle.exists() and recent.exists() and not idle_file.exists()
    with pytest.raises(ControlPlaneStorageGCError, match="scratch_window_invalid"):
        gc_module.build_scratch_manifest(scratch_roots=[root], minimum_age_seconds=-1, classifier=_noclass)


def test_run_cli_wires_windows_scratch_and_running_commit_from_the_unit_environment(
    tmp_path, monkeypatch, capsys
) -> None:
    import time as _time

    queue = tmp_path / "queue"
    _queue_row(queue, "pending", "stale", commit=STALE_COMMIT)
    scratch = tmp_path / "engineering"
    _scratch(scratch, "old-probe", age=10 * 86400, now=_time.time())
    evidence = tmp_path / "launch-runs"
    evidence.mkdir()
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_CONTENT_STORE_ROOTS", "")
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_DERIVED_ROOTS", "")
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS", str(queue))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_EVIDENCE_ROOTS", str(evidence))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_SCRATCH_ROOTS", str(scratch))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_SCRATCH_MINIMUM_AGE_SECONDS", str(7 * 86400))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_EVIDENCE_HOT_WINDOW_SECONDS", "172800")
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_EVIDENCE_ABANDONED_AFTER_SECONDS", "259200")
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_RUNNING_COMMIT", RUNNING_COMMIT)
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT", str(tmp_path / "pins"))
    monkeypatch.setattr(gc_module, "require_storage_class", _noclass)

    assert gc_main(["run"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["status"] == "dry_run"
    assert report["stranded_queue_rows"]["candidate_count"] == 1
    assert report["stranded_queue_rows"]["running_commit"] == RUNNING_COMMIT
    assert report["evidence_offload"]["hot_window_seconds"] == 172800
    assert report["evidence_offload"]["abandoned_after_seconds"] == 259200
    assert report["scratch_directories"]["candidate_count"] == 1
    assert report["scratch_directories"]["minimum_age_seconds"] == 7 * 86400

    monkeypatch.delenv("BLUEPRINT_CONTROL_PLANE_GC_RUNNING_COMMIT")
    monkeypatch.setattr(gc_module, "running_release_commit", lambda: "")
    assert gc_main(["run"]) == 0
    assert json.loads(capsys.readouterr().out)["stranded_queue_rows"] == {
        "status": "skipped",
        "reason": "running_commit_unknown",
    }


def _workspace(root: Path, name: str, *, age: float, now: float, with_bundle: bool = True) -> Path:
    workspace = root / name
    (workspace / "output").mkdir(parents=True)
    (workspace / "output" / "receipt.json").write_text('{"status": "completed"}\n', encoding="utf-8")
    if with_bundle:
        (workspace / "bundle" / "provider_runtime").mkdir(parents=True)
        (workspace / "bundle" / "provider_runtime" / "runtime.bin").write_bytes(b"r" * 4096)
    stamp = now - age
    for item in [workspace, *workspace.rglob("*")]:
        os.utime(item, (stamp, stamp))
    from blueprint_pipeline.control_plane_workspace_lock import workspace_lock
    with workspace_lock(workspace):
        pass
    return workspace


def test_workspace_bundles_are_reaped_only_from_idle_workspaces_and_keep_outputs(tmp_path) -> None:
    """2026-09-13: each retry left a 1.4 GB provider-runtime copy under
    semantic-pretraining/<digest>/bundle until the disk budget refused the next attempt."""
    root = tmp_path / "semantic-pretraining"
    now = 5_000_000.0
    idle = _workspace(root, "a" * 64, age=7 * 3600, now=now)
    touched = _workspace(root, "b" * 64, age=7 * 3600, now=now)
    recent = _workspace(root, "c" * 64, age=3600, now=now)
    bare = _workspace(root, "d" * 64, age=7 * 3600, now=now, with_bundle=False)

    manifest = gc_module.build_workspace_bundle_manifest(
        workspace_roots=[root], minimum_age_seconds=6 * 3600, now=lambda: now, classifier=_noclass
    )
    assert sorted(row["workspace"] for row in manifest["candidates"]) == ["a" * 64, "b" * 64]
    assert manifest["retained_counts"] == {"recent": 1, "unsafe": 0, "no_bundle": 1, "pinned": 0, "in_use": 0}
    assert manifest["candidate_bytes"] == 2 * 4096

    (touched / "output" / "receipt.json").write_text('{"status": "reviewing"}\n', encoding="utf-8")
    with pytest.raises(ControlPlaneStorageGCError, match="workspace_bundle_apply_not_authorized"):
        gc_module.apply_workspace_bundle_manifest(manifest, ack="wrong")
    receipt = gc_module.apply_workspace_bundle_manifest(
        manifest, ack=gc_module.WORKSPACE_BUNDLE_ACK, now=lambda: now
    )
    assert receipt["candidate_count"] == manifest["candidate_count"]
    assert receipt["candidate_bytes"] == manifest["candidate_bytes"]
    assert [row["workspace"] for row in receipt["removed"]] == ["a" * 64]
    assert receipt["skipped"] == [{"workspace": "b" * 64, "reason": "candidate_changed"}]
    assert receipt["evidence_removed"] is False
    assert not (idle / "bundle").exists() and (idle / "output" / "receipt.json").exists()
    marker = json.loads((idle / gc_module.WORKSPACE_BUNDLE_MARKER).read_text())
    assert marker["schema_version"] == gc_module.WORKSPACE_BUNDLE_MARKER_SCHEMA_VERSION
    assert marker["reaped_bytes"] == 4096 and marker["outputs_and_receipts_retained"] is True
    assert (touched / "bundle").exists() and (recent / "bundle").exists() and bare.exists()
    again = gc_module.build_workspace_bundle_manifest(
        workspace_roots=[root], minimum_age_seconds=6 * 3600, now=lambda: now + 8 * 3600, classifier=_noclass
    )
    # The touched workspace is recent again; the reaped one has no bundle left.
    assert [row["workspace"] for row in again["candidates"]] == ["c" * 64]
    assert again["retained_counts"] == {"recent": 1, "unsafe": 0, "no_bundle": 2, "pinned": 0, "in_use": 0}
    with pytest.raises(ControlPlaneStorageGCError, match="workspace_bundle_window_invalid"):
        gc_module.build_workspace_bundle_manifest(workspace_roots=[root], minimum_age_seconds=-1, classifier=_noclass)


def test_run_reaps_workspace_bundles_after_the_other_classes(tmp_path) -> None:
    root = tmp_path / "semantic-pretraining"
    now = 5_000_000.0
    idle = _workspace(root, "e" * 64, age=7 * 3600, now=now)
    common = dict(content_store_roots=[], derived_roots=[], queue_roots=[], pins_root=tmp_path / "pins",
                  workspace_bundle_roots=[root], workspace_bundle_minimum_age_seconds=6 * 3600,
                  now=lambda: now, classifier=_noclass)
    dry = run_storage_gc(**common)
    assert dry["workspace_bundles"]["candidate_count"] == 1 and (idle / "bundle").exists()
    applied = run_storage_gc(**common, apply=True, ack=RUN_ACK)
    assert applied["workspace_bundles"]["removed_count"] == 1
    assert not (idle / "bundle").exists() and (idle / "output" / "receipt.json").exists()


def test_gc_unit_can_write_every_workspace_bundle_root_it_names() -> None:
    """2026-09-13: the first reap failed with PermissionError because the unit runs under
    ProtectSystem=strict and its ReadWritePaths did not include the workspace root."""
    unit = (Path(__file__).resolve().parents[1] / "deploy/systemd/blueprint-control-plane-storage-gc.service").read_text(
        encoding="utf-8"
    )
    roots: list[str] = []
    writable: set[str] = set()
    for line in unit.splitlines():
        if line.startswith("Environment=" + gc_module.WORKSPACE_BUNDLE_ROOTS_ENV + "="):
            roots.extend(part for part in line.split("=", 2)[2].split(":") if part)
        if line.startswith("ReadWritePaths="):
            writable.update(part.lstrip("-") for part in line.split("=", 1)[1].split())
    assert roots, "the unit must name at least one workspace bundle root"
    missing = [root for root in roots if not any(root == path or root.startswith(path + "/") for path in writable)]
    assert missing == [], missing


@pytest.fixture(autouse=True)
def known_process_inventory(monkeypatch):
    monkeypatch.setattr(gc_module, "workspace_process_active", lambda workspace: False)


def _settlement_scene(root, *, scene: str, launch_run_name: str, reopened: str = "launch_profile.json"):
    """One settled attempt whose receipt reopens ``reopened`` under ``launch_run_name``."""
    directory = root / scene / "cancelled-unstarted-controls"
    directory.mkdir(parents=True)
    (directory / "controls-1.json").write_text(
        json.dumps(
            {
                "schema_version": "task_evaluation_terminal_scene_attempt_settlement.v1",
                "execution_terminal": {
                    "launch_id": launch_run_name,
                    "launch_receipt": {
                        "path": f"/var/lib/blueprint/launch-runs/{launch_run_name}/{reopened}",
                        "digest": "sha256:" + "0" * 64,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return directory


def _cold_evidence_run(evidence, name, now):
    run = evidence / name
    run.mkdir(parents=True)
    (run / "launch_receipt.json").write_text("{}", encoding="utf-8")
    old = now - 30 * 86400
    for path in (run / "launch_receipt.json", run):
        os.utime(path, (old, old))
    return run


def test_offload_retains_evidence_a_settlement_record_still_reopens(tmp_path) -> None:
    """Retention must not archive a launch run a settled attempt still reads.

    Offloading it leaves the spend/controls readers raising
    ``unstarted_controls_evidence_unsafe`` forever, which strands the intent.
    """

    now = 30_000_000.0
    evidence = tmp_path / "launch-runs"
    referenced = _cold_evidence_run(evidence, "run-referenced", now)
    orphan = _cold_evidence_run(evidence, "run-orphan", now)
    settlement = tmp_path / "scene-intents"
    _settlement_scene(settlement, scene="scene-abc", launch_run_name="run-referenced")
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    common = dict(
        content_store_roots=[],
        derived_roots=[],
        queue_roots=[queue],
        pins_root=tmp_path / "pins",
        evidence_roots=[evidence],
        settlement_roots=[settlement],
        now=lambda: now,
        classifier=_noclass,
    )

    client = _ContentAddressedClient()
    report = run_storage_gc(
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

    assert report["evidence_settlement_reference"]["unreadable_count"] == 0
    assert report["evidence_settlement_reference"]["protect_all"] is False
    # The referenced run stays readable; only the unreferenced one is archived.
    assert referenced.is_dir()
    assert (referenced / "launch_receipt.json").is_file()
    assert not orphan.exists()
    assert [row["name"] for row in report["evidence_offload"]["offloaded"]] == ["run-orphan"]


def test_offload_proceeds_when_a_settlement_reopens_only_a_retained_receipt(tmp_path) -> None:
    """A record that reopens ``launch_receipt.json`` keeps working from the pointer.

    The pointer retains that receipt byte-for-byte and ``read_receipt_bytes``
    serves it, so the reference must not pin gigabytes of bulk evidence. A
    bare mention of the run id is not a reopen either.
    """

    from blueprint_pipeline.control_plane_retained_receipt import read_receipt_bytes

    now = 30_000_000.0
    evidence = tmp_path / "launch-runs"
    receipt_only = _cold_evidence_run(evidence, "run-receipt-only", now)
    (receipt_only / "launch_receipt.json").write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    os.utime(receipt_only / "launch_receipt.json", (now - 30 * 86400, now - 30 * 86400))
    mentioned = _cold_evidence_run(evidence, "run-mentioned", now)
    reopened = _cold_evidence_run(evidence, "run-profile", now)
    settlement = tmp_path / "scene-intents"
    _settlement_scene(settlement, scene="scene-receipt", launch_run_name="run-receipt-only",
                      reopened="launch_receipt.json")
    (settlement / "scene-mention" / "attempts").mkdir(parents=True)
    (settlement / "scene-mention" / "attempts" / "a.json").write_text(
        json.dumps({"launch_id": "run-mentioned"}), encoding="utf-8")
    _settlement_scene(settlement, scene="scene-profile", launch_run_name="run-profile")
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    client = _ContentAddressedClient()
    report = run_storage_gc(
        content_store_roots=[], derived_roots=[], queue_roots=[queue], pins_root=tmp_path / "pins",
        evidence_roots=[evidence], settlement_roots=[settlement], now=lambda: now, classifier=_noclass,
        apply=True, ack=RUN_ACK, offload_enabled=True,
        publisher=functools.partial(store.publish_configured_scene_artifact, client=client,
                                    bucket="blueprint-production-inputs"),
    )

    assert sorted(row["name"] for row in report["evidence_offload"]["offloaded"]) == ["run-mentioned", "run-receipt-only"]
    assert reopened.is_dir() and not receipt_only.exists() and not mentioned.exists()
    assert json.loads(read_receipt_bytes(receipt_only / "launch_receipt.json")) == {"status": "blocked"}
    assert gc_module.settlement_reopens_beyond_retained_receipts(
        "run-x", '"/launch-runs/run-x/artifacts/policy/frames.tar"') is True
    assert gc_module.settlement_reopens_beyond_retained_receipts("run-x", '"launch_id": "run-x"') is False
    assert gc_module.settlement_reopens_beyond_retained_receipts("run-x", '"run-x/launch_receipt.json"') is False


def test_offload_protects_every_run_when_a_settlement_root_is_unreadable(tmp_path) -> None:
    """A settlement root that cannot be read is never read as "nothing referenced"."""

    now = 30_000_000.0
    evidence = tmp_path / "launch-runs"
    cold = _cold_evidence_run(evidence, "run-cold", now)
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)

    report = run_storage_gc(
        content_store_roots=[],
        derived_roots=[],
        queue_roots=[queue],
        pins_root=tmp_path / "pins",
        evidence_roots=[evidence],
        settlement_roots=[tmp_path / "scene-intents-that-do-not-exist"],
        now=lambda: now,
        classifier=_noclass,
        apply=True,
        ack=RUN_ACK,
        offload_enabled=True,
    )

    assert report["evidence_settlement_reference"]["unreadable_count"] == 1
    assert report["evidence_settlement_reference"]["protect_all"] is True
    assert report["evidence_offload"]["offloaded_count"] == 0
    assert cold.is_dir()


def test_run_cli_reads_settlement_roots_from_the_unit_environment(tmp_path, monkeypatch, capsys) -> None:
    now = 30_000_000.0
    evidence = tmp_path / "launch-runs"
    referenced = _cold_evidence_run(evidence, "run-referenced", now)
    settlement = tmp_path / "scene-intents"
    _settlement_scene(settlement, scene="scene-abc", launch_run_name="run-referenced")
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_EVIDENCE_ROOTS", str(evidence))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_SETTLEMENT_ROOTS", str(settlement))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS", str(queue))
    monkeypatch.setattr(gc_module, "require_storage_class", _noclass)

    assert gc_module.main(["run", "--pins-root", str(tmp_path / "pins")]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["evidence_settlement_reference"]["roots"] == [str(settlement)]
    assert report["evidence_offload"]["candidate_count"] == 0
    assert referenced.is_dir()


@pytest.mark.parametrize("damage", ["symlink", "oversized"])
def test_offload_protects_when_a_settlement_record_cannot_be_read(tmp_path, damage) -> None:
    """A record we decline to read must protect, never silently unprotect its run.

    A symlinked or oversized settlement record used to be skipped like an
    uninteresting queue message, so the launch run it names looked unreferenced
    and was archived.
    """

    now = 30_000_000.0
    evidence = tmp_path / "launch-runs"
    referenced = _cold_evidence_run(evidence, "run-referenced", now)
    settlement = tmp_path / "scene-intents"
    directory = _settlement_scene(
        settlement, scene="scene-abc", launch_run_name="run-referenced"
    )
    record = directory / "controls-1.json"
    if damage == "symlink":
        payload = record.read_text(encoding="utf-8")
        target = directory / "controls-1.actual.json"
        target.write_text(payload, encoding="utf-8")
        record.unlink()
        record.symlink_to(target)
    else:
        record.write_text(
            json.dumps({"pad": "x" * (gc_module._MAX_QUEUE_MESSAGE_BYTES + 1)}),
            encoding="utf-8",
        )
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)

    report = run_storage_gc(
        content_store_roots=[],
        derived_roots=[],
        queue_roots=[queue],
        pins_root=tmp_path / "pins",
        evidence_roots=[evidence],
        settlement_roots=[settlement],
        now=lambda: now,
        classifier=_noclass,
        apply=True,
        ack=RUN_ACK,
        offload_enabled=True,
    )

    assert report["evidence_settlement_reference"]["unreadable_count"] >= 1
    assert report["evidence_settlement_reference"]["protect_all"] is True
    assert report["evidence_offload"]["offloaded_count"] == 0
    assert referenced.is_dir()


def test_offload_rereads_settlement_records_before_each_eviction(tmp_path) -> None:
    """A settlement written after the manifest was built still protects its run.

    ``apply_evidence_offload`` re-checks protection immediately before evicting
    each candidate, so the check must re-read the records rather than reuse the
    text captured when the manifest was built.
    """

    now = 30_000_000.0
    evidence = tmp_path / "launch-runs"
    first = _cold_evidence_run(evidence, "run-a", now)
    late = _cold_evidence_run(evidence, "run-b", now)
    settlement = tmp_path / "scene-intents"
    (settlement / "scene-abc").mkdir(parents=True)
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)

    client = _ContentAddressedClient()

    def publisher(*args, **kwargs):
        # Between building the manifest and evicting "run-b", a settlement lands
        # that names it. The pre-eviction check has to see it.
        if not (settlement / "scene-abc" / "cancelled-unstarted-controls").exists():
            _settlement_scene(settlement, scene="scene-abc", launch_run_name="run-b")
        return store.publish_configured_scene_artifact(
            *args, client=client, bucket="blueprint-production-inputs", **kwargs
        )

    report = run_storage_gc(
        content_store_roots=[],
        derived_roots=[],
        queue_roots=[queue],
        pins_root=tmp_path / "pins",
        evidence_roots=[evidence],
        settlement_roots=[settlement],
        now=lambda: now,
        classifier=_noclass,
        apply=True,
        ack=RUN_ACK,
        offload_enabled=True,
        publisher=publisher,
    )

    offloaded = {row["name"] for row in report["evidence_offload"]["offloaded"]}
    assert "run-b" not in offloaded, "a settlement written mid-apply must still protect"
    assert late.is_dir()
    assert not first.exists()


def test_run_cli_honours_the_derived_minimum_age_from_the_unit_environment(tmp_path, monkeypatch, capsys):
    """The derived age must be configurable, like every other GC class.

    It was accepted by `run_storage_gc` but never passed by the CLI, so the 6h
    default always won and the unit environment was silently ignored. Scene
    840938, 2026-09-15: ~1.3 GiB of activation set per attempt, a new attempt
    roughly every 45 minutes, so they accumulated far faster than they aged out
    and disk blocked the run three times.
    """

    derived = tmp_path / "launch-activations"
    child = derived / "activation-1"
    child.mkdir(parents=True)
    (child / "bundle.zip").write_bytes(b"x" * 2048)
    # Two hours idle: older than a 1h setting, younger than the 6h default.
    idle = time.time() - 2 * 60 * 60
    for path in (child / "bundle.zip", child):
        os.utime(path, (idle, idle))
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_DERIVED_ROOTS", str(derived))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS", str(queue))
    monkeypatch.delenv("BLUEPRINT_CONTROL_PLANE_GC_DERIVED_MINIMUM_AGE_SECONDS", raising=False)
    monkeypatch.setattr(gc_module, "require_storage_class", _noclass)

    # The 6h default still protects it.
    assert gc_module.main(["run", "--pins-root", str(tmp_path / "pins")]) == 0
    assert json.loads(capsys.readouterr().out)["derived_directories"]["candidate_count"] == 0

    # The unit environment now actually takes effect.
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_DERIVED_MINIMUM_AGE_SECONDS", "3600")
    assert gc_module.main(["run", "--pins-root", str(tmp_path / "pins")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["derived_directories"]["candidate_count"] == 1
    assert report["derived_directories"]["candidates"][0]["name"] == "activation-1"

    # An explicit flag overrides the environment.
    assert gc_module.main([
        "run", "--pins-root", str(tmp_path / "pins"), "--derived-minimum-age-seconds", "86400",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["derived_directories"]["candidate_count"] == 0


# --- scene workspace retirement ---------------------------------------------------------------------


def _scene_tick(tmp_path, *, ack: bool = True):
    """One website scene staged through the real listener, and the tick arguments that reach it."""

    from tests.test_control_plane_evidence_streaming import MultipartClient
    from tests import test_website_scene_workspace_retention as scenes

    scene, cloud = scenes._scene(tmp_path, ack=ack)
    scenes._context(tmp_path)  # the pins, queue, intent and binding roots
    now = time.time() + 72 * 3600
    client = MultipartClient()
    common = dict(
        content_store_roots=[], derived_roots=[], queue_roots=[tmp_path / "queue"], pins_root=tmp_path / "pins",
        scene_workspace_roots=[tmp_path / "pubsub-handoffs"], scene_intent_root=tmp_path / "intents",
        scene_binding_root=tmp_path / "bindings", scene_cloud_factory=lambda: cloud,
        scene_stream_publisher=functools.partial(store.publish_configured_scene_stream, client=client,
                                                 bucket=scenes.ARTIFACT_BUCKET),
        scene_process_checker=lambda _path: False, now=lambda: now, classifier=_noclass,
    )
    return scene, common


def test_gc_phase_retires_verified_terminal_workspace(tmp_path) -> None:
    scene, common = _scene_tick(tmp_path)

    report = run_storage_gc(**common, apply=True, ack=RUN_ACK, scene_workspace_retirement_enabled=True)

    phase = report["scene_workspaces"]
    receipt = scene.parent / "scene-1.retired.v1.json"
    assert (phase["status"], phase["enabled"]) == ("applied", True)
    assert (phase["candidate_count"], phase["retired_count"]) == (1, 1)
    assert phase["retired_bytes"] > 0 and phase["archive_bytes"] > 0 and phase["retained_counts"] == {}
    assert phase["results"] == [{"bucket": "capture-bucket", "scene_id": "scene-1", "status": "retired",
                                 "receipt": str(receipt), "removal_complete": True}]
    assert not scene.exists() and receipt.is_file()
    assert "phase_errors" not in report

    again = run_storage_gc(**common, apply=True, ack=RUN_ACK, scene_workspace_retirement_enabled=True)
    assert again["scene_workspaces"]["candidate_count"] == 0 and receipt.is_file()


def test_gc_phase_only_plans_without_the_opt_in(tmp_path) -> None:
    scene, common = _scene_tick(tmp_path)

    report = run_storage_gc(**common, apply=True, ack=RUN_ACK, scene_workspace_retirement_enabled=False)

    phase = report["scene_workspaces"]
    assert (phase["status"], phase["enabled"], phase["candidate_count"], phase["retired_count"]) == (
        "dry_run", False, 1, 0)
    assert phase["results"][0]["status"] == "retirable" and scene.is_dir()
    dry = run_storage_gc(**common, scene_workspace_retirement_enabled=True)  # no --apply: a dry run too
    assert dry["scene_workspaces"]["status"] == "dry_run" and scene.is_dir()


def test_gc_phase_counts_why_scenes_are_retained(tmp_path) -> None:
    scene, common = _scene_tick(tmp_path, ack=False)

    phase = run_storage_gc(**common, apply=True, ack=RUN_ACK, scene_workspace_retirement_enabled=True)[
        "scene_workspaces"]

    assert phase["retained_counts"] == {"acknowledgement_unproven": 1} and phase["retired_count"] == 0
    assert phase["results"] == [{"bucket": "capture-bucket", "scene_id": "scene-1", "status": "retained",
                                 "reasons": ["acknowledgement_unproven:capture-1"]}]
    assert scene.is_dir()


@pytest.mark.parametrize(("retirement", "offload", "enabled", "alert"), [
    (None, None, False, None),
    (None, "1", False, None),  # the offload opt-in never enables retirement
    ("1", None, True, None),
    ("true", "0", True, None),
    ("0", "1", False, None),
    ("maybe", "1", False, "scene_workspace_retirement_setting_invalid"),
])
def test_scene_retirement_needs_its_own_explicit_opt_in(monkeypatch, retirement, offload, enabled, alert):
    for name, value in ((gc_module.SCENE_WORKSPACE_RETIREMENT_ENV, retirement), (gc_module.EVIDENCE_OFFLOAD_ENV, offload)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert gc_module.scene_workspace_retirement_setting() == (enabled, alert)


def test_an_invalid_retirement_setting_only_plans_and_alerts_without_aborting(tmp_path) -> None:
    scene, common = _scene_tick(tmp_path)
    alert = "scene_workspace_retirement_setting_invalid"

    report = run_storage_gc(**common, apply=True, ack=RUN_ACK, scene_workspace_retirement_enabled=False,
                            scene_workspace_retirement_alert=alert)

    assert report["alerts"] == [alert] and "phase_errors" not in report
    phase = report["scene_workspaces"]
    assert (phase["status"], phase["candidate_count"], phase["alerts"]) == ("dry_run", 1, [alert])
    assert scene.is_dir()


def test_the_command_line_reads_the_opt_in_from_the_environment(tmp_path, monkeypatch, capsys) -> None:
    seen: list[dict] = []

    def run(**kwargs):
        seen.append(kwargs)
        return {"schema_version": gc_module.RUN_SCHEMA_VERSION, "report_digest": "sha256:0"}

    monkeypatch.setattr(gc_module, "run_storage_gc", run)
    monkeypatch.setenv(gc_module.EVIDENCE_OFFLOAD_ENV, "1")
    monkeypatch.setenv(gc_module.SCENE_WORKSPACE_RETIREMENT_ENV, "sometimes")

    assert gc_main(["run", "--pins-root", str(tmp_path / "pins")]) == 0

    assert (seen[0]["scene_workspace_retirement_enabled"], seen[0]["scene_workspace_retirement_alert"]) == (
        False, "scene_workspace_retirement_setting_invalid")
    assert "scene_workspace_retirement_setting_invalid" in capsys.readouterr().err


def test_a_failing_phase_does_not_abort_the_tick(tmp_path, monkeypatch, capsys) -> None:
    now = 5_000_000.0
    derived = tmp_path / "prepared-references"
    derived.mkdir()
    scratch = tmp_path / "engineering"
    _scratch(scratch, "old-probe", age=4 * 86400, now=now)

    def broken(**_kwargs):
        raise RuntimeError("derived manifest failed")

    monkeypatch.setattr(gc_module, "build_derived_directory_manifest", broken)
    report = run_storage_gc(content_store_roots=[], derived_roots=[derived], queue_roots=[], pins_root=tmp_path / "pins",
                            scratch_roots=[scratch], scratch_minimum_age_seconds=3 * 86400, now=lambda: now,
                            classifier=_noclass)

    assert report["derived_directories"] == {"status": "error", "error": "RuntimeError"}
    assert report["phase_errors"] == ["derived_directories"]
    assert report["scratch_directories"]["candidate_count"] == 1, "later phases still run"
    assert report["report_digest"] == gc_module.canonical_digest(report, digest_field="report_digest")

    # The command line still writes the whole report, then fails so the unit shows the error.
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_DERIVED_ROOTS", str(derived))
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS", "")
    monkeypatch.setattr(gc_module, "require_storage_class", _noclass)
    out = tmp_path / "storage-gc" / "latest.json"
    assert gc_main(["run", "--pins-root", str(tmp_path / "pins"), "--report-out", str(out)]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["phase_errors"] == ["derived_directories"]
    assert "derived manifest failed" in capsys.readouterr().err


def test_gc_unit_can_write_scene_workspace_roots() -> None:
    unit = (Path(__file__).resolve().parents[1] / "deploy/systemd/blueprint-control-plane-storage-gc.service").read_text(
        encoding="utf-8"
    )
    roots: list[str] = []
    writable: set[str] = set()
    read_only: set[str] = set()
    for line in unit.splitlines():
        if line.startswith("Environment=" + gc_module.SCENE_WORKSPACE_ROOTS_ENV + "="):
            roots.extend(part for part in line.split("=", 2)[2].split(":") if part)
        if line.startswith("ReadWritePaths="):
            writable.update(part.lstrip("-") for part in line.split("=", 1)[1].split())
        if line.startswith("ReadOnlyPaths="):
            read_only.update(part.lstrip("-") for part in line.split("=", 1)[1].split())
    assert roots == ["/var/lib/blueprint/pubsub-handoffs"]
    assert all(any(root == path or root.startswith(path + "/") for path in writable) for root in roots)
    # A read-only entry at or below a writable root would win over it.
    assert not any(path == root or path.startswith(root + "/") for root in roots for path in read_only)
    assert f"Environment={gc_module.SCENE_INTENT_ROOT_ENV}=" in unit
    for opt_in in (gc_module.SCENE_WORKSPACE_RETIREMENT_ENV, gc_module.EVIDENCE_OFFLOAD_ENV):
        assert f"Environment={opt_in}=" not in unit, "retirement stays an operator opt-in"


def test_each_tick_first_finishes_removals_a_crash_left_behind(tmp_path) -> None:
    scene, common = _scene_tick(tmp_path)
    copy = tmp_path / "scene-before-retirement"
    shutil.copytree(scene, copy)
    run_storage_gc(**common, apply=True, ack=RUN_ACK, scene_workspace_retirement_enabled=True)
    receipt = json.loads((scene.parent / "scene-1.retired.v1.json").read_text(encoding="utf-8"))
    leftover = scene.parent / f".retiring-scene-1-{receipt['retiring_token']}"
    os.rename(copy, leftover)

    dry = run_storage_gc(**common, scene_workspace_retirement_enabled=True)["scene_workspaces"]
    assert leftover.is_dir() and dry["retiring_removable_count"] == 1, "a dry-run tick deletes nothing"

    # The opt-in also governs the crash-left sweep.
    off = run_storage_gc(**common, apply=True, ack=RUN_ACK, scene_workspace_retirement_enabled=False)[
        "scene_workspaces"]
    assert leftover.is_dir() and off["retiring_removable_count"] == 1
    phase = run_storage_gc(**common, apply=True, ack=RUN_ACK, scene_workspace_retirement_enabled=True)[
        "scene_workspaces"]

    assert not leftover.exists()
    assert phase["retiring_removed_count"] == 1 and phase["retiring_kept_without_receipt"] == []


def test_the_tick_bounds_retirement_attempts_not_only_successes(tmp_path, monkeypatch) -> None:
    """Each attempt can publish a large archive, so a failing publisher must not be retried per scene."""

    from blueprint_pipeline import website_scene_workspace_retention as retention

    attempts: list[str] = []
    monkeypatch.setattr(retention, "scene_workspaces",
                        lambda root: [("bucket", f"scene-{index}", root / f"scene-{index}") for index in range(3)])
    monkeypatch.setattr(retention, "sweep_retiring_workspaces",
                        lambda root, apply=True: {"removed" if apply else "removable": [], "kept_without_receipt": []})
    monkeypatch.setattr(retention, "build_reference_index", lambda context, now: None)
    monkeypatch.setattr(retention, "plan_scene_workspace_retirement", lambda **kwargs: {
        "status": "retirable", "reasons": [], "scene_id": kwargs["scene_id"],
        "totals": {"workspace_allocated_bytes": 10, "archive_bytes": 5}})

    def failing_apply(plan, **_kwargs):
        attempts.append(plan["scene_id"])
        return {"status": "skipped", "reason": "archive_readback_failed"}

    monkeypatch.setattr(retention, "apply_scene_workspace_retirement", failing_apply)

    report = gc_module.retire_scene_workspaces(
        storage_roots=[tmp_path], context_factory=lambda root: SimpleNamespace(storage_root=root), apply=True,
        enabled=True, now=1.0, cloud_factory=lambda: None, max_retirements=2)

    assert attempts == ["scene-0", "scene-1"]
    assert (report["attempted_count"], report["retired_count"], report["candidate_count"]) == (2, 0, 3)
    assert [row["status"] for row in report["results"]] == ["skipped", "skipped", "retirable"]


def test_post_upload_archive_reference_survives_result_truncation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(retention_module, "scene_workspaces", lambda root: [
        ("bucket", f"scene-{index}", root / f"scene-{index}") for index in range(51)])
    monkeypatch.setattr(retention_module, "sweep_retiring_workspaces", lambda root, apply=True: {
        "removed" if apply else "removable": [], "kept_without_receipt": []})
    monkeypatch.setattr(retention_module, "sweep_retirement_temporaries", lambda root, now: [])
    monkeypatch.setattr(retention_module, "build_reference_index", lambda context, now: None)
    monkeypatch.setattr(retention_module, "plan_scene_workspace_retirement", lambda **kwargs: {
        "status": "retirable", "reasons": [], "scene_id": kwargs["scene_id"],
        "totals": {"workspace_allocated_bytes": 10, "archive_bytes": 5}})

    def skipped(plan, **_kwargs):
        result = {"status": "skipped", "reason": "candidate_changed_during_archive"}
        if plan["scene_id"] == "scene-50":
            result["published_archive"] = {"uri": "s3://example/orphan", "digest": "sha256:abc", "size_bytes": 5}
        return result

    monkeypatch.setattr(retention_module, "apply_scene_workspace_retirement", skipped)
    report = gc_module.retire_scene_workspaces(
        storage_roots=[tmp_path], context_factory=lambda root: SimpleNamespace(storage_root=root, inventory_cache_root=None),
        apply=True, enabled=True, now=1.0, cloud_factory=lambda: object(), max_retirements=51)

    assert report["result_count"] == 51 and len(report["results"]) == 50
    assert report["published_archives"] == [{"bucket": "bucket", "scene_id": "scene-50",
                                              "uri": "s3://example/orphan", "digest": "sha256:abc", "size_bytes": 5}]


def test_ticks_reuse_cached_digests_and_drop_them_once_a_scene_is_retired(tmp_path) -> None:
    scene, common = _scene_tick(tmp_path)
    cache_root = tmp_path / "storage-gc" / "scene-workspace-inventory"
    cache = cache_root / "capture-bucket" / "scene-1.json"

    first = run_storage_gc(**common, scene_inventory_cache_root=cache_root)["scene_workspaces"]
    second = run_storage_gc(**common, scene_inventory_cache_root=cache_root)["scene_workspaces"]

    assert first["hashed_bytes"] > 0 and second["hashed_bytes"] == 0 and cache.is_file()
    run_storage_gc(**common, apply=True, ack=RUN_ACK, scene_workspace_retirement_enabled=True,
                   scene_inventory_cache_root=cache_root)
    assert not scene.exists() and not cache.exists()


def test_a_tick_defers_scenes_beyond_its_hashing_budget(tmp_path) -> None:
    scene, common = _scene_tick(tmp_path)

    phase = run_storage_gc(**common, scene_hash_budget_bytes=16)["scene_workspaces"]

    assert phase["retained_counts"] == {"inventory_deferred": 1} and phase["hashed_bytes"] <= 16


def test_the_command_line_caches_scene_digests_with_the_spool(tmp_path, monkeypatch) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(gc_module, "run_storage_gc", lambda **kwargs: seen.append(kwargs) or {"report_digest": ""})
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_GC_REPORT_ROOT", str(tmp_path / "storage-gc"))

    assert gc_main(["run", "--pins-root", str(tmp_path / "pins")]) == 0

    assert seen[0]["scene_inventory_cache_root"] is None  # context derives each spool's cache root
