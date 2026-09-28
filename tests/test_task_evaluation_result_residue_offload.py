# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_result_residue_offload.py
#   src/blueprint_pipeline/task_evaluation_result_residue_scan.py
#   src/blueprint_pipeline/task_evaluation_result_residue_restore.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
#   src/blueprint_pipeline/control_plane_storage_gc_reasons.py
#   src/blueprint_pipeline/control_plane_evidence_offload.py
#   src/blueprint_pipeline/control_plane_storage_references.py
#   src/blueprint_pipeline/completed_replay_cache_retention.py
#   src/blueprint_pipeline/task_evaluation_configured_scene_object_store.py
"""A sealed result run's residue moves to the artifact store behind a verified pointer, and comes back.

On 2026-09-27 evidence offload kept 27 result-registry runs (12.94 GB): whole-run
offload skips every run with a result registry, and per-artifact offload moves only
registered bulk payloads. Everything else in those runs stayed forever.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import control_plane_evidence_offload as evidence
from blueprint_pipeline import control_plane_storage_gc as gc_module
from blueprint_pipeline import task_evaluation_configured_scene_object_store as store
from blueprint_pipeline import task_evaluation_result_artifact_store as artifacts
from blueprint_pipeline import task_evaluation_result_residue_offload as residue
from blueprint_pipeline import task_evaluation_result_residue_scan as scan
from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk
from blueprint_pipeline.control_plane_replay_cache_gc import replay_cache_retention_setting
from blueprint_pipeline.control_plane_storage_gc import RUN_ACK, run_storage_gc
from blueprint_pipeline.control_plane_storage_gc_reasons import build_storage_gc_summary
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from tests.test_task_evaluation_configured_scene_object_store import _ContentAddressedClient

BUCKET = "blueprint-production-inputs"
NOW = 50_000_000.0
DAY = 86400
OLD = NOW - 30 * DAY
# Registered artifacts, relative to the run: a bulk payload the fixture offloads, a bulk-role
# frame below the 64 KiB floor and a non-bulk score. Only the first ever leaves.
REGISTERED = {
    "evidence/review.mp4": ("review_video", b"v" * 100_000),
    "evidence/frame.png": ("retained_lossless_frame", b"f" * 2_000),
    "evidence/score.json": ("task_score", b'{"score": 1}'),
}
RESIDUE = {
    "logs/worker.log": b"stage line\n" * 400,
    "work/stage/state.npz": b"s" * 120_000,
    "provider/outputs.zip": b"z" * 70_000,
}
# Unregistered files a live reader reopens after the seal. The dispatch receipt binds the terminal
# result by path; billing re-validation and spend ledgers reopen what that result names in turn.
TERMINAL_RESULT = "policy_canary_terminal_result.json"
ADAPTER_RESULT = "attempts/attempt_001/vast_provider_run/vast_provider_adapter_result.json"
ARTIFACT_MANIFEST = "attempts/attempt_001/artifact_manifest.json"
INSTANCE_ID = "attempts/attempt_001/started_vast_instance_id.txt"
KEPT_FOR_READERS = {TERMINAL_RESULT: "reader_reopened", ADAPTER_RESULT: "receipt_referenced",
                    ARTIFACT_MANIFEST: "receipt_referenced", INSTANCE_ID: "receipt_referenced"}
BY_DESIGN = frozenset(KEPT_FOR_READERS.values())


@pytest.fixture(autouse=True)
def hermetic_host(tmp_path, monkeypatch):
    ledger = tmp_path / "disk-reservations"
    roomy = functools.partial(
        reserve_control_plane_disk,
        disk_usage=lambda _: SimpleNamespace(total=100 * 1024**3, free=80 * 1024**3),
    )
    monkeypatch.setattr(evidence, "reserve_control_plane_disk", roomy)
    monkeypatch.setattr(evidence, "DEFAULT_RESERVATION_ROOT", ledger)
    monkeypatch.setattr(artifacts, "reserve_control_plane_disk", functools.partial(roomy, reservation_root=ledger))
    # An empty process table: nothing holds any run.
    monkeypatch.setattr("blueprint_pipeline.completed_replay_cache_retention.process_reference", lambda _, **kw: None)


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _age(root: Path, when: float = OLD) -> None:
    for path in (root, *root.rglob("*")):
        if not path.is_symlink():
            os.utime(path, (when, when))


def _sealed_run(evidence_root: Path, name: str = "run-1", *, residue_files=RESIDUE, bulk_remote: bool = True,
                client=None) -> SimpleNamespace:
    """A terminal, sealed result run: registry, delivery, closure receipts, registered files and residue."""

    run = evidence_root / name
    evidence_dir = run / "evidence"
    evidence_dir.mkdir(parents=True)
    records = []
    for index, (relative, (role, data)) in enumerate(REGISTERED.items()):
        (run / relative).write_bytes(data)
        records.append({"artifact_id": f"artifact-{index}", "role": role,
                        "relative_path": Path(relative).relative_to("evidence").as_posix(),
                        "evidence_root": str(evidence_dir), "sha256": _sha(data), "size_bytes": len(data)})
    closure = {}
    for kind in ("billing", "teardown", "provider_zero"):
        data = b'{"status":"closed"}'
        path = run / "closure" / f"{kind}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(data)
        records.append({"artifact_id": kind, "role": f"closure_{kind}", "relative_path": f"closure/{kind}.json",
                        "evidence_root": str(run), "sha256": _sha(data), "size_bytes": len(data)})
        closure[f"{kind}_receipt"] = {"artifact_id": kind, "digest": _sha(data), "size_bytes": len(data)}
    closure["provider_zero_receipt"]["provider_zero_verified"] = True
    delivery = {"schema_version": "task_evaluation_result_delivery.v2", "run_id": f"{name}-id",
                "result_status": "completed_unqualified", "reproducibility": closure, "delivery_digest": ""}
    delivery["delivery_digest"] = cross_runtime_canonical_digest(delivery, digest_field="delivery_digest")
    registry = {"schema_version": "task_evaluation_result_artifact_registry.v1", "run_id": f"{name}-id",
                "delivery_digest": delivery["delivery_digest"], "artifacts": records, "registry_digest": ""}
    registry["registry_digest"] = canonical_digest(registry, digest_field="registry_digest")
    delivery_dir = run / "artifacts" / "result_delivery"
    delivery_dir.mkdir(parents=True)
    (delivery_dir / "artifact_registry.json").write_text(json.dumps(registry), encoding="utf-8")
    (delivery_dir / "delivery.json").write_text(json.dumps(delivery), encoding="utf-8")
    documents = {
        # An absolute path names the adapter result; the adapter result names its instance id relatively.
        TERMINAL_RESULT: {"adapter_result_path": str(run / ADAPTER_RESULT),
                          "artifact_manifest_path": str(run / ARTIFACT_MANIFEST)},
        ADAPTER_RESULT: {"started_instance_id_path": INSTANCE_ID, "vast_instance_ids": [7]},
        ARTIFACT_MANIFEST: {"artifacts": []},
    }
    files = {relative: json.dumps(value).encode() for relative, value in documents.items()}
    files[INSTANCE_ID] = b"7\n"
    files["launch_receipt.json"] = b'{"launch": 1}'
    for relative, data in {**files, **residue_files}.items():
        (run / relative).parent.mkdir(parents=True, exist_ok=True)
        (run / relative).write_bytes(data)
    receipt = {"schema_version": "task_evaluation_policy_canary_dispatch.v1", "status": "completed",
               "run_id": registry["run_id"], "run_kind": "internal_policy_canary",
               "terminal_result": {"path": str(run / TERMINAL_RESULT)}, "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    (run / "dispatch_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    _age(run)
    client = client or _ContentAddressedClient()
    if bulk_remote:
        result = artifacts.offload_result_artifacts(
            run_root=run, apply=True, ack=artifacts.APPLY_ACK, hot_window_seconds=0, now=lambda: NOW,
            publisher=functools.partial(store.publish_configured_scene_artifact, client=client, bucket=BUCKET))
        assert (result["offloaded_count"], result["skipped"]) == (1, [])
    return SimpleNamespace(run=run, evidence=evidence_root, client=client, registry=registry,
                           pointer=evidence_root / f"{name}{residue.POINTER_SUFFIX}",
                           publisher=functools.partial(store.publish_configured_scene_artifact,
                                                       client=client, bucket=BUCKET))


def _offload(f, **kwargs):
    options = {"hot_window_seconds": 2 * DAY, "publisher": f.publisher, "now": lambda: NOW,
               "archive_verifier": functools.partial(store.verify_configured_scene_artifact, client=f.client,
                                                     bucket=BUCKET), **kwargs}
    return residue.offload_result_residue(run_root=f.run, apply=True, ack=residue.APPLY_ACK, **options)


def _local_files(run: Path) -> dict[str, bytes]:
    return {path.relative_to(run).as_posix(): path.read_bytes()
            for path in run.rglob("*") if path.is_file() and not path.is_symlink()}


def _kept_after_offload(run: Path) -> set[str]:
    """What must stay local however the residue goes: registry metadata, receipts and registered files."""
    return {path for path in _local_files(run) if path not in RESIDUE}


def _changed(rows) -> list[dict]:
    """Skipped rows other than the files kept for their readers, which every fixture run has."""
    return [row for row in rows if row["reason"] not in BY_DESIGN]


def _reader_bytes(run: Path, reason: str) -> dict[str, int]:
    kept = [relative for relative, why in KEPT_FOR_READERS.items() if why == reason]
    return {"count": len(kept), "bytes": sum((run / relative).stat().st_size for relative in kept)}


def test_residue_excludes_registry_receipts_and_registered_files(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries")
    run = f.run
    (run / "work" / "nested").mkdir()
    (run / "work" / "nested" / "launch_receipt.json").write_text("{}", encoding="utf-8")
    # A link out of the run keeps nothing inside it; links that stay inside are tested below.
    (tmp_path / "outside.bin").write_bytes(b"not in the run")
    (run / "work" / "link.bin").symlink_to(tmp_path / "outside.bin")
    os.mkfifo(run / "work" / "pipe")
    (run / "logs" / "fresh.log").write_text("written after the seal", encoding="utf-8")
    # Readers reopen these whole directories and names; a kept document that does not even parse
    # still keeps the file it names.
    (run / "episode_interpretation").mkdir()
    (run / "episode_interpretation" / "notes.log").write_text("interpretation", encoding="utf-8")
    (run / "episode_interpretation" / "broken.json").write_text(
        '{"source": "' + str(run / "work" / "cited.bin") + '", ', encoding="utf-8")
    (run / "work" / "cited.bin").write_bytes(b"cited by an interpretation")
    _age(run / "work")
    _age(run / "episode_interpretation")
    os.utime(run / "logs" / "fresh.log", (OLD + DAY, OLD + DAY))

    plan = residue.offload_result_residue(run_root=run, hot_window_seconds=2 * DAY, now=lambda: NOW)

    assert plan["status"] == "dry_run" and plan["retained_reason"] is None
    assert plan["candidate_count"] == len(RESIDUE)
    assert plan["candidate_bytes"] == sum(len(data) for data in RESIDUE.values())
    assert plan["registry_digest"] == f.registry["registry_digest"]
    reader_bytes = _reader_bytes(run, "reader_reopened")
    reader_bytes["count"] += 2
    reader_bytes["bytes"] += sum((run / "episode_interpretation" / name).stat().st_size
                                 for name in ("notes.log", "broken.json"))
    referenced = _reader_bytes(run, "receipt_referenced")
    referenced["count"] += 1
    referenced["bytes"] += len(b"cited by an interpretation")
    assert plan["skipped_by_reason"] == {
        "symlink": {"count": 1, "bytes": (run / "work" / "link.bin").lstat().st_size},
        "special_file": {"count": 1, "bytes": 0},
        "newer_than_registry": {"count": 1, "bytes": len("written after the seal")},
        "reader_reopened": reader_bytes,
        "receipt_referenced": referenced,
    }
    assert {row["relative_path"]: row["reason"] for row in plan["skipped"]} == {
        "work/link.bin": "symlink", "work/pipe": "special_file", "logs/fresh.log": "newer_than_registry",
        "episode_interpretation/notes.log": "reader_reopened",
        "episode_interpretation/broken.json": "reader_reopened", "work/cited.bin": "receipt_referenced",
        **KEPT_FOR_READERS}
    # A dry run changes nothing and publishes nothing.
    assert not f.pointer.exists() and f.client.upload_count == 1

    receipt = _offload(f)
    assert (receipt["status"], receipt["offloaded_count"]) == ("applied", len(RESIDUE))
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    assert sorted(row["relative_path"] for row in pointer["members"]) == sorted(RESIDUE)
    left = _local_files(run)
    assert not set(RESIDUE) & set(left)
    for relative in ("artifacts/result_delivery/artifact_registry.json", "artifacts/result_delivery/delivery.json",
                     "evidence/frame.png", "evidence/score.json", "closure/billing.json", "closure/teardown.json",
                     "closure/provider_zero.json", "dispatch_receipt.json", "launch_receipt.json",
                     "work/nested/launch_receipt.json", "logs/fresh.log", "work/cited.bin",
                     "episode_interpretation/notes.log", *KEPT_FOR_READERS):
        assert relative in left, relative
    remote = [path for path in left if path.startswith("artifacts/result_delivery/remote_artifacts/")]
    assert len(remote) == 1
    assert (run / "work" / "link.bin").is_symlink() and stat.S_ISFIFO((run / "work" / "pipe").lstat().st_mode)
    # The bulk payload stayed remote, the registered small files stayed local.
    assert not (run / "evidence" / "review.mp4").exists()


@pytest.mark.parametrize("case", [
    "kept_link_to_residue", "directory_link", "jsonl_document", "text_document", "skipped_named_document",
    "document_relative", "ancestor_relative", "evidence_root_relative", "link_to_run_root",
])
def test_what_a_reader_can_reach_from_a_kept_file_stays(tmp_path, case) -> None:
    """Whatever a reader can reach from a kept file stays: a link's target inside the run, a file
    any kept or named text document names (JSON or not, absolutely, relative to the run, the
    evidence root or any directory above the document), and what that file names in turn."""

    f = _sealed_run(tmp_path / "canaries")
    run = f.run
    reached, reason = {"logs/worker.log"}, "receipt_referenced"
    interpretation = run / "episode_interpretation"
    interpretation.mkdir()
    if case == "kept_link_to_residue":
        (interpretation / "state.npz").symlink_to(Path("..") / "work" / "stage" / "state.npz")
        reached, reason = {"work/stage/state.npz"}, "symlink_target"
    elif case == "directory_link":
        (run / "latest").symlink_to(Path("work") / "stage", target_is_directory=True)
        reached, reason = {"work/stage/state.npz"}, "symlink_target"
    elif case == "link_to_run_root":
        (run / "work" / "everything").symlink_to(run, target_is_directory=True)
        reached, reason = set(RESIDUE), "symlink_target"
    elif case == "jsonl_document":
        (interpretation / "rows.jsonl").write_text(
            json.dumps({"row": 1}) + "\n" + json.dumps({"log": str(run / "logs" / "worker.log")}) + "\n",
            encoding="utf-8")
    elif case == "text_document":
        (interpretation / "notes.txt").write_text("see logs/worker.log for the stage output\n", encoding="utf-8")
    elif case == "skipped_named_document":
        # The named JSON is newer than the registry, so it stays, and it is searched anyway.
        (interpretation / "index.json").write_text(json.dumps({"next": str(run / "work" / "late.json")}),
                                                   encoding="utf-8")
        (run / "work" / "late.json").write_text(json.dumps({"log": "logs/worker.log"}), encoding="utf-8")
    elif case == "document_relative":
        adapter = run / ADAPTER_RESULT
        adapter.write_text(json.dumps({**json.loads(adapter.read_text()), "log": "provider.log"}), encoding="utf-8")
        (adapter.parent / "provider.log").write_text("provider output\n", encoding="utf-8")
        reached = {"attempts/attempt_001/vast_provider_run/provider.log"}
    elif case == "ancestor_relative":
        adapter = run / ADAPTER_RESULT
        adapter.write_text(json.dumps({**json.loads(adapter.read_text()), "log": "vast_provider_run/provider.log"}),
                           encoding="utf-8")
        (adapter.parent / "provider.log").write_text("provider output\n", encoding="utf-8")
        reached = {"attempts/attempt_001/vast_provider_run/provider.log"}
    elif case == "evidence_root_relative":
        (interpretation / "index.json").write_text(json.dumps({"log": f"{run.name}/logs/worker.log"}),
                                                   encoding="utf-8")
    _age(run)
    if case == "skipped_named_document":
        os.utime(run / "work" / "late.json", (OLD + DAY, OLD + DAY))
    before = {relative: (run / relative).read_bytes() for relative in reached}

    plan = residue.offload_result_residue(run_root=run, hot_window_seconds=2 * DAY, now=lambda: NOW)

    reasons = {row["relative_path"]: row["reason"] for row in plan["skipped"]}
    assert {relative: reasons.get(relative) for relative in reached} == dict.fromkeys(reached, reason)
    assert _offload(f)["status"] == "applied"
    assert {relative: (run / relative).read_bytes() for relative in reached} == before


def test_a_large_reached_text_is_searched_as_a_stream(tmp_path, monkeypatch) -> None:
    """A kept or reached text is searched a chunk at a time, whatever its size: a log many
    chunks long naming a file keeps that file, even where the name straddles two chunks, and
    the run plans instead of failing. Memory stays at a chunk and a carried token. The chunk
    is shrunk to 64 KiB so a 3 MiB log spans the same forty-odd chunks a 70 MiB one would."""

    monkeypatch.setattr(scan, "_SCAN_CHUNK_BYTES", 64 * 1024)
    f = _sealed_run(tmp_path / "canaries")
    interpretation = f.run / "episode_interpretation"
    interpretation.mkdir()
    chunk, reference, line = scan._SCAN_CHUNK_BYTES, b"logs/worker.log", b"stage 0000001 ok\n"
    boundary = 40 * chunk
    with (interpretation / "rollout.log").open("wb") as stream:
        lines = (boundary - 7) // len(line)
        stream.write(line * lines)
        stream.write(b" " * (boundary - 7 - lines * len(line)))
        stream.write(reference + b"\n")  # seven bytes before the 41st chunk begins, eight after
        stream.write(line * ((3 * 1024 * 1024 - stream.tell()) // len(line) + 1))
    assert (interpretation / "rollout.log").stat().st_size > 3 * 1024 * 1024
    _age(f.run)

    plan = residue.offload_result_residue(run_root=f.run, hot_window_seconds=2 * DAY, now=lambda: NOW)

    assert plan["status"] == "dry_run"
    reasons = {row["relative_path"]: row["reason"] for row in plan["skipped"]}
    assert reasons["logs/worker.log"] == "receipt_referenced"
    assert plan["candidate_count"] == len(RESIDUE) - 1


@pytest.mark.parametrize("case", ["linked_reader_directory", "linked_kept_document", "unlistable_directory"])
def test_a_kept_directory_or_document_that_cannot_be_searched_keeps_the_whole_run(tmp_path, monkeypatch, case):
    """A reader follows a link the walk will not: whatever the target names inside the run is
    unknown, and so is what an unlistable directory holds. Nothing moves."""

    f = _sealed_run(tmp_path / "canaries")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "notes.json").write_text(json.dumps({"cites": str(f.run / "logs" / "worker.log")}), encoding="utf-8")
    if case == "linked_reader_directory":
        (f.run / "episode_interpretation").symlink_to(elsewhere, target_is_directory=True)
    elif case == "linked_kept_document":
        (f.run / TERMINAL_RESULT).unlink()
        (f.run / TERMINAL_RESULT).symlink_to(elsewhere / "notes.json")
    else:
        real_walk = os.walk

        def failing_walk(top, onerror=None, **kwargs):
            onerror(PermissionError(13, "unlistable", str(Path(top) / "logs")))
            yield from real_walk(top, onerror=onerror, **kwargs)

        monkeypatch.setattr(residue.os, "walk", failing_walk)
    before = _local_files(f.run)

    result = _offload(f)

    assert (result["status"], result["retained_reason"]) == ("retained", "plan_failed")
    assert result["failure"]["error_type"] == "ResultResidueOffloadError"
    assert _local_files(f.run) == before and not f.pointer.exists()


def test_a_hardlinked_member_moves_with_every_name_and_one_linked_elsewhere_stays(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries")
    os.link(f.run / "logs" / "worker.log", f.run / "logs" / "worker-copy.log")
    os.link(f.run / "evidence" / "score.json", f.run / "work" / "score-copy.json")

    plan = residue.offload_result_residue(run_root=f.run, hot_window_seconds=2 * DAY, now=lambda: NOW)

    assert plan["candidate_count"] == len(RESIDUE) + 1
    # One inode, two names: its bytes count once.
    assert plan["candidate_bytes"] == sum(len(data) for data in RESIDUE.values())
    assert _changed(plan["skipped"]) == [{"relative_path": "work/score-copy.json", "reason": "linked_outside_residue"}]

    result = _offload(f)
    assert (result["offloaded_count"], result["offloaded_bytes"]) == (len(RESIDUE) + 1, plan["candidate_bytes"])
    assert not (f.run / "logs" / "worker-copy.log").exists()
    assert (f.run / "work" / "score-copy.json").read_bytes() == REGISTERED["evidence/score.json"][1]
    restored = residue.restore_result_residue(run_root=f.run, now=lambda: NOW, materializer=functools.partial(
        store.materialize_configured_scene_artifact, client=f.client, bucket=BUCKET))
    assert restored["restored_count"] == len(RESIDUE) + 1
    assert (f.run / "logs" / "worker-copy.log").read_bytes() == RESIDUE["logs/worker.log"]


def test_residue_refuses_until_bulk_is_remote(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries", bulk_remote=False)

    waiting = _offload(f)

    assert (waiting["status"], waiting["retained_reason"]) == ("retained", "bulk_not_remote")
    assert waiting["candidate_bytes"] is None and not f.pointer.exists()
    assert set(RESIDUE) <= set(_local_files(f.run)) and f.client.upload_count == 0

    artifacts.offload_result_artifacts(run_root=f.run, apply=True, ack=artifacts.APPLY_ACK, hot_window_seconds=0,
                                       now=lambda: NOW, publisher=f.publisher)
    ready = _offload(f)
    assert (ready["status"], ready["offloaded_count"]) == ("applied", len(RESIDUE))


@pytest.mark.parametrize("case", ["hot", "protected_reason", "protected_true", "pointed", "symlinked_root",
                                  "unsealed", "not_authorized", "locked", "operator_run", "foreign_receipt"])
def test_residue_refuses_hot_or_protected_or_already_pointed_runs(tmp_path, case) -> None:
    import fcntl

    f = _sealed_run(tmp_path / "canaries")
    run_root = f.run
    options: dict = {}
    expected = case
    if case == "locked":
        # A per-artifact offload of the same run holds its exclusive lock.
        holder = (f.run / "artifacts/result_delivery/.offload.lock").open("a+b")
        fcntl.flock(holder, fcntl.LOCK_EX)
        expected = "offload_locked"
    if case == "hot":
        os.utime(f.run / "artifacts/result_delivery/artifact_registry.json", (NOW - 3600, NOW - 3600))
    elif case == "protected_reason":
        options["protection_checker"] = lambda root: "protected_queue" if root == f.run.resolve() else None
        expected = "protected_queue"
    elif case == "protected_true":
        options["protection_checker"] = lambda _root: True
        expected = "protected"
    elif case == "pointed":
        # A pointer that does not verify: the run is left alone, never resumed or re-offloaded.
        f.pointer.write_text("{}", encoding="utf-8")
        expected = "pointer_invalid"
    elif case == "symlinked_root":
        run_root = tmp_path / "canaries" / "alias"
        run_root.symlink_to(f.run)
        expected = "run_root_invalid"
    elif case == "unsealed":
        delivery = f.run / "artifacts/result_delivery/delivery.json"
        value = json.loads(delivery.read_text(encoding="utf-8"))
        value["result_status"] = "running"
        delivery.write_text(json.dumps(value), encoding="utf-8")
        _age(f.run)
        expected = "registry_unsealed"
    elif case == "operator_run":
        # An operator run has a sealed registry but no dispatch receipt; its continuation, terminal
        # delivery and download route keep reopening its files.
        (f.run / "dispatch_receipt.json").unlink()
        expected = "dispatch_receipt_missing"
    elif case == "foreign_receipt":
        receipt = {"run_id": "another-run", "receipt_digest": ""}
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        (f.run / "dispatch_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
        _age(f.run)
        expected = "dispatch_receipt_invalid"
    before = _local_files(f.run)

    if case == "not_authorized":
        with pytest.raises(residue.ResultResidueOffloadError, match="not_authorized"):
            residue.offload_result_residue(run_root=run_root, apply=True, ack="wrong", now=lambda: NOW)
    else:
        result = residue.offload_result_residue(
            run_root=run_root, apply=True, ack=residue.APPLY_ACK, hot_window_seconds=2 * DAY,
            publisher=f.publisher, now=lambda: NOW, **options)
        assert (result["status"], result["retained_reason"]) == ("retained", expected)
        assert result["result_digest"] == canonical_digest(result, digest_field="result_digest")

    assert _local_files(f.run) == before
    assert f.client.upload_count == 1  # only the fixture's bulk artifact
    if case != "pointed":
        assert not f.pointer.exists()


def test_residue_offload_writes_pointer_before_evicting(tmp_path, monkeypatch) -> None:
    f = _sealed_run(tmp_path / "canaries")
    events: list[tuple] = []
    names = {Path(relative).name for relative in RESIDUE}

    def publisher(**kwargs):
        events.append(("publish", f.pointer.exists(), set(RESIDUE) <= set(_local_files(f.run))))
        return f.publisher(**kwargs)

    real_replace, real_unlink = os.replace, os.unlink

    def replace(source, destination, *args, **kwargs):
        outcome = real_replace(source, destination, *args, **kwargs)
        if str(destination).endswith(residue.POINTER_SUFFIX):
            events.append(("pointer", json.loads(f.pointer.read_text(encoding="utf-8"))["state"]))
        return outcome

    def unlink(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None and os.fspath(path) in names:
            events.append(("unlink", f.pointer.is_file(), os.fspath(path)))
        return real_unlink(path, *args, **kwargs)

    modes = {relative: stat.S_IMODE((f.run / relative).stat().st_mode) for relative in RESIDUE}
    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "unlink", unlink)

    result = _offload(f, publisher=publisher)

    assert result["status"] == "applied" and result["offloaded_count"] == len(RESIDUE)
    kinds = [event[0] for event in events]
    assert kinds[0] == "publish" and events[0][1:] == (False, True)
    assert kinds.index("pointer") < kinds.index("unlink")
    # Written evicting before the first unlink, and offloaded once the last is done, even with nothing kept.
    assert [event[1] for event in events if event[0] == "pointer"] == ["evicting", "offloaded"]
    assert len(kinds) - 1 - kinds[::-1].index("pointer") > len(kinds) - 1 - kinds[::-1].index("unlink")
    unlinks = [event for event in events if event[0] == "unlink"]
    assert all(pointer_present for _kind, pointer_present, _name in unlinks)
    assert sorted(name for *_rest, name in unlinks) == sorted(names)
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    assert pointer["schema_version"] == "control_plane_result_residue_pointer.v1"
    assert pointer["pointer_digest"] == canonical_digest(pointer, digest_field="pointer_digest")
    assert (pointer["run"], pointer["registry_digest"], pointer["kept"], pointer["state"]) == (
        f.run.name, f.registry["registry_digest"], [], "offloaded")
    stored = f.client.objects[(BUCKET, pointer["archive"]["uri"].split(f"s3://{BUCKET}/", 1)[1])]
    assert (_sha(stored), len(stored)) == (pointer["archive"]["sha256"], pointer["archive"]["size_bytes"])
    # One inode group per name here, numbered in the plan's order.
    assert sorted(pointer["members"], key=lambda row: row["relative_path"]) == [
        {"relative_path": relative, "size_bytes": len(RESIDUE[relative]), "sha256": _sha(RESIDUE[relative]),
         "mode": modes[relative], "group": index}
        for index, relative in enumerate(sorted(RESIDUE))
    ]
    assert stat.S_IMODE(f.pointer.stat().st_mode) == 0o440
    assert not [path for path in f.evidence.iterdir() if path.name.startswith(".")]


def test_residue_streams_without_a_local_archive(tmp_path, monkeypatch) -> None:
    """The unit passes no file publisher: the tar is hashed, then streamed as a multipart upload
    and read back, and only the pointer needs local headroom."""

    from tests.test_control_plane_evidence_streaming import MultipartClient

    f = _sealed_run(tmp_path / "canaries")
    client = MultipartClient()
    reserved: list[int] = []

    def reserve(*args, **kwargs):
        reserved.append(kwargs["expected_bytes"])
        return reserve_control_plane_disk(*args, **{**kwargs, "disk_usage": lambda _: SimpleNamespace(
            total=100 * 1024**3, free=80 * 1024**3)})

    real_mkstemp = evidence.tempfile.mkstemp

    def no_archive(*args, **kwargs):
        assert ".residue-" not in kwargs.get("prefix", ""), "the stream path must not stage a tar"
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(evidence, "reserve_control_plane_disk", reserve)
    monkeypatch.setattr(evidence.tempfile, "mkstemp", no_archive)

    result = residue.offload_result_residue(
        run_root=f.run, apply=True, ack=residue.APPLY_ACK, hot_window_seconds=2 * DAY, now=lambda: NOW,
        stream_publisher=functools.partial(store.publish_configured_scene_stream, client=client, bucket=BUCKET))

    assert (result["status"], result["offloaded_count"]) == ("applied", len(RESIDUE))
    assert reserved and max(reserved) < 2 * 1024**2
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    key = pointer["archive"]["uri"].split(f"s3://{BUCKET}/", 1)[1]
    assert key.endswith("/residue.tar") and client.upload_count == 1
    assert _sha(client.objects[(BUCKET, key)]) == pointer["archive"]["sha256"]
    restored = residue.restore_result_residue(run_root=f.run, now=lambda: NOW, materializer=functools.partial(
        store.materialize_configured_scene_artifact, client=client, bucket=BUCKET))
    assert restored["restored_count"] == len(RESIDUE)
    assert {relative: (f.run / relative).read_bytes() for relative in RESIDUE} == RESIDUE


def test_readback_failure_or_changed_member_keeps_files(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries")
    before = _local_files(f.run)

    f.client.corrupt_readback = True
    corrupt = _offload(f)
    f.client.corrupt_readback = False
    assert (corrupt["status"], corrupt["retained_reason"]) == ("retained", "publication_failed")
    assert corrupt["failure"] == {"error_type": "TaskEvaluationConfiguredSceneObjectStoreError", "errno": None,
                                  "stage": "publish"}

    def lying(*, path, artifact_kind):
        return {"uri": f"s3://{BUCKET}/elsewhere", "digest": "sha256:" + "0" * 64,
                "size_bytes": Path(path).stat().st_size, "full_byte_service_account_readback_passed": True}

    lied = _offload(f, publisher=lying)
    assert (lied["status"], lied["retained_reason"]) == ("retained", "publication_failed")
    # The whole-run offload's own publication check refused it.
    assert lied["failure"] == {"error_type": "ControlPlaneEvidenceOffloadError", "errno": None, "stage": "publish"}
    assert _local_files(f.run) == before and not f.pointer.exists()
    assert not [path for path in f.evidence.iterdir() if path.name.startswith(".")]

    changed = f.run / "logs" / "worker.log"

    def touching(**kwargs):
        reference = f.publisher(**kwargs)
        changed.write_bytes(b"a line written after the archive was packed\n")
        return reference

    result = _offload(f, publisher=touching)

    assert result["status"] == "applied"
    assert result["offloaded_count"] == len(RESIDUE) - 1
    assert _changed(result["skipped"]) == [{"relative_path": "logs/worker.log", "reason": "member_changed"}]
    assert changed.read_bytes() == b"a line written after the archive was packed\n"
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    assert pointer["kept"] == [{"relative_path": "logs/worker.log", "reason": "member_changed"}]
    assert pointer["pointer_digest"] == canonical_digest(pointer, digest_field="pointer_digest")
    assert not (f.run / "work/stage/state.npz").exists() and not (f.run / "provider/outputs.zip").exists()


@pytest.mark.parametrize("swap", ["symlink", "fifo", "replaced"])
def test_a_member_swapped_while_it_is_packed_fails_publication(tmp_path, monkeypatch, swap) -> None:
    """The packer opens each member without following a link or blocking and requires the planned
    regular file, so a name swapped after it was listed can neither put another object in the
    archive (which restore would refuse) nor hang the tick on a FIFO."""

    f = _sealed_run(tmp_path / "canaries")
    (tmp_path / "secret.zip").write_bytes(b"outside the run")
    target = f.run / "provider" / "outputs.zip"
    real_listed = evidence._listed_members

    def racing(directory, members):
        for path, relative in real_listed(directory, members):
            if relative == "provider/outputs.zip":
                target.unlink()
                if swap == "symlink":
                    target.symlink_to(tmp_path / "secret.zip")
                elif swap == "fifo":
                    os.mkfifo(target)
                else:
                    target.write_bytes(RESIDUE["provider/outputs.zip"])
            yield path, relative

    monkeypatch.setattr(evidence, "_listed_members", racing)

    result = _offload(f)

    assert (result["status"], result["retained_reason"]) == ("retained", "publication_failed")
    assert result["failure"] == {"error_type": "ControlPlaneEvidenceOffloadError", "errno": None, "stage": "publish"}
    assert not f.pointer.exists() and f.client.upload_count == 1
    assert {relative: (f.run / relative).read_bytes() for relative in ("logs/worker.log", "work/stage/state.npz")} == {
        relative: RESIDUE[relative] for relative in ("logs/worker.log", "work/stage/state.npz")}
    assert not [path for path in f.evidence.iterdir() if path.name.startswith(".")]


def test_a_partly_unlinked_hardlink_group_is_restorable(tmp_path, monkeypatch) -> None:
    """A group's names go one by one. When a later unlink fails, the names already gone are
    offloaded (restore brings them back) and only the names still present are kept."""

    f = _sealed_run(tmp_path / "canaries")
    os.link(f.run / "logs" / "worker.log", f.run / "logs" / "worker-copy.log")
    real_unlink = os.unlink

    def sticky(path, *args, **kwargs):
        # The second name of the group (sorted: worker-copy.log, then worker.log) cannot go.
        if kwargs.get("dir_fd") is not None and os.fspath(path) == "worker.log":
            raise PermissionError(1, "Operation not permitted")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", sticky)
    result = _offload(f)
    monkeypatch.setattr(os, "unlink", real_unlink)

    assert result["status"] == "applied"
    assert _changed(result["skipped"]) == [
        {"relative_path": "logs/worker.log", "reason": "unlink_failed:PermissionError"}]
    # Two whole groups and one name of the third went; the third's blocks are still held.
    assert result["offloaded_count"] == len(RESIDUE)
    assert result["offloaded_bytes"] == len(RESIDUE["work/stage/state.npz"]) + len(RESIDUE["provider/outputs.zip"])
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    assert pointer["kept"] == [{"relative_path": "logs/worker.log", "reason": "unlink_failed:PermissionError"}]
    assert not (f.run / "logs" / "worker-copy.log").exists()

    restored = residue.restore_result_residue(run_root=f.run, now=lambda: NOW, materializer=functools.partial(
        store.materialize_configured_scene_artifact, client=f.client, bucket=BUCKET))

    assert restored["restored_count"] == len(RESIDUE)
    assert (f.run / "logs" / "worker-copy.log").read_bytes() == RESIDUE["logs/worker.log"]
    assert (f.run / "logs" / "worker.log").read_bytes() == RESIDUE["logs/worker.log"]


def test_a_run_whose_members_all_stay_is_planned_again(tmp_path, monkeypatch) -> None:
    """If nothing could be removed, the pointer is withdrawn so the next tick tries again; a
    pointer would otherwise mark the run offloaded forever."""

    f = _sealed_run(tmp_path / "canaries")

    def unavailable(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    with monkeypatch.context() as patched:
        patched.setattr(residue.held_files, "_HeldChild", unavailable)
        stuck = _offload(f)

    assert (stuck["status"], stuck["retained_reason"]) == ("retained", "nothing_evicted")
    assert {row["reason"] for row in _changed(stuck["skipped"])} == {"root_unavailable:PermissionError"}
    assert not f.pointer.exists() and set(RESIDUE) <= set(_local_files(f.run))

    again = _offload(f)
    assert (again["status"], again["offloaded_count"]) == ("applied", len(RESIDUE))


def test_residue_member_swapped_for_symlink_is_kept(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "outputs.zip").write_bytes(b"not evidence")
    (outside / "state.npz").write_bytes(b"not evidence either")

    def swapping(**kwargs):
        reference = f.publisher(**kwargs)
        # A file becomes a link out of the run, and a directory becomes a link to another tree.
        (f.run / "provider" / "outputs.zip").unlink()
        (f.run / "provider" / "outputs.zip").symlink_to(outside / "outputs.zip")
        (f.run / "work" / "stage").rename(f.run / "work" / "stage.moved")
        (f.run / "work" / "stage").symlink_to(outside, target_is_directory=True)
        return reference

    result = _offload(f, publisher=swapping)

    assert result["status"] == "applied" and result["offloaded_count"] == 1
    reasons = {row["relative_path"]: row["reason"] for row in result["skipped"]}
    assert reasons["provider/outputs.zip"] == "member_changed"
    assert reasons["work/stage/state.npz"].startswith("recheck_failed:")
    assert (outside / "outputs.zip").read_bytes() == b"not evidence"
    assert (outside / "state.npz").read_bytes() == b"not evidence either"
    assert (f.run / "provider" / "outputs.zip").is_symlink() and (f.run / "work" / "stage").is_symlink()
    assert (f.run / "work" / "stage.moved" / "state.npz").read_bytes() == RESIDUE["work/stage/state.npz"]
    assert not (f.run / "logs" / "worker.log").exists()
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    assert {row["relative_path"] for row in pointer["kept"]} == {"provider/outputs.zip", "work/stage/state.npz"}


def test_residue_pointer_is_not_an_unsafe_evidence_entry(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries")
    _offload(f)

    manifest = evidence.build_evidence_offload_manifest(
        evidence_roots=[f.evidence], hot_window_seconds=0, now=lambda: NOW, classifier=lambda *a, **k: None)

    assert set(manifest["retained_by_reason"]) == {"result_registry"}


def _tick(f, pins, queue, **kwargs):
    options = {"content_store_roots": [], "derived_roots": [], "queue_roots": [queue], "pins_root": pins,
               "evidence_roots": [f.evidence], "hot_window_seconds": 2 * DAY, "now": lambda: NOW,
               "classifier": lambda *a, **k: None, "publisher": f.publisher, **kwargs}
    return run_storage_gc(**options)


def test_gc_residue_offload_is_plan_only_until_enabled(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries")
    pins, queue = tmp_path / "pins", tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    residue_bytes = sum(len(data) for data in RESIDUE.values())

    for apply, offload, enabled in ((True, True, False), (False, True, True), (True, False, True)):
        report = _tick(f, pins, queue, apply=apply, ack=RUN_ACK if apply else "", offload_enabled=offload,
                       result_residue_offload_enabled=enabled)
        phase = report["result_residue_offload"]
        assert (phase["status"], phase["enabled"]) == ("dry_run", enabled)
        assert (phase["candidate_bytes"], phase["offloaded_bytes"]) == (residue_bytes, 0)
        assert [row["status"] for row in phase["runs"]] == ["dry_run"]
        assert report["opt_in"]["result_residue_offload"] is enabled
        assert set(RESIDUE) <= set(_local_files(f.run)) and not f.pointer.exists()
    assert f.client.upload_count == 1
    kept = _kept_after_offload(f.run)

    applied = _tick(f, pins, queue, apply=True, ack=RUN_ACK, offload_enabled=True,
                    result_residue_offload_enabled=True)
    phase = applied["result_residue_offload"]
    assert (phase["status"], phase["enabled"]) == ("applied", True)
    assert (phase["candidate_bytes"], phase["offloaded_bytes"]) == (residue_bytes, residue_bytes)
    assert f.pointer.is_file() and not set(RESIDUE) & set(_local_files(f.run))
    assert kept == set(_local_files(f.run))
    assert "phase_errors" not in applied

    again = _tick(f, pins, queue, apply=True, ack=RUN_ACK, offload_enabled=True,
                  result_residue_offload_enabled=True)
    assert [row["retained_reason"] for row in again["result_residue_offload"]["runs"]] == ["already_offloaded"]
    assert f.client.upload_count == 2


@pytest.mark.parametrize("raw", ["1", "true", "YES", " yes ", "", "0", "false", "no", "maybe", "2", "on"])
def test_residue_setting_parses_like_other_opt_ins(raw) -> None:
    ours = residue.result_residue_offload_setting({residue.RESIDUE_OFFLOAD_ENV: raw})
    theirs = replay_cache_retention_setting({"BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_RETENTION": raw})

    assert ours[0] is theirs[0]
    assert ours[1] == (None if theirs[1] is None else residue.RESIDUE_OFFLOAD_INVALID)
    # It is its own decision: the evidence offload opt-in never enables it.
    assert residue.result_residue_offload_setting({"BLUEPRINT_CONTROL_PLANE_EVIDENCE_OFFLOAD": "1"}) == (False, None)


def _unit_environment(monkeypatch, values: dict[str, str]) -> None:
    """The unit's environment and none of the shell's: every storage GC variable is cleared,
    whatever it is, and only ``values`` are set."""

    for name in [name for name in os.environ if name.startswith("BLUEPRINT_CONTROL_PLANE_GC_")]:
        monkeypatch.delenv(name)
    # The storage GC reads these too, outside that prefix.
    for name in (gc_module.EVIDENCE_OFFLOAD_ENV, gc_module.EVIDENCE_HOT_WINDOW_ENV, gc_module.EVIDENCE_ABANDONED_AFTER_ENV,
                 gc_module.SCENE_WORKSPACE_RETIREMENT_ENV, gc_module.SCENE_BINDING_ROOT_ENV, gc_module.PINS_ROOT_ENV):
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def test_invalid_residue_setting_only_plans_and_alerts(tmp_path, monkeypatch, capsys) -> None:
    f = _sealed_run(tmp_path / "canaries")
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    _unit_environment(monkeypatch, {gc_module.QUEUE_ROOTS_ENV: str(queue), gc_module.EVIDENCE_ROOTS_ENV: str(f.evidence),
                                     gc_module.EVIDENCE_OFFLOAD_ENV: "1", residue.RESIDUE_OFFLOAD_ENV: "maybe"})
    monkeypatch.setattr(gc_module, "require_storage_class", lambda *a, **k: None)
    # The bulk offload runs as the unit would; its publisher is the fixture's fake store.
    monkeypatch.setattr(artifacts, "_artifact_object_store_client", lambda: (f.client, BUCKET))

    code = gc_module.main(["run", "--apply", "--ack", RUN_ACK, "--pins-root", str(tmp_path / "pins")])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert code == 0
    assert "storage_gc_alert:result_residue_offload_setting_invalid" in captured.err
    assert residue.RESIDUE_OFFLOAD_INVALID in report["alerts"]
    phase = report["result_residue_offload"]
    assert (phase["status"], phase["enabled"], phase["alerts"]) == ("dry_run", False, [residue.RESIDUE_OFFLOAD_INVALID])
    assert report["opt_in"]["result_residue_offload"] is False
    assert set(RESIDUE) <= set(_local_files(f.run)) and not f.pointer.exists()


def test_summary_reports_residue_candidates_offloads_retained_reasons_and_opt_in(tmp_path) -> None:
    evidence_root = tmp_path / "canaries"
    client = _ContentAddressedClient()
    ready = _sealed_run(evidence_root, "run-ready-secret", client=client)
    (tmp_path / "elsewhere.log").write_text("outside the run", encoding="utf-8")
    (ready.run / "logs" / "link.log").symlink_to(tmp_path / "elsewhere.log")
    waiting = _sealed_run(evidence_root, "run-waiting-secret", bulk_remote=False, client=client)
    hot = _sealed_run(evidence_root, "run-hot-secret", client=client)
    os.utime(hot.run / "artifacts/result_delivery/artifact_registry.json", (NOW - 3600, NOW - 3600))
    pins, queue = tmp_path / "pins", tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    residue_bytes = sum(len(data) for data in RESIDUE.values())
    link_bytes = (ready.run / "logs" / "link.log").lstat().st_size

    def for_readers(*runs) -> dict:
        return {f"member_skipped:{reason}": {key: sum(_reader_bytes(run.run, reason)[key] for run in runs)
                                             for key in ("count", "bytes")}
                for reason in sorted(BY_DESIGN)}

    planned = _tick(ready, pins, queue, apply=False)
    assert build_storage_gc_summary(planned)["opt_in"]["result_residue_offload"] is False
    phase = build_storage_gc_summary(planned)["phases"]["result_residue_offload"]

    assert phase == {
        "status": "dry_run", "enabled": False, "candidate_bytes": residue_bytes, "removed_or_offloaded_bytes": 0,
        "retained_by_reason": {
            "member_skipped:symlink": {"count": 1, "bytes": link_bytes},
            **for_readers(ready),
            "bulk_not_remote": {"count": 1, "bytes": None},
            "hot": {"count": 1, "bytes": None},
        },
    }

    applied = _tick(ready, pins, queue, apply=True, ack=RUN_ACK, offload_enabled=True,
                    result_residue_offload_enabled=True)
    summary = build_storage_gc_summary(applied)
    assert summary["opt_in"]["result_residue_offload"] is True
    phase = summary["phases"]["result_residue_offload"]
    # This tick offloaded the waiting run's bulk artifact first, so its residue followed in the same tick.
    assert (phase["status"], phase["enabled"]) == ("applied", True)
    assert phase["candidate_bytes"] == phase["removed_or_offloaded_bytes"] == 2 * residue_bytes
    assert phase["retained_by_reason"] == {
        "member_skipped:symlink": {"count": 1, "bytes": link_bytes}, **for_readers(ready, waiting),
        "hot": {"count": 1, "bytes": None}}
    text = json.dumps(summary)
    assert "secret" not in text and str(tmp_path) not in text


def test_summary_names_member_skips_by_their_typed_reason() -> None:
    """A run row keeps the exception type (``recheck_failed:OSError``); the summary copies only
    typed lower-case reasons, so the phase counts the member under ``recheck_failed``."""

    row = {"status": "applied", "candidate_count": 3, "candidate_bytes": 30, "offloaded_count": 1,
           "offloaded_bytes": 10, "skipped_by_reason": {
               "member_changed": {"count": 1, "bytes": 10}, "recheck_failed:OSError": {"count": 1, "bytes": 10}}}
    failed = {"status": "error", "run": "run-secret", "error_type": "PermissionError", "errno": 13, "stage": "residue"}

    phase = build_storage_gc_summary({"result_residue_offload": residue.residue_phase(
        [row, failed], enabled=True, applying=True)})["phases"]["result_residue_offload"]

    assert phase == {
        "status": "applied", "enabled": True, "removed_or_offloaded_bytes": 10,
        # A run that raised leaves the planned bytes unknown.
        "candidate_bytes": None,
        "retained_by_reason": {
            "member_skipped:member_changed": {"count": 1, "bytes": 10},
            "member_skipped:recheck_failed": {"count": 1, "bytes": 10},
            "residue_offload_failed": {"count": 1, "bytes": None},
        },
    }


def test_scene_attempt_recovery_ownership_records_stay(tmp_path) -> None:
    """Scene-attempt recovery scans every canary run for ``*.lease.json`` and
    ``pending_teardowns/*.json`` (or ``pending-teardowns``) and counts an open record as a
    blocker; evicting one would silently lift an ambiguous-create blocker."""

    f = _sealed_run(tmp_path / "canaries")
    ownership = {
        "allocator/paid-lane.lease.json": b'{"schema_version": "lease"}',
        "pending_teardowns/x.json": b'{"status": "open"}',
        "attempts/attempt_001/pending-teardowns/y.json": b'{"status": "open"}',
    }
    for relative, data in ownership.items():
        (f.run / relative).parent.mkdir(parents=True, exist_ok=True)
        (f.run / relative).write_bytes(data)
    # No recovery glob matches this one, so it is residue like any other file.
    (f.run / "pending_teardowns" / "notes.txt").write_bytes(b"not an ownership record")
    _age(f.run)

    plan = residue.offload_result_residue(run_root=f.run, hot_window_seconds=2 * DAY, now=lambda: NOW)

    reasons = {row["relative_path"]: row["reason"] for row in plan["skipped"]}
    assert {relative: reasons.get(relative) for relative in ownership} == dict.fromkeys(ownership, "reader_reopened")
    assert plan["candidate_count"] == len(RESIDUE) + 1
    assert _offload(f)["status"] == "applied"
    assert {relative: (f.run / relative).read_bytes() for relative in ownership} == ownership
    assert not (f.run / "pending_teardowns" / "notes.txt").exists()


def test_residue_mirrors_the_recovery_scan_globs() -> None:
    """The recovery module inlines its globs; if they change, the residue rules must follow."""

    import inspect

    from blueprint_pipeline import task_evaluation_scene_progression_recovery as recovery

    source = inspect.getsource(recovery.reconcile_ownership)
    assert 'rglob("*' + residue.OWNERSHIP_RECORD_SUFFIX + '")' in source
    for directory in residue.OWNERSHIP_RECORD_DIRECTORIES:
        assert f'glob("**/{directory}/*.json")' in source
    assert residue.OWNERSHIP_RECORD_DIRECTORIES == frozenset({"pending_teardowns", "pending-teardowns"})


def _dispatch_queue(root: Path) -> Path:
    for state in ("pending", "processing", "completed", "blocked"):
        (root / state).mkdir(parents=True, exist_ok=True)
    return root


@pytest.mark.parametrize("state", ["pending", "processing"])
def test_a_run_a_live_dispatch_row_names_stays_whole(tmp_path, state) -> None:
    """In queue mode the dispatcher runs a pending or processing envelope whether or not its run
    is sealed, and so reopens that run's authority, bundle and allocator records. A run such a
    row names stays whole until the row completes."""

    f = _sealed_run(tmp_path / "canaries")
    queue = _dispatch_queue(tmp_path / "dispatches")
    row = queue / state / "envelope-1.json"
    row.write_text(json.dumps({"activation_id": f.run.name}), encoding="utf-8")
    before = _local_files(f.run)

    kept = _offload(f, queue_roots=[queue])

    assert (kept["status"], kept["retained_reason"]) == ("retained", "dispatch_row_pending")
    assert _local_files(f.run) == before and not f.pointer.exists() and f.client.upload_count == 1
    os.replace(row, queue / "completed" / row.name)
    assert _offload(f, queue_roots=[queue])["status"] == "applied"


@pytest.mark.parametrize("damage", ["linked_row", "oversized_row", "linked_state", "fifo_row", "non_utf8_row"])
def test_a_dispatch_row_that_cannot_be_read_keeps_every_run(tmp_path, monkeypatch, damage) -> None:
    """A row that cannot be read might name any run, so none moves. The residue reads its queues
    through the storage GC's one strict reader."""

    from blueprint_pipeline import control_plane_storage_references as references

    f = _sealed_run(tmp_path / "canaries")
    queue = _dispatch_queue(tmp_path / "dispatches")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "row.json").write_text("{}", encoding="utf-8")
    if damage == "linked_row":
        (queue / "pending" / "linked.json").symlink_to(tmp_path / "elsewhere" / "row.json")
    elif damage == "oversized_row":
        monkeypatch.setattr(references, "MAX_QUEUE_MESSAGE_BYTES", 16)
        (queue / "processing" / "large.json").write_text(json.dumps({"activation_id": "another-run"}), encoding="utf-8")
    elif damage == "linked_state":
        (queue / "processing").rmdir()
        (queue / "processing").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    elif damage == "fifo_row":
        os.mkfifo(queue / "pending" / "row.json")
    else:
        (queue / "pending" / "row.json").write_bytes(b"\xff\xfe")
    reads: list[bool] = []
    real_read = references.queue_reference_text
    monkeypatch.setattr(references, "queue_reference_text",
                        lambda roots, *args, **kwargs: (reads.append(kwargs.get("strict")), real_read(
                            roots, *args, **kwargs))[1])
    before = _local_files(f.run)

    result = _offload(f, queue_roots=[queue])

    assert (result["status"], result["retained_reason"]) == ("retained", "dispatch_queue_unreadable")
    assert _local_files(f.run) == before and not f.pointer.exists()
    assert reads and set(reads) == {True}


def test_a_dispatch_row_written_during_publication_keeps_the_run(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries")
    queue = _dispatch_queue(tmp_path / "dispatches")

    def enqueuing(**kwargs):
        reference = f.publisher(**kwargs)
        (queue / "pending" / "late.json").write_text(json.dumps({"activation_id": f.run.name}), encoding="utf-8")
        return reference

    result = _offload(f, publisher=enqueuing, queue_roots=[queue])

    assert (result["status"], result["retained_reason"]) == ("retained", "dispatch_row_pending")
    assert not f.pointer.exists() and set(RESIDUE) <= set(_local_files(f.run))


def test_the_gc_reads_its_queues_for_rows_that_name_a_run(tmp_path) -> None:
    """The tick hands the residue its queue roots. A linked row is invisible to the queue
    protection, which skips what it cannot read; the residue's strict read keeps the run."""

    f = _sealed_run(tmp_path / "canaries")
    queue = _dispatch_queue(tmp_path / "dispatches")
    (tmp_path / "elsewhere.json").write_text("{}", encoding="utf-8")
    (queue / "pending" / "linked.json").symlink_to(tmp_path / "elsewhere.json")

    report = _tick(f, tmp_path / "pins", queue, apply=True, ack=RUN_ACK, offload_enabled=True,
                   result_residue_offload_enabled=True)

    assert [row["retained_reason"] for row in report["result_residue_offload"]["runs"]] == [
        "dispatch_queue_unreadable"]
    assert not f.pointer.exists() and set(RESIDUE) <= set(_local_files(f.run))


@pytest.mark.parametrize("field,value", [
    ("schema_version", "task_evaluation_policy_canary_dispatch.v0"),
    ("run_kind", "operator_policy_canary"),
    ("run_id", ""),
])
def test_a_dispatch_receipt_of_another_kind_keeps_the_run(tmp_path, field, value) -> None:
    """Only the canary dispatcher's own sealed receipt admits a run, checked as the terminal
    index checks it: schema, digest, run kind and run id."""

    f = _sealed_run(tmp_path / "canaries")
    path = f.run / "dispatch_receipt.json"
    receipt = {**json.loads(path.read_text(encoding="utf-8")), field: value}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    path.write_text(json.dumps(receipt), encoding="utf-8")
    _age(f.run)

    result = _offload(f)

    assert (result["status"], result["retained_reason"]) == ("retained", "dispatch_receipt_invalid")
    assert not f.pointer.exists() and set(RESIDUE) <= set(_local_files(f.run))


#: A tick time whose hour starts the cap's rotation at the first of three runs.
FIRST_OF_THREE = (int(NOW // 3600) // 3) * 3 * 3600


def _three_ready_runs(tmp_path, client=None) -> list[SimpleNamespace]:
    client = client or _ContentAddressedClient()
    return [_sealed_run(tmp_path / "canaries", f"run-{index}", client=client) for index in range(3)]


def test_a_tick_publishes_at_most_its_cap_and_defers_the_rest(tmp_path) -> None:
    """Like scene retirement, a tick attempts at most ``max_runs`` publications; the rest are
    planned and reported as ``deferred_tick_cap`` for a later tick."""

    runs = _three_ready_runs(tmp_path)
    pins, queue = tmp_path / "pins", tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    residue_bytes = sum(len(data) for data in RESIDUE.values())

    first = _tick(runs[0], pins, queue, apply=True, ack=RUN_ACK, offload_enabled=True,
                  result_residue_offload_enabled=True, result_residue_max_runs_per_tick=2, now=lambda: FIRST_OF_THREE)

    phase = first["result_residue_offload"]
    assert (phase["max_runs_per_tick"], phase["attempted_count"]) == (2, 2)
    assert [(row["run"], row["status"], row["retained_reason"]) for row in phase["runs"]] == [
        ("run-0", "applied", None), ("run-1", "applied", None), ("run-2", "retained", "deferred_tick_cap")]
    assert phase["retained_by_reason"]["deferred_tick_cap"] == {"count": 1, "bytes": residue_bytes}
    assert (phase["candidate_bytes"], phase["offloaded_bytes"]) == (3 * residue_bytes, 2 * residue_bytes)
    assert set(RESIDUE) <= set(_local_files(runs[2].run)) and not runs[2].pointer.exists()

    second = _tick(runs[0], pins, queue, apply=True, ack=RUN_ACK, offload_enabled=True,
                   result_residue_offload_enabled=True, result_residue_max_runs_per_tick=2, now=lambda: FIRST_OF_THREE)
    assert [(row["run"], row["retained_reason"]) for row in second["result_residue_offload"]["runs"]] == [
        ("run-0", "already_offloaded"), ("run-1", "already_offloaded"), ("run-2", None)]
    assert runs[2].pointer.is_file()


def test_a_failing_publisher_costs_at_most_the_cap_per_tick(tmp_path) -> None:
    """A failed publication counts against the cap and is not retried in the same tick."""

    client = _ContentAddressedClient()
    runs = _three_ready_runs(tmp_path, client)
    calls: list[str] = []

    def failing(**kwargs):
        calls.append(Path(kwargs["path"]).name)
        raise store.TaskEvaluationConfiguredSceneObjectStoreError("configured_scene_artifact_publication_failed")

    pins, queue = tmp_path / "pins", tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    report = _tick(runs[0], pins, queue, apply=True, ack=RUN_ACK, offload_enabled=True,
                   result_residue_offload_enabled=True, result_residue_max_runs_per_tick=2, publisher=failing,
                   now=lambda: FIRST_OF_THREE)

    phase = report["result_residue_offload"]
    assert len(calls) == 2 and phase["attempted_count"] == 2
    assert [row["retained_reason"] for row in phase["runs"]] == [
        "publication_failed", "publication_failed", "deferred_tick_cap"]
    assert [row["failure"]["stage"] for row in phase["runs"][:2]] == ["publish", "publish"]
    assert all(not run.pointer.exists() and set(RESIDUE) <= set(_local_files(run.run)) for run in runs)


@pytest.mark.parametrize("raw,expected", [("", 5), ("0", 0), ("12", 12)])
def test_the_residue_cap_reads_from_the_unit_environment(tmp_path, monkeypatch, capsys, raw, expected) -> None:
    f = _sealed_run(tmp_path / "canaries")
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    _unit_environment(monkeypatch, {gc_module.QUEUE_ROOTS_ENV: str(queue), gc_module.EVIDENCE_ROOTS_ENV: str(f.evidence),
                                     gc_module.EVIDENCE_OFFLOAD_ENV: "1", residue.RESIDUE_OFFLOAD_ENV: "1",
                                     residue.RESIDUE_MAX_RUNS_ENV: raw})
    monkeypatch.setattr(gc_module, "require_storage_class", lambda *a, **k: None)
    monkeypatch.setattr(artifacts, "_artifact_object_store_client", lambda: (f.client, BUCKET))
    # The unit's stream publisher, against the fixture's fake store.
    monkeypatch.setattr(evidence, "publish_configured_scene_stream", lambda **kwargs: (_ for _ in ()).throw(
        store.TaskEvaluationConfiguredSceneObjectStoreError("configured_scene_artifact_publication_failed")))

    assert gc_module.main(["run", "--apply", "--ack", RUN_ACK, "--pins-root", str(tmp_path / "pins")]) == 0

    phase = json.loads(capsys.readouterr().out)["result_residue_offload"]
    assert phase["max_runs_per_tick"] == expected
    assert [row["retained_reason"] for row in phase["runs"]] == [
        "deferred_tick_cap" if expected == 0 else "publication_failed"]


def test_the_residue_bytes_count_once_across_phases() -> None:
    """What the residue keeps lies inside evidence offload's ``result_registry`` bytes, so the
    residue phase shows its own breakdown but never adds to the totals across phases."""

    report = {
        "status": "applied", "observed_at_epoch": NOW,
        "evidence_offload": {"status": "applied", "candidate_bytes": 0, "offloaded_bytes": 0,
                             "retained_by_reason": {"result_registry": {"count": 2, "bytes": 1000}}},
        "result_residue_offload": residue.residue_phase([
            {"status": "applied", "candidate_count": 1, "candidate_bytes": 100, "offloaded_count": 1,
             "offloaded_bytes": 100, "skipped_by_reason": {"reader_reopened": {"count": 3, "bytes": 600}}},
            {"status": "retained", "retained_reason": "deferred_tick_cap", "candidate_count": 1,
             "candidate_bytes": 200},
        ], enabled=True, applying=True),
    }

    summary = build_storage_gc_summary(report)

    assert summary["phases"]["result_residue_offload"]["retained_by_reason"] == {
        "member_skipped:reader_reopened": {"count": 3, "bytes": 600},
        "deferred_tick_cap": {"count": 1, "bytes": 200},
    }
    assert summary["top_retained_reasons"] == [{"reason": "result_registry", "bytes": 1000}]
    assert summary["top_retained"] == [
        {"phase": "evidence_offload", "reason": "result_registry", "count": 2, "bytes": 1000}]


def test_a_member_whose_directory_vanished_is_neither_offloaded_nor_kept(tmp_path) -> None:
    """A name whose directory disappeared before its unlink was not removed by the offload: it is
    ``member_vanished``, not offloaded, and not kept, so restore still brings its bytes back."""

    f = _sealed_run(tmp_path / "canaries")

    def moving(**kwargs):
        reference = f.publisher(**kwargs)
        (f.run / "work" / "stage").rename(tmp_path / "moved-stage")
        return reference

    result = _offload(f, publisher=moving)

    assert result["status"] == "applied"
    assert _changed(result["skipped"]) == [{"relative_path": "work/stage/state.npz", "reason": "member_vanished"}]
    assert result["offloaded_count"] == len(RESIDUE) - 1
    assert result["offloaded_bytes"] == sum(len(data) for data in RESIDUE.values()) - len(RESIDUE["work/stage/state.npz"])
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    assert pointer["kept"] == []
    assert (tmp_path / "moved-stage" / "state.npz").read_bytes() == RESIDUE["work/stage/state.npz"]

    restored = residue.restore_result_residue(run_root=f.run, now=lambda: NOW, materializer=functools.partial(
        store.materialize_configured_scene_artifact, client=f.client, bucket=BUCKET))

    assert restored["restored_count"] == len(RESIDUE)
    assert (f.run / "work" / "stage" / "state.npz").read_bytes() == RESIDUE["work/stage/state.npz"]


def test_a_run_whose_only_change_is_a_vanished_member_keeps_its_pointer(tmp_path, monkeypatch) -> None:
    """The archive is the only record of a vanished member's bytes, so its pointer stays even
    when nothing else could be evicted."""

    f = _sealed_run(tmp_path / "canaries", residue_files={"work/stage/state.npz": RESIDUE["work/stage/state.npz"]})

    def moving(**kwargs):
        reference = f.publisher(**kwargs)
        (f.run / "work" / "stage").rename(tmp_path / "moved-stage")
        return reference

    result = _offload(f, publisher=moving)

    assert (result["status"], result["offloaded_count"]) == ("applied", 0)
    assert f.pointer.is_file()


def _crash_on_second_group(monkeypatch, f) -> None:
    """Offload ``f`` with a crash while its second member group is being evicted."""

    real_remove, calls = residue.held_files._remove_group, []

    def crashing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("killed mid-eviction")
        return real_remove(*args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(residue.held_files, "_remove_group", crashing)
        with pytest.raises(RuntimeError, match="killed mid-eviction"):
            _offload(f)


def test_an_eviction_a_crash_cut_short_resumes_on_the_next_tick(tmp_path, monkeypatch) -> None:
    """A crash after the pointer is written leaves members behind it. Each later tick reports
    the residue bytes still local, and one that applies and passes every gate under the lock
    evicts the listed members that still match the pointer, then rewrites what it keeps."""

    f = _sealed_run(tmp_path / "canaries")
    _crash_on_second_group(monkeypatch, f)
    left = sorted(relative for relative in RESIDUE if (f.run / relative).exists())
    assert f.pointer.is_file() and len(left) == len(RESIDUE) - 1
    assert json.loads(f.pointer.read_text(encoding="utf-8"))["state"] == "evicting"

    plan = residue.offload_result_residue(run_root=f.run, hot_window_seconds=2 * DAY, now=lambda: NOW)
    left_bytes = sum(len(RESIDUE[relative]) for relative in left)
    assert (plan["status"], plan["resume"]) == ("dry_run", True)
    assert (plan["candidate_count"], plan["candidate_bytes"]) == (len(left), left_bytes)
    assert (plan["pointed_remaining_count"], plan["pointed_remaining_bytes"]) == (len(left), left_bytes)

    resumed = _offload(f)

    assert (resumed["status"], resumed["resume"], resumed["offloaded_count"]) == ("applied", True, len(left))
    assert resumed["offloaded_bytes"] == left_bytes and resumed["pointed_remaining_bytes"] == 0
    assert not set(RESIDUE) & set(_local_files(f.run))
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    assert pointer["kept"] == [] and pointer["pointer_digest"] == canonical_digest(pointer, digest_field="pointer_digest")
    assert pointer["state"] == "offloaded"
    assert f.client.upload_count == 2  # the resume published nothing
    again = _offload(f)
    assert (again["status"], again["retained_reason"], again["pointed_remaining_bytes"]) == (
        "retained", "already_offloaded", 0)


def test_a_resumed_member_that_no_longer_matches_the_pointer_is_kept(tmp_path, monkeypatch) -> None:
    f = _sealed_run(tmp_path / "canaries")
    _crash_on_second_group(monkeypatch, f)
    left = sorted(relative for relative in RESIDUE if (f.run / relative).exists())
    changed = f.run / left[0]
    changed.write_bytes(b"written after the crash")
    # Its old mtime keeps it residue for the resume's fresh plan: only its bytes tell it changed.
    os.utime(changed, (OLD, OLD))

    resumed = _offload(f)

    assert (resumed["status"], resumed["offloaded_count"]) == ("applied", len(left) - 1)
    assert _changed(resumed["skipped"]) == [{"relative_path": left[0], "reason": "member_changed"}]
    assert changed.read_bytes() == b"written after the crash"
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    assert pointer["kept"] == [{"relative_path": left[0], "reason": "member_changed"}]
    again = _offload(f)
    assert (again["retained_reason"], again["pointed_remaining_count"], again["pointed_remaining_bytes"]) == (
        "already_offloaded", 1, len(b"written after the crash"))


def test_a_pointer_that_does_not_verify_is_left_alone(tmp_path, monkeypatch) -> None:
    f = _sealed_run(tmp_path / "canaries")
    _crash_on_second_group(monkeypatch, f)
    value = json.loads(f.pointer.read_text(encoding="utf-8"))
    value["members"][0]["size_bytes"] += 1
    f.pointer.chmod(0o640)
    f.pointer.write_text(json.dumps(value), encoding="utf-8")
    before = _local_files(f.run)

    result = _offload(f)

    assert (result["status"], result["retained_reason"]) == ("retained", "pointer_invalid")
    assert _local_files(f.run) == before


@pytest.mark.parametrize("case", [
    "relative_to_another_directory", "name_unsupported_document", "newer_document", "linked_outside_document",
])
def test_a_file_that_stays_is_searched_whatever_kept_it(tmp_path, case) -> None:
    """Every file that stays is searched, whatever kept it, and a relative path is matched as a
    tail of any file of the run, since it may be written relative to a directory the search
    cannot guess."""

    f = _sealed_run(tmp_path / "canaries")
    run, target = f.run, "logs/worker.log"
    if case == "relative_to_another_directory":
        (run / "preprovider_evidence" / "logs").mkdir(parents=True)
        (run / "preprovider_evidence" / "logs" / "x.log").write_bytes(b"preprovider output")
        (run / TERMINAL_RESULT).write_text(json.dumps({"log": "logs/x.log"}), encoding="utf-8")
        target = "preprovider_evidence/logs/x.log"
    elif case == "name_unsupported_document":
        (run / "notes with spaces.txt").write_text("see logs/worker.log\n", encoding="utf-8")
    elif case == "newer_document":
        (run / "late.txt").write_text("see logs/worker.log\n", encoding="utf-8")
    else:
        (run / "work" / "linked.txt").write_text("see logs/worker.log\n", encoding="utf-8")
        os.link(run / "work" / "linked.txt", tmp_path / "outside-link.txt")
    _age(run)
    if case == "newer_document":
        os.utime(run / "late.txt", (OLD + DAY, OLD + DAY))

    plan = residue.offload_result_residue(run_root=run, hot_window_seconds=2 * DAY, now=lambda: NOW)

    reasons = {row["relative_path"]: row["reason"] for row in plan["skipped"]}
    assert reasons.get(target) == "receipt_referenced"
    assert _offload(f)["status"] == "applied"
    assert (run / target).exists()


def test_a_directory_on_another_filesystem_keeps_the_whole_run(tmp_path, monkeypatch) -> None:
    """A mounted directory is never entered, so whatever in it names a file is unknown."""

    f = _sealed_run(tmp_path / "canaries")
    mounted = str(f.run / "work" / "stage")
    real_lstat = os.lstat

    def mounting(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        if os.fspath(path) == mounted:
            return os.stat_result((info.st_mode, info.st_ino, info.st_dev + 1, *tuple(info)[3:10]))
        return info

    monkeypatch.setattr(os, "lstat", mounting)
    result = _offload(f)

    assert (result["status"], result["retained_reason"]) == ("retained", "plan_failed")
    assert result["failure"]["error_type"] == "ResultResidueOffloadError"
    assert not f.pointer.exists()


def test_a_kept_document_that_cannot_be_read_keeps_the_whole_run(tmp_path) -> None:
    if os.geteuid() == 0:
        pytest.skip("root reads a file whatever its mode")
    f = _sealed_run(tmp_path / "canaries")
    (f.run / TERMINAL_RESULT).chmod(0)
    try:
        result = _offload(f)
    finally:
        (f.run / TERMINAL_RESULT).chmod(0o644)

    assert (result["status"], result["retained_reason"]) == ("retained", "plan_failed")
    assert result["failure"] == {"error_type": "ResultResidueOffloadError", "errno": None, "stage": "plan"}
    assert not f.pointer.exists() and set(RESIDUE) <= set(_local_files(f.run))


def test_the_capped_runs_rotate_from_tick_to_tick(tmp_path) -> None:
    """Each hour's tick starts the cap at another run, so runs that keep failing cannot starve
    the ones after them."""

    runs = _three_ready_runs(tmp_path)
    attempted: list[str] = []

    def failing(**kwargs):
        attempted.append(Path(kwargs["path"]).name.split(".")[1])
        raise store.TaskEvaluationConfiguredSceneObjectStoreError("configured_scene_artifact_publication_failed")

    pins, queue = tmp_path / "pins", tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    for hour in range(3):
        report = _tick(runs[0], pins, queue, apply=True, ack=RUN_ACK, offload_enabled=True,
                       result_residue_offload_enabled=True, result_residue_max_runs_per_tick=1, publisher=failing,
                       now=lambda hour=hour: FIRST_OF_THREE + hour * 3600)
        # The rows keep the runs' order, whichever ran first.
        assert [row["run"] for row in report["result_residue_offload"]["runs"]] == ["run-0", "run-1", "run-2"]

    assert attempted == ["run-0", "run-1", "run-2"]


def test_a_dry_run_tick_reads_the_queues_once_and_trusts_the_bulk_result(tmp_path, monkeypatch) -> None:
    """While the phase only plans, a tick reads its queues once for every run and takes each run's
    bulk result from the per-artifact offload it already ran, instead of running it again."""

    runs = _three_ready_runs(tmp_path)
    pins, queue = tmp_path / "pins", tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    reads, bulk_checks = [], []
    real_snapshot, real_bulk = residue.queue_snapshot, residue.offload_result_artifacts
    monkeypatch.setattr(residue, "queue_snapshot", lambda roots: (reads.append(1), real_snapshot(roots))[1])
    monkeypatch.setattr(residue, "offload_result_artifacts", lambda **kw: (bulk_checks.append(1), real_bulk(**kw))[1])

    report = _tick(runs[0], pins, queue, apply=False)

    assert [row["status"] for row in report["result_residue_offload"]["runs"]] == ["dry_run"] * 3
    assert (len(reads), len(bulk_checks)) == (1, 0)


def test_a_run_added_twice_in_a_tick_is_planned_once(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries")
    tick = residue.ResidueTick(applying=False, enabled=False, hot_window_seconds=2 * DAY, protection_checker=None,
                               publisher=None, now=lambda: NOW, queue_roots=())
    bulk = {"status": "dry_run", "candidate_count": 0, "skipped": []}

    tick.add(f.run, bulk)
    tick.add(f.evidence / ".." / f.evidence.name / f.run.name, bulk)

    assert [row["run"] for row in tick.phase()["runs"]] == [f.run.name]


def test_both_offloads_publish_through_one_helper(tmp_path, monkeypatch) -> None:
    """Whole-run evidence offload and the residue pack, reserve, publish and check the readback
    through one helper, so neither can drift from the other."""

    f = _sealed_run(tmp_path / "canaries")
    plain = tmp_path / "launches" / "run-9"
    plain.mkdir(parents=True)
    (plain / "launch_receipt.json").write_text("{}", encoding="utf-8")
    (plain / "stage.log").write_text("a line\n", encoding="utf-8")
    _age(plain)
    published: list[str] = []
    real = evidence.publish_archive

    def recording(directory, **kwargs):
        published.append(kwargs["filename"])
        return real(directory, **kwargs)

    monkeypatch.setattr(evidence, "publish_archive", recording)
    assert _offload(f)["status"] == "applied"
    manifest = evidence.build_evidence_offload_manifest(
        evidence_roots=[plain.parent], hot_window_seconds=0, now=lambda: NOW, classifier=lambda *a, **k: None)
    applied = evidence.apply_evidence_offload(manifest, ack=evidence.EXECUTE_ACK, publisher=f.publisher,
                                              now=lambda: NOW)

    assert applied["offloaded_count"] == 1 and not plain.exists()
    assert published == ["residue.tar", "evidence.tar"]


def test_the_names_a_cut_short_group_lost_are_the_ones_it_unlinked(tmp_path, monkeypatch) -> None:
    """``_remove_group`` unlinks a store-copy name last, whatever order the names sort in, so the
    names a failed group removal took are the ones it reports unlinking, not a prefix of them."""

    f = _sealed_run(tmp_path / "canaries")
    digest = hashlib.sha256(b"a store-shaped name").hexdigest()
    store_name = f.run / "prepared-references" / "content-addressed" / "sha256" / digest
    store_name.parent.mkdir(parents=True)
    os.link(f.run / "logs" / "worker.log", store_name)
    os.link(f.run / "logs" / "worker.log", f.run / "zzz.log")
    _age(f.run)
    real_unlink = os.unlink

    def sticky(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None and os.fspath(path) == digest:
            raise PermissionError(1, "Operation not permitted")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(os, "unlink", sticky)
        result = _offload(f)

    kept = store_name.relative_to(f.run).as_posix()
    assert _changed(result["skipped"]) == [{"relative_path": kept, "reason": "unlink_failed:PermissionError"}]
    assert "member_vanished" not in result["skipped_by_reason"]
    # Every name went but the store-shaped one: the three groups' other names.
    assert result["offloaded_count"] == len(RESIDUE) + 1
    assert not (f.run / "logs" / "worker.log").exists() and not (f.run / "zzz.log").exists()
    assert json.loads(f.pointer.read_text(encoding="utf-8"))["kept"] == [
        {"relative_path": kept, "reason": "unlink_failed:PermissionError"}]


def test_a_run_whose_registry_does_not_seal_is_registry_unsealed(tmp_path) -> None:
    """A G1 review has a registry but no delivery, so its per-artifact offload refuses it while
    reading the registry. Its residue row says so (``registry_unsealed``, with that failure's type
    and stage) instead of ``bulk_offload_failed``, which stays for a bulk offload that failed later."""

    f = _sealed_run(tmp_path / "canaries")
    (f.run / "artifacts" / "result_delivery" / "delivery.json").unlink()
    pins, queue = tmp_path / "pins", tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)

    report = _tick(f, pins, queue, apply=False)

    [row] = report["result_residue_offload"]["runs"]
    assert (row["status"], row["retained_reason"]) == ("retained", "registry_unsealed")
    assert row["failure"] == {"error_type": "TaskEvaluationResultDeliveryError", "errno": None, "stage": "registry"}
    assert build_storage_gc_summary(report)["phases"]["result_residue_offload"]["retained_by_reason"] == {
        "registry_unsealed": {"count": 1, "bytes": None}}
    failed = {"status": "retained", "reason": "OSError", "error_type": "OSError", "errno": 5, "stage": "publish"}
    later = residue.residue_row(f.run, failed, apply=False, hot_window_seconds=2 * DAY, protection_checker=None,
                                publisher=None, now=lambda: NOW)
    assert (later["retained_reason"], later["failure"]) == (
        "bulk_offload_failed", {"error_type": "OSError", "errno": 5, "stage": "publish"})


@pytest.mark.parametrize("damage", ["source_changed", "remote_reference_invalid"])
def test_a_bulk_offload_that_failed_past_the_registry_is_not_registry_unsealed(tmp_path, damage) -> None:
    """Code review of 10d: the artifact store's own checks after the seal (a registered file with
    other bytes, a remote reference that does not verify, an alias conflict, a path outside the
    run) also failed at the default ``registry`` stage, so their runs read ``registry_unsealed``.
    They now fail at ``plan``: the run is ``bulk_offload_failed``, with the failure's type."""

    f = _sealed_run(tmp_path / "canaries", bulk_remote=damage != "source_changed")
    if damage == "source_changed":
        (f.run / "evidence" / "review.mp4").write_bytes(b"w" * len(REGISTERED["evidence/review.mp4"][1]))
    else:
        [reference] = (f.run / "artifacts" / "result_delivery" / artifacts.REMOTE_DIRECTORY).glob("*.json")
        value = json.loads(reference.read_text(encoding="utf-8"))
        value["run_id"] = "another-run"
        reference.chmod(0o640)
        reference.write_text(json.dumps(value), encoding="utf-8")
    pins, queue = tmp_path / "pins", tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)

    report = _tick(f, pins, queue, apply=False)

    [bulk] = report["result_artifact_offload"]
    assert (bulk["status"], bulk["stage"]) == ("retained", "plan")
    [row] = report["result_residue_offload"]["runs"]
    assert (row["status"], row["retained_reason"]) == ("retained", "bulk_offload_failed")
    assert row["failure"] == {"error_type": "TaskEvaluationResultDeliveryError", "errno": None, "stage": "plan"}


def test_a_linked_run_root_is_run_root_invalid_whatever_its_bulk_offload_says(tmp_path) -> None:
    """The artifact store refuses a linked run root before it reads the registry, at its default
    ``registry`` stage; the residue names it for what it is."""

    f = _sealed_run(tmp_path / "canaries")
    alias = f.evidence / "alias"
    alias.symlink_to(f.run)
    refused = {"status": "retained", "reason": "TaskEvaluationResultDeliveryError",
               "error_type": "TaskEvaluationResultDeliveryError", "errno": None, "stage": "registry"}

    row = residue.residue_row(alias, refused, apply=False, hot_window_seconds=2 * DAY, protection_checker=None,
                              publisher=None, now=lambda: NOW)

    assert (row["status"], row["retained_reason"]) == ("retained", "run_root_invalid")


def test_a_member_whose_name_the_search_cannot_read_stays(tmp_path) -> None:
    """The search reads a path only as a run of name characters, so it cannot prove that no kept
    document names a file whose name holds any other character: such a file stays
    (``name_unsupported``), and is searched like everything that stays."""

    f = _sealed_run(tmp_path / "canaries")
    unsupported = {"logs/worker log.txt": b"a spaced name\n", "logs/r\u00e9sum\u00e9.log": b"an accented name\n",
                   "work/semi;colon.txt": b"see work/stage/state.npz\n"}
    for relative, data in unsupported.items():
        (f.run / relative).write_bytes(data)
    _age(f.run)

    plan = residue.offload_result_residue(run_root=f.run, hot_window_seconds=2 * DAY, now=lambda: NOW)

    reasons = {row["relative_path"]: row["reason"] for row in plan["skipped"]}
    assert {relative: reasons.get(relative) for relative in unsupported} == dict.fromkeys(unsupported, "name_unsupported")
    assert plan["skipped_by_reason"]["name_unsupported"] == {
        "count": len(unsupported), "bytes": sum(len(data) for data in unsupported.values())}
    assert reasons.get("work/stage/state.npz") == "receipt_referenced"
    assert not any(scan.name_supported(relative) for relative in unsupported)
    assert _offload(f)["status"] == "applied"
    assert {relative: (f.run / relative).read_bytes() for relative in unsupported} == unsupported
    assert (f.run / "work/stage/state.npz").read_bytes() == RESIDUE["work/stage/state.npz"]


def _restore(f) -> dict:
    return residue.restore_result_residue(run_root=f.run, now=lambda: NOW, materializer=functools.partial(
        store.materialize_configured_scene_artifact, client=f.client, bucket=BUCKET))


def test_a_restored_run_is_never_offloaded_again_by_a_tick(tmp_path) -> None:
    """Code review of 10d: after a restore every member is local, not kept and matches the pointer,
    which is exactly what an eviction cut short looks like, so the next applying tick resumed and
    evicted all of them again. The pointer now says which it is: only an ``evicting`` pointer is
    resumed, and a restore records ``restored``, which no tick offloads again."""

    f = _sealed_run(tmp_path / "canaries")
    assert _offload(f)["status"] == "applied"
    assert json.loads(f.pointer.read_text(encoding="utf-8"))["state"] == "offloaded"

    assert _restore(f)["restored_count"] == len(RESIDUE)
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    assert pointer["state"] == "restored"
    assert pointer["pointer_digest"] == canonical_digest(pointer, digest_field="pointer_digest")

    planned = residue.offload_result_residue(run_root=f.run, hot_window_seconds=2 * DAY, now=lambda: NOW)
    applied = _offload(f)
    pins, queue = tmp_path / "pins", tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    ticked = _tick(f, pins, queue, apply=True, ack=RUN_ACK, offload_enabled=True,
                   result_residue_offload_enabled=True)["result_residue_offload"]

    for row in (planned, applied, *ticked["runs"]):
        assert (row["status"], row["retained_reason"], row.get("resume")) == ("retained", "restored", None)
    assert ticked["retained_by_reason"] == {"restored": {"count": 1, "bytes": None}}
    assert {relative: (f.run / relative).read_bytes() for relative in RESIDUE} == RESIDUE
    assert f.client.upload_count == 2  # the fixture's bulk artifact and the one residue archive


def test_a_pointer_without_a_state_is_never_resumed(tmp_path, monkeypatch) -> None:
    """A pointer that does not say its eviction is running is read as offloaded, so nothing
    resumes behind it: what is local stays local."""

    f = _sealed_run(tmp_path / "canaries")
    _crash_on_second_group(monkeypatch, f)
    value = json.loads(f.pointer.read_text(encoding="utf-8"))
    del value["state"]
    value["pointer_digest"] = canonical_digest(value, digest_field="pointer_digest")
    f.pointer.chmod(0o640)
    f.pointer.write_text(json.dumps(value), encoding="utf-8")
    before = _local_files(f.run)

    result = _offload(f)

    assert (result["status"], result["retained_reason"], result.get("resume")) == (
        "retained", "already_offloaded", None)
    assert _local_files(f.run) == before


def test_a_restore_and_a_resume_never_interleave(tmp_path, monkeypatch) -> None:
    """Restore holds the run's offload lock for its whole pass, so no tick resumes an eviction while
    members come back, and a restore refuses to start while a tick holds the lock."""

    import fcntl

    f = _sealed_run(tmp_path / "canaries")
    _crash_on_second_group(monkeypatch, f)
    before = _local_files(f.run)
    holder = (f.run / "artifacts/result_delivery/.offload.lock").open("a+b")
    try:
        fcntl.flock(holder, fcntl.LOCK_EX)
        with pytest.raises(residue.ResultResidueOffloadError, match="restore_locked"):
            _restore(f)
    finally:
        holder.close()
    assert _local_files(f.run) == before
    assert not (f.evidence / f"{f.run.name}{residue.RESTORE_RECEIPT_SUFFIX}").exists()

    materialize = functools.partial(store.materialize_configured_scene_artifact, client=f.client, bucket=BUCKET)
    during: list[dict] = []

    def materializing(**kwargs):
        during.append(_offload(f))  # a tick while the restore runs
        return materialize(**kwargs)

    restored = residue.restore_result_residue(run_root=f.run, now=lambda: NOW, materializer=materializing)

    assert [(row["retained_reason"], row.get("resume")) for row in during] == [("offload_locked", True)]
    assert restored["status"] == "restored"
    assert _offload(f)["retained_reason"] == "restored"
    assert {relative: (f.run / relative).read_bytes() for relative in RESIDUE} == RESIDUE


@pytest.mark.parametrize("damage", ["missing", "digest_differs", "size_differs"])
def test_a_resume_evicts_nothing_behind_an_archive_it_cannot_see(tmp_path, monkeypatch, damage) -> None:
    """Before a resume evicts, a HEAD request (no bytes read) must find the pointer's archive with its
    size and digest; otherwise every member stays (``archive_unverified``) and the pointer still says
    ``evicting``, so a later tick tries again."""

    f = _sealed_run(tmp_path / "canaries")
    _crash_on_second_group(monkeypatch, f)
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    key = (BUCKET, pointer["archive"]["uri"].split(f"s3://{BUCKET}/", 1)[1])
    stored, metadata = f.client.objects[key], dict(f.client.metadata[key])
    if damage == "missing":
        del f.client.objects[key]
    elif damage == "digest_differs":
        f.client.metadata[key]["sha256"] = "0" * 64
    else:
        f.client.objects[key] = stored + b"x"
    before = _local_files(f.run)

    kept = _offload(f)

    assert (kept["status"], kept["retained_reason"], kept["resume"]) == ("retained", "archive_unverified", True)
    assert kept["failure"] == {"error_type": "TaskEvaluationConfiguredSceneObjectStoreError", "errno": None,
                               "stage": "verify"}
    assert _local_files(f.run) == before
    assert json.loads(f.pointer.read_text(encoding="utf-8"))["state"] == "evicting"
    f.client.objects[key], f.client.metadata[key] = stored, metadata
    resumed = _offload(f)
    assert (resumed["status"], resumed["offloaded_count"]) == ("applied", len(RESIDUE) - 1)


def test_a_resume_keeps_a_member_a_reader_can_now_reach(tmp_path, monkeypatch) -> None:
    """A resume plans the run again and evicts only members that are still residue: a kept
    document may have come to name one since the original plan."""

    f = _sealed_run(tmp_path / "canaries")
    _crash_on_second_group(monkeypatch, f)
    left = sorted(relative for relative in RESIDUE if (f.run / relative).exists())
    named, other = left[0], left[1]
    (f.run / "launch_receipt.json").write_text(json.dumps({"log": named}), encoding="utf-8")

    plan = residue.offload_result_residue(run_root=f.run, hot_window_seconds=2 * DAY, now=lambda: NOW)
    resumed = _offload(f)

    for row in (plan, resumed):
        assert (row["candidate_count"], row["candidate_bytes"]) == (1, len(RESIDUE[other]))
        assert {skip["relative_path"]: skip["reason"] for skip in row["skipped"]}[named] == "receipt_referenced"
    assert (resumed["status"], resumed["offloaded_count"]) == ("applied", 1)
    assert (f.run / named).read_bytes() == RESIDUE[named] and not (f.run / other).exists()
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    assert pointer["state"] == "offloaded"
    assert pointer["kept"] == [{"relative_path": named, "reason": "no_longer_residue"}]


def test_a_restore_that_lands_before_the_tick_takes_the_lock_is_not_undone(tmp_path, monkeypatch) -> None:
    """Code review of 10d: a tick read the pointer before it took the run lock. A restore that
    finished in between left it acting on its stale ``evicting`` copy: it evicted every restored
    member and rewrote the pointer ``offloaded``. Under the lock the tick reads the pointer again,
    and goes on only while it is still the ``evicting`` pointer it saw."""

    f = _sealed_run(tmp_path / "canaries")
    _crash_on_second_group(monkeypatch, f)
    real_snapshot, restored = residue.queue_snapshot, []

    def restore_first(roots):
        if not restored:
            restored.append(_restore(f))  # lands inside the tick's gates, before its lock
        return real_snapshot(roots)

    monkeypatch.setattr(residue, "queue_snapshot", restore_first)
    result = _offload(f)

    assert [receipt["status"] for receipt in restored] == ["restored"]
    assert (result["status"], result["retained_reason"], result["offloaded_count"]) == ("retained", "restored", 0)
    assert {relative: (f.run / relative).read_bytes() for relative in RESIDUE} == RESIDUE
    assert json.loads(f.pointer.read_text(encoding="utf-8"))["state"] == "restored"


def test_a_pointer_written_before_the_tick_takes_the_lock_stops_a_new_offload(tmp_path, monkeypatch) -> None:
    """The same holds for a run with no pointer at first: an offload that lands before the lock
    leaves a pointer the tick then reads, and the tick publishes and writes nothing."""

    f = _sealed_run(tmp_path / "canaries")
    real_snapshot, landed = residue.queue_snapshot, []

    def offload_first(roots):
        if not landed:
            landed.append(True)
            landed.append(_offload(f))  # another tick's offload, before this one's lock
        return real_snapshot(roots)

    monkeypatch.setattr(residue, "queue_snapshot", offload_first)
    result = _offload(f)

    assert landed[1]["status"] == "applied"
    assert (result["status"], result["retained_reason"], result["offloaded_count"]) == (
        "retained", "already_offloaded", 0)
    assert f.client.upload_count == 2  # the fixture's bulk artifact and the one residue archive
