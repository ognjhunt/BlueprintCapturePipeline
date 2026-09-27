# Covers (for impacted-test selection):
#   src/blueprint_pipeline/task_evaluation_result_residue_offload.py
#   src/blueprint_pipeline/control_plane_storage_gc.py
#   src/blueprint_pipeline/control_plane_storage_gc_reasons.py
#   src/blueprint_pipeline/control_plane_evidence_offload.py
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
    options = {"hot_window_seconds": 2 * DAY, "publisher": f.publisher, "now": lambda: NOW, **kwargs}
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


def test_a_large_reached_text_is_searched_as_a_stream(tmp_path) -> None:
    """A kept or reached text is searched a chunk at a time, whatever its size: a 70 MiB log
    naming a file keeps that file, even where the name straddles two chunks, and the run plans
    instead of failing. Memory stays at a chunk and a carried token."""

    f = _sealed_run(tmp_path / "canaries")
    interpretation = f.run / "episode_interpretation"
    interpretation.mkdir()
    chunk, reference, line = residue._SCAN_CHUNK_BYTES, b"logs/worker.log", b"stage 0000001 ok\n"
    boundary = 40 * chunk
    with (interpretation / "rollout.log").open("wb") as stream:
        lines = (boundary - 7) // len(line)
        stream.write(line * lines)
        stream.write(b" " * (boundary - 7 - lines * len(line)))
        stream.write(reference + b"\n")  # seven bytes before the 41st chunk begins, eight after
        stream.write(line * ((70 * 1024 * 1024 - stream.tell()) // len(line) + 1))
    assert (interpretation / "rollout.log").stat().st_size > 70 * 1024 * 1024
    _age(f.run)

    plan = residue.offload_result_residue(run_root=f.run, hot_window_seconds=2 * DAY, now=lambda: NOW)

    assert plan["status"] == "dry_run"
    reasons = {row["relative_path"]: row["reason"] for row in plan["skipped"]}
    assert reasons["logs/worker.log"] == "receipt_referenced"
    assert plan["candidate_count"] == len(RESIDUE) - 1


def test_the_stream_search_carries_a_token_and_reads_escaped_slashes() -> None:
    """A token cut by a chunk boundary is read whole, ``\\/`` reads as ``/``, and a token longer
    than any path keeps only its tail, so memory stays bounded."""

    import io

    def tokens(raw: bytes, chunk: int = 8) -> set[str]:
        found: set[str] = set()
        for part in residue._stream_tokens(io.BytesIO(raw), chunk_bytes=chunk):
            found |= part
        return found

    assert "logs/worker.log" in tokens(b'{"log": "logs/worker.log"}')
    assert "logs/worker.log" in tokens(b'{"log": "logs\\/worker.log"}', chunk=11)
    assert "logs/worker.log" in tokens(b"a logs/worker.log b", chunk=5)
    long = tokens(b"x" * (3 * residue._MAX_TOKEN_CHARS) + b"/run-1/logs/worker.log", chunk=4096)
    assert any(token.endswith("/run-1/logs/worker.log") for token in long)
    assert all(len(token) <= residue._MAX_TOKEN_CHARS for token in long)
    # A binary names nothing.
    assert tokens(b"\x00" + b"logs/worker.log") == set()


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
        f.pointer.write_text("{}", encoding="utf-8")
        expected = "already_offloaded"
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
            events.append(("pointer", f.pointer.is_file()))
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
    assert kinds.count("pointer") == 1  # nothing was kept, so the pointer is written once
    unlinks = [event for event in events if event[0] == "unlink"]
    assert all(pointer_present for _kind, pointer_present, _name in unlinks)
    assert sorted(name for *_rest, name in unlinks) == sorted(names)
    pointer = json.loads(f.pointer.read_text(encoding="utf-8"))
    assert pointer["schema_version"] == "control_plane_result_residue_pointer.v1"
    assert pointer["pointer_digest"] == canonical_digest(pointer, digest_field="pointer_digest")
    assert (pointer["run"], pointer["registry_digest"], pointer["kept"]) == (
        f.run.name, f.registry["registry_digest"], [])
    stored = f.client.objects[(BUCKET, pointer["archive"]["uri"].split(f"s3://{BUCKET}/", 1)[1])]
    assert (_sha(stored), len(stored)) == (pointer["archive"]["sha256"], pointer["archive"]["size_bytes"])
    assert sorted(pointer["members"], key=lambda row: row["relative_path"]) == [
        {"relative_path": relative, "size_bytes": len(RESIDUE[relative]), "sha256": _sha(RESIDUE[relative]),
         "mode": modes[relative]}
        for relative in sorted(RESIDUE)
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

    real_mkstemp = residue.tempfile.mkstemp

    def no_archive(*args, **kwargs):
        assert ".residue-" not in kwargs.get("prefix", ""), "the stream path must not stage a tar"
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(evidence, "reserve_control_plane_disk", reserve)
    monkeypatch.setattr(residue.tempfile, "mkstemp", no_archive)

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
    assert lied["failure"]["error_type"] == "ResultResidueOffloadError"
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


def test_residue_restore_round_trips(tmp_path) -> None:
    f = _sealed_run(tmp_path / "canaries")
    os.chmod(f.run / "logs" / "worker.log", 0o600)
    modes = {relative: stat.S_IMODE((f.run / relative).stat().st_mode) for relative in RESIDUE}
    before = _local_files(f.run)
    assert _offload(f)["offloaded_count"] == len(RESIDUE)
    pointer_bytes = f.pointer.read_bytes()
    materializer = functools.partial(store.materialize_configured_scene_artifact, client=f.client, bucket=BUCKET)

    restored = residue.restore_result_residue(run_root=f.run, materializer=materializer, now=lambda: NOW)

    assert (restored["status"], restored["restored_count"], restored["conflicts"]) == ("restored", len(RESIDUE), [])
    assert restored["restored_bytes"] == sum(len(data) for data in RESIDUE.values())
    assert _local_files(f.run) == before
    assert {relative: stat.S_IMODE((f.run / relative).stat().st_mode) for relative in RESIDUE} == modes
    assert (f.run / "logs/worker.log").stat().st_mtime == OLD
    assert f.pointer.read_bytes() == pointer_bytes
    receipt_path = f.evidence / f"{f.run.name}{residue.RESTORE_RECEIPT_SUFFIX}"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt == {**restored} and receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest")
    assert not [path for path in f.evidence.iterdir() if path.name.startswith(".")]

    # A second restore finds every member in place; a different local file is never overwritten.
    (f.run / "logs" / "worker.log").chmod(0o644)
    (f.run / "logs" / "worker.log").write_bytes(b"newer local truth\n")
    again = residue.restore_result_residue(run_root=f.run, materializer=materializer, now=lambda: NOW)
    assert (again["status"], again["restored_count"], again["already_present_count"]) == (
        "restored_with_conflicts", 0, len(RESIDUE) - 1)
    assert again["conflicts"] == [{"relative_path": "logs/worker.log", "reason": "existing_file_differs"}]
    assert (f.run / "logs" / "worker.log").read_bytes() == b"newer local truth\n"

    # A tampered pointer restores nothing.
    tampered = json.loads(pointer_bytes)
    tampered["members"][0]["size_bytes"] += 1
    f.pointer.chmod(0o640)
    f.pointer.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(residue.ResultResidueOffloadError, match="pointer_invalid"):
        residue.restore_result_residue(run_root=f.run, materializer=materializer, now=lambda: NOW)


def test_the_service_owner_is_given_each_file_last(tmp_path, monkeypatch) -> None:
    """The GC is root with CAP_CHOWN but not CAP_FOWNER: once a file belongs to the service
    user, root may no longer set its mode or times. So both happen first, the owner last."""

    f = _sealed_run(tmp_path / "canaries")
    calls: list[tuple] = []
    stranger = SimpleNamespace(st_uid=os.getuid() + 1, st_gid=os.getgid() + 1)
    descriptor = os.open(tmp_path / "probe", os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        with monkeypatch.context() as patched:
            patched.setattr(os, "fchmod", lambda fd, mode: calls.append(("chmod", mode)))
            patched.setattr(os, "fchown", lambda fd, uid, gid: calls.append(("chown", uid, gid)))
            residue._adopt_owner(descriptor, stranger, mode=0o440)
    finally:
        os.close(descriptor)
    assert calls == [("chmod", 0o440), ("chown", stranger.st_uid, stranger.st_gid)]

    _offload(f)
    order: list[str] = []
    real_utime, real_adopt = os.utime, residue._adopt_owner
    monkeypatch.setattr(os, "utime", lambda *args, **kwargs: (order.append("utime"), real_utime(*args, **kwargs))[1])
    monkeypatch.setattr(residue, "_adopt_owner", lambda *args, **kwargs: (
        order.append("adopt"), real_adopt(*args, **kwargs))[1])
    residue.restore_result_residue(run_root=f.run, now=lambda: NOW, materializer=functools.partial(
        store.materialize_configured_scene_artifact, client=f.client, bucket=BUCKET))
    # One (utime, adopt) pair per restored file, then the receipt's own adopt.
    assert order == ["utime", "adopt"] * len(RESIDUE) + ["adopt"]


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


def test_invalid_residue_setting_only_plans_and_alerts(tmp_path, monkeypatch, capsys) -> None:
    f = _sealed_run(tmp_path / "canaries")
    queue = tmp_path / "queue"
    (queue / "pending").mkdir(parents=True)
    for name in (gc_module.CONTENT_STORE_ROOTS_ENV, gc_module.DERIVED_ROOTS_ENV, gc_module.PLAN_ONLY_DERIVED_ROOTS_ENV,
                 gc_module.SETTLEMENT_ROOTS_ENV, gc_module.SCRATCH_ROOTS_ENV, gc_module.WORKSPACE_BUNDLE_ROOTS_ENV,
                 gc_module.SCENE_WORKSPACE_ROOTS_ENV, "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_PARENT_ROOTS",
                 gc_module.SCENE_WORKSPACE_RETIREMENT_ENV, "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_RETENTION",
                 gc_module.EVIDENCE_ABANDONED_AFTER_ENV):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv(gc_module.QUEUE_ROOTS_ENV, str(queue))
    monkeypatch.setenv(gc_module.EVIDENCE_ROOTS_ENV, str(f.evidence))
    monkeypatch.setenv(gc_module.EVIDENCE_OFFLOAD_ENV, "1")
    monkeypatch.setenv(residue.RESIDUE_OFFLOAD_ENV, "maybe")
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


def test_every_way_a_kept_document_names_a_run_file_keeps_it() -> None:
    """A run file may be named absolutely (the run's own name may recur deeper in the path),
    relative to the evidence root, or relative to the run; each keeps it."""

    import io

    name = "run-7"
    value = {"nested": f"/var/lib/canaries/{name}/work/{name}/state.npz",
             "rooted": [f"{name}/logs/worker.log"], "relative": "provider/outputs.zip",
             "beside": "notes.txt", "above": "stage/state.npz"}

    def strings(raw: bytes) -> set[str]:
        return set().union(*residue._stream_tokens(io.BytesIO(raw)))

    named = set(residue._named_paths(strings(json.dumps(value).encode()), "work/stage/index.json", name))

    assert {f"work/{name}/state.npz", "logs/worker.log", "provider/outputs.zip", "work/stage/notes.txt",
            "work/stage/state.npz"} <= named
    # Free text and JSON lines name files too.
    assert "logs/worker.log" in strings(b"see logs/worker.log, then retry\n")
    assert f"/x/{name}/a.bin" in strings(b'{"a": 1}\n{"b": "/x/' + name.encode() + b'/a.bin"}\n')


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


@pytest.mark.parametrize("damage", ["linked_row", "oversized_row", "linked_state"])
def test_a_dispatch_row_that_cannot_be_read_keeps_every_run(tmp_path, monkeypatch, damage) -> None:
    """A row that cannot be read might name any run, so none moves."""

    f = _sealed_run(tmp_path / "canaries")
    queue = _dispatch_queue(tmp_path / "dispatches")
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "row.json").write_text("{}", encoding="utf-8")
    if damage == "linked_row":
        (queue / "pending" / "linked.json").symlink_to(tmp_path / "elsewhere" / "row.json")
    elif damage == "oversized_row":
        monkeypatch.setattr(residue, "MAX_QUEUE_MESSAGE_BYTES", 16)
        (queue / "processing" / "large.json").write_text(json.dumps({"activation_id": "another-run"}), encoding="utf-8")
    else:
        (queue / "processing").rmdir()
        (queue / "processing").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    before = _local_files(f.run)

    result = _offload(f, queue_roots=[queue])

    assert (result["status"], result["retained_reason"]) == ("retained", "dispatch_queue_unreadable")
    assert _local_files(f.run) == before and not f.pointer.exists()


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
                  result_residue_offload_enabled=True, result_residue_max_runs_per_tick=2)

    phase = first["result_residue_offload"]
    assert (phase["max_runs_per_tick"], phase["attempted_count"]) == (2, 2)
    assert [(row["run"], row["status"], row["retained_reason"]) for row in phase["runs"]] == [
        ("run-0", "applied", None), ("run-1", "applied", None), ("run-2", "retained", "deferred_tick_cap")]
    assert phase["retained_by_reason"]["deferred_tick_cap"] == {"count": 1, "bytes": residue_bytes}
    assert (phase["candidate_bytes"], phase["offloaded_bytes"]) == (3 * residue_bytes, 2 * residue_bytes)
    assert set(RESIDUE) <= set(_local_files(runs[2].run)) and not runs[2].pointer.exists()

    second = _tick(runs[0], pins, queue, apply=True, ack=RUN_ACK, offload_enabled=True,
                   result_residue_offload_enabled=True, result_residue_max_runs_per_tick=2)
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
                   result_residue_offload_enabled=True, result_residue_max_runs_per_tick=2, publisher=failing)

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
    for name in (gc_module.CONTENT_STORE_ROOTS_ENV, gc_module.DERIVED_ROOTS_ENV, gc_module.PLAN_ONLY_DERIVED_ROOTS_ENV,
                 gc_module.SETTLEMENT_ROOTS_ENV, gc_module.SCRATCH_ROOTS_ENV, gc_module.WORKSPACE_BUNDLE_ROOTS_ENV,
                 gc_module.SCENE_WORKSPACE_ROOTS_ENV, "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_PARENT_ROOTS",
                 gc_module.SCENE_WORKSPACE_RETIREMENT_ENV, "BLUEPRINT_CONTROL_PLANE_GC_REPLAY_CACHE_RETENTION",
                 gc_module.EVIDENCE_ABANDONED_AFTER_ENV):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv(gc_module.QUEUE_ROOTS_ENV, str(queue))
    monkeypatch.setenv(gc_module.EVIDENCE_ROOTS_ENV, str(f.evidence))
    monkeypatch.setenv(gc_module.EVIDENCE_OFFLOAD_ENV, "1")
    monkeypatch.setenv(residue.RESIDUE_OFFLOAD_ENV, "1")
    monkeypatch.setenv(residue.RESIDUE_MAX_RUNS_ENV, raw)
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
