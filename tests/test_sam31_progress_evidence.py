"""The SAM read seam preserves immutable evidence and progress admission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_sam31_preparation_queue as queue
from blueprint_pipeline import task_evaluation_sam31_progress_evidence as evidence
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_live_pipeline_import_isolation import HOT_LANE_MODULES, _transitive_local_modules


@pytest.fixture(params=(evidence, queue), ids=("pure_reader", "queue_reexport"))
def reader(request):
    return request.param


def _reference(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data)}


def _checkpoint(sequence: int, previous: dict | None = None) -> dict:
    value = {"schema_version": evidence.PROGRESS_SCHEMA,
             "request_digest": "sha256:" + "a" * 64,
             "sequence": sequence,
             "previous_progress_digest": previous["progress_digest"] if previous else None,
             "status": "waiting_for_child"}
    value["progress_digest"] = canonical_digest(value, digest_field="progress_digest")
    return value


def _store_progress(root: Path, values: list[dict]) -> None:
    directory = root / "source-progress" / "prep"
    directory.mkdir(parents=True, exist_ok=True)
    for index, value in enumerate(values, 1):
        (directory / f"{index:06d}.json").write_text(json.dumps(value))


def _snapshot(root: Path) -> dict:
    return {path.relative_to(root): (path.stat().st_mode, path.stat().st_mtime_ns,
                                    path.read_bytes() if path.is_file() else None)
            for path in root.rglob("*")}


def test_queue_reexports_preserve_object_identity():
    for name in ("Sam31PreparationQueueError", "_require", "_read", "PROGRESS_SCHEMA",
                 "load_progress", "verify_evidence_reference"):
        assert getattr(queue, name) is getattr(evidence, name)


def test_pure_readers_do_not_reach_preparation_or_execution_services():
    reachable = _transitive_local_modules(("task_evaluation_sam31_progress_evidence",))
    assert "decision_evidence_contracts" in reachable
    assert not reachable.intersection(HOT_LANE_MODULES)
    assert not reachable.intersection({
        "task_evaluation_sam31_preparation_queue",
        "task_evaluation_scene_configuration_sam31_preparation_driver",
        "task_evaluation_sam31_preparation_execution",
        "task_evaluation_sam31_preparation_stages",
        "task_evaluation_sam31_preparation_paid_stages",
        "task_evaluation_stage_replay",
    })


def test_valid_reads_accept_exact_bytes_and_digest_alias_without_writes(reader, tmp_path):
    artifact = tmp_path / "child-result.json"
    artifact.write_bytes(b"child evidence" * 100_000)
    reference = _reference(artifact)
    first = _checkpoint(1)
    second = _checkpoint(2, first)
    _store_progress(tmp_path, [first, second])
    before = _snapshot(tmp_path)

    assert reader.verify_evidence_reference(reference, (tmp_path,)) == artifact
    alias = {"path": reference["path"], "digest": reference["sha256"],
             "size_bytes": reference["size_bytes"]}
    assert reader.verify_evidence_reference(alias, (tmp_path,)) == artifact
    assert reader.load_progress(tmp_path, "prep.json", first["request_digest"]) == second
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("change", ["same_size_tamper", "digest", "size", "boolean_size", "zero_size"])
def test_evidence_rejects_byte_digest_and_size_changes(reader, tmp_path, change):
    artifact = tmp_path / "child.json"
    artifact.write_bytes(b"original")
    reference = _reference(artifact)
    if change == "same_size_tamper":
        artifact.write_bytes(b"tampered")
    elif change == "digest":
        reference["sha256"] = "sha256:" + "0" * 64
    elif change == "size":
        reference["size_bytes"] += 1
    elif change == "boolean_size":
        reference["size_bytes"] = True
    else:
        reference["size_bytes"] = 0
    with pytest.raises(reader.Sam31PreparationQueueError,
                       match="^sam31_preparation_evidence_readback_mismatch$"):
        reader.verify_evidence_reference(reference, (tmp_path,))


@pytest.mark.parametrize("change", ["relative", "parent_segment", "missing", "outside_root",
                                   "no_roots", "file_symlink", "ancestor_symlink"])
def test_evidence_rejects_unsafe_paths_and_unapproved_roots(reader, tmp_path, change):
    admitted = tmp_path / "admitted"
    admitted.mkdir()
    artifact = admitted / "child.json"
    artifact.write_bytes(b"child")
    reference = _reference(artifact)
    roots = (admitted,)
    if change == "relative":
        reference["path"] = "admitted/child.json"
    elif change == "parent_segment":
        reference["path"] = str(admitted / ".." / "admitted" / "child.json")
    elif change == "missing":
        reference["path"] = str(admitted / "missing.json")
    elif change == "outside_root":
        sibling = tmp_path / "admitted-sibling"
        sibling.mkdir()
        outside = sibling / "child.json"
        outside.write_bytes(b"child")
        reference["path"] = str(outside)
    elif change == "no_roots":
        roots = ()
    elif change == "file_symlink":
        link = admitted / "linked.json"
        link.symlink_to(artifact)
        reference["path"] = str(link)
    else:
        link = tmp_path / "linked-directory"
        link.symlink_to(admitted, target_is_directory=True)
        reference["path"] = str(link / artifact.name)
        roots = (tmp_path,)
    with pytest.raises(reader.Sam31PreparationQueueError,
                       match="^sam31_preparation_evidence_path_invalid$"):
        reader.verify_evidence_reference(reference, roots)


@pytest.mark.parametrize("change", ["schema", "request", "first_predecessor", "sequence_gap",
                                   "sequence_duplicate", "predecessor", "canonical_digest"])
def test_progress_rejects_rebound_or_broken_chains(reader, tmp_path, change):
    first = _checkpoint(1)
    second = _checkpoint(2, first)
    changed = first if change == "first_predecessor" else second
    if change == "schema":
        changed["schema_version"] = "different.v1"
    elif change == "request":
        changed["request_digest"] = "sha256:" + "b" * 64
    elif change == "first_predecessor":
        changed["previous_progress_digest"] = "sha256:" + "0" * 64
    elif change == "sequence_gap":
        changed["sequence"] = 3
    elif change == "sequence_duplicate":
        changed["sequence"] = 1
    elif change == "predecessor":
        changed["previous_progress_digest"] = "sha256:" + "0" * 64
    else:
        changed["status"] = "ready"
    if change != "canonical_digest":
        changed["progress_digest"] = canonical_digest(changed, digest_field="progress_digest")
    _store_progress(tmp_path, [first, second])
    with pytest.raises(reader.Sam31PreparationQueueError,
                       match="^sam31_preparation_progress_chain_invalid$"):
        reader.load_progress(tmp_path, "prep.json", first["request_digest"])


def test_progress_rejects_changed_requested_identity(reader, tmp_path):
    _store_progress(tmp_path, [_checkpoint(1)])
    with pytest.raises(reader.Sam31PreparationQueueError,
                       match="^sam31_preparation_progress_chain_invalid$"):
        reader.load_progress(tmp_path, "prep.json", "sha256:" + "b" * 64)


def test_missing_progress_does_not_create_queue_directories(reader, tmp_path):
    root = tmp_path / "missing"
    assert reader.load_progress(root, "prep.json", "sha256:" + "a" * 64) is None
    assert not root.exists()


def test_progress_rejects_symlinked_directory(reader, tmp_path):
    first = _checkpoint(1)
    actual = tmp_path / "actual"
    _store_progress(actual, [first])
    (tmp_path / "source-progress").mkdir()
    (tmp_path / "source-progress" / "prep").symlink_to(
        actual / "source-progress" / "prep", target_is_directory=True)
    with pytest.raises(reader.Sam31PreparationQueueError,
                       match="^sam31_preparation_progress_path_invalid$"):
        reader.load_progress(tmp_path, "prep.json", first["request_digest"])


def test_record_reader_retains_exact_four_mib_limit(reader, tmp_path):
    record = tmp_path / "record.json"
    record.write_bytes(b"{}" + b" " * (4 * 1024 * 1024 - 2))
    assert reader._read(record) == {}
    with record.open("ab") as target:
        target.write(b" ")
    with pytest.raises(reader.Sam31PreparationQueueError,
                       match="^sam31_preparation_record_path_invalid$"):
        reader._read(record)


@pytest.mark.parametrize("change", ["not_mapping", "file_symlink", "ancestor_symlink"])
def test_record_reader_rejects_untrusted_record_shapes_and_paths(reader, tmp_path, change):
    directory = tmp_path / "records"
    directory.mkdir()
    record = directory / "record.json"
    record.write_text("[]" if change == "not_mapping" else "{}")
    code = "record_invalid" if change == "not_mapping" else "record_path_invalid"
    if change == "file_symlink":
        link = directory / "linked.json"
        link.symlink_to(record)
        record = link
    elif change == "ancestor_symlink":
        link = tmp_path / "linked"
        link.symlink_to(directory, target_is_directory=True)
        record = link / record.name
    with pytest.raises(reader.Sam31PreparationQueueError, match=f"^sam31_preparation_{code}$"):
        reader._read(record)
