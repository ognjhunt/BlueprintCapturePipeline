"""Scene-scale inputs are read in blocks, never whole, and every recorded identity is what a whole read gave.

The launch dispatcher held a 1.2 GB provider bundle and its readback in memory (2.4 GB of anon memory per
dispatch on an 8.3 GB host with no swap); the resolver, the runtime-source packet check, the wheelhouse check
and the screening file record each held one whole file or member.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

import blueprint_pipeline.launch_immutable_input_writer as writer
import blueprint_pipeline.task_evaluation_launch_dispatcher as dispatcher
from blueprint_pipeline import beta_policy_screening_files, native_task_runtime_source_packet
from blueprint_pipeline import task_evaluation_scene_configuration_python_wheelhouse as wheelhouse
from blueprint_pipeline.task_evaluation_immutable_input_resolver import (
    STAGING_RECEIPT_ENV,
    ImmutableInputResolutionError,
    resolve_immutable_input,
)

BLOCK = 64 * 1024


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


@pytest.fixture
def no_whole_reads(monkeypatch):
    """Small copy blocks, and any whole-file read of a guarded path fails the test."""

    guarded: list[Path] = []
    original = Path.read_bytes

    def read_bytes(self):
        if any(self.resolve().is_relative_to(root) for root in guarded):
            raise AssertionError(f"whole read of {self}")
        return original(self)

    monkeypatch.setattr(writer, "COPY_BLOCK_BYTES", BLOCK)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return guarded


def _stage(tmp_path: Path, *, projection: bool):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    bundle = source_dir / "provider_bundle.zip"
    bundle.write_bytes(os.urandom(5 * BLOCK + 123))
    receipt = source_dir / "bundle.receipt.json"
    receipt.write_bytes(b'{"bundle": "provider_bundle.zip"}\n')
    inputs = [{"name": "provider_bundle", "path": str(bundle), "digest": _digest(bundle.read_bytes())},
              {"name": "source_bundle_manifest", "path": str(receipt), "digest": _digest(receipt.read_bytes())},
              {"name": "evaluation_run_spec", "path": str(receipt), "digest": _digest(receipt.read_bytes())}]
    profile = {"profile_id": "p", "profile_digest": "sha256:" + "0" * 64, "immutable_inputs": inputs}
    argv = ["python", "--bundle", str(bundle)] + (["--bundle-dir", str(source_dir)] if projection else [])
    sources = {row["path"]: Path(row["path"]).resolve() for row in inputs}
    projections, bindings = dispatcher._immutable_input_directory_projections(set(sources.values()), argv)
    return dict(profile=profile, run_root=tmp_path / "run", allocator_argv=argv, projections=projections,
                planned_sources=sources, directory_bindings=bindings), bundle, receipt


@pytest.mark.parametrize("projection", [False, True])
def test_dispatch_staging_copies_in_blocks_and_records_what_a_whole_read_gave(tmp_path, no_whole_reads, projection):
    arguments, bundle, receipt = _stage(tmp_path, projection=projection)
    expected = {path: path.read_bytes() for path in (bundle, receipt)}
    no_whole_reads.extend([tmp_path / "source", tmp_path / "run" / "immutable_inputs"])
    staged, rewritten = dispatcher._stage_profile_immutable_inputs_reserved(**arguments)
    rows = staged["inputs"]
    # Each row is the whole bytes' identity; the duplicate name shares the first copy, as before.
    for row in rows:
        data = expected[Path(row["source_path"])]
        assert row["staged_size_bytes"] == len(data)
        assert row["staged_digest"] == row["expected_digest"] == _digest(data)
        assert Path(row["staged_path"]).stat().st_size == len(data)
    assert rows[1]["staged_path"] == rows[2]["staged_path"]
    assert {Path(row["staged_path"]).name for row in rows} == {
        f"000-{_digest(expected[bundle])[7:]}.input", f"001-{_digest(expected[receipt])[7:]}.input"}
    no_whole_reads.clear()
    for row in rows:
        path = Path(row["staged_path"])
        assert path.read_bytes() == expected[Path(row["source_path"])]
        assert path.stat().st_mode & 0o777 == 0o600
    if projection:
        (view,) = staged["directory_projections"]
        assert sorted((row["relative_path"], row["digest"], row["size_bytes"]) for row in view["inputs"]) == sorted(
            (path.name, _digest(data), len(data)) for path, data in expected.items())
        for row in view["inputs"]:
            assert Path(row["projected_path"]).read_bytes() == expected[Path(row["source_path"])]
        assert rewritten[-1] == view["staged_directory"]
    else:
        assert staged["directory_projections"] == []
    assert rewritten[2] == rows[0]["staged_path"]
    on_disk = json.loads((tmp_path / "run" / "immutable_input_staging_receipt.json").read_text())
    assert on_disk == staged


def test_a_source_that_changes_after_its_digest_check_is_refused_and_never_staged(tmp_path, monkeypatch):
    arguments, bundle, _ = _stage(tmp_path, projection=False)
    copy = writer.copy_exclusive_private_file

    def tampered(path, source, **options):
        if source == bundle.resolve():
            bundle.write_bytes(b"tampered after the digest check")
        return copy(path, source, **options)

    monkeypatch.setattr(writer, "copy_exclusive_private_file", tampered)
    with pytest.raises(dispatcher.TaskEvaluationLaunchError, match="immutable_input_staging_readback_mismatch"):
        dispatcher._stage_profile_immutable_inputs_reserved(**arguments)
    stage = tmp_path / "run" / "immutable_inputs"
    assert sorted(p.name for p in stage.iterdir()) == []
    assert not (tmp_path / "run" / "immutable_input_staging_receipt.json").exists()


def test_an_existing_copy_with_other_bytes_is_a_conflict(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(os.urandom(3 * BLOCK))
    target = tmp_path / "staged" / "input"
    assert writer.copy_exclusive_private_file(target, source)[0] is True
    created, digest, size = writer.copy_exclusive_private_file(target, source)
    assert (created, digest, size) == (False, hashlib.sha256(source.read_bytes()).hexdigest(), 3 * BLOCK)
    source.write_bytes(os.urandom(3 * BLOCK))
    with pytest.raises(writer.TaskEvaluationLaunchError, match="immutable_input_staging_conflict"):
        writer.copy_exclusive_private_file(target, source)
    assert sorted(p.name for p in target.parent.iterdir()) == ["input"]


def test_resolver_checks_a_staged_input_in_blocks(tmp_path, monkeypatch, no_whole_reads):
    stage = tmp_path / "run" / "immutable_inputs"
    stage.mkdir(parents=True)
    data = os.urandom(4 * BLOCK + 7)
    staged = stage / "000.input"
    staged.write_bytes(data)
    original = tmp_path / "bundle.zip"
    receipt = {"schema_version": "task_evaluation_immutable_input_staging.v1", "status": "staged",
               "raw_secret_values_recorded": False, "receipt_digest": "",
               "inputs": [{"source_path": str(original), "expected_digest": _digest(data),
                           "staged_digest": _digest(data), "staged_size_bytes": len(data),
                           "staged_path": str(staged)}]}
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    receipt_path = tmp_path / "run" / "immutable_input_staging_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    monkeypatch.setenv(STAGING_RECEIPT_ENV, str(receipt_path))
    no_whole_reads.append(stage)
    assert resolve_immutable_input(original, expected_digest=_digest(data), expected_size_bytes=len(data)) == staged
    with staged.open("r+b") as stream:
        stream.seek(BLOCK)
        stream.write(b"\0")
    with pytest.raises(ImmutableInputResolutionError, match="target_identity_mismatch"):
        resolve_immutable_input(original, expected_digest=_digest(data), expected_size_bytes=len(data))


def test_runtime_source_member_identity_is_read_in_blocks(tmp_path, monkeypatch):
    wheel = os.urandom(3 * (1 << 20) + 11)
    packet = tmp_path / "packet.zip"
    with zipfile.ZipFile(packet, "w") as archive:
        archive.writestr("wheels/torch.whl", wheel)
    monkeypatch.setattr(zipfile.ZipFile, "read", lambda *_: pytest.fail("whole member read"))
    with zipfile.ZipFile(packet) as archive:
        assert native_task_runtime_source_packet._member_size_and_sha256(archive, "wheels/torch.whl") == (
            len(wheel), _digest(wheel))
        with pytest.raises(KeyError):
            native_task_runtime_source_packet._member_size_and_sha256(archive, "wheels/missing.whl")


def test_screening_record_and_wheel_digest_are_the_whole_bytes_identity(tmp_path, no_whole_reads):
    data = os.urandom(3 * (1 << 20) + 5)
    collider = tmp_path / "collision.usda"
    collider.write_bytes(data)
    no_whole_reads.append(tmp_path)
    assert beta_policy_screening_files._record(collider) == {
        "path": str(collider.resolve()), "size_bytes": len(data), "sha256": _digest(data)}
    assert wheelhouse._file_sha256(collider) == _digest(data)
