"""A new attempt shares an earlier attempt's verified staged inputs instead of copying them.

capture 819be, 2026-11-02: every preparation attempt of one scene copied the
same 0.44 GiB of USD into ``<factory>/<intent>/<attempt>/materialized/
submission/website/inputs``; 28 attempts held 10.5 GiB of identical bytes.
"""
from __future__ import annotations

import errno
import os
import stat

import pytest

from blueprint_pipeline import task_evaluation_scene_configuration_submission_inputs as inputs
from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import Staging
from blueprint_pipeline.website_scene_dispatch import earlier_attempt_submissions

PAYLOAD = b"usd scene bytes " * 4096  # 64 KiB


@pytest.fixture(autouse=True)
def small_threshold(monkeypatch):
    monkeypatch.setattr(inputs, "REUSE_MIN_BYTES", 1024)


def _source(tmp_path, payload=PAYLOAD, name="scene.usda"):
    path = tmp_path / "workspace" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _stage(tmp_path, name, *, reuse_roots):
    return Staging(tmp_path / name / "submission", "namespace", reuse_roots=reuse_roots)


def _writable(path):
    return bool(stat.S_IMODE(path.stat().st_mode) & 0o222)


def test_legacy_staging_still_copies_and_leaves_modes_alone(tmp_path):
    source = _source(tmp_path)
    stage = Staging(tmp_path / "legacy", "namespace")
    stage.copy(source, "website/inputs/000.usda")
    target = stage.root / "website/inputs/000.usda"
    assert target.read_bytes() == PAYLOAD
    assert target.stat().st_ino != source.stat().st_ino
    assert _writable(target) and stage.reused == {"files": 0, "bytes": 0}


def test_second_attempt_links_the_first_attempts_sealed_copy(tmp_path):
    source = _source(tmp_path)
    first = _stage(tmp_path, "source-a", reuse_roots=())
    first_ref = first.copy(source, "website/inputs/000.usda")
    sealed = first.root / "website/inputs/000.usda"
    # The first copy is private (not the workspace inode) and sealed read-only.
    assert sealed.stat().st_ino != source.stat().st_ino and not _writable(sealed)
    second = _stage(tmp_path, "source-b", reuse_roots=(first.root,))
    second_ref = second.copy(source, "website/inputs/000.usda")
    shared = second.root / "website/inputs/000.usda"
    assert shared.stat().st_ino == sealed.stat().st_ino and shared.stat().st_nlink == 2
    assert second.reused == {"files": 1, "bytes": len(PAYLOAD)}
    # The recorded reference is exactly what a copy would have recorded.
    assert {k: v for k, v in second_ref.items() if k != "uri"} == {
        k: v for k, v in first_ref.items() if k != "uri"}
    # The workspace source is never linked or chmodded.
    assert source.stat().st_nlink == 1 and _writable(source)


def test_small_inputs_are_copied_and_left_writable(tmp_path, monkeypatch):
    monkeypatch.setattr(inputs, "REUSE_MIN_BYTES", len(PAYLOAD) + 1)
    source = _source(tmp_path)
    first = _stage(tmp_path, "source-a", reuse_roots=())
    first.copy(source, "website/inputs/000.usda")
    second = _stage(tmp_path, "source-b", reuse_roots=(first.root,))
    second.copy(source, "website/inputs/000.usda")
    a, b = (stage.root / "website/inputs/000.usda" for stage in (first, second))
    assert a.stat().st_ino != b.stat().st_ino and _writable(a) and _writable(b)


def test_different_bytes_of_the_same_size_are_never_linked(tmp_path):
    first = _stage(tmp_path, "source-a", reuse_roots=())
    first.copy(_source(tmp_path, PAYLOAD.replace(b"usd", b"USD"), "old.usda"), "website/inputs/000.usda")
    second = _stage(tmp_path, "source-b", reuse_roots=(first.root,))
    second.copy(_source(tmp_path), "website/inputs/000.usda")
    target = second.root / "website/inputs/000.usda"
    assert target.read_bytes() == PAYLOAD
    assert target.stat().st_ino != (first.root / "website/inputs/000.usda").stat().st_ino
    assert second.reused["files"] == 0


def test_a_writable_earlier_copy_is_not_shared(tmp_path):
    # Attempts staged before this change hold writable copies; they stay private.
    source = _source(tmp_path)
    old = Staging(tmp_path / "source-old" / "submission", "namespace")
    old.copy(source, "website/inputs/000.usda")
    second = _stage(tmp_path, "source-b", reuse_roots=(old.root,))
    second.copy(source, "website/inputs/000.usda")
    assert ((second.root / "website/inputs/000.usda").stat().st_ino
            != (old.root / "website/inputs/000.usda").stat().st_ino)
    assert second.reused["files"] == 0


def test_a_symlinked_candidate_is_never_followed(tmp_path):
    source = _source(tmp_path)
    first = _stage(tmp_path, "source-a", reuse_roots=())
    first.copy(source, "website/inputs/000.usda")
    elsewhere = tmp_path / "elsewhere" / "submission"
    (elsewhere / "website").mkdir(parents=True)
    (elsewhere / "website" / "inputs").symlink_to(first.root / "website" / "inputs")
    second = _stage(tmp_path, "source-b", reuse_roots=(elsewhere,))
    second.copy(source, "website/inputs/000.usda")
    assert second.reused["files"] == 0
    assert ((second.root / "website/inputs/000.usda").stat().st_ino
            != (first.root / "website/inputs/000.usda").stat().st_ino)


@pytest.mark.parametrize("code", [errno.EXDEV, errno.EPERM, errno.EMLINK])
def test_a_refused_link_falls_back_to_a_verified_copy(tmp_path, monkeypatch, code):
    source = _source(tmp_path)
    first = _stage(tmp_path, "source-a", reuse_roots=())
    first.copy(source, "website/inputs/000.usda")
    second = _stage(tmp_path, "source-b", reuse_roots=(first.root,))

    def refuse(*_args, **_kwargs):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(inputs.os, "link", refuse)
    second.copy(source, "website/inputs/000.usda")
    target = second.root / "website/inputs/000.usda"
    assert target.read_bytes() == PAYLOAD and second.reused["files"] == 0
    assert target.stat().st_ino != (first.root / "website/inputs/000.usda").stat().st_ino


def test_an_unexpected_link_error_is_not_hidden(tmp_path, monkeypatch):
    source = _source(tmp_path)
    first = _stage(tmp_path, "source-a", reuse_roots=())
    first.copy(source, "website/inputs/000.usda")
    second = _stage(tmp_path, "source-b", reuse_roots=(first.root,))

    def broken(*_args, **_kwargs):
        raise OSError(errno.EIO, os.strerror(errno.EIO))

    monkeypatch.setattr(inputs.os, "link", broken)
    with pytest.raises(OSError):
        second.copy(source, "website/inputs/000.usda")


def _attempt(intent, name, mtime_ns):
    root = intent / name / "materialized" / "submission"
    root.mkdir(parents=True)
    os.utime(root, ns=(mtime_ns, mtime_ns))
    return root


def test_earlier_attempts_are_this_intents_other_attempts_newest_first(tmp_path):
    intent = tmp_path / "factory" / "scene-abc"
    roots = [_attempt(intent, f"source-{index:024d}", (index + 1) * 10**9) for index in range(5)]
    own = intent / "source-own" / "materialized"
    own.mkdir(parents=True)
    (intent / "website-source").mkdir()
    (intent / "source-not-materialized").mkdir()
    (intent / "source-linked").symlink_to(roots[0].parent.parent)
    found = earlier_attempt_submissions(own, "source-own")
    assert found == (roots[4], roots[3], roots[2])


def test_another_layout_shares_nothing(tmp_path):
    intent = tmp_path / "factory" / "scene-abc"
    _attempt(intent, "source-earlier", 10**9)
    output = intent / "source-own" / "elsewhere"
    output.mkdir(parents=True)
    assert earlier_attempt_submissions(output, "source-own") == ()
    assert earlier_attempt_submissions(intent / "source-own" / "materialized", "source-other") == ()
