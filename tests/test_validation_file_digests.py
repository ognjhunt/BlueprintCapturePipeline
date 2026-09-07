import hashlib
import os

import pytest

from blueprint_pipeline import validation_file_digests as subject


def sealed(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"a" * subject.MINIMUM_BYTES)
    path.chmod(0o440)
    return path


def count_hashes(monkeypatch):
    calls = []
    real = hashlib.sha256
    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(subject.hashlib, "sha256", counted)
    return calls


def test_nested_scope_hashes_sealed_bytes_once_but_next_attempt_reopens(tmp_path, monkeypatch):
    path = sealed(tmp_path)
    calls = count_hashes(monkeypatch)
    with subject.file_digest_scope():
        first = subject.sha256_file(path)
        with subject.file_digest_scope():
            assert subject.sha256_file(path) == first
        assert len(calls) == 1
    with subject.file_digest_scope():
        assert subject.sha256_file(path) == first
    assert len(calls) == 2


def test_tamper_with_restored_mode_size_and_mtime_invalidates(tmp_path, monkeypatch):
    path = sealed(tmp_path)
    before = path.stat()
    calls = count_hashes(monkeypatch)
    with subject.file_digest_scope():
        first = subject.sha256_file(path)
        path.chmod(0o640)
        path.write_bytes(b"b" * before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        path.chmod(0o440)
        assert subject.sha256_file(path) != first
    assert len(calls) == 2


@pytest.mark.parametrize("large,writable", [(False, False), (True, True)])
def test_small_authority_or_writable_file_never_cached(tmp_path, monkeypatch, large, writable):
    path = sealed(tmp_path)
    path.chmod(0o640)
    if not large:
        path.write_text('{"authorized": true}')
    if not writable:
        path.chmod(0o440)
    calls = count_hashes(monkeypatch)
    with subject.file_digest_scope():
        subject.sha256_file(path)
        subject.sha256_file(path)
    assert len(calls) == 2


def test_replaced_path_does_not_reuse_inode_hash(tmp_path):
    path = sealed(tmp_path)
    with subject.file_digest_scope():
        first = subject.sha256_file(path)
        alternate = tmp_path / "replacement"
        alternate.write_bytes(b"b" * subject.MINIMUM_BYTES)
        alternate.chmod(0o440)
        alternate.replace(path)
        assert subject.sha256_file(path) != first


def test_symlink_after_cached_read_is_refused(tmp_path):
    path = sealed(tmp_path)
    with subject.file_digest_scope():
        subject.sha256_file(path)
        other = tmp_path / "original"
        path.rename(other)
        path.symlink_to(other)
        with pytest.raises(ValueError, match="symlink_forbidden"):
            subject.sha256_file(path)


def test_exception_drops_scope_cache(tmp_path, monkeypatch):
    path = sealed(tmp_path)
    calls = count_hashes(monkeypatch)
    with pytest.raises(RuntimeError), subject.file_digest_scope():
        subject.sha256_file(path)
        raise RuntimeError("validation refused")
    with subject.file_digest_scope():
        subject.sha256_file(path)
    assert len(calls) == 2


def test_measurement_cache_binds_pixels_parameters_and_copies_results(tmp_path, monkeypatch):
    from PIL import Image
    from blueprint_pipeline import source_calibration_camera_resolution as camera
    groups = {}
    for role, color in (("images", "white"), ("target_support", "white"),
                        ("scene_without_target", "black")):
        root = tmp_path / role
        (root / "frames").mkdir(parents=True)
        Image.new("RGB", (16, 16), color).save(root / "frames" / "source.png")
        groups[role] = {"root": root}
    gate = {"support_threshold_8bit": 1, "visual_contribution_threshold_8bit": 8,
            "minimum_visible_target_fraction": .01}
    real = camera._measure_pixels
    calls = []
    def counted(*args):
        calls.append(1)
        return real(*args)
    monkeypatch.setattr(camera, "_measure_pixels", counted)
    with subject.file_digest_scope():
        first = camera.measure_candidate(groups, "source", gate)
        first["support_contribution_pixels"] = -1
        assert camera.measure_candidate(groups, "source", gate)["support_contribution_pixels"] == 256
        assert len(calls) == 1
        camera.measure_candidate(groups, "source", {**gate, "support_threshold_8bit": 256})
        assert len(calls) == 2
        Image.new("RGB", (16, 16), "white").save(
            groups["scene_without_target"]["root"] / "frames" / "source.png")
        assert camera.measure_candidate(groups, "source", gate)["support_contribution_pixels"] == 0
        assert len(calls) == 3
    camera.measure_candidate(groups, "source", gate)
    assert len(calls) == 4
