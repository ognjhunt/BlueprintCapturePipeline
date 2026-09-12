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


@pytest.mark.parametrize("mode", [0o440, 0o640])
def test_small_authority_file_never_cached_regardless_of_mode(tmp_path, monkeypatch, mode):
    path = sealed(tmp_path)
    path.chmod(0o640)
    path.write_text('{"authorized": true}')
    path.chmod(mode)
    calls = count_hashes(monkeypatch)
    with subject.file_digest_scope():
        subject.sha256_file(path)
        subject.sha256_file(path)
    assert len(calls) == 2


@pytest.mark.parametrize("mode", [0o600, 0o644])
def test_large_writable_retained_artifact_hashed_once_per_scope(tmp_path, monkeypatch, mode):
    """Producers retain PLY/bundle/video artifacts as 0600/0644.

    A write bit is not a change witness; the nanosecond change time is. Refusing
    to reuse these hashes made one restart re-hash 4.6 GB about 70 times.
    """
    path = tmp_path / "source_standard.ply"
    path.write_bytes(b"a" * subject.MINIMUM_BYTES)
    path.chmod(mode)
    calls = count_hashes(monkeypatch)
    with subject.file_digest_scope():
        first = subject.sha256_file(path)
        for _ in range(14 * 6):  # every reuse candidate times every reusable phase
            assert subject.sha256_file(path) == first
    assert len(calls) == 1
    with subject.file_digest_scope():
        assert subject.sha256_file(path) == first
    assert len(calls) == 2


@pytest.mark.parametrize("mode", [0o600, 0o644])
def test_writable_artifact_rewrite_with_restored_size_and_mtime_invalidates(tmp_path, monkeypatch, mode):
    path = tmp_path / "source_standard.ply"
    path.write_bytes(b"a" * subject.MINIMUM_BYTES)
    path.chmod(mode)
    before = path.stat()
    calls = count_hashes(monkeypatch)
    with subject.file_digest_scope():
        first = subject.sha256_file(path)
        path.write_bytes(b"b" * before.st_size)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        assert subject.sha256_file(path) != first
        assert subject.sha256_file(path) != first
    assert len(calls) == 2


def test_production_loop_shape_hashes_each_distinct_artifact_once(tmp_path, monkeypatch):
    """Factory shape: N prior attempts x M reusable phases, each reopening the same
    retained artifacts through the submission-inputs ``sha`` helper."""
    from blueprint_pipeline.task_evaluation_scene_configuration_submission_inputs import sha
    artifacts = []
    for name, mode in (("source_standard.ply", 0o600), ("images.ply", 0o600),
                       ("input_bundle.tar", 0o644), ("retained_sequence.mp4", 0o440)):
        path = tmp_path / name
        path.write_bytes(name.encode() * (subject.MINIMUM_BYTES // len(name) + 1))
        path.chmod(mode)
        artifacts.append(path)
    calls = count_hashes(monkeypatch)
    with subject.file_digest_scope():
        digests = {path: sha(path) for path in artifacts}
        for _candidate in range(14):
            for _phase in range(6):
                for path in artifacts:
                    assert sha(path) == digests[path]
    assert len(calls) == len(artifacts)


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
