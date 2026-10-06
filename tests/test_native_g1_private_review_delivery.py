from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from pathlib import Path
import stat

import pytest

from blueprint_pipeline.live_pipeline_result_artifact_resolution import (
    resolve_live_pipeline_result_artifact,
)
from blueprint_pipeline.native_g1_private_review_delivery import _stage_review_artifacts
from blueprint_pipeline import native_g1_private_review_delivery as delivery
from blueprint_pipeline.task_evaluation_result_delivery import TaskEvaluationResultDeliveryError


def _artifact(root: Path, relative: str, payload: bytes) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "relative_path": relative,
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _review(source: Path) -> dict[str, object]:
    episodes = []
    for index in range(4):
        pair = "manipulation_pair" if index < 2 else "movement_pair"
        prefix = f"{pair}/candidate-{index}/episode"
        episodes.append({
            "frame_manifest": _artifact(
                source, f"{prefix}/frames.json", f"frames-{index}".encode()
            ),
            "review_videos": {
                camera: _artifact(
                    source, f"{prefix}/{camera}.mp4", f"{camera}-{index}".encode()
                ) for camera in ("head", "overview")
            },
        })
    return {
        "status": "verified_private_development_review",
        "claim_ceiling": "development_only",
        "public_redistribution_authorized": False,
        "physical_outcome_claimed": False,
        "review_digest": "sha256:" + "a" * 64,
        "episodes": episodes,
    }


def test_g1_private_media_uses_live_authenticated_artifact_resolver(tmp_path: Path) -> None:
    source = tmp_path / "immutable_execution"
    source.mkdir()
    review = _review(source)
    result_root = tmp_path / "policy-results"
    result_root.mkdir()
    run_id = "g1-841757-test"
    registered = _stage_review_artifacts(
        review=review, source_root=source, result_root=result_root, run_id=run_id
    )
    assert registered["status"] == "registered_private_development_review"
    assert registered["artifact_count"] == 12
    assert registered["public_redistribution_authorized"] is False
    registry_root = result_root / f"{run_id}-activation"
    assert registry_root.is_dir()
    for episode in review["episodes"]:
        for role, artifact in (
            ("g1_frame_manifest", episode["frame_manifest"]),
            ("g1_head_video", episode["review_videos"]["head"]),
            ("g1_overview_video", episode["review_videos"]["overview"]),
        ):
            artifact_id = hashlib.sha256(
                f"{role}\0{artifact['relative_path']}\0{artifact['sha256']}".encode()
            ).hexdigest()[:32]
            path, record = resolve_live_pipeline_result_artifact(
                legacy_state_root=tmp_path / "legacy",
                policy_canary_result_root=result_root,
                run_id=run_id,
                artifact_id=artifact_id,
            )
            assert path.read_bytes() == (source / artifact["relative_path"]).read_bytes()
            assert record["sha256"] == artifact["sha256"]
            assert record["content_type"] == (
                "application/json" if role == "g1_frame_manifest" else "video/mp4"
            )
    assert _stage_review_artifacts(
        review=review, source_root=source, result_root=result_root, run_id=run_id
    ) == registered


def test_g1_private_media_rejects_path_escape_and_changed_bytes(tmp_path: Path) -> None:
    source = tmp_path / "immutable_execution"
    source.mkdir()
    review = _review(source)
    result_root = tmp_path / "policy-results"
    result_root.mkdir()
    review["episodes"][0]["frame_manifest"]["relative_path"] = "../outside.json"
    with pytest.raises(ValueError, match="artifact_path_invalid"):
        _stage_review_artifacts(
            review=review, source_root=source, result_root=result_root, run_id="g1-escape"
        )
    assert not (result_root / "g1-escape-activation").exists()

    review = _review(source)
    first = source / review["episodes"][0]["review_videos"]["head"]["relative_path"]
    first.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact_identity_invalid"):
        _stage_review_artifacts(
            review=review, source_root=source, result_root=result_root, run_id="g1-tamper"
        )
    assert not (result_root / "g1-tamper-activation").exists()


def test_g1_private_media_resolver_detects_post_registration_tamper(tmp_path: Path) -> None:
    source = tmp_path / "immutable_execution"
    source.mkdir()
    review = _review(source)
    result_root = tmp_path / "policy-results"
    result_root.mkdir()
    run_id = "g1-tamper-read"
    _stage_review_artifacts(
        review=review, source_root=source, result_root=result_root, run_id=run_id
    )
    artifact = review["episodes"][0]["frame_manifest"]
    artifact_id = hashlib.sha256(
        f"g1_frame_manifest\0{artifact['relative_path']}\0{artifact['sha256']}".encode()
    ).hexdigest()[:32]
    path = result_root / f"{run_id}-activation/evidence" / artifact["relative_path"]
    path.write_bytes(b"changed")
    with pytest.raises(TaskEvaluationResultDeliveryError, match="reverification_failed"):
        _stage_review_artifacts(
            review=review, source_root=source, result_root=result_root, run_id=run_id
        )
    with pytest.raises(TaskEvaluationResultDeliveryError, match="reverification_failed"):
        resolve_live_pipeline_result_artifact(
            legacy_state_root=tmp_path / "legacy",
            policy_canary_result_root=result_root,
            run_id=run_id,
            artifact_id=artifact_id,
        )


def test_shared_source_and_inherited_group_cannot_expose_private_registration(tmp_path, monkeypatch):
    source, result = tmp_path / 'shared-source', tmp_path / 'shared-result-parent'
    source.mkdir()
    result.mkdir()
    result.chmod(0o2770)
    review = _review(source)
    for path in source.rglob('*'):
        if path.is_file():
            path.chmod(0o666)
    original_source = {path: (path.stat().st_ino, stat.S_IMODE(path.stat().st_mode))
                       for path in source.rglob('*') if path.is_file()}
    monkeypatch.setattr(delivery.os, 'link', lambda *args, **kwargs: pytest.fail('source hardlink forbidden'))
    original_umask = os.umask(0)
    try:
        registration = _stage_review_artifacts(review=review, source_root=source,
            result_root=result, run_id='g1-private-permissions')
    finally:
        os.umask(original_umask)
    target = Path(registration['run_root'])
    for path in [target, *target.rglob('*')]:
        metadata = path.lstat()
        assert metadata.st_uid == os.geteuid()
        assert stat.S_IMODE(metadata.st_mode) == (0o700 if path.is_dir() else 0o600)
        # These real permission bits deny all non-owner synthetic identities,
        # including an outsider who belongs to the inherited result-root GID.
        assert metadata.st_mode & 0o077 == 0
        if path.is_file():
            assert metadata.st_nlink == 1
    for path, identity in original_source.items():
        assert (path.stat().st_ino, stat.S_IMODE(path.stat().st_mode)) == identity
        destination = target / 'evidence' / path.relative_to(source)
        assert destination.read_bytes() == path.read_bytes()
        assert (destination.stat().st_dev, destination.stat().st_ino) != (path.stat().st_dev, path.stat().st_ino)
    assert _stage_review_artifacts(review=review, source_root=source,
        result_root=result, run_id='g1-private-permissions') == registration


def test_registration_has_independent_bytes_when_shared_source_is_modified(tmp_path):
    source, result = tmp_path / 'source', tmp_path / 'result'
    source.mkdir()
    result.mkdir()
    review = _review(source)
    artifact = review['episodes'][0]['frame_manifest']
    original = source / artifact['relative_path']
    original.chmod(0o666)
    registration = _stage_review_artifacts(review=review, source_root=source,
        result_root=result, run_id='g1-independent-copy')
    retained = Path(registration['run_root']) / 'evidence' / artifact['relative_path']
    expected = retained.read_bytes()
    original.write_bytes(b'changed by synthetic shared-source writer')
    artifact_id = hashlib.sha256(f"g1_frame_manifest\0{artifact['relative_path']}\0{artifact['sha256']}".encode()).hexdigest()[:32]
    path, record = resolve_live_pipeline_result_artifact(legacy_state_root=tmp_path / 'legacy',
        policy_canary_result_root=result, run_id='g1-independent-copy', artifact_id=artifact_id)
    assert path.read_bytes() == expected
    assert record['sha256'] == artifact['sha256']
    assert retained.read_bytes() == expected
    assert stat.S_IMODE(original.stat().st_mode) == 0o666


@pytest.mark.parametrize('change', ['run_directory', 'nested_directory', 'registry_file',
    'artifact_file', 'hardlinked_artifact', 'hardlinked_registry'])
def test_reopening_does_not_endorse_unadmitted_legacy_readership(tmp_path, change):
    source, result = tmp_path / 'source', tmp_path / 'result'
    source.mkdir()
    result.mkdir()
    review = _review(source)
    registration = _stage_review_artifacts(review=review, source_root=source,
        result_root=result, run_id='g1-legacy-policy')
    target = Path(registration['run_root'])
    artifact = target / 'evidence' / review['episodes'][0]['frame_manifest']['relative_path']
    registry = target / 'artifacts/result_delivery/artifact_registry.json'
    changed = {'run_directory': target, 'nested_directory': artifact.parent,
        'registry_file': registry, 'artifact_file': artifact}.get(change)
    if changed is not None:
        changed.chmod(0o750 if changed.is_dir() else 0o640)
    else:
        changed = artifact if change == 'hardlinked_artifact' else registry
        os.link(changed, result / 'unadmitted-reader-alias')
    identity = changed.stat()
    with pytest.raises(ValueError, match='private_permissions_invalid'):
        _stage_review_artifacts(review=review, source_root=source,
            result_root=result, run_id='g1-legacy-policy')
    assert changed.stat() == identity  # No implicit chmod, copy migration or deletion.


def test_non_owner_reopen_is_refused_before_any_registered_artifact_read(tmp_path, monkeypatch):
    source, result = tmp_path / 'source', tmp_path / 'result'
    source.mkdir()
    result.mkdir()
    review = _review(source)
    _stage_review_artifacts(review=review, source_root=source, result_root=result, run_id='g1-owner')
    owner = os.geteuid()
    monkeypatch.setattr(delivery.os, 'geteuid', lambda: owner + 10000)
    monkeypatch.setattr(delivery, '_read', lambda *args, **kwargs: pytest.fail('outsider registry read'))
    monkeypatch.setattr(delivery, '_sha256', lambda *args, **kwargs: pytest.fail('outsider media read'))
    with pytest.raises(ValueError, match='private_permissions_invalid'):
        _stage_review_artifacts(review=review, source_root=source, result_root=result, run_id='g1-owner')


def test_copy_refuses_changed_source_before_publication(tmp_path, monkeypatch):
    source, result = tmp_path / 'source', tmp_path / 'result'
    source.mkdir()
    result.mkdir()
    review = _review(source)
    original = delivery._private_copy
    def change_then_copy(src, destination, *, size):
        src.write_bytes(b'x' * size)
        original(src, destination, size=size)
    monkeypatch.setattr(delivery, '_private_copy', change_then_copy)
    with pytest.raises(ValueError, match='staged_identity_invalid'):
        _stage_review_artifacts(review=review, source_root=source,
            result_root=result, run_id='g1-copy-race')
    assert list(result.iterdir()) == []


def test_source_growth_during_copy_never_writes_beyond_admitted_size(tmp_path, monkeypatch):
    source, result = tmp_path / 'source', tmp_path / 'result'
    source.mkdir()
    result.mkdir()
    review = _review(source)
    first = review['episodes'][0]['frame_manifest']
    original_writer = delivery._private_writer
    written = []

    @contextmanager
    def grow_source(path):
        with original_writer(path) as output:
            with (source / first['relative_path']).open('ab') as stream:
                stream.write(b'unadmitted bytes' * 100)
            try:
                yield output
            finally:
                output.flush()
                written.append(os.fstat(output.fileno()).st_size)

    monkeypatch.setattr(delivery, '_private_writer', grow_source)
    with pytest.raises(ValueError, match='staged_identity_invalid'):
        _stage_review_artifacts(review=review, source_root=source,
            result_root=result, run_id='g1-growth-race')
    assert written == [first['size_bytes']]
    assert list(result.iterdir()) == []
