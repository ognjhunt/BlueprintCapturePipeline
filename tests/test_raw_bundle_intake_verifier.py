from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Callable

import pytest

from blueprint_pipeline import materialization
from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.ios_manifest import verify_canonical_raw_bundle_path
from blueprint_pipeline.object_index_stage import run_object_index_stage


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _bundle_sha(artifacts: dict[str, str]) -> str:
    canonical = "\n".join(f"{name}:{artifacts[name]}" for name in sorted(artifacts))
    return sha256(canonical.encode("utf-8")).hexdigest()


def _rehash(raw: Path, *, transform: Callable[[dict[str, str]], dict[str, str]] | None = None) -> None:
    artifacts = {
        path.relative_to(raw).as_posix(): _file_sha(path)
        for path in sorted(raw.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name != "hashes.json"
    }
    if transform is not None:
        artifacts = transform(artifacts)
    _write_json(
        raw / "hashes.json",
        {
            "schema_version": "v1",
            "bundle_sha256": _bundle_sha(artifacts),
            "artifacts": artifacts,
        },
    )


def _v3_capture(tmp_path: Path) -> tuple[Path, Path]:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    raw = capture_root / "raw"
    raw.mkdir(parents=True)
    _write_json(
        raw / "manifest.json",
        {
            "schema_version": "v3",
            "capture_schema_version": "3.0.0",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "video_uri": "walkthrough.mov",
            "capture_source": "fixture",
            "capture_tier_hint": "test_fixture",
            "capture_profile_id": "fixture_video_only",
            "capture_capabilities": {},
            "coordinate_frame_session_id": "cfs-fixture-1",
            "capture_start_epoch_ms": 1_700_000_000_000,
            "app_version": "1.0.0",
            "app_build": "1",
            "ios_version": "18.0",
            "ios_build": "22A",
            "hardware_model_identifier": "FixtureDevice1,1",
            "device_model_marketing": "Fixture Device",
            "has_lidar": False,
            "depth_supported": False,
            "fps_source": 30.0,
            "width": 1920,
            "height": 1080,
            "rights_profile": "test_fixture",
            "requested_outputs": ["qualification"],
        },
    )
    _write_json(
        raw / "capture_context.json",
        {"schema_version": "v1", "scene_id": "scene-1", "capture_id": "capture-1"},
    )
    _write_json(
        raw / "capture_upload_complete.json",
        {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "raw_prefix": "scenes/scene-1/captures/capture-1/raw",
            "completed_at": "2026-07-09T12:00:00Z",
            "status": "complete",
        },
    )
    _write_json(
        raw / "intake_packet.json",
        {
            "workflowName": "Inspect aisle",
            "taskSteps": ["walk aisle"],
            "zone": "aisle",
            "owner": "ops",
        },
    )
    (raw / "walkthrough.mov").write_bytes(b"video-bytes")
    _rehash(raw)
    return capture_root, raw


def _raw_snapshot(raw: Path) -> dict[str, tuple[str, bytes | str]]:
    snapshot: dict[str, tuple[str, bytes | str]] = {}
    for path in sorted(raw.rglob("*")):
        relative = path.relative_to(raw).as_posix()
        if path.is_symlink():
            snapshot[relative] = ("symlink", os.readlink(path))
        elif path.is_file():
            snapshot[relative] = ("file", path.read_bytes())
    return snapshot


def _assert_quarantined_without_derived(capture_root: Path) -> None:
    with pytest.raises(PipelineError, match="raw_bundle_quarantined"):
        materialization.materialize_capture_bundle(
            bucket="bucket",
            scene_id="scene-1",
            capture_id="capture-1",
            gcs_root=capture_root.parents[3],
        )
    assert not (capture_root / "capture_descriptor.json").exists()
    assert not (capture_root / "qa_report.json").exists()
    assert not (capture_root / "frames").exists()
    assert not (capture_root / "pipeline").exists()
    quarantine_records = list((capture_root / "quarantine" / "raw_intake").glob("*.json"))
    assert quarantine_records
    assert json.loads(quarantine_records[-1].read_text(encoding="utf-8"))["status"] == "quarantined"


def test_current_v3_verification_persists_digest_and_materialization_preserves_raw_bytes(
    tmp_path: Path,
) -> None:
    capture_root, raw = _v3_capture(tmp_path)
    before = _raw_snapshot(raw)

    report = verify_canonical_raw_bundle_path(
        raw,
        expected_scene_id="scene-1",
        expected_capture_id="capture-1",
    )
    assert report["status"] == "verified"
    assert report["valid_for_derivation"] is True
    assert report["intake_digest"] == report["hash_verification"]["bundle_sha256_actual"]
    assert report["hash_verification"]["total_size_bytes"] > 0

    materialization.materialize_capture_bundle(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        gcs_root=tmp_path,
    )

    assert _raw_snapshot(raw) == before
    pointer = json.loads(
        (capture_root / "pipeline" / "intake" / "current.json").read_text(encoding="utf-8")
    )
    assert pointer["status"] == "verified"
    assert pointer["intake_digest"] == report["intake_digest"]


@pytest.mark.parametrize(
    "corrupt,expected_reason",
    [
        (
            lambda raw: (raw / "walkthrough.mov").write_bytes(b"tampered"),
            "hash_mismatch:walkthrough.mov",
        ),
        (
            lambda raw: _rehash(
                raw,
                transform=lambda artifacts: {
                    name: digest for name, digest in artifacts.items() if name != "walkthrough.mov"
                },
            ),
            "hash_coverage_missing:walkthrough.mov",
        ),
        (
            lambda raw: (
                (raw / "capture_context.json").write_text("{malformed", encoding="utf-8"),
                _rehash(raw),
            ),
            "malformed_sidecar:capture_context.json",
        ),
        (
            lambda raw: (
                _write_json(
                    raw / "capture_upload_complete.json",
                    {
                        "schema_version": "v1",
                        "scene_id": "scene-1",
                        "capture_id": "capture-1",
                        "raw_prefix": "scenes/scene-1/captures/capture-1/raw",
                        "completed_at": "2026-07-09T12:00:00Z",
                        "status": "uploading",
                    },
                ),
                _rehash(raw),
            ),
            "upload_not_complete",
        ),
        (
            lambda raw: _rehash(
                raw,
                transform=lambda artifacts: {**artifacts, "/etc/passwd": "0" * 64},
            ),
            "invalid_hash_path:/etc/passwd:path_escape",
        ),
        (
            lambda raw: _rehash(
                raw,
                transform=lambda artifacts: {**artifacts, "../outside": "0" * 64},
            ),
            "invalid_hash_path:../outside:path_escape",
        ),
        (
            lambda raw: (
                _write_json(raw / "object_index_build_report.json", {"status": "built"}),
                _rehash(raw),
            ),
            "pipeline_derivative_forbidden_in_raw:object_index_build_report.json",
        ),
        (
            lambda raw: (
                _write_json(
                    raw / "manifest.json",
                    {
                        key: value
                        for key, value in json.loads(
                            (raw / "manifest.json").read_text(encoding="utf-8")
                        ).items()
                        if key != "hardware_model_identifier"
                    },
                ),
                _rehash(raw),
            ),
            "manifest_missing_field:hardware_model_identifier",
        ),
    ],
)
def test_current_v3_failures_are_typed_quarantine_with_zero_derived_writes(
    tmp_path: Path,
    corrupt: Callable[[Path], object],
    expected_reason: str,
) -> None:
    capture_root, raw = _v3_capture(tmp_path)
    corrupt(raw)
    report = verify_canonical_raw_bundle_path(
        raw,
        expected_scene_id="scene-1",
        expected_capture_id="capture-1",
    )
    assert report["status"] == "quarantined"
    assert any(reason.startswith(expected_reason) for reason in report["quarantine_reasons"])
    _assert_quarantined_without_derived(capture_root)


def test_current_v3_symlink_is_quarantined_without_following_target(tmp_path: Path) -> None:
    capture_root, raw = _v3_capture(tmp_path)
    outside = tmp_path / "outside-secret"
    outside.write_bytes(b"outside")
    (raw / "linked.bin").symlink_to(outside)
    hashes = json.loads((raw / "hashes.json").read_text(encoding="utf-8"))
    hashes["artifacts"]["linked.bin"] = _file_sha(outside)
    hashes["bundle_sha256"] = _bundle_sha(hashes["artifacts"])
    _write_json(raw / "hashes.json", hashes)

    report = verify_canonical_raw_bundle_path(
        raw,
        expected_scene_id="scene-1",
        expected_capture_id="capture-1",
    )
    assert "raw_symlink_forbidden:linked.bin" in report["quarantine_reasons"]
    assert "invalid_hash_path:linked.bin:symlink" in report["quarantine_reasons"]
    _assert_quarantined_without_derived(capture_root)


def test_raw_change_during_materialization_is_quarantined_before_product_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root, raw = _v3_capture(tmp_path)
    original = materialization._discover_raw_sidecars

    def mutate_after_verification(**kwargs):  # type: ignore[no-untyped-def]
        result = original(**kwargs)
        _write_json(
            raw / "rights_consent.json",
            {"scene_id": "scene-1", "capture_id": "capture-1", "consent_status": "revoked"},
        )
        return result

    monkeypatch.setattr(materialization, "_discover_raw_sidecars", mutate_after_verification)
    with pytest.raises(PipelineError, match="raw_bundle_changed_during_materialization"):
        materialization.materialize_capture_bundle(
            bucket="bucket",
            scene_id="scene-1",
            capture_id="capture-1",
            gcs_root=tmp_path,
        )
    assert not (capture_root / "capture_descriptor.json").exists()
    assert not (capture_root / "qa_report.json").exists()
    assert not (capture_root / "frames").exists()
    assert list((capture_root / "quarantine" / "raw_intake").glob("*.json"))


def test_changed_rights_with_new_valid_hashes_cannot_reuse_persisted_intake_identity(
    tmp_path: Path,
) -> None:
    capture_root, raw = _v3_capture(tmp_path)
    materialization.materialize_capture_bundle(
        bucket="bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        gcs_root=tmp_path,
    )
    original_pointer = json.loads(
        (capture_root / "pipeline" / "intake" / "current.json").read_text(encoding="utf-8")
    )
    _write_json(
        raw / "rights_consent.json",
        {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "consent_status": "revoked",
        },
    )
    _rehash(raw)
    assert verify_canonical_raw_bundle_path(
        raw,
        expected_scene_id="scene-1",
        expected_capture_id="capture-1",
    )["status"] == "verified"

    with pytest.raises(PipelineError, match="immutable_intake_digest_changed"):
        run_object_index_stage(capture_root=capture_root)

    assert not (capture_root / "pipeline" / "derived" / "object_index").exists()
    assert json.loads(
        (capture_root / "pipeline" / "intake" / "current.json").read_text(encoding="utf-8")
    ) == original_pointer


def test_legacy_intake_is_explicitly_degraded_not_current_integrity_proof(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_json(raw / "manifest.json", {"scene_id": "legacy", "video_uri": "walkthrough.mov"})
    (raw / "walkthrough.mov").write_bytes(b"legacy")

    report = verify_canonical_raw_bundle_path(raw)

    assert report["status"] == "legacy_degraded"
    assert report["valid_for_derivation"] is True
    assert report["current_schema"] is False
    assert "not_public_launch_proof" in report["claim_boundary"]
