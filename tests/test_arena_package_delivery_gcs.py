"""Tests for the GCS Arena package delivery producer (audit R002)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import arena_package_delivery_gcs as gcs


class _FakeBlob:
    def __init__(self, store: dict, key: str) -> None:
        self._store = store
        self._key = key

    def upload_from_filename(self, filename: str) -> None:
        self._store[self._key] = Path(filename).read_bytes()


class _FakeBucket:
    def __init__(self, store: dict, name: str) -> None:
        self._store = store
        self.name = name

    def blob(self, key: str) -> _FakeBlob:
        return _FakeBlob(self._store, key)


class _FakeClient:
    def __init__(self) -> None:
        self.store: dict = {}

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(self.store, name)


class _RaisingClient:
    def bucket(self, name: str):  # noqa: ANN001
        raise RuntimeError("boom")


def _make_bundle(root: Path) -> None:
    bundle = root / "delivery_bundle"
    (bundle / "clips").mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    (bundle / "clips" / "clip_0.mp4").write_bytes(b"video-bytes")


def test_uploads_bundle_and_records_gs_uris(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gcs.GATE_ENV, "true")
    _make_bundle(tmp_path)
    client = _FakeClient()

    manifest = gcs.build_gcs_delivery_command_manifest(
        output_dir=tmp_path,
        entitlement_id="ent_123",
        destination_bucket="blueprint-delivery",
        storage_client=client,
    )

    assert manifest["storage_upload_performed"] is True
    assert manifest["blockers"] == []
    assert manifest["provider"] == "gcs"
    assert manifest["object_count"] == 2
    keys = {obj["object_key"] for obj in manifest["objects"]}
    assert keys == {
        "marketplace-artifacts/ent_123/manifest.json",
        "marketplace-artifacts/ent_123/clips/clip_0.mp4",
    }
    for obj in manifest["objects"]:
        assert obj["gs_uri"] == f"gs://blueprint-delivery/{obj['object_key']}"
        assert len(obj["sha256"]) == 64
    assert manifest["delivery_base_uri"] == "gs://blueprint-delivery/marketplace-artifacts/ent_123"
    # WebApp ingestion contract carries the entitlement + source + object keys.
    ingestion = manifest["webapp_ingestion"]
    assert ingestion["entitlement_id"] == "ent_123"
    assert ingestion["requires_webapp_entitlement_and_consent_check"] is True
    assert set(ingestion["object_keys"]) == keys
    # Objects were actually written through the (fake) client.
    assert set(client.store) == keys
    # Manifest persisted for the WebApp/ingest to consume.
    written = json.loads((tmp_path / gcs.OUTPUT_FILENAME).read_text())
    assert written["storage_upload_performed"] is True
    # Conservative claim boundary preserved.
    assert manifest["public_claim_upgrade_allowed"] is False
    assert manifest["claim_boundary"]["deployment_approval_proven"] is False


def test_fail_closed_without_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(gcs.GATE_ENV, raising=False)
    _make_bundle(tmp_path)
    manifest = gcs.build_gcs_delivery_command_manifest(
        output_dir=tmp_path,
        entitlement_id="ent_123",
        destination_bucket="blueprint-delivery",
        storage_client=_FakeClient(),
    )
    assert manifest["storage_upload_performed"] is False
    assert manifest["status"] == "blocked"
    assert f"missing_env_{gcs.GATE_ENV}" in manifest["blockers"]
    assert manifest["objects"] == []


def test_missing_entitlement_and_bucket_block(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gcs.GATE_ENV, "true")
    monkeypatch.delenv(gcs.BUCKET_ENV, raising=False)
    _make_bundle(tmp_path)
    manifest = gcs.build_gcs_delivery_command_manifest(output_dir=tmp_path, storage_client=_FakeClient())
    assert manifest["storage_upload_performed"] is False
    assert f"missing_env_{gcs.BUCKET_ENV}" in manifest["blockers"]
    assert "missing_entitlement_id" in manifest["blockers"]


def test_upload_failure_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(gcs.GATE_ENV, "true")
    _make_bundle(tmp_path)
    manifest = gcs.build_gcs_delivery_command_manifest(
        output_dir=tmp_path,
        entitlement_id="ent_123",
        destination_bucket="blueprint-delivery",
        storage_client=_RaisingClient(),
    )
    assert manifest["storage_upload_performed"] is False
    assert "storage_upload_failed" in manifest["blockers"]
    assert manifest["upload_error"] is not None
    assert manifest["objects"] == []
