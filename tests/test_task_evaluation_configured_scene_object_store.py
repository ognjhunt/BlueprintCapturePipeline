from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_configured_scene_object_store as store


class _Client:
    def __init__(self, *, corrupt_readback: bool = False) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.corrupt_readback = corrupt_readback

    def upload_file(self, source: str, bucket: str, key: str) -> None:
        self.objects[(bucket, key)] = Path(source).read_bytes()

    def get_object(self, *, Bucket: str, Key: str):
        payload = self.objects[(Bucket, Key)]
        if self.corrupt_readback:
            payload += b"changed"
        return {"Body": io.BytesIO(payload)}


def test_configured_scene_objects_are_content_addressed_and_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "configured_scene.usda"
    source.write_bytes(b"#usda 1.0\n")
    client = _Client()
    monkeypatch.setattr(
        store, "_object_store_client", lambda: (client, "blueprint-inputs")
    )

    publish = store.configured_scene_object_store_publisher()
    result = publish(
        path=source,
        object_name="team-a/scene/revision/configured_scene.usda",
    )

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["uri"] == (
        "s3://blueprint-inputs/blueprint/arm-decision-proof-v1/"
        "configured-scenes/team-a/scene/revision/sha256/"
        f"{digest}/configured_scene.usda"
    )
    assert result["digest"] == "sha256:" + digest
    assert result["full_byte_service_account_readback_passed"] is True
    assert result["content_addressed_key"] is True
    assert result["raw_secret_values_recorded"] is False


def test_configured_scene_publication_refuses_changed_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "configured_scene.usda"
    source.write_bytes(b"#usda 1.0\n")
    monkeypatch.setattr(
        store,
        "_object_store_client",
        lambda: (_Client(corrupt_readback=True), "blueprint-inputs"),
    )

    with pytest.raises(
        store.TaskEvaluationConfiguredSceneObjectStoreError,
        match="configured_scene_object_store_readback_mismatch",
    ):
        store.configured_scene_object_store_publisher()(
            path=source,
            object_name="team-a/scene/revision/configured_scene.usda",
        )


def test_configured_scene_publication_refuses_unsafe_object_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "configured_scene.usda"
    source.write_bytes(b"#usda 1.0\n")
    monkeypatch.setattr(
        store, "_object_store_client", lambda: (_Client(), "blueprint-inputs")
    )

    with pytest.raises(
        store.TaskEvaluationConfiguredSceneObjectStoreError,
        match="configured_scene_object_store_object_name_invalid",
    ):
        store.configured_scene_object_store_publisher()(
            path=source,
            object_name="../other-tenant/configured_scene.usda",
        )


def test_reads_only_an_exact_digest_bound_object_from_the_configured_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"exact-private-thumbnail"
    digest = hashlib.sha256(payload).hexdigest()
    key = (
        "blueprint/arm-decision-proof-v1/configured-scenes/team-a/thumbnail/"
        f"sha256/{digest}/thumbnail.png"
    )
    client = _Client()
    client.objects[("blueprint-inputs", key)] = payload
    monkeypatch.setattr(
        store, "_object_store_client", lambda: (client, "blueprint-inputs")
    )

    assert store.read_configured_scene_object(
        reference={
            "uri": f"s3://blueprint-inputs/{key}",
            "digest": "sha256:" + digest,
            "size_bytes": len(payload),
        }
    ) == payload

    with pytest.raises(
        store.TaskEvaluationConfiguredSceneObjectStoreError,
        match="configured_scene_object_store_read_reference_invalid",
    ):
        store.read_configured_scene_object(
            reference={
                "uri": f"s3://other-team-bucket/{key}",
                "digest": "sha256:" + digest,
                "size_bytes": len(payload),
            }
        )
