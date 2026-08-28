from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path
from types import SimpleNamespace

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


class _ContentAddressedClient(_Client):
    def __init__(self, *, corrupt_readback: bool = False) -> None:
        super().__init__(corrupt_readback=corrupt_readback)
        self.metadata: dict[tuple[str, str], dict[str, str]] = {}
        self.upload_count = 0

    def upload_file(
        self,
        source: str,
        bucket: str,
        key: str,
        ExtraArgs: dict[str, object] | None = None,
    ) -> None:
        self.upload_count += 1
        super().upload_file(source, bucket, key)
        extra = ExtraArgs or {}
        self.metadata[(bucket, key)] = dict(extra.get("Metadata") or {})

    def head_object(self, *, Bucket: str, Key: str):
        payload = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(payload),
            "Metadata": self.metadata[(Bucket, Key)],
        }


def test_local_readiness_constructs_client_without_object_store_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _Client()
    monkeypatch.setattr(
        store, "_object_store_client", lambda: (client, "blueprint-inputs")
    )

    result = store.validate_configured_scene_object_store_configuration()

    assert result == {
        "schema_version": (
            "task_evaluation_configured_scene_object_store_readiness.v1"
        ),
        "status": "locally_configured",
        "key_prefix": store.DEFAULT_KEY_PREFIX,
        "credential_files_validated": True,
        "client_constructed": True,
        "remote_bucket_authority_verified": False,
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
    }
    assert client.objects == {}


def test_partial_dedicated_artifact_store_never_falls_back_to_staging_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    access = tmp_path / "b2-access"
    access.write_text("dedicated-access\n", encoding="utf-8")
    access.chmod(0o600)
    monkeypatch.setenv(
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ACCESS_KEY_ID_FILE",
        str(access),
    )
    for field, value in (
        ("ACCESS_KEY_ID", "legacy-access"),
        ("SECRET_ACCESS_KEY", "legacy-secret"),
        ("BUCKET", "legacy-bucket"),
    ):
        path = tmp_path / f"legacy-{field.lower()}"
        path.write_text(value + "\n", encoding="utf-8")
        path.chmod(0o600)
        monkeypatch.setenv(f"BLUEPRINT_WAM_OBJECT_STORE_{field}_FILE", str(path))

    with pytest.raises(
        store.TaskEvaluationConfiguredSceneObjectStoreError,
        match=(
            "configured_scene_object_store_configuration_missing:"
            "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE"
        ),
    ):
        store._artifact_object_store_client()


def test_dedicated_artifact_store_presigns_exact_verified_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = {
        "ACCESS_KEY_ID": "test-access",
        "SECRET_ACCESS_KEY": "test-secret",
        "BUCKET": "scene-artifacts",
        "ENDPOINT_URL": "https://s3.us-west-004.backblazeb2.com",
        "REGION": "us-west-004",
    }
    for field, value in values.items():
        path = tmp_path / field.lower()
        path.write_text(value + "\n", encoding="utf-8")
        path.chmod(0o600)
        monkeypatch.setenv(
            f"BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_{field}_FILE",
            str(path),
        )

    calls: list[tuple[str, dict[str, object]]] = []

    class FakeConfig:
        def __init__(self, **_kwargs):
            pass

    class FakeClient:
        def generate_presigned_url(self, operation: str, **kwargs):
            calls.append((operation, kwargs))
            return "https://s3.us-west-004.backblazeb2.com/signed?redacted"

    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda _service, **_kwargs: FakeClient()),
    )
    monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "botocore.client", SimpleNamespace(Config=FakeConfig)
    )
    digest = "a" * 64
    key = (
        f"{store.LARGE_ARTIFACT_KEY_PREFIX}/provider-bundle/sha256/"
        f"{digest}/provider-bundle.zip"
    )
    url = store.presign_configured_scene_artifact(
        reference={
            "schema_version": "task_evaluation_scene_artifact_reference.v1",
            "status": "remote_verified",
            "artifact_kind": "provider-bundle",
            "uri": f"s3://scene-artifacts/{key}",
            "digest": "sha256:" + digest,
            "size_bytes": 123,
            "content_addressed_key": True,
            "remote_identity_verified": True,
            "full_byte_service_account_readback_passed": True,
        },
        expiration_seconds=600,
    )

    assert url.startswith("https://s3.us-west-004.backblazeb2.com/")
    assert calls == [
        (
            "get_object",
            {
                "Params": {
                    "Bucket": "scene-artifacts",
                    "Key": key,
                    "ResponseCacheControl": "no-store, max-age=0",
                },
                "ExpiresIn": 600,
                "HttpMethod": "GET",
            },
        )
    ]


def test_dedicated_artifact_store_does_not_reroute_configured_scene_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for prefix, values in (
        (
            "BLUEPRINT_WAM_OBJECT_STORE",
            {
                "ACCESS_KEY_ID": "spaces-access",
                "SECRET_ACCESS_KEY": "spaces-secret",
                "BUCKET": "configured-scenes",
                "ENDPOINT_URL": "https://spaces.example.test",
                "REGION": "spaces-region",
            },
        ),
        (
            "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE",
            {
                "ACCESS_KEY_ID": "b2-access",
                "SECRET_ACCESS_KEY": "b2-secret",
                "BUCKET": "large-artifacts",
                "ENDPOINT_URL": "https://b2.example.test",
                "REGION": "b2-region",
            },
        ),
    ):
        for field, value in values.items():
            path = tmp_path / f"{prefix.lower()}-{field.lower()}"
            path.write_text(value + "\n", encoding="utf-8")
            path.chmod(0o600)
            monkeypatch.setenv(f"{prefix}_{field}_FILE", str(path))

    client_calls: list[dict[str, object]] = []

    class FakeConfig:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(
            client=lambda _service, **kwargs: (
                client_calls.append(kwargs) or object()
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "botocore.client", SimpleNamespace(Config=FakeConfig)
    )

    _scene_client, scene_bucket = store._object_store_client()
    _artifact_client, artifact_bucket = store._artifact_object_store_client()

    assert scene_bucket == "configured-scenes"
    assert artifact_bucket == "large-artifacts"
    assert [call["endpoint_url"] for call in client_calls] == [
        "https://spaces.example.test",
        "https://b2.example.test",
    ]
    assert [call["aws_access_key_id"] for call in client_calls] == [
        "spaces-access",
        "b2-access",
    ]


def test_production_artifact_store_refuses_wrong_bucket_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = {
        "ACCESS_KEY_ID": "test-access",
        "SECRET_ACCESS_KEY": "test-secret",
        "BUCKET": "blueprint-task-evaluation-artifacts-dev",
        "ENDPOINT_URL": "https://s3.us-east-005.backblazeb2.com",
        "REGION": "us-east-005",
    }
    for field, value in values.items():
        path = tmp_path / field.lower()
        path.write_text(value + "\n", encoding="utf-8")
        path.chmod(0o600)
        monkeypatch.setenv(
            f"BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_{field}_FILE",
            str(path),
        )
    monkeypatch.setenv(
        "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET",
        "blueprint-task-evaluation-artifacts-prod",
    )
    monkeypatch.setattr(
        store,
        "_client_from_file_environment",
        lambda *_args, **_kwargs: (object(), values["BUCKET"]),
    )

    with pytest.raises(
        store.TaskEvaluationConfiguredSceneObjectStoreError,
        match="configured_scene_artifact_store_bucket_identity_mismatch",
    ):
        store._artifact_object_store_client()


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


def test_large_artifact_publication_reuses_verified_content_address(
    tmp_path: Path,
) -> None:
    source = tmp_path / "provider-output.zip"
    source.write_bytes((b"scene-output\n" * 4096) + b"terminal")
    client = _ContentAddressedClient()

    first = store.publish_configured_scene_artifact(
        path=source,
        artifact_kind="provider-output",
        client=client,
        bucket="blueprint-inputs",
    )
    second = store.publish_configured_scene_artifact(
        path=source,
        artifact_kind="provider-output",
        client=client,
        bucket="blueprint-inputs",
    )

    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert first["uri"].endswith(
        f"/artifacts/provider-output/sha256/{digest}/provider-output.zip"
    )
    assert first["upload_performed"] is True
    assert first["cache_hit"] is False
    assert second["upload_performed"] is False
    assert second["cache_hit"] is True
    assert second["remote_identity_verified"] is True
    assert second["full_byte_service_account_readback_passed"] is True
    assert client.upload_count == 1


def test_large_artifact_publication_refuses_existing_identity_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "provider-bundle.zip"
    source.write_bytes(b"exact bundle")
    client = _ContentAddressedClient()
    reference = store.publish_configured_scene_artifact(
        path=source,
        artifact_kind="provider-bundle",
        client=client,
        bucket="blueprint-inputs",
    )
    key = reference["uri"].split("blueprint-inputs/", 1)[1]
    client.metadata[("blueprint-inputs", key)]["sha256"] = "0" * 64

    with pytest.raises(
        store.TaskEvaluationConfiguredSceneObjectStoreError,
        match="configured_scene_artifact_existing_identity_mismatch",
    ):
        store.publish_configured_scene_artifact(
            path=source,
            artifact_kind="provider-bundle",
            client=client,
            bucket="blueprint-inputs",
        )


def test_large_artifact_materialization_streams_and_verifies_before_exposure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "checkpoint.zip"
    source.write_bytes(b"checkpoint" * 1_000_000)
    client = _ContentAddressedClient()
    reference = store.publish_configured_scene_artifact(
        path=source,
        artifact_kind="diagnostic-checkpoint",
        client=client,
        bucket="blueprint-inputs",
    )
    destination = tmp_path / "materialized" / "checkpoint.zip"

    result = store.materialize_configured_scene_artifact(
        reference=reference,
        destination=destination,
        maximum_size_bytes=source.stat().st_size,
        client=client,
        bucket="blueprint-inputs",
    )

    assert result["status"] == "completed"
    assert result["local_full_byte_readback_passed"] is True
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_mode & 0o777 == 0o440
    assert not list(destination.parent.glob("*.partial"))


def test_large_artifact_materialization_removes_corrupt_partial(
    tmp_path: Path,
) -> None:
    source = tmp_path / "checkpoint.zip"
    source.write_bytes(b"checkpoint")
    client = _ContentAddressedClient()
    reference = store.publish_configured_scene_artifact(
        path=source,
        artifact_kind="diagnostic-checkpoint",
        client=client,
        bucket="blueprint-inputs",
    )
    client.corrupt_readback = True
    destination = tmp_path / "materialized" / "checkpoint.zip"

    with pytest.raises(
        store.TaskEvaluationConfiguredSceneObjectStoreError,
        match="configured_scene_artifact_materialization_exceeds_limit",
    ):
        store.materialize_configured_scene_artifact(
            reference=reference,
            destination=destination,
            maximum_size_bytes=source.stat().st_size,
            client=client,
            bucket="blueprint-inputs",
        )

    assert not destination.exists()
    assert not list(destination.parent.glob("*.partial"))


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
