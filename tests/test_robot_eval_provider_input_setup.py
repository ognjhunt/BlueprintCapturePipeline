from __future__ import annotations

import hashlib
import io
import json
import sys
import types
from pathlib import Path

import pytest

from blueprint_pipeline import robot_eval_provider_input_setup as setup


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_upload_helpers_cover_supported_schemes_and_error_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")

    assert setup._string("text") == "text"
    assert setup._string(123) == ""

    uploaded: dict[str, object] = {}

    class FakeBlob:
        def upload_from_filename(self, filename: str) -> None:
            uploaded["gs_filename"] = filename

    class FakeBucket:
        def blob(self, key: str) -> FakeBlob:
            uploaded["gs_key"] = key
            return FakeBlob()

    class FakeStorageClient:
        def bucket(self, bucket: str) -> FakeBucket:
            uploaded["gs_bucket"] = bucket
            return FakeBucket()

    storage_module = types.SimpleNamespace(Client=lambda: FakeStorageClient())
    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    cloud_module.storage = storage_module
    google_module.cloud = cloud_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)
    assert setup._upload_file_to_gs(source, "gs://bucket/path/source.txt")["status"] == "uploaded"
    assert uploaded == {
        "gs_bucket": "bucket",
        "gs_key": "path/source.txt",
        "gs_filename": str(source),
    }
    monkeypatch.setattr(
        setup,
        "_upload_file_to_gs",
        lambda path, destination_uri: {
            "status": "uploaded",
            "source": str(path),
            "destination_uri": destination_uri,
            "storage_scheme": "gs",
        },
    )
    assert setup.upload_file(source, "gs://bucket/path/source.txt")["storage_scheme"] == "gs"

    class FakeBoto3:
        def client(self, service: str, **kwargs: object) -> object:
            uploaded["s3_service"] = service
            uploaded["s3_kwargs"] = kwargs

            class Client:
                def upload_file(self, filename: str, bucket: str, key: str) -> None:
                    uploaded["s3_upload"] = (filename, bucket, key)

            return Client()

    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3())
    monkeypatch.setenv("BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL", "https://r2.example")
    monkeypatch.setenv("AWS_REGION", "auto")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fixture-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fixture-secret")
    r2_upload = setup._upload_file_to_s3_compatible(source, "r2://bucket/path/source.txt")
    assert r2_upload["storage_scheme"] == "r2"
    assert uploaded["s3_kwargs"] == {
        "endpoint_url": "https://r2.example",
        "region_name": "auto",
        "aws_access_key_id": "fixture-access",
        "aws_secret_access_key": "fixture-secret",
    }
    monkeypatch.delenv("BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    access_file = tmp_path / "access-key-id"
    secret_file = tmp_path / "secret-access-key"
    endpoint_file = tmp_path / "spaces-endpoint"
    region_file = tmp_path / "spaces-region"
    access_file.write_text("file-access-key", encoding="utf-8")
    secret_file.write_text("file-secret-key", encoding="utf-8")
    endpoint_file.write_text("https://file-r2.example", encoding="utf-8")
    region_file.write_text("auto", encoding="utf-8")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID_FILE", str(access_file))
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY_FILE", str(secret_file))
    monkeypatch.setenv("BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL_FILE", str(endpoint_file))
    monkeypatch.setenv("AWS_REGION_FILE", str(region_file))

    file_r2_upload = setup._upload_file_to_s3_compatible(
        source,
        "r2://bucket/path/source.txt",
    )
    assert file_r2_upload["storage_scheme"] == "r2"
    assert file_r2_upload["secret_values_recorded"] is False
    assert file_r2_upload["file_based_secret_env_vars_present"] == [
        "AWS_ACCESS_KEY_ID_FILE",
        "AWS_SECRET_ACCESS_KEY_FILE",
    ]
    assert uploaded["s3_kwargs"] == {
        "endpoint_url": "https://file-r2.example",
        "region_name": "auto",
        "aws_access_key_id": "file-access-key",
        "aws_secret_access_key": "file-secret-key",
    }
    file_destination = tmp_path / "copied.txt"
    assert setup.upload_file(source, str(file_destination))["storage_scheme"] == "file"
    assert file_destination.read_text(encoding="utf-8") == "payload"
    assert setup.upload_file(source, "ftp://example/source.txt")["blockers"] == [
        "unsupported_upload_uri_scheme:ftp"
    ]

    def denied_upload(
        path: Path, destination_uri: str, *, exclusive: bool = False
    ) -> dict[str, object]:
        raise RuntimeError("AccessDenied")

    monkeypatch.setattr(setup, "_upload_file_to_s3_compatible", denied_upload)
    blocked = setup.upload_file(source, "s3://bucket/source.txt")
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == ["upload_failed:s3_access_denied"]
    assert setup._classify_upload_error(
        scheme="gs",
        error=RuntimeError("Billing account absent"),
    ) == "upload_failed:gs_billing_account_disabled"
    assert setup._classify_upload_error(scheme="s3", error=RuntimeError("403")) == (
        "upload_failed:s3_forbidden"
    )
    assert setup._classify_upload_error(scheme="file", error=OSError("disk full")) == (
        "upload_failed:OSError"
    )


def test_s3_validation_streams_full_bytes_with_bare_internal_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL", "https://nyc3.digitaloceanspaces.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fixture-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fixture-secret")
    source = tmp_path / "source.bin"
    payload = b"a" * (1024 * 1024 + 17)
    source.write_bytes(payload)
    observed: dict[str, object] = {}

    class Client:
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            observed.update(bucket=Bucket, key=Key)
            return {"Body": io.BytesIO(payload)}

    class Boto3:
        @staticmethod
        def client(service: str, **kwargs: object) -> Client:
            observed.update(service=service, kwargs=kwargs)
            return Client()

    monkeypatch.setitem(sys.modules, "boto3", Boto3())
    validation = setup._validate_uploaded_object(
        source, "s3://fixture-bucket/path/source.bin"
    )

    digest = hashlib.sha256(payload).hexdigest()
    assert validation["status"] == "validated"
    assert validation["object_sha256"] == digest
    assert validation["expected_sha256"] == digest
    assert validation["full_byte_readback_performed"] is True
    assert observed["bucket"] == "fixture-bucket"
    assert observed["key"] == "path/source.bin"


def test_s3_content_addressed_publication_is_exclusive_and_fully_read_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL", "https://nyc3.digitaloceanspaces.com")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fixture-access")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fixture-secret")
    source = tmp_path / "request.json"
    source.write_text('{"schema_version":"request.v1"}\n', encoding="utf-8")
    objects: dict[tuple[str, str], bytes] = {}
    observed: list[dict[str, object]] = []

    class Client:
        def put_object(self, **kwargs: object) -> None:
            observed.append(dict(kwargs))
            identity = (str(kwargs["Bucket"]), str(kwargs["Key"]))
            if identity in objects:
                raise RuntimeError("PreconditionFailed")
            assert kwargs["IfNoneMatch"] == "*"
            objects[identity] = bytes(kwargs["Body"])

        def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            return {"Body": io.BytesIO(objects[(Bucket, Key)])}

    class Boto3:
        @staticmethod
        def client(service: str, **kwargs: object) -> Client:
            assert service == "s3"
            return Client()

    monkeypatch.setitem(sys.modules, "boto3", Boto3())
    receipt = setup.publish_content_addressed_manifest(
        source=source,
        destination_prefix="s3://fixture-bucket/launch-manifests",
        receipt_path=tmp_path / "publication.json",
        profile_builder="build_retained_scene_render_live_profile.py",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["published_uri"].endswith(f"/sha256/{digest[:2]}/{digest}.json")
    assert receipt["storage_scheme"] == "s3"
    assert receipt["provider_full_byte_readback_verified"] is True
    assert observed[0]["IfNoneMatch"] == "*"

    blocked = setup.upload_file(source, receipt["published_uri"], exclusive=True)
    assert blocked["status"] == "blocked"


@pytest.mark.parametrize("scheme", ["s3", "r2"])
def test_s3_compatible_publication_accepts_canonical_wam_file_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheme: str
) -> None:
    source = tmp_path / "request.json"
    source.write_text('{"schema_version":"request.v1"}\n', encoding="utf-8")
    access_key = tmp_path / "wam-access-key"
    secret_key = tmp_path / "wam-secret-key"
    endpoint = tmp_path / "wam-endpoint"
    region = tmp_path / "wam-region"
    access_key.write_text("wam-access-value", encoding="utf-8")
    secret_key.write_text("wam-secret-value", encoding="utf-8")
    endpoint.write_text("https://wam-r2.example", encoding="utf-8")
    region.write_text("auto", encoding="utf-8")
    for name, path in {
        "BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE": access_key,
        "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE": secret_key,
        "BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE": endpoint,
        "BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE": region,
    }.items():
        monkeypatch.setenv(name, str(path))
    for name in (
        "AWS_ACCESS_KEY_ID",
        "AWS_ACCESS_KEY_ID_FILE",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SECRET_ACCESS_KEY_FILE",
        "AWS_REGION",
        "AWS_REGION_FILE",
        "AWS_DEFAULT_REGION",
        "AWS_DEFAULT_REGION_FILE",
        "BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL",
        "BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL_FILE",
        "R2_ENDPOINT_URL",
        "R2_ENDPOINT_URL_FILE",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_FILE",
    ):
        monkeypatch.delenv(name, raising=False)

    objects: dict[tuple[str, str], bytes] = {}
    client_kwargs: list[dict[str, object]] = []

    class Client:
        def put_object(self, **kwargs: object) -> None:
            objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))] = bytes(kwargs["Body"])

        def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            return {"Body": io.BytesIO(objects[(Bucket, Key)])}

    class Boto3:
        @staticmethod
        def client(service: str, **kwargs: object) -> Client:
            assert service == "s3"
            client_kwargs.append(dict(kwargs))
            return Client()

    monkeypatch.setitem(sys.modules, "boto3", Boto3())
    receipt = setup.publish_content_addressed_manifest(
        source=source,
        destination_prefix=f"{scheme}://fixture-bucket/launch-manifests",
        receipt_path=tmp_path / "publication.json",
        profile_builder="build_semantic_teacher_image_edit_live_profile.py",
    )

    assert receipt["status"] == "published"
    assert receipt["provider_full_byte_readback_verified"] is True
    assert client_kwargs == [
        {
            "endpoint_url": "https://wam-r2.example",
            "region_name": "auto",
            "aws_access_key_id": "wam-access-value",
            "aws_secret_access_key": "wam-secret-value",
        },
        {
            "endpoint_url": "https://wam-r2.example",
            "region_name": "auto",
            "aws_access_key_id": "wam-access-value",
            "aws_secret_access_key": "wam-secret-value",
        },
    ]
    retained = json.dumps(receipt, sort_keys=True)
    for secret in (
        "wam-access-value",
        "wam-secret-value",
        str(access_key),
        str(secret_key),
    ):
        assert secret not in retained


def test_standard_aws_credentials_keep_precedence_over_wam_file_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    access_key = tmp_path / "wam-access-key"
    secret_key = tmp_path / "wam-secret-key"
    access_key.write_text("wam-access-value", encoding="utf-8")
    secret_key.write_text("wam-secret-value", encoding="utf-8")
    monkeypatch.setenv(
        "BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE", str(access_key)
    )
    monkeypatch.setenv(
        "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE", str(secret_key)
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-access-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret-value")

    assert setup._env_or_file_value("AWS_ACCESS_KEY_ID") == "aws-access-value"
    assert setup._env_or_file_value("AWS_SECRET_ACCESS_KEY") == "aws-secret-value"


def test_content_addressed_publication_requires_full_digest_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "request.json"
    source.write_text(json.dumps({"schema_version": "request.v1"}), encoding="utf-8")
    objects: dict[tuple[str, str], bytes] = {}
    tamper = {"enabled": False}

    class Blob:
        def __init__(self, bucket: str, key: str) -> None:
            self.bucket = bucket
            self.key = key

        def upload_from_filename(self, filename: str, *, if_generation_match: int) -> None:
            assert if_generation_match == 0
            identity = (self.bucket, self.key)
            if identity in objects:
                raise RuntimeError("precondition failed")
            payload = Path(filename).read_bytes()
            objects[identity] = (b"x" * len(payload)) if tamper["enabled"] else payload

        def download_to_filename(self, filename: str) -> None:
            Path(filename).write_bytes(objects[(self.bucket, self.key)])

    class Bucket:
        def __init__(self, name: str) -> None:
            self.name = name

        def blob(self, key: str) -> Blob:
            return Blob(self.name, key)

    class Client:
        def bucket(self, name: str) -> Bucket:
            return Bucket(name)

    storage_module = types.SimpleNamespace(Client=Client)
    google_module = types.ModuleType("google")
    cloud_module = types.ModuleType("google.cloud")
    cloud_module.storage = storage_module
    google_module.cloud = cloud_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage_module)

    receipt = setup.publish_content_addressed_manifest(
        source=source,
        destination_prefix="gs://fixture-bucket/launch-manifests",
        receipt_path=tmp_path / "publication.json",
        profile_builder="build_retained_scene_render_live_profile.py",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["published_uri"].endswith(f"/sha256/{digest[:2]}/{digest}.json")
    assert receipt["provider_full_byte_readback_verified"] is True
    assert receipt["remote_sha256"] == "sha256:" + digest

    second = tmp_path / "second.json"
    second.write_text(json.dumps({"schema_version": "request.v2"}), encoding="utf-8")
    tamper["enabled"] = True
    with pytest.raises(ValueError, match="uploaded_object_digest_mismatch"):
        setup.publish_content_addressed_manifest(
            source=second,
            destination_prefix="gs://fixture-bucket/launch-manifests",
            receipt_path=tmp_path / "tampered-publication.json",
            profile_builder="build_retained_scene_render_live_profile.py",
        )


