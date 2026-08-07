from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

from blueprint_pipeline import wam_provider_object_store as object_store


def test_presigned_url_expiry_metadata_uses_generated_at() -> None:
    meta = object_store._presigned_url_expiry_metadata(
        "2026-06-29T12:00:00Z",
        600,
    )

    assert meta["expires_at"] == "2026-06-29T12:10:00Z"
    assert meta["expiry_warning"] is True
    assert meta["raw_url_values_recorded"] is False


def test_staging_binding_reservation_is_exclusive_and_preserves_first_writer(
    tmp_path: Path,
) -> None:
    binding_path = tmp_path / object_store.STAGING_BINDING_FILENAME
    first = {"staging_binding_sha256": "a" * 64}
    second = {"staging_binding_sha256": "b" * 64}

    object_store._write_staging_binding_exclusive(binding_path, first)
    first_bytes = binding_path.read_bytes()
    try:
        object_store._write_staging_binding_exclusive(binding_path, second)
    except FileExistsError:
        pass
    else:  # pragma: no cover - contract failure made explicit
        raise AssertionError("second staging writer replaced the first binding")

    assert binding_path.read_bytes() == first_bytes
    assert json.loads(first_bytes)["staging_binding_sha256"] == "a" * 64
    assert oct(binding_path.stat().st_mode & 0o777) == "0o600"


def test_wam_provider_object_store_blocks_without_file_based_credentials(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    bundle = tmp_path / "provider_bundle.zip"
    bundle.write_bytes(b"bundle")

    manifest = object_store.stage_wam_provider_bundle_object_store(
        job_dir=tmp_path / "job",
        bundle_path=bundle,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "missing_object_store_access_key_id_file" in manifest["blockers"]
    assert "missing_object_store_secret_access_key_file" in manifest["blockers"]
    assert "missing_object_store_bucket_or_network_volume_id_file" in manifest["blockers"]
    access_candidates = manifest["object_store"]["access_key_id"]["candidate_files"]
    assert access_candidates
    assert all(row["path_redacted"] is True for row in access_candidates)
    assert all("path" not in row for row in access_candidates)
    assert manifest["secret_artifact_policy"]["local_secret_file_paths_recorded"] is False
    assert manifest["bundle_key"] is None
    assert manifest["output_key"] is None
    assert manifest["bundle_key_content_addressed"] is False
    assert manifest["output_key_run_unique"] is False
    assert not (tmp_path / "job" / object_store.STAGING_BINDING_FILENAME).exists()
    assert object_store._existing_staging_initialization(tmp_path / "job")["initialized"] is False
    persisted = (tmp_path / "job" / "wam_provider_object_store_staging_manifest.json").read_text(
        encoding="utf-8"
    )
    assert ".blueprint-secrets" not in persisted
    assert (
        "raw_secret" not in persisted.lower() or '"raw_secret_values_recorded": false' in persisted
    )


def test_refresh_existing_output_get_url_preserves_object_and_extends_access(
    tmp_path: Path, monkeypatch
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    output_key = "blueprint/wam-test/job/checkpoint.part-001"
    (job / "wam_provider_object_store_staging_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": object_store.SCHEMA_VERSION,
                "status": "completed",
                "output_key": output_key,
            }
        ),
        encoding="utf-8",
    )
    (job / "provider_output_put_url.txt").write_text(
        f"https://nyc3.digitaloceanspaces.com/blueprint-wam/{output_key}?X-Amz-Signature=old-put\n",
        encoding="utf-8",
    )
    access = tmp_path / "access"
    secret = tmp_path / "secret"
    bucket_file = tmp_path / "bucket"
    access.write_text("access-value\n", encoding="utf-8")
    secret.write_text("secret-value\n", encoding="utf-8")
    bucket_file.write_text("blueprint-wam\n", encoding="utf-8")

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeClient:
        def head_object(self, *, Bucket: str, Key: str):
            assert Bucket == "blueprint-wam"
            assert Key == output_key
            return {"ContentLength": 123}

        def generate_presigned_url(self, operation: str, *, Params, ExpiresIn, HttpMethod):
            assert operation == "get_object"
            assert Params == {"Bucket": "blueprint-wam", "Key": output_key}
            assert ExpiresIn == 600
            assert HttpMethod == "GET"
            return (
                "https://nyc3.digitaloceanspaces.com/blueprint-wam/"
                f"{output_key}?X-Amz-Signature=refreshed"
            )

    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda *_args, **_kwargs: FakeClient()),
    )
    monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "botocore.client", SimpleNamespace(Config=FakeConfig))

    result = object_store.refresh_wam_provider_output_get_url(
        job_dir=job,
        access_key_id_file=access,
        secret_access_key_file=secret,
        endpoint_url="https://nyc3.digitaloceanspaces.com",
        bucket_file=bucket_file,
        region="nyc3",
        expiration_seconds=600,
        generated_at="2026-07-18T12:00:00Z",
    )

    assert result["status"] == "completed"
    assert result["object_size_bytes"] == 123
    assert result["output_object_mutated"] is False
    assert result["presigned_url_expiry"]["expires_at"] == "2026-07-18T12:10:00Z"
    get_url = job / "provider_output_get_url.txt"
    assert oct(get_url.stat().st_mode & 0o777) == "0o600"
    assert "X-Amz-Signature=refreshed" in get_url.read_text(encoding="utf-8")
    persisted = (job / "wam_provider_object_store_staging_manifest.json").read_text(
        encoding="utf-8"
    )
    refresh = (job / "wam_provider_object_store_get_refresh.json").read_text(encoding="utf-8")
    assert "X-Amz-Signature" not in persisted
    assert "X-Amz-Signature" not in refresh


def test_wam_provider_object_store_writes_0600_signed_url_files_without_leaking_query(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bundle = tmp_path / "provider_bundle.zip"
    bundle.write_bytes(b"bundle")
    access = tmp_path / "spaces-access-key"
    secret = tmp_path / "spaces-secret-key"
    bucket_file = tmp_path / "spaces-bucket"
    access.write_text("access-value-should-not-leak\n", encoding="utf-8")
    secret.write_text("secret-value-should-not-leak\n", encoding="utf-8")
    bucket_file.write_text("blueprint-wam\n", encoding="utf-8")

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeClient:
        def __init__(self) -> None:
            self.uploads: list[tuple[str, str, str]] = []
            self.deletes: list[tuple[str, str]] = []
            self.objects: dict[tuple[str, str], bytes] = {}
            self.presign_params: list[tuple[str, dict[str, str]]] = []

        def delete_object(self, *, Bucket: str, Key: str):
            self.deletes.append((Bucket, Key))
            self.objects.pop((Bucket, Key), None)
            return {"ResponseMetadata": {"HTTPStatusCode": 204}}

        def head_object(self, *, Bucket: str, Key: str):
            if (Bucket, Key) in self.objects:
                return {"ResponseMetadata": {"HTTPStatusCode": 200}}
            error = RuntimeError("not found")
            error.response = {  # type: ignore[attr-defined]
                "ResponseMetadata": {"HTTPStatusCode": 404},
                "Error": {"Code": "NoSuchKey"},
            }
            raise error

        def upload_file(self, source: str, bucket: str, key: str) -> None:
            self.uploads.append((source, bucket, key))

        def generate_presigned_url(self, operation: str, *, Params, ExpiresIn, HttpMethod):
            self.presign_params.append((operation, dict(Params)))
            key = Params["Key"]
            return (
                f"https://nyc3.digitaloceanspaces.com/{Params['Bucket']}/{key}"
                f"?X-Amz-Signature=fake-signature-{operation}&Expires={ExpiresIn}&Method={HttpMethod}"
            )

    fake_client = FakeClient()

    def fake_boto3_client(service: str, **kwargs):
        assert service == "s3"
        assert kwargs["endpoint_url"] == "https://nyc3.digitaloceanspaces.com"
        assert kwargs["aws_access_key_id"] == "access-value-should-not-leak"
        assert kwargs["aws_secret_access_key"] == "secret-value-should-not-leak"
        return fake_client

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=fake_boto3_client))
    monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "botocore.client", SimpleNamespace(Config=FakeConfig))

    def fake_safe_http_request(url: str, *, method: str, data=None, **_kwargs):
        parsed = urlparse(url)
        path_parts = parsed.path.lstrip("/").split("/", 1)
        request_bucket, key = path_parts
        assert request_bucket == "blueprint-wam"
        if method == "PUT":
            fake_client.objects[(request_bucket, key)] = bytes(data)
            return SimpleNamespace(status=200, body=b"")
        assert method == "GET"
        return SimpleNamespace(
            status=200,
            body=fake_client.objects[(request_bucket, key)],
        )

    monkeypatch.setattr(object_store, "safe_http_request", fake_safe_http_request)

    manifest = object_store.stage_wam_provider_bundle_object_store(
        job_dir=tmp_path / "job",
        bundle_path=bundle,
        access_key_id_file=access,
        secret_access_key_file=secret,
        endpoint_url="https://nyc3.digitaloceanspaces.com",
        bucket_file=bucket_file,
        region="nyc3",
        key_prefix="blueprint/wam-test",
        output_content_type="application/json",
        expiration_seconds=600,
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["presigned_url_expiry"]["expiration_seconds"] == 600
    assert manifest["presigned_url_expiry"]["expiry_warning"] is True
    assert manifest["object_store"]["expires_at"]
    assert manifest["provider_bundle_url_file"]["mode_is_0600"] is True
    assert manifest["provider_output_put_url_file"]["mode_is_0600"] is True
    bundle_sha256 = hashlib.sha256(b"bundle").hexdigest()
    expected_bundle_key = (
        f"blueprint/wam-test/{object_store._job_key_component((tmp_path / 'job').resolve())}/"
        f"bundles/sha256/{bundle_sha256}.zip"
    )
    bundle_get_params = [
        params
        for operation, params in fake_client.presign_params
        if operation == "get_object" and params.get("ResponseCacheControl")
    ]
    assert bundle_get_params == [
        {
            "Bucket": "blueprint-wam",
            "Key": expected_bundle_key,
            "ResponseCacheControl": "no-store, max-age=0",
        }
    ]
    uploaded_keys = [row[2] for row in fake_client.uploads]
    assert uploaded_keys == [expected_bundle_key]
    assert manifest["bundle_sha256"] == bundle_sha256
    assert manifest["bundle_key"] == expected_bundle_key
    assert manifest["bundle_key_content_addressed"] is True
    assert manifest["staging_binding_file"]["mode_is_0600"] is True
    binding = json.loads(
        (tmp_path / "job" / object_store.STAGING_BINDING_FILENAME).read_text(encoding="utf-8")
    )
    assert binding["bundle_sha256"] == bundle_sha256
    assert binding["bundle_key"] == expected_bundle_key
    assert binding["staging_binding_sha256"] == manifest["staging_binding_sha256"]
    assert re.fullmatch(
        f"blueprint/wam-test/{object_store._job_key_component((tmp_path / 'job').resolve())}"
        r"/runpod_provider_runtime_output_[0-9a-f]{32}\.json",
        manifest["output_key"],
    )
    actual_output_put_params = [
        params
        for operation, params in fake_client.presign_params
        if operation == "put_object" and params.get("Key") == manifest["output_key"]
    ]
    assert actual_output_put_params == [
        {
            "Bucket": "blueprint-wam",
            "Key": manifest["output_key"],
            "ContentType": "application/json",
        }
    ]
    assert manifest["object_store"]["output_content_type"] == "application/json"
    round_trip = manifest["signed_output_round_trip"]
    assert round_trip["status"] == "passed"
    assert round_trip["put"]["status"] == "passed"
    assert round_trip["get"]["exact_bytes_and_sha256"] is True
    assert round_trip["cleanup"]["absence_confirmed"] is True
    assert round_trip["actual_output_key_was_not_used"] is True
    assert manifest["fresh_output_key_absence"]["absence_confirmed"] is True
    assert manifest["output_key_run_unique"] is True
    assert len(manifest["output_url_object_binding_sha256"]) == 64
    assert all(key != manifest["output_key"] for _, key in fake_client.deletes)
    assert ("blueprint-wam", manifest["output_key"]) not in fake_client.objects
    assert manifest["provider_bundle_url_redacted"].endswith("?REDACTED_QUERY")
    persisted = (tmp_path / "job" / "wam_provider_object_store_staging_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "access-value-should-not-leak" not in persisted
    assert "secret-value-should-not-leak" not in persisted
    assert "fake-signature" not in persisted
    assert "fake-signature" in (tmp_path / "job" / "provider_bundle_url.txt").read_text(
        encoding="utf-8"
    )

    canonical_manifest_path = tmp_path / "job" / object_store.STAGING_MANIFEST_FILENAME
    canonical_manifest_before = canonical_manifest_path.read_bytes()
    signed_urls_before = {
        name: (tmp_path / "job" / name).read_bytes() for name in object_store.SIGNED_URL_FILENAMES
    }
    mutation_counts_before = (
        len(fake_client.uploads),
        len(fake_client.presign_params),
        len(fake_client.deletes),
    )
    replacement_bytes = b"different bundle"
    bundle.write_bytes(replacement_bytes)

    rejected = object_store.stage_wam_provider_bundle_object_store(
        job_dir=tmp_path / "job",
        bundle_path=bundle,
        access_key_id_file=access,
        secret_access_key_file=secret,
        endpoint_url="https://nyc3.digitaloceanspaces.com",
        bucket_file=bucket_file,
        region="nyc3",
        key_prefix="blueprint/wam-test",
        expiration_seconds=600,
        generated_at="later",
    )

    assert rejected["status"] == "blocked"
    assert rejected["blockers"] == ["wam_provider_staging_job_dir_already_initialized"]
    assert rejected["requested_bundle_sha256"] == hashlib.sha256(replacement_bytes).hexdigest()
    assert rejected["provider_mutations_performed"] is False
    assert rejected["canonical_staging_manifest_preserved"] is True
    assert rejected["signed_url_files_preserved"] is True
    assert mutation_counts_before == (
        len(fake_client.uploads),
        len(fake_client.presign_params),
        len(fake_client.deletes),
    )
    assert canonical_manifest_path.read_bytes() == canonical_manifest_before
    assert {
        name: (tmp_path / "job" / name).read_bytes() for name in object_store.SIGNED_URL_FILENAMES
    } == signed_urls_before
    rejection_files = list(
        (tmp_path / "job").glob("wam_provider_object_store_staging_reuse_rejection_*.json")
    )
    assert len(rejection_files) == 1
    assert json.loads(rejection_files[0].read_text(encoding="utf-8")) == rejected


def test_signed_output_round_trip_blocks_mismatch_but_still_proves_cleanup(
    monkeypatch,
) -> None:
    sentinel_key = "blueprint/preflight/sentinel.bin"

    class FakeNotFound(RuntimeError):
        response = {
            "ResponseMetadata": {"HTTPStatusCode": 404},
            "Error": {"Code": "NoSuchKey"},
        }

    class Client:
        def __init__(self) -> None:
            self.deleted = False

        def generate_presigned_url(self, operation: str, *, Params, **_kwargs):
            assert Params["Key"] == sentinel_key
            return (
                f"https://objects.example/bucket/{sentinel_key}"
                f"?X-Amz-Signature=must-not-leak-{operation}"
            )

        def delete_object(self, *, Bucket: str, Key: str):
            assert (Bucket, Key) == ("bucket", sentinel_key)
            self.deleted = True
            return {"ResponseMetadata": {"HTTPStatusCode": 204}}

        def head_object(self, *, Bucket: str, Key: str):
            assert self.deleted is True
            raise FakeNotFound("deleted")

    def fake_request(_url: str, *, method: str, **_kwargs):
        return SimpleNamespace(
            status=200,
            body=b"" if method == "PUT" else b"wrong-sentinel",
        )

    monkeypatch.setattr(object_store, "safe_http_request", fake_request)
    result = object_store._signed_output_round_trip_preflight(
        Client(),
        bucket="bucket",
        sentinel_key=sentinel_key,
        expiration_seconds=600,
    )

    assert result["status"] == "blocked"
    assert "signed_output_sentinel_get_mismatch" in result["blockers"]
    assert result["cleanup"]["status"] == "passed"
    assert result["cleanup"]["absence_confirmed"] is True
    persisted = json.dumps(result, sort_keys=True)
    assert "must-not-leak" not in persisted
    assert "https://" not in persisted


def test_signed_output_round_trip_blocks_when_cleanup_cannot_be_confirmed(
    monkeypatch,
) -> None:
    payload: dict[str, bytes] = {}

    class Client:
        def generate_presigned_url(self, operation: str, *, Params, **_kwargs):
            return (
                f"https://objects.example/bucket/{Params['Key']}?X-Amz-Signature=secret-{operation}"
            )

        def delete_object(self, **_kwargs):
            raise urllib.error.HTTPError(
                "https://objects.example/?X-Amz-Signature=must-not-leak",
                403,
                "forbidden",
                {},
                None,
            )

    def fake_request(_url: str, *, method: str, data=None, **_kwargs):
        if method == "PUT":
            payload["value"] = bytes(data)
            return SimpleNamespace(status=200, body=b"")
        return SimpleNamespace(status=200, body=payload["value"])

    monkeypatch.setattr(object_store, "safe_http_request", fake_request)
    result = object_store._signed_output_round_trip_preflight(
        Client(),
        bucket="bucket",
        sentinel_key="prefix/preflight/sentinel.bin",
        expiration_seconds=600,
    )

    assert result["status"] == "blocked"
    assert "signed_output_sentinel_cleanup_unverified" in result["blockers"]
    assert result["cleanup"]["http_status_code"] == 403
    assert "must-not-leak" not in json.dumps(result, sort_keys=True)


def test_warm_inbox_compatibility_entrypoint_requires_durable_broker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    base_url_file = tmp_path / "broker-base-url"
    token_file = tmp_path / "broker-token"
    base_url_file.write_text("https://warm-broker.example", encoding="utf-8")
    token_file.write_text("s" * 64, encoding="utf-8")
    token_file.chmod(0o600)
    monkeypatch.setenv(
        "BLUEPRINT_WARM_RENDER_BROKER_BASE_URL_FILE",
        str(base_url_file),
    )
    monkeypatch.setenv(
        "BLUEPRINT_WARM_RENDER_BROKER_TOKEN_FILE",
        str(token_file),
    )

    manifest = object_store.presign_warm_inbox_channel(
        tmp_path / "job",
        key_prefix="blueprint/test",
        expiration_seconds=600,
    )

    assert manifest["status"] == "completed"
    assert manifest["transport"] == "durable_warm_render_broker"
    assert manifest["single_object_transport_enabled"] is False
    assert manifest["server_canonical_job_ids_required"] is True
    assert manifest["server_idempotency_required"] is True
    assert manifest["inbox_key"] is None
    assert manifest["warm_inbox_get_url_file"] is None
    assert manifest["warm_inbox_put_url_file"] is None
    assert "s" * 64 not in json.dumps(manifest)
    assert (tmp_path / "job" / "warm_broker_token.txt").stat().st_mode & 0o777 == 0o600


def test_wam_provider_object_store_bucket_cli_and_main_edges(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    bundle = tmp_path / "provider_bundle.zip"
    bundle.write_bytes(b"bundle")
    access = tmp_path / "access"
    secret = tmp_path / "secret"
    access.write_text("access\n", encoding="utf-8")
    secret.write_text("secret\n", encoding="utf-8")

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeClient:
        def __init__(self) -> None:
            self.objects: dict[tuple[str, str], bytes] = {}

        def upload_file(self, _source: str, _bucket: str, _key: str) -> None:
            return None

        def delete_object(self, *, Bucket: str, Key: str):
            self.objects.pop((Bucket, Key), None)
            return {"ResponseMetadata": {"HTTPStatusCode": 204}}

        def head_object(self, *, Bucket: str, Key: str):
            if (Bucket, Key) in self.objects:
                return {"ResponseMetadata": {"HTTPStatusCode": 200}}
            error = RuntimeError("not found")
            error.response = {  # type: ignore[attr-defined]
                "ResponseMetadata": {"HTTPStatusCode": 404},
                "Error": {"Code": "NoSuchKey"},
            }
            raise error

        def generate_presigned_url(self, operation: str, *, Params, ExpiresIn, HttpMethod):
            del ExpiresIn, HttpMethod
            return f"https://object.example/{Params['Bucket']}/{Params['Key']}?sig={operation}"

    fake_client = FakeClient()
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda _service, **_kwargs: fake_client),
    )
    monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "botocore.client", SimpleNamespace(Config=FakeConfig))

    def fake_request(url: str, *, method: str, data=None, **_kwargs):
        bucket, key = urlparse(url).path.lstrip("/").split("/", 1)
        if method == "PUT":
            fake_client.objects[(bucket, key)] = bytes(data)
            return SimpleNamespace(status=200, body=b"")
        return SimpleNamespace(status=200, body=fake_client.objects[(bucket, key)])

    monkeypatch.setattr(object_store, "safe_http_request", fake_request)

    manifest = object_store.stage_wam_provider_bundle_object_store(
        job_dir=tmp_path / "cli-bucket",
        bundle_path=bundle,
        access_key_id_file=access,
        secret_access_key_file=secret,
        endpoint_url="https://object.example",
        bucket="cli-bucket",
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["object_store"]["bucket"]["source"] == "cli_argument"
    assert "--provider-output-get-url-file" in manifest["runpod_create_command_template"]
    assert "--provider-bundle-url-file" in manifest["vast_create_command_template"]
    assert "--provider-output-put-url-file" in manifest["vast_create_command_template"]
    assert "--provider-output-get-url-file" in manifest["vast_create_command_template"]

    monkeypatch.setattr(
        object_store,
        "stage_wam_provider_bundle_object_store",
        lambda **_kwargs: {"status": "completed"},
    )
    assert (
        object_store.main(
            [
                "--job-dir",
                str(tmp_path / "main"),
                "--bundle-path",
                str(bundle),
                "--bucket",
                "cli-bucket",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "completed"

    monkeypatch.setattr(
        object_store,
        "stage_wam_provider_bundle_object_store",
        lambda **_kwargs: {"status": "blocked"},
    )
    assert (
        object_store.main(
            [
                "--job-dir",
                str(tmp_path / "main-blocked"),
                "--bundle-path",
                str(bundle),
            ]
        )
        == 1
    )


def test_cleanup_staged_objects_is_exact_and_absence_proven(tmp_path: Path, monkeypatch) -> None:
    job = tmp_path / "job"
    job.mkdir()
    keys = ["blueprint/task/job/bundle.zip", "blueprint/task/job/output.zip"]
    (job / "wam_provider_object_store_staging_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": object_store.SCHEMA_VERSION,
                "status": "completed",
                "object_store": {"key_prefix": "blueprint/task"},
                "bundle_key": keys[0],
                "output_key": keys[1],
            }
        ),
        encoding="utf-8",
    )
    for name in (
        "provider_bundle_url.txt",
        "provider_output_put_url.txt",
        "provider_output_get_url.txt",
    ):
        (job / name).write_text("https://signed.example/?secret\n", encoding="utf-8")
    access = tmp_path / "access"
    secret = tmp_path / "secret"
    access.write_text("access\n", encoding="utf-8")
    secret.write_text("secret\n", encoding="utf-8")

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeNotFound(RuntimeError):
        response = {
            "ResponseMetadata": {"HTTPStatusCode": 404},
            "Error": {"Code": "NoSuchKey"},
        }

    class FakeClient:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        def delete_object(self, *, Bucket: str, Key: str):
            assert Bucket == "bucket"
            self.deleted.append(Key)
            return {"ResponseMetadata": {"HTTPStatusCode": 204}}

        def head_object(self, *, Bucket: str, Key: str):
            assert Bucket == "bucket"
            assert Key in self.deleted
            raise FakeNotFound("absent")

    client = FakeClient()
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda _service, **_kwargs: client),
    )
    monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "botocore.client", SimpleNamespace(Config=FakeConfig))

    result = object_store.cleanup_staged_wam_provider_objects(
        job,
        access_key_id_file=access,
        secret_access_key_file=secret,
        bucket="bucket",
    )

    assert result["status"] == "completed"
    assert client.deleted == keys
    assert result["all_objects_absent"] is True
    assert result["signed_url_files_removed"] is True
    persisted = (job / "wam_provider_object_store_cleanup.json").read_text(encoding="utf-8")
    assert "signed.example" not in persisted
    assert keys[0] not in persisted


def test_cleanup_retries_transient_transport_failure_then_proves_absence(
    tmp_path: Path, monkeypatch
) -> None:
    """A one-off SSLError must not strand staged objects and live signed URLs.

    A live run lost cleanup to exactly this and left two staged objects plus
    three signed-URL files behind, which had to be swept by hand.  Delete and
    absence-proof are idempotent, so the transient case is retried.
    """

    job = tmp_path / "job"
    job.mkdir()
    keys = ["blueprint/task/job/bundle.zip", "blueprint/task/job/output.zip"]
    (job / "wam_provider_object_store_staging_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": object_store.SCHEMA_VERSION,
                "status": "completed",
                "object_store": {"key_prefix": "blueprint/task"},
                "bundle_key": keys[0],
                "output_key": keys[1],
            }
        ),
        encoding="utf-8",
    )
    for name in object_store.SIGNED_URL_FILENAMES:
        (job / name).write_text("https://signed.example/?secret\n", encoding="utf-8")
    access = tmp_path / "access"
    secret = tmp_path / "secret"
    access.write_text("access\n", encoding="utf-8")
    secret.write_text("secret\n", encoding="utf-8")

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class SSLError(RuntimeError):
        """Mimics botocore's transient transport error by name."""

    class FakeNotFound(RuntimeError):
        response = {
            "ResponseMetadata": {"HTTPStatusCode": 404},
            "Error": {"Code": "NoSuchKey"},
        }

    class FakeClient:
        def __init__(self) -> None:
            self.deleted: list[str] = []
            self.attempts = 0

        def delete_object(self, *, Bucket: str, Key: str):
            if Key == keys[0]:
                self.attempts += 1
                if self.attempts == 1:
                    raise SSLError("transient tls failure")
            self.deleted.append(Key)
            return {"ResponseMetadata": {"HTTPStatusCode": 204}}

        def head_object(self, *, Bucket: str, Key: str):
            assert Key in self.deleted
            raise FakeNotFound("absent")

    client = FakeClient()
    monkeypatch.setattr(object_store, "CLEANUP_TRANSIENT_RETRY_SLEEP_SECONDS", 0.0)
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda _service, **_kwargs: client),
    )
    monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "botocore.client", SimpleNamespace(Config=FakeConfig))

    result = object_store.cleanup_staged_wam_provider_objects(
        job,
        access_key_id_file=access,
        secret_access_key_file=secret,
        bucket="bucket",
    )

    assert result["status"] == "completed"
    assert result["all_objects_absent"] is True
    assert result["signed_url_files_removed"] is True
    assert result["cleanup_attempts"] == 2
    assert sorted(client.deleted) == sorted(keys)
    assert not [name for name in object_store.SIGNED_URL_FILENAMES if (job / name).exists()]


def test_cleanup_does_not_retry_a_non_transient_failure(tmp_path: Path, monkeypatch) -> None:
    """A permission or configuration error must fail closed on the first attempt."""

    job = tmp_path / "job"
    job.mkdir()
    keys = ["blueprint/task/job/bundle.zip", "blueprint/task/job/output.zip"]
    (job / "wam_provider_object_store_staging_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": object_store.SCHEMA_VERSION,
                "status": "completed",
                "object_store": {"key_prefix": "blueprint/task"},
                "bundle_key": keys[0],
                "output_key": keys[1],
            }
        ),
        encoding="utf-8",
    )
    access = tmp_path / "access"
    secret = tmp_path / "secret"
    access.write_text("access\n", encoding="utf-8")
    secret.write_text("secret\n", encoding="utf-8")

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class AccessDenied(RuntimeError):
        pass

    class FakeClient:
        def __init__(self) -> None:
            self.attempts = 0

        def delete_object(self, *, Bucket: str, Key: str):
            self.attempts += 1
            raise AccessDenied("nope")

    client = FakeClient()
    monkeypatch.setattr(object_store, "CLEANUP_TRANSIENT_RETRY_SLEEP_SECONDS", 0.0)
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda _service, **_kwargs: client),
    )
    monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "botocore.client", SimpleNamespace(Config=FakeConfig))

    result = object_store.cleanup_staged_wam_provider_objects(
        job,
        access_key_id_file=access,
        secret_access_key_file=secret,
        bucket="bucket",
    )

    assert result["status"] == "blocked"
    assert result["cleanup_attempts"] == 1
    assert client.attempts == 1
    assert any("AccessDenied" in blocker for blocker in result["blockers"])
