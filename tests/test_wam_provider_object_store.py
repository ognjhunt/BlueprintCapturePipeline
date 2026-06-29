from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline import wam_provider_object_store as object_store


def test_presigned_url_expiry_metadata_uses_generated_at() -> None:
    meta = object_store._presigned_url_expiry_metadata(
        "2026-06-29T12:00:00Z",
        600,
    )

    assert meta["expires_at"] == "2026-06-29T12:10:00Z"
    assert meta["expiry_warning"] is True
    assert meta["raw_url_values_recorded"] is False


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
    assert any("digitalocean_spaces_access_key_id" in row["path"] for row in access_candidates)
    persisted = (tmp_path / "job" / "wam_provider_object_store_staging_manifest.json").read_text(
        encoding="utf-8"
    )
    assert "raw_secret" not in persisted.lower() or '"raw_secret_values_recorded": false' in persisted


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

        def delete_object(self, *, Bucket: str, Key: str) -> None:
            self.deletes.append((Bucket, Key))

        def upload_file(self, source: str, bucket: str, key: str) -> None:
            self.uploads.append((source, bucket, key))

        def generate_presigned_url(self, operation: str, *, Params, ExpiresIn, HttpMethod):
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

    manifest = object_store.stage_wam_provider_bundle_object_store(
        job_dir=tmp_path / "job",
        bundle_path=bundle,
        access_key_id_file=access,
        secret_access_key_file=secret,
        endpoint_url="https://nyc3.digitaloceanspaces.com",
        bucket_file=bucket_file,
        region="nyc3",
        key_prefix="blueprint/wam-test",
        expiration_seconds=600,
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["presigned_url_expiry"]["expiration_seconds"] == 600
    assert manifest["presigned_url_expiry"]["expiry_warning"] is True
    assert manifest["object_store"]["expires_at"]
    assert manifest["provider_bundle_url_file"]["mode_is_0600"] is True
    assert manifest["provider_output_put_url_file"]["mode_is_0600"] is True
    uploaded_keys = [row[2] for row in fake_client.uploads]
    assert uploaded_keys == [
        f"blueprint/wam-test/{object_store._job_key_component((tmp_path / 'job').resolve())}/provider_bundle.zip"
    ]
    assert manifest["output_key"] == (
        f"blueprint/wam-test/{object_store._job_key_component((tmp_path / 'job').resolve())}"
        "/runpod_provider_runtime_output.zip"
    )
    # a fresh run clears any stale output object at the key so the poll cannot grab a
    # pre-existing object and falsely report a completed fresh model run
    assert ("blueprint-wam", manifest["output_key"]) in fake_client.deletes
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


def test_presign_warm_inbox_channel_records_expiry_without_leaking_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    home = tmp_path / "home"
    secrets = home / ".blueprint-secrets"
    secrets.mkdir(parents=True)
    (secrets / "digitalocean_spaces_access_key_id").write_text("access\n", encoding="utf-8")
    (secrets / "digitalocean_spaces_secret_access_key").write_text("secret\n", encoding="utf-8")
    (secrets / "digitalocean_spaces_bucket").write_text("blueprint-wam\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeClient:
        def __init__(self) -> None:
            self.puts: list[dict] = []

        def put_object(self, **kwargs):
            self.puts.append(kwargs)

        def generate_presigned_url(self, operation: str, *, Params, ExpiresIn, HttpMethod):
            return (
                f"https://object.example/{Params['Bucket']}/{Params['Key']}"
                f"?signature=secret-{operation}&expires={ExpiresIn}&method={HttpMethod}"
            )

    fake_client = FakeClient()
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda _service, **_kwargs: fake_client),
    )
    monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "botocore.client", SimpleNamespace(Config=FakeConfig))

    manifest = object_store.presign_warm_inbox_channel(
        tmp_path / "job",
        key_prefix="blueprint/test",
        expiration_seconds=600,
    )

    assert manifest["status"] == "completed"
    assert manifest["presigned_url_expiry"]["expiry_warning"] is True
    assert manifest["expires_at"]
    assert manifest["warm_inbox_get_url_redacted"].endswith("?REDACTED_QUERY")
    assert "signature=secret" not in json.dumps(manifest)
    assert (tmp_path / "job" / "warm_inbox_get_url.txt").stat().st_mode & 0o777 == 0o600
    assert fake_client.puts and fake_client.puts[0]["ContentType"] == "application/json"


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
        def upload_file(self, _source: str, _bucket: str, _key: str) -> None:
            return None

        def generate_presigned_url(self, operation: str, *, Params, ExpiresIn, HttpMethod):
            del ExpiresIn, HttpMethod
            return f"https://object.example/{Params['Bucket']}/{Params['Key']}?sig={operation}"

    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda _service, **_kwargs: FakeClient()),
    )
    monkeypatch.setitem(sys.modules, "botocore", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "botocore.client", SimpleNamespace(Config=FakeConfig))

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
    assert object_store.main(
        [
            "--job-dir",
            str(tmp_path / "main"),
            "--bundle-path",
            str(bundle),
            "--bucket",
            "cli-bucket",
        ]
    ) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"

    monkeypatch.setattr(
        object_store,
        "stage_wam_provider_bundle_object_store",
        lambda **_kwargs: {"status": "blocked"},
    )
    assert object_store.main(
        [
            "--job-dir",
            str(tmp_path / "main-blocked"),
            "--bundle-path",
            str(bundle),
        ]
    ) == 1
