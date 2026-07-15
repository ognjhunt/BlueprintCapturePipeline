from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.common import write_json
from blueprint_pipeline.groot_oscar_model_cache import (
    MANIFEST_NAME,
    REQUIRED_MODEL_FILES,
    build_manifest,
)
from blueprint_pipeline.groot_oscar_runpod_s3_model_cache import (
    DEFAULT_REMOTE_PREFIX,
    endpoint_for_data_center,
    preflight_runpod_s3,
    upload_and_verify_model_cache,
)


class FakeS3:
    def __init__(
        self,
        *,
        corrupt_download: bool = False,
        fail_cleanup_list: bool = False,
    ) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.corrupt_download = corrupt_download
        self.fail_cleanup_list = fail_cleanup_list
        self.upload_calls = 0
        self.delete_calls = 0

    def list_buckets(self):  # type: ignore[no-untyped-def]
        return {"Buckets": [{"Name": "volume-1"}]}

    def head_bucket(self, *, Bucket: str) -> None:  # noqa: N803
        assert Bucket == "volume-1"

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        self.upload_calls += 1
        self.objects[(bucket, key)] = Path(path).read_bytes()

    def list_objects_v2(self, **kwargs):  # type: ignore[no-untyped-def]
        if self.fail_cleanup_list and self.delete_calls:
            raise RuntimeError("cleanup inventory failed")
        bucket = kwargs["Bucket"]
        prefix = kwargs["Prefix"]
        return {
            "IsTruncated": False,
            "Contents": [
                {"Key": key}
                for stored_bucket, key in sorted(self.objects)
                if stored_bucket == bucket and key.startswith(prefix)
            ],
        }

    def download_file(self, bucket: str, key: str, path: str) -> None:
        payload = self.objects[(bucket, key)]
        if self.corrupt_download and key.endswith("config.json"):
            payload += b"corrupt"
        Path(path).write_bytes(payload)

    def delete_object(self, *, Bucket: str, Key: str) -> None:  # noqa: N803
        self.delete_calls += 1
        self.objects.pop((Bucket, Key), None)


def _credentials(tmp_path: Path) -> tuple[Path, Path]:
    access = tmp_path / "runpod_s3_access_key"
    secret = tmp_path / "runpod_s3_secret_key"
    access.write_text("user_test\n", encoding="utf-8")
    secret.write_text("rps_test\n", encoding="utf-8")
    access.chmod(0o600)
    secret.chmod(0o600)
    return access, secret


def _cache(tmp_path: Path) -> Path:
    root = tmp_path / "cache"
    for model_name, relatives in REQUIRED_MODEL_FILES.items():
        for relative in relatives:
            path = root / model_name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{model_name}:{relative}".encode())
    write_json(root / MANIFEST_NAME, build_manifest(root))
    return root


def _volume_evidence(volume_id: str = "volume-1") -> dict[str, object]:
    return {
        "schema_version": "groot_oscar_runpod_network_volume_evidence.v1",
        "status": "verified",
        "id": volume_id,
        "data_center_id": "US-WA-1",
        "size_bytes": 50 * 1024**3,
    }


def test_us_wa_1_endpoint_is_exact() -> None:
    assert endpoint_for_data_center("us-wa-1") == "https://s3api-us-wa-1.runpod.io/"


def test_default_prefix_maps_to_runtime_mount() -> None:
    assert DEFAULT_REMOTE_PREFIX == ".blueprint-model-cache/blueprint-groot-oscar-v1"


def test_missing_credentials_block_before_live_probe(tmp_path: Path) -> None:
    result = preflight_runpod_s3(
        data_center_id="US-WA-1",
        access_key_file=tmp_path / "missing-access",
        secret_key_file=tmp_path / "missing-secret",
        client=object(),
    )
    assert result["status"] == "blocked"
    assert "runpod_s3_access_key_file_missing" in result["blockers"]
    assert "runpod_s3_secret_key_file_missing" in result["blockers"]
    assert result["live_probe_performed"] is False
    assert result["gpu_compute_allocated"] is False


def test_private_file_credentials_pass_live_read_only_probe(tmp_path: Path) -> None:
    access, secret = _credentials(tmp_path)
    result = preflight_runpod_s3(
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        client=FakeS3(),
    )
    assert result["status"] == "ready"
    assert result["visible_network_volume_count"] == 1
    assert result["raw_secret_values_recorded"] is False


def test_upload_requires_full_redownload_and_manifest_hash_verification(
    tmp_path: Path,
) -> None:
    access, secret = _credentials(tmp_path)
    client = FakeS3()
    result = upload_and_verify_model_cache(
        cache_root=_cache(tmp_path),
        verification_root=tmp_path / "redownload",
        volume_id="volume-1",
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        volume_evidence=_volume_evidence(),
        client=client,
        available_bytes=100 * 1024**3,
    )
    assert result["status"] == "completed"
    assert result["verification_method"] == (
        "full_s3_redownload_and_sha256_manifest_verification"
    )
    assert result["multipart_etag_used_as_integrity_proof"] is False
    assert client.upload_calls == result["remote_object_count"]


def test_corrupt_redownload_fails_closed(tmp_path: Path) -> None:
    access, secret = _credentials(tmp_path)
    result = upload_and_verify_model_cache(
        cache_root=_cache(tmp_path),
        verification_root=tmp_path / "redownload",
        volume_id="volume-1",
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        volume_evidence=_volume_evidence(),
        client=FakeS3(corrupt_download=True),
        available_bytes=100 * 1024**3,
    )
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["gpu_compute_allocated"] is False
    assert result["partial_upload_cleanup_verified"] is True


def test_source_and_verification_roots_must_not_overlap(tmp_path: Path) -> None:
    access, secret = _credentials(tmp_path)
    cache = _cache(tmp_path)
    result = upload_and_verify_model_cache(
        cache_root=cache,
        verification_root=cache / "redownload",
        volume_id="volume-1",
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        volume_evidence=_volume_evidence(),
        client=FakeS3(),
        available_bytes=100 * 1024**3,
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["model_cache_verification_root_overlaps_source"]


def test_unmanifested_extra_file_is_not_uploaded(tmp_path: Path) -> None:
    access, secret = _credentials(tmp_path)
    cache = _cache(tmp_path)
    (cache / "unmanifested-secret.txt").write_text("do not upload", encoding="utf-8")
    client = FakeS3()
    result = upload_and_verify_model_cache(
        cache_root=cache,
        verification_root=tmp_path / "redownload",
        volume_id="volume-1",
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        volume_evidence=_volume_evidence(),
        client=client,
        available_bytes=100 * 1024**3,
    )
    assert result["status"] == "completed"
    assert all(not key.endswith("unmanifested-secret.txt") for _, key in client.objects)


def test_expected_volume_must_be_visible_to_s3_credentials(tmp_path: Path) -> None:
    access, secret = _credentials(tmp_path)
    result = preflight_runpod_s3(
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        expected_volume_id="different-volume",
        client=FakeS3(),
    )
    assert result["status"] == "blocked"
    assert result["live_probe_error_type"] == "ValueError"


def test_nonempty_remote_prefix_blocks_before_upload(tmp_path: Path) -> None:
    access, secret = _credentials(tmp_path)
    client = FakeS3()
    client.objects[("volume-1", ".blueprint-model-cache/blueprint-groot-oscar-v1/old")] = b"old"
    result = upload_and_verify_model_cache(
        cache_root=_cache(tmp_path),
        verification_root=tmp_path / "redownload",
        volume_id="volume-1",
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        volume_evidence=_volume_evidence(),
        client=client,
        available_bytes=100 * 1024**3,
    )
    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert client.upload_calls == 0
    assert client.delete_calls == 0


def test_failed_cleanup_requires_outer_volume_deletion(tmp_path: Path) -> None:
    access, secret = _credentials(tmp_path)
    client = FakeS3(corrupt_download=True, fail_cleanup_list=True)
    result = upload_and_verify_model_cache(
        cache_root=_cache(tmp_path),
        verification_root=tmp_path / "redownload",
        volume_id="volume-1",
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        volume_evidence=_volume_evidence(),
        client=client,
        available_bytes=100 * 1024**3,
    )
    assert result["status"] == "failed"
    assert result["partial_upload_cleanup_verified"] is False
    assert result["outer_volume_deletion_required"] is True
