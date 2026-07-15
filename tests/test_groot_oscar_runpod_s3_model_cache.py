from __future__ import annotations

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import inspect

import pytest

import blueprint_pipeline.groot_oscar_runpod_s3_model_cache as s3_transport

from blueprint_pipeline.common import write_json
from blueprint_pipeline.groot_oscar_infrastructure_admission import (
    RUNPOD_NETWORK_VOLUME_DATA_CENTER_IDS,
)
from blueprint_pipeline.groot_oscar_model_cache import (
    MANIFEST_NAME,
    REQUIRED_MODEL_FILES,
    build_manifest,
    verify_model_cache,
)
from blueprint_pipeline.groot_oscar_runpod_s3_model_cache import (
    DEFAULT_REMOTE_PREFIX,
    REMOTE_SAFETY_HEADROOM_BYTES,
    RUNPOD_S3_VOLUME_DATA_CENTER_IDS,
    _TransportExecutionCapability,
    _issue_transport_execution_capability,
    _upload_and_verify_model_cache_impl,
    endpoint_for_data_center,
    main,
    preflight_runpod_s3,
    upload_and_verify_model_cache,
)
from blueprint_pipeline.paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission,
)


ALLOCATION_NONCE = "nonce1234"


@pytest.fixture(autouse=True)
def _large_test_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        s3_transport,
        "_filesystem_available_bytes",
        lambda _path: 100 * 1024**3,
    )


def test_public_upload_api_has_no_caller_claimed_disk_headroom_override() -> None:
    assert "available_bytes" not in inspect.signature(upload_and_verify_model_cache).parameters


def _direct_transport(capability: object | None) -> dict:
    return _upload_and_verify_model_cache_impl(
        cache_root="/does/not/matter",
        verification_root="/also/unused",
        volume_id="!invalid",
        data_center_id="US-WA-1",
        access_key_file="/missing",
        secret_key_file="/missing",
        volume_evidence={},
        allocation_nonce="invalid",
        execution_capability=capability,  # type: ignore[arg-type]
    )


def test_private_transport_rejects_missing_and_forged_capability_before_network() -> None:
    missing = _direct_transport(None)
    forged = _direct_transport(_TransportExecutionCapability(object()))
    assert missing["blockers"] == ["runpod_s3_transport_execution_capability_invalid"]
    assert forged["blockers"] == ["runpod_s3_transport_execution_capability_invalid"]
    assert missing["provider_mutations_performed"] == 0
    assert forged["provider_mutations_performed"] == 0
    with pytest.raises(PaidResourceAdmissionBlocked):
        _issue_transport_execution_capability()
    with pytest.raises(PaidResourceAdmissionBlocked):
        _issue_transport_execution_capability(
            remote_parent_binding={},
            remote_parent_capability=b"x" * 32,
            remote_packet={},
        )


def test_private_transport_capability_is_one_shot_and_concurrency_safe() -> None:
    capability = _issue_transport_execution_capability(
        paid_resource_admission_grant=_grant()
    )
    first = _direct_transport(capability)
    second = _direct_transport(capability)
    assert "runpod_s3_volume_id_invalid" in first["blockers"]
    assert second["blockers"] == ["runpod_s3_transport_execution_capability_invalid"]

    concurrent_capability = _issue_transport_execution_capability(
        paid_resource_admission_grant=_grant()
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                _direct_transport,
                [concurrent_capability, concurrent_capability],
            )
        )
    assert sum(
        result["blockers"] == ["runpod_s3_transport_execution_capability_invalid"]
        for result in results
    ) == 1
    assert all(result["provider_mutations_performed"] == 0 for result in results)


class FakeS3:
    def __init__(
        self,
        *,
        corrupt_download: bool = False,
        fail_cleanup_list: bool = False,
        fail_upload_on: int | None = None,
        write_before_upload_failure: bool = False,
        fail_delete: bool = False,
    ) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.corrupt_download = corrupt_download
        self.fail_cleanup_list = fail_cleanup_list
        self.fail_upload_on = fail_upload_on
        self.write_before_upload_failure = write_before_upload_failure
        self.fail_delete = fail_delete
        self.upload_calls = 0
        self.delete_calls = 0

    def list_buckets(self):  # type: ignore[no-untyped-def]
        return {"Buckets": [{"Name": "volume-1"}]}

    def head_bucket(self, *, Bucket: str) -> None:  # noqa: N803
        assert Bucket == "volume-1"

    def upload_file(self, path: str, bucket: str, key: str) -> None:
        self.upload_calls += 1
        if self.fail_upload_on == self.upload_calls:
            if self.write_before_upload_failure:
                self.objects[(bucket, key)] = Path(path).read_bytes()
            raise RuntimeError("injected upload failure")
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
        if self.fail_delete:
            raise RuntimeError("injected delete failure")
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
        "name": f"blueprint-groot-oscar-model-{ALLOCATION_NONCE}",
        "data_center_id": "US-WA-1",
        "size_bytes": 50 * 1024**3,
        "allocation_nonce": ALLOCATION_NONCE,
        "allocation_name_verified": True,
    }


def _grant() -> PaidResourceAdmissionGrant:
    return require_paid_resource_admission(
        {
            "schema_version": "test_model_volume_admission.v1",
            "status": "admitted",
            "resource_class": "model_volume",
            "blockers": [],
        },
        resource_class="model_volume",
        expected_schema_version="test_model_volume_admission.v1",
    )


def test_us_wa_1_endpoint_is_exact() -> None:
    assert endpoint_for_data_center("us-wa-1") == "https://s3api-us-wa-1.runpod.io/"


def test_s3_endpoint_support_uses_authoritative_volume_data_centers() -> None:
    assert RUNPOD_S3_VOLUME_DATA_CENTER_IDS == {
        "EU-CZ-1",
        "EU-RO-1",
        "EUR-IS-1",
        "EUR-NO-1",
        "US-CA-2",
        "US-IL-1",
        "US-MO-2",
        "US-NC-1",
        "US-NE-1",
        "US-WA-1",
    }
    assert RUNPOD_S3_VOLUME_DATA_CENTER_IDS <= RUNPOD_NETWORK_VOLUME_DATA_CENTER_IDS
    for data_center_id in RUNPOD_S3_VOLUME_DATA_CENTER_IDS:
        assert endpoint_for_data_center(data_center_id).startswith("https://s3api-")
    for unsupported in ("AP-IN-2", "US-NC-2"):
        with pytest.raises(ValueError, match="runpod_s3_data_center_not_supported"):
            endpoint_for_data_center(unsupported)


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
        allocation_nonce=ALLOCATION_NONCE,
        client=client,
        paid_resource_admission_grant=_grant(),
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
        allocation_nonce=ALLOCATION_NONCE,
        client=FakeS3(corrupt_download=True),
        paid_resource_admission_grant=_grant(),
    )
    assert result["status"] == "failed"
    assert result["error_type"] == "RuntimeError"
    assert result["gpu_compute_allocated"] is False
    assert result["partial_upload_cleanup_verified"] is True


@pytest.mark.parametrize(
    ("client", "delete_successes", "prefix_empty", "outer_delete"),
    [
        (FakeS3(fail_upload_on=3), 2, True, False),
        (FakeS3(fail_upload_on=3, fail_delete=True), 0, False, True),
        (
            FakeS3(fail_upload_on=3, write_before_upload_failure=True),
            2,
            False,
            True,
        ),
    ],
)
def test_partial_upload_failure_accounts_for_cleanup_and_observed_state(
    tmp_path: Path,
    client: FakeS3,
    delete_successes: int,
    prefix_empty: bool,
    outer_delete: bool,
) -> None:
    access, secret = _credentials(tmp_path)
    result = upload_and_verify_model_cache(
        cache_root=_cache(tmp_path),
        verification_root=tmp_path / "redownload",
        volume_id="volume-1",
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        volume_evidence=_volume_evidence(),
        allocation_nonce=ALLOCATION_NONCE,
        client=client,
        paid_resource_admission_grant=_grant(),
    )
    assert result["status"] == "failed"
    assert result["upload_attempt_count"] == 3
    assert result["upload_success_count"] == 2
    assert result["cleanup_delete_attempt_count"] == 2
    assert result["cleanup_delete_success_count"] == delete_successes
    assert result["provider_mutations_performed"] == 5
    assert result["final_provider_observed_prefix_empty"] is prefix_empty
    assert result["outer_volume_deletion_required"] is outer_delete


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
        allocation_nonce=ALLOCATION_NONCE,
        client=FakeS3(),
        paid_resource_admission_grant=_grant(),
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
        allocation_nonce=ALLOCATION_NONCE,
        client=client,
        paid_resource_admission_grant=_grant(),
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


def test_nonempty_dedicated_volume_blocks_before_upload(tmp_path: Path) -> None:
    access, secret = _credentials(tmp_path)
    client = FakeS3()
    client.objects[("volume-1", "unrelated/old")] = b"old"
    result = upload_and_verify_model_cache(
        cache_root=_cache(tmp_path),
        verification_root=tmp_path / "redownload",
        volume_id="volume-1",
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        volume_evidence=_volume_evidence(),
        allocation_nonce=ALLOCATION_NONCE,
        client=client,
        paid_resource_admission_grant=_grant(),
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
        allocation_nonce=ALLOCATION_NONCE,
        client=client,
        paid_resource_admission_grant=_grant(),
    )
    assert result["status"] == "failed"
    assert result["partial_upload_cleanup_verified"] is False
    assert result["outer_volume_deletion_required"] is True


def test_missing_grant_blocks_before_s3_mutation(tmp_path: Path) -> None:
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
        allocation_nonce=ALLOCATION_NONCE,
        client=client,
    )
    assert result["status"] == "blocked"
    assert "paid_resource_admission_grant_missing" in result["blockers"]
    assert client.upload_calls == 0
    assert client.delete_calls == 0


def test_remote_capacity_requires_cache_bytes_plus_headroom(tmp_path: Path) -> None:
    access, secret = _credentials(tmp_path)
    cache = _cache(tmp_path)
    evidence = _volume_evidence()
    evidence["size_bytes"] = (
        verify_model_cache(cache)["verified_size_bytes"]
        + REMOTE_SAFETY_HEADROOM_BYTES
        - 1
    )
    client = FakeS3()
    result = upload_and_verify_model_cache(
        cache_root=cache,
        verification_root=tmp_path / "redownload",
        volume_id="volume-1",
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        volume_evidence=evidence,
        allocation_nonce=ALLOCATION_NONCE,
        client=client,
        paid_resource_admission_grant=_grant(),
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == [
        "runpod_rest_volume_capacity_headroom_insufficient"
    ]
    assert client.upload_calls == 0


def test_allocation_nonce_mismatch_blocks_before_s3_mutation(tmp_path: Path) -> None:
    access, secret = _credentials(tmp_path)
    evidence = _volume_evidence()
    evidence["allocation_nonce"] = "different9"
    client = FakeS3()
    result = upload_and_verify_model_cache(
        cache_root=_cache(tmp_path),
        verification_root=tmp_path / "redownload",
        volume_id="volume-1",
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        volume_evidence=evidence,
        allocation_nonce=ALLOCATION_NONCE,
        client=client,
        paid_resource_admission_grant=_grant(),
    )
    assert result["status"] == "blocked"
    assert "runpod_rest_volume_allocation_identity_mismatch" in result["blockers"]
    assert client.upload_calls == 0
    assert client.delete_calls == 0


def test_verification_parent_is_never_recursively_deleted(tmp_path: Path) -> None:
    access, secret = _credentials(tmp_path)
    verification_parent = tmp_path / "verification-job"
    verification_parent.mkdir()
    sentinel = verification_parent / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    result = upload_and_verify_model_cache(
        cache_root=_cache(tmp_path),
        verification_root=verification_parent,
        volume_id="volume-1",
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        volume_evidence=_volume_evidence(),
        allocation_nonce=ALLOCATION_NONCE,
        client=FakeS3(),
        paid_resource_admission_grant=_grant(),
    )
    assert result["status"] == "completed"
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_public_upload_cli_is_hard_disabled(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(
        [
            "upload-verify",
            "--data-center-id",
            "US-WA-1",
            "--access-key-file",
            str(tmp_path / "missing-access"),
            "--secret-key-file",
            str(tmp_path / "missing-secret"),
        ]
    ) == 2
    assert "legacy_runpod_s3_model_cache_mutation_cli_disabled" in capsys.readouterr().out
