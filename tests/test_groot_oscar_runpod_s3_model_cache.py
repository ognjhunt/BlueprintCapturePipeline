from __future__ import annotations

from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import inspect
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import blueprint_pipeline.groot_oscar_runpod_s3_model_cache as s3_transport

from blueprint_pipeline.common import write_json
from blueprint_pipeline.groot_oscar_infrastructure_admission import (
    RUNPOD_NETWORK_VOLUME_DATA_CENTER_IDS,
)
from blueprint_pipeline.groot_oscar_model_cache import (
    COSMOS_RUNTIME_MODEL_RELATIVE_PATH,
    COSMOS_SELECTOR_ANCHOR_RELATIVE_PATH,
    MANIFEST_NAME,
    REQUIRED_MODEL_FILES,
    build_manifest,
    verify_model_cache,
)
from blueprint_pipeline.groot_oscar_runpod_s3_model_cache import (
    DEFAULT_REMOTE_PREFIX,
    REMOTE_SAFETY_HEADROOM_BYTES,
    RUNPOD_S3_MAX_CONCURRENCY,
    RUNPOD_S3_COMPLETE_MULTIPART_MAX_ATTEMPTS,
    RUNPOD_S3_AMBIGUOUS_COMPLETE_HEAD_INTERVAL_SECONDS,
    RUNPOD_S3_AMBIGUOUS_COMPLETE_HEAD_MAX_ATTEMPTS,
    RUNPOD_S3_MULTIPART_CHUNK_BYTES,
    RUNPOD_S3_MULTIPART_THRESHOLD_BYTES,
    RUNPOD_S3_VOLUME_DATA_CENTER_IDS,
    _TransportExecutionCapability,
    _client,
    _issue_transport_execution_capability,
    _runpod_transfer_contract,
    _retry_runpod_complete_multipart_gateway_timeout,
    _sanitized_s3_exception,
    _upload_file_with_ambiguous_completion_recovery,
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
_REAL_RUNPOD_TRANSFER_CONFIG_FACTORY = s3_transport._runpod_transfer_config


@pytest.fixture(autouse=True)
def _large_test_filesystem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        s3_transport,
        "_filesystem_available_bytes",
        lambda _path: 100 * 1024**3,
    )
    monkeypatch.setattr(
        s3_transport,
        "_runpod_transfer_config",
        lambda: SimpleNamespace(
            multipart_threshold=RUNPOD_S3_MULTIPART_THRESHOLD_BYTES,
            multipart_chunksize=RUNPOD_S3_MULTIPART_CHUNK_BYTES,
            max_concurrency=RUNPOD_S3_MAX_CONCURRENCY,
            use_threads=False,
        ),
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
        fail_multipart_listing: bool = False,
        fail_multipart_listing_on: int | None = None,
        visible_multipart_uploads: int = 0,
    ) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.corrupt_download = corrupt_download
        self.fail_cleanup_list = fail_cleanup_list
        self.fail_upload_on = fail_upload_on
        self.write_before_upload_failure = write_before_upload_failure
        self.fail_delete = fail_delete
        self.fail_multipart_listing = fail_multipart_listing
        self.fail_multipart_listing_on = fail_multipart_listing_on
        self.visible_multipart_uploads = visible_multipart_uploads
        self.upload_calls = 0
        self.delete_calls = 0
        self.abort_calls = 0
        self.multipart_list_calls = 0
        self.multipart_list_kwargs: list[dict[str, object]] = []
        self.upload_configs: list[object] = []
        self.head_calls = 0
        self.get_calls = 0

    def list_buckets(self):  # type: ignore[no-untyped-def]
        return {"Buckets": [{"Name": "volume-1"}]}

    def head_bucket(self, *, Bucket: str) -> None:  # noqa: N803
        assert Bucket == "volume-1"

    def upload_file(
        self,
        path: str,
        bucket: str,
        key: str,
        *,
        Config: object | None = None,  # noqa: N803
    ) -> None:
        self.upload_calls += 1
        self.upload_configs.append(Config)
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

    def list_multipart_uploads(self, **kwargs):  # type: ignore[no-untyped-def]
        self.multipart_list_calls += 1
        self.multipart_list_kwargs.append(dict(kwargs))
        if self.fail_multipart_listing or (
            self.fail_multipart_listing_on == self.multipart_list_calls
        ):
            raise RuntimeError("multipart listing unavailable")
        return {
            "Uploads": [
                {"Key": f"multipart-{index}", "UploadId": f"upload-{index}"}
                for index in range(self.visible_multipart_uploads)
            ]
        }

    def abort_multipart_upload(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.abort_calls += 1
        self.visible_multipart_uploads = max(0, self.visible_multipart_uploads - 1)

    def get_object(self, *, Bucket: str, Key: str):  # noqa: N803
        self.get_calls += 1
        payload = self.objects[(Bucket, Key)]
        if self.corrupt_download and Key.endswith("config.json"):
            payload += b"corrupt"
        return {"Body": BytesIO(payload)}

    def head_object(self, *, Bucket: str, Key: str):  # noqa: N803
        self.head_calls += 1
        return {"ContentLength": len(self.objects[(Bucket, Key)])}

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
    selector_anchor = root / COSMOS_SELECTOR_ANCHOR_RELATIVE_PATH
    selector_anchor.parent.mkdir(parents=True, exist_ok=True)
    selector_anchor.write_text("selector anchor\n", encoding="utf-8")
    (root / "sonic/config.json").write_text(
        json.dumps(
            {"model_name": str(root.resolve() / COSMOS_RUNTIME_MODEL_RELATIVE_PATH)}
        ),
        encoding="utf-8",
    )
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
    assert all(config is not None for config in client.upload_configs)
    assert all(
        config.multipart_threshold == RUNPOD_S3_MULTIPART_THRESHOLD_BYTES
        and config.multipart_chunksize == RUNPOD_S3_MULTIPART_CHUNK_BYTES
        and config.max_concurrency == RUNPOD_S3_MAX_CONCURRENCY
        and config.use_threads is False
        for config in client.upload_configs
    )
    assert result["upload_transfer_contract"] == {
        "multipart_threshold_bytes": RUNPOD_S3_MULTIPART_THRESHOLD_BYTES,
        "multipart_chunk_bytes": RUNPOD_S3_MULTIPART_CHUNK_BYTES,
        "max_concurrency": RUNPOD_S3_MAX_CONCURRENCY,
        "use_threads": False,
        "client_retry_mode": "standard",
        "client_max_attempts": 10,
        "complete_multipart_retry_http_statuses": [524],
        "complete_multipart_max_attempts": 4,
        "ambiguous_complete_head_max_attempts": 30,
        "ambiguous_complete_head_interval_seconds": 2.0,
    }
    assert result["multipart_cleanup_required"] is False
    assert result["multipart_listing_supported"] is True
    assert result["multipart_absence_verified"] is True
    assert client.multipart_list_calls == 4
    assert client.multipart_list_kwargs == [{"Bucket": "volume-1"}] * 4


def test_upload_verifies_runtime_artifacts_separately_from_model_manifest(
    tmp_path: Path,
) -> None:
    access, secret = _credentials(tmp_path)
    archive = tmp_path / "groot_oscar_runtime.tar.gz"
    archive.write_bytes(b"runtime archive bytes")
    runtime_manifest = tmp_path / "runtime_bundle_manifest.json"
    runtime_manifest.write_text('{"schema_version":"runtime.v1"}\n', encoding="utf-8")
    artifacts = [
        {
            "local_path": str(archive),
            "remote_key": ".blueprint-runtime/blueprint-groot-oscar-v1/"
            "groot_oscar_runtime.tar.gz",
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        },
        {
            "local_path": str(runtime_manifest),
            "remote_key": ".blueprint-runtime/blueprint-groot-oscar-v1/"
            "runtime_bundle_manifest.json",
            "sha256": hashlib.sha256(runtime_manifest.read_bytes()).hexdigest(),
        },
    ]

    result = upload_and_verify_model_cache(
        cache_root=_cache(tmp_path),
        verification_root=tmp_path / "redownload",
        volume_id="volume-1",
        data_center_id="US-WA-1",
        access_key_file=access,
        secret_key_file=secret,
        volume_evidence=_volume_evidence(),
        allocation_nonce=ALLOCATION_NONCE,
        client=FakeS3(),
        paid_resource_admission_grant=_grant(),
        additional_artifacts=artifacts,
    )

    assert result["status"] == "completed"
    assert result["additional_artifact_count"] == 2
    assert result["additional_artifact_verification_method"] == (
        "full_s3_redownload_and_sha256"
    )
    assert all(
        row["full_redownload_sha256_verified"] is True
        for row in result["additional_artifact_verification"]
    )
    assert result["model_manifest_digest"] == result["remote_model_manifest_digest"]


def test_additional_artifact_contract_blocks_before_s3_mutation(tmp_path: Path) -> None:
    access, secret = _credentials(tmp_path)
    artifact = tmp_path / "runtime.tar.gz"
    artifact.write_bytes(b"runtime")
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
        additional_artifacts=(
            {
                "local_path": str(artifact),
                "remote_key": "../escape",
                "sha256": "0" * 64,
            },
        ),
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["runpod_s3_additional_artifact_contract_invalid"]
    assert result["provider_mutations_performed"] == 0
    assert client.upload_calls == 0


def test_runpod_transfer_contract_is_large_chunked_and_single_threaded() -> None:
    assert _runpod_transfer_contract() == {
        "multipart_threshold_bytes": 64 * 1024**2,
        "multipart_chunk_bytes": 128 * 1024**2,
        "max_concurrency": 1,
        "use_threads": False,
        "client_retry_mode": "standard",
        "client_max_attempts": 10,
        "complete_multipart_retry_http_statuses": [524],
        "complete_multipart_max_attempts": 4,
        "ambiguous_complete_head_max_attempts": 30,
        "ambiguous_complete_head_interval_seconds": 2.0,
    }


def test_runpod_client_registers_bounded_complete_multipart_524_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registrations: list[tuple[str, object, str]] = []
    captured: dict[str, object] = {}

    class Events:
        def register(self, name: str, handler: object, *, unique_id: str) -> None:
            registrations.append((name, handler, unique_id))

    fake_client = SimpleNamespace(meta=SimpleNamespace(events=Events()))

    def client(name: str, **kwargs: object) -> object:
        captured.update({"name": name, **kwargs})
        return fake_client

    class Config:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    boto3_module = ModuleType("boto3")
    boto3_module.client = client  # type: ignore[attr-defined]
    botocore_module = ModuleType("botocore")
    botocore_module.__path__ = []  # type: ignore[attr-defined]
    config_module = ModuleType("botocore.config")
    config_module.Config = Config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3_module)
    monkeypatch.setitem(sys.modules, "botocore", botocore_module)
    monkeypatch.setitem(sys.modules, "botocore.config", config_module)

    result = _client(
        data_center_id="US-WA-1", access_key="access", secret_key="secret"
    )

    assert result is fake_client
    assert captured["name"] == "s3"
    assert captured["endpoint_url"] == "https://s3api-us-wa-1.runpod.io/"
    assert len(registrations) == 1
    name, handler, unique_id = registrations[0]
    assert name == "needs-retry.s3.CompleteMultipartUpload"
    assert unique_id == "blueprint-runpod-complete-multipart-524"
    assert isinstance(handler, partial)
    assert handler.func is _retry_runpod_complete_multipart_gateway_timeout
    assert handler.keywords["retry_state"] is result._blueprint_complete_multipart_retry_state


@pytest.mark.parametrize(
    ("status", "attempts", "expected"),
    [
        (524, 1, 1.0),
        (524, 2, 2.0),
        (524, 3, 4.0),
        (524, RUNPOD_S3_COMPLETE_MULTIPART_MAX_ATTEMPTS, None),
        (504, 1, None),
        (524, None, None),
    ],
)
def test_complete_multipart_retry_is_exact_and_bounded(
    status: int, attempts: int | None, expected: float | None
) -> None:
    response = (SimpleNamespace(status_code=status), {})
    assert (
        _retry_runpod_complete_multipart_gateway_timeout(
            response=response, attempts=attempts
        )
        == expected
    )


def test_524_then_invalid_part_recovers_only_after_exact_materialization(
    tmp_path: Path,
) -> None:
    class InvalidPart(RuntimeError):
        operation_name = "CompleteMultipartUpload"
        response = {
            "Error": {"Code": "InvalidPart"},
            "ResponseMetadata": {"HTTPStatusCode": 400, "RetryAttempts": 1},
        }

    class AmbiguousS3(FakeS3):
        def __init__(self) -> None:
            super().__init__()
            self._blueprint_complete_multipart_retry_state = {
                "gateway_timeout_observed": False,
                "retry_attempt_count": 0,
            }

        def upload_file(
            self,
            path: str,
            bucket: str,
            key: str,
            *,
            Config: object | None = None,  # noqa: N803
        ) -> None:
            self.upload_calls += 1
            self.upload_configs.append(Config)
            self.objects[(bucket, key)] = Path(path).read_bytes()
            self._blueprint_complete_multipart_retry_state.update(
                gateway_timeout_observed=True,
                retry_attempt_count=1,
            )
            wrapper = RuntimeError("upload failed")
            wrapper.__cause__ = InvalidPart("provider detail")
            raise wrapper

    path = tmp_path / "large.bin"
    path.write_bytes(b"verified later by full redownload")
    client = AmbiguousS3()

    recovered = _upload_file_with_ambiguous_completion_recovery(
        client,
        path=path,
        volume_id="volume-1",
        key="cache/large.bin",
        transfer_config=object(),
        sleeper=lambda _seconds: None,
    )

    assert recovered is True
    assert client.head_calls == 1


def test_invalid_part_without_observed_524_is_not_recovered(tmp_path: Path) -> None:
    class InvalidPart(RuntimeError):
        operation_name = "CompleteMultipartUpload"
        response = {
            "Error": {"Code": "InvalidPart"},
            "ResponseMetadata": {"HTTPStatusCode": 400, "RetryAttempts": 0},
        }

    class InvalidPartS3(FakeS3):
        _blueprint_complete_multipart_retry_state = {
            "gateway_timeout_observed": False,
            "retry_attempt_count": 0,
        }

        def upload_file(self, *_args: object, **_kwargs: object) -> None:
            raise InvalidPart("provider detail")

    path = tmp_path / "large.bin"
    path.write_bytes(b"payload")

    with pytest.raises(InvalidPart):
        _upload_file_with_ambiguous_completion_recovery(
            InvalidPartS3(),
            path=path,
            volume_id="volume-1",
            key="cache/large.bin",
            transfer_config=object(),
            sleeper=lambda _seconds: None,
        )

    assert RUNPOD_S3_AMBIGUOUS_COMPLETE_HEAD_MAX_ATTEMPTS == 30
    assert RUNPOD_S3_AMBIGUOUS_COMPLETE_HEAD_INTERVAL_SECONDS == 2.0


@pytest.mark.parametrize(
    ("corrupt_download", "expected_status"),
    [(False, "completed"), (True, "failed")],
)
def test_ambiguous_completion_still_requires_full_redownload_sha256(
    tmp_path: Path,
    corrupt_download: bool,
    expected_status: str,
) -> None:
    class InvalidPart(RuntimeError):
        operation_name = "CompleteMultipartUpload"
        response = {
            "Error": {"Code": "InvalidPart"},
            "ResponseMetadata": {"HTTPStatusCode": 400, "RetryAttempts": 1},
        }

    class OneAmbiguousUploadS3(FakeS3):
        def __init__(self) -> None:
            super().__init__(corrupt_download=corrupt_download)
            self._blueprint_complete_multipart_retry_state = {
                "gateway_timeout_observed": False,
                "retry_attempt_count": 0,
            }

        def upload_file(
            self,
            path: str,
            bucket: str,
            key: str,
            *,
            Config: object | None = None,  # noqa: N803
        ) -> None:
            self.upload_calls += 1
            self.upload_configs.append(Config)
            self.objects[(bucket, key)] = Path(path).read_bytes()
            if self.upload_calls == 1:
                self._blueprint_complete_multipart_retry_state.update(
                    gateway_timeout_observed=True,
                    retry_attempt_count=1,
                )
                wrapper = RuntimeError("upload failed")
                wrapper.__cause__ = InvalidPart("provider detail")
                raise wrapper

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
        client=OneAmbiguousUploadS3(),
        paid_resource_admission_grant=_grant(),
    )

    assert result["status"] == expected_status
    if expected_status == "completed":
        assert result["ambiguous_complete_recovery_count"] == 1
        assert result["ambiguous_completion_size_is_integrity_proof"] is False
        assert result["ambiguous_completion_requires_full_redownload_sha256"] is True
        assert result["verification_method"] == (
            "full_s3_redownload_and_sha256_manifest_verification"
        )
    else:
        assert result["partial_upload_cleanup_verified"] is True


def test_runpod_transfer_config_passes_exact_constructor_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_transfer_config(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    boto3_module = ModuleType("boto3")
    boto3_module.__path__ = []  # type: ignore[attr-defined]
    s3_module = ModuleType("boto3.s3")
    s3_module.__path__ = []  # type: ignore[attr-defined]
    transfer_module = ModuleType("boto3.s3.transfer")
    transfer_module.TransferConfig = fake_transfer_config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3_module)
    monkeypatch.setitem(sys.modules, "boto3.s3", s3_module)
    monkeypatch.setitem(sys.modules, "boto3.s3.transfer", transfer_module)

    config = _REAL_RUNPOD_TRANSFER_CONFIG_FACTORY()

    assert captured == {
        "multipart_threshold": 64 * 1024**2,
        "multipart_chunksize": 128 * 1024**2,
        "max_concurrency": 1,
        "use_threads": False,
    }
    assert config.multipart_chunksize == 128 * 1024**2


def test_missing_transfer_config_blocks_before_any_s3_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access, secret = _credentials(tmp_path)
    client = FakeS3()

    def unavailable() -> object:
        raise RuntimeError("boto3_required_for_runpod_s3")

    monkeypatch.setattr(s3_transport, "_runpod_transfer_config", unavailable)
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
    assert result["blockers"] == ["runpod_s3_transfer_config_unavailable"]
    assert result["error_type"] == "RuntimeError"
    assert result["provider_mutations_performed"] == 0
    assert result["outer_volume_deletion_required"] is True
    assert client.multipart_list_calls == 0
    assert client.upload_calls == 0
    assert client.delete_calls == 0


def test_s3_failure_evidence_preserves_only_sanitized_provider_metadata() -> None:
    class ProviderError(RuntimeError):
        operation_name = "UploadPart"
        response = {
            "Error": {"Code": "AccessDenied", "Message": "secret-bearing detail"},
            "ResponseMetadata": {"HTTPStatusCode": 403, "RetryAttempts": 10},
        }

    provider_error = ProviderError("secret-bearing detail")
    wrapper = RuntimeError("secret-bearing wrapper")
    wrapper.__cause__ = provider_error

    assert _sanitized_s3_exception(wrapper) == {
        "error_type": "RuntimeError",
        "error_code": "AccessDenied",
        "error_operation": "UploadPart",
        "error_http_status": 403,
        "error_retry_attempts": 10,
    }
    assert "secret-bearing" not in json.dumps(_sanitized_s3_exception(wrapper))


def test_visible_multipart_abort_is_counted_as_provider_mutation(tmp_path: Path) -> None:
    access, secret = _credentials(tmp_path)
    client = FakeS3(fail_upload_on=3, visible_multipart_uploads=1)
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
    assert result["multipart_listing_supported"] is True
    assert result["multipart_abort_attempt_count"] == 1
    assert result["multipart_abort_success_count"] == 1
    assert result["multipart_absence_verified"] is True
    assert result["provider_mutations_performed"] == 6
    assert result["partial_upload_cleanup_verified"] is True


def test_visible_stale_multipart_is_aborted_before_successful_upload(
    tmp_path: Path,
) -> None:
    access, secret = _credentials(tmp_path)
    client = FakeS3(visible_multipart_uploads=1)
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
    assert result["multipart_abort_attempt_count"] == 1
    assert result["multipart_abort_success_count"] == 1
    assert result["multipart_absence_verified"] is True
    assert result["provider_mutations_performed"] == result["remote_object_count"] + 1


def test_unsupported_multipart_listing_requires_outer_volume_deletion(
    tmp_path: Path,
) -> None:
    access, secret = _credentials(tmp_path)
    client = FakeS3(fail_upload_on=3, fail_multipart_listing_on=3)
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
    assert result["multipart_listing_supported"] is False
    assert result["multipart_absence_verified"] is False
    assert result["partial_upload_cleanup_verified"] is False
    assert result["outer_volume_deletion_required"] is True


def test_unverified_initial_multipart_state_blocks_before_upload(
    tmp_path: Path,
) -> None:
    access, secret = _credentials(tmp_path)
    client = FakeS3(fail_multipart_listing=True)
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
    assert result["blockers"] == [
        "runpod_s3_dedicated_volume_multipart_state_unverified"
    ]
    assert result["provider_mutations_performed"] == 0
    assert result["outer_volume_deletion_required"] is True
    assert client.upload_calls == 0


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


def test_full_redownload_uses_get_object_without_head(tmp_path: Path) -> None:
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
    assert client.get_calls == result["remote_object_count"]
    assert client.head_calls == 0


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
