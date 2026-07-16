"""Bounded RunPod network-volume I/O for one GR00T + OSCAR campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .groot_oscar_runpod_s3_model_cache import (
    _client,
    _remote_keys,
    _runpod_transfer_config,
    _secret_file,
)
from .groot_oscar_runpod_serverless_campaign_worker import _validate_campaign_input
from .groot_oscar_runpod_serverless_campaign_worker import (
    ARTIFACT_SCHEMA_VERSION,
    ATTEMPT_SCHEMA_VERSION,
    EXPECTED_ATTEMPTS,
    SCHEMA_VERSION as CAMPAIGN_RESULT_SCHEMA_VERSION,
)


SCHEMA_VERSION = "groot_oscar_runpod_serverless_campaign_io.v1"
EVIDENCE_SCHEMA_VERSION = "groot_oscar_runpod_serverless_campaign_io_input.v1"
_REQUIRED_LOCAL_FILE_COUNT = 6
_UPLOAD_ATTEMPTS = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_key(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or ".." in path.parts
        or any(not part or part == "." for part in path.parts)
    ):
        raise ValueError("campaign_io_relative_key_invalid")
    return path.as_posix()


def _validated_key(value: Any, blockers: list[str], blocker: str) -> str:
    try:
        return _safe_key(value)
    except ValueError:
        blockers.append(blocker)
        return ""


def validate_campaign_io_evidence(
    evidence_path: str | Path,
    *,
    source_commit: str,
    image_ref: str,
    model_manifest_digest: str,
    volume_id: str,
    data_center_id: str,
) -> dict[str, Any]:
    evidence_file = Path(evidence_path).expanduser().resolve()
    value = json.loads(evidence_file.read_text(encoding="utf-8"))
    evidence = _object(value)
    blockers: list[str] = []
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        blockers.append("campaign_io_evidence_schema_invalid")
    if evidence.get("source_commit") != source_commit:
        blockers.append("campaign_io_source_commit_mismatch")
    if evidence.get("worker_image_ref") != image_ref:
        blockers.append("campaign_io_worker_image_mismatch")
    if evidence.get("model_manifest_digest") != model_manifest_digest:
        blockers.append("campaign_io_model_manifest_mismatch")
    if evidence.get("network_volume_id") != volume_id:
        blockers.append("campaign_io_network_volume_mismatch")
    if str(evidence.get("data_center_id") or "").upper() != data_center_id.upper():
        blockers.append("campaign_io_data_center_mismatch")
    campaign_prefix = _validated_key(
        evidence.get("campaign_prefix"), blockers, "campaign_io_prefix_invalid"
    )
    if not campaign_prefix.startswith(".blueprint-campaigns/"):
        blockers.append("campaign_io_prefix_not_reserved")
    output_relative_path = _validated_key(
        evidence.get("output_relative_path"),
        blockers,
        "campaign_io_output_relative_path_invalid",
    )
    if not output_relative_path.startswith(f"{campaign_prefix}/output/"):
        blockers.append("campaign_io_output_outside_prefix")
    rows = evidence.get("files")
    rows = list(rows) if isinstance(rows, list) else []
    normalized: list[dict[str, Any]] = []
    remote_index: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = _object(raw)
        local = Path(str(row.get("local_path") or "")).expanduser().resolve()
        remote = _validated_key(
            row.get("relative_path"), blockers, "campaign_io_input_key_invalid"
        )
        expected = str(row.get("sha256") or "").lower()
        if not remote.startswith(f"{campaign_prefix}/input/"):
            blockers.append("campaign_io_input_outside_prefix")
        if not local.is_file():
            blockers.append(f"campaign_io_local_file_missing:{local.name}")
            size = 0
            actual = ""
        else:
            size = local.stat().st_size
            actual = _sha256(local)
        if actual != expected or size != int(row.get("size_bytes") or -1):
            blockers.append(f"campaign_io_local_file_integrity_failed:{local.name}")
        normalized_row = {
            "local_path": str(local),
            "relative_path": remote,
            "sha256": expected,
            "size_bytes": size,
        }
        normalized.append(normalized_row)
        if remote in remote_index:
            blockers.append("campaign_io_duplicate_remote_key")
        remote_index[remote] = normalized_row
    if len(normalized) != _REQUIRED_LOCAL_FILE_COUNT:
        blockers.append("campaign_io_requires_manifest_bundle_and_four_attempts")
    campaign_ref = _object(evidence.get("campaign_manifest"))
    campaign_key = _validated_key(
        campaign_ref.get("relative_path"),
        blockers,
        "campaign_io_manifest_key_invalid",
    )
    campaign_row = remote_index.get(campaign_key)
    if campaign_row is None or campaign_row.get("sha256") != campaign_ref.get("sha256"):
        blockers.append("campaign_io_manifest_file_reference_invalid")
        campaign_payload: dict[str, Any] = {}
    else:
        try:
            campaign_payload = _object(
                json.loads(
                    Path(campaign_row["local_path"]).read_text(encoding="utf-8")
                )
            )
            _validate_campaign_input(
                campaign_payload,
                source_commit=source_commit,
                image_ref=image_ref,
                model_manifest_digest=model_manifest_digest,
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            campaign_payload = {}
            blockers.append("campaign_io_campaign_manifest_contract_invalid")
    referenced: dict[str, str] = {}
    bundle = _object(campaign_payload.get("payload_bundle"))
    if bundle:
        bundle_key = _validated_key(
            bundle.get("relative_path"), blockers, "campaign_io_bundle_key_invalid"
        )
        if bundle_key:
            referenced[bundle_key] = str(bundle.get("sha256") or "")
    attempts = campaign_payload.get("attempts")
    for raw in attempts if isinstance(attempts, list) else []:
        attempt_ref = _object(_object(raw).get("attempt_manifest"))
        attempt_key = _validated_key(
            attempt_ref.get("relative_path"),
            blockers,
            "campaign_io_attempt_key_invalid",
        )
        if attempt_key:
            referenced[attempt_key] = str(attempt_ref.get("sha256") or "")
    for remote, expected in referenced.items():
        row = remote_index.get(remote)
        if row is None or row.get("sha256") != expected:
            blockers.append("campaign_io_referenced_input_missing_or_mismatched")
    if set(remote_index) != {campaign_key, *referenced.keys()}:
        blockers.append("campaign_io_unreferenced_input_file")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "campaign_prefix": campaign_prefix,
        "campaign_manifest_relative_path": campaign_key,
        "campaign_manifest_sha256": campaign_ref.get("sha256"),
        "output_relative_path": output_relative_path,
        "files": normalized,
        "network_volume_id": volume_id,
        "data_center_id": data_center_id,
        "raw_secret_values_recorded": False,
    }


def _credentials(
    access_key_file: str | Path, secret_key_file: str | Path
) -> tuple[str, str, dict[str, Any]]:
    access, access_status = _secret_file(access_key_file, label="runpod_s3_access_key")
    secret, secret_status = _secret_file(secret_key_file, label="runpod_s3_secret_key")
    blockers = [*access_status["blockers"], *secret_status["blockers"]]
    if blockers:
        raise ValueError(",".join(blockers))
    return access, secret, {
        "access_key": access_status,
        "secret_key": secret_status,
        "raw_secret_values_recorded": False,
    }


def _delete_keys(client: Any, *, volume_id: str, keys: list[str]) -> None:
    """Delete keys using the operation RunPod's S3 API actually supports."""

    for key in keys:
        client.delete_object(Bucket=volume_id, Key=key)


def _upload_file_with_recovery(
    client: Any,
    *,
    local_path: str,
    volume_id: str,
    relative_path: str,
    size_bytes: int,
    transfer: Any,
) -> bool:
    """Retry transient uploads and accept an ambiguous completion only by size."""

    for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
        try:
            client.upload_file(
                local_path,
                volume_id,
                relative_path,
                Config=transfer,
            )
        except Exception:
            try:
                head = client.head_object(Bucket=volume_id, Key=relative_path)
            except Exception:
                if attempt == _UPLOAD_ATTEMPTS:
                    raise
                continue
            if int(head.get("ContentLength") or -1) == size_bytes:
                return True
            if attempt == _UPLOAD_ATTEMPTS:
                raise
            continue
        head = client.head_object(Bucket=volume_id, Key=relative_path)
        if int(head.get("ContentLength") or -1) != size_bytes:
            raise RuntimeError("campaign_io_uploaded_size_mismatch")
        return False
    raise RuntimeError("campaign_io_upload_attempts_exhausted")


def stage_campaign_inputs(
    contract: Mapping[str, Any],
    *,
    access_key_file: str | Path,
    secret_key_file: str | Path,
) -> dict[str, Any]:
    if contract.get("status") != "passed":
        raise ValueError("campaign_io_contract_not_passed")
    access, secret, credential_status = _credentials(access_key_file, secret_key_file)
    volume_id = str(contract["network_volume_id"])
    prefix = str(contract["campaign_prefix"])
    client = _client(
        data_center_id=str(contract["data_center_id"]),
        access_key=access,
        secret_key=secret,
    )
    existing = _remote_keys(client, volume_id=volume_id, prefix=f"{prefix}/")
    expected_inputs = {
        str(row["relative_path"]): int(row["size_bytes"]) for row in contract["files"]
    }
    if existing:
        if any(key not in expected_inputs for key in existing):
            raise ValueError("campaign_io_remote_prefix_contains_unowned_artifacts")
        for key in existing:
            head = client.head_object(Bucket=volume_id, Key=key)
            if int(head.get("ContentLength") or -1) != expected_inputs[key]:
                raise ValueError("campaign_io_stale_input_size_mismatch")
        _delete_keys(client, volume_id=volume_id, keys=existing)
        if _remote_keys(client, volume_id=volume_id, prefix=f"{prefix}/"):
            raise RuntimeError("campaign_io_stale_input_cleanup_failed")
    transfer = _runpod_transfer_config()
    uploaded: list[dict[str, Any]] = []
    ambiguous_upload_completions = 0
    try:
        for row in contract["files"]:
            recovered = _upload_file_with_recovery(
                client,
                local_path=str(row["local_path"]),
                volume_id=volume_id,
                relative_path=str(row["relative_path"]),
                size_bytes=int(row["size_bytes"]),
                transfer=transfer,
            )
            ambiguous_upload_completions += int(recovered)
            uploaded.append(
                {
                    "relative_path": row["relative_path"],
                    "size_bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                }
            )
    except Exception:
        if uploaded:
            _delete_keys(
                client,
                volume_id=volume_id,
                keys=[str(row["relative_path"]) for row in uploaded],
            )
        if _remote_keys(client, volume_id=volume_id, prefix=f"{prefix}/"):
            raise RuntimeError("campaign_io_partial_upload_cleanup_failed")
        raise
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "uploaded_file_count": len(uploaded),
        "deleted_stale_input_file_count": len(existing),
        "ambiguous_upload_completion_count": ambiguous_upload_completions,
        "uploaded": uploaded,
        "credential_status": credential_status,
        "raw_secret_values_recorded": False,
    }


def retrieve_campaign_outputs(
    contract: Mapping[str, Any],
    *,
    destination: str | Path,
    access_key_file: str | Path,
    secret_key_file: str | Path,
) -> dict[str, Any]:
    access, secret, credential_status = _credentials(access_key_file, secret_key_file)
    volume_id = str(contract["network_volume_id"])
    output_prefix = str(contract["output_relative_path"]).rstrip("/")
    client = _client(
        data_center_id=str(contract["data_center_id"]),
        access_key=access,
        secret_key=secret,
    )
    keys = _remote_keys(client, volume_id=volume_id, prefix=f"{output_prefix}/")
    root = Path(destination).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("campaign_io_download_destination_not_empty")
    root.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, Any]] = []
    transfer_blockers: list[str] = []
    for key in keys:
        relative = PurePosixPath(key).relative_to(PurePosixPath(output_prefix))
        target = (root / Path(*relative.parts)).resolve()
        if target != root and root not in target.parents:
            raise ValueError("campaign_io_download_path_escape")
        target.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(volume_id, key, str(target))
        head = client.head_object(Bucket=volume_id, Key=key)
        if target.stat().st_size != int(head.get("ContentLength") or -1):
            transfer_blockers.append("campaign_io_downloaded_size_mismatch")
        downloaded.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    artifact_path = root / "campaign_artifact_manifest.json"
    result_path = root / "campaign_result.json"
    blockers: list[str] = list(transfer_blockers)
    if not artifact_path.is_file():
        blockers.append("campaign_artifact_manifest_missing")
        artifact = {}
    else:
        artifact = _object(json.loads(artifact_path.read_text(encoding="utf-8")))
        if (
            artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION
            or artifact.get("status") != "completed"
        ):
            blockers.append("campaign_artifact_manifest_schema_or_status_invalid")
    if not result_path.is_file():
        blockers.append("campaign_result_missing")
        campaign_result: dict[str, Any] = {}
    else:
        campaign_result = _object(json.loads(result_path.read_text(encoding="utf-8")))
        runs = campaign_result.get("runs")
        runs = list(runs) if isinstance(runs, list) else []
        if not (
            campaign_result.get("schema_version") == CAMPAIGN_RESULT_SCHEMA_VERSION
            and campaign_result.get("status") == "completed"
            and campaign_result.get("smoke_passed") is True
            and campaign_result.get("all_dynamic_episodes_completed") is True
            and [str(_object(row).get("attempt_id") or "") for row in runs]
            == [row[0] for row in EXPECTED_ATTEMPTS]
        ):
            blockers.append("campaign_result_schema_or_structure_invalid")
    expected_rows = artifact.get("files")
    expected_rows = list(expected_rows) if isinstance(expected_rows, list) else []
    if (
        int(artifact.get("file_count") or -1) != len(expected_rows)
        or int(artifact.get("total_size_bytes") or -1)
        != sum(int(_object(row).get("size_bytes") or 0) for row in expected_rows)
    ):
        blockers.append("campaign_artifact_manifest_aggregate_invalid")
    actual = {row["relative_path"]: row for row in downloaded}
    for raw in expected_rows:
        row = _object(raw)
        found = actual.get(str(row.get("relative_path") or ""))
        if (
            found is None
            or found["sha256"] != row.get("sha256")
            or found["size_bytes"] != row.get("size_bytes")
        ):
            blockers.append("campaign_artifact_hash_or_size_mismatch")
            break
    expected_paths = {str(_object(row).get("relative_path") or "") for row in expected_rows}
    if expected_paths and set(actual) != {
        *expected_paths,
        "campaign_artifact_manifest.json",
    }:
        blockers.append("campaign_artifact_file_set_mismatch")
    for attempt_id, _kind, _seed, _timeout in EXPECTED_ATTEMPTS:
        attempt_path = root / attempt_id / "attempt_result.json"
        if not attempt_path.is_file():
            blockers.append(f"campaign_attempt_result_missing:{attempt_id}")
            continue
        attempt_result = _object(json.loads(attempt_path.read_text(encoding="utf-8")))
        if (
            attempt_result.get("schema_version") != ATTEMPT_SCHEMA_VERSION
            or attempt_result.get("attempt_id") != attempt_id
            or attempt_result.get("status") != "completed"
        ):
            blockers.append(f"campaign_attempt_result_schema_invalid:{attempt_id}")
    json_decode_failures = []
    for path in root.rglob("*.json"):
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            json_decode_failures.append(path.relative_to(root).as_posix())
            continue
        if not isinstance(decoded, Mapping):
            json_decode_failures.append(path.relative_to(root).as_posix())
    if json_decode_failures:
        blockers.append("campaign_json_artifact_decode_failed")
    transfer_completed = not transfer_blockers and len(downloaded) == len(keys)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not blockers else "blocked",
        "transfer_status": "completed" if transfer_completed else "blocked",
        "blockers": sorted(set(blockers)),
        "downloaded_file_count": len(downloaded),
        "downloaded_total_size_bytes": sum(row["size_bytes"] for row in downloaded),
        "downloaded": downloaded,
        "artifact_manifest_schema_version": artifact.get("schema_version"),
        "campaign_result_schema_version": campaign_result.get("schema_version"),
        "json_decode_failure_count": len(json_decode_failures),
        "destination": str(root),
        "credential_status": credential_status,
        "raw_secret_values_recorded": False,
    }


def cleanup_campaign_storage(
    contract: Mapping[str, Any],
    *,
    access_key_file: str | Path,
    secret_key_file: str | Path,
) -> dict[str, Any]:
    access, secret, credential_status = _credentials(access_key_file, secret_key_file)
    volume_id = str(contract["network_volume_id"])
    prefix = str(contract["campaign_prefix"])
    client = _client(
        data_center_id=str(contract["data_center_id"]),
        access_key=access,
        secret_key=secret,
    )
    keys = _remote_keys(client, volume_id=volume_id, prefix=f"{prefix}/")
    _delete_keys(client, volume_id=volume_id, keys=keys)
    remaining = _remote_keys(client, volume_id=volume_id, prefix=f"{prefix}/")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not remaining else "blocked",
        "deleted_file_count": len(keys),
        "remaining_file_count": len(remaining),
        "credential_status": credential_status,
        "raw_secret_values_recorded": False,
    }
