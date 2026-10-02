"""Bounded registered checkpoint transfer kept separate from the generic object store."""
from __future__ import annotations

from blueprint_pipeline.s3_compatible_transport import s3_compatible_client

import hashlib
from pathlib import Path

from . import wam_provider_object_store as wam
from .wam_provider_object_store import (
    DEFAULT_ACCESS_KEY_FILES, DEFAULT_SECRET_KEY_FILES, DEFAULT_ENDPOINT_FILES,
    DEFAULT_BUCKET_FILES, DEFAULT_REGION_FILES, RUNTIME_DEPENDENCY_URL_FILENAME,
    RUNTIME_DEPENDENCY_CACHE_SCHEMA_VERSION, _sha256_file, ensure_dir, _mapping,
    _string, _write_sensitive_file, _file_status, utc_now_iso, write_json,
)

def _registered_call(use, method, **kwargs):
    use.check()
    result = method(**kwargs)
    use.check()
    return result


def _upload_registered_checkpoint_file(client, *, bucket, key, expected, row, use):
    """One proven payload FD; synchronous SDK calls receive bounded bytes only."""
    from .control_plane_registered_checkpoint_cache import NeededCheckpointCacheError
    use.check()
    upload_id, stream, parts, buffer = None, None, [], bytearray()
    digest, total, completed = hashlib.sha256(), 0, False
    try:
        created = client.create_multipart_upload(Bucket=bucket, Key=key, Metadata={"sha256": expected})
        if (type(created) is not dict or len(created) > 64 or type(created.get("UploadId")) is not str
                or not 0 < len(created["UploadId"].encode()) <= 4096):
            raise NeededCheckpointCacheError("needed_cache_upload_identity_invalid")
        # Retain the bounded native handle BEFORE current authority can refuse.
        # This selects cleanup only, never permission to send another byte.
        upload_id = created["UploadId"]
        use.check()
        stream = use.chunks(use._root / row["relative_path"], role="upload")
        for chunk in stream:
            digest.update(chunk)
            total += len(chunk)
            buffer.extend(chunk)
            if len(buffer) == 8 * 1024 * 1024 or total == row["size_bytes"]:
                body = bytes(buffer)
                buffer.clear()
                response = _registered_call(use, client.upload_part, Bucket=bucket, Key=key,
                    UploadId=upload_id, PartNumber=len(parts)+1, Body=body)
                if (type(response) is not dict or len(response) > 64 or type(response.get("ETag")) is not str
                        or not 0 < len(response["ETag"].encode()) <= 1024):
                    raise NeededCheckpointCacheError("needed_cache_upload_part_identity_invalid")
                parts.append({"PartNumber": len(parts)+1, "ETag": response["ETag"]})
                if len(parts) > (row["size_bytes"]+8*1024*1024-1)//(8*1024*1024):
                    raise NeededCheckpointCacheError("needed_cache_upload_part_limit")
        if total != row["size_bytes"] or digest.hexdigest() != expected or buffer:
            raise NeededCheckpointCacheError("needed_cache_upload_payload_changed")
        _registered_call(use, client.complete_multipart_upload, Bucket=bucket, Key=key, UploadId=upload_id,
                         MultipartUpload={"Parts": parts})
        completed = True
    finally:
        # Close/join owned local work even after a sticky current-authority refusal.
        try:
            if stream is not None:
                stream.close()
        except Exception:
            use._failure = use._failure or "needed_cache_upload_cleanup_unresolved"
            raise
        finally:
            if upload_id is not None and not completed:
                try:
                    client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
                except Exception:
                    use._failure = use._failure or "needed_cache_upload_cleanup_unresolved"
                    raise NeededCheckpointCacheError("needed_cache_upload_cleanup_unresolved") from None


def _stage_registered_runtime_dependency(*, job_dir, dependency_path, expected_sha256, key_prefix,
        expiration_seconds, generated_at, artifact_kind, use):
    """Actual WAM enrolled branch; generic None branch remains literal above."""
    from .control_plane_registered_checkpoint_cache import require_cache_use, NeededCheckpointCacheError
    require_cache_use(use)
    row = use.row(Path(dependency_path))
    use.check()
    if artifact_kind != "g1_checkpoint" or expected_sha256 != row["sha256"]:
        raise NeededCheckpointCacheError("needed_cache_dependency_unpinned")
    job = Path(job_dir)
    if not job.is_absolute() or job == use._root or use._root in job.parents:
        raise NeededCheckpointCacheError("needed_cache_staging_target_invalid")
    digest = _sha256_file(Path(dependency_path), _cache_use=use)
    use.check()
    ensure_dir(job)
    credentials = {}
    for name, env_name, paths, allow in (
        ("access_key_id", "BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID", DEFAULT_ACCESS_KEY_FILES, False),
        ("secret_access_key", "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY", DEFAULT_SECRET_KEY_FILES, False),
        ("endpoint_url", "BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL", DEFAULT_ENDPOINT_FILES, True),
        ("bucket", "BLUEPRINT_WAM_OBJECT_STORE_BUCKET", DEFAULT_BUCKET_FILES, True),
        ("region", "BLUEPRINT_WAM_OBJECT_STORE_REGION", DEFAULT_REGION_FILES, True),
    ):
        use.check()
        credentials[name] = wam._read_first_file(explicit_path=None, env_name=env_name, default_paths=paths,
                                             label="object_store_"+name, **({"allow_env_value": True} if allow else {}))
        use.check()
    prefix = key_prefix.strip("/ ") or "blueprint/wam-provider"
    key = f"{prefix}/g1-checkpoints/sha256/{digest}.bin"
    blockers, url, cache_hit, uploaded, remote = [], "", False, False, False
    client = None
    try:
        if not all(credentials[name][0] for name in ("access_key_id", "secret_access_key", "bucket")):
            blockers.append("runtime_dependency_object_store_credentials_missing")
        else:
            import boto3
            from botocore.client import Config
            options = dict(aws_access_key_id=credentials["access_key_id"][0],
                aws_secret_access_key=credentials["secret_access_key"][0],
                region_name=credentials["region"][0] or "us-east-1",
                config=Config(signature_version="s3v4", connect_timeout=45, read_timeout=45,
                              retries={"total_max_attempts": 1}))
            if credentials["endpoint_url"][0]:
                options["endpoint_url"] = credentials["endpoint_url"][0]
            use.check()
            client = s3_compatible_client(boto3, **options)
            use.check()
            bucket = credentials["bucket"][0]
            try:
                head = _registered_call(use, client.head_object, Bucket=bucket, Key=key)
                cache_hit = True
            except NeededCheckpointCacheError:
                raise
            except Exception as exc:
                response = _mapping(getattr(exc, "response", {}))
                status = _mapping(response.get("ResponseMetadata")).get("HTTPStatusCode")
                code = _string(_mapping(response.get("Error")).get("Code"))
                if status != 404 and code.lower() not in {"404", "nosuchkey", "notfound"}:
                    raise
                _upload_registered_checkpoint_file(client, bucket=bucket, key=key, expected=digest, row=row, use=use)
                uploaded = True
                head = _registered_call(use, client.head_object, Bucket=bucket, Key=key)
            remote = (type(head) is dict and len(head) <= 64 and type(head.get("ContentLength")) is int
                      and head["ContentLength"] == row["size_bytes"]
                      and _mapping(head.get("Metadata")).get("sha256") == digest)
            if not remote:
                blockers.append("runtime_dependency_remote_identity_mismatch")
            else:
                use.check()
                url = client.generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key,
                    "ResponseCacheControl": "no-store, max-age=0"}, ExpiresIn=int(expiration_seconds), HttpMethod="GET")
                use.check()
    except Exception as exc:
        blockers.append("runtime_dependency_cache_failed:"+type(exc).__name__)
        if isinstance(exc, NeededCheckpointCacheError) and str(exc) == "needed_cache_upload_cleanup_unresolved":
            blockers.append("runtime_dependency_upload_cleanup_unresolved")
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                use._failure = use._failure or "needed_cache_client_cleanup_unresolved"
                blockers.append("runtime_dependency_client_cleanup_unresolved")
    if not blockers:
        use.check()
    url_path = job / RUNTIME_DEPENDENCY_URL_FILENAME
    url_status = (_write_sensitive_file(url_path, url, label="provider_runtime_dependency_url")
                  if url and not blockers else _file_status(url_path, label="provider_runtime_dependency_url"))
    result = dict(schema_version=RUNTIME_DEPENDENCY_CACHE_SCHEMA_VERSION,
        generated_at=generated_at or utc_now_iso(), status="completed" if url and not blockers else "blocked",
        dependency_path=str(dependency_path), dependency_sha256="sha256:"+digest, artifact_kind=artifact_kind,
        dependency_size_bytes=row["size_bytes"], content_addressed_key_sha256=hashlib.sha256(key.encode()).hexdigest(),
        cache_hit=cache_hit, upload_performed=uploaded, remote_identity_verified=remote, signed_url_file=url_status,
        object_store={name: value[1] for name, value in credentials.items()} | {"expiration_seconds": int(expiration_seconds)},
        cache_object_retained_for_reuse=remote, blockers=sorted(set(blockers)), raw_secret_values_recorded=False)
    write_json(job / "wam_provider_runtime_dependency_cache.json", result)
    return result
