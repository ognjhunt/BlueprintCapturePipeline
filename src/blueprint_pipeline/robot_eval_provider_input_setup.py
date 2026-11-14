"""Publish live-profile manifests to exclusive content-addressed objects.

Uploads one local file to object storage, reads the complete object back, and
seals a secret-free receipt. It does not call any GPU provider.
"""

from __future__ import annotations

from blueprint_pipeline.s3_compatible_transport import s3_compatible_client

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence
from urllib.parse import urlparse

from .adp009d_live_readiness import LEGACY_LIVE_PROFILE_BUILDER
from .common import ensure_dir, parse_gs_uri, read_json_any, utc_now_iso
from .decision_evidence_contracts import canonical_digest


R2_ENDPOINT_ENV_VAR_ALTERNATIVES = (
    "BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL",
    "R2_ENDPOINT_URL",
    "AWS_ENDPOINT_URL",
)
S3_SECRET_ENV_VARS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
S3_REGION_ENV_VAR_ALTERNATIVES = ("AWS_REGION", "AWS_DEFAULT_REGION")

# Production launch services expose the object-store credentials through the
# WAM file contract.  The publication helper historically understood only the
# AWS/R2 spellings, so invoking it from the canonical service environment fell
# through to boto3's ambient credential lookup.  Keep the existing AWS names
# first and treat these as file-path aliases; secret values remain in memory
# only long enough to construct the storage client and are never copied into
# the process environment or a receipt.
FILE_ENV_VAR_ALIASES: Mapping[str, tuple[str, ...]] = {
    "AWS_ACCESS_KEY_ID": ("BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE",),
    "AWS_SECRET_ACCESS_KEY": (
        "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE",
    ),
    "BLUEPRINT_OBJECT_STORAGE_ENDPOINT_URL": (
        "BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE",
    ),
    "AWS_REGION": ("BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE",),
}

# Every website-reachable live-profile builder accepts an operator-supplied
# manifest URI. Generated manifests for these builders share this one
# content-addressed, full-byte-readback publication contract.
LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS = {
    LEGACY_LIVE_PROFILE_BUILDER: "exact_commit_raw_github",
    "build_arena_native_control_live_profile.py": "content_addressed_full_readback",
    "build_artifixer3d_live_profile.py": "content_addressed_full_readback",
    "build_content_agents_live_profile.py": "content_addressed_full_readback",
    "build_gaussian_excision_live_profile.py": "content_addressed_full_readback",
    "build_joint_agent_live_profile.py": "content_addressed_full_readback",
    "build_native_task_arena_live_profile.py": "content_addressed_full_readback",
    "build_new_site_diagnostic_canary_live_profile.py": "content_addressed_full_readback",
    "build_paired_target_native_import_live_profile.py": "content_addressed_full_readback",
    "build_retained_scene_render_live_profile.py": "content_addressed_full_readback",
    "build_sam31_source_tracks_live_profile.py": "content_addressed_full_readback",
    "build_semantic_teacher_image_edit_live_profile.py": "content_addressed_full_readback",
    "build_simready_isaac_live_profile.py": "content_addressed_full_readback",
    "build_task_evaluation_scene_configuration_live_profile.py": "content_addressed_full_readback",
}


def _file_env_name(env_name: str) -> str:
    return f"{env_name}_FILE"


def _file_env_names(env_name: str) -> tuple[str, ...]:
    return (_file_env_name(env_name), *FILE_ENV_VAR_ALIASES.get(env_name, ()))


def _read_named_file_env_value(file_env_name: str) -> str:
    path_text = os.getenv(file_env_name)
    if not path_text:
        return ""
    path = Path(path_text).expanduser()
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _read_file_env_value(env_name: str) -> str:
    for file_env_name in _file_env_names(env_name):
        value = _read_named_file_env_value(file_env_name)
        if value:
            return value
    return ""


def _env_or_file_value(env_name: str) -> str:
    return os.getenv(env_name) or _read_file_env_value(env_name)



def _first_env_or_file_value(env_names: Sequence[str]) -> str:
    for env_name in env_names:
        value = _env_or_file_value(env_name)
        if value:
            return value
    return ""


def _present_file_env_vars(env_names: Sequence[str]) -> list[str]:
    return [
        file_env_name
        for env_name in env_names
        for file_env_name in _file_env_names(env_name)
        if _read_named_file_env_value(file_env_name)
    ]


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""



def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _upload_file_to_gs(
    source: Path, destination_uri: str, *, exclusive: bool = False
) -> Dict[str, Any]:
    parsed = parse_gs_uri(destination_uri)
    try:
        from google.cloud import storage as gcs_storage  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("google-cloud-storage is required for gs:// uploads") from exc
    client = gcs_storage.Client()
    kwargs = {"if_generation_match": 0} if exclusive else {}
    client.bucket(parsed.bucket).blob(parsed.key).upload_from_filename(
        str(source), **kwargs
    )
    return {
        "status": "uploaded",
        "source": str(source),
        "destination_uri": destination_uri,
        "storage_scheme": "gs",
    }


def _upload_file_to_s3_compatible(
    source: Path, destination_uri: str, *, exclusive: bool = False
) -> Dict[str, Any]:
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("boto3 is required for s3:// or r2:// uploads") from exc
    parsed = urlparse(destination_uri)
    endpoint_url = _first_env_or_file_value(R2_ENDPOINT_ENV_VAR_ALTERNATIVES) or None
    kwargs: Dict[str, Any] = {}
    # The canonical production object store is S3-compatible DigitalOcean
    # Spaces and intentionally uses ordinary ``s3://`` references.  Honor the
    # operator-owned endpoint for both accepted S3-compatible spellings; only
    # doing so for ``r2://`` silently sent ``s3://`` publications to AWS.
    if parsed.scheme in {"s3", "r2"} and endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
    region = _first_env_or_file_value(S3_REGION_ENV_VAR_ALTERNATIVES)
    if region:
        kwargs["region_name"] = region
    access_key = _env_or_file_value("AWS_ACCESS_KEY_ID")
    secret_key = _env_or_file_value("AWS_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
    client = s3_compatible_client(boto3, **kwargs)
    if exclusive:
        # A content-addressed publication must never silently replace an
        # object. S3-compatible stores expose this conditional PutObject
        # contract in the same role as GCS generation-match-zero.
        client.put_object(
            Bucket=parsed.netloc,
            Key=parsed.path.lstrip("/"),
            Body=source.read_bytes(),
            IfNoneMatch="*",
        )
    else:
        client.upload_file(str(source), parsed.netloc, parsed.path.lstrip("/"))
    return {
        "status": "uploaded",
        "source": str(source),
        "destination_uri": destination_uri,
        "storage_scheme": parsed.scheme,
        "endpoint_configured": bool(endpoint_url),
        "file_based_secret_env_vars_present": _present_file_env_vars(S3_SECRET_ENV_VARS),
        "file_based_endpoint_env_vars_present": _present_file_env_vars(
            R2_ENDPOINT_ENV_VAR_ALTERNATIVES
        ),
        "secret_values_recorded": False,
    }


def _validate_uploaded_object(source: Path, destination_uri: str) -> Dict[str, Any]:
    scheme = urlparse(destination_uri).scheme
    expected_size_bytes = source.stat().st_size if source.is_file() else None
    expected_sha256 = _sha256_file(source) if source.is_file() else None
    try:
        if scheme == "gs":
            parsed = parse_gs_uri(destination_uri)
            from google.cloud import storage as gcs_storage  # type: ignore[import-untyped]

            client = gcs_storage.Client()
            blob = client.bucket(parsed.bucket).blob(parsed.key)
            with tempfile.NamedTemporaryFile() as temporary:
                blob.download_to_filename(temporary.name)
                temporary.flush()
                readback = Path(temporary.name)
                object_size_bytes = readback.stat().st_size
                observed_sha256 = _sha256_file(readback)
        elif scheme in {"s3", "r2"}:
            try:
                import boto3  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError("boto3 is required for s3:// or r2:// validation") from exc
            parsed = urlparse(destination_uri)
            endpoint_url = _first_env_or_file_value(R2_ENDPOINT_ENV_VAR_ALTERNATIVES) or None
            kwargs: Dict[str, Any] = {}
            if parsed.scheme in {"s3", "r2"} and endpoint_url:
                kwargs["endpoint_url"] = endpoint_url
            region = _first_env_or_file_value(S3_REGION_ENV_VAR_ALTERNATIVES)
            if region:
                kwargs["region_name"] = region
            access_key = _env_or_file_value("AWS_ACCESS_KEY_ID")
            secret_key = _env_or_file_value("AWS_SECRET_ACCESS_KEY")
            if access_key and secret_key:
                kwargs["aws_access_key_id"] = access_key
                kwargs["aws_secret_access_key"] = secret_key
            response = s3_compatible_client(boto3, **kwargs).get_object(
                Bucket=parsed.netloc,
                Key=parsed.path.lstrip("/"),
            )
            body = response["Body"]
            digest = hashlib.sha256()
            object_size_bytes = 0
            for chunk in iter(lambda: body.read(1024 * 1024), b""):
                digest.update(chunk)
                object_size_bytes += len(chunk)
            # Upload receipts use bare SHA-256 internally.  The typed
            # task-evaluation publication receipt adds the ``sha256:`` prefix.
            observed_sha256 = digest.hexdigest()
        elif scheme in {"", "file"}:
            destination = Path(urlparse(destination_uri).path if scheme else destination_uri).resolve()
            if not destination.is_file():
                return {
                    "status": "blocked",
                    "destination_uri": destination_uri,
                    "storage_scheme": scheme or "file",
                    "blockers": ["uploaded_object_missing_after_copy"],
                    "secret_values_recorded": False,
                }
            object_size_bytes = destination.stat().st_size
            observed_sha256 = _sha256_file(destination)
        else:
            return {
                "status": "blocked",
                "destination_uri": destination_uri,
                "storage_scheme": scheme or "local",
                "blockers": [f"unsupported_validation_uri_scheme:{scheme}"],
                "secret_values_recorded": False,
            }
    except Exception as exc:
        return {
            "status": "blocked",
            "destination_uri": destination_uri,
            "storage_scheme": scheme or "file",
            "blockers": [f"uploaded_object_validation_failed:{type(exc).__name__}"],
            "error": str(exc),
            "secret_values_recorded": False,
        }
    size_matches = (
        expected_size_bytes is None or int(object_size_bytes) == int(expected_size_bytes)
    )
    digest_matches = expected_sha256 is None or observed_sha256 == expected_sha256
    return {
        "status": "validated" if size_matches and digest_matches else "blocked",
        "destination_uri": destination_uri,
        "storage_scheme": scheme or "file",
        "object_size_bytes": object_size_bytes,
        "expected_size_bytes": expected_size_bytes,
        "object_sha256": observed_sha256,
        "expected_sha256": expected_sha256,
        "size_matches_source": size_matches,
        "digest_matches_source": digest_matches,
        "provider_fetchable_object_probe": size_matches and digest_matches,
        "full_byte_readback_performed": True,
        "multipart_etag_used_as_digest": False,
        "blockers": (
            []
            if size_matches and digest_matches
            else [
                *([] if size_matches else ["uploaded_object_size_mismatch"]),
                *([] if digest_matches else ["uploaded_object_digest_mismatch"]),
            ]
        ),
        "secret_values_recorded": False,
    }


def _classify_upload_error(*, scheme: str, error: BaseException) -> str:
    text = str(error).lower()
    if scheme == "gs" and "billing account" in text and (
        "disabled" in text or "absent" in text or "accountdisabled" in text
    ):
        return "upload_failed:gs_billing_account_disabled"
    if "accessdenied" in text or "access denied" in text:
        return f"upload_failed:{scheme}_access_denied"
    if "forbidden" in text or "403" in text:
        return f"upload_failed:{scheme}_forbidden"
    return f"upload_failed:{type(error).__name__}"


def _seal_upload_receipt(result: Dict[str, Any], *, source: Path) -> Dict[str, Any]:
    """Attach immutable source/readback facts to every successful upload."""

    validation = result.get("post_upload_validation")
    if result.get("status") != "uploaded" or not isinstance(validation, Mapping):
        return result
    sealed = {
        "schema_version": "provider_input_immutable_publication.v1",
        **result,
        "source_size_bytes": source.stat().st_size,
        "source_sha256": _sha256_file(source),
        "remote_size_bytes": validation.get("object_size_bytes"),
        "remote_sha256": validation.get("object_sha256"),
        "full_byte_readback_verified": (
            validation.get("status") == "validated"
            and validation.get("full_byte_readback_performed") is True
            and validation.get("digest_matches_source") is True
        ),
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    sealed["receipt_digest"] = canonical_digest(
        sealed, digest_field="receipt_digest"
    )
    return sealed


def upload_file(
    source: str | Path, destination_uri: str, *, exclusive: bool = False
) -> Dict[str, Any]:
    path = Path(source).resolve()
    scheme = urlparse(destination_uri).scheme
    if exclusive and scheme not in {"gs", "s3", "r2"}:
        return {
            "status": "blocked",
            "source": str(path),
            "destination_uri": destination_uri,
            "storage_scheme": scheme or "file",
            "blockers": [f"exclusive_create_unsupported_for_scheme:{scheme or 'file'}"],
            "raw_secret_values_recorded": False,
        }
    try:
        if scheme == "gs":
            result = _upload_file_to_gs(path, destination_uri, exclusive=exclusive)
            validation = _validate_uploaded_object(path, destination_uri)
            if validation.get("status") != "validated":
                return {
                    **result,
                    "status": "blocked",
                    "blockers": validation.get("blockers") or ["uploaded_object_validation_failed"],
                    "post_upload_validation": validation,
                }
            return _seal_upload_receipt(
                {**result, "post_upload_validation": validation}, source=path
            )
        if scheme in {"s3", "r2"}:
            result = _upload_file_to_s3_compatible(
                path, destination_uri, exclusive=exclusive
            )
            validation = _validate_uploaded_object(path, destination_uri)
            if validation.get("status") != "validated":
                return {
                    **result,
                    "status": "blocked",
                    "blockers": validation.get("blockers") or ["uploaded_object_validation_failed"],
                    "post_upload_validation": validation,
                }
            return _seal_upload_receipt(
                {**result, "post_upload_validation": validation}, source=path
            )
        if scheme in {"", "file"}:
            destination = Path(urlparse(destination_uri).path if scheme else destination_uri).resolve()
            ensure_dir(destination.parent)
            destination.write_bytes(path.read_bytes())
            validation = _validate_uploaded_object(path, destination_uri)
            if validation.get("status") != "validated":
                return {
                    "status": "blocked",
                    "source": str(path),
                    "destination_uri": str(destination),
                    "storage_scheme": "file",
                    "blockers": validation.get("blockers") or ["uploaded_object_validation_failed"],
                    "post_upload_validation": validation,
                }
            return _seal_upload_receipt({
                "status": "uploaded",
                "source": str(path),
                "destination_uri": str(destination),
                "storage_scheme": "file",
                "post_upload_validation": validation,
            }, source=path)
    except Exception as exc:
        blocker = _classify_upload_error(scheme=scheme or "file", error=exc)
        return {
            "status": "blocked",
            "source": str(path),
            "destination_uri": destination_uri,
            "storage_scheme": scheme or "file",
            "blockers": [blocker],
            "error": str(exc),
        }
    return {
        "status": "blocked",
        "source": str(path),
        "destination_uri": destination_uri,
        "blockers": [f"unsupported_upload_uri_scheme:{scheme}"],
    }


def publish_content_addressed_manifest(
    *,
    source: str | Path,
    destination_prefix: str,
    receipt_path: str | Path,
    profile_builder: str,
) -> Dict[str, Any]:
    """Publish one JSON manifest at an exclusive SHA-addressed object key.

    This is intentionally a storage mutation, not a compute/provider mutation.
    The returned receipt is useful only after the object was downloaded again
    and its complete bytes matched the local source.  An existing object is a
    conflict even when its name is content-addressed: silently accepting it
    would lose evidence of which invocation published the profile input.
    """

    path = Path(source).expanduser().resolve()
    output = Path(receipt_path).expanduser().resolve()
    if profile_builder not in LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS:
        raise ValueError("immutable_manifest_profile_builder_unregistered")
    if (
        LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS[profile_builder]
        != "content_addressed_full_readback"
    ):
        raise ValueError("immutable_manifest_profile_builder_not_generated")
    if path.is_symlink() or not path.is_file():
        raise ValueError("immutable_manifest_source_missing")
    if output.exists() or output.is_symlink():
        raise ValueError("immutable_manifest_publication_receipt_exists")
    try:
        manifest = read_json_any(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("immutable_manifest_source_invalid") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("immutable_manifest_source_not_object")
    prefix = str(destination_prefix or "").strip().rstrip("/")
    parsed = urlparse(prefix)
    if parsed.scheme not in {"gs", "s3", "r2"} or not parsed.netloc:
        raise ValueError("immutable_manifest_destination_prefix_invalid")
    source_sha256 = "sha256:" + _sha256_file(path)
    identity = source_sha256.removeprefix("sha256:")
    destination_uri = f"{prefix}/sha256/{identity[:2]}/{identity}.json"
    publication = upload_file(path, destination_uri, exclusive=True)
    if (
        publication.get("status") != "uploaded"
        or publication.get("full_byte_readback_verified") is not True
        or publication.get("source_sha256") != identity
        or publication.get("remote_sha256") != identity
        or publication.get("source_size_bytes") != path.stat().st_size
        or publication.get("remote_size_bytes") != path.stat().st_size
    ):
        blockers = publication.get("blockers") or [
            "immutable_manifest_provider_readback_invalid"
        ]
        raise ValueError(
            "immutable_manifest_publication_blocked:" + ",".join(map(str, blockers))
        )
    receipt: Dict[str, Any] = {
        "schema_version": "task_evaluation_immutable_manifest_publication.v1",
        "status": "published",
        "generated_at": utc_now_iso(),
        "source": {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": source_sha256,
        },
        "profile_builder": profile_builder,
        "publication_seam": LIVE_PROFILE_MANIFEST_PUBLICATION_SEAMS[profile_builder],
        "published_uri": destination_uri,
        "storage_scheme": parsed.scheme,
        "content_addressed_key": True,
        "exclusive_create": True,
        "provider_full_byte_readback_verified": True,
        "remote_size_bytes": publication["remote_size_bytes"],
        "remote_sha256": source_sha256,
        "upload_receipt_digest": publication["receipt_digest"],
        "provider_compute_mutation_performed": False,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": [],
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    ensure_dir(output.parent)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return receipt
