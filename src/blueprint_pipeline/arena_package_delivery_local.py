"""Arena package delivery command hook.

This command is a gated delivery hook. It can copy the Arena
``delivery_bundle`` into a configured local delivery root and, when a GCS prefix
is configured, upload the same bundle to cloud storage so buyer delivery
surfaces have a concrete ``gs://`` package source.

It does not verify customer entitlement or upgrade proof claims. Signed URLs are
optional because the WebApp can create buyer-scoped signed URLs from the
persisted ``gs://`` package URI.
"""

from __future__ import annotations

import argparse
import hashlib
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .arena_result_ingest import CLAIM_BOUNDARY, DELIVERY_COMMAND_MANIFEST_SCHEMA_VERSION
from .common import ensure_dir, utc_now_iso, write_json


OUTPUT_FILENAME = "delivery_upload.command.json"
GATE_ENV = "BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD"
DELIVERY_ROOT_ENV = "BLUEPRINT_LOCAL_DELIVERY_ROOT"
GCS_PREFIX_ENV = "BLUEPRINT_PACKAGE_DELIVERY_GCS_PREFIX"
ENTITLEMENT_ID_ENV = "BLUEPRINT_PACKAGE_DELIVERY_ENTITLEMENT_ID"
MARKETPLACE_ITEM_ID_ENV = "BLUEPRINT_PACKAGE_DELIVERY_MARKETPLACE_ITEM_ID"
BUYER_USER_ID_ENV = "BLUEPRINT_PACKAGE_DELIVERY_BUYER_USER_ID"
SIGNED_URLS_ENV = "BLUEPRINT_PACKAGE_DELIVERY_SIGNED_URLS"
SIGNED_URL_TTL_SECONDS_ENV = "BLUEPRINT_PACKAGE_DELIVERY_SIGNED_URL_TTL_SECONDS"
DEFAULT_SIGNED_URL_TTL_SECONDS = 15 * 60


def _string(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "on"}


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(_string(value))
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _relative_or_absolute(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _parse_gs_prefix(value: str) -> Tuple[str, str]:
    text = _string(value)
    if not text.startswith("gs://"):
        raise ValueError("gcs_prefix_must_start_with_gs://")
    without_scheme = text[5:]
    bucket, separator, prefix = without_scheme.partition("/")
    if not bucket or not separator or not prefix.strip("/"):
        raise ValueError("gcs_prefix_must_include_bucket_and_object_prefix")
    return bucket, prefix.strip("/")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_type(path: Path) -> str | None:
    return mimetypes.guess_type(path.name)[0]


def _load_storage_client() -> Any:
    try:
        from google.cloud import storage  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by env-dependent installs.
        raise RuntimeError("google-cloud-storage is required for GCS package delivery") from exc
    return storage.Client()


def _generate_signed_url(blob: Any, *, ttl_seconds: int) -> str:
    return str(blob.generate_signed_url(expiration=ttl_seconds, method="GET"))


def _empty_entitlement_patch(*, generated_at: str, reason: str) -> Dict[str, Any]:
    return {
        "schema_version": "marketplace_entitlement_artifact_patch.v1",
        "generated_at": generated_at,
        "status": "review_required",
        "reason": reason,
        "target_collection": "marketplaceEntitlements",
        "entitlement_id": None,
        "marketplace_item_id": None,
        "buyer_user_id": None,
        "fields": {},
    }


def _build_entitlement_patch(
    *,
    generated_at: str,
    artifact_uri: str | None,
    delivery_manifest_uri: str | None,
    uploaded_objects: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not artifact_uri:
        return _empty_entitlement_patch(
            generated_at=generated_at,
            reason="missing_cloud_artifact_uri",
        )

    entitlement_id = _string(os.getenv(ENTITLEMENT_ID_ENV)) or None
    marketplace_item_id = _string(os.getenv(MARKETPLACE_ITEM_ID_ENV)) or None
    buyer_user_id = _string(os.getenv(BUYER_USER_ID_ENV)) or None
    fields = {
        "artifact_uri": artifact_uri,
        "post_training_data_package_uri": artifact_uri,
        "delivery_manifest_uri": delivery_manifest_uri,
        "artifact_object_count": len(uploaded_objects),
        "artifact_delivery_provider": "gcs",
        "artifact_delivery_updated_at": generated_at,
    }
    status = "ready_for_webapp_patch" if entitlement_id else "review_required"
    return {
        "schema_version": "marketplace_entitlement_artifact_patch.v1",
        "generated_at": generated_at,
        "status": status,
        "reason": None if entitlement_id else f"missing_env_{ENTITLEMENT_ID_ENV}",
        "target_collection": "marketplaceEntitlements",
        "entitlement_id": entitlement_id,
        "marketplace_item_id": marketplace_item_id,
        "buyer_user_id": buyer_user_id,
        "fields": fields,
    }


def _select_primary_artifact_uri(
    uploaded_objects: Sequence[Mapping[str, Any]],
) -> str | None:
    preferred_suffixes = (
        "archives/post_training_data_package.tar.gz",
        "post_training_data_package.tar.gz",
        "package_index.json",
        "post_training_data_package_export_manifest.json",
    )
    for suffix in preferred_suffixes:
        for item in uploaded_objects:
            uri = _string(item.get("gcs_uri"))
            if uri.endswith(suffix):
                return uri
    return _string(uploaded_objects[0].get("gcs_uri")) if uploaded_objects else None


def _upload_delivery_bundle_to_gcs(
    *,
    bundle_dir: Path,
    output_dir: Path,
    gcs_prefix: str,
    generated_at: str,
    storage_client: Any | None = None,
    sign_urls: bool = False,
    signed_url_ttl_seconds: int = DEFAULT_SIGNED_URL_TTL_SECONDS,
) -> Dict[str, Any]:
    bucket_name, prefix = _parse_gs_prefix(gcs_prefix)
    target_prefix = f"{prefix.rstrip('/')}/{output_dir.name}"
    client = storage_client or _load_storage_client()
    bucket = client.bucket(bucket_name)
    uploaded_objects: List[Dict[str, Any]] = []
    signed_urls: List[str] = []
    signed_url_errors: List[Dict[str, str]] = []
    for source in sorted(path for path in bundle_dir.rglob("*") if path.is_file()):
        relative = source.relative_to(bundle_dir)
        object_name = f"{target_prefix}/{relative.as_posix()}"
        blob = bucket.blob(object_name)
        blob.upload_from_filename(str(source), content_type=_content_type(source))
        gcs_uri = f"gs://{bucket_name}/{object_name}"
        uploaded = {
            "relative_path": str(relative),
            "source_path": _relative_or_absolute(output_dir, source),
            "gcs_uri": gcs_uri,
            "size_bytes": source.stat().st_size,
            "sha256": _sha256(source),
        }
        if sign_urls:
            try:
                url = _generate_signed_url(blob, ttl_seconds=signed_url_ttl_seconds)
            except Exception as exc:  # pragma: no cover - provider-specific auth failure.
                signed_url_errors.append(
                    {"gcs_uri": gcs_uri, "error": f"{type(exc).__name__}: {exc}"}
                )
            else:
                signed_urls.append(url)
                uploaded["signed_url_generated"] = True
        uploaded_objects.append(uploaded)

    artifact_uri = _select_primary_artifact_uri(uploaded_objects)
    delivery_manifest_uri = next(
        (
            _string(item.get("gcs_uri"))
            for item in uploaded_objects
            if _string(item.get("relative_path")) == "delivery_manifest.json"
        ),
        None,
    )
    return {
        "schema_version": "arena_gcs_delivery_upload.v1",
        "generated_at": generated_at,
        "bucket": bucket_name,
        "prefix": target_prefix,
        "artifact_uri": artifact_uri,
        "delivery_manifest_uri": delivery_manifest_uri,
        "uploaded_objects": uploaded_objects,
        "signed_urls": signed_urls,
        "signed_url_errors": signed_url_errors,
    }


def build_local_delivery_command_manifest(
    *,
    output_dir: str | Path = ".",
    delivery_root: str | Path | None = None,
    capture_root: str | Path | None = None,
    gcs_prefix: str | None = None,
    storage_client: Any | None = None,
) -> Dict[str, Any]:
    resolved_output = Path(output_dir).resolve()
    generated_at = utc_now_iso()
    root_text = _string(delivery_root or os.getenv(DELIVERY_ROOT_ENV))
    gcs_prefix_text = _string(gcs_prefix or os.getenv(GCS_PREFIX_ENV))
    blockers: List[str] = []
    if not _truthy(os.getenv(GATE_ENV)):
        blockers.append(f"missing_env_{GATE_ENV}")
    if not root_text and not gcs_prefix_text:
        blockers.append(f"missing_env_{DELIVERY_ROOT_ENV}")
    bundle_dir = resolved_output / "delivery_bundle"
    if not bundle_dir.is_dir():
        blockers.append("missing_delivery_bundle")

    # Rights are authoritative continuously: an open consent takedown on the
    # source capture must stop this buyer-bundle copy. When a capture root is
    # provided, re-read consent live and block on revocation; when it is not,
    # record that the gate was not evaluated (visible, not a silent bypass).
    consent_gate: Dict[str, Any] = {
        "evaluated": False,
        "reason": "no_capture_root_provided",
    }
    if capture_root is not None:
        from .consent_takedown import evaluate_delivery_time_takedown_gate

        gate = evaluate_delivery_time_takedown_gate(
            capture_root=Path(capture_root).expanduser(),
            surface="arena_local_delivery",
        )
        consent_gate = {
            "evaluated": True,
            "status": _string(gate.get("status")),
            "serve_allowed": bool(gate.get("serve_allowed")),
            "blockers": [str(item) for item in (gate.get("blockers") or [])],
            "capture_root": str(Path(capture_root).expanduser()),
        }
        if not gate.get("serve_allowed"):
            blockers.append(f"consent_takedown_open:{_string(gate.get('status'))}")

    local_access_paths: List[Dict[str, Any]] = []
    delivery_root_path = Path(root_text).expanduser().resolve() if root_text else None
    target_dir = None
    gcs_upload: Dict[str, Any] | None = None
    uploaded_objects: List[Dict[str, Any]] = []
    signed_urls: List[str] = []
    artifact_uri: str | None = None
    if not blockers and delivery_root_path:
        target_dir = delivery_root_path / resolved_output.name
        ensure_dir(target_dir)
        for source in sorted(path for path in bundle_dir.rglob("*") if path.is_file()):
            relative = source.relative_to(bundle_dir)
            target = target_dir / relative
            ensure_dir(target.parent)
            shutil.copy2(source, target)
            local_access_paths.append(
                {
                    "relative_path": str(relative),
                    "source_path": _relative_or_absolute(resolved_output, source),
                    "delivered_path": str(target),
                    "size_bytes": target.stat().st_size,
                }
            )
        if not local_access_paths:
            blockers.append("delivery_bundle_empty")
    elif not blockers and not gcs_prefix_text:
        blockers.append("missing_delivery_destination")

    if not blockers and gcs_prefix_text:
        try:
            gcs_upload = _upload_delivery_bundle_to_gcs(
                bundle_dir=bundle_dir,
                output_dir=resolved_output,
                gcs_prefix=gcs_prefix_text,
                generated_at=generated_at,
                storage_client=storage_client,
                sign_urls=_truthy(os.getenv(SIGNED_URLS_ENV)),
                signed_url_ttl_seconds=_positive_int(
                    os.getenv(SIGNED_URL_TTL_SECONDS_ENV),
                    default=DEFAULT_SIGNED_URL_TTL_SECONDS,
                ),
            )
        except Exception as exc:
            blockers.append(f"gcs_upload_failed:{type(exc).__name__}")
            gcs_upload = {
                "schema_version": "arena_gcs_delivery_upload.v1",
                "generated_at": generated_at,
                "status": "blocked",
                "blockers": [f"{type(exc).__name__}: {exc}"],
            }
        else:
            uploaded_objects = list(gcs_upload.get("uploaded_objects") or [])
            signed_urls = [str(item) for item in (gcs_upload.get("signed_urls") or [])]
            artifact_uri = _string(gcs_upload.get("artifact_uri")) or None
            if not uploaded_objects:
                blockers.append("gcs_delivery_bundle_empty")
            if not artifact_uri:
                blockers.append("missing_primary_gcs_artifact_uri")

    storage_upload_performed = bool(uploaded_objects and not blockers)
    entitlement_patch = _build_entitlement_patch(
        generated_at=generated_at,
        artifact_uri=artifact_uri,
        delivery_manifest_uri=_string((gcs_upload or {}).get("delivery_manifest_uri")) or None,
        uploaded_objects=uploaded_objects,
    )
    if blockers:
        status = "blocked"
        provider = "gcs" if gcs_prefix_text else "local_filesystem"
    elif storage_upload_performed and signed_urls:
        status = "signed_cloud_delivery_ready"
        provider = "gcs"
    elif storage_upload_performed:
        status = "cloud_delivery_artifact_ready_review_required"
        provider = "gcs"
    else:
        status = "local_delivery_ready_review_required"
        provider = "local_filesystem"

    manifest = {
        "schema_version": DELIVERY_COMMAND_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "provider": provider,
        "delivery_root": str(delivery_root_path) if delivery_root_path else None,
        "target_dir": str(target_dir) if target_dir else None,
        "gcs_prefix": gcs_prefix_text or None,
        "blockers": blockers,
        "consent_gate": consent_gate,
        "artifact_uri": artifact_uri,
        "post_training_data_package_uri": artifact_uri,
        "artifact_uris": {
            "post_training_data_package_uri": artifact_uri,
            "delivery_manifest_uri": _string((gcs_upload or {}).get("delivery_manifest_uri"))
            or None,
        },
        "signed_urls": signed_urls,
        "local_access_paths": local_access_paths,
        "uploaded_objects": uploaded_objects,
        "gcs_upload": gcs_upload,
        "marketplace_entitlement_patch": entitlement_patch,
        "buyer_access_check": {
            "entitlement_verified": False,
            "artifact_uri_ready": bool(artifact_uri),
            "webapp_entitlement_patch_ready": entitlement_patch.get("status")
            == "ready_for_webapp_patch",
            "claim_boundary": "artifact_uri_ready_is_not_buyer_authorization",
        },
        "storage_upload_performed": storage_upload_performed,
        "signed_access_performed": bool(signed_urls),
        "entitlement_verified": False,
        "public_claim_upgrade_allowed": False,
        "proof_effect": "none",
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "local_filesystem_delivery_performed": bool(local_access_paths and not blockers),
            "cloud_storage_delivery_performed": storage_upload_performed,
            "webapp_entitlement_patch_prepared": entitlement_patch.get("status")
            == "ready_for_webapp_patch",
            "signed_delivery_access_proven": bool(signed_urls),
            "delivery_access_is_deployment_approval": False,
            "package_delivery_is_deployment_approval": False,
            "deployment_approval_proven": False,
            "physical_robot_readiness_proven": False,
            "safety_validation_proven": False,
            "public_claim_upgrade_allowed": False,
        },
    }
    write_json(resolved_output / OUTPUT_FILENAME, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy an Arena package delivery bundle to a local delivery root"
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--delivery-root", default=None)
    parser.add_argument(
        "--capture-root",
        default=None,
        help="Source capture root; delivery is blocked if its consent is revoked.",
    )
    parser.add_argument(
        "--gcs-prefix",
        default=None,
        help=f"GCS destination prefix, e.g. gs://bucket/deliveries (or {GCS_PREFIX_ENV}).",
    )
    args = parser.parse_args(argv)
    result = build_local_delivery_command_manifest(
        output_dir=args.output_dir,
        delivery_root=args.delivery_root,
        capture_root=args.capture_root,
        gcs_prefix=args.gcs_prefix,
    )
    print(f"[arena-delivery] manifest={Path(args.output_dir).resolve() / OUTPUT_FILENAME}")
    print(f"[arena-delivery] status={result['status']}")
    if result["blockers"]:
        print(f"[arena-delivery] blockers={len(result['blockers'])}")
    ready_statuses = {
        "local_delivery_ready_review_required",
        "cloud_delivery_artifact_ready_review_required",
        "signed_cloud_delivery_ready",
    }
    return 0 if result["status"] in ready_statuses else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
