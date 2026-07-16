"""Cloud (GCS) Arena package delivery producer.

Unlike :mod:`arena_package_delivery_local` (local-filesystem only,
``storage_upload_performed=False``), this command performs a REAL upload of the
built ``delivery_bundle`` to a GCS bucket under the
``marketplace-artifacts/{entitlement_id}/`` prefix that the WebApp signed-URL
route reads, and records the ``gs://`` object URIs + per-object checksums into a
delivery-command manifest that the WebApp entitlement/inventory ingestion
consumes to mint short-lived signed URLs.

This closes the gap where the pipeline never uploaded packages to cloud, so the
WebApp signed-URL handoff had no ``gs://`` source. It is fail-closed and gated,
and does NOT upgrade proof/deployment claims — it only makes the artifact
delivery *producer* real. A live end-to-end (real bucket + credentials + buyer
signed-URL download) remains a deploy/ops verification step; the storage client
is injectable so the producer is deterministically unit-tested.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .arena_result_ingest import CLAIM_BOUNDARY, DELIVERY_COMMAND_MANIFEST_SCHEMA_VERSION
from .common import utc_now_iso, write_json

OUTPUT_FILENAME = "delivery_upload.command.json"
GATE_ENV = "BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD"
BUCKET_ENV = "BLUEPRINT_PACKAGE_DELIVERY_BUCKET"
PREFIX_ENV = "BLUEPRINT_PACKAGE_DELIVERY_PREFIX"
DEFAULT_PREFIX = "marketplace-artifacts"


class _BlobLike(Protocol):
    def upload_from_filename(self, filename: str) -> Any: ...


class _BucketLike(Protocol):
    def blob(self, key: str) -> _BlobLike: ...


class _StorageClientLike(Protocol):
    def bucket(self, name: str) -> _BucketLike: ...


def _string(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "on"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_storage_client_factory() -> _StorageClientLike:
    from google.cloud import storage as gcs_storage  # type: ignore[import-untyped]

    return gcs_storage.Client()


def build_gcs_delivery_command_manifest(
    *,
    output_dir: str | Path = ".",
    entitlement_id: str | None = None,
    destination_bucket: str | None = None,
    destination_prefix: str | None = None,
    storage_client: Optional[_StorageClientLike] = None,
    storage_client_factory: Optional[Callable[[], _StorageClientLike]] = None,
) -> Dict[str, Any]:
    """Upload the built ``delivery_bundle`` to GCS and write the delivery manifest.

    Fail-closed: returns a ``blocked`` manifest (no upload performed) when the
    gate env, bucket, entitlement id, or bundle are missing. The storage client
    is injectable for deterministic testing; in production the default factory
    lazily constructs ``google.cloud.storage.Client()``.
    """

    resolved_output = Path(output_dir).resolve()
    generated_at = utc_now_iso()

    entitlement = _string(entitlement_id)
    bucket_name = _string(destination_bucket or os.getenv(BUCKET_ENV))
    prefix = _string(destination_prefix or os.getenv(PREFIX_ENV)) or DEFAULT_PREFIX
    prefix = prefix.strip("/")

    blockers: List[str] = []
    if not _truthy(os.getenv(GATE_ENV)):
        blockers.append(f"missing_env_{GATE_ENV}")
    if not bucket_name:
        blockers.append(f"missing_env_{BUCKET_ENV}")
    if not entitlement:
        blockers.append("missing_entitlement_id")

    bundle_dir = resolved_output / "delivery_bundle"
    if not bundle_dir.is_dir():
        blockers.append("missing_delivery_bundle")

    files: List[Path] = []
    if bundle_dir.is_dir():
        files = sorted(p for p in bundle_dir.rglob("*") if p.is_file())
        if not files:
            blockers.append("empty_delivery_bundle")

    uploaded_objects: List[Dict[str, Any]] = []
    storage_upload_performed = False
    upload_error: Optional[str] = None

    if not blockers:
        base_key = f"{prefix}/{entitlement}"
        try:
            client = storage_client or (storage_client_factory or _default_storage_client_factory)()
            bucket = client.bucket(bucket_name)
            for path in files:
                rel = path.relative_to(bundle_dir).as_posix()
                key = f"{base_key}/{rel}"
                bucket.blob(key).upload_from_filename(str(path))
                uploaded_objects.append(
                    {
                        "relative_path": rel,
                        "object_key": key,
                        "gs_uri": f"gs://{bucket_name}/{key}",
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256_file(path),
                    }
                )
            storage_upload_performed = True
        except Exception as exc:  # noqa: BLE001 - surfaced as a blocker, fail-closed
            upload_error = f"{type(exc).__name__}: {exc}"
            blockers.append("storage_upload_failed")
            uploaded_objects = []

    delivery_base_uri = (
        f"gs://{bucket_name}/{prefix}/{entitlement}"
        if bucket_name and entitlement
        else None
    )

    manifest: Dict[str, Any] = {
        "schema_version": DELIVERY_COMMAND_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "cloud_delivery_uploaded_review_required" if storage_upload_performed else "blocked",
        "provider": "gcs",
        "destination_bucket": bucket_name or None,
        "destination_prefix": prefix,
        "entitlement_id": entitlement or None,
        "delivery_base_uri": delivery_base_uri if storage_upload_performed else None,
        "blockers": blockers,
        "signed_urls": [],
        "objects": uploaded_objects,
        "object_count": len(uploaded_objects),
        "storage_upload_performed": storage_upload_performed,
        "signed_access_performed": False,
        "entitlement_verified": False,
        "upload_error": upload_error,
        # Contract the WebApp entitlement/inventory ingestion consumes to attach a
        # gs:// source to marketplaceEntitlements/publishedMarketplaceInventory
        # and (only after its own entitlement + consent checks) mint signed URLs.
        "webapp_ingestion": {
            "entitlement_id": entitlement or None,
            "delivery_base_uri": delivery_base_uri if storage_upload_performed else None,
            "object_keys": [obj["object_key"] for obj in uploaded_objects],
            "signed_urls_minted_by": "webapp_marketplace_entitlements_route",
            "requires_webapp_entitlement_and_consent_check": True,
        },
        "public_claim_upgrade_allowed": False,
        "proof_effect": "none",
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "cloud_upload_performed": storage_upload_performed,
            "signed_delivery_access_proven": False,
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


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload an Arena delivery bundle to GCS.")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--entitlement-id", default=None)
    parser.add_argument("--destination-bucket", default=None)
    parser.add_argument("--destination-prefix", default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    manifest = build_gcs_delivery_command_manifest(
        output_dir=args.output_dir,
        entitlement_id=args.entitlement_id,
        destination_bucket=args.destination_bucket,
        destination_prefix=args.destination_prefix,
    )
    return 0 if manifest.get("storage_upload_performed") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
