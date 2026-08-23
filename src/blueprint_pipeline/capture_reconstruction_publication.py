"""Immutable, full-byte-readback publication for capture reconstruction artifacts."""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "capture_reconstruction_publication.v1"
_PUBLISHED_KINDS = {
    "standard_3dgs_ply",
    "compressed_3dgs_spz_v4",
    "postshot_project",
    "postshot_coordinate_binding",
    "colmap_camera_intrinsics",
    "colmap_camera_extrinsics_and_order",
    "colmap_seed_points",
    "training_log",
}


class CaptureReconstructionPublicationError(ValueError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _safe_member(name: Any) -> PurePosixPath:
    value = str(name or "")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or not value.startswith("results/")
    ):
        raise CaptureReconstructionPublicationError(
            "capture_reconstruction_publication_member_path_invalid"
        )
    return path


def publish_postshot_output_bundle(
    *,
    output_bundle: str | Path,
    capture_id: str,
    capture_digest: str,
    bucket_name: str,
    publication_root: str | Path,
    storage_client: Any = None,
) -> dict[str, Any]:
    """Validate, upload with create-only semantics, then read every byte back."""

    bundle = Path(output_bundle).expanduser().resolve()
    root = Path(publication_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not bundle.is_file() or not capture_id or not bucket_name:
        raise CaptureReconstructionPublicationError(
            "capture_reconstruction_publication_input_invalid"
        )
    try:
        with zipfile.ZipFile(bundle, "r") as archive:
            manifest = json.loads(
                archive.read("canonical_3dgs_vast_output_manifest.json")
            )
            worker = json.loads(archive.read("results/worker_receipt.json"))
            if not isinstance(manifest, Mapping) or not isinstance(worker, Mapping):
                raise ValueError("control")
            artifacts = {
                str(row.get("kind")): dict(row)
                for row in worker.get("artifacts") or []
                if isinstance(row, Mapping) and row.get("kind") in _PUBLISHED_KINDS
            }
            if not _PUBLISHED_KINDS.issubset(artifacts):
                raise ValueError("artifacts")
            extraction = Path(
                tempfile.mkdtemp(
                    prefix="capture-reconstruction-publish-",
                    dir=root,
                )
            )
            extracted: list[tuple[str, Path, dict[str, Any]]] = []
            try:
                for kind in sorted(_PUBLISHED_KINDS):
                    row = artifacts[kind]
                    relative = _safe_member("results/" + str(row.get("relative_path") or ""))
                    info = archive.getinfo(relative.as_posix())
                    if info.is_dir() or info.file_size <= 0:
                        raise ValueError("member")
                    destination = extraction / Path(*relative.parts[1:])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, destination.open("wb") as target:
                        for chunk in iter(lambda: source.read(1024 * 1024), b""):
                            target.write(chunk)
                    if _sha(destination) != row.get("digest"):
                        raise ValueError("digest")
                    extracted.append((kind, destination, row))

                if storage_client is None:  # pragma: no cover - production seam
                    from google.cloud import storage

                    storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                published: list[dict[str, Any]] = []
                prefix = (
                    f"scenes-derived/captures/{capture_id}/canonical-3dgs/"
                    f"{capture_digest.removeprefix('sha256:')}"
                )
                for kind, path, row in extracted:
                    digest = str(row["digest"])
                    object_name = (
                        f"{prefix}/{kind}/sha256/"
                        f"{digest.removeprefix('sha256:')}/{path.name}"
                    )
                    blob = bucket.blob(object_name)
                    try:
                        blob.upload_from_filename(str(path), if_generation_match=0)
                    except Exception as exc:  # pre-existing is safe only if bytes match
                        if type(exc).__name__ not in {"PreconditionFailed", "Conflict"}:
                            raise
                    with tempfile.NamedTemporaryFile(dir=extraction, delete=False) as tmp:
                        readback = Path(tmp.name)
                    try:
                        blob.download_to_filename(str(readback))
                        if _sha(readback) != digest or readback.stat().st_size != path.stat().st_size:
                            raise CaptureReconstructionPublicationError(
                                "capture_reconstruction_publication_readback_mismatch"
                            )
                    finally:
                        readback.unlink(missing_ok=True)
                    published.append(
                        {
                            "artifact_id": kind,
                            "digest": digest,
                            "bytes": path.stat().st_size,
                            "uri": f"gs://{bucket_name}/{object_name}",
                            "create_only": True,
                            "full_byte_readback_verified": True,
                        }
                    )
            finally:
                import shutil

                shutil.rmtree(extraction, ignore_errors=True)
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, CaptureReconstructionPublicationError):
            raise
        raise CaptureReconstructionPublicationError(
            "capture_reconstruction_publication_bundle_invalid"
        ) from exc

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "published",
        "capture_id": capture_id,
        "capture_digest": capture_digest,
        "output_bundle_digest": _sha(bundle),
        "artifacts": published,
        "artifact_count": len(published),
        "all_create_only": True,
        "all_full_byte_readback_verified": True,
        "metric_alignment_qualified": False,
        "physical_truth_inferred": False,
        "published_at": utc_now_iso(),
    }
    receipt["publication_digest"] = canonical_digest(
        receipt, digest_field="publication_digest"
    )
    write_json(root / "capture_reconstruction_publication.json", receipt)
    return receipt


__all__ = [
    "CaptureReconstructionPublicationError",
    "SCHEMA_VERSION",
    "publish_postshot_output_bundle",
]
