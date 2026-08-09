"""Apply suppression volumes at the latest point each renderer allows.

Renderers we implement in Python can skip suppressed rows directly with
:func:`suppression_render_mask`. Renderers we only invoke - the Spark harness,
and anything downstream that consumes a scene file such as the Isaac/NuRec
lane - take a file, so suppression is applied by materializing a payload that
omits those rows.

The payload is never the truth. Retained rows are copied byte-for-byte in
source order, so the payload is a pure function of (canonical scan, volume
set) and is regenerable at any time; the canonical scan is never opened for
writing. Two lifetimes share one materializer: ``transient`` for a single
render invocation, deleted afterwards, and ``cached`` for closed renderers
that need a stable path, written under a content-addressed name so the same
inputs always resolve to the same file.

With no receipts the canonical scan is handed through untouched, which is what
makes turning a task off a genuine restore rather than a regeneration.
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .gaussian_splat_decode import (
    read_standard_3dgs_ply,
    verify_standard_3dgs_ply_subset_exact,
    write_standard_3dgs_ply_subset_exact,
)
from .gaussian_suppression_volume import (
    GaussianSuppressionVolumeError,
    compose_suppression_volumes,
)


SUPPRESSED_PAYLOAD_SCHEMA_VERSION = "gaussian_suppressed_render_package.v1"
PAYLOAD_BASENAME = "suppressed_scene"


class GaussianSuppressionRenderError(ValueError):
    """Stable, sorted suppression-render failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


@dataclass(frozen=True)
class SuppressedPayload:
    """A renderer-ready scene path plus the record of how it was produced."""

    path: Path
    record: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _resolve(
    canonical_ply_path: str | Path, receipts: Sequence[Mapping[str, Any]]
) -> tuple[Path, np.ndarray, dict[str, Any] | None]:
    canonical = Path(canonical_ply_path).expanduser().resolve()
    if not canonical.is_file():
        raise GaussianSuppressionRenderError(["suppression_canonical_scan_missing"])
    if not receipts:
        return canonical, np.zeros(0, dtype=np.int64), None
    try:
        composite = compose_suppression_volumes(
            canonical_ply_path=canonical, receipts=list(receipts)
        )
    except GaussianSuppressionVolumeError as exc:
        raise GaussianSuppressionRenderError(exc.errors) from exc
    indices = np.zeros(0, dtype=np.int64)
    for receipt in receipts:
        from .gaussian_suppression_volume import resolve_suppressed_indices

        resolved, _ = resolve_suppressed_indices(
            canonical_ply_path=canonical, receipt=receipt
        )
        indices = np.union1d(indices, resolved)
    return canonical, indices.astype(np.int64), composite


def suppression_render_mask(
    *,
    canonical_ply_path: str | Path,
    receipts: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Boolean mask of canonical rows a renderer must skip."""

    canonical, indices, _ = _resolve(canonical_ply_path, receipts)
    count = int(read_standard_3dgs_ply(canonical).count)
    mask = np.zeros(count, dtype=bool)
    if indices.size:
        mask[indices] = True
    return mask


def _payload_name(canonical: Path, composite: Mapping[str, Any] | None) -> str:
    scan = _sha256_file(canonical).split(":", 1)[1][:16]
    volume = (
        str(composite.get("suppressed_index_digest") or "").split(":", 1)[-1][:16]
        if composite
        else "none"
    )
    return f"{PAYLOAD_BASENAME}__{scan}__{volume}.ply"


def _materialize(
    *, canonical: Path, indices: np.ndarray, destination: Path
) -> dict[str, Any]:
    total = int(read_standard_3dgs_ply(canonical).count)
    retained = np.setdiff1d(np.arange(total, dtype=np.int64), indices)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_standard_3dgs_ply_subset_exact(canonical, destination, retained)
    proof = verify_standard_3dgs_ply_subset_exact(canonical, destination, retained)
    if not proof.get("retained_rows_byte_exact", False) or not proof.get(
        "retained_order_matches_source", False
    ):
        raise GaussianSuppressionRenderError(
            ["suppression_payload_rows_not_byte_exact"]
        )
    return {
        "path": str(destination),
        "sha256": _sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "canonical_vertex_count": total,
        "suppressed_vertex_count": int(indices.size),
        "retained_vertex_count": int(retained.size),
        "retained_rows_byte_exact": True,
        "retained_order_matches_source": True,
        "retained_indices": retained.tolist() if retained.size <= 4096 else None,
        "is_derived_cache_artifact": True,
    }


@contextlib.contextmanager
def suppressed_render_payload(
    *,
    canonical_ply_path: str | Path,
    receipts: Sequence[Mapping[str, Any]],
    cache_dir: str | Path | None = None,
) -> Iterator[SuppressedPayload]:
    """Yield a renderer-ready path with the volume set applied.

    Transient by default: the payload lives only for the duration of the block.
    With ``cache_dir`` it is written under a content-addressed name and kept,
    which is what closed renderers need. With no receipts the canonical scan is
    yielded directly - nothing is copied and nothing is produced.
    """

    canonical, indices, composite = _resolve(canonical_ply_path, receipts)
    if not receipts:
        yield SuppressedPayload(
            path=canonical,
            record={
                "lifetime": "canonical_passthrough",
                "path": str(canonical),
                "sha256": _sha256_file(canonical),
                "suppressed_vertex_count": 0,
                "is_derived_cache_artifact": False,
            },
        )
        return

    name = _payload_name(canonical, composite)
    if cache_dir is not None:
        cache = Path(cache_dir).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        destination = cache / name
        if destination.is_file():
            record = {
                "lifetime": "cached",
                "cache_hit": True,
                "path": str(destination),
                "sha256": _sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "suppressed_vertex_count": int(indices.size),
                "is_derived_cache_artifact": True,
            }
            yield SuppressedPayload(path=destination, record=record)
            return
        record = _materialize(
            canonical=canonical, indices=indices, destination=destination
        )
        record.update({"lifetime": "cached", "cache_hit": False})
        yield SuppressedPayload(path=destination, record=record)
        return

    scratch = Path(tempfile.mkdtemp(prefix="blueprint-suppressed-"))
    try:
        record = _materialize(
            canonical=canonical, indices=indices, destination=scratch / name
        )
        record.update({"lifetime": "transient", "cache_hit": False})
        yield SuppressedPayload(path=scratch / name, record=record)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def build_suppressed_render_package(
    *,
    canonical_ply_path: str | Path,
    receipts: Sequence[Mapping[str, Any]],
    destination: str | Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the persisted payload plus the digest chain a closed renderer needs."""

    if not receipts:
        raise GaussianSuppressionRenderError(["suppression_package_requires_receipts"])
    canonical, indices, composite = _resolve(canonical_ply_path, receipts)
    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = _materialize(
        canonical=canonical,
        indices=indices,
        destination=output / _payload_name(canonical, composite),
    )
    package: dict[str, Any] = {
        "schema_version": SUPPRESSED_PAYLOAD_SCHEMA_VERSION,
        "status": "suppressed_render_package_ready",
        "canonical_scan": {
            "path": str(canonical),
            "sha256": _sha256_file(canonical),
            "vertex_count": payload["canonical_vertex_count"],
        },
        "suppression": {
            "task_ids": list(composite["task_ids"]) if composite else [],
            "receipt_digests": [
                str(receipt.get("receipt_digest")) for receipt in receipts
            ],
            "suppressed_index_count": int(indices.size),
            "suppressed_index_digest": composite["suppressed_index_digest"]
            if composite
            else None,
            "composite_digest": composite["composite_digest"] if composite else None,
        },
        "payload": {
            key: value
            for key, value in payload.items()
            if key not in {"retained_indices"}
        },
        "claim_boundary": {
            "canonical_scan_modified": False,
            "payload_is_regenerable_cache": True,
            "payload_is_not_a_capture_artifact": True,
            "suppression_is_visibility_not_ownership": True,
        },
        "package_digest": "",
    }
    if generated_at is not None:
        package["generated_at"] = generated_at
    package["package_digest"] = canonical_digest(
        package, digest_field="package_digest"
    )
    write_json(output / "suppressed_render_package.json", package)
    return package


__all__ = [
    "PAYLOAD_BASENAME",
    "SUPPRESSED_PAYLOAD_SCHEMA_VERSION",
    "GaussianSuppressionRenderError",
    "SuppressedPayload",
    "build_suppressed_render_package",
    "suppressed_render_payload",
    "suppression_render_mask",
]
