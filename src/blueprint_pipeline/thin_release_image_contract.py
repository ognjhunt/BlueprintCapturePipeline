"""Measure the frequently pulled OCI delta above a cached foundation."""

from __future__ import annotations

import re
from typing import Any, Mapping


_DIGEST_REF = re.compile(r"\A[^\s@]+@sha256:[0-9a-f]{64}\Z")


def _layer_rows(payload: Mapping[str, Any]) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    layers = payload.get("layers")
    if not isinstance(layers, list):
        return rows
    for item in layers:
        if not isinstance(item, Mapping):
            continue
        digest = item.get("digest")
        size = item.get("size_bytes")
        if isinstance(digest, str) and type(size) is int:
            rows.append((digest, size))
    return rows


def build_thin_release_contract(
    release: Mapping[str, Any],
    foundation: Mapping[str, Any],
    *,
    max_release_bytes: int = 2 * 1024**3,
) -> dict[str, Any]:
    blockers: list[str] = []
    release_layer_rows = _layer_rows(release)
    release_layers = dict(release_layer_rows)
    foundation_layer_rows = _layer_rows(foundation)
    foundation_layers = dict(foundation_layer_rows)
    release_ref = str(release.get("resolved_digest_ref") or "")
    foundation_ref = str(foundation.get("resolved_digest_ref") or "")
    if not _DIGEST_REF.fullmatch(release_ref):
        blockers.append("thin_release_registry_ref_not_digest_pinned")
    if not _DIGEST_REF.fullmatch(foundation_ref):
        blockers.append("foundation_registry_ref_not_digest_pinned")
    if release.get("status") != "completed" or not release_layers:
        blockers.append("thin_release_registry_layers_missing")
    if foundation.get("status") != "completed" or not foundation_layers:
        blockers.append("foundation_registry_layers_missing")
    foundation_prefix = release_layer_rows[: len(foundation_layer_rows)]
    if foundation_layer_rows and foundation_prefix != foundation_layer_rows:
        blockers.append("release_does_not_extend_exact_foundation_layers")
    delta = {
        digest: size
        for digest, size in release_layers.items()
        if digest not in foundation_layers
    }
    delta_bytes = sum(delta.values()) if not blockers else None
    within_budget = delta_bytes is not None and delta_bytes <= max_release_bytes
    if delta_bytes is not None and not within_budget:
        blockers.append("thin_release_compressed_delta_exceeds_budget")
    return {
        "schema_version": "groot_oscar_thin_release_image_contract.v1",
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "foundation_image_ref": foundation_ref or None,
        "release_image_ref": release_ref or None,
        "foundation_layer_count": len(foundation_layers),
        "release_layer_count": len(release_layers),
        "release_delta_layer_count": len(delta),
        "release_delta_compressed_size_bytes": delta_bytes,
        "release_delta_largest_layer_size_bytes": max(delta.values()) if delta else 0,
        "release_delta_budget_bytes": max_release_bytes,
        "release_delta_budget_passed": within_budget,
        "models_externalized": True,
        "claim_boundary": {
            "cached_foundation_bytes_are_not_frequently_pulled_release_bytes": True,
            "release_size_is_not_live_startup_proof": True,
        },
    }
