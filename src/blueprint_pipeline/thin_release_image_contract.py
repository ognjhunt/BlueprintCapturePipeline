"""Measure the frequently pulled OCI delta above a cached foundation."""

from __future__ import annotations

from typing import Any, Mapping


def build_thin_release_contract(
    release: Mapping[str, Any],
    foundation: Mapping[str, Any],
    *,
    max_release_bytes: int = 2 * 1024**3,
) -> dict[str, Any]:
    blockers: list[str] = []
    release_layers = {
        str(item.get("digest")): int(item.get("size_bytes"))
        for item in release.get("layers", [])
        if isinstance(item, Mapping)
        and isinstance(item.get("digest"), str)
        and type(item.get("size_bytes")) is int
    }
    foundation_layers = {
        str(item.get("digest"))
        for item in foundation.get("layers", [])
        if isinstance(item, Mapping) and isinstance(item.get("digest"), str)
    }
    if release.get("status") != "completed" or not release_layers:
        blockers.append("thin_release_registry_layers_missing")
    if foundation.get("status") != "completed" or not foundation_layers:
        blockers.append("foundation_registry_layers_missing")
    if foundation_layers and not foundation_layers.issubset(release_layers):
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
        "foundation_image_ref": foundation.get("resolved_digest_ref"),
        "release_image_ref": release.get("resolved_digest_ref"),
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
