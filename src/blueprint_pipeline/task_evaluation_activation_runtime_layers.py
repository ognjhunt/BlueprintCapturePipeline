"""Validate external runtime layers while activating a prepared launch."""

from __future__ import annotations

import zipfile
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from .task_evaluation_native_arena_preparation_adapter import (
    MANIFEST_NAME as ADAPTER_MANIFEST_NAME,
    RUNTIME_SOURCE_LAYER_CONTRACT_PREFIX,
    TaskEvaluationNativeArenaAdapterError,
    read_runtime_source_external_layers,
)


class ActivationRuntimeLayerError(ValueError):
    """A prepared runtime wrapper or its reference set is inconsistent."""


def collect_request_references(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect every canonical URI, digest, and size contract recursively."""

    rows: list[dict[str, Any]] = []

    def visit(node: Any, path: tuple[str, ...]) -> None:
        if isinstance(node, Mapping):
            if set(node) == {"uri", "digest", "size_bytes"}:
                rows.append(
                    {
                        "contract_path": ".".join(path),
                        "uri": str(node["uri"]),
                        "digest": str(node["digest"]),
                        "size_bytes": int(node["size_bytes"]),
                    }
                )
                return
            for key, child in node.items():
                visit(child, (*path, str(key)))
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for index, child in enumerate(node):
                visit(child, (*path, str(index)))

    visit(value, ())
    return rows


def derive_runtime_source_external_layer_references(
    *, request: Mapping[str, Any], wrapper_path: Path
) -> list[dict[str, Any]]:
    """Validate a typed wrapper and derive its external-layer contracts."""

    if not zipfile.is_zipfile(wrapper_path):
        return []
    try:
        with zipfile.ZipFile(wrapper_path) as archive:
            if ADAPTER_MANIFEST_NAME not in archive.namelist():
                return []
    except (OSError, zipfile.BadZipFile):
        return []
    try:
        layers = read_runtime_source_external_layers(
            bundle_path=wrapper_path,
            request=request,
        )
    except TaskEvaluationNativeArenaAdapterError as exc:
        raise ActivationRuntimeLayerError(
            f"launch_activation_runtime_source_bundle_invalid:{exc}"
        ) from exc
    return [
        {
            "contract_path": f"{RUNTIME_SOURCE_LAYER_CONTRACT_PREFIX}{index}",
            "uri": layer["uri"],
            "digest": layer["sha256"],
            "size_bytes": layer["size_bytes"],
        }
        for index, layer in enumerate(layers)
    ]


def augment_runtime_source_external_layer_references(
    *,
    request: Mapping[str, Any],
    materialized_references: Mapping[str, Path],
    expected_references: MutableMapping[str, tuple[str, int]],
) -> dict[str, str]:
    """Add exact layer identities and return their required URIs."""

    runtime_source = materialized_references.get(
        "execution_adapter.runtime_source_bundle"
    )
    if runtime_source is None:
        return {}
    uris: dict[str, str] = {}
    for reference in derive_runtime_source_external_layer_references(
        request=request,
        wrapper_path=runtime_source,
    ):
        contract_path = str(reference["contract_path"])
        if contract_path in expected_references:
            raise ActivationRuntimeLayerError(
                "launch_activation_runtime_source_layer_contract_conflict"
            )
        expected_references[contract_path] = (
            str(reference["digest"]),
            int(reference["size_bytes"]),
        )
        uris[contract_path] = str(reference["uri"])
    return uris


def runtime_source_reference_matches(
    *,
    contract_path: str,
    row: Mapping[str, Any],
    expected: tuple[str, int],
    external_layer_uris: Mapping[str, str],
) -> bool:
    """Whether one prepared row matches digest, size, and layer URI."""

    return (row.get("digest"), row.get("size_bytes")) == expected and (
        contract_path not in external_layer_uris
        or row.get("uri") == external_layer_uris[contract_path]
    )


__all__ = [
    "ActivationRuntimeLayerError",
    "augment_runtime_source_external_layer_references",
    "collect_request_references",
    "derive_runtime_source_external_layer_references",
    "runtime_source_reference_matches",
]
