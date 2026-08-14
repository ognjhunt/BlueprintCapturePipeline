"""Load the data-driven registry of admissible image-editing backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


REGISTRY_SCHEMA_VERSION = "image_editor_backend_registry.v1"
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "arm_decision_proof_v1"
    / "manifests"
    / "image_editor_backends.v1.json"
)
NO_DIRECT_EDITOR = "none"
ARTIFIXER_DIRECT_CAPABILITY = "artifixer_direct"
SEMANTIC_TEACHER_IMAGE_EDIT_CAPABILITY = "semantic_teacher_image_edit"
SUPPORTED_CAPABILITIES = frozenset(
    {ARTIFIXER_DIRECT_CAPABILITY, SEMANTIC_TEACHER_IMAGE_EDIT_CAPABILITY}
)
REQUIRED_FIELDS = (
    "backend_id",
    "capability",
    "model_identity",
    "license",
    "license_url",
    "commercial_use_permitted",
    "recorded_on",
)


class ImageEditorRegistryError(ValueError):
    """The registry could not be read, or an entry was not admissible."""


def _entries(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ImageEditorRegistryError("image_editor_registry_schema_invalid")
    rows = payload.get("backends")
    if not isinstance(rows, list) or not rows:
        raise ImageEditorRegistryError("image_editor_registry_empty")
    return rows


def load_registry(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Read the registry and refuse entries with unrecorded terms."""

    resolved = Path(path or DEFAULT_REGISTRY_PATH).expanduser()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImageEditorRegistryError("image_editor_registry_unreadable") from exc
    if not isinstance(payload, dict):
        raise ImageEditorRegistryError("image_editor_registry_unreadable")

    registry: dict[str, dict[str, Any]] = {}
    for row in _entries(payload):
        if not isinstance(row, Mapping):
            raise ImageEditorRegistryError("image_editor_registry_entry_invalid")
        missing = [field for field in REQUIRED_FIELDS if row.get(field) in (None, "")]
        if missing:
            raise ImageEditorRegistryError(
                f"image_editor_registry_entry_incomplete:{','.join(sorted(missing))}"
            )
        if not isinstance(row.get("commercial_use_permitted"), bool):
            raise ImageEditorRegistryError("image_editor_registry_terms_unrecorded")
        if row.get("capability") not in SUPPORTED_CAPABILITIES:
            raise ImageEditorRegistryError("image_editor_registry_capability_invalid")
        backend_id = str(row["backend_id"])
        if backend_id == NO_DIRECT_EDITOR:
            raise ImageEditorRegistryError("image_editor_registry_reserved_backend_id")
        if backend_id in registry:
            raise ImageEditorRegistryError(
                f"image_editor_registry_duplicate:{backend_id}"
            )
        registry[backend_id] = dict(row)
    return registry


def registered_backend_ids(
    path: str | Path | None = None, *, capability: str | None = None
) -> frozenset[str]:
    registry = load_registry(path)
    if capability is None:
        return frozenset(registry)
    if capability not in SUPPORTED_CAPABILITIES:
        raise ImageEditorRegistryError("image_editor_registry_capability_invalid")
    return frozenset(
        backend_id
        for backend_id, row in registry.items()
        if row["capability"] == capability
    )


def admissible_for_delivery(
    backend_id: str, *, path: str | Path | None = None
) -> bool:
    """Return whether recorded terms allow customer-facing outputs."""

    registry = load_registry(path)
    if backend_id not in registry:
        raise ImageEditorRegistryError(
            f"image_editor_backend_unregistered:{backend_id}"
        )
    return bool(registry[backend_id]["commercial_use_permitted"])


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "ARTIFIXER_DIRECT_CAPABILITY",
    "ImageEditorRegistryError",
    "NO_DIRECT_EDITOR",
    "REGISTRY_SCHEMA_VERSION",
    "REQUIRED_FIELDS",
    "SEMANTIC_TEACHER_IMAGE_EDIT_CAPABILITY",
    "SUPPORTED_CAPABILITIES",
    "admissible_for_delivery",
    "load_registry",
    "registered_backend_ids",
]
