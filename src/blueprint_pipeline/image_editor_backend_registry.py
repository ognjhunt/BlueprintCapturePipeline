"""Which image-editing backends this pipeline may use, and on what terms.

The set of admissible editors was a frozenset of three literals in the
ArtiFixer3D bundle module, so adopting a newly released model meant editing
code and shipping a release. The best image-editing models change every few
months; a pipeline that needs a code change to follow them will simply stop
following them.

So the set is data. Adding a backend is adding a row here.

It is deliberately not *just* a list of names. These are third-party models with
genuinely different terms -- some research-only, some non-commercial, some
permissive -- and this pipeline produces customer-facing artifacts, so the
terms travel with the name and an entry that does not record them is refused.
That keeps the seam open for the next model without opening it for a model
nobody checked the license on.

`commercial_use_permitted` is the field that decides whether a backend may
produce delivered work, and it is required rather than defaulted: an unset
value is the one case where guessing is worst.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

REGISTRY_SCHEMA_VERSION = "image_editor_backend_registry.v1"

#: The registry that ships with the repository.
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "arm_decision_proof_v1"
    / "manifests"
    / "image_editor_backends.v1.json"
)

#: The value meaning "no direct editor ran", which is not a backend.
NO_DIRECT_EDITOR = "none"

REQUIRED_FIELDS = (
    "backend_id",
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
    """Read the registry and refuse any entry with unrecorded terms."""

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
        backend_id = str(row["backend_id"])
        if backend_id == NO_DIRECT_EDITOR:
            raise ImageEditorRegistryError("image_editor_registry_reserved_backend_id")
        if backend_id in registry:
            raise ImageEditorRegistryError(f"image_editor_registry_duplicate:{backend_id}")
        registry[backend_id] = dict(row)
    return registry


def registered_backend_ids(path: str | Path | None = None) -> frozenset[str]:
    return frozenset(load_registry(path))


def admissible_for_delivery(backend_id: str, *, path: str | Path | None = None) -> bool:
    """Whether this backend's terms allow producing customer-facing work.

    A caller asking about an unregistered backend gets a refusal rather than a
    `False`, because "not allowed" and "never checked" should not look alike.
    """

    registry = load_registry(path)
    if backend_id not in registry:
        raise ImageEditorRegistryError(f"image_editor_backend_unregistered:{backend_id}")
    return bool(registry[backend_id]["commercial_use_permitted"])


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "ImageEditorRegistryError",
    "NO_DIRECT_EDITOR",
    "REGISTRY_SCHEMA_VERSION",
    "REQUIRED_FIELDS",
    "admissible_for_delivery",
    "load_registry",
    "registered_backend_ids",
]
