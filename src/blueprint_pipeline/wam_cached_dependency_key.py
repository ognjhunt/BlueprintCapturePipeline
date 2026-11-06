"""The content-addressed object key one cached WAM provider dependency lives under.

``wam_provider_object_store.stage_cached_runtime_dependency_object_store`` writes
an immutable dependency once under this key and reuses it after; the G1 runtime
source restore (``native_g1_runtime_source_restore``) reads a retained copy back
by the same key.
"""

from __future__ import annotations


def cached_dependency_object_key(
    key_prefix: str, expected_sha256: str, artifact_kind: str = "runtime_dependency"
) -> str:
    """The run's key prefix, the dependency's sha256 and its kind name the object."""

    expected = str(expected_sha256 or "").removeprefix("sha256:")
    safe_prefix = key_prefix.strip("/ ") or "blueprint/wam-provider"
    if artifact_kind == "runtime_dependency":
        return f"{safe_prefix}/runtime-dependencies/sha256/{expected}.zip"
    if artifact_kind == "g1_checkpoint":
        return f"{safe_prefix}/g1-checkpoints/sha256/{expected}.bin"
    raise ValueError("cached_dependency_artifact_kind_invalid")
