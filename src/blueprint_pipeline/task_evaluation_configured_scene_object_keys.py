"""Configured-scene object-store key names, with no client, credential or write in reach.

The object store (``task_evaluation_configured_scene_object_store``) and the native-Arena adapter share
these names; the adapter reads them to bind runtime-source layer URIs.  Keeping them here keeps the
object store, and the admission it imports, out of what a compile loads (plan 14 review M3: the remote
worker's compile stage).
"""

from __future__ import annotations

DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/configured-scenes"
LARGE_ARTIFACT_KEY_PREFIX = f"{DEFAULT_KEY_PREFIX}/artifacts"
# Runtime-source wrapper layers are published under this artifact kind; the
# wrapper builder embeds the resulting URI, so the two must agree exactly.
EXTERNAL_LAYER_ARTIFACT_KIND = "native-runtime-source-layer"

__all__ = ["DEFAULT_KEY_PREFIX", "EXTERNAL_LAYER_ARTIFACT_KIND", "LARGE_ARTIFACT_KEY_PREFIX"]
