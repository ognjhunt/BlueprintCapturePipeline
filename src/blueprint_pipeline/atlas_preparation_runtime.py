"""Captured source callbacks for active preparation adapters; no authority is issued."""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from .owned_capture_atlas_context import SCHEMA

def preparation_source_checkpoint(context: Mapping[str, Any]):
    """Owned provider adapters require the worker's fresh byte/source fence."""
    if context.get("schema_version") != SCHEMA:
        return None
    from .owned_capture_atlas_runtime import owned_atlas_source_checkpoint
    return owned_atlas_source_checkpoint(context)


def preparation_disclosure_opener(context: Mapping[str, Any], opener):
    """Reopen an owned source at the image request's actual network boundary.

    Capture the callback in the worker scope so SDK threads retain the same
    immutable source. Completed output reads and teardown remain possible.
    """
    checkpoint = preparation_source_checkpoint(context)
    if checkpoint is None:
        return opener

    def send(request, *args, **kwargs):
        method = request.get_method() if hasattr(request, "get_method") else "POST"
        if method not in {"GET", "HEAD", "DELETE"}:
            checkpoint("owned_image_provider_disclosure")
        return opener(request, *args, **kwargs)
    return send
