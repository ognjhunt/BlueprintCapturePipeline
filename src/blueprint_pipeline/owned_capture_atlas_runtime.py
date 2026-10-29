"""Reopen an owned upload before each new provider disclosure.

This source fence grants no provider terms, allowance, execution, or metric
authority. The existing paid allocator and source-specific WebApp grant must
independently admit every paid stage. Reads, settlement and cleanup can finish
after processing authority ends.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .owned_capture_atlas_context import require_owned_capture_atlas_source_binding, validate_owned_capture_atlas_context


@dataclass(frozen=True)
class _SourceFence:
    context_digest: str
    check: Callable[[str], None]


_FENCE: ContextVar[_SourceFence | None] = ContextVar("owned_capture_atlas_source_fence", default=None)


def owned_atlas_source_checkpoint(context: Mapping[str, Any]) -> Callable[[str], None]:
    """Capture the installed source fence, including across later worker threads."""
    value = validate_owned_capture_atlas_context(context)
    fence = _FENCE.get()
    if fence is None or fence.context_digest != value["context_digest"]:
        raise ValueError("owned_capture_atlas_source_fence_required")
    return fence.check


@contextmanager
def bind_owned_atlas_source(
    *, state_root: Path, capture_store_root: Path, plan_id: str
) -> Iterator[dict[str, Any]]:
    """Only the worker's retained plan selects the source; no client paths do."""
    from .core.security_controls import strict_identifier
    from .reconstruction_control_plane import _read_object, _run_root, _source_binding

    key = strict_identifier(plan_id, field="plan_id", max_length=192)
    plan = _read_object(_run_root(state_root, key) / "artifacts" / "owned_atlas_preparation_plan.json",
                        code="owned_atlas_preparation_plan_missing")
    if (plan.get("schema_version") != "owned_capture_atlas_preparation_plan.v1"
            or plan.get("plan_id") != key
            or plan.get("plan_digest") != cross_runtime_canonical_digest(plan, digest_field="plan_digest")):
        raise ValueError("owned_capture_atlas_plan_binding_invalid")
    context = validate_owned_capture_atlas_context(plan["source_context"])
    if context["request_id"] != key or plan.get("source_video_digest") != context["source_capture"]["capture_digest"]:
        raise ValueError("owned_capture_atlas_plan_binding_invalid")

    def check(boundary: str) -> None:
        source = _source_binding(capture_store_root=capture_store_root,
                                 capture_session_id=context["capture_id"], intake_id=context["intake_id"])
        require_owned_capture_atlas_source_binding(context, source=source, plan_id=key)

    token = _FENCE.set(_SourceFence(context["context_digest"], check))
    try:
        # Callers may build derived task data; they never receive the private
        # snapshot to which existing upload/inference callbacks remain bound.
        yield deepcopy(context)
    finally:
        _FENCE.reset(token)
