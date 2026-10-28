"""Beta intake effect checkpoints; no new lease or retry authority is created."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar

from .common import PipelineError, write_json


class HandoffLeaseError(PipelineError):
    """The current handoff must stop new effects and retain reconciliation state."""

    def __init__(self, state: str, boundary: str = "") -> None:
        self.state = state
        self.boundary = boundary
        super().__init__(f"handoff_job_lease_{state}")


_CHECKPOINT: ContextVar[Callable[[str], None] | None] = ContextVar(
    "blueprint_handoff_lease_checkpoint", default=None
)
_EFFECT: ContextVar[Callable[[str], object] | None] = ContextVar("blueprint_handoff_effect_scope", default=None)


def handoff_lease_checkpoint(boundary: str) -> None:
    """Refuse the next stage/effect when this invocation has lost its lease.

    Standalone callers without a handoff retain their existing authority checks.
    An already-started resource must still run its own finally/teardown path.
    """
    check = _CHECKPOINT.get()
    if check is not None:
        check(boundary)


@contextmanager
def handoff_effect(boundary: str) -> Iterator[None]:
    """Hold the current generation through an atomic local promotion."""
    effect = _EFFECT.get()
    if effect is None:
        handoff_lease_checkpoint(boundary)
        yield
    else:
        with effect(boundary):
            yield


def guarded_handoff_write_json(path: object, value: Mapping) -> None:
    with handoff_effect("shared_artifact_promotion"):
        write_json(path, value)


@contextmanager
def bind_handoff_lease(check: Callable[[str], None], effect: Callable[[str], object] | None = None) -> Iterator[None]:
    token = _CHECKPOINT.set(check)
    effect_token = _EFFECT.set(effect)
    try:
        yield
    finally:
        _EFFECT.reset(effect_token)
        _CHECKPOINT.reset(token)
