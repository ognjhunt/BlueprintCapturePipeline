"""Beta intake effect checkpoints; no new lease or retry authority is created."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from .common import PipelineError


class HandoffLeaseError(PipelineError):
    """The current handoff must stop new effects and retain reconciliation state."""

    def __init__(self, state: str, boundary: str = "") -> None:
        self.state = state
        self.boundary = boundary
        super().__init__(f"handoff_job_lease_{state}")


_CHECKPOINT: ContextVar[Callable[[str], None] | None] = ContextVar(
    "blueprint_handoff_lease_checkpoint", default=None
)


def handoff_lease_checkpoint(boundary: str) -> None:
    """Refuse the next stage/effect when this invocation has lost its lease.

    Standalone callers without a handoff retain their existing authority checks.
    An already-started resource must still run its own finally/teardown path.
    """
    check = _CHECKPOINT.get()
    if check is not None:
        check(boundary)


@contextmanager
def bind_handoff_lease(check: Callable[[str], None]) -> Iterator[None]:
    token = _CHECKPOINT.set(check)
    try:
        yield
    finally:
        _CHECKPOINT.reset(token)
