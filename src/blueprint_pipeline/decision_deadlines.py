"""The instants at which a decision just made against the clock would change.

An owner execution window, a visual-review grant and a preparation grant are
valid until an expiry (a preparation grant also not before it was issued). A
decision made against the clock flips at that instant with no file changing,
so anything that caches the decision needs the instant. Code that compares such
an instant with the clock reports it here; the runtime env guard's
catalog-phase verdict stops serving a cached pass before the earliest one.

Reporting is free when no collector is open, and never changes the decision.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TypeVar

_Instant = TypeVar("_Instant")
_COLLECTORS: ContextVar[tuple[list[float], ...]] = ContextVar("decision_deadlines", default=())


def note_deadline(instant: _Instant) -> _Instant:
    """Report an epoch at which a decision against the clock flips; returns it unchanged.

    Anything but a finite number is ignored here; the caller's own validation
    decides what such a value means.
    """
    if (
        not isinstance(instant, bool)
        and isinstance(instant, (int, float))
        and math.isfinite(instant)
    ):
        for collector in _COLLECTORS.get():
            collector.append(float(instant))
    return instant


@contextmanager
def collecting_deadlines() -> Iterator[list[float]]:
    """Every instant reported inside the block; blocks nest, and each sees its own and inner ones."""
    found: list[float] = []
    token = _COLLECTORS.set(_COLLECTORS.get() + (found,))
    try:
        yield found
    finally:
        _COLLECTORS.reset(token)


__all__ = ["collecting_deadlines", "note_deadline"]
