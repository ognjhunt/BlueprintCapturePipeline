"""Where a release's runtime provisioning spends its time, for the deploy receipt."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import time
from typing import Iterator

_CLOCK: ContextVar[dict | None] = ContextVar("provision_phase_clock", default=None)


@contextmanager
def phase_clock() -> Iterator[dict[str, float]]:
    """Collect ``mark_phase`` timings in the yielded mapping for the length of the block."""
    phases: dict[str, float] = {}
    token = _CLOCK.set({"phases": phases, "since": time.monotonic()})
    try:
        yield phases
    finally:
        _CLOCK.reset(token)


def mark_phase(name: str) -> None:
    """Record the seconds since the previous mark; does nothing outside a clock."""
    clock = _CLOCK.get()
    if clock is None:
        return
    now = time.monotonic()
    clock["phases"][name] = round(now - clock["since"], 1)
    clock["since"] = now
