"""Refuse a new website upstream reservation while a scope's stop check says the job is held.

Kept free of imports so the reservation path (``website_task_context``) reaches nothing
new. ``website_early_task_hold.stop_paid_work_when_held`` installs the check.
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from contextvars import ContextVar


class TaskPreparationHeld(Exception):
    """A new upstream reservation refused because the capture's job is held for task preparation.

    Not a failure: the lane ends with a held outcome. Deliberately not a ``ValueError``,
    so the per-target recoveries that catch those do not turn it into another attempt.
    """


_CHECK: ContextVar[Callable[[], bool] | None] = ContextVar("blueprint_website_paid_work_stop", default=None)


@contextlib.contextmanager
def paid_work_stop(check: Callable[[], bool] | None) -> Iterator[None]:
    """Within this scope, ``check()`` true refuses each new website upstream reservation."""
    token = _CHECK.set(check)
    try:
        yield
    finally:
        _CHECK.reset(token)


def refuse_paid_work_when_held() -> None:
    """Called before a website upstream reservation; raises ``TaskPreparationHeld`` while the job is held."""
    check = _CHECK.get()
    if check is not None and check():
        raise TaskPreparationHeld("website_task_preparation_held")
