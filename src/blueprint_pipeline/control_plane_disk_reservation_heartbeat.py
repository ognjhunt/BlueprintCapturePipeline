"""Keep a disk reservation live while an individual copy blocks on I/O."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from .control_plane_disk_budget import ControlPlaneDiskBudgetError, DiskReservation


class ReservationHealth:
    def __init__(self, failures: list[Exception]) -> None:
        self._failures = failures

    def check(self) -> None:
        if self._failures:
            raise ControlPlaneDiskBudgetError(
                "control_plane_disk_budget_heartbeat_failed"
            ) from self._failures[0]


@contextmanager
def keep_reservation_live(
    reservation: DiskReservation, *, interval_seconds: float | None = None
) -> Iterator[ReservationHealth]:
    """Renew well before TTL, and surface a failed renewal after the copy exits."""

    interval = interval_seconds if interval_seconds is not None else min(
        15 * 60, reservation.ttl_seconds / 4
    )
    if interval <= 0 or interval >= reservation.ttl_seconds:
        raise ControlPlaneDiskBudgetError("control_plane_disk_budget_heartbeat_invalid")
    stopped = threading.Event()
    failures: list[Exception] = []
    health = ReservationHealth(failures)

    def pulse() -> None:
        while not stopped.wait(interval):
            try:
                reservation.renew()
            except Exception as exc:
                failures.append(exc)
                stopped.set()
                return

    thread = threading.Thread(target=pulse, name="disk-reservation-heartbeat", daemon=True)
    thread.start()
    try:
        yield health
    finally:
        stopped.set()
        thread.join()
        health.check()
