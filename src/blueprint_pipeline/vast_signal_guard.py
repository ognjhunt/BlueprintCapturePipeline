"""One stop signal per Vast provider run.

The adapter turns its first SIGINT/SIGTERM into an interrupt that unwinds into
its own teardown. That teardown is the stop. A repeated signal, or any signal
once teardown has begun, is recorded and ignored: a second interrupt there
would skip the instance DELETE (which catches only ``Exception``) and the run's
receipts, leaving the instance billing until the watchdog's hard TTL.
"""

from __future__ import annotations

from pathlib import Path

from .common import utc_now_iso, write_json

RECORD_FILENAME = "vast_signal_teardown_guard.json"


class OneStopSignal:
    """The run's one stop; afterwards every signal is recorded and ignored."""

    def __init__(self, job_dir: Path) -> None:
        self.record_path = Path(job_dir) / RECORD_FILENAME
        self.taken: str | None = None
        self.ignored: dict[str, int] = {}

    def absorbs(self, signum: int) -> bool:
        """False exactly once, for the stop itself; True for every later signal."""
        if self.taken is None:
            self.taken = "interrupt"
            return False
        key = str(signum)
        self.ignored[key] = self.ignored.get(key, 0) + 1
        try:
            write_json(self.record_path, {
                "schema_version": "vast_signal_teardown_guard.v1", "generated_at": utc_now_iso(),
                "status": "ignored_after_stop_or_teardown_start", "stop_taken_by": self.taken,
                "ignored_signal": signum, "ignored_signal_counts": dict(self.ignored),
                "raw_secret_values_recorded": False})
        except Exception:  # noqa: BLE001 - a record must never interrupt the teardown it protects
            pass
        return True

    def teardown(self) -> None:
        """Teardown has begun: from here on, every signal is ignored."""
        if self.taken is None:
            self.taken = "teardown"


__all__ = ["OneStopSignal", "RECORD_FILENAME"]
