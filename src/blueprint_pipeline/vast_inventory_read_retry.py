"""Bounded retry for rate-limited Vast inventory reads, never mutations."""
from __future__ import annotations

from email.utils import parsedate_to_datetime
import json
import logging
import math
import time
from urllib.error import HTTPError

from .transport_retry_policy import bounded_read_retry

logger = logging.getLogger(__name__)


class _InventoryRateLimited(Exception):
    def __init__(self, original):
        super().__init__("vast_inventory_read_rate_limited")
        self.original = original


def inventory_read_retry(*, sleep=None, evidence_hook=None):
    """At most three GETs with existing request timeouts and bounded waits.

    The shared 30-second retry stop is evaluated after a failed request; it
    is not an end-to-end deadline. Each server-directed wait is at most 10s.
    Launchers must recheck their watchdog after inventory reads.
    """
    sleeper = sleep or time.sleep
    retry_after = [0.0]

    def evidence(row):
        safe = {**row, "http_status": 429, "provider_mutation_performed": False}
        safe["delay_seconds"] = max(float(row.get("delay_seconds") or 0), retry_after[0])
        if evidence_hook is not None:
            evidence_hook(safe)
        else:
            logger.warning("vast_inventory_read_retry %s", json.dumps(safe, sort_keys=True))

    def decorator(call):
        def once():
            try:
                return call()
            except HTTPError as exc:
                if exc.code != 429:
                    raise
                raw = (exc.headers or {}).get("Retry-After", "0")
                try:
                    delay = float(raw)
                except (TypeError, ValueError):
                    try:
                        delay = parsedate_to_datetime(raw).timestamp() - time.time()
                    except (TypeError, ValueError, OverflowError):
                        delay = 0.0
                if not math.isfinite(delay) or delay > 10:
                    raise  # Never retry sooner than a longer server-directed delay.
                retry_after[0] = max(0.0, delay)
                raise _InventoryRateLimited(exc) from exc

        retry = bounded_read_retry(operation="vast_inventory_get", exception_allowlist=(_InventoryRateLimited,),
            max_attempts=3, max_delay_seconds=30, evidence_hook=evidence,
            jitter_initial_seconds=0.5, jitter_max_seconds=2,
            sleep=lambda delay: sleeper(max(delay, retry_after[0])))

        def run():
            try:
                return retry(once)()
            except _InventoryRateLimited as exc:
                raise exc.original from None
        return run
    return decorator
