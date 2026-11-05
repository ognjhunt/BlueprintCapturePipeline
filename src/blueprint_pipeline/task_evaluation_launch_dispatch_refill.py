"""Keep free dispatcher workers useful while a long paid run holds another.

The launch dispatcher is a systemd oneshot that claims pending launches only
when it starts, then blocks until every claimed paid run is terminal (up to the
unit's 9 h). With ``--max-concurrency`` above one, a website job that arrives
while another job's paid run is in flight would still wait for that whole run.
``--refill-poll-seconds`` (opt-in) re-reads ``pending/`` every interval and
claims new requests into free workers, up to ``--max-messages`` in total.
Every claimed request still goes through ``dispatch_launch_request`` and its
standing authorization exactly as before; this changes only *when* a request
is claimed, never what admits it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from .launch_immutable_input_writer import TaskEvaluationLaunchError

MAX_REFILL_POLL_SECONDS = 3600


def require_refill_poll(value: object) -> None:
    """``None`` (refill off) or a positive interval of at most an hour."""

    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < value <= MAX_REFILL_POLL_SECONDS
    ):
        raise TaskEvaluationLaunchError("launch_dispatch_refill_poll_invalid")


def dispatch_with_refill(
    *,
    initial: Sequence[Path],
    dispatch: Callable[[Path], dict[str, Any]],
    claim: Callable[[Path], Path | None],
    pending_sources: Callable[[], Sequence[Path]],
    max_concurrency: int,
    max_messages: int,
    poll_seconds: float,
) -> list[dict[str, Any]]:
    """Run claimed launches on ``max_concurrency`` workers, refilling from pending.

    Requests already claimed but not yet running wait their turn (the order is
    the stable pending order); new pending requests are claimed only into a
    free worker, so nothing sits in ``processing/`` waiting behind a long run.
    """

    queued = list(initial)
    claimed_total = len(queued)
    results: dict[int, dict[str, Any]] = {}
    running: dict[Any, int] = {}
    order = 0
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        while True:
            while queued and len(running) < max_concurrency:
                running[executor.submit(dispatch, queued.pop(0))] = order
                order += 1
            if not running:
                break
            done, _ = wait(list(running), timeout=poll_seconds, return_when=FIRST_COMPLETED)
            for future in done:
                results[running.pop(future)] = future.result()
            free = max_concurrency - len(running) - len(queued)
            if free > 0 and claimed_total < max_messages:
                for source in pending_sources():
                    if free <= 0 or claimed_total >= max_messages:
                        break
                    claimed = claim(source)
                    if claimed is None:
                        continue
                    queued.append(claimed)
                    claimed_total += 1
                    free -= 1
    return [results[index] for index in sorted(results)]


def run_claimed_launches(
    *,
    initial: Sequence[Path],
    dispatch: Callable[[Path], dict[str, Any]],
    claim: Callable[[Path], Path | None],
    pending_sources: Callable[[], Sequence[Path]],
    max_concurrency: int,
    max_messages: int,
    refill_poll_seconds: float | None,
) -> list[dict[str, Any]]:
    """Serial, overlapped, or overlapped-with-refill dispatch of claimed launches."""

    if max_concurrency == 1:
        return [dispatch(claimed) for claimed in initial]
    if refill_poll_seconds is None:
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            # executor.map preserves the stable pending-file order in the
            # queue receipt while the independently claimed launches overlap.
            return list(executor.map(dispatch, initial))
    return dispatch_with_refill(
        initial=initial, dispatch=dispatch, claim=claim, pending_sources=pending_sources,
        max_concurrency=max_concurrency, max_messages=max_messages,
        poll_seconds=float(refill_poll_seconds),
    )
