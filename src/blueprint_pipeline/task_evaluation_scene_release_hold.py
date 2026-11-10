"""Keep a deploy from retiring a scene attempt whose paid launch already left progression.

Rule B for the paid scene configuration ("an attempt in flight finishes on the
release it started on"): once an attempt's launch is queued, running, in transit
through the WebApp, or ended without a reconciled receipt, the attempt is not
retired by a release change. The dispatcher runs a queued launch from its own
profile whatever release is live, and the completion join reads its receipt
under its own commit, so the job closes from that launch. Retiring it instead
rebuilt the attempt on the new release and could reserve and launch the same
paid construction a second time.

Read-only: nothing here writes, retries or settles. ``docs/architecture/release-binding.md``
maps every release check of the website chain.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .task_evaluation_scene_configuration_submission_inputs import read
from .task_evaluation_terminal_scene_attempt_settlement import _launch_state, launch_id_for_preparation

LAUNCH_QUEUED_STATUS = "scene_configuration_launch_queued"


def previous_launch_in_flight(*, activation_link_path: Path | None, launch_record: Mapping[str, Any] | None,
                              config: Mapping[str, Any]) -> str | None:
    """Why the attempt's paid launch may still run or spend, or None when it never left or has ended.

    ``activation_link_path`` is the attempt's verified activation link (None when
    no paid reservation was ever made, so no launch can exist). ``launch_record``
    is the progression's own ``scene_configuration_launch`` row, which names a
    launch the tick already submitted to the WebApp.
    """
    if activation_link_path is None:
        return None
    link = read(activation_link_path, digest_field="link_digest")
    launch_id = launch_id_for_preparation(str(link["preparation_id"]))
    execution = _launch_state(launch_id=launch_id, launch_execution_root=Path(config["launch_execution_root"]),
                              launch_queue_root=Path(config["launch_queue_root"]))
    if execution is None:
        # Queued, claimed, running, or a run directory without a terminal receipt.
        return "previous_release_launch_not_terminal"
    record = launch_record if isinstance(launch_record, Mapping) else {}
    if (execution.get("launch_never_queued") and record.get("status") == LAUNCH_QUEUED_STATUS
            and record.get("launch_id") == launch_id):
        # Submitted to the WebApp, not yet in the queue: the intake still accepts it.
        return "previous_release_launch_in_transit"
    return None
