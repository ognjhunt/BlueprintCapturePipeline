"""Seal rigid task-control results without weakening diagnostic refusal."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

def finalize_rigid_task_controls(
    *,
    result: dict[str, Any],
    selection: str,
    diagnostic_only: bool,
    control_plan: Mapping[str, Any],
    announce: Callable[[str, str], None],
    pair: Mapping[str, Any] | None = None,
    episode: Mapping[str, Any] | None = None,
) -> int:
    """Record one rigid control result and preserve every qualification gate."""

    if pair is not None:
        result["control_pair"] = pair
        result["controls_qualified"] = (
            False if diagnostic_only else pair["cell_admitted_for_policy_execution"]
        )
        result["blockers"].extend(pair["policy_execution_blockers"])
    elif episode is not None:
        result.update(
            {
                "control_episode": episode,
                "selected_control_id": selection,
                "selected_control_passed": episode.get("control_passed"),
                "deterministic_task_succeeded": (episode.get("score") or {}).get(
                    "task_succeeded"
                ),
                "controls_qualified": False,
            }
        )
        result["blockers"].extend(episode["blockers"])
        if episode.get("control_passed") is not True:
            result["blockers"].append(f"selected_control_failed:{selection}")
    else:
        raise ValueError("rigid_task_control_result_missing")
    if diagnostic_only:
        result.update(
            {
                "controls_qualified": False,
                "diagnostic_only": True,
                "development_only": True,
                "qualification_effect": "none",
                "upstream_construction_blockers": list(
                    control_plan["upstream_construction_blockers"]
                ),
                "claim_boundary": control_plan["claim_boundary"],
            }
        )
        result["blockers"].extend(control_plan["upstream_construction_blockers"])
    result["blockers"] = sorted(set(result["blockers"]))
    result["status"] = (
        "diagnostic_completed"
        if diagnostic_only
        else ("completed" if not result["blockers"] else "blocked")
    )
    result["phase_reached"] = "selected_control_complete"
    announce(
        "required_controls",
        "completed" if diagnostic_only or not result["blockers"] else "blocked",
    )
    return 0 if result["status"] in {"completed", "diagnostic_completed"} else 1
