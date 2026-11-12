"""Whether a scene review's abstention is final: a leaf the control-plane lane can read.

Owner direction 2026-11-11: a scene review abstains only when the video cannot serve at all (no frame
shows the task object or its region, or no frame is usable). An abstention made under older
instructions, or for a doubt the current instructions propose through, is reviewed once more; only a
final one ends the job, as an ask for a new video. Standard library only: no planner, pipeline or
provider import, so status readers and contracts can use it.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: The scene review's instruction revision, recorded in each new receipt's binding.
INSTRUCTIONS_REVISION = "atlas_scene_planner.r2"
FINAL_ABSTENTION_REASONS = frozenset({"task_object_or_region_not_visible", "frames_unusable"})
#: Where an Atlas job keeps its observed-room selection (``atlas_observed_room``).
SELECTION = "observed_room/selection.json"


def final_abstention(receipt: Any) -> bool:
    """A retained review that abstained under the current instructions for a cause the video cannot get past."""
    if not isinstance(receipt, Mapping):
        return False
    proposal, binding = receipt.get("proposal") or {}, receipt.get("binding") or {}
    reasons = proposal.get("abstention_reasons")
    return (receipt.get("status") == "completed" and proposal.get("status") == "abstained"
            and binding.get("instructions_revision") == INSTRUCTIONS_REVISION
            and isinstance(reasons, list) and bool(reasons) and set(reasons) <= FINAL_ABSTENTION_REASONS)


def retained_selection(job_dir: Path) -> dict[str, Any] | None:
    """The Atlas job's retained observed-room selection, or None."""
    path = Path(job_dir) / SELECTION
    if path.is_symlink():
        raise ValueError("atlas_observed_room_path_unsafe")
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else None


def final_room(job_dir: Path) -> bool:
    """Whether the job's observed room rests on a final abstention; otherwise its task is reviewed once more."""
    selection = retained_selection(job_dir)
    return isinstance(selection, dict) and final_abstention(selection.get("planner_receipt"))


def abstention_reasons(job_dir: Path) -> list[str]:
    selection = retained_selection(job_dir) or {}
    reasons = ((selection.get("planner_receipt") or {}).get("proposal") or {}).get("abstention_reasons")
    return [str(reason) for reason in reasons] if isinstance(reasons, list) else []


def reopens_for_review(job_dir: Path) -> str | None:
    """The retained selection's digest when a ready observed-room job must review its task once more, else None."""
    selection = retained_selection(job_dir)
    if not isinstance(selection, dict) or final_room(job_dir):
        return None
    return selection.get("digest")
