"""The words a policy is given for a task, built once from its labels.

Pure and dependency-free so every runtime bundle that renders an instruction can
ship it. A subject is named once, without its own location ("tray on the table"
is the tray), and a destination by the marked area the policy can see.
"""
from __future__ import annotations

import re
from typing import Any

_RELATIONS = frozenset({"on", "in", "into", "inside", "onto", "at", "from", "near", "under", "beside", "by",
                        "next", "with", "behind", "above", "below", "against", "over", "to", "for", "that",
                        "which", "and", "or"})

def object_label(label: Any) -> str | None:
    """A label's object noun phrase, without its location ("tray on the table" -> "tray")."""
    if not isinstance(label, str) or not label.strip():
        return None
    words = label.strip().split()
    end = next((index for index, word in enumerate(words) if word.lower().strip(",.;:") in _RELATIONS
                and index > 0), len(words))
    phrase = " ".join(words[:end]).strip(" ,.;:")
    phrase = re.sub(r"^(the|a|an)\s+", "", phrase, flags=re.IGNORECASE)
    return phrase or None


def placement_instruction(*, subject_label: Any, destination_label: Any, relation: str, colour: str) -> str:
    """Name the subject once, the destination once, and the marked area the policy can see.

    The subject's own location never leaks into it ("tray on the table" is the tray), so
    a destination on the surface the subject already stands on is not named twice.
    """
    subject = object_label(subject_label)
    destination = object_label(destination_label)
    if subject is None:
        raise ValueError("task_spec_subject_label_missing")
    if relation not in ("on", "inside"):
        raise ValueError("task_spec_destination_relation_invalid")
    if relation == "inside":
        if destination is None:
            raise ValueError("task_spec_destination_label_missing")
        return f"Pick up the {subject} and put it inside the {destination}."
    marked = f"the {colour} marked area" + (f" on the {destination}" if destination else "")
    return f"Pick up the {subject} and place it on {marked}."


__all__ = ["object_label", "placement_instruction"]
