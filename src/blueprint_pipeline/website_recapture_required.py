"""A website preparation that only a new video can move on: it ends at once and asks the site for one.

Two stops of scene preparation are decided by the video alone, so the same video always gives the
same answer:

- ``website_front_normal_undetermined``: the object was filmed only from above, so no side of it was
  seen and which way it faces cannot be read (``website_task_preparation``);
- the object's observed shape fits the reconstructed scene nowhere, or in several places, or not
  where the reconstruction's own anchor puts it (``REGISTRATION_CODES``).

Before 2026-11-11 the handoff recorded them as ``awaiting_inputs``, the stage failed with the code,
and the preparation retry re-ran the same video six times over about two days; the site read nothing.
Now the handoff records the stop (``FIELD``), the stage completes as a hold (Pub/Sub acknowledges it
and nothing retries it), and preparation status v7 names the code as the job's outcome, which the
website turns into what to film. Nothing is bought: the stop is found after the scene is built, and
the hold refuses no gate and holds none.

A new video is a new capture: the owner's "update video" on the job page issues a replacement
capture, which the website makes the active one, and that capture runs its own Atlas job and
preparation from the start. The capture that ended here is not run again.

Not here, on purpose: ``website_registration_anchor_frame_missing`` and
``website_registration_anchor_frame_invalid`` name a binding between Blueprint's own records, not
anything the video lacks; ``website_registration_scale_conflicts_declared`` is a preparation blocker
read as missing task evidence (``website_preparation_outcome``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest

__all__ = ["CODES", "FIELD", "FRONT_NORMAL_CODE", "REGISTRATION_CODES", "SCHEMA", "record", "retains_recapture_required"]

SCHEMA = "website_recapture_required.v1"
FIELD = "recapture_required"
FRONT_NORMAL_CODE = "website_front_normal_undetermined"
REGISTRATION_CODES = frozenset({
    "website_registration_ambiguous", "website_registration_poor_fit",
    "website_registration_conflicts_provider_anchor", "website_registration_anchor_deviation",
    "website_registration_ground_plane_inconsistent",
})
CODES = REGISTRATION_CODES | {FRONT_NORMAL_CODE}


def record(code: str, *, task_context_digest: str) -> dict[str, Any]:
    """What the handoff keeps for a stop only a new video can clear."""
    if code not in CODES:
        raise ValueError("website_recapture_code_unknown")
    return {"schema_version": SCHEMA, "code": code, "task_context_digest": task_context_digest,
            "next_actor": "site", "provider_mutation_performed": False}


def retains_recapture_required(handoff: Mapping[str, Any], task_context: Mapping[str, Any]) -> bool:
    """A completed scene-preparation stage that ended on a stop only a new video can clear."""
    if not isinstance(handoff, Mapping) or not isinstance(task_context, Mapping):
        return False
    stop = handoff.get(FIELD)
    return (isinstance(stop, Mapping) and stop.get("schema_version") == SCHEMA and stop.get("code") in CODES
            and stop.get("task_context_digest") == task_context.get("context_digest")
            and stop.get("provider_mutation_performed") is False
            and handoff.get("schema_version") == "website_scene_handoff.v1"
            and handoff.get("digest") == canonical_digest(handoff, digest_field="digest")
            and all(handoff.get(key) == task_context.get(key) for key in ("scene_id", "capture_id"))
            and handoff.get("status") == "needs_input" and handoff.get("blockers") == [stop["code"]]
            and not any(key in handoff for key in ("source_registration", "website_intake_outbox")))
