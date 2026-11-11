"""A refusal that the same inputs and the same code always repeat, and how a controller waits on it.

``job_retry_policy`` gives a refusal nobody classified six timed retries over about two days. That is
right for a refusal that may pass on its own (a provider's answer, a file still being written), and
wrong for one decided only by retained inputs and code: the same video and the same release give the
same refusal, and a re-run can only repeat work (capture-5b067498, 2026-11-06 to 11-07: levelling and
mesh registration refusals re-ran on the schedule and refused again each time).

A step raises ``DeterministicRefusal`` with its code, the step's revision constant and the digests of
the inputs it decided on. ``key`` binds the three. A controller keeps ``record()`` on the job and does
not run the step again while ``unchanged`` says the release it runs computes the same key: only a
deploy that changes the step's revision re-runs it, once, and a repeat refusal under the new revision
waits again. The job's own deadline still ends it; nothing here ends a job or holds a gate.

Pure apart from reading a step's revision constant from its module. Standard library only.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping
from typing import Any

__all__ = ["DeterministicRefusal", "SCHEMA", "STEP_REVISIONS", "current_revision", "find", "unchanged"]

SCHEMA = "deterministic_refusal.v1"

#: Each step that raises a deterministic refusal, and the module constant naming its decision's revision.
#: A release that changes how the step decides changes that constant, and so the key.
STEP_REVISIONS: dict[str, tuple[str, str]] = {
    "atlas_proposal_review": ("blueprint_pipeline.atlas_scene_planner", "PROPOSAL_REVIEW_REVISION"),
    "atlas_mesh_registration": ("blueprint_pipeline.atlas_mesh_registration", "REGISTRATION_REVISION"),
    "website_front_normal": ("blueprint_pipeline.website_task_preparation", "FRONT_NORMAL_REVISION"),
}


def _key(code: str, step: str, revision: str, inputs: Mapping[str, str]) -> str:
    body = json.dumps({"code": code, "step": step, "revision": revision, "inputs": dict(inputs)},
                      sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def current_revision(step: object) -> str | None:
    """The revision the running release has for ``step``; None for a step this release does not know."""
    if not isinstance(step, str) or step not in STEP_REVISIONS:
        return None
    module, attribute = STEP_REVISIONS[step]
    try:
        value = getattr(importlib.import_module(module), attribute)
    except (ImportError, AttributeError):
        return None
    return value if isinstance(value, str) else None


class DeterministicRefusal(ValueError):
    """A typed refusal (``str(error)`` is its code) that the same inputs and revision always repeat."""

    def __init__(self, code: str, *, step: str, inputs: Mapping[str, str]) -> None:
        super().__init__(code)
        revision = current_revision(step)
        if revision is None:
            raise ValueError("deterministic_refusal_step_unknown")
        self.code, self.step, self.revision = code, step, revision
        self.inputs = {str(name): str(value) for name, value in inputs.items()}
        self.key = _key(code, step, revision, self.inputs)

    def record(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA, "code": self.code, "step": self.step, "revision": self.revision,
                "inputs": dict(self.inputs), "key": self.key}


def unchanged(record: object) -> bool:
    """Whether the running release would decide ``record``'s step on its inputs exactly as before.

    False for anything that is not a well-formed record, or whose step this release does not know:
    such a record never holds a job back, so a malformed one costs one re-run, never a stuck job.
    """
    if not isinstance(record, Mapping) or record.get("schema_version") != SCHEMA:
        return False
    code, step, inputs, key = (record.get(name) for name in ("code", "step", "inputs", "key"))
    revision = current_revision(step)
    if (revision is None or not isinstance(code, str) or not isinstance(inputs, Mapping)
            or not all(isinstance(value, str) for value in inputs.values())):
        return False
    return _key(code, step, revision, inputs) == key


def find(error: BaseException | None, *, limit: int = 16) -> DeterministicRefusal | None:
    """The deterministic refusal ``error`` is or was raised from (its cause or context chain), else None."""
    seen: set[int] = set()
    current = error
    for _ in range(limit):
        if current is None or id(current) in seen:
            return None
        if isinstance(current, DeterministicRefusal):
            return current
        seen.add(id(current))
        current = current.__cause__ if current.__cause__ is not None else current.__context__
    return None
