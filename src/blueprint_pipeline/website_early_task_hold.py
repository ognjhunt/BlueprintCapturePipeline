"""Whether a family task's runtime is installed on this host (``task_family_runtime_gap``).

This module once held a website job right after its Atlas review, before its scene was bought, when the job
had no task profile or its family no installed runtime. Owner direction 2026-11-16: any job is accepted, so
nothing is held here any more. Every such job buys its scene and the scene handoff ships the room package
(``website_scene_handoff.write_room_package``); the handoff reads ``task_family_runtime_gap`` to choose it.
A handoff hold an earlier release wrote still reads as held (``holds_native_task_preparation``).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

FAMILY_RUNTIME_GAP = "website_task_family_runtime_preparation_required"
PROFILE_GAP = "website_native_task_profile_preparation_required"
COVERAGE_GAP = "website_task_operation_coverage_unresolved"


def task_family_runtime_gap(task_context: Mapping[str, Any], *, task_semantics: Mapping[str, Any] | None,
                            source_video_digest: str | None,
                            runtimes: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Hold a family task when no runtime on this host registers its families.

    ``runtimes`` is ``task_family_native_runtime.installed_task_family_runtimes()``
    (read when omitted). A catalog that is configured but unreadable holds nothing:
    the job goes on and the later family preparation reports what is missing.
    """
    from .task_family_scoring import canonical_task_family
    from .task_profile_contracts import required_task_families
    from .website_task_semantics import effective_task_profile
    family = canonical_task_family(effective_task_profile(task_context=task_context, task_semantics=task_semantics,
                                                          source_video_digest=source_video_digest))
    if family is None:
        return None
    if runtimes is None:
        from .task_family_native_runtime import installed_task_family_runtimes
        runtimes = installed_task_family_runtimes()
    required = required_task_families(family, task_semantics)
    if runtimes is None or set(required) <= set(runtimes.get("task_families") or []):
        return None
    general = task_context.get("task_family") == "general_task" and task_semantics is not None
    return {
        "code": FAMILY_RUNTIME_GAP, "task_family": task_context.get("task_family"),
        "task_context_digest": task_context.get("context_digest"), "job_disposition": "held_for_preparation",
        "description": task_context.get("description"),
        "reason": (f"The task is a {family.replace('_', ' ')} job. No simulation runtime for it is installed "
                   "yet; the job is held while one is prepared. This is not a finding that robot teams "
                   "cannot perform the job."),
        "missing": ["qualified_task_family_runtime"], "task_family_profile": family,
        "required_task_families": required,
        "runtime_catalog": {key: runtimes.get(key) for key in ("configured", "catalog_digest", "task_families")},
        **({"task_semantics": dict(task_semantics), "source_video_digest": source_video_digest} if general else {}),
        "claim_boundary": "Native evaluation implementation gap only; no robot-team capability verdict.",
    }
