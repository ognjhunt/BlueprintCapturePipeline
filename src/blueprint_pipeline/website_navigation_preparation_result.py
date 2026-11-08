"""A completed navigation intent's result reference: what the website's scene intake checks.

Kept apart from ``website_navigation_intake`` (which re-exports every name here, the same objects) so the
intake service, which only validates this reference, never imports the navigation record's packet and
G1 preparation modules.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .task_evaluation_scene_intent_contracts import NAVIGATION_STRATEGY, NAVIGATION_TASK_DIGESTS

#: The completed intent's result reference and phase (``scene_intent_status``; the website's
#: ``navigationPreparationReference``). Its digest is the record's ``scene_revision_digest``.
RESULT_SCHEMA = "website_navigation_preparation_result.v1"
RESULT_PHASE = "navigation_scene_prepared"


class WebsiteNavigationIntakeError(ValueError):
    """A typed reason a navigation intake cannot be built, kept or read."""


def navigation_preparation_result(request: Mapping[str, Any]) -> dict[str, Any]:
    """The completed navigation intent's result reference: the scene as filmed, nothing paid for or run.

    Its digest is the navigation record's ``scene_revision_digest``, as the request names it; the other
    digests are the request's own. It claims no native application, policy episode or construction.
    """
    task = request["task"]
    return {"schema_version": RESULT_SCHEMA, "digest": task["scene_revision_digest"],
            "task_kind": NAVIGATION_STRATEGY,
            **{key: task[key] for key in NAVIGATION_TASK_DIGESTS if key != "scene_revision_digest"},
            "native_application_claimed": False, "policy_episode_claimed": False,
            "paid_construction_performed": False}


def validate_navigation_preparation_result(value: Any, *, request: Mapping[str, Any], status: str,
                                           phase: str | None) -> dict[str, Any]:
    """The result reference a completed navigation intent reports, exactly; raises otherwise."""
    task = request.get("task") or {}
    if (task.get("strategy") != NAVIGATION_STRATEGY or status != "completed" or phase != RESULT_PHASE
            or (request.get("execution") or {}).get("purpose") != "scene_preparation"
            or (request.get("execution") or {}).get("policy_candidates") != []
            or not isinstance(value, Mapping) or dict(value) != navigation_preparation_result(request)):
        raise WebsiteNavigationIntakeError("website_navigation_preparation_result_reference_invalid")
    return dict(value)


def is_navigation_request(request: Any) -> bool:
    return isinstance(request, Mapping) and (request.get("task") or {}).get("strategy") == NAVIGATION_STRATEGY


__all__ = ["RESULT_PHASE", "RESULT_SCHEMA", "WebsiteNavigationIntakeError", "is_navigation_request",
           "navigation_preparation_result", "validate_navigation_preparation_result"]
