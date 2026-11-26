"""Pure retained scene-intent contracts, shared by issuance and evidence readers.

ADP-050/day-28 and ADP-009D: reopen owner consent without importing execution
or mutation services. Issuance, reservations and recovery remain in intake.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as canonical_digest
from .validation_file_digests import note_consulted
from .website_job_window import WEBSITE_JOB_WINDOW_MAX_SECONDS

from .task_family_scoring import FAMILY_ALIASES

_LOG = logging.getLogger(__name__)

#: A navigation job's website intake (``website_navigation_intake``): the robot walks to a goal in a scene
#: kept as filmed. It names its navigation record's digests and no subject, support or destination, and it
#: is preparation only. Owner, 2026-11-07: it reserves $0 and grants no construction allowance, because a
#: scene kept as filmed needs neither, so no attempt is ever reserved for it (``reserve_scene_attempt``).
NAVIGATION_STRATEGY = "goal_region_navigation"
NAVIGATION_GEOMETRY_ORIGIN = "kept_as_filmed"
NAVIGATION_TASK_DIGESTS = ("navigation_task_spec_digest", "compiled_task_spec_digest",
                           "task_success_contract_digest", "runtime_contract_digest", "scene_revision_digest")
TASK_STRATEGIES = ("pick_and_place", "articulated_open_close", NAVIGATION_STRATEGY, *FAMILY_ALIASES)
ARTICULATION_JOINT_TYPES = ("prismatic", "revolute")
#: What a pick-and-place or articulated task may carry besides its own parts: keys a release reads (whether completed
#: stages are reused; an evaluation request's robot binding and its digest, episode interpretation and evaluation
#: source, ``task_evaluation_public_scene_attempt_factory``, ``task_evaluation_controls_terminal_adoption``,
#: ``task_evaluation_team_run_controller``). Any other key is refused, never ignored.
TASK_OPTIONAL_FIELDS = ("reuse_completed_stages", "robot_binding_id", "robot_binding_digest", "episode_interpretation",
                        "evaluation_source")
#: A several-object job's further objects: built and composed back into the scene besides ``task.subject``, the
#: top-ranked object the success rule scores. Left out entirely for a one-object job, so its request and digests are
#: unchanged. The WebApp's rules, mirrored exactly: Blueprint-WebApp ``server/utils/taskEvaluationSceneIntake.ts``
#: (``ADDITIONAL_OBJECTS_MAX``, ``additionalObjects``, ``sponsoredSceneTask``: pick-and-place only, 1 to 7 rows, each
#: exactly ``{rank, subject}``, ranks 2 to 8 strictly rising, a non-empty subject) and
#: ``server/utils/websiteSceneSponsorship.ts`` (``WEBSITE_SUBJECT_ORIGINS`` in ``validateWebsiteSponsoredIntake``:
#: removed before reconstruction or designed, never a ``test_environment``). A rank left out is an object left in the
#: scan as filmed.
ADDITIONAL_OBJECTS_FIELD = "additional_objects"
ADDITIONAL_OBJECTS_MAX = 7
ADDITIONAL_OBJECT_RANKS = (2, ADDITIONAL_OBJECTS_MAX + 1)
ADDITIONAL_OBJECT_ORIGINS = ("removed_before_reconstruction", "synthetic_design")
ADDITIONAL_OBJECT_FORBIDDEN_SUBJECT_FIELDS = ("test_environment",)
REQUEST_SCHEMA = "task_evaluation_scene_intake_request.v1"
INTENT_SCHEMA = "task_evaluation_scene_intent.v1"
ATTEMPT_SCHEMA = "task_evaluation_scene_attempt.v1"
ROOT_ENV = "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT"
CLIENTS_ENV = "BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_CLIENT_IDS"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
#: The two frozen policy candidates this ADP-009D run actually supports end to
#: end (scene setup CANDIDATE_IDS, policy-canary handoff, dispatch). Intake
#: rejects any other pair up front instead of accepting it and failing late,
#: after construction spend, at the handoff (A10). Do not broaden this here.
SUPPORTED_POLICY_CANDIDATE_IDS = ("pi05_droid", "groot_n17_droid")


class SceneIntakeError(ValueError):
    pass


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise SceneIntakeError("scene_intake_" + code)


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = canonical_digest(result, digest_field=field)
    return result


def _validate_additional_objects(task: Mapping[str, Any]) -> None:
    """A task's further objects (``ADDITIONAL_OBJECTS_FIELD``), when it names any: only on a pick-and-place task."""
    if ADDITIONAL_OBJECTS_FIELD not in task:
        return
    _require(task["strategy"] == "pick_and_place", "task_additional_objects_strategy_invalid")
    rows = task[ADDITIONAL_OBJECTS_FIELD]
    _require(isinstance(rows, list) and 1 <= len(rows) <= ADDITIONAL_OBJECTS_MAX
             and all(isinstance(row, Mapping) and set(row) == {"rank", "subject"} for row in rows),
             "task_additional_objects_invalid")
    low, high = ADDITIONAL_OBJECT_RANKS
    ranks = [row["rank"] for row in rows]
    _require(all(type(rank) is int and low <= rank <= high for rank in ranks)
             and all(later > earlier for earlier, later in zip(ranks, ranks[1:])), "task_additional_objects_rank_invalid")
    _require(all(isinstance(row["subject"], Mapping) and bool(row["subject"])
                 and row["subject"].get("geometry_origin") in ADDITIONAL_OBJECT_ORIGINS for row in rows),
             "task_additional_object_subject_invalid")
    _require(not any(key in row["subject"] for row in rows for key in ADDITIONAL_OBJECT_FORBIDDEN_SUBJECT_FIELDS),
             "task_additional_object_test_environment_forbidden")


def task_fields_not_read(task: Mapping[str, Any]) -> list[str]:
    """The keys of a pick-and-place or articulated task this release does not read, sorted: kept as sent and flagged
    (``scene_intake_task_fields_not_read``), never refused. Empty for a family or navigation task, checked exactly."""
    strategy = task.get("strategy")
    parts = {"pick_and_place": ("subject", "support", "destination", "success"),
             "articulated_open_close": ("subject", "support", "articulation", "success")}.get(strategy)
    if parts is None:
        return []
    read = {"task_id", "strategy", *parts, *TASK_OPTIONAL_FIELDS}
    if strategy == "pick_and_place":
        read.add(ADDITIONAL_OBJECTS_FIELD)
    return sorted(str(key) for key in task if key not in read)


def validate_request(value: Mapping[str, Any], *, now: float) -> dict[str, Any]:
    _require(set(value) == {"schema_version", "submission_id", "owner", "source", "task",
                            "execution", "consent"}, "request_fields_invalid")
    _require(value.get("schema_version") == REQUEST_SCHEMA, "schema_invalid")
    _require(_identifier(value.get("submission_id")), "submission_id_invalid")
    owner = value.get("owner")
    _require(isinstance(owner, Mapping) and set(owner) == {"user_id", "organization_id"}
             and all(isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._@-]{0,127}", v)
                     for v in owner.values()), "owner_invalid")
    source = value.get("source")
    _require(isinstance(source, Mapping) and set(source) - {"collision_mesh"} == {"kind", "binding_id", "content_digest"},
             "source_invalid")
    _require(source["kind"] in {"capture_bundle", "mesh", "gaussian_splat", "public_scene"}
             and _identifier(source["binding_id"])
             and isinstance(source["content_digest"], str)
             and _DIGEST.fullmatch(source["content_digest"]) is not None, "source_invalid")
    if "collision_mesh" in source:
        companion = source["collision_mesh"]
        _require(source["kind"] == "gaussian_splat" and isinstance(companion, Mapping)
                 and set(companion) == {"binding_id", "content_digest", "rights_reference", "frame_relation"}
                 and _identifier(companion.get("binding_id")) and companion["binding_id"] != source["binding_id"]
                 and isinstance(companion.get("content_digest"), str)
                 and _DIGEST.fullmatch(companion["content_digest"]) is not None
                 and isinstance(companion.get("rights_reference"), str)
                 and _DIGEST.fullmatch(companion["rights_reference"]) is not None
                 and companion.get("frame_relation") == "owner_declared_common_frame", "collision_mesh_binding_invalid")
    task = value.get("task")
    _require(isinstance(task, Mapping) and task.get("strategy") in TASK_STRATEGIES
             and _identifier(task.get("task_id")), "task_invalid")
    _require(type(task.get("reuse_completed_stages", True)) is bool, "task_reuse_mode_invalid")
    # Before the family and navigation checks, so further objects on either are refused by their own code.
    _validate_additional_objects(task)
    from .task_family_scoring import canonical_task_family
    family = canonical_task_family(task["strategy"])
    if family is not None:
        from .task_family_dispatch_contracts import validate_family_intake_task
        try:
            validate_family_intake_task(task)
        except (ValueError, KeyError, TypeError) as exc:
            raise SceneIntakeError("scene_intake_task_family_invalid:" + str(exc)) from exc
    navigation = task["strategy"] == NAVIGATION_STRATEGY
    from .website_preparation_revision_identity import REISSUE_FIELD, reissue
    if navigation:
        # The scene as filmed: its record's digests, and nothing placed, removed or moved. A navigation
        # intake re-issued for the same capture (its scene half derived from a later Atlas result) also
        # names the capture, its parent submission and that result.
        _require(set(task) - {REISSUE_FIELD} == {"task_id", "strategy", "geometry_origin", *NAVIGATION_TASK_DIGESTS}
                 and task["geometry_origin"] == NAVIGATION_GEOMETRY_ORIGIN
                 and all(isinstance(task[key], str) and _DIGEST.fullmatch(task[key]) is not None
                         for key in NAVIGATION_TASK_DIGESTS), "task_navigation_invalid")
    # A relocation binds a destination; an articulated open/close binds the
    # mechanism instead (the moving part never leaves its assembly).
    required = (() if family is not None or navigation else ("subject", "support", "articulation", "success")
                if task["strategy"] == "articulated_open_close"
                else ("subject", "support", "destination", "success"))
    for key in required:
        _require(isinstance(task.get(key), Mapping) and bool(task[key]), "task_" + key + "_missing")
    if required:
        # A key this release does not read is kept and flagged, never a stop (only privacy, rights and money stop a
        # job; a review of PR #3754). Further objects are not such a key: they are read and checked above.
        unread = task_fields_not_read(task)
        if unread:
            _LOG.warning("scene_intake_task_fields_not_read", extra={"submission_id": value.get("submission_id"),
                                                                     "fields": unread})
    if task["strategy"] == "articulated_open_close":
        _require(task["articulation"].get("joint_type") in ARTICULATION_JOINT_TYPES
                 and isinstance(task["articulation"].get("part_label"), str)
                 and task["articulation"]["part_label"].strip() != "", "task_articulation_invalid")
    from .task_evaluation_scene_execution_scope import scene_preparation_only, validate_execution
    validate_execution(value, now=now)
    _require(not navigation or scene_preparation_only(value), "navigation_must_be_scene_preparation")
    # A re-issue's submission is its prefix and the digest of the rest of the request; any other use of the
    # marker or the prefix is refused.
    try:
        reissue(value)
    except (ValueError, TypeError, AttributeError):
        raise SceneIntakeError("scene_intake_navigation_reissue_invalid") from None
    consent = value.get("consent")
    _require(isinstance(consent, Mapping) and set(consent) == {
        "accepted_by", "accepted_at_epoch", "rights_reference", "provider_terms_reference",
        "private_processing_authorized", "provider_training_authorized", "task_confirmed",
        "spend_authorized"}, "consent_invalid")
    # A website intent carries its preparation grant's consent, so the consent
    # may be as old as the longest window a grant may carry.
    _require(consent["accepted_by"] == owner["user_id"]
             and _number(consent["accepted_at_epoch"])
             and now - WEBSITE_JOB_WINDOW_MAX_SECONDS <= consent["accepted_at_epoch"] <= now,
             "consent_actor_or_time_invalid")
    _require(all(isinstance(consent[k], str) and 1 <= len(consent[k]) <= 1000
                 for k in ("rights_reference", "provider_terms_reference")), "consent_references_missing")
    # The site's own confirmation of its brief, as the website read it; never set or assumed here.
    _require(not navigation or consent["task_confirmed"] is True, "task_context_not_confirmed")
    from .website_preparation_contracts import preparation_consent_valid
    _require(consent["private_processing_authorized"] is True
             and consent["provider_training_authorized"] is False
             and (consent["task_confirmed"] is True or preparation_consent_valid(value))
             and consent["spend_authorized"] is True,
             "consent_missing")
    # Detach mutable caller state and reject non-JSON/NaN task values.
    try:
        detached = json.loads(json.dumps(value, allow_nan=False))
        canonical_digest(detached)
        return detached
    except (TypeError, ValueError) as exc:
        raise SceneIntakeError("scene_intake_json_invalid") from exc


def _read(path: Path, field: str) -> dict[str, Any]:
    _require(not path.is_symlink(), "record_unsafe")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise SceneIntakeError("scene_intake_record_unreadable") from exc
    _require(isinstance(value, dict) and value.get(field) == canonical_digest(value, digest_field=field),
             "record_digest_invalid")
    note_consulted(path, raw)
    return value
