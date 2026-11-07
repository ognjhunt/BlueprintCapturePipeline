"""A navigation job's website scene intake, its receipt and the screening source the receipt binds.

A job whose robot walks to a goal in a scene kept as filmed (``goal_region_navigation``) has no scene to
construct, so it gets its scene intent from the website's scene intake with a navigation task: the
navigation record's digests (``website_navigation_scene_inputs``) and nothing placed, removed or moved.
Owner, 2026-11-07: a navigation intake reserves $0 and grants no construction allowance, because a scene
kept as filmed needs neither. The website's route reserves nothing for it and counts no allowance; the
Pipeline reserves no attempt for its intent (``reserve_scene_attempt``) and completes it without paid
configuration (``navigation_preparation_result``).

- ``navigation_intake_request``: the request, from the record and the job's own records. Owner, consent
  and rights are the job's sponsorship's, exactly as for a rigid intake. ``task_confirmed`` is the
  website's own reading of the site's brief (``confirmed``) as the sponsorship carries it: nothing here
  sets or assumes it. An unconfirmed brief is refused by the website, by name.
- ``enqueue_navigation_intake``: sends it through ``enqueue_website_prepared_scene`` and keeps the
  website's acceptance beside the record (the outbox). A typed refusal comes back by its name.
- ``bind_navigation_intake_receipt``: once the Pipeline has staged that exact request's intent, writes the
  intake receipt beside the record. It binds the record's six-field source
  (``native_g1_beta_authority.SOURCE_FIELDS``): the intent's id and digest, the capture, the task, the
  task context and the record's scene revision.
- ``read_navigation_intake_receipt``: reopens the receipt against its record, or None while there is none.

The outbox and receipt sit in the record's own directory inside the job's capture, so the site's
withdrawal deletes them with the record (``website_capture_withdrawal_deletion``). Nothing here spends,
calls a provider other than the website's own intake, runs a simulator or claims a result.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from .task_evaluation_scene_intent_contracts import (
    NAVIGATION_GEOMETRY_ORIGIN,
    NAVIGATION_STRATEGY,
    NAVIGATION_TASK_DIGESTS,
    REQUEST_SCHEMA,
    ROOT_ENV,
)

OUTBOX_SCHEMA = "blueprint.website_navigation_intake_outbox.v1"
OUTBOX_FILENAME = "website_navigation_intake_outbox.v1.json"
RECEIPT_SCHEMA = "blueprint.website_navigation_intake_receipt.v1"
RECEIPT_FILENAME = "website_navigation_intake_receipt.v1.json"
#: The completed intent's result reference and phase (``scene_intent_status``; the website's
#: ``navigationPreparationReference``). Its digest is the record's ``scene_revision_digest``.
RESULT_SCHEMA = "website_navigation_preparation_result.v1"
RESULT_PHASE = "navigation_scene_prepared"
SOURCE_FIELDS = ("scene_intent_id", "scene_intent_digest", "capture_id", "task_id",
                 "task_context_digest", "scene_revision_digest")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REFUSAL = re.compile(r"website_control_prepared-scene_http_409:([a-z][a-z0-9_]{0,99})\Z")


class WebsiteNavigationIntakeError(ValueError):
    """A typed reason a navigation intake cannot be built, kept or read."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise WebsiteNavigationIntakeError(code)


def navigation_task(record: Mapping[str, Any]) -> dict[str, Any]:
    """The intake's task member: the record's task and its digests (named as the record names them)."""
    return {"task_id": record["task_id"], "strategy": NAVIGATION_STRATEGY,
            "geometry_origin": NAVIGATION_GEOMETRY_ORIGIN, **{key: record[key] for key in NAVIGATION_TASK_DIGESTS}}


def navigation_intake_request(*, record: Mapping[str, Any], task_context: Mapping[str, Any],
                              authority: Mapping[str, Any], base_scene: Mapping[str, Any]) -> dict[str, Any]:
    """The website intake request for one navigation record, from the job's own records.

    ``authority`` is the job's scene sponsorship (``website_scene_sponsorship.v1``), the same record a
    rigid or family intake is built from: its owner and consent (rights reference, provider terms, the
    website's ``task_confirmed`` reading) go in unchanged, and its execution fields as it states them.
    The website checks every one against the current sponsorship and reserves nothing for this kind.
    """
    _require(authority.get("schema_version") == "website_scene_sponsorship.v1"
             and isinstance(authority.get("owner"), Mapping) and isinstance(authority.get("consent"), Mapping),
             "website_scene_execution_authority_required")
    _require(record.get("scene_id") == task_context.get("scene_id")
             and record.get("task_id") == "website-" + str(task_context.get("context_digest"))[7:27]
             and (record.get("source_gap") or {}).get("known", {}).get("task_context_digest")
             == task_context.get("context_digest"), "website_navigation_intake_task_context_mismatch")
    providers = ["vast", "openai"] + (["anthropic"] if authority.get("authoring_provider") == "anthropic" else [])
    return {"schema_version": REQUEST_SCHEMA, "submission_id": task_context["capture_id"],
            "owner": dict(authority["owner"]),
            "source": {"kind": "gaussian_splat", "binding_id": base_scene["splat_binding_id"],
                       "content_digest": base_scene["splat_digest"]},
            "task": navigation_task(record),
            "execution": {"purpose": "scene_preparation", "max_total_spend_usd": authority.get("max_total_spend_usd"),
                          "max_paid_attempts": authority.get("max_paid_attempts"), "max_retries": 0,
                          "expires_at_epoch": authority.get("expires_at_epoch"),
                          "allowed_providers": authority.get("allowed_providers", providers),
                          "policy_candidates": [], "claim_scope": "development_only"},
            "consent": dict(authority["consent"])}


def refusal_code(error: BaseException) -> str | None:
    """The website's typed refusal of a ``prepared-scene`` call, or None for any other failure."""
    match = _REFUSAL.fullmatch(str(error))
    return match.group(1) if match else None


def _capture_withdrawn(capture_root: Path) -> bool:
    from .website_atlas_preview_store import capture_withdrawn

    return capture_withdrawn(Path(capture_root))


def _publish(path: Path, value: Mapping[str, Any], code: str) -> dict[str, Any]:
    """Write once beside the record; the same bytes already there are kept, other bytes refused."""
    from .website_navigation_scene_inputs import _write_once

    try:
        _write_once(path, value)
    except FileExistsError:
        _require(_sealed(path, value["schema_version"]) == dict(value), code)
    return dict(value)


def _sealed(path: Path, schema: str) -> dict[str, Any]:
    digest_field = "outbox_digest" if schema == OUTBOX_SCHEMA else "receipt_digest"
    try:
        _require(not path.is_symlink() and path.is_file(), "website_navigation_intake_record_unreadable")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WebsiteNavigationIntakeError("website_navigation_intake_record_unreadable") from exc
    _require(isinstance(value, dict) and value.get("schema_version") == schema
             and value.get(digest_field) == canonical_digest(value, digest_field=digest_field),
             "website_navigation_intake_record_unsealed")
    return value


def enqueue_navigation_intake(*, record_path: str | Path, record: Mapping[str, Any], capture_root: str | Path,
                              task_context: Mapping[str, Any], authority: Mapping[str, Any],
                              base_scene: Mapping[str, Any]) -> dict[str, Any]:
    """Send the record's intake to the website and keep its acceptance beside the record.

    Returns ``{"status": "forward_pending", ...}`` once the website holds the intake, ``{"status":
    "refused", "reason": <its code>}`` for the website's typed refusal (an unconfirmed brief is
    ``website_task_context_not_confirmed``), and raises for anything else. A withdrawn capture sends
    and writes nothing. A record whose intake the website already holds (its outbox is there) is not
    sent again: the website keeps one intake per capture, and re-checks it before forwarding.
    """
    from .website_task_context import enqueue_website_prepared_scene

    record_path = Path(record_path)
    _require(not _capture_withdrawn(Path(capture_root)), "website_capture_withdrawn")
    outbox_path = record_path.parent / OUTBOX_FILENAME
    if outbox_path.exists() or outbox_path.is_symlink():
        kept = _sealed(outbox_path, OUTBOX_SCHEMA)
        _require(kept["scene_inputs_digest"] == record["scene_inputs_digest"]
                 and kept["scene_revision_digest"] == record["scene_revision_digest"],
                 "website_navigation_intake_outbox_record_mismatch")
        return {"status": "forward_pending", "intent_id": kept["intent_id"], "request_digest": kept["request_digest"],
                "outbox_path": str(outbox_path), "outbox_digest": kept["outbox_digest"]}
    request = navigation_intake_request(record=record, task_context=task_context, authority=authority,
                                        base_scene=base_scene)
    try:
        accepted = enqueue_website_prepared_scene(task_context=task_context, request=request)
    except ValueError as exc:
        code = refusal_code(exc)
        if code is None:
            raise
        return {"status": "refused", "reason": code}
    outbox = {"schema_version": OUTBOX_SCHEMA, "claim_ceiling": "development_only",
              "scene_inputs_digest": record["scene_inputs_digest"],
              "scene_revision_digest": record["scene_revision_digest"],
              "request": request, "request_digest": cross_runtime_canonical_digest(request),
              "website_intake": {key: accepted[key] for key in ("id", "request_digest", "state")},
              "intent_id": accepted["id"], "reserved_usd": 0, "construction_allowance_granted": False,
              "outbox_digest": ""}
    outbox["outbox_digest"] = canonical_digest(outbox, digest_field="outbox_digest")
    # Privacy: checked again before the first file this writes in the site's capture.
    _require(not _capture_withdrawn(Path(capture_root)), "website_capture_withdrawn")
    _publish(outbox_path, outbox, "website_navigation_intake_outbox_conflict")
    return {"status": "forward_pending", "intent_id": accepted["id"], "request_digest": outbox["request_digest"],
            "outbox_path": str(outbox_path), "outbox_digest": outbox["outbox_digest"]}


def _intent_root(intent_root: str | Path | None) -> Path | None:
    configured = intent_root if intent_root is not None else os.getenv(ROOT_ENV, "").strip()
    return Path(configured) if configured else None


def bind_navigation_intake_receipt(*, record_path: str | Path, capture_root: str | Path,
                                   intent_root: str | Path | None = None) -> dict[str, Any] | None:
    """Write this record's intake receipt once its intent is staged; the receipt, or None while there is none.

    Reads only: the outbox beside the record, the record itself (``read_navigation_scene_inputs``,
    every file re-verified) and the staged intent (``intent.json`` under the intake root,
    ``BLUEPRINT_TASK_EVALUATION_SCENE_INTAKE_ROOT``), which must hold exactly the outbox's request. A
    capture withdrawn on this host writes nothing.
    """
    from .task_evaluation_scene_intent_contracts import SceneIntakeError, _read as read_intent
    from .website_navigation_scene_inputs import read_navigation_scene_inputs

    record_path = Path(record_path)
    existing = read_navigation_intake_receipt(record_path)
    if existing is not None:
        return existing
    outbox_path = record_path.parent / OUTBOX_FILENAME
    root = _intent_root(intent_root)
    if not outbox_path.exists() or root is None:
        return None
    outbox = _sealed(outbox_path, OUTBOX_SCHEMA)
    intent_path = root / str(outbox["intent_id"]) / "intent.json"
    if not intent_path.is_file():
        return None
    record = read_navigation_scene_inputs(record_path)
    request = outbox["request"]
    _require(outbox["scene_inputs_digest"] == record["scene_inputs_digest"]
             and outbox["scene_revision_digest"] == record["scene_revision_digest"]
             and request.get("task") == navigation_task(record)
             and outbox["request_digest"] == cross_runtime_canonical_digest(request),
             "website_navigation_intake_outbox_record_mismatch")
    try:
        intent = read_intent(intent_path, "intent_digest")
    except SceneIntakeError as exc:
        raise WebsiteNavigationIntakeError("website_navigation_intake_intent_unreadable") from exc
    _require(intent.get("intent_id") == outbox["intent_id"] and intent.get("request") == request,
             "website_navigation_intake_intent_mismatch")
    capture = Path(capture_root)
    _require(capture.name == request["submission_id"] and record_path.absolute().is_relative_to(capture.absolute()),
             "website_navigation_intake_capture_mismatch")
    source = {"scene_intent_id": intent["intent_id"], "scene_intent_digest": intent["intent_digest"],
              "capture_id": request["submission_id"], "task_id": record["task_id"],
              "task_context_digest": record["source_gap"]["known"]["task_context_digest"],
              "scene_revision_digest": record["scene_revision_digest"]}
    receipt = {"schema_version": RECEIPT_SCHEMA, "claim_ceiling": "development_only",
               "scene_id": record["scene_id"], "scene_inputs_digest": record["scene_inputs_digest"],
               "scene_revision_digest": record["scene_revision_digest"],
               "outbox_digest": outbox["outbox_digest"], "request_digest": outbox["request_digest"],
               "intent": {"intent_id": intent["intent_id"], "intent_digest": intent["intent_digest"]},
               "source": source, "reserved_usd": 0, "construction_allowance_granted": False,
               "receipt_digest": ""}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    # Privacy: no new file in the site's capture once it is withdrawn.
    _require(not _capture_withdrawn(capture), "website_capture_withdrawn")
    _publish(record_path.parent / RECEIPT_FILENAME, receipt, "website_navigation_intake_receipt_conflict")
    return read_navigation_intake_receipt(record_path)


def receipt_exists(record_path: str | Path) -> bool:
    """Whether this record's intake receipt is there; a cheap look that reads nothing."""
    path = Path(record_path).parent / RECEIPT_FILENAME
    return path.is_file() and not path.is_symlink()


def read_navigation_intake_receipt(record_path: str | Path,
                                   record: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """The record's intake receipt, sealed and bound to the record; None while there is none.

    ``record``, when given, is the record its reader has just verified; otherwise the record file's
    own seal and identity are read (not its packet).
    """
    record_path = Path(record_path)
    path = record_path.parent / RECEIPT_FILENAME
    if not path.exists() and not path.is_symlink():
        return None
    receipt = _sealed(path, RECEIPT_SCHEMA)
    if record is None:
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise WebsiteNavigationIntakeError("website_navigation_intake_record_unreadable") from exc
        _require(isinstance(record, dict) and record.get("scene_inputs_digest")
                 == canonical_digest(record, digest_field="scene_inputs_digest"),
                 "website_navigation_intake_record_unreadable")
    source = receipt.get("source")
    _require(isinstance(source, dict) and set(source) == set(SOURCE_FIELDS)
             and all(isinstance(source[key], str) and source[key] for key in SOURCE_FIELDS)
             and all(_DIGEST.fullmatch(source[key]) for key in SOURCE_FIELDS if key.endswith("digest"))
             and receipt.get("scene_id") == record.get("scene_id")
             and receipt.get("scene_inputs_digest") == record.get("scene_inputs_digest")
             and receipt.get("scene_revision_digest") == source["scene_revision_digest"]
             == record.get("scene_revision_digest")
             and source["task_id"] == record.get("task_id")
             and source["scene_intent_id"] == (receipt.get("intent") or {}).get("intent_id")
             and source["scene_intent_digest"] == (receipt.get("intent") or {}).get("intent_digest"),
             "website_navigation_intake_receipt_record_mismatch")
    return json.loads(json.dumps(receipt, allow_nan=False))


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


__all__ = ["OUTBOX_FILENAME", "OUTBOX_SCHEMA", "RECEIPT_FILENAME", "RECEIPT_SCHEMA", "RESULT_PHASE",
           "RESULT_SCHEMA", "SOURCE_FIELDS", "WebsiteNavigationIntakeError", "bind_navigation_intake_receipt",
           "enqueue_navigation_intake", "is_navigation_request", "navigation_intake_request",
           "navigation_preparation_result", "navigation_task", "read_navigation_intake_receipt",
           "receipt_exists", "refusal_code", "validate_navigation_preparation_result"]
