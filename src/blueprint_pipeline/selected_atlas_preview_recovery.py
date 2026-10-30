"""Source-bound observations and requeue of one existing website Atlas preview.

Atlas status/requeue creates no grant or provider request. A separately selected
preparation resume invokes the existing native handler after a reviewed snapshot;
that handler retains its source, lease, completed-stage and spending admission.
"""
from __future__ import annotations

import fcntl
import hashlib
import math
import os
import re
import stat
import time
from pathlib import Path

from .decision_evidence_contracts import canonical_digest
from .website_atlas_preview_store import JOB_FILE, LOCK_FILE, MAXIMUM_JOB_SECONDS, SCHEMA

_SHA = re.compile(r"sha256:[0-9a-f]{64}")
_CODE = re.compile(r"[a-z][a-z0-9_]{0,99}")
_CONFIG_REFUSALS = frozenset({
    "website_atlas_consent_expired", "website_atlas_source_consent_required",
    "website_atlas_authorization_invalid", "website_atlas_source_owner_mismatch",
    "website_atlas_planner_consent_required", "provider_terms_not_configured_or_changed",
    "source_revoked", "task_brief_missing", "website_scene_sponsorship_binding_invalid",
    "website_atlas_runtime_policy_required",
    "website_atlas_reservation_accounting_invalid", "website_atlas_reservation_summary_too_large",
    "website_atlas_accounting_response_too_large",
})


def _require(condition, code):
    if not condition:
        raise ValueError("selected_handoff_atlas_" + code)


def _safe_path(path, root, *, directory=False):
    """Reject aliases and special files throughout the fixed selected namespace."""
    path, root = Path(path).absolute(), Path(root).absolute()
    _require(path.is_relative_to(root), "path_invalid")
    current = root
    for part in (None, *path.relative_to(root).parts):
        if part is not None:
            current /= part
        info = current.lstat()
        expected = stat.S_ISDIR if current != path or directory else stat.S_ISREG
        _require(expected(info.st_mode), "path_unsafe")
    return path


# Bounds the read, not the job: a native ledger grows with every attempt and
# lease (capture-447ae4f4's was 138 KB), and refusing it stopped the retry.
RECORD_LIMIT_BYTES = 16 * 1024 * 1024


def _read_json(path, root):
    from .selected_handoff_recovery import _json
    _safe_path(path, root)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        info = os.fstat(stream.fileno())
        _require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= RECORD_LIMIT_BYTES, "record_invalid")
        raw = stream.read(RECORD_LIMIT_BYTES + 1)
    _require(len(raw) == info.st_size, "record_changed")
    return _json(raw), "sha256:" + hashlib.sha256(raw).hexdigest()


def _number(value, *, positive=False, integer=False):
    return (type(value) in ({int} if integer else {int, float})
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def _equal_cost(left, right):
    return abs(left - right) < 0.000001


def _source(selection, observation, membership, job, capture_root, storage_root):
    _require(type(membership) is tuple and len(membership) == 3, "membership_required")
    record = membership[1]
    original = observation["producer_delivery"]["raw_video"]
    rows = [row for row in record["raw"] if row["object_name"] == original["object_name"]]
    _require(len(rows) == 1, "source_binding_invalid")
    row = rows[0]
    _require(job.get("source_video") == row["relative_path"], "source_binding_invalid")
    path = _safe_path(capture_root / row["relative_path"], storage_root)
    digest = hashlib.sha256()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and before.st_size == row["size_bytes"], "source_changed")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
        _require(all(getattr(before, key) == getattr(after, key)
                     for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")), "source_changed")
    _require("sha256:" + digest.hexdigest() == row["sha256"], "source_changed")
    return {"video_sha256": row["sha256"], "video_generation": row["generation"],
            "video_size_bytes": row["size_bytes"], "marker_generation": selection["marker_generation"]}


def _configuration(context, source_digest):
    from .website_atlas_pipeline import load_configuration
    return load_configuration(context, source_digest, create=False)


def _accounting(value, *, native=False):
    """Copy a finite, numeric ledger projection; never relay arbitrary server data."""
    schema = "website_native_preparation_accounting.v1" if native else "website_atlas_allowance_accounting.v1"
    scope = "request_native_preparation_ledger" if native else "request_atlas_allowance"
    totals = {"settled_cost_usd", "held_cost_usd", "reserved_or_settled_cost_usd", "request_count"}
    keys = totals | {"schema_version", "reservations", "source_scope", "external_capture_assessment_costs"}
    _require(type(value) is dict and set(value) == keys and value["schema_version"] == schema
             and value["source_scope"] == scope
             and value["external_capture_assessment_costs"] == "not_in_preparation_ledger"
             and all(_number(value[key], integer=key == "request_count") for key in totals)
             and type(value["reservations"]) is list and len(value["reservations"]) <= 128, "accounting_invalid")
    row_keys = {"allocation_binding_digest", "provider", "maximum_cost_usd", "settled_cost_usd",
                "accounted_cost_usd", "request_count", "status", "task_context_digest", "source_video_digest",
                "atlas_authorization_digest", "settlement_digest"}
    for row in value["reservations"]:
        _require(type(row) is dict and set(row) == row_keys
                 and row["provider"] in {"world_labs_atlas", "world_labs", "openai", "meta", "google", "vast"}
                 and row["status"] in {"held", "settled"}
                 and all(_number(row[key], positive=True, integer=key == "request_count")
                         for key in ("maximum_cost_usd", "request_count"))
                 and _number(row["accounted_cost_usd"])
                 and (row["settled_cost_usd"] is None or _number(row["settled_cost_usd"]))
                 and all(type(row[key]) is str and _SHA.fullmatch(row[key])
                         for key in ("allocation_binding_digest", "task_context_digest"))
                 and all(row[key] is None or type(row[key]) is str and _SHA.fullmatch(row[key])
                         for key in ("source_video_digest", "atlas_authorization_digest", "settlement_digest")),
                 "accounting_invalid")
        settled = row["status"] == "settled"
        _require(settled == (row["settled_cost_usd"] is not None)
                 and settled == (row["settlement_digest"] is not None)
                 and _equal_cost(row["accounted_cost_usd"],
                                 row["settled_cost_usd"] if settled else row["maximum_cost_usd"]),
                 "accounting_invalid")
    rows = value["reservations"]
    _require(len({row["allocation_binding_digest"] for row in rows}) == len(rows)
             and value["request_count"] == sum(row["request_count"] for row in rows)
             and _equal_cost(value["reserved_or_settled_cost_usd"], sum(row["accounted_cost_usd"] for row in rows))
             and _equal_cost(value["held_cost_usd"], sum(row["accounted_cost_usd"] for row in rows if row["status"] == "held"))
             and _equal_cost(value["settled_cost_usd"], sum(row["accounted_cost_usd"] for row in rows if row["status"] == "settled")),
             "accounting_invalid")
    # Row order is not accounting truth and must not invalidate a reviewed snapshot.
    return {**value, "reservations": sorted(value["reservations"], key=lambda row: row["allocation_binding_digest"])}


def _capture_inference_accounting(value, *, request_id, capture_id, source_video_digest):
    """Retained usage and unknowns outside preparation; never a run allowance.

    The optional signed getter owns exact capture/linked-run evidence acquisition.
    A linked assessment total may overlap the calls and is never added to them.
    """
    if value is None:
        return None  # An older getter provides no inference cost evidence.

    def sha(item):
        return type(item) is str and _SHA.fullmatch(item) is not None

    def nullable_sha(item):
        return item is None or sha(item)

    keys = {"schema_version", "request_id", "capture_id", "budget_digest", "source_video_digest",
            "ledger_status", "known_usage_cost_usd", "bounded_unknown_cost_usd", "unbounded_unknown_call_count",
            "calls", "linked_assessment", "privacy", "preparation_ledger_overlap"}
    _require(type(value) is dict and set(value) == keys
             and value["schema_version"] == "website_capture_inference_accounting.v1"
             and value["request_id"] == request_id and value["capture_id"] == capture_id
             and value["source_video_digest"] == source_video_digest and sha(value["budget_digest"])
             and type(value["ledger_status"]) is str
             and value["ledger_status"] in {"complete", "missing", "incomplete", "invalid", "unavailable"}
             and _number(value["known_usage_cost_usd"]) and _number(value["bounded_unknown_cost_usd"])
             and _number(value["unbounded_unknown_call_count"], integer=True)
             and value["preparation_ledger_overlap"] == "not_established"
             and type(value["calls"]) is list and len(value["calls"]) <= 64, "inference_accounting_invalid")
    row_keys = {"call_digest", "provider", "model", "state", "source_digest", "context_digest", "run_digest",
                "recorded_cost_usd", "bounded_unknown_cost_usd", "usage_digest"}
    for row in value["calls"]:
        _require(type(row) is dict and set(row) == row_keys
                 and all(type(row[key]) is str for key in ("provider", "model", "state"))
                 and (row["provider"], row["model"]) in {("openai", "gpt-6.1-sol"), ("gemini", "gemini-3.8-flash")}
                 and row["state"] in {"recorded", "admitted", "unknown"}
                 and all(sha(row[key]) for key in ("call_digest", "source_digest", "context_digest"))
                 and all(nullable_sha(row[key]) for key in ("run_digest", "usage_digest"))
                 and all(row[key] is None or _number(row[key])
                         for key in ("recorded_cost_usd", "bounded_unknown_cost_usd")), "inference_accounting_invalid")
        known = row["recorded_cost_usd"] is not None
        _require((not known or row["state"] == "recorded" and row["bounded_unknown_cost_usd"] is None)
                 and known == (row["usage_digest"] is not None), "inference_accounting_invalid")
    calls = value["calls"]
    unknown = sum(row["recorded_cost_usd"] is None and row["bounded_unknown_cost_usd"] is None for row in calls)
    _require(len({row["call_digest"] for row in calls}) == len(calls)
             and _equal_cost(value["known_usage_cost_usd"], sum(row["recorded_cost_usd"] or 0 for row in calls))
             and _equal_cost(value["bounded_unknown_cost_usd"], sum(row["bounded_unknown_cost_usd"] or 0 for row in calls))
             and value["unbounded_unknown_call_count"] >= unknown
             and (value["ledger_status"] != "complete" or value["unbounded_unknown_call_count"] == unknown)
             and (value["ledger_status"] not in {"missing", "invalid", "unavailable"}
                  or value["unbounded_unknown_call_count"] > 0), "inference_accounting_invalid")
    linked = value["linked_assessment"]
    _require(type(linked) is dict and set(linked) == {
        "status", "job_digest", "run_digest", "known_usage_cost_usd", "ledger_overlap"}
        and type(linked["status"]) is str and type(linked["ledger_overlap"]) is str
        and linked["status"] in {"verified", "not_linked", "missing", "mismatch", "unavailable"}
        and all(nullable_sha(linked[key]) for key in ("job_digest", "run_digest"))
        and (linked["known_usage_cost_usd"] is None or _number(linked["known_usage_cost_usd"]))
        and linked["ledger_overlap"] in {"matched_call_ids", "unknown", "not_available"}
        and (linked["status"] == "verified" or linked["known_usage_cost_usd"] is None)
        and (linked["status"] != "verified" or sha(linked["job_digest"]) and sha(linked["run_digest"]))
        and (linked["ledger_overlap"] != "matched_call_ids" or linked["status"] == "verified"
             and any(row["run_digest"] == linked["run_digest"] for row in calls)), "inference_accounting_invalid")
    privacy = value["privacy"]
    _require(type(privacy) is dict and set(privacy) == {"status", "historical_cost_status"}
             and type(privacy["status"]) is str
             and privacy["status"] in {"source_bound_not_reviewed", "source_bound_reviewed", "unknown"}
             and privacy["historical_cost_status"] == "unknown", "inference_accounting_invalid")
    return {**value, "calls": sorted(calls, key=lambda row: row["call_digest"])}


def _snapshot(selection, observation, membership, capture_root, storage_root, job):
    from .capture_original_owner_observer import validate_observation
    from .website_atlas_pipeline import _require_current_price_review
    from .website_task_context import load_current_website_task_context
    from .paid_resource_admission import require_paid_resource_budget

    _require(job.get("schema_version") == SCHEMA
             and all(job.get(key) == expected for key, expected in (
                 ("request_id", observation["request_id"]), ("scene_id", selection["scene_id"]),
                 ("capture_id", selection["capture_id"])))
             and job.get("state") in {"processing", "ready", "needs_input", "failed", "withdrawn"}
             and _number(job.get("revision"), positive=True, integer=True)
             and _number(job.get("enqueued_at_epoch"), positive=True, integer=True), "job_binding_invalid")
    disk_job, job_digest = _read_json(capture_root / "pipeline/atlas" / JOB_FILE, storage_root)
    _require(disk_job == job, "job_changed")
    source = _source(selection, observation, membership, job, capture_root, storage_root)
    context = load_current_website_task_context(request_id=observation["request_id"],
        scene_id=selection["scene_id"], capture_id=selection["capture_id"], purpose="scene_preparation")
    value = {"request_id": observation["request_id"], "scene_id": selection["scene_id"],
             "capture_id": selection["capture_id"], "source": source,
             "delivery_selector_digest": canonical_digest({key: item for key, item in selection.items()
                 if key not in {"mode", "atlas_status_digest"}}),
             "source_membership_sha256": "sha256:" + hashlib.sha256(membership[0]).hexdigest(),
             "task_context_digest": context["context_digest"], "job_sha256": job_digest,
             "job": {"state": job["state"], "revision": job["revision"],
                     "enqueued_at_epoch": job["enqueued_at_epoch"],
                     "expires_at_epoch": job["enqueued_at_epoch"] + MAXIMUM_JOB_SECONDS,
                     "retry_count": len(job.get("retries", []))}, "preflight_blockers": [],
             "capture_inference_accounting": None}
    # Old native failures may precede the Atlas-wait continuation seam. Keep
    # their entire ledger in the reviewed digest, including unknown outcomes,
    # attempts and lease history; never manufacture a successful/waiting row.
    from .pubsub_handoff_listener import JOB_LEDGER_FILENAME
    native_path = capture_root / JOB_LEDGER_FILENAME
    value["native_preparation"] = None
    if native_path.exists() or native_path.is_symlink():
        native, _ = _read_json(native_path, storage_root)
        _require(native.get("scene_id") == selection["scene_id"]
                 and native.get("capture_id") == selection["capture_id"]
                 and _number(native.get("revision"), positive=True, integer=True)
                 and _number(native.get("attempt_count"), integer=True), "native_ledger_invalid")
        value["native_preparation"] = {
            "ledger_digest": canonical_digest(native),
            **{key: native.get(key) for key in (
                "status", "revision", "attempt_count", "source_payload_sha256", "producer_delivery_key")},
        }
    if type(job.get("reason")) is str and _CODE.fullmatch(job["reason"]):
        value["job"]["reason"] = job["reason"]
    if type(job.get("stage")) is str and _CODE.fullmatch(job["stage"]):
        value["job"]["stage"] = job["stage"]
    if time.time() - job["enqueued_at_epoch"] > MAXIMUM_JOB_SECONDS:
        value["preflight_blockers"].append("atlas_preview_expired")
    try:
        config = _configuration(context, source["video_sha256"])
    except ValueError as exc:
        code = str(exc).partition(":")[2]
        value["preflight_blockers"].append(code if code in _CONFIG_REFUSALS
                                           else "atlas_current_configuration_unavailable")
    else:
        authority = config.get("authority", {})
        _require(all(authority.get(key) == expected for key, expected in (
            ("request_id", observation["request_id"]), ("capture_id", selection["capture_id"]),
            ("source_video_digest", source["video_sha256"])))
            and _number(authority.get("maximum_cost_usd"), positive=True)
            # Money is the hard stop; a request count is optional and only reported.
            and ("maximum_requests" not in authority
                 or _number(authority.get("maximum_requests"), positive=True, integer=True))
            and _number(authority.get("expires_at_epoch"), positive=True)
            and all(_number(config.get(key)) for key in (
                "reserved_or_settled_cost_usd", "remaining_cost_usd"))
            and ("remaining_requests" not in config or _number(config.get("remaining_requests"))),
            "configuration_invalid")
        value["allowance"] = {key: authority[key] for key in (
            "maximum_cost_usd", "maximum_requests", "expires_at_epoch") if key in authority}
        value["allowance"].update({key: config[key] for key in (
            "reserved_or_settled_cost_usd", "remaining_cost_usd", "remaining_requests") if key in config})
        value["allowance"]["authorization_digest"] = canonical_digest(authority)
        value["policy_digest"] = canonical_digest(config["policy"])
        # Reservation summaries come only from the bounded read-only WebApp getter.
        value["allowance_accounting"] = _accounting(config.get("allowance_accounting"))
        value["native_preparation_accounting"] = _accounting(config.get("native_preparation_accounting"), native=True)
        value["capture_inference_accounting"] = _capture_inference_accounting(
            config.get("capture_inference_accounting"), request_id=observation["request_id"],
            capture_id=selection["capture_id"], source_video_digest=source["video_sha256"])
        _require(len(value["allowance_accounting"]["reservations"])
                 + len(value["native_preparation_accounting"]["reservations"]) <= 128, "accounting_invalid")
        atlas_rows = value["allowance_accounting"]["reservations"]
        native_rows = value["native_preparation_accounting"]["reservations"]
        _require(not ({row["allocation_binding_digest"] for row in atlas_rows}
                      & {row["allocation_binding_digest"] for row in native_rows}), "accounting_invalid")
        _require(_equal_cost(config["reserved_or_settled_cost_usd"],
                             value["allowance_accounting"]["reserved_or_settled_cost_usd"])
                 and _equal_cost(config["remaining_cost_usd"], max(
                     0, authority["maximum_cost_usd"] - config["reserved_or_settled_cost_usd"]))
                 and ("remaining_requests" not in config or config["remaining_requests"] == max(
                     0, authority["maximum_requests"] - value["allowance_accounting"]["request_count"])),
                 "configuration_invalid")
        if authority["expires_at_epoch"] <= time.time():
            value["preflight_blockers"].append("website_atlas_consent_expired")
        if config["remaining_cost_usd"] <= 0 or config.get("remaining_requests", 1) <= 0:
            value["preflight_blockers"].append("website_scene_preparation_budget_exhausted")
        try:
            _require_current_price_review(config["policy"])
        except ValueError:
            value["preflight_blockers"].append("atlas_runtime_price_review_expired")
    try:
        require_paid_resource_budget(resource_class="provider_reconstruction_api")
    except Exception:
        value["preflight_blockers"].append("atlas_current_budget_admission_blocked")
    # Owner evidence is short-lived; a slow local hash cannot silently outlive it.
    validate_observation(observation, bucket=selection["bucket"], scene_id=selection["scene_id"],
        capture_id=selection["capture_id"], marker_generation=selection["marker_generation"],
        expected_purpose="scene_preparation")
    value["source_projection_digest"] = observation["source_projection_digest"]
    value["preflight_blockers"] = sorted(set(value["preflight_blockers"]))
    value["atlas_status_digest"] = canonical_digest(value)
    return value


def recover_selected_atlas_preview(selection, *, observation, membership, capture_root, storage_root,
                                   handoff_payload=None, listener=None):
    from .task_evaluation_scene_retirement_generations import _capture_admission
    from .website_atlas_preview import retry_failed_job
    from .website_atlas_preview_store import capture_withdrawn

    _require(selection["mode"] in {"atlas-status", "atlas-retry", "preparation-resume"}, "mode_invalid")

    expected = (Path(storage_root) / selection["bucket"] / "scenes" / selection["scene_id"]
                / "captures" / selection["capture_id"])
    _require(Path(capture_root).absolute() == expected.absolute(), "path_invalid")
    _safe_path(capture_root, storage_root, directory=True)
    job_path, lock_path = capture_root / "pipeline/atlas" / JOB_FILE, capture_root / "pipeline/atlas" / LOCK_FILE
    _safe_path(job_path, storage_root)
    _safe_path(lock_path, storage_root)
    with _capture_admission(capture_root, required=True):
        _require(not capture_withdrawn(capture_root), "capture_withdrawn")
        if selection["mode"] in {"atlas-status", "preparation-resume"}:
            with os.fdopen(os.open(lock_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError("selected_handoff_atlas_job_busy") from None
                job, _ = _read_json(job_path, storage_root)
                snapshot = _snapshot(selection, observation, membership, capture_root, storage_root, job)
            if selection["mode"] == "preparation-resume":
                # Atlas has finished and will not be repurchased. Its remaining
                # processing deadline, call quota and price-review date cannot
                # block reuse; native allocation checks its current authority.
                blockers = set(snapshot["preflight_blockers"]) - {
                    "atlas_preview_expired", "website_scene_preparation_budget_exhausted",
                    "atlas_runtime_price_review_expired"}
                _require(not blockers, "preflight_blocked")
                _require(snapshot["atlas_status_digest"] == selection["atlas_status_digest"], "status_changed")
                _require(job["state"] == "ready", "reconstruction_not_ready")
                native = snapshot["native_preparation"]
                _require(type(native) is dict and native["status"] == "failed_retryable"
                         and type(native["producer_delivery_key"]) is str
                         and _SHA.fullmatch(native["producer_delivery_key"])
                         and native["producer_delivery_key"] == observation["producer_delivery"].get("delivery_key"),
                         "native_resume_not_eligible")
                _require(type(handoff_payload) is bytes and listener is not None, "original_payload_required")
                provider = os.getenv("BLUEPRINT_PUBSUB_HANDOFF_PROVIDER", "")
                _require(provider in {"local", "claude", "openai"}, "current_provider_missing")
                # This is one explicit operator retry, not a broker reset or a
                # new wait/grant. The native claim checks the reviewed ledger
                # atomically, then increments its existing attempt history.
                from .website_reconstruction_resume import REVIEW_SCHEMA
                result = listener.process_handoff_payload(handoff_payload, storage_root=Path(storage_root),
                    provider=provider, expected_preparation_purpose="scene_preparation",
                    expected_reconstruction_resume={"review_schema": REVIEW_SCHEMA,
                        "ledger_digest": native["ledger_digest"],
                        "selected_payload_sha256": hashlib.sha256(handoff_payload).hexdigest()})
                complete = result.get("status") == "processed" and result.get("queue_disposition") == "terminal_success"
                outcome = {"status": "native_preparation_resumed" if complete else "blocked", "atlas": snapshot,
                        "atlas_retry_performed": False, "native_handler_invoked": True,
                        "native_status": result.get("status"), "native_disposition": result.get("queue_disposition"),
                        "blockers": [] if complete else ["selected_handoff_native_preparation_still_blocked"]}
                # Preserve the native stage's bounded refusal just as original
                # dispatch does; never expose raw provider/source exception text.
                from .capture_delivery_staging import STAGING_REASON_CODES, bounded_staging_error
                from .capture_original_owner_observer import OWNER_OBSERVATION_REASON_CODES
                for field, codes in (("staging_reason", STAGING_REASON_CODES),
                                     ("owner_observation_reason", OWNER_OBSERVATION_REASON_CODES)):
                    reason = result.get(field)
                    if type(reason) is str and reason in codes:
                        outcome[field] = reason
                # A generic or summary refusal keeps only its failure's type and
                # a redacted, trimmed message, bounded again here.
                for field in ("owner_observation_error", "staging_error"):
                    described = bounded_staging_error(result.get(field))
                    if described is not None:
                        outcome[field] = described
                return outcome
            return {"status": "atlas_status_returned", "atlas": snapshot, "atlas_retry_performed": False}

        def preflight(job):
            nonlocal snapshot
            # The controller already checks recoverability under its lock.
            # Its stopped needs_input jobs use the same source/accounting fence.
            snapshot = _snapshot(selection, observation, membership, capture_root, storage_root, job)
            _require(not snapshot["preflight_blockers"], "preflight_blocked")
            _require(snapshot["atlas_status_digest"] == selection["atlas_status_digest"], "status_changed")

        snapshot = None
        retried = retry_failed_job(job_path, preflight=preflight)
        return {"status": "atlas_retry_queued", "atlas": snapshot, "atlas_retry_performed": True,
                "atlas_job_revision": retried["revision"], "later_timer_may_make_paid_calls": True}
