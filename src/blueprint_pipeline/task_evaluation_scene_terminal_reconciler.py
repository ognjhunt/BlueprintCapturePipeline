"""Join retained downstream terminal receipts into the persistent owner status.

``task_evaluation_scene_progression._advance_intent`` drives a persistent owner
intent to ``awaiting_execution`` and then stops: it recognises an
already-``completed`` projection but never *emits* completion from the downstream
controls/policy/publication receipts that a later paid run retains. This module
is that missing terminal owner-result join.

It is deliberately read-only over evidence: it never launches, retries, tears
down, allocates a provider, resets a spend/retry cap, or reruns completed GPU
work. It only observes receipts that other producers already retained and, when
they cohere into an owner-bound terminal result with confirmed resource closure
and an authenticated Website readback, returns the truthful terminal owner
status. Completion is never inferred from a process exit code or an activation
record; it is proven from the retained policy result projection, the launch
reconciler's post-teardown provider-zero closure, and the authenticated Website
readback, all bound to the owner intent, attempt, exact inputs and release
through the ``scene_policy`` binding and the reserved owner attempt.

The join key (all must agree, or the result is treated as not this owner's):

* ``scene_policy_binding.scene_intent_digest == intent.intent_digest`` and its
  frozen policy-candidate pair equals the owner's;
* the reserved owner attempt named by the binding exists with
  ``source_commit == release.source_commit`` (release), ``input_digest`` and
  ``runtime_digest`` equal to the binding (exact inputs);
* the launch request/profile bridge ties that binding to the terminal policy
  ``run_id`` (``profile_digest`` links request→profile→binding, ``run_id`` links
  request→projection);
* the authenticated Website readback and the durable result publication carry the
  projection's own ``projection_digest``.

A completed-unqualified diagnostic, a blocked run and unfinished resource closure
are three distinct, explicit outcomes; none is ever silently upgraded past the
``development_only`` / ``diagnostic_policy_execution`` claim ceiling.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import task_evaluation_scene_intake as intake
from . import task_evaluation_scene_policy_binding as scene_policy
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_policy_canary_result import (
    TaskEvaluationPolicyCanaryResultError,
    validate_policy_canary_result,
)

JOIN_SCHEMA = "task_evaluation_scene_terminal_owner_result.v1"
TERMINAL_JOIN_FILENAME = "terminal_owner_result_join.json"
PROVIDER_ZERO_SCHEMA = "task_evaluation_post_teardown_provider_zero.v1"
DIAGNOSTIC_CLAIM_CEILING = "diagnostic_policy_execution"
POLICY_CANARY_RUN_KIND = "internal_policy_canary"
_RESULT_URI_PREFIXES = ("https://", "b2://", "gs://", "r2://")

CLOSURE_PENDING = "terminal_resource_closure_pending"
READBACK_PENDING = "terminal_website_readback_pending"
PUBLICATION_PENDING = "terminal_result_publication_pending"


class TerminalReconciliationError(ValueError):
    """A retained receipt is structurally invalid for the owner-result join."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise TerminalReconciliationError("scene_terminal_" + code)


def _safe_path(path: Path) -> Path:
    path = Path(path)
    _require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)), "path_unsafe")
    return path


def _record(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {"path": str(path), "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size}


def _read_json(path: Path) -> dict[str, Any] | None:
    if path.is_symlink() or not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and intake._DIGEST.fullmatch(value) is not None


def _owner_binding(directory: Path, *, intent: dict, config: dict, release: dict):
    """Return (binding, attempt, profile, launch_request, projection) or None.

    ``None`` means the retained receipts are not (yet) an owner-bound terminal
    result for this exact intent, attempt, inputs and release; the caller then
    keeps its explicit non-terminal status rather than adopting a stale receipt.
    """
    projection_raw = _read_json(directory / "policy_canary_result_projection.json")
    profile = _read_json(directory / "launch_profile.json")
    launch_request = _read_json(directory / "launch_request.json")
    if projection_raw is None or profile is None or launch_request is None:
        return None
    try:
        projection = validate_policy_canary_result(projection_raw)
    except TaskEvaluationPolicyCanaryResultError:
        return None
    if (projection.get("run_kind") != POLICY_CANARY_RUN_KIND
            or projection.get("claim_ceiling") != DIAGNOSTIC_CLAIM_CEILING):
        return None
    # Bridge coherence: request -> profile (profile_digest) and request -> result (run_id).
    profile_digest = profile.get("profile_digest")
    if (not _is_digest(profile_digest)
            or profile_digest != canonical_digest(profile, digest_field="profile_digest")
            or launch_request.get("launch_profile_digest") != profile_digest
            or launch_request.get("source_commit") != profile.get("source_commit")
            or launch_request.get("run_id") != projection.get("run_id")):
        return None
    plan = profile.get("internal_policy_canary_execution_plan")
    binding_value = plan.get("scene_policy_binding") if isinstance(plan, dict) else None
    if not isinstance(binding_value, dict):
        return None
    try:
        binding = scene_policy.validate_binding(binding_value)
    except scene_policy.ScenePolicyBindingError:
        return None
    # Owner binding: this run's execution plan is bound to this owner intent and
    # to the owner's exact frozen policy-candidate pair.
    if binding["scene_intent_digest"] != intent["intent_digest"]:
        return None
    try:
        if (scene_policy.candidate_map(binding["policy_candidates"])
                != scene_policy.candidate_map(intent["request"]["execution"]["policy_candidates"])):
            return None
    except scene_policy.ScenePolicyBindingError:
        return None
    # Reserved owner attempt: binds the exact inputs and the release commit. A
    # deploy that moved the release, or a receipt for other inputs, will not match
    # and is therefore never adopted here.
    intent_root = config.get("intent_root")
    if not intent_root:
        return None
    attempts = _safe_path(Path(intent_root) / intent["intent_id"] / "attempts")
    try:
        attempt = intake._read(attempts / (binding["attempt_id"] + ".json"), "attempt_digest")
    except intake.SceneIntakeError:
        return None
    if (attempt.get("intent_digest") != intent["intent_digest"]
            or attempt.get("attempt_id") != binding["attempt_id"]
            or attempt.get("source_commit") != release.get("source_commit")
            or attempt.get("runtime_digest") != binding["runtime_digest"]
            or attempt.get("input_digest") != binding["input_digest"]):
        return None
    return binding, attempt, profile, launch_request, projection


def _authenticated_readback(directory: Path, projection: dict) -> dict | None:
    value = _read_json(directory / "policy_canary_webapp_sync.json")
    if value is None:
        return None
    notification = value.get("notification_delivery")
    notification = notification if isinstance(notification, dict) else {}
    if (value.get("schema_version") != "task_evaluation_policy_canary_webapp_sync_result.v1"
            or value.get("status") != "succeeded"
            or value.get("run_id") != projection["run_id"]
            or value.get("request_digest") != projection["request_digest"]
            or value.get("configuration_digest") != projection["configuration_digest"]
            or value.get("result_status") != projection["result_status"]
            or value.get("policy_canary_projection_digest") != projection["projection_digest"]
            or notification.get("run_result_digest") != projection["projection_digest"]
            or notification.get("status") not in {"accepted", "delivered"}):
        return None
    return value


def _confirmed_closure(directory: Path, projection: dict, *, launch_request: dict, profile: dict) -> dict | None:
    value = _read_json(directory / "provider_zero_closure.json")
    if value is None:
        return None
    if (value.get("schema_version") != PROVIDER_ZERO_SCHEMA
            or value.get("status") != "provider_zero_confirmed"
            or value.get("provider_zero_verified") is not True
            or value.get("continuing_spend_from_this_run") is not False
            or value.get("allocator_invoked") is not False
            or value.get("provider_mutation_performed") is not False
            or value.get("automatic_retry_performed") is not False
            or value.get("blockers") != []
            or value.get("provider_zero_receipt_digest")
            != canonical_digest(value, digest_field="provider_zero_receipt_digest")
            or value.get("launch_profile_digest") != profile.get("profile_digest")
            or value.get("run_id") != projection["run_id"]
            or value.get("request_digest") != launch_request.get("request_digest")):
        return None
    return value


def _result_reference(directory: Path, projection: dict) -> dict | None:
    value = _read_json(directory / "terminal_result_publication.json")
    if value is None:
        return None
    uri = value.get("uri")
    if (value.get("digest") != projection["projection_digest"]
            or not isinstance(uri, str) or not uri.startswith(_RESULT_URI_PREFIXES) or "?" in uri
            or type(value.get("size_bytes")) is not int or value.get("size_bytes") <= 0
            or value.get("provider_allocated") is True):
        return None
    return {"uri": uri, "digest": value["digest"], "size_bytes": value["size_bytes"]}


def _failed_children(projection: dict) -> list[dict[str, Any]]:
    children: list[dict[str, Any]] = []
    for episode in projection.get("episodes", []):
        if episode.get("terminal_state") == "completed":
            continue
        evidence = episode.get("evidence") if isinstance(episode.get("evidence"), dict) else {}
        children.append({
            "episode_id": episode.get("episode_id"),
            "failure_taxonomy": episode.get("failure_taxonomy"),
            "typed_media_gap": evidence.get("typed_media_gap"),
            "evidence_gaps": list(evidence.get("evidence_gaps") or []),
        })
    return children


def _explicit(status_blockers: list[str], *, state: dict) -> dict[str, Any]:
    return {"terminal": False, "status": "running", "phase": "terminal_reconciliation",
            "blockers": sorted(set(status_blockers)), "result_reference": None, "state": state}


def reconcile_terminal_owner_result(*, intent: dict, config: dict, release: dict, now: float,
                                    output: str | Path | None = None) -> dict[str, Any] | None:
    """Return a truthful terminal owner-status descriptor, or ``None``.

    ``None`` means there is no owner-bound terminal result to act on yet (no
    terminal-result root configured, no retained projection, or the retained
    receipts are not bound to this intent/attempt/inputs/release). The caller
    keeps its existing explicit status. A returned descriptor is deterministic in
    the retained receipts (independent of ``now``) so duplicate ticks and a
    worker restart are byte-identical.
    """
    root = config.get("terminal_result_root")
    if not root:
        return None
    directory = _safe_path(Path(root) / intent["intent_id"])
    if not directory.is_dir():
        return None
    bound = _owner_binding(directory, intent=intent, config=config, release=release)
    if bound is None:
        return None
    binding, attempt, profile, launch_request, projection = bound

    owner_reference = {
        "intent_digest": intent["intent_digest"], "attempt_id": binding["attempt_id"],
        "input_digest": binding["input_digest"], "source_commit": attempt["source_commit"],
        "run_id": projection["run_id"], "request_digest": projection["request_digest"],
        "configuration_digest": projection["configuration_digest"],
        "projection_digest": projection["projection_digest"],
    }
    failed_children = _failed_children(projection)
    state: dict[str, Any] = {
        "terminal_owner_binding": binding,
        "terminal_result": _record(directory / "policy_canary_result_projection.json"),
        "terminal_failed_children": failed_children,
    }

    # Resource closure and the authenticated Website readback gate any terminal
    # status. Absent closure (an ambiguous create or an incomplete teardown) and
    # an absent/mismatched readback each stay explicit and never complete.
    closure = _confirmed_closure(directory, projection, launch_request=launch_request, profile=profile)
    readback = _authenticated_readback(directory, projection)
    pending: list[str] = []
    if closure is None:
        pending.append(CLOSURE_PENDING)
    if readback is None:
        pending.append(READBACK_PENDING)

    result_status = projection["result_status"]
    result_reference: dict | None = None
    if result_status == "completed_unqualified":
        result_reference = _result_reference(directory, projection)
        if result_reference is None:
            pending.append(PUBLICATION_PENDING)

    if pending:
        return _explicit(pending, state=state)

    state["terminal_website_readback"] = _record(directory / "policy_canary_webapp_sync.json")
    state["terminal_resource_closure"] = _record(directory / "provider_zero_closure.json")

    join = {
        "schema_version": JOIN_SCHEMA,
        "owner": owner_reference,
        "result_status": result_status,
        "claim_ceiling": projection["claim_ceiling"],
        "run_kind": projection["run_kind"],
        "scene_controls_status": projection["scene_controls_status"],
        "website_readback_digest": readback["policy_canary_projection_digest"],
        "provider_zero_receipt_digest": closure["provider_zero_receipt_digest"],
        "result_reference": result_reference,
        "failed_children": failed_children,
        "claim_upgraded": False,
        "provider_mutation_performed": False,
        "join_digest": "",
    }
    join["join_digest"] = canonical_digest(join, digest_field="join_digest")

    join_path = _write_join(join, output=output, directory=directory)
    state["terminal_join"] = _record(join_path)

    if result_status == "completed_unqualified":
        return {"terminal": True, "status": "completed", "phase": "policy_canary_complete",
                "blockers": [], "result_reference": result_reference, "state": state}
    blockers = sorted({str(b) for b in projection.get("blockers", []) if b})
    return {"terminal": True, "status": "blocked", "phase": "policy_canary_blocked",
            "blockers": blockers, "result_reference": None, "state": state}


def _write_join(join: dict, *, output: str | Path | None, directory: Path) -> Path:
    target_root = _safe_path(Path(output)) if output is not None else directory
    target_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    path = target_root / TERMINAL_JOIN_FILENAME
    if path.exists() or path.is_symlink():
        existing = _read_json(path)
        _require(existing == join, "join_record_conflict")
        return path
    intake.write_exclusive(path, join)
    return path


__all__ = [
    "JOIN_SCHEMA",
    "TERMINAL_JOIN_FILENAME",
    "TerminalReconciliationError",
    "reconcile_terminal_owner_result",
]
