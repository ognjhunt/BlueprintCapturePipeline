"""Recover a scene-configuration launch whose own records prove it spent nothing.

A Website scene is built on one rented Vast GPU. Until 2026-11-10 only three
pre-allocation refusals (disk, credit, authoring authentication) were retried
by themselves; every other launch that stopped before renting anything (no GPU
offer after the in-run search, an object-storage staging error, a Vast
inventory query error, a watchdog that would not arm, a refused paid admission)
parked at ``scene_configuration_failed`` until someone deployed code. A launch
that ended without an allocator result showed no blocker at all.

This module names those classes and the proof each needs. It grants no
allocation authority: a recovery is a new $0 preparation attempt, whose paid
launch is still reserved, admitted and capped exactly like the first.

* ``preallocation_transient``: the allocator result itself proves no machine
  and no paid API call -- no instance, no provider mutation, no API
  pretraining, no CPU prestage, no output, nothing still spending, every
  blocker one of the transient refusals below or a consequence of producing
  nothing -- and the launch's own digest-bound teardown manifest says no
  instance ever existed.
* ``launch_without_result``: no allocator result exists, and the dispatcher's
  sealed launch receipt proves the paid allocator never reached a provider: it
  was not invoked, it stopped before its paid admission, or its admission was
  refused for a stale or unreadable account spend report before the Vast lane
  ran. Any other receipt is a named blocker, never silence and never "$0".

Both share the existing bounded pre-allocation recovery counter
(``MAX_PREALLOCATION_CAPACITY_RECOVERIES``) and wait between recoveries on the
shared schedule for an unclassified refusal (``job_retry_policy``: 15 min, 1 h,
3 h, 6 h, 12 h, 24 h, about 46 h), long enough to outlast a GPU-market
shortage without rebuilding the CPU preparation in a tight loop.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

TRANSIENT_KIND = "preallocation_transient"
RECEIPT_KIND = "launch_without_result"
ZERO_SPEND_KINDS = frozenset({TRANSIENT_KIND, RECEIPT_KIND})
BACKOFF_BLOCKER = "scene_configuration_recovery_backoff"
#: The launch ended without an allocator result and its receipt cannot prove nothing was spent.
UNPROVEN_BLOCKER = "scene_configuration_launch_result_missing_spend_unproven"
#: No receipt long after the dispatcher's own allocator timeout: the dispatcher died.
LOST_BLOCKER = "scene_configuration_launch_dispatcher_lost"

#: Refusals that stop a launch before any machine exists and that the next
#: attempt can clear by itself. Codes are compared up to their first ':'.
TRANSIENT_BLOCKERS = frozenset({
    # Object storage (wam_provider_object_store) and the bundle's durable copy.
    "object_store_staging_blocked",
    "object_store_upload_or_presign_failed",
    "fresh_output_key_absence_unverified",
    "durable_provider_bundle_binding_readback_failed",
    "signed_output_sentinel_put_failed",
    "signed_output_sentinel_get_failed",
    "signed_output_sentinel_get_mismatch",
    "scene_configuration_provider_bundle_durable_publication_failed",
    # The independent watchdog could not arm, so the lane refused to allocate.
    "independent_watchdog_not_armed",
    # The adapter's prelaunch Vast inventory read failed.
    "vast_prelaunch_inventory_query_failed",
    # The in-run offer search found nothing to rent (vast_provider_adapter).
    "no_vast_offer_selected",
    "no_vast_offer_at_or_below_max_hourly_rate",
    "no_vast_offer_with_known_supported_isaac_driver_at_or_below_max_hourly_rate",
    "no_vast_offer_with_required_isaac_rt_capability",
    "no_vast_offer_meeting_minimum_driver_version",
    "no_vast_offer_meeting_min_compute_cap",
    "no_vast_offer_at_or_below_max_compute_cap",
    "no_vast_offer_in_allowed_geolocation_country",
    "vast_native_offer_search_coverage_incomplete",
})
#: Paid-admission refusals the spend guard clears by itself (account_monthly_spend).
#: A reached limit is a money stop and is never in this set.
ADMISSION_TRANSIENT_BLOCKERS = frozenset({
    "account_monthly_spend_report_stale",
    "account_monthly_spend_report_missing_or_invalid",
    "account_monthly_spend_limit_unavailable",
})
#: Past the dispatcher's own allocator timeout (hard TTL + 300 s) by this much.
LOST_DISPATCHER_GRACE_SECONDS = 900
_ADAPTER_REFUSAL_FIELDS = frozenset({"status", "blockers", "provider_mutations_performed", "retry_cap"})


def _code(blocker: Any) -> str:
    return str(blocker).split(":", 1)[0]


def _epoch(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)).timestamp()


def transient_launch_failure(result: Any) -> bool:
    """The allocator result proves a $0 refusal of one of the transient classes."""
    from .task_evaluation_scene_capacity_recovery import DEAD_MACHINE_CONSEQUENCE_BLOCKERS, nothing_started
    if not nothing_started(result) or result.get("cpu_prestage"):
        return False
    blockers = [str(b) for b in result.get("blockers") or []]
    return (bool(blockers)
            and any(_code(b) in TRANSIENT_BLOCKERS for b in blockers)
            and all(_code(b) in TRANSIENT_BLOCKERS or b in DEAD_MACHINE_CONSEQUENCE_BLOCKERS for b in blockers))


def unallocated_teardown(teardown: Any, adapter: Any = None) -> bool:
    """A teardown manifest that proves no Vast instance ever existed for this launch.

    The lane's own no-allocation seals say so by status. An adapter that searched
    and found no offer closes with ``completed``, no instance and no action; that
    counts only with the adapter result saying it never reached a create call.
    """
    from .task_evaluation_terminal_scene_attempt_settlement import UNALLOCATED_TEARDOWN_STATUSES
    if not (isinstance(teardown, Mapping) and teardown.get("schema_version") == "vast_teardown_manifest.v1"
            and teardown.get("vast_instance_ids") == []
            and teardown.get("continuing_spend_from_this_run") is False):
        return False
    if teardown.get("status") in UNALLOCATED_TEARDOWN_STATUSES:
        return True
    return (teardown.get("status") == "completed" and teardown.get("teardown_actions_performed") == []
            and isinstance(adapter, Mapping)
            and adapter.get("provider_create_attempted") is False
            and adapter.get("vast_side_effects_may_have_occurred") is False
            and adapter.get("vast_instance_ids") == []
            and adapter.get("continuing_spend_from_this_run") is False)


def adapter_result(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """The Vast adapter's own result beside an allocator result, when it ran."""
    from .task_evaluation_scene_configuration_submission_inputs import read
    raw = result.get("provider_adapter_result_path")
    if not isinstance(raw, str) or not raw or not Path(raw).is_file():
        return None
    try:
        return read(raw)
    except (OSError, ValueError):
        return None


def validate_transient_teardown(launch: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    """Reopen the launch's digest-bound teardown manifest; refuse unless it proves no instance."""
    from .task_evaluation_retained_controls_evidence import _file
    from .task_evaluation_scene_configuration_submission_inputs import read, require
    from .task_evaluation_scene_progression_contracts import safe_path
    ref = ((launch.get("terminal_evidence") or {}).get("artifacts") or {}).get("teardown_manifest_path") or {}
    require(ref.get("exists") is True and isinstance(ref.get("path"), str)
            and _file(safe_path(ref["path"]))["digest"] == ref.get("digest"), "transient_teardown_changed")
    require(unallocated_teardown(read(ref["path"]), adapter_result(result)), "transient_teardown_not_unallocated")


def _adapter_output_path(profile: Mapping[str, Any], run_root: Path) -> Path | None:
    from .task_evaluation_launch_dispatcher import _render_launch_path
    argv = [str(item) for item in ((profile.get("allocator") or {}).get("argv") or [])]
    for index, argument in enumerate(argv):
        raw = (argv[index + 1] if argument == "--adapter-output" and index + 1 < len(argv)
               else argument.split("=", 1)[1] if argument.startswith("--adapter-output=") else None)
        if raw:
            return Path(_render_launch_path(raw, run_root=run_root))
    return None


def receipt_zero_spend_proof(*, run_root: Path, receipt: Mapping[str, Any],
                             profile: Mapping[str, Any]) -> dict[str, Any] | None:
    """How a blocked receipt with no allocator result proves no provider was reached; else ``None``."""
    from .task_evaluation_scene_attempt_evidence import record
    from .task_evaluation_scene_capacity_recovery import RESULT_RELATIVE
    from .task_evaluation_scene_outcome import BLOCKER_REASONS, REASONS
    if (receipt.get("status") != "blocked" or receipt.get("launch_id") != run_root.name
            or (run_root / RESULT_RELATIVE).exists()):
        return None
    # A refusal the outcome registry already names as privacy, rights, money,
    # site action or capability is that job's answer, not an infrastructure miss.
    if any(REASONS[BLOCKER_REASONS[_code(b)]][0] != "setup_repair"
           for b in receipt.get("blockers") or [] if _code(b) in BLOCKER_REASONS):
        return None
    evidence = receipt.get("provider_mutation_evidence") or {}
    if (receipt.get("allocator_invoked") is False and receipt.get("allocator_exit_code") is None
            and receipt.get("provider_mutation_attempted") is False):
        return {"basis": "allocator_not_invoked"}
    if receipt.get("allocator_invoked") is not True:
        return None
    if (evidence.get("status") == "absent_before_paid_admission"
            and receipt.get("provider_mutation_attempted") is False):
        return {"basis": "absent_before_paid_admission"}
    path = _adapter_output_path(profile, run_root)
    if (evidence.get("status") != "allocator_boundary_artifacts_present"
            or evidence.get("terminal_result_artifact_present") is not False
            or evidence.get("bound_request_artifact_present") is not False
            or evidence.get("adapter_output_artifact_present") is not True
            or path is None or path.is_symlink() or not path.is_file()):
        return None
    from .task_evaluation_scene_configuration_submission_inputs import read
    try:
        refusal = read(path)
    except (OSError, ValueError):
        return None
    blockers = refusal.get("blockers")
    # The allocator writes exactly this before the Vast lane runs when its paid
    # admission is refused (task_evaluation_scene_configuration_allocator).
    if not (set(refusal) == _ADAPTER_REFUSAL_FIELDS and refusal["status"] == "blocked"
            and type(refusal["provider_mutations_performed"]) is int and refusal["provider_mutations_performed"] == 0
            and isinstance(blockers, list) and blockers
            and all(_code(b) in ADMISSION_TRANSIENT_BLOCKERS for b in blockers)):
        return None
    return {"basis": "paid_admission_refused", "adapter_output": record(path)}


def receipt_proof_still_holds(launch_path: Path) -> bool:
    """Settlement's re-check: this sealed receipt still proves no provider was reached."""
    from .task_evaluation_scene_configuration_submission_inputs import read
    try:
        run_root = launch_path.parent
        receipt = read(launch_path, digest_field="receipt_digest")
        profile = read(run_root / "launch_profile.json", digest_field="profile_digest")
        return (receipt.get("launch_profile_digest") == profile.get("profile_digest")
                and receipt_zero_spend_proof(run_root=run_root, receipt=receipt, profile=profile) is not None)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _ended_at(run_root: Path, receipt_path: Path) -> float:
    """When the launch ended: the write-once receipt's time, never before it started."""
    from .task_evaluation_scene_configuration_submission_inputs import read
    started_path = run_root / "launch_started.json"
    started = _epoch(read(started_path).get("started_at")) if started_path.is_file() else None
    return max(started or 0.0, receipt_path.stat().st_mtime)


def observe_without_result(*, directory: Path, profile: Mapping[str, Any], attempt: Mapping[str, Any],
                           now: float) -> dict[str, Any] | None:
    """Classify a launch of this attempt that has no allocator result yet.

    ``None`` while it may still be running. A receipt that proves no provider
    was reached is recoverable (for a $0 preparation attempt); any other
    receipt, or a dispatcher that never wrote one, is a named blocker.
    """
    from .task_evaluation_scene_attempt_evidence import record
    from .task_evaluation_scene_configuration_submission_inputs import read
    receipt_path = directory / "launch_receipt.json"
    started_path = directory / "launch_started.json"
    if not receipt_path.is_file():
        if not started_path.is_file():
            return None
        started = read(started_path, digest_field="started_digest")
        ttl = started.get("hard_ttl_seconds") or (profile.get("allocator") or {}).get("hard_ttl_seconds")
        begun = _epoch(started.get("started_at"))
        if type(ttl) not in (int, float) or begun is None or now < begun + ttl + 300 + LOST_DISPATCHER_GRACE_SECONDS:
            return None
        return {"recoverable": False, "blockers": [LOST_BLOCKER], "result": record(started_path)}
    receipt = read(receipt_path, digest_field="receipt_digest")
    if receipt.get("status") != "blocked":
        return None
    proof = (receipt_zero_spend_proof(run_root=directory, receipt=receipt, profile=profile)
             if receipt.get("launch_profile_digest") == profile.get("profile_digest") else None)
    blockers = sorted({str(b) for b in receipt.get("blockers") or [] if str(b)})
    if proof is None or attempt.get("schema_version") != "task_evaluation_scene_preparation_attempt.v1":
        return {"recoverable": False, "blockers": [UNPROVEN_BLOCKER, *blockers], "result": record(receipt_path)}
    ended = datetime.fromtimestamp(_ended_at(directory, receipt_path), timezone.utc).isoformat()
    return {"recoverable": True, "kind": RECEIPT_KIND, "blockers": blockers, "proof": proof,
            "result": record(receipt_path),
            # The successor's identity and the failure's time come from the receipt.
            "result_value": {"result_digest": receipt["receipt_digest"], "generated_at": ended}}


def receipt_values(observation: Mapping[str, Any], refs: Mapping[str, Any]) -> dict[str, Any]:
    """Reopen the bound records of a receipt-only launch for the recovery that follows."""
    from . import task_evaluation_scene_intent_contracts as intake
    from .task_evaluation_scene_configuration_submission_inputs import checked_file, read
    from .task_evaluation_scene_progression_contracts import safe_path
    fields = {"profile": "profile_digest", "request": "request_digest", "launch": "receipt_digest",
              "bundle": "receipt_digest", "authority": "authority_digest", "link": "link_digest",
              "factory": "factory_digest", "preparation": None}
    values = {name: read(checked_file(safe_path(refs[name]["path"]), refs[name]), digest_field=field)
              for name, field in fields.items()}
    values["configuration_attempt"] = intake._read(
        checked_file(safe_path(refs["configuration_attempt"]["path"]), refs["configuration_attempt"]), "attempt_digest")
    values["result"] = dict(observation["result_value"])
    return values


def backoff_seconds(prior_recoveries: int) -> int:
    """The shared refusal schedule's delay before the next zero-spend recovery; 0 once it is spent."""
    from .job_retry_policy import refusal_retry_delay
    return refusal_retry_delay(prior_recoveries) or 0


def backoff_blocker(due: float, now: float) -> str:
    """The wait's blocker, naming the whole hours left until the scheduled recovery.

    The stall alarm pages when a row's status, phase and blockers stay the same
    for 90 min. This code changes at least once an hour while the wait runs, so
    a scheduled wait does not page; a wait that does not end (0 hours left and
    still waiting) stays the same and does.
    """
    import math
    return f"{BACKOFF_BLOCKER}:due_in_hours:{max(0, math.ceil((due - now) / 3600))}"


def backoff_wait(observation: Mapping[str, Any], prior_recoveries: int, now: float) -> dict[str, Any] | None:
    """``None`` once this zero-spend failure has waited its backoff, else the wait to record."""
    if observation.get("kind") not in ZERO_SPEND_KINDS:
        return None
    failed = _epoch(observation["values"]["result"].get("generated_at"))
    if failed is None:
        return None
    due = failed + backoff_seconds(prior_recoveries)
    if now >= due:
        return None
    return {"kind": observation["kind"], "failed_at_epoch": failed, "prior_recoveries": prior_recoveries,
            "next_check_epoch": due, "blocker": backoff_blocker(due, now)}


def zero_spend_admission(observation: Mapping[str, Any], now: float) -> dict[str, Any]:
    """The recovery admission of a zero-spend class: its backoff has elapsed; nothing is measured.

    The successor builds and admits a new bundle; its disk, credit and spend
    gates run then exactly as for a first attempt.
    """
    value = {"schema_version": "task_evaluation_preallocation_capacity_admission.v1", "status": "admitted",
             "observed_at_epoch": now, "basis": "zero_spend_backoff_elapsed",
             "configuration_failure_kind": observation["kind"],
             "provider_mutation_performed": False, "historical_bundle_payload_required": False,
             "successor_requires_new_sealed_bundle": True}
    if observation.get("proof") is not None:
        value["launch_receipt_proof"] = observation["proof"]
    value["capacity_digest"] = canonical_digest(value, digest_field="capacity_digest")
    return value


__all__ = [
    "ADMISSION_TRANSIENT_BLOCKERS", "BACKOFF_BLOCKER", "LOST_BLOCKER", "RECEIPT_KIND", "TRANSIENT_BLOCKERS",
    "TRANSIENT_KIND", "UNPROVEN_BLOCKER", "ZERO_SPEND_KINDS", "backoff_seconds", "backoff_wait",
    "observe_without_result", "receipt_proof_still_holds", "receipt_values", "receipt_zero_spend_proof",
    "transient_launch_failure", "unallocated_teardown", "validate_transient_teardown", "zero_spend_admission",
]
