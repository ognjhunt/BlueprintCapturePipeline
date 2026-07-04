"""Provider reliability manifest — service-grade phase/blocker ledger for paid GPU runs.

Every paid provider run (RunPod, Vast, Lambda, DigitalOcean, or future providers)
moves through the same ordered phases:

1. pre_spend_preflight   — capacity, image, runtime-contract, credential checks. No money yet.
2. provider_launch       — the provider accepted the create/launch request and is billing.
3. container_startup     — the worker container booted and wrote its startup marker.
4. runtime_execution     — Isaac/WAM/simulator work is producing runtime progress markers.
5. artifact_collection   — run outputs were pulled back locally with checksums.
6. artifact_quality      — collected artifacts are fresh, decodable, and non-degenerate.
7. task_evaluation       — task-level judgment (delegated to success_claim_contracts).
8. teardown              — the billing allocation is proven terminated.

The manifest exists so ops never has to hunt old sessions: one JSON file per run
records exactly which phase the run reached, the exact blocker that stopped it,
spend accounting, and teardown proof. It fails closed everywhere: a phase with no
evidence is FAIL, not PASS, and provider/runtime success can never upgrade
artifact-quality or task-success claims (see ``success_claim_contracts``).

This module is pure and provider-agnostic: builders take evidence, return dicts.
Runners own the side effects (API calls, polling, kills) and feed evidence in.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .common import utc_now_iso
from .success_claim_contracts import coerce_strict_success

MANIFEST_SCHEMA_VERSION = "provider_reliability_manifest.v1"
PRE_SPEND_PREFLIGHT_SCHEMA_VERSION = "pre_spend_preflight.v1"
STALL_POLICY_SCHEMA_VERSION = "post_marker_stall_policy.v1"
TEARDOWN_PROOF_SCHEMA_VERSION = "provider_teardown_proof.v1"
ARTIFACT_COLLECTION_SCHEMA_VERSION = "provider_artifact_collection.v1"

# Ordered run phases. A run's ``furthest_phase_reached`` is the last phase with a
# recorded start; ``failed_phase`` is the first phase whose contract failed.
RUN_PHASES: tuple[str, ...] = (
    "pre_spend_preflight",
    "provider_launch",
    "container_startup",
    "runtime_execution",
    "artifact_collection",
    "artifact_quality",
    "task_evaluation",
    "teardown",
)

KNOWN_PROVIDERS = frozenset({"runpod", "vast", "lambda", "digitalocean", "fixture"})

# Blocker codes are stable, grep-able identifiers. Free-text detail goes after ':'.
BLOCKER_CAPACITY_UNAVAILABLE = "capacity_unavailable"
BLOCKER_IMAGE_CONTRACT_INVALID = "worker_image_contract_invalid"
BLOCKER_RUNTIME_CONTRACT_INVALID = "runtime_contract_invalid"
BLOCKER_CREDENTIAL_MISSING = "provider_credential_missing"
BLOCKER_SPEND_GATE_CLOSED = "spend_gate_closed"
BLOCKER_STARTUP_MARKER_TIMEOUT = "startup_marker_timeout"
BLOCKER_POST_MARKER_NO_PROGRESS = "post_marker_no_progress"
BLOCKER_RUNNER_FAILED = "runner_failed"
BLOCKER_TEARDOWN_UNPROVEN = "teardown_unproven"
BLOCKER_STALE_ARTIFACT = "stale_artifact_rejected"


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _dedupe(blockers: Sequence[str]) -> list[str]:
    return sorted({str(b).strip() for b in blockers if str(b).strip()})


def _result(
    schema_version: str,
    *,
    passed: bool,
    blockers: Sequence[str],
    claim_boundary: str,
    **fields: Any,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "generated_at": utc_now_iso(),
        "status": "PASS" if passed else "FAIL",
        "blockers": _dedupe(blockers),
        "claim_boundary": claim_boundary,
        **fields,
    }


def _passed(contract: Mapping[str, Any] | None) -> bool:
    if not isinstance(contract, Mapping):
        return False
    return str(contract.get("status") or "").strip().upper() == "PASS"


# ---------------------------------------------------------------------------
# Phase 1 — pre-spend preflight. Fail BEFORE money, not after.
# ---------------------------------------------------------------------------


def build_pre_spend_preflight(
    *,
    provider: str,
    credential_present: bool,
    capacity_evidence: Mapping[str, Any] | None,
    image_contract: Mapping[str, Any] | None,
    runtime_contract: Mapping[str, Any] | None,
    spend_gate_open: Any = None,
) -> dict[str, Any]:
    """Gate that must PASS before any billable provider request is sent.

    - ``capacity_evidence``: result of a read-only capacity/offer query
      (``available`` strict bool, optional ``offer_count``/``detail``). Missing or
      non-boolean availability fails closed — an unverified offer is not capacity.
    - ``image_contract``: the worker image policy check (``image_ref`` non-empty,
      ``pinned`` strict bool; optional ``digest``). Unpinned or unnamed images fail.
    - ``runtime_contract``: the worker runtime expectations (``startup_marker``
      and ``progress_marker`` names, ``startup_timeout_seconds`` > 0,
      ``no_progress_timeout_seconds`` > 0). A run without marker/timeout contracts
      cannot be monitored and must not spend.
    - ``spend_gate_open``: the explicit operator spend approval (strict bool).
    """
    blockers: list[str] = []
    provider_name = str(provider or "").strip().lower()
    if not provider_name:
        blockers.append(f"{BLOCKER_RUNTIME_CONTRACT_INVALID}:provider_missing")
    if not credential_present:
        blockers.append(f"{BLOCKER_CREDENTIAL_MISSING}:{provider_name or 'unknown'}")

    gate = coerce_strict_success(spend_gate_open)
    if gate is not True:
        blockers.append(f"{BLOCKER_SPEND_GATE_CLOSED}:explicit_spend_approval_missing")

    capacity = _mapping(capacity_evidence)
    available = coerce_strict_success(capacity.get("available"))
    if available is None:
        blockers.append(f"{BLOCKER_CAPACITY_UNAVAILABLE}:capacity_evidence_missing")
    elif available is False:
        detail = str(capacity.get("detail") or "no_matching_offer").strip()
        blockers.append(f"{BLOCKER_CAPACITY_UNAVAILABLE}:{detail}")

    image = _mapping(image_contract)
    image_ref = str(image.get("image_ref") or "").strip()
    if not image_ref:
        blockers.append(f"{BLOCKER_IMAGE_CONTRACT_INVALID}:image_ref_missing")
    pinned = coerce_strict_success(image.get("pinned"))
    if pinned is not True:
        blockers.append(f"{BLOCKER_IMAGE_CONTRACT_INVALID}:image_not_pinned:{image_ref or 'unknown'}")

    runtime = _mapping(runtime_contract)
    if not str(runtime.get("startup_marker") or "").strip():
        blockers.append(f"{BLOCKER_RUNTIME_CONTRACT_INVALID}:startup_marker_undefined")
    if not str(runtime.get("progress_marker") or "").strip():
        blockers.append(f"{BLOCKER_RUNTIME_CONTRACT_INVALID}:progress_marker_undefined")
    for field in ("startup_timeout_seconds", "no_progress_timeout_seconds"):
        raw = runtime.get(field)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 0:
            blockers.append(f"{BLOCKER_RUNTIME_CONTRACT_INVALID}:{field}_not_positive")

    return _result(
        PRE_SPEND_PREFLIGHT_SCHEMA_VERSION,
        passed=not blockers,
        blockers=blockers,
        claim_boundary=(
            "Pre-spend preflight proves the launch contracts are valid before spend. It "
            "does not prove the launch will succeed, the container will boot, or any "
            "artifact/task outcome."
        ),
        provider=provider_name or None,
        spend_allowed=not blockers,
        capacity_evidence=capacity or None,
        image_contract=image or None,
        runtime_contract=runtime or None,
    )


# ---------------------------------------------------------------------------
# Post-marker stall policy — a booted pod that stops progressing must die.
# ---------------------------------------------------------------------------


def evaluate_post_marker_stall(
    *,
    startup_marker_seen: bool,
    startup_elapsed_seconds: float,
    startup_timeout_seconds: float,
    last_progress_age_seconds: float | None,
    no_progress_timeout_seconds: float,
) -> dict[str, Any]:
    """Decide whether the monitored run has stalled and must be terminated.

    Two distinct stall modes, recorded distinctly:
    - startup-marker timeout: container never wrote its startup marker in time
      (provider launch succeeded, container startup failed);
    - post-marker no-progress: startup marker seen, but no runtime progress marker
      update within ``no_progress_timeout_seconds`` (container startup succeeded,
      runtime execution stalled).

    ``last_progress_age_seconds=None`` after startup means no progress marker has
    EVER been written; the age of the startup marker itself is then the stall
    clock — silence is a stall, never patience.
    """
    blockers: list[str] = []
    should_terminate = False
    stall_mode: str | None = None

    if not startup_marker_seen:
        if float(startup_elapsed_seconds) >= float(startup_timeout_seconds):
            stall_mode = "container_startup"
            should_terminate = True
            blockers.append(
                f"{BLOCKER_STARTUP_MARKER_TIMEOUT}:"
                f"{startup_elapsed_seconds:.0f}s>={startup_timeout_seconds:.0f}s"
            )
    else:
        progress_age = (
            float(last_progress_age_seconds)
            if last_progress_age_seconds is not None
            else float(startup_elapsed_seconds)
        )
        if progress_age >= float(no_progress_timeout_seconds):
            stall_mode = "runtime_execution"
            should_terminate = True
            detail = (
                "no_progress_marker_ever_written"
                if last_progress_age_seconds is None
                else f"last_progress_{progress_age:.0f}s_ago"
            )
            blockers.append(
                f"{BLOCKER_POST_MARKER_NO_PROGRESS}:{detail}"
                f":timeout_{no_progress_timeout_seconds:.0f}s"
            )

    return _result(
        STALL_POLICY_SCHEMA_VERSION,
        passed=not should_terminate,
        blockers=blockers,
        claim_boundary=(
            "Stall evaluation is a billing-protection decision about marker recency. A "
            "non-stalled run is not a successful run."
        ),
        should_terminate=should_terminate,
        stall_mode=stall_mode,
        startup_marker_seen=bool(startup_marker_seen),
    )


# ---------------------------------------------------------------------------
# Teardown proof — every paid run must end with evidence the billing stopped.
# ---------------------------------------------------------------------------


def build_teardown_proof(
    *,
    provider: str,
    allocation_id: str | None,
    terminate_requested: bool,
    provider_terminal_status: str | None,
    verified_at: str | None = None,
    keep_alive_requested: bool = False,
    keep_alive_reason: str | None = None,
) -> dict[str, Any]:
    """Teardown is proven only by a provider-reported terminal status.

    A terminate *request* alone is not proof — the provider must report the
    allocation terminal (or the operator must have explicitly kept it alive, in
    which case the manifest records an OPEN billing allocation, never silence).
    """
    blockers: list[str] = []
    terminal = str(provider_terminal_status or "").strip().lower()
    alloc = str(allocation_id or "").strip()

    if keep_alive_requested:
        if not str(keep_alive_reason or "").strip():
            blockers.append(f"{BLOCKER_TEARDOWN_UNPROVEN}:keep_alive_without_reason")
        blockers.append(f"{BLOCKER_TEARDOWN_UNPROVEN}:allocation_intentionally_kept_alive:{alloc or 'unknown'}")
    else:
        if not alloc:
            blockers.append(f"{BLOCKER_TEARDOWN_UNPROVEN}:allocation_id_missing")
        if not terminate_requested:
            blockers.append(f"{BLOCKER_TEARDOWN_UNPROVEN}:terminate_never_requested")
        if not terminal:
            blockers.append(f"{BLOCKER_TEARDOWN_UNPROVEN}:terminal_status_not_verified")
        elif terminal not in {
            "terminated",
            "destroyed",
            "deleted",
            "exited",
            "not_found",
        }:
            blockers.append(f"{BLOCKER_TEARDOWN_UNPROVEN}:non_terminal_status:{terminal}")
        if terminal and not str(verified_at or "").strip():
            blockers.append(f"{BLOCKER_TEARDOWN_UNPROVEN}:verification_timestamp_missing")

    return _result(
        TEARDOWN_PROOF_SCHEMA_VERSION,
        passed=not blockers,
        blockers=blockers,
        claim_boundary=(
            "Teardown proof shows the billing allocation reached a provider-reported "
            "terminal state. It says nothing about run success or artifact quality."
        ),
        provider=str(provider or "").strip().lower() or None,
        allocation_id=alloc or None,
        billing_stopped=not blockers,
        provider_terminal_status=terminal or None,
        keep_alive_requested=bool(keep_alive_requested),
        keep_alive_reason=str(keep_alive_reason or "").strip() or None,
        verified_at=str(verified_at or "").strip() or None,
    )


# ---------------------------------------------------------------------------
# Artifact collection — required vs optional artifacts, stale rejection.
# ---------------------------------------------------------------------------


def build_artifact_collection(
    *,
    required_artifacts: Sequence[Mapping[str, Any]],
    optional_artifacts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Collection contract over per-artifact evidence.

    Each artifact entry: ``name``, ``present`` (strict bool), optional
    ``freshness`` (an ``artifact_freshness_evidence.v1`` result), optional
    ``checksum``. Required artifacts must be present AND fresh; a present-but-stale
    artifact is rejected, not accepted. Optional artifacts may be explicitly
    skipped (``skipped=True`` with a ``skip_reason``) without failing the phase —
    but a skipped artifact is recorded, never silently absent.
    """
    blockers: list[str] = []
    collected: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def _check(entry: Mapping[str, Any], *, required: bool) -> None:
        item = _mapping(entry)
        name = str(item.get("name") or "").strip() or "unnamed_artifact"
        if coerce_strict_success(item.get("skipped")) is True:
            reason = str(item.get("skip_reason") or "").strip()
            if required:
                blockers.append(f"required_artifact_skipped:{name}")
            elif not reason:
                blockers.append(f"optional_artifact_skipped_without_reason:{name}")
            skipped.append({"name": name, "skip_reason": reason or None, "required": required})
            return
        present = coerce_strict_success(item.get("present"))
        if present is not True:
            if required:
                blockers.append(f"required_artifact_missing:{name}")
            else:
                blockers.append(f"optional_artifact_neither_present_nor_skipped:{name}")
            return
        freshness = item.get("freshness")
        if freshness is None:
            blockers.append(f"{BLOCKER_STALE_ARTIFACT}:{name}:freshness_evidence_missing")
            return
        if not _passed(freshness):
            stale_detail = ",".join(_string_list(_mapping(freshness).get("blockers"))) or "not_fresh"
            blockers.append(f"{BLOCKER_STALE_ARTIFACT}:{name}:{stale_detail}")
            return
        collected.append(
            {
                "name": name,
                "checksum": str(item.get("checksum") or "").strip() or None,
                "required": required,
            }
        )

    for entry in required_artifacts:
        _check(entry, required=True)
    for entry in optional_artifacts:
        _check(entry, required=False)

    return _result(
        ARTIFACT_COLLECTION_SCHEMA_VERSION,
        passed=not blockers,
        blockers=blockers,
        claim_boundary=(
            "Artifact collection proves this run's outputs were retrieved and are fresh. "
            "It is not artifact quality, task success, or any higher claim."
        ),
        collected=collected,
        skipped=skipped,
        required_count=len(list(required_artifacts)),
        optional_count=len(list(optional_artifacts)),
    )


# ---------------------------------------------------------------------------
# Composed manifest — the single ops-facing record per paid run.
# ---------------------------------------------------------------------------


def build_provider_reliability_manifest(
    *,
    run_id: str,
    provider: str,
    session_dir: str | None = None,
    launched_at: str | None = None,
    spend: Mapping[str, Any] | None = None,
    pre_spend_preflight: Mapping[str, Any] | None = None,
    provider_launch: Mapping[str, Any] | None = None,
    container_startup: Mapping[str, Any] | None = None,
    runtime_execution: Mapping[str, Any] | None = None,
    artifact_collection: Mapping[str, Any] | None = None,
    artifact_quality: Mapping[str, Any] | None = None,
    task_evaluation: Mapping[str, Any] | None = None,
    teardown: Mapping[str, Any] | None = None,
    not_applicable_phases: Sequence[str] = (),
) -> dict[str, Any]:
    """Compose per-phase contracts into one ops-facing reliability manifest.

    - ``furthest_phase_reached``: last phase with any recorded contract.
    - ``failed_phase`` + ``failure_blockers``: the FIRST failing phase in order —
      the exact place and reason the run stopped, so nobody replays a session log
      to find out what happened.
    - Later phases recorded after an earlier failure stay in the manifest but
      cannot flip the failed phase: the first failure is the run's story.
    - ``teardown`` is mandatory: a manifest without teardown proof reports an
      open billing risk regardless of how well the run went.
    - ``not_applicable_phases``: phases a lane explicitly does not perform (e.g.
      a render-only lane never does ``task_evaluation``). They are recorded as
      not-applicable — never as PASS — and excluded from the all-phases-present
      completion requirement. Teardown and pre-spend preflight can never be
      declared not applicable.
    """
    not_applicable = {
        str(p).strip() for p in not_applicable_phases if str(p).strip() in RUN_PHASES
    } - {"teardown", "pre_spend_preflight"}
    phases: dict[str, dict[str, Any] | None] = {
        "pre_spend_preflight": _mapping(pre_spend_preflight) or None,
        "provider_launch": _mapping(provider_launch) or None,
        "container_startup": _mapping(container_startup) or None,
        "runtime_execution": _mapping(runtime_execution) or None,
        "artifact_collection": _mapping(artifact_collection) or None,
        "artifact_quality": _mapping(artifact_quality) or None,
        "task_evaluation": _mapping(task_evaluation) or None,
        "teardown": _mapping(teardown) or None,
    }

    furthest: str | None = None
    failed_phase: str | None = None
    failure_blockers: list[str] = []
    for phase in RUN_PHASES:
        contract = phases[phase]
        if contract is not None and phase not in not_applicable:
            furthest = phase
            if failed_phase is None and not _passed(contract):
                failed_phase = phase
                failure_blockers = _string_list(contract.get("blockers"))

    teardown_contract = phases["teardown"]
    teardown_proven = _passed(teardown_contract)
    open_billing_risk = not teardown_proven

    blockers: list[str] = list(failure_blockers)
    if teardown_contract is None:
        blockers.append(f"{BLOCKER_TEARDOWN_UNPROVEN}:teardown_contract_missing")
    elif not teardown_proven:
        blockers.extend(_string_list(teardown_contract.get("blockers")))

    # A run "completed" only when every applicable phase passed AND teardown is
    # proven. Anything else names its failed phase.
    completed = failed_phase is None and teardown_proven and all(
        phases[p] is not None for p in RUN_PHASES if p not in not_applicable
    )

    provider_name = str(provider or "").strip().lower()
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "run_id": str(run_id or "").strip(),
        "provider": provider_name,
        "provider_known": provider_name in KNOWN_PROVIDERS,
        "session_dir": str(session_dir or "").strip() or None,
        "launched_at": str(launched_at or "").strip() or None,
        "run_completed": completed,
        "furthest_phase_reached": furthest,
        "failed_phase": failed_phase,
        "failure_blockers": _dedupe(failure_blockers),
        "open_billing_risk": open_billing_risk,
        "teardown_proven": teardown_proven,
        "spend": _mapping(spend) or None,
        "not_applicable_phases": sorted(not_applicable),
        "phases": {
            name: (
                {
                    "present": contract is not None,
                    "passed": _passed(contract) and name not in not_applicable,
                    "not_applicable": name in not_applicable,
                    "blockers": _string_list((contract or {}).get("blockers")),
                }
            )
            for name, contract in phases.items()
        },
        "phase_contracts": phases,
        "blockers": _dedupe(blockers),
        "claim_boundary": (
            "This manifest records provider-run reliability phases only. Provider launch, "
            "container startup, and runtime execution are infrastructure health; artifact "
            "quality is media evidence; task success claims live in the "
            "success_claim_ledger and are never implied by any phase here."
        ),
    }
    if not manifest["run_id"]:
        manifest["blockers"] = _dedupe([*manifest["blockers"], "run_id_missing"])
        manifest["run_completed"] = False
    return manifest
