"""Single fail-closed admission chokepoint for canonical paid allocators."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping


PAID_LANE_ADMISSION_SCHEMA_VERSION = "paid_lane_admission.v1"
PAID_RESOURCE_CLASSES = frozenset(
    {
        "cloud_run_cpu_job",
        "cpu_build",
        "evaluator_api",
        "fullenrich_work_email",
        "gpu_canary",
        "model_volume",
        "openai_api_candidate",
        "parallel_findall",
        "provider_reconstruction_api",
        "gpu_render",
        "lambda_provider_adapter",
        "runpod_provider_adapter",
        "runpod_serverless_active_worker",
        "runpod_live_execution",
        "runpod_wam_async",
        "skypilot_vast_pilot",
        "unitree_unifolm_runpod",
        "vast_provider_adapter",
        "vast_wam_async",
    }
)
_GRANT_ISSUER = object()

#: RunPod is retired (owner decision, 2026-11-05). No new RunPod pod, endpoint,
#: serverless run or network volume may start, so no grant is issued for a class
#: that starts RunPod spend. Reads and teardown stay available: the spend guard
#: still inventories RunPod and reaps an orphan while a RunPod key exists, and
#: ``runpod_live_execution`` is absent on purpose (its grant only stops or
#: terminates an existing pod). ``model_volume`` is a RunPod network volume; only
#: ``groot_oscar_runpod_storage_volume`` issues it. This is a code constant, not an
#: environment switch: re-enabling RunPod is a reviewed code change.
RUNPOD_PROVIDER_RETIRED = True
RUNPOD_RETIRED_BLOCKER = "runpod_provider_retired"
RUNPOD_ALLOCATION_RESOURCE_CLASSES = frozenset(
    {
        "model_volume",
        "runpod_provider_adapter",
        "runpod_serverless_active_worker",
        "runpod_wam_async",
        "unitree_unifolm_runpod",
    }
)


def runpod_retirement_blockers(resource_class: str | None = None) -> list[str]:
    """``[runpod_provider_retired]`` for a RunPod allocation, else ``[]``.

    Without ``resource_class`` the caller already knows the request is a RunPod
    allocation (the allocator CLI, the RunPod render launcher).
    """

    if not RUNPOD_PROVIDER_RETIRED:
        return []
    if resource_class is None or resource_class in RUNPOD_ALLOCATION_RESOURCE_CLASSES:
        return [RUNPOD_RETIRED_BLOCKER]
    return []


#: Canonical allocator routes whose own branch allocates RunPod. Subcommands: the
#: serverless warm worker and the RunPod network-volume model cache (build, run,
#: retention). ``gpu-canary`` probe kinds: the generic startup, strict-policy-smoke and
#: persistent WAM loop canaries (RunPod only), the two kitchen lanes when
#: ``--provider runpod`` is chosen, and the OpenPI lanes when ``--openpi-provider
#: runpod`` is. Other probe kinds require Vast, ignore ``--provider``, or reach RunPod
#: only through the shared admission or the RunPod render launcher, which refuse it
#: themselves. A terminal resource release only deletes one stopped Vast record.
RUNPOD_ONLY_ALLOCATOR_COMMANDS = frozenset({"gpu-warm-worker", "model-volume", "model-volume-run"})
RUNPOD_ONLY_PROBE_KINDS = frozenset({"startup", "strict-policy-smoke", "persistent-policy-wam-loop"})
RUNPOD_SELECTABLE_PROBE_KINDS = frozenset({"single-kitchen-episode", "single-kitchen-finetune"})
OPENPI_PROVIDER_PROBE_KINDS = frozenset({"openpi-policy-ranking", "new-site-diagnostic-canary"})


def runpod_allocator_request_blockers(args: Any) -> list[str]:
    """Refuse a canonical allocator request that would allocate, start or extend RunPod."""

    command = getattr(args, "command", None)
    if command in RUNPOD_ONLY_ALLOCATOR_COMMANDS:
        return runpod_retirement_blockers()
    if command != "gpu-canary" or getattr(args, "terminal_resource_release", None):
        return []
    probe_kind = getattr(args, "probe_kind", None)
    if (
        probe_kind in RUNPOD_ONLY_PROBE_KINDS
        or (probe_kind in RUNPOD_SELECTABLE_PROBE_KINDS and getattr(args, "provider", None) == "runpod")
        or (probe_kind in OPENPI_PROVIDER_PROBE_KINDS and getattr(args, "openpi_provider", None) == "runpod")
    ):
        return runpod_retirement_blockers()
    return []


@dataclass(frozen=True)
class PaidResourceAdmissionGrant:
    resource_class: str
    schema_version: str
    _issuer: object = field(repr=False, compare=False)
    allocation_binding_digest: str | None = None
    _lease_check: Any = field(default=None, repr=False, compare=False)


class PaidResourceAdmissionBlocked(RuntimeError):
    def __init__(self, blockers: list[str]):
        self.blockers = blockers
        super().__init__("paid_resource_admission_blocked:" + ",".join(blockers))


def _monthly_budget_blockers(resource_class: str) -> list[str]:
    explicit = os.getenv("BLUEPRINT_GCP_MONTHLY_BUDGET_REPORT", "").strip()
    reconstruction = (resource_class == "provider_reconstruction_api"
                      and os.getenv("BLUEPRINT_GCP_BUDGET_GUARD_CONFIG", "").strip())
    if not explicit and not reconstruction:
        return []
    try:
        from blueprint_pipeline.gcp_monthly_budget_admission import admission_blockers, project_admission_blockers

        if explicit:
            return admission_blockers()
        project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        return project_admission_blockers(project) if project else ["gcp_monthly_budget_project_binding_missing"]
    except Exception:
        return ["gcp_monthly_budget_report_unavailable"]


#: What a paid admission does to spend. ``new_spend`` (the default) starts or
#: extends spend and is checked against the account monthly limit. ``stop`` only
#: stops, tears down or destroys what is already running; the limit never blocks
#: it, so an operator can always stop spend. A stop must say so twice: the caller
#: passes ``spend_effect="stop"`` and the admission record it built carries
#: ``"spend_effect": "stop"``. Anything else is checked as new spend.
SPEND_EFFECT_NEW_SPEND = "new_spend"
SPEND_EFFECT_STOP = "stop"
SPEND_EFFECTS = frozenset({SPEND_EFFECT_NEW_SPEND, SPEND_EFFECT_STOP})


def _spend_effect_blockers(admission: Mapping[str, Any], spend_effect: str) -> tuple[list[str], bool]:
    """Blockers for the declared spend effect, and whether this is a clear stop."""
    if spend_effect not in SPEND_EFFECTS:
        return ["paid_resource_admission_spend_effect_invalid"], False
    if spend_effect != SPEND_EFFECT_STOP:
        return [], False
    if admission.get("spend_effect") != SPEND_EFFECT_STOP:
        return ["paid_resource_admission_spend_effect_mismatch"], False
    return [], True


def _account_monthly_limit_blockers() -> list[str]:
    """The account-wide monthly limit (``account_monthly_spend``).

    ``[]`` while the month is under its limit. With no limit file the
    production control plane refuses (``account_monthly_spend_limit_not_configured``);
    any other process has no limit and gets ``[]``.

    Checked where a new paid admission is made or about to be reserved, never
    for a clear stop (``spend_effect``) and never when an issued grant is
    re-validated: stops, work already admitted, teardown and reconciliation are
    never refused by it.
    """
    try:
        from blueprint_pipeline.account_monthly_spend import admission_blockers

        return admission_blockers()
    except Exception:
        return ["account_monthly_spend_limit_unavailable"]


def require_paid_resource_budget(*, resource_class: str) -> None:
    """Refuse a known budget blocker before reserving external sponsorship.

    This issues no grant. Dispatch still checks the complete admission and
    current budget again after the remote transaction.
    """
    blockers = _monthly_budget_blockers(resource_class)
    blockers.extend(_account_monthly_limit_blockers())
    if resource_class not in PAID_RESOURCE_CLASSES:
        blockers.append("paid_resource_class_invalid")
    blockers.extend(runpod_retirement_blockers(resource_class))
    if blockers:
        raise PaidResourceAdmissionBlocked(sorted(set(blockers)))


def require_paid_resource_admission(
    admission: Mapping[str, Any],
    *,
    resource_class: str,
    expected_schema_version: str,
    spend_effect: str = SPEND_EFFECT_NEW_SPEND,
) -> PaidResourceAdmissionGrant:
    """Return an in-process grant or raise before a provider mutation.

    ``spend_effect="stop"`` marks an admission that only stops or destroys
    running spend; see ``SPEND_EFFECTS``.
    """
    from .core.handoff_lease import handoff_lease_checkpoint, current_handoff_lease_check
    handoff_lease_checkpoint("paid_resource_admission")
    blockers: list[str] = _monthly_budget_blockers(resource_class)
    effect_blockers, clear_stop = _spend_effect_blockers(admission, spend_effect)
    blockers.extend(effect_blockers)
    if not clear_stop:
        blockers.extend(_account_monthly_limit_blockers())
    if resource_class not in PAID_RESOURCE_CLASSES:
        blockers.append("paid_resource_class_invalid")
    blockers.extend(runpod_retirement_blockers(resource_class))
    admission_resource_class = admission.get("resource_class")
    if (admission_resource_class is not None and admission_resource_class != resource_class) or (
        expected_schema_version == PAID_LANE_ADMISSION_SCHEMA_VERSION
        and admission_resource_class is None
    ):
        blockers.append("paid_resource_admission_class_mismatch")
    if admission.get("schema_version") != expected_schema_version:
        blockers.append("paid_resource_admission_schema_invalid")
    if admission.get("status") != "admitted":
        blockers.append("paid_resource_admission_not_admitted")
    raw_blockers = admission.get("blockers")
    if raw_blockers not in ([], None):
        blockers.append("paid_resource_admission_has_blockers")
    if blockers:
        raise PaidResourceAdmissionBlocked(sorted(set(blockers)))
    # Every grant is metered against its job (``paid_spend_meter``); never raises.
    from .paid_spend_meter import meter_chokepoint_admission

    meter_chokepoint_admission(admission, resource_class=resource_class, schema_version=expected_schema_version,
                               spend_effect=SPEND_EFFECT_STOP if clear_stop else SPEND_EFFECT_NEW_SPEND)
    return PaidResourceAdmissionGrant(
        resource_class=resource_class,
        schema_version=expected_schema_version,
        allocation_binding_digest=(
            str(admission.get("allocation_binding_digest"))
            if admission.get("allocation_binding_digest") is not None
            else None
        ),
        _issuer=_GRANT_ISSUER,
        _lease_check=current_handoff_lease_check(),
    )


def require_paid_resource_admission_grant(
    grant: PaidResourceAdmissionGrant | None,
    *,
    resource_class: str,
    allocation_binding_digest: str | None = None,
    require_allocation_binding: bool = False,
    permit_reconciliation: bool = False,
) -> None:
    """Validate the opaque in-process capability issued by the chokepoint."""

    if not permit_reconciliation:
        from .core.handoff_lease import handoff_lease_checkpoint
        if isinstance(grant, PaidResourceAdmissionGrant) and grant._lease_check is not None:
            grant._lease_check("paid_provider_effect")
        else:
            handoff_lease_checkpoint("paid_provider_effect")
    blockers: list[str] = _monthly_budget_blockers(resource_class)
    if not isinstance(grant, PaidResourceAdmissionGrant):
        blockers.append("paid_resource_admission_grant_missing")
    else:
        if grant._issuer is not _GRANT_ISSUER:
            blockers.append("paid_resource_admission_grant_issuer_invalid")
        if grant.resource_class != resource_class:
            blockers.append("paid_resource_admission_grant_class_mismatch")
        if require_allocation_binding and not grant.allocation_binding_digest:
            blockers.append("paid_resource_admission_grant_binding_missing")
        if (
            allocation_binding_digest is not None
            and grant.allocation_binding_digest != allocation_binding_digest
        ):
            blockers.append("paid_resource_admission_grant_binding_mismatch")
    if blockers:
        raise PaidResourceAdmissionBlocked(blockers)


def build_paid_lane_admission(
    *,
    resource_class: str,
    blockers: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Normalize an existing lane's checks into the one shared admission shape.

    Provider-specific preflights may remain richer, but no paid mutation may
    treat their bespoke status fields as an alternate admission mechanism.
    The caller passes every blocker accumulated immediately before mutation,
    then calls :func:`require_paid_resource_admission` on this record.
    """

    normalized = sorted({str(item).strip() for item in blockers if str(item).strip()})
    if resource_class not in PAID_RESOURCE_CLASSES:
        normalized.append("paid_resource_class_invalid")
    normalized = sorted({*normalized, *runpod_retirement_blockers(resource_class)})
    return {
        "schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION,
        "status": "admitted" if not normalized else "blocked",
        "resource_class": resource_class,
        "blockers": normalized,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
