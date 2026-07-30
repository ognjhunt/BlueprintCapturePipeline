"""Single fail-closed admission chokepoint for canonical paid allocators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


PAID_LANE_ADMISSION_SCHEMA_VERSION = "paid_lane_admission.v1"
PAID_RESOURCE_CLASSES = frozenset(
    {
        "cpu_build",
        "gpu_canary",
        "model_volume",
        "openai_api_candidate",
        "gpu_render",
        "lambda_provider_adapter",
        "runpod_provider_adapter",
        "runpod_serverless_active_worker",
        "runpod_live_execution",
        "runpod_wam_async",
        "unitree_unifolm_runpod",
        "vast_provider_adapter",
        "vast_wam_async",
    }
)
_GRANT_ISSUER = object()


@dataclass(frozen=True)
class PaidResourceAdmissionGrant:
    resource_class: str
    schema_version: str
    _issuer: object = field(repr=False, compare=False)


class PaidResourceAdmissionBlocked(RuntimeError):
    def __init__(self, blockers: list[str]):
        self.blockers = blockers
        super().__init__("paid_resource_admission_blocked:" + ",".join(blockers))


def require_paid_resource_admission(
    admission: Mapping[str, Any],
    *,
    resource_class: str,
    expected_schema_version: str,
) -> PaidResourceAdmissionGrant:
    """Return an in-process grant or raise before a provider mutation."""

    blockers: list[str] = []
    if resource_class not in PAID_RESOURCE_CLASSES:
        blockers.append("paid_resource_class_invalid")
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
    return PaidResourceAdmissionGrant(
        resource_class=resource_class,
        schema_version=expected_schema_version,
        _issuer=_GRANT_ISSUER,
    )


def require_paid_resource_admission_grant(
    grant: PaidResourceAdmissionGrant | None,
    *,
    resource_class: str,
) -> None:
    """Validate the opaque in-process capability issued by the chokepoint."""

    blockers: list[str] = []
    if not isinstance(grant, PaidResourceAdmissionGrant):
        blockers.append("paid_resource_admission_grant_missing")
    else:
        if grant._issuer is not _GRANT_ISSUER:
            blockers.append("paid_resource_admission_grant_issuer_invalid")
        if grant.resource_class != resource_class:
            blockers.append("paid_resource_admission_grant_class_mismatch")
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
        normalized = sorted(set(normalized))
    return {
        "schema_version": PAID_LANE_ADMISSION_SCHEMA_VERSION,
        "status": "admitted" if not normalized else "blocked",
        "resource_class": resource_class,
        "blockers": normalized,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
    }
