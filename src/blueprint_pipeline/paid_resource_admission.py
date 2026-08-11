"""Single fail-closed admission chokepoint for canonical paid allocators."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Mapping


PAID_LANE_ADMISSION_SCHEMA_VERSION = "paid_lane_admission.v1"
PAID_RESOURCE_CLASSES = frozenset(
    {
        "cpu_build",
        "evaluator_api",
        "gpu_canary",
        "model_volume",
        "openai_api_candidate",
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


def _binding_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PaidResourceAdmissionGrant:
    resource_class: str
    schema_version: str
    _issuer: object = field(repr=False, compare=False)
    allocation_binding_digest: str | None = None
    allowed_active_instance_ids: tuple[int, ...] = ()


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
    allocation_binding = admission.get("allocation_binding")
    binding: dict[str, Any] = {}
    recomputed_binding_digest: str | None = None
    if "allocation_binding" in admission:
        if not isinstance(allocation_binding, Mapping):
            blockers.append("paid_resource_allocation_binding_invalid")
        else:
            binding = dict(allocation_binding)
            try:
                recomputed_binding_digest = _binding_digest(binding)
            except (TypeError, ValueError):
                blockers.append("paid_resource_allocation_binding_not_canonical_json")
            if recomputed_binding_digest != admission.get("allocation_binding_digest"):
                blockers.append("paid_resource_allocation_binding_digest_mismatch")
    raw_allowed_ids = binding.get("allowed_active_vast_instance_ids", ())
    allowed_ids: tuple[int, ...] = ()
    if raw_allowed_ids not in (None, ()):
        if (
            not isinstance(raw_allowed_ids, Sequence)
            or isinstance(raw_allowed_ids, (str, bytes))
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in raw_allowed_ids
            )
        ):
            blockers.append("paid_resource_allowed_active_instance_ids_invalid")
        else:
            allowed_ids = tuple(sorted(set(raw_allowed_ids)))
            if len(allowed_ids) != len(raw_allowed_ids):
                blockers.append("paid_resource_allowed_active_instance_ids_not_unique")
    if blockers:
        raise PaidResourceAdmissionBlocked(sorted(set(blockers)))
    return PaidResourceAdmissionGrant(
        resource_class=resource_class,
        schema_version=expected_schema_version,
        allocation_binding_digest=(
            recomputed_binding_digest
            if recomputed_binding_digest is not None
            else (
                str(admission.get("allocation_binding_digest"))
                if admission.get("allocation_binding_digest") is not None
                else None
            )
        ),
        allowed_active_instance_ids=allowed_ids,
        _issuer=_GRANT_ISSUER,
    )


def require_paid_resource_admission_grant(
    grant: PaidResourceAdmissionGrant | None,
    *,
    resource_class: str,
    allocation_binding_digest: str | None = None,
    require_allocation_binding: bool = False,
    allowed_active_instance_ids: Sequence[int] | None = None,
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
        if require_allocation_binding and not grant.allocation_binding_digest:
            blockers.append("paid_resource_admission_grant_binding_missing")
        if (
            allocation_binding_digest is not None
            and grant.allocation_binding_digest != allocation_binding_digest
        ):
            blockers.append("paid_resource_admission_grant_binding_mismatch")
        if allowed_active_instance_ids is not None:
            supplied_ids: tuple[int, ...] = ()
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in allowed_active_instance_ids
            ):
                blockers.append("paid_resource_allowed_active_instance_ids_invalid")
            else:
                supplied_ids = tuple(sorted(set(allowed_active_instance_ids)))
                if len(supplied_ids) != len(allowed_active_instance_ids):
                    blockers.append("paid_resource_allowed_active_instance_ids_not_unique")
            if grant.allowed_active_instance_ids != supplied_ids:
                blockers.append("paid_resource_admission_grant_active_instances_mismatch")
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
