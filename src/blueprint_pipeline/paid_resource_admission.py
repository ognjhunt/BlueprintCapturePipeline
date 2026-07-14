"""Single fail-closed admission chokepoint for canonical paid allocators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class PaidResourceAdmissionGrant:
    resource_class: str
    schema_version: str


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
    if resource_class not in {"cpu_build", "gpu_canary", "model_volume"}:
        blockers.append("paid_resource_class_invalid")
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
    )
