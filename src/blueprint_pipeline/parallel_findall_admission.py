"""Exact-request admission for one Parallel FindAll create.

The daily research owner (``tools/daily_research/findall.py``) computes every
blocker immediately before this call from its shared paid expansion allowance and
live fences: ``allocation.problem(..., source="findall")`` against the run's frozen
grant, the brake, the reviewed release and the original deadline. Any blocker, or a
prepared request whose pricing and binding do not reproduce, refuses before a grant
exists. The grant binds that exact request's ``allocation_binding_digest``.

Standard library only. No credential, network, provider call or durable state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .paid_resource_admission import (
    PAID_LANE_ADMISSION_SCHEMA_VERSION,
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    build_paid_lane_admission,
    require_paid_resource_admission,
)
from .parallel_findall import FindAllError
from .parallel_findall_execution import PAID_RESOURCE_CLASS, prepare_submission


def admit_exact_request(
    prepared: Mapping[str, Any], *, blockers: Sequence[str]
) -> PaidResourceAdmissionGrant:
    """Issue the opaque ``parallel_findall`` grant for exactly this prepared request.

    ``blockers`` must hold every refusal the owner observed for this request; an
    empty sequence is the owner's admission. The prepared request is rebuilt from its
    own body, operation and ceiling and must match it exactly, so a changed body,
    ceiling, digest or pricing snapshot cannot borrow another request's admission.
    """
    found = [str(code) for code in blockers if str(code).strip()]
    digest = prepared.get("allocation_binding_digest") if isinstance(prepared, Mapping) else None
    try:
        rebuilt = prepare_submission(
            prepared["body_json"],
            operation_id=prepared["operation_id"],
            maximum_cost_usd=prepared["maximum_cost_usd"],
        )
    except (FindAllError, KeyError, TypeError):
        rebuilt = None
    if rebuilt is None or dict(prepared) != rebuilt or rebuilt["resource_class"] != PAID_RESOURCE_CLASS:
        found.append("parallel_findall_request_binding_invalid")
    if found:
        # Name the owner's own refusals; no admission record or grant exists.
        raise PaidResourceAdmissionBlocked(sorted(set(found)))
    admission = build_paid_lane_admission(resource_class=PAID_RESOURCE_CLASS, blockers=found)
    admission["allocation_binding_digest"] = digest
    return require_paid_resource_admission(
        admission,
        resource_class=PAID_RESOURCE_CLASS,
        expected_schema_version=PAID_LANE_ADMISSION_SCHEMA_VERSION,
    )
