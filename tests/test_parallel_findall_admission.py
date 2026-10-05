"""The exact-request parallel_findall issuer: owner blockers decide; nothing else mints a grant."""
import importlib.util
import json
from pathlib import Path

import pytest

from blueprint_pipeline import parallel_findall_admission as issuer
from blueprint_pipeline import parallel_findall_execution as execution
from blueprint_pipeline.paid_resource_admission import (
    PaidResourceAdmissionBlocked,
    PaidResourceAdmissionGrant,
    require_paid_resource_admission_grant,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "docs/examples/parallel_findall_spec.json").read_text())
OPERATION = "blueprint-researcher:2026-10-04:findall:call_fixture"


def prepared(**overrides):
    return execution.prepare_submission(SPEC, **{"operation_id": OPERATION, "maximum_cost_usd": "1.00", **overrides})


def test_no_blocker_issues_a_grant_bound_to_exactly_this_request():
    request = prepared()
    grant = issuer.admit_exact_request(request, blockers=[])
    assert isinstance(grant, PaidResourceAdmissionGrant)
    require_paid_resource_admission_grant(
        grant, resource_class="parallel_findall", allocation_binding_digest=request["allocation_binding_digest"],
        require_allocation_binding=True)
    with pytest.raises(PaidResourceAdmissionBlocked, match="binding_mismatch"):
        require_paid_resource_admission_grant(
            grant, resource_class="parallel_findall",
            allocation_binding_digest=prepared(maximum_cost_usd="2.00")["allocation_binding_digest"],
            require_allocation_binding=True)


@pytest.mark.parametrize("blockers", [["paid_expansion_disabled"], ["paid_expansion_cap_exceeds_remaining",
                                                                    "findall_stopped"]])
def test_every_owner_blocker_is_named_and_refuses(blockers):
    with pytest.raises(PaidResourceAdmissionBlocked) as refused:
        issuer.admit_exact_request(prepared(), blockers=blockers)
    assert refused.value.blockers == sorted(blockers)


@pytest.mark.parametrize("change", [
    {"maximum_cost_usd": "9"}, {"operation_id": "another"}, {"allocation_binding_digest": "sha256:" + "0" * 64},
    {"pricing_version": "parallel-findall-1999-01-01"}, {"resource_class": "gpu_canary"},
])
def test_a_request_whose_binding_does_not_reproduce_cannot_borrow_admission(change):
    # Rebuilding from the request's own body, operation and ceiling exposes any changed field.
    request = {**prepared(), **change}
    with pytest.raises(PaidResourceAdmissionBlocked, match="parallel_findall_request_binding_invalid"):
        issuer.admit_exact_request(request, blockers=[])


@pytest.mark.parametrize("value", [None, [], "prepared", {"body_json": {}}])
def test_malformed_requests_refuse_without_raising_other_errors(value):
    with pytest.raises(PaidResourceAdmissionBlocked, match="parallel_findall_request_binding_invalid"):
        issuer.admit_exact_request(value, blockers=[])


def test_issuer_is_a_registered_canonical_admission_surface_and_the_adapters_still_cannot_issue():
    spec = importlib.util.spec_from_file_location("verify_paid_resource_allocator",
                                                  ROOT / "scripts/verify_paid_resource_allocator.py")
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    path = "src/blueprint_pipeline/parallel_findall_admission.py"
    manifest = json.loads(verifier.MUTATION_SURFACE_MANIFEST.read_text())
    assert path in verifier.APPROVED_ADMISSION_ISSUERS and path in verifier.APPROVED_LANE_ADMISSION_BUILDERS
    assert path in manifest["issuer_allowlist"]["require_paid_resource_admission"]
    assert path in manifest["issuer_allowlist"]["build_paid_lane_admission"]
    assert verifier._direct_paid_mutation_signals((ROOT / path).read_text()) == set()
    for adapter in ("src/blueprint_pipeline/parallel_findall_execution.py", "src/blueprint_pipeline/parallel_findall_owner.py"):
        calls = verifier._all_calls(ROOT / adapter)
        assert not calls & {"require_paid_resource_admission", "build_paid_lane_admission"}
