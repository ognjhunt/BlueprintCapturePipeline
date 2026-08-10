"""Concurrency contract for the general evidence-plan executor.

Pins the harness parallelization decision: steps for different claims are
independent and may overlap when concurrency is requested; within one claim
the conditional-escalation gate behaves exactly as in serial execution; paid
steps (positive ``expected_cost_usd``) never overlap without explicit
authorization; adapter failures propagate unchanged; and results plus the
execution manifest stay byte-identical between serial and concurrent runs for
the same adapter outputs.
"""

from __future__ import annotations

import json
import threading

import pytest

from blueprint_pipeline.decision_evidence_contracts import DecisionEvidenceRequest
from blueprint_pipeline.decision_evidence_execution import (
    EvidenceMethodAdapterRegistry,
    execute_evidence_plan,
)
from blueprint_pipeline.decision_evidence_router import route_decision_evidence
from tests.test_decision_evidence_router import (
    SHA_D,
    _claim,
    _profile,
    _qualification,
    _registry,
    _request,
    _testbed,
    _FixtureAdapter,
)

_WAIT_SECONDS = 5.0
_NO_OVERLAP_PROBE_SECONDS = 0.3


def _two_claim_setup(*, primary_cost: float, escalation: bool):
    testbed = _testbed()
    request = _request(testbed)
    request.pop("request_digest")
    request["claims"] = [
        _claim("reach-a", "reachability"),
        _claim("reach-b", "reachability"),
    ]
    request = DecisionEvidenceRequest.from_mapping(request).to_mapping()
    profiles = [
        _profile(
            "primary-reach",
            "analytic_geometry_kinematics",
            ["reachability"],
            required_inputs=["metric_geometry"],
            authority=1,
            cost=primary_cost,
            latency=1,
            correlation_group="primary-geometry",
        )
    ]
    if escalation:
        profiles.append(
            _profile(
                "stronger-reach",
                "analytic_geometry_kinematics",
                ["reachability"],
                required_inputs=["metric_geometry"],
                authority=2,
                cost=max(primary_cost, 0.5),
                latency=2,
                correlation_group="independent-geometry",
            )
        )
    qualifications = [_qualification(profile, "reachability") for profile in profiles]
    plan = route_decision_evidence(request, testbed, profiles, qualifications).to_mapping()
    return testbed, request, profiles, qualifications, plan


_VALID_RAW = {
    "status": "valid",
    "supports_claim": True,
    "uncertainty": 0.01,
    "coverage": 0.95,
    "blockers": [],
    "invalid_rollout_reasons": [],
    "raw_artifact_references": [{"uri": "fixture://valid", "digest": SHA_D}],
    "provenance": {"fixture": True},
    "false_safe_risk": 0.01,
}

_INVALID_RAW = {
    "status": "invalid",
    "supports_claim": None,
    "uncertainty": 1.0,
    "coverage": 0.0,
    "blockers": ["invalid_geometry_artifact"],
    "invalid_rollout_reasons": ["checksum_mismatch"],
    "raw_artifact_references": [],
    "provenance": {"fixture": True},
    "false_safe_risk": 1.0,
}


class _ClaimRoutedAdapter:
    """Deterministic adapter returning a per-claim canned result."""

    def __init__(self, adapter_reference: str, by_claim: dict):
        self.adapter_reference = adapter_reference
        self._by_claim = by_claim

    def execute(self, **kwargs):
        claim_id = str(kwargs["claim"]["claim_id"])
        result = self._by_claim[claim_id]
        if isinstance(result, BaseException):
            raise result
        return {
            **result,
            "applicability_envelope": kwargs["method_profile"]["applicability_envelope"],
        }


class _RendezvousAdapter:
    """Adapter that proves both claims' steps were in flight simultaneously."""

    def __init__(self, adapter_reference: str, barrier: threading.Barrier):
        self.adapter_reference = adapter_reference
        self._barrier = barrier

    def execute(self, **kwargs):
        self._barrier.wait()
        return {
            **_VALID_RAW,
            "applicability_envelope": kwargs["method_profile"]["applicability_envelope"],
        }


class _OverlapProbeAdapter:
    """Adapter that records whether the two paid steps overlapped."""

    def __init__(self, adapter_reference: str):
        self.adapter_reference = adapter_reference
        self.first_saw_second = None
        self._second_started = threading.Event()

    def execute(self, **kwargs):
        claim_id = str(kwargs["claim"]["claim_id"])
        if claim_id == "reach-a":
            self.first_saw_second = self._second_started.wait(_NO_OVERLAP_PROBE_SECONDS)
        else:
            self._second_started.set()
        return {
            **_VALID_RAW,
            "applicability_envelope": kwargs["method_profile"]["applicability_envelope"],
        }


def _strip_concurrency_evidence(manifest: dict) -> dict:
    stripped = dict(manifest)
    stripped.pop("execution_concurrency", None)
    return stripped


def test_concurrent_execution_matches_serial_byte_for_byte() -> None:
    testbed, request, profiles, qualifications, plan = _two_claim_setup(
        primary_cost=0.1, escalation=True
    )
    primary, escalation = profiles

    def _registry_for_run() -> EvidenceMethodAdapterRegistry:
        return EvidenceMethodAdapterRegistry(
            [
                _ClaimRoutedAdapter(
                    primary["adapter_reference"],
                    {"reach-a": _INVALID_RAW, "reach-b": _VALID_RAW},
                ),
                _ClaimRoutedAdapter(
                    escalation["adapter_reference"],
                    {"reach-a": _VALID_RAW, "reach-b": _VALID_RAW},
                ),
            ]
        )

    serial = execute_evidence_plan(
        plan, request, testbed, profiles, qualifications, registry=_registry_for_run()
    )
    concurrent = execute_evidence_plan(
        plan,
        request,
        testbed,
        profiles,
        qualifications,
        registry=_registry_for_run(),
        max_concurrency=4,
        paid_concurrency_authorized=True,
    )

    serial_results = [result.to_mapping() for result in serial.results]
    concurrent_results = [result.to_mapping() for result in concurrent.results]
    assert json.dumps(serial_results, sort_keys=True) == json.dumps(
        concurrent_results, sort_keys=True
    )
    assert json.dumps(dict(serial.execution_manifest), sort_keys=True) == json.dumps(
        _strip_concurrency_evidence(dict(concurrent.execution_manifest)), sort_keys=True
    )
    concurrency_evidence = concurrent.execution_manifest["execution_concurrency"]
    assert concurrency_evidence["max_concurrency"] == 4
    assert concurrency_evidence["paid_concurrency_authorized"] is True

    # The escalation gate behaved exactly as in serial execution: reach-a's
    # invalid primary escalated, reach-b's sufficient primary skipped.
    statuses = {
        row["step_id"]: row["status"] for row in concurrent.execution_manifest["steps"]
    }
    assert sorted(statuses.values()) == sorted(
        ["invalid", "valid", "valid", "skipped_evidence_already_sufficient"]
    )
    skipped = [
        row
        for row in concurrent.execution_manifest["steps"]
        if row["status"] == "skipped_evidence_already_sufficient"
    ]
    assert len(skipped) == 1
    assert "reach-b" in skipped[0]["step_id"]


def test_cross_claim_steps_overlap_when_concurrency_requested() -> None:
    testbed, request, profiles, qualifications, plan = _two_claim_setup(
        primary_cost=0.0, escalation=False
    )
    barrier = threading.Barrier(2, timeout=_WAIT_SECONDS)
    execution = execute_evidence_plan(
        plan,
        request,
        testbed,
        profiles,
        qualifications,
        registry=EvidenceMethodAdapterRegistry(
            [_RendezvousAdapter(profiles[0]["adapter_reference"], barrier)]
        ),
        max_concurrency=2,
    )
    assert [result.to_mapping()["status"] for result in execution.results] == [
        "valid",
        "valid",
    ]
    assert execution.execution_manifest["execution_concurrency"]["observed_max_overlap"] == 2


def test_paid_steps_never_overlap_without_explicit_authorization() -> None:
    testbed, request, profiles, qualifications, plan = _two_claim_setup(
        primary_cost=0.1, escalation=False
    )
    probe = _OverlapProbeAdapter(profiles[0]["adapter_reference"])
    execution = execute_evidence_plan(
        plan,
        request,
        testbed,
        profiles,
        qualifications,
        registry=EvidenceMethodAdapterRegistry([probe]),
        max_concurrency=4,
    )
    assert probe.first_saw_second is False
    assert [result.to_mapping()["status"] for result in execution.results] == [
        "valid",
        "valid",
    ]


def test_paid_steps_overlap_only_with_explicit_authorization() -> None:
    testbed, request, profiles, qualifications, plan = _two_claim_setup(
        primary_cost=0.1, escalation=False
    )
    barrier = threading.Barrier(2, timeout=_WAIT_SECONDS)
    execution = execute_evidence_plan(
        plan,
        request,
        testbed,
        profiles,
        qualifications,
        registry=EvidenceMethodAdapterRegistry(
            [_RendezvousAdapter(profiles[0]["adapter_reference"], barrier)]
        ),
        max_concurrency=2,
        paid_concurrency_authorized=True,
    )
    assert execution.execution_manifest["execution_concurrency"]["observed_max_overlap"] == 2


@pytest.mark.parametrize("max_concurrency", [1, 4])
def test_adapter_exception_propagates_unchanged(max_concurrency: int) -> None:
    testbed, request, profiles, qualifications, plan = _two_claim_setup(
        primary_cost=0.0, escalation=False
    )
    error = RuntimeError("adapter_boom")
    registry = EvidenceMethodAdapterRegistry(
        [
            _ClaimRoutedAdapter(
                profiles[0]["adapter_reference"],
                {"reach-a": error, "reach-b": _VALID_RAW},
            )
        ]
    )
    with pytest.raises(RuntimeError, match="adapter_boom"):
        execute_evidence_plan(
            plan,
            request,
            testbed,
            profiles,
            qualifications,
            registry=registry,
            max_concurrency=max_concurrency,
        )


def test_full_registry_plan_serial_and_concurrent_agree() -> None:
    testbed = _testbed()
    request = _request(testbed)
    profiles, qualifications = _registry()
    plan = route_decision_evidence(request, testbed, profiles, qualifications).to_mapping()

    def _adapters() -> EvidenceMethodAdapterRegistry:
        rows = []
        for profile in profiles:
            if profile["method_id"] in {
                "analytic-reach",
                "captured-visibility",
                "fixture-mujoco",
            }:
                rows.append(
                    _FixtureAdapter(
                        profile["adapter_reference"],
                        supports_claim=True,
                        finding=f"{profile['method_id']}:supports",
                    )
                )
        return EvidenceMethodAdapterRegistry(rows)

    serial = execute_evidence_plan(
        plan,
        request,
        testbed,
        profiles,
        qualifications,
        registry=_adapters(),
        context={"ephemeral_fixture_root": "/tmp/not-persisted"},
    )
    concurrent = execute_evidence_plan(
        plan,
        request,
        testbed,
        profiles,
        qualifications,
        registry=_adapters(),
        context={"ephemeral_fixture_root": "/tmp/not-persisted"},
        max_concurrency=3,
    )
    assert json.dumps(
        [result.to_mapping() for result in serial.results], sort_keys=True
    ) == json.dumps([result.to_mapping() for result in concurrent.results], sort_keys=True)
    assert json.dumps(dict(serial.execution_manifest), sort_keys=True) == json.dumps(
        _strip_concurrency_evidence(dict(concurrent.execution_manifest)), sort_keys=True
    )
