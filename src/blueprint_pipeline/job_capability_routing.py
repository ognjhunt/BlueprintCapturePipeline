"""Route one job's task to the complete configurations Blueprint can run.

Task requirements, embodiment, policy interface and runtime are separate
facts. A job is routed by its declared task family against explicit profiles;
a family without a native runtime adapter and independent scorer is reported
as a capability gap rather than coerced into pick/place. Unavailable robots
and policies stay listed with their exact reason; an announcement or a vendor
name never makes one runnable.

Adding a site that uses a supported profile needs only its capture and task
data. Adding a task family or embodiment needs its adapter, reset/scoring and
evidence work, then one entry here; nothing else in the job path changes.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .task_profile_contracts import TASK_PROFILES, TaskProfile as TaskProfile, task_profile_for
from .task_evaluation_embodiment_registry import EMBODIMENT_REGISTRY, EmbodimentProfile
from .task_evaluation_policy_candidate_registry import (
    HOSTED_REGISTRY,
    REGISTRY,
    HostedPolicyCandidate,
    PolicyCandidate,
)

ROUTING_SCHEMA = "job_capability_routing.v1"
RoutingStatus = Literal["supported", "capability_gap", "needs_input"]
EMBODIMENT_PROFILES = EMBODIMENT_REGISTRY



def _policy_row(policy: PolicyCandidate | HostedPolicyCandidate) -> dict[str, Any]:
    if isinstance(policy, HostedPolicyCandidate):
        return {
            "candidate_id": policy.candidate_id, "display_name": policy.display_name,
            "kind": "hosted_api", "provider": policy.provider, "model_id": policy.model_id,
            "version_identity": policy.version_identity, "version_frozen": policy.version_frozen,
            "adapter_id": policy.adapter_id, "terms_status": policy.terms_status,
            "sends_site_images_to_provider": policy.sends_site_images_to_provider,
        }
    return {
        "candidate_id": policy.candidate_id, "display_name": policy.display_name,
        "kind": "checkpoint",
        "checkpoint_uri": policy.checkpoint_uri, "checkpoint_revision": policy.checkpoint_revision,
        "checkpoint_digest": policy.checkpoint_digest, "adapter_id": policy.adapter_id,
        "license_id": policy.license_id, "commercial_use": policy.commercial_use,
        "minimum_gpu_memory_gb": policy.minimum_gpu_memory_gb,
    }


def _actions(policy: PolicyCandidate | HostedPolicyCandidate) -> tuple[str, ...]:
    return policy.action_schema_ids if isinstance(policy, HostedPolicyCandidate) else (policy.action_schema_id,)


def _matches(policy: PolicyCandidate | HostedPolicyCandidate, body: EmbodimentProfile) -> str | None:
    """The shared action schema when the policy's interface fits this body's declared one."""

    if policy.observation_schema_id not in body.observation_schema_ids:
        return None
    return next((item for item in _actions(policy) if item in body.action_schema_ids), None)


def route_job_capabilities(
    task_family: str | None,
    *,
    embodiment_ids: Sequence[str] = (),
    provided_inputs: Sequence[str] = (),
    native_runtime_contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return supported complete configurations, or the exact reason none can run."""

    requested = tuple(dict.fromkeys(embodiment_ids))
    known = {row.embodiment_id for row in EMBODIMENT_REGISTRY}
    unavailable: list[dict[str, Any]] = [
        {"kind": "embodiment", "id": item, "reason": "No embodiment profile is registered for this robot."}
        for item in requested if item not in known
    ]
    unmatched: list[dict[str, Any]] = []
    profile = task_profile_for(task_family)
    configurations: list[dict[str, Any]] = []
    missing_inputs: list[str] = []
    gap: dict[str, Any] | None = None
    if not (task_family or "").strip():
        missing_inputs.append("task_family")
    elif profile is None:
        gap = {
            "task_family": task_family,
            "missing": ["native_task_runtime_adapter", "independent_task_scorer", "reset_contract"],
            "reason": "No supported task profile compiles and independently scores this task family yet.",
        }
    else:
        from .task_family_scoring import canonical_task_family
        family = canonical_task_family(task_family)
        family_runtimes = {}
        if family is not None:
            from .task_family_native_contracts import validate_task_family_native_runtime_contract
            for embodiment_id, raw in (native_runtime_contracts or {}).items():
                runtime = validate_task_family_native_runtime_contract(raw)
                if runtime["task_spec"]["task_kind"] != family:
                    raise ValueError("job_capability_native_task_family_mismatch")
                if runtime["robot_configuration"].get("embodiment_id") != embodiment_id:
                    raise ValueError("job_capability_native_embodiment_mismatch")
                family_runtimes[embodiment_id] = runtime
        supplied = set(provided_inputs)
        if family_runtimes:
            supplied.update(profile.required_inputs)
        missing_inputs.extend(item for item in profile.required_inputs if item not in supplied)
        bodies = [row for row in EMBODIMENT_REGISTRY if not requested or row.embodiment_id in requested]
        policies: list[PolicyCandidate | HostedPolicyCandidate] = [*REGISTRY, *HOSTED_REGISTRY]
        listed: set[str] = set()
        matched_policies: set[str] = set()
        for body in bodies:
            family_runtime = family_runtimes.get(body.embodiment_id)
            if family is not None and family_runtime is None:
                continue
            if family is None and profile.profile_id not in body.task_profile_ids:
                if requested:
                    unavailable.append({"kind": "embodiment", "id": body.embodiment_id,
                                        "reason": body.reason or f"No {profile.profile_id} adapter is admitted for this robot."})
                continue
            if not body.available:
                unavailable.append({"kind": "embodiment", "id": body.embodiment_id, "reason": body.reason,
                                    "missing": list(body.missing)})
            for policy in policies:
                action = _matches(policy, body)
                if action is None:
                    continue
                matched_policies.add(policy.candidate_id)
                if not policy.runnable:
                    if policy.candidate_id not in listed:
                        listed.add(policy.candidate_id)
                        unavailable.append({"kind": "policy", "id": policy.candidate_id,
                                            "reason": policy.reason or policy.status,
                                            "blockers": list(policy.blockers)})
                    continue
                if not body.available:
                    continue
                if family_runtime is not None:
                    configured = family_runtime["robot_configuration"]
                    if (configured.get("observation_schema_id") != policy.observation_schema_id
                            or configured.get("action_schema_id") != action):
                        continue
                configurations.append({
                    "task_profile_id": profile.profile_id,
                    "native_task_kind": profile.native_task_kind,
                    "runtime_adapter": profile.runtime_adapter,
                    "success_predicate": profile.success_predicate,
                    "embodiment_id": body.embodiment_id, "robot": body.robot, "tool": body.tool,
                    "observation_schema_id": policy.observation_schema_id,
                    "action_schema_id": action,
                    "controller": body.controller, "simulator_runtime_id": body.simulator_runtime_id,
                    "sensor_reference": body.sensor_reference,
                    "policy": _policy_row(policy),
                    **({"native_runtime_contract_digest": family_runtime["contract_digest"],
                        "task_success_contract_digest": family_runtime["task_spec"]["task_success_contract"]["contract_digest"]}
                       if family_runtime is not None else {}),
                })
        unmatched = [
            {"kind": "policy", "id": policy.candidate_id,
             "reason": "No listed robot setup declares this policy's camera and action interface."}
            for policy in policies if policy.candidate_id not in matched_policies
        ]
        if not configurations and gap is None and not (family is not None and missing_inputs):
            gap = {"task_family": task_family, "missing": ["runnable_configuration"],
                   "reason": "No available robot and runnable policy pair supports this task profile."}
    status: RoutingStatus = (
        "capability_gap" if gap else "needs_input" if missing_inputs else "supported"
    )
    value = {
        "schema_version": ROUTING_SCHEMA, "task_family": task_family,
        "task_profile_id": profile.profile_id if profile else None, "status": status,
        "configurations": configurations, "unavailable": unavailable,
        "unmatched_policies": unmatched,
        "missing_inputs": missing_inputs, "capability_gap": gap,
        "claim_boundary": "Routing lists what Blueprint can run; it is not a performance, delivery or physical-capability result.",
    }
    # Shortlist options retain their exact identity and setup requirements even
    # when no runnable configuration exists. They never enter configurations.
    from .deployment_embodiment_support import support_catalog

    support = support_catalog()
    value["support_options"] = support
    value["routing_digest"] = cross_runtime_canonical_digest(value, digest_field="routing_digest")
    return value


def testbed_task_family(testbed: Mapping[str, Any]) -> str | None:
    family = (testbed.get("task_distribution") or {}).get("task_family")
    return family if isinstance(family, str) else None


def routing_limitations(routing: Mapping[str, Any]) -> list[str]:
    gap = routing.get("capability_gap") or {}
    rows = [f"Capability gap for task family '{gap.get('task_family')}': {gap.get('reason')} Missing: {', '.join(gap.get('missing', []))}."] if gap else []
    rows += [f"Unavailable {row['kind']} {row['id']}: {row['reason']}" for row in routing.get("unavailable", [])]
    return rows


__all__ = [
    "EMBODIMENT_PROFILES", "ROUTING_SCHEMA", "TASK_PROFILES", "route_job_capabilities",
    "routing_limitations", "task_profile_for", "testbed_task_family",
]
