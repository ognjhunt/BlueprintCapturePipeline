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

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .decision_evidence_contracts import cross_runtime_canonical_digest
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


@dataclass(frozen=True)
class TaskProfile:
    profile_id: str
    native_task_kind: str
    catalog_task_family_id: str
    runtime_adapter: str
    success_predicate: str
    aliases: tuple[str, ...]
    required_inputs: tuple[str, ...]


# First supported profiles. Each names the native adapter that compiles it and
# the success predicate the independent scorer reads.
TASK_PROFILES: tuple[TaskProfile, ...] = (
    TaskProfile(
        profile_id="rigid_pick_place",
        native_task_kind="rigid_pick_place",
        catalog_task_family_id="rigid_relocation",
        runtime_adapter="task_evaluation_rigid_relocation_native_adapter",
        success_predicate="object_pose_within_target_region",
        aliases=("rigid_pick_place", "rigid_object_pick_place", "rigid_object_relocation",
                 "rigid_relocation"),
        required_inputs=("metric_object_geometry", "target_region"),
    ),
    TaskProfile(
        profile_id="articulated_open_close",
        native_task_kind="articulated_open_close",
        catalog_task_family_id="articulated_open_close",
        runtime_adapter="task_evaluation_articulated_open_close_native_adapter",
        success_predicate="target_joint_within_qualified_interval",
        aliases=("articulated_open_close",),
        required_inputs=("qualified_articulation_graph",),
    ),
)


def task_profile_for(task_family: str | None) -> TaskProfile | None:
    value = (task_family or "").strip().lower()
    return next((row for row in TASK_PROFILES if value in row.aliases), None)


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
        missing_inputs.extend(item for item in profile.required_inputs if item not in provided_inputs)
        bodies = [row for row in EMBODIMENT_REGISTRY if not requested or row.embodiment_id in requested]
        policies: list[PolicyCandidate | HostedPolicyCandidate] = [*REGISTRY, *HOSTED_REGISTRY]
        listed: set[str] = set()
        matched_policies: set[str] = set()
        for body in bodies:
            if profile.profile_id not in body.task_profile_ids:
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
                })
        unmatched = [
            {"kind": "policy", "id": policy.candidate_id,
             "reason": "No listed robot setup declares this policy's camera and action interface."}
            for policy in policies if policy.candidate_id not in matched_policies
        ]
        if not configurations and gap is None:
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
