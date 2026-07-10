"""Typed claim-boundary constants shared by robot-eval orchestration surfaces."""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class RobotEvalJobClaimBoundary(TypedDict):
    artifact_purpose: str
    repo_local_only_by_default: bool
    agent_operator_mode_allowed: bool
    agents_may_mutate_proof_booleans: bool
    live_provider_calls_performed: bool
    remote_asset_downloads_performed: bool
    gpu_provisioning_performed: bool
    simulators_run: bool
    gpu_training_run: bool
    messages_sent: bool
    payments_touched: bool
    deployments_performed: bool
    review_acceptance_proven: bool
    rights_privacy_scope_proven: bool
    signed_delivery_access_proven: bool
    customer_handoff_ready: bool
    delivery_access_is_deployment_approval: bool
    package_delivery_is_deployment_approval: bool
    deployment_approval_proven: bool
    physical_robot_readiness_proven: bool
    safety_validation_proven: bool
    simulator_execution_proven: bool
    rank_fidelity_result_proven: bool
    robot_policy_execution_proven: bool
    physics_contact_validated: bool
    non_ranking_operational_claim_validated: bool
    training_completed: bool
    public_claim_upgrade_allowed: bool
    disallowed_claims: List[str]
    proof_upgrade_requires: List[str]


ROBOT_EVAL_JOB_CLAIM_BOUNDARY: RobotEvalJobClaimBoundary = {
    "artifact_purpose": "robot_eval_job_orchestration_only",
    "repo_local_only_by_default": True,
    "agent_operator_mode_allowed": True,
    "agents_may_mutate_proof_booleans": False,
    "live_provider_calls_performed": False,
    "remote_asset_downloads_performed": False,
    "gpu_provisioning_performed": False,
    "simulators_run": False,
    "gpu_training_run": False,
    "messages_sent": False,
    "payments_touched": False,
    "deployments_performed": False,
    "review_acceptance_proven": False,
    "rights_privacy_scope_proven": False,
    "signed_delivery_access_proven": False,
    "customer_handoff_ready": False,
    "delivery_access_is_deployment_approval": False,
    "package_delivery_is_deployment_approval": False,
    "deployment_approval_proven": False,
    "physical_robot_readiness_proven": False,
    "safety_validation_proven": False,
    "simulator_execution_proven": False,
    "rank_fidelity_result_proven": False,
    "robot_policy_execution_proven": False,
    "physics_contact_validated": False,
    "non_ranking_operational_claim_validated": False,
    "training_completed": False,
    "public_claim_upgrade_allowed": False,
    "disallowed_claims": [
        "robot_ready",
        "deployment_ready",
        "simulator_execution_completed",
        "physics_contact_validated",
        "robot_policy_execution_passed",
        "training_completed",
        "non_ranking_operational_claim_validated",
        "public_deployment_ready",
        "deployment_approval_from_package_delivery",
        "physical_robot_readiness_from_signed_delivery",
    ],
    "proof_upgrade_requires": [
        "rights/privacy clearance for the exact use",
        "owner-system simulator load and action traces",
        "owner-system robot policy, teleoperation, or action logs",
        "training run manifest and checkpoint evidence",
        "physics/contact validation logs",
        "safety methodology and validation evidence",
        "actual outcome records",
    ],
}


def robot_eval_job_claim_boundary() -> Dict[str, Any]:
    """Return a defensive copy so artifacts cannot mutate the shared policy."""

    return dict(ROBOT_EVAL_JOB_CLAIM_BOUNDARY)


__all__ = [
    "ROBOT_EVAL_JOB_CLAIM_BOUNDARY",
    "RobotEvalJobClaimBoundary",
    "robot_eval_job_claim_boundary",
]
