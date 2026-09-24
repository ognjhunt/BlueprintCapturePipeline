"""Seal an articulated replacement candidate returned by an authoring backend.

Split from ``task_evaluation_scene_configuration_builtin_adapters`` to keep
that module inside its line budget. The rules are unchanged: one passive task
joint, per-part physics bounds equal to the configuration's, a digest-bound
receipt and graph, and no physics authority granted by authoring.
"""

from __future__ import annotations

from collections.abc import Mapping

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_adapters import (
    TaskEvaluationSceneConfigurationAdapterError,
)

#: The disclaimer each authoring backend stamps on a replacement candidate: the
#: geometry is neither observed truth nor physics authority. Same meaning, two
#: spellings, because the drivers were written at different times.
ACCEPTED_SOURCE_CANDIDATE_CLAIMS = (
    "sage_candidate_geometry_not_observed_truth_or_physics_authority",
    "source_geometry_not_observed_truth_or_physics_authority",
)


def verify_articulated_replacement_result(
    *, configuration, envelope, receipt, graph, asset, asset_record, source_candidate_record
) -> None:
    """Seal an articulated assembly candidate: one passive task joint, per-part bounds, no authority."""

    from .articulation_graph_contract import ArticulationGraphContractError, validate_articulation_graph

    identity = configuration.get("replacement_identity")
    required_output = configuration.get("required_output")
    mechanism = configuration.get("mechanism")
    completion = receipt.get("candidate_physics_completion")
    shared = {
        "static_friction": required_output.get("static_friction_bounds"),
        "dynamic_friction": required_output.get("dynamic_friction_bounds"),
        "restitution": required_output.get("restitution_bounds"),
    } if isinstance(required_output, Mapping) else {}
    expected_bounds = {
        "carcass": {**shared, "mass_kg": required_output.get("mass_kg_bounds")},
        "drawer": {**shared, "mass_kg": required_output.get("task_part_mass_kg_bounds")},
    } if isinstance(required_output, Mapping) else {}
    try:
        articulation = validate_articulation_graph(graph.get("articulation_graph") or {})
    except (ArticulationGraphContractError, ValueError, TypeError):
        articulation = None
    target_joints = [
        row for row in articulation["joints"] if row["role"] == "target"] if articulation else []
    hypothesis = configuration.get("development_geometry_hypothesis")
    plan = graph.get("assembly_plan")
    planned_hypothesis = plan.get("development_geometry_hypothesis") if isinstance(plan, Mapping) else None
    source_geometry = plan.get("source_geometry") if isinstance(plan, Mapping) else None
    source_envelope = configuration.get("metric_envelope")
    if ((hypothesis is None) != (planned_hypothesis is None)
            or (hypothesis is not None and (
                not isinstance(hypothesis, Mapping)
                or not isinstance(planned_hypothesis, Mapping)
                or any(planned_hypothesis.get(key) != value for key, value in hypothesis.items())
                or not isinstance(source_geometry, Mapping)
                or not isinstance(source_envelope, Mapping)
                or source_geometry.get("aabb_min_xyz_m") != source_envelope.get("minimum_xyz_m")
                or source_geometry.get("aabb_max_xyz_m") != source_envelope.get("maximum_xyz_m")))):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "content_agents_articulated_development_hypothesis_binding_invalid"
        )
    # A planned stroke shorter than the captured task stroke changes the task.
    # Keep this refusal distinct from a malformed authoring result so the
    # preregistered configuration can be corrected before another paid attempt.
    if (len(target_joints) == 1 and isinstance(mechanism, Mapping)
            and isinstance(mechanism.get("joint_limits"), (list, tuple))
            and len(mechanism["joint_limits"]) == 2
            and target_joints[0]["limits"] != [float(v) for v in mechanism["joint_limits"]]):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "content_agents_articulated_joint_limits_disagree_with_configuration"
        )
    if (
        identity != envelope["recipe"]["subject_identity"]
        or not isinstance(required_output, Mapping)
        or not isinstance(mechanism, Mapping)
        or required_output.get("format") != "OpenUSD"
        or required_output.get("articulation_root") is not True
        or required_output.get("single_articulation_root") is not True
        or required_output.get("task_joint_count") != 1
        or required_output.get("units") != "meters"
        or required_output.get("up_axis") != "Z"
        or configuration.get("physics_authority_granted_by_authoring") is not False
        or asset.suffix.lower() != ".usdz"
        or receipt.get("schema_version")
        != "task_evaluation_articulated_replacement_authoring_result.v1"
        or receipt.get("status") != "authored_candidate_pending_qualification"
        or receipt.get("asset_kind") != "articulated_assembly"
        or receipt.get("replacement_identity") != identity
        or receipt.get("source_candidate_digest") != source_candidate_record.get("digest")
        or receipt.get("source_candidate_claim") not in ACCEPTED_SOURCE_CANDIDATE_CLAIMS
        or receipt.get("output_usd", {}).get("sha256") != asset_record.get("digest")
        or receipt.get("output_usd", {}).get("size_bytes") != asset_record.get("size_bytes")
        or receipt.get("physics_authority_granted") is not False
        or receipt.get("result_digest") != canonical_digest(receipt, digest_field="result_digest")
        or not isinstance(completion, Mapping)
        or completion.get("schema_version")
        != "task_evaluation_articulated_candidate_physics_completion.v1"
        or completion.get("status") != "bounded_candidate_completed"
        or completion.get("asset_kind") != "articulated_assembly"
        or completion.get("physics_bounds") != expected_bounds
        or completion.get("candidate_prior_only") is not True
        or completion.get("physical_truth_claimed") is not False
        or completion.get("intra_assembly_collision_filtered") is not True
        or completion.get("completion_digest") != canonical_digest(completion, digest_field="completion_digest")
        or not str(completion.get("task_joint_prim_path") or "").startswith("/Asset/joints/")
        or not str(completion.get("fixed_base_body_prim_path") or "").startswith("/Asset/links/")
        or graph.get("schema_version") != "task_evaluation_articulated_replacement_graph.v1"
        or graph.get("asset_id") != identity.get("id")
        or graph.get("asset_version") != identity.get("version")
        or articulation is None
        or len(target_joints) != 1
        or target_joints[0]["joint_type"] != mechanism.get("joint_type")
        or target_joints[0]["limits"] != [float(v) for v in mechanism.get("joint_limits") or []]
        or target_joints[0]["drive"]["drive_type"] != "none"
        or target_joints[0]["drive"]["stiffness"] != 0.0
        or any(row["joint_type"] != "fixed" for row in articulation["joints"] if row["role"] != "target")
        or graph.get("task_joint_prim_path") != completion.get("task_joint_prim_path")
        or graph.get("fixed_base_body_prim_path") != completion.get("fixed_base_body_prim_path")
        or graph.get("physics_bounds") != expected_bounds
        or graph.get("physics_authority_granted") is not False
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "content_agents_replacement_result_invalid"
        )


__all__ = [
    "ACCEPTED_SOURCE_CANDIDATE_CLAIMS",
    "verify_articulated_replacement_result",
]
