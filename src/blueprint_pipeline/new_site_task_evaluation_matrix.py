"""Frozen five-policy by scenario Task Evaluation Run contracts and executor.

The v2 path extends the single-reset v1 compiler without changing its reader.
Scenario generation is non-authorizing: every perturbation, condition, and
variant entering an admitted scenario must already be bounded by a digest-bound
qualification or evidence ceiling.  Execution packets preserve the full grid,
including failed and missing cells, and aggregation uses only paired scenarios.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .new_site_task_evaluation_run import (
    EXPECTED_POLICY_CANDIDATES,
    METRIC_SCHEMA_VERSION,
    NewSiteTaskEvaluationError,
    _clone,
    _expected_robot,
    _is_digest,
    _metric_spec,
    _parse_timestamp,
    _placement_gate,
    _policy_candidates,
    _reconstruction_gate,
    _route_gate,
    _scene_composition_gate,
    _source_gate,
    _target_gate,
    _validate_canonical_artifact,
)


REQUEST_SCHEMA_VERSION_V2 = "new_site_task_evaluation_request.v2"
RESULT_SCHEMA_VERSION_V2 = "new_site_task_evaluation_run.v2"
SCENARIO_PACK_SCHEMA_VERSION = "new_site_task_scenario_pack.v1"
SCENARIO_SCHEMA_VERSION = "new_site_task_scenario.v1"
AGGREGATION_SCHEMA_VERSION = "paired_scenario_aggregation_rule.v1"
AUTHORIZATION_SCHEMA_VERSION_V2 = "new_site_policy_execution_authorization.v2"
EXECUTION_PLAN_SCHEMA_VERSION = "new_site_policy_scenario_execution_plan.v1"
EXECUTION_PACKET_SCHEMA_VERSION = "new_site_policy_scenario_execution_packet.v1"
MATRIX_ATTEMPT_SCHEMA_VERSION = "learned_policy_scenario_attempt_receipt.v2"
MIGRATION_SCHEMA_VERSION = "new_site_task_evaluation_v1_to_v2_migration.v1"

MIN_INSPECTION_SCENARIOS = 3
MAX_INSPECTION_SCENARIOS = 5
_SCENARIO_KINDS = {
    "nominal",
    "bounded_placement_observation_perturbation",
    "visibility_occlusion_stress",
}
_APPLIED_PERTURBATION_STATUS = "bounded_qualified"
_NON_APPLIED_PERTURBATION_STATUS = "not_applied"
_CELL_TERMINAL_STATUSES = {"completed", "failed", "missing", "abstained_pre_execution"}
_LEARNED_ACTION_SOURCES = {"learned_policy", "policy_endpoint", "vla_policy"}


def _require_nonempty(value: Any, *, code: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise NewSiteTaskEvaluationError([code])
    return text


def _require_digest(value: Any, *, code: str) -> str:
    if not _is_digest(value):
        raise NewSiteTaskEvaluationError([code])
    return str(value)


def _scenario_cell_id(candidate_id: str, scenario_id: str) -> str:
    return f"{candidate_id}::{scenario_id}"


def _scenario_bindings(
    *,
    site_id: str,
    task_id: str,
    source_profile_digest: str,
    reconstruction_digest: str,
    target_binding_digest: str,
    robot_id: str,
    placement_digest: str,
    task_class: str,
) -> dict[str, str]:
    return {
        "site_id": site_id,
        "task_id": task_id,
        "source_profile_digest": source_profile_digest,
        "reconstruction_digest": reconstruction_digest,
        "target_binding_digest": target_binding_digest,
        "robot_id": robot_id,
        "placement_digest": placement_digest,
        "task_class": task_class,
    }


def _validate_perturbation(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NewSiteTaskEvaluationError([f"scenario_{label}_invalid"])
    perturbation = _clone(dict(value))
    status = perturbation.get("status")
    if status not in {_NON_APPLIED_PERTURBATION_STATUS, _APPLIED_PERTURBATION_STATUS}:
        raise NewSiteTaskEvaluationError([f"scenario_{label}_status_invalid"])
    if status == _NON_APPLIED_PERTURBATION_STATUS:
        if perturbation.get("translation_meters") not in (None, [0.0, 0.0, 0.0]):
            raise NewSiteTaskEvaluationError([f"scenario_{label}_not_applied_nonzero"])
        if perturbation.get("rotation_degrees") not in (None, [0.0, 0.0, 0.0]):
            raise NewSiteTaskEvaluationError([f"scenario_{label}_not_applied_nonzero"])
        return perturbation
    for field in ("qualification_digest", "evidence_ceiling_digest"):
        _require_digest(
            perturbation.get(field), code=f"scenario_{label}_{field}_invalid"
        )
    if perturbation.get("task_valid") is not True:
        raise NewSiteTaskEvaluationError([f"scenario_{label}_task_validity_unqualified"])
    for field in ("translation_meters", "rotation_degrees"):
        vector = perturbation.get(field)
        if (
            not isinstance(vector, list)
            or len(vector) != 3
            or any(
                isinstance(row, bool)
                or not isinstance(row, (int, float))
                or not math.isfinite(float(row))
                for row in vector
            )
        ):
            raise NewSiteTaskEvaluationError([f"scenario_{label}_{field}_invalid"])
    bound = perturbation.get("maximum_norm")
    if (
        isinstance(bound, bool)
        or not isinstance(bound, (int, float))
        or not math.isfinite(float(bound))
        or float(bound) <= 0.0
    ):
        raise NewSiteTaskEvaluationError([f"scenario_{label}_maximum_norm_invalid"])
    return perturbation


def _validate_observation_settings(value: Any) -> dict[str, Any]:
    settings = _validate_canonical_artifact(
        value,
        label="scenario_observation_settings",
        digest_field="settings_digest",
    )
    for field in ("lighting", "sensor", "noise"):
        if not isinstance(settings.get(field), Mapping):
            raise NewSiteTaskEvaluationError(
                [f"scenario_observation_settings_{field}_invalid"]
            )
    _require_digest(
        settings.get("evidence_ceiling_digest"),
        code="scenario_observation_settings_evidence_ceiling_invalid",
    )
    if settings.get("settings_may_authorize_new_claims") is not False:
        raise NewSiteTaskEvaluationError(
            ["scenario_observation_settings_self_authorization_forbidden"]
        )
    return settings


def _validate_variants(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise NewSiteTaskEvaluationError(["scenario_geometry_material_variants_invalid"])
    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise NewSiteTaskEvaluationError(
                [f"scenario_geometry_material_variant_{index}_invalid"]
            )
        variant = _clone(dict(raw))
        variant_id = _require_nonempty(
            variant.get("variant_id"),
            code="scenario_geometry_material_variant_id_missing",
        )
        if variant_id in seen:
            raise NewSiteTaskEvaluationError(
                ["scenario_geometry_material_variant_id_duplicate"]
            )
        if variant.get("variant_kind") not in {"geometry", "material"}:
            raise NewSiteTaskEvaluationError(
                ["scenario_geometry_material_variant_kind_invalid"]
            )
        if variant.get("status") != "qualified_within_evidence_ceiling":
            raise NewSiteTaskEvaluationError(
                ["scenario_geometry_material_variant_unqualified"]
            )
        for field in ("asset_digest", "qualification_digest", "evidence_ceiling_digest"):
            _require_digest(
                variant.get(field),
                code=f"scenario_geometry_material_variant_{field}_invalid",
            )
        if variant.get("variant_may_authorize_new_claims") is not False:
            raise NewSiteTaskEvaluationError(
                ["scenario_geometry_material_variant_self_authorization_forbidden"]
            )
        seen.add(variant_id)
        variants.append(variant)
    return variants


def _validate_scenario(
    value: Any,
    *,
    pack_id: str,
    expected_bindings: Mapping[str, str],
    metric_spec_digest: str,
) -> dict[str, Any]:
    scenario = _validate_canonical_artifact(
        value,
        label="scenario",
        digest_field="scenario_digest",
        accepted_schemas={SCENARIO_SCHEMA_VERSION},
    )
    _require_nonempty(scenario.get("scenario_id"), code="scenario_id_missing")
    if scenario.get("scenario_pack_id") != pack_id:
        raise NewSiteTaskEvaluationError(["scenario_pack_binding_mismatch"])
    if scenario.get("scenario_kind") not in _SCENARIO_KINDS:
        raise NewSiteTaskEvaluationError(["scenario_kind_invalid"])
    if scenario.get("admission_status") != "admitted":
        raise NewSiteTaskEvaluationError(["scenario_definition_not_admitted"])
    if scenario.get("frozen_before_execution") is not True:
        raise NewSiteTaskEvaluationError(["scenario_not_frozen_before_execution"])
    _parse_timestamp(
        scenario.get("frozen_at"), code="scenario_frozen_at_invalid"
    )
    if scenario.get("bindings") != expected_bindings:
        raise NewSiteTaskEvaluationError(["scenario_scope_binding_mismatch"])
    if scenario.get("metric_spec_digest") != metric_spec_digest:
        raise NewSiteTaskEvaluationError(["scenario_metric_binding_mismatch"])
    _require_digest(
        scenario.get("reset_state_digest"), code="scenario_reset_state_digest_invalid"
    )
    _require_digest(
        scenario.get("initial_state_observation_digest"),
        code="scenario_initial_state_observation_digest_invalid",
    )
    seed = scenario.get("deterministic_simulator_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise NewSiteTaskEvaluationError(["scenario_simulator_seed_invalid"])
    for field in ("target_state", "distractor_state"):
        state = scenario.get(field)
        if not isinstance(state, Mapping):
            raise NewSiteTaskEvaluationError([f"scenario_{field}_invalid"])
        _require_digest(
            state.get("state_digest"), code=f"scenario_{field}_digest_invalid"
        )
        if state.get("policy_visibility") not in {"observed", "not_directly_observed"}:
            raise NewSiteTaskEvaluationError(
                [f"scenario_{field}_policy_visibility_invalid"]
            )
    perturbations = scenario.get("perturbations")
    if not isinstance(perturbations, Mapping):
        raise NewSiteTaskEvaluationError(["scenario_perturbations_invalid"])
    base = _validate_perturbation(
        perturbations.get("robot_base"), label="robot_base_perturbation"
    )
    camera = _validate_perturbation(
        perturbations.get("camera"), label="camera_perturbation"
    )
    scenario["perturbations"] = {"robot_base": base, "camera": camera}
    scenario["observation_settings"] = _validate_observation_settings(
        scenario.get("observation_settings")
    )
    scenario["geometry_material_variants"] = _validate_variants(
        scenario.get("geometry_material_variants")
    )
    _require_digest(
        scenario.get("policy_observation_spec_digest"),
        code="scenario_policy_observation_spec_digest_invalid",
    )
    _require_digest(
        scenario.get("evaluator_only_state_digest"),
        code="scenario_evaluator_only_state_digest_invalid",
    )
    if scenario.get("hidden_evaluator_data_in_policy_input") is not False:
        raise NewSiteTaskEvaluationError(
            ["scenario_hidden_evaluator_data_exposure_forbidden"]
        )
    if scenario.get("scenario_generation_may_authorize_new_claims") is not False:
        raise NewSiteTaskEvaluationError(["scenario_self_authorization_forbidden"])
    _require_nonempty(
        scenario.get("inclusion_rationale"), code="scenario_inclusion_rationale_missing"
    )
    if scenario.get("scenario_kind") == "nominal" and (
        base.get("status") != _NON_APPLIED_PERTURBATION_STATUS
        or camera.get("status") != _NON_APPLIED_PERTURBATION_STATUS
    ):
        raise NewSiteTaskEvaluationError(["nominal_scenario_has_perturbation"])
    if (
        scenario.get("scenario_kind")
        == "bounded_placement_observation_perturbation"
        and base.get("status") != _APPLIED_PERTURBATION_STATUS
        and camera.get("status") != _APPLIED_PERTURBATION_STATUS
    ):
        raise NewSiteTaskEvaluationError(["bounded_perturbation_scenario_has_no_perturbation"])
    if scenario.get("scenario_kind") == "visibility_occlusion_stress":
        occlusion = scenario["observation_settings"].get("occlusion")
        if not isinstance(occlusion, Mapping) or occlusion.get("status") != "qualified_bounded":
            raise NewSiteTaskEvaluationError(["occlusion_stress_qualification_missing"])
        _require_digest(
            occlusion.get("qualification_digest"),
            code="occlusion_stress_qualification_digest_invalid",
        )
        fraction = occlusion.get("maximum_target_mask_fraction")
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not 0.0 < float(fraction) < 1.0
        ):
            raise NewSiteTaskEvaluationError(["occlusion_stress_bound_invalid"])
    return scenario


def _validate_aggregation_rule(
    value: Any, *, metric: Mapping[str, Any]
) -> dict[str, Any]:
    rule = _validate_canonical_artifact(
        value,
        label="scenario_aggregation_rule",
        digest_field="aggregation_rule_digest",
        accepted_schemas={AGGREGATION_SCHEMA_VERSION},
    )
    expected = {
        "metric_spec_digest": metric.get("metric_spec_digest"),
        "direction": metric.get("direction"),
        "method": "paired_complete_scenario_mean",
        "uncertainty_method": "deterministic_paired_bootstrap_percentile_95",
        "unsupported_metric_policy": "exclude_scenario_from_paired_ranking",
        "catastrophic_rule": "any_supported_cell_at_or_beyond_threshold",
        "aggregate_may_mask_catastrophic_failure": False,
    }
    if any(rule.get(key) != expected_value for key, expected_value in expected.items()):
        raise NewSiteTaskEvaluationError(["scenario_aggregation_rule_invalid"])
    for field in ("minimum_paired_scenarios", "bootstrap_replicates", "bootstrap_seed"):
        number = rule.get(field)
        minimum = 1 if field != "bootstrap_seed" else 0
        if isinstance(number, bool) or not isinstance(number, int) or number < minimum:
            raise NewSiteTaskEvaluationError([f"scenario_aggregation_{field}_invalid"])
    if rule["bootstrap_replicates"] > 100_000:
        raise NewSiteTaskEvaluationError(["scenario_aggregation_bootstrap_replicates_unbounded"])
    for field in ("tie_tolerance", "catastrophic_failure_threshold"):
        number = rule.get(field)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
        ):
            raise NewSiteTaskEvaluationError([f"scenario_aggregation_{field}_invalid"])
    if float(rule["tie_tolerance"]) < 0.0:
        raise NewSiteTaskEvaluationError(["scenario_aggregation_tie_tolerance_invalid"])
    return rule


def validate_scenario_pack(
    value: Any,
    *,
    expected_bindings: Mapping[str, str],
    metric: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one immutable admitted scenario pack and all nested digests."""

    pack = _validate_canonical_artifact(
        value,
        label="scenario_pack",
        digest_field="scenario_pack_digest",
        accepted_schemas={SCENARIO_PACK_SCHEMA_VERSION},
    )
    _require_nonempty(pack.get("scenario_pack_id"), code="scenario_pack_id_missing")
    if pack.get("bindings") != expected_bindings:
        raise NewSiteTaskEvaluationError(["scenario_pack_scope_binding_mismatch"])
    if pack.get("frozen_before_execution") is not True:
        raise NewSiteTaskEvaluationError(["scenario_pack_not_frozen_before_execution"])
    _parse_timestamp(pack.get("frozen_at"), code="scenario_pack_frozen_at_invalid")
    if pack.get("scenario_generation_may_authorize_new_claims") is not False:
        raise NewSiteTaskEvaluationError(["scenario_pack_self_authorization_forbidden"])
    if pack.get("matrix_evidence_type") != "learned_policy_scenario_matrix":
        raise NewSiteTaskEvaluationError(["scenario_pack_evidence_type_invalid"])
    if pack.get("scripted_controller_matrix_separate") is not True:
        raise NewSiteTaskEvaluationError(["scripted_controller_matrix_separation_missing"])
    pack_metric = _validate_canonical_artifact(
        pack.get("preregistered_metric"),
        label="scenario_pack_metric",
        digest_field="metric_spec_digest",
        accepted_schemas={METRIC_SCHEMA_VERSION},
    )
    if pack_metric != metric:
        raise NewSiteTaskEvaluationError(["scenario_pack_metric_mismatch"])
    rule = _validate_aggregation_rule(pack.get("aggregation_rule"), metric=metric)
    definitions = pack.get("scenario_definitions")
    if not isinstance(definitions, list):
        raise NewSiteTaskEvaluationError(["scenario_definitions_invalid"])
    pack_kind = pack.get("pack_kind")
    if pack_kind == "inspection":
        if not MIN_INSPECTION_SCENARIOS <= len(definitions) <= MAX_INSPECTION_SCENARIOS:
            raise NewSiteTaskEvaluationError(["inspection_scenario_count_out_of_bounds"])
    elif pack_kind == "legacy_single_scenario_projection":
        if len(definitions) != 1 or pack.get("legacy_projection") is not True:
            raise NewSiteTaskEvaluationError(["legacy_scenario_projection_invalid"])
    else:
        raise NewSiteTaskEvaluationError(["scenario_pack_kind_invalid"])
    if pack.get("scenario_count") != len(definitions):
        raise NewSiteTaskEvaluationError(["scenario_pack_count_mismatch"])
    scenarios: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    digests: set[str] = set()
    pack_id = str(pack["scenario_pack_id"])
    for raw in definitions:
        scenario = _validate_scenario(
            raw,
            pack_id=pack_id,
            expected_bindings=expected_bindings,
            metric_spec_digest=str(metric["metric_spec_digest"]),
        )
        scenario_id = str(scenario["scenario_id"])
        scenario_digest = str(scenario["scenario_digest"])
        if scenario_id in identifiers or scenario_digest in digests:
            raise NewSiteTaskEvaluationError(["scenario_identity_duplicate"])
        identifiers.add(scenario_id)
        digests.add(scenario_digest)
        scenarios.append(scenario)
    if pack_kind == "inspection":
        kinds = {str(row["scenario_kind"]) for row in scenarios}
        if "nominal" not in kinds:
            raise NewSiteTaskEvaluationError(["inspection_nominal_scenario_missing"])
        if "bounded_placement_observation_perturbation" not in kinds:
            raise NewSiteTaskEvaluationError(["inspection_bounded_perturbation_missing"])
    excluded = pack.get("excluded_scenarios")
    if not isinstance(excluded, list):
        raise NewSiteTaskEvaluationError(["scenario_pack_exclusions_invalid"])
    for row in excluded:
        if not isinstance(row, Mapping):
            raise NewSiteTaskEvaluationError(["scenario_pack_exclusion_invalid"])
        _require_nonempty(
            row.get("scenario_id"), code="scenario_pack_exclusion_id_missing"
        )
        _require_nonempty(
            row.get("exclusion_rationale"),
            code="scenario_pack_exclusion_rationale_missing",
        )
        if row.get("scenario_id") in identifiers:
            raise NewSiteTaskEvaluationError(["scenario_included_and_excluded"])
    if rule["minimum_paired_scenarios"] > len(scenarios):
        raise NewSiteTaskEvaluationError(["scenario_aggregation_minimum_exceeds_pack"])
    pack["scenario_definitions"] = scenarios
    pack["aggregation_rule"] = rule
    return pack


def _common_v2_context(request: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source, missing = _source_gate(request)
    source_digest = source.get("source_profile_digest") if source else None
    if missing:
        return {}, {
            "stage": "capture_source_admission",
            "source_profile_digest": source_digest,
            **missing,
        }
    reconstruction, missing = _reconstruction_gate(request.get("reconstruction"))
    if missing:
        return {}, {
            "stage": "reconstruction_registration",
            "source_profile_digest": source_digest,
            **missing,
        }
    if reconstruction.get("source_profile_digest") != source_digest:
        raise NewSiteTaskEvaluationError(["reconstruction_source_profile_mismatch"])
    target, missing = _target_gate(
        request.get("target_orchestration"),
        reconstruction_digest=str(reconstruction["reconstruction_digest"]),
        source_scene_digest=str(
            reconstruction.get("source_scene_digest")
            or reconstruction["reconstruction_digest"]
        ),
        appearance_asset_digest=str(reconstruction["appearance_asset_digest"]),
    )
    if missing:
        return {}, {
            "stage": "automatic_task_target_binding",
            "source_profile_digest": source_digest,
            **missing,
        }
    selected_target = dict(target["selected_target"])
    robot_id = _expected_robot(selected_target)
    placement, missing = _placement_gate(
        request.get("robot_placement"),
        expected_robot=robot_id,
        target_binding_digest=str(selected_target["target_binding_digest"]),
    )
    if missing:
        return {}, {
            "stage": "robot_placement",
            "source_profile_digest": source_digest,
            "robot_id": robot_id,
            **missing,
        }
    requirement = target.get("task_zone_asset_requirement")
    requirement = dict(requirement) if isinstance(requirement, Mapping) else {}
    composition, missing = _scene_composition_gate(
        request.get("scene_composition"),
        task_zone_required=requirement.get("verified_simready_asset_required") is True,
    )
    if missing:
        return {}, {
            "stage": "qualified_scene_composition",
            "source_profile_digest": source_digest,
            "robot_id": robot_id,
            **missing,
        }
    route, missing = _route_gate(
        request.get("routing_inputs"),
        source_profile_digest=str(source_digest),
        target_binding_digest=str(selected_target["target_binding_digest"]),
        placement_digest=str(placement["placement_digest"]),
        robot_id=robot_id,
        task_class=str(selected_target["task_class"]),
    )
    if missing:
        return {}, {
            "stage": "task_site_engine_routing",
            "source_profile_digest": source_digest,
            "robot_id": robot_id,
            "routing_decision": route,
            **missing,
        }
    evaluation = request.get("policy_evaluation")
    if not isinstance(evaluation, Mapping):
        return {}, {
            "stage": "learned_policy_scenario_matrix",
            "source_profile_digest": source_digest,
            "robot_id": robot_id,
            "code": "five_policy_scenario_matrix_missing",
            "instruction": "Supply five learned policies and one frozen scenario pack.",
        }
    metric, metric_frozen_at = _metric_spec(evaluation.get("task_metric"))
    candidates = _policy_candidates(evaluation.get("policy_candidates"))
    route_inputs = request.get("routing_inputs")
    route_inputs = dict(route_inputs) if isinstance(route_inputs, Mapping) else {}
    site_profile = route_inputs.get("site_evidence_profile")
    site_profile = dict(site_profile) if isinstance(site_profile, Mapping) else {}
    site_id = _require_nonempty(
        site_profile.get("site_id") or site_profile.get("profile_id"),
        code="scenario_site_binding_missing",
    )
    task_id = _require_nonempty(
        selected_target.get("task_id") or selected_target.get("proposal_id"),
        code="scenario_task_binding_missing",
    )
    bindings = _scenario_bindings(
        site_id=site_id,
        task_id=task_id,
        source_profile_digest=str(source_digest),
        reconstruction_digest=str(reconstruction["reconstruction_digest"]),
        target_binding_digest=str(selected_target["target_binding_digest"]),
        robot_id=robot_id,
        placement_digest=str(placement["placement_digest"]),
        task_class=str(selected_target["task_class"]),
    )
    pack = validate_scenario_pack(
        evaluation.get("scenario_pack"),
        expected_bindings=bindings,
        metric=metric,
    )
    context = {
        "source": source,
        "reconstruction": reconstruction,
        "target": target,
        "selected_target": selected_target,
        "robot_id": robot_id,
        "placement": placement,
        "composition": composition,
        "route": route,
        "metric": metric,
        "metric_frozen_at": metric_frozen_at,
        "candidates": candidates,
        "scenario_pack": pack,
        "bindings": bindings,
    }
    return context, {}


def _authorization_v2(value: Any, *, context: Mapping[str, Any]) -> dict[str, Any]:
    authorization = _validate_canonical_artifact(
        value,
        label="policy_execution_authorization",
        digest_field="authorization_digest",
        accepted_schemas={AUTHORIZATION_SCHEMA_VERSION_V2},
    )
    route = dict(context["route"])
    placement = dict(context["placement"])
    metric = dict(context["metric"])
    candidates = list(context["candidates"])
    pack = dict(context["scenario_pack"])
    candidate_set_digest = canonical_digest(
        {
            "policy_identity_digests": sorted(
                str(row["policy_identity_digest"]) for row in candidates
            )
        }
    )
    expected = {
        "policy_execution_authorized": True,
        "physical_robot_execution_authorized": False,
        "routing_decision_digest": route.get("routing_decision_digest"),
        "placement_digest": placement.get("placement_digest"),
        "metric_spec_digest": metric.get("metric_spec_digest"),
        "candidate_set_digest": candidate_set_digest,
        "scenario_pack_digest": pack.get("scenario_pack_digest"),
        "aggregation_rule_digest": dict(pack["aggregation_rule"]).get(
            "aggregation_rule_digest"
        ),
        "matrix_evidence_type": "learned_policy_scenario_matrix",
    }
    errors = [
        f"policy_authorization_{key}_mismatch"
        for key, expected_value in expected.items()
        if authorization.get(key) != expected_value
    ]
    if errors:
        raise NewSiteTaskEvaluationError(errors)
    return authorization


def _request_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NewSiteTaskEvaluationError(["new_site_request_invalid"])
    request = _clone(dict(value))
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION_V2:
        raise NewSiteTaskEvaluationError(["new_site_request_schema_invalid"])
    supplied = request.pop("request_digest", None)
    digest_projection = _clone(request)
    evaluation = digest_projection.get("policy_evaluation")
    if isinstance(evaluation, dict):
        evaluation.pop("matrix_execution_packet", None)
    expected = canonical_digest(digest_projection)
    if supplied is not None and supplied != expected:
        raise NewSiteTaskEvaluationError(["new_site_request_digest_mismatch"])
    request["request_digest"] = expected
    _require_nonempty(request.get("run_id"), code="new_site_run_id_missing")
    return request


def _plan_from_context(
    request: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    pack = dict(context["scenario_pack"])
    cells: list[dict[str, Any]] = []
    for scenario in pack["scenario_definitions"]:
        for candidate in context["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            scenario_id = str(scenario["scenario_id"])
            policy_query_payload = {
                "candidate_id": candidate_id,
                "policy_identity_digest": candidate["policy_identity_digest"],
                "observation_schema_digest": candidate["observation_schema_digest"],
                "action_schema_digest": candidate["action_schema_digest"],
                "observation_sequence_spec_digest": candidate[
                    "observation_sequence_spec_digest"
                ],
                "policy_observation_spec_digest": scenario[
                    "policy_observation_spec_digest"
                ],
                "hidden_evaluator_data_included": False,
            }
            cell = {
                "cell_id": _scenario_cell_id(candidate_id, scenario_id),
                "candidate_id": candidate_id,
                "policy_identity_digest": candidate["policy_identity_digest"],
                "scenario_id": scenario_id,
                "scenario_digest": scenario["scenario_digest"],
                "scenario_pack_digest": pack["scenario_pack_digest"],
                "reset_state_digest": scenario["reset_state_digest"],
                "initial_state_observation_digest": scenario[
                    "initial_state_observation_digest"
                ],
                "deterministic_simulator_seed": scenario[
                    "deterministic_simulator_seed"
                ],
                "routing_decision_digest": dict(context["route"])[
                    "routing_decision_digest"
                ],
                "placement_digest": dict(context["placement"])["placement_digest"],
                "metric_spec_digest": dict(context["metric"])["metric_spec_digest"],
                "policy_query_payload": policy_query_payload,
            }
            cell["cell_plan_digest"] = canonical_digest(cell)
            cells.append(cell)
    plan = {
        "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
        "run_id": request["run_id"],
        "request_digest": request["request_digest"],
        "matrix_evidence_type": "learned_policy_scenario_matrix",
        "scenario_pack_id": pack["scenario_pack_id"],
        "scenario_pack_digest": pack["scenario_pack_digest"],
        "policy_count": len(context["candidates"]),
        "scenario_count": len(pack["scenario_definitions"]),
        "expected_cell_count": len(cells),
        "cells": cells,
    }
    plan["execution_plan_digest"] = canonical_digest(plan)
    return plan


def build_matrix_execution_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    """Build the exact deterministic 5 x N plan without invoking a policy."""

    request = _request_v2(value)
    context, missing = _common_v2_context(request)
    if missing:
        raise NewSiteTaskEvaluationError([str(missing["code"])])
    _authorization_v2(request.get("execution_authorization"), context=context)
    return _plan_from_context(request, context)


def _failed_cell(plan_cell: Mapping[str, Any], *, status: str, blocker: str) -> dict[str, Any]:
    cell = {
        "cell_id": plan_cell["cell_id"],
        "candidate_id": plan_cell["candidate_id"],
        "policy_identity_digest": plan_cell["policy_identity_digest"],
        "scenario_id": plan_cell["scenario_id"],
        "scenario_digest": plan_cell["scenario_digest"],
        "cell_plan_digest": plan_cell["cell_plan_digest"],
        "status": status,
        "attempt_receipt": None,
        "blockers": [blocker],
    }
    cell["cell_result_digest"] = canonical_digest(cell)
    return cell


def execute_policy_scenario_matrix(
    value: Mapping[str, Any],
    runner: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Execute every planned cell and retain failures instead of short-circuiting.

    The runner receives a cell plan whose policy query payload intentionally
    omits evaluator-only state.  It must return a digest-bound v2 attempt
    receipt.  Exceptions and missing returns become explicit terminal cells.
    """

    request = _request_v2(value)
    context, missing = _common_v2_context(request)
    if missing:
        raise NewSiteTaskEvaluationError([str(missing["code"])])
    _authorization_v2(request.get("execution_authorization"), context=context)
    plan = _plan_from_context(request, context)
    results: list[dict[str, Any]] = []
    for plan_cell in plan["cells"]:
        try:
            receipt = runner(_clone(plan_cell))
        except Exception as exc:  # executor must preserve the remaining grid
            blocker = f"runner_exception:{type(exc).__name__}"
            results.append(_failed_cell(plan_cell, status="failed", blocker=blocker))
            continue
        if receipt is None:
            results.append(
                _failed_cell(
                    plan_cell,
                    status="missing",
                    blocker="attempt_receipt_missing",
                )
            )
            continue
        if not isinstance(receipt, Mapping):
            results.append(
                _failed_cell(
                    plan_cell,
                    status="failed",
                    blocker="attempt_receipt_invalid",
                )
            )
            continue
        cell = {
            "cell_id": plan_cell["cell_id"],
            "candidate_id": plan_cell["candidate_id"],
            "policy_identity_digest": plan_cell["policy_identity_digest"],
            "scenario_id": plan_cell["scenario_id"],
            "scenario_digest": plan_cell["scenario_digest"],
            "cell_plan_digest": plan_cell["cell_plan_digest"],
            "status": "completed" if receipt.get("status") == "completed" else "failed",
            "attempt_receipt": _clone(dict(receipt)),
            "blockers": [],
        }
        if cell["status"] == "failed":
            blockers = receipt.get("blockers")
            cell["blockers"] = (
                sorted(set(str(row) for row in blockers))
                if isinstance(blockers, list) and blockers
                else ["attempt_failed_without_blocker"]
            )
        cell["cell_result_digest"] = canonical_digest(cell)
        results.append(cell)
    packet = {
        "schema_version": EXECUTION_PACKET_SCHEMA_VERSION,
        "run_id": request["run_id"],
        "request_digest": request["request_digest"],
        "matrix_evidence_type": "learned_policy_scenario_matrix",
        "execution_plan": plan,
        "expected_cell_count": plan["expected_cell_count"],
        "observed_cell_count": len(results),
        "status": (
            "completed"
            if all(row["status"] == "completed" for row in results)
            else "completed_with_failures"
        ),
        "cells": results,
    }
    packet["execution_packet_digest"] = canonical_digest(packet)
    return packet


def _cell_attempt(
    raw: Any,
    *,
    plan_cell: Mapping[str, Any],
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not isinstance(raw, Mapping):
        raise NewSiteTaskEvaluationError(["matrix_cell_invalid"])
    cell = _validate_canonical_artifact(
        raw, label="matrix_cell", digest_field="cell_result_digest"
    )
    bindings = {
        "cell_id": plan_cell["cell_id"],
        "candidate_id": plan_cell["candidate_id"],
        "policy_identity_digest": plan_cell["policy_identity_digest"],
        "scenario_id": plan_cell["scenario_id"],
        "scenario_digest": plan_cell["scenario_digest"],
        "cell_plan_digest": plan_cell["cell_plan_digest"],
    }
    if any(cell.get(key) != expected for key, expected in bindings.items()):
        raise NewSiteTaskEvaluationError(["matrix_cell_scope_binding_mismatch"])
    status = cell.get("status")
    if status not in _CELL_TERMINAL_STATUSES:
        raise NewSiteTaskEvaluationError(["matrix_cell_status_invalid"])
    blockers = cell.get("blockers")
    if not isinstance(blockers, list) or not all(isinstance(row, str) for row in blockers):
        raise NewSiteTaskEvaluationError(["matrix_cell_blockers_invalid"])
    receipt_value = cell.get("attempt_receipt")
    if receipt_value is None:
        if status not in {"missing", "abstained_pre_execution"} or not blockers:
            raise NewSiteTaskEvaluationError(["matrix_cell_missing_attempt_not_explained"])
        return cell, None
    receipt = _validate_canonical_artifact(
        receipt_value,
        label="matrix_attempt",
        digest_field="attempt_digest",
        accepted_schemas={MATRIX_ATTEMPT_SCHEMA_VERSION},
    )
    receipt_status = receipt.get("status")
    if receipt_status not in {"completed", "failed", "abstained"}:
        raise NewSiteTaskEvaluationError(["matrix_attempt_status_invalid"])
    if status in {"missing", "abstained_pre_execution"} or (
        (status == "completed") != (receipt_status == "completed")
    ):
        raise NewSiteTaskEvaluationError(["matrix_cell_attempt_status_mismatch"])
    expected_receipt = {
        "cell_id": plan_cell["cell_id"],
        "cell_plan_digest": plan_cell["cell_plan_digest"],
        "candidate_id": plan_cell["candidate_id"],
        "policy_identity_digest": plan_cell["policy_identity_digest"],
        "scenario_id": plan_cell["scenario_id"],
        "scenario_digest": plan_cell["scenario_digest"],
        "scenario_pack_digest": plan_cell["scenario_pack_digest"],
        "reset_state_digest": plan_cell["reset_state_digest"],
        "initial_state_observation_digest": plan_cell[
            "initial_state_observation_digest"
        ],
        "deterministic_simulator_seed": plan_cell["deterministic_simulator_seed"],
        "routing_decision_digest": dict(context["route"])["routing_decision_digest"],
        "placement_digest": dict(context["placement"])["placement_digest"],
    }
    if any(receipt.get(key) != expected for key, expected in expected_receipt.items()):
        raise NewSiteTaskEvaluationError(["matrix_attempt_scope_binding_mismatch"])
    for field in (
        "execution_receipt_digest",
        "initial_state_observation_digest",
        "observation_trace_digest",
        "action_trace_digest",
        "contact_evidence_digest",
        "collision_evidence_digest",
    ):
        _require_digest(receipt.get(field), code=f"matrix_attempt_{field}_invalid")
    if receipt.get("action_source") not in _LEARNED_ACTION_SOURCES:
        raise NewSiteTaskEvaluationError(["matrix_attempt_not_learned_policy"])
    if receipt.get("hidden_evaluator_data_accessed") is not False:
        raise NewSiteTaskEvaluationError(["matrix_attempt_hidden_evaluator_access_forbidden"])
    started = _parse_timestamp(receipt.get("started_at"), code="matrix_attempt_started_at_invalid")
    ended = _parse_timestamp(receipt.get("ended_at"), code="matrix_attempt_ended_at_invalid")
    if ended < started or started <= context["metric_frozen_at"]:
        raise NewSiteTaskEvaluationError(["task_metric_not_frozen_before_attempt"])
    for field in ("fresh_policy_query_count", "learned_policy_action_count"):
        count = receipt.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise NewSiteTaskEvaluationError(["matrix_attempt_real_execution_not_proven"])
    if receipt.get("learned_policy_action_proven") is not True or receipt.get(
        "reset_observed"
    ) is not True:
        raise NewSiteTaskEvaluationError(["matrix_attempt_real_execution_not_proven"])
    outcome = receipt.get("task_metric_result")
    if not isinstance(outcome, Mapping):
        raise NewSiteTaskEvaluationError(["matrix_attempt_metric_result_missing"])
    if outcome.get("metric_spec_digest") != dict(context["metric"])["metric_spec_digest"]:
        raise NewSiteTaskEvaluationError(["matrix_attempt_metric_spec_mismatch"])
    metric_value = outcome.get("value")
    if metric_value is not None and (
        isinstance(metric_value, bool)
        or not isinstance(metric_value, (int, float))
        or not math.isfinite(float(metric_value))
    ):
        raise NewSiteTaskEvaluationError(["matrix_attempt_metric_value_invalid"])
    metric_blockers = outcome.get("blockers")
    if not isinstance(metric_blockers, list) or not all(
        isinstance(row, str) for row in metric_blockers
    ):
        raise NewSiteTaskEvaluationError(["matrix_attempt_metric_blockers_invalid"])
    supported = bool(
        status == "completed"
        and receipt.get("status") == "completed"
        and outcome.get("supported_for_ranking") is True
        and metric_value is not None
        and not metric_blockers
        and not blockers
    )
    receipt["supported_for_ranking"] = supported
    return cell, receipt


def _validate_execution_packet(
    value: Any, *, request: Mapping[str, Any], context: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any] | None]]:
    packet = _validate_canonical_artifact(
        value,
        label="matrix_execution_packet",
        digest_field="execution_packet_digest",
        accepted_schemas={EXECUTION_PACKET_SCHEMA_VERSION},
    )
    if (
        packet.get("run_id") != request["run_id"]
        or packet.get("request_digest") != request["request_digest"]
        or packet.get("matrix_evidence_type") != "learned_policy_scenario_matrix"
    ):
        raise NewSiteTaskEvaluationError(["matrix_execution_packet_scope_mismatch"])
    expected_plan = _plan_from_context(request, context)
    supplied_plan = packet.get("execution_plan")
    if supplied_plan != expected_plan:
        raise NewSiteTaskEvaluationError(["matrix_execution_plan_mismatch"])
    raw_cells = packet.get("cells")
    if not isinstance(raw_cells, list):
        raise NewSiteTaskEvaluationError(["matrix_execution_cells_invalid"])
    if packet.get("expected_cell_count") != expected_plan["expected_cell_count"]:
        raise NewSiteTaskEvaluationError(["matrix_execution_expected_cell_count_mismatch"])
    if packet.get("observed_cell_count") != len(raw_cells):
        raise NewSiteTaskEvaluationError(["matrix_execution_observed_cell_count_mismatch"])
    expected_packet_status = (
        "completed"
        if len(raw_cells) == expected_plan["expected_cell_count"]
        and all(
            isinstance(row, Mapping) and row.get("status") == "completed"
            for row in raw_cells
        )
        else "completed_with_failures"
    )
    if packet.get("status") != expected_packet_status:
        raise NewSiteTaskEvaluationError(["matrix_execution_packet_status_mismatch"])
    cells_by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_cells:
        if not isinstance(raw, Mapping):
            raise NewSiteTaskEvaluationError(["matrix_cell_invalid"])
        cell_id = str(raw.get("cell_id") or "")
        if not cell_id or cell_id in cells_by_id:
            raise NewSiteTaskEvaluationError(["matrix_cell_identity_duplicate_or_missing"])
        cells_by_id[cell_id] = raw
    expected_ids = {str(row["cell_id"]) for row in expected_plan["cells"]}
    foreign_ids = sorted(set(cells_by_id) - expected_ids)
    if foreign_ids:
        raise NewSiteTaskEvaluationError(["matrix_execution_foreign_cell"])
    normalized_cells: list[dict[str, Any]] = []
    attempts: list[dict[str, Any] | None] = []
    for plan_cell in expected_plan["cells"]:
        raw = cells_by_id.get(str(plan_cell["cell_id"]))
        if raw is None:
            raw = _failed_cell(
                plan_cell,
                status="missing",
                blocker="matrix_cell_missing_from_execution_packet",
            )
        cell, attempt = _cell_attempt(raw, plan_cell=plan_cell, context=context)
        normalized_cells.append(cell)
        attempts.append(attempt)
    return packet, normalized_cells, attempts


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise NewSiteTaskEvaluationError(["scenario_uncertainty_values_missing"])
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _bootstrap_samples(rule: Mapping[str, Any], count: int) -> list[list[int]]:
    generator = random.Random(int(rule["bootstrap_seed"]))
    return [
        [generator.randrange(count) for _ in range(count)]
        for _ in range(int(rule["bootstrap_replicates"]))
    ]


def _is_catastrophic(value: float, *, rule: Mapping[str, Any]) -> bool:
    threshold = float(rule["catastrophic_failure_threshold"])
    return value <= threshold if rule["direction"] == "maximize" else value >= threshold


def _aggregate_matrix(
    *,
    context: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any]:
    pack = dict(context["scenario_pack"])
    rule = dict(pack["aggregation_rule"])
    candidates = list(context["candidates"])
    scenario_ids = [str(row["scenario_id"]) for row in pack["scenario_definitions"]]
    by_key: dict[tuple[str, str], tuple[Mapping[str, Any], Mapping[str, Any] | None]] = {}
    for cell, attempt in zip(cells, attempts, strict=True):
        by_key[(str(cell["candidate_id"]), str(cell["scenario_id"]))] = (cell, attempt)
    paired: list[str] = []
    excluded: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        unsupported_cells = []
        for candidate in candidates:
            key = (str(candidate["candidate_id"]), scenario_id)
            cell, attempt = by_key[key]
            if attempt is None or attempt.get("supported_for_ranking") is not True:
                unsupported_cells.append(str(cell["cell_id"]))
        if unsupported_cells:
            excluded.append(
                {
                    "scenario_id": scenario_id,
                    "reason": "not_supported_for_all_paired_policies",
                    "unsupported_cell_ids": sorted(unsupported_cells),
                }
            )
        else:
            paired.append(scenario_id)
    bootstrap = _bootstrap_samples(rule, len(paired)) if paired else []
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        observed_attempts = []
        supported_attempts = []
        catastrophic_cells = []
        paired_values: list[float] = []
        unsupported_metrics: list[dict[str, Any]] = []
        for scenario_id in scenario_ids:
            cell, attempt = by_key[(candidate_id, scenario_id)]
            if attempt is not None:
                observed_attempts.append(str(attempt["attempt_digest"]))
            if attempt is not None and attempt.get("supported_for_ranking") is True:
                value = float(dict(attempt["task_metric_result"])["value"])
                supported_attempts.append(str(attempt["attempt_digest"]))
                if _is_catastrophic(value, rule=rule):
                    catastrophic_cells.append(str(cell["cell_id"]))
                if scenario_id in paired:
                    paired_values.append(value)
            else:
                metric_blockers = []
                if attempt is not None:
                    outcome = attempt.get("task_metric_result")
                    if isinstance(outcome, Mapping):
                        metric_blockers = list(outcome.get("blockers") or [])
                unsupported_metrics.append(
                    {
                        "cell_id": cell["cell_id"],
                        "scenario_id": scenario_id,
                        "cell_status": cell["status"],
                        "blockers": sorted(set(list(cell["blockers"]) + metric_blockers)),
                    }
                )
        aggregate = sum(paired_values) / len(paired_values) if paired_values else None
        distributions = [
            sum(paired_values[index] for index in sample) / len(sample)
            for sample in bootstrap
        ]
        uncertainty = (
            {
                "method": rule["uncertainty_method"],
                "confidence_level": 0.95,
                "lower": _percentile(distributions, 0.025),
                "upper": _percentile(distributions, 0.975),
                "bootstrap_replicates": len(distributions),
                "bootstrap_seed": rule["bootstrap_seed"],
            }
            if distributions
            else None
        )
        summaries.append(
            {
                "candidate_id": candidate_id,
                "policy_identity_digest": candidate["policy_identity_digest"],
                "attempt_coverage": len(observed_attempts) / len(scenario_ids),
                "supported_metric_coverage": len(supported_attempts) / len(scenario_ids),
                "paired_scenario_coverage": len(paired_values) / len(scenario_ids),
                "observed_attempt_count": len(observed_attempts),
                "supported_metric_count": len(supported_attempts),
                "paired_metric_count": len(paired_values),
                "aggregate_score": aggregate,
                "uncertainty": uncertainty,
                "catastrophic_failure_count": len(catastrophic_cells),
                "catastrophic_cell_ids": sorted(catastrophic_cells),
                "aggregate_does_not_hide_catastrophic_failures": True,
                "unsupported_metrics": unsupported_metrics,
                "eligible_for_winner": not catastrophic_cells and aggregate is not None,
            }
        )
    direction = str(rule["direction"])
    ordered = sorted(
        summaries,
        key=lambda row: (
            int(row["catastrophic_failure_count"]),
            -float(row["aggregate_score"])
            if direction == "maximize" and row["aggregate_score"] is not None
            else float(row["aggregate_score"])
            if row["aggregate_score"] is not None
            else math.inf,
            str(row["candidate_id"]),
        ),
    )
    ranking: list[dict[str, Any]] = []
    last_key: tuple[int, float] | None = None
    rank = 0
    tolerance = float(rule["tie_tolerance"])
    for index, summary in enumerate(ordered, start=1):
        aggregate = summary["aggregate_score"]
        if aggregate is None:
            continue
        key = (int(summary["catastrophic_failure_count"]), float(aggregate))
        tied = bool(
            last_key is not None
            and key[0] == last_key[0]
            and abs(key[1] - last_key[1]) <= tolerance
        )
        if not tied:
            rank = index
            last_key = key
        ranking.append(
            {
                "rank": rank,
                "candidate_id": summary["candidate_id"],
                "policy_identity_digest": summary["policy_identity_digest"],
                "aggregate_score": aggregate,
                "uncertainty": summary["uncertainty"],
                "paired_scenario_count": summary["paired_metric_count"],
                "catastrophic_failure_count": summary["catastrophic_failure_count"],
                "eligible_for_winner": summary["eligible_for_winner"],
            }
        )
    eligible = [row for row in ranking if row["eligible_for_winner"]]
    best_rank = min((int(row["rank"]) for row in eligible), default=None)
    winners = sorted(
        str(row["candidate_id"])
        for row in eligible
        if int(row["rank"]) == best_rank
    )
    return {
        "paired_scenario_ids": paired,
        "paired_scenario_count": len(paired),
        "excluded_scenarios": excluded,
        "candidate_summaries": summaries,
        "ranking": ranking,
        "winner_candidate_ids": winners,
        "winner_candidate_id": winners[0] if len(winners) == 1 else None,
    }


def _cell_summaries(
    cells: Sequence[Mapping[str, Any]], attempts: Sequence[Mapping[str, Any] | None]
) -> list[dict[str, Any]]:
    summaries = []
    for cell, attempt in zip(cells, attempts, strict=True):
        metric = dict(attempt["task_metric_result"]) if attempt is not None else {}
        supported = bool(attempt is not None and attempt.get("supported_for_ranking") is True)
        blockers = sorted(
            set(list(cell["blockers"]) + list(metric.get("blockers") or []))
        )
        summary = {
            "cell_id": cell["cell_id"],
            "candidate_id": cell["candidate_id"],
            "policy_identity_digest": cell["policy_identity_digest"],
            "scenario_id": cell["scenario_id"],
            "scenario_digest": cell["scenario_digest"],
            "cell_status": cell["status"],
            "attempt_digest": attempt.get("attempt_digest") if attempt else None,
            "metric_supported": supported,
            "metric_value": metric.get("value") if supported else None,
            "blockers": blockers,
            "cell_abstention": (
                None
                if supported
                else {
                    "code": blockers[0] if blockers else "metric_not_supported",
                    "instruction": "Repair this exact policy/scenario cell and replay the frozen pack.",
                }
            ),
        }
        summaries.append(summary)
    return summaries


def _v2_abstention(
    *, request: Mapping[str, Any], missing: Mapping[str, Any]
) -> dict[str, Any]:
    result = {
        "schema_version": RESULT_SCHEMA_VERSION_V2,
        "run_id": str(request.get("run_id") or "unknown"),
        "status": "abstained",
        "request_digest": request.get("request_digest"),
        "source_profile_digest": missing.get("source_profile_digest"),
        "robot_id": missing.get("robot_id"),
        "scenario_pack_id": None,
        "scenario_pack_digest": None,
        "expected_cell_count": 0,
        "observed_attempt_count": 0,
        "cell_results": [],
        "paired_scenario_ids": [],
        "paired_scenario_count": 0,
        "excluded_scenarios": [],
        "candidate_summaries": [],
        "ranking": [],
        "winner_candidate_ids": [],
        "winner_candidate_id": None,
        "terminal_stage": missing["stage"],
        "blockers": [str(missing["code"])],
        "smallest_missing_measurement": {
            "code": str(missing["code"]),
            "instruction": str(missing["instruction"]),
            "stage": str(missing["stage"]),
        },
        "claim_boundary": {
            "task_evaluation_run_completed": False,
            "learned_policy_attempts_observed": False,
            "learned_policy_scenario_ranking_proven": False,
            "scripted_controller_matrix_is_learned_policy_matrix": False,
            "scenario_generation_authorized_new_claims": False,
            "physical_success_proven": False,
            "deployment_readiness_proven": False,
        },
    }
    if missing.get("routing_decision") is not None:
        result["routing_decision"] = _clone(missing["routing_decision"])
    result["run_digest"] = canonical_digest(result, digest_field="run_digest")
    return result


def _matrix_result(
    *,
    request: Mapping[str, Any],
    context: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any] | None],
    execution_packet_digest: str | None,
    forced_blocker: tuple[str, str] | None = None,
) -> dict[str, Any]:
    aggregation = _aggregate_matrix(context=context, cells=cells, attempts=attempts)
    pack = dict(context["scenario_pack"])
    rule = dict(pack["aggregation_rule"])
    cell_results = _cell_summaries(cells, attempts)
    missing_cells = [row for row in cell_results if row["cell_status"] == "missing"]
    paired_shortfall = aggregation["paired_scenario_count"] < rule[
        "minimum_paired_scenarios"
    ]
    no_winner = not aggregation["winner_candidate_ids"]
    if forced_blocker is not None:
        status = "abstained"
        blocker, instruction = forced_blocker
    elif missing_cells:
        status = "abstained"
        blocker = "matrix_missing_attempt_cells"
        instruction = "Execute the listed missing cells from their frozen scenario resets."
    elif paired_shortfall:
        status = "abstained"
        blocker = "insufficient_paired_scenario_coverage"
        instruction = "Repair unsupported cells until the preregistered paired-scenario minimum is met."
    elif no_winner:
        status = "abstained"
        blocker = "all_ranked_candidates_catastrophic"
        instruction = "Review catastrophic scenarios; no candidate is eligible for a winner claim."
    else:
        status = "completed"
        blocker = None
        instruction = None
    observed_attempt_count = sum(attempt is not None for attempt in attempts)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION_V2,
        "run_id": request["run_id"],
        "status": status,
        "request_digest": request["request_digest"],
        "source_profile_digest": dict(context["source"])["source_profile_digest"],
        "source_truth_class": (
            "provider_derived_support"
            if dict(context["source"])["status"] == "admitted_provider_derived_support"
            else "blueprint_raw_contract"
        ),
        "reconstruction_digest": dict(context["reconstruction"])["reconstruction_digest"],
        "target_binding_digest": dict(context["selected_target"])[
            "target_binding_digest"
        ],
        "robot_id": context["robot_id"],
        "placement_digest": dict(context["placement"])["placement_digest"],
        "routing_decision": _clone(context["route"]),
        "scenario_pack_id": pack["scenario_pack_id"],
        "scenario_pack_digest": pack["scenario_pack_digest"],
        "aggregation_rule_digest": rule["aggregation_rule_digest"],
        "execution_packet_digest": execution_packet_digest,
        "matrix_evidence_type": "learned_policy_scenario_matrix",
        "policy_count": EXPECTED_POLICY_CANDIDATES,
        "scenario_count": pack["scenario_count"],
        "expected_cell_count": EXPECTED_POLICY_CANDIDATES * pack["scenario_count"],
        "observed_attempt_count": observed_attempt_count,
        "cell_results": cell_results,
        **aggregation,
        "terminal_stage": "paired_scenario_aggregation",
        "blockers": [blocker] if blocker else [],
        "smallest_missing_measurement": (
            {"code": blocker, "instruction": instruction, "stage": "paired_scenario_aggregation"}
            if blocker
            else None
        ),
        "claim_boundary": {
            "task_evaluation_run_completed": status == "completed",
            "learned_policy_attempts_observed": observed_attempt_count > 0,
            "learned_policy_scenario_ranking_proven": status == "completed",
            "scripted_controller_matrix_is_learned_policy_matrix": False,
            "scenario_generation_authorized_new_claims": False,
            "aggregate_hides_catastrophic_failures": False,
            "physical_success_proven": False,
            "deployment_readiness_proven": False,
        },
    }
    result["run_digest"] = canonical_digest(result, digest_field="run_digest")
    return result


def compile_new_site_task_evaluation_run_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    """Compile the full frozen scenario matrix or a proof-bounded abstention."""

    request = _request_v2(value)
    context, missing = _common_v2_context(request)
    if missing:
        return _v2_abstention(request=request, missing=missing)
    _authorization_v2(request.get("execution_authorization"), context=context)
    evaluation = dict(request["policy_evaluation"])
    packet_value = evaluation.get("matrix_execution_packet")
    if not isinstance(packet_value, Mapping):
        plan = _plan_from_context(request, context)
        cells = [
            _failed_cell(
                row,
                status="missing",
                blocker="matrix_execution_packet_missing",
            )
            for row in plan["cells"]
        ]
        return _matrix_result(
            request=request,
            context=context,
            cells=cells,
            attempts=[None] * len(cells),
            execution_packet_digest=None,
            forced_blocker=(
                "matrix_execution_packet_missing",
                "Execute every policy/scenario cell and retain the complete packet.",
            ),
        )
    packet, cells, attempts = _validate_execution_packet(
        packet_value, request=request, context=context
    )
    return _matrix_result(
        request=request,
        context=context,
        cells=cells,
        attempts=attempts,
        execution_packet_digest=str(packet["execution_packet_digest"]),
    )


def migrate_v1_request_to_v2(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project a readable v1 request into an explicitly legacy one-scenario v2 pack.

    This migration preserves evidence; it does not upgrade a single-reset run to
    a scientifically useful multi-scenario matrix.  The returned record is a
    migration envelope so callers must explicitly select its ``v2_request``.
    """

    from .new_site_task_evaluation_run import (
        REQUEST_SCHEMA_VERSION,
        compile_new_site_task_evaluation_run_v1,
    )

    if value.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise NewSiteTaskEvaluationError(["v1_migration_source_schema_invalid"])
    original = _clone(dict(value))
    v1_result = compile_new_site_task_evaluation_run_v1(original)
    if v1_result.get("status") != "completed":
        raise NewSiteTaskEvaluationError(["v1_migration_requires_completed_run"])
    evaluation = dict(original["policy_evaluation"])
    metric = dict(evaluation["task_metric"])
    bindings = _scenario_bindings(
        site_id=str(
            dict(original["routing_inputs"])["site_evidence_profile"].get(
                "site_id"
            )
            or dict(original["routing_inputs"])["site_evidence_profile"][
                "profile_id"
            ]
        ),
        task_id=str(
            dict(original["target_orchestration"])["selected_target"].get(
                "task_id"
            )
            or dict(original["target_orchestration"])["selected_target"][
                "proposal_id"
            ]
        ),
        source_profile_digest=str(v1_result["source_profile_digest"]),
        reconstruction_digest=str(v1_result["reconstruction_digest"]),
        target_binding_digest=str(v1_result["target_binding_digest"]),
        robot_id=str(v1_result["robot_id"]),
        placement_digest=str(v1_result["placement_digest"]),
        task_class=str(v1_result["task_class"]),
    )
    rule = {
        "schema_version": AGGREGATION_SCHEMA_VERSION,
        "metric_spec_digest": metric["metric_spec_digest"],
        "method": "paired_complete_scenario_mean",
        "direction": metric["direction"],
        "minimum_paired_scenarios": 1,
        "uncertainty_method": "deterministic_paired_bootstrap_percentile_95",
        "bootstrap_replicates": 1,
        "bootstrap_seed": 0,
        "tie_tolerance": 0.0,
        "catastrophic_failure_threshold": -1.0e308
        if metric["direction"] == "maximize"
        else 1.0e308,
        "catastrophic_rule": "any_supported_cell_at_or_beyond_threshold",
        "unsupported_metric_policy": "exclude_scenario_from_paired_ranking",
        "aggregate_may_mask_catastrophic_failure": False,
    }
    rule["aggregation_rule_digest"] = canonical_digest(rule)
    pack_stub = {
        "schema_version": SCENARIO_PACK_SCHEMA_VERSION,
        "scenario_pack_id": f"{original['run_id']}-legacy-v1",
        "pack_kind": "legacy_single_scenario_projection",
        "legacy_projection": True,
        "bindings": bindings,
        "frozen_before_execution": True,
        "frozen_at": metric["frozen_at"],
        "matrix_evidence_type": "learned_policy_scenario_matrix",
        "scripted_controller_matrix_separate": True,
        "scenario_generation_may_authorize_new_claims": False,
        "preregistered_metric": metric,
        "aggregation_rule": rule,
        "scenario_count": 1,
        "scenario_definitions": [],
        "excluded_scenarios": [
            {
                "scenario_id": "additional-scenarios-not-present-in-v1",
                "exclusion_rationale": "The source v1 run froze only one reset.",
            }
        ],
    }
    scenario = {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "scenario_id": "legacy-nominal",
        "scenario_pack_id": pack_stub["scenario_pack_id"],
        "scenario_kind": "nominal",
        "admission_status": "admitted",
        "bindings": bindings,
        "metric_spec_digest": metric["metric_spec_digest"],
        "frozen_before_execution": True,
        "frozen_at": metric["frozen_at"],
        "reset_state_digest": v1_result["matched_reset_digest"],
        "initial_state_observation_digest": v1_result[
            "initial_state_observation_digest"
        ],
        "deterministic_simulator_seed": 0,
        "target_state": {
            "state_digest": v1_result["initial_state_observation_digest"],
            "policy_visibility": "observed",
        },
        "distractor_state": {
            "state_digest": canonical_digest({"legacy": "not_recorded"}),
            "policy_visibility": "not_directly_observed",
        },
        "perturbations": {
            "robot_base": {"status": "not_applied"},
            "camera": {"status": "not_applied"},
        },
        "observation_settings": {
            "lighting": {"status": "legacy_not_recorded"},
            "sensor": {"status": "legacy_not_recorded"},
            "noise": {"status": "legacy_not_recorded"},
            "evidence_ceiling_digest": canonical_digest(
                {"legacy": "settings_not_recorded"}
            ),
            "settings_may_authorize_new_claims": False,
        },
        "geometry_material_variants": [],
        "policy_observation_spec_digest": evaluation["policy_candidates"][0][
            "observation_sequence_spec_digest"
        ],
        "evaluator_only_state_digest": canonical_digest(
            {"legacy": "no_hidden_evaluator_state_recorded"}
        ),
        "hidden_evaluator_data_in_policy_input": False,
        "scenario_generation_may_authorize_new_claims": False,
        "inclusion_rationale": "Faithful projection of the one reset observed by v1.",
    }
    scenario["observation_settings"]["settings_digest"] = canonical_digest(
        scenario["observation_settings"]
    )
    scenario["scenario_digest"] = canonical_digest(scenario)
    pack_stub["scenario_definitions"] = [scenario]
    pack_stub["scenario_pack_digest"] = canonical_digest(
        pack_stub
    )
    migration = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "source_request_digest": original["request_digest"],
        "source_run_digest": v1_result["run_digest"],
        "status": "projected_without_claim_upgrade",
        "scenario_pack": pack_stub,
        "claim_boundary": {
            "multi_scenario_evidence_created": False,
            "v1_source_preserved": True,
            "ranking_claim_upgraded": False,
        },
    }
    migration["migration_digest"] = canonical_digest(migration)
    return migration


def project_v2_result_to_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    """Create a v1-readable compatibility projection without hiding matrix scope."""

    if value.get("schema_version") != RESULT_SCHEMA_VERSION_V2:
        raise NewSiteTaskEvaluationError(["v2_projection_source_schema_invalid"])
    result = _clone(dict(value))
    ranking = []
    for row in result.get("ranking") or []:
        aggregate_digest = canonical_digest(
            {
                "scenario_pack_digest": result.get("scenario_pack_digest"),
                "candidate_id": row["candidate_id"],
                "aggregate_score": row["aggregate_score"],
            }
        )
        ranking.append(
            {
                "rank": row["rank"],
                "candidate_id": row["candidate_id"],
                "policy_identity_digest": row["policy_identity_digest"],
                "attempt_digest": aggregate_digest,
                "metric_value": row["aggregate_score"],
            }
        )
    projection = {
        "schema_version": "new_site_task_evaluation_run.v1",
        "run_id": result["run_id"],
        "status": result["status"],
        "source_profile_digest": result.get("source_profile_digest"),
        "terminal_stage": result["terminal_stage"],
        "blockers": list(result["blockers"]),
        "smallest_missing_measurement": result["smallest_missing_measurement"],
        "robot_id": result.get("robot_id"),
        "policy_attempt_count": min(
            EXPECTED_POLICY_CANDIDATES, int(result.get("observed_attempt_count") or 0)
        ),
        "supported_ranking_candidate_count": len(ranking),
        "ranking": ranking,
        "winner_candidate_ids": list(result.get("winner_candidate_ids") or []),
        "winner_candidate_id": result.get("winner_candidate_id"),
        "v2_matrix_run_digest": result.get("run_digest"),
        "v2_scenario_pack_digest": result.get("scenario_pack_digest"),
        "claim_boundary": {
            "task_evaluation_run_completed": result["status"] == "completed",
            "learned_policy_attempts_observed": bool(result.get("observed_attempt_count")),
            "learned_policy_ranking_proven": result["status"] == "completed",
            "v1_projection_of_v2_matrix": True,
            "physical_success_proven": False,
            "deployment_readiness_proven": False,
        },
    }
    projection["run_digest"] = canonical_digest(projection, digest_field="run_digest")
    return projection


__all__ = [
    "AGGREGATION_SCHEMA_VERSION",
    "AUTHORIZATION_SCHEMA_VERSION_V2",
    "EXECUTION_PACKET_SCHEMA_VERSION",
    "EXECUTION_PLAN_SCHEMA_VERSION",
    "MATRIX_ATTEMPT_SCHEMA_VERSION",
    "MIGRATION_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION_V2",
    "RESULT_SCHEMA_VERSION_V2",
    "SCENARIO_PACK_SCHEMA_VERSION",
    "SCENARIO_SCHEMA_VERSION",
    "build_matrix_execution_plan",
    "compile_new_site_task_evaluation_run_v2",
    "execute_policy_scenario_matrix",
    "migrate_v1_request_to_v2",
    "project_v2_result_to_v1",
    "validate_scenario_pack",
]
