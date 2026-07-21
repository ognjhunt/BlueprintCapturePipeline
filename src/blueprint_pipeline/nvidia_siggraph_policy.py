"""Policy registry and stop-rule evaluator for the SIGGRAPH 2026 support lanes."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from .common import read_json_any, sha256_file, utc_now_iso, write_json
from .external_tool_runtime import PUBLIC_CLAIM_UPGRADE_KEY


SOURCE_SNAPSHOT_DATE = "2026-07-21"
POST_CONFERENCE_REFRESH_NOT_BEFORE = "2026-07-24"

COMPONENT_POLICY: dict[str, dict[str, Any]] = {
    "simready_foundation": {
        "role": "advisory_external_validator",
        "default_enabled": False,
        "authoritative_capture_truth": False,
        "production_gate_allowed": False,
        "promotion_requires_calibration": True,
    },
    "ovrtx": {
        "role": "experimental_sensor_preflight",
        "default_enabled": False,
        "prerelease": True,
        "replaces_isaac": False,
    },
    "ovphysx": {
        "role": "experimental_physics_smoke_preflight",
        "default_enabled": False,
        "prerelease": True,
        "replaces_isaac": False,
    },
    "ovstage": {
        "role": "internal_scene_state_dependency_not_standalone_product_lane",
        "default_enabled": False,
        "standalone_adoption_allowed": False,
        "allowed_as_dependency_of": ["ovrtx>=0.4", "ovrtx+ovphysx"],
        "reason": (
            "ovrtx 0.4 deprecates renderer-owned scene APIs and is transitioning "
            "to ovstage; Blueprint still does not expose ovstage as a durable contract"
        ),
    },
    "agent_toolkit": {
        "role": "optional_orchestration_not_durable_contract",
        "default_enabled": False,
        "required_by_blueprint_contracts": False,
        "second_generic_agent_framework_allowed": False,
    },
    "cad_to_simready_skill": {
        "role": "buyer_supplied_asset_conditioning_reference",
        "default_enabled": False,
        "primary_capture_reconstruction_path_allowed": False,
        "staged_validation_and_proposal_reports_only": True,
    },
    "cosmos3_edge": {
        "role": "experimental_distinct_wam_reasoner_profile",
        "default_enabled": False,
        "inherits_cosmos3_nano_sc3_qualification": False,
        "structured_physics_truth": False,
    },
    "usd_convert_gsplat": {
        "role": "optional_conformance_oracle",
        "default_enabled": False,
        "replaces_particlefield_authoring": False,
    },
    "content_agents": {
        "role": "deferred_candidate_asset_enrichment",
        "default_enabled": False,
        "proposal_only": True,
        "human_approval_required": True,
        "raw_capture_upload_allowed": False,
    },
    "simready_blender": {
        "role": "developer_review_tool_only",
        "default_enabled": False,
        "critical_path_allowed": False,
    },
    "artifixer": {
        "role": "optional_generated_support_enhancer",
        "default_enabled": False,
        "held_out_real_view_evaluation_required": True,
        "capture_truth": False,
        "geometry_truth": False,
    },
    "motionbricks_ardy_gpc_newton_research": {
        "role": "watchlist",
        "default_enabled": False,
        "stable_runtime_required_before_integration": True,
    },
    "cosmos_dreams_research": {
        "role": "watchlist",
        "default_enabled": False,
        "stable_runtime_required_before_integration": True,
    },
}

POST_CONFERENCE_REVIEW_SCHEMA = "nvidia_siggraph_post_conference_source_review.v1"


STOP_RULES: tuple[tuple[str, str], ...] = (
    ("component_version_pinned", "component_version_or_revision_not_pinned"),
    ("license_compatible", "license_not_verified_compatible"),
    ("stable_normalized_receipts", "api_churn_prevents_stable_receipts"),
    ("privacy_safe_inputs_only", "raw_or_unredacted_upload_required"),
    ("dependency_isolated", "dependency_conflicts_with_core_environment"),
    ("input_output_digests_preserved", "input_output_digest_or_provenance_missing"),
    ("proof_boundaries_separated", "claim_boundary_cannot_be_preserved"),
    ("useful_failure_class_or_cost_gain", "no_measurable_failure_or_cost_advantage"),
    ("paid_resource_admission_enforced", "paid_resource_admission_not_enforced"),
    ("provider_teardown_provable", "provider_teardown_not_provable"),
)


def evaluate_stop_rules(
    *,
    component: str,
    evidence: Mapping[str, Any],
    require_measured_value: bool = True,
) -> dict[str, Any]:
    if component not in COMPONENT_POLICY:
        raise ValueError(f"unknown NVIDIA component: {component}")
    triggered: list[dict[str, str]] = []
    deferred: list[dict[str, str]] = []
    for field, blocker in STOP_RULES:
        if field == "useful_failure_class_or_cost_gain" and not require_measured_value:
            if evidence.get(field) is not True:
                deferred.append({"field": field, "reason": blocker})
            continue
        if evidence.get(field) is not True:
            triggered.append({"field": field, "reason": blocker})
    return {
        "schema_version": "nvidia_siggraph_stop_rule_evaluation.v1",
        "generated_at": utc_now_iso(),
        "source_snapshot_date": SOURCE_SNAPSHOT_DATE,
        "component": component,
        "status": "stop" if triggered else "advisory" if deferred else "proceed",
        "triggered_stop_rules": triggered,
        "deferred_measurement_rules": deferred,
        "policy": dict(COMPONENT_POLICY[component]),
        PUBLIC_CLAIM_UPGRADE_KEY: False,
    }


def capability_registry() -> dict[str, Any]:
    return {
        "schema_version": "nvidia_siggraph_capability_registry.v1",
        "generated_at": utc_now_iso(),
        "source_snapshot_date": SOURCE_SNAPSHOT_DATE,
        "post_conference_refresh": {
            "required": True,
            "not_before": POST_CONFERENCE_REFRESH_NOT_BEFORE,
            "must_reverify": [
                "source revision or package version",
                "license and binary redistribution terms",
                "API migration notes",
                "Python CUDA driver and platform requirements",
                "production versus prerelease status",
            ],
        },
        "components": {name: dict(policy) for name, policy in COMPONENT_POLICY.items()},
        "durable_contract_owner": "BlueprintCapturePipeline",
        "agent_toolkit_required": False,
        "raw_capture_authority_preserved": True,
        "openusd_is_exchange_substrate_not_capture_truth": True,
    }


def validate_post_conference_source_review(
    payload: Mapping[str, Any], *, as_of_date: str
) -> dict[str, Any]:
    blockers: list[str] = []
    if payload.get("schema_version") != POST_CONFERENCE_REVIEW_SCHEMA:
        blockers.append("post_siggraph_source_review_schema_invalid")
    reviewed_at = str(payload.get("reviewed_at") or "").strip()
    try:
        reviewed_date = date.fromisoformat(reviewed_at[:10])
    except ValueError:
        reviewed_date = date.min
        blockers.append("post_siggraph_source_review_date_invalid")
    if reviewed_date < date.fromisoformat(POST_CONFERENCE_REFRESH_NOT_BEFORE):
        blockers.append("post_siggraph_source_review_too_early")
    if not str(payload.get("reviewer_id") or "").strip():
        blockers.append("post_siggraph_source_review_reviewer_missing")
    raw_components = payload.get("components")
    if not isinstance(raw_components, list):
        raw_components = []
        blockers.append("post_siggraph_source_review_components_missing")
    observed: set[str] = set()
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(raw_components):
        row = dict(value) if isinstance(value, Mapping) else {}
        component = str(row.get("component") or "").strip()
        if component not in COMPONENT_POLICY or component in observed:
            blockers.append(f"post_siggraph_source_review_component_invalid_or_duplicate:{index}")
        observed.add(component)
        for field in (
            "source_url",
            "source_revision_or_package_version",
            "license_id",
            "maturity",
            "decision",
        ):
            if not str(row.get(field) or "").strip():
                blockers.append(
                    f"post_siggraph_source_review_field_missing:{component or index}:{field}"
                )
        if row.get("license_compatible") not in {True, False}:
            blockers.append(
                f"post_siggraph_source_review_license_decision_missing:{component or index}"
            )
        evidence_urls = row.get("evidence_urls")
        if not isinstance(evidence_urls, list) or not all(
            isinstance(item, str) and item.strip() for item in evidence_urls
        ):
            blockers.append(
                f"post_siggraph_source_review_evidence_urls_invalid:{component or index}"
            )
        rows.append(row)
    missing = sorted(set(COMPONENT_POLICY) - observed)
    blockers.extend(f"post_siggraph_source_review_component_missing:{name}" for name in missing)
    as_of = date.fromisoformat(as_of_date)
    due = as_of >= date.fromisoformat(POST_CONFERENCE_REFRESH_NOT_BEFORE)
    return {
        "schema_version": "nvidia_siggraph_post_conference_source_review_validation.v1",
        "generated_at": utc_now_iso(),
        "as_of_date": as_of_date,
        "refresh_due": due,
        "status": "completed" if not blockers else "blocked",
        "component_count": len(rows),
        "expected_component_count": len(COMPONENT_POLICY),
        "blockers": list(dict.fromkeys(blockers)),
        PUBLIC_CLAIM_UPGRADE_KEY: False,
    }


def validate_post_conference_source_review_file(
    path: str | Path, *, as_of_date: str
) -> dict[str, Any]:
    source = Path(path).resolve()
    loaded = read_json_any(source)
    payload = dict(loaded) if isinstance(loaded, Mapping) else {}
    result = validate_post_conference_source_review(payload, as_of_date=as_of_date)
    result["source_path"] = str(source)
    result["source_sha256"] = sha256_file(source) if source.is_file() else None
    return result


def evaluate_component_activation(
    component: str,
    *,
    evidence: Mapping[str, Any],
    as_of_date: str,
) -> dict[str, Any]:
    """Apply component-specific adoption boundaries in addition to stop rules."""

    if component not in COMPONENT_POLICY:
        raise ValueError(f"unknown NVIDIA component: {component}")
    blockers: list[str] = []
    today = date.fromisoformat(as_of_date)
    refresh_due = today >= date.fromisoformat(POST_CONFERENCE_REFRESH_NOT_BEFORE)
    refresh = evidence.get("post_conference_source_review")
    refresh_completed = bool(
        isinstance(refresh, Mapping)
        and refresh.get("schema_version")
        == "nvidia_siggraph_post_conference_source_review_validation.v1"
        and refresh.get("status") == "completed"
        and refresh.get("as_of_date") == as_of_date
    )
    if refresh_due and not refresh_completed:
        blockers.append("post_siggraph_source_version_license_refresh_required")
    if evidence.get("explicit_opt_in") is not True:
        blockers.append("component_requires_explicit_opt_in")
    if component == "ovstage" and evidence.get("standalone_adoption") is True:
        blockers.append("ovstage_standalone_adoption_prohibited")
    if component == "agent_toolkit" and evidence.get("required_framework_dependency") is True:
        blockers.append("agent_toolkit_cannot_be_required_durable_dependency")
    if component == "cad_to_simready_skill":
        for field, blocker in (
            ("buyer_supplied_asset", "cad_to_simready_requires_buyer_supplied_asset"),
            ("staged_validation_evidence", "cad_to_simready_staged_validation_missing"),
            (
                "capture_reconstruction_primary_path",
                "cad_to_simready_cannot_replace_capture_reconstruction",
            ),
        ):
            expected = field != "capture_reconstruction_primary_path"
            if evidence.get(field) is not expected:
                blockers.append(blocker)
    if component == "content_agents":
        for field, blocker in (
            ("specific_buyer_asset_conditioning_need", "content_agents_buyer_need_missing"),
            ("immutable_before_after_evidence", "content_agents_before_after_evidence_missing"),
            ("human_approval", "content_agents_human_approval_missing"),
            ("privacy_safe_inputs_only", "content_agents_raw_capture_upload_prohibited"),
            (
                "physical_metadata_treated_as_proposal",
                "content_agents_physics_authority_prohibited",
            ),
        ):
            if evidence.get(field) is not True:
                blockers.append(blocker)
    if component == "simready_blender" and evidence.get("critical_path") is True:
        blockers.append("simready_blender_critical_path_prohibited")
    if (
        component == "artifixer"
        and evidence.get("held_out_real_view_evaluation_passed") is not True
    ):
        blockers.append("artifixer_held_out_real_view_evaluation_required")
    if component in {"motionbricks_ardy_gpc_newton_research", "cosmos_dreams_research"}:
        for field, blocker in (
            ("stable_runtime", "research_component_stable_runtime_missing"),
            ("license_compatible", "research_component_license_not_verified"),
            ("frozen_fixture", "research_component_frozen_fixture_missing"),
            ("measurable_blueprint_gap", "research_component_no_measurable_blueprint_gap"),
        ):
            if evidence.get(field) is not True:
                blockers.append(blocker)
    return {
        "schema_version": "nvidia_siggraph_component_activation.v1",
        "generated_at": utc_now_iso(),
        "as_of_date": as_of_date,
        "component": component,
        "status": "allowed_experimental" if not blockers else "blocked",
        "blockers": blockers,
        "policy": dict(COMPONENT_POLICY[component]),
        "production_promotion_allowed": False,
        PUBLIC_CLAIM_UPGRADE_KEY: False,
    }


def write_capability_registry(
    output_path: str | Path,
    *,
    as_of_date: str | None = None,
    source_review_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = capability_registry()
    effective_date = as_of_date or date.today().isoformat()
    payload["as_of_date"] = effective_date
    payload["post_conference_refresh"]["validation"] = (
        validate_post_conference_source_review_file(source_review_path, as_of_date=effective_date)
        if source_review_path is not None
        else {
            "schema_version": "nvidia_siggraph_post_conference_source_review_validation.v1",
            "as_of_date": effective_date,
            "refresh_due": date.fromisoformat(effective_date)
            >= date.fromisoformat(POST_CONFERENCE_REFRESH_NOT_BEFORE),
            "status": "not_provided",
            "blockers": ["post_siggraph_source_review_not_provided"],
        }
    )
    write_json(Path(output_path), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the NVIDIA SIGGRAPH capability policy")
    parser.add_argument("--output", required=True)
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--source-review", default=None)
    args = parser.parse_args(argv)
    payload = write_capability_registry(
        args.output,
        as_of_date=args.as_of_date,
        source_review_path=args.source_review,
    )
    print(json.dumps({"status": "completed", "component_count": len(payload["components"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
