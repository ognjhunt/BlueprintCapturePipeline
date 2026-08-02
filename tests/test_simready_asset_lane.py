from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.simready_asset_lane import (
    PROVIDER_R2_GATES,
    SimReadyAssetLaneError,
    build_simready_asset_request,
    compose_simready_scene_binding,
    generate_simready_asset_draft,
    merge_simready_candidate_evidence,
    plan_external_simready_generation,
    preflight_simready_scene,
    probe_simready_preflight_toolchain,
    validate_simready_asset_manifest,
)
from blueprint_pipeline.task_site_measurement_routing import (
    ALL_CAPABILITY_FIELDS,
    derive_task_measurement_requirements,
    route_task_site_measurement,
    validate_measurement_qualification,
    validate_method_capability_profile,
    validate_site_evidence_profile,
)


SHA_A = "sha256:" + "a" * 64

# A unit-ish box mesh (12 triangles) standing in for a segmented task object.
_BOX_VERTICES = [
    [-0.05, -0.05, 0.0], [0.05, -0.05, 0.0], [0.05, 0.05, 0.0], [-0.05, 0.05, 0.0],
    [-0.05, -0.05, 0.1], [0.05, -0.05, 0.1], [0.05, 0.05, 0.1], [-0.05, 0.05, 0.1],
]
_BOX_FACES = [
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
    [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
]
_BOX_VOLUME = 0.1 * 0.1 * 0.1


def _request(*, mode: str = "local_geometry_pipeline", provider: str = "") -> dict:
    return build_simready_asset_request(
        request_id="simready-request-mug",
        object_id="counter-mug",
        bundle_id="capture-1",
        bundle_hash=SHA_A,
        source_references={
            "segmentation_record_id": "record-object_segmentation-mug",
            "mesh_record_id": "record-mesh-mug",
            "splat_record_id": "record-gaussian_splat_appearance",
            "provenance_record_id": "provenance-fixture",
        },
        density_class="ceramic_glass",
        generation_mode=mode,
        provider_candidate_id=provider,
    )


def _draft() -> dict:
    return generate_simready_asset_draft(
        _request(),
        vertices=_BOX_VERTICES,
        faces=_BOX_FACES,
        generated_on="2026-08-02",
    )


def _site(extra_evidence: dict | None = None) -> dict:
    evidence = {
        "metric_scale": {
            "available": True, "validated": True, "record_id": "record-metric_scale",
        },
        "robot_site_registration": {
            "available": True, "validated": True,
            "record_id": "record-robot_site_registration",
        },
        "gaussian_splat_appearance": {
            "available": True, "validated": True,
            "record_id": "record-gaussian_splat_appearance",
        },
        "object_segmentation": {
            "available": True, "validated": True,
            "record_id": "record-object_segmentation",
        },
    }
    evidence.update(extra_evidence or {})
    return validate_site_evidence_profile({
        "schema_version": "site_evidence_profile.v1",
        "profile_id": "simready-site-v1",
        "bundle_id": "capture-1",
        "bundle_hash": SHA_A,
        "provenance_record_id": "provenance-fixture",
        "rights": {"commercial_evaluation_allowed": True},
        "privacy": {"external_processing_allowed": False},
        "coordinate_system": {"metric_scale_verified": True},
        "evidence": evidence,
        "limitations": {"known_missing_regions": [], "forbidden_claims": []},
    })


def test_checked_json_schema_accepts_each_simready_contract() -> None:
    schema = json.loads(
        (
            Path(__file__).parents[1] / "docs/schemas/simready_asset_lane.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    request = _request()
    draft = _draft()
    plan = plan_external_simready_generation(
        _request(mode="external_provider", provider="nvidia-usd-content-agents")
    )
    binding = compose_simready_scene_binding(_site(), [draft])
    toolchain = probe_simready_preflight_toolchain()
    preflight = preflight_simready_scene([draft], settle_steps=10)
    for artifact in (request, draft, plan, binding, toolchain, preflight):
        jsonschema.validate(artifact, schema)


def test_preflight_toolchain_probe_records_optional_agentic_validator_absence() -> None:
    probe = probe_simready_preflight_toolchain()
    assert probe["current_required_toolchain_available"] is True
    assert probe["install_performed"] is False
    assert probe["network_used"] is False
    assert probe["provider_call_performed"] is False
    for tool_id in ("blender_headless_validation", "nvidia_content_agent_validation"):
        row = probe["tools"][tool_id]
        assert isinstance(row["available"], bool)
        assert row["required_for_current_preflight"] is False


def test_local_draft_generates_real_geometry_estimates_and_flags() -> None:
    draft = _draft()
    geometry = draft["geometry"]
    assert geometry["watertight"] is True
    assert abs(geometry["volume_m3"] - _BOX_VOLUME) < 1e-9
    mass = draft["mass_estimate"]
    assert mass["estimated"] is True
    assert abs(mass["value_kg"] - _BOX_VOLUME * 2300.0) < 1e-6
    assert draft["friction_estimate"]["estimated"] is True
    assert draft["generator"]["decomposition_method"].startswith("convex_hull_fallback") or (
        draft["generator"]["decomposition_method"] == "vhacd"
    )
    assert draft["validation"]["validated"] is False
    assert draft["physics_authority_granted"] is False
    for row in draft["candidate_site_evidence"]:
        assert row["validated"] is False
    usd_export = draft["exports"]["usd"]
    assert usd_export["content"] is not None and "counter_mug" in usd_export["content"]


def test_mjcf_export_loads_and_simulates_in_real_mujoco() -> None:
    mujoco = pytest.importorskip("mujoco")
    draft = _draft()
    model = mujoco.MjModel.from_xml_string(draft["exports"]["mjcf"]["content"])
    data = mujoco.MjData(model)
    for _ in range(50):
        mujoco.mj_step(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "counter-mug")
    assert body_id >= 0
    assert abs(float(model.body_mass[body_id]) - draft["mass_estimate"]["value_kg"]) < 1e-6
    # Free fall under gravity with no floor: the draft actually simulates.
    assert float(data.qpos[2]) < 0.0


def test_manifest_tampering_toward_authority_fails_closed() -> None:
    draft = _draft()
    for mutation in (
        ("physics_authority_granted", True),
        ("routing_eligibility_granted", True),
        ("execution_authorized", True),
    ):
        tampered = copy.deepcopy(draft)
        tampered.pop("simready_asset_digest")
        tampered[mutation[0]] = mutation[1]
        with pytest.raises(SimReadyAssetLaneError, match="must_be_false"):
            validate_simready_asset_manifest(tampered)
    unflagged = copy.deepcopy(draft)
    unflagged.pop("simready_asset_digest")
    unflagged["mass_estimate"]["estimated"] = False
    with pytest.raises(SimReadyAssetLaneError, match="must_be_flagged_estimated"):
        validate_simready_asset_manifest(unflagged)
    upgraded = copy.deepcopy(draft)
    upgraded.pop("simready_asset_digest")
    upgraded["candidate_site_evidence"][0]["validated"] = True
    with pytest.raises(SimReadyAssetLaneError, match="must_be_unvalidated"):
        validate_simready_asset_manifest(upgraded)


def test_external_provider_generation_is_planned_behind_r2_gates_without_network() -> None:
    request = _request(mode="external_provider", provider="nvidia-usd-content-agents")
    blocked = plan_external_simready_generation(request)
    assert blocked["status"] == "blocked_r2_gates_unresolved"
    assert blocked["unresolved_r2_gates"] == list(PROVIDER_R2_GATES)
    assert blocked["live_call_performed"] is False
    assert blocked["network_used"] is False
    assert blocked["agent_may_resolve_gates"] is False
    partially = plan_external_simready_generation(
        request,
        resolved_gates={"commercial_use_terms": "contract://msa-draft-7"},
    )
    assert partially["status"] == "blocked_r2_gates_unresolved"
    assert "data_retention_terms" in partially["unresolved_r2_gates"]
    resolved = plan_external_simready_generation(
        request,
        resolved_gates={gate: f"contract://{gate}" for gate in PROVIDER_R2_GATES},
    )
    assert resolved["status"] == "gates_resolved_pending_r3_adapter_admission"
    assert resolved["provider_output_would_enter_evidence_validated"] is False
    with pytest.raises(SimReadyAssetLaneError, match="simready_provider_unknown"):
        plan_external_simready_generation(
            _request(mode="external_provider", provider="unknown-provider")
        )


def test_merged_candidate_evidence_makes_router_demand_collider_validation() -> None:
    site = _site()
    merged = merge_simready_candidate_evidence(site, [_draft()])
    for evidence_id in ("validated_collider", "mass_inertia", "friction_contact"):
        record = merged["evidence"][evidence_id]
        assert record["available"] is True
        assert record["validated"] is False
    # A fully capable, fixture-qualified rigid engine still cannot route:
    # SimReady candidates are present-but-unvalidated evidence.
    requirements = derive_task_measurement_requirements(
        {"claim_id": "pick-the-mug", "claim_type": "collision_contact"},
        {"task_distribution": {"measurement_task_class": "rigid_pick_place"}},
    )
    caps = set(requirements["required_capabilities"])
    values: dict = {field: False for field in ALL_CAPABILITY_FIELDS}
    for field in (
        "plugin_versions", "robot_model_formats", "supported_embodiments",
        "supported_end_effectors", "action_representation_types",
        "qualification_record_ids", "qualified_task_classes",
        "qualified_material_regimes", "qualified_robot_ids",
        "qualified_end_effector_ids", "qualified_controller_ids",
        "qualified_sensor_ids", "qualified_site_classes", "qualified_metric_ids",
        "known_failure_modes", "prohibited_extrapolations", "asset_license_ids",
        "model_license_ids", "subprocessor_regions", "output_formats",
    ):
        values[field] = []
    values.update({
        "method_id": "fixture-rigid-engine", "method_family": "traditional_simulation",
        "version": "1", "release_date": "2026-08-01", "commit_hash": "fixture",
        "container_digest": SHA_A, "solver_backend": "fixture",
        "numeric_precision": "float64", "deterministic_mode": "strict",
        "operating_system": "linux", "gpu_model": "none", "driver_version": "none",
        "random_seed_policy": "frozen", "contact_formulation": "fixture",
        "maximum_control_rate_hz": 1000, "qualified_parameter_ranges": {},
        "qualified_claim_ceiling": "C3", "qualification_expiration": "2027-08-01",
        "harmful_false_negative_bound": 0.01, "maximum_latency_class": "interactive",
        "maximum_compute_class": "cpu", "estimated_cost_class": "low",
        "data_retention_days": 0, "source_available": True,
        "local_offline_supported": True, "commercial_use_allowed": True,
        "provider_training_use_allowed": False, "deletion_right_supported": True,
        "output_export_supported": True,
    })
    for field in caps:
        values[field] = True
    profile = validate_method_capability_profile({
        "schema_version": "method_capability_profile.v1",
        "method_id": "fixture-rigid-engine",
        "capabilities": values,
        "evidence_quality": {"source": "development_fixture"},
        "expected_cost_usd": 1.0,
        "expected_latency_seconds": 1.0,
    })
    qualification = validate_measurement_qualification({
        "schema_version": "measurement_qualification_record.v1",
        "qualification_id": "fixture-development-qualification",
        "method_id": "fixture-rigid-engine",
        "method_version": "1",
        "capability_profile_digest": profile["capability_profile_digest"],
        "admission_record_digest": SHA_A,
        "admission_stage": "R7",
        "status": "approved",
        "qualified_capabilities": sorted(caps),
        "claim_ceiling": "C3",
        "scope": {
            "task_classes": ["rigid_pick_place"], "material_regimes": ["none"],
            "robot_ids": [], "end_effector_ids": [], "controller_ids": [],
            "sensor_ids": [], "metric_ids": [], "parameter_ranges": {},
        },
        "metrics": {
            "physical_accuracy_error": 0.01, "uncertainty": 0.02,
            "scope_distance": 0.0, "harmful_false_negative_rate": 0.001,
            "reproducibility_score": 1.0, "privacy_preference": 1.0,
        },
        "approval": {
            "signature_status": "verified",
            "signature_id": "development-fixture-signature",
            "approved_by": ["benchmark-owner", "independent-reviewer"],
            "agent_approved": False,
        },
        "expiration_date": "2027-08-01",
        "self_grading": False,
    })
    decision = route_task_site_measurement(
        requirements, merged, [profile], [qualification],
        catalog_snapshot_hash=SHA_A, as_of=date(2026, 8, 2),
    )
    assert decision["status"] == "abstention"
    action = decision["abstention"]["smallest_next_action"]
    assert action["action_type"] == "collider_validation"
    assert "validated_collider" in action["exact_scope"]


def test_scene_binding_keeps_3dgs_appearance_and_grants_no_authority() -> None:
    site = _site()
    draft = _draft()
    binding = compose_simready_scene_binding(site, [draft])
    assert binding["appearance_stays_3dgs"] is True
    assert binding["appearance_layer"]["available"] is True
    assert binding["appearance_layer"]["record_id"] == (
        "record-gaussian_splat_appearance"
    )
    assert binding["physics_authority_granted"] is False
    assert binding["world_model_physics_used"] is False
    slot = binding["object_slots"][0]
    assert slot["object_id"] == "counter-mug"
    assert slot["physics_source"] == "simready_draft_unvalidated"
    with pytest.raises(SimReadyAssetLaneError, match="duplicate_object"):
        compose_simready_scene_binding(site, [draft, draft])
    mismatched = copy.deepcopy(draft)
    mismatched.pop("simready_asset_digest")
    mismatched["bundle_hash"] = "sha256:" + "b" * 64
    mismatched = validate_simready_asset_manifest(mismatched)
    with pytest.raises(SimReadyAssetLaneError, match="bundle_binding_mismatch"):
        merge_simready_candidate_evidence(site, [mismatched])
