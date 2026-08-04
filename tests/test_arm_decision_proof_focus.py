"""Contract tests for Blueprint's sole active Arm Decision Proof program."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parents[1]
PROGRAM_ROOT = ROOT / "docs" / "arm_decision_proof_v1"
CONTRACT_PATH = PROGRAM_ROOT / "north_star_contract.json"
SCHEMA_PATH = ROOT / "docs" / "schemas" / "arm_decision_proof_north_star.v2.schema.json"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_north_star_contract_is_schema_valid() -> None:
    schema = _read_json(SCHEMA_PATH)
    contract = _read_json(CONTRACT_PATH)

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(contract)


def test_north_star_contract_preserves_proof_and_compatibility_boundaries() -> None:
    contract = _read_json(CONTRACT_PATH)

    assert contract["status"] == "sole_active_program"
    assert contract["execution_strategy"] == {
        "mode": "public_evidence_ladder_then_fresh_site_proof",
        "completed_precursor": "ADP-008_simpler_decision_harness",
        "active_item": "ADP-009_public_scene_qualification",
        "public_scene_qualification": "dominant_until_digest_bound_matrix_complete",
        "fresh_site_capture": (
            "mandatory_final_construction_input_after_public_qualification_and_protocol_freeze"
        ),
        "partner_recruitment_and_protocol": "parallel_human_lane",
    }
    assert contract["north_star_metric"] == {
        "name": "prospectively_physically_validated_new_site_task_decisions",
        "current": 0,
        "target": 1,
    }
    assert contract["development_substrates"]["claim_ceiling"] == "development_only"
    assert contract["development_substrates"]["historical_public_decision_reference"] == (
        "SIMPLER"
    )
    assert "sim_to_real_decision_fidelity" in contract["development_substrates"][
        "cannot_qualify"
    ]
    assert contract["compatibility"] == {
        "preserve_readers": True,
        "preserve_historical_evidence": True,
        "historical_adp_008_digest_unchanged": True,
        "allow_unrelated_new_work": False,
    }


def test_public_evidence_ladder_requires_new_datasets_editing_and_fresh_capture() -> None:
    contract = _read_json(CONTRACT_PATH)
    ladder = {row["id"]: row for row in contract["public_evidence_ladder"]}

    assert [row["order"] for row in contract["public_evidence_ladder"]] == list(range(10))
    assert ladder["released_inpainting_author_smokes"]["substrate"] == (
        "exact_Inpaint360GS_InFusion_and_AuraFusion360_revisions_with_author_inputs"
    )
    assert ladder["real_metrology_transfer"]["substrate"] == (
        "one_admitted_ScanNet++_scene"
    )
    assert ladder["real_metrology_transfer"]["status"] == (
        "required_after_access_and_terms_acceptance"
    )
    assert ladder["synthetic_hybrid_scene_control"]["substrate"] == (
        "matched_InteriorGS_and_SAGE-3D_scene"
    )
    assert ladder["synthetic_hybrid_scene_control"]["status"] == (
        "required_after_explicit_rights"
    )
    assert ladder["controlled_known_background_recovery"]["status"] == "required"
    assert ladder["fresh_site_acquisition_and_prospective_proof"]["status"] == (
        "mandatory_final_acquisition_phase"
    )
    assert ladder["simready_authoring_backend_bakeoff"]["substrate"].endswith(
        "NVIDIA_USD_Content_Agents_v0.5.2"
    )

    edit = contract["scene_edit_contract"]
    assert edit["released_reproducibility_control"].startswith(
        "Inpaint360GS_exact_revision"
    )
    assert edit["primary_released_interface_adapter"].startswith(
        "InFusion_exact_revision"
    )
    assert edit["released_360_quality_challenger"].startswith(
        "AuraFusion360_exact_revision"
    )
    assert "virtual_cameras" in edit["render_derived_observation_rule"]
    assert "splat_rendered_depth_or_disparity_inputs" in edit[
        "render_derived_observation_rule"
    ]
    assert "clean_background_truth_out_of_method_inputs" in edit[
        "render_derived_observation_rule"
    ]
    assert "without_running_an_unscaled_mapper" in edit[
        "render_derived_observation_rule"
    ]
    assert "not_the_primary_Blueprint_interface_adapter" in edit[
        "inpaint360gs_interface_rule"
    ]
    assert "supplemental_world_aligned_Gaussians" in edit["render_derived_edit_identity"]
    assert "preserve_spherical_harmonic_degree" in edit[
        "primary_adapter_required_patch"
    ]
    assert "TRACE" in edit["paper_only_exclusions"]
    assert "CoIn" not in edit["paper_only_exclusions"]
    assert any(item.startswith("CoIn_") for item in edit["released_noncritical_methods"])
    assert "fresh_site_clean_background_factual_recovery" in edit[
        "ground_truth_levels"
    ]
    assert contract["scope"]["replacement_asset"] == (
        "simready_usd_with_distinct_visual_and_collision_geometry"
    )
    assert any(
        item.startswith("render_derived_camera_bundles_are_labeled_synthetic")
        for item in contract["acceptance"]
    )
    assert "ARKitScenes" not in json.dumps(contract)
    assert "WildRGB-D" not in json.dumps(contract)


def test_metric_and_editing_claim_authority_stays_separate() -> None:
    contract = _read_json(CONTRACT_PATH)
    representation = contract["representation_contract"]

    assert representation["appearance"].startswith("registered_3dgs")
    assert representation["static_collision"].startswith("separate_simplified_mesh")
    assert "gaussian_extent_or_covariance" in representation[
        "non_authoritative_aids"
    ]
    assert "da3_depth_or_pose_cross_check" in representation[
        "non_authoritative_aids"
    ]
    assert representation["hidden_surface_rule"].endswith("never_factual_recovery")
    assert contract["method_admission"]["paper_only_rule"] == (
        "inadmissible_for_critical_path"
    )
    assert contract["method_admission"]["generative_editing_rule"].startswith(
        "mandatory_released_baseline"
    )
    authoring = contract["asset_authoring_contract"]
    assert authoring["provider_interface"] == "replaceable_simready_authoring_backend"
    assert authoring["required_control"] == "known_good_manually_authored_usd"
    assert "not_image_to_cad" in authoring["candidate_limit"]


def test_public_scene_day_gates_bind_the_complete_editing_rehearsal() -> None:
    contract = _read_json(CONTRACT_PATH)
    gates = {
        row["day"]: row
        for row in contract["gates"]
        if row["phase"] == "public_scene_qualification"
    }

    assert set(gates) == {7, 14, 21, 28}
    assert "suite index" in gates[7]["required_outcome"]
    assert "NVIDIA Content Agents" in gates[7]["required_outcome"]
    assert "InFusion world-frame supplemental-PLY adapter" in gates[14][
        "required_outcome"
    ]
    assert "AuraFusion360 multiview challenger" in gates[14]["required_outcome"]
    assert "without claiming measurement-authoritative local surface geometry" in gates[
        14
    ]["required_outcome"]
    assert "appearance and matching collision" in gates[14]["required_outcome"]
    assert "controlled clean-background truth" in gates[21]["required_outcome"]
    assert "ScanNet++ real measured transfer" in gates[21]["required_outcome"]
    assert "exact replacement receipts" in gates[28]["required_outcome"]
    assert "two-candidate Task Evaluation Run" in gates[28]["required_outcome"]


def test_canonical_active_documents_point_to_the_same_program() -> None:
    required_references = {
        ROOT / "AGENTS.md": "arm-decision-proof-v1",
        ROOT / "README.md": "Arm Decision Proof v1",
        ROOT / "PLATFORM_CONTEXT.md": "Arm Decision Proof v1",
        ROOT / "WORLD_MODEL_STRATEGY_CONTEXT.md": "Arm Decision Proof v1",
        ROOT / "VISION.md": "Arm Decision Proof v1",
        ROOT / "CLAUDE.md": "Arm Decision Proof v1",
        ROOT / "docs" / "README.md": "Arm Decision Proof v1",
        ROOT / "docs" / "architecture" / "ai-onboarding-map.md": "Arm Decision Proof v1",
    }

    for path, marker in required_references.items():
        assert marker in path.read_text(encoding="utf-8"), f"{path} lost the focus marker"


def test_master_goal_carries_scope_and_authority_guards() -> None:
    prompt = (PROGRAM_ROOT / "MASTER_GOAL_PROMPT.md").read_text(encoding="utf-8")

    required_phrases = (
        "SOLE-FOCUS TEST",
        "USE PUBLIC DATASETS AND RELEASED CODE BEFORE FRESH CAPTURE",
        "ENGINEERING ALLOCATION",
        "Every reused asset is `development_only`",
        "No paid compute, provider job/upload",
        "Two candidates do not establish rank correlation",
        "Do not start or expand humanoids/G1",
        "Inpaint360GS",
        "NVIDIA USD Content Agents",
        "SimReady USD",
        "ADP-009",
    )
    for phrase in required_phrases:
        assert phrase in prompt
