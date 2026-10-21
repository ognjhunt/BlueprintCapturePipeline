"""Real producer/validator joins with CPU-only synthetic control measurements."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_policy_rights import CandidatePolicyRightsError
from blueprint_pipeline.adp_episode_evidence_index import EpisodeEvidenceIndexError
from blueprint_pipeline.adp_task_scoring import score_task_episode_from_spec
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.exact_workcell_native_admission import (
    derive_native_cell_policy_admission,
    validate_native_source_policy_candidates,
)
from blueprint_pipeline.exact_workcell_variation_matrix import ExactWorkcellVariationError, REQUIRED_CONTROLS
from blueprint_pipeline.native_task_arena_policy_bundle import (
    ADP009D_POLICY_READINESS_PATH, ADP009D_SCENARIO_SUITE_PATH,
    build_native_task_policy_execution_spec,
)
from blueprint_pipeline.native_task_control_plan import materialize_native_task_control_plan
from blueprint_pipeline.policy_scientific_reset import REQUIRED_CHANNELS, seal_reset_readback
from tests.test_adp009d_episode_evidence_index import _receipt
from tests.test_adp_task_scoring import _articulated_spec, _sample
from tests.test_native_task_arena_bundle import (
    _articulated_packet, _qualified_construction, _qualified_controls,
)


def _inputs(root: Path):
    _, source = _articulated_packet(root)
    source["scene_id"] = "fixture-site"
    source["task_spec"] = {**_articulated_spec(), **source["task_spec"]}
    task = source["task_spec"]
    task["target_joint_id"] = "fixture_hinge"
    task["joint_reset_positions_rad"] = {"fixture_hinge": 0.0, "left_door": 0.0}
    task["joint_hard_limits_rad"] = {"fixture_hinge": [0.0, 1.919862177], "left_door": [-.01, 1.919862177]}
    source["plan_digest"] = canonical_digest(source, digest_field="plan_digest")
    source_path = root / "source_plan.json"
    source_path.write_text(json.dumps(source))
    construction = _qualified_construction(root, source)
    controls = _qualified_controls(root, source, construction)
    specs = {candidate: build_native_task_policy_execution_spec(
        candidate_id=candidate, scene_plan_path=source_path, construction_result_path=construction,
        control_result_path=controls, output_path=root / (candidate + ".json"),
    ) for candidate in ("pi05_droid", "groot_n17_droid")}
    return dict(source_scene_plan=source, candidate_specs=specs,
        candidate_identity_digests={key: value["execution_spec_digest"] for key, value in specs.items()},
        readiness=json.loads(ADP009D_POLICY_READINESS_PATH.read_text()),
        scenario_suite=json.loads(ADP009D_SCENARIO_SUITE_PATH.read_text()))


def _cell_inputs(root: Path):
    inputs = _inputs(root)
    source = inputs["source_scene_plan"]
    cell = copy.deepcopy(source)
    cell["scenario"].update(cell_id="fixture-cell-1", exact_workcell_binding={
        "source_plan_digest": source["plan_digest"], "cell_digest": "sha256:" + "c" * 64,
        "reset_digest": "sha256:" + "d" * 64,
    })
    cell["plan_digest"] = canonical_digest(cell, digest_field="plan_digest")
    construction = json.loads(_qualified_construction(root, cell).read_text())
    control_plan = materialize_native_task_control_plan(scene_plan=cell, construction_result=construction)
    paths = {}
    for control in REQUIRED_CONTROLS:
        episode_id = cell["scenario"]["cell_id"] + "." + control
        path = _receipt(root, episode_id=episode_id, subject_id=control, learned=False)
        receipt = json.loads(path.read_text())
        samples = []
        for index in range(45):
            sample = _sample(index, 1.0 if index > 0 and control == "deterministic_scripted_positive" else 0.0)
            sample["joint_positions_rad"]["fixture_hinge"] = sample["joint_positions_rad"].pop("right_door")
            sample["joint_velocities_rad_s"]["fixture_hinge"] = sample["joint_velocities_rad_s"].pop("right_door")
            samples.append(sample)
        score = score_task_episode_from_spec(task_spec=cell["task_spec"], samples=samples)
        receipt.update(
            task_spec=cell["task_spec"], task_spec_digest=canonical_digest(cell["task_spec"]),
            control_plan_digest=control_plan["plan_digest"], control_passed=True, blockers=[],
            candidate_policy_queried=False, phase_execution_blocker=None, score=score,
            observed_outcome=score["outcome"], state_trace=samples,
            scientific_reset=seal_reset_readback(binding={
                "candidate_id": control, "cell_id": cell["scenario"]["cell_id"], "seed": cell["scenario"]["seed"],
                "task_spec_digest": canonical_digest(cell["task_spec"]), "resolved_scenario_digest": "sha256:" + "e" * 64,
                "matrix_cell_digest": "sha256:" + "c" * 64, "matrix_reset_digest": "sha256:" + "d" * 64,
            }, observed={name: {"synthetic_test_measurement": 1.0} for name in REQUIRED_CHANNELS},
               sources={name: "cpu_test_double" for name in REQUIRED_CHANNELS}, gaps=[]),
        )
        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
        path.write_text(json.dumps(receipt))
        paths[control] = path
    return dict(source_scene_plan=source, source_candidate_specs=inputs["candidate_specs"],
        candidate_identity_digests=inputs["candidate_identity_digests"], readiness=inputs["readiness"],
        scenario_suite=inputs["scenario_suite"], scene_plan=cell, construction_result=construction,
        control_plan=control_plan, control_receipt_paths=paths, media_root=root)


def test_full_authoritative_policy_inputs_accepted_without_a_provider_call(tmp_path):
    inputs = _inputs(tmp_path)
    result = validate_native_source_policy_candidates(**inputs)
    assert result["status"] == "admitted_candidate_inputs"
    assert result["provider_mutation_performed"] is False
    assert result["cell_controls_qualification_inferred"] is False


@pytest.mark.parametrize("authority", ["readiness", "scenario_suite"])
def test_redigested_fake_full_authority_cannot_admit_a_narrow_rights_receipt(tmp_path, authority):
    inputs = _inputs(tmp_path)
    field = "readiness_digest" if authority == "readiness" else "suite_digest"
    inputs[authority]["task_id"] = "forged"
    inputs[authority][field] = canonical_digest(inputs[authority], digest_field=field)
    with pytest.raises(CandidatePolicyRightsError, match="authority_invalid"):
        validate_native_source_policy_candidates(**inputs)


def test_frozen_source_candidate_cannot_change_endpoint_or_query_duration(tmp_path):
    inputs = _inputs(tmp_path)
    spec = inputs["candidate_specs"]["pi05_droid"]
    spec["policy_endpoint"]["port"] += 1
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    with pytest.raises(ExactWorkcellVariationError, match="frozen_candidate_binding_mismatch"):
        validate_native_source_policy_candidates(**inputs)


def test_actual_cell_controls_derive_new_predecessors_with_frozen_model_and_interfaces(tmp_path):
    inputs = _cell_inputs(tmp_path)
    result = derive_native_cell_policy_admission(**inputs)
    assert result["status"] == "admitted_policy_inputs"
    assert result["qualification_authorized"] is False
    for candidate, spec in result["candidate_specs"].items():
        source = inputs["source_candidate_specs"][candidate]
        for field in ("policy_spec", "policy_endpoint", "max_policy_queries", "open_loop_horizon"):
            assert spec[field] == source[field]
        assert spec["scene_plan_digest"] == inputs["scene_plan"]["plan_digest"]
        assert spec["construction_result_digest"] == inputs["construction_result"]["result_digest"]
        assert spec["control_result_digest"] == result["control_result"]["result_digest"]
        assert spec["candidate_rights_binding"]["robot_binding"]["scene_plan_digest"] == inputs["scene_plan"]["plan_digest"]
        assert result["source_candidate_identity_digests"][candidate] == source["execution_spec_digest"]


def test_cell_controls_do_not_accept_a_retained_pass_flag_over_independent_scoring(tmp_path):
    inputs = _cell_inputs(tmp_path)
    path = inputs["control_receipt_paths"]["zero_action_negative"]
    receipt = json.loads(path.read_text())
    receipt["score"]["task_succeeded"] = True
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    path.write_text(json.dumps(receipt))
    with pytest.raises(ExactWorkcellVariationError, match="control_score_not_admitted"):
        derive_native_cell_policy_admission(**inputs)


def test_cell_controls_verify_retained_media_bytes_before_policy_admission(tmp_path):
    inputs = _cell_inputs(tmp_path)
    receipt = json.loads(inputs["control_receipt_paths"]["zero_action_negative"].read_text())
    artifact = next(row for row in receipt["media_artifacts"] if row["role"] == "camera_review_video")
    (tmp_path / artifact["relative_path"]).write_bytes(b"tampered")
    with pytest.raises(EpisodeEvidenceIndexError, match="digest_mismatch"):
        derive_native_cell_policy_admission(**inputs)


def test_another_cells_construction_is_not_admitted_by_copying_a_pass_flag(tmp_path):
    inputs = _cell_inputs(tmp_path)
    construction = inputs["construction_result"]
    construction["scene_plan_digest"] = inputs["source_scene_plan"]["plan_digest"]
    construction["result_digest"] = canonical_digest(construction, digest_field="result_digest")
    with pytest.raises(ValueError, match="construction_not_qualified"):
        derive_native_cell_policy_admission(**inputs)
