from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_policy_canary_control_gate import CONTROL_IDS, _file, execute_native_controls
from blueprint_pipeline.native_policy_canary_matrix_gate import execute_strict_matrix, RESULT_FILENAME


def _matrix_case(tmp_path, *, failed_control=None, failed_pair=False, failed_transfer=False, stage="all"):
    inputs = {"candidate_ids": ["pi05_droid", "groot_n17_droid"],
              "runtime_inputs_digest": "sha256:"+"1"*64, "task_success_contract": {},
              "task_success_contract_digest": "sha256:"+"2"*64,
              "cells": [{"cell_id": f"cell-{i}", "seed": i+1} for i in range(10)]}
    order = []

    def spawn(*, index, runtime_root, output_root, child_root, controls_only=False):
        order.append(("control" if controls_only else "policy", index))
        evidence = child_root / "evidence.json"
        evidence.write_text(json.dumps({"index": index}))
        files = [_file(evidence, child_root)]
        if controls_only:
            controls = [{"control_id": key, "control_passed": index != failed_control} for key in CONTROL_IDS]
            receipt = {"cell_id": f"cell-{index}", "seed": index+1,
                       "status": "blocked" if index == failed_control else "passed", "controls": controls,
                       "files": files, "receipt_digest": ""}
            receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
            (child_root / "policy_canary_cell_controls.v1.json").write_text(json.dumps(receipt))
        else:
            episodes = [{"candidate_id": candidate, "cell_id": f"cell-{index}", "seed": index+1,
                         "status": "completed", "candidate_policy_queried": True,
                         "embodiment_parity_diagnostic": {"status": "passed"},
                         "episode": {"score": {"status": "undetermined" if failed_pair else "scored",
                                                "measurements": {"destination_pose_readback_complete": True}},
                                     "visual_evidence": {"status": "complete"}},
                         "evidence_artifacts": {key: files[0] for key in ("frame_manifest", "review_video", "action_sequence", "state_trace")}}
                        for candidate in inputs["candidate_ids"]]
            (child_root / RESULT_FILENAME).write_text(json.dumps({"episodes": episodes, "candidate_policy_queried": True}))
        return 0

    def transfer(**kwargs):
        order.append(("transfer", 0))
        return {"status": "failed" if failed_transfer else "uploaded_and_readback_verified"}

    def aggregate(**kwargs):
        return {"status": "runtime_completed_unqualified_pending_closeout",
                "episodes": [row for child in kwargs["child_results"] for row in child["episodes"]]}

    def seal(*, result_path, result):
        result["result_digest"] = canonical_digest(result, digest_field="result_digest")
        result_path.write_text(json.dumps(result))

    code = execute_strict_matrix(runtime=tmp_path, output_root=tmp_path, inputs=inputs,
                                 authority={"authority_digest": "sha256:"+"3"*64}, spawn=spawn, aggregate=aggregate, seal=seal,
                                 construction_lineage_mode="fixture", transfer_pair=transfer, stage=stage)
    result_path = tmp_path / ("policy_canary_controls_prerequisite.v1.json" if stage == "controls" and code == 0 else RESULT_FILENAME)
    return code, order, json.loads(result_path.read_text())


def test_all_twenty_controls_precede_policy_then_pair_is_transferred_before_other_nine(tmp_path):
    code, order, result = _matrix_case(tmp_path)
    assert code == 0
    assert order[:10] == [("control", i) for i in range(10)]
    assert order[10:12] == [("policy", 0), ("transfer", 0)]
    assert order[12:] == [("policy", i) for i in range(1, 10)]
    assert result["control_episode_count"] == 20
    assert len(result["episodes"]) == 20
    assert result["controls_gate"]["status"] == "passed"


def test_any_control_failure_seals_evidence_and_queries_no_policy(tmp_path):
    code, order, result = _matrix_case(tmp_path, failed_control=2)
    assert code == 1
    assert order == [("control", i) for i in range(3)]
    assert result["candidate_policy_queried"] is False
    assert result["control_episode_count"] == 6
    assert result["unexecuted_learned_episode_count"] == 20
    assert result["artifact_inventory"]


@pytest.mark.parametrize("reason", ["score", "transfer"])
def test_paired_infrastructure_failure_seals_both_episodes_and_never_starts_other_nine(tmp_path, reason):
    code, order, result = _matrix_case(tmp_path, failed_pair=reason == "score", failed_transfer=reason == "transfer")
    assert code == 1
    assert [row for row in order if row[0] == "policy"] == [("policy", 0)]
    assert result["control_episode_count"] == 20
    assert len(result["episodes"]) == 2
    assert result["strict_gate_blockers"]


@pytest.mark.parametrize("force_miss", [False, True])
def test_cell_control_gate_reuses_real_control_runner_and_native_phase_targets(tmp_path, monkeypatch, force_miss):
    from blueprint_pipeline import adp009d_control_episode as runner
    from blueprint_pipeline import native_task_arena_controls_worker as worker
    from tests.test_native_task_control_plan import _rigid_scene, _rigid_construction, _RigidControlEnvironment

    scene = _rigid_scene(scene_id="fixture", asset_id="subject")
    construction = _rigid_construction(scene)
    environment = _RigidControlEnvironment(scene=scene, construction=construction)
    if force_miss:
        read_sample = environment.read_object_sample
        def shifted_readback():
            row = read_sample()
            row["grasp_frame_position_world_m"][0] += 0.1
            return row
        environment.read_object_sample = shifted_readback
    monkeypatch.setattr(runner, "_persist_observation", lambda *args, **kwargs: {"observation_index": 0, "kind": kwargs["kind"], "views": {}})
    monkeypatch.setattr(runner, "finalize_manipulation_evaluation_visual_evidence", lambda **kwargs: ({"status": "complete", "required_camera_ids": ["external", "wrist", "overview"]}, []))
    seen = []

    def solve(**kwargs):
        seen.extend(row["target_position_world_m"] for row in kwargs["control_plan"]["scripted_positive_actions"])
        return [], {"status": "all_unique_poses_solved_or_bound"}

    monkeypatch.setattr(worker, "_control_plan_global_ik_joint_targets", solve)
    native = SimpleNamespace(unwrapped=SimpleNamespace(scene={"robot": object()}), reset=lambda **kwargs: None)
    runtime = SimpleNamespace(
        gripper_probe=lambda **kwargs: {"status": "measured", "open_command": 0., "closed_command": 1.},
        make_servo=lambda **kwargs: object(), make_rigid_task_readback=lambda built: object(),
        build_episode_environment=lambda **kwargs: (environment, {}),
        wrap_rigid_scoring_environment=lambda **kwargs: kwargs["environment"], to_tensor=lambda value: value)
    receipt = execute_native_controls(cell_runtime=runtime, built=SimpleNamespace(env=native), scene_plan=scene,
                                      gate={"policy_observation_integrity_passed": True, "snapshot": {"cameras": [
                                          {"role": role, "observability": {"thresholds": {"effective_minimum_pixels": 4}},
                                           "semantic_label_pixels": {"task_object": 20, "task_support": 20}}
                                          for role in ("external", "wrist", "overview")]}}, output_root=tmp_path)
    if force_miss:
        assert receipt["status"] == "blocked"
        positive = next(row for row in receipt["controls"] if row["control_id"] == CONTROL_IDS[1])
        assert positive["control_passed"] is False
        assert max(row["attempt"] for row in positive["phase_arrivals"]) == 1
        assert positive["environment_steps"] <= scene["task_spec"]["maximum_action_steps"]
        return
    assert receipt["status"] == "passed", receipt["blockers"]
    assert seen == [row["position_world_m"] for row in construction["construction_phase_plan"]["phases"]]
    assert [row["control_id"] for row in receipt["controls"]] == list(CONTROL_IDS)
    assert all(row["control_passed"] for row in receipt["controls"])
    assert all(row["state_trace"] and row["action_trace"] and row["score"] for row in receipt["controls"])


def test_explicit_control_contract_cannot_load_policies_with_only_camera_gate(tmp_path):
    from copy import deepcopy
    from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
    from blueprint_pipeline.native_task_arena_policy_canary_session import execute_paired_session
    from tests.test_native_task_arena_policy_canary_session import _preload_gate_session_kwargs

    calls = {"open": 0, "close": 0, "loads": []}
    kwargs = _preload_gate_session_kwargs(tmp_path, calls)
    inputs = deepcopy(kwargs["runtime_inputs"])
    contract = inputs["task_success_contract"]
    contract["criteria"]["controls"] = {"mode": "required_per_cell", "control_ids": list(CONTROL_IDS)}
    contract["provenance"].update(author_source="task_owner", author_id="fixture", confirmed_by_team_id="fixture-team")
    contract["contract_digest"] = cross_runtime_canonical_digest(contract, digest_field="contract_digest")
    inputs["task_success_contract_digest"] = contract["contract_digest"]
    for cell in inputs["cells"]:
        cell["control_diagnostic"].update(mode="required_before_policy", policy_execution_blocked=True)
    inputs["runtime_inputs_digest"] = canonical_digest(inputs, digest_field="runtime_inputs_digest")
    authority = deepcopy(kwargs["authority"])
    authority.update(runtime_inputs_digest=inputs["runtime_inputs_digest"], task_success_contract_digest=contract["contract_digest"])
    authority["authority_digest"] = canonical_digest(authority, digest_field="authority_digest")
    kwargs.update(runtime_inputs=inputs, authority=authority)
    result = execute_paired_session(**kwargs, prepolicy_observation_gate=lambda session: {"policy_observation_integrity_passed": True})
    assert result["status"] == "blocked"
    assert result["policy_loads"] == []
    assert calls["loads"] == []


def test_strict_camera_gate_requires_book_and_tray_in_each_native_view():
    from blueprint_pipeline.native_policy_canary_control_gate import validate_strict_camera_gate

    gate = {"policy_observation_integrity_passed": True, "snapshot": {"cameras": [
        {"role": role, "observability": {"thresholds": {"effective_minimum_pixels": 8}},
         "semantic_label_pixels": {"task_object": 20, "task_support": 20}}
        for role in ("external", "wrist", "overview")]}}
    validate_strict_camera_gate(gate)
    gate["snapshot"]["cameras"][1]["semantic_label_pixels"]["task_support"] = 0
    with pytest.raises(RuntimeError, match="subject_destination_visibility_failed:wrist"):
        validate_strict_camera_gate(gate)


def test_controls_preprovision_stage_then_policy_stage_reuses_same_verified_controls(tmp_path):
    code, order, prerequisite = _matrix_case(tmp_path, stage="controls")
    assert code == 0 and prerequisite["status"] == "passed"
    assert order == [("control", i) for i in range(10)]
    assert not (tmp_path / RESULT_FILENAME).exists()
    code, order, result = _matrix_case(tmp_path, stage="policies")
    assert code == 0
    assert order[:2] == [("policy", 0), ("transfer", 0)]
    assert len([item for item in order if item[0] == "policy"]) == 10
    assert not any(item[0] == "control" for item in order)
    assert result["control_episode_count"] == 20
