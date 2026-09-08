from copy import deepcopy
from types import SimpleNamespace

import pytest

from blueprint_pipeline.policy_paired_summary import paired_summary
from blueprint_pipeline.policy_interface_binding import resolve_policy_interface
from blueprint_pipeline.exact_workcell_variation_runtime import (
    apply_runtime_cell, collect_evaluation_evidence, compile_evaluation_schedule,
    compile_isaac_lab_event_plan, compile_variation_matrix,
)
from tests.test_exact_workcell_variation_matrix import _request, _schedule_request

CANDIDATES = ["pi05_droid", "groot_n17_droid"]


def episodes():
    rows = []
    for cell in range(10):
        for index, candidate in enumerate(CANDIDATES):
            valid = cell < 6 if index == 0 else cell < 2 or cell >= 6
            win = cell < 2 if index == 0 else cell >= 6
            rows.append({"candidate_id": candidate, "cell_id": f"c{cell}", "seed": cell,
                "family": "canonical_anchor" if cell < 2 else "camera_sensor", "partition": "held_out" if cell >= 8 else "diagnosis",
                "status": "completed", "policy_outcome_interpretable": valid,
                "candidate_policy_queried": True,
                "scoring_authority": "deterministic_simulator_state",
                "episode": {"score": {"status": "scored" if valid else "undetermined", "task_succeeded": win if valid else None}}})
    return rows


def test_paired_and_marginal_cohorts_cannot_be_conflated():
    result = paired_summary(episodes(), candidate_ids=CANDIDATES)
    counts = result["overall"]["candidate_counts"]
    assert counts[CANDIDATES[0]]["marginal_success_rate"] == 2 / 6
    assert counts[CANDIDATES[1]]["marginal_success_rate"] == 4 / 6
    assert result["overall"]["mutually_scorable_pairs"] == 2
    # Summary canonical order is alphabetical: Groot is A, pi05 is B.
    assert result["overall"]["paired_delta_b_minus_a"] == 1
    assert result["overall"]["two_sided_exact_sign_test_p"] == .5
    assert result["qualification_authorized"] is False
    assert result["overall"]["measured_reset_pairs"] == 0
    assert "camera_sensor/held_out" in result["by_family_and_partition"]


def test_paired_evidence_order_is_irrelevant_but_duplicates_and_new_scores_are_not():
    rows = episodes()
    result = paired_summary(rows, candidate_ids=CANDIDATES)
    assert paired_summary(list(reversed(rows)), candidate_ids=list(reversed(CANDIDATES))) == result
    with pytest.raises(ValueError, match="duplicate"):
        paired_summary([*rows, rows[0]], candidate_ids=CANDIDATES)
    rows[0]["episode"]["score"]["task_succeeded"] = False
    assert paired_summary(rows, candidate_ids=CANDIDATES)["summary_digest"] != result["summary_digest"]


def test_no_shared_valid_pairs_does_not_produce_a_paired_rate():
    rows = episodes()
    for row in rows:
        if row["candidate_id"] == CANDIDATES[1] and row["seed"] < 2:
            row["policy_outcome_interpretable"] = False
    result = paired_summary(rows, candidate_ids=CANDIDATES)["overall"]
    assert result["mutually_scorable_pairs"] == 0
    assert result["paired_delta_b_minus_a"] is None
    assert result["two_sided_exact_sign_test_p"] is None


def test_policy_self_grade_never_enters_summary():
    rows = episodes()
    for row in rows:
        row["scoring_authority"] = "candidate_policy"
    assert paired_summary(rows, candidate_ids=CANDIDATES)["overall"]["mutually_scorable_pairs"] == 0


def test_unknown_candidate_never_falls_through_to_groot():
    with pytest.raises(ValueError, match="not_admitted"):
        resolve_policy_interface({"candidate_id": "unadmitted"})
    with pytest.raises(ValueError, match="mismatch"):
        resolve_policy_interface({"candidate_id": "pi05_droid", "policy_interface_id": "groot_n17_droid.v1"})
    assert resolve_policy_interface({"candidate_id": "team_registered", "policy_interface_id": "openpi_droid.v1"}) == "openpi_droid.v1"


@pytest.mark.parametrize("fault", [None, "native_value", "unit"])
def test_exact_workcell_events_read_the_native_value_after_the_real_seeded_reset(fault):
    request = _request()
    matrix = compile_variation_matrix(request)
    cell = compile_isaac_lab_event_plan(matrix, request=request)["cells"][1]
    state, events = {}, []
    bindings = {}
    for term in cell["terms"]:
        key = term["manager_target"]
        def apply(value, key=key):
            state[key] = value
            events.append("apply")
        def read(key=key, unit=term["unit"]):
            assert events[-1] == ("reset", cell["seed"])
            return {"value": state[key] + (1 if fault == "native_value" else 0),
                    "unit": "wrong" if fault == "unit" else unit, "source": "native_readback"}
        bindings[key] = SimpleNamespace(apply=apply, read=read)
    receipt = apply_runtime_cell(matrix=matrix, request=request, cell_id=cell["cell_id"],
        runtime_bindings=bindings, reset=lambda *, seed: events.append(("reset", seed)))
    assert receipt["status"] == ("blocked" if fault else "applied_and_readback_verified")
    assert receipt["policy_query_performed"] is False


def test_full_evaluation_collection_preserves_all_missing_controls_and_policy_cells(tmp_path):
    request = _request()
    matrix = compile_variation_matrix(request)
    bound = _schedule_request(matrix)
    schedule = compile_evaluation_schedule(matrix, request=request, schedule_request=bound)
    result = collect_evaluation_evidence(request=request, schedule_request=bound, matrix=matrix, schedule=schedule,
        episode_records=[], evidence_root=tmp_path, output_path=tmp_path / "result.json")
    assert result["status"] == "blocked"
    assert len(result["missing_episode_ids"]) == 400
    assert all(value["missing"] == 100 for value in result["comparison"]["overall"]["candidate_counts"].values())
    changed = deepcopy(schedule)
    changed["rows"][0]["seed"] += 1
    with pytest.raises(ValueError):
        collect_evaluation_evidence(request=request, schedule_request=bound, matrix=matrix, schedule=changed,
            episode_records=[], evidence_root=tmp_path)


def test_worker_refuses_resealed_checkpoint_change_before_isaac_or_client(tmp_path):
    import json
    from blueprint_pipeline import native_task_arena_policy_canary_worker as worker
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from tests.test_native_task_arena_policy_canary_lifecycle_rehearsal import _stage_runtime_root, FakeIsaac, _rehearsal_runtime
    runtime, output = _stage_runtime_root(tmp_path)
    path = runtime / "runtime_inputs/policy_execution_spec.pi05_droid.json"
    spec = json.loads(path.read_text())
    spec["checkpoint_digest"] = "sha256:" + "9" * 64
    spec["execution_spec_digest"] = canonical_digest(spec, digest_field="execution_spec_digest")
    path.write_text(json.dumps(spec))
    isaac = FakeIsaac(output / worker.PROVIDER_RESULT_FILENAME)
    with pytest.raises(RuntimeError, match="frozen_execution_spec_mismatch"):
        worker._run_selected_cell(0, runtime_root=runtime, output_root=output,
            provider_output_root=output, cell_runtime=_rehearsal_runtime(isaac))
    assert isaac.launches == 0


def test_publication_preserves_an_explicit_registered_pair(tmp_path):
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.task_evaluation_result_delivery import materialize_policy_canary_result_delivery
    from blueprint_pipeline.task_evaluation_policy_canary_result_projection import build_policy_canary_result_projection
    from tests.test_task_evaluation_policy_canary_result_delivery import _result, _closure
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    result = _result(evidence)
    pair = ["team_policy_alpha", "team_policy_beta"]
    template = result["episodes"][0]
    template["evidence_artifacts"]["review_video"] = next(row for row in result["artifact_inventory"] if row["role"] == "review_video")
    result["candidate_ids"] = pair
    result["episodes"] = [dict(deepcopy(template), candidate_id=candidate) for candidate in pair]
    for row in result["episodes"]:
        row["episode"]["episode_id"] = row["candidate_id"] + "-episode"
    result.update(run_id="registered-pair", configuration_digest="sha256:" + "9" * 64,
        status="blocked", blockers=["remaining_cells_incomplete"])
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    closure = {name: _closure(tmp_path / (name + ".json"), flag=flag) for name, flag in (
        ("billing", "official_billing_sealed"), ("teardown", "teardown_completed"), ("provider_zero", "provider_zero_verified"))}
    delivery = materialize_policy_canary_result_delivery(run_root=tmp_path, run_id=result["run_id"],
        result_status="blocked", session_result=result, evidence_root=evidence, closure_records=closure)
    projected = build_policy_canary_result_projection(setup={"scene_id": "fixture", "request_digest": "sha256:" + "a" * 64,
        "task_success_contract": result["task_success_contract"], "task_success_contract_digest": result["task_success_contract_digest"]},
        result=result, delivery=delivery)
    assert projected["candidate_ids"] == pair
    assert [row["candidate_id"] for row in projected["candidate_results"]] == pair
    assert {row["candidate_id"] for row in projected["episodes"]} == set(pair)


def test_concurrent_media_guard_accepts_new_request_artifacts_and_rejects_byte_changes(tmp_path):
    from blueprint_pipeline.policy_canary_media_integrity import require_completed_episode_media
    from tests.test_adp009d_policy_episode import _LifecycleEnvironment, _LifecyclePolicy, _run
    receipt = _run(environment=_LifecycleEnvironment(), policy=_LifecyclePolicy(), max_policy_queries=1,
        settle_window_samples=1, media_output_dir=tmp_path / "episodes", episode_id="media-guard",
        require_complete_multicamera_media=True, require_prestart_readiness=True)
    require_completed_episode_media(tmp_path, receipt)
    artifact = receipt["policy_request_artifacts"][0]
    path = tmp_path / "episodes" / artifact["relative_path"]
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="media_identity_invalid"):
        require_completed_episode_media(tmp_path, receipt)
