from copy import deepcopy

import pytest

from blueprint_pipeline.controlled_policy_configuration import canonical_request_digest
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

from blueprint_pipeline.team_candidate_episode_recovery import select_pre_policy_reset_recovery


@pytest.mark.parametrize("fault", [None, "changed_source", "changed_cells", "wrong_request", "wrong_admission", "bad_digest"])
def test_recovery_plan_binds_original_frozen_inputs_and_immutable_receipts(tmp_path, fault):
    import hashlib
    import json
    from blueprint_pipeline.team_candidate_episode_recovery import recovery_arguments

    request = {"job_id": "diagnostic", "capture_root": "/capture"}
    request_digest = canonical_request_digest(request)
    sources = {
        "original_request": request,
        "original_admission": {"admission_digest": "admitted"},
        "prior_terminal": {"status": "blocked", "provider_zero_verified": True,
                           "canonical_request_digest": request_digest},
        "original_shard": {"index": 0, "cells": [scored("a"), scored("b")]},
        "retained_native_session": {"episodes": [scored("a"), refused("b")]},
    }
    records = []
    for role, value in sources.items():
        path = tmp_path / (role + ".json")
        data = json.dumps(value).encode()
        path.write_bytes(data)
        records.append({"role": role, "path": str(path), "size_bytes": len(data),
                        "sha256": "sha256:" + hashlib.sha256(data).hexdigest()})
    plan = {"schema_version": "team_candidate_pre_policy_reset_recovery_plan.v1",
        "canonical_request_digest": request_digest, "admission_digest": "admitted", "source_records": records,
        "expected_cells": [scored("a"), scored("b")], "recovery_cell_ids": ["b"], "plan_digest": ""}
    if fault == "changed_source":
        (tmp_path / "retained_native_session.json").write_text("{}")
    if fault == "changed_cells":
        plan["expected_cells"][1]["seed"] += 1
    if fault == "wrong_request":
        plan["canonical_request_digest"] = "other"
    if fault == "wrong_admission":
        plan["admission_digest"] = "other"
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    if fault == "bad_digest":
        plan["plan_digest"] = "invalid"
    if fault:
        with pytest.raises(ValueError, match="team_candidate_recovery"):
            recovery_arguments(plan, request_digest=request_digest, admission_digest="admitted", episodes=2)
    else:
        assert recovery_arguments(plan, request_digest=request_digest, admission_digest="admitted", episodes=2) == {
            "expected_cells": [scored("a"), scored("b")], "pre_policy_recovery_cells": ["b"]}


def test_resume_layout_adopts_scores_and_places_refused_cell_first():
    from scripts.stage_internal_team_evaluation_resume import remaining_groups

    cells = [scored(str(index)) for index in range(200)]
    retained = cells[:25] + cells[50:74] + [refused("74")]
    groups, refused_ids = remaining_groups(cells, retained)
    assert [len(group) for group in groups] == [25, 25, 25, 25, 25, 25, 1]
    assert groups[0][0]["cell_id"] == "74" and refused_ids == ["74"]
    assert len({cell["cell_id"] for group in groups for cell in group}) == 151
    with pytest.raises(ValueError, match="not_unique"):
        remaining_groups(cells, retained + [cells[0]])


def scored(cell, success=False):
    return {
        "candidate_id": "team.policy",
        "cell_id": cell,
        "seed": 74,
        "cell_spec_digest": "sha256:" + "1" * 64,
        "checkpoint_digest": "sha256:" + "2" * 64,
        "resolved_scenario_digest": "sha256:" + "4" * 64,
        "runtime_identity_digest": "sha256:" + "3" * 64,
        "status": "completed",
        "candidate_policy_queried": True,
        "actions_reached_robot": True,
        "scoring_authority": "deterministic_simulator_state",
        "episode": {"score": {"status": "scored", "task_succeeded": success}},
    }


def refused(cell):
    value = scored(cell)
    value.update(
        status="blocked",
        episode_failure_stage="before_first_observation",
        failure_message="scientific_reset_scenario_application_mismatch",
        candidate_policy_query_attempted=False,
        candidate_policy_queried=False,
        actions_reached_robot=False,
        episode={
            "candidate_policy_query_attempted": False,
            "policy_response_status": "not_attempted",
            "score": {"status": "undetermined", "task_succeeded": None},
        },
    )
    return value


def test_recovery_keeps_task_failures_and_preserves_every_original_record():
    rows = [("old", scored("a")), ("old", refused("b")), ("new", scored("b"))]
    original = deepcopy(rows)
    selected, receipt = select_pre_policy_reset_recovery(
        rows, expected_cells=[scored("a"), scored("b")], recovery_cell_ids=["b"]
    )
    assert [row["cell_id"] for row in selected] == ["a", "b"]
    assert all(row["episode"]["score"]["task_succeeded"] is False for row in selected)
    assert rows == original and receipt["completed_policy_attempts_replaced"] == 0
    assert receipt["superseded_pre_policy_refusals"][0]["session_result_digest"] == "old"


@pytest.mark.parametrize(
    "fault",
    [
        "already_scored",
        "query_attempted",
        "query_unknown",
        "actions_unknown",
        "after_observation",
        "seed",
        "spec",
        "unregistered",
        "extra_cell",
        "unscored_retry",
    ],
)
def test_recovery_never_replaces_policy_results_or_unknown_attempts(fault):
    bad, retry = refused("b"), scored("b", success=True)
    allowed = ["b"]
    if fault == "already_scored":
        bad = scored("b", success=False)
    if fault == "query_attempted":
        bad["candidate_policy_query_attempted"] = True
    if fault == "query_unknown":
        bad.pop("candidate_policy_query_attempted")
    if fault == "actions_unknown":
        bad["actions_reached_robot"] = None
    if fault == "after_observation":
        bad["episode_failure_stage"] = "after_first_observation"
    if fault == "seed":
        bad["seed"] += 1
    if fault == "spec":
        bad["cell_spec_digest"] = "sha256:" + "4" * 64
    if fault == "unregistered":
        allowed = ["a"]
    if fault == "unscored_retry":
        retry["episode"]["score"]["status"] = "undetermined"
    rows = [("old", scored("a")), ("old", bad), ("new", retry)]
    if fault == "extra_cell":
        rows.append(("new", scored("c")))
    with pytest.raises(ValueError, match="team_candidate_recovery"):
        select_pre_policy_reset_recovery(
            rows, expected_cells=[scored("a"), scored("b")], recovery_cell_ids=allowed
        )
