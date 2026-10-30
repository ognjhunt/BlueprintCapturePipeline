"""Development-only scripted provenance survives the real publication boundary."""
from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from blueprint_pipeline.beta_policy_screening_result import screening_publication
from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest

KIND = "scripted_simulator_state_baseline"
ROOT = Path(__file__).parents[1]


def publication_inputs():
    original = json.loads((ROOT / "tests/fixtures/beta_policy_screening_publication.v2.json").read_text())
    row = original["candidates"][0]
    configured = {"candidate_id": row["candidate_id"], "configuration_digest": row["configuration_digest"],
                  "display_name": "Development-only scripted baseline", "episode_count": row["episodes_planned"],
                  "execution_evidence_kind": KIND}
    plan = {key: original[key] for key in ("campaign_id", "site_id", "plan_digest", "catalog_digest", "scenario_plan_digest", "source")}
    plan["candidates"] = [configured]
    summary = {key: row[key] for key in ("episodes_scored", "episodes_succeeded", "episodes_failed", "episodes_void",
                                        "success_rate", "session_result_digest")}
    summary.update(execution_evidence_kind=KIND, action_source="scripted_simulator_state", scripted_rollout_count=10,
                   learned_policy_rollout_count=0, scripted_controller_invoked=True, candidate_policy_queried=False)
    state = {"revision": original["revision"], "updated_at": original["generated_at"],
             "candidates": {row["candidate_id"]: {"status": "completed", "summary": summary}}}
    return {"plan": plan, "state": state, "budget": original["budget"]}


def test_real_publication_matches_cross_repo_scripted_fixture():
    value = screening_publication(**publication_inputs())
    expected = json.loads((ROOT / "tests/fixtures/beta_policy_screening_scripted_publication.v2.json").read_text())
    assert value == expected
    assert value["publication_digest"] == cross_runtime_canonical_digest(value, digest_field="publication_digest")
    Draft202012Validator(json.loads((ROOT / "docs/schemas/beta_policy_screening_publication.v2.schema.json").read_text())).validate(value)
    row = value["candidates"][0]
    assert row["episodes_scored"] == 3 and row["scripted_rollout_count"] == 10
    assert row["learned_policy_rollout_count"] == 0 and row["candidate_policy_queried"] is False


@pytest.mark.parametrize("status", ["pending", "blocked", "unavailable", "deferred"])
def test_unexecuted_scripted_candidates_are_typed_without_inventing_invocation(status):
    inputs = publication_inputs()
    state = next(iter(inputs["state"]["candidates"].values()))
    state.update(status=status, summary={})
    row = screening_publication(**inputs)["candidates"][0]
    assert row["execution_evidence_kind"] == KIND
    assert row["scripted_rollout_count"] == row["learned_policy_rollout_count"] == row["episodes_scored"] == 0
    assert row["scripted_controller_invoked"] is row["candidate_policy_queried"] is False


def test_retained_prestart_failures_do_not_invent_controller_invocation():
    inputs = publication_inputs()
    summary = next(iter(inputs["state"]["candidates"].values()))["summary"]
    summary.update(episodes_scored=0, episodes_succeeded=0, episodes_failed=0, episodes_void=10,
                   success_rate=None, scripted_controller_invoked=False)
    row = screening_publication(**inputs)["candidates"][0]
    assert row["scripted_rollout_count"] == 10 and row["scripted_controller_invoked"] is False


@pytest.mark.parametrize("kind", [KIND, "learned_policy"])
def test_invocation_requires_at_least_one_retained_episode_record(kind):
    inputs = publication_inputs()
    inputs["plan"]["candidates"][0]["execution_evidence_kind"] = kind
    summary = next(iter(inputs["state"]["candidates"].values()))["summary"]
    summary.update(episodes_scored=0, episodes_succeeded=0, episodes_failed=0, episodes_void=10,
        success_rate=None, execution_evidence_kind=kind, scripted_rollout_count=0, learned_policy_rollout_count=0,
        action_source="scripted_simulator_state" if kind == KIND else "learned_policy",
        scripted_controller_invoked=kind == KIND, candidate_policy_queried=kind == "learned_policy")
    with pytest.raises(ValueError, match="beta_screening_result_execution_evidence_invalid"):
        screening_publication(**inputs)


@pytest.mark.parametrize("change", [{"learned_policy_rollout_count": 1}, {"candidate_policy_queried": True},
    {"action_source": "learned_policy"}, {"execution_evidence_kind": "learned_policy"},
    {"scripted_controller_invoked": False}, {"scripted_rollout_count": 2}, {"scripted_rollout_count": True}])
def test_scripted_results_cannot_be_published_as_learned_or_unwitnessed(change):
    inputs = publication_inputs()
    next(iter(inputs["state"]["candidates"].values()))["summary"].update(change)
    with pytest.raises(ValueError, match="beta_screening_result_.*invalid"):
        screening_publication(**inputs)


def test_old_publications_keep_absent_execution_provenance_absent():
    inputs = publication_inputs()
    del inputs["plan"]["candidates"][0]["execution_evidence_kind"]
    inputs["plan"]["candidates"][0]["display_name"] = "candidate-000"
    summary = next(iter(inputs["state"]["candidates"].values()))["summary"]
    for key in ("execution_evidence_kind", "action_source", "scripted_rollout_count", "learned_policy_rollout_count",
                "scripted_controller_invoked", "candidate_policy_queried"):
        del summary[key]
    original = json.loads((ROOT / "tests/fixtures/beta_policy_screening_publication.v2.json").read_text())
    assert screening_publication(**deepcopy(inputs)) == original
    summary["execution_evidence_kind"] = "learned_policy"
    assert screening_publication(**inputs) == original
    summary["execution_evidence_kind"] = KIND
    with pytest.raises(ValueError, match="beta_screening_result_evidence_kind_invalid"):
        screening_publication(**inputs)
