import copy
from pathlib import Path

import pytest

from blueprint_pipeline import task_evaluation_unused_team_holds as holds
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline import task_evaluation_scene_execution_authority as authority
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_task_evaluation_terminal_adoption_retirement import adopted as adopted
from tests.test_task_evaluation_unstarted_controls_reservations import reserved as reserved, put


@pytest.fixture
def team(adopted):
    config, source, owner, old, old_path, reserve, run = adopted
    old.update(evaluation_run_id="team-eval-example", evaluation_authority={
        "source_launch_id": source["launch_id"], "scene_intent_digest": owner["intent_digest"]})
    old["intent_digest"] = canonical_digest(old, digest_field="intent_digest")
    put(old_path, old)
    current = copy.deepcopy(old)
    current["expected_production_commit"] = "b" * 40
    current["intent_digest"] = canonical_digest(current, digest_field="intent_digest")
    current_path = old_path.parent / "current.json"
    put(current_path, current)
    record = {"execution_source_commit": "b" * 40, "adoption": source["adoption"],
        "provisioning": {"intent_path": str(current_path)}}
    record["receipt_digest"] = canonical_digest(record, digest_field="receipt_digest")
    put(Path(config["controls_root"]) / "terminal-adoptions" / owner["intent_id"] / "current" / "terminal_adoption_provisioning.json", record)
    binding = Path(config["progression_root"]) / source["launch_id"] / holds.scoped_identity("cpu-robot-binding", old["evaluation_run_id"])
    kwargs = dict(config=config, intent_id=owner["intent_id"], current_intent_path=current_path)
    return kwargs, old, current, binding, reserve


def test_releases_only_old_unpaid_holds_and_preserves_cpu_and_active_work(team):
    kwargs, old, current, binding, reserve = team
    put(binding / "cpu-placement-checkpoints" / "old" / "geometry.json", {"cpu": True})
    current_agent = holds.unpaid_paths(binding, current["intent_digest"])[0]
    put(current_agent / "attempt_000" / "real_inference.json", {"started": True})
    old_cost = holds.unpaid_paths(binding, old["intent_digest"])[1]
    put(old_cost / "attempt_000" / "openai_scope_lock_acquired.v1.json", {})
    before = {p: p.read_bytes() for p in binding.rglob("*") if p.is_file()}
    preview = holds.retire(**kwargs, dry_run=True)
    assert len(preview) == 3
    rows = holds.retire(**kwargs)
    assert rows == preview == holds.retire(**kwargs)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    directory = Path(kwargs["config"]["scene_root"]) / kwargs["intent_id"]
    for row in rows:
        attempt = intake._read(directory / "attempts" / (row["attempt_id"] + ".json"), "attempt_digest")
        assert authority.scene_execution_authority_blockers(authority.bind_scene_attempt(attempt),
            source_commit=attempt["source_commit"], provider=attempt["provider"],
            maximum_spend_usd=attempt["maximum_spend_usd"], now=102) == ["scene_execution_owner_attempt_cancelled_before_execution"]
    for phase, amount, provider in [("construction", .45, "vast"), ("controls", .45, "vast"), ("placement", 2.56, "openai")]:
        reserve("controls-current-" + phase, amount, provider, commit="b")
    reserve("policy", 4, commit="b")
    with pytest.raises(ValueError, match="spend_cap_exhausted"):
        reserve("too-much", .10, commit="b")


@pytest.mark.parametrize("artifact", ["reservation", "agent_output", "checkpoint", "result", "symlink", "cost_parent_symlink"])
def test_any_paid_or_ambiguous_old_work_keeps_the_entire_hold(team, tmp_path, artifact):
    kwargs, old, current, binding, reserve = team
    agent, cost, checkpoint, result = holds.unpaid_paths(binding, old["intent_digest"])
    if artifact == "reservation":
        put(cost / "attempt_000" / "openai_official_cost_run_reservation.v1.json", {})
    elif artifact == "agent_output":
        put(agent / "attempt_000" / "output.json", {})
    elif artifact in {"checkpoint", "result"}:
        put(checkpoint if artifact == "checkpoint" else result, {})
    else:
        target = agent if artifact == "symlink" else cost.parent
        target.parent.mkdir(parents=True)
        target.symlink_to(tmp_path, target_is_directory=True)
    assert holds.retire(**kwargs) == []


@pytest.mark.parametrize("change", ["owner", "run", "source", "intent_digest", "release"])
def test_refuses_changed_authority_before_releasing_holds(team, change, monkeypatch):
    kwargs, old, current, binding, reserve = team
    if change == "owner":
        current["evaluation_authority"]["scene_intent_digest"] = "sha256:" + "0" * 64
    elif change == "run":
        current["evaluation_run_id"] = "another-team"
    elif change == "source":
        current["configuration_adoption"]["source_launch_id"] = "another-scene"
    elif change == "release":
        from blueprint_pipeline import task_evaluation_release_identity as release
        monkeypatch.setattr(release, "running_release_commit", lambda: "c" * 40)
    current["intent_digest"] = canonical_digest(current, digest_field="intent_digest")
    if change == "intent_digest":
        current["intent_digest"] = "sha256:" + "0" * 64
    put(kwargs["current_intent_path"], current)
    with pytest.raises(ValueError):
        holds.retire(**kwargs)
    directory = Path(kwargs["config"]["scene_root"]) / kwargs["intent_id"] / holds.DIRECTORY
    assert not list(directory.glob("controls-adopted-*.json"))
