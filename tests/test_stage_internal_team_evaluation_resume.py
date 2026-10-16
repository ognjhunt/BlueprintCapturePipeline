"""The staging command must preserve scored cells and the frozen recovery identity."""
import pytest

from tests.test_team_candidate_episode_recovery import scored, refused


def test_staging_rejects_stale_native_digest_before_joining_or_writing(tmp_path, monkeypatch):
    import json
    from scripts import stage_internal_team_evaluation_resume as staging
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    prior, root = tmp_path / "prior", tmp_path / "new"
    job = prior / "evaluation"
    job.mkdir(parents=True)
    cells = [{"cell_id": str(index)} for index in range(200)]
    shards = []
    for index in range(8):
        directory = job / "shards" / str(index)
        directory.mkdir(parents=True)
        shard = {"index": index, "root": str(directory), "cells": cells[index*25:(index+1)*25]}
        shards.append(shard)
        (directory / staging.launch.SHARD_FILENAME).write_text(json.dumps(shard))
    native = prior / "native.json"
    inner = {"episodes": [{"episode": {"score": {"task_succeeded": False}}}]}
    inner["result_digest"] = canonical_digest(inner, digest_field="result_digest")
    # The result and child can agree on a changed score while the result's
    # stale digest still agrees with its allocator. Joining must not reseal it.
    inner["episodes"][0]["episode"]["score"]["task_succeeded"] = True
    native.write_text(json.dumps(inner))
    (job / "shards/0/allocator_result.json").write_text(json.dumps({
        "native_control_result_path": str(native), "native_control_result_digest": inner["result_digest"]}))
    for path, value in [
        (prior / "diagnostic.json", {"job_dir": str(job), "registry_path": str(prior / "registry.json")}),
        (prior / "registry.json", {"profiles": [{}]}),
        (job / staging.launch.TERMINAL_FILENAME, {"status": "blocked", "provider_zero_verified": True}),
        (job / staging.launch.REQUEST_FILENAME, {"internal_diagnostic": True, "customer_recommendation_permitted": False}),
        (job / "team_candidate_admission.json", {"candidate_id": staging.diagnostic.CANDIDATE_ID}),
        (job / staging.launch.INTENT_FILENAME, {"shard_count": 8}),
    ]:
        path.write_text(json.dumps(value))
    monkeypatch.setenv(staging.REGISTRY_ENV, "original-test-value")
    monkeypatch.setattr(staging.launch, "_source_commit", lambda: "candidate")
    monkeypatch.setattr(staging.launch, "_default_provider_zero", lambda: {"provider_zero_verified": True})
    monkeypatch.setattr(staging.launch, "_current_shard", lambda job, index: shards[index])
    monkeypatch.setattr(staging.launch, "_shard_state", lambda shard: "returned")
    monkeypatch.setattr(staging.launch, "_session_cells", lambda **kwargs: cells)
    monkeypatch.setattr(staging, "offer_eligibility", lambda profile: ({}, []))
    monkeypatch.setattr(staging, "join_team_session_closeout",
        lambda **kwargs: pytest.fail("unverified native content reached the resealing join"))
    with pytest.raises(ValueError, match="team_candidate_provider_result_binding_mismatch"):
        staging.stage(prior, root, tmp_path / "billing.json", "candidate")
    assert not root.exists()


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

