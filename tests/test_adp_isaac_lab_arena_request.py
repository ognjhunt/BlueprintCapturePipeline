from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.adp_founder_sim_protocol import build_founder_sim_protocol
from blueprint_pipeline.adp_isaac_lab_arena_request import build_arena_worker_request


def test_request_compiles_exact_arena_jobs_without_authorizing_execution() -> None:
    request = build_arena_worker_request()

    assert request["status"] == "frozen_not_authorized_for_execution"
    assert request["job_count"] == 88
    assert request["scenario_cousins_enabled"] is False
    assert request["paid_compute_authorized"] is False
    assert request["production_simulation_started"] is False
    assert all(
        job["environment"]["type"] == "pick_and_place_maple_table" for job in request["jobs"]
    )
    assert all(job["environment"]["variations"] == {} for job in request["jobs"])
    assert all(job["rollout"]["num_envs"] == 1 for job in request["jobs"])
    for offset in range(0, request["job_count"], 2):
        pair = request["jobs"][offset : offset + 2]
        assert {job["candidate_role"] for job in pair} == {"baseline", "alternative"}
        assert len({job["rollout"]["seed"] for job in pair}) == 1
        assert len({job["reset_digest"] for job in pair}) == 1


def test_request_rejects_changed_protocol() -> None:
    protocol = copy.deepcopy(build_founder_sim_protocol())
    protocol["task"]["instruction"] = "changed after freeze"
    with pytest.raises(ValueError, match="protocol_not_canonical"):
        build_arena_worker_request(protocol)
