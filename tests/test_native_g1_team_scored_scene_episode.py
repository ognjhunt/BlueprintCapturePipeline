"""A team client should get the same G1 scene, score, and retained media."""

from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
)
from blueprint_pipeline.native_g1_team_scored_scene_episode import (
    RESULT_FILENAME,
    TRACE_FILENAME,
    run_g1_team_scored_scene_episode,
)
from tests.test_native_g1_shared_scene_episode import _Bridge, _Scene
from tests.test_native_g1_development_campaign import _book_handoff
from tests.test_team_policy_delivery_profile import OWNER, _profile


def _inputs(tmp_path, monkeypatch, *, movement: bool = False):
    setup = _book_handoff(tmp_path, monkeypatch)["setup"]
    profile = _profile(
        setup,
        {
            "mode": "container",
            "image_ref": "registry.example.org/team/g1@sha256:" + "a" * 64,
            "protocol": "jsonl_observation_action_v1",
        },
    )
    scene = _Scene()
    task = {
        "task_kind": "rigid_pick_place",
        "prompt": "pick the box",
        "task_success_contract_digest": setup["task_success_contract_digest"],
    }
    if movement:
        from blueprint_pipeline.native_g1_navigation_goal import PUBLISHED_TASK_INSTRUCTION

        task["visible_target_marker"] = {
            "schema_version": "native_task_target_marker.v1",
            "shape": "flat_green_disc",
            "non_colliding": True,
            "surface_position_world_m": [0.5, 0.0, 0.7],
            "radius_m": 0.06,
        }
        task["g1_navigation_goal"] = {
            "schema_version": "native_g1_navigation_goal.v1",
            "center_world_m": [2.0, 0.0, 0.0],
            "acceptance_radius_m": 0.3,
            "max_root_height_drift_m": 0.2,
            "settle_window_samples": 2,
            "task_instruction": PUBLISHED_TASK_INSTRUCTION,
            "visible_target_marker": {
                "schema_version": "native_task_target_marker.v1",
                "shape": "flat_yellow_disc",
                "non_colliding": True,
                "surface_position_world_m": [2.0, 0.0, 0.0],
                "radius_m": 0.4,
            },
        }
    scene.plan = {
        **scene.plan,
        "scene_id": setup["scene_id"],
        "task_id": setup["task_id"],
        "task_kind": "rigid_pick_place",
        "task_spec": task,
    }
    scene.plan["plan_digest"] = canonical_digest(scene.plan, digest_field="plan_digest")
    scene.read_state = lambda: {
        "step_index": scene.step,
        "root_position_world_m": [
            0.0 if scene.step == 0 else 1.8 + 0.1 * (scene.step - 1),
            0.0,
            0.85,
        ],
    }
    return setup, profile, scene


class _BoundPolicy:
    def __init__(self, profile_digest: str, expected_task: str):
        self.profile_digest = profile_digest
        self.expected_task = expected_task
        self.queries = 0

    def reset(self, *, seed: int) -> None:
        assert seed == 19

    def infer_chunk(self, *, front_rgb, observation_state, task):
        assert front_rgb.shape == (480, 640, 3)
        assert len(observation_state) == 64
        assert task == self.expected_task
        self.queries += 1
        action = [0.0] * 40
        action[3:9] = [1, 0, 0, 1, 0, 0]
        return [action]


def _run(setup, profile, scene, policy, output_dir, monkeypatch, *, objective_id):
    from blueprint_pipeline import native_g1_joint_episode_environment as g1_environment
    from blueprint_pipeline import native_task_arena_readback

    monkeypatch.setattr(g1_environment, "NativeG1JointEpisodeEnvironment", lambda **kwargs: scene)
    monkeypatch.setattr(
        native_task_arena_readback,
        "NativeRigidTaskArenaReadback",
        lambda built: type(
            "Readback", (), {"read_task_sample": lambda self: {"object_z_m": float(scene.step)}}
        )(),
    )
    return run_g1_team_scored_scene_episode(
        built=type("Built", (), {"plan": scene.plan})(),
        profile=profile,
        trusted_setup=setup,
        authenticated_owner=OWNER,
        policy_client=policy,
        sonic_bridge=_Bridge(),
        objective_id=objective_id,
        max_steps=2,
        output_dir=output_dir / "episode",
        to_tensor=lambda value: value,
        make_action_tensor=lambda value, **kwargs: value,
    )


def test_team_manipulation_runs_and_scores_with_lossless_media(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline import adp_task_scoring

    setup, profile, scene = _inputs(tmp_path, monkeypatch)
    policy = _BoundPolicy(profile["profile_digest"], "pick the box")
    monkeypatch.setattr(
        adp_task_scoring,
        "score_task_episode_from_spec",
        lambda *, task_spec, samples: {
            "status": "scored",
            "outcome": "failure",
            "samples": len(samples),
        },
    )
    result = _run(setup, profile, scene, policy, tmp_path, monkeypatch, objective_id="task_success")
    assert policy.queries == result["policy_query_count"] == 2
    assert result["score"] == {"status": "scored", "outcome": "failure", "samples": 3}
    assert result["profile_digest"] == profile["profile_digest"]
    assert result["owner"] == OWNER
    assert result["source_packet_receipt_digest"] == setup["source_packet_receipt_digest"]
    assert result["task_success_contract_digest"] == setup["task_success_contract_digest"]
    assert result["policy_runtime_identity_verified"] is False
    assert result["ranking_eligible"] is False
    assert result["public_redistribution_authorized"] is False
    assert (tmp_path / "episode" / TRACE_FILENAME).is_file()
    assert (tmp_path / "episode" / RESULT_FILENAME).is_file()
    assert list((tmp_path / "episode").rglob("*.mp4"))


def test_team_movement_uses_measured_navigation_score(tmp_path: Path, monkeypatch) -> None:
    from blueprint_pipeline.native_g1_navigation_goal import PUBLISHED_TASK_INSTRUCTION

    setup, profile, scene = _inputs(tmp_path, monkeypatch, movement=True)
    policy = _BoundPolicy(profile["profile_digest"], PUBLISHED_TASK_INSTRUCTION)
    result = _run(
        setup, profile, scene, policy, tmp_path, monkeypatch, objective_id="g1_navigation_goal"
    )
    assert result["objective_id"] == "g1_navigation_goal"
    assert result["score"]["status"] == "scored"
    assert result["score"]["outcome"] == "success"
    assert result["policy_query_count"] == 2


def test_team_profile_or_scene_mismatch_blocks_before_policy_query(
    tmp_path: Path, monkeypatch
) -> None:
    setup, profile, scene = _inputs(tmp_path, monkeypatch)
    policy = _BoundPolicy(profile["profile_digest"], "pick the box")
    scene.plan["task_id"] = "other-task"
    scene.plan["plan_digest"] = canonical_digest(scene.plan, digest_field="plan_digest")
    with pytest.raises(ValueError, match="binding_invalid"):
        _run(setup, profile, scene, policy, tmp_path, monkeypatch, objective_id="task_success")
    assert scene.seed is None
    assert policy.queries == 0
