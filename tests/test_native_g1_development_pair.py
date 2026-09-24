from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_g1_development_pair as pair
from blueprint_pipeline.native_g1_development_worker import RESULT_FILENAME
from blueprint_pipeline.native_g1_navigation_goal import seal_g1_navigation_goal_authority
from tests.test_native_g1_development_worker import _request as worker_request
from tests.test_native_g1_navigation_goal import _authority_plan


DP = "humanoidarena_dp_g1_dex3_sonic"
PI = "humanoidarena_pi05_g1_dex3_sonic"


def _seal_request(request: dict, *, candidate_id: str, plan: dict) -> dict:
    request["candidate_id"] = candidate_id
    request["rights_review"]["candidate_id"] = candidate_id
    request["rights_review"]["scene_plan_digest"] = plan["plan_digest"]
    request["rights_review"]["rights_review_digest"] = canonical_digest(
        request["rights_review"], digest_field="rights_review_digest"
    )
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


def _paired_requests(tmp_path: Path, *, navigation: bool = False) -> tuple[list[Path], dict]:
    first = worker_request(tmp_path)
    plan = _authority_plan() if navigation else {
        "scene_id": "scene-a", "task_id": "task-a", "task_kind": "rigid_pick_place",
        "robot": {"robot_id": "unitree_g1"},
    }
    if not navigation:
        plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    Path(first["bundle_root"], "native_task_arena_scene_plan.v1.json").write_text(
        json.dumps(plan), encoding="utf-8"
    )
    if navigation:
        first["navigation_goal_authority"] = seal_g1_navigation_goal_authority(
            plan=plan, confirmed_by_team_id="team-a", human_reviewer="owner-a"
        )
    second = json.loads(json.dumps(first))
    candidates = (DP + "_vision_navi", PI + "_vision_navi") if navigation else (DP, PI)
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    for path, request, candidate in zip(paths, (first, second), candidates, strict=True):
        path.write_text(json.dumps(_seal_request(request, candidate_id=candidate, plan=plan)))
    return paths, plan


def _fake_result(request: dict, output_dir: Path, *, blocked: bool = False) -> dict:
    output_dir.mkdir()
    plan = json.loads(Path(request["bundle_root"], "native_task_arena_scene_plan.v1.json").read_text())
    if blocked:
        result = {
            "candidate_id": request["candidate_id"], "request_digest": request["request_digest"],
            "scene_plan_digest": None, "status": "blocked", "phase_reached": "preflight",
            "blocker": {"type": "ValueError", "message": "model_bytes_mismatch"},
            "ranking_eligible": False, "physical_outcome_claimed": False,
        }
    else:
        score = {"status": "scored", "outcome": "failure"}
        score["score_digest"] = canonical_digest(score, digest_field="score_digest")
        episode = {
            "status": "development_only_scored_episode", "candidate_id": request["candidate_id"],
            "scene_plan_digest": plan["plan_digest"],
            "evaluation_task_kind": (
                "g1_navigation_goal" if request["candidate_id"].endswith("_vision_navi")
                else "rigid_pick_place"
            ),
            "score": score, "ranking_eligible": False, "physical_outcome_claimed": False,
        }
        episode["result_digest"] = canonical_digest(episode, digest_field="result_digest")
        episode_dir = output_dir / "episode"
        episode_dir.mkdir()
        (episode_dir / pair.EPISODE_FILENAME).write_text(json.dumps(episode))
        result = {
            "candidate_id": request["candidate_id"], "request_digest": request["request_digest"],
            "scene_plan_digest": plan["plan_digest"], "status": "completed_development_only",
            "supervised_episode": {"episode_result_digest": episode["result_digest"]},
            "navigation_goal_authority_digest": (
                request.get("navigation_goal_authority") or {}
            ).get("authority_digest"),
            "ranking_eligible": False, "physical_outcome_claimed": False,
        }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (output_dir / RESULT_FILENAME).write_text(json.dumps(result))
    return result


def test_pair_runs_same_scene_in_catalog_order_and_preserves_two_scores(tmp_path: Path) -> None:
    paths, plan = _paired_requests(tmp_path)
    calls = []

    def run(*, request: dict, output_dir: Path) -> dict:
        calls.append(request["candidate_id"])
        return _fake_result(request, output_dir)

    result = pair.run_g1_development_pair(
        request_paths=list(reversed(paths)), output_dir=tmp_path / "comparison",
        local_runner=run,
    )
    assert calls == [DP, PI]
    assert result["status"] == "completed_development_only"
    assert result["scene_plan_digest"] == plan["plan_digest"]
    assert len({a["score"]["score_digest"] for a in result["attempts"]}) == 1
    assert all(a["score"]["outcome"] == "failure" for a in result["attempts"])
    assert result["ranking_eligible"] is False
    assert json.loads((tmp_path / "comparison" / (pair.SCHEMA + ".json")).read_text()) == result


def test_preflight_block_retains_terminal_receipt_and_skips_second_candidate(tmp_path: Path) -> None:
    paths, _ = _paired_requests(tmp_path)
    calls = []

    def run(*, request: dict, output_dir: Path) -> dict:
        calls.append(request["candidate_id"])
        return _fake_result(request, output_dir, blocked=True)

    result = pair.run_g1_development_pair(
        request_paths=paths, output_dir=tmp_path / "comparison", local_runner=run,
    )
    assert calls == [DP]
    assert result["status"] == "blocked"
    assert result["not_attempted_candidate_ids"] == [PI]
    assert result["attempts"][0]["worker_result_digest"]
    assert result["attempts"][0]["blocker"]["message"] == "model_bytes_mismatch"


def test_pair_rejects_changed_runtime_and_rights_before_output(tmp_path: Path) -> None:
    paths, _ = _paired_requests(tmp_path)
    request = json.loads(paths[1].read_text())
    request["max_steps"] = 3
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    paths[1].write_text(json.dumps(request))
    with pytest.raises(ValueError, match="g1_pair_shared_runtime_or_scene_mismatch"):
        pair.validate_g1_development_pair(paths)
    request["max_steps"] = 2
    request["rights_review"]["human_reviewer"] = "changed"
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    paths[1].write_text(json.dumps(request))
    with pytest.raises(ValueError, match="g1_pair_rights_review_invalid"):
        pair.run_g1_development_pair(request_paths=paths, output_dir=tmp_path / "no-output")
    assert not (tmp_path / "no-output").exists()


def test_navigation_pair_requires_same_team_confirmed_goal(tmp_path: Path) -> None:
    paths, _ = _paired_requests(tmp_path, navigation=True)
    validated = pair.validate_g1_development_pair(paths)
    assert validated["objective_id"] == "g1_navigation_goal"
    assert validated["navigation_goal_authority_digest"]
    request = json.loads(paths[1].read_text())
    request.pop("navigation_goal_authority")
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    paths[1].write_text(json.dumps(request))
    with pytest.raises(ValueError):
        pair.validate_g1_development_pair(paths)


def test_tampered_episode_score_cannot_complete_pair(tmp_path: Path) -> None:
    paths, _ = _paired_requests(tmp_path)

    def run(*, request: dict, output_dir: Path) -> dict:
        result = _fake_result(request, output_dir)
        episode_path = output_dir / "episode" / pair.EPISODE_FILENAME
        episode = json.loads(episode_path.read_text())
        episode["score"]["outcome"] = "success"
        episode_path.write_text(json.dumps(episode))
        return result

    result = pair.run_g1_development_pair(
        request_paths=paths, output_dir=tmp_path / "comparison", local_runner=run,
    )
    assert result["status"] == "blocked"
    assert result["attempts"][0]["blocker"]["message"] == (
        "g1_pair_episode_or_score_receipt_invalid"
    )
    assert result["not_attempted_candidate_ids"] == [PI]
