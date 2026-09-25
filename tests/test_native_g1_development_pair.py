from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline import native_g1_development_pair as pair
from blueprint_pipeline.native_g1_development_worker import RESULT_FILENAME
from blueprint_pipeline.native_g1_navigation_goal import seal_g1_navigation_goal_authority
from blueprint_pipeline.native_g1_shared_scene_episode import run_g1_shared_scene_episode
from tests.test_native_g1_development_worker import _request as worker_request
from tests.test_native_g1_navigation_goal import _authority_plan
from tests.test_native_g1_shared_scene_episode import _Bridge, _Policy, _Scene


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
        episode_dir = output_dir / "episode"
        episode_dir.mkdir()
        media_dir = episode_dir / "media" / request["candidate_id"]
        media_dir.mkdir(parents=True)
        manifest = {
            "identity": {
                "candidate_id": request["candidate_id"],
                "scene_plan_digest": plan["plan_digest"],
            },
        }
        manifest["frame_manifest_digest"] = canonical_digest(
            manifest, digest_field="frame_manifest_digest"
        )
        manifest_path = media_dir / "multicamera_frame_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        artifacts = [{
            "role": "multicamera_observation_frame_manifest",
            "relative_path": manifest_path.relative_to(episode_dir).as_posix(),
            "sha256": "sha256:" + hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "size_bytes": manifest_path.stat().st_size,
        }]
        videos = {}
        for camera_id in ("head", "overview"):
            video_path = media_dir / f"{camera_id}.mp4"
            video_path.write_bytes(f"fixture:{request['candidate_id']}:{camera_id}".encode())
            video = {
                "relative_path": video_path.relative_to(episode_dir).as_posix(),
                "sha256": "sha256:" + hashlib.sha256(video_path.read_bytes()).hexdigest(),
                "size_bytes": video_path.stat().st_size,
                "derived_from_frame_manifest_digest": manifest["frame_manifest_digest"],
            }
            videos[camera_id] = video
            artifacts.append({
                "role": "camera_review_video", "camera_id": camera_id,
                "relative_path": video["relative_path"], "sha256": video["sha256"],
                "size_bytes": video["size_bytes"], "media_type": "video/mp4",
            })
        trace = {
            "status": "development_trace_recorded",
            "candidate_id": request["candidate_id"],
            "scene_plan_digest": plan["plan_digest"],
            "claim_ceiling": "simulator_only_unscored",
            "visual_evidence": {
                "status": "complete", "videos": videos,
                "required_camera_ids": ["head", "overview"],
                "frame_manifest_digest": manifest["frame_manifest_digest"],
            },
            "media_artifacts": artifacts,
        }
        trace["trace_digest"] = canonical_digest(trace, digest_field="trace_digest")
        (episode_dir / pair.TRACE_FILENAME).write_text(json.dumps(trace))
        episode = {
            "status": "development_only_scored_episode", "candidate_id": request["candidate_id"],
            "scene_plan_digest": plan["plan_digest"],
            "evaluation_task_kind": (
                "g1_navigation_goal" if request["candidate_id"].endswith("_vision_navi")
                else "rigid_pick_place"
            ),
            "score": score, "ranking_eligible": False, "physical_outcome_claimed": False,
            "trace_relative_path": pair.TRACE_FILENAME,
            "trace_digest": trace["trace_digest"],
        }
        episode["result_digest"] = canonical_digest(episode, digest_field="result_digest")
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
    assert all(set(a["review_media"]["review_videos"]) == {"head", "overview"}
               for a in result["attempts"])
    assert all(a["review_media"]["public_redistribution_authorized"] is False
               for a in result["attempts"])
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


def test_subprocess_native_exit_retains_pair_receipt_and_exit_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, _ = _paired_requests(tmp_path)
    launcher = tmp_path / "python.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    calls = []

    def crash(command, *, stdout, stderr, check):
        calls.append(command)
        stdout.write("native simulator stopped before Python could write a receipt\n")
        return SimpleNamespace(returncode=-11)

    monkeypatch.setattr(pair.sys, "platform", "linux")
    monkeypatch.setattr(pair.subprocess, "run", crash)
    result = pair.run_g1_development_pair(
        request_paths=paths, output_dir=tmp_path / "comparison",
        mode="subprocess", worker_launcher=launcher,
    )
    assert result["status"] == "blocked"
    assert result["not_attempted_candidate_ids"] == [PI]
    assert result["attempts"][0]["blocker"]["message"] == (
        "g1_pair_worker_exited_without_receipt:-11"
    )
    assert calls[0][:3] == [str(launcher), "-m", "blueprint_pipeline.native_g1_development_worker"]
    diagnostics = tmp_path / "comparison/_worker_diagnostics"
    assert json.loads((diagnostics / f"{DP}.exit.json").read_text()) == {"returncode": -11}
    assert "native simulator stopped" in (diagnostics / f"{DP}.log").read_text()
    assert json.loads((tmp_path / "comparison" / (pair.SCHEMA + ".json")).read_text()) == result


def test_subprocess_preserves_episode_layout_and_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, _ = _paired_requests(tmp_path)
    launcher = tmp_path / "python.sh"
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")

    def run(command, *, stdout, stderr, check):
        request = json.loads(Path(command[4]).read_text(encoding="utf-8"))
        _fake_result(request, Path(command[6]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pair.sys, "platform", "linux")
    monkeypatch.setattr(pair.subprocess, "run", run)
    result = pair.run_g1_development_pair(
        request_paths=paths, output_dir=tmp_path / "comparison",
        mode="subprocess", worker_launcher=launcher,
    )
    assert result["status"] == "completed_development_only"
    assert [row["candidate_id"] for row in result["attempts"]] == [DP, PI]
    assert all(row["review_media"] for row in result["attempts"])


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


def test_tampered_review_video_retains_worker_receipt_but_blocks_pair(tmp_path: Path) -> None:
    paths, _ = _paired_requests(tmp_path)

    def run(*, request: dict, output_dir: Path) -> dict:
        result = _fake_result(request, output_dir)
        (output_dir / "episode/media" / request["candidate_id"] / "head.mp4").write_bytes(
            b"changed"
        )
        return result

    result = pair.run_g1_development_pair(
        request_paths=paths, output_dir=tmp_path / "comparison", local_runner=run,
    )
    attempt = result["attempts"][0]
    assert attempt["status"] == "blocked"
    assert attempt["worker_status"] == "completed_development_only"
    assert attempt["worker_result_digest"]
    assert attempt["score"]["score_digest"]
    assert attempt["blocker"]["message"] in {
        "g1_pair_media_file_invalid", "g1_pair_media_digest_mismatch"
    }
    assert result["not_attempted_candidate_ids"] == [PI]


def test_review_index_accepts_actual_g1_frame_and_video_finalizer(tmp_path: Path) -> None:
    scene = _Scene()
    pair_root = tmp_path / "comparison"
    episode_root = pair_root / DP / "episode"
    trace = run_g1_shared_scene_episode(
        environment=scene,
        policy_client=_Policy(),
        sonic_bridge=_Bridge(),
        candidate_id=DP,
        task_prompt="pick the box",
        max_steps=1,
        output_dir=episode_root,
        read_task_sample=lambda: {"step_index": scene.step},
    )
    (episode_root / pair.TRACE_FILENAME).write_text(json.dumps(trace), encoding="utf-8")
    media = pair._verified_review_media(
        episode_root / pair.EPISODE_FILENAME,
        episode={
            "candidate_id": DP,
            "scene_plan_digest": scene.plan["plan_digest"],
            "trace_relative_path": pair.TRACE_FILENAME,
            "trace_digest": trace["trace_digest"],
        },
        pair_root=pair_root,
    )
    assert media["frame_manifest_digest"] == trace["visual_evidence"]["frame_manifest_digest"]
    assert all((pair_root / row["relative_path"]).is_file()
               for row in media["review_videos"].values())


def test_container_pair_uses_resealed_worker_request_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import native_g1_container_run
    from blueprint_pipeline import native_g1_container_host

    paths, _ = _paired_requests(tmp_path)
    container_digest = "sha256:" + "e" * 64
    staged: dict = {}

    def prepare(**kwargs: object) -> dict:
        output_dir = kwargs["output_dir"]
        output_dir.mkdir()
        (output_dir / "results").mkdir()
        staged.update(request_path=kwargs["request_path"], output_dir=output_dir)
        return {"command": ["docker", "run", "fixture"],
                "container_request_digest": container_digest}

    def run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        request = json.loads(staged["request_path"].read_text())
        request["request_digest"] = container_digest
        _fake_result(request, staged["output_dir"] / "results/episode", blocked=True)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(pair.sys, "platform", "linux")
    monkeypatch.setattr(native_g1_container_run, "prepare_g1_container_run", prepare)
    monkeypatch.setattr(native_g1_container_host, "record_g1_container_host", lambda **_kwargs: {})
    monkeypatch.setattr(pair.subprocess, "run", run)
    result = pair.run_g1_development_pair(
        request_paths=paths, output_dir=tmp_path / "comparison", mode="container",
        source_receipt_path=tmp_path / "source-receipt.json",
        source_packet_path=tmp_path / "source-packet.tar",
        policy_runtime_root=tmp_path / "policy-runtime",
    )
    assert result["status"] == "blocked"
    assert result["attempts"][0]["worker_result_digest"]
    assert result["attempts"][0]["blocker"]["message"] == "model_bytes_mismatch"
    assert result["not_attempted_candidate_ids"] == [PI]


def test_container_pair_refuses_unready_host_before_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from blueprint_pipeline import native_g1_container_run, native_g1_container_host

    paths, _ = _paired_requests(tmp_path)

    def prepare(**kwargs: object) -> dict:
        kwargs["output_dir"].mkdir()
        return {"command": ["docker", "run", "fixture"],
                "container_request_digest": "sha256:" + "e" * 64}

    def fail_host(**_kwargs: object) -> dict:
        raise ValueError("g1_container_gpu_zero_missing")

    def forbidden_docker(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("docker must not start")

    monkeypatch.setattr(pair.sys, "platform", "linux")
    monkeypatch.setattr(native_g1_container_run, "prepare_g1_container_run", prepare)
    monkeypatch.setattr(native_g1_container_host, "record_g1_container_host", fail_host)
    monkeypatch.setattr(pair.subprocess, "run", forbidden_docker)
    result = pair.run_g1_development_pair(
        request_paths=paths, output_dir=tmp_path / "comparison", mode="container",
        source_receipt_path=tmp_path / "source-receipt.json",
        source_packet_path=tmp_path / "source-packet.tar",
        policy_runtime_root=tmp_path / "policy-runtime",
    )
    assert result["status"] == "blocked"
    assert result["attempts"][0]["blocker"]["message"] == "g1_container_gpu_zero_missing"
    assert result["not_attempted_candidate_ids"] == [PI]
