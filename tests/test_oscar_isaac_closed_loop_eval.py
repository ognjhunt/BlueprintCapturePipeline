"""Hermetic test for the per-step policy <-> WAM <-> perception closed loop (no GPU).

Validates the loop STRUCTURE the real run depends on: each step the policy acts, a (stubbed)
WAM generates the next observation, and the perception harness runs on it immediately. The real
run swaps the stub WAM for per-step OSCAR-2B and the fixture harness backend for real SAM3/DA3,
along the same code path.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest


pytestmark = [pytest.mark.slow, pytest.mark.integration]
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline import oscar_isaac_closed_loop_eval as L
from blueprint_pipeline.oscar_wam_command_adapter import DEFAULT_NUM_FRAMES


def _write_frame(path: Path, seed: int) -> Path:
    # a non-flat, non-dark frame so the harness frame-quality gate does not reject it
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 48))
    pix = img.load()
    for y in range(48):
        for x in range(64):
            pix[x, y] = ((x * 4 + seed) % 256, (y * 5 + seed) % 256, (x + y + seed) % 256)
    img.save(path)
    return path


def _stub_wam(work: Path):
    """A local stand-in WAM: writes the next-observation frame (real PNG) per step."""

    def _generate(current_frame, action, step_index, history):
        out = work / "wam_generated" / f"step_{step_index:04d}.png"
        _write_frame(out, seed=step_index * 17)
        return {"generated_frame_path": str(out)}

    return _generate


def _stub_wam_with_video(work: Path):
    """A WAM stand-in that exposes a generated video artifact for consistency scoring."""

    def _generate(current_frame, action, step_index, history):
        out = work / "wam_generated" / f"step_{step_index:04d}.png"
        _write_frame(out, seed=step_index * 17)
        video = out.with_suffix(".mp4")
        video.write_bytes(b"fake mp4")
        return {
            "generated_frame_path": str(out),
            "generated_video_path": str(video),
            "wam_backend": "oscar_wam",
        }

    return _generate


def _write_episode_consistency_command(tmp_path: Path, *, inverse_consistent: bool) -> Path:
    command = tmp_path / (
        "wam_consistency_pass.py" if inverse_consistent else "wam_consistency_fail.py"
    )
    command.write_text(
        f"""
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["BLUEPRINT_WAM_CONSISTENCY_INPUT"]).read_text(encoding="utf-8"))
assert request["schema_version"] == "wam_episode_consistency_request.v1"
assert request["claim_boundary"]["scorer_is_separate_from_wam_execution_and_evaluator"] is True
rollout = request["rollouts"][0]
payload = {{
    "schema_version": "wam_episode_consistency.command.v1",
    "status": "completed",
    "provider": "fake-vlm-episode-consistency",
    "model": "fake-vlm",
    "rollout_checks": [
        {{
            "rollout_id": rollout["rollout_id"],
            "scenario_eval_run_id": rollout["scenario_eval_run_id"],
            "policy_id": rollout["policy_id"],
            "model_candidate": rollout.get("model_candidate"),
            "forward_consistent": True,
            "inverse_consistent": {inverse_consistent!r},
            "confidence": 0.91,
            "rationale": "The rollout was checked against action trace context.",
            "visual_evidence_used": True,
            "action_trace_evidence_used": True,
        }}
    ],
}}
Path(os.environ["BLUEPRINT_WAM_CONSISTENCY_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    return command


def _write_success_label_command(tmp_path: Path, *, success: bool) -> Path:
    command = tmp_path / ("wam_success_pass.py" if success else "wam_success_fail.py")
    command.write_text(
        f"""
import json
import os
from pathlib import Path

request = json.loads(Path(os.environ["BLUEPRINT_WAM_SUCCESS_LABEL_INPUT"]).read_text(encoding="utf-8"))
assert request["schema_version"] == "wam_success_label_request.v1"
assert request["sim_only_constraint"]["real_world_data_allowed"] == "site_capture_only"
assert request["claim_boundary"]["judge_input_is_generated_video_not_raw_robot_evidence"] is True
rollout = request["rollouts"][0]
payload = {{
    "schema_version": "wam_success_labels.command.v1",
    "status": "completed",
    "provider": "fake-generated-video-success",
    "model": "fake-vlm",
    "labels": [
        {{
            "rollout_id": rollout["rollout_id"],
            "scenario_eval_run_id": rollout["scenario_eval_run_id"],
            "policy_id": rollout["policy_id"],
            "success": {success!r},
            "confidence": 0.93,
            "rationale": "The generated video was reviewed against the task prompt.",
            "task_completion_evidence": ["visible robot-caused target state change"]
            if {success!r}
            else [],
            "failure_modes": []
            if {success!r}
            else ["target_state_change_not_visible"],
            "visual_evidence_used": True,
        }}
    ],
}}
Path(os.environ["BLUEPRINT_WAM_SUCCESS_LABEL_OUTPUT"]).write_text(json.dumps(payload), encoding="utf-8")
""".strip(),
        encoding="utf-8",
    )
    return command


def _passed_visual_smoke(**kwargs):
    return {
        "schema_version": "wam_generated_rollout_visual_smoke.v1",
        "generated_at": "now",
        "status": "passed_visual_quality_smoke",
        "blockers": [],
        "rollout_count": len(kwargs.get("rollouts") or []),
        "claim_boundary": {
            "visual_rollout_useful_for_task_success_review": True,
            "generated_video_is_not_task_success_proof": True,
        },
    }


def _write_seed_geometry_route(tmp_path: Path) -> tuple[Path, Path]:
    render_dir = tmp_path / "render"
    source_render = render_dir / "frames" / "robot_pov_0000.png"
    _write_frame(source_render, seed=18)
    seed = tmp_path / "selected_seed.jpg"
    _write_frame(seed, seed=19)
    (render_dir / "manipulation_pov_geometry.json").write_text(
        json.dumps(
            {
                "schema_version": "manipulation_pov_geometry_index.v1",
                "frames": [
                    {
                        "status": "PASS",
                        "camera": "robot_pov",
                        "seed_frame_quality": {"image_size_px": [64, 48]},
                        "target_projection": {"available": True, "u_px": 50, "v_px": 24},
                        "projected_landmarks": [
                            {
                                "landmark_id": "left_shoulder",
                                "link_role": "shoulder",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 18,
                                    "v_px": 34,
                                    "depth_m": 0.35,
                                },
                            },
                            {
                                "landmark_id": "left_elbow",
                                "link_role": "elbow",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 22,
                                    "v_px": 32,
                                    "depth_m": 0.34,
                                },
                            },
                            {
                                "landmark_id": "left_wrist",
                                "link_role": "wrist",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 26,
                                    "v_px": 30,
                                    "depth_m": 0.32,
                                },
                            },
                            {
                                "landmark_id": "left_hand",
                                "link_role": "hand",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 30,
                                    "v_px": 28,
                                    "depth_m": 0.3,
                                },
                            },
                            {
                                "landmark_id": "right_shoulder",
                                "link_role": "shoulder",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 46,
                                    "v_px": 34,
                                    "depth_m": 0.35,
                                },
                            },
                            {
                                "landmark_id": "right_elbow",
                                "link_role": "elbow",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 42,
                                    "v_px": 32,
                                    "depth_m": 0.34,
                                },
                            },
                            {
                                "landmark_id": "right_wrist",
                                "link_role": "wrist",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 38,
                                    "v_px": 30,
                                    "depth_m": 0.32,
                                },
                            },
                            {
                                "landmark_id": "right_hand",
                                "link_role": "hand",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 34,
                                    "v_px": 28,
                                    "depth_m": 0.3,
                                },
                            }
                        ],
                        "segments": [
                            {"from": "left_shoulder", "to": "left_elbow"},
                            {"from": "left_elbow", "to": "left_wrist"},
                            {"from": "left_wrist", "to": "left_hand"},
                            {"from": "right_shoulder", "to": "right_elbow"},
                            {"from": "right_elbow", "to": "right_wrist"},
                            {"from": "right_wrist", "to": "right_hand"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source_trace = render_dir / "trace.jsonl"
    source_trace.write_text("{}\n", encoding="utf-8")
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps(
            {
                "route_points": [[0.0, 0.0, 0.79], [1.0, 0.0, 0.79]],
                "source_trace": str(source_trace),
            }
        ),
        encoding="utf-8",
    )
    return seed, route


def _write_passed_short_visual_sanity_manifest(
    root: Path, policy_observation_path: Path
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source_qa = root / "source_policy_observation_visual_qa.json"
    report = root / "wam_rollout_visual_quality_report.json"
    contact_sheet = _write_frame(root / "wam_rollout_contact_sheet.jpg", seed=91)
    video_status = root / "video_review_status.json"
    review_video = root / "review_video" / "persistent_policy_wam_live_rollout_review.mp4"
    review_video.parent.mkdir(parents=True, exist_ok=True)
    source_qa.write_text(
        json.dumps({"status": "passed_visual_quality_gate"}),
        encoding="utf-8",
    )
    report.write_text(
        json.dumps(
            {
                "status": "passed_visual_quality_gate",
                "visual_profile": "review_quality",
                "visual_success": True,
                "profile_contract": {
                    "review_quality_profile": True,
                    "review_quality_minimum_satisfied": True,
                    "smoke_only": False,
                },
            }
        ),
        encoding="utf-8",
    )
    ffprobe_metadata = {
        "streams": [
            {
                "width": 640,
                "height": 480,
                "avg_frame_rate": "15/1",
                "nb_frames": "24",
            }
        ],
        "format": {"duration": "1.6", "size": "1000"},
    }
    video_status.write_text(
        json.dumps(
            {
                "status": "completed",
                "ffprobe_command_ran": True,
                "ffprobe_returncode": 0,
                "ffprobe_metadata": ffprobe_metadata,
            }
        ),
        encoding="utf-8",
    )
    review_video.write_bytes(b"mp4")
    task_success_judge = root / "persistent_wam_task_success_judge.json"
    task_success_judge.write_text(
        json.dumps(
            {
                "schema_version": "persistent_wam_task_success_judge.v1",
                "generated_at": "now",
                "status": "not_proven",
                "answer": "not_proven",
                "task_success_proven": False,
                "true_manipulation_success_proven": False,
                "blockers": ["true_manipulation_success_not_proven"],
                "generated_video_semantic_success_label": {
                    "status": "requires_review",
                    "success_label_from_generated_video": False,
                    "generated_video_semantic_success": False,
                    "generated_video_semantic_failure": False,
                },
                "claim_boundary": {
                    "visual_quality_is_not_task_success": True,
                    "generated_video_semantic_label_is_support_only": True,
                    "task_success_proof_requires_evaluator_or_physics_state": True,
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = root / "persistent_wam_short_visual_sanity_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "persistent_wam_short_visual_sanity.v1",
                "generated_at": "now",
                "status": "passed_short_visual_sanity",
                "short_visual_sanity_passed": True,
                "policy_observation_path": str(policy_observation_path.resolve()),
                "provider": "vast",
                "requested_transition_count": 2,
                "requested_loop_step_count": 3,
                "generated_transition_count": 2,
                "visual_profile": "review_quality",
                "claim_boundary": {
                    "short_visual_sanity_is_not_task_success_proof": True,
                    "visual_sanity_passed_is_not_task_success": True,
                    "visual_quality_is_not_task_success": True,
                    "task_success_judge_required_for_task_success_claim": True,
                    "generated_video_success_label_is_support_only": True,
                },
                "task_success_judge_path": str(task_success_judge),
                "task_success_judge_status": "not_proven",
                "task_success_proven": False,
                "true_manipulation_success_proven": False,
                "generated_video_task_success_label_from_generated_video": False,
                "generated_video_task_success_label_status": "requires_review",
                "generated_video_semantic_success": False,
                "generated_video_semantic_failure": False,
                "task_success_blockers": ["true_manipulation_success_not_proven"],
                "source_policy_observation_visual_qa_status": (
                    "passed_visual_quality_gate"
                ),
                "source_policy_observation_visual_qa_path": str(source_qa),
                "wam_rollout_visual_success": True,
                "wam_rollout_visual_quality_report_path": str(report),
                "wam_rollout_contact_sheet_path": str(contact_sheet),
                "video_review_status_path": str(video_status),
                "review_video_path": str(review_video),
                "ffprobe_command_ran": True,
                "ffprobe_returncode": 0,
                "ffprobe_metadata": ffprobe_metadata,
                "live_wam_generation_success_count": 2,
                "learned_wam_model_success_count": 2,
                "structural_fallback_used": False,
                "paid_provider": {
                    "provider": "vast",
                    "used": False,
                    "teardown_status": "not_required_no_paid_provider",
                    "teardown_performed": False,
                    "continuing_spend_from_this_run": False,
                },
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _allow_vast_paid_provider(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    key_file = tmp_path / "vast_api_key"
    key_file.write_text("redacted-vast-test-key\n", encoding="utf-8")
    budget = tmp_path / "fresh_vast_budget.json"
    budget.write_text(
        json.dumps({"schema_version": "vast_session_cost_summary.v4", "attempts": []}),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    monkeypatch.setenv("VAST_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("VAST_SESSION_BUDGET_LEDGER_FILE", str(budget))
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_MAX_HOURLY_RATE", "0.25")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_MAX_LIVE_MINUTES", "10")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_SESSION_MAX_LIVE_MINUTES", "30")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_HARD_CAP_USD", "0.50")
    return key_file


def test_closed_loop_runs_policy_wam_harness_per_step(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=3)
    route = [(-4.25, -3.35, 0.79), (-1.0, -1.0, 0.79), (1.75, 1.25, 0.79)]
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=route,
        wam_generate_next=_stub_wam(tmp_path),
        steps=4,
        harness_backend_kind="fixture",
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["loop_kind"] == "per_step_policy_wam_perception_closed_loop"
    assert manifest["steps_executed"] == 4
    assert manifest["real_perception_backend_used"] is False  # fixture in this hermetic test

    # the trace proves per-step: policy action -> WAM frame -> harness ran, each step
    trace = [
        json.loads(line)
        for line in Path(manifest["trace_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(trace) == 4
    assert [r["step_index"] for r in trace] == [1, 2, 3, 4]
    for row in trace:
        assert row["policy_action"]  # policy acted
        assert Path(row["wam_generated_frame"]).is_file()  # WAM produced the next obs
        assert row["harness_step_status"] == "completed"  # harness ran on it
    # feed-forward: step 2's source observation is step 1's generated frame
    assert trace[1]["source_observation_frame"] == trace[0]["wam_generated_frame"]


def test_closed_loop_requeries_learned_policy_on_wam_observation(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=5)
    route = [(-4.25, -3.35, 0.79), (-1.0, -1.0, 0.79), (1.75, 1.25, 0.79)]
    endpoint_calls: list[dict[str, object]] = []
    endpoint_actions: list[dict[str, object]] = []

    def _policy_endpoint(obs, history, step_index):
        frame_path = Path(str(obs.get("camera_frame_path")))
        nested_visual = obs.get("visual_observation") or {}
        assert str(frame_path) == str(nested_visual.get("camera_frame_path"))
        red_channel = Image.open(frame_path).getpixel((0, 0))[0]
        action = {
            "root_position": [round(red_channel / 100.0, 6), round(step_index / 10.0, 6), 0.79],
            "root_yaw_radians": round(red_channel / 1000.0, 6),
            "policy_action": "learned_policy_action",
        }
        endpoint_calls.append(
            {"step_index": step_index, "camera_frame_path": str(frame_path), "history": list(history)}
        )
        endpoint_actions.append(action)
        return action

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=route,
        wam_generate_next=_stub_wam(tmp_path),
        steps=3,
        harness_backend_kind="fixture",
        generated_at="now",
        policy_endpoint=_policy_endpoint,
    )

    assert manifest["status"] == "completed"
    assert manifest["proof"]["simulator_backend"] == "isaac"
    assert manifest["proof"]["learned_policy_requery_count"] == 3
    assert manifest["proof"]["policy_action_changed_count"] >= 2
    assert manifest["proof"]["policy_observes_wam_generated_next_observation"] is True
    assert not any(str(key).startswith("unitree_") for key in manifest["proof"])

    trace = [
        json.loads(line)
        for line in Path(manifest["trace_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(endpoint_calls) == manifest["steps_executed"] == 3
    for call, row in zip(endpoint_calls, trace):
        assert call["camera_frame_path"] == row["wam_generated_frame"]
        assert call["camera_frame_path"] != str(start.resolve())
    assert endpoint_actions[0]["root_position"] != endpoint_actions[1]["root_position"]
    assert trace[0]["policy_action_from_wam_requery"] is False
    assert trace[1]["policy_action_from_wam_requery"] is True
    assert trace[1]["root_position"] == endpoint_actions[0]["root_position"]
    assert trace[2]["root_position"] == endpoint_actions[1]["root_position"]
    assert all(row["policy_requeried_on_wam_observation"] is True for row in trace)
    assert all(row["requery_status"] == "completed" for row in trace)


def test_completion_does_not_overclaim_without_learned_requery(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=8)
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam(tmp_path),
        steps=4,
        harness_backend_kind="fixture",
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["steps_executed"] == 4
    assert manifest["policy_observes_wam_generated_next_observation"] is False
    assert manifest["wam_evaluator_in_control_loop"] is True
    assert manifest["clean_frame_reanchoring"] == {
        "enabled": False,
        "interval_steps": None,
        "source_frame_kind": "initial_policy_observation_clean_frame",
    }
    assert manifest["clean_frame_reanchor_event_count"] == 0
    assert manifest["periodic_clean_frame_reanchoring_used"] is False
    assert manifest["proof"]["fresh_learned_policy_requery_steps"] == 0
    assert manifest["proof"]["policy_observes_wam_generated_next_observation"] is False

    trace = [
        json.loads(line)
        for line in Path(manifest["trace_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert trace[1]["source_observation_frame"] == trace[0]["wam_generated_frame"]
    assert all(row["clean_frame_reanchor_applied"] is False for row in trace)


def test_require_fresh_learned_policy_requery_blocks_deterministic_and_passes_with_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=9)
    route = [(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)]
    blocked = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "blocked",
        start_frame_path=start,
        route_points=route,
        wam_generate_next=_stub_wam(tmp_path),
        steps=3,
        harness_backend_kind="fixture",
        generated_at="now",
        require_fresh_learned_policy_requery=True,
    )
    assert blocked["status"] == "blocked"
    assert "fresh_learned_policy_requery_not_proven" in blocked["blockers"]

    original_action_record = L.action_record

    def _fresh_action_record(**kwargs):
        record = original_action_record(**kwargs)
        record["policy_requeried_on_generated_observation"] = True
        record["policy_action_source"] = "test_fresh_learned_policy_requery"
        return record

    monkeypatch.setattr(L, "action_record", _fresh_action_record)
    completed = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "completed",
        start_frame_path=start,
        route_points=route,
        wam_generate_next=_stub_wam(tmp_path),
        steps=3,
        harness_backend_kind="fixture",
        generated_at="now",
        require_fresh_learned_policy_requery=True,
    )
    assert completed["status"] == "completed"
    assert completed["proof"]["fresh_learned_policy_requery_steps"] == 3
    assert completed["policy_observes_wam_generated_next_observation"] is True
    assert completed["wam_evaluator_in_control_loop"] is True


def test_clean_frame_reanchoring_feeds_seed_frame_back(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=10)
    route = [(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)]
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "reanchor",
        start_frame_path=start,
        route_points=route,
        wam_generate_next=_stub_wam(tmp_path),
        steps=4,
        harness_backend_kind="fixture",
        generated_at="now",
        clean_frame_reanchor_interval=2,
    )
    trace = [
        json.loads(line)
        for line in Path(manifest["trace_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resolved_start = str(start.resolve())
    assert manifest["status"] == "completed"
    assert manifest["clean_frame_reanchor_event_count"] == 2
    assert manifest["periodic_clean_frame_reanchoring_used"] is True
    assert manifest["clean_frame_reanchoring"]["enabled"] is True
    assert manifest["clean_frame_reanchoring"]["interval_steps"] == 2
    assert [event["step_index"] for event in manifest["clean_frame_reanchor_events"]] == [2, 4]
    assert manifest["clean_frame_reanchor_events"][0]["next_policy_observation_frame_path"] == resolved_start
    assert trace[1]["clean_frame_reanchor_applied"] is True
    assert trace[2]["source_observation_frame"] == resolved_start
    assert manifest["proof"]["feed_forward_verified"] is True

    default = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "default",
        start_frame_path=start,
        route_points=route,
        wam_generate_next=_stub_wam(tmp_path),
        steps=3,
        harness_backend_kind="fixture",
        generated_at="now",
        clean_frame_reanchor_interval=0,
    )
    default_trace = [
        json.loads(line)
        for line in Path(default["trace_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert default["clean_frame_reanchoring"]["enabled"] is False
    assert default["clean_frame_reanchor_event_count"] == 0
    assert default_trace[1]["source_observation_frame"] == default_trace[0]["wam_generated_frame"]


def test_closed_loop_emits_in_process_success_evaluator_not_proven_by_default(
    tmp_path: Path,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=11)
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam(tmp_path),
        steps=4,
        harness_backend_kind="fixture",
        generated_at="now",
    )

    judge_path = Path(manifest["manipulation_success_evaluator_results_path"])
    assert judge_path.is_file()
    judge = json.loads(judge_path.read_text(encoding="utf-8"))
    assert judge["schema_version"] == "isaac_manipulation_success_evaluator_results.v1"
    assert judge["simulator_backend"] == "isaac"
    assert judge["success_proof_separate_from_structural_loop_proof"] is True
    assert "mujoco" not in json.dumps(judge).lower()
    assert "vast" not in json.dumps(judge).lower()
    assert manifest["status"] == "completed"
    assert manifest["task_target_reached"] is True
    assert judge["answer"] == "not_proven"
    assert judge["manipulation_success_proven"] is False
    assert manifest["manipulation_success_proven"] is False
    assert manifest["success_proof"]["success_proof_separate_from_structural_loop_proof"] is True
    assert manifest["success_proof"]["structural_loop_completed"] is True
    assert "feed_forward_verified" in manifest["proof"]
    assert "external judge" not in manifest["claim_boundary"]
    assert "manipulation_success_evaluator" in manifest["claim_boundary"]


def test_closed_loop_external_episode_consistency_stays_separate_from_task_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=12)
    command = _write_episode_consistency_command(tmp_path, inverse_consistent=True)
    monkeypatch.setenv("BLUEPRINT_ALLOW_WAM_EPISODE_CONSISTENCY_SCORING", "true")
    monkeypatch.setattr(L, "visual_smoke_generated_rollouts_for_review", _passed_visual_smoke)

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam_with_video(tmp_path),
        steps=2,
        harness_backend_kind="fixture",
        generated_at="now",
        wam_consistency_command=f"{sys.executable} {command}",
        allow_wam_consistency_scoring=True,
        wam_consistency_timeout_seconds=5.0,
    )

    assert manifest["status"] == "completed"
    assert manifest["steps_executed"] == 2
    assert manifest["external_episode_consistency_scorer_ran"] is True
    assert manifest["forward_inverse_consistency_proven"] is True
    assert manifest["wam_episode_consistency_early_termination_recommended"] is False
    assert manifest["manipulation_success_proven"] is False
    assert manifest["success_proof"]["answer"] == "not_proven"
    assert "task success" in manifest["claim_boundary"]

    for request_path in manifest["wam_episode_consistency_request_paths"]:
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        assert request["schema_version"] == "wam_episode_consistency_request.v1"
        assert request["rollouts"][0]["generated_video_path"].endswith(".mp4")
    checks = json.loads(
        Path(manifest["wam_consistency_checks_paths"][0]).read_text(encoding="utf-8")
    )
    assert checks["forward_inverse_consistency_proven"] is True
    assert checks["claim_boundary"][
        "forward_inverse_consistency_does_not_prove_task_success"
    ] is True

    judge = json.loads(
        Path(manifest["manipulation_success_evaluator_results_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert judge["manipulation_success_proven"] is False


def test_closed_loop_generated_video_success_label_is_sim_only_and_required(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=31)
    consistency_command = _write_episode_consistency_command(
        tmp_path, inverse_consistent=True
    )
    success_command = _write_success_label_command(tmp_path, success=True)
    monkeypatch.setenv("BLUEPRINT_ALLOW_WAM_EPISODE_CONSISTENCY_SCORING", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_WAM_SUCCESS_LABELING", "true")
    monkeypatch.setattr(L, "visual_smoke_generated_rollouts_for_review", _passed_visual_smoke)

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam_with_video(tmp_path),
        steps=2,
        harness_backend_kind="fixture",
        generated_at="now",
        perception_target_prompts=["open the fridge"],
        wam_consistency_command=f"{sys.executable} {consistency_command}",
        allow_wam_consistency_scoring=True,
        require_forward_inverse_consistency=True,
        wam_success_label_command=f"{sys.executable} {success_command}",
        allow_wam_success_labeling=True,
        require_generated_video_success_label=True,
        wam_consistency_timeout_seconds=5.0,
        wam_success_label_timeout_seconds=5.0,
    )

    assert manifest["status"] == "completed"
    assert manifest["forward_inverse_consistency_proven"] is True
    assert manifest["generated_video_success_label_passed"] is True
    assert manifest["simulated_manipulation_success_shown"] is True
    assert manifest["real_world_task_success_proven"] is False
    assert manifest["manipulation_success_proven"] is False
    assert manifest["success_proof"]["generated_video_success_label_is_sim_only"] is True
    assert manifest["success_proof"]["real_world_task_success_proven"] is False
    assert "generated_video_success_label_not_proven" not in manifest["blockers"]

    request = json.loads(
        Path(manifest["generated_video_success_label_request_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert request["schema_version"] == "wam_success_label_request.v1"
    assert request["sim_only_constraint"]["real_world_data_allowed"] == "site_capture_only"
    assert request["sim_only_constraint"]["physical_robot_rollout_used"] is False
    assert request["rollouts"][0]["generated_video_path"].endswith(".mp4")
    assert request["task_prompts"][0]["task_prompt"] == "open the fridge"

    labels = json.loads(
        Path(manifest["generated_video_success_labels_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert labels["status"] == "completed"
    assert labels["labels"][0]["review_task_success"] is True
    assert labels["claim_boundary"][
        "success_label_does_not_prove_forward_inverse_consistency"
    ] is True


def test_closed_loop_required_generated_video_success_label_blocks_without_labeler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=32)
    monkeypatch.setattr(L, "visual_smoke_generated_rollouts_for_review", _passed_visual_smoke)

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam_with_video(tmp_path),
        steps=2,
        harness_backend_kind="fixture",
        generated_at="now",
        require_generated_video_success_label=True,
    )

    assert manifest["status"] == "blocked"
    assert manifest["generated_video_success_label_passed"] is False
    assert manifest["simulated_manipulation_success_shown"] is False
    assert "generated_video_success_label_not_proven" in manifest["blockers"]
    assert "generated_video_success_label:requires_wam_success_review" in manifest[
        "blockers"
    ]
    assert manifest["real_world_task_success_proven"] is False


def test_closed_loop_required_forward_inverse_consistency_blocks_without_scorer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=33)
    monkeypatch.setattr(L, "visual_smoke_generated_rollouts_for_review", _passed_visual_smoke)

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam_with_video(tmp_path),
        steps=2,
        harness_backend_kind="fixture",
        generated_at="now",
        require_forward_inverse_consistency=True,
    )

    assert manifest["status"] == "blocked"
    assert manifest["forward_inverse_consistency_proven"] is False
    assert "forward_inverse_consistency_not_proven" in manifest["blockers"]


def test_closed_loop_episode_consistency_failure_early_terminates_feed_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=13)
    command = _write_episode_consistency_command(tmp_path, inverse_consistent=False)
    monkeypatch.setenv("BLUEPRINT_ALLOW_WAM_EPISODE_CONSISTENCY_SCORING", "true")
    monkeypatch.setattr(L, "visual_smoke_generated_rollouts_for_review", _passed_visual_smoke)

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_stub_wam_with_video(tmp_path),
        steps=3,
        harness_backend_kind="fixture",
        generated_at="now",
        wam_consistency_command=f"{sys.executable} {command}",
        allow_wam_consistency_scoring=True,
        wam_consistency_timeout_seconds=5.0,
    )

    assert manifest["status"] == "blocked"
    assert manifest["steps_executed"] == 1
    assert manifest["external_episode_consistency_scorer_ran"] is True
    assert manifest["forward_inverse_consistency_proven"] is False
    assert manifest["wam_episode_consistency_early_termination_recommended"] is True
    assert any(
        blocker == "wam_episode_consistency_step_1:wam_consistency_inverse_not_proven"
        for blocker in manifest["blockers"]
    )
    assert manifest["manipulation_success_proven"] is False

    trace = [
        json.loads(line)
        for line in Path(manifest["trace_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(trace) == 1
    assert trace[0]["wam_episode_consistency_early_termination_recommended"] is True
    assert "wam_consistency_inverse_not_proven" in trace[0]["wam_episode_consistency_blockers"]


def test_isaac_manipulation_success_evaluator_proves_only_on_learned_signal() -> None:
    proof = {
        "learned_policy_requery_steps": 2,
        "manipulation_success_signal": "success",
        "fresh_oscar_provider_model_run_steps": 2,
        "real_perception_backend_steps": 1,
    }
    trace_rows = [
        {"policy_action_conditioned_on_wam_generated_observation": True},
        {"policy_action_conditioned_on_wam_generated_observation": True},
    ]
    judge = L.evaluate_isaac_manipulation_success(
        generated_at="now",
        status="completed",
        proof=proof,
        trace_rows=trace_rows,
        task_target_reached=True,
        perception_target_prompts=["open the fridge"],
    )
    assert judge["manipulation_success_proven"] is True
    assert judge["did_target_manipulation_succeed"] is True
    assert judge["answer"] == "yes"
    assert judge["question"] == "open the fridge"
    assert judge["success_proof_separate_from_structural_loop_proof"] is True

    unjudged = L.evaluate_isaac_manipulation_success(
        generated_at="now",
        status="completed",
        proof={key: value for key, value in proof.items() if key != "manipulation_success_signal"},
        trace_rows=trace_rows,
        task_target_reached=True,
        perception_target_prompts=[],
    )
    assert unjudged["manipulation_success_proven"] is False
    assert unjudged["answer"] == "not_proven"
    assert "no task-success signal" in unjudged["reason"]
    assert unjudged["task_target_reached"] is True
    assert unjudged["kinematic_route_reached_is_not_manipulation_success"] is True


def test_closed_loop_blocks_on_missing_wam_frame(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=1)

    def _bad_wam(current_frame, action, step_index, history):
        return {"generated_frame_path": str(tmp_path / "does_not_exist.png")}

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (2.0, 0.0, 0.79)],
        wam_generate_next=_bad_wam,
        steps=3,
        generated_at="now",
    )
    assert manifest["status"] == "blocked"
    assert any("wam_generation_missing_frame" in b for b in manifest["blockers"])


def test_build_oscar_per_step_request_shapes_conditioning(tmp_path: Path) -> None:
    action = {
        "policy_action": "accepted_direct_collision_checked_motion",
        "root_position": [1.0, 2.0, 0.79],
        "root_yaw_radians": 0.5,
    }
    landmarks = [{"landmark_id": "pelvis", "image_projection": {"available": True, "u_px": 1, "v_px": 2}}]
    req = L.build_oscar_per_step_request(
        current_frame_path="/frames/cur.png",
        action=action,
        step_index=3,
        task_prompt="walk to the sink",
        num_frames=8,
        output_dir=tmp_path,
        skeleton_landmarks=landmarks,
        seed=42,
    )
    assert req["reference_frame_path"] == "/frames/cur.png"
    assert req["task_prompt"] == "walk to the sink"
    assert req["num_frames"] == 8
    assert req["seed"] == 45  # base seed + step_index
    assert req["projected_landmark_count"] == 1
    assert req["skeleton_landmarks"] == landmarks
    assert req["output_dir"].endswith("oscar_step_0003")


def test_oscar_per_step_backend_drives_the_loop(tmp_path: Path) -> None:
    """The real GPU path with OSCAR mocked: each step calls per-step OSCAR generation, the
    harness runs on the generated frame. Swapping the mock for a real OSCAR pod + real SAM3
    backend is the only change for the GPU run.
    """
    calls: list[dict] = []

    def _fake_oscar_generate(request):
        calls.append(dict(request))
        frame = tmp_path / "oscar_out" / f"step_{request['step_index']:04d}.png"
        _write_frame(frame, seed=request["step_index"] * 23 + 5)
        return {
            "status": "completed",
            "generated_frame_path": str(frame),
            "generated_video_path": str(frame.with_suffix(".mp4")),
        }

    def _skeleton_for_action(action, step_index):
        return [{"landmark_id": "pelvis", "image_projection": {"available": True, "u_px": step_index, "v_px": 1}}]

    backend = L.make_oscar_per_step_wam_backend(
        oscar_generate=_fake_oscar_generate,
        work_dir=tmp_path / "oscar_work",
        task_prompt="walk to the sink",
        num_frames=8,
        skeleton_for_action=_skeleton_for_action,
    )
    start = _write_frame(tmp_path / "start.png", seed=2)
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(-4.25, -3.35, 0.79), (1.75, 1.25, 0.79)],
        wam_generate_next=backend,
        steps=3,
        harness_backend_kind="fixture",
        generated_at="now",
    )
    assert manifest["status"] == "completed"
    assert manifest["steps_executed"] == 3
    assert len(calls) == 3  # OSCAR called once per step
    # each per-step request carried the step's action + projected skeleton conditioning
    assert calls[0]["task_prompt"] == "walk to the sink"
    assert all(c["projected_landmark_count"] == 1 for c in calls)
    assert [c["step_index"] for c in calls] == [1, 2, 3]


def test_provider_command_backend_writes_step_input_and_extracts_next_frame(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=4)
    captured: dict[str, object] = {}
    captured_input_paths: list[str] = []
    projected_trace = tmp_path / "g1_projected_skeleton_trace.jsonl"
    projected_trace.write_text(
        json.dumps(
            {
                "schema_version": "blueprint.g1.projected_upper_body_skeleton.v1",
                "status": "completed",
                "projected_landmark_count": 1,
                "landmarks": [
                    {
                        "landmark_id": "right_hand",
                        "image_projection": {"available": True, "u_px": 10, "v_px": 12},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def _fake_adapter(argv):
        captured["argv"] = list(argv or [])
        input_path = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_INPUT"])
        output_path = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
        captured["input_path"] = str(input_path)
        captured_input_paths.append(str(input_path))
        captured["runtime_env"] = {
            "num_frames": os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_FRAMES"),
            "num_steps": os.environ.get("BLUEPRINT_OSCAR_WAM_NUM_STEPS"),
            "guidance": os.environ.get("BLUEPRINT_OSCAR_WAM_GUIDANCE"),
            "seed": os.environ.get("BLUEPRINT_OSCAR_WAM_SEED"),
            "height": os.environ.get("BLUEPRINT_OSCAR_WAM_HEIGHT"),
            "width": os.environ.get("BLUEPRINT_OSCAR_WAM_WIDTH"),
            "fps": os.environ.get("BLUEPRINT_OSCAR_WAM_FPS"),
        }
        video = output_path.parent / "oscar_generated_rollout.mp4"
        video.write_bytes(b"fake mp4")
        payload = {
            "status": "completed",
            "fresh_provider_model_run_claimed": True,
            "provider_learned_wam_model_ran": True,
            "provider_generated_video_is_model_output": True,
            "rollouts": [{"generated_video_path": str(video)}],
            "blockers": [],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _extract(video_path, out_dir):
        assert Path(video_path).is_file()
        return _write_frame(Path(out_dir) / "next.png", seed=19)

    backend = L.make_oscar_provider_command_wam_backend(
        work_dir=tmp_path / "provider_loop",
        task_prompt="walk to the sink",
        num_frames=12,
        num_steps=27,
        guidance=4.25,
        seed=100,
        height=240,
        width=320,
        fps=10.0,
        provider="vast",
        allow_paid_provider_launch=True,
        adapter_run=_fake_adapter,
        extract_next_frame=_extract,
        projected_skeleton_trace_path=projected_trace,
    )
    result = backend(
        str(start),
        {"policy_action": "accepted_direct_collision_checked_motion", "root_position": [0, 0, 0.79]},
        1,
        [],
    )

    assert result["status"] == "completed"
    assert result["wam_backend"] == "oscar_2b_per_step_provider"
    assert result["fresh_provider_model_run_claimed"] is True
    assert Path(result["generated_frame_path"]).is_file()
    assert "--allow-paid-provider-launch" in captured["argv"]
    assert captured["runtime_env"] == {
        "num_frames": "12",
        "num_steps": "27",
        "guidance": "4.25",
        "seed": "101",
        "height": "240",
        "width": "320",
        "fps": "10.0",
    }
    step_input = json.loads(Path(str(captured["input_path"])).read_text(encoding="utf-8"))
    assert step_input["schema_version"] == "wam_generation_step_input.v1"
    assert step_input["source_policy_action"]["task_prompt"] == "walk to the sink"
    visual = step_input["current_policy_observation"]["visual_observation"]
    assert visual["g1_projected_skeleton_trace_jsonl"] == str(projected_trace.resolve())

    step2 = backend(
        str(result["generated_frame_path"]),
        {
            "policy_action": "accepted_direct_collision_checked_motion",
            "root_position": [0.25, 0, 0.79],
        },
        2,
        [{"step_index": 1, "wam_generated_frame": result["generated_frame_path"]}],
    )
    assert step2["status"] == "completed"
    step2_input = json.loads(Path(captured_input_paths[-1]).read_text(encoding="utf-8"))
    step2_visual = step2_input["current_policy_observation"]["visual_observation"]
    assert step2_visual["g1_projected_skeleton_trace_jsonl"] == str(projected_trace.resolve())
    assert step2_visual["projected_skeleton_trace_path"] == str(projected_trace.resolve())


def test_provider_command_backend_blocks_failed_visual_smoke_before_extracting_frame(
    tmp_path: Path,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=41)
    extractor_called = False

    def _fake_adapter(_argv):
        output_path = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_OUTPUT"])
        video = output_path.parent / "oscar_generated_rollout.mp4"
        video.write_bytes(b"fake mp4")
        payload = {
            "status": "completed",
            "fresh_provider_model_run_claimed": True,
            "provider_learned_wam_model_ran": True,
            "provider_generated_video_is_model_output": True,
            "generated_rollout_visual_smoke_status": "failed_visual_quality_smoke",
            "generated_rollout_visual_quality_blockers": [
                "generated_rollout_later_frames_edge_structure_drift",
                "generated_rollout_later_frames_entropy_drift",
            ],
            "rollouts": [{"generated_video_path": str(video)}],
            "blockers": [],
        }
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def _extract(_video_path, _out_dir):
        nonlocal extractor_called
        extractor_called = True
        return None

    backend = L.make_oscar_provider_command_wam_backend(
        work_dir=tmp_path / "provider_loop",
        task_prompt="open the refrigerator",
        adapter_run=_fake_adapter,
        extract_next_frame=_extract,
    )
    result = backend(
        str(start),
        {"policy_action": "accepted_direct_collision_checked_motion"},
        1,
        [],
    )

    assert result["status"] == "blocked"
    assert result["generated_frame_path"] == ""
    assert Path(result["generated_video_path"]).is_file()
    assert extractor_called is False
    assert "provider_generated_rollout_visual_smoke_not_passed" in result["blockers"]
    assert (
        "generated_rollout_later_frames_edge_structure_drift"
        in result["blockers"]
    )
    assert "generated_rollout_later_frames_entropy_drift" in result["blockers"]


def test_materialize_projected_skeleton_trace_from_seed_geometry_scales_to_seed(
    tmp_path: Path,
) -> None:
    source_render = tmp_path / "render" / "frames" / "robot_pov_0000.png"
    _write_frame(source_render, seed=9)
    seed = tmp_path / "selected_seed.jpg"
    _write_frame(seed, seed=10)
    geometry = tmp_path / "render" / "manipulation_pov_geometry.json"
    geometry.write_text(
        json.dumps(
            {
                "schema_version": "manipulation_pov_geometry_index.v1",
                "frames": [
                    {
                        "status": "PASS",
                        "camera": "robot_pov",
                        "seed_frame_quality": {"image_size_px": [128, 96]},
                        "target_projection": {
                            "available": True,
                            "u_px": 100,
                            "v_px": 60,
                        },
                        "projected_landmarks": [
                            {
                                "landmark_id": "left_wrist_link",
                                "link_role": "wrist",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 64,
                                    "v_px": 48,
                                    "depth_m": 0.2,
                                },
                            },
                            {
                                "landmark_id": "left_hand_link",
                                "link_role": "hand",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 96,
                                    "v_px": 72,
                                    "depth_m": 0.3,
                                },
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    trace = tmp_path / "render" / "trace.jsonl"
    trace.write_text("{}\n", encoding="utf-8")

    out = L.materialize_projected_skeleton_trace_from_seed_geometry(
        route_payload={"source_trace": str(trace)},
        start_frame_path=seed,
        output_dir=tmp_path / "conditioning",
    )

    assert out is not None and out.is_file()
    rows = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == DEFAULT_NUM_FRAMES
    first = rows[0]
    last = rows[-1]
    assert first["image_size_px"] == [64, 48]
    assert first["source_image_size_px"] == [128, 96]
    assert first["target_projection"]["u_px"] == 50.0
    assert first["target_projection"]["v_px"] == 30.0
    assert first["projected_landmark_count"] == 2
    assert first["landmarks"][0]["image_projection"]["u_px"] == 32.0
    assert first["landmarks"][0]["image_projection"]["v_px"] == 24.0
    assert first["segments"] == [{"from": "left_wrist_link", "to": "left_hand_link"}]
    assert last["temporal_progress"] == 1.0
    assert last["landmarks"][1]["image_projection"]["u_px"] > first["landmarks"][1]["image_projection"]["u_px"]
    assert last["landmarks"][1]["image_projection"]["v_px"] < first["landmarks"][1]["image_projection"]["v_px"]
    assert (
        last["claim_boundary"][
            "temporal_rows_are_target_conditioning_from_resolved_affordance_projection"
        ]
        is True
    )


def _real_backend_command(*, depth_kind: str = "depth_anything_3"):
    code = f"""
import json, os
out = os.environ["BLUEPRINT_WAM_PERCEPTION_BACKEND_OUTPUT"]
payload = {{
  "schema_version": "wam_perception_backend_result.v1",
  "status": "completed",
  "backend": {{
    "kind": "real_provider_probe",
    "status": "completed",
    "real_sam_or_depth_model_ran": True,
    "blockers": [],
    "provider_statuses": [
      {{"provider": "sam3", "ran": True, "blockers": [], "object_count": 1}},
      {{"provider": "depth", "kind": {depth_kind!r}, "ran": True, "blockers": []}}
    ]
  }},
  "objects": [{{"object_id": "sam3_target_0000", "label": "sink", "bbox": [1, 2, 10, 20], "confidence": 0.8}}],
  "depth_estimates": [{{"object_id": "generated_frame", "relative_depth": 0.5, "confidence": 0.7}}],
  "pose_estimates": [],
  "claim_boundary": {{"harness_outputs_are_derived_from_generated_pixels": True}}
}}
open(out, "w", encoding="utf-8").write(json.dumps(payload))
"""
    return [sys.executable, "-c", code]


def test_closed_loop_does_not_consume_blocked_wam_output_with_frame(
    tmp_path: Path,
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=61)

    def _blocked_wam(_current_frame, _action, step_index, _history):
        frame = _write_frame(
            tmp_path / "degraded_wam" / f"step_{step_index:04d}.png",
            seed=step_index * 61,
        )
        return {
            "status": "blocked",
            "wam_backend": "oscar_2b_per_step_provider",
            "generated_frame_path": str(frame),
            "fresh_provider_model_run_claimed": False,
            "blockers": [
                "provider_generated_rollout_visual_smoke_not_passed",
                "generated_rollout_later_frames_edge_structure_drift",
                "generated_rollout_later_frames_entropy_drift",
            ],
        }

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_blocked_wam,
        steps=2,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert manifest["steps_executed"] == 0
    assert (
        "blocked_wam_generation_at_step_1:"
        "provider_generated_rollout_visual_smoke_not_passed"
        in manifest["blockers"]
    )
    assert (
        "blocked_wam_generation_at_step_1:"
        "generated_rollout_later_frames_edge_structure_drift"
        in manifest["blockers"]
    )
    assert (
        "blocked_wam_generation_at_step_1:"
        "generated_rollout_later_frames_entropy_drift"
        in manifest["blockers"]
    )


def test_closed_loop_proof_requirements_pass_with_fresh_oscar_sam3_da3(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=6)

    def _fresh_oscar(current_frame, action, step_index, history):
        frame = _write_frame(tmp_path / "oscar" / f"step_{step_index:04d}.png", seed=step_index * 31)
        video = tmp_path / "oscar" / f"step_{step_index:04d}.mp4"
        video.write_bytes(b"fake mp4")
        return {
            "status": "completed",
            "wam_backend": "oscar_2b_per_step_provider",
            "generated_frame_path": str(frame),
            "generated_video_path": str(video),
            "fresh_provider_model_run_claimed": True,
            "provider_payload": {
                "status": "completed",
                "fresh_provider_model_run_claimed": True,
                "provider_learned_wam_model_ran": True,
                "provider_generated_video_is_model_output": True,
            },
        }

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_fresh_oscar,
        steps=2,
        harness_backend_kind="real_provider_probe",
        harness_backend_command=_real_backend_command(depth_kind="depth_anything_3"),
        allow_external_backend=True,
        require_fresh_oscar_provider=True,
        require_real_perception_backend=True,
        require_sam3_completed=True,
        require_da3_completed=True,
        generated_at="now",
    )

    assert manifest["status"] == "completed"
    assert manifest["proof"]["fresh_oscar_provider_model_run_steps"] == 2
    assert manifest["proof"]["sam3_completed_steps"] == 2
    assert manifest["proof"]["da3_completed_steps"] == 2
    assert manifest["proof"]["feed_forward_verified"] is True


def test_closed_loop_proof_does_not_count_depth_v2_as_da3(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=7)

    def _fresh_oscar(current_frame, action, step_index, history):
        frame = _write_frame(tmp_path / "oscar" / f"step_{step_index:04d}.png", seed=step_index * 33)
        return {
            "status": "completed",
            "wam_backend": "oscar_2b_per_step_provider",
            "generated_frame_path": str(frame),
            "fresh_provider_model_run_claimed": True,
        }

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[(0.0, 0.0, 0.79), (1.0, 0.0, 0.79)],
        wam_generate_next=_fresh_oscar,
        steps=1,
        harness_backend_kind="real_provider_probe",
        harness_backend_command=_real_backend_command(depth_kind="transformers_depth_anything_v2"),
        allow_external_backend=True,
        require_da3_completed=True,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "da3_provider_not_completed_at_step_1" in manifest["blockers"]
    assert manifest["proof"]["depth_completed_steps"] == 1
    assert manifest["proof"]["da3_completed_steps"] == 0


def test_build_oscar_inference_argv_mirrors_entrypoint(tmp_path: Path) -> None:
    argv = L.build_oscar_inference_argv(
        python="python",
        oscar_repo="/opt/oscar",
        checkpoint="/models/oscar/ckpt",
        first_frame_path="/frames/cur.png",
        prompt="walk to the sink",
        num_frames=8,
        num_steps=35,
        guidance=6.0,
        seed=45,
        height=480,
        width=640,
        fps=15.0,
        output_video=tmp_path / "out.mp4",
        skeleton_video=tmp_path / "skel.mp4",
    )
    assert any("inference_oscar.py" in a for a in argv)
    assert argv[argv.index("--first-frame") + 1] == "/frames/cur.png"
    assert argv[argv.index("--prompt") + 1] == "walk to the sink"
    assert argv[argv.index("--num-frames") + 1] == "8"
    assert argv[argv.index("--skeleton-video") + 1].endswith("skel.mp4")


def test_local_oscar_subprocess_generate_runs_and_extracts(tmp_path: Path) -> None:
    seen_argv: list[list[str]] = []

    class _Done:
        returncode = 0
        stdout = "oscar stdout"
        stderr = "oscar stderr"

    def _fake_run(argv, **kwargs):
        seen_argv.append(list(argv))
        # simulate inference_oscar.py writing the output clip
        out = argv[argv.index("--output") + 1]
        Path(out).write_bytes(b"\x00fakeclip")
        return _Done()

    def _fake_extract(video_path: Path, out_dir: Path):
        frame = out_dir / "next_obs.png"
        _write_frame(frame, seed=11)
        return frame

    def _fake_skeleton_video(landmarks, out_dir: Path):
        v = out_dir / "skel.mp4"
        v.write_bytes(b"\x00skel")
        return v

    gen = L.make_local_oscar_subprocess_generate(
        oscar_repo="/opt/oscar",
        checkpoint="/models/oscar/ckpt",
        run=_fake_run,
        build_skeleton_video=_fake_skeleton_video,
        extract_next_frame=_fake_extract,
    )
    request = L.build_oscar_per_step_request(
        current_frame_path="/frames/cur.png",
        action={"root_position": [1, 2, 0.79]},
        step_index=1,
        task_prompt="walk to the sink",
        num_frames=8,
        output_dir=tmp_path,
        skeleton_landmarks=[{"landmark_id": "pelvis"}],
    )
    out = gen(request)
    assert out["status"] == "completed"
    assert Path(out["generated_frame_path"]).is_file()
    assert Path(out["stdout_log_path"]).read_text(encoding="utf-8") == "oscar stdout"
    assert Path(out["stderr_log_path"]).read_text(encoding="utf-8") == "oscar stderr"
    assert "--skeleton-video" in seen_argv[0]  # skeleton conditioning passed to OSCAR


def test_local_oscar_subprocess_generate_blocks_on_nonzero(tmp_path: Path) -> None:
    class _Fail:
        returncode = 1
        stdout = "usage"
        stderr = "error: --skeleton-video required"

    gen = L.make_local_oscar_subprocess_generate(
        oscar_repo="/opt/oscar",
        checkpoint="/c",
        run=lambda argv, **k: _Fail(),
        extract_next_frame=lambda v, d: None,
    )
    out = gen({"output_dir": str(tmp_path), "reference_frame_path": "/f.png", "task_prompt": "t", "num_frames": 8, "seed": 1})
    assert out["status"] == "blocked"
    assert any("returncode" in b for b in out["blockers"])
    assert Path(out["stdout_log_path"]).read_text(encoding="utf-8") == "usage"
    assert Path(out["stderr_log_path"]).read_text(encoding="utf-8") == "error: --skeleton-video required"


def test_extract_next_observation_selects_earliest_usable_future_frame(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    seed_frame = np.zeros((24, 32, 3), dtype=np.uint8)
    seed_frame[:, ::2] = 220
    usable_future = np.zeros((24, 32, 3), dtype=np.uint8)
    usable_future[::2, :] = (235, 235, 235)
    usable_future[:, ::4] = (24, 180, 240)
    dark_late = np.full((24, 32, 3), 8, dtype=np.uint8)
    for frame in (seed_frame, usable_future, dark_late, dark_late):
        writer.write(frame)
    writer.release()

    out = L.extract_next_observation_frame_from_video(video, tmp_path / "extracted")
    assert out is not None and out.is_file()
    got = cv2.imread(str(out))
    assert got is not None and got.shape == (24, 32, 3)
    assert int(got.mean()) > 80  # selected frame 1, not the late collapsed dark frame
    selection = json.loads((tmp_path / "extracted" / "next_observation_selection.json").read_text(encoding="utf-8"))
    assert selection["status"] == "completed"
    assert selection["selected_frame_index"] == 1
    assert selection["claim_boundary"]["scene_or_task_specific_pixels_used"] is False


def test_extract_next_observation_blocks_when_future_frames_are_not_useful(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24))
    seed_frame = np.zeros((24, 32, 3), dtype=np.uint8)
    seed_frame[:, ::2] = 220
    dark_future = np.full((24, 32, 3), 8, dtype=np.uint8)
    for frame in (seed_frame, dark_future, dark_future):
        writer.write(frame)
    writer.release()

    out = L.extract_next_observation_frame_from_video(video, tmp_path / "extracted")

    assert out is None
    selection = json.loads((tmp_path / "extracted" / "next_observation_selection.json").read_text(encoding="utf-8"))
    assert selection["status"] == "blocked"
    assert selection["selected_frame_index"] is None
    assert "no_usable_future_next_observation_frame" in selection["blockers"]
    assert any(
        "next_observation_candidate_too_dark" in candidate["blockers"]
        for candidate in selection["candidates"][1:]
    )


def test_extract_next_observation_blocks_static_noise_future_frame(tmp_path: Path) -> None:
    stats = {
        "mean_luma": 97.0,
        "std_luma": 16.0,
        "luma_range": 143,
        "dark_pixel_ratio": 0.001,
        "bright_pixel_ratio": 0.0,
        "edge_density": 0.203,
    }

    blockers = L._next_observation_signal_blockers(stats)

    assert "next_observation_candidate_static_noise_artifact" in blockers


def test_extract_last_frame_uses_ffmpeg_when_cv2_missing(tmp_path: Path, monkeypatch) -> None:
    import builtins
    import subprocess

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake-video")
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "cv2":
            raise ImportError("cv2 intentionally unavailable")
        return real_import(name, *args, **kwargs)

    def fake_run(argv: list[str], **_kwargs: object):
        Path(argv[-1]).write_bytes(b"png")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(subprocess, "run", fake_run)

    out = L.extract_next_observation_frame_from_video(video, tmp_path / "extracted")

    assert out == tmp_path / "extracted" / "next_observation.png"
    assert out.is_file()
    selection = json.loads((tmp_path / "extracted" / "next_observation_selection.json").read_text(encoding="utf-8"))
    assert selection["selected_frame_index"] == 1


def test_closed_loop_blocks_on_empty_route(tmp_path: Path) -> None:
    start = _write_frame(tmp_path / "start.png", seed=1)
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "loop",
        start_frame_path=start,
        route_points=[],
        wam_generate_next=_stub_wam(tmp_path),
        steps=3,
        generated_at="now",
    )
    assert manifest["status"] == "blocked"
    assert "blocked_empty_route" in manifest["blockers"]


def test_closed_loop_wam_backend_readiness_blocks_unwired_cosmos3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND", "python cosmos3_adapter.py")

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="cosmos3_wam",
        use_provider_command=True,
        oscar_repo=None,
        checkpoint=None,
        oscar_provider="vast",
        allow_paid_provider_launch=False,
    )

    assert readiness["status"] == "blocked"
    assert readiness["selected_wam_backend"] == "cosmos3_wam"
    assert readiness["explicit_provider_command_configured"] is True
    assert "blocked_cosmos3_wam_not_wired_into_isaac_closed_loop_runner" in readiness["blockers"]
    assert (
        readiness["claim_boundary"]["cosmos3_strategy_preference_does_not_imply_runtime_wired"]
        is True
    )


def test_closed_loop_wam_backend_readiness_blocks_unpinned_local_oscar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_SOURCE_URL", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_SOURCE_REF", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_HF_REPO", raising=False)
    monkeypatch.delenv("BLUEPRINT_OSCAR_WAM_HF_REVISION", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_EXPERIMENTAL_OSCAR_WAM_VERSION", raising=False)
    source = tmp_path / "oscar-source"
    checkpoint = tmp_path / "checkpoint"
    source.mkdir()
    checkpoint.mkdir()

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="oscar_wam",
        use_provider_command=False,
        oscar_repo=str(source),
        checkpoint=str(checkpoint),
        oscar_provider="vast",
        allow_paid_provider_launch=False,
    )

    assert readiness["status"] == "blocked"
    assert "official_oscar_source_url_mismatch" in readiness["blockers"]
    assert "official_oscar_source_commit_not_pinned" in readiness["blockers"]
    assert "official_oscar_hf_revision_not_pinned" in readiness["blockers"]
    assert readiness["official_oscar_release"]["official_release_match"] is False
    assert (
        readiness["claim_boundary"]["official_oscar_source_and_checkpoint_pinned"]
        is False
    )


def test_closed_loop_wam_backend_readiness_surfaces_vast_paid_gate_blockers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_VAST_API_CALLS", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", raising=False)
    monkeypatch.setenv("VAST_API_KEY_FILE", str(tmp_path / "missing_vast_api_key"))
    monkeypatch.setenv("VAST_SESSION_BUDGET_LEDGER_FILE", str(tmp_path / "budget.json"))

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="oscar_wam",
        use_provider_command=True,
        oscar_provider="vast",
        allow_paid_provider_launch=True,
    )

    assert readiness["status"] == "blocked"
    preflight = readiness["paid_provider_preflight"]
    assert preflight["status"] == "blocked"
    assert "missing_env_BLUEPRINT_ALLOW_VAST_API_CALLS" in readiness["blockers"]
    assert "missing_env_BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH" in readiness["blockers"]
    assert "missing_file_based_secret_VAST_API_KEY_FILE" in readiness["blockers"]
    assert preflight["claim_boundary"]["preflight_does_not_call_vast_api"] is True


def test_closed_loop_wam_backend_readiness_surfaces_vast_session_budget_blockers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "vast_api_key"
    key_file.write_text("redacted-test-key\n", encoding="utf-8")
    budget = tmp_path / "budget.json"
    budget.write_text(
        json.dumps(
            {
                "schema_version": "vast_session_cost_summary.v4",
                "attempts": [
                    {
                        "estimated_cost_usd": 0.60,
                        "actual_live_runtime_seconds_observed_by_adapter": 55 * 60,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    monkeypatch.setenv("VAST_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("VAST_SESSION_BUDGET_LEDGER_FILE", str(budget))
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_MAX_HOURLY_RATE", "0.45")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_MAX_LIVE_MINUTES", "45")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_SESSION_MAX_LIVE_MINUTES", "50")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_HARD_CAP_USD", "0.75")

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="oscar_wam",
        use_provider_command=True,
        oscar_provider="vast",
        allow_paid_provider_launch=True,
    )

    preflight = readiness["paid_provider_preflight"]
    assert readiness["status"] == "blocked"
    assert preflight["prior_estimated_cost_usd"] == 0.6
    assert preflight["prior_live_runtime_minutes"] == 55.0
    assert "session_live_runtime_limit_exhausted" in readiness["blockers"]
    assert "session_estimated_spend_hard_cap_exhausted" in readiness["blockers"]
    assert preflight["raw_secret_values_recorded"] is False
    assert "redacted-test-key" not in json.dumps(preflight)


def test_closed_loop_cli_writes_no_spend_backend_readiness_for_cosmos3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=13)
    route = tmp_path / "route.json"
    route.write_text(json.dumps({"route_points": [[0.0, 0.0, 0.79], [1.0, 0.0, 0.79]]}), encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND", "python cosmos3_adapter.py")

    exit_code = L.main(
        [
            "--start-frame",
            str(start),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "cosmos3_wam",
            "--use-provider-command",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    readiness_path = tmp_path / "closed_loop" / "closed_loop_wam_backend_readiness.json"
    plan_path = tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert readiness["status"] == "blocked"
    assert readiness["selected_wam_backend"] == "cosmos3_wam"
    assert plan["selected_wam_backend"] == "cosmos3_wam"
    assert plan["wam_backend_readiness_path"] == str(readiness_path)
    assert "blocked_cosmos3_wam_not_wired_into_isaac_closed_loop_runner" in plan["blockers"]
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_closed_loop_cli_blocks_paid_multi_step_provider_without_projected_skeleton(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    start = _write_frame(tmp_path / "start.png", seed=17)
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps({"route_points": [[0.0, 0.0, 0.79], [1.0, 0.0, 0.79]]}),
        encoding="utf-8",
    )

    exit_code = L.main(
        [
            "--start-frame",
            str(start),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--allow-paid-provider-launch",
            "--steps",
            "2",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    readiness_path = tmp_path / "closed_loop" / "closed_loop_wam_backend_readiness.json"
    plan_path = tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    blocker = "closed_loop_projected_skeleton_trace_missing_for_paid_multi_step_provider_wam"
    assert readiness["status"] == "blocked"
    assert readiness["oscar_provider"] == "vast"
    assert readiness["paid_provider_preflight"]["provider"] == "vast"
    assert plan["oscar_provider"] == "vast"
    assert plan["num_frames_per_step"] == DEFAULT_NUM_FRAMES
    assert plan["oscar_runtime_settings"]["num_frames"] == DEFAULT_NUM_FRAMES
    assert readiness["seed_conditioning_preflight"]["required"] is True
    assert blocker in readiness["blockers"]
    assert blocker in plan["blockers"]
    assert plan["seed_conditioning_preflight"]["projected_skeleton_trace_present"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_closed_loop_cli_dry_run_writes_provider_input_contract_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cv2")
    _allow_vast_paid_provider(monkeypatch, tmp_path)
    render_dir = tmp_path / "render"
    source_render = render_dir / "frames" / "robot_pov_0000.png"
    _write_frame(source_render, seed=18)
    seed = tmp_path / "selected_seed.jpg"
    _write_frame(seed, seed=19)
    (render_dir / "manipulation_pov_geometry.json").write_text(
        json.dumps(
            {
                "schema_version": "manipulation_pov_geometry_index.v1",
                "frames": [
                    {
                        "status": "PASS",
                        "camera": "robot_pov",
                        "seed_frame_quality": {"image_size_px": [64, 48]},
                        "target_projection": {"available": True, "u_px": 50, "v_px": 24},
                        "projected_landmarks": [
                            {
                                "landmark_id": "left_shoulder",
                                "link_role": "shoulder",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 18,
                                    "v_px": 34,
                                    "depth_m": 0.35,
                                },
                            },
                            {
                                "landmark_id": "left_elbow",
                                "link_role": "elbow",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 22,
                                    "v_px": 32,
                                    "depth_m": 0.34,
                                },
                            },
                            {
                                "landmark_id": "left_wrist",
                                "link_role": "wrist",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 26,
                                    "v_px": 30,
                                    "depth_m": 0.32,
                                },
                            },
                            {
                                "landmark_id": "left_hand",
                                "link_role": "hand",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 30,
                                    "v_px": 28,
                                    "depth_m": 0.3,
                                },
                            },
                            {
                                "landmark_id": "right_shoulder",
                                "link_role": "shoulder",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 46,
                                    "v_px": 34,
                                    "depth_m": 0.35,
                                },
                            },
                            {
                                "landmark_id": "right_elbow",
                                "link_role": "elbow",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 42,
                                    "v_px": 32,
                                    "depth_m": 0.34,
                                },
                            },
                            {
                                "landmark_id": "right_wrist",
                                "link_role": "wrist",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 38,
                                    "v_px": 30,
                                    "depth_m": 0.32,
                                },
                            },
                            {
                                "landmark_id": "right_hand",
                                "link_role": "hand",
                                "image_projection": {
                                    "available": True,
                                    "u_px": 34,
                                    "v_px": 28,
                                    "depth_m": 0.3,
                                },
                            }
                        ],
                        "segments": [
                            {"from": "left_shoulder", "to": "left_elbow"},
                            {"from": "left_elbow", "to": "left_wrist"},
                            {"from": "left_wrist", "to": "left_hand"},
                            {"from": "right_shoulder", "to": "right_elbow"},
                            {"from": "right_elbow", "to": "right_wrist"},
                            {"from": "right_wrist", "to": "right_hand"},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source_trace = render_dir / "trace.jsonl"
    source_trace.write_text("{}\n", encoding="utf-8")
    route = tmp_path / "route.json"
    route.write_text(
        json.dumps(
            {
                "route_points": [[0.0, 0.0, 0.79], [1.0, 0.0, 0.79]],
                "source_trace": str(source_trace),
            }
        ),
        encoding="utf-8",
    )

    exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "2",
            "--oscar-guidance",
            "4.25",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    plan = json.loads(
        (tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json").read_text(
            encoding="utf-8"
        )
    )
    preflight = plan["provider_input_contract_preflight"]
    assert preflight["status"] == "ready"
    assert preflight["contract_status"] == "warning_high_risk"
    assert preflight["autoregressive_risk_level"] == "high"
    assert "projected_skeleton_not_scene_faithful_policy_action_high_risk" in preflight[
        "high_risk_flags"
    ]
    assert "projected_skeleton_missing_scene_faithful_policy_action_bridge" in preflight[
        "ranking_risk_flags"
    ]
    assert preflight["policy_ranking_claim_safe"] is False
    assert plan["short_visual_sanity_launch_plan"]["status"] == "not_required"
    assert Path(preflight["bundle_manifest_path"]).is_file()
    assert json.loads(capsys.readouterr().out)["status"] == "prepared"


def test_closed_loop_paid_long_run_requires_short_visual_sanity_after_input_risk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cv2")
    _allow_vast_paid_provider(monkeypatch, tmp_path)
    seed, route = _write_seed_geometry_route(tmp_path)

    exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "4",
            "--oscar-guidance",
            "4.25",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    plan = json.loads(
        (tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json").read_text(
            encoding="utf-8"
        )
    )
    gate = plan["short_rollout_sanity_gate"]
    assert gate["status"] == "blocked"
    assert gate["required"] is True
    assert gate["risk_recommends_short_sanity"] is True
    assert "closed_loop_paid_long_wam_requires_passed_short_rollout_sanity" in gate[
        "blockers"
    ]
    assert "short_visual_sanity_manifest_env_missing" in gate["blockers"]
    assert "closed_loop_paid_long_wam_requires_passed_short_rollout_sanity" in plan[
        "blockers"
    ]
    launch_plan = plan["short_visual_sanity_launch_plan"]
    assert launch_plan["status"] == "ready"
    assert launch_plan["required"] is True
    assert launch_plan["provider"] == "vast"
    assert launch_plan["provider_resolution"] == "explicit_provider"
    assert launch_plan["blockers"] == []
    assert launch_plan["command_materialized"] is True
    assert launch_plan["provider_launch_allowed_now"] is True
    assert launch_plan["provider_launch_blockers"] == []
    policy_observation_path = Path(launch_plan["policy_observation_path"])
    assert policy_observation_path.is_file()
    policy_observation = json.loads(policy_observation_path.read_text(encoding="utf-8"))
    assert policy_observation["schema_version"] == "blueprint_policy_observation.v1"
    assert policy_observation["task_prompt"] == plan["task_prompt"]
    assert policy_observation["unitree_g1_sonic_state_source"] == (
        "neutral_unitree_g1_sonic_contract_state"
    )
    assert policy_observation["unitree_g1_sonic_state_metadata"]["complete"] is True
    assert policy_observation["unitree_g1_sonic_state_metadata"][
        "scene_or_task_specific_coordinates_hardcoded"
    ] is False
    state = policy_observation["unitree_g1_sonic_state"]
    assert {key: len(value) for key, value in state.items()} == {
        "left_leg": 6,
        "right_leg": 6,
        "waist": 3,
        "left_arm": 7,
        "right_arm": 7,
        "left_hand": 7,
        "right_hand": 7,
        "projected_gravity": 3,
    }
    assert state["projected_gravity"] == [0.0, 0.0, -1.0]
    assert policy_observation["visual_observation"]["camera_frame_path"] == str(
        seed.resolve()
    )
    assert "blueprint_pipeline.persistent_wam_short_visual_sanity" in launch_plan[
        "command_argv"
    ]
    assert launch_plan["command_argv"][
        launch_plan["command_argv"].index("--transition-count") + 1
    ] == "2"
    assert launch_plan["expected_manifest_path"].endswith(
        "persistent_wam_short_visual_sanity_manifest.json"
    )
    assert launch_plan["unlock_env"][
        L.PERSISTENT_WAM_SHORT_VISUAL_SANITY_MANIFEST_ENV
    ] == launch_plan["expected_manifest_path"]
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_closed_loop_short_sanity_manifest_must_match_policy_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cv2")
    _allow_vast_paid_provider(monkeypatch, tmp_path)
    seed, route = _write_seed_geometry_route(tmp_path)
    stale_observation = tmp_path / "stale_policy_observation.json"
    stale_observation.write_text(
        json.dumps(
            {
                "schema_version": "blueprint_policy_observation.v1",
                "visual_observation": {"camera_frame_path": str(seed.resolve())},
            }
        ),
        encoding="utf-8",
    )
    stale_manifest = _write_passed_short_visual_sanity_manifest(
        tmp_path / "stale_short_sanity", stale_observation
    )

    exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "4",
            "--oscar-guidance",
            "4.25",
            "--short-visual-sanity-manifest",
            str(stale_manifest),
            "--dry-run",
        ]
    )

    assert exit_code == 2
    plan = json.loads(
        (tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json").read_text(
            encoding="utf-8"
        )
    )
    gate = plan["short_rollout_sanity_gate"]
    assert gate["status"] == "blocked"
    assert "short_visual_sanity_policy_observation_mismatch" in gate["blockers"]
    assert gate["expected_policy_observation_path"] == plan[
        "short_visual_sanity_launch_plan"
    ]["policy_observation_path"]
    assert "short_visual_sanity_policy_observation_mismatch" in gate[
        "short_visual_sanity_validation"
    ]["blockers"]
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_closed_loop_matching_short_sanity_manifest_unlocks_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cv2")
    _allow_vast_paid_provider(monkeypatch, tmp_path)
    seed, route = _write_seed_geometry_route(tmp_path)
    output_dir = tmp_path / "closed_loop"

    first_exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(output_dir),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "4",
            "--oscar-guidance",
            "4.25",
            "--dry-run",
        ]
    )
    assert first_exit_code == 2
    first_plan = json.loads(
        (output_dir / "oscar_isaac_closed_loop_plan.json").read_text(encoding="utf-8")
    )
    policy_observation_path = Path(
        first_plan["short_visual_sanity_launch_plan"]["policy_observation_path"]
    )
    matching_manifest = _write_passed_short_visual_sanity_manifest(
        tmp_path / "matching_short_sanity", policy_observation_path
    )

    second_exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(output_dir),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "4",
            "--oscar-guidance",
            "4.25",
            "--short-visual-sanity-manifest",
            str(matching_manifest),
            "--dry-run",
        ]
    )

    assert second_exit_code == 0
    plan = json.loads(
        (output_dir / "oscar_isaac_closed_loop_plan.json").read_text(encoding="utf-8")
    )
    gate = plan["short_rollout_sanity_gate"]
    assert plan["status"] == "prepared"
    assert gate["status"] == "ready"
    assert gate["short_visual_sanity_manifest_path"] == str(matching_manifest)
    assert gate["expected_policy_observation_path"] == str(policy_observation_path)
    assert gate["short_visual_sanity_validation"]["status"] == "passed_short_visual_sanity"
    assert "closed_loop_paid_long_wam_requires_passed_short_rollout_sanity" not in plan[
        "blockers"
    ]
    captured = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert captured[-1]["status"] == "prepared"


def test_closed_loop_short_sanity_launch_plan_blocks_vast_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cv2")
    seed, route = _write_seed_geometry_route(tmp_path)
    monkeypatch.delenv("BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_VAST_API_CALLS", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", raising=False)
    monkeypatch.setenv("VAST_API_KEY_FILE", str(tmp_path / "missing_vast_api_key"))
    monkeypatch.setenv("VAST_SESSION_BUDGET_LEDGER_FILE", str(tmp_path / "budget.json"))

    exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "4",
            "--oscar-guidance",
            "4.25",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    plan = json.loads(
        (tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json").read_text(
            encoding="utf-8"
        )
    )
    launch_plan = plan["short_visual_sanity_launch_plan"]
    assert launch_plan["status"] == "blocked_provider_authorization"
    assert launch_plan["command_materialized"] is True
    assert launch_plan["provider_launch_allowed_now"] is False
    assert launch_plan["provider"] == "vast"
    assert Path(launch_plan["policy_observation_path"]).is_file()
    assert "missing_env_BLUEPRINT_ALLOW_VAST_API_CALLS" in launch_plan[
        "provider_launch_blockers"
    ]
    assert "missing_env_BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH" in launch_plan[
        "provider_launch_blockers"
    ]
    assert "missing_file_based_secret_VAST_API_KEY_FILE" in launch_plan["blockers"]
    assert launch_plan["paid_provider_preflight"]["status"] == "blocked"
    assert launch_plan["claim_boundary"]["plan_is_no_spend"] is True
    assert len(plan["blockers"]) == len(set(plan["blockers"]))
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_closed_loop_short_sanity_launch_plan_allows_fresh_vast_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("cv2")
    seed, route = _write_seed_geometry_route(tmp_path)
    key_file = tmp_path / "vast_api_key"
    key_file.write_text("redacted-test-key\n", encoding="utf-8")
    budget = tmp_path / "fresh_budget.json"
    budget.write_text(
        json.dumps({"schema_version": "vast_session_cost_summary.v4", "attempts": []}),
        encoding="utf-8",
    )
    monkeypatch.setenv("BLUEPRINT_ALLOW_PAID_VAST_WAM_PROVIDER_LAUNCH", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "true")
    monkeypatch.setenv("VAST_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("VAST_SESSION_BUDGET_LEDGER_FILE", str(budget))
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_MAX_HOURLY_RATE", "0.25")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_MAX_LIVE_MINUTES", "10")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_SESSION_MAX_LIVE_MINUTES", "30")
    monkeypatch.setenv("BLUEPRINT_VAST_WAM_HARD_CAP_USD", "0.50")

    exit_code = L.main(
        [
            "--start-frame",
            str(seed),
            "--route-file",
            str(route),
            "--output-dir",
            str(tmp_path / "closed_loop"),
            "--wam-backend",
            "oscar_wam",
            "--use-provider-command",
            "--oscar-provider",
            "vast",
            "--allow-paid-provider-launch",
            "--steps",
            "4",
            "--oscar-guidance",
            "4.25",
            "--dry-run",
        ]
    )

    assert exit_code == 2
    plan = json.loads(
        (tmp_path / "closed_loop" / "oscar_isaac_closed_loop_plan.json").read_text(
            encoding="utf-8"
        )
    )
    launch_plan = plan["short_visual_sanity_launch_plan"]
    assert launch_plan["status"] == "ready"
    assert launch_plan["command_materialized"] is True
    assert launch_plan["provider_launch_allowed_now"] is True
    assert launch_plan["provider_launch_blockers"] == []
    assert launch_plan["provider"] == "vast"
    assert launch_plan["paid_provider_preflight"]["status"] == "ready"
    assert launch_plan["paid_provider_preflight"]["budget_ledger_present"] is True
    assert launch_plan["paid_provider_preflight"]["attempt_count"] == 0
    assert Path(launch_plan["policy_observation_path"]).is_file()
    assert launch_plan["blockers"] == []
    assert "closed_loop_paid_long_wam_requires_passed_short_rollout_sanity" in plan[
        "blockers"
    ]
    assert "short_visual_sanity_manifest_env_missing" in plan["blockers"]
    assert "redacted-test-key" not in json.dumps(plan)
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_closed_loop_runpod_paid_preflight_ready_with_gates_and_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RunPod is a first-class paid closed-loop provider when its API/pod-launch
    gates are set, the key file exists, and the projected cost fits the cap."""
    key_file = tmp_path / "runpod_api_key"
    key_file.write_text("redacted-test-key\n", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH", "true")
    monkeypatch.setenv("RUNPOD_API_KEY_FILE", str(key_file))

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="oscar_wam",
        use_provider_command=True,
        oscar_provider="runpod",
        allow_paid_provider_launch=True,
    )

    preflight = readiness["paid_provider_preflight"]
    assert preflight["status"] == "ready"
    assert preflight["provider"] == "runpod"
    assert preflight["projected_max_incremental_cost_usd"] <= preflight["hard_cap_usd"]


def test_closed_loop_runpod_paid_preflight_blocks_without_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BLUEPRINT_ALLOW_RUNPOD_API_CALLS", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH", raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY_FILE", str(tmp_path / "missing_key"))

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="oscar_wam",
        use_provider_command=True,
        oscar_provider="runpod",
        allow_paid_provider_launch=True,
    )

    preflight = readiness["paid_provider_preflight"]
    assert preflight["status"] == "blocked"
    assert "missing_env_BLUEPRINT_ALLOW_RUNPOD_API_CALLS" in preflight["blockers"]
    assert "missing_env_BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH" in preflight["blockers"]
    assert "missing_file_based_secret_RUNPOD_API_KEY_FILE" in preflight["blockers"]


def test_closed_loop_runpod_paid_preflight_blocks_over_hard_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_file = tmp_path / "runpod_api_key"
    key_file.write_text("redacted-test-key\n", encoding="utf-8")
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH", "true")
    monkeypatch.setenv("RUNPOD_API_KEY_FILE", str(key_file))
    monkeypatch.setenv("BLUEPRINT_RUNPOD_WAM_MAX_HOURLY_RATE", "10.0")
    monkeypatch.setenv("BLUEPRINT_RUNPOD_WAM_MAX_LIVE_MINUTES", "120")
    monkeypatch.setenv("BLUEPRINT_RUNPOD_WAM_HARD_CAP_USD", "3.0")

    readiness = L.build_closed_loop_wam_backend_readiness(
        selected_backend="oscar_wam",
        use_provider_command=True,
        oscar_provider="runpod",
        allow_paid_provider_launch=True,
    )

    preflight = readiness["paid_provider_preflight"]
    assert preflight["status"] == "blocked"
    assert any("hard_cap" in b for b in preflight["blockers"])


def test_cli_default_num_frames_is_standard_oscar_clip_length() -> None:
    """A 'default' generation must be the standard 81-frame (5.4s) OSCAR clip.
    The old default of 8 produced ~5-frame unusable rollouts that looked like
    model failure (2026-07-02 sink_faucet incident)."""
    import argparse

    parser_actions = {}
    real_parse = argparse.ArgumentParser.parse_args
    try:
        argparse.ArgumentParser.parse_args = lambda self, *a, **k: parser_actions.update(
            {a.dest: a.default for a in self._actions}) or (_ for _ in ()).throw(SystemExit(0))
        try:
            L.main(["--start-frame", "x", "--route-file", "y", "--output-dir", "z"])
        except SystemExit:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    assert parser_actions.get("num_frames") == 81


def _write_clip(path, frames):
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (64, 48)
    )
    for frame in frames:
        writer.write(np.ascontiguousarray(frame))
    writer.release()


def test_generated_clip_coherence_measures_drift(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    rng = np.random.default_rng(7)
    seed = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    # 3 seed-anchored frames (tiny jitter), then pure noise = drift.
    coherent = [
        np.clip(seed.astype(np.int16) + rng.integers(-6, 6, seed.shape), 0, 255).astype(
            np.uint8
        )
        for _ in range(3)
    ]
    noise = [
        rng.integers(0, 255, size=seed.shape, dtype=np.uint8) for _ in range(5)
    ]
    clip = tmp_path / "clip.mp4"
    _write_clip(clip, [seed, *coherent, *noise])

    result = L.generated_clip_coherence(clip)
    assert result["status"] == "measured"
    assert result["frame_count"] == 9
    # The 3 jittered frames stay anchored; the noise tail does not.
    assert 2 <= result["coherent_horizon_frames"] <= 5
    assert result["min_correlation"] < 0.5
    assert result["claim_boundary"]

    missing = L.generated_clip_coherence(tmp_path / "nope.mp4")
    assert missing["status"] == "not_measured"


def test_closed_loop_blocks_on_incoherent_generated_clip(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    rng = np.random.default_rng(11)
    seed_frame = _write_frame(tmp_path / "seed.png", 3)
    noise_clip = tmp_path / "noise.mp4"
    seed_img = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    _write_clip(
        noise_clip,
        [seed_img]
        + [rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8) for _ in range(6)],
    )

    def wam_with_incoherent_video(frame, action, step, history):
        generated = tmp_path / f"gen_{step}.png"
        _write_frame(generated, 40 + step)
        return {
            "status": "completed",
            "generated_frame_path": str(generated),
            "generated_video_path": str(noise_clip),
        }

    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "out",
        start_frame_path=seed_frame,
        route_points=[[0.0, 0.0, 0.79], [1.0, 0.0, 0.79]],
        wam_generate_next=wam_with_incoherent_video,
        steps=2,
        min_coherent_horizon_frames=2,
    )
    assert manifest["status"] == "blocked"
    assert any(
        blocker.startswith("blocked_generated_clip_coherence_below_floor_at_step_")
        for blocker in manifest["blockers"]
    )
    coherence = manifest["generated_clip_coherence"]
    assert coherence["min_coherent_horizon_frames_required"] == 2
    assert coherence["per_step"][0]["status"] == "measured"

    # Gate disabled (0): the same clips only get recorded, never block.
    relaxed = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "out_relaxed",
        start_frame_path=seed_frame,
        route_points=[[0.0, 0.0, 0.79], [1.0, 0.0, 0.79]],
        wam_generate_next=wam_with_incoherent_video,
        steps=1,
        min_coherent_horizon_frames=0,
    )
    assert not any(
        blocker.startswith("blocked_generated_clip_coherence")
        for blocker in relaxed.get("blockers", [])
    )
    assert relaxed["generated_clip_coherence"]["per_step"]


def test_cli_blocks_sub_native_oscar_resolution_without_override(tmp_path):
    seed = _write_frame(tmp_path / "seed.png", 5)
    route = tmp_path / "route.json"
    route.write_text(json.dumps({"route_points": [[0, 0, 0.79], [1, 0, 0.79]]}))

    def run_cli(extra):
        out = tmp_path / f"out_{len(extra)}"
        L.main(
            [
                "--start-frame", str(seed),
                "--route-file", str(route),
                "--steps", "1",
                "--output-dir", str(out),
                "--dry-run",
                "--oscar-height", "240",
                "--oscar-width", "320",
                *extra,
            ]
        )
        return json.loads(
            (out / "closed_loop_wam_backend_readiness.json").read_text()
        )

    blocked = run_cli([])
    contract = blocked["oscar_generation_resolution_contract"]
    assert contract["native_match"] is False
    assert contract["requested_height"] == 240
    assert (
        "blocked_non_native_oscar_resolution_requires_explicit_override"
        in blocked["blockers"]
    )
    assert blocked["status"] == "blocked"

    overridden = run_cli(["--allow-non-native-oscar-resolution"])
    assert (
        "blocked_non_native_oscar_resolution_requires_explicit_override"
        not in overridden["blockers"]
    )
    assert overridden["oscar_generation_resolution_contract"]["override_used"] is True


def _route_walking_wam(tmp_path):
    def wam(frame, action, step, history):
        generated = tmp_path / f"dyn_gen_{step}.png"
        _write_frame(generated, 60 + step)
        return {"status": "completed", "generated_frame_path": str(generated)}

    return wam


def test_episode_ends_when_task_target_reached(tmp_path):
    # Short route: the deterministic walk reaches the target well before the
    # steps cap, so the episode should terminate early with the task reason.
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "dyn_out",
        start_frame_path=_write_frame(tmp_path / "dyn_seed.png", 9),
        route_points=[[0.0, 0.0, 0.79], [0.1, 0.0, 0.79]],
        wam_generate_next=_route_walking_wam(tmp_path),
        steps=12,
        stop_on_task_completion=True,
    )
    termination = manifest["episode_termination"]
    assert termination["stop_on_task_completion"] is True
    assert termination["task_completed_early"] is True
    assert termination["reason"].startswith("task_target_reached_at_step_")
    assert termination["steps_executed"] < 12
    assert termination["steps_cap"] == 12
    assert manifest["status"] == "completed"
    assert "route-geometry" in termination["claim_boundary"]


def test_episode_runs_full_cap_without_stop_flag(tmp_path):
    manifest = L.run_oscar_isaac_closed_loop(
        output_dir=tmp_path / "cap_out",
        start_frame_path=_write_frame(tmp_path / "cap_seed.png", 9),
        route_points=[[0.0, 0.0, 0.79], [0.1, 0.0, 0.79]],
        wam_generate_next=_route_walking_wam(tmp_path),
        steps=3,
    )
    termination = manifest["episode_termination"]
    assert termination["task_completed_early"] is False
    assert termination["reason"] == "steps_cap_reached"
    assert termination["steps_executed"] == 3


def test_sealed_plan_and_cli_carry_stop_on_task_completion(tmp_path):
    from blueprint_pipeline import groot_oscar_closed_loop_image as gocl

    plan = gocl.build_sealed_launch_plan(
        start_frame="/workspace/seed.png",
        route_file="/workspace/route.json",
        steps=12,
        task_prompt="open the fridge",
        output_dir="/workspace/t4_out",
        env={"BLUEPRINT_GROOT_OSCAR_SEALED_IMAGE": "true"},
    )
    if plan["closed_loop_command"]:
        assert "--stop-on-task-completion" in plan["closed_loop_command"]
