from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from blueprint_pipeline.synthetic_2d_wam_seed import build_synthetic_2d_wam_seed_job


def _png(path: Path, *, size: tuple[int, int] = (1280, 720)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(210, 220, 230)).save(path)
    return path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_build_synthetic_2d_wam_seed_job_writes_policy_and_truth_boundary(
    tmp_path: Path,
) -> None:
    selected = _png(tmp_path / "generated" / "sink_selected.png")
    alternate = _png(tmp_path / "generated" / "stovetop_alternate.png")
    job_dir = tmp_path / "robot_eval_jobs" / "gpt_image2_unitree_g1_2d_wam_seed_test"

    result = build_synthetic_2d_wam_seed_job(
        job_dir=job_dir,
        candidate_images=[selected, alternate],
        selected_image=selected,
        candidate_prompts=["sink prompt", "stovetop prompt"],
        selected_prompt="sink prompt",
        selection_rationale="selected sink image has the clearest reachable faucet handle",
        selected_visual_qa_passed=True,
        selected_visual_qa_notes="both hands visible, target visible, no text or watermark",
        generated_at="2026-06-25T00:00:00+00:00",
    )

    assert result["status"] == "ready_for_policy_wam_loop"
    assert result["blockers"] == []
    assert (job_dir / "seed_images" / "candidate_0000.png").is_file()
    assert (job_dir / "seed_images" / "candidate_0001.png").is_file()
    assert (job_dir / "selected_initial_policy_frame.png").is_file()

    generation = _read(job_dir / "seed_image_generation_manifest.json")
    assert generation["source_model"] == "gpt-image-2"
    assert generation["source_kind"] == "synthetic_gpt_image_2_seed"
    assert generation["candidate_count"] == 2
    assert generation["selected_prompt"] == "sink prompt"
    assert generation["claim_boundary"]["capture_truth"] is False
    assert generation["claim_boundary"]["geometry_truth"] is False
    assert generation["claim_boundary"]["collision_truth"] is False

    visual_qa = _read(job_dir / "seed_image_visual_qa.json")
    assert visual_qa["status"] == "passed"
    assert visual_qa["requirements"]["minimum_review_resolution"]["passed"] is True

    policy_input = _read(job_dir / "policy_input.json")
    observation = policy_input["observation"]
    assert observation["source_kind"] == "synthetic_gpt_image_2_seed"
    assert observation["task_id"] == "turn_on_sink_handle"
    assert observation["robot_profile_id"] == "unitree_g1_sonic"
    assert observation["visual_observation"]["camera_id"] == "synthetic_head_pov"
    assert observation["visual_observation"]["capture_truth"] is False
    assert observation["visual_observation"]["synthetic_camera_view"] is True
    assert observation["unitree_g1_sonic_state_source"] == (
        "synthetic_2d_seed_contract_probe_zero_state"
    )

    rollout = _read(job_dir / "wam_rollout_input_manifest.json")
    assert rollout["schema_version"] == "wam_rollout_input_manifest.v1"
    assert rollout["status"] == "ready_for_image_only_wam_seed"
    assert rollout["camera_id"] == "synthetic_head_pov"
    assert rollout["source_image_path"].endswith("selected_initial_policy_frame.png")
    assert "oscar_wam_next_observation_generation" in rollout["expected_loop_shape"]

    boundary = _read(job_dir / "claim_boundary.json")
    assert boundary["visual_seed_for_wam_experiment"] is True
    assert "geometry_truth" in boundary["forbidden_claims"]
    assert boundary["physical_robot_readiness_proven"] is False


def test_build_synthetic_2d_wam_seed_blocks_low_resolution_selection(
    tmp_path: Path,
) -> None:
    selected = _png(tmp_path / "small.png", size=(320, 180))

    result = build_synthetic_2d_wam_seed_job(
        job_dir=tmp_path / "job",
        candidate_images=[selected],
        selected_image=selected,
        candidate_prompts=["small prompt"],
        selected_prompt="small prompt",
        selection_rationale="too small",
        selected_visual_qa_passed=True,
        selected_visual_qa_notes="manual pass cannot override resolution floor",
        generated_at="2026-06-25T00:00:00+00:00",
    )

    assert result["status"] == "blocked_visual_qa"
    assert result["blockers"] == ["selected_seed_failed_visual_qa"]
    visual_qa = _read(tmp_path / "job" / "seed_image_visual_qa.json")
    assert visual_qa["status"] == "failed"
    assert visual_qa["requirements"]["minimum_review_resolution"]["passed"] is False
