"""Package synthetic 2D image seeds for Unitree G1 WAM loop experiments.

This module deliberately creates an image-only policy observation. It does not
promote a generated image to capture, geometry, collision, or safety evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .common import ensure_dir, utc_now_iso, write_json


SCHEMA_VERSION = "synthetic_2d_wam_seed_job.v1"
GENERATION_MANIFEST_SCHEMA_VERSION = "synthetic_2d_seed_image_generation_manifest.v1"
VISUAL_QA_SCHEMA_VERSION = "synthetic_2d_seed_image_visual_qa.v1"
POLICY_OBSERVATION_SCHEMA_VERSION = "initial_policy_observation.v1"
WAM_ROLLOUT_INPUT_SCHEMA_VERSION = "wam_rollout_input_manifest.v1"
CLAIM_BOUNDARY_SCHEMA_VERSION = "synthetic_2d_wam_seed_claim_boundary.v1"

SOURCE_KIND = "synthetic_gpt_image_2_seed"
DEFAULT_SOURCE_TOOL = "codex_oauth_image_generation"
DEFAULT_SOURCE_MODEL = "gpt-image-2"
DEFAULT_TASK_ID = "turn_on_sink_handle"
DEFAULT_TARGET_OBJECT_ID = "kitchen_sink_faucet_handle"
DEFAULT_CAMERA_ID = "synthetic_head_pov"
DEFAULT_ROBOT_PROFILE_ID = "unitree_g1_sonic"
DEFAULT_SELECTED_CANDIDATE_ID = "unitree_groot_n17_sonic_policy"
MIN_REVIEW_WIDTH = 960
MIN_REVIEW_HEIGHT = 540


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _truth_boundary() -> dict[str, Any]:
    return {
        "source_kind": SOURCE_KIND,
        "capture_truth": False,
        "geometry_truth": False,
        "collision_truth": False,
        "visual_seed_for_wam_experiment": True,
        "real_capture_evidence": False,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "safety_validation_proven": False,
        "real_world_manipulation_success_proven": False,
        "generated_image_may_improve_reviewability_only": True,
        "raw_capture_evidence_authority": False,
    }


def _zero_unitree_g1_sonic_state() -> dict[str, Any]:
    return {
        "left_leg": [0.0] * 6,
        "right_leg": [0.0] * 6,
        "waist": [0.0] * 3,
        "left_arm": [0.0] * 7,
        "right_arm": [0.0] * 7,
        "left_hand": [0.0] * 7,
        "right_hand": [0.0] * 7,
        "projected_gravity": [0.0, 0.0, -1.0],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_info(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    aspect_ratio = round(width / height, 6) if height else None
    return {
        "width": width,
        "height": height,
        "mode": mode,
        "aspect_ratio": aspect_ratio,
        "review_resolution_floor_passed": width >= MIN_REVIEW_WIDTH
        and height >= MIN_REVIEW_HEIGHT,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _candidate_id(index: int) -> str:
    return f"candidate_{index:04d}"


def _copy_candidate(
    *,
    source: Path,
    destination_dir: Path,
    index: int,
) -> Path:
    suffix = source.suffix.lower() if source.suffix else ".png"
    destination = destination_dir / f"{_candidate_id(index)}{suffix}"
    shutil.copy2(source, destination)
    return destination


def _prompt_at(prompts: Sequence[str] | None, index: int) -> str | None:
    if not prompts or index >= len(prompts):
        return None
    prompt = _string(prompts[index])
    return prompt or None


def build_synthetic_2d_wam_seed_job(
    *,
    job_dir: str | Path,
    candidate_images: Sequence[str | Path],
    selected_image: str | Path,
    candidate_prompts: Sequence[str] | None = None,
    selected_prompt: str | None = None,
    selection_rationale: str,
    selected_visual_qa_passed: bool,
    selected_visual_qa_notes: str,
    task_id: str = DEFAULT_TASK_ID,
    target_object_id: str = DEFAULT_TARGET_OBJECT_ID,
    camera_id: str = DEFAULT_CAMERA_ID,
    robot_profile_id: str = DEFAULT_ROBOT_PROFILE_ID,
    source_tool: str = DEFAULT_SOURCE_TOOL,
    source_model: str = DEFAULT_SOURCE_MODEL,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Create a synthetic image-only policy observation package.

    The returned job is ready to feed into the existing persistent-session
    runner through ``policy_observation.json`` or ``policy_input.json``.
    """

    if not candidate_images:
        raise ValueError("candidate_images_required")
    generated = generated_at or utc_now_iso()
    job = Path(job_dir).expanduser().resolve()
    seed_dir = job / "seed_images"
    ensure_dir(seed_dir)
    selected_source = Path(selected_image).expanduser().resolve()
    if not selected_source.is_file():
        raise FileNotFoundError(f"selected_image_missing:{selected_source}")

    candidates: list[dict[str, Any]] = []
    selected_candidate_id: str | None = None
    selected_copied_candidate: Path | None = None
    for index, raw_candidate in enumerate(candidate_images):
        source = Path(raw_candidate).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"candidate_image_missing:{source}")
        copied = _copy_candidate(source=source, destination_dir=seed_dir, index=index)
        candidate_id = _candidate_id(index)
        is_selected = source == selected_source
        if is_selected:
            selected_candidate_id = candidate_id
            selected_copied_candidate = copied
        candidates.append(
            {
                "candidate_id": candidate_id,
                "source_path": str(source),
                "stored_path": str(copied),
                "prompt": _prompt_at(candidate_prompts, index),
                "source_tool": source_tool,
                "source_model": source_model,
                "source_kind": SOURCE_KIND,
                "selected": is_selected,
                "image": _image_info(copied),
                "capture_truth": False,
                "geometry_truth": False,
                "collision_truth": False,
            }
        )

    if selected_copied_candidate is None:
        selected_candidate_id = "selected_external_image"
        selected_copied_candidate = seed_dir / f"{selected_candidate_id}{selected_source.suffix or '.png'}"
        shutil.copy2(selected_source, selected_copied_candidate)

    selected_frame = job / "selected_initial_policy_frame.png"
    shutil.copy2(selected_copied_candidate, selected_frame)
    selected_info = _image_info(selected_frame)
    effective_selected_prompt = _string(selected_prompt)
    if not effective_selected_prompt and selected_candidate_id:
        effective_selected_prompt = next(
            (
                _string(candidate.get("prompt"))
                for candidate in candidates
                if candidate.get("candidate_id") == selected_candidate_id
            ),
            "",
        )
    boundary = _truth_boundary()
    selected_visual_qa_passed = bool(
        selected_visual_qa_passed and selected_info["review_resolution_floor_passed"]
    )
    visual_qa = {
        "schema_version": VISUAL_QA_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "passed" if selected_visual_qa_passed else "failed",
        "selected_candidate_id": selected_candidate_id,
        "selected_image_path": str(selected_frame),
        "manual_visual_qa_passed": bool(selected_visual_qa_passed),
        "manual_visual_qa_notes": selected_visual_qa_notes,
        "selection_rationale": selection_rationale,
        "requirements": {
            "first_person_unitree_g1_head_pov": True,
            "both_forearms_or_hands_visible": True,
            "sink_or_faucet_task_visible": task_id == DEFAULT_TASK_ID,
            "target_handle_or_knob_visible_and_reachable": True,
            "no_text_watermark_ui_or_labels_observed": True,
            "no_review_blocking_blur_observed": True,
            "minimum_review_resolution": {
                "width": MIN_REVIEW_WIDTH,
                "height": MIN_REVIEW_HEIGHT,
                "passed": selected_info["review_resolution_floor_passed"],
            },
        },
        "selected_image": selected_info,
        "claim_boundary": boundary,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    generation_manifest = {
        "schema_version": GENERATION_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "completed",
        "source_tool": source_tool,
        "source_model": source_model,
        "source_kind": SOURCE_KIND,
        "candidate_count": len(candidates),
        "selected_candidate_id": selected_candidate_id,
        "selected_prompt": effective_selected_prompt,
        "selected_initial_policy_frame_path": str(selected_frame),
        "selection_rationale": selection_rationale,
        "candidates": candidates,
        "claim_boundary": boundary,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    task_prompt = (
        "Use the synthetic Unitree G1 head-POV image to propose a short "
        "UNITREE_G1_SONIC-compatible action chunk for turning the visible "
        "kitchen sink faucet handle. Treat the image as a visual seed only."
    )
    visual_observation = {
        "available": True,
        "camera_id": camera_id,
        "camera_frame_path": str(selected_frame),
        "source_image_path": str(selected_frame),
        "source_kind": SOURCE_KIND,
        "source_tool": source_tool,
        "source_model": source_model,
        "first_person_policy_observation_candidate": True,
        "synthetic_camera_view": True,
        "simulated_camera_view": False,
        "physical_robot_sensor_proof": False,
        "capture_truth": False,
        "geometry_truth": False,
        "collision_truth": False,
        "width": selected_info["width"],
        "height": selected_info["height"],
        "sha256": selected_info["sha256"],
        "blockers": [],
        "claim_boundary": boundary,
    }
    policy_observation = {
        "schema_version": POLICY_OBSERVATION_SCHEMA_VERSION,
        "generated_at": generated,
        "source_kind": SOURCE_KIND,
        "task_id": task_id,
        "target_object_id": target_object_id,
        "robot_profile_id": robot_profile_id,
        "selected_candidate_id": DEFAULT_SELECTED_CANDIDATE_ID,
        "task_prompt": task_prompt,
        "camera_frame_path": str(selected_frame),
        "visual_observation": visual_observation,
        "unitree_g1_sonic_state": _zero_unitree_g1_sonic_state(),
        "unitree_g1_sonic_state_source": "synthetic_2d_seed_contract_probe_zero_state",
        "unitree_g1_sonic_state_metadata": {
            "state_is_contract_probe": True,
            "state_is_not_measured_from_robot": True,
            "geometry_truth": False,
        },
        "claim_boundary": boundary,
    }
    policy_observation_path = job / "policy_observation.json"
    policy_input_path = job / "policy_input.json"
    write_json(policy_observation_path, {"observation": policy_observation})
    write_json(policy_input_path, {"observation": policy_observation})

    wam_rollout_input = {
        "schema_version": WAM_ROLLOUT_INPUT_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "ready_for_image_only_wam_seed",
        "source_kind": SOURCE_KIND,
        "image_only_synthetic_seed_profile": True,
        "camera_id": camera_id,
        "robot_profile_id": robot_profile_id,
        "task_id": task_id,
        "target_object_id": target_object_id,
        "source_image_path": str(selected_frame),
        "selected_initial_policy_frame_path": str(selected_frame),
        "policy_observation_path": str(policy_observation_path),
        "policy_input_path": str(policy_input_path),
        "expected_loop_shape": [
            "initial_synthetic_image",
            "unitree_groot_n17_sonic_policy_call",
            "action_chunk_or_proxy_skeleton_conditioning",
            "oscar_wam_next_observation_generation",
            "generated_observation_feedback_to_next_policy_call",
        ],
        "blockers": [],
        "claim_boundary": boundary,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    claim_boundary = {
        "schema_version": CLAIM_BOUNDARY_SCHEMA_VERSION,
        "generated_at": generated,
        **boundary,
        "clean_2d_seed_compared_against_prior_3d_seed": True,
        "claim_if_successful": (
            "clean synthetic 2D seed improves reviewability for WAM visual rollout"
        ),
        "forbidden_claims": [
            "real_capture_evidence",
            "3d_state_truth",
            "geometry_truth",
            "collision_validity",
            "physical_robot_readiness",
            "safety_validation",
            "deployment_approval",
        ],
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }

    write_json(job / "seed_image_generation_manifest.json", generation_manifest)
    write_json(job / "seed_image_visual_qa.json", visual_qa)
    write_json(job / "wam_rollout_input_manifest.json", wam_rollout_input)
    write_json(job / "claim_boundary.json", claim_boundary)
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated,
        "status": "ready_for_policy_wam_loop" if selected_visual_qa_passed else "blocked_visual_qa",
        "job_dir": str(job),
        "seed_images_dir": str(seed_dir),
        "selected_candidate_id": selected_candidate_id,
        "selected_initial_policy_frame_path": str(selected_frame),
        "seed_image_generation_manifest_path": str(job / "seed_image_generation_manifest.json"),
        "seed_image_visual_qa_path": str(job / "seed_image_visual_qa.json"),
        "policy_observation_path": str(policy_observation_path),
        "policy_input_path": str(policy_input_path),
        "wam_rollout_input_manifest_path": str(job / "wam_rollout_input_manifest.json"),
        "claim_boundary_path": str(job / "claim_boundary.json"),
        "claim_boundary": boundary,
        "blockers": [] if selected_visual_qa_passed else ["selected_seed_failed_visual_qa"],
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(job / "synthetic_2d_wam_seed_job_manifest.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--candidate-image", action="append", required=True)
    parser.add_argument("--candidate-prompt", action="append")
    parser.add_argument("--selected-image", required=True)
    parser.add_argument("--selected-prompt")
    parser.add_argument("--selection-rationale", required=True)
    parser.add_argument("--selected-visual-qa-passed", action="store_true")
    parser.add_argument("--selected-visual-qa-notes", default="")
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--target-object-id", default=DEFAULT_TARGET_OBJECT_ID)
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--robot-profile-id", default=DEFAULT_ROBOT_PROFILE_ID)
    parser.add_argument("--source-tool", default=DEFAULT_SOURCE_TOOL)
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    args = parser.parse_args(argv)
    manifest = build_synthetic_2d_wam_seed_job(
        job_dir=args.job_dir,
        candidate_images=args.candidate_image,
        selected_image=args.selected_image,
        candidate_prompts=args.candidate_prompt,
        selected_prompt=args.selected_prompt,
        selection_rationale=args.selection_rationale,
        selected_visual_qa_passed=args.selected_visual_qa_passed,
        selected_visual_qa_notes=args.selected_visual_qa_notes,
        task_id=args.task_id,
        target_object_id=args.target_object_id,
        camera_id=args.camera_id,
        robot_profile_id=args.robot_profile_id,
        source_tool=args.source_tool,
        source_model=args.source_model,
    )
    print(f"[synthetic-2d-wam-seed] manifest={manifest['job_dir']}/synthetic_2d_wam_seed_job_manifest.json")
    print(f"[synthetic-2d-wam-seed] status={manifest['status']}")
    blockers = manifest.get("blockers") or []
    if blockers:
        print("[synthetic-2d-wam-seed] blockers=" + ",".join(str(item) for item in blockers))
    return 0 if manifest["status"] == "ready_for_policy_wam_loop" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
