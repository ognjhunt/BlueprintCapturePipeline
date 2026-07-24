"""Run a sim-only WAM-derived observation harness end-to-end proof."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

from .common import ensure_dir, utc_now_iso, write_json
from .wam_derived_observation_harness import (
    run_wam_derived_observation_harness_step,
    summarize_wam_derived_observation_artifacts,
)
from .wam_real_provider_validation_probe import (
    AUTO_DA3_ENV,
    AUTO_DEPTH_ENV,
    AUTO_POSE_ENV,
    DA3_MODEL_ENV,
    DEFAULT_DA3_MODEL_ID,
    DEFAULT_DEPTH_MODEL_ID,
    DEFAULT_POSE_MODEL_PATH,
    DEPTH_MODEL_ENV,
    DEPTH_PROVIDER_KIND_ENV,
    POSE_MODEL_ENV,
    SAM3_CONFIDENCE_ENV,
    SAM3_WEIGHTS_ENV,
)


SIM_E2E_MANIFEST_SCHEMA_VERSION = "wam_sim_provider_e2e_manifest.v1"
SIM_E2E_TRACE_SCHEMA_VERSION = "wam_sim_provider_e2e_trace.v1"
DEFAULT_JOB_PREFIX = "wam_sim_provider_e2e"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _default_output_dir() -> Path:
    stamp = utc_now_iso().replace(":", "").replace("-", "").split(".")[0]
    return Path("robot_eval_jobs") / f"{DEFAULT_JOB_PREFIX}_{stamp}Z"


def _discover_generated_frame() -> Path | None:
    patterns = (
        "robot_eval_jobs/gpt_image2*/**/generated_rollout_frame_review/frames/*.jpg",
        "robot_eval_jobs/gpt_image2*/**/*.jpg",
        "robot_eval_jobs/**/generated_rollout_frame_review/frames/*.jpg",
        "robot_eval_jobs/**/generated_rollout_frame_review/frames/*.png",
        "robot_eval_jobs/**/robot_policy_wam_closed_loop/**/*frame*.jpg",
        "robot_eval_jobs/**/robot_policy_wam_closed_loop/**/*frame*.png",
    )
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(".").glob(pattern))
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    return sorted(existing, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _write_local_generated_start_frame(path: Path, target_prompt: str) -> Path:
    ensure_dir(path.parent)
    width, height = 768, 512
    image = Image.new("RGB", (width, height), (32, 36, 38))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 330, width, height), fill=(83, 88, 86))
    draw.rectangle((90, 130, 690, 348), fill=(172, 176, 170), outline=(210, 210, 200), width=3)
    draw.ellipse((280, 178, 486, 306), fill=(58, 84, 92), outline=(220, 235, 238), width=4)
    draw.rectangle((402, 110, 432, 190), fill=(185, 188, 180))
    draw.arc((374, 72, 488, 158), 180, 360, fill=(220, 225, 218), width=8)
    draw.line((575, 45, 430, 190), fill=(226, 203, 85), width=16)
    draw.line((430, 190, 320, 252), fill=(226, 203, 85), width=14)
    draw.ellipse((300, 232, 342, 274), fill=(224, 120, 68), outline=(255, 235, 210), width=3)
    draw.line((612, 92, 536, 110), fill=(62, 66, 70), width=8)
    draw.polygon([(530, 110), (548, 96), (550, 124)], fill=(62, 66, 70))
    label = f"sim-generated WAM seed: {target_prompt}"
    try:
        font = ImageFont.load_default()
        draw.text((22, 20), label, fill=(235, 235, 226), font=font)
    except Exception:
        draw.text((22, 20), label, fill=(235, 235, 226))
    image.save(path, quality=92)
    return path


def _select_start_frame(output_dir: Path, generated_frame_path: Path | None, target_prompt: str) -> tuple[Path, str]:
    if generated_frame_path and generated_frame_path.is_file():
        return generated_frame_path, "explicit_generated_frame_path"
    discovered = _discover_generated_frame()
    if discovered:
        source = "discovered_gpt_image2_generated_frame" if "gpt_image2" in str(discovered) else "discovered_generated_frame"
        return discovered, source
    return (
        _write_local_generated_start_frame(output_dir / "sim_generated_frames" / "step_0000_start.jpg", target_prompt),
        "local_synthetic_ai_style_generated_frame",
    )


def _write_next_sim_frame(
    *,
    source_frame: Path,
    output_path: Path,
    step_index: int,
    action: Mapping[str, Any],
    target_prompt: str,
) -> Path:
    ensure_dir(output_path.parent)
    try:
        image = Image.open(source_frame).convert("RGB")
    except Exception:
        image = Image.new("RGB", (768, 512), (36, 38, 42))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    progress = min(1.0, max(0.0, step_index / 6.0))
    x0 = int(width * (0.70 - 0.22 * progress))
    y0 = int(height * (0.17 + 0.18 * progress))
    x1 = int(width * (0.51 - 0.13 * progress))
    y1 = int(height * (0.36 + 0.12 * progress))
    draw.line((x0, y0, x1, y1), fill=(245, 198, 67), width=max(10, width // 70))
    draw.ellipse((x1 - 18, y1 - 18, x1 + 18, y1 + 18), fill=(238, 118, 68), outline=(255, 240, 220), width=3)
    draw.rectangle((12, height - 58, min(width - 12, 430), height - 16), fill=(24, 27, 29))
    text = f"sim WAM step {step_index}: {_string(action.get('action_type')) or 'policy_action'} -> {target_prompt}"
    draw.text((22, height - 46), text[:70], fill=(235, 235, 226))
    image.save(output_path, quality=92)
    return output_path


def _declared_schema(policy_schema: str) -> dict[str, Any]:
    if policy_schema == "rgb_only":
        return {
            "schema_version": "wam_sim_provider_e2e_policy_schema.v1",
            "rgb_only": True,
            "modalities": ["rgb"],
            "fields": ["camera_frame_path", "visual_observation"],
            "supports_depth": False,
            "supports_masks": False,
            "supports_state": False,
        }
    return {
        "schema_version": "wam_sim_provider_e2e_policy_schema.v1",
        "modalities": ["rgb", "depth", "mask", "pose", "state"],
        "fields": [
            "camera_frame_path",
            "visual_observation",
            "objects",
            "depth_estimates",
            "pose_estimates",
            "robot_state",
            "contact_likelihood",
            "uncertainty",
            "consistency_checks",
        ],
        "supports_depth": True,
        "supports_masks": True,
        "supports_state": True,
    }


def _policy_action(step_index: int, target_prompt: str) -> dict[str, Any]:
    return {
        "schema_version": "wam_sim_policy_action.v1",
        "action_type": "sim_policy_requery_action",
        "step_index": step_index,
        "target_prompt": target_prompt,
        "end_effector_delta_m": [round(0.015 * step_index, 4), 0.0, -0.005],
        "gripper_command": "close" if step_index >= 2 else "approach",
        "source": "deterministic_sim_policy_stub",
    }


def _base_observation(frame: Path, target_prompt: str) -> dict[str, Any]:
    return {
        "schema_version": "blueprint_policy_observation.v1",
        "camera_frame_path": str(frame),
        "camera_role": "robot_pov",
        "viewpoint_mode": "robot_head_mounted_egocentric",
        "policy_observation_eligible": True,
        "third_person_overview_included": False,
        "visual_observation": {
            "camera_frame_path": str(frame),
            "camera_role": "robot_pov",
            "viewpoint_mode": "robot_head_mounted_egocentric",
            "policy_observation_eligible": True,
            "third_person_overview_included": False,
            "wam_generated_observation": True,
            "sim_only_generated_observation": True,
            "physical_robot_sensor_proof": False,
        },
        "task_id": "sim_wam_provider_e2e",
        "task_prompt": f"interact with {target_prompt}",
        "target_object_id": target_prompt.replace(" ", "_"),
        "state": {
            "sim_only": True,
            "capture_truth": False,
            "physical_sensor_truth": False,
        },
    }


def _grounding(target_prompt: str) -> dict[str, Any]:
    return {
        "schema_version": "eval_ready_task_grounding.v1",
        "status": "sim_only_prompt_grounding",
        "task": {
            "task_id": "sim_wam_provider_e2e",
            "target_prompts_for_object_index_backends": [target_prompt],
        },
        "selected_task_target": {
            "object_id": target_prompt.replace(" ", "_"),
            "label": target_prompt,
            "source_prompt": target_prompt,
            "source": "sim_only_e2e_target_prompt",
        },
    }


def _env_patch_for_run(args: argparse.Namespace) -> dict[str, str | None]:
    patch: dict[str, str | None] = {}
    if args.sam3_weights:
        patch[SAM3_WEIGHTS_ENV] = str(args.sam3_weights)
    if args.sam3_confidence is not None:
        patch[SAM3_CONFIDENCE_ENV] = str(args.sam3_confidence)
    if args.provider_mode == "real":
        if args.depth_provider == "da3":
            patch[DEPTH_PROVIDER_KIND_ENV] = "da3"
            patch[AUTO_DA3_ENV] = "true"
            patch[DA3_MODEL_ENV] = args.da3_model_id
            patch[AUTO_DEPTH_ENV] = None
        elif args.depth_provider == "v2":
            patch[DEPTH_PROVIDER_KIND_ENV] = "transformers_depth_anything_v2"
            patch[AUTO_DEPTH_ENV] = "true"
            patch[DEPTH_MODEL_ENV] = args.depth_model_id
            patch[AUTO_DA3_ENV] = None
        patch[AUTO_POSE_ENV] = "true"
        patch[POSE_MODEL_ENV] = str(args.pose_model)
    return patch


def _apply_env_patch(patch: Mapping[str, str | None]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for key, value in patch.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    return previous


def _restore_env(previous: Mapping[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _provider_triplet_completed(step_records: Sequence[Mapping[str, Any]]) -> bool:
    if not step_records:
        return False
    for step in step_records:
        backend = _mapping(step.get("harness_backend"))
        statuses = [_mapping(row) for row in _sequence(backend.get("provider_statuses"))]
        providers = {row.get("provider"): row for row in statuses}
        for provider in ("sam3", "depth", "pose"):
            row = _mapping(providers.get(provider))
            if not row.get("ran") or _sequence(row.get("blockers")):
                return False
    return True


def run_sim_provider_e2e(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir or _default_output_dir()
    ensure_dir(output_dir)
    target_prompt = args.target_prompt or "robot arm"
    start_frame, start_source = _select_start_frame(output_dir, args.generated_frame, target_prompt)
    harness_dir = output_dir / "wam_derived_observation_harness"
    generated_dir = output_dir / "sim_generated_frames"
    trace_path = output_dir / "wam_sim_provider_e2e_trace.jsonl"
    declared_schema = _declared_schema(args.policy_schema)
    backend_kind = "fixture" if args.provider_mode == "fixture" else "real_provider_probe"
    backend_command = (
        None
        if args.provider_mode == "fixture"
        else [sys.executable, "-m", "blueprint_pipeline.wam_real_provider_validation_probe", "backend"]
    )
    previous = _apply_env_patch(_env_patch_for_run(args))
    step_records: list[dict[str, Any]] = []
    adapter_reports: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] | None = None
    current_frame = start_frame
    action_history: list[dict[str, Any]] = []
    try:
        for step_index in range(1, max(1, int(args.step_count)) + 1):
            action = _policy_action(step_index, target_prompt)
            generated_frame = _write_next_sim_frame(
                source_frame=current_frame,
                output_path=generated_dir / f"step_{step_index:04d}_wam_generated.jpg",
                step_index=step_index,
                action=action,
                target_prompt=target_prompt,
            )
            action_history.append(action)
            result = run_wam_derived_observation_harness_step(
                output_dir=harness_dir,
                step_index=step_index,
                source_generated_frame_path=generated_frame,
                source_wam_rollout_id=output_dir.name,
                transition_id=f"sim_provider_e2e_step_{step_index:04d}",
                source_policy_action=action,
                action_history=action_history,
                current_policy_observation=_base_observation(generated_frame, target_prompt),
                eval_ready_task_grounding=_grounding(target_prompt),
                previous_steps=step_records,
                previous_adapter_reports=adapter_reports,
                backend_kind=backend_kind,
                backend_command=backend_command,
                allow_external_backend=args.provider_mode == "real",
                backend_timeout_seconds=int(args.backend_timeout_seconds),
                policy_id=args.policy_id,
                declared_policy_observation_schema=declared_schema,
            )
            artifacts = result
            step = dict(result["step_record"])
            adapter = dict(result["policy_adapter_report"])
            contact_likelihood = _mapping(step.get("contact_likelihood"))
            uncertainty = _mapping(step.get("uncertainty"))
            scoring_allowed = _mapping(step.get("scoring_allowed"))
            step_records.append(step)
            adapter_reports.append(adapter)
            trace_rows.append(
                {
                    "schema_version": SIM_E2E_TRACE_SCHEMA_VERSION,
                    "step_index": step_index,
                    "policy_action": action,
                    "source_frame_path": str(current_frame),
                    "wam_generated_frame_path": str(generated_frame),
                    "generated_frame_path": str(generated_frame),
                    "source_generated_frame_path": step.get("source_generated_frame_path"),
                    "harness_status": step.get("status"),
                    "adapter_status": adapter.get("adapter_status"),
                    "safe_for_policy_requery": adapter.get("safe_for_policy_requery"),
                    "early_termination_recommended": uncertainty.get("early_termination_recommended"),
                    "objects": len(_sequence(step.get("objects"))),
                    "depth_estimates": len(_sequence(step.get("depth_estimates"))),
                    "pose_estimates": len(_sequence(step.get("pose_estimates"))),
                    "contact_likelihood": {
                        "value": contact_likelihood.get("value"),
                        "confidence": contact_likelihood.get("confidence"),
                        "physical_contact_proven": contact_likelihood.get("physical_contact_proven"),
                    },
                    "uncertainty": {
                        "overall_confidence": uncertainty.get("overall_confidence"),
                        "early_termination_recommended": uncertainty.get("early_termination_recommended"),
                        "reasons": uncertainty.get("reasons", []),
                    },
                    "scoring_allowed": {
                        "usable_for_diagnostics": scoring_allowed.get("usable_for_diagnostics"),
                        "usable_for_policy_requery": scoring_allowed.get("usable_for_policy_requery"),
                        "usable_for_success_scoring": scoring_allowed.get("usable_for_success_scoring"),
                    },
                    "claim_boundary": {
                        "sim_only": True,
                        "generated_pixels_not_capture_truth": True,
                        "physical_sensor_truth": False,
                    },
                }
            )
            current_frame = generated_frame
    finally:
        _restore_env(previous)

    trace_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in trace_rows) + ("\n" if trace_rows else ""),
        encoding="utf-8",
    )
    harness_summary = summarize_wam_derived_observation_artifacts(artifacts or {})
    all_adapters_completed = bool(adapter_reports) and all(
        row.get("adapter_status") == "completed" for row in adapter_reports
    )
    all_safe_for_requery = bool(adapter_reports) and all(
        bool(row.get("safe_for_policy_requery")) for row in adapter_reports
    )
    provider_triplet_completed = (
        _provider_triplet_completed(step_records) if args.provider_mode == "real" else False
    )
    harness_completed = harness_summary.get("status") == "completed"
    completed = bool(step_records) and harness_completed and all_adapters_completed
    if args.provider_mode == "real":
        completed = completed and provider_triplet_completed
    claim_boundary = {
        "sim_only": True,
        "generated_frames_are_not_capture_truth": True,
        "harness_outputs_are_derived_observations_not_real_sensors": True,
        "inferred_depth_is_not_sensor_depth": True,
        "sam3_masks_are_not_physical_truth": True,
        "contact_likelihood_is_not_physical_contact_proof": True,
        "perception_accuracy_against_truth_labels_proven": False,
        "non_ranking_operational_claim_proven": False,
        "generated_world_rank_fidelity_result_proven": False,
        "accepted_anchor_success_proven": False,
    }
    manifest = {
        "schema_version": SIM_E2E_MANIFEST_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "completed" if completed else "blocked",
        "provider_mode": args.provider_mode,
        "depth_provider": args.depth_provider if args.provider_mode == "real" else "fixture",
        "step_count_requested": int(args.step_count),
        "step_count_completed": len(step_records),
        "policy_requery_count": sum(1 for row in adapter_reports if row.get("adapter_status") == "completed"),
        "target_prompt": target_prompt,
        "start_frame_path": str(start_frame),
        "start_frame_source": start_source,
        "generated_frame_dir": str(generated_dir),
        "trace_path": str(trace_path),
        "harness_artifact_paths": harness_summary.get("artifact_paths", {}),
        "checks_status": harness_summary.get("status"),
        "adapter_status": "completed" if all_adapters_completed else "blocked",
        "safe_for_policy_requery_all_steps": all_safe_for_requery,
        "real_provider_triplet_completed": provider_triplet_completed,
        "optional_truth_label_validation_requested": False,
        "sim_only_provider_harness_e2e_completed": bool(completed),
        "blockers": []
        if completed
        else [
            *([] if harness_completed else ["harness_checks_not_completed"]),
            *([] if all_adapters_completed else ["policy_adapter_not_completed"]),
            *(
                []
                if args.provider_mode != "real" or provider_triplet_completed
                else ["real_provider_triplet_not_completed"]
            ),
        ],
        "claim_boundary": claim_boundary,
        "claim_boundaries": claim_boundary,
    }
    manifest_path = output_dir / "wam_sim_provider_e2e_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--generated-frame", type=Path, default=None)
    parser.add_argument("--step-count", type=int, default=2)
    parser.add_argument("--target-prompt", default="robot arm")
    parser.add_argument("--policy-id", default="wam_sim_provider_e2e_policy")
    parser.add_argument("--policy-schema", choices=["rgb_only", "rgbd_mask_pose"], default="rgbd_mask_pose")
    parser.add_argument("--provider-mode", choices=["fixture", "real"], default="real")
    parser.add_argument("--sam3-weights", type=Path, default=None)
    parser.add_argument("--sam3-confidence", type=float, default=0.01)
    parser.add_argument("--pose-model", default=DEFAULT_POSE_MODEL_PATH)
    parser.add_argument("--depth-provider", choices=["v2", "da3"], default="v2")
    parser.add_argument("--depth-model-id", default=DEFAULT_DEPTH_MODEL_ID)
    parser.add_argument("--da3-model-id", default=DEFAULT_DA3_MODEL_ID)
    parser.add_argument("--backend-timeout-seconds", type=int, default=900)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_sim_provider_e2e(args)
    print(json.dumps({"status": manifest["status"], "manifest_path": manifest["manifest_path"]}))
    return 0 if manifest["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
