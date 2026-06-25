"""Near-preserving image-model remediation for failed rendered policy POVs.

This module packages a failed 3D/simulator render and an image-model enhanced
replacement as review artifacts. The enhanced frame can improve WAM visual
reviewability, but it is not capture truth, geometry truth, or collision proof.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from .common import ensure_dir, write_json
from .wam_generated_video_review import assess_source_policy_observation_visual_qa


REQUEST_SCHEMA_VERSION = "image_model_render_remediation_request.v1"
MANIFEST_SCHEMA_VERSION = "image_model_render_remediation_manifest.v1"
CLAIM_BOUNDARY_SCHEMA_VERSION = "image_model_render_remediation_claim_boundary.v1"
SOURCE_KIND = "image_model_enhanced_3d_render_seed"
DEFAULT_MODEL = "gpt-image-2"
ENABLE_ENV = "BLUEPRINT_ALLOW_IMAGE_MODEL_RENDER_REMEDIATION"
COMMAND_ENV = "BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_COMMAND"
MODEL_ENV = "BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_MODEL"
TIMEOUT_ENV = "BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_TIMEOUT_SECONDS"
REQUEST_PATH_ENV = "BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_REQUEST_PATH"
OUTPUT_DIR_ENV = "BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_OUTPUT_DIR"
RESPONSE_PATH_ENV = "BLUEPRINT_IMAGE_MODEL_RENDER_REMEDIATION_RESPONSE_PATH"


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _truthy(value: str | None) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "y", "on"}


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
    return {
        "path": str(path),
        "width": width,
        "height": height,
        "mode": mode,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def image_model_render_remediation_enabled() -> bool:
    return _truthy(os.getenv(ENABLE_ENV))


def image_model_render_remediation_claim_boundary() -> dict[str, Any]:
    return {
        "schema_version": CLAIM_BOUNDARY_SCHEMA_VERSION,
        "source_kind": SOURCE_KIND,
        "capture_truth": False,
        "geometry_truth": False,
        "collision_truth": False,
        "raw_capture_evidence": False,
        "visual_seed_for_wam_experiment": True,
        "provider_success_separate_from_visually_useful_rollout": True,
        "visually_useful_rollout": False,
        "visual_seed_quality_is_not_rollout_quality": True,
        "near_preserving_image_enhancement_only": True,
        "original_3d_render_preserved_for_audit": True,
        "enhanced_frame_is_not_scene_state_authority": True,
        "enhanced_frame_is_not_geometry_or_collision_proof": True,
        "physical_robot_readiness_proven": False,
        "deployment_readiness_proven": False,
        "safety_validation_proven": False,
        "real_world_manipulation_success_proven": False,
        "raw_secret_values_recorded": False,
    }


def build_image_model_render_remediation_prompt(
    *,
    task_id: str | None,
    target_object_id: str | None,
    source_visual_qa: Mapping[str, Any],
    model: str,
) -> str:
    blockers = ", ".join(str(item) for item in source_visual_qa.get("blockers") or [])
    task_text = task_id or "the declared robot manipulation task"
    target_text = target_object_id or "the visible task target"
    return (
        f"Use {model} for a near-preserving enhancement of the provided first-person "
        f"robot policy observation. Preserve the camera viewpoint, robot pose, scene "
        f"layout, object positions, target identity, and composition as closely as "
        f"possible while improving reviewability only. The task is {task_text}; the "
        f"target is {target_text}. Address these visual QA blockers without adding "
        f"new task state or invented success evidence: {blockers or 'none provided'}. "
        "Improve exposure, sharpness, and local detail enough for WAM visual rollout "
        "review. Do not add text, UI, labels, watermarks, extra robot limbs, new "
        "objects, changed geometry, changed contact state, or changed target state."
    )


def _blocked_manifest(
    *,
    output_dir: Path,
    generated_at: str,
    original_frame_path: Path | None,
    source_visual_qa: Mapping[str, Any],
    blockers: list[str],
    model: str,
    packaged_original_frame_path: Path | None = None,
    request_path: Path | None = None,
    response_path: Path | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "source_kind": SOURCE_KIND,
        "model": model,
        "remediation_attempted": False,
        "near_preserving_enhancement_requested": True,
        "original_frame_path": str(original_frame_path) if original_frame_path else None,
        "packaged_original_frame_path": str(packaged_original_frame_path)
        if packaged_original_frame_path
        else None,
        "original_frame_info": _image_info(original_frame_path)
        if original_frame_path and original_frame_path.is_file()
        else None,
        "packaged_original_frame_info": _image_info(packaged_original_frame_path)
        if packaged_original_frame_path and packaged_original_frame_path.is_file()
        else None,
        "source_visual_qa_status": source_visual_qa.get("status"),
        "source_visual_qa_blockers": list(source_visual_qa.get("blockers") or []),
        "request_path": str(request_path) if request_path else None,
        "response_path": str(response_path) if response_path else None,
        "enhanced_frame_path": None,
        "enhanced_source_visual_qa_path": None,
        "enhanced_source_visual_qa_status": None,
        "blockers": sorted(set(blockers)),
        "claim_boundary": image_model_render_remediation_claim_boundary(),
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(output_dir / "image_model_render_remediation_manifest.json", manifest)
    return manifest


def _format_command(command: str, *, request_path: Path, output_dir: Path) -> list[str]:
    substitutions = {
        "request_path": str(request_path),
        "output_dir": str(output_dir),
    }
    try:
        formatted = command.format(**substitutions)
    except (KeyError, IndexError, ValueError):
        formatted = command
    return shlex.split(formatted)


def _canonicalize_enhanced_image(source: Path, destination: Path) -> Path:
    if source.resolve() == destination.resolve():
        return destination
    with Image.open(source) as image:
        image.convert("RGB").save(destination)
    return destination


def run_image_model_render_remediation(
    *,
    original_frame_path: str | Path | None,
    source_visual_qa: Mapping[str, Any],
    output_dir: str | Path,
    generated_at: str,
    task_id: str | None = None,
    target_object_id: str | None = None,
    object_index: Mapping[str, Any] | Sequence[Any] | None = None,
    eval_ready_task_grounding: Mapping[str, Any] | None = None,
    semantic_artifact_base_dir: str | Path | None = None,
    visual_profile: str = "review_quality",
    review_quality_required: bool = True,
    command: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run the configured enhancement command and package original/enhanced frames."""

    output = Path(output_dir).expanduser().resolve()
    ensure_dir(output)
    resolved_original = (
        Path(original_frame_path).expanduser().resolve() if original_frame_path else None
    )
    selected_model = _string(model or os.getenv(MODEL_ENV)) or DEFAULT_MODEL
    selected_command = _string(command or os.getenv(COMMAND_ENV))
    prompt = build_image_model_render_remediation_prompt(
        task_id=task_id,
        target_object_id=target_object_id,
        source_visual_qa=source_visual_qa,
        model=selected_model,
    )
    boundary = image_model_render_remediation_claim_boundary()
    request_path = output / "image_model_render_remediation_request.json"
    response_path = output / "image_model_render_remediation_response.json"

    if not resolved_original or not resolved_original.is_file():
        return _blocked_manifest(
            output_dir=output,
            generated_at=generated_at,
            original_frame_path=resolved_original,
            source_visual_qa=source_visual_qa,
            blockers=["image_model_render_remediation_original_frame_missing"],
            model=selected_model,
            request_path=request_path,
            response_path=response_path,
        )

    original_copy = output / f"original_initial_policy_frame{resolved_original.suffix or '.png'}"
    shutil.copy2(resolved_original, original_copy)
    request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_kind": SOURCE_KIND,
        "model": selected_model,
        "prompt": prompt,
        "original_frame_path": str(resolved_original),
        "packaged_original_frame_path": str(original_copy),
        "task_id": task_id,
        "target_object_id": target_object_id,
        "visual_profile": visual_profile,
        "review_quality_required": bool(review_quality_required),
        "failed_source_visual_qa": dict(source_visual_qa),
        "near_preserving_enhancement_requested": True,
        "claim_boundary": boundary,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(request_path, request)
    write_json(output / "image_model_render_remediation_claim_boundary.json", boundary)

    if not selected_command:
        return _blocked_manifest(
            output_dir=output,
            generated_at=generated_at,
            original_frame_path=resolved_original,
            packaged_original_frame_path=original_copy,
            source_visual_qa=source_visual_qa,
            blockers=["image_model_render_remediation_command_not_configured"],
            model=selected_model,
            request_path=request_path,
            response_path=response_path,
        )

    try:
        timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else (_string(os.getenv(TIMEOUT_ENV)) or 300.0)
        )
    except (TypeError, ValueError):
        timeout = 300.0
    args = _format_command(selected_command, request_path=request_path, output_dir=output)
    if not args:
        return _blocked_manifest(
            output_dir=output,
            generated_at=generated_at,
            original_frame_path=resolved_original,
            packaged_original_frame_path=original_copy,
            source_visual_qa=source_visual_qa,
            blockers=["image_model_render_remediation_command_empty"],
            model=selected_model,
            request_path=request_path,
            response_path=response_path,
        )

    env = dict(os.environ)
    env[REQUEST_PATH_ENV] = str(request_path)
    env[OUTPUT_DIR_ENV] = str(output)
    env[RESPONSE_PATH_ENV] = str(response_path)
    try:
        completed = subprocess.run(
            args,
            cwd=str(output),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _blocked_manifest(
            output_dir=output,
            generated_at=generated_at,
            original_frame_path=resolved_original,
            packaged_original_frame_path=original_copy,
            source_visual_qa=source_visual_qa,
            blockers=["image_model_render_remediation_command_timeout"],
            model=selected_model,
            request_path=request_path,
            response_path=response_path,
        )
    except OSError:
        return _blocked_manifest(
            output_dir=output,
            generated_at=generated_at,
            original_frame_path=resolved_original,
            packaged_original_frame_path=original_copy,
            source_visual_qa=source_visual_qa,
            blockers=["image_model_render_remediation_command_launch_failed"],
            model=selected_model,
            request_path=request_path,
            response_path=response_path,
        )

    command_execution = {
        "returncode": completed.returncode,
        "command_arg_count": len(args),
        "stdout_omitted_to_avoid_secret_leakage": True,
        "stderr_omitted_to_avoid_secret_leakage": True,
        "raw_secret_values_recorded": False,
    }
    if completed.returncode != 0:
        manifest = _blocked_manifest(
            output_dir=output,
            generated_at=generated_at,
            original_frame_path=resolved_original,
            packaged_original_frame_path=original_copy,
            source_visual_qa=source_visual_qa,
            blockers=["image_model_render_remediation_command_nonzero_exit"],
            model=selected_model,
            request_path=request_path,
            response_path=response_path,
        )
        manifest["command_execution"] = command_execution
        write_json(output / "image_model_render_remediation_manifest.json", manifest)
        return manifest

    response: dict[str, Any] = {}
    if response_path.is_file():
        value = json.loads(response_path.read_text(encoding="utf-8"))
        response = dict(value) if isinstance(value, Mapping) else {}
    response_image = _string(
        response.get("enhanced_image_path")
        or response.get("output_image_path")
        or response.get("image_path")
    )
    candidate = Path(response_image).expanduser() if response_image else output / "enhanced_initial_policy_frame.png"
    if not candidate.is_absolute():
        candidate = output / candidate
    if not candidate.is_file():
        return _blocked_manifest(
            output_dir=output,
            generated_at=generated_at,
            original_frame_path=resolved_original,
            packaged_original_frame_path=original_copy,
            source_visual_qa=source_visual_qa,
            blockers=["image_model_render_remediation_enhanced_frame_missing"],
            model=selected_model,
            request_path=request_path,
            response_path=response_path,
        )

    enhanced_frame = _canonicalize_enhanced_image(
        candidate,
        output / "enhanced_initial_policy_frame.png",
    )
    enhanced_qa = assess_source_policy_observation_visual_qa(
        enhanced_frame,
        generated_at=generated_at,
        target_object_id=target_object_id,
        task_id=task_id,
        object_index=object_index,
        eval_ready_task_grounding=eval_ready_task_grounding,
        semantic_artifact_base_dir=semantic_artifact_base_dir,
        visual_profile=visual_profile,
        review_quality_required=review_quality_required,
    )
    enhanced_qa_path = output / "enhanced_source_policy_observation_visual_qa.json"
    write_json(enhanced_qa_path, enhanced_qa)
    enhanced_passed = enhanced_qa.get("status") == "passed_visual_quality_gate"
    blockers = []
    if not enhanced_passed:
        blockers.append("image_model_render_remediation_enhanced_visual_qa_failed")
        blockers.extend(str(item) for item in enhanced_qa.get("blockers") or [])

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed" if enhanced_passed else "blocked",
        "source_kind": SOURCE_KIND,
        "model": _string(response.get("model")) or selected_model,
        "provider": _string(response.get("provider")) or None,
        "remediation_attempted": True,
        "near_preserving_enhancement_requested": True,
        "original_frame_path": str(resolved_original),
        "packaged_original_frame_path": str(original_copy),
        "original_frame_info": _image_info(original_copy),
        "enhanced_frame_path": str(enhanced_frame),
        "enhanced_frame_info": _image_info(enhanced_frame),
        "request_path": str(request_path),
        "response_path": str(response_path) if response_path.is_file() else None,
        "enhanced_source_visual_qa_path": str(enhanced_qa_path),
        "enhanced_source_visual_qa_status": enhanced_qa.get("status"),
        "source_visual_qa_status": source_visual_qa.get("status"),
        "source_visual_qa_blockers": list(source_visual_qa.get("blockers") or []),
        "command_execution": command_execution,
        "blockers": sorted(set(blockers)),
        "claim_boundary": boundary,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(output / "image_model_render_remediation_manifest.json", manifest)
    return manifest
