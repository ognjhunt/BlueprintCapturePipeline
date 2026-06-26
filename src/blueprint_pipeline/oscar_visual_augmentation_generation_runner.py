"""Run visual augmentation generation jobs for OSCAR/Cosmos-swappable packets.

This runner consumes an ``oscar_visual_augmentation_packet`` and fans out one
generation request per visual variant. The backend is intentionally generic:
OSCAR, Cosmos, or a future model can sit behind the same command contract. A
fixture backend exists only to make the lane locally testable.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .oscar_visual_augmentation_packet import (
    CLAIM_BOUNDARY,
    PACKET_MANIFEST_NAME,
    build_oscar_visual_augmentation_packet,
)
from .wam_generated_video_review import (
    validate_generated_mp4_for_review,
    visual_smoke_generated_rollouts_for_review,
)


GENERATION_RUN_SCHEMA_VERSION = "oscar_visual_augmentation_generation_run.v1"
GENERATION_REQUEST_SCHEMA_VERSION = "visual_augmentation_generation_request.v1"
GENERATION_RESULT_SCHEMA_VERSION = "visual_augmentation_backend_generation_result.v1"
GENERATION_QA_SCHEMA_VERSION = "visual_augmentation_generation_qa.v1"
TRAINING_READINESS_SCHEMA_VERSION = "visual_augmentation_training_readiness.v1"
DATASET_MANIFEST_SCHEMA_VERSION = "visual_augmentation_training_dataset_manifest.v1"

DEFAULT_BACKEND_ID = "oscar_wam"
FIXTURE_BACKEND_ID = "fixture_visual_augmentation"

GENERIC_COMMAND_ENVS = (
    "BLUEPRINT_VISUAL_AUGMENTATION_BACKEND_COMMAND",
    "BLUEPRINT_OSCAR_VISUAL_AUGMENTATION_COMMAND",
)
BACKEND_COMMAND_ENVS: dict[str, tuple[str, ...]] = {
    "oscar_wam": (
        "BLUEPRINT_OSCAR_VISUAL_AUGMENTATION_COMMAND",
        "BLUEPRINT_VISUAL_AUGMENTATION_BACKEND_COMMAND",
    ),
    "cosmos_wam": (
        "BLUEPRINT_COSMOS_VISUAL_AUGMENTATION_COMMAND",
        "BLUEPRINT_VISUAL_AUGMENTATION_BACKEND_COMMAND",
    ),
    "future_video_wam": GENERIC_COMMAND_ENVS,
}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _string(value).lower() in {"1", "true", "yes", "y", "on"}


def _jsonl_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    content = "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _sha_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: str | Path | None, *, base_dir: Path) -> dict[str, Any]:
    if path is None:
        return {"path": None, "absolute_path": None, "exists": False, "size_bytes": 0}
    resolved = Path(path).expanduser().resolve()
    return {
        "path": os.path.relpath(resolved, start=base_dir).replace("\\", "/"),
        "absolute_path": str(resolved),
        "exists": resolved.is_file(),
        "size_bytes": resolved.stat().st_size if resolved.is_file() else 0,
        "sha256": _sha_file(resolved),
    }


def _safe_component(value: Any, *, fallback: str = "variant") -> str:
    text = _string(value).lower()
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part) or fallback


def _load_packet(path: str | Path) -> tuple[Path, dict[str, Any]]:
    packet_path = Path(path).expanduser().resolve()
    payload = read_json_any(packet_path)
    return packet_path, dict(payload) if isinstance(payload, Mapping) else {}


def _resolve_asset_path(packet: Mapping[str, Any], name: str) -> Path | None:
    asset = _mapping(_mapping(packet.get("input_assets")).get(name))
    for key in ("absolute_path", "path"):
        value = _string(asset.get(key))
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                source_context = _mapping(packet.get("source_context"))
                job_dir = _string(source_context.get("job_dir"))
                if job_dir:
                    path = Path(job_dir) / path
            return path.resolve()
    return None


def _command_from_env(backend_id: str) -> tuple[str, str | None]:
    envs = BACKEND_COMMAND_ENVS.get(backend_id, GENERIC_COMMAND_ENVS)
    for name in envs:
        command = _string(os.getenv(name))
        if command:
            return command, name
    return "", None


def _backend_truth_from_result(payload: Mapping[str, Any]) -> dict[str, bool]:
    truth_boundary = _mapping(payload.get("truth_boundary"))
    generated_video_is_model_output = (
        payload.get("model_derived") is True
        or payload.get("generated_video_is_model_output") is True
        or payload.get("provider_generated_video_is_model_output") is True
        or truth_boundary.get("generated_video_is_model_output") is True
    )
    learned_model_ran = (
        payload.get("learned_model_ran") is True
        or payload.get("learned_wam_model_ran") is True
        or payload.get("fresh_model_run_claimed") is True
        or payload.get("fresh_provider_model_run_claimed") is True
    )
    return {
        "model_derived": bool(generated_video_is_model_output),
        "learned_model_execution_proven": bool(generated_video_is_model_output and learned_model_ran),
    }


def _result_video_path(payload: Mapping[str, Any], *, default_video: Path) -> Path | None:
    candidates = [
        payload.get("generated_video_path"),
        payload.get("video_path"),
        payload.get("output_video_path"),
    ]
    for rollout in payload.get("rollouts") or []:
        if isinstance(rollout, Mapping):
            candidates.extend(
                [
                    rollout.get("generated_video_path"),
                    rollout.get("video_path"),
                    rollout.get("output_video_path"),
                ]
            )
    for value in candidates:
        text = _string(value)
        if not text:
            continue
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = default_video.parent / path
        return path.resolve()
    return default_video.resolve() if default_video.is_file() else None


def _read_backend_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _fixture_frame(width: int, height: int, variant_id: str, frame_index: int) -> Any:
    import cv2  # type: ignore[import-not-found]
    import numpy as np

    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:, :, 0] = (x + frame_index * 5) % 255
    base[:, :, 1] = (y + frame_index * 3) % 255
    base[:, :, 2] = ((x // 2 + y // 2 + frame_index * 7) % 255).astype("uint8")
    step = max(12, width // 16)
    for offset in range(0, width + height, step * 2):
        cv2.line(
            base,
            (max(0, offset - height), min(height - 1, offset)),
            (min(width - 1, offset), max(0, offset - width)),
            (245, 245, 245),
            2,
        )
    box_size = max(24, min(width, height) // 7)
    x0 = int((frame_index * 11) % max(width - box_size, 1))
    y0 = int(height * 0.55 + ((frame_index % 7) - 3) * 4)
    y0 = max(0, min(height - box_size, y0))
    cv2.rectangle(base, (x0, y0), (x0 + box_size, y0 + box_size), (20, 220, 60), -1)
    cv2.putText(
        base,
        variant_id[:32],
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return base


def _write_fixture_video(
    *,
    first_frame_path: Path | None,
    output_video: Path,
    variant_id: str,
    frame_count: int = 24,
    fps: float = 15.0,
    width: int = 640,
    height: int = 480,
) -> None:
    import cv2  # type: ignore[import-not-found]

    ensure_dir(output_video.parent)
    decoded = None
    if first_frame_path and first_frame_path.is_file():
        decoded = cv2.imread(str(first_frame_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("fixture_video_writer_failed")
    try:
        for frame_index in range(frame_count):
            frame = _fixture_frame(width, height, variant_id, frame_index)
            if decoded is not None:
                source = cv2.resize(decoded, (width, height))
                alpha = 0.28
                frame = cv2.addWeighted(source, alpha, frame, 1.0 - alpha, 0.0)
            writer.write(frame)
    finally:
        writer.release()


def _run_fixture_backend(
    *,
    request: Mapping[str, Any],
    output_video: Path,
    output_result: Path,
    first_frame_path: Path | None,
    generated_at: str,
) -> dict[str, Any]:
    variant_id = _string(request.get("variant_id")) or "variant"
    _write_fixture_video(
        first_frame_path=first_frame_path,
        output_video=output_video,
        variant_id=variant_id,
    )
    result = {
        "schema_version": GENERATION_RESULT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "completed",
        "backend_id": FIXTURE_BACKEND_ID,
        "variant_id": variant_id,
        "generated_video_path": str(output_video),
        "fixture_backend_used": True,
        "model_derived": False,
        "learned_model_ran": False,
        "learned_model_execution_proven": False,
        "purpose": "local_plumbing_and_visual_qa_test_only",
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "fixture_output_is_not_training_data": True,
            "fixture_output_is_not_model_backend_output": True,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(output_result, result)
    return result


def _run_command_backend(
    *,
    backend_command: str,
    backend_command_env: str | None,
    request_path: Path,
    output_video: Path,
    output_result: Path,
    packet_manifest_path: Path,
    variant_id: str,
    timeout_seconds: float,
    generated_at: str,
) -> dict[str, Any]:
    argv = shlex.split(backend_command)
    if not argv:
        return {
            "schema_version": GENERATION_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "blocked",
            "variant_id": variant_id,
            "blockers": ["backend_command_empty"],
            "raw_credentials_written_to_artifacts": False,
            "secret_hashes_written_to_artifacts": False,
        }
    env = dict(os.environ)
    env.update(
        {
            "BLUEPRINT_VISUAL_AUGMENTATION_REQUEST": str(request_path),
            "BLUEPRINT_VISUAL_AUGMENTATION_OUTPUT": str(output_result),
            "BLUEPRINT_VISUAL_AUGMENTATION_OUTPUT_VIDEO": str(output_video),
            "BLUEPRINT_VISUAL_AUGMENTATION_PACKET": str(packet_manifest_path),
            "BLUEPRINT_VISUAL_AUGMENTATION_VARIANT_ID": variant_id,
        }
    )
    started = utc_now_iso()
    try:
        completed = subprocess.run(
            argv,
            cwd=str(output_result.parent),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        subprocess_detail = {
            "status": "completed" if completed.returncode == 0 else "failed",
            "started_at": started,
            "completed_at": utc_now_iso(),
            "returncode": completed.returncode,
            "timeout_seconds": timeout_seconds,
            "command_env": backend_command_env,
            "command_argv0": argv[0],
            "command_argc": len(argv),
            "stdout_bytes": len((completed.stdout or "").encode("utf-8", errors="replace")),
            "stderr_bytes": len((completed.stderr or "").encode("utf-8", errors="replace")),
            "raw_command_recorded": False,
        }
    except subprocess.TimeoutExpired as exc:
        subprocess_detail = {
            "status": "timeout",
            "started_at": started,
            "completed_at": utc_now_iso(),
            "timeout_seconds": timeout_seconds,
            "command_env": backend_command_env,
            "command_argv0": argv[0],
            "command_argc": len(argv),
            "stdout_bytes": len((exc.stdout or b"") if isinstance(exc.stdout, bytes) else (exc.stdout or "").encode()),
            "stderr_bytes": len((exc.stderr or b"") if isinstance(exc.stderr, bytes) else (exc.stderr or "").encode()),
            "raw_command_recorded": False,
        }

    payload = _read_backend_result(output_result)
    if not payload:
        payload = {
            "schema_version": GENERATION_RESULT_SCHEMA_VERSION,
            "generated_at": generated_at,
            "status": "completed" if subprocess_detail["status"] == "completed" else "blocked",
            "variant_id": variant_id,
        }
    payload["backend_subprocess"] = subprocess_detail
    payload.setdefault("raw_credentials_written_to_artifacts", False)
    payload.setdefault("secret_hashes_written_to_artifacts", False)
    write_json(output_result, payload)
    return payload


def _write_qa_and_readiness(
    *,
    output_dir: Path,
    generated_rows: Sequence[Mapping[str, Any]],
    generated_at: str,
    require_review_quality_profile: bool,
    fixture_backend_used: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    decode_validations: list[dict[str, Any]] = []
    rollouts: list[dict[str, Any]] = []
    for row in generated_rows:
        video_path = _string(row.get("generated_video_path"))
        if video_path:
            decode_validations.append(validate_generated_mp4_for_review(video_path))
            rollouts.append(
                {
                    "rollout_id": row.get("variant_id"),
                    "generated_video_path": video_path,
                    "model_candidate": row.get("model_backend_id"),
                }
            )
    smoke = visual_smoke_generated_rollouts_for_review(
        rollouts=rollouts,
        output_dir=output_dir,
        generated_at=generated_at,
        require_review_quality_profile=require_review_quality_profile,
    )
    decode_ok = bool(decode_validations) and all(
        row.get("status") == "completed" for row in decode_validations
    )
    all_model_derived = bool(generated_rows) and all(
        row.get("model_derived") is True for row in generated_rows
    )
    all_learned_proven = bool(generated_rows) and all(
        row.get("learned_model_execution_proven") is True for row in generated_rows
    )
    smoke_ok = smoke.get("status") == "passed_visual_quality_smoke"
    qa = {
        "schema_version": GENERATION_QA_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed_visual_qa_smoke" if decode_ok and smoke_ok else "blocked_or_review_needed",
        "generated_video_count": len(generated_rows),
        "decode_validations": decode_validations,
        "visual_smoke": smoke,
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "decode_success_is_not_training_readiness": True,
            "visual_smoke_is_not_policy_success": True,
        },
    }
    readiness_status = (
        "review_ready_model_derived_training_candidate"
        if decode_ok and smoke_ok and all_model_derived and all_learned_proven
        else "review_ready_backend_output_pending_model_truth"
        if decode_ok and smoke_ok and generated_rows and not fixture_backend_used
        else "fixture_plumbing_only_not_training_data"
        if fixture_backend_used
        else "blocked_training_use_pending_quality_or_review"
    )
    readiness = {
        "schema_version": TRAINING_READINESS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": readiness_status,
        "training_ready_without_review": False,
        "requires_human_or_vlm_review_before_training_use": True,
        "fixture_backend_used": fixture_backend_used,
        "all_outputs_model_derived": all_model_derived,
        "all_learned_model_execution_proven": all_learned_proven,
        "decode_level_reviewable": decode_ok,
        "visual_smoke_passed": smoke_ok,
        "usable_for_visual_distribution_shift_eval_after_review": bool(
            decode_ok and smoke_ok and not fixture_backend_used
        ),
        "usable_for_policy_pretraining_without_real_or_sim_truth_mix": False,
        "usable_for_site_task_finetuning_without_real_site_capture_review": False,
        "recommended_training_use": (
            "low_or_medium_weight_visual_augmentation_after_review_and_real_or_sim_truth_mix"
            if decode_ok and smoke_ok and all_model_derived and all_learned_proven
            else "not_training_data"
            if fixture_backend_used
            else "hold_for_backend_truth_and_visual_review"
        ),
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "training_candidate_manifest_is_not_training_approval": True,
            "fine_tuning_requires_task_owner_acceptance_or_real_site_anchor_review": True,
        },
    }
    dataset_rows: list[dict[str, Any]] = []
    for row in generated_rows:
        dataset_rows.append(
            {
                "schema_version": "visual_augmentation_training_candidate_episode.v1",
                "variant_id": row.get("variant_id"),
                "model_backend_id": row.get("model_backend_id"),
                "generated_video_path": row.get("generated_video_path"),
                "model_derived": bool(row.get("model_derived")),
                "learned_model_execution_proven": bool(row.get("learned_model_execution_proven")),
                "use_status": (
                    "candidate_requires_review"
                    if row.get("model_derived") is True and not fixture_backend_used
                    else "plumbing_fixture_not_training_data"
                    if fixture_backend_used
                    else "hold_for_model_truth_review"
                ),
                "contact_physics_proven": False,
                "real_robot_readiness_proven": False,
                "deployment_safety_proven": False,
            }
        )
    export_dir = output_dir / "exports" / "visual_augmentation"
    _jsonl_write(export_dir / "episodes.jsonl", dataset_rows)
    dataset_manifest = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": (
            "candidate_dataset_written_requires_review"
            if dataset_rows and not fixture_backend_used
            else "fixture_dataset_written_for_plumbing_only"
            if dataset_rows
            else "blocked_no_generated_rows"
        ),
        "episode_count": len(dataset_rows),
        "episodes_jsonl": str(export_dir / "episodes.jsonl"),
        "training_ready_without_review": False,
        "requires_review_before_training_use": True,
        "claim_boundary": dict(readiness["claim_boundary"]),
    }
    write_json(output_dir / "visual_augmentation_generation_qa_manifest.json", qa)
    write_json(output_dir / "visual_augmentation_training_readiness_manifest.json", readiness)
    write_json(output_dir / "visual_augmentation_training_dataset_manifest.json", dataset_manifest)
    return qa, readiness, dataset_manifest


def run_visual_augmentation_generation(
    *,
    packet_manifest: str | Path,
    output_dir: str | Path | None = None,
    backend_id: str | None = None,
    backend_command: str | None = None,
    backend_mode: str = "auto",
    allow_fixture_backend: bool = False,
    max_variants: int | None = None,
    timeout_seconds: float = 3600.0,
    require_review_quality_profile: bool = True,
) -> dict[str, Any]:
    packet_path, packet = _load_packet(packet_manifest)
    packet_dir = packet_path.parent
    resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir else packet_dir
    ensure_dir(resolved_output_dir)
    generated_at = utc_now_iso()
    selected_backend_id = _string(backend_id or packet.get("selected_backend_id")) or DEFAULT_BACKEND_ID

    variants = [dict(row) for row in packet.get("variant_requests") or [] if isinstance(row, Mapping)]
    if max_variants is not None and max_variants >= 0:
        variants = variants[:max_variants]

    first_frame_path = _resolve_asset_path(packet, "first_frame")
    skeleton_video_path = _resolve_asset_path(packet, "skeleton_conditioning_video")
    camera_provenance_path = _resolve_asset_path(packet, "camera_provenance")
    skeleton_provenance_path = _resolve_asset_path(packet, "skeleton_provenance")

    blockers: list[str] = []
    if _string(packet.get("status")).startswith("blocked"):
        blockers.append("packet_status_blocked")
        blockers.extend(str(item) for item in packet.get("blockers") or [])
    for blocker_name, path in (
        ("first_frame_missing", first_frame_path),
        ("skeleton_conditioning_video_missing", skeleton_video_path),
        ("camera_provenance_missing", camera_provenance_path),
        ("skeleton_provenance_missing", skeleton_provenance_path),
    ):
        if path is None or not path.is_file():
            blockers.append(blocker_name)
    if not variants:
        blockers.append("no_visual_variants_requested")

    command, command_env = (_string(backend_command), None)
    if not command:
        command, command_env = _command_from_env(selected_backend_id)

    use_fixture = False
    if backend_mode == "fixture":
        if not allow_fixture_backend:
            blockers.append("fixture_backend_requires_allow_fixture_backend")
        use_fixture = True
        selected_backend_id = FIXTURE_BACKEND_ID
    elif backend_mode == "command":
        if not command:
            blockers.append("backend_command_missing")
    elif backend_mode == "auto":
        if command:
            use_fixture = False
        elif allow_fixture_backend:
            use_fixture = True
            selected_backend_id = FIXTURE_BACKEND_ID
        else:
            blockers.append("backend_command_missing_and_fixture_not_authorized")
    else:
        blockers.append(f"unsupported_backend_mode:{backend_mode}")

    request_dir = resolved_output_dir / "generation_requests"
    result_dir = resolved_output_dir / "backend_results"
    video_dir = resolved_output_dir / "generated_videos"
    for path in (request_dir, result_dir, video_dir):
        ensure_dir(path)

    request_rows: list[dict[str, Any]] = []
    backend_results: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []

    if not blockers:
        for index, variant in enumerate(variants, start=1):
            variant_id = _string(variant.get("variant_id")) or f"variant_{index:03d}"
            safe_variant = _safe_component(variant_id, fallback=f"variant_{index:03d}")
            safe_backend = _safe_component(selected_backend_id, fallback="backend")
            output_video = video_dir / f"{safe_backend}_{safe_variant}.mp4"
            output_result = result_dir / f"{safe_backend}_{safe_variant}.json"
            request_path = request_dir / f"{safe_backend}_{safe_variant}.json"
            request = {
                "schema_version": GENERATION_REQUEST_SCHEMA_VERSION,
                "generated_at": generated_at,
                "packet_manifest_path": str(packet_path),
                "variant_index": index,
                "variant_id": variant_id,
                "variant_request": variant,
                "selected_backend_id": selected_backend_id,
                "backend_mode": "fixture" if use_fixture else "command",
                "first_frame_path": str(first_frame_path),
                "skeleton_conditioning_video_path": str(skeleton_video_path),
                "camera_provenance_path": str(camera_provenance_path),
                "skeleton_provenance_path": str(skeleton_provenance_path),
                "output_video_path": str(output_video),
                "output_result_path": str(output_result),
                "claim_boundary": dict(CLAIM_BOUNDARY),
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
            }
            write_json(request_path, request)
            request_rows.append(request)
            if use_fixture:
                result = _run_fixture_backend(
                    request=request,
                    output_video=output_video,
                    output_result=output_result,
                    first_frame_path=first_frame_path,
                    generated_at=generated_at,
                )
            else:
                result = _run_command_backend(
                    backend_command=command,
                    backend_command_env=command_env,
                    request_path=request_path,
                    output_video=output_video,
                    output_result=output_result,
                    packet_manifest_path=packet_path,
                    variant_id=variant_id,
                    timeout_seconds=timeout_seconds,
                    generated_at=generated_at,
                )
            result_video_path = _result_video_path(result, default_video=output_video)
            truth = _backend_truth_from_result(result)
            status = _string(result.get("status"))
            video_exists = bool(result_video_path and result_video_path.is_file())
            blockers_for_result = list(result.get("blockers") or [])
            if not video_exists:
                blockers_for_result.append("generated_video_missing")
            row = {
                "schema_version": GENERATION_RESULT_SCHEMA_VERSION,
                "generated_at": generated_at,
                "status": "completed" if status == "completed" and video_exists else "blocked",
                "variant_id": variant_id,
                "model_backend_id": selected_backend_id,
                "backend_mode": "fixture" if use_fixture else "command",
                "generated_video_path": str(result_video_path) if result_video_path else None,
                "backend_result_path": str(output_result),
                "request_path": str(request_path),
                "model_derived": truth["model_derived"],
                "learned_model_execution_proven": truth["learned_model_execution_proven"],
                "fixture_backend_used": use_fixture,
                "blockers": sorted(set(str(item) for item in blockers_for_result if str(item))),
                "raw_credentials_written_to_artifacts": False,
                "secret_hashes_written_to_artifacts": False,
            }
            backend_results.append(row)
            if video_exists:
                generated_rows.append(row)

    refreshed_packet: dict[str, Any] | None = None
    if generated_rows:
        source_context = _mapping(packet.get("source_context"))
        variant_specs_path = resolved_output_dir / "visual_augmentation_runner_variant_specs.json"
        write_json(variant_specs_path, {"variants": variants})
        refreshed_packet = build_oscar_visual_augmentation_packet(
            capture_root=source_context.get("capture_root") or Path.cwd(),
            job_dir=source_context.get("job_dir"),
            output_dir=packet_dir,
            source_input_package=source_context.get("source_input_package"),
            first_frame=first_frame_path,
            skeleton_video=skeleton_video_path,
            camera_provenance=camera_provenance_path,
            skeleton_provenance=skeleton_provenance_path,
            variant_specs=variant_specs_path,
            generated_videos=[
                {
                    "variant_id": row["variant_id"],
                    "path": row["generated_video_path"],
                    "model_backend_id": row["model_backend_id"],
                    "model_derived": row["model_derived"],
                    "generated_artifact_kind": (
                        "model_derived_visual_augmentation"
                        if row["model_derived"]
                        else "fixture_or_backend_generated_support_video"
                    ),
                }
                for row in generated_rows
            ],
            selected_backend_id=selected_backend_id,
        )

    qa: dict[str, Any] = {
        "schema_version": GENERATION_QA_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "not_run",
        "generated_video_count": 0,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    readiness: dict[str, Any] = {
        "schema_version": TRAINING_READINESS_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked_no_generated_outputs",
        "training_ready_without_review": False,
        "requires_human_or_vlm_review_before_training_use": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    dataset_manifest: dict[str, Any] = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked_no_generated_rows",
        "episode_count": 0,
        "training_ready_without_review": False,
        "requires_review_before_training_use": True,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    if generated_rows:
        qa, readiness, dataset_manifest = _write_qa_and_readiness(
            output_dir=resolved_output_dir,
            generated_rows=generated_rows,
            generated_at=generated_at,
            require_review_quality_profile=require_review_quality_profile,
            fixture_backend_used=use_fixture,
        )

    command_configured = bool(command) and not use_fixture
    status = (
        "blocked"
        if blockers
        else "completed_fixture_test_outputs"
        if use_fixture and generated_rows
        else "completed_with_model_derived_outputs"
        if generated_rows and all(row.get("model_derived") is True for row in generated_rows)
        else "completed_with_outputs_pending_model_truth_review"
        if generated_rows
        else "blocked_no_generated_outputs"
    )
    run_manifest = {
        "schema_version": GENERATION_RUN_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "packet_manifest_path": str(packet_path),
        "output_dir": str(resolved_output_dir),
        "selected_backend_id": selected_backend_id,
        "backend_mode": "fixture" if use_fixture else "command",
        "backend_command_configured": command_configured,
        "backend_command_env": command_env if command_configured else None,
        "backend_command_raw_value_recorded": False,
        "fixture_backend_used": use_fixture,
        "variant_count_requested": len(variants),
        "generation_request_count": len(request_rows),
        "backend_result_count": len(backend_results),
        "generated_video_count": len(generated_rows),
        "training_ready_without_review": False,
        "requires_human_or_vlm_review_before_training_use": True,
        "contact_physics_proven": False,
        "real_robot_readiness_proven": False,
        "deployment_safety_proven": False,
        "blockers": sorted(set(blockers)),
        "requests_path": str(request_dir),
        "backend_results_path": str(result_dir),
        "generated_videos_path": str(video_dir),
        "generation_results": backend_results,
        "refreshed_packet_manifest": str(packet_dir / PACKET_MANIFEST_NAME)
        if refreshed_packet
        else None,
        "refreshed_packet_status": refreshed_packet.get("status") if refreshed_packet else None,
        "qa_manifest_path": str(resolved_output_dir / "visual_augmentation_generation_qa_manifest.json"),
        "training_readiness_manifest_path": str(
            resolved_output_dir / "visual_augmentation_training_readiness_manifest.json"
        ),
        "training_dataset_manifest_path": str(
            resolved_output_dir / "visual_augmentation_training_dataset_manifest.json"
        ),
        "qa_status": qa.get("status"),
        "training_readiness_status": readiness.get("status"),
        "dataset_status": dataset_manifest.get("status"),
        "claim_boundary": {
            **dict(CLAIM_BOUNDARY),
            "generation_run_is_not_contact_physics_proof": True,
            "generation_run_is_not_real_robot_readiness": True,
            "generation_run_is_not_deployment_safety": True,
            "generated_video_requires_review_before_training_use": True,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    write_json(resolved_output_dir / "visual_augmentation_generation_run_manifest.json", run_manifest)
    _jsonl_write(resolved_output_dir / "visual_augmentation_generation_results.jsonl", backend_results)
    return run_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--backend-id")
    parser.add_argument("--backend-command")
    parser.add_argument("--backend-mode", choices=("auto", "command", "fixture"), default="auto")
    parser.add_argument("--allow-fixture-backend", action="store_true")
    parser.add_argument("--max-variants", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--no-require-review-quality-profile",
        action="store_true",
        help="Run decode and visual smoke checks without review-quality media thresholds.",
    )
    args = parser.parse_args(argv)
    result = run_visual_augmentation_generation(
        packet_manifest=args.packet_manifest,
        output_dir=args.output_dir,
        backend_id=args.backend_id,
        backend_command=args.backend_command,
        backend_mode=args.backend_mode,
        allow_fixture_backend=args.allow_fixture_backend,
        max_variants=args.max_variants,
        timeout_seconds=args.timeout_seconds,
        require_review_quality_profile=not args.no_require_review_quality_profile,
    )
    manifest_path = Path(args.output_dir or Path(args.packet_manifest).expanduser().resolve().parent)
    print(
        "[oscar-visual-augmentation-generation] "
        f"manifest={manifest_path / 'visual_augmentation_generation_run_manifest.json'}"
    )
    print(f"[oscar-visual-augmentation-generation] status={result['status']}")
    return 0 if not str(result["status"]).startswith("blocked") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
