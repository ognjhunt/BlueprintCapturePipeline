"""Run a bounded review-quality WAM visual sanity pass before long rollouts."""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .unitree_groot_n17_sonic_vast_persistent_session import (
    PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_HEIGHT,
    PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_WIDTH,
    PERSISTENT_WAM_SHORT_VISUAL_SANITY_SCHEMA_VERSION,
    _camera_frame_path,
    _current_wam_visual_profile_settings,
    _load_policy_observation,
    _mapping,
    _read_json,
    _string,
    assess_source_policy_observation_visual_qa,
    run_persistent_session,
    run_persistent_session_runpod,
    validate_persistent_wam_short_visual_sanity_manifest,
)


SHORT_SANITY_FILENAME = "persistent_wam_short_visual_sanity_manifest.json"
REVIEW_QUALITY_ENV = {
    "BLUEPRINT_OSCAR_WAM_VISUAL_PROFILE": "review_quality",
    "BLUEPRINT_OSCAR_WAM_NUM_FRAMES": "24",
    "BLUEPRINT_OSCAR_WAM_HEIGHT": "480",
    "BLUEPRINT_OSCAR_WAM_WIDTH": "640",
    "BLUEPRINT_OSCAR_WAM_FPS": "15",
}


def _intish(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@contextmanager
def _temporary_env(updates: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _read_json_if_present(path: Any) -> dict[str, Any]:
    text = _string(path)
    if not text:
        return {}
    candidate = Path(text).expanduser()
    if not candidate.is_file():
        return {}
    try:
        return _read_json(candidate)
    except Exception:
        return {}


def _resolve_artifact_path(path: Any) -> str | None:
    text = _string(path)
    if not text:
        return None
    return str(Path(text).expanduser().resolve())


def _first_ffprobe_video_stream(metadata: Mapping[str, Any]) -> dict[str, Any]:
    streams = metadata.get("streams")
    if not isinstance(streams, Sequence) or isinstance(streams, (str, bytes, bytearray)):
        return {}
    for stream in streams:
        if isinstance(stream, Mapping):
            return dict(stream)
    return {}


def _review_media_resolution(video_status: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _mapping(video_status.get("ffprobe_metadata"))
    stream = _first_ffprobe_video_stream(metadata)
    width = _intish(stream.get("width")) or 0
    height = _intish(stream.get("height")) or 0
    passed = (
        width >= PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_WIDTH
        and height >= PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_HEIGHT
    )
    return {
        "width": width or None,
        "height": height or None,
        "minimum_width": PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_WIDTH,
        "minimum_height": PERSISTENT_WAM_SHORT_VISUAL_SANITY_MIN_REVIEW_MEDIA_HEIGHT,
        "passed": passed,
    }


def _paid_provider_status(*, provider: str, result: Mapping[str, Any]) -> dict[str, Any]:
    details = _mapping(result.get("details"))
    if provider == "runpod":
        create_path = _resolve_artifact_path(
            result.get("runpod_create_manifest_path")
            or details.get("runpod_create_manifest_path")
        )
        poll_path = _resolve_artifact_path(
            result.get("runpod_poll_manifest_path")
            or details.get("runpod_poll_manifest_path")
        )
        teardown_path = _resolve_artifact_path(
            result.get("runpod_teardown_manifest_path")
            or details.get("runpod_delete_manifest_path")
        )
        create_manifest = _read_json_if_present(create_path)
        poll_manifest = _read_json_if_present(poll_path)
        teardown_manifest = _read_json_if_present(teardown_path)
        used = bool(
            create_manifest.get("status") == "pod_created"
            or poll_manifest.get("pod_id")
            or teardown_manifest.get("pod_id")
        )
        continuing_spend = (
            teardown_manifest.get("continuing_spend_from_this_run")
            if teardown_manifest
            else poll_manifest.get("continuing_spend_from_this_run")
        )
        teardown_performed = bool(
            poll_manifest.get("teardown_performed")
            or teardown_manifest.get("status") == "completed"
        )
        return {
            "provider": provider,
            "used": used,
            "create_manifest_path": create_path,
            "poll_manifest_path": poll_path,
            "teardown_manifest_path": teardown_path,
            "teardown_status": _string(teardown_manifest.get("status"))
            or ("completed" if teardown_performed and continuing_spend is False else "missing"),
            "teardown_performed": teardown_performed,
            "continuing_spend_from_this_run": bool(continuing_spend)
            if continuing_spend is not None
            else None,
            "raw_credentials_written_to_artifacts": False,
        }
    if provider == "vast":
        adapter_path = _resolve_artifact_path(
            result.get("vast_provider_adapter_result_path")
            or details.get("vast_provider_adapter_result_path")
        )
        teardown_path = _resolve_artifact_path(
            result.get("vast_teardown_manifest_path")
            or details.get("vast_teardown_manifest_path")
        )
        adapter_manifest = _read_json_if_present(adapter_path)
        teardown_manifest = _read_json_if_present(teardown_path)
        used = bool(
            adapter_manifest.get("api_call_performed")
            or adapter_manifest.get("vast_side_effects_may_have_occurred")
            or teardown_manifest.get("vast_instance_ids")
        )
        return {
            "provider": provider,
            "used": used,
            "provider_adapter_result_path": adapter_path,
            "teardown_manifest_path": teardown_path,
            "teardown_status": _string(teardown_manifest.get("status"))
            or ("not_required_no_paid_provider" if not used else "missing"),
            "teardown_performed": bool(
                teardown_manifest.get("runner_gpu_teardown_completed")
                or teardown_manifest.get("status") == "completed"
            ),
            "continuing_spend_from_this_run": bool(
                teardown_manifest.get("continuing_spend_from_this_run")
            )
            if teardown_manifest
            else None,
            "raw_credentials_written_to_artifacts": False,
        }
    return {
        "provider": provider,
        "used": False,
        "teardown_status": "not_required_no_paid_provider",
        "teardown_performed": False,
        "continuing_spend_from_this_run": False,
        "raw_credentials_written_to_artifacts": False,
    }


def _manifest_blockers(
    *,
    result: Mapping[str, Any],
    visual_report: Mapping[str, Any],
    video_status: Mapping[str, Any],
    source_qa_status: str,
    paid_provider: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if result.get("status") != "completed":
        blockers.extend(str(item) for item in result.get("blockers") or [])
        blockers.append("short_visual_sanity_persistent_session_not_completed")
    if source_qa_status != "passed_visual_quality_gate":
        blockers.append("short_visual_sanity_source_observation_qa_not_passed")
    if visual_report.get("visual_success") is not True:
        blockers.append("short_visual_sanity_wam_visual_quality_failed")
    if video_status.get("status") != "completed":
        blockers.append("short_visual_sanity_review_video_not_completed")
    if not _mapping(video_status.get("ffprobe_metadata")):
        blockers.append("short_visual_sanity_ffprobe_metadata_missing")
    if video_status.get("ffprobe_command_ran") is not True:
        blockers.append("short_visual_sanity_ffprobe_command_not_ran")
    if _intish(video_status.get("ffprobe_returncode")) != 0:
        blockers.append("short_visual_sanity_ffprobe_returncode_not_zero")
    if not _review_media_resolution(video_status)["passed"]:
        blockers.append("short_visual_sanity_review_video_below_minimum_resolution")
    contact_sheet_path = _resolve_artifact_path(
        result.get("wam_rollout_contact_sheet_path")
        or _mapping(result.get("postprocess_artifacts")).get("wam_rollout_contact_sheet")
        or visual_report.get("contact_sheet_path")
    )
    if not contact_sheet_path or not Path(contact_sheet_path).is_file():
        blockers.append("short_visual_sanity_contact_sheet_missing")
    if paid_provider.get("used") is True:
        if paid_provider.get("continuing_spend_from_this_run") is not False:
            blockers.append("short_visual_sanity_paid_provider_teardown_not_zero_spend")
        if not _string(paid_provider.get("teardown_manifest_path")):
            blockers.append("short_visual_sanity_paid_provider_teardown_manifest_missing")
    return sorted(set(blockers))


def _write_blocked_preflight_manifest(
    *,
    output_dir: Path,
    generated_at: str,
    policy_observation_path: Path,
    provider: str,
    transition_count: int,
    source_qa: Mapping[str, Any],
    review_quality_settings: Mapping[str, Any],
) -> dict[str, Any]:
    source_qa_path = output_dir / "source_policy_observation_visual_qa.json"
    write_json(source_qa_path, source_qa)
    manifest = {
        "schema_version": PERSISTENT_WAM_SHORT_VISUAL_SANITY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "short_visual_sanity_passed": False,
        "policy_observation_path": str(policy_observation_path),
        "provider": provider,
        "requested_transition_count": transition_count,
        "requested_loop_step_count": transition_count + 1,
        "visual_profile": "review_quality",
        "review_quality_settings": dict(review_quality_settings),
        "capture_truth": False,
        "geometry_truth": False,
        "collision_truth": False,
        "provider_success": False,
        "provider_success_separate_from_visually_useful_rollout": True,
        "visually_useful_rollout": False,
        "source_policy_observation_visual_qa_status": source_qa.get("status"),
        "source_policy_observation_visual_qa_path": str(source_qa_path),
        "paid_provider": {
            "provider": provider,
            "used": False,
            "teardown_status": "not_required_prelaunch_blocked",
            "teardown_performed": False,
            "continuing_spend_from_this_run": False,
            "raw_credentials_written_to_artifacts": False,
        },
        "blockers": list(source_qa.get("blockers") or ["source_policy_observation_visual_qa_failed"]),
        "claim_boundary": {
            "short_visual_sanity_is_not_task_success_proof": True,
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
            "valid_mp4_or_provider_completed_is_not_visual_success": True,
            "provider_success": False,
            "provider_success_separate_from_visually_useful_rollout": True,
            "visually_useful_rollout": False,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "real_world_manipulation_success_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    manifest_path = output_dir / SHORT_SANITY_FILENAME
    manifest["short_visual_sanity_manifest_path"] = str(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def _build_short_sanity_manifest(
    *,
    generated_at: str,
    policy_observation_path: Path,
    provider: str,
    transition_count: int,
    result: Mapping[str, Any],
    review_quality_settings: Mapping[str, Any],
    manifest_output_dir: Path | None = None,
) -> dict[str, Any]:
    postprocess = _mapping(result.get("postprocess_artifacts"))
    visual_report_path = _resolve_artifact_path(
        result.get("wam_rollout_visual_quality_report_path")
        or postprocess.get("wam_rollout_visual_quality_report")
    )
    video_status_path = _resolve_artifact_path(
        result.get("video_review_status_path") or postprocess.get("video_review_status")
    )
    source_qa_path = _resolve_artifact_path(
        result.get("source_policy_observation_visual_qa_path")
        or postprocess.get("source_policy_observation_visual_qa")
    )
    contact_sheet_path = _resolve_artifact_path(
        result.get("wam_rollout_contact_sheet_path")
        or postprocess.get("wam_rollout_contact_sheet")
    )
    frame_stats_path = _resolve_artifact_path(postprocess.get("wam_rollout_frame_stats"))
    review_video_path = _resolve_artifact_path(
        result.get("review_video_path") or postprocess.get("review_video_path")
    )
    visual_report = _read_json_if_present(visual_report_path)
    video_status = _read_json_if_present(video_status_path)
    source_qa = _read_json_if_present(source_qa_path)
    paid_provider = _paid_provider_status(provider=provider, result=result)
    source_qa_status = _string(source_qa.get("status")) or _string(
        visual_report.get("source_policy_observation_visual_qa_status")
    )
    blockers = _manifest_blockers(
        result=result,
        visual_report=visual_report,
        video_status=video_status,
        source_qa_status=source_qa_status,
        paid_provider=paid_provider,
    )
    generated_transition_count = int(result.get("generated_next_observation_count") or 0)
    if generated_transition_count < transition_count:
        blockers.append("short_visual_sanity_transition_count_not_completed")
    passed = not blockers
    provider_success = result.get("status") == "completed"
    visually_useful_rollout = visual_report.get("visual_success") is True
    manifest = {
        "schema_version": PERSISTENT_WAM_SHORT_VISUAL_SANITY_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "passed_short_visual_sanity" if passed else "blocked",
        "short_visual_sanity_passed": passed,
        "policy_observation_path": str(policy_observation_path),
        "provider": provider,
        "requested_transition_count": transition_count,
        "requested_loop_step_count": transition_count + 1,
        "generated_transition_count": generated_transition_count,
        "visual_profile": "review_quality",
        "review_quality_settings": dict(review_quality_settings),
        "capture_truth": False,
        "geometry_truth": False,
        "collision_truth": False,
        "provider_success": provider_success,
        "provider_success_separate_from_visually_useful_rollout": True,
        "visually_useful_rollout": visually_useful_rollout,
        "persistent_session_result_path": str(
            Path(str(result.get("job_dir"))) / "unitree_groot_n17_sonic_vast_persistent_session_result.json"
        )
        if result.get("job_dir")
        else None,
        "manifest_source": _string(result.get("_short_visual_sanity_manifest_source"))
        or "live_persistent_session_run",
        "source_policy_observation_visual_qa_status": source_qa_status,
        "source_policy_observation_visual_qa_path": source_qa_path,
        "wam_rollout_visual_success": visual_report.get("visual_success") is True,
        "wam_rollout_visual_quality_report_path": visual_report_path,
        "wam_rollout_contact_sheet_path": contact_sheet_path,
        "wam_rollout_frame_stats_path": frame_stats_path,
        "video_review_status_path": video_status_path,
        "review_video_path": review_video_path,
        "ffprobe_command_ran": bool(video_status.get("ffprobe_command_ran")),
        "ffprobe_returncode": video_status.get("ffprobe_returncode"),
        "ffprobe_metadata": _mapping(video_status.get("ffprobe_metadata")),
        "review_media_resolution": _review_media_resolution(video_status),
        "live_wam_generation_success_count": int(
            result.get("live_wam_generation_success_count") or 0
        ),
        "learned_wam_model_success_count": int(result.get("learned_wam_model_success_count") or 0),
        "structural_fallback_used": bool(visual_report.get("structural_fallback_used")),
        "paid_provider": paid_provider,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "short_visual_sanity_is_not_task_success_proof": True,
            "capture_truth": False,
            "geometry_truth": False,
            "collision_truth": False,
            "valid_mp4_or_provider_completed_is_not_visual_success": True,
            "provider_success": provider_success,
            "provider_success_separate_from_visually_useful_rollout": True,
            "visually_useful_rollout": visually_useful_rollout,
            "source_observation_qa_is_required_before_provider_run": True,
            "physical_robot_readiness_proven": False,
            "deployment_readiness_proven": False,
            "safety_validation_proven": False,
            "real_world_manipulation_success_proven": False,
        },
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    if result.get("_persistent_session_result_path"):
        manifest["persistent_session_result_path"] = str(result["_persistent_session_result_path"])
    job_dir = Path(str(result.get("job_dir"))).expanduser() if result.get("job_dir") else None
    output_dir = manifest_output_dir or job_dir or policy_observation_path.parent
    ensure_dir(output_dir)
    manifest_path = output_dir / SHORT_SANITY_FILENAME
    manifest["short_visual_sanity_manifest_path"] = str(manifest_path)
    write_json(manifest_path, manifest)
    validation = validate_persistent_wam_short_visual_sanity_manifest(
        manifest_path,
        policy_observation_path=policy_observation_path,
    )
    if validation.get("status") != "passed_short_visual_sanity":
        manifest["status"] = "blocked"
        manifest["short_visual_sanity_passed"] = False
        manifest["blockers"] = sorted(set(manifest["blockers"] + validation["blockers"]))
        write_json(manifest_path, manifest)
    return manifest


def run_short_visual_sanity(
    *,
    policy_observation_path: str | Path,
    job_dir: str | Path | None = None,
    provider: str = "runpod",
    transition_count: int = 2,
    task_prompt: str | None = None,
    timeout_seconds: float = 3600.0,
    max_wait_seconds: int | None = None,
    use_live_wam: bool = True,
    allow_structural_wam_fallback: bool = False,
    persistent_session_result_path: str | Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Run a 1-2 transition review-quality visual sanity pass."""
    if provider not in {"runpod", "vast"}:
        raise ValueError(f"unsupported_provider:{provider}")
    if transition_count not in {1, 2}:
        raise ValueError("transition_count_must_be_1_or_2")
    generated_at = utc_now_iso()
    root = Path(job_dir).expanduser().resolve() if job_dir else Path.cwd() / "persistent_wam_short_visual_sanity"
    ensure_dir(root)
    observation_path = Path(policy_observation_path).expanduser().resolve()
    with _temporary_env(
        {
            **REVIEW_QUALITY_ENV,
            "BLUEPRINT_OSCAR_WAM_NUM_STEPS": str(transition_count),
        }
    ):
        observation = _load_policy_observation(observation_path)
        frame_path = _camera_frame_path(observation)
        settings = _current_wam_visual_profile_settings()
        source_qa = assess_source_policy_observation_visual_qa(
            frame_path,
            generated_at=generated_at,
            target_object_id=_string(observation.get("target_object_id")) or None,
            task_id=_string(observation.get("task_id")) or None,
            visual_profile="review_quality",
            review_quality_required=True,
        )
        if source_qa.get("status") != "passed_visual_quality_gate":
            manifest = _write_blocked_preflight_manifest(
                output_dir=root,
                generated_at=generated_at,
                policy_observation_path=observation_path,
                provider=provider,
                transition_count=transition_count,
                source_qa=source_qa,
                review_quality_settings=settings,
            )
            return manifest, 2
        if persistent_session_result_path is not None:
            source_qa_path = root / "source_policy_observation_visual_qa.json"
            write_json(source_qa_path, source_qa)
            existing_result_path = Path(persistent_session_result_path).expanduser().resolve()
            result = _read_json(existing_result_path)
            result = dict(result)
            postprocess = _mapping(result.get("postprocess_artifacts"))
            postprocess["source_policy_observation_visual_qa"] = str(source_qa_path)
            result["postprocess_artifacts"] = postprocess
            result["source_policy_observation_visual_qa_path"] = str(source_qa_path)
            result["_persistent_session_result_path"] = str(existing_result_path)
            result["_short_visual_sanity_manifest_source"] = "imported_persistent_session_result"
            manifest = _build_short_sanity_manifest(
                generated_at=generated_at,
                policy_observation_path=observation_path,
                provider=provider,
                transition_count=transition_count,
                result=result,
                review_quality_settings=settings,
                manifest_output_dir=root,
            )
            return manifest, 0 if manifest.get("short_visual_sanity_passed") else 2
        runner = run_persistent_session_runpod if provider == "runpod" else run_persistent_session
        if provider == "runpod":
            result, exit_code = runner(
                policy_observation_path=observation_path,
                job_dir=root,
                loop_step_count=transition_count + 1,
                task_prompt=task_prompt,
                timeout_seconds=timeout_seconds,
                use_live_wam=use_live_wam,
                allow_structural_wam_fallback=allow_structural_wam_fallback,
                max_wait_seconds=max_wait_seconds,
            )
        else:
            result, exit_code = runner(
                policy_observation_path=observation_path,
                job_dir=root,
                loop_step_count=transition_count + 1,
                task_prompt=task_prompt,
                timeout_seconds=timeout_seconds,
                use_live_wam=use_live_wam,
                allow_structural_wam_fallback=allow_structural_wam_fallback,
            )
        manifest = _build_short_sanity_manifest(
            generated_at=generated_at,
            policy_observation_path=observation_path,
            provider=provider,
            transition_count=transition_count,
            result=result,
            review_quality_settings=settings,
        )
        return manifest, 0 if manifest.get("short_visual_sanity_passed") else (exit_code or 2)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-observation", required=True)
    parser.add_argument("--job-dir")
    parser.add_argument("--provider", choices=("runpod", "vast"), default="runpod")
    parser.add_argument("--transition-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--task-prompt")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--max-wait-seconds", type=int)
    parser.add_argument("--disable-live-wam", action="store_true")
    parser.add_argument("--allow-structural-wam-fallback", action="store_true")
    parser.add_argument(
        "--persistent-session-result",
        help=(
            "Build the short-sanity manifest from an existing persistent-session result "
            "instead of launching a provider; source observation QA is still re-run."
        ),
    )
    args = parser.parse_args(argv)
    manifest, exit_code = run_short_visual_sanity(
        policy_observation_path=args.policy_observation,
        job_dir=args.job_dir,
        provider=args.provider,
        transition_count=args.transition_count,
        task_prompt=args.task_prompt,
        timeout_seconds=args.timeout_seconds,
        max_wait_seconds=args.max_wait_seconds,
        use_live_wam=not args.disable_live_wam,
        allow_structural_wam_fallback=args.allow_structural_wam_fallback,
        persistent_session_result_path=args.persistent_session_result,
    )
    print(json.dumps(manifest, sort_keys=True))
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
