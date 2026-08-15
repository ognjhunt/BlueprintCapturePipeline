"""Run an authorized SAM 3.1 tracker into Blueprint's source-track contract.

This module deliberately stops at source-bound 2D mask tracks.  It does not
lift masks into 3D, infer metric geometry, grade a task, or qualify its own
evidence.  The heavy SAM runtime is injected so the contract remains testable
without torch, gated checkpoints, network access, or paid compute.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from .semantic_gaussian_lifting import canonical_json_digest
from .semantic_source_track_import import (
    MASK_ENCODING,
    PROVIDER_RESULT_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION as IMPORT_REQUEST_SCHEMA_VERSION,
)


RUN_REQUEST_SCHEMA_VERSION = "semantic_sam31_source_track_run_request.v1"
RUN_RESULT_SCHEMA_VERSION = "semantic_sam31_source_track_run_result.v1"
RUNTIME_API = "meta_sam3_object_multiplex_handle_request.v1"
CHECKPOINT_FAMILY = "facebook/sam3.1"
FRAME_INPUT_MODE = "ordered_hash_bound_jpeg_derivatives.v1"
# The released SAM 3.1 multiplex checkpoint is trained with sixteen slots.
# Upstream uses this value to size checkpoint-bound embeddings and memory
# convolutions, so it is a runtime architecture identity rather than a tunable
# per-request capacity.  ``max_num_objects`` remains the semantic output cap.
CHECKPOINT_MULTIPLEX_COUNT = 16

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MAX_FRAMES = 100_000
_MAX_PROMPTS = 64
_MAX_TRACKS = 10_000
_MAX_MASK_PIXELS = 100_000_000
_MAX_OBSERVATIONS = 1_000_000
_MAX_RUNS_PER_OBSERVATION = 1_000_000
_SAFE_EXECUTION_BLOCKERS = {
    "sam31_checkpoint_digest_mismatch",
    "sam31_checkpoint_missing_or_unsafe",
    "sam31_duplicate_frame_outputs_disagree",
    "sam31_installed_code_revision_mismatch",
    "sam31_installed_runtime_digest_mismatch",
    "sam31_mask_runs_exceed_limit",
    "sam31_observations_exceed_limit",
    "sam31_output_frame_dimensions_mismatch",
    "sam31_output_not_array:out_binary_masks",
    "sam31_output_not_array:out_obj_ids",
    "sam31_output_not_array:out_probs",
    "sam31_output_object_id_invalid",
    "sam31_output_score_invalid",
    "sam31_output_shape_mismatch",
    "sam31_outputs_missing_retained_frames",
    "sam31_runtime_not_installed",
    "sam31_session_start_invalid",
    "sam31_stream_frame_invalid",
    "sam31_stream_response_invalid",
    "sam31_tracks_exceed_limit",
}


PredictorFactory = Callable[[Mapping[str, Any]], Any]


def _valid_digest(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _same_digest(left: Any, right: Any) -> bool:
    return str(left or "").strip().lower() == str(right or "").strip().lower()


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _identifier(value: Any) -> str:
    text = str(value or "").strip()
    return text if _IDENTIFIER.fullmatch(text) else ""


def _terminal_result(
    request: Mapping[str, Any],
    *,
    status: str,
    blockers: Sequence[str],
    warnings: Sequence[str] = (),
    provider_result: Mapping[str, Any] | None = None,
    import_request: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    bindings = request.get("bindings")
    result: Dict[str, Any] = {
        "schema_version": RUN_RESULT_SCHEMA_VERSION,
        "status": status,
        "bindings": dict(bindings) if isinstance(bindings, Mapping) else {},
        "provider_result": dict(provider_result) if provider_result is not None else None,
        "source_track_import_request": (
            dict(import_request) if import_request is not None else None
        ),
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "claim_ceiling": (
            "source_bound_2d_binary_mask_tracks_only"
            if status in {"completed", "abstained"}
            else "none_sam31_execution_not_admitted_or_failed"
        ),
        "directly_observed_object_fact": False,
        "canonical_object_geometry": False,
        "metric_box_ready": False,
        "collision_ready": False,
        "physics_ready": False,
        "physical_task_success_established": False,
        "model_self_grading_permitted": False,
        "comparative_policy_ranking_verdict": "thesis_not_supported",
    }
    result["result_digest"] = canonical_json_digest(result)
    return result


def blocked_sam31_source_track_run(
    request: Mapping[str, Any], blockers: Sequence[str]
) -> Dict[str, Any]:
    """Return a deterministic fail-closed terminal result."""

    return _terminal_result(request, status="blocked", blockers=blockers)


def _validate_profile(request: Mapping[str, Any], blockers: list[str]) -> Dict[str, Any]:
    raw = request.get("provider_profile")
    if not isinstance(raw, Mapping):
        blockers.append("provider_profile_missing")
        return {}
    profile = dict(raw)
    supplied = profile.get("profile_digest")
    try:
        computed = canonical_json_digest(
            {key: value for key, value in profile.items() if key != "profile_digest"}
        )
    except (TypeError, ValueError):
        computed = ""
    if not _valid_digest(supplied) or supplied != computed:
        blockers.append("provider_profile_digest_mismatch")
    exact_values = {
        "method_id": "meta.sam3.1.object_multiplex",
        "runtime_api": RUNTIME_API,
        "checkpoint_family": CHECKPOINT_FAMILY,
        "frame_input_mode": FRAME_INPUT_MODE,
        "mask_encoding": MASK_ENCODING,
        "execution_mode": "local",
    }
    for field, expected in exact_values.items():
        if profile.get(field) != expected:
            blockers.append(f"provider_profile_{field}_mismatch")
    if not str(profile.get("method_version") or "").strip():
        blockers.append("provider_profile_method_version_missing")
    revision = str(profile.get("official_code_revision") or "").strip().lower()
    if not _GIT_REVISION.fullmatch(revision):
        blockers.append("provider_profile_official_code_revision_invalid")
    for field in (
        "runtime_digest",
        "model_digest",
        "checkpoint_digest",
        "license_terms_digest",
        "license_use_authorization_digest",
        "privacy_use_authorization_digest",
        "trade_controls_review_digest",
        "execution_authorization_digest",
    ):
        if not _valid_digest(profile.get(field)):
            blockers.append(f"provider_profile_{field}_invalid")
    if not _same_digest(profile.get("model_digest"), profile.get("checkpoint_digest")):
        blockers.append("provider_profile_model_checkpoint_digest_mismatch")
    required_true = (
        "checkpoint_access_authorized",
        "commercial_evidence_use_authorized",
        "persistent_track_ids",
        "model_self_grading_forbidden",
        "source_frames_are_hash_verified",
        "network_access_during_inference_forbidden",
    )
    for field in required_true:
        if profile.get(field) is not True:
            blockers.append(f"provider_profile_{field}_required")
    if profile.get("customer_data_training_allowed") is not False:
        blockers.append("provider_profile_customer_training_must_be_false")
    threshold = _finite(profile.get("output_probability_threshold"))
    if threshold is None or not 0.0 < threshold <= 1.0:
        blockers.append("provider_profile_output_probability_threshold_invalid")
    for field in ("max_num_objects", "multiplex_count"):
        value = _positive_int(profile.get(field))
        if value is None or value > 10_000:
            blockers.append(f"provider_profile_{field}_invalid")
    if profile.get("multiplex_count") != CHECKPOINT_MULTIPLEX_COUNT:
        blockers.append("provider_profile_multiplex_count_checkpoint_mismatch")
    for field in ("use_fa3", "compile", "warm_up", "async_loading_frames"):
        if not isinstance(profile.get(field), bool):
            blockers.append(f"provider_profile_{field}_declaration_missing")
    if profile.get("warm_up") is True and profile.get("compile") is not True:
        blockers.append("provider_profile_warm_up_requires_compile")
    allowed_uses = request.get("allowed_evidence_uses")
    if not isinstance(allowed_uses, list) or "semantic_analysis" not in allowed_uses:
        blockers.append("semantic_analysis_use_not_permitted")
    return profile


def _validate_frames(request: Mapping[str, Any], blockers: list[str]) -> list[dict[str, Any]]:
    raw_bindings = request.get("bindings")
    bindings: Mapping[str, Any] = raw_bindings if isinstance(raw_bindings, Mapping) else {}
    raw = request.get("frame_registry")
    if not isinstance(raw, list) or not raw or len(raw) > _MAX_FRAMES:
        blockers.append("frame_registry_missing_empty_or_too_large")
        return []
    try:
        digest = canonical_json_digest(raw)
    except (TypeError, ValueError):
        blockers.append("frame_registry_not_canonical_json")
        return []
    if not _same_digest(bindings.get("frame_registry_digest"), digest):
        blockers.append("frame_registry_digest_mismatch")
    frames: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    previous_pts = -1.0
    for expected_index, row in enumerate(raw):
        if not isinstance(row, Mapping):
            blockers.append("frame_registry_row_invalid")
            continue
        frame = dict(row)
        frame_id = _identifier(frame.get("source_frame_id"))
        model_index = frame.get("model_frame_index")
        if (
            not frame_id
            or frame_id in seen_ids
            or isinstance(model_index, bool)
            or not isinstance(model_index, int)
            or model_index != expected_index
            or model_index in seen_indices
        ):
            blockers.append("frame_registry_identity_or_order_invalid")
            continue
        seen_ids.add(frame_id)
        seen_indices.add(model_index)
        for field in (
            "source_frame_digest",
            "retained_video_digest",
            "sync_map_row_digest",
            "camera_record_digest",
            "analysis_jpeg_digest",
        ):
            if not _valid_digest(frame.get(field)):
                blockers.append(f"frame_registry_digest_invalid:{frame_id}:{field}")
        if not _same_digest(
            frame.get("retained_video_digest"), bindings.get("retained_video_digest")
        ):
            blockers.append(f"frame_registry_retained_video_mismatch:{frame_id}")
        pts = _finite(frame.get("decoded_pts_seconds"))
        if pts is None or pts < 0.0 or pts <= previous_pts:
            blockers.append(f"frame_registry_pts_not_strictly_increasing:{frame_id}")
        else:
            previous_pts = pts
        if frame.get("encoder_retained") is not True:
            blockers.append(f"frame_registry_encoder_retention_not_proven:{frame_id}")
        width = _positive_int(frame.get("width"))
        height = _positive_int(frame.get("height"))
        if width is None or height is None or width * height > _MAX_MASK_PIXELS:
            blockers.append(f"frame_registry_dimensions_invalid:{frame_id}")
        frames.append(frame)
    return frames


def _validate_prompts(request: Mapping[str, Any], blockers: list[str]) -> list[dict[str, str]]:
    raw = request.get("prompts")
    if not isinstance(raw, list) or not raw or len(raw) > _MAX_PROMPTS:
        blockers.append("prompts_missing_empty_or_too_large")
        return []
    prompts: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in raw:
        if not isinstance(row, Mapping):
            blockers.append("prompt_invalid")
            continue
        prompt_id = _identifier(row.get("prompt_id"))
        text = str(row.get("text") or "").strip()
        label = str(row.get("output_label") or "").strip()
        if (
            not prompt_id
            or prompt_id in seen
            or not text
            or len(text) > 256
            or not label
            or len(label) > 256
        ):
            blockers.append("prompt_identity_or_text_invalid")
            continue
        seen.add(prompt_id)
        prompts.append({"prompt_id": prompt_id, "text": text, "output_label": label})
    return prompts


def _validate_request(
    request: Mapping[str, Any], materialized_frame_directory: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, str]], list[str]]:
    blockers: list[str] = []
    if request.get("schema_version") != RUN_REQUEST_SCHEMA_VERSION:
        blockers.append("request_schema_version_mismatch")
    bindings = request.get("bindings")
    if not isinstance(bindings, Mapping):
        blockers.append("bindings_missing")
        bindings = {}
    for field in (
        "capture_digest",
        "retained_video_digest",
        "camera_solution_digest",
        "frame_registry_digest",
    ):
        if not _valid_digest(bindings.get(field)):
            blockers.append(f"binding_digest_invalid:{field}")
    profile = _validate_profile(request, blockers)
    frames = _validate_frames(request, blockers)
    prompts = _validate_prompts(request, blockers)
    if not materialized_frame_directory.is_dir():
        blockers.append("materialized_frame_directory_missing")
    else:
        names = sorted(path.name for path in materialized_frame_directory.iterdir())
        expected = [f"{index:06d}.jpg" for index in range(len(frames))]
        if names != expected:
            blockers.append("materialized_frame_directory_members_mismatch")
    return profile, frames, prompts, blockers


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"sam31_output_not_array:{name}") from exc


def _mask_runs(mask: np.ndarray, probability: float) -> list[dict[str, Any]]:
    flat = np.asarray(mask, dtype=bool).reshape(-1)
    indices = np.flatnonzero(flat)
    if indices.size == 0:
        return []
    starts = [int(indices[0])]
    lengths: list[int] = []
    previous = int(indices[0])
    current_length = 1
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index == previous + 1:
            current_length += 1
        else:
            lengths.append(current_length)
            starts.append(index)
            current_length = 1
        previous = index
    lengths.append(current_length)
    if len(starts) > _MAX_RUNS_PER_OBSERVATION:
        raise ValueError("sam31_mask_runs_exceed_limit")
    return [
        {"start": start, "length": length, "probability": probability}
        for start, length in zip(starts, lengths)
    ]


def _collect_frame_output(
    *,
    frame: Mapping[str, Any],
    prompt: Mapping[str, str],
    outputs: Mapping[str, Any],
    threshold: float,
    observations: dict[tuple[str, int], list[dict[str, Any]]],
) -> None:
    object_ids = _as_numpy(outputs.get("out_obj_ids"), name="out_obj_ids").reshape(-1)
    scores = _as_numpy(outputs.get("out_probs"), name="out_probs").reshape(-1)
    masks = _as_numpy(outputs.get("out_binary_masks"), name="out_binary_masks")
    if (
        masks.ndim != 3
        or len(object_ids) != len(scores)
        or len(object_ids) != masks.shape[0]
        or len(object_ids) > _MAX_TRACKS
    ):
        raise ValueError("sam31_output_shape_mismatch")
    height = int(frame["height"])
    width = int(frame["width"])
    if tuple(masks.shape[1:]) != (height, width):
        raise ValueError("sam31_output_frame_dimensions_mismatch")
    for index, raw_object_id in enumerate(object_ids.tolist()):
        if isinstance(raw_object_id, bool) or not isinstance(raw_object_id, (int, np.integer)):
            raise ValueError("sam31_output_object_id_invalid")
        object_id = int(raw_object_id)
        if object_id < 0:
            raise ValueError("sam31_output_object_id_invalid")
        score = _finite(float(scores[index]))
        if score is None or not 0.0 <= score <= 1.0:
            raise ValueError("sam31_output_score_invalid")
        if score < threshold:
            continue
        runs = _mask_runs(masks[index], score)
        if not runs:
            continue
        observation = {
            "source_frame_id": frame["source_frame_id"],
            "source_frame_digest": frame["source_frame_digest"],
            "decoded_pts_seconds": frame["decoded_pts_seconds"],
            "camera_record_digest": frame["camera_record_digest"],
            "width": width,
            "height": height,
            "mask_encoding": MASK_ENCODING,
            "runs": runs,
        }
        observations[(prompt["prompt_id"], object_id)].append(observation)


def _run_prompt(
    *,
    predictor: Any,
    session_id: str,
    prompt: Mapping[str, str],
    frames: Sequence[Mapping[str, Any]],
    threshold: float,
    observations: dict[tuple[str, int], list[dict[str, Any]]],
) -> None:
    predictor.handle_request(request={"type": "reset_session", "session_id": session_id})
    initial = predictor.handle_request(
        request={
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": prompt["text"],
            "output_prob_thresh": threshold,
        }
    )
    by_frame: dict[int, Mapping[str, Any]] = {}
    if (
        isinstance(initial, Mapping)
        and initial.get("frame_index") == 0
        and isinstance(initial.get("outputs"), Mapping)
    ):
        by_frame[0] = initial["outputs"]
    stream_seen: set[int] = set()
    stream = predictor.handle_stream_request(
        request={
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": "forward",
            "start_frame_index": 0,
            "max_frame_num_to_track": len(frames),
            "output_prob_thresh": threshold,
        }
    )
    for response in stream:
        if not isinstance(response, Mapping):
            raise ValueError("sam31_stream_response_invalid")
        frame_index = response.get("frame_index")
        outputs = response.get("outputs")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
            or frame_index >= len(frames)
            or not isinstance(outputs, Mapping)
        ):
            raise ValueError("sam31_stream_frame_invalid")
        if frame_index in stream_seen:
            first = by_frame[frame_index]
            if not _outputs_equal(first, outputs):
                raise ValueError("sam31_duplicate_frame_outputs_disagree")
        stream_seen.add(frame_index)
        by_frame[frame_index] = outputs
    if sorted(by_frame) != list(range(len(frames))):
        raise ValueError("sam31_outputs_missing_retained_frames")
    for frame_index, frame in enumerate(frames):
        _collect_frame_output(
            frame=frame,
            prompt=prompt,
            outputs=by_frame[frame_index],
            threshold=threshold,
            observations=observations,
        )


def _outputs_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    for key in ("out_obj_ids", "out_probs", "out_binary_masks"):
        left_value = _as_numpy(left.get(key), name=key)
        right_value = _as_numpy(right.get(key), name=key)
        if left_value.shape != right_value.shape or not np.array_equal(left_value, right_value):
            return False
    return True


def _provider_result(
    request: Mapping[str, Any],
    profile: Mapping[str, Any],
    prompts: Sequence[Mapping[str, str]],
    observations: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    prompt_by_id = {row["prompt_id"]: row for row in prompts}
    tracks: list[dict[str, Any]] = []
    observation_count = 0
    for (prompt_id, object_id), raw_observations in sorted(observations.items()):
        ordered = sorted(
            raw_observations,
            key=lambda row: int(
                next(
                    frame["model_frame_index"]
                    for frame in request["frame_registry"]
                    if frame["source_frame_id"] == row["source_frame_id"]
                )
            ),
        )
        observation_count += len(ordered)
        if observation_count > _MAX_OBSERVATIONS:
            raise ValueError("sam31_observations_exceed_limit")
        tracks.append(
            {
                "track_id": f"sam31-{prompt_id}-{object_id}",
                "label": prompt_by_id[prompt_id]["output_label"],
                "label_source": "model_inferred",
                "observations": [dict(row) for row in ordered],
            }
        )
    if len(tracks) > _MAX_TRACKS:
        raise ValueError("sam31_tracks_exceed_limit")
    bindings = dict(request["bindings"])
    result: Dict[str, Any] = {
        "schema_version": PROVIDER_RESULT_SCHEMA_VERSION,
        "bindings": bindings,
        "profile_digest": profile["profile_digest"],
        "model_digest": profile["model_digest"],
        "runtime_digest": profile["runtime_digest"],
        "tracks": tracks,
        "provider_metadata": {
            "runtime_api": RUNTIME_API,
            "checkpoint_family": CHECKPOINT_FAMILY,
            "official_code_revision": profile["official_code_revision"],
            "frame_input_mode": FRAME_INPUT_MODE,
            "mask_support": "thresholded_binary_mask",
            "run_probability": "object_detection_score_on_binary_support",
            "prompt_ids": [row["prompt_id"] for row in prompts],
            "cross_prompt_instance_deduplication_performed": False,
            "source_frame_semantics_only": True,
            "model_self_grading_permitted": False,
            "network_access_during_inference": False,
        },
    }
    result["result_digest"] = canonical_json_digest(result)
    return result


def _import_request(
    request: Mapping[str, Any],
    profile: Mapping[str, Any],
    provider_result: Mapping[str, Any],
) -> Dict[str, Any]:
    bindings = {
        **dict(request["bindings"]),
        "provider_result_digest": provider_result["result_digest"],
    }
    return {
        "schema_version": IMPORT_REQUEST_SCHEMA_VERSION,
        "bindings": bindings,
        "frame_registry": [dict(frame) for frame in request["frame_registry"]],
        "provider_profile": dict(profile),
        "allowed_evidence_uses": list(request["allowed_evidence_uses"]),
    }


def execute_sam31_source_track_request(
    request: Mapping[str, Any],
    *,
    predictor_factory: PredictorFactory,
    materialized_frame_directory: str | Path,
) -> Dict[str, Any]:
    """Execute Meta's pinned multiplex API and return normalized track inputs.

    The caller owns filesystem and checkpoint admission.  This function owns
    request validation, untrusted runtime-output validation, deterministic mask
    encoding, session cleanup, and the no-authority-upgrade result boundary.
    """

    frame_directory = Path(materialized_frame_directory)
    profile, frames, prompts, blockers = _validate_request(request, frame_directory)
    if blockers:
        return blocked_sam31_source_track_run(request, blockers)
    observations: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    predictor: Any = None
    session_id = ""
    terminal: Dict[str, Any]
    cleanup_failed = False
    try:
        predictor = predictor_factory(profile)
        started = predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": str(frame_directory),
                "offload_video_to_cpu": False,
                "offload_state_to_cpu": False,
            }
        )
        if not isinstance(started, Mapping) or not str(started.get("session_id") or ""):
            raise ValueError("sam31_session_start_invalid")
        session_id = str(started["session_id"])
        threshold = float(profile["output_probability_threshold"])
        for prompt in prompts:
            _run_prompt(
                predictor=predictor,
                session_id=session_id,
                prompt=prompt,
                frames=frames,
                threshold=threshold,
                observations=observations,
            )
        provider = _provider_result(request, profile, prompts, observations)
        import_request = _import_request(request, profile, provider)
        status = "completed" if provider["tracks"] else "abstained"
        warnings = [] if provider["tracks"] else ["sam31_returned_no_tracks"]
        terminal = _terminal_result(
            request,
            status=status,
            blockers=[],
            warnings=warnings,
            provider_result=provider,
            import_request=import_request,
        )
    except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
        reason = str(exc).strip()
        if reason not in _SAFE_EXECUTION_BLOCKERS:
            reason = f"sam31_runtime_failed:{type(exc).__name__}"
        terminal = blocked_sam31_source_track_run(request, [reason])
    finally:
        if predictor is not None and session_id:
            try:
                closed = predictor.handle_request(
                    request={
                        "type": "close_session",
                        "session_id": session_id,
                        "run_gc_collect": True,
                    }
                )
                if not isinstance(closed, Mapping) or closed.get("is_success") is not True:
                    cleanup_failed = True
            except Exception:
                cleanup_failed = True
    if cleanup_failed:
        return blocked_sam31_source_track_run(
            request,
            [*terminal.get("blockers", []), "sam31_session_cleanup_failed"],
        )
    return terminal


__all__ = [
    "CHECKPOINT_FAMILY",
    "CHECKPOINT_MULTIPLEX_COUNT",
    "FRAME_INPUT_MODE",
    "RUN_REQUEST_SCHEMA_VERSION",
    "RUN_RESULT_SCHEMA_VERSION",
    "RUNTIME_API",
    "blocked_sam31_source_track_run",
    "execute_sam31_source_track_request",
]
