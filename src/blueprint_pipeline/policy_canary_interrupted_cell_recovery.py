"""Seal interrupted-cell evidence without executing a policy or inventing outcomes.

Call only after the child has stopped. Existing evidence files are never edited;
offline callers must supply a scratch copy of immutable retained provider output.
"""
from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from .adp009d_droid_observation import CANDIDATE_VIEW_SHAPES, resize_with_pad
from .decision_evidence_contracts import canonical_digest
from .episode_visual_evidence import _encode_or_resume_episode_video
from .native_task_arena_policy_canary_session import (
    CANDIDATE_IDS, CLAIM_CEILING, PROVIDER_RESULT_FILENAME, RESULT_SCHEMA_VERSION,
    RUN_KIND, validate_runtime_input_manifest, validate_session_authority,
)
from .policy_request_evidence import restore_request, validate_request_evidence

SCHEMA_VERSION = "policy_canary_interrupted_cell_recovery.v1"
RECOVERY_DIRECTORY = "interrupted_cell_recovery"
FRAME_PATTERN = re.compile(r"^(\d{6})-(policy-input|review-sample|terminal-observation)\.png$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,511}$")


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"interrupted_cell_input_missing_or_symlink:{path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"interrupted_cell_input_not_mapping:{path.name}")
    return value


def _digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return "sha256:" + hasher.hexdigest()


def _inventory_digest(rows: list) -> str:
    return canonical_digest({"value": rows})


def _safe_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError("interrupted_cell_artifact_path_unsafe")
    path = root / value
    if any(part.is_symlink() for part in (path, *path.parents) if part != root.parent):
        raise ValueError("interrupted_cell_artifact_symlink_forbidden")
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError("interrupted_cell_artifact_missing_or_outside_root")
    return path


def _artifact(root: Path, path: Path, role: str) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    path = _safe_path(root, relative)
    return {"role": role, "relative_path": relative, "size_bytes": path.stat().st_size,
            "sha256": _digest_file(path),
            "media_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream"}


def _verify_artifact(root: Path, row: Mapping[str, Any]) -> None:
    path = _safe_path(root, str(row.get("relative_path") or ""))
    if path.stat().st_size != row.get("size_bytes") or _digest_file(path) != row.get("sha256"):
        raise ValueError("interrupted_cell_artifact_digest_mismatch")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise FileExistsError("interrupted_cell_overwrite_forbidden")
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError("interrupted_cell_overwrite_forbidden")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".sealing", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic publication without replacing any pre-existing evidence.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _frame(root: Path, path: Path) -> tuple[dict[str, Any], Any]:
    import numpy as np
    from PIL import Image

    match = FRAME_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError("interrupted_cell_frame_name_invalid")
    artifact = _artifact(root, path, "retained_lossless_frame")
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError("interrupted_cell_frame_not_lossless_rgb")
        pixels = np.asarray(image, dtype=np.uint8)
    return {**artifact, "frame_index": int(match[1]), "kind": match[2],
            "width": int(pixels.shape[1]), "height": int(pixels.shape[0]),
            "raw_rgb_sha256": "sha256:" + hashlib.sha256(pixels.tobytes()).hexdigest(),
            "native_timestamp_ns": None, "simulation_time_s": None}, pixels


def _wire_images(request: Mapping[str, Any], candidate: str) -> tuple[dict[str, Any], str]:
    import numpy as np

    if candidate == "pi05_droid":
        images = {role: request[f"observation/{key}"] for role, key in
                  (("external", "exterior_image_1_left"), ("wrist", "wrist_image_left"))}
        prompt = request.get("prompt")
    else:
        video = request.get("video") or {}
        images = {role: np.asarray(video[key])[0, 0] for role, key in
                  (("external", "exterior_image_1_left"), ("wrist", "wrist_image_left"))}
        if any(np.asarray(video[key]).shape[:2] != (1, 1) for key in video):
            raise ValueError("interrupted_cell_policy_video_axes_invalid")
        language = (request.get("language") or {}).get("annotation.language.language_instruction")
        if (not isinstance(language, list) or len(language) != 1
                or not isinstance(language[0], list) or len(language[0]) != 1):
            raise ValueError("interrupted_cell_policy_prompt_invalid")
        prompt = language[0][0]
    shape = (*CANDIDATE_VIEW_SHAPES[candidate], 3)
    if not isinstance(prompt, str) or any(
        np.asarray(image).dtype != np.uint8 or np.asarray(image).shape != shape
        for image in images.values()
    ):
        raise ValueError("interrupted_cell_policy_images_invalid")
    return images, prompt


def _load_binding(runtime_root: Path, selected_cell_index: int) -> tuple[dict, dict, dict, dict, dict]:
    # These are pure existing validators/resolution; import at call time so the
    # canary worker may call this helper without creating a module import cycle.
    from .native_task_arena_policy_canary_worker import _resolved_scene_plan, _validate_provider_manifest

    inputs = validate_runtime_input_manifest(_read(runtime_root / "runtime_inputs/policy_canary_runtime_inputs.json"))
    authority = validate_session_authority(_read(runtime_root / "runtime_inputs/policy_canary_session_authority.json"))
    if (type(selected_cell_index) is not int or not 0 <= selected_cell_index < len(inputs["cells"])
            or inputs["run_id"] != authority["run_id"]
            or inputs["runtime_inputs_digest"] != authority["runtime_inputs_digest"]
            or inputs["task_success_contract_digest"] != authority["task_success_contract_digest"]):
        raise ValueError("interrupted_cell_runtime_binding_mismatch")
    provider = _validate_provider_manifest(_read(runtime_root / "adp_arena_provider_manifest.json"),
        runtime_inputs=inputs, authority=authority)
    base = _read(runtime_root / "native_task_packet/native_task_arena_scene_plan.v1.json")
    if (base.get("plan_digest") != canonical_digest(base, digest_field="plan_digest")
            or provider.get("arena_scene_plan_digest") != base["plan_digest"]):
        raise ValueError("interrupted_cell_scene_plan_binding_mismatch")
    cell = inputs["cells"][selected_cell_index]
    plan = _resolved_scene_plan(base, cell, task_success_contract=inputs["task_success_contract"])
    specs = {}
    for candidate in CANDIDATE_IDS:
        spec = _read(runtime_root / f"runtime_inputs/policy_execution_spec.{candidate}.json")
        if (spec.get("candidate_id") != candidate
                or spec.get("execution_spec_digest") != canonical_digest(spec, digest_field="execution_spec_digest")
                or spec.get("execution_spec_digest") != (provider.get("execution_spec_digests") or {}).get(candidate)
                or spec.get("task_success_contract_digest") != inputs["task_success_contract_digest"]
                or any(not re.fullmatch(r"sha256:[0-9a-f]{64}", str(spec.get(key) or ""))
                       for key in ("checkpoint_digest", "runtime_identity_digest"))
                or type(spec.get("max_policy_queries")) is not int or spec["max_policy_queries"] < 1):
            raise ValueError("interrupted_cell_execution_spec_binding_mismatch")
        specs[candidate] = spec
    binding = {"run_id": inputs["run_id"], "cell_id": cell["cell_id"], "seed": cell["seed"],
               "cell_spec_digest": cell["cell_spec_digest"], "resolved_scenario_digest": cell["resolved_scenario_digest"],
               "task_spec_digest": canonical_digest(plan["task_spec"]),
               "task_success_contract_digest": inputs["task_success_contract_digest"],
               "runtime_inputs_digest": inputs["runtime_inputs_digest"], "authority_digest": authority["authority_digest"],
               "provider_manifest_digest": provider["input_digest"], "implementation_commit": provider.get("implementation_commit")}
    return inputs, authority, cell, specs, binding


def _request_records(root: Path, episode_id: str, candidate: str, spec: dict, binding: dict) -> tuple[list, dict]:
    records, images = [], {}
    directory = root / "episodes/media" / episode_id / "policy-requests"
    paths = sorted(directory.glob("*.json"))
    for index, path in enumerate(paths):
        if path.name != f"{index:06d}.json" or index >= int(spec["max_policy_queries"]):
            raise ValueError("interrupted_cell_request_sequence_invalid")
        record = validate_request_evidence(_read(path))
        expected = {key: binding[key] for key in ("cell_id", "seed", "resolved_scenario_digest", "task_spec_digest")}
        expected.update(candidate_id=candidate, episode_id=episode_id, query_index=index)
        if record.get("episode_binding") != expected or record.get("serialization_verified") is not True:
            raise ValueError("interrupted_cell_request_binding_mismatch")
        views, prompt = _wire_images(restore_request(record["request"]), candidate)
        if prompt != (str(spec["prompt"]).strip() if candidate == "groot_n17_droid" else spec["prompt"]):
            raise ValueError("interrupted_cell_request_prompt_mismatch")
        images[index] = views
        records.append({**_artifact(root, path, "exact_policy_request"), "query_index": index,
                        "request_digest": record["request_digest"], "evidence_digest": record["evidence_digest"]})
    return records, images


def _recover_frames(root: Path, episode_id: str, candidate: str, request_images: dict) -> tuple[list, dict, list]:
    import numpy as np

    directory = root / "episodes/media" / episode_id / "frames"
    composites, streams, gaps = [], {}, []
    for path in sorted(directory.glob("*.png")):
        row, pixels = _frame(root, path)
        if row["kind"] != "policy-input":
            raise ValueError("interrupted_cell_composite_kind_invalid")
        index = row["frame_index"]
        if index != len(composites):
            raise ValueError("interrupted_cell_composite_sequence_invalid")
        views = request_images.get(index)
        if views is not None:
            if not np.array_equal(pixels, np.concatenate([views["external"], views["wrist"]], axis=1)):
                raise ValueError("interrupted_cell_policy_pixels_mismatch")
            row.update(policy_request_index=index, exact_wire_pixels_verified=True)
        else:
            row.update(policy_request_index=None, exact_wire_pixels_verified=False)
            gaps.append("retained_input_without_request_receipt")
        composites.append(row)
    if any(index >= len(composites) for index in request_images):
        raise ValueError("interrupted_cell_request_lossless_input_missing")
    for camera in ("external", "wrist", "overview"):
        frames = []
        paths = sorted((directory / camera).glob("*.png"))
        policy_paths = [path for path in paths if path.name.endswith("-policy-input.png")]
        aligned = len(policy_paths) == len(composites)
        if composites and not aligned:
            gaps.append(f"{camera}_policy_input_query_alignment_unproven")
        policy_index = 0
        indices = set()
        for path in paths:
            row, pixels = _frame(root, path)
            if row["frame_index"] in indices:
                raise ValueError("interrupted_cell_camera_sequence_ambiguous")
            indices.add(row["frame_index"])
            row.update(camera_id=camera, policy_visible=False, policy_request_index=None)
            if row["kind"] == "policy-input":
                views = request_images.get(policy_index) if aligned else None
                if views is not None and camera in views:
                    height, width = CANDIDATE_VIEW_SHAPES[candidate]
                    if not np.array_equal(resize_with_pad(pixels, height=height, width=width), views[camera]):
                        raise ValueError("interrupted_cell_raw_camera_policy_pixels_mismatch")
                    row.update(policy_request_index=policy_index, source_of_verified_policy_input=True)
                policy_index += 1
            frames.append(row)
        if frames:
            streams[camera] = frames
            if indices != set(range(max(indices) + 1)):
                gaps.append(f"{camera}_retained_sequence_gap")
    if composites:
        streams["policy_input_composite"] = composites
    return composites, streams, sorted(set(gaps))


def _original_failure(root: Path, episode_id: str, candidate: str, binding: dict, spec: dict) -> tuple[dict | None, dict | None]:
    paths = [path for suffix in ("failure_evidence", "failure_gap")
             if (path := root / "episodes" / f"{episode_id}.{suffix}.json").exists()]
    if len(paths) > 1:
        raise ValueError("interrupted_cell_original_failure_ambiguous")
    if not paths:
        return None, None
    path = paths[0]
    value = _read(path)
    if (value.get("schema_version") != "policy_canary_episode_failure_evidence.v2"
            or value.get("gap_digest") != canonical_digest(value, digest_field="gap_digest")
            or value.get("candidate_id") != candidate or value.get("cell_id") != binding["cell_id"]
            or value.get("seed") != binding["seed"] or value.get("run_kind") != RUN_KIND
            or value.get("claim_ceiling") != CLAIM_CEILING or value.get("status") != "blocked"
            or (value.get("episode") or {}).get("episode_id") != episode_id):
        raise ValueError("interrupted_cell_original_failure_binding_mismatch")
    for key in ("checkpoint_digest", "runtime_identity_digest", "execution_spec_digest", "task_success_contract_digest"):
        if value.get(key) is not None and value[key] != spec.get(key):
            raise ValueError("interrupted_cell_original_failure_spec_mismatch")
    for row in (value.get("evidence_artifacts") or {}).values():
        if isinstance(row, Mapping):
            _verify_artifact(root, row)
    for row in (value.get("episode") or {}).get("media_artifacts") or []:
        _verify_artifact(root / "episodes", row)
    reset = value.get("scientific_reset")
    if isinstance(reset, Mapping):
        expected = {key: binding[key] for key in ("cell_id", "seed", "resolved_scenario_digest", "task_spec_digest")}
        expected["candidate_id"] = candidate
        if (reset.get("binding") != expected or reset.get("schema_version") != "policy_scientific_reset.v1"
                or reset.get("receipt_digest") != canonical_digest(reset, digest_field="receipt_digest")):
            raise ValueError("interrupted_cell_original_reset_binding_mismatch")
    return value, _artifact(root, path, "original_episode_failure_receipt")


def _source_inventory(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts[0] == RECOVERY_DIRECTORY or path.name in {PROVIDER_RESULT_FILENAME, "worker_console.log"}:
            continue
        if path.is_symlink():
            raise ValueError("interrupted_cell_artifact_symlink_forbidden")
        if not path.is_file():
            continue
        role = ("exact_policy_request" if "policy-requests" in relative.parts else
                "lossless_frame_manifest" if "manifest" in path.name and "frame" in path.name else
                "review_video" if path.suffix == ".mp4" else "retained_lossless_frame" if path.suffix == ".png"
                else "retained_interrupted_cell_evidence")
        records.append(_artifact(root, path, role))
    return records


def recover_interrupted_cell_result(
    *, runtime_root: Path, child_root: Path, selected_cell_index: int, reason: str,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Add a blocked child result from retained bytes; never resume policy execution."""

    if Path(child_root).is_symlink():
        raise ValueError("interrupted_cell_artifact_symlink_forbidden")
    runtime_root, child_root = Path(runtime_root).resolve(), Path(child_root).resolve()
    if not child_root.is_dir() or not reason.strip() or len(reason) > 512:
        raise ValueError("interrupted_cell_recovery_arguments_invalid")
    if timeout_seconds is not None and (isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("interrupted_cell_recovery_timeout_invalid")
    inputs, authority, cell, specs, binding = _load_binding(runtime_root, selected_cell_index)
    expected_ids = {f"{inputs['run_id']}--{cell['cell_id']}--{candidate}" for candidate in CANDIDATE_IDS}
    if any(path.parent.parent.name not in expected_ids
           for path in child_root.glob("episodes/media/*/policy-requests/*.json")):
        raise ValueError("interrupted_cell_unbound_request_artifact")
    result_path = child_root / PROVIDER_RESULT_FILENAME
    if result_path.exists() or result_path.is_symlink():
        existing = _read(result_path)
        recovery = existing.get("interrupted_cell_recovery") or {}
        if (existing.get("result_digest") != canonical_digest(existing, digest_field="result_digest")
                or recovery.get("schema_version") != SCHEMA_VERSION or recovery.get("binding") != binding
                or recovery.get("reason") != reason or recovery.get("timeout_seconds") != timeout_seconds
                or existing.get("selected_cell_index") != selected_cell_index or existing.get("status") != "blocked"
                or recovery.get("source_inventory_digest") != _inventory_digest(_source_inventory(child_root))):
            raise FileExistsError("interrupted_cell_result_overwrite_forbidden")
        for artifact in existing.get("artifact_inventory") or []:
            _verify_artifact(child_root, artifact)
        return existing
    source_inventory = _source_inventory(child_root)
    source_map = {row["relative_path"]: row for row in source_inventory}
    source_manifest = {"schema_version": SCHEMA_VERSION, "binding": binding,
                       "source_artifacts": source_inventory, "source_inventory_digest": _inventory_digest(source_inventory)}
    recovery_root = child_root / RECOVERY_DIRECTORY
    if recovery_root.is_symlink():
        raise ValueError("interrupted_cell_artifact_symlink_forbidden")
    _write_json(recovery_root / "source_manifest.json", source_manifest)
    episodes, derived = [], []
    for candidate in CANDIDATE_IDS:
        spec = specs[candidate]
        episode_id = f"{inputs['run_id']}--{cell['cell_id']}--{candidate}"
        if not IDENTIFIER_PATTERN.fullmatch(episode_id):
            raise ValueError("interrupted_cell_episode_id_invalid")
        episode_binding = {**binding, "candidate_id": candidate, "episode_id": episode_id,
                           "execution_spec_digest": spec["execution_spec_digest"]}
        original, original_ref = _original_failure(child_root, episode_id, candidate, binding, spec)
        requests, images = _request_records(child_root, episode_id, candidate, spec, binding)
        composites, streams, media_gaps = _recover_frames(child_root, episode_id, candidate, images)
        first_observation = bool(composites)
        candidate_root = recovery_root / candidate
        missing_terminal = [] if original is not None else ["terminal_episode_receipt_missing", "policy_response_receipts_unavailable",
            "action_delivery_readback_unavailable", "deterministic_score_unavailable", "native_timestamps_unavailable",
            "scientific_reset_unavailable", "native_camera_calibration_unavailable"]
        gaps = sorted(set(media_gaps + missing_terminal))
        manifest = {"schema_version": "policy_canary_interrupted_observation_manifest.v1",
            "binding": episode_binding, "retained_policy_requests": requests, "lossless_composite_frames": composites,
            "camera_streams": streams, "native_timestamps_recovered": False,
            "frame_order_source": "retained_sequence_filename", "playback_only": True,
            "terminal_observation_invented": False, "evidence_gaps": gaps, "frame_manifest_digest": ""}
        manifest["frame_manifest_digest"] = canonical_digest(manifest, digest_field="frame_manifest_digest")
        manifest_path = candidate_root / "interrupted_frame_manifest.json"
        _write_json(manifest_path, manifest)
        manifest_ref = _artifact(child_root, manifest_path, "lossless_frame_manifest")
        derived.append(manifest_ref)
        videos, video_gaps = {}, []
        # The original failed media is already sealed; preserve its references.
        if original is None:
            for camera, frames in streams.items():
                path = candidate_root / f"{camera}-playback.mp4"
                try:
                    video = _encode_or_resume_episode_video(
                        [child_root / row["relative_path"] for row in frames], video_path=path, frames_per_second=4.0)
                except (OSError, RuntimeError, ValueError) as exc:
                    video_gaps.append({"type": "derived_review_video_unavailable", "camera_id": camera,
                                       "reason": type(exc).__name__})
                    continue
                video.update(relative_path=path.relative_to(child_root).as_posix(), camera_id=camera,
                    playback_only=True, native_timestamps_recovered=False,
                    frame_order_source="retained_sequence_filename", derived_from_frame_manifest_digest=manifest["frame_manifest_digest"])
                videos[camera] = video
                derived.append(_artifact(child_root, path, "review_video"))
        response_proven = bool(original and original.get("candidate_policy_queried") is True)
        query = {"schema_version": "policy_canary_interrupted_query_attempts.v1", "binding": episode_binding,
            "candidate_policy_query_attempted": bool(requests) or response_proven, "retained_request_count": len(requests),
            "policy_response_status": "received" if response_proven else "unproven", "wire_dispatch_completed_proven": False,
            "candidate_policy_queried": response_proven, "candidate_action_returned": bool(original and original.get("candidate_action_returned") is True),
            "policy_request_artifacts": requests, "policy_queries": (original or {}).get("candidate_policy_action_queries") or [], "receipt_digest": ""}
        query["receipt_digest"] = canonical_digest(query, digest_field="receipt_digest")
        query_path = candidate_root / "interrupted.policy_query_receipt.json"
        _write_json(query_path, query)
        query_ref = _artifact(child_root, query_path, "policy_query_receipt")
        derived.append(query_ref)
        visual = {"schema_version": "policy_canary_interrupted_visual_evidence.v1", "status": "partial_retained_observations",
            "human_review_available": bool(videos), "playback_only": True, "native_timestamps_recovered": False,
            "candidate_exact_policy_input_frame_count": len(images), "retained_observation_frame_count": len(composites),
            "episode_terminal_status": "interrupted_after_first_observation" if first_observation else "interrupted_before_first_observation",
            "terminal_observation_present": False, "terminal_observation_invented": False,
            "frame_manifest_digest": manifest["frame_manifest_digest"], "frame_manifest": manifest_ref,
            "videos": videos, "video_gaps": video_gaps, "media_gap": {"type": "interrupted_after_first_observation" if first_observation else "before_first_observation", "reason": reason}}
        row = dict(original) if original is not None else {
            "status": "blocked", "candidate_policy_queried": False, "candidate_action_returned": False,
            "actions_reached_robot": False, "arm_moved": False, "policy_outcome_interpretable": False,
            "scientific_reset": None, "reset_state_digest": None, "scoring_authority": None,
            "first_observation_retained": first_observation,
            "typed_harness_failure": visual["episode_terminal_status"], "failure_type": "InterruptedCell",
            "failure_message": reason, "episode_failure_stage": visual["episode_terminal_status"],
            "visual_evidence": visual, "evidence_artifacts": {}, "candidate_policy_action_queries": [],
            "commanded_actions": [], "episode": {"episode_id": episode_id, "scientific_reset": None,
                "candidate_policy_action_queries": [], "commanded_actions": [], "policy_request_artifacts": requests,
                "visual_evidence": visual, "score": {"status": "not_scored", "blockers": ["interrupted_cell_score_unproven"]}}}
        row.update(**cell, run_kind=RUN_KIND, claim_ceiling=CLAIM_CEILING, candidate_id=candidate,
            checkpoint_digest=spec["checkpoint_digest"], runtime_identity_digest=spec["runtime_identity_digest"],
            execution_spec_digest=spec["execution_spec_digest"], task_success_contract_digest=inputs["task_success_contract_digest"],
            candidate_policy_query_attempted=bool(requests) or response_proven or row.get("candidate_policy_query_attempted") is True,
            retained_policy_request_count=len(requests), policy_response_status="received" if row.get("candidate_policy_queried") else "unproven",
            ranking_eligible=False, recovery_binding=episode_binding, original_failure_receipt=original_ref)
        row["evidence_artifacts"] = dict(row.get("evidence_artifacts") or {})
        row["evidence_artifacts"].setdefault("frame_manifest", manifest_ref if first_observation else None)
        if requests:
            row["evidence_artifacts"].setdefault("policy_query_receipt", query_ref)
        if videos:
            first_video = next(iter(videos.values()))
            row["evidence_artifacts"].setdefault("review_video", _artifact(child_root,
                child_root / first_video["relative_path"], "review_video"))
        row["recovery_evidence_gaps"] = gaps
        episodes.append(row)
    # Detect concurrent writers or changed originals before publishing a result.
    current_sources = _source_inventory(child_root)
    if {row["relative_path"]: row for row in current_sources} != source_map:
        raise ValueError("interrupted_cell_source_changed_during_recovery")
    derived.append(_artifact(child_root, recovery_root / "source_manifest.json", "interrupted_cell_source_manifest"))
    inventory = sorted([*source_inventory, *derived], key=lambda row: row["relative_path"])
    result = {"schema_version": RESULT_SCHEMA_VERSION, "status": "blocked", "run_id": inputs["run_id"],
        "run_kind": RUN_KIND, "claim_ceiling": CLAIM_CEILING, "candidate_ids": list(CANDIDATE_IDS),
        "selected_cell_index": selected_cell_index, "episodes": episodes,
        "task_success_contract": inputs["task_success_contract"], "task_success_contract_digest": inputs["task_success_contract_digest"],
        "runtime_inputs_digest": inputs["runtime_inputs_digest"], "authority_digest": authority["authority_digest"],
        "matrix_digest": inputs.get("matrix_digest"), "episodes_per_policy": 10, "learned_policy_rollout_count": 20,
        "retry_cap": 0, "automatic_retry_performed": False, "candidate_policy_queried": any(row.get("candidate_policy_queried") is True for row in episodes),
        "scene_promotion_performed": False, "official_ranking_performed": False, "provider_zero_required_after_return": True,
        "session_closeout": {"status": "interrupted_runtime_close_unproven", "runtime_closed": False, "provider_closeout_pending": True},
        "blockers": [reason], "artifact_inventory": inventory, "artifact_inventory_digest": _inventory_digest(inventory),
        "interrupted_cell_recovery": {"schema_version": SCHEMA_VERSION, "binding": binding,
            "reason": reason, "timeout_seconds": timeout_seconds, "source_inventory_digest": _inventory_digest(source_inventory),
            "original_bytes_preserved": True, "policy_execution_performed_by_recovery": False,
            "derived_media_only": True}, "result_digest": ""}
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    _write_json(result_path, result)
    return result


__all__ = ["recover_interrupted_cell_result"]
