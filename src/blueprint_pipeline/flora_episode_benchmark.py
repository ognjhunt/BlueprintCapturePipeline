"""Blind, replayable FLORA comparison against retained ADP-009D episodes.

This is a research-only sidecar. It never changes source scores or publishes a
customer result. Preparing a dataset copies and verifies retained evidence;
running it requires exact per-episode rights and an already admitted runtime.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import time
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json, cross_runtime_canonical_digest
from .episode_interpretation import (
    EpisodeInterpretationRequest,
    _frame_records,
    build_episode_interpretation_request,
    interpret_episode,
    validate_episode_interpretation_rights,
    validate_interpreter_adapter_metadata,
)
from .policy_canary_episode_interpretation_closeout import (
    _episode_evidence_root,
    _source_path,
)

DATASET_SCHEMA = "flora_episode_benchmark_dataset.v1"
REPORT_SCHEMA = "flora_episode_benchmark_report.v1"


class FloraBenchmarkError(ValueError):
    """A benchmark must retain unknowns instead of manufacturing a comparison."""


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FloraBenchmarkError("flora_benchmark_file_invalid")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FloraBenchmarkError("flora_benchmark_mapping_required")
    return value


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    content = canonical_json(dict(value)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FloraBenchmarkError("flora_benchmark_symlink_forbidden")
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != content:
            raise FloraBenchmarkError("flora_benchmark_immutable_output_conflict")


def _inside(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise FloraBenchmarkError("flora_benchmark_path_outside_root")
    candidate = root / path
    if any(parent.is_symlink() for parent in (candidate, *candidate.parents) if parent != root.parent):
        raise FloraBenchmarkError("flora_benchmark_symlink_forbidden")
    resolved = candidate.resolve()
    if root != resolved and root not in resolved.parents:
        raise FloraBenchmarkError("flora_benchmark_path_outside_root")
    if must_exist and not resolved.is_file():
        raise FloraBenchmarkError("flora_benchmark_source_missing")
    return resolved


def _copy_once(source: Path, target: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise FloraBenchmarkError("flora_benchmark_source_missing")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise FloraBenchmarkError("flora_benchmark_symlink_forbidden")
    if target.exists():
        if _sha(source) != _sha(target):
            raise FloraBenchmarkError("flora_benchmark_staged_file_conflict")
        return
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    if _sha(source) != _sha(target):
        raise FloraBenchmarkError("flora_benchmark_copy_verification_failed")


def _correction_key(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("candidate_id"), row.get("cell_id"), row.get("seed")


def _correction_payload(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read(path)
    if payload.get("schema_version") == "task_evaluation_policy_canary_score_correction.v1":
        return payload, payload
    if (payload.get("schema_version") != "task_evaluation_policy_canary_score_correction_ingest.v2"
            or payload.get("ingest_digest") != cross_runtime_canonical_digest(payload, digest_field="ingest_digest")
            or not isinstance(payload.get("correction"), dict)
            or not isinstance(payload.get("derived_rescore_receipts"), list)
            or payload.get("source_binding", {}).get("run_id") != payload["correction"].get("source_run_id")):
        raise FloraBenchmarkError("flora_benchmark_corrected_reference_invalid")
    return payload, payload["correction"]


def _stage_corrected_reference(*, source: Mapping[str, Any], source_sha256: str,
                               path: Path, destination: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Retain the exact correction packet; never rewrite the original result."""
    from .task_evaluation_policy_canary_rescore import validate_policy_canary_score_correction

    input_hash = _sha(path)
    payload, correction = _correction_payload(path)
    correction_id = correction.get("correction_id")
    if (not re.fullmatch(r"[0-9a-f]{24}", str(correction_id))
            or correction.get("source_result_file_sha256") != source_sha256):
        raise FloraBenchmarkError("flora_benchmark_corrected_reference_source_mismatch")
    root = destination / "corrected_reference" / correction_id
    _copy_once(path, root / "source_artifact.json")
    embedded = payload.get("derived_rescore_receipts")
    receipts = {row.get("receipt_digest"): row for row in embedded} if embedded is not None else None
    if embedded is not None and len(receipts) != len(embedded):
        raise FloraBenchmarkError("flora_benchmark_duplicate_correction_receipt")
    for update in correction["score_updates"]:
        record = update["derived_rescore_receipt"]
        target = _inside(root, record["relative_path"], must_exist=False)
        if receipts is None:
            _copy_once(_inside(path.parent.resolve(), record["relative_path"]), target)
        else:
            receipt = receipts.get(record["receipt_digest"])
            if receipt is None:
                raise FloraBenchmarkError("flora_benchmark_correction_receipt_missing")
            _write_once(target, receipt)
    _write_once(root / "score_correction.json", correction)
    validated = validate_policy_canary_score_correction(source_result=source, correction=correction, receipt_root=root)
    if _sha(path) != input_hash:
        raise FloraBenchmarkError("flora_benchmark_corrected_reference_changed")
    record = {
        "schema_version": "flora_corrected_reference_binding.v1",
        "relative_path": (root / "score_correction.json").relative_to(destination).as_posix(),
        "sha256": _sha(root / "score_correction.json"), "correction_digest": validated["correction_digest"],
        "scoring_version_digest": validated["scoring_version_digest"],
        "source_artifact": {"relative_path": (root / "source_artifact.json").relative_to(destination).as_posix(),
                            "sha256": input_hash, "schema_version": payload["schema_version"],
                            "ingest_digest": payload.get("ingest_digest")},
    }
    return validated, record


def _load_corrected_reference(root: Path, dataset: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any] | None:
    from .task_evaluation_policy_canary_rescore import validate_policy_canary_score_correction

    record = dataset.get("corrected_reference")
    if record is None:
        return None
    path = _inside(root, record["relative_path"])
    source_path = _inside(root, record["source_artifact"]["relative_path"])
    payload, source_correction = _correction_payload(source_path)
    correction = _read(path)
    if (record.get("schema_version") != "flora_corrected_reference_binding.v1"
            or _sha(path) != record.get("sha256") or correction != source_correction
            or _sha(source_path) != record["source_artifact"].get("sha256")
            or record["source_artifact"].get("schema_version") != payload.get("schema_version")
            or record["source_artifact"].get("ingest_digest") != payload.get("ingest_digest")
            or record.get("correction_digest") != correction.get("correction_digest")
            or record.get("scoring_version_digest") != correction.get("scoring_version_digest")
            or correction.get("source_result_file_sha256") != dataset["source_result"]["sha256"]):
        raise FloraBenchmarkError("flora_benchmark_corrected_reference_binding_invalid")
    return validate_policy_canary_score_correction(source_result=source, correction=correction, receipt_root=path.parent)


def _score_reference(score: Mapping[str, Any]) -> dict[str, Any]:
    return {"status": score.get("status"), "task_succeeded": score.get("task_succeeded"),
            "score_digest": score.get("report_digest")}


def _episode_correction_binding(correction: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    return {"correction_digest": correction["correction_digest"],
            "scoring_version_digest": correction["scoring_version_digest"],
            "source_episode_identity_digest": update["source_episode_identity_digest"],
            "derived_rescore_receipt_digest": update["derived_rescore_receipt"]["receipt_digest"],
            "old_score_digest": update["old_score_digest"], "new_score_digest": update["new_score_digest"]}


def _review_video_records(episode: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], str]:
    records = episode["evidence_artifacts"]
    if records.get("review_videos"):
        videos = records["review_videos"]
        if not isinstance(videos, list) or not all(isinstance(row, Mapping) for row in videos):
            raise FloraBenchmarkError("flora_benchmark_review_videos_missing")
        return videos, "run_evidence_root"
    visual = (episode.get("episode") or {}).get("visual_evidence", {})
    if isinstance(visual, Mapping) and visual.get("schema_version") == "adp_multicamera_episode_visual_evidence.v1":
        cameras, videos = visual.get("required_camera_ids"), visual.get("videos")
        if (visual.get("status") != "complete" or not isinstance(cameras, list)
                or not 1 <= len(cameras) <= 4 or len(set(cameras)) != len(cameras)
                or not isinstance(videos, Mapping) or set(videos) != set(cameras)
                or any(not isinstance(videos[camera], Mapping) or videos[camera].get("camera_id") != camera
                       or videos[camera].get("derived_from_frame_manifest_digest") != visual.get("frame_manifest_digest")
                       for camera in cameras)):
            raise FloraBenchmarkError("flora_benchmark_multicamera_binding_invalid")
        return [videos[camera] for camera in cameras], "episode_evidence_root"
    return [records["review_video"]], "run_evidence_root"


def _request(dataset_root: Path, row: Mapping[str, Any]) -> EpisodeInterpretationRequest:
    relative_root = str(row["evidence_root"])
    if Path(relative_root).is_absolute() or ".." in Path(relative_root).parts:
        raise FloraBenchmarkError("flora_benchmark_episode_root_invalid")
    evidence_root = (dataset_root / relative_root).resolve()
    if dataset_root not in evidence_root.parents:
        raise FloraBenchmarkError("flora_benchmark_episode_root_invalid")
    sources = row["sources"]
    for name, relative in sources.items():
        paths = relative if name == "review_video_paths" else [relative]
        for value in paths:
            _inside(evidence_root, str(value))
    result = build_episode_interpretation_request(
        episode_id=str(row["episode_id"]),
        candidate_policy_id=str(row["candidate_id"]),
        evidence_root=evidence_root,
        **sources,
    )
    if row.get("input_bundle_digest") not in (None, result.input_receipt["input_bundle_digest"]):
        raise FloraBenchmarkError("flora_benchmark_input_digest_mismatch")
    if row.get("reference") is not None and row["reference"] != {
        "status": result.deterministic_score.get("status"),
        "task_succeeded": result.deterministic_score.get("task_succeeded"),
        "score_digest": result.input_receipt["deterministic_score_digest"],
    }:
        raise FloraBenchmarkError("flora_benchmark_reference_score_mismatch")
    return result


def prepare_flora_benchmark(
    *,
    session_result_path: Path,
    evidence_root: Path,
    profile: Mapping[str, Any],
    output_dir: Path,
    episode_ids: Sequence[str] | None = None,
    corrected_reference_path: Path | None = None,
) -> dict[str, Any]:
    """Stage portable evidence without modifying the previous run's bytes."""
    from .flora_episode_interpretation import validate_flora_interpreter_profile

    validate_flora_interpreter_profile(profile)
    source = _read(session_result_path)
    if source.get("result_digest") != canonical_digest(source, digest_field="result_digest"):
        raise FloraBenchmarkError("flora_benchmark_source_result_digest_invalid")
    source_hash = _sha(session_result_path)
    root = evidence_root.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    if destination == root or root in destination.parents:
        raise FloraBenchmarkError("flora_benchmark_output_must_be_separate_from_source")
    episodes = source.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise FloraBenchmarkError("flora_benchmark_episode_inventory_missing")
    selected = set(episode_ids or [])
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    correction, correction_record = (None, None)
    if corrected_reference_path is not None:
        correction, correction_record = _stage_corrected_reference(
            source=source, source_sha256=source_hash, path=corrected_reference_path, destination=destination)
    corrected_by_key = {_correction_key(row): row for row in correction["score_updates"]} if correction else {}
    contract = source.get("task_success_contract")
    if not isinstance(contract, Mapping) and correction is None:
        raise FloraBenchmarkError("flora_benchmark_confirmed_task_contract_missing")
    if correction is None and contract.get("contract_digest") != profile["task"]["task_success_contract_digest"]:
        raise FloraBenchmarkError("flora_benchmark_profile_task_mismatch")
    for episode in episodes:
        identity = str((episode.get("episode") or {}).get("episode_id") or
                       f"{source.get('run_id')}--{episode.get('cell_id')}--{episode.get('candidate_id')}")
        if identity in seen:
            raise FloraBenchmarkError("flora_benchmark_duplicate_episode")
        seen.add(identity)
        if selected and identity not in selected:
            continue
        opaque_id = hashlib.sha256(identity.encode()).hexdigest()
        row: dict[str, Any] = {
            "episode_id": identity, "candidate_id": episode.get("candidate_id"),
            "cell_id": episode.get("cell_id"), "seed": episode.get("seed"),
            "source_episode_status": episode.get("status"),
            "evidence_root": "episodes/" + opaque_id,
            "input_status": "unavailable",
        }
        staged = destination / row["evidence_root"]
        update = corrected_by_key.get(_correction_key(episode))
        if correction is not None:
            if update is None:
                raise FloraBenchmarkError("flora_benchmark_corrected_episode_missing")
            contract = update["new_score"]["task_success_contract"]
            if contract["contract_digest"] != profile["task"]["task_success_contract_digest"]:
                raise FloraBenchmarkError("flora_benchmark_profile_task_mismatch")
            row["original_reference"] = _score_reference(episode["episode"]["score"])
            row["reference"] = _score_reference(update["new_score"])
            row["corrected_reference"] = _episode_correction_binding(correction, update)
        try:
            records = episode["evidence_artifacts"]
            source_paths = {
                key: _source_path(root, records[role])
                for key, role in (
                    ("deterministic_score_path", "score_receipt"),
                    ("state_trace_path", "state_trace"),
                    ("contact_force_trace_path", "contact_force_trace"),
                    ("frame_manifest_path", "frame_manifest"),
                )
            }
            manifest_path = source_paths["frame_manifest_path"]
            original_root = _episode_evidence_root(run_evidence_root=root, manifest_path=manifest_path)
            video_records, video_scope = _review_video_records(episode)
            if len(video_records) != len(profile["task"]["cameras"]):
                raise FloraBenchmarkError("flora_benchmark_profile_camera_count_mismatch")
            if video_scope == "episode_evidence_root" and any(
                record["derived_from_frame_manifest_digest"] != _read(manifest_path)["frame_manifest_digest"]
                for record in video_records
            ):
                raise FloraBenchmarkError("flora_benchmark_multicamera_manifest_mismatch")
            videos = [_source_path(original_root if video_scope == "episode_evidence_root" else root, record)
                      for record in video_records]
            row["review_video_source"] = video_scope
            row["review_video_camera_ids"] = [record.get("camera_id") for record in video_records]
            relative_sources: dict[str, Any] = {}
            for key, path in source_paths.items():
                relative = path.relative_to(original_root).as_posix()
                _copy_once(path, _inside(staged, relative, must_exist=False))
                relative_sources[key] = relative
            if update is not None:
                if _read(source_paths["deterministic_score_path"]) != episode["episode"]["score"]:
                    raise FloraBenchmarkError("flora_benchmark_original_score_binding_invalid")
                row["original_score_path"] = relative_sources["deterministic_score_path"]
                relative_sources["deterministic_score_path"] = "_flora/corrected_score.json"
                _write_once(staged / relative_sources["deterministic_score_path"], update["new_score"])
            for frame in _frame_records(_read(manifest_path)):
                relative = str(frame.get("relative_path") or frame.get("path") or "")
                _copy_once(_inside(original_root, relative), _inside(staged, relative, must_exist=False))
            relative_sources["review_video_paths"] = []
            for path in videos:
                relative = path.relative_to(original_root).as_posix()
                _copy_once(path, _inside(staged, relative, must_exist=False))
                relative_sources["review_video_paths"].append(relative)
            relative_sources["task_success_contract_path"] = "_flora/task_success_contract.json"
            _write_once(staged / relative_sources["task_success_contract_path"], contract)
            row["sources"] = relative_sources
            request = _request(destination, row)
            row["input_bundle_digest"] = request.input_receipt["input_bundle_digest"]
            row["reference"] = {
                "status": request.deterministic_score.get("status"),
                "task_succeeded": request.deterministic_score.get("task_succeeded"),
                "score_digest": request.input_receipt["deterministic_score_digest"],
            }
            row["input_status"] = "ready"
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row["input_gap"] = {"error_type": type(exc).__name__, "reason": str(exc)[:400]}
        rows.append(row)
    if selected - seen:
        raise FloraBenchmarkError("flora_benchmark_selected_episode_not_found")
    if not any(row["input_status"] == "ready" for row in rows):
        _write_once(destination / "preparation_gaps.json", {"episodes": rows})
        reasons = sorted({str(row.get("input_gap", {}).get("reason")) for row in rows})
        raise FloraBenchmarkError("flora_benchmark_no_valid_retained_episodes:" + ";".join(reasons[:3]))
    _copy_once(session_result_path, destination / "source_session_result.json")
    _write_once(destination / "profile.json", profile)
    dataset = {
        "schema_version": DATASET_SCHEMA,
        "run_id": source.get("run_id"),
        "source_result": {
            "relative_path": "source_session_result.json", "sha256": source_hash,
            "result_digest": source["result_digest"],
        },
        "source_episode_count": len(episodes), "selected_episode_count": len(rows),
        "interpreter_profile_digest": profile["profile_digest"],
        "episodes": rows, "research_only": True,
        "claim_ceiling": "development_only_visual_grader_comparison",
        "source_scores_modified": False, "dataset_digest": "",
        "reference_kind": "corrected_retained_deterministic_simulator_score" if correction else "retained_deterministic_simulator_score",
        "corrected_reference": correction_record,
    }
    dataset["dataset_digest"] = canonical_digest(dataset, digest_field="dataset_digest")
    _write_once(destination / "manifest.json", dataset)
    if _sha(session_result_path) != source_hash:
        raise FloraBenchmarkError("flora_benchmark_source_changed_during_preparation")
    return dataset


def load_flora_benchmark(
    manifest_path: Path, profile: Mapping[str, Any]
) -> tuple[dict[str, Any], list[EpisodeInterpretationRequest]]:
    from .flora_episode_interpretation import validate_flora_interpreter_profile

    validate_flora_interpreter_profile(profile)
    root = manifest_path.expanduser().resolve().parent
    dataset = _read(manifest_path)
    if (dataset.get("schema_version") != DATASET_SCHEMA
            or dataset.get("research_only") is not True
            or dataset.get("source_scores_modified") is not False
            or dataset.get("dataset_digest") != canonical_digest(dataset, digest_field="dataset_digest")
            or dataset.get("interpreter_profile_digest") != profile["profile_digest"]):
        raise FloraBenchmarkError("flora_benchmark_manifest_invalid")
    source_record = dataset["source_result"]
    source_path = _inside(root, source_record["relative_path"])
    source = _read(source_path)
    if (_sha(source_path) != source_record["sha256"]
            or source.get("result_digest") != source_record["result_digest"]
            or source.get("result_digest") != canonical_digest(source, digest_field="result_digest")
            or source.get("run_id") != dataset["run_id"]):
        raise FloraBenchmarkError("flora_benchmark_source_result_mismatch")
    correction = _load_corrected_reference(root, dataset, source)
    corrected_by_key = {_correction_key(row): row for row in correction["score_updates"]} if correction else {}
    expected_kind = "corrected_retained_deterministic_simulator_score" if correction else "retained_deterministic_simulator_score"
    if dataset.get("reference_kind") != expected_kind:
        raise FloraBenchmarkError("flora_benchmark_reference_kind_invalid")
    rows = dataset.get("episodes")
    if not isinstance(rows, list) or not rows or len(rows) != dataset["selected_episode_count"]:
        raise FloraBenchmarkError("flora_benchmark_episode_inventory_invalid")
    if len({row["episode_id"] for row in rows}) != len(rows):
        raise FloraBenchmarkError("flora_benchmark_duplicate_episode")
    source_rows = {}
    for row in source["episodes"]:
        identity = str((row.get("episode") or {}).get("episode_id") or
                       f"{source.get('run_id')}--{row.get('cell_id')}--{row.get('candidate_id')}")
        if identity in source_rows:
            raise FloraBenchmarkError("flora_benchmark_duplicate_source_episode")
        source_rows[identity] = row
    for row in rows:
        original = source_rows.get(row["episode_id"])
        if original is None or any(row.get(key) != original.get(key) for key in ("candidate_id", "cell_id", "seed")):
            raise FloraBenchmarkError("flora_benchmark_episode_source_binding_invalid")
        update = corrected_by_key.get(_correction_key(original))
        if correction is not None and (
            update is None or row.get("reference") != _score_reference(update["new_score"])
            or row.get("original_reference") != _score_reference(original["episode"]["score"])
            or row.get("corrected_reference") != _episode_correction_binding(correction, update)
        ):
            raise FloraBenchmarkError("flora_benchmark_corrected_episode_binding_invalid")
        if row["input_status"] != "ready":
            continue
        staged_root = (root / row["evidence_root"]).resolve()
        for key, role in (("deterministic_score_path", "score_receipt"), ("state_trace_path", "state_trace"),
                          ("contact_force_trace_path", "contact_force_trace"), ("frame_manifest_path", "frame_manifest")):
            relative = row["original_score_path"] if update is not None and key == "deterministic_score_path" else row["sources"][key]
            if _sha(_inside(staged_root, relative)) != original["evidence_artifacts"][role]["sha256"]:
                raise FloraBenchmarkError("flora_benchmark_staged_source_digest_mismatch")
        if update is not None and _read(_inside(staged_root, row["sources"]["deterministic_score_path"])) != update["new_score"]:
            raise FloraBenchmarkError("flora_benchmark_staged_correction_score_mismatch")
        video_records, video_scope = _review_video_records(original)
        if (row.get("review_video_source") != video_scope
                or row.get("review_video_camera_ids") != [record.get("camera_id") for record in video_records]
                or len(video_records) != len(profile["task"]["cameras"])):
            raise FloraBenchmarkError("flora_benchmark_video_selection_invalid")
        if video_scope == "episode_evidence_root" and any(
            record["derived_from_frame_manifest_digest"] != _read(
                _inside(staged_root, row["sources"]["frame_manifest_path"]))["frame_manifest_digest"]
            for record in video_records
        ):
            raise FloraBenchmarkError("flora_benchmark_multicamera_manifest_mismatch")
        video_paths = row["sources"]["review_video_paths"]
        if len(video_records) != len(video_paths) or any(
            _sha(_inside(staged_root, path)) != record["sha256"] for path, record in zip(video_paths, video_records)
        ):
            raise FloraBenchmarkError("flora_benchmark_staged_video_digest_mismatch")
    requests = [_request(root, row) for row in rows if row["input_status"] == "ready"]
    if not requests:
        raise FloraBenchmarkError("flora_benchmark_no_valid_retained_episodes")
    if any(request.task_success_contract["contract_digest"] !=
           profile["task"]["task_success_contract_digest"] for request in requests):
        raise FloraBenchmarkError("flora_benchmark_profile_task_mismatch")
    return dataset, requests


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    if denominator == 0:
        return {"numerator": numerator, "denominator": 0, "rate": None, "wilson_95": None}
    p = numerator / denominator
    z = 1.959963984540054
    adjustment = 1 + z * z / denominator
    center = (p + z * z / (2 * denominator)) / adjustment
    half = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / adjustment
    return {"numerator": numerator, "denominator": denominator, "rate": p,
            "wilson_95": [max(0.0, center - half), min(1.0, center + half)]}


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["comparison"] for row in rows)
    passed = counts["true_pass"] + counts["false_pass"]
    failed = counts["true_fail"] + counts["false_fail"]
    actual_failed = counts["false_pass"] + counts["true_fail"]
    actual_passed = counts["true_pass"] + counts["false_fail"]
    return {
        "episode_count": len(rows),
        "confusion_matrix": {key: counts[key] for key in ("true_pass", "false_pass", "true_fail", "false_fail")},
        "abstained_count": counts["abstain"],
        "missing_reference_count": sum(type(row.get("reference_success")) is not bool for row in rows),
        "coverage": _rate(passed + failed, len(rows)),
        "false_pass_rate_among_classified_failures": _rate(counts["false_pass"], actual_failed),
        "false_fail_rate_among_classified_successes": _rate(counts["false_fail"], actual_passed),
        "pass_precision": _rate(counts["true_pass"], passed),
        "agreement_among_classified": _rate(counts["true_pass"] + counts["true_fail"], passed + failed),
    }


def _receipt_execution_kind(receipt: Mapping[str, Any], profile: Mapping[str, Any],
                            expected: Any, *, raw_artifact_root: Path | None = None) -> str:
    """Validate FLORA provenance before allowing a visual comparison claim."""
    from .flora_episode_interpretation import SAMPLING, normalize_flora_rewards, verify_flora_raw_artifacts
    from .flora_rewarder_runtime import FLORA_MODEL, FLORA_REVISION, MODEL_FILES, MODES, SOURCE_SHA256

    identity_digest = canonical_digest(expected.identity.__dict__)
    prompt = expected.prompt_identity()
    if (receipt.get("schema_version") != "episode_interpretation_receipt.v1"
            or receipt.get("interpreter") != expected.identity.__dict__
            or receipt.get("interpreter_identity_digest") != identity_digest
            or any(receipt.get(key) != value for key, value in prompt.items())):
        raise FloraBenchmarkError("flora_benchmark_receipt_interpreter_mismatch")
    boundary = receipt.get("proof_boundary", {})
    if (any(boundary.get(key) is not True for key in (
            "learned_interpretation_only", "authoritative_task_success_unchanged",
            "deterministic_score_overwrite_forbidden", "candidate_policy_self_grading_forbidden",
            "review_video_is_derived_visual_evidence", "simulation_is_not_physical_truth"))
            or boundary.get("ranking_or_promotion_effect") != "none"):
        raise FloraBenchmarkError("flora_benchmark_receipt_proof_boundary_invalid")
    inputs = receipt.get("input_receipt", {})
    if (inputs.get("input_bundle_digest") != receipt.get("input_bundle_digest")
            or inputs.get("input_bundle_digest") != canonical_digest(inputs, digest_field="input_bundle_digest")):
        raise FloraBenchmarkError("flora_benchmark_receipt_input_invalid")
    metadata = receipt.get("adapter_metadata")
    if metadata is None:
        if receipt.get("status") == "abstained" and receipt.get("provider_called") is False:
            return "unknown"
        raise FloraBenchmarkError("flora_benchmark_receipt_metadata_missing")
    metadata = validate_interpreter_adapter_metadata(
        metadata, input_bundle_digest=receipt["input_bundle_digest"],
        interpreter_identity_digest=identity_digest,
    )
    fixed = {"schema_version": "flora_episode_interpretation_metadata.v1",
             "profile_digest": profile["profile_digest"], "source_revision": FLORA_REVISION,
             "source_manifest_digest": canonical_digest(SOURCE_SHA256), "sampling": SAMPLING,
             "model_provider_id": "nvidia", "compute_provider_id": profile.get("compute_provider_id", "local"),
             "generation_settings": {key: profile["runtime_config"][key]
                                     for key in ("seed", "rubric_num_samples", "tensor_parallel_size")}}
    if (any(metadata.get(key) != value for key, value in fixed.items())
            or any(metadata.get(key) is not False for key in (
                "reference_scores_disclosed", "candidate_identity_disclosed",
                "state_and_contact_traces_disclosed", "confidence_calibrated"))
            or metadata.get("confidence_zero_means_unavailable") is not True):
        raise FloraBenchmarkError("flora_benchmark_receipt_metadata_mismatch")
    selected = {key: profile["task"][key] for key in ("instruction", "robot", "cameras", "rubric")}
    selected.update(video_digests=[row["sha256"] for row in inputs["artifacts"]["review_videos"]],
                    sampling=SAMPLING, prompt_digest=prompt["prompt_digest"])
    selected["generation_settings"] = {key: profile["runtime_config"][key]
                                       for key in ("seed", "rubric_num_samples", "tensor_parallel_size")}
    if metadata["model_input_digest"] != canonical_digest(selected):
        raise FloraBenchmarkError("flora_benchmark_receipt_model_input_mismatch")
    outcome = receipt.get("learned_interpretation", {}).get("episode_outcome")
    if metadata.get("status") == "completed":
        job = {"id": "episode-" + receipt["input_bundle_digest"][7:39],
               "task": profile["task"]["instruction"], "rubric": profile["task"]["rubric"]}
        rewards = normalize_flora_rewards(metadata.get("raw_outputs", {}), job=job,
                                         sample_count=profile["runtime_config"]["rubric_num_samples"])
        all_pass = all(reward["apparent_pass"] for reward in rewards.values())
        expected_outcome = "appears_complete" if all_pass else (
            "appears_incomplete" if not rewards["terminal"]["apparent_pass"]
            and not rewards["rubric"]["apparent_pass"] else "unclear")
        if (metadata.get("rewards") != rewards or metadata.get("apparent_pass") is not all_pass
                or outcome != expected_outcome
                or receipt.get("status") != ("abstained" if outcome == "unclear" else "completed")):
            raise FloraBenchmarkError("flora_benchmark_receipt_reward_mismatch")
    elif metadata.get("status") != "abstained" or receipt.get("status") != "abstained" or outcome != "unclear":
        raise FloraBenchmarkError("flora_benchmark_receipt_status_invalid")
    verify_flora_raw_artifacts(metadata, artifact_root=raw_artifact_root)

    if metadata.get("runtime_kind") == "hermetic_injected_runner":
        if (metadata.get("inference_execution_kind") != "hermetic_test"
                or metadata.get("learned_model_executed") is not False
                or metadata.get("inference_performed") is not False
                or metadata.get("inference_mode_count") != 0):
            raise FloraBenchmarkError("flora_benchmark_fixture_execution_claim_invalid")
        return "fixture"
    if metadata.get("runtime_kind") != "live_subprocess":
        return "unknown"
    executions = metadata.get("runtime_execution", [])
    preflight = metadata.get("runtime_preflight", {})
    execution_verified = (
        metadata.get("inference_execution_kind") == "real_model"
        and metadata.get("learned_model_executed") is True and metadata.get("inference_performed") is True
        and type(metadata.get("inference_mode_count")) is int and metadata["inference_mode_count"] == len(MODES)
        and isinstance(executions, list) and len(executions) == len(MODES)
        and all(isinstance(row, Mapping) for row in executions)
        and {row.get("mode") for row in executions} == set(MODES)
        and all(type(row.get("returncode")) is int and row["returncode"] == 0
                and row.get("timed_out") is False for row in executions)
        and isinstance(preflight, Mapping) and preflight.get("execute_ready") is True
        and preflight.get("schema_version") == "flora_runtime_preflight.v1"
        and preflight.get("model") == FLORA_MODEL and preflight.get("revision") == FLORA_REVISION
        and preflight.get("source_manifest_digest") == canonical_digest(SOURCE_SHA256)
        and preflight.get("checkpoint_manifest_digest") == canonical_digest(MODEL_FILES)
        and preflight.get("blockers") == []
    )
    return "live" if execution_verified else "unknown"


def build_flora_benchmark_report(
    *, dataset: Mapping[str, Any], profile: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]],
    elapsed_seconds: float, raw_artifact_root: Path | None = None,
) -> dict[str, Any]:
    from .flora_episode_interpretation import FloraEpisodeInterpreter

    expected = FloraEpisodeInterpreter.from_profile(profile)
    if (dataset.get("interpreter_profile_digest") != profile["profile_digest"]
            or dataset.get("dataset_digest") != canonical_digest(dataset, digest_field="dataset_digest")):
        raise FloraBenchmarkError("flora_benchmark_report_dataset_invalid")
    by_id: dict[str, Mapping[str, Any]] = {}
    execution_kinds: dict[str, str] = {}
    manifest_rows = {row["episode_id"]: row for row in dataset["episodes"]}
    for receipt in receipts:
        episode_id = receipt.get("episode_id")
        if episode_id not in manifest_rows or episode_id in by_id:
            raise FloraBenchmarkError("flora_benchmark_receipt_identity_invalid")
        if receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest"):
            raise FloraBenchmarkError("flora_benchmark_receipt_digest_invalid")
        source = manifest_rows[episode_id]
        if (receipt.get("candidate_policy_id") != source["candidate_id"]
                or receipt.get("input_bundle_digest") != source.get("input_bundle_digest")
                or receipt.get("proof_boundary", {}).get("authoritative_task_success_unchanged") is not True):
            raise FloraBenchmarkError("flora_benchmark_receipt_binding_invalid")
        reference = receipt.get("authoritative_deterministic_result", {})
        if any(reference.get(key) != source.get("reference", {}).get(key)
               for key in ("status", "task_succeeded", "score_digest")):
            raise FloraBenchmarkError("flora_benchmark_receipt_reference_mismatch")
        execution_kinds[episode_id] = _receipt_execution_kind(receipt, profile, expected, raw_artifact_root=raw_artifact_root)
        by_id[episode_id] = receipt
    rows = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in dataset["episodes"]:
        receipt = by_id.get(source["episode_id"])
        reference = source.get("reference", {})
        outcome = receipt.get("learned_interpretation", {}).get("episode_outcome") if receipt else "unclear"
        truth = reference.get("task_succeeded") if reference.get("status") == "scored" else None
        if receipt is None or outcome == "unclear" or receipt.get("status") != "completed":
            comparison = "abstain"
        elif not isinstance(truth, bool):
            comparison = "reference_unavailable"
        elif outcome == "appears_complete":
            comparison = "true_pass" if truth else "false_pass"
        elif outcome == "appears_incomplete":
            comparison = "false_fail" if truth else "true_fail"
        else:
            raise FloraBenchmarkError("flora_benchmark_unknown_visual_outcome")
        row = {
            "episode_id": source["episode_id"], "candidate_id": source["candidate_id"],
            "cell_id": source.get("cell_id"), "seed": source.get("seed"),
            "input_bundle_digest": source.get("input_bundle_digest"),
            "reference_success": truth, "reference_score_digest": reference.get("score_digest"),
            "original_reference": source.get("original_reference"),
            "corrected_reference": source.get("corrected_reference"),
            "visual_outcome": outcome, "comparison": comparison,
            "receipt_digest": receipt.get("receipt_digest") if receipt else None,
            "abstention_reason": (receipt.get("abstention_reason") if receipt else source.get("input_gap")),
            "adapter_metadata": receipt.get("adapter_metadata") if receipt else None,
            "execution_kind": execution_kinds.get(source["episode_id"], "unknown"),
        }
        rows.append(row)
        grouped[str(row["candidate_id"])].append(row)
    execution_counts = Counter(row["execution_kind"] for row in rows)
    all_live = bool(rows) and execution_counts["live"] == len(rows)
    all_fixture = bool(rows) and execution_counts["fixture"] == len(rows)
    claim_ceiling = ("development_only_visual_grader_comparison" if all_live else
                     "hermetic_wiring_validation_only" if all_fixture else "unverified_execution_comparison")
    report = {
        "schema_version": REPORT_SCHEMA, "run_id": dataset["run_id"],
        "dataset_digest": dataset["dataset_digest"],
        "interpreter_profile_digest": profile["profile_digest"],
        "source_result_digest": dataset["source_result"]["result_digest"],
        "status": "completed" if len(by_id) == len(rows) and not execution_counts["unknown"] and all(
            row["comparison"] not in {"abstain", "reference_unavailable"} for row in rows) else "partial",
        "metrics": _metrics(rows),
        "per_candidate": {key: _metrics(value) for key, value in sorted(grouped.items())},
        "episodes": rows, "elapsed_seconds": elapsed_seconds,
        "research_only": True, "claim_ceiling": claim_ceiling,
        "live_episode_count": execution_counts["live"], "fixture_episode_count": execution_counts["fixture"],
        "unknown_episode_count": execution_counts["unknown"], "runtime_execution_verified": all_live,
        "raw_artifacts_verified": bool(rows) and all(row.get("adapter_metadata") is not None for row in rows),
        "inference_performed": True if execution_counts["live"] else False if all_fixture else None,
        "reference_kind": dataset.get("reference_kind"), "corrected_reference": dataset.get("corrected_reference"),
        "reference_labels_exposed_to_model": False if all(row.get("adapter_metadata") is not None for row in rows) else None,
        "source_scores_modified": False, "ranking_or_promotion_effect": "none",
        "deployment_accuracy_qualified": False,
        "uncertainty_note": "Wilson intervals assume independent episodes; correlated scenario cells weaken that assumption.",
        "coverage_note": "Error rates are conditional on classified episodes; abstentions and absent references remain separate.",
        "report_digest": "",
    }
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    return report


def _markdown(report: Mapping[str, Any]) -> str:
    def literal(value: Any) -> str:
        value = " ".join(str(value).splitlines())
        return "".join(f"&#{ord(char)};" if char in "\\`*_{}[]()#+-.!|" else html.escape(char)
                       for char in value)

    metrics = report["metrics"]
    lines = [f"# FLORA retained-run comparison: {literal(report['run_id'])}", "",
             "Research-only visual assessment against retained simulator scores. Original results are unchanged.", "",
             f"Status: {report['status']}. Episodes: {metrics['episode_count']}. Abstentions: {metrics['abstained_count']}.", "",
             f"Execution evidence: {report['live_episode_count']} live; {report['fixture_episode_count']} fixture; "
             f"{report['unknown_episode_count']} unknown. Claim ceiling: {report['claim_ceiling']}.", "",
             f"Reference source: {literal(report['reference_kind'])}.", "",
             "| Episode | Candidate | Original success | Compared reference success | Visual outcome | Comparison |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in report["episodes"]:
        original = row.get("original_reference") or {"task_succeeded": row["reference_success"]}
        values = [row["episode_id"], row["candidate_id"], original.get("task_succeeded"),
                  row["reference_success"], row["visual_outcome"], row["comparison"]]
        lines.append("| " + " | ".join(literal(value) for value in values) + " |")
    lines.extend(["", "Rates below exclude abstentions; inspect coverage before interpreting accuracy.", ""])
    for name in ("coverage", "false_pass_rate_among_classified_failures", "false_fail_rate_among_classified_successes", "pass_precision"):
        value = metrics[name]
        rate = "unavailable" if value["rate"] is None else f"{100 * value['rate']:.1f}%"
        lines.append(f"- {name}: {rate} ({value['numerator']}/{value['denominator']}).")
    lines.extend(["", report["uncertainty_note"], "", "This comparison does not qualify deployment accuracy or physical success.", ""])
    return "\n".join(lines)


def run_flora_benchmark(
    *, manifest_path: Path, profile: Mapping[str, Any], rights_root: Path,
    output_dir: Path, execute: bool = False, interpreter: Any = None, raw_artifact_root: Path | None = None,
) -> dict[str, Any]:
    from .flora_episode_interpretation import FloraEpisodeInterpreter

    dataset, requests = load_flora_benchmark(manifest_path, profile)
    output_dir = output_dir.expanduser().resolve()
    if raw_artifact_root is None and (output_dir / "raw").is_dir():
        raw_artifact_root = output_dir / "raw"
    if raw_artifact_root is not None:
        raw_artifact_root = raw_artifact_root.expanduser().resolve()
    selected = interpreter or FloraEpisodeInterpreter.from_profile(profile, output_root=output_dir / "raw")
    expected = FloraEpisodeInterpreter.from_profile(profile)
    if selected.identity != expected.identity or selected.prompt_identity() != expected.prompt_identity():
        raise FloraBenchmarkError("flora_benchmark_selected_interpreter_mismatch")
    rights: dict[str, Path] = {}
    blockers = []
    for request in requests:
        path = rights_root / (request.input_receipt["input_bundle_digest"][7:] + ".json")
        try:
            validate_episode_interpretation_rights(rights_path=path, request=request, interpreter=selected)
            rights[request.episode_id] = path
        except (ValueError, OSError) as exc:
            blockers.append({"episode_id": request.episode_id, "reason": str(exc)[:400]})
    preflight = {
        "schema_version": "flora_episode_benchmark_preflight.v1",
        "status": "blocked" if blockers else "inputs_verified",
        "dataset_digest": dataset["dataset_digest"], "profile_digest": profile["profile_digest"],
        "eligible_episode_count": len(requests), "source_episode_count": dataset["source_episode_count"],
        "selected_episode_count": dataset["selected_episode_count"], "blockers": blockers,
        "runtime_execution_verified": False, "inference_performed": False,
        "paid_resource_admission_required_separately": True,
    }
    preflight["preflight_digest"] = canonical_digest(preflight, digest_field="preflight_digest")
    _write_once(output_dir / "input_preflights" / (preflight["preflight_digest"][7:] + ".json"), preflight)
    if blockers or not execute:
        return preflight
    started = time.monotonic()
    receipts = []
    missing = []
    for request in requests:
        path = output_dir / "receipts" / (request.input_receipt["input_bundle_digest"][7:] + ".json")
        if path.exists():
            receipt = _read(path)
            if (receipt.get("input_bundle_digest") != request.input_receipt["input_bundle_digest"]
                    or receipt.get("interpreter") != selected.identity.__dict__
                    or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")):
                raise FloraBenchmarkError("flora_benchmark_replay_receipt_conflict")
            receipts.append(receipt)
        else:
            missing.append((request, path))
    marker = output_dir / "execution_attempt.json"
    if missing:
        if marker.exists():
            raise FloraBenchmarkError("flora_benchmark_prior_execution_ambiguous")
        _write_once(marker, {"dataset_digest": dataset["dataset_digest"], "profile_digest": profile["profile_digest"],
                             "retry_limit": 0, "input_bundle_digests": [req.input_receipt["input_bundle_digest"] for req, _ in missing]})
        selected.run_batch([request for request, _ in missing], rights_attestation_paths=rights)
        for request, path in missing:
            receipts.append(interpret_episode(request=request, interpreter=selected,
                                             rights_attestation_path=rights[request.episode_id], output_path=path))
    # Replay returns the sealed report, never mutating its measured original duration.
    report_path = output_dir / "benchmark_report.json"
    if report_path.exists():
        existing = _read(report_path)
        expected = build_flora_benchmark_report(dataset=dataset, profile=profile, receipts=receipts,
                                                elapsed_seconds=existing["elapsed_seconds"], raw_artifact_root=raw_artifact_root)
        if expected != existing:
            raise FloraBenchmarkError("flora_benchmark_replay_report_conflict")
        return existing
    report = build_flora_benchmark_report(dataset=dataset, profile=profile, receipts=receipts,
                                         elapsed_seconds=time.monotonic() - started, raw_artifact_root=raw_artifact_root)
    _write_once(report_path, report)
    text_path = output_dir / "benchmark_report.md"
    with text_path.open("x", encoding="utf-8") as stream:
        stream.write(_markdown(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--session-result", type=Path, required=True)
    prepare.add_argument("--evidence-root", type=Path, required=True)
    prepare.add_argument("--profile", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--episode-id", action="append")
    prepare.add_argument("--corrected-reference", type=Path)
    for name in ("preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--profile", type=Path, required=True)
        command.add_argument("--rights-root", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--raw-artifact-root", type=Path)
    args = parser.parse_args()
    profile = _read(args.profile)
    if args.command == "prepare":
        result = prepare_flora_benchmark(session_result_path=args.session_result,
            evidence_root=args.evidence_root, profile=profile, output_dir=args.output_dir, episode_ids=args.episode_id,
            corrected_reference_path=args.corrected_reference)
    else:
        result = run_flora_benchmark(manifest_path=args.manifest, profile=profile,
            rights_root=args.rights_root, output_dir=args.output_dir, execute=args.command == "run",
            raw_artifact_root=args.raw_artifact_root)
    print(json.dumps({"schema_version": result["schema_version"], "status": result.get("status", "prepared"),
                      "output_dir": str(args.output_dir), "blockers": result.get("blockers", [])}, sort_keys=True))
    return 2 if result.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
