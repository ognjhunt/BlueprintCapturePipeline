"""Post-Training Data Package export, checksum, and archive builder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tarfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .common import ensure_dir, read_json_any, utc_now_iso, write_json
from .local_capture import resolve_local_capture_context


POST_TRAINING_DATA_PACKAGE_EXPORT_SCHEMA_VERSION = "post_training_data_package_export.v1"

CLAIM_BOUNDARY: Dict[str, Any] = {
    "artifact_purpose": "post_training_data_package_export",
    "export_manifest_only": False,
    "export_files_written": True,
    "simulator_execution_proven": False,
    "robot_policy_execution_proven": False,
    "robot_readiness_proven": False,
    "training_completed": False,
    "public_claim_upgrade_allowed": False,
}


def _read_optional_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_any(path)
    return dict(payload) if isinstance(payload, Mapping) else {}


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(payload: Mapping[str, Any], key: str) -> List[Dict[str, Any]]:
    values = payload.get(key)
    if isinstance(values, list):
        return [dict(item) for item in values if isinstance(item, Mapping)]
    return []


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    content = "\n".join(json.dumps(dict(row), sort_keys=True) for row in rows)
    if content:
        content += "\n"
    path.write_text(content, encoding="utf-8")


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to(base_dir: Path, target: Path) -> str:
    return os.path.relpath(target.resolve(), start=base_dir.resolve()).replace("\\", "/")


def _artifact(base_dir: Path, path: Path) -> Dict[str, Any]:
    return {
        "path": _relative_to(base_dir, path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": _sha_file(path) if path.is_file() else None,
    }


def _job_artifact(job_dir: Path, name: str) -> str | None:
    path = job_dir / name
    return path.name if path.is_file() else None


def _pipeline_artifact(pipeline_dir: Path, relative_path: str) -> str | None:
    path = pipeline_dir / relative_path
    return relative_path if path.is_file() else None


def _optional_export_formats() -> Dict[str, Dict[str, Any]]:
    formats: Dict[str, Dict[str, Any]] = {}
    dependencies = {
        "rlds": ("rlds",),
        "lerobot": ("lerobot",),
        "hdf5": ("h5py",),
        "parquet": ("pyarrow", "pandas"),
    }
    for name, packages in dependencies.items():
        available = all(importlib.util.find_spec(package) is not None for package in packages)
        formats[name] = {
            "status": "available_not_written" if available else "blocked_optional_dependency_missing",
            "dependencies": list(packages),
            "format_written": False,
        }
    formats["video_bundle"] = {
        "status": "degraded_manifest_only",
        "dependencies": ["clips_manifest.json", "clip files when present"],
        "format_written": False,
    }
    return formats


def _rows_for_optional_exports(
    *,
    attempts: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    scene_id: str,
    capture_id: str,
) -> List[Dict[str, Any]]:
    labels_by_attempt: Dict[str, List[Dict[str, Any]]] = {}
    labels_by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for label in label_rows:
        label_payload = dict(label)
        attempt_id = str(label.get("attempt_id") or "").strip()
        scenario_id = str(label.get("scenario_id") or "").strip()
        if attempt_id:
            labels_by_attempt.setdefault(attempt_id, []).append(label_payload)
        if scenario_id:
            labels_by_scenario.setdefault(scenario_id, []).append(label_payload)

    rows: List[Dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        attempt_id = str(attempt.get("attempt_id") or f"attempt_{index}").strip()
        scenario_id = str(attempt.get("scenario_id") or "").strip()
        labels = labels_by_attempt.get(attempt_id) or labels_by_scenario.get(scenario_id) or []
        rows.append(
            {
                "episode_id": attempt_id,
                "episode_index": index - 1,
                "scene_id": scene_id,
                "capture_id": capture_id,
                "task_id": attempt.get("task_id"),
                "scenario_id": scenario_id or None,
                "policy_id": attempt.get("policy_id"),
                "success": bool(attempt.get("success")),
                "status": attempt.get("status") or "unknown",
                "metrics": dict(_mapping(attempt.get("metrics"))),
                "actions": attempt.get("actions") or attempt.get("action_trace") or [],
                "observations": attempt.get("observations") or attempt.get("observation_refs") or [],
                "failure_labels": labels,
                "package_metrics": dict(metrics),
                "source_format": "blueprint_normalized_attempt_trace.v1",
                "claim_boundary": dict(CLAIM_BOUNDARY),
            }
        )
    if rows:
        return rows
    return [
        {
            "episode_id": "missing_attempts",
            "episode_index": 0,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "task_id": None,
            "scenario_id": None,
            "policy_id": None,
            "success": False,
            "status": "missing_source_attempts",
            "metrics": dict(metrics),
            "actions": [],
            "observations": [],
            "failure_labels": list(label_rows),
            "package_metrics": dict(metrics),
            "source_format": "blueprint_normalized_attempt_trace.v1",
            "claim_boundary": dict(CLAIM_BOUNDARY),
        }
    ]


def _flat_export_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "episode_id": row.get("episode_id"),
                "episode_index": row.get("episode_index"),
                "scene_id": row.get("scene_id"),
                "capture_id": row.get("capture_id"),
                "task_id": row.get("task_id"),
                "scenario_id": row.get("scenario_id"),
                "policy_id": row.get("policy_id"),
                "success": bool(row.get("success")),
                "status": row.get("status"),
                "metrics_json": json.dumps(row.get("metrics") or {}, sort_keys=True),
                "actions_json": json.dumps(row.get("actions") or [], sort_keys=True),
                "observations_json": json.dumps(row.get("observations") or [], sort_keys=True),
                "failure_labels_json": json.dumps(
                    row.get("failure_labels") or [],
                    sort_keys=True,
                ),
            }
        )
    return out


def _write_native_hdf5(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError:
        return False
    ensure_dir(path.parent)
    payloads = [json.dumps(dict(row), sort_keys=True) for row in rows]
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "blueprint_post_training_hdf5.v1"
        handle.attrs["source_format"] = "blueprint_normalized_attempt_trace.v1"
        string_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset("episodes_json", data=payloads, dtype=string_dtype)
    return True


def _write_native_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    if importlib.util.find_spec("pyarrow") is None or importlib.util.find_spec("pandas") is None:
        return False
    import pandas as pd  # type: ignore[import-not-found]

    ensure_dir(path.parent)
    pd.DataFrame(_flat_export_rows(rows)).to_parquet(path, index=False)
    return True


def _write_optional_exports(
    *,
    output_dir: Path,
    attempts: Sequence[Mapping[str, Any]],
    label_rows: Sequence[Mapping[str, Any]],
    metrics: Mapping[str, Any],
    clips: Mapping[str, Any],
    generated_at: str,
    scene_id: str,
    capture_id: str,
) -> Dict[str, Any]:
    rows = _rows_for_optional_exports(
        attempts=attempts,
        label_rows=label_rows,
        metrics=metrics,
        scene_id=scene_id,
        capture_id=capture_id,
    )
    files: Dict[str, str] = {}
    formats = _optional_export_formats()

    rlds_path = output_dir / "exports" / "rlds" / "episodes.jsonl"
    _write_jsonl(rlds_path, rows)
    files["rlds_episodes"] = _relative_to(output_dir, rlds_path)
    formats["rlds"] = {
        **formats["rlds"],
        "status": "written_jsonl",
        "format_written": True,
        "path": files["rlds_episodes"],
        "episode_count": len(rows),
        "native_package_required": False,
    }

    lerobot_rows = [
        {
            "episode_index": row.get("episode_index"),
            "episode_id": row.get("episode_id"),
            "task": row.get("task_id"),
            "scenario": row.get("scenario_id"),
            "observation": row.get("observations") or [],
            "action": row.get("actions") or [],
            "reward_or_success": 1.0 if row.get("success") else 0.0,
            "metadata": {
                "scene_id": scene_id,
                "capture_id": capture_id,
                "policy_id": row.get("policy_id"),
                "failure_labels": row.get("failure_labels") or [],
                "claim_boundary": dict(CLAIM_BOUNDARY),
            },
        }
        for row in rows
    ]
    lerobot_path = output_dir / "exports" / "lerobot" / "episodes.jsonl"
    _write_jsonl(lerobot_path, lerobot_rows)
    files["lerobot_episodes"] = _relative_to(output_dir, lerobot_path)
    formats["lerobot"] = {
        **formats["lerobot"],
        "status": "written_jsonl",
        "format_written": True,
        "path": files["lerobot_episodes"],
        "episode_count": len(lerobot_rows),
        "native_package_required": False,
    }

    hdf5_path = output_dir / "exports" / "hdf5" / "episodes.hdf5"
    if _write_native_hdf5(hdf5_path, rows):
        files["hdf5_episodes"] = _relative_to(output_dir, hdf5_path)
        formats["hdf5"] = {
            **formats["hdf5"],
            "status": "written_native",
            "format_written": True,
            "path": files["hdf5_episodes"],
            "episode_count": len(rows),
        }
    else:
        hdf5_fallback = output_dir / "exports" / "hdf5" / "episodes.hdf5.jsonl"
        _write_jsonl(hdf5_fallback, rows)
        files["hdf5_episodes"] = _relative_to(output_dir, hdf5_fallback)
        formats["hdf5"] = {
            **formats["hdf5"],
            "status": "written_jsonl_fallback",
            "format_written": True,
            "path": files["hdf5_episodes"],
            "episode_count": len(rows),
            "fallback_reason": "optional_dependency_h5py_missing",
        }

    parquet_path = output_dir / "exports" / "parquet" / "episodes.parquet"
    if _write_native_parquet(parquet_path, rows):
        files["parquet_episodes"] = _relative_to(output_dir, parquet_path)
        formats["parquet"] = {
            **formats["parquet"],
            "status": "written_native",
            "format_written": True,
            "path": files["parquet_episodes"],
            "episode_count": len(rows),
        }
    else:
        parquet_fallback = output_dir / "exports" / "parquet" / "episodes.parquet.jsonl"
        _write_jsonl(parquet_fallback, _flat_export_rows(rows))
        files["parquet_episodes"] = _relative_to(output_dir, parquet_fallback)
        formats["parquet"] = {
            **formats["parquet"],
            "status": "written_jsonl_fallback",
            "format_written": True,
            "path": files["parquet_episodes"],
            "episode_count": len(rows),
            "fallback_reason": "optional_dependency_pyarrow_or_pandas_missing",
        }

    video_bundle_path = output_dir / "exports" / "video_bundle" / "clips_manifest.json"
    write_json(
        video_bundle_path,
        {
            "schema_version": "post_training_video_bundle_manifest.v1",
            "generated_at": generated_at,
            "status": "written_manifest",
            "source_clips": dict(clips),
            "clip_count": int(clips.get("clip_count") or 0) if clips else 0,
            "claim_boundary": dict(CLAIM_BOUNDARY),
        },
    )
    files["video_bundle_manifest"] = _relative_to(output_dir, video_bundle_path)
    formats["video_bundle"] = {
        **formats["video_bundle"],
        "status": "written_manifest",
        "format_written": True,
        "path": files["video_bundle_manifest"],
        "clip_count": int(clips.get("clip_count") or 0) if clips else 0,
    }

    return {
        "schema_version": "post_training_data_package_optional_exports.v1",
        "generated_at": generated_at,
        "formats": formats,
        "files": files,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _write_package_files(
    *,
    output_dir: Path,
    included_artifacts: Mapping[str, str],
    trace: Mapping[str, Any],
    labels: Mapping[str, Any],
    metrics: Mapping[str, Any],
    clips: Mapping[str, Any],
    generated_at: str,
    scene_id: str,
    capture_id: str,
) -> Dict[str, Any]:
    data_dir = output_dir / "data"
    attempts = _rows(trace, "attempts")
    label_rows = _rows(labels, "labels")
    _write_jsonl(data_dir / "attempts.jsonl", attempts)
    _write_jsonl(data_dir / "failure_labels.jsonl", label_rows)
    write_json(
        data_dir / "metrics.json",
        dict(metrics)
        if metrics
        else {
            "schema_version": "post_training_package_metrics.v1",
            "generated_at": generated_at,
            "status": "missing_source_metrics",
            "attempt_count": len(attempts),
            "failure_count": len(label_rows),
        },
    )
    write_json(
        output_dir / "clips_manifest.json",
        dict(clips)
        if clips
        else {
            "schema_version": "post_training_package_clips_manifest.v1",
            "generated_at": generated_at,
            "status": "missing_source_clips",
            "clip_count": 0,
            "clips": [],
        },
    )
    dataset_card = {
        "schema_version": "post_training_data_package_dataset_card.v1",
        "generated_at": generated_at,
        "scene_id": scene_id,
        "capture_id": capture_id,
        "dataset_type": "real_site_robot_eval_post_training_package",
        "attempt_count": len(attempts),
        "failure_label_count": len(label_rows),
        "source_artifacts": dict(included_artifacts),
        "proof_boundary": dict(CLAIM_BOUNDARY),
    }
    license_manifest = {
        "schema_version": "post_training_data_package_license_manifest.v1",
        "generated_at": generated_at,
        "status": "review_required",
        "rights_privacy_review_required": True,
        "commercial_use_requires_package_scope_clearance": True,
        "included_artifacts": dict(included_artifacts),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    optional_exports = _write_optional_exports(
        output_dir=output_dir,
        attempts=attempts,
        label_rows=label_rows,
        metrics=metrics,
        clips=clips,
        generated_at=generated_at,
        scene_id=scene_id,
        capture_id=capture_id,
    )
    write_json(output_dir / "dataset_card.json", dataset_card)
    write_json(output_dir / "license_manifest.json", license_manifest)
    write_json(output_dir / "optional_export_manifest.json", optional_exports)
    package_file_index = {
        "attempts_jsonl": "data/attempts.jsonl",
        "failure_labels_jsonl": "data/failure_labels.jsonl",
        "metrics_json": "data/metrics.json",
        "clips_manifest": "clips_manifest.json",
        "dataset_card": "dataset_card.json",
        "license_manifest": "license_manifest.json",
        "optional_export_manifest": "optional_export_manifest.json",
        **dict(optional_exports.get("files") or {}),
    }
    package_index = {
        "schema_version": "post_training_data_package_index.v1",
        "generated_at": generated_at,
        "files": package_file_index,
        "source_artifacts": dict(included_artifacts),
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "package_index.json", package_index)
    package_files = {
        key: _artifact(output_dir, output_dir / path)
        for key, path in package_index["files"].items()
    }
    checksums = {
        "schema_version": "post_training_data_package_checksums.v1",
        "generated_at": generated_at,
        "files": package_files,
    }
    write_json(output_dir / "checksums.json", checksums)
    return {
        "dataset_card": dataset_card,
        "license_manifest": license_manifest,
        "package_index": package_index,
        "checksums": checksums,
        "package_files": package_files,
    }


def _write_archive(output_dir: Path, generated_at: str) -> Dict[str, Any]:
    archive_dir = output_dir / "archives"
    ensure_dir(archive_dir)
    archive_path = archive_dir / "post_training_data_package.tar.gz"
    archive_inputs = [
        output_dir / "data" / "attempts.jsonl",
        output_dir / "data" / "failure_labels.jsonl",
        output_dir / "data" / "metrics.json",
        output_dir / "clips_manifest.json",
        output_dir / "dataset_card.json",
        output_dir / "license_manifest.json",
        output_dir / "optional_export_manifest.json",
        output_dir / "package_index.json",
        output_dir / "checksums.json",
    ]
    exports_dir = output_dir / "exports"
    if exports_dir.is_dir():
        archive_inputs.extend(
            path for path in sorted(exports_dir.rglob("*")) if path.is_file()
        )
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in archive_inputs:
            if path.is_file():
                tar.add(path, arcname=_relative_to(output_dir, path))
    archive_manifest = {
        "schema_version": "post_training_data_package_archive_manifest.v1",
        "generated_at": generated_at,
        "status": "created",
        "archive": _artifact(output_dir, archive_path),
        "included_files": [
            _relative_to(output_dir, path) for path in archive_inputs if path.is_file()
        ],
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(output_dir / "archive_manifest.json", archive_manifest)
    return archive_manifest


def build_post_training_data_package_export(
    *,
    capture_root: str | Path,
    job_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    pipeline_dir = context.pipeline_root
    resolved_job_dir = Path(job_dir).resolve() if job_dir else None
    resolved_output_dir = (
        Path(output_dir).resolve()
        if output_dir
        else resolved_job_dir
        if resolved_job_dir
        else pipeline_dir / "post_training_data_package"
    )
    ensure_dir(resolved_output_dir)
    generated_at = utc_now_iso()

    included_artifacts: Dict[str, str] = {}
    if resolved_job_dir:
        for key, name in (
            ("normalized_attempt_trace", "normalized_attempt_trace.json"),
            ("failure_labels", "failure_labels.json"),
            ("arena_eval_metrics", "arena_eval_metrics.json"),
            ("simulator_provider_adapter_manifest", "simulator_provider_adapter_manifest.json"),
            ("simulator_command_artifacts_manifest", "simulator_command_artifacts_manifest.json"),
            ("scenario_eval_matrix", "scenario_eval_matrix.json"),
            ("robot_pov_observation_manifest", "robot_pov_observation_manifest.json"),
            ("robot_pov_observations", "robot_pov_observations.jsonl"),
            ("robot_pov_frame_sequence_manifest", "robot_pov_frame_sequence_manifest.json"),
            ("robot_pov_render_storyboard", "robot_pov_render_storyboard.json"),
            ("policy_execution_manifest", "policy_execution_manifest.json"),
            ("policy_execution_trace", "policy_execution_trace.json"),
            ("clips_manifest", "clips_manifest.json"),
            ("accepted_failure_labels", "accepted_failure_labels.json"),
            ("review_resolution_ledger", "review_resolution_ledger.json"),
            ("customer_handoff_report", "customer_handoff_report.json"),
            ("delivery_manifest", "delivery_manifest.json"),
            ("live_operator_ledger", "live_operator_ledger.json"),
            ("arena_rerun_plan", "arena_rerun_plan.json"),
            ("policy_adapter_manifest", "policy_adapter_manifest.json"),
            ("arena_result_ingest_ledger", "arena_result_ingest_ledger.json"),
            ("prediction_outcome_ledger", "prediction_outcome_ledger.json"),
            ("calibration_report", "calibration_report.json"),
            ("deployment_outcome_intake_manifest", "deployment_outcome_intake_manifest.json"),
            ("deployment_outcome_ledger", "deployment_outcome_ledger.json"),
            ("sim_vs_real_calibration_report", "sim_vs_real_calibration_report.json"),
            (
                "prediction_vs_actual_deployment_summary",
                "prediction_vs_actual_deployment_summary.json",
            ),
            ("live_eval_closure_manifest", "live_eval_closure_manifest.json"),
            ("breakage_library", "breakage_library.json"),
            ("evaluation_result", "evaluation_result.json"),
            ("robot_eval_report", "robot_eval_report.json"),
            ("robot_eval_report_markdown", "robot_eval_report.md"),
            ("proof_boundary", "proof_boundary.json"),
        ):
            value = _job_artifact(resolved_job_dir, name)
            if value:
                included_artifacts[key] = value

    for key, relative_path in (
        ("site_card", "robot_eval_dataset/site_card.json"),
        ("task_cards", "robot_eval_dataset/task_cards.json"),
        ("scenario_cards", "robot_eval_dataset/scenario_cards.json"),
        ("eval_cards", "robot_eval_dataset/eval_cards.json"),
        ("rights_packet", "robot_eval_dataset/rights_packet.json"),
        ("proof_boundaries", "robot_eval_dataset/proof_boundaries.json"),
        ("robot_eval_dataset_manifest", "robot_eval_dataset/robot_eval_dataset_manifest.json"),
        ("worldlabs_export_manifest", "worldlabs_export_manifest.json"),
        ("arena_environment_packet", "simulation_automation/arena_environment_packet.json"),
        ("gpu_handoff_packet", "simulation_automation/gpu_handoff_packet.json"),
    ):
        value = _pipeline_artifact(pipeline_dir, relative_path)
        if value:
            included_artifacts[key] = _relative_to(resolved_output_dir, pipeline_dir / value)

    required = (
        "normalized_attempt_trace",
        "failure_labels",
        "prediction_outcome_ledger",
        "calibration_report",
        "breakage_library",
        "site_card",
        "task_cards",
        "scenario_cards",
        "eval_cards",
        "proof_boundaries",
    )
    missing = [key for key in required if key not in included_artifacts]
    status = "blocked_missing_inputs" if missing else "export_ready_review_required"
    trace = (
        _read_optional_mapping(resolved_job_dir / "normalized_attempt_trace.json")
        if resolved_job_dir
        else {}
    )
    labels = (
        _read_optional_mapping(resolved_job_dir / "failure_labels.json")
        if resolved_job_dir
        else {}
    )
    metrics = (
        _read_optional_mapping(resolved_job_dir / "arena_eval_metrics.json")
        if resolved_job_dir
        else {}
    )
    clips = (
        _read_optional_mapping(resolved_job_dir / "clips_manifest.json")
        if resolved_job_dir
        else {}
    )
    package_files = _write_package_files(
        output_dir=resolved_output_dir,
        included_artifacts=included_artifacts,
        trace=trace,
        labels=labels,
        metrics=metrics,
        clips=clips,
        generated_at=generated_at,
        scene_id=context.scene_id,
        capture_id=context.capture_id,
    )
    archive_manifest = _write_archive(resolved_output_dir, generated_at)

    manifest = {
        "schema_version": POST_TRAINING_DATA_PACKAGE_EXPORT_SCHEMA_VERSION,
        "generated_at": generated_at,
        "scene_id": context.scene_id,
        "capture_id": context.capture_id,
        "package_type": "post_training_data_package",
        "status": status,
        "blockers": [f"missing_{key}" for key in missing],
        "included_artifacts": included_artifacts,
        "manifest_counts": {
            "attempt_count": int(trace.get("attempt_count") or 0),
            "failure_label_count": int(labels.get("label_count") or 0),
            "clip_count": int(clips.get("clip_count") or 0),
        },
        "export_policy": {
            "curated_robot_pov_clips_required_for_richer_exports": True,
            "robot_pov_observations_included": "robot_pov_observation_manifest"
            in included_artifacts,
            "scenario_eval_matrix_included": "scenario_eval_matrix" in included_artifacts,
            "policy_execution_trace_included": "policy_execution_trace" in included_artifacts,
            "normalized_eval_attempts_included": "normalized_attempt_trace" in included_artifacts,
            "failure_labels_included": "failure_labels" in included_artifacts,
            "arena_metrics_included": bool(metrics),
            "clips_manifest_included": bool(clips),
            "calibration_included": "calibration_report" in included_artifacts,
            "simulator_provider_adapter_included": "simulator_provider_adapter_manifest"
            in included_artifacts,
            "sim_vs_real_calibration_included": "sim_vs_real_calibration_report"
            in included_artifacts,
            "deployment_outcome_intake_included": "deployment_outcome_intake_manifest"
            in included_artifacts,
            "deployment_outcomes_included": "deployment_outcome_ledger" in included_artifacts,
            "live_eval_closure_included": "live_eval_closure_manifest"
            in included_artifacts,
            "breakage_library_included": "breakage_library" in included_artifacts,
            "robot_eval_report_included": "robot_eval_report" in included_artifacts,
        },
        "package_files": package_files["package_files"],
        "dataset_card_path": "dataset_card.json",
        "license_manifest_path": "license_manifest.json",
        "package_index_path": "package_index.json",
        "checksums_path": "checksums.json",
        "archive_manifest_path": "archive_manifest.json",
        "archive": archive_manifest["archive"],
        "optional_export_manifest_path": "optional_export_manifest.json",
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }
    write_json(resolved_output_dir / "post_training_data_package_export_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a Post-Training Data Package export, checksum, and archive"
    )
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--job-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    result = build_post_training_data_package_export(
        capture_root=args.capture_root,
        job_dir=args.job_dir,
        output_dir=args.output_dir,
    )
    default_output_dir = Path(args.capture_root) / "pipeline" / "post_training_data_package"
    manifest_dir = Path(args.output_dir or args.job_dir or default_output_dir)
    manifest_path = manifest_dir / "post_training_data_package_export_manifest.json"
    print(f"[post-training-data-package] manifest={manifest_path}")
    print(f"[post-training-data-package] status={result['status']}")
    return 0 if result["status"] == "export_ready_review_required" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
