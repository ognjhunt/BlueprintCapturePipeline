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
    "artifact_purpose": "post_training_data_package_export_manifest",
    "export_manifest_only": True,
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
    optional_exports = {
        "schema_version": "post_training_data_package_optional_exports.v1",
        "generated_at": generated_at,
        "formats": _optional_export_formats(),
    }
    write_json(output_dir / "dataset_card.json", dataset_card)
    write_json(output_dir / "license_manifest.json", license_manifest)
    write_json(output_dir / "optional_export_manifest.json", optional_exports)
    package_index = {
        "schema_version": "post_training_data_package_index.v1",
        "generated_at": generated_at,
        "files": {
            "attempts_jsonl": "data/attempts.jsonl",
            "failure_labels_jsonl": "data/failure_labels.jsonl",
            "metrics_json": "data/metrics.json",
            "clips_manifest": "clips_manifest.json",
            "dataset_card": "dataset_card.json",
            "license_manifest": "license_manifest.json",
            "optional_export_manifest": "optional_export_manifest.json",
        },
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
            ("robot_pov_observation_manifest", "robot_pov_observation_manifest.json"),
            ("robot_pov_observations", "robot_pov_observations.jsonl"),
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
            ("deployment_outcome_ledger", "deployment_outcome_ledger.json"),
            ("sim_vs_real_calibration_report", "sim_vs_real_calibration_report.json"),
            (
                "prediction_vs_actual_deployment_summary",
                "prediction_vs_actual_deployment_summary.json",
            ),
            ("breakage_library", "breakage_library.json"),
            ("evaluation_result", "evaluation_result.json"),
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
            "deployment_outcomes_included": "deployment_outcome_ledger" in included_artifacts,
            "breakage_library_included": "breakage_library" in included_artifacts,
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
