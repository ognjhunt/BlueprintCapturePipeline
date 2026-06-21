from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

import blueprint_pipeline.post_training_data_package as package_module
from blueprint_pipeline.post_training_data_package import (
    CLAIM_BOUNDARY,
    _artifact,
    _read_optional_mapping,
    _rows,
    _write_native_hdf5,
    _write_native_parquet,
    build_post_training_data_package_export,
    main,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "storage" / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    (capture_root / "raw").mkdir(parents=True)
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    _write_json(
        capture_root / "raw" / "manifest.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    return capture_root


def _seed_required_pipeline_artifacts(capture_root: Path) -> None:
    dataset_root = capture_root / "pipeline" / "robot_eval_dataset"
    for name in (
        "site_card.json",
        "task_cards.json",
        "scenario_cards.json",
        "eval_cards.json",
        "proof_boundaries.json",
    ):
        _write_json(dataset_root / name, {"name": name})


def _seed_ready_job(job_dir: Path) -> None:
    _write_json(
        job_dir / "normalized_attempt_trace.json",
        {
            "attempt_count": 1,
            "attempts": [
                {
                    "attempt_id": "attempt-1",
                    "scenario_id": "scenario-1",
                    "task_id": "task-1",
                    "policy_id": "policy-1",
                    "success": True,
                    "status": "passed",
                    "metrics": {"score": 1.0},
                    "action_trace": [{"joint": "arm"}],
                    "observation_refs": [{"frame": "000001"}],
                }
            ],
        },
    )
    _write_json(
        job_dir / "failure_labels.json",
        {
            "label_count": 1,
            "labels": [
                {"attempt_id": "attempt-1", "label": "nominal"},
                {"scenario_id": "scenario-1", "label": "scenario-only"},
            ],
        },
    )
    _write_json(job_dir / "arena_eval_metrics.json", {"score": 1.0, "attempt_count": 1})
    _write_json(job_dir / "clips_manifest.json", {"clip_count": 1, "clips": ["clip-1.mp4"]})
    for name in (
        "prediction_outcome_ledger.json",
        "calibration_report.json",
        "breakage_library.json",
        "visual_review_ledger.json",
        "simulator_provider_adapter_manifest.json",
        "simulator_command_batch_attempt_trace.jsonl",
        "simulator_command_batch_contact_stream.jsonl",
        "simulator_command_batch_planner_state.jsonl",
        "simulator_command_batch_control_stream.jsonl",
        "sim_vs_real_calibration_report.json",
        "deployment_outcome_intake_manifest.json",
        "deployment_outcome_ledger.json",
        "real_world_validation_followup_plan.json",
        "real_world_validation_followup_request_queue.json",
        "live_eval_closure_manifest.json",
        "robot_eval_report.json",
    ):
        path = job_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")


def test_post_training_data_package_blocks_with_manifest_only_defaults(tmp_path: Path) -> None:
    capture_root = _capture_root(tmp_path)

    manifest = build_post_training_data_package_export(capture_root=capture_root)

    assert manifest["status"] == "blocked_missing_inputs"
    assert "missing_normalized_attempt_trace" in manifest["blockers"]
    output_dir = capture_root / "pipeline" / "post_training_data_package"
    assert (output_dir / "data" / "attempts.jsonl").is_file()
    metrics = json.loads((output_dir / "data" / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "missing_source_metrics"
    clips = json.loads((output_dir / "clips_manifest.json").read_text(encoding="utf-8"))
    assert clips["status"] == "missing_source_clips"
    optional = json.loads((output_dir / "optional_export_manifest.json").read_text(encoding="utf-8"))
    assert optional["formats"]["video_bundle"]["clip_count"] == 0
    assert manifest["claim_boundary"] == CLAIM_BOUNDARY


def test_post_training_data_package_exports_ready_package_with_policy_flags(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = _capture_root(tmp_path)
    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    output_dir = tmp_path / "package"

    def _write_fake_hdf5(path: Path, _rows_arg: object) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake-hdf5", encoding="utf-8")
        return True

    def _write_fake_parquet(path: Path, _rows_arg: object) -> bool:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake-parquet", encoding="utf-8")
        return True

    monkeypatch.setattr(package_module, "_write_native_hdf5", _write_fake_hdf5)
    monkeypatch.setattr(package_module, "_write_native_parquet", _write_fake_parquet)

    manifest = build_post_training_data_package_export(
        capture_root=capture_root,
        job_dir=job_dir,
        output_dir=output_dir,
    )

    assert manifest["status"] == "export_ready_review_required"
    assert manifest["blockers"] == []
    assert manifest["manifest_counts"] == {
        "attempt_count": 1,
        "failure_label_count": 1,
        "clip_count": 1,
    }
    assert manifest["export_policy"]["simulator_command_batch_trace_streams_included"] is True
    assert manifest["export_policy"]["deployment_outcomes_included"] is True
    assert manifest["export_policy"]["real_world_validation_followup_queue_included"] is True
    optional = json.loads((output_dir / "optional_export_manifest.json").read_text(encoding="utf-8"))
    assert optional["formats"]["hdf5"]["status"] == "written_native"
    assert optional["formats"]["parquet"]["status"] == "written_native"
    archive_members = json.loads((output_dir / "archive_manifest.json").read_text(encoding="utf-8"))
    assert "exports/rlds/episodes.jsonl" in archive_members["included_files"]


def test_post_training_data_package_main_returns_status_codes(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:  # type: ignore[no-untyped-def]
    capture_root = _capture_root(tmp_path)
    blocked_output = tmp_path / "blocked"

    assert main(["--capture-root", str(capture_root), "--output-dir", str(blocked_output)]) == 1
    assert "status=blocked_missing_inputs" in capsys.readouterr().out

    _seed_required_pipeline_artifacts(capture_root)
    job_dir = tmp_path / "job"
    _seed_ready_job(job_dir)
    ready_output = tmp_path / "ready"
    monkeypatch.setattr(package_module, "_write_native_hdf5", lambda path, rows: False)
    monkeypatch.setattr(package_module, "_write_native_parquet", lambda path, rows: False)

    assert (
        main(
            [
                "--capture-root",
                str(capture_root),
                "--job-dir",
                str(job_dir),
                "--output-dir",
                str(ready_output),
            ]
        )
        == 0
    )
    assert "status=export_ready_review_required" in capsys.readouterr().out


def test_post_training_data_package_private_helpers_cover_optional_edges(
    monkeypatch,
    tmp_path: Path,
) -> None:
    assert _read_optional_mapping(tmp_path / "missing.json") == {}
    non_mapping = tmp_path / "list.json"
    non_mapping.write_text("[]", encoding="utf-8")
    assert _read_optional_mapping(non_mapping) == {}
    assert _rows({"rows": [{"ok": True}, "skip"]}, "rows") == [{"ok": True}]
    assert _rows({"rows": "not-a-list"}, "rows") == []

    missing_artifact = _artifact(tmp_path, tmp_path / "missing.file")
    assert missing_artifact["exists"] is False
    assert missing_artifact["sha256"] is None

    monkeypatch.setitem(sys.modules, "h5py", None)
    assert _write_native_hdf5(tmp_path / "fallback" / "episodes.hdf5", []) is False

    class _FakeH5File:
        def __init__(self, path: Path, mode: str) -> None:
            self.path = path
            self.mode = mode
            self.attrs: dict[str, object] = {}

        def __enter__(self) -> "_FakeH5File":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("fake", encoding="utf-8")
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def create_dataset(self, _name: str, *, data: object, dtype: object) -> None:
            self.attrs["dataset"] = {"data": data, "dtype": dtype}

    monkeypatch.setitem(
        sys.modules,
        "h5py",
        types.SimpleNamespace(
            File=_FakeH5File,
            string_dtype=lambda encoding: f"string:{encoding}",
        ),
    )
    assert _write_native_hdf5(tmp_path / "native" / "episodes.hdf5", [{"episode": 1}]) is True

    monkeypatch.setattr(package_module.importlib.util, "find_spec", lambda _name: None)
    assert _write_native_parquet(tmp_path / "fallback" / "episodes.parquet", []) is False

    class _FakeDataFrame:
        def __init__(self, rows: object) -> None:
            self.rows = rows

        def to_parquet(self, path: Path, *, index: bool) -> None:
            assert index is False
            path.write_text(json.dumps(self.rows), encoding="utf-8")

    monkeypatch.setattr(package_module.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setitem(
        sys.modules,
        "pandas",
        types.SimpleNamespace(DataFrame=lambda rows: _FakeDataFrame(rows)),
    )
    assert _write_native_parquet(tmp_path / "native" / "episodes.parquet", [{"episode_id": "e"}]) is True
