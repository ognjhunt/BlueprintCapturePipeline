from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline import ctrl_world_current_reference_provider_bundle as bundle
from blueprint_pipeline.ctrl_world_current_reference_provider_runtime import (
    EXPECTED_STATE_STAT_SHA256,
    VIEW_ORDER,
)
from blueprint_pipeline.ctrl_world_current_reference_wam import (
    MODEL_FREEZE,
    stage_ctrl_world_current_reference_request,
)
from blueprint_pipeline.droid_ctrl_world_closed_loop_adapter import (
    CTRL_WORLD_SELECTED_HISTORY_INDICES,
)
from blueprint_pipeline.policy_ranking_thesis import file_sha256
from blueprint_pipeline.provider_runtime_bundle_contract import (
    provider_runtime_contract_blockers,
    wam_registered_alternative_inputs_present,
)


def _source(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for relative in bundle.SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")
    return root


def _request(tmp_path: Path) -> Path:
    histories: dict[str, list[dict[str, str]]] = {}
    for view_index, view_id in enumerate(VIEW_ORDER):
        histories[view_id] = []
        for frame_index in range(6):
            path = tmp_path / "frames" / f"v{view_index}_{frame_index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (320, 192), (view_index, frame_index, 0)).save(path)
            histories[view_id].append({"path": str(path), "sha256": file_sha256(path)})
    request: dict[str, Any] = {
        "schema_version": "blueprint_ctrl_world_current_reference_request.v1",
        "query_index": 0,
        "task_prompt": "Pick up the blue block.",
        "view_order": list(VIEW_ORDER),
        "selected_history_views": histories,
        "current_views": {view_id: dict(rows[-1]) for view_id, rows in histories.items()},
        "selected_history_indices": list(CTRL_WORLD_SELECTED_HISTORY_INDICES),
        "action_conditioning_7d": np.zeros((11, 7), dtype=np.float64),
        "action_conditioning_shape": [11, 7],
        "predicted_frame_count": 5,
        "executed_prefix_steps": 8,
        "executed_prefix_seconds": 8 / 15,
        "physical_future_observation_used": False,
    }
    request_dir = tmp_path / "request"
    stage_ctrl_world_current_reference_request(request, output_dir=request_dir, seed=31)
    return request_dir


def test_bundle_is_deterministic_and_contains_no_policy_or_future_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    monkeypatch.setattr(
        bundle, "_source_commit", lambda _: MODEL_FREEZE["ctrl_world_source"]["revision"]
    )
    monkeypatch.setattr(bundle, "_source_status", lambda _: "")
    monkeypatch.setattr(
        bundle,
        "EXPECTED_STATE_STAT_SHA256",
        file_sha256(source / "dataset_meta_info/droid/stat.json"),
    )
    request_dir = _request(tmp_path)

    first = bundle.build_ctrl_world_current_reference_provider_bundle(
        job_dir=tmp_path / "first",
        ctrl_world_source_dir=source,
        staged_request_dir=request_dir,
        generated_at="2026-07-30T00:00:00+00:00",
    )
    second = bundle.build_ctrl_world_current_reference_provider_bundle(
        job_dir=tmp_path / "second",
        ctrl_world_source_dir=source,
        staged_request_dir=request_dir,
        generated_at="2026-07-30T00:00:00+00:00",
    )

    assert first["status"] == "completed"
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["bundle_size_bytes"] == second["bundle_size_bytes"]
    with zipfile.ZipFile(first["bundle_path"]) as archive:
        entries = set(archive.namelist())
        runtime_manifest = json.loads(
            archive.read("provider_runtime/wam_provider_runtime_manifest.json")
        )
        rollout = json.loads(archive.read("provider_runtime/wam_rollout_input_manifest.json"))
    assert "provider_runtime/wam_provider_runtime_runner.py" in entries
    assert "provider_runtime/ctrl_world_provider_runtime_support.py" in entries
    assert not any(entry.endswith(".mp4") for entry in entries)
    assert not any("openpi" in entry.lower() or "policy" in entry.lower() for entry in entries)
    assert runtime_manifest["request_sha256"] == first["request_sha256"]
    assert rollout["physical_future_rgb_provided_to_model"] is False
    assert rollout["candidate_policy_loaded_by_wam_runtime"] is False
    assert rollout["recorded_action_trace_used"] is False
    with zipfile.ZipFile(first["bundle_path"]) as archive:
        entrypoint = archive.read("provider_runtime/run_wam_provider_runtime.sh").decode()
        runner = archive.read("provider_runtime/wam_provider_runtime_runner.py").decode()
    assert wam_registered_alternative_inputs_present(
        bundle_path=Path(first["bundle_path"]), zip_entries=first["zip_entries"]
    )
    assert (
        provider_runtime_contract_blockers(
            provider_bundle_kind="wam", entrypoint_text=entrypoint, runner_text=runner
        )
        == []
    )
    with zipfile.ZipFile(first["bundle_path"]) as archive:
        names = set(archive.namelist())
        runtime_manifest = json.loads(
            archive.read("provider_runtime/wam_provider_runtime_manifest.json")
        )
        rollout_manifest = json.loads(
            archive.read("provider_runtime/wam_rollout_input_manifest.json")
        )
        embedded, blockers = bundle.inspect_ctrl_world_current_reference_archive_inputs(
            archive,
            manifest=runtime_manifest,
            rollout_manifest=rollout_manifest,
            names=names,
        )
    assert blockers == []
    assert embedded == first["embedded_hashes"]


def test_archive_inspector_rejects_request_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    monkeypatch.setattr(
        bundle, "_source_commit", lambda _: MODEL_FREEZE["ctrl_world_source"]["revision"]
    )
    monkeypatch.setattr(bundle, "_source_status", lambda _: "")
    monkeypatch.setattr(
        bundle,
        "EXPECTED_STATE_STAT_SHA256",
        file_sha256(source / "dataset_meta_info/droid/stat.json"),
    )
    built = bundle.build_ctrl_world_current_reference_provider_bundle(
        job_dir=tmp_path / "built",
        ctrl_world_source_dir=source,
        staged_request_dir=_request(tmp_path),
    )
    drifted = tmp_path / "drifted.zip"
    with (
        zipfile.ZipFile(built["bundle_path"]) as source_archive,
        zipfile.ZipFile(drifted, "w") as target,
    ):
        for name in source_archive.namelist():
            value = source_archive.read(name)
            if name.endswith("action_conditioning_11x7.npy"):
                value += b"drift"
            target.writestr(name, value)
    with zipfile.ZipFile(drifted) as archive:
        names = set(archive.namelist())
        runtime_manifest = json.loads(
            archive.read("provider_runtime/wam_provider_runtime_manifest.json")
        )
        rollout_manifest = json.loads(
            archive.read("provider_runtime/wam_rollout_input_manifest.json")
        )
        _embedded, blockers = bundle.inspect_ctrl_world_current_reference_archive_inputs(
            archive,
            manifest=runtime_manifest,
            rollout_manifest=rollout_manifest,
            names=names,
        )
    assert "ctrl_world_current_reference_archive_request_file_hash_mismatch" in blockers


def test_bundle_refuses_nonempty_output_and_wrong_source_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    monkeypatch.setattr(
        bundle,
        "EXPECTED_STATE_STAT_SHA256",
        file_sha256(source / "dataset_meta_info/droid/stat.json"),
    )
    request = _request(tmp_path)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "prior.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="bundle_job_exists"):
        bundle.build_ctrl_world_current_reference_provider_bundle(
            job_dir=occupied,
            ctrl_world_source_dir=source,
            staged_request_dir=request,
        )

    monkeypatch.setattr(bundle, "_source_commit", lambda _: "0" * 40)
    monkeypatch.setattr(bundle, "_source_status", lambda _: "")
    blocked = bundle.build_ctrl_world_current_reference_provider_bundle(
        job_dir=tmp_path / "blocked",
        ctrl_world_source_dir=source,
        staged_request_dir=request,
    )
    assert blocked["status"] == "blocked"
    assert blocked["bundle_present"] is False
    assert "ctrl_world_current_reference_source_revision_mismatch" in blocked["blockers"]


def test_state_stat_freeze_is_full_hash() -> None:
    assert len(EXPECTED_STATE_STAT_SHA256) == 64
