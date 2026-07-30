from __future__ import annotations

import json
import hashlib
import zipfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from blueprint_pipeline import ctrl_world_provider_bundle as bundle
from blueprint_pipeline import ctrl_world_provider_runtime_runner as runtime
from blueprint_pipeline import policy_ranking_successor_gpu_admission as admission


def _source_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "ctrl_world_source"
    for relative in bundle.SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith(".json"):
            if relative.endswith("899.json"):
                path.write_text(
                    json.dumps(
                        {
                            "texts": ["Move the banana to the right"],
                            "states": [[0.0] * 7] * 20,
                            "joints": [[0.0] * 8] * 20,
                            "videos": [],
                        }
                    ),
                    encoding="utf-8",
                )
            else:
                path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_bytes(f"fixture:{relative}\n".encode())
    return root


def test_ctrl_world_bundle_is_deterministic_and_passes_frozen_inspection(
    tmp_path: Path, monkeypatch
) -> None:
    source = _source_fixture(tmp_path)
    monkeypatch.setattr(bundle, "_source_commit", lambda _: bundle.CTRL_WORLD_SOURCE_REVISION)
    monkeypatch.setattr(bundle, "_source_status", lambda _: "")

    first = bundle.build_ctrl_world_provider_bundle(
        job_dir=tmp_path / "first",
        ctrl_world_source_dir=source,
        generated_at="2026-07-30T00:00:00+00:00",
    )
    second = bundle.build_ctrl_world_provider_bundle(
        job_dir=tmp_path / "second",
        ctrl_world_source_dir=source,
        generated_at="2026-07-30T00:00:00+00:00",
    )

    assert first["status"] == "completed"
    assert first["attribution"] == "Ctrl-World_not_OSCAR_not_Cosmos"
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["bundle_size_bytes"] == second["bundle_size_bytes"]
    receipt = json.loads((tmp_path / "first" / "ctrl_world_replay_bundle_receipt.json").read_text())
    profile = replace(
        admission.CTRL_WORLD_REPLAY_PROFILE,
        expected_bundle_sha256=first["bundle_sha256"],
        expected_bundle_size_bytes=first["bundle_size_bytes"],
        expected_embedded_input_hashes=first["embedded_hashes"],
    )
    inspection = admission.inspect_successor_bundle(
        first["bundle_path"], receipt=receipt, smoke_inventory={}, profile=profile
    )
    assert inspection["status"] == "passed"
    assert inspection["blockers"] == []

    with zipfile.ZipFile(first["bundle_path"]) as archive:
        manifest = json.loads(archive.read("provider_runtime/wam_provider_runtime_manifest.json"))
        rollout = json.loads(archive.read("provider_runtime/wam_rollout_input_manifest.json"))
    assert manifest["model_name"] == "Ctrl-World"
    assert manifest["canary_settings"]["interaction_count"] == 1
    assert rollout["closed_loop"] is False
    assert rollout["physical_outcome_labels_accessed"] is False


def test_ctrl_world_profile_binds_exact_v4_bundle() -> None:
    profile = admission.CTRL_WORLD_REPLAY_PROFILE
    assert profile.authorization_ids_by_allocation_index == {
        8: "policy-ranking-cosmos3-edge-closed-loop-20260729-allocation-8",
        9: "policy-ranking-cosmos3-edge-closed-loop-20260729-allocation-9",
        10: "policy-ranking-cosmos3-edge-closed-loop-20260729-allocation-10",
        11: "policy-ranking-cosmos3-edge-closed-loop-20260729-allocation-11",
    }
    assert profile.expected_bundle_sha256 == (
        "b616118c7ad5b54f98401fb60d3b4ec8d60acd036fc11d3a9568cc8821d4e6de"
    )
    assert profile.expected_bundle_size_bytes == 2_579_223
    assert profile.expected_embedded_input_hashes["runtime_manifest_file_sha256"] == (
        "95fc73992ac7f92329963917b3ada6881838a4ccd155691ed3999722247195bf"
    )
    assert profile.expected_embedded_input_hashes["runner_sha256"] == (
        "f79b63c23dd5e0ae78e8962ffa764b5aaa5d55fe02e37e564d7397a20c532655"
    )
    assert profile.ctrl_world_replay_bundle is True
    assert profile.qualification_canary_request_count == 1
    assert profile.scientific_matrix_request_count == 0


def test_generated_only_redaction_removes_public_physical_comparison(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.mp4"
    writer = cv2.VideoWriter(str(comparison), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (96, 64))
    assert writer.isOpened()
    for frame_index in range(5):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:32, :] = (0, 0, 255)
        frame[32:, :32] = (0, 255, 0)
        frame[32:, 32:64] = (255, 0, 0)
        frame[32:, 64:] = (0, 255, 255)
        frame[32:, :] = np.clip(frame[32:, :] + frame_index, 0, 255)
        writer.write(frame)
    writer.release()

    result = runtime._extract_generated_only_views(
        comparison_video=comparison, output_dir=tmp_path / "output"
    )

    assert result["status"] == "completed"
    assert result["frame_count"] == 5
    assert len(result["media"]) == 4
    assert result["physical_comparison_pixels_removed"] is True
    assert result["public_comparison_video_deleted_after_redaction"] is True
    assert not comparison.exists()
    combined = cv2.VideoCapture(str(tmp_path / "output" / "ctrl_world_generated_three_view.mp4"))
    ok, frame = combined.read()
    combined.release()
    assert ok
    assert frame.shape[:2] == (32, 96)
    assert float(frame[:, :, 2].mean()) < 100.0


def test_runtime_input_validation_accepts_frozen_zero_byte_source(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    source_file = bundle_dir / "provider_runtime" / "ctrl_world_source" / "models" / "__init__.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"")
    manifest = {
        "source_files": [
            {
                "relative_path": "models/__init__.py",
                "size_bytes": 0,
                "sha256": hashlib.sha256(b"").hexdigest(),
            }
        ]
    }

    assert runtime._validate_packaged_inputs(bundle_dir=bundle_dir, manifest=manifest) == []


def test_runtime_input_validation_rejects_invalid_or_mismatched_sizes(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    source_file = bundle_dir / "provider_runtime" / "ctrl_world_source" / "models" / "__init__.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_bytes(b"")
    empty_sha256 = hashlib.sha256(b"").hexdigest()

    for invalid_size in (-1, "0", False, None):
        manifest = {
            "source_files": [
                {
                    "relative_path": "models/__init__.py",
                    "size_bytes": invalid_size,
                    "sha256": empty_sha256,
                }
            ]
        }
        assert runtime._validate_packaged_inputs(bundle_dir=bundle_dir, manifest=manifest) == [
            "ctrl_world_source_file_size_invalid:models/__init__.py"
        ]

    mismatch_manifest = {
        "source_files": [
            {
                "relative_path": "models/__init__.py",
                "size_bytes": 1,
                "sha256": empty_sha256,
            }
        ]
    }
    assert runtime._validate_packaged_inputs(bundle_dir=bundle_dir, manifest=mismatch_manifest) == [
        "ctrl_world_source_file_size_mismatch:models/__init__.py"
    ]


def test_runtime_result_contract_names_ctrl_world_and_preserves_claim_ceiling() -> None:
    text = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "Ctrl-World" in text
    assert "action_conditioned_video_rollout_generated" in text
    assert '"closed_loop_policy_evaluation": False' in text
    assert '"no_policy_ranking_credit": True' in text
    assert '"no_thesis_credit": True' in text
    assert 'model_executed = replay.get("status") == "completed"' in text
