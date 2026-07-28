from __future__ import annotations

import json
import zipfile
from pathlib import Path

from blueprint_pipeline.wam_provider_output import (
    inspect_provider_runtime_output_zip,
    summarize_runtime_result,
)


def test_runtime_summary_preserves_only_explicit_boolean_task_success() -> None:
    assert summarize_runtime_result(
        {"status": "completed", "task_success": True, "blockers": []}
    ) == {
        "status": "completed",
        "blockers": [],
        "action_conditioned_video_rollout_generated": False,
        "generated_rollout_video_present": False,
        "generated_rollout_video_filename": None,
        "repeated_policy_calls_count": None,
        "generated_next_observation_count": None,
        "live_wam_generation_success_count": None,
        "learned_wam_model_success_count": None,
        "policy_observes_wam_generated_next_observation": None,
        "provider_instance_reused_for_policy_and_wam_loop": None,
        "checkpoint_status": None,
        "cuda_probe_status": None,
        "torch_cuda_available": None,
        "cuda_device_count": None,
        "subprocess_status": None,
        "task_success": True,
        "raw_secret_values_recorded": False,
    }
    assert summarize_runtime_result(
        {"status": "completed", "task_success": "true"}
    )["task_success"] is None


def test_provider_output_inspection_is_provider_neutral(tmp_path: Path) -> None:
    output_zip = tmp_path / "provider-output.zip"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "nested/wam_runtime_result.json",
            json.dumps(
                {
                    "status": "completed",
                    "task_success": False,
                    "blockers": [],
                }
            ),
        )

    result = inspect_provider_runtime_output_zip(output_zip)

    assert result["status"] == "completed"
    assert result["runtime_result_status"] == "completed"
    assert result["runtime_result"]["task_success"] is False
    assert result["video_smoke_proven"] is False


def test_runtime_summary_adds_evaluator_fields_only_for_attributable_result() -> None:
    summary = summarize_runtime_result(
        {
            "status": "completed",
            "result_count": 7,
            "error_count": 0,
            "model": "nvidia/Cosmos3-Nano",
            "claim_class": "post_unseal_diagnostic_only",
        }
    )

    assert summary is not None
    assert summary["evaluator_result_count"] == 7
    assert summary["evaluator_error_count"] == 0
    assert summary["evaluator_model"] == "nvidia/Cosmos3-Nano"
    assert summary["claim_class"] == "post_unseal_diagnostic_only"


def test_provider_output_video_probe_is_injected(tmp_path: Path) -> None:
    output_zip = tmp_path / "provider-output.zip"
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("nested/camera.mp4", b"video")
    observed: list[Path] = []

    def probe(path: Path) -> dict[str, object]:
        observed.append(path)
        return {"status": "completed", "frame_count": 4, "duration_seconds": 1.0}

    result = inspect_provider_runtime_output_zip(
        output_zip,
        video_extract_dir=tmp_path / "videos",
        expected_video_count=1,
        video_probe=probe,
    )

    assert result["video_smoke_proven"] is True
    assert len(observed) == 1
    assert observed[0].name == "000_camera.mp4"
