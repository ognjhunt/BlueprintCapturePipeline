from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from tests.test_public_scene_artifixer3d_dual_target_inputs import _dual_candidate


def _runner_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/public_scene_artifixer3d_runner.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_public_scene_artifixer3d_dual_target_runner", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(runner) -> dict[str, object]:
    return {
        "pipeline_mode": runner.DUAL_TARGET_PIPELINE_MODE,
        "direct_editor_backend": "none",
        "semantic_editor": None,
        "semantic_editor_only": False,
        "model": None,
        "wan_base": None,
        "direct_inference": None,
        "direct_model_weights_required": False,
        "phases": runner.DUAL_TARGET_PHASES,
        "outside_exact_support_changed_pixels_permitted": (
            "unconstrained_for_raw_representation_review"
        ),
        "outside_support_invariance_gate": (
            "deferred_until_final_soft_composite"
        ),
        "artifixer3d": {
            "steps": 10,
            "config_name": "apps/colmap_3dgut_sparse_mcmc_lpips",
            "loss_overrides": runner.DUAL_TARGET_LOSS_OVERRIDES,
            "anchor_mask_reduction": "full_frame_mean",
        },
    }


def test_actual_materialized_dual_target_contract_and_masks_are_accepted(
    tmp_path: Path,
) -> None:
    runner = _runner_module()
    _dual_root, _source, _semantic, dual = _dual_candidate(tmp_path)
    assert runner._dual_target_request_is_bound(_request(runner))
    assert runner._dual_target_candidate_is_bound(dual)

    task = dual["tasks"][0]
    staged_task = Path(task["scene_directory"])
    task_output = tmp_path / "task_output"
    teacher_root, teacher_rows = runner._prepare_dual_target_teacher_frames(
        task=task,
        staged_task=staged_task,
        task_output=task_output,
    )
    assert sorted(path.name for path in teacher_root.iterdir()) == [
        "00001.png",
        "00003.png",
    ]
    assert [row["semantic_teacher_training_index"] for row in teacher_rows] == [
        1,
        3,
    ]

    distillation = tmp_path / "distillation"
    images = distillation / "images"
    images.mkdir(parents=True)
    for frame in task["frames"]:
        index = frame["anchor_training_index"]
        source = staged_task / frame["anchor_rgb"]["relative_path"]
        (images / f"frame_{index:05d}.png").symlink_to(source.resolve())
    mask_rows = runner._stage_dual_target_anchor_masks(
        task=task,
        staged_task=staged_task,
        distillation_input_dir=distillation,
    )
    assert [row["anchor_training_index"] for row in mask_rows] == [0, 2]
    assert sorted(path.name for path in images.glob("*_mask.png")) == [
        "frame_00000_mask.png",
        "frame_00002_mask.png",
    ]
    assert not (images / "frame_00001_mask.png").exists()
    assert not (images / "frame_00003_mask.png").exists()


def test_native_review_layout_is_byte_copied_to_provider_retained_directory(
    tmp_path: Path,
) -> None:
    runner = _runner_module()
    _dual_root, _source, _semantic, dual = _dual_candidate(
        tmp_path / "candidate", cameras_per_task=2
    )
    task = dual["tasks"][0]
    staged_task = Path(task["scene_directory"])
    task_output = tmp_path / "runtime_output" / "tasks" / task["task_id"]
    native_review = (
        task_output
        / "artifixer3d"
        / "recon_results"
        / task["task_id"]
        / "artifixer3d"
        / task["task_id"]
        / "ours_10"
        / "review_transforms"
    )
    renders = native_review / "renders"
    renders.mkdir(parents=True)
    source_bytes: list[bytes] = []
    for frame in task["frames"]:
        source = staged_task / frame["anchor_rgb"]["relative_path"]
        destination = renders / f"{frame['physical_camera_index']:05d}.png"
        shutil.copyfile(source, destination)
        source_bytes.append(destination.read_bytes())

    rows = runner._normalize_dual_target_review_frames(
        task=task,
        staged_task=staged_task,
        review_dir=native_review,
        task_output=task_output,
    )

    normalized = task_output / "artifixer3d_review_frames"
    assert [Path(row["path"]) for row in rows] == [
        normalized / "00000.png",
        normalized / "00001.png",
    ]
    assert [path.read_bytes() for path in sorted(normalized.iterdir())] == source_bytes
    assert [row["frame_index"] for row in rows] == [0, 1]
    assert [row["camera_id"] for row in rows] == [
        frame["camera_id"] for frame in task["frames"]
    ]


@pytest.mark.parametrize("failure", ["missing", "extra", "order", "size"])
def test_review_normalization_fails_closed_on_invalid_native_output(
    tmp_path: Path, failure: str
) -> None:
    runner = _runner_module()
    _dual_root, _source, _semantic, dual = _dual_candidate(
        tmp_path / "candidate", cameras_per_task=2
    )
    task = dual["tasks"][0]
    staged_task = Path(task["scene_directory"])
    task_output = tmp_path / "runtime_output" / "tasks" / task["task_id"]
    native_review = task_output / "native" / "review_transforms"
    renders = native_review / "renders"
    renders.mkdir(parents=True)
    for frame in task["frames"]:
        source = staged_task / frame["anchor_rgb"]["relative_path"]
        shutil.copyfile(
            source, renders / f"{frame['physical_camera_index']:05d}.png"
        )
    if failure == "missing":
        (renders / "00001.png").unlink()
    elif failure == "extra":
        shutil.copyfile(renders / "00000.png", renders / "00002.png")
    elif failure == "order":
        task["frames"] = list(reversed(task["frames"]))
    else:
        Image.new("RGB", (1, 1), "black").save(renders / "00001.png")

    with pytest.raises(ValueError, match="artifixer3d_dual_target_review_"):
        runner._normalize_dual_target_review_frames(
            task=task,
            staged_task=staged_task,
            review_dir=native_review,
            task_output=task_output,
        )


def test_dual_target_execute_skips_all_direct_model_downloads_and_3d_plus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_module()
    request = {
        **_request(runner),
        "runtime_request_digest": "sha256:" + "1" * 64,
        "task_ids": ["task_a"],
    }
    candidate = {
        "receipt_digest": "sha256:" + "2" * 64,
        "replacement_object_count": 1,
        "tasks": [{"task_id": "task_a"}],
    }
    manifest = {"manifest_digest": "sha256:" + "3" * 64}
    completed = {
        "task_id": "task_a",
        "pipeline_mode": runner.DUAL_TARGET_PIPELINE_MODE,
        "artifixer3d_review_frames": [{"frame_index": 0}],
        "final_candidate_frames": [{"frame_index": 0}],
        "outside_support_invariance_status": (
            "deferred_until_final_soft_composite"
        ),
        "outside_exact_support_invariance_proven": False,
        "outside_support_changed_pixels_total": None,
    }
    monkeypatch.setattr(
        runner,
        "_validate_bundle",
        lambda _root: (manifest, request, candidate),
    )
    monkeypatch.setattr(
        runner,
        "_download_models",
        lambda *_args, **_kwargs: pytest.fail("direct weights must not download"),
    )
    monkeypatch.setattr(
        runner,
        "_download_semantic_editor",
        lambda *_args, **_kwargs: pytest.fail("editor weights must not download"),
    )
    monkeypatch.setattr(runner, "_task_runtime", lambda **_kwargs: completed)

    result = runner.execute(
        bundle_root=tmp_path / "bundle",
        output_root=tmp_path / "output",
        rehearsal=False,
    )
    assert result["pipeline_mode"] == runner.DUAL_TARGET_PIPELINE_MODE
    assert result["artifixer_direct_inference_executed"] is False
    assert result["semantic_editor_inference_executed"] is False
    assert result["artifixer3d_distillation_executed"] is True
    assert result["artifixer3d_plus_inference_executed"] is False
    assert result["outside_exact_support_invariance_proven"] is False
    assert result["tasks"] == [completed]


def test_render_only_task_reuses_exact_checkpoint_and_normalizes_eight_cameras(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_module()
    task_id = "task_a"
    staged_task = tmp_path / "input" / task_id
    staged_task.mkdir(parents=True)
    frames: list[dict[str, object]] = []
    for index in range(8):
        anchor = staged_task / f"anchor_{index:05d}.png"
        Image.new("RGB", (12, 10), (index, index, index)).save(anchor)
        frames.append(
            {
                "physical_camera_index": index,
                "camera_id": f"task_a_camera_{index:05d}",
                "anchor_rgb": {
                    "relative_path": anchor.relative_to(staged_task).as_posix(),
                    "size_bytes": anchor.stat().st_size,
                    "sha256": runner._sha256(anchor),
                },
            }
        )
    checkpoint = tmp_path / "input/checkpoint_reuse/checkpoint_00000.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"sealed-checkpoint-bytes")
    checkpoint_record = {
        "relative_path": checkpoint.relative_to(tmp_path / "input").as_posix(),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": runner._sha256(checkpoint),
    }
    review_trajectory = staged_task / "review_transforms.json"
    review_trajectory.write_text("{}\n", encoding="utf-8")
    native_review = tmp_path / "native_review"
    renders = native_review / "renders"

    def render_artifixer3d(
        _scene,
        _paths,
        *,
        checkpoint: Path,
        checkpoint_reused: bool,
        replace: bool,
        render_trajectory_path: Path,
    ) -> Path:
        assert checkpoint.read_bytes() == b"sealed-checkpoint-bytes"
        assert runner._sha256(checkpoint) == checkpoint_record["sha256"]
        assert checkpoint_reused is True
        assert replace is False
        assert render_trajectory_path == review_trajectory
        renders.mkdir(parents=True)
        for index in range(8):
            Image.new("RGB", (12, 10), (index, index, index)).save(
                renders / f"{index:05d}.png"
            )
        return native_review

    data_processing = ModuleType("data_processing")
    artifixer3d = ModuleType("data_processing.artifixer3d")
    artifixer3d.render_artifixer3d = render_artifixer3d
    data_processing.artifixer3d = artifixer3d
    monkeypatch.setitem(sys.modules, "data_processing", data_processing)
    monkeypatch.setitem(sys.modules, "data_processing.artifixer3d", artifixer3d)

    def prepared_replay(**_kwargs):
        (tmp_path / "runtime_output/tasks/task_a/logs").mkdir(parents=True)
        return {
            "staged_task": staged_task,
            "teacher_rows": [],
            "review_trajectory": review_trajectory,
            "scene": object(),
            "paths": SimpleNamespace(),
            "anchor_mask_rows": [],
        }

    monkeypatch.setattr(
        runner,
        "_prepare_dual_target_distillation_replay",
        prepared_replay,
    )
    request = {
        "artifixer3d": {
            "checkpoint_reuse": {
                "reuse_digest": "sha256:" + "a" * 64,
                "checkpoints": [
                    {"task_id": task_id, "checkpoint": checkpoint_record}
                ],
            },
            "anchor_mask_reduction": "full_frame_mean",
            "loss_overrides": runner.DUAL_TARGET_LOSS_OVERRIDES,
        }
    }
    result = runner._dual_target_render_only_task_runtime(
        task={
            "task_id": task_id,
            "physical_camera_count": 8,
            "training_record_count": 16,
            "selected_anchor_indices": list(range(0, 16, 2)),
            "semantic_teacher_indices": list(range(1, 16, 2)),
            "frames": frames,
        },
        input_root=tmp_path / "input",
        output_root=tmp_path / "runtime_output",
        request=request,
    )

    assert result["training_executed"] is False
    assert result["direct_artifixer_executed"] is False
    assert result["artifixer3d_plus_executed"] is False
    assert result["checkpoint_reused"] is True
    assert result["artifixer3d_checkpoint"]["sha256"] == checkpoint_record["sha256"]
    assert [row["camera_id"] for row in result["artifixer3d_review_frames"]] == [
        frame["camera_id"] for frame in frames
    ]
    normalized = (
        tmp_path / "runtime_output/tasks/task_a/artifixer3d_review_frames"
    )
    assert [path.name for path in sorted(normalized.iterdir())] == [
        f"{index:05d}.png" for index in range(8)
    ]


def test_render_only_execute_skips_training_direct_and_3d_plus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner_module()
    reuse_digest = "sha256:" + "a" * 64
    request = {
        **_request(runner),
        "pipeline_mode": runner.DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
        "phases": runner.DUAL_TARGET_RENDER_ONLY_PHASES,
        "runtime_request_digest": "sha256:" + "1" * 64,
        "task_ids": ["task_a"],
    }
    request["artifixer3d"] = {
        **request["artifixer3d"],
        "training_permitted": False,
        "distillation_input_replay_only": True,
        "checkpoint_reuse": {"reuse_digest": reuse_digest},
    }
    candidate = {
        "receipt_digest": "sha256:" + "2" * 64,
        "replacement_object_count": 1,
        "tasks": [{"task_id": "task_a"}],
    }
    manifest = {"manifest_digest": "sha256:" + "3" * 64}
    review_rows = [
        {"frame_index": index, "camera_id": f"task_a_camera_{index:05d}"}
        for index in range(8)
    ]
    completed = {
        "task_id": "task_a",
        "pipeline_mode": runner.DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE,
        "checkpoint_reused": True,
        "checkpoint_reuse_digest": reuse_digest,
        "training_executed": False,
        "direct_artifixer_executed": False,
        "artifixer3d_plus_executed": False,
        "artifixer3d_review_frames": review_rows,
        "final_candidate_frames": review_rows,
        "outside_support_invariance_status": (
            "deferred_until_final_soft_composite"
        ),
        "outside_exact_support_invariance_proven": False,
        "outside_support_changed_pixels_total": None,
    }
    monkeypatch.setattr(
        runner,
        "_validate_bundle",
        lambda _root: (manifest, request, candidate),
    )
    monkeypatch.setattr(
        runner,
        "_download_models",
        lambda *_args, **_kwargs: pytest.fail("direct weights must not download"),
    )
    monkeypatch.setattr(
        runner,
        "_download_semantic_editor",
        lambda *_args, **_kwargs: pytest.fail("editor weights must not download"),
    )
    monkeypatch.setattr(runner, "_task_runtime", lambda **_kwargs: completed)

    result = runner.execute(
        bundle_root=tmp_path / "bundle",
        output_root=tmp_path / "output",
        rehearsal=False,
    )
    assert result["pipeline_mode"] == runner.DUAL_TARGET_RENDER_ONLY_PIPELINE_MODE
    assert result["checkpoint_reuse_digest"] == reuse_digest
    assert result["artifixer3d_checkpoint_reused"] is True
    assert result["artifixer3d_distillation_executed"] is False
    assert result["artifixer_direct_inference_executed"] is False
    assert result["semantic_editor_inference_executed"] is False
    assert result["artifixer3d_plus_inference_executed"] is False
    assert result["tasks"] == [completed]
