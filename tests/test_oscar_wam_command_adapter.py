from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from blueprint_pipeline import oscar_wam_command_adapter as adapter


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_review_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64))
    assert writer.isOpened()
    for index in range(4):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = 40 + index * 20
        frame[:, :, 1] = 90
        frame[:, :, 2] = 140
        writer.write(frame)
    writer.release()


def test_oscar_wam_command_adapter_materializes_inputs_and_blocks_without_cuda(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "oscar-source"
    (source_root / "inference").mkdir(parents=True)
    (source_root / "inference" / "inference_oscar.py").write_text("# entrypoint\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    review_video = tmp_path / "review" / "episode_0001__third_person.mp4"
    _write_review_video(review_video)
    trace_path = tmp_path / "g1_mujoco_locomotion_trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "episode_id": "episode_0001",
                "root_position": [0.0 + step * 0.05, 0.0, 0.79],
                "root_yaw_rad": 0.1 * step,
                "active_action": {"action_type": "base_velocity", "vx_mps": 0.2, "vy_mps": 0.0},
                "fall_detected": False,
            }
            for step in range(8)
        ],
    )
    rollout_input = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(
        rollout_input,
        {
            "source_mujoco_endpoint_eval_job_dir": str(tmp_path),
            "selected_review_videos": [{"path": str(review_video)}],
            "task_prompts": [{"task_prompt": "Move toward the target."}],
            "inputs": {"g1_mujoco_locomotion_trace_jsonl": str(trace_path)},
        },
    )
    output_path = tmp_path / "wam_provider_output.json"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' '{\"module_available\":{\"torch\":true,\"torchvision\":true,\"cv2\":true,\"decord\":true,\"einops\":true,\"diffusers\":true,\"transformers\":true,\"worldsim\":true},\"torch_cuda_available\":false,\"platform_system\":\"Linux\"}'\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output_path))

    payload = adapter.run(
        [
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--python",
            str(fake_python),
            "--work-dir",
            str(tmp_path / "work"),
            "--num-frames",
            "4",
            "--height",
            "64",
            "--width",
            "64",
            "--fps",
            "5",
            "--probe-only",
        ]
    )

    assert payload["status"] == "blocked"
    assert "blocked_oscar_requires_cuda_gpu_runtime" in payload["blockers"]
    written = json.loads(output_path.read_text(encoding="utf-8"))
    package = written["input_package"]
    assert Path(package["first_frame"]["path"]).is_file()
    assert Path(package["skeleton_video"]["path"]).is_file()
    assert package["claim_boundary"]["skeleton_conditioning_is_proxy_from_mujoco_trace"] is True
    assert written["raw_credentials_written_to_artifacts"] is False
    assert written["secret_hashes_written_to_artifacts"] is False


def test_oscar_wam_command_adapter_private_helper_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert adapter._number(True, 1.5) == 1.5
    assert adapter._number("bad", 2.5) == 2.5
    assert adapter._read_jsonl(tmp_path / "missing.jsonl") == []
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text('\n{"row": 1}\n["ignored"]\n', encoding="utf-8")
    assert adapter._read_jsonl(rows_path) == [{"row": 1}]
    assert adapter._repo_src_root().name == "src"

    existing = tmp_path / "existing"
    existing.mkdir()
    assert adapter._first_existing_path(["", str(existing)]) == existing.resolve()
    monkeypatch.setenv("BLUEPRINT_OSCAR_WAM_SOURCE_ROOT", str(existing))
    monkeypatch.setenv("BLUEPRINT_WAM_MODEL_CHECKPOINT", str(existing))
    assert adapter._source_root_from_env() == existing.resolve()
    assert adapter._checkpoint_from_env() == existing.resolve()

    review_video = tmp_path / "episode_0001__third_person.mp4"
    _write_review_video(review_video)
    selection_manifest = tmp_path / "selection.json"
    _write_json(selection_manifest, {"selected_review_videos": [{"path": str(review_video)}]})
    assert (
        adapter._selected_video_path(
            {"inputs": {"review_video_selection_manifest": str(selection_manifest)}}
        )
        == review_video.resolve()
    )
    with pytest.raises(FileNotFoundError, match="missing_selected_review_video"):
        adapter._selected_video_path({"selected_review_videos": [{"path": "missing.mp4"}]})
    assert adapter._task_prompt({}) == (
        "Predict the next robot-scene frames from Blueprint action conditioning."
    )

    trace_path = tmp_path / "trace.jsonl"
    _write_jsonl(trace_path, [{"episode_id": "other", "root_position": [1, 2, 3]}])
    assert adapter._trace_rows({"inputs": {"g1_mujoco_locomotion_trace_jsonl": str(trace_path)}}) == [
        {"episode_id": "other", "root_position": [1, 2, 3]}
    ]
    assert adapter._sample_rows([], 3) == []
    assert adapter._sample_rows([{"row": 1}], 3) == [{"row": 1}, {"row": 1}, {"row": 1}]
    assert adapter._point_from_root({"root_position": "bad"}) == (0.0, 0.0, 0.8)

    with pytest.raises(ValueError, match="missing_locomotion_trace"):
        adapter._render_proxy_skeleton_video(
            trace_rows=[],
            output_path=tmp_path / "empty.mp4",
            width=64,
            height=64,
            fps=5.0,
            num_frames=1,
        )

    class ClosedWriter:
        def isOpened(self) -> bool:
            return False

    monkeypatch.setattr(cv2, "VideoWriter", lambda *args, **kwargs: ClosedWriter())
    with pytest.raises(RuntimeError, match="cv2_video_writer_failed"):
        adapter._render_proxy_skeleton_video(
            trace_rows=[{"root_position": [0, 0, 0.8]}],
            output_path=tmp_path / "closed.mp4",
            width=64,
            height=64,
            fps=5.0,
            num_frames=1,
        )

    monkeypatch.undo()
    skeleton = adapter._render_proxy_skeleton_video(
        trace_rows=[
            {
                "root_position": [0, 0, 0.8],
                "root_yaw_rad": 0.1,
                "active_action": {"action_type": "inspect_look"},
                "fall_detected": True,
            },
            {
                "root_position": [0.1, 0.1, 0.82],
                "active_action": {"action_type": "unknown"},
            },
        ],
        output_path=tmp_path / "inspect.mp4",
        width=64,
        height=64,
        fps=5.0,
        num_frames=2,
    )
    assert skeleton["fall_frame_count"] == 1
    assert {"action_type": "inspect_look", "count": 1} in skeleton["action_type_counts"]

    bad_video = tmp_path / "bad.mp4"
    bad_video.write_bytes(b"not a video")
    with pytest.raises(ValueError, match="could_not_decode_selected_review_video_first_frame"):
        adapter._extract_first_frame(
            review_video=bad_video,
            output_path=tmp_path / "first.png",
            width=64,
            height=64,
        )


def test_oscar_wam_runtime_probe_subprocess_and_rollout_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "oscar-source"
    (source_root / "inference").mkdir(parents=True)
    (source_root / "inference" / "inference_oscar.py").write_text("# oscar\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("weights", encoding="utf-8")

    monkeypatch.setenv("PYTHONPATH", "existing")
    env = adapter._runtime_env(source_root)
    assert str(source_root) in env["PYTHONPATH"]
    assert "existing" in env["PYTHONPATH"]

    monkeypatch.setattr(adapter.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="{", stderr=""),
    )
    invalid_probe = adapter._run_import_probe(
        python=sys.executable,
        source_root=source_root,
        timeout_seconds=1,
    )
    assert invalid_probe["status"] == "blocked"
    assert "blocked_oscar_runtime_import_probe_failed" in invalid_probe["blockers"]

    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "module_available": {"torch": True, "worldsim": False},
                    "torch_cuda_available": False,
                    "platform_system": "Linux",
                }
            ),
            stderr="warn",
        ),
    )
    missing_probe = adapter._run_import_probe(
        python=sys.executable,
        source_root=source_root,
        timeout_seconds=1,
    )
    assert missing_probe["status"] == "blocked"
    assert missing_probe["missing_modules"] == ["worldsim"]
    assert missing_probe["stderr_omitted_to_avoid_secret_leakage"] is True

    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "module_available": {"torch": True},
                    "torch_cuda_available": True,
                    "platform_system": "Linux",
                }
            ),
            stderr="",
        ),
    )
    completed_probe = adapter._run_import_probe(
        python=sys.executable,
        source_root=source_root,
        timeout_seconds=1,
    )
    assert completed_probe["status"] == "completed"

    assert str(checkpoint) not in adapter._redacted_argv(["python", str(checkpoint)], checkpoint)

    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="out", stderr="err"),
    )
    failed = adapter._run_oscar(
        python=sys.executable,
        source_root=source_root,
        checkpoint=checkpoint,
        package_manifest={
            "first_frame": {"path": str(tmp_path / "first.png")},
            "skeleton_video": {"path": str(tmp_path / "skeleton.mp4")},
            "prompt": "Move.",
            "num_frames": 2,
            "height": 64,
            "width": 64,
            "fps": 5,
        },
        output_video=tmp_path / "out.mp4",
        timeout_seconds=1,
        num_steps=2,
        guidance=1.5,
        seed=3,
    )
    assert failed["status"] == "blocked"
    assert failed["blockers"] == ["oscar_inference_command_nonzero"]

    output_video = tmp_path / "rollout.mp4"
    completed_payload = adapter._rollout_payload(
        package_manifest={"source_review_video_path": "review.mp4"},
        checkpoint=checkpoint,
        source_root=source_root,
        subprocess_detail={"status": "completed"},
        output_video=output_video,
    )
    assert completed_payload["status"] == "blocked"
    assert completed_payload["blockers"] == ["blocked_no_generated_oscar_mp4"]
    output_video.write_bytes(b"mp4")
    completed_payload = adapter._rollout_payload(
        package_manifest={"source_review_video_path": "review.mp4"},
        checkpoint=checkpoint,
        source_root=source_root,
        subprocess_detail={"status": "completed"},
        output_video=output_video,
    )
    assert completed_payload["status"] == "completed"
    assert completed_payload["rollouts"][0]["generated_video_path"] == str(output_video)


def test_oscar_wam_run_main_and_module_guard_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "missing_runtime_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output_path))
    missing = adapter.run(
        [
            "--source-root",
            str(tmp_path / "missing-source"),
            "--checkpoint",
            str(tmp_path / "missing-checkpoint"),
            "--python",
            str(tmp_path / "missing-python"),
        ]
    )
    assert missing["status"] == "blocked"
    assert {
        "blocked_missing_oscar_inference_entrypoint",
        "blocked_configured_oscar_checkpoint_path_missing",
        "blocked_configured_python_missing",
    }.issubset(set(missing["blockers"]))

    source_root = tmp_path / "oscar-source"
    (source_root / "inference").mkdir(parents=True)
    (source_root / "inference" / "inference_oscar.py").write_text("# oscar\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("weights", encoding="utf-8")

    def raise_materialize(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise RuntimeError("bad inputs")

    monkeypatch.setattr(adapter, "_materialize_oscar_input_package", raise_materialize)
    materialize_blocked = adapter.run(
        [
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--python",
            sys.executable,
            "--work-dir",
            str(tmp_path / "materialize-work"),
        ]
    )
    assert materialize_blocked["blockers"] == [
        "blocked_oscar_input_package_materialization_failed:KeyError"
    ]

    def fake_materialize(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "schema_version": "blueprint_oscar_wam_input_package.v1",
            "first_frame": {"path": str(tmp_path / "first.png")},
            "skeleton_video": {"path": str(tmp_path / "skeleton.mp4")},
            "prompt": "Move.",
        }

    monkeypatch.setattr(adapter, "_materialize_oscar_input_package", fake_materialize)
    rollout_input = tmp_path / "rollout_input.json"
    _write_json(rollout_input, {})
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_input))
    monkeypatch.setattr(
        adapter,
        "_run_import_probe",
        lambda **kwargs: {"status": "blocked", "blockers": ["probe_blocked"]},
    )
    probe_blocked = adapter.run(
        [
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--python",
            sys.executable,
            "--work-dir",
            str(tmp_path / "probe-work"),
            "--probe-only",
        ]
    )
    assert probe_blocked["probe_only"] is True
    assert probe_blocked["blockers"] == ["probe_blocked"]

    monkeypatch.setattr(
        adapter,
        "_run_import_probe",
        lambda **kwargs: {"status": "completed", "blockers": []},
    )

    def fake_run_oscar(**kwargs: Any) -> dict[str, Any]:
        Path(kwargs["output_video"]).write_bytes(b"mp4")
        return {"status": "completed", "blockers": []}

    monkeypatch.setattr(adapter, "_run_oscar", fake_run_oscar)
    completed = adapter.run(
        [
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--python",
            sys.executable,
            "--work-dir",
            str(tmp_path / "completed-work"),
            "--num-frames",
            "2",
            "--height",
            "64",
            "--width",
            "64",
            "--fps",
            "5",
            "--num-steps",
            "3",
            "--guidance",
            "1.5",
            "--seed",
            "9",
            "--timeout-seconds",
            "10",
        ]
    )
    assert completed["status"] == "completed"
    assert completed["generated_video_count"] == 1

    monkeypatch.setattr(
        adapter,
        "_run_oscar",
        lambda **kwargs: {"status": "blocked", "blockers": ["oscar_failed"]},
    )
    blocked = adapter.run(
        [
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--python",
            sys.executable,
            "--work-dir",
            str(tmp_path / "blocked-work"),
        ]
    )
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == ["oscar_failed"]

    monkeypatch.setattr(adapter, "run", lambda argv: {"status": "completed"})
    assert adapter.main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"

    def raise_run(argv: Any) -> dict[str, Any]:
        del argv
        raise RuntimeError("boom")

    exception_output = tmp_path / "exception.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(exception_output))
    monkeypatch.setattr(adapter, "run", raise_run)
    assert adapter.main([]) == 2
    assert "oscar_wam_adapter_exception:RuntimeError" in json.loads(
        exception_output.read_text(encoding="utf-8")
    )["blockers"]

    module_output = tmp_path / "module.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(module_output))
    monkeypatch.setattr(sys, "argv", ["oscar_wam_command_adapter.py"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("blueprint_pipeline.oscar_wam_command_adapter", run_name="__main__")
    assert exc.value.code == 2
    assert json.loads(module_output.read_text(encoding="utf-8"))["status"] == "blocked"
