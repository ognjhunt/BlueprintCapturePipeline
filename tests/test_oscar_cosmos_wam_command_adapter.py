from __future__ import annotations

import json
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from blueprint_pipeline import oscar_cosmos_wam_command_adapter as adapter


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n\n", encoding="utf-8")


def _source_root(tmp_path: Path) -> Path:
    source_root = tmp_path / "cosmos-source"
    (source_root / "examples").mkdir(parents=True)
    (source_root / "examples" / "action_conditioned.py").write_text("# cosmos\n", encoding="utf-8")
    return source_root


def _rollout_manifest_path(tmp_path: Path) -> Path:
    review_video = tmp_path / "review" / "episode_0001__third_person.mp4"
    review_video.parent.mkdir(parents=True)
    review_video.write_bytes(b"fake mp4")
    trace_path = tmp_path / "normalized_policy_action_trace.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "normalized_action": {
                    "action_type": "waypoint",
                    "target_waypoint": [2.0, -2.0],
                }
            },
            {
                "normalized_action": {
                    "action_type": "manipulation_contact",
                    "vx_mps": 0.1,
                    "vy_mps": 0.2,
                    "yaw_rate_rad_s": 0.3,
                }
            },
        ],
    )
    manifest_path = tmp_path / "wam_rollout_input_manifest.json"
    _write_json(
        manifest_path,
        {
            "source_mujoco_endpoint_eval_job_dir": str(tmp_path / "mujoco-job"),
            "selected_review_videos": [{"path": str(review_video)}],
            "task_prompts": [{"task_prompt": "Predict the tote transfer."}],
            "inputs": {"normalized_policy_action_trace_jsonl": str(trace_path)},
        },
    )
    return manifest_path


def test_oscar_cosmos_helpers_materialize_action_conditioning_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert adapter._number(True, 4.5) == 4.5
    assert adapter._number("bad", 2.0) == 2.0
    array_json = tmp_path / "array.json"
    array_json.write_text("[]", encoding="utf-8")
    assert adapter._read_json(array_json) == {}
    assert adapter._read_jsonl(tmp_path / "missing.jsonl") == []
    rows_path = tmp_path / "rows.jsonl"
    _write_jsonl(rows_path, [{"ok": True}, ["ignored"]])
    assert adapter._read_jsonl(rows_path) == [{"ok": True}]

    existing = tmp_path / "exists"
    existing.mkdir()
    assert adapter._first_existing_path(["", str(existing)]) == existing.resolve()
    monkeypatch.setenv("BLUEPRINT_OSCAR_COSMOS_SOURCE_ROOT", str(existing))
    monkeypatch.setenv("BLUEPRINT_WAM_MODEL_CHECKPOINT", str(existing))
    assert adapter._source_root_from_env() == existing.resolve()
    assert adapter._checkpoint_from_env() == existing.resolve()

    rollout_manifest_path = _rollout_manifest_path(tmp_path)
    rollout_manifest = adapter._read_json(rollout_manifest_path)
    review_video = adapter._selected_video_path(rollout_manifest)
    assert review_video.name == "episode_0001__third_person.mp4"
    selection_manifest = tmp_path / "selection.json"
    _write_json(selection_manifest, {"selected_review_videos": [{"path": str(review_video)}]})
    assert (
        adapter._selected_video_path(
            {"inputs": {"review_video_selection_manifest": str(selection_manifest)}}
        )
        == review_video
    )
    with pytest.raises(FileNotFoundError, match="missing_selected_review_video"):
        adapter._selected_video_path({"selected_review_videos": [{"path": "missing.mp4"}]})

    assert adapter._task_prompt({}) == (
        "Predict the next robot-scene frames from Blueprint action conditioning."
    )
    assert adapter._task_prompt(rollout_manifest) == "Predict the tote transfer."
    assert len(adapter._action_trace_rows(rollout_manifest)) == 2
    assert adapter._action_vector({"normalized_action": {"action_type": "inspect_look", "vx_mps": 1}})[
        :2
    ] == [0.0, 0.0]
    assert adapter._action_vector(
        {"normalized_action": {"action_type": "stop", "vx_mps": 1, "yaw_rate_rad_s": 1}}
    ) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert adapter._action_vector(
        {"normalized_action": {"action_type": "manipulation_contact"}}
    )[2:] == [0.02, 0.0, 0.0, 0.0, 0.35]
    waypoint = adapter._action_vector(
        {"normalized_action": {"target_waypoint": [2.0, -2.0]}}
    )
    assert waypoint[:2] == [0.05, -0.05]
    assert adapter._action_sequence([], chunk_size=3) == [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]

    package = adapter._materialize_cosmos_input_package(
        rollout_manifest=rollout_manifest,
        work_dir=tmp_path / "work",
        chunk_size=4,
        resolution="256,320",
        guidance=3,
        num_steps=11,
    )
    annotation = json.loads(Path(package["annotation_path"]).read_text(encoding="utf-8"))
    inference = json.loads(Path(package["inference_params_path"]).read_text(encoding="utf-8"))
    assert annotation["texts"] == ["Predict the tote transfer."]
    assert len(annotation["action"]) == 4
    assert inference["guidance"] == 3
    assert inference["num_steps"] == 11
    assert Path(package["input_root"], "videos", "test", "blueprint_0", "rgb.mp4").is_file()


def test_oscar_cosmos_runtime_probe_subprocess_and_loader_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = _source_root(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("weights", encoding="utf-8")

    monkeypatch.setenv("PYTHONPATH", "existing-path")
    runtime_env = adapter._runtime_env(source_root)
    assert str(source_root) in runtime_env["PYTHONPATH"]
    assert "existing-path" in runtime_env["PYTHONPATH"]

    def probe_completed(*args: Any, **kwargs: Any) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"cosmos_oss": True, "cosmos_predict2": True, "tyro": True, "torch": True}
            ),
            stderr="",
        )

    monkeypatch.setattr(adapter.subprocess, "run", probe_completed)
    probe = adapter._run_import_probe(
        python=sys.executable,
        source_root=source_root,
        timeout_seconds=1,
    )
    assert probe["status"] == "completed"
    assert probe["blockers"] == []

    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="{", stderr=""),
    )
    invalid_json_probe = adapter._run_import_probe(
        python=sys.executable,
        source_root=source_root,
        timeout_seconds=1,
    )
    assert invalid_json_probe["status"] == "completed"

    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"cosmos_oss": False}),
            stderr="warning",
        ),
    )
    blocked_probe = adapter._run_import_probe(
        python=sys.executable,
        source_root=source_root,
        timeout_seconds=1,
    )
    assert blocked_probe["status"] == "blocked"
    assert blocked_probe["stderr_omitted_to_avoid_secret_leakage"] is True

    package_manifest = {
        "inference_params_path": str(tmp_path / "params.json"),
        "save_root": str(tmp_path / "generated"),
    }

    def raise_timeout(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise subprocess.TimeoutExpired("cosmos", 1, output="partial", stderr="slow")

    monkeypatch.setattr(adapter.subprocess, "run", raise_timeout)
    timeout = adapter._run_cosmos(
        python=sys.executable,
        source_root=source_root,
        checkpoint=checkpoint,
        package_manifest=package_manifest,
        output_dir=tmp_path / "out",
        model="model",
        experiment="experiment",
        context_parallel_size=1,
        timeout_seconds=1,
        extra_args=["--flag"],
    )
    assert timeout["status"] == "blocked"
    assert timeout["blockers"] == ["cosmos_action_conditioned_command_timeout"]
    assert str(checkpoint) not in timeout["argv_redacted"]

    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="CUDA extra not installed. No module named cosmos.",
            stderr="CUDA not available. MPS not supported. out of memory. checkpoint missing.",
        ),
    )
    failed = adapter._run_cosmos(
        python=sys.executable,
        source_root=source_root,
        checkpoint=checkpoint,
        package_manifest=package_manifest,
        output_dir=tmp_path / "out",
        model="model",
        experiment="experiment",
        context_parallel_size=1,
        timeout_seconds=1,
        extra_args=[],
    )
    assert {
        "cosmos_action_conditioned_command_nonzero",
        "blocked_cosmos_cuda_extra_not_installed",
        "blocked_cuda_not_available",
        "blocked_mps_not_available",
        "blocked_model_runtime_out_of_memory",
        "blocked_missing_python_module",
        "blocked_checkpoint_load_failed",
    }.issubset(set(failed["blockers"]))

    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    completed = adapter._run_cosmos(
        python=sys.executable,
        source_root=source_root,
        checkpoint=checkpoint,
        package_manifest=package_manifest,
        output_dir=tmp_path / "out",
        model="model",
        experiment="experiment",
        context_parallel_size=2,
        timeout_seconds=1,
        extra_args=[],
    )
    assert completed["status"] == "completed"

    fake_mediapy = SimpleNamespace(
        read_video=lambda path: np.zeros((2, 4, 4, 3), dtype=np.uint8),
        resize_image=lambda image, size: np.ones((size[0], size[1], 3), dtype=np.uint8),
    )
    monkeypatch.setitem(sys.modules, "mediapy", fake_mediapy)
    load_fn = adapter.load_blueprint_action_fn()
    with pytest.raises(ValueError, match=r"shaped \[N, 7\]"):
        load_fn({"action": [[0.0, 1.0]]}, "video.mp4", SimpleNamespace())
    loaded = load_fn(
        {"action": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]},
        "video.mp4",
        SimpleNamespace(start_frame_idx=99, resolution="2,3"),
    )
    assert loaded["initial_frame"].shape == (2, 3, 3)
    assert loaded["actions"].shape == (1, 7)


def test_oscar_cosmos_run_blocks_probes_and_materializes_rollouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "blocked_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(output_path))
    blocked = adapter.run(
        [
            "--source-root",
            str(tmp_path / "missing-source"),
            "--checkpoint",
            str(tmp_path / "missing-checkpoint"),
            "--python",
            str(tmp_path / "missing-python"),
            "--work-dir",
            str(tmp_path / "blocked-work"),
        ]
    )
    assert blocked["status"] == "blocked"
    assert {
        "blocked_missing_cosmos_action_conditioned_entrypoint",
        "blocked_configured_oscar_cosmos_checkpoint_path_missing",
        "blocked_configured_python_missing",
    }.issubset(set(blocked["blockers"]))
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "blocked"

    source_root = _source_root(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("weights", encoding="utf-8")
    rollout_manifest_path = _rollout_manifest_path(tmp_path)
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_INPUT", str(rollout_manifest_path))

    monkeypatch.setattr(
        adapter,
        "_run_import_probe",
        lambda **kwargs: {
            "schema_version": "probe",
            "status": "blocked",
            "blockers": ["blocked_missing_cosmos_runtime_import"],
        },
    )
    probe_only_output = tmp_path / "probe_only.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(probe_only_output))
    probe_only = adapter.run(
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
    assert probe_only["probe_only"] is True
    assert probe_only["status"] == "blocked"

    blocked_probe_output = tmp_path / "blocked_probe.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(blocked_probe_output))
    blocked_probe = adapter.run(
        [
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--python",
            sys.executable,
            "--work-dir",
            str(tmp_path / "blocked-probe-work"),
        ]
    )
    assert blocked_probe["status"] == "blocked"
    assert blocked_probe["import_probe"]["status"] == "blocked"

    monkeypatch.setattr(
        adapter,
        "_run_import_probe",
        lambda **kwargs: {"schema_version": "probe", "status": "completed", "blockers": []},
    )

    def fake_run_cosmos(**kwargs: Any) -> dict[str, Any]:
        save_root = Path(kwargs["package_manifest"]["save_root"])
        (save_root / "sample").mkdir(parents=True)
        (save_root / "sample" / "rollout.mp4").write_bytes(b"generated")
        return {"schema_version": "cosmos", "status": "completed", "blockers": []}

    monkeypatch.setattr(adapter, "_run_cosmos", fake_run_cosmos)
    completed_output = tmp_path / "completed.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(completed_output))
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
            "--model",
            "cosmos-model",
            "--experiment",
            "cosmos-experiment",
            "--context-parallel-size",
            "2",
            "--chunk-size",
            "2",
            "--resolution",
            "128,160",
            "--guidance",
            "4",
            "--num-steps",
            "8",
            "--timeout-seconds",
            "12",
            "--extra-arg",
            "--dry-run true",
        ]
    )
    assert completed["status"] == "completed"
    assert completed["generated_video_count"] == 1
    assert completed["rollouts"][0]["model"] == "cosmos-model"

    monkeypatch.setattr(
        adapter,
        "_run_cosmos",
        lambda **kwargs: {
            "schema_version": "cosmos",
            "status": "blocked",
            "blockers": ["cosmos_action_conditioned_command_nonzero"],
        },
    )
    blocked_full_output = tmp_path / "blocked_full.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(blocked_full_output))
    blocked_full = adapter.run(
        [
            "--source-root",
            str(source_root),
            "--checkpoint",
            str(checkpoint),
            "--python",
            sys.executable,
            "--work-dir",
            str(tmp_path / "blocked-full-work"),
        ]
    )
    assert blocked_full["status"] == "blocked"
    assert blocked_full["blockers"] == ["cosmos_action_conditioned_command_nonzero"]


def test_oscar_cosmos_rollout_payload_main_and_module_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_root = _source_root(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.write_text("weights", encoding="utf-8")
    save_root = tmp_path / "save-root"
    (save_root / "nested").mkdir(parents=True)
    (save_root / "nested" / "rollout.mp4").write_bytes(b"mp4")

    payload = adapter._rollout_payload(
        package_manifest={
            "save_root": str(save_root),
            "source_review_video_path": str(tmp_path / "review.mp4"),
        },
        checkpoint=checkpoint,
        source_root=source_root,
        subprocess_detail={"status": "completed"},
        model="model",
        experiment="experiment",
    )
    assert payload["status"] == "completed"
    assert payload["rollouts"][0]["generated_rollout_termination_reason"] == (
        "cosmos_command_completed"
    )

    blocked = adapter._rollout_payload(
        package_manifest={"save_root": str(tmp_path / "empty-save-root")},
        checkpoint=checkpoint,
        source_root=source_root,
        subprocess_detail={"status": "completed"},
        model="model",
        experiment="experiment",
    )
    assert blocked["status"] == "blocked"
    assert blocked["blockers"] == ["blocked_no_generated_cosmos_mp4"]

    monkeypatch.setattr(adapter, "run", lambda argv: {"status": "completed"})
    assert adapter.main(["--ignored"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"

    def raise_adapter(argv: Any) -> dict[str, Any]:
        del argv
        raise RuntimeError("boom")

    exception_output = tmp_path / "exception_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(exception_output))
    monkeypatch.setattr(adapter, "run", raise_adapter)
    assert adapter.main([]) == 2
    assert "oscar_cosmos_adapter_exception:RuntimeError" in json.loads(
        exception_output.read_text(encoding="utf-8")
    )["blockers"]

    module_output = tmp_path / "module_output.json"
    monkeypatch.setenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", str(module_output))
    monkeypatch.setattr(sys, "argv", ["oscar_cosmos_wam_command_adapter.py"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(
            "blueprint_pipeline.oscar_cosmos_wam_command_adapter",
            run_name="__main__",
        )
    assert exc.value.code == 2
    assert json.loads(module_output.read_text(encoding="utf-8"))["status"] == "blocked"
