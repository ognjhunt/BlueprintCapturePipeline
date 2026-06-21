from __future__ import annotations

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline.synthesis import cosmos_vast_training_wrapper as wrapper


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    payload = {
        "trainer_config": str(tmp_path / "trainer.json"),
        "output_dir": str(tmp_path / "out"),
        "export_manifest": str(tmp_path / "export.json"),
        "capture_root": str(tmp_path / "capture"),
        "paired_reference_target": "paired.json",
        "k_reference_conditioning": "krefs.json",
        "train_val_split": "split.json",
    }
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_cosmos_vast_standard_env_mock_checkpoint_and_prefix(monkeypatch, tmp_path: Path) -> None:
    args = _args(tmp_path)
    monkeypatch.setenv("COSMOS_MODEL_ID", "cosmos/model")
    env = wrapper._standard_env(args)
    assert env["COSMOS_TRAINER_CONFIG_PATH"] == str((tmp_path / "trainer.json").resolve())
    assert env["COSMOS_TRAINER_OUTPUT_DIR"] == str((tmp_path / "out").resolve())
    assert env["COSMOS_EXPORT_MANIFEST_PATH"] == str((tmp_path / "export.json").resolve())
    assert env["COSMOS_CAPTURE_ROOT"] == str((tmp_path / "capture").resolve())
    assert env["COSMOS_PAIRED_REFERENCE_TARGET_PATH"] == "paired.json"
    assert env["COSMOS_K_REFERENCE_CONDITIONING_PATH"] == "krefs.json"
    assert env["COSMOS_TRAIN_VAL_SPLIT_PATH"] == "split.json"
    assert env["COSMOS_MODEL_ID"] == "cosmos/model"

    minimal_env = wrapper._standard_env(_args(tmp_path, export_manifest=None, capture_root=None, paired_reference_target=None, k_reference_conditioning=None, train_val_split=None))
    assert minimal_env["COSMOS_EXPORT_MANIFEST_PATH"] == ""
    assert minimal_env["COSMOS_CAPTURE_ROOT"] == ""

    assert wrapper._write_mock_checkpoint(tmp_path / "mock") == 0
    assert (tmp_path / "mock" / "adapter_model.safetensors").read_bytes() == b"mock-cosmos-adapter"
    assert json.loads((tmp_path / "mock" / "trainer_state.json").read_text(encoding="utf-8")) == {"mode": "mock_success"}

    monkeypatch.setenv("COSMOS_TRAINER_NUM_PROCESSES", "2")
    monkeypatch.setenv("COSMOS_TRAINER_MIXED_PRECISION", "fp16")
    assert wrapper._accelerate_prefix() == ["accelerate", "launch", "--num_processes", "2", "--mixed_precision", "fp16"]
    monkeypatch.setenv("COSMOS_TRAINER_MIXED_PRECISION", " ")
    assert wrapper._accelerate_prefix() == ["accelerate", "launch", "--num_processes", "2"]


def test_cosmos_vast_trainer_command_modes(monkeypatch, tmp_path: Path) -> None:
    args = _args(tmp_path)

    monkeypatch.setenv("COSMOS_TRAINER_COMMAND", "python train.py --cfg {trainer_config_path} --out {output_dir} --capture {capture_root}")
    command = wrapper._trainer_command(args)
    assert command[:4] == ["python", "train.py", "--cfg", str((tmp_path / "trainer.json").resolve())]
    assert command[-2:] == ["--capture", str((tmp_path / "capture").resolve())]

    monkeypatch.delenv("COSMOS_TRAINER_COMMAND", raising=False)
    monkeypatch.delenv("COSMOS_TRAINING_COMMAND", raising=False)
    monkeypatch.delenv("COSMOS_TRAINER_ENTRYPOINT", raising=False)
    monkeypatch.delenv("COSMOS_VAST_TRAINER_ENTRYPOINT", raising=False)
    with pytest.raises(RuntimeError, match="COSMOS_TRAINER_ENTRYPOINT is not configured"):
        wrapper._trainer_command(args)

    monkeypatch.setenv("COSMOS_TRAINER_ENTRYPOINT", "blueprint.train")
    monkeypatch.setenv("COSMOS_TRAINER_ENTRYPOINT_MODE", "module")
    monkeypatch.setenv("COSMOS_TRAINER_LAUNCHER", "python")
    monkeypatch.setenv("COSMOS_TRAINER_EXTRA_ARGS", "--epochs 1")
    assert wrapper._trainer_command(args) == [
        "python",
        "-m",
        "blueprint.train",
        "--trainer-config",
        str((tmp_path / "trainer.json").resolve()),
        "--output-dir",
        str((tmp_path / "out").resolve()),
        "--export-manifest",
        str((tmp_path / "export.json").resolve()),
        "--capture-root",
        str((tmp_path / "capture").resolve()),
        "--paired-reference-target",
        "paired.json",
        "--k-reference-conditioning",
        "krefs.json",
        "--train-val-split",
        "split.json",
        "--epochs",
        "1",
    ]

    monkeypatch.setenv("COSMOS_TRAINER_ENTRYPOINT_MODE", "script")
    monkeypatch.setenv("COSMOS_TRAINER_LAUNCHER", "torchrun")
    monkeypatch.setenv("COSMOS_TRAINER_NUM_PROCESSES", "3")
    no_optional = _args(tmp_path, export_manifest=None, capture_root=None, paired_reference_target=None, k_reference_conditioning=None, train_val_split=None)
    assert wrapper._trainer_command(no_optional)[:3] == ["torchrun", "--nproc_per_node", "3"]

    monkeypatch.setenv("COSMOS_TRAINER_LAUNCHER", "accelerate")
    assert wrapper._trainer_command(no_optional)[:3] == ["accelerate", "launch", "--num_processes"]

    monkeypatch.setenv("COSMOS_TRAINER_LAUNCHER", "unsupported")
    with pytest.raises(RuntimeError, match="Unsupported COSMOS_TRAINER_LAUNCHER"):
        wrapper._trainer_command(args)


def test_cosmos_vast_main_print_run_and_module_guard(monkeypatch, tmp_path: Path, capsys) -> None:
    trainer_config = tmp_path / "trainer.json"
    trainer_config.write_text("{}", encoding="utf-8")
    base_argv = ["--trainer-config", str(trainer_config), "--output-dir", str(tmp_path / "out")]

    monkeypatch.setenv("COSMOS_TRAINER_MOCK_SUCCESS", "1")
    assert wrapper.main(base_argv) == 0
    assert (tmp_path / "out" / "adapter_model.safetensors").is_file()

    monkeypatch.delenv("COSMOS_TRAINER_MOCK_SUCCESS", raising=False)
    monkeypatch.setenv("COSMOS_TRAINER_COMMAND", "python train.py")
    assert wrapper.main(base_argv + ["--print-command"]) == 0
    assert json.loads(capsys.readouterr().out)["command"] == ["python", "train.py"]

    seen: dict[str, object] = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["timeout"] = kwargs["timeout"]
        seen["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)
    assert wrapper.main(base_argv + ["--timeout-seconds", "1"]) == 7
    assert seen["command"] == ["python", "train.py"]
    assert seen["timeout"] == 1
    assert "COSMOS_TRAINER_OUTPUT_DIR" in seen["env"]

    monkeypatch.setenv("COSMOS_TRAINER_MOCK_SUCCESS", "1")
    monkeypatch.setattr(sys, "argv", ["cosmos-vast", *base_argv])
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as excinfo:
            runpy.run_module("blueprint_pipeline.synthesis.cosmos_vast_training_wrapper", run_name="__main__")
    assert excinfo.value.code == 0
