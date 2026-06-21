from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from blueprint_pipeline.synthesis import cosmos_lora_training
from blueprint_pipeline.synthesis.cosmos_lora_training import run_cosmos_lora_training


def _write_ready_export(capture_root: Path) -> Path:
    export_root = capture_root / "pipeline" / "cosmos_training_export"
    export_root.mkdir(parents=True, exist_ok=True)
    (export_root / "manifest.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "source_mode": "dense_index",
                "paired_reference_target_path": str(export_root / "paired_reference_target.jsonl"),
                "k_reference_conditioning_path": str(export_root / "k_reference_conditioning.jsonl"),
                "train_val_split_path": str(export_root / "train_val_split.json"),
                "train_count": 4,
                "val_count": 1,
            }
        ),
        encoding="utf-8",
    )
    (export_root / "trainer_config.json").write_text(
        json.dumps({"model_family": "nvidia/Cosmos-Predict2.5-2B", "adapter_type": "lora"}),
        encoding="utf-8",
    )
    (export_root / "checkpoint_layout.json").write_text(
        json.dumps({"root_dir": str((export_root / "checkpoints").resolve())}),
        encoding="utf-8",
    )
    return export_root


def test_run_cosmos_lora_training_records_completed_checkpoint(tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_ready_export(capture_root)

    trainer_script = tmp_path / "trainer.py"
    trainer_script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "output_dir = Path(sys.argv[1])",
                "output_dir.mkdir(parents=True, exist_ok=True)",
                "(output_dir / 'adapter_model.safetensors').write_bytes(b'adapter')",
                "(output_dir / 'trainer_state.json').write_text('{}', encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )

    manifest = run_cosmos_lora_training(
        capture_root=capture_root,
        training_command=f"python {trainer_script} {{output_dir}}",
        timeout_seconds=60,
    )

    assert manifest["status"] == "completed"
    assert Path(str(manifest["checkpoint_path"])).is_file()
    assert Path(str(manifest["log_path"])).is_file()


def test_run_cosmos_lora_training_falls_back_to_sys_executable_when_python_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_ready_export(capture_root)

    trainer_script = tmp_path / "trainer.py"
    trainer_script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "output_dir = Path(sys.argv[1])",
                "output_dir.mkdir(parents=True, exist_ok=True)",
                "(output_dir / 'adapter_model.safetensors').write_bytes(b'adapter')",
                "(output_dir / 'trainer_state.json').write_text('{}', encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("blueprint_pipeline.synthesis.cosmos_lora_training.shutil.which", lambda _cmd: None)

    manifest = run_cosmos_lora_training(
        capture_root=capture_root,
        training_command=f"python {trainer_script} {{output_dir}}",
        timeout_seconds=60,
    )

    assert manifest["status"] == "completed"
    assert Path(str(manifest["checkpoint_path"])).is_file()


def test_cosmos_lora_helpers_blocked_failed_and_cli(monkeypatch, capsys, tmp_path: Path) -> None:
    assert cosmos_lora_training._optional_json(tmp_path / "missing.json") == {}
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    assert cosmos_lora_training._discover_adapter_checkpoint(checkpoints) is None
    direct = checkpoints / "adapter_model.safetensors"
    direct.write_bytes(b"direct")
    assert cosmos_lora_training._discover_adapter_checkpoint(checkpoints) == direct
    direct.unlink()
    nested = checkpoints / "nested" / "adapter_model.safetensors"
    nested.parent.mkdir()
    nested.write_bytes(b"nested")
    assert cosmos_lora_training._discover_adapter_checkpoint(checkpoints) == nested
    assert cosmos_lora_training._expand_training_command("{a}/{b}", {"a": "x", "b": 2}) == "x/2"
    assert cosmos_lora_training._normalize_python_command('"unterminated') == '"unterminated'
    assert cosmos_lora_training._normalize_python_command("") == ""

    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    export_root = capture_root / "pipeline" / "cosmos_training_export"
    export_root.mkdir(parents=True)
    (export_root / "manifest.json").write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    blocked = run_cosmos_lora_training(capture_root=capture_root, training_command="echo ok")
    assert blocked["reason"] == "training_export_not_ready"

    (export_root / "manifest.json").write_text(json.dumps({"status": "ready"}), encoding="utf-8")
    missing_command = run_cosmos_lora_training(capture_root=capture_root)
    assert missing_command["reason"] == "missing_training_command"

    _write_ready_export(capture_root)
    monkeypatch.setattr(
        cosmos_lora_training.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess("cmd", 7, stdout="out", stderr="err"),
    )
    failed = run_cosmos_lora_training(capture_root=capture_root, training_command="trainer {output_dir}", timeout_seconds=1)
    assert failed["reason"] == "trainer_exit_code:7"

    monkeypatch.setattr(
        cosmos_lora_training.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess("cmd", 0, stdout="out", stderr=""),
    )
    missing_checkpoint = run_cosmos_lora_training(capture_root=capture_root, training_command="trainer {output_dir}", timeout_seconds=1)
    assert missing_checkpoint["reason"] == "adapter_checkpoint_missing"

    monkeypatch.setattr(sys, "argv", ["cosmos-lora-training", "--capture-root", str(capture_root), "--training-command", "trainer {output_dir}"])
    monkeypatch.setattr(cosmos_lora_training, "run_cosmos_lora_training", lambda **_kwargs: {"status": "completed", "checkpoint_path": "/tmp/adapter"})
    assert cosmos_lora_training.main() == 0
    assert "checkpoint=/tmp/adapter" in capsys.readouterr().out
    monkeypatch.setattr(cosmos_lora_training, "run_cosmos_lora_training", lambda **_kwargs: {"status": "failed", "reason": "bad"})
    assert cosmos_lora_training.main() == 1
    assert "reason=bad" in capsys.readouterr().out
