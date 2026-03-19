from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from blueprint_pipeline.synthesis.cosmos_vast_training_wrapper import _trainer_command


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        trainer_config=str(tmp_path / "trainer_config.json"),
        output_dir=str(tmp_path / "output"),
        export_manifest=str(tmp_path / "export_manifest.json"),
        capture_root=str(tmp_path / "capture"),
        paired_reference_target=str(tmp_path / "paired.json"),
        k_reference_conditioning=str(tmp_path / "conditioning.json"),
        train_val_split=str(tmp_path / "split.json"),
        timeout_seconds=None,
        print_command=False,
    )


def test_trainer_command_accepts_legacy_training_command_alias(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("COSMOS_TRAINER_COMMAND", raising=False)
    monkeypatch.setenv(
        "COSMOS_TRAINING_COMMAND",
        "python train.py --trainer-config {trainer_config_path} --output-dir {output_dir}",
    )

    command = _trainer_command(_args(tmp_path))

    assert command == [
        "python",
        "train.py",
        "--trainer-config",
        str((tmp_path / "trainer_config.json").resolve()),
        "--output-dir",
        str((tmp_path / "output").resolve()),
    ]
