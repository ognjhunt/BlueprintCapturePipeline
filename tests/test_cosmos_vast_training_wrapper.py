from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.synthesis.cosmos_vast_training_wrapper import main


def test_cosmos_vast_training_wrapper_mock_success(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    trainer_config = tmp_path / "trainer_config.json"
    trainer_config.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("COSMOS_TRAINER_MOCK_SUCCESS", "1")

    exit_code = main(
        [
            "--trainer-config",
            str(trainer_config),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "adapter_model.safetensors").is_file()
    assert (output_dir / "trainer_state.json").is_file()
