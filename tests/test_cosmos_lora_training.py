from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.synthesis.cosmos_lora_training import run_cosmos_lora_training


def test_run_cosmos_lora_training_records_completed_checkpoint(tmp_path: Path) -> None:
    capture_root = tmp_path / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    export_root = capture_root / "pipeline" / "cosmos_training_export"
    export_root.mkdir(parents=True)

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
        training_command=f"python3 {trainer_script} {{output_dir}}",
        timeout_seconds=60,
    )

    assert manifest["status"] == "completed"
    assert Path(str(manifest["checkpoint_path"])).is_file()
    assert Path(str(manifest["log_path"])).is_file()
