"""Executable Cosmos Predict LoRA training runner and manifest writer."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ..common import ensure_dir, read_json, utc_now_iso, write_json
from ..local_capture import resolve_local_capture_context


def _optional_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return read_json(path)


def _discover_adapter_checkpoint(root_dir: Path) -> Optional[Path]:
    direct = root_dir / "adapter_model.safetensors"
    if direct.is_file():
        return direct
    for candidate in sorted(root_dir.rglob("adapter_model.safetensors")):
        if candidate.is_file():
            return candidate
    return None


def _expand_training_command(template: str, values: Mapping[str, Any]) -> str:
    mapping = {key: shlex.quote(str(value)) for key, value in values.items()}
    return template.format_map(mapping)


def _normalize_python_command(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        return command
    if not parts:
        return command
    if parts[0] in {"python", "python3"} and not shutil.which(parts[0]):
        parts[0] = sys.executable
        return shlex.join(parts)
    return command


def run_cosmos_lora_training(
    *,
    capture_root: str | Path,
    training_command: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    context = resolve_local_capture_context(capture_root)
    export_root = context.pipeline_root / "cosmos_training_export"
    export_manifest_path = export_root / "manifest.json"
    trainer_config_path = export_root / "trainer_config.json"
    checkpoint_layout_path = export_root / "checkpoint_layout.json"
    run_manifest_path = export_root / "training_run_manifest.json"
    run_request_path = export_root / "training_run_request.json"
    log_path = export_root / "training_run.log"

    export_manifest = _optional_json(export_manifest_path)
    trainer_config = _optional_json(trainer_config_path)
    checkpoint_layout = _optional_json(checkpoint_layout_path)

    checkpoint_root_dir = Path(
        str(checkpoint_layout.get("root_dir") or export_root / "checkpoints")
    ).resolve()
    ensure_dir(checkpoint_root_dir)
    run_id = utc_now_iso().replace(":", "").replace("+00:00", "Z")
    run_dir = checkpoint_root_dir / f"run_{run_id}"
    ensure_dir(run_dir)

    command_template = (
        str(training_command or "").strip()
        or str(os.getenv("COSMOS_TRAINING_COMMAND") or "").strip()
    )
    request = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "capture_id": context.capture_id,
        "scene_id": context.scene_id,
        "export_manifest_path": str(export_manifest_path.resolve()),
        "trainer_config_path": str(trainer_config_path.resolve()),
        "checkpoint_layout_path": str(checkpoint_layout_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_root_dir": str(checkpoint_root_dir.resolve()),
        "source_mode": export_manifest.get("source_mode"),
        "training_command_template": command_template or None,
    }
    write_json(run_request_path, request)

    if export_manifest.get("status") != "ready":
        manifest = {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "reason": "training_export_not_ready",
            "capture_id": context.capture_id,
            "scene_id": context.scene_id,
            "export_manifest_path": str(export_manifest_path.resolve()),
            "trainer_config_path": str(trainer_config_path.resolve()),
            "checkpoint_root_dir": str(checkpoint_root_dir.resolve()),
            "run_dir": str(run_dir.resolve()),
            "checkpoint_path": None,
        }
        write_json(run_manifest_path, manifest)
        return manifest

    if not command_template:
        manifest = {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "reason": "missing_training_command",
            "capture_id": context.capture_id,
            "scene_id": context.scene_id,
            "export_manifest_path": str(export_manifest_path.resolve()),
            "trainer_config_path": str(trainer_config_path.resolve()),
            "checkpoint_root_dir": str(checkpoint_root_dir.resolve()),
            "run_dir": str(run_dir.resolve()),
            "checkpoint_path": None,
            "blockers": [
                "Set COSMOS_TRAINING_COMMAND or pass training_command with placeholders such as "
                "{trainer_config_path} and {output_dir}.",
            ],
        }
        write_json(run_manifest_path, manifest)
        return manifest

    expanded_command = _expand_training_command(
        command_template,
        {
            "capture_root": str(context.capture_root.resolve()),
            "export_root": str(export_root.resolve()),
            "export_manifest_path": str(export_manifest_path.resolve()),
            "trainer_config_path": str(trainer_config_path.resolve()),
            "checkpoint_layout_path": str(checkpoint_layout_path.resolve()),
            "checkpoint_root_dir": str(checkpoint_root_dir.resolve()),
            "output_dir": str(run_dir.resolve()),
            "paired_reference_target_path": str(export_manifest.get("paired_reference_target_path") or ""),
            "k_reference_conditioning_path": str(export_manifest.get("k_reference_conditioning_path") or ""),
            "train_val_split_path": str(export_manifest.get("train_val_split_path") or ""),
            "source_mode": str(export_manifest.get("source_mode") or ""),
        },
    )
    expanded_command = _normalize_python_command(expanded_command)

    try:
        command_argv = shlex.split(expanded_command)
    except ValueError:
        command_argv = []
    if not command_argv:
        manifest = {
            "schema_version": "v1",
            "generated_at": utc_now_iso(),
            "status": "blocked",
            "reason": "invalid_training_command",
            "capture_id": context.capture_id,
            "scene_id": context.scene_id,
            "export_manifest_path": str(export_manifest_path.resolve()),
            "trainer_config_path": str(trainer_config_path.resolve()),
            "checkpoint_root_dir": str(checkpoint_root_dir.resolve()),
            "run_dir": str(run_dir.resolve()),
            "checkpoint_path": None,
            "blockers": ["training_command_could_not_be_parsed_as_nonempty_argv"],
        }
        write_json(run_manifest_path, manifest)
        return manifest

    timeout = max(1, int(timeout_seconds or int(os.getenv("COSMOS_TRAINING_TIMEOUT_SECONDS", "3600"))))
    result = subprocess.run(
        command_argv,
        shell=False,
        cwd=str(context.capture_root.resolve()),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "COSMOS_TRAINING_OUTPUT_DIR": str(run_dir.resolve())},
    )
    log_path.write_text(
        f"$ {shlex.join(command_argv)}\n\n[stdout]\n{result.stdout}\n\n[stderr]\n{result.stderr}\n",
        encoding="utf-8",
    )

    checkpoint_path = _discover_adapter_checkpoint(run_dir)
    status = "completed" if result.returncode == 0 and checkpoint_path else "failed"
    reason = None
    if result.returncode != 0:
        reason = f"trainer_exit_code:{result.returncode}"
    elif checkpoint_path is None:
        reason = "adapter_checkpoint_missing"

    manifest = {
        "schema_version": "v1",
        "generated_at": utc_now_iso(),
        "status": status,
        "reason": reason,
        "capture_id": context.capture_id,
        "scene_id": context.scene_id,
        "export_manifest_path": str(export_manifest_path.resolve()),
        "trainer_config_path": str(trainer_config_path.resolve()),
        "checkpoint_layout_path": str(checkpoint_layout_path.resolve()),
        "checkpoint_root_dir": str(checkpoint_root_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path else None,
        "training_command": expanded_command,
        "log_path": str(log_path.resolve()),
        "source_mode": export_manifest.get("source_mode"),
        "model_family": trainer_config.get("model_family"),
        "adapter_type": trainer_config.get("adapter_type"),
        "train_count": export_manifest.get("train_count"),
        "val_count": export_manifest.get("val_count"),
    }
    write_json(run_manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Cosmos Predict LoRA training for a capture export")
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--training-command", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    args = parser.parse_args()

    manifest = run_cosmos_lora_training(
        capture_root=args.capture_root,
        training_command=args.training_command,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"[cosmos-lora-training] status={manifest['status']}")
    if manifest.get("checkpoint_path"):
        print(f"[cosmos-lora-training] checkpoint={manifest['checkpoint_path']}")
    if manifest.get("reason"):
        print(f"[cosmos-lora-training] reason={manifest['reason']}")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
