"""Concrete Cosmos LoRA trainer wrapper for GPU VM / Vast instances."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Sequence

from ..model_access_env import normalize_model_access_env


normalize_model_access_env()


def _write_mock_checkpoint(output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adapter_model.safetensors").write_bytes(b"mock-cosmos-adapter")
    (output_dir / "trainer_state.json").write_text('{"mode":"mock_success"}\n', encoding="utf-8")
    return 0


def _standard_env(args: argparse.Namespace) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "COSMOS_TRAINER_CONFIG_PATH": str(Path(args.trainer_config).resolve()),
            "COSMOS_TRAINER_OUTPUT_DIR": str(Path(args.output_dir).resolve()),
            "COSMOS_EXPORT_MANIFEST_PATH": str(Path(args.export_manifest).resolve())
            if args.export_manifest
            else "",
            "COSMOS_CAPTURE_ROOT": str(Path(args.capture_root).resolve())
            if args.capture_root
            else "",
            "COSMOS_PAIRED_REFERENCE_TARGET_PATH": args.paired_reference_target or "",
            "COSMOS_K_REFERENCE_CONDITIONING_PATH": args.k_reference_conditioning or "",
            "COSMOS_TRAIN_VAL_SPLIT_PATH": args.train_val_split or "",
            "COSMOS_MODEL_ID": str(os.getenv("COSMOS_MODEL_ID") or "nvidia/Cosmos-Predict2.5-2B"),
            "COSMOS_MODEL_REVISION": str(
                os.getenv("COSMOS_MODEL_REVISION") or "0d37c7498f54cee3c599d438d895a0a4a8608064"
            ),
        }
    )
    return env


def _accelerate_prefix() -> list[str]:
    mixed_precision = str(os.getenv("COSMOS_TRAINER_MIXED_PRECISION") or "bf16").strip()
    num_processes = max(1, int(os.getenv("COSMOS_TRAINER_NUM_PROCESSES") or "1"))
    prefix = ["accelerate", "launch", "--num_processes", str(num_processes)]
    if mixed_precision:
        prefix.extend(["--mixed_precision", mixed_precision])
    return prefix


def _trainer_command(args: argparse.Namespace) -> list[str]:
    trainer_command_template = str(
        os.getenv("COSMOS_TRAINER_COMMAND") or os.getenv("COSMOS_TRAINING_COMMAND") or ""
    ).strip()
    if trainer_command_template:
        mapping = {
            "trainer_config_path": str(Path(args.trainer_config).resolve()),
            "output_dir": str(Path(args.output_dir).resolve()),
            "export_manifest_path": str(Path(args.export_manifest).resolve())
            if args.export_manifest
            else "",
            "capture_root": str(Path(args.capture_root).resolve()) if args.capture_root else "",
            "paired_reference_target_path": args.paired_reference_target or "",
            "k_reference_conditioning_path": args.k_reference_conditioning or "",
            "train_val_split_path": args.train_val_split or "",
        }
        return shlex.split(trainer_command_template.format_map(mapping))

    entrypoint = str(
        os.getenv("COSMOS_TRAINER_ENTRYPOINT") or os.getenv("COSMOS_VAST_TRAINER_ENTRYPOINT") or ""
    ).strip()
    if not entrypoint:
        raise RuntimeError(
            "COSMOS_TRAINER_ENTRYPOINT is not configured. "
            "Point it at the real trainer script or set COSMOS_TRAINER_COMMAND."
        )

    entrypoint_mode = str(os.getenv("COSMOS_TRAINER_ENTRYPOINT_MODE") or "script").strip().lower()
    launcher = str(os.getenv("COSMOS_TRAINER_LAUNCHER") or "accelerate").strip().lower()

    command: list[str] = []
    if launcher == "accelerate":
        command.extend(_accelerate_prefix())
    elif launcher == "python":
        command.append("python")
    elif launcher == "torchrun":
        nproc = max(1, int(os.getenv("COSMOS_TRAINER_NUM_PROCESSES") or "1"))
        command.extend(["torchrun", "--nproc_per_node", str(nproc)])
    else:
        raise RuntimeError(f"Unsupported COSMOS_TRAINER_LAUNCHER: {launcher}")

    if entrypoint_mode == "module":
        command.extend(["-m", entrypoint])
    else:
        command.append(entrypoint)

    command.extend(
        [
            "--trainer-config",
            str(Path(args.trainer_config).resolve()),
            "--output-dir",
            str(Path(args.output_dir).resolve()),
        ]
    )
    if args.export_manifest:
        command.extend(["--export-manifest", str(Path(args.export_manifest).resolve())])
    if args.capture_root:
        command.extend(["--capture-root", str(Path(args.capture_root).resolve())])
    if args.paired_reference_target:
        command.extend(["--paired-reference-target", args.paired_reference_target])
    if args.k_reference_conditioning:
        command.extend(["--k-reference-conditioning", args.k_reference_conditioning])
    if args.train_val_split:
        command.extend(["--train-val-split", args.train_val_split])

    extra_args = str(os.getenv("COSMOS_TRAINER_EXTRA_ARGS") or "").strip()
    if extra_args:
        command.extend(shlex.split(extra_args))
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the configured Cosmos LoRA trainer on a GPU VM"
    )
    parser.add_argument("--trainer-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--export-manifest", default=None)
    parser.add_argument("--capture-root", default=None)
    parser.add_argument("--paired-reference-target", default=None)
    parser.add_argument("--k-reference-conditioning", default=None)
    parser.add_argument("--train-val-split", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    parser.add_argument("--print-command", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if str(os.getenv("COSMOS_TRAINER_MOCK_SUCCESS") or "").strip() == "1":
        return _write_mock_checkpoint(Path(args.output_dir))

    command = _trainer_command(args)
    if args.print_command:
        print(json.dumps({"command": command}, indent=2))
        return 0

    timeout = max(
        1, int(args.timeout_seconds or int(os.getenv("COSMOS_TRAINER_TIMEOUT_SECONDS") or "86400"))
    )
    env = _standard_env(args)
    result = subprocess.run(
        command,
        env=env,
        text=True,
        timeout=timeout,
        check=False,
    )
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
