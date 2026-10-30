"""Bounded YAM bootstrap inside the existing admitted canonical GPU worker.

Checkpoint Python, tokenizer, normalization and weights are verified together
before offline model loading. There is no resolver, hub fallback or allocation.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import quote

from .local_policy_provisioning import (
    DEADLINE_ENV, TransferBudget, fetch_exact, remaining_seconds, _command, _extract_interpreter, _install_wheels,
)
from .molmoact2_yam_assets import MODEL_ID, MODEL_REVISION, verify_checkpoint_directory
from .molmoact2_yam_policy import MolmoAct2YAMPolicyClient
from .molmoact2_yam_runtime import checkpoint_identity, policy_binding, serialized_binding, validate_execution_spec


def _start_server(*, python: Path, runtime: Path, output: Path, root: Path, spec: dict, deadline: float) -> dict:
    binding_path = root / "binding.json"
    binding_path.write_text(json.dumps(serialized_binding(spec["policy_spec"]), sort_keys=True))
    command = [str(python), "-m", "blueprint_pipeline.molmoact2_yam_policy_server",
               "--checkpoint", str(root / "checkpoint"), "--binding", str(binding_path)]
    env = {key: value for key, value in os.environ.items()
           if key not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "PYTHONHOME", "VIRTUAL_ENV"}}
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = ":".join(part for part in env["LD_LIBRARY_PATH"].split(":") if "/isaac-sim" not in part)
    env.update(PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
               HF_MODULES_CACHE=str(root / "module-cache"), HF_HOME=str(root / "hub-cache"))
    log = output / f"adp009d_policy_server.{spec['candidate_id']}.log"
    receipt_path = output / f"adp009d_policy_server_receipt.{spec['candidate_id']}.json"
    receipt = {"schema_version": "adp009d_policy_server_receipt.v1", "candidate_id": spec["candidate_id"],
               "status": "starting", "command": command, "candidate_policy_queried": False,
               "candidate_inference_performed": False, "checkpoint_bytes_verified": True}
    process = None
    try:
        with log.open("wb") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        receipt["server_pid"] = process.pid
        receipt_path.write_text(json.dumps(receipt, sort_keys=True))
        readiness_deadline = min(deadline, time.time() + 900)
        while True:
            if process.poll() is not None:
                raise ValueError("molmoact2_server_exited_before_ready")
            client = MolmoAct2YAMPolicyClient(binding=policy_binding(spec["policy_spec"]),
                timeout_seconds=min(10, remaining_seconds(readiness_deadline)))
            try:
                # Reset binds the exact model/session identity and performs no inference.
                client.reset()
                receipt.update(status="ready", handshake_completed=True, exact_policy_identity_verified=True)
                break
            except (OSError, ValueError, TimeoutError):
                time.sleep(min(1, remaining_seconds(readiness_deadline)))
            finally:
                client.close()
        receipt_path.write_text(json.dumps(receipt, sort_keys=True))
        return receipt
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        receipt.update(status="blocked", blocker="molmoact2_server_start_failed",
                       process_terminated=process is None or process.poll() is not None)
        receipt_path.write_text(json.dumps(receipt, sort_keys=True))
        raise


def provision(runtime: Path, output: Path, spec: dict, *, root: Path = Path("/opt/blueprint-molmoact2-yam")) -> dict:
    from .molmoact2_yam_dependencies import DEPENDENCY_MANIFEST_FILENAME, verified_dependency_manifest

    validate_execution_spec(spec)
    deadline = float(os.environ.get(DEADLINE_ENV, "nan"))
    remaining_seconds(deadline)
    dependencies = verified_dependency_manifest(runtime / DEPENDENCY_MANIFEST_FILENAME)
    checkpoint = checkpoint_identity()
    budget = TransferBudget(dependencies["total_download_bytes"] + sum(row["size_bytes"] for row in checkpoint["files"]))
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    downloads = []
    interpreter = dependencies["interpreter"]
    interpreter_archive = root / interpreter["filename"]
    downloads.append(fetch_exact(interpreter, interpreter_archive, deadline=deadline, budget=budget))
    _extract_interpreter(interpreter_archive, root / "interpreter", maximum_bytes=interpreter["maximum_expanded_bytes"])
    python = root / "interpreter" / interpreter["python_relative_path"]
    _command([str(python), "-c", "import sys;assert sys.version.split()[0]==" + repr(interpreter["version"])], deadline=deadline)
    wheels = root / "wheels"
    wheels.mkdir(exist_ok=True)
    for row in dependencies["server_wheels"]:
        downloads.append(fetch_exact(row, wheels / row["filename"], deadline=deadline, budget=budget))
    _install_wheels(python, dependencies["server_wheels"], wheels, deadline=deadline)
    for file in checkpoint["files"]:
        row = {**file, "url": f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{quote(file['path'])}?download=true"}
        downloads.append(fetch_exact(row, root / "checkpoint" / file["path"], deadline=deadline, budget=budget))
    verify_checkpoint_directory(root / "checkpoint", deadline_epoch=deadline)
    remaining_seconds(deadline)
    server = _start_server(python=python, runtime=runtime, output=output, root=root, spec=spec, deadline=deadline)
    return {"status": "ready", "checkpoint_bytes_verified": True, "download_receipts": downloads,
            "server": server, "peak_gpu_memory_measured": False,
            "maximum_download_bytes": budget.maximum_bytes, "attempted_download_bytes": budget.attempted_bytes}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        result = provision(Path(args.runtime), output, json.loads(Path(args.spec).read_text()))
    except Exception as exc:
        reason = str(exc)
        code = reason if re.fullmatch(r"(?:molmoact2|local_policy)_[a-z0-9_]{1,120}", reason) else "molmoact2_provisioning_failed:" + type(exc).__name__
        result = {"status": "blocked", "blockers": [code], "candidate_policy_queried": False}
    (output / "molmoact2_yam_provisioning.v1.json").write_text(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
