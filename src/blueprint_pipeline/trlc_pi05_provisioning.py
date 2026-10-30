"""Bounded community TRLC Pi0.5 bootstrap inside the existing admitted canonical GPU worker.

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
    DEADLINE_ENV, TransferBudget, fetch_exact, remaining_seconds,
)
from .trlc_pi05_assets import MODEL_ID, MODEL_REVISION, verify_checkpoint_directory, verify_tokenizer_directory
from .trlc_pi05_policy import TRLCPi05PolicyClient
from .trlc_pi05_runtime import checkpoint_identity, policy_binding, serialized_binding, validate_execution_spec


def _start_server(*, python: Path, runtime: Path, output: Path, root: Path, spec: dict, deadline: float) -> dict:
    binding_path = root / "binding.json"
    binding_path.write_text(json.dumps(serialized_binding(spec["policy_spec"]), sort_keys=True))
    source = root / "source/lerobot/src"
    bootstrap = "import sys,runpy;sys.path[:0]=" + repr([str(runtime), str(source)]) + ";runpy.run_module('blueprint_pipeline.trlc_pi05_policy_server',run_name='__main__')"
    command = ["/isaac-sim/python.sh", "-I", "-B", "-c", bootstrap,
               "--checkpoint", str(root / "checkpoint"), "--tokenizer", str(runtime / "trlc_tokenizer"),
               "--binding", str(binding_path)]
    env = {key: value for key, value in os.environ.items()
           if key not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "PYTHONHOME",
                          "PYTHONPATH", "VIRTUAL_ENV", "PYTHONEXE"}}
    env.update(PYTHONEXE=str(python), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
               HF_HOME=str(root / "hub-cache"))
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
                raise ValueError("trlc_pi05_server_exited_before_ready")
            client = TRLCPi05PolicyClient(binding=policy_binding(spec["policy_spec"]),
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
        receipt.update(status="blocked", blocker="trlc_pi05_server_start_failed",
                       process_terminated=process is None or process.poll() is not None)
        receipt_path.write_text(json.dumps(receipt, sort_keys=True))
        raise


def _run(argv, *, deadline, python=None):
    environment = {key: value for key, value in os.environ.items()
                   if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONEXE"}}
    environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PIP_NO_INDEX="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if python is not None:
        environment["PYTHONEXE"] = str(python)
    subprocess.run(argv, env=environment, check=True, timeout=remaining_seconds(deadline))


def provision(runtime: Path, output: Path, spec: dict, *, root: Path = Path("/opt/blueprint-trlc-pi05")) -> dict:
    from .native_g1_beta_dependency_profile import (DEPENDENCY_MANIFEST_FILENAME,
        artifact_manifest, verify_wheelhouse, TOTAL_DOWNLOAD_BYTES)
    from .trlc_pi05_source import extract_source, SOURCE_ARCHIVE, SOURCE_MANIFEST, verify_source_directory

    validate_execution_spec(spec)
    deadline = float(os.environ.get(DEADLINE_ENV, "nan"))
    remaining_seconds(deadline)
    dependencies = artifact_manifest(runtime / DEPENDENCY_MANIFEST_FILENAME)
    checkpoint = checkpoint_identity()
    budget = TransferBudget(TOTAL_DOWNLOAD_BYTES + sum(row["size_bytes"] for row in checkpoint["files"]))
    if root.exists() or root.is_symlink():
        raise ValueError("trlc_pi05_existing_runtime_refused")
    root.mkdir(parents=True)
    output.mkdir(parents=True, exist_ok=True)
    extract_source(runtime / SOURCE_ARCHIVE, runtime / SOURCE_MANIFEST, root / "source")
    verify_tokenizer_directory(runtime / "trlc_tokenizer", deadline_epoch=deadline)
    downloads, paths = [], {}
    for row in dependencies["wheels"]:
        if row["source"] == "runtime_source_packet":
            path = runtime / "provisioned_runtime_sources" / row["archive_path"]
            if (path.is_symlink() or not path.is_file()
                    or not path.resolve().is_relative_to(runtime / "provisioned_runtime_sources")):
                raise ValueError("trlc_pi05_native_dependency_missing")
        else:
            path = root / "wheels" / row["filename"]
            downloads.append(fetch_exact(row, path, deadline=deadline, budget=budget))
        paths[row["filename"]] = path
    wheel_receipt = verify_wheelhouse(paths, deadline_epoch=deadline)
    environment = root / "policy-runtime"
    python = environment / "bin/python"
    _run(["/isaac-sim/python.sh", "-m", "venv", "--copies", str(environment)], deadline=deadline)
    requirements = root / "offline-requirements.txt"
    requirements.write_text("\n".join(str(paths[row["filename"]]) + " --hash=sha256:" + row["sha256"]
                                     for row in dependencies["wheels"]) + "\n")
    launcher = ["/isaac-sim/python.sh", "-I", "-B"]
    _run([*launcher, "-m", "pip", "install", "--no-index", "--no-deps", "--require-hashes", "--no-compile",
          "-r", str(requirements)], deadline=deadline, python=python)
    _run([*launcher, "-m", "pip", "check"], deadline=deadline, python=python)
    source = root / "source/lerobot/src"
    probe = ("import sys,pathlib,importlib.metadata as m;assert sys.version_info[:2]==(3,12);"
             + f"sys.path.insert(0,{str(source)!r});"
             + "import torch,torchvision,transformers,lerobot;from lerobot.policies.pi05.modeling_pi05 import PI05Policy;"
             + f"assert pathlib.Path(lerobot.__file__).resolve().is_relative_to(pathlib.Path({str(source)!r}));"
             + f"assert all(m.version(k)==v for k,v in {dependencies['core_versions']!r}.items());"
             + "assert torch.version.cuda=='12.8'")
    _run([*launcher, "-c", probe], deadline=deadline, python=python)
    for file in checkpoint["files"]:
        row = {**file, "url": f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{quote(file['path'])}?download=true"}
        downloads.append(fetch_exact(row, root / "checkpoint" / file["path"], deadline=deadline, budget=budget))
    verify_checkpoint_directory(root / "checkpoint", deadline_epoch=deadline)
    verify_source_directory(root / "source", runtime / SOURCE_MANIFEST)
    remaining_seconds(deadline)
    server = _start_server(python=python, runtime=runtime, output=output, root=root, spec=spec, deadline=deadline)
    return {"status": "ready", "checkpoint_bytes_verified": True, "download_receipts": downloads,
            "server": server, "peak_gpu_memory_measured": False, "wheelhouse_verification": wheel_receipt,
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
        code = reason if re.fullmatch(r"(?:trlc_pi05|local_policy)_[a-z0-9_]{1,120}", reason) else "trlc_pi05_provisioning_failed:" + type(exc).__name__
        result = {"status": "blocked", "blockers": [code], "candidate_policy_queried": False}
    (output / "trlc_pi05_provisioning.v1.json").write_text(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
