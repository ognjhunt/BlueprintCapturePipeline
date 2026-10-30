"""Bounded local OpenWAM bootstrap inside an already admitted canonical worker.

Every network input has a fixed release URL, byte bound and SHA256. Installation
and model loading use only these staged files. No provider is created here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import quote

from .openwam_piper_assets import EXPECTED_FILES, verify_checkpoint_directory
from .openwam_piper_policy import MODEL_ID, MODEL_REVISION, OpenWAMPiperPolicyClient
from .openwam_piper_runtime import policy_binding, serialized_binding, validate_execution_spec
from .openwam_piper_source import SOURCE_ARCHIVE, SOURCE_MANIFEST, extract_source

from .local_policy_provisioning import (
    DEADLINE_ENV, TransferBudget, fetch_exact, remaining_seconds,
    _sha256, _command, _extract_interpreter, _install_wheels,
)


def _start_server(*, python: Path, runtime: Path, output: Path, root: Path,
                  spec: dict, deadline: float) -> dict:
    binding_path = root / "binding.json"
    binding_path.write_text(json.dumps(serialized_binding(spec["policy_spec"]), sort_keys=True))
    command = [str(python), "-m", "blueprint_pipeline.openwam_piper_policy_server",
               "--checkpoint", str(root / "checkpoint"), "--source-root", str(root / "source"),
               "--source-manifest", str(runtime / SOURCE_MANIFEST), "--binding", str(binding_path)]
    env = {key: value for key, value in os.environ.items()
           if key not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "PYTHONHOME", "VIRTUAL_ENV"}}
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = ":".join(part for part in env["LD_LIBRARY_PATH"].split(":") if "/isaac-sim" not in part)
    env.update(PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", DIFFSYNTH_SKIP_DOWNLOAD="true", TOKENIZERS_PARALLELISM="false",
               WAM_ATTENTION_IMPL="sdpa", DIFFSYNTH_ATTENTION_IMPLEMENTATION="torch")
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
        # Identity handshake only; it performs no inference or policy-state advance.
        readiness_deadline = min(deadline, time.time() + 900)
        while True:
            if process.poll() is not None:
                raise ValueError("openwam_server_exited_before_ready")
            seconds = remaining_seconds(readiness_deadline)
            client = OpenWAMPiperPolicyClient(binding=policy_binding(spec["policy_spec"]),
                                              timeout_seconds=min(10, seconds))
            try:
                client._open()
                receipt.update(status="ready", handshake_completed=True, exact_policy_identity_verified=True)
                break
            except (OSError, ValueError, TimeoutError):
                remaining_seconds(readiness_deadline)
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
        receipt.update(status="blocked", blocker="openwam_server_start_failed",
                       process_terminated=process is None or process.poll() is not None)
        receipt_path.write_text(json.dumps(receipt, sort_keys=True))
        raise


def provision(runtime: Path, output: Path, spec: dict, *, root: Path = Path("/opt/blueprint-openwam-piper")) -> dict:
    from .openwam_piper_dependencies import verified_dependency_manifest

    validate_execution_spec(spec)
    deadline = float(os.environ.get(DEADLINE_ENV, "nan"))
    remaining_seconds(deadline)
    dependencies = verified_dependency_manifest(runtime / "openwam-piper-dependencies.v1.json")
    budget = TransferBudget(dependencies["total_download_bytes"] + sum(size for size, _ in EXPECTED_FILES.values()))
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    source = extract_source(runtime / SOURCE_ARCHIVE, runtime / SOURCE_MANIFEST, root / "source")
    downloads = []
    interpreter = dependencies["interpreter"]
    interpreter_archive = root / interpreter["filename"]
    downloads.append(fetch_exact(interpreter, interpreter_archive, deadline=deadline, budget=budget))
    _extract_interpreter(interpreter_archive, root / "interpreter",
                         maximum_bytes=interpreter["maximum_expanded_bytes"])
    python = root / "interpreter" / interpreter["python_relative_path"]
    _command([str(python), "-c", "import sys;assert sys.version.split()[0]==" + repr(interpreter["version"])], deadline=deadline)
    wheels = root / "wheels"
    wheels.mkdir(exist_ok=True)
    for row in {row["filename"]: row for row in [*dependencies["server_wheels"], *dependencies["thin_client_wheels"]]}.values():
        if row.get("bundled_path"):
            path = runtime / row["bundled_path"]
            if path.stat().st_size != row["size_bytes"] or _sha256(path) != "sha256:" + row["sha256"].removeprefix("sha256:"):
                raise ValueError("openwam_bundled_dependency_changed")
            shutil.copyfile(path, wheels / row["filename"])
        else:
            downloads.append(fetch_exact(row, wheels / row["filename"], deadline=deadline, budget=budget))
    _install_wheels(python, dependencies["server_wheels"], wheels, deadline=deadline)
    _install_wheels(Path(sys.executable), dependencies["thin_client_wheels"], wheels,
                    deadline=deadline, target=runtime / "openwam_client")
    client_wheels = runtime / "openwam_client_wheels"
    client_wheels.mkdir(exist_ok=True)
    for row in dependencies["thin_client_wheels"]:
        shutil.copyfile(wheels / row["filename"], client_wheels / row["filename"])
    from .openwam_piper_client_dependencies import activate_client_dependencies
    activate_client_dependencies(runtime)
    for name, (size, digest) in EXPECTED_FILES.items():
        row = {"url": f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{quote(name)}?download=true",
               "size_bytes": size, "sha256": "sha256:" + digest}
        downloads.append(fetch_exact(row, root / "checkpoint" / name, deadline=deadline, budget=budget))
    verify_checkpoint_directory(root / "checkpoint", deadline_epoch=deadline)
    remaining_seconds(deadline)
    server = _start_server(python=python, runtime=runtime, output=output, root=root, spec=spec, deadline=deadline)
    return {"status": "ready", "source_directory": str(source), "checkpoint_bytes_verified": True,
            "download_receipts": downloads, "server": server, "peak_gpu_memory_measured": False,
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
        # Provider URL/query or secret values never enter the public result.
        reason = str(exc)
        code = reason if re.fullmatch(r"(?:openwam|local_policy)_[a-z0-9_]{1,120}", reason) else "openwam_provisioning_failed:" + type(exc).__name__
        result = {"status": "blocked", "blockers": [code],
                  "candidate_policy_queried": False}
    (output / "openwam_piper_provisioning.v1.json").write_text(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
