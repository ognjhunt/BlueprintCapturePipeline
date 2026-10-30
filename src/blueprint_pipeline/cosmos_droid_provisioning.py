"""Pinned offline Cosmos setup inside an already admitted native GPU worker."""
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

from .cosmos_droid_bundle import CHECKPOINT_BYTES, profile, runtime_contract
from .cosmos_droid_dependencies import DEPENDENCY_MANIFEST_FILENAME, verified_dependency_manifest
from .cosmos_droid_source import SOURCE_ARCHIVE, SOURCE_MANIFEST, extract_source
from .local_policy_provisioning import (
    DEADLINE_ENV, TransferBudget, fetch_exact, remaining_seconds, _sha256,
    _command, _extract_interpreter, _install_wheels,
)


def client_wheels(dependencies: dict, python_minor: str) -> list[dict]:
    if python_minor not in dependencies["supported_thin_client_python_minors"]:
        raise ValueError("cosmos_client_interpreter_unsupported")
    rows = [row for row in dependencies["thin_client_wheels"]
            if row.get("python_minor", python_minor) == python_minor]
    if len(rows) != 4 or len({row["package"] for row in rows}) != 4:
        raise ValueError("cosmos_client_dependency_closure_invalid")
    return rows


def server_environment(runtime: Path, root: Path) -> dict:
    env = {key: value for key, value in os.environ.items()
           if key not in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "PYTHONHOME", "VIRTUAL_ENV"}}
    if "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = ":".join(part for part in env["LD_LIBRARY_PATH"].split(":") if "/isaac-sim" not in part)
    env.update(PYTHONPATH=str(runtime), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               COSMOS_TRAINING="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
               HF_HOME=str(root / "hub-cache"), HF_MODULES_CACHE=str(root / "module-cache"))
    return env


def _start_server(*, python: Path, runtime: Path, output: Path, root: Path, spec: dict, deadline: float) -> dict:
    selected = profile(spec["candidate_id"])
    command = [str(python), "-m", "blueprint_pipeline.cosmos_edge_droid_policy_server",
        "--candidate-id", spec["candidate_id"], "--checkpoint-path", str(root / "checkpoint"),
        "--snapshot-manifest", str(runtime / selected.SNAPSHOT_STAGED_NAME),
        "--source-root", str(root / "source"), "--source-manifest", str(runtime / SOURCE_MANIFEST),
        "--host", "127.0.0.1", "--port", "8000", "--output-dir", str(output / "cosmos_policy")]
    log = output / f"adp009d_policy_server.{spec['candidate_id']}.log"
    receipt_path = output / f"adp009d_policy_server_receipt.{spec['candidate_id']}.json"
    receipt = {"schema_version": "adp009d_policy_server_receipt.v1", "candidate_id": spec["candidate_id"],
        "status": "starting", "command": command, "candidate_policy_queried": False,
        "candidate_inference_performed": False, "checkpoint_bytes_verified": True,
        "provisioning_identity": runtime_contract(spec["candidate_id"])}
    process = None
    try:
        with log.open("wb") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=server_environment(runtime, root))
        receipt["server_pid"] = process.pid
        receipt_path.write_text(json.dumps(receipt, sort_keys=True))
        ready_deadline = min(deadline, time.time() + 900)
        client_type = getattr(selected, "CosmosNanoSimulatorClient", selected.CosmosEdgeSimulatorClient)
        while True:
            if process.poll() is not None:
                raise ValueError("cosmos_server_exited_before_ready")
            remaining_seconds(ready_deadline)
            client = client_type(policy_spec=spec["policy_spec"], host="127.0.0.1", port=8000,
                                 startup_timeout_seconds=min(10, remaining_seconds(ready_deadline)))
            try:
                readiness = client.preflight_readiness()
                receipt.update(status="ready", handshake_completed=True, exact_policy_identity_verified=True,
                               readiness=readiness)
                break
            except (OSError, TimeoutError):
                time.sleep(min(1, remaining_seconds(ready_deadline)))
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
        receipt.update(status="blocked", blocker="cosmos_server_start_failed",
                       process_terminated=process is None or process.poll() is not None)
        receipt_path.write_text(json.dumps(receipt, sort_keys=True))
        raise


def provision(runtime: Path, output: Path, spec: dict, *, root: Path = Path("/opt/blueprint-cosmos-droid")) -> dict:
    selected = profile(spec["candidate_id"])
    selected.validate_execution_spec(spec)
    deadline = float(os.environ.get(DEADLINE_ENV, "nan"))
    remaining_seconds(deadline)
    dependencies = verified_dependency_manifest(runtime / DEPENDENCY_MANIFEST_FILENAME)
    manifest = selected.verified_snapshot_manifest(runtime / selected.SNAPSHOT_STAGED_NAME)
    if sum(row["size_bytes"] for row in manifest["required_files"]) != CHECKPOINT_BYTES[spec["candidate_id"]]:
        raise ValueError("cosmos_checkpoint_download_bound_invalid")
    budget = TransferBudget(dependencies["total_download_bytes"] + CHECKPOINT_BYTES[spec["candidate_id"]])
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    source = extract_source(runtime / SOURCE_ARCHIVE, runtime / SOURCE_MANIFEST, root / "source")
    interpreter = dependencies["interpreter"]
    archive = root / interpreter["filename"]
    downloads = [fetch_exact(interpreter, archive, deadline=deadline, budget=budget)]
    _extract_interpreter(archive, root / "interpreter", maximum_bytes=interpreter["maximum_expanded_bytes"])
    python = root / "interpreter" / interpreter["python_relative_path"]
    _command([str(python), "-c", "import sys;assert sys.version.split()[0]==" + repr(interpreter["version"])], deadline=deadline)
    wheels = root / "wheels"
    wheels.mkdir(exist_ok=True)
    thin = client_wheels(dependencies, f"{sys.version_info.major}.{sys.version_info.minor}")
    for row in {row["filename"]: row for row in [*dependencies["server_wheels"], *thin]}.values():
        if row.get("bundled_path"):
            path = runtime / row["bundled_path"]
            if (path.is_symlink() or path.stat().st_size != row["size_bytes"]
                    or _sha256(path, deadline=deadline) != "sha256:" + row["sha256"].removeprefix("sha256:")):
                raise ValueError("cosmos_bundled_dependency_changed")
            shutil.copyfile(path, wheels / row["filename"])
        else:
            downloads.append(fetch_exact(row, wheels / row["filename"], deadline=deadline, budget=budget))
    _install_wheels(python, dependencies["server_wheels"], wheels, deadline=deadline)
    _install_wheels(Path(sys.executable), thin, wheels, deadline=deadline, target=runtime / "cosmos_policy_client")
    client_archive = runtime / "cosmos_policy_client_wheels"
    client_archive.mkdir(exist_ok=True)
    for row in thin:
        shutil.copyfile(wheels / row["filename"], client_archive / row["filename"])
    for row in manifest["required_files"]:
        download = {**row, "url": f"https://huggingface.co/{selected.MODEL_ID}/resolve/{selected.MODEL_REVISION}/{quote(row['path'])}?download=true"}
        downloads.append(fetch_exact(download, root / "checkpoint" / row["path"], deadline=deadline, budget=budget))
    # The server repeats full-byte verification immediately before model load.
    expected = {row["path"] for row in manifest["required_files"]}
    actual = {path.relative_to(root / "checkpoint").as_posix() for path in (root / "checkpoint").rglob("*") if not path.is_dir()}
    if actual != expected:
        raise ValueError("cosmos_checkpoint_file_closure_changed")
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
        reason = str(exc)
        code = reason if re.fullmatch(r"(?:cosmos|local_policy)_[a-z0-9_]{1,120}", reason) else "cosmos_provisioning_failed:" + type(exc).__name__
        result = {"status": "blocked", "blockers": [code], "candidate_policy_queried": False}
    (output / "cosmos_droid_provisioning.v1.json").write_text(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
