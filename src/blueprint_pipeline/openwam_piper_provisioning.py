"""Bounded local OpenWAM bootstrap inside an already admitted canonical worker.

Every network input has a fixed release URL, byte bound and SHA256. Installation
and model loading use only these staged files. No provider is created here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
from urllib.parse import quote

from .openwam_piper_assets import EXPECTED_FILES, verify_checkpoint_directory
from .openwam_piper_policy import MODEL_ID, MODEL_REVISION, OpenWAMPiperPolicyClient
from .openwam_piper_runtime import policy_binding, serialized_binding, validate_execution_spec
from .openwam_piper_source import SOURCE_ARCHIVE, SOURCE_MANIFEST, extract_source

DEADLINE_ENV = "BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH"
ALLOWED_ORIGINS = (
    "https://huggingface.co", "https://cdn-lfs.huggingface.co", "https://cdn-lfs-us-1.hf.co",
    "https://cdn-lfs.hf.co", "https://cas-bridge.xethub.hf.co", "https://us.aws.cdn.hf.co",
    "https://github.com", "https://release-assets.githubusercontent.com", "https://objects.githubusercontent.com",
    "https://files.pythonhosted.org", "https://download.pytorch.org", "https://download-r2.pytorch.org",
)


def remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.time()
    if not math.isfinite(remaining) or remaining <= 0:
        raise ValueError("openwam_provisioning_deadline_exceeded")
    return remaining


def _sha256(path: Path, *, deadline: float | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            if deadline is not None:
                remaining_seconds(deadline)
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclass
class TransferBudget:
    maximum_bytes: int
    attempted_bytes: int = 0

    def reserve(self, size: int) -> None:
        if type(size) is not int or size <= 0 or self.attempted_bytes + size > self.maximum_bytes:
            raise ValueError("openwam_total_download_bound_exceeded")
        self.attempted_bytes += size


def fetch_exact(row: dict, destination: Path, *, deadline: float, fetch=None,
                budget: TransferBudget | None = None) -> dict:
    """No retry, ambient credentials, arbitrary output tree or unbounded reads."""
    from .core.security_controls import fetch_bounded_https

    size = row["size_bytes"]
    expected = row["sha256"]
    if isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected):
        expected = "sha256:" + expected
    if type(size) is not int or size < 1 or not isinstance(expected, str) or len(expected) != 71:
        raise ValueError("openwam_download_identity_invalid")
    if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents):
        raise ValueError("openwam_download_path_invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size == size and _sha256(destination, deadline=deadline) == expected:
            return {"status": "verified", "size_bytes": size, "sha256": expected, "cache_hit": True}
        raise ValueError("openwam_download_existing_file_changed")
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("openwam_download_partial_exists")
    try:
        if budget is not None:
            budget.reserve(size)
        (fetch or fetch_bounded_https)(row["url"], allowed_origins=ALLOWED_ORIGINS,
            timeout_seconds=max(1, min(600, int(remaining_seconds(deadline)))), max_bytes=size,
            max_redirects=5, output_path=temporary)
        remaining_seconds(deadline)
        if temporary.stat().st_size != size or _sha256(temporary, deadline=deadline) != expected:
            raise ValueError("openwam_download_bytes_changed")
        os.replace(temporary, destination)
        return {"status": "verified", "size_bytes": size, "sha256": expected, "cache_hit": False}
    finally:
        temporary.unlink(missing_ok=True)


def _command(argv: list[str], *, deadline: float, env: dict | None = None) -> None:
    subprocess.run(argv, check=True, timeout=remaining_seconds(deadline), env=env)


def _extract_interpreter(path: Path, destination: Path, *, maximum_bytes: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        if sum(member.size for member in members) > maximum_bytes:
            raise ValueError("openwam_interpreter_expansion_exceeds_bound")
        # Python's data filter refuses outside links, devices and absolute paths.
        archive.extractall(destination, filter="data")


def _install_wheels(python: Path, rows: list[dict], wheels: Path, *, deadline: float,
                    target: Path | None = None) -> None:
    requirements = wheels / ("client-requirements.txt" if target else "server-requirements.txt")
    requirements.write_text("\n".join(str(wheels / row["filename"]) + " --hash=sha256:" + row["sha256"].removeprefix("sha256:")
                                      for row in rows) + "\n")
    argv = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-index", "--no-deps",
            "--require-hashes", "--no-compile", "-r", str(requirements)]
    if target is not None:
        argv += ["--target", str(target)]
    _command(argv, deadline=deadline)


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
        code = reason if re.fullmatch(r"openwam_[a-z0-9_]{1,120}", reason) else "openwam_provisioning_failed:" + type(exc).__name__
        result = {"status": "blocked", "blockers": [code],
                  "candidate_policy_queried": False}
    (output / "openwam_piper_provisioning.v1.json").write_text(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
