"""Run the existing G1 development worker in the pinned Isaac container.

This is transport and provisioning for the same sealed scene episode used by
the local worker. Planning is offline; execution requires a Linux NVIDIA Docker
host with the pinned image, released runtime source packet, policy assets, and
the candidate/scene-specific human rights review in the worker request. The
official policy server has a separate LeRobot environment from Isaac Lab; its
interpreter is mounted and checked before simulator provisioning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_development_worker import _request
from .native_g1_policy_server_supervisor import (
    PINNED_SOURCE_REVISION,
    _source_revision,
)
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


SCHEMA = "native_g1_container_run_plan.v1"
OUTPUT_ROOT = Path("/blueprint-g1-output")
RUNTIME_RECEIPT = OUTPUT_ROOT / "runtime/native_task_runtime_source_provisioning.v1.json"
CONTAINER_REQUEST = OUTPUT_ROOT / "work/native_g1_development_episode_request.v1.json"
CONTAINER_RESULT_DIR = OUTPUT_ROOT / "results/episode"
SOURCE_ROOT = Path("/blueprint-src")
FILE_INPUTS = (
    "inventory_path", "sonic_encoder", "sonic_decoder",
)
DIR_INPUTS = ("bundle_root", "checkpoint_root")


def _host_path(value: str | Path, *, kind: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or any(char in str(path) for char in ",\n\r"):
        raise ValueError("g1_container_input_path_invalid")
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("g1_container_input_symlink_forbidden")
    if kind == "file" and not path.is_file():
        raise ValueError("g1_container_input_file_missing")
    if kind == "dir" and not path.is_dir():
        raise ValueError("g1_container_input_directory_missing")
    return path


def _checkout_root(source: str, *, expected_relative: str) -> Path:
    path = _host_path(source, kind="file")
    try:
        result = subprocess.run(
            ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        root = Path(result.stdout.strip()).resolve(strict=True)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("g1_container_policy_checkout_invalid") from exc
    if (
        not (root / ".git").is_dir()
        or not path.is_relative_to(root)
        or path.relative_to(root).as_posix() != expected_relative
    ):
        raise ValueError("g1_container_policy_checkout_invalid")
    return root


def _policy_python(value: str, *, runtime_root: Path) -> Path:
    path = Path(value).expanduser()
    if (
        not path.is_absolute() or not path.is_file()
        or not os.access(path, os.X_OK)
        or not path.is_relative_to(runtime_root)
        or not path.resolve().is_relative_to(runtime_root)
        or any(char in str(path) for char in ",\n\r")
    ):
        raise ValueError("g1_container_policy_python_invalid")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _read_only_mounts(paths: Sequence[Path]) -> list[Path]:
    # A parent directory mount exposes every child at the same absolute path.
    # Avoid Docker rejecting overlapping file and directory mount targets.
    selected: list[Path] = []
    for path in sorted(set(paths), key=lambda item: (len(item.parts), str(item))):
        if any(parent.is_dir() and path.is_relative_to(parent) for parent in selected):
            continue
        selected.append(path)
    return selected


def prepare_g1_container_run(
    *,
    request_path: Path,
    source_receipt_path: Path,
    source_packet_path: Path,
    policy_runtime_root: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Stage one exact Docker invocation without starting a provider or GPU."""

    request_file = _host_path(request_path, kind="file")
    source_receipt = _host_path(source_receipt_path, kind="file")
    source_packet = _host_path(source_packet_path, kind="file")
    source = _host_path(repo_root / "src", kind="dir")
    if source.parent != repo_root or not (source / "blueprint_pipeline").is_dir():
        raise ValueError("g1_container_blueprint_source_invalid")
    request = _request(json.loads(request_file.read_text(encoding="utf-8")))
    policy_runtime = _host_path(policy_runtime_root, kind="dir")
    policy_python = _policy_python(request["python_executable"], runtime_root=policy_runtime)
    policy_checkout = _checkout_root(
        request["policy_server_source"],
        expected_relative="lerobot/scripts/serve_lerobot_vla_http.py",
    )
    sonic_checkout = _checkout_root(
        request["sonic_provider_source"],
        expected_relative="isaaclab_twist2_g1/action_provider/action_provider_sonic.py",
    )
    if policy_checkout != sonic_checkout:
        raise ValueError("g1_container_policy_checkout_mismatch")
    _source_revision(Path(request["policy_server_source"]))
    _source_revision(Path(request["sonic_provider_source"]), expected_parent="action_provider")
    lerobot_source = policy_checkout / "lerobot/src/lerobot"
    if not lerobot_source.is_dir():
        raise ValueError("g1_container_lerobot_source_missing")
    input_paths = [request_file, source_receipt, source_packet, source]
    input_paths += [_host_path(request[field], kind="file") for field in FILE_INPUTS]
    input_paths += [_host_path(request[field], kind="dir") for field in DIR_INPUTS]
    input_paths += [policy_checkout, policy_runtime]
    output = Path(output_dir).expanduser()
    if (
        not output.is_absolute() or output.exists() or output.is_symlink()
        or output.resolve() != output
        or any(char in str(output) for char in ",\n\r")
        or any(output.is_relative_to(path) or path.is_relative_to(output) for path in input_paths)
    ):
        raise ValueError("g1_container_output_directory_invalid")
    output.mkdir(parents=True)
    (output / "runtime").mkdir()
    (output / "work").mkdir()
    (output / "results").mkdir()

    container_request = dict(request)
    container_request["runtime_provisioning_receipt_path"] = str(RUNTIME_RECEIPT)
    container_request["request_digest"] = canonical_digest(
        container_request, digest_field="request_digest"
    )
    (output / "work" / CONTAINER_REQUEST.name).write_text(
        json.dumps(container_request, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    probe = (
        "import pathlib,torch,lerobot;"
        "from lerobot.configs.policies import PreTrainedConfig;"
        "from lerobot.policies.factory import get_policy_class;"
        f"expected=pathlib.Path({str(lerobot_source)!r}).resolve();"
        "actual=pathlib.Path(lerobot.__file__).resolve()\n"
        "if not actual.is_relative_to(expected): raise SystemExit('g1_policy_import_origin_mismatch')\n"
        "if not torch.cuda.is_available(): raise SystemExit('g1_policy_cuda_unavailable')\n"
        "if not torch.empty(1,device='cuda:0').is_cuda: raise SystemExit('g1_policy_cuda_probe_failed')"
    )
    command = [
        "docker", "run", "--rm", "--pull", "never", "--gpus", "device=0",
        "--network", "none", "--shm-size", "8g",
        "--env", "ACCEPT_EULA=Y", "--env", "PRIVACY_CONSENT=Y",
        "--env", "GIT_OPTIONAL_LOCKS=0",
        "--env", "NO_PROXY=127.0.0.1,localhost",
        "--env", f"PYTHONPATH={SOURCE_ROOT}:{policy_checkout / 'lerobot/src'}",
        "--workdir", str(OUTPUT_ROOT),
        "--mount", f"type=bind,src={source},dst={SOURCE_ROOT},readonly",
    ]
    for path in _read_only_mounts(input_paths):
        if path == source:
            continue
        command.extend(("--mount", f"type=bind,src={path},dst={path},readonly"))
    command.extend((
        "--mount", f"type=bind,src={output},dst={OUTPUT_ROOT}",
        "--entrypoint", "/bin/bash", NATIVE_TASK_ARENA_IMAGE,
        "-euo", "pipefail", "-c",
        " ".join((
            f"{shlex.quote(str(policy_python))} -c {shlex.quote(probe)}",
            "&&",
            "/isaac-sim/python.sh -m blueprint_pipeline.native_task_runtime_source_provision",
            f"--source-receipt {shlex.quote(str(source_receipt))}",
            f"--source-packet {shlex.quote(str(source_packet))}",
            f"--extraction-dir {shlex.quote(str(OUTPUT_ROOT / 'runtime/sources'))}",
            f"--output {shlex.quote(str(RUNTIME_RECEIPT))}",
            "&& /isaac-sim/python.sh -m blueprint_pipeline.native_g1_development_worker",
            f"--request {shlex.quote(str(CONTAINER_REQUEST))}",
            f"--output-dir {shlex.quote(str(CONTAINER_RESULT_DIR))}",
        )),
    ))
    plan = {
        "schema_version": SCHEMA,
        "status": "staged_not_executed",
        "image": NATIVE_TASK_ARENA_IMAGE,
        "source_request_sha256": _file_sha256(request_file),
        "container_request_digest": container_request["request_digest"],
        "source_packet_sha256": _file_sha256(source_packet),
        "source_receipt_sha256": _file_sha256(source_receipt),
        "policy_source_revision": PINNED_SOURCE_REVISION,
        "policy_python_sha256": _file_sha256(policy_python),
        "policy_runtime_root": str(policy_runtime),
        "container_request_path": str(output / "work" / CONTAINER_REQUEST.name),
        "runtime_receipt_path": str(output / "runtime" / RUNTIME_RECEIPT.name),
        "worker_result_path": str(output / "results/episode/native_g1_development_worker_result.v1.json"),
        "command": command,
        "ranking_eligible": False,
        "physical_outcome_claimed": False,
    }
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    (output / (SCHEMA + ".json")).write_text(
        json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-packet", type=Path, required=True)
    parser.add_argument("--policy-runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and sys.platform != "linux":
        parser.error("G1 container execution requires a Linux NVIDIA Docker host")
    plan = prepare_g1_container_run(
        request_path=args.request,
        source_receipt_path=args.source_receipt,
        source_packet_path=args.source_packet,
        policy_runtime_root=args.policy_runtime_root,
        output_dir=args.output_dir,
        repo_root=Path(__file__).resolve().parents[2],
    )
    if not args.execute:
        print(json.dumps({"status": plan["status"], "plan_path": str(args.output_dir / (SCHEMA + ".json"))}))
        return 0
    with (args.output_dir / "container.log").open("w", encoding="utf-8") as stream:
        try:
            completed = subprocess.run(plan["command"], stdout=stream, stderr=subprocess.STDOUT, check=False)
            return completed.returncode
        except OSError as exc:
            stream.write(f"container_launch_failed:{type(exc).__name__}:{exc}\n")
            return 2


if __name__ == "__main__":
    sys.exit(main())
