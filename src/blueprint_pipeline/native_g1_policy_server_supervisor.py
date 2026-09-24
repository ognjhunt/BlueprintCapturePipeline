"""Own one pinned HumanoidArena policy server for a G1 scene episode.

The server is a child process on loopback. This module checks source, staged
checkpoint files, listener ownership, and the publisher's reset handshake.
It does not claim an inference, task outcome, model-license admission, or a
physical result. The caller must close the returned lease in a ``finally``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_humanoidarena_policy_client import NativeG1HumanoidArenaPolicyClient
from .native_g1_run_preflight import preflight_g1_shared_scene_run


PINNED_SOURCE_REVISION = "68479287a784a69be9ce6ad739311d2f11f75ef9"
LOOPBACK_HOST = "127.0.0.1"


def _source_revision(source: Path) -> str:
    root = source.resolve().parents[1]
    if source.parent.name != "scripts" or not (root / "src").is_dir():
        raise ValueError("g1_server_source_tree_invalid")
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("g1_server_source_revision_unavailable") from exc
    if revision != PINNED_SOURCE_REVISION or dirty:
        raise ValueError("g1_server_source_revision_or_tree_mismatch")
    return revision


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _candidate_policy_dir(inventory_path: Path, candidate_id: str, checkpoint_root: Path) -> Path:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if inventory.get("source_revision") != PINNED_SOURCE_REVISION:
        raise ValueError("g1_server_inventory_source_revision_mismatch")
    candidates = inventory.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("g1_server_inventory_invalid")
    matches = [row for row in candidates if isinstance(row, dict) and row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise ValueError("g1_server_candidate_invalid")
    relative = PurePosixPath(str(matches[0].get("subdirectory") or ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("g1_server_candidate_directory_invalid")
    folder = checkpoint_root.joinpath(*relative.parts)
    config = folder / "config.json"
    if config.is_symlink() or not config.is_file():
        raise ValueError("g1_server_policy_config_missing")
    settings = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("g1_server_policy_config_invalid")
    base = settings.get("pretrained_path")
    if isinstance(base, str) and Path(base).is_absolute():
        # The released pi05 configs reference a publisher-local base model.
        # Its bytes and loader mapping need a separate exact admission.
        raise ValueError("g1_server_external_base_model_unverified")
    return folder


def _linux_listener_owner_pids(port: int, *, proc_root: Path = Path("/proc")) -> set[int]:
    inodes: set[str] = set()
    tables_read = 0
    for name in ("net/tcp", "net/tcp6"):
        try:
            lines = (proc_root / name).read_text(encoding="utf-8").splitlines()[1:]
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise ValueError("g1_server_listener_table_unavailable") from exc
        tables_read += 1
        for line in lines:
            fields = line.split()
            if (
                len(fields) >= 10 and fields[3] == "0A"
                and fields[1].rsplit(":", 1)[-1].upper() == f"{port:04X}"
            ):
                inodes.add(fields[9])
    if not tables_read:
        raise ValueError("g1_server_listener_table_unavailable")
    if not inodes:
        return set()
    owners: set[int] = set()
    expected_links = {f"socket:[{inode}]" for inode in inodes}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            for descriptor in (entry / "fd").iterdir():
                if os.readlink(descriptor) in expected_links:
                    owners.add(int(entry.name))
                    break
        except (FileNotFoundError, PermissionError, NotADirectoryError, ProcessLookupError):
            continue
    if not owners:
        raise ValueError("g1_server_listener_owner_unresolved")
    return owners


def listener_owner_pids(port: int) -> set[int]:
    if sys.platform.startswith("linux"):
        return _linux_listener_owner_pids(port)
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
                check=False, capture_output=True, text=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError("g1_server_listener_owner_unavailable") from exc
        if result.returncode == 1 and not result.stdout.strip():
            return set()
        if result.returncode != 0:
            raise ValueError("g1_server_listener_owner_unavailable")
        try:
            return {int(line) for line in result.stdout.splitlines() if line.strip()}
        except ValueError as exc:
            raise ValueError("g1_server_listener_owner_invalid") from exc
    raise ValueError("g1_server_listener_owner_platform_unsupported")


class NativeG1PolicyServerLease:
    def __init__(
        self, *, process: subprocess.Popen[Any], client: Any,
        receipt: dict[str, Any], log_path: Path,
    ) -> None:
        self.process = process
        self.client = client
        self.receipt = receipt
        self.log_path = log_path
        self.closed = False

    def close(self) -> dict[str, Any]:
        if self.closed:
            return {"status": "already_closed", "pid": self.process.pid}
        self.closed = True
        if self.process.poll() is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if self.process.poll() is None:
                    self.process.kill()
                self.process.wait(timeout=10)
        return {
            "status": "child_exited" if self.process.poll() is not None else "teardown_unverified",
            "pid": self.process.pid,
            "exit_code": self.process.poll(),
        }


def start_g1_policy_server(
    *,
    preflight_inputs: Mapping[str, Any],
    python_executable: Path,
    port: int,
    device: str,
    log_path: Path,
    startup_timeout_seconds: float = 900.0,
    owner_reader: Callable[[int], set[int]] = listener_owner_pids,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    client_factory: Callable[..., NativeG1HumanoidArenaPolicyClient] = NativeG1HumanoidArenaPolicyClient,
) -> NativeG1PolicyServerLease:
    """Launch exact staged bytes, wait for its own listener, and reset once."""

    if (
        isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
        or not isinstance(device, str) or not device.startswith("cuda:")
        or not device[5:].isdigit()
        or isinstance(startup_timeout_seconds, bool)
        or not isinstance(startup_timeout_seconds, (int, float))
        or not 1 <= startup_timeout_seconds <= 1800
        or log_path.exists() or log_path.is_symlink()
    ):
        raise ValueError("g1_server_launch_configuration_invalid")
    preflight = preflight_g1_shared_scene_run(**preflight_inputs)
    if preflight.get("status") != "staged_inputs_verified":
        raise ValueError("g1_server_preflight_incomplete")
    source = Path(preflight_inputs["policy_server_source"])
    revision = _source_revision(source)
    policy_dir = _candidate_policy_dir(
        Path(preflight_inputs["inventory_path"]),
        str(preflight["candidate_id"]),
        Path(preflight_inputs["checkpoint_root"]),
    )
    python = python_executable.expanduser().resolve(strict=True)
    if not python.is_file():
        raise ValueError("g1_server_python_unavailable")
    python_digest = _sha256_file(python)
    argv = [
        str(python), str(source.resolve()), "--policy-path", str(policy_dir.resolve()),
        "--device", device, "--host", LOOPBACK_HOST, "--port", str(port),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = None
    with log_path.open("x", encoding="utf-8") as log_stream:
        process = popen_factory(
            argv, stdin=subprocess.DEVNULL, stdout=log_stream,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    try:
        client = client_factory(base_url=f"http://{LOOPBACK_HOST}:{port}")
        deadline = time.monotonic() + startup_timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("g1_server_exited_before_ready")
            owners = owner_reader(port)
            if owners and owners != {process.pid}:
                raise RuntimeError("g1_server_listener_owner_mismatch")
            if owners == {process.pid}:
                try:
                    client.reset(seed=0)
                except (OSError, TimeoutError, ConnectionError):
                    pass
                else:
                    if process.poll() is None and owner_reader(port) == {process.pid}:
                        receipt = {
                            "schema_version": "native_g1_policy_server_lease.v1",
                            "status": "server_ready_process_bound",
                            "pid": process.pid,
                            "source_revision": revision,
                            "scene_plan_digest": preflight["scene_plan_digest"],
                            "candidate_id": preflight["candidate_id"],
                            "candidate_inventory_digest": preflight["candidate_inventory_digest"],
                            "preflight_receipt_digest": canonical_digest(preflight),
                            "launch_argv": argv,
                            "python_executable_sha256": python_digest,
                            "reset_ack_verified": True,
                            "listener_owner_verified": True,
                            "checkpoint_bytes_verified": True,
                            "loaded_checkpoint_identity_observed": False,
                            "inference_observed": False,
                            "task_outcome_observed": False,
                        }
                        receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
                        return NativeG1PolicyServerLease(
                            process=process, client=client, receipt=receipt, log_path=log_path
                        )
            time.sleep(0.25)
        raise TimeoutError("g1_server_startup_timeout")
    except BaseException:
        NativeG1PolicyServerLease(
            process=process, client=None,
            receipt={}, log_path=log_path,
        ).close()
        raise
