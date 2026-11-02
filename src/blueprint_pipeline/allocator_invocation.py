"""Durable local allocator jobs in the existing attempt queue (AUT-08).

One flock spans the canonical allocator invocation. Before any provider-create
boundary the allocator fsyncs a source-bound create intention. A reconciler may
fence an invocation only while holding that same lock and while no create intent
exists. A delayed launcher then sees the fence and cannot create. Unknown
provider outcomes remain reconciliation-only; no inventory-zero inference is made.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextlib import nullcontext
import fcntl
import json
import os
import stat
from pathlib import Path
import subprocess
import sys
from typing import Any

from .common import utc_now_iso
from .core.handoff_lease import handoff_lease_checkpoint
from .decision_evidence_contracts import canonical_digest

SCHEMA = "blueprint.allocator_invocation.v1"
ENV = "BLUEPRINT_ALLOCATOR_INVOCATION_MANIFEST"
LOCK_ENV = "BLUEPRINT_ALLOCATOR_INVOCATION_LOCK_FD"


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("allocator_invocation_path_unsafe")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("allocator_invocation_record_invalid")
    return value


def _once(path: Path, value: Mapping[str, Any]) -> None:
    from .beta_policy_screening_launch import _write
    try:
        _write(path, value)
    except FileExistsError:
        if _read(path) != value:
            raise ValueError("allocator_invocation_record_conflict") from None


def manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    value = _read(path)
    file_status, directory_status = path.stat(), path.parent.stat()
    if (not path.is_absolute() or any(p.is_symlink() for p in path.parents)
            or file_status.st_uid != os.geteuid() or file_status.st_mode & 0o022
            or directory_status.st_uid != os.geteuid() or directory_status.st_mode & 0o077
            or not stat.S_ISREG(file_status.st_mode)
            or value.get("schema_version") != SCHEMA
            or value.get("manifest_digest") != canonical_digest(value, digest_field="manifest_digest")
            or value.get("invocation_id") != canonical_digest(value["binding"])
            or value.get("root") != str(path.parent)
            or value.get("result_path") != str(path.parent.parent / "allocator_result.json")
            or not isinstance(value.get("argv"), list) or not value["argv"]
            or any(not isinstance(arg, str) or "\x00" in arg for arg in value["argv"])):
        raise ValueError("allocator_invocation_manifest_invalid")
    return value


def retain_invocation(*, attempt_dir: Path, binding: Mapping[str, Any],
                      argv: Sequence[str], control: Mapping[str, Any] | None = None) -> Path:
    root = Path(attempt_dir).resolve() / "invocation"
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink():
        raise ValueError("allocator_invocation_path_unsafe")
    value = {"schema_version": SCHEMA, "root": str(root), "binding": dict(binding),
             "invocation_id": canonical_digest(binding), "argv": list(argv),
             "result_path": str(root.parent / "allocator_result.json"),
             "control": dict(control) if control is not None else None, "manifest_digest": ""}
    value["manifest_digest"] = canonical_digest(value, digest_field="manifest_digest")
    path = root / "manifest.json"
    if path.exists():
        old = manifest(path)
        old_control, proposed = dict(old.get("control") or {}), dict(value.get("control") or {})
        for key in ("initial_observed_at_epoch", "execution_deadline_epoch"):
            old_control.pop(key, None)
            proposed.pop(key, None)
        if old["binding"] != value["binding"] or old["argv"] != value["argv"] or old_control != proposed:
            raise ValueError("allocator_invocation_record_conflict")
        return path  # Immutable retained timing can only be shortened by monitoring.
    _once(path, value)
    manifest(path)
    return path


@contextmanager
def _claim(value: Mapping[str, Any]) -> Iterator[int]:
    path = Path(value["root"]) / "invocation.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        os.close(fd)


def start_invocation(path: Path, *, log_path: Path) -> int:
    """A duplicate spawn is harmless: only the queue job's lock holder can run."""
    handoff_lease_checkpoint("allocator_invocation_spawn")
    manifest(path)
    with log_path.open("x") as stream:
        log_path.chmod(0o600)
        process = subprocess.Popen(
            [sys.executable, "-m", "blueprint_pipeline.allocator_invocation", "--manifest", str(path)],
            cwd=Path(__file__).resolve().parents[2], stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "BLUEPRINT_VAST_IGNORE_LOCAL_SIGTERM_DURING_PROVIDER_RUN": "false"},
        )
    return int(process.pid)


def fence_unstarted(path: Path) -> dict[str, Any]:
    """Durably prevent an unspent invocation from ever reaching provider create."""
    value = manifest(path)
    root = Path(value["root"])
    try:
        with _claim(value):
            if (root / "provider_create_intent.json").exists():
                return {"status": "reconciliation_required", "next_action": "reconcile_exact_provider_identity",
                        "invocation_id": value["invocation_id"]}
            proof_path = root / "no_create_fence.json"
            proof = {"schema_version": "blueprint.allocator_no_create_fence.v1",
                     "invocation_id": value["invocation_id"], "manifest_digest": value["manifest_digest"],
                     "binding": value["binding"], "provider_create_prevented": True,
                     "no_policy_api_calls_proven": True}
            _once(proof_path, proof)
            return {"status": "fenced_unspent", "proof_path": str(proof_path), **proof}
    except BlockingIOError:
        return {"status": "running", "invocation_id": value["invocation_id"]}


def before_provider_create(*, authority: Mapping[str, Any], bundle: Mapping[str, Any]) -> None:
    """A conservative create intent, retained before the adapter can mutate."""
    handoff_lease_checkpoint("canonical_provider_create")
    configured = os.getenv(ENV)
    if not configured:
        return  # Existing standalone allocator callers keep their own admission.
    value = manifest(Path(configured))
    root = Path(value["root"])
    binding = value["binding"]
    fd = int(os.environ.get(LOCK_ENV, "-1"))
    held, named = os.fstat(fd), (root / "invocation.lock").stat()
    started = _read(root / "started.json")
    if ((held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
            or started.get("pid") != os.getpid()
            or started.get("manifest_digest") != value["manifest_digest"]
            or (root / "no_create_fence.json").exists()
            or binding.get("authority_digest") != authority.get("authority_digest")
            or binding.get("resource_name") != authority.get("resource_name")
            or binding.get("bundle_sha256") != bundle.get("bundle_sha256")):
        raise ValueError("allocator_invocation_create_binding_invalid")
    if value.get("control") is not None:
        from .beta_policy_screening_control import observe
        control_state = observe(control=value["control"], root=root)
        if control_state.get("new_processing_authorized") is False or control_state.get("status") != "current":
            raise ValueError("beta_screening_current_source_unproven_before_create")
    from .beta_policy_screening_launch import _write
    _write(root / "provider_create_intent.json", {
        "schema_version": "blueprint.allocator_provider_create_intent.v1",
        "invocation_id": value["invocation_id"], "manifest_digest": value["manifest_digest"],
        "binding": binding, "provider": binding["provider"], "resource_name": binding["resource_name"],
        "create_outcome": "unknown_until_exact_provider_receipt",
    })


def run_invocation(path: Path, *, allocator: Callable[[Sequence[str]], int] | None = None) -> int:
    value = manifest(path)
    root = Path(value["root"])
    try:
        with _claim(value) as fd:
            if ((root / "no_create_fence.json").exists() or (root / "provider_create_intent.json").exists()
                    or Path(value["result_path"]).exists()):
                return 2
            from .common import write_json
            from .team_candidate_quick10_launch import _process_start_ticks
            write_json(root / "started.json", {"pid": os.getpid(), "start_ticks": _process_start_ticks(os.getpid()),
                "manifest_digest": value["manifest_digest"],
                "invocation_id": value["invocation_id"], "started_at": utc_now_iso()})
            previous = {key: os.environ.get(key) for key in (ENV, LOCK_ENV)}
            os.environ.update({ENV: str(path), LOCK_ENV: str(fd)})
            try:
                if allocator is None:
                    from .paid_resource_allocator import main as allocator
                from .beta_policy_screening_control import CurrentSourceMonitor
                monitor = CurrentSourceMonitor(value) if value.get("control") is not None else nullcontext()
                with monitor:
                    return allocator(value["argv"])
            finally:
                for key, old in previous.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old
    except BlockingIOError:
        return 2


def stop_invocation(path: Path) -> None:
    """Signal only this invocation's verified allocator, preserving its finally."""
    import signal
    from .team_candidate_quick10_launch import _process_start_ticks
    value = manifest(path)
    started = _read(Path(value["root"]) / "started.json")
    if (started.get("manifest_digest") != value["manifest_digest"] or not started.get("start_ticks")
            or _process_start_ticks(started["pid"]) != started["start_ticks"]):
        raise ValueError("beta_screening_allocator_stop_identity_unproven")
    os.kill(int(started["pid"]), signal.SIGTERM)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    return run_invocation(parser.parse_args(argv).manifest)


if __name__ == "__main__":
    raise SystemExit(main())
