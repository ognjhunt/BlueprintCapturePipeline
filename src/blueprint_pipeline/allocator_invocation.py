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
    control = value.get("control") or {}
    scope = control.get("source_control_binding")
    if scope is not None:
        from .beta_policy_screening_control_contracts import validate_binding
        validate_binding(scope)
        if (scope["invocation_id"] != value["invocation_id"] or scope["source"] != control.get("source")
                or scope["site_id"] != control.get("site_id")
                or scope["prepared_digest"] != control.get("prepared_digest")
                or scope["attempt_id"] != path.parent.parent.name
                or any(scope[key] != value["binding"].get(key) for key in (
                    "prepared_digest", "plan_digest", "candidate_id", "configuration_digest",
                    "authority_digest", "resource_name", "attempt_id"))):
            raise ValueError("allocator_invocation_control_binding_invalid")
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


def invocation_temp_dir(path: Path) -> Path:
    """The detached allocator's own temporary directory, beside its manifest.

    The allocator outlives the timer pass that spawns it. A unit with PrivateTmp
    gives that pass a private /tmp and /var/tmp that systemd removes when the
    pass ends, so a detached child left with them finds no usable temporary
    directory (2026-11-04: every learned-policy launch was refused in the sealed
    static preflight before any allocation). The invocation directory is the
    allocator's own writable root, kept for the life of the attempt.
    """
    temp = Path(path).parent / "tmp"
    temp.mkdir(mode=0o700, exist_ok=True)
    if temp.is_symlink() or not temp.is_dir():
        raise ValueError("allocator_invocation_temp_dir_invalid")
    return temp


def start_invocation(path: Path, *, log_path: Path) -> int:
    """A duplicate spawn is harmless: only the queue job's lock holder can run."""
    handoff_lease_checkpoint("allocator_invocation_spawn")
    manifest(path)
    temp = str(invocation_temp_dir(path))
    with log_path.open("x") as stream:
        log_path.chmod(0o600)
        process = subprocess.Popen(
            [sys.executable, "-m", "blueprint_pipeline.allocator_invocation", "--manifest", str(path)],
            cwd=Path(__file__).resolve().parents[2], stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "BLUEPRINT_VAST_IGNORE_LOCAL_SIGTERM_DURING_PROVIDER_RUN": "false",
                 "TMPDIR": temp, "TEMP": temp, "TMP": temp},
        )
    return int(process.pid)


def _fence_proof(value: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema_version": "blueprint.allocator_no_create_fence.v1",
            "invocation_id": value["invocation_id"], "manifest_digest": value["manifest_digest"],
            "binding": value["binding"], "provider_create_prevented": True,
            "no_policy_api_calls_proven": True}


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
            proof = _fence_proof(value)
            _once(proof_path, proof)
            return {"status": "fenced_unspent", "proof_path": str(proof_path), **proof}
    except BlockingIOError:
        return {"status": "running", "invocation_id": value["invocation_id"]}


def _held_invocation() -> tuple[dict[str, Any], Path] | None:
    """This process's own running invocation, proven by the job lock it holds."""
    configured = os.getenv(ENV)
    if not configured:
        return None  # Existing standalone allocator callers keep their own admission.
    value = manifest(Path(configured))
    root = Path(value["root"])
    fd = int(os.environ.get(LOCK_ENV, "-1"))
    held, named = os.fstat(fd), (root / "invocation.lock").stat()
    started = _read(root / "started.json")
    if ((held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
            or started.get("pid") != os.getpid()
            or started.get("manifest_digest") != value["manifest_digest"]):
        raise ValueError("allocator_invocation_create_binding_invalid")
    return value, root


def fence_before_create() -> None:
    """The allocator itself refuses its create for a reason a retry can clear.

    Called only before any create intent, while this process holds the job
    lock: the same proof a reconciler would write for an unstarted job, so
    recovery retries the candidate on a fresh attempt instead of blocking it.
    """
    held = _held_invocation()
    if held is None:
        return
    value, root = held
    if not (root / "provider_create_intent.json").exists():
        _once(root / "no_create_fence.json", _fence_proof(value))


def before_provider_create(*, authority: Mapping[str, Any], bundle: Mapping[str, Any]) -> None:
    """A conservative create intent, retained before the adapter can mutate."""
    handoff_lease_checkpoint("canonical_provider_create")
    held = _held_invocation()
    if held is None:
        return  # Existing standalone allocator callers keep their own admission.
    value, root = held
    binding = value["binding"]
    if ((root / "no_create_fence.json").exists()
            or binding.get("authority_digest") != authority.get("authority_digest")
            or binding.get("resource_name") != authority.get("resource_name")
            or binding.get("bundle_sha256") != bundle.get("bundle_sha256")):
        raise ValueError("allocator_invocation_create_binding_invalid")
    from .beta_policy_screening_control import CONTROL_UNAVAILABLE_BEFORE_CREATE, current_control_admission
    from .beta_policy_screening_launch import _write
    admission = (current_control_admission(control=value["control"], root=root)
                 if value.get("control") is not None else nullcontext())
    try:
        with admission:
            _write(root / "provider_create_intent.json", {
                "schema_version": "blueprint.allocator_provider_create_intent.v1",
                "invocation_id": value["invocation_id"], "manifest_digest": value["manifest_digest"],
                "binding": binding, "provider": binding["provider"], "resource_name": binding["resource_name"],
                "create_outcome": "unknown_until_exact_provider_receipt",
            })
    except ValueError as exc:
        if str(exc) == CONTROL_UNAVAILABLE_BEFORE_CREATE:
            fence_before_create()  # No spend happened: recovery retries, it does not block.
        raise


@contextmanager
def _one_stop_signal(root: Path) -> Iterator[list[int]]:
    """The allocator honours exactly one stop signal.

    Before this job retains a provider create intent, the first SIGTERM
    unwinds it: nothing was created, and recovery fences the job. The create
    intent is written only inside the adapter's pre-create hook, where the
    adapter's own handler turns that signal into its teardown; so if this
    handler sees a SIGTERM while the intent exists, the adapter has already
    torn down. That signal, and every one after the first, is ignored, so the
    watchdog close and the result write always finish.
    """
    import signal
    received: list[int] = []

    def handler(signum: int, _frame: Any) -> None:
        first = not received
        received.append(signum)
        if first and not (root / "provider_create_intent.json").exists():
            raise KeyboardInterrupt(f"allocator_invocation_stop_signal:{signum}")

    try:
        previous = signal.signal(signal.SIGTERM, handler)
    except (ValueError, OSError, AttributeError):
        yield received  # Not the main thread: the adapter still guards its own teardown.
        return
    try:
        yield received
    finally:
        try:
            signal.signal(signal.SIGTERM, previous)
        except (ValueError, OSError, AttributeError):
            pass


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
                from .beta_policy_screening_control import CONTROL_UNAVAILABLE_BEFORE_START, CurrentSourceMonitor
                monitor = CurrentSourceMonitor(value) if value.get("control") is not None else nullcontext()
                try:
                    with _one_stop_signal(root), monitor as active:
                        effective_argv = active.allocator_argv(value["argv"]) if active is not None else value["argv"]
                        return allocator(effective_argv)
                except ValueError as exc:
                    if str(exc) == CONTROL_UNAVAILABLE_BEFORE_START:
                        fence_before_create()  # Nothing started: recovery retries, counted toward its cap.
                    raise
            finally:
                for key, old in previous.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old
    except BlockingIOError:
        return 2


STOP_SIGNAL_RECORD = "stop_signalled.json"


def record_stop_signal(root: Path, *, sender: str, reason: str, pid: int) -> bool:
    """Claim this job's one stop signal before sending it.

    False only when another sender already recorded it. A record that cannot
    be written still lets the stop go out: the allocator ignores a repeat.
    """
    from .beta_policy_screening_launch import _write
    try:
        _write(Path(root) / STOP_SIGNAL_RECORD, {
            "schema_version": "blueprint.allocator_stop_signal.v1", "sender": sender, "reason": reason,
            "pid": pid, "signalled_at": utc_now_iso()})
    except FileExistsError:
        return False
    except OSError:
        return True
    return True


def stop_invocation(path: Path, *, reason: str) -> bool:
    """Signal only this invocation's verified allocator, once, preserving its finally."""
    import signal
    from .team_candidate_quick10_launch import _process_start_ticks
    value = manifest(path)
    root = Path(value["root"])
    started = _read(root / "started.json")
    if (started.get("manifest_digest") != value["manifest_digest"] or not started.get("start_ticks")
            or _process_start_ticks(started["pid"]) != started["start_ticks"]):
        raise ValueError("beta_screening_allocator_stop_identity_unproven")
    if not record_stop_signal(root, sender="screening_reconcile", reason=reason, pid=int(started["pid"])):
        return False
    os.kill(int(started["pid"]), signal.SIGTERM)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    return run_invocation(parser.parse_args(argv).manifest)


if __name__ == "__main__":
    raise SystemExit(main())
