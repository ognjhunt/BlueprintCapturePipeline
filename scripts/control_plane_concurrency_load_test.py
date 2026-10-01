"""Plan 13c measurement and acceptance for the joined no-paid scene harness.

ADP-009D/day 28. Reports are development_only. The measurement layer cannot
substitute for the required production stage transitions or live worker proof.
"""

from __future__ import annotations

import math
import os
import argparse
import sys
import socket
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from blueprint_pipeline.control_plane_disk_usage import tree_usage

REQUIRED_STAGES = (
    "scene_intake", "scene_preparation", "scene_configuration", "launch_preparation",
    "episode_compilation", "launch_activation", "policy_output_ingestion", "result_delivery", "retirement",
)


def child_environment(environment: dict[str, str]) -> dict[str, str]:
    """Allow runtime paths only; production credentials and root settings never leak."""
    return {**{key: value for key, value in environment.items()
              if key in {"PATH", "PYTHONPATH", "LANG", "LC_ALL"}}, "PYTHONDONTWRITEBYTECODE": "1"}


def install_child_fences(owned_roots: Sequence[Path]) -> None:
    """A fresh benchmark child has no network, provider subprocess or live writer."""
    roots = tuple(Path(root).resolve(strict=True) for root in owned_roots)

    def owned(path: Any) -> bool:
        if isinstance(path, int):
            # Only standard streams and descriptors already opened by this
            # fresh child arrive here; every path open was audited separately.
            return True
        try:
            target = Path(os.fsdecode(path)).resolve()
        except (TypeError, ValueError):
            return False
        return any(target == root or target.is_relative_to(root) for root in roots)

    def check(path: Any) -> None:
        if not owned(path):
            raise PermissionError("concurrency_harness_write_outside_owned_roots")

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event in {"socket.connect", "socket.connect_ex", "socket.bind"}:
            if args[0].family == socket.AF_UNIX and owned(args[1]):
                return
            raise PermissionError("concurrency_harness_network_denied")
        if event == "socket.getaddrinfo":
            raise PermissionError("concurrency_harness_network_denied")
        if event == "subprocess.Popen":
            command = args[1]
            arguments = list(command[1:]) if isinstance(command, (list, tuple)) else []
            if len(arguments) >= 3 and arguments[0] == "-C":
                arguments = arguments[2:]
            if (not isinstance(command, (list, tuple)) or len(command) < 2
                    or Path(command[0]).name != "git"
                    or not arguments or arguments[0] not in {"rev-parse", "status", "show", "ls-files", "archive"}
                    or any(arg.startswith(("--output", "--exec")) for arg in arguments)):
                raise PermissionError("concurrency_harness_provider_subprocess_denied")
        if event == "open":
            path, mode, flags = args
            if not isinstance(path, int):
                target = Path(os.fsdecode(path)).resolve()
                if (target.is_relative_to('/etc/blueprint')
                        or '.blueprint-secrets' in target.parts):
                    raise PermissionError("concurrency_harness_live_credentials_denied")
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                check(path)
        if event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.chown", "os.utime"}:
            check(args[0])
        if event in {"os.rename", "os.link"}:
            check(args[0])
            check(args[1])
        if event == "os.symlink":
            check(args[1])

    sys.addaudithook(audit)


def argument_parser() -> argparse.ArgumentParser:
    def positive_int(value: str) -> int:
        number = int(value)
        if number < 1 or number > 3600:
            raise argparse.ArgumentTypeError("must be an integer from 1 to 3600")
        return number

    def positive_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise argparse.ArgumentTypeError("must be finite and positive")
        return number

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-beta-concurrency", type=positive_int, required=True)
    parser.add_argument("--owner-confirmed-concurrency", action="store_true")
    parser.add_argument("--maximum-retained-gib", type=positive_float, required=True)
    parser.add_argument("--maximum-delta-gib", type=positive_float, required=True)
    for flag in ("control-plane-root", "object-store-root", "worker-root", "report"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--child-timeout-seconds", type=positive_int, default=300)
    parser.add_argument("--global-timeout-seconds", type=positive_int, default=900)
    return parser


def p95(values: Sequence[float]) -> float:
    """Nearest-rank p95; an empty or invalid sample is never zero."""
    if not values or any(isinstance(v, bool) or not isinstance(v, (float, int))
                         or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("unmeasured_or_invalid_samples")
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def allocated_tree_bytes(root: Path) -> int:
    usage = tree_usage(root)
    if usage.unreadable:
        raise ValueError("allocation_measurement_incomplete")
    return usage.allocated_bytes


class AllocationSampler:
    """Continuously measure allocated bytes, retaining every measurement error."""
    def __init__(self, root: Path, *, interval_seconds: float = 0.1):
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("allocation_sampling_interval_invalid")
        self.root, self.interval_seconds = root, interval_seconds
        self.stop = threading.Event()
        self.errors: list[Exception] = []
        self.sample_count = self.peak_bytes = self.final_bytes = 0
        self.incomplete_scan_count = 0

    def sample(self):
        # A production toolchain publication renames a whole temporary tree.
        # Retry for at most 380ms and retain the incomplete-scan count.
        for attempt in range(20):
            try:
                value = allocated_tree_bytes(self.root)
                break
            except ValueError as exc:
                if str(exc) != "allocation_measurement_incomplete":
                    raise
                self.incomplete_scan_count += 1
                if attempt == 19:
                    raise
                time.sleep(0.02)
        self.final_bytes = value
        self.peak_bytes = max(self.peak_bytes, value)
        self.sample_count += 1

    def __enter__(self):
        self.sample()
        self.initial_bytes = self.final_bytes
        def monitor():
            while not self.stop.wait(self.interval_seconds):
                try:
                    self.sample()
                except Exception as exc:
                    self.errors.append(exc)
                    return
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join()
        if kind is None:
            if self.errors:
                raise self.errors[0]
            self.sample()
        return False


def create_run_roots(control_plane: Path, objects: Path, workers: Path) -> dict[str, Path]:
    """Create exclusive owned roots after checking all paths; never reuse user work."""
    roots = {"control_plane": Path(control_plane), "objects": Path(objects), "workers": Path(workers)}
    resolved = []
    for path in roots.values():
        if path.exists() or path.is_symlink():
            raise ValueError("run_root_exists")
        for parent in path.parents:
            if parent.is_symlink():
                raise ValueError("run_root_parent_symlink")
        resolved.append(path.absolute())
    for i, first in enumerate(resolved):
        for second in resolved[i + 1:]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError("run_roots_overlap")
    for path in roots.values():
        path.mkdir(mode=0o700)
        marker = path / ".concurrency-harness-owner"
        with marker.open("x") as stream:
            stream.write(f"control_plane_concurrency_load_test.v1 uid={os.geteuid()}\n")
        marker.chmod(0o600)
    return roots


def build_summary(*, source_commit: str, expected_beta_concurrency: int, owner_confirmed: bool,
                  scenes: list[dict[str, Any]], measured_peak_concurrency: int, concurrency_hold_seconds: float,
                  baseline_allocated_bytes: int, final_allocated_bytes: int, maximum_retained_bytes: int,
                  maximum_delta_bytes: int, external_calls: int) -> dict[str, Any]:
    """Fail acceptance when even one chain, concurrency or retention gate is absent."""
    count = 2 * expected_beta_concurrency
    blockers: list[str] = []
    if len(scenes) != count or len({s["scene_id"] for s in scenes}) != count:
        blockers.append("requested_scene_count_not_completed")
    if measured_peak_concurrency != count or concurrency_hold_seconds <= 0:
        blockers.append("target_concurrency_not_observed")
    if final_allocated_bytes > maximum_retained_bytes:
        blockers.append("control_plane_retained_bytes_exceeded")
    delta = final_allocated_bytes - baseline_allocated_bytes
    if delta > maximum_delta_bytes:
        blockers.append("control_plane_disk_delta_exceeded")
    if external_calls:
        blockers.append("external_provider_calls_observed")
    samples: dict[str, list[dict[str, Any]]] = {name: [] for name in REQUIRED_STAGES}
    completed = 0
    for scene in scenes:
        identity = scene["scene_id"]
        rows = scene["stages"]
        good = [row.get("stage") for row in rows] == list(REQUIRED_STAGES)
        if not good:
            blockers.append(f"scene_chain_incomplete:{identity}")
        for row in rows:
            if row.get("status") != "completed":
                good = False
                blockers.extend(row.get("blockers") or [f"stage_incomplete:{identity}:{row.get('stage')}"])
            if row.get("stage") in samples and row.get("status") == "completed":
                samples[row["stage"]].append(row)
        for field in ("residual_leases", "residual_pins"):
            if scene.get(field) != 0:
                blockers.append(f"{field}:{identity}")
                good = False
        completed += int(good)
    metrics = {name: {field + "_p95": p95([row[field] for row in rows])
                     for field in ("wall_seconds", "cpu_seconds", "peak_allocated_bytes")}
               for name, rows in samples.items() if rows}
    return {"schema_version": "control_plane_concurrency_acceptance.v1", "source_commit": source_commit,
            "claim_ceiling": "development_only", "expected_beta_concurrency": expected_beta_concurrency,
            "requested_scenes": count, "completed_scenes": completed,
            "measured_peak_concurrency": measured_peak_concurrency,
            "concurrency_hold_seconds": concurrency_hold_seconds, "p95_method": "nearest_rank",
            "stage_metrics": metrics, "baseline_allocated_bytes": baseline_allocated_bytes,
            "final_allocated_bytes": final_allocated_bytes, "control_plane_delta_bytes": delta,
            "maximum_retained_bytes": maximum_retained_bytes, "maximum_delta_bytes": maximum_delta_bytes,
            "owner_confirmed_concurrency": owner_confirmed,
            "owner_sized_acceptance_complete": owner_confirmed and not blockers,
            "external_calls": external_calls, "status": "failed" if blockers else "passed",
            "blockers": sorted(set(blockers)), "scenes": scenes}
