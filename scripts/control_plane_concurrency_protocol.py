"""Owned file barriers and original reservation readback for isolated children."""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from pathlib import Path

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from scripts.control_plane_concurrency_load_test import install_child_fences, REQUIRED_STAGES


def seal_document(path: Path, value: dict) -> dict:
    value = json.loads(json.dumps(value, default=str))
    value["receipt_digest"] = canonical_digest(value, digest_field="receipt_digest")
    temporary = path.parent / (".receipt-" + uuid.uuid4().hex)
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink()
    return value


def observe_overlap(*, children: list, ready_root: Path, reservation_root: Path) -> int:
    """Read the original live positive holds; a flag or ready file alone is insufficient."""
    if reservation_root.is_symlink():
        raise ValueError("overlap_reservation_not_owned")
    if not reservation_root.exists():
        if any(ready_root.glob("*.json")):
            raise ValueError("overlap_reservation_not_owned")
        return 0
    root = reservation_root.resolve(strict=True)
    if (
        not children
        or len({row["pid"] for row in children}) != len(children)
        or len({row["scene_key"] for row in children}) != len(children)
    ):
        raise ValueError("overlap_child_identity_not_distinct")
    count = 0
    for child in children:
        if not child["running"]:
            raise ValueError("overlap_child_not_running")
        path = ready_root / (child["scene_key"] + ".json")
        if not path.exists():
            continue
        if path.is_symlink():
            raise ValueError("overlap_ready_path_unsafe")
        ready = json.loads(path.read_text())
        lease = Path(ready["lease_path"])
        if (
            not lease.is_absolute()
            or ".." in lease.parts
            or lease.parent != root
            or lease.is_symlink()
            or not lease.is_file()
        ):
            raise ValueError("overlap_reservation_not_owned")
        value = json.loads(lease.read_text())
        if (
            ready["scene_key"] != child["scene_key"]
            or ready["pid"] != child["pid"]
            or value.get("schema_version") != "control_plane_disk_reservation.v1"
            or value.get("pid") != child["pid"]
            or value.get("role") != "policy_canary_output"
            or type(value.get("expected_bytes")) is not int
            or value["expected_bytes"] <= 0
            or value["expected_bytes"] != ready["expected_bytes"]
            or value.get("device") != root.stat().st_dev
            or value.get("expires_at_epoch", 0) <= time.time()
            or lease.name != value.get("token", "") + ".json"
        ):
            raise ValueError("overlap_reservation_invalid")
        os.kill(child["pid"], 0)
        count += 1
    return count


class HeavySlot:
    """Serialize the original 20GiB preparation holds on the shared disk."""

    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        descriptor = os.open(self.path, os.O_RDWR | os.O_NOFOLLOW)
        self.stream = os.fdopen(descriptor, "r+b")
        fcntl.flock(self.stream, fcntl.LOCK_EX)
        return self

    def __exit__(self, *args):
        fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()


def _wait(path: Path, deadline: float):
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError("concurrency_barrier_timeout")
        time.sleep(0.05)
    if path.is_symlink():
        raise ValueError("concurrency_barrier_path_unsafe")


def run_child(spec_path: Path) -> int:
    """Fresh child: immutable source, no network or provider subprocess authority."""
    spec = json.loads(spec_path.read_text())
    roots = [
        Path(spec[name]).resolve(strict=True)
        for name in ("control_root", "object_root", "worker_root")
    ]
    os.environ["TMPDIR"] = str(Path(spec["temporary_root"]))
    install_child_fences(roots)
    deadline = time.monotonic() + spec["child_timeout_seconds"]
    stages = []
    try:
        seal_document(
            Path(spec["startup_ready_root"]) / (spec["scene_key"] + ".json"),
            {"pid": os.getpid(), "scene_key": spec["scene_key"]},
        )
        _wait(Path(spec["start_barrier"]), deadline)
        registry = json.loads((roots[0] / "producer-registry.json").read_text())
        if not any(
            row == {"pid": os.getpid(), "scene_key": spec["scene_key"]}
            for row in registry["producers"]
        ):
            raise ValueError("concurrency_child_not_registered")

        def reserved(hold):
            seal_document(
                Path(spec["ready_root"]) / (spec["scene_key"] + ".json"),
                {
                    "scene_key": spec["scene_key"],
                    "pid": os.getpid(),
                    "lease_path": str(hold.path),
                    "expected_bytes": hold.expected_bytes,
                    "observed_at_epoch": time.time(),
                },
            )
            _wait(Path(spec["release_barrier"]), deadline)

        from scripts.control_plane_concurrency_runner import run_scene

        result = run_scene(
            scene_key=spec["scene_key"],
            control_root=roots[0],
            object_root=roots[1],
            worker_root=roots[2],
            source_commit=spec["source_commit"],
            runtime_bundle=spec["runtime_bundle"],
            release_binding=spec["release_binding"],
            heavy_slot=HeavySlot(Path(spec["heavy_lock"])),
            on_reserved=reserved,
            stage_recorder=stages,
        )
        result["producer_pid"] = os.getpid()
        seal_document(Path(spec["result_path"]), result)
        return 0
    except Exception as exc:
        blockers = getattr(exc, "errors", None) or [type(exc).__name__ + ":" + str(exc)]
        stage = REQUIRED_STAGES[min(len(stages), len(REQUIRED_STAGES) - 1)]
        seal_document(
            Path(spec["result_path"]),
            {
                "scene_key": spec["scene_key"],
                "scene_id": spec["scene_key"],
                "source_commit": spec["source_commit"],
                "producer_pid": os.getpid(),
                "claim_ceiling": "development_only",
                "actual_provider_calls": 0,
                "stages": stages + [{"stage": stage, "status": "failed", "blockers": blockers}],
                "status": "failed",
                "blockers": blockers,
            },
        )
        return 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child-spec", type=Path, required=True)
    raise SystemExit(run_child(parser.parse_args().child_spec))
