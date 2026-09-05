"""The volume-mount procedure plans by default, refuses to move production roots without the acknowledgement, and moves them under a prefix."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy" / "host" / "mount_work_volume.sh"


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, check=False, timeout=120)


def _state(tmp_path: Path) -> Path:
    state = tmp_path / "var" / "lib" / "blueprint"
    for rel in ("task-evaluation-inputs/prepared-references", "pipeline-control-plane/task-evaluation-policy-canaries"):
        (state / rel).mkdir(parents=True)
        (state / rel / "payload.bin").write_bytes(b"x" * 4096)
    (state / "pipeline-control-plane" / "task-evaluation-launches" / "pending").mkdir(parents=True)
    return state


def test_plan_lists_every_bulk_root_and_changes_nothing(tmp_path: Path) -> None:
    state = _state(tmp_path)
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))

    completed = _run("--device", "/dev/null", "--root-prefix", str(tmp_path), "--plan")

    assert completed.returncode == 0, completed.stderr
    assert f"move     {state}/task-evaluation-inputs/prepared-references" in completed.stdout
    assert f"missing  {state}/task-evaluation-inputs/compiled-episodes" in completed.stdout
    assert "task-evaluation-launches" not in completed.stdout, "queues never move"
    assert "nothing changed" in completed.stdout
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before


def test_apply_refuses_without_the_acknowledgement_and_moves_roots_with_it(tmp_path: Path) -> None:
    state = _state(tmp_path)
    refused = _run("--device", "/dev/null", "--root-prefix", str(tmp_path), "--apply")
    assert refused.returncode == 2 and "move-work-roots-to-volume" in refused.stderr

    applied = _run("--device", "/dev/null", "--root-prefix", str(tmp_path), "--apply", "--ack", "move-work-roots-to-volume")

    assert applied.returncode == 0, applied.stderr + applied.stdout
    volume = tmp_path / "mnt" / "blueprint-work"
    moved = volume / "task-evaluation-inputs" / "prepared-references" / "payload.bin"
    assert moved.read_bytes() == b"x" * 4096
    assert (volume / "pipeline-control-plane" / "task-evaluation-policy-canaries" / "payload.bin").is_file()
    # The original root is swapped for an empty directory (the bind-mount target) and the copy removed.
    original = state / "task-evaluation-inputs" / "prepared-references"
    assert original.is_dir() and list(original.iterdir()) == []
    assert not (state / "task-evaluation-inputs" / "prepared-references.migrated-to-volume").exists()
    assert (state / "pipeline-control-plane" / "task-evaluation-launches" / "pending").is_dir()
    assert os.access(SCRIPT, os.X_OK)
