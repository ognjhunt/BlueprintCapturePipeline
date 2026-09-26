"""The volume-mount procedure plans by default, refuses to move production roots without the acknowledgement, and moves them under a prefix."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path, PurePosixPath

from blueprint_pipeline.control_plane_storage_roots import STORAGE_ROOTS

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy" / "host" / "mount_work_volume.sh"
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
STATE = PurePosixPath("/var/lib/blueprint")
# Classes whose bytes are bulk by nature; they belong on the growable volume.
BULK_CLASSES = frozenset({"cache", "evidence_cold", "scratch", "scene_workspace"})
# Bound as one tree because its stores hardlink into each other, so the small
# durable entries inside it travel with it (the documented exception).
INPUTS_TREE = STATE / "task-evaluation-inputs"
# Work roots that are bulk by nature: the handoff spool, which holds every
# scene's raw capture, and native run work.
BULK_WORK_ROOTS = frozenset(
    {STATE / "pubsub-handoffs", STATE / "pipeline-control-plane" / "native-g1-team-campaign-work"}
)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, check=False, timeout=120)


def _state(tmp_path: Path) -> Path:
    state = tmp_path / "var" / "lib" / "blueprint"
    for rel in ("task-evaluation-inputs/prepared-references", "pipeline-control-plane/task-evaluation-policy-canaries"):
        (state / rel).mkdir(parents=True)
        (state / rel / "payload.bin").write_bytes(b"x" * 4096)
    (state / "pipeline-control-plane" / "task-evaluation-launches" / "pending").mkdir(parents=True)
    return state


def _script_array(name: str) -> list[str]:
    """Read one bash array assignment out of the script text, ignoring comments."""
    words: list[str] = []
    inside = False
    for line in SCRIPT.read_text(encoding="utf-8").splitlines():
        code = line.split("#", 1)[0]
        if not inside:
            if not code.startswith(f"{name}=("):
                continue
            code, inside = code[len(name) + 2 :], True
        if ")" in code:
            return words + code.split(")", 1)[0].split()
        words += code.split()
    raise AssertionError(f"{name} is not assigned in {SCRIPT.name}")


def _within(path: PurePosixPath, root: PurePosixPath) -> bool:
    return path == root or root in path.parents


def _read_write_paths(unit: Path) -> list[PurePosixPath]:
    paths: list[PurePosixPath] = []
    for line in unit.read_text(encoding="utf-8").splitlines():
        if line.startswith("ReadWritePaths="):
            paths += [PurePosixPath(entry.lstrip("-").rstrip("/")) for entry in line.split("=", 1)[1].split()]
    return paths


def test_every_bulk_storage_class_root_is_on_the_volume_and_queues_never_move() -> None:
    volume_roots = [STATE / rel for rel in _script_array("ROOTS")]

    stranded = sorted(
        root.path
        for root in STORAGE_ROOTS
        if root.storage_class in BULK_CLASSES
        and _within(PurePosixPath(root.path), STATE)
        and not any(_within(PurePosixPath(root.path), moved) for moved in volume_roots)
    )
    assert stranded == [], "bulk roots left on the root disk"
    work_roots = {PurePosixPath(root.path) for root in STORAGE_ROOTS if root.storage_class == "work"}
    assert BULK_WORK_ROOTS <= work_roots
    assert sorted(str(root) for root in BULK_WORK_ROOTS if root not in volume_roots) == [], (
        "the handoff spool and native run work move too"
    )

    classified = {PurePosixPath(root.path) for root in STORAGE_ROOTS}
    assert [str(root) for root in volume_roots if root not in classified] == [], "every moved root is classified"
    nested = [(str(a), str(b)) for a in volume_roots for b in volume_roots if a != b and _within(b, a)]
    assert nested == [], "one bind per tree: no moved root lies inside another"

    carried = [
        root for root in STORAGE_ROOTS
        if any(_within(PurePosixPath(root.path), moved) for moved in volume_roots)
    ]
    kept_on_root_disk = sorted(
        root.path
        for root in carried
        if root.storage_class in {"work", "ledger", "evidence_hot"}
        and not _within(PurePosixPath(root.path), INPUTS_TREE)
        and not (root.storage_class == "work" and PurePosixPath(root.path) in BULK_WORK_ROOTS)
    )
    assert kept_on_root_disk == [], "queues, ledgers and hot evidence never move"

    # Roots outside the state tree are not in the storage table; each must be
    # production storage that a unit writes, bound to the same path on the volume.
    absolute_roots = [PurePosixPath(path) for path in _script_array("ABSOLUTE_ROOTS")]
    assert absolute_roots, "the hand-bound CPU prestage work is recorded"
    for absolute in absolute_roots:
        assert absolute.is_absolute() and not _within(absolute, STATE), absolute
        writers = [unit.name for unit in SYSTEMD_DIR.glob("blueprint-*.service") if absolute in _read_write_paths(unit)]
        assert writers, f"no production unit writes {absolute}"


def test_plan_lists_every_bulk_root_and_changes_nothing(tmp_path: Path) -> None:
    state = _state(tmp_path)
    volume = tmp_path / "mnt" / "blueprint-work"
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))

    completed = _run("--device", "/dev/null", "--root-prefix", str(tmp_path), "--plan")

    assert completed.returncode == 0, completed.stderr
    # The inputs tree moves as one root, never store by store.
    assert f"move     {state}/task-evaluation-inputs -> {volume}/task-evaluation-inputs (" in completed.stdout
    assert f"{state}/task-evaluation-inputs/prepared-references" not in completed.stdout
    assert f"missing  {state}/pubsub-handoffs" in completed.stdout
    assert f"missing  {tmp_path}/workspace" in completed.stdout
    assert "task-evaluation-launches" not in completed.stdout, "queues never move"
    assert "nothing changed" in completed.stdout
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before


def test_apply_refuses_without_the_acknowledgement_and_moves_roots_with_it(tmp_path: Path) -> None:
    state = _state(tmp_path)
    (tmp_path / "workspace" / "prestage").mkdir(parents=True)
    (tmp_path / "workspace" / "prestage" / "work.bin").write_bytes(b"w" * 1024)
    refused = _run("--device", "/dev/null", "--root-prefix", str(tmp_path), "--apply")
    assert refused.returncode == 2 and "move-work-roots-to-volume" in refused.stderr

    applied = _run("--device", "/dev/null", "--root-prefix", str(tmp_path), "--apply", "--ack", "move-work-roots-to-volume")

    assert applied.returncode == 0, applied.stderr + applied.stdout
    volume = tmp_path / "mnt" / "blueprint-work"
    moved = volume / "task-evaluation-inputs" / "prepared-references" / "payload.bin"
    assert moved.read_bytes() == b"x" * 4096
    assert (volume / "pipeline-control-plane" / "task-evaluation-policy-canaries" / "payload.bin").is_file()
    assert (volume / "workspace" / "prestage" / "work.bin").read_bytes() == b"w" * 1024
    # Each original root is swapped for an empty directory (the bind-mount target) and the copy removed.
    for original in (state / "task-evaluation-inputs", tmp_path / "workspace"):
        assert original.is_dir() and list(original.iterdir()) == []
        assert not original.with_name(original.name + ".migrated-to-volume").exists()
    assert (state / "pipeline-control-plane" / "task-evaluation-launches" / "pending").is_dir()
    assert os.access(SCRIPT, os.X_OK)
