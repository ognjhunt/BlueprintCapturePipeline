"""The volume-mount procedure plans by default, refuses to move production roots without the acknowledgement, and moves them under a prefix."""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from blueprint_pipeline.control_plane_storage_roots import STORAGE_ROOTS

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "deploy" / "host" / "mount_work_volume.sh"
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
ACK = "move-work-roots-to-volume"
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
# Intake stays up during the move: it writes queues, which never move, and the
# reproducible result artifact cache, where a write that lands mid-move is
# caught by the script's verification.
UNITS_LEFT_RUNNING = frozenset({"blueprint-pipeline-intake.service"})
_HOST_PATH = re.compile(r"(?<![A-Za-z0-9_.-])(/var/lib/blueprint[A-Za-z0-9_./-]*|/workspace[A-Za-z0-9_./-]*)")


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, check=False, timeout=120)


def _state(tmp_path: Path) -> Path:
    state = tmp_path / "var" / "lib" / "blueprint"
    for rel in ("task-evaluation-inputs/prepared-references", "pipeline-control-plane/task-evaluation-policy-canaries"):
        (state / rel).mkdir(parents=True)
        (state / rel / "payload.bin").write_bytes(b"x" * 4096)
    (state / "pipeline-control-plane" / "task-evaluation-launches" / "pending").mkdir(parents=True)
    return state


def _bound_state(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A host from the September layout: prepared-references bound from the volume on its own."""
    state = tmp_path / "var" / "lib" / "blueprint"
    volume = tmp_path / "mnt" / "blueprint-work"
    inputs = state / "task-evaluation-inputs"
    (inputs / "prepared-references").mkdir(parents=True)  # the old bind's mount point
    (inputs / "launch-activations").mkdir()
    (inputs / "launch-activations" / "payload.bin").write_bytes(b"a" * 4096)
    (volume / "task-evaluation-inputs" / "prepared-references").mkdir(parents=True)
    (volume / "task-evaluation-inputs" / "prepared-references" / "payload.bin").write_bytes(b"p" * 4096)
    bound = tmp_path / "bound-roots"
    bound.write_text("/var/lib/blueprint/task-evaluation-inputs/prepared-references\n", encoding="utf-8")
    return state, volume, bound


def _hermetic(tmp_path: Path, bound: Path) -> tuple[str, ...]:
    return ("--device", "/dev/null", "--root-prefix", str(tmp_path), "--bound-roots-file", str(bound))


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


def _script_array(name: str) -> list[str]:
    """Read one bash array assignment out of the script text, ignoring comments and quotes."""
    words: list[str] = []
    inside = False
    for line in SCRIPT.read_text(encoding="utf-8").splitlines():
        code = line.split("#", 1)[0]
        if not inside:
            if not code.startswith(f"{name}=("):
                continue
            code, inside = code[len(name) + 2 :], True
        words += [word.strip("'\"") for word in code.split(")", 1)[0].split()]
        if ")" in code:
            return words
    raise AssertionError(f"{name} is not assigned in {SCRIPT.name}")


def _within(path: PurePosixPath, root: PurePosixPath) -> bool:
    return path == root or root in path.parents


def _read_write_paths(unit: Path) -> list[PurePosixPath]:
    paths: list[PurePosixPath] = []
    for line in unit.read_text(encoding="utf-8").splitlines():
        if line.startswith("ReadWritePaths="):
            paths += [PurePosixPath(entry.lstrip("-").rstrip("/")) for entry in line.split("=", 1)[1].split()]
    return paths


def _volume_roots() -> list[PurePosixPath]:
    return [STATE / rel for rel in _script_array("ROOTS")] + [
        PurePosixPath(path) for path in _script_array("ABSOLUTE_ROOTS")
    ]


def _writes_under(service: Path, roots: list[PurePosixPath]) -> bool:
    """A writable sandbox path is a moved root or lies inside one, or covers a moved path the unit names."""
    writable = _read_write_paths(service)
    named = [
        PurePosixPath(match.rstrip("/"))
        for line in service.read_text(encoding="utf-8").splitlines()
        if not line.startswith(("ReadOnlyPaths=", "InaccessiblePaths=", "Condition", "#"))
        for match in _HOST_PATH.findall(line)
    ]
    return any(
        any(_within(path, root) for path in writable)
        or any(_within(path, root) and any(_within(path, w) for w in writable) for path in named)
        for root in roots
    )


def _triggers(service: Path) -> list[Path]:
    """Timers and path units that would start the service again."""
    triggers: list[Path] = []
    for unit in sorted([*SYSTEMD_DIR.glob("blueprint-*.timer"), *SYSTEMD_DIR.glob("blueprint-*.path")]):
        lines = unit.read_text(encoding="utf-8").splitlines()
        target = next((line.split("=", 1)[1].strip() for line in lines if line.startswith("Unit=")), f"{unit.stem}.service")
        if target == service.name:
            triggers.append(unit)
    return triggers


def test_every_unit_that_writes_under_a_moved_root_stops_for_the_move() -> None:
    listed = _script_array("WORKER_UNITS")
    assert sorted(unit for unit in listed if not (SYSTEMD_DIR / unit).is_file()) == [], "every listed unit exists"
    roots = _volume_roots()
    running = sorted(
        unit.name
        for service in SYSTEMD_DIR.glob("blueprint-*.service")
        if service.name not in UNITS_LEFT_RUNNING and _writes_under(service, roots)
        for unit in (service, *_triggers(service))
        if unit.name not in listed
    )
    assert running == [], "these units write under a moved root and would keep running during the move"


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
        if root.storage_class in {"work", "ledger"}
        and not _within(PurePosixPath(root.path), INPUTS_TREE)
        and not (root.storage_class == "work" and PurePosixPath(root.path) in BULK_WORK_ROOTS)
    )
    assert kept_on_root_disk == [], "queues and ledgers never move"
    # Hot evidence rides on the volume only where the script declares it, and the
    # plan names every declared entry for the owner.  A new hot root inside a moved
    # root fails here until it is declared in EVIDENCE_HOT_ON_VOLUME or moved out.
    carried_hot = sorted(root.path for root in carried if root.storage_class == "evidence_hot")
    declared_hot = sorted(str(STATE / rel) for rel in _script_array("EVIDENCE_HOT_ON_VOLUME"))
    assert carried_hot == declared_hot, "hot evidence on the volume must be declared"

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


def test_apply_consolidates_previously_bound_children_into_one_tree_bind(tmp_path: Path) -> None:
    state, volume, bound = _bound_state(tmp_path)

    applied = _run(*_hermetic(tmp_path, bound), "--apply", "--ack", ACK)

    assert applied.returncode == 0, applied.stderr + applied.stdout
    inputs = volume / "task-evaluation-inputs"
    assert (inputs / "prepared-references" / "payload.bin").read_bytes() == b"p" * 4096
    assert (inputs / "launch-activations" / "payload.bin").read_bytes() == b"a" * 4096
    local = state / "task-evaluation-inputs"
    assert local.is_dir() and list(local.iterdir()) == []
    assert not (state / "task-evaluation-inputs.migrated-to-volume").exists()
    # The per-store bind is gone and the tree is bound whole, so a second run has nothing to do.
    assert bound.read_text(encoding="utf-8").splitlines() == ["/var/lib/blueprint/task-evaluation-inputs"]
    again = _run(*_hermetic(tmp_path, bound), "--apply", "--ack", ACK)
    assert again.returncode == 0 and "nothing to move" in again.stdout, again.stderr


def test_plan_reports_consolidation_and_evidence_hot_on_volume(tmp_path: Path) -> None:
    state, volume, bound = _bound_state(tmp_path)
    (state / "pipeline-control-plane" / "engineering").mkdir(parents=True)
    with bound.open("a", encoding="utf-8") as handle:
        handle.write("/var/lib/blueprint/pipeline-control-plane/engineering\n")
    before = _tree(tmp_path)

    completed = _run(*_hermetic(tmp_path, bound), "--plan")

    assert completed.returncode == 0, completed.stderr
    assert (
        f"consolidate {state}/task-evaluation-inputs -> {volume}/task-evaluation-inputs ("
        in completed.stdout
    )
    assert "; bound children: prepared-references)" in completed.stdout
    assert f"bound    {state}/pipeline-control-plane/engineering (" in completed.stdout
    for rel in ("sam31-profile-registry", "task-evaluation-terminal-results", "g1-team-campaign-registry.json"):
        assert f"evidence_hot on volume: {state}/task-evaluation-inputs/{rel}" in completed.stdout
    assert "nothing changed" in completed.stdout
    assert _tree(tmp_path) == before


def test_hardlinks_across_input_stores_survive_the_move(tmp_path: Path) -> None:
    probe = subprocess.run(["rsync", "--help"], capture_output=True, text=True, check=False)
    if "--hard-links" not in probe.stdout + probe.stderr:
        pytest.skip("the local rsync does not offer --hard-links")
    state = tmp_path / "var" / "lib" / "blueprint"
    blob = state / "task-evaluation-inputs" / "prepared-references" / "content-addressed" / "sha256" / "ab"
    member = state / "task-evaluation-inputs" / "compiled-episodes" / "episode-1" / "member.bin"
    blob.parent.mkdir(parents=True)
    member.parent.mkdir(parents=True)
    blob.write_bytes(b"m" * 4096)
    os.link(blob, member)

    applied = _run("--device", "/dev/null", "--root-prefix", str(tmp_path), "--apply", "--ack", ACK)

    assert applied.returncode == 0, applied.stderr + applied.stdout
    volume = tmp_path / "mnt" / "blueprint-work"
    moved_blob = os.stat(volume / blob.relative_to(state))
    moved_member = os.stat(volume / member.relative_to(state))
    assert (moved_blob.st_ino, moved_blob.st_nlink) == (moved_member.st_ino, 2)


def test_apply_keeps_the_original_when_it_holds_bytes_the_volume_copy_lacks(tmp_path: Path) -> None:
    state, volume, bound = _bound_state(tmp_path)
    # Bytes written under the old bind's mount point before it was mounted are
    # hidden by it, and surface once it is unmounted.
    (state / "task-evaluation-inputs" / "prepared-references" / "hidden.bin").write_bytes(b"h" * 1024)

    applied = _run(*_hermetic(tmp_path, bound), "--apply", "--ack", ACK)

    assert applied.returncode == 3, applied.stderr + applied.stdout
    assert "hidden.bin" in applied.stderr
    kept = state / "task-evaluation-inputs.migrated-to-volume"
    assert (kept / "prepared-references" / "hidden.bin").read_bytes() == b"h" * 1024
    # The old bind's mount point is never copied onto its own volume copy.
    assert not (volume / "task-evaluation-inputs" / "prepared-references" / "hidden.bin").exists()
    assert (volume / "task-evaluation-inputs" / "prepared-references" / "payload.bin").is_file()
    replanned = _run(*_hermetic(tmp_path, bound), "--plan")
    assert f"kept     {kept} (" in replanned.stdout


def test_apply_rewrites_fstab_atomically_after_a_backup(tmp_path: Path) -> None:
    _bound_state(tmp_path)
    fstab = tmp_path / "etc" / "fstab"
    fstab.parent.mkdir()
    child = "/var/lib/blueprint/task-evaluation-inputs/prepared-references"
    original = [
        "UUID=root / ext4 defaults 0 1",
        "UUID=volume /mnt/blueprint-work ext4 defaults,nofail,noatime,discard 0 2",
        f"# /mnt/blueprint-work/task-evaluation-inputs/prepared-references {child} none bind 0 0",
        f"/mnt/blueprint-work/task-evaluation-inputs/prepared-references {child} none bind 0 0",
        "/mnt/blueprint-work/pipeline-control-plane/engineering /var/lib/blueprint/pipeline-control-plane/engineering none bind 0 0",
    ]
    fstab.write_text("\n".join(original) + "\n", encoding="utf-8")
    fstab.chmod(0o644)

    applied = _run(*_hermetic(tmp_path, tmp_path / "bound-roots"), "--apply", "--ack", ACK)

    assert applied.returncode == 0, applied.stderr + applied.stdout
    backups = sorted(fstab.parent.glob("fstab.blueprint-*.bak"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8").splitlines() == original
    assert fstab.read_text(encoding="utf-8").splitlines() == [
        *original[:3],
        original[4],
        "/mnt/blueprint-work/task-evaluation-inputs /var/lib/blueprint/task-evaluation-inputs none bind 0 0",
    ]
    assert stat.S_IMODE(fstab.stat().st_mode) == 0o644
    assert sorted(p.name for p in fstab.parent.iterdir()) == sorted(["fstab", backups[0].name])


def test_apply_refuses_to_move_a_root_over_a_copy_an_earlier_move_kept(tmp_path: Path) -> None:
    state = _state(tmp_path)
    kept = state / "task-evaluation-inputs.migrated-to-volume"
    kept.mkdir()
    (kept / "evidence.bin").write_bytes(b"e" * 64)

    applied = _run("--device", "/dev/null", "--root-prefix", str(tmp_path), "--apply", "--ack", ACK)

    assert applied.returncode == 2, applied.stderr + applied.stdout
    assert f"{kept}" in applied.stderr
    assert (kept / "evidence.bin").read_bytes() == b"e" * 64
    assert (state / "task-evaluation-inputs" / "prepared-references" / "payload.bin").is_file()
    assert not (tmp_path / "mnt" / "blueprint-work" / "task-evaluation-inputs").exists()


def test_apply_refuses_a_bound_child_the_volume_has_no_copy_of(tmp_path: Path) -> None:
    state, volume, bound = _bound_state(tmp_path)
    shutil.rmtree(volume / "task-evaluation-inputs" / "prepared-references")
    before = _tree(state), bound.read_text(encoding="utf-8")

    applied = _run(*_hermetic(tmp_path, bound), "--apply", "--ack", ACK)

    assert applied.returncode == 2, applied.stderr + applied.stdout
    assert "prepared-references" in applied.stderr
    assert (_tree(state), bound.read_text(encoding="utf-8")) == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through a read-only file")
def test_apply_keeps_the_old_binds_when_one_will_not_unmount(tmp_path: Path) -> None:
    state, volume, bound = _bound_state(tmp_path)
    bound.chmod(0o444)  # the hermetic stand-in for a busy mount point
    before = _tree(state), bound.read_text(encoding="utf-8")

    applied = _run(*_hermetic(tmp_path, bound), "--apply", "--ack", ACK)

    assert applied.returncode == 2, applied.stderr + applied.stdout
    assert "could not unmount /var/lib/blueprint/task-evaluation-inputs/prepared-references" in applied.stderr
    assert (_tree(state), bound.read_text(encoding="utf-8")) == before
    assert (volume / "task-evaluation-inputs" / "prepared-references" / "payload.bin").is_file()
