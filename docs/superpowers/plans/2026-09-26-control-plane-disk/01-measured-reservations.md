# PR 1 — Measured reservations (design doc 1a)

> Read `00-index.md` first (ground rules, commands, commit rules).

**Goal:** reservations and whole-chain admission use what each role actually wrote (p95 of
measured samples × 1.25, clamped to `[64 MiB, declared]`) instead of a flat 2 GiB. The
capacity controller, intake headroom and the chain preflight compute admission with the same
function the ledger uses.

**Branch / base / worktree:** `claude/disk-1a-measured-reservations` from `origin/main`,
`$WORKSPACE/BlueprintCapturePipeline-disk-1a-20260926`.

## Facts this PR is built on (verified in code)

- `src/blueprint_pipeline/control_plane_disk_budget.py` owns the ledger:
  - `reserve_control_plane_disk` (204-316) writes `<ledger>/<token>.json` with `pid`, `device`,
    `expected_bytes`, `created_at_epoch` and `expires_at_epoch`.
  - Liveness requires the same device, an unexpired TTL and a live pid (`_load_live_reservations`, 114-142).
  - `DiskReservation.release()` only unlinks the ledger file (157-161).
- `control_plane_capacity_controller.measure_mount` (113-154) and `whole_chain_admission` (157-173)
  re-implement admission with three differences:
  - they ignore `BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES`;
  - they ignore the `BLUEPRINT_CONTROL_PLANE_DISK_FOOTPRINT_<ROLE>_BYTES` overrides;
  - `live_reserved_bytes` (91-110) has no device filter and no pid liveness check.
- `task_evaluation_production_chain_preflight.disk_admission_check` (1550-1596) hard-codes a
  third copy of the footprints.
- Reservation targets are **shared parents**. The per-job workspace is a child:
  - R1 prep `prepared-references/<preparation_id>` (`task_evaluation_launch_preparation_worker.py:986`, release at 1673-1674);
  - R2 compile `compiled-episodes/<compilation_id>` (`task_evaluation_episode_compilation_worker.py:366`, 379-380, release 479-480);
  - R3 activation `launch-activations/<activation_id>` (`task_evaluation_launch_activation_worker.py:1665`, 1705-1706, release 1868-1869);
  - R5 canary `task-evaluation-policy-canaries/<activation_id>` (`task_evaluation_policy_canary_dispatcher.py:2331`, 2328);
  - R6 scene attempt `<factory_output_root>/<intent>/<attempt>` (`task_evaluation_scene_preparation_attempts.py:28`, caller `task_evaluation_scene_progression.py:652-668`);
  - R8 cpu prestage `work` (`task_evaluation_scene_configuration_cpu_prestage.py:354`);
  - R9 semantic pretraining `LOGICAL_ROOT/<key>` (`task_evaluation_artifixer_pretraining.py:309`);
  - R10 stage replay: `mkdtemp` created *after* the reservation (`task_evaluation_stage_replay.py:174`, 217, 423).
- New bytes land in shared content stores and are hardlinked into the per-job directory.
  Measuring unique inodes inside the job directory therefore over-counts cache hits. That is
  acceptable here because it errs conservative and the result is clamped to the declared
  ceiling.
- Long jobs (cpu_prestage, deploy) outlive the 2 h TTL, so their ledger entry is deleted as stale
  while they still run (`_load_live_reservations` + stale deletion at 260-261).
- `launch_dispatch` has no reservation call site, so it keeps its declared default. That is
  intended and conservative.

## File map

| File | Change |
|---|---|
| `src/blueprint_pipeline/control_plane_disk_usage.py` | **new**: `tree_usage()` unique-inode walker (PR 2 extends it) |
| `src/blueprint_pipeline/control_plane_disk_budget.py` | footprint history, `measured_footprint`, `effective_footprint_bytes`, `floor_bytes`, `live_reservations`, per-role TTL, workspace binding on `DiskReservation` |
| `src/blueprint_pipeline/control_plane_capacity_controller.py` | `measure_mount` / `whole_chain_admission` use the shared functions; report footprints and basis |
| `src/blueprint_pipeline/task_evaluation_production_chain_preflight.py` | use `effective_footprint_bytes` / `floor_bytes` |
| worker modules for R1, R2, R3, R5, R6, R8, R9, R10 | pass `workspace=` (or `bind_workspace` after `mkdtemp`) |
| `scripts/deploy_control_plane_commit.py` | reserve a git-tree estimate; observe real bytes; install `disk-reservations/history` |
| `tests/test_control_plane_disk_usage.py` | **new** |
| `tests/test_control_plane_disk_budget.py`, `tests/test_control_plane_capacity_controller.py`, `tests/test_deploy_control_plane_commit.py` | extend |
| `docs/CONTROL_PLANE_STORAGE.md` | admission section: measured footprints |

## Task 1.1 — unique-inode tree usage

**Files:** create `src/blueprint_pipeline/control_plane_disk_usage.py`, `tests/test_control_plane_disk_usage.py`.

- [ ] **Step 1: failing tests**

```python
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_disk_usage.py
from __future__ import annotations

import os

from blueprint_pipeline.control_plane_disk_usage import tree_usage


def _allocated(path):
    metadata = os.lstat(path)
    return getattr(metadata, "st_blocks", 0) * 512 or metadata.st_size


def test_hardlinked_file_counts_once(tmp_path):
    root = tmp_path / "work"
    root.mkdir()
    (root / "a.bin").write_bytes(b"x" * 100_000)
    os.link(root / "a.bin", root / "b.bin")
    usage = tree_usage(root)
    assert usage.unique_inodes == 2  # the directory and one file inode
    assert usage.files == 2  # two names
    assert usage.apparent_bytes == 100_000 + os.lstat(root).st_size
    assert usage.allocated_bytes == _allocated(root / "a.bin") + _allocated(root)
    assert usage.shared_inodes == 1


def test_missing_path_is_zero_and_symlinks_are_not_followed(tmp_path):
    assert tree_usage(tmp_path / "absent").allocated_bytes == 0
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"y" * 50_000)
    root = tmp_path / "work"
    root.mkdir()
    (root / "link").symlink_to(outside)
    usage = tree_usage(root)
    assert usage.apparent_bytes < 50_000


def test_unreadable_entries_are_counted_not_raised(tmp_path, monkeypatch):
    root = tmp_path / "work"
    (root / "sub").mkdir(parents=True)
    real_scandir = os.scandir

    def failing_scandir(path):
        if str(path).endswith("sub"):
            raise PermissionError("denied")
        return real_scandir(path)

    monkeypatch.setattr("blueprint_pipeline.control_plane_disk_usage.os.scandir", failing_scandir)
    assert tree_usage(root).unreadable == 1
```

- [ ] **Step 2:** run `…pytest tests/test_control_plane_disk_usage.py` → FAIL (module missing).
- [ ] **Step 3: implement**

```python
"""Measure disk usage the way the disk sees it: once per inode, in allocated blocks.

Hardlinks are how the control plane shares bytes between content stores, compiled
episodes and launch sets, so a byte count per name overstates usage (on 2026-09-26 two
cache roots "held" about 470 GB on a 165 GB disk). Every measurement here counts each
``(st_dev, st_ino)`` once and reports allocated bytes (``st_blocks * 512``).
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TreeUsage:
    allocated_bytes: int = 0
    apparent_bytes: int = 0
    files: int = 0
    directories: int = 0
    unique_inodes: int = 0
    shared_inodes: int = 0
    unreadable: int = 0


def allocated_bytes(metadata: os.stat_result) -> int:
    blocks = getattr(metadata, "st_blocks", None)
    return int(blocks) * 512 if isinstance(blocks, int) else int(metadata.st_size)


def tree_usage(path: str | Path) -> TreeUsage:
    """Unique-inode usage of ``path`` (a file or a directory tree); never follows symlinks."""

    root = Path(path)
    try:
        top = os.lstat(root)
    except FileNotFoundError:
        return TreeUsage()
    except OSError:
        return TreeUsage(unreadable=1)
    # Only multiply-linked inodes can be reached twice, so only they are remembered.
    seen: set[tuple[int, int]] = set()
    totals = {"allocated": 0, "apparent": 0, "files": 0, "directories": 0,
              "unique": 0, "shared": 0, "unreadable": 0}

    def account(metadata: os.stat_result) -> None:
        if not stat.S_ISDIR(metadata.st_mode) and metadata.st_nlink > 1:
            key = (metadata.st_dev, metadata.st_ino)
            if key in seen:
                return
            seen.add(key)
            totals["shared"] += 1
        totals["unique"] += 1
        totals["allocated"] += allocated_bytes(metadata)
        totals["apparent"] += int(metadata.st_size)

    account(top)
    if not stat.S_ISDIR(top.st_mode):
        totals["files"] += 1
    else:
        totals["directories"] += 1
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as iterator:
                    entries = list(iterator)
            except OSError:
                totals["unreadable"] += 1
                continue
            for entry in entries:
                try:
                    metadata = entry.stat(follow_symlinks=False)
                except OSError:
                    totals["unreadable"] += 1
                    continue
                account(metadata)
                if stat.S_ISDIR(metadata.st_mode):
                    totals["directories"] += 1
                    pending.append(Path(entry.path))
                else:
                    totals["files"] += 1
    return TreeUsage(
        allocated_bytes=totals["allocated"], apparent_bytes=totals["apparent"],
        files=totals["files"], directories=totals["directories"], unique_inodes=totals["unique"],
        shared_inodes=totals["shared"], unreadable=totals["unreadable"],
    )
```

Directories are never deduplicated. Their link count is `2 + subdirectories` and says nothing
about sharing.

- [ ] **Step 4:** tests pass. **Step 5:** ruff, then commit "Measure disk usage once per inode, in allocated blocks".

## Task 1.2 — footprint history and the measured footprint

**Files:** `control_plane_disk_budget.py`, `tests/test_control_plane_disk_budget.py`.

Semantics (implement exactly):

```python
FOOTPRINT_HISTORY_DIRNAME = "history"
FOOTPRINT_SAMPLE_SCHEMA = "control_plane_disk_footprint_sample.v1"
MEASURED_MINIMUM_SAMPLES = 10
MEASURED_WINDOW = 50            # newest completed samples considered
MEASURED_HEADROOM = 1.25
MEASURED_FLOOR_BYTES = 64 * 1024**2
HISTORY_MAX_LINES = 200         # compaction keeps the newest lines
ROLE_TTL_SECONDS = {            # pid liveness is primary; TTL is the backstop
    "cpu_prestage": 12 * 3600, "semantic_pretraining": 12 * 3600,
    "stage_replay": 6 * 3600, "control_plane_deploy": 4 * 3600,
}                               # every other role keeps DEFAULT_TTL_SECONDS (2 h)


def floor_bytes(total_bytes: int) -> int:
    """The admission floor, identical for the ledger, the controller and the preflight."""
    return max(_environment_int("BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES", DEFAULT_FLOOR_BYTES),
               int(total_bytes * DEFAULT_FLOOR_FRACTION))


def record_footprint_sample(*, reservation_root, role, observed_bytes, reserved_bytes, workload=None,
                            outcome="completed", duration_seconds=None, device=None, now=time.time) -> bool:
    """Append one sample to <ledger>/history/<role>.jsonl. Never raises; returns False on failure."""


def measured_footprint(role, *, reservation_root=DEFAULT_RESERVATION_ROOT) -> dict:
    """{"role", "bytes", "basis", "sample_count", "declared_bytes", "p95_bytes"}.

    basis is "measured_p95" when at least MEASURED_MINIMUM_SAMPLES completed samples exist
    in the newest MEASURED_WINDOW, else "declared_default" (bytes == declared).
    p95 is nearest-rank: sorted values s, k = ceil(0.95 * n) - 1, p95 = s[k].
    bytes = max(MEASURED_FLOOR_BYTES, min(declared, ceil(p95 * MEASURED_HEADROOM))).
    declared = footprint_bytes(role) (env override honoured). Unreadable history -> declared.
    """


def effective_footprint_bytes(role, *, reservation_root=DEFAULT_RESERVATION_ROOT) -> int:
    return int(measured_footprint(role, reservation_root=reservation_root)["bytes"])


def live_reservations(reservation_root, *, device, now, pid_alive=_pid_alive) -> tuple[int, int]:
    """(bytes, count) of live reservations on ``device``; read-only (never deletes)."""
```

History file rules:
- The path is `<reservation_root>/history/<role>.jsonl`. `history/` is created with mode
  `0o2770` on first write (reuse the root's group), files `0o660`.
- Each line is one sample: `{"schema_version", "role", "workload", "observed_bytes",
  "reserved_bytes", "outcome", "duration_seconds", "device", "recorded_at_epoch"}`, under 1 KiB.
- `_load_live_reservations` / `live_reservations` must ignore the `history` directory. Today
  they glob `*.json` in the root only, which is already safe; keep it that way.
- Compaction: when the file exceeds 64 KiB, under the ledger `.lock` (exclusive), rewrite it
  atomically keeping the newest `HISTORY_MAX_LINES` lines.
- Only `outcome == "completed"` samples count for the p95. Negative deltas are recorded as 0.

- [ ] **Step 1: failing tests** (add to `tests/test_control_plane_disk_budget.py`; reuse its
  `Usage` namedtuple and `GIB`):

```python
MIB = 1024**2


def _samples(ledger, role, values, outcome="completed"):
    for value in values:
        assert disk_budget.record_footprint_sample(
            reservation_root=ledger, role=role, observed_bytes=value,
            reserved_bytes=2 * GIB, outcome=outcome, now=lambda: 50.0)


def test_short_history_keeps_the_declared_constant(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 9)
    measured = disk_budget.measured_footprint("launch_activation", reservation_root=ledger)
    assert measured == {"role": "launch_activation", "bytes": 2 * GIB, "basis": "declared_default",
                        "sample_count": 9, "declared_bytes": 2 * GIB, "p95_bytes": None}


def test_p95_times_headroom_is_used_once_ten_samples_exist(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 9 + [400 * MIB])
    measured = disk_budget.measured_footprint("launch_activation", reservation_root=ledger)
    assert measured["basis"] == "measured_p95"
    assert measured["p95_bytes"] == 400 * MIB
    assert measured["bytes"] == 500 * MIB


def test_sample_above_the_clamp_cannot_raise_the_reservation(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [10 * GIB] * 12)
    assert disk_budget.effective_footprint_bytes("launch_activation", reservation_root=ledger) == 2 * GIB


def test_tiny_samples_are_floored_and_failed_samples_ignored(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [1] * 10)
    _samples(ledger, "launch_activation", [9 * GIB] * 5, outcome="failed")
    assert disk_budget.effective_footprint_bytes("launch_activation", reservation_root=ledger) == 64 * MIB


def test_unreadable_history_falls_back_to_declared(tmp_path):
    ledger = tmp_path / "ledger"
    (ledger / "history").mkdir(parents=True)
    (ledger / "history" / "launch_activation.jsonl").write_text("{not json\n" * 20)
    assert disk_budget.measured_footprint("launch_activation", reservation_root=ledger)["basis"] == "declared_default"


def test_history_is_compacted(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [MIB] * 400)
    lines = (ledger / "history" / "launch_activation.jsonl").read_text().splitlines()
    assert len(lines) <= 400 and len(lines) >= disk_budget.HISTORY_MAX_LINES


def test_live_reservations_filter_device_and_dead_pids(tmp_path):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    for name, device, pid, expires in (("a", 1, 11, 999.0), ("b", 2, 11, 999.0), ("c", 1, 12, 999.0), ("d", 1, 11, 1.0)):
        (ledger / f"{name}.json").write_text(json.dumps({"device": device, "pid": pid, "expected_bytes": GIB,
                                                         "expires_at_epoch": expires}))
    assert disk_budget.live_reservations(ledger, device=1, now=100.0, pid_alive=lambda pid: pid == 11) == (GIB, 1)
    assert sorted(p.name for p in ledger.glob("*.json")) == ["a.json", "b.json", "c.json", "d.json"]
```

- [ ] **Step 2:** run → FAIL. **Step 3:** implement per the semantics block. **Step 4:** pass. **Step 5:** commit "Keep a history of what each disk role really wrote and admit on its p95".

## Task 1.3 — reservations bind a workspace and record their sample on release

**Files:** `control_plane_disk_budget.py`, tests.

API (backward compatible; every existing caller keeps working unchanged):

```python
def reserve_control_plane_disk(role, *, target_root, expected_bytes=None, reservation_root=...,
                               ttl_seconds=None, disk_usage=..., now=..., pid_alive=..., evictor=None,
                               workspace=None, workload=None) -> DiskReservation:
    # ttl_seconds None -> ROLE_TTL_SECONDS.get(role, DEFAULT_TTL_SECONDS)
    # expected_bytes None -> measured_footprint(role)["bytes"], basis/sample_count recorded
    # expected_bytes given -> basis "caller_exact", sample_count None
    # workspace given -> baseline = tree_usage(workspace).allocated_bytes measured BEFORE the ledger lock


@dataclass
class DiskReservation:
    ...existing fields...
    reservation_root: Path | None = None
    device: int | None = None
    footprint_basis: str = "caller_exact"
    footprint_sample_count: int | None = None
    workload: str | None = None
    workspace: Path | None = None
    baseline_bytes: int = 0
    peak_delta_bytes: int | None = None     # None until a workspace or an observation exists
    started_at_epoch: float = 0.0

    def bind_workspace(self, path) -> None: ...      # baseline = current usage; for mkdtemp-after-reserve
    def sample(self) -> int | None: ...              # peak = max(peak, usage - baseline); never raises
    def observe(self, observed_bytes: int) -> None: ...  # external measurement (deploy)
    def release(self) -> None:                       # unlink entry; final sample; record_footprint_sample(outcome="completed")
    def __exit__(self, exc_type, exc, tb) -> None:   # outcome "failed" when exc_type is not None
    def receipt(self) -> dict:                       # adds "footprint_basis", "footprint_sample_count", "workload"
```

Rules:
- A sample is recorded only when a workspace was bound or `observe` was called.
- Recording failures are swallowed.
- `release()` stays idempotent.
- `disk_headroom()` computes `refused_roles` from `effective_footprint_bytes`, and adds
  `"footprints": {role: {"bytes", "basis", "sample_count"}}`.

- [ ] **Step 1: failing tests**

```python
def test_release_records_the_workspace_peak_in_unique_inodes(tmp_path):
    ledger, work = tmp_path / "ledger", tmp_path / "work" / "job-1"
    work.mkdir(parents=True)
    (work / "preexisting.bin").write_bytes(b"p" * 8192)  # part of the baseline
    reservation = reserve_control_plane_disk(
        "launch_activation", target_root=tmp_path, reservation_root=ledger, workspace=work,
        disk_usage=lambda _p: Usage(100 * GIB, 10 * GIB, 90 * GIB), now=lambda: 100.0,
        pid_alive=lambda _pid: True)
    (work / "new.bin").write_bytes(b"n" * 200_000)
    os.link(work / "new.bin", work / "new-link.bin")        # counted once
    reservation.release()
    rows = [json.loads(line) for line in (ledger / "history" / "launch_activation.jsonl").read_text().splitlines()]
    assert rows[-1]["outcome"] == "completed"
    assert 200_000 <= rows[-1]["observed_bytes"] < 200_000 + 64 * 1024


def test_context_manager_marks_a_failed_run(tmp_path):
    ledger, work = tmp_path / "ledger", tmp_path / "job"
    with pytest.raises(RuntimeError):
        with reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
                                        workspace=work, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB),
                                        now=lambda: 1.0, pid_alive=lambda _pid: True):
            raise RuntimeError("boom")
    rows = (ledger / "history" / "launch_activation.jsonl").read_text().splitlines()
    assert json.loads(rows[-1])["outcome"] == "failed"


def test_receipt_names_the_basis_and_sample_count(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 10)
    reservation = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0, pid_alive=lambda _pid: True)
    receipt = reservation.receipt()
    assert receipt["expected_bytes"] == 125 * MIB
    assert receipt["footprint_basis"] == "measured_p95" and receipt["footprint_sample_count"] == 10
    exact = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        expected_bytes=GIB, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0,
        pid_alive=lambda _pid: True)
    assert exact.receipt()["footprint_basis"] == "caller_exact"


def test_measurement_failure_never_breaks_release(tmp_path, monkeypatch):
    ledger, work = tmp_path / "ledger", tmp_path / "job"
    reservation = reserve_control_plane_disk("launch_activation", target_root=tmp_path, reservation_root=ledger,
        workspace=work, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 1.0,
        pid_alive=lambda _pid: True)
    monkeypatch.setattr(disk_budget, "record_footprint_sample", lambda **_k: (_ for _ in ()).throw(OSError("full")))
    reservation.release()
    assert not reservation.path.exists()


def test_long_roles_outlive_the_default_ttl(tmp_path):
    ledger = tmp_path / "ledger"
    reservation = reserve_control_plane_disk("cpu_prestage", target_root=tmp_path, reservation_root=ledger,
        expected_bytes=GIB, disk_usage=lambda _p: Usage(100 * GIB, 0, 90 * GIB), now=lambda: 0.0,
        pid_alive=lambda _pid: True)
    entry = json.loads(reservation.path.read_text())
    assert entry["expires_at_epoch"] == 12 * 3600


def test_headroom_refuses_roles_by_their_measured_footprint(tmp_path):
    ledger = tmp_path / "ledger"
    _samples(ledger, "launch_activation", [100 * MIB] * 10)
    usage = lambda _p: Usage(100 * GIB, 0, 8 * GIB + GIB)  # 1 GiB above the 8 GiB floor
    headroom = disk_headroom(target_root=tmp_path, reservation_root=ledger, disk_usage=usage,
                             now=lambda: 1.0, pid_alive=lambda _pid: True)
    assert "launch_activation" not in headroom["refused_roles"]
    assert "launch_preparation" in headroom["refused_roles"]
    assert headroom["footprints"]["launch_activation"]["basis"] == "measured_p95"
```

Before writing the implementation, update `test_env_footprint_override…` (currently at
`tests/test_control_plane_disk_budget.py:172-190`) only if its assertion changes. The env
override must still set the *declared* ceiling, and with no history the declared value is what
gets reserved.

- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** the whole
  `tests/test_control_plane_disk_budget.py` passes. **Step 5:** commit "Record each reservation's real footprint when it is released".

## Task 1.4 — one admission computation everywhere

**Files:** `control_plane_capacity_controller.py`, `task_evaluation_production_chain_preflight.py`, tests.

- `measure_mount`:
  - `floor = disk_budget.floor_bytes(usage.total)`;
  - `reserved, live = disk_budget.live_reservations(ledger, device=os.stat(mount).st_dev, now=observed)`
    (pid liveness via the default `_pid_alive`; injectable for tests);
  - each `CHAIN_ROLES` footprint via `measured_footprint(role, reservation_root=...)`;
  - add `"footprints": {role: {"bytes", "basis", "sample_count"}}`;
  - compute `free_needed_for_whole_chain_bytes` from the same footprints.
- `whole_chain_admission(mount, *, reservation_root=..., now=None, disk_usage=shutil.disk_usage)`:
  - `required = sum(footprint bytes)`;
  - add `"footprints"` and `"required_workspace_basis"`: `"measured_p95"` if every role is
    measured, `"declared_default"` if none is, otherwise `"mixed"`.
- `live_reserved_bytes` stays exported for compatibility but delegates to
  `disk_budget.live_reservations` for the mount's device.
- Preflight `disk_admission_check`: replace the hard-coded `footprints` dict and floor with
  `measured_footprint` / `floor_bytes` over `CHAIN_ROLES`, and the reservation loop with
  `live_reservations`.

- [ ] **Step 1: failing tests** (in `tests/test_control_plane_capacity_controller.py`):

```python
def test_measured_p95_admits_a_chain_the_constants_refuse(tmp_path):
    ledger = tmp_path / "ledger"
    for role in cap.CHAIN_ROLES:
        for _ in range(10):
            disk_budget.record_footprint_sample(reservation_root=ledger, role=role,
                observed_bytes=400 * MIB, reserved_bytes=2 * GIB, now=lambda: 1.0)
    usage = lambda _p: _usage(free_gib=8.0 + 5.0)            # 5 GiB above the 8 GiB floor
    admitted = cap.whole_chain_admission(tmp_path, reservation_root=ledger, now=1.0, disk_usage=usage)
    assert admitted["status"] == "admitted"
    assert admitted["required_workspace_bytes"] == 5 * 500 * MIB
    assert admitted["required_workspace_basis"] == "measured_p95"
    empty = tmp_path / "empty-ledger"
    refused = cap.whole_chain_admission(tmp_path, reservation_root=empty, now=1.0, disk_usage=usage)
    assert refused["status"] == "waiting_for_capacity"
    assert refused["required_workspace_bytes"] == 10 * GIB
    assert refused["required_workspace_basis"] == "declared_default"


def test_measure_mount_honors_the_floor_override_and_ignores_other_devices(tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_FLOOR_BYTES", str(4 * GIB))
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / "other.json").write_text(json.dumps({"device": -5, "pid": os.getpid(), "expected_bytes": 50 * GIB,
                                                   "expires_at_epoch": 1e12}))
    row = cap.measure_mount(tmp_path, reservation_root=ledger, disk_usage=lambda _p: _usage(free_gib=10.0), now=1.0)
    assert row["floor_bytes"] == max(4 * GIB, int(154 * GIB * 0.05))
    assert row["reserved_bytes"] == 0
```

Update the existing test that pins `free_needed_for_one_role_bytes == 10 * GIB`, and the
whole-chain test that patches `cap.measure_mount` (at 196-209), to the new keyword
`disk_usage`. Keep their numeric intent: with no history the chain needs 10 GiB. The
`_reservation` helper writes entries with no pid or device. Give it `device` = the tmp
mount's `st_dev` and `pid` = `os.getpid()`, so these entries stay live under the new
filter.

- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** pass these test files:
  `tests/test_control_plane_capacity_controller.py`, `tests/test_task_evaluation_scene_progression.py`,
  `tests/test_task_evaluation_scene_capacity_recovery.py`, `tests/test_scene_capacity_recovery_integration.py`,
  `tests/test_scene_preparation_installation.py`, and every preflight test (`grep -l disk_admission_check tests`).
  **Step 5:** commit "Compute admission one way: shared floor, device-scoped reservations, measured footprints".

## Task 1.5 — workers bind their per-job workspace

For each call site, pass the per-job directory. Do not change `target_root` (admission still
reads the parent's filesystem). Also pass `workload=`:

| Site | Change |
|---|---|
| R1 `_reserve_preparation_disk` | `workspace=Path(input_root) / preparation_id`, `workload="prepared_references"` (thread `preparation_id` in) |
| R2 compile | `workspace=outputs / envelope["compilation_id"]`, `workload="compiled_episode"` |
| R3 activation | `workspace=activation_base / request["activation_id"]`, `workload="launch_activation"` |
| R4 partial astra (nested in R3) | **no workspace** (R3 measures the same tree) |
| R5 canary | `workspace=outputs / activation_id`, `workload="policy_canary"` |
| R6 scene attempt | `workspace=output_root`, `workload="scene_preparation_attempt"` |
| R7 public bootstrap | `workspace=<factory_output_root>/<intent_id>/public-source`, `workload="public_scene_bootstrap"` |
| R8 cpu prestage | `workspace=work`, `workload="cpu_prestage"` |
| R9 semantic pretraining | `workspace=LOGICAL_ROOT / key`, `workload="semantic_pretraining"` |
| R10 stage replay | after `mkdtemp`: `reservation.bind_workspace(replay_dir)` |

Tests that replace `reserve_control_plane_disk` with fakes must keep passing. Where a fake
signature rejects the new keywords, widen the fake to accept `**kwargs`
(`tests/test_task_evaluation_runtime_source_external_layers.py:407-414` and similar).
`tests/test_task_evaluation_launch_preparation_deploy_wiring.py:160-186` asserts that
`reserve_control_plane_disk(` appears in each worker source; keep that literal.

- [ ] **Step 1:** add one hermetic test per changed worker family that asserts the
  `workspace=` keyword reaches the reservation. Use a recording fake:
  `calls.append(kwargs); return real(*a, **kwargs)` with a roomy `disk_usage`. Put them in
  the existing worker test files next to their current disk tests.
- [ ] **Step 2–4:** implement, then run the worker test files: `tests/test_task_evaluation_launch_preparation_worker.py`,
  `tests/test_task_evaluation_episode_compilation_worker.py`, `tests/test_task_evaluation_launch_activation_worker.py`,
  `tests/test_task_evaluation_policy_canary_dispatcher.py` (whichever exist), `tests/test_task_evaluation_stage_replay.py`,
  `tests/test_task_evaluation_public_scene_bootstrap.py`, `tests/test_task_evaluation_runtime_source_external_layers.py`,
  and the cpu-prestage / artifixer tests (`grep -l "cpu_prestage\|prepare_semantics_before_gpu" tests`).
- [ ] **Step 5:** commit "Measure each job's own workspace, not the shared parent it reserves against".

## Task 1.6 — deploy reserves the release it will stage

**Files:** `scripts/deploy_control_plane_commit.py`, `tests/test_deploy_control_plane_commit.py`.

- `_release_footprint_estimate(source: Path, commit: str) -> dict`:
  - run `git -C <source> ls-tree -r -l --full-tree <commit>` (fixed argv via `_git`) and sum
    the blob sizes (column 4; `-` for submodules is 0);
  - `bytes = ceil(tree_bytes * 1.25) + 4096 * file_count + 256 MiB` (block rounding plus the
    index and runtime trees);
  - return `{"bytes", "basis": "git_tree_estimate", "tree_bytes", "file_count"}`;
  - if git fails, return `{"basis": "measured_p95"|"declared_default", ...}` from
    `measured_footprint("control_plane_deploy")`.
- Reserve with `expected_bytes=estimate["bytes"]` and `workload="control_plane_release"`.
  Put `estimate` into the receipt as `disk_reservation_estimate`.
- After `_mark_stage("release_staged")`, measure what the deploy actually created:
  - if `staged_release["created_release_checkout"]`, measure `tree_usage(release_path).allocated_bytes`;
  - after runtime provisioning, add `tree_usage` of each `system-runtimes/<component>/<commit>` created by this deploy;
  - then call `disk_reservation.observe(total)` when the reservation exists.
- `_install_disk_reservation_runtime_prerequisites`: also install `<root>/history`
  (`root:<service group>`, `0o2770`, same repair and readback loop) and report it in `installed`.
- The fake `Reservation` at `tests/test_deploy_control_plane_commit.py:1522-1530` gains `observe(self, _bytes): pass`.

- [ ] **Step 1: failing tests**

```python
def test_deploy_reserves_a_git_tree_estimate_not_a_flat_two_gib(tmp_path):
    source = _git_repo_with_commit(tmp_path)          # reuse the file's existing repo helper
    estimate = deploy._release_footprint_estimate(source, _head(source))
    assert estimate["basis"] == "git_tree_estimate"
    assert estimate["bytes"] >= estimate["tree_bytes"]
    assert estimate["bytes"] < 2 * 1024**3


def test_deploy_installs_the_footprint_history_directory(tmp_path):
    ...  # call _install_disk_reservation_runtime_prerequisites with fake stat/chown, assert history/ at 2770
```

(Use the helpers this test file already has for building a source repository and faking
ownership, around the existing `_install_disk_reservation_runtime_prerequisites` tests.)

- [ ] **Step 2–4:** implement; run `tests/test_deploy_control_plane_commit.py`.
- [ ] **Step 5:** commit "Reserve what a release actually costs to stage".

## Task 1.7 — docs

- `docs/CONTROL_PLANE_STORAGE.md`, "Admission gate":
  - replace the fixed-footprint table's "Reserves" column with "declared ceiling";
  - add a paragraph on measured footprints (sample history path, p95 × 1.25, clamp, basis in
    receipts, per-role TTLs);
  - document `whole_chain_admission`'s `required_workspace_basis`.
- Commit "Document measured disk footprints".

## PR verification (record in the PR body)

- `tests/test_control_plane_disk_usage.py`, `tests/test_control_plane_disk_budget.py`: the measurement contract.
- `tests/test_control_plane_capacity_controller.py`, the scene progression and capacity-recovery tests: admission parity.
- The worker test files from Task 1.5: call sites.
- `tests/test_deploy_control_plane_commit.py`: deploy reservation and ledger installation.
- `python -m blueprint_pipeline.impacted_test_selection` output attached.
