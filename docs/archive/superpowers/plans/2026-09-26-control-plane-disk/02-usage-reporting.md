# PR 2 — Accurate usage reporting (design doc 1d)

> Read `00-index.md` first. This PR builds on PR 1 (`control_plane_disk_usage.py`).

**Goal:** a single door `status` call answers "what uses the space?". The report shows bytes per
storage class, per root and per owner, counted once per inode. It says how much of each mount's
used bytes that attribution covers, and names any unclassified root.

**Branch / base / worktree:** `claude/disk-1d-usage-reporting` from PR 1's branch,
`$WORKSPACE/BlueprintCapturePipeline-disk-1d-20260926`.

## Facts

- The capacity controller (`src/blueprint_pipeline/control_plane_capacity_controller.py`) runs every
  10 min as root (`deploy/systemd/blueprint-control-plane-capacity.service`), with
  `MemoryMax=512M` and `TimeoutStartSec=15min`.
  - It measures only `shutil.disk_usage` per mount.
  - `write_report` (269-278) creates the report directory with `mode=0o750` under `UMask=0077`,
    so on the host it is `0700`, and writes `latest.json` as `0600`.
- The door runs as `blueprint` without capabilities, so it cannot read those files.
  - Door `status` (`deploy/operator-door/operator_door/status.py:84-113`) builds sections through
    `_section`, reads JSON with `_small_json` (256 KiB cap plus credential scan), and hard-codes
    disk paths `("/", "/var/lib/blueprint", "/mnt/blueprint-work")`.
- `control_plane_storage_roots.classify_path` gives the most specific classified root. Paths
  outside the table return `None`.
- The work volume layout is `/mnt/blueprint-work/<rel>` bind-mounted at `/var/lib/blueprint/<rel>`
  (`deploy/host/mount_work_volume.sh`). `/workspace` (the CPU prestage work dir) is also
  bind-mounted onto the volume on the production host.

## Design

**Survey.** `survey_usage(mounts, *, aliases, classify, max_entries, max_seconds, clock)` in
`control_plane_disk_usage.py`:
- Walk each mount **within its own filesystem** (`du -x` semantics):
  - skip any directory whose `st_dev` differs from the mount's, and any mount point listed in
    `/proc/self/mountinfo` (if readable);
  - never follow symlinks;
  - dedupe multiply-linked inodes by `(st_dev, st_ino)`.
- **Canonical path** for attribution: apply `aliases` (default
  `{"/mnt/blueprint-work": "/var/lib/blueprint"}`) by longest prefix. So bytes on the volume are
  attributed to the path the pipeline uses.
- **Shared inodes** (`st_nlink > 1`) belong to their content store. A shared inode is attributed
  to the first name found under a `…/content-addressed/…` directory. If none of its names is in a
  store, it goes to the lexicographically smallest canonical path.
  - Keep a dict for shared inodes only: `(dev, ino) -> (attribution, bytes, is_store)`.
  - Move the bytes when a better name is seen later. Traversal order must not change the result.
  - Traverse in sorted order anyway (entries ascending by name) so reports diff cleanly.
- **Attribution** of each inode:
  - `storage_class`:
    - `classify_path(canonical).storage_class` when classified;
    - `"unclassified"` for paths under `/var/lib/blueprint`, `/opt/blueprint` or `/workspace` that
      the table does not know;
    - `"host"` for everything else (OS, journal, docker).
  - `root`: the classified root path. For `unclassified`, the first path component below
    `/var/lib/blueprint`, `/opt/blueprint` or `/workspace`. For `host`, the first two components
    (e.g. `/var/log`, `/usr/lib`).
  - `owner`, derived from the canonical path by these rules, first match wins:
    - `…/pubsub-handoffs/<bucket>/scenes/<scene>/…` → `scene:<scene>`
    - `…/system-runtimes/<component>/<sha>/…` and `…/task-evaluation-control-plane-releases/<sha>/…` → `release:<sha[:12]>`
    - `…/content-addressed/…` → `store:<root basename>`
    - `…/task-evaluation-launch-runs/<id>/…` and `…/task-evaluation-policy-canaries/<id>/…` → `run:<id>`
    - `…/task-evaluation-scene-intents/<id>/…` → `scene-intent:<id>`
    - otherwise, for classified roots → `<root basename>/<first child>`, or `<root basename>` for a file directly under the root
    - `unclassified` / `host` → the `root` value
- **Budget.** Stop after `max_entries` (default 3,000,000) or `max_seconds` (default 240) and mark
  `status: "truncated"`. This never raises. Unreadable entries are counted.
- **Output.** Schema `control_plane_disk_usage_survey.v1`:

```json
{
  "schema_version": "control_plane_disk_usage_survey.v1",
  "status": "complete",
  "observed_at_epoch": 1790000000.0,
  "duration_seconds": 41.2,
  "entries_visited": 812345,
  "unreadable": 0,
  "mounts": [
    {"mount": "/", "used_bytes": 156000000000, "surveyed_bytes": 151000000000,
     "classified_bytes": 139000000000, "attributed_fraction": 0.968}
  ],
  "by_class": [{"storage_class": "cache", "allocated_bytes": 1, "apparent_bytes": 1, "files": 1}],
  "top_roots": [{"root": "/var/lib/blueprint/task-evaluation-inputs/prepared-references", "storage_class": "cache", "allocated_bytes": 1}],
  "top_owners": [{"owner": "scene:site-capture-…", "root": "/var/lib/blueprint/pubsub-handoffs", "storage_class": "work", "allocated_bytes": 1}],
  "unclassified_roots": [{"root": "/var/lib/blueprint/something-new", "allocated_bytes": 1}],
  "hardlinks": {"shared_inodes": 1, "shared_bytes": 1, "duplicate_names_skipped": 1},
  "survey_digest": "sha256:…"
}
```

- `attributed_fraction = surveyed_bytes / used_bytes`, where
  `used_bytes = (f_blocks - f_bfree) * f_frsize` from `os.statvfs`, capped at 1.0. Everything
  surveyed has a class and an owner by construction; the fraction measures coverage of the
  mount.
- `top_roots` and `top_owners` hold 10 rows each, sorted by bytes descending then name.
  `by_class` holds all classes.

**Cadence.** The controller surveys at most every `BLUEPRINT_CAPACITY_SURVEY_INTERVAL_SECONDS`
(default 3600):
- It checks the age of `capacity/usage-latest.json` and re-surveys when stale or when `--survey` is
  passed. The survey mounts are the controller's `mounts` plus `/` if it is not already listed.
- It writes `capacity/usage-latest.json` (0644), and embeds a compact `usage` projection in
  `latest.json`: `observed_at_epoch`, `age_seconds`, `status`, `mounts`, `by_class`, `top_roots`,
  `top_owners`, `unclassified_roots`.
- It adds alerts:
  - `usage_unclassified_root` (warning) for each unclassified root over 1 GiB;
  - `usage_attribution_low` (warning) when any mount's `attributed_fraction < 0.9`.

**Door-readable summary.** After each tick, `write_report` also writes `capacity/summary.json`
(0644) and chmods the capacity directory to 0755. The summary is secret-free and has these fields:
- `schema_version: "control_plane_capacity_summary.v1"`, `observed_at_epoch`, `level`, `alerts`;
- `mounts`, projected to mount, total/free/used_fraction, floor, reserved, available,
  refused_roles, forecast and level;
- `usage`, the projection above;
- `volume_resize` (status and reason only).

It has no project spend, no provider funding and no URLs, and must stay under 128 KiB (truncate
lists if needed).

**Door status.**
- Add a `capacity` section reading `summary.json` with `_small_json`. The path comes from a new
  `DoorConfig.capacity_summary` field (default
  `/var/lib/blueprint/pipeline-control-plane/capacity/summary.json`, overridable in `door.json`).
- Project named keys only.
- Keep the existing `disk` section.
- Client: `scripts/operator_door.py status` already prints the whole document.
- Add a `usage` subcommand that prints only `capacity.usage`, formatted as a table.

## Tasks

### Task 2.1 — survey core

**Files:** `control_plane_disk_usage.py` (extend), `tests/test_control_plane_disk_usage.py` (extend).

- [ ] **Step 1: failing tests**

```python
from blueprint_pipeline.control_plane_disk_usage import survey_usage


def _classifier(table):
    def classify(path):
        best = None
        for root, cls in table.items():
            if path == root or path.startswith(root + "/"):
                if best is None or len(root) > len(best[0]):
                    best = (root, cls)
        return None if best is None else type("Root", (), {"path": best[0], "storage_class": best[1]})()
    return classify


def test_hardlinks_count_once_and_are_attributed_to_the_first_path(tmp_path):
    base = tmp_path / "var/lib/blueprint"
    store = base / "task-evaluation-inputs/prepared-references/content-addressed/sha256"
    job = base / "task-evaluation-inputs/prepared-references/prep-1"
    store.mkdir(parents=True); job.mkdir(parents=True)
    (store / ("a" * 64)).write_bytes(b"z" * 300_000)
    os.link(store / ("a" * 64), job / "ref.bin")
    survey = survey_usage([str(base)], aliases={}, prefixes=(str(base),),
        classify=_classifier({str(base / "task-evaluation-inputs/prepared-references"): "cache"}),
        statvfs=lambda _m: os.statvfs_result((4096, 4096, 10**6, 5 * 10**5, 5 * 10**5, 0, 0, 0, 0, 255)))
    cache = next(row for row in survey["by_class"] if row["storage_class"] == "cache")
    assert 300_000 <= cache["allocated_bytes"] < 400_000
    assert survey["hardlinks"]["duplicate_names_skipped"] == 1
    owners = {row["owner"] for row in survey["top_owners"]}
    assert "store:prepared-references" in owners


def test_unclassified_roots_are_reported(tmp_path):
    base = tmp_path / "var/lib/blueprint"
    (base / "mystery").mkdir(parents=True)
    (base / "mystery" / "blob").write_bytes(b"m" * 200_000)
    survey = survey_usage([str(base)], aliases={}, prefixes=(str(base),), classify=_classifier({}),
        statvfs=lambda _m: os.statvfs_result((4096, 4096, 10**6, 5 * 10**5, 5 * 10**5, 0, 0, 0, 0, 255)))
    assert survey["unclassified_roots"][0]["root"] == str(base / "mystery")


def test_scene_workspaces_are_owned_by_their_scene(tmp_path):
    base = tmp_path / "var/lib/blueprint"
    scene = base / "pubsub-handoffs/bucket/scenes/site-capture-1/captures/c1/raw"
    scene.mkdir(parents=True)
    (scene / "video.mov").write_bytes(b"v" * 500_000)
    survey = survey_usage([str(base)], aliases={}, prefixes=(str(base),),
        classify=_classifier({str(base / "pubsub-handoffs"): "work"}),
        statvfs=lambda _m: os.statvfs_result((4096, 4096, 10**6, 5 * 10**5, 5 * 10**5, 0, 0, 0, 0, 255)))
    assert survey["top_owners"][0]["owner"] == "scene:site-capture-1"


def test_volume_paths_are_attributed_through_aliases(tmp_path):
    volume = tmp_path / "mnt/blueprint-work"
    (volume / "task-evaluation-launch-runs/run-9").mkdir(parents=True)
    (volume / "task-evaluation-launch-runs/run-9/episode.bin").write_bytes(b"e" * 100_000)
    canonical = tmp_path / "var/lib/blueprint"
    survey = survey_usage([str(volume)], aliases={str(volume): str(canonical)}, prefixes=(str(canonical),),
        classify=_classifier({str(canonical / "task-evaluation-launch-runs"): "evidence_cold"}),
        statvfs=lambda _m: os.statvfs_result((4096, 4096, 10**6, 5 * 10**5, 5 * 10**5, 0, 0, 0, 0, 255)))
    assert survey["top_owners"][0]["owner"] == "run:run-9"
    assert survey["top_owners"][0]["storage_class"] == "evidence_cold"


def test_the_budget_truncates_instead_of_running_forever(tmp_path):
    base = tmp_path / "var/lib/blueprint"
    for index in range(50):
        (base / f"d{index}").mkdir(parents=True)
    survey = survey_usage([str(base)], aliases={}, prefixes=(str(base),), classify=_classifier({}),
        max_entries=10, statvfs=lambda _m: os.statvfs_result((4096, 4096, 10**6, 5 * 10**5, 5 * 10**5, 0, 0, 0, 0, 255)))
    assert survey["status"] == "truncated"


def test_survey_attributes_every_byte_to_a_class_and_owner(tmp_path):
    base = tmp_path / "var/lib/blueprint"
    (base / "pubsub-handoffs/b/scenes/s/captures/c").mkdir(parents=True)
    (base / "pubsub-handoffs/b/scenes/s/captures/c/f").write_bytes(b"x" * 100_000)
    survey = survey_usage([str(base)], aliases={}, prefixes=(str(base),),
        classify=_classifier({str(base / "pubsub-handoffs"): "work"}),
        statvfs=lambda _m: os.statvfs_result((4096, 4096, 10**6, 10**6 - 30, 10**6 - 30, 0, 0, 0, 0, 255)))
    total = sum(row["allocated_bytes"] for row in survey["by_class"])
    assert total == survey["mounts"][0]["surveyed_bytes"]
    assert all(row["owner"] for row in survey["top_owners"])
```

`prefixes` are the Blueprint prefixes whose unknown children count as `unclassified`
(production default: `("/var/lib/blueprint", "/opt/blueprint", "/workspace")`). `classify`
defaults to `control_plane_storage_roots.classify_path` and `statvfs` to `os.statvfs`.

- [ ] **Step 2–4:** implement until green (also keep the PR 1 `tree_usage` tests green).
- [ ] **Step 5:** commit "Attribute every surveyed byte to a storage class and an owner, once per inode".

### Task 2.2 — controller cadence, summary file, alerts

**Files:** `control_plane_capacity_controller.py`, `tests/test_control_plane_capacity_controller.py`.

- `run_controller(..., survey: Callable[..., dict] | None = None, survey_interval_seconds=3600, force_survey=False)`:
  - load `usage-latest.json`; if it is missing, older than the interval or `force_survey` is set,
    run `survey` (default `survey_usage` with production aliases and prefixes);
  - write `usage-latest.json` atomically and chmod it 0644;
  - embed the projection and add the alerts.
- `write_report`: after writing `latest.json`, write `summary.json` atomically (0644) and
  `os.chmod(report_root, 0o755)`.
- CLI: `--survey` forces a survey.

- [ ] **Step 1: failing tests**

```python
def test_summary_is_door_readable_and_secret_free(tmp_path):
    report = cap.run_controller(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
        reservation_root=tmp_path / "ledger", webhook_url="", volume=None, ack="", token="",
        disk_usage=lambda _p: _usage(free_gib=100.0), now=1_000.0,
        survey=lambda **_k: {"schema_version": "control_plane_disk_usage_survey.v1", "status": "complete",
                             "observed_at_epoch": 1_000.0, "mounts": [], "by_class": [], "top_roots": [],
                             "top_owners": [], "unclassified_roots": [{"root": "/var/lib/blueprint/x",
                                                                       "allocated_bytes": 2 * GIB}]})
    summary_path = tmp_path / "capacity" / "summary.json"
    assert oct(summary_path.stat().st_mode & 0o777) == "0o644"
    assert oct((tmp_path / "capacity").stat().st_mode & 0o777) == "0o755"
    summary = json.loads(summary_path.read_text())
    assert summary["schema_version"] == "control_plane_capacity_summary.v1"
    assert "project_spend" not in summary and "provider_funding" not in summary
    assert {"mount": None, "code": "usage_unclassified_root", "root": "/var/lib/blueprint/x"} in [
        {k: a.get(k) for k in ("mount", "code", "root")} for a in report["alerts"]]


def test_survey_runs_at_most_hourly(tmp_path):
    calls = []
    def survey(**_k):
        calls.append(1)
        return {"schema_version": "control_plane_disk_usage_survey.v1", "status": "complete",
                "observed_at_epoch": 1_000.0, "mounts": [], "by_class": [], "top_roots": [],
                "top_owners": [], "unclassified_roots": []}
    for now in (1_000.0, 1_600.0, 4_700.0):
        cap.run_controller(mounts=[str(tmp_path)], report_root=tmp_path / "capacity",
            reservation_root=tmp_path / "ledger", webhook_url="", volume=None, ack="", token="",
            disk_usage=lambda _p: _usage(free_gib=100.0), now=now, survey=survey)
    assert len(calls) == 2
```

`run_controller` calls `refresh_configured_scene_project_spend()`; existing tests show how to
isolate it (monkeypatch it to return `None`). Reuse that in both tests.

- [ ] **Step 2–4:** implement; run `tests/test_control_plane_capacity_controller.py` and `tests/test_deploy_systemd_contract.py` (capacity unit contract).
- [ ] **Step 5:** commit "Publish a door-readable capacity summary with an hourly usage survey".

### Task 2.3 — door status and client

**Files:**
- `deploy/operator-door/operator_door/config.py`: `capacity_summary: str` default and `door.json` key.
- `deploy/operator-door/operator_door/status.py`: `capacity` section.
- `scripts/operator_door.py`: `usage` subcommand.
- `tests/test_operator_door_status.py`, `tests/test_operator_door_config.py`, `tests/test_operator_door_client.py`.
- `docs/OPERATOR_DOOR.md`: the status row and usage example.

- [ ] **Step 1: failing test** (extend the `host_tree` fixture with a
  `capacity/summary.json` and point `DoorConfig(capacity_summary=…)` at it):

```python
def test_status_reports_capacity_usage(host_tree):
    status = build_status(host_tree.config, host_tree.host, caller=CALLER)
    assert status["capacity"]["level"] == "warning"
    assert status["capacity"]["usage"]["top_owners"][0]["owner"] == "scene:site-capture-1"


def test_status_capacity_section_fails_soft(host_tree):
    host_tree.summary.unlink()
    status = build_status(host_tree.config, host_tree.host, caller=CALLER)
    assert status["capacity"] == {"error": "capacity_unavailable:FileNotFoundError"}
```

Match `CALLER` and the fixture to what `tests/test_operator_door_status.py` already uses.

- [ ] **Step 2–4:** implement; run every `tests/test_operator_door_*.py`.
- [ ] **Step 5:** commit "Show capacity and usage attribution in door status".

### Task 2.4 — docs

- `docs/CONTROL_PLANE_STORAGE.md`: new section "Usage attribution". Cover what the survey
  counts, the owner rules, cadence, the files and their modes, and how to read
  `operator_door.py usage`.
- Also correct the stale reclaim-timer facts the research found. The GC runs as `root`, hourly
  (`OnUnitInactiveSec=1h`), derived age 3600 s, hot window 172800 s. The GC module docstring says
  "Three reclaim steps" but there are seven phases; fix that docstring too.
- Commit "Document usage attribution and correct the reclaim timer facts".

## PR verification

`tests/test_control_plane_disk_usage.py`, `tests/test_control_plane_capacity_controller.py`,
`tests/test_operator_door_status.py`, `tests/test_operator_door_config.py`,
`tests/test_operator_door_client.py`, `tests/test_deploy_systemd_contract.py`, and the impacted
selection output.
