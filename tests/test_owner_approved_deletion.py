"""Owner-approved deletions remove exactly the approved, unchanged, unreferenced paths, behind receipts."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/owner_approved_deletion.py
#   deploy/owner-approved-deletions/2026-11-04-arena-scratch.json
#   deploy/owner-approved-deletions/held/2026-11-04-authority-ended-scenes.json
#   deploy/owner-approved-deletions/2026-11-04-controlled-policy-integration.json
#   deploy/owner-approved-deletions/2026-11-12-old-development-folders.json
#   deploy/systemd/blueprint-owner-approved-deletion-plan.service
#   deploy/systemd/blueprint-owner-approved-deletion.service
#   tests/fixtures/owner_approved_deletion_2026_11_12_naming_files.json

from __future__ import annotations

import functools
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from blueprint_pipeline import owner_approved_deletion as oad
from blueprint_pipeline.control_plane_storage_pins import write_storage_pin

REPO_ROOT = Path(__file__).resolve().parents[1]
DAY = 86400
APPROVED_AT = "2026-11-04T15:00:00Z"
APPROVED_EPOCH = 1793804400.0  # 2026-11-04T15:00:00Z


@pytest.fixture(autouse=True)
def isolated_disk_ledger(tmp_path, monkeypatch):
    # The scene tests stage a capture through the real listener, which reserves disk.
    from blueprint_pipeline import pubsub_handoff_disk_admission as disk_admission
    from blueprint_pipeline.control_plane_disk_budget import reserve_control_plane_disk

    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT", str(tmp_path / "disk-reservations"))
    monkeypatch.delenv("BLUEPRINT_WEBSITE_SCENE_BINDING_ROOT", raising=False)
    monkeypatch.setattr(disk_admission, "reserve_control_plane_disk", functools.partial(
        reserve_control_plane_disk, disk_usage=lambda _: SimpleNamespace(total=100 * 1024**3, free=80 * 1024**3)))


class Host:
    """A small host: the allowlisted parents, pins, queues, units, an empty process table and reports."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.inputs = root / "task-evaluation-inputs"
        self.handoffs = root / "pubsub-handoffs"
        self.control = root / "pipeline-control-plane"
        for path in (self.inputs, self.handoffs, self.control / "storage-pins", self.control / "queue" / "pending",
                     self.control / "queue" / "processing", self.control / "task-evaluation-scene-intents",
                     self.control / "website-source-bindings", root / "units", root / "proc", root / "manifests",
                     root / "etc-blueprint", self.inputs / "public-source-bindings",
                     *(root / "release" / name for name in oad.RELEASE_SUBDIRECTORIES)):
            path.mkdir(parents=True, exist_ok=True)
        self.etc = root / "etc-blueprint"
        self.intents = self.control / "task-evaluation-scene-intents"
        self.bindings = self.inputs / "public-source-bindings"
        self.catalog = self.control / "task-evaluation-launch-profile-catalog.json"
        self.catalog.write_text(json.dumps({"profiles": []}), encoding="utf-8")
        self.reports = root / "reports"
        self.settings = oad.Settings(pins_root=self.control / "storage-pins", queue_roots=[self.control / "queue"],
                                     intent_root=self.intents, inputs_root=self.inputs, handoff_root=self.handoffs,
                                     unit_directories=[root / "units"], process_root=root / "proc",
                                     config_root=self.etc, binding_root=self.bindings,
                                     launch_profile_catalog=self.catalog,
                                     registry_roots=[self.control / "website-source-bindings",
                                                     self.control / "beta-policy-screening"],
                                     release_roots=[root / "release"], door_roots=[root / "door"])

    def write(self, path: Path, document) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document if isinstance(document, str) else json.dumps(document), encoding="utf-8")
        return path

    def folder(self, name: str, files: dict[str, bytes] | None = None, *, mtime: int = 1787000000) -> Path:
        folder = self.inputs / name
        for relative, data in (files or {"job/result.json": b"{}", "inputs.bin": b"x" * 4096}).items():
            path = folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for directory, names, leaves in os.walk(folder, topdown=False):
            for name_ in leaves + names:
                os.utime(Path(directory) / name_, (mtime, mtime), follow_symlinks=False)
            os.utime(directory, (mtime, mtime))
        return folder

    def folder_entry(self, folder: Path, **overrides) -> dict:
        files = [path for path in folder.rglob("*") if path.is_file() and not path.is_symlink()]
        newest = max(int(os.lstat(path).st_mtime) for path in [folder, *folder.rglob("*")])
        return {"class": oad.FOLDER, "path": str(folder), "newest_mtime_epoch": newest,
                "apparent_bytes": sum(path.stat().st_size for path in files), "file_count": len(files),
                "door_unlisted_entries": 0, **overrides}

    def manifest(self, entries: list[dict], *, manifest_id: str = "test-manifest", **overrides) -> Path:
        document = {"schema_version": oad.MANIFEST_SCHEMA, "manifest_id": manifest_id, "approved_by": "owner",
                    "approval_evidence": 'Owner: "yes delete"', "approved_at": APPROVED_AT,
                    "expires_at": "2026-11-18T15:00:00Z", "entries": entries, **overrides}
        path = self.root / "manifests" / f"{manifest_id}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def plan(self, manifest: Path, at: float = APPROVED_EPOCH + 60) -> dict:
        return oad.plan_manifest(manifest, report_root=self.reports, settings=self.settings, now=lambda: at)

    def apply(self, manifest: Path, at: float = APPROVED_EPOCH + 120) -> dict:
        return oad.apply_manifest(manifest, report_root=self.reports, settings=self.settings,
                                  ack=oad.APPLY_ACK, now=lambda: at)


@pytest.fixture
def host(tmp_path):
    return Host(tmp_path / "host")


def _statuses(plan: dict) -> dict[str, tuple[str, list[str]]]:
    return {Path(row["path"]).name: (row["status"], row["reasons"]) for row in plan["entries"]}


def test_plan_deletes_nothing_and_apply_deletes_with_a_receipt_per_path(host):
    first, second = host.folder("arena-launch-r1"), host.folder("arena-controls-c2")
    keep = host.folder("arena-launch-r10")
    manifest = host.manifest([host.folder_entry(first), host.folder_entry(second)])

    plan = host.plan(manifest)

    assert plan["mutations"] == 0 and plan["eligible_count"] == 2
    assert first.is_dir() and second.is_dir()
    published = json.loads((host.reports / "test-manifest" / "plan.json").read_text())
    assert published["plan_digest"] == plan["plan_digest"]

    result = host.apply(manifest)

    assert result["outcome_counts"] == {"deleted": 2}
    assert not first.exists() and not second.exists() and keep.is_dir()
    assert not any(path.name.startswith(oad.ASIDE_PREFIX) for path in host.inputs.iterdir())
    receipts = sorted((host.reports / "test-manifest" / "receipts").glob("*.json"))
    assert len(receipts) == 2
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == "deleted" and receipt["approval_evidence"] == 'Owner: "yes delete"'
    assert receipt["plan_digest"] == plan["plan_digest"] and receipt["identity"]["file_count"] == 2
    assert oct(receipts[0].stat().st_mode & 0o777) == "0o644"

    again = host.apply(manifest)
    assert again["outcome_counts"] == {"already_deleted": 2}


def test_a_folder_changed_since_approval_is_kept(host):
    folder = host.folder("arena-launch-r2")
    entry = host.folder_entry(folder)
    (folder / "job" / "late.json").write_text("{}")  # written after the owner approved
    plan = host.plan(host.manifest([entry]))

    assert _statuses(plan)["arena-launch-r2"] == ("skipped", ["changed_since_approval"])


def test_unlisted_door_names_allow_only_larger_trees(host):
    folder = host.folder("controlled-policy-integration-20260928")
    entry = host.folder_entry(folder)
    smaller = dict(entry, apparent_bytes=entry["apparent_bytes"] - 10, file_count=1, door_unlisted_entries=1)
    larger = dict(entry, apparent_bytes=entry["apparent_bytes"] + 10, door_unlisted_entries=1)

    assert host.plan(host.manifest([smaller]))["eligible_count"] == 1
    assert _statuses(host.plan(host.manifest([larger])))[folder.name][1] == ["changed_since_approval"]


def test_a_symlink_is_never_followed_or_deleted_through(host, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.json").write_text("{}")
    link = host.inputs / "arena-launch-sweep-clearance-r1"
    link.symlink_to(outside)
    target_entry = {"class": oad.FOLDER, "path": str(link), "newest_mtime_epoch": 0, "apparent_bytes": 0,
                    "file_count": 0, "door_unlisted_entries": 0}
    folder = host.folder("arena-rim-grasp-r1")
    (folder / "escape").symlink_to(outside)
    manifest = host.manifest([target_entry, host.folder_entry(folder)])

    plan = host.plan(manifest)
    assert _statuses(plan)["arena-launch-sweep-clearance-r1"] == ("skipped", ["not_a_directory"])
    host.apply(manifest)

    assert link.is_symlink() and (outside / "precious.json").is_file()
    assert not folder.exists()


def test_the_target_of_a_sibling_link_is_kept(host):
    folder = host.folder("arena-rim-grasp-sweep-clearance-12b73159-r1")
    (host.inputs / "arena-launch-sweep-clearance-r1").symlink_to(folder.name)

    plan = host.plan(host.manifest([host.folder_entry(folder)]))

    assert _statuses(plan)[folder.name] == ("skipped", ["symlink_target"])


def test_pins_queues_units_and_processes_keep_a_folder(host):
    pinned, queued, unit, used = (host.folder(name) for name in (
        "arena-launch-r3", "arena-launch-r4", "arena-runtime-preflight-79ef9a8f", "arena-controls-c1"))
    write_storage_pin(pins_root=host.control / "storage-pins", kind="preparation", owner_id="pin-1",
                      paths=[pinned / "job"], ttl_seconds=DAY, now=lambda: APPROVED_EPOCH)
    (host.control / "queue" / "pending" / "row.json").write_text(json.dumps({"input": str(queued)}))
    (host.root / "units" / "blueprint-preflight.service").write_text(f"[Service]\nExecStart=/bin/true {unit}\n")
    process = host.root / "proc" / "4242"
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python\0-m\0x\0")
    (process / "environ").write_bytes(b"")
    (process / "cwd").symlink_to(used / "job")
    manifest = host.manifest([host.folder_entry(path) for path in (pinned, queued, unit, used)])

    statuses = _statuses(host.plan(manifest))

    assert statuses["arena-launch-r3"] == ("skipped", ["pinned"])
    assert statuses["arena-launch-r4"] == ("skipped", ["queue_referenced"])
    assert statuses[unit.name] == ("skipped", ["unit_referenced:blueprint-preflight.service"])
    assert statuses["arena-controls-c1"] == ("skipped", ["in_use"])
    assert host.apply(manifest)["outcome_counts"] == {"not_eligible_in_plan": 4}
    assert all(path.is_dir() for path in (pinned, queued, unit, used))


def test_a_similar_name_is_not_a_reference(host):
    folder = host.folder("arena-controls-c2")
    (host.control / "queue" / "pending" / "row.json").write_text(json.dumps({"input": "arena-controls-c2-15hz"}))

    assert host.plan(host.manifest([host.folder_entry(folder)]))["eligible_count"] == 1


@pytest.mark.parametrize("path", [
    "/var/lib/blueprint/task-evaluation-inputs/lanes/arena/x",
    "/var/lib/blueprint/task-evaluation-inputs/../pipeline-control-plane",
    "/var/lib/blueprint/pipeline-control-plane/storage-pins",
    "/var/lib/blueprint/task-evaluation-inputs/.owner-approved-deletion-abc",
])
def test_paths_outside_the_allowlist_are_refused(tmp_path, path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({
        "schema_version": oad.MANIFEST_SCHEMA, "manifest_id": "m", "approved_by": "owner",
        "approval_evidence": "yes delete", "approved_at": APPROVED_AT, "expires_at": "2026-11-05T15:00:00Z",
        "entries": [{"class": oad.FOLDER, "path": path, "newest_mtime_epoch": 1, "apparent_bytes": 1,
                     "file_count": 1, "door_unlisted_entries": 0}]}))

    with pytest.raises(oad.OwnerApprovedDeletionError, match="entry_path_outside_allowlist"):
        oad.load_manifest(manifest)


def test_an_approval_longer_than_fourteen_days_or_a_misnamed_manifest_is_refused(host):
    folder = host.folder("arena-launch-r6")
    with pytest.raises(oad.OwnerApprovedDeletionError, match="approval_window_invalid"):
        host.plan(host.manifest([host.folder_entry(folder)], expires_at="2026-11-19T15:00:01Z"))
    path = host.manifest([host.folder_entry(folder)])
    renamed = path.with_name("other.json")
    path.rename(renamed)
    with pytest.raises(oad.OwnerApprovedDeletionError, match="manifest_id_invalid"):
        host.plan(renamed)


def test_apply_needs_the_ack_a_fresh_plan_of_the_same_bytes_and_an_unexpired_approval(host):
    folder = host.folder("arena-launch-r7")
    manifest = host.manifest([host.folder_entry(folder)])
    with pytest.raises(oad.OwnerApprovedDeletionError, match="apply_not_acknowledged"):
        oad.apply_manifest(manifest, report_root=host.reports, settings=host.settings, ack="yes")
    with pytest.raises(oad.OwnerApprovedDeletionError, match="plan_missing"):
        host.apply(manifest)
    host.plan(manifest)
    with pytest.raises(oad.OwnerApprovedDeletionError, match="plan_stale"):
        host.apply(manifest, at=APPROVED_EPOCH + 60 + oad.MAX_PLAN_AGE_SECONDS + 1)
    manifest.write_text(manifest.read_text().replace("yes delete", "yes, delete"))
    with pytest.raises(oad.OwnerApprovedDeletionError, match="plan_for_other_manifest"):
        host.apply(manifest)
    late = host.manifest([host.folder_entry(folder)])
    host.plan(late, at=APPROVED_EPOCH + 14 * DAY - 10)
    with pytest.raises(oad.OwnerApprovedDeletionError, match="approval_expired"):
        host.apply(late, at=APPROVED_EPOCH + 14 * DAY + 10)
    assert folder.is_dir()


def test_a_folder_changed_between_plan_and_apply_is_kept(host):
    folder = host.folder("arena-launch-r8")
    manifest = host.manifest([host.folder_entry(folder)])
    host.plan(manifest)
    (folder / "job").rename(folder / "job-moved")  # same files and bytes, but the tree changed

    result = host.apply(manifest)

    assert result["outcomes"][0]["outcome"] == "skipped" and folder.is_dir()


def test_a_run_stopped_after_the_rename_is_finished_and_one_stopped_before_it_starts_over(host, monkeypatch):
    first, second = host.folder("arena-launch-r9"), host.folder("arena-launch-r11")
    manifest = host.manifest([host.folder_entry(first), host.folder_entry(second)])
    host.plan(manifest)
    real_remove = oad._remove_aside
    monkeypatch.setattr(oad, "_remove_aside", lambda *a, **k: "remove_incomplete")
    stopped = host.apply(manifest)
    assert stopped["outcome_counts"] == {"remove_incomplete": 2}
    asides = [path for path in host.inputs.iterdir() if path.name.startswith(oad.ASIDE_PREFIX)]
    assert len(asides) == 2
    monkeypatch.setattr(oad, "_remove_aside", real_remove)
    # Simulate the second run stopping before its rename: put it back, its receipt still says deleting.
    receipt = host.reports / "test-manifest" / "receipts" / f"{oad._key(str(second))}.json"
    aside = json.loads(receipt.read_text())["aside_name"]
    (host.inputs / aside).rename(second)

    finished = host.apply(manifest)

    assert finished["outcome_counts"] == {"deleted": 2}
    assert not first.exists() and not second.exists()
    assert not any(path.name.startswith(oad.ASIDE_PREFIX) for path in host.inputs.iterdir())


# --- authority-ended website scene workspaces ---------------------------------------------------


def _authority_ended_scene(tmp_path: Path, host: Host) -> Path:
    from tests.test_website_scene_workspace_retention import _scene

    scene, _cloud = _scene(tmp_path / "staging", status="terminal_authority_ended")
    target = host.handoffs / scene.parent.parent.name / "scenes" / scene.name
    target.parent.mkdir(parents=True)
    scene.rename(target)
    return target


def _approval_around(now: float) -> dict:
    """An approval window that holds the clock a test plans at.

    The scene builder stamps a scene's ending from the real clock, and these tests plan up to eight days after it.
    With the default fixed expiry (2026-11-18) they began to fail on 2026-11-10, eight days before it, on every
    branch. The window is the longest the tool accepts, starting at the test's own clock.
    """

    def stamp(epoch: float) -> str:
        return datetime.fromtimestamp(int(epoch), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {"approved_at": stamp(now), "expires_at": stamp(int(now) + oad.MAX_APPROVAL_SECONDS)}


def _scene_entry(scene: Path) -> dict:
    captures = []
    for capture in sorted((scene / "captures").iterdir()):
        ledger = json.loads((capture / "pipeline_job_ledger.json").read_text())
        captures.append({"capture_id": capture.name, **{key: ledger[key] for key in (
            "terminal_code", "terminal_at", "terminal_payload_sha256")}})
    return {"class": oad.SCENE, "path": str(scene), "captures": captures}


def test_an_authority_ended_scene_is_deleted_only_after_the_redelivery_window(tmp_path, host):
    scene = _authority_ended_scene(tmp_path, host)
    now = time.time()
    manifest = host.manifest([_scene_entry(scene)], **_approval_around(now))

    early = host.plan(manifest, at=now + DAY)
    assert early["entries"][0]["status"] == "skipped"
    from tests.test_website_scene_workspace_retention import CAPTURE
    assert early["entries"][0]["reasons"] == [f"terminal_within_redelivery_window:{CAPTURE}"]

    later = now + 8 * DAY
    plan = host.plan(manifest, at=later)
    assert plan["entries"][0]["status"] == "eligible", plan["entries"][0]["reasons"]
    result = host.apply(manifest, at=later + 60)

    assert result["outcome_counts"] == {"deleted": 1} and not scene.exists()


def test_a_scene_whose_capture_did_not_end_by_authority_is_kept(tmp_path, host):
    from tests.test_website_scene_workspace_retention import _scene

    scene, _ = _scene(tmp_path / "staging")
    target = host.handoffs / scene.parent.parent.name / "scenes" / scene.name
    target.parent.mkdir(parents=True)
    scene.rename(target)
    from tests.test_website_scene_workspace_retention import CAPTURE
    ledger = json.loads((target / "captures" / CAPTURE / "pipeline_job_ledger.json").read_text())
    entry = {"class": oad.SCENE, "path": str(target), "captures": [{
        "capture_id": CAPTURE, "terminal_code": "consent_expired", "terminal_at": "2026-09-27T03:53:03+00:00",
        "terminal_payload_sha256": "a" * 64}]}
    assert ledger["status"] == "completed"

    now = time.time()
    plan = host.plan(host.manifest([entry], **_approval_around(now)), at=now + 8 * DAY)

    assert plan["entries"][0]["status"] == "skipped"
    assert "capture_authority_not_ended" in plan["entries"][0]["reasons"]
    assert host.apply(host.manifest([entry], **_approval_around(now)), at=now + 8 * DAY)["outcome_counts"] == {
        "not_eligible_in_plan": 1}
    assert target.is_dir()


def test_a_queued_authority_ended_scene_is_kept(tmp_path, host):
    scene = _authority_ended_scene(tmp_path, host)
    (host.control / "queue" / "processing" / "row.json").write_text(json.dumps({"scene_id": scene.name}))

    now = time.time()
    plan = host.plan(host.manifest([_scene_entry(scene)], **_approval_around(now)), at=now + 8 * DAY)

    assert "queue_referenced" in plan["entries"][0]["reasons"]


# --- the shipped manifests and units ------------------------------------------------------------


def test_the_shipped_manifests_validate_and_never_name_the_live_capture():
    manifests = oad.manifests(REPO_ROOT / "deploy" / "owner-approved-deletions")
    # The 2026-11-04 scenes manifest is held aside (deploy/owner-approved-deletions/held/README.md), unchanged.
    assert not (REPO_ROOT / "deploy/owner-approved-deletions/2026-11-04-authority-ended-scenes.json").exists()
    assert (REPO_ROOT / "deploy/owner-approved-deletions/held/2026-11-04-authority-ended-scenes.json").is_file()
    assert [path.stem for path in manifests] == ["2026-11-04-arena-scratch",
                                                 "2026-11-04-controlled-policy-integration",
                                                 "2026-11-12-old-development-folders"]
    for path in manifests:
        manifest, _ = oad.load_manifest(path)
        assert not any("447ae4f4" in json.dumps(entry) for entry in manifest["entries"])
        for entry in manifest["entries"]:
            if entry["class"] == oad.FOLDER:
                # Nothing written after 2026-09-01, except the separately approved 2026-09-28 integration folder
                # and the 2026-11-12 old development folders (nothing written since 2026-10-01).
                limit = (1790812800 if "controlled-policy" in entry["path"] or path.stem.startswith("2026-11-12")
                         else 1788307200)
                assert entry["newest_mtime_epoch"] < limit, entry["path"]


def test_a_manifest_held_aside_is_never_planned_or_applied(tmp_path, monkeypatch):
    directory = tmp_path / "owner-approved-deletions"
    (directory / "held").mkdir(parents=True)
    (directory / "active.json").write_text("{}", encoding="utf-8")
    (directory / "held" / "held-scenes.json").write_text("{}", encoding="utf-8")
    assert oad.manifests(directory) == [directory / "active.json"]
    seen = []
    monkeypatch.setattr(oad, "plan_manifest", lambda path, **_: seen.append(("plan", path)) or {
        "manifest_id": "m", "plan_digest": "d", "entry_count": 0, "eligible_count": 0,
        "eligible_apparent_bytes": 0, "skipped_counts": {}})
    monkeypatch.setattr(oad, "apply_manifest", lambda path, **_: seen.append(("apply", path)) or {
        "manifest_id": "m", "result_digest": "d", "deleted_count": 0, "deleted_apparent_bytes": 0,
        "outcome_counts": {}})
    monkeypatch.setenv(oad.QUEUE_ROOTS_ENV, str(tmp_path / "queue"))
    for action in (["plan"], ["apply", "--ack", oad.APPLY_ACK]):
        assert oad.main([*action, "--manifest-dir", str(directory), "--report-root", str(tmp_path / "reports"),
                         "--pins-root", str(tmp_path / "pins"), "--intent-root", str(tmp_path / "intents")]) == 0
    assert seen == [("plan", directory / "active.json"), ("apply", directory / "active.json")]


def test_the_units_run_by_hand_with_the_reclaim_timers_queues():
    systemd = REPO_ROOT / "deploy" / "systemd"
    gc = (systemd / "blueprint-control-plane-storage-gc.service").read_text()
    queues = next(line for line in gc.splitlines() if line.startswith("Environment=" + oad.QUEUE_ROOTS_ENV + "="))
    for name, action in (("blueprint-owner-approved-deletion-plan", " plan "),
                         ("blueprint-owner-approved-deletion", " apply --ack " + oad.APPLY_ACK + " ")):
        text = (systemd / f"{name}.service").read_text()
        assert queues + "\n" in text
        assert f"-m blueprint_pipeline.owner_approved_deletion{action}" in text
        assert "--manifest-dir /opt/blueprint/task-evaluation-control-plane/deploy/owner-approved-deletions" in text
        assert not (systemd / f"{name}.timer").exists() and not (systemd / f"{name}.path").exists()


def test_the_units_can_read_every_live_state_root():
    # ProtectSystem=strict leaves the host readable; only the provider key files are hidden, and the scan
    # never reads them. Nothing else may hide or replace a root the live-state scan needs.
    roots = [oad.DEFAULT_CONFIG_ROOT, oad.DEFAULT_INTENT_ROOT, oad.DEFAULT_PUBLIC_SOURCE_BINDING_ROOT,
             oad.DEFAULT_LAUNCH_PROFILE_CATALOG, oad.INPUTS_ROOT, oad.DEFAULT_PINS_ROOT, *oad.DEFAULT_REGISTRY_ROOTS,
             oad.DEFAULT_RELEASE_ROOT, oad.DEFAULT_DOOR_ROOT]
    for name in ("blueprint-owner-approved-deletion-plan", "blueprint-owner-approved-deletion"):
        lines = (REPO_ROOT / "deploy" / "systemd" / f"{name}.service").read_text().splitlines()
        assert "User=root" in lines and "ProtectSystem=strict" in lines
        assert "CapabilityBoundingSet=CAP_DAC_OVERRIDE CAP_SYS_PTRACE" in lines
        hidden = [line.split("=", 1)[1] for line in lines
                  if line.split("=", 1)[0] in ("InaccessiblePaths", "TemporaryFileSystem", "BindPaths",
                                               "BindReadOnlyPaths", "ProtectProc")]
        assert hidden == ["-/etc/blueprint/provider-secrets"], hidden
        assert {Path(path.lstrip("-")).name for path in hidden} <= oad.CONFIG_EXCLUDED_NAMES
        assert all(not any(str(root) == path.lstrip("-") or str(root).startswith(path.lstrip("-") + "/")
                           for path in hidden) for root in roots)


# --- live state: configuration and registries that still name a folder -------------------------


def _naming_sources(host: Host, folder: Path) -> dict[str, Path]:
    """One naming file per source the plan reads, each naming ``folder`` (or a file in it)."""

    asset = f"{folder}/franka_robotiq_2f_85_flattened.usd"
    retained = host.root / "completed-scene-preparation" / "scene-retained" / "public-source" / "binding.json"
    host.write(retained, {"references": {"installation_receipt": {"path": str(folder / "receipt.json")}}})
    return {
        "config_json": host.write(host.etc / "task-evaluation-controls-robot-catalog.json",
                                  {"bindings": [{"robot_asset_usd_path": asset}]}),
        "config_env": host.write(host.etc / "installed-source-bindings" / "scene.env", f"SOURCE_ROOT={folder}\n"),
        "configured_controls_intent": host.write(
            host.etc / "task-evaluation-configured-controls-intents" / "abc.json",
            {"status": "revoked", "artifact_inventory": {"robot_asset_usd_path": asset}}),
        "launch_profile": host.write(host.etc / "task-evaluation-launch-profiles" / "p.json", {
            "claim_ceiling": "development_only", "allocator": {"argv": ["run", f"--input={folder}/job"]}}),
        "machinery": host.write(host.etc / "task-evaluation-public-scene-machinery.json",
                                {"review_terms": f"file://{folder}/terms.md"}),
        "scene_intent": host.write(host.intents / "scene-a" / "intent.json",
                                   {"status": "ended", "source": {"path": str(folder)}}),
        "retained_binding": retained,
        "public_source_binding": host.write(host.bindings / "public-scene-1.json", {
            "references": {"publisher_intake": {"path": f"{folder}/intake.json"}}}),
        "launch_profile_catalog": host.write(host.catalog, {"profiles": [{"inputs": [str(folder)]}]}),
        "inputs_root_loose_json": host.write(host.inputs / "request.v1.json", {"segment": str(folder)}),
        "control_plane_registry": host.write(host.control / "website-source-bindings" / "b.json",
                                             {"path": str(folder)}),
    }


@pytest.mark.parametrize("source", [
    "config_json", "config_env", "configured_controls_intent", "launch_profile", "machinery", "scene_intent",
    "retained_binding", "public_source_binding", "launch_profile_catalog", "inputs_root_loose_json",
    "control_plane_registry"])
def test_a_folder_named_by_live_state_is_refused_with_the_naming_file(host, source):
    folder = host.folder("scene-841757-raw-30b10e56-20260905")
    keep = _naming_sources(host, folder)[source]
    for name, path in _naming_sources(host, folder).items():  # leave only the source under test naming it
        if name != source:
            path.write_text("[]" if path.suffix == ".json" else "\n", encoding="utf-8")
    if source == "retained_binding":
        host.write(host.intents / "scene-a" / "progression.json", {"state": {"source_binding": {"path": str(keep)}}})
    manifest = host.manifest([host.folder_entry(folder)])

    plan = host.plan(manifest)

    row = plan["entries"][0]
    assert (row["status"], row["reasons"], row["named_by"]) == ("skipped", ["named_by_live_state"], [str(keep)])
    assert plan["reference_scan"]["status"] == "complete" and plan["skipped_counts"] == {"named_by_live_state": 1}
    assert host.apply(manifest)["outcome_counts"] == {"not_eligible_in_plan": 1} and folder.is_dir()
    receipt = json.loads((host.reports / "test-manifest" / "receipts" / f"{oad._key(str(folder))}.json").read_text())
    assert (receipt["status"], receipt["reasons"], receipt["named_by"]) == (
        "skipped", ["named_by_live_state"], [str(keep)])


def test_a_link_in_the_configuration_names_its_target(host):
    folder = host.folder("native-droid-robot-isaac6-c8d72259")
    (host.etc / "current-robot").symlink_to(folder / "robot.usd")  # dangling: the file inside is not there

    row = host.plan(host.manifest([host.folder_entry(folder)]))["entries"][0]

    assert row["named_by"] == [str(host.etc / "current-robot")] and row["status"] == "skipped"


def test_named_paths_compare_whole_components(host):
    folder = host.folder("scene-84")
    host.write(host.etc / "names.json", {"near": [str(host.inputs / "scene-841757"), f"{folder}-other/x",
                                                  f"{host.inputs}/xscene-84", str(host.inputs), f"gs:/{folder}",
                                                  f"https://host{folder}"]})

    assert host.plan(host.manifest([host.folder_entry(folder)]))["eligible_count"] == 1


def test_an_unreadable_file_marks_every_entry_and_deletes_nothing(host, monkeypatch):
    first, second = host.folder("arena-launch-r1"), host.folder("arena-launch-r2")
    secret = host.write(host.etc / "installed-source-bindings" / "scene.env", "TOKEN=x\n")
    broken = host.write(host.etc / "task-evaluation-launch-profiles" / "half.json", '{"immutable_inputs": [')
    real_read = Path.read_bytes

    def read_bytes(path):
        if path == secret:
            raise PermissionError(13, "denied")
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    manifest = host.manifest([host.folder_entry(first), host.folder_entry(second)])

    plan = host.plan(manifest)

    assert plan["eligible_count"] == 0
    assert {row["status"] for row in plan["entries"]} == {"skipped"}
    assert all(row["reasons"] == ["reference_scan_incomplete"] for row in plan["entries"])
    assert plan["reference_scan"]["status"] == "incomplete"
    assert plan["reference_scan"]["unreadable"] == [{"path": str(secret), "code": "permission_denied"},
                                                    {"path": str(broken), "code": "not_json"}]
    assert host.apply(manifest)["outcome_counts"] == {"not_eligible_in_plan": 2}
    assert first.is_dir() and second.is_dir()
    receipt = json.loads((host.reports / "test-manifest" / "receipts" / f"{oad._key(str(first))}.json").read_text())
    assert receipt["reference_scan_unreadable"] == plan["reference_scan"]["unreadable"]


@pytest.mark.parametrize("missing", ["etc", "intents", "bindings", "catalog"])
def test_a_missing_root_marks_every_entry(host, missing):
    import shutil

    folder = host.folder("arena-launch-r1")
    target = getattr(host, missing)
    shutil.rmtree(target) if target.is_dir() else target.unlink()

    plan = host.plan(host.manifest([host.folder_entry(folder)]))

    assert plan["entries"][0]["reasons"] == ["reference_scan_incomplete"]
    assert plan["reference_scan"]["unreadable"] == [{"path": str(target), "code": "required_root_missing"}]


def test_an_optional_registry_may_be_absent_but_the_provider_secrets_are_never_read(host):
    folder = host.folder("arena-launch-r1")
    host.write(host.etc / "provider-secrets" / "names.json", {"path": str(folder)})

    plan = host.plan(host.manifest([host.folder_entry(folder)]))

    assert plan["eligible_count"] == 1, plan["entries"][0]
    assert not (host.control / "beta-policy-screening").exists()


def test_a_folder_may_name_its_own_files_and_a_deleted_folder_may_name_another(tmp_path):
    x, y, z = (str(tmp_path / name) for name in ("x", "y", "z"))
    state = oad.LiveState()
    state._add(f"{y}/cfg.json", {f"{x}/f"})  # a file inside another folder this run deletes
    state._add(f"{x}/binding.json", {f"{x}/own"})  # a folder naming its own files
    entries = [{"path": path, "class": oad.FOLDER} for path in (x, y)]

    verdicts = oad.live_state_reasons(entries, state, deleted=[x, y])
    assert [verdicts[path]["reasons"] for path in (x, y)] == [[], []]

    # The same file counts once the folder holding it stays.
    assert oad.live_state_reasons(entries, state, deleted=[x])[x] == {
        "reasons": ["named_by_live_state"], "named_by": [f"{y}/cfg.json"], "named_by_released_code": []}
    # ... and when it stays because live state names it in turn, found on a second pass.
    state._add(str(tmp_path / "etc" / "p.json"), {z, f"{y}/cfg.json"})
    chained = oad.live_state_reasons(entries, state, deleted=[x, y])
    assert chained[y]["reasons"] == ["named_by_live_state"]
    assert chained[x] == {"reasons": ["named_by_live_state"], "named_by": [f"{y}/cfg.json"],
                          "named_by_released_code": []}


def test_a_reference_from_a_folder_that_stays_still_counts(host):
    # A retained binding inside one folder names another; the intent that records it keeps the first, so
    # the binding's reference keeps the second (a second pass of the check).
    receipt, holder = host.folder("scene-841757-raw-30b10e56-20260905"), host.folder("completed-scene-preparation")
    binding = host.write(holder / "scene-a" / "public-source" / "binding.json",
                         {"installation_receipt": {"path": f"{receipt}/r.json"}})
    host.write(host.intents / "scene-a" / "progression.json", {"source_binding": {"path": str(binding)}})
    plan = host.plan(host.manifest([host.folder_entry(receipt), host.folder_entry(holder)]))

    rows = {Path(row["path"]).name: row for row in plan["entries"]}
    assert rows[holder.name]["named_by"] == [str(host.intents / "scene-a" / "progression.json")]
    assert (rows[receipt.name]["reasons"], rows[receipt.name]["named_by"]) == (["named_by_live_state"], [str(binding)])


def test_apply_lets_only_what_it_deletes_name_another_path(host, monkeypatch):
    first, second = host.folder("arena-launch-r1"), host.folder("arena-launch-r2")
    manifest = host.manifest([host.folder_entry(first), host.folder_entry(second)])
    assert host.plan(manifest)["eligible_count"] == 2
    write_storage_pin(pins_root=host.control / "storage-pins", kind="preparation", owner_id="pin-1",
                      paths=[second], ttl_seconds=DAY, now=lambda: APPROVED_EPOCH + 100)
    seen = []
    real = oad.live_state_reasons
    monkeypatch.setattr(oad, "live_state_reasons", lambda entries, state, deleted: seen.append(list(deleted))
                        or real(entries, state, deleted))

    result = host.apply(manifest)

    assert seen == [[str(first)]]  # not the plan's two: the pinned one stays, so its files would count
    assert result["outcome_counts"] == {"deleted": 1, "skipped": 1} and second.is_dir()


def test_a_folder_named_after_the_plan_is_kept_by_apply(host):
    folder = host.folder("arena-launch-r1")
    manifest = host.manifest([host.folder_entry(folder)])
    assert host.plan(manifest)["eligible_count"] == 1
    profile = host.write(host.etc / "task-evaluation-launch-profiles" / "new.json",
                         {"immutable_inputs": [{"path": f"{folder}/job/result.json"}]})

    result = host.apply(manifest)

    assert result["outcomes"] == [{"path": str(folder), "outcome": "skipped", "reasons": ["named_by_live_state"],
                                   "named_by": [str(profile)], "named_by_released_code": []}]
    assert folder.is_dir()


def test_the_2026_11_12_shapes_are_never_eligible(host):
    # The real shapes of the damage report's section 1a, each still marked ended or development-only.
    robot, receipt, authorization, inputs = (host.folder(name) for name in (
        "native-droid-robot-isaac6-c8d72259", "scene-841757-raw-30b10e56-20260905",
        "sam31-scene841757-authorities-20260905", "arena-launch-r5"))
    host.write(host.etc / "task-evaluation-controls-robot-catalog.json", {"bindings": [
        {"binding_id": "franka-pick-place-841757", "robot_asset_usd_path": f"{robot}/franka_robotiq_2f_85_flattened.usd"}]})
    host.write(host.bindings / "public-scene-841757-autonomous-20260907.json", {
        "rights_reference": "internal_noncommercial_research_only",
        "references": {"installation_receipt": {"path": f"{receipt}/installation_receipt.json"}}})
    machinery = host.write(host.etc / "task-evaluation-public-scene-machinery.renewal-20260921.json",
                           {"privacy_authorization": f"{authorization}/privacy.json"})
    host.write(host.etc / "task-evaluation-scene-progression.json", {"machinery_path": str(machinery)})
    host.write(host.etc / "task-evaluation-launch-profiles" / "arena.json", {
        "claim_ceiling": "development_only", "immutable_inputs": [{"path": f"{inputs}/scene.usd"}]})
    host.write(host.intents / "scene-0353eba5" / "intent.json", {
        "status": "revoked", "request": {"source": {"binding_id": "public-scene-841757-autonomous-20260907"}}})
    manifest = host.manifest([host.folder_entry(path) for path in (robot, receipt, authorization, inputs)])

    plan = host.plan(manifest)

    assert plan["eligible_count"] == 0
    assert {row["reasons"][0] for row in plan["entries"]} == {"named_by_live_state"}
    assert host.apply(manifest)["outcome_counts"] == {"not_eligible_in_plan": 4}
    assert all(path.is_dir() for path in (robot, receipt, authorization, inputs))


def test_the_2026_11_12_manifest_replayed_against_the_naming_files_refuses_the_14_live_named_folders(tmp_path):
    """The real manifest against the files the damage report found naming its folders (paths only)."""

    fixture = json.loads((REPO_ROOT / "tests/fixtures/owner_approved_deletion_2026_11_12_naming_files.json")
                         .read_text())
    mirror = tmp_path / "host"

    def mirrored(path: str) -> Path:
        return mirror / path.lstrip("/")

    by_file: dict[str, list[str]] = {}
    for row in fixture["rows"]:
        by_file.setdefault(row["naming_file"], []).append(row["named_path"])
    for name, paths in by_file.items():
        target = mirrored(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(sorted(set(paths))), encoding="utf-8")  # the paths, no invented fields
    for root in ("etc/blueprint", "var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents"):
        (mirror / root).mkdir(parents=True, exist_ok=True)
    catalog = mirror / "var/lib/blueprint/pipeline-control-plane/task-evaluation-launch-profile-catalog.json"
    catalog.write_text("[]", encoding="utf-8")
    manifest, _ = oad.load_manifest(REPO_ROOT / "deploy/owner-approved-deletions/2026-11-12-old-development-folders.json")
    paths = [entry["path"] for entry in manifest["entries"]]

    state = oad.scan_live_state(
        config_root=mirror / "etc/blueprint",
        intent_root=mirror / "var/lib/blueprint/pipeline-control-plane/task-evaluation-scene-intents",
        binding_root=mirrored(str(oad.DEFAULT_PUBLIC_SOURCE_BINDING_ROOT)), launch_profile_catalog=catalog,
        inputs_root=mirrored(str(oad.INPUTS_ROOT)))
    reasons = oad.live_state_reasons(manifest["entries"], state, deleted=paths)

    refused = {path: verdict["named_by"] for path, verdict in reasons.items() if verdict["reasons"]}
    expected: dict[str, set[str]] = {}
    for row in fixture["rows"]:
        expected.setdefault(row["deleted_folder"], set()).add(str(mirrored(row["naming_file"])))
    assert len(expected) == 14
    assert {path: set(files) for path, files in refused.items()} == expected
    assert all((reasons[path]["reasons"], reasons[path]["named_by"]) == (["named_by_live_state"], sorted(expected[path]))
               for path in expected)


# --- privacy-required deletions are never kept by a leftover reference ----------------------------


def test_every_class_is_either_privacy_required_or_housekeeping():
    # A new class must be put in exactly one of the two sets before a manifest may list it.
    assert set(oad.ENTRY_VALIDATORS) == oad.PRIVACY_REQUIRED_CLASSES | oad.HOUSEKEEPING_CLASSES
    assert not oad.PRIVACY_REQUIRED_CLASSES & oad.HOUSEKEEPING_CLASSES
    assert oad.PRIVACY_REQUIRED_CLASSES == {oad.SCENE} and oad.HOUSEKEEPING_CLASSES == {oad.FOLDER}


def _privacy_scene_named_by_a_binding(tmp_path, host):
    # A development website source registration nobody claims any more, as on the host: the retention
    # checks let it go, and the live-state scan sees that it still names the workspace.
    from blueprint_pipeline.decision_evidence_contracts import canonical_digest
    from blueprint_pipeline.website_scene_workspace_retention import REGISTRATION_SCHEMA

    scene = _authority_ended_scene(tmp_path, host)
    registration = {"schema_version": REGISTRATION_SCHEMA, "request_digest": "sha256:" + "a" * 64,
                    "claim_ceiling": "development_only",
                    "references": {"capture_source": {"path": f"{scene}/captures/w/raw"}}}
    registration["registration_digest"] = canonical_digest(registration, digest_field="registration_digest")
    binding = host.write(host.control / "website-source-bindings" / "a96.json", registration)
    old = time.time() - 60 * DAY
    os.utime(binding, (old, old))
    return scene, binding


def test_a_privacy_deletion_named_by_a_binding_is_eligible_and_records_the_reference(tmp_path, host, capsys):
    scene, binding = _privacy_scene_named_by_a_binding(tmp_path, host)
    now = time.time()
    later = now + 8 * DAY
    manifest = host.manifest([_scene_entry(scene)], **_approval_around(now))

    plan = host.plan(manifest, at=later)
    row = plan["entries"][0]
    assert oad._journal(plan["manifest_id"], plan["entries"], plan["reference_scan"])[
        "references_left_pointing_at_nothing_count"] == 1
    assert json.loads(capsys.readouterr().out) == {"manifest_id": "test-manifest", oad.LEFT_POINTING_AT_NOTHING: {
        "path": str(scene), "named_by": [str(binding)], "unreadable": []}}

    assert (row["status"], row["reasons"], row["named_by"]) == ("eligible", [], [str(binding)])
    assert row[oad.LEFT_POINTING_AT_NOTHING] == {"named_by": [str(binding)], "unreadable": []}
    result = host.apply(manifest, at=later + 60)
    assert result["outcome_counts"] == {"deleted": 1} and not scene.exists()
    receipt = json.loads((host.reports / "test-manifest" / "receipts" / f"{oad._key(str(scene))}.json").read_text())
    assert receipt["status"] == "deleted"
    assert receipt[oad.LEFT_POINTING_AT_NOTHING] == {"named_by": [str(binding)], "unreadable": []}


def test_the_same_reference_keeps_a_housekeeping_folder(host):
    folder = host.folder("arena-launch-r1")
    binding = host.write(host.control / "website-source-bindings" / "a96.json",
                         {"capture_source": {"canonical_path": f"{folder}/captures/w/raw"}})

    row = host.plan(host.manifest([host.folder_entry(folder)]))["entries"][0]

    assert (row["status"], row["reasons"], row["named_by"]) == ("skipped", ["named_by_live_state"], [str(binding)])
    assert oad.LEFT_POINTING_AT_NOTHING not in row


def test_a_privacy_deletion_goes_ahead_when_the_scan_is_incomplete_and_housekeeping_does_not(tmp_path, host):
    scene, binding = _privacy_scene_named_by_a_binding(tmp_path, host)
    folder = host.folder("arena-launch-r1")
    broken = host.write(host.etc / "bad.json", "{not json")
    now = time.time()
    later = now + 8 * DAY
    manifest = host.manifest([_scene_entry(scene), host.folder_entry(folder)], **_approval_around(now))

    rows = {row["class"]: row for row in host.plan(manifest, at=later)["entries"]}

    unreadable = [{"path": str(broken), "code": "not_json"}]
    assert (rows[oad.SCENE]["status"], rows[oad.SCENE]["reasons"]) == ("eligible", [])
    assert rows[oad.SCENE][oad.LEFT_POINTING_AT_NOTHING] == {"named_by": [str(binding)], "unreadable": unreadable}
    assert (rows[oad.FOLDER]["status"], rows[oad.FOLDER]["reasons"]) == ("skipped", ["reference_scan_incomplete"])
    assert host.apply(manifest, at=later + 60)["outcome_counts"] == {"deleted": 1, "not_eligible_in_plan": 1}
    assert not scene.exists() and folder.is_dir()


# --- scan roots, released code, and the scan's own limits ---------------------------------------


def test_a_folder_that_is_or_holds_a_scan_root_is_refused_and_the_next_plan_still_runs(host):
    holder = host.folder("registry-holder")
    host.settings.registry_roots = (*host.settings.registry_roots, holder / "nested" / "registry")
    manifest = host.manifest([host.folder_entry(host.bindings), host.folder_entry(holder)])

    plan = host.plan(manifest)

    assert {Path(row["path"]).name: row["reasons"] for row in plan["entries"]} == {
        "public-source-bindings": ["live_state_scan_root"], "registry-holder": ["live_state_scan_root"]}
    host.apply(manifest)
    later = host.plan(host.manifest([host.folder_entry(host.folder("c-folder"))], manifest_id="second"))
    assert later["reference_scan"]["status"] == "complete" and later["eligible_count"] == 1


@pytest.mark.parametrize("name", ["owner-source-store", "toolchains", "particlefield-runtime-assets",
                                  "stage-replays", "semantic-pretraining", "system-runtime-prerequisites"])
def test_a_folder_released_code_uses_as_a_path_component_is_refused(host, name):
    folder = host.folder(name)
    host.settings.release_roots = (REPO_ROOT,)  # the real source tree; its manifests are never read as code

    row = host.plan(host.manifest([host.folder_entry(folder)]))["entries"][0]

    assert row["reasons"] == ["named_by_released_code"]
    assert row["named_by_released_code"] and all(
        path.startswith(str(REPO_ROOT / "src")) or path.startswith(str(REPO_ROOT / "scripts"))
        or path.startswith(str(REPO_ROOT / "deploy")) for path in row["named_by_released_code"])


def test_the_released_code_never_counts_the_deletion_manifests(host):
    folder = host.folder("arena-launch-r1")
    host.write(host.root / "release" / "deploy" / "owner-approved-deletions" / "m.json", {"path": str(folder)})
    host.write(host.root / "release" / "src" / "other.py", 'NAME = "arena-launch-r10"\n')

    assert host.plan(host.manifest([host.folder_entry(folder)]))["eligible_count"] == 1


def test_a_tree_deeper_than_the_limit_is_an_incomplete_scan(host, monkeypatch):
    monkeypatch.setattr(oad, "MAX_SCAN_DEPTH", 2)
    folder = host.folder("arena-launch-r1")
    host.write(host.etc / "a" / "b" / "c.json", {"path": str(folder)})

    plan = host.plan(host.manifest([host.folder_entry(folder)]))

    assert plan["entries"][0]["reasons"] == ["reference_scan_incomplete"]
    assert plan["reference_scan"]["unreadable"] == [{"path": str(host.etc / "a" / "b"), "code": "scan_depth_limit"}]


def test_deeply_nested_json_is_unreadable_not_a_crash(host):
    folder = host.folder("deep")
    deep = host.write(host.etc / "deep.json", "[" * 100000 + "]" * 100000)

    plan = host.plan(host.manifest([host.folder_entry(folder)]))

    assert plan["entries"][0]["reasons"] == ["reference_scan_incomplete"]
    assert plan["reference_scan"]["unreadable"] == [{"path": str(deep), "code": "too_deeply_nested"}]


def test_a_file_gone_since_it_was_listed_names_nothing(host, monkeypatch):
    folder = host.folder("arena-launch-r1")
    moving = host.write(host.control / "beta-policy-screening" / "queue" / "row.json", {"path": str(folder)})
    real_read = Path.read_bytes

    def read_bytes(path):
        if path == moving:
            path.unlink()  # a queue consumer took it between the listing and the read
            raise FileNotFoundError(2, "gone")
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    plan = host.plan(host.manifest([host.folder_entry(folder)]))

    assert plan["reference_scan"]["status"] == "complete" and plan["eligible_count"] == 1


def test_a_linked_configuration_folder_is_read(host, tmp_path):
    folder = host.folder("arena-launch-r1")
    outside = tmp_path / "outside"
    host.write(outside / "c.json", {"p": str(folder)})
    (host.etc / "current").symlink_to(outside)

    row = host.plan(host.manifest([host.folder_entry(folder)]))["entries"][0]

    assert row["named_by"] == [str(host.etc / "current" / "c.json")]


def test_the_journal_names_each_refused_path_its_reasons_and_naming_files_but_no_values(host, monkeypatch, capsys):
    folder, other = host.folder("arena-launch-r1"), host.folder("arena-launch-r2")
    env = host.write(host.etc / "scene.env", f"API_TOKEN=s3cr3t-value\nSOURCE_ROOT={folder}\n")
    manifest = host.manifest([host.folder_entry(folder), host.folder_entry(other)])
    real_plan = oad.plan_manifest
    plans = [host.plan(manifest)]
    # main() plans with production roots; hand it this host's plan instead, as the unit would print it.
    monkeypatch.setattr(oad, "plan_manifest", lambda path, **_: plans[-1])
    monkeypatch.setenv(oad.QUEUE_ROOTS_ENV, str(host.control / "queue"))

    assert oad.main(["plan", "--manifest-dir", str(manifest.parent)]) == 0

    def journal(out: str) -> tuple[list[dict], dict]:
        lines = out.splitlines()
        rows = [json.loads(line) for line in lines if line.startswith('{"manifest_id"')]
        return rows, json.loads("\n".join(lines[len(rows):]))["manifests"][0]

    out = capsys.readouterr().out
    rows, summary = journal(out)
    assert rows == [{"manifest_id": "test-manifest", "refused": {
        "path": str(folder), "reasons": ["named_by_live_state"], "named_by": [str(env)], "named_by_released_code": []}}]
    assert (summary["reference_scan"], summary["refused_count"], summary["unreadable_count"]) == ("complete", 1, 0)
    assert "s3cr3t" not in out
    broken = host.write(host.etc / "bad.json", "{")
    plans.append(real_plan(manifest, report_root=host.reports, settings=host.settings, now=lambda: APPROVED_EPOCH + 60))
    assert oad.main(["plan", "--manifest-dir", str(manifest.parent)]) == 0
    rows, summary = journal(capsys.readouterr().out)
    assert [row["unreadable"] for row in rows if "unreadable" in row] == [{"path": str(broken), "code": "not_json"}]
    assert {row["refused"]["path"] for row in rows if "refused" in row} == {str(folder), str(other)}
    assert summary["reference_scan"] == "incomplete"


def test_someone_elses_malformed_binding_or_queue_file_never_keeps_a_privacy_deletion(tmp_path, host):
    scene = _authority_ended_scene(tmp_path, host)
    folder = host.folder("arena-launch-r1")
    binding = host.write(host.control / "website-source-bindings" / "other.json", "{half")
    queued = host.control / "queue" / "pending" / "row.json"
    queued.write_bytes(b"\xff\xfe not utf-8")
    now = time.time()
    later = now + 8 * DAY
    manifest = host.manifest([_scene_entry(scene), host.folder_entry(folder)], **_approval_around(now))

    rows = {row["class"]: row for row in host.plan(manifest, at=later)["entries"]}

    assert (rows[oad.SCENE]["status"], rows[oad.SCENE]["reasons"]) == ("eligible", [])
    left = rows[oad.SCENE][oad.LEFT_POINTING_AT_NOTHING]
    assert {"path": str(binding), "code": "reference_index_unreadable"} in left["unreadable"]
    assert {"path": str(queued), "code": "queue_inventory_unreadable"} in left["unreadable"]
    assert {"path": str(binding), "code": "not_json"} in left["unreadable"]
    assert oad.UNREADABLE_REFERENCES not in rows[oad.SCENE]["identity"]
    assert rows[oad.FOLDER]["status"] == "skipped"
    assert set(rows[oad.FOLDER]["reasons"]) == {"queue_inventory_unreadable", "reference_scan_incomplete"}
    assert host.apply(manifest, at=later + 60)["outcome_counts"] == {"deleted": 1, "not_eligible_in_plan": 1}
    receipt = json.loads((host.reports / "test-manifest" / "receipts" / f"{oad._key(str(scene))}.json").read_text())
    assert receipt[oad.LEFT_POINTING_AT_NOTHING] == left and not scene.exists() and folder.is_dir()


def test_an_open_process_or_a_pin_still_holds_a_privacy_deletion(tmp_path, host):
    scene = _authority_ended_scene(tmp_path, host)
    host.write(host.control / "website-source-bindings" / "other.json", "{half")
    process = host.root / "proc" / "4242"
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python\0-m\0x\0")
    (process / "environ").write_bytes(b"")
    (process / "cwd").symlink_to(scene)
    now = time.time()
    manifest = host.manifest([_scene_entry(scene)], **_approval_around(now))

    row = host.plan(manifest, at=now + 8 * DAY)["entries"][0]
    assert row["status"] == "skipped" and "in_use" in row["reasons"]

    (process / "cwd").unlink()
    write_storage_pin(pins_root=host.control / "storage-pins", kind="preparation", owner_id="pin-1",
                      paths=[scene], ttl_seconds=30 * DAY, now=lambda: now)
    row = host.plan(manifest, at=now + 8 * DAY)["entries"][0]
    assert row["status"] == "skipped" and "pinned" in row["reasons"]
