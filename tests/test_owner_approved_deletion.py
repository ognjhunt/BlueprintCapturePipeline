"""Owner-approved deletions remove exactly the approved, unchanged, unreferenced paths, behind receipts."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/owner_approved_deletion.py
#   deploy/owner-approved-deletions/2026-11-04-arena-scratch.json
#   deploy/owner-approved-deletions/2026-11-04-authority-ended-scenes.json
#   deploy/owner-approved-deletions/2026-11-04-controlled-policy-integration.json
#   deploy/systemd/blueprint-owner-approved-deletion-plan.service
#   deploy/systemd/blueprint-owner-approved-deletion.service

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
                     self.control / "website-source-bindings", root / "units", root / "proc", root / "manifests"):
            path.mkdir(parents=True, exist_ok=True)
        self.reports = root / "reports"
        self.settings = oad.Settings(pins_root=self.control / "storage-pins", queue_roots=[self.control / "queue"],
                                     intent_root=self.control / "task-evaluation-scene-intents",
                                     inputs_root=self.inputs, handoff_root=self.handoffs,
                                     unit_directories=[root / "units"], process_root=root / "proc")

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
    assert early["entries"][0]["reasons"] == ["terminal_within_redelivery_window:capture-1"]

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
    ledger = json.loads((target / "captures" / "capture-1" / "pipeline_job_ledger.json").read_text())
    entry = {"class": oad.SCENE, "path": str(target), "captures": [{
        "capture_id": "capture-1", "terminal_code": "consent_expired", "terminal_at": "2026-09-27T03:53:03+00:00",
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
    assert [path.stem for path in manifests] == ["2026-11-04-arena-scratch", "2026-11-04-authority-ended-scenes",
                                                 "2026-11-04-controlled-policy-integration"]
    for path in manifests:
        manifest, _ = oad.load_manifest(path)
        assert not any("447ae4f4" in json.dumps(entry) for entry in manifest["entries"])
        for entry in manifest["entries"]:
            if entry["class"] == oad.FOLDER:
                # Nothing written after 2026-09-01, except the separately approved 2026-09-28 integration folder.
                limit = 1790812800 if "controlled-policy" in entry["path"] else 1788307200
                assert entry["newest_mtime_epoch"] < limit, entry["path"]


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
