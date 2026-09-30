# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_disk_usage.py
from __future__ import annotations

import fnmatch
import os
from pathlib import PurePosixPath

from blueprint_pipeline import control_plane_disk_usage as usage_module
from blueprint_pipeline.control_plane_disk_usage import survey_usage, tree_usage
from blueprint_pipeline.control_plane_storage_roots import classify_path


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


def _classifier(table):
    def classify(path):
        best = None
        for root, cls in table.items():
            if path == root or path.startswith(root + "/"):
                if best is None or len(root) > len(best[0]):
                    best = (root, cls)
        return None if best is None else type("Root", (), {"path": best[0], "storage_class": best[1]})()
    return classify


def _statvfs(free_blocks=5 * 10**5):
    return lambda _mount: os.statvfs_result((4096, 4096, 10**6, free_blocks, free_blocks, 0, 0, 0, 0, 255))


def test_hardlinks_count_once_and_are_attributed_to_the_first_path(tmp_path):
    base = tmp_path / "var/lib/blueprint"
    store = base / "task-evaluation-inputs/prepared-references/content-addressed/sha256"
    job = base / "task-evaluation-inputs/prepared-references/prep-1"
    store.mkdir(parents=True)
    job.mkdir(parents=True)
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


def test_busy_first_mount_cannot_spend_the_second_mount_entry_budget(tmp_path):
    roots = [tmp_path / name for name in ("root", "work")]
    for root in roots:
        root.mkdir()
        for index in range(12):
            (root / f"{index:02d}.bin").write_bytes(b"x" * 4096)
    report = survey_usage(roots, aliases={}, prefixes=tuple(map(str, roots)),
                          classify=_classifier({str(root): "work" for root in roots}),
                          statvfs=_statvfs(), max_entries=8)
    assert report["status"] == "truncated"
    assert report["entries_visited"] <= 8
    counted = {row["root"] for row in report["top_roots"] if row["allocated_bytes"] > 0}
    assert counted == set(map(str, roots))


def test_busy_first_mount_cannot_spend_the_second_mount_time_budget(tmp_path):
    roots = [tmp_path / name for name in ("root", "work")]
    for root in roots:
        root.mkdir()
        for index in range(12):
            (root / f"{index:02d}.bin").write_bytes(b"x" * 4096)
    now = [0.0]
    classify = _classifier({str(root): "work" for root in roots})

    def slow_classify(path):
        now[0] += 0.1
        return classify(path)

    report = survey_usage(roots, aliases={}, prefixes=tuple(map(str, roots)),
                          classify=slow_classify, statvfs=_statvfs(),
                          max_seconds=0.8, clock=lambda: now[0])
    assert report["status"] == "truncated"
    counted = {row["root"] for row in report["top_roots"] if row["allocated_bytes"] > 0}
    assert counted == set(map(str, roots))


def test_fair_root_budgets_keep_cross_root_hardlinks_counted_once(tmp_path):
    roots = [tmp_path / name for name in ("root", "work")]
    for root in roots:
        root.mkdir()
    original = roots[0] / "blob.bin"
    original.write_bytes(b"x" * 4096)
    os.link(original, roots[1] / "same.bin")
    report = survey_usage(roots, aliases={}, prefixes=tuple(map(str, roots)),
                          classify=_classifier({str(root): "work" for root in roots}),
                          statvfs=_statvfs())
    assert report["status"] == "complete"
    assert report["hardlinks"]["duplicate_names_skipped"] == 1
    assert report["mounts"][0]["surveyed_bytes"] == sum(map(_allocated, roots)) + _allocated(original)


def test_slow_directory_enumeration_leaves_time_to_survey_later_roots(tmp_path, monkeypatch):
    roots = [tmp_path / name for name in ("root", "work")]
    for root in roots:
        root.mkdir()
        for index in range(12):
            (root / f"{index:02d}.bin").write_bytes(b"x" * 4096)
    now = [0.0]
    real_scandir = os.scandir

    class SlowScandir:
        def __init__(self, path):
            self.iterator = real_scandir(path)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.iterator.close()

        def __iter__(self):
            return self

        def __next__(self):
            entry = next(self.iterator)
            now[0] += 0.1
            return entry

    monkeypatch.setattr(usage_module.os, "scandir", SlowScandir)
    report = survey_usage(roots, aliases={}, prefixes=tuple(map(str, roots)),
                          classify=_classifier({str(root): "work" for root in roots}),
                          statvfs=_statvfs(), max_seconds=0.8, clock=lambda: now[0])
    assert report["status"] == "truncated"
    assert now[0] <= 0.8
    assert {row["root"] for row in report["top_roots"]} == set(map(str, roots))
    assert all(row["allocated_bytes"] >= _allocated(root) + 4096
               for root in roots for row in report["top_roots"] if row["root"] == str(root))


def test_deferred_hardlinks_cannot_spend_later_roots_owner_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_module, "SURVEY_MAX_BUFFERED_ENTRIES", 4)
    roots = [tmp_path / name for name in ("root", "work")]
    for root in roots:
        root.mkdir()
        (root / "a.bin").write_bytes(b"x" * 4096)
        (root / "b.bin").write_bytes(b"x" * 4096)
    os.link(roots[0] / "b.bin", tmp_path / "outside.bin")
    for name in ("a.bin", "b.bin"):
        os.link(roots[1] / name, tmp_path / f"outside-work-{name}")
    report = survey_usage(roots, aliases={}, prefixes=tuple(map(str, roots)),
                          classify=_classifier({}), statvfs=_statvfs())
    assert report["status"] == "truncated"
    assert len(report["top_owners"]) <= 4
    assert any(row["root"].startswith(str(roots[1])) and row["allocated_bytes"] >= 4096
               for row in report["top_roots"])


def test_busy_first_mount_cannot_spend_second_mount_owner_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_module, "SURVEY_MAX_BUFFERED_ENTRIES", 4)
    roots = [tmp_path / name for name in ("root", "work")]
    for root in roots:
        for index in range(4):
            (root / f"owner{index}").mkdir(parents=True)
    report = survey_usage(roots, aliases={}, prefixes=tuple(map(str, roots)),
                          classify=_classifier({}), statvfs=_statvfs())
    assert report["status"] == "truncated"
    assert len(report["top_owners"]) <= 4
    assert any(row["root"].startswith(str(roots[1])) for row in report["top_roots"])


def test_orphan_scratch_summary_counts_unique_bytes_and_newest_mtime(tmp_path):
    volume = tmp_path / "work"
    orphan = volume / "loose-run"
    orphan.mkdir(parents=True)
    data = orphan / "data.bin"
    data.write_bytes(b"x" * 4096)
    (orphan / "same.bin").hardlink_to(data)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"z" * 9000)
    (orphan / "outside-link").symlink_to(outside)
    os.utime(data, (1234, 1234))
    os.utime(orphan, (1200, 1200))
    survey = survey_usage([str(volume)], aliases={str(volume): "/mnt/blueprint-work"},
                          statvfs=_statvfs(), mountinfo=str(tmp_path / "no-mountinfo"))
    [row] = survey["orphan_scratch_roots"]
    assert row["root"] == "/mnt/blueprint-work/loose-run"
    assert row["allocated_bytes"] == sum(
        _allocated(path) for path in (orphan, data, orphan / "outside-link")
    )
    assert survey["hardlinks"]["duplicate_names_skipped"] == 1
    assert row["newest_mtime_epoch"] >= 1234
    assert survey["orphan_scratch_bytes"] == row["allocated_bytes"]
    assert survey["orphan_scratch_count"] == 1


def test_unleased_lane_folder_remains_unclassified(tmp_path):
    from blueprint_pipeline.control_plane_lane_scratch import create_lane_scratch

    volume = tmp_path / "work"
    lane_root = volume / "lanes"
    lane_root.mkdir(parents=True)
    registered = create_lane_scratch("g1", "registered", root=lane_root, owner="agent-a",
                                     run_ref="run-1", reason="diagnostic", class_intent="scratch",
                                     cleanup="owner_review", ttl_seconds=3600)
    (registered / "keep.bin").write_bytes(b"k" * 4096)
    orphan = lane_root / "g1" / "unleased"
    orphan.mkdir()
    (orphan / "blob.bin").write_bytes(b"b" * 4096)
    (orphan / ".lane-scratch.v1.json").symlink_to(registered / ".lane-scratch.v1.json")
    survey = survey_usage([str(volume)], aliases={str(volume): "/mnt/blueprint-work"},
                          statvfs=_statvfs(), mountinfo=str(tmp_path / "no-mountinfo"))
    assert any(row["root"] == "/mnt/blueprint-work/lanes/g1/unleased"
               for row in survey["orphan_scratch_roots"])
    assert any(row["owner"] == "lane:g1" and row["storage_class"] == "lane_scratch"
               for row in survey["top_owners"])


def test_orphan_scratch_rows_are_bounded_without_losing_the_total(tmp_path):
    volume = tmp_path / "work"
    volume.mkdir()
    for index in range(60):
        (volume / f"loose-{index:02d}").mkdir()
    survey = survey_usage([str(volume)], aliases={str(volume): "/mnt/blueprint-work"},
                          statvfs=_statvfs(), mountinfo=str(tmp_path / "no-mountinfo"))
    assert survey["orphan_scratch_count"] == 60
    assert len(survey["orphan_scratch_roots"]) == 50
    assert survey["orphan_scratch_bytes"] > sum(row["allocated_bytes"]
                                                for row in survey["orphan_scratch_roots"])


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


def test_shared_bytes_go_to_the_smallest_name_whatever_the_traversal_order(tmp_path):
    # ``a/g`` is visited before ``a/deep/f`` (a directory's files come before its
    # subdirectories), yet ``a/deep/f`` is the smaller path, so the bytes move to it.
    base = tmp_path / "var/lib/blueprint"
    (base / "a/deep").mkdir(parents=True)
    (base / "a/g").write_bytes(b"g" * 40_000)
    os.link(base / "a/g", base / "a/deep/f")
    survey = survey_usage([str(base)], aliases={}, prefixes=(str(base),),
                          classify=_classifier({str(base / "a"): "cache"}), statvfs=_statvfs())
    owners = {row["owner"]: row["allocated_bytes"] for row in survey["top_owners"]}
    assert owners["a/deep"] >= 40_000 > owners["a"]
    assert survey["hardlinks"] == {"shared_inodes": 1, "shared_bytes": owners["a/deep"] - _allocated(base / "a/deep"),
                                   "duplicate_names_skipped": 1}


def test_a_store_name_found_after_a_job_name_takes_the_shared_bytes(tmp_path):
    base = tmp_path / "var/lib/blueprint"
    references = base / "prepared-references"
    (references / "a-job").mkdir(parents=True)
    (references / "content-addressed/sha256").mkdir(parents=True)
    (references / "a-job/ref.bin").write_bytes(b"r" * 40_000)
    os.link(references / "a-job/ref.bin", references / "content-addressed/sha256" / ("b" * 64))
    survey = survey_usage([str(base)], aliases={}, prefixes=(str(base),),
                          classify=_classifier({str(references): "cache"}), statvfs=_statvfs())
    owners = {row["owner"]: row["allocated_bytes"] for row in survey["top_owners"]}
    assert owners["store:prepared-references"] >= 40_000 > owners["prepared-references/a-job"]


def test_pattern_roots_resolve_to_the_concrete_scene_directory(tmp_path):
    # Storage-root rows may carry ``*`` segments (one path component each); the
    # survey reports the concrete directory the pattern matched, never the pattern.
    base = tmp_path / "var/lib/blueprint"
    pattern = f"{base}/pubsub-handoffs/*/scenes/*"
    marker_pattern = f"{base}/pubsub-handoffs/*/scenes/*.retired.v1.json"
    rows = {str(base / "pubsub-handoffs"): "work", pattern: "scene_workspace", marker_pattern: "evidence_hot"}

    def classify(path):
        candidate = PurePosixPath(path).parts
        best = None
        for root, cls in rows.items():
            segments = PurePosixPath(root).parts
            if len(segments) <= len(candidate) and all(
                fnmatch.fnmatchcase(part, segment) for part, segment in zip(candidate, segments)
            ):
                rank = (len(segments), len(root.replace("*", "")))
                if best is None or rank > best[0]:
                    best = (rank, root, cls)
        return None if best is None else type("Root", (), {"path": best[1], "storage_class": best[2]})()

    scene = base / "pubsub-handoffs/bucket/scenes/site-capture-1"
    (scene / "captures").mkdir(parents=True)
    (scene / "captures/video.mov").write_bytes(b"v" * 60_000)
    (base / "pubsub-handoffs/bucket/scenes/site-capture-0.retired.v1.json").write_bytes(b"{}")
    survey = survey_usage([str(base)], aliases={}, prefixes=(str(base),), classify=classify, statvfs=_statvfs())

    roots = {row["root"]: row["storage_class"] for row in survey["top_roots"]}
    assert roots[str(scene)] == "scene_workspace"
    assert roots[str(scene.parent / "site-capture-0.retired.v1.json")] == "evidence_hot"
    assert not any("*" in row["root"] for row in survey["top_roots"] + survey["top_owners"])
    top = survey["top_owners"][0]
    assert (top["owner"], top["root"], top["storage_class"]) == ("scene:site-capture-1", str(scene), "scene_workspace")


def test_mounts_nested_on_one_filesystem_are_walked_once(tmp_path):
    base = tmp_path / "var/lib/blueprint"
    (base / "sub").mkdir(parents=True)
    (base / "sub/f").write_bytes(b"f" * 30_000)
    common = dict(aliases={}, prefixes=(str(base),), classify=_classifier({}), statvfs=_statvfs())
    alone = survey_usage([str(base)], **common)
    nested = survey_usage([str(base / "sub"), str(base)], **common)
    assert [row["mount"] for row in nested["mounts"]] == [str(base)]
    assert nested["mounts"][0]["surveyed_bytes"] == alone["mounts"][0]["surveyed_bytes"]
    assert nested["by_class"] == alone["by_class"]


def test_mount_points_listed_in_mountinfo_are_skipped(tmp_path):
    base = tmp_path / "var/lib/blueprint"
    (base / "kept").mkdir(parents=True)
    (base / "bound").mkdir()
    (base / "kept/f").write_bytes(b"k" * 20_000)
    (base / "bound/f").write_bytes(b"b" * 90_000)
    mountinfo = tmp_path / "mountinfo"
    escaped = str(base / "bound").replace(" ", "\\040")
    mountinfo.write_text(f"36 35 98:0 /bound {escaped} rw,noatime shared:1 - ext4 /dev/vdb rw\n", encoding="utf-8")
    survey = survey_usage([str(base)], aliases={}, prefixes=(str(base),), classify=_classifier({}),
                          statvfs=_statvfs(), mountinfo=str(mountinfo))
    roots = {row["root"] for row in survey["unclassified_roots"]}
    assert str(base / "kept") in roots and str(base / "bound") not in roots
    assert survey["mounts"][0]["surveyed_bytes"] < 90_000


def test_unreadable_directories_are_counted_and_the_survey_completes(tmp_path, monkeypatch):
    base = tmp_path / "var/lib/blueprint"
    (base / "sub").mkdir(parents=True)
    real_scandir = os.scandir

    def failing_scandir(path):
        if str(path).endswith("sub"):
            raise PermissionError("denied")
        return real_scandir(path)

    monkeypatch.setattr("blueprint_pipeline.control_plane_disk_usage.os.scandir", failing_scandir)
    survey = survey_usage([str(base)], aliases={}, prefixes=(str(base),), classify=_classifier({}),
                          statvfs=_statvfs())
    assert survey["status"] == "complete" and survey["unreadable"] == 1


def test_default_table_classification_matches_classify_path(tmp_path):
    # The production default prunes classification below directories no storage
    # root can lie under; it must attribute exactly as classify_path does.
    volume = tmp_path / "vol"
    files = {
        "task-evaluation-inputs/prepared-references/content-addressed/sha256/" + "c" * 64: 9_000,
        "pipeline-control-plane/live_pipeline_control_plane_manifest.json": 3_000,
        "pipeline-control-plane/new-thing/x.bin": 5_000,
        "pipeline-control-plane/task-evaluation-launch-runs/run-1/receipt.json": 2_000,
        "pubsub-handoffs/b/scenes/s/captures/v.mov": 7_000,
        "mystery/blob": 4_000,
    }
    for relative, size in files.items():
        (volume / relative).parent.mkdir(parents=True, exist_ok=True)
        (volume / relative).write_bytes(b"d" * size)
    (volume / "task-evaluation-inputs/prepared-references/prep-1").mkdir()
    os.link(volume / next(iter(files)), volume / "task-evaluation-inputs/prepared-references/prep-1/ref.bin")
    common = dict(aliases={str(volume): "/var/lib/blueprint"}, statvfs=_statvfs(),
                  mountinfo=str(tmp_path / "no-mountinfo"))
    pruned = survey_usage([str(volume)], **common)
    exhaustive = survey_usage([str(volume)], classify=classify_path, **common)

    for key in ("mounts", "by_class", "top_roots", "top_owners", "unclassified_roots", "hardlinks"):
        assert pruned[key] == exhaustive[key], key
    roots = {row["root"]: row["storage_class"] for row in pruned["top_roots"]}
    assert roots["/var/lib/blueprint/pipeline-control-plane/live_pipeline_control_plane_manifest.json"] == "evidence_hot"
    assert roots["/var/lib/blueprint/pipeline-control-plane/new-thing"] == "unclassified"
    owners = {row["owner"] for row in pruned["top_owners"]}
    assert {"store:prepared-references", "scene:s", "run:run-1"} <= owners


def test_work_volume_lane_and_orphan_paths_keep_their_physical_owner(tmp_path):
    volume = tmp_path / "blueprint-work"
    for relative in (
        "lanes/agent-a/job-1/output.bin",
        "g1-unregistered/cache.bin",
        "task-evaluation-inputs/prepared-references/cache.bin",
    ):
        path = volume / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 4096)
    aliases = {
        str(volume) + source[len("/mnt/blueprint-work"):]: target
        for source, target in usage_module.DEFAULT_SURVEY_ALIASES.items()
    }
    aliases.setdefault(str(volume), "/mnt/blueprint-work")

    survey = survey_usage([str(volume)], aliases=aliases, statvfs=_statvfs(),
                          mountinfo=str(tmp_path / "no-mountinfo"))

    assert any(row["storage_class"] == "lane_scratch" and row["owner"] == "lane:agent-a"
               for row in survey["top_owners"])
    assert any(row["root"] == "/mnt/blueprint-work/g1-unregistered"
               for row in survey["unclassified_roots"])
    assert all(row["root"] != "/mnt/blueprint-work/lanes"
               for row in survey["unclassified_roots"])
    assert any(row["root"] == "/var/lib/blueprint/task-evaluation-inputs/prepared-references"
               and row["storage_class"] == "cache" for row in survey["top_roots"])


def test_non_utf8_filename_is_escaped_in_the_report(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    base = tmp_path / "var/lib/blueprint"
    base.mkdir(parents=True)
    actual = base / "bad"
    actual.write_bytes(b"x")
    real_scandir = os.scandir

    def scandir(path):
        if str(path) == str(base):
            return nullcontext([SimpleNamespace(name="bad-\udcff", path=str(actual),
                                                stat=lambda **_kwargs: os.lstat(actual))])
        return real_scandir(path)

    monkeypatch.setattr(usage_module.os, "scandir", scandir)

    report = survey_usage([str(base)], aliases={}, prefixes=(str(base),),
                          classify=_classifier({}), statvfs=_statvfs())
    assert report["status"] == "complete"
    assert any("bad-\\xff" in row["root"] for row in report["unclassified_roots"])
    assert report["survey_digest"].startswith("sha256:")


def test_large_directory_is_truncated_before_its_entries_fill_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_module, "SURVEY_MAX_BUFFERED_ENTRIES", 2)
    base = tmp_path / "var/lib/blueprint"
    base.mkdir(parents=True)
    for index in range(3):
        (base / f"f{index}").write_bytes(b"x")

    report = survey_usage([str(base)], aliases={}, prefixes=(str(base),),
                          classify=_classifier({}), statvfs=_statvfs())
    assert report["status"] == "truncated"
    assert report["entries_visited"] <= 3


def test_shared_inode_map_is_bounded_and_reports_truncation(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_module, "SURVEY_MAX_SHARED_INODES", 2)
    base = tmp_path / "var/lib/blueprint"
    for directory in ("a", "b"):
        (base / directory).mkdir(parents=True)
        for index in range(2):
            original = base / directory / f"f{index}"
            original.write_bytes(b"x")
            os.link(original, tmp_path / f"outside-{directory}-{index}")

    report = survey_usage([str(base)], aliases={}, prefixes=(str(base),),
                          classify=_classifier({str(base): "work"}), statvfs=_statvfs())
    assert report["status"] == "truncated"
    assert report["hardlinks"]["shared_inodes"] <= 2


def test_owner_and_pending_directory_maps_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(usage_module, "SURVEY_MAX_BUFFERED_ENTRIES", 2)
    base = tmp_path / "var/lib/blueprint"
    for relative in ("a/a1", "a/a2", "b"):
        (base / relative).mkdir(parents=True)
    common = dict(aliases={}, prefixes=(str(base),), classify=_classifier({}), statvfs=_statvfs())

    owner_limited = survey_usage([str(base)], **common)
    assert owner_limited["status"] == "truncated"
    assert len(owner_limited["top_owners"]) <= 2

    monkeypatch.setattr(usage_module, "_owner", lambda *_args: "same")
    pending_limited = survey_usage([str(base)], **common)
    assert pending_limited["status"] == "truncated"
    assert pending_limited["entries_visited"] <= 5

def test_tree_walk_stops_at_entry_budget_without_materializing_directory(
    tmp_path, monkeypatch,
):
    root = tmp_path / "work"
    root.mkdir()
    for index in range(30):
        (root / f"{index}.bin").write_bytes(b"x")
    monkeypatch.setattr(usage_module, "MAX_TREE_SCAN_ENTRIES", 3, raising=False)
    real_scandir = os.scandir
    seen = []

    class CountingScandir:
        def __init__(self, path):
            self.iterator = real_scandir(path)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.iterator.close()

        def __iter__(self):
            return self

        def __next__(self):
            entry = next(self.iterator)
            seen.append(entry.name)
            return entry

    monkeypatch.setattr(usage_module.os, "scandir", CountingScandir)
    usage = tree_usage(root)
    assert usage.unreadable == 1
    assert len(seen) <= 4


def test_many_distinct_hardlinks_mark_measurement_incomplete(tmp_path, monkeypatch):
    root = tmp_path / "work"
    root.mkdir()
    other_links = tmp_path / "other-links"
    other_links.mkdir()
    for index in range(4):
        path = root / f"{index}.bin"
        path.write_bytes(b"x")
        os.link(path, other_links / path.name)
    monkeypatch.setattr(usage_module, "MAX_TRACKED_SHARED_INODES", 2, raising=False)
    usage = tree_usage(root)
    assert usage.unreadable == 1
    assert usage.shared_inodes == 2
