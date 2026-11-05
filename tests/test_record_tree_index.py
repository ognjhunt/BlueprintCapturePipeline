"""One-walk record discovery returns exactly what the replaced ``**`` globs did."""
import os
import random
import time

import pytest

from blueprint_pipeline import record_tree_index as index
from blueprint_pipeline.record_tree_index import Rule, find_records
from blueprint_pipeline.task_evaluation_scene_progression_recovery import OWNERSHIP_RECORD_RULES

OLD = time.time() - 3600


@pytest.fixture(autouse=True)
def fresh_cache():
    index.clear_cache()
    yield
    index.clear_cache()


def age(root):
    """Make every directory old enough for its listing to be reusable."""
    for directory, _, _ in os.walk(root):
        os.utime(directory, (OLD, OLD))
    os.utime(root, (OLD, OLD))


def ownership_globs(roots):
    paths = set()
    for root in roots:
        paths.update(root.rglob("*.lease.json"))
        paths.update(root.glob("**/pending_teardowns/*.json"))
        paths.update(root.glob("**/pending-teardowns/*.json"))
    return sorted(paths)


def build_tree(root, seed):
    rng = random.Random(seed)
    names = ["a", "b", ".hidden", "pending_teardowns", "pending-teardowns", "x.lease.json", "object_store_real_run"]
    files = ["r.lease.json", ".lease.json", "t.json", "t.json.tmp", "note.txt", "started_pod_id.txt"]
    dirs = [root]
    for _ in range(60):
        parent = rng.choice(dirs)
        child = parent / rng.choice(names)
        child.mkdir(exist_ok=True)
        dirs.append(child)
    for directory in dirs:
        for name in rng.sample(files, rng.randint(0, len(files))):
            (directory / name).write_text("{}")
    elsewhere = root.parent / (root.name + "-outside")
    (elsewhere / "pending_teardowns").mkdir(parents=True)
    (elsewhere / "pending_teardowns" / "far.json").write_text("{}")
    (elsewhere / "deep.lease.json").write_text("{}")
    (elsewhere / "nested.lease.json").mkdir()
    # Named parent that is a symlink: the pattern component follows it.
    os.symlink(elsewhere / "pending_teardowns", dirs[3] / "pending_teardowns", target_is_directory=True) \
        if not (dirs[3] / "pending_teardowns").exists() else None
    # ``**`` must not recurse through a symlinked directory, nor loop.
    os.symlink(elsewhere, dirs[5] / "linked", target_is_directory=True)
    os.symlink(root, dirs[7] / "loop", target_is_directory=True)
    os.symlink(elsewhere / "deep.lease.json", root / "linked.lease.json")


@pytest.mark.parametrize("seed", range(8))
def test_matches_replaced_ownership_globs(tmp_path, seed):
    roots = [tmp_path / "inputs", tmp_path / "state"]
    for number, root in enumerate(roots):
        root.mkdir()
        build_tree(root, seed * 10 + number)
    expected = ownership_globs(roots)
    assert expected, "fixture should contain records"
    assert sorted(set(find_records(roots, OWNERSHIP_RECORD_RULES))) == expected
    age(tmp_path)
    assert sorted(set(find_records(roots, OWNERSHIP_RECORD_RULES))) == expected  # listings from cache
    assert sorted(set(find_records(roots, OWNERSHIP_RECORD_RULES))) == expected


def test_root_named_like_a_parent_component_does_not_match_itself(tmp_path):
    root = tmp_path / "pending_teardowns"
    root.mkdir()
    (root / "top.json").write_text("{}")
    (root / "pending_teardowns").mkdir()
    (root / "pending_teardowns" / "inner.json").write_text("{}")
    assert find_records([root], OWNERSHIP_RECORD_RULES) == ownership_globs([root]) == [
        root / "pending_teardowns" / "inner.json"]


def test_missing_root_is_skipped_like_glob(tmp_path):
    assert find_records([tmp_path / "absent"], OWNERSHIP_RECORD_RULES) == []


def test_new_removed_and_renamed_records_are_seen_after_cached_scan(tmp_path):
    deep = tmp_path / "runs" / "one" / "lane"
    deep.mkdir(parents=True)
    (deep / "a.lease.json").write_text("{}")
    age(tmp_path)
    assert find_records([tmp_path], OWNERSHIP_RECORD_RULES) == [deep / "a.lease.json"]
    # A writer adds a lease in an old directory (its mtime moves).
    (deep / "b.lease.json").write_text("{}")
    assert find_records([tmp_path], OWNERSHIP_RECORD_RULES) == [deep / "a.lease.json", deep / "b.lease.json"]
    age(tmp_path)
    (deep / "a.lease.json").unlink()
    assert find_records([tmp_path], OWNERSHIP_RECORD_RULES) == [deep / "b.lease.json"]
    age(tmp_path)
    # A whole subtree renamed into place under an unchanged grandparent.
    staged = tmp_path / "staged" / "pending_teardowns"
    staged.mkdir(parents=True)
    (staged / "open.json").write_text("{}")
    age(tmp_path)
    find_records([tmp_path], OWNERSHIP_RECORD_RULES)
    os.rename(tmp_path / "staged", tmp_path / "runs" / "two")
    assert find_records([tmp_path], OWNERSHIP_RECORD_RULES) == [
        tmp_path / "runs" / "one" / "lane" / "b.lease.json", tmp_path / "runs" / "two" / "pending_teardowns" / "open.json"]


def test_racy_listing_is_never_reused(tmp_path):
    lane = tmp_path / "lane"
    lane.mkdir()
    (lane / "a.lease.json").write_text("{}")
    now = time.time()
    os.utime(lane, (now, now))
    find_records([tmp_path], OWNERSHIP_RECORD_RULES)
    assert (OWNERSHIP_RECORD_RULES, str(tmp_path), str(lane)) not in index._CACHE
    # Simulate a change inside the same coarse timestamp tick: same signature.
    stamp = os.stat(lane).st_mtime_ns
    (lane / "b.lease.json").write_text("{}")
    os.utime(lane, ns=(stamp, stamp))
    assert lane / "b.lease.json" in find_records([tmp_path], OWNERSHIP_RECORD_RULES)


def test_unchanged_directories_are_not_reread(tmp_path, monkeypatch):
    for number in range(20):
        lane = tmp_path / f"run-{number}" / "lane"
        lane.mkdir(parents=True)
        (lane / f"{number}.lease.json").write_text("{}")
    age(tmp_path)
    first = find_records([tmp_path], OWNERSHIP_RECORD_RULES)
    calls = []
    real = os.scandir
    monkeypatch.setattr(index.os, "scandir", lambda path: calls.append(path) or real(path))
    assert find_records([tmp_path], OWNERSHIP_RECORD_RULES) == first
    assert calls == []
    (tmp_path / "run-3" / "lane" / "new.lease.json").write_text("{}")
    assert len(find_records([tmp_path], OWNERSHIP_RECORD_RULES)) == 21
    assert calls == [str(tmp_path / "run-3" / "lane")]


def test_rule_requires_one_name_form():
    with pytest.raises(ValueError):
        Rule()
    with pytest.raises(ValueError):
        Rule(name="a", suffix=".json")
    with pytest.raises(ValueError):
        Rule(name="a/b")


def test_timing_against_replaced_globs(tmp_path):
    """A broad tree with few records: report old vs. first vs. repeated scans."""
    root = tmp_path / "state"
    for run in range(300):
        for step in range(5):
            directory = root / f"run-{run:03d}" / f"step-{step}" / "artifacts"
            directory.mkdir(parents=True)
            for item in range(20):
                (directory / f"frame-{item}.bin").write_bytes(b"")
        if run % 50 == 0:
            (root / f"run-{run:03d}" / "pending_teardowns").mkdir()
            (root / f"run-{run:03d}" / "pending_teardowns" / "t.json").write_text("{}")
            (root / f"run-{run:03d}" / "lane.lease.json").write_text("{}")
    age(root)
    repeats = 8  # a release succession reconciles several attempts in one pass

    started = time.perf_counter()
    for _ in range(repeats):
        old = ownership_globs([root])
    old_seconds = time.perf_counter() - started

    started = time.perf_counter()
    new = find_records([root], OWNERSHIP_RECORD_RULES)
    first_seconds = time.perf_counter() - started
    for _ in range(repeats - 1):
        assert find_records([root], OWNERSHIP_RECORD_RULES) == new
    new_seconds = time.perf_counter() - started
    assert new == old and len(new) == 12
    print(f"\n{repeats} ownership scans of 31,500 entries: globs {old_seconds:.3f}s, "
          f"indexed {new_seconds:.3f}s (first {first_seconds:.3f}s)")
    assert new_seconds < old_seconds


def test_spend_guard_owner_records_match_replaced_globs(tmp_path):
    from scripts import gpu_spend_guard as guard

    root = tmp_path / "pipeline-control-plane"
    rng = random.Random(7)
    dirs = [root]
    root.mkdir()
    for _ in range(80):
        child = rng.choice(dirs) / rng.choice(["a", "pipeline", "object_store_real_run", "independent_vast_watchdog"])
        child.mkdir(exist_ok=True)
        dirs.append(child)
    names = [*guard.OWNER_ID_FILENAMES, guard.WARM_SERVE_MARKER_FILENAME, guard.TASK_EVALUATION_WATCHDOG_FILENAME]
    for directory in dirs:
        for name in rng.sample(names, rng.randint(0, 3)):
            (directory / name).write_text("1")
    outside = tmp_path / "outside" / "object_store_real_run"
    outside.mkdir(parents=True)
    (outside / "started_pod_id.txt").write_text("1")
    (outside / "started_vast_instance_id.txt").write_text("1")
    os.symlink(outside, dirs[4] / "object_store_real_run") if not (dirs[4] / "object_store_real_run").exists() else None
    os.symlink(tmp_path / "outside", dirs[6] / "linked", target_is_directory=True)
    for _ in range(2):  # second pass answers from the directory cache
        age(tmp_path)
        for filename in guard.OWNER_ID_FILENAMES:
            expected = set(root.glob(f"**/object_store_real_run/{filename}"))
            if filename == "started_vast_instance_id.txt":
                expected |= set(root.glob(f"**/{filename}"))
            got = {path for path in guard._owner_records(root) if path.name == filename and (
                path.parent.name == "object_store_real_run" or filename == "started_vast_instance_id.txt")}
            assert got == expected and expected
        assert {p for p in guard._owner_records(root) if p.name == guard.WARM_SERVE_MARKER_FILENAME} == set(
            root.glob(f"**/{guard.WARM_SERVE_MARKER_FILENAME}"))
        assert {p for p in guard._owner_records(root) if p.name == guard.TASK_EVALUATION_WATCHDOG_FILENAME
                and p.parent.name == "independent_vast_watchdog"} == set(
            root.glob(f"**/independent_vast_watchdog/{guard.TASK_EVALUATION_WATCHDOG_FILENAME}"))
