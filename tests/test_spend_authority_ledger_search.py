"""The bounded legacy-ledger search finds exactly what the old full-depth glob found."""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import pytest

from blueprint_pipeline import spend_authority_ledger_migration as migration


def old_discover(base: Path, root: Path) -> list[Path]:
    """The pre-change search: resolve and test every entry to depth three."""
    candidates: list[Path] = []
    for depth in range(1, migration._SEARCH_DEPTH + 1):
        candidates.extend(base.glob("/".join(["*"] * depth)))
    found, seen = [], set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved == root.resolve() or resolved in seen or not migration._is_ledger(resolved):
            continue
        seen.add(resolved)
        found.append(resolved)
    return sorted(found)


@pytest.fixture(autouse=True)
def _no_home_or_named_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home-unused"))
    monkeypatch.delenv(migration.LEGACY_ROOTS_ENV, raising=False)


def build(base: Path, seed: int) -> None:
    rng = random.Random(seed)
    dirs = [base]
    for _ in range(70):
        child = rng.choice(dirs) / rng.choice(["a", "b", ".hidden", "spend-authority-home",
                                               ".blueprint-spend-authority", "runs"])
        child.mkdir(exist_ok=True)
        dirs.append(child)
    for directory in rng.sample(dirs[1:], 10):
        name = rng.choice(["consumed", "authorizations"])
        if (directory / name).exists():
            continue
        if rng.random() < 0.2:
            (directory / name).write_text("a file named like a ledger directory is not one")
        else:
            (directory / name).mkdir()
            (directory / name / "record.json").write_text("{}")
    for directory in dirs:
        if rng.random() < 0.5:
            (directory / "note.txt").write_text("x")
    outside = base.parent / f"outside-{seed}"
    (outside / "consumed").mkdir(parents=True)
    os.symlink(outside, dirs[3] / "linked-ledger", target_is_directory=True)
    os.symlink(base, dirs[5] / "loop", target_is_directory=True)


@pytest.mark.parametrize("seed", range(10))
def test_same_ledgers_as_the_old_search(tmp_path, seed):
    base = tmp_path / "var-lib-blueprint"
    root = base / "spend-authority"
    (root / "consumed").mkdir(parents=True)  # the configured root itself is never "legacy"
    build(base, seed)
    expected = old_discover(base, root)
    assert expected, "fixture should contain at least the symlinked ledger"
    assert migration.discover_legacy_ledgers(search_bases=[base], root=root) == expected


def test_search_stays_within_depth_three(tmp_path):
    base = tmp_path / "base"
    (base / "a" / "b" / "c" / "consumed").mkdir(parents=True)
    (base / "a" / "b" / "c" / "d" / "consumed").mkdir(parents=True)
    root = tmp_path / "root"
    root.mkdir()
    found = migration.discover_legacy_ledgers(search_bases=[base], root=root)
    assert found == [(base / "a" / "b" / "c").resolve()] == old_discover(base, root)


def test_search_is_faster_on_a_wide_data_root(tmp_path):
    base = tmp_path / "var-lib-blueprint"
    for top in range(40):
        for run in range(60):
            directory = base / f"tree-{top}" / f"run-{run}"
            directory.mkdir(parents=True)
            for item in range(10):
                (directory / f"f{item}.json").write_text("{}")
    (base / "spend-authority-home" / ".blueprint-spend-authority" / "consumed").mkdir(parents=True)
    root = base / "spend-authority"
    root.mkdir()
    started = time.perf_counter()
    old = old_discover(base, root)
    old_seconds = time.perf_counter() - started
    started = time.perf_counter()
    new = migration.discover_legacy_ledgers(search_bases=[base], root=root)
    new_seconds = time.perf_counter() - started
    assert new == old == [(base / "spend-authority-home" / ".blueprint-spend-authority").resolve()]
    print(f"\nlegacy-ledger search over 26,481 entries: old {old_seconds:.3f}s, new {new_seconds:.3f}s")
    assert new_seconds < old_seconds
