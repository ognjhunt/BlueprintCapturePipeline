"""The runtime tree walk reports what Path.rglob reported, without a path operation per file."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from blueprint_pipeline.immutable_tree_walk import path_order, walk_tree


@pytest.fixture()
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg-extra").mkdir()
    (root / "pkg" / "sub" / "module.py").write_text("x")
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg-extra" / "a.py").write_text("x")
    (root / "pkg.py").write_text("x")
    (root / ".hidden").write_text("x")
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret").write_text("x")
    return root


def test_it_lists_what_rglob_lists_with_the_same_relative_paths(tree: Path) -> None:
    (tree / "pkg" / "file-link").symlink_to(tree / "pkg.py")

    walked = {relative for relative, _entry in walk_tree(tree)}

    assert walked == {path.relative_to(tree).as_posix() for path in tree.rglob("*")}


def test_a_directory_comes_before_what_it_holds(tree: Path) -> None:
    seen: set[str] = set()
    for relative, _entry in walk_tree(tree):
        parent = os.path.dirname(relative)
        assert not parent or parent in seen, relative
        seen.add(relative)


def test_a_directory_link_is_reported_and_not_followed(tree: Path, tmp_path: Path) -> None:
    (tree / "dir-link").symlink_to(tmp_path / "outside", target_is_directory=True)

    walked = dict(walk_tree(tree))

    assert walked["dir-link"].is_symlink() and not walked["dir-link"].is_dir(follow_symlinks=False)
    assert "dir-link/secret" not in walked


def test_an_unreadable_directory_lists_nothing_as_rglob_does(tree: Path) -> None:
    locked = tree / "pkg" / "sub"
    locked.chmod(0o000)
    try:
        if os.access(locked, os.R_OK):
            pytest.skip("this user reads anything")
        walked = {relative for relative, _entry in walk_tree(tree)}
        assert "pkg/sub" in walked and "pkg/sub/module.py" not in walked
    finally:
        locked.chmod(0o755)


def test_a_missing_root_lists_nothing(tmp_path: Path) -> None:
    assert list(walk_tree(tmp_path / "absent")) == []


def test_path_order_is_the_order_manifests_already_hold(tree: Path) -> None:
    """A manifest's digest covers its order. Plain string order would put pkg-extra before pkg/."""
    relative = [relative for relative, _entry in walk_tree(tree)]

    assert sorted(relative, key=path_order) == [path.relative_to(tree).as_posix() for path in sorted(tree.rglob("*"))]
    assert sorted(relative, key=path_order) != sorted(relative)
