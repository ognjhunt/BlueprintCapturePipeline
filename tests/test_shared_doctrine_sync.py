"""Fast-lane tests for the single-source shared doctrine generator.

The splice/extract pair is the load-bearing invariant: whatever
`sync_shared_doctrine` writes must be exactly what `verify_shared_doctrine`
reads back, or propagation and enforcement disagree and the lock is worthless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.sync_shared_doctrine import (  # type: ignore[import-not-found]
    BLOCK_SOURCES,
    CANONICAL_REPO,
    DOCTRINE_DIRECTORY,
    DoctrineSyncError,
    discover_repos,
    read_fragment,
    splice,
    sync,
)
from scripts.verify_shared_doctrine import (  # type: ignore[import-not-found]
    LOCK_RELATIVE_PATH,
    digest_block,
    extract_block,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = "# Title\n\nintro\n\n<!-- {block}_START -->\nstale\n<!-- {block}_END -->\n\nfooter\n"


def test_splice_then_extract_roundtrips() -> None:
    block = "SHARED_PLATFORM_CONTEXT"
    body = "## New\n\ncontent line\n"
    spliced = splice(WRAPPER.format(block=block), block, body)
    assert extract_block(spliced, block) == body


def test_splice_preserves_wrapper_outside_markers() -> None:
    block = "SHARED_VISION"
    spliced = splice(WRAPPER.format(block=block), block, "body\n")
    assert spliced.startswith("# Title\n\nintro\n")
    assert spliced.rstrip("\n").endswith("footer")
    assert "stale" not in spliced


def test_splice_is_idempotent() -> None:
    block = "SHARED_VISION"
    once = splice(WRAPPER.format(block=block), block, "body\n")
    assert splice(once, block, "body\n") == once


def test_splice_preserves_interior_blank_lines() -> None:
    block = "SHARED_VISION"
    body = "a\n\nb\n"
    spliced = splice(WRAPPER.format(block=block), block, body)
    assert extract_block(spliced, block) == body


@pytest.mark.parametrize(
    "text",
    [
        "no markers\n",
        "<!-- SHARED_VISION_START -->\nbody\n",
        "<!-- SHARED_VISION_END -->\nbody\n<!-- SHARED_VISION_START -->\n",
    ],
)
def test_splice_fails_closed_on_malformed_markers(text: str) -> None:
    with pytest.raises(DoctrineSyncError):
        splice(text, "SHARED_VISION", "body\n")


def test_block_sources_cover_every_locked_block() -> None:
    lock = json.loads((REPO_ROOT / LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert set(BLOCK_SOURCES) == set(lock["blocks"])
    for block, (_, target) in BLOCK_SOURCES.items():
        assert lock["blocks"][block]["file"] == target


def test_missing_fragment_fails_closed(tmp_path: Path) -> None:
    (tmp_path / DOCTRINE_DIRECTORY).mkdir()
    with pytest.raises(DoctrineSyncError, match="missing canonical fragment"):
        read_fragment(tmp_path, "SHARED_VISION")


def test_empty_fragment_fails_closed(tmp_path: Path) -> None:
    directory = tmp_path / DOCTRINE_DIRECTORY
    directory.mkdir()
    filename, _ = BLOCK_SOURCES["SHARED_VISION"]
    (directory / filename).write_text("   \n\n", encoding="utf-8")
    with pytest.raises(DoctrineSyncError, match="is empty"):
        read_fragment(tmp_path, "SHARED_VISION")


def test_fragment_is_normalized_to_one_trailing_newline(tmp_path: Path) -> None:
    directory = tmp_path / DOCTRINE_DIRECTORY
    directory.mkdir()
    filename, _ = BLOCK_SOURCES["SHARED_VISION"]
    (directory / filename).write_text("body\n\n\n", encoding="utf-8")
    assert read_fragment(tmp_path, "SHARED_VISION") == "body\n"


def test_discover_repos_always_includes_canonical() -> None:
    assert discover_repos(REPO_ROOT)[CANONICAL_REPO] == REPO_ROOT


def test_discover_repos_skips_absent_siblings(tmp_path: Path) -> None:
    lock = json.loads((REPO_ROOT / LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
    lock["repos"] = [CANONICAL_REPO, "DefinitelyNotCheckedOut"]
    (tmp_path / "contracts").mkdir(parents=True)
    (tmp_path / LOCK_RELATIVE_PATH).write_text(json.dumps(lock), encoding="utf-8")
    assert "DefinitelyNotCheckedOut" not in discover_repos(tmp_path)


def test_sync_check_mode_does_not_write(tmp_path: Path) -> None:
    root = _stage_canonical_repo(tmp_path, body="fresh body\n")
    before = {
        target: (root / target).read_text(encoding="utf-8") for _, target in BLOCK_SOURCES.values()
    }
    report = sync(root, write=False)
    assert report["changed_targets"], "staged repo should be out of date"
    for target, original in before.items():
        assert (root / target).read_text(encoding="utf-8") == original


def test_sync_write_mode_converges_and_locks(tmp_path: Path) -> None:
    root = _stage_canonical_repo(tmp_path, body="fresh body\n")
    sync(root, write=True)

    lock = json.loads((root / LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert lock["status"] == "locked"

    for block, (_, target) in BLOCK_SOURCES.items():
        extracted = extract_block((root / target).read_text(encoding="utf-8"), block)
        assert extracted == "fresh body\n"
        assert lock["blocks"][block]["canonical_sha256"] == digest_block(extracted)
        assert "observed_sha256" not in lock["blocks"][block]

    assert not sync(root, write=False)["changed_targets"], "second run should be a no-op"


def _stage_canonical_repo(root: Path, *, body: str) -> Path:
    """Build a throwaway canonical repo: fragments, stale targets, and a lock."""

    (root / DOCTRINE_DIRECTORY).mkdir(parents=True)
    (root / "contracts").mkdir(parents=True)
    real_lock = json.loads((REPO_ROOT / LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
    real_lock["repos"] = [CANONICAL_REPO]
    (root / LOCK_RELATIVE_PATH).write_text(json.dumps(real_lock, indent=2), encoding="utf-8")

    for block, (filename, target) in BLOCK_SOURCES.items():
        (root / DOCTRINE_DIRECTORY / filename).write_text(body, encoding="utf-8")
        (root / target).write_text(WRAPPER.format(block=block), encoding="utf-8")
    return root
