"""Fast-lane contract tests for the cross-repo shared doctrine lock.

These run in the hermetic pre-push lane so a doctrine block cannot drift from
the committed lock without a merge-blocking failure, with no sibling checkout
and no network required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_shared_doctrine import (  # type: ignore[import-not-found]
    LOCK_RELATIVE_PATH,
    LOCK_SCHEMA_VERSION,
    REPO_NAME,
    DoctrineVerificationError,
    digest_block,
    extract_block,
    load_lock,
    verify,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_every_tracked_block_matches_the_lock() -> None:
    results = verify(REPO_ROOT)
    assert results, "lock declares no blocks"
    assert all(row["matched"] for row in results)


def test_lock_is_wellformed_and_covers_this_repo() -> None:
    lock = load_lock(REPO_ROOT)
    assert lock["schema_version"] == LOCK_SCHEMA_VERSION
    for block_name, entry in lock["blocks"].items():
        assert (REPO_ROOT / entry["file"]).is_file(), f"{block_name}: missing source"
        if lock["status"] == "locked":
            assert entry["canonical_sha256"], f"{block_name}: locked without canonical digest"
        else:
            assert entry["observed_sha256"].get(REPO_NAME), (
                f"{block_name}: no baseline recorded for {REPO_NAME}"
            )


def test_lock_lists_this_repo_in_the_repo_set() -> None:
    assert REPO_NAME in load_lock(REPO_ROOT)["repos"]


def test_extraction_is_exclusive_of_marker_lines() -> None:
    text = "before\n<!-- X_START -->\nbody line\n<!-- X_END -->\nafter\n"
    assert extract_block(text, "X") == "body line\n"


def test_extraction_preserves_interior_blank_lines() -> None:
    text = "<!-- X_START -->\na\n\nb\n<!-- X_END -->\n"
    assert extract_block(text, "X") == "a\n\nb\n"


@pytest.mark.parametrize(
    "text",
    [
        "no markers at all\n",
        "<!-- X_START -->\nbody\n",
        "<!-- X_END -->\nbody\n",
        "<!-- X_END -->\nbody\n<!-- X_START -->\n",
        "<!-- X_START -->\na\n<!-- X_START -->\nb\n<!-- X_END -->\n",
    ],
)
def test_malformed_markers_fail_closed(text: str) -> None:
    with pytest.raises(DoctrineVerificationError):
        extract_block(text, "X")


def test_crlf_and_cr_hash_identically_to_lf() -> None:
    """A CRLF checkout must produce the committed LF baseline digest.

    No .gitattributes rule pins these Markdown files to LF, so without folding
    line endings a Windows checkout would fail the gate on unmodified content.
    """

    lf = "<!-- X_START -->\na\n\nb\n<!-- X_END -->\n"
    digests = {
        digest_block(extract_block(text, "X"))
        for text in (lf, lf.replace("\n", "\r\n"), lf.replace("\n", "\r"))
    }
    assert len(digests) == 1


@pytest.mark.parametrize("separator", ["\x0b", "\x0c", " ", " ", "\x85"])
def test_unicode_line_boundaries_are_not_treated_as_line_breaks(separator: str) -> None:
    """Parity guard against `str.splitlines()`.

    `splitlines()` breaks on these characters but JavaScript's `split("\\n")`
    does not, so using it here would make the Python and TypeScript verifiers
    disagree on any document containing one.
    """

    body = extract_block(f"<!-- X_START -->\na{separator}b\n<!-- X_END -->\n", "X")
    assert body == f"a{separator}b\n"


def test_digest_is_stable_for_identical_bodies() -> None:
    a = extract_block("<!-- X_START -->\nsame\n<!-- X_END -->\n", "X")
    b = extract_block("head\n<!-- X_START -->\nsame\n<!-- X_END -->\ntail\n", "X")
    assert digest_block(a) == digest_block(b)


def test_missing_repo_baseline_fails_closed(tmp_path: Path) -> None:
    """An unreconciled lock with no baseline for this repo must not be skipped.

    Builds the unreconciled shape explicitly rather than mutating the committed
    lock, which is `locked` and carries no `observed_sha256` at all — deriving
    from it would couple this test to whichever phase the lock happens to be in.
    """

    lock = _lock_template(status="unreconciled")
    for entry in lock["blocks"].values():
        entry.pop("canonical_sha256", None)
        entry["observed_sha256"] = {REPO_NAME: None}
    _stage_repo(tmp_path, lock)
    with pytest.raises(DoctrineVerificationError, match="no baseline recorded"):
        verify(tmp_path)


def test_locked_status_without_canonical_digest_fails_closed(tmp_path: Path) -> None:
    """A locked lock missing its canonical digest must fail, not pass vacuously."""

    lock = _lock_template(status="locked")
    for entry in lock["blocks"].values():
        entry["canonical_sha256"] = None
        entry.pop("observed_sha256", None)
    _stage_repo(tmp_path, lock)
    with pytest.raises(DoctrineVerificationError, match="canonical_sha256 is absent"):
        verify(tmp_path)


def test_unknown_lock_schema_version_fails_closed(tmp_path: Path) -> None:
    lock = json.loads((REPO_ROOT / LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
    lock["schema_version"] = "blueprint.shared_doctrine_lock.v999"
    _stage_repo(tmp_path, lock)
    with pytest.raises(DoctrineVerificationError, match="unsupported lock schema_version"):
        verify(tmp_path)


def test_missing_lock_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(DoctrineVerificationError, match="missing lock file"):
        verify(tmp_path)


def _lock_template(*, status: str) -> dict:
    """Committed lock with an explicit status, for tests that need a given phase.

    Only `status` is forced; block names and file mappings stay real so these
    tests keep tracking the actual set of shared blocks.
    """

    lock = json.loads((REPO_ROOT / LOCK_RELATIVE_PATH).read_text(encoding="utf-8"))
    lock["status"] = status
    return lock


def _stage_repo(root: Path, lock: dict) -> None:
    """Write a throwaway repo root carrying the real blocks and a mutated lock."""

    (root / "contracts").mkdir(parents=True, exist_ok=True)
    (root / LOCK_RELATIVE_PATH).write_text(json.dumps(lock, indent=2), encoding="utf-8")
    for entry in lock["blocks"].values():
        source = REPO_ROOT / entry["file"]
        (root / entry["file"]).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
