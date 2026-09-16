"""The sweep/source join must survive a successor attempt reusing a retained splat."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_evaluation_scene_configuration_sam31_inputs import (
    _same_source_splat,
)


def _splat(path: Path, payload: bytes = b"standard splat bytes") -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def test_join_accepts_the_same_file_at_the_same_path(tmp_path: Path) -> None:
    source = tmp_path / "attempt-a" / "converted" / "source_standard.ply"
    reference = _splat(source)
    assert _same_source_splat(reference, source) is True


def test_join_accepts_a_retained_splat_reused_by_a_successor_attempt(tmp_path: Path) -> None:
    """Scene 840938, 2026-09-16: a dead provider machine forced a successor attempt.

    The successor reused the retained conversion rather than re-running it, so the
    freeze still referenced the previous attempt's materialized copy. Identical bytes,
    different attempt root -- the run was refused with sweep_source_join_invalid.
    """
    payload = b"identical standard splat bytes"
    retained = tmp_path / "attempt-a" / "converted" / "source_standard.ply"
    reference = _splat(retained, payload)

    # The successor materialises the same bytes under its own attempt root.
    successor = tmp_path / "attempt-b" / "converted" / "source_standard.ply"
    successor.parent.mkdir(parents=True, exist_ok=True)
    successor.write_bytes(payload)

    assert retained != successor
    assert _same_source_splat(reference, successor) is True


def test_join_refuses_a_different_source_splat(tmp_path: Path) -> None:
    """A genuinely different source must still fail closed."""
    reference = _splat(tmp_path / "attempt-a" / "source_standard.ply", b"one scene")
    other = tmp_path / "attempt-b" / "source_standard.ply"
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(b"a different scene entirely")
    assert _same_source_splat(reference, other) is False


def test_join_refuses_a_reference_whose_bytes_changed(tmp_path: Path) -> None:
    """The referenced file must still match the digest the freeze recorded."""
    path = tmp_path / "attempt-a" / "source_standard.ply"
    reference = _splat(path, b"frozen bytes")
    path.write_bytes(b"tampered bytes!")
    with pytest.raises(Exception):
        _same_source_splat(reference, path)
