"""Retained census annotations are proposals, never filesystem authority."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_scratch_decisions.py
#   scripts/lane_scratch_census.py

from __future__ import annotations

import hashlib
import json
import os

import pytest


def _json(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode()


def _inventory(rows=()):
    return {"schema_version": "control_plane_lane_scratch_census.v1",
            "status": "complete", "observed_at_epoch": 900,
            "rows": list(rows), "candidate_count": len(rows), "entries_visited": len(rows),
            "unique_allocated_bytes": sum(row["allocated_bytes"] for row in rows),
            "scan_errors": [], "mutations": 0}


def _annotations(census, decisions=()):
    return _json({"schema_version": "control_plane_lane_scratch_annotations.v1",
                  "census_digest": "sha256:" + hashlib.sha256(census).hexdigest(),
                  "decisions": list(decisions)})


def _validate(census, annotations, **kwargs):
    from blueprint_pipeline.control_plane_lane_scratch_decisions import validate_census_annotations

    return validate_census_annotations(census, annotations, now=1000,
                                       allowed_roots=("/work", "/inputs"), **kwargs)


def _refused(code, census, annotations, **kwargs):
    from blueprint_pipeline.control_plane_lane_scratch_decisions import CensusDecisionError

    with pytest.raises(CensusDecisionError, match=f"^{code}$"):
        _validate(census, annotations, **kwargs)


def test_empty_census_binds_exact_retained_bytes():
    census = _json(_inventory())
    annotations = _annotations(census)
    result = _validate(census, annotations)
    assert result["census_digest"] == "sha256:" + hashlib.sha256(census).hexdigest()
    assert result["annotations_digest"] == "sha256:" + hashlib.sha256(annotations).hexdigest()
    assert result["mutations"] == 0 and result["execution_authorized"] is False
    assert result["requires_fresh_reference_check"] is True
    assert result["decisions"] == []
    _refused("census_identity_mismatch", census + b" ", annotations)


@pytest.mark.parametrize("payload", [b'{"status":1,"status":2}', b"[]", b"not json",
                                    b'{"x":NaN}', b'{"x":Infinity}', b'"\xff"'])
def test_ambiguous_or_nonfinite_json_refuses(payload):
    _refused("census_json_invalid", payload, _annotations(payload))


def test_input_and_output_bounds_are_injected_without_large_payloads():
    census = _json(_inventory())
    annotations = _annotations(census)
    _refused("census_input_too_large", census, annotations, max_input_bytes=16)
    _refused("census_validation_output_too_large", census, annotations, max_output_bytes=16)


def test_reader_reads_regular_file_without_mutation(tmp_path):
    from blueprint_pipeline.control_plane_lane_scratch_decisions import read_census_input

    path = tmp_path / "census.json"
    path.write_bytes(b"{}")
    assert read_census_input(path) == b"{}"
    assert path.read_bytes() == b"{}"


@pytest.mark.parametrize("kind", ["file_symlink", "parent_symlink", "directory", "missing"])
def test_reader_refuses_unsafe_or_missing_inputs(tmp_path, kind):
    from blueprint_pipeline.control_plane_lane_scratch_decisions import (
        CensusDecisionError, read_census_input,
    )

    real = tmp_path / "real"
    real.mkdir()
    source = real / "census.json"
    source.write_bytes(b"{}")
    if kind == "file_symlink":
        path = tmp_path / "linked.json"
        path.symlink_to(source)
    elif kind == "parent_symlink":
        parent = tmp_path / "linked"
        parent.symlink_to(real, target_is_directory=True)
        path = parent / "census.json"
    elif kind == "directory":
        path = real
    else:
        path = tmp_path / "missing"
    with pytest.raises(CensusDecisionError):
        read_census_input(path)


def test_reader_refuses_oversized_or_growing_input(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_lane_scratch_decisions as decisions

    path = tmp_path / "census.json"
    path.write_bytes(b"{}")
    with pytest.raises(decisions.CensusDecisionError, match="census_input_too_large"):
        decisions.read_census_input(path, max_bytes=1)
    real_read = os.read
    changed = False

    def grow(descriptor, count):
        nonlocal changed
        if not changed:
            changed = True
            path.write_bytes(b"{} ")
        return real_read(descriptor, count)

    monkeypatch.setattr(decisions.os, "read", grow)
    with pytest.raises(decisions.CensusDecisionError, match="census_input_unsafe"):
        decisions.read_census_input(path)


def test_reader_refuses_unreadable_input_with_typed_code(tmp_path, monkeypatch):
    from blueprint_pipeline import control_plane_lane_scratch_decisions as decisions

    path = tmp_path / "census.json"
    path.write_bytes(b"{}")
    real_open = os.open

    def refuse(name, *args, **kwargs):
        if name == path.name:
            raise PermissionError("do not echo this private path")
        return real_open(name, *args, **kwargs)

    monkeypatch.setattr(decisions.os, "open", refuse)
    with pytest.raises(decisions.CensusDecisionError, match="^census_input_unreadable$"):
        decisions.read_census_input(path)
