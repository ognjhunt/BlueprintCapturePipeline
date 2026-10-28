"""Independent-review regressions: a replaced public ancestor never escapes."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import pytest

from blueprint_pipeline.production_preflight_summary import (
    PreflightSummaryError, build_safe_summary, write_safe_summary,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))
from operator_door.config import DoorConfig  # noqa: E402
from operator_door.status import _preflight  # noqa: E402


def _summary() -> dict:
    report = {"schema_version": "task_evaluation_production_chain_preflight.v1",
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "active_sha": "a" * 40, "lane": "diagnostic", "units": {}, "host_findings": []}
    return build_safe_summary(report, raw_report_bytes=json.dumps(report).encode())


@pytest.mark.parametrize("boundary", ["create", "replace"])
def test_summary_publication_anchors_parent_during_replacement(tmp_path: Path, monkeypatch, boundary: str) -> None:
    public = tmp_path / "preflight" / "public"
    outside = tmp_path / "outside"
    public.mkdir(parents=True)
    outside.mkdir()
    target = outside / "summary.json"
    target.write_bytes(b"outside-must-remain")
    moved = tmp_path / "moved-public"
    swapped = False
    original_open, original_replace = os.open, os.replace

    def swap() -> None:
        nonlocal swapped
        if not swapped:
            public.rename(moved)
            public.symlink_to(outside, target_is_directory=True)
            swapped = True

    def open_at(name, flags, *args, **kwargs):
        if boundary == "create" and isinstance(name, str) and name.startswith(".preflight-summary-"):
            swap()
        return original_open(name, flags, *args, **kwargs)

    def replace_at(source, destination, *args, **kwargs):
        if boundary == "replace":
            swap()
        return original_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_at)
    monkeypatch.setattr(os, "replace", replace_at)
    with pytest.raises(PreflightSummaryError, match="identity_changed"):
        write_safe_summary(public / "summary.json", _summary(), reader_ids=(os.getuid(), os.getgid()))
    assert swapped and target.read_bytes() == b"outside-must-remain"
    assert not list(moved.glob(".preflight-summary-*"))


def test_door_parent_replacement_cannot_read_through_outside_symlink(tmp_path: Path, monkeypatch) -> None:
    public, outside = tmp_path / "public", tmp_path / "outside"
    public.mkdir()
    outside.mkdir()
    (public / "summary.json").write_text(json.dumps(_summary()))
    (outside / "summary.json").write_text(json.dumps(_summary()))
    active_target = tmp_path / ("a" * 40)
    active_target.mkdir()
    active = tmp_path / "active"
    active.symlink_to(active_target)
    original_open = os.open
    swapped = False

    def open_at(name, flags, *args, **kwargs):
        nonlocal swapped
        if name == "summary.json" and not swapped:
            public.rename(tmp_path / "moved")
            public.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_at)
    result = _preflight(dataclasses.replace(DoorConfig(), preflight_summary=str(public / "summary.json"), active_release_link=str(active)))
    assert swapped
    assert result == {"status": "refused", "reason": "preflight_summary_directory_identity_changed"}
