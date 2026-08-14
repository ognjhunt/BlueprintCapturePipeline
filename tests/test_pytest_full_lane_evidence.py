from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline import pytest_full_lane_evidence as MODULE


def test_manifest_canonicalizes_nodeids_for_parallel_execution() -> None:
    items = [SimpleNamespace(nodeid="tests/test_b.py::test_b"), SimpleNamespace(nodeid="tests/test_a.py::test_a")]

    manifest = MODULE.build_manifest(items, phase="executed")

    assert manifest["nodeids"] == ["tests/test_a.py::test_a", "tests/test_b.py::test_b"]
    assert manifest["nodeids_sha256"] == MODULE.nodeids_sha256(manifest["nodeids"])


def test_only_gw0_writes_the_xdist_collection_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "executed.json"
    monkeypatch.setenv(MODULE.OUTPUT_ENV, str(output))
    monkeypatch.setenv(MODULE.PHASE_ENV, "executed")
    items = [SimpleNamespace(nodeid="tests/test_a.py::test_a")]

    MODULE.pytest_collection_finish(
        SimpleNamespace(
            items=items,
            config=SimpleNamespace(workerinput={"workerid": "gw1"}),
        )
    )
    assert not output.exists()

    MODULE.pytest_collection_finish(
        SimpleNamespace(
            items=items,
            config=SimpleNamespace(workerinput={"workerid": "gw0"}),
        )
    )
    assert json.loads(output.read_text(encoding="utf-8"))["nodeids"] == [
        "tests/test_a.py::test_a"
    ]
