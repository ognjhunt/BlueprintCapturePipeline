"""Scene preparation retains the robot-free readiness receipt where the team-evaluation gate reads it.

Beta journey step: L5 team policy evaluation. The revision, its documents and
the rigid physics trace are real artifacts of the rigid relocation fixture,
fetched by their published references through an in-memory store.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import robot_free_probe_gate as gate
from blueprint_pipeline import robot_free_readiness_retention as retention
from blueprint_pipeline import task_evaluation_scene_robot_free_readiness as articulated
from tests.test_rigid_robot_free_probe import recorded_trace, rigid_case
from tests.test_robot_free_probe_gate import scored_probe_receipt
from tests.test_scene_preparation_completion import prepared  # noqa: F401 - pytest fixture
from tests.test_task_evaluation_rigid_relocation_native_adapter import DEFINITION, _rewrite


def _published(tmp_path: Path, *, strategy: str = "planar_push") -> dict[str, Any]:
    (tmp_path / "sources").mkdir()
    configured, references, docs, spec = rigid_case(tmp_path / "sources", strategy=strategy)
    store = {row["uri"]: Path(row["materialized_path"]).read_bytes() for row in references.values()}
    return {"configured": configured, "references": references, "docs": docs, "spec": spec, "store": store}


def _reference(case: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(case["configured"], sort_keys=True).encode()
    reference = {"uri": "s3://blueprint-production-inputs/revision.json",
                 "digest": "sha256:" + hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    case["store"][reference["uri"]] = payload
    return reference


def _retain(case: dict[str, Any], run: Path, **kwargs: Any) -> dict[str, Any]:
    run.mkdir(exist_ok=True)
    return retention.retain_robot_free_readiness(
        run_root=run, revision_reference=_reference(case),
        revision_digest=kwargs.pop("revision_digest", case["configured"]["revision_digest"]),
        fetcher=kwargs.pop("fetcher", lambda reference: case["store"][reference["uri"]]))


def _place_trace(case: dict[str, Any], run: Path, **scene: Any) -> None:
    trace = recorded_trace(case["spec"], case["configured"]["revision_digest"], **scene)
    (run / retention.TRACE_FILENAME).write_text(json.dumps(trace))


def test_a_rigid_scene_is_retained_unavailable_until_its_physics_trace_arrives(tmp_path: Path) -> None:
    case, run = _published(tmp_path), tmp_path / "launch-1"
    revision = case["configured"]["revision_digest"]
    first = _retain(case, run)
    assert first["status"] == "probe_unavailable" and first["write"] == "written"
    retained = gate.retained_probe_receipt(run)
    assert retained["probe_error"] == "physics_trace_missing"
    with pytest.raises(gate.RobotFreeProbeGateError, match="not_passed"):
        gate.require_robot_free_probe(retained, scene_revision_digest=revision, task_kind="rigid_pick_place")
    _place_trace(case, run)
    second = _retain(case, run)
    assert (second["status"], second["write"]) == ("passed", "replaced_unavailable")
    admitted = gate.require_robot_free_probe(gate.retained_probe_receipt(run), scene_revision_digest=revision,
                                            task_kind="rigid_pick_place")
    assert admitted["receipt_digest"] == second["receipt_digest"]
    assert _retain(case, run)["write"] == "unchanged"


def test_a_decided_receipt_is_never_replaced(tmp_path: Path) -> None:
    case, run = _published(tmp_path), tmp_path / "launch-1"
    run.mkdir()
    _place_trace(case, run, support_max_xy=(3.05, -1.4))
    failed = _retain(case, run)
    assert failed["status"] == "failed"
    _place_trace(case, run)
    kept = _retain(case, run)
    assert (kept["status"], kept["write"], kept["receipt_digest"]) == (
        "failed", "kept_existing_decided", failed["receipt_digest"])


def test_an_articulated_scene_is_scored_by_the_articulated_scorer(tmp_path: Path, monkeypatch) -> None:
    case = _published(tmp_path)
    definition = copy.deepcopy(case["docs"][DEFINITION])
    definition["strategy"] = "articulated_open_close"
    _rewrite(tmp_path=tmp_path / "sources", configured=case["configured"], references=case["references"],
             contract_path=DEFINITION, document=definition)
    case["store"] = {row["uri"]: Path(row["materialized_path"]).read_bytes() for row in case["references"].values()}
    revision = case["configured"]["revision_digest"]
    calls = []
    monkeypatch.setattr(articulated, "robot_free_readiness_for_revision",
                        lambda **kwargs: calls.append(kwargs) or scored_probe_receipt(revision))
    result = _retain(case, tmp_path / "launch-1")
    assert result["status"] == "passed" and calls[0]["revision"]["revision_digest"] == revision
    gate.require_robot_free_probe(gate.retained_probe_receipt(tmp_path / "launch-1"), scene_revision_digest=revision,
                                  task_kind="articulated_open_close")


def test_refusals_retain_nothing_and_transient_failures_raise(tmp_path: Path) -> None:
    case, run = _published(tmp_path), tmp_path / "launch-1"
    mismatch = _retain(case, run, revision_digest="sha256:" + "9" * 64)
    assert mismatch == {"status": "refused", "reason": "robot_free_readiness_revision_digest_mismatch",
                        "retained": False}
    assert gate.retained_probe_receipt(run) is None

    def offline(_reference: Any) -> bytes:
        raise OSError("network unreachable")

    with pytest.raises(retention.RobotFreeReadinessRetentionError, match="retention_failed:OSError"):
        _retain(case, run, fetcher=offline)
    linked = tmp_path / "linked"
    linked.symlink_to(run)
    with pytest.raises(retention.RobotFreeReadinessRetentionError, match="path_unsafe"):
        retention.retain_robot_free_readiness(run_root=linked, revision_reference=_reference(case),
                                              revision_digest=case["configured"]["revision_digest"])


def test_a_task_kind_without_a_probe_retains_nothing(tmp_path: Path) -> None:
    case = _published(tmp_path)
    definition = copy.deepcopy(case["docs"][DEFINITION])
    definition["strategy"] = "deformable_fold"
    _rewrite(tmp_path=tmp_path / "sources", configured=case["configured"], references=case["references"],
             contract_path=DEFINITION, document=definition)
    case["store"] = {row["uri"]: Path(row["materialized_path"]).read_bytes() for row in case["references"].values()}
    result = _retain(case, tmp_path / "launch-1")
    assert result == {"status": "unavailable_for_task_kind", "reason": "deformable_fold", "retained": False}
    assert gate.retained_probe_receipt(tmp_path / "launch-1") is None


def test_a_completed_source_launch_is_retained_from_its_validated_terminal_evidence(
        prepared, monkeypatch) -> None:  # noqa: F811
    _args, run, result_path = prepared
    published = json.loads(result_path.read_text())
    calls = []
    monkeypatch.setattr(retention, "retain_robot_free_readiness",
                        lambda **kwargs: calls.append(kwargs) or {"status": "passed", "retained": True})
    assert retention.retain_for_source_launch(run) == {"status": "passed", "retained": True}
    assert calls == [{"run_root": run, "revision_reference": published["configured_scene_revision_reference"],
                      "revision_digest": published["configured_scene_revision_digest"], "fetcher": None}]
