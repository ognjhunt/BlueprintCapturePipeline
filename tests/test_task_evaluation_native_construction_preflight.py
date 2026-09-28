"""The preflight must refuse by name and must never imply provider work."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import blueprint_pipeline.task_evaluation_native_construction_preflight as preflight
from blueprint_pipeline.task_evaluation_native_construction_preflight import (
    NativeConstructionPreflightError,
    run_preflight,
)

COMMIT = "c32e1afb14b430fbe7d13d8b03c50cca1e364ce3"


def _context(packet_dir: Path, *, schema: str | None = None) -> dict:
    return {
        "schema_version": schema or preflight.CONTEXT_SCHEMA_VERSION,
        "lane": "native_task_arena_construction",
        "team_namespace": "blueprint-adp",
        "operations": {"source_commit": COMMIT},
        "references": {"scene": {"packet_dir": str(packet_dir)}},
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _universe(count: int = 2) -> dict:
    return {
        "schema_version": "native_construction_feedback_candidates.v1",
        "candidates": [{"candidate_id": f"candidate-{i:02d}"} for i in range(count)],
    }


def test_preflight_binds_packet_and_reports_no_provider_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = tmp_path / "construction-packet"
    packet.mkdir()
    context = _write(tmp_path / "context.json", _context(packet))
    universe = _write(tmp_path / "universe.json", _universe(3))

    class _Materialized:
        analytic_candidate_inventory = {"candidates": [1, 2, 3]}

    def _fake(**kwargs):
        assert kwargs["packet_dir"] == packet.resolve()
        assert kwargs["commit"] == COMMIT
        assert len(kwargs["universe"]["candidates"]) == 3
        return _Materialized(), "/workspace/adp_arena_provider_bundle/provider_runtime"

    monkeypatch.setattr(preflight, "materialize_remote_curobo_context", _fake)
    receipt = run_preflight(context_file=context, candidate_universe=universe)

    assert receipt["status"] == "completed"
    assert receipt["blockers"] == []
    assert receipt["expected_production_commit"] == COMMIT
    assert receipt["admitted_candidate_count"] == 3
    assert receipt["bound_candidate_count"] == 3
    # A preflight that ever implied paid work would defeat its own purpose.
    assert receipt["provider_allocation_performed"] is False
    assert receipt["paid_execution_requested"] is False


def test_preflight_names_the_failing_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = tmp_path / "construction-packet"
    packet.mkdir()
    universe = _write(tmp_path / "universe.json", _universe())

    wrong_schema = _write(
        tmp_path / "wrong.json", _context(packet, schema="native_task_arena.v1")
    )
    with pytest.raises(NativeConstructionPreflightError) as schema_error:
        run_preflight(context_file=wrong_schema, candidate_universe=universe)
    assert str(schema_error.value) == "preflight_context_schema_invalid"

    missing_packet = _write(
        tmp_path / "missing.json", _context(tmp_path / "absent-packet")
    )
    with pytest.raises(NativeConstructionPreflightError) as packet_error:
        run_preflight(context_file=missing_packet, candidate_universe=universe)
    assert str(packet_error.value) == "preflight_packet_dir_unreadable"

    context = _write(tmp_path / "context.json", _context(packet))
    empty = _write(tmp_path / "empty.json", {"candidates": []})
    with pytest.raises(NativeConstructionPreflightError) as universe_error:
        run_preflight(context_file=context, candidate_universe=empty)
    assert str(universe_error.value) == "preflight_candidate_universe_empty"

    def _boom(**kwargs):
        raise preflight.CuroboContextError("curobo_scene_collision_usd_path_missing")

    monkeypatch.setattr(preflight, "materialize_remote_curobo_context", _boom)
    with pytest.raises(NativeConstructionPreflightError) as context_error:
        run_preflight(context_file=context, candidate_universe=universe)
    # The upstream predicate survives, so the receipt names the real contract.
    assert str(context_error.value) == (
        "preflight_curobo_context_failed:curobo_scene_collision_usd_path_missing"
    )


def test_preflight_cli_writes_receipt_and_exits_two_when_blocked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    universe = _write(tmp_path / "universe.json", _universe())
    blocked = _write(
        tmp_path / "context.json", _context(tmp_path / "absent", schema="wrong.v1")
    )
    receipt_out = tmp_path / "receipt.json"
    exit_code = preflight.main(
        [
            "--context-file",
            str(blocked),
            "--candidate-universe",
            str(universe),
            "--receipt-out",
            str(receipt_out),
        ]
    )
    assert exit_code == 2
    written = json.loads(receipt_out.read_text(encoding="utf-8"))
    assert written["status"] == "blocked"
    assert written["blockers"] == ["preflight_context_schema_invalid"]
    assert written["provider_allocation_performed"] is False
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
