from __future__ import annotations

import json

import pytest

from blueprint_pipeline.g1_kitchen_run_index import append_run_index_event, load_run_index


def test_run_index_uses_relative_hashed_refs_and_retains_raw(tmp_path) -> None:
    raw = tmp_path / "attempts" / "a1" / "closure.json"
    raw.parent.mkdir(parents=True)
    raw.write_text(json.dumps({"status": "blocked"}), encoding="utf-8")
    event = append_run_index_event(
        run_root=tmp_path,
        event_type="attempt_terminalized",
        run_id="run",
        attempt_id="a1",
        artifact_paths=[raw],
        detail={"terminal_reason": "superseded"},
    )
    assert event["artifact_refs"][0]["relative_path"] == "attempts/a1/closure.json"
    assert len(event["artifact_refs"][0]["sha256"]) == 64
    assert raw.is_file()
    assert load_run_index(event["index_path"])[0]["retention"]["raw_evidence_retained"] is True


def test_run_index_rejects_external_artifacts_and_duplicate_terminal_events(tmp_path) -> None:
    outside = tmp_path.parent / "outside-g1-index.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="outside run root"):
        append_run_index_event(
            run_root=tmp_path,
            event_type="bundle_ineligible",
            run_id="run",
            artifact_paths=[outside],
        )
    closure = tmp_path / "closure.json"
    closure.write_text("{}", encoding="utf-8")
    append_run_index_event(
        run_root=tmp_path,
        event_type="attempt_terminalized",
        run_id="run",
        attempt_id="a1",
        artifact_paths=[closure],
    )
    with pytest.raises(ValueError, match="duplicate terminal attempt"):
        append_run_index_event(
            run_root=tmp_path,
            event_type="attempt_terminalized",
            run_id="run",
            attempt_id="a1",
            artifact_paths=[closure],
        )
    assert len(load_run_index(tmp_path / "g1_kitchen_run_index.jsonl")) == 1
