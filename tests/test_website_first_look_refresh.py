"""A job handed off before the first look gets one beside its hand-off, never by touching what was handed off."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_first_look_refresh.py
#   src/blueprint_pipeline/website_robot_fit_projection.py
#   src/blueprint_pipeline/website_atlas_preview.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import website_first_look_refresh as refresh
from blueprint_pipeline import website_robot_fit_projection as projection
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.robot_arm_chains import load_robot_arm_chains
from blueprint_pipeline.robot_spec_catalog import load_robot_spec_catalog
from blueprint_pipeline.website_robot_fit import robot_fit_table
from tests.test_website_robot_fit import _floor_at, _job, _verdict

CONTEXT = "sha256:" + "c" * 64
NOW = 1_800_000_000.0
#: What a row and a table carried only from the first look on; a table written before it has none of them.
FIRST_LOOK_ROW_FIELDS = ("first_look_verdict", "first_look_reasons", "conditions", "missing",
                         "verdict_if_conditions_hold", "rated_payload_kg", "payload_limit_kg")


def _capture(tmp_path: Path) -> Path:
    root = tmp_path / "storage" / "bucket" / "scenes" / "site-req1" / "captures" / "walkthrough-req1"
    write_json(root / "raw" / "manifest.json", {"site_submission_id": "req1", "scene_id": "site-req1",
                                                 "capture_id": "walkthrough-req1"})
    return root


def _old_format(table: dict) -> dict:
    """The table as the hand-off wrote it before the first look: each row's old verdict only."""
    old = {key: value for key, value in table.items() if key not in ("first_look_counts", "table_digest")}
    old["rows"] = [{key: value for key, value in row.items() if key not in FIRST_LOOK_ROW_FIELDS}
                   for row in table["rows"]]
    old["table_digest"] = canonical_digest(old, digest_field="table_digest")
    return old


def _handed_off(tmp_path: Path, monkeypatch, *, old: bool = True) -> tuple[Path, Path]:
    """A capture as the full-route hand-off leaves it: preparation, collider, verdict, inputs and its table."""
    _floor_at(monkeypatch)
    root = _capture(tmp_path)
    work = root / "pipeline" / "website_scene_preparation"
    native = work / "native"
    job = _job(weight="2 kg")
    job["binding"] = {"task_context_digest": CONTEXT}
    job["digest"] = canonical_digest(job, digest_field="digest")
    write_json(work / "preparation.json", job)
    collision = native / "collision" / "key" / "collision.usda"
    collision.parent.mkdir(parents=True)
    collision.write_bytes(b"#usda 1.0\n")
    verdict = _verdict(job, collision)
    write_json(native / "capability_verdict.json", verdict)
    runtime = {"schema_version": "website_scene_runtime_inputs.v1", "preparation_digest": job["digest"],
               "collision": {"path": str(collision)}}
    runtime["digest"] = canonical_digest(runtime, digest_field="digest")
    write_json(native / "runtime_inputs.json", runtime)
    table = robot_fit_table(preparation=job, collision_path=collision, verdict=verdict,
                            catalog=load_robot_spec_catalog(), arm_chains=load_robot_arm_chains())
    table = _old_format(table) if old else table
    write_json(native / "robot_fit_table.json", table)
    handoff = {"schema_version": "website_scene_handoff.v1", "status": "intake_ready", "blockers": [],
               "scene_id": "site-req1", "capture_id": "walkthrough-req1",
               "preparation_path": str(work / "preparation.json"), "preparation_digest": job["digest"],
               "runtime_inputs": {"status": runtime.get("status"), "digest": runtime["digest"],
                                  "path": str(native / "runtime_inputs.json")},
               "capability_verdict": {"status": verdict["status"], "blockers": verdict["blockers"],
                                      "path": str(native / "capability_verdict.json"),
                                      "digest": verdict["verdict_digest"]},
               "robot_fit_table": {"path": str(native / "robot_fit_table.json"), "digest": table["table_digest"],
                                   "counts": table["counts"]}}
    handoff["digest"] = canonical_digest(handoff, digest_field="digest")
    write_json(work / "handoff.json", handoff)
    return root, native / "robot_fit_table.json"


def _v2(root: Path, context: str = CONTEXT):
    return projection.read_robot_fit_v2(capture_root=root, task_context_digest=context)


def test_an_old_handoff_table_gets_a_refreshed_first_look_and_the_handoff_file_is_untouched(tmp_path, monkeypatch):
    root, handed = _handed_off(tmp_path, monkeypatch)
    before = handed.read_bytes()
    handed_digest = json.loads(before)["table_digest"]
    stale = _v2(root)
    assert stale["table_digest"] == handed_digest
    v1 = projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT)
    rows = projection.robot_fit_rows(capture_root=root, task_context_digest=CONTEXT)

    assert refresh.refresh_first_look(root, now=lambda: NOW) == {"status": "built"}

    assert handed.read_bytes() == before
    directory = root.joinpath(*refresh.DIRECTORY)
    built = json.loads((directory / refresh.TABLE_FILE).read_text())
    record = json.loads((directory / refresh.STATUS_FILE).read_text())
    assert record["status"] == "built" and record["table_digest"] == built["table_digest"]
    assert record["task_context_digest"] == CONTEXT and record["handoff_table_digest"] == handed_digest
    assert all("first_look_verdict" in row for row in built["rows"])
    counts = built["first_look_counts"]
    assert sum(counts.values()) == len(built["rows"]) and counts["unknown"] < len(built["rows"])
    fresh = _v2(root)
    assert fresh["table_digest"] == built["table_digest"] and fresh["counts"] == counts
    assert fresh["counts"] != stale["counts"]
    # v1 and the screening planner read the handed-off table exactly as before.
    assert projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT) == v1
    assert projection.robot_fit_rows(capture_root=root, task_context_digest=CONTEXT) == rows
    # Built and verified: never rebuilt for the same binding.
    assert refresh.refresh_first_look(root, now=lambda: NOW + 7200) == {"status": "current"}
    assert json.loads((directory / refresh.STATUS_FILE).read_text()) == record


def test_a_handoff_table_with_the_first_look_is_never_refreshed(tmp_path, monkeypatch):
    root, handed = _handed_off(tmp_path, monkeypatch, old=False)
    before = handed.read_bytes()

    assert refresh.refresh_first_look(root, now=lambda: NOW) == {"status": "not_needed"}

    assert not root.joinpath(*refresh.DIRECTORY).exists() and handed.read_bytes() == before
    assert _v2(root)["table_digest"] == json.loads(before)["table_digest"]


@pytest.mark.parametrize("field", ["task_context_digest", "handoff_table_digest", "preparation_digest"])
def test_a_refreshed_table_bound_to_anything_else_is_ignored(tmp_path, monkeypatch, field):
    root, handed = _handed_off(tmp_path, monkeypatch)
    assert refresh.refresh_first_look(root, now=lambda: NOW)["status"] == "built"
    status = root.joinpath(*refresh.DIRECTORY, refresh.STATUS_FILE)
    record = json.loads(status.read_text())
    record[field] = "sha256:" + "d" * 64
    record["digest"] = canonical_digest(record, digest_field="digest")
    write_json(status, record)

    assert _v2(root)["table_digest"] == json.loads(handed.read_text())["table_digest"]


def test_a_refreshed_table_that_does_not_verify_is_ignored(tmp_path, monkeypatch):
    root, handed = _handed_off(tmp_path, monkeypatch)
    assert refresh.refresh_first_look(root, now=lambda: NOW)["status"] == "built"
    table = root.joinpath(*refresh.DIRECTORY, refresh.TABLE_FILE)
    value = json.loads(table.read_text())
    value["rows"][0]["first_look_verdict"] = "fits"  # changed after it was sealed
    write_json(table, value)

    assert _v2(root)["table_digest"] == json.loads(handed.read_text())["table_digest"]


def test_a_failed_build_is_recorded_and_retried_after_the_hour(tmp_path, monkeypatch):
    from blueprint_pipeline import website_robot_fit

    root, handed = _handed_off(tmp_path, monkeypatch)
    real = website_robot_fit.robot_fit_table

    def broken(**_):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(website_robot_fit, "robot_fit_table", broken)
    assert refresh.refresh_first_look(root, now=lambda: NOW) == {"status": "failed"}
    record = json.loads(root.joinpath(*refresh.DIRECTORY, refresh.STATUS_FILE).read_text())
    assert record["status"] == "failed" and record["error"] == "RuntimeError" and record["table_digest"] is None
    assert record["attempted_at_epoch"] == int(NOW)
    assert _v2(root)["table_digest"] == json.loads(handed.read_text())["table_digest"]
    monkeypatch.setattr(website_robot_fit, "robot_fit_table", real)
    assert refresh.refresh_first_look(root, now=lambda: NOW + refresh.RETRY_SECONDS - 1) == {
        "status": "waiting_retry"}
    assert refresh.refresh_first_look(root, now=lambda: NOW + refresh.RETRY_SECONDS) == {"status": "built"}
    assert _v2(root)["table_digest"] != json.loads(handed.read_text())["table_digest"]


def test_a_missing_input_is_recorded_by_name_and_builds_nothing(tmp_path, monkeypatch):
    root, handed = _handed_off(tmp_path, monkeypatch)
    collision = Path(json.loads((handed.parent / "runtime_inputs.json").read_text())["collision"]["path"])
    collision.unlink()

    assert refresh.refresh_first_look(root, now=lambda: NOW) == {"status": "failed"}

    record = json.loads(root.joinpath(*refresh.DIRECTORY, refresh.STATUS_FILE).read_text())
    assert record["error"] == "ValueError" and record["table_digest"] is None
    assert not root.joinpath(*refresh.DIRECTORY, refresh.TABLE_FILE).exists()


def test_a_withdrawn_capture_is_skipped(tmp_path, monkeypatch):
    root, handed = _handed_off(tmp_path, monkeypatch)
    monkeypatch.setattr(refresh, "capture_withdrawn", lambda capture: True)

    assert refresh.refresh_first_look(root, now=lambda: NOW) == {"status": "withdrawn"}
    assert not root.joinpath(*refresh.DIRECTORY).exists()


def test_the_pass_counts_each_capture_and_one_failure_stops_no_other(tmp_path, monkeypatch):
    root, _handed = _handed_off(tmp_path, monkeypatch)
    other = root.parent / "walkthrough-req1-broken"
    write_json(other / "raw" / "manifest.json", {})
    real = refresh.refresh_first_look

    def flaky(capture, **options):
        if capture.name != root.name:
            raise OSError("synthetic")
        return real(capture, **options)

    monkeypatch.setattr("blueprint_pipeline.website_company_clip.capture_roots", lambda storage: [other, root])
    monkeypatch.setattr(refresh, "refresh_first_look", flaky)

    assert refresh.tick(tmp_path / "storage", now=lambda: NOW) == {"error": 1, "built": 1}
    assert refresh.tick(tmp_path / "storage", stop_reason=lambda: "quiesce") == {
        "deferred": 2, "stopped_quiesce": 1}


def test_the_preview_pass_runs_the_refresh_and_never_raises(tmp_path, monkeypatch):
    from blueprint_pipeline import website_atlas_preview as preview

    root, _handed = _handed_off(tmp_path, monkeypatch)
    assert preview.first_look_refresh_tick(tmp_path / "storage") == {"built": 1}
    monkeypatch.setattr(refresh, "tick", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert preview.first_look_refresh_tick(tmp_path / "storage") == {"tick_failed": 1}
