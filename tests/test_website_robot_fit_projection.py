"""The site reads its job's robot fit table through preparation status v5, never an unbound one."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/website_robot_fit_projection.py
#   src/blueprint_pipeline/website_preparation_status.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline import website_robot_fit_projection as projection
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest

CONTEXT = "sha256:" + "c" * 64


def _row(name, verdict, results):
    return {"model_id": name.lower().replace(" ", "_"), "recipe_id": None, "name": name, "manufacturer": "Maker",
            "body_type": "fixed_arm", "verdict": verdict, "reasons": [],
            "checks": [{"check": check, "result": result, "reason": "", "values": {}}
                       for check, result in results.items()]}


ROWS = [
    _row("Arm Unknown", "unknown", {"stance": "pass", "reach": "unknown", "payload": "unknown",
                                    "cycle_time": "not_requested"}),
    _row("Arm Short", "doesnt_fit", {"stance": "pass", "reach": "fail", "payload": "pass"}),
    _row("Arm Spec", "fits_spec", {"stance": "pass", "reach": "pass", "payload": "pass", "access": "unknown"}),
    _row("Arm Path", "fits", {"stance": "pass", "reach": "pass", "payload": "pass"}),
]


def capture_root(tmp_path: Path, scene_id: str = "site-req1", capture_id: str = "walkthrough-req1") -> Path:
    root = tmp_path / "storage" / "scenes" / scene_id / "captures" / capture_id
    root.mkdir(parents=True)
    return root


def write_fit_table(root: Path, *, route: str = "full", rows=ROWS, context: str = CONTEXT,
                    handoff: dict | None = None) -> dict:
    """A handoff with its preparation and fit table, as the scene handoff leaves them."""
    work = root / "pipeline" / "website_scene_preparation"
    preparation = {"schema_version": "website_scene_preparation.v1", "binding": {"task_context_digest": context}}
    preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    write_json(work / "preparation.json", preparation)
    table = {"schema_version": "website_robot_fit_table.v1", "preparation_digest": preparation["digest"],
             "evidence_label": "spec_bounded_estimate", "claim_ceiling": "development_only",
             "physical_world_truth_claimed": False, "rows": rows,
             "counts": {verdict: sum(row["verdict"] == verdict for row in rows) for verdict in projection.VERDICTS}}
    table["table_digest"] = canonical_digest(table, digest_field="table_digest")
    table_path = work / ("fit_check" if route == "geometry_only" else "native") / "robot_fit_table.json"
    write_json(table_path, table)
    reference = {"path": str(table_path), "digest": table["table_digest"], "counts": table["counts"]}
    value = {"schema_version": "website_scene_handoff.v1", "status": "intake_ready", "blockers": [],
             "scene_id": root.parent.parent.name, "capture_id": root.name,
             "preparation_path": str(work / "preparation.json"), "preparation_digest": preparation["digest"],
             **({"atlas_route": "geometry_only", "fit_check": {"status": "completed", "robot_fit_table": reference}}
                if route == "geometry_only" else {"robot_fit_table": reference}),
             **(handoff or {})}
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(work / "handoff.json", value)
    return value


def test_the_site_reads_the_full_routes_fit_table_with_the_robots_that_fit_first(tmp_path):
    root = capture_root(tmp_path)
    handoff = write_fit_table(root)
    value = projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT)
    table = json.loads(Path(handoff["robot_fit_table"]["path"]).read_text())
    assert value == {
        "schema_version": "website_robot_fit_projection.v1", "route": "full", "status": "completed", "reason": None,
        "evidence_label": "spec_bounded_estimate",
        "counts": {"fits": 1, "fits_spec": 1, "doesnt_fit": 1, "unknown": 1},
        # Only what the spec sheets and the scene could not settle, or ruled out, is named.
        "models": [{"name": "Arm Path", "manufacturer": "Maker", "verdict": "fits", "open_checks": []},
                   {"name": "Arm Spec", "manufacturer": "Maker", "verdict": "fits_spec", "open_checks": ["access"]},
                   {"name": "Arm Short", "manufacturer": "Maker", "verdict": "doesnt_fit", "open_checks": ["reach"]},
                   {"name": "Arm Unknown", "manufacturer": "Maker", "verdict": "unknown",
                    "open_checks": ["reach", "payload"]}],
        "models_truncated": False, "table_digest": table["table_digest"]}


def test_a_geometry_only_job_reads_its_fit_check(tmp_path):
    root = capture_root(tmp_path)
    write_fit_table(root, route="geometry_only")
    value = projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT)
    assert value["route"] == "geometry_only" and value["status"] == "completed" and len(value["models"]) == 4


@pytest.mark.parametrize("reason, expected", [("task_family_not_modelled", True), ("subject_not_excised", True),
                                              ("arbitrary_error", False)])
def test_a_geometry_only_check_that_did_not_run_says_why(tmp_path, reason, expected):
    root = capture_root(tmp_path)
    write_fit_table(root, route="geometry_only", handoff={"fit_check": {"status": "not_run", "reason": reason}})
    value = projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT)
    if not expected:
        assert value is None
        return
    assert value == {"schema_version": "website_robot_fit_projection.v1", "route": "geometry_only",
                     "status": "not_run", "reason": reason, "evidence_label": None, "counts": None, "models": [],
                     "models_truncated": False, "table_digest": None}


@pytest.mark.parametrize("fault", ["edited_table", "reference_digest", "other_context", "other_preparation",
                                   "outside_capture", "symlink", "physical_truth", "unknown_verdict",
                                   "other_capture", "unknown_route", "no_table"])
def test_a_fit_table_that_does_not_bind_to_this_job_is_left_out(tmp_path, fault):
    root = capture_root(tmp_path)
    if fault == "other_context":
        write_fit_table(root, context="sha256:" + "d" * 64)
    elif fault == "unknown_verdict":
        write_fit_table(root, rows=[_row("Arm", "probably", {"reach": "pass"})])
    elif fault == "other_capture":
        write_fit_table(root, handoff={"capture_id": "walkthrough-other"})
    elif fault == "unknown_route":
        write_fit_table(root, handoff={"atlas_route": "cheapest"})
    elif fault == "no_table":
        write_fit_table(root, handoff={"robot_fit_table": {"status": "failed", "error": "KeyError"}})
    else:
        handoff = write_fit_table(root)
        work = root / "pipeline" / "website_scene_preparation"
        table_path = Path(handoff["robot_fit_table"]["path"])
        table = json.loads(table_path.read_text())
        if fault == "edited_table":
            table["rows"][0]["verdict"] = "fits"
            write_json(table_path, table)
        elif fault == "physical_truth":
            table["physical_world_truth_claimed"] = True
            table["table_digest"] = canonical_digest(table, digest_field="table_digest")
            write_json(table_path, table)
        elif fault == "other_preparation":
            table["preparation_digest"] = "sha256:" + "e" * 64
            table["table_digest"] = canonical_digest(table, digest_field="table_digest")
            write_json(table_path, table)
        else:
            reference = dict(handoff["robot_fit_table"])
            if fault == "reference_digest":
                reference["digest"] = "sha256:" + "f" * 64
            elif fault == "outside_capture":
                outside = tmp_path / "elsewhere.json"
                write_json(outside, table)
                reference["path"] = str(outside)
            elif fault == "symlink":
                link = work / "linked.json"
                link.symlink_to(table_path)
                reference["path"] = str(link)
            handoff["robot_fit_table"] = reference
            handoff["digest"] = canonical_digest(handoff, digest_field="digest")
            write_json(work / "handoff.json", handoff)
    assert projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT) is None


def test_no_handoff_yet_projects_nothing(tmp_path):
    assert projection.read_robot_fit(capture_root=capture_root(tmp_path), task_context_digest=CONTEXT) is None


def test_a_long_table_is_cut_to_its_budget_keeping_the_robots_that_fit(tmp_path):
    rows = ([_row(f"Unknown Arm {index:02d}", "unknown", {"reach": "unknown", "payload": "unknown",
                                                          "access": "unknown"}) for index in range(70)]
            + [_row(f"Fitting Arm {index:02d}", "fits", {"reach": "pass"}) for index in range(10)])
    root = capture_root(tmp_path)
    write_fit_table(root, rows=rows)
    value = projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT)
    assert value["models_truncated"] is True
    assert len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()) <= projection.MAX_BYTES
    assert [row["verdict"] for row in value["models"][:10]] == ["fits"] * 10
    # The counts stay those of the whole table.
    assert value["counts"] == {"fits": 10, "fits_spec": 0, "doesnt_fit": 0, "unknown": 70}


def test_every_projection_matches_the_published_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).resolve().parents[1]
                         / "docs/schemas/website_robot_fit_projection.v1.schema.json").read_text())
    completed = capture_root(tmp_path / "completed")
    write_fit_table(completed)
    not_run = capture_root(tmp_path / "not_run")
    write_fit_table(not_run, route="geometry_only",
                    handoff={"fit_check": {"status": "not_run", "reason": "task_family_not_modelled"}})
    for root in (completed, not_run):
        jsonschema.validate(projection.read_robot_fit(capture_root=root, task_context_digest=CONTEXT), schema)
