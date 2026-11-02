"""ARTISAN's real output shapes, pinned so the glue is tested against them, not hand-made guesses.

capture-447ae4f4 (2026-11-02): every hand-made fixture said phase "DONE"; the
real bridge writes "done", so the first finished run was refused. The shapes
here were observed on that job's stopped and finished runs (scrubbed: keys and
enum values only). When upstream changes, update the fixture from a real run.
"""
import json
from pathlib import Path

import pytest

from blueprint_pipeline.task_object_artisan_exports import UPSTREAM_DONE_PHASE, upstream_exports
from blueprint_pipeline.task_object_authoring_contracts import AuthoringContractError

from .test_task_object_artisan_exports import export_fixture

SHAPES = json.loads((Path(__file__).parent / "fixtures/artisan_upstream_observed_shapes.v1.json").read_text())
FINISHED, STOPPED = SHAPES["terminals"]["finished_registered"], SHAPES["terminals"]["stopped_in_review"]
#: Terminal, manifest and report fields the glue reads (task_object_artisan_exports.upstream_exports,
#: the artisan driver and task_evaluation_artisan_continuation._source_state).
TERMINAL_READ = {"phase", "status", "registered", "returncode", "canonical_record_id", "candidate_id",
                 "reviewed_physics_report_sha256", "paths", "termination_reason", "run_id", "schema_version"}
TERMINAL_PATHS_READ = {"root", "exports", "candidate_manifest", "model", "blueprint", "physics_report"}
MANIFEST_READ = {"schema_version", "status", "candidate_id", "source_manifest_sha256", "model_sha256",
                 "blueprint_sha256", "artifacts", "formats", "physics_report_path", "physics_report_sha256"}


def test_every_field_the_glue_reads_exists_upstream():
    assert TERMINAL_READ <= set(SHAPES["terminal_keys"])
    assert TERMINAL_PATHS_READ <= set(SHAPES["terminal_path_keys"])
    assert MANIFEST_READ <= set(SHAPES["export_manifest"]["keys"])
    assert set(SHAPES["export_manifest"]["artifact_row_keys"]) == {"path", "sha256", "size_bytes"}
    assert "candidate_id" in SHAPES["physics_report_keys"]
    assert SHAPES["export_manifest"]["schema_version"] == "release-candidate-v7"
    assert SHAPES["export_manifest"]["status"] == "accepted" and "usd" in SHAPES["export_manifest"]["formats"]


def test_the_finished_phase_is_the_value_the_bridge_writes():
    assert FINISHED["phase"] == UPSTREAM_DONE_PHASE
    assert STOPPED["phase"] != UPSTREAM_DONE_PHASE


@pytest.mark.parametrize("shape,accepted", [(FINISHED, True), (STOPPED, False)])
def test_the_export_accepts_exactly_a_real_finished_terminal(tmp_path, shape, accepted):
    terminal, _manifest = export_fixture(tmp_path)
    terminal.update({key: shape[key] for key in ("phase", "status", "registered", "returncode")})
    if not shape["canonical_record_id_present"]:
        terminal["canonical_record_id"] = None
    if accepted:
        assert upstream_exports(terminal, tmp_path)["usd"]["path"] == str(tmp_path / "exports/object.usd")
    else:
        with pytest.raises(AuthoringContractError, match="artisan_not_done_and_registered"):
            upstream_exports(terminal, tmp_path)
