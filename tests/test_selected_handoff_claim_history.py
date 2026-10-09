"""Actual reader/claim functions with a fake lock; no provider or real flock."""

import ast
import json
from pathlib import Path

import pytest

from tests.test_selected_handoff_recovery import claim_function


def read_ledger(capture_root):
    source = Path(__file__).parents[1] / "src/blueprint_pipeline/handoff_job_state.py"
    node = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_read_job_ledger"
    )
    namespace = {
        "json": json,
        "JOB_LEDGER_FILENAME": "pipeline_job_ledger.json",
        "JOB_LEDGER_SCHEMA_VERSION": "isolated-test",
    }
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    return namespace["_read_job_ledger"](capture_root)


@pytest.mark.parametrize("existing", ["empty_object", "dangling_link", "directory"])
def test_selected_first_attempt_refuses_existing_unreadable_history_under_lock(tmp_path, existing):
    ledger_path = tmp_path / "pipeline_job_ledger.json"
    if existing == "empty_object":
        ledger_path.write_text("{}")
    elif existing == "dangling_link":
        ledger_path.symlink_to(tmp_path / "missing")
    else:
        ledger_path.mkdir()
    ledger = read_ledger(tmp_path)
    assert ledger == {}  # The actual existing reader reports no usable mapping.
    claim = claim_function(ledger)
    claim.__globals__["JOB_LEDGER_FILENAME"] = "pipeline_job_ledger.json"
    status, after = claim(
        tmp_path,
        scene_id="site-fixture",
        capture_id="walkthrough-fixture",
        owner="fixture",
        lease_seconds=30,
        require_unattempted_delivery=True,
    )
    assert status == "prior_delivery_effects_unresolved"
    assert after == ledger == {}
