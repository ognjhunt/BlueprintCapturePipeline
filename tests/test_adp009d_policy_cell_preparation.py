from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from blueprint_pipeline.adp009d_policy_cell_preparation import (
    CONTROLS_BLOCKER,
    materialize_policy_cell_matrix,
    prepare_policy_cell_matrix,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.dual_task_scenario_suite import COUSIN_BLOCKER


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "docs/arm_decision_proof_v1/manifests"
SUITE = MANIFESTS / "third_scene_840920_task_a_scenario_suite.v1.json"
READINESS = MANIFESTS / "adp009d_scene_840920_policy_readiness.v1.json"


def _inputs() -> tuple[dict, dict]:
    return (
        json.loads(SUITE.read_text(encoding="utf-8")),
        json.loads(READINESS.read_text(encoding="utf-8")),
    )


def test_committed_seven_cells_prepare_identical_candidate_inputs() -> None:
    suite, readiness = _inputs()
    receipt = prepare_policy_cell_matrix(
        scenario_suite=suite,
        policy_readiness=readiness,
    )

    assert receipt["cell_count"] == 7
    assert receipt["candidate_cell_count"] == 14
    assert receipt["provider_free_scenario_instances_materialized"] == 7
    assert receipt["native_packets_materialized"] == 0
    assert receipt["policy_execution_specs_materialized"] == 0
    assert receipt["executable_candidate_cells_before_controls"] == 0
    assert receipt["provider_mutation_performed"] is False
    assert receipt["paid_resource_allocation_performed"] is False
    assert receipt["learned_policy_outcomes_consulted"] is False
    assert receipt["materialization_digest"] == canonical_digest(
        receipt, digest_field="materialization_digest"
    )

    for cell in receipt["cells"]:
        left, right = cell["candidate_runs"]
        assert [left["candidate_id"], right["candidate_id"]] == [
            "pi05_droid",
            "groot_n17_droid",
        ]
        assert left["cell_id"] == right["cell_id"] == cell["cell_id"]
        assert left["seed"] == right["seed"] == cell["seed"]
        assert left["scenario_instance_digest"] == (
            right["scenario_instance_digest"]
        )
        assert left["blockers"] == right["blockers"] == cell["blockers"]
        instance = cell["scenario_instance"]
        assert instance["instance_digest"] == canonical_digest(
            instance, digest_field="instance_digest"
        )
        assert instance["caller_asserted_success"] is False
        assert instance["learned_policy_outcomes_consulted"] is False

    cousin = next(
        cell for cell in receipt["cells"] if cell["family"] == "admitted_object_cousin"
    )
    assert cousin["blockers"] == [COUSIN_BLOCKER]
    assert all(
        run["status"] == "blocked_before_packet_materialization"
        for run in cousin["candidate_runs"]
    )
    ordinary = [cell for cell in receipt["cells"] if cell is not cousin]
    assert len(ordinary) == 6
    assert all(cell["blockers"] == [CONTROLS_BLOCKER] for cell in ordinary)
    assert all(
        run["status"] == "waiting_for_cell_specific_controls"
        for cell in ordinary
        for run in cell["candidate_runs"]
    )


def test_preparation_refuses_changed_seed_or_unvalidated_rights() -> None:
    suite, readiness = _inputs()
    suite["cells"][0]["seed"] += 1
    with pytest.raises(ValueError, match="dual_task_scenario_suite_digest_invalid"):
        prepare_policy_cell_matrix(
            scenario_suite=suite,
            policy_readiness=readiness,
        )

    suite, readiness = _inputs()
    readiness["candidates"][0]["rights_ready"] = False
    readiness["readiness_digest"] = canonical_digest(
        readiness, digest_field="readiness_digest"
    )
    with pytest.raises(ValueError, match="rights_ready_invalid"):
        prepare_policy_cell_matrix(
            scenario_suite=suite,
            policy_readiness=readiness,
        )


def test_provider_free_cli_materializes_once_and_refuses_overwrite(tmp_path) -> None:
    output = tmp_path / "policy-cell-preparation.v1.json"
    command = [
        sys.executable,
        "scripts/materialize_adp009d_policy_cell_preparation.py",
        "--scenario-suite",
        str(SUITE),
        "--policy-readiness",
        str(READINESS),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status"] == "materialized"
    assert summary["cell_count"] == 7
    assert summary["candidate_cell_count"] == 14
    assert summary["executable_candidate_cells_before_controls"] == 0
    assert summary["provider_mutation_performed"] is False
    assert summary["paid_resource_allocation_performed"] is False
    assert json.loads(output.read_text(encoding="utf-8"))[
        "materialization_digest"
    ] == summary["materialization_digest"]

    repeated = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert repeated.returncode == 2
    assert "policy_cell_preparation_output_exists" in repeated.stdout


def test_materializer_rejects_non_mapping_inputs(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="policy_cell_preparation_input_invalid"):
        materialize_policy_cell_matrix(
            scenario_suite_path=bad,
            policy_readiness_path=READINESS,
            output_path=tmp_path / "must-not-exist.json",
        )
