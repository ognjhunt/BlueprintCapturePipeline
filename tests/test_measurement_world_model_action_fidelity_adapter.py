from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_execution_request,
)
from blueprint_pipeline.measurement_adapter_runtime import build_measurement_adapter_descriptor
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)
from blueprint_pipeline.measurement_world_model_action_fidelity_adapter import (
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    implementation_digest,
    run_world_model_action_fidelity_request,
)
from blueprint_pipeline.measurement_world_model_action_fidelity_suite import (
    WorldModelActionFidelitySuiteError,
    run_world_model_action_fidelity_development_suite,
    validate_world_model_action_fidelity_development_suite,
)


ROOT = Path(__file__).parents[1]
CORPUS = ROOT / "tests/fixtures/measurement_world_model_action_fidelity_v1/corpus.json"
CORPUS_SCHEMA = ROOT / "docs/schemas/world_model_action_fidelity_development_corpus.v1.schema.json"
SUITE_SCHEMA = ROOT / "docs/schemas/world_model_action_fidelity_development_suite.v1.schema.json"
Q = "sha256:" + "f" * 64
C = "sha256:" + "d" * 64


def _corpus() -> dict:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


def _digest() -> str:
    return "sha256:" + hashlib.sha256(CORPUS.read_bytes()).hexdigest()


def _spec() -> dict:
    return build_qualification_benchmark_spec(
        benchmark_id="world-model-action-fidelity",
        benchmark_version="development-contract-1",
        method_ids=["gigaworld-wmbench"],
        development_split_digest=_digest(),
        qualification_split_digest=Q,
        capture_bundle_digests=[_digest()],
        robot_controller_digests=[C],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 1.0,
        },
        compute_budget={"usd": 0.0},
    )


def _request(index: int = 0) -> dict:
    corpus = _corpus()
    row = dict(corpus["cases"][index])
    case_id = row.pop("case_id")
    case = build_benchmark_case_manifest(
        _spec(),
        case_id=case_id,
        split="development",
        input_artifact_digests=[_digest()],
        task_class="long_horizon_task_execution",
        material_regime="none",
        operating_point={**corpus["shared_operating_point"], **row},
    )
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("gigaworld-wmbench"),
        _spec(),
        case,
        execution_id=f"wm-fidelity-{index}",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="blueprint-strict-wam-action-consistency",
        precision="float64",
        seed=41,
        solver_settings={"protocol": "world_model_action_fidelity.v1", "replay_count": 2},
        timeout_seconds=30,
    )


def _rehash(value: dict, field: str) -> None:
    value.pop(field, None)
    value[field] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def test_world_model_corpus_schema_and_claim_ceiling() -> None:
    corpus = _corpus()
    jsonschema.validate(corpus, json.loads(CORPUS_SCHEMA.read_text(encoding="utf-8")))
    assert corpus["shared_operating_point"]["claim_scope"] == "evaluator_support_only"
    assert (
        corpus["shared_operating_point"]["historical_policy_ranking_verdict"]
        == "thesis_not_supported"
    )
    assert corpus["physical_outcomes_included"] is False
    assert corpus["policy_ranking_labels_included"] is False


def test_world_model_action_fidelity_suite_executes_both_outcomes() -> None:
    completed = run_world_model_action_fidelity_development_suite(
        CORPUS, qualification_split_digest=Q, controller_scope_digest=C, execute=True
    )
    assert completed["status"] == "completed_development_only"
    assert completed["all_replays_deterministic"] is True
    assert completed["aggregate_metrics"]["within_envelope_case_count"] == 1
    assert completed["aggregate_metrics"]["policy_ranking_case_count"] == 0
    assert [row["observed_metrics"]["task_outcome"] for row in completed["cases"]] == [
        "within_action_fidelity_envelope",
        "action_fidelity_envelope_exceeded",
    ]
    assert completed["historical_policy_ranking_verdict"] == "thesis_not_supported"
    for key in (
        "policy_ranking_scored",
        "physics_authority",
        "physical_success_established",
        "r5_evidence",
        "r6_decision",
        "r7_admission",
        "production_route_eligible",
        "agent_may_promote",
    ):
        assert completed[key] is False
    jsonschema.validate(completed, json.loads(SUITE_SCHEMA.read_text(encoding="utf-8")))


def test_world_model_worker_rejects_physical_labels_and_replay_reuse() -> None:
    physical = copy.deepcopy(_request())
    physical["case_manifest"]["operating_point"]["physical_outcomes_included"] = True
    _rehash(physical["case_manifest"], "case_manifest_digest")
    physical.pop("execution_request_digest")
    with pytest.raises(
        MeasurementAdapterExecutionError, match="physical_outcomes_included_invalid"
    ):
        run_world_model_action_fidelity_request(physical)

    reused = copy.deepcopy(_request())
    steps = reused["case_manifest"]["operating_point"]["action_steps"]
    steps[1]["motion_identity"] = steps[0]["motion_identity"]
    _rehash(reused["case_manifest"], "case_manifest_digest")
    reused.pop("execution_request_digest")
    result = run_world_model_action_fidelity_request(reused)
    assert result["status"] == "failed"
    assert any("generated_motion_reused" in code for code in result["failure_codes"])


def test_world_model_suite_rejects_split_and_thesis_upgrade() -> None:
    with pytest.raises(WorldModelActionFidelitySuiteError, match="split_leakage"):
        run_world_model_action_fidelity_development_suite(
            CORPUS, qualification_split_digest=_digest(), controller_scope_digest=C
        )
    planned = run_world_model_action_fidelity_development_suite(
        CORPUS, qualification_split_digest=Q, controller_scope_digest=C
    )
    planned["historical_policy_ranking_verdict"] = "thesis_supported"
    with pytest.raises(
        WorldModelActionFidelitySuiteError, match="historical_policy_ranking_verdict_invalid"
    ):
        validate_world_model_action_fidelity_development_suite(planned)


def test_world_model_worker_cannot_inherit_oscar_identity() -> None:
    request = _request()
    request["implementation"]["implementation_id"] = "oscar-world-model"
    request.pop("execution_request_digest")
    result = run_world_model_action_fidelity_request(request)
    assert result["status"] == "blocked"
    assert result["failure_codes"] == ["world_model_fidelity_implementation_id_mismatch"]
