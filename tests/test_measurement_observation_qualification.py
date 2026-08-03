from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.measurement_adapter_execution import (
    validate_measurement_adapter_execution_receipt,
)
from blueprint_pipeline.measurement_adapter_runtime import (
    build_measurement_adapter_descriptor,
)
from blueprint_pipeline.measurement_observation_qualification import (
    LIGHTING_CHALLENGES,
    MATERIAL_CHALLENGES,
    MeasurementObservationQualificationError,
    build_observation_challenge_case,
    build_observation_qualification_scope,
    build_observation_r5_candidate_stage_data,
    evaluate_observation_challenge_matrix,
    validate_observation_challenge_case,
    validate_observation_challenge_report,
    validate_observation_qualification_scope,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    MeasurementBenchmarkError,
    build_benchmark_case_manifest,
    build_benchmark_prediction,
    build_qualification_benchmark_spec,
    build_r5_stage_data,
    build_sealed_physical_label,
    evaluate_qualification_benchmark,
)
from blueprint_pipeline.measurement_sensor_stream_pairing import (
    build_sensor_stream_pairing_record,
)


ROOT = Path(__file__).parents[1]


def _digest(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _pairing() -> dict:
    modalities = ("rgb", "depth", "lidar")
    sensors = {modality: f"sensor-{modality}" for modality in modalities}
    streams = []
    for index, modality in enumerate(modalities):
        stream = {
            "sensor_id": sensors[modality],
            "modality": modality,
            "stream_digest": _digest(f"stream-{modality}"),
            "calibration_digest": _digest(f"calibration-{modality}"),
            "extrinsics_digest": _digest(f"extrinsics-{modality}"),
            "clock_domain": "site-ptp",
            "time_offset_ns": -index * 100,
            "timestamp_uncertainty_ns": 10,
            "independent_calibration": {
                "evaluator_id": f"calibration-lab-{modality}",
                "candidate_method_independent": True,
                "agent_is_evaluator": False,
                "signature_status": "verified",
                "approval_signature_id": f"signature-{modality}",
            },
            "samples": [
                {
                    "sample_id": f"{modality}-{replicate}",
                    "timestamp_ns": replicate * 1_000_000 + index * 100,
                    "artifact_digest": _digest(f"{modality}-{replicate}"),
                }
                for replicate in (1, 2)
            ],
        }
        if modality in {"rgb", "depth"}:
            stream["intrinsics_digest"] = _digest(f"intrinsics-{modality}")
        streams.append(stream)
    return build_sensor_stream_pairing_record(
        {
            "schema_version": "measurement_sensor_stream_pairing.v1",
            "pairing_id": "observation-qualification-pairing",
            "source_capture_digest": _digest("capture"),
            "site_frame_id": "site-frame",
            "clock_domain": "site-ptp",
            "required_modalities": list(modalities),
            "maximum_pair_delta_ns": 100,
            "streams": streams,
            "pair_groups": [
                {
                    "group_id": f"pair-{replicate}",
                    "samples": {
                        modality: {
                            "sensor_id": sensors[modality],
                            "sample_id": f"{modality}-{replicate}",
                        }
                        for modality in modalities
                    },
                }
                for replicate in (1, 2)
            ],
            "physical_measurements_included": True,
            "development_only": False,
            "candidate_self_graded": False,
            "agent_generated_calibration": False,
            "thresholds_modified_after_measurement": False,
            "agent_may_approve": False,
            "q_sensor_qualification_created": False,
            "r5_evidence_created": False,
            "r6_decision_created": False,
            "r7_admission_created": False,
            "physical_success_established": False,
        }
    )


def _spec() -> dict:
    return build_qualification_benchmark_spec(
        benchmark_id="capture-to-observation",
        benchmark_version="site-task-observation-1",
        method_ids=[
            "direct-captured-observations",
            "isaac-rtx-openusd-sensor-path",
        ],
        development_split_digest=_digest("development-split"),
        qualification_split_digest=_digest("qualification-split"),
        capture_bundle_digests=[_digest("capture")],
        robot_controller_digests=[_digest("controller")],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 0.2,
            "maximum_mismatch_rate": 0.1,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 1.0,
        },
        compute_budget={"usd": 100.0, "maximum_duration_seconds": 3600},
        minimum_repeated_trials=2,
    )


def _scope() -> dict:
    return build_observation_qualification_scope(
        _spec(),
        _pairing(),
        task_id="task-transparent-bin-pick",
        site_id="site-assembly-a",
        task_request_digest=_digest("task-request"),
        site_evidence_profile_digest=_digest("site-evidence-profile"),
        policy_checkpoints={
            f"policy-{index}": _digest(f"policy-checkpoint-{index}") for index in range(1, 4)
        },
    )


def _trials(scope: dict, *, repeats: int = 2) -> list[dict]:
    return [
        {
            "policy_id": policy_id,
            "policy_digest": policy_digest,
            "replicate_id": f"replicate-{replicate}",
            "physical_outcome_digest": _digest(f"physical-outcome-{policy_id}-{replicate}"),
            "synthetic_outcome_digests": {
                method_id: _digest(f"synthetic-outcome-{method_id}-{policy_id}-{replicate}")
                for method_id in scope["synthetic_method_ids"]
            },
        }
        for policy_id, policy_digest in scope["policy_checkpoints"].items()
        for replicate in range(1, repeats + 1)
    ]


def _challenge_case(
    scope: dict,
    *,
    case_id: str = "all-challenges",
    split: str = "qualification",
    materials: list[str] | None = None,
    lighting: list[str] | None = None,
    repeats: int = 2,
) -> dict:
    physical = {
        modality: _digest(f"physical-{case_id}-{modality}")
        for modality in scope["required_modalities"]
    }
    synthetic = {
        method_id: {
            modality: _digest(f"synthetic-{case_id}-{method_id}-{modality}")
            for modality in scope["required_modalities"]
        }
        for method_id in scope["synthetic_method_ids"]
    }
    return build_observation_challenge_case(
        scope,
        case_id=case_id,
        split=split,
        material_challenges=materials or sorted(MATERIAL_CHALLENGES),
        lighting_challenges=lighting or ["controlled"],
        physical_observation_artifacts=physical,
        synthetic_observation_artifacts=synthetic,
        policy_trials=_trials(scope, repeats=repeats),
    )


def _execution_receipt(descriptor: dict, case: dict, index: int) -> dict:
    receipt = {
        "schema_version": "measurement_adapter_execution_receipt.v1",
        "execution_id": f"observation-qualification-{index}",
        "execution_request_digest": _digest(f"request-{index}"),
        "candidate_id": descriptor["candidate_id"],
        "adapter_descriptor_digest": descriptor["adapter_descriptor_digest"],
        "benchmark_spec_digest": case["benchmark_spec_digest"],
        "case_manifest_digest": case["case_manifest_digest"],
        "split": case["split"],
        "status": "completed",
        "evidence_class": "independent_qualification_execution",
        "executor_id": "independent-observation-lab",
        "executor_independent_of_candidate": True,
        "clean_environment_verified": True,
        "immutable_runtime_identity_verified": True,
        "command_digest": _digest("command"),
        "command_executable": "observation-worker",
        "command_argc": 4,
        "started_at": "2026-08-02T12:00:00+00:00",
        "finished_at": "2026-08-02T12:00:01+00:00",
        "duration_seconds": 1.0,
        "exit_code": 0,
        "worker_result_digest": _digest(f"worker-{index}"),
        "stdout_digest": _digest(f"stdout-{index}"),
        "stdout_bytes": 0,
        "stdout_content_persisted": False,
        "stderr_digest": _digest(f"stderr-{index}"),
        "stderr_bytes": 0,
        "stderr_content_persisted": False,
        "runtime_observations": {"fixture": True},
        "host_runtime": {"fixture": True},
        "failure_codes": [],
        "host_process_isolation_only": False,
        "network_isolation_verified": True,
        "filesystem_isolation_verified": True,
        "secrets_persisted": False,
        "qualification_labels_accessed": False,
        "provider_spend_authorized": False,
        "physical_execution_authorized": False,
        "production_route_eligible": False,
        "r6_qualification_decision_created": False,
        "r7_catalog_admission_created": False,
        "agent_authorized": False,
    }
    receipt["execution_receipt_digest"] = _digest_payload(receipt, "execution_receipt_digest")
    return validate_measurement_adapter_execution_receipt(receipt)


def _digest_payload(value: dict, field: str) -> str:
    normalized = dict(value)
    normalized.pop(field, None)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _benchmark_report(spec: dict) -> dict:
    descriptor = build_measurement_adapter_descriptor("isaac-rtx-openusd-sensor-path")
    predictions = []
    labels = []
    for index in (1, 2):
        case = build_benchmark_case_manifest(
            spec,
            case_id=f"qualification-observation-{index}",
            split="qualification",
            input_artifact_digests=[_digest(f"observation-input-{index}")],
            task_class="transparent_bin_pick",
            material_regime="observation_challenge_matrix",
            operating_point={"replicate": index},
        )
        expected = {metric_id: float(index) for metric_id in spec["metric_ids"]}
        predictions.append(
            build_benchmark_prediction(
                descriptor,
                case,
                observed_metrics=expected,
                unsafe_condition_predicted=False,
                execution_receipt=_execution_receipt(descriptor, case, index),
            )
        )
        labels.append(
            build_sealed_physical_label(
                case,
                expected_metrics=expected,
                unsafe_condition_observed=False,
                physical_measurement_ids=[f"physical-measurement-{index}"],
                independent_evaluator_id="independent-observation-evaluator",
            )
        )
    return evaluate_qualification_benchmark(
        spec,
        predictions,
        labels,
        evaluator_id="independent-observation-evaluator",
        independent_execution=True,
    )


def test_scope_binds_exact_task_site_pairing_policy_and_full_challenge_taxonomy() -> None:
    scope = _scope()
    assert scope["task_id"] == "task-transparent-bin-pick"
    assert scope["site_id"] == "site-assembly-a"
    assert set(scope["required_material_challenges"]) == MATERIAL_CHALLENGES
    assert set(scope["required_lighting_challenges"]) == LIGHTING_CHALLENGES
    assert scope["required_modalities"] == ["rgb", "depth", "lidar"]
    assert scope["visual_similarity_is_primary"] is False
    assert scope["downstream_task_validity_is_primary"] is True
    assert validate_observation_qualification_scope(scope) == scope
    schema = json.loads(
        (ROOT / "docs/schemas/measurement_observation_qualification.v1.schema.json").read_text()
    )
    jsonschema.validate(scope, schema)


def test_complete_qualification_matrix_requires_paired_modalities_and_policy_repeats() -> None:
    scope = _scope()
    cases = [
        _challenge_case(scope, case_id=f"{lighting}-conditions", lighting=[lighting])
        for lighting in sorted(LIGHTING_CHALLENGES)
    ]
    report = evaluate_observation_challenge_matrix(
        scope,
        cases,
        evaluator_id="independent-observation-evaluator",
        evaluator_independent_of_candidates=True,
    )
    assert report["status"] == "qualification_matrix_ready"
    assert report["qualification_matrix_ready"] is True
    assert report["blockers"] == []
    assert len(report["coverage"]["material_lighting_pairs"]) == 21
    assert report["q_sensor_qualification_created"] is False
    assert validate_observation_challenge_report(report) == report
    schema = json.loads(
        (ROOT / "docs/schemas/measurement_observation_qualification.v1.schema.json").read_text()
    )
    for artifact in [*cases, report]:
        jsonschema.validate(artifact, schema)


def test_missing_strata_or_repeated_trials_block_matrix_without_fallback() -> None:
    scope = _scope()
    incomplete = _challenge_case(
        scope,
        materials=["opaque_control"],
        lighting=["controlled"],
        repeats=1,
    )
    report = evaluate_observation_challenge_matrix(
        scope,
        [incomplete],
        evaluator_id="independent-observation-evaluator",
        evaluator_independent_of_candidates=True,
    )
    assert report["status"] == "blocked"
    assert "observation_material_challenge_missing:transparent" in report["blockers"]
    assert "observation_lighting_challenge_missing:adverse" in report["blockers"]
    assert any("policy_repeats_insufficient" in item for item in report["blockers"])


def test_case_rejects_missing_modalities_synthetic_methods_and_tampering() -> None:
    scope = _scope()
    case = _challenge_case(scope)
    missing_modality = copy.deepcopy(case)
    missing_modality.pop("case_digest")
    missing_modality["physical_observation_artifacts"].pop("lidar")
    missing_modality["case_digest"] = _digest_payload(missing_modality, "case_digest")
    with pytest.raises(
        MeasurementObservationQualificationError,
        match="physical_artifacts_invalid",
    ):
        validate_observation_challenge_case(missing_modality, scope)

    tampered = copy.deepcopy(case)
    tampered["task_id"] = "different-task"
    with pytest.raises(
        MeasurementObservationQualificationError,
        match="task_id_mismatch",
    ):
        validate_observation_challenge_case(tampered, scope)


def test_observation_r5_requires_challenge_matrix_and_all_metrics() -> None:
    spec = _spec()
    scope = build_observation_qualification_scope(
        spec,
        _pairing(),
        task_id="task-transparent-bin-pick",
        site_id="site-assembly-a",
        task_request_digest=_digest("task-request"),
        site_evidence_profile_digest=_digest("site-evidence-profile"),
        policy_checkpoints={
            f"policy-{index}": _digest(f"policy-checkpoint-{index}") for index in range(1, 4)
        },
    )
    challenge = evaluate_observation_challenge_matrix(
        scope,
        [
            _challenge_case(scope, case_id=f"{lighting}-conditions", lighting=[lighting])
            for lighting in sorted(LIGHTING_CHALLENGES)
        ],
        evaluator_id="independent-observation-evaluator",
        evaluator_independent_of_candidates=True,
    )
    benchmark = _benchmark_report(spec)
    assert benchmark["evidence_status"] == "r5_evidence_candidate"
    with pytest.raises(
        MeasurementBenchmarkError,
        match="requires_task_site_challenge_report",
    ):
        build_r5_stage_data(benchmark)
    stage = build_observation_r5_candidate_stage_data(scope, challenge, benchmark)
    assert stage["evidence_status"] == "r5_evidence_candidate"
    assert stage["heldout_evaluation"]["task_id"] == scope["task_id"]
    assert set(stage["heldout_evaluation"]["measured_metric_ids"]) == set(
        scope["required_metric_ids"]
    )
    assert stage["q_sensor_qualification_created"] is False
    assert stage["policy_rank_fidelity_public_claim_eligible"] is False

    forged_coverage = copy.deepcopy(challenge)
    forged_coverage["coverage"]["material_lighting_pairs"].pop("transparent:adverse")
    forged_coverage["report_digest"] = _digest_payload(forged_coverage, "report_digest")
    with pytest.raises(
        MeasurementObservationQualificationError,
        match="condition_pair_coverage_invalid",
    ):
        build_observation_r5_candidate_stage_data(scope, forged_coverage, benchmark)


def test_development_matrix_and_unaccepted_pairing_cannot_upgrade_claims() -> None:
    scope = _scope()
    report = evaluate_observation_challenge_matrix(
        scope,
        [
            _challenge_case(
                scope,
                case_id=f"{lighting}-conditions",
                split="development",
                lighting=[lighting],
            )
            for lighting in sorted(LIGHTING_CHALLENGES)
        ],
        evaluator_id="independent-observation-evaluator",
        evaluator_independent_of_candidates=True,
    )
    assert report["status"] == "development_matrix_complete"
    assert report["qualification_matrix_ready"] is False

    pairing = _pairing()
    pairing.pop("pairing_digest")
    pairing["development_only"] = True
    pairing["physical_measurements_included"] = False
    pairing = build_sensor_stream_pairing_record(pairing)
    with pytest.raises(
        MeasurementObservationQualificationError,
        match="sensor_pairing_not_accepted",
    ):
        build_observation_qualification_scope(
            _spec(),
            pairing,
            task_id="task",
            site_id="site",
            task_request_digest=_digest("task"),
            site_evidence_profile_digest=_digest("site"),
            policy_checkpoints={
                f"policy-{index}": _digest(f"policy-{index}") for index in range(3)
            },
        )
