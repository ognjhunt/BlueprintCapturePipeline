"""Evaluate five realistic scenarios across BlueprintCapturePipeline capabilities."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from .common import ensure_dir, optional_read_json, utc_now_iso, write_json, write_text


MAJOR_CAPABILITY_SCENARIO_SUITE_SCHEMA_VERSION = "major_capability_scenario_suite.v1"
EVALUATION_METHOD_ID = "pass_fail_all_criteria.v1"

_MISSING = object()


_SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario_id": "capture_to_robot_eval_artifacts",
        "title": "Capture To Robot Eval Artifacts",
        "major_capability": "site/task/scenario/eval artifact packaging",
        "realistic_context": (
            "A real indoor capture is turned into buyer-usable Site, Task, Scenario, "
            "and Eval Cards while preserving raw capture authority."
        ),
        "evaluation_method_id": EVALUATION_METHOD_ID,
        "success_criteria": [
            "Raw capture evidence exists and carries workflow success criteria.",
            "Robot-eval dataset cards exist for site, task, scenario, and eval layers.",
            "Rights and proof-boundary artifacts preserve raw capture truth.",
            "No downstream artifact upgrades public or robot-readiness claims.",
        ],
        "criteria": [
            {
                "criterion_id": "raw_manifest_exists",
                "description": "Raw capture manifest is present.",
                "artifact": "raw/manifest.json",
                "check": "file_exists",
            },
            {
                "criterion_id": "raw_capture_has_success_criteria",
                "description": "Raw capture manifest records workflow success criteria.",
                "artifact": "raw/manifest.json",
                "check": "json_list_min_length",
                "field": "success_criteria",
                "min_length": 1,
            },
            {
                "criterion_id": "dataset_manifest_schema",
                "description": "Robot-eval dataset manifest uses the current schema.",
                "artifact": "pipeline/robot_eval_dataset/robot_eval_dataset_manifest.json",
                "check": "json_field_equals",
                "field": "schema_version",
                "expected": "real_site_robot_eval_dataset_manifest.v1",
            },
            {
                "criterion_id": "site_card_exists",
                "description": "Site Card exists.",
                "artifact": "pipeline/robot_eval_dataset/site_card.json",
                "check": "file_exists",
            },
            {
                "criterion_id": "task_cards_present",
                "description": "At least one Task Card exists.",
                "artifact": "pipeline/robot_eval_dataset/task_cards.json",
                "check": "json_list_min_length",
                "field": "cards",
                "min_length": 1,
            },
            {
                "criterion_id": "scenario_cards_present",
                "description": "At least one Scenario Card exists.",
                "artifact": "pipeline/robot_eval_dataset/scenario_cards.json",
                "check": "json_list_min_length",
                "field": "cards",
                "min_length": 1,
            },
            {
                "criterion_id": "eval_cards_present",
                "description": "At least one Eval Card exists.",
                "artifact": "pipeline/robot_eval_dataset/eval_cards.json",
                "check": "json_list_min_length",
                "field": "cards",
                "min_length": 1,
            },
            {
                "criterion_id": "rights_packet_accepted",
                "description": "Rights packet is accepted for package use.",
                "artifact": "pipeline/robot_eval_dataset/rights_packet.json",
                "check": "json_field_in",
                "field": "status",
                "allowed": ["accepted", "ready", "complete", "completed"],
            },
            {
                "criterion_id": "raw_capture_truth_preserved",
                "description": "Dataset proof boundary keeps raw capture authoritative.",
                "artifact": "pipeline/robot_eval_dataset/robot_eval_dataset_manifest.json",
                "check": "json_field_equals",
                "field": "proof_boundary.raw_capture_authoritative",
                "expected": True,
            },
            {
                "criterion_id": "dataset_public_claim_boundary",
                "description": "Dataset packaging does not upgrade public claims.",
                "artifact": "pipeline/robot_eval_dataset/robot_eval_dataset_manifest.json",
                "check": "json_field_equals",
                "field": "proof_boundary.public_claim_upgrade_allowed",
                "expected": False,
            },
        ],
    },
    {
        "scenario_id": "task_evaluation_run_execution",
        "title": "Task Evaluation Run Execution",
        "major_capability": "Task Evaluation Run artifact and simulator evidence",
        "realistic_context": (
            "A buyer-scoped robot-eval job runs a scenario matrix, records attempts, "
            "labels failures, and keeps simulator proof separate from real-world proof."
        ),
        "evaluation_method_id": EVALUATION_METHOD_ID,
        "success_criteria": [
            "A job request and scenario matrix exist for the same job.",
            "Execution attempts and failure labels are recorded.",
            "Policy execution evidence is present.",
            "Proof boundaries permit simulator proof without implying robot readiness.",
        ],
        "criteria": [
            {
                "criterion_id": "job_request_schema",
                "description": "Robot-eval job request uses the expected schema.",
                "artifact": "pipeline/robot_eval_jobs/{job_id}/job_request.json",
                "check": "json_field_equals",
                "field": "schema_version",
                "expected": "robot_eval_job_request.v1",
            },
            {
                "criterion_id": "scenario_matrix_rows",
                "description": "Scenario eval matrix contains at least one row.",
                "artifact": "pipeline/robot_eval_jobs/{job_id}/scenario_eval_matrix.json",
                "check": "json_list_min_length",
                "field": "runs|rows",
                "min_length": 1,
            },
            {
                "criterion_id": "attempt_trace_rows",
                "description": "Normalized attempt trace records at least one attempt.",
                "artifact": "pipeline/robot_eval_jobs/{job_id}/normalized_attempt_trace.json",
                "check": "json_list_min_length",
                "field": "attempts",
                "min_length": 1,
            },
            {
                "criterion_id": "failure_labels_present",
                "description": "Failure labels are recorded for review/evaluation.",
                "artifact": "pipeline/robot_eval_jobs/{job_id}/failure_labels.json",
                "check": "json_list_min_length",
                "field": "labels",
                "min_length": 1,
            },
            {
                "criterion_id": "policy_trace_present",
                "description": "Policy execution trace includes events.",
                "artifact": "pipeline/robot_eval_jobs/{job_id}/policy_execution_trace.json",
                "check": "json_list_min_length",
                "field": "events",
                "min_length": 1,
            },
            {
                "criterion_id": "task_eval_simulator_proof",
                "description": "Task Evaluation Run proves simulator execution.",
                "artifact": "pipeline/robot_eval_jobs/{job_id}/proof_boundary.json",
                "check": "json_field_equals",
                "field": "simulator_execution_proven",
                "expected": True,
            },
            {
                "criterion_id": "task_eval_robot_readiness_boundary",
                "description": "Task Evaluation Run does not imply robot readiness.",
                "artifact": "pipeline/robot_eval_jobs/{job_id}/proof_boundary.json",
                "check": "json_field_equals",
                "field": "robot_readiness_proven",
                "expected": False,
            },
            {
                "criterion_id": "task_eval_public_claim_boundary",
                "description": "Task Evaluation Run does not upgrade public claims.",
                "artifact": "pipeline/robot_eval_jobs/{job_id}/proof_boundary.json",
                "check": "json_field_equals",
                "field": "public_claim_upgrade_allowed",
                "expected": False,
            },
        ],
    },
    {
        "scenario_id": "post_training_data_package_export",
        "title": "Post-Training Data Package Export",
        "major_capability": "Post-Training Data Package artifact export",
        "realistic_context": (
            "A robot-eval job emits a package export with dataset metadata, rights, "
            "checksums, package index, and bounded training-use claims."
        ),
        "evaluation_method_id": EVALUATION_METHOD_ID,
        "success_criteria": [
            "Dataset, rights, index, checksum, and export manifests exist.",
            "The export contains at least one episode.",
            "The package remains a data package and does not claim training completion.",
            "The package does not imply robot readiness or public launch readiness.",
        ],
        "criteria": [
            {
                "criterion_id": "dataset_card_ready",
                "description": "Dataset card exists and is ready.",
                "artifact": "pipeline/robot_eval_jobs/{job_id}/dataset_card.json",
                "check": "json_field_in",
                "field": "status",
                "allowed": ["ready", "complete", "completed", "exported"],
            },
            {
                "criterion_id": "license_manifest_accepted",
                "description": "License manifest records accepted rights.",
                "artifact": "pipeline/robot_eval_jobs/{job_id}/license_manifest.json",
                "check": "json_field_in",
                "field": "rights_status",
                "allowed": ["accepted", "ready", "complete", "completed"],
            },
            {
                "criterion_id": "package_index_files",
                "description": "Package index references exported files.",
                "artifact": "pipeline/robot_eval_jobs/{job_id}/package_index.json",
                "check": "json_list_min_length",
                "field": "files",
                "min_length": 1,
            },
            {
                "criterion_id": "checksums_recorded",
                "description": "Checksum manifest records at least one file digest.",
                "artifact": "pipeline/robot_eval_jobs/{job_id}/checksums.json",
                "check": "json_dict_min_length",
                "field": "files",
                "min_length": 1,
            },
            {
                "criterion_id": "export_manifest_schema",
                "description": "Post-training export manifest uses the expected schema.",
                "artifact": (
                    "pipeline/robot_eval_jobs/{job_id}/"
                    "post_training_data_package_export_manifest.json"
                ),
                "check": "json_field_equals",
                "field": "schema_version",
                "expected": "post_training_data_package_export.v1",
            },
            {
                "criterion_id": "export_episode_count",
                "description": "Export includes at least one episode.",
                "artifact": (
                    "pipeline/robot_eval_jobs/{job_id}/"
                    "post_training_data_package_export_manifest.json"
                ),
                "check": "json_number_at_least",
                "field": "episode_count",
                "min_value": 1,
            },
            {
                "criterion_id": "package_training_boundary",
                "description": "Data package does not claim training completion.",
                "artifact": (
                    "pipeline/robot_eval_jobs/{job_id}/"
                    "post_training_data_package_export_manifest.json"
                ),
                "check": "json_field_equals",
                "field": "claim_boundary.training_completed",
                "expected": False,
            },
            {
                "criterion_id": "package_robot_readiness_boundary",
                "description": "Data package does not imply robot readiness.",
                "artifact": (
                    "pipeline/robot_eval_jobs/{job_id}/"
                    "post_training_data_package_export_manifest.json"
                ),
                "check": "json_field_equals",
                "field": "claim_boundary.robot_readiness_proven",
                "expected": False,
            },
            {
                "criterion_id": "package_public_claim_boundary",
                "description": "Data package does not upgrade public claims.",
                "artifact": (
                    "pipeline/robot_eval_jobs/{job_id}/"
                    "post_training_data_package_export_manifest.json"
                ),
                "check": "json_field_equals",
                "field": "claim_boundary.public_claim_upgrade_allowed",
                "expected": False,
            },
        ],
    },
    {
        "scenario_id": "hosted_runtime_session",
        "title": "Hosted Runtime Session",
        "major_capability": "hosted-session and runtime artifacts",
        "realistic_context": (
            "The generated site-world package is registered into a hosted runtime session "
            "contract without coupling the product to one permanent backend."
        ),
        "evaluation_method_id": EVALUATION_METHOD_ID,
        "success_criteria": [
            "Presentation-world and runtime demo manifests exist.",
            "The runtime session contract is explicit.",
            "Site-world registration exists and the health artifact is healthy.",
            "The backend is recorded as replaceable support infrastructure.",
        ],
        "criteria": [
            {
                "criterion_id": "presentation_world_ready",
                "description": "Presentation-world manifest is ready.",
                "artifact": "pipeline/presentation_world/presentation_world_manifest.json",
                "check": "json_field_in",
                "field": "status",
                "allowed": ["ready", "complete", "completed"],
            },
            {
                "criterion_id": "runtime_demo_ready",
                "description": "Runtime demo manifest is ready.",
                "artifact": "pipeline/presentation_world/runtime_demo_manifest.json",
                "check": "json_field_in",
                "field": "status",
                "allowed": ["ready", "complete", "completed"],
            },
            {
                "criterion_id": "runtime_contract_present",
                "description": "Runtime demo declares a session contract version.",
                "artifact": "pipeline/presentation_world/runtime_demo_manifest.json",
                "check": "json_field_present",
                "field": "session_contract_version",
            },
            {
                "criterion_id": "hosted_session_artifact",
                "description": "Runtime demo is explicitly a hosted-session artifact.",
                "artifact": "pipeline/presentation_world/runtime_demo_manifest.json",
                "check": "json_field_equals",
                "field": "hosted_session_artifact",
                "expected": True,
            },
            {
                "criterion_id": "replaceable_runtime_backend_recorded",
                "description": "Runtime demo records its backend as a replaceable engine.",
                "artifact": "pipeline/presentation_world/runtime_demo_manifest.json",
                "check": "json_field_present",
                "field": "model_backend",
            },
            {
                "criterion_id": "site_world_registration_ready",
                "description": "Site-world registration is ready for runtime use.",
                "artifact": "pipeline/evaluation_prep/site_world_registration.json",
                "check": "json_field_equals",
                "field": "runtime_registration_ready",
                "expected": True,
            },
            {
                "criterion_id": "site_world_health_healthy",
                "description": "Site-world health reports healthy.",
                "artifact": "pipeline/evaluation_prep/site_world_health.json",
                "check": "json_field_in",
                "field": "status",
                "allowed": ["healthy", "ready", "ok"],
            },
        ],
    },
    {
        "scenario_id": "support_assets_trust_and_policy_improvement",
        "title": "Support Assets, Trust, And Policy Improvement",
        "major_capability": (
            "generated/model-derived support assets, optional trust outputs, and "
            "policy-improvement support"
        ),
        "realistic_context": (
            "Simulation support artifacts, provider preview QA, production handoff readiness, "
            "and sim-only policy autoresearch are audited as support layers with explicit "
            "claim boundaries."
        ),
        "evaluation_method_id": EVALUATION_METHOD_ID,
        "success_criteria": [
            "Simulation automation support artifacts exist.",
            "Provider preview and production handoff trust outputs are present.",
            "Policy autoresearch reports heldout improvement with clean safety/contact gates.",
            "Support artifacts do not claim physical robot or public readiness.",
        ],
        "criteria": [
            {
                "criterion_id": "simulation_automation_ready",
                "description": "Simulation automation run manifest is ready.",
                "artifact": "pipeline/simulation_automation/simulation_automation_run_manifest.json",
                "check": "json_field_in",
                "field": "status",
                "allowed": ["ready_for_simulation_support", "ready", "complete", "completed"],
            },
            {
                "criterion_id": "scenario_variations_generated",
                "description": "Simulation automation generated scenario variations.",
                "artifact": "pipeline/simulation_automation/simulation_automation_run_manifest.json",
                "check": "json_number_at_least",
                "field": "scenario_variation_count",
                "min_value": 1,
            },
            {
                "criterion_id": "support_assets_generated",
                "description": "Generated/model-derived support assets are recorded.",
                "artifact": "pipeline/simulation_automation/simulation_automation_run_manifest.json",
                "check": "json_list_min_length",
                "field": "generated_support_assets",
                "min_length": 1,
            },
            {
                "criterion_id": "support_assets_robot_boundary",
                "description": "Support assets do not imply robot readiness.",
                "artifact": "pipeline/simulation_automation/proof_boundary.json",
                "check": "json_field_equals",
                "field": "robot_readiness_proven",
                "expected": False,
            },
            {
                "criterion_id": "provider_preview_ready",
                "description": "Provider preview QA is ready or passed.",
                "artifact": "pipeline/provider_preview_qa_manifest.json",
                "check": "json_field_in",
                "field": "status",
                "allowed": ["provider_preview_packet_ready", "ready", "passed", "complete"],
            },
            {
                "criterion_id": "privacy_safe_preview",
                "description": "Provider preview uses privacy-safe input, not raw bypass.",
                "artifact": "pipeline/provider_preview_qa_manifest.json",
                "check": "json_field_equals",
                "field": "privacy_safe_input_verified",
                "expected": True,
            },
            {
                "criterion_id": "raw_video_bypass_not_used",
                "description": "Provider preview did not rely on raw-video bypass.",
                "artifact": "pipeline/provider_preview_qa_manifest.json",
                "check": "json_field_equals",
                "field": "raw_video_bypass_used",
                "expected": False,
            },
            {
                "criterion_id": "production_handoff_status",
                "description": "Production handoff readiness artifact is present and bounded.",
                "artifact": "pipeline/production_handoff_readiness_manifest.json",
                "check": "json_field_in",
                "field": "status",
                "allowed": [
                    "ready_except_owner_gpu_simulator_execution",
                    "ready",
                    "passed",
                    "complete",
                    "completed",
                ],
            },
            {
                "criterion_id": "policy_autoresearch_completed",
                "description": "Policy autoresearch report completed.",
                "artifact": (
                    "pipeline/robot_eval_jobs/{job_id}/policy_autoresearch/"
                    "policy_autoresearch_report.json"
                ),
                "check": "json_field_equals",
                "field": "status",
                "expected": "completed",
            },
            {
                "criterion_id": "heldout_success_quality_bar",
                "description": "Policy autoresearch heldout success meets the quality bar.",
                "artifact": (
                    "pipeline/robot_eval_jobs/{job_id}/policy_autoresearch/"
                    "policy_autoresearch_report.json"
                ),
                "check": "json_number_at_least",
                "field": "heldout_eval.success_rate",
                "min_value": 0.8,
            },
            {
                "criterion_id": "heldout_safety_contact_clean",
                "description": "Policy autoresearch has zero heldout safety/contact failures.",
                "artifact": (
                    "pipeline/robot_eval_jobs/{job_id}/policy_autoresearch/"
                    "policy_autoresearch_report.json"
                ),
                "check": "json_field_equals",
                "field": "heldout_eval.safety_contact_failures",
                "expected": 0,
            },
            {
                "criterion_id": "policy_autoresearch_robot_boundary",
                "description": "Policy autoresearch does not prove robot readiness.",
                "artifact": (
                    "pipeline/robot_eval_jobs/{job_id}/policy_autoresearch/"
                    "policy_autoresearch_report.json"
                ),
                "check": "json_field_equals",
                "field": "proof_boundary.robot_readiness_proven",
                "expected": False,
            },
            {
                "criterion_id": "policy_autoresearch_public_boundary",
                "description": "Policy autoresearch does not upgrade public claims.",
                "artifact": (
                    "pipeline/robot_eval_jobs/{job_id}/policy_autoresearch/"
                    "policy_autoresearch_report.json"
                ),
                "check": "json_field_equals",
                "field": "proof_boundary.public_claim_upgrade_allowed",
                "expected": False,
            },
        ],
    },
]


def major_capability_scenario_definitions() -> list[dict[str, Any]]:
    """Return the immutable scenario definitions used by the suite."""

    return deepcopy(_SCENARIOS)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _find_primary_job_id(capture_root: Path, requested: str | None) -> str | None:
    if requested:
        return requested
    jobs_root = capture_root / "pipeline" / "robot_eval_jobs"
    if not jobs_root.is_dir():
        return None
    job_dirs = sorted(path for path in jobs_root.iterdir() if path.is_dir())
    for job_dir in job_dirs:
        if (job_dir / "job_request.json").is_file():
            return job_dir.name
    return job_dirs[0].name if job_dirs else None


def _resolve_artifact(capture_root: Path, artifact: str, job_id: str | None) -> Path:
    safe_job_id = job_id or "__missing_job_id__"
    return capture_root / artifact.format(job_id=safe_job_id)


def _field_value(payload: Any, field_path: str | None) -> Any:
    if not field_path:
        return payload
    for candidate in field_path.split("|"):
        current = payload
        for part in candidate.split("."):
            key = part.strip()
            if isinstance(current, Mapping) and key in current:
                current = current[key]
            else:
                current = _MISSING
                break
        if current is not _MISSING:
            return current
    return _MISSING


def _expected_for(criterion: Mapping[str, Any]) -> Any:
    check = _string(criterion.get("check"))
    if "expected" in criterion:
        return criterion.get("expected")
    if "allowed" in criterion:
        return criterion.get("allowed")
    if check in {"json_list_min_length", "json_dict_min_length"}:
        return f"length >= {criterion.get('min_length')}"
    if check == "json_number_at_least":
        return f">= {criterion.get('min_value')}"
    if check == "json_field_present":
        return "present"
    if check == "file_exists":
        return "file exists"
    return None


def _json_payload_for(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "artifact_missing"
    payload = optional_read_json(path)
    if not isinstance(payload, Mapping):
        return None, "artifact_not_json_object"
    return dict(payload), None


def _evaluate_json_criterion(
    *,
    criterion: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[bool, Any, str]:
    check = _string(criterion.get("check"))
    observed = _field_value(payload, _string(criterion.get("field")))
    if observed is _MISSING:
        return False, None, "field_missing"

    if check == "json_field_equals":
        expected = criterion.get("expected")
        return observed == expected, observed, (
            "matched_expected" if observed == expected else "value_mismatch"
        )

    if check == "json_field_in":
        allowed = list(criterion.get("allowed") or [])
        return observed in allowed, observed, (
            "value_allowed" if observed in allowed else "value_not_allowed"
        )

    if check == "json_field_present":
        passed = observed not in (None, "")
        return passed, observed, "present" if passed else "empty_value"

    if check == "json_list_min_length":
        min_length = int(criterion.get("min_length") or 0)
        length = len(observed) if isinstance(observed, list) else None
        passed = length is not None and length >= min_length
        return passed, length, "length_ok" if passed else "length_too_short"

    if check == "json_dict_min_length":
        min_length = int(criterion.get("min_length") or 0)
        length = len(observed) if isinstance(observed, Mapping) else None
        passed = length is not None and length >= min_length
        return passed, length, "length_ok" if passed else "length_too_short"

    if check == "json_number_at_least":
        min_value = float(criterion.get("min_value") or 0)
        observed_number = observed if isinstance(observed, (int, float)) else None
        passed = observed_number is not None and float(observed_number) >= min_value
        return passed, observed, "number_ok" if passed else "number_too_low"

    return False, observed, f"unsupported_check:{check}"


def _evaluate_criterion(
    *,
    capture_root: Path,
    job_id: str | None,
    criterion: Mapping[str, Any],
) -> dict[str, Any]:
    artifact = _string(criterion.get("artifact"))
    path = _resolve_artifact(capture_root, artifact, job_id)
    check = _string(criterion.get("check"))
    expected = _expected_for(criterion)

    if check == "file_exists":
        passed = path.is_file()
        return {
            "criterion_id": criterion.get("criterion_id"),
            "description": criterion.get("description"),
            "status": "passed" if passed else "failed",
            "passed": passed,
            "check": check,
            "artifact": artifact,
            "artifact_path": str(path),
            "field": criterion.get("field"),
            "expected": expected,
            "observed": "file exists" if passed else "missing",
            "message": "file_exists" if passed else "artifact_missing",
        }

    payload, payload_error = _json_payload_for(path)
    if payload_error or payload is None:
        return {
            "criterion_id": criterion.get("criterion_id"),
            "description": criterion.get("description"),
            "status": "failed",
            "passed": False,
            "check": check,
            "artifact": artifact,
            "artifact_path": str(path),
            "field": criterion.get("field"),
            "expected": expected,
            "observed": None,
            "message": payload_error,
        }

    passed, observed, message = _evaluate_json_criterion(criterion=criterion, payload=payload)
    return {
        "criterion_id": criterion.get("criterion_id"),
        "description": criterion.get("description"),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "check": check,
        "artifact": artifact,
        "artifact_path": str(path),
        "field": criterion.get("field"),
        "expected": expected,
        "observed": observed,
        "message": message,
    }


def _evaluate_scenario(
    *,
    capture_root: Path,
    job_id: str | None,
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    evidence = [
        _evaluate_criterion(capture_root=capture_root, job_id=job_id, criterion=criterion)
        for criterion in list(scenario.get("criteria") or [])
        if isinstance(criterion, Mapping)
    ]
    failed_criteria = [
        _string(item.get("criterion_id")) for item in evidence if item.get("passed") is not True
    ]
    passed = not failed_criteria
    return {
        "scenario_id": scenario.get("scenario_id"),
        "title": scenario.get("title"),
        "major_capability": scenario.get("major_capability"),
        "realistic_context": scenario.get("realistic_context"),
        "evaluation_method_id": scenario.get("evaluation_method_id"),
        "success_criteria": list(scenario.get("success_criteria") or []),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "failed_criteria": failed_criteria,
        "evidence": evidence,
    }


def _markdown_for_suite(suite: Mapping[str, Any]) -> str:
    summary = _as_mapping(suite.get("summary"))
    lines = [
        "# Major Capability Scenario Suite",
        "",
        f"- Status: `{suite.get('status')}`",
        f"- Capture root: `{suite.get('capture_root')}`",
        f"- Job ID: `{_as_mapping(suite.get('conditions')).get('job_id')}`",
        (
            f"- Scenarios passed: `{summary.get('passed_count', 0)}/"
            f"{summary.get('scenario_count', 0)}`"
        ),
        "",
        "## Evaluation Method",
        "",
        f"- Method: `{_as_mapping(suite.get('evaluation_method')).get('method_id')}`",
        "- Rule: every criterion in a scenario must pass; every scenario must pass.",
        "",
        "## Scenarios",
        "",
    ]
    for scenario_raw in list(suite.get("scenarios") or []):
        scenario = _as_mapping(scenario_raw)
        lines.append(f"### {scenario.get('title')}")
        lines.append(f"- Status: `{scenario.get('status')}`")
        lines.append(f"- Capability: {scenario.get('major_capability')}")
        failed = list(scenario.get("failed_criteria") or [])
        if failed:
            lines.append(f"- Failed criteria: `{', '.join(_string(item) for item in failed)}`")
        lines.append("- Evidence:")
        for evidence_raw in list(scenario.get("evidence") or []):
            item = _as_mapping(evidence_raw)
            lines.append(
                f"  - `{item.get('criterion_id')}`: `{item.get('status')}` "
                f"({item.get('message')})"
            )
        lines.append("")
    return "\n".join(lines)


def _scenario_coverage(scenarios: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": scenario.get("scenario_id"),
            "major_capability": scenario.get("major_capability"),
            "status": scenario.get("status"),
        }
        for scenario in scenarios
    ]


def build_major_capability_scenario_suite(
    *,
    capture_root: str | Path,
    output_dir: str | Path | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Run all five major-capability scenarios under one pass/fail method."""

    root = Path(capture_root).expanduser().resolve()
    primary_job_id = _find_primary_job_id(root, job_id)
    output_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root / "pipeline" / "major_capability_scenarios"
    )
    ensure_dir(output_root)

    scenario_results = [
        _evaluate_scenario(capture_root=root, job_id=primary_job_id, scenario=scenario)
        for scenario in major_capability_scenario_definitions()
    ]
    passed_count = sum(1 for scenario in scenario_results if scenario.get("passed") is True)
    failed_count = len(scenario_results) - passed_count
    report_path = output_root / "major_capability_scenario_report.json"
    markdown_path = output_root / "major_capability_scenario_report.md"
    blockers = [
        {
            "scenario_id": scenario.get("scenario_id"),
            "failed_criteria": scenario.get("failed_criteria"),
        }
        for scenario in scenario_results
        if scenario.get("passed") is not True
    ]
    suite: dict[str, Any] = {
        "schema_version": MAJOR_CAPABILITY_SCENARIO_SUITE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "passed" if failed_count == 0 else "failed",
        "capture_root": str(root),
        "output_dir": str(output_root),
        "evaluation_method": {
            "method_id": EVALUATION_METHOD_ID,
            "type": "pass_fail",
            "rule": (
                "A scenario passes only when every predeclared criterion passes. "
                "The suite passes only when all five scenarios pass."
            ),
            "criteria_defined_before_evaluation": True,
        },
        "conditions": {
            "same_conditions_applied": True,
            "condition_profile": "local_artifact_contract_audit",
            "job_id": primary_job_id,
            "network_calls_allowed": False,
            "external_provider_calls_allowed": False,
            "live_robot_or_gpu_required": False,
            "artifact_root": str(root / "pipeline"),
        },
        "summary": {
            "scenario_count": len(scenario_results),
            "passed_count": passed_count,
            "failed_count": failed_count,
            "coverage": _scenario_coverage(scenario_results),
        },
        "scenarios": scenario_results,
        "blockers": blockers,
        "artifacts": {
            "report": str(report_path),
            "report_markdown": str(markdown_path),
        },
        "claim_boundary": {
            "scenario_suite_is_artifact_contract_evidence": True,
            "scenario_suite_is_not_live_provider_or_robot_execution": True,
            "simulator_support_does_not_prove_physical_robot_readiness": True,
            "public_claim_upgrade_allowed": False,
        },
    }

    write_json(report_path, suite)
    write_text(markdown_path, _markdown_for_suite(suite))
    return suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--job-id", default=None)
    args = parser.parse_args(argv)

    suite = build_major_capability_scenario_suite(
        capture_root=args.capture_root,
        output_dir=args.output_dir,
        job_id=args.job_id,
    )
    print(suite["artifacts"]["report"])
    return 0 if suite["status"] == "passed" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
