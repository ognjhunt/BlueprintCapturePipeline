"""Reproduce software-only Pipeline publications; no provider or simulator runs."""
from pathlib import Path
import json
import tempfile

from blueprint_pipeline.decision_evidence_contracts import cross_runtime_canonical_digest
from blueprint_pipeline.decision_evidence_execution import build_decision_envelope, execute_evidence_plan
from blueprint_pipeline.decision_evidence_router import route_decision_evidence
from blueprint_pipeline.local_evidence_adapters import authorized_local_evidence_adapter_registry
from blueprint_pipeline.task_evaluation_policy_run_contract import policy_run_result_projection_digest, validate_policy_run_setup
from blueprint_pipeline.task_evaluation_result_delivery import materialize_task_evaluation_result_delivery
from blueprint_pipeline.task_evaluation_run_webapp_sync import build_task_evaluation_run_webapp_publication
from tests.test_task_evaluation_run_control_plane import _testbed, _request, _profile, _qualification
from tests.test_task_evaluation_policy_run_contract import result_projection, setup


def fixtures():
    testbed = _testbed()
    request = _request(testbed)
    method = _profile()
    qualifications = [_qualification(method)]
    plan = route_decision_evidence(request, testbed, [method], qualifications).to_mapping()
    execution = execute_evidence_plan(plan, request, testbed, [method], qualifications,
        registry=authorized_local_evidence_adapter_registry([method["adapter_reference"]]))
    envelope = build_decision_envelope(request, testbed, plan, [row.to_mapping() for row in execution.results]).to_mapping()
    v1 = build_task_evaluation_run_webapp_publication(capture_session_id="software-fixture-capture",
        intake_id="capture-1", run_id="software-fixture-run", state="decided", evidence_plan=plan, decision_envelope=envelope)
    # The incomplete policy matrix is an explicit abstention fixture. It
    # carries planned cells, zero completed episodes, and no scientific pass.
    no_methods_plan = route_decision_evidence(request, testbed, [], []).to_mapping()
    abstention = build_decision_envelope(request, testbed, no_methods_plan, []).to_mapping()
    with tempfile.TemporaryDirectory() as directory:
        delivery = materialize_task_evaluation_result_delivery(run_root=directory,
            run_id="policy-run-1", state="abstained", decision_envelope=abstention)
    projection = result_projection(state="abstained")
    projection["plan_digest"] = no_methods_plan["plan_digest"]
    projection["result_delivery_digest"] = delivery["delivery_digest"]
    projection["matrix"]["completed_episode_count"] = 0
    projection["paired_comparison"]["matched_episode_pairs"] = 0
    for candidate in projection["candidate_results"]:
        candidate["episodes_completed"] = 0
        candidate["evidence"] = {"lossless_frame_manifest_count": 0, "review_video_count": 0, "typed_media_gap_count": 0}
    projection["projection_digest"] = policy_run_result_projection_digest(projection)
    v3 = build_task_evaluation_run_webapp_publication(capture_session_id="software-fixture-capture",
        intake_id="capture-1", run_id="policy-run-1", state="abstained", evidence_plan=no_methods_plan,
        decision_envelope=abstention, result_delivery=delivery, policy_run_result=projection)
    published_setup = setup()
    # This fixture helper predates JCS. Reproduce the current real producer's
    # digest functions without altering its eight distinct variation families.
    template = published_setup["preparation_template"]
    template["template_digest"] = cross_runtime_canonical_digest(template, digest_field="template_digest")
    for preset in published_setup["presets"]:
        if "cells" in preset:
            preset["scenario_set_digest"] = cross_runtime_canonical_digest({"ordered_cells": preset["cells"]})
        preset["nesting_proof_digest"] = cross_runtime_canonical_digest({key: preset[key] for key in (
            "preset_id", "scenario_set_digest", "parent_preset_id", "parent_prefix_count")}
            | {"selection_rule": "published_ordered_prefix"})
    published_setup["setup_digest"] = cross_runtime_canonical_digest(published_setup, digest_field="setup_digest")
    validate_policy_run_setup(published_setup)
    return {"provenance": {"scope": "development_only", "provider_execution": False, "physical_evidence": False,
            "generator": "tests/fixtures/managed_job_publication_v1/generate.py"},
        "v1": v1, "v3": v3, "managed_setup": published_setup}


if __name__ == "__main__":
    target = Path(__file__).with_name("producer-examples.json")
    target.write_text(json.dumps(fixtures(), sort_keys=True, indent=2, ensure_ascii=False) + "\n")
