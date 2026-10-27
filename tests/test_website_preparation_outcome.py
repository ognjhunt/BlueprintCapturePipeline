# Covers: src/blueprint_pipeline/website_preparation_outcome.py
# Covers: src/blueprint_pipeline/website_preparation_status.py
"""Retained evidence, not agent assertions, drives customer abstention."""
import json

import pytest

from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.website_preparation_outcome import read_preparation_outcome
from blueprint_pipeline import website_preparation_status as status
from tests.test_website_preparation_status import _finish

pytest_plugins = ["tests.test_website_preparation_status"]


def _seal(path, value, field="digest"):
    value = {**value, field: canonical_digest(value, digest_field=field)}
    write_json(path, value)
    return value


def _evidence(root, context, blocker="robot_rigid_pick_place_unreachable:lift"):
    directory = root / "pipeline/website_scene_preparation"
    preparation = _seal(directory / "preparation.json", {"schema_version": "website_scene_preparation.v1",
        "binding": {"task_context_digest": context}})
    runtime = _seal(directory / "native/runtime_inputs.json", {"schema_version": "website_scene_runtime_inputs.v1",
        "preparation_digest": preparation["digest"], "collision": {"digest": "sha256:" + "a" * 64}})
    verdict = _seal(directory / "native/capability_verdict.json", {
        "schema_version": "website_rigid_task_capability_verdict.v1", "status": "abstained",
        "preparation_digest": preparation["digest"], "robot_id": "franka_panda", "blockers": [blocker],
        "collision_sha256": runtime["collision"]["digest"], "claim_ceiling": "development_only",
        "physical_world_truth_claimed": False}, "verdict_digest")
    handoff = _seal(directory / "handoff.json", {"schema_version": "website_scene_handoff.v1",
        "scene_id": root.parent.parent.name, "capture_id": root.name,
        "preparation_path": str(directory / "preparation.json"), "preparation_digest": preparation["digest"],
        "runtime_inputs": {"path": str(directory / "native/runtime_inputs.json"), "digest": runtime["digest"]},
        "capability_verdict": {"path": str(directory / "native/capability_verdict.json"),
            "status": "abstained", "digest": verdict["verdict_digest"]}})
    return directory, verdict, handoff


def test_signed_source_bound_v3_is_conservative_and_v1_v2_stay_unchanged(prepared_source):
    root, _, context, selectors, *_ = prepared_source
    _finish(prepared_source, "retryable_blocked")
    _, verdict, _ = _evidence(root, context["context_digest"])
    result = status.read_preparation_status(capture_root=root, selectors=selectors, schema_version=status.STATUS_V3)
    assert result["state"] == "awaiting_inputs"
    assert result["outcome"] == {"category": "robot_incompatibility", "reason": "robot_carry_path_rejected",
        "basis": "conservative_configuration_check", "robot_id": "franka_panda",
        "evidence_digest": verdict["verdict_digest"]}
    assert result["status_digest"] == cross_runtime_canonical_digest({k: v for k, v in result.items() if k != "status_digest"})
    for version in (status.STATUS_V1, status.STATUS_V2):
        assert "outcome" not in status.read_preparation_status(capture_root=root, selectors=selectors, schema_version=version)


@pytest.mark.parametrize("change", ["context", "verdict", "robot", "physical", "capture", "external_path", "collision", "traversal"])
def test_current_evidence_binding_cannot_be_forged(prepared_source, change):
    root, _, context, selectors, *_ = prepared_source
    _finish(prepared_source, "retryable_blocked")
    directory, verdict, handoff = _evidence(root, context["context_digest"])
    if change == "context":
        preparation = json.loads((directory / "preparation.json").read_text())
        preparation["binding"]["task_context_digest"] = "sha256:" + "e" * 64
        _seal(directory / "preparation.json", preparation)
    elif change in {"robot", "physical"}:
        verdict["robot_id" if change == "robot" else "physical_world_truth_claimed"] = "other_robot" if change == "robot" else True
        _seal(directory / "native/capability_verdict.json", verdict, "verdict_digest")
    elif change == "verdict":
        verdict["blockers"] = []
        write_json(directory / "native/capability_verdict.json", verdict)
    elif change == "capture":
        handoff["capture_id"] = "another-capture"
        _seal(directory / "handoff.json", handoff)
    elif change in {"external_path", "traversal"}:
        handoff["capability_verdict"]["path"] = ("/private/tmp/unrelated-verdict.json" if change == "external_path"
                                               else str(directory / "../website_scene_preparation/native/capability_verdict.json"))
        _seal(directory / "handoff.json", handoff)
    else:
        runtime = json.loads((directory / "native/runtime_inputs.json").read_text())
        runtime["collision"]["digest"] = "sha256:" + "b" * 64
        runtime = _seal(directory / "native/runtime_inputs.json", runtime)
        handoff["runtime_inputs"]["digest"] = runtime["digest"]
        _seal(directory / "handoff.json", handoff)
    with pytest.raises(status.PreparationStatusUnavailable):
        status.read_preparation_status(capture_root=root, selectors=selectors, schema_version=status.STATUS_V3)


def test_missing_station_is_not_a_robot_reach_verdict(prepared_source):
    root, _, context, *_ = prepared_source
    _evidence(root, context["context_digest"], "robot_station_floor_unobserved")
    result = read_preparation_outcome(capture_root=root, task_context_digest=context["context_digest"], state="awaiting_inputs")
    assert result["category"] == "insufficient_evidence" and result["robot_id"] is None


@pytest.mark.parametrize("state", ["preparing", "handed_off", "authority_ended"])
def test_active_successor_or_ended_authority_never_reuses_old_verdict(prepared_source, state):
    root, _, context, *_ = prepared_source
    _evidence(root, context["context_digest"])
    assert read_preparation_outcome(capture_root=root, task_context_digest=context["context_digest"], state=state) is None


def test_version_negotiation_acceptance_and_missing_evidence(prepared_source):
    root, _, _, selectors, *_ = prepared_source
    _finish(prepared_source)
    result = status.read_preparation_status(capture_root=root, selectors=selectors, schema_version=status.STATUS_V3)
    assert result["outcome"] is None
    assert status.requested_status_version("3") == status.STATUS_V3
    acceptance = {"schema_version": "website_preparation_status_acceptance.v3", **selectors,
        "accepted": True, "state": result["state"], "status_digest": result["status_digest"],
        "native_execution_complete": False, "correlation_id": "bp-prep-" + result["status_digest"][7:23]}
    assert status.accepted_status_version(status.validate_acceptance(acceptance, selectors)) == status.STATUS_V3
