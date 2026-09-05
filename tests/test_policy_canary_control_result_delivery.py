"""Real lossless manifests and sealed native controls survive private delivery."""
from __future__ import annotations

import hashlib
import json
import zipfile
from copy import deepcopy

import numpy as np
import pytest

from blueprint_pipeline import episode_visual_evidence as visual
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.policy_canary_control_result_delivery import CONTROL_IDS, WARNINGS
from blueprint_pipeline.task_evaluation_policy_canary_result_projection import build_policy_canary_result_projection
from blueprint_pipeline.task_evaluation_policy_canary_result import validate_policy_canary_result
from blueprint_pipeline.task_evaluation_result_delivery import (
    materialize_policy_canary_result_delivery, compact_policy_canary_website_delivery,
    resolve_task_evaluation_result_artifact, TaskEvaluationResultDeliveryError,
)
from blueprint_pipeline.task_evaluation_run_webapp_sync import build_task_evaluation_policy_canary_webapp_publication
from tests.test_episode_visual_evidence import _calibration
from tests.test_task_evaluation_policy_canary_result_delivery import _result, _closure


def _record(path, root):
    payload = path.read_bytes()
    return {"relative_path": path.relative_to(root).as_posix(), "size_bytes": len(payload),
            "sha256": "sha256:" + hashlib.sha256(payload).hexdigest()}


def _seal(path, value, field):
    value[field] = canonical_digest(value, digest_field=field)
    path.write_text(json.dumps(value))
    return value


@pytest.fixture
def case(tmp_path, monkeypatch):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    result = _result(evidence)
    first = result["episodes"][0]
    first["evidence_artifacts"]["review_video"] = next(row for row in result["artifact_inventory"] if row["role"] == "review_video")
    result.update(run_id="scene841757-controls", configuration_digest="sha256:" + "9"*64, blockers=[])
    result["episodes"] = []
    result["controls"] = []

    def encode(frame_paths, *, video_path, frames_per_second):
        video_path.write_bytes(b"same valid fixture video bytes")
        return {**_record(video_path, video_path.parent), "frame_count": len(frame_paths)}
    monkeypatch.setattr(visual, "_encode_or_resume_episode_video", encode)
    for index in range(10):
        cell = f"quick-cell-{index}"
        seed = 3100 + index
        for candidate in ("pi05_droid", "groot_n17_droid"):
            episode = deepcopy(first)
            episode.update(cell_id=cell, seed=seed, candidate_id=candidate)
            episode["episode"]["episode_id"] = f"{result['run_id']}--{cell}--{candidate}"
            result["episodes"].append(episode)
        cell_root = evidence / "control_runs" / f"{index:02d}"
        control_root = cell_root / "strict_controls"
        control_root.mkdir(parents=True)
        receipts = []
        for control in CONTROL_IDS:
            episode_id = f"{cell}-{control}"
            observations = []
            for observation_index, kind in enumerate(("policy-input", "terminal-observation")):
                images = {camera: np.full((4, 4, 3), 20 + observation_index, dtype=np.uint8)
                          for camera in ("external", "wrist", "overview")}
                observations.append(visual.persist_multicamera_observation(images, output_dir=control_root,
                    episode_id=episode_id, observation_index=observation_index, kind=kind,
                    timestamp_ns=100+observation_index, simulation_time_s=float(observation_index)/15,
                    calibrations={camera: _calibration(4, 4) for camera in images},
                    source_devices={camera: "cpu" for camera in images},
                    synchronizations={camera: {"host_bytes_ready": True, "method": "cpu_copy"} for camera in images}))
            media, artifacts = visual.finalize_manipulation_evaluation_visual_evidence(output_dir=control_root,
                episode_id=episode_id, identity={"control_id": control}, policy_input_observations=observations[:1],
                terminal_observation=observations[-1])
            positive = control == "deterministic_scripted_positive"
            receipt = {"schema_version": "adp_task_control_episode.v1", "control_id": control,
                "episode_id": episode_id, "task_spec_digest": "sha256:"+"a"*64,
                "control_plan_digest": "sha256:"+"b"*64, "control_passed": True, "blockers": [],
                "score": {"status": "scored", "task_succeeded": positive,
                          "outcome": "placed" if positive else "never_moved", "failed_criteria": []},
                "state_trace": [{"step_index": 0}], "action_trace": [],
                "state_trace_digest": canonical_digest({"samples": [{"step_index": 0}]}),
                "action_trace_digest": canonical_digest({"actions": []}),
                "visual_evidence": media, "media_artifacts": artifacts,
                "grader_authority": "deterministic_simulator_state", "candidate_policy_queried": False,
                "caller_asserted_success_accepted": False}
            receipts.append(_seal(control_root / f"adp_task_control_episode.{control}.json", receipt, "receipt_digest"))
        files = [_record(path, cell_root) for path in sorted(control_root.rglob("*")) if path.is_file()]
        parent_path = cell_root / "policy_canary_cell_controls.v1.json"
        parent = {"schema_version": "policy_canary_cell_controls.v1", "status": "passed", "cell_id": cell,
            "seed": seed, "scene_plan_digest": "sha256:"+"c"*64, "task_spec_digest": "sha256:"+"a"*64,
            "camera_binding_digest": "sha256:"+"d"*64, "candidate_policy_queried": False,
            "controls": receipts, "files": files, "blockers": []}
        _seal(parent_path, parent, "receipt_digest")
        for receipt in receipts:
            result["controls"].append({"cell_id": cell, "seed": seed, "control_id": receipt["control_id"],
                "control_passed": True, "receipt": receipt, "evidence_root": cell_root.relative_to(evidence).as_posix(),
                "evidence_files": files, "cell_receipt_artifact": _record(parent_path, evidence)})
        result["artifact_inventory"].extend({"role": "control_evidence", **_record(path, evidence)}
            for path in sorted(cell_root.rglob("*")) if path.is_file())
    result.update(control_episode_count=20,
        controls_gate={"status": "passed", "required_control_episode_count": 20, "candidate_policies_loaded_during_controls": False},
        strict_paired_gate={"schema_version": "policy_canary_strict_paired_gate.v1", "status": "passed", "blockers": [], "deterministic_success_required": False},
        strict_gate_blockers=[])
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    closure = {key: _closure(tmp_path / f"{key}.json", flag=flag) for key, flag in (
        ("billing", "official_billing_sealed"), ("teardown", "teardown_completed"), ("provider_zero", "provider_zero_verified"))}
    return result, evidence, closure


def _deliver(tmp_path, case, status="completed_unqualified"):
    result, evidence, closure = case
    return materialize_policy_canary_result_delivery(run_root=tmp_path, run_id=result["run_id"], result_status=status,
        session_result=result, evidence_root=evidence, closure_records=closure)


def test_twenty_native_controls_have_separate_private_artifacts_and_verified_status(tmp_path, case):
    result, _, _ = case
    delivery = _deliver(tmp_path, case)
    assert delivery["scene_controls_status"] == "controls_verified_development_only"
    assert delivery["controls_summary"] == {"expected_count": 20, "recorded_count": 20,
        "completed_count": 20, "passed_count": 20, "verified_cell_count": 10}
    assert len(delivery["episodes"]) == len(delivery["controls"]) == 20
    compact = compact_policy_canary_website_delivery(delivery)
    assert len(compact["artifacts"]) < len(delivery["artifacts"])
    projection = build_policy_canary_result_projection(setup={"scene_id": "841757", "request_digest": "sha256:"+"8"*64,
        "task_success_contract": result["task_success_contract"], "task_success_contract_digest": result["task_success_contract_digest"]},
        result=result, delivery=compact)
    assert projection["result_status"] == "completed_unqualified"
    assert projection["warning"] == WARNINGS["controls_verified_development_only"]
    assert projection["counts"]["completed_diagnostic_control_rollout_count"] == 20
    for control in projection["controls"]:
        assert set(control["videos"]) == {"external", "wrist", "overview"}
        assert control["receipt"]["artifact_id"]
        assert {row["role"] for row in control["artifacts"]} == {"control_cell_archive", "frame_manifest", "state_trace", "action_trace"}
    publication = build_task_evaluation_policy_canary_webapp_publication(capture_session_id="capture", intake_id="intake",
        run_id=result["run_id"], request_digest=projection["request_digest"], configuration_digest=result["configuration_digest"],
        result_status="completed_unqualified", result_delivery=compact, policy_canary_result=projection)
    assert publication["scene_controls_status"] == "controls_verified_development_only"
    path, _ = resolve_task_evaluation_result_artifact(run_root=tmp_path, run_id=result["run_id"],
        artifact_id=projection["controls"][0]["artifacts"][0]["artifact_id"])
    with zipfile.ZipFile(path) as archive:
        assert any(name.endswith("policy_canary_cell_controls.v1.json") for name in archive.namelist())
    assert projection["report"]["controls_csv"]["artifact_id"]


def test_foreign_cell_seed_and_tampered_native_receipts_never_verify(tmp_path, case):
    result, _, _ = case
    result["controls"][0]["seed"] += 1
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    with pytest.raises(TaskEvaluationResultDeliveryError, match="control_cell_binding_invalid"):
        _deliver(tmp_path, case)


def test_missing_control_is_visible_as_incomplete_without_a_claim_upgrade(tmp_path, case):
    result, _, _ = case
    result["controls"].pop()
    result.update(control_episode_count=19, blockers=["strict_controls_incomplete"])
    result["controls_gate"]["status"] = "blocked"
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    delivery = _deliver(tmp_path, case, "blocked")
    assert delivery["scene_controls_status"] == "controls_failed"
    assert len(delivery["controls"]) == 19
    assert delivery["controls_summary"]["verified_cell_count"] == 9
    assert delivery["warning"] == WARNINGS["controls_failed"]


def test_schema_rejects_resigned_verified_claim_without_controls():
    from tests.test_task_evaluation_policy_canary_webapp_sync import _projection
    _, result = _projection()
    result["scene_controls_status"] = "controls_verified_development_only"
    result["warning"] = WARNINGS["controls_verified_development_only"]
    result["projection_digest"] = cross_runtime_canonical_digest(result, digest_field="projection_digest")
    with pytest.raises(ValueError, match="controls_evidence_missing"):
        validate_policy_canary_result(result)


def test_native_control_failure_is_retained_separately_from_learned_results(tmp_path, case):
    result, evidence, _ = case
    control = result["controls"][1]
    receipt = control["receipt"]
    receipt.update(control_passed=False, blockers=["positive_control_failed"])
    receipt["score"].update(task_succeeded=False, outcome="never_moved", failed_criteria=["placement"])
    cell_root = evidence / control["evidence_root"]
    receipt_path = cell_root / "strict_controls" / f"adp_task_control_episode.{control['control_id']}.json"
    _seal(receipt_path, receipt, "receipt_digest")
    parent_path = evidence / control["cell_receipt_artifact"]["relative_path"]
    parent = json.loads(parent_path.read_text())
    parent.update(status="blocked", blockers=["strict_controls_pair_failed"])
    parent["controls"][1] = receipt
    parent["files"] = [_record(cell_root / row["relative_path"], cell_root) for row in parent["files"]]
    _seal(parent_path, parent, "receipt_digest")
    for row in result["controls"][:2]:
        row["control_passed"] = row["receipt"]["control_passed"]
        row["evidence_files"] = parent["files"]
        row["cell_receipt_artifact"] = _record(parent_path, evidence)
    result["artifact_inventory"] = [{**row, **_record(evidence / row["relative_path"], evidence)}
                                    for row in result["artifact_inventory"]]
    result["controls_gate"]["status"] = "blocked"
    result["blockers"] = ["strict_controls_pair_failed"]
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    delivery = _deliver(tmp_path, case, "blocked")
    assert delivery["controls_summary"]["completed_count"] == 20
    assert delivery["controls_summary"]["passed_count"] == 19
    assert delivery["controls"][1]["score"]["failed_criteria"] == ["placement"]
    assert len(delivery["episodes"]) == 20
    assert delivery["warning"] == WARNINGS["controls_failed"]


def test_lossless_control_frame_tampering_refuses_delivery(tmp_path, case):
    result, evidence, _ = case
    frame = next(row for row in result["artifact_inventory"] if row["relative_path"].endswith(".png"))
    (evidence / frame["relative_path"]).write_bytes(b"corrupted after provider return")
    with pytest.raises(TaskEvaluationResultDeliveryError, match="artifact_inventory_invalid"):
        _deliver(tmp_path, case)
