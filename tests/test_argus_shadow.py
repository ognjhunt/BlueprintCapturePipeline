"""Synthetic contract fixtures only: no retained real episode or model inference."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp_articulated_task_success_contract import (
    seal_articulated_task_success_contract,
)
from blueprint_pipeline.adp_task_scoring import score_task_episode_from_spec
from blueprint_pipeline.argus_shadow import (
    ARGUS_COMMIT, ARMS, SCHEMA, ShadowError, binding, compare, cost_proposal,
    normalize_argus, prepare, seal, validate_manifest, write_new_output, frame_rows,
)

REPO = Path(__file__).resolve().parents[1]
DELIVERABLE = REPO / "docs/experiments/argus_shadow_20261001"


def _write(root: Path, name: str, value) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return binding(root, name)


def _fixture(tmp_path: Path):
    spec = {"schema_version": "adp_task_spec.v1", "task_kind": "articulated_open_close",
            "target_joint_id": "right_door", "joint_reset_positions_rad": {"right_door": 0, "left_door": 0},
            "target_success_interval_rad": [.8, 1.4],
            "joint_hard_limits_rad": {"right_door": [0, 1.9], "left_door": [-.01, 1.9]},
            "settle_window_samples": 3, "maximum_settled_target_speed_rad_s": .05,
            "non_task_joint_motion_tolerance_rad": .001, "movement_epsilon_rad": .0001,
            "reset_tolerance_rad": .0001}
    contract = seal_articulated_task_success_contract(
        task_spec=spec, site_id="SYNTHETIC_SITE", task_id="SYNTHETIC_TASK",
        author_source="task_owner", author_id="SYNTHETIC_TEST_OWNER",
        confirmation_status="confirmed", confirmed_by_team_id="SYNTHETIC_TEST_TEAM")
    episodes = []
    scenarios = {"success": ([0, .4, .9, .9, .9, .9], "success", True),
                 "subtle_failure": ([0, .4, .9, .9, .9, .9], "failure", False),
                 "success_then_undone": ([0, .9, .9, .2, .2, .2], "failure", True),
                 "occlusion": ([0, .9], "unclear", True)}
    for name, (positions, truth, retreat) in scenarios.items():
        samples = [{"step_index": i, "joint_positions_rad": {"right_door": pos, "left_door": 0},
                    "joint_velocities_rad_s": {"right_door": 0, "left_door": 0},
                    "task_contact_active": False, "joint_limit_violation": False,
                    "containment_violation": False, "robot_collision_failure": False,
                    "scene_collision_failure": False, "retreat_completed": retreat}
                   for i, pos in enumerate(positions)]
        frames = []
        for i in range(6):
            ref = _write(tmp_path, f"{name}/f{i}.png", b"synthetic image placeholder")
            frames.append({**ref, "timestamp_s": i, "camera_id": "external"})
        artifacts = {"task_spec": _write(tmp_path, f"{name}/spec.json", spec),
                     "task_success_contract": _write(tmp_path, f"{name}/contract.json", contract),
                     "state_trace": _write(tmp_path, f"{name}/state.json", {"samples": samples}),
                     "review_video": _write(tmp_path, f"{name}/fixture.mp4", b"fixture not real video"),
                     "frame_manifest": _write(tmp_path, f"{name}/frames.json", {"frames": frames})}
        # Independent fixture labels are declared from each scenario's raw construction,
        # before invoking the current scorer. A simulated review is never a human review.
        label = {"episode_id": name, "outcome": truth, "authority": "human_review",
                 "reviewer_id": "SYNTHETIC_TEST_REVIEWER_NOT_A_REAL_REVIEW",
                 "review_status": "confirmed", "reviewed_criteria": list(contract["criteria"]),
                 "evidence": [{"role": role, "sha256": artifacts[role]["sha256"],
                               "finding": "Synthetic scenario declared in test, not real observation"}
                              for role in ("task_success_contract", "frame_manifest", "state_trace")],
                 "events": {"goal_reached_at_s": 1, "undone_at_s": 3} if name == "success_then_undone" else {}}
        artifacts["independent_label"] = _write(tmp_path, f"{name}/independent.json", label)
        score = score_task_episode_from_spec(task_spec=spec, samples=samples)
        artifacts["baseline_output"] = _write(tmp_path, f"{name}/current.json", score)
        episodes.append({"episode_id": name, "evidence_kind": "synthetic_fixture",
                         "intended_task": "Open right door, release, retreat, and hold within the interval",
                         "duration_s": 5, "categories": [name],
                         "provenance": {"source_kind": "synthetic_fixture", "synthetic": True, "generated": False},
                         "rights": {"offline_review_allowed": True, "basis": "Repository-owned synthetic test data"},
                         "artifacts": artifacts})
    manifest = seal({"schema_version": SCHEMA, "argus_commit": ARGUS_COMMIT,
                     "corpus_kind": "synthetic_fixture", "current_grader_kind": "adp_deterministic",
                     "current_grader_commit": "fixture_checkout",
                     "current_grader_sources": [_write(tmp_path, "source.py", b"fixture source pin")],
                     "selection": {"episode_ids": [e["episode_id"] for e in episodes]},
                     "prompts": {arm: _write(tmp_path, arm + ".txt", b"synthetic prompt") for arm in ARMS[1:]},
                     "episodes": episodes}, "manifest_digest")
    return manifest


def _responses(root: Path, manifest: dict):
    records = []
    for request in prepare(manifest, root, allow_fixtures=True)["requests"]:
        name, arm = request["episode_id"], request["arm"]
        outcome = {"success": "success", "subtle_failure": "failure",
                   "success_then_undone": "success_then_undone", "occlusion": "unclear"}[name]
        if arm == "argus_vanilla" and name == "subtle_failure":
            outcome = "success"
        completion = {"task_completed": outcome, "reason": "Synthetic response only", "confidence": .8}
        if outcome == "success_then_undone":
            completion.update(goal_reached_at_s=1, undone_at_s=3, completed_at_s=None)
        raw = _write(root, f"{name}/{arm}.json", {"parse_ok": True, "labels": {"completion": completion}})
        records.append({k: request[k] for k in ("episode_id", "arm", "request_digest", "manifest_digest", "episode_digest")})
        records[-1].update(raw_response=raw, inference_identity={"model_requested": request["model"],
                          "model_served": "SYNTHETIC_NO_MODEL_CALL", "provider": "SYNTHETIC",
                          "generation_id": "SYNTHETIC_NO_GENERATION"})
    return records


def test_paired_fixture_replays_actual_grader_and_reports_denominators(tmp_path):
    manifest = _fixture(tmp_path)
    before = copy.deepcopy(manifest)
    report = compare(manifest, tmp_path, _responses(tmp_path, manifest), allow_fixtures=True)
    assert report["status"] == "fixture_only"
    assert report["real_episode_count"] == report["model_calls"] == 0
    assert report["ground_truth_counts"] == {"success": 1, "failure": 2, "unclear": 1}
    assert report["metrics"]["current"]["false_success"]["count"] == 0
    assert report["metrics"]["argus_vanilla"]["false_success"] == {"count": 1, "denominator": 2, "rate": .5}
    assert report["metrics"]["argus_adapted"]["missed_failure"]["count"] == 0
    assert report["metrics"]["argus_adapted"]["timestamp_matched_event_count"] == 2
    assert report["metrics"]["argus_adapted"]["brier_score"] is None
    assert report["policy_ranking"]["status"] == "not_supported"
    assert before == manifest


def test_blinded_identical_inputs_deterministic_plan(tmp_path):
    manifest = _fixture(tmp_path)
    plan = prepare(manifest, tmp_path, allow_fixtures=True)
    assert plan == prepare(manifest, tmp_path, allow_fixtures=True)
    assert plan["model_calls"] == 0
    for i in range(0, len(plan["requests"]), 2):
        a, b = plan["requests"][i:i + 2]
        assert a["frames"] == b["frames"] and a["artifacts"] == b["artifacts"]
        assert {"baseline_output", "independent_label"}.isdisjoint(a["artifacts"])
        assert a["frames"][0]["timestamp_s"] == 0 and a["frames"][-1]["timestamp_s"] == 5
        assert a["target_inference_rights_admitted"] is False


def test_fixture_cannot_be_relabelled_real(tmp_path):
    manifest = _fixture(tmp_path)
    with pytest.raises(ShadowError, match="explicit_flag"):
        validate_manifest(manifest, tmp_path)
    manifest["corpus_kind"] = "retained_episodes"
    manifest = seal(manifest, "manifest_digest")
    with pytest.raises(ShadowError, match="current_grader_commit_not_immutable"):
        validate_manifest(manifest, tmp_path)


@pytest.mark.parametrize("change,error", [("missing", "paired_responses_missing"),
                                          ("duplicate", "duplicate_or_unselected"),
                                          ("digest", "paired_input"), ("model", "model_mismatch")])
def test_response_pair_and_identity_refusals(tmp_path, change, error):
    manifest = _fixture(tmp_path)
    records = _responses(tmp_path, manifest)
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records.append(records[0])
    elif change == "digest":
        records[0]["episode_digest"] = records[2]["episode_digest"]
    else:
        records[0]["inference_identity"]["model_requested"] = "other"
    with pytest.raises(ShadowError, match=error):
        compare(manifest, tmp_path, records, allow_fixtures=True)


def test_raw_evidence_and_retained_score_tampering_refused(tmp_path):
    manifest = _fixture(tmp_path)
    ep = manifest["episodes"][0]
    ref = ep["artifacts"]["baseline_output"]
    score = json.loads((tmp_path / ref["path"]).read_text())
    score["outcome"] = "invented"
    ep["artifacts"]["baseline_output"] = _write(tmp_path, ref["path"], score)
    manifest = seal(manifest, "manifest_digest")
    with pytest.raises(ShadowError, match="replay_differs"):
        compare(manifest, tmp_path, _responses(tmp_path, manifest), allow_fixtures=True)
    (tmp_path / "success/f0.png").write_bytes(b"changed")
    with pytest.raises(ShadowError, match="artifact_changed"):
        prepare(manifest, tmp_path, allow_fixtures=True)


@pytest.mark.parametrize("authority", ["grader_agreement", "argus", "policy_self_report"])
def test_circular_ground_truth_rejected(tmp_path, authority):
    manifest = _fixture(tmp_path)
    ep = manifest["episodes"][0]
    label_ref = ep["artifacts"]["independent_label"]
    label = json.loads((tmp_path / label_ref["path"]).read_text())
    label["authority"] = authority
    ep["artifacts"]["independent_label"] = _write(tmp_path, label_ref["path"], label)
    with pytest.raises(ShadowError, match="must_be_independent"):
        prepare(seal(manifest, "manifest_digest"), tmp_path, allow_fixtures=True)


def test_paths_and_rights_fail_closed(tmp_path):
    manifest = _fixture(tmp_path)
    manifest["episodes"][0]["rights"]["offline_review_allowed"] = False
    with pytest.raises(ShadowError, match="offline_rights"):
        prepare(seal(manifest, "manifest_digest"), tmp_path, allow_fixtures=True)
    with pytest.raises(ShadowError, match="outside_root"):
        binding(tmp_path, "../escape")
    (tmp_path / "symlink").symlink_to(tmp_path / "source.py")
    with pytest.raises(ShadowError, match="symlink"):
        binding(tmp_path, "symlink")


@pytest.mark.parametrize("completion,error", [
    ({"task_completed": "success_then_undone", "goal_reached_at_s": 3, "undone_at_s": 2}, "undo_timeline"),
    ({"task_completed": "success", "undone_at_s": 3}, "contradicts_undo"),
    ({"task_completed": "failure", "completed_at_s": float("nan")}, "invalid_number"),
    ({"task_completed": "unclear", "confidence": True}, "invalid_number"),
])
def test_invalid_outcomes_timing_confidence_refused(completion, error):
    with pytest.raises(ShadowError, match=error):
        normalize_argus({"completion": completion}, 5)


@pytest.mark.parametrize("raw", [[], {"labels": []}, {"completion": []}])
def test_malformed_response_objects_refused(raw):
    with pytest.raises(ShadowError, match="not_object"):
        normalize_argus(raw, 5)


def test_failure_cannot_have_terminal_completion_time():
    with pytest.raises(ShadowError, match="non_success"):
        normalize_argus({"completion": {"task_completed": "failure", "completed_at_s": 3}}, 5)


def test_served_model_pairing_and_raw_provenance(tmp_path):
    manifest = _fixture(tmp_path)
    records = _responses(tmp_path, manifest)
    records[1]["inference_identity"]["model_served"] = "DIFFERENT_MODEL"
    with pytest.raises(ShadowError, match="paired_served_model"):
        compare(manifest, tmp_path, records, allow_fixtures=True)
    records[1]["inference_identity"]["model_served"] = "SYNTHETIC_NO_MODEL_CALL"
    report = compare(manifest, tmp_path, records, allow_fixtures=True)
    arm = report["episodes"][0]["arms"]["argus_adapted"]
    assert arm["inference_identity"] == {**records[0]["inference_identity"], "system_fingerprint": None}
    assert arm["raw_response"] == records[0]["raw_response"]
    assert arm["request_digest"] == records[0]["request_digest"]


def test_native_multicamera_media_and_task_state_trace_are_supported(tmp_path):
    manifest = _fixture(tmp_path)
    ep = manifest["episodes"][0]
    ref = ep["artifacts"]["frame_manifest"]
    frames = json.loads((tmp_path / ref["path"]).read_text())["frames"]
    observations = [{"views": {f["camera_id"]: {"relative_path": f["path"], "png_sha256": f["sha256"],
                    "size_bytes": f["size_bytes"], "simulation_time_s": f["timestamp_s"],
                    "camera_id": f["camera_id"]}}} for f in frames]
    ep["artifacts"]["frame_manifest"] = _write(tmp_path, ref["path"], {
        "schema_version": "adp_multicamera_observation_frame_manifest.v1",
        "policy_input_observations": observations[:-1], "terminal_observation": observations[-1]})
    assert frame_rows(ep, tmp_path) == frames
    state_ref = ep["artifacts"]["state_trace"]
    trace = json.loads((tmp_path / state_ref["path"]).read_text())
    ep["artifacts"]["state_trace"] = _write(tmp_path, state_ref["path"], {
        "schema_version": "policy_episode_state_trace.v1", "task_state_samples": trace["samples"]})
    label_ref = ep["artifacts"]["independent_label"]
    label = json.loads((tmp_path / label_ref["path"]).read_text())
    for evidence in label["evidence"]:
        evidence["sha256"] = ep["artifacts"][evidence["role"]]["sha256"]
    ep["artifacts"]["independent_label"] = _write(tmp_path, label_ref["path"], label)
    manifest = seal(manifest, "manifest_digest")
    assert compare(manifest, tmp_path, _responses(tmp_path, manifest), allow_fixtures=True)["episode_count"] == 4


def test_malformed_response_envelope_is_typed_refusal(tmp_path):
    manifest = _fixture(tmp_path)
    records = _responses(tmp_path, manifest)
    records[0]["inference_identity"] = []
    with pytest.raises(ShadowError, match="identity_not_object"):
        compare(manifest, tmp_path, records, allow_fixtures=True)


def test_unparseable_and_partial_results_are_not_success():
    assert normalize_argus({"parse_ok": False, "labels": {"_raw": "{"}}, 5)["parse_failed"]
    assert normalize_argus({"completion": {"task_completed": "partial"}}, 5)["outcome"] == "failure"


def test_output_cannot_overwrite_source_or_evidence(tmp_path):
    retained = tmp_path / "retained.json"
    retained.write_text("retained evidence")
    with pytest.raises(ShadowError, match="no_overwrite"):
        write_new_output(retained, {"replacement": True})
    assert retained.read_text() == "retained evidence"
    output = tmp_path / "new.json"
    write_new_output(output, {"value": 1})
    write_new_output(output, {"value": 1})
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ShadowError, match="symlink"):
        write_new_output(tmp_path / "link/unsafe.json", {})


def test_articulated_grader_and_confirmed_acceptance_must_match(tmp_path):
    manifest = _fixture(tmp_path)
    ep = manifest["episodes"][0]
    ref = ep["artifacts"]["task_spec"]
    spec = json.loads((tmp_path / ref["path"]).read_text())
    spec["target_success_interval_rad"] = [1.1, 1.4]
    ep["artifacts"]["task_spec"] = _write(tmp_path, ref["path"], spec)
    with pytest.raises(ShadowError, match="acceptance_contract_differ"):
        prepare(seal(manifest, "manifest_digest"), tmp_path, allow_fixtures=True)


def test_committed_empty_corpus_is_honest_and_zero_cost():
    manifest = json.loads((DELIVERABLE / "argus_shadow_manifest.v1.json").read_text())
    assert validate_manifest(manifest, REPO) == []
    report = compare(manifest, REPO, [])
    assert report["status"] == "blocked_no_retained_episodes"
    assert report["episode_count"] == 0
    assert report["metrics"]["current"]["false_success"]["rate"] is None
    cost = cost_proposal(manifest, REPO)
    assert cost["estimated_argus_usd"] == cost["unrelated_budgets_available_usd"] == 0
    assert cost["paid_execution_authorized"] is False
    assert cost["conditional_pilot"]["estimated_usd"] == 10.40
