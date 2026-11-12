"""A beta try counts only if the robot's cameras could see the job.

The measurements below are capture-447ae4f4's pre-policy runtime gate (cell 00,
pi05_droid), copied field for field from the retained receipt
``policy_canary_runtime_observation_integrity_gate.v1.json``: the camera
qualification pixel counts and the composition check's interior and covered
pixel counts for the tray and the destination marker. Only the object noun is
kept; no path, digest or identifier of the site is.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
import blueprint_pipeline.native_task_arena_policy_canary_session as session
import blueprint_pipeline.policy_canary_cell_setup_failure as setup_failure
from blueprint_pipeline.policy_episode_lifecycle import (
    NOT_VISIBLE, UNMEASURED, VISIBLE, VISIBLE_WITH_CAVEAT, gate_flag_codes, observation_quality_flag_codes,
    visibility_verdict,
)

OCCLUDED = "composition_support_background_occlusion_requires_review"
# 447: (camera, target) -> (interior pixels, interior pixels covered by the room's appearance).
CAPTURE_447_COMPOSITION = {
    ("external", "task_object"): (16880, 912),
    ("external", "task_target_marker"): (16926, 2406),
    ("overview", "task_object"): (5406, 97),
    ("overview", "task_target_marker"): (12700, 449),
}
CAPTURE_447_QUALIFICATIONS = {
    "external": {"centroid_within_margin": True, "minimum_pixels": 200, "passed": True, "pixel_count": 17234,
                 "raw_observability_passed": True, "raw_render_passed": True, "render_passed": True,
                 "status": "centered", "wrist_robot_occlusion_qualified": False},
    "overview": {"centroid_within_margin": True, "minimum_pixels": 200, "passed": True, "pixel_count": 6095,
                 "raw_observability_passed": True, "raw_render_passed": True, "render_passed": True,
                 "status": "centered", "wrist_robot_occlusion_qualified": False},
    "wrist": {"centroid_within_margin": True, "minimum_pixels": 120, "passed": False, "pixel_count": 94,
              "raw_observability_passed": False, "raw_render_passed": True, "render_passed": True,
              "status": "insufficient_task_pixels", "wrist_robot_occlusion_qualified": False},
}


def _assessment(interior: int, covered: int) -> dict[str, Any]:
    blockers = [OCCLUDED] if covered else []
    return {"automatic_gaussian_deletion_authorized": False, "blockers": blockers, "interaction_visibility": None,
            "interior_pixels_occluded_by_appearance": covered, "minimum_support_interior_pixels": 64,
            "passed": not blockers, "pixel_cause_proven": False, "scope": "opaque_target_coverage",
            "silhouette_margin_pixels": 2, "silhouette_pixels_occluded_by_appearance": 0,
            "support_interior_pixels": interior, "whole_asset_visibility_qualified": not blockers}


def measured_gate(composition=None, qualifications=None) -> dict[str, Any]:
    """The runtime gate in the worker's shape, with 447's measurements unless changed."""
    from tests.test_beta_screening_worker_control import runtime_observation_gate

    composition = dict(CAPTURE_447_COMPOSITION if composition is None else composition)
    qualifications = deepcopy(CAPTURE_447_QUALIFICATIONS if qualifications is None else qualifications)
    views = [{"camera_role": role, "target_semantic_class": target, "assessment": _assessment(*pixels)}
             for (role, target), pixels in sorted(composition.items())]
    composition_blockers = [f"{row['camera_role']}:{row['target_semantic_class']}:{item}"
                            for row in views for item in row["assessment"]["blockers"]]
    blockers = composition_blockers + (["droid_wrist_task_pixels_below_threshold"]
                                       if not qualifications["wrist"]["passed"] else [])
    if blockers:
        blockers.append("policy_canary_task_semantic_visibility_failed")
    gate = runtime_observation_gate(backend_digest="sha256:" + "b" * 64, blockers=blockers or ["placeholder"])
    gate.update(camera_role_qualifications=qualifications,
                asset_composition_gate={"schema_version": "native_task_asset_composition_gate.v1",
                                        "status": "blocked" if composition_blockers else "passed",
                                        "passed": not composition_blockers, "blockers": composition_blockers,
                                        "target_semantic_classes": ["task_object", "task_target_marker"],
                                        "views": views})
    gate["gate_digest"] = canonical_digest(gate, digest_field="gate_digest")
    return gate


def test_capture_447_is_visible_with_a_caveat_and_says_what_was_hidden() -> None:
    verdict = visibility_verdict(measured_gate())
    assert verdict["outcome"] == VISIBLE_WITH_CAVEAT
    # Published: only whole-view flags (each target is partly hidden in every camera that measures coverage).
    assert verdict["flags"] == ["task_object_partly_hidden", "task_target_marker_partly_hidden"]
    # What each camera showed stays in the record, never published as a flag.
    assert verdict["camera_flags"] == ["task_object_not_in_view:wrist", "task_object_partly_hidden:external",
                                       "task_object_partly_hidden:overview", "task_target_marker_partly_hidden:external",
                                       "task_target_marker_partly_hidden:overview"]
    assert verdict["stops_the_tries"] is False
    # The flag claims only what was measured: the cameras the gate checked are named.
    assert verdict["checked_cameras"] == ["external", "overview", "wrist"]
    tray = verdict["targets"]["task_object"]
    assert tray["policy_cameras"]["external"]["hidden_share"] == round(912 / 16880, 6)
    # The wrist's coverage is not measured: reported as such, never as clean.
    assert tray["policy_cameras"]["wrist"] == {"pixel_count": 94, "minimum_pixels": 120, "seen": False,
                                               "clean": None, "coverage_measured": False,
                                               "source": "camera_qualification_semantic_segmentation"}
    # The overview camera is review-only: counted and flagged, never what the policy sees.
    assert set(tray["review_cameras"]) == {"overview"} and "overview" not in tray["policy_cameras"]
    assert verdict["targets"]["task_target_marker"]["policy_cameras"]["external"]["hidden_share"] == round(
        2406 / 16926, 6)


def test_the_verdict_replaces_the_visibility_blockers_and_keeps_every_other_blocker() -> None:
    gate = measured_gate()
    codes = observation_quality_flag_codes(gate["blockers"] + ["policy_canary_external_destination_visibility_failed"])
    assert gate_flag_codes(gate, codes) == sorted(["policy_canary_external_destination_visibility_failed",
                                                   "task_object_partly_hidden",
                                                   "task_target_marker_partly_hidden"])


def test_partly_hidden_is_published_only_when_no_checked_camera_shows_it_clean() -> None:
    clean_overview = {**CAPTURE_447_COMPOSITION, ("overview", "task_object"): (5406, 0)}
    verdict = visibility_verdict(measured_gate(clean_overview))
    assert "task_object_partly_hidden" not in verdict["flags"]
    assert "task_object_partly_hidden:external" in verdict["camera_flags"]
    assert verdict["flags"] == ["task_target_marker_partly_hidden"]
    # No flag ever names a camera: per-camera detail is the record's, not the website's.
    for composition in (CAPTURE_447_COMPOSITION, clean_overview,
                        {**CAPTURE_447_COMPOSITION, ("external", "task_object"): (16880, 16000)}):
        assert all(":" not in code for code in visibility_verdict(measured_gate(composition))["flags"])


def test_a_clean_view_in_one_policy_camera_is_visible() -> None:
    clean = {key: (interior, 0) for key, (interior, _covered) in CAPTURE_447_COMPOSITION.items()}
    qualifications = deepcopy(CAPTURE_447_QUALIFICATIONS)
    qualifications["wrist"].update(passed=True, pixel_count=4000)
    assert visibility_verdict(measured_gate(clean, qualifications))["outcome"] == VISIBLE
    # A few covered pixels (at most 1 %) are still clean; a gate blocked only by them keeps its blockers.
    few = {**clean, ("external", "task_object"): (16880, 168)}
    gate = measured_gate(few, qualifications)
    assert visibility_verdict(gate)["outcome"] == VISIBLE
    assert gate_flag_codes(gate, observation_quality_flag_codes(gate["blockers"])) == sorted(
        observation_quality_flag_codes(gate["blockers"]))


@pytest.mark.parametrize("target, other", [("task_object", "task_target_marker"),
                                           ("task_target_marker", "task_object")])
def test_more_than_half_covered_in_every_policy_camera_is_flagged_never_a_stop(target, other) -> None:
    composition = dict(CAPTURE_447_COMPOSITION)
    composition[("external", target)] = (16880, 8441)  # one pixel past half
    composition[("overview", target)] = (5406, 0)      # a clean review view: recorded, not what the policy sees
    gate = measured_gate(composition)
    verdict = visibility_verdict(gate)
    assert verdict["outcome"] == NOT_VISIBLE and verdict["stops_the_tries"] is False
    assert f"{target}_not_in_view_of_checked_cameras" in verdict["flags"]
    assert f"{target}_not_in_view:external" in verdict["camera_flags"]
    assert f"{target}_not_in_view:external" not in verdict["flags"]
    assert f"{target}_not_in_view_of_checked_cameras" in gate_flag_codes(gate, [])
    composition[("external", target)] = (16880, 8440)  # exactly half: still seen, partly hidden
    assert visibility_verdict(measured_gate(composition))["outcome"] == VISIBLE_WITH_CAVEAT


def test_the_wrist_alone_can_carry_the_object_and_too_few_pixels_are_flagged() -> None:
    composition = {**CAPTURE_447_COMPOSITION, ("external", "task_object"): (16880, 16000)}
    qualifications = deepcopy(CAPTURE_447_QUALIFICATIONS)
    qualifications["wrist"].update(passed=True, pixel_count=900)
    verdict = visibility_verdict(measured_gate(composition, qualifications))
    assert verdict["outcome"] == VISIBLE_WITH_CAVEAT and "task_object_not_in_view:external" in verdict["camera_flags"]
    assert "task_object_not_in_view_of_checked_cameras" not in verdict["flags"]
    small = {**CAPTURE_447_COMPOSITION, ("external", "task_object"): (63, 0)}
    assert "task_object_not_in_view_of_checked_cameras" in visibility_verdict(measured_gate(small))["flags"]


def test_a_gate_without_measurements_keeps_its_blockers_as_flags() -> None:
    from tests.test_beta_screening_worker_control import REAL_GATE_BLOCKERS, runtime_observation_gate
    gate = runtime_observation_gate(backend_digest="sha256:" + "b" * 64)
    assert visibility_verdict(gate)["outcome"] == UNMEASURED
    assert gate_flag_codes(gate, sorted(REAL_GATE_BLOCKERS)) == sorted(REAL_GATE_BLOCKERS)
    broken = measured_gate()
    broken["asset_composition_gate"]["views"][0]["assessment"]["scope"] = "initial_articulated_contact_patch"
    assert visibility_verdict(broken)["outcome"] == UNMEASURED


def test_the_session_counts_447s_tries_with_the_caveat_flags(tmp_path, monkeypatch) -> None:
    from tests.test_beta_screening_worker_control import stage_beta_bundle
    from tests.test_native_task_arena_policy_canary_session import _gated_session
    _runtime, _output, inputs, authority, *_ = stage_beta_bundle(tmp_path, monkeypatch)
    calls = {"loads": []}
    result = _gated_session(authority, inputs, measured_gate(), calls)
    assert result["status"] == "completed_unqualified" and calls["loads"] == inputs["candidate_ids"]
    assert result["observation_quality_flags"] == ["task_object_partly_hidden", "task_target_marker_partly_hidden"]
    whole = _gated_session(authority, inputs, measured_gate(), {"loads": []}, selected_cell_index=None)
    assert session.validate_session_result(whole)["observation_quality_flags"] == result["observation_quality_flags"]


def test_a_job_the_cameras_cannot_see_still_runs_its_tries_with_the_flag(tmp_path, monkeypatch) -> None:
    from tests.test_beta_screening_worker_control import stage_beta_bundle
    from tests.test_native_task_arena_policy_canary_session import _gated_session
    _runtime, _output, inputs, authority, *_ = stage_beta_bundle(tmp_path, monkeypatch)
    hidden = {**CAPTURE_447_COMPOSITION, ("external", "task_object"): (16880, 12000)}
    calls = {"loads": []}
    result = _gated_session(authority, inputs, measured_gate(hidden), calls)
    # The owner's rule: the tries run; nothing here is a stop, so nothing is retried with a paid start.
    assert result["status"] == "completed_unqualified" and calls["loads"] == inputs["candidate_ids"]
    assert "task_object_not_in_view_of_checked_cameras" in result["observation_quality_flags"]
    assert not result.get("session_failure_codes")


def test_the_not_visible_cell_is_sealed_as_our_scene_setup_fault(tmp_path) -> None:
    from tests.test_policy_canary_cell_setup_failure import _blocked_child, _inputs, _write_child
    inputs = _inputs(tmp_path)
    gate = measured_gate({**CAPTURE_447_COMPOSITION, ("external", "task_target_marker"): (16926, 16000)})
    child = _blocked_child(inputs, 1, run_kind=session.BETA_RUN_KIND, preload_observation_gate=gate,
                           session_failure_type="PolicyCanarySessionError",
                           session_failure_codes=["scene_task_not_visible:task_target_marker"],
                           session_failure_source_trace=[{"file": "native_task_arena_policy_canary_session.py",
                                                          "function": "execute_paired_session", "line": 1}])
    child_root = _write_child(tmp_path, 1, child,
                              ["static_preflight_passed", "isaac_launch_started", "isaac_launch_completed"])
    failure = setup_failure.cell_setup_failure(child=child, inputs=inputs, index=1, child_root=child_root,
                                               output_root=tmp_path)
    assert failure["blocker"] == "policy_canary_session_setup_failed:observation_gate"
    assert failure["observation_gate_verdict"] == "refused"
    assert failure["failure_codes"] == ["scene_task_not_visible:task_target_marker"]
    assert "observation_quality_flags" not in failure
