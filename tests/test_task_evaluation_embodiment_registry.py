"""Only an integrated robot setup may run; every other listed robot says why."""

from __future__ import annotations

import dataclasses

import pytest

from blueprint_pipeline import task_evaluation_embodiment_registry as registry


def test_only_the_integrated_droid_franka_runs_and_every_other_robot_names_its_gap() -> None:
    assert registry.embodiment_registry_violations() == []
    runnable = [row.embodiment_id for row in registry.EMBODIMENT_REGISTRY if row.available]
    assert runnable == ["droid_franka_robotiq_2f85"]
    for row in registry.EMBODIMENT_REGISTRY:
        if not row.available:
            assert row.reason and row.missing
    for robot in ("universal_robots_ur5e", "fanuc_crx_10ia", "abb_gofa_crb15000", "kinova_gen3"):
        assert registry.embodiment(robot).status == "not_integrated"


def test_flipping_a_robot_to_runnable_without_integration_is_caught(monkeypatch) -> None:
    ur5e = registry.embodiment("universal_robots_ur5e")
    monkeypatch.setattr(registry, "EMBODIMENT_REGISTRY", (dataclasses.replace(ur5e, status="runnable"),))
    assert registry.embodiment_registry_violations() == ["runnable_embodiment_incomplete:universal_robots_ur5e"]


# --- G1 navigation (docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md): the general development-run allowance ---

NAVIGATION = "goal_region_navigation"
NAVIGATION_CHECKPOINTS = ("humanoidarena_dp_g1_dex3_sonic_vision_navi",
                          "humanoidarena_pi05_g1_dex3_sonic_vision_navi")
MANIPULATION_CHECKPOINTS = ("humanoidarena_dp_g1_dex3_sonic", "humanoidarena_pi05_g1_dex3_sonic")
#: The owner's words of 2026-11-07, verbatim: each stand-in quotes them with the session and UTC time.
VERIFIED_EPISODE_WORDS = '"Dev check, then counts"'
CHECKPOINT_TERMS_WORDS = '"yes - make sure this is the case for all future runs/scenes"'
CONFIRMATION_WORDS = '"Yes, both confirmed"'


def _g1_with(allowance):
    return dataclasses.replace(registry.embodiment(registry.G1_EMBODIMENT_ID), development_run_allowances=(allowance,))


def _strings(value):
    """Every string value in a record (keys excluded): what a reader of the output takes as its claims."""
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _strings(item)]
    return [value] if isinstance(value, str) else []


def test_the_g1_stays_unavailable_with_every_gap_and_declares_one_general_navigation_allowance() -> None:
    g1 = registry.embodiment(registry.G1_EMBODIMENT_ID)
    # Nothing flips a verified site episode: the body is listed as it was, with both items missing.
    assert g1.status == "unavailable" and not g1.available
    assert g1.missing == ("verified_site_episode", "reviewed_checkpoint_terms")
    assert g1.task_profile_ids == ("rigid_pick_place", NAVIGATION)
    assert g1.development_run_allowances == (registry.G1_NAVIGATION_DEVELOPMENT_RUN,)
    assert [row.embodiment_id for row in registry.EMBODIMENT_REGISTRY if row.development_run_allowances] == [
        registry.G1_EMBODIMENT_ID]
    allowance = registry.G1_NAVIGATION_DEVELOPMENT_RUN
    # General: one task profile, and no site, scene or run field that could name one.
    assert [field.name for field in dataclasses.fields(allowance)] == ["task_profile_id", "stood_in_by",
                                                                       "claim_ceiling"]
    assert allowance.task_profile_id == NAVIGATION and allowance.claim_ceiling == "development_only"
    assert registry.DEVELOPMENT_RUN_TASK_PROFILES == frozenset({NAVIGATION})
    stand = {row.item: row for row in allowance.stood_in_by}
    assert list(stand) == ["verified_site_episode", "reviewed_checkpoint_terms"]
    # Each stand-in quotes the owner verbatim, with the session and UTC time, so a run record can be traced.
    episode = stand["verified_site_episode"].owner_decision
    assert episode == registry.VERIFIED_EPISODE_DECISION
    assert episode.startswith("owner, 2026-11-07 00:05Z, library-aisle session")
    assert VERIFIED_EPISODE_WORDS in episode and "G1 session at 13:43Z" in episode and CONFIRMATION_WORDS in episode
    assert stand["verified_site_episode"].checkpoint_review is None
    terms = stand["reviewed_checkpoint_terms"].owner_decision
    assert terms == registry.CHECKPOINT_TERMS_DECISION
    assert terms.startswith("owner, 2026-11-07 about 13:20Z, library-aisle session, its question 3")
    assert CHECKPOINT_TERMS_WORDS in terms and "G1 session at 13:43Z" in terms and CONFIRMATION_WORDS in terms
    assert "decision 1" not in episode + terms and "decision 2" not in episode + terms
    assert stand["reviewed_checkpoint_terms"].checkpoint_review is \
        registry.g1_navigation_checkpoint_development_rights_review
    assert registry.embodiment_registry_violations() == []


def test_each_navigation_checkpoints_own_review_stands_in_for_its_checkpoint_terms_and_no_other() -> None:
    from blueprint_pipeline import native_g1_beta_authority as authority

    for candidate in NAVIGATION_CHECKPOINTS:
        review = registry.g1_navigation_checkpoint_development_rights_review(candidate)
        # The same pinned file and digest every G1 run's authority re-validates (validate_beta_authority).
        assert review == {"artifact_sha256": "sha256:" + authority.MODEL_REVIEW_FILES[candidate],
                          "rights_review_digest": authority.REVIEW_DIGESTS[candidate],
                          "status": "approved_for_development_simulation"}
    # The owner's decision names the navigation checkpoints' reviews: a manipulation checkpoint's review stands in for
    # nothing here, and a checkpoint with no pinned review has none.
    for candidate in (*MANIPULATION_CHECKPOINTS, "pi05_droid", "humanoidarena_unreviewed_g1_vision_navi"):
        assert registry.g1_navigation_checkpoint_development_rights_review(candidate) is None


def test_a_changed_review_file_stands_in_for_nothing(monkeypatch) -> None:
    from blueprint_pipeline import native_g1_beta_authority as authority

    changed, kept = NAVIGATION_CHECKPOINTS
    monkeypatch.setitem(authority.MODEL_REVIEW_FILES, changed, "0" * 64)
    assert registry.g1_navigation_checkpoint_development_rights_review(changed) is None
    g1 = registry.embodiment(registry.G1_EMBODIMENT_ID)
    missing, allowance = g1.development_run_missing(NAVIGATION, candidate_id=changed)
    assert missing == ("reviewed_checkpoint_terms",)
    assert [row.item for row in allowance.stood_in_by] == ["verified_site_episode"]
    assert g1.development_run_missing(NAVIGATION, candidate_id=kept)[0] == ()


def test_only_a_development_only_navigation_run_may_proceed_without_the_stood_in_items() -> None:
    g1 = registry.embodiment(registry.G1_EMBODIMENT_ID)
    for candidate in NAVIGATION_CHECKPOINTS:
        missing, allowance = g1.development_run_missing(NAVIGATION, candidate_id=candidate)
        assert missing == () and allowance == registry.G1_NAVIGATION_DEVELOPMENT_RUN
        assert allowance.claim_ceiling == "development_only"
    # Every other profile keeps all the body's gaps, whatever the checkpoint.
    for profile in ("rigid_pick_place", "articulated_open_close"):
        for candidate in (*NAVIGATION_CHECKPOINTS, *MANIPULATION_CHECKPOINTS, None):
            assert g1.development_run_missing(profile, candidate_id=candidate) == (g1.missing, None)
    # Checkpoint terms are stood in for only on a navigation checkpoint with its own review.
    for candidate in (*MANIPULATION_CHECKPOINTS, None):
        missing, allowance = g1.development_run_missing(NAVIGATION, candidate_id=candidate)
        assert missing == ("reviewed_checkpoint_terms",)
        assert [row.item for row in allowance.stood_in_by] == ["verified_site_episode"]
    # An allowance stands in only for the items it names.
    partial = _g1_with(dataclasses.replace(registry.G1_NAVIGATION_DEVELOPMENT_RUN,
                                           stood_in_by=registry.G1_NAVIGATION_DEVELOPMENT_RUN.stood_in_by[:1]))
    assert partial.development_run_missing(NAVIGATION, candidate_id=NAVIGATION_CHECKPOINTS[0])[0] == (
        "reviewed_checkpoint_terms",)


@pytest.mark.parametrize("profile", ["rigid_pick_place", "articulated_open_close", "deformable_fold", ""])
def test_an_allowance_naming_any_other_task_profile_is_refused(monkeypatch, profile) -> None:
    g1 = registry.embodiment(registry.G1_EMBODIMENT_ID)
    other = dataclasses.replace(registry.G1_NAVIGATION_DEVELOPMENT_RUN, task_profile_id=profile)
    # Refused even on a body that lists the profile: only DEVELOPMENT_RUN_TASK_PROFILES may be named.
    listed = dataclasses.replace(g1, task_profile_ids=(*g1.task_profile_ids, profile),
                                 development_run_allowances=(other,))
    monkeypatch.setattr(registry, "EMBODIMENT_REGISTRY", (listed,))
    assert registry.embodiment_registry_violations() == [f"development_run_allowance_invalid:{g1.embodiment_id}"]


@pytest.mark.parametrize("change", [
    {"claim_ceiling": "diagnostic_policy_execution"},
    {"stood_in_by": (registry.StandIn("integration_canary_receipt", "invented", "owner decision"),)},
    {"stood_in_by": ()},
    {"stood_in_by": (registry.StandIn("verified_site_episode", "", "owner decision"),)},
    {"stood_in_by": (registry.StandIn("verified_site_episode", "basis", ""),)},
    # Checkpoint terms only per checkpoint, from its own review: never a blanket stand-in.
    {"stood_in_by": (registry.StandIn("reviewed_checkpoint_terms", "basis", "owner decision"),)},
    {"stood_in_by": registry.G1_NAVIGATION_DEVELOPMENT_RUN.stood_in_by[:1] * 2}])
def test_a_wider_or_unexplained_allowance_is_caught(monkeypatch, change) -> None:
    g1 = registry.embodiment(registry.G1_EMBODIMENT_ID)
    widened = dataclasses.replace(registry.G1_NAVIGATION_DEVELOPMENT_RUN, **change)
    monkeypatch.setattr(registry, "EMBODIMENT_REGISTRY",
                        (dataclasses.replace(g1, development_run_allowances=(widened,)),))
    assert registry.embodiment_registry_violations() == [f"development_run_allowance_invalid:{g1.embodiment_id}"]
    # Even the declared allowance is refused on a runnable body.
    runnable = dataclasses.replace(g1, status="runnable", missing=())
    monkeypatch.setattr(registry, "EMBODIMENT_REGISTRY", (runnable,))
    assert f"development_run_allowance_invalid:{g1.embodiment_id}" in registry.embodiment_registry_violations()


def test_a_navigation_walk_records_the_allowance_stood_in_and_development_only_never_verified() -> None:
    from blueprint_pipeline.task_spec_compiler import requirements
    from blueprint_pipeline.website_locomotion_task import NAVIGATION_PIPELINE_PIECES, runnable_options

    assert registry.STOOD_IN == "stood in by the development-run allowance (development_only)"
    for built in (None, frozenset(NAVIGATION_PIPELINE_PIECES)):
        walk = runnable_options(requirements(locomote=True, manipulate=False), built_pieces=built)
        g1_rows = [row for row in walk["policies"] if row["embodiment_id"] == registry.G1_EMBODIMENT_ID]
        # Both navigation rows are listed; nothing narrows a navigation site to one.
        assert sorted(row["candidate_id"] for row in g1_rows) == sorted(NAVIGATION_CHECKPOINTS)
        for row in g1_rows:
            assert not any(item.startswith("embodiment:") for item in row["missing"])
            assert row["claim_ceiling"] == "development_only"
            record = row["development_run_allowance"]
            assert record["task_profile_id"] == NAVIGATION and record["claim_ceiling"] == "development_only"
            assert list(record["stood_in"]) == ["verified_site_episode", "reviewed_checkpoint_terms"]
            assert {entry["status"] for entry in record["stood_in"].values()} == {registry.STOOD_IN}
            assert VERIFIED_EPISODE_WORDS in record["stood_in"]["verified_site_episode"]["owner_decision"]
            assert CHECKPOINT_TERMS_WORDS in record["stood_in"]["reviewed_checkpoint_terms"]["owner_decision"]
            assert record["stood_in"]["reviewed_checkpoint_terms"]["checkpoint_review"] == \
                registry.g1_navigation_checkpoint_development_rights_review(row["candidate_id"])
            # Stood in, never verified or satisfied: no value of the record says so.
            assert not any(word in text.lower() for text in _strings(record) for word in ("verified", "satisf"))
        # The body itself is listed as it is: unavailable, with every gap.
        body = next(row for row in walk["embodiments"] if row["embodiment_id"] == registry.G1_EMBODIMENT_ID)
        assert body["status"] == "unavailable"
        assert body["missing"] == ["verified_site_episode", "reviewed_checkpoint_terms"]
    # With no job's outputs known it is still held by the pipeline's own pieces, not by the body's items.
    walk = runnable_options(requirements(locomote=True, manipulate=False))
    assert all(not row["runnable"] and row["missing"] for row in walk["policies"])


def test_the_navigation_hold_names_neither_stood_in_item_and_records_the_allowance() -> None:
    from blueprint_pipeline.task_spec_compiler import requirements
    from blueprint_pipeline.website_locomotion_hold import HOLD_CODE
    from blueprint_pipeline.website_locomotion_task import runnable_policy_gap

    context = {"task_family": "navigation", "context_digest": "sha256:" + "a" * 64, "description": "Walk."}
    gap = runnable_policy_gap(task_context=context, requirements=requirements(locomote=True, manipulate=False),
                              spec_digest="sha256:" + "c" * 64)
    assert gap["code"] == HOLD_CODE
    reason_missing = gap["reason"].split("Missing: ", 1)[1].split(".", 1)[0]
    for item in ("embodiment:verified_site_episode", "embodiment:reviewed_checkpoint_terms"):
        assert item not in gap["missing"] and item not in reason_missing
    assert f"embodiment:reviewed_checkpoint_terms, embodiment:verified_site_episode: {registry.STOOD_IN}." \
        in gap["reason"]
    policies = gap["runnable_options"]["policies"]
    assert policies and all(row["development_run_allowance"]["claim_ceiling"] == "development_only"
                            for row in policies)


def test_the_g1_run_scope_a_stood_in_run_records_is_pinned_development_only() -> None:
    """Pin: whatever proceeds on the allowance, every G1 run's authority claims no more than development_only."""
    from blueprint_pipeline import native_g1_beta_authority as authority

    assert authority.SCOPE["claim_ceiling"] == "development_only"
    assert authority.SCOPE["ranking_eligible"] is False and authority.SCOPE["qualification_permitted"] is False
    assert authority.SCOPE["physical_outcome_claimed"] is False


def test_rigid_pick_place_on_the_g1_is_pinned_as_it_was() -> None:
    """Pin: manipulation routing, the manipulation catalog rows and their registration identity are unchanged."""
    import json
    from pathlib import Path

    from blueprint_pipeline.job_capability_routing import route_job_capabilities
    from blueprint_pipeline.task_spec_compiler import requirements
    from blueprint_pipeline.website_locomotion_task import runnable_options

    routing = route_job_capabilities("rigid_pick_place", embodiment_ids=[registry.G1_EMBODIMENT_ID])
    assert routing["configurations"] == []
    assert {"kind": "embodiment", "id": registry.G1_EMBODIMENT_ID, "missing": ["verified_site_episode",
            "reviewed_checkpoint_terms"], "reason": registry.embodiment(registry.G1_EMBODIMENT_ID).reason} \
        in routing["unavailable"]
    arm = runnable_options(requirements(locomote=False, manipulate=True))
    g1_arm = [row for row in arm["policies"] if row["embodiment_id"] == registry.G1_EMBODIMENT_ID]
    assert sorted(row["candidate_id"] for row in g1_arm) == sorted(MANIPULATION_CHECKPOINTS)
    for row in g1_arm:
        assert row == {"candidate_id": row["candidate_id"], "embodiment_id": registry.G1_EMBODIMENT_ID,
                       "runnable": False,
                       "missing": ["embodiment:verified_site_episode", "embodiment:reviewed_checkpoint_terms"]}
    rows = {row["candidate_id"]: row for row in json.loads(
        (Path(registry.__file__).resolve().parents[2] / "config/beta_policy_candidates.v1.json").read_text())["candidates"]}
    assert rows["humanoidarena_dp_g1_dex3_sonic"]["supported_task_kinds"] == ["rigid_pick_place"]
    assert rows["humanoidarena_pi05_g1_dex3_sonic"]["supported_task_kinds"] == ["rigid_pick_place"]
    assert rows["humanoidarena_dp_g1_dex3_sonic"]["configuration_digest"] == \
        "sha256:7642740f48911d8237591536827ef1327c6d92d41ae456093cf95d42009643be"
    assert rows["humanoidarena_pi05_g1_dex3_sonic"]["configuration_digest"] == \
        "sha256:98530d80577e61eba52599d9578d7bead1faec33502bf4edea4dbb2d8d604397"
    # The G1 row is main's but for the navigation allowance; no other body declares one.
    g1 = registry.embodiment(registry.G1_EMBODIMENT_ID)
    assert dataclasses.replace(g1, development_run_allowances=()) == registry.EmbodimentProfile(
        embodiment_id=registry.G1_EMBODIMENT_ID,
        robot="Unitree G1", tool="Dex3 hand", arm_count=2, mobility="humanoid",
        observation_schema_ids=("humanoidarena_head_rgb_state64_v1",),
        action_schema_ids=("humanoidarena_semantic_v3",),
        controller="SONIC whole-body controller", simulator_runtime_id="isaac_native_arena_g1_v1",
        sensor_reference="configs/g1_humanoidarena_checkpoint_inventory.v1.json",
        task_profile_ids=("rigid_pick_place", "goal_region_navigation"), status="unavailable",
        missing=("verified_site_episode", "reviewed_checkpoint_terms"),
        reason="No production policy-canary profile has executed a verified G1 episode on a site scene, and "
               "its checkpoint terms are pending review.",
        policy_robot_profile_ids=("unitree_g1_dex3_sonic_v1",),
    )
    assert all(not row.development_run_allowances for row in registry.EMBODIMENT_REGISTRY
               if row.embodiment_id != registry.G1_EMBODIMENT_ID)
