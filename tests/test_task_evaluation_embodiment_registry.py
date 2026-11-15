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


def test_the_g1_run_scope_a_stood_in_run_records_is_pinned_development_only() -> None:
    """Pin: whatever proceeds on the allowance, every G1 run's authority claims no more than development_only."""
    from blueprint_pipeline import native_g1_beta_authority as authority

    assert authority.SCOPE["claim_ceiling"] == "development_only"
    assert authority.SCOPE["ranking_eligible"] is False and authority.SCOPE["qualification_permitted"] is False
    assert authority.SCOPE["physical_outcome_claimed"] is False
