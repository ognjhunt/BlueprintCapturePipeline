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


# --- G1 navigation (docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md, part C): no development-run allowance yet ---

#: A test-only allowance, so the type, ``development_run_missing`` and the validator stay checked while no
#: body declares one. Its bases are test markers, not decisions.
ALLOWANCE = registry.DevelopmentRunAllowance(
    task_profile_id="goal_region_navigation",
    satisfied_by=(("verified_site_episode", "test_basis_verified_site_episode"),
                  ("reviewed_checkpoint_terms", "test_basis_reviewed_checkpoint_terms")),
)


def _g1_with(allowance):
    return dataclasses.replace(registry.embodiment(registry.G1_EMBODIMENT_ID), development_run_allowances=(allowance,))


def test_the_g1_stays_unavailable_and_lists_its_gaps_with_a_navigation_profile() -> None:
    g1 = registry.embodiment(registry.G1_EMBODIMENT_ID)
    assert g1.status == "unavailable" and not g1.available
    assert g1.missing == ("verified_site_episode", "reviewed_checkpoint_terms")
    assert g1.task_profile_ids == ("rigid_pick_place", "goal_region_navigation")
    # The first-navigation-run allowance is deferred: it needs a bound and the owner's decision.
    assert [row.embodiment_id for row in registry.EMBODIMENT_REGISTRY if row.development_run_allowances] == []
    assert g1.development_run_missing("goal_region_navigation") == (g1.missing, None)


def test_only_a_development_only_navigation_run_may_proceed_without_an_allowances_items(monkeypatch) -> None:
    g1 = _g1_with(ALLOWANCE)
    monkeypatch.setattr(registry, "EMBODIMENT_REGISTRY", (g1,))
    assert registry.embodiment_registry_violations() == []
    missing, allowance = g1.development_run_missing("goal_region_navigation")
    assert missing == () and allowance is ALLOWANCE and allowance.claim_ceiling == "development_only"
    # Every other profile keeps all the body's gaps.
    assert g1.development_run_missing("rigid_pick_place") == (g1.missing, None)
    assert g1.development_run_missing("articulated_open_close") == (g1.missing, None)
    # An allowance waives only the items it names.
    partial = _g1_with(registry.DevelopmentRunAllowance(
        task_profile_id="goal_region_navigation", satisfied_by=(ALLOWANCE.satisfied_by[0],)))
    assert partial.development_run_missing("goal_region_navigation")[0] == ("reviewed_checkpoint_terms",)


@pytest.mark.parametrize("change", [
    {"task_profile_id": "rigid_pick_place"}, {"claim_ceiling": "diagnostic_policy_execution"},
    {"satisfied_by": (("integration_canary_receipt", "invented"),)}, {"satisfied_by": ()},
    {"satisfied_by": (("verified_site_episode", ""),)}])
def test_a_wider_or_unexplained_allowance_is_caught(monkeypatch, change) -> None:
    g1 = registry.embodiment(registry.G1_EMBODIMENT_ID)
    widened = dataclasses.replace(ALLOWANCE, **change)
    monkeypatch.setattr(registry, "EMBODIMENT_REGISTRY",
                        (dataclasses.replace(g1, development_run_allowances=(widened,)),))
    assert registry.embodiment_registry_violations() == [f"development_run_allowance_invalid:{g1.embodiment_id}"]
    # Even a valid allowance is refused on a runnable body.
    runnable = dataclasses.replace(_g1_with(ALLOWANCE), status="runnable", missing=())
    monkeypatch.setattr(registry, "EMBODIMENT_REGISTRY", (runnable,))
    assert f"development_run_allowance_invalid:{g1.embodiment_id}" in registry.embodiment_registry_violations()


def test_a_navigation_walk_lists_every_g1_gap_and_reads_an_allowance_only_once_one_exists(monkeypatch) -> None:
    from blueprint_pipeline.task_spec_compiler import requirements
    from blueprint_pipeline.website_locomotion_task import NAVIGATION_PIPELINE_PIECES, runnable_options

    walk = runnable_options(requirements(locomote=True, manipulate=False))
    g1_rows = [row for row in walk["policies"] if row["embodiment_id"] == registry.G1_EMBODIMENT_ID]
    assert g1_rows
    for row in g1_rows:
        assert {"embodiment:verified_site_episode", "embodiment:reviewed_checkpoint_terms"} <= set(row["missing"])
        assert "claim_ceiling" not in row and "development_run_allowance" not in row and not row["runnable"]
    built = runnable_options(requirements(locomote=True, manipulate=False),
                             built_pieces=frozenset(NAVIGATION_PIPELINE_PIECES))
    for row in built["policies"]:
        if row["embodiment_id"] == registry.G1_EMBODIMENT_ID:
            assert {"embodiment:verified_site_episode", "embodiment:reviewed_checkpoint_terms"} <= set(row["missing"])
            assert not row["runnable"]
    # The body itself is listed as it is: unavailable, with every gap.
    body = next(row for row in walk["embodiments"] if row["embodiment_id"] == registry.G1_EMBODIMENT_ID)
    assert body["status"] == "unavailable" and body["missing"] == ["verified_site_episode", "reviewed_checkpoint_terms"]
    # Once a body declares an allowance, a walk reads it: its items go, its result is recorded development_only.
    monkeypatch.setattr(registry, "EMBODIMENT_REGISTRY", tuple(
        _g1_with(ALLOWANCE) if row.embodiment_id == registry.G1_EMBODIMENT_ID else row
        for row in registry.EMBODIMENT_REGISTRY))
    allowed = runnable_options(requirements(locomote=True, manipulate=False))
    g1_rows = [row for row in allowed["policies"] if row["embodiment_id"] == registry.G1_EMBODIMENT_ID]
    assert g1_rows
    for row in g1_rows:
        assert not any(item.startswith("embodiment:") for item in row["missing"])
        assert row["claim_ceiling"] == "development_only"
        assert row["development_run_allowance"] == {"task_profile_id": "goal_region_navigation",
                                                    "satisfied_by": dict(ALLOWANCE.satisfied_by)}
    body = next(row for row in allowed["embodiments"] if row["embodiment_id"] == registry.G1_EMBODIMENT_ID)
    assert body["missing"] == ["verified_site_episode", "reviewed_checkpoint_terms"]


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
    for row in arm["policies"]:
        if row["embodiment_id"] == registry.G1_EMBODIMENT_ID:
            assert {"embodiment:verified_site_episode", "embodiment:reviewed_checkpoint_terms"} <= set(row["missing"])
            assert "claim_ceiling" not in row and "development_run_allowance" not in row and not row["runnable"]
    rows = {row["candidate_id"]: row for row in json.loads(
        (Path(registry.__file__).resolve().parents[2] / "config/beta_policy_candidates.v1.json").read_text())["candidates"]}
    assert rows["humanoidarena_dp_g1_dex3_sonic"]["supported_task_kinds"] == ["rigid_pick_place"]
    assert rows["humanoidarena_pi05_g1_dex3_sonic"]["supported_task_kinds"] == ["rigid_pick_place"]
    assert rows["humanoidarena_dp_g1_dex3_sonic"]["configuration_digest"] == \
        "sha256:7642740f48911d8237591536827ef1327c6d92d41ae456093cf95d42009643be"
    assert rows["humanoidarena_pi05_g1_dex3_sonic"]["configuration_digest"] == \
        "sha256:98530d80577e61eba52599d9578d7bead1faec33502bf4edea4dbb2d8d604397"
