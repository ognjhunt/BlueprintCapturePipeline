"""Hermetic tests for the bounded feedback-driven stance-configuration search.

The agent must: derive quantitative feasible windows from sweep rejections, only
re-parameterize the caller's sweep (never invent poses or loosen gates), switch
approach mode through registered strategies, stop on full PASS, and distinguish a
proved infeasibility from a fail-closed budget/stall stop.
"""
from __future__ import annotations

import math

import pytest

from blueprint_pipeline import stance_configuration_agent as agent


def _limits(**overrides):
    values = {
        "min_standoff_m": 0.12,
        "max_standoff_m": 1.2,
        "max_rounds": 8,
        "max_staging_anchors": 2,
    }
    values.update(overrides)
    return agent.StanceSearchLimits(**values)


def _reach_estimate(effector_m: float, *, limit_m: float = 0.35, margin_m: float = 0.10):
    status = "PASS" if effector_m <= limit_m + margin_m else "FAIL"
    return {
        "status": status,
        "nearest_seed_effector_to_affordance_m": round(effector_m, 4),
        "required_max_seed_effector_to_affordance_m": limit_m,
        "approx_preselection_effector_margin_m": margin_m,
    }


def _candidate(
    *,
    standoff_m: float,
    angle_deg: float = 0,
    contact_count: int = 0,
    reach_effector_m: float | None = None,
    placement_status: str | None = None,
    placement_blockers: tuple[str, ...] = (),
    required_gap_m: float | None = None,
    measured_clearance_m: float | None = None,
    pose: tuple[float, float, float] | None = None,
):
    record = {
        "candidate_kind": "task_stance",
        "pose": list(pose) if pose is not None else [2.0 + standoff_m, 0.0, 0.79],
        "yaw": 3.14159,
        "standoff_from_target_surface_m": round(standoff_m, 4),
        "angle_offset_deg": angle_deg,
        "scene_collision_contact_count": contact_count,
    }
    if reach_effector_m is not None:
        record["reachability_estimate"] = _reach_estimate(reach_effector_m)
    if placement_status is not None:
        validation = {"status": placement_status, "blockers": list(placement_blockers)}
        if required_gap_m is not None:
            validation["required_target_gap_m"] = required_gap_m
        if measured_clearance_m is not None:
            validation["deterministic_geometry"] = {
                "min_obstacle_clearance_m": measured_clearance_m
            }
        record["placement_validation"] = validation
    return record


def _blocked_plan(candidates, blockers=("no_reach_seed_task_stance_candidate",)):
    return {
        "schema_version": "task_stance_plan.v1",
        "status": "blocked",
        "blockers": list(blockers),
        "task_affordance_xyz": [2.0, 0.0, 1.2],
        "candidates": list(candidates),
    }


def _accepted_plan():
    return {
        "schema_version": "task_stance_plan.v1",
        "status": "accepted",
        "accepted_pose": [2.13, 0.0, 0.79],
        "accepted_yaw": 3.14159,
        "candidates": [],
        "accepted_candidate_count": 1,
    }


def _initial_parameters(distances=(0.16, 0.24, 0.38)):
    return agent.StanceSweepParameters(
        approach_mode=agent.APPROACH_MODE_DIRECT,
        standoff_candidates_m=tuple(distances),
        angle_offsets_deg=None,
    )


# ---------------------------------------------------------------- analyzer


def test_reach_shortfall_uses_effector_limit_plus_margin() -> None:
    estimate = _reach_estimate(0.598)
    shortfall = agent._reach_shortfall_m(estimate)
    assert shortfall == pytest.approx(0.598 - 0.45, abs=1e-6)


def test_reach_shortfall_is_robot_agnostic_via_recorded_limits() -> None:
    # A short-reach tabletop arm (Panda-like) records tighter limits in its own
    # estimate; the agent must derive the shortfall from those recorded numbers,
    # not from any embodiment constant.
    panda_like = {
        "status": "FAIL",
        "nearest_seed_effector_to_affordance_m": 0.30,
        "required_max_seed_effector_to_affordance_m": 0.12,
        "approx_preselection_effector_margin_m": 0.03,
    }
    assert agent._reach_shortfall_m(panda_like) == pytest.approx(0.15, abs=1e-6)
    long_reach_humanoid = {
        "status": "PASS",
        "nearest_seed_effector_to_affordance_m": 0.30,
        "required_max_seed_effector_to_affordance_m": 0.60,
        "approx_preselection_effector_margin_m": 0.10,
    }
    assert agent._reach_shortfall_m(long_reach_humanoid) == pytest.approx(-0.40, abs=1e-6)


def test_analyzer_derives_reach_upper_bound_from_shortfall() -> None:
    plan = _blocked_plan(
        [_candidate(standoff_m=0.38, reach_effector_m=0.55)]
    )
    evidence = agent.analyze_stance_plan_feedback(plan, limits=_limits())
    record = evidence[0.0]
    assert record.reach_failures == 1
    # shortfall 0.55 - (0.35 + 0.10) = 0.10 -> upper bound 0.38 - 0.10 = 0.28
    assert record.upper_bound_m == pytest.approx(0.28, abs=1e-6)
    assert record.upper_bound_evidence[-1]["kind"] == "reach"
    assert record.upper_bound_evidence[-1]["reach_shortfall_m"] == pytest.approx(0.10, abs=1e-4)


def test_analyzer_derives_collision_lower_bound() -> None:
    plan = _blocked_plan(
        [_candidate(standoff_m=0.16, contact_count=3)],
        blockers=("no_collision_free_task_stance_candidate",),
    )
    evidence = agent.analyze_stance_plan_feedback(plan, limits=_limits())
    record = evidence[0.0]
    assert record.collision_failures == 1
    assert record.lower_bound_m == pytest.approx(0.21, abs=1e-6)
    assert record.lower_bound_evidence[-1]["kind"] == "collision"


def test_analyzer_uses_quantitative_clearance_shortfall_when_present() -> None:
    plan = _blocked_plan(
        [
            _candidate(
                standoff_m=0.24,
                placement_status="blocked",
                placement_blockers=("placed_robot_target_gap_below_threshold",),
                required_gap_m=0.10,
                measured_clearance_m=0.08,
            )
        ],
        blockers=("no_validated_task_stance_candidate",),
    )
    evidence = agent.analyze_stance_plan_feedback(plan, limits=_limits())
    record = evidence[0.0]
    assert record.placement_failures == 1
    assert record.min_clearance_shortfall_m == pytest.approx(0.02, abs=1e-6)
    # 0.24 + 0.02 shortfall + 0.01 epsilon
    assert record.lower_bound_m == pytest.approx(0.27, abs=1e-6)


def test_analyzer_accumulates_evidence_across_rounds() -> None:
    limits = _limits()
    first = agent.analyze_stance_plan_feedback(
        _blocked_plan([_candidate(standoff_m=0.16, contact_count=1)]),
        limits=limits,
    )
    second = agent.analyze_stance_plan_feedback(
        _blocked_plan([_candidate(standoff_m=0.55, reach_effector_m=0.60)]),
        limits=limits,
        previous=first,
    )
    record = second[0.0]
    assert record.collision_failures == 1 and record.reach_failures == 1
    assert record.lower_bound_m > limits.min_standoff_m
    assert math.isfinite(record.upper_bound_m)


# ---------------------------------------------------------------- strategies


def test_descend_strategy_proposes_untried_standoffs_below_ladder() -> None:
    limits = _limits()
    plan = _blocked_plan(
        [
            _candidate(standoff_m=s, reach_effector_m=0.47 + (s - 0.16))
            for s in (0.16, 0.24, 0.38)
        ]
    )
    evidence = agent.analyze_stance_plan_feedback(plan, limits=limits)
    eligible = agent.compute_eligible_strategies(
        evidence=evidence,
        limits=limits,
        approach_mode=agent.APPROACH_MODE_DIRECT,
        tried_signatures=set(),
        tried_angle_fans=[],
        staging_anchors_remaining=[],
        default_standoffs_m=(0.16, 0.24, 0.38),
    )
    assert eligible, "reach-bounded direction must yield a strategy"
    chosen = eligible[0]
    assert chosen.strategy_id == agent.STRATEGY_DESCEND_TO_REACH
    lo, hi = evidence[0.0].window()
    for standoff in chosen.parameters.standoff_candidates_m:
        assert limits.min_standoff_m <= standoff <= limits.max_standoff_m
        assert lo - 1e-9 <= standoff <= hi + 1e-9
    assert all(
        abs(standoff - tried) > 1e-3
        for standoff in chosen.parameters.standoff_candidates_m
        for tried in (0.16, 0.24, 0.38)
    )


def test_window_probe_strategy_when_both_bounds_measured() -> None:
    limits = _limits()
    plan = _blocked_plan(
        [
            _candidate(standoff_m=0.16, contact_count=2),
            _candidate(standoff_m=0.55, reach_effector_m=0.55),
        ]
    )
    evidence = agent.analyze_stance_plan_feedback(plan, limits=limits)
    lo, hi = evidence[0.0].window()
    assert lo == pytest.approx(0.21, abs=1e-6)
    assert hi == pytest.approx(0.45, abs=1e-6)
    eligible = agent.compute_eligible_strategies(
        evidence=evidence,
        limits=limits,
        approach_mode=agent.APPROACH_MODE_DIRECT,
        tried_signatures=set(),
        tried_angle_fans=[],
        staging_anchors_remaining=[],
        default_standoffs_m=(0.16, 0.55),
    )
    assert eligible[0].strategy_id == agent.STRATEGY_PROBE_FEASIBLE_WINDOW
    assert all(lo <= s <= hi for s in eligible[0].parameters.standoff_candidates_m)


def test_backoff_strategy_when_only_collision_bound() -> None:
    limits = _limits()
    plan = _blocked_plan(
        [_candidate(standoff_m=0.16, contact_count=4)],
        blockers=("no_collision_free_task_stance_candidate",),
    )
    evidence = agent.analyze_stance_plan_feedback(plan, limits=limits)
    eligible = agent.compute_eligible_strategies(
        evidence=evidence,
        limits=limits,
        approach_mode=agent.APPROACH_MODE_DIRECT,
        tried_signatures=set(),
        tried_angle_fans=[],
        staging_anchors_remaining=[],
        default_standoffs_m=(0.16,),
    )
    assert eligible[0].strategy_id == agent.STRATEGY_BACK_OFF_TO_CLEARANCE
    assert all(s > 0.21 - 1e-9 for s in eligible[0].parameters.standoff_candidates_m)


def test_angle_fan_refinement_proposes_untried_midpoints() -> None:
    limits = _limits()
    plan = _blocked_plan(
        [
            _candidate(standoff_m=0.16, angle_deg=0, reach_effector_m=0.48),
            _candidate(standoff_m=0.16, angle_deg=-15, reach_effector_m=0.52),
            _candidate(standoff_m=0.16, angle_deg=15, reach_effector_m=0.60),
        ]
    )
    evidence = agent.analyze_stance_plan_feedback(plan, limits=limits)
    fan = agent._refined_angle_fan(evidence, tried_fans=[(0.0, -15.0, 15.0)])
    assert fan is not None
    assert all(abs(a) == pytest.approx(7.5) for a in fan)


def test_restage_strategy_switches_mode_and_uses_measured_anchor_only() -> None:
    limits = _limits()
    anchors = [{"xyz": [3.4, 1.1, 0.79], "source": "collision_free_sweep_candidate"}]
    eligible = agent.compute_eligible_strategies(
        evidence={},
        limits=limits,
        approach_mode=agent.APPROACH_MODE_DIRECT,
        tried_signatures=set(),
        tried_angle_fans=[],
        staging_anchors_remaining=anchors,
        default_standoffs_m=(0.16, 0.24),
    )
    assert len(eligible) == 1
    restage = eligible[0]
    assert restage.strategy_id == agent.STRATEGY_RESTAGE_APPROACH_ANCHOR
    assert restage.parameters.approach_mode == agent.APPROACH_MODE_FINAL
    assert restage.parameters.approach_anchor_xyz == (3.4, 1.1, 0.79)


# ---------------------------------------------------------------- search loop


def test_search_accepts_when_descended_standoff_passes() -> None:
    ladder = (0.16, 0.24, 0.38)
    initial = _blocked_plan(
        [_candidate(standoff_m=s, reach_effector_m=0.47 + (s - 0.16)) for s in ladder]
    )
    sweeps: list[agent.StanceSweepParameters] = []

    def attempt_sweep(parameters):
        sweeps.append(parameters)
        # The re-swept closer standoff passes the caller's gates.
        return _accepted_plan()

    result = agent.run_stance_configuration_search(
        attempt_sweep=attempt_sweep,
        initial_plan=initial,
        initial_parameters=_initial_parameters(ladder),
        limits=_limits(),
    )
    assert result["status"] == "accepted"
    assert result["accepted_plan"]["status"] == "accepted"
    assert result["round_count"] == 2
    assert sweeps, "the agent must have re-invoked the caller's sweep"
    proposed = sweeps[0].standoff_candidates_m
    assert proposed and all(s < min(ladder) for s in proposed)
    assert result["rounds"][1]["strategy_id"] == agent.STRATEGY_DESCEND_TO_REACH
    # The accepted plan came from the sweep, not from the agent.
    assert "accepted_pose" in result["accepted_plan"]


def test_search_proves_infeasibility_from_contradictory_bounds() -> None:
    # Collision requires standoff > 0.35; reach requires standoff <= 0.28.
    initial = _blocked_plan(
        [
            _candidate(standoff_m=0.30, contact_count=2),
            _candidate(standoff_m=0.38, reach_effector_m=0.55),
        ]
    )

    def attempt_sweep(parameters):  # pragma: no cover - must not be called
        raise AssertionError("no strategy should be eligible")

    result = agent.run_stance_configuration_search(
        attempt_sweep=attempt_sweep,
        initial_plan=initial,
        initial_parameters=_initial_parameters((0.30, 0.38)),
        limits=_limits(),
    )
    assert result["status"] == "infeasible"
    proof = result["infeasibility_proof"]
    assert proof["per_direction"][0]["window_is_empty"] is True
    assert proof["assumptions"] == agent.INTERVAL_MODEL_ASSUMPTION
    assert proof["per_direction"][0]["lower_bound_evidence"]
    assert proof["per_direction"][0]["upper_bound_evidence"]


def test_search_budget_exhausted_is_not_a_proof() -> None:
    initial = _blocked_plan(
        [_candidate(standoff_m=0.38, reach_effector_m=0.55)]
    )
    round_standoff = {"value": 0.38}

    def attempt_sweep(parameters):
        # Every re-sweep fails reach again slightly closer, keeping a nonempty
        # window alive so the loop runs to its round budget.
        round_standoff["value"] = min(parameters.standoff_candidates_m)
        return _blocked_plan(
            [
                _candidate(
                    standoff_m=round_standoff["value"],
                    reach_effector_m=0.47,
                )
            ]
        )

    result = agent.run_stance_configuration_search(
        attempt_sweep=attempt_sweep,
        initial_plan=initial,
        initial_parameters=_initial_parameters((0.38,)),
        limits=_limits(max_rounds=3),
    )
    assert result["status"] in {"budget_exhausted", "infeasible", "search_stalled"}
    if result["status"] != "infeasible":
        assert "infeasibility_proof" not in result
    assert result["round_count"] <= 4


def test_search_stall_without_conclusive_windows_is_not_infeasible() -> None:
    initial = _blocked_plan(
        [_candidate(standoff_m=0.38, reach_effector_m=0.55)]
    )

    def attempt_sweep(parameters):
        # Misbehaving sweep: returns no candidate records at all, so evidence
        # never advances and the same parameters cannot be retried.
        return {"status": "blocked", "blockers": ["x"], "candidates": []}

    result = agent.run_stance_configuration_search(
        attempt_sweep=attempt_sweep,
        initial_plan=initial,
        initial_parameters=_initial_parameters((0.38,)),
        limits=_limits(max_rounds=8),
    )
    assert result["status"] in {"search_stalled", "budget_exhausted"}
    assert "infeasibility_proof" not in result


def test_search_restages_and_reaches_final_mode() -> None:
    # All direct-approach directions are contradictory -> only restage remains.
    initial = _blocked_plan(
        [
            _candidate(standoff_m=0.30, contact_count=2),
            _candidate(
                standoff_m=0.38,
                reach_effector_m=0.55,
                pose=[2.38, 0.0, 0.79],
            ),
        ]
    )
    anchors = [{"xyz": [3.2, 0.9, 0.79], "source": "collision_free_sweep_candidate"}]
    seen_modes: list[str] = []

    def attempt_sweep(parameters):
        seen_modes.append(parameters.approach_mode)
        assert parameters.approach_anchor_xyz == (3.2, 0.9, 0.79)
        return _accepted_plan()

    result = agent.run_stance_configuration_search(
        attempt_sweep=attempt_sweep,
        initial_plan=initial,
        initial_parameters=_initial_parameters((0.30, 0.38)),
        limits=_limits(),
        staging_anchor_candidates=anchors,
    )
    assert result["status"] == "accepted"
    assert seen_modes == [agent.APPROACH_MODE_FINAL]
    assert result["accepted_parameters"]["approach_mode"] == agent.APPROACH_MODE_FINAL
    assert (
        result["rounds"][1]["strategy_id"] == agent.STRATEGY_RESTAGE_APPROACH_ANCHOR
    )


def test_search_never_repeats_identical_sweep_parameters() -> None:
    initial = _blocked_plan(
        [_candidate(standoff_m=0.38, reach_effector_m=0.55)]
    )
    signatures: list[tuple] = []

    def attempt_sweep(parameters):
        signatures.append(parameters.signature())
        return _blocked_plan(
            [
                _candidate(
                    standoff_m=min(parameters.standoff_candidates_m),
                    reach_effector_m=0.47,
                )
            ]
        )

    agent.run_stance_configuration_search(
        attempt_sweep=attempt_sweep,
        initial_plan=initial,
        initial_parameters=_initial_parameters((0.38,)),
        limits=_limits(max_rounds=6),
    )
    assert len(signatures) == len(set(signatures))


def test_accepted_initial_plan_short_circuits_without_sweeps() -> None:
    def attempt_sweep(parameters):  # pragma: no cover - must not be called
        raise AssertionError("no sweep should run for an accepted plan")

    result = agent.run_stance_configuration_search(
        attempt_sweep=attempt_sweep,
        initial_plan=_accepted_plan(),
        initial_parameters=_initial_parameters(),
        limits=_limits(),
    )
    assert result["status"] == "accepted"
    assert result["round_count"] == 1


def test_sweep_exception_fails_closed_as_error() -> None:
    initial = _blocked_plan(
        [_candidate(standoff_m=0.38, reach_effector_m=0.55)]
    )

    def attempt_sweep(parameters):
        raise RuntimeError("physx probe crashed")

    result = agent.run_stance_configuration_search(
        attempt_sweep=attempt_sweep,
        initial_plan=initial,
        initial_parameters=_initial_parameters((0.38,)),
        limits=_limits(),
    )
    assert result["status"] == "error"
    assert "physx probe crashed" in result["error"]
    assert "infeasibility_proof" not in result


# ---------------------------------------------------------------- chooser seam


def test_external_chooser_may_only_pick_registered_eligible_strategies() -> None:
    initial = _blocked_plan(
        [_candidate(standoff_m=0.38, reach_effector_m=0.55)]
    )

    def rogue_chooser(eligible, context):
        return "teleport_robot_inside_microwave"

    def attempt_sweep(parameters):
        return _accepted_plan()

    result = agent.run_stance_configuration_search(
        attempt_sweep=attempt_sweep,
        initial_plan=initial,
        initial_parameters=_initial_parameters((0.38,)),
        limits=_limits(),
        strategy_chooser=rogue_chooser,
        chooser_kind="external",
    )
    assert result["status"] == "accepted"
    assert result["chooser_kind"] == "external"
    violations = result["chooser_violations"]
    assert violations and violations[0]["kind"] == (
        "chooser_returned_unregistered_or_ineligible_strategy"
    )
    # Deterministic fallback strategy actually ran.
    assert result["rounds"][1]["strategy_id"] in agent.REGISTERED_STRATEGY_IDS


def test_chooser_exception_falls_back_deterministically() -> None:
    initial = _blocked_plan(
        [_candidate(standoff_m=0.38, reach_effector_m=0.55)]
    )

    def crashing_chooser(eligible, context):
        raise ValueError("llm timeout")

    def attempt_sweep(parameters):
        return _accepted_plan()

    result = agent.run_stance_configuration_search(
        attempt_sweep=attempt_sweep,
        initial_plan=initial,
        initial_parameters=_initial_parameters((0.38,)),
        limits=_limits(),
        strategy_chooser=crashing_chooser,
        chooser_kind="external",
    )
    assert result["status"] == "accepted"
    assert result["chooser_violations"][0]["kind"] == "chooser_error"


def test_chooser_context_exposes_only_registered_strategy_metadata() -> None:
    initial = _blocked_plan(
        [_candidate(standoff_m=0.38, reach_effector_m=0.55)]
    )
    contexts: list[dict] = []

    def recording_chooser(eligible, context):
        contexts.append(dict(context))
        return eligible[0].strategy_id

    def attempt_sweep(parameters):
        return _accepted_plan()

    agent.run_stance_configuration_search(
        attempt_sweep=attempt_sweep,
        initial_plan=initial,
        initial_parameters=_initial_parameters((0.38,)),
        limits=_limits(),
        strategy_chooser=recording_chooser,
        chooser_kind="external",
    )
    assert contexts
    for item in contexts[0]["eligible"]:
        assert item["strategy_id"] in agent.REGISTERED_STRATEGY_IDS
        assert "parameters" in item and "rationale" in item


# ---------------------------------------------------------------- manifest claims


def test_result_claim_boundary_never_upgrades_authority() -> None:
    result = agent.run_stance_configuration_search(
        attempt_sweep=lambda parameters: _accepted_plan(),
        initial_plan=_accepted_plan(),
        initial_parameters=_initial_parameters(),
        limits=_limits(),
    )
    assert result["schema_version"] == agent.STANCE_SEARCH_SCHEMA_VERSION
    boundary = result["claim_boundary"]
    assert "unchanged sweep" in boundary
    assert "budget_exhausted" in boundary
    assert result["registered_strategy_ids"] == list(agent.REGISTERED_STRATEGY_IDS)
