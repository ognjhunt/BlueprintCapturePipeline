from blueprint_pipeline.captured_site_policy_ranking import (
    aggregate_policy_rankings,
    score_episode,
)


def _episode(policy_id: str, *, lift: float, final_x: float, final_y: float, contained: bool):
    return {
        "schema_version": "franka_droid_closed_loop.v1",
        "status": "completed",
        "policy_id": policy_id,
        "initial_can_position_m": [0.5, 0.075, 0.09],
        "gates": {"contract_valid": True},
        "metrics": {
            "lift_delta_m": lift,
            "final_spraycan_center_m": [final_x, final_y, 0.12],
            "contained_in_tray_interior": contained,
            "final_linear_speed_m_s": 0.001,
        },
    }


def test_phase_gating_gives_stationary_episode_no_stability_credit() -> None:
    scored = score_episode(
        _episode("stationary", lift=0.0, final_x=0.5, final_y=0.075, contained=False)
    )
    assert scored["stability"] is True
    assert scored["episode_progress_score"] == 0.0


def test_successful_episode_reaches_one() -> None:
    scored = score_episode(
        _episode("success", lift=0.06, final_x=0.45, final_y=0.32, contained=True)
    )
    assert scored["episode_progress_score"] == 1.0


def test_aggregate_emits_only_strictly_separated_ranking() -> None:
    strong = [_episode("strong", lift=0.06, final_x=0.45, final_y=0.32, contained=True)] * 3
    weak = [_episode("weak", lift=0.0, final_x=0.5, final_y=0.075, contained=False)] * 3
    ranked = aggregate_policy_rankings({"strong": strong, "weak": weak})
    assert ranked["total_ranking_emitted"] is True
    assert ranked["ranking"] == ["strong", "weak"]

    tied = aggregate_policy_rankings({"strong": strong, "also_strong": [
        _episode("also_strong", lift=0.06, final_x=0.45, final_y=0.32, contained=True)
    ] * 3})
    assert tied["total_ranking_emitted"] is False
    assert tied["abstained"] is True
