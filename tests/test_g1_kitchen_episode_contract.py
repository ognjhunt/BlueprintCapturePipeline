from blueprint_pipeline.g1_kitchen_episode_contract import build_episode_render_settings


def test_dynamic_contract_has_no_fixed_frame_expectation():
    result = build_episode_render_settings(
        steps=81,
        width=640,
        height=480,
        fps=20,
        warmup_frames=6,
        per_scenario_seconds=420,
        dynamic_episode_termination=True,
        episode_max_steps=10_000,
    )
    assert result["expected_frame_count_per_scenario"] is None
    assert result["episode_max_steps"] == 10_000


def test_fixed_diagnostic_retains_explicit_horizon():
    result = build_episode_render_settings(
        steps=81,
        width=640,
        height=480,
        fps=20,
        warmup_frames=6,
        per_scenario_seconds=420,
        dynamic_episode_termination=False,
        episode_max_steps=0,
    )
    assert result["expected_frame_count_per_scenario"] == 81
