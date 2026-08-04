from __future__ import annotations

from blueprint_pipeline import franka_droid_control_preflight as preflight


def _episode(*, episode_id: str, success: bool) -> dict:
    return {
        "episode_id": episode_id,
        "status": "completed",
        "task_success": success,
        "action_steps_executed": 24,
        "policy_query_count": 3,
        "metrics": {
            "lift_delta_m": 0.1 if success else 0.0,
            "contained_in_tray_interior": success,
        },
        "visual_evidence": {
            "frame_manifest_digest": "sha256:" + "1" * 64,
            "video": {"sha256": "sha256:" + "2" * 64},
        },
    }


def test_preflight_persists_reusable_positive_negative_gate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        preflight,
        "_verified_menagerie_revision",
        lambda _root: preflight.PINNED_MENAGERIE_REVISION,
    )
    monkeypatch.setattr(
        preflight,
        "prepare_franka_droid_runtime",
        lambda **_kwargs: {"targets": {"pregrasp": [0.0] * 7}},
    )
    results = iter(
        [
            _episode(episode_id="positive", success=True),
            _episode(episode_id="negative", success=False),
        ]
    )
    monkeypatch.setattr(
        preflight,
        "run_franka_droid_closed_loop",
        lambda **_kwargs: next(results),
    )
    monkeypatch.setattr(
        preflight,
        "validate_episode_evidence_contract",
        lambda result: {
            "status": "admitted",
            "episode_admission_digest": "sha256:" + result["episode_id"][0] * 64,
        },
    )

    receipt = preflight.run_franka_droid_control_preflight(
        menagerie_root=tmp_path / "menagerie" / "franka_emika_panda",
        output_dir=tmp_path / "preflight",
    )

    assert receipt["status"] == "passed"
    assert receipt["positive_control"]["task_success"] is True
    assert receipt["negative_control"]["task_success"] is False
    assert receipt["production_policy_episode_executed"] is False
    assert (tmp_path / "preflight" / "franka_droid_control_preflight.json").is_file()
