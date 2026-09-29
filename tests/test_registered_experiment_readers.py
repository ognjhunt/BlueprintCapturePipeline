"""Actual exported G1 readers acquire current lifetime before payload reads."""
import importlib
from pathlib import Path
import pytest


@pytest.mark.parametrize("module,method", [
    ("native_g1_development_pair", "_read_result"),
    ("native_g1_development_pair", "_score_from_episode"),
    ("native_g1_development_pair", "_verified_review_media"),
    ("native_g1_paid_campaign", "verify_g1_paid_output"),
    ("native_g1_team_paid_output", "verify_g1_team_paid_output"),
    ("native_g1_provider_runtime", "_query_count"),
])
def test_existing_reader_refuses_missing_current_generation_before_payload(tmp_path, monkeypatch, module, method):
    from blueprint_pipeline import control_plane_lane_experiment_consumer as lifetime
    root = tmp_path / "lanes"
    target = root / "g1" / ("registered-" + "a" * 32)
    monkeypatch.setattr(lifetime, "LANE_ROOTS", (root, tmp_path / "inputs"))
    monkeypatch.setattr(lifetime, "AUTHORITY_ROOT", tmp_path / "absent-authority")
    monkeypatch.setattr(lifetime, "_blueprint_gid", lambda: 0)
    monkeypatch.setattr(Path, "read_text", lambda *a, **kw: pytest.fail("unadmitted payload read"))
    code = importlib.import_module("blueprint_pipeline." + module)
    arguments = {
        "_read_result": {"path": target / "result.json", "candidate_id": "candidate", "scene_plan_digest": "scene", "request_digest": "request"},
        "_score_from_episode": {"path": target / "episode.json", "worker": {}, "objective_id": "objective"},
        "_verified_review_media": {"episode_path": target / "episode.json", "episode": {}, "pair_root": target},
        "verify_g1_paid_output": {"result": {"status": "completed", "continuing_spend_from_this_run": False, "attempt_root": str(target)}, "bundle": {}},
        "verify_g1_team_paid_output": {"output_dir": target, "execution_packet": {}, "scene_plan_digest": "scene", "scene_packet_receipt_digest": "receipt"},
        "_query_count": {"pair_output": target, "candidate": "candidate"},
    }
    with pytest.raises(ValueError, match="experiment_consumer_admission_failed"):
        getattr(code, method)(**arguments[method])
