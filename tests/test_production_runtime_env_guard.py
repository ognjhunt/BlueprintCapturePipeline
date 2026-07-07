from blueprint_pipeline.production_runtime_env_guard import (
    build_production_runtime_env_guard,
)


def test_production_runtime_env_guard_blocks_ambiguous_runtime_env():
    report = build_production_runtime_env_guard(env={})

    assert report["status"] == "blocked"
    assert "missing_BLUEPRINT_LAUNCH_PROOF_MODE_production" in report["blockers"]
    assert "missing_or_false_PRIVACY_PIPELINE_ENABLED" in report["blockers"]
    assert "missing_or_false_PIPELINE_SYNC_REQUIRED" in report["blockers"]


def test_production_runtime_env_guard_accepts_fail_closed_production_env():
    report = build_production_runtime_env_guard(
        env={
            "BLUEPRINT_LAUNCH_PROOF_MODE": "production",
            "PRIVACY_PIPELINE_ENABLED": "true",
            "PRIVACY_FAIL_CLOSED": "true",
            "PIPELINE_SYNC_REQUIRED": "true",
            "RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO": "true",
        }
    )

    assert report["status"] == "ready"
    assert report["blockers"] == []
