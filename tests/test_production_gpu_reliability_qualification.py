from blueprint_pipeline.production_gpu_reliability_qualification import qualify_release


def _campaign():
    return {"terminal": True, "attempts": [{"state": "passed"}] * 4}


def test_qualification_promotes_only_with_repeated_live_slo_evidence():
    result = qualify_release(
        release_fingerprint="sha256:" + "a" * 64,
        campaign_snapshots=[_campaign(), _campaign(), _campaign()],
        bind_latencies_seconds=[1, 2, 3],
        cold_replenishment_seconds=[300, 400, 500],
        rollback_drill_passed=True,
    )
    assert result["status"] == "promoted"
    assert result["metrics"]["attempt_pass_rate"] == 1.0


def test_qualification_quarantines_sparse_or_failed_release():
    result = qualify_release(
        release_fingerprint="sha256:" + "a" * 64,
        campaign_snapshots=[{"terminal": False, "attempts": [{"state": "failed"}]}],
        bind_latencies_seconds=[],
        cold_replenishment_seconds=[],
        rollback_drill_passed=False,
    )
    assert result["status"] == "quarantined"
    assert result["blockers"]
