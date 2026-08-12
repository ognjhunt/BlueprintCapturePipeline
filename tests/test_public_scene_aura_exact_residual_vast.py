from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.public_scene_aura_exact_residual_vast import (
    RESULT_SCHEMA_VERSION,
    run_aura_exact_residual_vast,
)


def test_exact_residual_vast_dry_run_has_no_provider_mutation(tmp_path: Path) -> None:
    result = run_aura_exact_residual_vast(
        job_dir=tmp_path,
        paid_resource_admission_grant=None,
        execute=False,
        prepared_bundle={
            "bundle_sha256": "sha256:" + "a" * 64,
            "preflight_digest": "sha256:" + "b" * 64,
            "allowed_active_instance_ids": [47373597],
        },
        max_hourly_rate_usd=1.5,
        hard_cap_usd=6.0,
        hard_ttl_seconds=14_400,
    )

    assert result["schema_version"] == RESULT_SCHEMA_VERSION
    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0
    assert result["retry_cap"] == 0
    retained = json.loads(
        (tmp_path / "public_scene_aura_exact_residual_vast_result.json").read_text()
    )
    assert retained == result
