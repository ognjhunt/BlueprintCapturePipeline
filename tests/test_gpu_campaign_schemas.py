import json
from pathlib import Path

import jsonschema


SCHEMAS = Path("docs/schemas")


def test_campaign_config_payload_matches_repository_schema():
    payload = {
        "schema_version": "provider_neutral_gpu_campaign_config.v1",
        "campaign_id": "campaign-1",
        "allocation_key": "g4-1",
        "source_sha": "a" * 40,
        "image_digest": "sha256:" + "b" * 64,
        "hourly_rate_usd": 4.5,
        "max_provider_seconds": 3900,
        "spend_authorization_usd": 20,
        "image_total_compressed_bytes": 10_000_000_000,
        "image_largest_layer_bytes": 5_000_000_000,
        "image_residency_evidence": None,
        "prior_exposure_usd": 0.0,
        "smoke_seed": 1000,
        "episode_seeds": [1001, 1002, 1003],
        "stage_deadlines_seconds": {
            "host_ready": 600,
            "image_ready": 1200,
            "runtime_health": 300,
            "canary": 300,
            "smoke": 300,
            "episodes": 2700,
            "artifact_retrieval": 300,
            "teardown": 300,
        },
        "reuse_validated_same_allocation_canary": False,
        "canary_handoff": None,
    }
    schema = json.loads((SCHEMAS / "provider_neutral_gpu_campaign_config.schema.json").read_text())
    jsonschema.validate(payload, schema)


def test_same_allocation_handoff_matches_repository_schema():
    payload = {
        "schema_version": "same_allocation_canary_handoff.v1",
        "source_sha": "a" * 40,
        "image_digest": "sha256:" + "b" * 64,
        "allocation_key": "g4-1",
        "allocation_id": "vm-1",
        "launch_nonce": "nonce-1",
        "teardown_owner": "owner-1",
        "provider_started_at_epoch_seconds": 1000.0,
        "runtime_health_passed": True,
        "review_media_valid": True,
        "allocation_still_owned": True,
        "teardown_requested": False,
    }
    schema = json.loads((SCHEMAS / "same_allocation_canary_handoff.schema.json").read_text())
    jsonschema.validate(payload, schema)
