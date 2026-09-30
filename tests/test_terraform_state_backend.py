from __future__ import annotations

from scripts.validate_terraform_state_backend import (
    MINIMUM_RETENTION_SECONDS,
    validate_bucket,
)
from pathlib import Path


BUCKET = "blueprint-terraform-state"
KMS = "projects/blueprint/locations/us/keyRings/state/cryptoKeys/terraform"


def _metadata() -> dict[str, object]:
    return {
        "name": BUCKET,
        "location": "US-CENTRAL1",
        "iamConfiguration": {
            "uniformBucketLevelAccess": {"enabled": True},
            "publicAccessPrevention": "enforced",
        },
        "versioning": {"enabled": True},
        "softDeletePolicy": {"retentionDurationSeconds": str(MINIMUM_RETENTION_SECONDS)},
        "encryption": {"defaultKmsKeyName": KMS},
    }


def test_state_backend_requires_us_versioned_recoverable_cmek_bucket() -> None:
    payload = _metadata()
    assert validate_bucket(
        payload,
        expected_bucket=f"gs://{BUCKET}",
        expected_kms_key=KMS,
    ) == []

    mutations = (
        (payload, "location", "EUROPE-WEST1"),
        (payload["iamConfiguration"]["uniformBucketLevelAccess"], "enabled", False),  # type: ignore[index]
        (payload["iamConfiguration"], "publicAccessPrevention", "inherited"),  # type: ignore[index]
        (payload["versioning"], "enabled", False),  # type: ignore[index]
        (payload["softDeletePolicy"], "retentionDurationSeconds", "60"),  # type: ignore[index]
        (payload["encryption"], "defaultKmsKeyName", "other"),  # type: ignore[index]
    )
    for target, key, invalid in mutations:
        previous = target[key]  # type: ignore[index]
        target[key] = invalid  # type: ignore[index]
        assert validate_bucket(
            payload,
            expected_bucket=BUCKET,
            expected_kms_key=KMS,
        )
        target[key] = previous  # type: ignore[index]


def test_existing_versioned_bucket_retention_remains_supported() -> None:
    payload = _metadata()
    payload.pop("softDeletePolicy")
    payload["retentionPolicy"] = {"retentionPeriod": str(MINIMUM_RETENTION_SECONDS)}
    assert validate_bucket(
        payload, expected_bucket=BUCKET, expected_kms_key=KMS,
    ) == []


def test_soft_delete_duration_must_be_present_and_parseable() -> None:
    for value in (None, {}, {"retentionDurationSeconds": "invalid"}):
        payload = _metadata()
        payload["softDeletePolicy"] = value
        assert "terraform_state_retention_below_30_days" in validate_bucket(
            payload, expected_bucket=BUCKET, expected_kms_key=KMS,
        )


def test_deploy_requests_json_api_bucket_metadata_for_the_validator() -> None:
    script = Path(__file__).resolve().parents[1] / "deploy/scripts/deploy.sh"
    assert ('gcloud storage buckets describe "gs://${TERRAFORM_STATE_BUCKET}" '
            '--raw --format=json') in script.read_text()
