from __future__ import annotations

from scripts.validate_terraform_state_backend import (
    MINIMUM_RETENTION_SECONDS,
    validate_bucket,
)


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
        "retentionPolicy": {"retentionPeriod": str(MINIMUM_RETENTION_SECONDS)},
        "encryption": {"defaultKmsKeyName": KMS},
    }


def test_state_backend_requires_us_locked_versioned_retained_cmek_bucket() -> None:
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
        (payload["retentionPolicy"], "retentionPeriod", "60"),  # type: ignore[index]
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
