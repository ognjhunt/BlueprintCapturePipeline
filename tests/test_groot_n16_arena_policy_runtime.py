from __future__ import annotations

import pytest

from blueprint_pipeline.groot_n16_arena_policy_runtime import (
    CHECKPOINT_REVISION,
    EMBODIMENT_TAG,
    GROOT_SOURCE_REVISION,
    MODEL_ID,
    GrootN16ArenaPolicySpec,
    validate_worker_identity_receipt,
)


def _receipt() -> dict[str, object]:
    return {
        "status": "verified",
        "model_id": MODEL_ID,
        "embodiment_tag": EMBODIMENT_TAG,
        "groot_source_revision": GROOT_SOURCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_files_sha256": "1" * 64,
        "environment_lock_sha256": "2" * 64,
    }


def test_n16_identity_is_exactly_the_arena_supported_droid_pair() -> None:
    identity = GrootN16ArenaPolicySpec().identity()
    assert identity["model_id"] == "nvidia/GR00T-N1.6-DROID"
    assert identity["embodiment_tag"] == "OXE_DROID"
    assert identity["groot_source_revision"] == ("e29d8fc50b0e4745120ae3fb72447986fe638aa6")
    assert identity["checkpoint_revision"] == ("ae3ebe8d288971ac53aa30c756ea5cba0f52611b")


def test_n16_worker_receipt_requires_materialized_bytes() -> None:
    assert (
        validate_worker_identity_receipt(_receipt(), expected=GrootN16ArenaPolicySpec())["status"]
        == "verified"
    )

    changed = _receipt()
    changed["checkpoint_files_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="checkpoint_files_sha256_invalid"):
        validate_worker_identity_receipt(changed, expected=GrootN16ArenaPolicySpec())


def test_n16_worker_receipt_rejects_n17_identity() -> None:
    changed = _receipt()
    changed["model_id"] = "nvidia/GR00T-N1.7-DROID"
    with pytest.raises(ValueError, match="model_id_mismatch"):
        validate_worker_identity_receipt(changed, expected=GrootN16ArenaPolicySpec())
