from __future__ import annotations

import pytest

from blueprint_pipeline.adp009d_checkpoint_materialization import (
    BLOCKER_BYTE_COUNT_MISMATCH,
    BLOCKER_CREDENTIALS_FORWARDED,
    BLOCKER_NOT_ON_WORKER,
    SOURCE_PUBLIC_GCS,
    SOURCE_PUBLIC_HUGGINGFACE,
    CheckpointMaterializationError,
    plan_checkpoint_materialization,
    verify_materialized_checkpoint,
)
from blueprint_pipeline.adp009d_policy_candidate_admission import EXPECTED_CANDIDATES

_PI05_BYTES = EXPECTED_CANDIDATES["pi05_droid"]["checkpoint_total_bytes"]
_PI05_REV = EXPECTED_CANDIDATES["pi05_droid"]["checkpoint_revision"]


def _objects(total: int, count: int = 4) -> list[dict]:
    """Split ``total`` across ``count`` named objects."""

    each, remainder = divmod(total, count)
    return [
        {"name": f"params/shard_{index:03d}", "size_bytes": each + (remainder if index == 0 else 0)}
        for index in range(count)
    ]


def test_pi05_plans_a_credential_free_public_gcs_fetch() -> None:
    plan = plan_checkpoint_materialization("pi05_droid")

    assert plan["source"] == SOURCE_PUBLIC_GCS
    assert plan["credentials_required"] is False
    assert plan["materialize_on"] == "gpu_worker"
    assert plan["stage_locally"] is False
    assert plan["listing_url"].startswith(
        "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_droid/"
    )
    assert plan["expected_total_bytes"] == _PI05_BYTES


def test_every_frozen_candidate_is_credential_free() -> None:
    """Measured: all four list anonymously, so no token reaches a rented host."""

    for candidate_id in EXPECTED_CANDIDATES:
        plan = plan_checkpoint_materialization(candidate_id)
        assert plan["credentials_required"] is False
        assert plan["materialize_on"] == "gpu_worker"
        assert plan["source"] in {SOURCE_PUBLIC_GCS, SOURCE_PUBLIC_HUGGINGFACE}


def test_huggingface_candidates_plan_a_revision_pinned_listing() -> None:
    plan = plan_checkpoint_materialization("groot_n17_droid")

    assert plan["source"] == SOURCE_PUBLIC_HUGGINGFACE
    assert plan["listing_url"] == (
        "https://huggingface.co/api/models/nvidia/GR00T-N1.7-DROID"
        "/revision/05e7cc97e40dbd33b0890c35cc0214fcb0547ab5"
    )


def test_unknown_candidate_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(CheckpointMaterializationError):
        plan_checkpoint_materialization("some_other_policy")
    with pytest.raises(CheckpointMaterializationError):
        verify_materialized_checkpoint(
            candidate_id="some_other_policy",
            objects=_objects(1),
            materialized_on="gpu_worker",
            credentials_forwarded=False,
        )


def test_exact_byte_count_admits_and_binds_the_manifest() -> None:
    receipt = verify_materialized_checkpoint(
        candidate_id="pi05_droid",
        objects=_objects(_PI05_BYTES),
        materialized_on="gpu_worker",
        credentials_forwarded=False,
        observed_revision=_PI05_REV,
    )

    assert receipt["status"] == "materialized"
    assert receipt["total_bytes"] == _PI05_BYTES
    assert receipt["object_count"] == 4
    assert receipt["object_manifest_sha256"].startswith("sha256:")
    assert receipt["credentials_forwarded"] is False
    assert receipt["staged_locally"] is False
    assert receipt["candidate_policy_queried"] is False

    from blueprint_pipeline.decision_evidence_contracts import canonical_digest

    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_a_single_wrong_byte_fails_closed() -> None:
    """The pinned total is the identity; near-enough is not enough."""

    with pytest.raises(CheckpointMaterializationError) as excinfo:
        verify_materialized_checkpoint(
            candidate_id="pi05_droid",
            objects=_objects(_PI05_BYTES - 1),
            materialized_on="gpu_worker",
            credentials_forwarded=False,
        )
    assert any(BLOCKER_BYTE_COUNT_MISMATCH in e for e in excinfo.value.errors)


def test_local_staging_is_refused() -> None:
    """Local bytes cannot evidence what the worker actually ran."""

    with pytest.raises(CheckpointMaterializationError) as excinfo:
        verify_materialized_checkpoint(
            candidate_id="pi05_droid",
            objects=_objects(_PI05_BYTES),
            materialized_on="orchestrator",
            credentials_forwarded=False,
        )
    assert any(BLOCKER_NOT_ON_WORKER in e for e in excinfo.value.errors)


def test_forwarding_credentials_for_a_public_artifact_is_a_blocker() -> None:
    """Every candidate is public; a token means something else was fetched."""

    with pytest.raises(CheckpointMaterializationError) as excinfo:
        verify_materialized_checkpoint(
            candidate_id="pi05_droid",
            objects=_objects(_PI05_BYTES),
            materialized_on="gpu_worker",
            credentials_forwarded=True,
        )
    assert BLOCKER_CREDENTIALS_FORWARDED in excinfo.value.errors


def test_a_revision_that_drifted_fails_closed() -> None:
    with pytest.raises(CheckpointMaterializationError) as excinfo:
        verify_materialized_checkpoint(
            candidate_id="pi05_droid",
            objects=_objects(_PI05_BYTES),
            materialized_on="gpu_worker",
            credentials_forwarded=False,
            observed_revision="gcs-generation-inventory:deadbeef",
        )
    assert any("revision_mismatch" in e for e in excinfo.value.errors)


def test_empty_and_malformed_object_rows_never_pass() -> None:
    with pytest.raises(CheckpointMaterializationError):
        verify_materialized_checkpoint(
            candidate_id="pi05_droid",
            objects=[],
            materialized_on="gpu_worker",
            credentials_forwarded=False,
        )
    with pytest.raises(CheckpointMaterializationError):
        verify_materialized_checkpoint(
            candidate_id="pi05_droid",
            objects=[{"name": "a", "size_bytes": "not-a-number"}],
            materialized_on="gpu_worker",
            credentials_forwarded=False,
        )


def test_manifest_digest_is_order_independent() -> None:
    """Listing order must not change the identity of the same bytes."""

    forward = _objects(_PI05_BYTES)
    backward = list(reversed(forward))

    first = verify_materialized_checkpoint(
        candidate_id="pi05_droid",
        objects=forward,
        materialized_on="gpu_worker",
        credentials_forwarded=False,
    )
    second = verify_materialized_checkpoint(
        candidate_id="pi05_droid",
        objects=backward,
        materialized_on="gpu_worker",
        credentials_forwarded=False,
    )
    assert first["object_manifest_sha256"] == second["object_manifest_sha256"]


def test_plan_agrees_with_the_policy_server_standup_contract() -> None:
    """Both must name the same checkpoint identity and the same worker."""

    from blueprint_pipeline.adp009d_policy_server_standup import describe_standup_plan

    for candidate_id in ("pi05_droid", "groot_n17_droid"):
        materialization = plan_checkpoint_materialization(candidate_id)
        standup = describe_standup_plan(candidate_id)
        assert materialization["checkpoint_repository"] == standup["checkpoint_repository"]
        assert materialization["checkpoint_revision"] == standup["checkpoint_revision"]
        assert materialization["expected_total_bytes"] == standup["checkpoint_total_bytes"]
        assert standup["checkpoint_materialized_on"] == materialization["materialize_on"]
