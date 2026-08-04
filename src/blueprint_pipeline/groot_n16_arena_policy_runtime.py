"""Identity contract for Arena's native GR00T-N1.6-DROID remote seam.

The pinned Isaac Lab-Arena revision vendors a GR00T client/server source that
documents N1.6-DROID. This module binds that known-working source and model
checkpoint without loading the model or contacting a provider.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .policy_ranking_thesis import canonical_sha256


MODEL_ID = "nvidia/GR00T-N1.6-DROID"
EMBODIMENT_TAG = "OXE_DROID"
GROOT_SOURCE_REVISION = "e29d8fc50b0e4745120ae3fb72447986fe638aa6"
CHECKPOINT_REVISION = "ae3ebe8d288971ac53aa30c756ea5cba0f52611b"


def _is_sha(value: str, *, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class GrootN16ArenaPolicySpec:
    model_id: str = MODEL_ID
    embodiment_tag: str = EMBODIMENT_TAG
    groot_source_revision: str = GROOT_SOURCE_REVISION
    checkpoint_revision: str = CHECKPOINT_REVISION

    def validate(self) -> None:
        if self.model_id != MODEL_ID:
            raise ValueError("groot_n16_arena_model_id_mismatch")
        if self.embodiment_tag != EMBODIMENT_TAG:
            raise ValueError("groot_n16_arena_embodiment_mismatch")
        if not _is_sha(self.groot_source_revision, length=40):
            raise ValueError("groot_n16_arena_source_revision_invalid")
        if not _is_sha(self.checkpoint_revision, length=40):
            raise ValueError("groot_n16_arena_checkpoint_revision_invalid")

    def identity(self) -> dict[str, Any]:
        self.validate()
        identity: dict[str, Any] = {
            "model_id": self.model_id,
            "embodiment_tag": self.embodiment_tag,
            "groot_source_revision": self.groot_source_revision,
            "checkpoint_revision": self.checkpoint_revision,
            "arena_policy_type": "gr00t_remote_closedloop",
        }
        identity["identity_sha256"] = canonical_sha256(identity)
        return identity


def validate_worker_identity_receipt(
    receipt: Mapping[str, Any], *, expected: GrootN16ArenaPolicySpec
) -> dict[str, Any]:
    expected_identity = expected.identity()
    blockers: list[str] = []
    if receipt.get("status") != "verified":
        blockers.append("groot_n16_arena_worker_receipt_not_verified")
    for key in (
        "model_id",
        "embodiment_tag",
        "groot_source_revision",
        "checkpoint_revision",
    ):
        if receipt.get(key) != expected_identity[key]:
            blockers.append(f"groot_n16_arena_worker_receipt_{key}_mismatch")
    checkpoint_digest = str(receipt.get("checkpoint_files_sha256") or "")
    environment_digest = str(receipt.get("environment_lock_sha256") or "")
    if not _is_sha(checkpoint_digest, length=64):
        blockers.append("groot_n16_arena_checkpoint_files_sha256_invalid")
    if not _is_sha(environment_digest, length=64):
        blockers.append("groot_n16_arena_environment_lock_sha256_invalid")
    if blockers:
        raise ValueError(";".join(sorted(blockers)))
    return dict(receipt)


__all__ = [
    "CHECKPOINT_REVISION",
    "EMBODIMENT_TAG",
    "GROOT_SOURCE_REVISION",
    "MODEL_ID",
    "GrootN16ArenaPolicySpec",
    "validate_worker_identity_receipt",
]
