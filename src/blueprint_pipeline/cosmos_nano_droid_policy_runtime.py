"""Exact public Cosmos3 Nano DROID contract; native inference is not yet measured."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .cosmos_edge_droid_policy_runtime import (
    CosmosEdgeDroidPolicySpec, canonical_sha256, validate_server_metadata,
    verify_local_policy_snapshot as _verify_snapshot,
)

MODEL_ID = "nvidia/Cosmos3-Nano-Policy-DROID"
MODEL_REVISION = "805c0d6d46196ecdc789213a6de72303c383895b"
MODEL_CONFIG_SHA256 = "11577909f49e95d62456b6a497cf7b883f8987a3186d720d838ea5de0e493366"
CHECKPOINT_CONFIG_SHA256 = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
SNAPSHOT_MANIFEST_SHA256 = "1f87ceffb8f1db9e0a3b6cf76e23d64e2d01047c8e39ffe92691f386aa2a3e88"
SNAPSHOT_MANIFEST_SCHEMA = "cosmos_nano_droid_policy_snapshot_manifest.v1"


@dataclass(frozen=True)
class CosmosNanoDroidPolicySpec(CosmosEdgeDroidPolicySpec):
    policy_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    model_config_sha256: str = MODEL_CONFIG_SHA256
    checkpoint_config_sha256: str = CHECKPOINT_CONFIG_SHA256
    snapshot_manifest_sha256: str = SNAPSHOT_MANIFEST_SHA256
    action_chunk_rows: int = 32
    format_prompt_as_json: bool = False

    def validate(self) -> None:
        actual, expected = asdict(self), asdict(CosmosNanoDroidPolicySpec())
        actual["required_policy_views"] = tuple(actual["required_policy_views"])
        if (actual != expected or any(type(actual[key]) is not type(value) for key, value in expected.items())
                or self.format_prompt_as_json is not False):
            raise ValueError("cosmos_nano_policy_exact_contract_mismatch")

    def server_metadata(self) -> dict:
        self.validate()
        payload = {"schema_version": "cosmos_nano_droid_policy_server_metadata.v1",
                   **asdict(self), "required_policy_views": list(self.required_policy_views),
                   "role": "candidate_policy_only", "neutral_wam": False, "self_evaluator": False}
        payload["identity_sha256"] = canonical_sha256(payload)
        return payload


def verify_local_policy_snapshot(*, spec, snapshot_dir, snapshot_manifest_path):
    return _verify_snapshot(spec=spec, snapshot_dir=snapshot_dir,
                            snapshot_manifest_path=snapshot_manifest_path,
                            expected_schema=SNAPSHOT_MANIFEST_SCHEMA, require_exact_files=True)


__all__ = ["CosmosNanoDroidPolicySpec", "validate_server_metadata", "verify_local_policy_snapshot"]
