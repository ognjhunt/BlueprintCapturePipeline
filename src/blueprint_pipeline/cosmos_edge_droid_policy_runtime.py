"""Identity-bound Cosmos3-Edge DROID policy client and snapshot contract.

The NVIDIA RoboLab policy server intentionally speaks OpenPI's websocket
protocol, but its stock metadata is empty.  Blueprint wraps that transport with
an immutable model/source identity and validates the exact three-view DROID
observation plus the returned 16x8 joint-position action chunk before a WAM can
consume it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .droid_policy_bridge import (
    DROID_ROBOARENA_CONCAT_VIEWS,
    validate_droid_action_chunk,
    validate_droid_observation,
)
from .policy_ranking_thesis import canonical_sha256, file_sha256


MODEL_ID = "nvidia/Cosmos3-Edge-Policy-DROID"
MODEL_REVISION = "3ea407af3e156c0af3b4bb6edd85842cc9a58777"
COSMOS_FRAMEWORK_REVISION = "2f603cb114ff8b335e116060444d0b6caee3a85e"
MODEL_CONFIG_SHA256 = "da6a23cbf4477aafda3e773874bf2c98d6869156e8a450d944e2f28c94eee00b"
CHECKPOINT_CONFIG_SHA256 = "a279d57b84d458c2aeeaf9698aee8c1b3830204422da67ab31c801ab57225019"
ACTION_CHUNK_ROWS = 16
ACTION_DIMENSION = 8
EXECUTED_PREFIX_STEPS = 8
CONDITIONING_FPS = 15.0
SERVER_METADATA_SCHEMA = "cosmos_edge_droid_policy_server_metadata.v1"
SNAPSHOT_MANIFEST_SCHEMA = "cosmos_edge_droid_policy_snapshot_manifest.v1"


def _sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


@dataclass(frozen=True)
class CosmosEdgeDroidPolicySpec:
    policy_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    cosmos_framework_revision: str = COSMOS_FRAMEWORK_REVISION
    model_config_sha256: str = MODEL_CONFIG_SHA256
    checkpoint_config_sha256: str = CHECKPOINT_CONFIG_SHA256
    snapshot_manifest_sha256: str = ""
    action_space: str = "absolute_joint_position_plus_gripper"
    action_chunk_rows: int = ACTION_CHUNK_ROWS
    action_dimension: int = ACTION_DIMENSION
    executed_prefix_steps: int = EXECUTED_PREFIX_STEPS
    conditioning_fps: float = CONDITIONING_FPS
    required_policy_views: tuple[str, ...] = DROID_ROBOARENA_CONCAT_VIEWS
    use_state: bool = True
    license_id: str = "OpenMDW-1.1"

    def validate(self) -> None:
        if self.policy_id != MODEL_ID or self.model_revision != MODEL_REVISION:
            raise ValueError("cosmos_edge_policy_model_identity_mismatch")
        if self.cosmos_framework_revision != COSMOS_FRAMEWORK_REVISION:
            raise ValueError("cosmos_edge_policy_source_revision_mismatch")
        if self.model_config_sha256 != MODEL_CONFIG_SHA256:
            raise ValueError("cosmos_edge_policy_model_config_mismatch")
        if self.checkpoint_config_sha256 != CHECKPOINT_CONFIG_SHA256:
            raise ValueError("cosmos_edge_policy_checkpoint_config_mismatch")
        if not _sha256(self.snapshot_manifest_sha256):
            raise ValueError("cosmos_edge_policy_snapshot_manifest_missing")
        if self.action_space != "absolute_joint_position_plus_gripper":
            raise ValueError("cosmos_edge_policy_action_space_mismatch")
        if (self.action_chunk_rows, self.action_dimension) != (
            ACTION_CHUNK_ROWS,
            ACTION_DIMENSION,
        ):
            raise ValueError("cosmos_edge_policy_action_shape_mismatch")
        if not 1 <= self.executed_prefix_steps <= self.action_chunk_rows:
            raise ValueError("cosmos_edge_policy_executed_prefix_invalid")
        if self.conditioning_fps != CONDITIONING_FPS:
            raise ValueError("cosmos_edge_policy_conditioning_fps_mismatch")
        if tuple(self.required_policy_views) != DROID_ROBOARENA_CONCAT_VIEWS:
            raise ValueError("cosmos_edge_policy_view_contract_mismatch")
        if self.use_state is not True or self.license_id != "OpenMDW-1.1":
            raise ValueError("cosmos_edge_policy_state_or_license_mismatch")

    def server_metadata(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "schema_version": SERVER_METADATA_SCHEMA,
            **asdict(self),
            "required_policy_views": list(self.required_policy_views),
            "role": "candidate_policy_only",
            "neutral_wam": False,
            "self_evaluator": False,
        }
        payload["identity_sha256"] = canonical_sha256(payload)
        return payload


def verify_local_policy_snapshot(
    *,
    spec: CosmosEdgeDroidPolicySpec,
    snapshot_dir: str | Path,
    snapshot_manifest_path: str | Path,
) -> dict[str, Any]:
    """Hash every frozen runtime file before the policy model is loaded."""

    spec.validate()
    snapshot = Path(snapshot_dir).expanduser().resolve()
    manifest_path = Path(snapshot_manifest_path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_digest = str(payload.get("manifest_sha256") or "")
    digest_payload = dict(payload)
    digest_payload.pop("manifest_sha256", None)
    actual_digest = canonical_sha256(digest_payload)
    if recorded_digest != actual_digest or actual_digest != spec.snapshot_manifest_sha256:
        raise ValueError("cosmos_edge_policy_snapshot_manifest_digest_mismatch")
    if (
        payload.get("schema_version") != SNAPSHOT_MANIFEST_SCHEMA
        or payload.get("model_id") != spec.policy_id
        or payload.get("model_revision") != spec.model_revision
    ):
        raise ValueError("cosmos_edge_policy_snapshot_manifest_identity_mismatch")
    files = payload.get("required_files")
    if not isinstance(files, list) or not files:
        raise ValueError("cosmos_edge_policy_snapshot_required_files_missing")
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in files:
        if not isinstance(row, Mapping):
            raise ValueError("cosmos_edge_policy_snapshot_file_entry_invalid")
        relative_text = str(row.get("path") or "")
        relative = PurePosixPath(relative_text)
        if (
            not relative_text
            or relative.is_absolute()
            or ".." in relative.parts
            or relative_text in seen
        ):
            raise ValueError("cosmos_edge_policy_snapshot_file_path_invalid")
        seen.add(relative_text)
        path = snapshot.joinpath(*relative.parts)
        if not path.is_file():
            raise FileNotFoundError(f"cosmos_edge_policy_snapshot_file_missing:{relative_text}")
        expected_size = int(row.get("size_bytes") or -1)
        expected_sha256 = str(row.get("sha256") or "")
        if path.stat().st_size != expected_size:
            raise ValueError(f"cosmos_edge_policy_snapshot_file_size_mismatch:{relative_text}")
        if not _sha256(expected_sha256) or file_sha256(path) != expected_sha256:
            raise ValueError(f"cosmos_edge_policy_snapshot_file_hash_mismatch:{relative_text}")
        verified.append(
            {
                "path": relative_text,
                "size_bytes": expected_size,
                "sha256": expected_sha256,
            }
        )
    receipt = {
        "schema_version": "cosmos_edge_droid_policy_snapshot_verification.v1",
        "model_id": spec.policy_id,
        "model_revision": spec.model_revision,
        "snapshot_manifest_sha256": actual_digest,
        "verified_files": verified,
    }
    return {
        "local_snapshot_verified": True,
        "local_snapshot_manifest_sha256": actual_digest,
        "local_snapshot_file_count": len(verified),
        "local_snapshot_size_bytes": sum(row["size_bytes"] for row in verified),
        "local_snapshot_verification_sha256": canonical_sha256(receipt),
    }


def validate_server_metadata(
    metadata: Mapping[str, Any], *, expected: CosmosEdgeDroidPolicySpec
) -> dict[str, Any]:
    expected_metadata = expected.server_metadata()
    actual = dict(metadata)
    mismatches = sorted(
        key for key, value in expected_metadata.items() if actual.get(key) != value
    )
    local_fields = {
        "local_snapshot_verified",
        "local_snapshot_manifest_sha256",
        "local_snapshot_file_count",
        "local_snapshot_size_bytes",
        "local_snapshot_verification_sha256",
    }
    unexpected = sorted(set(actual) - set(expected_metadata) - local_fields)
    if mismatches or unexpected:
        reasons = [*mismatches, *(f"unexpected:{key}" for key in unexpected)]
        raise ValueError(f"cosmos_edge_policy_server_identity_mismatch:{','.join(reasons)}")
    if actual.get("local_snapshot_verified") is not True:
        raise ValueError("cosmos_edge_policy_server_snapshot_not_verified")
    if actual.get("local_snapshot_manifest_sha256") != expected.snapshot_manifest_sha256:
        raise ValueError("cosmos_edge_policy_server_snapshot_manifest_mismatch")
    for field in ("local_snapshot_file_count", "local_snapshot_size_bytes"):
        if not isinstance(actual.get(field), int) or int(actual[field]) <= 0:
            raise ValueError(f"cosmos_edge_policy_server_{field}_invalid")
    if not _sha256(actual.get("local_snapshot_verification_sha256")):
        raise ValueError("cosmos_edge_policy_server_snapshot_verification_invalid")
    return actual


def _array_sha256(array: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(list(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


class CosmosEdgeDroidPolicyClient:
    """Verified OpenPI-protocol client for the frozen NVIDIA Edge policy."""

    learned_policy = True

    def __init__(
        self,
        *,
        spec: CosmosEdgeDroidPolicySpec,
        host: str,
        port: int,
        api_key: str | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        spec.validate()
        if not host.strip() or not 1 <= int(port) <= 65535:
            raise ValueError("cosmos_edge_policy_server_endpoint_invalid")
        if client_factory is None:
            try:
                from openpi_client import websocket_client_policy
            except ImportError as exc:  # pragma: no cover - GPU runtime dependency
                raise RuntimeError("openpi_client_not_installed") from exc
            client_factory = websocket_client_policy.WebsocketClientPolicy
        self.policy_id = spec.policy_id
        self.action_space = spec.action_space
        self.action_chunk_rows = spec.action_chunk_rows
        self.open_loop_horizon = spec.executed_prefix_steps
        self.required_policy_views = spec.required_policy_views
        self._spec = spec
        self._client = client_factory(host=host, port=int(port), api_key=api_key)
        raw_metadata = self._client.get_server_metadata()
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("cosmos_edge_policy_server_metadata_not_object")
        self.server_metadata = validate_server_metadata(raw_metadata, expected=spec)
        self._receipts: list[dict[str, Any]] = []

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        import numpy as np

        blockers = validate_droid_observation(
            observation, required_views=self.required_policy_views
        )
        if blockers:
            raise ValueError(f"cosmos_edge_policy_observation_invalid:{blockers[0]}")
        raw_response = self._client.infer(dict(observation))
        if not isinstance(raw_response, Mapping):
            raise ValueError("cosmos_edge_policy_response_not_object")
        action = raw_response.get("action", raw_response.get("actions"))
        action_blockers = validate_droid_action_chunk(
            action, expected_rows=self.action_chunk_rows
        )
        if action_blockers:
            raise ValueError(f"cosmos_edge_policy_action_invalid:{action_blockers[0]}")
        action_array = np.asarray(action, dtype=np.float64)
        observation_identity = {
            "prompt": str(observation["prompt"]),
            "views": {
                view: _array_sha256(observation[view]) for view in self.required_policy_views
            },
            "joint_position_sha256": _array_sha256(
                observation["observation/joint_position"]
            ),
            "gripper_position_sha256": _array_sha256(
                observation["observation/gripper_position"]
            ),
        }
        receipt = {
            "query_index": len(self._receipts),
            "observation_sha256": canonical_sha256(observation_identity),
            "action_sha256": _array_sha256(action_array),
            "action_shape": list(action_array.shape),
            "server_identity_sha256": self.server_metadata["identity_sha256"],
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        self._receipts.append(receipt)
        return {"action": action_array, "policy_request_receipt": receipt}

    def evidence_summary(self) -> dict[str, Any]:
        return {
            "transport": "openpi_websocket_msgpack_numpy",
            "identity_verified": True,
            "server_metadata": self.server_metadata,
            "request_count": len(self._receipts),
            "request_receipts": list(self._receipts),
        }


__all__ = [
    "CosmosEdgeDroidPolicyClient",
    "CosmosEdgeDroidPolicySpec",
    "validate_server_metadata",
    "verify_local_policy_snapshot",
]
