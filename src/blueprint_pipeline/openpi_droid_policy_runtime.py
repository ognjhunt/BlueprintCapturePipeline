"""Identity-bound OpenPI DROID policy server/client contracts.

The public OpenPI websocket protocol is intentionally small, but its default
DROID policy metadata is empty.  This module adds the minimum fail-closed
identity layer required for a policy-ranking experiment: policy/config identity,
action semantics, checkpoint object-manifest identity, and pinned source revision.
It does not download checkpoints or allocate paid compute.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .droid_policy_bridge import DROID_OPEN_LOOP_HORIZON, OPENPI_SOURCE_REVISION
from .policy_ranking_thesis import canonical_sha256


SCHEMA_VERSION = "openpi_droid_policy_runtime.v1"
SERVER_METADATA_SCHEMA_VERSION = "openpi_droid_policy_server_metadata.v1"
SUPPORTED_ACTION_SPACES = frozenset({"joint_position"})
SUPPORTED_ACTION_CHUNK_ROWS = frozenset({10, 15})
LOCAL_VERIFICATION_FIELDS = frozenset(
    {
        "local_checkpoint_verified",
        "local_checkpoint_verification_sha256",
        "local_checkpoint_object_count",
        "local_checkpoint_size_bytes",
    }
)


@dataclass(frozen=True)
class OpenPIDroidPolicySpec:
    policy_id: str
    config_name: str
    checkpoint_uri: str
    checkpoint_object_manifest_sha256: str
    checkpoint_generation_manifest_sha256: str
    checkpoint_inventory_sha256: str
    checkpoint_object_count: int
    checkpoint_size_bytes: int
    action_space: str
    action_chunk_rows: int
    open_loop_horizon: int = DROID_OPEN_LOOP_HORIZON
    openpi_revision: str = OPENPI_SOURCE_REVISION

    def validate(self) -> None:
        if not self.policy_id or self.config_name != self.policy_id:
            raise ValueError("policy_and_config_identity_mismatch")
        if not self.checkpoint_uri.startswith("gs://openpi-assets/checkpoints/polaris/"):
            raise ValueError("checkpoint_uri_not_frozen_openpi_polaris")
        digest = self.checkpoint_object_manifest_sha256
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("invalid_checkpoint_object_manifest_sha256")
        for name, generation_digest in (
            (
                "checkpoint_generation_manifest_sha256",
                self.checkpoint_generation_manifest_sha256,
            ),
            ("checkpoint_inventory_sha256", self.checkpoint_inventory_sha256),
        ):
            if len(generation_digest) != 64 or any(
                character not in "0123456789abcdef" for character in generation_digest
            ):
                raise ValueError(f"invalid_{name}")
        if self.checkpoint_object_count <= 0 or self.checkpoint_size_bytes <= 0:
            raise ValueError("invalid_checkpoint_inventory_summary")
        if self.action_space not in SUPPORTED_ACTION_SPACES:
            raise ValueError("unsupported_action_space")
        if self.action_chunk_rows not in SUPPORTED_ACTION_CHUNK_ROWS:
            raise ValueError("unsupported_action_chunk_rows")
        if not 1 <= self.open_loop_horizon <= self.action_chunk_rows:
            raise ValueError("invalid_open_loop_horizon")
        if self.openpi_revision != OPENPI_SOURCE_REVISION:
            raise ValueError("openpi_revision_mismatch")

    def server_metadata(self) -> dict[str, Any]:
        self.validate()
        identity = {
            "schema_version": SERVER_METADATA_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "config_name": self.config_name,
            "checkpoint_uri": self.checkpoint_uri,
            "checkpoint_object_manifest_sha256": self.checkpoint_object_manifest_sha256,
            "checkpoint_generation_manifest_sha256": self.checkpoint_generation_manifest_sha256,
            "checkpoint_inventory_sha256": self.checkpoint_inventory_sha256,
            "checkpoint_object_count": self.checkpoint_object_count,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "action_space": self.action_space,
            "action_chunk_rows": self.action_chunk_rows,
            "open_loop_horizon": self.open_loop_horizon,
            "openpi_revision": self.openpi_revision,
        }
        identity["identity_sha256"] = canonical_sha256(identity)
        return identity


def load_policy_spec(
    cohort_path: str | Path,
    *,
    policy_id: str,
) -> OpenPIDroidPolicySpec:
    payload = json.loads(Path(cohort_path).expanduser().read_text(encoding="utf-8"))
    if payload.get("schema_version") != "policy_ranking_warehouse_policy_cohort.v2":
        raise ValueError("unsupported_policy_cohort_schema")
    action_contract = payload.get("action_contract")
    if not isinstance(action_contract, Mapping):
        raise ValueError("missing_action_contract")
    matches = [
        row
        for row in payload.get("primary_cohort", [])
        if isinstance(row, Mapping) and row.get("policy_id") == policy_id
    ]
    if len(matches) != 1:
        raise ValueError("policy_id_not_unique_in_cohort")
    row = matches[0]
    inventory = payload.get("checkpoint_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("missing_checkpoint_inventory")
    spec = OpenPIDroidPolicySpec(
        policy_id=str(row["policy_id"]),
        config_name=str(row["policy_id"]),
        checkpoint_uri=str(row["checkpoint"]),
        checkpoint_object_manifest_sha256=str(row["public_object_manifest_sha256"]),
        checkpoint_generation_manifest_sha256=str(row["generation_manifest_sha256"]),
        checkpoint_inventory_sha256=str(inventory["inventory_sha256"]),
        checkpoint_object_count=int(row["checkpoint_object_count"]),
        checkpoint_size_bytes=int(row["checkpoint_size_bytes"]),
        action_space=str(action_contract["space"]).replace(
            "absolute_joint_position_plus_gripper_position", "joint_position"
        ),
        action_chunk_rows=int(row["action_horizon"]),
        open_loop_horizon=int(action_contract["executed_open_loop_horizon"]),
        openpi_revision=str(payload["openpi_revision"]),
    )
    spec.validate()
    return spec


def validate_server_metadata(
    metadata: Mapping[str, Any],
    *,
    expected: OpenPIDroidPolicySpec,
) -> dict[str, Any]:
    expected_metadata = expected.server_metadata()
    actual = {str(key): value for key, value in metadata.items()}
    mismatches = [
        key
        for key, expected_value in expected_metadata.items()
        if actual.get(key) != expected_value
    ]
    unexpected = sorted(set(actual) - set(expected_metadata) - LOCAL_VERIFICATION_FIELDS)
    if mismatches or unexpected:
        reasons = [*sorted(mismatches), *(f"unexpected:{key}" for key in unexpected)]
        raise ValueError(f"policy_server_identity_mismatch:{','.join(reasons)}")
    if actual.get("local_checkpoint_verified") is not True:
        raise ValueError("policy_server_local_checkpoint_not_verified")
    local_digest = actual.get("local_checkpoint_verification_sha256")
    if not isinstance(local_digest, str) or len(local_digest) != 64 or any(
        character not in "0123456789abcdef" for character in local_digest
    ):
        raise ValueError("policy_server_local_checkpoint_verification_invalid")
    if actual.get("local_checkpoint_object_count") != expected.checkpoint_object_count:
        raise ValueError("policy_server_local_checkpoint_object_count_mismatch")
    if actual.get("local_checkpoint_size_bytes") != expected.checkpoint_size_bytes:
        raise ValueError("policy_server_local_checkpoint_size_mismatch")
    return actual


def _file_md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def verify_local_checkpoint(
    *,
    spec: OpenPIDroidPolicySpec,
    checkpoint_dir: str | Path,
    checkpoint_inventory_path: str | Path,
) -> dict[str, Any]:
    """Verify every downloaded checkpoint object against the frozen GCS inventory."""
    spec.validate()
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    inventory_file = Path(checkpoint_inventory_path).expanduser().resolve()
    inventory = json.loads(inventory_file.read_text(encoding="utf-8"))
    if inventory.get("schema_version") != "openpi_checkpoint_inventory.v1":
        raise ValueError("unsupported_checkpoint_inventory_schema")
    if inventory.get("status") != "frozen" or inventory.get("blockers"):
        raise ValueError("checkpoint_inventory_not_frozen")
    if inventory.get("openpi_revision") != spec.openpi_revision:
        raise ValueError("checkpoint_inventory_openpi_revision_mismatch")
    declared_inventory_digest = inventory.get("inventory_sha256")
    digest_payload = dict(inventory)
    digest_payload.pop("inventory_sha256", None)
    actual_inventory_digest = canonical_sha256(digest_payload)
    if (
        declared_inventory_digest != actual_inventory_digest
        or actual_inventory_digest != spec.checkpoint_inventory_sha256
    ):
        raise ValueError("checkpoint_inventory_sha256_mismatch")
    matches = [
        row
        for row in inventory.get("entries", [])
        if isinstance(row, Mapping) and row.get("policy_id") == spec.policy_id
    ]
    if len(matches) != 1:
        raise ValueError("checkpoint_inventory_policy_not_unique")
    entry = matches[0]
    entry_checks = {
        "checkpoint_uri": spec.checkpoint_uri,
        "object_count": spec.checkpoint_object_count,
        "size_bytes": spec.checkpoint_size_bytes,
        "legacy_object_manifest_sha256": spec.checkpoint_object_manifest_sha256,
        "generation_manifest_sha256": spec.checkpoint_generation_manifest_sha256,
    }
    changed = sorted(key for key, value in entry_checks.items() if entry.get(key) != value)
    if changed:
        raise ValueError(f"checkpoint_inventory_entry_mismatch:{','.join(changed)}")
    objects = entry.get("objects")
    if not isinstance(objects, list) or len(objects) != spec.checkpoint_object_count:
        raise ValueError("checkpoint_inventory_objects_invalid")
    object_prefix = spec.checkpoint_uri.removeprefix("gs://openpi-assets/").rstrip("/") + "/"
    verified_objects: list[dict[str, Any]] = []
    expected_relative_paths: set[str] = set()
    for row in objects:
        if not isinstance(row, Mapping):
            raise ValueError("checkpoint_inventory_object_not_mapping")
        object_name = str(row.get("name", ""))
        if not object_name.startswith(object_prefix):
            raise ValueError("checkpoint_inventory_object_outside_prefix")
        relative_text = object_name.removeprefix(object_prefix)
        relative_path = PurePosixPath(relative_text)
        if not relative_text or relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("checkpoint_inventory_object_path_invalid")
        local_path = checkpoint.joinpath(*relative_path.parts)
        if not local_path.is_file() or local_path.is_symlink():
            raise FileNotFoundError(f"checkpoint_object_missing_or_not_regular:{relative_text}")
        expected_size = int(row.get("size", -1))
        actual_size = local_path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"checkpoint_object_size_mismatch:{relative_text}")
        expected_md5 = str(row.get("md5Hash", ""))
        actual_md5 = _file_md5_base64(local_path)
        if not expected_md5 or actual_md5 != expected_md5:
            raise ValueError(f"checkpoint_object_md5_mismatch:{relative_text}")
        expected_relative_paths.add(relative_text)
        verified_objects.append(
            {"relative_path": relative_text, "size": actual_size, "md5Hash": actual_md5}
        )
    local_files = {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    unexpected_files = sorted(local_files - expected_relative_paths)
    if unexpected_files:
        raise ValueError(f"checkpoint_unexpected_local_files:{','.join(unexpected_files)}")
    verified_objects.sort(key=lambda row: str(row["relative_path"]))
    verification_payload = {
        "schema_version": "openpi_local_checkpoint_verification.v1",
        "policy_id": spec.policy_id,
        "checkpoint_uri": spec.checkpoint_uri,
        "checkpoint_inventory_sha256": actual_inventory_digest,
        "checkpoint_generation_manifest_sha256": spec.checkpoint_generation_manifest_sha256,
        "objects": verified_objects,
    }
    return {
        "local_checkpoint_verified": True,
        "local_checkpoint_verification_sha256": canonical_sha256(verification_payload),
        "local_checkpoint_object_count": len(verified_objects),
        "local_checkpoint_size_bytes": sum(int(row["size"]) for row in verified_objects),
    }


class OpenPIWebsocketDroidPolicyClient:
    """OpenPI websocket client with mandatory server-identity verification."""

    learned_policy = True

    def __init__(
        self,
        *,
        spec: OpenPIDroidPolicySpec,
        host: str,
        port: int,
        api_key: str | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        spec.validate()
        if not host.strip() or not 1 <= int(port) <= 65535:
            raise ValueError("invalid_policy_server_endpoint")
        if client_factory is None:
            try:
                from openpi_client import websocket_client_policy
            except ImportError as exc:  # pragma: no cover - exercised on GPU runtime
                raise RuntimeError("openpi_client_not_installed") from exc
            client_factory = websocket_client_policy.WebsocketClientPolicy
        self.policy_id = spec.policy_id
        self.action_space = spec.action_space
        self.action_chunk_rows = spec.action_chunk_rows
        self.open_loop_horizon = spec.open_loop_horizon
        self._spec = spec
        self._client = client_factory(host=host, port=int(port), api_key=api_key)
        raw_metadata = self._client.get_server_metadata()
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("policy_server_metadata_not_object")
        self.server_metadata = validate_server_metadata(raw_metadata, expected=spec)

    def infer(self, observation: Mapping[str, Any]) -> Any:
        return self._client.infer(dict(observation))

    def evidence_summary(self) -> dict[str, Any]:
        return {
            "transport": "openpi_websocket_msgpack_numpy",
            "identity_verified": True,
            "server_metadata": self.server_metadata,
        }


def serve_identity_bound_policy(
    *,
    spec: OpenPIDroidPolicySpec,
    checkpoint_dir: str | Path,
    checkpoint_inventory_path: str | Path,
    host: str,
    port: int,
) -> None:
    """Load the pinned OpenPI config/checkpoint and serve identity metadata."""
    spec.validate()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("openpi_policy_server_must_be_loopback_only")
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    local_verification = verify_local_checkpoint(
        spec=spec,
        checkpoint_dir=checkpoint,
        checkpoint_inventory_path=checkpoint_inventory_path,
    )
    try:
        from openpi.policies import policy_config
        from openpi.serving import websocket_policy_server
        from openpi.training import config as training_config
    except ImportError as exc:  # pragma: no cover - exercised on GPU runtime
        raise RuntimeError("openpi_server_runtime_not_installed") from exc
    config = training_config.get_config(spec.config_name)
    configured_rows = int(config.model.action_horizon)
    if configured_rows != spec.action_chunk_rows:
        raise ValueError("openpi_config_action_horizon_mismatch")
    policy = policy_config.create_trained_policy(config, checkpoint)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=host,
        port=int(port),
        metadata={**spec.server_metadata(), **local_verification},
    )
    server.serve_forever()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-inventory", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    spec = load_policy_spec(args.cohort, policy_id=args.policy_id)
    serve_identity_bound_policy(
        spec=spec,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_inventory_path=args.checkpoint_inventory,
        host=args.host,
        port=args.port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OpenPIDroidPolicySpec",
    "OpenPIWebsocketDroidPolicyClient",
    "load_policy_spec",
    "serve_identity_bound_policy",
    "validate_server_metadata",
    "verify_local_checkpoint",
]
