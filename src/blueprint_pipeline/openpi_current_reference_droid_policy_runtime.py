"""Identity-bound runtime for current official OpenPI DROID checkpoints.

This module is deliberately separate from the historical Polaris joint-position
runtime.  It freezes the current public OpenPI joint-velocity checkpoints and
validates every downloaded GCS object before a policy can be queried.  It does
not allocate compute, execute a WAM, or assign task success.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from .droid_policy_bridge import (
    DROID_OPENPI_POLICY_VIEWS,
    OPENPI_SOURCE_REVISION,
    validate_droid_action_chunk,
    validate_droid_observation,
)
from .policy_ranking_thesis import canonical_sha256, file_sha256


SCHEMA_VERSION = "openpi_current_reference_droid_policy_runtime.v1"
SERVER_METADATA_SCHEMA_VERSION = (
    "openpi_current_reference_droid_policy_server_metadata.v1"
)
CURRENT_REFERENCE_ACTION_ROWS = {
    "pi0_droid": 10,
    "pi0_fast_droid": 10,
    "pi05_droid": 15,
}
CURRENT_REFERENCE_INVENTORY_FILES = {
    "pi0_droid": "openpi_pi0_droid_gcs_inventory_v1.json",
    "pi0_fast_droid": "openpi_pi0_fast_droid_gcs_inventory_v1.json",
    "pi05_droid": "openpi_pi05_droid_gcs_inventory_v1.json",
}


def _sha256_text(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


@dataclass(frozen=True)
class OpenPICurrentReferenceDroidPolicySpec:
    """Frozen identity for one current official OpenPI DROID policy."""

    policy_id: str
    config_name: str
    checkpoint_uri: str
    checkpoint_object_inventory_sha256: str
    checkpoint_manifest_sha256: str
    checkpoint_inventory_file_sha256: str
    checkpoint_object_count: int
    checkpoint_size_bytes: int
    action_chunk_rows: int
    openpi_revision: str = OPENPI_SOURCE_REVISION
    action_space: str = "joint_velocity_plus_gripper_position"
    executed_prefix_steps: int = 8
    control_hz: float = 15.0

    def validate(self) -> None:
        expected_rows = CURRENT_REFERENCE_ACTION_ROWS.get(self.policy_id)
        if self.config_name != self.policy_id or expected_rows is None:
            raise ValueError("openpi_current_reference_policy_identity_invalid")
        if self.action_chunk_rows != expected_rows:
            raise ValueError("openpi_current_reference_action_rows_invalid")
        expected_uri = f"gs://openpi-assets/checkpoints/{self.policy_id}"
        if self.checkpoint_uri != expected_uri:
            raise ValueError("openpi_current_reference_checkpoint_uri_invalid")
        for field_name, value in (
            ("checkpoint_object_inventory_sha256", self.checkpoint_object_inventory_sha256),
            ("checkpoint_manifest_sha256", self.checkpoint_manifest_sha256),
            ("checkpoint_inventory_file_sha256", self.checkpoint_inventory_file_sha256),
        ):
            if not _sha256_text(value):
                raise ValueError(f"openpi_current_reference_{field_name}_invalid")
        if self.checkpoint_object_count <= 0 or self.checkpoint_size_bytes <= 0:
            raise ValueError("openpi_current_reference_checkpoint_summary_invalid")
        if self.openpi_revision != OPENPI_SOURCE_REVISION:
            raise ValueError("openpi_current_reference_source_revision_invalid")
        if self.action_space != "joint_velocity_plus_gripper_position":
            raise ValueError("openpi_current_reference_action_space_invalid")
        if self.executed_prefix_steps != 8 or self.control_hz != 15.0:
            raise ValueError("openpi_current_reference_execution_contract_invalid")

    def server_metadata(self) -> dict[str, Any]:
        self.validate()
        payload = {
            "schema_version": SERVER_METADATA_SCHEMA_VERSION,
            **asdict(self),
            "required_policy_views": list(DROID_OPENPI_POLICY_VIEWS),
            "role": "candidate_policy_only",
            "neutral_wam": False,
            "self_evaluator": False,
        }
        payload["identity_sha256"] = canonical_sha256(payload)
        return payload


def load_current_reference_policy_specs(
    *, source_freeze_path: str | Path, checkpoint_inventory_dir: str | Path
) -> dict[str, OpenPICurrentReferenceDroidPolicySpec]:
    """Load all three identities from the prospective source freeze."""

    freeze_path = Path(source_freeze_path).expanduser().resolve()
    inventory_dir = Path(checkpoint_inventory_dir).expanduser().resolve()
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "policy_ranking_real_policy_closed_loop_source_freeze.v1":
        raise ValueError("openpi_current_reference_source_freeze_schema_invalid")
    openpi = payload.get("openpi")
    if not isinstance(openpi, Mapping) or openpi.get("revision") != OPENPI_SOURCE_REVISION:
        raise ValueError("openpi_current_reference_source_freeze_revision_invalid")
    specs: dict[str, OpenPICurrentReferenceDroidPolicySpec] = {}
    for row in payload.get("policies", []):
        if not isinstance(row, Mapping):
            raise ValueError("openpi_current_reference_policy_row_invalid")
        policy_id = str(row.get("policy_id") or "")
        inventory_name = CURRENT_REFERENCE_INVENTORY_FILES.get(policy_id)
        if inventory_name is None or policy_id in specs:
            raise ValueError("openpi_current_reference_policy_set_invalid")
        inventory_path = inventory_dir / inventory_name
        inventory_file_sha256 = file_sha256(inventory_path)
        if inventory_file_sha256 != row.get("external_inventory_file_sha256"):
            raise ValueError("openpi_current_reference_inventory_file_sha256_mismatch")
        spec = OpenPICurrentReferenceDroidPolicySpec(
            policy_id=policy_id,
            config_name=str(row.get("config_name") or ""),
            checkpoint_uri=str(row.get("checkpoint_uri") or ""),
            checkpoint_object_inventory_sha256=str(
                row.get("object_inventory_sha256") or ""
            ),
            checkpoint_manifest_sha256=str(row.get("manifest_sha256") or ""),
            checkpoint_inventory_file_sha256=inventory_file_sha256,
            checkpoint_object_count=int(row.get("object_count") or 0),
            checkpoint_size_bytes=int(row.get("total_bytes") or 0),
            action_chunk_rows=int((row.get("native_action_shape") or [0])[0]),
        )
        spec.validate()
        specs[policy_id] = spec
    if set(specs) != set(CURRENT_REFERENCE_ACTION_ROWS):
        raise ValueError("openpi_current_reference_policy_set_incomplete")
    return specs


def verify_local_current_reference_checkpoint(
    *,
    spec: OpenPICurrentReferenceDroidPolicySpec,
    checkpoint_dir: str | Path,
    checkpoint_inventory_path: str | Path,
) -> dict[str, Any]:
    """Verify every current-reference checkpoint object before model loading."""

    spec.validate()
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    inventory_path = Path(checkpoint_inventory_path).expanduser().resolve()
    if file_sha256(inventory_path) != spec.checkpoint_inventory_file_sha256:
        raise ValueError("openpi_current_reference_inventory_file_changed")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    recorded_manifest = str(inventory.get("manifest_sha256") or "")
    digest_payload = dict(inventory)
    digest_payload.pop("manifest_sha256", None)
    if (
        inventory.get("schema_version") != "public_gcs_checkpoint_inventory.v1"
        or recorded_manifest != canonical_sha256(digest_payload)
        or recorded_manifest != spec.checkpoint_manifest_sha256
    ):
        raise ValueError("openpi_current_reference_inventory_manifest_invalid")
    summary = {
        "source_uri": spec.checkpoint_uri,
        "object_count": spec.checkpoint_object_count,
        "total_bytes": spec.checkpoint_size_bytes,
        "object_inventory_sha256": spec.checkpoint_object_inventory_sha256,
    }
    changed = sorted(key for key, value in summary.items() if inventory.get(key) != value)
    if changed:
        raise ValueError(
            "openpi_current_reference_inventory_summary_mismatch:" + ",".join(changed)
        )
    objects = inventory.get("objects")
    if not isinstance(objects, list) or canonical_sha256(objects) != (
        spec.checkpoint_object_inventory_sha256
    ):
        raise ValueError("openpi_current_reference_object_inventory_invalid")
    prefix = spec.checkpoint_uri.removeprefix("gs://openpi-assets/").rstrip("/") + "/"
    expected_paths: set[str] = set()
    verified: list[dict[str, Any]] = []
    for row in objects:
        if not isinstance(row, Mapping):
            raise ValueError("openpi_current_reference_object_row_invalid")
        name = str(row.get("name") or "")
        if not name.startswith(prefix):
            raise ValueError("openpi_current_reference_object_outside_prefix")
        relative_text = name.removeprefix(prefix)
        relative = PurePosixPath(relative_text)
        if not relative_text or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("openpi_current_reference_object_path_invalid")
        path = checkpoint.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"openpi_current_reference_checkpoint_object_missing:{relative_text}"
            )
        size_bytes = int(row.get("size_bytes") or -1)
        if path.stat().st_size != size_bytes:
            raise ValueError(
                f"openpi_current_reference_checkpoint_object_size_mismatch:{relative_text}"
            )
        md5_base64 = str(row.get("md5_base64") or "")
        if not md5_base64 or _file_md5_base64(path) != md5_base64:
            raise ValueError(
                f"openpi_current_reference_checkpoint_object_md5_mismatch:{relative_text}"
            )
        expected_paths.add(relative_text)
        verified.append(
            {
                "relative_path": relative_text,
                "size_bytes": size_bytes,
                "generation": str(row.get("generation") or ""),
                "md5_base64": md5_base64,
            }
        )
    local_paths = {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    unexpected = sorted(local_paths - expected_paths)
    if unexpected:
        raise ValueError(
            "openpi_current_reference_checkpoint_unexpected_files:" + ",".join(unexpected)
        )
    receipt = {
        "schema_version": "openpi_current_reference_checkpoint_verification.v1",
        "policy_id": spec.policy_id,
        "checkpoint_uri": spec.checkpoint_uri,
        "checkpoint_manifest_sha256": spec.checkpoint_manifest_sha256,
        "verified_objects": verified,
    }
    return {
        "local_checkpoint_verified": True,
        "local_checkpoint_verification_sha256": canonical_sha256(receipt),
        "local_checkpoint_object_count": len(verified),
        "local_checkpoint_size_bytes": sum(row["size_bytes"] for row in verified),
    }


class OpenPICurrentReferenceDroidPolicyClient:
    """In-process client preserving deterministic request and native-response receipts."""

    learned_policy = True

    def __init__(
        self,
        *,
        spec: OpenPICurrentReferenceDroidPolicySpec,
        policy: Any,
        local_verification: Mapping[str, Any],
    ) -> None:
        spec.validate()
        if local_verification.get("local_checkpoint_verified") is not True:
            raise ValueError("openpi_current_reference_local_checkpoint_not_verified")
        if local_verification.get("local_checkpoint_object_count") != (
            spec.checkpoint_object_count
        ):
            raise ValueError("openpi_current_reference_local_checkpoint_count_mismatch")
        if local_verification.get("local_checkpoint_size_bytes") != spec.checkpoint_size_bytes:
            raise ValueError("openpi_current_reference_local_checkpoint_size_mismatch")
        self.policy_id = spec.policy_id
        self.action_space = spec.action_space
        self.action_chunk_rows = spec.action_chunk_rows
        self.open_loop_horizon = spec.executed_prefix_steps
        self._spec = spec
        self._policy = policy
        self._local_verification = dict(local_verification)
        self._receipts: list[dict[str, Any]] = []

    def infer(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        blockers = validate_droid_observation(
            observation, required_views=DROID_OPENPI_POLICY_VIEWS
        )
        if blockers:
            raise ValueError(f"openpi_current_reference_observation_invalid:{blockers[0]}")
        transport_observation = {
            view: observation[view] for view in DROID_OPENPI_POLICY_VIEWS
        }
        transport_observation.update(
            {
                "observation/joint_position": observation["observation/joint_position"],
                "observation/gripper_position": observation[
                    "observation/gripper_position"
                ],
                "prompt": observation["prompt"],
            }
        )
        raw_response = self._policy.infer(transport_observation)
        if not isinstance(raw_response, Mapping):
            raise ValueError("openpi_current_reference_response_not_mapping")
        action = raw_response.get("actions", raw_response.get("action"))
        action_blockers = validate_droid_action_chunk(
            action, expected_rows=self.action_chunk_rows
        )
        if action_blockers:
            raise ValueError(
                f"openpi_current_reference_action_invalid:{action_blockers[0]}"
            )
        native_action = np.asarray(action, dtype=np.float64).copy()
        request_identity = {
            "prompt": str(observation["prompt"]),
            "views": {
                view: _array_sha256(observation[view])
                for view in DROID_OPENPI_POLICY_VIEWS
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
            "policy_identity_sha256": self._spec.server_metadata()["identity_sha256"],
            "request_sha256": canonical_sha256(request_identity),
            "native_action_sha256": _array_sha256(native_action),
            "native_action_shape": list(native_action.shape),
            "native_action_semantics": self.action_space,
            "executed_prefix_steps": self.open_loop_horizon,
            "physical_outcome_accessed": False,
            "physical_future_observation_accessed": False,
        }
        receipt["receipt_sha256"] = canonical_sha256(receipt)
        self._receipts.append(receipt)
        return {"actions": native_action, "policy_request_receipt": receipt}

    def evidence_summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "transport": "in_process_openpi_policy_inference",
            "identity_verified": True,
            "policy_identity": self._spec.server_metadata(),
            "local_checkpoint_verification": self._local_verification,
            "request_count": len(self._receipts),
            "request_receipts": list(self._receipts),
        }


__all__ = [
    "CURRENT_REFERENCE_ACTION_ROWS",
    "CURRENT_REFERENCE_INVENTORY_FILES",
    "OpenPICurrentReferenceDroidPolicyClient",
    "OpenPICurrentReferenceDroidPolicySpec",
    "load_current_reference_policy_specs",
    "verify_local_current_reference_checkpoint",
]
