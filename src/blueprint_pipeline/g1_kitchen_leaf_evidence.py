"""Emit immutable, attempt-bound worker leaves for host-side closure.

The writer signs the exact bytes referenced by a proof row.  It never signs a
different canonical projection and never fills identity fields from runtime
results.  Attempt identity comes only from the immutable input manifest plus
the provider allocation identity supplied by the worker environment.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .g1_kitchen_attempt_closure import IDENTITY_FIELDS

PROVIDER_ALLOCATION_ID_ENV = "BLUEPRINT_PROVIDER_ALLOCATION_ID"
ROLE_PRIVATE_KEY_ENVS = {
    "startup": "BLUEPRINT_SC3_STARTUP_PRIVATE_KEY_FILE",
    "policy": "BLUEPRINT_SC3_POLICY_PRIVATE_KEY_FILE",
    "task_transition": "BLUEPRINT_SC3_TASK_COMPLETION_PRIVATE_KEY_FILE",
    "controller": "BLUEPRINT_SC3_FK_EXECUTOR_PRIVATE_KEY_FILE",
    "scorer": "BLUEPRINT_SC3_CONSISTENCY_SCORER_PRIVATE_KEY_FILE",
    "semantic_review": "BLUEPRINT_SC3_SEMANTIC_REVIEW_PRIVATE_KEY_FILE",
    "geometry": "BLUEPRINT_SC3_GEOMETRY_PRIVATE_KEY_FILE",
}


def _digest_ref(artifacts: Mapping[str, Any], name: str) -> str:
    ref = artifacts.get(name)
    return str(dict(ref).get("sha256") or "") if isinstance(ref, Mapping) else ""


def load_attempt_identity(
    attempt_manifest_path: str | Path,
    *,
    provider_allocation_id: str | None = None,
) -> dict[str, str]:
    attempt = json.loads(Path(attempt_manifest_path).read_text(encoding="utf-8"))
    if not isinstance(attempt, Mapping):
        raise ValueError("attempt_input_manifest_not_object")
    artifacts = dict(attempt.get("artifacts") or {})
    image = str(attempt.get("image_digest") or "").rsplit("@sha256:", 1)[-1]
    image = image.removeprefix("sha256:")
    identity = {
        "run_id": str(attempt.get("run_id") or ""),
        "attempt_id": str(attempt.get("attempt_id") or ""),
        "launch_nonce": str(attempt.get("launch_nonce") or ""),
        "source_commit": str(attempt.get("source_commit") or ""),
        "source_dirty_patch_sha256": str(
            attempt.get("source_dirty_patch_sha256") or ""
        ),
        "image_digest": image,
        "bundle_digest": _digest_ref(artifacts, "bundle"),
        "kitchen_asset_digest": _digest_ref(artifacts, "kitchen_inventory"),
        "active_selection_sha256": _digest_ref(artifacts, "selection"),
        "task_contract_sha256": _digest_ref(artifacts, "task_success_contract"),
        "provider_allocation_id": str(
            provider_allocation_id
            or os.environ.get(PROVIDER_ALLOCATION_ID_ENV)
            or ""
        ),
    }
    missing = [field for field in IDENTITY_FIELDS if not identity.get(field)]
    if missing:
        raise ValueError("attempt_leaf_identity_missing:" + ",".join(missing))
    return identity


def write_attested_leaf(
    *,
    payload: Mapping[str, Any],
    path: str | Path,
    identity: Mapping[str, Any],
    role: str,
    private_key_file: str | Path | None = None,
    reference_path: str | None = None,
) -> dict[str, Any]:
    """Write and Ed25519-sign the exact leaf bytes returned in the reference."""
    if role not in ROLE_PRIVATE_KEY_ENVS:
        raise ValueError(f"leaf_attestation_role_unsupported:{role}")
    observed = dict(payload)
    if not str(observed.get("schema_version") or ""):
        raise ValueError("leaf_schema_version_missing")
    if "identity_binding" in observed and dict(observed["identity_binding"]) != dict(identity):
        raise ValueError("leaf_identity_binding_preexisting_mismatch")
    observed["identity_binding"] = dict(identity)
    data = (json.dumps(observed, indent=2, sort_keys=True) + "\n").encode("utf-8")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise RuntimeError(f"leaf_artifact_already_exists:{target}") from exc

    key_path = Path(
        private_key_file or os.environ.get(ROLE_PRIVATE_KEY_ENVS[role]) or ""
    )
    if not key_path.is_file():
        raise ValueError(f"leaf_attestation_private_key_missing:{role}")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError("leaf_attestation_private_key_not_ed25519")
    public = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return {
        "path": reference_path or str(target),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "schema_version": observed["schema_version"],
        "attestation": {
            "algorithm": "ed25519",
            "role": role,
            "public_key_fingerprint": hashlib.sha256(public).hexdigest(),
            "signature_b64": base64.b64encode(key.sign(data)).decode("ascii"),
        },
    }
