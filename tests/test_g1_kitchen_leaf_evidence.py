from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from blueprint_pipeline.g1_kitchen_leaf_evidence import (
    load_attempt_identity,
    write_attested_leaf,
)


def _key(path: Path) -> Ed25519PrivateKey:
    key = Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return key


def _attempt(path: Path) -> None:
    digest = "a" * 64
    path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "attempt_id": "attempt-1",
                "launch_nonce": "nonce-1",
                "source_commit": "b" * 40,
                "source_dirty_patch_sha256": "c" * 64,
                "image_digest": f"sha256:{digest}",
                "artifacts": {
                    name: {"sha256": digest}
                    for name in (
                        "bundle",
                        "kitchen_inventory",
                        "selection",
                        "task_success_contract",
                    )
                },
            }
        ),
        encoding="utf-8",
    )


def test_exact_leaf_bytes_are_attempt_bound_and_signed(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt.json"
    _attempt(attempt)
    identity = load_attempt_identity(attempt, provider_allocation_id="do-1")
    key_path = tmp_path / "key.pem"
    key = _key(key_path)
    leaf = tmp_path / "leaf.json"
    ref = write_attested_leaf(
        payload={"schema_version": "example.v1", "passed": True},
        path=leaf,
        reference_path="closed_loop_out/proof_leaves/leaf.json",
        identity=identity,
        role="task_transition",
        private_key_file=key_path,
    )
    data = leaf.read_bytes()
    assert hashlib.sha256(data).hexdigest() == ref["sha256"]
    assert json.loads(data)["identity_binding"] == identity
    Ed25519PublicKey.from_public_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    ).verify(base64.b64decode(ref["attestation"]["signature_b64"]), data)


def test_missing_allocation_and_preexisting_identity_mismatch_block(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt.json"
    _attempt(attempt)
    with pytest.raises(ValueError, match="provider_allocation_id"):
        load_attempt_identity(attempt)
    identity = load_attempt_identity(attempt, provider_allocation_id="do-1")
    with pytest.raises(ValueError, match="preexisting_mismatch"):
        write_attested_leaf(
            payload={"schema_version": "example.v1", "identity_binding": {}},
            path=tmp_path / "leaf.json",
            identity=identity,
            role="task_transition",
            private_key_file=tmp_path / "missing.pem",
        )
