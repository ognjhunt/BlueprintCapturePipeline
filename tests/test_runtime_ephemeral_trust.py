from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline.g1_kitchen_proof_row_validation import (
    load_attestation_pins,
    verify_leaf_attestation,
)
from blueprint_pipeline.runtime_ephemeral_trust import SIGNERS, create_attempt_trust


def _identity() -> dict[str, str]:
    return {
        "run_id": "run-1",
        "attempt_id": "attempt-1",
        "launch_nonce": "nonce-1",
        "provider_allocation_id": "do-1",
    }


def test_attempt_trust_publishes_verifiable_role_pins_without_private_material(
    tmp_path: Path,
) -> None:
    secret_root = tmp_path / "secrets"
    environment_file = secret_root / "trust_env.sh"
    public_manifest = tmp_path / "runtime_ephemeral_trust.json"
    identity = _identity()

    manifest = create_attempt_trust(
        secret_root=secret_root,
        environment_file=environment_file,
        public_manifest=public_manifest,
        identity_binding=identity,
    )

    assert manifest["schema_version"] == "g1_kitchen_attestation_public_key_pins.v1"
    assert manifest["algorithm"] == "ed25519"
    assert manifest["identity_binding"] == identity
    assert manifest["private_keys_retained"] is False
    assert set(manifest["roles"]) == {row[1] for row in SIGNERS}
    assert len(manifest["public_keys"]) == len(SIGNERS)
    assert "PRIVATE KEY" not in public_manifest.read_text(encoding="utf-8")
    assert environment_file.stat().st_mode & 0o077 == 0

    pins = load_attestation_pins(public_manifest)
    assert pins is not None
    data = b"exact leaf bytes\n"
    for name, role, _, _ in SIGNERS:
        private_path = secret_root / f"{name}.pem"
        assert private_path.stat().st_mode & 0o077 == 0
        key = serialization.load_pem_private_key(private_path.read_bytes(), password=None)
        assert isinstance(key, Ed25519PrivateKey)
        raw = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        fingerprint = hashlib.sha256(raw).hexdigest()
        assert manifest["roles"][role] == [fingerprint]
        assert base64.b64decode(manifest["public_keys"][fingerprint]) == raw
        assert verify_leaf_attestation(
            data=data,
            attestation={
                "algorithm": "ed25519",
                "role": role,
                "public_key_fingerprint": fingerprint,
                "signature_b64": base64.b64encode(key.sign(data)).decode("ascii"),
            },
            expected_role=role,
            pins=pins,
        ) == []

