"""Create attempt-local runtime signer keys without retaining private material."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


SIGNERS = (
    (
        "startup",
        "BLUEPRINT_SC3_STARTUP_PRIVATE_KEY_FILE",
        "BLUEPRINT_SC3_STARTUP_TRUSTED_PUBLIC_KEY_SHA256",
    ),
    (
        "task_completion",
        "BLUEPRINT_SC3_TASK_COMPLETION_PRIVATE_KEY_FILE",
        "BLUEPRINT_SC3_TASK_COMPLETION_TRUSTED_PUBLIC_KEY_SHA256",
    ),
    (
        "policy",
        "BLUEPRINT_SC3_POLICY_PRIVATE_KEY_FILE",
        "BLUEPRINT_SC3_POLICY_TRUSTED_PUBLIC_KEY_SHA256",
    ),
    (
        "gear_sonic_fk",
        "BLUEPRINT_SC3_FK_EXECUTOR_PRIVATE_KEY_FILE",
        "BLUEPRINT_SC3_FK_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256",
    ),
    (
        "consistency_scorer",
        "BLUEPRINT_SC3_CONSISTENCY_SCORER_PRIVATE_KEY_FILE",
        "BLUEPRINT_SC3_CONSISTENCY_SCORER_TRUSTED_PUBLIC_KEY_SHA256",
    ),
    (
        "semantic_review",
        "BLUEPRINT_SC3_SEMANTIC_REVIEW_PRIVATE_KEY_FILE",
        "BLUEPRINT_SC3_SEMANTIC_REVIEW_TRUSTED_PUBLIC_KEY_SHA256",
    ),
    (
        "geometry",
        "BLUEPRINT_SC3_GEOMETRY_PRIVATE_KEY_FILE",
        "BLUEPRINT_SC3_GEOMETRY_TRUSTED_PUBLIC_KEY_SHA256",
    ),
)


def create_attempt_trust(
    *, secret_root: Path, environment_file: Path, public_manifest: Path
) -> dict[str, object]:
    secret_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    secret_root.chmod(0o700)
    rows: list[tuple[str, str, str, str]] = []
    for name, private_env, trust_env in SIGNERS:
        key = Ed25519PrivateKey.generate()
        private_path = secret_root / f"{name}.pem"
        private_path.write_bytes(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        private_path.chmod(0o600)
        public = key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        rows.append((private_env, str(private_path), trust_env, hashlib.sha256(public).hexdigest()))
    environment_file.write_text(
        "".join(
            f"export {private_env}='{private_path}'\nexport {trust_env}='{digest}'\n"
            for private_env, private_path, trust_env, digest in rows
        ),
        encoding="utf-8",
    )
    environment_file.chmod(0o600)
    manifest: dict[str, object] = {
        "schema_version": "g1_kitchen_runtime_ephemeral_trust.v1",
        "scope": "single_allocation_single_attempt",
        "private_keys_retained": False,
        "public_key_sha256": {trust_env: digest for _, _, trust_env, digest in rows},
    }
    public_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secret-root", required=True, type=Path)
    parser.add_argument("--environment-file", required=True, type=Path)
    parser.add_argument("--public-manifest", required=True, type=Path)
    args = parser.parse_args()
    create_attempt_trust(
        secret_root=args.secret_root,
        environment_file=args.environment_file,
        public_manifest=args.public_manifest,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
