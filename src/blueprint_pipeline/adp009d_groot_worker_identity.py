"""Observe and bind the exact GR00T worker source, checkpoint, and environment.

The public repository URI and pinned revision authorize a fetch; they do not
prove what bytes reached an ephemeral worker.  This helper runs after the fetch
and before server launch.  It hashes every materialized checkpoint file (while
excluding Hugging Face's local download metadata), verifies the observed byte
count and source checkout, and hashes the policy interpreter's installed
package lock.  A server receipt may then bind this independently observed
identity instead of asking the endpoint to identify itself.

It has no network or GPU dependency and is shipped in the flat provider bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "adp009d_groot_worker_identity.v1"
MODEL_ID = "nvidia/GR00T-N1.7-DROID"
EMBODIMENT_TAG = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
GROOT_SOURCE_REVISION = "b9955401d50c92a29258732e3ad6ccd579f1bdc0"
CHECKPOINT_REVISION = "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5"
EXPECTED_CHECKPOINT_BYTES = 6_914_267_987
EXPECTED_PUBLISHER_INVENTORY_SHA256 = (
    "5d1d83ab34215da2dcaa049d70e93ccec18687591ad5760c5183fc1fd6e035fd"
)
EXPECTED_CHECKPOINT_FILE_MANIFEST = (
    (".gitattributes", 1_578, "git_blob_sha1", "b63a0b1fdee3ac1542c4003322bdeefa02728f98"),
    ("README.md", 9_845, "git_blob_sha1", "2c1ba4777ef56afe180480aa9cf16e36fc6303ce"),
    ("SUCCESS", 0, "git_blob_sha1", "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"),
    ("config.json", 2_102, "git_blob_sha1", "11a846e5af6e5da63366a9b9f94f7a8b2d8da06b"),
    ("embodiment_id.json", 2_217, "git_blob_sha1", "272b0d0d047ad5d21ccd953549ec03554a539444"),
    ("experiment_cfg/conf.yaml", 8_855, "git_blob_sha1", "9f81ea0e195b88d908efc6a0d58986ce2b20e945"),
    ("experiment_cfg/config.yaml", 10_012, "git_blob_sha1", "1bcf8d6011a8ea8e4bfdf1d17855c0c7e1f3acac"),
    (
        "experiment_cfg/dataset_statistics.json", 288_228, "git_blob_sha1",
        "0d0383256ae692e8c97d274f630b6469c433e2c6",
    ),
    ("latest", 17, "git_blob_sha1", "11d33bb4033ce7ce7abb18127dacfbcf3592ab28"),
    (
        "model-00001-of-00002.safetensors", 4_990_519_232, "sha256",
        "68d885c9684bb7d4781389873e4b7d33202b5618e70a83f2e78187a5fb839202",
    ),
    (
        "model-00002-of-00002.safetensors", 1_919_980_184, "sha256",
        "aa4c6e553ea8454500354352368bcbb7e4f0fb32a9816b20d5b25c231f13a8fd",
    ),
    (
        "model-architecture.png", 3_184_280, "sha256",
        "bbf161c66daac88799b5bc7b5c110429f48d7f96ae315113e2b2e6d1f11c9e54",
    ),
    (
        "model.safetensors.index.json", 104_985, "git_blob_sha1",
        "4761851d9515098ac203474f61221a26cd3a98b6",
    ),
    (
        "processor_config.json", 2_833, "git_blob_sha1",
        "55b4d74b3565274662ba33eefe9bdb0ca75df3e9",
    ),
    (
        "scheduler.pt", 1_263, "sha256",
        "0e68cd5c0983910f321ec0ab989e4ea8c30670cfdbd27cf252aeb2b95587256c",
    ),
    (
        "statistics.json", 144_097, "git_blob_sha1",
        "03e76c7666bafe2e31fcc2320ee5ffcdddc6d675",
    ),
    (
        "training_args.bin", 8_259, "sha256",
        "8d636db8bb639e87538810ab7d76176a6a334d903728dd0bd0b3d2358f668e1b",
    ),
)


def expected_checkpoint_content_binding() -> dict[str, Any]:
    """Return the exact publisher-observed 17-file checkpoint identity."""

    files = [
        {
            "path": path,
            "size_bytes": size_bytes,
            "digest_algorithm": algorithm,
            "digest": digest,
        }
        for path, size_bytes, algorithm, digest in EXPECTED_CHECKPOINT_FILE_MANIFEST
    ]
    encoded = json.dumps(
        files, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "inventory_digest": "sha256:" + EXPECTED_PUBLISHER_INVENTORY_SHA256,
        "file_count": len(files),
        "total_bytes": sum(int(row["size_bytes"]) for row in files),
        "file_manifest": files,
        "file_manifest_digest": "sha256:" + hashlib.sha256(encoded).hexdigest(),
    }

# ``uv venv`` does not seed pip unless explicitly requested.  That is a valid,
# desirable policy environment: uv owns installation, while the venv stays
# isolated from Isaac.  Observe installed distributions through Python's
# standard-library metadata API so identity collection does not accidentally
# require a package manager inside the target environment.
_ENVIRONMENT_INVENTORY_CODE = r"""
import importlib.metadata as metadata
import json
import sys

rows = []
for distribution in metadata.distributions():
    name = distribution.metadata.get("Name")
    if not name:
        continue
    direct_url = distribution.read_text("direct_url.json")
    rows.append(
        {
            "name": name,
            "version": distribution.version,
            "direct_url": json.loads(direct_url) if direct_url else None,
        }
    )
rows.sort(
    key=lambda row: (
        row["name"].casefold(),
        row["version"],
        json.dumps(row["direct_url"], sort_keys=True, separators=(",", ":")),
    )
)
print(
    json.dumps(
        {
            "schema_version": "python_distribution_inventory.v1",
            "python": {"executable": sys.executable, "version": sys.version},
            "distributions": rows,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
""".strip()


def checkpoint_inventory(root: Path) -> dict[str, Any]:
    """Digest path, size, and content hash for all model files under ``root``."""

    root = root.resolve()
    if not root.is_dir():
        raise ValueError("groot_worker_checkpoint_root_missing")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if ".cache" in relative.parts or not path.is_file():
            continue
        size_bytes = path.stat().st_size
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1(usedforsecurity=False)
        git_blob.update(f"blob {size_bytes}\0".encode("ascii"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                sha256.update(chunk)
                git_blob.update(chunk)
        rows.append(
            {
                "path": relative.as_posix(),
                "size_bytes": size_bytes,
                "sha256": sha256.hexdigest(),
                "git_blob_sha1": git_blob.hexdigest(),
            }
        )
    if not rows:
        raise ValueError("groot_worker_checkpoint_files_missing")
    digest_rows = [
        {key: row[key] for key in ("path", "size_bytes", "sha256")}
        for row in rows
    ]
    encoded = json.dumps(
        digest_rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "checkpoint_files_sha256": hashlib.sha256(encoded).hexdigest(),
        "checkpoint_files": rows,
    }


def _checkpoint_content_blockers(inventory: Mapping[str, Any]) -> list[str]:
    expected = expected_checkpoint_content_binding()
    observed_by_path = {
        str(row.get("path") or ""): row
        for row in inventory.get("checkpoint_files") or []
        if isinstance(row, Mapping)
    }
    blockers: list[str] = []
    if inventory.get("file_count") != expected["file_count"]:
        blockers.append("groot_worker_checkpoint_file_count_mismatch")
    if inventory.get("total_bytes") != expected["total_bytes"]:
        blockers.append("groot_worker_checkpoint_byte_count_mismatch")
    expected_paths = {row["path"] for row in expected["file_manifest"]}
    if set(observed_by_path) != expected_paths:
        blockers.append("groot_worker_checkpoint_file_set_mismatch")
    for row in expected["file_manifest"]:
        observed = observed_by_path.get(row["path"])
        if observed is None:
            continue
        digest_field = str(row["digest_algorithm"])
        if (
            observed.get("size_bytes") != row["size_bytes"]
            or observed.get(digest_field) != row["digest"]
        ):
            blockers.append(
                f"groot_worker_checkpoint_file_identity_mismatch:{row['path']}"
            )
    return sorted(set(blockers))


def _observed_checkpoint_content_manifest_digest(
    inventory: Mapping[str, Any],
) -> str | None:
    observed_by_path = {
        str(row.get("path") or ""): row
        for row in inventory.get("checkpoint_files") or []
        if isinstance(row, Mapping)
    }
    rows: list[dict[str, Any]] = []
    for expected in expected_checkpoint_content_binding()["file_manifest"]:
        observed = observed_by_path.get(expected["path"])
        if observed is None:
            return None
        algorithm = str(expected["digest_algorithm"])
        rows.append(
            {
                "path": expected["path"],
                "size_bytes": observed.get("size_bytes"),
                "digest_algorithm": algorithm,
                "digest": observed.get(algorithm),
            }
        )
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _run_text(command: Sequence[str]) -> str:
    completed = subprocess.run(  # noqa: S603 - all argv are caller-bound paths
        list(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    return completed.stdout.strip()


def build_worker_identity(
    *, source_root: Path, checkpoint_root: Path, python: str
) -> dict[str, Any]:
    """Return a verified receipt or a typed blocked receipt."""

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "model_id": MODEL_ID,
        "embodiment_tag": EMBODIMENT_TAG,
        "groot_source_revision": GROOT_SOURCE_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "expected_checkpoint_bytes": EXPECTED_CHECKPOINT_BYTES,
        "publisher_inventory_sha256": EXPECTED_PUBLISHER_INVENTORY_SHA256,
        "publisher_inventory_role": "fetch_admission_not_local_content_digest",
    }
    blockers: list[str] = []
    try:
        observed_source_revision = _run_text(
            ("git", "-C", str(source_root), "rev-parse", "HEAD")
        )
        receipt["observed_groot_source_revision"] = observed_source_revision
        if observed_source_revision != GROOT_SOURCE_REVISION:
            blockers.append("groot_worker_source_revision_mismatch")
    except (OSError, subprocess.SubprocessError) as exc:
        receipt["source_observation_error"] = f"{type(exc).__name__}: {exc}"
        blockers.append("groot_worker_source_revision_unobserved")

    try:
        inventory = checkpoint_inventory(checkpoint_root)
        receipt.update(inventory)
        blockers.extend(_checkpoint_content_blockers(inventory))
        receipt["checkpoint_content_manifest_digest"] = (
            _observed_checkpoint_content_manifest_digest(inventory)
        )
    except (OSError, ValueError) as exc:
        receipt["checkpoint_observation_error"] = f"{type(exc).__name__}: {exc}"
        blockers.append("groot_worker_checkpoint_unobserved")

    try:
        environment_inventory_text = _run_text(
            (python, "-c", _ENVIRONMENT_INVENTORY_CODE)
        )
        environment_inventory = json.loads(environment_inventory_text)
        distributions = environment_inventory["distributions"]
        python_identity = environment_inventory["python"]
        environment_bytes = (environment_inventory_text + "\n").encode("utf-8")
        receipt.update(
            {
                "environment_lock_sha256": hashlib.sha256(environment_bytes).hexdigest(),
                "environment_lock_distribution_count": len(distributions),
                "environment_lock_observer": "stdlib_importlib_metadata",
                "python_identity": python_identity,
            }
        )
    except (
        KeyError,
        OSError,
        TypeError,
        subprocess.SubprocessError,
        json.JSONDecodeError,
    ) as exc:
        receipt["environment_observation_error"] = f"{type(exc).__name__}: {exc}"
        blockers.append("groot_worker_environment_unobserved")

    receipt["blockers"] = sorted(set(blockers))
    if not blockers:
        receipt["status"] = "verified"
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    receipt = build_worker_identity(
        source_root=Path(args.source_root),
        checkpoint_root=Path(args.checkpoint_root),
        python=args.python,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"BLUEPRINT_ADP009D_GROOT_WORKER_IDENTITY:{receipt['status']}")
    return 0 if receipt["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
