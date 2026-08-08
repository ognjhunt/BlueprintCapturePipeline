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
from collections.abc import Sequence
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        rows.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    if not rows:
        raise ValueError("groot_worker_checkpoint_files_missing")
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "checkpoint_files_sha256": hashlib.sha256(encoded).hexdigest(),
    }


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
        if inventory["total_bytes"] != EXPECTED_CHECKPOINT_BYTES:
            blockers.append("groot_worker_checkpoint_byte_count_mismatch")
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
