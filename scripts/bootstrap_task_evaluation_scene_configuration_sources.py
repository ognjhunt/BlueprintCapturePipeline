#!/usr/bin/env python3
"""Materialize the two pinned public scene-configuration source mirrors.

This command is a no-spend platform bootstrap.  It creates detached, clean
checkouts for the already admitted ArtiFixer and Content Agents releases and
refuses any commit/tree mismatch.  Per-scene Website runs never clone source.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from blueprint_pipeline.adp_content_agents_vast import (
    SOURCE_COMMIT as CONTENT_AGENTS_COMMIT,
    SOURCE_TREE as CONTENT_AGENTS_TREE,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, canonical_json
from blueprint_pipeline.public_scene_artifixer3d_bundle import (
    ARTIFIXER_COMMIT,
    ARTIFIXER_REPOSITORY,
    ARTIFIXER_TREE,
)


SCHEMA_VERSION = "task_evaluation_scene_configuration_source_mirrors.v1"
CONTENT_AGENTS_REPOSITORY = "https://github.com/NVIDIA-Omniverse/usd-content-agents"


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(  # nosec B603 B607 - fixed git argv
        [
            "git",
            "-c",
            f"safe.directory={root}",
            "-C",
            str(root),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode:
        raise ValueError("scene_configuration_source_mirror_git_failed")
    return completed.stdout.strip()


def ensure_pinned_source_mirror(
    *,
    repository: str,
    commit: str,
    tree: str,
    destination: str | Path,
) -> dict[str, Any]:
    """Create or validate one exact detached clean public checkout."""

    output = Path(destination).expanduser().absolute()
    created = False
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
        )
        shutil.rmtree(temporary)
        completed = subprocess.run(  # nosec B603 B607 - fixed git argv
            ["git", "clone", "--no-checkout", "--filter=blob:none", repository, str(temporary)],
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
        if completed.returncode:
            shutil.rmtree(temporary, ignore_errors=True)
            raise ValueError("scene_configuration_source_mirror_clone_failed")
        try:
            _git(temporary, "checkout", "--detach", commit)
            if output.exists() or output.is_symlink():
                raise ValueError("scene_configuration_source_mirror_destination_conflict")
            os.rename(temporary, output)
            created = True
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    if (
        output.is_symlink()
        or not output.is_dir()
        or _git(output, "rev-parse", "HEAD") != commit
        or _git(output, "rev-parse", "HEAD^{tree}") != tree
        or _git(output, "status", "--porcelain=v1")
    ):
        raise ValueError("scene_configuration_source_mirror_identity_invalid")
    return {
        "repository": repository,
        "commit": commit,
        "tree": tree,
        "path": str(output),
        "created": created,
        "clean": True,
    }


def bootstrap_scene_configuration_source_mirrors(
    *,
    source_root: str | Path,
) -> dict[str, Any]:
    root = Path(source_root).expanduser().absolute()
    observed_mirrors = {
        "artifixer": ensure_pinned_source_mirror(
            repository=ARTIFIXER_REPOSITORY,
            commit=ARTIFIXER_COMMIT,
            tree=ARTIFIXER_TREE,
            destination=root / f"artifixer-{ARTIFIXER_COMMIT[:8]}",
        ),
        "content_agents": ensure_pinned_source_mirror(
            repository=CONTENT_AGENTS_REPOSITORY,
            commit=CONTENT_AGENTS_COMMIT,
            tree=CONTENT_AGENTS_TREE,
            destination=root
            / f"usd-content-agents-v0.5.2-{CONTENT_AGENTS_COMMIT[:8]}",
        ),
    }
    mirrors = {
        name: {key: value for key, value in row.items() if key != "created"}
        for name, row in observed_mirrors.items()
    }
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "mirrors": mirrors,
        "provider_mutation_performed": False,
        "paid_resource_allocated": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = root / f"{SCHEMA_VERSION}.json"
    payload = (canonical_json(receipt) + "\n").encode("utf-8")
    if receipt_path.exists():
        if receipt_path.is_symlink() or receipt_path.read_bytes() != payload:
            raise ValueError("scene_configuration_source_mirror_receipt_conflict")
    else:
        with receipt_path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        receipt_path.chmod(0o444)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    args = parser.parse_args()
    value = bootstrap_scene_configuration_source_mirrors(source_root=args.source_root)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
