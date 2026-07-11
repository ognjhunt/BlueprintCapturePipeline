"""Source identity and strict schema compatibility for prepared G1 bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


COMPATIBILITY_SCHEMA_VERSION = "g1_kitchen_bundle_compatibility.v1"
REQUIRED_SCHEMAS = {
    "selection": "kitchen_random_task_selection.v1",
    "attempt_input": "g1_kitchen_attempt_input_manifest.v1",
    "controller_fk": "gear_sonic_controller_fk_execution.v1",
    "completion": "oscar_task_completion_evaluator_request.v1",
    "strict_scorer": "strict_action_aware_consistency_contract.v1",
    "closure": "g1_kitchen_attempt_closure.v1",
    "review_media": "isaac_full_ordered_episode_media_admission.v1",
    "worker_image_runtime_evidence": "g1_kitchen_worker_image_runtime_evidence.v1",
}


def _git(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, check=True
    )
    return completed.stdout


def build_source_tree_identity(repo_root: str | Path) -> dict[str, Any]:
    """Hash commit plus staged, unstaged, and untracked source bytes."""
    root = Path(repo_root).expanduser().resolve()
    commit = _git(root, "rev-parse", "HEAD").decode().strip()
    digest = hashlib.sha256()
    for label, args in (
        (b"staged\0", ("diff", "--binary", "--cached", "HEAD")),
        (b"unstaged\0", ("diff", "--binary",)),
    ):
        digest.update(label)
        digest.update(_git(root, *args))
    untracked = [
        line
        for line in _git(root, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
        if line
    ]
    for relative_bytes in sorted(untracked):
        relative = relative_bytes.decode("utf-8", errors="surrogateescape")
        path = root / relative
        if not path.is_file():
            continue
        digest.update(b"untracked\0" + relative_bytes + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return {
        "source_commit": commit,
        "source_dirty_patch_sha256": digest.hexdigest(),
        "dirty": bool(_git(root, "status", "--porcelain=v1", "-z")),
        "untracked_file_count": len(untracked),
        "identity_includes_staged_unstaged_and_untracked": True,
    }


def build_bundle_compatibility() -> dict[str, Any]:
    return {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "required_schemas": dict(REQUIRED_SCHEMAS),
        "prepared_before_strict_contracts_is_ineligible": True,
    }


def validate_bundle_compatibility(value: Any) -> dict[str, Any]:
    detail = dict(value) if isinstance(value, dict) else {}
    observed = detail.get("required_schemas")
    blockers: list[str] = []
    if detail.get("schema_version") != COMPATIBILITY_SCHEMA_VERSION:
        blockers.append("g1_bundle_compatibility_schema_mismatch")
    if not isinstance(observed, dict):
        blockers.append("g1_bundle_required_schemas_missing")
        observed = {}
    for name, expected in REQUIRED_SCHEMAS.items():
        if observed.get(name) != expected:
            blockers.append(f"g1_bundle_schema_incompatible:{name}")
    return {
        "status": "passed" if not blockers else "blocked",
        "blockers": blockers,
        "required_schemas": dict(REQUIRED_SCHEMAS),
        "observed_sha256": hashlib.sha256(
            json.dumps(detail, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    compatibility = manifest.get("compatibility") if isinstance(manifest, dict) else None
    result = validate_bundle_compatibility(compatibility)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
