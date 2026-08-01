"""Deterministic, non-authorizing license inventory for the worker image."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .reconstruction_worker_contracts import (
    REQUIREMENTS_LOCK_SHA256,
    ReconstructionWorkerContractError,
    build_worker_stack_manifest,
)


SCHEMA_VERSION = "reconstruction_worker_license_inventory.v1"
_REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s\\]+)\s*\\$")
_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})(?:\s*\\)?$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class ReconstructionWorkerLicenseInventoryError(ValueError):
    pass


def _normalize_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_hashed_requirements_lock(text: str) -> list[dict[str, Any]]:
    """Return exact requirements, rejecting unpinned or unhashed entries."""

    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line_number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement = _REQUIREMENT.fullmatch(line)
        if requirement is not None:
            if current is not None and not current["hashes"]:
                raise ReconstructionWorkerLicenseInventoryError(
                    f"requirement_hash_missing:{current['exact_requirement']}"
                )
            name, version = requirement.groups()
            current = {
                "name": _normalize_name(name),
                "recorded_name": name,
                "version": version,
                "exact_requirement": f"{_normalize_name(name)}=={version}",
                "hashes": [],
            }
            rows.append(current)
            continue
        digest = _HASH.fullmatch(line)
        if digest is not None and current is not None:
            current["hashes"].append("sha256:" + digest.group(1))
            continue
        raise ReconstructionWorkerLicenseInventoryError(
            f"requirements_lock_line_invalid:{line_number}"
        )
    if current is not None and not current["hashes"]:
        raise ReconstructionWorkerLicenseInventoryError(
            f"requirement_hash_missing:{current['exact_requirement']}"
        )
    if not rows:
        raise ReconstructionWorkerLicenseInventoryError("requirements_lock_empty")
    exact = [row["exact_requirement"] for row in rows]
    if len(exact) != len(set(exact)):
        raise ReconstructionWorkerLicenseInventoryError(
            "requirements_lock_inventory_duplicate"
        )
    return sorted(rows, key=lambda row: row["exact_requirement"])


def _policy_by_requirement(policy: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if policy.get("schema_version") != "blueprint.runtime_dependency_license_policy.v1":
        raise ReconstructionWorkerLicenseInventoryError("license_policy_schema_invalid")
    components = policy.get("components")
    if not isinstance(components, Mapping):
        raise ReconstructionWorkerLicenseInventoryError("license_policy_components_invalid")
    normalized: dict[str, Mapping[str, Any]] = {}
    for key, value in components.items():
        if not isinstance(key, str) or "==" not in key or not isinstance(value, Mapping):
            raise ReconstructionWorkerLicenseInventoryError("license_policy_entry_invalid")
        name, version = key.split("==", 1)
        canonical = f"{_normalize_name(name)}=={version}"
        if canonical in normalized:
            raise ReconstructionWorkerLicenseInventoryError("license_policy_duplicate_entry")
        normalized[canonical] = value
    return normalized


def build_reconstruction_worker_license_inventory(
    *,
    source_commit_sha: str,
    worker_stack_manifest: Mapping[str, Any],
    requirements_lock_path: str | Path,
    license_policy: Mapping[str, Any],
    as_of: date,
) -> dict[str, Any]:
    """Compile review gaps without granting internal-build or distribution rights."""

    if _COMMIT.fullmatch(source_commit_sha) is None:
        raise ReconstructionWorkerLicenseInventoryError("source_commit_sha_invalid")
    try:
        stack = build_worker_stack_manifest(worker_stack_manifest)
    except ReconstructionWorkerContractError as exc:
        raise ReconstructionWorkerLicenseInventoryError(
            "worker_stack_manifest_invalid"
        ) from exc
    if stack["source_commit_sha"] != source_commit_sha:
        raise ReconstructionWorkerLicenseInventoryError(
            "worker_stack_source_commit_mismatch"
        )
    lock_path = Path(requirements_lock_path)
    if not lock_path.is_file() or lock_path.is_symlink():
        raise ReconstructionWorkerLicenseInventoryError("requirements_lock_missing_or_unsafe")
    lock_digest = _sha256(lock_path)
    if lock_digest != REQUIREMENTS_LOCK_SHA256:
        raise ReconstructionWorkerLicenseInventoryError("requirements_lock_digest_mismatch")
    requirements = parse_hashed_requirements_lock(lock_path.read_text(encoding="utf-8"))
    policy = _policy_by_requirement(license_policy)
    blockers: list[str] = []
    dependency_rows: list[dict[str, Any]] = []
    for requirement in requirements:
        exact = requirement["exact_requirement"]
        review = policy.get(exact)
        expiry: date | None = None
        if review is not None:
            try:
                expiry = date.fromisoformat(str(review.get("expires_on") or ""))
            except ValueError:
                expiry = None
        approved = bool(
            review is not None
            and review.get("approved") is True
            and isinstance(review.get("license_expression"), str)
            and bool(review.get("license_expression"))
            and isinstance(review.get("owner"), str)
            and bool(review.get("owner"))
            and expiry is not None
            and expiry >= as_of
        )
        reason = None
        if review is None:
            reason = f"license_review_missing:{exact}"
        elif not approved:
            reason = f"license_review_invalid_or_expired:{exact}"
        if reason:
            blockers.append(reason)
        dependency_rows.append(
            {
                **requirement,
                "hashes": sorted(set(requirement["hashes"])),
                "review_status": "approved" if approved else "review_required",
                "license_expression": review.get("license_expression") if review else None,
                "review_owner": review.get("owner") if review else None,
                "reviewed_on": review.get("reviewed_on") if review else None,
                "expires_on": review.get("expires_on") if review else None,
                "review_source": review.get("source") if review else None,
            }
        )

    source_rows: list[dict[str, Any]] = []
    for component in stack["components"]:
        redistribution = str(component.get("redistribution") or "")
        review_required = "review_required" in redistribution
        if review_required:
            blockers.append(
                f"source_component_license_review_required:{component['component_id']}"
            )
        source_rows.append(
            {
                "component_id": component["component_id"],
                "name": component["name"],
                "version": component["version"],
                "source_revision": component["source_revision"],
                "license": component["license"],
                "redistribution": redistribution,
                "review_status": "review_required" if review_required else "declared",
            }
        )

    model_asset_rows: list[dict[str, Any]] = []
    for model_asset in stack["model_assets"]:
        license_declaration = str(model_asset.get("license") or "")
        review_required = "review_required" in license_declaration
        if review_required:
            blockers.append(
                f"model_asset_license_review_required:{model_asset['model_id']}"
            )
        model_asset_rows.append(
            {
                "model_id": model_asset["model_id"],
                "url": model_asset["url"],
                "digest": model_asset["digest"],
                "license": license_declaration,
                "review_status": "review_required" if review_required else "declared",
            }
        )

    review_policy = license_policy.get("review_policy")
    if not isinstance(review_policy, Mapping):
        raise ReconstructionWorkerLicenseInventoryError(
            "license_policy_review_policy_invalid"
        )
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_commit_sha": source_commit_sha,
        "worker_stack_manifest_digest": stack["worker_stack_manifest_digest"],
        "requirements_lock_digest": "sha256:" + lock_digest,
        "license_policy_schema_version": license_policy.get("schema_version"),
        "license_policy_digest": canonical_digest(license_policy),
        "license_policy_reviewed_on": review_policy.get("reviewed_on"),
        "as_of": as_of.isoformat(),
        "status": "review_required" if blockers else "policy_approved",
        "dependency_count": len(dependency_rows),
        "dependency_reviews": dependency_rows,
        "source_component_reviews": source_rows,
        "model_asset_reviews": model_asset_rows,
        "blockers": sorted(set(blockers)),
        "agent_approval_permitted": False,
        "internal_build_authorized": False,
        "redistribution_authorized": False,
        "commercial_distribution_authorized": False,
        "proof_effect": "none",
        "claim_ceiling": "license_inventory_and_review_gap_only",
    }
    artifact["license_inventory_digest"] = canonical_digest(
        artifact, digest_field="license_inventory_digest"
    )
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--worker-stack-manifest", required=True, type=Path)
    parser.add_argument("--requirements-lock", required=True, type=Path)
    parser.add_argument("--license-policy", required=True, type=Path)
    parser.add_argument("--as-of", required=True, type=date.fromisoformat)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    artifact = build_reconstruction_worker_license_inventory(
        source_commit_sha=args.source_commit,
        worker_stack_manifest=json.loads(
            args.worker_stack_manifest.read_text(encoding="utf-8")
        ),
        requirements_lock_path=args.requirements_lock,
        license_policy=json.loads(args.license_policy.read_text(encoding="utf-8")),
        as_of=args.as_of,
    )
    write_json(args.output, artifact)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "dependency_count": artifact["dependency_count"],
                "blocker_count": len(artifact["blockers"]),
                "license_inventory_digest": artifact["license_inventory_digest"],
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["status"] == "policy_approved" else 2


if __name__ == "__main__":
    raise SystemExit(main())
