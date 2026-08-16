#!/usr/bin/env python3
"""Validate the runtime SBOM/license inventory and emit SPDX/provenance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 CI
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]


LICENSE_POLICY_SCHEMA = "blueprint.runtime_dependency_license_policy.v1"
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
IMAGE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
RELEASE_ARTIFACT_PATTERN = re.compile(r"^.+(?:\.whl|\.tar\.gz)$")
LOCK_REQUIREMENT_PATTERN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s\\;]+)")
# Exact-pin locks for runtime images that are deliberately outside the uv.lock
# SBOM (reviewed in docs/runtime_dependency_license_policy.json all the same).
REVIEWED_REQUIREMENTS_LOCKS = (
    Path("deploy/docker/reconstruction_worker/requirements.lock"),
)


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_sha(root: Path) -> str:
    configured = str(os.environ.get("GITHUB_SHA") or "").strip().lower()
    if configured:
        return configured
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip().lower() if completed.returncode == 0 else ""


def _component_key(name: object, version: object) -> str:
    return f"{str(name or '').strip().lower()}=={str(version or '').strip()}"


def read_reviewed_lock_keys(paths: list[Path]) -> frozenset[str]:
    """Exact ``name==version`` keys pinned by additional reviewed lock files.

    License-policy entries covered by these locks are reviewed for a runtime
    image outside the uv.lock SBOM, so they are not ``orphaned_license_review``
    blockers.  A missing lock contributes no exemptions (fail-closed: absence
    can only produce more blockers, never fewer).
    """

    keys: set[str] = set()
    for path in paths:
        if not path.is_file() or path.is_symlink():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            match = LOCK_REQUIREMENT_PATTERN.match(raw.strip())
            if match is not None:
                name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
                keys.add(f"{name}=={match.group(2)}")
    return frozenset(keys)


def _spdx_id(index: int, name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9.-]+", "-", name).strip("-.") or "package"
    return f"SPDXRef-Package-{index}-{slug}"


def build_evidence(
    *,
    root: Path,
    cyclonedx: Mapping[str, Any],
    license_policy: Mapping[str, Any],
    repository_sha: str,
    image_digest: str | None,
    artifact_paths: list[Path],
    today: date,
    additional_reviewed_component_keys: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    blockers: list[str] = []
    if cyclonedx.get("bomFormat") != "CycloneDX" or cyclonedx.get("specVersion") != "1.5":
        blockers.append("cyclonedx_1_5_schema_required")
    if license_policy.get("schema_version") != LICENSE_POLICY_SCHEMA:
        blockers.append("license_policy_schema_invalid")
    if SHA_PATTERN.fullmatch(repository_sha) is None:
        blockers.append("repository_sha_invalid")
    if image_digest is not None and IMAGE_PATTERN.fullmatch(image_digest) is None:
        blockers.append("image_digest_invalid")

    raw_components = cyclonedx.get("components")
    components = [
        dict(item) for item in raw_components if isinstance(item, Mapping)
    ] if isinstance(raw_components, list) else []
    if isinstance(raw_components, list) and len(components) != len(raw_components):
        blockers.append("cyclonedx_component_not_object")
    if not components:
        blockers.append("cyclonedx_components_empty")
    raw_approvals = license_policy.get("components")
    approvals = _mapping(raw_approvals)
    license_rows: list[dict[str, Any]] = []
    current_keys: set[str] = set()
    for component in components:
        key = _component_key(component.get("name"), component.get("version"))
        if key == "==":
            blockers.append("cyclonedx_component_coordinate_missing")
        elif not str(component.get("name") or "").strip() or not str(
            component.get("version") or ""
        ).strip():
            blockers.append(f"cyclonedx_component_coordinate_incomplete:{key}")
        if not str(component.get("purl") or "").startswith("pkg:"):
            blockers.append(f"cyclonedx_component_purl_missing:{key}")
        if key in current_keys:
            blockers.append(f"cyclonedx_component_duplicate:{key}")
        current_keys.add(key)
        approval = _mapping(approvals.get(key))
        row = {
            "component": key,
            "purl": component.get("purl"),
            "license_expression": approval.get("license_expression"),
            "approved": approval.get("approved") is True,
            "owner": approval.get("owner"),
            "reviewed_on": approval.get("reviewed_on"),
            "expires_on": approval.get("expires_on"),
            "source": approval.get("source"),
        }
        if not approval:
            blockers.append(f"license_review_missing:{key}")
        else:
            if approval.get("approved") is not True:
                blockers.append(f"license_not_approved:{key}")
            if len(str(approval.get("license_expression") or "").strip()) < 3:
                blockers.append(f"license_expression_missing:{key}")
            if len(str(approval.get("owner") or "").strip()) < 3:
                blockers.append(f"license_owner_missing:{key}")
            if len(str(approval.get("source") or "").strip()) < 8:
                blockers.append(f"license_source_missing:{key}")
            try:
                reviewed_on = date.fromisoformat(str(approval.get("reviewed_on") or ""))
                expires_on = date.fromisoformat(str(approval.get("expires_on") or ""))
            except ValueError:
                blockers.append(f"license_review_dates_invalid:{key}")
            else:
                if reviewed_on > today:
                    blockers.append(f"license_review_future:{key}")
                if expires_on < today:
                    blockers.append(f"license_review_expired:{key}")
                if expires_on < reviewed_on:
                    blockers.append(f"license_review_interval_invalid:{key}")
        license_rows.append(row)
    for key in sorted(
        set(approvals) - current_keys - set(additional_reviewed_component_keys)
    ):
        blockers.append(f"orphaned_license_review:{key}")

    materials: list[dict[str, Any]] = []
    if SHA_PATTERN.fullmatch(repository_sha) is not None:
        materials.append(
            {
                "uri": (
                    "git+https://github.com/ognjhunt/BlueprintCapturePipeline.git@"
                    f"{repository_sha}"
                ),
                "digest": {"gitCommit": repository_sha},
            }
        )
    for relative in (
        "uv.lock",
        "requirements.txt",
        "requirements-geometry.txt",
        "pyproject.toml",
    ):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            blockers.append(f"supply_chain_input_missing:{relative}")
            continue
        materials.append({"uri": relative, "digest": {"sha256": _sha256(path)}})
    subjects: list[dict[str, Any]] = []
    subject_names: set[str] = set()
    if len(artifact_paths) != len(set(artifact_paths)):
        blockers.append("release_artifact_argument_duplicate")
    for path in sorted(set(artifact_paths), key=lambda item: item.as_posix()):
        if not path.is_file() or path.is_symlink():
            blockers.append(f"release_artifact_invalid:{path.name or 'unnamed'}")
            continue
        if RELEASE_ARTIFACT_PATTERN.fullmatch(path.name) is None:
            blockers.append(f"release_artifact_type_invalid:{path.name}")
            continue
        if path.name in subject_names:
            blockers.append(f"release_artifact_name_duplicate:{path.name}")
            continue
        subject_names.add(path.name)
        subjects.append({"name": path.name, "digest": {"sha256": _sha256(path)}})
    wheel_count = sum(subject["name"].endswith(".whl") for subject in subjects)
    sdist_count = sum(subject["name"].endswith(".tar.gz") for subject in subjects)
    if wheel_count != 1:
        blockers.append(f"release_wheel_subject_count_invalid:{wheel_count}")
    if sdist_count != 1:
        blockers.append(f"release_sdist_subject_count_invalid:{sdist_count}")

    generated_at = datetime.now(timezone.utc).isoformat()
    cdx_out = dict(cyclonedx)
    metadata = _mapping(cdx_out.get("metadata"))
    properties = metadata.get("properties")
    property_rows = [dict(item) for item in properties if isinstance(item, Mapping)] if isinstance(properties, list) else []
    property_rows.extend(
        [
            {"name": "blueprint:repository_sha", "value": repository_sha},
            {"name": "blueprint:uv_lock_sha256", "value": next((m["digest"]["sha256"] for m in materials if m["uri"] == "uv.lock"), "")},
            {"name": "blueprint:image_digest", "value": image_digest or "not_bound"},
        ]
    )
    metadata["properties"] = property_rows
    cdx_out["metadata"] = metadata

    document_seed = hashlib.sha256(
        json.dumps(cdx_out, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    try:
        project = _mapping(
            tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8")).get(
                "project"
            )
        )
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        project = {}
        blockers.append("project_metadata_unreadable")
    root_name = str(project.get("name") or "").strip()
    root_version = str(project.get("version") or "").strip()
    root_license = str(project.get("license") or "").strip()
    if not root_name or not root_version or not root_license:
        blockers.append("project_spdx_metadata_incomplete")
    root_package_id = "SPDXRef-RootPackage"
    packages: list[dict[str, Any]] = [
        {
            "name": root_name or "blueprint-capture-pipeline",
            "SPDXID": root_package_id,
            "versionInfo": root_version or "NOASSERTION",
            "downloadLocation": "https://github.com/ognjhunt/BlueprintCapturePipeline",
            "filesAnalyzed": False,
            "licenseConcluded": root_license or "NOASSERTION",
            "licenseDeclared": root_license or "NOASSERTION",
            "copyrightText": "NOASSERTION",
        }
    ]
    relationships: list[dict[str, str]] = [
        {
            "spdxElementId": "SPDXRef-DOCUMENT",
            "relationshipType": "DESCRIBES",
            "relatedSpdxElement": root_package_id,
        }
    ]
    for index, (component, license_row) in enumerate(zip(components, license_rows), start=1):
        package_id = _spdx_id(index, str(component.get("name") or "package"))
        packages.append(
            {
                "name": component.get("name"),
                "SPDXID": package_id,
                "versionInfo": component.get("version"),
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseConcluded": license_row.get("license_expression") or "NOASSERTION",
                "licenseDeclared": license_row.get("license_expression") or "NOASSERTION",
                "copyrightText": "NOASSERTION",
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": component.get("purl"),
                    }
                ] if component.get("purl") else [],
            }
        )
        relationships.append(
            {
                "spdxElementId": root_package_id,
                "relationshipType": "DEPENDS_ON",
                "relatedSpdxElement": package_id,
            }
        )
    spdx = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"blueprint-capture-pipeline-{repository_sha[:12] or 'unbound'}",
        "documentNamespace": f"https://tryblueprint.io/spdx/{document_seed}",
        "creationInfo": {
            "created": generated_at,
            "creators": ["Tool: Blueprint build_supply_chain_evidence.py"],
        },
        "packages": packages,
        "relationships": relationships,
    }
    github_actions_build = os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true"
    builder_id = (
        "https://github.com/ognjhunt/BlueprintCapturePipeline/actions"
        if github_actions_build
        else "https://tryblueprint.io/builders/local-unsigned"
    )
    provenance = {
        "_type": "https://in-toto.io/Statement/v1",
        "subject": subjects,
        "predicateType": "https://slsa.dev/provenance/v1",
        "predicate": {
            "buildDefinition": {
                "buildType": "https://tryblueprint.io/buildtypes/python-uv/v1",
                "externalParameters": {
                    "repository_sha": repository_sha,
                    "image_digest": image_digest,
                    "frozen_lock_required": True,
                },
                "internalParameters": {},
                "resolvedDependencies": materials,
            },
            "runDetails": {
                "builder": {"id": builder_id},
                "metadata": {
                    "invocationId": (
                        os.environ.get("GITHUB_RUN_ID") if github_actions_build else None
                    )
                },
                "byproducts": [],
            },
        },
    }
    blockers = sorted(set(blockers))
    report = {
        "schema_version": "blueprint.supply_chain_evidence.v1",
        "generated_at": generated_at,
        "status": "passed" if not blockers else "blocked",
        "repository_sha": repository_sha or None,
        "image_digest": image_digest,
        "component_count": len(components),
        "license_review_count": len(license_rows),
        "artifact_subject_count": len(subjects),
        "blockers": blockers,
        "signature_status": "external_signature_not_verified",
        "claim_boundary": {
            "sbom_and_provenance_generated": not blockers,
            "provenance_is_unsigned_until_external_attestation": True,
            "container_signature_not_verified_by_this_command": True,
            "deployed_digest_signature_and_sbom_match_require_live_release_evidence": True,
        },
    }
    return cdx_out, spdx, provenance, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cyclonedx", type=Path, required=True)
    parser.add_argument(
        "--license-policy",
        type=Path,
        default=Path("docs/runtime_dependency_license_policy.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-sha")
    parser.add_argument("--image-digest")
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--reviewed-requirements-lock",
        type=Path,
        action="append",
        default=None,
        help=(
            "Exact-pin lock files (root-relative unless absolute) whose "
            "reviewed policy entries are exempt from orphaned_license_review; "
            "defaults to the reconstruction worker image lock"
        ),
    )
    args = parser.parse_args(argv)
    try:
        cyclonedx = json.loads(args.cyclonedx.read_text(encoding="utf-8"))
        policy = json.loads(args.license_policy.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"[supply-chain] ERROR unreadable_input:{exc}", file=sys.stderr)
        return 1
    root = args.root.resolve()
    reviewed_locks = (
        list(REVIEWED_REQUIREMENTS_LOCKS)
        if args.reviewed_requirements_lock is None
        else args.reviewed_requirements_lock
    )
    cdx, spdx, provenance, report = build_evidence(
        root=root,
        cyclonedx=_mapping(cyclonedx),
        license_policy=_mapping(policy),
        repository_sha=(args.repository_sha or _repository_sha(root)).strip().lower(),
        image_digest=args.image_digest.strip().lower() if args.image_digest else None,
        artifact_paths=[path.resolve() for path in args.artifact],
        today=date.today(),
        additional_reviewed_component_keys=read_reviewed_lock_keys(
            [path if path.is_absolute() else root / path for path in reviewed_locks]
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "runtime.cyclonedx.json": cdx,
        "runtime.spdx.json": spdx,
        "provenance.intoto.json": provenance,
        "supply-chain-report.json": report,
    }
    for name, payload in outputs.items():
        (args.output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        f"[supply-chain] status={report['status']} components={report['component_count']} "
        f"output={args.output_dir}"
    )
    for blocker in report["blockers"]:
        print(f"[supply-chain] blocker={blocker}", file=sys.stderr)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
