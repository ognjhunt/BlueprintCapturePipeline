"""Mandatory attempt/image/source identity gate before any paid allocation.

Every equality below is compared, never repaired: attempt image digest ==
launch digest == registry digest == worker-evidence digest, and attempt source
identity == bundle source identity == worker-evidence source identity (==
current host identity when supplied). Bundle compatibility and exact artifact
bytes are validated before a single provider resource may be created.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from collections.abc import Mapping
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .g1_kitchen_bundle_compatibility import validate_bundle_compatibility
from .g1_kitchen_bundle_compatibility import build_source_tree_identity
from .g1_kitchen_worker_image_evidence import validate_worker_image_runtime_evidence

SCHEMA_VERSION = "g1_kitchen_pre_allocation_identity_gate.v1"
REGISTRY_EVIDENCE_SCHEMA_VERSION = "g1_kitchen_registry_image_evidence.v1"
ATTEMPT_INPUT_SCHEMA_VERSION = "g1_kitchen_attempt_input_manifest.v1"
_INSPECT_DIGEST = re.compile(r"^Digest:\s*sha256:([0-9a-f]{64})\s*$", re.MULTILINE)


def inspect_registry_image(image_ref: str) -> dict[str, Any]:
    """Resolve the digest from the registry through Docker, never from a caller JSON file."""
    completed = subprocess.run(
        ["docker", "buildx", "imagetools", "inspect", image_ref],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    match = _INSPECT_DIGEST.search(completed.stdout)
    if match is None:
        raise RuntimeError("registry_image_digest_not_reported")
    return {
        "schema_version": REGISTRY_EVIDENCE_SCHEMA_VERSION,
        "source": "registry_api",
        "image_ref": image_ref,
        "digest": f"sha256:{match.group(1)}",
    }


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _digest_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "@sha256:" in text:
        text = text.rsplit("@sha256:", 1)[-1]
    if text.startswith("sha256:"):
        text = text[7:]
    if len(text) == 64 and all(char in "0123456789abcdef" for char in text):
        return text
    return ""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_manifest(bundle_path: Path) -> dict[str, Any] | None:
    try:
        with zipfile.ZipFile(bundle_path) as archive:
            payload = json.loads(archive.read("bundle_manifest.json"))
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError):
        return None
    return _mapping(payload) or None


def enforce_pre_allocation_identity_gate(
    *,
    attempt_input_manifest_file: str | Path,
    launch_image_ref: str,
    expected_source_identity: Mapping[str, Any] | None = None,
    registry_evidence_resolver: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return PASS only when every identity equality and byte hash verifies."""
    blockers: list[str] = []
    manifest_path = Path(attempt_input_manifest_file)
    attempt: dict[str, Any] = {}
    try:
        attempt = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        blockers.append("attempt_input_manifest_unreadable")
    if attempt and attempt.get("schema_version") != ATTEMPT_INPUT_SCHEMA_VERSION:
        blockers.append("attempt_input_manifest_schema_mismatch")

    attempt_digest = _digest_text(attempt.get("image_digest"))
    if not attempt_digest:
        blockers.append("attempt_image_digest_missing_or_invalid")
    launch_ref = str(launch_image_ref or "")
    launch_digest = _digest_text(launch_ref) if "@sha256:" in launch_ref else ""
    if not launch_digest:
        blockers.append("launch_image_ref_not_digest_pinned")
    elif attempt_digest and launch_digest != attempt_digest:
        blockers.append("launch_image_digest_mismatch")

    try:
        registry = _mapping(
            (registry_evidence_resolver or inspect_registry_image)(launch_ref)
        )
    except (OSError, RuntimeError, subprocess.SubprocessError):
        registry = {}
        blockers.append("registry_image_live_inspection_failed")
    if not registry:
        blockers.append("registry_image_evidence_missing")
    else:
        if registry.get("schema_version") != REGISTRY_EVIDENCE_SCHEMA_VERSION:
            blockers.append("registry_image_evidence_schema_mismatch")
        if registry.get("source") != "registry_api":
            blockers.append("registry_image_evidence_source_not_registry_api")
        if str(registry.get("image_ref") or "") != launch_ref:
            blockers.append("registry_image_evidence_ref_mismatch")
        registry_digest = _digest_text(registry.get("digest"))
        if not registry_digest:
            blockers.append("registry_image_evidence_digest_invalid")
        elif attempt_digest and registry_digest != attempt_digest:
            blockers.append("registry_digest_mismatch")

    source_commit = str(attempt.get("source_commit") or "").strip().lower()
    dirty_patch = str(attempt.get("source_dirty_patch_sha256") or "").strip().lower()
    if not source_commit:
        blockers.append("attempt_source_commit_missing")
    if len(dirty_patch) != 64:
        blockers.append("attempt_source_dirty_patch_sha256_missing")
    expected = _mapping(expected_source_identity)
    if expected:
        if str(expected.get("source_commit") or "").lower() != source_commit:
            blockers.append("attempt_source_commit_mismatch:current_host_identity")
        if (
            str(expected.get("source_dirty_patch_sha256") or "").lower()
            != dirty_patch
        ):
            blockers.append("attempt_source_dirty_patch_mismatch:current_host_identity")

    artifacts = _mapping(attempt.get("artifacts"))
    verified_paths: dict[str, Path] = {}
    for name, raw_ref in sorted(artifacts.items()):
        ref = _mapping(raw_ref)
        path = Path(str(ref.get("path") or ""))
        declared = str(ref.get("sha256") or "").lower()
        if not path.is_file():
            blockers.append(f"attempt_artifact_missing:{name}")
            continue
        if _sha256_file(path) != declared:
            blockers.append(f"attempt_artifact_sha256_mismatch:{name}")
            continue
        verified_paths[name] = path

    bundle_path = verified_paths.get("bundle")
    if bundle_path is None:
        blockers.append("attempt_bundle_artifact_not_verified")
    else:
        bundle_manifest = _bundle_manifest(bundle_path)
        if bundle_manifest is None:
            blockers.append("bundle_manifest_missing_or_unreadable")
        else:
            compatibility = validate_bundle_compatibility(
                bundle_manifest.get("compatibility")
            )
            if compatibility["status"] != "passed":
                blockers.extend(compatibility["blockers"])
            bundle_source = _mapping(bundle_manifest.get("source_tree_identity"))
            if str(bundle_source.get("source_commit") or "").lower() != source_commit:
                blockers.append("bundle_source_commit_mismatch")
            if (
                str(bundle_source.get("source_dirty_patch_sha256") or "").lower()
                != dirty_patch
            ):
                blockers.append("bundle_source_dirty_patch_mismatch")

    evidence_path = verified_paths.get("worker_image_runtime_evidence")
    if evidence_path is None:
        blockers.append("worker_image_runtime_evidence_not_verified")
    else:
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            evidence = None
        validation = validate_worker_image_runtime_evidence(
            evidence,
            expected_image_digest=f"sha256:{attempt_digest}" if attempt_digest else None,
            expected_source_commit=source_commit or None,
            expected_dirty_patch_sha256=dirty_patch or None,
            expected_run_id=str(attempt.get("run_id") or "") or None,
            expected_attempt_id=str(attempt.get("attempt_id") or "") or None,
            expected_launch_nonce=str(attempt.get("launch_nonce") or "") or None,
        )
        if validation["status"] != "passed":
            blockers.extend(validation["blockers"])

    blockers = sorted(set(blockers))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "identity": {
            "run_id": attempt.get("run_id"),
            "attempt_id": attempt.get("attempt_id"),
            "launch_nonce": attempt.get("launch_nonce"),
            "image_digest": attempt_digest or None,
            "source_commit": source_commit or None,
            "source_dirty_patch_sha256": dirty_patch or None,
        },
        "claim_boundary": {
            "pre_allocation_identity_only": True,
            "task_success_proven": False,
        },
    }


def enforce_current_checkout_pre_allocation_identity(
    *, attempt_input_manifest_file: str | Path, launch_image_ref: str, repo_root: str | Path
) -> dict[str, Any]:
    """Bind a paid launch to the live checkout and live registry in one fail-closed call."""
    try:
        expected = build_source_tree_identity(Path(repo_root))
    except (OSError, subprocess.SubprocessError):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKED",
            "blockers": ["current_source_identity_unavailable"],
            "identity": {},
        }
    return enforce_pre_allocation_identity_gate(
        attempt_input_manifest_file=attempt_input_manifest_file,
        launch_image_ref=launch_image_ref,
        expected_source_identity=expected,
    )


def revalidate_attempt_artifact_bytes(attempt_input_manifest_file: str | Path) -> list[str]:
    """Re-hash every declared input immediately before materialization or launch."""
    try:
        attempt = json.loads(Path(attempt_input_manifest_file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["attempt_input_manifest_unreadable"]
    blockers: list[str] = []
    for name, raw in sorted(_mapping(attempt.get("artifacts")).items()):
        ref = _mapping(raw)
        path = Path(str(ref.get("path") or ""))
        if not path.is_file():
            blockers.append(f"attempt_artifact_missing:{name}")
        elif _sha256_file(path) != str(ref.get("sha256") or "").lower():
            blockers.append(f"attempt_artifact_sha256_mismatch:{name}")
    return blockers
