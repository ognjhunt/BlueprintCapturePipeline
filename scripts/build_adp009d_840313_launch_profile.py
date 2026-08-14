#!/usr/bin/env python3
"""Build the exact dry-only production launch profile for ADP-009D scene 840313.

The builder runs only from a clean immutable checkout at one protected-``main``
commit. Protected ``main`` may advance after that release is staged; it must not
silently change the source identity named by an already-built profile. The
builder re-verifies every materialized InteriorGS/SAGE byte, writes immutable
allocator-preflight inputs, and emits a Pipeline-owned profile whose mutable
outputs are rooted beneath ``{launch_run_root}`` by the dispatcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess  # nosec B404 - fixed git argv only
from pathlib import Path
from typing import Any, Mapping, Sequence

from blueprint_pipeline.evaluation_run_contract import validate_evaluation_run_spec
from blueprint_pipeline.adp009d_physics_backend_comparison import (
    DEFAULT_PHYSICS_BACKEND,
    build_backend_profile,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    LAUNCH_RUN_ROOT_PLACEHOLDER,
    canonical_digest,
    validate_launch_profile,
    verify_profile_immutable_inputs,
)
from blueprint_pipeline.task_evaluation_profile_preflight import (
    RELEASE_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
)


READINESS_PROFILE_ID = "adp009d-840313-franka-dry-v1"
PROFILE_ID_PREFIX = "adp009d-840313-franka-dry"
BUNDLE_ID = "adp009d-840313-interiorgs-sage-v1"
EXPECTED_BUNDLE_DIGEST = (
    "sha256:4cbf6781cd43cdf02353e0417aefd9ee4df1a65a99e7dbb2ef69a0a0170f22ba"
)
EXPECTED_SPEC_DIGEST = (
    "sha256:6e39daf5c5fc8a7e26d7cb34f53c6f9ac92756c1e86a5fc5ec70dd0e4e38b034"
)
EXPECTED_READINESS_DIGEST = (
    "sha256:7eb35f11e298038422cb5377ec60e35687f27532fd6430a7352ac58d80701e06"
)
#: The dry lane admits on *declared preconditions*, not on a measured readiness
#: receipt. The file previously declared `task_evaluation_runtime_readiness.v1`
#: -- the schema owned by `adp009d_live_readiness` -- while carrying three
#: blockers that module cannot emit and three of its six observation keys. Its
#: digest was self-consistent, so digest-binding could never catch it, and the
#: profile copied `allocator_artifact_manifest_not_emitted` into
#: `execution_admission` where it read as a measurement of the allocator. It
#: never was one. Pinning the schema keeps a placeholder from impersonating a
#: receipt again.
EXPECTED_READINESS_SCHEMA = "task_evaluation_runtime_readiness_precondition.v1"
MANIFEST_RELATIVE_ROOT = Path("docs/arm_decision_proof_v1/manifests")
SOURCE_MANIFEST_NAME = "adp009d_840313_interiorgs_sage_source_bundle.v1.json"
EVALUATION_SPEC_NAME = "adp009d_840313_evaluation_run.v1.json"
READINESS_NAME = "adp009d_840313_runtime_readiness.v1.json"
RAW_GITHUB_ROOT = "https://raw.githubusercontent.com/ognjhunt/BlueprintCapturePipeline"


class ProductionProfileBuildError(ValueError):
    """Raised when the exact production profile cannot be proven."""


def profile_id_for_source_commit(source_commit: str) -> str:
    """Give every immutable allocator identity its own publishable profile ID."""

    normalized = source_commit.strip().lower()
    if (
        len(normalized) != 40
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ProductionProfileBuildError("production_profile_source_commit_invalid")
    return f"{PROFILE_ID_PREFIX}-{normalized}"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProductionProfileBuildError(f"production_profile_input_invalid:{path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ProductionProfileBuildError(f"production_profile_input_invalid:{path.name}")
    return dict(value)


def _write_exact(path: Path, value: Mapping[str, Any]) -> None:
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        if path.read_bytes() != payload:
            raise ProductionProfileBuildError(f"immutable_profile_build_conflict:{path.name}")


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(  # nosec B603 - fixed git executable and internal argv
        ["git", "-C", str(repo_root), *args],
        shell=False,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise ProductionProfileBuildError("production_profile_git_identity_unavailable")
    return completed.stdout.strip()


def verify_protected_main_checkout(repo_root: Path, source_commit: str) -> None:
    """Require a clean checkout of a commit already protected by ``main``.

    Release worktrees are deliberately detached. Requiring their checkout to
    equal the *current* ``origin/main`` would race ordinary protected-main
    merges and turn an immutable profile into a moving target. A commit must
    instead be the exact clean checkout and an ancestor of the current protected
    ``origin/main`` reference.
    """

    normalized = source_commit.strip().lower()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ProductionProfileBuildError(
            "production_profile_checkout_not_exact_clean_main"
        )
    try:
        merged = _git(
            repo_root,
            "merge-base",
            "--is-ancestor",
            normalized,
            "origin/main",
        )
    except ProductionProfileBuildError:
        merged = None
    if (
        _git(repo_root, "rev-parse", "HEAD") != normalized
        or not _git(repo_root, "rev-parse", "origin/main")
        or merged is None
        or _git(repo_root, "status", "--porcelain")
    ):
        raise ProductionProfileBuildError("production_profile_checkout_not_exact_clean_main")


def verify_materialized_source_artifacts(
    source_bundle: Mapping[str, Any], production_input_root: Path
) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    rows = source_bundle.get("materialized_artifacts")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ProductionProfileBuildError("production_profile_source_artifact_set_invalid")
    for row_value in rows:
        row = dict(row_value) if isinstance(row_value, Mapping) else {}
        production_path = Path(str(row.get("production_path") or ""))
        expected_path = production_input_root / production_path.name
        if (
            not str(row.get("role") or "")
            or production_path.parent.name != BUNDLE_ID
            or expected_path.is_symlink()
            or not expected_path.is_file()
            or expected_path.stat().st_size != row.get("size_bytes")
            or _file_digest(expected_path) != row.get("sha256")
        ):
            raise ProductionProfileBuildError(
                f"production_profile_source_artifact_invalid:{row.get('role') or 'unknown'}"
            )
        verified.append(
            {
                "name": str(row["role"]),
                "path": str(expected_path.resolve()),
                "digest": str(row["sha256"]),
            }
        )
    return verified


def build_profile_release(
    *,
    source_commit: str,
    repo_root: str | Path,
    production_input_root: str | Path,
    provider_guard_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    inputs = Path(production_input_root).expanduser().resolve()
    guard = Path(provider_guard_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    verify_protected_main_checkout(repo, source_commit)
    profile_id = profile_id_for_source_commit(source_commit)

    manifest_root = repo / MANIFEST_RELATIVE_ROOT
    source_path = manifest_root / SOURCE_MANIFEST_NAME
    spec_path = manifest_root / EVALUATION_SPEC_NAME
    readiness_path = manifest_root / READINESS_NAME
    source_bundle = _read(source_path)
    evaluation_spec = _read(spec_path)
    readiness = _read(readiness_path)
    validation = validate_evaluation_run_spec(evaluation_spec)
    if (
        source_bundle.get("bundle_id") != BUNDLE_ID
        or source_bundle.get("bundle_digest") != EXPECTED_BUNDLE_DIGEST
        or validation.get("status") != "passed"
        or validation.get("spec_digest") != EXPECTED_SPEC_DIGEST
        or readiness.get("receipt_digest") != EXPECTED_READINESS_DIGEST
        or readiness.get("schema_version") != EXPECTED_READINESS_SCHEMA
        or readiness.get("profile_id") != READINESS_PROFILE_ID
        or readiness.get("status") != "blocked"
        or readiness.get("live_execution_enabled") is not False
    ):
        raise ProductionProfileBuildError("production_profile_frozen_contract_mismatch")
    live_blockers = readiness.get("blockers")
    if not isinstance(live_blockers, list) or not live_blockers:
        raise ProductionProfileBuildError("production_profile_readiness_blockers_missing")
    raw_inputs = verify_materialized_source_artifacts(source_bundle, inputs)

    tracked_inputs = {
        "source_bundle_manifest": source_path,
        "evaluation_run_spec": spec_path,
        "runtime_readiness": readiness_path,
    }
    tracked_refs = {
        name: {"path": str(path), "digest": _file_digest(path)}
        for name, path in tracked_inputs.items()
    }
    preflight_request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "profile_id": profile_id,
        "provider": "vast",
        "retry_cap": 0,
        "live_execution_authorized": False,
        "immutable_inputs": tracked_refs,
        "required_provider_zero": ["digitalocean", "runpod", "vast"],
        "max_guard_age_seconds": 300,
    }
    preflight_request["request_digest"] = canonical_digest(
        preflight_request, digest_field="request_digest"
    )
    release_evidence = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": "passed",
        "repository": "ognjhunt/BlueprintCapturePipeline",
        "source_commit": source_commit,
        "source_ref": "main",
        "tracked_state": "clean",
    }
    release_evidence["release_digest"] = canonical_digest(
        release_evidence, digest_field="release_digest"
    )
    preflight_path = output / "task_evaluation_allocator_preflight_request.v1.json"
    release_path = output / "task_evaluation_pipeline_release_evidence.v1.json"
    _write_exact(preflight_path, preflight_request)
    _write_exact(release_path, release_evidence)

    immutable_inputs = [
        *(
            {
                "name": name,
                "path": str(path),
                "digest": tracked_refs[name]["digest"],
            }
            for name, path in tracked_inputs.items()
        ),
        *raw_inputs,
        {
            "name": "allocator_preflight_request",
            "path": str(preflight_path),
            "digest": _file_digest(preflight_path),
        },
        {
            "name": "pipeline_release_evidence",
            "path": str(release_path),
            "digest": _file_digest(release_path),
        },
    ]
    source_uri = f"{RAW_GITHUB_ROOT}/{source_commit}/{MANIFEST_RELATIVE_ROOT.as_posix()}"
    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": profile_id,
        "program_id": "arm-decision-proof-v1",
        "physics_backend": DEFAULT_PHYSICS_BACKEND,
        "physics_backend_profile_digest": build_backend_profile(
            DEFAULT_PHYSICS_BACKEND
        )["profile_digest"],
        "source_bundle": {
            "bundle_id": BUNDLE_ID,
            "source_kind": "interiorgs_sage",
            "uri": f"{source_uri}/{SOURCE_MANIFEST_NAME}",
            "digest": EXPECTED_BUNDLE_DIGEST,
        },
        "evaluation_run_spec": {
            "uri": f"{source_uri}/{EVALUATION_SPEC_NAME}",
            "digest": EXPECTED_SPEC_DIGEST,
        },
        "immutable_inputs": immutable_inputs,
        "allocator": {
            "entrypoint": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "subcommand": "gpu-canary",
            "argv": [
                "--provider-launch-request",
                str(preflight_path),
                "--release-evidence",
                str(release_path),
                "--model-cache-evidence",
                str(readiness_path),
                "--preflight-bundle",
                str(guard),
                "--admission-out",
                f"{LAUNCH_RUN_ROOT_PLACEHOLDER}/allocator/admission.json",
                "--bound-request-out",
                f"{LAUNCH_RUN_ROOT_PLACEHOLDER}/allocator/bound-request.json",
                "--adapter-output",
                f"{LAUNCH_RUN_ROOT_PLACEHOLDER}/allocator/result.json",
                "--pod-name",
                profile_id,
                "--expected-source-commit",
                source_commit,
                "--provider",
                "vast",
                "--probe-kind",
                "task-evaluation-profile-preflight",
                "--adp-max-spend-usd",
                "6.0",
                "--adp-hard-ttl-seconds",
                "5400",
            ],
            "max_spend_usd": 6.0,
            "hard_ttl_seconds": 5400,
            "retry_cap": 0,
        },
        "runtime_environment": {},
        "execution_admission": {
            "live_enabled": False,
            "readiness_receipt": {
                "uri": f"{source_uri}/{READINESS_NAME}",
                "digest": EXPECTED_READINESS_DIGEST,
            },
            "blockers": list(live_blockers),
        },
        "reconciliation": {
            "required_providers": ["vast"],
            "max_guard_age_seconds": 300,
        },
        "webapp_sync": {"max_attempts": 20},
        "terminal_contract": {
            "result_path": f"{LAUNCH_RUN_ROOT_PLACEHOLDER}/allocator/result.json",
            "success_statuses": ["completed"],
            "required_values": {
                "continuing_spend_from_this_run": False,
                "retry_cap": 0,
            },
            "required_path_fields": [
                "teardown_manifest_path",
                "artifact_manifest_path",
            ],
        },
        "required_controls": {
            "canonical_allocator": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "secret_profile_id": "canonical-vast-adp",
            "watchdog_required": True,
            "artifact_storage_required": True,
            "teardown_required": True,
            "provider_zero_required": True,
            "webapp_status_sync_required": True,
            "retry_cap": 0,
        },
        "claim_ceiling": "development_only",
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    blockers = [*validate_launch_profile(profile), *verify_profile_immutable_inputs(profile)]
    if blockers:
        raise ProductionProfileBuildError(",".join(sorted(set(blockers))))
    profile_path = output / f"{profile_id}.json"
    _write_exact(profile_path, profile)
    receipt = {
        "schema_version": "adp009d_840313_launch_profile_build_receipt.v1",
        "status": "built",
        "source_commit": source_commit,
        "profile_id": profile_id,
        "profile_digest": profile["profile_digest"],
        "physics_backend": DEFAULT_PHYSICS_BACKEND,
        "physics_backend_profile_digest": profile["physics_backend_profile_digest"],
        "source_bundle_digest": EXPECTED_BUNDLE_DIGEST,
        "evaluation_run_spec_digest": EXPECTED_SPEC_DIGEST,
        "runtime_readiness_digest": EXPECTED_READINESS_DIGEST,
        "profile_path": str(profile_path),
        "preflight_request_path": str(preflight_path),
        "release_evidence_path": str(release_path),
        "provider_mutation_performed": False,
        "live_execution_enabled": False,
        "blockers": list(live_blockers),
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write_exact(output / "profile_build_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--production-input-root", required=True)
    parser.add_argument("--provider-guard-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        receipt = build_profile_release(
            source_commit=args.source_commit,
            repo_root=args.repo_root,
            production_input_root=args.production_input_root,
            provider_guard_path=args.provider_guard_path,
            output_dir=args.output_dir,
        )
    except (OSError, json.JSONDecodeError, ProductionProfileBuildError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "adp009d_840313_launch_profile_build_receipt.v1",
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
