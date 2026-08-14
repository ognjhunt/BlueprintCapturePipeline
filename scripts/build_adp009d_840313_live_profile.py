#!/usr/bin/env python3
"""Build the live 840313 website profile from one passing protected-main canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from blueprint_pipeline.adp009d_840313_runtime_bundle import (
    verify_materialized_runtime_inputs,
)
from blueprint_pipeline.adp009d_physics_backend_comparison import (
    DEFAULT_PHYSICS_BACKEND,
    build_backend_profile,
)
from blueprint_pipeline.adp009d_live_readiness import (
    APPEARANCE_DIGEST,
    EVALUATION_RUN_SPEC_DIGEST,
    # Re-exported deliberately: the builder does not read it, and the live
    # builder test asserts against it through this module. Ruff reports it
    # unused, which is true of the module and false of its test surface --
    # the same pattern common.py documents.
    PROFILE_ID as READINESS_PROFILE_ID,  # noqa: F401
    RUNTIME_BUNDLE_DIGEST,
    SOURCE_BUNDLE_DIGEST,
    build_live_readiness,
)
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    LAUNCH_RUN_ROOT_PLACEHOLDER,
    canonical_digest,
    validate_launch_profile,
    verify_profile_immutable_inputs,
)
from blueprint_pipeline.task_evaluation_live_profile import shared_control_surface
from blueprint_pipeline.task_evaluation_profile_preflight import (
    RELEASE_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
)
from scripts.build_adp009d_840313_launch_profile import (
    BUNDLE_ID,
    EVALUATION_SPEC_NAME,
    MANIFEST_RELATIVE_ROOT,
    RAW_GITHUB_ROOT,
    SOURCE_MANIFEST_NAME,
    ProductionProfileBuildError,
    _file_digest,
    _read,
    _write_exact,
    verify_materialized_source_artifacts,
    verify_protected_main_checkout,
)


RUNTIME_MANIFEST_NAME = "adp009d_840313_franka_runtime_bundle.v1.json"
LIVE_PROFILE_ID_PREFIX = "adp009d-840313-franka-live"


def live_profile_id_for_source_commit(source_commit: str) -> str:
    """Give every immutable live allocator identity its own publication ID."""

    normalized = source_commit.strip().lower()
    if (
        len(normalized) != 40
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ProductionProfileBuildError("live_profile_source_commit_invalid")
    return f"{LIVE_PROFILE_ID_PREFIX}-{normalized}"


def _read_evidence(path: str | Path) -> dict[str, Any]:
    return _read(Path(path).expanduser().resolve())


def build_live_profile_release(
    *,
    source_commit: str,
    repo_root: str | Path,
    source_input_root: str | Path,
    runtime_input_root: str | Path,
    provider_guard_path: str | Path,
    release_evidence_path: str | Path,
    control_bundle_receipt_path: str | Path,
    allocator_result_path: str | Path,
    control_pair_path: str | Path,
    artifact_manifest_path: str | Path,
    teardown_manifest_path: str | Path,
    provider_zero_guard_path: str | Path,
    readiness_uri: str,
    output_dir: str | Path,
    revision: str | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    source_inputs = Path(source_input_root).expanduser().resolve()
    runtime_inputs = Path(runtime_input_root).expanduser().resolve()
    guard = Path(provider_guard_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    verify_protected_main_checkout(repo, source_commit)
    profile_id = live_profile_id_for_source_commit(source_commit)
    if revision:
        # Published profiles are immutable, so a profile whose inputs changed
        # needs its own id rather than a conflicting rewrite. Inputs change at
        # a fixed commit more often than they look like they would, and the
        # collision surfaces as an immutable-input digest mismatch on a later
        # launch rather than at publish time.
        profile_id = f"{profile_id}-{revision}"
    if not readiness_uri.startswith(("s3://", "gs://", "https://")):
        raise ProductionProfileBuildError("live_profile_readiness_uri_invalid")

    manifest_root = repo / MANIFEST_RELATIVE_ROOT
    source_path = manifest_root / SOURCE_MANIFEST_NAME
    spec_path = manifest_root / EVALUATION_SPEC_NAME
    runtime_manifest_path = manifest_root / RUNTIME_MANIFEST_NAME
    source_bundle = _read(source_path)
    runtime_bundle = _read(runtime_manifest_path)
    raw_inputs = verify_materialized_source_artifacts(source_bundle, source_inputs)
    runtime_rows = verify_materialized_runtime_inputs(
        runtime_bundle,
        runtime_input_root=runtime_inputs,
        source_input_root=source_inputs,
        repo_root=repo,
    )
    runtime_by_name = {str(row["name"]): row for row in runtime_rows}
    release = _read_evidence(release_evidence_path)
    bundle_receipt = _read_evidence(control_bundle_receipt_path)
    allocator_result = _read_evidence(allocator_result_path)
    control_pair = _read_evidence(control_pair_path)
    artifact_manifest = _read_evidence(artifact_manifest_path)
    teardown = _read_evidence(teardown_manifest_path)
    provider_zero = _read_evidence(provider_zero_guard_path)
    readiness = build_live_readiness(
        source_commit=source_commit,
        release_evidence=release,
        bundle_receipt=bundle_receipt,
        allocator_result=allocator_result,
        control_pair=control_pair,
        control_pair_path=control_pair_path,
        artifact_manifest=artifact_manifest,
        teardown_manifest=teardown,
        provider_zero_guard=provider_zero,
    )
    if readiness.get("status") != "passed" or readiness.get("blockers"):
        raise ProductionProfileBuildError(
            "live_profile_readiness_blocked:" + ",".join(readiness.get("blockers") or [])
        )
    readiness_path = output / "adp009d_840313_live_runtime_readiness.v1.json"
    _write_exact(readiness_path, readiness)

    tracked_inputs = {
        "source_bundle_manifest": source_path,
        "evaluation_run_spec": spec_path,
        "runtime_bundle_manifest": runtime_manifest_path,
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
        "live_execution_authorized": True,
        "immutable_inputs": tracked_refs,
        "required_provider_zero": ["digitalocean", "runpod", "vast"],
        "max_guard_age_seconds": 300,
    }
    preflight_request["request_digest"] = canonical_digest(
        preflight_request, digest_field="request_digest"
    )
    release_for_profile = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "status": "passed",
        "repository": "ognjhunt/BlueprintCapturePipeline",
        "source_commit": source_commit,
        "source_ref": "main",
        "tracked_state": "clean",
    }
    release_for_profile["release_digest"] = canonical_digest(
        release_for_profile, digest_field="release_digest"
    )
    preflight_path = output / "task_evaluation_allocator_preflight_request.v1.json"
    profile_release_path = output / "task_evaluation_pipeline_release_evidence.v1.json"
    _write_exact(preflight_path, preflight_request)
    _write_exact(profile_release_path, release_for_profile)

    immutable_inputs = [
        *(
            {"name": name, "path": str(path), "digest": tracked_refs[name]["digest"]}
            for name, path in tracked_inputs.items()
        ),
        *raw_inputs,
        *(
            {
                "name": f"runtime_{row['name']}",
                "path": str(row["path"]),
                "digest": str(row["digest"]),
            }
            for row in runtime_rows
        ),
        {
            "name": "allocator_preflight_request",
            "path": str(preflight_path),
            "digest": _file_digest(preflight_path),
        },
        {
            "name": "pipeline_release_evidence",
            "path": str(profile_release_path),
            "digest": _file_digest(profile_release_path),
        },
    ]
    source_uri = f"{RAW_GITHUB_ROOT}/{source_commit}/{MANIFEST_RELATIVE_ROOT.as_posix()}"
    profile: dict[str, Any] = {
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
            "digest": SOURCE_BUNDLE_DIGEST,
        },
        "evaluation_run_spec": {
            "uri": f"{source_uri}/{EVALUATION_SPEC_NAME}",
            "digest": EVALUATION_RUN_SPEC_DIGEST,
        },
        "immutable_inputs": immutable_inputs,
        "allocator": {
            "entrypoint": CANONICAL_ALLOCATOR_ENTRYPOINT,
            "subcommand": "gpu-canary",
            "argv": [
                "--provider-launch-request",
                str(preflight_path),
                "--release-evidence",
                str(profile_release_path),
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
                "adp009d-franka-native-microcheck",
                "--adp009d-physics-backend",
                DEFAULT_PHYSICS_BACKEND,
                "--adp009d-approved-can",
                runtime_by_name["approved_simready_can"]["path"],
                "--adp009d-sage-collision",
                runtime_by_name["static_collision_geometry"]["path"],
                "--adp009d-harness-manifest",
                runtime_by_name["franka_evaluation_harness"]["path"],
                "--adp009d-aura-particlefield",
                runtime_by_name["aura_nurec_appearance"]["path"],
                "--adp009d-aura-particlefield-sha256",
                APPEARANCE_DIGEST,
                "--adp009d-policy-candidate",
                "pi05_droid,groot_n17_droid",
                "--adp009d-controls",
                "--adp009d-scenario-instance",
                runtime_by_name["canonical_scenario_instance"]["path"],
                "--adp009d-authorize-gated-backbone",
                "--adp-job-dir",
                f"{LAUNCH_RUN_ROOT_PLACEHOLDER}/allocator/adp009d-job",
                "--adp-max-hourly-rate-usd",
                "0.80",
                "--adp-max-spend-usd",
                "6.0",
                "--adp-hard-ttl-seconds",
                "5400",
            ],
            "max_spend_usd": 6.0,
            "hard_ttl_seconds": 5400,
            "retry_cap": 0,
        },
        "runtime_environment": {"BLUEPRINT_ADP009D_CAMERA_RESOLUTION": "policy"},
        "execution_admission": {
            "live_enabled": True,
            "readiness_receipt": {
                "uri": readiness_uri,
                "digest": readiness["receipt_digest"],
            },
            "blockers": [],
        },
        # This lane's inputs are a preflight request and release evidence rather
        # than a receipt-resolved archive, so it does not share the full
        # skeleton in `task_evaluation_live_profile`. It shares the part that
        # decides whether a run is provable, because two copies of that drift.
        **shared_control_surface(),
        "claim_ceiling": "development_only",
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    blockers = [*validate_launch_profile(profile), *verify_profile_immutable_inputs(profile)]
    if blockers:
        raise ProductionProfileBuildError(",".join(sorted(set(blockers))))
    profile_path = output / f"{profile_id}.json"
    _write_exact(profile_path, profile)
    receipt = {
        "schema_version": "adp009d_840313_live_profile_build_receipt.v1",
        "status": "built",
        "source_commit": source_commit,
        "profile_id": profile_id,
        "profile_digest": profile["profile_digest"],
        "physics_backend": DEFAULT_PHYSICS_BACKEND,
        "physics_backend_profile_digest": profile["physics_backend_profile_digest"],
        "source_bundle_digest": SOURCE_BUNDLE_DIGEST,
        "evaluation_run_spec_digest": EVALUATION_RUN_SPEC_DIGEST,
        "runtime_bundle_digest": RUNTIME_BUNDLE_DIGEST,
        "runtime_readiness_digest": readiness["receipt_digest"],
        "profile_path": str(profile_path),
        "readiness_path": str(readiness_path),
        "provider_mutation_performed": False,
        "live_execution_enabled": True,
        "blockers": [],
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write_exact(output / "live_profile_build_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-commit",
        "repo-root",
        "source-input-root",
        "runtime-input-root",
        "provider-guard-path",
        "release-evidence-path",
        "control-bundle-receipt-path",
        "allocator-result-path",
        "control-pair-path",
        "artifact-manifest-path",
        "teardown-manifest-path",
        "provider-zero-guard-path",
        "readiness-uri",
        "output-dir",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument(
        "--revision",
        help="Distinguish a rebuilt profile whose inputs changed at the same commit.",
    )
    args = parser.parse_args(argv)
    try:
        receipt = build_live_profile_release(**vars(args))
    except (OSError, json.JSONDecodeError, ProductionProfileBuildError) as exc:
        print(json.dumps({"status": "blocked", "blockers": [str(exc)], "provider_mutation_performed": False}, sort_keys=True))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
