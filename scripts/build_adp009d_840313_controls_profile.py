#!/usr/bin/env python3
"""Publish the website-triggerable ADP-009D controls-canary launch profile.

The controls canary was the last paid step still driven by a hand-written shell
runner over SSH, so every iteration needed an operator at a terminal.  This
builds the same invocation as an immutable ``task_evaluation_launch_profile.v1``
so the website queues it and the canonical dispatcher, allocator, watchdog,
reconciler, and provider-zero guard own execution exactly as they do for the
dry route.

The dry profile answers "may we execute the scientific evaluation", and its
readiness blockers include the controls pair that has not passed yet.  A
controls profile cannot inherit that gate without being circular: this run is
what produces the controls evidence.  Its admission is therefore its own -- the
exact protected-main source, the sealed and rehashed inputs, an API-confirmed
provider-zero guard, and the same spend/TTL/retry ceilings.  The claim ceiling
stays ``development_only``: a passing controls pair is harness evidence, never a
task or policy result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from blueprint_pipeline.adp009d_physics_backend_comparison import (
    DEFAULT_PHYSICS_BACKEND,
    build_backend_profile,
    normalize_physics_backend,
    validate_newton_canary_admission,
)
from blueprint_pipeline.evaluation_run_contract import validate_evaluation_run_spec
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

try:  # pytest imports scripts as a package; direct execution puts this dir on the path
    from scripts.build_adp009d_840313_launch_profile import (  # noqa: I001
        BUNDLE_ID,
        EVALUATION_SPEC_NAME,
        EXPECTED_BUNDLE_DIGEST,
        EXPECTED_SPEC_DIGEST,
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
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from build_adp009d_840313_launch_profile import (  # noqa: I001
        BUNDLE_ID,
        EVALUATION_SPEC_NAME,
        EXPECTED_BUNDLE_DIGEST,
        EXPECTED_SPEC_DIGEST,
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

PROFILE_ID_PREFIX = "adp009d-840313-franka-controls"
NEWTON_PROFILE_ID_PREFIX = "adp009d-840313-franka-newton-controls"
# The controls pair is currently blocked by a scene-geometry stall during
# descend, which halts every run before teardown evidence can show a terminal
# success.  Diagnostic mode runs the same immutable bundle, provider, scene
# construction, collider validation, resets, cameras, artifacts, and teardown
# with no control pair and no policy, so the production path can be exercised
# green while that physics problem is worked separately.  It proves the
# pipeline, never the task.
DIAGNOSTIC_PROFILE_ID_PREFIX = "adp009d-840313-franka-diagnostic"
EXECUTION_MODES = ("controls", "diagnostic")
MAX_SPEND_USD = 6.0
MAX_HOURLY_RATE_USD = 0.80
HARD_TTL_SECONDS = 5400
NEWTON_MAX_SPEND_USD = 2.0
# Controls-only inputs, resolved from the repository at the exact source commit
# and rehashed here so a drifted asset cannot reach a paid worker.
APPROVED_CAN_RELATIVE = Path(
    "docs/arm_decision_proof_v1/assets/adp009a_840313_canned_beverage_match_v2.usda"
)
HARNESS_MANIFEST_RELATIVE = MANIFEST_RELATIVE_ROOT / "adp009d_franka_eval_harness_manifest.v1.json"
SCENARIO_INSTANCE_RELATIVE = MANIFEST_RELATIVE_ROOT / "adp009d_canonical_scenario_instance.v1.json"
SAGE_COLLISION_NAME = "840313_collision.usd"
AURA_APPEARANCE_NAME = "aura_ghost_removed_appearance.usdz"
# Instances this lane creates, used to scope the pre-freeze liveness check.
LANE_INSTANCE_PREFIX = "blueprint-adp009d-"


def profile_id_for_source_commit(
    source_commit: str,
    *,
    mode: str = "controls",
    physics_backend: str = DEFAULT_PHYSICS_BACKEND,
) -> str:
    backend = normalize_physics_backend(physics_backend)
    if backend == "newton":
        if mode != "controls":
            raise ProductionProfileBuildError("newton_controls_profile_mode_invalid")
        prefix = NEWTON_PROFILE_ID_PREFIX
    else:
        prefix = PROFILE_ID_PREFIX if mode == "controls" else DIAGNOSTIC_PROFILE_ID_PREFIX
    return f"{prefix}-{source_commit}"


def build_controls_profile_release(
    *,
    source_commit: str,
    repo_root: str | Path,
    production_input_root: str | Path,
    runtime_input_root: str | Path,
    provider_guard_path: str | Path,
    output_dir: str | Path,
    mode: str = "controls",
    physics_backend: str = DEFAULT_PHYSICS_BACKEND,
    newton_canary_admission_path: str | Path | None = None,
) -> dict[str, Any]:
    if mode not in EXECUTION_MODES:
        raise ProductionProfileBuildError(f"controls_profile_mode_invalid:{mode}")
    backend = normalize_physics_backend(physics_backend)
    if backend == "newton" and mode != "controls":
        raise ProductionProfileBuildError("newton_controls_profile_mode_invalid")
    if backend == "physx" and newton_canary_admission_path is not None:
        raise ProductionProfileBuildError("physx_profile_newton_admission_forbidden")
    repo = Path(repo_root).expanduser().resolve()
    inputs = Path(production_input_root).expanduser().resolve()
    runtime_inputs = Path(runtime_input_root).expanduser().resolve()
    unresolved_guard = Path(provider_guard_path).expanduser()
    if unresolved_guard.is_symlink():
        raise ProductionProfileBuildError("controls_profile_provider_guard_symlink")
    guard = unresolved_guard.resolve()
    out = Path(output_dir).expanduser().resolve()

    verify_protected_main_checkout(repo, source_commit)

    source_path = repo / MANIFEST_RELATIVE_ROOT / SOURCE_MANIFEST_NAME
    spec_path = repo / MANIFEST_RELATIVE_ROOT / EVALUATION_SPEC_NAME
    source_bundle = _read(source_path)
    # The frozen digests are the ones the manifests declare for their own
    # content, not the hashes of the manifest files; the file hashes go into
    # `immutable_inputs` below, where the dispatcher rehashes them.
    if source_bundle.get("bundle_id") != BUNDLE_ID:
        raise ProductionProfileBuildError("controls_profile_bundle_id_mismatch")
    if source_bundle.get("bundle_digest") != EXPECTED_BUNDLE_DIGEST:
        raise ProductionProfileBuildError("controls_profile_source_bundle_digest_mismatch")
    validation = validate_evaluation_run_spec(_read(spec_path))
    if (
        validation.get("status") != "passed"
        or validation.get("spec_digest") != EXPECTED_SPEC_DIGEST
    ):
        raise ProductionProfileBuildError("controls_profile_spec_digest_mismatch")
    raw_inputs = list(verify_materialized_source_artifacts(source_bundle, inputs) or [])

    guard_report = _read(guard)
    if guard_report.get("status") != "passed":
        raise ProductionProfileBuildError("controls_profile_provider_guard_not_passed")
    # Scope the freeze to this lane rather than the whole fleet.  A concurrent
    # operator's unrelated instance says nothing about whether this profile's
    # inputs are sound, and demanding fleet-wide zero here would deadlock two
    # engineers sharing one provider account.  Fleet-wide zero remains the gate
    # for *claiming* provider zero after a run, which is a different claim.
    lane_instances = [
        row
        for row in (guard_report.get("instances") or [])
        if str((row or {}).get("name") or "").startswith(LANE_INSTANCE_PREFIX)
    ]
    newton_admission: dict[str, Any] | None = None
    newton_admission_path: Path | None = None
    allowed_active_instance_ids: list[int] = []
    if backend == "newton":
        if newton_canary_admission_path is None:
            raise ProductionProfileBuildError("newton_canary_admission_missing")
        unresolved_admission = Path(newton_canary_admission_path).expanduser()
        if unresolved_admission.is_symlink():
            raise ProductionProfileBuildError("newton_canary_admission_symlink")
        newton_admission_path = unresolved_admission.resolve()
        newton_admission = _read(newton_admission_path)
        admission_blockers = validate_newton_canary_admission(
            newton_admission,
            profile=build_backend_profile("newton"),
        )
        if admission_blockers:
            raise ProductionProfileBuildError(
                "newton_canary_admission_invalid:" + ",".join(admission_blockers)
            )
        max_spend_usd = float(newton_admission["max_spend_usd"])
        hard_ttl_seconds = int(newton_admission["hard_ttl_seconds"])
        if max_spend_usd > NEWTON_MAX_SPEND_USD:
            raise ProductionProfileBuildError("newton_controls_profile_spend_cap_exceeded")
        if newton_admission.get("provider_zero_precheck_digest") != canonical_digest(
            guard_report, digest_field="receipt_digest"
        ):
            raise ProductionProfileBuildError("newton_canary_admission_guard_mismatch")
        allowed_active_instance_ids = list(
            newton_admission.get("allowed_active_vast_instance_ids") or []
        )
    else:
        max_spend_usd = MAX_SPEND_USD
        hard_ttl_seconds = HARD_TTL_SECONDS
    allowed_active_set = set(allowed_active_instance_ids)
    invalid_lane_instance_id = any(not str(row.get("id") or "").isdigit() for row in lane_instances)
    lane_instance_ids = {
        int(str(row.get("id"))) for row in lane_instances if str(row.get("id") or "").isdigit()
    }
    if backend == "physx" and lane_instances:
        raise ProductionProfileBuildError(
            "controls_profile_lane_instance_live:"
            + ",".join(sorted(str(row.get("id")) for row in lane_instances))
        )
    if backend == "newton" and (
        invalid_lane_instance_id or lane_instance_ids != allowed_active_set
    ):
        raise ProductionProfileBuildError(
            "controls_profile_lane_instance_binding_invalid:"
            + ",".join(str(value) for value in sorted(lane_instance_ids))
        )

    approved_can = repo / APPROVED_CAN_RELATIVE
    harness_manifest = repo / HARNESS_MANIFEST_RELATIVE
    scenario_instance = repo / SCENARIO_INSTANCE_RELATIVE
    sage_collision = inputs / SAGE_COLLISION_NAME
    aura_appearance = runtime_inputs / AURA_APPEARANCE_NAME
    controls_inputs = {
        "approved_can": approved_can,
        "sage_collision": sage_collision,
        "harness_manifest": harness_manifest,
        "scenario_instance": scenario_instance,
        "aura_appearance": aura_appearance,
    }
    for name, path in controls_inputs.items():
        if not path.is_file():
            raise ProductionProfileBuildError(f"controls_profile_input_missing:{name}")
    aura_digest = _file_digest(aura_appearance)

    profile_id = profile_id_for_source_commit(source_commit, mode=mode, physics_backend=backend)
    readiness_path = repo / MANIFEST_RELATIVE_ROOT / "adp009d_840313_runtime_readiness.v1.json"
    tracked_inputs = {
        "source_bundle_manifest": source_path,
        "evaluation_run_spec": spec_path,
        "runtime_readiness": readiness_path,
        "provider_guard": guard,
        **controls_inputs,
    }
    if newton_admission_path is not None:
        tracked_inputs["newton_canary_admission"] = newton_admission_path
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
    preflight_path = out / "task_evaluation_allocator_preflight_request.v1.json"
    release_path = out / "task_evaluation_pipeline_release_evidence.v1.json"
    _write_exact(preflight_path, preflight_request)
    _write_exact(release_path, release_evidence)
    immutable_inputs = [
        *(
            {"name": name, "path": ref["path"], "digest": ref["digest"]}
            for name, ref in tracked_refs.items()
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
    max_hourly_rate_usd = min(MAX_HOURLY_RATE_USD, max_spend_usd)
    backend_profile = build_backend_profile(backend)
    allocator_argv = [
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
        "adp009d-franka-native-microcheck",
        "--adp009d-physics-backend",
        backend,
        "--adp009d-approved-can",
        str(approved_can),
        "--adp009d-sage-collision",
        str(sage_collision),
        "--adp009d-harness-manifest",
        str(harness_manifest),
        "--adp009d-scenario-instance",
        str(scenario_instance),
        "--adp009d-aura-particlefield",
        str(aura_appearance),
        "--adp009d-aura-particlefield-sha256",
        aura_digest,
        "--adp009d-controls" if mode == "controls" else "--adp009d-diagnostic-only",
        "--adp-job-dir",
        f"{LAUNCH_RUN_ROOT_PLACEHOLDER}/adp009d-job",
        "--adp-max-hourly-rate-usd",
        str(max_hourly_rate_usd),
        "--adp-max-spend-usd",
        str(max_spend_usd),
        "--adp-hard-ttl-seconds",
        str(hard_ttl_seconds),
    ]
    if newton_admission_path is not None:
        allocator_argv.extend(["--adp009d-newton-canary-admission", str(newton_admission_path)])
    for instance_id in allowed_active_instance_ids:
        allocator_argv.extend(["--adp-allowed-active-vast-instance-id", str(instance_id)])
    profile: dict[str, Any] = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": profile_id,
        "program_id": "arm-decision-proof-v1",
        "physics_backend": backend,
        "physics_backend_profile_digest": backend_profile["profile_digest"],
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
            "argv": allocator_argv,
            "max_spend_usd": max_spend_usd,
            "hard_ttl_seconds": hard_ttl_seconds,
            "retry_cap": 0,
        },
        "runtime_environment": {},
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
        # This profile executes the controls pair itself, so it cannot be gated
        # on that pair having already passed.  Its readiness is the exact
        # protected-main source, the rehashed sealed inputs, and provider zero,
        # all verified above before this field is written.
        "execution_admission": {
            "live_enabled": True,
            "readiness_receipt": {
                "uri": (f"{source_uri}/adp009d_840313_runtime_readiness.v1.json"),
                "digest": _file_digest(readiness_path),
            },
            "blockers": [],
        },
        # A passing controls pair is harness evidence and never a task result.
        "claim_ceiling": "development_only",
        "reconciliation": {
            "required_providers": ["vast"],
            "max_guard_age_seconds": 300,
        },
        "webapp_sync": {"max_attempts": 20},
        "terminal_contract": {
            "result_path": f"{LAUNCH_RUN_ROOT_PLACEHOLDER}/allocator/result.json",
            # A blocked controls pair is a real scientific outcome, not a
            # successful run: only `completed` may close this profile.
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
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")

    blockers = [*validate_launch_profile(profile), *verify_profile_immutable_inputs(profile)]
    if blockers:
        raise ProductionProfileBuildError(",".join(sorted(set(blockers))))

    out.mkdir(parents=True, exist_ok=True)
    profile_path = out / f"{profile_id}.json"
    _write_exact(profile_path, profile)
    receipt = {
        "schema_version": "adp009d_840313_controls_profile_build_receipt.v1",
        "status": "built",
        "mode": mode,
        "source_commit": source_commit,
        "profile_id": profile_id,
        "profile_digest": profile["profile_digest"],
        "physics_backend": backend,
        "physics_backend_profile_digest": backend_profile["profile_digest"],
        "newton_canary_admission_digest": (
            newton_admission.get("admission_digest") if newton_admission else None
        ),
        "profile_path": str(profile_path),
        "live_execution_enabled": True,
        "claim_ceiling": "development_only",
        "provider_mutation_performed": False,
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    _write_exact(out / "controls_profile_build_receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--production-input-root", required=True)
    parser.add_argument("--runtime-input-root", required=True)
    parser.add_argument("--provider-guard-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=EXECUTION_MODES, default="controls")
    parser.add_argument(
        "--physics-backend",
        choices=("physx", "newton"),
        default=DEFAULT_PHYSICS_BACKEND,
    )
    parser.add_argument("--newton-canary-admission")
    args = parser.parse_args(argv)
    try:
        receipt = build_controls_profile_release(
            source_commit=args.source_commit,
            repo_root=args.repo_root,
            production_input_root=args.production_input_root,
            runtime_input_root=args.runtime_input_root,
            provider_guard_path=args.provider_guard_path,
            output_dir=args.output_dir,
            mode=args.mode,
            physics_backend=args.physics_backend,
            newton_canary_admission_path=args.newton_canary_admission,
        )
    except (OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": "adp009d_840313_controls_profile_build_receipt.v1",
                    "status": "blocked",
                    "error_type": type(exc).__name__,
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
