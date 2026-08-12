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

from blueprint_pipeline.evaluation_run_contract import validate_evaluation_run_spec
from blueprint_pipeline.task_evaluation_launch_dispatcher import (
    CANONICAL_ALLOCATOR_ENTRYPOINT,
    LAUNCH_RUN_ROOT_PLACEHOLDER,
    canonical_digest,
    validate_launch_profile,
    verify_profile_immutable_inputs,
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


def profile_id_for_source_commit(source_commit: str, *, mode: str = "controls") -> str:
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
) -> dict[str, Any]:
    if mode not in EXECUTION_MODES:
        raise ProductionProfileBuildError(f"controls_profile_mode_invalid:{mode}")
    repo = Path(repo_root).expanduser().resolve()
    inputs = Path(production_input_root).expanduser().resolve()
    runtime_inputs = Path(runtime_input_root).expanduser().resolve()
    guard = Path(provider_guard_path).expanduser().resolve()
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
    if validation.get("status") != "passed" or validation.get("spec_digest") != EXPECTED_SPEC_DIGEST:
        raise ProductionProfileBuildError("controls_profile_spec_digest_mismatch")
    verify_materialized_source_artifacts(source_bundle, inputs)

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
    if lane_instances:
        raise ProductionProfileBuildError(
            "controls_profile_lane_instance_live:"
            + ",".join(sorted(str(row.get("id")) for row in lane_instances))
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

    profile_id = profile_id_for_source_commit(source_commit, mode=mode)
    immutable_inputs = [
        {"name": "source_bundle_manifest", "path": str(source_path), "digest": _file_digest(source_path)},
        {"name": "evaluation_run_spec", "path": str(spec_path), "digest": _file_digest(spec_path)},
        *(
            {"name": name, "path": str(path), "digest": _file_digest(path)}
            for name, path in sorted(controls_inputs.items())
        ),
    ]
    source_uri = f"{RAW_GITHUB_ROOT}/{source_commit}/{MANIFEST_RELATIVE_ROOT.as_posix()}"
    profile: dict[str, Any] = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": profile_id,
        "program_id": "arm-decision-proof-v1",
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
                str(out / "task_evaluation_allocator_preflight_request.v1.json"),
                "--release-evidence",
                str(out / "task_evaluation_pipeline_release_evidence.v1.json"),
                "--model-cache-evidence",
                str(repo / MANIFEST_RELATIVE_ROOT / "adp009d_840313_runtime_readiness.v1.json"),
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
                # Exactly one execution mode: the allocator rejects a request
                # that names more than one.
                "--adp009d-controls" if mode == "controls" else "--adp009d-diagnostic-only",
                "--adp-job-dir",
                f"{LAUNCH_RUN_ROOT_PLACEHOLDER}/adp009d-job",
                "--adp-max-hourly-rate-usd",
                str(MAX_HOURLY_RATE_USD),
                "--adp-max-spend-usd",
                str(MAX_SPEND_USD),
                "--adp-hard-ttl-seconds",
                str(HARD_TTL_SECONDS),
            ],
            "max_spend_usd": MAX_SPEND_USD,
            "hard_ttl_seconds": HARD_TTL_SECONDS,
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
                "uri": (
                    f"{source_uri}/adp009d_840313_runtime_readiness.v1.json"
                ),
                "digest": _file_digest(
                    repo / MANIFEST_RELATIVE_ROOT / "adp009d_840313_runtime_readiness.v1.json"
                ),
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
        "profile_path": str(profile_path),
        "live_execution_enabled": True,
        "claim_ceiling": "development_only",
        "provider_mutation_performed": False,
    }
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
