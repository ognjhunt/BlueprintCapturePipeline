"""Build a canonical learned-policy diagnostic without qualifying the policy.

This seam is deliberately separate from :mod:`native_task_arena_policy_bundle`.
It permits one frozen candidate to act from the canonical scene reset while the
deterministic positive control is still unqualified, but it cannot score, rank,
qualify, or admit that candidate.  Construction and the zero-action negative
remain mandatory and digest-bound.
"""

from __future__ import annotations

import json
import tempfile
import argparse
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adp009d_policy_episode import maximum_policy_queries_for_task_spec
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_bundle import (
    POLICY_RUNTIME_ROOT_MODULE_NAMES,
    build_native_task_arena_bundle,
)
from .native_task_arena_controls_bundle import controls_runtime_sources
from .native_task_arena_execution_contract import POLICY_EXTRA_RUNTIME_MODULE_NAMES
from .native_task_arena_policy_bundle import (
    EXECUTION_SPEC_SCHEMA_VERSION,
    _candidate_runtime_binding,
    _read,
    _sha256,
    _verified_openpi_checkpoint_inventory,
    validate_native_task_policy_execution_spec,
)
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .native_task_runtime_contract import FROZEN_CANDIDATES


PROBE_KIND = "native-task-arena-policy-diagnostic"
RESULT_FILENAME = "native_task_arena_policy_diagnostic_result.v1.json"
RESULT_SCHEMA_VERSION = "native_task_arena_policy_diagnostic_result.v1"
DIAGNOSTIC_EXECUTION_AUTHORITY = (
    "development_only_unqualified_controls_canonical_diagnostic"
)
DIAGNOSTIC_CLAIM_CEILING = (
    "development_only_policy_motion_diagnostic_not_scoring_not_ranking_"
    "not_qualification"
)


def _zero_action_negative_passed(pair: Mapping[str, Any]) -> bool:
    controls = pair.get("controls")
    return isinstance(controls, list) and any(
        isinstance(row, Mapping)
        and row.get("control_id") == "zero_action_negative"
        and row.get("control_passed") is True
        and row.get("observed_outcome") == "never_moved"
        and str(row.get("receipt_digest") or "").startswith("sha256:")
        for row in controls
    )


def validate_policy_diagnostic_execution_spec(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(value))
    errors: list[str] = []
    if payload.get("execution_authority") != DIAGNOSTIC_EXECUTION_AUTHORITY:
        errors.append("native_task_policy_diagnostic_authority_invalid")
    expected = {
        "claim_ceiling": DIAGNOSTIC_CLAIM_CEILING,
        "initial_state": "canonical_scene_reset",
        "controls_qualified": False,
        "zero_action_negative_bound_separately": True,
        "scientific_scoring_permitted": False,
        "ranking_permitted": False,
        "qualification_permitted": False,
    }
    for field, expected_value in expected.items():
        if payload.get(field) != expected_value:
            errors.append(f"native_task_policy_diagnostic_{field}_invalid")
    if payload.get("execution_spec_digest") != canonical_digest(
        payload, digest_field="execution_spec_digest"
    ):
        errors.append("native_task_policy_diagnostic_spec_digest_invalid")

    # Reuse every shape, candidate, endpoint, checkpoint, and query-budget
    # validation from the qualified spec after removing diagnostic authority.
    qualified_shape = dict(payload)
    for field in (*expected, "execution_authority", "execution_spec_digest"):
        qualified_shape.pop(field, None)
    qualified_shape["execution_spec_digest"] = canonical_digest(
        qualified_shape, digest_field="execution_spec_digest"
    )
    try:
        validate_native_task_policy_execution_spec(qualified_shape)
    except ValueError as exc:
        errors.append(f"native_task_policy_diagnostic_common_spec_invalid:{exc}")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    return payload


def build_policy_diagnostic_execution_spec(
    *,
    candidate_id: str,
    scene_plan_path: str | Path,
    construction_result_path: str | Path,
    control_result_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    scene = _read(scene_plan_path, error="native_task_policy_scene_plan_invalid")
    construction = _read(
        construction_result_path,
        error="native_task_policy_construction_result_invalid",
    )
    controls = _read(
        control_result_path, error="native_task_policy_control_result_invalid"
    )
    task_spec = scene.get("task_spec") or {}
    pair = controls.get("control_pair") or {}
    scene_digest = scene.get("plan_digest")
    construction_digest = construction.get("result_digest")
    control_digest = controls.get("result_digest")
    pair_digest = pair.get("pair_digest") if isinstance(pair, Mapping) else None
    cell_id = (scene.get("scenario") or {}).get("cell_id")
    errors: list[str] = []
    if (
        scene.get("schema_version") != "native_task_arena_scene_plan.v1"
        or scene_digest != canonical_digest(scene, digest_field="plan_digest")
        or not isinstance(task_spec, Mapping)
        or not str(scene.get("task_id") or "")
        or not str(cell_id or "")
        or not str(task_spec.get("prompt") or "")
    ):
        errors.append("native_task_policy_scene_plan_invalid")
    if (
        construction.get("schema_version")
        != "native_task_arena_construction_result.v1"
        or construction.get("status") != "completed"
        or construction.get("construction_gate_qualified") is not True
        or construction.get("scene_plan_digest") != scene_digest
        or construction_digest
        != canonical_digest(construction, digest_field="result_digest")
    ):
        errors.append("native_task_policy_construction_not_qualified")
    if (
        controls.get("schema_version") != "native_task_arena_control_result.v1"
        or controls.get("status") not in {"blocked", "completed"}
        or controls.get("controls_qualified") is not False
        or controls.get("candidate_policy_queried") is not False
        or controls.get("scene_plan_digest") != scene_digest
        or controls.get("construction_result_digest") != construction_digest
        or control_digest != canonical_digest(controls, digest_field="result_digest")
        or not isinstance(pair, Mapping)
        or pair.get("schema_version") != "adp_task_control_pair.v1"
        or pair.get("cell_id") != cell_id
        or pair.get("task_spec_digest") != canonical_digest(task_spec)
        or pair.get("cell_admitted_for_policy_execution") is not False
        or pair.get("candidate_policy_queried") is not False
        or pair_digest != canonical_digest(pair, digest_field="pair_digest")
        or not _zero_action_negative_passed(pair)
    ):
        errors.append("native_task_policy_diagnostic_controls_invalid")
    if candidate_id not in FROZEN_CANDIDATES:
        errors.append("native_task_policy_candidate_invalid")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))

    policy, endpoint, identity = _candidate_runtime_binding(candidate_id)
    request = {
        "schema_version": EXECUTION_SPEC_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "task_id": scene["task_id"],
        "cell_id": cell_id,
        "prompt": task_spec["prompt"],
        "scene_plan_digest": scene_digest,
        "construction_result_digest": construction_digest,
        "control_result_digest": control_digest,
        "control_pair_digest": pair_digest,
        "policy_endpoint": endpoint,
        "policy_spec": asdict(policy),
        "policy_identity_receipt": identity,
        "max_policy_queries": maximum_policy_queries_for_task_spec(
            task_spec, open_loop_horizon=policy.open_loop_horizon
        ),
        "open_loop_horizon": policy.open_loop_horizon,
        "overview_camera_policy_input": False,
        "policy_may_grade_itself": False,
        "execution_authority": DIAGNOSTIC_EXECUTION_AUTHORITY,
        "claim_ceiling": DIAGNOSTIC_CLAIM_CEILING,
        "initial_state": "canonical_scene_reset",
        "controls_qualified": False,
        "zero_action_negative_bound_separately": True,
        "scientific_scoring_permitted": False,
        "ranking_permitted": False,
        "qualification_permitted": False,
    }
    request["execution_spec_digest"] = canonical_digest(
        request, digest_field="execution_spec_digest"
    )
    validated = validate_policy_diagnostic_execution_spec(request)
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("native_task_policy_execution_spec_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validated


def build_native_task_arena_policy_diagnostic_bundle(
    *,
    job_dir: str | Path,
    packet_dir: str | Path,
    construction_result_path: str | Path,
    control_result_path: str | Path,
    policy_execution_spec: Mapping[str, Any],
    runtime_source_packet_receipt: str | Path,
    implementation_commit: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    packet = Path(packet_dir).expanduser().resolve()
    scene = _read(
        packet / "native_task_arena_scene_plan.v1.json",
        error="native_task_policy_scene_plan_invalid",
    )
    construction_path = Path(construction_result_path).expanduser().resolve()
    controls_path = Path(control_result_path).expanduser().resolve()
    construction = _read(
        construction_path, error="native_task_policy_construction_result_invalid"
    )
    controls = _read(controls_path, error="native_task_policy_control_result_invalid")
    spec = validate_policy_diagnostic_execution_spec(policy_execution_spec)
    pair = controls.get("control_pair") or {}
    errors: list[str] = []
    if (
        construction.get("result_digest") != spec.get("construction_result_digest")
        or controls.get("result_digest") != spec.get("control_result_digest")
        or pair.get("pair_digest") != spec.get("control_pair_digest")
        or scene.get("plan_digest") != spec.get("scene_plan_digest")
        or scene.get("task_id") != spec.get("task_id")
        or (scene.get("scenario") or {}).get("cell_id") != spec.get("cell_id")
        or not _zero_action_negative_passed(pair)
    ):
        errors.append("native_task_policy_diagnostic_binding_invalid")
    if errors:
        raise ValueError(";".join(errors))

    package = Path(__file__).resolve().parent
    sources = set(controls_runtime_sources())
    sources.update(package / name for name in POLICY_EXTRA_RUNTIME_MODULE_NAMES)
    with tempfile.TemporaryDirectory(
        prefix="blueprint-native-task-policy-diagnostic-"
    ) as raw:
        execution_path = Path(raw) / (
            "native_task_arena_policy_execution_spec.v1.json"
        )
        execution_path.write_text(
            json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        inputs: dict[str, Path] = {
            "native_task_arena_construction_result.v1.json": construction_path,
            "native_task_arena_control_result.v1.json": controls_path,
            execution_path.name: execution_path,
        }
        if spec["candidate_id"] == "pi05_droid":
            inputs["openpi_polaris_checkpoint_inventory.json"] = (
                _verified_openpi_checkpoint_inventory(spec["policy_spec"])
            )
        receipt = build_native_task_arena_bundle(
            job_dir=job_dir,
            packet_dir=packet,
            runtime_source_packet_receipt=runtime_source_packet_receipt,
            worker_source=package / "native_task_arena_policy_worker.py",
            runtime_module_sources=sorted(sources),
            implementation_commit=implementation_commit,
            execution_mode="policy_diagnostic",
            policy_candidate_id=spec["candidate_id"],
            expected_output_filename=RESULT_FILENAME,
            container_image=NATIVE_TASK_ARENA_IMAGE,
            bound_runtime_inputs=inputs,
            generated_at=generated_at,
        )
    return receipt


def load_verified_native_task_arena_policy_diagnostic_bundle(
    receipt_path: str | Path,
    *,
    expected_implementation_commit: str,
    expected_packet_receipt_digest: str | None = None,
    expected_runtime_source_packet_digest: str | None = None,
) -> dict[str, Any]:
    receipt = _read(receipt_path, error="native_task_policy_diagnostic_bundle_invalid")
    bundle = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    errors: list[str] = []
    candidate_id = str(receipt.get("policy_candidate_id") or "")
    expected_inputs = {
        "native_task_arena_construction_result.v1.json",
        "native_task_arena_control_result.v1.json",
        "native_task_arena_policy_execution_spec.v1.json",
    }
    if candidate_id == "pi05_droid":
        expected_inputs.add("openpi_polaris_checkpoint_inventory.json")
    input_names = {
        Path(str(row.get("relative_path") or "")).name
        for row in receipt.get("bound_runtime_inputs") or []
        if isinstance(row, Mapping)
    }
    runtime_root_names = {
        str(row.get("relative_path") or "")
        for row in receipt.get("runtime_root_modules") or []
        if isinstance(row, Mapping)
    }
    if (
        receipt.get("schema_version") != "native_task_arena_provider_bundle.v1"
        or receipt.get("status") != "ready"
        or receipt.get("execution_mode") != "policy_diagnostic"
        or receipt.get("policy_candidate_id") not in FROZEN_CANDIDATES
        or receipt.get("candidate_policy_queried") is not False
        or receipt.get("expected_output_filename") != RESULT_FILENAME
        or receipt.get("implementation_commit") != expected_implementation_commit
        or receipt.get("container_image") != NATIVE_TASK_ARENA_IMAGE
        or input_names != expected_inputs
        or runtime_root_names != set(POLICY_RUNTIME_ROOT_MODULE_NAMES)
        or receipt.get("policy_provisioning_script")
        != f"adp009d_policy_provisioning.{candidate_id}.sh"
        or (receipt.get("policy_provisioning") or {}).get("relative_path")
        != f"adp009d_policy_provisioning.{candidate_id}.sh"
    ):
        errors.append("native_task_policy_diagnostic_bundle_contract_invalid")
    if expected_packet_receipt_digest and receipt.get(
        "packet_receipt_digest"
    ) != expected_packet_receipt_digest:
        errors.append("native_task_policy_diagnostic_packet_mismatch")
    source = receipt.get("runtime_source_packet") or {}
    if expected_runtime_source_packet_digest and source.get(
        "receipt_digest"
    ) != expected_runtime_source_packet_digest:
        errors.append("native_task_policy_diagnostic_sources_mismatch")
    manifest = {
        key: value
        for key, value in receipt.items()
        if key not in {"bundle_path", "bundle_size_bytes", "bundle_sha256"}
    }
    if receipt.get("input_digest") != canonical_digest(
        manifest, digest_field="input_digest"
    ):
        errors.append("native_task_policy_diagnostic_input_digest_invalid")
    if (
        not bundle.is_file()
        or receipt.get("bundle_size_bytes") != bundle.stat().st_size
        or receipt.get("bundle_sha256") != _sha256(bundle)
    ):
        errors.append("native_task_policy_diagnostic_bytes_identity_mismatch")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    return receipt


__all__ = [
    "DIAGNOSTIC_CLAIM_CEILING",
    "DIAGNOSTIC_EXECUTION_AUTHORITY",
    "PROBE_KIND",
    "RESULT_FILENAME",
    "RESULT_SCHEMA_VERSION",
    "build_native_task_arena_policy_diagnostic_bundle",
    "build_policy_diagnostic_execution_spec",
    "load_verified_native_task_arena_policy_diagnostic_bundle",
    "validate_policy_diagnostic_execution_spec",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--packet-dir", required=True)
    parser.add_argument("--construction-result", required=True)
    parser.add_argument("--control-result", required=True)
    parser.add_argument("--runtime-source-packet-receipt", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--policy-execution-spec", required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    try:
        spec = json.loads(
            Path(args.policy_execution_spec).read_text(encoding="utf-8")
        )
        receipt = build_native_task_arena_policy_diagnostic_bundle(
            job_dir=args.job_dir,
            packet_dir=args.packet_dir,
            construction_result_path=args.construction_result,
            control_result_path=args.control_result,
            runtime_source_packet_receipt=args.runtime_source_packet_receipt,
            implementation_commit=args.implementation_commit,
            policy_execution_spec=spec,
            **({"generated_at": args.generated_at} if args.generated_at else {}),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "ready_diagnostic",
                "bundle_sha256": receipt["bundle_sha256"],
                "candidate_id": receipt["policy_candidate_id"],
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
