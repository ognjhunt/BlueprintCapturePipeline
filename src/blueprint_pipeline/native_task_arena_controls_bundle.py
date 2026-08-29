"""Freeze the task-neutral native Arena controls provider bundle."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .native_task_arena_bundle import (
    build_native_task_arena_bundle,
    digest_pinned_container_image,
)
from .native_task_arena_execution_contract import CONTROLS_RUNTIME_MODULE_NAMES
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE
from .native_task_control_plan import materialize_native_task_control_plan


PROBE_KIND = "native-task-arena-controls"
PROVIDER_BUNDLE_KIND = "native_task_arena"
RESULT_SCHEMA_VERSION = "native_task_arena_control_result.v1"
RESULT_FILENAME = "native_task_arena_control_result.v1.json"
DOWNSTREAM_DIAGNOSTIC_REQUEST_FILENAME = (
    "adp_task_synthetic_post_phase5_downstream_diagnostic_request.v1.json"
)
CONTROL_EXECUTION_SPEC_FILENAME = "adp_task_control_execution_spec.v1.json"
ZERO_ACTION_RESULT_FILENAME = "native_task_arena_zero_action_result.v1.json"
CONTROL_PAIR = "control_pair"
ZERO_ACTION_NEGATIVE = "zero_action_negative"
SCRIPTED_POSITIVE = "deterministic_scripted_positive"
CONTROL_SELECTIONS = frozenset(
    {CONTROL_PAIR, ZERO_ACTION_NEGATIVE, SCRIPTED_POSITIVE}
)

def controls_runtime_sources() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    return tuple(package / name for name in CONTROLS_RUNTIME_MODULE_NAMES)


def _read_mapping(path: Path, *, error: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(error) from exc
    if not isinstance(value, dict):
        raise ValueError(error)
    return value


def _validated_zero_action_result(
    path: str | Path,
    *,
    scene_plan: Mapping[str, Any],
    construction: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError("native_task_controls_zero_action_result_invalid")
    source = unresolved.resolve()
    value = _read_mapping(
        source, error="native_task_controls_zero_action_result_invalid"
    )
    episode = value.get("control_episode")
    visual = episode.get("visual_evidence") if isinstance(episode, Mapping) else None
    if (
        value.get("schema_version") != RESULT_SCHEMA_VERSION
        or value.get("status") != "completed"
        or value.get("blockers") != []
        or value.get("control_selection") != ZERO_ACTION_NEGATIVE
        or value.get("controls_qualified") is not False
        or value.get("scene_plan_digest") != scene_plan.get("plan_digest")
        or value.get("construction_result_digest")
        != construction.get("result_digest")
        or not isinstance(episode, Mapping)
        or episode.get("schema_version") != "adp_task_control_episode.v1"
        or episode.get("control_id") != ZERO_ACTION_NEGATIVE
        or episode.get("control_passed") is not True
        or episode.get("observed_outcome") != "never_moved"
        or episode.get("grader_authority")
        != "deterministic_simulator_state"
        or episode.get("candidate_policy_queried") is not False
        or not isinstance(visual, Mapping)
        or visual.get("status") != "complete"
        or not isinstance(episode.get("media_artifacts"), list)
        or not episode["media_artifacts"]
        or episode.get("receipt_digest")
        != canonical_digest(episode, digest_field="receipt_digest")
        or value.get("result_digest")
        != canonical_digest(value, digest_field="result_digest")
    ):
        raise ValueError("native_task_controls_zero_action_result_invalid")
    return source, value


def build_native_task_arena_controls_bundle(
    *,
    job_dir: str | Path,
    packet_dir: str | Path,
    construction_result_path: str | Path,
    runtime_source_packet_receipt: str | Path,
    implementation_commit: str,
    container_image: str = NATIVE_TASK_ARENA_IMAGE,
    generated_at: str | None = None,
    control_selection: str = CONTROL_PAIR,
    zero_action_result_path: str | Path | None = None,
    enable_synthetic_post_phase5_downstream_diagnostic: bool = False,
    bounded_orientation_reference_joint_positions_rad: Sequence[float]
    | None = None,
    allow_unqualified_construction_diagnostic: bool = False,
) -> dict[str, Any]:
    """Bind construction evidence to qualifying or diagnostic controls.

    The diagnostic option carries a blocked rigid construction forward only
    for downstream runtime measurement.  The frozen plan and execution spec
    both bind that nonqualification, and the worker must preserve it even if a
    later episode happens to meet its task scorer.
    """

    packet = Path(packet_dir).expanduser().resolve()
    scene_plan = _read_mapping(
        packet / "native_task_arena_scene_plan.v1.json",
        error="native_task_controls_scene_plan_invalid",
    )
    construction_path = Path(construction_result_path).expanduser().resolve()
    construction = _read_mapping(
        construction_path,
        error="native_task_controls_construction_result_invalid",
    )
    control_plan = materialize_native_task_control_plan(
        scene_plan=scene_plan,
        construction_result=construction,
        allow_unqualified_construction_diagnostic=(
            allow_unqualified_construction_diagnostic
        ),
    )
    if allow_unqualified_construction_diagnostic and (
        control_selection != CONTROL_PAIR
        or scene_plan.get("task_kind") != "rigid_pick_place"
    ):
        raise ValueError("native_task_controls_diagnostic_mode_invalid")
    if control_selection not in CONTROL_SELECTIONS:
        raise ValueError("native_task_controls_selection_invalid")
    if control_selection != CONTROL_PAIR and not (
        scene_plan.get("task_kind") == "rigid_pick_place"
        and (scene_plan.get("task_spec") or {}).get("schema_version")
        == "adp_task_spec.v2"
    ):
        raise ValueError("native_task_controls_single_selection_unsupported")
    if (
        control_selection != CONTROL_PAIR
        and enable_synthetic_post_phase5_downstream_diagnostic
    ):
        raise ValueError("native_task_controls_selection_diagnostic_conflict")
    zero_action_result: dict[str, Any] | None = None
    zero_action_path: Path | None = None
    if control_selection == SCRIPTED_POSITIVE:
        if zero_action_result_path is None:
            raise ValueError("native_task_controls_zero_action_result_required")
        zero_action_path, zero_action_result = _validated_zero_action_result(
            zero_action_result_path,
            scene_plan=scene_plan,
            construction=construction,
        )
    elif zero_action_result_path is not None:
        raise ValueError("native_task_controls_zero_action_result_unexpected")
    if bounded_orientation_reference_joint_positions_rad is not None:
        try:
            reference = [
                float(value)
                for value in bounded_orientation_reference_joint_positions_rad
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "native_task_controls_bounded_orientation_reference_invalid"
            ) from exc
        if len(reference) != 7 or not all(
            math.isfinite(value) for value in reference
        ):
            raise ValueError(
                "native_task_controls_bounded_orientation_reference_invalid"
            )
        control_plan["bounded_orientation_reference_joint_positions_rad"] = (
            reference
        )
        control_plan["plan_digest"] = canonical_digest(
            control_plan, digest_field="plan_digest"
        )
    with tempfile.TemporaryDirectory(prefix="blueprint-native-task-controls-") as raw:
        frozen_plan = Path(raw) / "adp_task_control_plan.v1.json"
        frozen_plan.write_text(
            json.dumps(control_plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bound_runtime_inputs: dict[str, Path] = {
            "native_task_arena_construction_result.v1.json": construction_path,
            "adp_task_control_plan.v1.json": frozen_plan,
        }
        execution_spec: dict[str, Any] = {
            "schema_version": "adp_task_control_execution_spec.v1",
            "control_selection": control_selection,
            "task_kind": scene_plan["task_kind"],
            "scene_plan_digest": scene_plan["plan_digest"],
            "construction_result_digest": construction["result_digest"],
            "control_plan_digest": control_plan["plan_digest"],
            "candidate_policy_queried": False,
            "prior_zero_action_result_digest": (
                zero_action_result["result_digest"]
                if zero_action_result is not None
                else None
            ),
            "execution_spec_digest": "",
        }
        if allow_unqualified_construction_diagnostic:
            execution_spec.update(
                {
                    "diagnostic_only": True,
                    "qualification_effect": "none",
                    "upstream_construction_blockers": list(
                        control_plan["upstream_construction_blockers"]
                    ),
                }
            )
        execution_spec["execution_spec_digest"] = canonical_digest(
            execution_spec, digest_field="execution_spec_digest"
        )
        execution_spec_path = Path(raw) / CONTROL_EXECUTION_SPEC_FILENAME
        execution_spec_path.write_text(
            json.dumps(execution_spec, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bound_runtime_inputs[CONTROL_EXECUTION_SPEC_FILENAME] = (
            execution_spec_path
        )
        if zero_action_path is not None:
            bound_runtime_inputs[ZERO_ACTION_RESULT_FILENAME] = zero_action_path
        if enable_synthetic_post_phase5_downstream_diagnostic:
            request: dict[str, Any] = {
                "schema_version": (
                    "adp_task_synthetic_post_phase5_downstream_"
                    "diagnostic_request.v1"
                ),
                "enabled": True,
                "development_only": True,
                "qualification_effect": "none",
                "request_digest": "",
            }
            request["request_digest"] = canonical_digest(
                request, digest_field="request_digest"
            )
            request_path = Path(raw) / DOWNSTREAM_DIAGNOSTIC_REQUEST_FILENAME
            request_path.write_text(
                json.dumps(request, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            bound_runtime_inputs[DOWNSTREAM_DIAGNOSTIC_REQUEST_FILENAME] = (
                request_path
            )
        return build_native_task_arena_bundle(
            job_dir=job_dir,
            packet_dir=packet,
            runtime_source_packet_receipt=runtime_source_packet_receipt,
            worker_source=(
                Path(__file__).resolve().parent
                / "native_task_arena_controls_worker.py"
            ),
            runtime_module_sources=controls_runtime_sources(),
            implementation_commit=implementation_commit,
            execution_mode="controls",
            expected_output_filename=RESULT_FILENAME,
            container_image=container_image,
            bound_runtime_inputs=bound_runtime_inputs,
            generated_at=generated_at,
        )


def load_verified_native_task_arena_controls_bundle(
    receipt_path: str | Path,
    *,
    expected_implementation_commit: str,
    expected_packet_receipt_digest: str | None = None,
    expected_runtime_source_packet_digest: str | None = None,
) -> dict[str, Any]:
    """Reverify the immutable controls bundle without rebuilding its bytes."""

    path = Path(receipt_path).expanduser().resolve()
    receipt = _read_mapping(
        path, error="native_task_arena_controls_bundle_receipt_invalid"
    )
    bundle_path = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    digest = hashlib.sha256()
    try:
        with bundle_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError("native_task_arena_controls_bundle_bytes_missing") from exc
    manifest = {
        key: value
        for key, value in receipt.items()
        if key not in {"bundle_path", "bundle_size_bytes", "bundle_sha256"}
    }
    errors: list[str] = []
    input_names = {
        Path(str(row.get("relative_path") or "")).name
        for row in receipt.get("bound_runtime_inputs") or []
        if isinstance(row, dict)
    }
    if (
        receipt.get("schema_version") != "native_task_arena_provider_bundle.v1"
        or receipt.get("status") != "ready"
        or receipt.get("execution_mode") != "controls"
        or receipt.get("policy_candidate_id") is not None
        or receipt.get("candidate_policy_queried") is not False
        or receipt.get("expected_output_filename") != RESULT_FILENAME
        or input_names
        not in (
            {
                "native_task_arena_construction_result.v1.json",
                "adp_task_control_plan.v1.json",
                CONTROL_EXECUTION_SPEC_FILENAME,
            },
            {
                "native_task_arena_construction_result.v1.json",
                "adp_task_control_plan.v1.json",
                CONTROL_EXECUTION_SPEC_FILENAME,
                DOWNSTREAM_DIAGNOSTIC_REQUEST_FILENAME,
            },
            {
                "native_task_arena_construction_result.v1.json",
                "adp_task_control_plan.v1.json",
                CONTROL_EXECUTION_SPEC_FILENAME,
                ZERO_ACTION_RESULT_FILENAME,
            },
        )
    ):
        errors.append("native_task_arena_controls_bundle_contract_invalid")
    if receipt.get("implementation_commit") != expected_implementation_commit:
        errors.append("native_task_arena_controls_bundle_commit_mismatch")
    if not digest_pinned_container_image(receipt.get("container_image")):
        errors.append("native_task_arena_controls_bundle_image_mismatch")
    if expected_packet_receipt_digest and (
        receipt.get("packet_receipt_digest") != expected_packet_receipt_digest
    ):
        errors.append("native_task_arena_controls_bundle_packet_mismatch")
    source_packet = receipt.get("runtime_source_packet") or {}
    if expected_runtime_source_packet_digest and (
        source_packet.get("receipt_digest")
        != expected_runtime_source_packet_digest
    ):
        errors.append("native_task_arena_controls_bundle_sources_mismatch")
    if receipt.get("input_digest") != canonical_digest(
        manifest, digest_field="input_digest"
    ):
        errors.append("native_task_arena_controls_bundle_input_digest_invalid")
    if (
        receipt.get("bundle_size_bytes") != bundle_path.stat().st_size
        or receipt.get("bundle_sha256") != "sha256:" + digest.hexdigest()
    ):
        errors.append("native_task_arena_controls_bundle_bytes_identity_mismatch")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    return receipt


__all__ = [
    "CONTROLS_RUNTIME_MODULE_NAMES",
    "CONTROL_EXECUTION_SPEC_FILENAME",
    "CONTROL_PAIR",
    "CONTROL_SELECTIONS",
    "DOWNSTREAM_DIAGNOSTIC_REQUEST_FILENAME",
    "PROBE_KIND",
    "PROVIDER_BUNDLE_KIND",
    "RESULT_FILENAME",
    "RESULT_SCHEMA_VERSION",
    "SCRIPTED_POSITIVE",
    "ZERO_ACTION_NEGATIVE",
    "ZERO_ACTION_RESULT_FILENAME",
    "build_native_task_arena_controls_bundle",
    "controls_runtime_sources",
    "load_verified_native_task_arena_controls_bundle",
]

def main(argv: list[str] | None = None) -> int:
    """Build the sealed native Arena controls packet.

    The Arena family is a chain -- construction, then controls, then policy --
    and each link consumes the previous link's result. None of the three could
    be produced except by calling a Python function, so the whole chain was
    unreachable from any production path.

    The allocator refuses a bundle whose commit is not the one the control
    plane is running, and every deploy moves that commit, so a bundle that can
    only be built by hand is launchable at most once.

    Performs no provider mutation and rents nothing.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Build the sealed native Arena controls packet.")
    parser.add_argument("--job-dir", dest="job_dir", required=True)
    parser.add_argument("--packet-dir", dest="packet_dir", required=True)
    parser.add_argument("--construction-result", dest="construction_result_path", required=True)
    parser.add_argument("--runtime-source-packet-receipt", dest="runtime_source_packet_receipt", required=True)
    parser.add_argument("--implementation-commit", dest="implementation_commit", required=True)
    parser.add_argument("--container-image", default=NATIVE_TASK_ARENA_IMAGE)
    parser.add_argument("--generated-at", dest="generated_at")
    parser.add_argument(
        "--control-selection",
        choices=sorted(CONTROL_SELECTIONS),
        default=CONTROL_PAIR,
    )
    parser.add_argument("--zero-action-result")
    parser.add_argument(
        "--enable-synthetic-post-phase5-downstream-diagnostic",
        action="store_true",
        help=(
            "Seal a development-only, non-qualifying phases 6-11 diagnostic "
            "request into this bundle"
        ),
    )
    parser.add_argument(
        "--allow-unqualified-construction-diagnostic",
        action="store_true",
        help=(
            "Carry one digest-bound blocked rigid construction into an "
            "explicitly nonqualifying controls diagnostic"
        ),
    )
    parser.add_argument(
        "--bounded-orientation-reference-joint-positions-rad",
        nargs=7,
        type=float,
        help=(
            "Seal one previously measured seven-joint posture into the control "
            "plan as a reference seed for bounded orientation search. The "
            "runtime re-solves every candidate and never directly replays this "
            "posture."
        ),
    )
    args = parser.parse_args(argv)

    try:
        receipt = build_native_task_arena_controls_bundle(
            job_dir=args.job_dir,
            packet_dir=args.packet_dir,
            construction_result_path=args.construction_result_path,
            runtime_source_packet_receipt=args.runtime_source_packet_receipt,
            implementation_commit=args.implementation_commit,
            container_image=args.container_image,
            **({"generated_at": args.generated_at} if args.generated_at else {}),
            control_selection=args.control_selection,
            zero_action_result_path=args.zero_action_result,
            enable_synthetic_post_phase5_downstream_diagnostic=(
                args.enable_synthetic_post_phase5_downstream_diagnostic
            ),
            allow_unqualified_construction_diagnostic=(
                args.allow_unqualified_construction_diagnostic
            ),
            bounded_orientation_reference_joint_positions_rad=(
                args.bounded_orientation_reference_joint_positions_rad
            ),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") in {"ready", "sealed"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
