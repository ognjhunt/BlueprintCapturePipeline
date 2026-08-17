#!/usr/bin/env python3
"""Freeze one scene-bound SimReady probe from paired-native terminal evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from blueprint_pipeline.paired_native_simready_transition import (
    PairedNativeSimReadyTransitionError,
    materialize_paired_native_simready_probe,
)
from blueprint_pipeline.articulated_native_probe import (
    COMMANDED_ARTICULATION_MODE,
    LOCKED_HINGE_RIGID_MODE,
)


def _reset_positions(values: Sequence[str]) -> dict[str, float]:
    resets: dict[str, float] = {}
    for value in values:
        path, separator, raw = value.rpartition("=")
        if not separator or not path.startswith("/"):
            raise ValueError("simready_reset_joint_position_invalid")
        resets[path] = float(raw)
    return resets


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--candidate-usd", required=True)
    parser.add_argument("--paired-bundle-receipt", required=True)
    parser.add_argument("--paired-request", required=True)
    parser.add_argument("--paired-terminal-result", required=True)
    parser.add_argument("--paired-runtime-result", required=True)
    parser.add_argument("--paired-candidate-probe", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument(
        "--validation-mode",
        choices=(COMMANDED_ARTICULATION_MODE, LOCKED_HINGE_RIGID_MODE),
        default=COMMANDED_ARTICULATION_MODE,
    )
    parser.add_argument("--task-joint-prim-path", default="")
    parser.add_argument("--locked-joint-prim-path", action="append", default=[])
    parser.add_argument(
        "--commanded-sweep-degrees", type=float, nargs="*", default=[]
    )
    parser.add_argument(
        "--reset-joint-position",
        action="append",
        default=[],
        metavar="/PRIM/PATH=RADIANS",
    )
    parser.add_argument(
        "--locked-joint-motion-tolerance-rad", type=float, required=True
    )
    parser.add_argument("--settle-samples", type=int, required=True)
    parser.add_argument("--control-frequency-hz", type=float, required=True)
    parser.add_argument("--probe-drive-stiffness", type=float, default=0.0)
    parser.add_argument("--probe-drive-damping", type=float, default=0.0)
    parser.add_argument("--probe-drive-max-force", type=float, default=0.0)
    parser.add_argument("--fixed-step-seconds", type=float, default=1.0 / 120.0)
    args = parser.parse_args(argv)
    try:
        receipt = materialize_paired_native_simready_probe(
            scene_id=args.scene_id,
            task_id=args.task_id,
            asset_id=args.asset_id,
            candidate_usd_path=args.candidate_usd,
            paired_bundle_receipt_path=args.paired_bundle_receipt,
            paired_request_path=args.paired_request,
            paired_terminal_result_path=args.paired_terminal_result,
            paired_runtime_result_path=args.paired_runtime_result,
            paired_candidate_probe_path=args.paired_candidate_probe,
            destination=args.destination,
            task_joint_prim_path=args.task_joint_prim_path,
            locked_joint_prim_paths=args.locked_joint_prim_path,
            commanded_sweep_degrees=args.commanded_sweep_degrees,
            reset_joint_positions_rad=_reset_positions(args.reset_joint_position),
            locked_joint_motion_tolerance_rad=(
                args.locked_joint_motion_tolerance_rad
            ),
            settle_samples=args.settle_samples,
            control_frequency_hz=args.control_frequency_hz,
            validation_mode=args.validation_mode,
            probe_drive_stiffness=args.probe_drive_stiffness,
            probe_drive_damping=args.probe_drive_damping,
            probe_drive_max_force=args.probe_drive_max_force,
            fixed_step_seconds=args.fixed_step_seconds,
        )
    except (OSError, ValueError, PairedNativeSimReadyTransitionError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [str(exc)],
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": "materialized",
                "scene_id": receipt["scene_id"],
                "candidate_usd_sha256": receipt["candidate_usd_sha256"],
                "predecessor_binding_digest": receipt[
                    "paired_native_predecessor"
                ]["binding_digest"],
                "receipt_digest": receipt["receipt_digest"],
                "spec_path": str(Path(receipt["spec_path"])),
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
