"""Provider worker for one development-only vectorized Isaac Lab sweep."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .native_task_arena_runtime import build_native_task_arena_environment
from .native_task_isaaclab_control_sweep_runtime import (
    NativeIsaacLabControlSweepWaveRunner,
)
from .native_task_isaaclab_launch import (
    NATIVE_TASK_ARENA_DEVICE,
    launch_native_task_isaaclab,
)
from .task_evaluation_control_search_funnel import (
    validate_control_search_funnel_plan,
)
from .task_evaluation_isaaclab_control_sweep import (
    execute_isaaclab_control_sweep,
    validate_isaaclab_control_sweep_schedule,
)


def _mapping(path: str | Path, *, blocker: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(blocker) from exc
    if source.is_symlink() or not isinstance(value, Mapping):
        raise ValueError(blocker)
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_control_sweep_worker(
    *,
    plan_path: str | Path,
    schedule_path: str | Path,
    candidate_inventory_path: str | Path,
    scene_plan_path: str | Path,
    packet_root: str | Path,
    provisioning_receipt_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Execute one sealed sweep and close the SimulationApp on every return."""

    plan = validate_control_search_funnel_plan(
        _mapping(plan_path, blocker="control_search_worker_plan_invalid")
    )
    schedule = validate_isaaclab_control_sweep_schedule(
        _mapping(schedule_path, blocker="control_search_worker_schedule_invalid"),
        plan=plan,
    )
    inventory = _mapping(
        candidate_inventory_path,
        blocker="control_search_worker_inventory_invalid",
    )
    if inventory.get("inventory_digest") != schedule.get(
        "candidate_inventory_digest"
    ):
        raise ValueError("control_search_worker_inventory_invalid")
    scene_plan = _mapping(
        scene_plan_path, blocker="control_search_worker_scene_plan_invalid"
    )
    packet = Path(packet_root).expanduser().resolve()
    if packet.is_symlink() or not packet.is_dir():
        raise ValueError("control_search_worker_packet_root_invalid")
    simulation_app = None
    try:
        simulation_app, _launch_receipt = launch_native_task_isaaclab(
            provisioning_receipt_path,
            device=NATIVE_TASK_ARENA_DEVICE,
        )
        runner = NativeIsaacLabControlSweepWaveRunner(
            plan=plan,
            schedule=schedule,
            # The pinned Arena DROID action contract uses 1=open and 0=closed.
            # This remains development-only and full replay remeasures the live
            # physical-pad convention before any qualifying contact claim.
            gripper_open_command=1.0,
            gripper_closed_command=0.0,
        )
        result = execute_isaaclab_control_sweep(
            plan=plan,
            schedule=schedule,
            candidate_inventory=inventory,
            scene_plan=scene_plan,
            bundle_root=packet,
            wave_runner=runner,
            environment_builder=build_native_task_arena_environment,
        )
        _write_json(Path(output_path).expanduser().resolve(), result)
        return result
    finally:
        if simulation_app is not None:
            simulation_app.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--candidate-inventory", required=True)
    parser.add_argument("--scene-plan", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--provisioning-receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    run_control_sweep_worker(
        plan_path=args.plan,
        schedule_path=args.schedule,
        candidate_inventory_path=args.candidate_inventory,
        scene_plan_path=args.scene_plan,
        packet_root=args.packet_root,
        provisioning_receipt_path=args.provisioning_receipt,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_control_sweep_worker"]
