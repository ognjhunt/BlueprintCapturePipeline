#!/usr/bin/env python3
"""Rebuild the articulated scene spec the Arena payload consumes.

The first version of this spec was authored inline in a session and never
committed, which meant a regenerated twin, a corrected control plan, or a new
asset binding all had to be re-applied by hand into a file nobody owned. This
is that spec as code: the same inputs, the same planners, and a digest so two
builds can be compared.

Nothing here contacts a runtime. It resolves spawn types, the joint binding,
and the scripted trajectory, and it records the alias names a provider bundle
will rename the assets to - all of which are decisions the payload must not be
left to guess on paid hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

try:  # flat provider bundle
    from articulated_runtime_composition import plan_articulated_runtime_composition
except ModuleNotFoundError:  # repository checkout
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from blueprint_pipeline.articulated_runtime_composition import (
        plan_articulated_runtime_composition,
    )


SPEC_SCHEMA_VERSION = "adp009d_articulated_scene_spec.v1"
# What asset bindings rename each role to inside a provider bundle. These are
# lane constants, not properties of this scene: the rigid can lane named them,
# and every payload shipped through the same transport inherits them.
BUNDLE_ASSET_ALIASES = {
    "task_object": ["approved_can.usda"],
    "scene_collision": ["sage_collision.usd"],
}


def _canonical_digest(value: dict[str, Any], *, field: str) -> str:
    payload = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_articulated_scene_spec(
    *,
    task_spec: dict[str, Any],
    control_plan: dict[str, Any],
    robot_base: dict[str, Any],
    articulated_joints: Sequence[dict[str, Any]],
    twin_usd_filename: str,
    scene_collision_filename: str,
    seed: int,
    episode_length_s: float,
    gripper_open_command: float,
    asset_filename_aliases: dict[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Assemble the spec from planners rather than by hand."""

    # The planner needs the joint inventory; the spec persists only the
    # narrower runtime task_spec, so the two are recombined here rather than
    # the inventory being re-derived from the asset (which would let a
    # renamed joint pass silently).
    planner_task_spec = dict(task_spec)
    planner_task_spec["articulated_joints"] = [dict(row) for row in articulated_joints]
    composition = plan_articulated_runtime_composition(
        task_spec=planner_task_spec,
        twin_usd_filename=twin_usd_filename,
        scene_collision_filename=scene_collision_filename,
        asset_filename_aliases=(
            BUNDLE_ASSET_ALIASES
            if asset_filename_aliases is None
            else asset_filename_aliases
        ),
    )
    spec: dict[str, Any] = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "composition": composition,
        "control_plan": control_plan,
        "robot_base": robot_base,
        "task_spec": task_spec,
        "seed": int(seed),
        "episode_length_s": float(episode_length_s),
        "gripper_open_command": float(gripper_open_command),
        "spec_digest": "",
    }
    spec["spec_digest"] = _canonical_digest(spec, field="spec_digest")
    return spec


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-spec", required=True, help="spec to rebuild from")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    source = json.loads(Path(arguments.source_spec).read_text(encoding="utf-8"))
    objects = {
        str(row.get("semantic_role")): row
        for row in (source.get("composition") or {}).get("objects") or []
    }
    binding = (source.get("composition") or {}).get("task_sample_binding") or {}
    prim_paths = binding.get("joint_prim_paths") or {}
    roles = binding.get("joint_roles") or {}
    spec = build_articulated_scene_spec(
        articulated_joints=[
            {
                "joint_id": joint_id,
                "joint_prim_path": prim_paths.get(joint_id),
                "role": roles.get(joint_id),
            }
            for joint_id in binding.get("joint_ids") or []
        ],
        task_spec=source["task_spec"],
        control_plan=source["control_plan"],
        robot_base=source["robot_base"],
        twin_usd_filename=str(objects["task_object"]["usd_filename"]),
        scene_collision_filename=str(objects["scene_collision"]["usd_filename"]),
        seed=int(source.get("seed") or 0),
        episode_length_s=float(source.get("episode_length_s") or 0.0),
        gripper_open_command=float(source.get("gripper_open_command") or 0.0),
    )
    output = Path(arguments.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(output), "spec_digest": spec["spec_digest"]}))
    return 0


__all__ = [
    "BUNDLE_ASSET_ALIASES",
    "SPEC_SCHEMA_VERSION",
    "build_articulated_scene_spec",
]


if __name__ == "__main__":
    raise SystemExit(main())
