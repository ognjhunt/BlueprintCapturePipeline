"""Isaac scene files for a website scene kept as filmed, made on the controller with no spend.

A navigation job keeps its scene exactly as filmed: no object removed, edited or authored. So it
has no paid native construction and no configured scene bundle. The arena still needs the two
files a configured bundle would hold: a ParticleField appearance USD and a collision USD, Z up,
in metres. Both converters already exist and are what the pick-and-place handoff uses on the
controller:

- ``website_native_appearance.prepare_native_appearance`` (``particlefield_usd``);
- ``website_scene_runtime_inputs.normalize_runtime_collision`` (``normalize_completed_mesh``).

They read only a preparation's splat and collider digests and its coordinate frame, so this
module gives them that frame from the base scene itself. The frame is the declared Atlas scale
and the Y-up to Z-up turn the locomotion compiler uses, so the compiled start and goal land
where the scene files put the floor and shelves. Nothing is bought, measured or qualified here;
everything stays ``development_only``. Part of docs/G1_NAVIGATION_FROM_COMPILED_SPEC.md.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest

SCHEMA = "website_navigation_scene_assets.v1"
FILE = "navigation_scene_assets.json"


def navigation_frame(base_scene: Mapping[str, Any]) -> dict[str, Any]:
    """The preparation-shaped record both converters read: digests and the declared frame."""
    up = base_scene.get("up_axis")
    mpu = base_scene.get("meters_per_unit")
    if (up not in {"Y", "-Y", "Z"} or isinstance(mpu, bool) or not isinstance(mpu, (int, float))
            or not math.isfinite(mpu) or mpu <= 0):
        raise ValueError("website_navigation_scene_frame_invalid")
    sign = -1 if up == "-Y" else 1
    rotation = np.array([[1, 0, 0], [0, 0, -sign], [0, sign, 0]]) if up in {"Y", "-Y"} else np.eye(3)
    transform = np.eye(4)
    transform[:3, :3] = float(mpu) * rotation
    preparation = {
        "schema_version": "website_navigation_scene_frame.v1",
        "binding": {"splat_digest": base_scene["splat_digest"],
                    "collision_mesh_digest": base_scene["collision_mesh_digest"]},
        "coordinate_frame": {"runtime_to_simulator": transform.tolist(), "declared_meters_per_unit": float(mpu),
                             "declared_up_axis": up, "physical_scale_measured": False},
    }
    preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    return preparation


def navigation_scene_assets(*, base_scene: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    """Author the appearance and collision USDs for a scene kept as filmed; idempotent."""
    from .website_native_appearance import prepare_native_appearance
    from .website_scene_runtime_inputs import normalize_runtime_collision

    preparation = navigation_frame(base_scene)
    root = Path(output_root).resolve()
    appearance = prepare_native_appearance(preparation=preparation, base_scene=base_scene,
                                           output_root=root / "appearance")
    collision_root, collision = normalize_runtime_collision(preparation=preparation, base_scene=base_scene,
                                                            output_root=root)
    output = collision["output"]
    value = {
        "schema_version": SCHEMA,
        "claim_ceiling": "development_only",
        "frame": preparation["coordinate_frame"],
        "frame_digest": preparation["digest"],
        "appearance": {"path": appearance["artifact"]["path"], "sha256": appearance["artifact"]["digest"],
                       "size_bytes": appearance["artifact"]["size_bytes"], "receipt_digest": appearance["digest"]},
        "collision": {"path": str(collision_root / output["relative_path"]), "sha256": output["sha256"],
                      "size_bytes": output["size_bytes"],
                      "receipt_digest": collision.get("normalization_digest")},
        "source": {"splat_digest": base_scene["splat_digest"],
                   "collision_mesh_digest": base_scene["collision_mesh_digest"]},
        "paid_construction_performed": False,
        "scene_edited": False,
        "physical_scale_measured": False,
    }
    value["digest"] = canonical_digest(value, digest_field="digest")
    return value


__all__ = ["FILE", "SCHEMA", "navigation_frame", "navigation_scene_assets"]
