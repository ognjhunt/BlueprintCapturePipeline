"""Container smoke for the pinned GEAR-SONIC WBC tree and real robot model XML.

Loads the pinned MuJoCo model from the ``/opt/wbc`` deployment tree, validates
its joint set against the protocol-v4 joint-order contract, and proves that two
asymmetric perturbations move the intended distinct joints and produce distinct
FK states. Fail-closes with a precise blocker when the WBC tree or model is
absent; the hermetic committed fixture under ``tests/fixtures/gear_sonic_g1_min``
exercises the same code path without the container.

Run inside the container with::

    python -m blueprint_pipeline.gear_sonic_container_smoke
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .gear_sonic_joint_order_contract import (
    JOINT_ORDER_SCHEMA_VERSION,
    PROTOCOL_V4_BODY_JOINT_NAMES,
    PROTOCOL_V4_MAPPING_DIGEST,
)
from .gear_sonic_official_zmq_executor import (
    DEFAULT_MODEL,
    DEFAULT_ROOT,
    HAND_DIM,
    MODEL_ENV,
    ROOT_ENV,
    _official_mujoco_fk,
    _sha256_file,
)

SMOKE_SCHEMA_VERSION = "gear_sonic_container_smoke.v1"
LEFT_PERTURBED_JOINT = "left_elbow_joint"
RIGHT_PERTURBED_JOINT = "right_elbow_joint"
PERTURBATION_RADIANS = 0.6


def _landmark_map(landmarks: list[dict[str, Any]]) -> dict[str, tuple[float, float, float]]:
    return {row["name"]: (row["x"], row["y"], row["z"]) for row in landmarks}


def run_container_smoke(
    *, root: Path | str | None = None, model_path: Path | str | None = None
) -> dict[str, Any]:
    resolved_root = Path(root or os.getenv(ROOT_ENV, DEFAULT_ROOT)).expanduser().resolve()
    resolved_model = (
        Path(model_path or os.getenv(MODEL_ENV, DEFAULT_MODEL)).expanduser().resolve()
    )
    if resolved_root.name != "wbc" or not (resolved_root / "gear_sonic_deploy").is_dir():
        raise RuntimeError("official_gear_sonic_container_smoke_wbc_tree_missing")
    if not resolved_model.is_file() or resolved_root not in resolved_model.parents:
        raise RuntimeError("official_gear_sonic_container_smoke_robot_model_missing")
    try:
        import mujoco  # type: ignore # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "official_gear_sonic_container_smoke_mujoco_unavailable"
        ) from error

    neutral_body = [0.0] * len(PROTOCOL_V4_BODY_JOINT_NAMES)
    left_body = list(neutral_body)
    left_body[PROTOCOL_V4_BODY_JOINT_NAMES.index(LEFT_PERTURBED_JOINT)] = PERTURBATION_RADIANS
    right_body = list(neutral_body)
    right_body[PROTOCOL_V4_BODY_JOINT_NAMES.index(RIGHT_PERTURBED_JOINT)] = (
        PERTURBATION_RADIANS
    )
    hands = {"left_hand": [0.0] * HAND_DIM, "right_hand": [0.0] * HAND_DIM}
    _, _, neutral_marks, applied = _official_mujoco_fk(
        model_path=resolved_model, body_positions=neutral_body, **hands
    )
    _, _, left_marks, _ = _official_mujoco_fk(
        model_path=resolved_model, body_positions=left_body, **hands
    )
    _, _, right_marks, _ = _official_mujoco_fk(
        model_path=resolved_model, body_positions=right_body, **hands
    )
    neutral_map = _landmark_map(neutral_marks)
    left_map = _landmark_map(left_marks)
    right_map = _landmark_map(right_marks)
    states_distinct = (
        left_map != right_map and left_map != neutral_map and right_map != neutral_map
    )
    if not states_distinct:
        raise RuntimeError("official_gear_sonic_container_smoke_fk_states_not_distinct")
    return {
        "schema_version": SMOKE_SCHEMA_VERSION,
        "status": "passed",
        "wbc_root": str(resolved_root),
        "robot_model_path": str(resolved_model),
        "robot_model_sha256": _sha256_file(resolved_model),
        "joint_order_schema_version": JOINT_ORDER_SCHEMA_VERSION,
        "mapping_digest": PROTOCOL_V4_MAPPING_DIGEST,
        "applied_dof_mapping": applied,
        "perturbation_evidence": {
            "left_perturbed_joint": LEFT_PERTURBED_JOINT,
            "right_perturbed_joint": RIGHT_PERTURBED_JOINT,
            "perturbation_radians": PERTURBATION_RADIANS,
            "fk_states_distinct": True,
        },
    }


def main() -> int:
    try:
        report = run_container_smoke()
    except (RuntimeError, ValueError) as error:
        print(json.dumps({"status": "blocked", "blocker": str(error)}), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
