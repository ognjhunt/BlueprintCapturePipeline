from __future__ import annotations

import json

from blueprint_pipeline.native_task_arena_controls_admission import (
    validate_native_task_controls_admission,
)
from tests.test_native_task_arena_bundle import (
    _articulated_packet,
    _qualified_construction,
    _qualified_controls,
)


def test_shared_policy_and_website_controls_admission_accepts_exact_pair(
    tmp_path,
) -> None:
    _packet, scene = _articulated_packet(tmp_path)
    construction_path = _qualified_construction(tmp_path, scene)
    controls_path = _qualified_controls(tmp_path, scene, construction_path)

    admission = validate_native_task_controls_admission(
        scene_plan=scene,
        construction_result=json.loads(construction_path.read_text()),
        control_result=json.loads(controls_path.read_text()),
    )

    assert admission["scene_plan_digest"] == scene["plan_digest"]
    assert [row["control_id"] for row in admission["controls"]] == [
        "zero_action_negative",
        "deterministic_scripted_positive",
    ]
