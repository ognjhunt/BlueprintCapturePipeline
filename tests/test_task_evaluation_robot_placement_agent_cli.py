from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from blueprint_pipeline.task_evaluation_robot_placement_agent_cli import (
    _persist_images,
    _read_mapping,
    run_robot_placement_cli,
)


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_cli_persists_digest_bound_preview_without_embedding_data_url(tmp_path) -> None:
    digest = "sha256:" + hashlib.sha256(_ONE_PIXEL_PNG).hexdigest()
    records = _persist_images(
        [
            {
                "label": "top_down",
                "digest": digest,
                "image_url": "data:image/png;base64,"
                + base64.b64encode(_ONE_PIXEL_PNG).decode("ascii"),
                "detail": "high",
            }
        ],
        output_dir=tmp_path,
        prefix="candidate-00",
    )

    assert records[0]["digest"] == digest
    assert records[0]["size_bytes"] == len(_ONE_PIXEL_PNG)
    assert records[0]["path"].endswith("candidate-00-00-top_down.png")


def test_cli_reads_only_mapping_bindings(tmp_path) -> None:
    path = tmp_path / "binding.json"
    path.write_text('{"schema_version":"fixture.v1"}\n', encoding="utf-8")

    assert _read_mapping(path, label="scene_binding") == {
        "schema_version": "fixture.v1"
    }


def test_cli_draws_trajectory_but_keeps_analytic_gate_point_scoped(
    tmp_path, monkeypatch
) -> None:
    import blueprint_pipeline.task_evaluation_robot_placement_agent_cli as module

    trajectory = {
        "phases": [
            {
                "phase_id": "precontact",
                "position_world_m": [2.79, -6.76, 0.818],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
            },
            {
                "phase_id": "push_contact",
                "position_world_m": [2.91, -6.76, 0.818],
                "orientation_world_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
            },
        ]
    }
    proposal = {
        "candidate_id": "candidate",
        "support_surface_id": "surface",
        "pose": {
            "position_world_m": [3.0, -6.3, 0.75],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
    rendered_waypoints = []

    monkeypatch.setattr(module.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(module, "build_robot_placement_geometry_index", lambda **_: object())
    monkeypatch.setattr(module, "summarize_robot_placement_geometry", lambda *_a, **_k: {})
    def enumerate_candidates(**kwargs):
        assert kwargs["trajectory_waypoints_world_m"] == [
            [2.79, -6.76, 0.818],
            [2.91, -6.76, 0.818],
        ]
        assert kwargs["trajectory_phase_ids"] == ["precontact", "push_contact"]
        assert kwargs["trajectory_orientations_world_xyzw"] == [
            [0.0, 0.70710678, 0.0, 0.70710678],
            [0.0, 0.70710678, 0.0, 0.70710678],
        ]
        assert kwargs["geometry_worker_count"] == 4
        assert kwargs["trajectory_worker_count"] == 4
        return [proposal]

    monkeypatch.setattr(
        module, "enumerate_robot_placement_geometry_candidates", enumerate_candidates
    )

    def render(*, trajectory_waypoints_world_m=(), **_kwargs):
        rendered_waypoints.append(list(trajectory_waypoints_world_m))
        digest = "sha256:" + hashlib.sha256(_ONE_PIXEL_PNG).hexdigest()
        return [
            {
                "label": "top_down_xy",
                "digest": digest,
                "image_url": "data:image/png;base64,"
                + base64.b64encode(_ONE_PIXEL_PNG).decode("ascii"),
                "detail": "high",
            }
        ]

    def validate(
        *,
        index,
        proposal,
        target_position_world_m,
        robot_id,
        trajectory_waypoints_world_m,
        trajectory_phase_ids,
        trajectory_orientations_world_xyzw,
    ):
        assert index is not None
        assert proposal["candidate_id"] == "candidate"
        assert target_position_world_m == [3.0, -6.7, 0.8]
        assert robot_id == "franka_panda"
        assert trajectory_waypoints_world_m == [
            [2.79, -6.76, 0.818],
            [2.91, -6.76, 0.818],
        ]
        assert trajectory_phase_ids == ["precontact", "push_contact"]
        assert trajectory_orientations_world_xyzw == [
            [0.0, 0.70710678, 0.0, 0.70710678],
            [0.0, 0.70710678, 0.0, 0.70710678],
        ]
        return {"status": "passed"}

    def run_agent(**kwargs):
        inventory = kwargs["scene_context"][
            "deterministic_geometry_passing_candidate_inventory"
        ]
        assert kwargs["scene_context"][
            "deterministic_geometry_passing_candidate_inventory_digest"
        ] == module.canonical_digest(
            {"trajectory_digest": None, "candidates": inventory}
        )
        assert kwargs["scene_context"]["model_must_select_exact_inventory_member"] is True
        kwargs["validate_candidate"](proposal)
        kwargs["render_candidate"](proposal, 0)
        assert kwargs["task_trajectory"] == trajectory
        return {"receipt_digest": "sha256:" + "a" * 64}

    monkeypatch.setattr(module, "render_robot_placement_geometry_previews", render)
    monkeypatch.setattr(module, "validate_robot_placement_geometry_candidate", validate)
    monkeypatch.setattr(module, "OpenAIAgentsSDKInvoker", lambda _config: object())
    monkeypatch.setattr(module, "run_task_evaluation_robot_placement_agent", run_agent)

    run_robot_placement_cli(
        run_id="trajectory-preview",
        scene_collision_usd=Path("scene.usda"),
        robot_asset_usd=Path("robot.usda"),
        target_position_world_m=[3.0, -6.7, 0.8],
        scene_binding={"scene": "839873"},
        task_binding={"task": "push"},
        overview_image_paths=[],
        output_dir=tmp_path / "output",
        max_rounds=2,
        candidate_inventory_cap=12,
        max_input_tokens=1_000,
        max_inference_cost_usd=1.0,
        allow_live_invocation=False,
        tracing_disabled=True,
        task_trajectory=trajectory,
    )

    assert rendered_waypoints == [
        [[2.79, -6.76, 0.818], [2.91, -6.76, 0.818]],
        [[2.79, -6.76, 0.818], [2.91, -6.76, 0.818]],
    ]
