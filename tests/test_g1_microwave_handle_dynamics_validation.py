from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline import g1_microwave_handle_dynamics_validation as dynamics


class _FakeMujoco:
    mjtObj = SimpleNamespace(mjOBJ_BODY=1)

    @staticmethod
    def mj_id2name(model, _object_type, body_id):
        return model.body_names[body_id]


def test_right_hand_geom_ids_selects_only_finger_bodies():
    model = SimpleNamespace(
        ngeom=4,
        geom_bodyid=np.asarray([0, 1, 2, 3]),
        body_names=(
            "floor",
            "right_hand_index_link",
            "right_hand_thumb_link",
            "right_wrist_yaw_link",
        ),
    )

    assert dynamics._right_hand_geom_ids(_FakeMujoco, model) == {1, 2}


def test_right_hand_geom_ids_fails_closed_when_missing():
    model = SimpleNamespace(
        ngeom=2,
        geom_bodyid=np.asarray([0, 1]),
        body_names=("floor", "right_wrist_yaw_link"),
    )

    with pytest.raises(
        ValueError,
        match="g1_microwave_handle_dynamics_right_hand_geoms_missing",
    ):
        dynamics._right_hand_geom_ids(_FakeMujoco, model)


def test_positive_handle_contact_requires_observed_contact_and_force():
    assert dynamics._positive_handle_contact_proven(
        contact_step_count=2,
        positive_force_contact_count=1,
        peak_normal_force_n=4.0,
    )
    assert not dynamics._positive_handle_contact_proven(
        contact_step_count=0,
        positive_force_contact_count=0,
        peak_normal_force_n=0.0,
    )
    assert not dynamics._positive_handle_contact_proven(
        contact_step_count=2,
        positive_force_contact_count=0,
        peak_normal_force_n=0.0,
    )


def test_main_writes_bound_trace_and_qualified_report(tmp_path, monkeypatch):
    trace = np.asarray(
        (
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.001, -0.02, -0.1, 1.0, 4.0),
        )
    )
    report = {
        "schema_version": dynamics.SCHEMA_VERSION,
        "status": "qualified_contact_driven_handle_only_partial_articulation",
        "qualification": {
            "contact_driven_door_articulation_proven": True,
            "requested_opening_within_tolerance_proven": True,
        },
    }
    monkeypatch.setattr(
        dynamics,
        "validate_handle_only_dynamics",
        lambda **_kwargs: (report, trace),
    )
    report_out = tmp_path / "report.json"
    trace_out = tmp_path / "trace.npy"

    exit_code = dynamics.main(
        [
            "--model",
            "model.xml",
            "--standing-initialization",
            "standing.json",
            "--initial-policy-observation",
            "observation.json",
            "--target-focus-report",
            "focus.json",
            "--grasp-report",
            "grasp.json",
            "--trajectory",
            "trajectory.npy",
            "--report-out",
            str(report_out),
            "--trace-out",
            str(trace_out),
        ]
    )

    assert exit_code == 0
    assert np.array_equal(np.load(trace_out, allow_pickle=False), trace)
    written = json.loads(report_out.read_text(encoding="utf-8"))
    assert written["trace"]["row_count"] == 2
    assert written["trace"]["path"] == str(trace_out)
    assert len(written["trace"]["sha256"]) == 64


def test_main_returns_nonzero_when_articulation_is_not_proven(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dynamics,
        "validate_handle_only_dynamics",
        lambda **_kwargs: (
            {
                "schema_version": dynamics.SCHEMA_VERSION,
                "status": "blocked_no_contact_driven_articulation",
                "qualification": {
                    "contact_driven_door_articulation_proven": False,
                    "requested_opening_within_tolerance_proven": False,
                },
            },
            np.zeros((1, 5)),
        ),
    )

    exit_code = dynamics.main(
        [
            "--model",
            "model.xml",
            "--standing-initialization",
            "standing.json",
            "--initial-policy-observation",
            "observation.json",
            "--target-focus-report",
            "focus.json",
            "--grasp-report",
            "grasp.json",
            "--trajectory",
            "trajectory.npy",
            "--report-out",
            str(tmp_path / "report.json"),
            "--trace-out",
            str(tmp_path / "trace.npy"),
        ]
    )

    assert exit_code == 1
