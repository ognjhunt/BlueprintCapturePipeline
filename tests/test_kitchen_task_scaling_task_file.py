"""Hermetic tests: kitchen task scaling accepts arbitrary task specs from a
JSON file (tasks are data, not code) and threads robot identity through.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.kitchen_task_scaling_preflight import (
    build_arg_parser,
    default_task_specs,
    load_task_specs_file,
    _selected_specs,
)


def _write_tasks(tmp_path: Path, tasks) -> Path:
    f = tmp_path / "tasks.json"
    f.write_text(json.dumps(tasks), encoding="utf-8")
    return f


def test_load_task_specs_file_fills_defaults(tmp_path):
    f = _write_tasks(tmp_path, [
        {"task_id": "microwave_door", "description": "Open the microwave door.",
         "required_target_terms": ["microwave", "door"]},
    ])
    specs = load_task_specs_file(f)
    assert len(specs) == 1
    spec = specs[0]
    assert spec["task_id"] == "microwave_door"
    # auto-derived scenario id; stance distances stay ABSENT so the runner
    # derives a robot-footprint-scaled ladder (robot-adaptive, not G1-pinned)
    assert spec["scenario_id"].endswith("microwave_door")
    assert "stance_distance_candidates_m" not in spec
    assert "preferred_stance_distance_m" not in spec
    assert spec["zone"]


def test_load_task_specs_file_accepts_tasks_wrapper(tmp_path):
    f = _write_tasks(tmp_path, {"tasks": [
        {"task_id": "a", "description": "Do a.", "required_target_terms": ["a"]}]})
    assert load_task_specs_file(f)[0]["task_id"] == "a"


def test_load_task_specs_file_rejects_missing_fields(tmp_path):
    f = _write_tasks(tmp_path, [{"task_id": "x"}])
    with pytest.raises(ValueError):
        load_task_specs_file(f)
    f2 = _write_tasks(tmp_path, [{"description": "no id"}])
    with pytest.raises(ValueError):
        load_task_specs_file(f2)


def test_load_task_specs_file_rejects_duplicate_ids(tmp_path):
    f = _write_tasks(tmp_path, [
        {"task_id": "a", "description": "Do a.", "required_target_terms": ["a"]},
        {"task_id": "a", "description": "Do a again.", "required_target_terms": ["a"]},
    ])
    with pytest.raises(ValueError):
        load_task_specs_file(f)


def test_selected_specs_merges_extra_specs_and_selects():
    extra = [{"task_id": "microwave_door", "scenario_id": "custom_04_microwave",
              "description": "Open the microwave door.",
              "required_target_terms": ["microwave"], "zone": "custom",
              "preferred_stance_distance_m": 0.24,
              "stance_distance_candidates_m": [0.24, 0.3]}]
    merged = _selected_specs((), extra_specs=extra)
    ids = [s["task_id"] for s in merged]
    assert "microwave_door" in ids
    assert {"sink_faucet", "stovetop_knob", "top_cabinet"} <= set(ids)
    only = _selected_specs(("microwave_door",), extra_specs=extra)
    assert [s["task_id"] for s in only] == ["microwave_door"]


def test_extra_spec_overrides_default_with_same_id():
    extra = [dict(default_task_specs()[0], preferred_stance_distance_m=0.99)]
    merged = _selected_specs(("sink_faucet",), extra_specs=extra)
    assert merged[0]["preferred_stance_distance_m"] == pytest.approx(0.99)
    assert len(merged) == 1


def test_build_request_omits_absent_stance_hints(tmp_path):
    from blueprint_pipeline.kitchen_task_scaling_preflight import build_request

    f = _write_tasks(tmp_path, [
        {"task_id": "microwave_door", "description": "Open the microwave door.",
         "required_target_terms": ["microwave"]}])
    specs = load_task_specs_file(f)
    request = build_request(kitchen_usd=tmp_path / "k.usd", task_specs=specs)
    scenario = request["scenarios"][0]
    assert "stance_distance_candidates_m" not in scenario
    assert "preferred_stance_distance_m" not in scenario
    # built-in specs still pin their tuned ladders
    request2 = build_request(kitchen_usd=tmp_path / "k.usd",
                             task_specs=default_task_specs())
    assert request2["scenarios"][0]["stance_distance_candidates_m"]


def test_cli_accepts_task_file_and_custom_task_id(tmp_path):
    f = _write_tasks(tmp_path, [
        {"task_id": "microwave_door", "description": "Open the microwave door.",
         "required_target_terms": ["microwave"]}])
    parser = build_arg_parser()
    args = parser.parse_args(["--task-file", str(f), "--task", "microwave_door"])
    assert args.task_file == str(f)
    assert args.task == ["microwave_door"]


def test_cli_accepts_robot_flags(tmp_path):
    parser = build_arg_parser()
    args = parser.parse_args(["--robot-id", "unitree_g1",
                              "--robot-profile-json", str(tmp_path / "p.json")])
    assert args.robot_id == "unitree_g1"
    assert args.robot_profile_json == str(tmp_path / "p.json")


def test_build_dry_render_command_threads_robot_flags(tmp_path):
    from blueprint_pipeline.kitchen_task_scaling_preflight import build_dry_render_command

    cmd = build_dry_render_command(
        request_path=tmp_path / "req.json",
        kitchen_usd=tmp_path / "k.usd",
        out_dir=tmp_path,
        robot_id="custom_bot",
        robot_profile_json=str(tmp_path / "p.json"),
    )
    assert "--robot-id" in cmd and cmd[cmd.index("--robot-id") + 1] == "custom_bot"
    assert "--robot-profile-json" in cmd
    assert cmd[cmd.index("--manipulation-reach-arm") + 1] == "auto"
    # default: no robot flags injected
    cmd2 = build_dry_render_command(
        request_path=tmp_path / "req.json", kitchen_usd=tmp_path / "k.usd", out_dir=tmp_path)
    assert "--robot-id" not in cmd2


def test_run_preflight_manifest_records_robot_profile_id(tmp_path):
    from blueprint_pipeline.kitchen_task_scaling_preflight import run_preflight

    manifest = run_preflight(
        out_dir=tmp_path / "out",
        kitchen_usd=tmp_path / "missing.usd",  # forces blocked manifest, no subprocess
        robot_id="custom_bot",
    )
    assert manifest["robot_profile_id"] == "custom_bot"
