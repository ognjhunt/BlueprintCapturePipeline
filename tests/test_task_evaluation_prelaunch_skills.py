from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

import blueprint_pipeline.task_evaluation_prelaunch_skills as skills
from blueprint_pipeline.task_evaluation_prelaunch_skills import (
    CAD_CONFIG_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    canonical_digest,
    execute_prelaunch_skill_plan,
    validate_prelaunch_skill_plan,
    validate_profile_prelaunch_skill_plan,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _box(ins_id: str, label: str, center: tuple[float, float, float]) -> dict:
    cx, cy, cz = center
    corners = []
    for x in (cx - 0.1, cx + 0.1):
        for y in (cy - 0.1, cy + 0.1):
            for z in (cz - 0.1, cz + 0.1):
                corners.append({"x": x, "y": y, "z": z})
    return {"ins_id": ins_id, "label": label, "bounding_box": corners}


def _room_sources(root: Path) -> tuple[Path, Path]:
    structure = root / "structure.json"
    labels = root / "labels.json"
    _write(
        structure,
        {
            "rooms": [{"profile": [[0, 0], [2, 0], [2, 2], [0, 2]]}],
            "walls": [],
            "holes": [],
        },
    )
    _write(labels, [_box("10", "cup", (1.0, 1.0, 0.8))])
    return structure, labels


def _plan(*, source_bundle: dict, steps: list[dict]) -> dict:
    value = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "program_id": "arm-decision-proof-v1",
        "plan_id": "prelaunch-skill-plan-001",
        "source_bundle": {
            "bundle_id": source_bundle["bundle_id"],
            "digest": source_bundle["digest"],
        },
        "steps": steps,
    }
    value["plan_digest"] = canonical_digest(value, digest_field="plan_digest")
    return value


def _profile(*, root: Path, plan_path: Path, plan: dict, inputs: dict[str, Path]) -> dict:
    source_bundle = {
        "bundle_id": "interiorgs-sage-001",
        "source_kind": "interiorgs_sage",
        "uri": "gs://blueprint-runs/interiorgs-sage-001.json",
        "digest": "sha256:" + "a" * 64,
    }
    immutable_inputs = [
        {"name": "prelaunch_skill_plan", "path": str(plan_path), "digest": _digest(plan_path)},
        *(
            {"name": name, "path": str(path), "digest": _digest(path)}
            for name, path in sorted(inputs.items())
        ),
    ]
    return {
        "source_bundle": source_bundle,
        "immutable_inputs": immutable_inputs,
        "prelaunch_skill_plan": {
            "plan_id": plan["plan_id"],
            "path": str(plan_path),
            "digest": _digest(plan_path),
        },
    }


def _room_profile(root: Path) -> tuple[dict, dict]:
    structure, labels = _room_sources(root / "inputs")
    source_bundle = {
        "bundle_id": "interiorgs-sage-001",
        "source_kind": "interiorgs_sage",
        "uri": "gs://blueprint-runs/interiorgs-sage-001.json",
        "digest": "sha256:" + "a" * 64,
    }
    plan = _plan(
        source_bundle=source_bundle,
        steps=[
            {
                "step_id": "room-survey",
                "adapter": "interiorgs_room_survey",
                "structure_input": "structure",
                "labels_input": "labels",
                "scene_id": "scene-001",
                "target_ins_id": "10",
                "timeout_seconds": 60,
            }
        ],
    )
    plan_path = root / "inputs" / "prelaunch-plan.json"
    _write(plan_path, plan)
    profile = _profile(
        root=root,
        plan_path=plan_path,
        plan=plan,
        inputs={"structure": structure, "labels": labels},
    )
    return profile, plan


def test_executes_room_survey_from_profile_bound_inputs_and_retains_receipt(tmp_path: Path) -> None:
    profile, plan = _room_profile(tmp_path)

    result = execute_prelaunch_skill_plan(profile=profile, run_root=tmp_path / "run")

    assert validate_prelaunch_skill_plan(plan) == []
    assert validate_profile_prelaunch_skill_plan(profile) == []
    assert result["status"] == "passed", result
    assert result["provider_mutation_performed"] is False
    assert result["allocator_invoked"] is False
    assert result["agent_operator_used"] is False
    assert result["steps"][0]["adapter"] == "interiorgs_room_survey"
    output = Path(result["steps"][0]["output"]["path"])
    survey = json.loads(output.read_text(encoding="utf-8"))
    assert survey["target_closeup"]["target_ins_id"] == "10"
    assert survey["claim_boundary"]["survey_previews_are_not_method_inputs"] is True
    assert result["execution_digest"] == canonical_digest(result, digest_field="execution_digest")

    replay = execute_prelaunch_skill_plan(profile=profile, run_root=tmp_path / "run")
    assert replay == result


def test_plan_rejects_smuggled_command_or_unknown_step_field(tmp_path: Path) -> None:
    profile, plan = _room_profile(tmp_path)
    plan["steps"][0]["argv"] = ["--execute"]
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")

    blockers = validate_prelaunch_skill_plan(plan)

    assert "prelaunch_skill_room_survey_fields_invalid:room-survey" in blockers
    assert validate_profile_prelaunch_skill_plan(profile) == []


def test_plan_source_binding_mismatch_is_retained_without_running_a_skill(tmp_path: Path) -> None:
    profile, plan = _room_profile(tmp_path)
    plan["source_bundle"]["digest"] = "sha256:" + "b" * 64
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    plan_path = Path(profile["prelaunch_skill_plan"]["path"])
    _write(plan_path, plan)
    profile["prelaunch_skill_plan"]["digest"] = _digest(plan_path)
    profile["immutable_inputs"][0]["digest"] = _digest(plan_path)

    result = execute_prelaunch_skill_plan(profile=profile, run_root=tmp_path / "run")

    assert result["status"] == "blocked"
    assert "prelaunch_skill_plan_source_bundle_binding_mismatch" in result["blockers"]
    assert result["steps"] == []
    assert not (tmp_path / "run" / "prelaunch_skills" / "room-survey.json").exists()


def test_executor_rejects_a_plan_pointer_outside_immutable_profile_inputs(tmp_path: Path) -> None:
    profile, _ = _room_profile(tmp_path)
    profile["immutable_inputs"] = [
        item for item in profile["immutable_inputs"] if item["name"] != "prelaunch_skill_plan"
    ]

    result = execute_prelaunch_skill_plan(profile=profile, run_root=tmp_path / "run")

    assert result["status"] == "blocked"
    assert "launch_profile_prelaunch_skill_plan_not_immutable" in result["blockers"]
    assert result["steps"] == []


def test_earthtojake_adapter_uses_only_pinned_configuration_and_profile_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bundle = {
        "bundle_id": "interiorgs-sage-001",
        "source_kind": "interiorgs_sage",
        "uri": "gs://blueprint-runs/interiorgs-sage-001.json",
        "digest": "sha256:" + "a" * 64,
    }
    step_path = tmp_path / "repo" / "assets" / "control.step"
    step_path.parent.mkdir(parents=True)
    step_path.write_text("ISO-10303-21;\nEND-ISO-10303-21;\n", encoding="utf-8")
    config = {
        "schema_version": CAD_CONFIG_SCHEMA_VERSION,
        "repo_root": str((tmp_path / "repo").resolve()),
        "cad_skill_root": str((tmp_path / "pinned-skill").resolve()),
        "cad_python": str(Path(sys.executable).resolve()),
        "expected_commit": "1" * 40,
        "expected_tree": "2" * 40,
    }
    config["config_digest"] = canonical_digest(config, digest_field="config_digest")
    config_path = tmp_path / "inputs" / "cad-config.json"
    _write(config_path, config)
    plan = _plan(
        source_bundle=source_bundle,
        steps=[
            {
                "step_id": "cad-inspection",
                "adapter": "earthtojake_step_inspection",
                "step_input": "step",
                "configuration_input": "cad-config",
                "timeout_seconds": 45,
            }
        ],
    )
    plan_path = tmp_path / "inputs" / "prelaunch-plan.json"
    _write(plan_path, plan)
    profile = _profile(
        root=tmp_path,
        plan_path=plan_path,
        plan=plan,
        inputs={"step": step_path, "cad-config": config_path},
    )
    observed: dict[str, object] = {}

    def fake_capture(**kwargs: object) -> dict:
        observed.update(kwargs)
        output = Path(str(kwargs["output_path"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        _write(output, {"ok": True, "tokens": [{}], "errors": []})
        return {"ok": True, "tokens": [{}], "errors": []}

    monkeypatch.setattr(skills, "capture_cad_inspection", fake_capture)

    result = execute_prelaunch_skill_plan(profile=profile, run_root=tmp_path / "run")

    assert result["status"] == "passed"
    assert observed["step_path"] == step_path.resolve()
    assert observed["expected_commit"] == "1" * 40
    assert observed["timeout_seconds"] == 45
    assert result["steps"][0]["output"]["digest"].startswith("sha256:")
