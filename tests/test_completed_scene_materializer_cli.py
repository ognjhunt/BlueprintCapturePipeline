"""Recovery entry points retain real parameter types and preserve fail-closed producers."""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import inspect
import json
from pathlib import Path

from PIL import Image
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.materializer_cli import build_parser, call_arguments

ROOT = Path(__file__).resolve().parents[1]


def _script(name="materialize_completed_scene_inputs"):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", [
    "legacy-placement-alias", "astra-phase-adoption", "astra-automatic-phase-adoption", "packaged-blender-runtime",
    "astra-native-adoption", "completed-scene-attempt", "completed-mesh-inputs",
    "completed-scene-submission", "repair-support",
])
def test_every_completed_scene_command_supplies_required_inputs(name, tmp_path, monkeypatch, capsys):
    cli = _script()
    step = cli.STEPS[name]
    signature = inspect.signature(step.materialize)
    required = {key for key, param in signature.parameters.items()
                if param.default is inspect.Parameter.empty}
    assert required <= step.params.keys()
    args = [name]
    expected = {}
    for key, param in step.params.items():
        if param.json_file:
            value = {"retained": key}
            path = tmp_path / f"{key}.json"
            path.write_text(json.dumps(value))
            text = str(path)
        elif param.type is Path:
            value = tmp_path / key
            text = str(value)
        elif param.type is int:
            value, text = 1, "1"
        else:
            value, text = key, key
        args.extend([param.flag, text])
        if param.accumulate:
            args.extend([param.flag, "second"])
            value = (value, "second")
        expected[key] = value
    received = []

    def capture(**kwargs):
        signature.bind(**kwargs)  # No invented argument or missing required producer input.
        received.append(kwargs)
        return {"schema_version": "cli_fixture.v1"}

    monkeypatch.setitem(cli.STEPS, name, replace(step, materialize=capture))
    assert cli.main(args) == 0
    assert received == [expected]
    assert json.loads(capsys.readouterr().out)["provider_mutation_performed"] is False


def test_semantic_teacher_cli_supplies_optional_selection_json(tmp_path):
    cli = _script("prepare_artifixer3d_inputs")
    selection = tmp_path / "selection.json"
    selection.write_text('{"selection_digest":"retained"}')
    args = ["semantic-teacher", "--source-candidate-inputs", "source.json", "--task-id", "task",
            "--semantic-teacher-frames-root", "frames", "--editor-identity", str(selection),
            "--prompt-policy", "retained-policy", "--output", "receipt.json"]
    parser = build_parser(cli.STEPS)
    omitted = call_arguments(cli.STEPS["semantic-teacher"], parser.parse_args(args))
    supplied = call_arguments(cli.STEPS["semantic-teacher"],
                              parser.parse_args([*args, "--training-view-selection", str(selection)]))
    assert omitted["training_view_selection"] is None
    assert supplied["training_view_selection"] == {"selection_digest": "retained"}


def test_repair_support_cli_calls_real_producer_and_preserves_source_masks(tmp_path, capsys):
    cli = _script()
    frame, calibrated, sam = [tmp_path / name for name in ("frame.png", "calibrated.png", "sam.png")]
    Image.new("RGB", (80, 80), "white").save(frame)
    mask = Image.new("L", (80, 80), 0)
    mask.paste(255, (35, 35, 45, 45))
    mask.save(calibrated)
    Image.new("L", (80, 80), 0).save(sam)
    original = [path.read_bytes() for path in (frame, calibrated, sam)]
    output = tmp_path / "support"
    args = ["repair-support", "--calibrated-mask", str(calibrated), "--sam-mask", str(sam),
            "--source-frame", str(frame), "--calibration-digest", "sha256:" + "a" * 64,
            "--output-root", str(output)]
    assert cli.main(args) == 0
    with Image.open(output / "object-core.png") as core:
        assert core.tobytes() == mask.tobytes()
    assert [path.read_bytes() for path in (frame, calibrated, sam)] == original
    capsys.readouterr()
    assert cli.main(args) == 2
    assert "output_exists" in capsys.readouterr().out


def test_phase_descriptor_is_retained_without_rewriting_a_prior_file(tmp_path, monkeypatch, capsys):
    cli = _script()
    receipt = {"schema_version": "fixture.v1", "phases": ["source_analysis"]}
    receipt["adoption_digest"] = canonical_digest(receipt, digest_field="adoption_digest")
    monkeypatch.setattr(cli, "materialize_phase_adoption", lambda **kwargs: receipt)
    output = tmp_path / "adoption.json"
    args = ["astra-phase-adoption", "--prior-runtime", str(tmp_path), "--phase", "source_analysis",
            "--output", str(output)]
    assert cli.main(args) == 0
    assert json.loads(output.read_text()) == receipt
    assert json.loads(capsys.readouterr().out)["adoption_digest"] == receipt["adoption_digest"]
    assert cli.main(args) == 2
    assert json.loads(output.read_text()) == receipt


def test_missing_adoption_inputs_fail_before_any_output(tmp_path, capsys):
    cli = _script()
    output = tmp_path / "adoption.json"
    assert cli.main(["astra-phase-adoption", "--prior-runtime", str(tmp_path / "absent"),
                     "--phase", "source_analysis", "--output", str(output)]) == 2
    assert "astra_phase_adoption_descriptor_invalid" in capsys.readouterr().out
    assert not output.exists()
