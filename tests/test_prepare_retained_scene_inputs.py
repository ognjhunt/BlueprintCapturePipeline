"""Exercise local replay commands through their real producer validators."""

from __future__ import annotations

import importlib.util
import inspect
import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.materializer_cli import Step
from blueprint_pipeline.semantic_teacher_candidate_reuse import load_retained_selection

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "prepare_retained_scene_inputs.py"


@pytest.fixture
def cli(monkeypatch):
    def refuse_network(*_args, **_kwargs):
        pytest.fail("local materialization attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse_network)
    monkeypatch.setattr(socket, "getaddrinfo", refuse_network)
    spec = importlib.util.spec_from_file_location("prepare_retained_scene_inputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(path, value):
    path.write_text(json.dumps(value))
    return path


def _snapshot(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _retained_source(root, camera_id):
    """Small, explicitly hermetic unreviewed candidates, never provider proof."""
    root.mkdir()
    image = root / "candidate.png"
    Image.new("RGB", (4, 3), "#345678").save(image)
    rgb_digest, mask_digest = "sha256:" + "a" * 64, "sha256:" + "b" * 64
    frame = {"camera_id": camera_id, "input_rgb": {"sha256": rgb_digest},
             "edit_mask": {"sha256": mask_digest}}
    request = {
        "schema_version": "semantic_teacher_image_edit_runtime_request.v1",
        "backend": {"execution": {"model_snapshot": "hermetic-fixture", "mask_encoding": "fixture"},
                    "registry_entry": {"backend_id": "hermetic-fixture"}},
        "tasks": [{"task_id": "fixture-task", "frames": [frame]}],
        "claim_ceiling": "development_only",
    }
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    result = {
        "schema_version": "semantic_teacher_image_edit_runtime_result.v1",
        "source_runtime_request_digest": request["request_digest"],
        "status": "completed_unreviewed_semantic_teacher_candidates",
        "model_snapshot": "hermetic-fixture", "backend_id": "hermetic-fixture",
        "tasks": [{"task_id": "fixture-task", "frames": [{
            "camera_id": camera_id, "terminal_state": "completed_unreviewed_candidate",
            "source_rgb_sha256": rgb_digest, "edit_mask_sha256": mask_digest,
            "semantic_teacher_frame": {"relative_path": image.name,
                                       "size_bytes": image.stat().st_size,
                                       "sha256": _sha256_file(image)},
            "provider_usage": {"fixture_only": True}, "computed_editor_cost_usd": 0.0,
        }]}], "claim_ceiling": "development_only",
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    return {"request_path": str(_json(root / "request.json", request)),
            "result_path": str(_json(root / "result.json", result)),
            "output_root": str(root), "camera_ids": [camera_id]}


def _selection_args(mode, sources, output, tmp_path):
    if mode == "retained-selection":
        source = sources[0]
        return [mode, "--source-request", source["request_path"],
                "--source-result", source["result_path"], "--source-output-root", source["output_root"],
                "--camera-id", source["camera_ids"][0], "--output-root", str(output)]
    return [mode, "--sources", str(_json(tmp_path / "sources.json", sources)),
            "--output-root", str(output)]


@pytest.mark.parametrize("step", ["retained-selection", "retained-selection-from-sources",
                                  "sage-collision-partition", "object-observations", "website-inputs"])
def test_dispatch_supplies_every_upstream_parameter(cli, tmp_path, step, monkeypatch, capsys):
    producers = {
        "retained-selection": cli.materialize_retained_selection,
        "retained-selection-from-sources": cli.materialize_retained_selection_from_sources,
        "sage-collision-partition": cli.materialize_sage_collision_partition,
        "object-observations": cli.materialize_object_observations,
        "website-inputs": cli.materialize_website_inputs,
    }
    entry = cli.STEPS[step]
    upstream = {name for name, param in inspect.signature(producers[step]).parameters.items()
                if param.kind == inspect.Parameter.KEYWORD_ONLY}
    assert upstream == set(entry.params)
    argv, expected = [step], {}
    for name, param in entry.params.items():
        if param.json_file:
            value = [] if name == "sources" else {"fixture": name}
            argv.extend([param.flag, str(_json(tmp_path / f"{name}.json", value))])
        elif param.accumulate:
            value = ("first", "second")
            for item in value:
                argv.extend([param.flag, item])
        else:
            value = tmp_path / name
            argv.extend([param.flag, str(value)])
        expected[name] = value
    received = []

    def capture(**kwargs):
        received.append(kwargs)
        return {}

    monkeypatch.setitem(cli.STEPS, step, Step(entry.summary, capture, entry.params))
    assert cli.main(argv) == 0
    assert received == [expected]
    assert json.loads(capsys.readouterr().out)["provider_mutation_performed"] is False


@pytest.mark.parametrize("mode", ["retained-selection", "retained-selection-from-sources"])
def test_selection_dispatch_preserves_original_receipts_candidates_and_lineage(cli, tmp_path, mode, capsys):
    sources = [_retained_source(tmp_path / "source-1", "camera-1")]
    if mode.endswith("from-sources"):
        sources.append(_retained_source(tmp_path / "source-2", "camera-2"))
    before = [_snapshot(Path(source["output_root"])) for source in sources]
    output = tmp_path / "selected"
    assert cli.main(_selection_args(mode, sources, output, tmp_path)) == 0
    summary = json.loads(capsys.readouterr().out)
    path = output / "selection.json"
    selection = json.loads(path.read_text())
    assert summary["receipt_path"] == str(path)
    assert summary["receipt_digest"] == canonical_digest(selection, digest_field="selection_digest")
    assert selection["status"] == "selected_unreviewed_candidates"
    rows = load_retained_selection(selection_path=path, render={"derived_frames": [
        {"camera_id": source["camera_ids"][0], "digest": "sha256:" + "a" * 64} for source in sources]})
    assert len(rows) == len(sources)
    for row, source, original in zip(rows, sources, before, strict=True):
        assert Path(row["source_runtime_request"]["path"]).read_bytes() == original[Path("request.json")]
        assert Path(row["source_runtime_result"]["path"]).read_bytes() == original[Path("result.json")]
        assert Path(row["candidate"]["path"]).read_bytes() == original[Path("candidate.png")]
        assert _snapshot(Path(source["output_root"])) == original
    selection_bytes = path.read_bytes()
    assert cli.main(_selection_args(mode, sources, output, tmp_path)) == 2
    assert path.read_bytes() == selection_bytes


@pytest.mark.parametrize("mode", ["retained-selection", "retained-selection-from-sources"])
@pytest.mark.parametrize("corruption", ["candidate", "result"])
def test_selection_refuses_changed_bytes_without_reporting_success(cli, tmp_path, mode, corruption, capsys):
    sources = [_retained_source(tmp_path / "source", "camera-1")]
    source = sources[0]
    if corruption == "candidate":
        (Path(source["output_root"]) / "candidate.png").write_bytes(b"changed")
    else:
        path = Path(source["result_path"])
        value = json.loads(path.read_text())
        value["model_snapshot"] = "changed-without-resealing"
        _json(path, value)
    before = _snapshot(Path(source["output_root"]))
    output = tmp_path / "selected"
    assert cli.main(_selection_args(mode, sources, output, tmp_path)) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    selection_path = output / "selection.json"
    if selection_path.exists():
        # The existing producer can leave a partial output before final
        # readback fails. That output must still fail its original validator.
        with pytest.raises(ValueError, match="receipt_invalid"):
            load_retained_selection(selection_path=selection_path, render={"derived_frames": [
                {"camera_id": "camera-1", "digest": "sha256:" + "a" * 64}]})
    assert _snapshot(Path(source["output_root"])) == before


def test_multi_source_selection_refuses_duplicate_camera_identity(cli, tmp_path, capsys):
    sources = [_retained_source(tmp_path / "first", "same-camera"),
               _retained_source(tmp_path / "second", "same-camera")]
    output = tmp_path / "selected"
    assert cli.main(_selection_args("retained-selection-from-sources", sources, output, tmp_path)) == 2
    assert "output_invalid" in json.loads(capsys.readouterr().out)["blockers"][0]
    assert not output.exists()


def test_partition_dispatch_keeps_source_faces_and_rechecks_output(cli, tmp_path, capsys):
    from tests.test_sage_collision_partition import combined_scene

    source, labels = combined_scene(tmp_path)
    before = source.read_bytes(), labels.read_bytes()
    output = tmp_path / "partition"
    argv = ["sage-collision-partition", "--source", str(source), "--labels", str(labels),
            "--instance-id", "subject", "--instance-id", "support", "--output-root", str(output)]
    assert cli.main(argv) == 0
    receipt = json.loads((output / "collision_partition.json").read_text())
    assert receipt["receipt_digest"] == json.loads(capsys.readouterr().out)["receipt_digest"]
    assert receipt["source_faces_deleted"] == 0
    assert receipt["native_collision_cooking_qualified"] is False
    assert sorted(index for row in receipt["face_partitions"] for index in row["source_face_indices"]) == list(range(18))
    assert (source.read_bytes(), labels.read_bytes()) == before
    assert cli.main(argv) == 0
    (output / "partitioned_collision.usd").write_bytes(b"changed")
    capsys.readouterr()
    assert cli.main(argv) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert (source.read_bytes(), labels.read_bytes()) == before


@pytest.mark.parametrize("ids", [[], ["subject"], ["subject", "subject"], ["subject", "missing"]])
def test_partition_refuses_invalid_selection_before_output(cli, tmp_path, ids, capsys):
    from tests.test_sage_collision_partition import combined_scene

    source, labels = combined_scene(tmp_path)
    output = tmp_path / "partition"
    argv = ["sage-collision-partition", "--source", str(source), "--labels", str(labels),
            "--output-root", str(output)]
    for instance_id in ids:
        argv.extend(["--instance-id", instance_id])
    assert cli.main(argv) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    assert not output.exists()


def _observation_args(tmp_path):
    from blueprint_pipeline.website_task_preparation import compile_website_scene_preparation
    from tests.test_website_task_preparation import _arguments

    inputs = _arguments(tmp_path)
    preparation = compile_website_scene_preparation(**inputs)
    output = tmp_path / "observations"
    argv = ["object-observations", "--preparation", str(_json(tmp_path / "preparation.json", preparation)),
            "--source-geometry", str(_json(tmp_path / "geometry.json", inputs["source_geometry"])),
            "--task-masks", str(_json(tmp_path / "masks.json", inputs["task_masks"])),
            "--output-root", str(output)]
    return inputs, preparation, argv, output


def test_observation_dispatch_retains_estimated_scope_and_reopens_sources(cli, tmp_path, capsys):
    inputs, preparation, argv, output = _observation_args(tmp_path)
    before = _snapshot(tmp_path)
    assert cli.main(argv) == 0
    manifest_path = output / preparation["digest"][7:] / "observations.json"
    value = json.loads(manifest_path.read_text())
    assert value["digest"] == canonical_digest(value, digest_field="digest")
    assert value["claim_ceiling"] == "development_only"
    assert value["physical_measurement_proven"] is value["complete_object_geometry"] is False
    for path, original in before.items():
        assert (tmp_path / path).read_bytes() == original
    assert cli.main(argv) == 0
    Path(inputs["source_geometry"]["frames"][0]["image_path"]).write_bytes(b"changed")
    capsys.readouterr()
    assert cli.main(argv) == 2
    assert "source_changed" in json.loads(capsys.readouterr().out)["blockers"][0]


def test_observation_dispatch_refuses_resealed_wrong_source_binding(cli, tmp_path, capsys):
    _, preparation, argv, output = _observation_args(tmp_path)
    preparation["binding"]["source_geometry_digest"] = "sha256:" + "c" * 64
    preparation["digest"] = canonical_digest(preparation, digest_field="digest")
    _json(tmp_path / "preparation.json", preparation)
    assert cli.main(argv) == 2
    assert "source_mismatch" in json.loads(capsys.readouterr().out)["blockers"][0]
    assert not output.exists()


def _website_args(tmp_path):
    from tests.test_website_native_inputs import packet

    envelope, configurations = packet(tmp_path)
    output = tmp_path / "website-inputs"
    argv = ["website-inputs", "--envelope", str(_json(tmp_path / "envelope.json", envelope)),
            "--stage-one-configuration", str(_json(tmp_path / "stage-one.json", configurations["stage-1"])),
            "--output-root", str(output)]
    return envelope, configurations, argv, output


def test_website_dispatch_reopens_full_binding_and_preserves_no_spend_scope(cli, tmp_path, capsys):
    _, _, argv, output = _website_args(tmp_path)
    before = _snapshot(tmp_path)
    assert cli.main(argv) == 0
    result = json.loads((output / "task_evaluation_scene_configuration_render_inputs.v1.json").read_text())
    assert result["result_digest"] == canonical_digest(result, digest_field="result_digest")
    assert result["provider_mutation_performed"] is result["paid_execution_requested"] is False
    assert result["physical_truth_claimed"] is result["renderer_qualified"] is False
    assert result["website_binding"]["captured_frame_count"] > 0
    assert json.loads(capsys.readouterr().out)["result_digest"] == result["result_digest"]
    for path, original in before.items():
        assert (tmp_path / path).read_bytes() == original


@pytest.mark.parametrize("corruption", ["configuration", "runtime_bytes", "disclosure"])
def test_website_dispatch_refuses_changed_contracts_before_output(cli, tmp_path, corruption, capsys):
    envelope, configurations, argv, output = _website_args(tmp_path)
    if corruption == "configuration":
        config = {**configurations["stage-1"], "marker": "changed"}
        _json(tmp_path / "stage-one.json", config)
    elif corruption == "runtime_bytes":
        row = next(row for row in envelope["materialized_references"] if row["contract_path"].endswith(".runtime_inputs"))
        Path(row["materialized_path"]).write_bytes(b"changed")
    else:
        envelope["request"]["scene"]["rights"]["provider_disclosure_scope"] = "raw_capture"
        _json(tmp_path / "envelope.json", envelope)
    assert cli.main(argv) == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "blocked"
    predicate = {
        "configuration": "website_native_inputs_stage_one_changed",
        "runtime_bytes": "scene_configuration_materialized_reference_invalid",
        "disclosure": "website_native_inputs_disclosure_scope_invalid",
    }[corruption]
    assert predicate in summary["blockers"][0]
    assert not output.exists()


def test_cli_refuses_bad_json_and_missing_required_flags(cli, tmp_path, capsys):
    malformed = tmp_path / "sources.json"
    malformed.write_text("{not JSON")
    assert cli.main(["retained-selection-from-sources", "--sources", str(malformed),
                     "--output-root", str(tmp_path / "output")]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
    with pytest.raises(SystemExit) as exc:
        cli.main(["website-inputs"])
    assert exc.value.code == 2
    assert not (tmp_path / "output").exists()


@pytest.mark.slow
def test_script_entrypoint_dispatches_real_candidate_selection(tmp_path):
    source = _retained_source(tmp_path / "source", "camera-1")
    output = tmp_path / "selected"
    result = subprocess.run([sys.executable, str(SCRIPT), *_selection_args(
        "retained-selection", [source], output, tmp_path)], cwd=REPO_ROOT,
        capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["receipt_path"] == str(output / "selection.json")
    assert (output / "selection.json").is_file()
