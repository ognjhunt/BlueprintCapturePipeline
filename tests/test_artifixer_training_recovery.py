import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.test_public_scene_artifixer3d_dual_target_runner import _runner_module, _request


def test_completed_training_export_refusal_survives_provider_archive(tmp_path, monkeypatch):
    runner = _runner_module()
    output = tmp_path / "round/artifixer_output"
    task_output = output / "tasks/task_a"
    log = task_output / "logs/artifixer3d_dual_target.log"
    log.parent.mkdir(parents=True)
    log.write_text("training started\n")
    reference = tmp_path / "reference.ply"
    reference.write_bytes(b"retained-reference")
    checkpoint = task_output / "ckpt_30000.pt"
    prepared = {
        "staged_task": tmp_path,
        "scene": SimpleNamespace(scene_id="scene"),
        "steps": 30000,
        "paths": SimpleNamespace(
            distillation_input_dir=tmp_path,
            run_root=task_output,
            distillation_selected_indices_path=tmp_path / "indices",
            override_image_dir=tmp_path / "images",
        ),
    }
    monkeypatch.setattr(
        runner, "_prepare_dual_target_distillation_replay", lambda **kwargs: prepared
    )
    monkeypatch.setattr(runner, "_retained_reference_gaussian_ply", lambda _: reference)
    package = ModuleType("data_processing")
    package.artifixer3d = SimpleNamespace(artifixer3d_checkpoint=lambda *args: checkpoint)

    def train(*args):
        checkpoint.write_bytes(b"completed-optimization-checkpoint-with-config")
        print("Training Complete 30000")

    package.threedgrut_training = SimpleNamespace(
        train_3dgrut=train, DEFAULT_THREEDGRUT_CONFIG_DIR=tmp_path
    )
    monkeypatch.setitem(sys.modules, "data_processing", package)
    error = ValueError("artifixer3d_native_export_gaussian_field_drift_invalid:outlier")
    error.geometry_quality = {"status": "blocked", "metrics": {"max_center_ratio": 827}}

    def refuse(**kwargs):
        recovery = output.parent / "retained_training_evidence/task_a"
        assert (recovery / "checkpoint.pt").read_bytes() == checkpoint.read_bytes()
        raise error

    monkeypatch.setattr(runner, "_export_checkpoint_native_appearance", refuse)
    with pytest.raises(ValueError, match="gaussian_field_drift_invalid"):
        runner._dual_target_task_runtime(
            task={"task_id": "task_a"},
            input_root=tmp_path,
            source_root=tmp_path,
            output_root=output,
            request=_request(runner),
        )
    # Use the actual provider output filter, extracted from its generated program.
    import ast

    adapter = Path(runner.__file__).parents[1] / "src/blueprint_pipeline/vast_provider_adapter.py"
    constants = [
        n.value
        for n in ast.walk(ast.parse(adapter.read_text()))
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and "excluded_parts = {" in n.value
    ]
    line = next(
        line
        for text in constants
        for line in text.splitlines()
        if line.startswith("excluded_parts = {")
    )
    excluded = ast.literal_eval(line.partition(" = ")[2])
    archive = tmp_path / "output.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in output.parent.rglob("*"):
            if path.is_file() and excluded.isdisjoint(path.relative_to(output.parent).parts):
                bundle.write(path, path.relative_to(output.parent))
    with zipfile.ZipFile(archive) as bundle:
        prefix = "retained_training_evidence/task_a/"
        assert bundle.read(prefix + "checkpoint.pt") == checkpoint.read_bytes()
        receipt = json.loads(bundle.read(prefix + "training_recovery.json"))
        assert receipt["optimization_complete"] and not receipt["export_qualified"]
        assert (
            receipt["files"]["checkpoint.pt"]["sha256"]
            == "sha256:" + hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        )
        outcome = json.loads(bundle.read(prefix + "export_outcome.json"))
        assert outcome["geometry_quality"] == error.geometry_quality
        assert b"Training Complete 30000" in bundle.read(prefix + "training.log")
    monkeypatch.setattr(
        sys, "argv", ["runner", "--bundle-root", str(tmp_path), "--output-root", str(output)]
    )
    monkeypatch.setattr(runner, "execute", refuse)
    assert runner.main() == 2
    result = json.loads((output / "public_scene_artifixer3d_runtime_result.json").read_text())
    assert result["completed_task_count"] == 0
    assert result["optimization_complete_task_count"] == 1
    assert result["partial_task_evidence_preserved"] is True


def test_review_rejection_archive_keeps_exact_exports_and_review_pngs(tmp_path):
    import ast
    from blueprint_pipeline.artifixer_training_recovery import (
        retain_native_exports,
        retain_review_frames,
    )
    from blueprint_pipeline.artifixer_source_geometry_admission import _record

    original = tmp_path / "round/artifixer_output"
    original.mkdir(parents=True)
    native = {"status": "export_completed_requires_review"}
    for key, name in (
        ("standard_gaussian_ply", "repaired_scene.ply"),
        ("isaac_nurec_usdz", "repaired_scene.usdz"),
    ):
        path = original / name
        path.write_bytes((name + "-exact-native-bytes").encode())
        native[key] = _record(path)
    frame = original / "00000.png"
    frame.write_bytes(b"exact-review-image-bytes")
    rows = [{"frame_index": 0, "camera_id": "source-01", **_record(frame)}]
    recovery = original.parent / "retained_training_evidence/task"
    retain_native_exports(recovery, native)
    retain_review_frames(recovery, rows)
    (original.parent / "review_rejection.json").write_text('{"decision":"rejected"}')
    adapter = Path(__file__).parents[1] / "src/blueprint_pipeline/vast_provider_adapter.py"
    strings = [
        n.value
        for n in ast.walk(ast.parse(adapter.read_text()))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    line = next(
        line
        for text in strings
        for line in text.splitlines()
        if line.startswith("excluded_parts = {")
    )
    excluded = ast.literal_eval(line.partition(" = ")[2])
    with zipfile.ZipFile(tmp_path / "returned.zip", "w") as bundle:
        for path in original.parent.rglob("*"):
            if path.is_file() and excluded.isdisjoint(path.relative_to(original.parent).parts):
                bundle.write(path, path.relative_to(original.parent))
    with zipfile.ZipFile(tmp_path / "returned.zip") as bundle:
        prefix = "retained_training_evidence/task/"
        for name in ("repaired_scene.ply", "repaired_scene.usdz"):
            assert bundle.read(prefix + "native_exports/" + name) == (original / name).read_bytes()
        assert bundle.read(prefix + "review_frames/00000.png") == frame.read_bytes()
        receipt = json.loads(bundle.read(prefix + "review_frames/review_frame_recovery.json"))
        assert receipt["frames"][0]["retained_frame"]["sha256"] == rows[0]["sha256"]
        assert receipt["physical_or_deployment_evidence"] is False
