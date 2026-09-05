"""Resume a closed GPU stage across CPU publication interruptions, never reallocate."""
import json
from pathlib import Path

import pytest

from blueprint_pipeline import sam31_source_calibration_stage as stage
from blueprint_pipeline.source_calibration_render_return import record, materialize_source_calibration_closed_return
from tests.test_retained_source_calibration_lifecycle import _closed_source_fixture, _seal_json


def _closed_job(tmp_path, monkeypatch):
    prepared, render_return, closure, _evidence = _closed_source_fixture(tmp_path, monkeypatch)
    root = tmp_path / "phase"
    root.mkdir()
    prepared_path = Path(prepared["preparation_path"])
    (root / "cpu_preparation_outcome.json").write_text(json.dumps({
        "prepared_inputs": record(prepared_path), "calibrated_view_request": prepared["request_file"]}))
    (root / "bundle").mkdir()
    (root / "bundle" / stage.RECEIPT_NAME).write_text("{}")
    (root / "paid_attempt_authority.json").write_text("{}")
    execution = json.loads(Path(closure["provider_execution"]["path"]).read_text())
    execution.update(source_calibration_return={"return_path": str(render_return)},
                     independent_watchdog={"provider_absence_confirmed": True})
    _seal_json(root / "allocator_result.json", execution, "receipt_digest")
    closure["provider_execution"] = record(root / "allocator_result.json")
    materialize_source_calibration_closed_return(prepared_inputs=prepared, returned_group_path=render_return,
        execution_closure=closure, output_path=root / "source_calibration_closed_return.v1.json")
    monkeypatch.setattr(stage, "_posted_charge", lambda *_: Path(closure["official_billing"]["path"]))
    task = root / "task.json"
    task.write_text(json.dumps({"human_authority": {"accepted_by": "fixture-owner"}}))
    avoidlist = root / "avoidlist.json"
    avoidlist.write_text(json.dumps({"machine_ids": []}))
    job = {"output_root": str(root), "repo_root": prepared["context"]["paths"]["repo"],
           "expected_source_commit": prepared["repository"]["commit"], "resume_only": True,
           "plan": {"host_inputs": {"task_request": record(task)}},
           "server_profile": {"calibrated_views": {"execution_site": "provider_gpu", "hardware_required": True,
               "max_spend_usd": 1., "hard_ttl_seconds": 1800, "max_hourly_rate_usd": .5, "retry_cap": 0,
               "maximum_resource_count": 1, "allowed_geolocation_country_codes": ["US"], "machine_avoidlist": record(avoidlist)}}}
    return job, prepared, root


def _no_allocation(*args, **kwargs):
    pytest.fail("a retained closed GPU run must not allocate again")


def test_terminal_cpu_receipt_is_adopted_after_checkpoint_write_failure(tmp_path, monkeypatch):
    from blueprint_pipeline import public_scene_inpainting_inputs as cpu
    job, prepared, root = _closed_job(tmp_path, monkeypatch)
    finalize = cpu.finalize_public_scene_inpainting_inputs
    def fail_after_terminal_receipt(**kwargs):
        finalize(**kwargs)
        raise OSError("injected CPU checkpoint write failure")
    monkeypatch.setattr(cpu, "finalize_public_scene_inpainting_inputs", fail_after_terminal_receipt)
    with pytest.raises(OSError, match="checkpoint write failure"):
        stage.execute_source_calibration_stage(job, allocator_runner=_no_allocation)
    terminal = Path(prepared["preparation_path"]).parent / "public_scene_interiorgs_edit_input_receipt.v2.json"
    before = terminal.read_bytes()
    assert not (root / "cpu_finalization_outcome.json").exists()
    monkeypatch.setattr(cpu, "finalize_public_scene_inpainting_inputs", finalize)
    outcome = stage.execute_source_calibration_stage(job, allocator_runner=_no_allocation)
    assert outcome["status"] == "completed"
    assert terminal.read_bytes() == before


@pytest.mark.parametrize("corrupt", [False, True])
def test_partial_verified_copy_resumes_without_reallocation_or_overwriting_mismatch(tmp_path, monkeypatch, corrupt):
    from blueprint_pipeline import public_scene_inpainting_preparation as cpu
    job, prepared, _root = _closed_job(tmp_path, monkeypatch)
    copied = []
    copy = cpu.shutil.copyfile
    def fail_after_copy(source, target):
        copy(source, target)
        copied.append(Path(target))
        if len(copied) == 2:
            raise OSError("injected partial copy failure")
    monkeypatch.setattr(cpu.shutil, "copyfile", fail_after_copy)
    with pytest.raises(OSError, match="partial copy"):
        stage.execute_source_calibration_stage(job, allocator_runner=_no_allocation)
    monkeypatch.setattr(cpu.shutil, "copyfile", copy)
    first_png = copied[-1]
    before = first_png.read_bytes()
    if corrupt:
        first_png.write_bytes(before + b"changed")
        with pytest.raises(ValueError, match="artifact_conflict"):
            stage.execute_source_calibration_stage(job, allocator_runner=_no_allocation)
        assert first_png.read_bytes() == before + b"changed"
    else:
        outcome = stage.execute_source_calibration_stage(job, allocator_runner=_no_allocation)
        assert outcome["status"] == "completed"
        assert first_png.read_bytes() == before
        assert len(list(Path(prepared["preparation_path"]).parent.glob("*/frames/*.png"))) == 48


@pytest.mark.parametrize("corrupt", ["request", "closed_return", "mask"])
def test_terminal_receipt_adoption_reopens_exact_bindings_and_mask_pixels(tmp_path, monkeypatch, corrupt):
    from PIL import Image
    job, prepared, _root = _closed_job(tmp_path, monkeypatch)
    stage.execute_source_calibration_stage(job, allocator_runner=_no_allocation)
    terminal = Path(prepared["preparation_path"]).parent / "public_scene_interiorgs_edit_input_receipt.v2.json"
    receipt = json.loads(terminal.read_text())
    if corrupt == "request":
        receipt["request_digest"] = "sha256:" + "b" * 64
    elif corrupt == "closed_return":
        receipt["source_calibration_render"]["return_digest"] = "sha256:" + "b" * 64
    else:
        mask_row = receipt["derived_artifacts"]["masks"][0]
        mask = terminal.parent / mask_row["relative_path"]
        with Image.open(mask) as image:
            changed = image.convert("L")
        changed.putpixel((0, 0), 255 if changed.getpixel((0, 0)) == 0 else 0)
        changed.save(mask)
        mask_row.update(sha256=record(mask)["sha256"], size_bytes=mask.stat().st_size)
    _seal_json(terminal, receipt, "receipt_digest")
    before = terminal.read_bytes()
    with pytest.raises(ValueError, match="retained_(receipt_binding_mismatch|mask_changed)"):
        stage.execute_source_calibration_stage(job, allocator_runner=_no_allocation)
    assert terminal.read_bytes() == before
