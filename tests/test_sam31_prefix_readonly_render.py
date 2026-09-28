"""A retained render reader cannot silently repair historical output files."""
from pathlib import Path

import pytest

from blueprint_pipeline.public_scene_inpainting_preparation import adopt_finalized_public_scene_inpainting_inputs
from blueprint_pipeline import sam31_source_calibration_stage as stage
from tests.test_source_calibration_finalization_reentry import _closed_job, _no_allocation


@pytest.mark.parametrize('missing', ['frame', 'manifest'])
def test_retained_render_validation_refuses_missing_files_without_recreating_them(tmp_path, monkeypatch, missing):
    job, prepared, root = _closed_job(tmp_path, monkeypatch)
    stage.execute_source_calibration_stage(job, allocator_runner=_no_allocation)
    output = Path(prepared['preparation_path']).parent
    target = (next((output / 'images/frames').glob('*.png')) if missing == 'frame'
              else output / 'images/sealed_camera_render_manifest.v1.json')
    target.unlink()
    before = {path: path.read_bytes() for path in output.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match='retained_artifact_missing'):
        adopt_finalized_public_scene_inpainting_inputs(preparation_path=prepared['preparation_path'],
            returned_group_path=root / 'source_calibration_closed_return.v1.json')
    assert not target.exists()
    assert {path: path.read_bytes() for path in output.rglob('*') if path.is_file()} == before


@pytest.mark.slow
def test_recovered_camera_file_is_not_recreated_by_readonly_adoption(tmp_path, monkeypatch):
    from blueprint_pipeline import public_scene_inpainting_preparation as preparation
    from tests.test_source_calibration_camera_resolution import (
        test_actual_packet_worker_and_host_agree_on_bounded_repair as run_fixture,
    )
    real = preparation.adopt_finalized_public_scene_inpainting_inputs
    observed = {}
    def capture(**kwargs):
        observed.update(kwargs)
        return real(**kwargs)
    monkeypatch.setattr(preparation, 'adopt_finalized_public_scene_inpainting_inputs', capture)
    run_fixture(tmp_path, monkeypatch)
    root = Path(observed['preparation_path']).parent
    target = root / 'resolved_cameras.json'
    target.unlink()
    before = {path: path.read_bytes() for path in root.rglob('*') if path.is_file()}
    with pytest.raises(ValueError, match='retained_artifact_missing'):
        real(**observed)
    assert not target.exists()
    assert {path: path.read_bytes() for path in root.rglob('*') if path.is_file()} == before
