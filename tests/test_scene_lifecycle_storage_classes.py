# Covers (for impacted-test selection): src/blueprint_pipeline/control_plane_storage_roots.py
"""Producer-owned mixed preparation roots are work; SAM global law stays cache."""
from blueprint_pipeline.control_plane_storage_roots import classify_path
import pytest


def test_installer_preparation_roots_have_explicit_work_law():
    for name in ('completed-scene-preparation', 'completed-scene-preparation-inputs'):
        path = '/var/lib/blueprint/task-evaluation-inputs/'+name
        row = classify_path(path+'/owned-child')
        assert row.path == path and row.storage_class == 'work'
    assert classify_path('/var/lib/blueprint/task-evaluation-inputs/sam31-preparations/child').storage_class == 'cache'


@pytest.mark.slow
def test_actual_installer_config_roots_follow_the_same_storage_law(tmp_path, monkeypatch):
    from pathlib import Path
    from tests.test_scene_preparation_installation import _installed
    _, _, config, _ = _installed(tmp_path, monkeypatch)
    prefix = str(Path(config['factory_output_root']).parent)
    canonical = '/var/lib/blueprint/task-evaluation-inputs'
    for actual in (config['factory_output_root'], config['preparation_worker']['input_root']):
        owned = actual.replace(prefix, canonical, 1)
        assert classify_path(owned+'/owned-child').storage_class == 'work'
    assert classify_path(config['child_execution_root'].replace(prefix, canonical, 1)+'/child').storage_class == 'cache'
