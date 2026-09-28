# Covers (for impacted-test selection): src/blueprint_pipeline/control_plane_storage_roots.py
"""Producer-owned mixed preparation roots are work; SAM global law stays cache."""
from blueprint_pipeline.control_plane_storage_roots import classify_path


def test_installer_preparation_roots_have_explicit_work_law():
    for name in ('completed-scene-preparation', 'completed-scene-preparation-inputs'):
        path = '/var/lib/blueprint/task-evaluation-inputs/'+name
        row = classify_path(path+'/owned-child')
        assert row.path == path and row.storage_class == 'work'
    assert classify_path('/var/lib/blueprint/task-evaluation-inputs/sam31-preparations/child').storage_class == 'cache'
