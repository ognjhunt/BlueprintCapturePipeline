"""Import-time derived data must invalidate independently of function tracing."""
from blueprint_pipeline import validation_code_dependencies as dependencies
from tests.test_speedup_audit_regressions import probe_modules  # noqa: F401


def test_import_time_helper_dependencies_are_bound(probe_modules):  # noqa: F811
    probe_modules('audit_source_rule', 'LIMIT = 1\n')
    probe_modules('audit_derived_rule',
        'from blueprint_pipeline.audit_source_rule import LIMIT\n'
        'def build():\n    return LIMIT * 2\nTHRESHOLD = build()\n')
    assert 'blueprint_pipeline.audit_source_rule' in dependencies.data_closure(['blueprint_pipeline.audit_derived_rule'])


def test_unused_local_import_is_not_import_time_data(probe_modules):  # noqa: F811
    probe_modules('audit_idle_rule',
        'def unused():\n    from blueprint_pipeline import control_plane_storage_gc\n    return control_plane_storage_gc\nLIMIT = 1\n')
    assert 'blueprint_pipeline.control_plane_storage_gc' not in dependencies.data_closure(['blueprint_pipeline.audit_idle_rule'])
