"""Portable original descriptor/accounting contracts; no Linux clearance proof."""
import os

import pytest

from tests.test_owner_target_version_publication import root_metadata, protected_root_tmp_path  # noqa: F401


@pytest.mark.parametrize('change', ['none', 'replacement'])
def test_reference_rereads_retain_one_original_descriptor_and_charge_actual_bytes(
    protected_root_tmp_path, root_metadata, change  # noqa: F811
):
    from blueprint_pipeline.control_plane_lane_disk_diagnostic_references import _ReferenceFiles
    from blueprint_pipeline.control_plane_lane_experiment_work import _ActionFiles
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    path = protected_root_tmp_path / 'selected.json'
    raw = b'{"observed":true}\n'
    path.write_bytes(raw)
    path.chmod(0o600)
    action = _ActionFiles(now=lambda: 1000)
    budget = ReferenceCollectionBudget()
    observer = _ReferenceFiles(budget, action)
    try:
        first, record = observer.read(path, cap=64, protected=True, mode=0o600)
        owned = dict(observer.owned)
        assert first == raw
        if change == 'replacement':
            foreign = path.with_name('owned-foreign-fixture.json')
            foreign.write_bytes(raw)
            foreign.chmod(0o600)
            foreign.replace(path)
            with pytest.raises(ValueError):
                observer.read(path, cap=64, protected=True, mode=0o600)
            assert path.read_bytes() == raw
            assert observer.owned == owned
        else:
            for _ in range(30):
                observed, repeated = observer.read(path, cap=64, protected=True, mode=0o600)
                assert observed == raw and repeated is record
            assert observer.owned == owned and len(observer.records) == 1
            assert budget.counts['raw_bytes'] == len(raw) * 31
            budget.close()
            with pytest.raises(ValueError, match='reference_budget_closed'):
                observer.read(path, cap=64, protected=True, mode=0o600)
        assert os.fstat(record.fd).st_ino == record.info.st_ino
    finally:
        observer.finish()
        action.finish()
        budget.close()
        action.budget.close()
