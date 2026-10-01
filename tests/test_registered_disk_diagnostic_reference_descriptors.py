"""Portable original descriptor/accounting contracts; no Linux clearance proof."""
import os

import pytest

from tests.test_owner_target_version_publication import root_metadata, protected_root_tmp_path  # noqa: F401


def test_native_limit_observer_rethrows_actual_refusal_and_retains_counters(
    protected_root_tmp_path, root_metadata  # noqa: F811
):
    from blueprint_pipeline.control_plane_lane_disk_diagnostic_references import _ReferenceFiles
    from blueprint_pipeline.control_plane_lane_experiment_work import _ActionFiles
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    from tests.test_registered_feature_linux import _retain_native_limit
    path = protected_root_tmp_path / 'limit.json'
    path.write_bytes(b'12345678')
    path.chmod(0o600)
    action = _ActionFiles(now=lambda: 1000)
    budget = ReferenceCollectionBudget()
    observer = _ReferenceFiles(budget, action)
    observer.raw_cap = 8  # Lowered CPU boundary, never a native allowance.
    failures = []
    observer.read_bytes = _retain_native_limit(_ReferenceFiles.read_bytes, failures).__get__(observer)
    try:
        with pytest.raises(ValueError, match='owner_target_resource_exhausted'):
            observer.read(path, cap=8, protected=True, mode=0o600)
        assert len(failures) == 1
        row = failures[0]
        assert row['metadata_bytes'] == row['budget_counts']['raw_bytes'] == row['raw_cap'] == 8
        assert row['budget_limits'] == dict(budget.limits)
        assert row['budget_deadline'] == budget.deadline
        assert any(frame['function'] == 'read_bytes' for frame in row['frames'])
        assert path.read_bytes() == b'12345678'
    finally:
        observer.finish()
        action.finish()
        budget.close()
        action.budget.close()


def test_shared_actual_scan_bytes_do_not_consume_metadata_owner_allowance(
    protected_root_tmp_path, root_metadata  # noqa: F811
):
    from blueprint_pipeline.control_plane_lane_disk_diagnostic_references import _ReferenceFiles
    from blueprint_pipeline.control_plane_lane_experiment_work import _ActionFiles
    from blueprint_pipeline.control_plane_lane_historical_processes import _Scan
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    path = protected_root_tmp_path / 'current.json'
    raw = b'{"current":true}\n'
    path.write_bytes(raw)
    path.chmod(0o600)
    observed = protected_root_tmp_path / 'scan-bytes'
    observed.write_bytes(b'x' * 65536)
    action = _ActionFiles(now=lambda: 1000)
    budget = ReferenceCollectionBudget()
    observer = _ReferenceFiles(budget, action)
    directory = os.open(protected_root_tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        first, record = observer.read(path, cap=64, protected=True, mode=0o600)
        owned = dict(observer.owned)
        scan = _Scan(budget.tick, budget)
        for _ in range(33):
            assert scan.read(directory, observed.name) == b'x' * 65536
        assert scan.raw_bytes > observer.raw_cap
        again, selected = observer.read(path, cap=64, protected=True, mode=0o600)
        assert first == again == raw and selected is record
        assert observer.owned == owned
        assert budget.counts['raw_bytes'] == scan.raw_bytes + 2 * len(raw)
        assert budget.counts['raw_bytes'] < budget.limits['raw_bytes']
    finally:
        os.close(directory)
        observer.finish()
        action.finish()
        budget.close()
        action.budget.close()


@pytest.mark.parametrize('limit', ['metadata', 'shared'])
def test_reference_metadata_rereads_preserve_both_original_byte_limits(
    protected_root_tmp_path, root_metadata, limit  # noqa: F811
):
    from blueprint_pipeline.control_plane_lane_disk_diagnostic_references import _ReferenceFiles
    from blueprint_pipeline.control_plane_lane_experiment_work import _ActionFiles
    from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget
    path = protected_root_tmp_path / 'bounded.json'
    raw = b'x' * 1024
    path.write_bytes(raw)
    path.chmod(0o600)
    action = _ActionFiles(now=lambda: 1000)
    budget = ReferenceCollectionBudget(monotonic=lambda: 1000)
    observer = _ReferenceFiles(budget, action)
    try:
        observer.read(path, cap=1024, protected=True, mode=0o600)
        owned = dict(observer.owned)
        if limit == 'metadata':
            # The production allowance remains exactly two MiB. Repeated
            # original-FD observations consume it without any refund/reset.
            assert observer.raw_cap == 2 * 1024**2
            for _ in range(2046):
                observer.read(path, cap=1024, protected=True, mode=0o600)
        else:
            # CPU counter boundary only; this supplies no kernel observation.
            budget.charge('raw_bytes', budget.limits['raw_bytes'] - budget.counts['raw_bytes'])
        before = budget.counts['raw_bytes']
        with pytest.raises(ValueError):
            observer.read(path, cap=1024, protected=True, mode=0o600)
        assert observer.owned == owned and path.read_bytes() == raw
        assert budget.counts['raw_bytes'] >= before
        assert budget.counts['raw_bytes'] <= budget.limits['raw_bytes']
    finally:
        observer.finish()
        action.finish()
        budget.close()
        action.budget.close()


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
