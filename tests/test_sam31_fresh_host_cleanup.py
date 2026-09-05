"""A fresh release must survive an idle SAM worker and its real cleanup hook."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import stat

import pytest

from blueprint_pipeline import task_evaluation_sam31_preparation_execution as worker
from blueprint_pipeline.task_evaluation_dispatcher_cgroup_cleanup import (
    DispatcherCgroupCleanupError, cleanup_dispatcher_cgroup,
)
from tests.test_deploy_control_plane_commit import _stage_real_units, deploy


SERVICE = 'blueprint-task-evaluation-sam31-preparation-execution.service'


def test_fresh_deploy_provisions_idle_sam_worker_and_exec_stop_roots(tmp_path, monkeypatch):
    release = _stage_real_units(tmp_path)
    unit = (release/'deploy/systemd'/SERVICE).read_text()
    stop = next(line for line in unit.splitlines() if line.startswith('ExecStopPost='))
    command = shlex.split(shlex.split(stop.partition('=')[2])[-1])
    state = Path(command[command.index('--state-root')+1])
    receipts = Path(command[command.index('--receipt-dir')+1])
    assert state == worker.DEFAULT_EXECUTION_ROOT
    assert receipts.parent == worker.DEFAULT_QUEUE
    assert 'KillMode=process' in unit
    host = tmp_path/'host'
    # This is existing host configuration, not a service state directory that
    # the release installer may silently invent.
    (host/'etc/blueprint/task-evaluation-launch-profiles').mkdir(parents=True)
    ids = (os.getuid(), os.getgid())
    result = deploy._install_unit_sandbox_paths(release_path=release, units=(SERVICE,),
        root_prefix=host, owner_ids=ids)
    created = {row['path'] for row in result['created']}
    expected = (state, worker.DEFAULT_QUEUE, receipts)
    assert {str(path) for path in expected} <= created
    for path in expected:
        installed = host/str(path).lstrip('/')
        metadata = installed.stat()
        assert installed.is_dir() and not installed.is_symlink()
        assert (metadata.st_uid, metadata.st_gid) == ids
        assert stat.S_IMODE(metadata.st_mode) == 0o750
    monkeypatch.setattr(worker, '_verified_checkout_head', lambda: 'a'*40)
    idle = worker.process_sam31_phase_queue(
        queue_root=host/str(worker.DEFAULT_QUEUE).lstrip('/'),
        execution_root=host/str(state).lstrip('/'), source_commit='a'*40,
        phase_executor=lambda _: pytest.fail('an idle service must not execute a phase'))
    assert idle['status'] == 'idle' and idle['results'] == []
    cgroup = tmp_path/'cgroup.procs'
    cgroup.write_text('999\n')
    state_root = host/str(state).lstrip('/')
    receipt_dir = host/str(receipts).lstrip('/')
    cleanup = cleanup_dispatcher_cgroup(state_root=state_root, receipt_dir=receipt_dir,
        cgroup_procs_path=cgroup, proc_root=tmp_path/'proc', self_pid=999,
        killer=lambda *_: pytest.fail('idle cleanup must signal no process'))
    assert cleanup['status'] == 'reconciled' and cleanup['blockers'] == []
    assert cleanup['cgroup_process_count'] == 0 and cleanup['preserved_watchdogs'] == []
    assert len(list(receipt_dir.glob('dispatcher-cgroup-cleanup-*.json'))) == 1
    # Provisioning fixes the missing root; the cleanup guard still refuses a
    # genuinely absent root rather than calling that state successful.
    state_root.rmdir()
    with pytest.raises(DispatcherCgroupCleanupError, match='dispatcher_cgroup_cleanup_root_invalid'):
        cleanup_dispatcher_cgroup(state_root=state_root, receipt_dir=receipt_dir,
                                  cgroup_procs_path=cgroup, self_pid=999)
