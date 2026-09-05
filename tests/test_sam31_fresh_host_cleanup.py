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


def test_parent_preparation_sandbox_can_publish_sam_child_jobs(tmp_path):
    import hashlib
    from blueprint_pipeline.task_evaluation_sam31_phase_queue import enqueue_sam31_phase
    from blueprint_pipeline.task_evaluation_scene_configuration_sam31_preparation_driver import DEFAULT_CHILD_QUEUE
    release = _stage_real_units(tmp_path)
    service = 'blueprint-task-evaluation-launch-preparation.service'
    text = (release/'deploy/systemd'/service).read_text()
    writable = {path for path, optional, directive in deploy._unit_sandbox_entries(text)
                if directive == 'ReadWritePaths' and not optional}
    # This exact path was read-only in the actual service namespace even
    # though the same blueprint account could write it outside that namespace.
    assert str(DEFAULT_CHILD_QUEUE) in writable
    host = tmp_path/'host'
    ids = (os.getuid(), os.getgid())
    provisioned = deploy._install_unit_sandbox_paths(release_path=release,
        units=(service,), root_prefix=host, owner_ids=ids)
    assert str(DEFAULT_CHILD_QUEUE) in {row['path'] for row in provisioned['created']}
    queue = host/str(DEFAULT_CHILD_QUEUE).lstrip('/')
    assert (queue.stat().st_uid, queue.stat().st_gid) == ids
    assert stat.S_IMODE(queue.stat().st_mode) == 0o750
    plan = host/'var/lib/blueprint/task-evaluation-inputs/fixture-plan.json'
    plan.write_text('{"fixture_only":true}')
    ref = {'path':str(plan), 'sha256':'sha256:'+hashlib.sha256(plan.read_bytes()).hexdigest(),
           'size_bytes':plan.stat().st_size}
    result = enqueue_sam31_phase(queue_root=queue, parent_preparation_id='fixture-preparation',
        parent_request_digest='sha256:'+'a'*64, expected_source_commit='a'*40,
        plan_ref=ref, phase='source_selections', inputs={})
    assert result['status'] == 'queued' and Path(result['job_path']).is_file()
    assert not Path(result['result_path']).exists()  # Publication never executes the child.


def test_sam_paid_phase_can_consume_existing_ledger_without_resetting_it(tmp_path, monkeypatch):
    from blueprint_pipeline.adp_retained_scene_render_vast import consume_retained_scene_render_paid_attempt_authority_once
    from blueprint_pipeline.spend_authority_consumption_root import SPEND_AUTHORITY_ROOT_ENV
    release = _stage_real_units(tmp_path)
    text = (release/'deploy/systemd'/SERVICE).read_text()
    ledger_root = Path('/var/lib/blueprint/spend-authority')
    writable = {path for path, optional, directive in deploy._unit_sandbox_entries(text)
                if directive == 'ReadWritePaths' and not optional}
    assert str(ledger_root) in writable
    host = tmp_path/'host'
    (host/'etc/blueprint/task-evaluation-launch-profiles').mkdir(parents=True)
    ledger = host/str(ledger_root).lstrip('/')
    consumed = ledger/'consumed'
    consumed.mkdir(parents=True)
    consumed.chmod(0o700)
    previous = consumed/'retained-existing-consumption.json'
    previous.write_bytes(b'previous immutable consumption evidence')
    previous.chmod(0o600)
    deploy._install_unit_sandbox_paths(release_path=release, units=(SERVICE,),
        root_prefix=host, owner_ids=(os.getuid(), os.getgid()))
    monkeypatch.setenv(SPEND_AUTHORITY_ROOT_ENV, str(ledger))
    authority = {'authorization_digest':'sha256:'+'a'*64, 'bundle_sha256':'sha256:'+'b'*64}
    first = consume_retained_scene_render_paid_attempt_authority_once(authority, blueprint_commit='c'*40)
    assert first['status'] == 'consumed'
    second = consume_retained_scene_render_paid_attempt_authority_once(authority, blueprint_commit='c'*40)
    assert second == {'status':'blocked', 'blockers':['attempt_authority_already_consumed']}
    assert previous.read_bytes() == b'previous immutable consumption evidence'
    assert stat.S_IMODE(consumed.stat().st_mode) == 0o700
