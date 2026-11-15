"""The parent preparation sandbox can still publish SAM child jobs."""
from __future__ import annotations

import os
from pathlib import Path
import stat

from tests.test_deploy_control_plane_commit import _stage_real_units, deploy


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
