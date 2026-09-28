"""Actual finite registered batch; tiny payloads, no paid/provider operation."""
# ruff: noqa: F811
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_experiment_retirement.py
#   src/blueprint_pipeline/control_plane_lane_experiment_birth.py
import json
from pathlib import Path

from tests.test_registered_experiment_issuer import installation, issue  # noqa: F401
from tests.test_registered_experiment_birth import birth


def test_actual_sixteen_authentic_experiments_keep_independent_owner_generation(installation):
    from blueprint_pipeline import control_plane_lane_scratch as scratch

    grants, targets = [], []
    for index in range(16):
        grant = issue(installation, reference_value=f'finite-batch-{index:02d}')
        born = birth(installation, grant)
        path = Path(born['path'])
        assert path.is_dir()
        lease = json.loads((path / scratch.LEASE_FILE).read_bytes())
        assert lease['owner'] == 'owner'
        assert lease['run_ref'] == f'finite-batch-{index:02d}'
        assert lease['expires_at_epoch'] == 2800
        assert born['generation'] != grant['intent_id']
        grants.append(grant)
        targets.append((path, path.stat().st_ino, born['generation']))
    head = json.loads((installation[2].parents[1] / 'experiment-authority/HEAD.json').read_bytes())
    record = json.loads((installation[2].parents[1] / 'experiment-authority' / head['record_name']).read_bytes())
    assert len(record['enrollments']) == 16
    assert {entry['intent_id'] for entry in record['enrollments']} == {grant['intent_id'] for grant in grants}
    assert len({entry['generation'] for entry in record['enrollments']}) == 16
    assert all(path.stat().st_ino == inode for path, inode, _ in targets)
