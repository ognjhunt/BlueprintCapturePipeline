"""Installed GC sandbox must admit only the supported finite experiment roots."""
# Covers (for impacted-test selection):
#   deploy/systemd/blueprint-control-plane-storage-gc.service
from pathlib import Path

import pytest

UNIT = Path(__file__).resolve().parents[1] / 'deploy/systemd/blueprint-control-plane-storage-gc.service'


@pytest.mark.parametrize('path', [
    '/mnt/blueprint-work/lanes/g1',
    '/var/lib/blueprint/task-evaluation-inputs/lanes/g1',
    '/var/lib/blueprint-operator-door/requests/experiment-records',
    '/var/lib/blueprint-operator-door/experiment-authority',
])
def test_existing_gc_sandbox_admits_fixed_absent_safe_experiment_path(path):
    text = UNIT.read_text()
    writable = {part for line in text.splitlines() if line.startswith('ReadWritePaths=')
                for part in line.split('=', 1)[1].split()}
    assert '-' + path in writable
    assert 'ProtectSystem=strict\n' in text
    assert 'NoNewPrivileges=true\n' in text
    assert 'ReadOnlyPaths=/etc/blueprint/provider-secrets\n' in text
    assert 'ReadWritePaths=/var/lib/blueprint\n' not in text
    assert 'ReadWritePaths=/mnt/blueprint-work\n' not in text
    assert 'ReadWritePaths=/var/lib/blueprint-operator-door\n' not in text
