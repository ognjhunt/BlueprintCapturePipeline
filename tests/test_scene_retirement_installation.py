# Covers: scripts/install_live_pipeline_control_plane.sh
"""Exercise the actual installer's protected-store block without installing units."""
from pathlib import Path
import os
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/install_live_pipeline_control_plane.sh'
pytestmark = pytest.mark.slow


def block():
    source = SCRIPT.read_text()
    start = source.index('# BEGIN scene retirement stores')
    end = source.index('# END scene retirement stores', start)
    return source[start:end]


def execute(tmp_path, *, setup=None):
    root = tmp_path.resolve() / 'scene-retirement'
    if setup:
        setup(root)
    # The actual source block supplies all modes/owners. Intercept mutations;
    # exercise real path and symlink checks against this tiny local tree.
    source = block().replace('/var/lib/blueprint/scene-retirement', str(root))
    script = '''set -euo pipefail
SERVICE_USER=blueprint
SERVICE_GROUP=blueprint
DRY_RUN=true
run() { printf 'COMMAND'; printf ' <%s>' "$@"; printf '\\n'; }
''' + source
    return root, subprocess.run(['bash', '-c', script], capture_output=True, text=True,
                                env=dict(os.environ), timeout=10)


def test_actual_installer_provisions_distinct_private_and_readable_stores(tmp_path):
    root, result = execute(tmp_path)
    assert result.returncode == 0, result.stderr
    commands = result.stdout
    for suffix, mode, owner in [('', '0755', 'root'), ('/coordinator', '0755', 'root'),
                                ('/generations', '0700', 'blueprint'), ('/journals', '0700', 'root'),
                                ('/journals/processes', '0700', 'root'),
                                ('/journals/retired', '0700', 'root'), ('/journals.metadata', '0750', 'root'),
                                ('/consents', '0700', 'root')]:
        assert f'<install> <-d> <-m> <{mode}> <-o> <{owner}> <-g> <blueprint> <{root}{suffix}>' in commands
    assert not root.exists()  # A dry run never creates or adopts authority.
    assert 'chmod -R' not in commands and 'chown -R' not in commands


@pytest.mark.parametrize('link', ['root', 'journals', 'processes', 'ancestor'])
def test_installer_refuses_linked_stores_before_any_mutation(tmp_path, link):
    foreign = tmp_path.resolve() / 'foreign'
    foreign.mkdir()
    (foreign / 'private-proof').write_bytes(b'preserve-this')

    def setup(root):
        if link == 'root':
            root.symlink_to(foreign)
        elif link == 'journals':
            root.mkdir()
            (root / 'journals').symlink_to(foreign)
        elif link == 'processes':
            (root / 'journals').mkdir(parents=True)
            (root / 'journals' / 'processes').symlink_to(foreign)
        else:
            # Exercise the ancestry checks with the real root's parent linked.
            root.parent.rename(root.parent.with_name(root.parent.name + '-real'))
            root.parent.symlink_to(root.parent.with_name(root.parent.name + '-real'))

    _, result = execute(tmp_path, setup=setup)
    assert result.returncode != 0
    assert 'COMMAND' not in result.stdout
    assert b'preserve-this' in [p.read_bytes() for p in foreign.resolve().iterdir()]


def test_authority_store_is_outside_recursive_service_migration_and_before_units():
    source = SCRIPT.read_text()
    block()
    assert source.index('# BEGIN scene retirement stores') > source.index('run chown -R --no-dereference')
    assert source.index('# END scene retirement stores') < source.index('systemctl daemon-reload')
    assert 'BLUEPRINT_SCENE_RETIREMENT_POLICY_FILE=' not in block()
    assert 'BLUEPRINT_CONTROL_PLANE_GC_SCENE' not in block()
