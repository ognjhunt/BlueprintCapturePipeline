"""ADP-009D/day28: unit labels and supplied claims never prove native rights."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_unit.py
import sys

import pytest


def test_non_linux_or_ordinary_uid_never_reads_claimed_unit(monkeypatch):
    from blueprint_pipeline.control_plane_lane_historical_unit import prove_historical_unit
    import os
    monkeypatch.setattr(os, 'geteuid', lambda: 501)
    with pytest.raises(ValueError, match='native_unavailable'):
        prove_historical_unit('a' * 32, '/work/selected', '/private/journals')


@pytest.mark.skipif(sys.platform == 'linux', reason='Mac refusal is separate from actual Linux proof')
def test_mac_root_metadata_is_still_not_linux_unit_proof(monkeypatch):
    from blueprint_pipeline.control_plane_lane_historical_unit import prove_historical_unit
    import os
    monkeypatch.setattr(os, 'geteuid', lambda: 0)
    with pytest.raises(ValueError, match='native_unavailable'):
        prove_historical_unit('a' * 32, '/work/selected', '/private/journals')


@pytest.mark.parametrize('changed', ['uid', 'caps', 'pid', 'mount', 'command', 'readonly', 'namespace'])
def test_kernel_or_manager_drift_cannot_clear_installed_rights(changed):
    from blueprint_pipeline.control_plane_lane_historical_unit import _validate_unit_observation, _REQUIREMENTS
    action = 'a' * 32
    unit = 'blueprint-historical-generation-' + action + '.service'
    status = {'Uid': '0\t0\t0\t0', 'Gid': '0\t0\t0\t0', 'CapEff': '000000000008000f',
              'CapBnd': '000000000008000f', 'NoNewPrivs': '1', 'Seccomp': '2'}
    fields = _REQUIREMENTS | dict(Id=unit, MainPID='1234', ControlGroup='/system.slice/' + unit,
        ReadWritePaths='/work/selected /private/journals', TimeoutStartUSec='4h',
        SystemCallFilter='read write openat close landlock_create_ruleset landlock_add_rule landlock_restrict_self',
        CapabilityBoundingSet='cap_chown cap_dac_override cap_dac_read_search cap_fowner cap_sys_ptrace',
        ExecStart='{ path=/opt/blueprint/operator-door/bin/blueprint-historical-generation-action ; '
                  'argv[]=/opt/blueprint/operator-door/bin/blueprint-historical-generation-action '
                  + action + ' ; ignore_errors=no ; pid=1234 ; }')
    cgroup = '0::/system.slice/' + unit + '\n'
    # Parser baseline is accepted; supplied facts alone are not native proof.
    _validate_unit_observation(action, '/work/selected', '/private/journals',
                               fields, status, cgroup, pid=1234)
    if changed == 'uid':
        status['Uid'] = '0\t501\t0\t0'
    elif changed == 'caps':
        status['CapEff'] = '000000000028000f'  # Additional CAP_SYS_ADMIN.
    elif changed == 'pid':
        fields['MainPID'] = '9999'
    elif changed == 'mount':
        fields['ReadWritePaths'] += ' /work'
    elif changed == 'command':
        fields['ExecStart'] = fields['ExecStart'].replace(' ; ignore_errors', ' extra ; ignore_errors')
    elif changed == 'readonly':
        fields['ProtectSystem'] = 'no'
    else:
        cgroup = '0::/user.slice/fake.service\n'
    with pytest.raises(ValueError, match='unit_rights_unknown'):
        _validate_unit_observation(action, '/work/selected', '/private/journals',
                                   fields, status, cgroup, pid=1234)
