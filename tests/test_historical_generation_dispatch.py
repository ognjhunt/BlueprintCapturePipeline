"""ADP-009D/day28: exact protected decommission selects one installed action."""
# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_historical_dispatch.py

import pytest

from tests.test_historical_generation_authority import (
    decision, historical_installation, packet)  # noqa: F401
from tests.test_registered_experiment_issuer import installation  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401

# Imported pytest fixtures are intentionally named in the dependent tests.
# ruff: noqa: F811


def select(installed, action, **changes):
    from blueprint_pipeline.control_plane_lane_historical_dispatch import select_historical_action
    return select_historical_action(installed_config_path=installed[0], action_id=action['action_id'],
                                    now=1030, monotonic=lambda: 0, **changes)


def test_dispatch_derives_exact_target_and_private_store_from_approved_id(historical_installation):
    approved = decision(historical_installation, packet(historical_installation))
    before = (historical_installation[1] / 'one.log').read_bytes()
    selected = select(historical_installation, approved)
    assert selected['action_id'] == approved['action_id']
    assert selected['generation_digest'] == approved['generation_digest']
    assert selected['target_path'] == str(historical_installation[1])
    assert selected['read_write_paths'] == [str(historical_installation[1]),
        str(historical_installation[3].parent / 'historical-generation-journals')]
    assert str(historical_installation[3]) not in selected['read_write_paths']
    assert str(historical_installation[1].parent) not in selected['read_write_paths']
    assert selected['execution_authorized'] is False
    assert selected['action_unit_started'] is False
    assert (historical_installation[1] / 'one.log').read_bytes() == before


@pytest.mark.parametrize('change', ['payload', 'config', 'policy', 'packet', 'decision', 'manifest'])
def test_dispatch_reopens_current_authority_and_generation(historical_installation, change):
    selected = packet(historical_installation)
    approved = decision(historical_installation, selected)
    config, target, policy, store, _ = historical_installation
    path = {'payload': target / 'one.log', 'config': config, 'policy': policy,
            'packet': store / (selected['packet_id'] + '.json'),
            'decision': store / (approved['action_id'] + '.json'),
            'manifest': store / (selected['packet_id'] + '.manifest.json')}[change]
    path.write_bytes(path.read_bytes() + b'changed')
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    with pytest.raises(ValueError):
        select(historical_installation, approved)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before


@pytest.mark.parametrize('identifier', ['../escape', '', 'a' * 31, 'a' * 33])
def test_dispatch_accepts_only_an_immutable_action_id(historical_installation, identifier):
    with pytest.raises(ValueError):
        select(historical_installation, {'action_id': identifier})


def test_owner_review_never_produces_a_mutation_unit(historical_installation):
    approved = decision(historical_installation, packet(historical_installation), action='owner_review')
    with pytest.raises(ValueError, match='owner_review'):
        select(historical_installation, approved)


def test_writable_original_parent_keeps_target(historical_installation):
    approved = decision(historical_installation, packet(historical_installation))
    historical_installation[1].parent.chmod(0o777)
    with pytest.raises(ValueError):
        select(historical_installation, approved)
    assert (historical_installation[1] / 'one.log').is_file()


def test_action_expiry_during_hash_is_not_extended(historical_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_generation as generation
    approved = decision(historical_installation, packet(historical_installation))
    elapsed = [0]
    actual = generation.inventory_historical_generation

    def slow_hash(*args, **kwargs):
        result = actual(*args, **kwargs)
        elapsed[0] = 71
        return result

    monkeypatch.setattr(generation, 'inventory_historical_generation', slow_hash)
    with pytest.raises(ValueError, match='expired'):
        from blueprint_pipeline.control_plane_lane_historical_dispatch import select_historical_action
        select_historical_action(installed_config_path=historical_installation[0],
            action_id=approved['action_id'], now=1030, monotonic=lambda: elapsed[0])
    assert (historical_installation[1] / 'one.log').is_file()


def test_planned_unit_is_fixed_root_target_only_with_foreign_reference_rights(historical_installation):
    approved = decision(historical_installation, packet(historical_installation))
    selected = select(historical_installation, approved)
    assert selected['unit_name'] == 'blueprint-historical-generation-' + approved['action_id'] + '.service'
    assert selected['exec_start'] == ['/opt/blueprint/operator-door/bin/blueprint-historical-generation-action',
                                      approved['action_id']]
    props = selected['service_properties']
    assert props['User'] == props['Group'] == 'root'
    assert props['ProtectSystem'] == 'strict'
    assert props['ReadWritePaths'] == selected['read_write_paths']
    assert props['NoNewPrivileges'] is True
    assert props['PrivateUsers'] is False
    assert props['ProtectProc'] == 'default' and props['ProcSubset'] == 'all'
    assert 'CAP_SYS_PTRACE' in props['CapabilityBoundingSet']
    assert 'CAP_FOWNER' in props['CapabilityBoundingSet']
    assert 'CAP_SYS_ADMIN' not in props['CapabilityBoundingSet']
    assert props['TimeoutStartSec'] == 4 * 3600 and props['Restart'] == 'no'
    assert selected['execution_authorized'] is False and selected['action_unit_started'] is False


@pytest.mark.parametrize('change', ['rewrite', 'symlink', 'hardlink', 'nested_addition'])
def test_payload_rewrite_after_inventory_keeps_target(historical_installation, monkeypatch, change):
    import os
    from blueprint_pipeline import control_plane_lane_historical_dispatch as dispatch
    target = historical_installation[1]
    (target / 'nested').mkdir()
    (target / 'nested' / 'two.log').write_bytes(b'original nested bytes')
    approved = decision(historical_installation, packet(historical_installation))
    real = dispatch._selection
    calls = [0]

    def changed(*args, **kwargs):
        result = real(*args, **kwargs)
        calls[0] += 1
        if calls[0] == 2:
            if change == 'rewrite':
                (target / 'nested' / 'two.log').write_bytes(b'new writer bytes')
            elif change == 'symlink':
                (target / 'one.log').unlink()
                (target / 'one.log').symlink_to(target / 'nested' / 'two.log')
            elif change == 'hardlink':
                os.link(target / 'one.log', target / 'new-link')
            else:
                (target / 'nested' / 'new.log').write_bytes(b'new writer bytes')
        return result

    monkeypatch.setattr(dispatch, '_selection', changed)
    with pytest.raises(ValueError, match='changed'):
        select(historical_installation, approved)
    assert (target / 'one.log').exists() and (target / 'nested' / 'two.log').exists()


def test_owner_expiry_during_final_member_check_keeps_target(historical_installation, monkeypatch):
    from blueprint_pipeline import control_plane_lane_historical_generation as generation
    from blueprint_pipeline.control_plane_lane_historical_dispatch import select_historical_action
    approved = decision(historical_installation, packet(historical_installation))
    elapsed = [0]
    real = generation.verify_historical_member_versions

    def expired(*args, **kwargs):
        result = real(*args, **kwargs)
        elapsed[0] = 71
        return result

    monkeypatch.setattr(generation, 'verify_historical_member_versions', expired)
    with pytest.raises(ValueError, match='expired|deadline'):
        select_historical_action(installed_config_path=historical_installation[0],
            action_id=approved['action_id'], now=1030, monotonic=lambda: elapsed[0])
    assert (historical_installation[1] / 'one.log').is_file()


def test_unit_assignments_preserve_two_syscall_filters_and_only_selected_mounts(historical_installation):
    from blueprint_pipeline.control_plane_lane_historical_dispatch import _unit_property_assignments
    target, store = historical_installation[1], historical_installation[3]
    assignments = _unit_property_assignments(target, store)
    filters = [row for row in assignments if row.startswith('SystemCallFilter=')]
    assert filters == ['SystemCallFilter=@system-service seccomp landlock_create_ruleset landlock_add_rule landlock_restrict_self',
                       'SystemCallFilter=~ptrace process_vm_readv process_vm_writev']
    assert [row for row in assignments if row.startswith('ReadWritePaths=')] == [
        'ReadWritePaths=' + str(target) + ' ' + str(store)]
    assert 'PrivateUsers=no' in assignments and 'NoNewPrivileges=yes' in assignments


@pytest.mark.parametrize('suffix', [' with space', '%n', '\nReadWritePaths=/', '"', '\\'])
def test_unsupported_unit_path_never_expands_into_write_authority(historical_installation, suffix):
    from blueprint_pipeline.control_plane_lane_historical_dispatch import _unit_property_assignments
    with pytest.raises(ValueError, match='unit_path_unsupported'):
        _unit_property_assignments(str(historical_installation[1]) + suffix, historical_installation[3])
