"""ADP-009D/day28: authentic stage conflict refusal and original-journal replay.

Portable metadata and object transport only; installed Linux clearance remains
mandatory. This test does not supply process absence or completed robot proof.
"""
# Covers (for impacted-test selection):
#   tests/registered_disk_diagnostic_native_acceptance.py
#   src/blueprint_pipeline/control_plane_lane_experiment_restore.py
#   src/blueprint_pipeline/control_plane_lane_experiment_restore_reconcile.py

import os
import stat
import sys
from pathlib import Path

from tests.test_registered_disk_diagnostic_actions import configure, seal
from tests.test_registered_experiment_issuer import installation, encoded  # noqa: F401
from tests.test_owner_target_version_publication import root_metadata  # noqa: F401
from tests.historical_generation_fake_cloud import Cloud
from tests.test_registered_experiment_retirement_flow import _current_entry
from tests.registered_disk_diagnostic_native_acceptance import _restore_conflict_before_publication


def test_foreign_destination_refusal_can_replay_authenticated_original_stage(
        installation, monkeypatch):  # noqa: F811
    from blueprint_pipeline import control_plane_lane_experiment_archive as archive
    from blueprint_pipeline import control_plane_lane_experiment_restore as restoration
    from blueprint_pipeline import control_plane_lane_experiment_retirement as issuer
    from blueprint_pipeline import control_plane_lane_experiment_actions as actions
    from blueprint_pipeline import control_plane_lane_legacy_owner as legacy
    from blueprint_pipeline import control_plane_lane_disk_diagnostic_references as references
    value = installation
    configure(value)
    config, settings, _, _ = value
    tables = {}
    for kind in ('queue', 'evidence', 'settlement'):
        tables[kind] = config.parent / kind
        tables[kind].mkdir(mode=0o700)
    unit = config.parent / 'gc.service'
    unit.write_bytes(b'[Service]\n')
    unit.chmod(0o600)
    monkeypatch.setattr(legacy, '_GC_UNIT', unit)
    environment = config.parent / 'gc.env'
    environment.write_text(environment.read_text() + ''.join(
        'BLUEPRINT_CONTROL_PLANE_GC_' + key.upper() + '_ROOTS=' + str(path) + '\n'
        for key, path in tables.items()))
    release = config.parent / 'active-release'
    release.symlink_to(config.parent / 'installed')
    settings.update(control_plane_state=settings['state_root'], active_release_link=str(release))
    config.write_bytes(encoded(settings))
    _, _, grant, born, _ = seal(value, monkeypatch, 'root_disk_diagnostic_evidence.v1')
    intent_id = grant['intent_id']
    target = Path(born['path'])
    report = target / 'disk-capacity-report.v1.json'
    original = report.read_bytes()
    cloud = Cloud()
    monkeypatch.setattr(archive, '_client', lambda *args: (cloud, 'development-only'))
    # Only the kernel-specific process methods are substituted in this portable
    # test; actual table/pin/inode/namespace/journal checks remain in place.
    monkeypatch.setattr(references.DiagnosticReferences, '_own_handles', lambda *args: None)
    monkeypatch.setattr(references, 'refuse_historical_process_references', lambda *args, **kwargs: None)
    if sys.platform == 'darwin':
        # Darwin retains nlink=1 on an open unlinked directory. The installed
        # Linux method requires nlink=0 and is exercised unchanged in hosted CI.
        # Here independently prove that the actual named stage is gone, with
        # the original owned open directory still held, without faking fstat.
        def removed_stage(reader, fd):
            assert reader.stage == fd
            files = reader.action_files
            parent, name, _ = files.bindings[fd]
            assert parent == reader.target_fd and stat.S_ISDIR(os.fstat(fd).st_mode)
            try:
                os.stat(name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('portable restore retained its stage name')
            info = os.fstat(fd)
            reader.original.add((info.st_dev, info.st_ino))
            reader.stage = None
        monkeypatch.setattr(references.DiagnosticReferences, 'removed_stage', removed_stage)
    action = issuer.issue_experiment_action_intent(intent_id, principal='operator', owner='owner',
        action='offload', expires_at_epoch=3500, installed_config_path=config, now=lambda: 2900)
    retired = actions.run_action(action['action_id'], expected_action_intent=action['action_intent'],
        installed_config_path=config, now=lambda: 2901, _pins_root=config.parent / 'pins')
    assert retired['decision'] == 'retired' and not report.exists()
    restore = issuer.issue_experiment_restore_intent(intent_id, principal='operator', owner='owner',
        lease_ttl_seconds=600, expires_at_epoch=3400, installed_config_path=value[0], now=lambda: 2901)

    class Reservation:
        def release(self, **kwargs):
            pass

    monkeypatch.setattr(restoration, 'reserve_control_plane_disk', lambda *args, **kwargs: Reservation())
    _restore_conflict_before_publication({'config': value[0]}, restore, value[0].parent / 'pins', report,
                                         now=lambda: 2902)
    # A destination collision must reach the real publication boundary, rather
    # than contaminating the retired directory before its first admission.
    assert _current_entry(value, intent_id)['state'] == 'restoring'
    outcome = issuer.restore_registered_experiment(restore['action_id'],
        expected_restore_intent=restore['restore_intent'], installed_config_path=value[0],
        now=lambda: 2902, _pins_root=value[0].parent / 'pins')
    assert outcome['decision'] == 'restored' and report.read_bytes() == original
