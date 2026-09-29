# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_legacy_owner.py
#   deploy/operator-door/install.sh
"""Installed owner attribution reads only current root-controlled selectors."""

from pathlib import Path
from types import SimpleNamespace

import pytest


class _SourceFiles:
    def __init__(self, unit: bytes, env: bytes):
        self.unit = unit
        self.env = env

    def read(self, path, *, cap, protected):
        from blueprint_pipeline.control_plane_lane_legacy_owner import _GC_UNIT
        return (self.unit if Path(path) == _GC_UNIT else self.env), None


def test_reference_selection_requires_current_installed_queue_pin_and_run_roots():
    from blueprint_pipeline.control_plane_lane_legacy_owner import (
        LegacyOwnerError, _reference_settings,
    )

    unit = b"\n".join((
        b"Environment=BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS=/state/q1:/state/q2",
        b"Environment=BLUEPRINT_CONTROL_PLANE_GC_EVIDENCE_ROOTS=/state/runs",
        b"Environment=BLUEPRINT_CONTROL_PLANE_GC_SETTLEMENT_ROOTS=/state/intents",
        b"Environment=BLUEPRINT_CONTROL_PLANE_STORAGE_PINS_ROOT=/state/pins",
    ))
    config = SimpleNamespace(experiment_gc_environment_file="/etc/blueprint/pipeline-control-plane.env")
    selected = _reference_settings(_SourceFiles(unit, b"OTHER=value\n"), config)
    assert selected["queue_roots"] == (Path("/state/q1"), Path("/state/q2"))
    assert selected["pins_root"] == Path("/state/pins")
    assert selected["active_run_roots"] == (Path("/state/runs"), Path("/state/intents"))
    with pytest.raises(LegacyOwnerError, match="legacy_owner_references_incomplete"):
        _reference_settings(_SourceFiles(unit.replace(b"BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS", b"OTHER"), b""), config)
    with pytest.raises(LegacyOwnerError, match="legacy_owner_references_incomplete"):
        _reference_settings(_SourceFiles(unit, b"BLUEPRINT_CONTROL_PLANE_GC_QUEUE_ROOTS=relative\n"), config)


def test_installer_provisions_only_external_root_owned_registry():
    source = (Path(__file__).resolve().parents[1] / "deploy/operator-door/install.sh").read_text()
    block = source.split("# LEGACY OWNER REVIEW PROVISIONING BEGIN", 1)[1].split(
        "# LEGACY OWNER REVIEW PROVISIONING END", 1)[0]
    assert 'legacy_owner_store="$state_root/requests/legacy-owner-registrations"' in block
    assert "install -d -o root -g root -m 0700" in block
    assert 'legacy_owner_lock="$legacy_owner_store/.legacy-owner.lock"' in block
    assert "legacy_owner_provisioning_unsafe" in block
    assert "rm -" not in block and "chown -R" not in block
