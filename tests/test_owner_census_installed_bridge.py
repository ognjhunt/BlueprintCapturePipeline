"""Fixed installed root code and bounded config acquisition; no repo fallback."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_lane_owner_consents.py
#   deploy/operator-door/operator_door/config.py
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_lane_owner_consents as c
from blueprint_pipeline.control_plane_reference_budget import ReferenceCollectionBudget


def installed(tmp_path, monkeypatch):
    root = tmp_path / "installed"
    package = root / "operator_door"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"")
    actual = Path(__file__).parents[1] / "deploy/operator-door/operator_door/config.py"
    (package / "config.py").write_bytes(actual.read_bytes())
    path = tmp_path / "door.json"
    path.write_bytes(b'{"owner_census_decisions_enabled":1}')
    path.chmod(0o600)
    monkeypatch.setattr(c, "INSTALLED_PACKAGE_ROOT", root)
    monkeypatch.setattr(c, "_protected", lambda info, **kw: None)
    return path, package


def test_only_fixed_acquired_installed_bytes_supply_the_bridge(tmp_path, monkeypatch):
    path, _ = installed(tmp_path, monkeypatch)
    files = c._Files(ReferenceCollectionBudget(monotonic=lambda: 0))
    try:
        result = c._installed_config(files, path)
        assert result.owner_census_decisions_enabled == 1
        assert result.owner_consent_store.endswith("/requests/owner-consents")
        assert files.budget.counts["raw_bytes"] > len(path.read_bytes())
    finally:
        files.finish()


def test_missing_installed_source_refuses_instead_of_importing_repo(tmp_path, monkeypatch):
    path, package = installed(tmp_path, monkeypatch)
    (package / "config.py").unlink()
    files = c._Files(ReferenceCollectionBudget(monotonic=lambda: 0))
    try:
        with pytest.raises(
            c.OwnerCensusConsentError, match="owner_consent_installed_bridge_invalid"
        ):
            c._installed_config(files, path)
    finally:
        files.finish()


def test_changed_source_is_rechecked_after_code_execution(tmp_path, monkeypatch):
    path, package = installed(tmp_path, monkeypatch)
    source = package / "config.py"
    source.write_text(
        source.read_text() + "\nfrom pathlib import Path\nPath(__file__).chmod(0o777)\n"
    )
    files = c._Files(ReferenceCollectionBudget(monotonic=lambda: 0))
    try:
        with pytest.raises(c.OwnerCensusConsentError, match="owner_consent_record_changed"):
            c._installed_config(files, path)
    finally:
        files.finish()
