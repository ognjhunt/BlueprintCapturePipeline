"""ADP-009D/day28: production compiler pins must follow the worker lock."""
# Covers (for impacted-test selection):
#   deploy/systemd/production-control-plane-requirements.txt
#   scripts/install_live_pipeline_control_plane.sh

from pathlib import Path
import re
import tomllib


def test_observed_compile_differences_are_pinned_to_verified_worker_wheels():
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "deploy/systemd/production-control-plane-requirements.txt").read_text()
    packages = {p["name"]: p for p in tomllib.loads((root / "uv.lock").read_text())["package"]}
    for name in ("annotated-types", "packaging", "typing-inspection", "usd-core"):
        package = packages[name]
        pin = re.search(r"^" + re.escape(name) + r"==([^\s]+)\s+\\\n\s+--hash=(sha256:[a-f0-9]{64})",
                        requirements, re.MULTILINE)
        assert pin is not None, name
        assert pin.group(1) == package["version"]
        assert pin.group(2) in {w["hash"] for w in package["wheels"]}


def test_installer_synchronizes_all_pins_when_rfc8785_is_already_present():
    root = Path(__file__).resolve().parents[1]
    installer = (root / "scripts/install_live_pipeline_control_plane.sh").read_text()
    block = installer.split('RUNTIME_REQUIREMENTS=', 1)[1].split('# Build the Windows worker', 1)[0]
    assert 'if runuser' not in block
    assert '--no-deps --only-binary=:all:' in block
    assert '--require-hashes --requirement "${RUNTIME_REQUIREMENTS}"' in block
