"""The operator door's systemd units agree with its code and keep their sandbox."""

from __future__ import annotations

import configparser
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
sys.path.insert(0, str(REPO_ROOT / "deploy" / "operator-door"))

from operator_door.config import DoorConfig  # noqa: E402

DOOR = SYSTEMD_DIR / "blueprint-operator-door.service"
RUNNER = SYSTEMD_DIR / "blueprint-operator-door-runner.service"
TRIGGER = SYSTEMD_DIR / "blueprint-operator-door-runner.path"


def _unit(path: Path) -> configparser.RawConfigParser:
    parser = configparser.RawConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # keep systemd's key case
    parser.read_string(path.read_text(encoding="utf-8"))
    return parser


def test_door_runs_the_installed_package_on_system_python() -> None:
    service = _unit(DOOR)["Service"]
    assert service["ExecStart"] == "/usr/bin/python3 -m operator_door serve"
    assert f"PYTHONPATH={DoorConfig().install_root}" in DOOR.read_text(encoding="utf-8")
    assert service["User"] == "blueprint" and service["SupplementaryGroups"] == "systemd-journal"


def test_door_hides_every_secret_location_except_its_own_token_store() -> None:
    text = DOOR.read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if line.startswith("InaccessiblePaths="))
    hidden = {path.lstrip("-") for path in line.split("=", 1)[1].split()}
    config = DoorConfig()
    # The door must read its own token file; the file view refuses that directory instead.
    assert hidden == set(config.hidden_paths) - {"/etc/blueprint-operator-door"}


def test_door_can_only_write_its_state_and_only_talk_to_loopback() -> None:
    service = _unit(DOOR)["Service"]
    assert service["ReadWritePaths"] == DoorConfig().state_root
    assert service["IPAddressDeny"] == "any" and service["IPAddressAllow"] == "localhost"
    assert service["CapabilityBoundingSet"] == "" and service["ProtectSystem"] == "strict"


def test_runner_is_a_sandboxed_root_oneshot_without_network() -> None:
    service = _unit(RUNNER)["Service"]
    assert service["Type"] == "oneshot" and service["User"] == "root"
    assert service["ExecStart"] == "/usr/bin/python3 -m operator_door run-spool"
    assert service["CapabilityBoundingSet"] == "CAP_DAC_OVERRIDE"
    assert service["PrivateNetwork"] == "true" and service["RestrictAddressFamilies"] == "AF_UNIX"
    assert service["ReadWritePaths"] == DoorConfig().spool_root


def test_path_unit_watches_the_spool_the_door_writes() -> None:
    path = _unit(TRIGGER)["Path"]
    assert path["PathExistsGlob"] == f"{DoorConfig().spool_root}/pending/*.json"
    assert path["Unit"] == RUNNER.name
