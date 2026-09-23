"""The operator door's configuration: safe defaults, strict overrides."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door.config import DoorConfig, DoorConfigError, load_config  # noqa: E402


def test_defaults_load_without_a_config_file(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.json")
    assert isinstance(config, DoorConfig)
    assert config.listen_host == "127.0.0.1"
    assert config.listen_port == 8767
    assert "/var/lib/blueprint" in config.read_roots
    assert "/etc/blueprint/provider-secrets" in config.hidden_paths


def test_defaults_hide_every_known_secret_location() -> None:
    hidden = set(DoorConfig().hidden_paths)
    assert {
        "/etc/blueprint/provider-secrets",
        "/etc/blueprint/credentials",
        "/etc/blueprint/secrets",
        "/var/lib/blueprint/spend-authority-home",
        "/var/lib/blueprint/pipeline-control-plane/agent-execution-rollout",
        "/var/lib/blueprint/pipeline-control-plane/episode-interpreter-service-account.json",
        "/etc/blueprint-operator-door",
    } <= hidden


def test_override_replaces_only_named_keys(tmp_path: Path) -> None:
    path = tmp_path / "door.json"
    path.write_text(json.dumps({"listen_port": 9001, "max_list_entries": 10}), encoding="utf-8")
    config = load_config(path)
    assert config.listen_port == 9001
    assert config.max_list_entries == 10
    assert config.listen_host == "127.0.0.1"


def test_override_lists_become_tuples(tmp_path: Path) -> None:
    path = tmp_path / "door.json"
    path.write_text(json.dumps({"read_roots": ["/srv/a", "/srv/b"]}), encoding="utf-8")
    assert load_config(path).read_roots == ("/srv/a", "/srv/b")


def test_unknown_keys_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "door.json"
    path.write_text(json.dumps({"listen_prot": 1}), encoding="utf-8")
    with pytest.raises(DoorConfigError, match="door_config_unknown_key:listen_prot"):
        load_config(path)


@pytest.mark.parametrize("key", ["read_roots", "hidden_paths"])
def test_roots_and_hidden_paths_must_be_absolute(tmp_path: Path, key: str) -> None:
    path = tmp_path / "door.json"
    path.write_text(json.dumps({key: ["relative/path"]}), encoding="utf-8")
    with pytest.raises(DoorConfigError, match=f"door_config_path_not_absolute:{key}"):
        load_config(path)


def test_listener_must_stay_on_loopback(tmp_path: Path) -> None:
    path = tmp_path / "door.json"
    path.write_text(json.dumps({"listen_host": "0.0.0.0"}), encoding="utf-8")
    with pytest.raises(DoorConfigError, match="door_config_listener_not_loopback"):
        load_config(path)


def test_wrong_value_types_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "door.json"
    path.write_text(json.dumps({"listen_port": "8767"}), encoding="utf-8")
    with pytest.raises(DoorConfigError, match="door_config_type:listen_port"):
        load_config(path)


def test_derived_spool_paths_follow_the_state_root(tmp_path: Path) -> None:
    path = tmp_path / "door.json"
    path.write_text(json.dumps({"state_root": str(tmp_path / "state")}), encoding="utf-8")
    config = load_config(path)
    assert config.spool_root == str(tmp_path / "state" / "requests")
    assert config.audit_path == str(tmp_path / "state" / "audit" / "audit.jsonl")
