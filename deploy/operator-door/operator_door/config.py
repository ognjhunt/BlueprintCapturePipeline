"""Door configuration: conservative defaults, strict optional overrides.

The defaults describe the production host. ``/etc/blueprint-operator-door/door.json``
may override individual keys; an unknown key or a wrong type is refused rather
than ignored, because a typo in a hidden-path list must never widen access.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DoorConfigError(ValueError):
    pass


_CONTROL_PLANE = "/var/lib/blueprint/pipeline-control-plane"


@dataclass(frozen=True)
class DoorConfig:
    listen_host: str = "127.0.0.1"
    listen_port: int = 8767
    state_root: str = "/var/lib/blueprint-operator-door"
    token_file: str = "/etc/blueprint-operator-door/tokens.json"
    install_root: str = "/opt/blueprint/operator-door"
    read_roots: tuple[str, ...] = (
        "/var/lib/blueprint",
        "/opt/blueprint",
        "/workspace",
        "/mnt/blueprint-work",
        "/etc/blueprint",
        "/var/lib/blueprint-operator-door",
    )
    # Hidden by systemd (InaccessiblePaths=) and refused again here.
    hidden_paths: tuple[str, ...] = (
        "/etc/blueprint/provider-secrets",
        "/etc/blueprint/credentials",
        "/etc/blueprint/secrets",
        "/etc/blueprint/agent-execution-admissions",
        "/etc/blueprint-operator-door",
        "/var/lib/blueprint/spend-authority",
        "/var/lib/blueprint/spend-authority-home",
        f"{_CONTROL_PLANE}/agent-execution-rollout",
        f"{_CONTROL_PLANE}/episode-interpreter-service-account.json",
    )
    intake_version_url: str = "http://127.0.0.1:8765/api/live-pipeline/version"
    control_plane_state: str = _CONTROL_PLANE
    active_release_link: str = "/opt/blueprint/task-evaluation-control-plane"
    unit_prefix: str = "blueprint-"
    controller_units: tuple[str, ...] = (
        "blueprint-pipeline-intake.service",
        "blueprint-pubsub-handoff-listener.timer",
        "blueprint-task-evaluation-scene-progression.timer",
        "blueprint-task-evaluation-scene-progression.service",
        "blueprint-task-evaluation-configured-controls-progression.timer",
        "blueprint-task-evaluation-configured-controls-progression.service",
        "blueprint-task-evaluation-launch-activation.service",
        "blueprint-task-evaluation-launch-dispatcher.path",
        "blueprint-task-evaluation-policy-canary-dispatcher.path",
        "blueprint-gpu-spend-guard.timer",
    )
    max_read_bytes: int = 16 * 1024 * 1024
    max_archive_bytes: int = 512 * 1024 * 1024
    max_list_entries: int = 2000
    max_journal_lines: int = 2000
    max_request_body: int = 64 * 1024
    source_clone: str = "/opt/blueprint/control-plane-config-tools/operator-door-source"
    reference_repo: str = "/opt/blueprint/BlueprintCapturePipeline"
    upstream_url: str = "https://github.com/ognjhunt/BlueprintCapturePipeline.git"
    venv_python: str = "/opt/blueprint/BlueprintCapturePipeline/.venv/bin/python"
    idle_wait_units: tuple[str, ...] = (
        "blueprint-task-evaluation-scene-progression.service",
        "blueprint-task-evaluation-configured-controls-progression.service",
    )
    idle_wait_seconds: int = 1800

    @property
    def spool_root(self) -> str:
        return str(Path(self.state_root) / "requests")

    @property
    def audit_path(self) -> str:
        return str(Path(self.state_root) / "audit" / "audit.jsonl")


_PATH_TUPLES = ("read_roots", "hidden_paths")
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _coerce(name: str, value: Any, default: Any) -> Any:
    if isinstance(default, tuple):
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise DoorConfigError(f"door_config_type:{name}")
        return tuple(value)
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise DoorConfigError(f"door_config_type:{name}")
        return value
    if not isinstance(value, str):
        raise DoorConfigError(f"door_config_type:{name}")
    return value


def load_config(path: str | os.PathLike[str] = "/etc/blueprint-operator-door/door.json") -> DoorConfig:
    config = DoorConfig()
    source = Path(path)
    if not source.is_file():
        return _validated(config)
    try:
        overrides = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DoorConfigError("door_config_unreadable") from error
    if not isinstance(overrides, dict):
        raise DoorConfigError("door_config_not_object")
    known = {item.name for item in dataclasses.fields(DoorConfig)}
    values: dict[str, Any] = {}
    for name, value in overrides.items():
        if name not in known:
            raise DoorConfigError(f"door_config_unknown_key:{name}")
        values[name] = _coerce(name, value, getattr(config, name))
    return _validated(dataclasses.replace(config, **values))


def _validated(config: DoorConfig) -> DoorConfig:
    for name in _PATH_TUPLES:
        if not all(os.path.isabs(item) for item in getattr(config, name)):
            raise DoorConfigError(f"door_config_path_not_absolute:{name}")
    if config.listen_host not in _LOOPBACK:
        raise DoorConfigError("door_config_listener_not_loopback")
    return config
