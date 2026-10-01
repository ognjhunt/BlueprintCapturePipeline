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
    # Only *.json is readable here: /etc/blueprint holds ~150 env files and backups.
    json_only_roots: tuple[str, ...] = ("/etc/blueprint",)
    intake_version_url: str = "http://127.0.0.1:8765/api/live-pipeline/version"
    control_plane_state: str = _CONTROL_PLANE
    # The capacity controller's secret-free summary: its other reports are root-only.
    capacity_summary: str = f"{_CONTROL_PLANE}/capacity/summary.json"
    lane_scratch_work_root: str = "/mnt/blueprint-work/lanes"
    lane_scratch_inputs_root: str = "/var/lib/blueprint/task-evaluation-inputs/lanes"
    owner_census_decisions_enabled: int = 0
    experiment_creation_enabled: bool = False
    experiment_retirement_enabled: bool = False
    historical_generation_actions_enabled: bool = False
    experiment_gc_environment_file: str = "/etc/blueprint/pipeline-control-plane.env"
    needed_checkpoint_cache_creation_enabled: bool = False
    needed_checkpoint_cache_inventory_file: str = "/opt/blueprint/control-plane-config-tools/operator-door-source/configs/g1_humanoidarena_checkpoint_inventory.v1.json"
    lane_owner_policy_file: str = "/etc/blueprint-operator-door/lane-owner-policy.json"
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
        "blueprint-native-g1-team-campaign-dispatcher.timer",
        "blueprint-native-g1-team-campaign-settlement.timer",
        "blueprint-gpu-spend-guard.timer",
    )
    max_read_bytes: int = 16 * 1024 * 1024
    max_archive_bytes: int = 512 * 1024 * 1024
    max_list_entries: int = 2000
    max_journal_lines: int = 2000
    max_request_body: int = 64 * 1024
    request_timeout_seconds: int = 30
    audit_rotate_bytes: int = 64 * 1024 * 1024
    spool_retention_days: int = 30
    source_clone: str = "/opt/blueprint/control-plane-config-tools/operator-door-source"
    reference_repo: str = "/opt/blueprint/BlueprintCapturePipeline"
    upstream_url: str = "https://github.com/ognjhunt/BlueprintCapturePipeline.git"
    # The repository is private and the host has no other GitHub credential: a
    # read-only deploy key in a root-only directory the door process cannot enter.
    github_deploy_key: str = "/etc/blueprint-operator-door/deploy-key/github"
    github_known_hosts: str = "/etc/blueprint-operator-door/deploy-key/known_hosts"
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
    def owner_consent_store(self) -> str:
        return str(Path(self.spool_root) / "owner-consents")

    @property
    def needed_checkpoint_cache_record_store(self) -> str:
        return str(Path(self.state_root) / "requests" / "needed-checkpoint-cache-records")

    @property
    def needed_checkpoint_cache_registration_root(self) -> str:
        return str(Path(self.state_root) / "needed-checkpoint-cache-registration")

    @property
    def needed_checkpoint_cache_authority_root(self) -> str:
        return str(Path(self.state_root) / "needed-checkpoint-cache-registration" / "authority")

    @property
    def experiment_record_store(self) -> str:
        return str(Path(self.spool_root) / "experiment-records")

    @property
    def experiment_authority_root(self) -> str:
        return str(Path(self.state_root) / "experiment-authority")

    @property
    def audit_path(self) -> str:
        return str(Path(self.state_root) / "audit" / "audit.jsonl")


_PATH_TUPLES = ("read_roots", "hidden_paths", "json_only_roots")
_PATH_SCALARS = (
    "state_root", "token_file", "install_root", "control_plane_state", "active_release_link",
    "source_clone", "reference_repo", "github_deploy_key", "github_known_hosts", "venv_python",
    "capacity_summary",
    "lane_scratch_work_root", "lane_scratch_inputs_root", "lane_owner_policy_file",
    "needed_checkpoint_cache_inventory_file", "experiment_gc_environment_file",
)
_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _coerce(name: str, value: Any, default: Any, *, _work_budget=None) -> Any:
    if _work_budget is not None:
        _work_budget.tick()
        _work_budget.measure(value, cap=64 * 1024)
    if type(default) is bool:
        if type(value) is not bool:
            raise DoorConfigError(f"door_config_type:{name}")
        return value
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
    return config_from_mapping(overrides)


def config_from_mapping(overrides, *, _work_budget=None) -> DoorConfig:
    """Validate one finite mapping; optional private caller shares its budget."""
    config = DoorConfig()
    if _work_budget is not None:
        _work_budget.tick()
        _work_budget.measure(overrides, cap=64 * 1024)
    if not isinstance(overrides, dict):
        raise DoorConfigError("door_config_not_object")
    known = {item.name for item in dataclasses.fields(DoorConfig)}
    values: dict[str, Any] = {}
    for name, value in overrides.items():
        if _work_budget is not None:
            _work_budget.charge("entries")
        if name not in known:
            raise DoorConfigError(f"door_config_unknown_key:{name}")
        values[name] = _coerce(name, value, getattr(config, name), _work_budget=_work_budget)
    return _validated(dataclasses.replace(config, **values), _work_budget=_work_budget)


def _validated(config: DoorConfig, *, _work_budget=None) -> DoorConfig:
    if type(config.owner_census_decisions_enabled) is not int or config.owner_census_decisions_enabled not in (0, 1):
        raise DoorConfigError("door_config_owner_enablement_invalid")
    for name in _PATH_TUPLES:
        for item in getattr(config, name):
            if _work_budget is not None:
                _work_budget.charge("entries")
                _work_budget.measure(item, cap=4098)
            if not os.path.isabs(item):
                raise DoorConfigError(f"door_config_path_not_absolute:{name}")
    for name in _PATH_SCALARS:
        if _work_budget is not None:
            _work_budget.charge("entries")
            _work_budget.measure(getattr(config, name), cap=4098)
        if not os.path.isabs(getattr(config, name)):
            raise DoorConfigError(f"door_config_path_not_absolute:{name}")
    trusted_parent = Path(DoorConfig().source_clone).parent
    source_clone = Path(config.source_clone)
    if _work_budget is not None:
        _work_budget.tick()
    linked = source_clone.is_symlink()
    if _work_budget is not None:
        _work_budget.tick()
    if source_clone.parent != trusted_parent or linked:
        raise DoorConfigError("door_config_source_clone_outside_trusted_root")
    if config.listen_host not in _LOOPBACK:
        raise DoorConfigError("door_config_listener_not_loopback")
    return config
