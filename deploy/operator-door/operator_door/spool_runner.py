"""Root oneshot that turns spooled requests into a small set of fixed commands.

Started by ``blueprint-operator-door-runner.path`` whenever ``pending/*.json``
exists. It trusts nothing in the spool: each file is opened without following
symlinks, bounded in size, checked against its own id and revalidated with the
same schemas the door used. It then either runs ``systemctl --no-block`` for a
unit action or starts one transient unit running a script installed with the
door, passing only validated values as environment variables. Every pending
file is drained, even junk or an unreadable file, so the path unit can never
spin; only ``pending/`` is writable by the door, everything after it is root's.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import activation_hold, holds
from .config import DoorConfig
from .hostinfo import CommandRunner, SubprocessRunner
from .requests import (
    SCHEMA,
    RequestRefused,
    _json_or_none,
    load_request_file,
    validate_request,
    validate_request_id,
)

RESULT_SCHEMA = "blueprint_operator_door_result.v1"
# The door token names auth.py admits; only these reach a script as the caller.
_CALLER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_ACTIVE_STATES = "active,activating,deactivating,reloading"
_UNIT_OBSERVATION_PROPERTIES = "ActiveState,SubState,Result,InvocationID,ExecMainStatus,Type"
_PARKED_SCHEMA = "blueprint_operator_door_parked_deploy.v1"
_PARKED_MAX = 16
_PARKED_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z-deploy-[0-9a-f]{8}")
# As long as a client waits: two deploys at their 3 h limit, and the deploy lock. A request
# this old has nobody waiting for it and may no longer be wanted, and a new one deploys.
_PARKED_MAX_AGE_SECONDS = 8 * 3600
_UNKNOWN_DEPLOY_ACTIVITY = "unknown:deploy_units_unreadable"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _write_result(results: Path, request_id: str, payload: dict[str, Any]) -> None:
    document = {"schema": RESULT_SCHEMA, "id": request_id, "finished_at": _now(), **payload}
    handle, temporary = tempfile.mkstemp(dir=results, prefix=f".{request_id}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
        os.chmod(temporary, 0o644)
        os.replace(temporary, results / f"{request_id}.json")
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _source_environment(config: DoorConfig, request: dict[str, Any]) -> dict[str, str]:
    """What a script that fetches and runs a commit on main needs."""

    return {
        "DOOR_COMMIT": request["commit"],
        "DOOR_INSTALL_ROOT": config.install_root,
        "DOOR_SOURCE_CLONE": config.source_clone,
        "DOOR_REFERENCE_REPO": config.reference_repo,
        "DOOR_UPSTREAM_URL": config.upstream_url,
        "DOOR_GITHUB_KEY": config.github_deploy_key,
        "DOOR_GITHUB_KNOWN_HOSTS": config.github_known_hosts,
        "DOOR_VENV_PYTHON": config.venv_python,
        "DOOR_STATE_ROOT": config.control_plane_state,
    }


def _deploy_environment(config: DoorConfig, request: dict[str, Any]) -> dict[str, str]:
    return {
        **_source_environment(config, request),
        "DOOR_WAIT_FOR_IDLE": "1" if request["wait_for_idle"] else "0",
        "DOOR_IDLE_UNITS": ",".join(config.idle_wait_units),
        "DOOR_IDLE_WAIT_SECONDS": str(config.idle_wait_seconds),
        # activation_hold.py reads the running release and, to park a direct deploy, the queue.
        # Its own name: DOOR_PARKED_DIR comes only with DOOR_PARKED, the parked-deploy mark.
        "DOOR_ACTIVATION_HOLD_PARKED_DIR": str(Path(config.spool_root) / "parked"),
        "DOOR_ACTIVE_RELEASE_LINK": config.active_release_link,
        **({"DOOR_ACTIVATION_HOLD_OVERRIDE": request["activation_hold_override"]}
           if request.get("activation_hold_override") else {}),
    }


_PUBSUB_HANDOFFS = "/var/lib/blueprint/pubsub-handoffs"


def _retire_environment(config: DoorConfig, request: dict[str, Any]) -> dict[str, str]:
    values = {"DOOR_VENV_PYTHON": config.venv_python, "DOOR_SCENE_ID": request["scene_id"]}
    if request.get("bucket"):
        values["DOOR_BUCKET"] = request["bucket"]
    if request["apply"]:
        values["DOOR_APPLY"] = "1"
    return values


def _restore_environment(config: DoorConfig, request: dict[str, Any]) -> dict[str, str]:
    return {"DOOR_VENV_PYTHON": config.venv_python, "DOOR_SCENE_ID": request["scene_id"],
            "DOOR_BUCKET": request["bucket"]}


def _scratch_root(config: DoorConfig, request: dict[str, Any]) -> str:
    return config.lane_scratch_work_root if request["root"] == "work" else config.lane_scratch_inputs_root


def _scratch_environment(config: DoorConfig, request: dict[str, Any]) -> dict[str, str]:
    values = {"DOOR_VENV_PYTHON": config.venv_python, "DOOR_CONTROL_PLANE_REPO": config.active_release_link,
              "DOOR_SCRATCH_ROOT": _scratch_root(config, request),
              "DOOR_SCRATCH_ACTION": request["action"], "DOOR_SCRATCH_LANE": request["lane"]}
    for field in ("name", "owner", "expected_digest", "ttl_seconds", "limit", "offset"):
        if field in request:
            values["DOOR_SCRATCH_" + field.upper()] = str(request[field])
    return values


def _scratch_properties(config: DoorConfig, request: dict[str, Any]) -> tuple[str, ...]:
    results = str(Path(config.spool_root) / "results")
    return ("ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "PrivateDevices=yes",
            "ProtectHome=yes", "ProtectKernelTunables=yes", "ProtectControlGroups=yes",
            f"ReadWritePaths={_scratch_root(config, request)} {results}")



def _owner_environment(config: DoorConfig, request: dict[str, Any]) -> dict[str, str]:
    return {"DOOR_VENV_PYTHON": config.venv_python, "DOOR_CONTROL_PLANE_REPO": config.active_release_link,
            "DOOR_CONFIG_PATH": "/etc/blueprint-operator-door/door.json",
            "DOOR_CONSENT_ID": request["consent_id"], "DOOR_CONSENT_SHA256": request["expected_sha256"],
            "DOOR_CONSENT_SIZE_BYTES": str(request["expected_size_bytes"])}


def _lifecycle_environment(config: DoorConfig, request: dict[str, Any]) -> dict[str, str]:
    return {"DOOR_VENV_PYTHON": config.venv_python,
            "DOOR_CONTROL_PLANE_REPO": config.active_release_link,
            "DOOR_SCENE_ACTION": "retire" if request["kind"] == "retire-scene" else "restore",
            "DOOR_SCENE_INTENT_ID": request["intent_id"],
            "DOOR_SCENE_CONSENT_ID": request["consent_id"],
            "DOOR_SCENE_CONSENT_SHA256": request["expected_sha256"],
            "DOOR_SCENE_CONSENT_SIZE_BYTES": str(request["expected_size_bytes"]),
            "DOOR_SCENE_APPLY": "1" if request.get("apply", True) else "0"}


def _lifecycle_properties(config: DoorConfig, _request: dict[str, Any]) -> tuple[str, ...]:
    # Fixed installed containers only. The native engine further requires exact
    # consented members, installed policy and generation/reference admission.
    roots = " ".join("/var/lib/blueprint/" + root for root in
                     ("scene-retirement", "pubsub-handoffs", "task-evaluation-inputs", "pipeline-control-plane"))
    return ("ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "PrivateDevices=yes",
            "ProtectHome=yes", "ProtectKernelTunables=yes", "ProtectControlGroups=yes",
            "CapabilityBoundingSet=CAP_DAC_OVERRIDE CAP_CHOWN CAP_FOWNER CAP_SYS_PTRACE",
            f"ReadWritePaths={roots} /mnt/blueprint-work {Path(config.spool_root) / 'results'}")


def _legacy_owner_environment(config: DoorConfig, _request: dict[str, Any]) -> dict[str, str]:
    return {"DOOR_VENV_PYTHON": config.venv_python,
            "DOOR_CONTROL_PLANE_REPO": config.active_release_link,
            "DOOR_CONFIG_PATH": "/etc/blueprint-operator-door/door.json"}


def _owner_properties(config: DoorConfig, _request: dict[str, Any]) -> tuple[str, ...]:
    return ("ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "PrivateDevices=yes",
            "ProtectHome=yes", "ProtectKernelTunables=yes", "ProtectControlGroups=yes", "PrivateNetwork=yes",
            "CapabilityBoundingSet=", "AmbientCapabilities=",
            f"ReadOnlyPaths={config.owner_consent_store} {config.lane_owner_policy_file}",
            f"ReadWritePaths={Path(config.spool_root) / 'results'}")


def _legacy_owner_properties(config: DoorConfig, _request: dict[str, Any]) -> tuple[str, ...]:
    """Fixed read-only inputs; the only writable location is this door's results."""
    registry = Path(config.owner_consent_store).parent / "legacy-owner-registrations"
    # CAP_PERFMON admits read-only foreign /proc metadata but cannot open
    # foreign /proc/PID/mem. This transient must also inherit
    # the door's secret hides. The config/policy files under the door root are
    # required; hide its token and deploy key individually instead.
    hidden = [path for path in config.hidden_paths if path != "/etc/blueprint-operator-door"]
    hidden.extend((config.token_file, "/etc/blueprint-operator-door/deploy-key"))
    return ("ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "PrivateDevices=yes",
            "ProtectHome=yes", "ProtectKernelTunables=yes", "ProtectControlGroups=yes", "PrivateNetwork=yes",
            "SystemCallFilter=@system-service",
            "SystemCallFilter=~ptrace process_vm_readv process_vm_writev",
            "SystemCallErrorNumber=EPERM",
            "CapabilityBoundingSet=CAP_DAC_READ_SEARCH CAP_PERFMON",
            "AmbientCapabilities=CAP_DAC_READ_SEARCH CAP_PERFMON",
            "InaccessiblePaths=" + " ".join("-" + path for path in hidden),
            f"ReadOnlyPaths={registry} {config.lane_owner_policy_file} "
            f"{config.experiment_gc_environment_file} "
            f"-/etc/systemd/system/blueprint-control-plane-storage-gc.service "
            f"{config.lane_scratch_work_root} {config.lane_scratch_inputs_root}",
            f"ReadWritePaths={Path(config.spool_root) / 'results'}")


def _retire_properties(config: DoorConfig, _request: dict[str, Any]) -> tuple[str, ...]:
    """Limit retirement to its spool, coordination locks, and result."""

    results = str(Path(config.spool_root) / "results")
    reservations = str(Path(config.control_plane_state) / "disk-reservations")
    pins = str(Path(config.control_plane_state) / "storage-pins")
    return ("ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "PrivateDevices=yes",
            "ProtectHome=yes", "ProtectKernelTunables=yes", "ProtectControlGroups=yes",
            "CapabilityBoundingSet=CAP_DAC_OVERRIDE CAP_CHOWN CAP_SYS_PTRACE",
            "AmbientCapabilities=CAP_DAC_OVERRIDE",
            f"ReadWritePaths={_PUBSUB_HANDOFFS} {reservations} {pins} {results}")


def _canary_root(config: DoorConfig) -> str:
    return str(Path(config.control_plane_state) / "task-evaluation-policy-canaries")


def _selected_handoff_environment(config: DoorConfig, request: dict[str, Any]) -> dict[str, str]:
    selection = {key: value for key, value in request.items() if key != "resume_request_id"}
    return {"DOOR_VENV_PYTHON": config.venv_python, "DOOR_CONTROL_PLANE_REPO": config.active_release_link,
            "DOOR_SERVICE_USER": "blueprint", "DOOR_SELECTED_HANDOFF_REQUEST": json.dumps(selection, sort_keys=True)}


def _embodiment_preparation_environment(config: DoorConfig, request: dict[str, Any]) -> dict[str, str]:
    return {"DOOR_VENV_PYTHON": config.venv_python, "DOOR_CONTROL_PLANE_REPO": config.active_release_link,
            "DOOR_SERVICE_USER": "blueprint",
            "DOOR_EMBODIMENT_PREPARATION_REQUEST": json.dumps(request["request"], sort_keys=True, allow_nan=False)}


def _embodiment_preparation_properties(config: DoorConfig, _request: dict[str, Any]) -> tuple[str, ...]:
    return ("ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "PrivateDevices=yes",
            "ProtectHome=yes", "ProtectKernelTunables=yes", "ProtectControlGroups=yes", "PrivateNetwork=yes",
            "RestrictAddressFamilies=AF_UNIX", "UMask=0077",
            "CapabilityBoundingSet=CAP_SETUID CAP_SETGID CAP_CHOWN CAP_FOWNER CAP_DAC_OVERRIDE",
            "InaccessiblePaths=" + " ".join("-" + path for path in config.hidden_paths),
            "ReadWritePaths=/var/lib/blueprint/pipeline-control-plane/embodiment-preparation "
            + str(Path(config.spool_root) / "results"))


def _public_policy_environment(config: DoorConfig, request: dict[str, Any]) -> dict[str, str]:
    return {"DOOR_VENV_PYTHON": config.venv_python, "DOOR_CONTROL_PLANE_REPO": config.active_release_link,
            "DOOR_SERVICE_USER": "blueprint",
            "DOOR_PUBLIC_POLICY_OPERATION": request["kind"],
            "DOOR_PUBLIC_POLICY_REQUEST": json.dumps(request["request"], sort_keys=True, allow_nan=False)}


def _public_policy_properties(config: DoorConfig, _request: dict[str, Any]) -> tuple[str, ...]:
    return ("ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "PrivateDevices=yes",
            "ProtectHome=yes", "ProtectKernelTunables=yes", "ProtectControlGroups=yes", "PrivateNetwork=yes",
            "RestrictAddressFamilies=AF_UNIX", "UMask=0077",
            "CapabilityBoundingSet=CAP_SETUID CAP_SETGID CAP_CHOWN CAP_FOWNER CAP_DAC_OVERRIDE",
            "InaccessiblePaths=" + " ".join("-" + path for path in config.hidden_paths),
            "ReadWritePaths=/etc/blueprint /var/lib/blueprint/pipeline-control-plane/public-policy-registrations "
            "/var/lib/blueprint-operator-door/policy-registrations " + str(Path(config.spool_root) / "results"))


def _native_g1_properties(config: DoorConfig, request: dict[str, Any]) -> tuple[str, ...]:
    # Fixed production site-packet store, never a request-selected write path.
    return tuple(value + " /var/lib/blueprint/task-evaluation-inputs/team-scene-halves"
                 if value.startswith("ReadWritePaths=") else value
                 for value in _public_policy_properties(config, request))


def _selected_handoff_properties(config: DoorConfig, request: dict[str, Any]) -> tuple[str, ...]:
    writable = str(Path(config.spool_root) / "results")
    if request["mode"] in {"dispatch", "preparation-resume"}:
        writable += " /var/lib/blueprint"
    elif request["mode"] == "atlas-retry":
        # Only the existing selected capture's Atlas job may be requeued.
        writable += (f" /var/lib/blueprint/pubsub-handoffs/{request['bucket']}/scenes/"
                     f"{request['scene_id']}/captures/{request['capture_id']}/pipeline/atlas")
    return ("ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "PrivateDevices=yes",
            "ProtectHome=yes", "ProtectKernelTunables=yes", "ProtectControlGroups=yes",
            "UMask=0077", f"ReadWritePaths={writable}")


def _scene_window_environment(config: DoorConfig, request: dict[str, Any]) -> dict[str, str]:
    return {"DOOR_VENV_PYTHON": config.venv_python, "DOOR_CONTROL_PLANE_REPO": config.active_release_link,
            "DOOR_SERVICE_USER": "blueprint",
            "DOOR_SCENE_WINDOW_REQUEST": json.dumps(request, sort_keys=True, allow_nan=False)}


def _scene_window_properties(config: DoorConfig, _request: dict[str, Any]) -> tuple[str, ...]:
    """Writes one record under the scene intents and its result. The signed WebApp
    notice needs the network and the Pipeline sync token in provider-secrets;
    every other hidden secret stays hidden."""
    hidden = [path for path in config.hidden_paths if path != "/etc/blueprint/provider-secrets"]
    return ("ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "PrivateDevices=yes",
            "ProtectHome=yes", "ProtectKernelTunables=yes", "ProtectControlGroups=yes", "UMask=0077",
            "CapabilityBoundingSet=CAP_SETUID CAP_SETGID CAP_CHOWN CAP_FOWNER CAP_DAC_OVERRIDE",
            "InaccessiblePaths=" + " ".join("-" + path for path in hidden),
            "ReadWritePaths=/var/lib/blueprint/pipeline-control-plane " + str(Path(config.spool_root) / "results"))


def _scene_budget_environment(config: DoorConfig, request: dict[str, Any]) -> dict[str, str]:
    return {"DOOR_VENV_PYTHON": config.venv_python, "DOOR_CONTROL_PLANE_REPO": config.active_release_link,
            "DOOR_SERVICE_USER": "blueprint",
            "DOOR_SCENE_BUDGET_REQUEST": json.dumps(request, sort_keys=True, allow_nan=False)}


def _scene_budget_properties(config: DoorConfig, _request: dict[str, Any]) -> tuple[str, ...]:
    """Writes one grant under the scene intents and its result; no network, every secret hidden."""
    return ("ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "PrivateDevices=yes",
            "ProtectHome=yes", "ProtectKernelTunables=yes", "ProtectControlGroups=yes", "UMask=0077",
            "PrivateNetwork=yes",
            "CapabilityBoundingSet=CAP_SETUID CAP_SETGID CAP_CHOWN CAP_FOWNER CAP_DAC_OVERRIDE",
            "InaccessiblePaths=" + " ".join("-" + path for path in config.hidden_paths),
            "ReadWritePaths=/var/lib/blueprint/pipeline-control-plane " + str(Path(config.spool_root) / "results"))


def _output_resume_environment(config: DoorConfig, request: dict[str, Any]) -> dict[str, str]:
    values = {"DOOR_VENV_PYTHON": config.venv_python, "DOOR_CONTROL_PLANE_REPO": config.active_release_link,
              "DOOR_CANARY_ROOT": _canary_root(config), "DOOR_RUN": request["run"],
              "DOOR_ATTEMPT": str(request["attempt"]), "DOOR_SERVICE_USER": "blueprint"}
    if request["ingest"]:
        values["DOOR_INGEST"] = "1"
    return values


def _output_resume_properties(config: DoorConfig, _request: dict[str, Any]) -> tuple[str, ...]:
    """The canary tree, the disk ledger (ingestion's hold) and the result; nothing else is writable."""

    results = str(Path(config.spool_root) / "results")
    reservations = str(Path(config.control_plane_state) / "disk-reservations")
    return ("ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "PrivateDevices=yes",
            "ProtectHome=yes", "ProtectKernelTunables=yes", "ProtectControlGroups=yes",
            f"ReadWritePaths={_canary_root(config)} {reservations} {results}")


@dataclass(frozen=True)
class _LaunchSpec:
    """How one request kind becomes one transient ``.service`` unit running one installed script."""

    unit_prefix: str
    script: str
    #: Bounds both the start and the whole run: ``TimeoutStartSec`` alone does not bound an exec unit.
    runtime_max: str
    label: Callable[[dict[str, Any]], str]
    environment: Callable[[DoorConfig, dict[str, Any]], dict[str, str]]
    properties: Callable[[DoorConfig, dict[str, Any]], tuple[str, ...]] = lambda _config, _request: ()


_LAUNCHES: dict[str, _LaunchSpec] = {
    "install-native-g1-policies": _LaunchSpec(
        "blueprint-operator-door-native-g1-policy", "door-install-public-piper-policy.sh", "15min",
        lambda request: request["request"]["request_digest"].removeprefix("sha256:")[:12],
        _public_policy_environment, _native_g1_properties),
    "install-public-trlc-policy": _LaunchSpec(
        "blueprint-operator-door-public-trlc-policy", "door-install-public-piper-policy.sh", "5min",
        lambda request: request["request"]["request_digest"].removeprefix("sha256:")[:12],
        _public_policy_environment, _public_policy_properties),
    "install-public-franka-policy": _LaunchSpec(
        "blueprint-operator-door-public-franka-policy", "door-install-public-piper-policy.sh", "5min",
        lambda request: request["request"]["request_digest"].removeprefix("sha256:")[:12],
        _public_policy_environment, _public_policy_properties),
    "install-scripted-robot-baselines": _LaunchSpec(
        "blueprint-operator-door-scripted-baselines", "door-install-public-piper-policy.sh", "10min",
        lambda request: request["request"]["request_digest"].removeprefix("sha256:")[:12],
        _public_policy_environment, _public_policy_properties),
    "install-public-yam-policy": _LaunchSpec(
        "blueprint-operator-door-public-yam-policy", "door-install-public-piper-policy.sh", "5min",
        lambda request: request["request"]["request_digest"].removeprefix("sha256:")[:12],
        _public_policy_environment, _public_policy_properties),
    "install-public-piper-policy": _LaunchSpec(
        "blueprint-operator-door-public-policy", "door-install-public-piper-policy.sh", "5min",
        lambda request: request["request"]["request_digest"].removeprefix("sha256:")[:12],
        _public_policy_environment, _public_policy_properties),
    "public-embodiment-preparation": _LaunchSpec(
        "blueprint-operator-door-embodiment-preparation", "door-public-embodiment-preparation.sh", "2min",
        lambda request: request["request"]["request_digest"].removeprefix("sha256:")[:12],
        _embodiment_preparation_environment, _embodiment_preparation_properties),
    "retire-scene": _LaunchSpec("blueprint-operator-door-scene-retire", "door-scene-lifecycle.sh", "2h",
                                lambda request: request["consent_id"][:12],
                                _lifecycle_environment, _lifecycle_properties),
    "restore-scene": _LaunchSpec("blueprint-operator-door-scene-restore", "door-scene-lifecycle.sh", "2h",
                                 lambda request: request["consent_id"][:12],
                                 _lifecycle_environment, _lifecycle_properties),
    "unit": _LaunchSpec(
        "blueprint-operator-door-notifier-repair", "door-repair-notifier-binding.sh", "60s",
        lambda _request: "postchecks",
        lambda config, request: {
            "DOOR_INSTALL_ROOT": config.install_root,
            "DOOR_EXPECTED_POSTCHECK_SHA256": request["expected_postcheck_sha256"],
            "DOOR_EXPECTED_SOURCE_COMMIT": request["expected_source_commit"],
        },
        lambda config, _request: (
            "ProtectSystem=strict", "PrivateTmp=yes", "NoNewPrivileges=yes", "PrivateDevices=yes",
            "ProtectHome=yes", "PrivateNetwork=yes", "RestrictAddressFamilies=AF_UNIX",
            "CapabilityBoundingSet=", "AmbientCapabilities=",
            "InaccessiblePaths=" + " ".join("-" + path for path in config.hidden_paths),
            "ReadWritePaths=/etc/systemd/system/blueprint-pipeline-control-plane.service.d "
            + str(Path(config.spool_root) / "results"),
        ),
    ),
    "legacy-owner-census": _LaunchSpec("blueprint-operator-door-legacy-owner-census",
                                      "door-legacy-owner-census.sh", "5min",
                                      lambda _request: "current", _legacy_owner_environment,
                                      _legacy_owner_properties),
    "owner-census-decision": _LaunchSpec("blueprint-operator-door-owner-census", "door-owner-census.sh", "10s",
                                         lambda request: request["consent_id"][:12], _owner_environment, _owner_properties),
    "deploy": _LaunchSpec("blueprint-operator-door-deploy", "door-deploy.sh", "3h",
                          lambda request: request["commit"][:12], _deploy_environment),
    "door-upgrade": _LaunchSpec("blueprint-operator-door-upgrade", "door-upgrade.sh", "30min",
                                lambda request: request["commit"][:12], _source_environment),
    # Named by a hash of the scene id, never the id itself: caller text in a unit name could
    # match `blueprint-*deploy*` and make every deploy wait on a retirement.
    "retire-scene-workspace": _LaunchSpec("blueprint-operator-door-retire", "door-retire-scene-workspace.sh", "2h",
                                          lambda request: hashlib.sha256(request["scene_id"].encode("utf-8"))
                                          .hexdigest()[:12], _retire_environment, _retire_properties),
    "restore-scene-workspace": _LaunchSpec("blueprint-operator-door-restore", "door-restore-scene-workspace.sh", "2h",
                                           lambda request: hashlib.sha256(request["scene_id"].encode("utf-8"))
                                           .hexdigest()[:12], _restore_environment, _retire_properties),
    # Promotion reads and writes up to one archive's bytes to B2 and back: at most 2 h.
    "selected-handoff": _LaunchSpec("blueprint-operator-door-selected-handoff", "door-selected-handoff.sh",
                                      "2h", lambda request: hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:12],
                                      _selected_handoff_environment, _selected_handoff_properties),
    "provider-output-resume": _LaunchSpec("blueprint-operator-door-output-resume", "door-provider-output-resume.sh",
                                          "2h", lambda request: hashlib.sha256(
                                              f"{request['run']}/{request['attempt']}".encode("utf-8"))
                                          .hexdigest()[:12], _output_resume_environment, _output_resume_properties),
    # Named by a hash of the intent id; the caller text never reaches a unit name.
    "extend-scene-execution-window": _LaunchSpec(
        "blueprint-operator-door-scene-window", "door-extend-scene-execution-window.sh", "5min",
        lambda request: hashlib.sha256(request["intent_id"].encode("utf-8")).hexdigest()[:12],
        _scene_window_environment, _scene_window_properties),
    # Named by a hash of the intent id; the caller text never reaches a unit name.
    "extend-scene-execution-budget": _LaunchSpec(
        "blueprint-operator-door-scene-budget", "door-extend-scene-execution-budget.sh", "5min",
        lambda request: hashlib.sha256(request["intent_id"].encode("utf-8")).hexdigest()[:12],
        _scene_budget_environment, _scene_budget_properties),
    "lane-scratch": _LaunchSpec("blueprint-operator-door-scratch", "door-lane-scratch.sh", "2min",
                                lambda request: hashlib.sha256((request["lane"] + "/" + request.get("name", ""))
                                                                .encode("utf-8")).hexdigest()[:12],
                                _scratch_environment, _scratch_properties),
}



def _stage_release_provenance(config: DoorConfig, request_id: str, request: dict[str, Any]) -> dict[str, str]:
    if request["kind"] != "deploy" or "release_provenance_json" not in request:
        return {}
    destination = Path(config.spool_root) / "results" / f"{request_id}.release-provenance.json"
    # Exclusive creation refuses an existing file or symlink rather than replacing it.
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(request["release_provenance_json"].encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    return {"DOOR_RELEASE_PROVENANCE_FILE": str(destination),
            "DOOR_RELEASE_PROVENANCE_SHA256": request["release_provenance_sha256"]}


def _recovery_json(directory_fd: int, name: str) -> dict[str, Any]:
    """Read retained root evidence without trusting paths or permissive JSON."""
    def pairs(items):
        document = {}
        for key, value in items:
            if key in document:
                raise ValueError("duplicate key")
            document[key] = value
        return document

    def nonfinite(_value):
        raise ValueError("nonfinite value")

    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_mode & 0o022 or info.st_nlink != 1 or not 0 < info.st_size <= 16384):
                raise ValueError("untrusted evidence")
            raw = stream.read(16385)
        if not 0 < len(raw) <= 16384:
            raise ValueError("unbounded evidence")
        document = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
        if not isinstance(document, dict):
            raise ValueError("invalid document")
        return document
    except (OSError, ValueError) as error:
        raise RequestRefused("selected_handoff_recovery_evidence_invalid") from error


def _stage_selected_recovery(config: DoorConfig, request_id: str, request: dict[str, Any]) -> None:
    """Consume one linked successor for a proven preclaim failure, before launch."""
    if request["kind"] != "selected-handoff" or "resume_request_id" not in request:
        return
    prior = request["resume_request_id"]
    if prior == request_id:
        raise RequestRefused("selected_handoff_recovery_self_reference")
    results = Path(config.spool_root) / "results"
    selection = {key: value for key, value in request.items() if key != "resume_request_id"}
    digest = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()
    try:
        directory_fd = os.open(results, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise RequestRefused("selected_handoff_recovery_evidence_invalid") from error
    try:
        info = os.fstat(directory_fd)
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise RequestRefused("selected_handoff_recovery_evidence_invalid")
        outcome = _recovery_json(directory_fd, f"{prior}.outcome.json")
        native = _recovery_json(directory_fd, f"{prior}.selected-handoff.json")
        if (outcome.get("schema") != "blueprint_operator_door_outcome.v1"
                or outcome.get("status") != "failed" or outcome.get("code") != "selected_handoff_blocked"
                or type(outcome.get("exit_code")) is not int or outcome["exit_code"] != 1
                or outcome.get("result") != str(results / f"{prior}.selected-handoff.json")
                or native.get("schema_version") != "selected_handoff_recovery.v1"
                or native.get("mode") != "dispatch" or native.get("status") != "blocked"
                # Both native returns occur before _claim_job_lease. Staging
                # can leave local birth/data state, so the successor still
                # refuses nonempty workspaces and requires an unattempted
                # delivery; this proof does not assert zero prior charges.
                or type(native.get("native_status")) is not str
                or native["native_status"] not in {
                    "capture_owner_observation_unavailable_retryable",
                    "capture_source_membership_unavailable_retryable",
                }
                or native.get("native_disposition") != "retryable"
                or native.get("source_membership_verified") is not True
                or native.get("current_admission_verified") is not True
                or native.get("selector_sha256") != digest
                or native.get("blockers") != ["selected_handoff_native_admission_or_result_blocked"]):
            raise RequestRefused("selected_handoff_recovery_not_preclaim_failure")
        marker = {"schema": "blueprint_selected_handoff_recovery.v1", "prior_request_id": prior,
                  "request_id": request_id, "selector_sha256": digest, "created_at": _now()}
        try:
            fd = os.open(f"{prior}.selected-handoff-recovery.json",
                         os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
        except FileExistsError as error:
            raise RequestRefused("selected_handoff_recovery_already_consumed") from error
        # Never remove this marker after a write/launch failure: admission is uncertain.
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(marker, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _script_env(config: DoorConfig, request_id: str, request: dict[str, Any],
                requested_by: str = "", extra: dict[str, str] | None = None) -> list[str]:
    """Only the values the request's own kind defines, each already validated."""

    _stage_selected_recovery(config, request_id, request)
    values = {
        "DOOR_REQUEST_ID": request_id,
        "DOOR_RESULTS_DIR": str(Path(config.spool_root) / "results"),
        **_LAUNCHES[request["kind"]].environment(config, request),
        **_stage_release_provenance(config, request_id, request),
        **(extra or {}),
    }
    if request["kind"] in {"extend-scene-execution-window", "extend-scene-execution-budget"}:
        # The authenticated token name, recorded in the extension as who asked.
        if not isinstance(requested_by, str) or _CALLER.fullmatch(requested_by) is None:
            raise RequestRefused("scene_window_caller_invalid" if request["kind"].endswith("window")
                                 else "scene_budget_caller_invalid")
        values["DOOR_REQUESTED_BY"] = requested_by
    return [f"--setenv={key}={value}" for key, value in values.items()]


def _launch(config: DoorConfig, runner: CommandRunner, request_id: str, request: dict[str, Any],
            requested_by: str = "", *, extra_environment: dict[str, str] | None = None) -> dict[str, Any]:
    spec = _LAUNCHES[request["kind"]]
    # A .service name is what `unit_properties` accepts, so the request's live state can be shown.
    unit = f"{spec.unit_prefix}-{spec.label(request)}-{request_id[-8:]}.service"
    argv = [
        "systemd-run", f"--unit={unit}", "--collect", "--service-type=exec",
        f"--property=TimeoutStartSec={spec.runtime_max}", "--setenv=PYTHONDONTWRITEBYTECODE=1",
        f"--property=RuntimeMaxSec={spec.runtime_max}",
        *(f"--property={value}" for value in spec.properties(config, request)),
        *_script_env(config, request_id, request, requested_by, extra_environment),
        "--", "/bin/bash", f"{config.install_root}/{spec.script}",
    ]
    result = runner.run(argv, timeout=60)
    if result.returncode != 0:
        return {"status": "failed", "unit": unit, "returncode": result.returncode,
                "stderr_tail": result.stderr[-2000:]}
    return {"status": "launched", "unit": unit}


def _active_deploy_unit(runner: CommandRunner) -> str | None:
    result = runner.run(
        ["systemctl", "list-units", "--all", "--plain", "--no-legend", "--no-pager",
         f"--state={_ACTIVE_STATES}", "--", "blueprint-*deploy*"],
        timeout=15,
    )
    if result.returncode != 0:
        # A failed or timed-out listing must not read as "nothing is deploying".
        return _UNKNOWN_DEPLOY_ACTIVITY
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].startswith("blueprint-"):
            return parts[0]
    return None


def _parkable(request: dict[str, Any]) -> bool:
    """A canonical deploy carries one commit's provenance, so it never waits to be replaced."""

    return request["kind"] == "deploy" and "release_provenance_json" not in request


def _active_release_commit(config: DoorConfig) -> str:
    """What runs now, read without git; empty when the link cannot be read."""

    try:
        name = Path(os.readlink(config.active_release_link)).name
    except OSError:
        return ""
    return name if re.fullmatch(r"[0-9a-f]{40}", name) else ""


def _park_deploy(config: DoorConfig, request_id: str, request: dict[str, Any], busy: str) -> dict[str, Any]:
    """Keep what the deploy needs in a record only root can write.

    The request's own file in completed/ is the door's, so it is never read again.
    The active commit now tells a stale request from a rollback when this one starts.
    """

    parked = Path(config.spool_root) / "parked"
    if len(list(parked.glob("*.json"))) >= _PARKED_MAX:
        return {"status": "refused", "code": f"deploy_in_progress:{busy}"}
    record = {"schema": _PARKED_SCHEMA, "id": request_id, "commit": request["commit"],
              "wait_for_idle": request["wait_for_idle"], "parked_at_epoch": time.time(),
              "active_release_at_park": _active_release_commit(config),
              **({"activation_hold_override": request["activation_hold_override"]}
                 if request.get("activation_hold_override") else {})}
    handle, temporary = tempfile.mkstemp(dir=parked, prefix=f".{request_id}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(record, stream, sort_keys=True)
        os.chmod(temporary, 0o644)
        os.replace(temporary, parked / f"{request_id}.json")
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return {"status": "parked", "parked_behind": busy, "commit": request["commit"]}


def _parked_request(path: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """The checks a request passes on its way out of pending/, repeated before it starts."""

    request_id = validate_request_id(path.stem)
    record = load_request_file(path)
    parked_at = record.get("parked_at_epoch")
    if (_PARKED_ID.fullmatch(request_id) is None or record.get("schema") != _PARKED_SCHEMA
            or record.get("id") != request_id or isinstance(parked_at, bool)
            or not isinstance(parked_at, (int, float)) or not math.isfinite(parked_at)):
        raise RequestRefused("deploy_parked_record_invalid")
    request = validate_request({"kind": "deploy", "commit": record.get("commit"),
                                "wait_for_idle": record.get("wait_for_idle"),
                                **({"activation_hold_override": record["activation_hold_override"]}
                                   if "activation_hold_override" in record else {})})
    return request_id, request, record


def _drop_parked_record(results: Path, path: Path, error: Exception) -> None:
    """One file nothing can use must not hold up the requests after it."""

    print(f"parked_record_dropped:{path.name}:{type(error).__name__}", file=sys.stderr)
    if _PARKED_ID.fullmatch(path.stem):
        _write_result(results, path.stem, {"status": "refused", "code": "deploy_parked_record_invalid"})
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _parked_still_waiting(results: Path, path: Path, now: float) -> tuple[str, dict[str, Any]] | None:
    """The record's request if it can still start. Otherwise end it, saying why."""

    request_id, request, record = _parked_request(path)
    if _json_or_none(results / f"{request_id}.outcome.json") is not None:
        path.unlink(missing_ok=True)  # finished; only the record's removal was lost
        return None
    if record.get("deploying_commit"):
        code = "parked_deploy_interrupted"
    elif now - record["parked_at_epoch"] > _PARKED_MAX_AGE_SECONDS:
        code = "deploy_parked_expired"
    else:
        return request_id, request
    _write_result(results, request_id, {"status": "refused", "code": code})
    path.unlink(missing_ok=True)
    return None


def _activation_hold_recheck_pending(parked: Path, now: float) -> bool:
    """True while door-deploy.sh's last activation-hold check says not to look again yet.

    It saves a fetch every minute for hours. The file is root's, in root's directory;
    a time too far ahead is not trusted, so a bad file delays a deploy by one recheck at most.
    """

    hold = _json_or_none(parked / activation_hold.HOLD_FILE)
    recheck = (hold or {}).get("recheck_after_epoch")
    if (not hold or hold.get("schema") != activation_hold.HOLD_SCHEMA or isinstance(recheck, bool)
            or not isinstance(recheck, (int, float)) or not math.isfinite(recheck)):
        return False
    return now < recheck <= now + activation_hold.RECHECK_SECONDS


def _release_parked(config: DoorConfig, runner: CommandRunner) -> None:
    """When no deploy runs, start the newest parked deploy.

    This process has no network and no git, on purpose, and a fresh commit may be in
    no local clone. So it decides nothing about commits: door-deploy.sh does, after
    its fetch, through parked_settle.py. A record stays here until its outcome is
    written. One that says it was deploying, with no deploy running, was killed.
    """

    spool = Path(config.spool_root)
    results = spool / "results"
    paths = sorted((spool / "parked").glob("*.json"), key=lambda path: path.name)
    if not paths or _active_deploy_unit(runner) is not None:
        return
    now = time.time()
    live: list[tuple[str, dict[str, Any]]] = []
    for path in paths:
        try:
            waiting = _parked_still_waiting(results, path, now)
            if waiting:
                live.append(waiting)
        except Exception as error:  # noqa: BLE001 - whatever is wrong with this file, the queue goes on
            try:
                _drop_parked_record(results, path, error)
            except (OSError, RequestRefused) as drop_error:
                print(f"parked_record_kept:{path.name}:{type(drop_error).__name__}", file=sys.stderr)
    if not live:
        return
    override = next((row["activation_hold_override"] for _, row in live if row.get("activation_hold_override")), "")
    if not override and _activation_hold_recheck_pending(spool / "parked", now):
        return  # a live scene-configuration activation holds them; door-deploy.sh rechecks later
    request_id, request = live[-1]
    request = {**request, "wait_for_idle": any(row["wait_for_idle"] for _, row in live)}
    if override:
        # Any of them may be the urgent fix, and the deploy that goes out covers it.
        request["activation_hold_override"] = override
    outcome = _launch(config, runner, request_id,
                      # The deploy that goes out may be any of theirs, so any of them can ask it to drain.
                      request,
                      extra_environment={
        "DOOR_PARKED": "1",
        "DOOR_PARKED_DIR": str(spool / "parked"),
        "DOOR_ACTIVE_RELEASE_LINK": config.active_release_link,
    })
    if outcome["status"] == "launched":
        _write_result(results, request_id, outcome)
    else:
        # A failed launch says nothing about this request: it stays parked for the next wake-up.
        print(f"parked_launch_failed:{request_id}:{outcome.get('returncode')}:{outcome.get('stderr_tail', '')[-200:]}",
              file=sys.stderr)


def _has_hold_guard(runner: CommandRunner, unit: str, root: Path) -> bool:
    """Only hold a loaded unit whose effective source includes our crash guard."""

    result = runner.run(["systemctl", "cat", "--", unit], timeout=30)
    if result.returncode != 0:
        return False
    expected = f"!{root / f'{unit}.json'}"
    section = ""
    guarded = False
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line
        elif section == "[Unit]" and line.startswith("ConditionPathExists="):
            value = line.partition("=")[2].strip()
            if not value:
                guarded = False  # a later drop-in can reset earlier conditions
            elif value == expected:
                guarded = True
    if not guarded:
        return False
    loaded = runner.run(["systemctl", "show", "--property=NeedDaemonReload", "--value", "--", unit], timeout=30)
    return loaded.returncode == 0 and loaded.stdout.strip() == "no"


def _act_hold(
    config: DoorConfig, runner: CommandRunner, request_id: str, request: dict[str, Any], requested_by: str,
) -> dict[str, Any]:
    root = Path(config.spool_root) / "holds"
    unit = request["unit"]
    with holds.locked(root):
        if holds.read(root / "releasing", unit) is not None:
            return {"status": "refused", "code": "hold_release_in_progress"}
        current = holds.read(root, unit)
        prior_active = (current is not None and current["status"] == "active"
                        and (current["expires_at_epoch"] > time.time()
                             or current.get("require_explicit_release") is True))
        if prior_active and current["owner"] != request["owner"]:
            return {"status": "refused", "code": f"hold_active:{current['owner']}"}
        if (prior_active and current.get("require_explicit_release") is True
                and request.get("require_explicit_release") is not True):
            return {"status": "refused", "code": "hold_explicit_release_required"}
        if not _has_hold_guard(runner, unit, root):
            return {"status": "refused", "code": "hold_unit_guard_missing"}
        if current is not None and current["status"] == "active" and isinstance(current.get("enabled_before"), bool):
            enabled_before = current["enabled_before"]
        else:
            enabled = runner.run(["systemctl", "is-enabled", "--", unit], timeout=30)
            state = enabled.stdout.strip()
            if state not in {"enabled", "disabled"}:
                return {"status": "refused", "code": "hold_unit_enabled_state_unknown"}
            enabled_before = state == "enabled"
        activity = runner.run(["systemctl", "is-active", "--", unit], timeout=30)
        if activity.stdout.strip() not in {"active", "inactive"}:
            return {"status": "refused", "code": "hold_unit_active_state_unknown"}
        active_before = activity.stdout.strip() == "active"

        def restore_unheld() -> int:
            if current is not None and current["status"] == "active":
                return 0  # the previous hold still owns this trigger
            if enabled_before:
                restored_boot = runner.run(["systemctl", "enable", "--", unit], timeout=30)
                if restored_boot.returncode != 0:
                    return restored_boot.returncode
            if active_before:
                return runner.run(["systemctl", "--no-block", "start", "--", unit], timeout=30).returncode
            return 0

        stopped = runner.run(["systemctl", "stop", "--", unit], timeout=30)
        if stopped.returncode != 0:
            rollback_returncode = restore_unheld()
            return {"status": "failed", "code": "hold_stop_failed", "returncode": stopped.returncode,
                    "rollback_returncode": rollback_returncode, "stderr_tail": stopped.stderr[-2000:]}
        state = runner.run(["systemctl", "is-active", "--", unit], timeout=30)
        if state.stdout.strip() != "inactive":
            rollback_returncode = restore_unheld()
            return {"status": "failed", "code": "hold_stop_incomplete", "returncode": state.returncode,
                    "rollback_returncode": rollback_returncode, "stderr_tail": state.stderr[-2000:]}
        now = int(time.time())
        expires_at_epoch = max(
            now + request["expires_in_seconds"],
            current["expires_at_epoch"] if prior_active else 0,
        )
        record = {"schema": holds.SCHEMA, "unit": unit, "owner": request["owner"],
                  "reason": request["reason"], "requested_by": requested_by, "request_id": request_id,
                  "created_at": holds.timestamp(now), "expires_at": holds.timestamp(expires_at_epoch),
                  "expires_at_epoch": expires_at_epoch, "enabled_before": enabled_before,
                  "status": "active", **({"require_explicit_release": True}
                                         if request.get("require_explicit_release") is True else {})}

        def rollback() -> int:
            if prior_active:
                # The old expiry timer still exists. Restore its generation;
                # starting here would silently undo the owner's prior hold.
                saved = holds.read(root, unit)
                if saved is None or saved["request_id"] != current["request_id"]:
                    holds.write(root, unit, current)
                return 0
            if holds.read(root, unit) is not None:
                holds.begin_release(root, unit, record, released_by="runner", status="failed_released",
                                    restart_on_release=active_before)
                return holds.finish_release(root, unit, command=lambda argv: runner.run(argv, timeout=30))
            return restore_unheld()

        try:
            holds.write(root, unit, record)
            stopped = runner.run(["systemctl", "stop", "--", unit], timeout=30)
            if stopped.returncode != 0:
                rollback_returncode = rollback()
                return {"status": "failed", "code": "hold_stop_failed", "returncode": stopped.returncode,
                        "rollback_returncode": rollback_returncode, "stderr_tail": stopped.stderr[-2000:]}
            state = runner.run(["systemctl", "is-active", "--", unit], timeout=30)
            if state.stdout.strip() != "inactive":
                rollback_returncode = rollback()
                return {"status": "failed", "code": "hold_stop_incomplete", "returncode": state.returncode,
                        "rollback_returncode": rollback_returncode, "stderr_tail": state.stderr[-2000:]}
            disabled = runner.run(["systemctl", "disable", "--", unit], timeout=30)
            if disabled.returncode != 0:
                rollback_returncode = rollback()
                return {"status": "failed", "code": "hold_disable_failed", "returncode": disabled.returncode,
                        "rollback_returncode": rollback_returncode, "stderr_tail": disabled.stderr[-2000:]}
            launch = runner.run([
                "systemd-run", f"--unit=blueprint-operator-door-hold-expiry-{request_id[-8:]}",
                f"--on-active={expires_at_epoch - now}s", "--collect",
                f"--setenv=DOOR_HOLD_UNIT={unit}", f"--setenv=DOOR_HOLD_REQUEST_ID={request_id}",
                f"--setenv=DOOR_HOLDS_DIR={root}", "--", "/bin/bash",
                f"{config.install_root}/door-hold-expire.sh",
            ], timeout=60)
        except Exception:
            rollback()
            raise
        if launch.returncode != 0:
            rollback_returncode = rollback()
            return {"status": "failed", "code": "hold_expiry_schedule_failed", "returncode": launch.returncode,
                    "rollback_returncode": rollback_returncode, "stderr_tail": launch.stderr[-2000:]}
        return {"status": "done", "hold": record}


def _act_release_hold(
    config: DoorConfig, runner: CommandRunner, request: dict[str, Any], requested_by: str,
) -> dict[str, Any]:
    root = Path(config.spool_root) / "holds"
    unit = request["unit"]
    with holds.locked(root):
        record = holds.read(root, unit)
        if record is None or record["status"] != "active":
            return {"status": "refused", "code": "hold_not_active"}
        holds.begin_release(root, unit, record, released_by=requested_by, status="released")
        pending = holds.read(root / "releasing", unit)
        if pending is None:
            raise holds.HoldError("hold_release_intent_missing")
        result = holds.finish_release(root, unit, command=lambda argv: runner.run(argv, timeout=30))
        if result != 0:
            return {"status": "failed", "code": "hold_release_start_failed", "returncode": result}
        return {"status": "done", "hold": {**record, "status": "released",
                                           "released_by": requested_by, "released_at": pending["released_at"]}}


def _observe_unit(runner: CommandRunner, unit: str) -> dict[str, str]:
    observation = runner.run(["systemctl", "show", "--no-pager",
        "--property=" + _UNIT_OBSERVATION_PROPERTIES, "--", unit], timeout=10)
    if observation.returncode != 0:
        return {}
    return dict(line.split("=", 1) for line in observation.stdout.splitlines() if "=" in line)


def _act(config: DoorConfig, runner: CommandRunner, request_id: str, request: dict[str, Any],
         requested_by: str = "") -> dict[str, Any]:
    if request["kind"] in {"owner-census-decision", "legacy-owner-census"} and config.owner_census_decisions_enabled != 1:
        return {"status": "refused", "code": "owner_consent_disabled"}
    if request["kind"] == "unit":
        if request["action"] == "repair-notifier-binding":
            busy = _active_deploy_unit(runner)
            if busy is not None:
                return {"status": "refused", "code": "notifier_repair_deploy_in_progress"}
            return _launch(config, runner, request_id, request)
        prior = _observe_unit(runner, request["unit"])
        result = runner.run(
            ["systemctl", request["action"], "--", request["unit"]], timeout=30
        )
        if result.returncode == 124:
            properties = _observe_unit(runner, request["unit"])
            # systemctl's caller timed out, but its submitted job can continue.
            # Bind only a newly observed invocation: an old success receipt must
            # not complete this operation while a new job is still queued.
            if (not prior or not properties.get("InvocationID")
                    or properties.get("InvocationID") == prior.get("InvocationID")):
                properties.pop("InvocationID", None)
            return {"status": "unknown", "code": "unit_action_timed_out",
                    "unit_action": request, "unit": request["unit"],
                    "prior_unit_observation": prior, "unit_observation": properties,
                    "returncode": result.returncode, "stderr_tail": result.stderr[-2000:]}
        if result.returncode == 0:
            properties = _observe_unit(runner, request["unit"])
            success = bool(properties) and (
                (request["action"] in {"start", "restart"} and properties.get("ActiveState") == "active"
                 and (not request["unit"].endswith(".service") or properties.get("Type") != "oneshot"))
                or (request["action"] == "reset-failed" and properties.get("ActiveState") in {"active", "inactive"})
                or (request["action"] == "stop" and properties.get("ActiveState") == "inactive")
                or (request["action"] == "start" and properties.get("ActiveState") == "inactive"
                    and properties.get("Result") == "success" and properties.get("ExecMainStatus") == "0"
                    and bool(properties.get("InvocationID"))))
            failure = properties.get("ActiveState") == "failed" and bool(properties.get("InvocationID"))
            running = properties.get("ActiveState") in {"activating", "active"} and properties.get("Type") == "oneshot" and bool(properties.get("InvocationID"))
            return {"status": "done" if success else "failed" if failure else "running" if running else "unknown", "unit_action": request,
                    "unit": request["unit"],
                    "unit_observation": properties, "returncode": result.returncode}

        return {"status": "accepted" if result.returncode == 0 else "failed", "unit_action": request,
                "returncode": result.returncode, "stderr_tail": result.stderr[-2000:]}
    if request["kind"] == "hold":
        return _act_hold(config, runner, request_id, request, requested_by)
    if request["kind"] == "release-hold":
        return _act_release_hold(config, runner, request, requested_by)
    if request["kind"] == "deploy":
        busy = _active_deploy_unit(runner)
        if busy is not None:
            if _parkable(request):
                return _park_deploy(config, request_id, request, busy)
            return {"status": "refused", "code": f"deploy_in_progress:{busy}"}
    if request["kind"] not in _LAUNCHES:
        return {"status": "refused", "code": "kind_unknown"}
    return _launch(config, runner, request_id, request, requested_by)


def _process_one(config: DoorConfig, runner: CommandRunner, claimed: Path, request_id: str) -> None:
    spool = Path(config.spool_root)
    results = spool / "results"
    if (spool / "completed" / claimed.name).exists():
        claimed.unlink()
        return
    try:
        document = load_request_file(claimed)
        if document.get("schema") != SCHEMA:
            raise RequestRefused("spool_schema_invalid")
        if document.get("id") != request_id:
            raise RequestRefused("spool_id_mismatch")
        request = validate_request(document.get("request"))
        if request_id.split("-", 1)[1].rsplit("-", 1)[0] != request["kind"]:
            raise RequestRefused("spool_kind_mismatch")
    except RequestRefused as refusal:
        os.replace(claimed, spool / "completed" / claimed.name)
        _write_result(results, request_id, {"status": "refused", "code": refusal.code})
        return
    os.replace(claimed, spool / "completed" / claimed.name)
    unit_identity = ({"unit": request["unit"], "unit_action": request}
                     if request["kind"] == "unit" and request["action"] != "repair-notifier-binding"
                     else {})
    _write_result(results, request_id, {"status": "accepted", **unit_identity})
    try:
        outcome = _act(config, runner, request_id, request, document["requested_by"])
    except RequestRefused as refusal:
        outcome = {"status": "refused", "code": refusal.code}
    except Exception as error:  # noqa: BLE001 - record, never crash the oneshot
        outcome = {"status": "unknown", "code": f"runner_error:{type(error).__name__}"}
    _write_result(results, request_id, {**unit_identity, **outcome})


def _quarantine(spool: Path, path: Path) -> None:
    """Move an unprocessable file out of the way; delete it if even that fails."""

    try:
        os.replace(path, spool / "completed" / f"{path.name}.invalid-{secrets.token_hex(4)}")
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass


def _prune(spool: Path, retention_days: int) -> None:
    """Drop old finished requests, fail stranded claims, clear abandoned temp files."""

    now = time.time()
    cutoff = now - retention_days * 86400
    for state in ("completed", "results"):
        for path in (spool / state).iterdir():
            try:
                if path.is_file() and not path.is_symlink() and path.lstat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue
    for path in (spool / "parked").glob(".*.tmp"):
        try:
            if path.lstat().st_mtime < now - 3600:
                path.unlink()
        except OSError:
            continue
    for path in (spool / "processing").glob("*.json"):
        try:
            if path.lstat().st_mtime < now - 3600:
                request_id = validate_request_id(path.stem)
                os.replace(path, spool / "completed" / path.name)
                _write_result(spool / "results", request_id, {"status": "unknown", "code": "stranded"})
        except (OSError, RequestRefused):
            _quarantine(spool, path)
    for path in (spool / "pending").glob(".*.tmp"):
        try:
            if path.lstat().st_mtime < now - 3600:
                path.unlink()
        except OSError:
            continue


def wake_parked(config: DoorConfig, *, runner: CommandRunner | None = None) -> int:
    """Start the runner if a deploy is parked. The hold sweep's one-minute timer calls this.

    A new request starts the runner by itself, but nothing else notices a deploy
    ending. This timer already runs every minute and at boot, so nothing can strand
    a parked request by losing a race with its own re-arming.
    """

    if not any((Path(config.spool_root) / "parked").glob("*.json")):
        return 0
    runner = runner or SubprocessRunner()
    return runner.run(["systemctl", "start", "--no-block", "blueprint-operator-door-runner.service"],
                      timeout=30).returncode


def process_spool(config: DoorConfig, *, runner: CommandRunner | None = None) -> int:
    runner = runner or SubprocessRunner()
    spool = Path(config.spool_root)
    for state in ("pending", "processing", "completed", "results", "parked"):
        (spool / state).mkdir(parents=True, exist_ok=True)
    _prune(spool, config.spool_retention_days)
    for _ in range(100):  # requests that arrive while we work are picked up too
        pending = sorted(spool.joinpath("pending").glob("*.json"), key=lambda path: path.name)
        if not pending:
            break
        for path in pending:
            try:
                request_id = validate_request_id(path.stem)
            except RequestRefused:
                _quarantine(spool, path)
                continue
            claimed = spool / "processing" / path.name
            try:
                os.replace(path, claimed)
            except FileNotFoundError:
                continue  # another runner instance claimed it
            except OSError:
                _quarantine(spool, path)
                continue
            try:
                _process_one(config, runner, claimed, request_id)
            except OSError:
                _quarantine(spool, claimed)
    # After pending/: a queue that cannot start must not keep a new direct deploy, which
    # may be what repairs it, behind a unit that only comes back to say it cannot start.
    try:
        _release_parked(config, runner)
    except Exception as error:  # noqa: BLE001 - parked deploys must never stop every other request
        print(f"parked_release_failed:{type(error).__name__}", file=sys.stderr)
    return 0
