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
import os
import secrets
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import holds
from .config import DoorConfig
from .hostinfo import CommandRunner, SubprocessRunner
from .requests import (
    SCHEMA,
    RequestRefused,
    load_request_file,
    validate_request,
    validate_request_id,
)

RESULT_SCHEMA = "blueprint_operator_door_result.v1"
_ACTIVE_STATES = "active,activating,deactivating,reloading"


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
    "provider-output-resume": _LaunchSpec("blueprint-operator-door-output-resume", "door-provider-output-resume.sh",
                                          "2h", lambda request: hashlib.sha256(
                                              f"{request['run']}/{request['attempt']}".encode("utf-8"))
                                          .hexdigest()[:12], _output_resume_environment, _output_resume_properties),
    "lane-scratch": _LaunchSpec("blueprint-operator-door-scratch", "door-lane-scratch.sh", "2min",
                                lambda request: hashlib.sha256((request["lane"] + "/" + request.get("name", ""))
                                                                .encode("utf-8")).hexdigest()[:12],
                                _scratch_environment, _scratch_properties),
}


def _script_env(config: DoorConfig, request_id: str, request: dict[str, Any]) -> list[str]:
    """Only the values the request's own kind defines, each already validated."""

    values = {
        "DOOR_REQUEST_ID": request_id,
        "DOOR_RESULTS_DIR": str(Path(config.spool_root) / "results"),
        **_LAUNCHES[request["kind"]].environment(config, request),
    }
    return [f"--setenv={key}={value}" for key, value in values.items()]


def _launch(config: DoorConfig, runner: CommandRunner, request_id: str, request: dict[str, Any]) -> dict[str, Any]:
    spec = _LAUNCHES[request["kind"]]
    # A .service name is what `unit_properties` accepts, so the request's live state can be shown.
    unit = f"{spec.unit_prefix}-{spec.label(request)}-{request_id[-8:]}.service"
    argv = [
        "systemd-run", f"--unit={unit}", "--collect", "--service-type=exec",
        f"--property=TimeoutStartSec={spec.runtime_max}", "--setenv=PYTHONDONTWRITEBYTECODE=1",
        f"--property=RuntimeMaxSec={spec.runtime_max}",
        *(f"--property={value}" for value in spec.properties(config, request)),
        *_script_env(config, request_id, request),
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
    for line in result.stdout.splitlines():
        parts = line.split()
        if parts and parts[0].startswith("blueprint-"):
            return parts[0]
    return None


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
                        and current["expires_at_epoch"] > time.time())
        if prior_active and current["owner"] != request["owner"]:
            return {"status": "refused", "code": f"hold_active:{current['owner']}"}
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
                  "status": "active"}

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


def _act(config: DoorConfig, runner: CommandRunner, request_id: str, request: dict[str, Any],
         requested_by: str = "") -> dict[str, Any]:
    if request["kind"] in {"owner-census-decision", "legacy-owner-census"} and config.owner_census_decisions_enabled != 1:
        return {"status": "refused", "code": "owner_consent_disabled"}
    if request["kind"] == "unit":
        result = runner.run(
            ["systemctl", "--no-block", request["action"], "--", request["unit"]], timeout=30
        )
        return {"status": "done" if result.returncode == 0 else "failed", "unit_action": request,
                "returncode": result.returncode, "stderr_tail": result.stderr[-2000:]}
    if request["kind"] == "hold":
        return _act_hold(config, runner, request_id, request, requested_by)
    if request["kind"] == "release-hold":
        return _act_release_hold(config, runner, request, requested_by)
    if request["kind"] == "deploy":
        busy = _active_deploy_unit(runner)
        if busy is not None:
            return {"status": "refused", "code": f"deploy_in_progress:{busy}"}
    if request["kind"] not in _LAUNCHES:
        return {"status": "refused", "code": "kind_unknown"}
    return _launch(config, runner, request_id, request)


def _process_one(config: DoorConfig, runner: CommandRunner, claimed: Path, request_id: str) -> None:
    spool = Path(config.spool_root)
    results = spool / "results"
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
    _write_result(results, request_id, {"status": "accepted"})
    try:
        outcome = _act(config, runner, request_id, request, document["requested_by"])
    except Exception as error:  # noqa: BLE001 - record, never crash the oneshot
        outcome = {"status": "failed", "code": f"runner_error:{type(error).__name__}"}
    _write_result(results, request_id, outcome)


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
    for path in (spool / "processing").glob("*.json"):
        try:
            if path.lstat().st_mtime < now - 3600:
                request_id = validate_request_id(path.stem)
                os.replace(path, spool / "completed" / path.name)
                _write_result(spool / "results", request_id, {"status": "failed", "code": "stranded"})
        except (OSError, RequestRefused):
            _quarantine(spool, path)
    for path in (spool / "pending").glob(".*.tmp"):
        try:
            if path.lstat().st_mtime < now - 3600:
                path.unlink()
        except OSError:
            continue


def process_spool(config: DoorConfig, *, runner: CommandRunner | None = None) -> int:
    runner = runner or SubprocessRunner()
    spool = Path(config.spool_root)
    for state in ("pending", "processing", "completed", "results"):
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
    return 0
