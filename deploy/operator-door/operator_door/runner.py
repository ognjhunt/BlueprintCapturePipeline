"""Root oneshot that turns spooled requests into a small set of fixed commands.

Started by ``blueprint-operator-door-runner.path`` whenever ``pending/*.json``
exists. It trusts nothing in the spool: each file is opened without following
symlinks, bounded in size, checked against its own id and revalidated with the
same schemas the door used. It then either runs ``systemctl --no-block`` for a
unit action or starts one transient unit running a script installed with the
door, passing only validated values as environment variables. Every pending
file is drained, even junk, so the path unit can never spin.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Any

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


def _script_env(config: DoorConfig, request_id: str, request: dict[str, Any]) -> list[str]:
    values = {
        "DOOR_REQUEST_ID": request_id,
        "DOOR_RESULTS_DIR": str(Path(config.spool_root) / "results"),
        "DOOR_COMMIT": request["commit"],
        "DOOR_INSTALL_ROOT": config.install_root,
        "DOOR_SOURCE_CLONE": config.source_clone,
        "DOOR_REFERENCE_REPO": config.reference_repo,
        "DOOR_UPSTREAM_URL": config.upstream_url,
        "DOOR_VENV_PYTHON": config.venv_python,
        "DOOR_STATE_ROOT": config.control_plane_state,
    }
    if request["kind"] == "deploy":
        values["DOOR_MODE"] = request["mode"]
        values["DOOR_WAIT_FOR_IDLE"] = "1" if request["wait_for_idle"] else "0"
        values["DOOR_IDLE_UNITS"] = ",".join(config.idle_wait_units)
        values["DOOR_IDLE_WAIT_SECONDS"] = str(config.idle_wait_seconds)
    if request["kind"] == "stage-replay":
        values["DOOR_CHILD"] = request["child"] or ""
        values["DOOR_PARENT"] = request["parent"] or ""
    return [f"--setenv={key}={value}" for key, value in values.items()]


def _launch(config: DoorConfig, runner: CommandRunner, request_id: str, request: dict[str, Any]) -> dict[str, Any]:
    kind, short = request["kind"], request_id[-8:]
    if kind == "deploy":
        unit, script, timeout = f"blueprint-operator-door-deploy-{request['commit'][:12]}-{short}", "door-deploy.sh", "3h"
    elif kind == "stage-replay":
        unit, script, timeout = f"blueprint-operator-door-replay-{short}", "door-replay.sh", "1h"
    else:
        unit, script, timeout = f"blueprint-operator-door-upgrade-{request['commit'][:12]}-{short}", "door-upgrade.sh", "30min"
    argv = [
        "systemd-run", f"--unit={unit}", "--collect", "--service-type=exec",
        f"--property=TimeoutStartSec={timeout}", "--setenv=PYTHONDONTWRITEBYTECODE=1",
        *_script_env(config, request_id, request),
        "--", "/bin/bash", f"{config.install_root}/{script}",
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


def _act(config: DoorConfig, runner: CommandRunner, request_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if request["kind"] == "unit":
        result = runner.run(
            ["systemctl", "--no-block", request["action"], "--", request["unit"]], timeout=30
        )
        return {"status": "done" if result.returncode == 0 else "failed", "unit_action": request,
                "returncode": result.returncode, "stderr_tail": result.stderr[-2000:]}
    if request["kind"] == "deploy":
        busy = _active_deploy_unit(runner)
        if busy is not None:
            return {"status": "refused", "code": f"deploy_in_progress:{busy}"}
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
    except RequestRefused as refusal:
        os.replace(claimed, spool / "completed" / claimed.name)
        _write_result(results, request_id, {"status": "refused", "code": refusal.code})
        return
    os.replace(claimed, spool / "completed" / claimed.name)
    _write_result(results, request_id, {"status": "accepted"})
    try:
        outcome = _act(config, runner, request_id, request)
    except Exception as error:  # noqa: BLE001 - record, never crash the oneshot
        outcome = {"status": "failed", "code": f"runner_error:{type(error).__name__}"}
    _write_result(results, request_id, outcome)


def process_spool(config: DoorConfig, *, runner: CommandRunner | None = None) -> int:
    runner = runner or SubprocessRunner()
    spool = Path(config.spool_root)
    for state in ("pending", "processing", "completed", "results"):
        (spool / state).mkdir(parents=True, exist_ok=True)
    pending = sorted(
        spool.joinpath("pending").glob("*.json"),
        key=lambda path: (path.lstat().st_mtime, path.name),
    )
    for path in pending:
        try:
            request_id = validate_request_id(path.stem)
        except RequestRefused:
            junk = spool / "completed" / f"{path.name}.invalid-{secrets.token_hex(4)}"
            os.replace(path, junk)
            continue
        claimed = spool / "processing" / path.name
        try:
            os.replace(path, claimed)
        except FileNotFoundError:
            continue  # another runner instance claimed it
        _process_one(config, runner, claimed, request_id)
    return 0
