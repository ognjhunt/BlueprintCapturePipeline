"""Privileged requests: strict schemas and a spool the root runner can trust."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door.config import DoorConfig  # noqa: E402
from operator_door.requests import (  # noqa: E402
    RequestRefused,
    enqueue,
    list_requests,
    load_request_file,
    request_state,
    required_scope,
    validate_request,
)

SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture()
def config(tmp_path: Path) -> DoorConfig:
    for state in ("pending", "processing", "completed", "results"):
        (tmp_path / "requests" / state).mkdir(parents=True)
    return DoorConfig(state_root=str(tmp_path))


def test_deploy_defaults_to_main_mode_and_waiting_for_idle() -> None:
    assert validate_request({"kind": "deploy", "commit": SHA}) == {
        "kind": "deploy", "commit": SHA, "mode": "main", "wait_for_idle": True,
    }
    assert required_scope("deploy") == "deploy"


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"kind": "deploy", "commit": SHA[:12]}, "commit_invalid"),
        ({"kind": "deploy", "commit": SHA.upper()}, "commit_invalid"),
        ({"kind": "deploy", "commit": SHA, "mode": "promote"}, "mode_invalid"),
        ({"kind": "deploy", "commit": SHA, "wait_for_idle": "yes"}, "wait_for_idle_invalid"),
        ({"kind": "deploy", "commit": SHA, "flags": "--arm-path-units"}, "request_key_unknown:flags"),
        ({"kind": "shell", "command": "id"}, "kind_unknown"),
        ({"commit": SHA}, "kind_unknown"),
        ("not an object", "request_not_object"),
    ],
)
def test_invalid_requests_are_refused(body: object, code: str) -> None:
    with pytest.raises(RequestRefused) as caught:
        validate_request(body)  # type: ignore[arg-type]
    assert caught.value.code == code


def test_unit_actions_are_limited_to_safe_shapes() -> None:
    assert validate_request({"kind": "unit", "unit": "blueprint-gpu-spend-guard.service", "action": "start"})
    assert validate_request(
        {"kind": "unit", "unit": "blueprint-pubsub-handoff-listener.timer", "action": "stop"}
    )["action"] == "stop"
    assert required_scope("unit") == "operate"


@pytest.mark.parametrize(
    ("unit", "action", "code"),
    [
        ("blueprint-pipeline-intake.service", "stop", "unit_action_not_allowed"),
        ("blueprint-pipeline-intake.service", "restart", "unit_action_not_allowed"),
        ("blueprint-pipeline-intake.service", "kill", "unit_action_invalid"),
        ("sshd.service", "start", "unit_name_invalid"),
        ("blueprint-operator-door-runner.path", "stop", "unit_is_door"),
        ("blueprint-operator-door.service", "start", "unit_is_door"),
    ],
)
def test_unsafe_unit_actions_are_refused(unit: str, action: str, code: str) -> None:
    with pytest.raises(RequestRefused) as caught:
        validate_request({"kind": "unit", "unit": unit, "action": action})
    assert caught.value.code == code


def test_stage_replay_needs_exactly_one_target_and_a_commit() -> None:
    child = "sam31-" + "ab" * 16
    assert validate_request({"kind": "stage-replay", "commit": SHA, "child": child}) == {
        "kind": "stage-replay", "commit": SHA, "child": child, "parent": None,
    }
    assert required_scope("stage-replay") == "operate"
    for body in (
        {"kind": "stage-replay", "commit": SHA},
        {"kind": "stage-replay", "commit": SHA, "child": child, "parent": "prep-1234"},
        {"kind": "stage-replay", "commit": SHA, "child": "sam31-../../etc"},
    ):
        with pytest.raises(RequestRefused):
            validate_request(body)


def test_door_upgrade_needs_a_commit() -> None:
    assert validate_request({"kind": "door-upgrade", "commit": SHA}) == {"kind": "door-upgrade", "commit": SHA}
    assert required_scope("door-upgrade") == "deploy"


def test_enqueue_writes_an_atomic_world_readable_spool_file(config: DoorConfig) -> None:
    request_id = enqueue(config, validate_request({"kind": "deploy", "commit": SHA}), requested_by="cloud")
    assert re.fullmatch(r"\d{8}T\d{6}Z-deploy-[0-9a-f]{8}", request_id)
    path = Path(config.spool_root) / "pending" / f"{request_id}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["id"] == request_id and document["requested_by"] == "cloud"
    assert document["request"] == {"kind": "deploy", "commit": SHA, "mode": "main", "wait_for_idle": True}
    assert oct(path.stat().st_mode & 0o777) == oct(0o644)
    assert [p.name for p in (Path(config.spool_root) / "pending").iterdir()] == [path.name]


def test_load_request_file_refuses_symlinks_and_oversize(config: DoorConfig, tmp_path: Path) -> None:
    pending = Path(config.spool_root) / "pending"
    target = tmp_path / "elsewhere.json"
    target.write_text("{}", encoding="utf-8")
    os.symlink(target, pending / "link.json")
    with pytest.raises(RequestRefused) as caught:
        load_request_file(pending / "link.json")
    assert caught.value.code == "spool_file_unsafe"
    (pending / "huge.json").write_text("{" + " " * 70_000 + "}", encoding="utf-8")
    with pytest.raises(RequestRefused) as caught:
        load_request_file(pending / "huge.json")
    assert caught.value.code == "spool_file_too_large"


def test_request_state_merges_request_result_outcome_and_log(config: DoorConfig) -> None:
    request_id = enqueue(config, validate_request({"kind": "deploy", "commit": SHA}), requested_by="cloud")
    spool = Path(config.spool_root)
    os.replace(spool / "pending" / f"{request_id}.json", spool / "completed" / f"{request_id}.json")
    (spool / "results" / f"{request_id}.json").write_text(json.dumps({"status": "launched", "unit": "u"}))
    (spool / "results" / f"{request_id}.outcome.json").write_text(json.dumps({"exit_code": 0}))
    (spool / "results" / f"{request_id}.log").write_text("line 1\nPIPELINE_SYNC_TOKEN=abc12345\nline 3\n")
    state = request_state(config, request_id)
    assert state["state"] == "completed"
    assert state["result"] == {"status": "launched", "unit": "u"}
    assert state["outcome"] == {"exit_code": 0}
    assert "abc12345" not in state["log_tail"] and "line 3" in state["log_tail"]


def test_request_state_validates_ids(config: DoorConfig) -> None:
    with pytest.raises(RequestRefused) as caught:
        request_state(config, "../../etc/passwd")
    assert caught.value.code == "request_id_invalid"
    assert request_state(config, "20260923T000000Z-deploy-00000000")["state"] == "unknown"


def test_list_requests_newest_first(config: DoorConfig) -> None:
    first = enqueue(config, validate_request({"kind": "door-upgrade", "commit": SHA}), requested_by="a")
    second = enqueue(config, validate_request({"kind": "deploy", "commit": SHA}), requested_by="b")
    os.utime(Path(config.spool_root) / "pending" / f"{first}.json", (1, 1))
    listed = list_requests(config)
    assert [item["id"] for item in listed] == [second, first]
    assert listed[0]["state"] == "pending" and listed[0]["kind"] == "deploy"
