"""Privileged requests: strict schemas and a spool the root runner can trust."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/requests.py

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


def test_explicit_release_policy_is_dispatcher_only_and_typed() -> None:
    body = {"kind": "hold", "unit": "blueprint-agent-run-dispatcher.timer", "owner": "founder-stop",
            "reason": "Founder stop", "expires_in_seconds": 60, "require_explicit_release": True}
    assert validate_request(body) == body
    for change, code in [({"require_explicit_release": "yes"}, "hold_release_policy_invalid"),
                         ({"unit": "blueprint-scene-progression.timer"}, "hold_explicit_release_unit_refused")]:
        with pytest.raises(RequestRefused) as error:
            validate_request({**body, **change})
        assert error.value.code == code


@pytest.fixture()
def config(tmp_path: Path) -> DoorConfig:
    for state in ("pending", "processing", "completed", "results"):
        (tmp_path / "requests" / state).mkdir(parents=True)
    return DoorConfig(state_root=str(tmp_path))


def test_deploy_defaults_to_waiting_for_idle() -> None:
    assert validate_request({"kind": "deploy", "commit": SHA}) == {
        "kind": "deploy", "commit": SHA, "wait_for_idle": True,
    }
    assert required_scope("deploy") == "deploy"


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"kind": "deploy", "commit": SHA[:12]}, "commit_invalid"),
        ({"kind": "deploy", "commit": SHA.upper()}, "commit_invalid"),
        ({"kind": "deploy", "commit": SHA, "mode": "canary"}, "request_key_unknown:mode"),
        ({"kind": "deploy", "commit": SHA + "\n"}, "commit_invalid"),
        ({"kind": "stage-replay", "commit": SHA, "child": "sam31-" + "ab" * 16}, "kind_unknown"),
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
        {"kind": "unit", "unit": "blueprint-pubsub-handoff-listener.timer", "action": "restart"}
    )["action"] == "restart"
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
        ("blueprint-gpu-spend-guard.timer", "stop", "unit_safety_critical"),
        ("blueprint-existing-policy-canary-watchdog.timer", "restart", "unit_safety_critical"),
        ("blueprint-pubsub-handoff-listener.timer", "stop", "unit_stop_requires_hold"),
        ("blueprint-scene-progression.path", "stop", "unit_stop_requires_hold"),
        ("blueprint-task-evaluation-terminal-resource-release.path", "restart", "unit_safety_critical"),
        ("blueprint-control-plane-storage-gc.timer", "restart", "unit_safety_critical"),
        ("blueprint-control-plane-capacity.timer", "restart", "unit_safety_critical"),
        ("blueprint-completed-replay-cache-gc.timer", "restart", "unit_safety_critical"),
        ("blueprint-task-evaluation-preflight.timer", "restart", "unit_safety_critical"),
        ("blueprint-gpu-spend-guard.timer\n", "start", "unit_name_invalid"),
    ],
)
def test_unsafe_unit_actions_are_refused(unit: str, action: str, code: str) -> None:
    with pytest.raises(RequestRefused) as caught:
        validate_request({"kind": "unit", "unit": unit, "action": action})
    assert caught.value.code == code


@pytest.mark.parametrize("kind", [["deploy"], {"deploy": 1}, None, 7])
def test_a_kind_that_is_not_a_known_string_is_refused(kind: object) -> None:
    with pytest.raises(RequestRefused) as caught:
        validate_request({"kind": kind, "commit": SHA})
    assert caught.value.code == "kind_unknown"


def test_request_ids_name_exactly_the_known_kinds() -> None:
    from operator_door.requests import _SCOPES, validate_request_id

    for kind in _SCOPES:
        assert validate_request_id(f"20260926T000000Z-{kind}-0123abcd")
        assert required_scope(kind) in {"deploy", "operate"}
    for bad in ("20260926T000000Z-shell-0123abcd", "20260926T000000Z-deploy-0123ABCD"):
        with pytest.raises(RequestRefused):
            validate_request_id(bad)


def test_lane_scratch_requests_are_bounded_and_require_operate_scope() -> None:
    digest = "sha256:" + "a" * 64
    assert required_scope("lane-scratch") == "operate"
    assert validate_request({"kind": "lane-scratch", "action": "ls", "root": "work", "lane": "g1",
                             "limit": 20, "offset": 0}) == {
        "kind": "lane-scratch", "action": "ls", "root": "work", "lane": "g1", "limit": 20, "offset": 0}
    assert validate_request({"kind": "lane-scratch", "action": "renew", "root": "inputs", "lane": "g1",
                             "name": "run-1", "owner": "agent-1", "expected_digest": digest,
                             "ttl_seconds": 3600})["ttl_seconds"] == 3600
    assert validate_request({"kind": "lane-scratch", "action": "release", "root": "work", "lane": "g1",
                             "name": "run-1", "owner": "agent-1", "expected_digest": digest})["action"] == "release"


@pytest.mark.parametrize("change", [
    {"root": "/tmp"}, {"root": "other"}, {"lane": "../g1"}, {"lane": "."},
    {"name": "../../etc"}, {"owner": "a/b"}, {"expected_digest": "sha256:bad"},
    {"ttl_seconds": 0}, {"ttl_seconds": 14 * 86400 + 1}, {"path": "/etc/passwd"},
])
def test_lane_scratch_rejects_unsafe_renewals(change: dict) -> None:
    body = {"kind": "lane-scratch", "action": "renew", "root": "work", "lane": "g1",
            "name": "run-1", "owner": "agent-1", "expected_digest": "sha256:" + "a" * 64,
            "ttl_seconds": 3600, **change}
    with pytest.raises(RequestRefused):
        validate_request(body)


def test_hold_and_release_hold_have_operate_scope_and_strict_fields() -> None:
    hold = {"kind": "hold", "unit": "blueprint-scene-progression.timer", "owner": "alice@example.org",
            "reason": "pause for inspected capture", "expires_in_seconds": 3600}
    assert validate_request(hold) == hold
    assert required_scope("hold") == "operate"
    release = {"kind": "release-hold", "unit": hold["unit"]}
    assert validate_request(release) == release
    assert required_scope("release-hold") == "operate"


@pytest.mark.parametrize(("change", "code"), [
    ({"owner": ""}, "hold_owner_invalid"),
    ({"owner": "bad name"}, "hold_owner_invalid"),
    ({"reason": ""}, "hold_reason_invalid"),
    ({"reason": "x" * 201}, "hold_reason_invalid"),
    ({"reason": "line\nbreak"}, "hold_reason_invalid"),
    ({"expires_in_seconds": True}, "hold_expiry_invalid"),
    ({"expires_in_seconds": 59}, "hold_expiry_invalid"),
    ({"expires_in_seconds": 86401}, "hold_expiry_invalid"),
    ({"unit": "blueprint-scene-progression.service"}, "hold_unit_invalid"),
    ({"unit": "blueprint-control-plane-storage-gc.timer"}, "unit_safety_critical"),
    ({"extra": "ignored"}, "request_key_unknown:extra"),
])
def test_hold_refuses_invalid_fields(change: dict, code: str) -> None:
    body = {"kind": "hold", "unit": "blueprint-scene-progression.timer", "owner": "alice",
            "reason": "pause", "expires_in_seconds": 60, **change}
    with pytest.raises(RequestRefused) as caught:
        validate_request(body)
    assert caught.value.code == code


@pytest.mark.parametrize(("unit", "code"), [
    ("blueprint-scene-progression.service", "hold_unit_invalid"),
    ("blueprint-gpu-spend-guard.timer", "unit_safety_critical"),
    ("blueprint-operator-door-runner.path", "unit_is_door"),
])
def test_release_hold_refuses_unsafe_units(unit: str, code: str) -> None:
    with pytest.raises(RequestRefused) as caught:
        validate_request({"kind": "release-hold", "unit": unit})
    assert caught.value.code == code


def test_restore_requires_an_exact_scene_and_bucket() -> None:
    assert validate_request({"kind": "restore-scene-workspace", "scene_id": "scene-1",
                             "bucket": "blueprint-8c1ca.appspot.com"}) == {
        "kind": "restore-scene-workspace", "scene_id": "scene-1", "bucket": "blueprint-8c1ca.appspot.com"}
    assert required_scope("restore-scene-workspace") == "operate"
    for body in ({"kind": "restore-scene-workspace", "scene_id": "scene-1"},
                 {"kind": "restore-scene-workspace", "scene_id": "../bad", "bucket": "valid.example"},
                 {"kind": "restore-scene-workspace", "scene_id": "scene-1", "bucket": "a/unsafe"}):
        with pytest.raises(RequestRefused):
            validate_request(body)


def test_provider_output_resume_names_one_canary_attempt() -> None:
    """One canary dispatch directory and one attempt number select exactly one attempt tree."""
    assert validate_request({"kind": "provider-output-resume", "run": "activation-1", "attempt": 1}) == {
        "kind": "provider-output-resume", "run": "activation-1", "attempt": 1, "ingest": False}
    assert validate_request({"kind": "provider-output-resume", "run": "scene-839873.r4_a", "attempt": 12,
                             "ingest": True})["ingest"] is True
    assert required_scope("provider-output-resume") == "operate"
    for body, code in (
            ({"kind": "provider-output-resume", "run": "../activation-1", "attempt": 1}, "provider_output_resume_run_invalid"),
            ({"kind": "provider-output-resume", "run": "..", "attempt": 1}, "provider_output_resume_run_invalid"),
            ({"kind": "provider-output-resume", "run": "a/b", "attempt": 1}, "provider_output_resume_run_invalid"),
            ({"kind": "provider-output-resume", "run": "activation-1", "attempt": 0}, "provider_output_resume_attempt_invalid"),
            ({"kind": "provider-output-resume", "run": "activation-1", "attempt": 1000},
             "provider_output_resume_attempt_invalid"),
            ({"kind": "provider-output-resume", "run": "activation-1", "attempt": True},
             "provider_output_resume_attempt_invalid"),
            ({"kind": "provider-output-resume", "run": "activation-1", "attempt": "1"},
             "provider_output_resume_attempt_invalid"),
            ({"kind": "provider-output-resume", "run": "activation-1", "attempt": 1, "ingest": "yes"},
             "provider_output_resume_ingest_invalid"),
            ({"kind": "provider-output-resume", "run": "activation-1", "attempt": 1, "path": "/etc"},
             "request_key_unknown:path")):
        with pytest.raises(RequestRefused) as caught:
            validate_request(body)
        assert caught.value.code == code


def test_door_upgrade_needs_a_commit() -> None:
    assert validate_request({"kind": "door-upgrade", "commit": SHA}) == {"kind": "door-upgrade", "commit": SHA}
    assert required_scope("door-upgrade") == "deploy"


def test_enqueue_writes_an_atomic_world_readable_spool_file(config: DoorConfig) -> None:
    request_id = enqueue(config, validate_request({"kind": "deploy", "commit": SHA}), requested_by="cloud")
    assert re.fullmatch(r"\d{8}T\d{6}Z-deploy-[0-9a-f]{8}", request_id)
    path = Path(config.spool_root) / "pending" / f"{request_id}.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["id"] == request_id and document["requested_by"] == "cloud"
    assert document["request"] == {"kind": "deploy", "commit": SHA, "wait_for_idle": True}
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


SCENE_OK = ["site-capture-1", "a", "A.b_c-d", "x" * 128]
SCENE_BAD = ["", ".", "..", "-leading", ".hidden", "a/b", "a b", "x" * 129, "scene\n", 7, None]
BUCKET_OK = ["blueprint-8c1ca.appspot.com", "abc", "a" * 222]
BUCKET_BAD = ["", "ab", "Upper-bucket", "-bucket", "bucket-", "a..b", "a" * 223, "bucket/x", 3]


@pytest.mark.parametrize("scene_id", SCENE_OK)
def test_retire_scene_workspace_accepts_scene_ids_the_listener_can_create(scene_id: str) -> None:
    assert validate_request({"kind": "retire-scene-workspace", "scene_id": scene_id}) == {
        "kind": "retire-scene-workspace", "scene_id": scene_id, "apply": False}


@pytest.mark.parametrize("scene_id", SCENE_BAD)
def test_retire_scene_workspace_refuses_unsafe_scene_ids(scene_id: object) -> None:
    with pytest.raises(RequestRefused) as caught:
        validate_request({"kind": "retire-scene-workspace", "scene_id": scene_id})
    assert caught.value.code == "scene_id_invalid"


@pytest.mark.parametrize("bucket", BUCKET_OK)
def test_retire_scene_workspace_takes_an_optional_bucket(bucket: str) -> None:
    assert validate_request({"kind": "retire-scene-workspace", "scene_id": "s", "bucket": bucket, "apply": True}) == {
        "kind": "retire-scene-workspace", "scene_id": "s", "bucket": bucket, "apply": True}


@pytest.mark.parametrize("bucket", BUCKET_BAD)
def test_retire_scene_workspace_refuses_invalid_buckets(bucket: object) -> None:
    with pytest.raises(RequestRefused) as caught:
        validate_request({"kind": "retire-scene-workspace", "scene_id": "s", "bucket": bucket})
    assert caught.value.code == "bucket_invalid"


@pytest.mark.parametrize(("body", "code"), [
    ({"kind": "retire-scene-workspace", "scene_id": "s", "apply": "yes"}, "apply_invalid"),
    ({"kind": "retire-scene-workspace", "scene_id": "s", "apply": 1}, "apply_invalid"),
    ({"kind": "retire-scene-workspace", "scene_id": "s", "commit": SHA}, "request_key_unknown:commit"),
    ({"kind": "retire-scene-workspace", "scene_id": "s", "ack": "retire-scene-workspace"}, "request_key_unknown:ack"),
])
def test_retire_scene_workspace_refuses_everything_else(body: dict, code: str) -> None:
    with pytest.raises(RequestRefused) as caught:
        validate_request(body)
    assert caught.value.code == code


def test_retire_scene_workspace_is_an_operate_request_with_its_own_id(config: DoorConfig) -> None:
    assert required_scope("retire-scene-workspace") == "operate"
    request_id = enqueue(config, validate_request({"kind": "retire-scene-workspace", "scene_id": "s"}),
                         requested_by="cloud")
    assert re.fullmatch(r"\d{8}T\d{6}Z-retire-scene-workspace-[0-9a-f]{8}", request_id)
    assert request_state(config, request_id)["request"]["scene_id"] == "s"
    assert list_requests(config)[0]["kind"] == "retire-scene-workspace"


def test_operation_key_replays_and_conflicts_without_duplicate_spool(config):
    request = {"kind": "deploy", "commit": SHA}
    key = "test-lost-response-123"
    first = enqueue(config, request, requested_by="owner", operation_key=key)
    assert enqueue(config, request, requested_by="owner", operation_key=key) == first
    assert len(list((Path(config.spool_root) / "pending").glob("*.json"))) == 1
    with pytest.raises(RequestRefused, match="operation_key_conflict"):
        enqueue(config, {**request, "commit": "f" * 40}, requested_by="owner", operation_key=key)
    assert enqueue(config, request, requested_by="other", operation_key=key) != first
    (Path(config.spool_root) / "pending" / f"{first}.json").unlink()
    assert enqueue(config, request, requested_by="owner", operation_key=key) == first
    assert not (Path(config.spool_root) / "pending" / f"{first}.json").exists()


def test_concurrent_operation_key_has_one_identity(config):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: enqueue(config, {"kind": "deploy", "commit": SHA},
            requested_by="owner", operation_key="concurrent-key-1234"), range(16)))
    assert len(set(ids)) == 1


def test_unit_completion_requires_the_same_observed_invocation():
    from operator_door.requests import observed_unit_outcome
    state = {"result": {"unit_observation": {"InvocationID": "bound"}},
             "unit_state": [{"InvocationID": "bound", "ActiveState": "inactive", "Result": "success", "ExecMainStatus": "0"}]}
    assert observed_unit_outcome(state)["status"] == "observed_completed"
    state["unit_state"][0]["ActiveState"] = "failed"
    assert observed_unit_outcome(state)["status"] == "observed_failed"
    state["unit_state"][0]["InvocationID"] = "newer"
    assert observed_unit_outcome(state)["status"] == "unknown"
    assert observed_unit_outcome({"result": {"status": "done"}}) is None
