"""Existing authenticated door/spool gets one finite report-only consent request."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/requests.py
#   deploy/operator-door/operator_door/server.py
#   deploy/operator-door/operator_door/spool_runner.py
from pathlib import Path

import pytest

from tests.test_operator_door_server import door as _door_fixture, _call, _json, READER, DEPLOYER
from tests.test_operator_door_runner import FakeRunner
from operator_door import requests, spool_runner
from operator_door.config import DoorConfig

door = _door_fixture


def request():
    return dict(
        kind="owner-census-decision",
        consent_id="a" * 32,
        expected_sha256="sha256:" + "b" * 64,
        expected_size_bytes=100,
    )


def test_finite_kind_operate_scope_and_id_normalization():
    body = request()
    assert requests.validate_request(body) == body
    assert requests.required_scope(body["kind"]) == "operate"
    assert requests.validate_request_id(requests.new_request_id(body["kind"]))


@pytest.mark.parametrize(
    "change",
    [
        {"PRIVATE" * 400: 1},
        {"apply": True},
        {"owner": "forged"},
        {"consent_id": "../private"},
        {"expected_size_bytes": True},
        {"expected_size_bytes": 524289},
        {"expected_sha256": "invalid"},
    ],
)
def test_request_has_fixed_bounded_no_echo_errors(change):
    with pytest.raises(requests.RequestRefused) as exc:
        requests.validate_request(request() | change)
    assert exc.value.code == "owner_consent_options_invalid"


def test_disabled_http_refuses_before_enqueue(door):
    before = list((door["state"] / "requests/pending").iterdir())
    value = _call(door, "/requests", token=DEPLOYER, body=request())
    assert value[0] == 403 and _json(value) == {"error": "owner_consent_disabled"}
    assert list((door["state"] / "requests/pending").iterdir()) == before


def test_reader_scope_cannot_request_report(door):
    value = _call(door, "/requests", token=READER, body=request())
    assert value[0] == 403 and _json(value) == {"error": "scope_missing:operate"}


def test_enabled_runner_launch_is_fixed_readonly_except_results(tmp_path):
    config = DoorConfig(state_root=str(tmp_path), owner_census_decisions_enabled=1)
    fake = FakeRunner()
    result = spool_runner._act(
        config, fake, "20260928T000000Z-owner-census-decision-deadbeef", request()
    )
    assert result["status"] == "launched"
    argv = fake.calls[0]
    assert argv[-1] == config.install_root + "/door-owner-census.sh"
    assert "--property=RuntimeMaxSec=10s" in argv
    assert "--property=ProtectSystem=strict" in argv
    writable = [v for v in argv if v.startswith("--property=ReadWritePaths=")]
    assert writable == ["--property=ReadWritePaths=" + str(Path(config.spool_root) / "results")]
    env = [v for v in argv if v.startswith("--setenv=DOOR_")]
    assert {v.split("=", 2)[1] for v in env} == {
        "DOOR_REQUEST_ID",
        "DOOR_RESULTS_DIR",
        "DOOR_CONSENT_ID",
        "DOOR_CONSENT_SHA256",
        "DOOR_CONSENT_SIZE_BYTES",
        "DOOR_CONFIG_PATH",
        "DOOR_VENV_PYTHON",
        "DOOR_CONTROL_PLANE_REPO",
    }
    assert not any("OWNER=" in v or "POLICY=" in v or "APPLY" in v for v in env)


def test_disabled_spooled_request_never_launches(tmp_path):
    fake = FakeRunner()
    result = spool_runner._act(
        DoorConfig(state_root=str(tmp_path)),
        fake,
        "20260928T000000Z-owner-census-decision-deadbeef",
        request(),
    )
    assert result == {"status": "refused", "code": "owner_consent_disabled"} and not fake.calls


def test_enabled_http_normalizes_only_fixed_consent_identity(door):
    import json

    object.__setattr__(door["config"], "owner_census_decisions_enabled", 1)
    response = _call(door, "/requests", token=DEPLOYER, body=request())
    assert response[0] == 202
    request_id = _json(response)["id"]
    pending = door["state"] / "requests/pending" / (request_id + ".json")
    retained = json.loads(pending.read_bytes())
    assert retained["request"] == request()
    assert retained["requested_by"] == "deployer"
    assert not any(key in retained["request"] for key in ("owner", "apply", "policy", "path"))
