"""The door's HTTP surface, exercised through a real server on a loopback port."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/server.py
#   deploy/operator-door/operator_door/__main__.py

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator, Sequence

import pytest

DOOR_ROOT = Path(__file__).resolve().parents[1] / "deploy" / "operator-door"
sys.path.insert(0, str(DOOR_ROOT))

from operator_door import API_PREFIX  # noqa: E402
from operator_door.auth import add_token, hash_token  # noqa: E402
from operator_door.config import DoorConfig  # noqa: E402
from operator_door.hostinfo import CommandResult, HostInfo  # noqa: E402
from operator_door.server import make_server  # noqa: E402

READER = "r" * 40
DEPLOYER = "d" * 40
SHA = "0123456789abcdef0123456789abcdef01234567"


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, argv: Sequence[str], timeout: float) -> CommandResult:
        self.calls.append(list(argv))
        if argv[:2] == ["journalctl", "-u"]:
            return CommandResult(0, "started\nOPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz\n", "")
        if argv[:2] == ["systemctl", "list-units"]:
            return CommandResult(0, "blueprint-a.timer loaded active waiting A\n", "")
        if argv[:2] == ["systemctl", "show"]:
            return CommandResult(0, "Id=blueprint-a.timer\nActiveState=active\n", "")
        return CommandResult(0, "", "")


@pytest.fixture()
def door(tmp_path: Path) -> Iterator[dict[str, Any]]:
    base = tmp_path.resolve()
    data = base / "data"
    (data / "run").mkdir(parents=True)
    (data / "run" / "progression.json").write_text('{"stage": "marble"}', encoding="utf-8")
    (data / "run" / "release.env").write_text("X=1\n", encoding="utf-8")
    state = base / "door"
    for sub in ("pending", "processing", "completed", "results"):
        (state / "requests" / sub).mkdir(parents=True)
    tokens = base / "tokens.json"
    add_token(tokens, name="reader", sha256=hash_token(READER), scopes=["read"])
    add_token(tokens, name="deployer", sha256=hash_token(DEPLOYER), scopes=["read", "operate", "deploy"])
    config = DoorConfig(
        read_roots=(str(data),), hidden_paths=(str(base / "hidden"),), state_root=str(state),
        token_file=str(tokens), listen_port=0, control_plane_state=str(base / "cp"),
        active_release_link=str(base / "missing-link"),
    )
    runner = FakeRunner()
    host = HostInfo(config, runner=runner, proc_locks_path=str(base / "no-locks"),
                    fetch_json=lambda url: {"source_commit": SHA, "commit_proven": True, "blockers": []})
    server = make_server(config, host=host)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield {"url": f"http://127.0.0.1:{server.server_address[1]}{API_PREFIX}", "config": config,
               "data": data, "state": state, "runner": runner}
    finally:
        server.shutdown()
        server.server_close()


def _call(door: dict[str, Any], path: str, *, token: str | None = READER, body: Any = None,
          method: str | None = None) -> tuple[int, dict[str, str], bytes]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(door["url"] + path, data=data, method=method)
    if token is not None:
        request.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=10) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def _json(result: tuple[int, dict[str, str], bytes]) -> Any:
    return json.loads(result[2])


def test_healthz_needs_no_token_and_reveals_nothing(door: dict[str, Any]) -> None:
    status, _, body = _call(door, "/healthz", token=None)
    assert status == 200 and set(json.loads(body)) == {"ok", "version"}


@pytest.mark.parametrize("token", [None, "x" * 40])
def test_missing_or_unknown_tokens_are_unauthorized_and_audited(door: dict[str, Any], token: str | None) -> None:
    status, headers, body = _call(door, "/status", token=token)
    assert status == 401 and json.loads(body) == {"error": "unauthorized"}
    audit = Path(door["config"].audit_path).read_text(encoding="utf-8").splitlines()
    entry = json.loads(audit[-1])
    assert entry["outcome"] == "denied" and entry["token"] is None and entry["status"] == 401


def test_every_response_is_uncacheable_and_nosniff(door: dict[str, Any]) -> None:
    _, headers, _ = _call(door, "/whoami")
    assert headers["Cache-Control"] == "no-store" and headers["X-Content-Type-Options"] == "nosniff"


def test_whoami_reports_the_identity(door: dict[str, Any]) -> None:
    assert _json(_call(door, "/whoami")) == {"name": "reader", "scopes": ["read"]}


def test_status_is_served_to_readers(door: dict[str, Any]) -> None:
    status = _json(_call(door, "/status"))
    assert status["deployed"]["source_commit"] == SHA
    assert status["door"]["caller"] == {"name": "reader", "scopes": ["read"]}


def test_file_listing_and_reading(door: dict[str, Any]) -> None:
    run = door["data"] / "run"
    listing = _json(_call(door, f"/fs/list?path={run}"))
    assert {entry["name"] for entry in listing["entries"]} == {"progression.json", "release.env"}
    status, headers, body = _call(door, f"/fs/read?path={run / 'progression.json'}")
    assert status == 200 and json.loads(body) == {"stage": "marble"}
    assert headers["X-Door-Eof"] == "true" and headers["X-Door-Size"] == str(len(body))


def test_refusals_are_403_with_the_rule(door: dict[str, Any]) -> None:
    status, _, body = _call(door, f"/fs/read?path={door['data'] / 'run' / 'release.env'}")
    assert status == 403 and json.loads(body) == {"error": "secret_name_refused"}
    status, _, body = _call(door, "/fs/read?path=/etc/passwd")
    assert status == 403 and json.loads(body) == {"error": "path_outside_roots"}


def test_missing_files_are_404(door: dict[str, Any]) -> None:
    status, _, body = _call(door, f"/fs/read?path={door['data'] / 'nope.json'}")
    assert status == 404 and json.loads(body) == {"error": "not_found"}


def test_archive_streams_a_tarball(door: dict[str, Any]) -> None:
    status, headers, body = _call(door, f"/fs/archive?path={door['data'] / 'run'}")
    assert status == 200 and headers["Content-Type"] == "application/gzip"
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
        assert "progression.json" in archive.getnames() and "release.env" not in archive.getnames()


def test_journal_is_redacted_text(door: dict[str, Any]) -> None:
    status, headers, body = _call(door, "/journal?unit=blueprint-pipeline-intake.service&lines=20")
    assert status == 200 and headers["Content-Type"].startswith("text/plain")
    assert b"sk-abcdef" not in body and b"started" in body


def test_journal_refuses_foreign_units(door: dict[str, Any]) -> None:
    status, _, body = _call(door, "/journal?unit=sshd.service")
    assert status == 403 and json.loads(body) == {"error": "unit_name_invalid"}


def test_units_listing(door: dict[str, Any]) -> None:
    units = _json(_call(door, "/units?pattern=blueprint-*"))
    assert units["units"][0]["unit"] == "blueprint-a.timer"


def test_post_requires_the_kind_scope(door: dict[str, Any]) -> None:
    status, _, body = _call(door, "/requests", body={"kind": "deploy", "commit": SHA})
    assert status == 403 and json.loads(body) == {"error": "scope_missing:deploy"}


def test_deploy_request_is_spooled_and_readable(door: dict[str, Any]) -> None:
    status, _, body = _call(door, "/requests", token=DEPLOYER, body={"kind": "deploy", "commit": SHA})
    assert status == 202
    request_id = json.loads(body)["id"]
    assert (door["state"] / "requests" / "pending" / f"{request_id}.json").exists()
    state = _json(_call(door, f"/requests/{request_id}"))
    assert state["state"] == "pending" and state["request"]["commit"] == SHA
    assert _json(_call(door, "/requests"))["requests"][0]["id"] == request_id


def test_invalid_request_bodies(door: dict[str, Any]) -> None:
    status, _, body = _call(door, "/requests", token=DEPLOYER, body={"kind": "shell"})
    assert status == 400 and json.loads(body) == {"error": "kind_unknown"}
    request = urllib.request.Request(door["url"] + "/requests", data=b"{not json", method="POST")
    request.add_header("Authorization", f"Bearer {DEPLOYER}")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=10)
    assert caught.value.code == 400


def test_oversized_bodies_are_rejected(door: dict[str, Any]) -> None:
    status, _, _ = _call(door, "/requests", token=DEPLOYER, body={"kind": "deploy", "pad": "x" * 70_000})
    assert status == 413


def test_unknown_routes_are_404(door: dict[str, Any]) -> None:
    assert _call(door, "/admin")[0] == 404
    assert _call(door, "/requests/../../etc")[0] in (403, 404)


def test_self_test_command(tmp_path: Path) -> None:
    config = tmp_path / "door.json"
    config.write_text(json.dumps({"read_roots": [str(tmp_path)], "state_root": str(tmp_path / "s"),
                                  "token_file": str(tmp_path / "tokens.json")}), encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-m", "operator_door", "self-test", "--config", str(config)],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(DOOR_ROOT)}, check=False,
    )
    report = json.loads(done.stdout)
    assert report["config"] == "ok" and report["tokens"] == 0
    assert done.returncode == 1  # no tokens configured yet is a failure worth surfacing



def _small_server(tmp_path: Path, **overrides: Any) -> tuple[Any, DoorConfig]:
    tokens = tmp_path / "tokens-small.json"
    add_token(tokens, name="reader", sha256=hash_token(READER), scopes=["read"])
    config = DoorConfig(read_roots=(str(tmp_path),), hidden_paths=(str(tmp_path / "hidden"),),
                        state_root=str(tmp_path / "small-door"), token_file=str(tokens), listen_port=0,
                        **overrides)
    host = HostInfo(config, runner=FakeRunner(), proc_locks_path=str(tmp_path / "none"),
                    fetch_json=lambda url: {})
    server = make_server(config, host=host)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    return server, config


def test_idle_connections_time_out(tmp_path: Path) -> None:
    import socket

    server, _ = _small_server(tmp_path.resolve(), request_timeout_seconds=1)
    try:
        with socket.create_connection(("127.0.0.1", server.server_address[1]), timeout=5) as idle:
            idle.settimeout(10)
            assert idle.recv(1) == b""  # closed by the door, not held open
    finally:
        server.shutdown()
        server.server_close()


def test_audit_log_rotates_when_large(tmp_path: Path) -> None:
    server, config = _small_server(tmp_path.resolve(), audit_rotate_bytes=1000)
    try:
        audit = Path(config.audit_path)
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text("x" * 1001, encoding="utf-8")
        request = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}{API_PREFIX}/whoami",
                                         headers={"Authorization": f"Bearer {READER}"})
        urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=10).read()
        assert audit.with_name("audit.jsonl.1").stat().st_size == 1001
        assert json.loads(audit.read_text(encoding="utf-8").splitlines()[-1])["route"] == "/whoami"
    finally:
        server.shutdown()
        server.server_close()
