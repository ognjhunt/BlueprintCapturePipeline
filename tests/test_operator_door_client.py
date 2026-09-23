"""scripts/operator_door.py against a real door on a loopback port."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import threading
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "deploy" / "operator-door"))

from operator_door import API_PREFIX  # noqa: E402
from operator_door.auth import add_token, hash_token  # noqa: E402
from operator_door.config import DoorConfig  # noqa: E402
from operator_door.hostinfo import CommandResult, HostInfo  # noqa: E402
from operator_door.server import make_server  # noqa: E402

spec = importlib.util.spec_from_file_location("operator_door_client", REPO_ROOT / "scripts" / "operator_door.py")
client = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(client)

TOKEN = "t" * 40
READ_ONLY = "o" * 40
SHA = "0123456789abcdef0123456789abcdef01234567"


class Runner:
    def run(self, argv: Any, timeout: float) -> CommandResult:
        if argv[:2] == ["journalctl", "-u"]:
            return CommandResult(0, "line one\nline two\n", "")
        return CommandResult(0, "", "")


@pytest.fixture()
def door(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Any]]:
    base = tmp_path.resolve()
    data = base / "data"
    (data / "run" / "nested").mkdir(parents=True)
    (data / "run" / "progression.json").write_text('{"stage": "sam"}', encoding="utf-8")
    (data / "run" / "nested" / "frame.bin").write_bytes(bytes(range(256)) * 40)
    (data / "run" / "release.env").write_text("K=1\n", encoding="utf-8")
    state = base / "door"
    for sub in ("pending", "processing", "completed", "results"):
        (state / "requests" / sub).mkdir(parents=True)
    tokens = base / "tokens.json"
    add_token(tokens, name="cloud", sha256=hash_token(TOKEN), scopes=["read", "operate", "deploy"])
    add_token(tokens, name="viewer", sha256=hash_token(READ_ONLY), scopes=["read"])
    config = DoorConfig(read_roots=(str(data),), hidden_paths=(str(base / "hidden"),), state_root=str(state),
                        token_file=str(tokens), listen_port=0, max_read_bytes=4096,
                        control_plane_state=str(base / "cp"), active_release_link=str(base / "none"))
    host = HostInfo(config, runner=Runner(), proc_locks_path=str(base / "locks"),
                    fetch_json=lambda url: {"source_commit": SHA, "commit_proven": True, "blockers": []})
    server = make_server(config, host=host)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_URL", f"http://127.0.0.1:{server.server_address[1]}{API_PREFIX}")
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", TOKEN)
    monkeypatch.delenv("BLUEPRINT_OPERATOR_DOOR_TOKEN_FILE", raising=False)
    try:
        yield {"data": data, "state": state, "base": base}
    finally:
        server.shutdown()
        server.server_close()


def _run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with redirect_stdout(out):
        code = client.main(list(argv))
    return code, out.getvalue()


def test_whoami_prints_identity_json(door: dict[str, Any]) -> None:
    code, out = _run("whoami")
    assert code == 0 and json.loads(out) == {"name": "cloud", "scopes": ["deploy", "operate", "read"]}


def test_unauthorized_exits_3(door: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", "x" * 40)
    assert _run("whoami")[0] == 3


def test_network_errors_exit_4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_URL", f"http://127.0.0.1:9{API_PREFIX}")
    assert _run("whoami")[0] == 4


def test_no_token_sends_no_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", raising=False)
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN_FILE", "/nonexistent/token")
    assert client.auth_headers() == {}


def test_token_file_is_read_without_echoing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token = tmp_path / "token"
    token.write_text(TOKEN + "\n", encoding="utf-8")
    monkeypatch.delenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", raising=False)
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN_FILE", str(token))
    assert client.auth_headers() == {"Authorization": f"Bearer {TOKEN}"}


def test_status_and_ls_and_cat(door: dict[str, Any]) -> None:
    code, out = _run("status")
    assert code == 0 and json.loads(out)["deployed"]["source_commit"] == SHA
    code, out = _run("ls", str(door["data"] / "run"))
    names = {entry["name"]: entry for entry in json.loads(out)["entries"]}
    assert code == 0 and names["release.env"]["refused"] == "secret_name"
    code, out = _run("cat", str(door["data"] / "run" / "progression.json"))
    assert code == 0 and json.loads(out) == {"stage": "sam"}


def test_refusals_print_the_rule_and_exit_2(door: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = _run("cat", str(door["data"] / "run" / "release.env"))
    assert code == 2 and "secret_name_refused" in capsys.readouterr().err


def test_pull_pages_a_large_file(door: dict[str, Any], tmp_path: Path) -> None:
    target = tmp_path / "frame.bin"
    code, _ = _run("pull", str(door["data"] / "run" / "nested" / "frame.bin"), str(target))
    assert code == 0 and target.read_bytes() == bytes(range(256)) * 40


def test_pull_extracts_a_directory_and_reports_skips(door: dict[str, Any], tmp_path: Path) -> None:
    target = tmp_path / "copy"
    code, out = _run("pull", str(door["data"] / "run"), str(target))
    manifest = json.loads(out)
    assert code == 0 and (target / "nested" / "frame.bin").exists() and (target / "progression.json").exists()
    assert not (target / "release.env").exists()
    assert {"path": "release.env", "reason": "secret_name"} in manifest["skipped"]


def test_journal_prints_text(door: dict[str, Any]) -> None:
    code, out = _run("journal", "blueprint-pipeline-intake.service", "-n", "5")
    assert code == 0 and out == "line one\nline two\n"


def test_deploy_spools_a_request_and_request_shows_it(door: dict[str, Any]) -> None:
    code, out = _run("deploy", SHA, "--mode", "canary", "--no-wait-for-idle")
    request_id = json.loads(out)["id"]
    assert code == 0
    spooled = json.loads((door["state"] / "requests" / "pending" / f"{request_id}.json").read_text())
    assert spooled["request"] == {"kind": "deploy", "commit": SHA, "mode": "canary", "wait_for_idle": False}
    code, out = _run("request", request_id)
    assert code == 0 and json.loads(out)["state"] == "pending"


def test_unit_and_replay_and_upgrade_requests(door: dict[str, Any]) -> None:
    assert _run("unit", "start", "blueprint-gpu-spend-guard.service")[0] == 0
    assert _run("replay", "--child", "sam31-" + "ab" * 16, "--commit", SHA)[0] == 0
    assert _run("upgrade-door", SHA)[0] == 0
    kinds = sorted(item["kind"] for item in json.loads(_run("requests")[1])["requests"])
    assert kinds == ["door-upgrade", "stage-replay", "unit"]


def test_wait_returns_when_the_outcome_lands(door: dict[str, Any]) -> None:
    code, out = _run("deploy", SHA)
    request_id = json.loads(out)["id"]
    results = door["state"] / "requests" / "results"
    (results / f"{request_id}.json").write_text(json.dumps({"status": "launched", "unit": "u"}))
    (results / f"{request_id}.outcome.json").write_text(json.dumps({"status": "deployed", "exit_code": 0}))
    code, out = _run("request", request_id, "--wait", "--poll", "0.05", "--timeout", "5")
    assert code == 0 and json.loads(out)["outcome"]["status"] == "deployed"


def test_wait_fails_on_a_refusal(door: dict[str, Any]) -> None:
    code, out = _run("deploy", SHA)
    request_id = json.loads(out)["id"]
    results = door["state"] / "requests" / "results"
    (results / f"{request_id}.json").write_text(json.dumps({"status": "refused", "code": "deploy_in_progress:x"}))
    code, _ = _run("request", request_id, "--wait", "--poll", "0.05", "--timeout", "5")
    assert code == 1


def test_scope_refusal_on_post_exits_3(door: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", READ_ONLY)
    assert _run("deploy", SHA)[0] == 3
