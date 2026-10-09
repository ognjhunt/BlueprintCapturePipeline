"""scripts/operator_door.py against a real door on a loopback port."""

# Covers (for impacted-test selection):
#   scripts/operator_door.py

from __future__ import annotations

import importlib.util
import hashlib
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
                        control_plane_state=str(base / "cp"), active_release_link=str(base / "none"),
                        capacity_summary=str(base / "cp" / "capacity" / "summary.json"))
    host = HostInfo(config, runner=Runner(), proc_locks_path=str(base / "locks"),
                    fetch_json=lambda url: {"source_commit": SHA, "commit_proven": True, "blockers": []})
    server = make_server(config, host=host)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_URL", f"http://127.0.0.1:{server.server_address[1]}{API_PREFIX}")
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", TOKEN)
    monkeypatch.delenv("BLUEPRINT_OPERATOR_DOOR_TOKEN_FILE", raising=False)
    try:
        yield {"data": data, "state": state, "base": base, "config": config}
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


def test_hold_client_explicit_release_remains_an_owned_bounded_request(door: dict[str, Any]) -> None:
    code, out = _run("hold", "blueprint-agent-run-dispatcher.timer", "--owner", "founder-stop",
                     "--reason", "Founder stop", "--for", "24h", "--until-released")
    assert code == 0
    request_id = json.loads(out)["id"]
    request = json.loads((door["state"] / "requests" / "pending" / f"{request_id}.json").read_text())["request"]
    assert request["require_explicit_release"] is True
    assert request["expires_in_seconds"] == 86400


def test_notifier_repair_client_requires_and_preserves_expected_identity(door: dict[str, Any]) -> None:
    digest = "sha256:" + "a" * 64
    code, out = _run("unit", "repair-notifier-binding", "blueprint-pipeline-control-plane.service",
                     "--expected-postcheck-sha256", digest, "--expected-source-commit", SHA)
    assert code == 0
    request_id = json.loads(out)["id"]
    request = json.loads((door["state"] / "requests/pending" / f"{request_id}.json").read_text())["request"]
    assert request == {"kind": "unit", "unit": "blueprint-pipeline-control-plane.service",
                       "action": "repair-notifier-binding", "expected_postcheck_sha256": digest,
                       "expected_source_commit": SHA}
    assert _run("unit", "repair-notifier-binding", "blueprint-pipeline-control-plane.service")[0] == 2


def test_lane_scratch_client_submits_bounded_commands(door: dict[str, Any]) -> None:
    digest = "sha256:" + "a" * 64
    def submitted(out: str) -> dict[str, Any]:
        request_id = json.loads(out)["id"]
        path = door["state"] / "requests" / "pending" / f"{request_id}.json"
        return json.loads(path.read_text(encoding="utf-8"))["request"]

    code, out = _run("lane-scratch", "ls", "g1", "--root", "work", "--limit", "10", "--offset", "20")
    assert code == 0
    assert submitted(out) == {"kind": "lane-scratch", "action": "ls", "lane": "g1",
                              "root": "work", "limit": 10, "offset": 20}
    code, out = _run("lane-scratch", "renew", "g1", "run-1", "--root", "inputs", "--owner", "agent-1",
                     "--digest", digest, "--for", "2d")
    assert code == 0
    assert submitted(out)["ttl_seconds"] == 172800
    code, out = _run("lane-scratch", "release", "g1", "run-1", "--root", "work", "--owner", "agent-1",
                     "--digest", digest)
    assert code == 0 and submitted(out)["action"] == "release"


def test_legacy_owner_census_client_submits_only_fixed_report_kind(door: dict[str, Any]) -> None:
    object.__setattr__(door["config"], "owner_census_decisions_enabled", 1)
    code, out = _run("legacy-owner-census")
    assert code == 0
    request_id = json.loads(out)["id"]
    path = door["state"] / "requests" / "pending" / f"{request_id}.json"
    assert json.loads(path.read_text(encoding="utf-8"))["request"] == {
        "kind": "legacy-owner-census"}


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
    code, out = _run("deploy", SHA, "--no-wait-for-idle")
    request_id = json.loads(out)["id"]
    assert code == 0
    spooled = json.loads((door["state"] / "requests" / "pending" / f"{request_id}.json").read_text())
    assert spooled["request"] == {"kind": "deploy", "commit": SHA, "wait_for_idle": False}
    code, out = _run("request", request_id)
    assert code == 0 and json.loads(out)["state"] == "pending"


def test_canonical_deploy_client_preserves_verified_provenance_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = ('{\n "schema_version": "blueprint.deploy_release_provenance.v1",'
               ' "git_sha": "' + SHA + '", "status": "verified"\n}\n').encode()
    provenance = tmp_path / "official-provenance.json"
    provenance.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    submitted = []
    monkeypatch.setattr(client, "_submit", lambda body, args: submitted.append(body) or 0)
    assert _run("deploy", SHA, "--release-provenance", str(provenance),
                "--release-provenance-sha256", digest)[0] == 0
    assert submitted == [{"kind": "deploy", "commit": SHA, "wait_for_idle": True,
                          "release_provenance_json": payload.decode(), "release_provenance_sha256": digest}]


@pytest.mark.parametrize("case", ["missing_hash", "missing_file", "bad_hash", "hash_mismatch",
                                  "oversized", "invalid_json", "array", "invalid_utf8", "symlink"])
def test_canonical_deploy_client_refuses_invalid_transport_before_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    payload = b'{"status":"verified"}'
    if case == "oversized":
        payload = b" " * (16 * 1024 + 1)
    if case == "invalid_json":
        payload = b"{"
    if case == "array":
        payload = b"[]"
    if case == "invalid_utf8":
        payload = b"\xff"
    provenance = tmp_path / "receipt.json"
    provenance.write_bytes(payload)
    if case == "symlink":
        link = tmp_path / "receipt-link.json"
        link.symlink_to(provenance)
        provenance = link
    digest = hashlib.sha256(payload).hexdigest()
    if case == "bad_hash":
        digest = "invalid"
    if case == "hash_mismatch":
        digest = "0" * 64
    args = ["deploy", SHA]
    if case != "missing_file":
        args += ["--release-provenance", str(provenance)]
    if case != "missing_hash":
        args += ["--release-provenance-sha256", digest]
    submitted = []
    monkeypatch.setattr(client, "_submit", lambda body, args: submitted.append(body) or 0)
    assert _run(*args)[0] == 2
    assert submitted == []


def test_canonical_deploy_still_requires_deploy_scope(
    door: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A valid request must reach the scope check rather than fail schema admission.
    payload = json.dumps({"schema_version": "blueprint.deploy_release_provenance.v1",
                          "status": "verified", "git_sha": SHA, "workflow_name": "Full Test Lane",
                          "workflow_path": ".github/workflows/full-test-lane.yml",
                          "job_name": "Full pytest lane on CPU runner", "run_id": 123,
                          "collection": {"test_count": 42},
                          "claim_boundary": {"canonical_full_lane_verified": True}}).encode()
    provenance = tmp_path / "receipt.json"
    provenance.write_bytes(payload)
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", READ_ONLY)
    assert _run("deploy", SHA, "--release-provenance", str(provenance),
                "--release-provenance-sha256", hashlib.sha256(payload).hexdigest())[0] == 3
    assert not list((door["state"] / "requests" / "pending").iterdir())


def test_unit_and_upgrade_requests(door: dict[str, Any]) -> None:
    assert _run("unit", "start", "blueprint-gpu-spend-guard.service")[0] == 0
    assert _run("upgrade-door", SHA)[0] == 0
    kinds = sorted(item["kind"] for item in json.loads(_run("requests")[1])["requests"])
    assert kinds == ["door-upgrade", "unit"]


@pytest.mark.parametrize(("duration", "seconds"), [("2h", 7200), ("90m", 5400), ("3600s", 3600)])
def test_hold_client_spools_owner_reason_and_duration(door: dict[str, Any], duration: str, seconds: int) -> None:
    code, out = _run("hold", "blueprint-scene-progression.timer", "--owner", "alice",
                     "--reason", "inspect capture", "--for", duration)
    request_id = json.loads(out)["id"]
    spooled = json.loads((door["state"] / "requests" / "pending" / f"{request_id}.json").read_text())
    assert code == 0 and spooled["request"] == {"kind": "hold", "unit": "blueprint-scene-progression.timer",
                                                "owner": "alice", "reason": "inspect capture",
                                                "expires_in_seconds": seconds}
    code, out = _run("release-hold", "blueprint-scene-progression.timer")
    release = json.loads((door["state"] / "requests" / "pending" / f"{json.loads(out)['id']}.json").read_text())
    assert code == 0 and release["request"] == {"kind": "release-hold", "unit": "blueprint-scene-progression.timer"}


@pytest.mark.parametrize("duration", ["0s", "59s", "25h", "1d", "1.5h", "60m; echo bad"])
def test_hold_client_rejects_invalid_duration(duration: str) -> None:
    with pytest.raises(SystemExit) as caught:
        client.build_parser().parse_args(["hold", "blueprint-scene-progression.timer", "--owner", "alice",
                                          "--reason", "inspect", "--for", duration])
    assert caught.value.code == 2


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


def test_usage_prints_the_capacity_usage_as_tables(door: dict[str, Any]) -> None:
    summary = door["base"] / "cp" / "capacity" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({
        "schema_version": "control_plane_capacity_summary.v1", "level": "warning", "alerts": [], "mounts": [],
        "usage": {
            "status": "complete", "observed_at_epoch": 900.0, "age_seconds": 1200.0,
            "mounts": [{"mount": "/", "used_bytes": 150 * 1024**3, "surveyed_bytes": 145 * 1024**3,
                        "classified_bytes": 130 * 1024**3, "attributed_fraction": 0.9667}],
            "by_class": [{"storage_class": "work", "allocated_bytes": 60 * 1024**3,
                          "apparent_bytes": 58 * 1024**3, "files": 1234}],
            "top_roots": [{"root": "/var/lib/blueprint/pubsub-handoffs", "storage_class": "work",
                           "allocated_bytes": 60 * 1024**3}],
            "top_owners": [{"owner": "scene:site-capture-1", "root": "/var/lib/blueprint/pubsub-handoffs",
                            "storage_class": "work", "allocated_bytes": 12 * 1024**3}],
            "unclassified_roots": [{"root": "/var/lib/blueprint/something-new", "allocated_bytes": 2 * 1024**2}],
            "orphan_scratch_bytes": 6 * 1024**3, "orphan_scratch_count": 3,
            "orphan_scratch_roots": [{"root": "/mnt/blueprint-work/loose-run",
                                      "allocated_bytes": 4 * 1024**3,
                                      "newest_mtime_epoch": 1_700_000_000}],
        },
    }), encoding="utf-8")
    code, out = _run("usage")
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == "usage survey: complete, 1200 s old"
    assert any(line.split() == ["/", "150.0", "GiB", "145.0", "GiB", "130.0", "GiB", "96.7%"] for line in lines)
    assert any(line.split()[:3] == ["scene:site-capture-1", "work", "12.0"] for line in lines)
    assert any(line.split() == ["/var/lib/blueprint/something-new", "2.0", "MiB"] for line in lines)
    assert "unowned scratch: 6.0 GiB in 3 folders" in out
    assert "/mnt/blueprint-work/loose-run" in out
    assert "2023-11-14T22:13:20Z" in out
    assert "{" not in out


def test_usage_without_a_capacity_summary_exits_1(door: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _run("usage")
    assert code == 1 and out == ""
    assert "capacity_unavailable:FileNotFoundError" in capsys.readouterr().err

def test_retire_scene_workspace_spools_a_plan_or_an_apply(door: dict[str, Any]) -> None:
    code, out = _run("retire-scene-workspace", "site-capture-1")
    spooled = json.loads((door["state"] / "requests" / "pending" / f"{json.loads(out)['id']}.json").read_text())
    assert code == 0 and spooled["request"] == {"kind": "retire-scene-workspace", "scene_id": "site-capture-1",
                                                "apply": False}
    code, out = _run("retire-scene-workspace", "site-capture-1", "--bucket", "blueprint-8c1ca.appspot.com", "--apply")
    spooled = json.loads((door["state"] / "requests" / "pending" / f"{json.loads(out)['id']}.json").read_text())
    assert code == 0 and spooled["request"] == {"kind": "retire-scene-workspace", "scene_id": "site-capture-1",
                                                "bucket": "blueprint-8c1ca.appspot.com", "apply": True}


@pytest.mark.parametrize(("status", "exit_code"), [("planned", 0), ("retired", 0), ("retained", 1), ("failed", 1)])
def test_waiting_on_a_retirement_exits_by_its_outcome(door: dict[str, Any], status: str, exit_code: int) -> None:
    code, out = _run("retire-scene-workspace", "site-capture-1")
    request_id = json.loads(out)["id"]
    results = door["state"] / "requests" / "results"
    (results / f"{request_id}.json").write_text(json.dumps({"status": "launched", "unit": "u.service"}))
    outcome = {"status": status, "code": "pinned" if status == "retained" else None, "exit_code": 0}
    (results / f"{request_id}.outcome.json").write_text(json.dumps(outcome))

    code, out = _run("request", request_id, "--wait", "--poll", "0.05", "--timeout", "5")

    assert code == exit_code and json.loads(out)["outcome"] == outcome  # a retained scene says why


def test_waiting_on_notifier_repair_accepts_its_success_outcome(door: dict[str, Any]) -> None:
    code, out = _run("unit", "repair-notifier-binding", "blueprint-pipeline-control-plane.service",
                     "--expected-postcheck-sha256", "sha256:" + "a" * 64, "--expected-source-commit", SHA)
    request_id = json.loads(out)["id"]
    results = door["state"] / "requests" / "results"
    (results / f"{request_id}.json").write_text(json.dumps({"status": "launched", "unit": "u.service"}))
    outcome = {"status": "repaired", "exit_code": 0}
    (results / f"{request_id}.outcome.json").write_text(json.dumps(outcome))
    code, out = _run("request", request_id, "--wait", "--poll", "0.05", "--timeout", "5")
    assert code == 0 and json.loads(out)["outcome"] == outcome


def test_client_submits_a_canonical_scene_restore(door: dict[str, Any]) -> None:
    code, out = _run("restore-scene-workspace", "scene-1", "--bucket", "blueprint-8c1ca.appspot.com")
    request_id = json.loads(out)["id"]
    spooled = json.loads((door["state"] / "requests" / "pending" / f"{request_id}.json").read_text())
    assert code == 0 and spooled["request"] == {"kind": "restore-scene-workspace", "scene_id": "scene-1",
                                                "bucket": "blueprint-8c1ca.appspot.com"}


def test_client_submits_one_canary_attempt_s_provider_output_resume(door: dict[str, Any]) -> None:
    code, out = _run("provider-output-resume", "activation-1", "3", "--ingest")
    request_id = json.loads(out)["id"]
    spooled = json.loads((door["state"] / "requests" / "pending" / f"{request_id}.json").read_text())
    assert code == 0 and spooled["request"] == {"kind": "provider-output-resume", "run": "activation-1",
                                                "attempt": 3, "ingest": True}


def test_a_retirement_needs_the_operate_scope(door: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", READ_ONLY)
    assert _run("retire-scene-workspace", "site-capture-1")[0] == 3


def test_historical_unit_acceptance_is_not_objective_completion():
    assert client._terminal({"request": {"kind": "unit"}, "result": {"status": "done"}}) is None
    assert client._terminal({"observed_outcome": {"status": "observed_completed"}}) is True
    assert client._terminal({"observed_outcome": {"status": "observed_failed"}}) is False


@pytest.mark.parametrize("dispatch", [False, True])
def test_selected_handoff_cli_defaults_to_inspect_and_dispatch_requires_flag(door: dict[str, Any], dispatch: bool) -> None:
    from test_operator_door_requests import _selected_request
    body = _selected_request(); body.pop("kind"); body.pop("mode")
    selector = door["base"] / "selected.json"; selector.write_text(json.dumps(body))
    args = ["selected-handoff", "--selector-file", str(selector)]
    if dispatch: args.append("--dispatch")
    code, out = _run(*args)
    assert code == 0
    request_id = json.loads(out)["id"]
    request = json.loads((door["state"] / "requests" / "pending" / f"{request_id}.json").read_text())["request"]
    assert request == {**body, "kind": "selected-handoff", "mode": "dispatch" if dispatch else "inspect"}


def test_selected_handoff_cli_refuses_oversized_input_and_read_only_actor(door: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    from test_operator_door_requests import _selected_request
    selector = door["base"] / "selected.json"; selector.write_text(" " * 4097)
    assert _run("selected-handoff", "--selector-file", str(selector))[0] == 2
    assert not list((door["state"] / "requests" / "pending").iterdir())
    body = _selected_request(); body.pop("kind"); body.pop("mode"); selector.write_text(json.dumps(body))
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", READ_ONLY)
    assert _run("selected-handoff", "--selector-file", str(selector), "--dispatch")[0] == 3
    assert not list((door["state"] / "requests" / "pending").iterdir())
