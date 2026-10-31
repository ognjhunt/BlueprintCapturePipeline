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
import time
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


def _asset_grant(**changes):
    value = {"schema_version": "embodiment_asset_preparation_request.v1", "authorization_id": "development-only-fixture",
             "release_commit": "a" * 40, "embodiment_ids": ["universal_robots_ur5e", "fanuc_crx_10ia"],
             "max_spend_usd": 4.0, "max_hourly_rate_usd": .8, "ttl_seconds": 14_400,
             "expires_at_epoch": int(time.time()) + 86400, "authorization_reference": "development_only fixture"}
    value.update(changes)
    value["request_digest"] = "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    return value


def _public_policy_request(*, yam=False):
    from operator_door.public_policy_registration_contract import digest
    value = {"schema_version": "openwam_piper_policy_install_request.v1", "installation_id": "development-only",
        "release_commit": "a" * 40, "preparation_authorization_id": "development-only-assets",
        "preparation_publication_digest": "sha256:" + "b" * 64,
        "candidate_id": "openwam_alpha_piper_bimanual", "team_id": "blueprint-public-models",
        "checkpoint_id": "openwam-alpha-piper-12bfca6f", "policy_profile": "openwam_alpha_real_piper.v1",
        "expected_registry_digest": "sha256:" + "c" * 64, "authorization_reference": "development_only public policy request"}
    if yam:
        value.update(schema_version="molmoact2_yam_policy_install_request.v1", candidate_id="molmoact2_bimanual_yam",
            checkpoint_id="molmoact2-bimanual-yam-8dcbed66", policy_profile="molmoact2_bimanual_yam.v1")
    value["request_digest"] = digest(value)
    return value


@pytest.mark.parametrize("yam", [False, True])
def test_public_policy_install_cli_http_keeps_exact_request_and_retry_identity(door, tmp_path, monkeypatch, yam):
    request = _public_policy_request(yam=yam)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request))
    kind = "install-public-yam-policy" if yam else "install-public-piper-policy"
    args = (kind, "--request-file", str(path))
    code, output = _run(*args)
    assert code == 0
    request_id = json.loads(output)["id"]
    retry_code, retry = _run(*args)
    assert retry_code == 0 and json.loads(retry)["id"] == request_id
    saved = json.loads((door["state"] / "requests" / "pending" / (request_id + ".json")).read_text())
    assert saved["request"] == {"kind": kind, "request": request}
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", READ_ONLY)
    assert _run(*args)[0] == 3


def test_public_policy_install_cli_rejects_unsealed_or_duplicate_fields_without_network(tmp_path, monkeypatch):
    path = tmp_path / "request.json"
    request = _public_policy_request()
    request["expected_registry_digest"] = "sha256:" + "d" * 64
    path.write_text(json.dumps(request))
    monkeypatch.setattr(client, "_request", lambda *_a, **_k: pytest.fail("malformed request reached network"))
    assert _run("install-public-piper-policy", "--request-file", str(path))[0] == 2
    path.write_text('{"installation_id":"first",' + json.dumps(_public_policy_request())[1:])
    assert _run("install-public-piper-policy", "--request-file", str(path))[0] == 2


def test_public_policy_install_cli_rejects_wrong_fixed_model_operation_before_http(tmp_path, monkeypatch):
    path = tmp_path / "request.json"
    path.write_text(json.dumps(_public_policy_request()))
    monkeypatch.setattr(client, "_request", lambda *_a, **_k: pytest.fail("wrong profile reached network"))
    assert _run("install-public-yam-policy", "--request-file", str(path))[0] == 2


def test_public_asset_batch_cli_api_preserves_authority_and_retry_identity(door, tmp_path):
    grant = _asset_grant()
    path = tmp_path / "grant.json"
    path.write_text(json.dumps(grant))
    arguments = ("public-embodiment-preparation", "--request-file", str(path))
    code, output = _run(*arguments)
    assert code == 0
    request_id = json.loads(output)["id"]
    retry_code, retry = _run(*arguments)
    assert retry_code == 0 and json.loads(retry)["id"] == request_id
    saved = json.loads((door["state"] / "requests" / "pending" / (request_id + ".json")).read_text())
    assert saved["request"] == {"kind": "public-embodiment-preparation", "request": grant}
    path.write_text(json.dumps(_asset_grant(embodiment_ids=["fanuc_crx_10ia"])))
    assert _run(*arguments)[0] == 2


def test_public_asset_batch_requires_operate_scope(door, tmp_path, monkeypatch):
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", READ_ONLY)
    path = tmp_path / "grant.json"
    path.write_text(json.dumps(_asset_grant()))
    assert _run("public-embodiment-preparation", "--request-file", str(path))[0] == 3
    assert not list((door["state"] / "requests" / "pending").glob("*.json"))


@pytest.mark.parametrize("malformation", ["duplicate", "nonfinite", "digest", "large", "symlink", "missing", "extra"])
def test_public_asset_client_rejects_ambiguous_files_before_http(tmp_path, monkeypatch, malformation):
    monkeypatch.setattr(client, "_json", lambda *a, **kw: pytest.fail("invalid file reached API"))
    value = _asset_grant()
    path = tmp_path / "grant.json"
    raw = json.dumps(value)
    if malformation == "duplicate":
        raw = raw.replace('"max_spend_usd": 4.0', '"max_spend_usd": 400, "max_spend_usd": 4.0')
    elif malformation == "nonfinite":
        raw = raw.replace('"max_spend_usd": 4.0', '"max_spend_usd": NaN')
    elif malformation == "digest":
        raw = raw.replace('"max_spend_usd": 4.0', '"max_spend_usd": 3.0')
    elif malformation == "large":
        raw = raw + " " * 16384
    elif malformation == "extra":
        value["command"] = "run-queue"
        raw = json.dumps(value)
    if malformation == "symlink":
        source = tmp_path / "source.json"
        source.write_text(raw)
        path.symlink_to(source)
    elif malformation != "missing":
        path.write_text(raw)
    assert _run("public-embodiment-preparation", "--request-file", str(path))[0] == 2


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


def _settle(door: dict[str, Any], request_id: str, *, result: dict | None = None, outcome: dict | None = None) -> None:
    results = door["state"] / "requests" / "results"
    if result is not None:
        (results / f"{request_id}.json").write_text(json.dumps(result))
    if outcome is not None:
        (results / f"{request_id}.outcome.json").write_text(json.dumps(outcome))


def test_wait_keeps_waiting_while_a_deploy_is_parked(door: dict[str, Any]) -> None:
    _, out = _run("deploy", SHA)
    request_id = json.loads(out)["id"]
    _settle(door, request_id, result={"status": "parked", "parked_behind": "blueprint-x-deploy.service"})
    code, _ = _run("request", request_id, "--wait", "--poll", "0.05", "--timeout", "0.2")
    assert code == 1  # gave up waiting: parked is not a verdict

    _settle(door, request_id, outcome={"status": "deployed"})
    code, _ = _run("request", request_id, "--wait", "--poll", "0.05", "--timeout", "5")
    assert code == 0


@pytest.mark.parametrize(("newer_outcome", "expected"), [("deployed", 0), ("failed", 1)])
def test_wait_on_a_superseded_deploy_reports_the_newer_deploys_result(
    door: dict[str, Any], newer_outcome: str, expected: int,
) -> None:
    _, out = _run("deploy", SHA)
    older = json.loads(out)["id"]
    _, out = _run("deploy", "f" * 40)
    newer = json.loads(out)["id"]
    _settle(door, older, outcome={"status": "superseded", "superseded_by": newer})
    _settle(door, newer, outcome={"status": newer_outcome})

    code, printed = _run("request", older, "--wait", "--poll", "0.05", "--timeout", "5")

    assert code == expected
    assert json.loads(printed)["id"] == newer


def test_wait_ends_well_when_the_active_release_already_contains_the_commit(door: dict[str, Any]) -> None:
    _, out = _run("deploy", SHA)
    request_id = json.loads(out)["id"]
    _settle(door, request_id, outcome={"status": "superseded", "superseded_by_commit": "f" * 40})

    code, _ = _run("request", request_id, "--wait", "--poll", "0.05", "--timeout", "5")

    assert code == 0


def test_wait_fails_when_a_parked_commit_was_not_contained(door: dict[str, Any]) -> None:
    _, out = _run("deploy", SHA)
    request_id = json.loads(out)["id"]
    _settle(door, request_id, outcome={"status": "failed", "code": "parked_commit_not_contained"})

    code, _ = _run("request", request_id, "--wait", "--poll", "0.05", "--timeout", "5")

    assert code == 1


class _GitHub:
    """The read-only GitHub answers --latest-attested may ask for, and a record of every question."""

    def __init__(self, monkeypatch, runs, jobs, compare="ahead"):
        self.routes: list[str] = []
        self.runs, self.jobs, self.compare = runs, jobs, compare
        monkeypatch.setattr(client, "_github_json", self.read)

    def read(self, route: str) -> dict[str, Any]:
        self.routes.append(route)
        if route.startswith("/actions/workflows/ci.yml/runs?"):
            return {"workflow_runs": self.runs}
        if route.startswith("/compare/"):
            return {"status": self.compare}
        run_id = int(route.split("/")[3])
        return {"jobs": self.jobs.get(run_id, [])}


ATTESTED = [{"name": "SBOM, license, and provenance contract", "conclusion": "success"}]
UNATTESTED = [{"name": "SBOM, license, and provenance contract", "conclusion": "skipped"},
              {"name": "Ruff and claim lint", "conclusion": "failure"}]
NEW, MIDDLE, OLD = "a" * 40, "b" * 40, "c" * 40


def _ci_run(run_id: int, sha: str, status: str = "completed") -> dict[str, Any]:
    return {"id": run_id, "head_sha": sha, "status": status}


def _status_with_active(monkeypatch, active: str | None) -> None:
    real = client._json

    def fake(method, route, **kwargs):
        if route == "/status":
            return {"active_release": {"commit": active} if active else None}
        return real(method, route, **kwargs)

    monkeypatch.setattr(client, "_json", fake)


def test_latest_attested_deploys_the_newest_commit_whose_job_succeeded(door, monkeypatch) -> None:
    github = _GitHub(monkeypatch, [_ci_run(3, NEW, "queued"), _ci_run(2, MIDDLE), _ci_run(1, OLD)],
                     {2: UNATTESTED, 1: ATTESTED})
    _status_with_active(monkeypatch, "d" * 40)

    code, out = _run("deploy", "--latest-attested")

    assert code == 0
    request_id = json.loads(out)["id"]
    spooled = json.loads((door["state"] / "requests" / "pending" / f"{request_id}.json").read_text())
    assert spooled["request"]["commit"] == OLD
    assert spooled["request"]["kind"] == "deploy"
    # A run still queued cannot have attested, so it costs no request against the hourly limit.
    assert "/actions/runs/3/jobs" not in " ".join(github.routes)
    assert github.routes[0] == "/actions/workflows/ci.yml/runs?branch=main&event=push&per_page=30"


def test_latest_attested_refuses_a_commit_that_is_already_running(door, monkeypatch, capsys) -> None:
    _GitHub(monkeypatch, [_ci_run(1, NEW)], {1: ATTESTED})
    _status_with_active(monkeypatch, NEW)

    code, _ = _run("deploy", "--latest-attested")

    assert code == 2
    assert "already the active release" in capsys.readouterr().err
    assert not list((door["state"] / "requests" / "pending").glob("*.json"))


def test_latest_attested_refuses_to_roll_back_what_is_running(door, monkeypatch, capsys) -> None:
    github = _GitHub(monkeypatch, [_ci_run(1, OLD)], {1: ATTESTED}, compare="behind")
    _status_with_active(monkeypatch, NEW)

    code, _ = _run("deploy", "--latest-attested")

    assert code == 2
    assert f"/compare/{NEW}...{OLD}?per_page=1" in github.routes
    assert "already contains" in capsys.readouterr().err
    assert not list((door["state"] / "requests" / "pending").glob("*.json"))


def test_latest_attested_refuses_when_nothing_is_attested(door, monkeypatch, capsys) -> None:
    _GitHub(monkeypatch, [_ci_run(2, NEW), _ci_run(1, OLD)], {2: UNATTESTED, 1: []})

    code, _ = _run("deploy", "--latest-attested")

    assert code == 2
    assert "no attested commit" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ("deploy", SHA, "--latest-attested"),
    ("deploy", "--latest-attested", "--release-provenance", "receipt.json"),
    ("deploy", "--latest-attested", "--release-provenance-sha256", "a" * 64),
    ("deploy",),
])
def test_deploy_needs_exactly_one_way_to_name_its_commit(door, monkeypatch, capsys, argv) -> None:
    monkeypatch.setattr(client, "_github_json", lambda _route: pytest.fail("no GitHub read expected"))

    code, _ = _run(*argv)

    assert code == 2
    assert not list((door["state"] / "requests" / "pending").glob("*.json"))


def test_a_github_read_never_carries_the_door_token(monkeypatch) -> None:
    seen: list[Any] = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    def opened(request, timeout):
        seen.append(request)
        return Response(b'{"workflow_runs": []}')

    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", "secret-door-token")
    monkeypatch.setattr(client.urllib.request, "urlopen", opened)

    assert client._github_json("/actions/workflows/ci.yml/runs") == {"workflow_runs": []}
    assert seen[0].full_url.startswith("https://api.github.com/repos/ognjhunt/BlueprintCapturePipeline/")
    assert seen[0].get_method() == "GET"
    assert seen[0].get_header("Authorization") is None


def test_github_failure_is_a_network_error_not_a_crash(monkeypatch) -> None:
    def down(_request, timeout):
        raise OSError("down")

    monkeypatch.setattr(client.urllib.request, "urlopen", down)

    with pytest.raises(client.DoorError) as failure:
        client._github_json("/x")
    assert failure.value.exit_code == 4


def test_wait_fails_on_a_refusal(door: dict[str, Any]) -> None:
    code, out = _run("deploy", SHA)
    request_id = json.loads(out)["id"]
    results = door["state"] / "requests" / "results"
    (results / f"{request_id}.json").write_text(json.dumps({"status": "refused", "code": "deploy_in_progress:x"}))
    code, _ = _run("request", request_id, "--wait", "--poll", "0.05", "--timeout", "5")
    assert code == 1


@pytest.mark.parametrize("kind", ["public-embodiment-preparation", "install-public-piper-policy",
    "install-public-yam-policy", "install-public-franka-policy", "install-scripted-robot-baselines"])
@pytest.mark.parametrize("change,expected", [({}, 0), ({"status": "failed"}, 1),
    ({"exit_code": 2}, 1), ({"exit_code": False}, 1), ({"exit_code": None}, 1),
    ({"disposition": "wrong_operation"}, 1)])
def test_public_operation_submit_wait_reports_only_its_completed_command(
    door: dict[str, Any], tmp_path: Path, kind: str, change: dict, expected: int,
) -> None:
    if kind == "public-embodiment-preparation":
        value, disposition = _asset_grant(), "grant_queued"
    elif kind == "install-scripted-robot-baselines":
        from tests.test_operator_door_scripted_registration import request
        value, disposition = request(), "public_policy_registered"
    else:
        from tests.test_operator_door_public_policy_registration import request
        value = request(yam=kind == "install-public-yam-policy", franka=kind == "install-public-franka-policy")
        disposition = "public_policy_registered"
    path = tmp_path / "public-operation.json"
    path.write_text(json.dumps(value))
    arguments = (kind, "--request-file", str(path))
    code, output = _run(*arguments)
    assert code == 0
    request_id = json.loads(output)["id"]
    root = door["state"] / "requests"
    (root / "pending" / f"{request_id}.json").rename(root / "completed" / f"{request_id}.json")
    outcome = {"status": "completed", "exit_code": 0, "disposition": disposition, **change}
    (root / "results" / f"{request_id}.outcome.json").write_text(json.dumps(outcome))

    # Retry the exact submission through real HTTP and its actual --wait path.
    # A completed enqueue/registration does not claim the simulator ran.
    code, output = _run(*arguments, "--wait", "--poll", "0.01", "--timeout", "5")
    final = json.loads(output)
    assert code == expected and final["id"] == request_id and final["outcome"] == outcome
    assert not list((root / "pending").glob("*.json"))


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
def test_selected_handoff_cli_defaults_to_inspect_and_dispatch_requires_flag(
    door: dict[str, Any], dispatch: bool
) -> None:
    from tests.test_operator_door_requests import _selected_request

    body = _selected_request()
    body.pop("kind")
    body.pop("mode")
    selector = door["base"] / "selected.json"
    selector.write_text(json.dumps(body))
    args = ["selected-handoff", "--selector-file", str(selector)]
    if dispatch:
        args.append("--dispatch")
    code, out = _run(*args)
    assert code == 0
    request_id = json.loads(out)["id"]
    request = json.loads(
        (door["state"] / "requests" / "pending" / f"{request_id}.json").read_text()
    )["request"]
    assert request == {
        **body,
        "kind": "selected-handoff",
        "mode": "dispatch" if dispatch else "inspect",
    }


def test_selected_handoff_cli_refuses_oversized_input_and_read_only_actor(
    door: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_operator_door_requests import _selected_request

    selector = door["base"] / "selected.json"
    selector.write_text(" " * 4097)
    assert _run("selected-handoff", "--selector-file", str(selector))[0] == 2
    assert not list((door["state"] / "requests" / "pending").iterdir())
    body = _selected_request()
    body.pop("kind")
    body.pop("mode")
    selector.write_text(json.dumps(body))
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", READ_ONLY)
    assert _run("selected-handoff", "--selector-file", str(selector), "--dispatch")[0] == 3
    assert not list((door["state"] / "requests" / "pending").iterdir())


@pytest.mark.parametrize("dispatch", [False, True])
@pytest.mark.parametrize("status,exit_code,expected", [("completed", 0, 0), ("completed", 1, 1), ("failed", 0, 1)])
def test_waiting_on_selected_handoff_reports_command_outcome(
    door: dict[str, Any], dispatch: bool, status: str, exit_code: int, expected: int
) -> None:
    from tests.test_operator_door_requests import _selected_request

    body = _selected_request()
    body.pop("kind")
    body.pop("mode")
    selector = door["base"] / "selected-wait.json"
    selector.write_text(json.dumps(body))
    args = ["selected-handoff", "--selector-file", str(selector)]
    if dispatch:
        args.append("--dispatch")
    accepted, output = _run(*args)
    assert accepted == 0
    request_id = json.loads(output)["id"]
    root = door["state"] / "requests"
    (root / "pending" / f"{request_id}.json").rename(root / "completed" / f"{request_id}.json")
    outcome = {"status": status, "exit_code": exit_code}
    (root / "results" / f"{request_id}.outcome.json").write_text(json.dumps(outcome))
    code, output = _run("request", request_id, "--wait", "--poll", "0.05", "--timeout", "5")
    assert code == expected
    assert json.loads(output)["outcome"] == outcome


def test_selected_handoff_cli_links_explicit_recovery_and_keeps_selector_plain(door):
    from tests.test_operator_door_requests import _selected_request
    body = {key: value for key, value in _selected_request().items() if key not in {"kind", "mode"}}
    selector = door["base"] / "selected-resume.json"
    selector.write_text(json.dumps(body))
    prior = "20261009T154718Z-selected-handoff-9ef60661"
    args = ["selected-handoff", "--selector-file", str(selector), "--resume-request", prior]
    assert _run(*args)[0] == 2
    assert not list((door["state"] / "requests" / "pending").iterdir())
    code, output = _run(*args, "--dispatch")
    assert code == 0
    request_id = json.loads(output)["id"]
    request = json.loads((door["state"] / "requests" / "pending" / f"{request_id}.json").read_text())["request"]
    assert request == {**body, "kind": "selected-handoff", "mode": "dispatch", "resume_request_id": prior}
    assert json.loads(_run(*args, "--dispatch")[1])["id"] == request_id
    selector.write_text(json.dumps({**body, "resume_request_id": prior}))
    assert _run(*args, "--dispatch")[0] == 2


@pytest.mark.parametrize("mode", ["atlas-status", "atlas-retry", "preparation-resume"])
def test_selected_atlas_cli_requires_explicit_snapshot_review_and_excludes_native_dispatch(door, mode):
    from tests.test_operator_door_requests import _selected_request
    body = {key: value for key, value in _selected_request().items() if key not in {"kind", "mode"}}
    selector = door["base"] / "selected-atlas.json"
    selector.write_text(json.dumps(body))
    args = ["selected-handoff", "--selector-file", str(selector),
            "--resume-preparation" if mode == "preparation-resume" else "--" + mode]
    retry = mode != "atlas-status"
    digest = "sha256:" + "c" * 64
    if retry:
        assert _run(*args)[0] == 2
        args += ["--atlas-status-digest", digest]
    else:
        assert _run(*args, "--atlas-status-digest", digest)[0] == 2
    code, output = _run(*args)
    assert code == 0
    request_id = json.loads(output)["id"]
    request = json.loads((door["state"] / "requests" / "pending" / f"{request_id}.json").read_text())["request"]
    assert request == {**body, "kind": "selected-handoff", "mode": mode,
                       **({"atlas_status_digest": digest} if retry else {})}
    with pytest.raises(SystemExit) as conflict:
        _run(*args, "--dispatch")
    assert conflict.value.code == 2
    assert _run(*args, "--resume-request", "20261009T154718Z-selected-handoff-9ef60661")[0] == 2


def test_selected_handoff_cli_discovers_original_selectors_without_execution(door):
    from tests.test_operator_door_requests import _selected_request
    fields = {"bucket", "scene_id", "capture_id", "marker_generation"}
    body = {key: value for key, value in _selected_request().items() if key in fields}
    selector = door["base"] / "selected-discovery.json"
    selector.write_text(json.dumps(body))
    args = ["selected-handoff", "--selector-file", str(selector), "--discover-selector"]
    code, output = _run(*args)
    assert code == 0
    request_id = json.loads(output)["id"]
    request = json.loads((door["state"] / "requests" / "pending" / f"{request_id}.json").read_text())["request"]
    assert request == {**body, "kind": "selected-handoff", "mode": "discover-selector"}
    for flag in ("--dispatch", "--atlas-status", "--atlas-retry"):
        with pytest.raises(SystemExit):
            _run(*args, flag)
    assert _run(*args, "--atlas-status-digest", "sha256:" + "c" * 64)[0] == 2
    assert _run(*args, "--resume-request", "20261009T154718Z-selected-handoff-9ef60661")[0] == 2


def test_franka_policy_cli_http_preserves_exact_model_batch_and_retry(door, tmp_path, monkeypatch):
    from tests.test_operator_door_public_policy_registration import request
    value = request(franka=True)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(value))
    arguments = ("install-public-franka-policy", "--request-file", str(path))
    code, output = _run(*arguments)
    assert code == 0
    request_id = json.loads(output)["id"]
    retry_code, retry = _run(*arguments)
    assert retry_code == 0 and json.loads(retry)["id"] == request_id
    saved = json.loads((door["state"] / "requests" / "pending" / (request_id + ".json")).read_text())
    assert saved["request"] == {"kind": "install-public-franka-policy", "request": value}
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", READ_ONLY)
    assert _run(*arguments)[0] == 3


def test_trlc_policy_cli_http_preserves_exact_model_batch_and_retry(door, tmp_path, monkeypatch):
    from tests.test_operator_door_trlc_registration import request
    value = request()
    path = tmp_path / "request.json"
    path.write_text(json.dumps(value))
    arguments = ("install-public-trlc-policy", "--request-file", str(path))
    code, output = _run(*arguments)
    assert code == 0
    request_id = json.loads(output)["id"]
    retry_code, retry = _run(*arguments)
    assert retry_code == 0 and json.loads(retry)["id"] == request_id
    saved = json.loads((door["state"] / "requests" / "pending" / (request_id + ".json")).read_text())
    assert saved["request"] == {"kind": "install-public-trlc-policy", "request": value}
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", READ_ONLY)
    assert _run(*arguments)[0] == 3


def test_scripted_batch_cli_http_binds_exact_batch_and_idempotent_operation(door, tmp_path, monkeypatch):
    from tests.test_operator_door_scripted_registration import request
    value = request()
    path = tmp_path / "request.json"
    path.write_text(json.dumps(value))
    arguments = ("install-scripted-robot-baselines", "--request-file", str(path))
    code, output = _run(*arguments)
    assert code == 0
    request_id = json.loads(output)["id"]
    retry_code, retry = _run(*arguments)
    assert retry_code == 0 and json.loads(retry)["id"] == request_id
    saved = json.loads((door["state"] / "requests" / "pending" / (request_id + ".json")).read_text())
    assert saved["request"] == {"kind": "install-scripted-robot-baselines", "request": value}
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", READ_ONLY)
    assert _run(*arguments)[0] == 3


def test_native_g1_install_real_cli_http_preserves_exact_source_request_and_retry(door, tmp_path, monkeypatch):
    from tests.test_operator_door_g1_registration import request
    value = request()
    path = tmp_path / "g1-request.json"
    path.write_text(json.dumps(value))
    args = ("install-native-g1-policies", "--request-file", str(path))
    code, output = _run(*args)
    assert code == 0
    request_id = json.loads(output)["id"]
    retry_code, retry = _run(*args)
    assert retry_code == 0 and json.loads(retry)["id"] == request_id
    saved = json.loads((door["state"] / "requests/pending" / (request_id + ".json")).read_text())
    assert saved["request"] == {"kind": "install-native-g1-policies", "request": value}
    monkeypatch.setenv("BLUEPRINT_OPERATOR_DOOR_TOKEN", READ_ONLY)
    assert _run(*args)[0] == 3


def _scene_window_args(*extra: str, expires: str | None = None) -> list[str]:
    instant = expires or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + 3 * 86400))
    return ["extend-scene-execution-window", "scene-" + "a" * 64, "--intent-digest", "sha256:" + "b" * 64,
            "--expires-at", instant, "--authorization-reference", "founder approval 2026-10-31 (time only)", *extra]


def test_scene_window_client_plans_by_default_and_applies_only_with_ack(door: dict[str, Any]) -> None:
    code, out = _run(*_scene_window_args())
    assert code == 0
    request_id = json.loads(out)["id"]
    spooled = json.loads((door["state"] / "requests" / "pending" / f"{request_id}.json").read_text())
    assert spooled["requested_by"] == "cloud"
    assert spooled["request"]["apply"] is False and spooled["request"]["ack"] == "extend-scene-execution-window"
    assert set(spooled["request"]) == {"kind", "intent_id", "intent_digest", "expires_at",
                                       "authorization_reference", "ack", "apply"}
    assert _run(*_scene_window_args("--apply"))[0] == 2
    code, out = _run(*_scene_window_args("--apply", "--ack", "extend-scene-execution-window"))
    applied = json.loads((door["state"] / "requests" / "pending" / f"{json.loads(out)['id']}.json").read_text())
    assert code == 0 and applied["request"]["apply"] is True


@pytest.mark.parametrize("expires", ["2020-01-01T00:00:00Z", "2099-01-01T00:00:00Z", "tomorrow"])
def test_scene_window_client_refuses_out_of_bounds_expiry_before_http(expires: str, monkeypatch) -> None:
    monkeypatch.setattr(client, "_json", lambda *a, **k: pytest.fail("no request may be sent"))
    assert _run(*_scene_window_args(expires=expires))[0] == 2


def test_scene_window_outcomes_finish_a_wait() -> None:
    for status, verdict in (("planned", True), ("extended", True), ("already_covered", True),
                            ("failed", False), ("refused", False)):
        state = {"request": {"kind": "extend-scene-execution-window"}, "outcome": {"status": status}}
        assert client._terminal(state) is verdict
