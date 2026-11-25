"""The door's ``test-company``: one signed website call for Blueprint's internal test company. Fakes only.

The website opens internal test jobs to that company only (Blueprint-WebApp ``server/utils/internalTestCompany.ts``,
owner 2026-11-25); the door only asks it, with the same signed transport as ``test-job``.
"""
# Covers: src/blueprint_pipeline/internal_test_company_door.py
# Covers: src/blueprint_pipeline/internal_test_job_door.py
# Covers: deploy/operator-door/operator_door/requests.py
# Covers: deploy/operator-door/operator_door/spool_runner.py
# Covers: deploy/operator-door/door-test-company.sh
# Covers: deploy/operator-door/install.sh
# Covers: scripts/operator_door.py
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import internal_test_company_door as company
from blueprint_pipeline.internal_test_job_door import Refused
from tests.test_operator_door_scripts import (
    SETPRIV_STUB,
    _run,
    _write_stub,
    env,  # noqa: F401 - a fixture
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "operator-door"))
# Bind the door package now (see tests/test_internal_test_jobs.py): the client is also named ``operator_door``.
if not hasattr(sys.modules.get("operator_door"), "__path__"):
    sys.modules.pop("operator_door", None)
importlib.import_module("operator_door")

JOB = "itest-" + "c" * 40
THIS = "20261125T050000Z-test-company-0000abcd"
ANSWER = {"kind": "test-company", "action": "answer", "job_id": JOB, "state": "interested",
          "inputs": "We run this job at two sites."}


@pytest.mark.parametrize(("body", "expected"), [
    ({"kind": "test-company", "action": "ensure"}, {"kind": "test-company", "action": "ensure"}),
    ({"kind": "test-company", "action": "send", "job_id": JOB}, {"kind": "test-company", "action": "send", "job_id": JOB}),
    ({"kind": "test-company", "action": "site-link", "job_id": JOB},
     {"kind": "test-company", "action": "site-link", "job_id": JOB}),
    (ANSWER, ANSWER),
    ({key: value for key, value in ANSWER.items() if key != "inputs"}, {**ANSWER, "inputs": ""}),
])
def test_the_door_admits_one_action_for_the_test_company(body, expected):
    from operator_door.requests import required_scope, validate_request, validate_request_id

    assert required_scope("test-company") == "operate" and validate_request_id(THIS) == THIS
    assert validate_request(body) == expected


@pytest.mark.parametrize(("body", "code"), [
    ({"kind": "test-company"}, "test_company_action_invalid"),
    ({"kind": "test-company", "action": "delete"}, "test_company_action_invalid"),
    ({"kind": "test-company", "action": "ensure", "job_id": JOB}, "request_key_unknown:job_id"),
    ({"kind": "test-company", "action": "send"}, "test_company_job_invalid"),
    ({"kind": "test-company", "action": "send", "job_id": "8d3f0c2a-9e7b-4c1d-a0f5-6b2e9d4c7a31"},
     "test_company_job_invalid"),                                        # a site's job id: never a real job
    ({"kind": "test-company", "action": "site-link", "job_id": JOB + "/../x"}, "test_company_job_invalid"),
    ({**ANSWER, "state": "accepted"}, "test_company_state_invalid"),
    ({**ANSWER, "inputs": "x" * 2001}, "test_company_inputs_invalid"),
    ({**ANSWER, "inputs": "a\x00b"}, "test_company_inputs_invalid"),
    ({**ANSWER, "email": "someone@example.com"}, "request_key_unknown:email"),   # it names no address
    ({**ANSWER, "spend_usd": 5}, "request_key_unknown:spend_usd"),
])
def test_the_door_refuses_anything_else(body, code):
    from operator_door.requests import RequestRefused, validate_request

    with pytest.raises(RequestRefused, match=f"^{code}$"):
        validate_request(body)


def test_the_restricted_door_admits_it_on_the_owners_approval():
    text = (ROOT / "deploy/operator-door/operator_door/requests.py").read_text()
    assert '"test-company"' in text.split("if DISPATCHER_HOLD_ONLY:")[1].split("_COMMIT =")[0]
    assert 'with "Both"' in text


def test_the_door_runs_one_signed_call_with_only_the_sync_token_and_the_result_writable():
    from operator_door import spool_runner
    from operator_door.config import DoorConfig
    from operator_door.requests import validate_request

    config, request = DoorConfig(), validate_request(ANSWER)
    launch = spool_runner._LAUNCHES["test-company"]
    properties = launch.properties(config, request)
    assert launch.script == "door-test-company.sh" and JOB not in launch.label(request)
    assert "PrivateNetwork=yes" not in properties  # the website is called
    assert "BindReadOnlyPaths=/etc/blueprint/provider-secrets/pipeline_sync_token" in properties
    assert "TemporaryFileSystem=/etc/blueprint/provider-secrets:ro" in properties
    hidden = next(item for item in properties if item.startswith("InaccessiblePaths=")).split("=", 1)[1].split()
    assert {"-" + path for path in config.hidden_paths if path != "/etc/blueprint/provider-secrets"} == set(hidden)
    writable = next(item for item in properties if item.startswith("ReadWritePaths=")).split("=", 1)[1].split()
    assert writable == ["/var/lib/blueprint-operator-door/requests/results"]
    arguments = spool_runner._script_env(config, THIS, request, "cloud-agent")
    sent = next(item for item in arguments if item.startswith("--setenv=DOOR_TEST_COMPANY_REQUEST="))
    assert json.loads(sent.split("=", 2)[2]) == ANSWER
    assert (ROOT / "deploy/operator-door/door-test-company.sh").exists()
    assert '"$source_dir"/door-test-company.sh' in (ROOT / "deploy/operator-door/install.sh").read_text()


@pytest.mark.parametrize(("request_", "path", "body"), [
    ({"kind": "test-company", "action": "ensure"}, "/api/internal/pipeline/test-company", {}),
    ({"kind": "test-company", "action": "send", "job_id": JOB}, "/api/internal/pipeline/test-company/send",
     {"job_id": JOB}),
    ({"kind": "test-company", "action": "site-link", "job_id": JOB},
     f"/api/internal/pipeline/test-jobs/{JOB}/site-link", {}),
    (ANSWER, "/api/internal/pipeline/test-company/answer",
     {"job_id": JOB, "state": "interested", "inputs": "We run this job at two sites."}),
])
def test_each_action_is_one_signed_call_to_its_website_route(request_, path, body):
    calls = []

    def post(path_, body_, *, prefix):
        calls.append((path_, body_, prefix))
        return {"ok": True}
    assert company.run(request_, post=post) == {"door_outcome": "done", "action": request_["action"], "website": {"ok": True}}
    assert calls == [(path, body, "test_company")]


def test_the_websites_refusal_is_the_doors_refusal_and_an_outage_its_failure():
    def refused(*_args, **_kwargs):
        raise ValueError("test_company_webapp_http_409:not_an_internal_test_job")
    with pytest.raises(Refused, match="^test_company_webapp_http_409:not_an_internal_test_job$"):
        company.run({"kind": "test-company", "action": "send", "job_id": JOB}, post=refused)

    def down(*_args, **_kwargs):
        raise ValueError("test_company_webapp_http_503")
    with pytest.raises(ValueError, match="^test_company_webapp_http_503$"):
        company.run({"kind": "test-company", "action": "send", "job_id": JOB}, post=down)
    with pytest.raises(Refused, match="^test_company_request_invalid$"):
        company.run({"kind": "test-company", "action": "ensure", "job_id": JOB}, post=down)


def test_the_test_job_route_keeps_its_own_path_and_codes(monkeypatch):
    from blueprint_pipeline import internal_test_job_door as door_job

    seen = []
    monkeypatch.setattr(door_job, "post_signed", lambda path, body, *, prefix: seen.append((path, prefix)) or {})
    door_job.post_test_job({"idempotency_key": "k" * 8})
    assert seen == [("/api/internal/pipeline/test-jobs", "test_job")]


@pytest.fixture()
def company_env(env, tmp_path):  # noqa: F811 - the scripts module's fixture
    repository = tmp_path / "repository"
    (repository / "src" / "blueprint_pipeline").mkdir(parents=True)
    (repository / "src" / "blueprint_pipeline" / "internal_test_company_door.py").write_text("")
    _write_stub(tmp_path / "stubs" / "setpriv", SETPRIV_STUB)
    _write_stub(tmp_path / "job-python", "#!/bin/bash\nprintf 'job %s\\n' \"$*\" >> \"$STUB_LOG\"\n"
                                         "printf '%s\\n' \"${FAKE_JOB:-}\"\nexit \"${FAKE_TOOL_RC:-0}\"\n")
    return {**env, "DOOR_VENV_PYTHON": str(tmp_path / "job-python"), "DOOR_CONTROL_PLANE_REPO": str(repository),
            "DOOR_SERVICE_USER": "blueprint", "DOOR_TEST_COMPANY_REQUEST": json.dumps(ANSWER)}


@pytest.mark.parametrize(("rc", "status", "code"), [(0, "acted", None), (2, "refused", "test_company_refused"),
                                                    (1, "failed", "test_company_failed")])
def test_the_door_script_runs_the_release_module_as_the_service_user(company_env, rc, status, code):
    answer = {"door_outcome": status}
    done, outcome, calls = _run("door-test-company.sh", company_env, DOOR_REQUEST_ID=THIS,
                                FAKE_JOB=json.dumps(answer), FAKE_TOOL_RC=str(rc))
    assert (done, outcome["status"], outcome["code"]) == (rc, status, code)
    assert calls[0] == "setpriv --reuid=blueprint --regid=blueprint --init-groups --inh-caps=-all --"
    assert calls[2].startswith("job -m blueprint_pipeline.internal_test_company_door --request-json ")
    assert json.loads(Path(outcome["result"]).read_text()) == answer


def test_the_door_script_says_when_the_release_lacks_the_module(company_env, tmp_path):
    (tmp_path / "repository" / "src" / "blueprint_pipeline" / "internal_test_company_door.py").unlink()
    done, outcome, _calls = _run("door-test-company.sh", company_env, DOOR_REQUEST_ID=THIS)
    assert (done, outcome["status"], outcome["code"]) == (2, "refused", "test_company_release_lacks_module")


def _client():
    spec = importlib.util.spec_from_file_location("operator_door_client_test_company", ROOT / "scripts" / "operator_door.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_client_sends_one_action(monkeypatch):
    client, sent = _client(), []
    monkeypatch.setattr(client, "_submit", lambda body, args: sent.append(body) or 0)
    for argv in (["test-company", "ensure"], ["test-company", "send", JOB], ["test-company", "site-link", JOB],
                 ["test-company", "answer", JOB, "--state", "interested", "--inputs", ANSWER["inputs"]]):
        assert client.run(client.build_parser().parse_args(argv)) == 0
    assert sent == [{"kind": "test-company", "action": "ensure"}, {"kind": "test-company", "action": "send", "job_id": JOB},
                    {"kind": "test-company", "action": "site-link", "job_id": JOB}, ANSWER]
    with pytest.raises(SystemExit, match="name the test job"):
        client.run(client.build_parser().parse_args(["test-company", "send"]))
    with pytest.raises(SystemExit, match="--state is required"):
        client.run(client.build_parser().parse_args(["test-company", "answer", JOB]))
    assert client._terminal({"request": ANSWER, "outcome": {"status": "acted"}}) is True
