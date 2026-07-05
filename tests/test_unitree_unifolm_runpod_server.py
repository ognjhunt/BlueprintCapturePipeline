import json
from pathlib import Path

from blueprint_pipeline import unitree_unifolm_runpod_server as server

import pytest

pytestmark = pytest.mark.slow


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object], status: int = 200) -> None:
        self._payload = payload
        self.status = status

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_unitree_unifolm_runpod_server_launch_blocks_without_gates(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("BLUEPRINT_ALLOW_RUNPOD_API_CALLS", raising=False)
    monkeypatch.delenv("BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH", raising=False)
    monkeypatch.setattr(server, "_read_runpod_api_key", lambda: ("", {"configured": False}))
    monkeypatch.setattr(
        server,
        "_read_model_secret_env",
        lambda: ({}, {"status": "not_configured", "raw_secret_values_recorded": False}),
    )

    manifest = server.launch_unitree_unifolm_runpod_server(
        job_dir=tmp_path / "job",
        allow_paid_runpod_launch=False,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "paid_runpod_launch_not_authorized_by_runner_flag" in manifest["blockers"]
    assert "missing_runpod_api_key_or_file" in manifest["blockers"]
    assert manifest["raw_secret_values_recorded"] is False
    assert (
        tmp_path / "job" / "unitree_unifolm_runpod_server_launch_manifest.json"
    ).is_file()


def test_unitree_unifolm_runpod_server_launch_writes_proxy_url_without_secret_values(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH", "true")
    monkeypatch.setattr(
        server,
        "_read_runpod_api_key",
        lambda: (
            "runtime-runpod-secret",
            {"api_key_configured": True, "raw_secret_values_recorded": False},
        ),
    )
    monkeypatch.setattr(
        server,
        "_read_model_secret_env",
        lambda: (
            {"HF_TOKEN": "runtime-hf-secret", "HUGGING_FACE_HUB_TOKEN": "runtime-hf-secret"},
            {
                "status": "configured",
                "env_keys_forwarded": ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"],
                "raw_secret_values_recorded": False,
            },
        ),
    )

    def fake_request(**kwargs):
        captured["request"] = kwargs
        return 201, {"id": "podabc123"}

    monkeypatch.setattr(server, "_runpod_request", fake_request)

    manifest = server.launch_unitree_unifolm_runpod_server(
        job_dir=tmp_path / "job",
        image_name="docker.io/nijelhunt/blueprint-unitree-unifolm:test",
        gpu_type_ids=["NVIDIA GeForce RTX 4090"],
        max_spend_usd=0.75,
        allow_paid_runpod_launch=True,
        generated_at="now",
    )

    assert manifest["status"] == "pod_created"
    assert manifest["server_url"] == "https://podabc123-8777.proxy.runpod.net/act"
    assert manifest["prelaunch_spend_guard"]["can_launch"] is True
    assert manifest["prelaunch_spend_guard"]["requested_budget_usd"] == 0.75
    assert manifest["redacted_pod_payload"]["ports"] == ["8777/http"]
    assert "HF_TOKEN" in manifest["redacted_pod_payload"]["env_keys"]
    assert "runtime-hf-secret" not in json.dumps(manifest)
    assert "runtime-runpod-secret" not in json.dumps(manifest)
    assert (tmp_path / "job" / "unitree_unifolm_runpod_server_state.json").is_file()

    payload = captured["request"]["payload"]  # type: ignore[index]
    assert payload["ports"] == ["8777/http"]  # type: ignore[index]
    assert "blueprint_unitree_unifolm_proxy.py" in payload["dockerStartCmd"][0]  # type: ignore[index]
    assert "run_unitree_unifolm_vla_server" in payload["dockerStartCmd"][0]  # type: ignore[index]
    assert payload["env"]["BLUEPRINT_UNITREE_UNIFOLM_VLA_CHECKPOINT"] == (  # type: ignore[index]
        "unitreerobotics/UnifoLM-VLA-Base"
    )


def test_unitree_unifolm_runpod_server_launch_blocks_before_post_without_budget(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH", "true")
    monkeypatch.delenv(server.RUNPOD_UNIFOLM_MAX_SPEND_USD_ENV, raising=False)
    monkeypatch.setattr(
        server,
        "_read_runpod_api_key",
        lambda: (
            "runtime-runpod-secret",
            {"api_key_configured": True, "raw_secret_values_recorded": False},
        ),
    )
    monkeypatch.setattr(
        server,
        "_read_model_secret_env",
        lambda: ({}, {"status": "not_configured", "raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(
        server,
        "_runpod_request",
        lambda **kwargs: calls.append(dict(kwargs)) or (201, {"id": "podabc123"}),
    )

    manifest = server.launch_unitree_unifolm_runpod_server(
        job_dir=tmp_path / "job",
        image_name="docker.io/nijelhunt/blueprint-unitree-unifolm:test",
        gpu_type_ids=["NVIDIA GeForce RTX 4090"],
        allow_paid_runpod_launch=True,
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "unitree_unifolm_runpod_prelaunch_spend_guard_not_passed" in manifest["blockers"]
    assert "unitree_unifolm_runpod_max_spend_usd_missing" in manifest["blockers"]
    assert manifest["prelaunch_spend_guard"]["required_before_provider_launch"] is True
    assert manifest["prelaunch_spend_guard"]["can_launch"] is False
    assert calls == []


def test_unitree_unifolm_runpod_server_launch_accepts_budget_from_env(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_API_CALLS", "true")
    monkeypatch.setenv("BLUEPRINT_ALLOW_RUNPOD_POD_LAUNCH", "true")
    monkeypatch.setenv(server.RUNPOD_UNIFOLM_MAX_SPEND_USD_ENV, "0.5")
    monkeypatch.setattr(
        server,
        "_read_runpod_api_key",
        lambda: (
            "runtime-runpod-secret",
            {"api_key_configured": True, "raw_secret_values_recorded": False},
        ),
    )
    monkeypatch.setattr(
        server,
        "_read_model_secret_env",
        lambda: ({}, {"status": "not_configured", "raw_secret_values_recorded": False}),
    )
    monkeypatch.setattr(server, "_runpod_request", lambda **_kwargs: (201, {"id": "podabc123"}))

    manifest = server.launch_unitree_unifolm_runpod_server(
        job_dir=tmp_path / "job",
        image_name="docker.io/nijelhunt/blueprint-unitree-unifolm:test",
        gpu_type_ids=["NVIDIA GeForce RTX 4090"],
        allow_paid_runpod_launch=True,
        generated_at="now",
    )

    assert manifest["status"] == "pod_created"
    assert manifest["prelaunch_spend_guard"]["budget_source"] == "env"
    assert manifest["prelaunch_spend_guard"]["requested_budget_usd"] == 0.5


def test_unitree_unifolm_runpod_server_delete_uses_state(
    tmp_path: Path, monkeypatch
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "unitree_unifolm_runpod_server_state.json").write_text(
        json.dumps({"pod_id": "podabc123"}), encoding="utf-8"
    )
    monkeypatch.setattr(
        server,
        "_read_runpod_api_key",
        lambda: (
            "runtime-runpod-secret",
            {"api_key_configured": True, "raw_secret_values_recorded": False},
        ),
    )
    monkeypatch.setattr(
        server,
        "_delete_pod",
        lambda **kwargs: {
            "status": "completed",
            "pod_id": kwargs["pod_id"],
            "raw_secret_values_recorded": False,
        },
    )

    manifest = server.delete_unitree_unifolm_runpod_server(job_dir=job, generated_at="now")

    assert manifest["status"] == "completed"
    assert manifest["pod_id"] == "podabc123"
    assert (
        job / "unitree_unifolm_runpod_server_delete_manifest.json"
    ).is_file()


def test_unitree_unifolm_runpod_server_probe_reports_running_backend(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "unitree_unifolm_runpod_server_state.json").write_text(
        json.dumps({"server_url": "https://podabc123-8777.proxy.runpod.net/act"}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return _FakeHTTPResponse(
            {
                "schema_version": "unitree_unifolm_runpod_status_proxy.v1",
                "backend_process_running": True,
                "backend_start_error": "",
            }
        )

    monkeypatch.setattr(server.urllib.request, "urlopen", fake_urlopen)

    manifest = server.probe_unitree_unifolm_runpod_server(
        job_dir=job,
        timeout_seconds=3,
        generated_at="now",
    )

    assert manifest["status"] == "running"
    assert manifest["blockers"] == []
    assert manifest["status_url"] == "https://podabc123-8777.proxy.runpod.net/status"
    assert captured["timeout"] == 3
    assert (
        job / "unitree_unifolm_runpod_server_proxy_probe.json"
    ).is_file()


def test_unitree_unifolm_runpod_server_probe_reports_backend_start_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = tmp_path / "job"
    job.mkdir()

    def fake_urlopen(_request, timeout):  # type: ignore[no-untyped-def]
        return _FakeHTTPResponse(
            {
                "schema_version": "unitree_unifolm_runpod_status_proxy.v1",
                "backend_process_running": False,
                "backend_start_error": "FileNotFoundError: run_unitree_unifolm_vla_server",
            }
        )

    monkeypatch.setattr(server.urllib.request, "urlopen", fake_urlopen)

    manifest = server.probe_unitree_unifolm_runpod_server(
        job_dir=job,
        server_url="https://podabc123-8777.proxy.runpod.net/act",
        generated_at="now",
    )

    assert manifest["status"] == "blocked"
    assert "blocked_unitree_unifolm_backend_start_error" in manifest["blockers"]
    assert manifest["backend_start_error_present"] is True
