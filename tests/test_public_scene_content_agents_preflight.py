from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.public_scene_content_agents_preflight import (
    ContentAgentsPreflightError,
    materialize_content_agents_preflight,
)


COMMIT = "1" * 40
TREE = "2" * 40
IMAGE_DIGEST = "sha256:" + "3" * 64


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    source = tmp_path / "usd-content-agents"
    output = repo / "receipt.json"
    (repo / "assets").mkdir(parents=True)
    (repo / "assets" / "container.Dockerfile").write_text("FROM test\n", encoding="utf-8")
    (source / "LICENSE").parent.mkdir(parents=True)
    (source / "LICENSE").write_text("Apache-2.0", encoding="utf-8")
    skills = []
    for name in ("material", "texture", "physics", "validation"):
        relative = f".agents/skills/{name}/SKILL.md"
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {name}\n", encoding="utf-8")
        skills.append(relative)

    content = data / "content"
    configs: dict[str, str] = {}
    for name in ("material", "texture", "physics"):
        relative = f"content/{name}.yaml"
        path = data / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"project:\n  name: {name}\n", encoding="utf-8")
        configs[name] = relative
    (content / "native.log").write_text(
        "usd-exchange==2.3.0 has no matching macosx platform wheel",
        encoding="utf-8",
    )
    (content / "freeze.txt").write_text("usd-exchange==2.3.0\n", encoding="utf-8")
    (content / "stage.usda").write_text("#usda 1.0\n", encoding="utf-8")
    stage_receipt = {"schema_version": "test", "receipt_digest": ""}
    stage_receipt["receipt_digest"] = canonical_digest(
        stage_receipt, digest_field="receipt_digest"
    )
    _write_json(content / "stage.receipt.json", stage_receipt)

    request = {
        "schema_version": "adp009a_usd_content_agents_preflight_request.v1",
        "source": {
            "repository": "https://example.test/usd-content-agents",
            "commit": COMMIT,
            "tree": TREE,
            "version": "0.5.2",
            "license": "Apache-2.0",
            "license_path": "LICENSE",
            "skill_paths": skills,
        },
        "image": {
            "reference": "example/content-agents:0.5.2",
            "digest": IMAGE_DIGEST,
            "platform": "linux/arm64",
        },
        "dockerfile_path": "assets/container.Dockerfile",
        "native_install_failure_log": "content/native.log",
        "installed_packages_path": "content/freeze.txt",
        "agents": {
            name: {
                "config_path": configs[name],
                "dry_run_log_path": f"content/{name}.log",
                "smallest_blocker": f"{name}_execution_missing",
            }
            for name in ("material", "texture", "physics")
        },
    }
    request["agents"]["validation"] = {
        "stage_path": "content/stage.usda",
        "stage_receipt_path": "content/stage.receipt.json",
        "output_dir": "content/validation-output",
        "log_path": "content/validation.log",
        "task": "Validate static physics authoring only.",
    }
    request_path = repo / "request.json"
    _write_json(request_path, request)
    return {
        "repo": repo,
        "data": data,
        "source": source,
        "request": request_path,
        "output": output,
    }


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
    *, material_marker: str = "Dry run complete",
) -> None:
    def fake_git(_repo: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return COMMIT
        if args == ("rev-parse", "HEAD^{tree}"):
            return TREE
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    def completed(command: list[str], stdout: str = "", returncode: int = 0):
        return subprocess.CompletedProcess(command, returncode, stdout, "")

    def fake_run(command):
        command = list(command)
        if command[1:3] == ["image", "inspect"]:
            return completed(
                command,
                json.dumps(
                    [
                        {
                            "RepoDigests": [f"example/content-agents@{IMAGE_DIGEST}"],
                            "Id": "sha256:" + "4" * 64,
                            "Os": "linux",
                            "Architecture": "arm64",
                        }
                    ]
                ),
            )
        entrypoint = command[command.index("--entrypoint") + 1]
        if command[-1] == "--version":
            return completed(command, f"{entrypoint} version 0.5.2\n")
        if entrypoint == "material-agent":
            return completed(command, material_marker)
        if entrypoint == "texture-agent":
            return completed(command, "Dry run -- execution plan")
        if entrypoint == "physics-agent":
            return completed(command, "Dry run complete")
        if entrypoint == "validation-agent":
            output_dir = paths["data"] / "content" / "validation-output"
            _write_json(output_dir / "validation_request.json", {"task": "physics_sane"})
            _write_json(output_dir / "validation_plan.json", {"template": "physics_sane"})
            _write_json(
                output_dir / "validation_result.json",
                {
                    "verdict": "pass",
                    "template_results": [
                        {
                            "template_name": "physics_sane",
                            "status": "passed",
                            "metrics": {
                                "physics_scene_count": 1,
                                "rigid_body_count": 1,
                                "collider_count": 1,
                                "mass_api_count": 1,
                                "material_api_count": 1,
                            },
                        }
                    ],
                    "metadata": {"dry_run": False},
                },
            )
            return completed(command, '{"verdict":"pass"}\n')
        raise AssertionError(command)

    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_content_agents_preflight._git", fake_git
    )
    monkeypatch.setattr(
        "blueprint_pipeline.public_scene_content_agents_preflight._run", fake_run
    )


def _run(paths: dict[str, Path]) -> dict:
    return materialize_content_agents_preflight(
        request_path=paths["request"],
        repo_root=paths["repo"],
        data_root=paths["data"],
        content_agents_root=paths["source"],
        receipt_output=paths["output"],
    )


def test_preflight_executes_three_dry_runs_and_real_static_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _install_fakes(monkeypatch, paths)

    receipt = _run(paths)

    assert receipt["status"] == "prepared_static_validation_passed"
    assert receipt["runtime"]["paid_resource_allocated"] is False
    assert receipt["agents"]["material"]["dry_run_executed"] is True
    assert receipt["agents"]["physics"]["full_agent_executed"] is False
    assert receipt["agents"]["validation"]["executed"] is True
    assert receipt["agents"]["validation"]["dry_run"] is False
    assert receipt["agents"]["joint"]["applicable"] is False
    assert receipt["receipt_digest"] == canonical_digest(
        receipt, digest_field="receipt_digest"
    )


def test_preflight_rejects_missing_native_dry_run_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _install_fakes(monkeypatch, paths, material_marker="not a completed plan")

    with pytest.raises(ContentAgentsPreflightError, match="dry_run_marker_missing:material"):
        _run(paths)


def test_preflight_rejects_caller_asserted_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    request = json.loads(paths["request"].read_text(encoding="utf-8"))
    request["status"] = "admitted"
    _write_json(paths["request"], request)
    _install_fakes(monkeypatch, paths)

    with pytest.raises(ContentAgentsPreflightError, match="caller_asserted_status_forbidden"):
        _run(paths)


def test_the_model_secret_is_resolved_the_way_the_deployment_installs_it(
    tmp_path, monkeypatch
) -> None:
    """Control-plane units run as `blueprint` with ProtectHome=true and home
    `/nonexistent`, so resolving only the environment and `~/.blueprint-secrets`
    meant the paid preflight could never prove model access on the deployed
    host -- the same defect PR #449 fixed for provider keys."""

    from blueprint_pipeline import adp_content_agents_bundle_preflight as preflight
    from blueprint_pipeline.gpu_render_providers import PROVIDER_SECRETS_DIR_ENV

    for name in preflight.SECRET_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    secrets_dir = tmp_path / "provider-secrets"
    secrets_dir.mkdir()
    (secrets_dir / "openai_api_key").write_text("k-from-the-deployment\n", encoding="utf-8")
    monkeypatch.setenv(PROVIDER_SECRETS_DIR_ENV, str(secrets_dir))

    assert preflight._secret() == "k-from-the-deployment"


def test_an_explicit_secret_file_override_wins(tmp_path, monkeypatch) -> None:
    from blueprint_pipeline import adp_content_agents_bundle_preflight as preflight

    for name in preflight.SECRET_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    explicit = tmp_path / "elsewhere.key"
    explicit.write_text("k-explicit\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY_FILE", str(explicit))

    assert preflight._secret() == "k-explicit"


def test_no_secret_anywhere_resolves_empty_rather_than_raising(monkeypatch) -> None:
    """An absent secret is a blocked preflight, not a crash."""

    from blueprint_pipeline import adp_content_agents_bundle_preflight as preflight
    from blueprint_pipeline.gpu_render_providers import PROVIDER_SECRETS_DIR_ENV

    for name in preflight.SECRET_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENAI_API_KEY_FILE", raising=False)
    monkeypatch.setenv(PROVIDER_SECRETS_DIR_ENV, "/nonexistent-secrets-dir")

    assert preflight._secret() == ""
