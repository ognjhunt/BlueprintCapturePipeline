from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import adp_content_agents_vast as content_agents
from blueprint_pipeline import adp_content_agents_bundle_preflight as bundle_preflight
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_env,
    _probe_shell_script,
    _resolve_launch_mode,
)
from blueprint_pipeline.wam_provider_output import inspect_provider_runtime_output_zip


ROOT = Path(__file__).resolve().parents[1]


def _fake_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "content-agents"
    source.mkdir()
    material_library = (
        source
        / "apps/material_agent/data/materials/material_libs_default/materials.yaml"
    )
    material_library.parent.mkdir(parents=True)
    material_library.write_text("materials: []\n", encoding="utf-8")

    def fake_git(_repo: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return content_agents.SOURCE_COMMIT
        if args == ("rev-parse", "HEAD^{tree}"):
            return content_agents.SOURCE_TREE
        if args == ("status", "--porcelain"):
            return ""
        raise AssertionError(args)

    original_run = content_agents.subprocess.run

    def fake_run(command, **kwargs):
        if command[:4] == ["git", "-C", str(source), "archive"]:
            output = next(item.removeprefix("--output=") for item in command if item.startswith("--output="))
            with zipfile.ZipFile(output, "w") as archive:
                archive.writestr("LICENSE", "Apache-2.0\n")
            return None
        return original_run(command, **kwargs)

    monkeypatch.setattr(content_agents, "_git", fake_git)
    monkeypatch.setattr(content_agents.subprocess, "run", fake_run)
    return source


def _reference_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "reference.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nblueprint-owned-test")
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(content_agents, "REFERENCE_IMAGE_SHA256", digest)
    return path


def test_bundle_is_deterministic_and_provider_preflight_accepts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    reference = _reference_image(tmp_path, monkeypatch)
    first = content_agents.build_content_agents_vast_bundle(
        repo_root=ROOT,
        content_agents_root=source,
        reference_image_path=reference,
        job_dir=tmp_path / "first",
        generated_at="fixed",
    )
    second = content_agents.build_content_agents_vast_bundle(
        repo_root=ROOT,
        content_agents_root=source,
        reference_image_path=reference,
        job_dir=tmp_path / "second",
        generated_at="fixed",
    )

    assert first["blockers"] == []
    assert first["status"] == "ready"
    assert first["bundle_sha256"] == second["bundle_sha256"]
    assert first["container_image"] == content_agents.DEFAULT_IMAGE
    assert first["remote_config_contract_validated"] is True
    assert first["retry_cap"] == 0
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_content_agents",
        bundle_path=Path(first["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?signature=redacted",
        provider_output_put_url="https://example.com/output.zip?signature=redacted",
    )
    assert preflight["status"] == "passed"
    assert preflight["blockers"] == []


def test_bundle_rejects_changed_reference_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    reference = _reference_image(tmp_path, monkeypatch)
    reference.write_bytes(reference.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="reference_image_identity_mismatch"):
        content_agents.build_content_agents_vast_bundle(
            repo_root=ROOT,
            content_agents_root=source,
            reference_image_path=reference,
            job_dir=tmp_path / "job",
        )


def test_vast_adapter_uses_gpu_rendering_and_bounded_bundle_path(tmp_path: Path) -> None:
    assert (
        _resolve_launch_mode(
            requested="auto",
            enable_isaac_smoke=False,
            enable_blueprint_bundle=True,
            provider_bundle_kind="adp_content_agents",
        )
        == "ssh_direct"
    )
    env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_content_agents",
        forward_hf_token=False,
    )
    assert env["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert "ACCEPT_EULA" not in env
    script = _probe_shell_script(
        "https://example.com",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_content_agents",
    )
    assert "run_adp_content_agents_provider_runtime.sh" in script
    assert "adp_content_agents_provider_runtime_output.zip" in script
    assert "apt-get install" in script


def test_provider_output_inspector_recognizes_content_agents_result(tmp_path: Path) -> None:
    output = tmp_path / "output.zip"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "adp_content_agents_vast_result.json",
            json.dumps({"status": "blocked", "blockers": ["typed_runtime_failure"]}),
        )

    inspection = inspect_provider_runtime_output_zip(output)

    assert inspection["runtime_result_present"] is True
    assert inspection["runtime_result_status"] == "blocked"
    assert inspection["runtime_result_blockers"] == ["typed_runtime_failure"]


def _passing_config_preflight(tmp_path: Path, bundle_receipt: dict) -> Path:
    executions = {}
    for name in ("material", "texture", "physics"):
        log_path = tmp_path / f"{name}.log"
        log_path.write_text(bundle_preflight.DRY_RUN_MARKERS[name], encoding="utf-8")
        executions[name] = {
            "entrypoint": bundle_preflight.ENTRYPOINTS[name],
            "arguments": [
                "run",
                "/bundle/" + bundle_preflight.CONFIG_MEMBERS[name],
                "--dry-run",
            ],
            "secret_environment_names_passed_by_name": list(
                bundle_preflight.SECRET_ENV_NAMES
            ),
            "returncode": 0,
            "required_marker": bundle_preflight.DRY_RUN_MARKERS[name],
            "log_path": str(log_path.resolve()),
            "log_size_bytes": log_path.stat().st_size,
            "log_sha256": "sha256:"
            + hashlib.sha256(log_path.read_bytes()).hexdigest(),
        }
    bundle_receipt_path = tmp_path / "receipt.json"
    receipt = {
        "schema_version": bundle_preflight.SCHEMA_VERSION,
        "generated_at": "fixed",
        "generated_by": "blueprint_pipeline.adp_content_agents_bundle_preflight",
        "orchestrator_source_identity": {
            "commit": "a" * 40,
            "tree": "b" * 40,
            "checkout_clean": True,
        },
        "status": "passed",
        "bundle_receipt_path": str(bundle_receipt_path),
        "bundle_receipt_sha256": "sha256:"
        + hashlib.sha256(bundle_receipt_path.read_bytes()).hexdigest(),
        "bundle_path": bundle_receipt["bundle_path"],
        "bundle_sha256": bundle_receipt["bundle_sha256"],
        "content_agents_source_commit": content_agents.SOURCE_COMMIT,
        "content_agents_source_tree": content_agents.SOURCE_TREE,
        "local_container_image": {
            "reference": bundle_preflight.LOCAL_IMAGE,
            "id": bundle_preflight.LOCAL_IMAGE_ID,
            "platform": bundle_preflight.LOCAL_IMAGE_PLATFORM,
        },
        "configs": bundle_preflight._bundle_config_records(
            Path(bundle_receipt["bundle_path"])
        ),
        "executions": executions,
        "all_required_dry_runs_executed": True,
        "provider_mutations_performed": 0,
        "paid_resource_allocated": False,
        "raw_secret_values_recorded": False,
        "blockers": [],
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    path = tmp_path / "config-preflight.json"
    write_json(path, receipt)
    return path


def _allocator_bundle(tmp_path: Path, *, content: bytes = b"config") -> tuple[Path, dict]:
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        for member in bundle_preflight.CONFIG_MEMBERS.values():
            archive.writestr(member, content)
    receipt_value = {
        "status": "ready",
        "source_commit": content_agents.SOURCE_COMMIT,
        "source_tree": content_agents.SOURCE_TREE,
        "container_image": content_agents.DEFAULT_IMAGE,
        "retry_cap": 0,
        "blockers": [],
        "bundle_path": str(bundle),
        "bundle_sha256": "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest(),
    }
    receipt = tmp_path / "receipt.json"
    write_json(receipt, receipt_value)
    return receipt, receipt_value


def _allocator_args(
    tmp_path: Path, receipt: Path, preflight: Path, *, execute: bool
) -> list[str]:
    args = [
        "gpu-canary",
        "--probe-kind",
        content_agents.PROBE_KIND,
        "--provider",
        "vast",
        "--provider-launch-request",
        str(tmp_path / "unused-request.json"),
        "--release-evidence",
        str(tmp_path / "unused-release.json"),
        "--model-cache-evidence",
        str(tmp_path / "unused-model.json"),
        "--preflight-bundle",
        str(tmp_path / "unused-preflight.json"),
        "--admission-out",
        str(tmp_path / "admission.json"),
        "--bound-request-out",
        str(tmp_path / "unused-bound.json"),
        "--adapter-output",
        str(tmp_path / "adapter.json"),
        "--pod-name",
        "adp-content-agents",
        "--adp-content-agents-bundle-receipt",
        str(receipt),
        "--adp-content-agents-config-preflight-receipt",
        str(preflight),
        "--adp-job-dir",
        str(tmp_path / "job"),
        "--adp-max-hourly-rate-usd",
        "0.75",
        "--adp-max-spend-usd",
        "2.00",
        "--adp-hard-ttl-seconds",
        "7200",
    ]
    if execute:
        args.append("--execute")
    return args


@pytest.mark.parametrize("execute", [False, True])
def test_canonical_allocator_issues_grant_only_for_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    receipt, receipt_value = _allocator_bundle(tmp_path)
    preflight = _passing_config_preflight(tmp_path, receipt_value)
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_content_agents_vast", fake_run)
    assert allocator.main(_allocator_args(tmp_path, receipt, preflight, execute=execute)) == 0
    assert observed["execute"] is execute
    assert isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is (
        execute
    )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["retry_cap"] == 0
    assert admission["private_or_licensed_dataset_bytes_uploaded"] is False
    assert admission["hard_cap_usd"] == 2.0
    assert admission["allocation_binding"]["config_preflight_receipt_sha256"]


def test_canonical_allocator_rejects_mutated_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, receipt_value = _allocator_bundle(tmp_path, content=b"before")
    preflight = _passing_config_preflight(tmp_path, receipt_value)
    bundle = Path(receipt_value["bundle_path"])
    bundle.write_bytes(b"after")
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    assert allocator.main(_allocator_args(tmp_path, receipt, preflight, execute=False)) == 2
    result = json.loads((tmp_path / "adapter.json").read_text())
    assert "adp_content_agents_bundle_binding_invalid" in result["blockers"]


def test_canonical_allocator_requires_exact_bundle_config_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, receipt_value = _allocator_bundle(tmp_path)
    preflight = _passing_config_preflight(tmp_path, receipt_value)
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    arguments = _allocator_args(tmp_path, receipt, preflight, execute=False)
    flag = arguments.index("--adp-content-agents-config-preflight-receipt")
    del arguments[flag : flag + 2]

    assert allocator.main(arguments) == 2
    result = json.loads((tmp_path / "adapter.json").read_text())
    assert "adp_content_agents_config_preflight_receipt" in result["blockers"]


def test_canonical_allocator_rejects_changed_preflight_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, receipt_value = _allocator_bundle(tmp_path)
    preflight = _passing_config_preflight(tmp_path, receipt_value)
    (tmp_path / "texture.log").write_text("changed after preflight", encoding="utf-8")
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    assert allocator.main(_allocator_args(tmp_path, receipt, preflight, execute=False)) == 2
    result = json.loads((tmp_path / "adapter.json").read_text())
    assert (
        "adp_content_agents_config_preflight_execution_invalid:texture"
        in result["blockers"]
    )


def _executable_preflight_fixture(tmp_path: Path) -> tuple[Path, str]:
    secret = "unit-test-secret-that-must-not-be-recorded"
    source_buffer = io.BytesIO()
    with zipfile.ZipFile(source_buffer, "w") as archive:
        archive.writestr(
            "apps/material_agent/data/materials/material_libs_default/materials.yaml",
            "materials: []\n",
        )
    bundle = tmp_path / "exact-bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr(
            "provider_runtime/content_agents_source.zip", source_buffer.getvalue()
        )
        for member in bundle_preflight.CONFIG_MEMBERS.values():
            archive.writestr(member, "project:\n  name: exact-bundle-test\n")
    bundle_receipt = tmp_path / "exact-bundle-receipt.json"
    write_json(
        bundle_receipt,
        {
            "status": "ready",
            "source_commit": content_agents.SOURCE_COMMIT,
            "source_tree": content_agents.SOURCE_TREE,
            "bundle_path": str(bundle),
            "bundle_sha256": "sha256:"
            + hashlib.sha256(bundle.read_bytes()).hexdigest(),
        },
    )
    return bundle_receipt, secret


def test_exact_bundle_preflight_executes_all_clis_and_never_records_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_receipt, secret = _executable_preflight_fixture(tmp_path)
    observed_commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        command = list(command)
        observed_commands.append(command)
        if command[0] == "git":
            if command[-2:] == ["rev-parse", "HEAD"]:
                stdout = "c" * 40 + "\n"
            elif command[-2:] == ["rev-parse", "HEAD^{tree}"]:
                stdout = "d" * 40 + "\n"
            elif command[-2:] == ["status", "--porcelain"]:
                stdout = ""
            else:
                raise AssertionError(command)
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if command[1:3] == ["image", "inspect"]:
            stdout = json.dumps(
                [
                    {
                        "Id": bundle_preflight.LOCAL_IMAGE_ID,
                        "Os": "linux",
                        "Architecture": "arm64",
                    }
                ]
            )
            return subprocess.CompletedProcess(command, 0, stdout, "")
        assert kwargs["env"]["GEMINI_API_KEY"] == secret
        assert secret not in " ".join(command)
        entrypoint = command[command.index("--entrypoint") + 1]
        name = next(
            name
            for name, expected in bundle_preflight.ENTRYPOINTS.items()
            if expected == entrypoint
        )
        return subprocess.CompletedProcess(
            command, 0, bundle_preflight.DRY_RUN_MARKERS[name], ""
        )

    monkeypatch.setattr(bundle_preflight.subprocess, "run", fake_run)
    monkeypatch.setattr(bundle_preflight, "_secret", lambda: secret)
    evidence = tmp_path / "evidence"

    receipt = bundle_preflight.materialize_bundle_config_preflight(
        bundle_receipt_path=bundle_receipt,
        evidence_dir=evidence,
        generated_at="fixed",
    )

    assert len(observed_commands) == 7
    assert receipt["status"] == "passed"
    assert receipt["all_required_dry_runs_executed"] is True
    assert bundle_preflight.validate_bundle_config_preflight(
        preflight=receipt,
        prepared_bundle=json.loads(bundle_receipt.read_text()),
        preflight_receipt_path=(
            evidence / "adp_content_agents_bundle_config_preflight.json"
        ),
        expected_orchestrator_source_commit="c" * 40,
    ) == []
    assert secret not in json.dumps(receipt)
    assert secret not in "".join(path.read_text() for path in evidence.glob("*.log"))
    assert not list(tmp_path.glob("adp-content-agents-preflight-*"))


def test_exact_bundle_preflight_fails_before_receipt_when_any_cli_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle_receipt, secret = _executable_preflight_fixture(tmp_path)

    def fake_run(command, **kwargs):
        command = list(command)
        if command[0] == "git":
            if command[-2:] == ["rev-parse", "HEAD"]:
                stdout = "c" * 40 + "\n"
            elif command[-2:] == ["rev-parse", "HEAD^{tree}"]:
                stdout = "d" * 40 + "\n"
            elif command[-2:] == ["status", "--porcelain"]:
                stdout = ""
            else:
                raise AssertionError(command)
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    [
                        {
                            "Id": bundle_preflight.LOCAL_IMAGE_ID,
                            "Os": "linux",
                            "Architecture": "arm64",
                        }
                    ]
                ),
                "",
            )
        entrypoint = command[command.index("--entrypoint") + 1]
        return subprocess.CompletedProcess(
            command,
            1 if entrypoint == "texture-agent" else 0,
            "failed" if entrypoint == "texture-agent" else "Dry run complete",
            "",
        )

    monkeypatch.setattr(bundle_preflight.subprocess, "run", fake_run)
    monkeypatch.setattr(bundle_preflight, "_secret", lambda: secret)
    evidence = tmp_path / "failed-evidence"

    with pytest.raises(
        bundle_preflight.ContentAgentsBundlePreflightError,
        match="dry_run_failed:texture",
    ):
        bundle_preflight.materialize_bundle_config_preflight(
            bundle_receipt_path=bundle_receipt,
            evidence_dir=evidence,
        )

    assert not (evidence / "adp_content_agents_bundle_config_preflight.json").exists()
