from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import adp_content_agents_vast as content_agents
from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.common import write_json
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


def _allocator_args(tmp_path: Path, receipt: Path, *, execute: bool) -> list[str]:
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
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"immutable-bundle")
    receipt = tmp_path / "receipt.json"
    write_json(
        receipt,
        {
            "status": "ready",
            "source_commit": content_agents.SOURCE_COMMIT,
            "source_tree": content_agents.SOURCE_TREE,
            "container_image": content_agents.DEFAULT_IMAGE,
            "retry_cap": 0,
            "blockers": [],
            "bundle_path": str(bundle),
            "bundle_sha256": "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest(),
        },
    )
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
    assert allocator.main(_allocator_args(tmp_path, receipt, execute=execute)) == 0
    assert observed["execute"] is execute
    assert isinstance(observed["paid_resource_admission_grant"], PaidResourceAdmissionGrant) is (
        execute
    )
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["retry_cap"] == 0
    assert admission["private_or_licensed_dataset_bytes_uploaded"] is False
    assert admission["hard_cap_usd"] == 2.0


def test_canonical_allocator_rejects_mutated_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"before")
    receipt = tmp_path / "receipt.json"
    write_json(
        receipt,
        {
            "status": "ready",
            "source_commit": content_agents.SOURCE_COMMIT,
            "source_tree": content_agents.SOURCE_TREE,
            "container_image": content_agents.DEFAULT_IMAGE,
            "retry_cap": 0,
            "blockers": [],
            "bundle_path": str(bundle),
            "bundle_sha256": "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest(),
        },
    )
    bundle.write_bytes(b"after")
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    assert allocator.main(_allocator_args(tmp_path, receipt, execute=False)) == 2
    result = json.loads((tmp_path / "adapter.json").read_text())
    assert "adp_content_agents_bundle_binding_invalid" in result["blockers"]
