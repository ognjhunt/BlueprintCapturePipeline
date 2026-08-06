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
                info = zipfile.ZipInfo("LICENSE", date_time=(1980, 1, 1, 0, 0, 0))
                archive.writestr(info, "Apache-2.0\n")
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


def _write_receipt(path: Path, payload: dict) -> dict:
    payload = dict(payload)
    payload["receipt_digest"] = canonical_digest(payload, digest_field="receipt_digest")
    write_json(path, payload)
    return payload


def _match_v2_evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    usd = repo / "docs/arm_decision_proof_v1/assets/match-v2.usda"
    snapshot = evidence / "simready/cad_match_v2/snapshot.png"
    usd.parent.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    snapshot.write_bytes(b"\x89PNG\r\n\x1a\nmatch-v2")
    control = _write_receipt(
        repo / content_agents.MATCH_V2_RECEIPT_RELATIVE_PATH,
        {
            "control_id": "adp009a-840313-canned-beverage-multiview-match-v2",
            "status": "prepared_for_independent_validation",
            "checks": {
                "cad_inspection_passed": True,
                "target_dimensions_derived_not_caller_asserted": True,
            },
            "visual_match_evidence": {"projected_scale_and_pose_gate_passed": True},
            "usd": {
                "relative_path": usd.relative_to(repo).as_posix(),
                "sha256": "sha256:" + hashlib.sha256(usd.read_bytes()).hexdigest(),
            },
            "cad_evidence": {
                "snapshot": {
                    "relative_path": snapshot.relative_to(evidence).as_posix(),
                    "sha256": "sha256:"
                    + hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                }
            },
        },
    )
    replacement = _write_receipt(
        repo / content_agents.MATCH_V2_REPLACEMENT_RECEIPT_RELATIVE_PATH,
        {
            "status": "composed_static_candidate",
            "bindings": {"simready_control_receipt_digest": control["receipt_digest"]},
        },
    )
    _write_receipt(
        repo / content_agents.MATCH_V2_HUMAN_REVIEW_RELATIVE_PATH,
        {
            "status": "human_accepted_for_native_validation",
            "technical_admission": False,
            "artifact_chain": {
                "replacement_receipt_digest": replacement["receipt_digest"]
            },
        },
    )
    return repo, evidence, snapshot


def test_match_v2_variant_binds_approved_receipt_chain(tmp_path: Path) -> None:
    repo, evidence, snapshot = _match_v2_evidence(tmp_path)

    resolved = content_agents._resolve_input_variant(
        repo=repo,
        evidence_root=evidence,
        reference_source=snapshot,
        variant="match_v2",
    )

    assert resolved["variant"] == "match_v2"
    assert resolved["control_receipt_digest"].startswith("sha256:")
    assert resolved["replacement_receipt_digest"].startswith("sha256:")
    assert resolved["human_review_receipt_digest"].startswith("sha256:")
    assert "interiorgs_dataset_bytes" in resolved["reference_image_authority"]


def test_match_v2_variant_rejects_changed_approved_snapshot(tmp_path: Path) -> None:
    repo, evidence, snapshot = _match_v2_evidence(tmp_path)
    snapshot.write_bytes(snapshot.read_bytes() + b"changed")

    with pytest.raises(ValueError, match="source_identity_mismatch"):
        content_agents._resolve_input_variant(
            repo=repo,
            evidence_root=evidence,
            reference_source=snapshot,
            variant="match_v2",
        )


def test_match_v2_configs_remove_unproven_aluminum_assertion(tmp_path: Path) -> None:
    assets = ROOT / "docs" / "arm_decision_proof_v1" / "assets"
    sources = {
        "material_agent.yaml": assets / "adp009a_content_agents_material.vast.yaml",
        "texture_agent.yaml": assets / "adp009a_content_agents_texture.vast.yaml",
        "physics_agent.yaml": assets / "adp009a_content_agents_physics.vast.yaml",
    }
    destination = tmp_path / "configs"
    destination.mkdir()

    hashes = content_agents._materialize_remote_configs(
        config_sources=sources, destination=destination, variant="match_v2"
    )

    material = (destination / "material_agent.yaml").read_text(encoding="utf-8")
    texture = (destination / "texture_agent.yaml").read_text(encoding="utf-8")
    assert set(hashes) == set(sources)
    assert "Do not assume aluminum" in material
    assert "bright green aluminum" not in texture
    assert "pale mint green" in texture


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
    assert first["input_usd_normalization"]["default_purpose_bbox_nonempty"] is True
    assert (
        first["input_usd_normalization"]["source_input_usd_sha256"]
        != first["input_usd_sha256"]
    )
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


@pytest.mark.parametrize(
    ("filename", "before", "after"),
    [
        ("material_agent.yaml", "on_failure: warn", "on_failure: fail"),
        ("texture_agent.yaml", "model: gpt-image-1", "model: unavailable-image"),
        (
            "physics_agent.yaml",
            "    enabled: false\n    vlm:\n      backend: openai",
            "    enabled: true\n    vlm:\n      backend: openai",
        ),
    ],
)
def test_remote_config_contract_rejects_known_paid_runtime_failure_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    before: str,
    after: str,
) -> None:
    source = _fake_source(tmp_path, monkeypatch)
    assets = ROOT / "docs" / "arm_decision_proof_v1" / "assets"
    config_sources = {}
    for name in ("material_agent.yaml", "texture_agent.yaml", "physics_agent.yaml"):
        source_name = "adp009a_content_agents_" + name.removesuffix("_agent.yaml") + ".vast.yaml"
        destination = tmp_path / name
        value = (assets / source_name).read_text(encoding="utf-8")
        if name == filename:
            assert before in value
            value = value.replace(before, after, 1)
        destination.write_text(value, encoding="utf-8")
        config_sources[name] = destination

    with pytest.raises(ValueError, match="remote_config_contract_invalid"):
        content_agents._validate_remote_configs(
            source=source,
            config_sources=config_sources,
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


def test_provider_runtime_pins_native_dependency_closure_before_agent_execution() -> None:
    runtime = (ROOT / "scripts/run_adp_content_agents_provider_runtime.sh").read_text()
    runner = (ROOT / "scripts/adp_content_agents_provider_runner.py").read_text()

    assert 'NATIVE_OVRTX_ENV="${SOURCE_DIR}/.ovrtx_native_venv"' in runtime
    assert '"ovrtx==0.4.0.346409"' in runtime
    assert '"ovstage==0.1.0.346039"' in runtime
    assert 'm.version("ovrtx") == "0.4.0.346409"' in runtime
    assert 'm.version("ovstage") == "0.1.0.346039"' in runtime
    assert "ovrtx.__version__" not in runtime
    assert "content_agents_native_ovrtx_dependency_closure_failed" in runtime
    assert 'content_agents_source/.ovrtx_native_venv/bin/python' in runner
    assert runner.index("native, native_blockers = _native_probes(") < runner.index(
        'for name in ("material", "texture", "physics"):'
    )
    assert "skipped_after_native_probe_failure" in runner


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
    validation_log = tmp_path / "material-validate.log"
    validation_log.write_text(
        bundle_preflight.MATERIAL_VALIDATE_MARKER, encoding="utf-8"
    )
    executions["material_validate_input"] = {
        "entrypoint": bundle_preflight.ENTRYPOINTS["material"],
        "arguments": [
            "run",
            "/bundle/" + bundle_preflight.CONFIG_MEMBERS["material"],
            "--only",
            "validate_input",
            "--clean",
        ],
        "secret_environment_names_passed_by_name": ["OPENAI_API_KEY"],
        "returncode": 0,
        "required_marker": bundle_preflight.MATERIAL_VALIDATE_MARKER,
        "log_path": str(validation_log.resolve()),
        "log_size_bytes": validation_log.stat().st_size,
        "log_sha256": "sha256:"
        + hashlib.sha256(validation_log.read_bytes()).hexdigest(),
    }
    bbox_log = tmp_path / "bbox.log"
    bbox_log.write_text(bundle_preflight.USD_BBOX_MARKER, encoding="utf-8")
    executions["usd_default_purpose_bbox"] = {
        "entrypoint": "python",
        "arguments": ["-c", bundle_preflight.USD_BBOX_SCRIPT],
        "secret_environment_names_passed_by_name": [],
        "returncode": 0,
        "required_marker": bundle_preflight.USD_BBOX_MARKER,
        "log_path": str(bbox_log.resolve()),
        "log_size_bytes": bbox_log.stat().st_size,
        "log_sha256": "sha256:" + hashlib.sha256(bbox_log.read_bytes()).hexdigest(),
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
        "model_access": {
            "provider": "openai",
            "models": {
                model: {"http_status": 200, "returned_id": model}
                for model in bundle_preflight.REQUIRED_MODELS
            },
            "paid_inference_performed": False,
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


def _allocator_bundle(
    tmp_path: Path, *, content: bytes = b"config", native_probe: bool = False
) -> tuple[Path, dict]:
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
    if native_probe:
        receipt_value["native_probe"] = {
            "sage_collision_sha256": "sha256:" + "c" * 64
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


def test_canonical_allocator_discloses_public_sage_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, receipt_value = _allocator_bundle(tmp_path, native_probe=True)
    preflight = _passing_config_preflight(tmp_path, receipt_value)
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    monkeypatch.setattr(
        allocator,
        "run_content_agents_vast",
        lambda **_kwargs: {"status": "dry_run_ready"},
    )

    assert allocator.main(_allocator_args(tmp_path, receipt, preflight, execute=False)) == 0
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["private_or_licensed_dataset_bytes_uploaded"] is True
    assert admission["private_or_gated_dataset_bytes_uploaded"] is False
    assert admission["public_licensed_sage_collision_bytes_uploaded"] is True
    assert admission["public_licensed_dataset_identity"]["license"] == "CC-BY-NC-4.0"
    assert admission["input_is_blueprint_owned_parametric_control"] is False
    assert admission["input_contains_blueprint_owned_parametric_control"] is True


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
        assert secret not in " ".join(command)
        entrypoint = command[command.index("--entrypoint") + 1]
        if entrypoint == "python":
            return subprocess.CompletedProcess(
                command, 0, bundle_preflight.USD_BBOX_MARKER, ""
            )
        assert kwargs["env"]["OPENAI_API_KEY"] == secret
        if "--only" in command:
            return subprocess.CompletedProcess(
                command, 0, bundle_preflight.MATERIAL_VALIDATE_MARKER, ""
            )
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
    monkeypatch.setattr(
        bundle_preflight,
        "_probe_model_access",
        lambda _secret: {
            "provider": "openai",
            "models": {
                model: {"http_status": 200, "returned_id": model}
                for model in bundle_preflight.REQUIRED_MODELS
            },
            "paid_inference_performed": False,
        },
    )
    evidence = tmp_path / "evidence"

    receipt = bundle_preflight.materialize_bundle_config_preflight(
        bundle_receipt_path=bundle_receipt,
        evidence_dir=evidence,
        generated_at="fixed",
    )

    assert len(observed_commands) == 9
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
    monkeypatch.setattr(
        bundle_preflight,
        "_probe_model_access",
        lambda _secret: {
            "provider": "openai",
            "models": {
                model: {"http_status": 200, "returned_id": model}
                for model in bundle_preflight.REQUIRED_MODELS
            },
            "paid_inference_performed": False,
        },
    )
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
