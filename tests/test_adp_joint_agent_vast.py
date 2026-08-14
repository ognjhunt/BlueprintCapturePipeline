from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import adp_joint_agent_vast as joint_vast
from blueprint_pipeline.adp_joint_agent_vast import (
    CONSTRUCTION_RENDER_MODE,
    CONSTRUCTION_RENDER_SENSOR_UPDATES,
    DEFAULT_IMAGE,
    PROBE_KIND,
    PROVIDER_BUNDLE_KIND,
    SOURCE_TREE,
)
from blueprint_pipeline.adp_joint_agent_vast import (
    build_joint_agent_vast_bundle,
    recover_joint_agent_local_closeout,
    run_joint_agent_vast,
)
from blueprint_pipeline.common import write_json
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.nvidia_nim_model_preflight import (
    DEFAULT_ENDPOINT as NIM_ENDPOINT,
    DEFAULT_MODEL as NIM_MODEL,
    SCHEMA_VERSION as NIM_SCHEMA_VERSION,
)
from blueprint_pipeline.usd_content_joint_agent_packet import JOINT_AGENT_IDENTITY
from blueprint_pipeline.vast_provider_adapter import (
    _blueprint_bundle_preflight,
    _probe_env,
    _probe_shell_script,
    _resolve_launch_mode,
)


def _provider_runner_module():
    path = Path(__file__).resolve().parents[1] / "scripts/adp_joint_agent_provider_runner.py"
    spec = importlib.util.spec_from_file_location("adp_joint_agent_provider_runner_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_provider_runner_retains_downstream_artifacts_outside_working_tree(
    tmp_path: Path,
) -> None:
    runner = _provider_runner_module()
    work = tmp_path / "provider_runtime/runtime_output/joint_agent_work"
    work.mkdir(parents=True)
    sources = {
        "articulation_candidates": work / "candidates.json",
        "owned_core_rigged_asset": work / "rigged.usdz",
        "owned_core_diagnostics": work / "diagnostics.json",
    }
    for role, path in sources.items():
        path.write_bytes(f"retained-{role}".encode())
    output = tmp_path / "provider_return"

    rows = runner.retain_joint_agent_artifacts(
        output_root=output,
        artifacts=sources,
    )

    assert {row["role"] for row in rows} == set(sources)
    for row in rows:
        retained = output / row["relative_path"]
        assert retained.is_file()
        assert row["size_bytes"] == retained.stat().st_size
        assert row["sha256"] == runner._sha256(retained)


def test_provider_runner_fails_closed_when_required_artifact_is_missing(
    tmp_path: Path,
) -> None:
    runner = _provider_runner_module()

    with pytest.raises(
        ValueError,
        match="joint_agent_retained_artifact_invalid:owned_core_rigged_asset",
    ):
        runner.retain_joint_agent_artifacts(
            output_root=tmp_path / "return",
            artifacts={"owned_core_rigged_asset": tmp_path / "missing.usdz"},
        )


def test_provider_return_requires_every_digest_bound_construction_artifact(
    tmp_path: Path,
) -> None:
    rows = []
    for role in sorted(joint_vast.REQUIRED_RETAINED_ARTIFACT_ROLES):
        path = tmp_path / "retained" / f"{role}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(role.encode())
        rows.append(
            {
                "role": role,
                "relative_path": path.relative_to(tmp_path).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": joint_vast._sha256(path),
            }
        )

    assert joint_vast._retained_execution_artifact_blockers(
        {"retained_artifacts": rows}, extracted_root=tmp_path
    ) == []

    rows.pop()
    assert any(
        blocker.startswith("joint_agent_retained_artifact_missing:")
        for blocker in joint_vast._retained_execution_artifact_blockers(
            {"retained_artifacts": rows}, extracted_root=tmp_path
        )
    )


def _checkout(root: Path) -> Path:
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for relative, digest in JOINT_AGENT_IDENTITY["files"].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "VERSION.md":
            path.write_text("0.5.2", encoding="utf-8")
        else:
            # The builder's checkout inspector is exercised separately against
            # the real locked checkout; this fixture injects its own identity by
            # constructing the exact release files below in the test monkeypatch.
            path.write_text(digest, encoding="utf-8")
    build_projects = {
        "pyproject.toml": ["hatchling<1.27.0", "uv-dynamic-versioning"],
        "apps/joint_agent/pyproject.toml": [
            "hatchling<1.27.0",
            "uv-dynamic-versioning",
        ],
        "apps/ovrtx_rendering_api/pyproject.toml": [
            "setuptools>=68.0",
            "setuptools-scm>=8.0",
        ],
    }
    for relative, requirements in build_projects.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        source = (
            "[build-system]\n"
            f"requires = {json.dumps(requirements)}\n"
            'build-backend = "hatchling.build"\n'
        )
        if relative == "apps/joint_agent/pyproject.toml":
            source += (
                "\n[tool.uv.sources]\n"
                'world-understanding = { path = "../..", editable = true }\n'
            )
        path.write_text(source, encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"],
        cwd=root,
        check=True,
    )
    return root


def test_builder_binds_scene_neutral_joint_runtime(monkeypatch, tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    checkout = _checkout(tmp_path / "checkout")
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\ndef Xform \"asset\" {}\n", encoding="utf-8")
    source_digest = "sha256:" + __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    source_receipt_digest = "sha256:" + "1" * 64
    config = {
        "project": {"name": "fixture", "working_dir": str(tmp_path / "work")},
        "input": {"usd_path": str(source)},
        "steps": {
            "optimize_usd": {"enabled": True},
            "identify_asset": {
                "enabled": True,
                "renderer": {"backend": "remote"},
                "vlm": {"backend": "nim"},
            },
            "analyze_structure": {"enabled": True, "llm": {"backend": "nim"}},
            "build_dataset_usd": {"enabled": True, "renderer": {"backend": "remote"}},
            "build_dataset_prepare_dataset": {"enabled": True},
            "predict": {
                "enabled": True,
                "completion_retries": 0,
                "vlm": {"backend": "nim"},
            },
            "consistency_pass": {"enabled": True},
            "infer_articulation_candidates": {
                "enabled": True,
                "vlm": {"backend": "nim"},
            },
            "apply_joint_rigger": {"enabled": False},
            "author_physics_schemas": {"enabled": False},
        },
    }
    config_path = tmp_path / "joint.yaml"
    import yaml
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    packet = {
        "schema_version": "usd_content_joint_agent_packet.v1",
        "source_asset": {
            "path": str(source),
            "sha256": source_digest,
            "source_receipt_digest": source_receipt_digest,
        },
        "config": {"path": str(config_path)},
        "packet_digest": "",
    }
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    freeze = {
        "scene": {"publisher_scene_id": "840796"},
        "member_geometry_observation": {
            "joint_axis_world": [0.0, 0.0, 1.0],
            "upper_member_vertical_interval_m": [0.94, 1.632],
        },
        "task_spec": {
            "target_joint_id": "upper_hinge",
            "non_task_joint_motion_tolerance_rad": 0.001,
        },
        "freeze_digest": "",
    }
    freeze["freeze_digest"] = canonical_digest(freeze, digest_field="freeze_digest")
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    scope_amendment = {
        "task_family": "one_commanded_joint_in_bounded_multi_joint_articulated_assembly",
        "joint_scope": {
            "minimum_assembly_joint_count": 1,
            "maximum_assembly_joint_count": 4,
            "commanded_task_joint_count": 1,
            "required_articulation_root_count": 1,
            "non_task_joint_mode": "locked_at_frozen_reset_with_native_readback",
            "non_task_joint_motion_tolerance": 0.001,
        },
        "amendment_digest": "",
    }
    scope_amendment["amendment_digest"] = canonical_digest(
        scope_amendment, digest_field="amendment_digest"
    )
    scope_amendment_path = tmp_path / "scope_amendment.json"
    scope_amendment_path.write_text(json.dumps(scope_amendment), encoding="utf-8")
    nim_preflight = {
        "schema_version": NIM_SCHEMA_VERSION,
        "status": "qualified",
        "endpoint": NIM_ENDPOINT,
        "model": NIM_MODEL,
        "http_status": 200,
        "credential_validated": True,
        "required_model_present": True,
        "paid_inference_performed": False,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "blockers": [],
        "receipt_digest": "",
    }
    nim_preflight["receipt_digest"] = canonical_digest(
        nim_preflight, digest_field="receipt_digest"
    )
    nim_preflight_path = tmp_path / "nim_preflight.json"
    nim_preflight_path.write_text(json.dumps(nim_preflight), encoding="utf-8")
    authority = {
        "authorization_digest": "",
        "joint_agent_source_asset_digest": source_digest,
        "joint_agent_source_receipt_digest": source_receipt_digest,
        "publisher_scene_id": "840796",
        "freeze_digest": freeze["freeze_digest"],
    }
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.validate_public_scene_execution_authority",
        lambda value: value,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.inspect_joint_agent_checkout",
        lambda value: {"commit": JOINT_AGENT_IDENTITY["commit"], "version": "0.5.2"},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.SOURCE_TREE",
        subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=checkout, check=True, capture_output=True, text=True
        ).stdout.strip(),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast._blueprint_identity",
        lambda value: {"commit": "a" * 40, "tree": "b" * 40, "dirty": False},
    )
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    optimizer_zip = tmp_path / "scene_optimizer_core.zip"
    with zipfile.ZipFile(optimizer_zip, "w") as archive:
        for subdir in ("python", "lib", "extraLibs", "usdpy"):
            archive.writestr(f"{subdir}/.keep", subdir)
    optimizer_digest = "sha256:" + __import__("hashlib").sha256(
        optimizer_zip.read_bytes()
    ).hexdigest()
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.SCENE_OPTIMIZER_CORE_SHA256",
        optimizer_digest,
    )

    receipt = build_joint_agent_vast_bundle(
        repo_root=repo,
        joint_agent_root=checkout,
        packet_path=packet_path,
        execution_authority_path=authority_path,
        freeze_path=freeze_path,
        scope_amendment_path=scope_amendment_path,
        nim_preflight_path=nim_preflight_path,
        scene_optimizer_core_zip_path=optimizer_zip,
        job_dir=tmp_path / "bundle",
        generated_at="2026-08-08T00:00:00+00:00",
    )

    assert receipt["status"] == "ready"
    assert receipt["scene_optimizer_core"] == {
        "role": "released_code_public_build_resource",
        "package": "scene_optimizer_core_usd_25.11_py_3.12",
        "release": "NVIDIA-Omniverse/usd-optimize v1.0.3",
        "license": "Apache-2.0",
        "sha256": optimizer_digest,
        "size_bytes": optimizer_zip.stat().st_size,
    }
    assert (
        tmp_path / "bundle/provider_runtime/scene_optimizer_core.zip"
    ).is_file()
    assert receipt["provider_bundle_kind"] == "adp_joint_agent"
    assert receipt["runtime_resource_requirements"] == {
        "requested_disk_gb": joint_vast.RUNTIME_DISK_GB,
        "minimum_free_bytes_before_dependency_install": (
            joint_vast.MINIMUM_FREE_BYTES_BEFORE_DEPENDENCY_INSTALL
        ),
        "measurement_path": "provider_runtime",
        "failure_blocker": "joint_agent_runtime_disk_headroom_insufficient",
    }
    assert receipt["python_build_dependency_plan"]["requirements"] == [
        "editables>=0.3",
        "hatchling<1.27.0",
        "setuptools-scm>=8.0",
        "setuptools>=68.0",
        "uv-dynamic-versioning",
    ]
    assert receipt["input_usd_sha256"] == source_digest
    assert receipt["renderer"]["scene_bytes_leave_vast_instance"] is False
    assert receipt["renderer"] == {
        "implementation": "released_code_local_ovrtx_rendering_api",
        "endpoint": "http://127.0.0.1:8001",
        "purpose": "joint_agent_construction_preview_only",
        "render_mode": CONSTRUCTION_RENDER_MODE,
        "num_sensor_updates": CONSTRUCTION_RENDER_SENSOR_UPDATES,
        "profile_basis": "released_ovrtx_in_process_backend_defaults",
        "evaluation_authorized": False,
        "policy_input": False,
        "scene_bytes_leave_vast_instance": False,
    }
    runtime_script = (
        tmp_path / "bundle/provider_runtime/run_adp_joint_agent_provider_runtime.sh"
    ).read_text(encoding="utf-8")
    assert f'export OVRTX_RENDER_MODE="{CONSTRUCTION_RENDER_MODE}"' in runtime_script
    assert (
        f'export OVRTX_NUM_SENSOR_UPDATES="{CONSTRUCTION_RENDER_SENSOR_UPDATES}"'
        in runtime_script
    )
    assert 'export OVRTX_RENDER_MODE="pt"' not in runtime_script
    assert 'export WU_SO_PACKAGE_DIR=' in runtime_script
    assert "joint_agent_scene_optimizer_core_missing" in runtime_script
    assert receipt["blueprint_source"]["commit"] == "a" * 40
    assert receipt["completion_retries"] == 0
    assert receipt["execution_role"] == "optional_construction_enrichment"
    assert receipt["failure_blocks_native_simulator_qualification"] is False
    assert receipt["agent_output_is_simready_authority"] is False
    assert receipt["scope_amendment_digest"] == scope_amendment["amendment_digest"]
    assert receipt["nim_preflight_receipt_digest"] == nim_preflight["receipt_digest"]
    review_contract = json.loads(
        (tmp_path / "bundle/provider_runtime/joint_review_contract.json").read_text()
    )
    assert review_contract["maximum_assembly_joint_count"] == 4
    assert review_contract["commanded_task_joint_count"] == 1
    assert review_contract["scope_amendment_digest"] == scope_amendment["amendment_digest"]
    assert "840313" not in (tmp_path / "bundle/provider_runtime/joint_agent.yaml").read_text()
    preflight = _blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight",
        generated_at="fixed",
        enable_blueprint_bundle=True,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_joint_agent",
        bundle_path=Path(receipt["bundle_path"]),
        provider_bundle_url="https://example.com/bundle.zip?signature=redacted",
        provider_output_put_url="https://example.com/output.zip?signature=redacted",
    )
    assert preflight["status"] == "passed"


def test_joint_agent_provider_uses_gpu_graphics_and_distinct_runtime(tmp_path: Path) -> None:
    assert _resolve_launch_mode(
        requested="auto",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_joint_agent",
    ) == "ssh_direct"
    env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_joint_agent",
        forward_hf_token=False,
    )
    assert env["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert "ACCEPT_EULA" not in env
    script = _probe_shell_script(
        "https://example.com",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_joint_agent",
    )
    assert "run_adp_joint_agent_provider_runtime.sh" in script
    assert "adp_joint_agent_provider_runtime_output.zip" in script
    assert "adp_content_agents_provider_runtime_output.zip" not in script


def test_provider_config_routes_replaceable_hosted_model_backend(
    tmp_path: Path,
) -> None:
    import yaml

    base = {
        "project": {"working_dir": "old"},
        "input": {"usd_path": "old"},
        "steps": {
            "analyze_structure": {"llm": {"backend": "nim", "temperature": 0}},
            "predict": {"vlm": {"backend": "nim"}},
            "infer_articulation_candidates": {"vlm": {"backend": "nim"}},
            # The released BYOA config omits this node and silently injects a
            # NIM default. The harness must materialize it before rewriting.
            "identify_asset": {},
            "build_dataset_usd": {},
            "apply_joint_rigger": {},
            "future_nested_step": {
                "workers": [{"vlm": {"backend": "nim", "custom": "retained"}}]
            },
        },
    }
    path = tmp_path / "joint.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    config = joint_vast._provider_config(
        {"config": {"path": str(path)}},
        model_backend="openai",
        model_id="gpt-5.6-luna",
        model_options={"reasoning_effort": "max"},
    )

    for step, key in (
        ("analyze_structure", "llm"),
        ("predict", "vlm"),
        ("infer_articulation_candidates", "vlm"),
        ("identify_asset", "vlm"),
    ):
        assert config["steps"][step][key]["backend"] == "openai"
        assert config["steps"][step][key]["model"] == "gpt-5.6-luna"
        assert config["steps"][step][key]["reasoning_effort"] == "max"
    assert config["steps"]["analyze_structure"]["llm"]["temperature"] == 0
    assert config["steps"]["future_nested_step"]["workers"][0]["vlm"] == {
        "backend": "openai",
        "model": "gpt-5.6-luna",
        "reasoning_effort": "max",
        "custom": "retained",
    }


def test_joint_model_admission_accepts_capability_proven_model_not_static_default() -> None:
    receipt = {
        "schema_version": "hosted_model_inference_preflight.v2",
        "status": "qualified",
        "backend": "openai",
        "endpoint": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-5.6-luna",
        "credential_validated": True,
        "inference_http_status": 200,
        "choice_count": 1,
        "probe_profile": "multimodal_structured_json.v1",
        "verified_capabilities": ["image_input", "structured_json"],
        "probe_response_validated": True,
        "probe_image": {"uploaded_scene_bytes": False},
        "reasoning_effort": "max",
        "inference_probe_performed": True,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "blockers": [],
    }

    assert joint_vast._admitted_hosted_model(receipt) == (
        "openai",
        "openai",
        "gpt-5.6-luna",
        {"reasoning_effort": "max"},
    )

    receipt["verified_capabilities"] = ["structured_json"]
    with pytest.raises(
        ValueError, match="adp_joint_agent_model_capability_preflight_failed"
    ):
        joint_vast._admitted_hosted_model(receipt)


def _prepared_bundle(tmp_path: Path) -> dict:
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("provider_runtime/fixture", "fixture")
    digest = "sha256:" + __import__("hashlib").sha256(bundle.read_bytes()).hexdigest()
    return {
        "status": "ready",
        "bundle_path": str(bundle),
        "bundle_sha256": digest,
        "exact_bundle_entrypoint_rehearsal": {
            "status": "passed",
            "bundle_sha256": digest,
            "entrypoint_relative_path": (
                "provider_runtime/run_adp_joint_agent_provider_runtime.sh"
            ),
            "returncode": 0,
            "gpu_runtime_started": False,
            "paid_inference_performed": False,
            "provider_mutations_performed": 0,
        },
    }


def _ample_disk(free_bytes: int = 64 * 1024**3):
    """A disk report the test states, rather than whatever the host has.

    The lane refuses below `MINIMUM_LOCAL_EVIDENCE_FREE_BYTES` (2 GiB), and a
    developer machine sitting near that line turns every test below into a
    headroom failure that says nothing about what it was asserting.
    """

    return lambda path: shutil._ntuple_diskusage(free_bytes * 4, free_bytes * 3, free_bytes)


def test_run_dry_run_is_zero_mutation_and_requires_bound_bundle(tmp_path: Path) -> None:
    result = run_joint_agent_vast(
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=None,
        execute=False,
        prepared_bundle=_prepared_bundle(tmp_path),
        disk_usage=_ample_disk(),
    )

    assert result["status"] == "dry_run_ready"
    assert result["provider_mutations_performed"] == 0
    assert result["retry_cap"] == 0
    assert result["local_evidence_capacity"]["status"] == "passed"
    assert result["admitted_live_minutes"] >= 90


def test_run_dry_run_rejects_budget_below_minimum_live_window(
    tmp_path: Path,
) -> None:
    result = run_joint_agent_vast(
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=None,
        execute=False,
        prepared_bundle=_prepared_bundle(tmp_path),
        max_hourly_rate_usd=0.8,
        hard_cap_usd=0.8,
        hard_ttl_seconds=5400,
        disk_usage=_ample_disk(),
    )

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert result["blockers"] == [
        "adp_joint_agent_budget_below_minimum_live_window"
    ]


def test_a_host_without_room_for_the_evidence_still_refuses_the_run(
    tmp_path: Path,
) -> None:
    """Stating the free space must not disable the guard, only pin it.

    This is the case the two tests above used to reach by accident on a full
    machine, which is why it is now asserted on purpose.
    """

    result = run_joint_agent_vast(
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=None,
        execute=False,
        prepared_bundle=_prepared_bundle(tmp_path),
        disk_usage=_ample_disk(free_bytes=1),
    )

    assert result["status"] == "blocked"
    assert result["provider_mutations_performed"] == 0
    assert result["blockers"] == ["adp_joint_agent_local_evidence_headroom_insufficient"]


@pytest.mark.parametrize("execute", [False, True])
@pytest.mark.parametrize("concurrent", [False, True])
@pytest.mark.parametrize("model_backend", ["legacy_nim", "openai"])
def test_canonical_allocator_binds_joint_agent_bundle_and_grant(
    monkeypatch, tmp_path: Path, execute: bool, concurrent: bool, model_backend: str
) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"immutable-joint-agent-runtime")
    bundle_digest = "sha256:" + __import__("hashlib").sha256(bundle.read_bytes()).hexdigest()
    receipt = {
        "status": "ready",
        "provider_bundle_kind": PROVIDER_BUNDLE_KIND,
        "container_image": DEFAULT_IMAGE,
        "source_tree": SOURCE_TREE,
        "blueprint_source": {"commit": "a" * 40, "dirty": False},
        "completion_retries": 0,
        "automatic_paid_retry_allowed": False,
        "provider_zero_required_after_return": True,
        "scope_amendment_digest": "sha256:" + "1" * 64,
        "one_instance_at_a_time": not concurrent,
        "maximum_concurrent_paid_instances": 2 if concurrent else 1,
        "blockers": [],
        "bundle_path": str(bundle),
        "bundle_sha256": bundle_digest,
    }
    if model_backend == "legacy_nim":
        receipt["nim_preflight_receipt_digest"] = "sha256:" + "2" * 64
    else:
        receipt["model_preflight_receipt_digest"] = "sha256:" + "2" * 64
        receipt["model"] = {
            "backend": "openai",
            "id": "gpt-5.6-luna",
            "options": {"reasoning_effort": "xhigh"},
            "capability_preflight": {
                "schema_version": "hosted_model_inference_preflight.v2",
                "probe_profile": "multimodal_structured_json.v1",
                "verified_capabilities": ["image_input", "structured_json"],
                "receipt_digest": "sha256:" + "2" * 64,
            },
        }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: (
            [],
            {"orchestrator_source_commit": "a" * 40, "checkout_clean": True},
        ),
    )
    observed: dict = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_joint_agent_vast", fake_run)
    args = [
        "gpu-canary",
        "--probe-kind",
        PROBE_KIND,
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
        "adp-joint-agent",
        "--expected-source-commit",
        "a" * 40,
        "--adp-joint-agent-bundle-receipt",
        str(receipt_path),
        "--adp-job-dir",
        str(tmp_path / "run"),
        "--adp-max-hourly-rate-usd",
        "1.00",
        "--adp-max-spend-usd",
        "3.00",
        "--adp-hard-ttl-seconds",
        "10800",
    ]
    if execute:
        args.append("--execute")
    if concurrent:
        args.extend(["--adp-allowed-active-vast-instance-id", "47226054"])

    assert allocator.main(args) == 0
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["status"] == "admitted"
    assert admission["allocation_binding"]["bundle_sha256"] == bundle_digest
    assert (
        admission["allocation_binding"]["model_preflight_receipt_digest"]
        == "sha256:" + "2" * 64
    )
    assert admission["raw_interiorgs_downloaded_bytes_uploaded"] is False
    assert admission["explicit_concurrent_gpu_authority_bound"] is concurrent
    assert observed["execute"] is execute
    assert (observed["paid_resource_admission_grant"] is not None) is execute


def test_live_run_arms_watchdog_before_adapter_and_forwards_only_nvidia_key(
    monkeypatch, tmp_path: Path
) -> None:
    events: list[str] = []
    started_path = tmp_path / "started_instance.txt"
    staging = tmp_path / "job/object_store_staging"

    def fake_stage(**kwargs):
        staging.mkdir(parents=True)
        for name in (
            "provider_bundle_url.txt",
            "provider_output_put_url.txt",
            "provider_output_get_url.txt",
        ):
            (staging / name).write_text("https://example.com/private", encoding="utf-8")
        return {"status": "completed"}

    def fake_arm(**kwargs):
        events.append("watchdog")
        assert kwargs["allowed_active_instance_ids"] == ()
        return {"status": "armed"}, SimpleNamespace(started_instance_id_path=started_path)

    def fake_adapter(**kwargs):
        events.append("adapter")
        assert kwargs["started_instance_id_path"] == started_path
        assert kwargs["provider_bundle_kind"] == "adp_joint_agent"
        assert kwargs["paid_resource_admission_grant"] is grant
        assert __import__("os").environ["NVIDIA_API_KEY"] == "fixture-nvidia"
        assert __import__("os").environ["BLUEPRINT_VAST_FORWARD_SECRET_ENV_VARS"] == "NVIDIA_API_KEY"
        output_zip = Path(kwargs["provider_runtime_output_zip"])
        output_zip.parent.mkdir(parents=True)
        retained_rows = []
        retained_payloads = {}
        for role in sorted(joint_vast.REQUIRED_RETAINED_ARTIFACT_ROLES):
            payload = role.encode()
            relative_path = f"retained/{role}.bin"
            retained_payloads[relative_path] = payload
            retained_rows.append(
                {
                    "role": role,
                    "relative_path": relative_path,
                    "size_bytes": len(payload),
                    "sha256": "sha256:"
                    + __import__("hashlib").sha256(payload).hexdigest(),
                }
            )
        with zipfile.ZipFile(output_zip, "w") as archive:
            archive.writestr(
                "adp_joint_agent_result.json",
                json.dumps(
                    {
                        "status": "completed",
                        "blockers": [],
                        "retained_artifacts": retained_rows,
                    }
                ),
            )
            for relative_path, payload in retained_payloads.items():
                archive.writestr(relative_path, payload)
        write_json(
            output_zip.parent / "vast_teardown_manifest.json",
            {"vast_instance_ids": [7], "continuing_spend_from_this_run": False},
        )
        return {"status": "completed", "blockers": [], "estimated_cost_usd": 0.5}

    monkeypatch.setattr("blueprint_pipeline.adp_joint_agent_vast._remaining_minutes", lambda **kwargs: 100)
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast._local_evidence_capacity",
        lambda **kwargs: {"status": "passed", "blockers": []},
    )
    monkeypatch.setattr("blueprint_pipeline.adp_joint_agent_vast.stage_wam_provider_bundle_object_store", fake_stage)
    monkeypatch.setattr("blueprint_pipeline.adp_joint_agent_vast.arm_independent_vast_watchdog", fake_arm)
    monkeypatch.setattr("blueprint_pipeline.adp_joint_agent_vast.run_vast_provider_adapter", fake_adapter)
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.cleanup_staged_wam_provider_objects",
        lambda value: {"all_objects_absent": True},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.close_independent_vast_watchdog",
        lambda **kwargs: {"status": "provider_terminal"},
    )
    monkeypatch.setenv("NVIDIA_API_KEY", "fixture-nvidia")
    grant = object()

    result = run_joint_agent_vast(
        job_dir=tmp_path / "job",
        paid_resource_admission_grant=grant,  # type: ignore[arg-type]
        execute=True,
        prepared_bundle=_prepared_bundle(tmp_path),
    )

    assert events == ["watchdog", "adapter"]
    assert result["status"] == "completed"
    assert result["continuing_spend_from_this_run"] is False
    assert result["retry_cap"] == 0


def test_local_evidence_capacity_blocks_before_paid_mutation(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle.zip"
    bundle.write_bytes(b"bundle")
    usage = SimpleNamespace(total=10_000, used=9_000, free=1_000)

    receipt = joint_vast._local_evidence_capacity(
        job=tmp_path,
        bundle_path=bundle,
        disk_usage=lambda path: usage,
    )

    assert receipt["status"] == "blocked"
    assert receipt["blockers"] == [
        "adp_joint_agent_local_evidence_headroom_insufficient"
    ]
    assert receipt["minimum_free_bytes"] >= 2 * 1024**3


def test_local_closeout_recovery_requires_provider_and_object_store_zero(
    tmp_path: Path,
) -> None:
    job = tmp_path / "job"
    provider = job / "vast_provider_run"
    object_store = job / "object_store_staging"
    watchdog = job / "independent_vast_watchdog"
    execution = job / "immutable_execution"
    for path in (provider, object_store, watchdog, execution):
        path.mkdir(parents=True)
    write_json(
        provider / "vast_teardown_manifest.json",
        {"vast_instance_ids": [7], "continuing_spend_from_this_run": False},
    )
    write_json(
        provider / "vast_provider_adapter_result.json",
        {"status": "completed", "blockers": [], "estimated_cost_usd": 0.25},
    )
    write_json(
        object_store / "wam_provider_object_store_cleanup.json",
        {"all_objects_absent": True},
    )
    write_json(
        watchdog / "groot_oscar_runpod_canary_watchdog_cancel.json",
        {"instance_id": "7", "provider_absence_confirmed": True},
    )
    write_json(
        execution / "adp_joint_agent_result.json",
        {"status": "blocked", "blockers": ["scientific_null"]},
    )
    zero = {
        "schema_version": "adp_paid_provider_zero.v1",
        "provider": "vast",
        "provider_zero": True,
        "global_live_resource_count": 0,
        "provider_zero_digest": "",
    }
    zero["provider_zero_digest"] = canonical_digest(
        zero, digest_field="provider_zero_digest"
    )

    result = recover_joint_agent_local_closeout(
        job_dir=job, provider_zero_receipt=zero
    )

    assert result["status"] == "blocked"
    assert result["blockers"] == ["scientific_null"]
    assert result["continuing_spend_from_this_run"] is False
    assert result["independent_watchdog"]["status"] == "provider_terminal"
    assert result["local_receipt_recovered_after_provider_teardown"] is True


def test_builder_preflight_failure_leaves_no_partial_output(
    monkeypatch, tmp_path: Path
) -> None:
    packet = {"packet_digest": ""}
    packet["packet_digest"] = canonical_digest(packet, digest_field="packet_digest")
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    authority = {"authorization_digest": ""}
    authority["authorization_digest"] = canonical_digest(
        authority, digest_field="authorization_digest"
    )
    authority_path = tmp_path / "authority.json"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    freeze = {"freeze_digest": ""}
    freeze["freeze_digest"] = canonical_digest(freeze, digest_field="freeze_digest")
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    scope_amendment = {"amendment_digest": ""}
    scope_amendment["amendment_digest"] = canonical_digest(
        scope_amendment, digest_field="amendment_digest"
    )
    scope_amendment_path = tmp_path / "scope_amendment.json"
    scope_amendment_path.write_text(json.dumps(scope_amendment), encoding="utf-8")
    nim_preflight = {
        "schema_version": NIM_SCHEMA_VERSION,
        "status": "qualified",
        "endpoint": NIM_ENDPOINT,
        "model": NIM_MODEL,
        "http_status": 200,
        "credential_validated": True,
        "required_model_present": True,
        "paid_inference_performed": False,
        "provider_mutations_performed": 0,
        "raw_secret_values_recorded": False,
        "blockers": [],
        "receipt_digest": "",
    }
    nim_preflight["receipt_digest"] = canonical_digest(
        nim_preflight, digest_field="receipt_digest"
    )
    nim_preflight_path = tmp_path / "nim_preflight.json"
    nim_preflight_path.write_text(json.dumps(nim_preflight), encoding="utf-8")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.validate_public_scene_execution_authority",
        lambda value: value,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.inspect_joint_agent_checkout",
        lambda value: (_ for _ in ()).throw(ValueError("release mismatch")),
    )
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast._blueprint_identity",
        lambda value: {"commit": "a" * 40, "tree": "b" * 40, "dirty": False},
    )
    optimizer_zip = tmp_path / "scene_optimizer_core.zip"
    with zipfile.ZipFile(optimizer_zip, "w") as archive:
        for subdir in ("python", "lib", "extraLibs", "usdpy"):
            archive.writestr(f"{subdir}/.keep", subdir)
    monkeypatch.setattr(
        "blueprint_pipeline.adp_joint_agent_vast.SCENE_OPTIMIZER_CORE_SHA256",
        "sha256:" + __import__("hashlib").sha256(optimizer_zip.read_bytes()).hexdigest(),
    )
    destination = tmp_path / "bundle"

    with pytest.raises(ValueError, match="release mismatch"):
        build_joint_agent_vast_bundle(
            repo_root=Path(__file__).resolve().parents[1],
            joint_agent_root=checkout,
            packet_path=packet_path,
            execution_authority_path=authority_path,
            freeze_path=freeze_path,
            scope_amendment_path=scope_amendment_path,
            nim_preflight_path=nim_preflight_path,
            scene_optimizer_core_zip_path=optimizer_zip,
            job_dir=destination,
        )

    assert not destination.exists()


def test_provider_runner_retains_available_artifacts_and_skips_missing(
    tmp_path: Path,
) -> None:
    runner = _provider_runner_module()
    work = tmp_path / "provider_runtime/runtime_output/joint_agent_work"
    work.mkdir(parents=True)
    present = work / "articulation_candidates.json"
    present.write_text('{"candidates": []}', encoding="utf-8")
    output = tmp_path / "provider_return"

    rows = runner.retain_available_joint_agent_artifacts(
        output_root=output,
        artifacts={
            "articulation_candidates": present,
            "optimized_source": work / "missing_optimized.usdc",
        },
    )

    assert [row["role"] for row in rows] == ["articulation_candidates"]
    retained = output / rows[0]["relative_path"]
    assert retained.is_file()
    assert rows[0]["sha256"] == runner._sha256(retained)


def test_provider_runner_review_failure_retains_topology_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _provider_runner_module()
    root = tmp_path / "provider_runtime"
    output = tmp_path / "runtime_output"
    root.mkdir(parents=True)
    manifest = {
        "status": "ready",
        "model": {"backend": "nvidia_nim", "id": NIM_MODEL},
    }
    (root / "adp_joint_agent_provider_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    contract = {
        "schema_version": "joint_agent_task_topology_review_contract.v2",
        "minimum_assembly_joint_count": 1,
        "maximum_assembly_joint_count": 4,
        "commanded_task_joint_count": 1,
        "required_articulation_root_count": 1,
        "non_task_joint_mode": "locked_at_frozen_reset_with_native_readback",
        "non_task_joint_motion_tolerance": 0.001,
        "allowed_joint_types": ["revolute", "prismatic"],
        "target_joint_type": "revolute",
        "target_axis_world": [0.0, 0.0, 1.0],
        "target_axis_absolute_dot_minimum": 0.99,
        "target_member_projection_constraints": [
            {
                "axis_world": [0.0, 0.0, 1.0],
                "interval_m": [0.94, 1.632],
                "minimum_overlap_fraction": 0.85,
            }
        ],
        "compatibility_adapter": "legacy_rotation_observation_v1",
        "task_joint_id": "upper_hinge",
        "freeze_digest": "sha256:" + "2" * 64,
        "scope_amendment_digest": "sha256:" + "3" * 64,
        "contract_digest": "",
    }
    contract["contract_digest"] = canonical_digest(
        contract, digest_field="contract_digest"
    )
    (root / "joint_review_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    candidates_path = (
        root / "runtime_output/joint_agent_work/articulation_candidates/articulation_candidates.json"
    )
    optimized_path = (
        root / "runtime_output/joint_agent_work/optimized/articulated_source_optimized.usdc"
    )

    def fake_run(command: list[str], log_name: str) -> dict:
        output.mkdir(parents=True, exist_ok=True)
        if log_name == "joint_agent_inference.log":
            candidates_path.parent.mkdir(parents=True, exist_ok=True)
            candidates_path.write_text(
                json.dumps(
                    {
                        "schema_version": "joint-agent-stage2-v0",
                        "candidates": [
                            {
                                "candidate_id": "upper",
                                "joint_type_hint": "revolute",
                                "review_status": "needs_more_context",
                                "unresolved_reason_codes": [],
                                "motion_axis_world": [0.0, 0.0, 1.0],
                                "moving_part_prims": ["/Asset/upper"],
                            }
                        ],
                        "summary": {"candidate_count": 1},
                    }
                ),
                encoding="utf-8",
            )
            optimized_path.parent.mkdir(parents=True, exist_ok=True)
            optimized_path.write_bytes(b"fixture-optimized-usd")
        log = output / log_name
        log.write_text("fixture", encoding="utf-8")
        return {
            "returncode": 0,
            "started_at": "start",
            "ended_at": "end",
            "duration_seconds": 0.0,
            "log_path": str(log),
            "log_sha256": runner._sha256(log),
        }

    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(runner, "scene_optimizer_probe", lambda *, output_root: [])
    monkeypatch.setattr(
        runner,
        "_candidate_bounds",
        lambda stage_path, document: {
            "upper": {
                "status": "measured_from_optimized_usd",
                "moving_part_prims": ["/Asset/upper"],
                "aabb_min": [0.0, 0.0, 0.94],
                "aabb_max": [1.0, 1.0, 1.632],
            }
        },
    )
    monkeypatch.setenv("NVIDIA_API_KEY", "hermetic-test-placeholder")

    assert runner.main() == 2

    result = json.loads((output / "adp_joint_agent_result.json").read_text())
    assert result["status"] == "blocked"
    assert any(
        blocker.startswith("joint_agent_deterministic_review_failed:")
        for blocker in result["blockers"]
    )
    rows = {row["role"]: row for row in result["retained_artifacts"]}
    assert set(rows) == {"articulation_candidates", "optimized_source"}
    for row in rows.values():
        retained = output / row["relative_path"]
        assert retained.is_file()
        assert row["sha256"] == runner._sha256(retained)


def test_provider_runner_resolves_one_non_symlink_output_by_role(
    tmp_path: Path,
) -> None:
    runner = _provider_runner_module()
    working = tmp_path / "work"
    optimized = working / "optimized/fixture_optimized.usdc"
    optimized.parent.mkdir(parents=True)
    optimized.write_bytes(b"usd")

    assert runner.resolve_joint_agent_output(
        working_dir=working,
        role="optimized_source",
        relative_glob="optimized/*_optimized.usd*",
    ) == optimized.resolve()

    second = optimized.with_name("other_optimized.usda")
    second.write_bytes(b"usd")
    with pytest.raises(
        ValueError,
        match="joint_agent_output_role_not_unique:optimized_source:observed=2",
    ):
        runner.resolve_joint_agent_output(
            working_dir=working,
            role="optimized_source",
            relative_glob="optimized/*_optimized.usd*",
        )


@pytest.mark.parametrize(
    ("backend", "present_env", "expected"),
    [
        ("openai", "OPENAI_API_KEY", []),
        ("nvidia_nim", "NVIDIA_API_KEY", []),
        ("unknown", "OPENAI_API_KEY", ["joint_agent_model_backend_not_supported"]),
    ],
)
def test_provider_runner_binds_credentials_to_model_backend(
    monkeypatch, backend: str, present_env: str, expected: list[str]
) -> None:
    runner = _provider_runner_module()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv(present_env, "hermetic-test-placeholder")

    assert runner.model_credential_blockers(
        {"model": {"backend": backend, "id": "fixture-model"}}
    ) == expected


def test_provider_runner_reports_typed_openai_credential_gap_before_gpu_work(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _provider_runner_module()
    root = tmp_path / "provider_runtime"
    output = tmp_path / "runtime_output"
    root.mkdir()
    manifest = {
        "status": "ready",
        "model": {"backend": "openai", "id": "fixture-openai-model"},
    }
    (root / "adp_joint_agent_provider_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    contract = {"contract_digest": ""}
    contract["contract_digest"] = canonical_digest(
        contract, digest_field="contract_digest"
    )
    (root / "joint_review_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "OUTPUT", output)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        runner,
        "scene_optimizer_probe",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("GPU work started")),
    )

    assert runner.main() == 2
    result = json.loads((output / "adp_joint_agent_result.json").read_text())
    assert result["blockers"] == ["joint_agent_model_api_key_missing:openai"]


def test_nim_resolver_prefers_inference_key_over_registry_key(
    monkeypatch, tmp_path: Path
) -> None:
    from blueprint_pipeline.gpu_render_providers import PROVIDER_SECRETS_DIR_ENV

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    (tmp_path / "nvidia_nim_api_key").write_text("inference-key", encoding="utf-8")
    (tmp_path / "ngc_api_key").write_text("registry-key", encoding="utf-8")
    # The configured secrets directory, which is how a deployed host supplies
    # these. The resolver used to take the root as a parameter defaulting to a
    # developer home -- unreadable under ProtectHome=true, which is what made
    # this lane fail after admission.
    monkeypatch.setenv(PROVIDER_SECRETS_DIR_ENV, str(tmp_path))

    assert joint_vast._model_api_key("nvidia_nim") == (
        "inference-key",
        "NVIDIA_API_KEY",
    )


def test_output_zip_inspection_recognizes_joint_agent_runtime_result(
    tmp_path: Path,
) -> None:
    from blueprint_pipeline.wam_provider_output import (
        inspect_provider_runtime_output_zip,
    )

    output_zip = tmp_path / "vast_provider_runtime_output.zip"
    with zipfile.ZipFile(output_zip, "w") as archive:
        archive.writestr(
            "adp_joint_agent_result.json",
            json.dumps({"status": "completed", "blockers": []}),
        )

    inspection = inspect_provider_runtime_output_zip(output_zip)

    assert inspection["runtime_result_present"] is True
    assert inspection["runtime_result_status"] == "completed"


def test_runtime_script_probes_ovrtx_daemon_before_service() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_adp_joint_agent_provider_runtime.sh"
    ).read_text(encoding="utf-8")

    assert "ovrtx_daemon_probe.log" in script
    assert "joint_agent_ovrtx_daemon_probe_failed" in script
    assert script.index("ovrtx_daemon_probe.log") < script.index(
        "uvicorn service.main:app"
    )
    assert "env -u PYTHONPATH" in script
    assert "runtime_disk_headroom.json" in script
    assert "joint_agent_runtime_disk_headroom_insufficient" in script
    assert script.index("runtime_disk_headroom.json") < script.index(
        "pip install --disable-pip-version-check"
    )
    assert "UV_NATIVE_TLS=true" in script
    assert "UV_DEFAULT_INDEX" in script
    assert "python_build_dependency_plan.json" in script
    assert "--no-build-isolation" in script
    assert '-e "${SOURCE_DIR}"' not in script
    assert (
        '"${SOURCE_DIR}/.venv/bin/python" '
        '"${SCRIPT_DIR}/provider_python_build_plan.py"'
    ) in script
    assert script.index("python_build_dependency_plan.json") < script.index(
        '"${SOURCE_DIR}/apps/joint_agent"'
    )


def test_scene_optimizer_probe_retains_real_failure_diagnostics(
    monkeypatch, tmp_path: Path
) -> None:
    runner = _provider_runner_module()

    def _failing_loader():
        def _optimize(*args, **kwargs):
            raise RuntimeError(
                "Scene Optimizer subprocess failed (exit code 127)\n"
                "--- stderr (last 2000) ---\nlibExample.so: cannot open shared object file"
            )

        return _optimize

    monkeypatch.setattr(runner, "_load_optimize_usd_local", _failing_loader)
    blockers = runner.scene_optimizer_probe(output_root=tmp_path)

    assert blockers == ["joint_agent_scene_optimizer_probe_failed"]
    log = (tmp_path / "scene_optimizer_probe.log").read_text(encoding="utf-8")
    assert "cannot open shared object file" in log

    def _passing_loader():
        def _optimize(input_path, output_path, config):
            assert Path(input_path).is_file()
            settings = config["scene_optimizer_settings"]
            assert settings["enable_split_meshes"] is True
            return {"status": "completed", "operations_executed": ["splitMeshes"]}

        return _optimize

    monkeypatch.setattr(runner, "_load_optimize_usd_local", _passing_loader)
    assert runner.scene_optimizer_probe(output_root=tmp_path) == []
    log = (tmp_path / "scene_optimizer_probe.log").read_text(encoding="utf-8")
    assert "scene_optimizer_probe_ok" in log
