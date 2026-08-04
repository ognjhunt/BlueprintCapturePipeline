from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline.paid_resource_admission import PaidResourceAdmissionGrant
from blueprint_pipeline.provider_runtime_bundle_contract import (
    provider_runtime_contract_blockers,
)
from blueprint_pipeline.simpler_public_vast import (
    ADMITTED_MACHINE_IDS,
    PROBE_KIND,
    _adp_session_budget_ledger,
    _vast_authority_environment,
    build_simpler_public_vast_bundle,
)
from blueprint_pipeline.simpler_public_runtime_worker import (
    _activate_verified_source_roots,
    _cuda_toolkit_evidence,
    _vulkan_runtime_evidence,
)
from blueprint_pipeline.vast_provider_adapter import (
    _probe_env,
    _probe_shell_script,
    _search_payload,
)


ROOT = Path(__file__).parents[1]
MANIFEST = (
    ROOT
    / "docs"
    / "arm_decision_proof_v1"
    / "manifests"
    / "simpler_google_robot_pick_coke_can.v1.json"
)


def test_bundle_contains_public_runtime_but_not_physical_outcome_values(
    tmp_path: Path,
) -> None:
    receipt = build_simpler_public_vast_bundle(
        manifest_path=MANIFEST, job_dir=tmp_path / "bundle", generated_at="fixed"
    )

    assert receipt["status"] == "ready"
    assert receipt["physical_outcome_values_bundled"] is False
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        names = set(archive.namelist())
        entrypoint = archive.read("provider_runtime/run_adp_simpler_provider_runtime.sh").decode()
        runner = archive.read("provider_runtime/adp_simpler_provider_runner.py").decode()
        bundled_manifest = json.loads(
            archive.read("provider_runtime/public_reference_manifest.json")
        )
    assert "provider_runtime/adp_simpler_provider_runner.py" in names
    assert not any("physical_outcomes" in name for name in names)
    assert "cells" not in bundled_manifest["physical_reference"]
    assert 'manifest["runtime"]["environment_lock"]["container_image"]' in runner
    assert "cuda_toolkit_ptxas_missing" in runner
    assert "cuda_toolkit_libdevice_missing" in runner
    assert "--xla_gpu_cuda_data_dir=" in runner
    assert "nvidia_vulkan_device_not_observed" in runner
    assert "VK_ICD_FILENAMES" in runner
    assert "paid_runtime_plan" not in runner
    assert "BLUEPRINT_WAM_RUNTIME_PHASE:adp_simpler" in runner
    assert 'source_dir / "ManiSkill2_real2sim"' in runner
    assert 'SCHEMA_VERSION = "simpler_closed_loop_execution.v2"' in runner
    assert "observation_frame_manifest" in runner
    assert "episode_video" in runner
    assert "environment_step_info.success" in runner
    assert '"vlm_used": False' in runner
    assert (
        provider_runtime_contract_blockers(
            provider_bundle_kind="adp_simpler",
            entrypoint_text=entrypoint,
            runner_text=runner,
        )
        == []
    )
    second = build_simpler_public_vast_bundle(
        manifest_path=MANIFEST, job_dir=tmp_path / "bundle-2", generated_at="fixed"
    )
    assert second["bundle_sha256"] == receipt["bundle_sha256"]


def test_canonical_grant_bridge_sets_and_restores_adapter_mutation_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BLUEPRINT_ALLOW_VAST_API_CALLS", raising=False)
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "prior")

    with _vast_authority_environment():
        assert allocator.os.environ["BLUEPRINT_ALLOW_VAST_API_CALLS"] == "1"
        assert allocator.os.environ["BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"] == "1"

    assert "BLUEPRINT_ALLOW_VAST_API_CALLS" not in allocator.os.environ
    assert allocator.os.environ["BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"] == "prior"


def test_adp_budget_ledger_is_run_local(tmp_path: Path) -> None:
    assert _adp_session_budget_ledger(tmp_path) == (
        tmp_path.resolve() / "adp_vast_session_budget.json"
    )


def test_adp_runtime_pins_only_observed_vulkan_capable_machine() -> None:
    assert ADMITTED_MACHINE_IDS == (41950,)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["runtime"]["environment_lock"]["admitted_vast_machine_ids"] == [41950]
    assert _search_payload(
        limit=100,
        max_hourly_rate=0.8,
        allowed_machine_ids=ADMITTED_MACHINE_IDS,
    )["machine_id"] == {"in": [41950]}


def test_worker_activates_only_verified_editable_source_roots(tmp_path: Path) -> None:
    source = tmp_path / "SimplerEnv"
    original = list(sys.path)
    try:
        roots = _activate_verified_source_roots({"source_dir": source})
        assert roots == [str(source / "ManiSkill2_real2sim"), str(source)]
        assert all(root in sys.path for root in roots)
    finally:
        sys.path[:] = original


def test_worker_binds_xla_compiler_inputs_into_runtime_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cuda_root = tmp_path / "cuda"
    libdevice = cuda_root / "nvvm" / "libdevice" / "libdevice.10.bc"
    libdevice.parent.mkdir(parents=True)
    libdevice.write_bytes(b"immutable-libdevice")
    monkeypatch.setattr(
        "blueprint_pipeline.simpler_public_runtime_worker.shutil.which",
        lambda name: "/usr/local/cuda/bin/ptxas" if name == "ptxas" else None,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.simpler_public_runtime_worker._run",
        lambda *args, **kwargs: {"returncode": 0, "output_tail": "ptxas 12.2"},
    )

    evidence = _cuda_toolkit_evidence(cuda_root)

    assert evidence["xla_flags"] == f"--xla_gpu_cuda_data_dir={cuda_root}"
    assert evidence["libdevice_files"][0]["sha256"].startswith("sha256:")


def test_worker_rejects_image_without_xla_compiler_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "blueprint_pipeline.simpler_public_runtime_worker.shutil.which",
        lambda name: None,
    )

    with pytest.raises(RuntimeError, match="cuda_toolkit_ptxas_missing"):
        _cuda_toolkit_evidence(tmp_path / "cuda")


def test_worker_binds_observed_nvidia_vulkan_icd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    icd = tmp_path / "nvidia_icd.json"
    icd.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "all")
    monkeypatch.setattr(
        "blueprint_pipeline.simpler_public_runtime_worker.shutil.which",
        lambda name: "/usr/bin/vulkaninfo" if name == "vulkaninfo" else None,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.simpler_public_runtime_worker._run",
        lambda command, **kwargs: {
            "returncode": 0,
            "output_tail": (
                "GPU0: NVIDIA GeForce RTX 4090"
                if command[0].endswith("vulkaninfo")
                else "libegl1=1.4.0\nlibxext6=1.3.4\nlibvulkan1=1.3.204\nvulkan-tools=1.3.204"
            ),
        },
    )

    evidence = _vulkan_runtime_evidence((icd,))

    assert evidence["driver_capabilities"] == "all"
    assert evidence["vk_driver_files"] == str(icd)
    assert evidence["vk_icd_filenames"] == str(icd)
    assert evidence["icd_sha256"].startswith("sha256:")
    assert "libegl1=1.4.0" in evidence["system_packages"]


def test_adp_vast_env_requests_graphics_without_isaac_terms(tmp_path: Path) -> None:
    env = _probe_env(
        job_dir=tmp_path,
        enable_isaac_smoke=False,
        provider_bundle_kind="adp_simpler",
        forward_hf_token=False,
    )

    assert env["NVIDIA_DRIVER_CAPABILITIES"] == "all"
    assert "ACCEPT_EULA" not in env
    assert "PRIVACY_CONSENT" not in env

    script = _probe_shell_script(
        "https://example.com",
        enable_isaac_smoke=False,
        enable_blueprint_bundle=True,
        provider_bundle_kind="adp_simpler",
    )
    assert "libegl1 libxext6" in script


def test_canonical_grant_bridge_sets_and_restores_adapter_mutation_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BLUEPRINT_ALLOW_VAST_API_CALLS", raising=False)
    monkeypatch.setenv("BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH", "prior")

    with _vast_authority_environment():
        assert allocator.os.environ["BLUEPRINT_ALLOW_VAST_API_CALLS"] == "1"
        assert allocator.os.environ["BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"] == "1"

    assert "BLUEPRINT_ALLOW_VAST_API_CALLS" not in allocator.os.environ
    assert allocator.os.environ["BLUEPRINT_ALLOW_VAST_INSTANCE_LAUNCH"] == "prior"


def _allocator_args(tmp_path: Path, *, execute: bool) -> list[str]:
    values = [
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
        "adp-simpler",
        "--adp-public-reference-manifest",
        str(MANIFEST),
        "--adp-job-dir",
        str(tmp_path / "job"),
    ]
    if execute:
        values.append("--execute")
    return values


@pytest.mark.parametrize("execute", [False, True])
def test_canonical_allocator_issues_grant_only_for_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execute: bool
) -> None:
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "completed" if kwargs["execute"] else "dry_run_ready"}

    monkeypatch.setattr(allocator, "run_simpler_public_vast", fake_run)

    assert allocator.main(_allocator_args(tmp_path, execute=execute)) == 0
    assert observed["execute"] is execute
    grant = observed["paid_resource_admission_grant"]
    assert isinstance(grant, PaidResourceAdmissionGrant) is execute
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["retry_cap"] == 0
    assert admission["physical_outcome_values_uploaded"] is False
    assert admission["hard_cap_usd"] == 2.0
    assert admission["hard_ttl_seconds"] == 7200
    assert (
        admission["allocation_binding"]["bundle_sha256"]
        == observed["prepared_bundle"]["bundle_sha256"]
    )
    assert admission["allocation_binding_digest"].startswith("sha256:")
    assert admission["allocation_binding"]["machine_avoidlist_digest"] is None


def test_allocator_digest_binds_reviewed_machine_avoidlist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    avoidlist = tmp_path / "avoidlist.json"
    avoidlist.write_text(
        json.dumps(
            {
                "schema_version": "vast_machine_avoidlist.v1",
                "status": "completed",
                "machine_ids": [56730],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        allocator,
        "_control_plane_checkout_blockers",
        lambda: ([], {"orchestrator_source_commit": "a" * 40, "checkout_clean": True}),
    )
    observed: dict = {}
    monkeypatch.setattr(
        allocator,
        "run_simpler_public_vast",
        lambda **kwargs: observed.update(kwargs) or {"status": "dry_run_ready"},
    )
    args = _allocator_args(tmp_path, execute=False) + [
        "--adp-machine-avoidlist",
        str(avoidlist),
    ]

    assert allocator.main(args) == 0
    admission = json.loads((tmp_path / "admission.json").read_text())
    assert admission["allocation_binding"]["machine_avoidlist_digest"].startswith("sha256:")
    assert observed["machine_avoidlist_path"] == str(avoidlist)
