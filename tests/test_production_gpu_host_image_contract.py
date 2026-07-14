from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.production_gpu_worker_pool import readiness_main


ROOT = Path(__file__).resolve().parents[1]


def test_packer_host_image_requires_digest_and_bakes_runtime_and_worker_cache() -> None:
    template = (ROOT / "deploy/packer/gcp_g4_gpu_worker_host.pkr.hcl").read_text()
    bake = (ROOT / "deploy/packer/scripts/bake_gpu_worker_host.sh").read_text()

    assert "worker_image_ref must be digest pinned" in template
    assert "omit_external_ip        = true" in template
    assert "use_internal_ip         = true" in template
    assert "nvidia-container-toolkit" in bake
    assert 'docker pull "$image_ref"' in bake
    assert "/etc/blueprint/worker-image-ref" in bake
    assert 'rm -f /root/.docker/config.json' in bake
    assert "\ndocker image prune" not in bake


def test_runtime_cloud_init_never_installs_or_pulls() -> None:
    provider = (ROOT / "src/blueprint_pipeline/cloud_vm_render_providers.py").read_text()
    startup_function = provider.split("def _worker_cloud_init", 1)[1].split(
        "class GCPRenderProvider", 1
    )[0]
    assert "apt-get" not in startup_function
    assert "docker pull" not in startup_function
    assert "docker login" not in startup_function
    assert "docker image inspect" in startup_function
    assert "/etc/blueprint/worker-image-ref" in startup_function


def test_readiness_cli_writes_blocked_local_gate_without_live_evidence(
    tmp_path: Path,
) -> None:
    output = tmp_path / "readiness.json"
    rc = readiness_main(
        [
            "--host-image-id",
            "projects/test/global/images/host-v1",
            "--worker-image-ref",
            "registry.example/worker@sha256:" + "a" * 64,
            "--gpu-family",
            "g4-rtx-pro-6000",
            "--output",
            str(output),
        ]
    )
    assert rc == 1
    assert '"status": "local_contract_ready_live_proof_required"' in output.read_text()


def test_production_pool_systemd_unit_is_private_and_restartable() -> None:
    unit = (
        ROOT / "deploy/systemd/blueprint-production-gpu-worker-pool.service"
    ).read_text()
    assert "--host 127.0.0.1" in unit
    assert "Restart=on-failure" in unit
    assert "NoNewPrivileges=true" in unit
    assert "ProtectSystem=strict" in unit
