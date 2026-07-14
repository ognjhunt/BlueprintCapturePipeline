from pathlib import Path


ROOT = Path("infra/gcp/g4_host_image")


def test_packer_host_is_date_pinned_and_preloads_only_exact_worker_closure():
    template = (ROOT / "g4-host.pkr.hcl").read_text()
    install = (ROOT / "install-pinned-host.sh").read_text()
    assert "source_image" in template
    assert "source_image_family" not in template
    assert "[a-z0-9-]+-v[0-9]{8}" in template
    assert 'source_image_project_id = ["ubuntu-os-cloud"]' in template
    assert 'scopes                  = ["https://www.googleapis.com/auth/cloud-platform"]' in template
    assert 'blueprint-managed = "true"' in template
    assert "nvidia_driver_sha256" in template
    assert "nvidia_container_toolkit_version" in template
    assert 'default = "1.19.1-1"' in template
    assert "worker_image_digest_ref" in template
    assert "worker_source_sha" in template
    assert 'disk_size               = 300' in template
    assert 'docker pull "$WORKER_IMAGE_DIGEST_REF"' in install
    assert 'docker image inspect "$WORKER_IMAGE_DIGEST_REF"' in install
    assert "application_or_model_code_outside_worker_image=false" in install
    assert '"linux-headers-$(uname -r)"' in install
    assert "build-essential" in install and "dkms" in install
    assert "Metadata-Flavor: Google" in install
    assert "/opt/blueprint" not in install
    assert "/opt/gr00t" not in install


def test_host_self_test_requires_driver_docker_and_nvidia_runtime():
    script = (ROOT / "blueprint-g4-host-self-test.sh").read_text()
    assert "nvidia-smi" in script
    assert "docker version" in script
    assert "Runtimes" in script and "nvidia" in script
    assert 'docker image inspect "$worker_ref"' in script
    assert '"image_present_before_allocation": True' in script
    assert '"cold_pull_required_during_campaign": False' in script
    assert '"status": "passed"' in script
