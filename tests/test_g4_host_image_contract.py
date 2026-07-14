from pathlib import Path


ROOT = Path("infra/gcp/g4_host_image")


def test_packer_host_is_date_pinned_and_never_bakes_application_code():
    template = (ROOT / "g4-host.pkr.hcl").read_text()
    install = (ROOT / "install-pinned-host.sh").read_text()
    assert "source_image" in template
    assert "source_image_family" not in template
    assert "nvidia_driver_sha256" in template
    assert "nvidia_container_toolkit_version" in template
    assert "application_or_model_code_baked=false" in install
    assert "/opt/blueprint" not in install
    assert "/opt/gr00t" not in install


def test_host_self_test_requires_driver_docker_and_nvidia_runtime():
    script = (ROOT / "blueprint-g4-host-self-test.sh").read_text()
    assert "nvidia-smi" in script
    assert "docker version" in script
    assert "Runtimes" in script and "nvidia" in script
    assert '"status": "passed"' in script
