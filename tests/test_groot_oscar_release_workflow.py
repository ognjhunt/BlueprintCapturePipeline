from pathlib import Path

from scripts.verify_groot_oscar_thin_architecture import verify


def test_release_workflow_uses_known_amd64_docker_builder_and_digest_handoff() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/groot-oscar-thin-release.yml").read_text(
        encoding="utf-8"
    )
    assert "runs-on: [self-hosted, linux, x64, blueprint-large-docker]" in workflow
    assert "test \"$(uname -m)\" = x86_64" in workflow
    assert "120 * 1024 * 1024" in workflow
    assert "groot_oscar_infrastructure_admission build" in workflow
    assert "remote_build_groot_oscar_thin_images.sh" in workflow
    assert "release_buildx_metadata.json" in workflow
    assert "groot_oscar_thin_remote_build_result.json" in workflow
    assert "runpod.io" not in workflow.lower()


def test_thin_architecture_has_a_dedicated_network_free_ci_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "scripts/verify_groot_oscar_thin_architecture.py" in workflow
    assert verify() == []
