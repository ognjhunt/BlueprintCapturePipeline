import json
from pathlib import Path

from scripts.verify_groot_oscar_live_prerequisites import (
    _verify_isaac_base_image,
    verify_static,
)
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


def test_paid_foundation_build_has_a_free_live_prerequisite_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "foundation-prerequisites:" in workflow
    assert "runs-on: ubuntu-24.04" in workflow
    assert "scripts/verify_groot_oscar_live_prerequisites.py" in workflow
    assert "--live" in workflow
    assert verify_static() == []


def test_isaac_base_manifest_verification_fails_closed_without_leaking_token() -> None:
    token = "ephemeral-test-token"
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "manifests": [
                {"platform": {"os": "linux", "architecture": "amd64"}}
            ],
        },
        separators=(",", ":"),
    ).encode()

    def fake_fetch(url: str) -> bytes:
        assert url.startswith("https://nvcr.io/proxy_auth?")
        return json.dumps({"token": token}).encode()

    def fake_authorized_fetch(url: str, supplied_token: str) -> bytes:
        assert url.startswith("https://nvcr.io/v2/nvidia/isaac-sim/manifests/sha256:")
        assert supplied_token == token
        return manifest

    blockers, check = _verify_isaac_base_image(fake_fetch, fake_authorized_fetch)
    assert blockers == ["isaac_base_image_digest_mismatch"]
    assert check["linux_amd64_present"] is True
    assert token not in json.dumps(check)
