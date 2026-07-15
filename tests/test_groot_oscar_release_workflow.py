import json
import urllib.request
from pathlib import Path

import pytest

from scripts.verify_groot_oscar_live_prerequisites import (
    _AllowlistedRedirectHandler,
    _verify_isaac_base_image,
    summarize_required_model_metadata,
    verify_static,
)
from scripts.verify_groot_oscar_thin_architecture import verify


WORKFLOW = Path(".github/workflows/groot-oscar-thin-release.yml")


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
    prerequisite = workflow.index("verify_groot_oscar_live_prerequisites.py")
    allocator = workflow.index("paid_resource_allocator cpu-build")
    assert prerequisite < allocator
    assert "--live" in workflow[prerequisite:allocator]
    assert "groot_oscar_live_prerequisites.json" in workflow
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


def test_model_metadata_sizing_fails_closed_for_missing_or_unknown_sizes() -> None:
    available, missing, invalid_sizes, required_bytes = summarize_required_model_metadata(
        [
            {"rfilename": "weights.bin", "size": 123},
            {"rfilename": "config.json", "size": None},
            {"rfilename": "ignored.txt", "size": 999},
        ],
        ("weights.bin", "config.json", "tokenizer.json"),
    )
    assert set(available) == {"weights.bin", "config.json", "ignored.txt"}
    assert missing == ["tokenizer.json"]
    assert invalid_sizes == ["config.json"]
    assert required_bytes == 123


def test_prerequisite_redirect_handler_reapplies_outbound_allowlist() -> None:
    handler = _AllowlistedRedirectHandler()
    request = urllib.request.Request("https://github.com/source")
    with pytest.raises(ValueError, match="prerequisite_url_not_allowlisted"):
        handler.redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://attacker.example/payload",
        )


def test_release_workflow_is_serialized_and_uses_one_canonical_allocator():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "cancel-in-progress: false" in text
    assert "paid_resource_allocator cpu-build" in text
    assert "--execution-plane local" in text
    assert "cpu-build-local" not in text
    assert "scripts/build_push_groot_oscar_closed_loop_image.sh" not in text
    assert "release_sbom.spdx.json" in text
    assert "release_provenance.json" in text
    assert "release_supply_chain_manifest.json" in text
    assert not Path(".github/workflows/groot-oscar-release-image.yml").exists()


def test_release_workflow_uses_ephemeral_file_credentials_and_always_cleans_them():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "registry-secrets/username" in text
    assert "registry-secrets/password" in text
    assert "BLUEPRINT_DOCKER_USERNAME_FILE" in text
    assert "BLUEPRINT_DOCKER_PASSWORD_FILE" in text
    assert "DOCKER_CONFIG: ${{ runner.temp }}/blueprint-docker-config" in text
    assert "if: always()" in text
    assert "Upload immutable release evidence" in text
    assert 'rm -rf "$RUNNER_TEMP/registry-secrets" "$RUNNER_TEMP/blueprint-docker-config"' in text
