from pathlib import Path


LEGACY_SCRIPT = Path("scripts/build_push_groot_oscar_closed_loop_image.sh")
PACKET = Path("src/blueprint_pipeline/groot_oscar_thin_remote_build_packet.py")
FOUNDATION = Path(
    "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Foundation.Dockerfile"
)
ENTRYPOINT = Path(
    "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/thin_release_entrypoint.sh"
)
RELEASE = Path(
    "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Release.Dockerfile"
)


def test_legacy_monolithic_build_is_fail_closed() -> None:
    lines = LEGACY_SCRIPT.read_text(encoding="utf-8").splitlines()
    disable = lines.index(
        'echo "legacy build path disabled; use paid_resource_allocator cpu-build" >&2'
    )
    assert lines[disable + 1] == "exit 2"
    assert disable < 20


def test_canonical_thin_build_attests_and_scans_only_registry_digest() -> None:
    text = PACKET.read_text(encoding="utf-8")
    compact = " ".join(text.split())
    assert "--attest type=sbom --attest type=provenance,mode=max" in compact
    assert '\"registry:$release_exact\"' in text
    assert 'syft_bin\" \"$release_exact' not in text
    assert "release_supply_chain_disk_admission.json" in text
    assert "release_buildkit_sbom_attestation.json" in text
    assert "release_buildkit_provenance_attestation.json" in text
    assert "release_supply_chain_manifest.json" in text
    assert "SYFT_ARCHIVE_SHA256" in text


def test_foundation_uses_runtime_only_wbc_multistage_closure() -> None:
    text = FOUNDATION.read_text(encoding="utf-8")
    assert "AS wbc-builder" in text
    runtime = text.index("FROM tensorrt-base\n")
    assert "COPY --from=wbc-builder" in text[runtime:]
    assert "cuda-compiler" not in text[runtime:]


def test_thin_entrypoint_uses_installed_absolute_worker_executable() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "/opt/oscar-venv/bin/blueprint-run-robot-eval-worker" in text
    assert "set -- blueprint-run-robot-eval-worker" not in text
    assert (
        "test -x /opt/oscar-venv/bin/blueprint-run-robot-eval-worker"
        in RELEASE.read_text(encoding="utf-8")
    )
