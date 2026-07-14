from pathlib import Path


SCRIPT = Path("scripts/build_push_groot_oscar_closed_loop_image.sh")


def test_official_build_enables_buildkit_attestations_and_scans_registry_digest():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--attest type=sbom --provenance mode=max" in text
    assert 'syft "registry:${exact_digest_ref}"' in text
    assert 'syft "$image_ref"' not in text
    assert "groot_oscar_closed_loop_supply_chain_evidence_failed" in text
    assert "groot_oscar_closed_loop_disk_admission.json" in text
    assert "expected_unpacked_gib" in text
    assert "--format '{{json .SBOM}}'" in text
    assert "--format '{{json .Provenance}}'" in text
    assert '"WBC_SOURCE_REF=$wbc_ref"' in text
    assert '"GEAR_SONIC_CHECKPOINT_REVISION=$gear_checkpoint_revision"' in text


def test_release_runtime_smoke_uses_the_pushed_immutable_digest():
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'runtime_image_ref="${runtime_image_ref%:*}@${build_digest}"' in text
    assert 'docker pull "$runtime_image_ref"' in text
    assert 'docker run --rm --entrypoint /bin/bash "$runtime_image_ref"' in text
