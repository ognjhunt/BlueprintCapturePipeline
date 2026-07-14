from pathlib import Path


SCRIPT = Path("scripts/build_push_groot_oscar_closed_loop_image.sh")


def test_official_build_enables_buildkit_attestations_and_scans_registry_digest():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "--attest type=sbom" in " ".join(text.split())
    assert "--provenance mode=max" in " ".join(text.split())
    assert 'syft "registry:${exact_digest_ref}"' in text
    assert 'syft "$image_ref"' not in text
    assert "groot_oscar_closed_loop_supply_chain_evidence_failed" in text
    assert "groot_oscar_closed_loop_disk_admission.json" in text
    assert "expected_unpacked_gib" in text
    assert "--format '{{json .SBOM}}'" in text
    assert "--format '{{json .Provenance}}'" in text
    assert '"WBC_SOURCE_REF=$wbc_ref"' in text
    assert '"GEAR_SONIC_CHECKPOINT_REVISION=$gear_checkpoint_revision"' in text
    assert '"size_bytes": row.get("size")' in text
    assert '"size_bytes": row.get("size_bytes")' not in text


def test_release_runtime_smoke_precedes_push_and_binds_published_digest():
    text = SCRIPT.read_text(encoding="utf-8")
    smoke_index = text.index('docker run --rm --entrypoint /bin/bash "$runtime_image_ref"')
    publish_index = text.index("publish_build_args=(")
    assert smoke_index < publish_index
    assert "--load" in text[smoke_index - 2500 : smoke_index]
    assert "--push" in text[publish_index : publish_index + 800]
    assert 'runtime_image_ref="${runtime_image_ref%:*}@${build_digest}"' in text
    assert "published_config_digest" in text
    assert "published_runtime_identity_matches_smoked_local_image" in text
    assert 'docker pull "$runtime_image_ref"' not in text
    assert 'payload.get("Manifest")' in text
    assert 'payload.get("manifest")' not in text


def test_release_image_uses_runtime_only_wbc_multistage_closure():
    dockerfile = Path(
        "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile"
    ).read_text(encoding="utf-8")
    from_line = "FROM --platform=linux/amd64 ${ISAAC_SIM_BASE_IMAGE}"
    assert dockerfile.count(from_line) == 2
    assert "AS gear_sonic_builder" in dockerfile
    assert "AS runtime" in dockerfile
    assert "COPY --from=gear_sonic_builder /opt/wbc/gear_sonic_deploy" in dockerfile


def test_oscar_checkpoint_override_is_the_runtime_provenance_revision():
    dockerfile = Path(
        "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile"
    ).read_text(encoding="utf-8")
    assert "ARG OSCAR_CHECKPOINT_REVISION=" in dockerfile
    assert "BLUEPRINT_OSCAR_WAM_HF_REVISION=${OSCAR_CHECKPOINT_REVISION}" in dockerfile
