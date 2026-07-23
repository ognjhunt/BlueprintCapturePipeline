from pathlib import Path


LEGACY_SCRIPT = Path("scripts/build_push_groot_oscar_closed_loop_image.sh")
PACKET = Path("src/blueprint_pipeline/groot_oscar_thin_remote_build_packet.py")
FOUNDATION = Path(
    "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Foundation.Dockerfile"
)
APT_TRANSPORT_HARDENING = Path(
    "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/apt_transport_hardening.conf"
)
ENTRYPOINT = Path(
    "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/thin_release_entrypoint.sh"
)
HEALTHCHECK = Path(
    "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
    "groot_oscar_closed_loop_image_healthcheck.py"
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
    assert "PYTHONPATH=/opt/wbc:/opt/OSCAR" in text
    assert "BLUEPRINT_GEAR_SONIC_SOURCE_REVISION=${WBC_SOURCE_REF}" in text


def test_foundation_retries_package_fetches_and_uses_tls_ubuntu_mirrors() -> None:
    dockerfile = FOUNDATION.read_text(encoding="utf-8")
    apt_config = APT_TRANSPORT_HARDENING.read_text(encoding="utf-8")

    assert 'Acquire::Retries "10";' in apt_config
    assert 'Acquire::http::Timeout "30";' in apt_config
    assert 'Acquire::https::Timeout "30";' in apt_config
    assert 'Acquire::http::Pipeline-Depth "0";' in apt_config
    assert 'Acquire::https::Pipeline-Depth "0";' in apt_config
    assert dockerfile.count(
        "COPY deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
        "apt_transport_hardening.conf"
    ) == 2
    assert dockerfile.count("find /etc/apt -maxdepth 2") == 2
    assert "https://archive.ubuntu.com/ubuntu" in dockerfile
    assert "https://security.ubuntu.com/ubuntu" in dockerfile
    assert "http://(archive|security)\\.ubuntu\\.com/ubuntu" in dockerfile


def test_thin_entrypoint_uses_installed_absolute_worker_executable() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "/opt/oscar-venv/bin/blueprint-run-robot-eval-worker" in text
    assert "set -- blueprint-run-robot-eval-worker" not in text
    assert (
        "test -x /opt/oscar-venv/bin/blueprint-run-robot-eval-worker"
        in RELEASE.read_text(encoding="utf-8")
    )


def test_release_asset_modes_are_explicit_and_fail_closed() -> None:
    dockerfile = RELEASE.read_text(encoding="utf-8")
    entrypoint = ENTRYPOINT.read_text(encoding="utf-8")
    assert "ARG FOUNDATION_MODEL_ASSETS=external" in dockerfile
    assert "external) test ! -e /opt/blueprint/ckpts" in dockerfile
    assert "embedded)" in dockerfile
    assert "BLUEPRINT_GROOT_OSCAR_FOUNDATION_MODEL_ASSETS" in dockerfile
    assert 'model_asset_mode="${BLUEPRINT_GROOT_OSCAR_FOUNDATION_MODEL_ASSETS:-external}"' in entrypoint
    assert 'elif [[ "$model_asset_mode" == "embedded" ]]' in entrypoint
    assert "groot_oscar_closed_loop_image_healthcheck.py --require-cuda" in entrypoint
    assert "groot_oscar_closed_loop_image_healthcheck.py --build-time" not in entrypoint
    assert "invalid BLUEPRINT_GROOT_OSCAR_FOUNDATION_MODEL_ASSETS" in entrypoint
    healthcheck = HEALTHCHECK.read_text(encoding="utf-8")
    assert "BLUEPRINT_GROOT_OSCAR_FOUNDATION_MODEL_ASSETS" in healthcheck
    assert 'and model_asset_mode == "external"' not in healthcheck
    assert "rm -rf /opt/wbc/gear_sonic_deploy/build" in dockerfile
    assert "BLUEPRINT_WORKER_IMAGE_VARIANT=groot-oscar-thin-release" in dockerfile
    assert (
        'BLUEPRINT_GROOT_OSCAR_FOUNDATION_MODEL_ASSETS="${FOUNDATION_MODEL_ASSETS}"'
        in dockerfile
    )


def test_release_source_overlay_does_not_require_foundation_venv_pip() -> None:
    dockerfile = RELEASE.read_text(encoding="utf-8")

    assert "/opt/oscar-venv/bin/python -m pip install" not in dockerfile
    assert "/opt/gr00t-venv/bin/python -m pip install" not in dockerfile
    assert "/isaac-sim/python.sh -m pip install" not in dockerfile
    assert "/opt/blueprint/release-src" in dockerfile
    assert "blueprint_release_override.pth" in dockerfile
    assert dockerfile.count("blueprint_pipeline.__file__.startswith") == 1
    assert "from blueprint_pipeline.robot_eval_worker import main" in dockerfile
    assert "chmod 0755 /opt/oscar-venv/bin/blueprint-run-robot-eval-worker" in dockerfile


def test_release_repairs_missing_embedded_carrier_opencv_without_mutating_other_dependencies() -> None:
    dockerfile = RELEASE.read_text(encoding="utf-8")
    lock = Path(
        "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/"
        "requirements_embedded_carrier_opencv.lock"
    ).read_text(encoding="utf-8")

    assert "if ! /opt/oscar-venv/bin/python -c 'import cv2'; then" in dockerfile
    assert "/opt/runpod-serverless-venv/bin/python -m pip install" in dockerfile
    assert '--target "${oscar_site_packages}"' in dockerfile
    assert "--no-deps --require-hashes" in dockerfile
    assert "/opt/oscar-venv/bin/python -c 'import cv2; assert cv2.__version__'" in dockerfile
    assert "opencv-python-headless==4.11.0.86" in lock
    assert "sha256:0e0a27c19dd1f40ddff94976cfe43066fbbe9dfbb2ec1907d66c19caef42a57b" in lock
