from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
IMAGE_DIR = REPO / "deploy" / "docker" / "robot_eval_worker" / "unitree_groot_sonic_wam"


def test_unitree_groot_sonic_wam_dockerfile_seals_runtime_contract() -> None:
    text = (IMAGE_DIR / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM --platform=linux/amd64 ${BASE_IMAGE}" in text
    assert "GROOT_SOURCE_REF=e5749287857afd97b78f1147166137de29746392" in text
    assert "BLUEPRINT_UNITREE_GROOT_N17_SONIC_SKIP_SYSTEM_PYTHON_DEPS_INSTALL=true" in text
    assert "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_SEALED_IMAGE_CONFIRMED=true" in text
    assert "--mount=type=secret,id=hf_token,required=false" in text
    assert "snapshot_download(" in text
    assert "unitree_groot_sonic_wam_image_healthcheck.py --build-time" in text
    assert ":latest" not in text


def test_unitree_groot_sonic_wam_requirements_pin_known_working_pydantic_pair() -> None:
    requirements = (
        IMAGE_DIR / "requirements_unitree_groot_sonic_system_python.txt"
    ).read_text(encoding="utf-8")

    assert "pydantic==2.13.4" in requirements
    assert "pydantic-core==2.46.4" in requirements
    assert "albumentations==1.4.18" in requirements
    assert "albucore==0.0.17" in requirements
    assert "diffusers==0.35.1" in requirements
    assert "scipy==1.15.3" in requirements
    assert "transformers==4.57.3" in requirements
    assert "pyzmq==27.0.1" in requirements


def test_unitree_groot_sonic_wam_build_script_blocks_unversioned_and_low_disk() -> None:
    script = (
        REPO / "scripts" / "build_push_unitree_groot_sonic_wam_image.sh"
    ).read_text(encoding="utf-8")

    assert "missing_BLUEPRINT_RUNPOD_UNITREE_GROOT_SONIC_WAM_IMAGE_REF" in script
    assert "unitree_groot_sonic_wam_image_ref_must_be_versioned" in script
    assert "unitree_groot_sonic_wam_image_ref_refuses_unstable_tag" in script
    assert "insufficient_local_disk_for_unitree_groot_sonic_wam_image_build" in script
    assert "available_free_gib" in script
    assert "required_free_gib" in script
    assert "--secret \"id=hf_token,src=$hf_token_file\"" in script
    assert "BLUEPRINT_ALLOW_RUNPOD_UNITREE_GROOT_SONIC_WAM_IMAGE_PUSH" in script
    assert "unitree_groot_sonic_wam_image_build_manifest.v1" in script
