from __future__ import annotations

import hashlib
from pathlib import Path

from blueprint_pipeline.reconstruction_worker_contracts import (
    PINNED_WORKER_COMPONENTS,
    REQUIREMENTS_LOCK_SHA256,
)
from blueprint_pipeline.reconstruction_worker_image_healthcheck import (
    COLMAP_REVISION,
    COLMAP_VERSION,
    CMAKE_VERSION,
    GCC_VERSION,
    GSPLAT_REVISION,
    MODEL_DIGESTS,
    NINJA_VERSION,
    THREEDGRUT_REVISION,
    run_reconstruction_worker_healthcheck,
)


ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = ROOT / "deploy" / "docker" / "reconstruction_worker"


def _env() -> dict[str, str]:
    return {
        "BLUEPRINT_WORKER_IMAGE_FAMILY": "blueprint-reconstruction-worker",
        "BLUEPRINT_SOURCE_COMMIT": "a" * 40,
        "BLUEPRINT_CONTAINER_IMAGE_DIGEST": (
            "registry.example/blueprint/reconstruction@sha256:" + "b" * 64
        ),
        "BLUEPRINT_RECONSTRUCTION_MODEL_ROOT": "/opt/models/colmap",
    }


def _exists(path: Path) -> bool:
    return str(path).startswith(
        ("/opt/colmap/", "/opt/gsplat/", "/opt/3dgrut/", "/opt/models/colmap/")
    )


def _digest(path: Path) -> str:
    return MODEL_DIGESTS[path.name]


def _importer(_name: str) -> object:
    return object()


def _revision(path: Path) -> str:
    return {
        "colmap": COLMAP_REVISION,
        "gsplat": GSPLAT_REVISION,
        "3dgrut": THREEDGRUT_REVISION,
    }[path.parts[2]]


def _command(argv):
    if argv[0] == "ffmpeg":
        return 0, "ffmpeg version 6.1.1"
    if argv[0] == "colmap":
        return 0, f"COLMAP {COLMAP_VERSION}"
    if argv[0] == "nvidia-smi":
        return 0, "550.90, 49140 MiB, 8.9"
    if argv[0] == "gcc":
        return 0, f"gcc (Ubuntu) {GCC_VERSION}"
    if argv[0] == "cmake":
        return 0, f"cmake version {CMAKE_VERSION}"
    if argv[0] == "ninja":
        return 0, NINJA_VERSION
    return 127, ""


def test_reconstruction_worker_recipe_is_digest_and_revision_pinned():
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:" in dockerfile
    assert "ARG PYTHON_VERSION=3.11.9" in dockerfile
    assert (
        "ARG PYTHON_SOURCE_SHA256="
        "9b1e896523fc510691126c864406d9360a3d1e986acbda59cda57b5abda45b87"
        in dockerfile
    )
    assert "python3.11 python3.11-dev" not in dockerfile
    assert "https://www.python.org/ftp/python/${PYTHON_VERSION}/" in dockerfile
    assert "sha256sum --check --strict" in dockerfile
    assert "/opt/python-${PYTHON_VERSION}/bin/python3.11 -m venv /opt/venv" in dockerfile
    stack = next(
        row for row in PINNED_WORKER_COMPONENTS if row["component_id"] == "python_ml_runtime"
    )
    assert "9b1e896523fc510691126c864406d9360a3d1e986acbda59cda57b5abda45b87" in stack[
        "source_revision"
    ]
    assert "ARG FFMPEG_VERSION=6.1.1" in dockerfile
    assert (
        "ARG FFMPEG_SOURCE_SHA256="
        "8684f4b00f94b85461884c3719382f1261f0d9eb3d59640a1f4ac0873616f968"
        in dockerfile
    )
    assert '"https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz"' in dockerfile
    ffmpeg = next(
        row for row in PINNED_WORKER_COMPONENTS if row["component_id"] == "ffmpeg"
    )
    assert "8684f4b00f94b85461884c3719382f1261f0d9eb3d59640a1f4ac0873616f968" in ffmpeg[
        "source_revision"
    ]
    assert f"ARG REQUIREMENTS_LOCK_SHA256={REQUIREMENTS_LOCK_SHA256}" in dockerfile
    assert '"${REQUIREMENTS_LOCK_SHA256}" /opt/blueprint/requirements.lock' in dockerfile
    assert f"ARG COLMAP_REVISION={COLMAP_REVISION}" in dockerfile
    assert f"ARG GSPLAT_REVISION={GSPLAT_REVISION}" in dockerfile
    assert f"ARG THREEDGRUT_REVISION={THREEDGRUT_REVISION}" in dockerfile
    assert "-DGUI_ENABLED=OFF" in dockerfile
    assert "-DCUDA_ENABLED=ON" in dockerfile
    assert "-DONNX_ENABLED=ON" in dockerfile
    assert "cmake==3.28.3" in dockerfile
    assert "ninja==1.11.1.1" in dockerfile
    assert "pip install --no-deps --editable /opt/3dgrut" in dockerfile
    assert 'python -c "import threedgrut"' in dockerfile
    assert "sha256sum --check --strict" in dockerfile
    assert ":latest" not in dockerfile
    readme = (IMAGE_ROOT / "README.md").read_text(encoding="utf-8")
    assert "blueprint_pipeline.reconstruction_gaussian_trainer" in readme
    assert "independent evaluator" in readme
    requirements = (IMAGE_ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
    assert requirements
    assert all("==" in line for line in requirements)
    assert (
        hashlib.sha256((IMAGE_ROOT / "requirements.lock").read_bytes()).hexdigest()
        == REQUIREMENTS_LOCK_SHA256
    )
    for component_id in ("python_ml_runtime", "deterministic_qa"):
        component = next(
            row for row in PINNED_WORKER_COMPONENTS if row["component_id"] == component_id
        )
        assert REQUIREMENTS_LOCK_SHA256 in component["source_revision"]


def test_build_healthcheck_passes_without_display_or_gpu_claim():
    result = run_reconstruction_worker_healthcheck(
        build_time=True,
        env=_env(),
        command_runner=_command,
        importer=_importer,
        path_exists=_exists,
        file_digest=_digest,
        file_text=_revision,
    )
    assert result["status"] == "passed"
    assert result["display_attached"] is False
    assert result["scientific_qualification_inferred"] is False
    assert result["hidden_heldout_observations_accessed"] is False
    assert result["runtime_identity"]["source_commit_sha"] == "a" * 40
    assert result["runtime_identity"]["container_image_digest"].endswith("b" * 64)
    assert result["claim_ceiling"] == "worker_image_compatibility_only"
    assert all(row["check_id"] != "nvidia_runtime" for row in result["checks"])


def test_runtime_healthcheck_requires_gpu_and_fails_closed_on_model_drift():
    result = run_reconstruction_worker_healthcheck(
        build_time=False,
        env={**_env(), "DISPLAY": ":0"},
        command_runner=lambda argv: (1, "") if argv[0] == "nvidia-smi" else _command(argv),
        importer=_importer,
        path_exists=_exists,
        file_digest=lambda path: "0" * 64 if path.name == "aliked-n16rot.onnx" else _digest(path),
        file_text=_revision,
    )
    assert result["status"] == "failed"
    assert "reconstruction_worker_display_attached" in result["blockers"]
    assert "reconstruction_worker_nvidia_runtime_unavailable" in result["blockers"]
    assert "reconstruction_worker_model_digest_invalid:aliked-n16rot.onnx" in result["blockers"]
    assert result["proof_effect"] == "none"


def test_healthcheck_rejects_a_different_colmap_version() -> None:
    result = run_reconstruction_worker_healthcheck(
        build_time=True,
        env=_env(),
        command_runner=lambda argv: (
            (0, "COLMAP 4.1.1") if argv[0] == "colmap" else _command(argv)
        ),
        importer=_importer,
        path_exists=_exists,
        file_digest=_digest,
        file_text=_revision,
    )

    assert result["status"] == "failed"
    assert "reconstruction_worker_colmap_unavailable" in result["blockers"]
