from __future__ import annotations

from pathlib import Path

from blueprint_pipeline.reconstruction_worker_image_healthcheck import (
    COLMAP_REVISION,
    GSPLAT_REVISION,
    MODEL_DIGESTS,
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
        return 0, "COLMAP 4.1.1"
    if argv[0] == "nvidia-smi":
        return 0, "550.90, 49140 MiB, 8.9"
    return 127, ""


def test_reconstruction_worker_recipe_is_digest_and_revision_pinned():
    dockerfile = (IMAGE_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:" in dockerfile
    assert f"ARG COLMAP_REVISION={COLMAP_REVISION}" in dockerfile
    assert f"ARG GSPLAT_REVISION={GSPLAT_REVISION}" in dockerfile
    assert f"ARG THREEDGRUT_REVISION={THREEDGRUT_REVISION}" in dockerfile
    assert "-DGUI_ENABLED=OFF" in dockerfile
    assert "-DCUDA_ENABLED=ON" in dockerfile
    assert "-DONNX_ENABLED=ON" in dockerfile
    assert "pip install --no-deps --editable /opt/3dgrut" in dockerfile
    assert 'python -c "import threedgrut"' in dockerfile
    assert "sha256sum --check --strict" in dockerfile
    assert ":latest" not in dockerfile
    requirements = (IMAGE_ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
    assert requirements
    assert all("==" in line for line in requirements)


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
