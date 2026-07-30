"""Fail-closed build/runtime checks for the headless reconstruction worker."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .common import utc_now_iso, write_json
from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "reconstruction_worker_image_healthcheck.v1"
WORKER_FAMILY = "blueprint-reconstruction-worker"
COLMAP_REVISION = "a0d785fba74b2664f31edc4a29026a8b27c00f67"
GSPLAT_REVISION = "937e29912570c372bed6747a5c9bf85fed877bae"
THREEDGRUT_REVISION = "0a5832248698ab8456b181d6ea17fe02eda58637"
MODEL_DIGESTS = {
    "aliked-n16rot.onnx": "39c423d0a6f03d39ec89d3d1d61853765c2fb6a8b8381376c703e5758778a547",
    "aliked-lightglue.onnx": "b9a5de7204648b18a8cf5dcac819f9d30de1a5961ef03756803c8b86c2dceb8d",
    "sift-lightglue.onnx": "e0500228472b43f92b3d36881a09b3310d3b058b56187b246cc7b9ab6429096e",
    "bruteforce-matcher.onnx": "3c1282f96d83f5ffc861a873298d08bbe5219f59af59223f5ceab5c41a182a47",
}
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _default_command(argv: Sequence[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(list(argv), check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        return 127, type(exc).__name__
    return result.returncode, (result.stdout + result.stderr)[:16_384]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_reconstruction_worker_healthcheck(
    *,
    build_time: bool,
    env: Mapping[str, str] | None = None,
    command_runner: Callable[[Sequence[str]], tuple[int, str]] | None = None,
    importer: Callable[[str], Any] | None = None,
    path_exists: Callable[[Path], bool] | None = None,
    file_digest: Callable[[Path], str] | None = None,
    file_text: Callable[[Path], str] | None = None,
) -> dict[str, Any]:
    """Check image contents without reading a dataset or held-out observation."""

    runtime_env = dict(env or os.environ)
    run_command = command_runner or _default_command
    import_name = importer or importlib.import_module
    exists = path_exists or Path.exists
    digest_file = file_digest or _sha256
    read_text = file_text or (lambda path: path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    family = str(runtime_env.get("BLUEPRINT_WORKER_IMAGE_FAMILY") or "")
    family_ok = family == WORKER_FAMILY
    checks.append({"check_id": "worker_family", "status": "passed" if family_ok else "failed"})
    if not family_ok:
        blockers.append("reconstruction_worker_family_invalid")

    source_commit = str(runtime_env.get("BLUEPRINT_SOURCE_COMMIT") or "").lower()
    source_ok = _COMMIT.fullmatch(source_commit) is not None
    checks.append({"check_id": "source_commit", "status": "passed" if source_ok else "failed"})
    if not source_ok:
        blockers.append("reconstruction_worker_source_commit_invalid")

    display_free = not any(runtime_env.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY"))
    checks.append(
        {"check_id": "headless_environment", "status": "passed" if display_free else "failed"}
    )
    if not display_free:
        blockers.append("reconstruction_worker_display_attached")

    revision_paths = {
        "colmap_revision": (Path("/opt/colmap/.blueprint-source-revision"), COLMAP_REVISION),
        "gsplat_revision": (Path("/opt/gsplat/.blueprint-source-revision"), GSPLAT_REVISION),
        "threedgrut_revision": (
            Path("/opt/3dgrut/.blueprint-source-revision"),
            THREEDGRUT_REVISION,
        ),
    }
    for check_id, (path, expected) in revision_paths.items():
        present = bool(exists(path))
        try:
            actual = read_text(path).strip() if present else ""
        except (OSError, UnicodeError):
            actual = ""
        passed = actual == expected
        checks.append({"check_id": check_id, "status": "passed" if passed else "failed"})
        if not passed:
            blockers.append(f"reconstruction_worker_{check_id}_invalid")

    for binary, expected_token in (("ffmpeg", "6.1.1"), ("colmap", "COLMAP")):
        returncode, output = run_command((binary, "-version" if binary == "ffmpeg" else "-h"))
        passed = returncode == 0 and expected_token in output
        checks.append(
            {"check_id": f"{binary}_headless", "status": "passed" if passed else "failed"}
        )
        if not passed:
            blockers.append(f"reconstruction_worker_{binary}_unavailable")

    for module in (
        "torch",
        "onnxruntime",
        "gsplat",
        "threedgrut",
        "numpy",
        "cv2",
        "trimesh",
        "pxr",
    ):
        try:
            import_name(module)
            passed = True
        except Exception:  # noqa: BLE001 - a typed blocker is the public result
            passed = False
        checks.append({"check_id": f"import_{module}", "status": "passed" if passed else "failed"})
        if not passed:
            blockers.append(f"reconstruction_worker_import_failed:{module}")

    model_root = Path(
        runtime_env.get("BLUEPRINT_RECONSTRUCTION_MODEL_ROOT") or "/opt/models/colmap"
    )
    for filename, expected in MODEL_DIGESTS.items():
        path = model_root / filename
        present = bool(exists(path))
        try:
            actual = digest_file(path) if present else ""
        except OSError:
            actual = ""
        passed = actual == expected
        checks.append(
            {"check_id": f"model_digest:{filename}", "status": "passed" if passed else "failed"}
        )
        if not passed:
            blockers.append(f"reconstruction_worker_model_digest_invalid:{filename}")

    if not build_time:
        returncode, output = run_command(
            (
                "nvidia-smi",
                "--query-gpu=driver_version,memory.total,compute_cap",
                "--format=csv,noheader",
            )
        )
        gpu_ok = returncode == 0 and bool(output.strip())
        checks.append({"check_id": "nvidia_runtime", "status": "passed" if gpu_ok else "failed"})
        if not gpu_ok:
            blockers.append("reconstruction_worker_nvidia_runtime_unavailable")

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "status": "passed" if not blockers else "failed",
        "mode": "build_time" if build_time else "gpu_runtime",
        "checks": checks,
        "blockers": sorted(set(blockers)),
        "display_attached": not display_free,
        "runtime_identity": {
            "worker_family": family or None,
            "source_commit_sha": source_commit or None,
            "container_image_digest": (
                str(runtime_env.get("BLUEPRINT_CONTAINER_IMAGE_DIGEST") or "").strip() or None
            ),
        },
        "hidden_heldout_observations_accessed": False,
        "scientific_qualification_inferred": False,
        "proof_effect": "none",
        "claim_ceiling": "worker_image_compatibility_only",
    }
    result["healthcheck_digest"] = canonical_digest(result, digest_field="healthcheck_digest")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-time", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_reconstruction_worker_healthcheck(build_time=args.build_time)
    if args.output:
        write_json(args.output, result)
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
