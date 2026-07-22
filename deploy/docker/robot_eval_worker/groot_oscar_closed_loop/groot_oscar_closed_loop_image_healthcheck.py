#!/usr/bin/env python3
"""Build/runtime healthcheck for the sealed ``blueprint-groot-oscar-eval`` image.

Fail-closed: proves the two Python environments and the baked assets are present
before a paid pod trusts the image. Runs in the MAIN env; the GR00T env is
checked out-of-process against ``/opt/gr00t-venv``.

Claim boundary: this proves build/runtime readiness only — not provider startup,
GR00T inference, WAM quality, or task success.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


OSCAR_REPO = os.environ.get("BLUEPRINT_GROOT_OSCAR_OSCAR_REPO", "/opt/OSCAR")
OSCAR_CHECKPOINT = os.environ.get(
    "BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT", "/opt/blueprint/ckpts/oscar"
)
GROOT_ROOT = os.environ.get("BLUEPRINT_GROOT_OSCAR_GROOT_ROOT", "/opt/gr00t")
GROOT_VENV_PYTHON = os.environ.get(
    "BLUEPRINT_GROOT_OSCAR_GROOT_VENV_PYTHON", "/opt/gr00t-venv/bin/python"
)
SONIC_CHECKPOINT = os.environ.get(
    "BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT", "/opt/blueprint/ckpts/sonic"
)
ISAAC_PYTHON = os.environ.get("BLUEPRINT_ISAAC_PYTHON", "/isaac-sim/python.sh")
G1_USD = os.environ.get(
    "BLUEPRINT_ISAAC_UNITREE_G1_USD",
    "/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd",
)
WBC_ROOT = os.environ.get("BLUEPRINT_GEAR_SONIC_ROOT", "/opt/wbc")
WBC_SOURCE_REVISION = os.environ.get("BLUEPRINT_GEAR_SONIC_SOURCE_REVISION", "")
GEAR_SONIC_CHECKPOINT_REPO = os.environ.get("GEAR_SONIC_CHECKPOINT_REPO", "nvidia/GEAR-SONIC")
GEAR_SONIC_CHECKPOINT_REVISION = os.environ.get(
    "GEAR_SONIC_CHECKPOINT_REVISION",
    "5e22ddc69abcea2a9aafc40536b14c232d3f9d7f",
)
COSMOS_BACKBONE_REPO = os.environ.get("COSMOS_BACKBONE_REPO", "nvidia/Cosmos-Reason2-2B")
COSMOS_BACKBONE_REVISION = os.environ.get("COSMOS_BACKBONE_REVISION", "")

# IMGFIX-004: the Isaac 6 kit creates DerivedDataCache under kit/cache and
# documents/ under kit/data at renderer startup. The 2026-07-12 live A40
# canary wedged because the base image ships these writable only by uid 1234.
# The healthcheck runs as the runtime user (build step 7), so a real
# create/delete probe here fails the BUILD instead of wedging a paid pod.
ISAAC_KIT_RUNTIME_WRITE_DIRS = (
    "/isaac-sim/kit/cache",
    "/isaac-sim/kit/data",
    "/isaac-sim/kit/logs",
)

# Main-env modules the closed loop imports (import name : label).
MAIN_ENV_MODULES = {
    "blueprint_pipeline": "blueprint_pipeline",
    "mujoco": "mujoco",
    "zmq": "pyzmq",
    "msgpack_numpy": "msgpack_numpy",
    "imageio": "imageio",
    "PIL": "pillow",
    "huggingface_hub": "huggingface_hub",
}


def _dir_has_files(path: Path) -> bool:
    return path.is_dir() and any(path.rglob("*"))


def _dir_writable_by_current_user(path: Path) -> bool:
    """Real create/delete probe: os.access lies under CAP_DAC_OVERRIDE/ACLs."""
    if not path.is_dir():
        return False
    probe = path / f".blueprint_write_probe_{os.getpid()}"
    try:
        with open(probe, "x", encoding="utf-8") as handle:
            handle.write("probe")
    except OSError:
        return False
    try:
        probe.unlink()
    except OSError:
        pass
    return True


def compute_asset_binding(env=None) -> tuple[dict, bool]:
    """Same asset-binding semantics as the generic worker healthcheck, fail-closed."""
    try:
        from blueprint_pipeline.isaac_worker_image_healthcheck import (
            asset_binding_check,
        )
    except ImportError:
        return (
            {
                "name": "unitree_g1_asset_binding",
                "status": "blocked",
                "blockers": ["blueprint_pipeline_asset_binding_check_unavailable"],
            },
            False,
        )
    return asset_binding_check(runtime_env=dict(env or os.environ), path_exists=Path.exists)


def build_runtime_metadata(
    *,
    env,
    isaac_imported: bool,
    g1_exists: bool,
    asset_binding_valid: bool,
    official_assets_exist: bool,
    source_commit: str,
    dirty_patch: str,
) -> dict:
    """One shared runtime-metadata schema with the generic worker image.

    ``configured_g1_usd_exists`` (the file is present) and
    ``configured_g1_asset_binding_valid`` (the configured binding resolves)
    are distinct claims and are both always emitted.
    """
    return {
        "image_family": env.get("BLUEPRINT_WORKER_IMAGE_FAMILY"),
        "image_variant": env.get("BLUEPRINT_WORKER_IMAGE_VARIANT"),
        "simulator_family": env.get("BLUEPRINT_SIMULATOR_FRAMEWORK"),
        "simulator_major_version": int(env.get("BLUEPRINT_ISAAC_SIM_MAJOR_VERSION", "0") or 0),
        "blueprint_pipeline_imported": isaac_imported,
        "configured_g1_usd_exists": g1_exists,
        "configured_g1_asset_binding_valid": asset_binding_valid,
        "official_gear_sonic_assets_exist": official_assets_exist,
        "source_commit": source_commit,
        "source_dirty_patch_sha256": dirty_patch,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-time", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    blockers: list[str] = []
    payload: dict[str, Any] = {
        "schema_version": "groot_oscar_closed_loop_image_healthcheck.v1",
        "python": sys.version.split()[0],
        "build_time": args.build_time,
        "raw_secret_values_recorded": False,
    }
    source_commit = os.environ.get("BLUEPRINT_SOURCE_COMMIT", "").strip()
    worker_image_digest = os.environ.get("BLUEPRINT_WORKER_IMAGE_DIGEST", "").strip()
    if "@sha256:" in worker_image_digest:
        worker_image_digest = worker_image_digest.rsplit("@", 1)[1]
    payload["worker_image_digest"] = worker_image_digest
    dirty_patch = os.environ.get("BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256", "").strip()
    if not source_commit:
        blockers.append("blueprint_source_commit_missing")
    if len(dirty_patch) != 64:
        blockers.append("blueprint_source_dirty_patch_sha256_missing")
    if args.require_cuda and re.fullmatch(r"sha256:[0-9a-f]{64}", worker_image_digest) is None:
        blockers.append("blueprint_worker_image_digest_missing_or_invalid")

    # --- main env: torch (from base) ---
    try:
        import torch

        payload["torch_version"] = torch.__version__
        if not torch.__version__.startswith("2.10.0"):
            blockers.append("torch_version_not_2_10_0")
        if "+cu128" not in torch.__version__:
            blockers.append("torch_not_built_for_cu128")
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        if args.require_cuda and not torch.cuda.is_available():
            blockers.append("torch_cuda_unavailable")
    except Exception as exc:  # pragma: no cover - image-only path
        payload["torch_error_type"] = type(exc).__name__
        blockers.append("torch_import_failed")

    # --- main env: closed-loop imports ---
    payload["main_env_imports"] = {}
    for module, label in MAIN_ENV_MODULES.items():
        spec = importlib.util.find_spec(module)
        payload["main_env_imports"][label] = spec is not None
        if spec is None:
            blockers.append(f"{label}_not_importable")

    # ``find_spec`` only proves that the package exists. Import the exact live
    # episode entrypoint so a missing transitive runtime dependency fails the
    # immutable image build instead of consuming a paid GPU allocation.
    try:
        importlib.import_module("blueprint_pipeline.oscar_isaac_closed_loop_eval")
        payload["closed_loop_episode_entrypoint_importable"] = True
    except Exception as exc:  # pragma: no cover - image-only failure path
        payload["closed_loop_episode_entrypoint_importable"] = False
        payload["closed_loop_episode_entrypoint_error_type"] = type(exc).__name__
        blockers.append("closed_loop_episode_entrypoint_not_importable")

    try:
        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # type: ignore
            build_command_message,
            pack_pose_message,
        )

        payload["gear_sonic_zmq_python_runtime_importable"] = bool(
            callable(build_command_message) and callable(pack_pose_message)
        )
    except Exception as exc:  # pragma: no cover - image-only path
        payload["gear_sonic_zmq_python_runtime_importable"] = False
        payload["gear_sonic_zmq_python_runtime_error_type"] = type(exc).__name__
    if not payload["gear_sonic_zmq_python_runtime_importable"]:
        blockers.append("gear_sonic_zmq_python_runtime_not_importable")

    cuda_development_markers: set[str] = set()
    for cuda_root in Path("/usr/local").glob("cuda*"):
        for marker in (cuda_root / "bin/nvcc", cuda_root / "include", cuda_root / "lib64/stubs"):
            if marker.exists():
                cuda_development_markers.add(str(marker.resolve()))
        if cuda_root.is_dir():
            cuda_development_markers.update(
                str(path.resolve()) for path in cuda_root.rglob("*.a") if path.is_file()
            )
    payload["cuda_compiler_or_development_markers"] = sorted(cuda_development_markers)
    payload["cuda_compiler_and_development_files_removed"] = not cuda_development_markers
    if cuda_development_markers:
        blockers.append("cuda_compiler_or_development_files_present")

    # --- OSCAR source on PYTHONPATH ---
    oscar_entrypoint = Path(OSCAR_REPO) / "inference" / "inference_oscar.py"
    payload["oscar_repo"] = OSCAR_REPO
    payload["oscar_inference_entrypoint_present"] = oscar_entrypoint.is_file()
    if not oscar_entrypoint.is_file():
        blockers.append("oscar_inference_entrypoint_missing")
    oscar_dynamic_config = None
    if oscar_entrypoint.is_file():
        oscar_env = os.environ.copy()
        oscar_env["PYTHONPATH"] = os.pathsep.join(
            value for value in (OSCAR_REPO, oscar_env.get("PYTHONPATH", "").strip()) if value
        )
        try:
            oscar_dynamic_config = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import importlib.metadata; "
                        "import inference.inference_oscar; "
                        "from worldsim._src.configs.agibot_control.config import make_config; "
                        "assert importlib.metadata.version('pytest') == '9.1.1'; "
                        "assert make_config() is not None; "
                        "print('BLUEPRINT_OSCAR_DYNAMIC_CONFIG_OK')"
                    ),
                ],
                cwd=OSCAR_REPO,
                env=oscar_env,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            payload["oscar_dynamic_config_error_type"] = type(exc).__name__
    payload["oscar_dynamic_config_constructible"] = bool(
        oscar_dynamic_config is not None and oscar_dynamic_config.returncode == 0
    )
    if oscar_dynamic_config is not None:
        payload["oscar_dynamic_config_returncode"] = oscar_dynamic_config.returncode
        if oscar_dynamic_config.returncode != 0:
            payload["oscar_dynamic_config_stdout_tail"] = oscar_dynamic_config.stdout[-1000:]
            payload["oscar_dynamic_config_stderr_tail"] = oscar_dynamic_config.stderr[-2000:]
    if not payload["oscar_dynamic_config_constructible"]:
        blockers.append("oscar_inference_dynamic_config_not_importable")

    # --- exact Isaac carrier and official G1/WBC assets ---
    isaac_import = subprocess.run(
        [
            ISAAC_PYTHON,
            "-c",
            (
                "import blueprint_pipeline; "
                "from isaacsim import SimulationApp; "
                "app = SimulationApp({'headless': True}); "
                "from isaacsim.core.prims import SingleArticulation; "
                "import blueprint_pipeline.isaac_runtime_task_backend; "
                "print('BLUEPRINT_ISAAC_6_ARTICULATION_API_OK'); "
                "app.close()"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    isaac_imported = isaac_import.returncode == 0
    if not isaac_imported:
        payload["isaac_python_import_stdout_tail"] = isaac_import.stdout[-1000:]
        payload["isaac_python_import_stderr_tail"] = isaac_import.stderr[-2000:]
        blockers.append("isaac_python_blueprint_import_failed")
    g1_exists = Path(G1_USD).is_file()
    if not g1_exists:
        blockers.append("configured_g1_usd_missing")
    wbc_root = Path(WBC_ROOT)
    required_wbc_assets = {
        "robot_model": wbc_root / "gear_sonic_deploy/g1/g1_29dof_with_hand.xml",
        "runtime_binary": wbc_root / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref",
        "policy_encoder": wbc_root / "gear_sonic_deploy/policy/release/model_encoder.onnx",
        "policy_decoder": wbc_root / "gear_sonic_deploy/policy/release/model_decoder.onnx",
        "observation_config": wbc_root / "gear_sonic_deploy/policy/release/observation_config.yaml",
        "planner": wbc_root / "gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx",
    }
    wbc_asset_checks = {
        label: path.is_file() and path.stat().st_size > 0
        for label, path in required_wbc_assets.items()
    }
    payload["gear_sonic_checkpoint_repo"] = GEAR_SONIC_CHECKPOINT_REPO
    payload["gear_sonic_checkpoint_revision"] = GEAR_SONIC_CHECKPOINT_REVISION
    revision_marker = wbc_root / ".blueprint-source-revision"
    sealed_wbc_revision = (
        revision_marker.read_text(encoding="utf-8").strip().lower()
        if revision_marker.is_file()
        else ""
    )
    payload["gear_sonic_source_revision"] = sealed_wbc_revision
    payload["gear_sonic_source_revision_marker"] = str(revision_marker)
    payload["gear_sonic_deployment_assets"] = wbc_asset_checks
    if not WBC_SOURCE_REVISION or sealed_wbc_revision != WBC_SOURCE_REVISION.lower():
        blockers.append("official_gear_sonic_source_revision_mismatch")
    if not GEAR_SONIC_CHECKPOINT_REVISION:
        blockers.append("gear_sonic_checkpoint_revision_missing")
    if not all(wbc_asset_checks.values()):
        blockers.append("official_gear_sonic_deployment_assets_missing")
    wbc_build = wbc_root / "gear_sonic_deploy/build"
    model_asset_mode = os.environ.get(
        "BLUEPRINT_GROOT_OSCAR_FOUNDATION_MODEL_ASSETS", "external"
    )
    thin_release = bool(
        os.environ.get("BLUEPRINT_WORKER_IMAGE_VARIANT") == "groot-oscar-thin-release"
        and model_asset_mode == "external"
    )
    payload["foundation_model_assets"] = model_asset_mode
    payload["gear_sonic_build_tree_absent"] = not wbc_build.exists()
    payload["gear_sonic_build_tree_writable"] = (
        False if thin_release else _dir_writable_by_current_user(wbc_build)
    )
    if thin_release and not payload["gear_sonic_build_tree_absent"]:
        blockers.append("thin_release_contains_official_gear_sonic_build_tree")
    if not thin_release and not payload["gear_sonic_build_tree_writable"]:
        blockers.append("official_gear_sonic_build_tree_not_writable")

    # --- GR00T env (out-of-process) ---
    payload["groot_venv_python"] = GROOT_VENV_PYTHON
    if Path(GROOT_VENV_PYTHON).exists():
        proc = subprocess.run(
            [GROOT_VENV_PYTHON, "-c", "from gr00t.policy.gr00t_policy import Gr00tPolicy"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        payload["groot_policy_importable"] = proc.returncode == 0
        if proc.returncode != 0:
            payload["groot_import_stderr_tail"] = proc.stderr[-500:]
            blockers.append("groot_policy_not_importable")

        processor_proc = subprocess.run(
            [
                GROOT_VENV_PYTHON,
                "-c",
                (
                    "import json; from pathlib import Path; import gr00t.model; "
                    "from gr00t.model.gr00t_n1d7.gr00t_n1d7 import get_backbone_cls; "
                    "from transformers import AutoConfig, AutoProcessor, Qwen3VLProcessor; "
                    f"checkpoint=Path({SONIC_CHECKPOINT!r}); "
                    "cfg=AutoConfig.from_pretrained(str(checkpoint), local_files_only=True, trust_remote_code=True); "
                    "p=Path(cfg.model_name); "
                    "assert p.is_absolute() and p.is_dir(); "
                    "processor_cfg=json.loads((checkpoint/'processor/processor_config.json').read_text()); "
                    "assert processor_cfg['processor_kwargs']['model_name'] == cfg.model_name; "
                    "backbone_cls=get_backbone_cls(cfg); "
                    "assert backbone_cls.__name__ == 'Qwen3Backbone'; "
                    "AutoConfig.from_pretrained(str(p), local_files_only=True, trust_remote_code=True); "
                    "Qwen3VLProcessor.from_pretrained(str(p), local_files_only=True); "
                    "AutoProcessor.from_pretrained(str(checkpoint/'processor'), local_files_only=True); "
                    "print('BLUEPRINT_OFFLINE_GROOT_SELECTOR_AND_COSMOS_PROCESSOR_OK')"
                ),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        payload["groot_checkpoint_selector_offline_constructible"] = processor_proc.returncode == 0
        payload["groot_nested_processor_offline_constructible"] = processor_proc.returncode == 0
        if processor_proc.returncode != 0:
            payload["groot_nested_processor_stderr_tail"] = processor_proc.stderr[-1000:]
            blockers.append(
                "groot_checkpoint_selector_or_nested_processor_not_offline_constructible"
            )
    else:
        payload["groot_policy_importable"] = False
        blockers.append("groot_venv_python_missing")

    # --- baked checkpoints ---
    for label, path in (("sonic", SONIC_CHECKPOINT), ("oscar", OSCAR_CHECKPOINT)):
        present = _dir_has_files(Path(path))
        payload[f"{label}_checkpoint_present"] = present
        payload[f"{label}_checkpoint_path"] = path
        if not present:
            blockers.append(f"{label}_checkpoint_missing")

    try:
        from huggingface_hub import try_to_load_from_cache

        cosmos_config = try_to_load_from_cache(
            COSMOS_BACKBONE_REPO,
            "config.json",
            revision=COSMOS_BACKBONE_REVISION,
        )
        cosmos_cached = isinstance(cosmos_config, str) and Path(cosmos_config).is_file()
        cosmos_default_config = try_to_load_from_cache(
            COSMOS_BACKBONE_REPO,
            "config.json",
            revision="main",
        )
        cosmos_default_is_pinned = (
            cosmos_cached
            and isinstance(cosmos_default_config, str)
            and Path(cosmos_default_config).is_file()
            and Path(cosmos_default_config).resolve() == Path(cosmos_config).resolve()
        )
    except Exception as exc:  # pragma: no cover - image-only path
        payload["cosmos_backbone_cache_error_type"] = type(exc).__name__
        cosmos_cached = False
        cosmos_default_is_pinned = False
    payload["cosmos_backbone_repo"] = COSMOS_BACKBONE_REPO
    payload["cosmos_backbone_revision"] = COSMOS_BACKBONE_REVISION
    payload["cosmos_backbone_config_cached"] = cosmos_cached
    payload["cosmos_backbone_default_ref_pinned"] = cosmos_default_is_pinned
    if not COSMOS_BACKBONE_REVISION:
        blockers.append("cosmos_backbone_revision_missing")
    if not cosmos_cached:
        blockers.append("cosmos_backbone_not_sealed_in_hf_cache")
    if not cosmos_default_is_pinned:
        blockers.append("cosmos_backbone_default_ref_not_pinned")

    # --- IMGFIX-004: kit runtime dirs writable by the executing (runtime) user ---
    payload["isaac_kit_runtime_write_dirs"] = {}
    for kit_dir in ISAAC_KIT_RUNTIME_WRITE_DIRS:
        writable = _dir_writable_by_current_user(Path(kit_dir))
        payload["isaac_kit_runtime_write_dirs"][kit_dir] = writable
        if not writable:
            blockers.append(f"isaac_kit_dir_not_writable:{kit_dir}")

    asset_binding, asset_binding_valid = compute_asset_binding(os.environ)
    payload["unitree_g1_asset_binding"] = asset_binding
    if not asset_binding_valid:
        blockers.append("unitree_g1_asset_binding_invalid")
    payload["status"] = "passed" if not blockers else "blocked"
    payload["blockers"] = blockers
    payload["runtime_metadata"] = build_runtime_metadata(
        env=os.environ,
        isaac_imported=isaac_imported,
        g1_exists=g1_exists,
        asset_binding_valid=asset_binding_valid,
        official_assets_exist=all(wbc_asset_checks.values()),
        source_commit=source_commit,
        dirty_patch=dirty_patch,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
