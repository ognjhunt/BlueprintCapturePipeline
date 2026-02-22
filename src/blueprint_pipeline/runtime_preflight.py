"""Runtime preflight validation for production NuRec swap orchestration."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .common import StageError


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
        }


def _env_any(*keys: str) -> str:
    for key in keys:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return ""


def _parse_provider_chain(raw_chain: str) -> list[str]:
    providers = [part.strip().lower() for part in raw_chain.split(",") if part.strip()]
    return providers or ["image_to_3d", "proxy_box"]


def _is_truthy(raw_value: str) -> bool:
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _import_from_blueprintpipeline(root: Path, module_name: str) -> None:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.append(root_str)
    importlib.import_module(module_name)


def _validate_provider_env(provider_chain: list[str]) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    if "sam3d" in provider_chain:
        host = _env_any("TEXT_SAM3D_API_HOST", "SAM3D_API_HOST", "TEXT_SAM3D_BASE_URL")
        key = _env_any("TEXT_SAM3D_API_KEY", "SAM3D_API_KEY")
        checks.append(
            PreflightCheck(
                "provider_sam3d",
                bool(host and key),
                "sam3d host+key present"
                if host and key
                else "missing SAM3D credentials — set TEXT_SAM3D_API_HOST + TEXT_SAM3D_API_KEY",
            )
        )

    if "hunyuan3d" in provider_chain:
        host = _env_any("TEXT_HUNYUAN_API_HOST", "HUNYUAN_API_HOST", "TEXT_HUNYUAN_BASE_URL")
        key = _env_any("TEXT_HUNYUAN_API_KEY", "HUNYUAN_API_KEY")
        checks.append(
            PreflightCheck(
                "provider_hunyuan3d",
                bool(host and key),
                "hunyuan host+key present"
                if host and key
                else "missing Hunyuan credentials — set TEXT_HUNYUAN_API_HOST + TEXT_HUNYUAN_API_KEY",
            )
        )

    return checks


def _validate_interactive_backend_env() -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    particulate_mode = (os.getenv("PARTICULATE_MODE") or "remote").strip().lower()
    checks.append(
        PreflightCheck(
            "particulate_mode",
            particulate_mode in {"remote", "local", "mock", "skip"},
            f"mode={particulate_mode}",
        )
    )

    if particulate_mode == "remote":
        endpoint = _env_any("PARTICULATE_ENDPOINT")
        checks.append(
            PreflightCheck(
                "particulate_remote_endpoint",
                bool(endpoint),
                "PARTICULATE_ENDPOINT present"
                if endpoint
                else "missing PARTICULATE_ENDPOINT for PARTICULATE_MODE=remote",
            )
        )
    elif particulate_mode == "local":
        endpoint = _env_any("PARTICULATE_LOCAL_ENDPOINT")
        model = _env_any("PARTICULATE_LOCAL_MODEL")
        checks.append(
            PreflightCheck(
                "particulate_local_endpoint",
                bool(endpoint),
                "PARTICULATE_LOCAL_ENDPOINT present"
                if endpoint
                else "missing PARTICULATE_LOCAL_ENDPOINT for PARTICULATE_MODE=local",
            )
        )
        checks.append(
            PreflightCheck(
                "particulate_local_model",
                bool(model),
                "PARTICULATE_LOCAL_MODEL present"
                if model
                else "missing PARTICULATE_LOCAL_MODEL for PARTICULATE_MODE=local",
            )
        )

    articulation_backend = (os.getenv("ARTICULATION_BACKEND") or "auto").strip().lower()
    checks.append(
        PreflightCheck(
            "articulation_backend",
            articulation_backend in {"auto", "particulate", "heuristic", "infinigen", "physx_anything"},
            f"backend={articulation_backend}",
        )
    )
    return checks


def _validate_nurec_worker(worker_mode: str, worker_command: str) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    mode = (worker_mode or "").strip().lower()
    if mode == "local_worker":
        command = (os.getenv("NUREC_PIPELINE_COMMAND") or "").strip()
        skip = (os.getenv("NUREC_SKIP_PIPELINE_COMMAND") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        ok = bool(command or skip)
        checks.append(
            PreflightCheck(
                "nurec_local_worker",
                ok,
                "local worker configured" if ok else "set NUREC_PIPELINE_COMMAND (or NUREC_SKIP_PIPELINE_COMMAND=true)",
            )
        )
    elif mode == "command":
        ok = bool(worker_command.strip())
        checks.append(
            PreflightCheck(
                "nurec_command_worker",
                ok,
                "worker command configured" if ok else "NUREC_WORKER_COMMAND is required for command mode",
            )
        )
    elif mode == "external_markers":
        allow = (os.getenv("ALLOW_EXTERNAL_NUREC_MARKERS") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        checks.append(
            PreflightCheck(
                "nurec_external_markers",
                allow,
                "external marker mode explicitly allowed"
                if allow
                else "set ALLOW_EXTERNAL_NUREC_MARKERS=true to use external marker mode",
            )
        )
    else:
        checks.append(
            PreflightCheck(
                "nurec_worker_mode",
                False,
                f"unsupported NUREC_WORKER_MODE: {worker_mode}",
            )
        )
    return checks


def _validate_swap_policy_path(swap_policy_path: str) -> list[PreflightCheck]:
    path = (swap_policy_path or "").strip()
    if not path:
        return [PreflightCheck("swap_policy_path", True, "using built-in policy defaults")]
    policy_path = Path(path)
    return [
        PreflightCheck(
            "swap_policy_path",
            policy_path.is_file(),
            f"found {policy_path}" if policy_path.is_file() else f"missing {policy_path}",
        )
    ]


def _validate_quality_gate_dependencies(*, advanced_quality_gates_enabled: bool) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    if not advanced_quality_gates_enabled:
        checks.append(
            PreflightCheck(
                "advanced_quality_gate_dependencies",
                True,
                "advanced quality gates disabled",
            )
        )
        return checks
    for dep in ("trimesh", "pybullet"):
        try:
            importlib.import_module(dep)
            checks.append(PreflightCheck(f"dep_{dep}", True, f"{dep} import ok"))
        except Exception as exc:
            checks.append(PreflightCheck(f"dep_{dep}", False, f"{dep} import failed: {exc}"))
    return checks


def _validate_blueprintpipeline_runtime(
    root: Path,
    *,
    standalone_mode: bool,
    completion_mode: str,
    data_gen_enabled: bool = False,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []

    if standalone_mode:
        checks.append(
            PreflightCheck(
                "blueprintpipeline_runtime_mode",
                True,
                "standalone mode enabled; external BlueprintPipeline runtime checks skipped",
            )
        )
        return checks

    checks.append(
        PreflightCheck(
            "blueprintpipeline_root",
            root.exists() and root.is_dir(),
            f"found {root}" if root.exists() and root.is_dir() else f"missing {root}",
        )
    )

    required_scripts = [
        root / "interactive-job/run_interactive_assets.py",
        root / "simready-job/prepare_simready_assets.py",
        root / "usd-assembly-job/assemble_scene.py",
        root / "tools/source_pipeline/adapter.py",
    ]

    normalized_mode = (completion_mode or "best_effort").strip().lower()
    strict_downstream = normalized_mode == "full_required"

    # ``data_gen_enabled`` is retained only for backward compatibility with
    # older callers that still pass this flag.
    if strict_downstream or data_gen_enabled:
        required_scripts.extend([
            root / "replicator-job/generate_replicator_bundle.py",
            root / "variation-asset-pipeline-job/run_variation_asset_pipeline.py",
            root / "genie-sim-export-job/export_to_geniesim.py",
        ])

    for script in required_scripts:
        checks.append(
            PreflightCheck(
                f"script_{script.name}",
                script.is_file(),
                f"found {script}" if script.is_file() else f"missing {script}",
            )
        )

    try:
        _import_from_blueprintpipeline(root, "tools.source_pipeline.adapter")
        _import_from_blueprintpipeline(root, "tools.scene_manifest.loader")
        checks.append(
            PreflightCheck("blueprintpipeline_imports", True, "required BlueprintPipeline modules import")
        )
    except Exception as exc:
        checks.append(
            PreflightCheck(
                "blueprintpipeline_imports",
                False,
                f"failed to import BlueprintPipeline modules: {exc}",
            )
        )

    return checks


def _validate_risky_overrides(
    *,
    standalone_mode: bool,
    completion_mode: str,
    advanced_quality_gates_enabled: bool,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []
    normalized_mode = (completion_mode or "best_effort").strip().lower()
    if normalized_mode not in {"full_required", "best_effort"}:
        normalized_mode = "best_effort"

    heuristic_explicit = _is_truthy(os.getenv("SWAP_INCLUDE_HEURISTIC_AS_EXPLICIT", "false"))
    checks.append(
        PreflightCheck(
            "swap_heuristic_explicit_override",
            not heuristic_explicit,
            "heuristic entries remain non-explicit"
            if not heuristic_explicit
            else "unsafe override: SWAP_INCLUDE_HEURISTIC_AS_EXPLICIT=true",
        )
    )

    preflight_enabled_env = _is_truthy(os.getenv("RUNTIME_PREFLIGHT_ENABLED", "true"))
    checks.append(
        PreflightCheck(
            "runtime_preflight_toggle",
            preflight_enabled_env,
            "RUNTIME_PREFLIGHT_ENABLED=true"
            if preflight_enabled_env
            else "unsafe override: RUNTIME_PREFLIGHT_ENABLED=false",
        )
    )

    strict_mode_ok = not (normalized_mode == "full_required" and standalone_mode)
    checks.append(
        PreflightCheck(
            "full_completion_standalone_guard",
            strict_mode_ok,
            (
                "standalone disabled in full_required mode"
                if strict_mode_ok
                else "unsafe override: PIPELINE_STANDALONE_MODE=true with PIPELINE_COMPLETION_MODE=full_required"
            ),
        )
    )

    strict_quality_ok = not (normalized_mode == "full_required" and not advanced_quality_gates_enabled)
    checks.append(
        PreflightCheck(
            "full_completion_quality_guard",
            strict_quality_ok,
            (
                "advanced quality gates enabled"
                if strict_quality_ok
                else "unsafe override: ADVANCED_QUALITY_GATES_ENABLED=false in full_required mode"
            ),
        )
    )
    return checks


def _validate_data_gen_requirements(*, completion_mode: str, data_gen_enabled: bool = False) -> list[PreflightCheck]:
    """Validate dependencies for strict downstream data-generation stages."""
    checks: list[PreflightCheck] = []
    normalized_mode = (completion_mode or "best_effort").strip().lower()
    strict_downstream = normalized_mode == "full_required"
    if not strict_downstream and not data_gen_enabled:
        checks.append(
            PreflightCheck(
                "data_gen_stack",
                True,
                "downstream data generation not required in best_effort mode",
            )
        )
        return checks

    # Gemini API key — required by variation-asset-pipeline-job.
    gemini_key = _env_any("GOOGLE_GENAI_API_KEY", "GEMINI_API_KEY")
    checks.append(
        PreflightCheck(
            "data_gen_gemini_key",
            bool(gemini_key),
            "Gemini API key present"
            if gemini_key
            else "missing GOOGLE_GENAI_API_KEY for full_required downstream stages",
        )
    )

    # blueprint_sim importability (best-effort)
    try:
        importlib.import_module("blueprint_sim")
        checks.append(
            PreflightCheck("data_gen_blueprint_sim", True, "blueprint_sim importable")
        )
    except Exception:
        checks.append(
            PreflightCheck(
                "data_gen_blueprint_sim",
                False,
                "blueprint_sim not importable — install BlueprintPipeline (pip install -e .)",
            )
        )

    return checks


def validate_runtime_preflight(
    *,
    gcs_root: Path,
    blueprintpipeline_root: Path,
    generation_provider_chain: str,
    swap_policy_path: str,
    nurec_worker_mode: str,
    nurec_worker_command: str,
    advanced_quality_gates_enabled: bool,
    standalone_mode: bool = False,
    completion_mode: str = "best_effort",
    data_gen_enabled: bool = False,
) -> list[PreflightCheck]:
    checks: list[PreflightCheck] = []

    checks.append(
        PreflightCheck(
            "gcs_root",
            gcs_root.exists() and gcs_root.is_dir(),
            f"found {gcs_root}" if gcs_root.exists() and gcs_root.is_dir() else f"missing {gcs_root}",
        )
    )

    checks.extend(
        _validate_blueprintpipeline_runtime(
            blueprintpipeline_root,
            standalone_mode=standalone_mode,
            completion_mode=completion_mode,
            data_gen_enabled=data_gen_enabled,
        )
    )
    checks.extend(_validate_provider_env(_parse_provider_chain(generation_provider_chain)))
    checks.extend(_validate_swap_policy_path(swap_policy_path))
    checks.extend(_validate_interactive_backend_env())
    checks.extend(_validate_nurec_worker(nurec_worker_mode, nurec_worker_command))
    checks.extend(
        _validate_risky_overrides(
            standalone_mode=standalone_mode,
            completion_mode=completion_mode,
            advanced_quality_gates_enabled=advanced_quality_gates_enabled,
        )
    )
    checks.extend(
        _validate_quality_gate_dependencies(
            advanced_quality_gates_enabled=advanced_quality_gates_enabled
        )
    )
    checks.extend(
        _validate_data_gen_requirements(
            completion_mode=completion_mode,
            data_gen_enabled=data_gen_enabled,
        )
    )

    return checks


def enforce_preflight(checks: list[PreflightCheck]) -> None:
    failed = [check for check in checks if not check.passed]
    if not failed:
        return
    messages = "; ".join(f"{check.name}: {check.detail}" for check in failed)
    raise StageError("runtime_preflight", messages)
