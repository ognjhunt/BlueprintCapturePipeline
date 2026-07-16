"""Image-policy contract for the RunPod GR00T/SONIC persistent session."""

from __future__ import annotations

import os
from typing import Any, Sequence

RUNPOD_UNITREE_GROOT_SONIC_ALLOW_RUNTIME_BOOTSTRAP_ENV = (
    "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_ALLOW_RUNTIME_BOOTSTRAP"
)
RUNPOD_UNITREE_GROOT_SONIC_SEALED_IMAGE_CONFIRMED_ENV = (
    "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_SEALED_IMAGE_CONFIRMED"
)
RUNPOD_UNITREE_GROOT_SONIC_REQUIRE_SEALED_IMAGE_ENV = (
    "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_REQUIRE_SEALED_IMAGE"
)


def _string(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: str | None) -> bool:
    return _string(value).lower() in {"1", "true", "yes", "on"}


def _legacy_int_env(name: str, default: int) -> int:
    try:
        return int(_string(os.getenv(name)) or default)
    except (TypeError, ValueError):
        return default


def runpod_unitree_groot_sonic_image_contract_policy(
    *,
    provider_bundle_kind: str,
    image_name: str,
    bootstrap_mode: str | None,
    external_runtime_bundle_verified: bool = False,
) -> dict[str, Any]:
    is_wam_carrier = _string(provider_bundle_kind).lower() == "wam"
    runtime_bootstrap_allowed = _truthy(
        os.getenv(RUNPOD_UNITREE_GROOT_SONIC_ALLOW_RUNTIME_BOOTSTRAP_ENV)
    )
    sealed_image_confirmed = _truthy(
        os.getenv(RUNPOD_UNITREE_GROOT_SONIC_SEALED_IMAGE_CONFIRMED_ENV)
    )
    require_sealed_image = not runtime_bootstrap_allowed and not external_runtime_bundle_verified
    explicit_require = _string(
        os.getenv(RUNPOD_UNITREE_GROOT_SONIC_REQUIRE_SEALED_IMAGE_ENV)
    )
    if explicit_require:
        require_sealed_image = _truthy(explicit_require)
    bootstrap = _string(bootstrap_mode)
    runtime_bootstrap_mode = bootstrap in {
        "system_python_minimal",
        "system_python",
        "uv_sync",
        "runtime_clone",
    }
    blockers: list[str] = []
    if is_wam_carrier and require_sealed_image and not sealed_image_confirmed:
        blockers.append("runpod_unitree_groot_sonic_wam_carrier_image_not_sealed")
    if (
        is_wam_carrier
        and require_sealed_image
        and runtime_bootstrap_mode
        and not runtime_bootstrap_allowed
    ):
        blockers.append("runpod_unitree_groot_sonic_runtime_bootstrap_disallowed")
    return {
        "schema_version": "runpod_unitree_groot_sonic_image_contract_policy.v1",
        "status": "blocked" if blockers else "allowed",
        "provider_bundle_kind": _string(provider_bundle_kind),
        "image_name": _string(image_name),
        "wam_carrier": is_wam_carrier,
        "bootstrap_mode": bootstrap or None,
        "runtime_bootstrap_mode": runtime_bootstrap_mode,
        "runtime_bootstrap_allowed": runtime_bootstrap_allowed,
        "runtime_bootstrap_override_env": (
            RUNPOD_UNITREE_GROOT_SONIC_ALLOW_RUNTIME_BOOTSTRAP_ENV
        ),
        "require_sealed_image": require_sealed_image,
        "require_sealed_image_env": RUNPOD_UNITREE_GROOT_SONIC_REQUIRE_SEALED_IMAGE_ENV,
        "sealed_image_confirmed": sealed_image_confirmed,
        "external_runtime_bundle_verified": external_runtime_bundle_verified,
        "sealed_image_confirmed_env": RUNPOD_UNITREE_GROOT_SONIC_SEALED_IMAGE_CONFIRMED_ENV,
        "runtime_dependency_install_disallowed_for_paid_launch": bool(
            is_wam_carrier and require_sealed_image and not runtime_bootstrap_allowed
        ),
        "blockers": blockers,
        "blocker": blockers[0] if blockers else None,
        "safe_next_path": (
            "Use a sealed GR00T/SONIC WAM carrier image with sources, compatible Python deps, "
            "server runtime, and checkpoint cache already present; then set "
            f"{RUNPOD_UNITREE_GROOT_SONIC_SEALED_IMAGE_CONFIRMED_ENV}=true. For an explicit "
            "debug run that may install/clone/download in-pod, set "
            f"{RUNPOD_UNITREE_GROOT_SONIC_ALLOW_RUNTIME_BOOTSTRAP_ENV}=true."
        ),
        "claim_boundary": (
            "This is a paid-provider image contract gate. It does not prove policy inference, "
            "WAM rollout quality, semantic task success, or physical robot readiness."
        ),
        "raw_secret_values_recorded": False,
    }


def runpod_unitree_groot_sonic_should_default_to_sealed_bootstrap(
    *,
    provider_bundle_kind: str,
    previous_bootstrap_mode: str | None,
) -> bool:
    if _string(provider_bundle_kind).lower() != "wam":
        return False
    if previous_bootstrap_mode is not None:
        return False
    runtime_bootstrap_allowed = _truthy(
        os.getenv(RUNPOD_UNITREE_GROOT_SONIC_ALLOW_RUNTIME_BOOTSTRAP_ENV)
    )
    if runtime_bootstrap_allowed:
        return False
    require_sealed_image = True
    explicit_require = _string(
        os.getenv(RUNPOD_UNITREE_GROOT_SONIC_REQUIRE_SEALED_IMAGE_ENV)
    )
    if explicit_require:
        require_sealed_image = _truthy(explicit_require)
    return bool(
        require_sealed_image
        and _truthy(os.getenv(RUNPOD_UNITREE_GROOT_SONIC_SEALED_IMAGE_CONFIRMED_ENV))
    )


def resolve_runpod_provider_shape(
    *,
    gpu_type_ids: Sequence[str],
    default_gpu_type_ids: Sequence[str],
    container_disk_gb: int | None,
    volume_gb: int | None,
    allowed_cuda_versions: Sequence[str],
) -> dict[str, Any]:
    """Resolve legacy env defaults while letting admitted campaigns bind exactly."""

    return {
        "gpu_type_ids": tuple(gpu_type_ids) or tuple(default_gpu_type_ids),
        "container_disk_gb": int(
            container_disk_gb
            if container_disk_gb is not None
            else _legacy_int_env(
                "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_DISK_GB", 240
            )
        ),
        "volume_gb": int(
            volume_gb
            if volume_gb is not None
            else _legacy_int_env(
                "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_VOLUME_GB", 120
            )
        ),
        "allowed_cuda_versions": tuple(allowed_cuda_versions),
    }
