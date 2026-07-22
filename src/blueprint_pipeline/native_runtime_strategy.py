"""Provider-neutral strategy selection for the hosted native runtime.

The runtime-service API is stable; this catalog controls only which synthesis
implementation sits behind it.  Legacy mode names remain accepted as aliases
for one compatibility window, but callers should select a backend by the
neutral ``BLUEPRINT_NATIVE_RUNTIME_BACKEND`` setting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, MutableMapping


NATIVE_RUNTIME_STRATEGY_SCHEMA_VERSION = "native_runtime_strategy_catalog.v1"
NATIVE_RUNTIME_BACKEND_ENV = "BLUEPRINT_NATIVE_RUNTIME_BACKEND"
LEGACY_SYNTHESIS_MODE_ENV = "NATIVE_WORLD_MODEL_SYNTHESIS_MODE"
DEFAULT_NATIVE_RUNTIME_BACKEND = "site_splat"


@dataclass(frozen=True)
class NativeRuntimeStrategy:
    backend_id: str
    synthesis_mode: str
    model_family: str
    render_source: str
    requires_model_runtime: bool
    legacy_backend: bool
    wam_backend_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": NATIVE_RUNTIME_STRATEGY_SCHEMA_VERSION,
            **asdict(self),
            "claim_boundary": {
                "selection_does_not_prove_runtime_execution": True,
                "selection_does_not_prove_generated_media_fidelity": True,
                "selection_does_not_prove_task_or_policy_success": True,
            },
        }


_STRATEGIES: dict[str, NativeRuntimeStrategy] = {
    "site_splat": NativeRuntimeStrategy(
        backend_id="site_splat",
        synthesis_mode="splat_only",
        model_family="site_splat_truthful_preview",
        render_source="truthful_preview_splat",
        requires_model_runtime=False,
        legacy_backend=False,
    ),
    "cosmos_wam": NativeRuntimeStrategy(
        backend_id="cosmos_wam",
        synthesis_mode="cosmos_i2w",
        model_family="cosmos_i2w_native",
        render_source="cosmos_i2w",
        requires_model_runtime=True,
        legacy_backend=True,
        wam_backend_id="cosmos_wam",
    ),
}

_ALIASES = {
    "site_splat": "site_splat",
    "splat_only": "site_splat",
    "cosmos_wam": "cosmos_wam",
    "cosmos_i2w": "cosmos_wam",
}


def _normalize_backend_id(raw_value: object, *, setting_name: str) -> str | None:
    value = str(raw_value or "").strip().lower()
    if not value:
        return None
    backend_id = _ALIASES.get(value)
    if backend_id is None:
        supported = ",".join(sorted(_STRATEGIES))
        raise ValueError(
            f"native_runtime_backend_unknown:{setting_name}:{value}:supported={supported}"
        )
    return backend_id


def native_runtime_strategy_catalog() -> dict[str, dict[str, object]]:
    return {
        backend_id: strategy.to_dict()
        for backend_id, strategy in sorted(_STRATEGIES.items())
    }


def native_runtime_strategy_for_mode(mode: str) -> NativeRuntimeStrategy:
    backend_id = _normalize_backend_id(mode, setting_name="synthesis_mode")
    if backend_id is None:
        raise ValueError("native_runtime_synthesis_mode_required")
    return _STRATEGIES[backend_id]


def resolve_native_runtime_strategy(
    environ: Mapping[str, str],
) -> NativeRuntimeStrategy:
    configured_backend = _normalize_backend_id(
        environ.get(NATIVE_RUNTIME_BACKEND_ENV),
        setting_name=NATIVE_RUNTIME_BACKEND_ENV,
    )
    legacy_backend = _normalize_backend_id(
        environ.get(LEGACY_SYNTHESIS_MODE_ENV),
        setting_name=LEGACY_SYNTHESIS_MODE_ENV,
    )
    if configured_backend and legacy_backend and configured_backend != legacy_backend:
        raise ValueError(
            "native_runtime_backend_conflict:"
            f"{NATIVE_RUNTIME_BACKEND_ENV}={configured_backend}:"
            f"{LEGACY_SYNTHESIS_MODE_ENV}={legacy_backend}"
        )
    backend_id = configured_backend or legacy_backend or DEFAULT_NATIVE_RUNTIME_BACKEND
    return _STRATEGIES[backend_id]


def cosmos_refinement_enabled(
    *,
    strategy: NativeRuntimeStrategy,
    readiness: Mapping[str, object],
    explicit: object,
    truthful_preview: bool,
) -> bool:
    """Resolve legacy Cosmos refinement without ambient-provider activation."""

    setting = str(explicit or "").strip().lower()
    cosmos_ready = bool(readiness.get("cosmos_ready", readiness.get("ready")))
    if setting in {"0", "false", "no", "off"}:
        return False
    if setting in {"1", "true", "yes", "on"}:
        return cosmos_ready
    return strategy.backend_id == "cosmos_wam" and truthful_preview and cosmos_ready


def bind_selected_runtime_identity(
    readiness: MutableMapping[str, object],
    identity: Mapping[str, object],
) -> None:
    """Attach neutral strategy identity to a runtime-readiness payload."""

    readiness["selected_runtime_path"] = identity["selected_runtime_path"]
    readiness["selected_backend_id"] = identity["backend_id"]
