"""Typed appearance render backend contract for the native task arena.

Scene 839873's render-only probe loaded a ParticleField while the policy
canary worker launched Isaac with no backend named at all and inherited a
legacy default.  Both receipts were internally valid and disagreed with each
other.  This module makes the backend a discriminated, digest-bound value that
the render probe, the policy session, the observation-integrity authority and
the Website report must all carry unchanged.

Nothing here renders or converts.  The contract records which renderer and
which conversion produced the asset the arena composes, so a same-pose parity
receipt can be bound to exactly that backend and to nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest

APPEARANCE_RENDER_BACKEND_SCHEMA_VERSION = "appearance_render_backend.v1"

#: Backends that may serve a policy observation once same-pose parity and
#: human review are sealed against them.
BACKEND_NRE_NATIVE_GRPC = "nre_native_grpc"
BACKEND_ISAAC_NATIVE_NUREC = "isaac_native_nurec"
BACKEND_PARTICLEFIELD_3DGRUT_TRANSCODE = "particlefield_3dgrut_transcode"
#: Blueprint's own NuRec-tensor-to-PLY-to-ParticleField conversion.  Proven
#: attribute-identical to the 3DGRUT direct transcode on the arrays it emits
#: (see ``tests/test_particlefield_upstream_parity.py``), but it remains a
#: private reinterpretation of a private container and is therefore a typed
#: development comparator, never a production default.
BACKEND_PARTICLEFIELD_BLUEPRINT_PRIVATE = "particlefield_blueprint_private_tensor_conversion"

PRODUCTION_BACKEND_KINDS = frozenset(
    {
        BACKEND_NRE_NATIVE_GRPC,
        BACKEND_ISAAC_NATIVE_NUREC,
        BACKEND_PARTICLEFIELD_3DGRUT_TRANSCODE,
    }
)
DEVELOPMENT_BACKEND_KINDS = frozenset({BACKEND_PARTICLEFIELD_BLUEPRINT_PRIVATE})
APPEARANCE_RENDER_BACKEND_KINDS = PRODUCTION_BACKEND_KINDS | DEVELOPMENT_BACKEND_KINDS

#: Isaac launch render path each backend composes as.
BACKEND_LAUNCH_RENDER_PATHS: dict[str, str] = {
    BACKEND_ISAAC_NATIVE_NUREC: "plain_nurec_volume",
    BACKEND_PARTICLEFIELD_3DGRUT_TRANSCODE: "particlefield_3d_gaussian_splat",
    BACKEND_PARTICLEFIELD_BLUEPRINT_PRIVATE: "particlefield_3d_gaussian_splat",
}

CAMERA_FRAME_CONTRACTS = frozenset({"nurec_space", "registered_world"})


class AppearanceRenderBackendError(ValueError):
    """Fail-closed backend contract errors."""

    def __init__(self, errors: list[str], *, diagnostics: Mapping[str, Any] | None = None):
        self.errors = tuple(sorted(set(errors)))
        self.diagnostics = dict(diagnostics or {})
        super().__init__(";".join(self.errors))


def _digest_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if not text.startswith("sha256:") or len(text) != len("sha256:") + 64:
        raise AppearanceRenderBackendError(["appearance_render_backend_digest_invalid"])
    return text


def build_appearance_render_backend(
    *,
    kind: str,
    source_asset_digest: str,
    derived_asset_digest: str | None,
    renderer_identity: str,
    conversion_identity: str | None,
    camera_frame_contract: str,
    development_only: bool = False,
) -> dict[str, Any]:
    """Seal one backend choice.

    A development comparator must be declared as such; a production kind must
    not be.  ``derived_asset_digest`` is required whenever a conversion took
    place and forbidden when the source is rendered directly, so the receipt
    cannot describe a conversion that did not happen.
    """

    errors: list[str] = []
    if kind not in APPEARANCE_RENDER_BACKEND_KINDS:
        errors.append("appearance_render_backend_kind_unknown")
    if kind in DEVELOPMENT_BACKEND_KINDS and development_only is not True:
        errors.append("appearance_render_backend_development_kind_requires_declaration")
    if kind in PRODUCTION_BACKEND_KINDS and development_only:
        errors.append("appearance_render_backend_production_kind_declared_development")
    if camera_frame_contract not in CAMERA_FRAME_CONTRACTS:
        errors.append("appearance_render_backend_camera_frame_contract_invalid")
    if not str(renderer_identity or "").strip():
        errors.append("appearance_render_backend_renderer_identity_missing")
    converted = kind in {
        BACKEND_PARTICLEFIELD_3DGRUT_TRANSCODE,
        BACKEND_PARTICLEFIELD_BLUEPRINT_PRIVATE,
    }
    if converted and (derived_asset_digest is None or not str(conversion_identity or "").strip()):
        errors.append("appearance_render_backend_conversion_identity_missing")
    if not converted and (derived_asset_digest is not None or conversion_identity is not None):
        errors.append("appearance_render_backend_conversion_declared_without_conversion")
    if errors:
        raise AppearanceRenderBackendError(errors, diagnostics={"kind": kind})
    contract = {
        "schema_version": APPEARANCE_RENDER_BACKEND_SCHEMA_VERSION,
        "kind": kind,
        "development_only": bool(development_only),
        "source_asset_digest": _digest_or_none(source_asset_digest),
        "derived_asset_digest": _digest_or_none(derived_asset_digest),
        "renderer_identity": str(renderer_identity),
        "conversion_identity": (
            str(conversion_identity) if conversion_identity is not None else None
        ),
        "camera_frame_contract": camera_frame_contract,
        "launch_render_path": BACKEND_LAUNCH_RENDER_PATHS.get(kind),
    }
    if contract["source_asset_digest"] is None:
        raise AppearanceRenderBackendError(["appearance_render_backend_source_digest_missing"])
    contract["receipt_digest"] = canonical_digest(contract, digest_field="receipt_digest")
    return contract


def validate_appearance_render_backend(value: Any) -> dict[str, Any]:
    """Re-validate a sealed backend contract and its digest."""

    if not isinstance(value, Mapping):
        raise AppearanceRenderBackendError(["appearance_render_backend_invalid"])
    contract = dict(value)
    if contract.get("schema_version") != APPEARANCE_RENDER_BACKEND_SCHEMA_VERSION:
        raise AppearanceRenderBackendError(["appearance_render_backend_schema_invalid"])
    rebuilt = build_appearance_render_backend(
        kind=str(contract.get("kind")),
        source_asset_digest=contract.get("source_asset_digest"),
        derived_asset_digest=contract.get("derived_asset_digest"),
        renderer_identity=str(contract.get("renderer_identity") or ""),
        conversion_identity=contract.get("conversion_identity"),
        camera_frame_contract=str(contract.get("camera_frame_contract") or ""),
        development_only=bool(contract.get("development_only")),
    )
    if rebuilt["receipt_digest"] != contract.get("receipt_digest"):
        raise AppearanceRenderBackendError(["appearance_render_backend_receipt_digest_mismatch"])
    return rebuilt


def backend_launch_render_path(contract: Mapping[str, Any]) -> str:
    """The Isaac launch render path a sealed backend composes as."""

    kind = str(contract.get("kind") or "")
    render_path = BACKEND_LAUNCH_RENDER_PATHS.get(kind)
    if render_path is None:
        raise AppearanceRenderBackendError(
            ["appearance_render_backend_not_isaac_composable"], diagnostics={"kind": kind}
        )
    return render_path


__all__ = [
    "APPEARANCE_RENDER_BACKEND_KINDS",
    "APPEARANCE_RENDER_BACKEND_SCHEMA_VERSION",
    "BACKEND_ISAAC_NATIVE_NUREC",
    "BACKEND_LAUNCH_RENDER_PATHS",
    "BACKEND_NRE_NATIVE_GRPC",
    "BACKEND_PARTICLEFIELD_3DGRUT_TRANSCODE",
    "BACKEND_PARTICLEFIELD_BLUEPRINT_PRIVATE",
    "CAMERA_FRAME_CONTRACTS",
    "DEVELOPMENT_BACKEND_KINDS",
    "PRODUCTION_BACKEND_KINDS",
    "AppearanceRenderBackendError",
    "backend_launch_render_path",
    "build_appearance_render_backend",
    "validate_appearance_render_backend",
]
