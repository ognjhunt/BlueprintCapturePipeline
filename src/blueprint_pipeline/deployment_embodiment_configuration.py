"""Inspect an exact robot configuration from the existing signed Website surface.

This is structural preparation, not a task qualification or an execution receipt.
The normal team-run admission still binds the real scene, license and authority.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .core.security_controls import fetch_bounded_https
from .decision_evidence_contracts import cross_runtime_canonical_digest
from .deployment_embodiment_support import DeploymentSupportError, support_catalog
from .team_embodiment import (
    model_structure_blockers,
    read_model_structure,
    validate_embodiment_manifest,
    verify_model_file,
)

SCHEMA = "deployment_embodiment_configuration_preflight.v1"
MAX_MODEL_BYTES = 512 * 1024 * 1024


def _has_external_assets(layer, sdf) -> bool:
    """Inspect authored fields without composing or resolving unbound assets."""
    if layer.GetExternalReferences() or layer.GetExternalAssetDependencies():
        return True

    def asset_value(value):
        if isinstance(value, sdf.AssetPath):
            return bool(value.path)
        if isinstance(value, Mapping):
            return any(asset_value(item) for item in value.values())
        if isinstance(value, (list, tuple, sdf.AssetPathArray)):
            return any(asset_value(item) for item in value)
        return False

    found = False

    def inspect(path):
        nonlocal found
        if found:
            return
        spec = layer.GetObjectAtPath(path)
        if spec is not None:
            found = any(asset_value(spec.GetInfo(key)) for key in spec.ListInfoKeys())

    layer.Traverse(sdf.Path.absoluteRootPath, inspect)
    return found


def validate_catalog_configuration(embodiment_id: str, value: Mapping[str, Any]) -> dict[str, Any]:
    """Exact variant, arm DoF and independent tool declarations; no status promotion."""
    body = next((row for row in support_catalog()["embodiments"]
                 if row["embodiment_id"] == embodiment_id), None)
    if body is None:
        raise DeploymentSupportError("support_native_embodiment_unknown")
    manifest = validate_embodiment_manifest(value)
    if manifest["embodiment_id"] != embodiment_id:
        raise DeploymentSupportError("support_native_embodiment_identity_mismatch")
    expected_form = "bimanual_arm" if body["arm_count"] == 2 else "fixed_base_arm"
    if manifest["form"] != expected_form or manifest["base"]["mode"] != "fixed":
        raise DeploymentSupportError("support_native_form_mismatch")
    if body["arm_dof"] is not None and len(manifest["joints"]) != sum(body["arm_dof"]):
        raise DeploymentSupportError("support_native_arm_joint_count_mismatch")
    if body["arm_count"] == 2 and (manifest["schema_version"] != "blueprint.robot_embodiment.v2"
                                  or len(manifest["grippers"]) != 2):
        raise DeploymentSupportError("independent_bimanual_tool_channels_required")
    return manifest


def _fetch_model(url: str, path: Path) -> None:
    parsed = urlsplit(url)
    fetch_bounded_https(url, allowed_origins=[f"https://{parsed.netloc}"],
                        timeout_seconds=60, max_bytes=MAX_MODEL_BYTES,
                        max_redirects=0, output_path=path)


def inspect_configuration(
    value: Mapping[str, Any], *, fetch: Callable[[str, Path], None] | None = None,
) -> dict[str, Any]:
    """Read the actual digest-bound USD; never accept a browser's scene or a fake run.

    Public, bounded HTTPS only. Private models continue through the existing
    operator-bound storage/staging route rather than copying a secret into a URL.
    Repeating this read creates no run, allocation, notification or durable grant.
    """
    if not isinstance(value, Mapping) or set(value) != {"embodiment_id", "robot_embodiment"}:
        raise DeploymentSupportError("support_native_request_invalid")
    manifest = validate_catalog_configuration(value["embodiment_id"], value["robot_embodiment"])
    with tempfile.TemporaryDirectory(prefix="blueprint-robot-inspection-") as directory:
        path = Path(directory) / "robot.usd"
        (fetch or _fetch_model)(manifest["model"]["url"], path)
        record = verify_model_file(manifest, path)
        # Opening a referenced stage could read unbound local/remote layers. This
        # public preparation route accepts a flattened model only; the existing
        # private source-bundle route retains its own dependency admission.
        from pxr import Sdf

        layer = Sdf.Layer.FindOrOpen(str(path))
        if layer is None or _has_external_assets(layer, Sdf):
            raise DeploymentSupportError("support_native_self_contained_usd_required")
        structure = read_model_structure(path)
        blockers = model_structure_blockers(manifest, structure)
    result = {
        "schema_version": SCHEMA, "claim_ceiling": "configuration_only",
        "status": "requires_setup" if blockers else "configured_not_executed",
        "embodiment_id": manifest["embodiment_id"],
        "request_digest": cross_runtime_canonical_digest(dict(value)),
        "catalog_digest": support_catalog()["catalog_digest"],
        "embodiment_digest": manifest["embodiment_digest"],
        "model": record, "model_file_digest_verified": True,
        "structure": {"joint_count": len(structure["joint_names"]),
                      "rigid_body_count": len(structure["body_paths"])},
        "blockers": blockers,
        "limitations": ["Source provenance and use rights remain required.",
                        "Task fit, controller, reset and independent episode scoring are not established by a model inspection.",
                        "Policy access, runtime qualification and spending authority use the existing run admission."],
        "execution_authorized": False, "provider_mutation_performed": False,
    }
    # Model record contains only identity/size, never the scratch path.
    result["preflight_digest"] = cross_runtime_canonical_digest(result, digest_field="preflight_digest")
    return result
