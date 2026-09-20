"""Author the prepared website splat in the collider's frame, ADP-040/day 28.

Reuse the existing ParticleField writer. A USD transform preserves local
Gaussian covariance and view-dependent radiance while placing the entire field
in the same estimated frame as the collider and separately authored object.
This is format preparation, not another reconstruction or an RTX qualification.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .gaussian_splat_decode import (
    convert_to_standard_ply, find_splat_transform_cli, read_standard_3dgs_ply, splat_decoder_input_suffix,
)
from .local_reconstruction_adapters import _sha256_file
from . import particlefield_usd
from .task_evaluation_splat_render_runtime import runtime_from_environment
from .website_object_observations import _record


def prepare_native_appearance(*, preparation: Mapping[str, Any], base_scene: Mapping[str, Any],
                              output_root: Path, runtime_resolver=runtime_from_environment) -> dict[str, Any]:
    source = Path(base_scene["splat_path"])
    source_digest = _sha256_file(source)
    if (source_digest != base_scene["splat_digest"]
            or source_digest != preparation["binding"]["splat_digest"]
            or preparation.get("digest") != canonical_digest(preparation, digest_field="digest")):
        raise ValueError("website_native_appearance_source_changed")
    coordinate = preparation["coordinate_frame"]
    transform = np.asarray(coordinate["runtime_to_simulator"], dtype=float)
    scale = coordinate["declared_meters_per_unit"]
    sign = -1 if coordinate["declared_up_axis"] == "-Y" else 1
    rotation = (np.array([[1, 0, 0], [0, 0, -sign], [0, sign, 0]])
                if coordinate["declared_up_axis"] in {"Y", "-Y"} else np.eye(3))
    expected = np.eye(4)
    expected[:3, :3] = float(scale) * rotation
    if (coordinate["declared_up_axis"] not in {"Y", "-Y", "Z"} or not np.isfinite(expected).all() or scale <= 0
            or coordinate.get("physical_scale_measured") is not False
            or transform.shape != (4, 4) or not np.allclose(transform, expected, atol=1e-10, rtol=0)):
        raise ValueError("website_native_appearance_frame_invalid")
    identity = {"preparation_digest": preparation["digest"], "source_digest": source_digest,
                "writer_digest": _sha256_file(Path(particlefield_usd.__file__)),
                "adapter_digest": _sha256_file(Path(__file__))}
    root = output_root.resolve() / canonical_digest(identity)[7:]
    receipt_path = root / "appearance.json"
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        if (prior.get("digest") != canonical_digest(prior, digest_field="digest") or prior.get("binding") != identity
                or _record(Path(prior["artifact"]["path"])) != prior["artifact"]):
            raise ValueError("website_native_appearance_cache_changed")
        return prior
    try:
        read_standard_3dgs_ply(source)
        standard = source
        decoder = None
    except ValueError:
        if not splat_decoder_input_suffix(source):
            raise ValueError("website_native_appearance_format_invalid") from None
        runtime = runtime_resolver(repo_root=Path(__file__).resolve().parents[2])
        cli = find_splat_transform_cli(runtime["renderer_root"])
        package = json.loads((cli.parent.parent / "package.json").read_text()) if cli else {}
        if package.get("name") != "@playcanvas/splat-transform" or package.get("version") != "3.2.0":
            raise ValueError("website_native_appearance_decoder_unadmitted")
        standard = root / "decoded.ply"
        decoded = convert_to_standard_ply(source, standard, repo_root=runtime["renderer_root"], node=runtime["node"])
        if decoded.get("status") != "completed":
            raise ValueError("website_native_appearance_decode_failed:" + ";".join(decoded.get("blockers", [])))
        decoder = {"id": package["name"], "version": package["version"],
                   "runtime_digest": runtime["identity"]["runtime_digest"], "output_digest": _sha256_file(standard)}
    artifact = root / "background.usdc"
    authored = particlefield_usd.write_particlefield_usd(standard, artifact,
        expected_source_sha256=_sha256_file(standard), layer_transform_row_major=transform.T.tolist())
    if authored.get("status") != "completed":
        raise ValueError("website_native_appearance_authoring_failed:" + ";".join(authored.get("blockers", [])))
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(artifact))
    prim = stage.GetPrimAtPath(authored["prim_path"])
    actual = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    if (not stage.GetDefaultPrim() or UsdGeom.GetStageUpAxis(stage) != "Z"
            or UsdGeom.GetStageMetersPerUnit(stage) != 1.0 or not np.allclose(actual, transform.T)
            or _sha256_file(source) != source_digest):
        raise ValueError("website_native_appearance_readback_failed")
    value = {"schema_version": "website_native_appearance.v1", "status": "native_appearance_authored",
             "binding": identity, "artifact": _record(artifact), "decoder": decoder,
             "authoring": authored, "runtime_to_simulator": transform.tolist(),
             "source_bytes_unchanged": True, "appearance_removal_performed": False,
             "reconstruction_performed": False, "physical_measurement_proven": False,
             "renderer_qualified": False, "claim_ceiling": "development_only"}
    value["digest"] = canonical_digest(value, digest_field="digest")
    write_json(receipt_path, value)
    return value
