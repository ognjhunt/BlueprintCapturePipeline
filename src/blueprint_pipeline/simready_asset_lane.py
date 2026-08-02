"""On-demand SimReady object lane for captured scenes.

Converts a captured, segmented task object into a physics-ready draft asset so
a 3DGS scene becomes interactable without world-model physics: the splat stays
the appearance layer, and per-object physics slots swap to SimReady assets.

Doctrine (from the measurement-routing research, enforced structurally):

- generation accelerates authoring, never truth: every generated collider,
  mass, friction, or articulation value is flagged ``estimated`` and every
  candidate evidence record enters site evidence ``validated=False`` until the
  existing collider-qualification / articulation-measurement /
  material-identification gates pass;
- a SimReady draft never grants physics authority, qualification, routing
  eligibility, or execution authorization;
- external providers (Lightwheel-class, usd-content-agents with a remote VLM
  backend) are planned behind explicit R2 rights/retention gates and this
  module performs no live calls;
- environment capabilities are probed per instance (for example the VHACD
  decomposition binary) and absences are recorded as typed fallbacks, never
  silently.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from typing import Any, Mapping, Sequence

from .measurement_method_research_catalog import research_intake_catalog
from .task_site_measurement_routing import validate_site_evidence_profile


SIMREADY_REQUEST_SCHEMA_VERSION = "simready_asset_request.v1"
SIMREADY_MANIFEST_SCHEMA_VERSION = "simready_asset_manifest.v1"
SIMREADY_PROVIDER_PLAN_SCHEMA_VERSION = "simready_provider_generation_plan.v1"
SIMREADY_SCENE_BINDING_SCHEMA_VERSION = "simready_scene_binding.v1"
SIMREADY_PREFLIGHT_TOOLCHAIN_SCHEMA_VERSION = "simready_preflight_toolchain_probe.v1"

GENERATION_MODES = frozenset({"local_geometry_pipeline", "external_provider"})
TARGET_FORMATS = frozenset({"mjcf", "usd"})

# Density and friction classes are coarse authoring priors, never material
# identification. Values are kg/m^3 and dimensionless sliding friction.
DENSITY_CLASSES: dict[str, float] = {
    "cardboard_paper": 300.0,
    "wood": 700.0,
    "rigid_plastic": 950.0,
    "ceramic_glass": 2300.0,
    "aluminum": 2700.0,
    "steel": 7850.0,
}
FRICTION_CLASS_ESTIMATES: dict[str, float] = {
    "cardboard_paper": 0.50,
    "wood": 0.45,
    "rigid_plastic": 0.35,
    "ceramic_glass": 0.40,
    "aluminum": 0.40,
    "steel": 0.50,
}

PROVIDER_R2_GATES = (
    "commercial_use_terms",
    "data_retention_terms",
    "training_use_prohibition",
    "provenance_export",
    "output_portability",
    "subprocessor_regions",
)

_CANDIDATE_EVIDENCE_IDS = ("validated_collider", "mass_inertia", "friction_contact")


class SimReadyAssetLaneError(ValueError):
    def __init__(self, *codes: str):
        self.codes = tuple(sorted(set(code for code in codes if code)))
        super().__init__("; ".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise SimReadyAssetLaneError("simready_artifact_not_json") from exc
    if not isinstance(result, dict):
        raise SimReadyAssetLaneError("simready_artifact_not_object")
    return result


def _digest(value: Mapping[str, Any], field: str) -> str:
    normalized = {key: item for key, item in value.items() if key != field}
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _text_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def validate_simready_asset_request(value: Mapping[str, Any]) -> dict[str, Any]:
    request = _clone(value)
    errors: list[str] = []
    if request.get("schema_version") != SIMREADY_REQUEST_SCHEMA_VERSION:
        errors.append("simready_request_schema_invalid")
    for key in ("request_id", "object_id", "bundle_id"):
        if not _string(request.get(key)):
            errors.append(f"simready_request_{key}_missing")
    if not _string(request.get("bundle_hash")).startswith("sha256:"):
        errors.append("simready_request_bundle_hash_invalid")
    sources = request.get("source_references")
    sources = dict(sources) if isinstance(sources, Mapping) else {}
    if not (_string(sources.get("mesh_record_id")) or _string(sources.get("segmentation_record_id"))):
        errors.append("simready_request_geometry_source_missing")
    if not _string(sources.get("provenance_record_id")):
        errors.append("simready_request_provenance_missing")
    formats = request.get("target_formats")
    if (
        not isinstance(formats, list)
        or not formats
        or any(_string(item) not in TARGET_FORMATS for item in formats)
    ):
        errors.append("simready_request_target_formats_invalid")
    mode = _string(request.get("generation_mode"))
    if mode not in GENERATION_MODES:
        errors.append("simready_request_generation_mode_invalid")
    if mode == "external_provider" and not _string(request.get("provider_candidate_id")):
        errors.append("simready_request_provider_candidate_missing")
    rights = request.get("rights_gates")
    rights = dict(rights) if isinstance(rights, Mapping) else {}
    if rights.get("provider_training_use_allowed") is not False:
        errors.append("simready_request_training_use_must_be_prohibited")
    if not isinstance(rights.get("data_retention_allowed"), bool):
        errors.append("simready_request_retention_gate_missing")
    density_class = _string(request.get("density_class"))
    if density_class not in DENSITY_CLASSES:
        errors.append(f"simready_request_density_class_unknown:{density_class or 'missing'}")
    if errors:
        raise SimReadyAssetLaneError(*errors)
    request["source_references"] = sources
    request["rights_gates"] = rights
    request["simready_request_digest"] = _digest(request, "simready_request_digest")
    return request


def build_simready_asset_request(
    *,
    request_id: str,
    object_id: str,
    bundle_id: str,
    bundle_hash: str,
    source_references: Mapping[str, Any],
    density_class: str,
    generation_mode: str = "local_geometry_pipeline",
    provider_candidate_id: str = "",
    target_formats: Sequence[str] = ("mjcf", "usd"),
    data_retention_allowed: bool = False,
) -> dict[str, Any]:
    return validate_simready_asset_request(
        {
            "schema_version": SIMREADY_REQUEST_SCHEMA_VERSION,
            "request_id": request_id,
            "object_id": object_id,
            "bundle_id": bundle_id,
            "bundle_hash": bundle_hash,
            "source_references": dict(source_references),
            "target_formats": list(target_formats),
            "generation_mode": generation_mode,
            "provider_candidate_id": provider_candidate_id,
            "density_class": density_class,
            "rights_gates": {
                "provider_training_use_allowed": False,
                "data_retention_allowed": data_retention_allowed,
            },
        }
    )


def _mjcf_from_parts(
    object_id: str,
    parts: Sequence[Mapping[str, Any]],
    mass_kg: float,
    friction: float,
) -> str:
    assets: list[str] = []
    geoms: list[str] = []
    part_mass = mass_kg / max(1, len(parts))
    for index, part in enumerate(parts):
        vertex_text = " ".join(f"{value:.6f}" for value in part["vertices_flat"])
        face_text = " ".join(str(value) for value in part["faces_flat"])
        assets.append(
            f'<mesh name="{object_id}-part-{index}" vertex="{vertex_text}" face="{face_text}"/>'
        )
        geoms.append(
            f'<geom name="{object_id}-collider-{index}" type="mesh" '
            f'mesh="{object_id}-part-{index}" mass="{part_mass:.6f}" '
            f'friction="{friction:.4f} 0.005 0.0001"/>'
        )
    newline = "\n      "
    return f"""
<mujoco model="simready-{object_id}">
  <asset>
    {newline.join(assets)}
  </asset>
  <worldbody>
    <body name="{object_id}" pos="0 0 0">
      <freejoint/>
      {newline.join(geoms)}
    </body>
  </worldbody>
</mujoco>
""".strip()


def _usd_prim_name(object_id: str) -> str:
    sanitized = "".join(
        char if char.isalnum() or char == "_" else "_" for char in object_id
    )
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"object_{sanitized}"
    return sanitized


def _usd_from_mesh(object_id: str, vertices: Any, faces: Any) -> str | None:
    try:
        from pxr import Usd, UsdGeom  # noqa: PLC0415 - probed optional export
    except ImportError:
        return None
    prim_name = _usd_prim_name(object_id)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, f"/{prim_name}")
    mesh = UsdGeom.Mesh.Define(stage, f"/{prim_name}/geometry")
    mesh.CreatePointsAttr([tuple(float(v) for v in row) for row in vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr([int(i) for row in faces for i in row])
    return stage.GetRootLayer().ExportToString()


def generate_simready_asset_draft(
    request_value: Mapping[str, Any],
    *,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    generated_on: str,
) -> dict[str, Any]:
    """Local geometry pipeline: segmented mesh -> watertight repair -> convex
    collider parts -> density-class mass/friction estimates -> MJCF/USD draft.

    The result is an authoring draft. Its estimates are flagged, its evidence
    records are ``validated=False``, and it grants no physics authority.
    """

    request = validate_simready_asset_request(request_value)
    if request["generation_mode"] != "local_geometry_pipeline":
        raise SimReadyAssetLaneError("simready_external_generation_requires_provider_plan")
    if not _string(generated_on):
        raise SimReadyAssetLaneError("simready_generated_on_missing")
    try:
        import numpy as np  # noqa: PLC0415
        import trimesh  # noqa: PLC0415
    except ImportError as exc:
        raise SimReadyAssetLaneError("simready_local_toolchain_unavailable") from exc

    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
    )
    if mesh.vertices.shape[0] < 4 or mesh.faces.shape[0] < 4:
        raise SimReadyAssetLaneError("simready_source_mesh_degenerate")
    repair_method = "already_watertight"
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        repair_method = "fill_holes"
    if not mesh.is_watertight:
        mesh = mesh.convex_hull
        repair_method = "convex_hull_replacement"

    vhacd_binary = shutil.which("TestVHACD") or shutil.which("vhacd")
    if vhacd_binary:
        decomposition_method = "vhacd"
        try:
            raw_parts = trimesh.decomposition.convex_decomposition(mesh)
            part_meshes = [
                trimesh.Trimesh(vertices=part["vertices"], faces=part["faces"])
                if isinstance(part, Mapping)
                else part
                for part in raw_parts
            ]
        except BaseException:  # noqa: BLE001 - typed fallback, never a crash
            decomposition_method = "convex_hull_fallback_vhacd_failed"
            part_meshes = [mesh.convex_hull]
    else:
        decomposition_method = "convex_hull_fallback_vhacd_unavailable"
        part_meshes = [mesh.convex_hull]

    density_class = request["density_class"]
    density = DENSITY_CLASSES[density_class]
    volume = float(sum(abs(part.volume) for part in part_meshes))
    if volume <= 0:
        raise SimReadyAssetLaneError("simready_collider_volume_degenerate")
    mass = volume * density
    friction = FRICTION_CLASS_ESTIMATES[density_class]

    parts_payload = []
    for part in part_meshes:
        part_vertices = np.asarray(part.vertices, dtype=np.float64)
        part_faces = np.asarray(part.faces, dtype=np.int64)
        parts_payload.append(
            {
                "vertex_count": int(part_vertices.shape[0]),
                "face_count": int(part_faces.shape[0]),
                "volume_m3": float(abs(part.volume)),
                "vertices_flat": [float(v) for v in part_vertices.reshape(-1)],
                "faces_flat": [int(i) for i in part_faces.reshape(-1)],
            }
        )

    mjcf_xml = _mjcf_from_parts(request["object_id"], parts_payload, mass, friction)
    exports: dict[str, Any] = {
        "mjcf": {"content": mjcf_xml, "content_digest": _text_digest(mjcf_xml)},
    }
    if "usd" in request["target_formats"]:
        usd_text = _usd_from_mesh(request["object_id"], mesh.vertices, mesh.faces)
        exports["usd"] = (
            {"content": usd_text, "content_digest": _text_digest(usd_text)}
            if usd_text is not None
            else {"content": None, "unavailable_reason": "pxr_not_installed"}
        )

    asset_id = f"simready-{request['object_id']}-{request['simready_request_digest'][-12:]}"
    manifest = {
        "schema_version": SIMREADY_MANIFEST_SCHEMA_VERSION,
        "asset_id": asset_id,
        "object_id": request["object_id"],
        "simready_request_digest": request["simready_request_digest"],
        "bundle_id": request["bundle_id"],
        "bundle_hash": request["bundle_hash"],
        "generated_on": generated_on,
        "generator": {
            "generator_id": "blueprint-local-simready-geometry-pipeline",
            "generator_version": "1",
            "repair_method": repair_method,
            "decomposition_method": decomposition_method,
            "vhacd_binary_probed": bool(vhacd_binary),
        },
        "geometry": {
            "vertex_count": int(mesh.vertices.shape[0]),
            "face_count": int(mesh.faces.shape[0]),
            "watertight": bool(mesh.is_watertight),
            "volume_m3": volume,
        },
        "colliders": [
            {key: value for key, value in part.items() if not key.endswith("_flat")}
            for part in parts_payload
        ],
        "mass_estimate": {
            "value_kg": mass,
            "method": "density_class_volume_product",
            "density_class": density_class,
            "density_kg_m3": density,
            "estimated": True,
        },
        "friction_estimate": {
            "value": friction,
            "method": "density_class_lookup",
            "estimated": True,
        },
        "articulation": {"inferred": False, "estimated": True},
        "exports": exports,
        "candidate_site_evidence": [
            {
                "evidence_id": evidence_id,
                "available": True,
                "validated": False,
                "record_id": f"{asset_id}-{evidence_id}",
                "derived_from": "simready_draft",
            }
            for evidence_id in _CANDIDATE_EVIDENCE_IDS
        ],
        "validation": {
            "validated": False,
            "qualification_status": "unvalidated_candidate",
            "collider_qualification_required": True,
            "articulation_measurement_required": False,
            "material_identification_required": True,
        },
        "physics_authority_granted": False,
        "routing_eligibility_granted": False,
        "execution_authorized": False,
    }
    manifest["simready_asset_digest"] = _digest(manifest, "simready_asset_digest")
    return validate_simready_asset_manifest(manifest)


def validate_simready_asset_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _clone(value)
    errors: list[str] = []
    if manifest.get("schema_version") != SIMREADY_MANIFEST_SCHEMA_VERSION:
        errors.append("simready_manifest_schema_invalid")
    for key in ("asset_id", "object_id", "simready_request_digest", "generated_on"):
        if not _string(manifest.get(key)):
            errors.append(f"simready_manifest_{key}_missing")
    for estimate_key in ("mass_estimate", "friction_estimate", "articulation"):
        estimate = manifest.get(estimate_key)
        if not isinstance(estimate, Mapping) or estimate.get("estimated") is not True:
            errors.append(f"simready_manifest_{estimate_key}_must_be_flagged_estimated")
    validation = manifest.get("validation")
    validation = dict(validation) if isinstance(validation, Mapping) else {}
    if validation.get("validated") is not False:
        errors.append("simready_manifest_validation_must_be_false")
    if validation.get("qualification_status") != "unvalidated_candidate":
        errors.append("simready_manifest_qualification_status_invalid")
    for key in (
        "physics_authority_granted",
        "routing_eligibility_granted",
        "execution_authorized",
    ):
        if manifest.get(key) is not False:
            errors.append(f"simready_manifest_{key}_must_be_false")
    evidence = manifest.get("candidate_site_evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("simready_manifest_candidate_evidence_missing")
    else:
        for row in evidence:
            if not isinstance(row, Mapping) or row.get("validated") is not False:
                errors.append("simready_manifest_candidate_evidence_must_be_unvalidated")
                break
    expected = _digest(manifest, "simready_asset_digest")
    supplied = manifest.get("simready_asset_digest")
    if supplied is not None and supplied != expected:
        errors.append("simready_manifest_digest_mismatch")
    if errors:
        raise SimReadyAssetLaneError(*errors)
    manifest["simready_asset_digest"] = expected
    return manifest


def plan_external_simready_generation(
    request_value: Mapping[str, Any],
    *,
    resolved_gates: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Plan (never perform) an external-provider SimReady generation.

    Every R2 gate must carry an explicit contract/record reference to count as
    resolved; anything else blocks. No network call is made on any path.
    """

    request = validate_simready_asset_request(request_value)
    if request["generation_mode"] != "external_provider":
        raise SimReadyAssetLaneError("simready_provider_plan_requires_external_mode")
    provider_id = request["provider_candidate_id"]
    catalog = {row["candidate_id"]: row for row in research_intake_catalog()}
    provider = catalog.get(provider_id)
    if provider is None:
        raise SimReadyAssetLaneError(f"simready_provider_unknown:{provider_id}")
    resolutions = {
        _string(key): _string(reference)
        for key, reference in dict(resolved_gates or {}).items()
        if _string(reference)
    }
    unresolved = [gate for gate in PROVIDER_R2_GATES if gate not in resolutions]
    plan = {
        "schema_version": SIMREADY_PROVIDER_PLAN_SCHEMA_VERSION,
        "simready_request_digest": request["simready_request_digest"],
        "provider_candidate_id": provider_id,
        "provider_classification": provider["classification"],
        "provider_access": dict(provider["access"]),
        "required_r2_gates": list(PROVIDER_R2_GATES),
        "resolved_r2_gates": resolutions,
        "unresolved_r2_gates": unresolved,
        "status": (
            "blocked_r2_gates_unresolved"
            if unresolved
            else "gates_resolved_pending_r3_adapter_admission"
        ),
        "live_call_performed": False,
        "network_used": False,
        "provider_output_would_enter_evidence_validated": False,
        "agent_may_resolve_gates": False,
    }
    plan["simready_provider_plan_digest"] = _digest(plan, "simready_provider_plan_digest")
    return plan


def merge_simready_candidate_evidence(
    site_profile_value: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Add SimReady candidate evidence to a site profile, fail-closed.

    Records are only added for evidence identifiers the site does not already
    carry, always with ``validated=False``. Existing records are never
    overwritten and never downgraded or upgraded.
    """

    site = validate_site_evidence_profile(site_profile_value)
    merged = _clone(site)
    merged.pop("site_evidence_digest", None)
    evidence = dict(merged.get("evidence") or {})
    for manifest_value in manifests:
        manifest = validate_simready_asset_manifest(manifest_value)
        if manifest["bundle_id"] != site["bundle_id"] or (
            manifest["bundle_hash"] != site["bundle_hash"]
        ):
            raise SimReadyAssetLaneError("simready_merge_bundle_binding_mismatch")
        for row in manifest["candidate_site_evidence"]:
            evidence_id = row["evidence_id"]
            if evidence_id in evidence:
                continue
            evidence[evidence_id] = {
                "available": True,
                "validated": False,
                "record_id": row["record_id"],
                "derived_from": "simready_draft",
                "simready_asset_digest": manifest["simready_asset_digest"],
            }
    merged["evidence"] = evidence
    return validate_site_evidence_profile(merged)


def compose_simready_scene_binding(
    site_profile_value: Mapping[str, Any],
    manifests: Sequence[Mapping[str, Any]],
    *,
    gaussian_object_partitions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Bind per-object SimReady assets into the captured scene.

    The 3DGS layer remains the appearance source; each object slot points at a
    draft physics asset that is explicitly unvalidated and authority-free. A
    slot is dynamically renderable without a duplicate only when an exact
    Gaussian object partition is supplied. The legacy no-partition form stays
    static-only and says so explicitly.
    """

    site = validate_site_evidence_profile(site_profile_value)
    if not manifests:
        raise SimReadyAssetLaneError("simready_binding_requires_at_least_one_asset")
    evidence = dict(site.get("evidence") or {})
    splat = evidence.get("gaussian_splat_appearance")
    partition_by_object: dict[str, dict[str, Any]] = {}
    if gaussian_object_partitions:
        from .gaussian_object_partition import (  # noqa: PLC0415
            validate_gaussian_object_partition,
            verify_gaussian_object_partition_files,
        )

        for value in gaussian_object_partitions:
            partition = validate_gaussian_object_partition(value)
            object_id = partition["object_id"]
            if object_id in partition_by_object:
                raise SimReadyAssetLaneError(
                    f"simready_binding_duplicate_partition:{object_id}"
                )
            verification = verify_gaussian_object_partition_files(partition)
            if verification["status"] != "passed":
                raise SimReadyAssetLaneError(*verification["errors"])
            partition_by_object[object_id] = partition
    slots = []
    seen: set[str] = set()
    for manifest_value in manifests:
        manifest = validate_simready_asset_manifest(manifest_value)
        if manifest["object_id"] in seen:
            raise SimReadyAssetLaneError(
                f"simready_binding_duplicate_object:{manifest['object_id']}"
            )
        seen.add(manifest["object_id"])
        partition = partition_by_object.get(manifest["object_id"])
        slots.append(
            {
                "object_id": manifest["object_id"],
                "simready_asset_digest": manifest["simready_asset_digest"],
                "physics_source": "simready_draft_unvalidated",
                "appearance_source": (
                    "movable_object_gaussian_partition"
                    if partition is not None
                    else "scene_gaussian_splat_static_only"
                ),
                "gaussian_object_partition_digest": (
                    partition["gaussian_object_partition_digest"]
                    if partition is not None
                    else None
                ),
                "background_gaussian_digest": (
                    partition["artifacts"]["background"]["digest"]
                    if partition is not None
                    else None
                ),
                "object_gaussian_digest": (
                    partition["artifacts"]["object"]["digest"]
                    if partition is not None
                    else None
                ),
                "object_absent_from_static_background": partition is not None,
                "dynamic_object_renderable_without_duplicate": partition is not None,
            }
        )
    unmatched_partitions = sorted(set(partition_by_object) - seen)
    if unmatched_partitions:
        raise SimReadyAssetLaneError(
            "simready_binding_partition_without_asset:" + ",".join(unmatched_partitions)
        )
    binding = {
        "schema_version": SIMREADY_SCENE_BINDING_SCHEMA_VERSION,
        "site_evidence_profile_id": site["profile_id"],
        "site_evidence_digest": site["site_evidence_digest"],
        "appearance_layer": {
            "kind": "gaussian_splat",
            "record_id": (
                dict(splat).get("record_id") if isinstance(splat, Mapping) else None
            ),
            "available": isinstance(splat, Mapping),
        },
        "object_slots": sorted(slots, key=lambda row: row["object_id"]),
        "appearance_stays_3dgs": True,
        "dynamic_rendering_requires_gaussian_object_partition": True,
        "physics_authority_granted": False,
        "world_model_physics_used": False,
        "routing_still_requires_validated_evidence": True,
    }
    binding["simready_scene_binding_digest"] = _digest(
        binding, "simready_scene_binding_digest"
    )
    return binding


SIMREADY_PREFLIGHT_SCHEMA_VERSION = "simready_scene_preflight.v1"


def probe_simready_preflight_toolchain() -> dict[str, Any]:
    """Report the exact local preflight surface without installing or calling it."""

    tools = {
        "trimesh_geometry_checks": {
            "available": importlib.util.find_spec("trimesh") is not None,
            "required_for_local_generation": True,
        },
        "pxr_usd_authoring": {
            "available": importlib.util.find_spec("pxr") is not None,
            "required_for_usd_export": True,
        },
        "mujoco_headless_dynamics": {
            "available": importlib.util.find_spec("mujoco") is not None,
            "required_for_current_preflight": True,
        },
        "blender_headless_validation": {
            "available": shutil.which("blender") is not None,
            "executable": shutil.which("blender"),
            "required_for_current_preflight": False,
            "status_if_unavailable": "typed_optional_validator_unavailable",
        },
        "nvidia_content_agent_validation": {
            "available": shutil.which("validation-agent") is not None,
            "executable": shutil.which("validation-agent"),
            "required_for_current_preflight": False,
            "status_if_unavailable": "typed_optional_validator_unavailable",
        },
    }
    probe = {
        "schema_version": SIMREADY_PREFLIGHT_TOOLCHAIN_SCHEMA_VERSION,
        "tools": tools,
        "current_required_toolchain_available": all(
            row["available"]
            for row in tools.values()
            if row.get("required_for_current_preflight") is True
            or row.get("required_for_local_generation") is True
            or row.get("required_for_usd_export") is True
        ),
        "optional_validator_absence_blocks_current_preflight": False,
        "install_performed": False,
        "network_used": False,
        "provider_call_performed": False,
    }
    probe["simready_preflight_toolchain_probe_digest"] = _digest(
        probe, "simready_preflight_toolchain_probe_digest"
    )
    return probe


def preflight_simready_scene(
    manifests: Sequence[Mapping[str, Any]], *, settle_steps: int = 100
) -> dict[str, Any]:
    """Test SimReady drafts before they head to any simulator.

    Headless load of each asset's MJCF, structural invariants (body present,
    positive mass, at least one collider geom), and a short dynamics probe
    (finite, non-exploding state after ``settle_steps``). A pass means
    *loadable and numerically stable* — never physically valid; validity
    still belongs to the collider/articulation/material qualification gates.
    """

    if not isinstance(settle_steps, int) or isinstance(settle_steps, bool) or not (
        1 <= settle_steps <= 10_000
    ):
        raise SimReadyAssetLaneError("simready_preflight_settle_steps_invalid")
    toolchain_probe = probe_simready_preflight_toolchain()
    try:
        import mujoco  # noqa: PLC0415 - probed engine dependency
        import numpy as np  # noqa: PLC0415
    except ImportError:
        raise SimReadyAssetLaneError("simready_preflight_runtime_unavailable") from None
    rows: list[dict[str, Any]] = []
    for manifest_value in manifests:
        manifest = validate_simready_asset_manifest(manifest_value)
        mjcf = dict(dict(manifest.get("exports") or {}).get("mjcf") or {})
        content = mjcf.get("content")
        row: dict[str, Any] = {
            "asset_id": manifest["asset_id"],
            "object_id": manifest["object_id"],
            "simready_asset_digest": manifest["simready_asset_digest"],
            "loaded": False,
            "structural_checks": {},
            "stability": {},
            "failure_codes": [],
            "passed": False,
        }
        if not isinstance(content, str) or not content.strip():
            row["failure_codes"].append("preflight_mjcf_export_missing")
            rows.append(row)
            continue
        try:
            model = mujoco.MjModel.from_xml_string(content)
        except Exception as exc:  # noqa: BLE001 - typed forwarding
            row["failure_codes"].append(
                f"preflight_mjcf_load_failed:{type(exc).__name__}"
            )
            rows.append(row)
            continue
        row["loaded"] = True
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, manifest["object_id"]
        )
        mass = float(model.body_mass[body_id]) if body_id >= 0 else 0.0
        structural = {
            "object_body_present": body_id >= 0,
            "collider_geom_count": int(model.ngeom),
            "body_mass_kg": mass,
            "mass_positive": mass > 0.0,
        }
        row["structural_checks"] = structural
        if not (structural["object_body_present"] and structural["mass_positive"] and model.ngeom > 0):
            row["failure_codes"].append("preflight_structural_check_failed")
            rows.append(row)
            continue
        data = mujoco.MjData(model)
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        speed = float(np.linalg.norm(data.qvel[:3])) if data.qvel.shape[0] >= 3 else 0.0
        exploded = (not finite) or speed > 1000.0
        row["stability"] = {
            "steps": settle_steps,
            "state_finite": finite,
            "peak_linear_speed_m_s": speed,
            "exploded": exploded,
        }
        if exploded:
            row["failure_codes"].append("preflight_dynamics_unstable")
            rows.append(row)
            continue
        row["passed"] = True
        rows.append(row)
    report = {
        "schema_version": SIMREADY_PREFLIGHT_SCHEMA_VERSION,
        "toolchain_probe": toolchain_probe,
        "assets": rows,
        "asset_count": len(rows),
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "preflight_pass_means_loadable_and_stable_only": True,
        "physical_validity_established": False,
        "qualification_created": False,
    }
    report["simready_preflight_digest"] = _digest(report, "simready_preflight_digest")
    return report


__all__ = [
    "DENSITY_CLASSES", "FRICTION_CLASS_ESTIMATES", "GENERATION_MODES",
    "SIMREADY_PREFLIGHT_SCHEMA_VERSION", "SIMREADY_PREFLIGHT_TOOLCHAIN_SCHEMA_VERSION",
    "preflight_simready_scene", "probe_simready_preflight_toolchain",
    "PROVIDER_R2_GATES", "SIMREADY_MANIFEST_SCHEMA_VERSION",
    "SIMREADY_PROVIDER_PLAN_SCHEMA_VERSION", "SIMREADY_REQUEST_SCHEMA_VERSION",
    "SIMREADY_SCENE_BINDING_SCHEMA_VERSION", "SimReadyAssetLaneError",
    "build_simready_asset_request", "compose_simready_scene_binding",
    "generate_simready_asset_draft", "merge_simready_candidate_evidence",
    "plan_external_simready_generation", "validate_simready_asset_manifest",
    "validate_simready_asset_request",
]
