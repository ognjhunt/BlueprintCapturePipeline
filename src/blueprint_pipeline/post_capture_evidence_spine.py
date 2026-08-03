"""Produce digest-bound prerequisites for a new-site Task Evaluation Run.

This module is deliberately an evidence producer, not another permissive final
compiler.  It accepts native artifacts emitted by the existing ARKit,
reconstruction, target, placement, and measurement-routing lanes, verifies
their bytes and joins, and writes one content-addressed run directory.  A
missing qualification stops the run at the first unavailable gate and records
the smallest measurement needed to continue.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .arkit_raw_contract_validation import (
    SCHEMA_VERSION as ARKIT_RAW_VALIDATION_SCHEMA,
    ArkitRawContractValidationError,
    validate_arkit_raw_contract_validation,
)
from .arkit_depth_surface_compiler import RESULT_SCHEMA as ARKIT_DEPTH_RESULT_SCHEMA
from .arkitscenes_raw_proxy import ARKITSCENES_PROXY_SCHEMA_VERSION
from .decision_evidence_contracts import canonical_digest, canonical_json
from .external_scene_robot_placement import propose_external_scene_robot_placement
from .new_site_task_evaluation_run import (
    AUTHORIZATION_SCHEMA_VERSION,
    EXPECTED_POLICY_CANDIDATES,
    REQUEST_SCHEMA_VERSION as NEW_SITE_REQUEST_SCHEMA,
    compile_new_site_task_evaluation_run,
    select_robot_for_target,
    validate_policy_candidates,
    validate_task_metric,
)
from .rendered_scene_task_target_orchestrator import (
    run_rendered_scene_task_target_pipeline,
)
from .task_site_measurement_routing import (
    MeasurementRoutingError,
    route_task_site_measurement,
)


SOURCE_PROFILE_SCHEMA = "post_capture_source_profile.v1"
GEOMETRY_SCHEMA = "derived_site_geometry.v1"
GEOMETRY_QUALIFICATION_SCHEMA = "site_geometry_qualification.v1"
NATIVE_3DGS_SCHEMA = "native_3dgs_candidate.v1"
CANONICAL_REGISTERED_APPEARANCE_SCHEMA = "canonical_registered_appearance.v1"
CANONICAL_REGISTRATION_MEASUREMENT_SCHEMA = "canonical_3dgs_registration_measurement.v1"
TELEPORT_RUN_RECEIPT_SCHEMA = "teleport_provider_run_receipt.v1"
PROVIDER_SPLAT_IMPORT_RECEIPT_SCHEMA = "provider_splat_import_receipt.v1"
REGISTRATION_QUALIFICATION_SCHEMA = "scene_registration_qualification.v1"
REGISTERED_RECONSTRUCTION_SCHEMA = "registered_site_reconstruction.v1"
ROBOT_SELECTION_SCHEMA = "task_robot_selection.v1"
PLACEMENT_QUALIFICATION_SCHEMA = "qualified_robot_placement.v1"
PLACEMENT_MEASUREMENT_SCHEMA = "robot_placement_qualification_measurement.v1"
SCENE_COMPOSITION_SCHEMA = "qualified_scene_composition.v1"
AUTHORIZATION_DECISION_SCHEMA = "post_capture_policy_execution_decision.v1"
RUN_SCHEMA = "post_capture_evidence_run.v1"
ROUTING_INPUTS_SCHEMA = "post_capture_routing_inputs.v1"

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class PostCaptureEvidenceError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PostCaptureEvidenceError(["post_capture_value_not_json"]) from exc


def _digest(value: Any) -> bool:
    return _DIGEST.fullmatch(str(value or "")) is not None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _finalize(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = _clone(dict(value))
    result[field] = canonical_digest(result, digest_field=field)
    return result


def _validate_artifact(
    value: Any,
    *,
    schema: str,
    digest_field: str,
    code: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PostCaptureEvidenceError([f"{code}_missing"])
    result = _clone(dict(value))
    if (
        result.get("schema_version") != schema
        or result.get(digest_field)
        != canonical_digest(result, digest_field=digest_field)
    ):
        raise PostCaptureEvidenceError([f"{code}_invalid"])
    return result


def _safe_bound_file(root: Path, relative: Any, expected_digest: Any, *, code: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise PostCaptureEvidenceError([f"{code}_path_invalid"])
    candidate = root / relative
    if candidate.is_symlink():
        raise PostCaptureEvidenceError([f"{code}_symlink_forbidden"])
    try:
        resolved_root = root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PostCaptureEvidenceError([f"{code}_missing"] ) from exc
    if resolved_root != resolved and resolved_root not in resolved.parents:
        raise PostCaptureEvidenceError([f"{code}_path_escape"])
    if not resolved.is_file() or _sha256(resolved) != expected_digest:
        raise PostCaptureEvidenceError([f"{code}_digest_mismatch"])
    return resolved


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
            raise PostCaptureEvidenceError(["post_capture_output_conflict"])


def build_source_profile(
    *, source_artifact: Mapping[str, Any], source_root: str | Path
) -> dict[str, Any]:
    """Admit an ARKit proxy or strict Raw V3.2 receipt at its exact ceiling."""

    source = _clone(dict(source_artifact))
    schema = source.get("schema_version")
    root = Path(source_root).expanduser().resolve(strict=True)
    if schema == ARKITSCENES_PROXY_SCHEMA_VERSION:
        digest_field = "arkitscenes_proxy_compilation_digest"
        if source.get(digest_field) != canonical_digest(source, digest_field=digest_field):
            raise PostCaptureEvidenceError(["arkitscenes_source_artifact_digest_mismatch"])
        references = source.get("original_file_references")
        truth_status = "admitted_provider_derived_support"
        source_kind = "arkitscenes_public_dataset_proxy"
        raw_truth = False
        provider_support = True
    elif schema == ARKIT_RAW_VALIDATION_SCHEMA:
        try:
            source = validate_arkit_raw_contract_validation(source)
        except ArkitRawContractValidationError as exc:
            raise PostCaptureEvidenceError(
                [f"raw_v32_source_invalid:{code}" for code in exc.codes]
            ) from exc
        digest_field = "arkit_raw_contract_validation_digest"
        references = source.get("original_file_references_and_digests")
        truth_status = "admitted_blueprint_raw_contract"
        source_kind = "blueprint_arkit_raw_contract_3_2"
        raw_truth = True
        provider_support = False
    else:
        raise PostCaptureEvidenceError(["post_capture_source_schema_unsupported"])
    if not isinstance(references, list) or not references:
        raise PostCaptureEvidenceError(["post_capture_source_references_missing"])
    verified: list[dict[str, Any]] = []
    for index, row in enumerate(references):
        if not isinstance(row, Mapping) or not _digest(row.get("digest")):
            raise PostCaptureEvidenceError(["post_capture_source_reference_invalid"])
        path = _safe_bound_file(
            root,
            row.get("relative_path"),
            row.get("digest"),
            code=f"source_byte_{index}",
        )
        verified.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "digest": row["digest"],
                "size_bytes": path.stat().st_size,
            }
        )
    profile = {
        "schema_version": SOURCE_PROFILE_SCHEMA,
        "status": truth_status,
        "source_kind": source_kind,
        "source_capture_identity": source.get("source_capture_identity"),
        "source_capture_digest": source.get("source_capture_digest"),
        "source_artifact_schema": schema,
        "source_artifact_digest": source[digest_field],
        "verified_source_files": verified,
        "verified_source_file_set_digest": canonical_digest(
            {"verified_source_files": verified}
        ),
        "source_bytes_verified": True,
        "metric_scale_status": source.get("metric_scale_status"),
        "upstream_blockers": list(source.get("blockers") or []),
        "smallest_missing_measurement": None,
        "claim_boundary": {
            "provider_derived_support": provider_support,
            "blueprint_raw_contract_truth": raw_truth,
            "raw_capture_authority_upgraded": False,
            "metric_scale_proven": False,
            "metric_geometry_proven": False,
            "collision_geometry_proven": False,
            "physical_success_proven": False,
            "deployment_readiness_proven": False,
        },
        "producer_authority": {
            "producer": "blueprint.post_capture_source_profile",
            "agent_or_provider_self_authorization": False,
        },
    }
    if not _digest(profile["source_capture_digest"]):
        raise PostCaptureEvidenceError(["post_capture_source_capture_digest_invalid"])
    return _finalize(profile, "source_profile_digest")


def build_derived_site_geometry(
    *,
    source_profile: Mapping[str, Any],
    depth_surface_result: Mapping[str, Any],
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Normalize observed ARKit depth into a hole-preserving collider candidate."""

    source = _validate_artifact(
        source_profile,
        schema=SOURCE_PROFILE_SCHEMA,
        digest_field="source_profile_digest",
        code="source_profile",
    )
    depth = _validate_artifact(
        depth_surface_result,
        schema=ARKIT_DEPTH_RESULT_SCHEMA,
        digest_field="arkit_depth_surface_compilation_result_digest",
        code="arkit_depth_surface_result",
    )
    if depth.get("source_capture_digest") != source.get("source_capture_digest"):
        raise PostCaptureEvidenceError(["site_geometry_source_profile_mismatch"])
    asset = depth.get("surface_asset")
    if not isinstance(asset, Mapping) or not _digest(asset.get("digest")):
        raise PostCaptureEvidenceError(["site_geometry_surface_asset_invalid"])
    _safe_bound_file(
        Path(artifact_root).expanduser().resolve(strict=True),
        asset.get("relative_path"),
        asset.get("digest"),
        code="site_geometry_surface_asset",
    )
    metric_qualified = depth.get("metric_scale_status") == "validated"
    geometry = {
        "schema_version": GEOMETRY_SCHEMA,
        "status": "derived_candidate_unqualified",
        "source_profile_digest": source["source_profile_digest"],
        "source_capture_digest": source["source_capture_digest"],
        "depth_surface_result_digest": depth[
            "arkit_depth_surface_compilation_result_digest"
        ],
        "geometry_asset_digest": asset["digest"],
        "geometry_asset_relative_path": asset["relative_path"],
        "collider_candidate_digest": canonical_digest(
            {
                "surface_asset_digest": asset["digest"],
                "generated_fill_used": depth.get("generated_fill_used"),
                "unsupported_region_ids": depth.get("unsupported_region_ids") or [],
            }
        ),
        "coordinate_frame_declaration": depth.get("coordinate_frame_declaration"),
        "scale_source": {
            "kind": "arkit_sensor_depth",
            "status": depth.get("metric_scale_status"),
            "independently_qualified": metric_qualified,
        },
        "coverage_and_uncertainty": {
            "observed_region_ids": list(depth.get("observed_region_ids") or []),
            "unsupported_region_ids": list(depth.get("unsupported_region_ids") or []),
            "accepted_high_confidence_pixel_count": depth.get(
                "accepted_high_confidence_pixel_count"
            ),
            "rejected_or_missing_pixel_count": depth.get(
                "rejected_or_missing_pixel_count"
            ),
            "discontinuity_rejected_triangle_count": depth.get(
                "discontinuity_rejected_triangle_count"
            ),
            "generated_fill_used": False,
            "unseen_or_rejected_depth_filled": False,
            "uncertainty_model": "captured_confidence_and_explicit_unsupported_regions",
        },
        "qualification_state": {
            "metric_scale": "qualified" if metric_qualified else "unqualified",
            "collision_geometry": "unqualified",
            "isaac_contact": "unqualified",
            "candidate_may_self_qualify": False,
        },
        "smallest_missing_measurement": {
            "code": (
                "independent_metric_scale_measurement_missing"
                if not metric_qualified
                else "independent_collider_qualification_missing"
            ),
            "instruction": (
                "Measure scale against an independent site reference."
                if not metric_qualified
                else "Qualify the exact collider against independent collision/contact evidence."
            ),
            "stage": "derived_site_geometry",
        },
        "claim_boundary": {
            "observed_depth_surface_candidate": True,
            "holes_preserved": True,
            "metric_geometry_proven": metric_qualified,
            "collision_geometry_proven": False,
            "physical_surface_proven": False,
        },
    }
    return _finalize(geometry, "derived_site_geometry_digest")


def build_native_3dgs_candidate(
    *,
    source_profile: Mapping[str, Any],
    provider_receipt: Mapping[str, Any],
    appearance_asset_digest: str,
    provider_identity: str,
    provider_receipt_digest_field: str,
    full_resolution_appearance_preserved: bool,
) -> dict[str, Any]:
    """Normalize a canonical or provider receipt without upgrading its quality."""

    source = _validate_artifact(
        source_profile,
        schema=SOURCE_PROFILE_SCHEMA,
        digest_field="source_profile_digest",
        code="source_profile",
    )
    receipt = _clone(dict(provider_receipt))
    receipt_digest = receipt.get(provider_receipt_digest_field)
    if (
        not _digest(receipt_digest)
        or receipt_digest
        != canonical_digest(receipt, digest_field=provider_receipt_digest_field)
        or not _digest(appearance_asset_digest)
        or not str(provider_identity).strip()
    ):
        raise PostCaptureEvidenceError(["native_3dgs_provider_receipt_invalid"])
    receipt_source = receipt.get("source_capture_digest")
    if receipt_source is not None and receipt_source != source["source_capture_digest"]:
        raise PostCaptureEvidenceError(["native_3dgs_source_capture_mismatch"])
    candidate = {
        "schema_version": NATIVE_3DGS_SCHEMA,
        "status": "candidate",
        "source_profile_digest": source["source_profile_digest"],
        "source_capture_digest": source["source_capture_digest"],
        "provider_identity": str(provider_identity),
        "provider_receipt_schema": receipt.get("schema_version"),
        "provider_receipt_digest": receipt_digest,
        "appearance_format": "native_3dgs",
        "appearance_asset_digest": appearance_asset_digest,
        "full_resolution_appearance_preserved": bool(
            full_resolution_appearance_preserved
        ),
        "provider_self_qualified": False,
        "appearance_is_geometry_authority": False,
        "claim_boundary": {
            "appearance_candidate": True,
            "appearance_quality_qualified": receipt.get("status")
            in {"quality_winner_selected", "qualified"},
            "metric_registration_proven": False,
            "collision_geometry_proven": False,
        },
    }
    return _finalize(candidate, "native_3dgs_candidate_digest")


def build_native_3dgs_candidate_from_canonical(
    *,
    source_profile: Mapping[str, Any],
    registered_appearance: Mapping[str, Any],
) -> dict[str, Any]:
    """Adapt the canonical producer without treating appearance as geometry authority."""

    registered = _validate_artifact(
        registered_appearance,
        schema=CANONICAL_REGISTERED_APPEARANCE_SCHEMA,
        digest_field="canonical_registered_appearance_digest",
        code="canonical_registered_appearance",
    )
    if (
        registered.get("appearance_format") != "native_3dgs"
        or registered.get("full_resolution_appearance_preserved") is not True
        or not _digest(registered.get("appearance_asset_digest"))
        or registered.get("metric_geometry_proven") is not False
        or registered.get("collision_geometry_validated") is not False
        or registered.get("candidate_may_self_authorize") is not False
        or registered.get("claim_ceiling") != "registered_appearance_only"
    ):
        raise PostCaptureEvidenceError(["canonical_registered_appearance_invalid"])
    return build_native_3dgs_candidate(
        source_profile=source_profile,
        provider_receipt=registered,
        appearance_asset_digest=registered["appearance_asset_digest"],
        provider_identity="canonical_3dgs",
        provider_receipt_digest_field="canonical_registered_appearance_digest",
        full_resolution_appearance_preserved=True,
    )


def build_native_3dgs_candidate_from_teleport(
    *,
    source_profile: Mapping[str, Any],
    run_receipt: Mapping[str, Any],
    import_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize exact Teleport lifecycle and imported native PLY receipts."""

    run = _validate_artifact(
        run_receipt,
        schema=TELEPORT_RUN_RECEIPT_SCHEMA,
        digest_field="teleport_provider_run_receipt_digest",
        code="teleport_provider_run_receipt",
    )
    imported = _validate_artifact(
        import_receipt,
        schema=PROVIDER_SPLAT_IMPORT_RECEIPT_SCHEMA,
        digest_field="provider_splat_import_receipt_digest",
        code="provider_splat_import_receipt",
    )
    splats = [
        row
        for row in imported.get("imported_assets") or []
        if isinstance(row, Mapping) and row.get("artifact_kind") == "splat_ply"
    ]
    if (
        run.get("status") != "succeeded_unqualified"
        or run.get("provider_identity") != "teleport"
        or run.get("provider_splat_import_receipt_digest")
        != imported.get("provider_splat_import_receipt_digest")
        or not _digest(run.get("provider_execution_receipt_digest"))
        or run.get("provider_execution_receipt_digest")
        != imported.get("provider_execution_receipt_digest")
        or imported.get("provider_identity") != "teleport"
        or imported.get("provider_native_output_preserved_unchanged") is not True
        or imported.get("provider_success_is_blueprint_qualification") is not False
        or imported.get("metric_scale_proven") is not False
        or imported.get("collision_geometry_validated") is not False
        or run.get("metric_scale_proven") is not False
        or run.get("collision_geometry_validated") is not False
        or len(splats) != 1
        or not _digest(splats[0].get("digest"))
    ):
        raise PostCaptureEvidenceError(["teleport_native_3dgs_join_invalid"])
    candidate = build_native_3dgs_candidate(
        source_profile=source_profile,
        provider_receipt=imported,
        appearance_asset_digest=str(splats[0]["digest"]),
        provider_identity="teleport",
        provider_receipt_digest_field="provider_splat_import_receipt_digest",
        full_resolution_appearance_preserved=True,
    )
    candidate["teleport_provider_run_receipt_digest"] = run[
        "teleport_provider_run_receipt_digest"
    ]
    candidate["claim_boundary"]["appearance_quality_qualified"] = False
    candidate.pop("native_3dgs_candidate_digest")
    return _finalize(candidate, "native_3dgs_candidate_digest")


def build_registration_qualification_from_canonical(
    *,
    source_profile: Mapping[str, Any],
    appearance_candidate: Mapping[str, Any],
    site_geometry: Mapping[str, Any],
    registered_appearance: Mapping[str, Any],
    registration_measurement: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the canonical measurement to the exact normalized appearance and geometry."""

    source = _validate_artifact(
        source_profile,
        schema=SOURCE_PROFILE_SCHEMA,
        digest_field="source_profile_digest",
        code="source_profile",
    )
    appearance = _validate_artifact(
        appearance_candidate,
        schema=NATIVE_3DGS_SCHEMA,
        digest_field="native_3dgs_candidate_digest",
        code="native_3dgs_candidate",
    )
    geometry = _validate_artifact(
        site_geometry,
        schema=GEOMETRY_SCHEMA,
        digest_field="derived_site_geometry_digest",
        code="derived_site_geometry",
    )
    registered = _validate_artifact(
        registered_appearance,
        schema=CANONICAL_REGISTERED_APPEARANCE_SCHEMA,
        digest_field="canonical_registered_appearance_digest",
        code="canonical_registered_appearance",
    )
    measurement = _validate_artifact(
        registration_measurement,
        schema=CANONICAL_REGISTRATION_MEASUREMENT_SCHEMA,
        digest_field="canonical_3dgs_registration_measurement_digest",
        code="canonical_registration_measurement",
    )
    if any(
        (
            source.get("source_capture_digest")
            != registered.get("source_capture_digest"),
            source.get("source_capture_digest")
            != measurement.get("source_capture_digest"),
            appearance.get("appearance_asset_digest")
            != registered.get("appearance_asset_digest"),
            appearance.get("appearance_asset_digest")
            != measurement.get("appearance_asset_digest"),
            geometry.get("geometry_asset_digest")
            != registered.get("geometry_asset_digest"),
            registered.get("scene_registration_digest")
            != measurement.get("canonical_3dgs_registration_measurement_digest"),
            registered.get("world_frame") != measurement.get("world_frame"),
            dict(geometry.get("coordinate_frame_declaration") or {}).get("frame")
            != measurement.get("world_frame"),
            registered.get("registration_transform_appearance_to_site")
            != measurement.get("transform_appearance_to_site"),
            registered.get("registration_residual_summary")
            != measurement.get("residual_summary"),
        )
    ):
        raise PostCaptureEvidenceError(["canonical_registration_exact_join_mismatch"])
    measurement_digest = measurement["canonical_3dgs_registration_measurement_digest"]
    transform_digest = canonical_digest(
        {
            "schema_version": "scene_registration_transform.v1",
            "canonical_registration_measurement_digest": measurement_digest,
            "transform_appearance_to_site": measurement.get(
                "transform_appearance_to_site"
            ),
        }
    )
    residual_digest = canonical_digest(
        {
            "schema_version": "scene_registration_residual_measurement.v1",
            "canonical_registration_measurement_digest": measurement_digest,
            "residual_summary": measurement.get("residual_summary"),
            "thresholds_m": measurement.get("thresholds_m"),
        }
    )
    scene_registration_digest = canonical_digest(
        {
            "schema_version": "scene_registration.v1",
            "source_profile_digest": source["source_profile_digest"],
            "native_3dgs_candidate_digest": appearance[
                "native_3dgs_candidate_digest"
            ],
            "derived_site_geometry_digest": geometry[
                "derived_site_geometry_digest"
            ],
            "registration_transform_digest": transform_digest,
            "residual_measurement_digest": residual_digest,
        }
    )
    qualified = (
        registered.get("status") == "qualified"
        and registered.get("registration_status") == "qualified"
        and registered.get("heldout_appearance_status") == "qualified"
        and measurement.get("status") == "qualified"
        and measurement.get("registration_gate_passed") is True
    )
    result = {
        "schema_version": REGISTRATION_QUALIFICATION_SCHEMA,
        "status": "qualified" if qualified else "unqualified",
        "source_profile_digest": source["source_profile_digest"],
        "native_3dgs_candidate_digest": appearance[
            "native_3dgs_candidate_digest"
        ],
        "appearance_asset_digest": appearance["appearance_asset_digest"],
        "derived_site_geometry_digest": geometry["derived_site_geometry_digest"],
        "geometry_asset_digest": geometry["geometry_asset_digest"],
        "canonical_registered_appearance_digest": registered[
            "canonical_registered_appearance_digest"
        ],
        "canonical_registration_measurement_digest": measurement_digest,
        "scene_registration_digest": scene_registration_digest,
        "registration_transform_digest": transform_digest,
        "residual_measurement_digest": residual_digest,
        "qualifier_identity": "canonical-registration:" + str(measurement["method_id"]),
        "candidate_may_self_qualify": False,
        "smallest_missing_measurement": (
            None
            if qualified
            else {
                "code": "splat_metric_frame_registration_missing",
                "instruction": "Pass the frozen canonical appearance-to-site residual gate.",
                "stage": "reconstruction_registration",
            }
        ),
    }
    return _finalize(result, "registration_qualification_digest")


def build_qualified_site_geometry(
    *,
    geometry_candidate: Mapping[str, Any],
    independent_qualification: Mapping[str, Any],
) -> dict[str, Any]:
    """Upgrade only the exact geometry candidate measured by an independent gate."""

    geometry = _validate_artifact(
        geometry_candidate,
        schema=GEOMETRY_SCHEMA,
        digest_field="derived_site_geometry_digest",
        code="derived_site_geometry",
    )
    qualification = _validate_artifact(
        independent_qualification,
        schema=GEOMETRY_QUALIFICATION_SCHEMA,
        digest_field="geometry_qualification_digest",
        code="site_geometry_qualification",
    )
    expected = {
        "derived_site_geometry_digest": geometry["derived_site_geometry_digest"],
        "geometry_asset_digest": geometry["geometry_asset_digest"],
        "collider_candidate_digest": geometry["collider_candidate_digest"],
    }
    if any(qualification.get(key) != value for key, value in expected.items()):
        raise PostCaptureEvidenceError(["site_geometry_qualification_join_mismatch"])
    if (
        qualification.get("candidate_may_self_qualify") is not False
        or qualification.get("qualifier_identity")
        == geometry.get("producing_method")
    ):
        raise PostCaptureEvidenceError(["site_geometry_self_qualification_forbidden"])
    qualified = bool(
        qualification.get("status") == "qualified"
        and qualification.get("metric_scale_qualified") is True
        and qualification.get("collision_geometry_qualified") is True
        and qualification.get("blockers") in ([], ())
    )
    result = {
        **geometry,
        "status": "qualified" if qualified else "derived_candidate_unqualified",
        "unqualified_candidate_digest": geometry["derived_site_geometry_digest"],
        "qualification_state": {
            "metric_scale": (
                "qualified"
                if qualification.get("metric_scale_qualified") is True
                else "unqualified"
            ),
            "collision_geometry": (
                "qualified"
                if qualification.get("collision_geometry_qualified") is True
                else "unqualified"
            ),
            "isaac_contact": (
                "qualified"
                if qualification.get("isaac_contact_qualified") is True
                else "unqualified"
            ),
            "candidate_may_self_qualify": False,
        },
        "geometry_qualification_digest": qualification[
            "geometry_qualification_digest"
        ],
        "smallest_missing_measurement": (
            None
            if qualified
            else qualification.get("smallest_missing_measurement")
            or {
                "code": "independent_collider_qualification_missing",
                "instruction": "Qualify metric scale and collision geometry for the exact surface bytes.",
                "stage": "derived_site_geometry",
            }
        ),
        "claim_boundary": {
            **dict(geometry.get("claim_boundary") or {}),
            "metric_geometry_proven": qualification.get("metric_scale_qualified")
            is True,
            "collision_geometry_proven": qualification.get(
                "collision_geometry_qualified"
            )
            is True,
            "physical_surface_proven": False,
        },
    }
    result.pop("derived_site_geometry_digest", None)
    return _finalize(result, "derived_site_geometry_digest")


def build_registered_site_reconstruction(
    *,
    source_profile: Mapping[str, Any],
    appearance_candidate: Mapping[str, Any] | None,
    site_geometry: Mapping[str, Any] | None,
    registration_qualification: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Join exact appearance, geometry, transform, and residual evidence."""

    source = _validate_artifact(
        source_profile,
        schema=SOURCE_PROFILE_SCHEMA,
        digest_field="source_profile_digest",
        code="source_profile",
    )
    appearance = (
        _validate_artifact(
            appearance_candidate,
            schema=NATIVE_3DGS_SCHEMA,
            digest_field="native_3dgs_candidate_digest",
            code="native_3dgs_candidate",
        )
        if appearance_candidate is not None
        else None
    )
    geometry = (
        _validate_artifact(
            site_geometry,
            schema=GEOMETRY_SCHEMA,
            digest_field="derived_site_geometry_digest",
            code="derived_site_geometry",
        )
        if site_geometry is not None
        else None
    )
    missing: dict[str, str] | None = None
    if appearance is None:
        missing = {
            "code": "native_3dgs_appearance_missing",
            "instruction": "Produce a full-resolution native 3DGS from the admitted source.",
        }
    elif appearance.get("source_profile_digest") != source["source_profile_digest"]:
        raise PostCaptureEvidenceError(["registered_reconstruction_appearance_source_mismatch"])
    elif appearance.get("full_resolution_appearance_preserved") is not True:
        missing = {
            "code": "full_resolution_appearance_truth_missing",
            "instruction": "Preserve and hash the provider-native full-resolution 3DGS.",
        }
    elif geometry is None:
        missing = {
            "code": "derived_site_geometry_missing",
            "instruction": "Compile the source depth or mesh into a hole-preserving site geometry candidate.",
        }
    elif geometry.get("source_profile_digest") != source["source_profile_digest"]:
        raise PostCaptureEvidenceError(["registered_reconstruction_geometry_source_mismatch"])
    elif geometry.get("status") != "qualified":
        smallest = geometry.get("smallest_missing_measurement")
        smallest = dict(smallest) if isinstance(smallest, Mapping) else {}
        missing = {
            "code": str(
                smallest.get("code") or "independent_collider_qualification_missing"
            ),
            "instruction": str(
                smallest.get("instruction")
                or "Qualify metric scale and collision geometry for the exact site geometry."
            ),
        }
    qualification = (
        _validate_artifact(
            registration_qualification,
            schema=REGISTRATION_QUALIFICATION_SCHEMA,
            digest_field="registration_qualification_digest",
            code="scene_registration_qualification",
        )
        if registration_qualification is not None
        else None
    )
    if missing is None and qualification is None:
        missing = {
            "code": "splat_metric_frame_registration_missing",
            "instruction": "Measure the splat-to-site transform and residuals independently.",
        }
    if qualification is not None and appearance is not None and geometry is not None:
        expected = {
            "source_profile_digest": source["source_profile_digest"],
            "native_3dgs_candidate_digest": appearance["native_3dgs_candidate_digest"],
            "appearance_asset_digest": appearance["appearance_asset_digest"],
            "derived_site_geometry_digest": geometry["derived_site_geometry_digest"],
            "geometry_asset_digest": geometry["geometry_asset_digest"],
        }
        if any(qualification.get(key) != value for key, value in expected.items()):
            raise PostCaptureEvidenceError(["scene_registration_exact_join_mismatch"])
        if (
            qualification.get("candidate_may_self_qualify") is not False
            or qualification.get("qualifier_identity")
            == appearance.get("provider_identity")
        ):
            raise PostCaptureEvidenceError(["scene_registration_self_qualification_forbidden"])
        for field in (
            "scene_registration_digest",
            "registration_transform_digest",
            "residual_measurement_digest",
        ):
            if not _digest(qualification.get(field)):
                raise PostCaptureEvidenceError([f"scene_registration_{field}_invalid"])
        if qualification.get("status") != "qualified":
            smallest = qualification.get("smallest_missing_measurement")
            smallest = dict(smallest) if isinstance(smallest, Mapping) else {}
            missing = {
                "code": str(
                    smallest.get("code") or "splat_metric_frame_registration_missing"
                ),
                "instruction": str(
                    smallest.get("instruction")
                    or "Reduce and independently qualify the registration residual."
                ),
            }
    qualified = missing is None and qualification is not None
    result = {
        "schema_version": REGISTERED_RECONSTRUCTION_SCHEMA,
        "status": "qualified" if qualified else "abstained",
        "source_profile_digest": source["source_profile_digest"],
        "source_capture_digest": source["source_capture_digest"],
        "appearance_format": "native_3dgs",
        "appearance_asset_digest": (
            appearance.get("appearance_asset_digest") if appearance else None
        ),
        "native_3dgs_candidate_digest": (
            appearance.get("native_3dgs_candidate_digest") if appearance else None
        ),
        "full_resolution_appearance_preserved": bool(
            appearance
            and appearance.get("full_resolution_appearance_preserved") is True
        ),
        "geometry_asset_digest": geometry.get("geometry_asset_digest") if geometry else None,
        "source_scene_digest": geometry.get("geometry_asset_digest") if geometry else None,
        "derived_site_geometry_digest": (
            geometry.get("derived_site_geometry_digest") if geometry else None
        ),
        "geometry_qualification_status": (
            dict(geometry.get("qualification_state") or {}).get("collision_geometry")
            if geometry
            else "missing"
        ),
        "scene_registration_digest": (
            qualification.get("scene_registration_digest") if qualification else None
        ),
        "registration_transform_digest": (
            qualification.get("registration_transform_digest") if qualification else None
        ),
        "residual_measurement_digest": (
            qualification.get("residual_measurement_digest") if qualification else None
        ),
        "registration_qualification_digest": (
            qualification.get("registration_qualification_digest")
            if qualification
            else None
        ),
        "registration_status": "qualified" if qualified else "abstained",
        "presentation_output_used_as_evaluation_evidence": False,
        "smallest_missing_measurement": (
            {**missing, "stage": "reconstruction_registration"}
            if missing is not None
            else None
        ),
        "claim_boundary": {
            "appearance_quality_is_metric_registration": False,
            "appearance_used_as_dynamics_authority": False,
            "registration_proven": qualified,
            "collision_geometry_proven": bool(
                geometry
                and dict(geometry.get("qualification_state") or {}).get(
                    "collision_geometry"
                )
                == "qualified"
            ),
            "physical_success_proven": False,
        },
    }
    return _finalize(result, "reconstruction_digest")


def build_automatic_task_target(
    *,
    registered_reconstruction: Mapping[str, Any],
    target_pipeline_request: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the current analyzer and deterministic 3D binder on registered views."""

    reconstruction = _validate_artifact(
        registered_reconstruction,
        schema=REGISTERED_RECONSTRUCTION_SCHEMA,
        digest_field="reconstruction_digest",
        code="registered_reconstruction",
    )
    if reconstruction.get("status") != "qualified":
        raise PostCaptureEvidenceError(["automatic_target_reconstruction_unqualified"])
    request = _clone(dict(target_pipeline_request))
    if (
        request.get("source_scene_digest")
        != reconstruction.get("source_scene_digest")
        or request.get("metric_scale_status") != "validated"
    ):
        raise PostCaptureEvidenceError(["automatic_target_registration_scope_mismatch"])
    try:
        result = run_rendered_scene_task_target_pipeline(request)
    except ValueError as exc:
        raise PostCaptureEvidenceError(["automatic_task_target_producer_failed"]) from exc
    if (
        result.get("source_scene_digest") != reconstruction.get("source_scene_digest")
        or result.get("analysis_splat_digest")
        != reconstruction.get("appearance_asset_digest")
    ):
        raise PostCaptureEvidenceError(["automatic_task_target_output_join_mismatch"])
    return result


def build_task_robot_selection(target_orchestration: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the current selected target to the doctrine robot deterministically."""

    target = _clone(dict(target_orchestration))
    digest_field = (
        "orchestration_digest"
        if target.get("schema_version") == "rendered_scene_task_target_orchestration.v1"
        else "target_orchestration_digest"
    )
    if target.get(digest_field) != canonical_digest(target, digest_field=digest_field):
        raise PostCaptureEvidenceError(["task_target_orchestration_invalid"])
    selected = (
        dict(dict(target.get("target_analysis") or {}).get("selected_target") or {})
        if digest_field == "orchestration_digest"
        else dict(target.get("selected_target") or {})
    )
    if not selected:
        raise PostCaptureEvidenceError(["task_robot_selection_target_missing"])
    try:
        robot_id = select_robot_for_target(selected)
    except ValueError as exc:
        raise PostCaptureEvidenceError(["task_robot_selection_invalid"]) from exc
    binding_digest = selected.get("target_binding_digest")
    if not _digest(binding_digest):
        binding_rows = target.get("binding_results")
        matches = [
            row
            for row in (binding_rows or [])
            if isinstance(row, Mapping)
            and row.get("proposal_id") == selected.get("proposal_id")
        ]
        binding = matches[0].get("binding") if len(matches) == 1 else None
        binding_digest = (
            binding.get("binding_evidence_digest")
            if isinstance(binding, Mapping)
            else None
        )
    if not _digest(binding_digest):
        raise PostCaptureEvidenceError(["task_robot_selection_binding_missing"])
    result = {
        "schema_version": ROBOT_SELECTION_SCHEMA,
        "status": "selected",
        "target_orchestration_digest": target[digest_field],
        "target_binding_digest": binding_digest,
        "proposal_id": selected.get("proposal_id"),
        "task_family": selected.get("task_family"),
        "task_class": selected.get("task_class"),
        "robot_id": robot_id,
        "selection_policy": "explicit_requirement_then_task_semantics_then_franka_default",
        "selection_is_qualification": False,
    }
    return _finalize(result, "robot_selection_digest")


def build_qualified_robot_placement(
    *,
    placement_candidate: Mapping[str, Any],
    robot_selection: Mapping[str, Any],
    independent_qualification: Mapping[str, Any],
) -> dict[str, Any]:
    """Join an analytic placement candidate to an independent qualification."""

    selection = _validate_artifact(
        robot_selection,
        schema=ROBOT_SELECTION_SCHEMA,
        digest_field="robot_selection_digest",
        code="robot_selection",
    )
    candidate = _clone(dict(placement_candidate))
    candidate_field = (
        "placement_proposal_digest"
        if "placement_proposal_digest" in candidate
        else "placement_digest"
    )
    if candidate.get(candidate_field) != canonical_digest(
        candidate, digest_field=candidate_field
    ):
        raise PostCaptureEvidenceError(["robot_placement_candidate_invalid"])
    qualification = _validate_artifact(
        independent_qualification,
        schema=PLACEMENT_MEASUREMENT_SCHEMA,
        digest_field="placement_qualification_digest",
        code="placement_qualification",
    )
    expected = {
        "placement_candidate_digest": candidate[candidate_field],
        "robot_selection_digest": selection["robot_selection_digest"],
        "target_binding_digest": selection["target_binding_digest"],
        "robot_id": selection["robot_id"],
    }
    if any(qualification.get(key) != value for key, value in expected.items()):
        raise PostCaptureEvidenceError(["robot_placement_qualification_join_mismatch"])
    if (
        qualification.get("candidate_may_self_qualify") is not False
        or qualification.get("qualifier_identity")
        == candidate.get("producing_method")
    ):
        raise PostCaptureEvidenceError(["robot_placement_self_qualification_forbidden"])
    qualified = bool(
        qualification.get("status") == "qualified"
        and qualification.get("reachable") is True
        and qualification.get("footprint_clear") is True
        and qualification.get("collision_aware") is True
        and qualification.get("blockers") in ([], ())
    )
    result = {
        "schema_version": PLACEMENT_QUALIFICATION_SCHEMA,
        "status": "qualified" if qualified else "unqualified",
        "robot_id": selection["robot_id"],
        "target_binding_digest": selection["target_binding_digest"],
        "robot_selection_digest": selection["robot_selection_digest"],
        "placement_candidate_digest": candidate[candidate_field],
        "placement_pose": candidate.get("robot_pose_xyzyaw_collision_stage")
        or candidate.get("placement_pose"),
        "reachable": qualification.get("reachable") is True,
        "footprint_clear": qualification.get("footprint_clear") is True,
        "collision_aware": qualification.get("collision_aware") is True,
        "source_collider_qualified": qualification.get("source_collider_qualified")
        is True,
        "qualification_evidence_digest": qualification[
            "placement_qualification_digest"
        ],
        "blockers": list(qualification.get("blockers") or []),
        "smallest_missing_measurement": (
            None
            if qualified
            else qualification.get("smallest_missing_measurement")
            or {
                "code": "qualified_robot_placement_missing",
                "instruction": "Measure reach, footprint clearance, and collision at the exact proposed pose.",
                "stage": "robot_placement",
            }
        ),
        "candidate_may_self_authorize": False,
        "physical_execution_authorized": False,
    }
    return _finalize(result, "placement_digest")


def build_robot_placement_candidate(
    *,
    robot_selection: Mapping[str, Any],
    target_orchestration: Mapping[str, Any],
    placement_request: Mapping[str, Any],
    collision_glb_path: str | Path,
) -> dict[str, Any]:
    """Run the current collision-aware placement producer for the selected robot."""

    selection = _validate_artifact(
        robot_selection,
        schema=ROBOT_SELECTION_SCHEMA,
        digest_field="robot_selection_digest",
        code="robot_selection",
    )
    target = _clone(dict(target_orchestration))
    analysis = target.get("target_analysis")
    if not isinstance(analysis, Mapping):
        raise PostCaptureEvidenceError(["placement_target_analysis_missing"])
    request = _clone(dict(placement_request))
    if (
        request.get("robot_id") != selection["robot_id"]
        or request.get("target_binding_digest")
        != selection["target_binding_digest"]
    ):
        raise PostCaptureEvidenceError(["placement_request_robot_target_scope_mismatch"])
    if selection["robot_id"] != "franka_panda":
        raise PostCaptureEvidenceError(
            ["unitree_g1_collision_aware_placement_producer_unavailable"]
        )
    try:
        packet = propose_external_scene_robot_placement(
            collision_glb_path=collision_glb_path,
            request=request,
            target_analysis=analysis,
        )
    except ValueError as exc:
        raise PostCaptureEvidenceError(["robot_placement_candidate_producer_failed"]) from exc
    placement = packet.get("placement")
    if not isinstance(placement, Mapping):
        raise PostCaptureEvidenceError(["robot_placement_candidate_output_missing"])
    result = _clone(dict(placement))
    if (
        result.get("robot_id") != selection["robot_id"]
        or result.get("target_binding_digest")
        != selection["target_binding_digest"]
    ):
        raise PostCaptureEvidenceError(["robot_placement_candidate_output_join_mismatch"])
    return result


def build_scene_composition_decision(
    *,
    target_orchestration: Mapping[str, Any],
    qualified_placement: Mapping[str, Any],
    simready_task_zone_qualification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply inspection-versus-interaction composition semantics."""

    placement = _validate_artifact(
        qualified_placement,
        schema=PLACEMENT_QUALIFICATION_SCHEMA,
        digest_field="placement_digest",
        code="qualified_placement",
    )
    target = _clone(dict(target_orchestration))
    requirement = target.get("task_zone_asset_requirement")
    requirement = dict(requirement) if isinstance(requirement, Mapping) else {}
    task_zone_required = requirement.get("verified_simready_asset_required") is True
    if placement.get("status") != "qualified":
        raise PostCaptureEvidenceError(["scene_composition_placement_unqualified"])
    floor_status = (
        "not_required" if placement.get("source_collider_qualified") is True else "unqualified"
    )
    zone_status = "not_required"
    zone_digest: str | None = None
    smallest: dict[str, str] | None = None
    if floor_status != "not_required":
        smallest = {
            "code": "qualified_floor_support_mount_missing",
            "instruction": "Qualify the source collider or a bounded support mount.",
            "stage": "qualified_scene_composition",
        }
    if task_zone_required:
        zone = _clone(dict(simready_task_zone_qualification or {}))
        zone_field = "qualification_digest"
        valid_zone = bool(
            zone.get("status") == "qualified"
            and _digest(zone.get(zone_field))
            and zone.get(zone_field) == canonical_digest(zone, digest_field=zone_field)
            and zone.get("candidate_may_self_qualify") is False
            and zone.get("target_binding_digest") == placement["target_binding_digest"]
        )
        if valid_zone:
            zone_status = "qualified"
            zone_digest = zone[zone_field]
        elif smallest is None:
            zone_status = "unqualified"
            smallest = {
                "code": "qualified_simready_task_zone_missing",
                "instruction": "Insert and independently qualify a SimReady task-zone asset for this interaction.",
                "stage": "qualified_scene_composition",
            }
    result = {
        "schema_version": SCENE_COMPOSITION_SCHEMA,
        "status": "qualified" if smallest is None else "abstained",
        "target_binding_digest": placement["target_binding_digest"],
        "placement_digest": placement["placement_digest"],
        "task_semantics": "interaction" if task_zone_required else "inspection",
        "floor_support_mount": {
            "status": floor_status,
            "qualification_digest": (
                placement.get("qualification_evidence_digest")
                if floor_status == "not_required"
                else None
            ),
        },
        "task_zone_replacement": {
            "status": zone_status,
            "qualification_digest": zone_digest,
        },
        "smallest_missing_measurement": smallest,
        "candidate_may_self_authorize": False,
    }
    return _finalize(result, "scene_composition_digest")


def build_routing_inputs_and_decision(
    *,
    source_profile: Mapping[str, Any],
    robot_selection: Mapping[str, Any],
    qualified_placement: Mapping[str, Any],
    requirements: Mapping[str, Any],
    site_evidence_profile: Mapping[str, Any],
    method_capability_profiles: Sequence[Mapping[str, Any]],
    measurement_qualifications: Sequence[Mapping[str, Any]],
    catalog_snapshot_hash: str,
    routing_as_of: date,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create exact task/site routing inputs and execute the deterministic router."""

    source = _validate_artifact(
        source_profile,
        schema=SOURCE_PROFILE_SCHEMA,
        digest_field="source_profile_digest",
        code="source_profile",
    )
    selection = _validate_artifact(
        robot_selection,
        schema=ROBOT_SELECTION_SCHEMA,
        digest_field="robot_selection_digest",
        code="robot_selection",
    )
    placement = _validate_artifact(
        qualified_placement,
        schema=PLACEMENT_QUALIFICATION_SCHEMA,
        digest_field="placement_digest",
        code="qualified_placement",
    )
    if placement.get("status") != "qualified":
        raise PostCaptureEvidenceError(["routing_placement_unqualified"])
    inputs = {
        "schema_version": ROUTING_INPUTS_SCHEMA,
        "requirements": _clone(dict(requirements)),
        "site_evidence_profile": _clone(dict(site_evidence_profile)),
        "method_capability_profiles": _clone(list(method_capability_profiles)),
        "measurement_qualifications": _clone(list(measurement_qualifications)),
        "catalog_snapshot_hash": catalog_snapshot_hash,
        "routing_as_of": routing_as_of.isoformat(),
        "source_profile_digest": source["source_profile_digest"],
        "target_binding_digest": selection["target_binding_digest"],
        "placement_digest": placement["placement_digest"],
        "robot_id": selection["robot_id"],
        "task_class": selection["task_class"],
    }
    inputs = _finalize(inputs, "routing_inputs_digest")
    try:
        decision = route_task_site_measurement(
            requirements,
            site_evidence_profile,
            method_capability_profiles,
            measurement_qualifications,
            catalog_snapshot_hash=catalog_snapshot_hash,
            as_of=routing_as_of,
        )
    except (MeasurementRoutingError, TypeError, ValueError) as exc:
        raise PostCaptureEvidenceError(["post_capture_routing_inputs_invalid"]) from exc
    return inputs, decision


def build_policy_execution_decision(
    *,
    source_profile: Mapping[str, Any],
    registered_reconstruction: Mapping[str, Any],
    target_orchestration: Mapping[str, Any],
    routing_inputs: Mapping[str, Any],
    routing_decision: Mapping[str, Any],
    qualified_placement: Mapping[str, Any],
    scene_composition: Mapping[str, Any],
    task_metric: Mapping[str, Any],
    policy_candidates: Sequence[Mapping[str, Any]],
    authorizer_identity: str,
) -> dict[str, Any]:
    """Authorize only the exact independently qualified site/task/robot join."""

    source = _validate_artifact(
        source_profile,
        schema=SOURCE_PROFILE_SCHEMA,
        digest_field="source_profile_digest",
        code="source_profile",
    )
    reconstruction = _validate_artifact(
        registered_reconstruction,
        schema=REGISTERED_RECONSTRUCTION_SCHEMA,
        digest_field="reconstruction_digest",
        code="registered_reconstruction",
    )
    target = _clone(dict(target_orchestration))
    target_field = (
        "orchestration_digest"
        if target.get("schema_version") == "rendered_scene_task_target_orchestration.v1"
        else "target_orchestration_digest"
    )
    if target.get(target_field) != canonical_digest(target, digest_field=target_field):
        raise PostCaptureEvidenceError(["policy_authorization_target_invalid"])
    route_inputs = _validate_artifact(
        routing_inputs,
        schema=ROUTING_INPUTS_SCHEMA,
        digest_field="routing_inputs_digest",
        code="routing_inputs",
    )
    route = _clone(dict(routing_decision))
    placement = _validate_artifact(
        qualified_placement,
        schema=PLACEMENT_QUALIFICATION_SCHEMA,
        digest_field="placement_digest",
        code="qualified_placement",
    )
    composition = _validate_artifact(
        scene_composition,
        schema=SCENE_COMPOSITION_SCHEMA,
        digest_field="scene_composition_digest",
        code="scene_composition",
    )
    resolved_selection = build_task_robot_selection(target)
    target_binding_digest = resolved_selection["target_binding_digest"]
    exact_join = {
        "reconstruction_source_profile_digest": (
            reconstruction.get("source_profile_digest"),
            source["source_profile_digest"],
        ),
        "routing_source_profile_digest": (
            route_inputs.get("source_profile_digest"),
            source["source_profile_digest"],
        ),
        "routing_target_binding_digest": (
            route_inputs.get("target_binding_digest"),
            target_binding_digest,
        ),
        "routing_placement_digest": (
            route_inputs.get("placement_digest"),
            placement["placement_digest"],
        ),
        "routing_robot_id": (
            route_inputs.get("robot_id"),
            resolved_selection["robot_id"],
        ),
        "composition_placement_digest": (
            composition.get("placement_digest"),
            placement["placement_digest"],
        ),
    }
    if target_field == "orchestration_digest":
        exact_join.update(
            {
                "target_source_scene_digest": (
                    target.get("source_scene_digest"),
                    reconstruction.get("source_scene_digest"),
                ),
                "target_analysis_appearance_digest": (
                    target.get("analysis_splat_digest"),
                    reconstruction.get("appearance_asset_digest"),
                ),
            }
        )
    else:
        exact_join["target_reconstruction_digest"] = (
            target.get("reconstruction_digest"),
            reconstruction["reconstruction_digest"],
        )
    bad_joins = [name for name, values in exact_join.items() if values[0] != values[1]]
    if bad_joins:
        raise PostCaptureEvidenceError(
            [f"policy_authorization_{name}_mismatch" for name in bad_joins]
        )
    try:
        metric = validate_task_metric(task_metric)
    except (TypeError, ValueError) as exc:
        raise PostCaptureEvidenceError(
            ["policy_authorization_metric_invalid"]
        ) from exc
    candidates = [_clone(dict(row)) for row in policy_candidates]
    authorizer = str(authorizer_identity).strip()
    selected_method_ids = {
        str(row.get("method_id") or "")
        for row in dict(route.get("selected_route") or {}).get("stages", [])
        if isinstance(row, Mapping)
    }
    if not authorizer or authorizer in selected_method_ids:
        raise PostCaptureEvidenceError(["policy_authorizer_independence_invalid"])
    missing: dict[str, str] | None = None
    if route.get("status") != "route_selected":
        missing = {
            "code": "no_exact_qualified_measurement_route",
            "instruction": "Collect the deterministic router's smallest missing qualification.",
        }
    elif placement.get("status") != "qualified":
        missing = {
            "code": "qualified_robot_placement_missing",
            "instruction": "Qualify the exact placement before policy execution.",
        }
    elif composition.get("status") != "qualified":
        smallest = composition.get("smallest_missing_measurement")
        smallest = dict(smallest) if isinstance(smallest, Mapping) else {}
        missing = {
            "code": str(smallest.get("code") or "scene_composition_qualification_missing"),
            "instruction": str(
                smallest.get("instruction") or "Qualify exact scene composition."
            ),
        }
    elif len(candidates) != EXPECTED_POLICY_CANDIDATES:
        missing = {
            "code": "exactly_five_learned_policy_candidates_required",
            "instruction": "Bind exactly five immutable learned-policy candidates.",
        }
    for field in ("routing_decision_digest",):
        if (
            not _digest(route.get(field))
            or route.get(field) != canonical_digest(route, digest_field=field)
        ):
            raise PostCaptureEvidenceError([f"policy_authorization_{field}_invalid"])
    try:
        expected_route = route_task_site_measurement(
            route_inputs["requirements"],
            route_inputs["site_evidence_profile"],
            route_inputs["method_capability_profiles"],
            route_inputs["measurement_qualifications"],
            catalog_snapshot_hash=str(route_inputs["catalog_snapshot_hash"]),
            as_of=date.fromisoformat(str(route_inputs["routing_as_of"])),
        )
    except (KeyError, MeasurementRoutingError, TypeError, ValueError) as exc:
        raise PostCaptureEvidenceError(
            ["policy_authorization_routing_inputs_invalid"]
        ) from exc
    if expected_route.get("routing_decision_digest") != route.get(
        "routing_decision_digest"
    ):
        raise PostCaptureEvidenceError(["policy_authorization_route_replay_mismatch"])
    if missing is None:
        try:
            candidates = validate_policy_candidates(candidates)
        except (TypeError, ValueError) as exc:
            raise PostCaptureEvidenceError(
                ["policy_authorization_candidate_invalid"]
            ) from exc
        identity_digests = [row["policy_identity_digest"] for row in candidates]
    else:
        identity_digests = [canonical_digest(row) for row in candidates]
    candidate_set_digest = canonical_digest(
        {"policy_identity_digests": sorted(str(value) for value in identity_digests)}
    )
    if missing is not None:
        return _finalize(
            {
                "schema_version": AUTHORIZATION_DECISION_SCHEMA,
                "status": "abstained",
                "policy_execution_authorized": False,
                "routing_decision_digest": route.get("routing_decision_digest"),
                "routing_inputs_digest": route_inputs["routing_inputs_digest"],
                "source_profile_digest": source["source_profile_digest"],
                "reconstruction_digest": reconstruction["reconstruction_digest"],
                "target_orchestration_digest": target[target_field],
                "target_binding_digest": target_binding_digest,
                "placement_digest": placement["placement_digest"],
                "scene_composition_digest": composition["scene_composition_digest"],
                "metric_spec_digest": metric["metric_spec_digest"],
                "candidate_set_digest": candidate_set_digest,
                "authorizer_identity": authorizer,
                "smallest_missing_measurement": {
                    **missing,
                    "stage": "policy_execution_authorization",
                },
            },
            "authorization_decision_digest",
        )
    authorization = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "policy_execution_authorized": True,
        "physical_robot_execution_authorized": False,
        "routing_decision_digest": route["routing_decision_digest"],
        "routing_inputs_digest": route_inputs["routing_inputs_digest"],
        "source_profile_digest": source["source_profile_digest"],
        "reconstruction_digest": reconstruction["reconstruction_digest"],
        "target_orchestration_digest": target[target_field],
        "target_binding_digest": target_binding_digest,
        "placement_digest": placement["placement_digest"],
        "scene_composition_digest": composition["scene_composition_digest"],
        "metric_spec_digest": metric["metric_spec_digest"],
        "candidate_set_digest": candidate_set_digest,
        "authorizer_identity": authorizer,
        "agent_or_provider_self_authorized": False,
    }
    return _finalize(authorization, "authorization_digest")


def _artifact_reference(path: Path, value: Mapping[str, Any], digest_field: str) -> dict[str, Any]:
    return {
        "schema_version": value.get("schema_version"),
        "digest": value[digest_field],
        "relative_path": path.name,
    }


def run_post_capture_evidence_spine(
    *,
    run_id: str,
    source_artifact: Mapping[str, Any],
    source_root: str | Path,
    output_root: str | Path,
    appearance_candidate: Mapping[str, Any] | None = None,
    canonical_registered_appearance: Mapping[str, Any] | None = None,
    canonical_registration_measurement: Mapping[str, Any] | None = None,
    teleport_run_receipt: Mapping[str, Any] | None = None,
    teleport_import_receipt: Mapping[str, Any] | None = None,
    depth_surface_result: Mapping[str, Any] | None = None,
    depth_surface_root: str | Path | None = None,
    geometry_qualification: Mapping[str, Any] | None = None,
    registration_qualification: Mapping[str, Any] | None = None,
    target_orchestration: Mapping[str, Any] | None = None,
    target_pipeline_request: Mapping[str, Any] | None = None,
    placement_candidate: Mapping[str, Any] | None = None,
    placement_request: Mapping[str, Any] | None = None,
    collision_glb_path: str | Path | None = None,
    placement_qualification: Mapping[str, Any] | None = None,
    simready_task_zone_qualification: Mapping[str, Any] | None = None,
    routing_bundle: Mapping[str, Any] | None = None,
    task_metric: Mapping[str, Any] | None = None,
    policy_candidates: Sequence[Mapping[str, Any]] = (),
    policy_attempts: Sequence[Mapping[str, Any]] = (),
    authorizer_identity: str = "blueprint-post-capture-admission",
) -> dict[str, Any]:
    """Execute all available stages and emit authorization or exact abstention."""

    if not str(run_id).strip():
        raise PostCaptureEvidenceError(["post_capture_run_id_missing"])
    source = build_source_profile(source_artifact=source_artifact, source_root=source_root)
    appearance_input_count = sum(
        value is not None
        for value in (
            appearance_candidate,
            canonical_registered_appearance,
            teleport_run_receipt,
        )
    )
    if appearance_input_count > 1:
        raise PostCaptureEvidenceError(["native_3dgs_input_ambiguous"])
    if (teleport_run_receipt is None) != (teleport_import_receipt is None):
        raise PostCaptureEvidenceError(["teleport_receipt_pair_incomplete"])
    if (
        registration_qualification is not None
        and canonical_registration_measurement is not None
    ):
        raise PostCaptureEvidenceError(["registration_qualification_input_ambiguous"])
    invocation = {
        "schema_version": "post_capture_evidence_invocation.v1",
        "run_id": str(run_id),
        "source_profile_digest": source["source_profile_digest"],
        "appearance_candidate_digest": (
            appearance_candidate.get("native_3dgs_candidate_digest")
            if isinstance(appearance_candidate, Mapping)
            else None
        ),
        "canonical_registered_appearance_digest": (
            canonical_registered_appearance.get(
                "canonical_registered_appearance_digest"
            )
            if isinstance(canonical_registered_appearance, Mapping)
            else None
        ),
        "canonical_registration_measurement_digest": (
            canonical_registration_measurement.get(
                "canonical_3dgs_registration_measurement_digest"
            )
            if isinstance(canonical_registration_measurement, Mapping)
            else None
        ),
        "teleport_provider_run_receipt_digest": (
            teleport_run_receipt.get("teleport_provider_run_receipt_digest")
            if isinstance(teleport_run_receipt, Mapping)
            else None
        ),
        "teleport_provider_splat_import_receipt_digest": (
            teleport_import_receipt.get("provider_splat_import_receipt_digest")
            if isinstance(teleport_import_receipt, Mapping)
            else None
        ),
        "depth_surface_result_digest": (
            depth_surface_result.get("arkit_depth_surface_compilation_result_digest")
            if isinstance(depth_surface_result, Mapping)
            else None
        ),
        "registration_qualification_digest": (
            registration_qualification.get("registration_qualification_digest")
            if isinstance(registration_qualification, Mapping)
            else None
        ),
        "geometry_qualification_digest": (
            geometry_qualification.get("geometry_qualification_digest")
            if isinstance(geometry_qualification, Mapping)
            else None
        ),
        "target_input_digest": (
            canonical_digest(dict(target_orchestration))
            if isinstance(target_orchestration, Mapping)
            else None
        ),
        "target_pipeline_request_digest": (
            canonical_digest(dict(target_pipeline_request))
            if isinstance(target_pipeline_request, Mapping)
            else None
        ),
        "placement_input_digest": (
            canonical_digest(dict(placement_candidate))
            if isinstance(placement_candidate, Mapping)
            else None
        ),
        "placement_request_digest": (
            canonical_digest(dict(placement_request))
            if isinstance(placement_request, Mapping)
            else None
        ),
        "collision_glb_digest": (
            _sha256(Path(collision_glb_path).expanduser().resolve(strict=True))
            if collision_glb_path is not None
            else None
        ),
        "placement_qualification_input_digest": (
            canonical_digest(dict(placement_qualification))
            if isinstance(placement_qualification, Mapping)
            else None
        ),
        "simready_qualification_input_digest": (
            canonical_digest(dict(simready_task_zone_qualification))
            if isinstance(simready_task_zone_qualification, Mapping)
            else None
        ),
        "routing_bundle_digest": (
            canonical_digest(dict(routing_bundle))
            if isinstance(routing_bundle, Mapping)
            else None
        ),
        "task_metric_input_digest": (
            canonical_digest(dict(task_metric))
            if isinstance(task_metric, Mapping)
            else None
        ),
        "policy_candidate_input_digest": canonical_digest(
            {"policy_candidates": list(policy_candidates)}
        ),
        "policy_attempt_input_digest": canonical_digest(
            {"policy_attempts": list(policy_attempts)}
        ),
    }
    invocation = _finalize(invocation, "invocation_digest")
    run_root = Path(output_root).expanduser().resolve() / (
        "post_capture_" + invocation["invocation_digest"][7:23]
    )
    source_path = run_root / "01_source_profile.json"
    _write_immutable(source_path, source)
    artifacts = [_artifact_reference(source_path, source, "source_profile_digest")]
    appearance = None
    if appearance_input_count:
        appearance = (
            build_native_3dgs_candidate_from_teleport(
                source_profile=source,
                run_receipt=teleport_run_receipt or {},
                import_receipt=teleport_import_receipt or {},
            )
            if teleport_run_receipt is not None
            else (
                build_native_3dgs_candidate_from_canonical(
                    source_profile=source,
                    registered_appearance=canonical_registered_appearance or {},
                )
                if canonical_registered_appearance is not None
                else _validate_artifact(
                    appearance_candidate or {},
                    schema=NATIVE_3DGS_SCHEMA,
                    digest_field="native_3dgs_candidate_digest",
                    code="native_3dgs_candidate",
                )
            )
        )
        appearance_path = run_root / "02_native_3dgs_candidate.json"
        _write_immutable(appearance_path, appearance)
        artifacts.append(
            _artifact_reference(
                appearance_path, appearance, "native_3dgs_candidate_digest"
            )
        )
    geometry = None
    if depth_surface_result is not None:
        if depth_surface_root is None:
            raise PostCaptureEvidenceError(["depth_surface_root_missing"])
        geometry = build_derived_site_geometry(
            source_profile=source,
            depth_surface_result=depth_surface_result,
            artifact_root=depth_surface_root,
        )
        if geometry_qualification is not None:
            geometry = build_qualified_site_geometry(
                geometry_candidate=geometry,
                independent_qualification=geometry_qualification,
            )
        geometry_path = run_root / "03_derived_site_geometry.json"
        _write_immutable(geometry_path, geometry)
        artifacts.append(
            _artifact_reference(
                geometry_path, geometry, "derived_site_geometry_digest"
            )
        )
    effective_registration_qualification = registration_qualification
    if canonical_registration_measurement is not None:
        if canonical_registered_appearance is None:
            raise PostCaptureEvidenceError(
                ["canonical_registered_appearance_required_for_measurement"]
            )
        if appearance is None or geometry is None:
            effective_registration_qualification = None
        else:
            effective_registration_qualification = (
                build_registration_qualification_from_canonical(
                    source_profile=source,
                    appearance_candidate=appearance,
                    site_geometry=geometry,
                    registered_appearance=canonical_registered_appearance,
                    registration_measurement=canonical_registration_measurement,
                )
            )
    if effective_registration_qualification is not None:
        effective_registration_qualification = _validate_artifact(
            effective_registration_qualification,
            schema=REGISTRATION_QUALIFICATION_SCHEMA,
            digest_field="registration_qualification_digest",
            code="scene_registration_qualification",
        )
        registration_path = run_root / "03b_scene_registration_qualification.json"
        _write_immutable(registration_path, effective_registration_qualification)
        artifacts.append(
            _artifact_reference(
                registration_path,
                effective_registration_qualification,
                "registration_qualification_digest",
            )
        )
    reconstruction = build_registered_site_reconstruction(
        source_profile=source,
        appearance_candidate=appearance,
        site_geometry=geometry,
        registration_qualification=effective_registration_qualification,
    )
    reconstruction_path = run_root / "04_registered_site_reconstruction.json"
    _write_immutable(reconstruction_path, reconstruction)
    artifacts.append(
        _artifact_reference(
            reconstruction_path, reconstruction, "reconstruction_digest"
        )
    )
    request = {
        "schema_version": NEW_SITE_REQUEST_SCHEMA,
        "run_id": str(run_id),
        "source_profile": source,
        "reconstruction": reconstruction,
    }
    route_decision: dict[str, Any] | None = None
    selected_placement: dict[str, Any] | None = None
    composition: dict[str, Any] | None = None
    if reconstruction.get("status") == "qualified" and (
        target_orchestration is not None or target_pipeline_request is not None
    ):
        if target_orchestration is not None and target_pipeline_request is not None:
            raise PostCaptureEvidenceError(["automatic_target_input_ambiguous"])
        target = (
            build_automatic_task_target(
                registered_reconstruction=reconstruction,
                target_pipeline_request=target_pipeline_request or {},
            )
            if target_pipeline_request is not None
            else _clone(dict(target_orchestration or {}))
        )
        target_field = (
            "orchestration_digest"
            if target.get("schema_version")
            == "rendered_scene_task_target_orchestration.v1"
            else "target_orchestration_digest"
        )
        if target.get(target_field) != canonical_digest(target, digest_field=target_field):
            raise PostCaptureEvidenceError(["target_orchestration_input_invalid"])
        target_path = run_root / "05_target_orchestration.json"
        _write_immutable(target_path, target)
        artifacts.append(_artifact_reference(target_path, target, target_field))
        request["target_orchestration"] = target
        selection = build_task_robot_selection(target)
        selection_path = run_root / "06_task_robot_selection.json"
        _write_immutable(selection_path, selection)
        artifacts.append(
            _artifact_reference(selection_path, selection, "robot_selection_digest")
        )
        if placement_candidate is not None and placement_request is not None:
            raise PostCaptureEvidenceError(["placement_candidate_input_ambiguous"])
        effective_placement_candidate = placement_candidate
        if placement_request is not None:
            if collision_glb_path is None:
                raise PostCaptureEvidenceError(["placement_collision_glb_missing"])
            effective_placement_candidate = build_robot_placement_candidate(
                robot_selection=selection,
                target_orchestration=target,
                placement_request=placement_request,
                collision_glb_path=collision_glb_path,
            )
        if effective_placement_candidate is not None:
            if placement_qualification is not None:
                selected_placement = build_qualified_robot_placement(
                    placement_candidate=effective_placement_candidate,
                    robot_selection=selection,
                    independent_qualification=placement_qualification,
                )
                placement_field = "placement_digest"
            else:
                selected_placement = _clone(dict(effective_placement_candidate))
                placement_field = (
                    "placement_proposal_digest"
                    if "placement_proposal_digest" in selected_placement
                    else "placement_digest"
                )
                if selected_placement.get(placement_field) != canonical_digest(
                    selected_placement, digest_field=placement_field
                ):
                    raise PostCaptureEvidenceError(["placement_candidate_input_invalid"])
            placement_path = run_root / "07_robot_placement.json"
            _write_immutable(placement_path, selected_placement)
            artifacts.append(
                _artifact_reference(
                    placement_path, selected_placement, placement_field
                )
            )
            request["robot_placement"] = selected_placement
        if (
            selected_placement is not None
            and selected_placement.get("status") == "qualified"
        ):
            composition = build_scene_composition_decision(
                target_orchestration=target,
                qualified_placement=selected_placement,
                simready_task_zone_qualification=simready_task_zone_qualification,
            )
            composition_path = run_root / "08_scene_composition.json"
            _write_immutable(composition_path, composition)
            artifacts.append(
                _artifact_reference(
                    composition_path, composition, "scene_composition_digest"
                )
            )
            request["scene_composition"] = composition
        if (
            routing_bundle is not None
            and selected_placement is not None
            and selected_placement.get("status") == "qualified"
        ):
            bundle = _clone(dict(routing_bundle))
            required_keys = {
                "requirements",
                "site_evidence_profile",
                "method_capability_profiles",
                "measurement_qualifications",
                "catalog_snapshot_hash",
                "routing_as_of",
            }
            if not required_keys.issubset(bundle):
                raise PostCaptureEvidenceError(["routing_bundle_incomplete"])
            try:
                routing_date = date.fromisoformat(str(bundle["routing_as_of"]))
            except ValueError as exc:
                raise PostCaptureEvidenceError(["routing_bundle_date_invalid"]) from exc
            routing_inputs, route_decision = build_routing_inputs_and_decision(
                source_profile=source,
                robot_selection=selection,
                qualified_placement=selected_placement,
                requirements=bundle["requirements"],
                site_evidence_profile=bundle["site_evidence_profile"],
                method_capability_profiles=bundle["method_capability_profiles"],
                measurement_qualifications=bundle["measurement_qualifications"],
                catalog_snapshot_hash=str(bundle["catalog_snapshot_hash"]),
                routing_as_of=routing_date,
            )
            routing_inputs_path = run_root / "09_routing_inputs.json"
            _write_immutable(routing_inputs_path, routing_inputs)
            artifacts.append(
                _artifact_reference(
                    routing_inputs_path, routing_inputs, "routing_inputs_digest"
                )
            )
            request["routing_inputs"] = routing_inputs
            route_path = run_root / "09_routing_decision.json"
            _write_immutable(route_path, route_decision)
            artifacts.append(
                _artifact_reference(
                    route_path, route_decision, "routing_decision_digest"
                )
            )
        if task_metric is not None or policy_candidates or policy_attempts:
            request["policy_evaluation"] = {
                "task_metric": _clone(dict(task_metric or {})),
                "policy_candidates": _clone(list(policy_candidates)),
                "attempts": _clone(list(policy_attempts)),
            }
        if (
            route_decision is not None
            and selected_placement is not None
            and composition is not None
            and task_metric is not None
            and policy_candidates
        ):
            authorization = build_policy_execution_decision(
                source_profile=source,
                registered_reconstruction=reconstruction,
                target_orchestration=target,
                routing_inputs=routing_inputs,
                routing_decision=route_decision,
                qualified_placement=selected_placement,
                scene_composition=composition,
                task_metric=task_metric,
                policy_candidates=policy_candidates,
                authorizer_identity=authorizer_identity,
            )
            authorization_field = (
                "authorization_digest"
                if authorization.get("schema_version") == AUTHORIZATION_SCHEMA_VERSION
                else "authorization_decision_digest"
            )
            authorization_path = run_root / "10_policy_execution_authorization.json"
            _write_immutable(authorization_path, authorization)
            artifacts.append(
                _artifact_reference(
                    authorization_path, authorization, authorization_field
                )
            )
            if authorization.get("policy_execution_authorized") is True:
                request["execution_authorization"] = authorization
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    terminal = compile_new_site_task_evaluation_run(request)
    terminal_path = run_root / "terminal_new_site_task_evaluation_run.json"
    _write_immutable(terminal_path, terminal)
    artifacts.append(_artifact_reference(terminal_path, terminal, "run_digest"))
    manifest = {
        "schema_version": RUN_SCHEMA,
        "run_id": str(run_id),
        "status": terminal["status"],
        "invocation_digest": invocation["invocation_digest"],
        "source_profile_digest": source["source_profile_digest"],
        "terminal_stage": terminal["terminal_stage"],
        "smallest_missing_measurement": terminal["smallest_missing_measurement"],
        "artifacts": artifacts,
        "idempotent_content_addressing": True,
        "upstream_digest_change_creates_new_run_directory": True,
        "fixture_evidence_used": False,
    }
    manifest = _finalize(manifest, "post_capture_evidence_run_digest")
    manifest_path = run_root / "post_capture_evidence_run.json"
    _write_immutable(manifest_path, manifest)
    return {"run_root": str(run_root), "manifest": manifest, "terminal": terminal}


def main(argv: Sequence[str] | None = None) -> int:
    from .post_capture_evidence_cli import main as cli_main

    return cli_main(argv)


__all__ = [
    "AUTHORIZATION_DECISION_SCHEMA",
    "CANONICAL_REGISTERED_APPEARANCE_SCHEMA",
    "CANONICAL_REGISTRATION_MEASUREMENT_SCHEMA",
    "GEOMETRY_SCHEMA",
    "GEOMETRY_QUALIFICATION_SCHEMA",
    "NATIVE_3DGS_SCHEMA",
    "PLACEMENT_QUALIFICATION_SCHEMA",
    "PLACEMENT_MEASUREMENT_SCHEMA",
    "PostCaptureEvidenceError",
    "REGISTERED_RECONSTRUCTION_SCHEMA",
    "REGISTRATION_QUALIFICATION_SCHEMA",
    "ROBOT_SELECTION_SCHEMA",
    "ROUTING_INPUTS_SCHEMA",
    "RUN_SCHEMA",
    "SCENE_COMPOSITION_SCHEMA",
    "SOURCE_PROFILE_SCHEMA",
    "build_derived_site_geometry",
    "build_automatic_task_target",
    "build_native_3dgs_candidate",
    "build_native_3dgs_candidate_from_canonical",
    "build_native_3dgs_candidate_from_teleport",
    "build_qualified_site_geometry",
    "build_policy_execution_decision",
    "build_robot_placement_candidate",
    "build_qualified_robot_placement",
    "build_registered_site_reconstruction",
    "build_registration_qualification_from_canonical",
    "build_routing_inputs_and_decision",
    "build_scene_composition_decision",
    "build_source_profile",
    "build_task_robot_selection",
    "main",
    "run_post_capture_evidence_spine",
]


if __name__ == "__main__":
    raise SystemExit(main())
