"""Qualify one supplemental passive destination inside the scene-configuration run.

A pick_and_place run may carry one passive destination (for example a tray)
that has no source object to remove.  It enters the run as exact
request-declared bytes plus the recipe-bound authoring receipt and SimReady
result, and leaves with the same independent static and native-import
qualifications the subject replacement receives.  Nothing here invents
source-removal lineage for it.  Stage 4 calls
``supplemental_destination_static_artifacts``; stage 5 calls
``supplemental_destination_native_artifacts``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_configuration_adapters import (
    TaskEvaluationSceneConfigurationAdapterError,
)
from .task_evaluation_scene_configuration_static_qualification import (
    SCHEMA_VERSION as STATIC_QUALIFICATION_SCHEMA_VERSION,
    TaskEvaluationSceneConfigurationStaticQualificationError,
    qualify_scene_configuration_rigid_asset_static,
)


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _materialized_reference(
    envelope: Mapping[str, Any], *, contract_path: str
) -> tuple[Mapping[str, Any], Path]:
    matches = [
        row
        for row in envelope.get("materialized_references") or []
        if isinstance(row, Mapping) and row.get("contract_path") == contract_path
    ]
    if len(matches) != 1:
        raise TaskEvaluationSceneConfigurationAdapterError(
            f"scene_configuration_materialized_reference_missing:{contract_path}"
        )
    row = matches[0]
    path = Path(str(row.get("materialized_path") or "")).resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256_and_size(path) != (row.get("digest"), row.get("size_bytes"))
        or row.get("full_byte_service_account_readback_passed") is not True
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            f"scene_configuration_materialized_reference_invalid:{contract_path}"
        )
    return row, path


def _dependency_artifact(
    dependency_results: tuple[Mapping[str, Any], ...], *, role: str
) -> tuple[Mapping[str, Any], Path]:
    matches = [
        artifact
        for result in dependency_results
        for artifact in result.get("output_artifacts") or []
        if isinstance(artifact, Mapping) and artifact.get("role") == role
    ]
    if len(matches) != 1:
        raise TaskEvaluationSceneConfigurationAdapterError(
            f"scene_configuration_dependency_artifact_missing:{role}"
        )
    artifact = matches[0]
    path = Path(str(artifact.get("path") or "")).resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256_and_size(path) != (artifact.get("digest"), artifact.get("size_bytes"))
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            f"scene_configuration_dependency_artifact_invalid:{role}"
        )
    return artifact, path


def _provider_runtime_artifact(
    provider_runtime_artifacts: tuple[Mapping[str, Any], ...], *, role: str
) -> tuple[Mapping[str, Any], Path]:
    matches = [
        artifact
        for artifact in provider_runtime_artifacts
        if isinstance(artifact, Mapping) and artifact.get("role") == role
    ]
    if len(matches) != 1:
        raise TaskEvaluationSceneConfigurationAdapterError(
            f"scene_configuration_provider_runtime_artifact_missing:{role}"
        )
    artifact = matches[0]
    path = Path(str(artifact.get("path") or "")).resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256_and_size(path) != (artifact.get("digest"), artifact.get("size_bytes"))
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            f"scene_configuration_provider_runtime_artifact_invalid:{role}"
        )
    return artifact, path


def _copy_artifact(source: Path, destination: Path) -> dict[str, Any]:
    shutil.copyfile(source, destination)
    if _sha256_and_size(destination) != _sha256_and_size(source):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "scene_configuration_provider_runtime_artifact_copy_mismatch"
        )
    return {
        "path": str(destination),
        "digest": _sha256_and_size(destination)[0],
        "size_bytes": _sha256_and_size(destination)[1],
    }


SUPPLEMENTAL_DESTINATION_ASSET_CONTRACT_PATH = "task.destination.asset"
SUPPLEMENTAL_DESTINATION_STATIC_CONTRACT_PATH = (
    "task.destination.static_qualification"
)
SUPPLEMENTAL_DESTINATION_RIGHTS_CONTRACT_PATH = "task.destination.rights_admission"
SUPPLEMENTAL_DESTINATION_AUTHORING_CONTRACT_PATH = (
    "construction.recipe.supplemental_destination.authoring_receipt"
)
SUPPLEMENTAL_DESTINATION_SIMREADY_CONTRACT_PATH = (
    "construction.recipe.supplemental_destination.simready_result"
)
_SUPPLEMENTAL_DESTINATION_CONTRACT_PATHS = {
    "asset": SUPPLEMENTAL_DESTINATION_ASSET_CONTRACT_PATH,
    "static_qualification": SUPPLEMENTAL_DESTINATION_STATIC_CONTRACT_PATH,
    "rights_admission": SUPPLEMENTAL_DESTINATION_RIGHTS_CONTRACT_PATH,
    "authoring_receipt": SUPPLEMENTAL_DESTINATION_AUTHORING_CONTRACT_PATH,
    "simready_result": SUPPLEMENTAL_DESTINATION_SIMREADY_CONTRACT_PATH,
}
PASSIVE_DESTINATION_SIMREADY_SCHEMA_VERSION = (
    "task_evaluation_passive_destination_simready.v1"
)
DESTINATION_RIGHTS_SCHEMA_VERSION = (
    "task_evaluation_rigid_destination_rights_admission.v1"
)


def _load_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationAdapterError(code) from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationAdapterError(code)
    return dict(value)


def supplemental_destination_inputs(
    envelope: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve and cross-bind every supplemental destination input, or ``None``.

    Returns the recipe binding, the exact materialized files, and their parsed
    receipts after every digest join has been checked.  Raises a typed adapter
    error when the recipe, request references, or receipts disagree.
    """

    recipe = envelope.get("recipe")
    destination = (
        recipe.get("supplemental_destination") if isinstance(recipe, Mapping) else None
    )
    if destination is None:
        return None
    if not isinstance(destination, Mapping) or not isinstance(
        destination.get("identity"), Mapping
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_supplemental_destination_recipe_invalid"
        )
    identity = dict(destination["identity"])
    rows: dict[str, Mapping[str, Any]] = {}
    paths: dict[str, Path] = {}
    for name, contract_path in _SUPPLEMENTAL_DESTINATION_CONTRACT_PATHS.items():
        row, path = _materialized_reference(envelope, contract_path=contract_path)
        expected = destination.get(name)
        if not isinstance(expected, Mapping) or any(
            row.get(key) != expected.get(key)
            for key in ("uri", "digest", "size_bytes")
        ):
            raise TaskEvaluationSceneConfigurationAdapterError(
                f"simready_supplemental_destination_recipe_binding_invalid:{name}"
            )
        rows[name] = row
        paths[name] = path
    asset_digest, asset_size = _sha256_and_size(paths["asset"])
    static = _load_json(
        paths["static_qualification"],
        code="simready_supplemental_destination_input_invalid:static_qualification",
    )
    rights = _load_json(
        paths["rights_admission"],
        code="simready_supplemental_destination_input_invalid:rights_admission",
    )
    authoring = _load_json(
        paths["authoring_receipt"],
        code="simready_supplemental_destination_input_invalid:authoring_receipt",
    )
    simready = _load_json(
        paths["simready_result"],
        code="simready_supplemental_destination_input_invalid:simready_result",
    )
    completion = authoring.get("candidate_physics_completion")
    interior = simready.get("interior_bounds_body_frame_m")
    support_bodies = simready.get("intended_support_prim_paths")
    support_colliders = simready.get("intended_support_collision_prim_paths")
    structure = static.get("observed_structure") or {}
    if (
        static.get("schema_version") != STATIC_QUALIFICATION_SCHEMA_VERSION
        or static.get("status") != "authored_structure_statically_qualified"
        or static.get("authored_structure_statically_qualified") is not True
        or static.get("structural_findings") != []
        or static.get("replacement_identity") != identity
        or (static.get("replacement_usd") or {}).get("sha256") != asset_digest
        or (static.get("replacement_usd") or {}).get("size_bytes") != asset_size
        or (static.get("claim_boundary") or {}).get("native_simulator_import_qualified")
        is not False
        or static.get("result_digest")
        != canonical_digest(static, digest_field="result_digest")
        or rights.get("schema_version") != DESTINATION_RIGHTS_SCHEMA_VERSION
        or rights.get("status") != "admitted"
        or rights.get("destination_identity") != identity
        or rights.get("private_provider_processing_allowed") is not True
        or rights.get("rights_admission_digest")
        != canonical_digest(rights, digest_field="rights_admission_digest")
        or authoring.get("schema_version")
        != "task_evaluation_rigid_replacement_authoring_result.v1"
        or authoring.get("status") != "authored_candidate_pending_qualification"
        or authoring.get("replacement_identity") != identity
        or authoring.get("physics_authority_granted") is not False
        or (authoring.get("output_usd") or {}).get("sha256") != asset_digest
        or (authoring.get("output_usd") or {}).get("size_bytes") != asset_size
        or authoring.get("result_digest")
        != canonical_digest(authoring, digest_field="result_digest")
        or not isinstance(completion, Mapping)
        or not isinstance(completion.get("physics_bounds"), Mapping)
        or simready.get("schema_version") != PASSIVE_DESTINATION_SIMREADY_SCHEMA_VERSION
        or simready.get("destination_identity") != identity
        or (simready.get("asset") or {}).get("sha256") != asset_digest
        or (simready.get("static_qualification") or {}).get("sha256")
        != rows["static_qualification"]["digest"]
        or (simready.get("authoring_receipt") or {}).get("sha256")
        != rows["authoring_receipt"]["digest"]
        or (simready.get("rights_admission") or {}).get("sha256")
        != rows["rights_admission"]["digest"]
        or simready.get("static_result_digest") != static.get("result_digest")
        or not isinstance(support_bodies, list)
        or not support_bodies
        or any(
            not str(path).startswith("/")
            or path not in (structure.get("rigid_body_paths") or [])
            for path in support_bodies
        )
        or not isinstance(support_colliders, list)
        or not support_colliders
        or any(
            not str(path).startswith("/")
            or path not in (structure.get("collision_prim_paths") or [])
            for path in support_colliders
        )
        or not isinstance(interior, Mapping)
        or simready.get("result_digest")
        != canonical_digest(simready, digest_field="result_digest")
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_supplemental_destination_binding_invalid"
        )
    return {
        "identity": identity,
        "relation": destination.get("relation"),
        "binding": dict(destination),
        "rows": rows,
        "paths": paths,
        "asset_digest": asset_digest,
        "asset_size_bytes": asset_size,
        "static_qualification": static,
        "rights_admission": rights,
        "authoring_receipt": authoring,
        "simready_result": simready,
    }


def _requalification_comparable(receipt: Mapping[str, Any]) -> dict[str, Any]:
    comparable = json.loads(json.dumps(dict(receipt)))
    comparable.pop("result_digest", None)
    usd = comparable.get("replacement_usd")
    if isinstance(usd, dict):
        usd.pop("path", None)
    return comparable


def supplemental_destination_static_artifacts(
    *, envelope: Mapping[str, Any], output_root: Path
) -> list[dict[str, Any]]:
    inputs = supplemental_destination_inputs(envelope)
    if inputs is None:
        return []
    identity = inputs["identity"]
    authoring = inputs["authoring_receipt"]
    graph = {
        "schema_version": "task_evaluation_rigid_replacement_graph.v1",
        "asset_id": identity["id"],
        "asset_version": identity["version"],
        "articulation_graph": {"joints": []},
        "single_rigid_candidate": True,
        "physics_bounds": dict(
            authoring["candidate_physics_completion"]["physics_bounds"]
        ),
        "physics_authority_granted": False,
    }
    requalification_path = (
        output_root / "destination_static_requalification_receipt.v1.json"
    )
    try:
        requalified = qualify_scene_configuration_rigid_asset_static(
            asset_path=inputs["paths"]["asset"],
            graph_spec=graph,
            authoring_receipt=authoring,
            replacement_identity=identity,
            output_path=requalification_path,
        )
    except TaskEvaluationSceneConfigurationStaticQualificationError as exc:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_static_destination_requalification_failed:"
            + ";".join(exc.codes)
        ) from exc
    if _requalification_comparable(requalified) != _requalification_comparable(
        inputs["static_qualification"]
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_static_destination_requalification_mismatch"
        )
    asset = inputs["paths"]["asset"]
    retained_asset = _copy_artifact(
        asset, output_root / f"statically_qualified_destination_asset{asset.suffix}"
    )
    retained_static = _copy_artifact(
        inputs["paths"]["static_qualification"],
        output_root / "destination_static_qualification_receipt.v1.json",
    )
    return [
        {"role": "statically_qualified_destination_asset", **retained_asset},
        {"role": "destination_static_qualification_receipt", **retained_static},
        {
            "role": "destination_static_requalification_receipt",
            "path": str(requalification_path),
            "digest": _sha256_and_size(requalification_path)[0],
            "size_bytes": _sha256_and_size(requalification_path)[1],
        },
    ]


def supplemental_destination_native_artifacts(
    *,
    envelope: Mapping[str, Any],
    dependency_results: tuple[Mapping[str, Any], ...],
    provider_runtime_artifacts: tuple[Mapping[str, Any], ...],
    output_root: Path,
) -> list[dict[str, Any]]:
    """Admit the destination's native readback with the subject's exact rules."""

    recipe = envelope.get("recipe")
    destination = (
        recipe.get("supplemental_destination") if isinstance(recipe, Mapping) else None
    )
    if destination is None:
        return []
    identity = destination.get("identity") if isinstance(destination, Mapping) else None
    if not isinstance(identity, Mapping):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_supplemental_destination_recipe_invalid"
        )
    _asset_record, asset = _dependency_artifact(
        dependency_results, role="statically_qualified_destination_asset"
    )
    _static_record, static_receipt = _dependency_artifact(
        dependency_results, role="destination_static_qualification_receipt"
    )
    _runtime_record, runtime_path = _provider_runtime_artifact(
        provider_runtime_artifacts, role="destination_native_import_runtime_result"
    )
    runtime = _load_json(
        runtime_path, code="simready_native_import_destination_result_invalid"
    )
    if (
        runtime.get("schema_version")
        != "task_evaluation_replacement_native_import_result.v1"
        or runtime.get("status") != "qualified"
        or runtime.get("replacement_identity") != identity
        or runtime.get("asset_digest") != _sha256_and_size(asset)[0]
        or runtime.get("static_qualification_digest")
        != _sha256_and_size(static_receipt)[0]
        or runtime.get("native_isaac_executed") is not True
        or runtime.get("native_simulator_import_qualified") is not True
        or runtime.get("support_contact_observed") is not True
        or runtime.get("deterministic_reset_state_digest_repeat_count") != 3
        or runtime.get("blockers") != []
        or runtime.get("result_digest")
        != canonical_digest(runtime, digest_field="result_digest")
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_native_import_destination_result_invalid"
        )
    retained_asset = _copy_artifact(
        asset, output_root / f"native_qualified_destination_asset{asset.suffix}"
    )
    retained_receipt = _copy_artifact(
        runtime_path,
        output_root / "destination_native_import_qualification_receipt.v1.json",
    )
    return [
        {"role": "native_qualified_destination_asset", **retained_asset},
        {"role": "destination_native_import_qualification_receipt", **retained_receipt},
    ]

__all__ = [
    "DESTINATION_RIGHTS_SCHEMA_VERSION",
    "PASSIVE_DESTINATION_SIMREADY_SCHEMA_VERSION",
    "supplemental_destination_inputs",
    "supplemental_destination_native_artifacts",
    "supplemental_destination_static_artifacts",
]
