"""Materialize a portable, digest-bound native Arena construction packet.

The task request names evidence-root-relative source bytes and contains only
scene/task data.  This module verifies and copies those bytes, then invokes the
shared runtime-contract and Arena-plan compilers.  It never imports Isaac and
does not claim native application; the provider worker still owns that gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .common import usd_payload_format_matches, write_json
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_scene_plan import (
    materialize_native_task_arena_scene_plan,
)
from .native_task_arena_runtime import author_gpu_compatible_scene_collision
from .native_task_execution_admission import seal_native_task_execution_admission
from .native_task_runtime_contract import materialize_native_task_runtime_contract
from .paired_target_native_construction_bindings import (
    SCHEMA_VERSION as PAIRED_CONSTRUCTION_SCHEMA_VERSION,
)


REQUEST_SCHEMA_VERSION = "native_task_arena_packet_request.v1"
RECEIPT_SCHEMA_VERSION = "native_task_arena_packet_receipt.v1"
CONSTRUCTION_CANARY_SCHEMA_VERSION = "native_task_construction_canary.v1"
EVALUATION_INSTANCE_SCHEMA_VERSION = "adp009d_scenario_instance.v1"


class NativeTaskArenaPacketError(ValueError):
    """Stable packet-construction failures before native execution."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _has_symlink_component(path: Path, *, root: Path) -> bool:
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _clone_request(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        request = json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise NativeTaskArenaPacketError(
            ["native_task_arena_packet_request_invalid"]
        ) from exc
    if (
        not isinstance(request, dict)
        or request.get("schema_version") != REQUEST_SCHEMA_VERSION
    ):
        raise NativeTaskArenaPacketError(
            ["native_task_arena_packet_request_invalid"]
        )
    if request.get("request_digest") != canonical_digest(
        request, digest_field="request_digest"
    ):
        raise NativeTaskArenaPacketError(
            ["native_task_arena_packet_request_digest_invalid"]
        )
    return request


def validate_native_task_arena_packet_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and clone a packet request without materializing its assets."""

    return _clone_request(value)


def materialize_native_task_arena_appearance_variant_request(
    *,
    base_request_path: str | Path,
    appearance_authoring_receipt_path: str | Path,
    evidence_root: str | Path,
    output_path: str | Path,
    filename: str = "scene_appearance.usdc",
) -> dict[str, Any]:
    """Derive one packet request with a sealed ParticleField appearance."""

    root = Path(evidence_root).expanduser().resolve()
    try:
        base = json.loads(Path(base_request_path).read_text(encoding="utf-8"))
        appearance = json.loads(
            Path(appearance_authoring_receipt_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeTaskArenaPacketError(
            ["native_task_arena_appearance_variant_input_invalid"]
        ) from exc
    request = _clone_request(base)
    sh_degree = appearance.get("sh_degree")
    sh_element_size = (
        (sh_degree + 1) ** 2
        if isinstance(sh_degree, int)
        and not isinstance(sh_degree, bool)
        and 0 <= sh_degree <= 3
        else None
    )
    if (
        appearance.get("schema_version")
        != "particlefield_3dgs_authoring_receipt.v1"
        or appearance.get("status") != "completed"
        or appearance.get("schema") != "ParticleField3DGaussianSplat"
        or appearance.get("sh_primvar_element_size") != sh_element_size
        or appearance.get("sh_primvar_interpolation") != "vertex"
        or appearance.get("display_color_fallback_authored") is not True
        or appearance.get("particlefield_emissive_material_binding_authored")
        is not True
        or appearance.get("particlefield_emissive_material_inputs")
        != "mdl_defaults"
        or appearance.get("receipt_digest")
        != canonical_digest(appearance, digest_field="receipt_digest")
    ):
        raise NativeTaskArenaPacketError(
            ["native_task_arena_appearance_variant_receipt_invalid"]
        )
    asset = Path(str(appearance.get("output") or "")).expanduser().resolve()
    outside = asset != root and root not in asset.parents
    if (
        outside
        or asset.is_symlink()
        or not asset.is_file()
        or asset.stat().st_size != int(appearance.get("output_bytes") or 0)
        or _sha256(asset) != appearance.get("output_sha256")
    ):
        raise NativeTaskArenaPacketError(
            ["native_task_arena_appearance_variant_asset_invalid"]
        )
    relative = asset.relative_to(root).as_posix()
    rows = [
        row
        for row in request.get("assets") or []
        if isinstance(row, dict) and row.get("semantic_role") == "scene_appearance"
    ]
    if len(rows) != 1:
        raise NativeTaskArenaPacketError(
            ["native_task_arena_appearance_variant_role_not_exact"]
        )
    rows[0]["filename"] = filename
    rows[0]["source"] = {
        "root": "evidence",
        "relative_path": relative,
        "size_bytes": asset.stat().st_size,
        "sha256": appearance["output_sha256"],
    }
    request["appearance_variant"] = {
        "base_request_digest": base["request_digest"],
        "representation": "particlefield_3d_gaussian_splat",
        "authoring_receipt_digest": appearance["receipt_digest"],
        "source_gaussian_sha256": appearance.get("source_sha256"),
        "splat_count": appearance.get("splat_count"),
        "sh_degree": sh_degree,
        "sh_primvar_element_size": sh_element_size,
        "sh_primvar_interpolation": "vertex",
        "display_color_fallback_authored": True,
        "particlefield_emissive_material_binding_authored": True,
        "particlefield_emissive_material_inputs": "mdl_defaults",
    }
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    write_json(Path(output_path), request)
    return request


def _asset_source(
    row: Mapping[str, Any], *, evidence_root: Path
) -> tuple[Path, str, int]:
    role = str(row.get("semantic_role") or "")
    source = row.get("source")
    if not isinstance(source, Mapping) or source.get("root") != "evidence":
        raise NativeTaskArenaPacketError(
            [f"native_task_arena_packet_asset_source_invalid:{role}"]
        )
    relative = str(source.get("relative_path") or "")
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or ".." in pure.parts
        or pure.name in {"", ".", ".."}
    ):
        raise NativeTaskArenaPacketError(
            [f"native_task_arena_packet_asset_source_invalid:{role}"]
        )
    candidate = evidence_root.joinpath(*pure.parts)
    resolved = candidate.resolve()
    outside = resolved != evidence_root and evidence_root not in resolved.parents
    if (
        _has_symlink_component(candidate, root=evidence_root)
        or not resolved.is_file()
        or outside
    ):
        raise NativeTaskArenaPacketError(
            [f"native_task_arena_packet_asset_source_missing:{role}"]
        )
    try:
        size = int(source["size_bytes"])
        digest = str(source["sha256"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeTaskArenaPacketError(
            [f"native_task_arena_packet_asset_identity_invalid:{role}"]
        ) from exc
    if size <= 0 or resolved.stat().st_size != size or _sha256(resolved) != digest:
        raise NativeTaskArenaPacketError(
            [f"native_task_arena_packet_asset_identity_mismatch:{role}"]
        )
    return resolved, digest, size


def _validated_scenario_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeTaskArenaPacketError(
            ["native_task_arena_packet_scenario_invalid"]
        )
    scenario = json.loads(json.dumps(value))
    kind = str(scenario.get("context_kind") or "")
    expected_schema, digest_field = {
        "construction_canary": (CONSTRUCTION_CANARY_SCHEMA_VERSION, "context_digest"),
        "evaluation_cell": (EVALUATION_INSTANCE_SCHEMA_VERSION, "instance_digest"),
    }.get(kind, (None, None))
    document = scenario.get("context_document")
    if not isinstance(document, Mapping) or expected_schema is None or digest_field is None:
        raise NativeTaskArenaPacketError(
            ["native_task_arena_packet_scenario_invalid"]
        )
    context = json.loads(json.dumps(document))
    errors: list[str] = []
    if context.get("schema_version") != expected_schema:
        errors.append("native_task_arena_packet_scenario_schema_invalid")
    if (
        context.get("cell_id") != scenario.get("cell_id")
        or context.get("seed") != scenario.get("seed")
    ):
        errors.append("native_task_arena_packet_scenario_binding_mismatch")
    expected_digest = canonical_digest(context, digest_field=digest_field)
    if (
        context.get(digest_field) != expected_digest
        or scenario.get("instance_digest") != expected_digest
    ):
        errors.append("native_task_arena_packet_scenario_digest_invalid")
    if context.get("policy_neutral") is not True:
        errors.append("native_task_arena_packet_scenario_policy_neutrality_invalid")
    if context.get("caller_asserted_success") is not False:
        errors.append("native_task_arena_packet_scenario_caller_success_invalid")
    if (
        kind == "construction_canary"
        and context.get("learned_policy_outcomes_consulted") is not False
    ):
        errors.append("native_task_arena_packet_scenario_policy_leakage")
    if errors:
        raise NativeTaskArenaPacketError(errors)
    return scenario


def _scenario_parameter_bindings(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    context = scenario["context_document"]
    records = context.get("factor_records") or []
    resolved = context.get("resolved_parameters") or {}
    if not isinstance(records, list) or not isinstance(resolved, Mapping):
        raise NativeTaskArenaPacketError(
            ["native_task_arena_scenario_parameters_invalid"]
        )
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise NativeTaskArenaPacketError(
                [f"native_task_arena_scenario_parameter_invalid:{index}"]
            )
        parameter_id = str(record.get("parameter_id") or "")
        target = str(record.get("runtime_target") or "")
        unit = str(record.get("unit") or "")
        try:
            nominal = float(record["nominal_value"])
            value = float(record["resolved_value"])
            tolerance = float(record.get("application_tolerance"))
        except (KeyError, TypeError, ValueError):
            tolerance = {
                "m": 1.0e-4,
                "degrees": 1.0e-3,
                "ratio": 1.0e-6,
                "K": 0.5,
                "kg": 1.0e-4,
                "coefficient": 1.0e-6,
            }.get(unit, float("nan"))
            try:
                nominal = float(record["nominal_value"])
                value = float(record["resolved_value"])
            except (KeyError, TypeError, ValueError) as exc:
                raise NativeTaskArenaPacketError(
                    [f"native_task_arena_scenario_parameter_invalid:{index}"]
                ) from exc
        try:
            context_value = float(resolved[parameter_id])
        except (KeyError, TypeError, ValueError):
            context_value = float("nan")
        if (
            not parameter_id
            or not target
            or not unit
            or not all(math.isfinite(item) for item in (nominal, value, tolerance))
            or tolerance <= 0.0
            or context_value != value
        ):
            raise NativeTaskArenaPacketError(
                [f"native_task_arena_scenario_parameter_invalid:{index}"]
            )
        rows.append(
            {
                "parameter_id": parameter_id,
                "runtime_target": target,
                "unit": unit,
                "nominal_value": nominal,
                "resolved_value": value,
                "application_tolerance": tolerance,
            }
        )
    return rows


def materialize_native_task_arena_packet(
    *,
    request: Mapping[str, Any],
    evidence_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Verify source bytes and build one immutable local construction packet."""

    frozen = _clone_request(request)
    evidence = Path(evidence_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not evidence.is_dir() or evidence.is_symlink():
        raise NativeTaskArenaPacketError(
            ["native_task_arena_packet_evidence_root_invalid"]
        )
    if output.exists():
        raise NativeTaskArenaPacketError(
            ["native_task_arena_packet_output_exists"]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    assets_dir = output / "assets"
    assets_dir.mkdir()
    source_bindings: list[dict[str, Any]] = []
    runtime_assets: list[dict[str, Any]] = []
    try:
        raw_assets = frozen.get("assets")
        if not isinstance(raw_assets, list) or not raw_assets:
            raise NativeTaskArenaPacketError(
                ["native_task_arena_packet_assets_invalid"]
            )
        for raw in raw_assets:
            if not isinstance(raw, Mapping):
                raise NativeTaskArenaPacketError(
                    ["native_task_arena_packet_assets_invalid"]
                )
            role = str(raw.get("semantic_role") or "")
            filename = str(raw.get("filename") or "")
            if PurePosixPath(filename).name != filename or not filename:
                raise NativeTaskArenaPacketError(
                    [f"native_task_arena_packet_asset_filename_invalid:{role}"]
                )
            source_path, source_digest, source_size = _asset_source(
                raw, evidence_root=evidence
            )
            if not usd_payload_format_matches(source_path, filename):
                raise NativeTaskArenaPacketError(
                    [f"native_task_arena_packet_asset_format_invalid:{role}"]
                )
            destination = assets_dir / filename
            if destination.exists():
                raise NativeTaskArenaPacketError(
                    [f"native_task_arena_packet_asset_filename_duplicate:{filename}"]
                )
            collision_adaptation = None
            if role == "scene_collision":
                try:
                    collision_adaptation = author_gpu_compatible_scene_collision(
                        source_path, destination
                    )
                except Exception as exc:
                    raise NativeTaskArenaPacketError(
                        ["native_task_arena_packet_scene_collision_adaptation_failed"]
                    ) from exc
            if collision_adaptation is None:
                shutil.copyfile(source_path, destination)
            staged_size = destination.stat().st_size
            staged_digest = _sha256(destination)
            if collision_adaptation is None and (
                staged_size != source_size or staged_digest != source_digest
            ):
                raise NativeTaskArenaPacketError(
                    [f"native_task_arena_packet_asset_copy_mismatch:{role}"]
                )
            source = dict(raw["source"])
            source_bindings.append(
                {
                    "semantic_role": role,
                    "asset_id": raw.get("source_asset_id", raw.get("asset_id")),
                    **(
                        {"runtime_asset_id": raw.get("asset_id")}
                        if raw.get("source_asset_id") is not None
                        else {}
                    ),
                    "source": source,
                    "staged_relative_path": f"assets/{filename}",
                    "staged_size_bytes": staged_size,
                    "staged_sha256": staged_digest,
                    **(
                        {"static_scene_collision_adaptation": collision_adaptation}
                        if collision_adaptation is not None
                        else {}
                    ),
                }
            )
            runtime_assets.append(
                {
                    "semantic_role": role,
                    "name": str(raw.get("name") or role),
                    **(
                        {"asset_id": raw.get("asset_id")}
                        if raw.get("asset_id") is not None
                        else {}
                    ),
                    **(
                        {"object_type": raw.get("object_type")}
                        if raw.get("object_type") is not None
                        else {}
                    ),
                    **(
                        {"reset_state": raw.get("reset_state")}
                        if raw.get("reset_state") is not None
                        else {}
                    ),
                    "visible": bool(raw.get("visible", True)),
                    "filename": filename,
                    "sha256": staged_digest,
                    "pose_world": raw.get("pose_world"),
                    # the grounded articulated asset's declared derivation --
                    # the runtime contract joins the GPU collision
                    # qualification through it
                    **(
                        {
                            "articulation_adaptation": dict(
                                raw["articulation_adaptation"]
                            )
                        }
                        if isinstance(raw.get("articulation_adaptation"), Mapping)
                        else {}
                    ),
                }
            )

        scenario = _validated_scenario_context(frozen.get("scenario"))
        scenario_parameter_bindings = _scenario_parameter_bindings(scenario)
        contract_path = output / "native_task_runtime_contract.v1.json"
        contract = materialize_native_task_runtime_contract(
            scene_id=str(frozen.get("scene_id") or ""),
            task_id=str(frozen.get("task_id") or ""),
            task_spec=frozen.get("task_spec") or {},
            task_joint_bindings=frozen.get("task_joint_bindings"),
            task_state_binding=frozen.get("task_state_binding"),
            assets=runtime_assets,
            robot_base_pose_world=frozen.get("robot_base_pose_world") or {},
            robot_joint_reset_positions_rad=(
                frozen.get("robot_joint_reset_positions_rad") or {}
            ),
            cameras=frozen.get("cameras") or [],
            scenario_cell_id=str(scenario.get("cell_id") or ""),
            scenario_instance_digest=str(scenario.get("instance_digest") or ""),
            seed=scenario.get("seed"),
            scenario_context_kind=str(
                scenario.get("context_kind") or "evaluation_cell"
            ),
            construction_bindings=frozen.get("construction_bindings"),
            task_freeze_digest=frozen.get("task_freeze_digest"),
            scenario_parameter_bindings=scenario_parameter_bindings,
            destination=contract_path,
        )
        plan_path = output / "native_task_arena_scene_plan.v1.json"
        plan = materialize_native_task_arena_scene_plan(
            runtime_contract=contract,
            provider_asset_directory=assets_dir,
            physics_frequency_hz=frozen.get("physics_frequency_hz"),
            published_asset_directory="assets",
            destination=plan_path,
        )
        write_json(output / "native_task_arena_packet_request.v1.json", frozen)
        artifacts = [
            {
                "role": role,
                "relative_path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for role, path in (
                ("packet_request", output / "native_task_arena_packet_request.v1.json"),
                ("runtime_contract", contract_path),
                ("arena_scene_plan", plan_path),
            )
        ]
        receipt: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "construction_packet_completed",
            "scene_id": contract["scene_id"],
            "task_id": contract["task_id"],
            "request_digest": frozen["request_digest"],
            "runtime_contract_digest": contract["contract_digest"],
            "arena_scene_plan_digest": plan["plan_digest"],
            "task_freeze_digest": contract.get("task_freeze_digest"),
            "shared_construction_digest": (
                (contract.get("construction_bindings") or {}).get(
                    "construction_digest"
                )
            ),
            "scenario_instance_digest": scenario["instance_digest"],
            "source_bindings": source_bindings,
            "artifacts": artifacts,
            "source_bytes_mutated": False,
            "native_application_claimed": False,
            "policy_episode_claimed": False,
            "simulator_execution_is_not_physical_truth": True,
            "receipt_digest": "",
        }
        for binding, raw in zip(source_bindings, raw_assets, strict=True):
            path, digest, size = _asset_source(raw, evidence_root=evidence)
            if digest != binding["source"]["sha256"] or _sha256(path) != digest:
                raise NativeTaskArenaPacketError(
                    ["native_task_arena_packet_sealed_source_mutated"]
                )
            if (
                "static_scene_collision_adaptation" not in binding
                and (
                    digest != binding["staged_sha256"]
                    or size != binding["staged_size_bytes"]
                )
            ):
                raise NativeTaskArenaPacketError(
                    ["native_task_arena_packet_asset_copy_mismatch"]
                )
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
        write_json(output / "native_task_arena_packet_receipt.v1.json", receipt)
        construction = contract.get("construction_bindings") or {}
        if construction.get("schema_version") == PAIRED_CONSTRUCTION_SCHEMA_VERSION:
            candidate_record = construction.get("native_execution_candidate") or {}
            runtime_record = construction.get("native_import_result") or {}
            try:
                candidate = json.loads(
                    Path(str(candidate_record.get("path") or "")).read_text(
                        encoding="utf-8"
                    )
                )
                runtime_result = json.loads(
                    Path(str(runtime_record.get("path") or "")).read_text(
                        encoding="utf-8"
                    )
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise NativeTaskArenaPacketError(
                    ["native_task_execution_admission_evidence_missing"]
                ) from exc
            seal_native_task_execution_admission(
                candidate=candidate,
                runtime_result=runtime_result,
                packet_receipt=receipt,
                scene_plan=plan,
                task_id=contract["task_id"],
                destination=output / "native_task_execution_admission.v1.json",
            )
        return json.loads(json.dumps(receipt))
    except Exception:
        shutil.rmtree(output)
        raise


__all__ = [
    "NativeTaskArenaPacketError",
    "CONSTRUCTION_CANARY_SCHEMA_VERSION",
    "EVALUATION_INSTANCE_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "materialize_native_task_arena_appearance_variant_request",
    "materialize_native_task_arena_packet",
    "validate_native_task_arena_packet_request",
]
