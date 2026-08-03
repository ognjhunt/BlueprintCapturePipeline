"""Execution handlers for appearance-fidelity supervisor tools."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Mapping

from ..appearance_fidelity import build_appearance_fidelity_qualification
from ..decision_evidence_contracts import canonical_digest
from .phase2_artifacts import write_phase2_artifact


APPEARANCE_TOOL_IDS = (
    "qualify_appearance_fidelity",
    "render_native_3dgs",
)


def appearance_tool_output_schemas(
    output_schema: Callable[[Mapping[str, Any]], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        "qualify_appearance_fidelity": output_schema(
            {
                "contract_present": {"const": True},
                "digest_matches": {"const": True},
                "qualification_digest": {"type": "string"},
                "status": {"enum": ["qualified", "blocked"]},
                "retained_splat_fraction": {"type": "number"},
                "evaluation_render_authorized": {"type": "boolean"},
                "global_decimation_forbidden": {"const": True},
                "claim_ceiling": {"enum": ["qualified_appearance_render", "none"]},
                "proof_state_changed": {"const": False},
            }
        ),
        "render_native_3dgs": output_schema(
            {
                "contract_present": {"const": True},
                "digest_matches": {"const": True},
                "render_result_digest": {"type": "string"},
                "status": {"enum": ["completed", "failed"]},
                "source_appearance_digest": {"type": "string"},
                "native_3dgs": {"const": True},
                "full_resolution_source_preserved": {"const": True},
                "global_decimation_applied": {"const": False},
                "evaluation_render_authorized": {"const": False},
                "claim_ceiling": {"const": "native_appearance_render_candidate"},
                "proof_state_changed": {"const": False},
            }
        ),
    }


def appearance_tool_descriptors(descriptor: Callable[..., Any]) -> tuple[Any, ...]:
    return (
        descriptor(
            "qualify_appearance_fidelity",
            "appearance_fidelity_qualification",
            expected_artifacts=["appearance_fidelity_qualification.v1"],
            input_properties={"appearance_fidelity_qualification_digest": {"type": "string"}},
            required_inputs=["appearance_fidelity_qualification_digest"],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend",
            timeout_seconds=30.0,
            idempotency="digest_bound_deterministic_fidelity_qualification",
        ),
        descriptor(
            "render_native_3dgs",
            "native_3dgs_rendering",
            expected_artifacts=["native_3dgs_render_result.v1"],
            input_properties={"native_3dgs_render_request_digest": {"type": "string"}},
            required_inputs=["native_3dgs_render_request_digest"],
            mutability="reversible_mutation",
            allowed_modes=["execute_non_spend", "execute_preauthorized"],
            minimum_mode="execute_non_spend",
            timeout_seconds=3_600.0,
            idempotency="content_addressed_full_fidelity_native_render",
        ),
    )


def appearance_tool_available(tool_id: str, context: Any) -> bool:
    if tool_id == "qualify_appearance_fidelity":
        return isinstance(getattr(context, "appearance_fidelity_qualification", None), Mapping)
    if tool_id == "render_native_3dgs":
        return isinstance(
            getattr(context, "native_3dgs_render_request", None), Mapping
        ) and callable(getattr(context, "native_3dgs_renderer", None))
    raise ValueError(f"unknown_appearance_tool:{tool_id}")


def execute_appearance_tool(
    *, tool_id: str, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if tool_id == "qualify_appearance_fidelity":
        return qualify_appearance_fidelity(context=context, arguments=arguments)
    if tool_id == "render_native_3dgs":
        return render_native_3dgs(context=context, arguments=arguments)
    raise ValueError(f"unknown_appearance_tool:{tool_id}")


def qualify_appearance_fidelity(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    source_value = getattr(context, "appearance_fidelity_qualification", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("registered_tool_execution_scope_missing:qualify_appearance_fidelity")
    if not isinstance(source_value, Mapping):
        raise ValueError("appearance_fidelity_qualification_not_injected")
    try:
        qualification = build_appearance_fidelity_qualification(source_value)
    except ValueError as exc:
        raise ValueError("appearance_fidelity_qualification_contract_invalid") from exc
    digest = qualification["appearance_fidelity_qualification_digest"]
    if arguments.get("appearance_fidelity_qualification_digest") != digest:
        raise ValueError("registered_tool_source_digest_mismatch:qualify_appearance_fidelity")
    path = write_phase2_artifact(
        root_value,
        "generated/qualify_appearance_fidelity/appearance_fidelity_qualification.json",
        qualification,
    )
    return {
        "contract_present": True,
        "digest_matches": True,
        "qualification_digest": digest,
        "status": qualification["status"],
        "retained_splat_fraction": qualification["retained_splat_fraction"],
        "evaluation_render_authorized": qualification["evaluation_render_authorized"],
        "global_decimation_forbidden": True,
        "claim_ceiling": qualification["claim_ceiling"],
        "proof_state_changed": False,
    }, [
        {
            "artifact_path": str(path.relative_to(Path(root_value))),
            "artifact_digest": digest,
            "artifact_type": "appearance_fidelity_qualification.v1",
        }
    ]


def _sha256_digest(value: Any) -> bool:
    text = str(value or "")
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", text))


def render_native_3dgs(
    *, context: Any, arguments: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root_value = getattr(context, "supervisor_output_dir", None)
    request_value = getattr(context, "native_3dgs_render_request", None)
    renderer = getattr(context, "native_3dgs_renderer", None)
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("registered_tool_execution_scope_missing:render_native_3dgs")
    if not isinstance(request_value, Mapping) or not callable(renderer):
        raise ValueError("native_3dgs_render_runtime_not_injected")
    request = dict(request_value)
    digest_field = "native_3dgs_render_request_digest"
    request_digest = request.get(digest_field)
    if not _sha256_digest(request_digest) or request_digest != canonical_digest(
        request, digest_field=digest_field
    ):
        raise ValueError("native_3dgs_render_request_contract_invalid")
    if arguments.get(digest_field) != request_digest:
        raise ValueError("registered_tool_source_digest_mismatch:render_native_3dgs")
    for field in ("source_appearance_digest", "render_input_digest", "camera_set_digest"):
        if not _sha256_digest(request.get(field)):
            raise ValueError(f"native_3dgs_render_request_{field}_invalid")
    if request.get("global_decimation_applied") is not False:
        raise ValueError("native_3dgs_render_global_decimation_forbidden")
    output_root = Path(root_value) / "generated" / "render_native_3dgs"
    emitted = renderer(request=dict(request), output_root=output_root)
    if not isinstance(emitted, Mapping):
        raise ValueError("native_3dgs_render_result_not_object")
    result = dict(emitted)
    result["schema_version"] = "native_3dgs_render_result.v1"
    errors: list[str] = []
    for field in ("source_appearance_digest", "render_input_digest", "camera_set_digest"):
        if result.get(field) != request.get(field):
            errors.append(f"native_3dgs_render_{field}_lineage_mismatch")
    for field in ("implementation_digest", "runtime_digest"):
        if not _sha256_digest(result.get(field)):
            errors.append(f"native_3dgs_render_{field}_invalid")
    if result.get("status") not in {"completed", "failed"}:
        errors.append("native_3dgs_render_status_invalid")
    if result.get("native_3dgs") is not True:
        errors.append("native_3dgs_renderer_required")
    if result.get("full_anisotropic_gaussians") is not True:
        errors.append("native_3dgs_full_anisotropic_gaussians_required")
    if result.get("full_resolution_source_preserved") is not True:
        errors.append("native_3dgs_full_resolution_source_not_preserved")
    if result.get("global_decimation_applied") is not False:
        errors.append("native_3dgs_global_decimation_forbidden")
    if result.get("status") == "completed" and not _sha256_digest(
        result.get("frame_manifest_digest")
    ):
        errors.append("native_3dgs_frame_manifest_digest_invalid")
    if errors:
        raise ValueError(";".join(sorted(errors)))
    result["evaluation_render_authorized"] = False
    result["claim_ceiling"] = "native_appearance_render_candidate"
    result_digest = canonical_digest(result, digest_field="native_3dgs_render_result_digest")
    result["native_3dgs_render_result_digest"] = result_digest
    path = write_phase2_artifact(
        root_value,
        "generated/render_native_3dgs/native_3dgs_render_result.json",
        result,
    )
    return {
        "contract_present": True,
        "digest_matches": True,
        "render_result_digest": result_digest,
        "status": result["status"],
        "source_appearance_digest": result["source_appearance_digest"],
        "native_3dgs": True,
        "full_resolution_source_preserved": True,
        "global_decimation_applied": False,
        "evaluation_render_authorized": False,
        "claim_ceiling": "native_appearance_render_candidate",
        "proof_state_changed": False,
    }, [
        {
            "artifact_path": str(path.relative_to(Path(root_value))),
            "artifact_digest": result_digest,
            "artifact_type": "native_3dgs_render_result.v1",
        }
    ]


__all__ = [
    "APPEARANCE_TOOL_IDS",
    "appearance_tool_available",
    "appearance_tool_descriptors",
    "appearance_tool_output_schemas",
    "execute_appearance_tool",
    "qualify_appearance_fidelity",
    "render_native_3dgs",
]
