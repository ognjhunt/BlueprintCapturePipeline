"""Verify the exact three-layer, sixteen-camera GPU source-calibration return."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image

from .decision_evidence_contracts import canonical_digest, canonical_json, cross_runtime_canonical_digest
from .sealed_camera_render import _camera_specs_from_calibration_file, _standard_ply_vertex_count
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, sha

ROLES = ("images", "target_support", "scene_without_target")
PREPARED_SCHEMA = "public_scene_inpainting_prepared_inputs.v1"
RESULT_SCHEMA = "adp009d_source_calibration_gpu_render_result.v1"
RETURN_SCHEMA = "source_calibration_render_return.v1"


def require(value: bool, code: str) -> None:
    if not value:
        raise ValueError("source_calibration_render_" + code)


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def validate_prepared_render_inputs(value: Mapping[str, Any]) -> dict[str, Any]:
    prepared = dict(value)
    require(prepared.get("schema_version") == PREPARED_SCHEMA
            and prepared.get("status") == "prepared_for_exact_render"
            and prepared.get("preparation_digest") == canonical_digest(prepared, digest_field="preparation_digest"),
            "preparation_invalid")
    require(set(prepared.get("layers", {})) == set(ROLES), "layer_inventory_invalid")
    camera = prepared["camera_file"]
    path = checked_file(camera["path"], camera)
    cameras = _camera_specs_from_calibration_file(path)
    require(len(cameras) == 16 and len({row["id"] for row in cameras}) == 16, "sixteen_cameras_required")
    require(json.loads(path.read_text()) == prepared.get("cameras"), "prepared_cameras_changed")
    for row in cameras:
        require(row["spec"]["intrinsics"]["width"] == 1280
                and row["spec"]["intrinsics"]["height"] == 1280, "master_dimensions_invalid")
    for layer in prepared["layers"].values():
        source = checked_file(layer["path"], layer)
        require(type(layer.get("retained_gaussian_count")) is int
                and layer["retained_gaussian_count"] > 0
                and _standard_ply_vertex_count(source) == layer["retained_gaussian_count"], "layer_count_invalid")
    require(prepared["layers"]["target_support"]["retained_gaussian_count"]
            + prepared["layers"]["scene_without_target"]["retained_gaussian_count"]
            == prepared["layers"]["images"]["retained_gaussian_count"], "layer_partition_count_invalid")
    return prepared


def _local_manifest(result_root: Path, reference: Mapping[str, Any]) -> Path:
    relative = Path(str(reference.get("relative_path") or ""))
    require(not relative.is_absolute() and ".." not in relative.parts and bool(str(relative)), "manifest_path_invalid")
    return checked_file(result_root/relative, dict(reference))


def _read_node_record(path: Path, digest_field: str) -> dict:
    value = read(path)
    require(value.get("digest_canonicalization") == "rfc8785"
            and value.get(digest_field) == cross_runtime_canonical_digest(value, digest_field=digest_field),
            "cross_runtime_digest_invalid")
    return value


def _verify_group(prepared: dict, role: str, manifest_path: Path) -> dict[str, Any]:
    manifest = _read_node_record(manifest_path, "sealed_camera_render_manifest_digest")
    layer = prepared["layers"][role]
    expected_cameras = _camera_specs_from_calibration_file(Path(prepared["camera_file"]["path"]))
    require(manifest.get("schema_version") == "sealed_camera_render_manifest.v1"
            and manifest.get("status") == "rendered_exact_cameras"
            and manifest.get("authorization_class") == "method_input"
            and manifest.get("source_layer_role") == role
            and manifest.get("source_calibration_preparation_digest") == prepared["preparation_digest"]
            and manifest.get("calibrated_cameras") == expected_cameras
            and manifest.get("calibrated_camera_file", {}).get("digest") == prepared["camera_file"]["sha256"]
            and manifest.get("source_splat", {}).get("digest") == layer["sha256"]
            and manifest.get("source_splat", {}).get("retained_gaussian_count") == layer["retained_gaussian_count"],
            "group_scientific_binding_invalid")
    identity = manifest.get("renderer_identity", {})
    graphics = identity.get("graphics_diagnostics", {})
    renderer = str(graphics.get("renderer") or "").lower()
    require(manifest.get("rendered_by_gpu") is True
            and manifest.get("gpu_identity", {}).get("nvidia_smi_detected") is True
            and bool(manifest["gpu_identity"].get("gpu_rows"))
            and identity.get("repository", {}).get("commit") == prepared["repository"]["commit"]
            and identity.get("graphics_backend") == "egl" and graphics.get("webgl_available") is True
            and bool(renderer) and not any(word in renderer for word in ("software", "swiftshader", "llvmpipe")),
            "hardware_identity_invalid")
    require(manifest.get("rendered_by") == "reference_spark_renderer_exact_camera"
            and manifest.get("camera_set_label") == layer["camera_set_label"]
            and manifest.get("purpose") == layer["purpose"]
            and manifest.get("provider_splat_import_receipt_digest") == layer["provider_splat_import_receipt_digest"]
            and manifest.get("alignment_digest") == layer["alignment_digest"]
            and manifest.get("candidate_policy_queried") is False
            and manifest.get("paid_inference_performed") is False, "group_execution_scope_invalid")
    options = prepared["render_options"]
    settings = {"dimensions": {"width": 1280, "height": 1280},
                "supersampling": options.get("supersampling", 1),
                "color_space": options.get("color_space", "srgb"),
                "alpha_mode": options.get("alpha_mode", "opaque_rgb"),
                "background_rgb": f"#{int(options.get('background_rgb', 0)):06x}",
                "exposure": {"mode": options.get("exposure_mode", "renderer_default_unmodified"), "ev": None}}
    require(manifest.get("render_settings") == settings, "render_settings_changed")
    repo = Path(prepared["context"]["paths"]["repo"])
    for name, relative in (("harness_sha256", "render_splat.mjs"),
                           ("render_entry_sha256", "src/render_entry.mjs"),
                           ("package_manifest_sha256", "package.json"),
                           ("package_lock_sha256", "package-lock.json")):
        require(identity.get(name) == sha(repo/"tools/splat_render"/relative), "renderer_source_changed")
    rows = manifest.get("renders", [])
    require(len(rows) == 16 and manifest.get("render_count") == 16
            and {row.get("camera_id") for row in rows} == {row["id"] for row in expected_cameras}, "frame_inventory_invalid")
    for row in rows:
        require(row.get("relative_path") == f"frames/{row['camera_id']}.png"
                and row.get("width") == 1280 and row.get("height") == 1280, "frame_path_or_dimensions_invalid")
        frame = checked_file(manifest_path.parent/row["relative_path"],
                             {"sha256": row["digest"], "size_bytes": row["size_bytes"]})
        with Image.open(frame) as image:
            require(image.size == (1280, 1280), "frame_dimensions_invalid")
    require({p.name for p in (manifest_path.parent/'frames').glob('*.png')}
            == {f"{row['id']}.png" for row in expected_cameras}, "extra_or_missing_frame")
    return {"manifest": manifest, "root": manifest_path.parent, "manifest_path": manifest_path}


def verify_source_calibration_return(prepared_inputs: Mapping[str, Any], returned_group_path: str | Path) -> dict[str, dict]:
    prepared = validate_prepared_render_inputs(prepared_inputs)
    path = Path(returned_group_path)
    returned = read(path, digest_field="return_digest")
    require(returned.get("schema_version") == RETURN_SCHEMA
            and returned.get("status") == "rendered_exact_layer_camera_matrix"
            and returned.get("preparation_digest") == prepared["preparation_digest"]
            and returned.get("blueprint_commit") == prepared["repository"]["commit"]
            and returned.get("candidate_policy_queried") is False
            and returned.get("paid_inference_performed") is False, "return_identity_invalid")
    return _verify_return_value(prepared, returned)


def _verify_return_value(prepared: dict, returned: dict) -> dict[str, dict]:
    result_ref = returned.get("provider_result", {})
    result_path = checked_file(result_ref.get("path", ""), result_ref)
    result = _read_node_record(result_path, "result_digest")
    require(result.get("schema_version") == RESULT_SCHEMA and result.get("status") == "completed"
            and result.get("preparation_digest") == prepared["preparation_digest"]
            and result.get("blueprint_commit") == prepared["repository"]["commit"]
            and result.get("candidate_policy_queried") is False and result.get("paid_inference_performed") is False
            and result.get("provider_mutations_performed") == 0
            and result.get("render_scope") == "source_calibration", "provider_result_invalid")
    groups = result.get("render_groups", [])
    require(len(groups) == 3 and {row.get("role") for row in groups} == set(ROLES), "group_inventory_invalid")
    verified = {row["role"]: _verify_group(prepared, row["role"],
                _local_manifest(result_path.parent, row["manifest"])) for row in groups}
    require(returned.get("render_groups") == {role: record(row["manifest_path"]) for role, row in verified.items()},
            "return_group_paths_changed")
    return verified


def materialize_source_calibration_return(*, prepared_inputs: Mapping[str, Any], result_path: Path,
                                         output_path: Path) -> dict[str, Any]:
    result = _read_node_record(result_path, "result_digest")
    groups = result.get("render_groups", [])
    value = {"schema_version": RETURN_SCHEMA, "status": "rendered_exact_layer_camera_matrix",
             "preparation_digest": prepared_inputs["preparation_digest"],
             "blueprint_commit": prepared_inputs["repository"]["commit"],
             "provider_result": record(result_path),
             "render_groups": {row["role"]: record(_local_manifest(result_path.parent, row["manifest"])) for row in groups},
             "candidate_policy_queried": False, "paid_inference_performed": False, "return_digest": ""}
    value["return_digest"] = canonical_digest(value, digest_field="return_digest")
    _verify_return_value(validate_prepared_render_inputs(prepared_inputs), value)
    with output_path.open('x', encoding='utf-8') as stream:
        stream.write(canonical_json(value)+'\n')
    return value


def require_source_calibration_closure(prepared: Mapping[str, Any], returned_group_path: str | Path) -> dict:
    from .source_calibration_render_closure import require_source_calibration_closure as validate
    return validate(prepared, returned_group_path)


def materialize_source_calibration_closed_return(**kwargs) -> dict:
    from .source_calibration_render_closure import materialize_source_calibration_closed_return as materialize
    return materialize(**kwargs)


def main(argv=None) -> int:
    """Retain verified returned artifacts without launching provider work."""
    import argparse
    from .public_scene_inpainting_preparation import validate_prepared_inputs
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('render-return', 'closed-return'))
    parser.add_argument('--prepared-inputs', required=True)
    parser.add_argument('--provider-result')
    parser.add_argument('--render-return')
    parser.add_argument('--execution-closure')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    prepared = validate_prepared_inputs(args.prepared_inputs)
    if args.operation == 'render-return':
        require(bool(args.provider_result) and not args.render_return and not args.execution_closure,
                'render_return_cli_inputs_invalid')
        materialize_source_calibration_return(prepared_inputs=prepared,
            result_path=Path(args.provider_result), output_path=Path(args.output))
    else:
        require(bool(args.render_return) and bool(args.execution_closure) and not args.provider_result,
                'closed_return_cli_inputs_invalid')
        materialize_source_calibration_closed_return(prepared_inputs=prepared,
            returned_group_path=args.render_return, execution_closure=read(args.execution_closure),
            output_path=args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
