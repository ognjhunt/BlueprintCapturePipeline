"""Native render-only preflight payload for exact mesh/ParticleField attribution."""

from __future__ import annotations

import json
import math
import os
import argparse
from pathlib import Path
from types import SimpleNamespace

from blueprint_pipeline.native_task_composition_diagnostic import (
    CompositionAdapters,
    CompositionDiagnosticError,
    file_record,
    run_composition_diagnostic,
    seal,
    validate_request,
    verify_record,
)

RESULT_FILENAME = "native_task_arena_runtime_preflight.v1.json"


def _array(value):
    return getattr(value, "torch", value)


def _plain(value):
    value = _array(value)
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def native_physics_clock(sim):
    """Read the pinned Isaac Lab public clock, never legacy Isaac Sim fields."""
    step = sim.get_physics_step_count()
    dt = float(sim.get_physics_dt())
    if type(step) is not int or step < 0 or not math.isfinite(dt) or dt <= 0:
        raise CompositionDiagnosticError("composition_native_physics_clock_invalid")
    return {
        "physics_step_index": step,
        "physics_dt_seconds": dt,
        "physics_time_seconds": step * dt,
        "physics_time_basis": "derived_from_native_physics_step_count_times_fixed_dt",
    }


def _geometry_visibility_targets(stage, env_path, appearance_path):
    """Hide geometry leaves, preserving embedded lights and original hidden SAGE."""
    from pxr import Usd, UsdGeom, UsdLux

    targets = {"appearance": [], "native_meshes": []}
    root = stage.GetPrimAtPath(env_path)
    if not root:
        raise CompositionDiagnosticError("composition_native_environment_missing")
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        is_field = str(prim.GetTypeName()) == "ParticleField3DGaussianSplat"
        if not is_field and not prim.IsA(UsdGeom.Gprim):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        original_path = str(prim.GetPath())
        while prim.IsInstanceProxy():
            prim = prim.GetParent()
        if any(
            child.HasAPI(UsdLux.LightAPI)
            for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
        ):
            raise CompositionDiagnosticError("composition_geometry_visibility_would_change_lights")
        key = "appearance" if original_path.startswith(appearance_path + "/") else "native_meshes"
        path = str(prim.GetPath())
        if path not in targets[key]:
            targets[key].append(path)
    if not all(targets.values()):
        raise CompositionDiagnosticError("composition_geometry_classes_missing")
    return targets


def make_native_adapters(*, built, stage, request):
    import carb
    from pxr import Usd, UsdGeom, UsdLux
    from blueprint_pipeline.adp009d_isaac_runtime import _save_camera

    env = built.env.unwrapped
    env_path = str(env.scene.env_prim_paths[0])
    appearance = next(
        row for row in built.plan["objects"] if row["semantic_role"] == "scene_appearance"
    )
    appearance_path = appearance["prim_path"].replace("{ENV_REGEX_NS}", env_path)
    targets = _geometry_visibility_targets(stage, env_path, appearance_path)
    paths = sorted(set(targets["appearance"] + targets["native_meshes"]))
    session_layer = stage.GetSessionLayer()
    camera = env.scene[built.camera_scene_names[request["camera_role"]]]
    settings = carb.settings.get_settings()
    setting_names = (
        "/rtx/rendermode",
        "/omni/rtx/nre/compositing/rendererHints",
        "/rtx/rtpt/gaussian/skipTonemapping/enabled",
        "/renderer/multiGpu/enabled",
        "/UJITSO/geometry",
        "/rtx/post/aa/op",
        "/rtx/post/dlss/execMode",
        "/rtx-transient/dlssg/enabled",
    )
    light_prims = [p for p in stage.Traverse() if p.HasAPI(UsdLux.LightAPI)]

    def snapshot():
        result = {}
        for path in paths:
            attr = UsdGeom.Imageable(stage.GetPrimAtPath(path)).GetVisibilityAttr()
            spec = session_layer.GetAttributeAtPath(attr.GetPath())
            result[path] = {
                "authored": bool(spec and spec.HasInfo("default")),
                "value": str(spec.default) if spec and spec.HasInfo("default") else None,
            }
        return result

    def restore(snapshot):
        with Usd.EditContext(stage, session_layer):
            for path, old in snapshot.items():
                attr = UsdGeom.Imageable(stage.GetPrimAtPath(path)).GetVisibilityAttr()
                if old["authored"]:
                    attr.Set(old["value"])
                else:
                    attr.Clear()

    def apply(label):
        hidden = (
            targets["native_meshes"]
            if label == "appearance_only"
            else targets["appearance"]
            if label == "native_meshes_only"
            else []
        )
        with Usd.EditContext(stage, session_layer):
            for path in hidden:
                UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()

    def fixed_state():
        objects = {}
        for name in ["robot", *built.scene_asset_names.values()]:
            asset = env.scene[name]
            data = getattr(asset, "data", None)
            objects[name] = {
                field: _plain(getattr(data, field))
                for field in ("root_pose_w", "joint_pos")
                if hasattr(data, field)
            }
        return {
            "native_objects": objects,
            **native_physics_clock(env.sim),
            "renderer_settings": {key: settings.get(key) for key in setting_names},
            "lights": {
                str(p.GetPath()): {
                    a.GetName(): str(a.Get())
                    for a in p.GetAttributes()
                    if a.GetName().startswith(("inputs:", "xformOp", "visibility"))
                }
                for p in light_prims
            },
            "camera": {
                "intrinsic_matrix": _plain(camera.data.intrinsic_matrices),
                "position_world_m": _plain(camera.data.pos_w),
                "quaternion_world_opengl_xyzw": _plain(camera.data.quat_w_opengl),
            },
            "geometry_visibility_targets": targets,
        }

    def generation():
        value = _plain(camera.frame)
        return int(value[0] if isinstance(value, list) else value)

    def capture(label, output):
        import numpy as np

        for _ in range(request["render_refresh_count"]):
            env.sim.render()
            camera.update(0.0, force_recompute=True)
        # Explicit ProxyArray -> Torch before the shared legacy AOV writer indexes
        # batch zero. Keep the raw sensor bytes, calibration and label conventions.
        native = camera.data
        data = SimpleNamespace(
            output={key: _array(value) for key, value in native.output.items()},
            info=native.info,
            intrinsic_matrices=_array(native.intrinsic_matrices),
            pos_w=_array(native.pos_w),
            quat_w_opengl=_array(native.quat_w_opengl),
        )
        wrapped = SimpleNamespace(data=data, prim_path=getattr(camera, "prim_path", None))
        clock = native_physics_clock(env.sim)
        row = _save_camera(
            output,
            request["camera_role"],
            wrapped,
            frame_index=0,
            sim_time=clock["physics_time_seconds"],
            require_metric_depth=False,
        )
        row["native_physics_clock"] = clock
        if "distance_to_camera" not in data.output:
            raise CompositionDiagnosticError("composition_native_metric_depth_channel_missing")
        raw = data.output["distance_to_camera"][0].detach().cpu().numpy()
        path = output / "camera_frames" / request["camera_role"] / "000000.distance_to_camera.npy"
        np.save(path, raw, allow_pickle=False)
        finite = np.isfinite(raw)
        row["metric_depth"] = {
            "status": "valid"
            if finite.any() and not (raw[finite] < 0).any()
            else "invalid_native_aov",
            "aov": "distance_to_camera",
            "units": "meter",
            "dtype": str(raw.dtype),
            "path": str(path.relative_to(output)),
            **file_record(path),
        }
        # Preserve native label IDs/dtype (the shared episode writer historically
        # narrows to int32). Its paths, labels and camera-calibration conventions
        # remain shared; every diagnostic AOV retains its actual sensor values.
        semantic = data.output["semantic_segmentation"][0].detach().cpu().numpy()
        if semantic.ndim == 3 and semantic.shape[-1] == 1:
            semantic = semantic[..., 0]
        path = output / row["semantic_segmentation"]["path"]
        np.save(path, semantic, allow_pickle=False)
        ids, counts = np.unique(semantic, return_counts=True)
        row["semantic_segmentation"].update(
            **file_record(path),
            dtype=str(semantic.dtype),
            pixel_counts_by_id={
                str(int(label)): int(count) for label, count in zip(ids, counts, strict=True)
            },
        )
        return row

    return CompositionAdapters(snapshot, apply, restore, fixed_state, capture, generation)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root")
    args = parser.parse_args(argv)
    runtime = (
        Path(args.runtime_root).resolve() if args.runtime_root else Path(__file__).resolve().parent
    )
    output = Path(os.environ.get("BLUEPRINT_ADP_ARENA_OUTPUT_DIR", runtime / "runtime_output"))
    output.mkdir(parents=True, exist_ok=True)
    app = env = None
    result = {
        "schema_version": "native_task_arena_runtime_preflight.v1",
        "status": "blocked",
        "diagnostic_kind": "fixed_native_mesh_particlefield_composition",
        "preflight_only": True,
        "task_motion_executed": False,
        "candidate_policy_queried": False,
        "candidate_outcomes_accessed": False,
        "provider_zero_required_after_return": True,
        "occlusion_qualified": False,
        "blockers": [],
    }
    try:
        from blueprint_pipeline.native_task_arena_construction_worker import (
            _load_and_verify_manifest,
            preflight_native_dependency_matrix,
        )
        from blueprint_pipeline.native_task_arena_runtime import build_native_task_arena_environment
        from blueprint_pipeline.native_task_arena_preconstruction import (
            prepare_native_task_arena_preconstruction,
        )
        from blueprint_pipeline.native_task_isaaclab_launch import (
            launch_native_task_isaaclab,
            NATIVE_TASK_ARENA_DEVICE,
        )
        from blueprint_pipeline.native_task_nurec_render_setup import (
            setup_and_warm_native_nurec_renderer,
        )

        manifest = _load_and_verify_manifest(runtime, expected_execution_mode="runtime_preflight")
        module_entry = next(
            (
                row
                for row in manifest["runtime_modules"]
                if row["relative_path"] == "blueprint_pipeline/" + Path(__file__).name
            ),
            None,
        )
        expected_worker = (
            module_entry["sha256"]
            if args.runtime_root and module_entry
            else manifest["worker_source_sha256"]
        )
        if (
            file_record(__file__)["sha256"] != expected_worker
            or args.runtime_root
            and Path(__file__).resolve() != runtime / "blueprint_pipeline" / Path(__file__).name
        ):
            raise CompositionDiagnosticError("composition_actual_worker_digest_mismatch")
        for row in [*manifest["bound_runtime_inputs"], *manifest["runtime_modules"]]:
            verify_record(runtime / row["relative_path"], row)
        packet = runtime / "native_task_packet"
        for row in manifest["packet_files"]:
            verify_record(packet / row["relative_path"], row)
        request = validate_request(
            json.loads((runtime / "runtime_inputs/composition_request.json").read_text())
        )
        plan_path = runtime / "runtime_inputs/composition_scene_plan.json"
        verify_record(plan_path, request["resolved_scene_plan"])
        plan = json.loads(plan_path.read_text())
        if (
            plan["plan_digest"] != request["resolved_scene_plan_digest"]
            or plan["scenario"]["seed"] != request["seed"]
            or manifest["packet_receipt_digest"] != request["packet_receipt_digest"]
        ):
            raise CompositionDiagnosticError("composition_plan_binding_invalid")
        result.update(
            request_digest=request["request_digest"],
            implementation_commit=manifest["implementation_commit"],
            bundle_input_digest=manifest["input_digest"],
            container_image=manifest["container_image"],
            source_packet_receipt_digest=manifest["packet_receipt_digest"],
        )
        app, launch = launch_native_task_isaaclab(
            output / "native_task_runtime_source_provisioning.v1.json",
            device=NATIVE_TASK_ARENA_DEVICE,
            appearance_render_path="particlefield_3d_gaussian_splat",
        )
        result["isaaclab_launch"] = launch
        dependency = preflight_native_dependency_matrix(robot_id=plan["robot"]["robot_id"])
        if not dependency["all_required_available"]:
            raise CompositionDiagnosticError("composition_native_dependencies_missing")
        preconstruction = prepare_native_task_arena_preconstruction(
            expected_device=NATIVE_TASK_ARENA_DEVICE
        )
        if not preconstruction["passed"]:
            raise CompositionDiagnosticError("composition_native_device_unqualified")
        built = build_native_task_arena_environment(
            plan,
            device=NATIVE_TASK_ARENA_DEVICE,
            bundle_root=packet,
            preconstruction_receipt=preconstruction,
        )
        env = built.env
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        warmup = setup_and_warm_native_nurec_renderer(
            app, stage, require_display_referred_particlefield=True
        )
        result["renderer_warmup"] = warmup
        if not warmup["passed"]:
            raise CompositionDiagnosticError("composition_native_renderer_unqualified")
        env.reset(seed=request["seed"])
        # One reset establishes the original spawn. There is no settle/controller
        # episode and no reset, simulation step or camera selection between passes.
        adapters = make_native_adapters(built=built, stage=stage, request=request)
        diagnostic = run_composition_diagnostic(
            request, output_root=output / "composition", adapters=adapters
        )
        result["composition_diagnostic"] = diagnostic
        result["status"] = "completed" if diagnostic["status"] == "captured" else "blocked"
        result["blockers"] = diagnostic["blockers"]
    except Exception as exc:
        result["blockers"].append(type(exc).__name__ + ":" + str(exc))
    finally:
        seal(result, "result_digest")
        (output / RESULT_FILENAME).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        if env is not None:
            env.close()
        if app is not None:
            app.close()
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
