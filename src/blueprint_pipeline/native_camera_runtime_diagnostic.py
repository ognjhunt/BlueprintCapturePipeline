"""Bounded camera/GPU diagnostic on explicit synthetic geometry; never a task."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from .decision_evidence_contracts import canonical_digest
from .scene_component_process import run_component_process

FILENAME = "native_camera_runtime_diagnostic.v1.json"
MAX_SECONDS = 90.
MAX_TICKS = 120


class _KitCamera:
    native = True

    def __init__(self, app):
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux
        self.app, self.rep = app, rep
        self.stage = omni.usd.get_context().get_stage()
        if self.stage is None:
            omni.usd.get_context().new_stage()
            self.stage = omni.usd.get_context().get_stage()
        self.root = "/BlueprintRuntimeDiagnostic"
        if self.stage.GetPrimAtPath(self.root).IsValid():
            raise ValueError("camera_diagnostic_prim_already_exists")
        UsdGeom.Xform.Define(self.stage, self.root)
        cube = UsdGeom.Cube.Define(self.stage, self.root + "/Cube")
        cube.CreateSizeAttr(1.)
        cube.CreateDisplayColorAttr([(0.8, 0.1, 0.1)])
        light = UsdLux.DomeLight.Define(self.stage, self.root + "/Light")
        light.CreateIntensityAttr(500.)
        camera = UsdGeom.Camera.Define(self.stage, self.root + "/Camera")
        camera.AddTranslateOp().Set(Gf.Vec3d(0., 0., 3.))
        camera.CreateClippingRangeAttr(Gf.Vec2f(.1, 10.))
        camera.CreateFocalLengthAttr(24.)
        self.product = rep.create.render_product(str(camera.GetPath()), (64, 64))
        self.annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cuda")
        self.annotator.attach([self.product])

    def frame(self):
        import torch
        import warp
        self.app.update()
        data = self.annotator.get_data()
        if data is None or not hasattr(data, "shape") or len(data.shape) < 3:
            return None
        tensor = data if isinstance(data, torch.Tensor) else warp.to_torch(data)
        if tensor.numel() == 0:
            return None
        rgb = tensor[..., :3].contiguous()
        return {"shape": list(rgb.shape), "device": str(rgb.device), "dtype": str(rgb.dtype),
                "finite": bool(torch.isfinite(rgb).all().item()), "minimum": float(rgb.min().item()),
                "maximum": float(rgb.max().item()),
                "sha256": "sha256:" + hashlib.sha256(rgb.cpu().numpy().tobytes()).hexdigest()}

    def close(self):
        self.annotator.detach([self.product])
        self.product.destroy()
        self.stage.RemovePrim(self.root)


def run_camera_diagnostic(app, output: Path, *, deadline_epoch: float, backend_factory=_KitCamera,
                          wall=time.time):
    if not math.isfinite(deadline_epoch):
        raise ValueError("camera_diagnostic_deadline_invalid")
    started = wall()
    deadline = min(float(deadline_epoch), started + MAX_SECONDS)
    inherited = os.environ.get("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH")
    if inherited:
        deadline = min(deadline, float(inherited) - 600.)
    result = {"schema_version": "native_camera_runtime_diagnostic.v1", "status": "failed",
              "claim_ceiling": "synthetic_camera_gpu_runtime_diagnostic", "task_executed": False,
              "policy_executed": False, "site_appearance_exercised": False, "native_isaac_executed": False,
              "deadline_epoch": deadline, "maximum_ticks": MAX_TICKS, "ticks": 0, "reasons": []}
    backend = None
    try:
        if wall() >= deadline:
            raise TimeoutError("camera_diagnostic_deadline_exhausted")
        backend = backend_factory(app)
        result["native_isaac_executed"] = backend.native is True
        for tick in range(MAX_TICKS):
            if wall() >= deadline:
                raise TimeoutError("camera_diagnostic_deadline_exhausted")
            frame = backend.frame()
            result["ticks"] = tick + 1
            if wall() >= deadline:
                raise TimeoutError("camera_diagnostic_deadline_exhausted")
            if frame is None:
                continue
            result["frame"] = frame
            if (frame.get("shape") == [64, 64, 3] and frame.get("device") == "cuda:0"
                    and frame.get("finite") is True and frame.get("dtype") == "torch.uint8"
                    and 0 <= frame["minimum"] < frame["maximum"] <= 255 and frame.get("sha256")):
                result["status"] = "passed"
                break
        if result["status"] != "passed":
            result["reasons"].append("camera_diagnostic_gpu_frame_unqualified")
    except Exception as exc:
        result["reasons"].append(type(exc).__name__ + ":" + str(exc))
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:
                result["status"] = "failed"
                result["reasons"].append("camera_diagnostic_cleanup_failed:" + type(exc).__name__)
    result["elapsed_seconds"] = max(0., wall() - started)
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    output.mkdir(parents=True, exist_ok=True)
    (output / FILENAME).write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_bounded_camera_process(*, provisioning_receipt: Path, output: Path, deadline_epoch: float,
                               controller_backend="joint", run=run_component_process, wall=time.time):
    """Bound blocked Kit updates/startup on an admitted GPU; allocate no resource."""
    if not math.isfinite(deadline_epoch):
        raise ValueError("camera_diagnostic_deadline_invalid")
    if (output / FILENAME).exists():
        return {"status": "failed", "reason": "camera_diagnostic_result_already_exists"}
    deadline = min(deadline_epoch, wall() + MAX_SECONDS)
    inherited = os.environ.get("BLUEPRINT_POLICY_CANARY_HARD_DEADLINE_EPOCH")
    if inherited:
        deadline = min(deadline, float(inherited) - 600.)
    if deadline - wall() <= 5:
        return {"status": "not_run", "reason": "camera_diagnostic_deadline_exhausted"}
    output.mkdir(parents=True, exist_ok=True)
    command = ["/isaac-sim/python.sh", "-m", "blueprint_pipeline.native_camera_runtime_diagnostic",
               "--provisioning-receipt", str(provisioning_receipt), "--output", str(output),
               "--deadline-epoch", str(deadline - 5.), "--controller-backend", controller_backend]
    try:
        with (output / "camera-runtime.log").open("w") as log:
            completed = run(command, stdout=log, stderr=subprocess.STDOUT, capture_output=False,
                            timeout=max(.1, deadline - wall()), check=False)
        if completed.returncode != 0:
            return {"status": "failed", "reason": "camera_diagnostic_process_nonzero", "returncode": completed.returncode}
        result = json.loads((output / FILENAME).read_text())
        if (result.get("result_digest") != canonical_digest(result, digest_field="result_digest")
                or result.get("task_executed") is not False or result.get("policy_executed") is not False
                or result.get("site_appearance_exercised") is not False):
            raise ValueError("camera_diagnostic_result_changed")
        return {"status": result["status"], "returncode": completed.returncode, "result_digest": result["result_digest"]}
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        return {"status": "failed", "reason": "camera_diagnostic_process_failed:" + type(exc).__name__}


def main(argv=None):
    from .native_task_isaaclab_launch import launch_native_task_isaaclab
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provisioning-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--controller-backend", choices=("joint", "g1_joint", "pink_franka"), default="joint")
    args = parser.parse_args(argv)
    if time.time() >= args.deadline_epoch:
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        app, launch = launch_native_task_isaaclab(args.provisioning_receipt, device="cuda:0",
            appearance_render_path="usd_geometry", controller_backend=args.controller_backend, enable_cameras=True)
    except Exception as exc:
        (args.output / "launch.json").write_text(json.dumps({"status": "failed", "error_type": type(exc).__name__,
            "reason": str(exc), "diagnostics": getattr(exc, "diagnostics", {}), "task_executed": False,
            "policy_executed": False}, indent=2) + "\n")
        return 1
    try:
        (args.output / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
        result = run_camera_diagnostic(app, args.output, deadline_epoch=args.deadline_epoch)
        return 0 if result["status"] == "passed" else 1
    finally:
        # Required evidence above is sealed before process-terminating SDK close.
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
