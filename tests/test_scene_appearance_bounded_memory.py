"""The appearance step reads, measures, presses and writes a large Gaussian field in bounded blocks.

The field is generated here (several million particles, nothing committed). The step runs in a
fresh interpreter under ``tracemalloc``: the arrays it allocates stay under a bound that does not
grow with the field beyond the pressed positions USD needs whole (12 bytes a particle), where the
whole-array code it replaced held about 220 bytes a particle (890 MB for 4 million). A small
block size gives the same files and records as the default one.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
PARTICLES = 3_000_000
PLANE = 0.8
# The pressed positions are authored whole (float32, 12 bytes a particle); everything else is a
# fixed working set of blocks. Measured peak: 53 MB; the whole-array code allocated 703 MB here.
BOUND_BYTES = 12 * PARTICLES + 48 * 2**20

GENERATE = textwrap.dedent("""
    import sys
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    path, count, plane = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    rng = np.random.default_rng(7)
    positions = rng.uniform([-3, -3, 0], [3, 3, 2.5], size=(count, 3)).astype(np.float32)
    table = count // 250  # 0.4 % of the field: table particles reaching up under the object
    positions[:table] = np.column_stack([rng.uniform(-0.14, 0.14, table), rng.uniform(-0.24, 0.24, table),
                                         rng.uniform(plane - 0.03, plane - 0.01, table)])
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())
    prim = stage.DefinePrim("/World/Gaussians", "ParticleField3DGaussianSplat")
    frame = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0.3, 0.2, 1.0), 2.0))
    frame.SetTranslateOnly(Gf.Vec3d(0.004, -0.003, 0.001))
    UsdGeom.Xformable(prim).AddTransformOp().Set(frame)
    prim.CreateAttribute("positions", Sdf.ValueTypeNames.Point3fArray).Set(Vt.Vec3fArray.FromNumpy(positions))
    del positions
    prim.CreateAttribute("scales", Sdf.ValueTypeNames.Float3Array).Set(
        Vt.Vec3fArray.FromNumpy(rng.uniform(0.002, 0.01, (count, 3)).astype(np.float32)))
    prim.CreateAttribute("orientations", Sdf.ValueTypeNames.QuatfArray).Set(
        Vt.QuatfArray.FromNumpy(np.tile(np.array([1, 0, 0, 0], np.float32), (count, 1))))
    prim.CreateAttribute("opacities", Sdf.ValueTypeNames.FloatArray).Set(
        Vt.FloatArray.FromNumpy(rng.uniform(0.05, 1.0, count).astype(np.float32)))
    stage.Save()
""")

RUN = textwrap.dedent("""
    import json, sys, tracemalloc
    from pathlib import Path
    from blueprint_pipeline import gaussian_field_quality, scene_appearance_task_clearance
    from blueprint_pipeline import task_evaluation_native_arena_episode_compiler as compiler

    source, out, plane, rows = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    if rows:
        gaussian_field_quality._blocks.__defaults__ = (rows,)
        scene_appearance_task_clearance.CHUNK_ROWS = rows
        scene_appearance_task_clearance.streamed_clearance_plan.__kwdefaults__["rows"] = rows
    spec = {"start_pose_world": [0.0, 0.0, plane + 0.01, 0.0, 0.0, 0.0, 1.0],
            "subject_collision_bounds_scoring_frame_m": {"minimum": [-0.15, -0.25, -0.01],
                                                         "maximum": [0.15, 0.25, 0.03]},
            "visible_target_marker": {"radius_m": 0.3, "surface_position_world_m": [1.0, 1.0, plane]}}
    tracemalloc.start()
    appearance = compiler._materialize_native_particlefield_appearance(
        source_path=source, output_root=out / "native-appearance")
    appearance = compiler.apply_task_volume_clearance(appearance, task_spec=spec, output_root=out)
    peak = tracemalloc.get_traced_memory()[1]
    text = json.dumps(appearance, sort_keys=True).replace(str(out), "<out>")
    print(json.dumps({"peak": peak, "appearance": json.loads(text)}))
""")


def _python(code: str, *args: object) -> str:
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")]))
    completed = subprocess.run([sys.executable, "-c", code, *map(str, args)], capture_output=True, text=True,
                               env=environment, timeout=600, check=False)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return completed.stdout


def _field(path: Path, count: int) -> Path:
    _python(GENERATE, path, count, PLANE)
    return path


def _appearance(source: Path, out: Path, rows: int = 0) -> dict:
    out.mkdir()
    return json.loads(_python(RUN, source, out, PLANE, rows).splitlines()[-1])


def test_a_field_of_millions_of_particles_is_pressed_within_a_fixed_working_set(tmp_path) -> None:
    result = _appearance(_field(tmp_path / "field.usdc", PARTICLES), tmp_path / "out")
    clearance = result["appearance"]["appearance_task_clearance"]
    assert clearance["status"] == "pressed" and clearance["particle_count"] == PARTICLES
    assert result["appearance"]["gaussian_field_quality"]["metrics"]["count"] == PARTICLES
    assert result["peak"] < BOUND_BYTES, (result["peak"], BOUND_BYTES)


def test_the_block_size_changes_no_byte_of_the_field_or_the_records(tmp_path) -> None:
    source = _field(tmp_path / "field.usdc", 60_001)
    whole, blocks = _appearance(source, tmp_path / "whole"), _appearance(source, tmp_path / "blocks", rows=777)
    assert blocks["appearance"] == whole["appearance"]
    assert whole["appearance"]["appearance_task_clearance"]["status"] == "pressed"
    derived = [Path(str(run["appearance"]["path"]).replace("<out>", str(tmp_path / name)))
               for run, name in ((whole, "whole"), (blocks, "blocks"))]
    assert derived[0].read_bytes() == derived[1].read_bytes()
