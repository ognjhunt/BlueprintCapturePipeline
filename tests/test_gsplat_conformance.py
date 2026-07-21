from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import jsonschema
import numpy as np

from blueprint_pipeline.gaussian_splat_decode import SplatData, write_standard_3dgs_ply
from blueprint_pipeline.gsplat_conformance import (
    compare_particlefield_usd,
    run_gsplat_conformance_oracle,
)
from blueprint_pipeline.particlefield_usd import write_particlefield_usd


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _capture(tmp_path: Path) -> Path:
    root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write(root / "capture_descriptor.json", {"scene_id": "scene-1", "capture_id": "capture-1"})
    _write(root / "raw" / "manifest.json", {"scene_id": "scene-1", "capture_id": "capture-1"})
    return root


def _splat() -> SplatData:
    return SplatData(
        count=3,
        xyz=np.asarray([[0, 0, 0], [1, 2, 3], [-1, 0.5, 2]], dtype=np.float32),
        opacity=np.asarray([0.0, 1.0, -1.0], dtype=np.float32),
        f_dc=np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32),
        scales=np.asarray([[0, 0, 0], [0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]], dtype=np.float32),
        quats=np.asarray([[1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [-1, 0, 0, 0]], dtype=np.float32),
        properties=(),
    )


def test_particlefield_comparison_is_quaternion_sign_invariant_and_detects_drift(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.usdc"
    second = tmp_path / "second.usdc"
    assert write_particlefield_usd(_splat(), first)["status"] == "completed"
    assert write_particlefield_usd(_splat(), second)["status"] == "completed"
    comparison = compare_particlefield_usd(first, second)
    assert comparison["status"] == "conformant"

    from pxr import Usd

    stage = Usd.Stage.Open(str(second))
    prim = next(
        prim for prim in stage.Traverse() if prim.GetTypeName() == "ParticleField3DGaussianSplat"
    )
    scales = prim.GetAttribute("scales").Get()
    scales[0] = (99.0, 99.0, 99.0)
    prim.GetAttribute("scales").Set(scales)
    stage.GetRootLayer().Save()
    drift = compare_particlefield_usd(first, second)
    assert drift["status"] == "nonconformant"
    assert "particlefield_attribute_mismatch:scales" in drift["blockers"]


def test_external_converter_remains_advisory_oracle(tmp_path: Path) -> None:
    root = _capture(tmp_path / "capture")
    ply = root / "pipeline" / "geometry" / "scene.ply"
    ply.parent.mkdir(parents=True, exist_ok=True)
    write_standard_3dgs_ply(_splat(), ply)
    worker = tmp_path / "oracle-worker"
    worker.write_text(
        """#!/bin/sh
raw_report="$1"
oracle="$2"
blueprint="$3"
cp "$blueprint" "$oracle"
printf '{"status":"completed","converter_version":"0.1.0-fixture","source_revision":"621017ebf78394488260c70ec4eadd70ff621131"}\n' > "$raw_report"
""",
        encoding="utf-8",
    )
    worker.chmod(worker.stat().st_mode | stat.S_IXUSR)
    result = run_gsplat_conformance_oracle(
        capture_root=root,
        source_ply=ply,
        converter_command=[str(worker), "{output}", "{oracle_output}", "{blueprint_output}"],
        converter_version="0.1.0-fixture",
        source_revision="621017ebf78394488260c70ec4eadd70ff621131",
        license_compatible=True,
        env={"PATH": f"{worker.parent}:{os.defpath}"},
    )
    assert result["status"] == "conformant_advisory"
    assert result["claim_boundary"]["blueprint_particlefield_authoring_replaced"] is False
    assert result["claim_boundary"]["isaac_or_ovrtx_render_proven"] is False
    assert result["comparison"]["status"] == "conformant"
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "schemas"
            / "usd_convert_gsplat_conformance_result.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(result)
