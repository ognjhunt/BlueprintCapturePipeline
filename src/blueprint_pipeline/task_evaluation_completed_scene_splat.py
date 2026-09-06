"""Normalize a completed standard splat's declared frame with the pinned converter."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess  # nosec B404 - fixed, sealed local format converter; no provider call
import tempfile

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .gaussian_splat_decode import find_splat_transform_cli, read_standard_3dgs_ply
from .task_evaluation_scene_configuration_submission_inputs import checked_file, read, require, sha
from .task_evaluation_splat_render_runtime import runtime_from_environment


def normalize_completed_splat(*, source: Path, coordinate_frame: dict, output_root: Path,
                              runtime_resolver=runtime_from_environment):
    scale, axis = coordinate_frame["meters_per_unit"], coordinate_frame["up_axis"]
    if scale == 1.0 and axis == "Z":
        return None
    require(type(scale) in (int, float) and math.isfinite(scale) and scale > 0 and axis in {"Y", "Z"},
            "completed_splat_frame_invalid")
    source_digest = sha(source)
    receipt_path = output_root / "splat_normalization.v1.json"
    if receipt_path.exists():
        prior = read(receipt_path, digest_field="normalization_digest")
        require(prior.get("source_digest") == source_digest and prior.get("declared_coordinate_frame") == coordinate_frame,
                "completed_splat_normalization_conflict")
        checked_file(output_root / prior["output"]["relative_path"], prior["output"])
        return prior
    runtime = runtime_resolver(repo_root=Path(__file__).resolve().parents[2])
    cli = find_splat_transform_cli(runtime["renderer_root"])
    require(cli is not None, "completed_splat_converter_not_installed")
    package = json.loads((cli.parent.parent / "package.json").read_text())
    require(package.get("name") == "@playcanvas/splat-transform" and package.get("version") == "3.2.0",
            "completed_splat_converter_version_unadmitted")
    original = read_standard_3dgs_ply(source)
    output_root.mkdir(parents=True, exist_ok=True, mode=0o750)
    with tempfile.TemporaryDirectory(prefix=".normalize-", dir=output_root) as temporary:
        temporary = Path(temporary)
        named = temporary / "source.ply"
        try:
            os.link(source, named, follow_symlinks=False)
        except OSError:
            shutil.copyfile(source, named)
        output = temporary / "normalized.ply"
        arguments = [str(runtime["node"]), str(cli), "-w", "-q", str(named), "--scale", str(scale)]
        if axis == "Y":
            # PlayCanvas applies actions in its corrected PLY engine frame.
            # -90 there is +90 about X in the retained standard-PLY frame.
            # Verify the output XYZ and covariance rotations independently below.
            arguments += ["--rotate", "-90,0,0"]
        arguments.append(str(output))
        result = subprocess.run(arguments, capture_output=True, timeout=600, check=False)  # nosec B603
        require(result.returncode == 0 and output.is_file(), "completed_splat_conversion_failed")
        converted = read_standard_3dgs_ply(output)
        expected_xyz = original.xyz.astype(np.float64) * scale
        if axis == "Y":
            expected_xyz = expected_xyz[:, [0, 2, 1]] * [1, -1, 1]
        require(converted.count == original.count
                and np.allclose(converted.xyz, expected_xyz, rtol=1e-5, atol=1e-6)
                and np.allclose(converted.opacity, original.opacity, rtol=1e-5, atol=1e-6)
                and np.allclose(converted.f_dc, original.f_dc, rtol=1e-5, atol=1e-6)
                and np.allclose(converted.scales, original.scales + math.log(scale), rtol=1e-5, atol=1e-6)
                and (converted.sh_rest is None) == (original.sh_rest is None),
                "completed_splat_conversion_readback_mismatch")
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_euler("x", 90 if axis == "Y" else 0, degrees=True).as_matrix()
        before = Rotation.from_quat(original.quats[:, [1, 2, 3, 0]]).as_matrix()
        after = Rotation.from_quat(converted.quats[:, [1, 2, 3, 0]]).as_matrix()
        require(np.allclose(after, rotation @ before, rtol=1e-5, atol=1e-5),
                "completed_splat_orientation_readback_mismatch")
        require(sha(source) == source_digest, "completed_splat_source_changed")
        destination = output_root / "normalized.ply"
        output_digest = sha(output)
        if destination.exists():
            require(sha(destination) == output_digest, "completed_splat_partial_output_conflict")
        else:
            output.chmod(0o440)
            os.link(output, destination, follow_symlinks=False)
    value = {"schema_version": "task_evaluation_completed_splat_normalization.v1",
        "source_digest": source_digest, "declared_coordinate_frame": coordinate_frame,
        "runtime_frame": {"meters_per_unit": 1.0, "up_axis": "Z"},
        "output": {"relative_path": destination.name, "sha256": output_digest, "size_bytes": destination.stat().st_size},
        "retained_gaussian_count": original.count, "source_bytes_unchanged": True,
        "conversion_method": {"id": "playcanvas_splat_transform", "version": package["version"],
            "runtime_digest": runtime["identity"]["runtime_digest"], "entrypoint_digest": sha(cli)},
        "reconstruction_performed": False, "physical_scale_measured": False, "renderer_qualified": False}
    value["normalization_digest"] = canonical_digest(value, digest_field="normalization_digest")
    from .task_evaluation_scene_intake import write_exclusive
    write_exclusive(receipt_path, value)
    return value
