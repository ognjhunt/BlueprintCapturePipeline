"""Fail-closed contracts for the optional 3dgrut/Isaac export adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from blueprint_pipeline.isaac_nurec_export import (
    THREEDGRUT_PINNED_REVISION,
    TRANSCODE_FORWARDED_ENVIRONMENT,
    build_nurec_transcode_command,
    transcode_nurec_usdz_to_particlefield,
    validate_transcoded_particlefield,
)

from blueprint_pipeline.isaac_nurec_export import (
    convert_ply_to_isaac_usd,
    threedgrut_available,
)

_HAS_PXR = importlib.util.find_spec("pxr") is not None


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_isaac_splat_nurec_render.py"


def test_threedgrut_unavailable_locally() -> None:
    assert threedgrut_available(sys.executable) is False


def test_convert_blocked_unsupported_format(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"ply\n")
    result = convert_ply_to_isaac_usd(
        ply,
        tmp_path / "o.usdz",
        fmt="bogus",
        python=sys.executable,
    )
    assert result["status"] == "blocked"
    assert "unsupported_isaac_splat_format" in result["blockers"]


def test_convert_blocked_missing_ply(tmp_path: Path) -> None:
    result = convert_ply_to_isaac_usd(
        tmp_path / "nope.ply",
        tmp_path / "o.usdz",
        python=sys.executable,
    )
    assert result["status"] == "blocked"
    assert "standard_ply_missing" in result["blockers"]


def test_convert_blocked_threedgrut_unavailable(tmp_path: Path) -> None:
    ply = tmp_path / "scene.ply"
    ply.write_bytes(b"ply\nformat binary_little_endian 1.0\nend_header\n")
    result = convert_ply_to_isaac_usd(
        ply,
        tmp_path / "o.usdz",
        python=sys.executable,
    )
    assert result["status"] == "blocked"
    assert "threedgrut_unavailable" in result["blockers"]
    assert "remediation" in result


def test_runner_py_compiles() -> None:
    process = subprocess.run(
        [sys.executable, "-m", "py_compile", str(RUNNER)],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def test_runner_help_exits_zero() -> None:
    process = subprocess.run(
        [sys.executable, str(RUNNER), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 0
    assert "nurec" in process.stdout.lower()


def test_runner_requires_ply_or_usdz(tmp_path: Path) -> None:
    cameras = tmp_path / "cameras.json"
    cameras.write_text("[]", encoding="utf-8")
    process = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--cameras",
            str(cameras),
            "--out-dir",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert process.returncode == 2
    assert "ply" in process.stderr.lower()


# ---------------------------------------------------------------------------
# Direct NuRec USDZ -> LightField transcode wrapper (Scene 839873 render audit)
# ---------------------------------------------------------------------------



def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_direct_transcode_command_is_pinned_and_carries_no_secrets(tmp_path: Path) -> None:
    command = build_nurec_transcode_command(tmp_path / "in.usdz", tmp_path / "out.usdz")
    assert command[1:] == [
        "-m",
        "threedgrut.export.scripts.transcode",
        str(tmp_path / "in.usdz"),
        "-o",
        str(tmp_path / "out.usdz"),
        "--format",
        "lightfield",
    ]
    assert "--apply-coordinate-transform" not in command
    assert THREEDGRUT_PINNED_REVISION == "a37ef721012dea0f29c0fcfff2d525023b4e854a"
    assert "NGC_API_KEY" not in TRANSCODE_FORWARDED_ENVIRONMENT
    with pytest.raises(ValueError):
        build_nurec_transcode_command(tmp_path / "in.usdz", tmp_path / "o.usdz", render_order_hint="nearest")


def test_direct_transcode_refuses_identity_mismatch_before_invoking_anything(tmp_path: Path) -> None:
    source = tmp_path / "configured_appearance.usdz"
    source.write_bytes(b"not-the-sealed-asset")
    invoked: list[list[str]] = []
    result = transcode_nurec_usdz_to_particlefield(
        source,
        tmp_path / "out.usdz",
        expected_source_sha256="sha256:" + "0" * 64,
        runner=lambda cmd, *, env, timeout: invoked.append(cmd),
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["nurec_transcode_source_identity_mismatch"]
    assert invoked == []
    assert result["nurec_state_reinterpreted_by_blueprint"] is False


def test_direct_transcode_refuses_an_unpinned_revision(tmp_path: Path) -> None:
    source = tmp_path / "configured_appearance.usdz"
    source.write_bytes(b"sealed")
    invoked: list[list[str]] = []
    result = transcode_nurec_usdz_to_particlefield(
        source,
        tmp_path / "out.usdz",
        expected_source_sha256=_digest(source),
        threedgrut_revision="main",
        runner=lambda cmd, *, env, timeout: invoked.append(cmd),
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["nurec_transcode_revision_unpinned"]
    assert invoked == []


def test_direct_transcode_forwards_only_the_environment_whitelist_and_scrubs_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NGC_API_KEY", "nvapi-secret-value")
    monkeypatch.setenv("HOME", str(tmp_path))
    source = tmp_path / "configured_appearance.usdz"
    source.write_bytes(b"sealed")
    seen: dict = {}

    def runner(cmd, *, env, timeout):
        seen["env"] = dict(env)
        return SimpleNamespace(returncode=1, stdout="", stderr="api_key=nvapi-secret-value failed")

    result = transcode_nurec_usdz_to_particlefield(
        source, tmp_path / "out.usdz", expected_source_sha256=_digest(source), runner=runner
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["threedgrut_transcode_failed"]
    assert "NGC_API_KEY" not in seen["env"]
    assert set(seen["env"]) <= set(TRANSCODE_FORWARDED_ENVIRONMENT)
    assert "nvapi-secret-value" not in result["stderr_tail"]
    assert "<redacted>" in result["stderr_tail"]
    assert result["environment_keys_forwarded"] == sorted(seen["env"])


def _write_particlefield(path: Path, *, count: int, degree: int, hints: bool, color_space: bool):
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdVol, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    field = UsdVol.ParticleField3DGaussianSplat.Define(stage, "/World/Gaussians/gaussians")
    rng = np.random.default_rng(1)
    field.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(rng.normal(size=(count, 3)).astype(np.float32)))
    field.CreateScalesAttr().Set(Vt.Vec3fArray.FromNumpy(np.full((count, 3), 0.01, np.float32)))
    field.CreateOrientationsAttr().Set(Vt.QuatfArray([Gf.Quatf(1.0, 0.0, 0.0, 0.0)] * count))
    field.CreateOpacitiesAttr().Set(Vt.FloatArray.FromNumpy(np.full(count, 0.5, np.float32)))
    elements = (degree + 1) ** 2
    sh = field.CreateRadianceSphericalHarmonicsCoefficientsAttr()
    sh.Set(Vt.Vec3fArray.FromNumpy(rng.normal(size=(count * elements, 3)).astype(np.float32)))
    sh.SetMetadata("elementSize", elements)
    field.CreateRadianceSphericalHarmonicsDegreeAttr().Set(degree)
    if hints:
        field.CreateProjectionModeHintAttr().Set("perspective")
        field.CreateSortingModeHintAttr().Set("cameraDistance")
    if color_space:
        Usd.ColorSpaceAPI.Apply(field.GetPrim()).CreateColorSpaceNameAttr().Set("srgb_rec709_display")
    stage.GetRootLayer().Save()
    return Sdf  # keep import alive for linters


@pytest.mark.skipif(not _HAS_PXR, reason="usd-core unavailable")
def test_direct_transcode_validates_the_lightfield_output_and_seals_digests(tmp_path: Path) -> None:
    source = tmp_path / "configured_appearance.usdz"
    source.write_bytes(b"sealed")
    out = tmp_path / "lightfield.usdc"

    def runner(cmd, *, env, timeout):
        _write_particlefield(Path(cmd[cmd.index("-o") + 1]), count=8, degree=3, hints=True, color_space=True)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    receipt_path = tmp_path / "receipt.json"
    result = transcode_nurec_usdz_to_particlefield(
        source,
        out,
        expected_source_sha256=_digest(source),
        expected_gaussian_count=8,
        container_image_digest="sha256:" + "c" * 64,
        runner=runner,
        receipt_path=receipt_path,
    )
    assert result["status"] == "completed", result
    validation = result["output_validation"]
    assert validation["passed"] is True
    assert validation["sorting_mode_hint"] == "cameraDistance"
    assert validation["color_space"] == "srgb_rec709_display"
    assert set(validation["attribute_digests"]) == {
        "positions", "scales", "orientations", "opacities", "sh_coefficients"
    }
    assert result["output_sha256"] == _digest(out)
    assert result["receipt_digest"].startswith("sha256:")
    assert receipt_path.is_file()


@pytest.mark.skipif(not _HAS_PXR, reason="usd-core unavailable")
@pytest.mark.parametrize(
    ("count", "degree", "hints", "color_space", "expected"),
    [
        (7, 3, True, True, {"nurec_transcode_particlefield_count_unexpected"}),
        (8, 2, True, True, {"nurec_transcode_sh_degree_unexpected", "nurec_transcode_sh_coefficient_count_unexpected"}),
        (8, 3, False, True, {"nurec_transcode_projection_hint_unexpected", "nurec_transcode_sorting_hint_unexpected"}),
        (8, 3, True, False, {"nurec_transcode_color_space_unexpected"}),
    ],
)
def test_direct_transcode_output_contract_drift_fails_closed(
    tmp_path: Path, count: int, degree: int, hints: bool, color_space: bool, expected: set[str]
) -> None:
    source = tmp_path / "configured_appearance.usdz"
    source.write_bytes(b"sealed")

    def runner(cmd, *, env, timeout):
        _write_particlefield(
            Path(cmd[cmd.index("-o") + 1]), count=count, degree=degree, hints=hints, color_space=color_space
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = transcode_nurec_usdz_to_particlefield(
        source,
        tmp_path / "lightfield.usdc",
        expected_source_sha256=_digest(source),
        expected_gaussian_count=8,
        expected_sh_degree=3,
        runner=runner,
    )
    assert result["status"] == "blocked"
    assert set(result["blockers"]) == expected


def test_validation_without_usd_runtime_is_a_typed_refusal(tmp_path: Path, monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pxr":
            raise ImportError("pxr unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = validate_transcoded_particlefield(tmp_path / "missing.usdc")
    assert result == {"passed": False, "blockers": ["nurec_transcode_validation_runtime_unavailable"]}


def test_os_environment_is_not_captured_at_import_time() -> None:
    assert "PATH" in TRANSCODE_FORWARDED_ENVIRONMENT
    assert all(isinstance(key, str) for key in TRANSCODE_FORWARDED_ENVIRONMENT)
    assert os.environ is not TRANSCODE_FORWARDED_ENVIRONMENT
