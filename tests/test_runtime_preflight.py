"""Tests for runtime preflight validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline.runtime_preflight import (
    PreflightCheck,
    enforce_preflight,
    format_failed_preflight_checks,
    validate_runtime_preflight,
)


def _make_blueprintpipeline_stub(root: Path, *, include_downstream: bool = False) -> None:
    (root / "interactive-job").mkdir(parents=True, exist_ok=True)
    (root / "simready-job").mkdir(parents=True, exist_ok=True)
    (root / "usd-assembly-job").mkdir(parents=True, exist_ok=True)
    (root / "tools/source_pipeline").mkdir(parents=True, exist_ok=True)
    (root / "tools/scene_manifest").mkdir(parents=True, exist_ok=True)

    (root / "interactive-job/run_interactive_assets.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "simready-job/prepare_simready_assets.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "usd-assembly-job/assemble_scene.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "tools/source_pipeline/adapter.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    (root / "tools/scene_manifest/loader.py").write_text("VALUE = 'ok'\n", encoding="utf-8")
    (root / "tools/__init__.py").write_text("", encoding="utf-8")
    (root / "tools/source_pipeline/__init__.py").write_text("", encoding="utf-8")
    (root / "tools/scene_manifest/__init__.py").write_text("", encoding="utf-8")

    if include_downstream:
        (root / "replicator-job").mkdir(parents=True, exist_ok=True)
        (root / "variation-asset-pipeline-job").mkdir(parents=True, exist_ok=True)
        (root / "genie-sim-export-job").mkdir(parents=True, exist_ok=True)
        (root / "replicator-job/generate_replicator_bundle.py").write_text(
            "print('ok')\n",
            encoding="utf-8",
        )
        (root / "variation-asset-pipeline-job/run_variation_asset_pipeline.py").write_text(
            "print('ok')\n",
            encoding="utf-8",
        )
        (root / "genie-sim-export-job/export_to_geniesim.py").write_text(
            "print('ok')\n",
            encoding="utf-8",
        )


def _set_default_neoverse_local_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEOVERSE_EXECUTABLE", "/opt/neoverse/bin/neoverse_local")
    monkeypatch.delenv("NEOVERSE_CMD_TEMPLATE", raising=False)


def test_runtime_preflight_passes_with_configured_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bp_root = tmp_path / "BlueprintPipeline"
    _make_blueprintpipeline_stub(bp_root)
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("TEXT_SAM3D_API_HOST", "https://sam3d.example.internal")
    monkeypatch.setenv("TEXT_SAM3D_API_KEY", "sam3d-key")
    _set_default_neoverse_local_env(monkeypatch)
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    monkeypatch.setenv("RUNTIME_PREFLIGHT_ENABLED", "true")

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=bp_root,
        generation_provider_chain="sam3d",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
    )
    enforce_preflight(checks)
    assert all(item.passed for item in checks)


def test_runtime_preflight_fails_when_provider_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bp_root = tmp_path / "BlueprintPipeline"
    _make_blueprintpipeline_stub(bp_root)
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.delenv("TEXT_SAM3D_API_HOST", raising=False)
    monkeypatch.delenv("TEXT_SAM3D_API_KEY", raising=False)
    _set_default_neoverse_local_env(monkeypatch)
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    monkeypatch.setenv("RUNTIME_PREFLIGHT_ENABLED", "true")

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=bp_root,
        generation_provider_chain="sam3d",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
    )
    with pytest.raises(Exception):
        enforce_preflight(checks)

    failed_names = {item.name for item in checks if not item.passed}
    assert "provider_sam3d" in failed_names


def test_runtime_preflight_accepts_ttt_lrm_with_command_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bp_root = tmp_path / "BlueprintPipeline"
    _make_blueprintpipeline_stub(bp_root)
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("STAGE_D_TTTLRM_IMAGE_TO_3D_COMMAND", "ttt_lrm_cli --input {REFERENCE_IMAGE} --out {OUTPUT_GLB}")
    _set_default_neoverse_local_env(monkeypatch)
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    monkeypatch.setenv("RUNTIME_PREFLIGHT_ENABLED", "true")

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=bp_root,
        generation_provider_chain="ttt_lrm",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
    )
    enforce_preflight(checks)
    failed_names = {item.name for item in checks if not item.passed}
    assert "provider_ttt_lrm" not in failed_names


def test_runtime_preflight_fails_when_ttt_lrm_unconfigured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bp_root = tmp_path / "BlueprintPipeline"
    _make_blueprintpipeline_stub(bp_root)
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.delenv("STAGE_D_TTTLRM_IMAGE_TO_3D_COMMAND", raising=False)
    monkeypatch.delenv("STAGE_D_TTT_LRM_IMAGE_TO_3D_COMMAND", raising=False)
    monkeypatch.delenv("TEXT_TTTLRM_API_HOST", raising=False)
    monkeypatch.delenv("TEXT_TTTLRM_API_KEY", raising=False)
    monkeypatch.delenv("TEXT_TTT_LRM_API_HOST", raising=False)
    monkeypatch.delenv("TEXT_TTT_LRM_API_KEY", raising=False)
    _set_default_neoverse_local_env(monkeypatch)
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    monkeypatch.setenv("RUNTIME_PREFLIGHT_ENABLED", "true")

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=bp_root,
        generation_provider_chain="ttt_lrm",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
    )
    failed_names = {item.name for item in checks if not item.passed}
    assert "provider_ttt_lrm" in failed_names


def test_runtime_preflight_requires_neoverse_local_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bp_root = tmp_path / "BlueprintPipeline"
    _make_blueprintpipeline_stub(bp_root)
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("RECONSTRUCTION_BACKEND", "neoverse")
    monkeypatch.delenv("NEOVERSE_CMD_TEMPLATE", raising=False)
    monkeypatch.delenv("NEOVERSE_EXECUTABLE", raising=False)
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    monkeypatch.setenv("RUNTIME_PREFLIGHT_ENABLED", "true")

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=bp_root,
        generation_provider_chain="sam3d",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
    )
    failed_names = {item.name for item in checks if not item.passed}
    assert "reconstruction_neoverse_local" in failed_names


def test_runtime_preflight_requires_gen3c_service_and_conditioning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bp_root = tmp_path / "BlueprintPipeline"
    _make_blueprintpipeline_stub(bp_root)
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("RECONSTRUCTION_BACKEND", "gen3c")
    monkeypatch.setenv("GEN3C_SERVICE_URL", "https://gen3c.example.internal")
    monkeypatch.setenv("GEN3C_SERVICE_API_KEY", "secret")
    monkeypatch.delenv("RECONSTRUCTION_ARKIT_POSES_PATH", raising=False)
    monkeypatch.delenv("RECONSTRUCTION_ARKIT_INTRINSICS_PATH", raising=False)
    monkeypatch.delenv("RECONSTRUCTION_ARKIT_DEPTH_DIR", raising=False)
    monkeypatch.delenv("RECONSTRUCTION_ADVANCED_GEOMETRY_BUNDLE_PATH", raising=False)
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    monkeypatch.setenv("RUNTIME_PREFLIGHT_ENABLED", "true")

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=bp_root,
        generation_provider_chain="sam3d",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
    )
    failed_names = {item.name for item in checks if not item.passed}
    assert "reconstruction_gen3c_conditioning" in failed_names


def test_runtime_preflight_accepts_gen3c_with_service_and_conditioning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bp_root = tmp_path / "BlueprintPipeline"
    _make_blueprintpipeline_stub(bp_root)
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)
    poses = tmp_path / "poses.jsonl"
    intrinsics = tmp_path / "intrinsics.json"
    depth_dir = tmp_path / "depth"
    poses.write_text("{}\n", encoding="utf-8")
    intrinsics.write_text("{}", encoding="utf-8")
    depth_dir.mkdir()

    monkeypatch.setenv("RECONSTRUCTION_BACKEND", "gen3c")
    monkeypatch.setenv("GEN3C_SERVICE_URL", "https://gen3c.example.internal")
    monkeypatch.setenv("GEN3C_SERVICE_API_KEY", "secret")
    monkeypatch.setenv("RECONSTRUCTION_ARKIT_POSES_PATH", str(poses))
    monkeypatch.setenv("RECONSTRUCTION_ARKIT_INTRINSICS_PATH", str(intrinsics))
    monkeypatch.setenv("RECONSTRUCTION_ARKIT_DEPTH_DIR", str(depth_dir))
    monkeypatch.setenv("TEXT_SAM3D_API_HOST", "https://sam3d.example.internal")
    monkeypatch.setenv("TEXT_SAM3D_API_KEY", "sam3d-key")
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    monkeypatch.setenv("RUNTIME_PREFLIGHT_ENABLED", "true")

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=bp_root,
        generation_provider_chain="sam3d",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
    )
    failed_names = {item.name for item in checks if not item.passed}
    assert "reconstruction_gen3c_service" not in failed_names
    assert "reconstruction_gen3c_conditioning" not in failed_names


def test_runtime_preflight_standalone_mode_skips_blueprintpipeline_requirements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)
    missing_bp_root = tmp_path / "does_not_exist"

    _set_default_neoverse_local_env(monkeypatch)
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    monkeypatch.setenv("RUNTIME_PREFLIGHT_ENABLED", "true")

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=missing_bp_root,
        generation_provider_chain="image_to_3d,proxy_box",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
        standalone_mode=True,
    )
    enforce_preflight(checks)
    check_names = {item.name for item in checks}
    assert "blueprintpipeline_runtime_mode" in check_names


def test_runtime_preflight_flags_unsafe_heuristic_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bp_root = tmp_path / "BlueprintPipeline"
    _make_blueprintpipeline_stub(bp_root)
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setenv("TEXT_SAM3D_API_HOST", "https://sam3d.example.internal")
    monkeypatch.setenv("TEXT_SAM3D_API_KEY", "sam3d-key")
    _set_default_neoverse_local_env(monkeypatch)
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    monkeypatch.setenv("SWAP_INCLUDE_HEURISTIC_AS_EXPLICIT", "true")

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=bp_root,
        generation_provider_chain="sam3d",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
    )
    failed = {item.name for item in checks if not item.passed}
    assert "swap_heuristic_explicit_override" in failed


def test_runtime_preflight_rejects_standalone_in_full_required_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)
    missing_bp_root = tmp_path / "does_not_exist"

    _set_default_neoverse_local_env(monkeypatch)
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    monkeypatch.setenv("RUNTIME_PREFLIGHT_ENABLED", "true")

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=missing_bp_root,
        generation_provider_chain="image_to_3d,proxy_box",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=True,
        standalone_mode=True,
        completion_mode="full_required",
    )
    failed = {item.name for item in checks if not item.passed}
    assert "full_completion_standalone_guard" in failed


def test_runtime_preflight_full_required_requires_downstream_scripts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bp_root = tmp_path / "BlueprintPipeline"
    _make_blueprintpipeline_stub(bp_root, include_downstream=False)
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)

    _set_default_neoverse_local_env(monkeypatch)
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    monkeypatch.setenv("RUNTIME_PREFLIGHT_ENABLED", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key")

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=bp_root,
        generation_provider_chain="image_to_3d,proxy_box",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
        completion_mode="full_required",
    )
    failed = {item.name for item in checks if not item.passed}
    assert "script_generate_replicator_bundle.py" in failed
    assert "script_run_variation_asset_pipeline.py" in failed
    assert "script_export_to_geniesim.py" in failed


def test_runtime_preflight_full_required_requires_gemini_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bp_root = tmp_path / "BlueprintPipeline"
    _make_blueprintpipeline_stub(bp_root, include_downstream=True)
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)

    _set_default_neoverse_local_env(monkeypatch)
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    monkeypatch.setenv("RUNTIME_PREFLIGHT_ENABLED", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=bp_root,
        generation_provider_chain="image_to_3d,proxy_box",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
        completion_mode="full_required",
    )
    failed = {item.name for item in checks if not item.passed}
    assert "data_gen_gemini_key" in failed


def test_runtime_preflight_best_effort_does_not_require_downstream_scripts_or_gemini(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bp_root = tmp_path / "BlueprintPipeline"
    _make_blueprintpipeline_stub(bp_root, include_downstream=False)
    gcs_root = tmp_path / "gcs"
    gcs_root.mkdir(parents=True, exist_ok=True)

    _set_default_neoverse_local_env(monkeypatch)
    monkeypatch.setenv("PARTICULATE_MODE", "skip")
    monkeypatch.setenv("ARTICULATION_BACKEND", "auto")
    monkeypatch.setenv("NUREC_SKIP_PIPELINE_COMMAND", "true")
    monkeypatch.setenv("RUNTIME_PREFLIGHT_ENABLED", "true")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_API_KEY", raising=False)

    checks = validate_runtime_preflight(
        gcs_root=gcs_root,
        blueprintpipeline_root=bp_root,
        generation_provider_chain="image_to_3d,proxy_box",
        swap_policy_path="",
        nurec_worker_mode="local_worker",
        nurec_worker_command="",
        advanced_quality_gates_enabled=False,
        completion_mode="best_effort",
    )
    failed = {item.name for item in checks if not item.passed}
    assert "script_generate_replicator_bundle.py" not in failed
    assert "script_run_variation_asset_pipeline.py" not in failed
    assert "script_export_to_geniesim.py" not in failed
    assert "data_gen_gemini_key" not in failed


def test_format_failed_preflight_checks_is_multiline_and_actionable() -> None:
    checks = [
        PreflightCheck(name="provider_sam3d", passed=False, detail="missing host"),
        PreflightCheck(name="data_gen_gemini_key", passed=False, detail="missing GOOGLE_GENAI_API_KEY"),
    ]
    text = format_failed_preflight_checks(checks)
    assert text.startswith("2 check(s) failed:")
    assert "- provider_sam3d: missing host" in text
    assert "- data_gen_gemini_key: missing GOOGLE_GENAI_API_KEY" in text


def test_enforce_preflight_uses_structured_failure_text() -> None:
    checks = [
        PreflightCheck(name="gcs_root", passed=True, detail="ok"),
        PreflightCheck(name="provider_sam3d", passed=False, detail="missing host+key"),
    ]
    with pytest.raises(Exception) as exc:
        enforce_preflight(checks)
    message = str(exc.value)
    assert "runtime_preflight:" in message
    assert "1 check(s) failed:" in message
    assert "- provider_sam3d: missing host+key" in message
