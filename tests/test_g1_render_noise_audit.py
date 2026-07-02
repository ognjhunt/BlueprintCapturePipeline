"""Hermetic tests for the G1 textured-robot render-noise audit module (no GPU, no Isaac)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from blueprint_pipeline import g1_render_noise_audit as A


# ---------------------------------------------------------------- variant plan

def test_default_variant_matrix_matches_spec_rows() -> None:
    variants = {v.variant_id: v for v in A.build_default_variant_matrix()}
    assert sorted(variants) == ["A", "B", "C", "D", "E", "F", "G"]
    assert variants["A"].robot_material == A.VARIANT_MATERIAL_WHITE_PROXY
    assert variants["A"].denoiser_enabled and variants["A"].render_budget == A.RENDER_BUDGET_CURRENT_DEFAULT
    assert variants["B"].robot_material == A.VARIANT_MATERIAL_TEXTURED_ORIGINAL
    assert not variants["B"].denoiser_enabled
    assert variants["C"].denoiser_enabled
    assert variants["D"].render_budget == A.RENDER_BUDGET_HIGH and not variants["D"].denoiser_enabled
    assert variants["E"].render_budget == A.RENDER_BUDGET_HIGH and variants["E"].denoiser_enabled
    assert variants["F"].robot_material == A.VARIANT_MATERIAL_SIMPLIFIED_DIFFUSE
    assert variants["G"].lighting_boost and variants["G"].robot_material == A.VARIANT_MATERIAL_TEXTURED_ORIGINAL
    assert not any(v.exploratory for v in variants.values())


def test_variant_plan_declares_only_single_variable_comparisons() -> None:
    plan = A.build_variant_plan()
    assert plan["blockers"] == []
    assert plan["schema_version"] == A.VARIANT_PLAN_SCHEMA_VERSION
    pairs = {tuple(p["pair"]) for p in plan["single_variable_comparison_pairs"]}
    assert ("B", "C") in pairs and ("D", "E") in pairs and ("C", "F") in pairs and ("C", "G") in pairs


def test_variant_execution_order_is_material_monotonic() -> None:
    order = A.default_execution_order(A.build_default_variant_matrix())
    assert order == ["B", "C", "D", "E", "G", "F", "A"]  # textured first, white proxy last


def test_validate_variant_plan_flags_multi_variable_pairs_and_duplicates() -> None:
    plan = A.build_variant_plan()
    plan["single_variable_comparison_pairs"].append({"pair": ["A", "D"], "isolates": "bogus"})
    blockers = A.validate_variant_plan(plan)
    assert any(b.startswith("variant_plan_comparison_pair_not_single_variable:A:D") for b in blockers)

    dup = A.build_variant_plan()
    dup["variants"].append(dict(dup["variants"][0]))
    assert "variant_plan_duplicate_variant_ids" in A.validate_variant_plan(dup)


# ---------------------------------------------------------------- frame stats

def _rng() -> np.random.Generator:
    return np.random.default_rng(20260701)


def test_frame_stats_flat_frame_is_clean() -> None:
    arr = np.full((96, 128, 3), 128, dtype=np.uint8)
    stats = A.compute_frame_stats(arr)
    assert stats["mean_luma"] == 128.0
    assert stats["edge_density"] == 0.0
    assert stats["high_frequency_noise_estimate"] < 0.5
    assert stats["black_edge_wedge_ratio"] == 0.0
    assert A.noise_grade(stats) == "clean"


def test_frame_stats_speckle_noise_raises_noise_estimate() -> None:
    rng = _rng()
    base = np.full((96, 128), 120.0)
    noisy = np.clip(base + rng.normal(0.0, 40.0, base.shape), 0, 255).astype(np.uint8)
    noisy_stats = A.compute_frame_stats(np.stack([noisy] * 3, axis=2))
    clean_stats = A.compute_frame_stats(np.full((96, 128, 3), 120, dtype=np.uint8))
    assert noisy_stats["high_frequency_noise_estimate"] > 10.0
    assert noisy_stats["high_frequency_noise_estimate"] > clean_stats["high_frequency_noise_estimate"]
    assert A.noise_grade(noisy_stats) == "noisy"


def test_frame_stats_detects_black_border_wedge() -> None:
    arr = np.full((120, 160, 3), 150, dtype=np.uint8)
    arr[:, :56] = 4  # large dark wedge touching the left frame edge (35% of frame)
    stats = A.compute_frame_stats(arr)
    assert stats["black_edge_wedge_ratio"] > 0.30
    # interior dark blob NOT touching the border is not an edge wedge
    interior = np.full((120, 160, 3), 150, dtype=np.uint8)
    interior[40:80, 60:100] = 4
    assert A.compute_frame_stats(interior)["black_edge_wedge_ratio"] == 0.0


def test_frame_stats_reads_png_from_disk(tmp_path: Path) -> None:
    png = tmp_path / "frame.png"
    Image.fromarray(np.full((32, 48, 3), 200, dtype=np.uint8)).save(png)
    stats = A.compute_frame_stats(png)
    assert stats["width"] == 48 and stats["height"] == 32
    assert abs(stats["mean_luma"] - 200.0) < 1.0


# ---------------------------------------------------------------- material resolution

def _raw_materials(missing: int = 0, present: int = 2) -> dict:
    refs = [
        {"input": f"tex_{i}", "authored_path": f"./textures/t{i}.png",
         "resolved_path": f"/assets/t{i}.png", "exists": True}
        for i in range(present)
    ]
    refs += [
        {"input": f"missing_{i}", "authored_path": f"./textures/m{i}.png",
         "resolved_path": None, "exists": False}
        for i in range(missing)
    ]
    return {
        "robot_prim_path": "/World/G1",
        "robot_asset_uri": "Isaac/Robots/Unitree/G1/g1.usd",
        "resolved_visual_asset": "omniverse://assets/G1/g1.usd",
        "gprim_count": 40,
        "mesh_count": 38,
        "gprims_without_material": 0,
        "materials": [{"path": "/World/G1/Looks/Body", "shader_ids": ["UsdPreviewSurface"],
                       "texture_refs": refs}],
    }


def test_summarize_material_resolution_counts_and_blockers() -> None:
    ok = A.summarize_material_resolution(_raw_materials())
    assert ok["status"] == "completed"
    assert ok["texture_reference_count"] == 2
    assert ok["texture_reference_missing_count"] == 0
    assert ok["textured_material_evidence_present"] is True

    broken = A.summarize_material_resolution(_raw_materials(missing=3))
    assert broken["status"] == "blocked"
    assert broken["texture_reference_missing_count"] == 3
    assert "robot_texture_references_missing" in broken["blockers"]
    assert broken["textured_material_evidence_present"] is False


def test_classify_robot_material_mode_never_upgrades_unverified_textures() -> None:
    verified = A.classify_robot_material_mode(
        requested_material=A.VARIANT_MATERIAL_TEXTURED_ORIGINAL,
        material_resolution=A.summarize_material_resolution(_raw_materials()),
    )
    assert verified["robot_material_mode"] == A.ROBOT_MATERIAL_MODE_VERIFIED_TEXTURED

    unverified = A.classify_robot_material_mode(
        requested_material=A.VARIANT_MATERIAL_TEXTURED_ORIGINAL,
        material_resolution=A.summarize_material_resolution(_raw_materials(missing=1)),
    )
    assert unverified["robot_material_mode"] == A.ROBOT_MATERIAL_MODE_TEXTURED_UNVERIFIED
    assert A.TEXTURED_MATERIAL_UNVERIFIED_BLOCKER in unverified["blockers"]

    no_refs = A.classify_robot_material_mode(
        requested_material=A.VARIANT_MATERIAL_TEXTURED_ORIGINAL,
        material_resolution=A.summarize_material_resolution(_raw_materials(present=0)),
    )
    assert no_refs["robot_material_mode"] == A.ROBOT_MATERIAL_MODE_TEXTURED_UNVERIFIED

    assert A.classify_robot_material_mode(
        requested_material=A.VARIANT_MATERIAL_WHITE_PROXY, material_resolution=None,
    )["robot_material_mode"] == A.ROBOT_MATERIAL_MODE_WHITE_PROXY


def test_normalize_legacy_robot_material_mode() -> None:
    assert A.normalize_legacy_robot_material_mode("neutral_matte_untextured_g1") == A.ROBOT_MATERIAL_MODE_WHITE_PROXY
    assert (
        A.normalize_legacy_robot_material_mode("preserve_authored_g1_materials_when_available")
        == A.ROBOT_MATERIAL_MODE_TEXTURED_UNVERIFIED
    )
    assert A.normalize_legacy_robot_material_mode("white_proxy") == "white_proxy"
    assert A.normalize_legacy_robot_material_mode("") is None
    assert A.normalize_legacy_robot_material_mode("mystery") is None


# ---------------------------------------------------------------- gates

def _stats(mean_luma=120.0, edge_density=0.05, wedge=0.0, noise=1.0,
           center_mean=110.0, center_dark=0.05) -> dict:
    return {
        "mean_luma": mean_luma,
        "edge_density": edge_density,
        "black_edge_wedge_ratio": wedge,
        "high_frequency_noise_estimate": noise,
        "center_crop": {"mean_luma": center_mean, "dark_pixel_ratio": center_dark},
    }


_VIS_OK = {
    "left_arm_visible": True, "right_arm_visible": True,
    "both_end_effectors_visible": True, "target_in_frame": True,
}


def _variant(vid: str) -> A.RenderNoiseAuditVariant:
    return {v.variant_id: v for v in A.build_default_variant_matrix()}[vid]


def test_gates_denoiser_regression_darker_than_raw_fails() -> None:
    raw = _stats(mean_luma=120.0, edge_density=0.08)
    denoised = _stats(mean_luma=95.0, edge_density=0.03)  # darker + lower structure
    gate = A.evaluate_variant_gates(
        variant=_variant("C"), stats=denoised, proxy_stats=_stats(edge_density=0.06),
        raw_textured_stats=raw, visibility=_VIS_OK,
        material_resolution=A.summarize_material_resolution(_raw_materials()),
    )
    assert gate["gates"]["denoised_not_darker_or_lower_structure_than_raw"] is False
    assert gate["denoiser_regression"]["darker_than_raw_beyond_tolerance"] is True
    assert gate["passed"] is False


def test_gates_pass_for_clean_denoised_frame() -> None:
    gate = A.evaluate_variant_gates(
        variant=_variant("C"), stats=_stats(), proxy_stats=_stats(edge_density=0.06),
        raw_textured_stats=_stats(mean_luma=118.0, edge_density=0.06), visibility=_VIS_OK,
        material_resolution=A.summarize_material_resolution(_raw_materials()),
    )
    assert gate["passed"] is True
    assert gate["robot_material_mode"] == A.ROBOT_MATERIAL_MODE_VERIFIED_TEXTURED


def test_gates_missing_textures_taint_material_mode_without_failing_frame_gates() -> None:
    gate = A.evaluate_variant_gates(
        variant=_variant("C"), stats=_stats(), proxy_stats=None,
        raw_textured_stats=None, visibility=_VIS_OK,
        material_resolution=A.summarize_material_resolution(_raw_materials(missing=2)),
    )
    assert gate["robot_material_mode"] == A.ROBOT_MATERIAL_MODE_TEXTURED_UNVERIFIED
    assert A.TEXTURED_MATERIAL_UNVERIFIED_BLOCKER in gate["blockers"]
    assert gate["passed"] is True  # frame gates still pass; the LABEL is what is bounded


def test_gates_black_wedge_and_edge_structure_vs_proxy() -> None:
    gate = A.evaluate_variant_gates(
        variant=_variant("B"), stats=_stats(wedge=0.4, edge_density=0.004),
        proxy_stats=_stats(edge_density=0.08), raw_textured_stats=None, visibility=_VIS_OK,
        material_resolution=A.summarize_material_resolution(_raw_materials()),
    )
    assert gate["gates"]["no_large_black_edge_wedge"] is False
    assert gate["gates"]["edge_structure_preserved_vs_proxy"] is False
    assert gate["passed"] is False


# ---------------------------------------------------------------- interpretation

def _gate_stub(passed=True, wedge_ok=True, arms_ok=True, luma_ok=True, regression=None) -> dict:
    return {
        "passed": passed,
        "gates": {
            "no_large_black_edge_wedge": wedge_ok,
            "both_arms_visible": arms_ok,
            "luma_not_collapsed_in_task_region": luma_ok,
        },
        "denoiser_regression": regression,
    }


_CLEAN = _stats(noise=1.0)
_NOISY = _stats(noise=20.0)


def test_interpret_sample_starvation_when_high_budget_is_clean() -> None:
    verdict = A.interpret_audit(
        stats_by_id={"A": _CLEAN, "B": _NOISY, "C": _NOISY, "D": _CLEAN, "E": _CLEAN,
                     "F": _CLEAN, "G": _NOISY},
        gates_by_id={vid: _gate_stub() for vid in "ABCDEFG"},
        material_resolution=A.summarize_material_resolution(_raw_materials()),
    )
    assert verdict["primary_diagnosis"] == "render_budget_sample_starvation"


def test_interpret_denoiser_failure_when_high_sample_denoised_regresses() -> None:
    gates = {vid: _gate_stub() for vid in "ABCDEFG"}
    gates["E"] = _gate_stub(regression={
        "darker_than_raw_beyond_tolerance": True,
        "lower_structure_than_raw_beyond_tolerance": False,
    })
    verdict = A.interpret_audit(
        stats_by_id={"A": _CLEAN, "B": _NOISY, "C": _NOISY, "D": _CLEAN, "E": _NOISY,
                     "F": _CLEAN, "G": _NOISY},
        gates_by_id=gates,
        material_resolution=A.summarize_material_resolution(_raw_materials()),
    )
    assert verdict["primary_diagnosis"] == "denoiser_path_failure"


def test_interpret_pbr_response_when_only_simplified_diffuse_is_clean() -> None:
    verdict = A.interpret_audit(
        stats_by_id={"A": _CLEAN, "B": _NOISY, "C": _NOISY, "D": _NOISY, "E": _NOISY,
                     "F": _CLEAN, "G": _NOISY},
        gates_by_id={vid: _gate_stub() for vid in "ABCDEFG"},
        material_resolution=A.summarize_material_resolution(_raw_materials()),
    )
    assert verdict["primary_diagnosis"] == "pbr_specular_material_response"
    rules = {f["rule"] for f in verdict["findings"]}
    assert "white_proxy_bounded_workaround_available" in rules


def test_interpret_lighting_underexposure() -> None:
    verdict = A.interpret_audit(
        stats_by_id={"A": _CLEAN, "B": _NOISY, "C": _NOISY, "D": _NOISY, "E": _NOISY,
                     "F": _NOISY, "G": _CLEAN},
        gates_by_id={vid: _gate_stub() for vid in "ABCDEFG"},
        material_resolution=A.summarize_material_resolution(_raw_materials()),
    )
    assert verdict["primary_diagnosis"] == "lighting_underexposure"


def test_interpret_camera_clipping_outranks_material_findings() -> None:
    verdict = A.interpret_audit(
        stats_by_id={vid: _NOISY for vid in "ABCDEFG"},
        gates_by_id={vid: _gate_stub(passed=False, wedge_ok=False) for vid in "ABCDEFG"},
        material_resolution=A.summarize_material_resolution(_raw_materials(missing=1)),
    )
    assert verdict["primary_diagnosis"] == "camera_pose_clipping"
    rules = {f["rule"] for f in verdict["findings"]}
    assert "missing_texture_assets" in rules  # still reported, just not primary


def test_interpret_missing_variants_recorded() -> None:
    verdict = A.interpret_audit(
        stats_by_id={"A": _CLEAN, "C": _NOISY},
        gates_by_id={"A": _gate_stub(), "C": _gate_stub()},
        material_resolution=A.summarize_material_resolution(_raw_materials()),
    )
    assert set(verdict["missing_variants"]) == {"B", "D", "E", "F", "G"}


# ---------------------------------------------------------------- WAM contract

def test_wam_contract_white_proxy_allowed_only_as_simplified_proxy() -> None:
    contract = A.build_wam_seed_media_contract(
        robot_material_mode=A.ROBOT_MATERIAL_MODE_WHITE_PROXY,
        seed_frame_visual_quality_status="completed",
    )
    assert contract["wam_conditioning_allowed"] is True
    assert contract["simplified_robot_visual_proxy"] is True
    assert contract["textured_robot_visual_fidelity_claimed"] is False


def test_wam_contract_blocks_unverified_textured_and_incomplete_status() -> None:
    contract = A.build_wam_seed_media_contract(
        robot_material_mode=A.ROBOT_MATERIAL_MODE_TEXTURED_UNVERIFIED,
        seed_frame_visual_quality_status="completed",
    )
    assert contract["wam_conditioning_allowed"] is False
    assert A.TEXTURED_MATERIAL_UNVERIFIED_BLOCKER in contract["blockers"]

    pending = A.build_wam_seed_media_contract(
        robot_material_mode=A.ROBOT_MATERIAL_MODE_WHITE_PROXY,
        seed_frame_visual_quality_status="pending",
    )
    assert pending["wam_conditioning_allowed"] is False


def test_wam_contract_noisy_textured_requires_visual_smoke_acceptance() -> None:
    rejected = A.build_wam_seed_media_contract(
        robot_material_mode=A.ROBOT_MATERIAL_MODE_VERIFIED_TEXTURED,
        seed_frame_visual_quality_status="completed",
        noise_grade_value="noisy",
        visual_smoke_passed=None,
    )
    assert rejected["wam_conditioning_allowed"] is False
    assert "noisy_textured_seed_requires_visual_smoke_acceptance" in rejected["blockers"]

    accepted = A.build_wam_seed_media_contract(
        robot_material_mode=A.ROBOT_MATERIAL_MODE_VERIFIED_TEXTURED,
        seed_frame_visual_quality_status="completed",
        noise_grade_value="noisy",
        visual_smoke_passed=True,
    )
    assert accepted["wam_conditioning_allowed"] is True
    assert accepted["noisy_textured_seed"] is True  # recorded, never silent


# ---------------------------------------------------------------- run analysis (end to end)

def _write_frame(path: Path, *, noisy: bool, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    # structured scene stand-in: horizontal luma gradient + a few darker fixtures with edges
    base = np.tile(np.linspace(90.0, 190.0, 160), (120, 1))
    base[30:70, 40:80] -= 55.0
    base[80:110, 100:150] -= 40.0
    if noisy:
        base = base + rng.normal(0.0, 45.0, base.shape)
    arr = np.clip(base, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.stack([arr] * 3, axis=2)).save(path)


def _worker_fixture(run_dir: Path, *, noisy_ids: set[str], material_raw: dict) -> Path:
    audit_dir = run_dir / A.AUDIT_SUBDIR_NAME
    plan = A.build_variant_plan()
    variant_results = []
    for row in plan["variants"]:
        vid = row["variant_id"]
        _write_frame(audit_dir / "variants" / vid / "frame_raw.png", noisy=vid in noisy_ids)
        variant_results.append({
            "variant_id": vid,
            "status": "completed",
            "frame_png": f"variants/{vid}/frame_raw.png",
            "render_settings": {"samples_per_pixel": 64},
        })
    material_resolution = A.summarize_material_resolution(material_raw)
    (audit_dir / A.MATERIAL_RESOLUTION_MANIFEST_NAME).write_text(json.dumps(material_resolution))
    (audit_dir / A.RENDER_SETTINGS_MANIFEST_NAME).write_text(json.dumps({
        "schema_version": A.RENDER_SETTINGS_MANIFEST_SCHEMA_VERSION,
        "renderer": "rtx_pathtracing",
        "resolution": [160, 120],
        "lighting_summary": {"light_count": 3},
        "runtime_metadata": {"gpu_name": "NVIDIA L40S", "isaac_version": "6.0.0"},
    }))
    (audit_dir / A.CAMERA_CONTRACT_NAME).write_text(json.dumps({
        "available": True, "camera_source": "derived_head_camera",
        "resolution": [160, 120], "pitch_down_deg": 22.0,
    }))
    (audit_dir / A.WORKER_RUN_MANIFEST_NAME).write_text(json.dumps({
        "schema_version": "g1_render_noise_audit_worker_run.v1",
        "task": "open the fridge door",
        "target_resolution": {"status": "resolved", "selected": {"target_object_label": "fridge"}},
        "stance_plan_summary": {"status": "accepted"},
        "placement_validation": {"status": "PASS"},
        "robot_asset": {"requested_g1_usd": "Isaac/Robots/Unitree/G1/g1.usd"},
        "arm_visibility": dict(_VIS_OK),
        "variant_plan": plan,
        "variant_results": variant_results,
    }))
    return audit_dir


def test_analyze_run_end_to_end_diagnoses_sample_starvation(tmp_path: Path) -> None:
    run_dir = tmp_path / "render_output"
    audit_dir = _worker_fixture(run_dir, noisy_ids={"B", "C"}, material_raw=_raw_materials())

    manifest = A.analyze_render_noise_audit_run(run_dir)

    assert manifest["status"] == "completed"
    assert manifest["variants_executed"] == ["A", "B", "C", "D", "E", "F", "G"]
    assert manifest["interpretation"]["primary_diagnosis"] == "render_budget_sample_starvation"
    assert manifest["gates"]["A"]["passed"] is True
    assert manifest["gates"]["C"]["robot_material_mode"] == A.ROBOT_MATERIAL_MODE_VERIFIED_TEXTURED
    assert (audit_dir / A.AUDIT_MANIFEST_NAME).is_file()
    assert (audit_dir / A.FRAME_STATS_NAME).is_file()
    assert (audit_dir / A.CONTACT_SHEET_NAME).is_file()
    assert manifest["contact_sheet"]["frame_count"] == 7
    assert not any(
        b.startswith("required_recorded_input_missing") for b in manifest["blockers"]
    )


def test_analyze_run_missing_textures_blocks_verified_textured_label(tmp_path: Path) -> None:
    run_dir = tmp_path / "render_output"
    _worker_fixture(run_dir, noisy_ids=set(), material_raw=_raw_materials(missing=2))

    manifest = A.analyze_render_noise_audit_run(run_dir)

    for vid in ("B", "C", "D", "E", "G"):
        assert manifest["gates"][vid]["robot_material_mode"] == A.ROBOT_MATERIAL_MODE_TEXTURED_UNVERIFIED
    rules = {f["rule"] for f in manifest["interpretation"]["findings"]}
    assert "missing_texture_assets" in rules


def test_analyze_run_blocked_without_worker_manifest(tmp_path: Path) -> None:
    manifest = A.analyze_render_noise_audit_run(tmp_path)
    assert manifest["status"] == "blocked"
    assert "audit_worker_run_manifest_missing" in manifest["blockers"]


def test_analyze_run_blocks_when_planned_variant_result_row_is_absent(tmp_path: Path) -> None:
    run_dir = tmp_path / "render_output"
    audit_dir = _worker_fixture(run_dir, noisy_ids=set(), material_raw=_raw_materials())
    worker = json.loads((audit_dir / A.WORKER_RUN_MANIFEST_NAME).read_text())
    # truncated worker upload: variant G never wrote a variant_results row at all
    worker["variant_results"] = [
        row for row in worker["variant_results"] if row["variant_id"] != "G"
    ]
    (audit_dir / A.WORKER_RUN_MANIFEST_NAME).write_text(json.dumps(worker))

    manifest = A.analyze_render_noise_audit_run(run_dir)

    assert manifest["status"] == "blocked"
    assert "planned_variant_result_missing:G" in manifest["blockers"]
    assert "G" in manifest["interpretation"]["missing_variants"]
    # the variants that did run are still analyzed for partial evidence
    assert manifest["gates"]["A"]["passed"] is True


def test_cli_plan_and_analyze(tmp_path: Path, capsys) -> None:
    plan_path = tmp_path / "plan.json"
    assert A.main(["plan", "--out", str(plan_path)]) == 0
    plan = json.loads(plan_path.read_text())
    assert [v["variant_id"] for v in plan["variants"]] == list("ABCDEFG")

    run_dir = tmp_path / "render_output"
    _worker_fixture(run_dir, noisy_ids={"B", "C"}, material_raw=_raw_materials())
    assert A.main(["analyze", "--run-dir", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "status=completed" in out
    assert "render_budget_sample_starvation" in out
