"""Hermetic tests for the GPU runner's render-noise-audit helpers (no isaacsim import)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_isaac_g1_kitchen_parity_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("parity_runner_audit", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # would raise if it imported isaacsim at module load
    return mod


M = _load()


def test_runner_exposes_render_noise_audit_entrypoints() -> None:
    assert hasattr(M, "run_render_noise_audit")
    assert hasattr(M, "default_render_noise_audit_variants")
    assert hasattr(M, "render_noise_audit_plan_from_request")


def test_default_variant_fallback_matches_spec_matrix() -> None:
    variants = {v["variant_id"]: v for v in M.default_render_noise_audit_variants()}
    assert sorted(variants) == ["A", "B", "C", "D", "E", "F", "G"]
    assert variants["A"]["robot_material"] == "white_proxy"
    assert variants["B"]["denoiser_enabled"] is False
    assert variants["D"]["render_budget"] == "high"
    assert variants["F"]["robot_material"] == "simplified_diffuse"
    assert variants["G"]["lighting_boost"] is True


def test_execution_order_is_material_monotonic() -> None:
    order = M.audit_variant_execution_order(M.default_render_noise_audit_variants())
    assert order == ["B", "C", "D", "E", "G", "F", "A"]
    ranks = [
        M._AUDIT_MATERIAL_MONOTONIC_RANK[
            next(v for v in M.default_render_noise_audit_variants() if v["variant_id"] == vid)["robot_material"]
        ]
        for vid in order
    ]
    assert ranks == sorted(ranks)


def test_plan_from_request_prefers_shipped_plan_and_repairs_order() -> None:
    shipped = {
        "render_noise_audit": {
            "schema_version": "g1_render_noise_audit_variant_plan.v1",
            "variants": [
                {"variant_id": "A", "robot_material": "white_proxy",
                 "denoiser_enabled": True, "render_budget": "current_default",
                 "lighting_boost": False},
                {"variant_id": "B", "robot_material": "textured_original",
                 "denoiser_enabled": False, "render_budget": "current_default",
                 "lighting_boost": False},
            ],
            "execution_order": ["Z", "Q"],  # stale/invalid -> recomputed
        },
    }
    plan = M.render_noise_audit_plan_from_request(shipped)
    assert plan["source"] == "request"
    assert plan["execution_order"] == ["B", "A"]

    fallback = M.render_noise_audit_plan_from_request({})
    assert fallback["source"] == "runner_default_matrix"
    assert [v["variant_id"] for v in fallback["variants"]] == list("ABCDEFG")


def test_audit_samples_per_pixel_budget_mapping_and_clamp() -> None:
    assert M.audit_samples_per_pixel("current_default", default_spp=64, high_spp=384) == 64
    assert M.audit_samples_per_pixel("high", default_spp=64, high_spp=384) == 384
    assert M.audit_samples_per_pixel("high", default_spp=64, high_spp=9999) == 512
    assert M.audit_samples_per_pixel("", default_spp=64, high_spp=384) == 64


def test_audit_arm_visibility_from_pov_geometry() -> None:
    visibility = M.audit_arm_visibility_from_pov_geometry({
        "target_in_frame": True,
        "arm_roles_in_frame_by_arm": {
            "left": ["shoulder", "elbow", "hand"],
            "right": ["elbow"],
        },
    })
    assert visibility["left_arm_visible"] is True
    assert visibility["right_arm_visible"] is True
    assert visibility["left_end_effector_visible"] is True
    assert visibility["right_end_effector_visible"] is False
    assert visibility["both_end_effectors_visible"] is False
    assert visibility["target_in_frame"] is True
    assert visibility["evidence_source"] == "projected_usd_arm_link_geometry"

    empty = M.audit_arm_visibility_from_pov_geometry({})
    assert empty["left_arm_visible"] is False
    assert empty["both_end_effectors_visible"] is False


def test_arg_parser_accepts_render_noise_audit_flags() -> None:
    ap = M.build_arg_parser()
    args = ap.parse_args([
        "--out-dir", "/tmp/out",
        "--render-noise-audit",
        "--audit-high-spp", "256",
        "--audit-warmup-frames", "5",
        "--audit-boost-light-intensity", "5000",
    ])
    assert args.render_noise_audit is True
    assert args.audit_high_spp == 256
    assert args.audit_warmup_frames == 5
    assert args.audit_boost_light_intensity == 5000.0


def test_audit_writes_result_before_simulation_close() -> None:
    source = _RUNNER.read_text()
    audit_fn = source[source.index("def run_render_noise_audit("):]
    audit_fn = audit_fn[: audit_fn.index("\ndef build_arg_parser")]
    assert "_write_result()" in audit_fn
    assert "sim.close()" in audit_fn
    # the completed-result write happens before the finally-close block
    assert audit_fn.rindex("_write_result()") < audit_fn.rindex("sim.close()")


def test_audit_captures_raw_frames_without_software_denoise() -> None:
    source = _RUNNER.read_text()
    audit_fn = source[source.index("def run_render_noise_audit("):]
    audit_fn = audit_fn[: audit_fn.index("\ndef build_arg_parser")]
    assert "software_denoise=False" in audit_fn
