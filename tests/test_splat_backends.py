"""Tests for the swappable 3DGS backend registry."""

from __future__ import annotations

import pytest

import blueprint_pipeline.splat_backends as splat_backends
from blueprint_pipeline.splat_backends import (
    BACKEND_KINDS,
    SplatBackend,
    get_backend,
    list_backends,
    register_backend,
)


def test_builtins_registered() -> None:
    names = {backend["name"] for backend in list_backends()}
    assert {
        "splat_transform",
        "spark",
        "threedgrut",
        "particlefield_usd",
        "isaac_nurec",
        "artifixer",
    } <= names


def test_exporters_include_particlefield() -> None:
    exporters = {backend["name"] for backend in list_backends("exporter")}
    assert {"threedgrut", "particlefield_usd"} <= exporters


def test_describe_has_kind_and_available_bool() -> None:
    for backend in list_backends():
        assert backend["kind"] in BACKEND_KINDS
        assert isinstance(backend["available"], bool)  # probes never raise


def test_list_by_kind() -> None:
    renderers = {backend["name"] for backend in list_backends("renderer")}
    assert {"spark", "isaac_nurec"} <= renderers
    assert "splat_transform" not in renderers  # it is a decoder
    assert {backend["name"] for backend in list_backends("enhancer")} == {
        "artifixer",
        "difix3d",
        "harmonizer",
    }


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_backend("nonexistent_backend")


def test_register_validates_kind() -> None:
    with pytest.raises(ValueError):
        register_backend(
            SplatBackend("x", "badkind", "s", (), lambda: False, lambda **kwargs: {})
        )


def test_artifixer_fail_closed() -> None:
    backend = get_backend("artifixer")
    assert backend.kind == "enhancer"
    result = backend.run(
        checkpoint_pt="/x.pt",
        save_dir="/out",
        split_path="/s.json",
    )
    assert result["status"] == "blocked"
    assert "artifixer_unavailable" in result["blockers"]
    assert "remediation" in result


def test_artifixer_installed_runtime_still_requires_exact_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(splat_backends, "_artifixer_available", lambda: True)
    result = get_backend("artifixer").run(
        checkpoint_pt="/x.pt",
        save_dir="/out",
        split_path="/s.json",
    )
    assert result["status"] == "blocked"
    assert "artifixer_checkpoint_digest_missing_or_invalid" in result["blockers"]
    assert "artifixer_frozen_real_heldout_manifest_required" in result["blockers"]
    assert "commercial_model_use_not_qualified" in result["blockers"]
    assert result["enhancement_method_audit"]["status"].startswith("rejected_")
    assert result["claim_ceiling"] == "generated_visual_support"


@pytest.mark.parametrize(
    ("backend_name", "expected_blocker"),
    [
        ("difix3d", "source_and_model_license_noncommercial"),
        ("harmonizer", "checkpoint_digest_not_pinned_in_worker"),
    ],
)
def test_unqualified_enhancement_candidates_emit_deterministic_rejection(
    backend_name: str, expected_blocker: str
) -> None:
    backend = get_backend(backend_name)
    assert backend.kind == "enhancer"
    assert backend.available() is False
    result = backend.run()
    assert result["status"] == "blocked"
    assert expected_blocker in result["blockers"]
    assert result["enhancement_method_audit"]["independent_evaluator_required"] is True
    assert result["enhancement_method_audit"]["metric_or_collision_proof_effect"] is False


def test_isaac_nurec_is_gpu_worker_only() -> None:
    result = get_backend("isaac_nurec").run()
    assert result["status"] == "blocked"
    assert "isaac_nurec_render_is_gpu_worker_only" in result["blockers"]
    assert result["runner"].endswith("run_isaac_splat_nurec_render.py")


def test_threedgrut_backend_fail_closed(tmp_path) -> None:
    result = get_backend("threedgrut").run(
        tmp_path / "missing.ply",
        tmp_path / "o.usdz",
    )
    assert result["status"] == "blocked"
