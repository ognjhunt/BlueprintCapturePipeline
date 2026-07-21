"""Tests for the swappable 3DGS backend registry."""

from __future__ import annotations

import pytest

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
    assert {backend["name"] for backend in list_backends("enhancer")} == {"artifixer"}


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
