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
    names = {b["name"] for b in list_backends()}
    assert {
        "splat_transform", "spark", "threedgrut", "particlefield_usd", "isaac_nurec", "artifixer"
    } <= names


def test_exporters_include_particlefield() -> None:
    exporters = {b["name"] for b in list_backends("exporter")}
    assert {"threedgrut", "particlefield_usd"} <= exporters


def test_describe_has_kind_and_available_bool() -> None:
    for b in list_backends():
        assert b["kind"] in BACKEND_KINDS
        assert isinstance(b["available"], bool)  # probes never raise


def test_list_by_kind() -> None:
    renderers = {b["name"] for b in list_backends("renderer")}
    assert {"spark", "isaac_nurec"} <= renderers
    assert "splat_transform" not in renderers  # it's a decoder
    assert {b["name"] for b in list_backends("enhancer")} == {"artifixer"}


def test_get_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_backend("nonexistent_backend")


def test_register_validates_kind() -> None:
    with pytest.raises(ValueError):
        register_backend(SplatBackend("x", "badkind", "s", (), lambda: False, lambda **k: {}))


def test_artifixer_fail_closed() -> None:
    b = get_backend("artifixer")
    assert b.kind == "enhancer"
    res = b.run(checkpoint_pt="/x.pt", save_dir="/out", split_path="/s.json")
    assert res["status"] == "blocked"
    assert "artifixer_unavailable" in res["blockers"]
    assert "remediation" in res


def test_isaac_nurec_is_gpu_worker_only() -> None:
    res = get_backend("isaac_nurec").run()
    assert res["status"] == "blocked"
    assert "isaac_nurec_render_is_gpu_worker_only" in res["blockers"]
    assert res["runner"].endswith("run_isaac_splat_nurec_render.py")


def test_threedgrut_backend_fail_closed(tmp_path) -> None:
    res = get_backend("threedgrut").run(tmp_path / "missing.ply", tmp_path / "o.usdz")
    assert res["status"] == "blocked"
