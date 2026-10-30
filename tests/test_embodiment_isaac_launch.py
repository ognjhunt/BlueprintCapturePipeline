"""Pinned SDK dependency graph replay; no native importer or GPU execution."""
from __future__ import annotations

import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from blueprint_pipeline import deployment_embodiment_isaac_launch as launch
from tests.test_native_task_isaaclab_launch import _receipt

FIXTURES = Path(__file__).parent / "fixtures"
GRAPH = json.loads((FIXTURES / "isaacsim_601_importer_dependencies.json").read_text())["extensions"]


def resolve_extensions(requested, excluded):
    """Replay official extension.toml dependencies, not a fake successful import."""
    enabled = set()

    def visit(name):
        if name in excluded:
            raise ValueError("excluded_dependency:" + name)
        if name not in enabled:
            enabled.add(name)
            for dependency in GRAPH.get(name, {}).get("dependencies", []):
                visit(dependency)

    for name in requested:
        visit(name)
    return enabled


@pytest.fixture
def source(tmp_path):
    receipt = _receipt(tmp_path)
    root = Path(json.loads(receipt.read_text())["runtime_experience"]["path"]).parent
    original = (FIXTURES / "isaaclab_headless_ffff603e.kit").read_bytes()
    (root / "isaaclab.python.headless.kit").write_bytes(original)
    return receipt, root, original


class App:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def fake_sdk(calls, app):
    def factory(**kwargs):
        calls.append(kwargs)
        experience_path = Path(kwargs["experience"])
        # The native importer rejected the former hyphenated experience names
        # before loading any body. Use a conservative dotted Kit identifier.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", experience_path.stem):
            raise ValueError("kit_experience_extension_identifier_invalid")
        experience = experience_path.read_text()
        excluded = json.loads(re.search(r"^app.extensions.excluded = (.*)$", experience, re.M)[1])
        args = kwargs["kit_args"].split()
        requested = [args[i + 1] for i, value in enumerate(args) if value == "--enable"]
        app.enabled = resolve_extensions(requested, excluded)
        return SimpleNamespace(app=app)

    def probe(importer):
        return {"extensions": {name: name in app.enabled for name in
                [*launch.UPSTREAM_EXCLUSIONS, *launch.IMPORTERS.values()]},
                "modules": {"torch": {"version": "2.10.0+cu128", "path": "/development/torch"},
                            "warp": {"version": "1.13.0", "path": "/development/warp"}},
                "importer": {"module": importer} if importer else None}

    return factory, probe


def test_production_failure_has_two_independent_pinned_dependency_conflicts():
    with pytest.raises(ValueError, match="omni.isaac.ml_archive"):
        resolve_extensions(["isaacsim.replicator.nurec_utils"], launch.UPSTREAM_EXCLUSIONS)
    for importer in launch.IMPORTERS.values():
        with pytest.raises(ValueError, match="omni.kit.pip_archive|isaacsim.pip.newton"):
            resolve_extensions([importer], launch.UPSTREAM_EXCLUSIONS)
        # Removing Newton alone still leaves its Kit archive dependency blocked.
        with pytest.raises(ValueError, match="omni.kit.pip_archive"):
            resolve_extensions([importer], ["omni.kit.pip_archive", "omni.isaac.ml_archive", "omni.warp.core"])


@pytest.mark.parametrize("source_format", ["urdf", "mjcf", None])
def test_asset_profiles_resolve_official_graph_and_retain_source_and_warp(source, tmp_path, source_format):
    receipt, root, original = source
    app, calls = App(), []
    factory, probe = fake_sdk(calls, app)
    returned, recorded = launch.launch_embodiment_isaaclab(receipt, output=tmp_path / "launch",
        source_format=source_format, app_launcher_factory=factory, runtime_probe=probe)
    assert returned is app and not app.closed
    assert (root / "isaaclab.python.headless.kit").read_bytes() == original
    actual = calls[0]
    assert actual["headless"] and not actual["enable_cameras"] and actual["device"] == "cuda:0"
    assert "nurec" not in actual["kit_args"] and "pink" not in actual["kit_args"] and "spg" not in actual["kit_args"]
    assert not app.enabled.intersection({"omni.isaac.ml_archive", "omni.warp.core"})
    if source_format:
        assert {launch.IMPORTERS[source_format], "isaacsim.pip.newton", "omni.kit.pip_archive"} <= app.enabled
        text = Path(actual["experience"]).read_text()
        assert "${app}" not in text and str(root) + "/../source" in text
    else:
        assert Path(actual["experience"]) == root / "isaaclab.python.headless.kit"
        assert recorded["profile"]["excluded_extensions"] == launch.UPSTREAM_EXCLUSIONS
    assert json.loads((tmp_path / "launch/launch.json").read_text()) == recorded


@pytest.mark.parametrize("source_format", ["urdf", "mjcf"])
def test_former_importer_experience_names_fail_before_extension_loading(source, tmp_path, source_format):
    _, root, original = source
    legacy = tmp_path / ("embodiment-import-" + source_format + ".kit")
    legacy.write_bytes(original)
    app = App()
    factory, _ = fake_sdk([], app)
    with pytest.raises(ValueError, match="kit_experience_extension_identifier_invalid"):
        factory(experience=str(legacy), kit_args="--enable " + launch.IMPORTERS[source_format])
    assert not hasattr(app, "enabled")
    assert (root / "isaaclab.python.headless.kit").read_bytes() == original


@pytest.mark.parametrize("change,reason", [
    ("warp_extension", "excluded_extension_loaded"),
    ("wrong_warp", "runtime_dependency_changed"),
    ("importer_missing", "official_importer_not_enabled"),
])
def test_post_start_readback_rejects_real_dependency_drift_and_flushes_reason(source, tmp_path, capsys, change, reason):
    app = App()
    factory, probe = fake_sdk([], app)

    def altered(importer):
        value = probe(importer)
        if change == "warp_extension":
            value["extensions"]["omni.warp.core"] = True
        elif change == "wrong_warp":
            value["modules"]["warp"]["version"] = "0.0-development"
        else:
            value["extensions"][importer] = False
        return value

    with pytest.raises(ValueError, match=reason):
        launch.launch_embodiment_isaaclab(source[0], output=tmp_path / "launch", source_format="urdf",
            app_launcher_factory=factory, runtime_probe=altered)
    assert app.closed and reason in capsys.readouterr().err
    assert not (tmp_path / "launch/launch.json").exists()


def test_changed_experience_cannot_silently_remove_more_exclusions(source, tmp_path):
    receipt, root, original = source
    (root / "isaaclab.python.headless.kit").write_bytes(original.replace(b"omni.isaac.ml_archive", b"different_archive"))
    with pytest.raises(ValueError, match="headless_source_changed"):
        launch.launch_embodiment_isaaclab(receipt, output=tmp_path / "launch")


def test_fixture_is_exact_pinned_headless_source():
    assert launch._sha((FIXTURES / "isaaclab_headless_ffff603e.kit").read_bytes()) == launch.HEADLESS_SHA256
