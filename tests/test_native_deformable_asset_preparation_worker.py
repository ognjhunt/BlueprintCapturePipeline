from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

from blueprint_pipeline import native_deformable_asset_preparation_worker as worker


def _blob(content: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(content)}\0".encode() + content, usedforsecurity=False
    ).hexdigest()


def _fake_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "IsaacLab"
    rows = {
        worker.DEFORMABLE_MATERIAL_API: (
            "source/material.py",
            "def spawn(prim_path, cfg):\n    return object()\n",
            "spawn",
            ["prim_path", "cfg"],
        ),
        worker.DEFORMABLE_AUTHORING_API: (
            "source/author.py",
            "def author(prim_path, cfg, stage):\n    return None\n",
            "author",
            ["prim_path", "cfg", "stage"],
        ),
        worker.DEFORMABLE_PHYSICS_BINDING_API: (
            "source/bind.py",
            (
                "def bind(prim_path, material_path, stage, stronger_than_descendants):\n"
                "    return None\n"
            ),
            "bind",
            ["prim_path", "material_path", "stage", "stronger_than_descendants"],
        ),
        worker.DEFORMABLE_BODY_CFG: (
            "source/body_cfg.py",
            "class BodyCfg:\n    pass\n",
            "BodyCfg",
            None,
        ),
        worker.DEFORMABLE_MATERIAL_CFG: (
            "source/material_cfg.py",
            "class MaterialCfg:\n    pass\n",
            "MaterialCfg",
            None,
        ),
    }
    modules: dict[str, types.ModuleType] = {}
    contract = {
        "material_spawn": {},
        "deformable_authoring": {},
        "physics_material_binding": {},
        "configuration_sources": {},
    }
    keys = ["material_spawn", "deformable_authoring", "physics_material_binding"]
    for index, (symbol, (relative, source, name, parameters)) in enumerate(rows.items()):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        module_name = symbol.split(":", 1)[0]
        module = types.ModuleType(module_name)
        module.__file__ = str(path)
        exec(compile(source, str(path), "exec"), module.__dict__)
        modules[module_name] = module
        row = {
            "symbol": symbol,
            "source_relative_path": relative,
            "source_git_blob_sha1": _blob(source.encode()),
        }
        if parameters is not None:
            row["parameters"] = parameters
        if index < 3:
            contract[keys[index]] = row
        else:
            contract["configuration_sources"][symbol] = row
        assert callable(getattr(module, name))
        setattr(module, symbol.split(":", 1)[1], getattr(module, name))
    cook_module = types.ModuleType("omni.physx.scripts.deformableUtils")
    cook_module.add_physx_deformable_body = lambda **_kwargs: True
    modules[cook_module.__name__] = cook_module
    monkeypatch.setattr(worker, "PINNED_NATIVE_CALL_CONTRACT", contract)
    return root, modules


def test_registry_binds_exact_source_blobs_origins_and_signatures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, modules = _fake_runtime(tmp_path, monkeypatch)
    registry = worker._build_registry(
        isaaclab_source_root=root, importer=lambda name: modules[name]
    )
    assert set(registry) == set(worker._ALL_SYMBOLS)

    material = Path(modules[worker.DEFORMABLE_MATERIAL_API.split(":")[0]].__file__)
    material.write_text(material.read_text() + "# drift\n", encoding="utf-8")
    with pytest.raises(worker.NativeDeformableAssetPreparationWorkerError) as exc:
        worker._build_registry(isaaclab_source_root=root, importer=lambda name: modules[name])
    assert "source_blob_mismatch" in str(exc.value)


def test_registry_rejects_signature_and_origin_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root, modules = _fake_runtime(tmp_path, monkeypatch)
    module = modules[worker.DEFORMABLE_AUTHORING_API.split(":")[0]]
    module.author = lambda prim_path, cfg: None
    setattr(module, worker.DEFORMABLE_AUTHORING_API.split(":")[1], module.author)
    with pytest.raises(worker.NativeDeformableAssetPreparationWorkerError) as exc:
        worker._build_registry(isaaclab_source_root=root, importer=lambda name: modules[name])
    assert "symbol_signature_invalid" in str(exc.value)

    _root, modules = _fake_runtime(tmp_path / "again", monkeypatch)
    module = modules[worker.DEFORMABLE_BODY_CFG.split(":")[0]]
    module.__file__ = str(tmp_path / "elsewhere.py")
    with pytest.raises(worker.NativeDeformableAssetPreparationWorkerError) as exc:
        worker._build_registry(isaaclab_source_root=_root, importer=lambda name: modules[name])
    assert "symbol_origin_invalid" in str(exc.value)


def test_pinned_native_call_contract_matches_runtime_source_packet_v14_blobs():
    expected = {
        worker.DEFORMABLE_MATERIAL_API: "8c12bee9442dbf4122b67234ff9ccca40cc02a74",
        worker.DEFORMABLE_AUTHORING_API: "8bd2c314bf931afe160759fb1ac3f92e24358ff3",
        worker.DEFORMABLE_PHYSICS_BINDING_API: "d0f0e8d9042a531ce617645cdc158fa4ac81f754",
        worker.DEFORMABLE_BODY_CFG: "d6dc99a847482a96fc7db07df023ad4f16584138",
        worker.DEFORMABLE_MATERIAL_CFG: "5c88731cf8d5b056812eb4713e534312eab1dc68",
    }

    rows = worker._source_rows()
    assert {symbol: rows[symbol]["source_git_blob_sha1"] for symbol in expected} == expected


def test_worker_reads_one_frozen_plan_and_delegates_without_claim_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    plan = tmp_path / "plan.json"
    plan.write_text('{"plan_digest":"sha256:' + "a" * 64 + '"}\n', encoding="utf-8")
    observed = {}

    def execute(**kwargs):
        observed.update(kwargs)
        return {"worker_result_digest": "sha256:" + "b" * 64}

    monkeypatch.setattr(worker, "execute_native_deformable_asset_preparation", execute)
    result = worker.run_native_deformable_asset_preparation_worker(
        plan_path=plan,
        expected_plan_digest="sha256:" + "a" * 64,
        package_root=tmp_path / "package",
        output_root=tmp_path / "output",
        isaaclab_source_root=tmp_path / "source",
        stage_api="stage-api",
        registry_builder=lambda **_kwargs: {
            symbol: (lambda: None) for symbol in worker._ALL_SYMBOLS
        },
    )
    assert result == {"worker_result_digest": "sha256:" + "b" * 64}
    assert observed["plan"] == {"plan_digest": "sha256:" + "a" * 64}
    assert observed["stage_api"] == "stage-api"
    assert set(observed["native_api_registry"]) == set(worker._ALL_SYMBOLS)


def test_plan_reader_rejects_symlink_fifo_and_oversize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(worker.NativeDeformableAssetPreparationWorkerError):
        worker.run_native_deformable_asset_preparation_worker(
            plan_path=link,
            expected_plan_digest="sha256:" + "a" * 64,
            package_root=tmp_path,
            output_root=tmp_path / "out",
            isaaclab_source_root=tmp_path,
            registry_builder=lambda **_kwargs: {},
        )

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(worker.NativeDeformableAssetPreparationWorkerError):
        worker._snapshot_regular_file(fifo, maximum_size=32, error="bad")

    monkeypatch.setattr(worker, "MAX_PLAN_BYTES", 1)
    with pytest.raises(worker.NativeDeformableAssetPreparationWorkerError):
        worker.run_native_deformable_asset_preparation_worker(
            plan_path=target,
            expected_plan_digest="sha256:" + "a" * 64,
            package_root=tmp_path,
            output_root=tmp_path / "out",
            isaaclab_source_root=tmp_path,
            registry_builder=lambda **_kwargs: {},
        )


def test_terminal_receipt_never_claims_execution_authority(tmp_path: Path):
    path = tmp_path / "terminal.json"
    value = {
        "schema_version": worker.WORKER_SCHEMA_VERSION,
        "status": "worker_payload_materialized_pending_trusted_execution_join",
        "claim_boundary": {
            "worker_payload_only": True,
            "trusted_execution_authority": False,
            "native_cook_qualified": False,
            "simulator_qualified": False,
            "physical_material_equivalence": False,
        },
    }
    worker._write_terminal(path, value)
    assert json.loads(path.read_text())["claim_boundary"]["native_cook_qualified"] is False
    with pytest.raises(worker.NativeDeformableAssetPreparationWorkerError):
        worker._write_terminal(path, value)


def test_main_starts_and_closes_one_app_and_retains_payload_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    events: list[str] = []

    class Application:
        def __init__(self, configuration):
            assert configuration == {"headless": True, "renderer": "RayTracedLighting"}
            events.append("start")

        def close(self):
            events.append("close")

    package = types.ModuleType("isaacsim")
    module = types.ModuleType("isaacsim.simulation_app")
    module.SimulationApp = Application
    monkeypatch.setitem(sys.modules, "isaacsim", package)
    monkeypatch.setitem(sys.modules, "isaacsim.simulation_app", module)
    monkeypatch.setattr(
        worker,
        "run_native_deformable_asset_preparation_worker",
        lambda **_kwargs: {"worker_result_digest": "sha256:" + "b" * 64},
    )
    terminal = tmp_path / "terminal.json"
    exit_code = worker.main(
        [
            "--plan",
            str(tmp_path / "plan.json"),
            "--expected-plan-digest",
            "sha256:" + "a" * 64,
            "--package-root",
            str(tmp_path / "package"),
            "--output-root",
            str(tmp_path / "output"),
            "--isaaclab-source-root",
            str(tmp_path / "source"),
            "--terminal-output",
            str(terminal),
        ]
    )
    assert exit_code == 0
    assert events == ["start", "close"]
    receipt = json.loads(terminal.read_text())
    assert receipt["status"] == "worker_payload_materialized_pending_trusted_execution_join"
    assert receipt["claim_boundary"] == {
        "worker_payload_only": True,
        "trusted_execution_authority": False,
        "native_cook_qualified": False,
        "simulator_qualified": False,
        "physical_material_equivalence": False,
    }


def test_main_writes_terminal_before_simulation_app_close_can_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class Application:
        def __init__(self, _configuration):
            pass

        def close(self):
            raise SystemExit(0)

    package = types.ModuleType("isaacsim")
    module = types.ModuleType("isaacsim.simulation_app")
    module.SimulationApp = Application
    monkeypatch.setitem(sys.modules, "isaacsim", package)
    monkeypatch.setitem(sys.modules, "isaacsim.simulation_app", module)
    monkeypatch.setattr(
        worker,
        "run_native_deformable_asset_preparation_worker",
        lambda **_kwargs: {"worker_result_digest": "sha256:" + "c" * 64},
    )
    terminal = tmp_path / "terminal.json"
    with pytest.raises(SystemExit):
        worker.main(
            [
                "--plan",
                str(tmp_path / "plan.json"),
                "--expected-plan-digest",
                "sha256:" + "a" * 64,
                "--package-root",
                str(tmp_path / "package"),
                "--output-root",
                str(tmp_path / "output"),
                "--isaaclab-source-root",
                str(tmp_path / "source"),
                "--terminal-output",
                str(terminal),
            ]
        )

    receipt = json.loads(terminal.read_text())
    assert receipt["status"] == "worker_payload_materialized_pending_trusted_execution_join"
    assert receipt["worker_result_digest"] == "sha256:" + "c" * 64
