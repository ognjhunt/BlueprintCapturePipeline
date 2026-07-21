from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import jsonschema

from blueprint_pipeline.omniverse_library_preflight import (
    ENABLE_ENV,
    build_omniverse_preflight_benchmark,
    build_omniverse_preflight_benchmark_suite,
    inspect_usd_features,
    required_checks_for,
    run_ovphysx_preflight,
    run_ovrtx_preflight,
)
from blueprint_pipeline.simready_assets import build_simready_assets
from tests.test_simready_assets import (
    _build_capture_root,
    _object_geometry_manifest,
    _site_world_spec,
    _task_anchor_manifest,
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_schema(artifact: Path, schema_name: str) -> None:
    schema = _read(Path(__file__).resolve().parents[1] / "docs" / "schemas" / schema_name)
    jsonschema.Draft202012Validator(schema).validate(_read(artifact))


def _capture(tmp_path: Path) -> Path:
    root = _build_capture_root(tmp_path)
    build_simready_assets(
        capture_root=root,
        object_geometry_manifest=_object_geometry_manifest(),
        task_anchor_manifest=_task_anchor_manifest(),
        site_world_spec=_site_world_spec(),
    )
    return root


def _fake_worker(tmp_path: Path) -> Path:
    path = tmp_path / "fake-omniverse-worker"
    path.write_text(
        """#!/usr/bin/env python3
import hashlib, json, pathlib, sys
output, input_path, config_path, mode, component, version, revision, output_dir = sys.argv[1:]
config = json.loads(pathlib.Path(config_path).read_text())
digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
checks = {
 "ovrtx": ["usd_scene_load", "requested_sensor_outputs_nonempty", "sensor_metadata_complete", "semantic_id_map"],
 "ovphysx": ["usd_scene_load", "gravity_and_rigid_body_integration", "collider_presence_and_penetration", "joint_and_limit_inspection", "mass_and_friction_bounds", "fixed_step_state_snapshot"],
}[component]
out = pathlib.Path(output_dir)
outputs = []
if component == "ovrtx":
  for kind in ("rgb", "depth", "semantic_segmentation"):
    artifact = out / (kind + ".bin")
    artifact.write_bytes((kind + "-stable").encode())
    outputs.append({"kind": kind, "path": artifact.name, "metadata": {"width": 8, "height": 8}})
else:
  artifact = out / "state_snapshots.json"
  artifact.write_text('{"states":[[0,0,0],[0,0,-1]]}\\n')
  outputs.append({"kind": "state_snapshots", "path": artifact.name, "metadata": {"steps": 2}})
report = {
 "component_name": component,
 "component_version": version,
 "source_revision": revision,
 "configuration_sha256": digest,
 "runtime": {"python_version": "3.12.0", "cuda_version": "13.0", "driver_version": "590.1", "gpu_identity": {"name": "fixture", "uuid": "GPU-fixture"}},
 "checks": [{"name": name, "status": "passed"} for name in checks],
 "outputs": outputs,
 "failure_classes_checked": ["usd_scene_load", "empty_sensor_output"],
 "required_sensor_metadata_preserved": True,
}
pathlib.Path(output).write_text(json.dumps(report, sort_keys=True) + "\\n")
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _command(worker: Path, component: str) -> list[str]:
    return [
        str(worker),
        "{output}",
        "{input}",
        "{config}",
        "{mode}",
        component,
        "{component_version}",
        "{source_revision}",
        "{output_dir}",
    ]


def test_ovrtx_preflight_is_dual_gated_and_passes_stable_fake_worker(tmp_path: Path) -> None:
    root = _capture(tmp_path / "capture")
    worker = _fake_worker(tmp_path)
    blocked = run_ovrtx_preflight(
        capture_root=root,
        worker_command=_command(worker, "ovrtx"),
        component_version="0.4.0",
        source_revision="fixture-ovrtx-revision",
        license_id="LicenseRef-NvidiaProprietary",
        license_compatible=True,
        sensor_configuration={
            "runtime_expectations": {
                "python_version": "3.12.0",
                "cuda_version": "13.0",
                "driver_version": "590.1",
                "gpu_uuid": "GPU-fixture",
                "shader_configuration_id": "fixture-shaders-v1",
            }
        },
    )
    assert blocked["status"] == "blocked"
    assert blocked["component_ran"] is False

    passed = run_ovrtx_preflight(
        capture_root=root,
        worker_command=_command(worker, "ovrtx"),
        component_version="0.4.0",
        source_revision="fixture-ovrtx-revision",
        license_id="LicenseRef-NvidiaProprietary",
        license_compatible=True,
        sensor_configuration={
            "runtime_expectations": {
                "python_version": "3.12.0",
                "cuda_version": "13.0",
                "driver_version": "590.1",
                "gpu_uuid": "GPU-fixture",
                "shader_configuration_id": "fixture-shaders-v1",
            }
        },
        allow_external_preflight=True,
        env={ENABLE_ENV: "true", "PATH": f"{worker.parent}:{os.defpath}"},
    )
    assert passed["status"] == "passed_advisory"
    result = _read(Path(passed["result_path"]))
    request = _read(Path(passed["request_path"]))
    receipt = _read(Path(passed["runtime_receipt_path"]))
    assert result["required_sensor_metadata_preserved"] is True
    assert receipt["repeatable_output_digests"] is True
    assert "semantic_id_map" in request["required_checks"]  # type: ignore[operator]
    assert result["claim_boundary"]["rank_fidelity_result_proven"] is False  # type: ignore[index]
    _validate_schema(Path(passed["request_path"]), "omniverse_preflight_request.schema.json")
    _validate_schema(Path(passed["result_path"]), "omniverse_preflight_result.schema.json")
    _validate_schema(
        Path(passed["runtime_receipt_path"]),
        "omniverse_preflight_runtime_receipt.schema.json",
    )
    _validate_schema(
        Path(passed["claim_boundary_path"]),
        "omniverse_preflight_claim_boundary.schema.json",
    )


def test_ovphysx_preflight_passes_fake_worker_without_claim_upgrade(tmp_path: Path) -> None:
    root = _capture(tmp_path / "capture")
    worker = _fake_worker(tmp_path)
    passed = run_ovphysx_preflight(
        capture_root=root,
        worker_command=_command(worker, "ovphysx"),
        component_version="0.4.13",
        source_revision="fixture-ovphysx-revision",
        license_id="BSD-3-Clause-and-NVIDIA-Omniverse-License",
        license_compatible=True,
        physics_configuration={
            "device": "cpu",
            "runtime_expectations": {
                "python_version": "3.12.0",
                "solver_configuration_id": "fixture-solver-v1",
            },
        },
        allow_external_preflight=True,
        env={ENABLE_ENV: "true", "PATH": f"{worker.parent}:{os.defpath}"},
    )
    assert passed["status"] == "passed_advisory"
    result = _read(Path(passed["result_path"]))
    assert result["claim_boundary"]["physics_contact_task_success_proven"] is False  # type: ignore[index]
    assert result["claim_boundary"]["isaac_sim_execution_proven"] is False  # type: ignore[index]


def test_particlefield_and_episode_features_add_specific_ovrtx_checks(tmp_path: Path) -> None:
    usd = tmp_path / "episode.usda"
    usd.write_text(
        '#usda 1.0\ndef ParticleField3DGaussianSplat "Scene" { double3 xformOp:translate.timeSamples = { 0: (0,0,0) } }\n',
        encoding="utf-8",
    )
    features = inspect_usd_features(usd)
    checks = required_checks_for(
        "ovrtx",
        configuration={"episode_mode": True},
        usd_features=features,
        required_output_kinds=("rgb", "semantic_segmentation"),
    )
    assert features["particlefield_gaussian_splat"] is True
    assert "particlefield_gaussian_splat_render" in checks
    assert "dynamic_transform_update" in checks
    assert "semantic_id_map" in checks


def test_benchmark_requires_accepted_same_scene_isaac_baseline(tmp_path: Path) -> None:
    scene_hash = "a" * 64
    result = {
        "status": "passed_advisory",
        "input_sha256": scene_hash,
        "failure_classes_checked": ["usd_scene_load"],
        "required_sensor_metadata_preserved": True,
    }
    receipt = {
        "repeatable_output_digests": True,
        "cold_run": {"execution": {"duration_seconds": 1.0}},
        "warm_run": {"execution": {"duration_seconds": 0.5}},
    }
    isaac = {
        "accepted_fixture": True,
        "isaac_execution_proven": True,
        "input_sha256": scene_hash,
        "failure_classes_checked": ["usd_scene_load"],
        "runtime": {"cold_start_seconds": 10.0, "warm_start_seconds": 5.0},
    }
    paths: dict[str, Path] = {}
    for name, payload in (("result", result), ("receipt", receipt), ("isaac", isaac)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(payload), encoding="utf-8")
    benchmark = build_omniverse_preflight_benchmark(
        output_path=tmp_path / "benchmark.json",
        ovrtx_result_path=paths["result"],
        ovrtx_receipt_path=paths["receipt"],
        ovphysx_result_path=paths["result"],
        ovphysx_receipt_path=paths["receipt"],
        isaac_baseline_path=paths["isaac"],
    )
    assert benchmark["decisions"]["ovrtx"]["decision"] == "candidate_retained"
    assert benchmark["claim_boundary"]["component_retention_is_production_qualification"] is False
    _validate_schema(tmp_path / "benchmark.json", "omniverse_preflight_benchmark.schema.json")


def test_benchmark_suite_requires_valid_negative_memory_and_same_scene(tmp_path: Path) -> None:
    scene_hash = "b" * 64
    cases = []
    for kind in ("valid", "negative"):
        isaac = {
            "accepted_fixture": True,
            "isaac_execution_proven": True,
            "input_sha256": scene_hash,
            "runtime": {"cold_start_seconds": 10.0, "warm_start_seconds": 5.0},
            "failure_classes_detected": ["usd_scene_load"] if kind == "negative" else [],
        }
        isaac_path = tmp_path / f"{kind}-isaac.json"
        isaac_path.write_text(json.dumps(isaac), encoding="utf-8")
        case = {
            "fixture_id": f"{kind}-fixture",
            "kind": kind,
            "isaac_baseline_path": isaac_path.name,
            "expected_failure_classes": {"ovrtx": "usd_scene_load", "ovphysx": "usd_scene_load"}
            if kind == "negative"
            else {},
        }
        for component in ("ovrtx", "ovphysx"):
            result = {
                "status": "passed_advisory" if kind == "valid" else "blocked",
                "input_sha256": scene_hash,
                "failure_classes_detected": ["usd_scene_load"] if kind == "negative" else [],
                "required_sensor_metadata_preserved": True,
            }
            run = {
                "execution": {
                    "duration_seconds": 1.0,
                    "resource_usage": {"maximum_resident_set_size_platform_units": 4096},
                },
                "metrics": {
                    "gpu_memory_baseline_bytes": 1024,
                    "gpu_memory_peak_observed_bytes": 2048,
                },
            }
            receipt = {
                "input_sha256": scene_hash,
                "repeatable_output_digests": True,
                "cold_run": run,
                "warm_run": run,
            }
            result_path = tmp_path / f"{kind}-{component}-result.json"
            receipt_path = tmp_path / f"{kind}-{component}-receipt.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            case[f"{component}_result_path"] = result_path.name
            case[f"{component}_receipt_path"] = receipt_path.name
        cases.append(case)
    manifest = tmp_path / "suite.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "omniverse_preflight_benchmark_suite_manifest.v1",
                "frozen": True,
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "suite-result.json"
    result = build_omniverse_preflight_benchmark_suite(manifest_path=manifest, output_path=output)
    assert result["status"] == "completed"
    assert result["decisions"]["ovrtx"]["decision"] == "candidate_retained"
    assert result["decisions"]["ovphysx"]["decision"] == "candidate_retained"
    assert result["claim_boundary"]["candidate_retention_is_production_qualification"] is False
    _validate_schema(output, "omniverse_preflight_benchmark_suite.schema.json")
