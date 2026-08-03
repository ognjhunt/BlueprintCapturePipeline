from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from blueprint_pipeline.measurement_adapter_execution import (
    MeasurementAdapterExecutionError,
    build_measurement_adapter_execution_request,
    run_measurement_adapter_execution,
)
from blueprint_pipeline.measurement_adapter_runtime import build_measurement_adapter_descriptor
from blueprint_pipeline.measurement_chrono_dem_cuda_adapter import (
    BINARY_NAME,
    EXPECTED_ENGINE_VERSION,
    EXPECTED_SOURCE_COMMIT,
    IMPLEMENTATION_ID,
    IMPLEMENTATION_VERSION,
    WORKER_SCRIPT,
    implementation_digest,
    run_chrono_dem_cuda_request,
)
from blueprint_pipeline.measurement_qualification_benchmarks import (
    build_benchmark_case_manifest,
    build_qualification_benchmark_spec,
)


ROOT = Path(__file__).parents[1]
CORPUS_PATH = ROOT / "tests/fixtures/measurement_chrono_dem_cuda_v1/corpus.json"
QUALIFICATION_SPLIT_DIGEST = "sha256:" + "f" * 64
CONTROLLER_SCOPE_DIGEST = (
    "sha256:" + hashlib.sha256(b"chrono-dem-cuda-development-no-controller").hexdigest()
)


def _request(case_index: int = 0) -> dict:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    corpus_digest = "sha256:" + hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()
    spec = build_qualification_benchmark_spec(
        benchmark_id="capture-to-deformation",
        benchmark_version="development-chrono-dem-cuda-synthetic-1",
        method_ids=["project-chrono-10"],
        development_split_digest=corpus_digest,
        qualification_split_digest=QUALIFICATION_SPLIT_DIGEST,
        capture_bundle_digests=[corpus_digest],
        robot_controller_digests=[CONTROLLER_SCOPE_DIGEST],
        acceptance_thresholds={
            "maximum_mean_absolute_error": 1.0,
            "maximum_mismatch_rate": 0.0,
            "maximum_harmful_false_negative_rate": 0.0,
            "minimum_coverage": 4 / 6,
        },
        compute_budget={"usd": 1.0, "maximum_duration_seconds": 1800},
        minimum_repeated_trials=2,
        lane="granular",
    )
    row = dict(corpus["cases"][case_index])
    case_id = row.pop("case_id")
    case = build_benchmark_case_manifest(
        spec,
        case_id=case_id,
        split="development",
        input_artifact_digests=[corpus_digest],
        task_class="granular_manipulation",
        material_regime="synthetic_chrono_dem_granular_media",
        operating_point={**corpus["shared_operating_point"], **row},
    )
    return build_measurement_adapter_execution_request(
        build_measurement_adapter_descriptor("project-chrono-10"),
        spec,
        case,
        execution_id=f"chrono-dem-cuda-synthetic-{case_index + 1:03d}",
        implementation_id=IMPLEMENTATION_ID,
        implementation_version=IMPLEMENTATION_VERSION,
        implementation_digest=implementation_digest(),
        backend_id="chrono-dem-cuda",
        precision="float32",
        seed=0,
        solver_settings={
            "binary_name": BINARY_NAME,
            "cuda_device_count": 1,
            "module": "chrono_dem",
            "replay_count": 2,
            "source_commit": EXPECTED_SOURCE_COMMIT,
        },
        timeout_seconds=120,
    )


def _fake_probe(
    tmp_path: Path,
    *,
    cuda_device_count: int = 1,
    density_override: float | None = None,
) -> Path:
    path = tmp_path / BINARY_NAME
    source = (
        "#!/usr/bin/env python3\n"
        "import argparse, json, math\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--density-g-cm3', type=float, required=True)\n"
        "p.add_argument('--friction', type=float, required=True)\n"
        "p.add_argument('--rolling-friction', type=float, required=True)\n"
        "p.add_argument('--duration-s', type=float, required=True)\n"
        "p.add_argument('--timestep-s', type=float, required=True)\n"
        "p.add_argument('--settle-speed-threshold-cm-s', type=float, required=True)\n"
        "a=p.parse_args()\n"
        f"density={density_override!r} if {density_override is not None!r} else a.density_g_cm3\n"
        "weight=27*(4/3)*math.pi*density*980*1e-5\n"
        "trace=[{'time_s':(i+1)*a.duration_s/20,'centroid_m':[0.0,0.0,-0.07],"
        "'horizontal_span_m':0.05,'maximum_speed_m_s':0.0,'settled_fraction':1.0,"
        "'contact_count':5,'kinetic_energy_native':0.0,'ground_reaction_force_n':weight} "
        "for i in range(20)]\n"
        "result={'schema_version':'measurement_chrono_dem_cuda_probe_result.v1',"
        "'status':'completed','chrono_version':" + repr(EXPECTED_ENGINE_VERSION) + ","
        "'source_commit':" + repr(EXPECTED_SOURCE_COMMIT) + ","
        "'chrono_dem_module_used':True,'cuda_device_count':" + repr(cuda_device_count) + ","
        "'cuda_device_name':'fixture CUDA device','cuda_compute_capability':'8.9',"
        "'particle_count':27,'density_g_cm3':density,'friction':a.friction,"
        "'rolling_friction':a.rolling_friction,'duration_s':a.duration_s,"
        "'timestep_s':a.timestep_s,'initial_horizontal_span_m':0.042,"
        "'final_horizontal_span_m':0.05,'spread_ratio':1.19047619,"
        "'final_settled_fraction':1.0,'final_maximum_speed_m_s':0.0,"
        "'maximum_contact_count':5,'expected_static_weight_n':weight,"
        "'final_ground_reaction_force_n':weight,'maximum_ground_reaction_force_n':weight,"
        "'penetration_m':0.0,'trace':trace}\n"
        "print(json.dumps(result, sort_keys=True))\n"
    )
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def test_chrono_descriptor_supports_conda_core_and_exact_source_dem() -> None:
    descriptor = build_measurement_adapter_descriptor("project-chrono-10")
    assert descriptor["target_version"] == EXPECTED_ENGINE_VERSION
    assert descriptor["execution_mode"] == "isolated_external_conda_or_exact_source_build"
    assert descriptor["probe_contract"]["python_distributions"] == []
    assert descriptor["probe_contract"]["executables"] == []


def test_chrono_dem_worker_executes_two_fake_cuda_cases_without_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_probe(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    bundles = [
        run_measurement_adapter_execution(
            _request(index),
            command_argv=[os.sys.executable, str(WORKER_SCRIPT)],
            execute=True,
        )
        for index in range(2)
    ]
    assert [row["receipt"]["status"] for row in bundles] == ["completed", "completed"]
    for bundle in bundles:
        runtime = bundle["receipt"]["runtime_observations"]
        assert runtime["chrono_dem_module_used"] is True
        assert runtime["cuda_available"] is True
        assert runtime["cuda_device_count"] == 1
        assert runtime["cpu_fallback_used"] is False
        assert runtime["deterministic_replay_match"] is True
        assert runtime["material_characterization_scope"] == "synthetic_parameters_only"
        assert runtime["q_gran_qualification_created"] is False
        assert runtime["r7_admission_created"] is False
        assert runtime["physical_success_established"] is False
        assert bundle["prediction"]["unsafe_condition_predicted"] is False
        assert bundle["qualification_created"] is False
        assert bundle["catalog_mutated"] is False


def test_chrono_dem_worker_rejects_solver_before_binary_lookup() -> None:
    request = copy.deepcopy(_request())
    request["runtime_configuration"]["solver_settings"]["module"] = "chrono_core"
    encoded = json.dumps(
        request["runtime_configuration"]["solver_settings"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    request["runtime_configuration"]["solver_settings_digest"] = (
        "sha256:" + hashlib.sha256(encoded).hexdigest()
    )
    request.pop("execution_request_digest")
    with pytest.raises(MeasurementAdapterExecutionError, match="solver_settings_invalid"):
        run_chrono_dem_cuda_request(request)


def test_chrono_dem_worker_rejects_non_single_cuda_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_probe(tmp_path, cuda_device_count=2)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    with pytest.raises(MeasurementAdapterExecutionError, match="cuda_device_count_invalid"):
        run_chrono_dem_cuda_request(_request())


def test_chrono_dem_worker_rejects_probe_parameter_binding_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_probe(tmp_path, density_override=9.0)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    with pytest.raises(MeasurementAdapterExecutionError, match="density_g_cm3_binding_mismatch"):
        run_chrono_dem_cuda_request(_request())
