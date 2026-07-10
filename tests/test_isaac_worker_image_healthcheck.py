from pathlib import Path

from blueprint_pipeline.isaac_worker_image_healthcheck import run_image_healthcheck


def test_static_image_healthcheck_passes_required_worker_contract() -> None:
    result = run_image_healthcheck(
        build_time=True,
        env={
            "BLUEPRINT_WORKER_IMAGE_FAMILY": "isaac-eval-worker",
            "BLUEPRINT_ISAAC_PYTHON": "/isaac-sim/python.sh",
            "BLUEPRINT_ISAAC_UNITREE_G1_USD": "/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd",
        },
        exists=lambda path: path in {
            Path("/isaac-sim/python.sh"),
            Path("/isaac-sim/Isaac/Robots/Unitree/G1/g1.usd"),
        },
        importer=lambda name: object() if name == "blueprint_pipeline" else None,
    )
    assert result["status"] == "passed"
    assert result["blockers"] == []


def test_static_image_healthcheck_fails_closed_on_missing_g1_and_wrong_family() -> None:
    result = run_image_healthcheck(
        build_time=True,
        env={"BLUEPRINT_WORKER_IMAGE_FAMILY": "wrong"},
        exists=lambda _path: False,
        importer=lambda _name: (_ for _ in ()).throw(ImportError("missing")),
    )
    assert result["status"] == "blocked"
    assert set(result["blockers"]) == {
        "isaac_python_missing",
        "unitree_g1_usd_missing",
        "blueprint_pipeline_import_failed",
        "isaac_worker_image_family_invalid",
    }
