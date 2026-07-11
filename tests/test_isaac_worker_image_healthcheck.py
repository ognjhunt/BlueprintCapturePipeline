from pathlib import Path

from blueprint_pipeline.isaac_worker_image_healthcheck import run_image_healthcheck


def test_static_image_healthcheck_passes_required_worker_contract() -> None:
    result = run_image_healthcheck(
        build_time=True,
        env={
            "BLUEPRINT_WORKER_IMAGE_FAMILY": "isaac-eval-worker",
            "BLUEPRINT_ISAAC_PYTHON": "/isaac-sim/python.sh",
            "ISAACSIM_ASSET_ROOT": "https://assets.example/Assets/Isaac/6.0",
            "BLUEPRINT_ISAAC_UNITREE_G1_USD": "Isaac/Robots/Unitree/G1/g1.usd",
            "BLUEPRINT_SIMULATOR_FRAMEWORK": "isaac_sim",
            "BLUEPRINT_ISAAC_SIM_MAJOR_VERSION": "6",
            "BLUEPRINT_SOURCE_COMMIT": "abc1234",
            "BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256": "a" * 64,
        },
        exists=lambda path: path in {
            Path("/isaac-sim/python.sh"),
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
        "unitree_g1_asset_binding_invalid",
        "blueprint_pipeline_import_failed",
        "isaac_worker_image_family_invalid",
        "isaac_worker_simulator_family_invalid",
        "isaac_worker_simulator_major_version_invalid",
        "isaac_worker_source_commit_invalid",
        "isaac_worker_source_dirty_patch_sha256_invalid",
    }


def test_static_image_healthcheck_treats_unreadable_isaac_path_as_blocked() -> None:
    def unreadable(_path: Path) -> bool:
        raise PermissionError("no traversal")

    result = run_image_healthcheck(
        build_time=False,
        env={
            "BLUEPRINT_WORKER_IMAGE_FAMILY": "isaac-eval-worker",
            "BLUEPRINT_SIMULATOR_FRAMEWORK": "isaac_sim",
            "BLUEPRINT_ISAAC_SIM_MAJOR_VERSION": "6",
            "BLUEPRINT_SOURCE_COMMIT": "abc1234",
            "BLUEPRINT_SOURCE_DIRTY_PATCH_SHA256": "a" * 64,
        },
        exists=unreadable,
        importer=lambda _name: object(),
    )

    assert result["status"] == "blocked"
    assert "isaac_python_missing" in result["blockers"]
