from __future__ import annotations

from blueprint_pipeline.cpu_env_doctor import CPU_ENV_MODULES, check_cpu_env


def test_cpu_env_doctor_reports_shape() -> None:
    report = check_cpu_env(["json", "definitely_missing_blueprint_module"])

    assert report["schema_version"] == "cpu_env_doctor.v1"
    assert report["sys_executable"]
    assert report["sys_version"]
    assert report["modules"]["json"]["present"] is True
    assert report["modules"]["definitely_missing_blueprint_module"]["present"] is False
    assert "definitely_missing_blueprint_module" in report["missing"]
    assert report["ok"] is False


def test_canonical_cpu_env_has_no_skip_stack() -> None:
    report = check_cpu_env(CPU_ENV_MODULES)

    assert report["missing"] == []
    for module in ("PIL", "pxr", "mujoco", "trimesh", "boto3", "botocore"):
        assert report["modules"][module]["present"] is True
