from blueprint_pipeline.unitree_groot_n17_sonic_runpod_image_contract import (
    resolve_runpod_provider_shape,
)


def _resolve() -> dict:
    return resolve_runpod_provider_shape(
        gpu_type_ids=(),
        default_gpu_type_ids=("NVIDIA L40S",),
        container_disk_gb=None,
        volume_gb=None,
        allowed_cuda_versions=("12.8",),
    )


def test_shape_defaults_when_legacy_size_env_vars_are_empty(monkeypatch) -> None:
    monkeypatch.setenv("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_DISK_GB", "")
    monkeypatch.setenv("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_VOLUME_GB", "")

    shape = _resolve()

    assert shape["container_disk_gb"] == 240
    assert shape["volume_gb"] == 120


def test_shape_defaults_when_legacy_size_env_vars_are_malformed(monkeypatch) -> None:
    monkeypatch.setenv(
        "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_DISK_GB", "not-a-number"
    )
    monkeypatch.setenv("BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_VOLUME_GB", "12.5")

    shape = _resolve()

    assert shape["container_disk_gb"] == 240
    assert shape["volume_gb"] == 120


def test_explicit_admitted_shape_overrides_malformed_legacy_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_DISK_GB", "not-a-number"
    )
    monkeypatch.setenv(
        "BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_VOLUME_GB", "not-a-number"
    )

    shape = resolve_runpod_provider_shape(
        gpu_type_ids=("NVIDIA A40",),
        default_gpu_type_ids=("NVIDIA L40S",),
        container_disk_gb=80,
        volume_gb=50,
        allowed_cuda_versions=("12.8",),
    )

    assert shape["container_disk_gb"] == 80
    assert shape["volume_gb"] == 50
