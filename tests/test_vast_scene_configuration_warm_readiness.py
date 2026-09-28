from blueprint_pipeline.vast_scene_configuration_warm_readiness import (
    ARTIFIXER_WARM_RUNTIME_READY_MARKER,
    observed_scene_configuration_warm_readiness,
    scene_configuration_warm_validation_fields,
)


def test_specialized_marker_is_distinct_and_satisfies_runtime_root() -> None:
    assert observed_scene_configuration_warm_readiness(
        ARTIFIXER_WARM_RUNTIME_READY_MARKER
    ) == (False, True)
    fields = scene_configuration_warm_validation_fields(
        {
            "provider_bundle_kind": "task_evaluation_scene_configuration",
            "provider_bundle_downloaded": True,
            "provider_entrypoint_started": True,
            "scene_configuration_warm_runtime_ready": False,
            "scene_configuration_artifixer_warm_runtime_ready": True,
        }
    )
    assert fields == {
        "scene_configuration_runtime_root_ready": True,
        "scene_configuration_artifixer_warm_runtime_ready": True,
    }


def test_runtime_root_refuses_marker_without_started_scene_runtime() -> None:
    assert scene_configuration_warm_validation_fields(
        {
            "provider_bundle_kind": "task_evaluation_scene_configuration",
            "provider_bundle_downloaded": True,
            "provider_entrypoint_started": False,
            "scene_configuration_artifixer_warm_runtime_ready": True,
        }
    ) == {
        "scene_configuration_runtime_root_ready": False,
        "scene_configuration_artifixer_warm_runtime_ready": True,
    }
