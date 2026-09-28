from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.scene_object_discovery_contract import (
    SceneObjectDiscoveryContractError,
    scene_object_discovery_request_digest,
    validate_scene_object_discovery_request,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _ref(character: str) -> dict[str, object]:
    return {
        "uri": f"https://objects.example/{character}.json",
        "digest": _digest(character),
        "size_bytes": 100,
    }


def request() -> dict[str, object]:
    return {
        "schema_version": "scene_object_discovery_request.v1",
        "discovery_id": "discover-scene-001",
        "expected_production_commit": "a" * 40,
        "team_namespace": "robot-team-001",
        "scene": {
            "identity": {"id": "scene-001", "version": "v1"},
            "source_splat": _ref("a"),
            "scene_analysis": _ref("b"),
            "metric_registration": _ref("c"),
            "renderer_qualification": _ref("d"),
            "retained_gaussian_count": 1234,
        },
        "task": {
            "kind": "rigid_relocation",
            "strategy": "pick_and_place",
            "task_statement": "Pick the red tote",
            "target_hint": "red tote",
        },
        "analysis": {
            "analyzers": ["splat_analyzer", "sam31"],
            "prompts": ["red tote", "container"],
            "minimum_confidence": 0.5,
            "minimum_task_relevance": 0.5,
            "require_metric_source_object": True,
            "full_scene_survey_required": True,
        },
        "rights": {
            "admission": _ref("e"),
            "human_authority_record": _ref("f"),
            "source_bytes_redistributable": False,
            "provider_disclosure_scope": "derived_only",
        },
        "execution": {"mode": "qualified_local_runtime"},
        "publication": {
            "input_namespace": "scene-001-discovery",
            "service_account_readback_required": True,
        },
    }


def test_discovery_request_validates_and_is_digest_bound() -> None:
    value = request()
    assert validate_scene_object_discovery_request(value) == value
    assert scene_object_discovery_request_digest(value).startswith("sha256:")


def test_provider_execution_requires_source_disclosure_authority() -> None:
    value = copy.deepcopy(request())
    value["execution"] = {
        "mode": "provider_gpu_after_activation",
        "selected_provider": "vast",
    }
    with pytest.raises(SceneObjectDiscoveryContractError) as exc:
        validate_scene_object_discovery_request(value)
    assert str(exc.value) == "scene_object_discovery_provider_source_disclosure_not_authorized"


def test_duplicate_analyzers_and_wrong_strategy_fail_closed() -> None:
    duplicate = copy.deepcopy(request())
    duplicate["analysis"]["analyzers"] = ["sam31", "sam31"]
    with pytest.raises(SceneObjectDiscoveryContractError) as exc:
        validate_scene_object_discovery_request(duplicate)
    assert str(exc.value) == "scene_object_discovery_analyzers_duplicate"

    wrong = copy.deepcopy(request())
    wrong["task"]["strategy"] = "articulated_open_close"
    with pytest.raises(SceneObjectDiscoveryContractError) as exc:
        validate_scene_object_discovery_request(wrong)
    assert str(exc.value) == "scene_object_discovery_task_strategy_kind_mismatch"
