from __future__ import annotations

from copy import deepcopy

import pytest

from blueprint_pipeline.native_task_entity_contract import (
    NativeTaskEntityContractError,
    TASK_KIND_ARTICULATED_OPEN_CLOSE,
    TASK_KIND_DEFORMABLE_TRANSFER,
    TASK_KIND_RIGID_PICK_PLACE,
    materialize_native_task_entity_contract,
    project_legacy_task_object_entity,
)


_PHYSICS_BY_ROLE = {
    "movable_rigid": "rigid_body",
    "articulated_fixture": "articulation",
    "movable_deformable": "deformable_volume",
    "destination_receptacle": "static_collider",
    "support_surface": "static_collider",
    "obstacle": "static_collider",
    "robot": "robot_articulation",
}

_RESET_BY_PHYSICS = {
    "rigid_body": "native_rigid_state",
    "articulation": "native_articulation_state",
    "deformable_volume": "native_deformable_state",
    "static_collider": "immutable_scene_state",
    "robot_articulation": "native_robot_state",
}

_CONTACT_BY_ROLE = {
    "movable_rigid": "manipulated_rigid",
    "articulated_fixture": "manipulated_articulation",
    "movable_deformable": "manipulated_deformable",
    "destination_receptacle": "destination_volume",
    "support_surface": "supporting_surface",
    "obstacle": "collision_obstacle",
    "robot": "manipulator",
}

_SCORING_BY_ROLE = {
    "movable_rigid": "movable_target",
    "articulated_fixture": "articulated_target",
    "movable_deformable": "deformable_target",
    "destination_receptacle": "destination",
    "support_surface": "support_context",
    "obstacle": "collision_context",
    "robot": "robot_context",
}


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _sha_offset(character: str, offset: int) -> str:
    alphabet = "0123456789abcdef"
    return _sha(alphabet[(alphabet.index(character) + offset) % len(alphabet)])


def _entity(
    entity_id: str,
    semantic_role: str,
    *,
    legacy_task_object: bool = False,
    inserted: bool = False,
    replaced: bool = False,
    digest_character: str = "a",
) -> dict:
    physics_type = _PHYSICS_BY_ROLE[semantic_role]
    source_digest = _sha(digest_character)
    asset_digest = _sha_offset(digest_character, 1)
    state_digest = _sha_offset(digest_character, 2)
    configuration_digest = _sha_offset(digest_character, 3)

    if replaced:
        removal_policy = {
            "source_entity_action": "remove",
            "gaussian_action": "delete_owned",
            "collider_action": "delete_subtree",
            "receipt_sha256": _sha("e"),
        }
        replacement_policy = {
            "action": "insert_runtime_asset",
            "replacement_required": True,
            "receipt_sha256": _sha("f"),
        }
    elif inserted:
        removal_policy = {
            "source_entity_action": "not_present",
            "gaussian_action": "not_applicable",
            "collider_action": "not_applicable",
            "receipt_sha256": _sha("e"),
        }
        replacement_policy = {
            "action": "insert_runtime_asset",
            "replacement_required": True,
            "receipt_sha256": _sha("f"),
        }
    else:
        removal_policy = {
            "source_entity_action": "retain",
            "gaussian_action": "retain",
            "collider_action": "retain",
            "receipt_sha256": _sha("e"),
        }
        replacement_policy = {
            "action": "retain_registered_source",
            "replacement_required": False,
            "receipt_sha256": _sha("f"),
        }

    role = "task_object" if legacy_task_object else semantic_role
    source_kind = (
        "legacy_runtime_contract"
        if legacy_task_object
        else "runtime_embodiment"
        if semantic_role == "robot"
        else "observed_dataset_entity"
        if replaced
        else "registered_scene_geometry"
    )
    binding_kind = (
        "runtime_embodiment"
        if semantic_role == "robot"
        else "usd_asset"
        if inserted or replaced or legacy_task_object
        else "registered_scene_geometry"
    )
    disclosure_class = (
        "runtime_bundled"
        if semantic_role == "robot" or legacy_task_object
        else "restricted_private_processing"
    )

    return {
        "entity_id": entity_id,
        "name": entity_id,
        "semantic_role": role,
        "source_observation": {
            "observation_id": f"observation:{entity_id}",
            "source_kind": source_kind,
            "source_reference": f"sources/{entity_id}",
            "source_sha256": source_digest,
            "observed": not legacy_task_object and semantic_role != "robot",
        },
        "physics_type": physics_type,
        "runtime_asset": {
            "asset_id": f"asset:{entity_id}",
            "binding_kind": binding_kind,
            "source_reference": f"assets/{entity_id}.usd",
            "sha256": asset_digest,
        },
        "initial_state": {
            "pose_world": {
                "position_world_m": [0.1, 0.2, 0.3],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "state_sha256": state_digest,
            "settled_state_required": True,
            "initial_penetration_allowed": False,
        },
        "reset_method": {
            "kind": _RESET_BY_PHYSICS[physics_type],
            "state_id": f"reset:{entity_id}",
            "native_readback_required": True,
            "direct_state_write_after_episode_start_allowed": False,
        },
        "contact_role": {
            "kind": _CONTACT_BY_ROLE[semantic_role],
            "native_contact_readback_required": True,
        },
        "scoring_role": {
            "kind": _SCORING_BY_ROLE[semantic_role],
            "deterministic_state_readback_required": True,
            "policy_self_grading_allowed": False,
        },
        "removal_policy": removal_policy,
        "replacement_policy": replacement_policy,
        "provenance": {
            "source_id": f"source:{entity_id}",
            "source_revision": "fixture-revision-1",
            "source_path": f"fixture/{entity_id}",
            "source_size_bytes": 1024,
            "license_id": "fixture-license",
            "public_source_rights_id": "fixture-public-rights-receipt",
            "derived_processing_authority_id": "fixture-processing-authority",
            "provider_terms_id": "fixture-provider-terms",
            "output_rights_id": "fixture-output-rights",
            "attribution": "Hermetic test fixture",
            "disclosure_class": disclosure_class,
            "upload_permitted": semantic_role == "robot" or legacy_task_object,
            "raw_redistribution_permitted": semantic_role == "robot",
            "provider_retention_permitted": False,
            "provider_training_permitted": False,
        },
        "digests": {
            "source_sha256": source_digest,
            "runtime_asset_sha256": asset_digest,
            "initial_state_sha256": state_digest,
            "configuration_sha256": configuration_digest,
        },
    }


def _robot() -> dict:
    return _entity("franka", "robot", inserted=True, digest_character="1")


def _legacy_rigid_pick_place_fixture() -> tuple[str, list[dict]]:
    return (
        TASK_KIND_RIGID_PICK_PLACE,
        [
            _entity(
                "840313_pick_object",
                "movable_rigid",
                legacy_task_object=True,
                inserted=True,
                digest_character="2",
            ),
            _robot(),
        ],
    )


def _legacy_articulated_refrigerator_fixture() -> tuple[str, list[dict]]:
    return (
        TASK_KIND_ARTICULATED_OPEN_CLOSE,
        [
            _entity(
                "840796_refrigerator",
                "articulated_fixture",
                legacy_task_object=True,
                inserted=True,
                digest_character="3",
            ),
            _robot(),
        ],
    )


def _deformable_receptacle_fixture() -> tuple[str, list[dict]]:
    return (
        TASK_KIND_DEFORMABLE_TRANSFER,
        [
            _entity(
                "cloth",
                "movable_deformable",
                replaced=True,
                digest_character="4",
            ),
            _entity(
                "basket",
                "destination_receptacle",
                inserted=True,
                digest_character="5",
            ),
            _entity("counter", "support_surface", digest_character="6"),
            _entity("wall", "obstacle", digest_character="7"),
            _entity("chair", "obstacle", digest_character="8"),
            _robot(),
        ],
    )


@pytest.mark.parametrize(
    ("fixture", "expected_roles", "legacy_projection_count"),
    [
        (
            _legacy_rigid_pick_place_fixture,
            {"movable_rigid", "robot"},
            1,
        ),
        (
            _legacy_articulated_refrigerator_fixture,
            {"articulated_fixture", "robot"},
            1,
        ),
        (
            _deformable_receptacle_fixture,
            {
                "movable_deformable",
                "destination_receptacle",
                "support_surface",
                "obstacle",
                "robot",
            },
            0,
        ),
    ],
)
def test_contract_passes_all_three_fixture_families(
    fixture,
    expected_roles: set[str],
    legacy_projection_count: int,
) -> None:
    task_kind, entities = fixture()

    contract = materialize_native_task_entity_contract(
        task_kind=task_kind, task_entities=entities
    )

    assert set(contract["semantic_role_index"]) == expected_roles
    assert contract["legacy_task_object_projection_count"] == legacy_projection_count
    assert contract["contract_digest"].startswith("sha256:")
    assert all(
        entity["semantic_role"] != "task_object"
        for entity in contract["task_entities"]
    )


def test_repeated_semantic_roles_are_indexed_by_unique_entity_id() -> None:
    task_kind, entities = _deformable_receptacle_fixture()

    forward = materialize_native_task_entity_contract(
        task_kind=task_kind, task_entities=entities
    )
    reverse = materialize_native_task_entity_contract(
        task_kind=task_kind, task_entities=list(reversed(entities))
    )

    assert forward["semantic_role_index"]["obstacle"] == ["chair", "wall"]
    assert forward == reverse


def test_duplicate_entity_ids_fail_closed_even_when_roles_differ() -> None:
    task_kind, entities = _deformable_receptacle_fixture()
    entities[1]["entity_id"] = "cloth"

    with pytest.raises(NativeTaskEntityContractError) as exc_info:
        materialize_native_task_entity_contract(
            task_kind=task_kind, task_entities=entities
        )

    assert "native_task_entity_id_duplicate:cloth" in exc_info.value.errors


def test_missing_required_role_fails_closed() -> None:
    task_kind, entities = _deformable_receptacle_fixture()
    entities = [
        entity
        for entity in entities
        if entity["semantic_role"] != "destination_receptacle"
    ]

    with pytest.raises(NativeTaskEntityContractError) as exc_info:
        materialize_native_task_entity_contract(
            task_kind=task_kind, task_entities=entities
        )

    assert (
        "native_task_entity_role_missing:destination_receptacle"
        in exc_info.value.errors
    )


@pytest.mark.parametrize(
    ("field", "expected_error"),
    [
        ("entity_id", "native_task_entity_id_invalid:0"),
        ("semantic_role", "native_task_entity_semantic_role_invalid:cloth"),
        ("source_observation", "native_task_entity_source_observation_invalid:cloth"),
        ("physics_type", "native_task_entity_physics_type_invalid:cloth"),
        ("runtime_asset", "native_task_entity_runtime_asset_invalid:cloth"),
        ("initial_state", "native_task_entity_initial_state_invalid:cloth"),
        ("reset_method", "native_task_entity_reset_method_invalid:cloth"),
        ("contact_role", "native_task_entity_contact_role_invalid:cloth"),
        ("scoring_role", "native_task_entity_scoring_role_invalid:cloth"),
        ("removal_policy", "native_task_entity_removal_policy_invalid:cloth"),
        (
            "replacement_policy",
            "native_task_entity_replacement_policy_invalid:cloth",
        ),
        ("provenance", "native_task_entity_provenance_invalid:cloth"),
        ("digests", "native_task_entity_digests_invalid:cloth"),
    ],
)
def test_every_required_entity_field_fails_closed(
    field: str, expected_error: str
) -> None:
    task_kind, entities = _deformable_receptacle_fixture()
    entities[0].pop(field)

    with pytest.raises(NativeTaskEntityContractError) as exc_info:
        materialize_native_task_entity_contract(
            task_kind=task_kind, task_entities=entities
        )

    assert expected_error in exc_info.value.errors


def _mutate_bad_digest(entity: dict) -> None:
    entity["digests"]["runtime_asset_sha256"] = "not-a-digest"


def _mutate_bad_pose(entity: dict) -> None:
    entity["initial_state"]["pose_world"]["orientation_xyzw"] = [0.0, 0.0, 0.0, 2.0]


def _mutate_bad_reset(entity: dict) -> None:
    entity["reset_method"]["direct_state_write_after_episode_start_allowed"] = True


def _mutate_bad_contact(entity: dict) -> None:
    entity["contact_role"]["kind"] = "manipulated_rigid"


def _mutate_bad_scoring(entity: dict) -> None:
    entity["scoring_role"]["policy_self_grading_allowed"] = True


def _mutate_bad_removal(entity: dict) -> None:
    entity["removal_policy"]["source_entity_action"] = "retain"


def _mutate_bad_replacement(entity: dict) -> None:
    entity["replacement_policy"]["replacement_required"] = False


def _mutate_bad_provenance(entity: dict) -> None:
    entity["provenance"]["source_size_bytes"] = 0


@pytest.mark.parametrize(
    ("mutator", "expected_error"),
    [
        (_mutate_bad_digest, "native_task_entity_digests_invalid:cloth"),
        (_mutate_bad_pose, "native_task_entity_initial_state_pose_invalid:cloth"),
        (_mutate_bad_reset, "native_task_entity_reset_method_invalid:cloth"),
        (_mutate_bad_contact, "native_task_entity_contact_role_invalid:cloth"),
        (_mutate_bad_scoring, "native_task_entity_scoring_role_invalid:cloth"),
        (_mutate_bad_removal, "native_task_entity_removal_policy_invalid:cloth"),
        (
            _mutate_bad_replacement,
            "native_task_entity_replacement_policy_invalid:cloth",
        ),
        (_mutate_bad_provenance, "native_task_entity_provenance_invalid:cloth"),
    ],
)
def test_invalid_entity_subcontracts_fail_closed(mutator, expected_error: str) -> None:
    task_kind, entities = _deformable_receptacle_fixture()
    mutator(entities[0])

    with pytest.raises(NativeTaskEntityContractError) as exc_info:
        materialize_native_task_entity_contract(
            task_kind=task_kind, task_entities=entities
        )

    assert expected_error in exc_info.value.errors


def test_legacy_projection_maps_only_identity_and_role() -> None:
    _, entities = _legacy_articulated_refrigerator_fixture()
    legacy = entities[0]
    legacy.pop("entity_id")

    projected = project_legacy_task_object_entity(
        task_kind=TASK_KIND_ARTICULATED_OPEN_CLOSE,
        task_object=legacy,
    )

    assert projected["entity_id"] == "840796_refrigerator"
    assert projected["semantic_role"] == "articulated_fixture"
    assert projected["provenance"] == legacy["provenance"]


def test_legacy_projection_does_not_invent_missing_provenance() -> None:
    task_kind, entities = _legacy_rigid_pick_place_fixture()
    entities = deepcopy(entities)
    entities[0].pop("provenance")

    with pytest.raises(NativeTaskEntityContractError) as exc_info:
        materialize_native_task_entity_contract(
            task_kind=task_kind, task_entities=entities
        )

    assert "native_task_entity_provenance_invalid:840313_pick_object" in (
        exc_info.value.errors
    )
