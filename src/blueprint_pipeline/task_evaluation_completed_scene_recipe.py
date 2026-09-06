"""Bind completed assets to the installed six-stage construction contract."""
from __future__ import annotations

from . import task_evaluation_scene_configuration_submission_records as records
from .decision_evidence_contracts import canonical_digest
from .task_evaluation_scene_construction_recipe import validate_scene_construction_recipe

MESH_APPEARANCE_ADAPTER = "provided_mesh_appearance_excision"
SOURCE_AUTHORING_ADAPTER = "provided_mesh_rigid_authoring"


def stage_configurations(*, task: dict, collision_digest: str) -> list[dict]:
    subject, support = task["subject"], task["support"]
    lower, upper = subject["aabb_min_xyz_m"], subject["aabb_max_xyz_m"]
    center = [(a + b) / 2 for a, b in zip(lower, upper, strict=True)]
    # The existing ArtiFixer transport calls this key publisher_instance_id.
    # Its owner-object namespace is an internal compatibility identifier, not
    # a claimed InteriorGS label. Retain the actual provided prim separately.
    source_key = "owner-object-" + canonical_digest({
        "source": task["source_binding"]["sha256"], "prim": subject["source_object_id"]})[7:31]
    selection = {"publisher_instance_id": source_key, "publisher_label": subject["review_label"],
        "review_label": subject["review_label"], "aabb_min_xyz_m": lower,
        "aabb_max_xyz_m": upper, "center_xyz_m": center}
    first = records.stage_one_configuration(scene_id=task["scene_identity"]["id"],
        source_object=selection, support_label=support["label"], human_authority=task["human_authority"])
    first["appearance_authority"] = "owner-provided completed appearance candidate"
    first["source_origin"] = "owner_provided_completed_asset"
    first["source_object"].update(source_object_id=subject["source_object_id"],
        runtime_prim_path=subject["runtime_prim_path"], identity_basis="exact_owner_mesh_object")
    if task["appearance_kind"] == "other_observed":
        first = {"schema_version": "task_evaluation_provided_mesh_appearance_excision.v1",
            "source_origin": "owner_provided_completed_asset",
            "exact_target_prim": subject["runtime_prim_path"],
            "collision_source_digest": collision_digest,
            "source_object": first["source_object"],
            "source_bytes_unchanged_required": True, "unobserved_surfaces_recovered": False,
            "physical_truth_claimed": False, "generated_appearance": False}
    second = records.stage_two_configuration(scene_id=task["scene_identity"]["id"],
        collision_source_digest=collision_digest, target_match={"prim_path": subject["runtime_prim_path"],
            "world_aabb_min_m": lower, "world_aabb_max_m": upper,
            "point_count": subject["point_count"], "face_count": subject["face_count"]},
        support_prim_path=support["runtime_prim_path"])
    second["claim_boundary"] = "Provided geometry is a collision candidate, not independent physical evidence."
    third = records.stage_three_configuration(scene_id=task["scene_identity"]["id"],
        replacement_identity=subject["identity"], source_instance_id=source_key,
        authoring_target=subject["review_label"], source_min=lower, source_max=upper,
        dimension_tolerance=0.0001, physics_bounds=subject["physics_bounds"])
    third.update(authoring_method="exact_provided_mesh_with_bounded_simulation_physics",
        source_object_identity=subject["source_object_id"],
        geometry_support="exact_provided_mesh_in_owner_declared_frame",
        appearance_inputs="provided_mesh_visual_geometry_preserved")
    assembly = records.stage_six_configuration(scene_identity=task["scene_identity"],
        support_plane={"sage_prim_path": support["runtime_prim_path"],
            "publisher_instance_id": "owner-support-" + canonical_digest(support)[7:31],
            "top_z_m": support["aabb_max_xyz_m"][2],
            "bounds_min_xyz_m": support["aabb_min_xyz_m"], "bounds_max_xyz_m": support["aabb_max_xyz_m"]},
        start_center=center, bottom_z=lower[2])
    assembly["appearance"]["source"] = "owner_provided_completed_asset"
    assembly["collision"]["source"] = "normalized_owner_provided_mesh"
    assembly["support_plane"].update(source_object_id=support["source_object_id"],
        physical_scale_measured=False, authority="owner_declared_asset_frame")
    return [first, second, third,
        records.stage_four_configuration(replacement_identity=subject["identity"], dimension_tolerance=0.0001),
        records.stage_five_configuration(replacement_identity=subject["identity"]), assembly]


def construction_recipe(*, run_id: str, task: dict, source_manifest_digest: str,
                        rights_admission_digest: str, configurations: list[dict],
                        supplemental_destination: dict) -> dict:
    value = records.recipe(recipe_id=run_id + "-recipe", team_namespace=task["team_namespace"],
        scene_identity=task["scene_identity"], task_identity=task["task_identity"],
        subject_identity=task["subject"]["identity"], output_identity=task["output_identity"],
        source_manifest_digest=source_manifest_digest, rights_admission_digest=rights_admission_digest,
        stage_configuration_references=configurations, supplemental_destination=supplemental_destination)
    if task["appearance_kind"] == "other_observed":
        value["stage_sequence"][0].update(adapter={"id": MESH_APPEARANCE_ADAPTER, "version": "v1"},
                                         execution_class="no_spend")
    value["stage_sequence"][2].update(adapter={"id": SOURCE_AUTHORING_ADAPTER, "version": "v1"},
                                     execution_class="no_spend")
    value["recipe_digest"] = canonical_digest(value, digest_field="recipe_digest")
    return validate_scene_construction_recipe(value)
