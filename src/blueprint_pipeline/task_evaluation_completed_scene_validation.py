"""Pure configuration checks for observed completed-scene meshes."""
from __future__ import annotations

def mesh_appearance_configuration_refusal(configuration, envelope):
    rows = [row for row in envelope.get("materialized_references", [])
            if row.get("contract_path") == "scene.geometry.collision"]
    first = ((envelope.get("recipe") or {}).get("stage_sequence") or [{}])[0]
    if (configuration.get("schema_version") != "task_evaluation_provided_mesh_appearance_excision.v1"
            or first.get("adapter", {}).get("id") != "provided_mesh_appearance_excision"
            or first.get("execution_class") != "no_spend"
            or envelope.get("request", {}).get("scene", {}).get("appearance", {}).get("kind") != "other_observed"
            or configuration.get("source_origin") != "owner_provided_completed_asset"
            or configuration.get("source_bytes_unchanged_required") is not True
            or configuration.get("unobserved_surfaces_recovered") is not False
            or configuration.get("physical_truth_claimed") is not False
            or configuration.get("generated_appearance") is not False
            or len(rows) != 1
            or configuration.get("collision_source_digest") != rows[0].get("digest")
            or not str(configuration.get("exact_target_prim", "")).startswith("/Root/")):
        return "provided_mesh_source_binding"
    return None
