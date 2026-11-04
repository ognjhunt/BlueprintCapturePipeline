"""Development-only depth prior for a drawer cabinet or mobile pedestal; never a measured source repair.

A video often shows only the closed front of a drawer cabinet, so its visible
box is far too shallow to build from. Public specifications of comparable
office pedestals and drawer units give a broad depth and mass interval instead.

The prior is category knowledge, so it is general: every task subject the
articulated planner builds as a stacked drawer cabinet (``assembly_family``:
a declared ``stacked_drawer_cabinet``, or a legacy prismatic configuration)
gets it, in any scene. A mobile pedestal is one of those. A hinged door, any
other family and every rigid subject get ``None``.

Each record is bound when it is made to the identity its caller supplies: the
configuration's scene and subject, and the preparation, observation manifest and
observed frames the caller checked. No identity is ever a constant, so a record
always says which scene and observations it was applied to and cannot carry
another capture's identity. A field the caller does not know stays ``None``.

The record keeps the shape, values and ``schema_version`` of the exact-scene
records it replaces (``website_drawer_depth_prior.v1``). Retained depth
hypotheses carry that version and the category values, and nothing digests the
record itself, so they still validate and a hypothesis re-derived from the same
inputs is byte-identical.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = "website_drawer_depth_prior.v1"

#: Public comparisons for an office drawer cabinet or mobile pedestal, retrieved 2026-09-23.
CABINET_PRIOR: Mapping[str, Any] = {
    "nominal_depth_m": 0.55,
    "depth_interval_m": [0.45, 0.65],
    "whole_assembly_mass_interval_kg": [15.0, 50.0],
    "revised_part_mass_bounds_kg": {"carcass": [4.0, 30.0], "drawer": [0.5, 9.0]},
    "reference_retrieved_date": "2026-09-23",
    "example_depth_range_m": [0.50, 0.610],
    "example_width_range_m": [0.30, 0.42],
    "example_height_range_m": [0.53, 0.75],
    "example_weight_range_kg_approx": [18.0, 34.0],
    "manufacturer_examples": [
        {"model": "Steelcase Edvi mobile pedestal", "url": "https://shop.steelcase.com/products/edvi-storage-copy",
         "depth_m": 0.50, "width_m": 0.33, "height_m": 0.53},
        {"model": "Herman Miller Kumi pedestal", "url": "https://ukstore.hermanmiller.com/pages/product-details-kumi-pedestal",
         "depth_m": 0.565, "width_m": [0.30, 0.42], "height_m": 0.567, "weight_kg_lower": 20.0},
        {"model": "IKEA MICKE three-drawer unit", "url": "https://www.ikea.com/us/en/p/micke-drawer-unit-drop-file-storage-white-50213080/",
         "depth_m": 0.50, "width_m": 0.35, "height_m": 0.75, "packed_weight_kg_approx": 18.0},
        {"model": "Global wood veneer three-drawer mobile BBF", "url": "https://admin.globalfurnituregroup.com/storage/96152/Wood_Veneer_Price_List_01_23_26.pdf",
         "depth_m": [0.508, 0.610], "width_m": 0.394, "height_m": 0.699,
         "weight_kg": [31.8, 34.0]},
    ],
    "limitations": "Comparison models differ from the filmed cabinet; rear geometry and physical mass remain unmeasured.",
}


def is_drawer_cabinet(configuration: Mapping[str, Any]) -> bool:
    """Whether the planner builds this task subject as a stacked drawer cabinet."""
    from .task_object_articulated_packaging import DRAWER_FAMILY, AssetAuthoringError, assembly_family

    try:
        return assembly_family(configuration) == DRAWER_FAMILY
    except (AssetAuthoringError, AttributeError, TypeError):
        return False


def prior_for(configuration: Mapping[str, Any], *,
              preparation_digest: str | None = None,
              observation_manifest_digest: str | None = None,
              original_frame_sha256s: Iterable[str] = ()) -> dict[str, Any] | None:
    """The cabinet prior bound to this configuration's own scene and subject, or ``None``.

    ``None`` for a subject that is not a drawer cabinet, or for a configuration
    without a scene id and subject identity to bind the record to.
    """
    scene_id = configuration.get("scene_id")
    subject = configuration.get("replacement_identity")
    if (not isinstance(scene_id, str) or not scene_id
            or not isinstance(subject, Mapping) or not isinstance(subject.get("id"), str) or not subject["id"]
            or not is_drawer_cabinet(configuration)):
        return None
    values = copy.deepcopy(dict(CABINET_PRIOR))
    limitations = values.pop("limitations")
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_ceiling": "development_only",
        "scene_id": scene_id,
        "preparation_digest": preparation_digest,
        "observation_manifest_digest": observation_manifest_digest,
        "subject_identity": copy.deepcopy(dict(subject)),
        "original_frame_sha256s": sorted({str(digest) for digest in original_frame_sha256s}),
        **values,
        "limitations": limitations,
    }
