"""Exact-scene, development-only cabinet prior; never a measured source repair."""

from __future__ import annotations

from typing import Any, Mapping

PRIOR = {
    "schema_version": "website_drawer_depth_prior.v1",
    "claim_ceiling": "development_only",
    "scene_id": "site-capture-7d655e25-34f2-4961-ae9c-6da11621ce8f-development",
    "preparation_digest": "sha256:cfec5e38e056748c73d2d93660aae81c1f703e81f6bf8041dd88a3914d64cdb6",
    "observation_manifest_digest": "sha256:028cc3a1ff6c0093d13850859e3220bede6b2ef919a0a985570d858b56e91681",
    "subject_identity": {"id": "website-subject-0300250645e1def96386", "version": "v1"},
    "original_frame_sha256s": [
        "sha256:0205303c4faa9892114349af39bc5908b07c145dcc58eaeedd9b5fb74e4336af",
        "sha256:7c711762f672a41fd90803b4c5c658a81ee23e9dc3b0fcc71dd57ed243bfcc4a",
        "sha256:afba0d20718721e00312255a4be823a42af42f36e6640598008eee2e2bbe01f9",
        "sha256:aba7b9e216a351510bf07b16bc8362ec0a68306c94306bbf705f60a0e8874157",
        "sha256:b6f03d21cf7d274c7cc4228fca289e242e9eeb347f97986c4af39d2e00812a25",
    ],
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

# Each entry is bound to its own scene, preparation, observed frames and object
# identity. A later capture of the same video still needs a new entry: the old
# capture's identity and authority cannot be borrowed for the new run.
BC15_PRIOR = {
    # This capture contains the same source video as PRIOR, so the observed
    # original-frame hashes and public comparison examples are the same. Its
    # preparation and observation receipts, object identity and scene are new.
    **PRIOR,
    "scene_id": "site-capture-bc15f409-09b7-438c-9891-519ba24d728f-development",
    "preparation_digest": "sha256:2e1254a642a9a3a77de7f4ae421cc59e68ca8ef8b75da6d14617207afaba1f59",
    "observation_manifest_digest": "sha256:5d780038e8d92460645b15bccf2146a3dcc2436dd230ea1b236dfdb07a494e73",
    "subject_identity": {"id": "website-subject-cc9b93603a73eb7cbace", "version": "v1"},
    "original_frame_sha256s": [
        "sha256:0205303c4faa9892114349af39bc5908b07c145dcc58eaeedd9b5fb74e4336af",
        "sha256:7c711762f672a41fd90803b4c5c658a81ee23e9dc3b0fcc71dd57ed243bfcc4a",
        "sha256:afba0d20718721e00312255a4be823a42af42f36e6640598008eee2e2bbe01f9",
        "sha256:aba7b9e216a351510bf07b16bc8362ec0a68306c94306bbf705f60a0e8874157",
        "sha256:b6f03d21cf7d274c7cc4228fca289e242e9eeb347f97986c4af39d2e00812a25",
    ],
}

ADDITIONAL_PRIORS: tuple[Mapping[str, Any], ...] = (BC15_PRIOR,)


def prior_for(*, scene_id: str, subject_identity: Mapping[str, Any]) -> Mapping[str, Any] | None:
    matches = [prior for prior in (PRIOR, *ADDITIONAL_PRIORS)
               if prior.get("scene_id") == scene_id
               and prior.get("subject_identity") == subject_identity]
    if len(matches) > 1:
        raise ValueError("website_drawer_depth_prior_ambiguous")
    return matches[0] if matches else None
