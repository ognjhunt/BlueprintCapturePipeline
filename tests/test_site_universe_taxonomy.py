"""taxonomy.json schema, data-driven category mapping and bounded Overpass queries."""

import copy
import json
import re
import urllib.parse

import pytest

from tools.site_universe import taxonomy
from tools.site_universe.adapters import osm_overpass
from tools.site_universe.taxonomy import TaxonomyError

DOCUMENT = json.loads(taxonomy.TAXONOMY_PATH.read_text(encoding="utf-8"))
# The owner asked for at least these rows to be active in the first Texas build (2026-10-04).
REQUIRED_ACTIVE_CAPABILITIES = {
    "fixed_arm_machine_tending", "kitting_assembly", "mobile_manipulator_case_picking",
    "palletizing_depalletizing", "bimanual_folding", "recycling_sorting", "food_prep_manipulation",
    "shelf_restocking", "hospital_logistics",
}


def doc():
    return copy.deepcopy(DOCUMENT)


def test_taxonomy_file_satisfies_the_schema():
    loaded = taxonomy.load()
    assert DOCUMENT["schema"] == "blueprint.site_universe.taxonomy.v1"
    assert re.match(r"^\d{4}-\d{2}-\d{2}\.\d+$", loaded.version)
    assert loaded.sha256 == taxonomy.file_sha256()
    active = {row["capability"] for row in loaded.active_rows}
    assert active >= REQUIRED_ACTIVE_CAPABILITIES
    for row in DOCUMENT["rows"]:
        assert row["status"] in ("active", "candidate")
        assert re.match(r"^\d{4}-\d{2}-\d{2}$", row["added_at"])
        assert row["task_families"] and row["site_types"]
        if row["status"] == "active":
            assert row["evidence"], row["row_id"]
        for item in row["evidence"]:
            assert item["url"].startswith("https://") and item["claim"] and item["verified_at"]
    names = [site_type["id"] for site_type in DOCUMENT["site_types"]]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    ("code", "system", "site_type", "category"),
    [
        ("493120", "naics", "warehouse_cold_storage", "warehousing_storage"),
        ("493110", "naics", "warehouse_general", "warehousing_storage"),
        ("424410", "naics", "wholesale_grocery_food", "distribution_wholesale"),
        ("423830", "naics", "wholesale_distribution", "distribution_wholesale"),
        ("311615", "naics", "plant_meat_poultry_egg", "food_processing"),
        ("311991", "naics", "plant_food", "food_processing"),
        ("332710", "naics", "plant_fabricated_metal", "manufacturing"),
        ("336111", "naics", "plant_transportation_equipment", "manufacturing"),
        ("812332", "naics", "industrial_laundry", "industrial_laundry"),
        ("562920", "naics", "recycling_mrf", "recycling_waste"),
        ("622110", "naics", "hospital", "healthcare_hospital"),
        ("455211", "naics", "warehouse_club", "retail_general"),
        ("3599", "sic", "plant_machinery", "manufacturing"),
        ("2015", "sic", "plant_meat_poultry_egg", "food_processing"),
    ],
)
def test_codes_map_to_site_types_and_categories_from_data(code, system, site_type, category):
    loaded = taxonomy.load()
    result = loaded.classify_codes([code], system=system, primary=code)
    assert result.primary == site_type
    assert loaded.category(result.primary) == category
    assert result.matched_by == {site_type: (f"{system}:{code}",)}


@pytest.mark.parametrize(
    ("codes", "reason"),
    [
        (["424710"], "no_matching_code"),  # petroleum bulk stations are excluded from wholesale
        (["324110"], "no_matching_code"),  # refineries are not a site type
        (["445131"], "inactive_site_type"),  # convenience stores are a candidate row only
        (["423930"], "inactive_site_type"),  # scrap yards are a candidate row only
    ],
)
def test_codes_outside_active_rows_are_dropped(codes, reason):
    assert taxonomy.load().classify_codes(codes).drop_reason == reason


def test_osm_classification_rules():
    loaded = taxonomy.load()
    plant = loaded.classify_osm({"building": "industrial", "man_made": "works"}, "Ironclad Foundry")
    assert plant.site_types == ("plant_general",)
    generic = loaded.classify_osm({"landuse": "industrial"}, "Westside Industrial Park")
    assert generic.site_types == ("industrial_general",)
    assert loaded.classify_osm({"landuse": "industrial", "industrial": "gas"}, "Quillmoor Gas Plant").drop_reason == "excluded_tag"
    assert loaded.classify_osm({"landuse": "industrial"}, "Quillmoor Pump").drop_reason == "excluded_name"
    assert loaded.classify_osm({"building": "warehouse"}, "Public Storage").drop_reason == "excluded_name"
    assert loaded.classify_osm({"shop": "convenience"}, "Quick Mart").drop_reason == "inactive_site_type"
    multi = loaded.classify_osm({"shop": "supermarket", "building": "warehouse"}, "Fresh Market")
    assert set(multi.site_types) == {"grocery_store", "warehouse_general"}


def test_category_mapping_is_data_not_code():
    changed = doc()
    for site_type in changed["site_types"]:
        if site_type["id"] == "warehouse_general":
            site_type["category"] = "distribution_wholesale"
    loaded = taxonomy.Taxonomy(changed)
    assert loaded.category(loaded.classify_codes(["493110"]).primary) == "distribution_wholesale"


def test_adding_an_active_row_is_a_data_change():
    changed = doc()
    for row in changed["rows"]:
        if row["row_id"] == "hotel_delivery":
            row["status"] = "active"
    loaded = taxonomy.Taxonomy(changed)
    assert loaded.classify_codes(["721110"]).primary == "hotel"
    assert loaded.matches(["hotel"])["rows"] == ["hotel_delivery"]
    assert any('"tourism"="hotel"' in query for _, query in loaded.overpass_queries("TX"))


def test_overpass_queries_are_bounded_named_and_stable():
    loaded = taxonomy.load()
    queries = loaded.overpass_queries("TX")
    assert 1 <= len(queries) <= taxonomy.MAX_OVERPASS_QUERIES_PER_STATE
    assert queries == loaded.overpass_queries("tx")
    for group, query in queries:
        assert group in ("buildings", "industrial_areas", "retail", "amenities")
        assert query.startswith("[out:json][timeout:300];")
        assert 'area["ISO3166-2"="US-TX"]' in query and query.endswith("out geom;")
        assert " meta" not in query
        clauses = [line for line in query.splitlines() if line.strip().startswith("nwr")]
        assert clauses and all('["name"](area.state);' in line for line in clauses)
        url = osm_overpass.query_url(query)
        assert urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["data"] == [query]
    with pytest.raises(TaxonomyError):
        loaded.overpass_queries("ZZ")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["site_types"][0]["selectors"].update(osm=[{"building": "house"}]),
        lambda d: d["site_types"][0]["selectors"].update(osm=[{"amenity": "doctors"}]),
        lambda d: d["site_types"][0]["selectors"].update(osm=[{"building": "yes"}]),
        lambda d: d["site_types"][0]["selectors"].update(osm=[{"name": "acme"}]),
        lambda d: d["rows"][0].update(site_types=["no_such_site_type"]),
        lambda d: d["rows"][0].update(evidence=[]),
        lambda d: d["rows"][0].update(status="maybe"),
        lambda d: d["rows"][0]["evidence"][0].update(url="http://insecure.example"),
        lambda d: d["site_types"][1]["selectors"].update(naics_prefixes=["49312"]),
        lambda d: d["site_types"][0]["selectors"].update(naics_prefixes=["49x"]),
        lambda d: d.update(schema="other"),
    ],
)
def test_invalid_taxonomy_documents_are_refused(mutate):
    changed = doc()
    mutate(changed)
    with pytest.raises(TaxonomyError):
        taxonomy.Taxonomy(changed)


def test_unreferenced_site_types_are_refused():
    changed = doc()
    changed["site_types"].append({
        "id": "orphan", "category": "manufacturing", "label": "Orphan",
        "selectors": {"naics_prefixes": ["3999"]},
    })
    with pytest.raises(TaxonomyError, match="not used by any row"):
        taxonomy.Taxonomy(changed)
