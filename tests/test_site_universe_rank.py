"""Ranking v1: config contract, size curve, exclusions, determinism, manifest and CLI (hermetic)."""

import copy
import gzip
import hashlib
import json

import pytest

from tests.site_universe_fixture import seeded_cache
from tools.site_universe import SCHEMA_VERSION, build, rank, taxonomy
from tools.site_universe.cli import main
from tools.site_universe.rank import RankError

TAXONOMY = taxonomy.load()
CONFIG = rank.load_config()
CONFIG_DOCUMENT = json.loads(rank.RANK_CONFIG_PATH.read_text(encoding="utf-8"))
PLANT = ("plant_transportation_equipment",)
WAREHOUSE = ("warehouse_general",)
GROCERY = ("grocery_store",)
HOSPITAL = ("hospital",)
FRS = ("epa_frs",)
OSHA_FRS = ("osha_ita", "epa_frs")


def make_site(key, name, *, site_types=("plant_fabricated_metal",), primary=None, employees=180,
              naics="332710", operator=None, city="DALLAS", street="100 MAIN ST", lat=32.78,
              lon=-96.80, sources=("osha_ita",), frs=None, frs_codes=(), osha_type="private",
              osha_year=2025, building_area=None, names=None):
    """A site shaped like a snapshot row, with only the fields the ranker reads plus ids."""
    primary = primary or site_types[0]
    records = []
    if "osha_ita" in sources:
        attributes = {"establishment_type": osha_type, "year_filing_for": osha_year}
        records.append({"attributes": attributes, "naics": naics, "source_id": "osha_ita",
                        "source_record_id": f"osha-{key}"})
    if "epa_frs" in sources:
        codes = sorted({code for code in (naics, *frs_codes) if code})
        records.append({"attributes": {"activity_status": frs or "unknown", "naics_codes": codes},
                        "naics": naics, "source_id": "epa_frs", "source_record_id": f"frs-{key}"})
    if "osm_overpass" in sources:
        records.append({"attributes": {}, "naics": None, "source_id": "osm_overpass",
                        "source_record_id": f"way/{key}"})
    matches = TAXONOMY.matches(site_types)
    matches.update(primary_site_type=primary, matched_by={})
    return {
        "attribution_required": "osm_overpass" in sources,
        "building_area_m2": building_area,
        "category": TAXONOMY.category(primary),
        "city": city,
        "employees": employees,
        "features": {
            "building_area_m2": building_area,
            "coordinate_precision": "precise" if lat is not None else None,
            "employees": employees,
            "frs_activity_status": (frs or "unknown") if "epa_frs" in sources else None,
            "has_coordinates": lat is not None,
            "has_street_address": street is not None,
            "source_count": len(sources),
            "sources": sorted(sources),
        },
        "lat": lat,
        "lon": lon,
        "naics": naics,
        "name": name,
        "names": sorted(names or [name]),
        "operator": operator,
        "postal_code": "75201",
        "records": records,
        "site_id": hashlib.sha256(key.encode("utf-8")).hexdigest(),
        "state": "TX",
        "street": street,
        "taxonomy_matches": matches,
    }


def write_snapshot(directory, sites):
    """Write sites.jsonl.gz and manifest.json the way the builder does."""
    directory.mkdir(parents=True, exist_ok=True)
    ordered = sorted(sites, key=lambda site: site["site_id"])
    payload = build._gzip_lines([build.canonical_json(site) for site in ordered])
    (directory / build.SITES_FILE).write_bytes(payload)
    manifest = {
        "attribution_required": ["© OpenStreetMap contributors"],
        "license_union": [{"id": "ODbL-1.0", "source_id": "osm_overpass"}],
        "schema": build.MANIFEST_SCHEMA,
        "site_schema": SCHEMA_VERSION,
        "snapshot_id": hashlib.sha256(payload).hexdigest(),
        "states": ["TX"],
        "taxonomy": TAXONOMY.summary(),
    }
    (directory / build.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    return directory


def synthetic_sites():
    return [
        make_site("machine", "Prairie Precision Machining", employees=180),
        make_site("molder", "Gulf Coast Molding", site_types=("plant_plastics_rubber",),
                  naics="326199", employees=240, sources=OSHA_FRS, frs="active"),
        make_site("twin-a", "Twin Fab Alpha", naics="332999", employees=300),
        make_site("twin-b", "Twin Fab Bravo", naics="332999", employees=300),
        make_site("giant", "Giant Assembly Plant", naics="332999", employees=17000),
        make_site("tiny", "Tiny Lathe Shop", employees=8),
        make_site("aero", "Mega Aero Works", site_types=PLANT, naics="336411"),
        make_site("lockheed", "Lockheed Martin Corporation Example Campus", site_types=PLANT,
                  naics="336414", operator="Lockheed Martin Corporation", employees=3900),
        make_site("frs-code", "Bluebonnet Parts", sources=OSHA_FRS, frs="active",
                  frs_codes=("336414",)),
        make_site("amazon", "Amazon.com Services LLC - XMP7", site_types=WAREHOUSE,
                  naics="493110", operator="Amazon.com Services LLC", employees=2700),
        make_site("gxo", "TX - Sample Account", site_types=WAREHOUSE, naics="493110",
                  operator="GXO, Inc.", employees=64),
        make_site("support-office", "50001 Example Support Office", site_types=GROCERY,
                  naics="445110", employees=2100),
        make_site("hq-naics", "Acme Holdings", site_types=("wholesale_distribution",),
                  naics="551114", sources=FRS, employees=None, frs="active"),
        make_site("office-warehouse", "Dallas Office/Warehouse", site_types=WAREHOUSE,
                  naics="493110", employees=120),
        make_site("rsc", "Example Grocer Retail Support Center", site_types=WAREHOUSE,
                  naics="493110", employees=None, sources=FRS, frs="active"),
        make_site("closed", "Closed Plating Works", sources=FRS, employees=None, frs="inactive"),
        make_site("rescued", "Rescued Springs", sources=OSHA_FRS, frs="inactive"),
        make_site("nowhere", "Nowhere Fabrication", street=None, lat=None, lon=None, sources=FRS,
                  employees=None),
        make_site("hospital", "Dunmere Community Hospital", site_types=HOSPITAL, naics="622110",
                  employees=400),
        make_site("cold", "Osprella Cold Storage", site_types=("warehouse_cold_storage",),
                  naics="493120", employees=300),
        make_site("plant-dc", "Brazos Pump Works", site_types=("plant_machinery", *WAREHOUSE),
                  naics="333914", employees=300),
        make_site("chem", "Bayou Specialty Chemical",
                  site_types=("plant_chemical_pharma", "industrial_general"), naics="325998",
                  employees=300, sources=("osha_ita", "osm_overpass")),
        make_site("pecan-1", "Pecan Hospital Larkstone", site_types=HOSPITAL, naics="622110",
                  operator="Pecan Healthcare PH WEXMOOR COUNTY LARKSTONE", city="LARKSTONE"),
        make_site("pecan-2", "Pecan Hospital Fernmoor", site_types=HOSPITAL, naics="622110",
                  operator="Pecan Healthcare PH FERNMOOR MEDICAL CENTER", city="FERNMOOR"),
        make_site("pecan-3", "Pecan Hospital San Quillo", site_types=HOSPITAL, naics="622110",
                  operator="Pecan Healthcare", city="SAN QUILLO"),
        make_site("grocer", "GROCER #962", site_types=GROCERY, naics="445110", employees=200),
        make_site("post-office", "SAMPLE TOWN_0000001", site_types=("parcel_hub",), naics="491110",
                  employees=80, osha_type="state_government"),
    ]


def site_named(name):
    return next(site for site in synthetic_sites() if site["name"] == name)


@pytest.fixture(scope="module")
def snapshot_dir(tmp_path_factory):
    return write_snapshot(tmp_path_factory.mktemp("rank_snapshot"), synthetic_sites())


@pytest.fixture(scope="module")
def run(snapshot_dir, tmp_path_factory):
    return rank.write_ranking(snapshot_dir, tmp_path_factory.mktemp("rank_out"))


def rows_by_name(rows):
    return {row["name"]: row for row in rows}


def read_ranked(path):
    lines = gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()
    return [json.loads(line) for line in lines]


def context():
    return rank.build_context(synthetic_sites(), TAXONOMY)


def components(site, **kwargs):
    return rank.score(site, CONFIG, **kwargs)["components"]


# --- config ---------------------------------------------------------------------------------
def test_committed_config_is_valid_and_covers_the_taxonomy():
    assert CONFIG.sha256 == hashlib.sha256(rank.RANK_CONFIG_PATH.read_bytes()).hexdigest()
    assert sum(CONFIG.weights.values()) == pytest.approx(100.0)
    assert set(CONFIG.weights) == set(rank.COMPONENTS)
    assert {row["capability"] for row in TAXONOMY.rows} <= set(CONFIG.capability_weights)
    assert set(TAXONOMY.categories) <= set(CONFIG.category_weights)
    assert set(CONFIG.site_type_weights) <= set(TAXONOMY.site_types)
    assert CONFIG.unspecific_site_types <= set(TAXONOMY.site_types)
    assert {rule.id for rule in CONFIG.rules} >= {
        "in_house_robotics", "defense_itar", "office_headquarters", "frs_inactive", "no_location",
    }
    for raw in CONFIG_DOCUMENT["exclusions"]:
        assert raw["reason"].strip() and raw["evidence"]
        assert all(item["url"].startswith("https://") for item in raw["evidence"])
        if raw["id"] in ("in_house_robotics", "known_robot_deployment", "defense_itar"):
            for entry in raw["match"]["phrases"]["entries"]:
                assert entry["evidence_url"].startswith("https://"), entry["label"]
    defense = next(rule for rule in CONFIG.rules if rule.id == "defense_itar")
    assert {"336411", "336414", "336419", "336992"} <= set(defense.naics_prefixes)
    office = next(rule for rule in CONFIG.rules if rule.id == "office_headquarters")
    assert office.naics_prefixes == ("551114",) and office.naics_scope == "primary"
    words = {phrase for _, phrases in office.phrase_entries for phrase in phrases}
    assert {"OFFICE", "HEADQUARTERS", "CORPORATE", "SUPPORT CENTER"} <= words


def bad(mutate):
    document = copy.deepcopy(CONFIG_DOCUMENT)
    mutate(document)
    return document


def first_entry(document):
    return document["exclusions"][0]["match"]["phrases"]["entries"][0]


def rule_doc(document, rule_id):
    return next(rule for rule in document["exclusions"] if rule["id"] == rule_id)


def rename(mapping, old, new):
    mapping[new] = mapping.pop(old)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["weights"].update(size_fit=30), "sum to 100"),
        (lambda d: d["weights"].pop("ownership"), "exactly the components"),
        (lambda d: d["size"]["employees_curve"].reverse(), "increase strictly"),
        (lambda d: d["size"].update(interpolation="linear"), "log10"),
        (lambda d: d["capability_weights"]["kitting_assembly"].update(weight=1.5),
         "kitting_assembly"),
        (lambda d: d["exclusions"][0].update(evidence=[]), "evidence"),
        (lambda d: d["exclusions"][0]["evidence"][0].update(url="http://example.invalid"), "https"),
        (lambda d: d["exclusions"][0].update(reason=""), "reason"),
        (lambda d: first_entry(d).update(phrases=["Amazon"]), "uppercase"),
        (lambda d: d["exclusions"][0]["match"].update(regex="x"), "unknown matchers"),
        (lambda d: d["exclusions"][1].update(id=d["exclusions"][0]["id"]), "duplicate rule ids"),
        (lambda d: d["task_evidence"]["naics_by_capability"].update(teleport=[]),
         "unknown capability"),
        # A typo must fail closed: it would otherwise switch a weight or a rule off silently.
        (lambda d: rename(d["site_type_weights"], "plant_general", "plant_genral"),
         r"site_type_weights: unknown site types \['plant_genral'\]"),
        (lambda d: d["capability_scope"].update(
            unspecific_site_types=["industrial_genral", "plant_general"]),
         r"unspecific_site_types: unknown site types \['industrial_genral'\]"),
        (lambda d: rename(rule_doc(d, "office_headquarters")["match"]["phrases"], "unless_phrases",
                          "unless_phrase"),
         r"match.phrases: unknown keys \['unless_phrase'\]"),
        (lambda d: rename(rule_doc(d, "frs_inactive")["match"]["frs_activity_status"],
                          "unless_osha_filing_year_at_least", "unless_osha_year"),
         r"frs_activity_status: unknown keys \['unless_osha_year'\]"),
        (lambda d: d["source_corroboration"].update(values_by_source_count={"2": 0.7, "3": 1.0}),
         "must define '1'"),
        (lambda d: d["source_corroboration"]["values_by_source_count"].update({"01": 0.5}),
         "not a source count"),
        (lambda d: d.update(weigths={}), r"rank config: unknown keys \['weigths'\]"),
        (lambda d: d.pop("score_meaning"), r"rank config: missing keys \['score_meaning'\]"),
        (lambda d: rename(d["capability_weights"], "hospital_logistics", "hospital_logistic"),
         r"unknown capabilities \['hospital_logistic'\]"),
        (lambda d: rename(d["category_weights"], "manufacturing", "manufacturng"),
         r"unknown categories \['manufacturng'\]"),
        (lambda d: d["capability_weights"]["kitting_assembly"].update(wieght=0.5),
         r"unknown keys \['wieght'\]"),
        (lambda d: d["size"].pop("unknown"), r"size: missing keys \['unknown'\]"),
        (lambda d: d["operator_scale"].update(numbered_unit_sites=25), "numbered_unit_sites"),
        (lambda d: rename(d["exclusions"][0]["evidence"][0], "verified_at", "verifed_at"),
         r"missing keys \['verified_at'\]; unknown keys \['verifed_at'\]"),
        (lambda d: rename(first_entry(d), "evidence_url", "evidence_link"), "evidence_link"),
        (lambda d: first_entry(d).pop("verified_at"), "needs verified_at"),
        (lambda d: d["exclusions"][0].update(notes="x"), r"unknown keys \['notes'\]"),
        (lambda d: rule_doc(d, "defense_itar")["match"]["naics"].update(scop="any"), "scop"),
        (lambda d: rename(d["task_evidence"]["naics_by_capability"]["kitting_assembly"][0], "value",
                          "valeu"), "valeu"),
        (lambda d: d["site_type_weight_notes"].update(plant_food="no weight for this type"),
         "have no entry in site_type_weights"),
        (lambda d: first_entry(d).update(acronyms=["AMAZON COM"]), "one uppercase word"),
        (lambda d: first_entry(d).update(acronyms=[]), "non-empty list of acronyms"),
        (lambda d: rule_doc(d, "frs_inactive")["match"]["frs_activity_status"].update(
            unless_fsis_listed="yes"), "expected true or false"),
    ],
)
def test_config_validation_fails_closed(mutate, message):
    with pytest.raises(RankError, match=message):
        rank.config_from_document(bad(mutate))


def test_site_type_typo_is_refused_instead_of_falling_back_to_the_category_weight():
    """Before the fix 'plant_genral' was accepted and plant_general sites got the category weight."""
    site = make_site("pg", "Generic Works", site_types=("plant_general",), naics=None,
                     sources=("osm_overpass",))
    ctx = rank.build_context([site], TAXONOMY)
    assert components(site, context=ctx)["category_fit"]["basis"] == "site type plant_general"
    typo = bad(lambda d: rename(d["site_type_weights"], "plant_general", "plant_genral"))
    with pytest.raises(RankError, match="plant_genral"):
        rank.config_from_document(typo)


def test_a_built_config_is_checked_again_against_the_ranking_taxonomy(snapshot_dir):
    document = copy.deepcopy(TAXONOMY.document)
    old, new = "drycleaning_laundry_service", "drycleaning_storefront"
    for site_type in document["site_types"]:
        if site_type["id"] == old:
            site_type["id"] = new
    for row in document["rows"]:
        row["site_types"] = [new if item == old else item for item in row["site_types"]]
    renamed = taxonomy.Taxonomy(document)
    with pytest.raises(RankError, match=r"site_type_weights: unknown site types \['drycleaning_laundry"):
        rank.rank(snapshot_dir, CONFIG, taxonomy=renamed)


def test_snapshot_capability_without_a_weight_is_refused(tmp_path):
    document = bad(lambda d: d["capability_weights"].pop("hospital_logistics"))
    with pytest.raises(RankError, match="no capability weight"):
        rank.rank(write_snapshot(tmp_path / "snap", synthetic_sites()), document)


# --- size curve -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("employees", "expected"),
    [(0, -0.5), (5, -0.5), (20, -0.25), (50, 0.0), (100, 1.0), (500, 1.0), (1000, 1.0),
     (2500, 0.0), (5000, -0.5), (10000, -1.0), (61144, -1.0)],
)
def test_size_curve_points(employees, expected):
    assert rank.interpolate(CONFIG.employees_curve, employees) == pytest.approx(expected)


def test_size_curve_peaks_at_100_to_1000_and_penalizes_both_tails():
    value = {n: rank.interpolate(CONFIG.employees_curve, n) for n in
             (10, 30, 49, 60, 80, 100, 400, 1000, 1500, 2400, 2600, 4000)}
    assert value[10] < value[30] < value[49] < 0
    assert 0 < value[60] < value[80] < value[100] == 1.0
    assert value[400] == value[1000] == 1.0
    assert 1.0 > value[1500] > value[2400] > 0 > value[2600] > value[4000]
    unknown = components(make_site("u", "Unknown Shop", employees=None, sources=FRS))
    assert unknown["size_fit"] == {
        "basis": "size unknown", "points": 0.0, "value": CONFIG.size_unknown, "weight": 22.0,
    }
    footprint = make_site("f", "Footprint Shop", employees=None, sources=("osm_overpass",),
                          building_area=20000)
    size = components(footprint)["size_fit"]
    assert size["value"] == pytest.approx(CONFIG.building_area_confidence)
    assert "building footprint" in size["basis"]


def test_size_is_not_a_contest(run):
    rows = rows_by_name(read_ranked(run.ranked_path))
    giant, twin = rows["Giant Assembly Plant"], rows["Twin Fab Alpha"]
    tiny = rows["Tiny Lathe Shop"]
    assert giant["status"] == twin["status"] == tiny["status"] == "ranked"
    assert twin["rank"] < giant["rank"] and twin["rank"] < tiny["rank"]
    assert giant["components"]["size_fit"]["value"] == -1.0
    assert tiny["components"]["size_fit"]["value"] < 0


# --- exclusions -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("name", "rule", "detail"),
    [
        ("Amazon.com Services LLC - XMP7", "in_house_robotics", "name contains 'AMAZON'"),
        ("TX - Sample Account", "known_robot_deployment", "operator is 'GXO'"),
        ("Lockheed Martin Corporation Example Campus", "defense_itar",
         "name contains 'LOCKHEED'"),
        ("Mega Aero Works", "defense_itar", "any NAICS 336411"),
        ("Bluebonnet Parts", "defense_itar", "any NAICS 336414"),
        ("50001 Example Support Office", "office_headquarters", "name contains 'SUPPORT OFFICE'"),
        ("Acme Holdings", "office_headquarters", "primary NAICS 551114"),
        ("Closed Plating Works", "frs_inactive", "EPA FRS activity status inactive"),
        ("Nowhere Fabrication", "no_location", "no street address and no coordinates"),
    ],
)
def test_each_exclusion_rule_is_recorded_with_its_rule_id(run, name, rule, detail):
    row = rows_by_name(read_ranked(run.ranked_path))[name]
    assert row["status"] == "excluded" and row["rank"] is None
    assert rule in row["excluded"]
    assert any(item["rule"] == rule and detail in item["detail"]
               for item in row["exclusion_details"])
    expected = rank.score(site_named(name), CONFIG, context=context())
    assert row["components"] == expected["components"] and row["score"] == expected["score"]
    assert run.manifest["counts"]["exclusions_by_rule"][rule] >= 1


def test_office_misclassified_as_a_store_is_excluded_but_physical_sites_are_kept(run):
    rows = rows_by_name(read_ranked(run.ranked_path))
    office = rows["50001 Example Support Office"]
    assert office["primary_site_type"] == "grocery_store"
    assert office["excluded"] == ["office_headquarters"]
    kept = ("Dallas Office/Warehouse", "Example Grocer Retail Support Center", "Rescued Springs")
    for name in kept:
        assert rows[name]["status"] == "ranked", name
    whole_foods = make_site("wf", "Whole Foods Market", site_types=GROCERY, naics="445110",
                            operator="Amazon.com")
    assert rank.score(whole_foods, CONFIG)["excluded"] == ["in_house_robotics"]
    for name in ("Amazonas Tool & Die", "Bell Machine Works", "Officer Industries"):
        assert rank.score(make_site(name, name), CONFIG)["excluded"] == [], name
    secondary_hq = make_site("hq2", "Bluebonnet Stampings", sources=OSHA_FRS, frs="active",
                             frs_codes=("551114",))
    assert rank.score(secondary_hq, CONFIG)["excluded"] == []


@pytest.mark.parametrize(
    ("name", "operator", "rule", "detail"),
    [
        ("Sample Parcel Hub", "UPS", "known_robot_deployment", "operator is 'UPS'"),
        ("UPS Sample Hub", None, "known_robot_deployment", "name starts with 'UPS'"),
        ("Sample Hub", "United Parcel Service Inc", "known_robot_deployment",
         "operator contains 'UNITED PARCEL'"),
        ("Sample DC", "DHL Supply Chain", "known_robot_deployment",
         "operator contains 'DHL SUPPLY CHAIN'"),
        ("Sample DC", "GXO Logistics Sample LLC", "known_robot_deployment",
         "operator contains 'GXO LOGISTICS'"),
        ("Sample Avionics", "RTX Corporation", "defense_itar", "operator is 'RTX'"),
        # Spelling variants seen in filings: a plural name, and a site run on another's behalf.
        ("Sample Hub", "United Parcels Sample Co", "known_robot_deployment",
         "operator contains 'UNITED PARCELS'"),
        ("Sample Freight Desk Care of DHL", None, "known_robot_deployment",
         "name contains 'CARE OF DHL'"),
    ],
)
def test_company_acronyms_match_in_a_company_name_position(name, operator, rule, detail):
    result = rank.score(make_site(name, name, operator=operator), CONFIG)
    assert result["excluded"] == [rule]
    assert detail in result["exclusion_details"][0]["detail"]


@pytest.mark.parametrize(
    ("name", "operator"),
    [
        ("Quillmoor Machine Works", "UPS Holdings LLC"),  # another company that shortens to UPS
        ("Roll Ups Packaging", None),
        ("Quillmoor Fabrication", "Quillmoor DHL Freight Services"),
        ("Sample Works", "Sample RTX Parts Inc"),
    ],
)
def test_company_acronyms_elsewhere_in_a_name_or_operator_do_not_match(name, operator):
    assert rank.score(make_site(name, name, operator=operator), CONFIG)["excluded"] == []


def test_inactive_frs_site_with_a_recent_osha_filing_is_kept():
    rescued = make_site("r", "Rescued Springs", sources=OSHA_FRS, frs="inactive")
    assert rank.score(rescued, CONFIG)["excluded"] == []
    stale = make_site("s", "Stale Springs", sources=OSHA_FRS, frs="inactive", osha_year=2019)
    assert rank.score(stale, CONFIG)["excluded"] == ["frs_inactive"]


def test_inactive_frs_site_with_an_fsis_listing_is_kept():
    def plant(sources):
        return make_site("m", "Thistlemoor Meats", site_types=("plant_meat_poultry_egg",),
                         naics="311612", employees=None, sources=sources, frs="inactive")

    listed = rank.score(plant(("epa_frs", "fsis_mpi")), CONFIG)
    assert listed["excluded"] == []
    assert listed["components"]["activity_evidence"]["basis"] == "listed in the USDA FSIS directory"
    assert rank.score(plant(FRS), CONFIG)["excluded"] == ["frs_inactive"]
    without_override = bad(lambda d: rule_doc(d, "frs_inactive")["match"]["frs_activity_status"].pop(
        "unless_fsis_listed"))
    assert rank.score(plant(("epa_frs", "fsis_mpi")), without_override)["excluded"] == ["frs_inactive"]


def test_exclusions_are_never_dropped_by_the_limit(snapshot_dir):
    rows = rank.rank(snapshot_dir, CONFIG, limit=2)
    assert [row["rank"] for row in rows if row["status"] == "ranked"] == [1, 2]
    excluded = [row for row in rows if row["status"] == "excluded"]
    full = rank.rank(snapshot_dir, CONFIG)
    assert excluded == [row for row in full if row["status"] == "excluded"]
    assert len(excluded) == 9
    assert [row["site_id"] for row in excluded] == sorted(row["site_id"] for row in excluded)


# --- scoring --------------------------------------------------------------------------------
def test_score_returns_fit_components_only_and_adds_up():
    site = make_site("m", "Prairie Precision Machining")
    before = copy.deepcopy(site)
    result = rank.score(site, CONFIG, context=context())
    assert site == before and rank.score(site, CONFIG, context=context()) == result
    assert set(result) == {"components", "excluded", "exclusion_details", "explanation", "score"}
    assert list(result["components"]) == list(rank.COMPONENTS)
    for name, item in result["components"].items():
        assert set(item) == {"basis", "points", "value", "weight"}
        assert -1.0 <= item["value"] <= 1.0 and item["weight"] == CONFIG.weights[name]
        assert item["points"] == round(item["weight"] * item["value"], 3)
    total = sum(item["points"] for item in result["components"].values())
    assert result["score"] == round(total, 3)
    task = result["components"]["task_evidence"]["basis"]
    assert task.endswith("NAICS 332710 (Machine shops)")
    assert "interest" not in result["explanation"].lower()


def test_capability_fit_uses_specific_site_types_and_discounts_secondary_ones():
    ctx = context()
    chem = site_named("Bayou Specialty Chemical")
    assert list(rank.capability_support(chem, CONFIG, ctx)) == ["palletizing_depalletizing"]
    assert components(chem, context=ctx)["category_fit"]["value"] == 0.6
    plant = site_named("Brazos Pump Works")
    fit = components(plant, capabilities={"mobile_manipulator_case_picking"}, context=ctx)
    assert fit["capability_fit"]["value"] == pytest.approx(0.6 * CONFIG.secondary_factor)
    assert "secondary site type warehouse_general" in fit["capability_fit"]["basis"]
    assert fit["category_fit"]["basis"] == "category warehousing_storage"
    whole = components(plant, context=ctx)
    assert whole["capability_fit"]["basis"] == "fixed_arm_machine_tending"
    assert whole["category_fit"]["value"] == 1.0


def test_operator_scale_sees_chains_by_stem_and_unit_number():
    ctx = context()
    pecan = site_named("Pecan Hospital Larkstone")
    assert rank.operator_keys(pecan)[0] == "PECAN"
    scale = components(pecan, context=ctx)["operator_scale"]
    assert scale["basis"] == "3 sites share the operator key 'PECAN'"
    grocer = components(site_named("GROCER #962"), context=ctx)["operator_scale"]
    assert "unit number '#962'" in grocer["basis"] and grocer["value"] == 0.0
    assert rank.numbered_unit("2001 Example Road") is None
    assert rank.numbered_unit("61730 BRINDLECOTE SURGICAL CENTER") == "61730"
    machine = site_named("Prairie Precision Machining")
    assert components(machine, context=ctx)["operator_scale"]["value"] == 1.0
    assert components(machine)["operator_scale"]["basis"] == "no snapshot context"


@pytest.mark.parametrize(
    ("name", "street", "unit"),
    [
        ("4400 Commerce St Plant", None, None),  # a thoroughfare word before the last word
        ("7215 HWY 9999 NORTH", None, None),
        ("2001 Example Road", None, None),
        ("8100 QUILLMOOR", "8100 QUILLMOOR ROW", None),  # the site's own house number
        ("8100 QUILLMOOR", "12 MAIN ST", "8100"),
        ("0952 ST QUILL HOSPITAL", None, "0952"),  # ST after the number is Saint
        ("1397 - St Quill Health", None, "1397"),
        ("61730 BRINDLECOTE SURGICAL CENTER", None, "61730"),
        ("GROCER #962", None, "#962"),
        ("Market Store 718 / 2046", None, "Store 718"),
        ("Quillmoor Machine Works", None, None),
    ],
)
def test_numbered_unit_skips_addresses_used_as_names(name, street, unit):
    assert rank.numbered_unit(name, street) == unit


def test_an_address_used_as_a_name_is_not_scored_as_a_chain():
    ctx = context()
    site = make_site("addr", "4400 Commerce St Plant", street="4400 COMMERCE ST")
    scale = components(site, context=rank.build_context([*synthetic_sites(), site], TAXONOMY))
    assert scale["operator_scale"]["basis"] == "1 site shares the name key 'COMMERCE'"
    assert scale["operator_scale"]["value"] == 1.0
    assert "unit number" in components(site_named("GROCER #962"), context=ctx)["operator_scale"]["basis"]


def test_public_sector_sites_score_lower_than_private_ones():
    ctx = context()
    ownership = components(site_named("SAMPLE TOWN_0000001"), context=ctx)["ownership"]
    assert ownership["value"] == -1.0 and "state_government" in ownership["basis"]
    epa_only = make_site("po2", "US POST OFFICE", site_types=("parcel_hub",), naics="491110",
                         sources=FRS, employees=None)
    assert components(epa_only, context=ctx)["ownership"]["value"] == -1.0
    private = components(site_named("Prairie Precision Machining"), context=ctx)["ownership"]
    assert private["value"] == 1.0


# --- determinism and binding ----------------------------------------------------------------
def test_order_is_by_score_then_site_id_and_outputs_are_byte_identical(snapshot_dir, run,
                                                                        tmp_path):
    ranked = [row for row in read_ranked(run.ranked_path) if row["status"] == "ranked"]
    assert ranked == sorted(ranked, key=lambda row: (-row["score"], row["site_id"]))
    assert [row["rank"] for row in ranked] == list(range(1, len(ranked) + 1))
    twins = [row for row in ranked if row["name"].startswith("Twin Fab")]
    assert twins[0]["score"] == twins[1]["score"]
    assert twins[0]["site_id"] < twins[1]["site_id"] and twins[1]["rank"] == twins[0]["rank"] + 1
    again = rank.write_ranking(snapshot_dir, tmp_path / "again")
    for name in (rank.RANKED_FILE, rank.RANK_MANIFEST_FILE, rank.REVIEW_FILE):
        assert (again.out_dir / name).read_bytes() == (run.out_dir / name).read_bytes(), name
    # A two-source machine shop (NAICS 332710) leads; a 2025 OSHA filing overrides its closed
    # FRS record.
    assert [row["name"] for row in ranked[:2]] == ["Rescued Springs", "Gulf Coast Molding"]


def test_outputs_are_bound_to_the_config_bytes(snapshot_dir, run, tmp_path):
    assert run.manifest["rank_config"] == {
        "sha256": hashlib.sha256(rank.RANK_CONFIG_PATH.read_bytes()).hexdigest(),
        "updated_at": CONFIG_DOCUMENT["updated_at"],
        "version": CONFIG_DOCUMENT["version"],
        "weights": CONFIG.weights,
    }
    tuned = copy.deepcopy(CONFIG_DOCUMENT)
    tuned["weights"].update(capability_fit=15, size_fit=32)
    path = tmp_path / "tuned.json"
    path.write_text(json.dumps(tuned, indent=2), encoding="utf-8")
    other = rank.write_ranking(snapshot_dir, tmp_path / "tuned", path)
    assert other.manifest["rank_config"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert other.manifest["rank_config"]["sha256"] != run.manifest["rank_config"]["sha256"]
    ranked_sha = run.manifest["files"][rank.RANKED_FILE]["sha256"]
    assert other.manifest["files"][rank.RANKED_FILE]["sha256"] != ranked_sha
    reformatted = tmp_path / "same.json"
    reformatted.write_text(json.dumps(CONFIG_DOCUMENT), encoding="utf-8")
    assert rank.load_config(reformatted).sha256 != CONFIG.sha256  # the id is the file bytes
    copied = rank.config_from_document(copy.deepcopy(CONFIG_DOCUMENT))
    assert rank.config_from_document(CONFIG_DOCUMENT).sha256 == copied.sha256


def test_tampered_snapshot_is_refused(tmp_path):
    directory = write_snapshot(tmp_path / "snap", synthetic_sites())
    sites = directory / build.SITES_FILE
    sites.write_bytes(sites.read_bytes() + b"x")
    with pytest.raises(RankError, match="does not match snapshot_id"):
        rank.rank(directory, CONFIG)


# --- capability filter ----------------------------------------------------------------------
def test_capability_filter_limits_the_scope(snapshot_dir):
    rows = rank.rank(snapshot_dir, CONFIG, capabilities=["hospital_logistics"])
    assert {row["name"] for row in rows} == {
        "Dunmere Community Hospital", "Pecan Hospital Larkstone", "Pecan Hospital Fernmoor",
        "Pecan Hospital San Quillo",
    }
    for row in rows:
        assert row["capabilities"] == ["hospital_logistics"]
        assert row["components"]["capability_fit"]["basis"] == "hospital_logistics"
    picking = rank.rank(snapshot_dir, CONFIG, capabilities=["mobile_manipulator_case_picking"])
    names = [row["name"] for row in picking if row["status"] == "ranked"]
    assert names.index("Osprella Cold Storage") < names.index("Brazos Pump Works")
    with pytest.raises(RankError, match="unknown capabilities"):
        rank.rank(snapshot_dir, CONFIG, capabilities=["teleportation"])


# --- exclusions input -----------------------------------------------------------------------
def test_exclusions_input_applies_rows_with_reasons(snapshot_dir, tmp_path):
    machine = site_named("Prairie Precision Machining")["site_id"]
    rejection = {"name": "gulf coast molding", "city": "Dallas", "source": "rejection",
                 "reason": "rejected 2026-09-30: no budget", "added_at": "2026-09-30"}
    path = tmp_path / "exclusions.jsonl"
    path.write_text("\n".join([
        json.dumps({"site_id": machine, "reason": "already in CRM", "source": "crm"}),
        json.dumps(rejection),
        json.dumps({"name": "Twin Fab Alpha", "city": "Austin", "reason": "wrong city"}),
        "",
        json.dumps({"operator": "Pecan Healthcare", "reason": "system contacted",
                    "source": "crm"}),
    ]) + "\n", encoding="utf-8")
    result = rank.write_ranking(snapshot_dir, tmp_path / "out", CONFIG, exclusions_input=path)
    rows = rows_by_name(read_ranked(result.ranked_path))
    machine_row = rows["Prairie Precision Machining"]
    assert machine_row["excluded"] == ["input:crm"]
    assert "already in CRM" in machine_row["exclusion_details"][0]["detail"]
    assert rows["Gulf Coast Molding"]["excluded"] == ["input:rejection"]
    assert rows["Pecan Hospital San Quillo"]["excluded"] == ["input:crm"]
    assert rows["Pecan Hospital Larkstone"]["status"] == "ranked"  # operator rows match exactly
    assert rows["Twin Fab Alpha"]["status"] == "ranked"
    assert result.manifest["exclusions_input"] == {
        "rows": 4, "rows_matched": 3, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "unmatched_lines": [3],
    }
    assert result.manifest["counts"]["exclusions_by_rule"]["input:crm"] == 2


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "not a JSON object"),
        (json.dumps({"name": "Acme"}), "reason"),
        (json.dumps({"name": "Acme", "site_id": "0" * 64, "reason": "x"}), "exactly one"),
        (json.dumps({"site_id": "abc", "reason": "x"}), "64 hex"),
        (json.dumps({"name": "Acme", "reason": "x", "source": "CRM!"}), "source"),
        (json.dumps({"operator": "Acme", "city": "Dallas", "reason": "x"}), "a name row only"),
        (json.dumps({"name": "Acme", "reason": "x", "colour": "red"}), "unknown fields"),
    ],
)
def test_malformed_exclusions_input_fails_closed(tmp_path, line, message):
    path = tmp_path / "bad.jsonl"
    good = json.dumps({"name": "Fine", "reason": "ok"})
    path.write_text(f"{good}\n{line}\n", encoding="utf-8")
    with pytest.raises(RankError, match=message) as error:
        rank.load_exclusions_input(path)
    assert "bad.jsonl:2" in str(error.value)


# --- manifest, review and CLI ---------------------------------------------------------------
def test_manifest_records_counts_exclusions_percentiles_and_files(run, snapshot_dir):
    manifest = run.manifest
    assert json.loads(run.manifest_path.read_text(encoding="utf-8")) == manifest
    assert manifest["schema"] == "blueprint.site_universe.rank_manifest.v1"
    assert manifest["distribution"] == "internal_only"
    snapshot = json.loads((snapshot_dir / build.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["snapshot"]["snapshot_id"] == snapshot["snapshot_id"]
    assert manifest["taxonomy_used"]["matches_snapshot"] is True
    assert manifest["license_union"] == snapshot["license_union"]
    assert manifest["attribution_required"] == ["© OpenStreetMap contributors"]
    counts = manifest["counts"]
    assert counts["sites_in_snapshot"] == len(synthetic_sites()) == counts["sites_in_scope"]
    assert counts["sites_ranked"] + counts["sites_excluded"] == counts["sites_in_scope"]
    assert counts["sites_excluded"] == 9
    assert counts["exclusions_by_rule"] == {
        "in_house_robotics": 1, "known_robot_deployment": 1, "defense_itar": 3,
        "office_headquarters": 2, "frs_inactive": 1, "no_location": 1,
    }
    assert counts["by_capability"]["hospital_logistics"] == {
        "excluded": 0, "in_scope": 4, "ranked": 4,
    }
    for cap, row in counts["by_capability"].items():
        assert row["in_scope"] == row["ranked"] + row["excluded"], cap
    percentiles = manifest["score_percentiles"]
    assert list(percentiles) == ["p0", "p10", "p25", "p50", "p75", "p90", "p99", "p100"]
    rows = read_ranked(run.ranked_path)
    scores = sorted(row["score"] for row in rows if row["status"] == "ranked")
    assert percentiles["p0"] == scores[0] and percentiles["p100"] == scores[-1]
    for name in (rank.RANKED_FILE, rank.REVIEW_FILE):
        data = (run.out_dir / name).read_bytes()
        assert manifest["files"][name]["sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["files"][rank.RANKED_FILE]["lines"] == counts["sites_in_scope"]
    assert "not a measure of buying interest" in manifest["score_meaning"]
    unverified = manifest["exclusion_evidence_unverified"]
    assert unverified["count"] == len(CONFIG.unverified_evidence())


def test_review_lists_the_top_sites_per_capability(run):
    review = run.review_path.read_text(encoding="utf-8")
    assert review.startswith("# Site ranking review")
    assert "Do not commit this file" in review and "not a measure of buying interest" in review
    for cap in run.manifest["counts"]["by_capability"]:
        assert f"## `{cap}`" in review
    section = review.split("## `fixed_arm_machine_tending`")[1].split("\n## ")[0]
    assert "| 1 | Rescued Springs | DALLAS | plant_fabricated_metal | 332710 | 180 |" in section
    assert "| 2 | Gulf Coast Molding | DALLAS | plant_plastics_rubber | 326199 | 240 |" in section
    assert "capability_fit +25.0" in section
    assert "### `office_headquarters`" in review and "SUPPORT OFFICE" in review


def test_builder_snapshot_ranks_end_to_end(tmp_path):
    """Seam test: the builder's hermetic snapshot feeds the ranker without adaptation."""
    snapshot = build.build(["TX"], out_dir=tmp_path / "snap", cache=seeded_cache(tmp_path / "raw"))
    result = rank.write_ranking(snapshot.out_dir, tmp_path / "rank")
    rows = rows_by_name(read_ranked(result.ranked_path))
    assert len(rows) == snapshot.manifest["counts"]["sites"]
    assert len(rows) == result.manifest["counts"]["sites_in_scope"]
    assert rows["GULF COAST RECOVERY MRF"]["excluded"] == ["frs_inactive"]
    tool = rows["PRAIRIE TOOL WORKS"]
    assert tool["status"] == "ranked"
    assert tool["components"]["capability_fit"]["basis"] == "fixed_arm_machine_tending"
    ctx = rank.build_context(list(build.read_sites(snapshot.out_dir)), TAXONOMY)
    loaded = {site["site_id"]: site for site in rank.load_snapshot(snapshot.out_dir).sites}
    for full in build.read_sites(snapshot.out_dir):
        slim = loaded[full["site_id"]]
        assert rank.score(full, CONFIG, context=ctx) == rank.score(slim, CONFIG, context=ctx)


def test_cli_rank_writes_the_three_outputs(snapshot_dir, tmp_path, capsys):
    out = tmp_path / "cli"
    base = ["rank", "--snapshot", str(snapshot_dir), "--out", str(out)]
    code = main(base + ["--top", "2", "--capabilities",
                        "fixed_arm_machine_tending,hospital_logistics"])
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    snapshot = json.loads((snapshot_dir / build.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert summary["snapshot_id"] == snapshot["snapshot_id"]
    assert sorted(path.name for path in out.iterdir()) == sorted(
        [rank.RANKED_FILE, rank.RANK_MANIFEST_FILE, rank.REVIEW_FILE])
    rows = read_ranked(out / rank.RANKED_FILE)
    assert [row["rank"] for row in rows if row["status"] == "ranked"] == [1, 2]
    manifest = json.loads((out / rank.RANK_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["scope"] == {
        "capabilities": ["fixed_arm_machine_tending", "hospital_logistics"], "limit": 2,
    }
    review = (out / rank.REVIEW_FILE).read_text(encoding="utf-8")
    assert "## `fixed_arm_machine_tending`" in review and "## `hospital_logistics`" in review
    assert "## `palletizing_depalletizing`" not in review
    assert main(base + ["--capabilities", "nope"]) == 1
    assert main(base + ["--top", "0"]) == 1
    inside = rank.RANK_CONFIG_PATH.parent / "_refused_rank_output_never_created"
    assert main(["rank", "--snapshot", str(snapshot_dir), "--out", str(inside)]) == 1
    assert "inside the repository worktree" in capsys.readouterr().err
    assert not inside.exists()
