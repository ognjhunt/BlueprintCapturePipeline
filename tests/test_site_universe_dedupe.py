"""Dedupe of source records into sites (pure, no network)."""

import random

from tools.site_universe import dedupe
from tools.site_universe.records import SourceRecord

AT = "2026-10-04T12:00:00Z"


def rec(source, rid, name, street=None, *, city="DALLAS", postal="75201", lat=None, lon=None,
        precision=None, footprint=None, operator=None):
    attributes = {"coordinate_precision": precision} if precision else {}
    return SourceRecord(
        source_id=source, source_record_id=rid, name=name, operator=operator, street=street, unit=None,
        city=city, state="TX", postal_code=postal, country="US", lat=lat, lon=lon, naics=None,
        category=None, employees=None, building_area_m2=None, retrieved_at=AT, raw_sha256="0" * 64,
        attributes=attributes, footprint=footprint,
    )


def groups(clusters):
    return sorted(sorted(record.ref for record in cluster.members) for cluster in clusters)


def test_exact_address_merges_similar_names():
    records = [
        rec("epa_frs", "1", "ACME COLD STORAGE LLC", "100 INDUSTRIAL BLVD"),
        rec("epa_frs", "3", "ACME COLD STORAGE", "100 INDUSTRIAL BLVD"),
        rec("osha_ita", "500002", "Acme Cold Storage - Dallas", "100 INDUSTRIAL BLVD"),
    ]
    clusters, stats = dedupe.cluster(records)
    assert groups(clusters) == [["epa_frs:1", "epa_frs:3", "osha_ita:500002"]]
    assert {edge["reason"] for edge in clusters[0].merges} == {"address_key"}
    assert len(clusters[0].merges) == 2
    assert stats["merges_address_key"] == 2 and stats["records_merged_away"] == 2


def test_different_names_at_one_address_stay_apart():
    records = [
        rec("epa_frs", "1", "ACME COLD STORAGE LLC", "100 INDUSTRIAL BLVD"),
        rec("epa_frs", "2", "BOLT MANUFACTURING INC", "100 INDUSTRIAL BLVD"),
        rec("osha_ita", "500003", "Bolt Manufacturing", "100 INDUSTRIAL BLVD"),
        rec("epa_frs", "7", "ZORVEL CLEANERS", "4317 WRENFIELD LN"),
        rec("epa_frs", "8", "ODALYNS CLEANER", "4317 WRENFIELD LN"),
        rec("epa_frs", "9", "VELLUMA DRYCLEAN", "4317 WRENFIELD LN"),
    ]
    clusters, stats = dedupe.cluster(records)
    assert groups(clusters) == [
        ["epa_frs:1"], ["epa_frs:2", "osha_ita:500003"], ["epa_frs:7"], ["epa_frs:8"], ["epa_frs:9"],
    ]
    assert stats["address_keys_with_several_sites"] == 2
    assert stats["address_pairs_kept_apart_by_name"] >= 5


def test_a_third_name_cannot_chain_two_tenants_of_one_address():
    """A name that resembles both tenants joins one of them; the tenants stay two sites."""
    records = [
        rec("epa_frs", "1", "ACME MOTOR WORKS", "100 INDUSTRIAL BLVD"),
        rec("epa_frs", "2", "BOLT AUTOMOTIVE", "100 INDUSTRIAL BLVD"),
        rec("osha_ita", "3", "BOLT AUTOMOTIVE ACME COMPLEX", "100 INDUSTRIAL BLVD"),
    ]
    rng = random.Random(11)
    for _ in range(6):
        shuffled = records[:]
        rng.shuffle(shuffled)
        clusters, stats = dedupe.cluster(shuffled)
        # The best match (score 100) joins first; the weaker one (score 60) would chain ACME to BOLT.
        assert groups(clusters) == [["epa_frs:1"], ["epa_frs:2", "osha_ita:3"]]
        assert stats["joins_refused_address_key"] == 1 and stats["merges_address_key"] == 1


def test_a_store_filed_under_its_number_still_joins_through_its_operator():
    """The operator names the business: 'QM 4410' run by Quillmoor is the Quillmoor store."""
    records = [
        rec("epa_frs", "1", "QUILLMOOR SUPERCENTER 4410", "100 INDUSTRIAL BLVD"),
        rec("osha_ita", "2", "QM 4410", "100 INDUSTRIAL BLVD", operator="Quillmoor Stores Texas LLC"),
        rec("osm_overpass", "way/3", "Quillmoor Supercenter", "100 INDUSTRIAL BLVD"),
    ]
    clusters, stats = dedupe.cluster(records)
    assert groups(clusters) == [["epa_frs:1", "osha_ita:2", "osm_overpass:way/3"]]
    assert "joins_refused_address_key" not in stats
    # Without the operator, the store-number name and the brand name look like two tenants.
    records[1] = rec("osha_ita", "2", "QM 4410", "100 INDUSTRIAL BLVD")
    clusters, stats = dedupe.cluster(records)
    assert groups(clusters) == [["epa_frs:1", "osm_overpass:way/3"], ["osha_ita:2"]]
    assert stats["joins_refused_address_key"] == 1


def test_a_nearby_name_cannot_chain_two_tenants_of_one_address():
    records = [
        rec("epa_frs", "1", "ACME", "100 INDUSTRIAL BLVD", lat=32.78, lon=-96.8, precision="precise"),
        rec("epa_frs", "2", "BOLT", "100 INDUSTRIAL BLVD", lat=32.7801, lon=-96.8, precision="precise"),
        rec("osm_overpass", "node/3", "Acme Bolt", city=None, postal=None, lat=32.7802, lon=-96.8),
    ]
    clusters, stats = dedupe.cluster(records)
    assert groups(clusters) == [["epa_frs:1"], ["epa_frs:2", "osm_overpass:node/3"]]
    assert stats["joins_refused_proximity_name"] == 1


def test_proximity_and_similar_name_merge_within_75_metres():
    records = [
        rec("epa_frs", "13", "KESTRAMAR BREWING CO", "7410 MALTRIDGE LN", city="LARKSTONE", postal="77161",
            lat=29.4, lon=-98.5, precision="precise"),
        rec("osm_overpass", "node/1010", "Kestramar Brewing", city=None, postal=None, lat=29.4003, lon=-98.5001),
    ]
    clusters, stats = dedupe.cluster(records)
    assert len(clusters) == 1
    edge = clusters[0].merges[0]
    assert edge["reason"] == "proximity_name" and 30 < edge["distance_m"] < 40
    assert stats["merges_proximity_name"] == 1


def test_proximity_needs_a_matching_name_and_the_distance_bound():
    near_other_name = [
        rec("epa_frs", "1", "KESTRAMAR BREWING CO", lat=29.4, lon=-98.5, postal=None),
        rec("osm_overpass", "node/2", "Pellwick Foundry", lat=29.4003, lon=-98.5001, postal=None),
    ]
    assert len(dedupe.cluster(near_other_name)[0]) == 2
    far_same_name = [
        rec("epa_frs", "1", "KESTRAMAR BREWING CO", lat=29.4, lon=-98.5, postal=None),
        rec("osm_overpass", "node/2", "Kestramar Brewing", lat=29.402, lon=-98.5, postal=None),
    ]
    assert len(dedupe.cluster(far_same_name)[0]) == 2


def test_approximate_and_stacked_coordinates_never_merge_by_proximity():
    approximate = [
        rec("epa_frs", "12", "NORTH TEXAS LINEN SUPPLY", lat=33.2148, lon=-97.1331, precision="approximate",
            postal=None),
        rec("osm_overpass", "way/5", "North Texas Linen Supply", lat=33.2148, lon=-97.1331, postal=None),
    ]
    assert len(dedupe.cluster(approximate)[0]) == 2
    stacked = [
        rec("epa_frs", str(index), f"PLANT {name}", lat=31.0, lon=-97.0, precision="precise", postal=None)
        for index, name in enumerate(["ALPHA", "ALPHA", "BETA", "GAMMA", "DELTA"])
    ]
    clusters, stats = dedupe.cluster(stacked)
    assert len(clusters) == 5 and stats["records_with_proximity_coordinates"] == 0


def test_point_inside_an_osm_footprint_merges():
    ring = [(32.779, -96.803), (32.779, -96.799), (32.7802, -96.799), (32.7802, -96.803)]
    records = [
        rec("epa_frs", "1", "ACME COLD STORAGE LLC", "100 INDUSTRIAL BLVD", lat=32.78, lon=-96.8,
            precision="precise"),
        rec("osm_overpass", "way/1001", "Acme Cold Storage", city=None, postal=None, lat=32.7796, lon=-96.801,
            footprint=([ring], [])),
        rec("epa_frs", "2", "BOLT MANUFACTURING INC", "100 INDUSTRIAL BLVD", lat=32.7801, lon=-96.8001,
            precision="precise"),
    ]
    clusters, _ = dedupe.cluster(records)
    assert groups(clusters) == [["epa_frs:1", "osm_overpass:way/1001"], ["epa_frs:2"]]
    edge = next(c for c in clusters if len(c.members) == 2).merges[0]
    assert edge["inside_footprint"] is True and edge["distance_m"] == 0.0


def test_one_street_written_two_ways_merges_by_zip_and_house_number():
    records = [
        rec("epa_frs", "1", "QUILLMOOR STARCH & CHEMICAL CO", "2715 N PELLWICK SW PKWY", city="FERNMOOR",
            postal="77162"),
        rec("epa_frs", "2", "QUILLMOOR STARCH AND CHEMICAL CO", "2715 N PELLWICK SOUTHWEST PKWY",
            city="FERNMOOR", postal="77162"),
        rec("epa_frs", "3", "QUILLMOOR STARCH AND CHEMICAL CO", "2715 ELM ST", city="FERNMOOR",
            postal="77162"),
    ]
    clusters, _ = dedupe.cluster(records)
    assert groups(clusters) == [["epa_frs:1", "epa_frs:2"], ["epa_frs:3"]]
    assert clusters[0].merges[0]["reason"] == "postal_house_number_name"


def test_records_without_address_merge_on_identical_name_in_one_zip():
    records = [
        rec("epa_frs", "1", "TALLOWMERE FRESH MEATS", city="LARKSTONE", postal="77163"),
        rec("epa_frs", "2", "TALLOWMERE FRESH MEATS INC", city="LARKSTONE", postal="77163"),
        rec("epa_frs", "3", "TALLOWMERE FRESH MEATS", city="LARKSTONE", postal="77164"),
        rec("epa_frs", "4", "CLEANERS", city="LARKSTONE", postal="77163"),
        rec("epa_frs", "5", "CLEANERS", city="LARKSTONE", postal="77163"),
    ]
    clusters, _ = dedupe.cluster(records)
    assert groups(clusters) == [["epa_frs:1", "epa_frs:2"], ["epa_frs:3"], ["epa_frs:4"], ["epa_frs:5"]]
    assert clusters[0].merges[0]["reason"] == "name_postal_no_address"


def test_partition_does_not_depend_on_input_order():
    records = [
        rec("epa_frs", "1", "ACME COLD STORAGE LLC", "100 INDUSTRIAL BLVD", lat=32.78, lon=-96.8,
            precision="precise"),
        rec("osha_ita", "2", "Acme Cold Storage", "100 INDUSTRIAL BLVD"),
        rec("epa_frs", "3", "BOLT MANUFACTURING INC", "100 INDUSTRIAL BLVD"),
        rec("osm_overpass", "node/4", "Acme Cold Storage", city=None, postal=None, lat=32.7801, lon=-96.8),
        rec("epa_frs", "5", "ZETA PLASTICS", "9 ELM ST"),
    ]
    expected = groups(dedupe.cluster(records)[0])
    rng = random.Random(4)
    for _ in range(10):
        shuffled = records[:]
        rng.shuffle(shuffled)
        assert groups(dedupe.cluster(shuffled)[0]) == expected
