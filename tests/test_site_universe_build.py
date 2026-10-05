"""End-to-end snapshot build on the hermetic raw cache: content, determinism, manifest, CLI."""

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from tests import site_universe_fixture
from tests.site_universe_fixture import (
    FETCHED_AT,
    FSIS_IMPORTED_AT,
    OSHA_IMPORTED_AT,
    PERSONAL_STRINGS,
    seeded_cache,
)
from tools.site_universe import build, taxonomy
from tools.site_universe.cli import main
from tools.site_universe.fetch import RAW_MANIFEST_NAME, RawCache


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    root = tmp_path_factory.mktemp("site_universe")
    cache = seeded_cache(root / "raw")
    return build.build(["TX"], out_dir=root / "out", cache=cache)


def _sites(snapshot):
    return list(build.read_sites(snapshot.out_dir))


def _by_name(snapshot):
    return {site["name"]: site for site in _sites(snapshot)}


def test_sites_merge_across_sources_and_keep_provenance(snapshot):
    sites = _by_name(snapshot)
    assert sorted(sites) == sorted([
        "Acme Cold Storage - Dallas", "Bolt Manufacturing", "Lone Pine Poultry - Houston",
        "GULF COAST RECOVERY MRF", "PRAIRIE TOOL WORKS", "NORTH TEXAS LINEN SUPPLY", "Kestramar Brewing",
        "H-E-B", "Thistlemoor Custom Meats", "Westside Industrial Park", "Glimmerdale Regional Hospital",
        "Ironclad Foundry",
    ])
    acme = sites["Acme Cold Storage - Dallas"]
    refs = [f"{r['source_id']}:{r['source_record_id']}" for r in acme["records"]]
    assert refs == ["epa_frs:110000000001", "epa_frs:110000000003", "osha_ita:500002", "osm_overpass:way/1001"]
    assert len(acme["merges"]) == 3 and all(edge["reason"] for edge in acme["merges"])
    assert {edge["reason"] for edge in acme["merges"]} <= {"address_key", "proximity_name"}
    assert acme["employees"] == 85 and acme["employees_source"] == "osha_ita"
    assert (acme["street"], acme["postal_code"], acme["address_source"]) == ("100 INDUSTRIAL BLVD", "75201", "epa_frs")
    assert acme["coordinate_source"] == "osm_overpass" and acme["building_area_m2"] > 40_000
    assert acme["category"] == "warehousing_storage" and acme["naics"] == "493120"
    assert acme["attribution_required"] is True and acme["attributions"] == ["© OpenStreetMap contributors"]
    assert {item["id"] for item in acme["licenses"]} == {"US-PD", "US-Gov-Work", "ODbL-1.0"}
    # A second business at the same address stays a separate site with its own id.
    bolt = sites["Bolt Manufacturing"]
    assert bolt["unit"] == "STE 5" and bolt["employees"] == 40 and bolt["operator"] is None
    assert acme["id_basis"] == bolt["id_basis"] == "address_name" and acme["site_id"] != bolt["site_id"]
    poultry = sites["Lone Pine Poultry - Houston"]
    assert poultry["features"]["sources"] == ["epa_frs", "fsis_mpi", "osha_ita"]
    assert poultry["employees"] == 450 and poultry["category"] == "food_processing"
    assert poultry["taxonomy_matches"]["primary_site_type"] == "plant_meat_poultry_egg"
    assert "fsis_mpi:fsis_mpi:directory" in poultry["taxonomy_matches"]["matched_by"]["plant_meat_poultry_egg"]
    brewery = sites["Kestramar Brewing"]
    assert [m["reason"] for m in brewery["merges"]] == ["proximity_name"]
    assert sites["H-E-B"]["features"]["sources"] == ["epa_frs", "osm_overpass"]
    mrf = sites["GULF COAST RECOVERY MRF"]
    assert mrf["id_basis"] == "geohash7_name" and mrf["street"] is None
    assert mrf["features"]["frs_activity_status"] == "inactive"


def test_taxonomy_matches_and_ranker_features(snapshot):
    sites = _by_name(snapshot)
    tool = sites["PRAIRIE TOOL WORKS"]
    assert tool["taxonomy_matches"]["site_types"] == ["plant_machinery"]
    assert tool["taxonomy_matches"]["matched_by"] == {"plant_machinery": ["epa_frs:sic:3599"]}
    assert "fixed_arm_machine_tending" in tool["taxonomy_matches"]["capabilities"]
    hospital = sites["Glimmerdale Regional Hospital"]
    assert hospital["taxonomy_matches"]["rows"] == ["hospital_logistics"]
    for site in sites.values():
        features = site["features"]
        assert set(features) >= {
            "category", "employees", "building_area_m2", "source_count", "sources", "capability_count",
            "site_type_count", "taxonomy_row_count", "has_coordinates", "has_street_address",
        }
        assert features["source_count"] == len(features["sources"]) == len({r["source_id"] for r in site["records"]})
        assert features["capability_count"] == len(site["taxonomy_matches"]["capabilities"])
        assert "score" not in features and "rank" not in features
    linen = sites["NORTH TEXAS LINEN SUPPLY"]
    assert linen["features"]["coordinate_precision"] == "approximate"


def test_no_personal_data_reaches_the_snapshot(snapshot):
    sites_text = gzip.decompress(snapshot.sites_path.read_bytes()).decode("utf-8")
    for value in PERSONAL_STRINGS:
        assert value not in sites_text + snapshot.manifest_path.read_text(encoding="utf-8"), value
    for field in ('"ein"', '"phone"', '"email"', '"change_reason"', '"user"', '"uid"', '"contact:phone"'):
        assert field not in sites_text


def test_snapshot_file_is_canonical_sorted_and_gzip_deterministic(snapshot):
    payload = snapshot.sites_path.read_bytes()
    assert payload[3] & 0x08 == 0  # no file name in the gzip header
    assert payload[4:8] == b"\x00\x00\x00\x00"  # mtime 0
    lines = gzip.decompress(payload).decode("utf-8").splitlines()
    ids = [json.loads(line)["site_id"] for line in lines]
    assert ids == sorted(ids) and len(ids) == len(set(ids))
    for line in lines:
        assert build.canonical_json(json.loads(line)) == line
    assert snapshot.snapshot_id == hashlib.sha256(payload).hexdigest()
    for site in map(json.loads, lines):
        assert site["site_id"] == hashlib.sha256(site["id_key"].encode()).hexdigest()


def _record_refs(site):
    return tuple(f"{record['source_id']}:{record['source_record_id']}" for record in site["records"])


def test_site_id_depends_only_on_the_sites_own_records(tmp_path, monkeypatch, snapshot):
    """Removing the second tenant at Acme's address must not move the id of any other site."""
    fixtures = tmp_path / "fixtures"
    shutil.copytree(site_universe_fixture.FIXTURES, fixtures)
    for relative, marker in (("epa_frs_tx/TX_FACILITY_FILE.CSV", b",110000000002,"),
                             ("osha_ita_300a.csv", b",500003,")):
        path = fixtures / relative
        lines = path.read_bytes().splitlines(keepends=True)
        path.write_bytes(b"".join(line for line in lines if marker not in line))
    monkeypatch.setattr(site_universe_fixture, "FIXTURES", fixtures)
    without = build.build(["TX"], out_dir=tmp_path / "out", cache=seeded_cache(tmp_path / "raw"))
    before = {_record_refs(site): site for site in _sites(snapshot)}
    after = {_record_refs(site): site for site in build.read_sites(without.out_dir)}
    assert "Bolt Manufacturing" not in {site["name"] for site in after.values()}
    assert len(after) == len(before) - 1 and set(after) < set(before)
    for refs, site in after.items():
        assert site["site_id"] == before[refs]["site_id"], site["name"]
        assert site["id_basis"] == before[refs]["id_basis"]


def _id_site(anchor, *, address_key="100 MAIN ST|DALLAS|TX|75201", name="ACME"):
    return {"_address_key": address_key, "_anchor_ref": anchor, "_name_tokens": name, "city": "DALLAS",
            "lat": None, "lon": None, "postal_code": "75201", "state": "TX"}


def test_a_shared_id_key_stays_with_the_earliest_anchor_record():
    alone = [_id_site("epa_frs:110000000020")]
    build._assign_ids(alone)
    shared = [_id_site("osm_overpass:way/9"), _id_site("epa_frs:110000000020"), _id_site("epa_frs:1", name="BOLT")]
    stats = build._assign_ids(shared)
    assert shared[1]["site_id"] == alone[0]["site_id"]  # keeps its id when a later site shares its key
    assert shared[0]["id_key"] == "100 MAIN ST|DALLAS|TX|75201|ACME|osm_overpass:way/9"
    assert shared[2]["id_key"] == "100 MAIN ST|DALLAS|TX|75201|BOLT"
    assert stats["id_key_collisions"] == 2
    assert {site["id_basis"] for site in shared} == {"address_name"}


def test_same_raw_inputs_give_byte_identical_output(tmp_path, snapshot):
    first = build.build(["TX"], out_dir=tmp_path / "a", cache=seeded_cache(tmp_path / "raw_a"))
    second = build.build(["TX"], out_dir=tmp_path / "b", cache=seeded_cache(tmp_path / "raw_b"))
    rebuilt = build.build(
        ["TX"], out_dir=tmp_path / "c", cache=RawCache(tmp_path / "raw_a", allow_network=False)
    )
    for other in (second, rebuilt, snapshot):
        assert other.sites_path.read_bytes() == first.sites_path.read_bytes()
        assert other.manifest_path.read_bytes() == first.manifest_path.read_bytes()
        assert other.snapshot_id == first.snapshot_id


def test_manifest_records_inputs_counts_merges_and_licenses(snapshot, tmp_path):
    manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "blueprint.site_universe.manifest.v1"
    assert manifest["site_schema"] == "blueprint.site_universe.snapshot.v1"
    assert manifest["distribution"] == "internal_only"
    assert manifest["snapshot_id"] == snapshot.snapshot_id
    assert manifest["files"]["sites.jsonl.gz"] == {
        "bytes": snapshot.sites_path.stat().st_size, "lines": 12, "sha256": snapshot.snapshot_id,
    }
    raw = json.loads((snapshot.out_dir.parent / "raw" / RAW_MANIFEST_NAME).read_text(encoding="utf-8"))
    for row in manifest["inputs"]:
        assert raw["entries"][row["url"]]["sha256"] == row["sha256"]
    by_source = {row["source_id"] for row in manifest["inputs"]}
    assert by_source == {"epa_frs", "fsis_mpi", "osha_ita", "osm_overpass"}
    assert {row["retrieved_at"] for row in manifest["inputs"]} == {FETCHED_AT, OSHA_IMPORTED_AT, FSIS_IMPORTED_AT}
    assert len([row for row in manifest["inputs"] if row["source_id"] == "osm_overpass"]) == len(
        taxonomy.load().overpass_queries("TX")
    )
    counts = manifest["counts"]
    assert counts["sites"] == 12
    assert sum(counts["sites_by_category"].values()) == 12
    assert counts["sites_by_category"]["manufacturing"] == 3
    assert counts["sites_with_employees"] == 3
    records = counts["records_by_source"]
    assert records["epa_frs"]["kept"] == 9 and records["epa_frs"]["dropped_placeholder_name"] == 1
    assert records["epa_frs"]["dropped_inactive_site_type"] == 1
    assert records["osm_overpass"]["duplicate_across_raw_inputs"] == 1
    assert records["osm_overpass"]["dropped_generic_name"] == 1
    assert records["osm_overpass"]["dropped_excluded_tag"] == 1
    assert records["osha_ita"]["dropped_no_matching_code"] == 1
    assert counts["sites_by_taxonomy_row"]["hospital_logistics"] == 1
    assert counts["sites_by_site_type"]["plant_meat_poultry_egg"] == 2
    merges = manifest["merge_stats"]
    assert merges["records_in"] - merges["records_merged_away"] == merges["sites_out"] == 12
    assert merges["multi_source_sites"] == 5
    assert sum(merges["sites_by_source_combination"].values()) == 12
    licenses = {item["source_id"]: item for item in manifest["license_union"]}
    assert licenses["osm_overpass"]["id"] == "ODbL-1.0" and licenses["osm_overpass"]["share_alike"] is True
    assert manifest["attribution_required"] == ["© OpenStreetMap contributors"]
    assert {entry["id"] for entry in manifest["sources_used"]} == set(licenses)
    assert manifest["taxonomy"]["sha256"] == taxonomy.file_sha256()
    assert manifest["sources_skipped"] == []


def test_missing_manual_import_is_recorded_as_a_skip(tmp_path):
    cache = seeded_cache(tmp_path / "raw", manual=False)
    result = build.build(["TX"], out_dir=tmp_path / "out", cache=cache)
    skipped = {row["source_id"]: row["reason"] for row in result.manifest["sources_skipped"]}
    assert set(skipped) == {"fsis_mpi", "osha_ita"} and "manual import" in skipped["osha_ita"]
    assert result.manifest["counts"]["sites_with_employees"] == 0
    with pytest.raises(Exception, match="manual import"):
        build.build(["TX"], out_dir=tmp_path / "strict", cache=cache, strict=True)


def test_cli_build_stats_and_integrity_check(tmp_path, capsys):
    seeded_cache(tmp_path / "out" / "raw")
    assert main(["build", "--states", "TX", "--out", str(tmp_path / "out"), "--offline"]) == 0
    built = json.loads(capsys.readouterr().out)
    assert built["sites"] == 12
    assert main(["stats", str(tmp_path / "out"), "--sample", "5"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["snapshot_id"] == built["snapshot_id"] and len(stats["sample"]) == 5
    assert set(stats["sample"][0]) == {"name", "city", "category"}
    sites = tmp_path / "out" / build.SITES_FILE
    sites.write_bytes(sites.read_bytes() + b"x")
    assert main(["stats", str(tmp_path / "out")]) == 2


REPOSITORY = Path(build.__file__).resolve().parents[2]


def _fake_worktree(root, *, producer=False):
    (root / ".git").mkdir(parents=True)
    if producer:
        (root / "tools" / "site_universe").mkdir(parents=True)
    return root


def test_output_inside_the_repository_is_refused_before_anything_is_written(tmp_path, capsys):
    inside = REPOSITORY / "tools" / "site_universe" / "_refused_output_never_created"
    try:
        with pytest.raises(build.BuildError, match="inside the repository worktree"):
            build.check_output_dir(inside)
        with pytest.raises(build.BuildError, match="inside the repository worktree"):
            build.build(["TX"], out_dir=inside, cache=seeded_cache(tmp_path / "raw"))
        with pytest.raises(build.BuildError, match="inside the repository worktree"):
            build.build(["TX"], out_dir=tmp_path / "out", raw_dir=inside / "raw",
                        cache=seeded_cache(tmp_path / "raw2"))
        for argv in (
            ["build", "--states", "TX", "--out", str(inside), "--offline"],
            ["build", "--states", "TX", "--out", str(tmp_path / "ok"), "--raw-dir", str(inside), "--offline"],
            ["import-raw", "--source", "osha_ita", "--file", str(tmp_path / "x.csv"), "--raw-dir", str(inside)],
        ):
            assert main(argv) == 1
            assert "inside the repository worktree" in capsys.readouterr().err
        assert not inside.exists()
    finally:
        if inside.exists():
            shutil.rmtree(inside)
    assert build.check_output_dir(tmp_path / "out") == (tmp_path / "out").resolve()


def test_other_checkouts_of_the_repository_are_refused(tmp_path, monkeypatch):
    main_tree = _fake_worktree(tmp_path / "main")
    linked = tmp_path / "linked"
    gitdir = main_tree / ".git" / "worktrees" / "linked"
    gitdir.mkdir(parents=True)
    (gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    clone = _fake_worktree(tmp_path / "clone", producer=True)
    unrelated = _fake_worktree(tmp_path / "home")
    monkeypatch.setattr(build, "_code_worktree", lambda: main_tree)
    for refused in (main_tree / "out", linked / "out" / "deeper", clone / "out"):
        with pytest.raises(build.BuildError, match="inside the repository worktree"):
            build.check_output_dir(refused)
    assert build.check_output_dir(unrelated / "data") == (unrelated / "data").resolve()
    assert build.check_output_dir(tmp_path / "scratch") == (tmp_path / "scratch").resolve()


def test_outputs_are_git_ignored():
    ignored = (REPOSITORY / ".gitignore").read_text(encoding="utf-8").splitlines()
    for name in ("sites.jsonl.gz", "ranked.jsonl.gz", "review-top.md"):
        assert name in ignored, name


def test_cli_import_raw_validates_the_file(tmp_path, capsys):
    good = tmp_path / "ita.csv"
    good.write_bytes((Path(__file__).parent / "fixtures/site_universe/osha_ita_300a.csv").read_bytes())
    assert main(["import-raw", "--source", "osha_ita", "--file", str(good), "--raw-dir", str(tmp_path / "raw"),
                 "--retrieved-at", "2026-10-04T08:00:00Z"]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row["acquisition"] == "manual_import" and row["source_id"] == "osha_ita"
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"name,address\nAcme,1 Main\n")
    assert main(["import-raw", "--source", "osha_ita", "--file", str(bad), "--raw-dir", str(tmp_path / "raw2")]) == 1
