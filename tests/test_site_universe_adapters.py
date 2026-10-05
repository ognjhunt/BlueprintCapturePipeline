"""Source adapters on hermetic fixtures: canonical records, dropped personal fields, fail-closed headers."""

import io
import json
import zipfile

import pytest

from tests.site_universe_fixture import FIXTURES, PERSONAL_STRINGS, frs_zip_bytes
from tools.site_universe.adapters import AdapterError, epa_frs, fsis_mpi, osha_ita, osm_overpass

SHA = "a" * 64
AT = "2026-10-04T12:00:00Z"
CANONICAL_FIELDS = {
    "source_id", "source_record_id", "name", "operator", "street", "city", "state", "postal_code",
    "country", "lat", "lon", "naics", "category", "employees", "building_area_m2", "retrieved_at",
    "raw_sha256",
}


def _by_id(result):
    return {record.source_record_id: record for record in result.records}


def _no_personal_strings(result):
    text = json.dumps([record.to_dict() for record in result.records], sort_keys=True)
    for value in PERSONAL_STRINGS:
        assert value not in text, value


def test_epa_frs_reads_only_allowed_members(monkeypatch):
    opened = []
    original = zipfile.ZipFile.open

    def spy(self, name, *args, **kwargs):
        opened.append(name if isinstance(name, str) else name.filename)
        return original(self, name, *args, **kwargs)

    raw = frs_zip_bytes()
    monkeypatch.setattr(zipfile.ZipFile, "open", spy)
    epa_frs.parse(raw, state="TX", raw_sha256=SHA, retrieved_at=AT)
    assert sorted(set(opened)) == [
        "TX_ENVIRONMENTAL_INTEREST_FILE.CSV", "TX_FACILITY_FILE.CSV", "TX_NAICS_FILE.CSV", "TX_SIC_FILE.CSV",
    ]


def test_epa_frs_records():
    result = epa_frs.parse(frs_zip_bytes(), state="TX", raw_sha256=SHA, retrieved_at=AT)
    records = _by_id(result)
    assert result.stats["dropped_other_state"] == 1
    assert result.stats["dropped_person_name"] == 1
    assert "dropped_no_industry_code" not in result.stats
    assert "110000000005" not in records and "110000000008" not in records
    acme = records["110000000001"]
    assert set(acme.to_dict()) >= CANONICAL_FIELDS | {"unit", "attributes", "site_type_matches"}
    assert (acme.street, acme.city, acme.state, acme.postal_code) == ("100 INDUSTRIAL BLVD", "DALLAS", "TX", "75201")
    assert (acme.lat, acme.lon, acme.naics, acme.country) == (32.78, -96.8, "493120", "US")
    assert acme.attributes["activity_status"] == "active"
    assert acme.attributes["last_activity_year"] == 2026
    assert acme.attributes["coordinate_precision"] == "precise"
    assert acme.attributes["code_system"] == "naics"
    assert acme.attributes["program_systems"] == ["RCRAINFO", "TX-TCEQ ACR"]
    assert acme.employees is None and acme.building_area_m2 is None
    bolt = records["110000000002"]
    assert (bolt.street, bolt.unit) == ("100 INDUSTRIAL BLVD", "STE 5")
    assert bolt.attributes["activity_status"] == "inactive"
    poultry = records["110000000004"]
    assert poultry.street == "2200 FM 1960 W"
    assert poultry.codes == ("311615", "493120") and poultry.attributes["primary_code"] == "311615"
    tool = records["110000000010"]
    assert tool.code_system == "sic" and tool.codes == ("3599",) and tool.naics is None
    assert tool.city == "FORT WORTH"
    mrf = records["110000000006"]
    assert mrf.street is None
    assert mrf.attributes["location_description"] == "8 MI N OF ALVIN ON STATE HWY 35"
    assert mrf.attributes["last_activity_year"] == 2020
    linen = records["110000000012"]
    assert linen.attributes["coordinate_precision"] == "approximate"
    assert records["110000000011"].postal_code is None
    _no_personal_strings(result)


def test_epa_frs_refuses_a_zip_without_the_state_files():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("OK_FACILITY_FILE.CSV", "REGISTRY_ID\n1\n")
    with pytest.raises(AdapterError):
        epa_frs.parse(buffer.getvalue(), state="TX", raw_sha256=SHA, retrieved_at=AT)
    with pytest.raises(AdapterError):
        epa_frs.parse(b"not a zip", state="TX", raw_sha256=SHA, retrieved_at=AT)


def _osm_payload(group):
    return json.dumps(json.loads((FIXTURES / "osm_overpass_tx.json").read_text())[group]).encode()


def test_osm_records_footprints_and_tag_allowlist():
    result = osm_overpass.parse(_osm_payload("buildings"), state="TX", raw_sha256=SHA, retrieved_at=AT)
    records = _by_id(result)
    assert result.stats["osm_base_timestamp"] == "2026-10-04T00:00:00Z"
    acme = records["way/1001"]
    assert acme.building_area_m2 == pytest.approx(133.4 * 374.3, rel=0.01)
    assert acme.footprint is not None and acme.operator == "Acme Cold Storage LLC"
    assert "phone" not in acme.attributes["osm_tags"]
    foundry = records["relation/1009"]
    # Outer ring split across two member ways, minus the inner ring.
    assert foundry.building_area_m2 == pytest.approx(10_598 - 424, rel=0.01)
    assert "email" not in foundry.attributes["osm_tags"]
    assert foundry.lat == pytest.approx(31.0005, abs=1e-4)
    _no_personal_strings(result)


def test_osm_drops_homes_and_reads_addresses():
    retail = osm_overpass.parse(_osm_payload("retail"), state="TX", raw_sha256=SHA, retrieved_at=AT)
    assert retail.stats["dropped_residential_tag"] == 1
    heb = _by_id(retail)["way/1003"]
    assert (heb.street, heb.city, heb.postal_code, heb.state) == ("500 MAIN ST", "AUSTIN", "78701", "TX")
    areas = osm_overpass.parse(_osm_payload("industrial_areas"), state="TX", raw_sha256=SHA, retrieved_at=AT)
    park = _by_id(areas)["way/1005"]
    assert park.building_area_m2 is None and park.attributes["site_area_m2"] > 1_000_000


def test_osm_partial_response_is_refused(tmp_path):
    raw = (FIXTURES / "osm_overpass_partial.json").read_bytes()
    with pytest.raises(AdapterError, match="partial"):
        osm_overpass.parse(raw, state="TX", raw_sha256=SHA, retrieved_at=AT)
    path = tmp_path / "partial.json"
    path.write_bytes(raw)
    with pytest.raises(AdapterError):
        osm_overpass.validate_payload(path)


def test_osha_ita_drops_ein_and_injury_fields():
    raw = (FIXTURES / "osha_ita_300a.csv").read_bytes()
    result = osha_ita.parse(raw, state="TX", raw_sha256=SHA, retrieved_at=AT)
    records = _by_id(result)
    assert sorted(records) == ["500001", "500002", "500003", "500005"]
    assert result.stats["dropped_other_state"] == 1
    assert result.stats["dropped_superseded_row"] == 1
    assert result.stats["operator_person_name_removed"] == 1
    poultry = records["500001"]
    assert (poultry.street, poultry.employees, poultry.naics) == ("2200 FM 1960 W", 450, "311615")
    assert poultry.operator == "Lone Pine Foods Inc"
    assert poultry.attributes["osha_size_band"] == "250+"
    assert poultry.attributes["establishment_type"] == "private"
    bolt = records["500003"]
    assert bolt.employees == 40 and bolt.operator is None and bolt.unit == "STE 5"
    for record in result.records:
        flat = json.dumps(record.to_dict())
        for field in ("ein", "change_reason", "total_hours_worked", "total_injuries", "created_timestamp"):
            assert field not in flat
    _no_personal_strings(result)


def test_osha_ita_accepts_a_zip_and_refuses_an_unknown_header():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("ITA_300A.csv", (FIXTURES / "osha_ita_300a.csv").read_bytes())
    result = osha_ita.parse(buffer.getvalue(), state="TX", raw_sha256=SHA, retrieved_at=AT)
    assert result.stats["records"] == 4
    with pytest.raises(AdapterError, match="header lacks"):
        osha_ita.parse(b"name,address\nAcme,1 Main\n", state="TX", raw_sha256=SHA, retrieved_at=AT)


def test_fsis_drops_phone_and_sole_proprietor_names():
    raw = (FIXTURES / "fsis_mpi.csv").read_bytes()
    result = fsis_mpi.parse(raw, state="TX", raw_sha256=SHA, retrieved_at=AT)
    records = _by_id(result)
    assert sorted(records) == ["M1234+P1234", "M9012"]
    assert result.stats["dropped_person_name"] == 1
    assert result.stats["company_person_name_replaced"] == 1
    assert records["M9012"].name == "Thistlemoor Custom Meats"
    plant = records["M1234+P1234"]
    assert plant.street == "2200 FM 1960 W"
    assert plant.attributes["fsis_activities"] == ["Meat Processing", "Poultry Slaughter"]
    _no_personal_strings(result)


def test_fsis_refuses_an_undocumented_header():
    with pytest.raises(AdapterError, match="documented columns"):
        fsis_mpi.parse(b"Establishment,Address\nX,Y\n", state="TX", raw_sha256=SHA, retrieved_at=AT)
