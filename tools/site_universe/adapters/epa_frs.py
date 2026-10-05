"""EPA Facility Registry Service state combined CSV adapter (pure).

Only four members of the state zip are opened: the facility file, the NAICS
file, the SIC file and the environmental interest file. The contact,
organization, mailing address and program files hold person names, phone
numbers, email addresses, EINs and user ids, and are never read.
"""

from __future__ import annotations

import csv
import io
import zipfile
from collections import Counter, defaultdict

from tools.site_universe import normalize
from tools.site_universe.adapters import AdapterError, ParseResult
from tools.site_universe.records import (
    SourceRecord,
    looks_like_person_name,
    round_coordinate,
    valid_us_coordinates,
)

SOURCE_ID = "epa_frs"
URL_TEMPLATE = "https://ordsext.epa.gov/FLA/www3/state_files/state_combined_{state_lower}.zip"
MEMBER_SUFFIXES = ("FACILITY_FILE.CSV", "NAICS_FILE.CSV", "SIC_FILE.CSV", "ENVIRONMENTAL_INTEREST_FILE.CSV")
FACILITY_COLUMNS = (
    "REGISTRY_ID", "PRIMARY_NAME", "LOCATION_ADDRESS", "SUPPLEMENTAL_LOCATION", "CITY_NAME",
    "COUNTY_NAME", "STATE_CODE", "POSTAL_CODE", "LATITUDE83", "LONGITUDE83",
)
CODE_COLUMNS = ("REGISTRY_ID", "PRIMARY_INDICATOR")
INTEREST_COLUMNS = ("REGISTRY_ID", "ACTIVE_STATUS", "LAST_REPORTED_DATE", "UPDATE_DATE")
LIVE_STATUSES = frozenset({"ACTIVE", "EFFECTIVE", "OPERATING", "Y"})
DEAD_STATUSES = frozenset({"TERMINATED", "EXPIRED", "INACTIVE", "N", "CLOSED"})


APPROXIMATE_METHODS = ("CENTROID", "CENSUS", "INTERSECTION", "UNKNOWN")
APPROXIMATE_ACCURACY_M = 150.0


def _precision(method: str | None, accuracy_m: float | None) -> str:
    """'approximate' coordinates never drive a proximity merge."""
    upper = (method or "").upper()
    if any(word in upper for word in APPROXIMATE_METHODS):
        return "approximate"
    if accuracy_m is not None and accuracy_m > APPROXIMATE_ACCURACY_M:
        return "approximate"
    return "precise" if method or accuracy_m is not None else "unknown"


def download_url(state: str) -> str:
    return URL_TEMPLATE.format(state_lower=state.lower())


def _rows(archive: zipfile.ZipFile, member: str, required: tuple[str, ...]):
    with archive.open(member) as handle:
        text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text)
        header = set(reader.fieldnames or ())
        missing = [column for column in required if column not in header]
        if missing:
            raise AdapterError(f"{member}: missing columns {missing}")
        yield from reader


def _member(archive: zipfile.ZipFile, state: str, suffix: str) -> str:
    wanted = f"{state}_{suffix}"
    for name in archive.namelist():
        if name.upper() == wanted:
            return name
    raise AdapterError(f"EPA FRS zip has no {wanted}")


_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def _year(text: str, pivot: int) -> int | None:
    """Year of a 'DD-MON-YY' date; two-digit years above ``pivot`` belong to the 1900s."""
    parts = (text or "").strip().upper().split("-")
    if len(parts) != 3 or parts[1] not in _MONTHS or not (parts[0].isdigit() and parts[2].isdigit()):
        return None
    if len(parts[2]) != 2 or not 1 <= int(parts[0]) <= 31:
        return None
    year = 2000 + int(parts[2])
    if year > pivot:
        year -= 100
    return year


def _codes(archive, state: str, suffix: str, code_column: str, skip=None):
    by_registry: dict[str, Counter] = defaultdict(Counter)
    primary: dict[str, Counter] = defaultdict(Counter)
    for row in _rows(archive, _member(archive, state, suffix), CODE_COLUMNS + (code_column,)):
        registry_id = row["REGISTRY_ID"].strip()
        code = "".join(ch for ch in row[code_column] if ch.isdigit())
        if not registry_id or len(code) < 2 or (skip is not None and registry_id in skip):
            continue
        by_registry[registry_id][code] += 1
        if row["PRIMARY_INDICATOR"].strip().upper() == "PRIMARY":
            primary[registry_id][code] += 1
    return by_registry, primary


def _primary(codes: Counter, primary: Counter) -> str:
    pool = primary or codes
    return min(pool.items(), key=lambda item: (-item[1], item[0]))[0]


def parse(raw: bytes, *, state: str, raw_sha256: str, retrieved_at: str) -> ParseResult:
    state = normalize.normalize_state(state)
    if not state:
        raise AdapterError("a two-letter state code is required")
    pivot = int(retrieved_at[:4]) + 1
    result = ParseResult()
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as error:
        raise AdapterError(f"EPA FRS raw file is not a zip: {error}") from error
    with archive:
        naics, naics_primary = _codes(archive, state, "NAICS_FILE.CSV", "NAICS_CODE")
        sic, sic_primary = _codes(archive, state, "SIC_FILE.CSV", "SIC_CODE", skip=naics)
        coded = set(naics) | set(sic)
        statuses: dict[str, set] = defaultdict(set)
        last_year: dict[str, int] = {}
        interest = _member(archive, state, "ENVIRONMENTAL_INTEREST_FILE.CSV")
        for row in _rows(archive, interest, INTEREST_COLUMNS):
            registry_id = row["REGISTRY_ID"].strip()
            if registry_id not in coded:
                continue
            status = row["ACTIVE_STATUS"].strip().upper()
            if status and not status.startswith("*"):
                statuses[registry_id].add(status)
            for column in ("LAST_REPORTED_DATE", "UPDATE_DATE"):
                year = _year(row[column], pivot)
                if year and year > last_year.get(registry_id, 0):
                    last_year[registry_id] = year
        facility = _member(archive, state, "FACILITY_FILE.CSV")
        for row in _rows(archive, facility, FACILITY_COLUMNS):
            result.count("rows_read")
            registry_id = row["REGISTRY_ID"].strip()
            if normalize.normalize_state(row["STATE_CODE"]) != state:
                result.count("dropped_other_state")
                continue
            if registry_id in naics:
                codes, system = naics[registry_id], "naics"
                primary = _primary(codes, naics_primary.get(registry_id, Counter()))
            elif registry_id in sic:
                codes, system = sic[registry_id], "sic"
                primary = _primary(codes, sic_primary.get(registry_id, Counter()))
            else:
                result.count("dropped_no_industry_code")
                continue
            name = normalize.clean_text(row["PRIMARY_NAME"])
            if not name:
                result.count("dropped_no_name")
                continue
            if looks_like_person_name(name):
                result.count("dropped_person_name")
                continue
            street, unit = normalize.normalize_street(
                row["LOCATION_ADDRESS"], row.get("SUPPLEMENTAL_LOCATION")
            )
            description = None
            if street is None and normalize.is_descriptive_location(row["LOCATION_ADDRESS"]):
                description = normalize.clean_text(row["LOCATION_ADDRESS"])
            lat = round_coordinate(row["LATITUDE83"])
            lon = round_coordinate(row["LONGITUDE83"])
            if (lat is not None or lon is not None) and not valid_us_coordinates(lat, lon):
                result.count("coordinates_rejected")
                lat = lon = None
            year = last_year.get(registry_id)
            for column in ("UPDATE_DATE", "CREATE_DATE"):
                candidate = _year(row.get(column, ""), pivot)
                if candidate and (year is None or candidate > year):
                    year = candidate
            seen = statuses.get(registry_id, set())
            if seen & LIVE_STATUSES:
                activity = "active"
            elif seen and seen <= DEAD_STATUSES:
                activity = "inactive"
            else:
                activity = "unknown"
            attributes = {
                "activity_status": activity,
                "code_system": system,
                "county": normalize.clean_text(row["COUNTY_NAME"]),
                "frs_registry_id": registry_id,
                "location_description": description,
                "last_activity_year": year,
                f"{system}_codes": sorted(codes),
                "primary_code": primary,
                "program_systems": sorted(
                    {
                        item.split(":", 1)[0].strip()
                        for item in (row.get("PGM_SYS_ACRNMS") or "").split(",")
                        if item.strip()
                    }
                ),
            }
            accuracy = normalize.clean_text(row.get("ACCURACY_VALUE"))
            accuracy_m = float(accuracy) if accuracy and accuracy.replace(".", "", 1).isdigit() else None
            method = normalize.clean_text(row.get("COLLECT_DESC"))
            if lat is not None:
                if accuracy_m is not None:
                    attributes["coordinate_accuracy_m"] = accuracy_m
                if method:
                    attributes["coordinate_method"] = method
                attributes["coordinate_precision"] = _precision(method, accuracy_m)
            result.records.append(
                SourceRecord(
                    source_id=SOURCE_ID,
                    source_record_id=registry_id,
                    name=name,
                    operator=None,
                    street=street,
                    unit=unit,
                    city=normalize.normalize_city(row["CITY_NAME"]),
                    state=state,
                    postal_code=normalize.normalize_postal(row["POSTAL_CODE"]),
                    country="US",
                    lat=lat,
                    lon=lon,
                    naics=primary if system == "naics" else None,
                    category=None,
                    employees=None,
                    building_area_m2=None,
                    retrieved_at=retrieved_at,
                    raw_sha256=raw_sha256,
                    attributes=attributes,
                    codes=tuple(sorted(codes)),
                    code_system=system,
                )
            )
            result.count("records")
    return result
