"""USDA FSIS Meat, Poultry and Egg Product Inspection Directory adapter (pure).

The publisher refuses automated clients and the exact CSV header could not be
read on 2026-10-04. The adapter therefore accepts only a header that has the
documented columns (EstNumber, Company, Street, City, State, Zip; any case,
spaces and underscores ignored) and refuses anything else. The phone number is
dropped. A company name that looks like a person's name (a sole proprietor) is
replaced by the first DBA name that is not one, or the record is dropped.
"""

from __future__ import annotations

import csv
import io
import re

from tools.site_universe import normalize
from tools.site_universe.adapters import AdapterError, ParseResult
from tools.site_universe.records import (
    SourceRecord,
    looks_like_person_name,
    round_coordinate,
    valid_us_coordinates,
)

SOURCE_ID = "fsis_mpi"
REQUIRED_COLUMNS = ("estnumber", "company", "street", "city", "state", "zip")


def _key(column: str) -> str:
    return re.sub(r"[\s_]+", "", column.strip().lower())


def _split(value: str | None) -> list[str]:
    text = normalize.clean_text(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r"[;|]", text) if part.strip()]


def parse(raw: bytes, *, state: str, raw_sha256: str, retrieved_at: str) -> ParseResult:
    state = normalize.normalize_state(state)
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = [_key(column) for column in next(reader)]
    except StopIteration as error:
        raise AdapterError("FSIS MPI file is empty") from error
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise AdapterError(f"FSIS MPI header lacks documented columns {missing}; header was {header}")
    index = {column: position for position, column in enumerate(header)}

    def cell(row, column):
        position = index.get(column)
        return row[position] if position is not None and position < len(row) else ""

    result = ParseResult()
    seen: set[str] = set()
    for row in reader:
        result.count("rows_read")
        if normalize.normalize_state(cell(row, "state")) != state:
            result.count("dropped_other_state")
            continue
        number = normalize.clean_text(cell(row, "estnumber"))
        if not number or number in seen:
            result.count("dropped_duplicate_or_no_id")
            continue
        seen.add(number)
        dbas = [item for item in _split(cell(row, "dbas")) if not looks_like_person_name(item, plain=True)]
        name = normalize.clean_text(cell(row, "company"))
        if name and looks_like_person_name(name, plain=True):
            result.count("company_person_name_replaced" if dbas else "dropped_person_name")
            name = dbas[0] if dbas else None
            if not name:
                continue
        if not name:
            result.count("dropped_no_name")
            continue
        street, unit = normalize.normalize_street(cell(row, "street"))
        lat = round_coordinate(cell(row, "latitude"))
        lon = round_coordinate(cell(row, "longitude"))
        if not valid_us_coordinates(lat, lon):
            lat = lon = None
        attributes = {
            "fsis_activities": _split(cell(row, "activities")),
            "fsis_dbas": dbas,
            "fsis_establishment_number": number,
            "fsis_grant_date": normalize.clean_text(cell(row, "grantdate")),
            "fsis_size": normalize.clean_text(cell(row, "size")),
        }
        result.records.append(
            SourceRecord(
                source_id=SOURCE_ID,
                source_record_id=number,
                name=name,
                operator=None,
                street=street,
                unit=unit,
                city=normalize.normalize_city(cell(row, "city")),
                state=state,
                postal_code=normalize.normalize_postal(cell(row, "zip")),
                country="US",
                lat=lat,
                lon=lon,
                naics=None,
                category=None,
                employees=None,
                building_area_m2=None,
                retrieved_at=retrieved_at,
                raw_sha256=raw_sha256,
                attributes=attributes,
                codes=(),
                code_system="fsis",
            )
        )
        result.count("records")
    return result
