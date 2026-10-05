"""OSHA ITA Form 300A establishment summary adapter (pure).

The publisher refuses automated clients, so the raw file arrives by manual
import. Columns follow the OSHA 300A summary data dictionary (April 2024).
The EIN, the free-text change reason, the submission timestamp, hours worked
and every injury and illness count are dropped at parse time.
"""

from __future__ import annotations

import csv
import io
import zipfile

from tools.site_universe import normalize
from tools.site_universe.adapters import AdapterError, ParseResult
from tools.site_universe.records import SourceRecord, looks_like_person_name, parse_int

SOURCE_ID = "osha_ita"
REQUIRED_COLUMNS = (
    "establishment_name", "street_address", "city", "state", "zip_code", "naics_code",
    "annual_average_employees",
)
SIZE_BANDS = {"1": "<20", "2": "20-249", "21": "20-99", "22": "100-249", "3": "250+"}
ESTABLISHMENT_TYPES = {"1": "private", "2": "state_government", "3": "local_government"}


def _text(raw: bytes) -> str:
    if raw[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(members) != 1:
                raise AdapterError(f"OSHA ITA zip must hold exactly one CSV, found {members}")
            raw = archive.read(members[0])
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def parse(raw: bytes, *, state: str, raw_sha256: str, retrieved_at: str) -> ParseResult:
    state = normalize.normalize_state(state)
    reader = csv.reader(io.StringIO(_text(raw), newline=""))
    try:
        header = [column.strip().lower() for column in next(reader)]
    except StopIteration as error:
        raise AdapterError("OSHA ITA file is empty") from error
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise AdapterError(f"OSHA ITA header lacks {missing}; header was {header}")
    index = {column: position for position, column in enumerate(header)}
    id_column = "establishment_id" if "establishment_id" in index else "id"
    if id_column not in index:
        raise AdapterError("OSHA ITA header has neither establishment_id nor id")

    def cell(row, column):
        position = index.get(column)
        return row[position] if position is not None and position < len(row) else ""

    result = ParseResult()
    latest: dict[str, tuple] = {}
    for row in reader:
        result.count("rows_read")
        if normalize.normalize_state(cell(row, "state")) != state:
            result.count("dropped_other_state")
            continue
        record_id = normalize.clean_text(cell(row, id_column))
        if not record_id:
            result.count("dropped_no_id")
            continue
        order = (parse_int(cell(row, "year_filing_for")) or 0, parse_int(cell(row, "id")) or 0)
        if record_id in latest and latest[record_id][0] >= order:
            result.count("dropped_superseded_row")
            continue
        if record_id in latest:
            result.count("dropped_superseded_row")
        latest[record_id] = (order, row)
    for record_id in sorted(latest):
        row = latest[record_id][1]
        name = normalize.clean_text(cell(row, "establishment_name"))
        if not name or looks_like_person_name(name):
            result.count("dropped_no_name" if not name else "dropped_person_name")
            continue
        operator = normalize.clean_text(cell(row, "company_name"))
        if operator and looks_like_person_name(operator, plain=True):
            result.count("operator_person_name_removed")
            operator = None
        street, unit = normalize.normalize_street(cell(row, "street_address"))
        naics = "".join(ch for ch in cell(row, "naics_code") if ch.isdigit()) or None
        attributes = {
            "establishment_type": ESTABLISHMENT_TYPES.get(cell(row, "establishment_type").strip()),
            "industry_description": normalize.clean_text(cell(row, "industry_description")),
            "naics_year": parse_int(cell(row, "naics_year")),
            "osha_size_band": SIZE_BANDS.get(cell(row, "size").strip()),
            "year_filing_for": parse_int(cell(row, "year_filing_for")),
        }
        result.records.append(
            SourceRecord(
                source_id=SOURCE_ID,
                source_record_id=record_id,
                name=name,
                operator=operator,
                street=street,
                unit=unit,
                city=normalize.normalize_city(cell(row, "city")),
                state=state,
                postal_code=normalize.normalize_postal(cell(row, "zip_code")),
                country="US",
                lat=None,
                lon=None,
                naics=naics,
                category=None,
                employees=parse_int(cell(row, "annual_average_employees")),
                building_area_m2=None,
                retrieved_at=retrieved_at,
                raw_sha256=raw_sha256,
                attributes=attributes,
                codes=(naics,) if naics else (),
                code_system="naics",
            )
        )
        result.count("records")
    return result
