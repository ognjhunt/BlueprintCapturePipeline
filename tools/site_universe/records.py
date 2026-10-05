"""Canonical source records shared by every adapter."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from tools.site_universe import normalize

# Business words that make a short name clearly not a person's name.
_BUSINESS_WORDS = frozenset(
    {
        "ABATTOIR", "ASSOCIATES", "BAKERY", "BBQ", "BEEF", "BROS", "BROTHERS", "BUTCHER", "CATTLE",
        "CENTER", "CO", "COMPANY", "CORP", "CUSTOM", "DELI", "DIST", "DISTRIBUTING", "DISTRIBUTION",
        "ENTERPRISES", "FARM", "FARMS", "FOOD", "FOODS", "GROCERY", "GROUP", "HOSPITAL", "INC",
        "INDUSTRIES", "JERKY", "LLC", "LLP", "LP", "LTD", "MANUFACTURING", "MARKET", "MEAT",
        "MEATS", "MFG", "PACKERS", "PACKING", "PLANT", "PORK", "POULTRY", "PROCESSING", "PRODUCTS",
        "RANCH", "SAUSAGE", "SERVICES", "SMOKEHOUSE", "SONS", "STORE", "SUPPLY", "TRADING",
        "WAREHOUSE", "WHOLESALE", "WORKS",
    }
)
_BUSINESS_WORDS = (
    _BUSINESS_WORDS | normalize.WEAK_NAME_TOKENS | normalize.GENERIC_WORDS | normalize.LEGAL_WORDS
)
_WORD = r"[A-Z][a-z'\-]+"
_INITIAL = r"[A-Z]\.?"
_PERSON_COMMA = re.compile(rf"^{_WORD}, {_WORD}( {_INITIAL})?$")
_PERSON_PLAIN = re.compile(rf"^{_WORD} ({_INITIAL} )?{_WORD}$")


def looks_like_person_name(value: str | None, *, plain: bool = False) -> bool:
    """Conservative check for a bare personal name such as 'Smith, John A'.

    It flags only two to four alphabetic words with no business words, no
    digits and no '&'. Business names that contain a surname
    ('Smith Meat Processing') are kept. With ``plain`` it also flags the
    mixed-case 'John A Smith' form; that form also matches some two-word
    brands, so adapters use it only for owner and company fields.
    """
    text = normalize.clean_text(value)
    if not text or any(ch.isdigit() for ch in text) or "&" in text:
        return False
    words = re.sub(r"[.,]", " ", text).upper().split()
    if not 2 <= len(words) <= 4 or any(word in _BUSINESS_WORDS for word in words):
        return False
    if "," in text:
        return bool(_PERSON_COMMA.match(" ".join(text.title().split())))
    # Without a comma only flag mixed-case "First M Last" forms; all-caps strings are usually brands.
    return plain and text != text.upper() and bool(_PERSON_PLAIN.match(text))


@dataclass
class SourceRecord:
    source_id: str
    source_record_id: str
    name: str | None
    operator: str | None
    street: str | None
    unit: str | None
    city: str | None
    state: str | None
    postal_code: str | None
    country: str | None
    lat: float | None
    lon: float | None
    naics: str | None
    category: str | None
    employees: int | None
    building_area_m2: float | None
    retrieved_at: str
    raw_sha256: str
    attributes: dict = field(default_factory=dict)
    site_types: tuple = ()
    site_type_matches: dict = field(default_factory=dict)
    # In-memory only: (outer rings, inner rings) of (lat, lon). Never serialized.
    footprint: tuple | None = None
    # Industry codes used for classification (NAICS, or SIC when no NAICS).
    codes: tuple = ()
    code_system: str = "naics"

    def to_dict(self) -> dict:
        return {
            "attributes": dict(sorted(self.attributes.items())),
            "building_area_m2": self.building_area_m2,
            "category": self.category,
            "city": self.city,
            "country": self.country,
            "employees": self.employees,
            "lat": self.lat,
            "lon": self.lon,
            "name": self.name,
            "naics": self.naics,
            "operator": self.operator,
            "postal_code": self.postal_code,
            "raw_sha256": self.raw_sha256,
            "retrieved_at": self.retrieved_at,
            "site_type_matches": {
                key: list(value) for key, value in sorted(self.site_type_matches.items())
            },
            "source_id": self.source_id,
            "source_record_id": self.source_record_id,
            "state": self.state,
            "street": self.street,
            "unit": self.unit,
        }

    @property
    def ref(self) -> str:
        return f"{self.source_id}:{self.source_record_id}"


def round_coordinate(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 6)


def valid_us_coordinates(lat: float | None, lon: float | None) -> bool:
    """Rough bounding box for the US and territories; rejects 0,0 and swapped signs."""
    if lat is None or lon is None:
        return False
    return 13.0 <= lat <= 72.0 and -180.0 <= lon <= -64.0


def parse_int(value) -> int | None:
    text = normalize.clean_text(value)
    if text is None:
        return None
    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return round(number)
