"""Pure address and name normalization for site matching.

Addresses follow USPS Publication 28 style: uppercase ASCII, standard street
suffix and directional abbreviations, and secondary unit designators moved to a
separate ``unit`` field. Texas highway forms (FM, RM, State Highway, IH) are
folded to one spelling so that two sources that write the same road differently
produce the same address key.

Names are compared as token sets after legal suffixes, generic facility words
and the city name are removed. :func:`token_set_ratio` is a self-contained
implementation of the usual token-set ratio over a bit-parallel LCS.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import pairwise

STATE_CODES = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL",
    "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA",
    "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR",
    "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA",
    "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "PUERTO RICO": "PR",
}
STATE_NAMES = {code: name for name, code in STATE_CODES.items()}

# USPS Publication 28, Appendix C1 (common street suffixes).
STREET_SUFFIXES = {
    "ALLEY": "ALY", "ANNEX": "ANX", "ARCADE": "ARC", "AVENUE": "AVE", "AVENU": "AVE", "AV": "AVE",
    "BAYOU": "BYU", "BEND": "BND", "BLUFF": "BLF", "BOULEVARD": "BLVD", "BOULV": "BLVD",
    "BRANCH": "BR", "BRIDGE": "BRG", "BYPASS": "BYP", "CAUSEWAY": "CSWY", "CENTER": "CTR",
    "CIRCLE": "CIR", "CIRCL": "CIR", "CRCL": "CIR", "COURT": "CT", "COURTS": "CTS", "COVE": "CV",
    "CREEK": "CRK", "CROSSING": "XING", "CRSSNG": "XING", "DRIVE": "DR", "DRV": "DR",
    "ESTATE": "EST", "ESTATES": "ESTS", "EXPRESSWAY": "EXPY", "EXPRESS": "EXPY", "EXPW": "EXPY",
    "EXTENSION": "EXT", "FREEWAY": "FWY", "FRWY": "FWY", "GARDENS": "GDNS", "GATEWAY": "GTWY",
    "GROVE": "GRV", "HARBOR": "HBR", "HEIGHTS": "HTS", "HIGHWAY": "HWY", "HIWAY": "HWY",
    "HOLLOW": "HOLW", "JUNCTION": "JCT", "LAKE": "LK", "LAKES": "LKS", "LANDING": "LNDG",
    "LANE": "LN", "MANOR": "MNR", "MEADOWS": "MDWS", "MOTORWAY": "MTWY", "PARKWAY": "PKWY",
    "PARKWY": "PKWY", "PKY": "PKWY", "PASSAGE": "PSGE", "PLACE": "PL", "PLAZA": "PLZ",
    "POINT": "PT", "PORT": "PRT", "PRAIRIE": "PR", "RANCH": "RNCH", "RIDGE": "RDG",
    "ROAD": "RD", "ROUTE": "RTE", "SQUARE": "SQ", "STATION": "STA", "STREET": "ST", "STR": "ST",
    "TERRACE": "TER", "TRACE": "TRCE", "TRAFFICWAY": "TRFY", "TRAIL": "TRL", "TRAILS": "TRL",
    "TURNPIKE": "TPKE", "VALLEY": "VLY", "VIEW": "VW", "VILLAGE": "VLG", "VISTA": "VIS",
    "WALK": "WALK", "WAY": "WAY", "WELLS": "WLS",
}
DIRECTIONALS = {
    "NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W", "NORTHEAST": "NE", "NORTHWEST": "NW",
    "SOUTHEAST": "SE", "SOUTHWEST": "SW", "NO": "N", "SO": "S",
}
ORDINAL_WORDS = {
    "FIRST": "1ST", "SECOND": "2ND", "THIRD": "3RD", "FOURTH": "4TH", "FIFTH": "5TH",
    "SIXTH": "6TH", "SEVENTH": "7TH", "EIGHTH": "8TH", "NINTH": "9TH", "TENTH": "10TH",
}
# Multi-word road designators, longest first. Texas forms fold to one spelling.
ROAD_DESIGNATORS = (
    ("FARM TO MARKET ROAD", "FM"), ("FARM TO MARKET RD", "FM"), ("FARM TO MARKET", "FM"),
    ("FARM ROAD", "FM"), ("FM ROAD", "FM"), ("FM RD", "FM"), ("F M", "FM"),
    ("RANCH TO MARKET ROAD", "RM"), ("RANCH TO MARKET RD", "RM"), ("RANCH TO MARKET", "RM"),
    ("RM ROAD", "RM"), ("RM RD", "RM"), ("RANCH ROAD", "RR"),
    ("STATE HIGHWAY", "STATE HWY"), ("STATE HWY", "STATE HWY"), ("ST HWY", "STATE HWY"),
    ("TEXAS HIGHWAY", "STATE HWY"), ("TX HWY", "STATE HWY"), ("SH", "STATE HWY"),
    ("STATE ROAD", "STATE RD"), ("STATE ROUTE", "STATE RTE"),
    ("UNITED STATES HIGHWAY", "US HWY"), ("U S HIGHWAY", "US HWY"), ("U S HWY", "US HWY"),
    ("US HIGHWAY", "US HWY"), ("US HWY", "US HWY"),
    ("INTERSTATE HIGHWAY", "I"), ("INTERSTATE HWY", "I"), ("INTERSTATE", "I"), ("IH", "I"),
    ("FM", "FM"), ("RM", "RM"), ("RR", "RR"), ("I", "I"),
    ("COUNTY ROAD", "COUNTY RD"), ("COUNTY RD", "COUNTY RD"), ("CO RD", "COUNTY RD"),
    ("CR", "COUNTY RD"), ("PRIVATE ROAD", "PRIVATE RD"), ("PR", "PRIVATE RD"),
)
# USPS Publication 28, Appendix C2 (secondary unit designators).
UNIT_DESIGNATORS = {
    "APARTMENT": "APT", "APT": "APT", "BASEMENT": "BSMT", "BSMT": "BSMT", "BUILDING": "BLDG",
    "BLDG": "BLDG", "BLD": "BLDG", "DEPARTMENT": "DEPT", "DEPT": "DEPT", "FLOOR": "FL", "FL": "FL",
    "FRONT": "FRNT", "HANGAR": "HNGR", "HNGR": "HNGR", "LOBBY": "LBBY", "LOT": "LOT",
    "LOWER": "LOWR", "OFFICE": "OFC", "OFC": "OFC", "PIER": "PIER", "REAR": "REAR", "ROOM": "RM",
    "RM": "RM", "SIDE": "SIDE", "SLIP": "SLIP", "SPACE": "SPC", "SPC": "SPC", "STOP": "STOP",
    "SUITE": "STE", "STE": "STE", "TRAILER": "TRLR", "TRLR": "TRLR", "UNIT": "UNIT",
    "UPPER": "UPPR", "DOCK": "DOCK", "BAY": "BAY", "GATE": "GATE", "#": "#",
}
CITY_ABBREVIATIONS = {"FT": "FORT", "MT": "MOUNT", "ST": "SAINT", "STE": "SAINTE", "PT": "PORT"}

LEGAL_WORDS = frozenset(
    {
        "INC", "INCORPORATED", "LLC", "LLP", "LP", "LTD", "LIMITED", "CORP", "CORPORATION", "CO",
        "COMPANY", "COMPANIES", "PLLC", "PC", "PA", "HOLDINGS", "HOLDING", "GROUP", "THE", "OF",
        "AND", "AT", "DBA", "USA", "US", "NA", "AMERICA", "AMERICAS",
    }
)
GENERIC_WORDS = frozenset(
    {
        "PLANT", "PLANTS", "FACILITY", "FACILITIES", "SITE", "WAREHOUSE", "WAREHOUSES",
        "WAREHOUSING", "DISTRIBUTION", "DISTRIBUTING", "DISTRIBUTOR", "DISTRIBUTORS", "CENTER",
        "CENTRE", "CTR", "DC", "FC", "BUILDING", "BLDG", "INDUSTRIES", "INDUSTRY", "INDUSTRIAL",
        "MANUFACTURING", "MFG", "MANUFACTURER", "MANUFACTURERS", "OPERATIONS", "OPS", "DIVISION",
        "DIV", "DEPT", "LOCATION", "STORE", "STORES", "SUPERCENTER", "NUMBER", "NO", "UNIT",
        "SUITE", "STE", "SERVICES", "SERVICE", "SYSTEMS", "PRODUCTS", "PRODUCT", "SUPPLY",
        "SUPPLIES", "LOGISTICS", "INTERNATIONAL", "INTL", "NATIONAL", "TEXAS", "TX", "SAINT", "ST",
        "NORTH", "SOUTH", "EAST", "WEST", "N", "S", "E", "W", "NE", "NW", "SE", "SW", "PARK",
        "COMPLEX", "CAMPUS", "WORKS",
    }
)
NAME_ABBREVIATIONS = {
    "MFG": "MANUFACTURING", "MANUF": "MANUFACTURING", "DIST": "DISTRIBUTION", "DISTR": "DISTRIBUTION",
    "WHSE": "WAREHOUSE", "WHS": "WAREHOUSE", "CTR": "CENTER", "INTL": "INTERNATIONAL",
    "NATL": "NATIONAL", "SVCS": "SERVICES", "SVC": "SERVICE", "HOSP": "HOSPITAL", "MED": "MEDICAL",
    "MEM": "MEMORIAL", "REG": "REGIONAL", "TECH": "TECHNOLOGY", "TECHNOLOGIES": "TECHNOLOGY",
    "&": "AND", "SAINT": "ST",
}

_SPACE = re.compile(r"\s+")
_JOINED_PUNCT = re.compile(r"(?<=[A-Z0-9])[-'.&/](?=[A-Z0-9])")
_PUNCT = re.compile(r"[^A-Z0-9# ]+")
_PO_BOX = re.compile(r"^(P\s*O\s*BOX|POST OFFICE BOX|BOX)\b")
_DIRECTION = r"(N|S|E|W|NE|NW|SE|SW|NORTH|SOUTH|EAST|WEST|NORTHEAST|NORTHWEST|SOUTHEAST|SOUTHWEST)"
_DESCRIPTIVE = re.compile(
    r"^\d+(\.\d+)?\s*(M|MI|MIS|MILE|MILES|KM|FT|FEET|YDS)\b"
    rf"|\b(MILES?|MI)\s+{_DIRECTION}\b"
    rf"|\b{_DIRECTION}\s+OF\b"
    r"|\b(CORNER|INTERSECTION|JUNCTION|JCT)\s+OF\b"
)
_ZIP = re.compile(r"(\d{5})")


def ascii_upper(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    return _SPACE.sub(" ", text.upper()).strip()


def clean_text(value: str | None) -> str | None:
    """Trim and collapse whitespace; empty strings become None."""
    if value is None:
        return None
    text = _SPACE.sub(" ", str(value)).strip()
    return text or None


def normalize_state(value: str | None) -> str | None:
    text = ascii_upper(value).replace(".", "")
    if not text:
        return None
    if len(text) == 2 and text.isalpha():
        return text
    return STATE_CODES.get(text)


def normalize_postal(value: str | None) -> str | None:
    """Five-digit ZIP code, or None. ZIP+4 and nine-digit forms keep the first five."""
    text = ascii_upper(value)
    if not text:
        return None
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) in (5, 9):
        zip5 = digits[:5]
    else:
        match = _ZIP.match(text)
        zip5 = match.group(1) if match and len(digits) >= 5 else None
    return None if zip5 == "00000" else zip5


def normalize_city(value: str | None) -> str | None:
    text = ascii_upper(value)
    text = _PUNCT.sub(" ", _JOINED_PUNCT.sub("", text).replace("#", " "))
    tokens = [CITY_ABBREVIATIONS.get(token, token) for token in text.split()]
    return " ".join(tokens) or None


_REDUNDANT_AFTER_ROUTE = frozenset({"RD", "ROAD", "HWY", "HIGHWAY"})


def _replace_designators(tokens: list[str]) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(tokens):
        for phrase, short in ROAD_DESIGNATORS:
            words = phrase.split()
            if tokens[index : index + len(words)] == words:
                nxt = tokens[index + len(words)] if index + len(words) < len(tokens) else ""
                # Single-letter or short forms (SH, CR, PR, IH) only count before a road number.
                if len(words) == 1 and not nxt[:1].isdigit():
                    continue
                out.extend(short.split())
                index += len(words)
                # "FM 1960 RD W" and "FM 1960 W" are one road: drop RD/HWY after the route number.
                if nxt[:1].isdigit():
                    out.append(nxt)
                    index += 1
                    if index < len(tokens) and tokens[index] in _REDUNDANT_AFTER_ROUTE:
                        index += 1
                break
        else:
            out.append(tokens[index])
            index += 1
    return out


def is_descriptive_location(value: str | None) -> bool:
    """True for directions such as '3 MI S ON STATE HWY 163' that are not street addresses."""
    return _descriptive_cut(ascii_upper(value).replace(",", " ")) is None


def _descriptive_cut(text: str) -> str | None:
    """Text before a trailing description ('4210 STATE HWY 12 W OF TOWN'), or None when the
    whole value is a description. Values without a description come back unchanged."""
    match = _DESCRIPTIVE.search(text)
    if not match:
        return text
    head = text[: match.start()].strip()
    words = head.split()
    if len(words) >= 2 and words[0][:1].isdigit() and not _DESCRIPTIVE.match(head):
        return head
    return None


def normalize_street(value: str | None, supplemental: str | None = None) -> tuple[str | None, str | None]:
    """Return ``(street, unit)`` in USPS style.

    The unit designator and everything after it move to ``unit``. A PO Box or
    a description such as '2 MI N OF TOWN' is not a street address, so the
    street is None.
    """
    text = _descriptive_cut(ascii_upper(value).replace(",", " "))
    if not text:
        return None, _normalize_unit(supplemental)
    text = text.replace(",", " ")
    text = _JOINED_PUNCT.sub(lambda m: "" if m.group(0) in "'." else " ", text)
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"#\s*", " # ", text)
    tokens = text.split()
    if not tokens or _PO_BOX.match(" ".join(tokens)):
        return None, _normalize_unit(supplemental)
    unit_tokens: list[str] = []
    for index, token in enumerate(tokens):
        if index == 0:
            continue
        if token in UNIT_DESIGNATORS and _looks_like_unit(tokens, index):
            unit_tokens = tokens[index:]
            tokens = tokens[:index]
            break
    tokens = _replace_designators(tokens)
    normalized: list[str] = []
    for index, token in enumerate(tokens):
        if token in ORDINAL_WORDS:
            token = ORDINAL_WORDS[token]
        elif token in DIRECTIONALS and _is_directional_position(tokens, index):
            token = DIRECTIONALS[token]
        elif token in STREET_SUFFIXES and index > 0 and index >= len(tokens) - 2:
            token = STREET_SUFFIXES[token]
        normalized.append(token)
    street = " ".join(normalized) or None
    unit = _normalize_unit(" ".join(unit_tokens)) if unit_tokens else None
    extra = _normalize_unit(supplemental)
    if extra and extra != unit:
        unit = f"{unit} {extra}" if unit else extra
    return street, unit


def _looks_like_unit(tokens: list[str], index: int) -> bool:
    token = tokens[index]
    nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
    if token == "#":
        return bool(nxt)
    if token in {"REAR", "FRONT", "UPPER", "LOWER", "SIDE", "LOBBY", "BASEMENT", "BSMT"}:
        return index == len(tokens) - 1
    # "LOT", "BAY", "GATE" and "DOCK" are also street name words; require a short identifier.
    return bool(nxt) and (any(ch.isdigit() for ch in nxt) or len(nxt) <= 2)


def _is_directional_position(tokens: list[str], index: int) -> bool:
    if len(tokens) == 1:
        return False
    if index == 1 and tokens[0][:1].isdigit():
        return True
    return index == len(tokens) - 1 or (index == 0 and len(tokens) > 1)


def _normalize_unit(value: str | None) -> str | None:
    text = ascii_upper(value)
    if not text:
        return None
    text = _PUNCT.sub(" ", text.replace(",", " "))
    tokens = [UNIT_DESIGNATORS.get(token, token) for token in text.split()]
    # "STE # 4" -> "STE 4": a number sign after another designator adds nothing.
    tokens = [
        token for position, token in enumerate(tokens)
        if not (token == "#" and position > 0 and tokens[position - 1] in UNIT_DESIGNATORS.values())
    ]
    return " ".join(tokens) or None


def house_number(street: str | None) -> str | None:
    if not street:
        return None
    first = street.split(" ", 1)[0]
    return first if first[:1].isdigit() else None


def address_key(street: str | None, city: str | None, state: str | None, postal: str | None) -> str | None:
    """Canonical street+city+state+postal key, or None without a numbered street."""
    if not street or not state or not house_number(street) or not (city or postal):
        return None
    return "|".join((street, city or "", state, postal or ""))


# --- names -------------------------------------------------------------------------
# Industry and place words. A shared weak word alone is not evidence that two
# names are one business (two dry cleaners in one strip mall share CLEANER).
WEAK_NAME_TOKENS = frozenset(
    {
        "ALLIED", "ALUMINUM", "AMERICAN", "ASSOCIATE", "ASSOCIATES", "AUTO", "AUTOMOTIVE", "BAKERY",
        "BAKING", "BEEF", "BEST", "BEVERAGE", "BOX", "BOXE", "BUILDER", "BUILDERS", "BUILDING",
        "CABINET", "CANDY", "CARE", "CARGO", "CHEMICAL", "CHILDREN", "CITY", "CLEANER", "CLEANING",
        "CLINIC", "COAST", "COATING", "COFFEE", "COMMUNITY", "CONCRETE", "CONSTRUCTION", "CONTAINER",
        "CONTRACTOR", "CONTROL", "CORRUGATED", "COUNTY", "CUSTOM", "DAIRY", "DELIVERY", "DOOR",
        "DRY", "DRYCLEAN", "DRYCLEANER", "DRYCLEANING", "ELECTRIC", "ELECTRICAL", "ELECTRONIC",
        "ENERGY", "ENGINEERED", "ENGINEERING", "ENTERPRISE", "EQUIPMENT", "EXPORT", "EXPRES",
        "EXPRESS", "FAB", "FABRICATION", "FABRICATOR", "FARM", "FIRST", "FREIGHT", "FUEL",
        "FURNITURE", "GAS", "GENERAL", "GLAS", "GLASS", "GRAPHIC", "GROCERY", "GULF", "HEALTH",
        "HEALTHCARE", "HOME", "HOSPITAL", "ICE", "IMPORT", "IRON", "LABEL", "LAUNDRY", "LINEN",
        "LONE", "LONESTAR", "LUMBER", "MACHINE", "MACHINERY", "MACHINING", "MARKET", "MART",
        "MEAT", "MEDICAL", "MEMORIAL", "METAL", "MILLWORK", "MOTOR", "OIL", "PACKAGE", "PACKAGING",
        "PAINT", "PAPER", "PART", "PARTS", "PETROLEUM", "PIPE", "PLASTIC", "POLYMER", "PORK",
        "POULTRY", "POWER", "PRECISION", "PREMIER", "PRINT", "PRINTING", "PRODUCE", "QUALITY",
        "RECYCLER", "RECYCLING", "REGIONAL", "RESIN", "RUBBER", "SALE", "SALES", "SALVAGE",
        "SCRAP", "SEAFOOD", "SHIPPING", "SNACK", "SOLUTION", "SOUTHERN", "SOUTHWEST",
        "SOUTHWESTERN", "STAR", "STATE", "STEEL", "STONE", "SUPERMARKET", "TECHNOLOGY", "TOOL",
        "TOOLING", "TRADING", "TRAILER", "TRANSPORT", "TRANSPORTATION", "TRUCK", "TRUCKING",
        "TUBE", "TUBING", "UNIFORM", "UNITED", "VEHICLE", "WASTE", "WATER", "WELD", "WELDING",
        "WHOLESALE", "WINDOW", "WIRE", "WOOD", "WOODWORK",
    }
)


def _ordered_name_tokens(name: str | None, drop: set[str]) -> list[str]:
    text = ascii_upper(name)
    if not text:
        return []
    text = re.sub(r"(?<=[A-Z0-9])&(?=[A-Z0-9])", "", text)
    text = _JOINED_PUNCT.sub(lambda m: "" if m.group(0) in "-'." else " ", text)
    text = _PUNCT.sub(" ", text.replace("#", " ").replace("&", " AND "))
    words: list[str] = []
    letters: list[str] = []
    for word in text.split() + [""]:
        if len(word) == 1 and word.isalpha():
            letters.append(word)
            continue
        if len(letters) > 1:
            words.append("".join(letters))
        letters = []
        if word:
            words.append(word)
    out = []
    for token in words:
        token = NAME_ABBREVIATIONS.get(token, token)
        if token in LEGAL_WORDS or token in GENERIC_WORDS or token in drop:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        if len(token) > 4 and token.endswith("S") and not token.endswith("SS"):
            token = token[:-1]
        out.append(token)
    return out


def name_tokens(name: str | None, *, city: str | None = None) -> tuple[str, ...]:
    """Distinctive name tokens: legal suffixes, generic words and the city removed."""
    drop = set((normalize_city(city) or "").split())
    return tuple(sorted(set(_ordered_name_tokens(name, drop))))


@dataclass(frozen=True)
class NameProfile:
    tokens: frozenset
    joins: tuple


def name_profile(name: str | None, *, city: str | None = None) -> NameProfile:
    """Tokens for matching, falling back to tokens without the city, then to the raw words."""
    ordered = _ordered_name_tokens(name, set((normalize_city(city) or "").split()))
    if not ordered:
        ordered = _ordered_name_tokens(name, set())
    if not ordered:
        ordered = [word for word in ascii_upper(name).split() if word.isalnum()]
    joins = tuple(
        (first, second, first + second) for first, second in pairwise(ordered) if first != second
    )
    return NameProfile(tokens=frozenset(ordered), joins=joins)


def name_similarity(first: NameProfile, second: NameProfile) -> tuple[float, bool]:
    """Token-set ratio and whether the names share a distinctive (not weak) token.

    Adjacent tokens are joined when the other name spells them as one word, so
    'WAL MART' matches 'WALMART'.
    """
    left, right = set(first.tokens), set(second.tokens)
    for a, b, joined in first.joins:
        if joined in right and a in left and b in left:
            left -= {a, b}
            left.add(joined)
    for a, b, joined in second.joins:
        if joined in left and a in right and b in right:
            right -= {a, b}
            right.add(joined)
    score = token_set_ratio(left, right)
    strong = any(token not in WEAK_NAME_TOKENS for token in left & right)
    return score, strong


def lcs_length(a: str, b: str) -> int:
    """Length of the longest common subsequence (bit-parallel, Hyyro 2004)."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    masks: dict[str, int] = {}
    for position, char in enumerate(a):
        masks[char] = masks.get(char, 0) | (1 << position)
    full = (1 << len(a)) - 1
    row = full
    for char in b:
        matches = row & masks.get(char, 0)
        row = ((row + matches) | (row - matches)) & full
    return len(a) - row.bit_count()


def ratio(a: str, b: str) -> float:
    """Normalized similarity 0-100 from the LCS: 200 * LCS / (len(a) + len(b))."""
    if not a and not b:
        return 100.0
    return 200.0 * lcs_length(a, b) / (len(a) + len(b))


def token_set_ratio(a_tokens, b_tokens) -> float:
    """Token-set ratio of two token collections, 0-100.

    One token set contained in the other scores 100. Otherwise the score is the
    best ratio among the shared tokens and each side's full sorted token string.
    """
    first, second = set(a_tokens), set(b_tokens)
    if not first or not second:
        return 0.0
    shared = sorted(first & second)
    only_first = sorted(first - second)
    only_second = sorted(second - first)
    if shared and (not only_first or not only_second):
        return 100.0
    base = " ".join(shared)
    left = " ".join(shared + only_first)
    right = " ".join(shared + only_second)
    scores = [ratio(left, right)]
    if base:
        scores.extend((ratio(base, left), ratio(base, right)))
    return round(max(scores), 2)
