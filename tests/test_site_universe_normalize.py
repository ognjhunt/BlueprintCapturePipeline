"""Address, name and geometry normalization for the site universe (pure, no network)."""

import random

import pytest

from tools.site_universe import geo, normalize
from tools.site_universe.records import looks_like_person_name


@pytest.mark.parametrize(
    ("raw", "supplemental", "street", "unit"),
    [
        ("4718 Ellery Quill Dr.", None, "4718 ELLERY QUILL DR", None),
        ("100 N Main Street Suite #4", None, "100 N MAIN ST", "STE 4"),
        ("3150 Brindlecote Commerce Parkway Bldg B", None, "3150 BRINDLECOTE COMMERCE PKWY", "BLDG B"),
        ("100 INDUSTRIAL BOULEVARD", "SUITE 5", "100 INDUSTRIAL BLVD", "STE 5"),
        ("2200 Farm to Market Road 1960 West", None, "2200 FM 1960 W", None),
        ("2200 FM 1960 Rd W", None, "2200 FM 1960 W", None),
        ("5000 SH 6 South", None, "5000 STATE HWY 6 S", None),
        ("800 US Highway 290 East", None, "800 US HWY 290 E", None),
        ("4517 S IH-35", None, "4517 S I 35", None),
        ("123 CR 101", None, "123 COUNTY RD 101", None),
        ("77 SH Lane", None, "77 SH LN", None),
        ("48120 E US HWY 90 E OF QUILLTOWN", None, "48120 E US HWY 90", None),
        ("250 First Avenue", None, "250 1ST AVE", None),
        ("PO Box 1234", None, None, None),
        ("8 MI N OF ALVIN ON STATE HWY 35", None, None, None),
        ("CORNER OF HWY 6 AND FM 529", None, None, None),
        ("", None, None, None),
    ],
)
def test_street_normalization_is_usps_style(raw, supplemental, street, unit):
    assert normalize.normalize_street(raw, supplemental) == (street, unit)


def test_postal_state_and_city():
    assert normalize.normalize_postal("77165-0412") == "77165"
    assert normalize.normalize_postal("771650412") == "77165"
    assert normalize.normalize_postal("00000") is None
    assert normalize.normalize_postal("7500") is None
    assert normalize.normalize_state("Texas") == "TX"
    assert normalize.normalize_state("tx") == "TX"
    assert normalize.normalize_state("Atlantis") is None
    assert normalize.normalize_city("Ft. Worth") == "FORT WORTH"
    assert normalize.normalize_city("san antonio") == "SAN ANTONIO"


def test_address_key_needs_a_numbered_street():
    assert normalize.address_key("100 MAIN ST", "DALLAS", "TX", "75201") == "100 MAIN ST|DALLAS|TX|75201"
    assert normalize.address_key("MAIN ST", "DALLAS", "TX", "75201") is None
    assert normalize.address_key("100 MAIN ST", None, "TX", None) is None


def test_name_tokens_drop_legal_generic_and_city_words():
    assert normalize.name_tokens("Acme Cold Storage, LLC") == ("ACME", "COLD", "STORAGE")
    assert normalize.name_tokens("Tallowmere Foods - Larkstone Plant", city="Larkstone") == (
        "FOOD", "TALLOWMERE",
    )
    assert normalize.name_tokens("H-E-B") == ("HEB",)
    assert normalize.name_tokens("H E B Plus") == ("HEB", "PLUS")
    assert normalize.name_tokens("Wal-Mart Supercenter #0000") == ("0000", "WALMART")
    assert normalize.name_tokens("C&S Wholesale Grocers") == ("CS", "GROCER", "WHOLESALE")


def test_name_similarity_needs_a_distinctive_shared_token():
    def similar(a, b, city=None):
        return normalize.name_similarity(
            normalize.name_profile(a, city=city), normalize.name_profile(b, city=city)
        )

    assert similar("WAL MART SUPERCENTER #0000", "Walmart Supercenter") == (100.0, True)
    assert similar("HEB GROCERY 123", "H-E-B") == (100.0, True)
    score, strong = similar("Tallowmere Foods - Larkstone", "TALLOWMERE FRESH MEATS INC LARKSTONE PLANT",
                            "LARKSTONE")
    assert strong and score >= 60
    # Two dry cleaners share only the industry word CLEANER.
    score, strong = similar("ZORVEL CLEANERS", "ODALYNS CLEANER")
    assert not strong and score < 92
    assert similar("ACME COLD STORAGE", "BOLT MANUFACTURING")[1] is False


def test_token_set_ratio_properties():
    assert normalize.token_set_ratio(("ACME",), ("ACME", "BOLT")) == 100.0
    assert normalize.token_set_ratio(("ACME", "BOLT"), ("BOLT", "ACME")) == 100.0
    assert normalize.token_set_ratio((), ("ACME",)) == 0.0
    forward = normalize.token_set_ratio(("ACME", "FOOD"), ("ACME", "FRESH", "MEAT"))
    backward = normalize.token_set_ratio(("ACME", "FRESH", "MEAT"), ("ACME", "FOOD"))
    assert forward == backward
    assert normalize.token_set_ratio(("ACME", "COLD"), ("ZETA", "BOLT")) < 50


def _naive_lcs(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i, x in enumerate(a, 1):
        for j, y in enumerate(b, 1):
            table[i][j] = table[i - 1][j - 1] + 1 if x == y else max(table[i - 1][j], table[i][j - 1])
    return table[-1][-1]


def test_bit_parallel_lcs_matches_dynamic_programming():
    rng = random.Random(20261004)
    for _ in range(400):
        a = "".join(rng.choice("ABC D") for _ in range(rng.randint(0, 40)))
        b = "".join(rng.choice("ABC D") for _ in range(rng.randint(0, 40)))
        assert normalize.lcs_length(a, b) == _naive_lcs(a, b)


def test_geohash_and_distance():
    assert geo.geohash(57.64911, 10.40744, 11) == "u4pruydqqvj"
    assert geo.geohash(42.6, -5.6, 5) == "ezs42"
    assert len(geo.geohash(32.78, -96.8)) == 7
    assert geo.haversine_m(32.78, -96.8, 32.78, -96.8) == 0.0
    assert geo.haversine_m(0.0, 0.0, 1.0, 0.0) == pytest.approx(111_195, rel=1e-4)


def test_footprint_area_and_containment():
    square = [(0.0, 0.0), (0.0, 0.001), (0.001, 0.001), (0.001, 0.0)]
    assert geo.ring_area_m2(square) == pytest.approx(111.195**2, rel=5e-3)
    hole = [(0.0004, 0.0004), (0.0004, 0.0006), (0.0006, 0.0006), (0.0006, 0.0004)]
    assert geo.polygon_area_m2([square], [hole]) == pytest.approx(111.195**2 * 0.96, rel=5e-3)
    footprint = ([square], [hole])
    assert geo.point_in_footprint(0.0002, 0.0002, footprint)
    assert not geo.point_in_footprint(0.0005, 0.0005, footprint)
    assert not geo.point_in_footprint(0.002, 0.002, footprint)


@pytest.mark.parametrize(
    ("value", "plain", "expected"),
    [
        ("SMITH, JOHN A", False, True),
        ("Garcia, Maria L.", False, True),
        ("John A Smith", True, True),
        ("John A Smith", False, False),
        ("Blue Bell", False, False),
        ("JOHN SMITH", True, False),
        ("Smith Meat Processing", True, False),
        ("TYSON FOODS, INC", True, False),
    ],
)
def test_person_name_check_is_conservative(value, plain, expected):
    assert looks_like_person_name(value, plain=plain) is expected
