"""Fail-closed geography policy for provider workers that call OpenAI APIs."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


OPENAI_API_SUPPORTED_COUNTRIES_SOURCE = (
    "https://help.openai.com/en/articles/5347006-openai-api-supported-countries-and-territories"
)
OPENAI_API_SUPPORTED_COUNTRIES_REVIEWED_ON = "2026-08-15"

# ISO 3166-1 alpha-2 codes corresponding to the official source above.  Vast
# reports locations as slugs ending in one of these codes (for example,
# ``california_us``).  This is intentionally an allowlist: unknown or malformed
# locations fail closed for workloads that forward an OpenAI credential.
OPENAI_API_SUPPORTED_COUNTRY_CODES = frozenset(
    {
        "ad",
        "ae",
        "af",
        "ag",
        "al",
        "am",
        "ao",
        "ar",
        "at",
        "au",
        "az",
        "ba",
        "bb",
        "bd",
        "be",
        "bf",
        "bg",
        "bh",
        "bi",
        "bj",
        "bn",
        "bo",
        "br",
        "bs",
        "bt",
        "bw",
        "bz",
        "ca",
        "cd",
        "cf",
        "cg",
        "ch",
        "ci",
        "cl",
        "cm",
        "co",
        "cr",
        "cv",
        "cy",
        "cz",
        "de",
        "dj",
        "dk",
        "dm",
        "do",
        "dz",
        "ec",
        "ee",
        "eg",
        "er",
        "es",
        "et",
        "fi",
        "fj",
        "fm",
        "fr",
        "ga",
        "gb",
        "gd",
        "ge",
        "gh",
        "gm",
        "gn",
        "gq",
        "gr",
        "gt",
        "gw",
        "gy",
        "hn",
        "hr",
        "ht",
        "hu",
        "id",
        "ie",
        "il",
        "in",
        "iq",
        "is",
        "it",
        "jm",
        "jo",
        "jp",
        "ke",
        "kg",
        "kh",
        "ki",
        "km",
        "kn",
        "kr",
        "kw",
        "kz",
        "la",
        "lb",
        "lc",
        "li",
        "lk",
        "lr",
        "ls",
        "lt",
        "lu",
        "lv",
        "ly",
        "ma",
        "mc",
        "md",
        "me",
        "mg",
        "mh",
        "mk",
        "ml",
        "mm",
        "mn",
        "mr",
        "mt",
        "mu",
        "mv",
        "mw",
        "mx",
        "my",
        "mz",
        "na",
        "ne",
        "ng",
        "ni",
        "nl",
        "no",
        "np",
        "nr",
        "nz",
        "om",
        "pa",
        "pe",
        "pg",
        "ph",
        "pk",
        "pl",
        "ps",
        "pt",
        "pw",
        "py",
        "qa",
        "ro",
        "rs",
        "rw",
        "sa",
        "sb",
        "sc",
        "sd",
        "se",
        "sg",
        "si",
        "sk",
        "sl",
        "sm",
        "sn",
        "so",
        "sr",
        "ss",
        "st",
        "sv",
        "sz",
        "td",
        "tg",
        "th",
        "tj",
        "tl",
        "tm",
        "tn",
        "to",
        "tr",
        "tt",
        "tv",
        "tw",
        "tz",
        "ua",
        "ug",
        "us",
        "uy",
        "uz",
        "va",
        "vc",
        "vn",
        "vu",
        "ws",
        "ye",
        "za",
        "zm",
        "zw",
    }
)


def vast_geolocation_country_code(value: object) -> str | None:
    """Return the terminal Vast country code, or ``None`` if unprovable.

    Vast reports geolocation in two shapes: underscore slugs
    (``california_us``) and display strings from the live offers endpoint
    (``Sweden, SE`` — sometimes with an empty region, ``, CA``). The code is
    accepted only when a separator proves the terminal two letters are a
    country suffix rather than the tail of a word; anything else stays
    unprovable and fails closed.
    """

    text = str(value or "").strip().lower()
    if len(text) == 2 and text.isalpha():
        return text
    if len(text) >= 4 and text[-2:].isalpha() and text[-3] in {"_", " ", ","}:
        return text[-2:]
    return None


def openai_vast_country_allowlist(model_backend: object) -> frozenset[str]:
    """Return the mandatory Vast geography allowlist for an OpenAI worker."""

    if str(model_backend or "").strip().lower() == "openai":
        return OPENAI_API_SUPPORTED_COUNTRY_CODES
    return frozenset()


def normalize_vast_country_allowlist(values: Iterable[object]) -> frozenset[str]:
    """Normalize and validate an optional ISO alpha-2 country allowlist."""

    countries = frozenset(str(value).strip().lower() for value in values if str(value).strip())
    if any(not re.fullmatch(r"[a-z]{2}", value) for value in countries):
        raise ValueError("invalid_vast_allowed_geolocation_country_code")
    return countries


def vast_geolocation_allowed(value: object, countries: Iterable[str]) -> bool:
    allowed = frozenset(countries)
    return not allowed or vast_geolocation_country_code(value) in allowed


def vast_country_policy_manifest(
    offers: Sequence[Mapping[str, Any]], countries: Iterable[str]
) -> dict[str, Any]:
    allowed = sorted(set(countries))
    return {
        "allowed_geolocation_country_codes": allowed,
        "geolocation_country_allowlist_active": bool(allowed),
        "geolocation_country_allowed_offer_count": (
            sum(
                vast_geolocation_country_code(offer.get("geolocation")) in allowed
                for offer in offers
            )
            if allowed
            else None
        ),
    }
