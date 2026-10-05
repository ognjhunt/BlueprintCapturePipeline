"""Source registry for the site universe producer.

Every bulk source records its publisher, URLs, license, required attribution,
share-alike requirement, allowed uses, refresh cadence, provided fields and
what was verified on the web and when. :func:`check_source` fails closed: the
builder refuses any source without a recorded license, attribution and
share-alike requirements, allowed uses and a verification date.

Statuses:
- ``enabled``: fetched automatically through ``fetch.py``.
- ``manual_import_only``: the publisher refuses automated clients. A person
  downloads the file in a browser and records it with ``import-raw``; the build
  uses it only when it is in the raw cache, and otherwise records a skip.
- ``skipped``: never used (for example, the license is unclear).
"""

from __future__ import annotations

import copy

REGISTRY_SCHEMA = "blueprint.site_universe.source_registry.v1"
REQUIRED_USES = ("commercial_use", "internal_derivative_database")
STATUSES = ("enabled", "manual_import_only", "skipped")
VERIFIED_AT = "2026-10-04"

# Field precedence when several records of one site provide the same field.
FIELD_PRIORITY = {
    "employees": ("osha_ita",),
    "address": ("epa_frs", "osha_ita", "fsis_mpi", "osm_overpass"),
    "coordinates": ("osm_overpass", "epa_frs", "fsis_mpi"),
    "building_area_m2": ("osm_overpass",),
    "name": ("osha_ita", "fsis_mpi", "osm_overpass", "epa_frs"),
    "operator": ("osha_ita", "fsis_mpi", "osm_overpass", "epa_frs"),
    "naics": ("osha_ita", "epa_frs"),
    "category": ("osha_ita", "epa_frs", "fsis_mpi", "osm_overpass"),
}


class SourceRefused(ValueError):
    """The registry entry lacks a license, allowed uses or verification."""


SOURCES = (
    {
        "id": "osha_ita",
        "name": "OSHA Injury Tracking Application (ITA) establishment-specific Form 300A summary data",
        "publisher": "U.S. Department of Labor, Occupational Safety and Health Administration",
        "landing_url": "https://www.osha.gov/Establishment-Specific-Injury-and-Illness-Data",
        "download_url": (
            "https://www.osha.gov/sites/default/files/"
            "ITA_300A_Summary_Data_2025_through_03-15-2026_v2.csv"
        ),
        "format": "csv (one national file per calendar year; older years are zip archives)",
        "status": "manual_import_only",
        "license": {
            "id": "US-Gov-Work",
            "url": "https://www.usa.gov/government-copyright",
            "basis": "17 U.S.C. 105: works of the U.S. federal government are not subject to copyright.",
        },
        "attribution": None,
        "attribution_required": False,
        "share_alike": False,
        "allowed_uses": ["commercial_use", "internal_derivative_database", "redistribution"],
        "refresh_cadence": "annual; the 300A summary file for a calendar year appears after the March 2 "
        "submission deadline and is revised during the year",
        "fields_provided": [
            "name", "operator", "street", "city", "state", "postal_code", "naics", "employees",
        ],
        "personal_data_policy": {
            "dropped_at_parse": [
                "ein", "change_reason", "created_timestamp", "total_hours_worked",
                "all injury and illness counts",
            ],
            "notes": "Form 300A case detail files (300/301) are never read. A company_name or "
            "establishment_name that looks like a person's name is dropped.",
        },
        "verification": {
            "verified_at": VERIFIED_AT,
            "evidence": [
                {
                    "url": "https://www.osha.gov/Establishment-Specific-Injury-and-Illness-Data",
                    "observed": "Page lists the 2025 Summary Data CSV "
                    "(ITA_300A_Summary_Data_2025_through_03-15-2026_v2.csv), 2024 and 2023 summary "
                    "zip files and the summary data dictionary. No license statement on the page.",
                },
                {
                    "url": "https://www.osha.gov/sites/default/files/summary_data_dictionary.pdf",
                    "observed": "Data dictionary (April 2024): columns id, establishment_name, "
                    "establishment_id, ein, company_name, street_address, city, state, zip_code, "
                    "naics_code, naics_year, industry_description, establishment_type, size, "
                    "annual_average_employees, total_hours_worked, injury counts, "
                    "created_timestamp, change_reason, year_filing_for.",
                },
                {
                    "url": "https://catalog.data.gov/dataset/osha-form-300a",
                    "observed": "Data.gov entry: access level public, no license field, landing page "
                    "as above, metadata modified 2026-03-27.",
                },
                {
                    "url": "https://www.osha.gov/robots.txt",
                    "observed": "HTTP 403 from CloudFront for curl and Python urllib, including "
                    "robots.txt and the data files; only the agent web-fetch tool could read the "
                    "landing page and the dictionary. The producer does not work around bot "
                    "protection, so this source needs a manual browser download.",
                },
                {
                    "url": "https://dataportal.dol.gov/registration",
                    "observed": "API key registration page (a JavaScript app). DOL pages found by "
                    "search say the data portal API needs a registered key, so it is not used (no "
                    "credentials). Whether it serves ITA establishment data was not confirmed.",
                },
            ],
        },
    },
    {
        "id": "epa_frs",
        "name": "EPA Facility Registry Service (FRS) state combined CSV files",
        "publisher": "U.S. Environmental Protection Agency",
        "landing_url": "https://www.epa.gov/frs/epa-state-combined-csv-download-files",
        "download_url_template": (
            "https://ordsext.epa.gov/FLA/www3/state_files/state_combined_{state_lower}.zip"
        ),
        "format": "zip of CSV files per state",
        "status": "enabled",
        "license": {
            "id": "US-PD",
            "url": "https://edg.epa.gov/EPA_Data_License.html",
            "basis": "EPA Standard Open Data License: unless otherwise specified, data produced by EPA "
            "is in the public domain (17 U.S.C. 105); U.S. Public Domain label "
            "https://www.usa.gov/publicdomain/label/1.0/.",
        },
        "attribution": None,
        "attribution_required": False,
        "share_alike": False,
        "allowed_uses": ["commercial_use", "internal_derivative_database", "redistribution"],
        "refresh_cadence": "EPA refreshes the state files regularly; the TX file was last modified "
        "2026-09-08 when retrieved",
        "fields_provided": [
            "name", "street", "city", "state", "postal_code", "lat", "lon", "naics", "sic",
            "activity_status", "last_activity_year",
        ],
        "personal_data_policy": {
            "members_read": [
                "<ST>_FACILITY_FILE.CSV", "<ST>_NAICS_FILE.CSV", "<ST>_SIC_FILE.CSV",
                "<ST>_ENVIRONMENTAL_INTEREST_FILE.CSV",
            ],
            "members_never_opened": [
                "<ST>_CONTACT_FILE.CSV (person names, phone numbers, email addresses)",
                "<ST>_ORGANIZATION_FILE.CSV (EIN, DUNS, phone numbers, email addresses)",
                "<ST>_MAILING_ADDRESS_FILE.CSV", "<ST>_PROGRAM_FILE.CSV (includes a USER_ID column)",
                "<ST>_ALTERNATIVE_NAME_FILE.CSV", "<ST>_SUPP_INTEREST_FILE.CSV",
                "<ST>_PROGRAM_GIS_FILE.CSV",
            ],
            "notes": "A PRIMARY_NAME that looks like a person's name is dropped with its record.",
        },
        "verification": {
            "verified_at": VERIFIED_AT,
            "evidence": [
                {
                    "url": "https://www.epa.gov/frs/epa-state-combined-csv-download-files",
                    "observed": "Page (last updated 2026-04-30) offers state combined CSV zips with "
                    "facility, geospatial, interest, organization, NAICS/SIC, alternative name, "
                    "contact and mailing address files; documentation is inside each zip.",
                },
                {
                    "url": "https://ordsext.epa.gov/FLA/www3/state_files/state_combined_tx.zip",
                    "observed": "HTTP 200, 100,779,107 bytes, Last-Modified 2026-09-08, automated "
                    "download allowed.",
                },
                {
                    "url": "https://edg.epa.gov/EPA_Data_License.html",
                    "observed": "States that, unless otherwise specified, all data produced by EPA is "
                    "in the public domain under 17 U.S.C. 105.",
                },
                {
                    "url": "https://catalog.data.gov/dataset/facility-registry-service-frs",
                    "observed": "Data.gov FRS entry: license https://edg.epa.gov/EPA_Data_License.htm, "
                    "access level public.",
                },
            ],
            "caveats": [
                (
                    "FRS integrates state program records (for example TCEQ). Names, addresses "
                    "and codes are factual records that EPA republishes without stated restrictions."
                ),
                (
                    "FRS keeps closed facilities; the adapter records activity status and the last "
                    "activity year from the environmental interest file."
                ),
            ],
        },
    },
    {
        "id": "osm_overpass",
        "name": "OpenStreetMap through the Overpass API",
        "publisher": "OpenStreetMap contributors; queried on the public Overpass API instance",
        "landing_url": "https://www.openstreetmap.org/copyright",
        "api_url": "https://overpass-api.de/api/interpreter",
        "format": "Overpass JSON (out geom)",
        "status": "enabled",
        "license": {
            "id": "ODbL-1.0",
            "url": "https://opendatacommons.org/licenses/odbl/1-0/",
            "basis": "OpenStreetMap data is licensed under the Open Data Commons Open Database "
            "License by the OpenStreetMap Foundation.",
        },
        "attribution": "© OpenStreetMap contributors",
        "attribution_required": True,
        "share_alike": True,
        "allowed_uses": [
            "commercial_use", "internal_derivative_database",
            "public_use_only_with_attribution_and_odbl_share_alike",
        ],
        "usage_policy": {
            "url": "https://dev.overpass-api.de/overpass-doc/en/preface/commons.html",
            "publisher_limits": "about 10,000 requests and 1 GB per day per user; HTTP 429 when "
            "over the rate limit; default timeout 180 s and memory 512 MiB",
            "producer_bounds": "at most 4 queries per state per build, run one at a time at least "
            "15 s apart, with a 300 s server timeout; every response is cached and reused",
        },
        "refresh_cadence": "continuously edited upstream; refreshed per snapshot",
        "fields_provided": [
            "name", "operator", "street", "city", "state", "postal_code", "lat", "lon",
            "building_area_m2", "site_area_m2", "tags",
        ],
        "personal_data_policy": {
            "dropped_at_parse": [
                "phone", "fax", "email", "contact:*",
                "every tag outside the classification and address allowlist",
            ],
            "notes": "Queries never request metadata (no user names, user ids or changesets). "
            "Residential and personal feature tags are refused as selectors.",
        },
        "verification": {
            "verified_at": VERIFIED_AT,
            "evidence": [
                {
                    "url": "https://www.openstreetmap.org/copyright",
                    "observed": "OpenStreetMap is open data under the ODbL; credit OpenStreetMap and "
                    "its contributors; derived databases may be distributed only under the same "
                    "license.",
                },
                {
                    "url": "https://opendatacommons.org/licenses/odbl/1-0/",
                    "observed": "ODbL 1.0 legal code resolves (HTTP 200).",
                },
                {
                    "url": "https://osmfoundation.org/wiki/Licence/Attribution_Guidelines",
                    "observed": "Attribution guidelines resolve (HTTP 200).",
                },
                {
                    "url": "https://dev.overpass-api.de/overpass-doc/en/preface/commons.html",
                    "observed": "Public instance policy: about 10,000 queries and 1 GB per day; "
                    "rate limited per IP; set up your own instance for heavy use.",
                },
                {
                    "url": "https://overpass-api.de/api/status",
                    "observed": "HTTP 200; rate limit 2 slots for this client.",
                },
            ],
        },
    },
    {
        "id": "fsis_mpi",
        "name": "USDA FSIS Meat, Poultry and Egg Product Inspection (MPI) Directory",
        "publisher": "U.S. Department of Agriculture, Food Safety and Inspection Service",
        "landing_url": (
            "https://www.fsis.usda.gov/inspection/establishments/"
            "meat-poultry-and-egg-product-inspection-directory"
        ),
        "download_url": None,
        "format": "csv (directory by establishment name or number)",
        "status": "manual_import_only",
        "license": {
            "id": "CC0-1.0",
            "url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "basis": "Data.gov metadata for the FSIS MPI Directory records the CC0 1.0 public domain "
            "dedication; also a U.S. federal government work (17 U.S.C. 105).",
        },
        "attribution": None,
        "attribution_required": False,
        "share_alike": False,
        "allowed_uses": ["commercial_use", "internal_derivative_database", "redistribution"],
        "refresh_cadence": "weekly to monthly; each edition replaces the previous one",
        "fields_provided": ["name", "street", "city", "state", "postal_code", "activities"],
        "schema_verified": False,
        "personal_data_policy": {
            "dropped_at_parse": ["phone"],
            "notes": "A company name that looks like a person's name (a sole proprietor) is replaced "
            "by the first DBA name, or the record is dropped.",
        },
        "verification": {
            "verified_at": VERIFIED_AT,
            "evidence": [
                {
                    "url": "https://catalog.data.gov/dataset/"
                    "fsis-mpi-meat-poultry-and-egg-inspection-directory-by-establishment-number",
                    "observed": "Data.gov entry: license CC0 1.0, access level public, publisher FSIS, "
                    "metadata modified 2026-08-31; its resource link is the directory page.",
                },
                {
                    "url": "https://www.fsis.usda.gov/inspection/establishments/"
                    "meat-poultry-and-egg-product-inspection-directory",
                    "observed": "HTTP 403 for every automated client tried (curl, Python urllib and "
                    "the agent fetch tool), including robots.txt and the CSV paths. The exact CSV "
                    "link and its header could not be read, so the adapter checks the documented "
                    "columns and refuses any other header.",
                },
            ],
        },
    },
)


def registry() -> dict[str, dict]:
    return {entry["id"]: copy.deepcopy(entry) for entry in SOURCES}


def check_source(entry: dict, *, required_uses=REQUIRED_USES) -> dict:
    """Return the entry when it may be used; raise :class:`SourceRefused` otherwise."""
    source_id = entry.get("id") or "<missing id>"
    for field in ("id", "name", "publisher", "landing_url", "refresh_cadence", "fields_provided"):
        if not entry.get(field):
            raise SourceRefused(f"{source_id}: missing registry field {field!r}")
    license_ = entry.get("license") or {}
    if not license_.get("id") or not license_.get("url"):
        raise SourceRefused(f"{source_id}: no recorded license (SPDX-like id and URL are required)")
    uses = entry.get("allowed_uses")
    if not uses:
        raise SourceRefused(f"{source_id}: no recorded allowed uses")
    lacking = [use for use in required_uses if use not in uses]
    if lacking:
        raise SourceRefused(f"{source_id}: allowed uses do not include {', '.join(lacking)}")
    if "attribution_required" not in entry:
        raise SourceRefused(f"{source_id}: attribution requirement not recorded")
    if entry["attribution_required"] and not entry.get("attribution"):
        raise SourceRefused(f"{source_id}: attribution is required but its text is not recorded")
    if not isinstance(entry.get("share_alike"), bool):
        raise SourceRefused(f"{source_id}: share-alike requirement not recorded (true or false)")
    if not (entry.get("verification") or {}).get("verified_at"):
        raise SourceRefused(f"{source_id}: no recorded verification date")
    status = entry.get("status")
    if status not in STATUSES:
        raise SourceRefused(f"{source_id}: unknown status {status!r}")
    if status == "skipped":
        reason = entry.get("skip_reason") or "marked skipped in the registry"
        raise SourceRefused(f"{source_id}: {reason}")
    if status == "enabled" and not (
        entry.get("download_url") or entry.get("download_url_template") or entry.get("api_url")
    ):
        raise SourceRefused(f"{source_id}: enabled without a download or API URL")
    return entry


def license_entry(entry: dict) -> dict:
    license_ = entry["license"]
    return {
        "attribution": entry.get("attribution"),
        "attribution_required": bool(entry.get("attribution_required")),
        "id": license_["id"],
        "share_alike": bool(entry.get("share_alike")),
        "source_id": entry["id"],
        "url": license_["url"],
    }
