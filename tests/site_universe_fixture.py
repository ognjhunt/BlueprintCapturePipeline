"""Hermetic raw-cache fixtures for the site universe tests. No network is used."""

from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from pathlib import Path

from tools.site_universe import sources, taxonomy
from tools.site_universe.adapters import epa_frs, osm_overpass
from tools.site_universe.fetch import RawCache

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "site_universe"
FETCHED_AT = "2026-10-04T12:00:00Z"
OSHA_IMPORTED_AT = "2026-10-04T12:10:00Z"
FSIS_IMPORTED_AT = "2026-10-04T12:15:00Z"
OSHA_URL = sources.registry()["osha_ita"]["download_url"]
FSIS_URL = "https://www.fsis.usda.gov/sites/default/files/media_file/documents/MPI_Directory_by_Establishment_Name.csv"
# Strings planted in the fixtures that must never reach a snapshot.
PERSONAL_STRINGS = (
    "Pat Fixture-Contact", "pat.contact@example.invalid", "214-555-0142", "owner@example.invalid",
    "11-1111111", "98-7654321", "22-2222222", "Pat Fixture-Reviewer", "John Q Public", "SMITH, JOHN A",
    "Garcia, Maria L", "Doe, Jane", "(281) 555-0100", "(830) 555-0177", "+1 512 555 0199",
    "info@glimmerdale.example.invalid", "sales@ironclad.example.invalid", "+1 214 555 0100", "+1 210 555 0111",
)


def frs_zip_bytes() -> bytes:
    """Deterministic zip of the EPA FRS fixture CSVs."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((FIXTURES / "epa_frs_tx").iterdir()):
            info = zipfile.ZipInfo(path.name, date_time=(2026, 9, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def osm_payloads() -> dict[str, bytes]:
    groups = json.loads((FIXTURES / "osm_overpass_tx.json").read_text(encoding="utf-8"))
    out = {}
    for group, query in taxonomy.load().overpass_queries("TX"):
        payload = groups.get(group, {"elements": [], "version": 0.6})
        out[osm_overpass.query_url(query)] = json.dumps(payload, sort_keys=True).encode()
    return out


class FakeResponse:
    def __init__(self, body: bytes, content_type: str = "application/octet-stream"):
        self._body = io.BytesIO(body)
        self.status = 200
        self.headers = {"Content-Type": content_type}
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def close(self) -> None:
        self.closed = True


class FakeOpener:
    """Serves fixture bytes by URL; records every request; raises HTTPError for listed codes."""

    def __init__(self, bodies: dict[str, bytes], errors: dict[str, list[int]] | None = None):
        self.bodies = dict(bodies)
        self.errors = {url: list(codes) for url, codes in (errors or {}).items()}
        self.requests: list[str] = []

    def __call__(self, request, timeout):
        url = request.full_url
        self.requests.append(url)
        codes = self.errors.get(url)
        if codes:
            code = codes.pop(0)
            raise urllib.error.HTTPError(url, code, "fixture error", {}, None)
        if url not in self.bodies:
            raise urllib.error.HTTPError(url, 404, "not in fixture", {}, None)
        return FakeResponse(self.bodies[url])


def fixture_bodies() -> dict[str, bytes]:
    bodies = {epa_frs.download_url("TX"): frs_zip_bytes()}
    bodies.update(osm_payloads())
    return bodies


def seeded_cache(raw_dir: Path, *, manual: bool = True, opener: FakeOpener | None = None) -> RawCache:
    """A raw cache filled through a fake opener (FRS and Overpass) plus manual imports (OSHA, FSIS)."""
    cache = RawCache(
        raw_dir,
        opener=opener or FakeOpener(fixture_bodies()),
        clock=lambda: FETCHED_AT,
        sleep=lambda seconds: None,
    )
    for url in sorted(fixture_bodies()):
        source_id = "epa_frs" if url.endswith(".zip") else "osm_overpass"
        cache.get(url, source_id=source_id, suffix="" if source_id == "epa_frs" else ".json")
    if manual:
        cache.import_file(OSHA_URL, FIXTURES / "osha_ita_300a.csv", source_id="osha_ita",
                          retrieved_at=OSHA_IMPORTED_AT)
        cache.import_file(FSIS_URL, FIXTURES / "fsis_mpi.csv", source_id="fsis_mpi",
                          retrieved_at=FSIS_IMPORTED_AT)
    return cache
