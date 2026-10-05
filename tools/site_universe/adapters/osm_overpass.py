"""OpenStreetMap Overpass JSON adapter (pure).

Queries are built from the taxonomy (see ``Taxonomy.overpass_queries``) and
never request metadata, so responses carry no user names, user ids or
changesets. Only an allowlist of classification and address tags is kept;
phone, email and contact tags are dropped. A response that reports a runtime
error (a timeout or memory stop) is partial and is refused.
"""

from __future__ import annotations

import json
import urllib.parse

from tools.site_universe import geo, normalize
from tools.site_universe.adapters import AdapterError, ParseResult
from tools.site_universe.records import SourceRecord, round_coordinate, valid_us_coordinates

SOURCE_ID = "osm_overpass"
API_URL = "https://overpass-api.de/api/interpreter"
# Features that are homes are dropped even when another tag matched a selector.
RESIDENTIAL_TAGS = {
    "building": frozenset(
        {"apartments", "bungalow", "cabin", "detached", "dormitory", "house", "houseboat", "hut",
         "residential", "semidetached_house", "static_caravan", "terrace"}
    ),
    "landuse": frozenset({"residential"}),
}
KEPT_TAGS = frozenset(
    {
        "addr:city", "addr:housenumber", "addr:postcode", "addr:state", "addr:street", "addr:unit",
        "aeroway", "amenity", "beds", "brand", "brand:wikidata", "building", "building:levels",
        "craft", "healthcare", "industrial", "landuse", "man_made", "name", "official_name",
        "operator", "operator:wikidata", "power", "product", "shop", "tourism", "website", "wikidata",
    }
)


def query_url(query: str) -> str:
    """GET URL for one Overpass query; the URL is the raw-cache key."""
    return API_URL + "?" + urllib.parse.urlencode({"data": query})


def validate_payload(path) -> None:
    """Fetch-time check: refuse partial or malformed Overpass responses before caching."""
    with open(path, "rb") as handle:
        _load(handle.read())


def _load(raw: bytes) -> dict:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterError(f"Overpass response is not JSON: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("elements"), list):
        raise AdapterError("Overpass response has no elements list")
    remark = str(payload.get("remark") or "")
    if "runtime error" in remark.lower() or "timed out" in remark.lower():
        raise AdapterError(f"Overpass response is partial: {remark.strip()[:200]}")
    return payload


def _points(geometry) -> list[tuple[float, float]]:
    points = []
    for point in geometry or ():
        if isinstance(point, dict) and point.get("lat") is not None and point.get("lon") is not None:
            points.append((float(point["lat"]), float(point["lon"])))
    return points


def _close(ring):
    if len(ring) >= 4 and ring[0] == ring[-1]:
        return ring[:-1]
    return None


def _assemble(segments) -> list[list[tuple[float, float]]]:
    """Join way segments end to end into closed rings; drop anything left open."""
    rings = []
    pending = [list(segment) for segment in segments if len(segment) >= 2]
    while pending:
        current = pending.pop(0)
        changed = True
        while current[0] != current[-1] and changed:
            changed = False
            for index, segment in enumerate(pending):
                if segment[0] == current[-1]:
                    current.extend(segment[1:])
                elif segment[-1] == current[-1]:
                    current.extend(reversed(segment[:-1]))
                elif segment[-1] == current[0]:
                    current[:0] = segment[:-1]
                elif segment[0] == current[0]:
                    current[:0] = list(reversed(segment[1:]))
                else:
                    continue
                pending.pop(index)
                changed = True
                break
        ring = _close(current)
        if ring:
            rings.append(ring)
    return rings


def _footprint(element):
    kind = element.get("type")
    if kind == "way":
        ring = _close(_points(element.get("geometry")))
        return ([ring], []) if ring else None
    if kind == "relation":
        outers, inners = [], []
        for member in element.get("members") or ():
            if member.get("type") != "way":
                continue
            target = inners if member.get("role") == "inner" else outers
            target.append(_points(member.get("geometry")))
        outer_rings = _assemble(outers)
        return (outer_rings, _assemble(inners)) if outer_rings else None
    return None


def _center(element, footprint):
    if element.get("type") == "node":
        return element.get("lat"), element.get("lon")
    if footprint:
        largest = max(footprint[0], key=geo.ring_area_m2)
        return geo.ring_centroid(largest)
    points = _points(element.get("geometry"))
    if points:
        return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
    bounds = element.get("bounds") or {}
    if all(key in bounds for key in ("minlat", "maxlat", "minlon", "maxlon")):
        return ((bounds["minlat"] + bounds["maxlat"]) / 2, (bounds["minlon"] + bounds["maxlon"]) / 2)
    return None, None


def _residential(tags: dict) -> bool:
    return any(tags.get(key) in values for key, values in RESIDENTIAL_TAGS.items())


def parse(raw: bytes, *, state: str, raw_sha256: str, retrieved_at: str) -> ParseResult:
    payload = _load(raw)
    state = normalize.normalize_state(state)
    result = ParseResult()
    base = (payload.get("osm3s") or {}).get("timestamp_osm_base")
    if base:
        result.stats["osm_base_timestamp"] = base
    for element in payload["elements"]:
        result.count("elements_read")
        kind, osm_id = element.get("type"), element.get("id")
        tags = element.get("tags") or {}
        if kind not in ("node", "way", "relation") or osm_id is None:
            result.count("dropped_malformed")
            continue
        name = normalize.clean_text(tags.get("name"))
        if not name:
            result.count("dropped_no_name")
            continue
        if _residential(tags):
            result.count("dropped_residential_tag")
            continue
        footprint = _footprint(element)
        lat, lon = _center(element, footprint)
        lat, lon = round_coordinate(lat), round_coordinate(lon)
        if not valid_us_coordinates(lat, lon):
            result.count("dropped_bad_coordinates")
            continue
        area = round(geo.polygon_area_m2(*footprint), 1) if footprint else None
        has_building = tags.get("building") not in (None, "no")
        house = normalize.clean_text(tags.get("addr:housenumber"))
        street_name = normalize.clean_text(tags.get("addr:street"))
        street, unit = normalize.normalize_street(
            f"{house} {street_name}" if house and street_name else None, tags.get("addr:unit")
        )
        record_state = normalize.normalize_state(tags.get("addr:state")) or state
        kept = {key: str(value) for key, value in tags.items() if key in KEPT_TAGS}
        attributes = {"osm_tags": dict(sorted(kept.items())), "osm_type": kind}
        if area is not None and not has_building:
            attributes["site_area_m2"] = area
        result.records.append(
            SourceRecord(
                source_id=SOURCE_ID,
                source_record_id=f"{kind}/{osm_id}",
                name=name,
                operator=normalize.clean_text(tags.get("operator") or tags.get("brand")),
                street=street,
                unit=unit,
                city=normalize.normalize_city(tags.get("addr:city")),
                state=record_state,
                postal_code=normalize.normalize_postal(tags.get("addr:postcode")),
                country="US",
                lat=lat,
                lon=lon,
                naics=None,
                category=None,
                employees=None,
                building_area_m2=area if has_building else None,
                retrieved_at=retrieved_at,
                raw_sha256=raw_sha256,
                attributes=attributes,
                footprint=footprint,
                codes=(),
                code_system="osm",
            )
        )
        result.count("records")
    return result
