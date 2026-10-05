"""Pure geometry helpers: geohash, great-circle distance, footprint area."""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6_371_008.8
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash(lat: float, lon: float, precision: int = 7) -> str:
    """Standard base-32 geohash. Precision 7 is a cell of about 153 m by 153 m."""
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        raise ValueError(f"coordinate out of range: {lat}, {lon}")
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    out = []
    bits = 0
    bit_count = 0
    even = True
    while len(out) < precision:
        if even:
            middle = (lon_range[0] + lon_range[1]) / 2
            if lon >= middle:
                bits = (bits << 1) | 1
                lon_range[0] = middle
            else:
                bits <<= 1
                lon_range[1] = middle
        else:
            middle = (lat_range[0] + lat_range[1]) / 2
            if lat >= middle:
                bits = (bits << 1) | 1
                lat_range[0] = middle
            else:
                bits <<= 1
                lat_range[1] = middle
        even = not even
        bit_count += 1
        if bit_count == 5:
            out.append(_BASE32[bits])
            bits = 0
            bit_count = 0
    return "".join(out)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def _project(ring, lat0: float, lon0: float):
    cos_lat = math.cos(math.radians(lat0))
    scale = math.pi / 180.0 * EARTH_RADIUS_M
    return [((lon - lon0) * scale * cos_lat, (lat - lat0) * scale) for lat, lon in ring]


def ring_area_m2(ring) -> float:
    """Unsigned area of one ring of (lat, lon) points, by a local equirectangular projection."""
    if len(ring) < 3:
        return 0.0
    lat0 = sum(point[0] for point in ring) / len(ring)
    lon0 = sum(point[1] for point in ring) / len(ring)
    points = _project(ring, lat0, lon0)
    twice = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        twice += x1 * y2 - x2 * y1
    return abs(twice) / 2.0


def polygon_area_m2(outers, inners=()) -> float:
    area = sum(ring_area_m2(ring) for ring in outers) - sum(ring_area_m2(ring) for ring in inners)
    return max(area, 0.0)


def ring_centroid(ring) -> tuple[float, float]:
    """Area-weighted centroid of a ring of (lat, lon); falls back to the vertex mean."""
    if not ring:
        raise ValueError("empty ring")
    lat0 = sum(point[0] for point in ring) / len(ring)
    lon0 = sum(point[1] for point in ring) / len(ring)
    points = _project(ring, lat0, lon0)
    twice = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        cross = x1 * y2 - x2 * y1
        twice += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(twice) < 1e-9:
        return lat0, lon0
    cx /= 3.0 * twice
    cy /= 3.0 * twice
    scale = math.pi / 180.0 * EARTH_RADIUS_M
    return lat0 + cy / scale, lon0 + cx / (scale * math.cos(math.radians(lat0)))


def point_in_ring(lat: float, lon: float, ring) -> bool:
    """Even-odd rule on (lat, lon) vertices; adequate for building-sized rings."""
    inside = False
    count = len(ring)
    for index in range(count):
        lat_i, lon_i = ring[index]
        lat_j, lon_j = ring[index - 1]
        if (lat_i > lat) != (lat_j > lat):
            crossing = lon_i + (lat - lat_i) * (lon_j - lon_i) / (lat_j - lat_i)
            if lon < crossing:
                inside = not inside
    return inside


def point_in_footprint(lat: float, lon: float, footprint) -> bool:
    """True when the point is inside an outer ring and outside every inner ring."""
    if not footprint:
        return False
    outers, inners = footprint
    if not any(point_in_ring(lat, lon, ring) for ring in outers):
        return False
    return not any(point_in_ring(lat, lon, ring) for ring in inners)
