"""Deduplicate source records into sites (pure).

Every merge edge needs a name match, so two businesses at one address stay two
sites. A name match is either a token-set ratio at or above the rule's minimum
together with a shared distinctive token (not an industry word such as
CLEANER or STEEL), or a ratio of at least ``NEAR_IDENTICAL`` on its own.

Edges, in order:

1. ``address_key``: same normalized street, city, state and ZIP; ratio at
   least ``ADDRESS_NAME_MIN``.
2. ``postal_house_number_name``: same ZIP and house number and similar street
   names (one street written two ways); ratio at least ``NAME_MIN``.
3. ``proximity_name``: precise coordinates within ``PROXIMITY_M`` metres, or a
   point inside another record's OpenStreetMap footprint; ratio at least
   ``NAME_MIN``. Approximate and stacked coordinates (ZIP or area centroids)
   never take part.
4. ``name_postal_no_address``: neither record has a street address or a
   precise coordinate; same ZIP and identical name tokens, at least one of
   them distinctive.

Records join by union-find. Only edges that join two different groups are
recorded, so each site carries one reason per merge. A join is refused when
the joined group would hold two records with the same address key that name
different businesses: their names do not match (rule 1) and neither record's
operator matches the other record. So a third name that resembles two
tenants of one building cannot chain them into one site, while a store filed
under a store number by its operator still joins its other records. Within
each pass the edges are applied in a fixed order (best name score first, then
record references), so the partition does not depend on the input order.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from tools.site_universe import geo, normalize

ADDRESS_NAME_MIN = 60.0
NAME_MIN = 85.0
NEAR_IDENTICAL = 92.0
STREET_MIN = 60.0
PROXIMITY_M = 75.0
STACKED_COORDINATE_MIN = 5
MAX_FOOTPRINT_CELLS = 40_000
_CELL_DEG = 0.001


@dataclass
class Prepared:
    index: int
    record: object
    profile: normalize.NameProfile
    address_key: str | None
    zip5: str | None
    house: str | None
    street_tokens: tuple
    point: tuple | None
    operator_profile: normalize.NameProfile | None = None


@dataclass
class Cluster:
    members: list
    merges: list = field(default_factory=list)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: int, b: int) -> bool:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        if root_b < root_a:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        return True


def matching_tokens(record) -> tuple:
    return tuple(sorted(normalize.name_profile(record.name, city=record.city).tokens))


def names_match(score: float, strong: bool, minimum: float) -> bool:
    return (strong and score >= minimum) or score >= NEAR_IDENTICAL


def prepare(records) -> list[Prepared]:
    stacked = defaultdict(int)
    for record in records:
        if record.lat is not None and record.lon is not None:
            stacked[(record.source_id, record.lat, record.lon)] += 1
    prepared = []
    for index, record in enumerate(records):
        point = None
        if (
            record.lat is not None
            and record.lon is not None
            and record.attributes.get("coordinate_precision") != "approximate"
            and stacked[(record.source_id, record.lat, record.lon)] < STACKED_COORDINATE_MIN
        ):
            point = (record.lat, record.lon)
        house = normalize.house_number(record.street)
        prepared.append(
            Prepared(
                index=index,
                record=record,
                profile=normalize.name_profile(record.name, city=record.city),
                address_key=normalize.address_key(
                    record.street, record.city, record.state, record.postal_code
                ),
                zip5=record.postal_code,
                house=house,
                street_tokens=tuple(record.street.split()[1:]) if house else (),
                point=point,
                operator_profile=(
                    normalize.name_profile(record.operator, city=record.city) if record.operator else None
                ),
            )
        )
    return prepared


def _cell(lat: float, lon: float) -> tuple[int, int]:
    return (math.floor(lat / _CELL_DEG), math.floor(lon / _CELL_DEG))


def _pair_key(a: Prepared, b: Prepared) -> tuple[str, str]:
    return (a.record.ref, b.record.ref) if a.record.ref <= b.record.ref else (b.record.ref, a.record.ref)


def cluster(records) -> tuple[list[Cluster], dict]:
    """Group records into clusters; return the clusters and merge statistics."""
    items = prepare(records)
    union = _UnionFind(len(items))
    edges: list[dict] = []
    stats = defaultdict(int)
    cache: dict[tuple[int, int], tuple[float, bool]] = {}
    # Group root -> address key -> the group's records with that key.
    tenants = {item.index: {item.address_key: [item]} for item in items if item.address_key}

    def similarity(a: Prepared, b: Prepared) -> tuple[float, bool]:
        key = (a.index, b.index) if a.index < b.index else (b.index, a.index)
        if key not in cache:
            cache[key] = normalize.name_similarity(a.profile, b.profile)
        return cache[key]

    same_business_cache: dict[tuple[int, int], bool] = {}

    def same_business(a: Prepared, b: Prepared) -> bool:
        """Whether two records at one address name one business: the names match, or the
        operator of one matches the name or operator of the other ('QM 4410' run by
        'Quillmoor Stores Texas LLC' and 'Quillmoor Supercenter')."""
        key = (a.index, b.index) if a.index < b.index else (b.index, a.index)
        if key not in same_business_cache:
            pairs = [(a.profile, b.operator_profile), (a.operator_profile, b.profile),
                     (a.operator_profile, b.operator_profile)]
            same_business_cache[key] = names_match(*similarity(a, b), ADDRESS_NAME_MIN) or any(
                names_match(*normalize.name_similarity(left, right), ADDRESS_NAME_MIN)
                for left, right in pairs if left is not None and right is not None
            )
        return same_business_cache[key]

    def tenant_conflict(root_a: int, root_b: int) -> bool:
        """True when joining the groups would put two businesses at one address key into one site."""
        left, right = tenants.get(root_a, {}), tenants.get(root_b, {})
        if len(left) > len(right):
            left, right = right, left
        for key, members in left.items():
            for first in members:
                for second in right.get(key, ()):
                    if not same_business(first, second):
                        return True
        return False

    def join(a: Prepared, b: Prepared, reason: str, score: float, extra: dict) -> None:
        root_a, root_b = union.find(a.index), union.find(b.index)
        if root_a == root_b:
            return
        if tenant_conflict(root_a, root_b):
            stats[f"joins_refused_{reason}"] += 1
            return
        union.union(a.index, b.index)
        merged, other = tenants.pop(root_a, {}), tenants.pop(root_b, {})
        if len(merged) < len(other):
            merged, other = other, merged
        for key, members in other.items():
            merged.setdefault(key, []).extend(members)
        if merged:
            tenants[union.find(a.index)] = merged
        stats[f"merges_{reason}"] += 1
        first, second = _pair_key(a, b)
        edges.append({"a": first, "b": second, "name_score": score, "reason": reason, **extra})

    def apply(candidates: list, reason: str) -> None:
        """Join candidate edges, best name score first, then by distance and record references."""
        stats[f"edges_{reason}"] += len(candidates)
        for _, a, b, score, extra in sorted(candidates, key=lambda candidate: candidate[0]):
            join(a, b, reason, score, extra)

    # 1. Exact address key.
    by_address = defaultdict(list)
    for item in items:
        if item.address_key:
            by_address[item.address_key].append(item)
    candidates = []
    for key in sorted(by_address):
        group = by_address[key]
        for position, first in enumerate(group):
            for second in group[position + 1 :]:
                score, strong = similarity(first, second)
                if names_match(score, strong, ADDRESS_NAME_MIN):
                    candidates.append(((-score, _pair_key(first, second)), first, second, score, {}))
                else:
                    stats["address_pairs_kept_apart_by_name"] += 1
    apply(candidates, "address_key")

    # 2. Same ZIP and house number, street written differently.
    by_house = defaultdict(list)
    for item in items:
        if item.zip5 and item.house:
            by_house[(item.zip5, item.house)].append(item)
    candidates = []
    for key in sorted(by_house):
        group = by_house[key]
        for position, first in enumerate(group):
            for second in group[position + 1 :]:
                if first.address_key == second.address_key:
                    continue
                street_score = normalize.token_set_ratio(first.street_tokens, second.street_tokens)
                if street_score < STREET_MIN:
                    continue
                score, strong = similarity(first, second)
                if names_match(score, strong, NAME_MIN):
                    candidates.append(((-score, -street_score, _pair_key(first, second)), first, second,
                                       score, {"street_score": street_score}))
    apply(candidates, "postal_house_number_name")

    # 3. Proximity, or a point inside an OpenStreetMap footprint.
    grid = defaultdict(list)
    for item in items:
        if item.point:
            grid[_cell(*item.point)].append(item)
    candidates = []
    for item in items:
        if not item.point:
            continue
        row, column = _cell(*item.point)
        for d_row in (-1, 0, 1):
            for d_column in (-1, 0, 1):
                for other in grid.get((row + d_row, column + d_column), ()):
                    if other.index <= item.index:
                        continue
                    distance = geo.haversine_m(*item.point, *other.point)
                    if distance > PROXIMITY_M:
                        continue
                    score, strong = similarity(item, other)
                    if names_match(score, strong, NAME_MIN):
                        distance = round(distance, 1)
                        candidates.append(((-score, distance, _pair_key(item, other)), item, other, score,
                                           {"distance_m": distance}))
    apply(candidates, "proximity_name")
    candidates = []
    for item in items:
        footprint = item.record.footprint
        if not footprint:
            continue
        lats = [lat for ring in footprint[0] for lat, _ in ring]
        lons = [lon for ring in footprint[0] for _, lon in ring]
        low_row, low_column = _cell(min(lats), min(lons))
        high_row, high_column = _cell(max(lats), max(lons))
        if (high_row - low_row + 1) * (high_column - low_column + 1) > MAX_FOOTPRINT_CELLS:
            stats["footprints_too_large_for_containment"] += 1
            continue
        for cell_row in range(low_row, high_row + 1):
            for cell_column in range(low_column, high_column + 1):
                for other in grid.get((cell_row, cell_column), ()):
                    if other.index == item.index or union.find(other.index) == union.find(item.index):
                        continue
                    score, strong = similarity(item, other)
                    if not names_match(score, strong, NAME_MIN):
                        continue
                    if geo.point_in_footprint(other.point[0], other.point[1], footprint):
                        candidates.append(((-score, _pair_key(item, other)), item, other, score,
                                           {"distance_m": 0.0, "inside_footprint": True}))
    apply(candidates, "proximity_name")

    # 4. No street address on either side: identical names in one ZIP.
    by_postal_name = defaultdict(list)
    for item in items:
        if not item.address_key and not item.point and item.zip5 and item.profile.tokens:
            by_postal_name[(item.zip5, tuple(sorted(item.profile.tokens)))].append(item)
    candidates = []
    for key in sorted(by_postal_name):
        if not any(token not in normalize.WEAK_NAME_TOKENS for token in key[1]):
            continue
        group = sorted(by_postal_name[key], key=lambda item: item.record.ref)
        for other in group[1:]:
            candidates.append(((-100.0, _pair_key(group[0], other)), group[0], other, 100.0, {}))
    apply(candidates, "name_postal_no_address")

    groups = defaultdict(list)
    for item in items:
        groups[union.find(item.index)].append(item.index)
    ref_to_index = {item.record.ref: item.index for item in items}
    edges_by_root = defaultdict(list)
    for edge in edges:
        edges_by_root[union.find(ref_to_index[edge["a"]])].append(edge)
    clusters = [
        Cluster(members=[items[i].record for i in groups[root]], merges=edges_by_root[root])
        for root in sorted(groups)
    ]
    stats["address_keys_with_several_sites"] = sum(
        1 for group in by_address.values() if len({union.find(item.index) for item in group}) > 1
    )
    stats["records_in"] = len(items)
    stats["sites_out"] = len(clusters)
    stats["records_merged_away"] = len(items) - len(clusters)
    stats["records_with_proximity_coordinates"] = sum(1 for item in items if item.point)
    stats["records_with_address_key"] = sum(1 for item in items if item.address_key)
    return clusters, dict(sorted(stats.items()))
