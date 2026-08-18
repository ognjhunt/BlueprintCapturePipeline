"""Give the deterministic review named links instead of anonymous blobs.

The Joint Agent review anchors link membership on the source asset receipt's
connected components: bounding boxes recovered by partitioning one mesh, each
identified only by an index.  That is what you get when the input is an
undifferentiated mesh, and on 2026-08-17 it is also what made Scene 840920's
paid run useless -- the agent was asked to name parents among parts that did
not exist as prims, and resolved none of them.

Feeding it our authored replacement fixes the question, and this closes the
last gap: the replacement has no connected components to project, because its
parts are already separate.  It has six named links.

So this synthesizes the review's own input shape from those links, one
component per link, with the bounds computed from the link's authored collision
geometry.  Oriented boxes are rotated corner by corner rather than approximated,
and cylinders are bounded by the box that contains them, so a component is never
smaller than the geometry it stands for -- an undersized bound would silently
drop a joint out of its own link.

The result is strictly better evidence than the mesh partition it replaces.
Membership stops being "this blob overlaps where a door should be" and becomes
"this is the door", because the link carries its name from the authored graph.
Nothing here infers topology: it reports bounds for links we authored, and the
admission already caps the claim accordingly.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "authored_link_source_components.v1"

#: Collision primitives an authored link may use.  Anything else fails closed:
#: a bound we cannot compute is not a bound we may guess.
SUPPORTED_GEOMETRY_KINDS = ("box", "cylinder", "sphere", "capsule")


class AuthoredLinkSourceComponentsError(ValueError):
    """Fail-closed refusal to synthesize bounds we cannot compute."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(error) for error in errors if str(error)}))
        super().__init__(";".join(self.errors))


def _vector(value: Any, length: int = 3) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        out.append(number)
    return out


def _rotate(quaternion: Sequence[float], point: Sequence[float]) -> list[float]:
    """Rotate a point by an xyzw quaternion."""

    x, y, z, w = quaternion
    # v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + w * v)
    cx = y * point[2] - z * point[1] + w * point[0]
    cy = z * point[0] - x * point[2] + w * point[1]
    cz = x * point[1] - y * point[0] + w * point[2]
    return [
        point[0] + 2.0 * (y * cz - z * cy),
        point[1] + 2.0 * (z * cx - x * cz),
        point[2] + 2.0 * (x * cy - y * cx),
    ]


def _half_extent(geometry: Mapping[str, Any]) -> list[float] | None:
    """Half extents of the axis-aligned box that contains this primitive."""

    kind = str(geometry.get("kind") or "")
    if kind == "box":
        size = _vector(geometry.get("size_m"))
        if size is None or any(value <= 0.0 for value in size):
            return None
        return [value / 2.0 for value in size]
    if kind == "sphere":
        radius = geometry.get("radius_m")
        if isinstance(radius, bool) or not isinstance(radius, (int, float)):
            return None
        radius = float(radius)
        return [radius] * 3 if radius > 0.0 and math.isfinite(radius) else None
    if kind in ("cylinder", "capsule"):
        radius = geometry.get("radius_m")
        height = geometry.get("height_m")
        if (
            isinstance(radius, bool)
            or isinstance(height, bool)
            or not isinstance(radius, (int, float))
            or not isinstance(height, (int, float))
        ):
            return None
        radius = float(radius)
        height = float(height)
        if radius <= 0.0 or height <= 0.0:
            return None
        # A capsule's caps add a radius at each end; bounding it as a cylinder
        # would cut them off.
        half_height = height / 2.0 + (radius if kind == "capsule" else 0.0)
        return [radius, radius, half_height]
    return None


def _geometry_bounds(
    geometry: Mapping[str, Any],
    *,
    rest_translation: Sequence[float] = (0.0, 0.0, 0.0),
    rest_orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
) -> tuple[list[float], list[float]] | None:
    """Asset-frame bounds of one collision primitive on a posed link."""

    half = _half_extent(geometry)
    translation = _vector(geometry.get("translation_m")) or [0.0, 0.0, 0.0]
    orientation = _vector(geometry.get("orientation_xyzw"), 4) or [0.0, 0.0, 0.0, 1.0]
    if half is None:
        return None
    norm = math.sqrt(sum(value * value for value in orientation))
    if norm <= 0.0:
        return None
    orientation = [value / norm for value in orientation]

    lower = [math.inf] * 3
    upper = [-math.inf] * 3
    for index in range(8):
        corner = [
            half[axis] if (index >> axis) & 1 else -half[axis] for axis in range(3)
        ]
        local = _rotate(orientation, corner)
        local = [local[axis] + translation[axis] for axis in range(3)]
        rotated = _rotate(rest_orientation, local)
        for axis in range(3):
            value = rotated[axis] + rest_translation[axis]
            lower[axis] = min(lower[axis], value)
            upper[axis] = max(upper[axis], value)
    return lower, upper


def build_authored_link_source_components(
    *,
    spec: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Return review-shaped components, one per authored link.

    The spec supplies the geometry and the receipt supplies the link prims the
    bundle will actually ship, so both are required and they must agree: the
    receipt names the spec it was authored from, and every link in one has to
    appear in the other.  A link present in only one of them means the two
    documents describe different assets, and no bound synthesized across that
    gap would mean anything.
    """

    errors: list[str] = []
    spec_digest = receipt.get("spec_digest")
    if not isinstance(spec_digest, str) or spec.get("spec_digest") != spec_digest:
        errors.append("authored_link_components_spec_receipt_mismatch")
    if spec.get("spec_digest") != canonical_digest(spec, digest_field="spec_digest"):
        errors.append("authored_link_components_spec_digest_invalid")
    if receipt.get("receipt_digest") != canonical_digest(
        receipt, digest_field="receipt_digest"
    ):
        errors.append("authored_link_components_receipt_digest_invalid")

    link_paths = receipt.get("link_paths")
    if not isinstance(link_paths, Mapping) or not link_paths:
        errors.append("authored_link_components_link_paths_invalid")
        link_paths = {}
    links = spec.get("links")
    if not isinstance(links, list) or not links:
        errors.append("authored_link_components_spec_links_invalid")
        links = []
    spec_ids = {
        str(link.get("link_id"))
        for link in links
        if isinstance(link, Mapping) and link.get("link_id")
    }
    if link_paths and spec_ids and spec_ids != set(map(str, link_paths)):
        errors.append("authored_link_components_link_set_mismatch")
    if errors:
        raise AuthoredLinkSourceComponentsError(errors)

    components: list[dict[str, Any]] = []
    # Sorted by link id so the component index is a property of the asset, not
    # of document order: the same asset must always produce the same indices.
    for index, link_id in enumerate(sorted(spec_ids)):
        link = next(
            link
            for link in links
            if isinstance(link, Mapping) and str(link.get("link_id")) == link_id
        )
        # Geometry is authored in the link's own frame; the review anchors
        # membership in the asset frame. Skipping this places every child link
        # at the origin -- the door would sit inside the drum, and membership
        # would be assigned to whichever blob happened to overlap.
        rest_pose = link.get("rest_pose")
        rest_translation = (
            _vector((rest_pose or {}).get("translation_m"))
            if isinstance(rest_pose, Mapping)
            else None
        )
        rest_orientation = (
            _vector((rest_pose or {}).get("orientation_xyzw"), 4)
            if isinstance(rest_pose, Mapping)
            else None
        )
        if rest_translation is None or rest_orientation is None:
            errors.append(f"authored_link_components_rest_pose_invalid:{link_id}")
            continue
        rest_norm = math.sqrt(sum(value * value for value in rest_orientation))
        if rest_norm <= 0.0:
            errors.append(f"authored_link_components_rest_pose_invalid:{link_id}")
            continue
        rest_orientation = [value / rest_norm for value in rest_orientation]

        geometries = link.get("geometry")
        if not isinstance(geometries, list) or not geometries:
            errors.append(f"authored_link_components_geometry_missing:{link_id}")
            continue
        lower = [math.inf] * 3
        upper = [-math.inf] * 3
        for position, geometry in enumerate(geometries):
            if not isinstance(geometry, Mapping) or str(
                geometry.get("kind") or ""
            ) not in SUPPORTED_GEOMETRY_KINDS:
                errors.append(
                    f"authored_link_components_geometry_kind_unsupported:{link_id}:{position}"
                )
                continue
            bounds = _geometry_bounds(
                geometry,
                rest_translation=rest_translation,
                rest_orientation=rest_orientation,
            )
            if bounds is None:
                errors.append(
                    f"authored_link_components_geometry_invalid:{link_id}:{position}"
                )
                continue
            for axis in range(3):
                lower[axis] = min(lower[axis], bounds[0][axis])
                upper[axis] = max(upper[axis], bounds[1][axis])
        if any(not math.isfinite(value) for value in [*lower, *upper]):
            errors.append(f"authored_link_components_bounds_unresolved:{link_id}")
            continue
        components.append(
            {
                "component_index": index,
                "link_id": link_id,
                "link_prim_path": str(link_paths[link_id]),
                "aabb_min_asset_m": lower,
                "aabb_max_asset_m": upper,
            }
        )
    if errors:
        raise AuthoredLinkSourceComponentsError(errors)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "authored_link_bounds_synthesized",
        "spec_digest": str(spec_digest),
        "source_receipt_digest": str(receipt.get("receipt_digest")),
        "connected_component_count": len(components),
        "connected_components": components,
        "claim_boundary": {
            # Bounds for links we authored. The review may anchor membership on
            # them; it may not read them as independent observation.
            "connected_components_are_not_rigid_links": False,
            "components_are_authored_links": True,
            "independent_topology_inference": False,
            "joint_topology_qualified": False,
            "simready_qualified": False,
            "physical_equivalence_proven": False,
        },
        "provider_mutation_performed": False,
        "spend_incurred_usd": 0.0,
        "components_digest": "",
    }
    payload["components_digest"] = canonical_digest(
        payload, digest_field="components_digest"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        spec = json.loads(Path(args.spec).expanduser().read_text(encoding="utf-8"))
        receipt = json.loads(
            Path(args.receipt).expanduser().read_text(encoding="utf-8")
        )
        payload = build_authored_link_source_components(spec=spec, receipt=receipt)
    except (OSError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": ["authored_link_components_input_invalid"],
                    "provider_mutation_performed": False,
                    "detail": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    except AuthoredLinkSourceComponentsError as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": list(exc.errors),
                    "provider_mutation_performed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "components_digest": payload["components_digest"],
                "connected_component_count": payload["connected_component_count"],
                "output": str(destination),
                "provider_mutation_performed": False,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_GEOMETRY_KINDS",
    "AuthoredLinkSourceComponentsError",
    "build_authored_link_source_components",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
