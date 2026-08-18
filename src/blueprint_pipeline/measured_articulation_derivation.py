"""Derive articulation from the scan, so nobody ever types physics again.

Scene 840920's washer door was sealed with a hand-typed hinge axis of ``+Z``
where the measured geometry demands ``-Z``.  Two paid runs read the jammed
result -- 6.01 degrees at every commanded angle -- and correcting it meant
amending a prospectively sealed freeze.  The same week, scene 840796's fridge
twin was authored by partitioning the frozen scan along measured planes, and
its hinge never had this failure mode available: every parameter was an
observation.

This module restores that principle as a general contract:

  * **The scan is the only author of physical claims.**  Separating planes are
    found in the vertex distribution of the frozen source bytes; part clusters
    are the vertices a measured plane isolates; hinge columns and pivots are
    extremes of measured clusters.
  * **A joint's axis sign is never an input.**  "Opening" means the commanded
    travel increases clearance from the parent, so the sign is computed from
    that requirement.  An inverted hinge is not a bug this representation can
    express.
  * **Replayable.**  Same frozen bytes, same sealed parameters, byte-identical
    derivation receipt.  Agents may *propose* where to look; nothing they say
    becomes input until the measurement confirms it.

The derivation emits measurements and a derived articulation in the same
vocabulary the graph-asset author consumes, so the proven downstream chain --
visual binding, composition, registration, qualification, import, probe --
is unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest


SCHEMA_VERSION = "measured_articulation_derivation.v1"

#: Vertex-histogram bin for plane finding, metres.  Coarse enough to survive
#: scan noise, fine enough to separate a door shell from its cabinet face.
PLANE_BIN_M = 0.01

#: A separating plane must clear its neighbours by this many vertices to count
#: as a plate rather than noise.
MIN_PLATE_VERTICES = 100




class MeasuredArticulationError(ValueError):
    """Fail-closed refusal to derive from unusable measurements."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted({str(e) for e in errors if str(e)}))
        super().__init__(";".join(self.errors))


def _load_world_points(
    source_usd: Path,
) -> tuple[list[tuple[float, float, float]], int]:
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - environment guard
        raise MeasuredArticulationError(["openusd_runtime_missing"]) from exc

    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise MeasuredArticulationError(["measured_source_unreadable"])
    up_token = UsdGeom.GetStageUpAxis(stage)
    up_axis = {"X": 0, "Y": 1, "Z": 2}.get(str(up_token))
    if up_axis is None:
        raise MeasuredArticulationError(["measured_stage_up_axis_unreadable"])
    points: list[tuple[float, float, float]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        for point in mesh.GetPointsAttr().Get() or []:
            world = transform.Transform(Gf.Vec3d(point))
            points.append((float(world[0]), float(world[1]), float(world[2])))
    if len(points) < MIN_PLATE_VERTICES:
        raise MeasuredArticulationError(["measured_source_too_sparse"])
    return points, up_axis


def measure_front_plate_plane(
    points: Sequence[tuple[float, float, float]],
    *,
    axis: int = 1,
    facing_outward_sign: int,
    facing_proposed_by: str,
) -> dict[str, Any]:
    """Validate a proposed facing direction and measure its plate.

    The facing direction is a PROPOSAL, never a measurement: on the real
    840920 scan the rear jumble of hoses and panels out-shells the actual
    door 5807 vertices to 1850, so any dominant-shell auto-pick chooses the
    back of the machine.  The proposal comes from scene context or an agent
    looking at renders, carries its provenance, and is verified here -- the
    proposed side must actually hold a plate and a protruding shell, and the
    receipt records both sides' measurements so the decision is inspectable.
    Everything downstream of the (verified) proposal is measured.
    """

    if facing_outward_sign not in (-1, 1):
        raise MeasuredArticulationError(["measured_facing_proposal_invalid"])
    if not str(facing_proposed_by or "").strip():
        raise MeasuredArticulationError(["measured_facing_provenance_required"])
    centroid = sum(p[axis] for p in points) / len(points)
    candidates = []
    for outward in (-1, 1):
        half = [p for p in points if (p[axis] - centroid) * outward > 0]
        if len(half) < MIN_PLATE_VERTICES:
            continue
        histogram = Counter(
            round(p[axis] / PLANE_BIN_M) * PLANE_BIN_M for p in half
        )
        plane_value, plate_vertices = max(histogram.items(), key=lambda kv: kv[1])
        if plate_vertices < MIN_PLATE_VERTICES:
            continue
        shell_vertices = sum(
            1 for p in half if (p[axis] - plane_value) * outward > PLANE_BIN_M / 2
        )
        candidates.append(
            {
                "axis_index": axis,
                "outward_sign": outward,
                "plane_m": round(plane_value, 6),
                "plate_vertex_count": plate_vertices,
                "shell_vertex_count": shell_vertices,
                "measurement": "front_plate_histogram_peak",
                "bin_m": PLANE_BIN_M,
            }
        )
    proposed = [c for c in candidates if c["outward_sign"] == facing_outward_sign]
    if not proposed or proposed[0]["shell_vertex_count"] < MIN_PLATE_VERTICES:
        # The proposal named a side with no plate or no protruding shell:
        # refuse it rather than quietly measuring a different face.
        raise MeasuredArticulationError(["measured_facing_proposal_refuted"])
    selected = dict(proposed[0])
    selected["facing_proposed_by"] = str(facing_proposed_by).strip()
    selected["facing_is_proposal_not_measurement"] = True
    selected["other_side_measurements"] = [
        {k: c[k] for k in ("outward_sign", "plane_m", "shell_vertex_count")}
        for c in candidates
        if c["outward_sign"] != facing_outward_sign
    ]
    return selected


def measure_forward_shell(
    points: Sequence[tuple[float, float, float]],
    *,
    plane: Mapping[str, Any],
    margin_m: float = 0.005,
) -> dict[str, Any]:
    """Everything protruding beyond the measured plate: the door assembly."""

    axis = int(plane["axis_index"])
    outward = int(plane["outward_sign"])
    limit = float(plane["plane_m"]) + outward * margin_m
    shell = [p for p in points if (p[axis] - limit) * outward > 0]
    if len(shell) < MIN_PLATE_VERTICES:
        raise MeasuredArticulationError(["measured_forward_shell_too_sparse"])
    lateral = [i for i in range(3) if i != axis]
    a, b = lateral
    return {
        "vertex_count": len(shell),
        "extent_min_m": [
            round(min(p[i] for p in shell), 6) for i in range(3)
        ],
        "extent_max_m": [
            round(max(p[i] for p in shell), 6) for i in range(3)
        ],
        "lateral_axes": [a, b],
        "front_m": round(
            (max if outward > 0 else min)(p[axis] for p in shell), 6
        ),
        "outward_sign": outward,
        "measurement": "vertices_beyond_front_plate",
        "margin_m": margin_m,
    }


def derive_hinge(
    *,
    plane: Mapping[str, Any],
    shell: Mapping[str, Any],
    up_axis: int,
    hinge_side: str = "left",
    swing_probe_deg: float = 10.0,
) -> dict[str, Any]:
    """Hinge pivot at the shell rim; axis vertical; **sign by clearance**.

    The sign is the derivation's whole point.  Both candidate signs are swept
    through a probe angle; the one that moves the shell's far edge *away* from
    the measured plate is the opening direction and becomes the axis.  The
    other is recorded, labelled with the jam it would cause, so the receipt
    shows the decision instead of asserting it.
    """

    if hinge_side not in {"left", "right"}:
        raise MeasuredArticulationError(["measured_hinge_side_invalid"])
    face_axis = int(plane["axis_index"])
    # A round door's lateral spans are equal by construction, so no span
    # heuristic can orient the hinge.  The stage's declared up-axis can: a
    # side-hinged door swings about vertical, so the hinge columns run along
    # the remaining lateral axis.  The up-axis is read from the stage, not
    # assumed -- and a face that IS the up-axis (a top-loader lid) is refused
    # here until that geometry gets its own derivation.
    if up_axis == face_axis:
        raise MeasuredArticulationError(["measured_face_axis_is_up_axis"])
    vertical_axis = up_axis
    hinge_axis_lateral = next(
        i for i in range(3) if i not in (face_axis, vertical_axis)
    )
    spans = {
        i: float(shell["extent_max_m"][i]) - float(shell["extent_min_m"][i])
        for i in (hinge_axis_lateral, vertical_axis)
    }
    edge_key = "extent_min_m" if hinge_side == "left" else "extent_max_m"
    pivot_lateral = float(shell[edge_key][hinge_axis_lateral])
    pivot_face = float(shell["front_m"])
    diameter = spans[hinge_axis_lateral]
    if diameter <= 0:
        raise MeasuredArticulationError(["measured_shell_degenerate"])

    plate = float(plane["plane_m"])
    candidates = []
    for sign in (1, -1):
        # Rotating by +probe about (vertical * sign): the far edge, a diameter
        # from the hinge, moves along the face axis by sign * d * sin(theta)
        # (orientation fixed by the right-hand rule with the lateral/face pair).
        delta_face = sign * diameter * math.sin(math.radians(swing_probe_deg))
        if hinge_side == "right":
            delta_face = -delta_face
        # The shell sits on the open side of the plate (front_m < plane_m for a
        # front-facing object), so clearance increases when the far edge moves
        # further from the plate along the shell's own side.
        away_from_plate = -1.0 if pivot_face < plate else 1.0
        opens = (delta_face * away_from_plate) > 0
        candidates.append(
            {
                "axis_sign": sign,
                "probe_deg": swing_probe_deg,
                "far_edge_delta_face_m": round(delta_face, 6),
                "verdict": "opens_clear_of_parent" if opens else "jams_into_parent",
            }
        )
    opening = [c for c in candidates if c["verdict"] == "opens_clear_of_parent"]
    if len(opening) != 1:
        raise MeasuredArticulationError(["measured_axis_sign_underdetermined"])
    axis_vector = [0.0, 0.0, 0.0]
    axis_vector[vertical_axis] = float(opening[0]["axis_sign"])

    pivot = [0.0, 0.0, 0.0]
    pivot[hinge_axis_lateral] = round(pivot_lateral, 6)
    pivot[face_axis] = round(pivot_face, 6)
    pivot[vertical_axis] = round(
        (
            float(shell["extent_min_m"][vertical_axis])
            + float(shell["extent_max_m"][vertical_axis])
        )
        / 2.0,
        6,
    )
    return {
        "pivot_asset_m": pivot,
        "axis": axis_vector,
        "hinge_side": hinge_side,
        "diameter_m": round(diameter, 6),
        "sign_candidates": candidates,
        "sign_rule": "commanded_travel_must_increase_clearance_from_parent",
    }


def derive_measured_articulation(
    *,
    source_usd_path: str | Path,
    facing_outward_sign: int,
    facing_proposed_by: str,
    hinge_side: str = "left",
) -> dict[str, Any]:
    """Verify the facing proposal, then measure and derive. No typed physics."""

    source = Path(source_usd_path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise MeasuredArticulationError(["measured_source_missing"])
    points, up_axis = _load_world_points(source)
    plane = measure_front_plate_plane(
        points,
        facing_outward_sign=facing_outward_sign,
        facing_proposed_by=facing_proposed_by,
    )
    shell = measure_forward_shell(points, plane=plane)
    hinge = derive_hinge(
        plane=plane, shell=shell, up_axis=up_axis, hinge_side=hinge_side
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "derived_from_measurement",
        "source_vertex_count": len(points),
        "stage_up_axis_index": up_axis,
        "front_plate": plane,
        "forward_shell": shell,
        "target_joint": hinge,
        "claim_boundary": {
            "physics_typed_by_hand": False,
            "facing_direction_is_verified_proposal": True,
            "axis_sign_is_derived_not_input": True,
            "simready_qualified": False,
            "physical_equivalence_proven": False,
        },
        "provider_mutation_performed": False,
        "spend_incurred_usd": 0.0,
        "derivation_digest": "",
    }
    payload["derivation_digest"] = canonical_digest(
        payload, digest_field="derivation_digest"
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-usd", required=True)
    parser.add_argument(
        "--facing-outward-sign",
        required=True,
        type=int,
        choices=[-1, 1],
        help="Proposed facing direction along the face axis; verified, never inferred.",
    )
    parser.add_argument(
        "--facing-proposed-by",
        required=True,
        help="Provenance of the facing proposal (scene context, agent, operator).",
    )
    parser.add_argument("--hinge-side", default="left", choices=["left", "right"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        payload = derive_measured_articulation(
            source_usd_path=args.source_usd,
            facing_outward_sign=args.facing_outward_sign,
            facing_proposed_by=args.facing_proposed_by,
            hinge_side=args.hinge_side,
        )
    except MeasuredArticulationError as exc:
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
                "axis": payload["target_joint"]["axis"],
                "pivot_asset_m": payload["target_joint"]["pivot_asset_m"],
                "derivation_digest": payload["derivation_digest"],
                "output": str(destination),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "SCHEMA_VERSION",
    "MeasuredArticulationError",
    "derive_hinge",
    "derive_measured_articulation",
    "measure_forward_shell",
    "measure_front_plate_plane",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
