"""Decide whether an agent-enriched asset may replace the candidate.

Content Agents add appearance and physical priors on top of an asset Blueprint
authored. That division only holds if it is checked: the 840796 pass showed an
agent applying per-component mass to twenty-six prims, which was harmless, and
a Material Agent rebinding surfaces, which was the point. Nothing in either run
verified that the articulation, the authored link masses, or the reachable
interior survived.

This gate answers one question - is the enriched asset still the asset we
qualified? - by comparing it against the baseline it was derived from. Agents
may add materials, textures, and priors freely. They may not change the joint
graph, move a mass Blueprint authored, seal the interior, or drop a render
binding a later texture stage depends on. Any of those and the enrichment is
rejected rather than silently promoted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .articulated_interior_exposure import (
    ArticulatedInteriorExposureError,
    evaluate_interior_exposure,
)
from .common import write_json
from .decision_evidence_contracts import canonical_digest


AGENT_ENRICHMENT_ACCEPTANCE_SCHEMA_VERSION = "agent_enrichment_acceptance.v1"
DEFAULT_MASS_TOLERANCE_KG = 1e-6


class AgentEnrichmentAcceptanceError(ValueError):
    """Stable, sorted acceptance-gate failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _articulation(stage) -> dict[str, Any]:
    from pxr import UsdPhysics

    joints = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        axis = prim.GetAttribute("physics:axis")
        revolute = UsdPhysics.RevoluteJoint(prim) if prim.IsA(UsdPhysics.RevoluteJoint) else None
        joints[str(prim.GetPath())] = {
            "type": "revolute" if revolute else "other",
            "axis": str(axis.Get()) if axis and axis.HasAuthoredValue() else "",
            "lower": float(revolute.GetLowerLimitAttr().Get())
            if revolute and revolute.GetLowerLimitAttr().Get() is not None
            else None,
            "upper": float(revolute.GetUpperLimitAttr().Get())
            if revolute and revolute.GetUpperLimitAttr().Get() is not None
            else None,
        }
    return {
        "joints": dict(sorted(joints.items())),
        "articulation_roots": sorted(
            str(p.GetPath())
            for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.ArticulationRootAPI)
        ),
        "rigid_bodies": sorted(
            str(p.GetPath())
            for p in stage.Traverse()
            if p.HasAPI(UsdPhysics.RigidBodyAPI)
        ),
    }


def _link_masses(stage) -> dict[str, float]:
    from pxr import UsdPhysics

    masses: dict[str, float] = {}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        value = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        if value is not None:
            masses[str(prim.GetPath())] = float(value)
    return dict(sorted(masses.items()))


def _bound_render_materials(stage, prim_paths: Sequence[str]) -> dict[str, str | None]:
    from pxr import UsdShade

    out: dict[str, str | None] = {}
    for path in prim_paths:
        prim = stage.GetPrimAtPath(str(path))
        if not prim.IsValid():
            out[str(path)] = None
            continue
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        target = bound.GetPrim() if bound else None
        out[str(path)] = str(target.GetPath()) if target and target.IsValid() else None
    return dict(sorted(out.items()))


def accept_agent_enriched_asset(
    *,
    baseline_usd_path: str | Path,
    enriched_usd_path: str | Path,
    support_link_path: str,
    task_door_link_path: str,
    interior_prim_paths: Sequence[str],
    aperture_plane_y_m: float,
    aperture_x_interval_m: Sequence[float],
    aperture_z_interval_m: Sequence[float],
    required_render_material_prim_paths: Sequence[str] = (),
    samples_per_axis: int = 11,
    mass_tolerance_kg: float = DEFAULT_MASS_TOLERANCE_KG,
    destination: str | Path | None = None,
) -> dict[str, Any]:
    """Compare an enriched asset against its baseline and accept or reject it."""

    try:
        from pxr import Usd
    except ImportError as exc:  # pragma: no cover - environment guard
        raise AgentEnrichmentAcceptanceError(
            ["agent_enrichment_openusd_runtime_missing"]
        ) from exc

    baseline = Path(baseline_usd_path).expanduser().resolve()
    enriched = Path(enriched_usd_path).expanduser().resolve()
    if not baseline.is_file():
        raise AgentEnrichmentAcceptanceError(["agent_enrichment_baseline_missing"])
    if not enriched.is_file():
        raise AgentEnrichmentAcceptanceError(["agent_enrichment_enriched_asset_missing"])

    baseline_stage = Usd.Stage.Open(str(baseline))
    enriched_stage = Usd.Stage.Open(str(enriched))
    if baseline_stage is None or enriched_stage is None:
        raise AgentEnrichmentAcceptanceError(["agent_enrichment_asset_unreadable"])

    blockers: list[str] = []
    before = _articulation(baseline_stage)
    after = _articulation(enriched_stage)
    articulation_preserved = before == after
    if not articulation_preserved:
        blockers.append("agent_enrichment_articulation_changed")

    baseline_masses = _link_masses(baseline_stage)
    enriched_masses = _link_masses(enriched_stage)
    mass_rows: dict[str, dict[str, float | None]] = {}
    masses_unchanged = True
    for path, value in baseline_masses.items():
        observed = enriched_masses.get(path)
        row = {"baseline_kg": value, "enriched_kg": observed}
        mass_rows[path] = row
        if observed is None or abs(observed - value) > float(mass_tolerance_kg):
            masses_unchanged = False
    if not masses_unchanged:
        blockers.append("agent_enrichment_authored_link_mass_changed")

    required = [str(path) for path in required_render_material_prim_paths]
    bindings = _bound_render_materials(enriched_stage, required)
    materials_bound = all(value for value in bindings.values()) if required else True
    if not materials_bound:
        blockers.append("agent_enrichment_render_material_unbound")

    exposure: dict[str, Any] | None = None
    interior_exposed = False
    try:
        exposure = evaluate_interior_exposure(
            replacement_usd_path=enriched,
            support_link_path=support_link_path,
            task_door_link_path=task_door_link_path,
            interior_prim_paths=interior_prim_paths,
            aperture_plane_y_m=aperture_plane_y_m,
            aperture_x_interval_m=aperture_x_interval_m,
            aperture_z_interval_m=aperture_z_interval_m,
            samples_per_axis=samples_per_axis,
        )
        interior_exposed = bool(exposure["interior_exposed"])
    except ArticulatedInteriorExposureError as exc:
        blockers.append(
            "agent_enrichment_interior_exposure_uncheckable:" + ";".join(exc.errors)
        )
    if exposure is not None and not interior_exposed:
        blockers.append("agent_enrichment_interior_no_longer_exposed")

    accepted = not blockers
    receipt: dict[str, Any] = {
        "schema_version": AGENT_ENRICHMENT_ACCEPTANCE_SCHEMA_VERSION,
        "status": "agent_enrichment_accepted"
        if accepted
        else "agent_enrichment_rejected",
        "accepted": accepted,
        "baseline_usd_path": str(baseline),
        "baseline_usd_sha256": _sha256(baseline),
        "enriched_usd_path": str(enriched),
        "enriched_usd_sha256": _sha256(enriched),
        "checks": {
            "articulation_preserved": articulation_preserved,
            "authored_link_masses_unchanged": masses_unchanged,
            "interior_still_exposed": interior_exposed,
            "render_materials_still_bound": materials_bound,
        },
        "articulation": {"baseline": before, "enriched": after},
        "link_masses": mass_rows,
        "render_material_bindings": bindings,
        "interior_exposure": exposure,
        "blockers": sorted(set(blockers)),
        "claim_boundary": {
            "agents_may_add_priors_not_rewrite_authored_physics": True,
            "agent_priors_are_estimates_not_measurements": True,
            "native_simulator_qualified": False,
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    if destination is not None:
        write_json(Path(destination).expanduser().resolve(), receipt)
    return json.loads(json.dumps(receipt))


__all__ = [
    "AGENT_ENRICHMENT_ACCEPTANCE_SCHEMA_VERSION",
    "AgentEnrichmentAcceptanceError",
    "accept_agent_enriched_asset",
]
