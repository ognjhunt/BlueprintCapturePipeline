"""The look of a company simulation package, written again from the splat it was authored from.

Review, 2026-11-18: ``particlefield_usd.write_particlefield_usd`` stored every Gaussian's rotation scrambled until
that day, and nothing authors a finished job's look again. So the package (``website_company_simulation``) finds the
look's own authoring receipt (``website_native_appearance``) and the source splat it names (``look_source``), writes
the look again from that source with today's writer, in the job's own frame (``reauthor_look``), and writes the
standard 3DGS PLY of the same source, placed as that look places it (``look_ply``). CPU only, local files, buys
nothing. Anything that stops it is returned as a reason, never raised: the package then ships the look as its half
holds it, with no PLY, and its README says so.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import company_share_permission as permission
from .decision_evidence_contracts import canonical_digest

#: The largest source splat written again in the pass. The ParticleField writer peaks at 5.5 times its source at
#: SH degree 0 and 4.1 times at degree 3 (2M Gaussians, measured 2026-11-18: 586 MiB and 1.8 GiB), so 512 MiB stays
#: near 3 GiB, inside the preview unit's MemoryMax=5.5G with the rest of the pass; 2M Gaussians at degree 3 are
#: 450 MiB. The PLY is written a chunk at a time (``write_placed_viewer_ply``) and needs no bound of its own.
REAUTHOR_MAX_SOURCE_BYTES = 512 * 1024 * 1024
#: Where the website's look authoring (``website_native_appearance``) seals its receipts in each preparation
#: workspace: the runtime inputs' own folder, or one below it (``native``, ``room_package``, a floor's).
APPEARANCE_RECEIPTS = ("appearance/*/appearance.json", "*/appearance/*/appearance.json")
NATIVE_APPEARANCE_SCHEMA = "website_native_appearance.v1"


def look_source(capture_root: Path, look_sha256: str) -> dict[str, Any] | None:
    """The splat and frame the job's look was authored from (``website_native_appearance``), or None.

    Found by the look's digest among the sealed receipts in the capture's preparation workspaces. The source is
    the standard PLY the receipt names by digest: the decoded copy beside the receipt, else a base scene's splat
    recorded with that digest (the levelled splat, or the provider's it was levelled from). Small records only:
    no splat is hashed here; the build checks the source's bytes before it writes anything from them. ``path`` is
    None when the receipt is found and its source is not.
    ``first_rotations_scrambled``: the receipt's writer predates 2026-11-18 (``orientation_storage``), so the
    look as first written holds every Gaussian's rotation in the wrong order.
    """
    from .website_company_simulation import _DIGEST, _inside, _read
    root = Path(capture_root)
    preparation = root / "pipeline" / "website_scene_preparation"
    try:
        workspaces = [preparation, *sorted(path for path in (preparation / "revisions").glob("*")
                                           if path.is_dir() and not path.is_symlink())]
    except OSError:
        return None
    bases: list[tuple[Any, Path]] = []
    named = _read(preparation / "handoff.json").get("base_scene_path")
    for record in [*(workspace / "base_scene.json" for workspace in workspaces),
                   *([Path(named)] if isinstance(named, str) else [])]:
        found = _inside(root, record)
        base = _read(found) if found is not None else {}
        levelling = base.get("levelling") if isinstance(base.get("levelling"), Mapping) else {}
        for splat, digest in ((base.get("splat_path"), base.get("splat_digest")),
                              (levelling.get("provider_splat_path"), levelling.get("provider_splat_digest"))):
            path = _inside(root, Path(splat)) if isinstance(splat, str) else None
            if path is not None:
                bases.append((digest, path))
    for workspace in workspaces:
        for pattern in APPEARANCE_RECEIPTS:
            for path in sorted(workspace.glob(pattern)):
                receipt = _read(path) if _inside(root, path) is not None else {}
                artifact = receipt.get("artifact") if isinstance(receipt.get("artifact"), Mapping) else {}
                transform = receipt.get("runtime_to_simulator")
                if (receipt.get("schema_version") != NATIVE_APPEARANCE_SCHEMA
                        or receipt.get("status") != "native_appearance_authored"
                        or artifact.get("digest") != look_sha256
                        or receipt.get("digest") != canonical_digest(receipt, digest_field="digest")
                        or not (isinstance(transform, list) and len(transform) == 4
                                and all(isinstance(row, list) and len(row) == 4
                                        and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                                                and math.isfinite(value) for value in row) for row in transform))):
                    continue
                decoder = receipt.get("decoder")
                if isinstance(decoder, Mapping):
                    wanted, places = decoder.get("output_digest"), [_inside(root, path.parent / "decoded.ply")]
                else:
                    wanted = (receipt.get("binding") or {}).get("source_digest")
                    places = [splat for digest, splat in bases if digest == wanted]
                if not _DIGEST.fullmatch(str(wanted)):
                    continue
                place = next((item for item in places if item is not None), None)
                # A receipt whose source is gone still names the look: it ships as first written, and says so.
                return {"path": str(place) if place is not None else None, "sha256": wanted,
                        "runtime_to_simulator": [[float(value) for value in row] for row in transform],
                        "first_rotations_scrambled": (receipt.get("authoring") or {}).get("orientation_storage")
                        != "gf_real_first"}
    return None


def reauthor_look(look: Mapping[str, Any], target: Path) -> str | None:
    """Write the look again at ``target`` from its source splat (``look_source``), as ``website_native_appearance``
    authors it, with today's writer; None when written, else why not (nothing is written then).

    CPU only, buys nothing. The source must be the bytes the receipt names and at most
    ``REAUTHOR_MAX_SOURCE_BYTES``, checked before it is read.
    """
    from . import particlefield_usd
    source = Path(look["path"]) if look.get("path") else None
    try:
        if source is None or source.is_symlink() or not source.is_file():
            return "website_company_simulation_look_source_missing"
        if source.stat().st_size > REAUTHOR_MAX_SOURCE_BYTES:
            return "website_company_simulation_look_source_over_memory_bound"
        if permission.sha256_file(source) != look["sha256"]:
            return "website_company_simulation_look_source_changed"
        transform = [list(row) for row in zip(*look["runtime_to_simulator"])]  # its rows: the writer's row-major form
        authored = particlefield_usd.write_particlefield_usd(source, target, expected_source_sha256=look["sha256"],
                                                             layer_transform_row_major=transform)
    except Exception as error:  # noqa: BLE001 - a look not written again ships as first written, never raises
        target.unlink(missing_ok=True)
        return f"website_company_simulation_look_not_written:{type(error).__name__}"
    if authored.get("status") != "completed":
        target.unlink(missing_ok=True)
        return "website_company_simulation_look_not_written:" + ";".join(authored.get("blockers") or ["blocked"])
    return None


def look_ply(look: Mapping[str, Any], target: Path) -> str | None:
    """Write the look's source splat, placed as its USD places it, as the standard 3DGS PLY at ``target``; None when
    written, else why not (no PLY then). Written a chunk at a time, and dropped if the source changed meanwhile."""
    from .particlefield_usd import write_placed_viewer_ply
    try:
        write_placed_viewer_ply(look["path"], target, look["runtime_to_simulator"])
        if permission.sha256_file(Path(look["path"])) == look["sha256"]:
            return None
        reason = "website_company_simulation_look_source_changed"
    except Exception as error:  # noqa: BLE001 - a PLY not written is flagged, never raised
        reason = str(error) if isinstance(error, ValueError) else f"website_company_simulation_look_ply:{type(error).__name__}"
    target.unlink(missing_ok=True)
    return reason
