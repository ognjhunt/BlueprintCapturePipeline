"""Pinned, sandboxed CAD for designed task primitives; no model/provider calls."""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
import subprocess
from types import SimpleNamespace
from typing import Literal
import zipfile

import numpy as np
from pydantic import BaseModel, ConfigDict

from .decision_evidence_contracts import canonical_digest, canonical_json

CAD_BACKEND = "pinned_cad_primitive_v1"
LEGACY_BACKEND = "legacy_primitive_mesh_v1"
MAX_EVIDENCE_BYTES = 64 * 1024**2


class SyntheticAuthoring(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["website_synthetic_authoring.v1"] = "website_synthetic_authoring.v1"
    backend: Literal["pinned_cad_primitive_v1", "legacy_primitive_mesh_v1"] = CAD_BACKEND
    physics_profile: Literal["bounded_synthetic_rigid_v1"] | None = None
    physics_bounds: dict[str, list[float]] | None = None


def selection(value):
    return SyntheticAuthoring.model_validate(value).model_dump(exclude_none=True)


def resolved_selection(value, design):
    chosen = selection(value)
    if chosen["backend"] != CAD_BACKEND:
        if "physics_bounds" in chosen or "physics_profile" in chosen:
            raise ValueError("website_synthetic_cad_legacy_physics_changed")
        return chosen
    mass = float(design["mass_kg"])
    chosen.setdefault("physics_profile", "bounded_synthetic_rigid_v1")
    chosen.setdefault("physics_bounds", {"mass_kg": [mass*.9, mass*1.1], "static_friction": [.6, .8],
                                         "dynamic_friction": [.4, .5], "restitution": [.1, .4]})
    bounds = chosen["physics_bounds"]
    if (set(bounds) != {"mass_kg", "static_friction", "dynamic_friction", "restitution"}
            or any(len(row) != 2 or not all(math.isfinite(v) for v in row) or not 0 <= row[0] <= row[1]
                   for row in bounds.values())
            or bounds["mass_kg"][0] <= 0 or not bounds["mass_kg"][0] <= mass <= bounds["mass_kg"][1]
            or bounds["static_friction"][0] < bounds["dynamic_friction"][1]
            or bounds["restitution"][1] > 1):
        raise ValueError("website_synthetic_cad_physics_invalid")
    return chosen


def physical_review(design, chosen, parameter_digest):
    """Independent existing validator, using the exact proposed production bounds."""
    from .task_object_physical_property_review import (
        PhysicalPropertyReviewInput, PhysicalPropertyReviewProposal, review_physical_properties,
    )
    bounds = resolved_selection(chosen, design)["physics_bounds"]
    def value(nominal, low, high):
        return {"value": nominal, "basis": "estimated", "interval": {"lower": low, "upper": high},
            "rationale": "Synthetic simulation design choice, not a captured measurement.",
            "uncertainty": "Bounded proposed physics; native sensitivity testing remains necessary.",
            "evidence_ids": ["synthetic-design"]}
    dims = {axis: value(float(d), float(d)*.999, float(d)*1.001)
            for axis, d in zip(("x_m", "y_m", "z_m"), design["dimensions_m"], strict=True)}
    properties = {name: value((low+high)/2, low, high) for name, (low, high) in bounds.items()}
    optical = {"name": "SyntheticOpaqueRigid", "transmission": 0., "opacity": 1.}
    data = {"object_id": "synthetic-task-primitive", "object_description": "Designed " + design["shape"],
        "material_description": "Assigned constant-density rigid simulation material; no physical material claim.",
        "appearance": "opaque", "dimensions": dims, "proposed": properties,
        "measured": {k: None for k in properties}, "optical_material": optical,
        "admitted_restitution": {"lower": bounds["restitution"][0], "upper": bounds["restitution"][1]},
        "evidence": [{"evidence_id": "synthetic-design", "uri": "retained:parameters.json",
            "sha256": parameter_digest[7:], "kind": "primary_reference",
            "excerpt": "Primary synthetic parameter record under delegated design authority; no owner numeric confirmation or physical measurement claimed."}]}
    fill = math.pi/6 if design["shape"] == "sphere" else math.pi/4 if design["shape"] == "cylinder" else 1.
    volume = float(np.prod(design["dimensions_m"]))*fill
    proposal = {"object_id": data["object_id"], "dimensions": dims, "properties": properties,
        "optical_material": optical, "review_rationale": "Validate exact proposed production bounds independently.",
        "mass_model": {"method": "density_fill",
            "density_kg_m3": {"lower": bounds["mass_kg"][0]/volume, "upper": bounds["mass_kg"][1]/volume},
            "envelope_fill_fraction": {"lower": fill, "upper": fill}, "sheet_count": None,
            "sheet_area_m2": None, "grammage_g_m2": None, "cover_mass_kg": None,
            "rationale": "Assigned synthetic density times exact primitive volume.",
            "uncertainty": "Synthetic model range, not a measured physical material.", "evidence_ids": ["synthetic-design"]}}
    result = review_physical_properties(PhysicalPropertyReviewInput.model_validate(data),
                                        PhysicalPropertyReviewProposal.model_validate(proposal))
    return data, proposal, result.model_dump(mode="json")


def primitive_program(design):
    from .clean_plate_removal_analysis_gemini import validate_synthetic_subject
    if validate_synthetic_subject(design):
        raise ValueError("website_synthetic_cad_design_invalid")
    x, y, z = [float(v) * 1000 for v in design["dimensions_m"]]
    shape = design["shape"]
    expression = (f"Sphere(radius={x / 2!r})" if shape == "sphere" else
                  f"Cylinder(radius={x / 2!r}, height={z!r})" if shape == "cylinder" else
                  f"Box(length={x!r}, width={y!r}, height={z!r})")
    return ("from build123d import Sphere, Cylinder, Box\n\n"
            "def gen_step():\n"
            f"    candidate = {expression}\n"
            f"    candidate.label = 'Synthetic_task_{shape}'\n"
            "    return candidate\n")


def _record(path):
    data = path.read_bytes()
    return {"digest": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def verify_evidence(path, *, design, expected_manifest_digest=None, authoring=None):
    try:
        return _verify_evidence(path, design=design, expected_manifest_digest=expected_manifest_digest, authoring=authoring)
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError) as exc:
        raise ValueError("website_synthetic_cad_evidence_invalid") from exc


def _verify_evidence(path, *, design, expected_manifest_digest=None, authoring=None):
    """Read bounded ZIP members without extraction; verify every retained export."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ValueError("website_synthetic_cad_evidence_invalid")
    with zipfile.ZipFile(path) as archive:
        from .production_cad_skill_sources import SOURCE_SPECS
        rows = archive.infolist()
        names = [r.filename for r in rows]
        if (len(rows) > 32 or len(names) != len(set(names))
                or sum(r.file_size for r in rows) > MAX_EVIDENCE_BYTES
                or any(r.flag_bits & 1 for r in rows)
                or any(PurePosixPath(n).name != n or n in {"", ".", ".."} or "\\" in n for n in names)):
            raise ValueError("website_synthetic_cad_evidence_invalid")
        manifest = json.loads(archive.read("manifest.json"))
        if (not isinstance(manifest, dict) or not isinstance(manifest.get("step_readback"), dict)
                or not isinstance(manifest.get("stl_readback"), dict) or not isinstance(manifest.get("files"), dict)):
            raise ValueError("website_synthetic_cad_evidence_invalid")
        if (manifest.get("schema_version") != "website_synthetic_cad_candidate.v1"
                or manifest.get("claim_ceiling") != "development_only"
                or manifest.get("provider_called") is not False
                or manifest.get("native_qualified") is not False
                or manifest.get("source_pins") != {s["id"]: s["commit"] for s in SOURCE_SPECS}
                or manifest.get("design") != dict(design)
                or manifest.get("manifest_digest") != canonical_digest(manifest, digest_field="manifest_digest")
                or expected_manifest_digest is not None and manifest["manifest_digest"] != expected_manifest_digest
                or manifest.get("step_readback", {}).get("passed") is not True
                or manifest.get("stl_readback", {}).get("passed") is not True):
            raise ValueError("website_synthetic_cad_evidence_invalid")
        files = manifest.get("files") or {}
        required = {"asset.py", "candidate.step", "candidate.stl", "parameters.json", "source-receipt.json",
                    "step-readback.json", "candidate-receipt.json", "stl-readback.json", "snapshot-result.json",
                    "physical-review-input.json", "physical-review-proposal.json", "physical-review-result.json"}
        if not required <= set(files) or set(names) != {*files, "manifest.json"}:
            raise ValueError("website_synthetic_cad_evidence_invalid")
        for name, record in files.items():
            data = archive.read(name)
            if record != {"digest": "sha256:" + hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}:
                raise ValueError("website_synthetic_cad_evidence_changed")
        if (archive.read("asset.py").decode() != primitive_program(design)
                or json.loads(archive.read("parameters.json"))["design"] != dict(design)
                or json.loads(archive.read("step-readback.json")) != manifest["step_readback"]
                or json.loads(archive.read("stl-readback.json")) != manifest["stl_readback"]):
            raise ValueError("website_synthetic_cad_evidence_invalid")
        parameters = json.loads(archive.read("parameters.json"))
        chosen = resolved_selection(parameters["synthetic_authoring"], design)
        if authoring is not None and chosen != resolved_selection(authoring, design):
            raise ValueError("website_synthetic_cad_physics_binding_invalid")
        reviewed = physical_review(design, chosen, files["parameters.json"]["digest"])
        if (reviewed[2]["accepted"] is None or reviewed[2]["blockers"]
                or any(json.loads(archive.read(name)) != expected for name, expected in zip(
                    ("physical-review-input.json", "physical-review-proposal.json", "physical-review-result.json"), reviewed, strict=True))):
            raise ValueError("website_synthetic_cad_physics_review_invalid")
        sources = json.loads(archive.read("source-receipt.json"))
        compiled = json.loads(archive.read("candidate-receipt.json"))
        if (not isinstance(sources, dict) or not isinstance(compiled, dict)
                or not isinstance(sources.get("cad"), dict) or not isinstance(sources.get("mac"), dict)
                or sources.get("cad", {}).get("commit") != SOURCE_SPECS[0]["commit"]
                or sources.get("mac", {}).get("commit") != SOURCE_SPECS[1]["commit"]
                or compiled.get("passed") is not True or compiled.get("source_unchanged_after") is not True
                or compiled.get("readback") != manifest["step_readback"]
                or manifest["step_readback"].get("build123d") != "0.11.1"):
            raise ValueError("website_synthetic_cad_evidence_invalid")
    return manifest


def author_candidate(design, *, path, authoring):
    """Use the installed host CAD sources and existing OS isolation boundary."""
    from .asset_authoring_sandbox import AssetSandboxError, SandboxedAssetRunner
    from .astra_cad_skill_runtime import AstraCADRuntimeBlocked
    from .production_cad_skill_sources import DEFAULT_ROOT, SOURCE_SPECS
    from .task_object_agent_cad import execute_cad_program
    from .task_object_authoring_contracts import AssetAuthoringError
    import trimesh

    program = primitive_program(design)
    chosen = resolved_selection(authoring, design)
    archive_path = path.parent / "synthetic_cad_evidence.zip"
    if archive_path.exists():
        manifest = verify_evidence(archive_path, design=design, authoring=chosen)
        with zipfile.ZipFile(archive_path) as archive:
            mesh = trimesh.load(io.BytesIO(archive.read("candidate.stl")), file_type="stl", force="mesh")
        mesh.apply_scale(.001)
        return mesh, _metadata(archive_path, manifest)
    if importlib.metadata.version("build123d") != "0.11.1":
        raise ValueError("website_synthetic_cad_kernel_not_pinned")
    sources = [Path(DEFAULT_ROOT) / f"{s['id']}-{s['commit'][:8]}" for s in SOURCE_SPECS]
    root = path.parent / "cad_authoring"
    root.mkdir(exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
    parameter_path = attempt / "parameters.json"
    parameter_path.write_text(canonical_json({"design": dict(design), "synthetic_authoring": chosen,
        "length_unit": "millimeter", "simulation_length_unit": "meter", "stl_to_simulation_scale": .001,
        "dimension_authority": "synthetic_design_choice", "physical_measurement": False,
        "owner_numeric_confirmation_claimed": False}) + "\n")
    reviewed = physical_review(design, chosen, _record(parameter_path)["digest"])
    for name, value in zip(("physical-review-input.json", "physical-review-proposal.json", "physical-review-result.json"),
                           reviewed, strict=True):
        (attempt / name).write_text(canonical_json(value) + "\n")
    if reviewed[2]["accepted"] is None or reviewed[2]["blockers"]:
        raise ValueError("website_synthetic_cad_physics_review_rejected")
    python_roots = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    python_roots += [Path(p).resolve() for p in sys.path if p and Path(p).is_dir()
                     and "site-packages" in p]
    runner = SandboxedAssetRunner(read_roots=[*python_roots, *sources, Path(__file__).parent.parent],
                                 executable_roots=python_roots, write_root=attempt)
    try:
        receipt = execute_cad_program(program=program, output_root=attempt / "candidate",
            request=SimpleNamespace(dimensions_m=tuple(design["dimensions_m"]), maximum_export_error_m=.0001),
            cad_root=sources[0], mac_root=sources[1], sandbox=runner)
    except AssetSandboxError as exc:
        raise ValueError("website_synthetic_cad_sandbox_unavailable:" + str(exc)) from exc
    except (AstraCADRuntimeBlocked, AssetAuthoringError, subprocess.CalledProcessError) as exc:
        raise ValueError("website_synthetic_cad_authoring_rejected:" + str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("website_synthetic_cad_authoring_timeout") from exc
    output = attempt / "candidate"
    mesh = trimesh.load(receipt["stl_path"], force="mesh")
    mesh.apply_scale(.001)
    dimensions = np.asarray(design["dimensions_m"], dtype=float)
    expected_volume = (4 / 3 * math.pi * (dimensions[0] / 2)**3 if design["shape"] == "sphere" else
                       math.pi * (dimensions[0] / 2)**2 * dimensions[2] if design["shape"] == "cylinder" else
                       float(np.prod(dimensions)))
    readback = {"watertight": bool(mesh.is_watertight), "convex": bool(mesh.is_convex),
        "winding_consistent": bool(mesh.is_winding_consistent), "body_count": int(mesh.body_count),
        "dimensions_m": mesh.extents.tolist(), "volume_m3": float(mesh.volume),
        "expected_volume_m3": expected_volume, "trimesh": importlib.metadata.version("trimesh")}
    readback["passed"] = bool(readback["watertight"] and readback["convex"] and readback["winding_consistent"]
        and readback["body_count"] == 1 and np.isfinite(mesh.vertices).all()
        and np.allclose(mesh.extents, dimensions, rtol=0, atol=.0001)
        and np.allclose(mesh.bounds.mean(axis=0), 0, rtol=0, atol=.0001)
        and math.isclose(mesh.volume, expected_volume, rel_tol=.01))
    (output / "stl-readback.json").write_text(canonical_json(readback) + "\n")
    if not readback["passed"]:
        raise ValueError("website_synthetic_cad_mesh_rejected")
    (output / "parameters.json").write_bytes(parameter_path.read_bytes())
    for name, value in zip(("physical-review-input.json", "physical-review-proposal.json", "physical-review-result.json"),
                           reviewed, strict=True):
        (output / name).write_bytes((attempt / name).read_bytes())
    # Diagnostic snapshot failure remains explicit; it cannot grant geometry or native authority.
    try:
        rendered = runner([sys.executable, str(sources[0] / "skills/cad/scripts/snapshot"),
            "--input", receipt["step_path"], "--output", str(output / "snapshot.png")],
            cwd=str(output), timeout=60, capture_output=True, text=True, check=False)
    except subprocess.TimeoutExpired:
        rendered = SimpleNamespace(returncode=124, stdout="", stderr="CAD snapshot exceeded 60 seconds")
    snapshots = list(output.glob("snapshot*.png"))
    (output / "snapshot-result.json").write_text(canonical_json({"returncode": rendered.returncode,
        "status": "rendered_pending_review" if rendered.returncode == 0 and snapshots else "unavailable",
        "stdout": (rendered.stdout or "")[-4000:], "stderr": (rendered.stderr or "")[-4000:]}) + "\n")
    names = ["asset.py", "candidate.step", "candidate.stl", "parameters.json", "source-receipt.json",
             "step-readback.json", "candidate-receipt.json", "stl-readback.json", "snapshot-result.json",
             "physical-review-input.json", "physical-review-proposal.json", "physical-review-result.json"]
    names += [p.name for p in snapshots[:1]]
    manifest = {"schema_version": "website_synthetic_cad_candidate.v1", "design": dict(design),
        "claim_ceiling": "development_only", "provider_called": False, "native_qualified": False,
        "source_pins": {s["id"]: s["commit"] for s in SOURCE_SPECS}, "step_readback": receipt["readback"],
        "stl_readback": readback, "files": {n: _record(output / n) for n in names}}
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    temporary_archive = output / "evidence.zip"
    with zipfile.ZipFile(temporary_archive, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", canonical_json(manifest) + "\n")
        for name in names:
            archive.write(output / name, name)
    verify_evidence(temporary_archive, design=design, expected_manifest_digest=manifest["manifest_digest"], authoring=chosen)
    os.link(temporary_archive, archive_path)  # Publish only a complete verified archive; never overwrite one.
    return mesh, _metadata(archive_path, manifest)


def _metadata(path, manifest):
    return {"backend": CAD_BACKEND, "manifest_digest": manifest["manifest_digest"],
            "evidence": {"path": path.name, **_record(path)},
            "step_readback": manifest["step_readback"], "stl_readback": manifest["stl_readback"]}
