"""Registered public source choices for authenticated scene intake.

Registration pins publisher bytes and a reviewable task proposal. It is not an
installation receipt, provider grant, or qualification result. The persistent
scene controller, after owner consent, downloads and verifies these sources.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

SCHEMA = "task_evaluation_public_scene_catalog.v1"
SOURCE_SCHEMA = "task_evaluation_public_scene_source_choice.v1"
CATALOG_ENV = "BLUEPRINT_TASK_EVALUATION_PUBLIC_SCENE_CATALOG_FILE"
DEFAULT_CATALOG = "/etc/blueprint/task-evaluation-public-scene-catalog.json"
ROLE_REPOSITORIES = {
    "appearance_3dgs": "InteriorGS", "semantic_metadata": "InteriorGS", "scene_structure": "InteriorGS",
    "collision_usd": "SAGE-3D_Collision_Mesh", "publisher_scene_usdz": "SAGE-3D_InteriorGS_usdz",
}
RIGHTS_ROLES = {"interiorgs_terms", "interiorgs_readme", "sage_readme"}


def _require(condition, code):
    if not condition:
        raise ValueError("public_scene_catalog_" + code)


def content_digest(scene_id, files):
    return canonical_digest({"publisher_scene_id": scene_id, "assets": sorted(
        [{k: r[k] for k in ("role", "sha256", "size_bytes")} for r in files], key=lambda r: r["role"])})


def validate_source_choice(value):
    value = json.loads(json.dumps(value, allow_nan=False))
    _require(value.get("schema_version") == SOURCE_SCHEMA and value.get("source_kind") == "public_scene"
             and value.get("claim_scope") == "development_only", "source_scope_invalid")
    scene = value.get("publisher_scene_id")
    _require(isinstance(scene, str) and re.fullmatch(r"[0-9]{6}", scene), "scene_id_invalid")
    _require(isinstance(value.get("binding_id"), str)
             and re.fullmatch(r"public-[A-Za-z0-9][A-Za-z0-9._-]{0,119}", value["binding_id"]), "binding_id_invalid")
    files = value.get("files")
    _require(isinstance(files, list) and len(files) == 5 and {r.get("role") for r in files} == set(ROLE_REPOSITORIES),
             "source_roles_invalid")
    for row in files:
        _require(set(row) == {"role", "publisher_url", "publisher_revision", "sha256", "size_bytes"}, "file_fields_invalid")
        _require(re.fullmatch(r"sha256:[0-9a-f]{64}", str(row["sha256"]))
                 and type(row["size_bytes"]) is int and 0 < row["size_bytes"] <= 256 * 1024**2
                 and re.fullmatch(r"[0-9a-f]{40}", str(row["publisher_revision"])), "file_identity_invalid")
        uri = urlsplit(row["publisher_url"])
        prefix = f"/datasets/spatialverse/{ROLE_REPOSITORIES[row['role']]}/resolve/{row['publisher_revision']}/"
        _require(uri.scheme == "https" and uri.netloc == "huggingface.co" and not uri.query and not uri.fragment
                 and uri.path.startswith(prefix), "publisher_url_invalid")
        relative = uri.path.removeprefix(prefix)
        expected = {
            "appearance_3dgs": rf"[0-9]{{4}}_{scene}/3dgs_compressed\.ply",
            "semantic_metadata": rf"[0-9]{{4}}_{scene}/labels\.json",
            "scene_structure": rf"[0-9]{{4}}_{scene}/structure\.json",
            "collision_usd": rf"Collision_Mesh/{scene}/{scene}_collision\.usd",
            "publisher_scene_usdz": rf"InteriorGS_usdz/{scene}\.usdz",
        }[row["role"]]
        _require(re.fullmatch(expected, relative), "publisher_scene_path_mismatch")
    _require(value.get("source_content_digest") == content_digest(scene, files), "source_content_digest_invalid")
    rights = value.get("rights")
    _require(isinstance(rights, dict) and rights.get("use_scope") == "noncommercial_internal_research"
             and rights.get("raw_redistribution_allowed") is False
             and rights.get("provider_training_allowed") is False, "rights_scope_invalid")
    evidence = rights.get("evidence")
    _require(isinstance(evidence, list) and len(evidence) == 3
             and {r.get("role") for r in evidence} == RIGHTS_ROLES, "rights_evidence_missing")
    for row in evidence:
        _require(set(row) == {"role", "publisher_url", "sha256", "size_bytes"}
                 and re.fullmatch(r"sha256:[0-9a-f]{64}", str(row["sha256"]))
                 and type(row["size_bytes"]) is int and 0 < row["size_bytes"] <= 2 * 1024**2,
                 "rights_evidence_invalid")
        url = row["publisher_url"]
        if row["role"] == "interiorgs_terms":
            valid = url == "https://kloudsim-usa-cos.kujiale.com/InteriorGS/InteriorGS_Terms_of_Use.pdf"
        else:
            role = "appearance_3dgs" if row["role"] == "interiorgs_readme" else "collision_usd"
            revision = next(f["publisher_revision"] for f in files if f["role"] == role)
            valid = url == f"https://huggingface.co/datasets/spatialverse/{ROLE_REPOSITORIES[role]}/resolve/{revision}/README.md"
        _require(valid, "rights_publisher_url_invalid")
    _require(value.get("rights_reference") == cross_runtime_canonical_digest(rights), "rights_digest_invalid")
    task = value.get("task_proposal")
    _require(isinstance(task, dict) and task.get("strategy") == "pick_and_place"
             and all(isinstance(task.get(k), dict) for k in ("subject", "support", "destination", "success")),
             "task_proposal_invalid")
    _require(value.get("task_proposal_digest") == cross_runtime_canonical_digest(task), "task_proposal_digest_invalid")
    _require(value.get("required_providers") == ["vast", "openai"], "providers_invalid")
    _require(value.get("choice_digest") == cross_runtime_canonical_digest(value, digest_field="choice_digest"),
             "choice_digest_invalid")
    return value


def load_catalog(path=None):
    source = Path(path or os.environ.get(CATALOG_ENV) or DEFAULT_CATALOG)
    if not source.exists():
        result = {"schema_version": SCHEMA, "sources": [], "provider_mutation_performed": False}
    else:
        _require(not any(p.is_symlink() for p in (source, *source.parents)) and source.stat().st_size <= 2 * 1024**2,
                 "file_invalid")
        result = json.loads(source.read_text())
        _require(result.get("schema_version") == SCHEMA and isinstance(result.get("sources"), list)
                 and len(result["sources"]) <= 32 and result.get("provider_mutation_performed") is False,
                 "schema_invalid")
        _require(result.get("catalog_digest") == cross_runtime_canonical_digest(result, digest_field="catalog_digest"),
                 "digest_invalid")
        result["sources"] = [validate_source_choice(row) for row in result["sources"]]
        _require(len({r["binding_id"] for r in result["sources"]}) == len(result["sources"]), "duplicate_binding")
    result["catalog_digest"] = cross_runtime_canonical_digest(result, digest_field="catalog_digest")
    return result


def source_choice(binding_id, *, catalog_path=None):
    rows = [row for row in load_catalog(catalog_path)["sources"] if row["binding_id"] == binding_id]
    _require(len(rows) == 1, "source_not_registered")
    return rows[0]
