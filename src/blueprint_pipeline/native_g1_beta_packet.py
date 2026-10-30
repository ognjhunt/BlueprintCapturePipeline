"""Derive frozen-reset beta cells without changing retained G1 scene geometry.

The base packet remains immutable. Only policy-neutral scenario metadata and
its enclosing digests change; this is a simulation seed derivation, not a new
capture, task definition, controller calibration or human review.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_authority import derive_cell_authority, validate_frozen_cell
from .native_task_arena_bundle import verify_native_task_arena_packet

REQUEST = "native_task_arena_packet_request.v1.json"
CONTRACT = "native_task_runtime_contract.v1.json"
PLAN = "native_task_arena_scene_plan.v1.json"
RECEIPT = "native_task_arena_packet_receipt.v1.json"
DERIVATION = "native_g1_beta_cell_packet.v1.json"


def _sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = json.loads(json.dumps(value, allow_nan=False))
    if result.get(field) != canonical_digest(result, digest_field=field):
        raise ValueError("g1_beta_base_packet_digest_invalid")
    return result


def scenario_for_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """One native evaluation context with no camera/placement perturbations."""
    row = validate_frozen_cell(cell)
    context = {"schema_version": "adp009d_scenario_instance.v1",
        "cell_id": row["cell_id"], "seed": row["seed"],
        "policy_neutral": True, "caller_asserted_success": False,
        "learned_policy_outcomes_consulted": False,
        "factor_records": [], "resolved_parameters": {},
        "beta_cell_spec_digest": row["cell_spec_digest"],
        "scenario_scope": "native_frozen_reset_seeds"}
    context["instance_digest"] = canonical_digest(context, digest_field="instance_digest")
    return {"context_kind": "evaluation_cell", "cell_id": row["cell_id"],
            "seed": row["seed"], "instance_digest": context["instance_digest"],
            "context_document": context}


def derive_documents(*, base_request: Mapping[str, Any], base_contract: Mapping[str, Any],
                     base_scene: Mapping[str, Any], cell: Mapping[str, Any]) -> dict[str, Any]:
    request = _sealed(base_request, "request_digest")
    contract = _sealed(base_contract, "contract_digest")
    scene = _sealed(base_scene, "plan_digest")
    if (scene.get("runtime_contract_digest") != contract["contract_digest"]
            or scene.get("robot", {}).get("robot_id") != "unitree_g1"
            or scene.get("task_kind") != "rigid_pick_place"
            or any(request.get(key) != scene.get(key) or contract.get(key) != scene.get(key)
                   for key in ("scene_id", "task_id"))
            or contract.get("task_spec") != scene.get("task_spec")
            or not isinstance(contract.get("scenario"), dict)
            or not isinstance(scene.get("scenario"), dict)):
        raise ValueError("g1_beta_base_packet_binding_invalid")
    # These are repetitions of the authored base scene. Existing parameter
    # applications remain frozen; a new cell does not claim to reapply them.
    scenario = scenario_for_cell(cell)
    request["scenario"] = scenario
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    update = {key: scenario[key] for key in ("context_kind", "cell_id", "instance_digest", "seed")}
    contract["scenario"].update(update)
    contract["contract_digest"] = canonical_digest(contract, digest_field="contract_digest")
    scene["scenario"].update(update)
    scene["runtime_contract_digest"] = contract["contract_digest"]
    scene["plan_digest"] = canonical_digest(scene, digest_field="plan_digest")
    return {"request": request, "contract": contract, "scene": scene}


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _file_row(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"relative_path": path.name, "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + digest.hexdigest()}


def expected_cell_packet_receipt(original, derived, cell):
    if original.get("g1_beta_cell_derivation") is not None:
        raise ValueError("g1_beta_nested_cell_packet_forbidden")
    receipt = deepcopy(original)
    receipt.update(request_digest=derived["request"]["request_digest"],
                   runtime_contract_digest=derived["contract"]["contract_digest"],
                   arena_scene_plan_digest=derived["scene"]["plan_digest"],
                   scenario_instance_digest=derived["scene"]["scenario"]["instance_digest"])
    rewritten = {}
    for key, name in (("request", REQUEST), ("contract", CONTRACT), ("scene", PLAN)):
        data = (json.dumps(derived[key], indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        rewritten[name] = {"relative_path": name, "size_bytes": len(data),
                           "sha256": "sha256:" + hashlib.sha256(data).hexdigest()}
    receipt["artifacts"] = [{**row, **rewritten.get(row["relative_path"], {})}
                            for row in receipt["artifacts"]]
    receipt["g1_beta_cell_derivation"] = {
        "base_packet_receipt_digest": original["receipt_digest"],
        "base_scene_plan_digest": original["arena_scene_plan_digest"],
        "cell": dict(cell), "geometry_changed": False, "new_human_review_claimed": False}
    receipt["receipt_digest"] = canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


def materialize_cell_packet(*, base_packet: Path, output_dir: Path,
                            cell: Mapping[str, Any], base_authority: Mapping[str, Any]) -> dict[str, Any]:
    root, original, rows = verify_native_task_arena_packet(base_packet)
    documents = {name: json.loads((root / filename).read_text()) for name, filename in
                 (("request", REQUEST), ("contract", CONTRACT), ("scene", PLAN))}
    if (original.get("arena_scene_plan_digest") != documents["scene"].get("plan_digest")
            or original.get("runtime_contract_digest") != documents["contract"].get("contract_digest")
            or original.get("request_digest") != documents["request"].get("request_digest")
            or base_authority.get("scene_plan_digest") != documents["scene"]["plan_digest"]):
        raise ValueError("g1_beta_base_packet_authority_mismatch")
    derived = derive_documents(base_request=documents["request"], base_contract=documents["contract"],
                               base_scene=documents["scene"], cell=cell)
    destination = Path(output_dir)
    if (not destination.is_absolute() or destination.exists() or destination.is_symlink()
            or destination.resolve().is_relative_to(root) or root.is_relative_to(destination.resolve())):
        raise ValueError("g1_beta_cell_packet_destination_invalid")
    destination.mkdir(parents=True)
    # Copy verified source files; never overwrite an asset in the retained base.
    for row in rows:
        target = destination / row["relative_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / row["relative_path"], target)
    for name, filename in (("request", REQUEST), ("contract", CONTRACT), ("scene", PLAN)):
        _write(destination / filename, derived[name])
    receipt = expected_cell_packet_receipt(original, derived, cell)
    _write(destination / RECEIPT, receipt)
    authority = derive_cell_authority(base_authority=base_authority, cell=cell,
        scene_plan_digest=derived["scene"]["plan_digest"], packet_receipt_digest=receipt["receipt_digest"])
    derivation = {"schema_version": "native_g1_beta_cell_packet.v1", "cell": dict(cell),
        "base_packet_receipt_digest": original["receipt_digest"],
        "packet_receipt_digest": receipt["receipt_digest"],
        "scene_plan_digest": derived["scene"]["plan_digest"], "rights_authority": authority}
    derivation["derivation_digest"] = canonical_digest(derivation, digest_field="derivation_digest")
    _write(destination / DERIVATION, derivation)
    verify_cell_packet(base_packet=root, cell_packet=destination, cell=cell, base_authority=base_authority)
    return derivation


def verify_cell_packet(*, base_packet: Path, cell_packet: Path, cell: Mapping[str, Any],
                       base_authority: Mapping[str, Any]) -> dict[str, Any]:
    base, original, base_rows = verify_native_task_arena_packet(base_packet)
    root, receipt, rows = verify_native_task_arena_packet(cell_packet)
    documents = {name: json.loads((base / filename).read_text()) for name, filename in
                 (("request", REQUEST), ("contract", CONTRACT), ("scene", PLAN))}
    expected = derive_documents(base_request=documents["request"], base_contract=documents["contract"],
                                base_scene=documents["scene"], cell=cell)
    for name, filename in (("request", REQUEST), ("contract", CONTRACT), ("scene", PLAN)):
        if json.loads((root / filename).read_text()) != expected[name]:
            raise ValueError("g1_beta_cell_packet_document_changed")
    if receipt != expected_cell_packet_receipt(original, expected, cell):
        raise ValueError("g1_beta_cell_packet_receipt_changed")
    mutable = {REQUEST, CONTRACT, PLAN, RECEIPT, DERIVATION}
    by_path = {row["relative_path"]: row for row in rows}
    if set(by_path) != {row["relative_path"] for row in base_rows} | {DERIVATION}:
        raise ValueError("g1_beta_cell_packet_closure_changed")
    for row in base_rows:
        if row["relative_path"] not in mutable and by_path[row["relative_path"]] != row:
            raise ValueError("g1_beta_cell_packet_asset_changed")
    derivation = _sealed(json.loads((root / DERIVATION).read_text()), "derivation_digest")
    expected_authority = derive_cell_authority(base_authority=base_authority, cell=cell,
        scene_plan_digest=expected["scene"]["plan_digest"], packet_receipt_digest=receipt["receipt_digest"])
    if (derivation.get("schema_version") != "native_g1_beta_cell_packet.v1"
            or derivation.get("cell") != cell
            or derivation.get("base_packet_receipt_digest") != original["receipt_digest"]
            or derivation.get("packet_receipt_digest") != receipt["receipt_digest"]
            or derivation.get("scene_plan_digest") != expected["scene"]["plan_digest"]
            or derivation.get("rights_authority") != expected_authority):
        raise ValueError("g1_beta_cell_packet_derivation_changed")
    return derivation
