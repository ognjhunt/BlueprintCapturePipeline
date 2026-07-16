#!/usr/bin/env python3
"""Prepare exact local inputs for one bounded RunPod Serverless kitchen campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
import zipfile
from pathlib import Path

from blueprint_pipeline.common import utc_now_iso, write_json
from blueprint_pipeline.g1_kitchen_bundle_compatibility import build_source_tree_identity
from blueprint_pipeline.groot_oscar_runpod_serverless_campaign_io import (
    EVIDENCE_SCHEMA_VERSION,
    validate_campaign_io_evidence,
)
from blueprint_pipeline.groot_oscar_runpod_serverless_campaign_worker import (
    EXPECTED_ATTEMPTS,
    INPUT_SCHEMA_VERSION,
)
from blueprint_pipeline.kitchen_attempt_lineage import ATTEMPT_INPUT_SCHEMA_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file(path: Path, relative_path: str) -> dict:
    return {
        "local_path": str(path.resolve()),
        "relative_path": relative_path,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--strict-bundle-dir", required=True, type=Path)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--model-manifest-digest", required=True)
    parser.add_argument("--network-volume-id", required=True)
    parser.add_argument("--data-center-id", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    repo = args.repo_root.expanduser().resolve()
    source = build_source_tree_identity(repo)
    if source["dirty"]:
        raise SystemExit("campaign_preparation_requires_clean_source_worktree")
    if "@sha256:" not in args.image_ref:
        raise SystemExit("campaign_preparation_requires_digest_pinned_image")
    image_digest = args.image_ref.rsplit("@", 1)[-1]
    bundle_root = args.strict_bundle_dir.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()
    if out.exists():
        raise SystemExit("campaign_preparation_output_must_not_exist")
    out.mkdir(parents=True)
    local_input = out / "input"
    local_input.mkdir()
    paths = {
        "selection": next(bundle_root.glob("selection_generations/*/random_task_selection.json")),
        "scenario": bundle_root / "selected_isaac_scenario.json",
        "route": bundle_root / "route.json",
        "task_success_contract": bundle_root / "task_success_contract.json",
        "kitchen_inventory": bundle_root / "kitchen_asset_inventory_reroll_002.json",
        "bundle": bundle_root / "immutable_payload_bundle.zip",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit("campaign_preparation_inputs_missing:" + ",".join(missing))
    scenario = json.loads(paths["scenario"].read_text(encoding="utf-8"))
    stance = dict(scenario.get("accepted_stance_contract") or {})
    affordance = dict(stance.get("resolved_affordance") or {})
    articulation_prim_path = str(affordance.get("prim_path") or "")
    if not articulation_prim_path.startswith("/"):
        raise SystemExit("campaign_preparation_articulation_prim_path_missing")
    task_contract = json.loads(paths["task_success_contract"].read_text(encoding="utf-8"))
    criteria = [dict(row) for row in task_contract.get("registered_criteria") or []]
    if len(criteria) != 1:
        raise SystemExit("campaign_preparation_requires_one_completion_criterion")
    criteria[0]["articulation_prim_path"] = articulation_prim_path
    task_contract["registered_criteria"] = criteria
    task_contract_path = local_input / "task_success_contract.json"
    write_json(task_contract_path, task_contract)
    payload_copy = local_input / "immutable_payload_bundle.zip"
    with (
        zipfile.ZipFile(paths["bundle"]) as source_archive,
        zipfile.ZipFile(payload_copy, "w") as destination_archive,
    ):
        for member in source_archive.infolist():
            if member.filename == "task_success_contract.json":
                destination_archive.writestr(member, task_contract_path.read_bytes())
            elif member.is_dir():
                destination_archive.writestr(member, b"")
            else:
                with (
                    source_archive.open(member) as source_handle,
                    destination_archive.open(member, "w") as destination_handle,
                ):
                    shutil.copyfileobj(source_handle, destination_handle, 1024 * 1024)
    with zipfile.ZipFile(payload_copy) as archive:
        bundle_inventory = archive.read("kitchen_asset_inventory_checksums.json")
    bundle_inventory_path = local_input / "kitchen_asset_inventory_checksums.json"
    bundle_inventory_path.write_bytes(bundle_inventory)
    artifact_paths = {
        **paths,
        "kitchen_inventory": bundle_inventory_path,
        "task_success_contract": task_contract_path,
        "bundle": payload_copy,
    }
    artifact_refs = {
        name: {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in artifact_paths.items()
    }
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    selected_task_id = str(selection.get("selected_task_id") or "")
    if not selected_task_id:
        raise SystemExit("campaign_preparation_selected_task_missing")
    campaign_prefix = f".blueprint-campaigns/{args.campaign_id}"
    remote_input = f"{campaign_prefix}/input"
    attempts: list[dict] = []
    files: list[dict] = []
    for attempt_id, kind, seed, timeout_seconds in EXPECTED_ATTEMPTS:
        attempt_dir = local_input / attempt_id
        attempt_dir.mkdir()
        manifest = {
            "schema_version": ATTEMPT_INPUT_SCHEMA_VERSION,
            "generated_at": utc_now_iso(),
            "run_id": args.campaign_id,
            "attempt_id": attempt_id,
            "launch_nonce": f"serverless-{uuid.uuid4().hex}",
            "provider": "runpod_serverless",
            "selected_task_id": selected_task_id,
            "source_commit": source["source_commit"],
            "source_dirty_patch_sha256": source["source_dirty_patch_sha256"],
            "image_digest": image_digest,
            "artifacts": artifact_refs,
            "serverless_runtime_qualification_contract": {
                "schema_version": "g1_kitchen_serverless_runtime_qualification.v1",
                "startup_reverified_in_campaign_job": True,
                "strict_three_action_probe_required_before_campaign": True,
                "same_runtime_worker_identity_required": True,
                "runtime_proof_is_not_semantic_task_success": True,
            },
            "compatibility": {
                "attempt_input_schema": ATTEMPT_INPUT_SCHEMA_VERSION,
                "runtime_qualification_schema": (
                    "g1_kitchen_serverless_runtime_qualification.v1"
                ),
            },
        }
        attempt_path = attempt_dir / "attempt_input_manifest.json"
        write_json(
            attempt_path,
            manifest,
        )
        remote = f"{remote_input}/{attempt_id}.json"
        files.append(_file(attempt_path, remote))
        attempts.append(
            {
                "attempt_id": attempt_id,
                "kind": kind,
                "seed": seed,
                "timeout_seconds": timeout_seconds,
                "attempt_manifest": {
                    "relative_path": remote,
                    "sha256": _sha256(attempt_path),
                },
            }
        )
    payload_remote = f"{remote_input}/immutable_payload_bundle.zip"
    files.append(_file(payload_copy, payload_remote))
    campaign_manifest = local_input / "campaign_manifest.json"
    write_json(
        campaign_manifest,
        {
            "schema_version": INPUT_SCHEMA_VERSION,
            "campaign_id": args.campaign_id,
            "source_commit": source["source_commit"],
            "worker_image_ref": args.image_ref,
            "model_manifest_digest": args.model_manifest_digest,
            "payload_bundle": {
                "relative_path": payload_remote,
                "sha256": _sha256(payload_copy),
            },
            "runtime": {
                "dynamic_episode_termination": True,
                "stop_immediately_on_declared_completion": True,
                "fixed_frame_count": None,
                "review_width": 640,
                "review_height": 480,
            },
            "attempts": attempts,
            "raw_secret_values_recorded": False,
        },
    )
    manifest_remote = f"{remote_input}/campaign_manifest.json"
    files.append(_file(campaign_manifest, manifest_remote))
    evidence_path = out / "campaign_io_evidence.json"
    write_json(
        evidence_path,
        {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "source_commit": source["source_commit"],
            "worker_image_ref": args.image_ref,
            "model_manifest_digest": args.model_manifest_digest,
            "network_volume_id": args.network_volume_id,
            "data_center_id": args.data_center_id,
            "campaign_prefix": campaign_prefix,
            "campaign_manifest": {
                "relative_path": manifest_remote,
                "sha256": _sha256(campaign_manifest),
            },
            "output_relative_path": f"{campaign_prefix}/output/artifacts",
            "files": files,
            "raw_secret_values_recorded": False,
        },
    )
    validation = validate_campaign_io_evidence(
        evidence_path,
        source_commit=str(source["source_commit"]),
        image_ref=args.image_ref,
        model_manifest_digest=args.model_manifest_digest,
        volume_id=args.network_volume_id,
        data_center_id=args.data_center_id,
    )
    write_json(out / "campaign_io_validation.json", validation)
    if validation.get("status") != "passed":
        raise SystemExit("campaign_io_validation_failed")
    print(json.dumps({"status": "prepared", "campaign_io_evidence": str(evidence_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
