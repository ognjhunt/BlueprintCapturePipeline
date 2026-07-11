#!/usr/bin/env python3
"""Regenerate an attempt-bound G1 kitchen prepared bundle without provider spend."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

from blueprint_pipeline.g1_kitchen_bundle_compatibility import build_source_tree_identity
from blueprint_pipeline.g1_kitchen_run_index import append_run_index_event
from blueprint_pipeline.groot_oscar_closed_loop_image import (
    IMAGE_REF_ENV,
    SEALED_CONFIRMED_ENV,
    build_sealed_launch_plan,
)
from blueprint_pipeline.groot_oscar_digitalocean_closed_loop_job import (
    DEFAULT_CONFIGURED_WAM_CONSISTENCY_COMMAND,
    run_groot_oscar_digitalocean_closed_loop_job,
)
from blueprint_pipeline.groot_oscar_digitalocean_job_inputs import _write_payload_bundle
from blueprint_pipeline.kitchen_attempt_lineage import (
    build_attempt_input_manifest,
    sha256_file,
    write_attempt_input_manifest,
)


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required:{path}")
    return value


def _scenario(value: dict) -> dict:
    rows = value.get("scenarios")
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        return dict(rows[0])
    raise ValueError("exactly one scenario required")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--scenario", required=True, type=Path)
    parser.add_argument("--start-frame", required=True, type=Path)
    parser.add_argument("--kitchen-assets", required=True, type=Path)
    parser.add_argument("--kitchen-inventory", required=True, type=Path)
    parser.add_argument("--worker-image-runtime-evidence", required=True, type=Path)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--provider", default="digitalocean")
    parser.add_argument("--articulation-prim-path")
    parser.add_argument("--steps", type=int, default=48)
    args = parser.parse_args()

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    selection = _load(args.selection.resolve())
    scenario_document = _load(args.scenario.resolve())
    scenario = _scenario(scenario_document)
    selection_sha = sha256_file(args.selection)
    task_id = str(selection.get("selected_task_id") or "")
    if not task_id or scenario.get("task_id") != task_id:
        raise ValueError("scenario task does not match selection")
    stance = dict(scenario.get("accepted_stance_contract") or {})
    scenario_selection_sha = scenario.get("source_selection_sha256") or stance.get(
        "source_selection_sha256"
    )
    if scenario_selection_sha != selection_sha:
        raise ValueError("scenario selection checksum mismatch")
    pose = stance.get("pose_xyz")
    if not isinstance(pose, list) or len(pose) != 3:
        raise ValueError("accepted attempt-bound stance pose required")

    selection_dir = out / "selection_generations" / selection_sha
    selection_dir.mkdir(parents=True)
    selection_path = selection_dir / "random_task_selection.json"
    shutil.copyfile(args.selection, selection_path)
    scenario_path = out / "selected_isaac_scenario.json"
    scenario_path.write_text(
        json.dumps(
            {**scenario, "source_selection_sha256": selection_sha},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    route_path = out / "route.json"
    route_path.write_text(
        json.dumps(
            {
                "schema_version": "kitchen_selected_task_route.v2",
                "task_id": task_id,
                "source_selection_sha256": selection_sha,
                "route_points": [pose, pose],
                "accepted_stance_yaw_rad": stance.get("yaw_rad"),
                "route_semantics": "Attempt begins at the accepted manipulation stance.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    completion = dict(scenario.get("task_success_contract") or {})
    criteria = [dict(row) for row in completion.get("registered_criteria") or []]
    if len(criteria) != 1:
        raise ValueError("one registered completion criterion required")
    if args.articulation_prim_path:
        criteria[0]["articulation_prim_path"] = args.articulation_prim_path
    task_contract_path = out / "task_success_contract.json"
    task_contract_path.write_text(
        json.dumps(
            {
                **completion,
                "schema_version": "g1_kitchen_task_success_contract.v1",
                "task_id": task_id,
                "source_selection_sha256": selection_sha,
                "registered_criteria": criteria,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    plan = build_sealed_launch_plan(
        start_frame="/workspace/initial_policy_frame.png",
        route_file="/workspace/route.json",
        steps=args.steps,
        task_prompt=str(scenario.get("task_instruction") or scenario.get("task") or ""),
        output_dir="/workspace/closed_loop_out",
        wam_consistency_command=DEFAULT_CONFIGURED_WAM_CONSISTENCY_COMMAND,
        env={SEALED_CONFIRMED_ENV: "true", IMAGE_REF_ENV: args.image_ref},
    )
    if plan.get("sealed_active") is not True or plan.get("blockers"):
        raise ValueError(f"sealed launch plan blocked:{plan.get('blockers')}")
    payload = _write_payload_bundle(
        payload_zip=out / "immutable_payload_bundle.zip",
        plan=plan,
        route_payload=_load(route_path),
        seed_path=args.start_frame.resolve(),
        task_prompt=str(scenario.get("task_instruction") or scenario.get("task") or ""),
        seed_provenance={
            "source": "historical_provider_frame_for_preparation_only",
            "task_success_or_media_validity_proven": False,
        },
        task_success_contract_path=task_contract_path,
        kitchen_asset_archive_path=args.kitchen_assets.resolve(),
    )
    source = build_source_tree_identity(Path(__file__).resolve().parents[1])
    attempt_dir = out / "attempts" / args.attempt_id
    attempt_dir.mkdir(parents=True)
    attempt = build_attempt_input_manifest(
        run_id=args.run_id,
        attempt_id=args.attempt_id,
        launch_nonce=f"prepared-{uuid.uuid4().hex}",
        provider=args.provider,
        artifacts={
            "selection": selection_path,
            "scenario": scenario_path,
            "route": route_path,
            "task_success_contract": task_contract_path,
            "kitchen_inventory": args.kitchen_inventory.resolve(),
            "bundle": payload,
            "worker_image_runtime_evidence": args.worker_image_runtime_evidence.resolve(),
        },
        image_digest=args.image_ref.rsplit("@", 1)[-1],
        source_commit=source["source_commit"],
        source_dirty_patch_sha256=source["source_dirty_patch_sha256"],
    )
    attempt_path = write_attempt_input_manifest(
        attempt_dir=attempt_dir, manifest=attempt
    )
    manifest = run_groot_oscar_digitalocean_closed_loop_job(
        start_frame=args.start_frame,
        route_file=route_path,
        task_prompt=str(scenario.get("task_instruction") or scenario.get("task") or ""),
        out_dir=out / "prepared_job",
        steps=args.steps,
        image_ref=args.image_ref,
        wam_consistency_command=DEFAULT_CONFIGURED_WAM_CONSISTENCY_COMMAND,
        seed_provenance={
            "source": "historical_provider_frame_for_preparation_only",
            "task_success_or_media_validity_proven": False,
        },
        task_success_contract_file=task_contract_path,
        attempt_input_manifest_file=attempt_path,
        kitchen_asset_archive_file=args.kitchen_assets,
    )
    if manifest.get("status") != "prepared":
        raise RuntimeError(f"strict bundle preparation blocked:{manifest.get('blockers')}")
    append_run_index_event(
        run_root=out,
        event_type="attempt_allocated",
        run_id=args.run_id,
        attempt_id=args.attempt_id,
        artifact_paths=[attempt_path, payload, Path(manifest["bundle_zip"])],
        detail={"status": "prepared_not_executed", "provider_spend": False},
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
