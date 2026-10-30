"""Prepare a fixed site-specific G1 registration; no network, rights grant or compute."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

from .deployment_embodiment_preparation_contract import write_once

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = REPO_ROOT / "config/policy_runtimes"
_spec = importlib.util.spec_from_file_location("_native_g1_registration_contract",
    REPO_ROOT / "deploy/operator-door/operator_door/native_g1_registration_contract.py")
assert _spec and _spec.loader
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)
common = contract.common


def prepare(request, *, registry_path=common.REGISTRY_PATH, state_root=common.STATE_ROOT,
            profile_root=PROFILE_ROOT, release_commit=None, shared_registry_path=contract.SHARED_REGISTRY,
            source_queue=contract.SOURCE_QUEUE):
    request = contract.validate_request(request)
    actual = release_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    contract.require(actual == request["release_commit"], "installed_release_mismatch")
    destination = common.safe_path(state_root / request["installation_id"])
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_once(destination / "request.json", request)
    if (destination / "proposal.json").exists():
        proposal = common.read(destination / "proposal.json")
        contract.require(proposal.get("request_digest") == request["request_digest"], "retained_request_changed")
        return _receipt(request, proposal)
    registry = common.read(registry_path)
    contract.require(registry.get("registry_digest") == request["expected_registry_digest"]
        == common.digest(registry, "registry_digest"), "registry_compare_failed")
    _prepare_current_packet(request, registry, profile_root=profile_root,
        shared_registry_path=shared_registry_path, source_queue=source_queue)
    # Preparation is deterministic and task-scoped. The independently installed
    # root verifier reopens the resulting files rather than trusting this helper.
    proposal = contract.reconstruct(request, registry, profile_root=profile_root,
        shared_registry_path=shared_registry_path, source_queue=source_queue)
    write_once(destination / "proposal.json", proposal)
    return _receipt(request, proposal)


def _prepare_current_packet(request, registry, *, profile_root, shared_registry_path, source_queue):
    """Use only the current site and source-registry-selected exact robot bytes."""
    if all(row["scene_packet"] is not None for row in request["candidates"]):
        return
    site_root, inputs, receipt, _base_request, _base_scene = contract.site_inputs(request, registry, source_queue=source_queue)
    destination = site_root / "native-g1" / receipt["receipt_digest"][7:] / common.digest(request["source"])[7:]
    if (destination / "native_task_arena_packet_receipt.v1.json").exists():
        return  # Independently reverified by reconstruct below, never overwritten.
    try:
        from .beta_policy_screening_g1_prepare import prepare_native_g1_beta_packet
        profile = common.read(profile_root / "native_g1_registration_profile.json")
        shared = contract.sealed(contract.protected(shared_registry_path), "native_g1_team_campaign_registry.v1", "registry_digest")
        contract.require(shared["registry_digest"] == request["shared_registry_digest"], "shared_registry_changed")
        matches = []
        for row in shared["bindings"]:
            try:
                contract.bound(Path(row["publisher_source_dir"]) / "native_g1_publisher_source_stage.v1.json", request["publisher_source"])
                contract.bound(Path(row["runtime_source_receipt_path"]), request["runtime_source"])
                # Historical task/rights are never copied. Only the exact reviewed
                # Arena robot asset is read from the already protected locator.
                packet_root = common.safe_path(Path(row["manipulation_packet_dir"]))
                old = common.read(packet_root / "native_task_arena_scene_plan.v1.json")
                raw = Path(old["robot"]["usd_path"])
                asset = raw if raw.is_absolute() else contract._child(packet_root, str(raw))
                actual = contract.bound(asset, {"sha256": profile["robot_usd_sha256"], "size_bytes": profile["robot_usd_size_bytes"]})
                if actual not in matches:
                    matches.append(actual)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        contract.require(len(matches) == 1, "retained_robot_asset_missing_or_ambiguous")
        prepare_native_g1_beta_packet(base_packet_record=inputs["base_native_packet"], source=request["source"],
            site_id=request["site_id"], output_root=site_root, robot_asset_record=matches[0],
            task_success_contract=inputs["task_success_contract"])
    except (OSError, ValueError, KeyError, TypeError):
        # Root reconstructs per-candidate missing/current-scene reasons. This
        # service helper never invents pose/task semantics to force activation.
        return


def _receipt(request, proposal):
    return {"status": "prepared", "installation_id": request["installation_id"],
        "request_digest": request["request_digest"], "registered_count": len(proposal["registrations"]),
        "results": proposal["results"], "execution_authority_granted": False,
        "candidate_policy_queried": False, "provider_allocations_performed": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-json", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(common.decode(args.request_json)), sort_keys=True))


if __name__ == "__main__":
    main()
