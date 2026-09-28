"""Run one selected G1 provider worker after reviewed runtime provisioning.

Allocation, current host authority, paired leases, credential resolution and
posted billing belong to the canonical controller. This entry point rechecks
the transported inputs and expiry before any policy can see site observations.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .decision_evidence_contracts import canonical_digest
from .native_g1_team_provider_bundle import (
    MANIFEST, PACKET_RELATIVE_PATH, SCENE_RELATIVE_ROOT, RESULT_SCHEMA, RESULT_FILENAME,
    SCHEMA as BUNDLE_SCHEMA, _json, _sonic_assets, _SONIC_INVENTORY,
    _verify_artifacts, verify_g1_team_scene_packet, verify_g1_team_manifest_binding,
)
from .native_g1_publisher_source_stage import verify_g1_publisher_source
from .native_g1_team_policy_worker import _credential, _execution_packet
from .native_g1_team_worker_supervisor import run_supervised_g1_team_worker


CREDENTIAL_FILE_ENV = "BLUEPRINT_G1_TEAM_CREDENTIAL_FILE"
POLICY_RELAY_FILE_ENV = "BLUEPRINT_G1_TEAM_PRIVATE_RELAY_CONFIG"


def verify_g1_team_sealed_inputs(runtime_root: Path) -> dict[str, Any]:
    root = Path(runtime_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("g1_team_provider_root_invalid")
    bundle_root = root.parent
    manifest = _json(bundle_root / MANIFEST)
    if (
        manifest.get("schema_version") != BUNDLE_SCHEMA
        or manifest.get("status") != "sealed_not_admitted"
        or manifest.get("claim_ceiling") != "development_only"
        or manifest.get("credential_value_included") is not False
        or manifest.get("provider_mutation_performed") is not False
        or manifest.get("manifest_digest") != canonical_digest(manifest, digest_field="manifest_digest")
    ):
        raise ValueError("g1_team_provider_manifest_invalid")

    def open_artifact(relative: str):
        path = bundle_root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(bundle_root.resolve()):
            raise ValueError("g1_team_provider_artifact_path_invalid")
        return path.open("rb")

    _verify_artifacts(
        manifest, open_artifact,
        {row["relative_path"] for row in manifest["artifacts"]} | {MANIFEST},
    )
    packet = _execution_packet(bundle_root / PACKET_RELATIVE_PATH, manifest["implementation_commit"])
    verify_g1_team_manifest_binding(manifest, packet)
    _, scene, _ = verify_g1_team_scene_packet(bundle_root / SCENE_RELATIVE_ROOT, packet)
    if (scene["receipt_digest"] != manifest["scene_packet_receipt_digest"]
            or scene["arena_scene_plan_digest"] != manifest["scene_plan_digest"]):
        raise ValueError("g1_team_provider_scene_binding_invalid")
    publisher = verify_g1_publisher_source(root / "publisher-source")
    if {key: value for key, value in publisher.items() if key not in {"source_root", "receipt_digest"}} != manifest["publisher_source_identity"]:
        raise ValueError("g1_team_provider_publisher_binding_invalid")
    assets = _sonic_assets(root / "inputs/sonic", _SONIC_INVENTORY)
    return {"manifest": manifest, "packet": packet, "sonic_assets": assets}


def verify_g1_team_provider_inputs(runtime_root: Path) -> dict[str, Any]:
    inputs = verify_g1_team_sealed_inputs(runtime_root)
    root = Path(runtime_root)
    bundle_root = root.parent
    manifest = inputs["manifest"]
    source = _json(root / "native_task_runtime_sources/native_task_runtime_source_packet.v1.json")
    provision = _json(bundle_root / "runtime_output/native_task_runtime_source_provisioning.v1.json")
    if (
        provision.get("status") != "completed" or provision.get("runtime_profile") != "unitree_g1"
        or provision.get("receipt_digest") != canonical_digest(provision, digest_field="receipt_digest")
        or source.get("receipt_digest") != manifest["runtime_source_packet"]["receipt_digest"]
        or provision.get("source_packet_sha256") != manifest["runtime_source_packet"]["packet_sha256"]
    ):
        raise ValueError("g1_team_provider_provisioning_binding_invalid")
    return inputs


def run_g1_team_provider_runtime(
    *, runtime_root: Path, output_dir: Path, credential_file_path: Path | None = None,
    policy_relay_config_path: Path | None = None,
) -> dict[str, Any]:
    """Produce a terminal provider receipt without claiming paid closeout."""

    output = Path(output_dir)
    if not output.is_absolute() or output.is_symlink() or (output / RESULT_FILENAME).exists():
        raise ValueError("g1_team_provider_output_invalid")
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA, "status": "blocked", "claim_ceiling": "development_only",
        "stage_reached": "input_verification", "blocker_type": None, "blocker_code": None,
        "execution_packet_digest": None, "supervised_result_digest": None,
        "worker_output_relative_path": "selected-worker/worker", "verified_output": None,
        "candidate_policy_queried": False,
        "provider_teardown_verified": False, "official_billing_reconciled": False,
        "public_redistribution_authorized": False,
    }
    try:
        inputs = verify_g1_team_provider_inputs(runtime_root)
        manifest = inputs["manifest"]
        result["execution_packet_digest"] = manifest["execution_packet_digest"]
        result["stage_reached"] = "policy_runtime_binding"
        if manifest["policy_runtime_required"] and policy_relay_config_path is None:
            # A paid dispatcher must admit a separate policy runtime before it
            # allocates Isaac for these modes. Never fall back to nested Docker.
            result["blocker_code"] = "g1_team_provider_paired_policy_runtime_required"
            raise ValueError("g1_team_provider_paired_policy_runtime_required")
        if policy_relay_config_path is not None:
            from .native_g1_team_relay_runtime_session import read_g1_team_relay_config
            packet = inputs["packet"]
            read_g1_team_relay_config(
                config_path=policy_relay_config_path, execution_packet_digest=packet["packet_digest"],
                profile=packet["request"]["policy_profile"], trusted_setup=packet["trusted_setup"],
                authenticated_owner=packet["request"]["owner"],
            )
        _credential(credential_file_path, required=not manifest["policy_runtime_required"])
        assets = {row["role"]: row for row in inputs["sonic_assets"]}
        root = Path(runtime_root)
        result["stage_reached"] = "supervised_worker"
        # A started child can query before failing. Until retained output is
        # verified, neither zero queries nor successful inference is proven.
        result["candidate_policy_queried"] = None
        supervised = run_supervised_g1_team_worker(
            worker_arguments={
                "execution_packet_path": root.parent / PACKET_RELATIVE_PATH,
                "expected_implementation_commit": manifest["implementation_commit"],
                "scene_packet_root": root.parent / SCENE_RELATIVE_ROOT,
                "runtime_provisioning_receipt_path": root.parent / "runtime_output/native_task_runtime_source_provisioning.v1.json",
                "sonic_provider_source": root / "publisher-source/source/isaaclab_twist2_g1/action_provider/action_provider_sonic.py",
                "sonic_encoder": Path(assets["encoder"]["path"]),
                "sonic_encoder_sha256": assets["encoder"]["sha256"],
                "sonic_decoder": Path(assets["decoder"]["path"]),
                "sonic_decoder_sha256": assets["decoder"]["sha256"],
                "credential_file_path": credential_file_path, "max_steps": 3000,
                **({"policy_relay_config_path": policy_relay_config_path} if policy_relay_config_path else {}),
            },
            output_dir=output / "selected-worker", worker_launcher=Path("/isaac-sim/python.sh"),
        )
        result["supervised_result_digest"] = supervised["result_digest"]
        if supervised["status"] != "completed_development_only":
            raise ValueError("g1_team_provider_worker_incomplete")
        result["verified_output"] = supervised["verified_output"]
        queries = result["verified_output"].get("policy_query_count")
        if type(queries) is not int or queries < 1:
            raise ValueError("g1_team_provider_policy_query_required")
        result["candidate_policy_queried"] = True
        result["status"] = "completed_development_only"
    except Exception as exc:  # noqa: BLE001 - retain typed terminal proof, never raw endpoint details
        result["blocker_type"] = type(exc).__name__
        result["blocker_code"] = result["blocker_code"] or "g1_team_provider_stage_failed"
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    with (output / RESULT_FILENAME).open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    credential = os.environ.get(CREDENTIAL_FILE_ENV)
    relay = os.environ.get(POLICY_RELAY_FILE_ENV)
    result = run_g1_team_provider_runtime(
        runtime_root=args.runtime_root, output_dir=args.output_dir,
        credential_file_path=Path(credential) if credential else None,
        policy_relay_config_path=Path(relay) if relay else None,
    )
    print(json.dumps({"status": result["status"], "result_digest": result["result_digest"]}))
    return 0 if result["status"] == "completed_development_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
