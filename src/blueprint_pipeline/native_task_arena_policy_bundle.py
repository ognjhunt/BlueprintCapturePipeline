"""Freeze one task/cell-qualified learned-policy Arena provider bundle."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adp009d_policy_episode import maximum_policy_queries_for_task_spec
from .decision_evidence_contracts import canonical_digest
from .native_task_arena_bundle import (
    POLICY_RUNTIME_ROOT_MODULE_NAMES,
    build_native_task_arena_bundle,
)
from .native_task_arena_controls_bundle import controls_runtime_sources
from .native_task_arena_execution_contract import POLICY_EXTRA_RUNTIME_MODULE_NAMES
from .native_task_runtime_contract import FROZEN_CANDIDATES
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


RESULT_SCHEMA_VERSION = "native_task_arena_policy_result.v1"
RESULT_FILENAME = "native_task_arena_policy_result.v1.json"
EXECUTION_SPEC_SCHEMA_VERSION = "native_task_arena_policy_execution_spec.v1"
PROBE_KIND = "native-task-arena-policy"
GROOT_RUNTIME_IDENTITY_FILENAME = (
    "adp009d_groot_worker_identity.groot_n17_droid.json"
)
GROOT_RUNTIME_IDENTITY_DECLARATION = {
    "status": "runtime_measurement_required",
    "relative_path": GROOT_RUNTIME_IDENTITY_FILENAME,
}
QUALIFIED_EXECUTION_AUTHORITY = "qualified_controls_evaluation"
OPENPI_CHECKPOINT_INVENTORY_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/experiments/policy_ranking_thesis_20260726/"
    "openpi_polaris_checkpoint_inventory.json"
)


def _read(path: str | Path, *, error: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(error) from exc
    if not isinstance(value, dict):
        raise ValueError(error)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _verified_openpi_checkpoint_inventory(
    policy_spec: Mapping[str, Any],
) -> Path:
    from .openpi_droid_policy_runtime import canonical_sha256

    path = OPENPI_CHECKPOINT_INVENTORY_PATH
    inventory = _read(
        path, error="native_task_policy_openpi_checkpoint_inventory_invalid"
    )
    digest_payload = dict(inventory)
    digest_payload.pop("inventory_sha256", None)
    matches = [
        row
        for row in inventory.get("entries", [])
        if isinstance(row, Mapping)
        and row.get("policy_id") == policy_spec.get("policy_id")
    ]
    if (
        inventory.get("schema_version") != "openpi_checkpoint_inventory.v1"
        or inventory.get("status") != "frozen"
        or inventory.get("blockers")
        or inventory.get("inventory_sha256") != canonical_sha256(digest_payload)
        or inventory.get("inventory_sha256")
        != policy_spec.get("checkpoint_inventory_sha256")
        or len(matches) != 1
    ):
        raise ValueError("native_task_policy_openpi_checkpoint_inventory_invalid")
    entry = matches[0]
    expected = {
        "checkpoint_uri": policy_spec.get("checkpoint_uri"),
        "object_count": policy_spec.get("checkpoint_object_count"),
        "size_bytes": policy_spec.get("checkpoint_size_bytes"),
        "legacy_object_manifest_sha256": policy_spec.get(
            "checkpoint_object_manifest_sha256"
        ),
        "generation_manifest_sha256": policy_spec.get(
            "checkpoint_generation_manifest_sha256"
        ),
    }
    if any(entry.get(field) != value for field, value in expected.items()):
        raise ValueError("native_task_policy_openpi_checkpoint_inventory_invalid")
    return path


def validate_native_task_policy_execution_spec(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(value))
    errors: list[str] = []
    candidate = str(payload.get("candidate_id") or "")
    if payload.get("execution_authority") not in (
        None,
        QUALIFIED_EXECUTION_AUTHORITY,
    ):
        errors.append("native_task_policy_execution_authority_invalid")
    endpoint = payload.get("policy_endpoint")
    if payload.get("schema_version") != EXECUTION_SPEC_SCHEMA_VERSION:
        errors.append("native_task_policy_execution_spec_schema_invalid")
    if candidate not in FROZEN_CANDIDATES:
        errors.append("native_task_policy_candidate_invalid")
    if not isinstance(endpoint, Mapping):
        errors.append("native_task_policy_endpoint_invalid")
    else:
        host = str(endpoint.get("host") or "")
        port = endpoint.get("port")
        credential_env = str(endpoint.get("credential_env") or "")
        if (
            not host
            or isinstance(port, bool)
            or not isinstance(port, int)
            or not 1 <= port <= 65535
            or not credential_env.startswith("BLUEPRINT_")
        ):
            errors.append("native_task_policy_endpoint_invalid")
    for field in ("task_id", "cell_id", "prompt"):
        if not str(payload.get(field) or "").strip():
            errors.append(f"native_task_policy_{field}_missing")
    for field in (
        "scene_plan_digest",
        "construction_result_digest",
        "control_result_digest",
        "control_pair_digest",
    ):
        text = str(payload.get(field) or "")
        if len(text) != 71 or not text.startswith("sha256:"):
            errors.append(f"native_task_policy_{field}_invalid")
    for field in ("policy_spec", "policy_identity_receipt"):
        if not isinstance(payload.get(field), Mapping):
            errors.append(f"native_task_policy_{field}_invalid")
    if isinstance(payload.get("policy_spec"), Mapping):
        try:
            if candidate == "pi05_droid":
                from .openpi_droid_policy_runtime import (
                    OpenPIDroidPolicySpec,
                    validate_arena_candidate_policy_binding,
                )

                policy_spec = OpenPIDroidPolicySpec(**payload["policy_spec"])
                validate_arena_candidate_policy_binding(
                    candidate_id=candidate, spec=policy_spec
                )
            elif candidate == "groot_n17_droid":
                from .groot_n17_droid_policy_runtime import (
                    GrootN17DroidPolicySpec,
                )

                policy_spec = GrootN17DroidPolicySpec(**payload["policy_spec"])
                policy_spec.validate()
                # NVIDIA's PolicyClient pings the server but does not attest
                # which checkpoint bytes the server loaded.  Provisioning
                # measures those bytes only after this immutable spec has been
                # built, so a pre-filled "verified" receipt here would be a
                # declaration masquerading as an observation.  Require the
                # spec to declare the runtime measurement and let the episode
                # worker consume the resulting receipt.
                if (
                    payload.get("policy_identity_receipt")
                    != GROOT_RUNTIME_IDENTITY_DECLARATION
                ):
                    raise ValueError("groot_runtime_identity_declaration_invalid")
        except (TypeError, ValueError):
            errors.append("native_task_policy_spec_or_identity_invalid")
        if payload["policy_spec"].get("open_loop_horizon") != payload.get(
            "open_loop_horizon"
        ):
            errors.append("native_task_policy_open_loop_horizon_mismatch")
    for field in ("max_policy_queries", "open_loop_horizon"):
        raw = payload.get(field)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            errors.append(f"native_task_policy_{field}_invalid")
    if payload.get("overview_camera_policy_input") is not False:
        errors.append("native_task_policy_overview_input_invalid")
    if payload.get("policy_may_grade_itself") is not False:
        errors.append("native_task_policy_self_grading_invalid")
    if payload.get("execution_spec_digest") != canonical_digest(
        payload, digest_field="execution_spec_digest"
    ):
        errors.append("native_task_policy_execution_spec_digest_invalid")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    return payload


def materialize_native_task_policy_execution_spec(
    *, request: Mapping[str, Any], output_path: str | Path
) -> dict[str, Any]:
    """Seal one frozen policy request without contacting its endpoint."""

    try:
        payload = json.loads(json.dumps(dict(request), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("native_task_policy_execution_spec_request_invalid") from exc
    supplied_digest = payload.pop("execution_spec_digest", None)
    payload["execution_spec_digest"] = canonical_digest(
        payload, digest_field="execution_spec_digest"
    )
    if supplied_digest not in (None, "", payload["execution_spec_digest"]):
        raise ValueError("native_task_policy_execution_spec_digest_invalid")
    validated = validate_native_task_policy_execution_spec(payload)
    destination = Path(output_path).expanduser().resolve()
    if destination.exists() or destination.is_symlink():
        raise ValueError("native_task_policy_execution_spec_output_exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validated


def build_native_task_policy_execution_spec(
    *,
    candidate_id: str,
    scene_plan_path: str | Path,
    construction_result_path: str | Path,
    control_result_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Derive and seal one candidate spec from authoritative predecessors.

    This is the provider-free bridge between a completed controls receipt and
    the policy bundle builder.  Callers choose only one of the already-frozen
    candidates; task, cell, prompt, query budget, checkpoint identity, and all
    predecessor digests are derived here instead of being copied by hand.
    """

    scene = _read(scene_plan_path, error="native_task_policy_scene_plan_invalid")
    construction = _read(
        construction_result_path,
        error="native_task_policy_construction_result_invalid",
    )
    controls = _read(
        control_result_path, error="native_task_policy_control_result_invalid"
    )
    pair = controls.get("control_pair") or {}
    task_spec = scene.get("task_spec") or {}
    scene_digest = scene.get("plan_digest")
    construction_digest = construction.get("result_digest")
    control_digest = controls.get("result_digest")
    pair_digest = pair.get("pair_digest") if isinstance(pair, Mapping) else None
    cell_id = (scene.get("scenario") or {}).get("cell_id")
    errors: list[str] = []
    if (
        scene.get("schema_version") != "native_task_arena_scene_plan.v1"
        or scene_digest != canonical_digest(scene, digest_field="plan_digest")
        or not isinstance(task_spec, Mapping)
        or not str(scene.get("task_id") or "")
        or not str(cell_id or "")
        or not str(task_spec.get("prompt") or "")
    ):
        errors.append("native_task_policy_scene_plan_invalid")
    if (
        construction.get("schema_version")
        != "native_task_arena_construction_result.v1"
        or construction.get("status") != "completed"
        or construction.get("construction_gate_qualified") is not True
        or construction.get("scene_plan_digest") != scene_digest
        or construction_digest
        != canonical_digest(construction, digest_field="result_digest")
    ):
        errors.append("native_task_policy_construction_not_qualified")
    if (
        controls.get("schema_version") != "native_task_arena_control_result.v1"
        or controls.get("status") != "completed"
        or controls.get("controls_qualified") is not True
        or controls.get("candidate_policy_queried") is not False
        or controls.get("scene_plan_digest") != scene_digest
        or controls.get("construction_result_digest") != construction_digest
        or control_digest != canonical_digest(controls, digest_field="result_digest")
        or not isinstance(pair, Mapping)
        or pair.get("schema_version") != "adp_task_control_pair.v1"
        or pair.get("cell_id") != cell_id
        or pair.get("task_spec_digest") != canonical_digest(task_spec)
        or pair.get("cell_admitted_for_policy_execution") is not True
        or pair.get("policy_execution_blockers") != []
        or pair.get("candidate_policy_queried") is not False
        or pair_digest != canonical_digest(pair, digest_field="pair_digest")
    ):
        errors.append("native_task_policy_controls_not_qualified")
    if candidate_id not in FROZEN_CANDIDATES:
        errors.append("native_task_policy_candidate_invalid")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))

    policy, endpoint, identity = _candidate_runtime_binding(candidate_id)

    request = {
        "schema_version": EXECUTION_SPEC_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "task_id": scene["task_id"],
        "cell_id": cell_id,
        "prompt": task_spec["prompt"],
        "scene_plan_digest": scene_digest,
        "construction_result_digest": construction_digest,
        "control_result_digest": control_digest,
        "control_pair_digest": pair_digest,
        "policy_endpoint": endpoint,
        "policy_spec": asdict(policy),
        "policy_identity_receipt": identity,
        "max_policy_queries": maximum_policy_queries_for_task_spec(
            task_spec, open_loop_horizon=policy.open_loop_horizon
        ),
        "open_loop_horizon": policy.open_loop_horizon,
        "overview_camera_policy_input": False,
        "policy_may_grade_itself": False,
        "execution_authority": QUALIFIED_EXECUTION_AUTHORITY,
    }
    return materialize_native_task_policy_execution_spec(
        request=request, output_path=output_path
    )


def _candidate_runtime_binding(
    candidate_id: str,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Return the frozen runtime binding without inspecting outcome receipts."""

    if candidate_id == "pi05_droid":
        from .openpi_droid_policy_runtime import OpenPIDroidPolicySpec

        inventory = _read(
            OPENPI_CHECKPOINT_INVENTORY_PATH,
            error="native_task_policy_openpi_checkpoint_inventory_invalid",
        )
        entries = [
            row
            for row in inventory.get("entries") or []
            if isinstance(row, Mapping)
            and row.get("policy_id") == "pi05_droid_jointpos_polaris"
        ]
        if len(entries) != 1:
            raise ValueError("native_task_policy_openpi_checkpoint_inventory_invalid")
        entry = entries[0]
        policy = OpenPIDroidPolicySpec(
            policy_id="pi05_droid_jointpos_polaris",
            config_name="pi05_droid_jointpos_polaris",
            checkpoint_uri=str(entry.get("checkpoint_uri") or ""),
            checkpoint_object_manifest_sha256=str(
                entry.get("legacy_object_manifest_sha256") or ""
            ),
            checkpoint_generation_manifest_sha256=str(
                entry.get("generation_manifest_sha256") or ""
            ),
            checkpoint_inventory_sha256=str(
                inventory.get("inventory_sha256") or ""
            ),
            checkpoint_object_count=int(entry.get("object_count") or 0),
            checkpoint_size_bytes=int(entry.get("size_bytes") or 0),
            action_space="joint_position",
            action_chunk_rows=15,
        )
        policy.validate()
        endpoint: dict[str, Any] = {
            "host": "127.0.0.1",
            "port": 8000,
            "credential_env": "BLUEPRINT_PI05_API_KEY",
        }
        identity: dict[str, Any] = {"identity_verified": True}
    elif candidate_id == "groot_n17_droid":
        from .groot_n17_droid_policy_runtime import GrootN17DroidPolicySpec

        policy = GrootN17DroidPolicySpec()
        policy.validate()
        endpoint = {
            "host": "127.0.0.1",
            "port": 5555,
            "credential_env": "BLUEPRINT_GROOT_API_TOKEN",
        }
        identity = dict(GROOT_RUNTIME_IDENTITY_DECLARATION)
    else:
        raise ValueError("native_task_policy_candidate_invalid")
    return policy, endpoint, identity


def build_native_task_arena_policy_bundle(
    *,
    job_dir: str | Path,
    packet_dir: str | Path,
    construction_result_path: str | Path,
    control_result_path: str | Path,
    policy_execution_spec: Mapping[str, Any],
    runtime_source_packet_receipt: str | Path,
    implementation_commit: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Require successful construction and controls before bundling a candidate."""

    packet = Path(packet_dir).expanduser().resolve()
    scene_plan = _read(
        packet / "native_task_arena_scene_plan.v1.json",
        error="native_task_policy_scene_plan_invalid",
    )
    construction_path = Path(construction_result_path).expanduser().resolve()
    controls_path = Path(control_result_path).expanduser().resolve()
    construction = _read(
        construction_path, error="native_task_policy_construction_result_invalid"
    )
    controls = _read(controls_path, error="native_task_policy_control_result_invalid")
    spec = validate_native_task_policy_execution_spec(policy_execution_spec)
    pair = controls.get("control_pair") or {}
    errors: list[str] = []
    task_spec = scene_plan.get("task_spec") or {}
    try:
        expected_policy_queries = maximum_policy_queries_for_task_spec(
            task_spec,
            open_loop_horizon=int(spec["open_loop_horizon"]),
        )
    except (TypeError, ValueError):
        expected_policy_queries = None
        errors.append("native_task_policy_shared_query_budget_invalid")
    if spec.get("prompt") != task_spec.get("prompt"):
        errors.append("native_task_policy_prompt_task_spec_mismatch")
    if spec.get("max_policy_queries") != expected_policy_queries:
        errors.append("native_task_policy_shared_query_budget_mismatch")
    if (
        construction.get("schema_version")
        != "native_task_arena_construction_result.v1"
        or construction.get("status") != "completed"
        or construction.get("construction_gate_qualified") is not True
        or construction.get("scene_plan_digest") != scene_plan.get("plan_digest")
        or construction.get("result_digest")
        != spec.get("construction_result_digest")
    ):
        errors.append("native_task_policy_construction_not_qualified")
    if (
        controls.get("schema_version") != "native_task_arena_control_result.v1"
        or controls.get("status") != "completed"
        or controls.get("controls_qualified") is not True
        or controls.get("result_digest") != spec.get("control_result_digest")
        or not isinstance(pair, Mapping)
        or pair.get("cell_admitted_for_policy_execution") is not True
        or pair.get("pair_digest") != spec.get("control_pair_digest")
    ):
        errors.append("native_task_policy_controls_not_qualified")
    if (
        spec.get("scene_plan_digest") != scene_plan.get("plan_digest")
        or spec.get("task_id") != scene_plan.get("task_id")
        or spec.get("cell_id") != (scene_plan.get("scenario") or {}).get("cell_id")
        or pair.get("cell_id") != spec.get("cell_id")
        or pair.get("task_spec_digest")
        != canonical_digest(task_spec)
    ):
        errors.append("native_task_policy_task_cell_binding_mismatch")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))

    with tempfile.TemporaryDirectory(prefix="blueprint-native-task-policy-") as raw:
        execution_path = Path(raw) / "native_task_arena_policy_execution_spec.v1.json"
        execution_path.write_text(
            json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        package = Path(__file__).resolve().parent
        sources = set(controls_runtime_sources())
        sources.update(package / name for name in POLICY_EXTRA_RUNTIME_MODULE_NAMES)
        bound_inputs = {
            "native_task_arena_construction_result.v1.json": construction_path,
            "native_task_arena_control_result.v1.json": controls_path,
            execution_path.name: execution_path,
        }
        if spec["candidate_id"] == "pi05_droid":
            bound_inputs["openpi_polaris_checkpoint_inventory.json"] = (
                _verified_openpi_checkpoint_inventory(spec["policy_spec"])
            )
        receipt = build_native_task_arena_bundle(
            job_dir=job_dir,
            packet_dir=packet,
            runtime_source_packet_receipt=runtime_source_packet_receipt,
            worker_source=package / "native_task_arena_policy_worker.py",
            runtime_module_sources=sorted(sources),
            implementation_commit=implementation_commit,
            execution_mode="policy",
            policy_candidate_id=spec["candidate_id"],
            expected_output_filename=RESULT_FILENAME,
            container_image=NATIVE_TASK_ARENA_IMAGE,
            bound_runtime_inputs=bound_inputs,
            generated_at=generated_at,
        )
    return receipt


def load_verified_native_task_arena_policy_bundle(
    receipt_path: str | Path,
    *,
    expected_implementation_commit: str,
    expected_packet_receipt_digest: str | None = None,
    expected_runtime_source_packet_digest: str | None = None,
) -> dict[str, Any]:
    """Reverify immutable policy bundle bytes without rebuilding the episode."""

    receipt = _read(
        receipt_path, error="native_task_policy_bundle_receipt_invalid"
    )
    bundle = Path(str(receipt.get("bundle_path") or "")).expanduser().resolve()
    errors: list[str] = []
    input_names = {
        Path(str(row.get("relative_path") or "")).name
        for row in receipt.get("bound_runtime_inputs") or []
        if isinstance(row, Mapping)
    }
    candidate_id = str(receipt.get("policy_candidate_id") or "")
    expected_input_names = {
        "native_task_arena_construction_result.v1.json",
        "native_task_arena_control_result.v1.json",
        "native_task_arena_policy_execution_spec.v1.json",
    }
    if candidate_id == "pi05_droid":
        expected_input_names.add("openpi_polaris_checkpoint_inventory.json")
    runtime_root_names = {
        str(row.get("relative_path") or "")
        for row in receipt.get("runtime_root_modules") or []
        if isinstance(row, Mapping)
    }
    if (
        receipt.get("schema_version") != "native_task_arena_provider_bundle.v1"
        or receipt.get("status") != "ready"
        or receipt.get("execution_mode") != "policy"
        or receipt.get("policy_candidate_id") not in FROZEN_CANDIDATES
        or receipt.get("candidate_policy_queried") is not False
        or receipt.get("expected_output_filename") != RESULT_FILENAME
        or input_names != expected_input_names
        or receipt.get("policy_provisioning_script")
        != f"adp009d_policy_provisioning.{candidate_id}.sh"
        or (receipt.get("policy_provisioning") or {}).get("relative_path")
        != f"adp009d_policy_provisioning.{candidate_id}.sh"
        or runtime_root_names
        != set(POLICY_RUNTIME_ROOT_MODULE_NAMES)
    ):
        errors.append("native_task_policy_bundle_contract_invalid")
    if receipt.get("implementation_commit") != expected_implementation_commit:
        errors.append("native_task_policy_bundle_commit_mismatch")
    if receipt.get("container_image") != NATIVE_TASK_ARENA_IMAGE:
        errors.append("native_task_policy_bundle_image_mismatch")
    if expected_packet_receipt_digest and (
        receipt.get("packet_receipt_digest") != expected_packet_receipt_digest
    ):
        errors.append("native_task_policy_bundle_packet_mismatch")
    source = receipt.get("runtime_source_packet") or {}
    if expected_runtime_source_packet_digest and (
        source.get("receipt_digest") != expected_runtime_source_packet_digest
    ):
        errors.append("native_task_policy_bundle_sources_mismatch")
    manifest = {
        key: value
        for key, value in receipt.items()
        if key not in {"bundle_path", "bundle_size_bytes", "bundle_sha256"}
    }
    if receipt.get("input_digest") != canonical_digest(
        manifest, digest_field="input_digest"
    ):
        errors.append("native_task_policy_bundle_input_digest_invalid")
    if (
        not bundle.is_file()
        or receipt.get("bundle_size_bytes") != bundle.stat().st_size
        or receipt.get("bundle_sha256") != _sha256(bundle)
    ):
        errors.append("native_task_policy_bundle_bytes_identity_mismatch")
    if errors:
        raise ValueError(";".join(sorted(set(errors))))
    return receipt


__all__ = [
    "EXECUTION_SPEC_SCHEMA_VERSION",
    "GROOT_RUNTIME_IDENTITY_DECLARATION",
    "GROOT_RUNTIME_IDENTITY_FILENAME",
    "PROBE_KIND",
    "RESULT_FILENAME",
    "RESULT_SCHEMA_VERSION",
    "build_native_task_arena_policy_bundle",
    "build_native_task_policy_execution_spec",
    "load_verified_native_task_arena_policy_bundle",
    "materialize_native_task_policy_execution_spec",
    "validate_native_task_policy_execution_spec",
]

def main(argv: list[str] | None = None) -> int:
    """Build the sealed native Arena policy packet.

    The Arena family is a chain -- construction, then controls, then policy --
    and each link consumes the previous link's result. None of the three could
    be produced except by calling a Python function, so the whole chain was
    unreachable from any production path.

    The allocator refuses a bundle whose commit is not the one the control
    plane is running, and every deploy moves that commit, so a bundle that can
    only be built by hand is launchable at most once.

    Performs no provider mutation and rents nothing.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Build the sealed native Arena policy packet.")
    parser.add_argument("--job-dir", dest="job_dir", required=True)
    parser.add_argument("--packet-dir", dest="packet_dir", required=True)
    parser.add_argument("--construction-result", dest="construction_result_path", required=True)
    parser.add_argument("--control-result", dest="control_result_path", required=True)
    parser.add_argument("--runtime-source-packet-receipt", dest="runtime_source_packet_receipt", required=True)
    parser.add_argument("--implementation-commit", dest="implementation_commit", required=True)
    parser.add_argument("--generated-at", dest="generated_at")
    parser.add_argument("--policy-execution-spec", dest="policy_execution_spec", required=True,
                        help="Path to the JSON execution spec.")
    args = parser.parse_args(argv)

    try:
        spec = json.loads(Path(args.policy_execution_spec).read_text(encoding="utf-8"))
        receipt = build_native_task_arena_policy_bundle(
            job_dir=args.job_dir,
            packet_dir=args.packet_dir,
            construction_result_path=args.construction_result_path,
            control_result_path=args.control_result_path,
            runtime_source_packet_receipt=args.runtime_source_packet_receipt,
            implementation_commit=args.implementation_commit,
            policy_execution_spec=spec,
            **({"generated_at": args.generated_at} if args.generated_at else {}),
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": [f"{type(exc).__name__}:{exc}"],
                    "provider_mutation_performed": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt.get("status") in {"ready", "sealed"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
