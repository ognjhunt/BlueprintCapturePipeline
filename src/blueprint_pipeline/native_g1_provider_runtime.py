"""Execute the selected G1 book and movement pairs inside pinned Isaac.

The paid allocator owns the GPU, watchdog, output upload, and provider teardown.
This runner owns the four development episodes and their lossless media. It
expects the source packet to have been provisioned by the provider entrypoint
before this process starts, so newly installed Isaac packages are importable.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import threading
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_development_campaign import _movement_handoff, plan_g1_development_campaign
from .native_g1_development_pair import (
    EPISODE_FILENAME,
    PAIR_ORDER,
    TRACE_FILENAME,
    run_g1_development_pair,
)
from .native_g1_development_selection import stage_g1_development_selection
from .native_g1_policy_runtime_build import (
    PINNED_IMAGE_ENV,
    execute_g1_policy_runtime_build,
    prepare_g1_policy_runtime_build,
)
from .native_g1_publisher_source_stage import verify_g1_publisher_source
from .native_task_isaaclab_launch import NATIVE_TASK_ARENA_IMAGE


SCHEMA = "native_g1_provider_campaign_result.v1"
RESULT_FILENAME = SCHEMA + ".json"
CAMPAIGN_FILENAME = "native_g1_development_campaign.v1.json"
MANIPULATION = "task_success"
MOVEMENT = "g1_navigation_goal"
G1_RUNTIME_IMPORTS = (
    "isaaclab_arena_g1", "pinocchio", "pink", "scipy", "qpsolvers",
    "onnxruntime", "google.protobuf",
)


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("g1_provider_runtime_input_missing_or_symlink")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("g1_provider_runtime_input_invalid")
    return value


def _preflight_g1_runtime_imports(output: Path) -> dict[str, Any]:
    """Retain the G1 import closure before policy build and model downloads."""

    rows: list[dict[str, Any]] = []
    for name in G1_RUNTIME_IMPORTS:
        try:
            module = importlib.import_module(name)
            rows.append({
                "module": name, "available": True,
                "version": str(getattr(module, "__version__", "unreported")),
            })
        except Exception as exc:  # noqa: BLE001 - complete paid preflight evidence
            rows.append({
                "module": name, "available": False,
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
            })
    result = {
        "schema_version": "native_g1_runtime_import_preflight.v1",
        "status": "passed" if all(row["available"] for row in rows) else "blocked",
        "imports": rows,
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    (output / "native_g1_runtime_import_preflight.v1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _rights_paths(root: Path) -> dict[str, Path]:
    return {candidate: root / "inputs/rights" / (candidate + ".json") for candidate in PAIR_ORDER}


def verify_g1_provider_inputs(runtime_root: Path) -> dict[str, Any]:
    """Recompute owner and source bindings before model download or inference."""

    root = Path(runtime_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("g1_provider_runtime_root_invalid")
    inputs = root / "inputs"
    campaign = plan_g1_development_campaign(
        book_handoff_path=inputs / "book_handoff.json",
        movement_handoff_path=(
            inputs / "movement_handoff.json"
            if (inputs / "movement_handoff.json").is_file() else None
        ),
        manipulation_packet=inputs / "scene_packets/manipulation",
        movement_packet=inputs / "scene_packets/movement",
        inventory_path=root.parent / "configs/g1_humanoidarena_checkpoint_inventory.v1.json",
        rights_review_paths=_rights_paths(root),
        navigation_authority_path=inputs / "navigation_authority.json",
    )
    if campaign != _json(inputs / CAMPAIGN_FILENAME):
        raise ValueError("g1_provider_campaign_plan_changed")
    source = verify_g1_publisher_source(root / "publisher-source")
    staged_source = _json(root / "publisher-source/native_g1_publisher_source_stage.v1.json")
    if staged_source.get("receipt_digest") != canonical_digest(
        staged_source, digest_field="receipt_digest"
    ) or any(
        source.get(field) != staged_source.get(field)
        for field in (
            "source_repository",
            "source_revision",
            "inventory_file_sha256",
            "policy_server_sha256",
            "sonic_provider_sha256",
            "lerobot_pyproject_sha256",
            "model_weights_included",
            "gpu_allocated",
        )
    ):
        raise ValueError("g1_provider_publisher_source_changed")
    provision = _json(
        root.parent / "runtime_output/native_task_runtime_source_provisioning.v1.json"
    )
    source_receipt = _json(
        root / "native_task_runtime_sources/native_task_runtime_source_packet.v1.json"
    )
    if (
        provision.get("status") != "completed"
        or provision.get("source_packet_sha256") != source_receipt.get("packet_sha256")
        or source_receipt.get("status") != "ready"
    ):
        raise ValueError("g1_provider_runtime_source_provisioning_incomplete")
    return {
        "campaign": campaign,
        "publisher_source": source,
        "runtime_source_receipt_digest": source_receipt["receipt_digest"],
        "runtime_source_packet_sha256": source_receipt["packet_sha256"],
        "runtime_provisioning_receipt_path": str(
            root.parent / "runtime_output/native_task_runtime_source_provisioning.v1.json"
        ),
    }


def _load_script(path: Path, name: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError("g1_provider_fetcher_missing")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("g1_provider_fetcher_invalid")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Heartbeat:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self.stop.wait(30):
            print("BLUEPRINT_G1_STAGE_PROGRESS:" + self.stage, flush=True)

    def __enter__(self) -> "_Heartbeat":
        print("BLUEPRINT_G1_STAGE_STARTED:" + self.stage, flush=True)
        self.thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop.set()
        self.thread.join(timeout=5)
        print("BLUEPRINT_G1_STAGE_FINISHED:" + self.stage, flush=True)


def _stage_models(root: Path, output: Path) -> dict[str, Any]:
    inventory = root.parent / "configs/g1_humanoidarena_checkpoint_inventory.v1.json"
    sonic_inventory = root.parent / "configs/g1_sonic_default_asset_inventory.v1.json"
    checkpoint_fetcher = _load_script(
        root / "scripts/fetch_g1_humanoidarena_checkpoint.py", "g1_checkpoint_fetcher"
    )
    sonic_fetcher = _load_script(root / "scripts/fetch_g1_sonic_assets.py", "g1_sonic_fetcher")
    models = output / "models"
    models.mkdir()
    checkpoints = models / "checkpoints"

    def fetch_checkpoint(candidate: str) -> dict[str, Any]:
        with _Heartbeat("checkpoint:" + candidate):
            return checkpoint_fetcher.materialize_candidate(
                inventory_path=inventory,
                candidate_id=candidate,
                output_dir=checkpoints,
            )

    # Candidate downloads use distinct pinned paths. Fetch them together so
    # large pi0.5 weights do not consume the whole bounded GPU lease serially.
    with ThreadPoolExecutor(max_workers=len(PAIR_ORDER)) as pool:
        receipts = list(pool.map(fetch_checkpoint, PAIR_ORDER))
    for candidate, receipt in zip(PAIR_ORDER, receipts, strict=True):
        (models / (candidate + ".json")).write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    sonic_root = models / "sonic"
    with _Heartbeat("sonic-assets"):
        sonic = sonic_fetcher.stage_sonic_assets(
            inventory_path=sonic_inventory, output_dir=sonic_root
        )
    (models / "sonic.json").write_text(
        json.dumps(sonic, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"checkpoints": receipts, "sonic": sonic, "root": str(models)}


def _template(
    *,
    root: Path,
    output: Path,
    packet: Path,
    objective_name: str,
    models: Mapping[str, Any],
    runtime_python: Path,
    provisioning_path: Path,
) -> Path:
    sonic = {row["role"]: row for row in models["sonic"]["files"]}
    source = root / "publisher-source/source"
    value = {
        "schema_version": "native_g1_development_episode_request.v1",
        "bundle_root": str(packet),
        "inventory_path": str(
            root.parent / "configs/g1_humanoidarena_checkpoint_inventory.v1.json"
        ),
        "checkpoint_root": str(output / "models/checkpoints"),
        "policy_server_source": str(source / "lerobot/scripts/serve_lerobot_vla_http.py"),
        "sonic_provider_source": str(
            source / "isaaclab_twist2_g1/action_provider/action_provider_sonic.py"
        ),
        "sonic_encoder": str(output / "models/sonic/model_encoder.onnx"),
        "sonic_decoder": str(output / "models/sonic/model_decoder.onnx"),
        "sonic_encoder_sha256": sonic["encoder"]["sha256"],
        "sonic_decoder_sha256": sonic["decoder"]["sha256"],
        "python_executable": str(runtime_python),
        "runtime_provisioning_receipt_path": str(provisioning_path),
        "port": 8765,
        "max_steps": 3000,
        "device": "cuda:0",
    }
    if objective_name not in {"movement", "manipulation"}:
        raise ValueError("g1_provider_template_objective_invalid")
    path = output / (objective_name + "_runtime_template.json")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _query_count(pair_output: Path, candidate: str) -> int:
    episode = _json(pair_output / candidate / "episode" / EPISODE_FILENAME)
    trace = _json(pair_output / candidate / "episode" / TRACE_FILENAME)
    if (
        episode.get("trace_digest") != trace.get("trace_digest")
        or trace.get("trace_digest") != canonical_digest(trace, digest_field="trace_digest")
        or trace.get("candidate_id") != candidate
        or not isinstance(trace.get("policy_query_count"), int)
        or trace["policy_query_count"] < 1
    ):
        raise ValueError("g1_provider_policy_query_evidence_invalid")
    return trace["policy_query_count"]


def run_g1_provider_campaign(runtime_root: Path, output_dir: Path) -> dict[str, Any]:
    """Run the verified campaign once; always retain a terminal result."""

    root = Path(runtime_root)
    output = Path(output_dir)
    if (
        not output.is_absolute()
        or not output.is_dir()
        or output.is_symlink()
        or output.resolve() != output
        or (output / RESULT_FILENAME).exists()
        or os.environ.get(PINNED_IMAGE_ENV) != NATIVE_TASK_ARENA_IMAGE
        or not Path("/isaac-sim/python.sh").is_file()
    ):
        raise ValueError("g1_provider_runtime_environment_invalid")
    stage = "input-verification"
    pair_results: list[dict[str, Any]] = []
    queries: dict[str, int] = {}
    blockers: list[str] = []
    identity: dict[str, Any] = {}
    try:
        with _Heartbeat(stage):
            identity = verify_g1_provider_inputs(root)
        stage = "runtime-import-preflight"
        with _Heartbeat(stage):
            runtime_imports = _preflight_g1_runtime_imports(output)
        if runtime_imports["status"] != "passed":
            raise ValueError("g1_provider_runtime_dependency_preflight_blocked")
        stage = "policy-runtime-build"
        with _Heartbeat(stage):
            build = execute_g1_policy_runtime_build(
                plan=prepare_g1_policy_runtime_build(
                    checkout=root / "publisher-source/source",
                    output_dir=output / "policy-runtime-build",
                    execution_mode="inside_isaac_container",
                )
            )
        if build["status"] != "built_import_probe_passed_no_cuda_probe":
            raise ValueError("g1_provider_policy_runtime_build_blocked")
        stage = "model-staging"
        models = _stage_models(root, output)
        source_handoff = _json(root / "inputs/book_handoff.json")
        supplied_movement_handoff = root / "inputs/movement_handoff.json"
        movement_handoff = (
            _json(supplied_movement_handoff)
            if supplied_movement_handoff.is_file()
            else _movement_handoff(source_handoff)
        )
        movement_handoff_path = output / "movement_handoff.json"
        movement_handoff_path.write_text(
            json.dumps(movement_handoff, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for objective, packet_name, handoff_path, candidates in (
            (MANIPULATION, "manipulation", root / "inputs/book_handoff.json", PAIR_ORDER[:2]),
            (MOVEMENT, "movement", movement_handoff_path, PAIR_ORDER[2:]),
        ):
            packet = root / "inputs/scene_packets" / packet_name
            template_path = _template(
                root=root,
                output=output,
                packet=packet,
                objective_name=packet_name,
                models=models,
                runtime_python=output / "policy-runtime-build/policy-runtime/bin/python",
                provisioning_path=Path(identity["runtime_provisioning_receipt_path"]),
            )
            stage = "selection:" + objective
            selected = stage_g1_development_selection(
                setup_path=None,
                selection_path=None,
                runtime_template_path=template_path,
                rights_review_paths={
                    candidate: _rights_paths(root)[candidate] for candidate in candidates
                },
                output_dir=output / (packet_name + "_selection"),
                navigation_authority_path=(
                    root / "inputs/navigation_authority.json" if objective == MOVEMENT else None
                ),
                handoff_path=handoff_path,
            )
            stage = "episodes:" + objective
            with _Heartbeat(stage):
                pair = run_g1_development_pair(
                    request_paths=[Path(path) for path in selected["request_paths"]],
                    output_dir=output / (packet_name + "_pair"),
                    mode="subprocess",
                    worker_launcher=Path("/isaac-sim/python.sh"),
                )
            pair_results.append(
                {
                    "objective_id": objective,
                    "pair_result_digest": pair["result_digest"],
                    "pair_relative_path": packet_name + "_pair/native_g1_development_pair.v1.json",
                    "status": pair["status"],
                    "candidate_ids": pair["candidate_ids"],
                }
            )
            if pair["status"] != "completed_development_only":
                raise ValueError("g1_provider_" + packet_name + "_pair_blocked")
            for candidate in candidates:
                queries[candidate] = _query_count(output / (packet_name + "_pair"), candidate)
    except Exception as exc:  # noqa: BLE001 - terminal paid-run evidence must survive
        blockers.append(type(exc).__name__ + ":" + str(exc)[:300])
    for packet_name, candidates in (("manipulation", PAIR_ORDER[:2]), ("movement", PAIR_ORDER[2:])):
        for candidate in candidates:
            if candidate in queries:
                continue
            trace_path = output / (packet_name + "_pair") / candidate / "episode" / TRACE_FILENAME
            if not trace_path.is_file() or trace_path.is_symlink():
                continue
            try:
                trace = _json(trace_path)
                if (
                    trace.get("candidate_id") == candidate
                    and trace.get("trace_digest")
                    == canonical_digest(trace, digest_field="trace_digest")
                    and isinstance(trace.get("policy_query_count"), int)
                    and trace["policy_query_count"] > 0
                ):
                    queries[candidate] = trace["policy_query_count"]
            except (OSError, ValueError, KeyError, TypeError):
                pass
    completed = (
        not blockers
        and len(pair_results) == 2
        and [row["status"] for row in pair_results] == ["completed_development_only"] * 2
        and list(queries) == list(PAIR_ORDER)
    )
    result = {
        "schema_version": SCHEMA,
        "status": "completed" if completed else "blocked",
        "claim_ceiling": "development_only",
        "campaign_plan_digest": (identity.get("campaign") or {}).get("plan_digest"),
        "publisher_source_receipt_digest": (identity.get("publisher_source") or {}).get(
            "receipt_digest"
        ),
        "runtime_source_packet_sha256": identity.get("runtime_source_packet_sha256"),
        "pairs": pair_results,
        "policy_query_counts": queries,
        "candidate_policy_queried": bool(queries),
        "ranking_eligible": False,
        "physical_outcome_claimed": False,
        "stage_reached": stage,
        "blockers": blockers,
    }
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    (output / RESULT_FILENAME).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_g1_provider_campaign(args.runtime_root, args.output_dir)
    print(json.dumps({"status": result["status"], "result_digest": result["result_digest"]}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
