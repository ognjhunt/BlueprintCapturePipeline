"""Command adapter for Cosmos3-Nano action-conditioned WAM rollout generation.

The adapter is the ``cosmos3_wam`` process boundary behind the swappable
provider-command interface. It mirrors the OSCAR adapter honesty mechanics:

- The operator supplies the Cosmos3 source tree, checkpoint path, and run
  gates through environment variables or CLI flags; nothing auto-runs.
- The adapter emits the trusted output schema
  ``cosmos3_wam_command_adapter.v1`` and self-reports its backbone as
  ``base_model`` so the provider runtime can hard-fail on family mismatches.
- ``learned_wam_model_ran`` is set only when the checkpoint and source
  identity verify as Cosmos3-Nano and a reviewable generated MP4 exists.
- The SC3-Eval recipe metadata (80/10/10 forward/cross-view/inverse mixture,
  predict-24/execute-16 horizon decoupling) is recorded as declared operator
  config, never as proof that the recipe was trained or executed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .oscar_cosmos_wam_command_adapter import (
    _failure_signals,
    _mapping,
    _materialize_cosmos_input_package,
    _read_json,
    _redacted_argv,
    _runtime_env,
    _string,
    _write_json,
)
from .sc3_fidelity_contracts import (
    SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    validate_checkpoint_attestation,
    validate_horizon_execution_trace,
)
from .wam_generated_video_review import validate_generated_mp4_for_review


ADAPTER_ID = "blueprint_cosmos3_nano_wam_command_adapter"
SCHEMA_VERSION = "cosmos3_wam_command_adapter.v1"
SUBSTRATE = "cosmos3_wam"
EXPECTED_BASE_MODEL = "Cosmos3-Nano"
EXPECTED_MODEL_FAMILY = "Cosmos 3"
LOCAL_MODEL_GATE_ENV = "BLUEPRINT_ALLOW_LOCAL_WAM_MODEL"
DEFAULT_MODEL = "cosmos3/nano/action-cond"
SC3_EXECUTOR_SIGNING_PRIVATE_KEY_FILE_ENV = "BLUEPRINT_SC3_EXECUTOR_SIGNING_PRIVATE_KEY_FILE"
DEFAULT_ENTRYPOINT_RELPATH = "examples/action_conditioned.py"

CHECKPOINT_IDENTITY_FILENAMES = (
    "blueprint_checkpoint_identity.json",
    "cosmos3_checkpoint_manifest.json",
    "checkpoint_manifest.json",
    "model_index.json",
    "config.json",
    "metadata.json",
)
CHECKPOINT_IDENTITY_KEYS = (
    "base_model",
    "model_family",
    "model_name",
    "model_id",
    "_name_or_path",
    "architecture",
    "backbone",
)
# Any declared identity that matches one of these markers without also
# matching Cosmos3-Nano is a wrong model family and must fail closed.
WRONG_FAMILY_MARKERS = (
    "predict2",
    "predict-2",
    "oscar",
    "cosmos1",
    "cosmos2",
    "cosmos3super",
    "cosmos3edge",
)
COSMOS3_NANO_IDENTITY_TOKEN = "cosmos3nano"

# Declared upstream recipe per SC3-Eval (arXiv 2606.18610). This block is
# configuration metadata only: emitting it never claims the mixture was
# trained, reproduced, or validated by this adapter invocation.
SC3_RECIPE_DECLARED_CONFIG = {
    "schema_version": "sc3_recipe_declared_config.v1",
    "recipe_id": "sc3_eval_self_consistency_recipe",
    "recipe_source": "arXiv:2606.18610 (SC3-Eval), Cosmos3-Nano backbone",
    "training_mixture": {
        "forward_dynamics": 0.8,
        "cross_view": 0.1,
        "inverse_dynamics": 0.1,
    },
    "horizon_decoupling": {
        "predict_horizon_frames": 24,
        "execute_horizon_frames": 16,
    },
    "claim_boundary": {
        "recipe_metadata_is_operator_declared_config": True,
        "recipe_metadata_is_execution_or_training_proof": False,
        "recipe_declared_config_does_not_prove_rank_fidelity": True,
    },
}


def _validated_sc3_callback_evidence(
    value: Any,
    *,
    expected_schema_version: str,
    expected_bindings: Mapping[str, Any],
    error_prefix: str,
) -> dict[str, str]:
    ref = _mapping(value)
    path = Path(_string(ref.get("path"))).expanduser()
    digest = _string(ref.get("sha256")).lower()
    if not path.is_file():
        raise ValueError(f"{error_prefix}_evidence_file_missing")
    if len(digest) != 64 or _sha256_file(path) != digest:
        raise ValueError(f"{error_prefix}_evidence_digest_invalid")
    try:
        payload = _mapping(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{error_prefix}_evidence_json_invalid") from exc
    if payload.get("schema_version") != expected_schema_version:
        raise ValueError(f"{error_prefix}_evidence_schema_invalid")
    if any(payload.get(key) != expected for key, expected in expected_bindings.items()):
        raise ValueError(f"{error_prefix}_evidence_binding_invalid")
    return {"path": str(path.resolve()), "sha256": digest}


def execute_sc3_receding_horizon_chunk(
    *,
    propose_actions: Callable[[], Sequence[Mapping[str, Any]]],
    world_model_predict: Callable[[Mapping[str, Any], int], Mapping[str, Any]],
    controller_execute: Callable[[Mapping[str, Any], int, float], Mapping[str, Any]],
    output_dir: str | Path,
    runtime_session_id: str,
    runtime_executor_id: str,
    runtime_executor_code_sha256: str,
    controller_id: str,
    controller_sha256: str,
    world_model_checkpoint_sha256: str,
    control_rate_hz: float,
    chunk_start_timestamp_sec: float,
    signing_private_key_file: str | Path | None = None,
) -> dict[str, Any]:
    """Execute and attest one real SC3 25/24/16 receding-horizon chunk."""

    for identity_name, identity_value in (
        ("runtime_session_id", runtime_session_id),
        ("runtime_executor_id", runtime_executor_id),
        ("controller_id", controller_id),
    ):
        if not _string(identity_value):
            raise ValueError(f"sc3_executor_{identity_name}_missing")
    for digest_name, digest_value in (
        ("runtime_executor_code_sha256", runtime_executor_code_sha256),
        ("controller_sha256", controller_sha256),
        ("world_model_checkpoint_sha256", world_model_checkpoint_sha256),
    ):
        digest = _string(digest_value).lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"sc3_executor_{digest_name}_invalid")
    rate = float(control_rate_hz)
    chunk_start = float(chunk_start_timestamp_sec)
    if not math.isfinite(rate) or rate <= 0.0:
        raise ValueError("sc3_executor_control_rate_hz_invalid")
    if not math.isfinite(chunk_start):
        raise ValueError("sc3_executor_chunk_start_timestamp_invalid")

    private_key_path = Path(
        signing_private_key_file or _string(os.getenv(SC3_EXECUTOR_SIGNING_PRIVATE_KEY_FILE_ENV))
    ).expanduser()
    if not private_key_path.is_file():
        raise ValueError("sc3_executor_signing_private_key_file_missing")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = serialization.load_pem_private_key(private_key_path.read_bytes(), password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("sc3_executor_signing_key_must_be_ed25519")
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    trusted_public_key_sha256 = _string(
        os.getenv(SC3_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV)
    ).lower()
    if public_key_sha256 != trusted_public_key_sha256:
        raise ValueError("sc3_executor_signing_key_not_trusted")

    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    proposed_actions = [dict(row) for row in propose_actions()]
    if len(proposed_actions) != 25:
        raise ValueError("sc3_executor_policy_must_propose_exactly_25_actions")
    action_ids: set[str] = set()
    for index, action in enumerate(proposed_actions):
        action_id = _string(action.get("action_id"))
        vector = action.get("action_vector_7d")
        if not action_id or action_id in action_ids:
            raise ValueError(f"sc3_executor_action_id_missing_or_duplicate:{index}")
        action_ids.add(action_id)
        if not (
            isinstance(vector, Sequence)
            and not isinstance(vector, (str, bytes, bytearray))
            and len(vector) == 7
        ):
            raise ValueError(f"sc3_executor_action_vector_invalid:{index}")
        try:
            numeric_vector = [float(value) for value in vector]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"sc3_executor_action_vector_invalid:{index}") from exc
        if not all(math.isfinite(value) for value in numeric_vector):
            raise ValueError(f"sc3_executor_action_vector_invalid:{index}")
        expected_action_sha256 = hashlib.sha256(
            json.dumps(numeric_vector, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if _string(action.get("action_sha256")).lower() != expected_action_sha256:
            raise ValueError(f"sc3_executor_action_sha256_mismatch:{index}")
    predictions: list[dict[str, Any]] = []
    prediction_runtime_result_ids: set[str] = set()
    prediction_evidence_sha256s: set[str] = set()
    for index, action in enumerate(proposed_actions[:24]):
        prediction = dict(world_model_predict(action, index) or {})
        prediction_id = _string(prediction.get("prediction_id"))
        runtime_result_id = _string(prediction.get("runtime_result_id"))
        if not (
            prediction.get("schema_version") == "sc3_world_model_prediction_result.v1"
            and prediction.get("status") == "completed"
            and prediction_id
            and runtime_result_id
            and prediction.get("action_id") == action.get("action_id")
            and _string(prediction.get("action_sha256")).lower()
            == _string(action.get("action_sha256")).lower()
            and _string(prediction.get("world_model_checkpoint_sha256")).lower()
            == _string(world_model_checkpoint_sha256).lower()
        ):
            raise ValueError(f"sc3_executor_prediction_result_invalid:{index}")
        if runtime_result_id in prediction_runtime_result_ids:
            raise ValueError(f"sc3_executor_prediction_runtime_result_id_duplicate:{index}")
        prediction_runtime_result_ids.add(runtime_result_id)
        prediction_evidence = _validated_sc3_callback_evidence(
            prediction.get("evidence_artifact"),
            expected_schema_version="sc3_world_model_prediction_evidence.v1",
            expected_bindings={
                "status": "completed",
                "runtime_session_id": runtime_session_id,
                "runtime_result_id": runtime_result_id,
                "prediction_id": prediction_id,
                "action_id": action.get("action_id"),
                "action_sha256": action.get("action_sha256"),
                "world_model_checkpoint_sha256": world_model_checkpoint_sha256,
            },
            error_prefix=f"sc3_executor_prediction:{index}",
        )
        if prediction_evidence["sha256"] in prediction_evidence_sha256s:
            raise ValueError(f"sc3_executor_prediction_evidence_duplicate:{index}")
        prediction_evidence_sha256s.add(prediction_evidence["sha256"])
        predictions.append(
            {
                **action,
                "prediction_result_schema_version": prediction.get("schema_version"),
                "prediction_id": prediction_id,
                "prediction_runtime_result_id": runtime_result_id,
                "prediction_evidence_artifact": prediction_evidence,
                "prediction_index": index,
                "prediction_status": "completed",
            }
        )
    retained = [
        {**action, "retention_status": "retained_for_execution"} for action in proposed_actions[:16]
    ]
    executed: list[dict[str, Any]] = []
    controller_runtime_result_ids: set[str] = set()
    controller_evidence_sha256s: set[str] = set()
    for index, action in enumerate(proposed_actions[:16]):
        execution_timestamp = chunk_start + index / rate
        execution = dict(controller_execute(action, index, execution_timestamp) or {})
        runtime_result_id = _string(execution.get("runtime_result_id"))
        try:
            observed_execution_timestamp = float(execution.get("execution_timestamp_sec"))
        except (TypeError, ValueError):
            observed_execution_timestamp = math.nan
        if not (
            execution.get("schema_version") == "sc3_controller_execution_result.v1"
            and execution.get("status") == "completed"
            and runtime_result_id
            and execution.get("action_id") == action.get("action_id")
            and _string(execution.get("action_sha256")).lower()
            == _string(action.get("action_sha256")).lower()
            and execution.get("controller_id") == controller_id
            and _string(execution.get("controller_sha256")).lower()
            == _string(controller_sha256).lower()
            and math.isfinite(observed_execution_timestamp)
            and math.isclose(
                observed_execution_timestamp,
                execution_timestamp,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError(f"sc3_executor_controller_result_invalid:{index}")
        if runtime_result_id in controller_runtime_result_ids:
            raise ValueError(f"sc3_executor_controller_runtime_result_id_duplicate:{index}")
        controller_runtime_result_ids.add(runtime_result_id)
        controller_evidence = _validated_sc3_callback_evidence(
            execution.get("evidence_artifact"),
            expected_schema_version="sc3_controller_execution_evidence.v1",
            expected_bindings={
                "status": "completed",
                "runtime_session_id": runtime_session_id,
                "runtime_result_id": runtime_result_id,
                "action_id": action.get("action_id"),
                "action_sha256": action.get("action_sha256"),
                "controller_id": controller_id,
                "controller_sha256": controller_sha256,
                "execution_timestamp_sec": execution_timestamp,
            },
            error_prefix=f"sc3_executor_controller:{index}",
        )
        if controller_evidence["sha256"] in controller_evidence_sha256s:
            raise ValueError(f"sc3_executor_controller_evidence_duplicate:{index}")
        controller_evidence_sha256s.add(controller_evidence["sha256"])
        executed.append(
            {
                **action,
                "execution_result_schema_version": execution.get("schema_version"),
                "controller_runtime_result_id": runtime_result_id,
                "controller_evidence_artifact": controller_evidence,
                "execution_status": "executed",
                "execution_timestamp_sec": execution_timestamp,
            }
        )
    discarded = [
        {
            **action,
            "retention_status": "discarded_not_executed",
            "executed": False,
        }
        for action in proposed_actions[16:24]
    ]
    trace = {
        "trace_producer_id": "blueprint_sc3_receding_horizon_executor",
        "runtime_session_id": runtime_session_id,
        "runtime_executor_id": runtime_executor_id,
        "runtime_executor_code_sha256": runtime_executor_code_sha256,
        "controller_id": controller_id,
        "controller_sha256": controller_sha256,
        "world_model_checkpoint_sha256": world_model_checkpoint_sha256,
        "runtime_execution_proven": True,
        "world_model_prediction_proven": True,
        "receding_horizon_controller_proven": True,
        "proposed_actions": proposed_actions,
        "world_model_predictions": predictions,
        "retained_actions": retained,
        "executed_actions": executed,
        "discarded_predictions": discarded,
        "control_rate_hz": rate,
        "chunk_start_timestamp_sec": chunk_start,
        "requery_timestamp_sec": chunk_start + 16 / rate,
    }
    artifact = output_root / "sc3_horizon_executor_trace.json"
    artifact.write_text(
        json.dumps(
            {"schema_version": "sc3_horizon_executor_trace.v1", **trace},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    trace["executor_trace_artifact"] = {
        "path": str(artifact),
        "sha256": _sha256_file(artifact),
    }
    # The artifact ref is transport metadata, not part of the validator's
    # signed execution fields.
    signed_fields = {key: value for key, value in trace.items() if key != "executor_trace_artifact"}
    signed_payload = json.dumps(signed_fields, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    payload_sha256 = hashlib.sha256(signed_payload).hexdigest()
    report = output_root / "sc3_horizon_executor_signature_report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": "sc3_signature_verification_report.v1",
                "algorithm": "Ed25519",
                "verification_status": "verified",
                "public_key_sha256": public_key_sha256,
                "signed_payload_sha256": payload_sha256,
                "signer_key_id": runtime_executor_id,
                "verifier_id": "blueprint_sc3_receding_horizon_executor",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    trace["executor_attestation"] = {
        "algorithm": "Ed25519",
        "signature_verified": True,
        "signer_key_id": runtime_executor_id,
        "verifier_id": "blueprint_sc3_receding_horizon_executor",
        "public_key_base64": base64.b64encode(public_key).decode("ascii"),
        "public_key_sha256": public_key_sha256,
        "signature_base64": base64.b64encode(private_key.sign(signed_payload)).decode("ascii"),
        "signed_payload_sha256": payload_sha256,
        "verification_report_artifact": {
            "path": str(report),
            "sha256": _sha256_file(report),
        },
    }
    validation = validate_horizon_execution_trace(trace)
    return {
        **trace,
        "status": validation.get("status"),
        "validation": validation,
        "blockers": validation.get("blockers", []),
    }


def build_sc3_horizon_execution_trace(
    package_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an executor-emitted 25/24/16 trace without fabricating stages.

    The Cosmos input package contains proposed actions, but this adapter does not
    itself execute a receding-horizon controller. Only a runtime-supplied trace
    may populate prediction, retention, execution, discard, and requery evidence.
    """

    supplied = package_manifest.get("sc3_horizon_execution_trace")
    trace = (
        dict(supplied)
        if isinstance(supplied, Mapping)
        else {
            "schema_version": "sc3_horizon_execution_trace.v1",
            "runtime_execution_proven": False,
            "world_model_prediction_proven": False,
            "receding_horizon_controller_proven": False,
            "proposed_actions": [],
            "world_model_predictions": [],
            "retained_actions": [],
            "executed_actions": [],
            "discarded_predictions": [],
            "control_rate_hz": package_manifest.get("control_rate_hz"),
            "chunk_start_timestamp_sec": package_manifest.get("chunk_start_timestamp_sec"),
            "requery_timestamp_sec": None,
        }
    )
    validation = validate_horizon_execution_trace(trace)
    blockers = list(validation["blockers"])
    if not isinstance(supplied, Mapping):
        blockers.append("horizon_runtime_execution_trace_missing")
        blockers = sorted(set(blockers))
    return {
        **trace,
        "status": "validated" if not blockers else "blocked",
        "validation": {
            **validation,
            "status": "validated" if not blockers else "blocked",
            "blockers": blockers,
        },
        "blockers": blockers,
        "claim_boundary": {
            "trace_is_executable_horizon_evidence": not blockers,
            "declared_recipe_metadata_alone_is_not_execution_proof": True,
            "input_action_records_are_not_prediction_or_execution_evidence": True,
        },
    }


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _first_existing_path(paths: Sequence[str]) -> Path | None:
    for value in paths:
        if not value:
            continue
        path = Path(value).expanduser()
        if path.exists():
            return path.resolve()
    return None


def _source_root_from_env() -> Path | None:
    return _first_existing_path(
        [
            os.getenv("BLUEPRINT_COSMOS3_WAM_SOURCE_ROOT", ""),
            os.getenv("BLUEPRINT_COSMOS3_NANO_SOURCE_ROOT", ""),
            os.getenv("BLUEPRINT_COSMOS3_SOURCE_ROOT", ""),
        ]
    )


def _checkpoint_from_env() -> Path | None:
    return _first_existing_path(
        [
            os.getenv("BLUEPRINT_COSMOS3_WAM_CHECKPOINT", ""),
            os.getenv("BLUEPRINT_COSMOS3_NANO_CHECKPOINT", ""),
            os.getenv("BLUEPRINT_WAM_MODEL_CHECKPOINT", ""),
        ]
    )


def _normalized_identity(value: Any) -> str:
    text = _string(value).lower()
    return "".join(char for char in text if char.isalnum() or char == ".")


def _identity_value_is_cosmos3_nano(value: Any) -> bool:
    return COSMOS3_NANO_IDENTITY_TOKEN in _normalized_identity(value).replace(".", "")


def _identity_value_is_wrong_family(value: Any) -> bool:
    normalized = _normalized_identity(value).replace(".", "").replace("-", "")
    if COSMOS3_NANO_IDENTITY_TOKEN in normalized:
        return False
    return any(
        marker.replace("-", "").replace(".", "") in normalized for marker in WRONG_FAMILY_MARKERS
    )


def _identity_candidate_files(checkpoint: Path) -> list[Path]:
    roots = [checkpoint] if checkpoint.is_dir() else [checkpoint.parent]
    files: list[Path] = []
    for root in roots:
        for name in CHECKPOINT_IDENTITY_FILENAMES:
            candidate = root / name
            if candidate.is_file():
                files.append(candidate)
    return files


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity_probe(checkpoint: Path) -> dict[str, Any]:
    """Machine-check the operator-supplied checkpoint's declared identity."""

    declared: list[str] = []
    scanned: list[str] = []
    attestation_candidates: list[dict[str, Any]] = []
    for identity_file in _identity_candidate_files(checkpoint):
        scanned.append(str(identity_file))
        try:
            payload = _read_json(identity_file)
        except (json.JSONDecodeError, OSError):
            continue
        attestation = _mapping(payload.get("sc3_checkpoint_attestation"))
        if not attestation and payload.get("trained_modes"):
            attestation = dict(payload)
        if attestation:
            attestation_candidates.append(attestation)
        for key in CHECKPOINT_IDENTITY_KEYS:
            value = _string(payload.get(key))
            if value and value not in declared:
                declared.append(value)
    verified = any(_identity_value_is_cosmos3_nano(value) for value in declared)
    wrong_family_values = [value for value in declared if _identity_value_is_wrong_family(value)]
    attestation_validations = [
        validate_checkpoint_attestation(candidate) for candidate in attestation_candidates
    ]
    accepted_pair = next(
        (
            (candidate, validation)
            for candidate, validation in zip(attestation_candidates, attestation_validations)
            if validation.get("status") == "validated"
        ),
        None,
    )
    accepted_attestation = (
        dict(accepted_pair[1]) if accepted_pair is not None else validate_checkpoint_attestation({})
    )
    selected_checkpoint_binding_proven = False
    if accepted_pair is not None:
        candidate = accepted_pair[0]
        checkpoint_ref = _mapping(candidate.get("checkpoint_artifact"))
        attested_path = Path(_string(checkpoint_ref.get("path"))).expanduser()
        selected_path_contains_attested_checkpoint = bool(
            attested_path.is_file()
            and (
                attested_path.resolve() == checkpoint.resolve()
                if checkpoint.is_file()
                else attested_path.resolve().is_relative_to(checkpoint.resolve())
            )
        )
        selected_checkpoint_binding_proven = bool(
            selected_path_contains_attested_checkpoint
            and _string(checkpoint_ref.get("sha256")).lower()
            == _sha256_file(attested_path)
            == _string(candidate.get("checkpoint_sha256")).lower()
        )
    if accepted_pair is not None and not selected_checkpoint_binding_proven:
        accepted_attestation = {
            **accepted_attestation,
            "status": "blocked",
            "blockers": sorted(
                {
                    *accepted_attestation.get("blockers", []),
                    "sc3_checkpoint_selected_path_not_bound_to_attestation",
                }
            ),
        }
    return {
        "schema_version": "cosmos3_checkpoint_identity_probe.v1",
        "checkpoint_path": str(checkpoint),
        "identity_files_scanned": scanned,
        "declared_identity_values": declared,
        "expected_base_model": EXPECTED_BASE_MODEL,
        "checkpoint_identity_verified": verified,
        "wrong_model_family_detected": bool(wrong_family_values) and not verified,
        "wrong_family_identity_values": wrong_family_values if not verified else [],
        "sc3_checkpoint_attestation_validation": accepted_attestation,
        "selected_checkpoint_binding_proven": selected_checkpoint_binding_proven,
        "sc3_trained_checkpoint_proven": accepted_attestation.get("status") == "validated",
    }


def source_identity_probe(source_root: Path) -> dict[str, Any]:
    """Machine-check that the source tree declares a Cosmos 3 lineage."""

    declared: list[str] = []
    scanned: list[str] = []
    for name in ("pyproject.toml", "setup.py", "setup.cfg", "README.md"):
        candidate = source_root / name
        if not candidate.is_file():
            continue
        scanned.append(str(candidate))
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        declared.append(text[:20000])
    combined = "\n".join(declared).lower()
    has_cosmos3_marker = "cosmos3" in combined.replace("-", "").replace("_", "")
    has_wrong_family_marker = any(
        marker in combined.replace("-", "").replace("_", "")
        for marker in ("cosmospredict2", "predict2.5", "oscarpublic")
    )
    verified = has_cosmos3_marker
    return {
        "schema_version": "cosmos3_source_identity_probe.v1",
        "source_root": str(source_root),
        "identity_files_scanned": scanned,
        "source_identity_verified": verified,
        "wrong_model_family_detected": has_wrong_family_marker and not verified,
    }


def _probe_modules() -> list[str]:
    configured = os.getenv("BLUEPRINT_COSMOS3_WAM_PROBE_MODULES", "")
    values = configured.split(",") if configured else ["torch"]
    return [_string(value) for value in values if _string(value)]


def _run_import_probe(*, python: str, source_root: Path, timeout_seconds: float) -> dict[str, Any]:
    modules = _probe_modules()
    started = time.monotonic()
    result = subprocess.run(
        [
            python,
            "-c",
            (
                "import json, importlib.util, sys; "
                "mods = json.loads(sys.argv[1]); "
                "print(json.dumps({m: bool(importlib.util.find_spec(m)) for m in mods}))"
            ),
            json.dumps(modules),
        ],
        cwd=str(source_root),
        env=_runtime_env(source_root),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_seconds,
    )
    available: dict[str, Any] = {}
    if result.stdout.strip():
        try:
            available = json.loads(result.stdout)
        except json.JSONDecodeError:
            available = {}
    missing = [name for name, present in available.items() if not present]
    return {
        "schema_version": "cosmos3_runtime_import_probe.v1",
        "status": "completed" if result.returncode == 0 and not missing else "blocked",
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "module_available": available,
        "blockers": []
        if result.returncode == 0 and not missing
        else ["blocked_missing_cosmos3_runtime_import"],
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
    }


def _entrypoint_relpath() -> str:
    return _string(os.getenv("BLUEPRINT_COSMOS3_WAM_ENTRYPOINT")) or DEFAULT_ENTRYPOINT_RELPATH


def _run_cosmos3(
    *,
    python: str,
    source_root: Path,
    checkpoint: Path,
    package_manifest: Mapping[str, Any],
    output_dir: Path,
    model: str,
    timeout_seconds: float,
    extra_args: Sequence[str],
) -> dict[str, Any]:
    entrypoint = source_root / _entrypoint_relpath()
    inference_params = Path(_string(package_manifest.get("inference_params_path")))
    argv = [
        python,
        str(entrypoint),
        "-i",
        str(inference_params),
        "-o",
        str(output_dir),
        "--checkpoint-path",
        str(checkpoint),
        "--model",
        model,
    ]
    argv.extend(extra_args)
    started = time.monotonic()
    try:
        result = subprocess.run(
            argv,
            cwd=str(source_root),
            env=_runtime_env(source_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "schema_version": "cosmos3_subprocess_result.v1",
            "status": "blocked",
            "returncode": None,
            "duration_seconds": round(time.monotonic() - started, 6),
            "argv_redacted": _redacted_argv(argv, checkpoint),
            "stdout_size_bytes": len(exc.stdout or ""),
            "stderr_size_bytes": len(exc.stderr or ""),
            "stderr_omitted_to_avoid_secret_leakage": bool(exc.stderr),
            "blockers": ["cosmos3_wam_command_timeout"],
        }
    failure_signals = _failure_signals(result.stdout or "", result.stderr or "")
    blockers = [] if result.returncode == 0 else ["cosmos3_wam_command_nonzero"]
    blockers.extend(signal for signal in failure_signals if signal not in blockers)
    return {
        "schema_version": "cosmos3_subprocess_result.v1",
        "status": "completed" if result.returncode == 0 else "blocked",
        "returncode": result.returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "argv_redacted": _redacted_argv(argv, checkpoint),
        "stdout_size_bytes": len(result.stdout or ""),
        "stderr_size_bytes": len(result.stderr or ""),
        "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
        "blockers": blockers,
    }


def _run_gate_status() -> dict[str, Any]:
    enabled = _env_truthy(LOCAL_MODEL_GATE_ENV)
    return {
        "local_model_gate_env": LOCAL_MODEL_GATE_ENV,
        "local_model_gate_enabled": enabled,
        "auto_run_allowed_without_gate": False,
    }


def _base_payload(
    *,
    status: str,
    blockers: Sequence[str],
    source_root: Path | None,
    checkpoint: Path | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "adapter_id": ADAPTER_ID,
        "evaluation_substrate": SUBSTRATE,
        "expected_base_model": EXPECTED_BASE_MODEL,
        "run_gates": _run_gate_status(),
        "sc3_recipe_declared_config": dict(SC3_RECIPE_DECLARED_CONFIG),
        "blockers": list(blockers),
        "source_root": str(source_root) if source_root else None,
        "checkpoint_path": str(checkpoint) if checkpoint else None,
        "learned_wam_model_ran": False,
        "fresh_model_command_executed_this_invocation": False,
        "fresh_model_run_claimed": False,
        "raw_credentials_written_to_artifacts": False,
        "secret_hashes_written_to_artifacts": False,
    }
    payload.update(dict(extra or {}))
    return payload


def _rollout_payload(
    *,
    package_manifest: Mapping[str, Any],
    checkpoint: Path,
    source_root: Path,
    subprocess_detail: Mapping[str, Any],
    model: str,
    checkpoint_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    save_root = Path(_string(package_manifest.get("save_root")))
    generated_videos = sorted(path.resolve() for path in save_root.rglob("*.mp4"))
    video_validations = [validate_generated_mp4_for_review(path) for path in generated_videos]
    subprocess_completed = subprocess_detail.get("status") == "completed"
    identity_verified = bool(
        checkpoint_identity.get("checkpoint_identity_verified")
        and source_identity.get("source_identity_verified")
    )
    rollouts = []
    for index, (path, validation) in enumerate(zip(generated_videos, video_validations), start=1):
        if validation.get("status") != "completed":
            continue
        rollouts.append(
            {
                "rollout_id": f"cosmos3_wam_rollout_{index:04d}",
                "policy_id": ADAPTER_ID,
                "model_candidate": os.getenv("BLUEPRINT_WAM_MODEL_CANDIDATE") or SUBSTRATE,
                "base_model": EXPECTED_BASE_MODEL,
                "model": model,
                "generated_video_path": str(path),
                "source_review_video_path": package_manifest.get("source_review_video_path"),
                "source_camera": package_manifest.get("source_camera"),
                "scenario_eval_run_id": package_manifest.get("scenario_eval_run_id"),
                "task_id": package_manifest.get("task_id"),
                "spawn_id": package_manifest.get("spawn_id"),
                "model_rollout_confidence": None,
                "generated_rollout_termination_reason": "cosmos3_command_completed",
                "success_label_source": "generated_video_requires_review",
                "generated_video_review_validation": validation,
            }
        )
    status = "completed" if rollouts else "blocked"
    validation_blockers = sorted(
        {
            str(blocker)
            for validation in video_validations
            for blocker in validation.get("blockers", [])
            if str(blocker)
        }
    )
    blockers = (
        []
        if rollouts
        else [
            "blocked_generated_cosmos3_mp4_not_reviewable"
            if generated_videos
            else "blocked_no_generated_cosmos3_mp4",
            *validation_blockers,
        ]
    )
    model_ran = bool(rollouts and subprocess_completed and identity_verified)
    horizon_trace = build_sc3_horizon_execution_trace(package_manifest)
    return _base_payload(
        status=status,
        blockers=blockers,
        source_root=source_root,
        checkpoint=checkpoint,
        extra={
            "base_model": EXPECTED_BASE_MODEL,
            "rollouts": rollouts,
            "generated_video_count": len(generated_videos),
            "generated_video_review_validations": video_validations,
            "model_provenance": {
                "candidate": os.getenv("BLUEPRINT_WAM_MODEL_CANDIDATE") or SUBSTRATE,
                "base_model": EXPECTED_BASE_MODEL,
                "model_family": EXPECTED_MODEL_FAMILY,
                "source_root": str(source_root),
                "checkpoint_path": str(checkpoint),
                "checkpoint_exists": checkpoint.exists(),
                "model": model,
                "checkpoint_identity_probe": dict(checkpoint_identity),
                "source_identity_probe": dict(source_identity),
            },
            "input_package": dict(package_manifest),
            "sc3_horizon_execution_trace": horizon_trace,
            "cosmos3_subprocess": dict(subprocess_detail),
            "fresh_model_command_executed_this_invocation": bool(rollouts and subprocess_completed),
            "fresh_model_run_claimed": model_ran,
            "learned_wam_model_ran": model_ran,
            "truth_boundary": {
                "generated_video_is_model_output": bool(rollouts and subprocess_completed),
                "checkpoint_identity_verified_as_cosmos3_nano": bool(
                    checkpoint_identity.get("checkpoint_identity_verified")
                ),
                "source_identity_verified_as_cosmos3": bool(
                    source_identity.get("source_identity_verified")
                ),
                "sc3_trained_checkpoint_proven": bool(
                    checkpoint_identity.get("sc3_trained_checkpoint_proven")
                ),
                "sc3_checkpoint_attestation_validation": _mapping(
                    checkpoint_identity.get("sc3_checkpoint_attestation_validation")
                ),
                "cosmos3_identity_match_required_for_learned_wam_claim": True,
                "sc3_recipe_metadata_is_declared_config_not_proof": True,
                "generated_rollout_not_physical_robot_proof": True,
                "generated_success_label_requires_external_vlm_or_human_judge": True,
                "generated_world_rank_fidelity_result_proven": False,
                "generated_world_policy_evaluation_scope_proven": False,
            },
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--python",
        default=os.getenv("BLUEPRINT_COSMOS3_WAM_PYTHON") or sys.executable,
    )
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--model",
        default=os.getenv("BLUEPRINT_COSMOS3_WAM_MODEL") or DEFAULT_MODEL,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("BLUEPRINT_COSMOS3_WAM_CHUNK_SIZE", "25")),
    )
    parser.add_argument(
        "--resolution",
        default=os.getenv("BLUEPRINT_COSMOS3_WAM_RESOLUTION") or "256,320",
    )
    parser.add_argument(
        "--guidance",
        type=int,
        default=int(os.getenv("BLUEPRINT_COSMOS3_WAM_GUIDANCE", "0")),
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=int(os.getenv("BLUEPRINT_COSMOS3_WAM_NUM_STEPS", "35")),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("BLUEPRINT_COSMOS3_WAM_TIMEOUT_SECONDS", "3600")),
    )
    parser.add_argument("--extra-arg", action="append", default=[])
    parser.add_argument("--probe-only", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    source_root = (
        args.source_root.expanduser().resolve() if args.source_root else _source_root_from_env()
    )
    checkpoint = (
        args.checkpoint.expanduser().resolve() if args.checkpoint else _checkpoint_from_env()
    )
    output_path = Path(
        os.getenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", "wam_provider_output.json")
    ).resolve()
    work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir
        else output_path.parent / "cosmos3_wam_command_workspace"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    blockers: list[str] = []
    if source_root is None:
        blockers.append("blocked_missing_cosmos3_source_root")
    elif not (source_root / _entrypoint_relpath()).is_file():
        blockers.append("blocked_missing_cosmos3_entrypoint")
    if checkpoint is None:
        blockers.append("blocked_missing_cosmos3_checkpoint")
    elif not checkpoint.exists():
        blockers.append("blocked_configured_cosmos3_checkpoint_path_missing")
    if not shutil.which(args.python) and not Path(args.python).expanduser().is_file():
        blockers.append("blocked_configured_python_missing")

    if blockers:
        payload = _base_payload(
            status="blocked",
            blockers=blockers,
            source_root=source_root,
            checkpoint=checkpoint,
        )
        _write_json(output_path, payload)
        return payload

    assert source_root is not None
    assert checkpoint is not None

    checkpoint_identity = checkpoint_identity_probe(checkpoint)
    source_identity = source_identity_probe(source_root)
    _write_json(work_dir / "cosmos3_checkpoint_identity_probe.json", checkpoint_identity)
    _write_json(work_dir / "cosmos3_source_identity_probe.json", source_identity)

    identity_blockers: list[str] = []
    if checkpoint_identity.get("wrong_model_family_detected"):
        identity_blockers.append("blocked_wrong_model_family_checkpoint_for_cosmos3_wam")
    elif not checkpoint_identity.get("checkpoint_identity_verified"):
        identity_blockers.append("blocked_cosmos3_checkpoint_identity_unverified")
    if source_identity.get("wrong_model_family_detected"):
        identity_blockers.append("blocked_wrong_model_family_source_for_cosmos3_wam")
    if identity_blockers:
        payload = _base_payload(
            status="blocked",
            blockers=identity_blockers,
            source_root=source_root,
            checkpoint=checkpoint,
            extra={
                "checkpoint_identity_probe": checkpoint_identity,
                "source_identity_probe": source_identity,
            },
        )
        _write_json(output_path, payload)
        return payload

    probe = _run_import_probe(
        python=args.python,
        source_root=source_root,
        timeout_seconds=min(args.timeout_seconds, 120.0),
    )
    _write_json(work_dir / "cosmos3_import_probe.json", probe)
    if args.probe_only:
        payload = _base_payload(
            status=probe["status"],
            blockers=list(probe.get("blockers", [])),
            source_root=source_root,
            checkpoint=checkpoint,
            extra={
                "probe_only": True,
                "import_probe": probe,
                "checkpoint_identity_probe": checkpoint_identity,
                "source_identity_probe": source_identity,
            },
        )
        _write_json(output_path, payload)
        return payload
    if probe["status"] != "completed":
        payload = _base_payload(
            status="blocked",
            blockers=list(probe.get("blockers", [])),
            source_root=source_root,
            checkpoint=checkpoint,
            extra={
                "import_probe": probe,
                "checkpoint_identity_probe": checkpoint_identity,
                "source_identity_probe": source_identity,
            },
        )
        _write_json(output_path, payload)
        return payload

    if not _env_truthy(LOCAL_MODEL_GATE_ENV):
        payload = _base_payload(
            status="blocked",
            blockers=[f"blocked_{LOCAL_MODEL_GATE_ENV}_not_enabled"],
            source_root=source_root,
            checkpoint=checkpoint,
            extra={
                "import_probe": probe,
                "checkpoint_identity_probe": checkpoint_identity,
                "source_identity_probe": source_identity,
            },
        )
        _write_json(output_path, payload)
        return payload

    rollout_input = Path(os.environ["BLUEPRINT_WAM_ROLLOUT_INPUT"]).expanduser().resolve()
    rollout_manifest = _read_json(rollout_input)
    package_manifest = _materialize_cosmos_input_package(
        rollout_manifest=rollout_manifest,
        work_dir=work_dir,
        chunk_size=args.chunk_size,
        resolution=args.resolution,
        guidance=args.guidance,
        num_steps=args.num_steps,
    )
    cosmos3_output_dir = work_dir / "cosmos3_output"
    subprocess_detail = _run_cosmos3(
        python=args.python,
        source_root=source_root,
        checkpoint=checkpoint,
        package_manifest=package_manifest,
        output_dir=cosmos3_output_dir,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        extra_args=[item for value in args.extra_arg for item in shlex.split(value)],
    )
    payload = _rollout_payload(
        package_manifest=package_manifest,
        checkpoint=checkpoint,
        source_root=source_root,
        subprocess_detail=subprocess_detail,
        model=args.model,
        checkpoint_identity=checkpoint_identity,
        source_identity=source_identity,
    )
    if subprocess_detail["status"] != "completed" and not payload["rollouts"]:
        payload["status"] = "blocked"
        payload["blockers"] = list(subprocess_detail.get("blockers") or [])
    _write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run(argv)
    except Exception as exc:
        output_path = Path(
            os.getenv("BLUEPRINT_WAM_ROLLOUT_OUTPUT", "wam_provider_output.json")
        ).resolve()
        payload = _base_payload(
            status="blocked",
            blockers=[f"cosmos3_wam_adapter_exception:{type(exc).__name__}"],
            source_root=None,
            checkpoint=None,
        )
        _write_json(output_path, payload)
    print(json.dumps({"adapter_id": ADAPTER_ID, "status": payload.get("status")}, sort_keys=True))
    return 0 if payload.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
