"""Provider-side Cosmos3-Nano Reasoner evaluator diagnostic runtime.

This file is copied verbatim into a paid-provider bundle. It uses only the
reasoner surface and never invokes Cosmos video generation or a robot endpoint.
All results are post-unseal diagnostics and receive no confirmatory credit.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping


MODEL = "nvidia/Cosmos3-Nano"
REVISION = "411f42a8fdfb8c5b2583cb8786e0938f49796eaa"
NATIVE_REASONER_ARCHITECTURE = "Cosmos3ForConditionalGeneration"
MODEL_CONFIG_SHA256 = "c32f2468a54542c21946bc8eab6172b911dcec9a7193a94c023ea2d4073bcda6"
CLAIM_CLASS = "post_unseal_diagnostic_only"
PORT = 8000
EXPECTED_KEYS = {
    "preferred_episode",
    "episode_a_progress_0_to_5",
    "episode_b_progress_0_to_5",
    "stable_success_a",
    "stable_success_b",
    "comparison_confidence",
    "uncertainty",
    "decisive_evidence",
    "artifact_flags_a",
    "artifact_flags_b",
    "abstention_factors",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected_json_object:{path.name}")
    return dict(value)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _http_json(
    method: str, url: str, payload: Mapping[str, Any] | None = None, timeout: int = 60
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(dict(payload)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(  # nosec B310 - fixed loopback reasoner endpoint
        request, timeout=timeout
    ) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("provider_response_not_object")
    return dict(value)


def _wait_for_server(
    process: subprocess.Popen[str], timeout_seconds: int = 3600
) -> tuple[dict[str, Any], float]:
    started = time.monotonic()
    last_error = "not_started"
    while time.monotonic() - started < timeout_seconds:
        if process.poll() is not None:
            raise RuntimeError(f"reasoner_server_exited:{process.returncode}")
        try:
            models = _http_json("GET", f"http://127.0.0.1:{PORT}/v1/models", timeout=10)
            if isinstance(models.get("data"), list) and models["data"]:
                return models, time.monotonic() - started
        except Exception as exc:  # noqa: BLE001
            last_error = type(exc).__name__
        time.sleep(5)
    raise TimeoutError(f"reasoner_server_start_timeout:{last_error}")


def _extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Reasoner responses may prefix a valid object with analysis or fences.
        parsed = None
    if isinstance(parsed, Mapping):
        return dict(parsed)
    decoder = json.JSONDecoder()
    valid: list[dict[str, Any]] = []
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            valid.append(dict(parsed))
    if not valid:
        raise ValueError("structured_json_object_missing")
    return valid[-1]


def _validate_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != EXPECTED_KEYS:
        raise ValueError("structured_output_keys_invalid")
    if value["preferred_episode"] not in {"A", "B", "tie", "abstain"}:
        raise ValueError("preferred_episode_invalid")
    for key in ("episode_a_progress_0_to_5", "episode_b_progress_0_to_5"):
        if type(value[key]) is not int or not 0 <= value[key] <= 5:
            raise ValueError(f"{key}_invalid")
    for key in ("stable_success_a", "stable_success_b"):
        if type(value[key]) is not bool:
            raise ValueError(f"{key}_invalid")
    for key in ("comparison_confidence", "uncertainty"):
        number = value[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not 0 <= float(number) <= 1
        ):
            raise ValueError(f"{key}_invalid")
    for key, maximum in (
        ("decisive_evidence", 4),
        ("artifact_flags_a", 6),
        ("artifact_flags_b", 6),
        ("abstention_factors", 4),
    ):
        items = value[key]
        if (
            not isinstance(items, list)
            or len(items) > maximum
            or any(not isinstance(item, str) for item in items)
        ):
            raise ValueError(f"{key}_invalid")
    return dict(value)


def _message_content(response: Mapping[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("reasoner_choices_missing")
    message = choices[0].get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise ValueError("reasoner_message_content_missing")
    return str(message["content"])


def _request_payload(
    row: Mapping[str, Any], bundle_root: Path, served_model: str
) -> dict[str, Any]:
    video_a = (bundle_root / str(row["episode_a_video"])).resolve().as_uri()
    video_b = (bundle_root / str(row["episode_b_video"])).resolve().as_uri()
    prompt = (
        str(row["prompt"])
        + "\n"
        + json.dumps(
            {
                "task_instruction": row["task_instruction"],
                "episode_a": "first complete chronological generated-only video",
                "episode_b": "second complete chronological generated-only video",
                "claim_boundary": "generated_episode_pair_diagnostic_not_physical_success",
            },
            sort_keys=True,
        )
    )
    return {
        "model": served_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "EPISODE A"},
                    {"type": "video_url", "video_url": {"url": video_a}},
                    {"type": "text", "text": "EPISODE B"},
                    {"type": "video_url", "video_url": {"url": video_b}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0,
        "seed": 0,
    }


def _reasoner_server_command(bundle_root: Path) -> list[str]:
    """Build the pinned native-architecture Reasoner command.

    Cosmos3-Nano's frozen model config declares ``Cosmos3ForConditionalGeneration``.
    The serving runtime must honor that native declaration. Forcing the separate
    Cosmos Framework class name is incompatible with the pinned vLLM runtime.
    """

    return [
        "vllm",
        "serve",
        MODEL,
        "--revision",
        REVISION,
        "--tensor-parallel-size",
        "1",
        "--mm-encoder-tp-mode",
        "data",
        "--async-scheduling",
        "--allowed-local-media-path",
        str(bundle_root),
        "--media-io-kwargs",
        '{"video":{"num_frames":32}}',
        "--max-model-len",
        "131072",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.92",
        "--port",
        str(PORT),
    ]


def run() -> int:
    bundle_root = Path(os.environ["BLUEPRINT_EVALUATOR_PROVIDER_BUNDLE_DIR"]).resolve()
    output_dir = Path(os.environ["BLUEPRINT_EVALUATOR_PROVIDER_OUTPUT_DIR"]).resolve()
    manifest = _read_json(Path(os.environ["BLUEPRINT_EVALUATOR_INPUT"]).resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = output_dir / "reasoner_server.log"
    command = _reasoner_server_command(bundle_root)
    process: subprocess.Popen[str] | None = None
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    load_seconds = 0.0
    started_wall = time.time()
    try:
        with server_log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            models, load_seconds = _wait_for_server(process)
            served_model = str(models["data"][0].get("id") or MODEL)
            for row in manifest["pairs"]:
                pair_id = str(row["pair_id"])
                request_started = time.monotonic()
                raw_text = ""
                usage: dict[str, Any] = {}
                try:
                    response = _http_json(
                        "POST",
                        f"http://127.0.0.1:{PORT}/v1/chat/completions",
                        _request_payload(row, bundle_root, served_model),
                        timeout=1800,
                    )
                    usage = dict(response.get("usage") or {})
                    raw_text = _message_content(response)
                    structured = _validate_payload(_extract_json_object(raw_text))
                    results.append(
                        {
                            "schema_version": "policy_ranking_pair_result.v1",
                            "pair_id": pair_id,
                            "arm_id": "cosmos3_nano_reasoner",
                            "provider": "self_hosted_vast",
                            "model": MODEL,
                            "model_revision": REVISION,
                            "structured_response": structured,
                            "raw_response_text": raw_text,
                            "usage": usage,
                            "latency_seconds": time.monotonic() - request_started,
                            "transport": "vllm_openai_compatible_native_video",
                            "claim_class": CLAIM_CLASS,
                            "policy_identity_sent_to_evaluator": False,
                            "physical_outcome_sent_to_evaluator": False,
                            "physical_ground_truth_pixels_sent_to_evaluator": False,
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "pair_id": pair_id,
                            "error_type": type(exc).__name__,
                            "error": str(exc)[:500],
                            "raw_response_text": raw_text,
                            "usage": usage,
                            "latency_seconds": time.monotonic() - request_started,
                        }
                    )
    except Exception as exc:  # noqa: BLE001
        errors.append({"pair_id": None, "error_type": type(exc).__name__, "error": str(exc)[:500]})
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "pair_id": None,
                            "error_type": "ReasonerProcessCleanupError",
                            "error": f"{type(exc).__name__}:{str(exc)[:400]}",
                        }
                    )
    _write_json(output_dir / "pair_results.json", {"results": results, "errors": errors})
    runtime = {
        "schema_version": "policy_ranking_cosmos_reasoner_runtime.v1",
        "status": "completed"
        if len(results) == int(manifest["pair_count"]) and not errors
        else "blocked",
        "model": MODEL,
        "model_revision": REVISION,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "native_reasoner_architecture": NATIVE_REASONER_ARCHITECTURE,
        "architecture_selection": "native_frozen_model_config",
        "architecture_override_used": False,
        "surface": "reasoner_only_vllm",
        "claim_class": CLAIM_CLASS,
        "result_count": len(results),
        "error_count": len(errors),
        "model_load_seconds": load_seconds,
        "total_wall_seconds": time.time() - started_wall,
        "server_command": command,
        "server_log_path": server_log_path.name,
        "blockers": [] if not errors else ["one_or_more_reasoner_rows_failed"],
        "evaluator_runtime_executed": bool(results),
        "action_conditioned_video_rollout_generated": False,
        "physical_robot_endpoint_accessed": False,
        "policy_identity_sent_to_evaluator": False,
        "physical_outcome_sent_to_evaluator": False,
        "physical_ground_truth_pixels_sent_to_evaluator": False,
        "raw_secret_values_recorded": False,
    }
    _write_json(output_dir / "evaluator_runtime_result.json", runtime)
    print(
        "BLUEPRINT_COSMOS_REASONER_EVALUATOR_COMPLETED"
        if runtime["status"] == "completed"
        else "BLUEPRINT_COSMOS_REASONER_EVALUATOR_BLOCKED",
        flush=True,
    )
    return 0 if runtime["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(run())
