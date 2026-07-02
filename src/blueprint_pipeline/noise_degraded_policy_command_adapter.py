"""Noise-degraded policy command adapter for ranker-validation ladders.

The adapter wraps an inner policy command adapter (for example the GR00T
N1.7 + SONIC adapter), runs it unchanged on the same observation packet, and
then injects deterministic seeded Gaussian noise into the numeric action
values it returns. The result is a synthetic "worse" variant of the inner
policy with a known ordering relative to the clean policy, which is what the
evaluation-ranker validation ladder needs.

The degraded variant never fabricates an action when the inner adapter is
blocked, and its claim boundary marks the output as a synthetic degradation
for ranker validation only — it is not the behavior of any real checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "noise_degraded_policy_command_adapter.v1"
POLICY_ID_PREFIX = "noise_degraded"
INNER_COMMAND_ENV = "BLUEPRINT_NOISE_DEGRADED_INNER_COMMAND"
AMPLITUDE_ENV = "BLUEPRINT_NOISE_DEGRADED_AMPLITUDE"
SEED_ENV = "BLUEPRINT_NOISE_DEGRADED_SEED"
POLICY_ID_ENV = "BLUEPRINT_NOISE_DEGRADED_POLICY_ID"
DEFAULT_SEED = 1337
DEFAULT_TIMEOUT_SECONDS = 180.0


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _read_payload() -> dict[str, Any]:
    input_path = os.getenv("BLUEPRINT_POLICY_ACTION_INPUT", "").strip()
    if input_path:
        payload = json.loads(Path(input_path).expanduser().read_text(encoding="utf-8"))
    else:
        raw = sys.stdin.read().strip()
        payload = json.loads(raw) if raw else {}
    if not isinstance(payload, Mapping):
        raise ValueError("policy input must be a JSON object")
    return dict(payload)


def _write_payload(payload: Mapping[str, Any]) -> None:
    output_path = os.getenv("BLUEPRINT_POLICY_ACTION_OUTPUT", "").strip()
    encoded = json.dumps(dict(payload), sort_keys=True)
    if output_path:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


def _observation(payload: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("observation"), Mapping):
        return dict(payload["observation"])  # type: ignore[index]
    return dict(payload)


def amplitude_label(amplitude: float) -> str:
    return f"{amplitude:g}".replace("-", "m").replace(".", "p")


def noise_degraded_policy_id(inner_policy_id: str, amplitude: float) -> str:
    base = _string(inner_policy_id) or "policy"
    return f"{base}_noise_{amplitude_label(amplitude)}"


def derive_noise_rng_seed(
    *,
    seed: int,
    amplitude: float,
    observation: Mapping[str, Any],
) -> int:
    visual = _mapping(observation.get("visual_observation"))
    identity = {
        "seed": int(seed),
        "amplitude": float(amplitude),
        "task_id": _string(observation.get("task_id")),
        "scenario_id": _string(observation.get("scenario_id")),
        "scenario_eval_run_id": _string(observation.get("scenario_eval_run_id")),
        "scenario_variation_instance_id": _string(
            observation.get("scenario_variation_instance_id")
        ),
        "step_index": _string(observation.get("step_index")),
        "camera_frame_path": _string(
            visual.get("camera_frame_path") or observation.get("camera_frame_path")
        ),
    }
    digest = sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _perturb(value: Any, rng: "_DeterministicGaussian", amplitude: float, *, in_sequence: bool) -> tuple[Any, int]:
    if isinstance(value, bool):
        return value, 0
    if isinstance(value, (int, float)):
        if not in_sequence:
            return value, 0
        return float(value) + amplitude * rng.gauss(), 1
    if isinstance(value, Mapping):
        perturbed: dict[str, Any] = {}
        count = 0
        for key in sorted(value, key=str):
            child, child_count = _perturb(value[key], rng, amplitude, in_sequence=False)
            perturbed[str(key)] = child
            count += child_count
        return perturbed, count
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        perturbed_list = []
        count = 0
        for item in value:
            child, child_count = _perturb(item, rng, amplitude, in_sequence=True)
            perturbed_list.append(child)
            count += child_count
        return perturbed_list, count
    return value, 0


class _DeterministicGaussian:
    """Box-Muller gaussian over a SHA-256 counter stream.

    Avoids the `random` module so the noise stream is stable across Python
    versions and platforms for the same seed.
    """

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self._counter = 0
        self._spare: float | None = None

    def _uniform(self) -> float:
        digest = sha256(f"{self._seed}:{self._counter}".encode("utf-8")).hexdigest()
        self._counter += 1
        return (int(digest[:13], 16) + 0.5) / float(1 << 52)

    def gauss(self) -> float:
        import math

        if self._spare is not None:
            value = self._spare
            self._spare = None
            return value
        u1 = self._uniform()
        u2 = self._uniform()
        radius = math.sqrt(-2.0 * math.log(u1))
        self._spare = radius * math.sin(2.0 * math.pi * u2)
        return radius * math.cos(2.0 * math.pi * u2)


def perturb_action(
    action: Mapping[str, Any],
    *,
    amplitude: float,
    rng_seed: int,
) -> tuple[dict[str, Any], int]:
    """Perturb numeric values inside sequences of the action payload.

    Scalar metadata fields directly on mappings (counts, flags, ids) are left
    untouched; only numbers inside lists — action chunks, joint targets,
    latent vectors — receive noise.
    """
    rng = _DeterministicGaussian(rng_seed)
    perturbed, count = _perturb(dict(action), rng, float(amplitude), in_sequence=False)
    return _mapping(perturbed), count


def _claim_boundary(
    *,
    inner_claim_boundary: Mapping[str, Any],
    inner_command_ran: bool,
    action_values_perturbed: bool,
) -> dict[str, Any]:
    return {
        **_mapping(inner_claim_boundary),
        "noise_degraded_policy_command_ran": bool(inner_command_ran),
        "synthetic_noise_degradation_injected": bool(action_values_perturbed),
        "degraded_variant_for_ranker_validation_only": True,
        "degraded_action_is_not_policy_checkpoint_behavior": True,
        "known_ordering_ladder_member": True,
        "generated_world_rank_fidelity_result_proven": False,
        "non_ranking_operational_claim_proven": False,
    }


def _blocked_payload(
    *,
    blockers: Sequence[str],
    observation: Mapping[str, Any],
    inner_command: str | None,
    amplitude: float | None,
    seed: int,
    policy_id: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked",
        "policy_id": _string(policy_id) or _string(os.getenv(POLICY_ID_ENV)) or POLICY_ID_PREFIX,
        "inner_policy_id": None,
        "model_ran": False,
        "task_id": observation.get("task_id"),
        "blockers": sorted(set(blockers)),
        "inner_command_env": INNER_COMMAND_ENV,
        "inner_command_configured": bool(inner_command),
        "noise_injection": {
            "amplitude": amplitude,
            "seed": seed,
            "action_values_perturbed": False,
            "perturbed_value_count": 0,
        },
        "claim_boundary": _claim_boundary(
            inner_claim_boundary={},
            inner_command_ran=False,
            action_values_perturbed=False,
        ),
    }


def _run_inner_command(
    *,
    command: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="blueprint-noise-degraded-") as tmp:
        temp_dir = Path(tmp)
        input_path = temp_dir / "policy_input.json"
        output_path = temp_dir / "policy_output.json"
        input_path.write_text(json.dumps(dict(payload), sort_keys=True) + "\n", encoding="utf-8")
        env = {
            **os.environ,
            "BLUEPRINT_POLICY_ACTION_INPUT": str(input_path),
            "BLUEPRINT_POLICY_ACTION_OUTPUT": str(output_path),
        }
        result = subprocess.run(
            shlex.split(command),
            input=json.dumps(dict(payload)),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
            env=env,
        )
        meta = {
            "command_exit_code": result.returncode,
            "stderr_size_bytes": len(result.stderr or ""),
            "stderr_omitted_to_avoid_secret_leakage": bool(result.stderr),
            "stdout_size_bytes": len(result.stdout or ""),
            "subprocess_spawned": True,
            "policy_output_file_used": output_path.is_file(),
        }
        if output_path.is_file():
            value = json.loads(output_path.read_text(encoding="utf-8"))
        else:
            value = json.loads(result.stdout or "{}")
        if not isinstance(value, Mapping):
            raise RuntimeError("noise_degraded_inner_policy_output_not_json_object")
        return dict(value), meta


def run_noise_degraded_policy(
    *,
    payload: Mapping[str, Any],
    inner_command: str | None,
    amplitude: float | None,
    seed: int = DEFAULT_SEED,
    policy_id: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], int]:
    observation = _observation(payload)
    blockers: list[str] = []
    if not _string(inner_command):
        blockers.append(f"set_{INNER_COMMAND_ENV}_to_runnable_inner_policy_adapter_command")
    if amplitude is None:
        blockers.append(f"set_{AMPLITUDE_ENV}_to_nonnegative_noise_amplitude")
    elif float(amplitude) < 0.0:
        blockers.append("noise_amplitude_must_be_nonnegative")
    if blockers:
        return (
            _blocked_payload(
                blockers=blockers,
                observation=observation,
                inner_command=inner_command,
                amplitude=amplitude,
                seed=seed,
                policy_id=policy_id,
            ),
            2,
        )

    try:
        inner_payload, meta = _run_inner_command(
            command=_string(inner_command),
            payload={"observation": observation},
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        return (
            _blocked_payload(
                blockers=[f"blocked_noise_degraded_inner_command_failed:{type(exc).__name__}"],
                observation=observation,
                inner_command=inner_command,
                amplitude=amplitude,
                seed=seed,
                policy_id=policy_id,
            ),
            2,
        )

    inner_policy_id = _string(inner_payload.get("policy_id"))
    if inner_payload.get("status") != "completed" or not isinstance(
        inner_payload.get("action"), Mapping
    ):
        inner_blockers = [
            _string(item) for item in inner_payload.get("blockers", []) if _string(item)
        ] or ["blocked_noise_degraded_inner_command_returned_no_action"]
        return (
            _blocked_payload(
                blockers=[f"inner:{blocker}" for blocker in inner_blockers],
                observation=observation,
                inner_command=inner_command,
                amplitude=amplitude,
                seed=seed,
                policy_id=policy_id,
            )
            | {
                "inner_policy_id": inner_policy_id or None,
                "inner_runner_metadata": meta,
            },
            2,
        )

    effective_amplitude = float(amplitude if amplitude is not None else 0.0)
    rng_seed = derive_noise_rng_seed(
        seed=seed,
        amplitude=effective_amplitude,
        observation=observation,
    )
    action, perturbed_count = perturb_action(
        _mapping(inner_payload.get("action")),
        amplitude=effective_amplitude,
        rng_seed=rng_seed,
    )
    action_values_perturbed = bool(perturbed_count and effective_amplitude > 0.0)
    action["noise_injected"] = action_values_perturbed
    action["noise_amplitude"] = effective_amplitude
    effective_policy_id = (
        _string(policy_id)
        or noise_degraded_policy_id(inner_policy_id or POLICY_ID_PREFIX, effective_amplitude)
    )
    return (
        {
            "schema_version": SCHEMA_VERSION,
            "status": "completed",
            "policy_id": effective_policy_id,
            "selected_candidate_id": effective_policy_id,
            "policy_kind": "noise_degraded_ranker_validation_variant",
            "inner_policy_id": inner_policy_id or None,
            "inner_schema_version": _string(inner_payload.get("schema_version")) or None,
            "model_ran": bool(inner_payload.get("model_ran")),
            "inner_policy_command_ran": True,
            "task_id": observation.get("task_id"),
            "action": action,
            "noise_injection": {
                "amplitude": effective_amplitude,
                "seed": int(seed),
                "rng_seed": rng_seed,
                "rng_basis": "sha256_counter_box_muller",
                "perturbed_value_count": perturbed_count,
                "action_values_perturbed": action_values_perturbed,
            },
            "inner_runner_metadata": meta,
            "inner_claim_boundary": _mapping(inner_payload.get("claim_boundary")),
            "adapter_metadata": {
                "adapter_family": POLICY_ID_PREFIX,
                "wraps_inner_policy_adapter": True,
                "deterministic_given_seed_and_observation": True,
            },
            "claim_boundary": _claim_boundary(
                inner_claim_boundary=_mapping(inner_payload.get("claim_boundary")),
                inner_command_ran=True,
                action_values_perturbed=action_values_perturbed,
            ),
        },
        0,
    )


def adapter_manifest() -> dict[str, Any]:
    return {
        "schema_version": "policy_command_adapter_manifest.v1",
        "policy_id": POLICY_ID_PREFIX,
        "adapter_family": POLICY_ID_PREFIX,
        "wraps_inner_policy_adapter": True,
        "reads_json_from_stdin": True,
        "also_reads_BLUEPRINT_POLICY_ACTION_INPUT": True,
        "writes_json_to_stdout": True,
        "also_writes_BLUEPRINT_POLICY_ACTION_OUTPUT": True,
        "required_env": [INNER_COMMAND_ENV, AMPLITUDE_ENV],
        "optional_env": [SEED_ENV, POLICY_ID_ENV],
        "claim_boundary": _claim_boundary(
            inner_claim_boundary={},
            inner_command_ran=False,
            action_values_perturbed=False,
        ),
    }


def _float_or_none(value: Any) -> float | None:
    text = _string(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner-command")
    parser.add_argument("--noise-amplitude", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--policy-id")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--print-manifest", action="store_true")
    args = parser.parse_args(argv)
    if args.print_manifest:
        _write_payload(adapter_manifest())
        return 0
    amplitude = (
        args.noise_amplitude
        if args.noise_amplitude is not None
        else _float_or_none(os.getenv(AMPLITUDE_ENV))
    )
    seed_env = _string(os.getenv(SEED_ENV))
    seed = args.seed if args.seed is not None else int(seed_env) if seed_env.isdigit() else DEFAULT_SEED
    response, exit_code = run_noise_degraded_policy(
        payload=_read_payload(),
        inner_command=args.inner_command or os.getenv(INNER_COMMAND_ENV),
        amplitude=amplitude,
        seed=seed,
        policy_id=args.policy_id or os.getenv(POLICY_ID_ENV),
        timeout_seconds=args.timeout_seconds,
    )
    _write_payload(response)
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
