"""Stock RoboLab episode/inference orchestration for Blueprint's actual task.

RoboLab owns its normal per-step loop. The native environment generator owns
Blueprint's reconstructed scene, calibrated robot, actuator conversion,
scientific reset, exact policy-input evidence and independent success scorer.
Only policy observations cross the inference boundary. Benchmark scenes,
benchmark predicates and privileged ground truth are never substituted.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Generator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import zipfile

from .adp_articulated_task_success_contract import task_kind_of_contract, validate_task_success_contract
from .decision_evidence_contracts import canonical_digest
from .policy_episode_events import NativeStep, PolicyQuery
from .policy_interface_binding import TEAM_HTTPS_INTERFACES, resolve_policy_interface

BACKEND = "robolab.v1"
ROBOLAB_COMMIT = "b6cfcf6ccd2113c4cd1216d35b33024a7d5691da"
ROBOLAB_ARCHIVE_SHA256 = "sha256:16092baa43c5a2eb78f6dc8a330330e8e8bfc6a8cafce59efff560b26d6af278"

IMAGEIO_WHEEL = "imageio-2.37.2-py3-none-any.whl"
IMAGEIO_SHA256 = "sha256:ad9adfb20335d718c03de457358ed69f141021a333c40a53e57273d8a5bd0b9b"
ROBOLAB_MANIFEST_SHA256 = "sha256:b19532cef9431b39e586310cf169ba09145cacb570592ed0c143ce5649a8fd22"

def validate_robolab_policy_execution_spec(spec: Mapping[str, Any], *,
                                           candidate_id: str | None = None) -> dict[str, Any]:
    value = json.loads(json.dumps(spec, allow_nan=False))
    if (value.get("schema_version") != "native_task_arena_policy_canary_execution_spec.v1"
            or value.get("evaluation_backend") != BACKEND
            or (candidate_id is not None and value.get("candidate_id") != candidate_id)
            or value.get("execution_spec_digest") != canonical_digest(value, digest_field="execution_spec_digest")
            or not isinstance(value.get("task_success_contract"), dict)
            or value.get("task_success_contract_digest") != value["task_success_contract"].get("contract_digest")):
        raise ValueError("beta_robolab_execution_spec_binding_invalid")
    for key in ("checkpoint_digest", "runtime_identity_digest"):
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 71 or not digest.startswith("sha256:"):
            raise ValueError("beta_robolab_policy_identity_invalid")
        try:
            int(digest[7:], 16)
        except ValueError as exc:
            raise ValueError("beta_robolab_policy_identity_invalid") from exc
    validate_task_success_contract(value["task_success_contract"],
                                   task_kind=task_kind_of_contract(value["task_success_contract"]),
                                   require_confirmed=False)
    interface = resolve_policy_interface(value)
    if interface in TEAM_HTTPS_INTERFACES:
        endpoint = value.get("policy_endpoint")
        if (not isinstance(endpoint, dict) or set(endpoint) != {"credential_env"}
                or re.fullmatch(r"[A-Z][A-Z0-9_]{2,58}_FILE", str(endpoint.get("credential_env") or "")) is None):
            raise ValueError("beta_robolab_https_credential_binding_invalid")
    for field in ("max_policy_queries", "open_loop_horizon"):
        if type(value.get(field)) is not int or value[field] < 1:
            raise ValueError("beta_robolab_episode_limits_invalid")
    if not isinstance(value.get("prompt"), str) or not value["prompt"].strip():
        raise ValueError("beta_robolab_instruction_missing")
    return value


def stage_robolab_source(runtime: Path) -> dict[str, Any]:
    """Install exact upstream Python sources from the reviewed, offline archive."""
    vendor = Path(__file__).resolve().parent / "_vendor"
    archive_path = vendor / f"robolab-{ROBOLAB_COMMIT}.zip"
    raw = archive_path.read_bytes()
    if "sha256:" + hashlib.sha256(raw).hexdigest() != ROBOLAB_ARCHIVE_SHA256:
        raise ValueError("beta_robolab_source_archive_digest_mismatch")
    manifest_bytes = (vendor / "robolab-source.v1.json").read_bytes()
    if "sha256:" + hashlib.sha256(manifest_bytes).hexdigest() != ROBOLAB_MANIFEST_SHA256:
        raise ValueError("beta_robolab_source_manifest_digest_mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest["commit"] != ROBOLAB_COMMIT or manifest["archive_sha256"] != ROBOLAB_ARCHIVE_SHA256:
        raise ValueError("beta_robolab_source_manifest_mismatch")
    with zipfile.ZipFile(archive_path) as archive:
        for row in manifest["files"]:
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("beta_robolab_source_path_invalid")
            content = archive.read(row["path"])
            if "sha256:" + hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise ValueError("beta_robolab_source_file_digest_mismatch")
            target = runtime / (relative if relative.parts[0] == "robolab" else Path("robolab-LICENSE"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    wheel_bytes = (vendor / IMAGEIO_WHEEL).read_bytes()
    if "sha256:" + hashlib.sha256(wheel_bytes).hexdigest() != IMAGEIO_SHA256:
        raise ValueError("beta_robolab_imageio_wheel_digest_mismatch")
    (runtime / IMAGEIO_WHEEL).write_bytes(wheel_bytes)
    with zipfile.ZipFile(vendor / IMAGEIO_WHEEL) as wheel:
        for entry in wheel.infolist():
            relative = Path(entry.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("beta_robolab_imageio_path_invalid")
            if not entry.is_dir():
                target = runtime / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(wheel.read(entry))
    (runtime / "robolab-source.v1.json").write_bytes(manifest_bytes)
    return {"distribution": "robolab", "version": manifest["release_version"],
            "source_commit": ROBOLAB_COMMIT, "archive_sha256": ROBOLAB_ARCHIVE_SHA256,
            "files": manifest["files"],
            "python_dependencies": [{"distribution": "imageio", "version": "2.37.2",
                                     "wheel": IMAGEIO_WHEEL, "sha256": IMAGEIO_SHA256}]}


def verify_loaded_robolab() -> None:
    import robolab

    runtime = Path(robolab.__file__).resolve().parent.parent
    manifest_bytes = (runtime / "robolab-source.v1.json").read_bytes()
    if "sha256:" + hashlib.sha256(manifest_bytes).hexdigest() != ROBOLAB_MANIFEST_SHA256:
        raise ValueError("beta_robolab_loaded_manifest_digest_mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest.get("commit") != ROBOLAB_COMMIT or manifest.get("archive_sha256") != ROBOLAB_ARCHIVE_SHA256:
        raise ValueError("beta_robolab_loaded_source_unpinned")
    for row in manifest["files"]:
        if row["path"].startswith("robolab/"):
            content = (runtime / row["path"]).read_bytes()
            if "sha256:" + hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise ValueError("beta_robolab_loaded_source_digest_mismatch")

    wheel_path = runtime / IMAGEIO_WHEEL
    if "sha256:" + hashlib.sha256(wheel_path.read_bytes()).hexdigest() != IMAGEIO_SHA256:
        raise ValueError("beta_robolab_loaded_imageio_wheel_digest_mismatch")
    with zipfile.ZipFile(wheel_path) as wheel:
        for entry in wheel.infolist():
            if not entry.is_dir() and (runtime / entry.filename).read_bytes() != wheel.read(entry):
                raise ValueError("beta_robolab_loaded_imageio_source_digest_mismatch")


def preflight_robolab_runtime() -> dict[str, Any]:
    """Resolve real runtime imports after Isaac launch, before policy clients."""
    verify_loaded_robolab()
    from robolab.eval import InferenceClient, run_episode
    if not isinstance(InferenceClient, type) or not callable(run_episode):
        raise ValueError("beta_robolab_upstream_entrypoints_unavailable")
    return {"backend": BACKEND, "source_commit": ROBOLAB_COMMIT, "status": "passed"}


class NativeEpisodeEnvironment:
    """Single active native scene registered with RoboLab's environment protocol.

    Upstream's two startup resets acknowledge the one scientific reset already
    performed by the generator. They must not reset physics a second time.
    """
    num_envs = 1
    device = "cpu"
    recorder_manager = None

    def __init__(self, events: Generator, *, policy: Any, maximum_steps: int, action_dimension: int):
        self.events, self.policy = events, policy
        self.max_episode_length = maximum_steps
        self.action_manager = SimpleNamespace(total_action_dim=action_dimension)
        self.action_dimension = action_dimension
        self.result: dict[str, Any] | None = None
        self.event: PolicyQuery | NativeStep | None = None
        self.steps = 0
        self.started = False
        self._advance(initial=True)

    def _advance(self, response: Any = None, *, initial: bool = False, error: Exception | None = None):
        try:
            self.event = (next(self.events) if initial else
                          self.events.throw(error) if error is not None else self.events.send(response))
            if not isinstance(self.event, (PolicyQuery, NativeStep)):
                raise ValueError("beta_robolab_episode_event_invalid")
        except StopIteration as completed:
            if not isinstance(completed.value, dict):
                raise ValueError("beta_robolab_native_result_missing") from None
            self.result, self.event = completed.value, None

    def reset(self):
        if self.started:
            raise ValueError("beta_robolab_reset_after_start_refused")
        return self.observation(), {}

    def observation(self) -> Mapping[str, Any]:
        return self.event.observation if isinstance(self.event, PolicyQuery) else {}

    @property
    def active_env_ids(self):
        return [] if self.all_terminated else [0]

    @property
    def all_terminated(self):
        return self.result is not None

    def command(self):
        import numpy as np

        self.started = True
        if isinstance(self.event, PolicyQuery):
            try:
                response = self.policy.infer(self.event.observation)
            except Exception as exc:
                self._advance(error=exc)
            else:
                self._advance(response)
        if self.result is not None:
            return np.zeros(self.action_dimension, dtype=np.float32)
        if not isinstance(self.event, NativeStep):
            raise ValueError("beta_robolab_native_step_missing")
        values = np.asarray(self.event.action, dtype=np.float32)
        if values.shape != (self.action_dimension,) or not np.isfinite(values).all():
            raise ValueError("beta_robolab_native_action_invalid")
        return values

    def step(self, actions):
        import numpy as np

        if self.result is None:
            if not isinstance(self.event, NativeStep):
                raise ValueError("beta_robolab_step_without_command")
            received = actions.detach().cpu().numpy() if hasattr(actions, "detach") else np.asarray(actions)
            expected = np.asarray(self.event.action, dtype=np.float32)
            if received.shape != (1, self.action_dimension) or not np.array_equal(received[0], expected):
                raise ValueError("beta_robolab_native_action_changed")
            self.steps += 1
            self._advance()
        return self.observation(), 0.0, self.all_terminated, False, {}

    def get_env_results(self):
        if self.result is None:
            raise ValueError("beta_robolab_episode_incomplete")
        score = self.result.get("score") or {}
        return [{"env_id": 0, "success": score.get("task_succeeded") is True, "step": self.steps,
                 "scoring_authority": "blueprint_independent_simulator", "score": score}]


def make_inference_client(environment: NativeEpisodeEnvironment, client_base: type):
    """Use upstream InferenceClient's normal prepare/infer/batch/reset pipeline."""
    import numpy as np

    class BlueprintInferenceClient(client_base):
        def __init__(self):
            super().__init__()
            self.open_loop_horizon = 1

        def _extract_observation(self, raw_obs, *, env_id=0):
            if env_id != 0:
                raise ValueError("beta_robolab_env_id_invalid")
            return raw_obs

        def _pack_request(self, extracted_obs, instruction):
            # Mark the actual job instruction as used. Pixel/state transport
            # remains the retained native policy client's exact request.
            instruction["default"]
            return extracted_obs

        def _query_server(self, request):
            return environment.command()

        def _unpack_response(self, response):
            return np.asarray(response, dtype=np.float32)[None, :]

    return BlueprintInferenceClient()


def run_robolab_episode(events: Generator, *, policy: Any, prompt: str, maximum_steps: int,
                        control_hz: float, action_dimension: int) -> dict[str, Any]:
    verify_loaded_robolab()
    # These are the unchanged, digest-verified upstream implementations.
    from robolab.eval import InferenceClient, run_episode

    environment = NativeEpisodeEnvironment(events, policy=policy, maximum_steps=maximum_steps,
                                           action_dimension=action_dimension)
    client = make_inference_client(environment, InferenceClient)
    config = SimpleNamespace(instruction=prompt, sim=SimpleNamespace(dt=1.0 / control_hz, render_interval=1))
    try:
        if environment.result is None:
            results, _subtasks, timing = run_episode(environment, config, 0, client, headless=True,
                                                    save_videos=False, video_mode="none", enable_gt_state=False)
        else:
            results, timing = environment.get_env_results(), {}
        result = environment.result
        if result is None:
            raise ValueError("beta_robolab_episode_step_budget_exhausted")
        result["evaluation_backend"] = {
            "backend": BACKEND, "source_commit": ROBOLAB_COMMIT,
            "source_archive_sha256": ROBOLAB_ARCHIVE_SHA256,
            "episode_loop": "robolab.eval.run_episode", "client": "robolab.eval.InferenceClient",
            "native_action_steps": environment.steps, "timing": timing, "results": results,
            "ground_truth_supplied_to_policy": False,
            "task_source": "sealed_blueprint_native_task_packet",
            "scoring_authority": "blueprint_independent_simulator",
        }
        if "receipt_digest" in result:
            result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
        return result
    finally:
        events.close()
