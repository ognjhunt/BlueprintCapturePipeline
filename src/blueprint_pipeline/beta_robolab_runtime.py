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
import shutil
import sys
from collections.abc import Generator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import zipfile

from .decision_evidence_contracts import canonical_digest
from .policy_episode_events import NativeStep, PolicyQuery

BACKEND = "robolab.v1"
ROBOLAB_COMMIT = "b6cfcf6ccd2113c4cd1216d35b33024a7d5691da"
ROBOLAB_ARCHIVE_SHA256 = "sha256:16092baa43c5a2eb78f6dc8a330330e8e8bfc6a8cafce59efff560b26d6af278"

IMAGEIO_WHEEL = "imageio-2.37.2-py3-none-any.whl"
IMAGEIO_SHA256 = "sha256:ad9adfb20335d718c03de457358ed69f141021a333c40a53e57273d8a5bd0b9b"
ROBOLAB_MANIFEST_SHA256 = "sha256:b19532cef9431b39e586310cf169ba09145cacb570592ed0c143ce5649a8fd22"
OPENCV_WHEEL = "opencv_python_headless-4.10.0.84-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
OPENCV_SHA256 = "sha256:377d08a7e48a1405b5e84afcbe4798464ce7ee17081c1c23619c8b398ff18295"

# Third-party modules RoboLab's run path imports that neither the native
# dependency contract nor the Isaac Sim launcher promises. Each ships as its
# unchanged PyPI wheel, is digest-checked when staged and again after load.
IMAGEIO_DEPENDENCY = {
    # RoboLab imports its video utility even when recording is disabled.
    "distribution": "imageio", "version": "2.37.2", "wheel": IMAGEIO_WHEEL, "sha256": IMAGEIO_SHA256,
    "size_bytes": 317_646, "license_spdx": "BSD-2-Clause", "import_names": ["imageio"], "delivery": "vendored",
}
OPENCV_DEPENDENCY = {
    # robolab.eval.episode imports cv2 at module level; it calls cv2.imshow only
    # when not headless, and Blueprint always runs headless. This is the exact
    # owner-approved version in docs/runtime_dependency_license_policy.json.
    # Its abi3 binary targets the NumPy 2.0 C API, so it loads with the NumPy 2.x
    # that Isaac Lab 3 requires (2.3.1 in the G1 runtime packet).
    "distribution": "opencv-python-headless", "version": "4.10.0.84", "wheel": OPENCV_WHEEL,
    "sha256": OPENCV_SHA256, "size_bytes": 49_858_781, "license_spdx": "Apache-2.0",
    "import_names": ["cv2"], "delivery": "pinned_download",
    "url": ("https://files.pythonhosted.org/packages/d1/09/"
            "248f86a404567303cdf120e4a301f389b68e3b18e5c0cc428de327da609c/" + OPENCV_WHEEL),
}
PYTHON_DEPENDENCY_WHEELS = (IMAGEIO_DEPENDENCY, OPENCV_DEPENDENCY)


def _dependency_code(row: Mapping[str, Any]) -> str:
    return str(row["import_names"][0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _stage_dependency_wheel(runtime: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one pinned wheel beside RoboLab and extract it unchanged."""
    code = _dependency_code(row)
    if row["delivery"] == "vendored":
        source = Path(__file__).resolve().parent / "_vendor" / row["wheel"]
    else:
        # Bundle building runs on the control plane; the worker never fetches.
        from .pinned_wheel_cache import fetch_pinned_wheel

        source = fetch_pinned_wheel(row, destination_dir=runtime)
    staged = runtime / row["wheel"]
    if source.stat().st_size != row["size_bytes"] or _sha256_file(source) != row["sha256"]:
        raise ValueError(f"beta_robolab_{code}_wheel_digest_mismatch")
    if source.resolve() != staged.resolve():
        shutil.copyfile(source, staged)
    if _sha256_file(staged) != row["sha256"]:
        raise ValueError(f"beta_robolab_{code}_wheel_digest_mismatch")
    with zipfile.ZipFile(staged) as wheel:
        for entry in wheel.infolist():
            relative = Path(entry.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"beta_robolab_{code}_path_invalid")
            if not entry.is_dir():
                target = runtime / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(wheel.read(entry))
    return {key: row[key] for key in ("distribution", "version", "wheel", "sha256", "size_bytes",
                                      "license_spdx", "import_names", "delivery")}


def stage_robolab_source(runtime: Path) -> dict[str, Any]:
    """Install exact upstream Python sources and their pinned wheels into a bundle.

    RoboLab's sources and the small ImageIO wheel are vendored; the OpenCV wheel
    comes from the control plane's pinned wheel cache. The worker stays offline.
    """
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
    dependencies = [_stage_dependency_wheel(runtime, row) for row in PYTHON_DEPENDENCY_WHEELS]
    (runtime / "robolab-source.v1.json").write_bytes(manifest_bytes)
    return {"distribution": "robolab", "version": manifest["release_version"],
            "source_commit": ROBOLAB_COMMIT, "archive_sha256": ROBOLAB_ARCHIVE_SHA256,
            "files": manifest["files"], "python_dependencies": dependencies}


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

    for row in PYTHON_DEPENDENCY_WHEELS:
        code = _dependency_code(row)
        wheel_path = runtime / row["wheel"]
        if _sha256_file(wheel_path) != row["sha256"]:
            raise ValueError(f"beta_robolab_loaded_{code}_wheel_digest_mismatch")
        with zipfile.ZipFile(wheel_path) as wheel:
            for entry in wheel.infolist():
                if not entry.is_dir() and (runtime / entry.filename).read_bytes() != wheel.read(entry):
                    raise ValueError(f"beta_robolab_loaded_{code}_source_digest_mismatch")


def _loaded_dependency_origins(runtime: Path) -> dict[str, Any]:
    """Which copy each pinned dependency actually imported from (recorded, not enforced)."""
    origins = {}
    for row in PYTHON_DEPENDENCY_WHEELS:
        for name in row["import_names"]:
            module = sys.modules.get(name)
            source = getattr(module, "__file__", None)
            path = Path(source).resolve() if source else None
            origins[name] = {"distribution": row["distribution"], "pinned_version": row["version"],
                             "loaded": module is not None, "path": str(path) if path else None,
                             "from_staged_wheel": path is not None and path.is_relative_to(runtime)}
    return origins


def preflight_robolab_runtime() -> dict[str, Any]:
    """Resolve real runtime imports after Isaac launch, before policy clients."""
    verify_loaded_robolab()
    from robolab.eval import InferenceClient, run_episode
    if not isinstance(InferenceClient, type) or not callable(run_episode):
        raise ValueError("beta_robolab_upstream_entrypoints_unavailable")
    import robolab

    origins = _loaded_dependency_origins(Path(robolab.__file__).resolve().parent.parent)
    return {"backend": BACKEND, "source_commit": ROBOLAB_COMMIT, "status": "passed",
            "python_dependency_origins": origins,
            "warnings": sorted(f"beta_robolab_dependency_not_from_staged_wheel:{name}"
                               for name, origin in origins.items()
                               if origin["loaded"] and not origin["from_staged_wheel"])}


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
