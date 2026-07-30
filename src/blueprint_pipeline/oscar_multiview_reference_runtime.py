"""Pinned resident OSCAR runtime for the new-site two-view canary.

This module adapts the existing line-delimited resident OSCAR worker to the
``CallableMultiViewOscarWamArm`` generator contract.  It validates the exact
first-frame and skeleton-conditioning bytes for every view before dispatch,
loads the model once, and fails closed on any provenance, asset, or worker
protocol defect.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .droid_oscar_closed_loop_adapter import EXTERIOR_VIEW, WRIST_VIEW
from .oscar_resident_worker import ResidentOscarWorker, build_resident_worker_argv
from .oscar_runtime_asset_contract import offline_preflight
from .oscar_runtime_source_provenance import verify_source_tree
from .policy_ranking_thesis import file_sha256


OSCAR_RUNTIME_PYTHON_ENV = "BLUEPRINT_OSCAR_RUNTIME_PYTHON"
OSCAR_RUNTIME_REPO_ENV = "BLUEPRINT_OSCAR_RUNTIME_REPO"
OSCAR_RUNTIME_CHECKPOINT_ENV = "BLUEPRINT_OSCAR_RUNTIME_CHECKPOINT"
OSCAR_RUNTIME_SOURCE_SEAL_ENV = "BLUEPRINT_OSCAR_RUNTIME_SOURCE_SEAL"
OSCAR_RUNTIME_ASSET_CACHE_ENV = "BLUEPRINT_OSCAR_RUNTIME_ASSET_CACHE"

DEFAULT_OSCAR_RUNTIME_PYTHON = "/opt/oscar-venv/bin/python"
DEFAULT_OSCAR_RUNTIME_REPO = "/opt/OSCAR"
DEFAULT_OSCAR_RUNTIME_CHECKPOINT = "/opt/blueprint/ckpts/oscar"
DEFAULT_OSCAR_RUNTIME_SOURCE_SEAL = "/opt/blueprint/oscar_source_provenance.json"
DEFAULT_OSCAR_RUNTIME_ASSET_CACHE = "/opt/blueprint/oscar-runtime-assets"

OSCAR_NUM_STEPS = 35
OSCAR_GUIDANCE = 6.0
OSCAR_SHIFT = 5.0
OSCAR_HEIGHT = 480
OSCAR_WIDTH = 640
OSCAR_FPS = 15.0
OSCAR_NUM_FRAMES = 81
OSCAR_REQUIRED_VIEWS = (EXTERIOR_VIEW, WRIST_VIEW)


def _safe_file(value: str | Path, *, reason: str) -> Path:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise ValueError(reason)
    resolved = unresolved.resolve()
    if not resolved.is_file():
        raise ValueError(reason)
    return resolved


def _safe_dir(value: str | Path, *, reason: str) -> Path:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise ValueError(reason)
    resolved = unresolved.resolve()
    if not resolved.is_dir():
        raise ValueError(reason)
    return resolved


class OscarMultiViewReferenceRuntime:
    """One resident official OSCAR model serving both frozen camera views."""

    def __init__(
        self,
        *,
        python: str | Path,
        oscar_repo: str | Path,
        checkpoint: str | Path,
        source_seal: str | Path,
        asset_cache: str | Path,
        evidence_dir: str | Path,
        worker_factory: Callable[..., ResidentOscarWorker] = ResidentOscarWorker,
        provenance_verifier: Callable[..., Mapping[str, Any]] = verify_source_tree,
        asset_preflight: Callable[..., Mapping[str, Any]] = offline_preflight,
    ) -> None:
        self.python = Path(python).expanduser()
        self.oscar_repo = Path(oscar_repo).expanduser()
        self.checkpoint = Path(checkpoint).expanduser()
        self.source_seal = Path(source_seal).expanduser()
        self.asset_cache = Path(asset_cache).expanduser()
        self.evidence_dir = Path(evidence_dir).expanduser().resolve()
        self._worker_factory = worker_factory
        self._provenance_verifier = provenance_verifier
        self._asset_preflight = asset_preflight
        self._worker: ResidentOscarWorker | None = None
        self._seen_output_paths: set[Path] = set()

    @classmethod
    def from_environment(cls, *, evidence_dir: str | Path) -> "OscarMultiViewReferenceRuntime":
        return cls(
            python=os.getenv(OSCAR_RUNTIME_PYTHON_ENV, DEFAULT_OSCAR_RUNTIME_PYTHON),
            oscar_repo=os.getenv(OSCAR_RUNTIME_REPO_ENV, DEFAULT_OSCAR_RUNTIME_REPO),
            checkpoint=os.getenv(OSCAR_RUNTIME_CHECKPOINT_ENV, DEFAULT_OSCAR_RUNTIME_CHECKPOINT),
            source_seal=os.getenv(OSCAR_RUNTIME_SOURCE_SEAL_ENV, DEFAULT_OSCAR_RUNTIME_SOURCE_SEAL),
            asset_cache=os.getenv(OSCAR_RUNTIME_ASSET_CACHE_ENV, DEFAULT_OSCAR_RUNTIME_ASSET_CACHE),
            evidence_dir=evidence_dir,
        )

    def start(self) -> dict[str, Any]:
        if self._worker is not None:
            raise RuntimeError("new_site_oscar_resident_worker_already_started")
        python = _safe_file(self.python, reason="new_site_oscar_runtime_python_missing_or_unsafe")
        repo = _safe_dir(self.oscar_repo, reason="new_site_oscar_runtime_repo_missing_or_unsafe")
        checkpoint = _safe_dir(
            self.checkpoint,
            reason="new_site_oscar_runtime_checkpoint_missing_or_unsafe",
        )
        source_seal = _safe_file(
            self.source_seal,
            reason="new_site_oscar_runtime_source_seal_missing_or_unsafe",
        )
        asset_cache = _safe_dir(
            self.asset_cache,
            reason="new_site_oscar_runtime_asset_cache_missing_or_unsafe",
        )
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        provenance = dict(
            self._provenance_verifier(
                source_root=repo,
                seal_path=source_seal,
                artifact_path=self.evidence_dir / "oscar_runtime_source_provenance.json",
            )
        )
        if provenance.get("status") != "passed" or provenance.get("blockers"):
            raise ValueError("new_site_oscar_runtime_source_provenance_blocked")
        assets = dict(
            self._asset_preflight(
                asset_cache,
                oscar_checkpoint_root=checkpoint,
                evidence_output_path=self.evidence_dir / "oscar_runtime_asset_preflight.json",
            )
        )
        if assets.get("status") != "passed" or assets.get("blockers"):
            raise ValueError("new_site_oscar_runtime_asset_preflight_blocked")
        worker = self._worker_factory(
            argv=build_resident_worker_argv(
                python=str(python),
                oscar_repo=repo,
                checkpoint=checkpoint,
                num_steps=OSCAR_NUM_STEPS,
                guidance=OSCAR_GUIDANCE,
                shift=OSCAR_SHIFT,
                height=OSCAR_HEIGHT,
                width=OSCAR_WIDTH,
                fps=OSCAR_FPS,
            ),
            cwd=repo,
            env=os.environ.copy(),
            max_restarts=0,
            require_gpu_residency=True,
        )
        ready = dict(worker.start())
        checkpoint_sha256 = str(ready.get("checkpoint_sha256") or "")
        if len(checkpoint_sha256) != 64:
            worker.close()
            raise ValueError("new_site_oscar_resident_checkpoint_identity_missing")
        self._worker = worker
        return {
            "status": "ready",
            "runtime": "resident_official_oscar_multiview",
            "ready": ready,
            "source_provenance": provenance,
            "asset_preflight": assets,
            "required_views": list(OSCAR_REQUIRED_VIEWS),
            "num_frames": OSCAR_NUM_FRAMES,
        }

    def __call__(
        self,
        *,
        view_id: str,
        view_request: Mapping[str, Any],
        task_prompt: str,
        negative_prompt: str,
        output_dir: Path,
        seed: int,
    ) -> dict[str, Any]:
        if self._worker is None:
            raise RuntimeError("new_site_oscar_resident_worker_not_started")
        if view_id not in OSCAR_REQUIRED_VIEWS:
            raise ValueError("new_site_oscar_runtime_view_invalid")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("new_site_oscar_runtime_seed_invalid")
        first_frame = _safe_file(
            str(view_request.get("first_frame_path") or ""),
            reason=f"new_site_oscar_first_frame_missing_or_unsafe:{view_id}",
        )
        skeleton_video = _safe_file(
            str(view_request.get("skeleton_video_path") or ""),
            reason=f"new_site_oscar_skeleton_video_missing_or_unsafe:{view_id}",
        )
        if view_request.get("first_frame_sha256") != file_sha256(first_frame):
            raise ValueError(f"new_site_oscar_first_frame_sha256_mismatch:{view_id}")
        if view_request.get("skeleton_video_sha256") != file_sha256(skeleton_video):
            raise ValueError(f"new_site_oscar_skeleton_video_sha256_mismatch:{view_id}")
        if len(str(view_request.get("camera_calibration_sha256") or "")) != 64:
            raise ValueError(f"new_site_oscar_camera_calibration_identity_missing:{view_id}")
        resolved_output = Path(output_dir).expanduser().resolve()
        resolved_output.mkdir(parents=True, exist_ok=True)
        output_video = resolved_output / "oscar_generated.mp4"
        if output_video in self._seen_output_paths or output_video.exists():
            raise ValueError("new_site_oscar_runtime_output_reuse_forbidden")
        response = dict(
            self._worker.generate(
                {
                    "reference_frame_path": str(first_frame),
                    "task_prompt": str(task_prompt),
                    "negative_prompt": str(negative_prompt),
                    "num_frames": OSCAR_NUM_FRAMES,
                    "seed": seed,
                    "output_video": str(output_video),
                    "skeleton_video": str(skeleton_video),
                }
            )
        )
        if response.get("status") != "ok" or response.get("blockers"):
            raise RuntimeError(f"new_site_oscar_generation_blocked:{view_id}")
        generated = _safe_file(
            output_video,
            reason=f"new_site_oscar_generated_video_missing_or_unsafe:{view_id}",
        )
        if Path(str(response.get("output_video") or "")).expanduser().resolve() != generated:
            raise ValueError(f"new_site_oscar_generated_video_identity_mismatch:{view_id}")
        self._seen_output_paths.add(generated)
        return {
            "generated_video_path": str(generated),
            "generated_video_sha256": file_sha256(generated),
            "runtime_result_id": response.get("runtime_result_id"),
            "provider": "resident_official_oscar_multiview",
            "view_id": view_id,
            "seed": seed,
            "num_frames": OSCAR_NUM_FRAMES,
            "official_negative_prompt_parameter_supported": True,
        }

    def close(self) -> dict[str, Any] | None:
        worker = self._worker
        if worker is None:
            return None
        self._worker = None
        return worker.close_and_report(self.evidence_dir)

    def __enter__(self) -> "OscarMultiViewReferenceRuntime":
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


__all__ = [
    "OSCAR_REQUIRED_VIEWS",
    "OscarMultiViewReferenceRuntime",
]
