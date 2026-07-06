"""Launcher-facing contract for the sealed ``blueprint-groot-oscar-eval`` image.

This module is the seam between the closed-loop eval lane
(``oscar_isaac_closed_loop_eval`` + ``groot_sonic_policy_endpoint``) and a
prebaked worker image that freezes tonight's dependency archaeology (torch /
cuDNN / transformer-engine + OSCAR + Isaac-GR00T venv + checkpoints + our
package). When the sealed image ref is configured and confirmed, a paid pod is
``docker pull`` + start the baked GR00T server + run the closed-loop CLI — with
no clone / venv / download churn.

Everything here is pure and hermetic (no Docker, no pod, no network): it
resolves configuration and emits *plans* (arg-lists, path allow-lists, env
markers). Side-effecting build/launch belongs in the shell scripts and the
runbook that consume these plans.

Claim boundary: a configured/confirmed sealed image proves build + runtime
readiness only. It does not prove provider startup, GR00T policy inference, WAM
generation quality, or semantic task success.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .oscar_official_release import OFFICIAL_OSCAR_HF_REVISION

# --------------------------------------------------------------------------- #
# configuration keys
# --------------------------------------------------------------------------- #

IMAGE_REF_ENV = "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF"
IMAGE_REF_FILE_ENV = "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF_FILE"
ROBOT_EVAL_WORKER_IMAGE_REF_ENV = "BLUEPRINT_ROBOT_EVAL_WORKER_IMAGE_REF"
DEFAULT_IMAGE_REF_FILE = "~/.blueprint-secrets/groot_oscar_closed_loop_image_ref"

SEALED_CONFIRMED_ENV = "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_SEALED_IMAGE_CONFIRMED"

# Baked-path env vars (defaults match the image layout; overridable so the same
# wiring works whether the image was produced by the Dockerfile or by snapshot).
OSCAR_REPO_ENV = "BLUEPRINT_GROOT_OSCAR_OSCAR_REPO"
OSCAR_CHECKPOINT_ENV = "BLUEPRINT_GROOT_OSCAR_OSCAR_CHECKPOINT"
GROOT_ROOT_ENV = "BLUEPRINT_GROOT_OSCAR_GROOT_ROOT"
SONIC_CHECKPOINT_ENV = "BLUEPRINT_GROOT_OSCAR_SONIC_CHECKPOINT"

DEFAULT_OSCAR_REPO = "/opt/OSCAR"
DEFAULT_OSCAR_CHECKPOINT = "/opt/blueprint/ckpts/oscar"
DEFAULT_GROOT_ROOT = "/opt/gr00t"
DEFAULT_SONIC_CHECKPOINT = "/opt/blueprint/ckpts/sonic"
DEFAULT_GROOT_VENV = "/opt/gr00t-venv"
DEFAULT_WBC_ROOT = "/opt/wbc"

POLICY_SERVER_PORT = 5550
POLICY_SERVER_URL = f"tcp://127.0.0.1:{POLICY_SERVER_PORT}"
SONIC_EMBODIMENT_TAG = "UNITREE_G1_SONIC"

_UNSTABLE_TAGS = {"latest", "local", "dev", "test"}

_CLAIM_BOUNDARY = (
    "A configured/confirmed sealed image proves build + runtime readiness only. "
    "It does not prove provider startup, GR00T policy inference, WAM generation "
    "quality, or semantic task success."
)


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


# --------------------------------------------------------------------------- #
# image-ref resolution (mirrors _configured_isaac_worker_image_ref order)
# --------------------------------------------------------------------------- #

def configured_image_ref(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Resolve the sealed image ref: explicit env > secret file > generic env."""
    env = _env(env)
    explicit = _string(env.get(IMAGE_REF_ENV))
    if explicit:
        return {
            "image_ref": explicit,
            "source": IMAGE_REF_ENV,
            "configured": True,
            "image_ref_file": None,
            "image_ref_file_present": False,
            "raw_secret_values_recorded": False,
        }
    file_value = _string(env.get(IMAGE_REF_FILE_ENV))
    image_ref_file = Path(file_value or DEFAULT_IMAGE_REF_FILE).expanduser()
    if image_ref_file.is_file():
        image_ref = image_ref_file.read_text(encoding="utf-8").strip()
        return {
            "image_ref": image_ref,
            "source": IMAGE_REF_FILE_ENV if file_value else "default_blueprint_secret_file_path",
            "configured": bool(image_ref),
            "image_ref_file": str(image_ref_file),
            "image_ref_file_present": True,
            "raw_secret_values_recorded": False,
        }
    generic = _string(env.get(ROBOT_EVAL_WORKER_IMAGE_REF_ENV))
    if generic:
        return {
            "image_ref": generic,
            "source": ROBOT_EVAL_WORKER_IMAGE_REF_ENV,
            "configured": True,
            "image_ref_file": str(image_ref_file),
            "image_ref_file_present": False,
            "raw_secret_values_recorded": False,
        }
    return {
        "image_ref": "",
        "source": None,
        "configured": False,
        "image_ref_file": str(image_ref_file),
        "image_ref_file_present": False,
        "raw_secret_values_recorded": False,
    }


def image_ref_launch_blockers(image_ref: str) -> list[str]:
    """Refuse missing/unversioned/unstable refs (same rule as the build script)."""
    ref = _string(image_ref)
    if not ref:
        return ["missing_image_ref"]
    if ":" not in ref and "@sha256:" not in ref:
        return ["image_ref_must_be_versioned"]
    tag = ref.rsplit("@", maxsplit=1)[0].rsplit(":", maxsplit=1)[-1] if "@sha256:" not in ref else ""
    if tag in _UNSTABLE_TAGS:
        return ["image_ref_refuses_unstable_tag"]
    return []


# --------------------------------------------------------------------------- #
# sealed-mode gate
# --------------------------------------------------------------------------- #

def sealed_image_contract(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Fail-closed sealed-mode contract for the closed-loop launcher.

    ``sealed_active`` is True only when the image ref is configured, launchable
    (versioned, not an unstable tag), and explicitly confirmed. Otherwise the
    legacy runtime-bootstrap path must be used and ``blockers`` names why.
    """
    env = _env(env)
    resolved = configured_image_ref(env=env)
    image_ref = _string(resolved.get("image_ref"))
    configured = bool(resolved.get("configured"))

    blockers = list(image_ref_launch_blockers(image_ref))
    sealed_confirmed = _string(env.get(SEALED_CONFIRMED_ENV)).lower() == "true"
    if not sealed_confirmed:
        blockers.append("sealed_image_not_confirmed")

    sealed_active = configured and not blockers

    return {
        "schema_version": "groot_oscar_closed_loop_sealed_image_contract.v1",
        "image_ref": image_ref,
        "image_ref_source": resolved.get("source"),
        "image_ref_configured": configured,
        "sealed_confirmed": sealed_confirmed,
        "sealed_active": sealed_active,
        "oscar_repo": _string(env.get(OSCAR_REPO_ENV)) or DEFAULT_OSCAR_REPO,
        "oscar_checkpoint": _string(env.get(OSCAR_CHECKPOINT_ENV)) or DEFAULT_OSCAR_CHECKPOINT,
        "groot_root": _string(env.get(GROOT_ROOT_ENV)) or DEFAULT_GROOT_ROOT,
        "groot_venv": DEFAULT_GROOT_VENV,
        "sonic_checkpoint": _string(env.get(SONIC_CHECKPOINT_ENV)) or DEFAULT_SONIC_CHECKPOINT,
        "policy_server_url": POLICY_SERVER_URL,
        "policy_server_port": POLICY_SERVER_PORT,
        "oscar_hf_revision": OFFICIAL_OSCAR_HF_REVISION,
        "blockers": blockers,
        "raw_secret_values_recorded": False,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


# --------------------------------------------------------------------------- #
# launch plan (what the sealed image actually runs)
# --------------------------------------------------------------------------- #

def build_sealed_launch_plan(
    *,
    start_frame: str,
    route_file: str,
    steps: int,
    task_prompt: str,
    output_dir: str,
    oscar_height: int = 240,
    oscar_width: int = 320,
    device: str = "cuda:0",
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Ordered command plan: start the baked GR00T server, then the closed loop.

    Commands are arg-lists (not shell strings) so they are safe to render and
    exact to assert. When the sealed contract is not active the plan is blocked
    with empty commands and the caller must fall back to runtime bootstrap.
    """
    contract = sealed_image_contract(env=env)
    plan: dict[str, Any] = {
        "schema_version": "groot_oscar_closed_loop_sealed_launch_plan.v1",
        "sealed_active": contract["sealed_active"],
        "blockers": list(contract["blockers"]),
        "image_ref": contract["image_ref"],
        "policy_server_url": contract["policy_server_url"],
        "policy_server_port": contract["policy_server_port"],
        "groot_server_command": [],
        "closed_loop_command": [],
        "env": {},
        "raw_secret_values_recorded": False,
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    if not contract["sealed_active"]:
        return plan

    groot_root = contract["groot_root"]
    oscar_repo = contract["oscar_repo"]
    plan["groot_server_command"] = [
        f"{groot_root}/.venv/bin/python",
        f"{groot_root}/gr00t/eval/run_gr00t_server.py",
        "--model-path", contract["sonic_checkpoint"],
        "--embodiment-tag", SONIC_EMBODIMENT_TAG,
        "--device", str(device),
        "--port", str(contract["policy_server_port"]),
    ]
    plan["closed_loop_command"] = [
        "python", "-m", "blueprint_pipeline.oscar_isaac_closed_loop_eval",
        "--start-frame", start_frame,
        "--route-file", route_file,
        "--steps", str(int(steps)),
        "--task-prompt", task_prompt,
        "--oscar-repo", oscar_repo,
        "--checkpoint", contract["oscar_checkpoint"],
        "--output-dir", output_dir,
        "--groot-sonic-policy-server-url", contract["policy_server_url"],
        "--groot-root", groot_root,
        "--require-fresh-learned-policy-requery",
        "--harness-backend-kind", "fixture",
        "--oscar-height", str(int(oscar_height)),
        "--oscar-width", str(int(oscar_width)),
    ]
    plan["env"] = {
        "MUJOCO_GL": "osmesa",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "PYTHONPATH": oscar_repo,
        "BLUEPRINT_OSCAR_WAM_HF_REVISION": contract["oscar_hf_revision"],
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT": groot_root,
        "BLUEPRINT_UNITREE_GROOT_N17_SONIC_POLICY_SERVER_URL": contract["policy_server_url"],
    }
    return plan


# --------------------------------------------------------------------------- #
# snapshot layer plan (source of truth for the crane-snapshot build)
# --------------------------------------------------------------------------- #

def build_snapshot_layer_plan(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Allow-list of pod paths + image env markers for the crane-snapshot script.

    The snapshot build (primary path) tars exactly these trees onto the pod's
    base image so no ephemeral/customer cruft is captured, and stamps the sealed
    markers so the resulting image is self-describing.
    """
    contract = sealed_image_contract(env=env)
    snapshot_paths = [
        contract["oscar_repo"],
        contract["groot_root"],
        DEFAULT_GROOT_VENV,
        DEFAULT_WBC_ROOT,
        contract["oscar_checkpoint"],
        contract["sonic_checkpoint"],
        "/opt/blueprint",
    ]
    # de-dup while preserving order
    seen: set[str] = set()
    ordered_paths = [p for p in snapshot_paths if not (p in seen or seen.add(p))]
    return {
        "schema_version": "groot_oscar_closed_loop_snapshot_layer_plan.v1",
        "snapshot_paths": ordered_paths,
        "image_env": {
            SEALED_CONFIRMED_ENV: "true",
            OSCAR_REPO_ENV: contract["oscar_repo"],
            OSCAR_CHECKPOINT_ENV: contract["oscar_checkpoint"],
            GROOT_ROOT_ENV: contract["groot_root"],
            SONIC_CHECKPOINT_ENV: contract["sonic_checkpoint"],
            "MUJOCO_GL": "osmesa",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "PYTHONPATH": contract["oscar_repo"],
            "BLUEPRINT_OSCAR_WAM_HF_REVISION": contract["oscar_hf_revision"],
            "BLUEPRINT_WORKER_IMAGE_FAMILY": "groot-oscar-closed-loop-eval",
        },
        "raw_secret_values_recorded": False,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _emit(payload: Mapping[str, Any]) -> int:
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not payload.get("blockers") and payload.get("sealed_active", True) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the sealed blueprint-groot-oscar-eval image contract."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--print-sealed-contract", action="store_true", default=True)
    mode.add_argument("--print-launch-plan", action="store_true")
    mode.add_argument("--print-snapshot-plan", action="store_true")
    parser.add_argument("--start-frame", default="/workspace/initial_policy_frame.png")
    parser.add_argument("--route-file", default="/workspace/route.json")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--task-prompt", default="open the fridge")
    parser.add_argument("--output-dir", default="/workspace/t4_out")
    parser.add_argument("--oscar-height", type=int, default=240)
    parser.add_argument("--oscar-width", type=int, default=320)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    if args.print_snapshot_plan:
        return _emit(build_snapshot_layer_plan())
    if args.print_launch_plan:
        return _emit(
            build_sealed_launch_plan(
                start_frame=args.start_frame,
                route_file=args.route_file,
                steps=args.steps,
                task_prompt=args.task_prompt,
                output_dir=args.output_dir,
                oscar_height=args.oscar_height,
                oscar_width=args.oscar_width,
                device=args.device,
            )
        )
    return _emit(sealed_image_contract())


if __name__ == "__main__":
    raise SystemExit(main())
