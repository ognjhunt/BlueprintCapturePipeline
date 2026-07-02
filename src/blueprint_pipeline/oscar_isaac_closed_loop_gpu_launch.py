"""Build the RunPod pod startup that runs the per-step OSCAR-2B <-> SAM3 closed loop on GPU.

This packages the on-pod setup so the bounded closed-loop run (oscar_isaac_closed_loop_eval) is
one launch: install Blueprint (public repo) + clone OSCAR-2B + fetch its checkpoint, drop the
start observation and route on the pod, run the closed-loop CLI, and upload the per-step results.

Startup construction is pure string building (no GPU, no network), so it is unit-testable; the
actual RunPod create is the caller's gated step. Mirrors the OSCAR provider's proven runtime
setup (clone wuzy2115/oscar-public, download zywu2115/OSCAR-2B, the same import deps) so the
co-located run reuses what already generated a real OSCAR-2B rollout.
"""
from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path
from typing import Sequence

from .oscar_official_release import (
    OFFICIAL_OSCAR_HF_REPO,
    OFFICIAL_OSCAR_HF_REVISION,
    OFFICIAL_OSCAR_SOURCE_COMMIT,
    OFFICIAL_OSCAR_SOURCE_URL,
)
from .oscar_wam_command_adapter import DEFAULT_NUM_FRAMES as DEFAULT_OSCAR_NUM_FRAMES

DEFAULT_BLUEPRINT_REPO_URL = "https://github.com/ognjhunt/BlueprintCapturePipeline.git"
DEFAULT_OSCAR_SOURCE_URL = OFFICIAL_OSCAR_SOURCE_URL
DEFAULT_OSCAR_SOURCE_REF = OFFICIAL_OSCAR_SOURCE_COMMIT
DEFAULT_OSCAR_HF_REPO = OFFICIAL_OSCAR_HF_REPO
DEFAULT_OSCAR_HF_REVISION = OFFICIAL_OSCAR_HF_REVISION
# The import deps OSCAR inference needs on top of the pytorch base image — the required set from
# the OSCAR provider's _ensure_dependencies (hf_transfer is required: HF_HUB_ENABLE_HF_TRANSFER=1).
OSCAR_RUNTIME_PIP_PACKAGES = (
    "huggingface_hub",
    "hf_transfer",
    "opencv-python-headless",
    "imageio",
    "imageio-ffmpeg",
    "ffmpegcv",
    "peft",
)


def _route_points_json(route_points: Sequence[Sequence[float]]) -> str:
    return json.dumps({"route_points": [[float(c) for c in p] for p in route_points]})


def _closed_loop_optional_args(
    *,
    perception_target_prompts: Sequence[str],
    require_real_perception_backend: bool,
    require_sam3_completed: bool,
    require_da3_completed: bool,
) -> str:
    args: list[str] = []
    for prompt in perception_target_prompts:
        cleaned = str(prompt).strip()
        if cleaned:
            args.extend(["--perception-target-prompt", cleaned])
    if require_real_perception_backend:
        args.append("--require-real-perception-backend")
    if require_sam3_completed:
        args.append("--require-sam3-completed")
    if require_da3_completed:
        args.append("--require-da3-completed")
    return " ".join(shlex.quote(arg) for arg in args)


def build_closed_loop_pod_startup(
    *,
    start_frame_path: str | Path,
    route_points: Sequence[Sequence[float]],
    steps: int,
    task_prompt: str = "walk to the sink",
    num_frames: int = DEFAULT_OSCAR_NUM_FRAMES,
    harness_backend_kind: str = "fixture",
    perception_target_prompts: Sequence[str] = (),
    require_real_perception_backend: bool = False,
    require_sam3_completed: bool = False,
    require_da3_completed: bool = False,
    output_get_url: str = "",
    output_put_url: str = "",
    blueprint_repo_url: str = DEFAULT_BLUEPRINT_REPO_URL,
    oscar_source_url: str = DEFAULT_OSCAR_SOURCE_URL,
    oscar_source_ref: str = DEFAULT_OSCAR_SOURCE_REF,
    oscar_hf_repo: str = DEFAULT_OSCAR_HF_REPO,
    oscar_hf_revision: str = DEFAULT_OSCAR_HF_REVISION,
    workdir: str = "/workspace/closed_loop",
) -> str:
    """Return the bash startup script the pod runs. The start frame + route are baked in (base64 /
    inline) so no extra staging is needed; results upload to ``output_put_url`` when given.
    """
    start_b64 = base64.b64encode(Path(start_frame_path).expanduser().read_bytes()).decode("ascii")
    route_json = _route_points_json(route_points)
    pip_pkgs = " ".join(OSCAR_RUNTIME_PIP_PACKAGES)
    task_prompt_arg = shlex.quote(str(task_prompt))
    harness_backend_kind_arg = shlex.quote(str(harness_backend_kind))
    optional_loop_args = _closed_loop_optional_args(
        perception_target_prompts=perception_target_prompts,
        require_real_perception_backend=require_real_perception_backend,
        require_sam3_completed=require_sam3_completed,
        require_da3_completed=require_da3_completed,
    )
    optional_loop_line = f"  {optional_loop_args} \\\n" if optional_loop_args else ""
    # marker prints bracket each phase so the heartbeat/poll can see progress, like the OSCAR lane.
    return f"""#!/usr/bin/env bash
set -uo pipefail
echo "BLUEPRINT_CLOSED_LOOP_START $(date -u +%FT%TZ)"
export HF_HUB_ENABLE_HF_TRANSFER=1
export DEBIAN_FRONTEND=noninteractive
mkdir -p {workdir} /opt /models
apt-get update -y >/dev/null 2>&1 && apt-get install -y git ffmpeg >/dev/null 2>&1 || true

echo "BLUEPRINT_CLOSED_LOOP_PHASE deps_install"
python -m pip install -q --upgrade pip || true
python -m pip install -q {pip_pkgs} || echo "BLUEPRINT_CLOSED_LOOP_WARN optional_pip_failed"

echo "BLUEPRINT_CLOSED_LOOP_PHASE blueprint_clone"
git clone --depth 1 {blueprint_repo_url} /opt/blueprint || echo "BLUEPRINT_CLOSED_LOOP_BLOCK blueprint_clone_failed"
python -m pip install -q -e /opt/blueprint || echo "BLUEPRINT_CLOSED_LOOP_WARN blueprint_install_partial"

echo "BLUEPRINT_CLOSED_LOOP_PHASE oscar_clone"
rm -rf /opt/oscar
git init /opt/oscar >/dev/null 2>&1 \
  && git -C /opt/oscar remote add origin {oscar_source_url} \
  && git -C /opt/oscar fetch --depth 1 origin {oscar_source_ref} >/dev/null 2>&1 \
  && git -C /opt/oscar checkout --detach FETCH_HEAD >/dev/null 2>&1 \
  || echo "BLUEPRINT_CLOSED_LOOP_BLOCK oscar_clone_failed"
OSCAR_SOURCE_COMMIT="$(git -C /opt/oscar rev-parse HEAD 2>/dev/null || true)"
[ "$OSCAR_SOURCE_COMMIT" = "{oscar_source_ref}" ] || echo "BLUEPRINT_CLOSED_LOOP_BLOCK oscar_source_commit_mismatch"
export BLUEPRINT_OSCAR_WAM_SOURCE_URL="{oscar_source_url}"
export BLUEPRINT_OSCAR_WAM_SOURCE_REF="{oscar_source_ref}"
export BLUEPRINT_OSCAR_WAM_HF_REPO="{oscar_hf_repo}"
export BLUEPRINT_OSCAR_WAM_HF_REVISION="{oscar_hf_revision}"
[ -f /opt/oscar/requirements.txt ] && python -m pip install -q -r /opt/oscar/requirements.txt || true

echo "BLUEPRINT_CLOSED_LOOP_PHASE checkpoint_download"
python -m huggingface_hub.commands.huggingface_cli download {oscar_hf_repo} --revision {oscar_hf_revision} --local-dir /models/oscar-2b >/dev/null 2>&1 \
  || huggingface-cli download {oscar_hf_repo} --revision {oscar_hf_revision} --local-dir /models/oscar-2b >/dev/null 2>&1 \
  || echo "BLUEPRINT_CLOSED_LOOP_BLOCK checkpoint_download_failed"

echo "BLUEPRINT_CLOSED_LOOP_PHASE inputs"
echo "{start_b64}" | base64 -d > {workdir}/start.png
cat > {workdir}/route.json <<'ROUTE_EOF'
{route_json}
ROUTE_EOF

echo "BLUEPRINT_CLOSED_LOOP_PHASE run_loop"
python -m blueprint_pipeline.oscar_isaac_closed_loop_eval \
	  --start-frame {workdir}/start.png \
	  --route-file {workdir}/route.json \
	  --steps {int(steps)} \
	  --task-prompt {task_prompt_arg} \
	  --num-frames {int(num_frames)} \
	  --oscar-repo /opt/oscar \
	  --checkpoint /models/oscar-2b \
	  --harness-backend-kind {harness_backend_kind_arg} \
{optional_loop_line}\
	  --output-dir {workdir}/out 2>&1 | tee {workdir}/run.log
RUN_RC=${{PIPESTATUS[0]}}
echo "BLUEPRINT_CLOSED_LOOP_PHASE upload run_rc=$RUN_RC"
cd {workdir} && tar czf /workspace/closed_loop_output.tgz out run.log 2>/dev/null || true
if [ -n "{output_put_url}" ]; then
  curl -sS -X PUT -T /workspace/closed_loop_output.tgz "{output_put_url}" \
    && echo "BLUEPRINT_CLOSED_LOOP_UPLOAD_OK" || echo "BLUEPRINT_CLOSED_LOOP_BLOCK upload_failed"
fi
echo "BLUEPRINT_CLOSED_LOOP_DONE rc=$RUN_RC"
"""
