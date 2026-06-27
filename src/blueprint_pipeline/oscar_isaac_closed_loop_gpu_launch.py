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
from pathlib import Path
from typing import Sequence

DEFAULT_BLUEPRINT_REPO_URL = "https://github.com/ognjhunt/BlueprintCapturePipeline.git"
DEFAULT_OSCAR_SOURCE_URL = "https://github.com/wuzy2115/oscar-public.git"
DEFAULT_OSCAR_HF_REPO = "zywu2115/OSCAR-2B"
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


def build_closed_loop_pod_startup(
    *,
    start_frame_path: str | Path,
    route_points: Sequence[Sequence[float]],
    steps: int,
    task_prompt: str = "walk to the sink",
    num_frames: int = 8,
    harness_backend_kind: str = "fixture",
    output_get_url: str = "",
    output_put_url: str = "",
    blueprint_repo_url: str = DEFAULT_BLUEPRINT_REPO_URL,
    oscar_source_url: str = DEFAULT_OSCAR_SOURCE_URL,
    oscar_hf_repo: str = DEFAULT_OSCAR_HF_REPO,
    workdir: str = "/workspace/closed_loop",
) -> str:
    """Return the bash startup script the pod runs. The start frame + route are baked in (base64 /
    inline) so no extra staging is needed; results upload to ``output_put_url`` when given.
    """
    start_b64 = base64.b64encode(Path(start_frame_path).expanduser().read_bytes()).decode("ascii")
    route_json = _route_points_json(route_points)
    pip_pkgs = " ".join(OSCAR_RUNTIME_PIP_PACKAGES)
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
git clone --depth 1 {oscar_source_url} /opt/oscar || echo "BLUEPRINT_CLOSED_LOOP_BLOCK oscar_clone_failed"
[ -f /opt/oscar/requirements.txt ] && python -m pip install -q -r /opt/oscar/requirements.txt || true

echo "BLUEPRINT_CLOSED_LOOP_PHASE checkpoint_download"
python -m huggingface_hub.commands.huggingface_cli download {oscar_hf_repo} --local-dir /models/oscar-2b >/dev/null 2>&1 \
  || huggingface-cli download {oscar_hf_repo} --local-dir /models/oscar-2b >/dev/null 2>&1 \
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
  --task-prompt "{task_prompt}" \
  --num-frames {int(num_frames)} \
  --oscar-repo /opt/oscar \
  --checkpoint /models/oscar-2b \
  --harness-backend-kind {harness_backend_kind} \
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
