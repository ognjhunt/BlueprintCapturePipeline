# Design: `blueprint-groot-oscar-eval` sealed worker image

Date: 2026-07-06
Owner: closed-loop eval lane (GR00T N1.7 + SONIC ⨯ OSCAR-2B WAM)
Status: approved (approach + decisions), authoring in progress

## Problem

Every GR00T + OSCAR closed-loop eval pod (`oscar_isaac_closed_loop_eval` +
`groot_sonic_policy_endpoint`, the "T4" lane) currently boots from a naked
pytorch image and pays **40–70 minutes** of runtime setup: apt, our source
install, a Python-3.10 GR00T venv + Isaac-GR00T + ~14 GB checkpoint, the OSCAR
repo + ~4 GB (fp32 ≈ 22 GB resident) weights + its dependency chain. This was
paid three times in one night. Setup is also the least reliable phase (HF
rate-limits, transformer-engine build, host/network duds).

Goal: one sealed `blueprint-groot-oscar-eval` image so a paid pod is
`docker pull + go`, freezing tonight's dependency archaeology permanently.

## Key finding — the hard layer is already frozen

The expensive OSCAR archaeology is **already a digest-pinned, pushed image**:

- `docker.io/nijelhunt/blueprint-oscar-wam@sha256:b0f3f675…` (tag
  `:20260701-cu128-ropefix`; amd64 digest `sha256:dc233346…`), recorded in
  `src/blueprint_pipeline/oscar_official_release.py`.
- It bakes: CUDA 12.8, `nvidia-cudnn-cu12>=9.10` with headers on `CPATH`,
  `torch==2.10.0+cu128` / `torchvision==0.25.0`, transformer-engine (shim by
  default; real via `NVTE_FRAMEWORK=pytorch --no-build-isolation`), OSCAR source
  `github.com/wuzy2115/oscar-public@4dea2f65` at `/opt/oscar-public`, OSCAR
  deps (incl. the `pytest`-import and torch double-pin fixes), Python 3.10 main
  env. Built from `robot_eval_jobs/oscar_wam_gpu_image_20260701_ropefix/`.

What is **not** sealed anywhere, and is exactly the nightly tax:

1. The GR00T runtime: `/opt/gr00t` (Isaac-GR00T `@e5749287`) + a **separate
   Python-3.10 `uv` venv** `/opt/gr00t-venv` with `gr00t` installed, running
   `gr00t/eval/run_gr00t_server.py` on ZMQ `:5550`.
2. Model checkpoints (SONIC `LucaFrat/groot-bs16`; OSCAR-2B `zywu2115/OSCAR-2B`
   `@c9781ffa`).
3. **Our package** `blueprint_pipeline` + `mujoco pyzmq msgpack-numpy imageio
   pillow` + `libosmesa6` in the main env.
4. Launcher wiring to select the sealed image and skip runtime bootstrap.

So this is a **thin, low-risk layer on a proven base**, not a from-scratch
build. The genuinely painful parts (torch/cuDNN/TE/OSCAR deps) are inherited by
digest and never re-derived.

## Ground truth (tonight's proven recipe)

From `output/kitchen_g1_groot_sonic_eval/groot-t4-20260706T050514Z/run_t4.sh`
and `setup_t4.log` (the corrected recipe; the log also captured a hallucinated
`hpcai-tech/OSCAR` 404 → the real sources are `wuzy2115`/`zywu2115`):

Main env:
```
apt-get install -y libosmesa6 ffmpeg
pip install -e /workspace mujoco pillow pyzmq msgpack msgpack-numpy imageio uv huggingface_hub
export MUJOCO_GL=osmesa
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/opt/OSCAR                 # OSCAR imported by PYTHONPATH, no setup.py
export BLUEPRINT_OSCAR_WAM_HF_REVISION=c9781ffa7dd8556d862d7d9f338a2ea008a58ca6
```
GR00T env:
```
git clone --depth 1 https://github.com/NVIDIA/Isaac-GR00T.git /opt/gr00t
git clone --depth 1 https://github.com/NVlabs/GR00T-WholeBodyControl.git /opt/wbc
uv venv /opt/gr00t-venv --python 3.10 && ln -sfn /opt/gr00t-venv /opt/gr00t/.venv
cd /opt/gr00t && VIRTUAL_ENV=/opt/gr00t-venv uv pip install -e .   # no '.[base]' extra
/opt/gr00t-venv/bin/python -c "from gr00t.policy.gr00t_policy import Gr00tPolicy"
```
Run shape (server on cpu to free VRAM on 24 GB; on A6000 it runs on GPU):
```
cd /opt/gr00t && .venv/bin/python gr00t/eval/run_gr00t_server.py \
  --model-path <sonic_ckpt> --embodiment-tag UNITREE_G1_SONIC --device {cpu|cuda:0} --port 5550
python -m blueprint_pipeline.oscar_isaac_closed_loop_eval \
  --start-frame <frame.png> --route-file <route.json> --steps 3 --task-prompt "…" \
  --oscar-repo /opt/OSCAR --checkpoint <oscar_ckpt> --output-dir <out> \
  --groot-sonic-policy-server-url tcp://127.0.0.1:5550 --groot-root /opt/gr00t \
  --require-fresh-learned-policy-requery --harness-backend-kind fixture \
  --oscar-height 240 --oscar-width 320
```

## Decisions

- **Bake both weight sets in** (user: optimize for fast + reliable bring-up).
  Removes the flakiest runtime steps. Image ~50–70 GB; mitigate pull time by
  splitting large weight layers for parallel pull (`split_worker_image_layer.py`
  precedent). Departs, deliberately and for this lane only, from the base
  image's "no checkpoints" policy.
- **Build by crane-snapshotting tonight's healthy pod** after its run finishes
  (env already resident → lowest spend + highest fidelity). Dockerfile is the
  reproducible spec + clean-build fallback. Paid step gated on explicit go +
  spend-guard loop.
- **`blueprint_pipeline` baked** at build time; runtime may still `pip install
  -e` a newer bundle over it (deps already satisfied → seconds) for dev freshness.

## Image contents (`blueprint-groot-oscar-eval`)

Base: `blueprint-oscar-wam@sha256:b0f3f675…` (digest-pinned). Two Python envs.

| Path | Contents |
|---|---|
| `/opt/oscar-public` (+ symlink `/opt/OSCAR`) | OSCAR source (from base), on `PYTHONPATH` |
| `/opt/gr00t` | Isaac-GR00T `@e5749287`; `.venv → /opt/gr00t-venv` |
| `/opt/gr00t-venv` | py3.10 venv with `gr00t` installed (`uv pip install -e .`) |
| `/opt/wbc` | GR00T-WholeBodyControl (for the endpoint lane; harmless for T4) |
| `/opt/blueprint/ckpts/sonic` | `LucaFrat/groot-bs16` (allow-patterns: config, index, `model-*.safetensors`, `processor/*`, `experiment_cfg/*`) |
| `/opt/blueprint/ckpts/oscar` | `zywu2115/OSCAR-2B@c9781ffa` |
| main site-packages | `blueprint_pipeline`, `mujoco pyzmq msgpack msgpack-numpy imageio pillow huggingface_hub uv` |

Baked ENV: `MUJOCO_GL=osmesa`, `PYTORCH_ALLOC_CONF=expandable_segments:True`,
`PYTHONPATH=/opt/oscar-public`, `BLUEPRINT_OSCAR_WAM_HF_REVISION=c9781ffa…`,
`BLUEPRINT_UNITREE_GROOT_N17_SONIC_ROOT=/opt/gr00t`,
`…_WBC_ROOT=/opt/wbc`, `…_SONIC_CHECKPOINT=/opt/blueprint/ckpts/sonic`,
`BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_SEALED_IMAGE_CONFIRMED=true`,
`BLUEPRINT_WORKER_IMAGE_FAMILY=groot-oscar-closed-loop-eval`.

Build-time healthcheck (`--build-time`, fail-closed): main env imports torch
(`2.10.0+cu128`), `blueprint_pipeline`, `mujoco`, `pyzmq`, `msgpack_numpy`;
OSCAR importable via `PYTHONPATH`; GR00T venv imports `Gr00tPolicy`; both
checkpoint dirs present with expected marker files.

## Build strategy

**Primary — crane snapshot of tonight's pod** (`scripts/snapshot_groot_oscar_eval_pod.sh`):
1. Pre-spend + spend-guard confirm; SSH via `~/.ssh/id_ed25519`.
2. On the pod: relocate the two checkpoints from ephemeral `/workspace/*` into
   `/opt/blueprint/ckpts/*`; write the baked-env marker; install `crane`.
3. `crane auth login docker.io` with the Docker PAT (stdin, never echoed).
4. FIFO-stream a layer of the added paths onto the pod's base image via
   `crane append` (global CLAUDE.md method), tag
   `docker.io/nijelhunt/blueprint-groot-oscar-eval:20260706-cu128-amd64`.
5. Optionally split the largest layers for parallel pull; push; record a build
   manifest.

**Secondary — clean Dockerfile build**
(`deploy/docker/robot_eval_worker/groot_oscar_closed_loop/Dockerfile` +
`scripts/build_push_groot_oscar_closed_loop_image.sh`): FROM the pinned base,
encodes the recipe above, BuildKit `--secret` HF token, versioned-tag refusal +
disk check + build manifest (mirrors `build_push_unitree_groot_sonic_wam_image.sh`).
Reproducible source of truth even though the first real build is the snapshot.

## Launcher wiring (additive, default-off)

- Image-ref config key `BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF` (+ file
  `~/.blueprint-secrets/groot_oscar_closed_loop_image_ref`), same resolution
  order as the Isaac worker (`_configured_isaac_worker_image_ref` pattern).
- Sealed-mode gate mirroring
  `BLUEPRINT_RUNPOD_UNITREE_GROOT_N17_SONIC_SEALED_IMAGE_CONFIRMED`: when the
  sealed image ref + confirmation are set, the launch script assumes the baked
  env and **skips clone/venv/download**, just: start GR00T server from the baked
  venv+ckpt, then run the closed-loop CLI with baked `--oscar-repo`,
  `--checkpoint`, `--groot-root`. Absent the gate, the existing runtime-bootstrap
  path is untouched (no regression to tonight's healthy run or other lanes).

## Verification & spend gates

- No paid pod without explicit user go + `scripts/gpu_spend_guard.py` 0-live
  confirm before **and** after; one pod, per-minute watch; `pending_teardown` +
  `build_teardown_proof(status_source="provider_api")`.
- Snapshot verification: after push, a fresh pod (or the same pod) does
  `docker pull` + the `--build-time` healthcheck + a **1-step** closed-loop
  smoke (real GR00T requery, tiny resolution) to prove the sealed image runs
  end-to-end before it is trusted as the default.
- Image ref promoted to the secrets file only after that smoke passes; prior
  ref kept as `.bak`.

## Tests (hermetic, no spend, `pytest` fast lane)

- Image spec/manifest builder: env keys, paths, checkpoint targets, versioned-tag
  refusal, `raw_secret_values_recorded=false`.
- Launcher wiring: image-ref resolution order; sealed-gate on ⇒ bootstrap
  skipped; sealed-gate off ⇒ legacy path unchanged; missing confirmation ⇒
  blocked with a named blocker.
- Snapshot script arg/So-C contract via a dry-run (`--print-plan`) that emits the
  layer path-list + tag without touching a pod or the network.

## Risks & mitigations

- **Snapshot captures pod cruft / wrong base.** Mitigate: tar an explicit
  allow-list of paths; `crane append` onto the pod's exact base digest; verify by
  fresh-pull smoke before trust.
- **Weights bloat pull time.** Mitigate: layer split for parallel pull; document
  size in the manifest.
- **Two-env dep conflicts (OSCAR vs GR00T).** Already isolated by the separate
  `/opt/gr00t-venv`; healthcheck imports both.
- **Code freeze in image.** Runtime `pip install -e <bundle>` override kept for
  dev; sealed default for paid speed.

## Out of scope

Isaac-render lanes (separate `blueprint-isaac-eval-worker` split image);
semantic task-success claims (the image proves build/runtime readiness only, not
inference quality — same claim boundary as the existing sealed carrier).
