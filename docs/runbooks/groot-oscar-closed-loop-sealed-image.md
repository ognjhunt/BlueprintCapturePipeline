# Runbook: sealed `blueprint-groot-oscar-eval` image

Turns the 40-70 min/pod hand-bootstrap of the GR00T x OSCAR closed-loop lane
(`oscar_isaac_closed_loop_eval` + `groot_sonic_policy_endpoint`) into
`docker pull` + go. Design: `docs/superpowers/specs/2026-07-06-groot-oscar-eval-worker-image-design.md`.

## What the image contains

Base: `blueprint-oscar-wam@sha256:b0f3f675…` (torch 2.10.0+cu128, cuDNN, TE, OSCAR
source + worldsim deps). Added last mile:

- `/opt/gr00t` (Isaac-GR00T `@e5749287`) + `/opt/gr00t-venv` (py3.10 `uv` venv, `gr00t` installed)
- `/opt/wbc` (GR00T-WholeBodyControl)
- `/opt/blueprint/ckpts/sonic` (`LucaFrat/groot-bs16`) + `/opt/blueprint/ckpts/oscar` (`zywu2115/OSCAR-2B@c9781ffa`) — **baked**
- main env: `blueprint_pipeline` + `mujoco pyzmq msgpack-numpy imageio pillow`, `libosmesa6`
- baked env: `MUJOCO_GL=osmesa`, `PYTORCH_ALLOC_CONF=expandable_segments:True`, `PYTHONPATH=/opt/OSCAR`, GR00T paths, `…_SEALED_IMAGE_CONFIRMED=true`

Claim boundary: proves build/runtime readiness only — not provider startup, GR00T
inference, WAM quality, or task success.

## Spend discipline (read first)

- No paid pod without explicit go. Run `python scripts/gpu_spend_guard.py` and
  confirm the expected live pods **before and after**.
- The snapshot operates on a pod you already own; it does not launch/terminate.
- Keep the `pending_teardown` + `build_teardown_proof(status_source="provider_api")`
  discipline for whatever pod you use.
- Secrets stay file-based in `~/.blueprint-secrets/`; never echo them. The PAT is
  piped to `crane auth login … --password-stdin`.

## Build — Path A (PRIMARY): crane-snapshot a healthy pod

Freezes tonight's exact, proven environment with no re-derivation.

```bash
# 0. hermetic dry run — prints the snapshot layer plan, no pod touched
./scripts/snapshot_groot_oscar_eval_pod.sh --print-plan

# 1. confirm the target pod is healthy (GR00T server up, run finished/idle) and
#    note its base image ref (what it was created from) + ssh coords.

# 2. snapshot (needs go). Weights already resident on the pod → fast.
BLUEPRINT_ALLOW_GROOT_OSCAR_SNAPSHOT=true \
BLUEPRINT_GROOT_OSCAR_SNAPSHOT_SSH="root@<host> -p <port> -i $HOME/.ssh/id_ed25519" \
BLUEPRINT_GROOT_OSCAR_SNAPSHOT_BASE_IMAGE="<pod-base-image-ref>" \
BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF="docker.io/nijelhunt/blueprint-groot-oscar-eval:20260706-cu128-amd64" \
BLUEPRINT_DOCKER_PAT_FILE="$HOME/.blueprint-secrets/docker_pat" \
./scripts/snapshot_groot_oscar_eval_pod.sh
```

The script (idempotently) relocates the ephemeral `/workspace/*_ckpt` checkpoints
into `/opt/blueprint/ckpts/*`, stamps `/opt/blueprint/groot_oscar_sealed_env.sh`,
discovers the pod's `site-packages`, installs `crane`, logs in (PAT via stdin),
and FIFO-streams a layer of `/opt/*` + `site-packages` + `/usr/local/bin` onto the
pod's base image. Set `BLUEPRINT_GROOT_OSCAR_SNAPSHOT_TRIM_TORCH=true` to drop
base-duplicated torch/nvidia dirs from the layer if the base already carries them.

## Build — Path B (fallback): reproducible clean build

From the pinned base; re-derives the venv + re-pulls checkpoints. Use on any
amd64 Docker host with ≥120 GiB free (`BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_MIN_FREE_GIB`).

```bash
BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_IMAGE_REF="docker.io/nijelhunt/blueprint-groot-oscar-eval:20260706-cu128-amd64" \
BLUEPRINT_ALLOW_GROOT_OSCAR_CLOSED_LOOP_IMAGE_PUSH=true \
BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_HF_TOKEN_FILE="$HOME/.blueprint-secrets/hf_token" \
./scripts/build_push_groot_oscar_closed_loop_image.sh
```

## Verify before trust (mandatory)

Whichever path built it, on a fresh pod:

```bash
docker pull <image_ref>
python3 /opt/blueprint/groot_oscar_closed_loop_image_healthcheck.py --require-cuda   # exit 0
# 1-step closed-loop smoke (real GR00T requery, tiny resolution):
source /opt/blueprint/groot_oscar_sealed_env.sh
python -m blueprint_pipeline.groot_oscar_closed_loop_image --print-launch-plan --steps 1   # then run the two commands it prints
```

Only after the smoke passes, promote the ref:

```bash
cp ~/.blueprint-secrets/groot_oscar_closed_loop_image_ref{,.bak_$(date -u +%Y%m%d)} 2>/dev/null || true
printf '%s\n' "<image_ref>" > ~/.blueprint-secrets/groot_oscar_closed_loop_image_ref
```

## Launch with the sealed image

Once the ref file is set, a paid closed-loop pod is `docker pull <ref>` then the
plan from:

```bash
export BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_SEALED_IMAGE_CONFIRMED=true
python -m blueprint_pipeline.groot_oscar_closed_loop_image \
  --print-launch-plan --steps 3 --task-prompt "open the fridge" \
  --start-frame /workspace/initial_policy_frame.png --route-file /workspace/route.json \
  --output-dir /workspace/out
```

`--print-sealed-contract` returns exit 1 with a named blocker
(`missing_image_ref`, `image_ref_must_be_versioned`, `sealed_image_not_confirmed`,
…) whenever sealed mode is not fully configured — in which case fall back to the
legacy runtime-bootstrap recipe (`run_t4.sh`).

### Direct DigitalOcean launcher

Use this path when the sealed image has been pushed and the remaining blocker is
DigitalOcean GPU capacity. Prepared mode is local-only: it validates the sealed
contract, writes the input bundle and launch plan, and does not stage to object
store or call the DigitalOcean capacity API.

```bash
blueprint-run-groot-oscar-digitalocean-closed-loop \
  --start-frame <initial_policy_frame.png> \
  --route-file <route.json> \
  --task-prompt "Open the dishwasher door; if the dishwasher is already open, close the dishwasher door." \
  --seed-provenance-file <seed_provenance.json> \
  --out-dir <run_dir>
```

When capacity should be checked and a real droplet may be launched, add
`--allow-paid --max-spend-usd <budget>`. The launcher fails closed in this order:

1. sealed image contract and launch plan
2. spend guard
3. read-only DigitalOcean GPU size/region capacity preflight
4. object-store staging
5. droplet launch with pending-teardown record
6. worker collection and provider-API teardown proof

The worker runs the baked healthcheck, starts the GR00T policy server, then runs
`oscar_isaac_closed_loop_eval` at native OSCAR resolution with
`--require-fresh-learned-policy-requery` and `--stop-on-task-completion`. The
`--steps` value is a safety cap, not a fixed frame count.
