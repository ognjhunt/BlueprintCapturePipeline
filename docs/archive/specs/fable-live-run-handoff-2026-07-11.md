# FABLE live-run handoff — 2026-07-11 (Fable → whoever finishes the episode)

> Archived cloud-agent handoff snapshot. Not a current operator instruction.

Written after standing down on paid infra to avoid colliding with the
concurrent agent that is driving the same branch/DO account. This records the
exact state so the canary + microwave episode can be completed without
re-deriving anything.

## Verified state

- **Code (FABLE-001–007): complete** on `origin/codex/fable-001-007-closure`,
  hosted CI green historically. Details in
  `docs/specs/fable-remediation-and-live-readiness-2026-07-11.md`.
- **Sealed image (FABLE-008): built and pushed.**
  `docker.io/nijelhunt/blueprint-groot-oscar-eval:20260711-cu128-amd64`
  → `sha256:b15624dba322f45b68eb7ad952dc05ff2c7caffe76fddff9e8e28916243dbfbc`.
  Build manifest: `output/groot_oscar_image_build_20260711/final_7ad011de/`,
  `status: completed`, `pushed: true`.
- **All seven Dockerfile build blockers are fixed** and committed on the branch
  (they had never been run against the pinned Isaac-Sim base): `USER root`;
  PEP-668 `--break-system-packages uv` + `python3-pip`; `oscar_wam_gpu_image`
  rename; `sudo` for the WBC install script; pinned TensorRT
  `10.4.0.26-1+cuda12.6` + `FindTensorRT.cmake` nvparsers patch; the full
  TensorRT dependency-closure pin; and the checkpoint prefetch running under
  `/opt/oscar-venv/bin/python`.
- **Spend: ~$0.66**, two build droplets, both destroyed with API-confirmed
  teardown proofs. **Zero live droplets** (DO API authoritative), zero billing
  risk. Ledger: `output/groot_oscar_image_build_20260711/build_droplet_spend.json`.

## The one blocker before a *clean* identity-gated episode

The pushed image was built from a **dirty tree**:
`source_commit: 7ad011de` **+** `source_dirty_patch_sha256: 1bcd3e1a...`
(non-empty → uncommitted local changes at build time, likely the in-flight
`consent_normalization` refactor).

The FABLE-005 pre-allocation identity gate
(`g1_kitchen_pre_allocation_identity.py`) requires
`attempt source identity == image source identity == live host identity`. A
dirty `source_dirty_patch_sha256` cannot be reproduced from a clean checkout,
so an attempt bundle built from a committed SHA will (correctly) refuse this
image. **For a legitimately clean, identity-bound closure, rebuild the image
from a clean checkout of a committed HEAD** (empty-diff dirty patch), then build
the attempt bundle from that same SHA.

If a dirty-source run is acceptable for a first smoke, the image is usable as-is
but the closure's identity rows will not be a clean pass — do not claim a clean
identity-bound success from it.

## Runbook to finish (single operator only — do not run two agents)

1. Clean rebuild: `scripts/build_push_groot_oscar_closed_loop_image.sh` from a
   clean checkout of the committed branch HEAD (needs ~120 GiB + docker; DO
   `s-8vcpu-16gb-amd` @ $0.167/hr works — add 24 GiB swap for the WBC C++
   compile, and run the build under `setsid` so a torn-down SSH session can't
   cancel it). Resolve the pushed digest with
   `docker buildx imagetools inspect`.
2. GPU canary on that exact digest (DO `gpu-6000adax1-48gb`/`gpu-l40sx1-48gb`
   @ $1.57/hr, region atl1): run the real sealed healthcheck `--require-cuda`,
   the fast RTX preflight, and the 640x480 review-renderer canary on ONE
   allocation; then assemble `worker_image_runtime_evidence` via
   `g1_kitchen_worker_image_evidence.py` and tear down with proof + zero
   inventory. Staged runbook: `scratchpad/canary_remote.sh`.
3. Regenerate the strict bundle: `scripts/prepare_strict_g1_kitchen_bundle.py`
   with the new evidence + digest-pinned image ref. Frozen task inputs are in
   `output/kitchen_random_task_e2e_20260710T131557Z/`: selection
   `random_task_selection_reroll_002.json` (microwave_door), scenario
   `selected_isaac_scenario_attempt_014.json` (stance `[-1.229635, 1.471274,
   0.84]`, yaw `3.141593`), `scene/kitchen_asset_inventory_reroll_002.json`,
   kitchen assets from `attempt_023_.../kitchen_assets.zip`, start frame from a
   prior payload bundle.
4. Launch: `run_groot_oscar_digitalocean_closed_loop_job` with `--allow-paid`,
   `--max-spend-usd`, and `--registry-image-evidence-file`. The pre-allocation
   identity gate runs between the spend guard and capacity preflight; then
   canary→asset gate→scene→baseline→GR00T→controller/FK→transitions→media→
   closure.

## Known truthful-blocked rows even in a perfect episode

The strict forward/inverse action-recovery scorer and the external
semantic-review service are not configured, and the Ed25519 attestation
signing keys / closure pins are not provisioned. Those closure rows will be
truthfully `blocked`; that is expected and honest (see FABLE-010). The episode
still proves the live geometry / baseline / GR00T / controller / transition /
attested-media chain.

## Coordination note

A concurrent autonomous agent (the codex connector) shares this branch, DO
account, and scratchpad; it built+pushed the image and reaped the build box via
`scratchpad/build_droplet.py`. Two agents driving paid infra collide (a build
was cancelled when the box was torn down mid-run). **Only one operator should
drive the paid canary+episode.**
