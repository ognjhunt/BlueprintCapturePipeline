# MuSHRoom Bakeoff Decision Scorecard — 2026-08-02 (v2)

Point-in-time snapshot. This is the dated successor revision required by the
2026-08-01 scorecard's freeze clause; that document is preserved unchanged at
`docs/MUSHROOM_BAKEOFF_DECISION_SCORECARD_2026-08-01.md`
(sha256 `ab746239758ef3e00505fe2d2d3975ee1e17934f4b543bc5f893637ac7cfb02f`).
No provider output existed for this bakeoff when v2 was frozen.

## Inherited unchanged from v1 (binding by reference)

- Evaluation proxy: compiler-v3 `mushroom_proxy_fea6da5dfeca8e6a`; frozen
  split digest
  `sha256:75c0a00f8b70d05bcbdf406f3266026105348b8d02f4bf25de8965a6b9d3712b`.
- Evaluator pin: `heldout_appearance_evaluation_v2` (windowed SSIM
  Wang-2004 11×11 σ1.5; LPIPS `lpips_alex_v0.1` with the v1-pinned
  checkpoint and backbone digests), rendered through `sealed_camera_render`
  after `provider_splat_import` + `align_provider_reconstruction`, with the
  v1 pre-gate (≥8 candidate views rendered, PSNR vs candidate pixels
  ≥ 15 dB mean).
- Floors (per trajectory, both must pass, never averaged): mean PSNR
  ≥ 16.5 dB · mean windowed SSIM ≥ 0.55 · mean global SSIM ≥ 0.50 · mean
  absolute error ≤ 0.12 · mean LPIPS ≤ 0.55; plus the v1 alignment gates.
- Decision rules and verdict enum: exactly v1 §Decision rules (primary mean
  LPIPS per trajectory; the win margins, conditional-default and
  no-backend-qualified outcomes are unchanged). Existing arms T1, P1, P2,
  P0 (conditional) are unchanged.
- Hidden-view discipline: no provider or trainer receives hidden filenames,
  pixels, cameras, or metrics; 29 held-out + 121 independent-short views
  remain evaluator-only.

## New in v2 — Linux open-trainer arms

| Arm | Trainer | Input | Notes |
| --- | --- | --- | --- |
| G1 | Splatfacto — `nerfstudio==1.1.5` + `gsplat==1.4.0` (DefaultStrategy) | Byte-identical copy of P1's point-seeded COLMAP text dataset block (copied, digest-verified, from the frozen Postshot execution packet) | Apache-2.0, headless, ordinary Linux Vast lane |
| G2 | Splatfacto-MCMC — `nerfstudio @ git+…@50e0e3c70c775e89333256213363badbf074f29d` + `gsplat==1.4.0` (MCMCStrategy, `max_gs_num` 1,000,000) | same | The MCMC strategy field does not exist in the 1.1.5 PyPI release; the exact main commit is pinned and license-reviewed |

Packet: `scripts/build_splatfacto_execution_packet.py` →
`provider_packets/splatfacto/splatfacto_execution_packet.v1.json`
(self-digested, write-once, refuses hidden-path references; arms pin
strategy, seed 42, 30k iterations, environment files
`requirements/splatfacto-arm-g{1,2}.txt`, venvs via
`scripts/setup_splatfacto_venv.sh`). Worker execution receipts must record
exact argv, `pip freeze`, and durations. Poses are fixed from the
point-seeded dataset (`pose_estimation_by_provider: false`); any pose
convention error surfaces at the v1 pre-gate.

Benchmark context only (not evidence for our capture profile): gsplat's
MCMC strategy at 3M Gaussians reports PSNR 29.65 / SSIM 0.89 / LPIPS 0.12 on
Mip-NeRF 360 (https://docs.gsplat.studio/main/tests/eval.html). Whether it
reaches parity on Blueprint captures is exactly what these arms measure.

Budget: no new paid runs are implied by this revision; G1/G2 ride the
already-gated bakeoff budget behind the canonical paid-resource seam.

## Postshot lane status (v2 posture)

Postshot remains a **separately licensed Windows exception lane**, not a
fleet-tooling candidate and not permanently banished either:

- EULA facts: one concurrently used device per license; internet connection
  required; CLI automation gated to the Studio tier (€39/mo). The reviewed
  EULA does not expressly resolve ephemeral cloud-VM use — that is
  procurement ambiguity, not a proven prohibition.
- Procurement action (owner: @ognjhunt): obtain written vendor confirmation
  on cloud-VM/ephemeral-worker use before scaling the Windows lane; live
  terms/pricing/CLI reverification at execution time stays binding (v1).
- Fleet layer: SkyPilot is Windows-free, so the Postshot worker can never
  ride the SkyPilot lane; it stays behind Blueprint's provider-neutral job
  contract on AWS G6/G5 or Azure NVadsA10 Windows as v1 already anticipated.
- Demotion rule: if G1 or G2 reaches parity on our capture profile under
  the v1 floors and decision rules, the Windows worker is demoted to a
  licensed-tool exception lane (or deleted) per the build-on-top audit.

## Arm-entry conditions (all arms, unchanged in substance)

1. Pinned environment digests recorded before training
   (`docs/architecture/isolated-component-license-inventory.md` for G1/G2).
2. Candidate inputs bound by digest to the frozen split; hidden views
   unreachable.
3. Evaluator-only scoring through the v1-pinned evaluator; no
   self-reported metrics enter the decision.
4. Every paid execution passes the canonical fail-closed paid-resource
   seam; teardown proof with `status_source="provider_api"`.
