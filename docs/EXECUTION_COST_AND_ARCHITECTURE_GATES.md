# Execution cost, GPU policy, and architecture/reachability gates

This document covers a second group of changes, following
`EVALUATOR_ATTRIBUTION_AND_PUBLIC_ANCHOR.md`. They share a theme: constraints
that were written for one workload had quietly become platform-wide
invariants, and several of them were costing throughput, hardware access, or
honesty about what could ever be satisfied.

Nothing here measures a robot, runs a world model, or upgrades a claim.

## 1. The cold-start tax

`make_local_oscar_subprocess_generate` spawned a full `torch.distributed.run`
invocation for **every closed-loop step**, so a 2B checkpoint was read from disk,
moved to device and torn down once per generated observation. A 300-step rollout
paid 300 model loads to produce 300 frames — and `make_oscar_per_step_wam_backend`'s
own docstring already described the intended design as "a thin call into a
persistent OSCAR-2B pod".

This is measurement-limiting, not merely wasteful: rank-fidelity confidence is
bought with rollouts, and generation cost per step sets how many policies a
campaign can afford to cover.

`blueprint_pipeline.oscar_resident_worker` keeps one worker alive across a
rollout, speaking line-delimited JSON:

- the worker loads the checkpoint once and announces readiness with its load
  time and device identity;
- each request returns the clip plus its own warm timing;
- the client records cold-start and warm-step latency separately and writes
  `oscar_resident_worker_throughput.json`.

Failure handling is loud by design. A dead or desynchronised worker fails the
step **closed** rather than falling back to per-step spawning, because a silent
fallback would restore the cold-start cost while continuing to report
resident-path timings. Restarts must be explicitly budgeted
(`--oscar-resident-worker-max-restarts`, default 0) and are counted, so a
crash-loop appears as a restart count instead of hiding inside an average.
Out-of-order responses and replayed runtime result ids are rejected: a
desynchronised stream would attribute one step's output to another step's action.

Enable with `--oscar-resident-worker`. The worker is torn down and its report
written in a `finally`, so a run that dies halfway still leaves its timings —
which is exactly when they are worth reading.

## 2. GPU selection is a workload policy, not a global rule

`DISALLOWED_ISAAC_GPU_KEYWORDS = ("A100", "H100")` was applied as an
unconditional filter term to **all** Vast offer selection, while only one call
site was Isaac-scoped. A pure generation or training campaign was therefore
barred from exactly the hardware it needs.

GPU choice is now an explicit, named policy:

| policy | denylist | intent |
| --- | --- | --- |
| `isaac_rendering` | A100, H100, H200, B200, GB200 | RTX rendering requires RT cores |
| `generation` | none | compute/VRAM bound; constrain with `min_gpu_ram_mb` |
| `training` | none | compute/VRAM bound |
| `open` | none | no workload-specific constraint |

Resolution rules: an explicit policy wins; otherwise the workload is inferred
from `prefer_isaac_rt`, so existing Isaac callers are unchanged. An **unknown
policy name fails closed to the Isaac policy** rather than silently opening
selection. Inline policies (`denied_gpu_keywords` / `allowed_gpu_keywords`) are
accepted for one-off cases. The resolved policy is recorded in the offer
selection manifest.

Blackwell and large-VRAM parts are now first-class: `RTX PRO 6000` and
`RTX 5090` are recognised RT candidates (they keep RT cores, so they are
eligible for Isaac rendering *and* generation), and the VRAM table gained
RTX PRO 6000 Blackwell 96GB, H200, B200 and RTX 5090.

Because rate ceilings were tuned for short Isaac smoke attempts, a generation
campaign inheriting them would match no offers at all. Each policy therefore
carries a recommended envelope (`recommended_max_hourly_rate`,
`recommended_hard_cap_usd`, `recommended_min_gpu_ram_mb`), and
`--gpu-selection-policy` selects one. Rate and cap flags remain explicit
overrides.

## 3. Judge spend governance

Rented GPUs get an hourly rate ceiling, a target spend, a hard cap, a TTL, a
watchdog armed before allocation, and a ledger. Judge inference — also metered,
also billable, also launched from automation — had a single boolean environment
flag.

The asymmetry worsens precisely as graded progress scoring is adopted: a binary
label reads a handful of frames, a 0–5 rubric over a 300-frame episode reads an
order of magnitude more, multiplied by policies × sites × trials.

`blueprint_pipeline.judge_spend_governor` gives judge spend the same shape:

- **policy** — target spend, hard cap, request ceiling, frame ceiling, TTL;
- **ledger** — every reservation and settlement, optionally appended to disk;
- **cohort hard stop** — once the cap is reached every later reservation is
  denied, so overspend is bounded by one in-flight request.

Two deliberate choices. **Prices are never invented**: a policy without
operator-supplied rates cannot price a request, and an unpriceable request is
denied rather than waved through. **Failed requests are still settled**: a
governor that only counts successes systematically under-reports spend.

The graded-progress lane treats an absent policy as a refusal
(`progress_judge_spend_policy_not_configured`) rather than a default-allow.

## 4. The 7-D action invariant is retired

Two independent hardcoded `7`s governed action handling — the normalization
contract and the Cosmos command adapter — while the executing Unitree G1
whole-body action is 78-dimensional. The embodiment the pipeline drives could
not be described by its own action contract.

The fix is not to relax the check. A 7-D delta end-effector vector, a 43-joint
arm/hand chunk and a 78-D whole-body command are different physical objects that
happen to share a Python type, and letting them through one adapter because they
are all arrays of numbers is the failure mode the strict contract was defending
against.

`blueprint_pipeline.action_space_registry` makes dimensionality a property of a
**registered action space** — id, dimension, ordered component names, units and
representation aliases. Callers name the space they intend; an unregistered name
fails closed; a vector is validated against that space's exact layout with the
same strictness the 7-D path always had. The default remains the SC3 space, so
existing callers and blocker strings are unchanged.

## 5. A second registered embodiment

Only one robot profile was ever registered, so nothing exercised the claim that
placement, export and action paths are embodiment-parameterised.
`FIXED_BASE_SINGLE_ARM_PROFILE` is a zero-GPU conformance fixture that differs
on the axes that matter: fixed base rather than legged, one arm rather than two,
7-D delta end-effector rather than 78-D whole-body, external plus wrist cameras
rather than a head rig. It deliberately matches the DROID single-arm family that
public leaderboards evaluate, so the harness can be exercised against an
embodiment whose reference outcomes already exist.

`get_robot_profile` now raises `UnknownRobotProfileError` (a `KeyError`
subclass, so existing handlers keep working) — giving call sites that accept a
job-request-supplied `robot_id` something specific to catch.

## 6. Adoption and admission are different decisions

The RoboWorld admission checklist conflated two things behind one
`awaiting_upstream_release` status:

1. **Admitting the upstream backend** — running released code and weights and
   reproducing published results. Legitimately blocked: nothing to pin.
2. **Adopting the architectural recipe** — building an action-conditioned world
   model from published design ideas on components Blueprint already holds under
   permissive licences.

The second is blocked only by contract text. Blueprint already pins
`Wan-AI/Wan2.1-T2V-1.3B` (Apache-2.0, immutable revision) and uses only its VAE.

`blueprint_pipeline.world_model_architecture_adoption` separates the tracks.
Separating them makes claims **stricter**, not looser: a model built on track 2
is Blueprint-authored, may not use the upstream project's name or reported
metrics, may not use upstream weights or code, and must pass ordinary evaluator
qualification. Authorisation is *to build*, and confers no evaluator standing.

It also records `backend_selection_principle()`: order backends by measured
fidelity, throughput, licence and abstention — not parameter scale. The
catalogue's scale ladder is unsupported by the repository's own recorded
evidence, where the published correlations run inversely to model size.

## 7. Gate reachability

"38 blockers" mixes at least four situations: satisfiable now, awaiting
execution, awaiting upstream release, and **unreachable by construction** — a
defect wearing a status's clothes.

`blueprint_pipeline.gate_reachability_audit` classifies gates by **probing**
rather than asserting: it calls the real validators and scans the real source,
so a repaired gate stops being reported as dead without editing the audit.

Current findings, all reproducible via `blueprint-audit-gate-reachability`:

- `validate_external_study` returns `external_proof_required` for every input,
  and the string `"validated"` does not appear anywhere in that module. So
  `sc3_eval_protocol`'s `public_rank_fidelity_claim_eligible`, `claim_ready`,
  and `eligible_preregistered_external_rank_fidelity` — whose conjunction
  requires that status — are **unreachable by construction**.
- `benchmark_uncertainty.public_rank_fidelity_claim_eligible` and
  `oscar_cosmos_wam_evaluator.full_closed_loop_episode_proven` are emitted as
  literal `False` (the latter at six distinct sites), so no caller can influence
  them.
- The two OOD axis vocabularies disagree (8 frozen SC3 axes versus 5
  decision-grade axes; `appearance` and `viewpoint` appear in neither direction).
  They bind *different* artifacts, so this is reported as `divergent_registry`
  rather than unreachable — the honest label.

`classify_blockers()` splits a blocker list into what waiting could clear and
what it never will.

## Commands

```bash
blueprint-audit-gate-reachability --fail-on-unreachable
blueprint-plan-architecture-adoption --input proposal.json --output plan.json
python -m blueprint_pipeline.oscar_isaac_closed_loop_eval --oscar-resident-worker ...
python -m blueprint_pipeline.vast_provider_adapter --gpu-selection-policy generation ...
```
