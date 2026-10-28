# Platform Context

The block below is generated from `doctrine/platform-context.md` by
`python3 scripts/sync_shared_doctrine.py --write`. Edit the fragment, not this
file. It covers the free beta and Arm Decision Proof v1.

<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

Owner direction, 2026-10-28: simplify. This short block replaces the earlier
long doctrine; git history keeps the old text for reference only.

### What Blueprint Does

A site sends a video of one recurring job. Blueprint turns the video into a
simulation of that site, runs the candidate robot policies and embodiments in
it itself, and shows the site a bounded estimate of what works. Blueprint then
works alongside a deployment company that can deliver the winning approach on
the real site. Blueprint does not rely on robot teams to run evaluations;
a technology provider supplies access or an integration only when Blueprint
cannot run a policy itself.

The journey is four steps:

1. Email and video of the job.
2. Blueprint drafts the brief from the video; the site corrects it if needed.
3. Blueprint prepares the site simulation and evaluates the candidates.
4. The site gets the result: a simulation estimate, or "held — we can't
   simulate this kind of job yet", which is counted as demand for what to
   build next.

The beta is free. Blueprint pays for preparation and evaluation inside a fixed
per-site budget. No job is refused by task kind.

### The Only Hard Stops

Fail closed only for:

- **Privacy:** people in captures are screened, and consent is respected.
- **Rights:** capture permissions, revocation, and dataset or model licenses.
- **Money:** spend stays inside its budget, and paid resources go through the
  canonical allocator.

Everything else keeps going. When something is missing, ambiguous or
unverified, flag it where the person can see it and continue the same job.
Ask at most one question at a time, inside that job. Fixing a failure means
fixing its cause; do not add a new gate, approval step, review stage or
readiness check unless it protects privacy, rights or money.

### Two Truths

- Don't make things up: no invented results, figures, customers, readiness or
  approvals. Unknown stays unknown.
- Simulation results are bounded estimates. They never certify physical
  performance or safety.

### Arm Decision Proof v1

Arm Decision Proof v1 (`arm-decision-proof-v1`) is a separate evidence
program: it tests whether a simulated decision between two frozen candidates
agrees with held-out physical trials on one fixed-arm workcell. Its protocol
lives in `BlueprintCapturePipeline/docs/arm_decision_proof_v1/` and applies
only to that program. It alone may make physical-validation claims; beta
results never enter its evidence. The historical program identifier
`free-beta-evaluated-pilot` stays readable on existing beta records.
<!-- SHARED_PLATFORM_CONTEXT_END -->

## Pipeline Embodiment Rule

The admitted design partner's fixed arm and working interface determine the Arm
Decision Proof embodiment. Franka/DROID assets may serve as development-only
defaults before admission, but they cannot substitute for the partner binding.
This rule is for Arm Decision Proof only; beta evaluations may use any
embodiment Blueprint can run, humanoids included.
