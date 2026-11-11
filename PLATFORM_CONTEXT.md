# Platform Context

The block below is generated from `doctrine/platform-context.md` by
`python3 scripts/sync_shared_doctrine.py --write`. Edit the fragment, not this
file. It covers the free beta and Arm Decision Proof v1.

<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

Owner direction, 2026-10-28: simplify. This short block replaces the earlier
long doctrine; git history keeps the old text for reference only.

Owner direction, 2026-11-11: the site gets one answer, and the answer becomes a
deployment opportunity. Later the same day the owner settled what the answer
is: the result of Blueprint's own simulation test of AI robots on the site's
job. The 2026-10-28 direction stands except where this changes it.

### What Blueprint Does

A site sends a video of one recurring job. Blueprint reads the video, turns it
into a simulation of that site, and runs the top AI robots on the job in it
itself. An AI robot here is a robot together with the learned, camera-driven
software that controls it. Each gets about ten tries, with the set-up varied
between tries. The site's answer is which did best and how far that result can
be trusted. A pre-programmed (scripted) motion is never one of the contenders;
Blueprint uses one only to check its own simulation.

Before a test result exists, and when no test can run, the answer is the first
look on paper: real robot models checked against the job by their published
specifications, with the robot that most likely fits, a runner-up, what was
ruled out and why, who already uses those robots for work like it, and how sure
Blueprint is. The paper check also decides what is worth entering in the test.

The site can then share an anonymised job card, with the answer, with
deployment companies Blueprint has approved. Blueprint works alongside the
company that takes the job on; the site and that company agree any pilot
directly. Blueprint does not rely on robot teams to run evaluations; a
technology provider supplies access or an integration only when Blueprint
cannot run a policy itself.

The journey is four steps:

1. Email and video of the job. Blueprint drafts the brief from the video; the
   site corrects it if needed.
2. A first look on paper, as soon as the video is read: a fit check against
   published specifications and a cited track record. It never waits on the
   simulation.
3. The simulation test: Blueprint prepares the site simulation and runs the AI
   robots in it. The site sees which did best, with the published accuracy of
   the method and its conditions, or a plain statement that none finished or
   that there is no clear winner, or "held — we can't simulate this kind of
   job yet", which is counted as demand for what to build next.
4. A deployment company: the site chooses whether to share its job card.

The beta is free. Blueprint pays for preparation and evaluation inside a fixed
per-site budget. No job is refused by task kind.

### How Capture Works

One capture path is in use (owner, 2026-11-11): a person who works at the site
films the job with their own smartphone camera and uploads it in the browser,
through the WebApp's onboarding and site-task flow. No capture app is in use.
Nobody else films for a site: there are no outside capturers, no Blueprint
filming visits, no capturer marketplace and no capturer payouts.

The `BlueprintCapture` repository is kept for one reason: its cloud function
`extractFrames` still receives every website upload and writes the delivery
records the Pipeline starts from. The phone apps in that repository are not in
use. Documents and code that describe the app, capturers or payouts are
history, not current practice.

### The Only Hard Stops

Fail closed only for:

- **Privacy:** the site's consent is respected, including its withdrawal.
  Faces are blurred in the frames sent to World Labs; people in captures are
  not otherwise screened or blurred.
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
- The simulation test's result and the first look on paper are bounded
  estimates.
  They never certify physical performance or safety. A stated accuracy carries
  its source and its conditions.

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
