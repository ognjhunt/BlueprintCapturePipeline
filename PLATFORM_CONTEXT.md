# Platform Context

The block below is generated from `doctrine/platform-context.md` by
`python3 scripts/sync_shared_doctrine.py --write`. Edit the fragment, not this
file. It covers the free beta and Arm Decision Proof v1.

<!-- SHARED_PLATFORM_CONTEXT_START -->
## Shared Platform Doctrine

Owner direction, 2026-10-28: simplify. This short block replaces the earlier
long doctrine; git history keeps the old text for reference only.

Owner direction, 2026-11-12: the job packet is the product. It replaces the
direction of 2026-11-11 about what the site's answer is. Where any other
document disagrees with this block, this block is right and the other document
is out of date.

### What Blueprint Does

A site films one recurring job once. Blueprint turns the video into a **job
packet** that any robot deployment company can evaluate without visiting,
sends it to deployment companies Blueprint has approved, and tells the site
who will take the job and on what evidence.

Our edge is the site side: real jobs arriving as videos. The reconstruction
and simulation technology and the openly available robot software are not our
edge; others have them. The simulation is how a job becomes testable without a
visit. It is a means, not the product.

Who it is for:

- **The site** is the user and pays nothing. It sends one video and gets a
  first look on paper, a simulation of its site with the job in it, and the
  responses of deployment companies: who will take the job, who will not, and
  why.
- **The deployment company** is the customer. It gets qualified jobs it would
  not otherwise see, in one format, with a simulation attached, so it can
  decide without a site visit. How deployment companies pay (per introduction,
  or a fee on a signed deal) is not decided. No price is stated anywhere.
- Later, and only if they show they want it: large multi-site buyers (an
  independent comparison across deployment companies) and robot-software
  makers (real commercial jobs to test against, only with the sites' rights).

The job packet has one format for every job:

1. The site's own sentence for the job, and its constraints.
2. The video, with the site's rights and permissions recorded.
3. The first look on paper: which kinds of robot most likely fit, from
   published specifications and a cited track record.
4. The simulation of the site, with the job's object in it.
5. The success rule: what counts as the job done.
6. Where one exists, the result and a clip of Blueprint's own early trial.

The site's answer is who will take the job, and on what evidence: the
responses of deployment companies. Until responses exist the site sees the
first look and the simulation, and is told plainly where the job stands: not
yet sent, sent and waiting, or no response yet. Nothing promises a response
that has not been given.

Blueprint's own trial of openly available robot software stays as supporting
evidence inside the packet. It is labelled **an early trial of openly
available robot software**. It is not the site's answer, and it does not
predict what a deployment company's own robot would do. A zero is a valid
result of the trial. A scripted run only checks Blueprint's own simulation.

A site's video and simulation go to a deployment company only with the site's
permission, and only to companies Blueprint has approved. Every approach to an
outside company needs the owner's go-ahead first.

As of 2026-11-12 no recorded permission covers sending a site's video or the
simulation of its site to a deployment company: every statement a site has
agreed to says they stay private. Until the owner sets a new permission and
its wording, a deployment company receives a summary of the job only (the
site's sentence, the first look, the success rule and the early trial's
result), without the site's name or address. Nobody but the owner changes the
wording of anything a site agrees to.

V1 is by hand. One video becomes a packet; the Pipeline is the packet factory.
Blueprint carries the packet to five to ten deployment companies and collects
their responses, and the site sees them.

Paused (keep the code, start no new work): hardening the automatic test for
any kind of job, for several objects and for unattended scale; standing
re-tests of every robot on every open job.

Later, only after deployment companies show they want it: a deployment company
runs its own software on the site's job in Blueprint's simulation; each new
robot release is re-run against the open jobs; simulated results are compared
with real outcomes across companies.

If sites are hard to reach, the fallback is to offer the video intake to
deployment companies for their own leads and, with the site's consent, pass on
the jobs they decline.

The long-term vision is an independent proving ground where every job is
tested against every robot: a site posts once and is told the moment a robot
can do its job, and the backlog of jobs is a live map of real demand. It is a
direction, not current work.

Words to use: "deployment company" (not vendor, integrator or OEM), "job
packet", "first look", "simulation of your site", "early trial", "responses".

The beta is free for sites. Blueprint pays for preparation and evaluation
inside a fixed per-site budget. No job is refused by task kind.

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

- Don't make things up: no invented results, figures, customers, readiness,
  approvals or responses. Unknown stays unknown. A page, an email or a packet
  says only what is true.
- The first look on paper and the early trial's result are bounded estimates.
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
