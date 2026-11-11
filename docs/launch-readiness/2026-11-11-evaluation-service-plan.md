# The evaluation service: standing tests, cheap policy additions, hosted policies

Plan, 2026-11-11. Owner direction of the same day: robot policies will become
like language-model APIs, nearly all accessible and testable. Blueprint's job is
to be the evaluation service for them on real sites' jobs. We test each policy as
its maker released it. We do not train anything.

The facts behind this plan were read from the code and the host on 2026-11-11.
Where something has never been measured or could not be read, it says so.

## Where we stand

- 70 learned-policy rows in the catalog. 15 can run. 14 are blocked by limits in
  our own runner. 41 wait on an outside party (10 of those are hosted).
- Two robots have policies we can run: the DROID Franka arm and the Unitree G1.
- A clean test of one policy on one job takes 101–160 minutes and costs
  $1.40–$1.70 of GPU. The cap per policy is $4.00.
- Every policy that has run on a site's job so far scored zero. That is a valid
  result: the policy, as released, could not do that job.

## Plan 1. Every job is a standing test

When a policy is released, run it on every stored job, and tell the site when the
answer changes.

**What exists**

- A catalog or registry change already gives every live job a new campaign.
- A plan can be limited to chosen policies (`candidate_ids`).
- Each campaign is published as its own run on the job page.

**What is in the way**

1. The job's window. The WebApp stops every run after the preparation grant
   expires, plus at most seven days. A standing test needs its own authority.
2. Money is per site ($30), and it is the preparation money. The one finished
   job has $3.37 left.
3. A re-plan runs every runnable policy again, including ones that already ran.
4. A finished job goes cold: the launch run after two days, the scene workspace
   seven days after the job ends. No door operation restores a launch run. The
   work disk is below its floor.
5. The page shows only the newest campaign, and no email exists for a second
   result.
6. The site agreed to "run candidate robot policies in it". Whether that covers
   tests months later is not settled.

**What to build, in order**

| Step | What | Where | Needs |
|---|---|---|---|
| 1.1 | "Test policy X on job Y": a caller over the existing plan, queue and settle functions, under a new campaign id | Pipeline | nothing new |
| 1.2 | Never test the same pair twice: record policy version, scene revision and harness revision per job; a catalog change plans only new pairs | Pipeline | nothing new |
| 1.3 | Restore on demand: a door operation over the existing restore function for launch runs | Pipeline, door | a host operation |
| 1.4 | One list per job: the page merges campaigns on the same scene revision, shows when each was tested, and flags a result made under a different harness revision | WebApp | nothing new |
| 1.5 | Tell the site only when the headline changes (a first finisher, or a new leader) | WebApp | owner: yes or no |
| 1.6 | A standing-test allowance: a cap per policy per job and a monthly total, outside the seven-day window, stopped at once by withdrawal or run closure | both | owner: money |
| 1.7 | Consent: new jobs agree to "keep testing new AI robots on this job"; existing jobs are asked, not assumed | WebApp | owner: rights |

**Cost.** About $1.50 per policy per job, plus a restore when the job is cold
(cost and time of a restore have never been measured). Twenty stored jobs and
four new policies in a month is about $120. The account month is $200 and stood
at $147.63 on 2026-11-11.

## Plan 2. Adding a policy is cheap

**What exists**

- The last addition (MolmoAct2) took about 10 new files and edits to about 18
  shared Pipeline files and 3 WebApp files. Its first scored result cost $4.24,
  including two failed paid attempts. One policy cost $5.92 and never completed.
- No interface is shared by most runnable rows.
- Two HTTPS contracts exist (`openpi_droid_https.v1`, `blueprint_policy_https.v1`).
  With them a new policy is a registry entry and no new code. They have never
  run in screening, and only one remote policy is allowed per GPU session.

**What to build, in order**

| Step | What | Needs |
|---|---|---|
| 2.1 | Make the HTTPS contract the one standard connection. A policy runs behind a small server that speaks it, whether a provider hosts it or we run it in a container beside the simulator on the same rented GPU. Prove it with a policy we already run (π0.5): same job, same tries, native path against the contract, and compare | one paid run, about $2–4 |
| 2.2 | Unblock the 7 joint-velocity checkpoints. A velocity-to-position adapter already exists in another path; the runner refuses them only by its allowlist, checkpoint prefix and frozen checkpoint list | first runs about $2–6 each; owner: count them beside the others? |
| 2.3 | Bring the 4 rows that need a newer openpi version in as containers on the standard connection, so the 4 openpi policies already running keep their identity | after 2.1 |
| 2.4 | A first-run check for every new policy: two tries on a stored reference job, paid from an onboarding allowance, before any site's money is spent | owner: money |
| 2.5 | Policy names come from one mirrored catalog file, so adding a policy needs no WebApp code | nothing new |

Not now: the two rows whose checkpoint pairing is unstated, and the two-arm TRLC
policy (it needs a two-arm job, and none exists).

## Plan 3. Hosted policies

**What exists**

- The hosted path is complete in code: a price per request, a request limit, a
  bounded charge, and a client for one provider. No hosted policy is set up.
- Each call sends two 224×224 pictures drawn from the simulation, the arm's
  joint and gripper state, and the task sentence. Never the site's video.

**What is in the way**

1. Rights. The site's agreement does not clearly cover it. The privacy annex
   expects rendered views to go to a robot team's policy, but limits sharing to
   the listing or a written agreement, and no policy provider is a listed
   processor.
2. Money. API fees count against the site's $30 only. The $200 account month and
   the spend guard never see them.
3. An outage is scored as the policy failing.
4. The provider's token is copied onto a rented third-party GPU.
5. No provider has given us a contract or terms.

**What to build, in order**

| Step | What | Needs |
|---|---|---|
| 3.1 | API fees appear in the account month and the spend guard | nothing new |
| 3.2 | A provider error or timeout makes a try "not counted", never a failure | nothing new |
| 3.3 | Calls go through our own relay, so a provider's token never sits on a rented GPU | nothing new |
| 3.4 | Privacy Policy, processor list and the start form say that drawn views of the simulation go to AI providers under terms that forbid any other use; existing jobs are asked | owner: rights |
| 3.5 | The first provider: account, terms, data use, billing | owner: outside contact |

Steps 3.1–3.3 need no outside party: the container from step 2.1 stands in for a
provider.

## Order

1. Step 2.1. It is the base of Plans 2 and 3.
2. Steps 1.1, 1.2, 1.3.
3. Step 2.2.
4. Steps 1.4, 1.5, then 1.6 and 1.7 once the owner has decided.
5. Steps 3.1–3.3, then 3.4 and 3.5 with the owner.

## Decisions only the owner can make

1. The standing-test allowance: a cap per policy per job, a monthly total, and
   its place beside the $200 account month.
2. Consent for later tests: the wording for new jobs, and asking existing jobs.
3. Whether a site is emailed only when the headline changes.
4. Whether results from the joint-velocity adapter are counted beside the others.
5. An onboarding allowance for a new policy's first runs.
6. Hosted policies: which provider to contact first, who pays API fees, and the
   Privacy Policy and Terms changes.
7. Keep finished jobs ready on disk, or restore them on demand.
8. Licences: the Gemma terms (openpi), and MolmoAct2's "intended for research and
   educational use".

## Not in this plan

Checking a simulation ranking against a real robot, and conventionally
programmed robots as contenders. The owner set both aside on 2026-11-11.
