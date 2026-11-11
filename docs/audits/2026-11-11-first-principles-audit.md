# First-principles audit: does Blueprint answer the site's question?

Read-only audit of `BlueprintCapturePipeline` and `Blueprint-WebApp`, done
2026-11-11. Nothing was deployed, run on the host, spent or sent. It is a dated
evidence snapshot, not roadmap authority; its recommendations wait for the
owner.

The question it tests, in the owner's words: *are we creating a service that
answers the question of a site that wants to know which robot or policy will
most likely work best in real life on its task at its site, in the easiest and
simplest manner possible as of November 2026 — no more or less?* The owner
added a yardstick during the audit: simple is better; the service should be a
thin layer of AI over the best available APIs and frameworks, with little that
is custom.

## Verdict

**Not yet, and not on the current path.**

- **Asking is nearly right.** A site can start a job from one form with no
  account. That part is close to "easiest and simplest".
- **The answer is not yet an answer to the site's question.** Five website jobs
  have been attempted, all on the owner's own footage or stock clips. One
  reached a result. In every scored episode in every site-derived scene, no
  learned policy has succeeded once. No result has ever told two candidates
  apart.
- **Even a good result would answer a different question.** The eight arm
  candidates are public research checkpoints on one research arm. They compare
  policies, not robots, and none maps to a system a site can buy or a company
  it can hire.
- **Nothing ties the number to reality.** Scale, mass and friction are
  estimates. No known-answer check runs in the scene. The program that would
  compare simulated and physical outcomes has no partner and a count of zero.
- **It is far more than the question needs.** The Pipeline is 1.31 million
  lines in 2,505 modules. One job crosses 14 stages and about 15 outside
  services, costs about $50–60 and took about a week and about eight deploys.
- **The simplest credible answer is already inside the product.** A 512-line
  check rates 22 real robot models from 13 makers against the site's measured
  scene by spec sheet, and the WebApp already reads the video and keeps a
  researched registry of robot teams. Today that sits beneath the simulation as
  a secondary section. From first principles it should be the spine, and the
  simulation one line of evidence within it.

## The yardstick

For the site, an answer is worth having only if all five of these hold.

1. **Easy to ask.** Email and a video, nothing the site has to learn.
2. **About options it can actually get.** The things compared are systems a
   site can buy or have deployed, by someone it can contact.
3. **Predictive.** Whatever ranks the options has a known link to what happens
   on a real floor.
4. **Decisive.** One page says which option, how sure, what was ruled out and
   what to do next.
5. **Simple to run.** Fast and cheap enough to serve every invited site, with
   no engineer driving each job.

"No more" means nothing is built or shown that does not serve those five.
"No less" means none of the five is missing.

## Scorecard

| Link | State | Evidence |
|---|---|---|
| 1. Easy to ask | Mostly yes | One form, no account, a private link (`SiteCaptureStart.tsx`, `captureUploadToken.ts:59`). It asks for a typed task, a tick box and a country besides email and video (`SiteCaptureStart.tsx:836-866`). |
| 2. Options it can get | No | 12 of 136 catalog rows can run (`config/beta_policy_candidates.v1.json`). 8 are public checkpoints on the DROID Franka arm with no deployment record; 4 are on the Unitree G1 and have no valid score. |
| 3. Predictive | No | No calibration, validation set or known-answer check on the beta path. Arm Decision Proof: north-star metric 0 of 1, no admitted partner (`north_star_contract.json`, `CURRENT_STATE.md`). |
| 4. Decisive | Partly | A "Chosen in simulation" sentence exists when something succeeds (`website_candidate_selection.py`). The only real result was all zeros and was still headed "Your simulation estimate is ready" (`betaPolicyScreeningContract.ts:421`). The next step shown is "Nothing is needed from you." |
| 5. Simple to run | No | 14 stages, about 15 outside services, at least $51.7 on the one finished job, about 7 days, about 8 deploys (`docs/SYSTEM_MAP.md`; tracker lines 423-448; goal brief line 120). |

## What the record shows

### 1. No result has yet told two candidates apart

Source: `docs/launch-readiness/2026-11-beta-launch-tracker.md` and the WebApp
end-to-end records.

| Job | Footage | End state | Scored successes |
|---|---|---|---|
| Blue container (September) | uploader not recorded | 11 scored episodes on an authored surface, not the captured room | 0 |
| Drawer (September) | owner's clip | blocked, window expired, 16 of 16 paid attempts used | 0 |
| Dishwasher (September) | owner's home | later steps unproven, retired 10-04 | none scored |
| capture-447, tray pick and place | stock clip | terminal `estimate` on 11-06 | 0 of 39 |
| Library aisle, navigation | stock clip | not ended; held on the job ceiling | 0 valid episodes |

- On capture-447, π0.5 scored 0 of 9, GR00T N1.7 0 of 10, MolmoAct 2 0 of 10
  and a π0 variant 0 of 10. Cosmos 3 never loaded. Two more were deferred when
  the $30 site budget ran out (tracker 446-448).
- In those 39 episodes the tray moved once, pushed 0.18 m without a grasp.
- No job has come from an outside site, no invitee list exists, and no
  deployment-company introduction is recorded (readiness plan O-04; tracker
  F-01, F-09).

### 2. The candidates are not things a site can get

- All 8 runnable arm candidates use the same body: the DROID Franka Panda with
  a Robotiq gripper. On arms, "which robot" is never compared.
- All 12 are public checkpoints run as published. None is adapted to the site.
- None of the 8 arm rows carries a deployment record, so a winner would show
  "No recorded deployment partner for this configuration"
  (`website_candidate_selection.py:55-60`).
- The DROID Panda is not among the 16 bodies in
  `deployment_embodiment_support.v1.json`.
- 110 catalog rows name real companies, drawn from public pages rather than
  agreements, and all of those are disabled; 102 rows wait on
  `prepared_robot_policy_registration_missing`.
- There is no runnable mobile manipulator or mobile robot
  (`mobile_platform_not_modeled`).
- The last recorded read of the robot-team registry found 83 teams and no
  supply agreement (WebApp 2026-10-18 audit).

A site does not buy a policy checkpoint. It buys a working system from someone
who will install and support it. That is the unit the answer should rank.

### 3. Nothing ties the number to reality

- **Scale** is `atlas_reference_estimate`: a model's web research on typical
  object sizes (`website_atlas_scene_source.py:38`).
- **Mass and friction** are checked against fixed default ranges or wide
  priors; `website_articulated_mass.py` says "Nothing here is a measurement".
- **Task objects** are authored by a language-model agent.
- **Floors** are filled in flat where the video did not show them.
- **Trials.** At most 10 episodes per candidate. Variation is small (object
  shift up to 3 cm, light ±20%, camera ±2 cm) and is removed altogether for
  fixed objects and for the task-family back end, so the trials are close to
  repeats (`beta_policy_screening_plan.py:348-391`).
- **Choice rule.** The highest point estimate wins if it has at least one
  success. There is no significance test: 3 of 10 beats 2 of 10
  (`website_candidate_selection.py`).
- **Success criteria** are Blueprint's defaults, taken as confirmed when the
  site submits (`website_owner_success_assumption.py`).
- **Harness faults have already changed results.** Until 11-05 every DROID
  policy was told its gripper was closed when it was open; until 11-08 a G1
  falling through the floor counted as a policy failure
  (`beta_policy_screening_plan.py:39-52`).
- **No known-answer check.** The WebApp reads `known_good_control_*` flags
  (`betaPolicyScreeningContract.ts:324-332`); no Pipeline module writes them.

### 4. The method is ahead of the published evidence

A research pass read the sources below on 2026-11-11. The lead did not re-read
the papers dated after June 2026.

- The strongest published agreement between simulation and real trials is on a
  known laboratory arm, in tabletop scenes, with a scale reference in the scan
  and policies co-trained on simulated demonstrations. PolaRiS reports
  correlation near 0.9 for 4 policies on the DROID platform and says that
  without co-training correlation is "too low to accurately rank policy
  performance" (arXiv 2512.16881). REALM needed aligned robot control
  (arXiv 2512.19562).
- No source found supports predicting absolute success rates. The pass found
  no evidence that an uncalibrated phone video of an arbitrary site ranks
  policies the way reality does.
- A study from October 2026 reports that reconstruction defects changed
  outcomes and that the results "do not yet establish a generally reliable
  scene-acceptance rule" (arXiv 2610.17521).
- The newest open generalist model found reports about 40% success on unseen
  tasks on its own robot without demonstrations (NVIDIA GR00T 2 page, November
  2026).
- Sites choose automation through integrators, vendor proofs and paid pilots.
  Two companies already turn a site video into a simulation for sales, with no
  validation shown.

The catalog already uses the PolaRiS co-trained checkpoints, which is the right
instinct. The validated result belongs to that framework's own scan recipe and
simulator, not to a different scene pipeline.

### 5. One job costs about $50–60, a week and about 15 outside services

- **Stages:** 14, of which the reconstruction stage has about 8 paid sub-steps
  (`docs/SYSTEM_MAP.md`).
- **Outside services for one job:** Firebase and Google Cloud storage,
  Pub/Sub, Cloud Functions, Firestore, Render, the DigitalOcean host, Gemini,
  OpenAI, Meta SAM, World Labs, Vast, Backblaze, Hugging Face, NVIDIA NGC and
  Resend.
- **Cost:** at least $25.08 of preparation plus $26.63 of screening on
  capture-447. The readiness plan uses $60 per job plus about $110 a month of
  fixed cost. Against the $200 monthly limit that is one or two jobs a month;
  the planned cohort is 3–5 sites.
- **Time and effort:** about 7 days and about 8 deploys for the one finished
  job. The library job took about 36 deploys in 4 days with no valid episode.
- **Stop reasons:** roughly 1,000 distinct reason codes in the 118 `website_*`
  modules alone (a string scan, approximate).

### 6. The most useful finding so far was free, and the site never saw it

On capture-447 a CPU check flagged, before any paid run, that the place point
was 1.086 m from the arm's shoulder against a reach bound of 0.990 m. The
tracker records that the flag "was not visible beside the results" (lines
471-476). $26.63 of GPU screening then returned zeros that mostly restate it.

## The website

Checked by running the client locally with the repository's own test fixtures
and photographing each page at desktop and phone size, and by reading the copy
in code.

**What works**

- The home page says what the site gets and what to do within a few seconds.
- No account, password or budget is needed. One form leads to a private link.
- Held, new-video and ended states are plain and say whose move it is.
- The phone layout has no horizontal scroll on any page tested.

**Where it asks for more than email and video**

- The form requires a typed task, a "This is the task I want Blueprint to
  evaluate" tick box and a country. The line under the home page button says
  "Start with a video; we draft the job description from it."
- Filming on a phone adds a second recording-permission step on the job page.
- The form sends `publicTaskListing.consent: true` by default, so each job
  becomes an anonymised card for approved robot teams unless the site hides it
  (`SiteCaptureStart.tsx:435-437`). The notice is one line of small print.

**Where the result gives less than the promise**

- The home page promises "One job. A clear pilot plan." The result page gives
  a table of checkpoint names, such as "Cosmos 3 Edge Policy DROID" and
  "π0 DROID joint position 100k (PolaRiS)", with "episodes" and "unscorable".
- A finished estimate ends with "Nothing is needed from you." The only route
  to a pilot is a sentence in the email: reply for an introduction.
- "Blueprint uses this estimate to recommend the next step" has no code that
  delivers a recommendation (`jobManagedEvaluation.ts:302`).
- An all-zero result is headed "Your simulation estimate is ready" and carries
  the same pilot-introduction offer.
- Nothing says how long a result takes. "We usually reply within an hour" sits
  beside a real time of about a week.
- Videos of the robot attempting the job appear only in the optional account
  workspace, not on the link every email points to.
- The section that names real robots, "Robots that may fit this job", is
  below the simulation table.

**Smaller items**

- About 28 distinct email subjects and 12 screening outcomes, for a journey
  described as four steps.
- Many nouns for one thing: job, task, brief, assessment, scene, simulation,
  estimate, screening, evaluation, candidate, episode, practice run, job card.
- The header shows "Your workspace" or "Sign in" although no account is
  needed; the footer shows "Beta & pricing" and "Refunds" in a free beta.
- The "Chosen" cell breaks mid-word at desktop and phone widths.
- Held and reference-only pages list a dozen rows of "Did not run · 0/0
  scored".

## More than the question needs

| Area | Size | Note |
|---|---|---|
| Pipeline source | 2,505 modules, 1.31 M lines | Tests add 1.06 M lines. |
| Pipeline lanes off the website path | about 800 modules, 520k lines | G1 and humanoid 120k; WAM 72k; OSCAR 57k; splat lanes 43k; RunPod 25k (retired); MuJoCo 22k. Name-based classification. |
| Coupling | one import cycle of 1,137 modules, 709k lines | Every website unit loads about 1,900 modules. |
| Deployed units | 86 systemd units | 45 services, 30 timers, 18 path units in the repo. |
| Candidate catalog | 136 rows, 124 cannot run | |
| Simulators and scene back ends present | Isaac, MuJoCo, Newton, Chrono, Drake; Atlas plus about ten other reconstruction lanes | One of each is live. |
| WebApp server | 205.6k lines without tests | Site-journey route files are 9.5k lines; intake imports the agent runtime, so the journey loads 68% of the server (`inbound-request.ts:40`). |
| WebApp pages | 65 pages, 19.7k lines | 6 pages and 2.8k lines are the site's journey. 80 of 156 routes are redirects. |
| WebApp other estates | `ops/paperclip` about 172k lines; 94 npm scripts, 47 for agents and automation; a worker with 21 automation lanes | Stripe, payouts and marketplace routes are mounted though the beta is free. |

The November simplification audit reached the same conclusion from inside and
has removed several lanes. This audit adds the product side: much of what
remains serves generality (humanoids, navigation, seven task families, 124
placeholder rows) before one job kind has produced one useful result.

## Less than the question needs

1. Candidates that are systems a site can obtain.
2. A basis for "most likely" that has been checked against reality.
3. One page that says which, how sure, what was ruled out and what next.
4. A truthful reading of a zero result, with the scene finding beside it.
5. One outside site.

## The simple version

What a site cannot do itself, and what current models and APIs do well, is
this:

1. **Take the email and the video.** Keep the country field; it protects
   privacy.
2. **Read the video.** The existing site-assessment agent already extracts
   the job, the object, sizes, weights, heights, distance and people nearby as
   sourced estimates (`server/agents/site-assessment.ts`).
3. **Measure the scene.** Keep the reconstruction, used for distances and
   clearances only.
4. **Check every real robot against it.** `website_robot_fit.py` already does
   this for 22 models from ABB, FANUC, KUKA, Universal Robots, Yaskawa,
   Standard Bots, Franka, Kinova, UFactory, Unitree, Hello Robot and others,
   deciding each check across the whole uncertainty band and answering
   "unknown" when it cannot tell.
5. **Find who has done this job with that kind of robot.** A research agent
   and the existing team registry, with every claim quoted from its source.
6. **Write one page.** The pick, the runner-up, what was ruled out and why,
   what the video could not tell, how sure in plain words, and who could pilot
   it.
7. **Add simulation as one line of evidence**, only for a job kind where the
   simulated ranking has been checked against real trials.

Steps 2, 3 and 4 exist. Step 5 is half built. Step 6 is the missing product.
This drops the GPU stages, the authored objects and the fleet for most jobs,
which is where the week and most of the stop reasons come from. Its cost and
time are not measured here.

Why this ranks better than a simulated success count today: the two best
predictors a site can have in November 2026 are whether a system physically
fits the job and whether it already does that kind of job somewhere. Both are
checkable now. A zero-shot simulated success rate from an unmeasured scene is
not.

If simulation stays as a second tier, the simple form of it is one lane (a
fixed arm, tabletop pick and place), built on the published framework whose
ranking has been checked against real trials, with a known-answer run in every
scene, and with paired real trials to earn the word "likely". That is the work
Arm Decision Proof was created for; it has no partner yet.

## Keep, freeze, cut

- **Keep:** the intake and private link; consent, withdrawal and money
  controls; the video reader; the reconstruction as a measuring tool; the
  spec-sheet fit check; the plain held and new-video copy.
- **Freeze until one job kind gives a useful result:** humanoid and navigation
  lanes, the seven task families, the 124 disabled catalog rows, more
  per-scene levelling.
- **Cut from the site's view:** checkpoint names, episode tables, the list of
  candidates that did not run, workspace and pricing links, and all but about
  four emails (your link, a question if there is one, your result, held or
  ended).
- **Separate from the customer app:** the autonomous-organisation and agent
  estate, payouts and marketplace routes. Intake should not load them.

## Decisions for the owner

1. **Which evidence leads the answer.** The direction of 2026-10-21 says to
   reuse the capture and simulation path and "not replace them with text-only
   advice or supplier matching". This audit recommends a different order, not
   text-only advice: video-measured fit and cited deployment evidence lead,
   and simulation follows where it has been validated. The record since that
   date is the reason to revisit it.
2. **Scope.** Whether to freeze every lane except one job kind.
3. **The zero result.** Whether an all-zero run is shown as an estimate, and
   whether a scene finding such as "beyond reach" leads the page.
4. **Default sharing.** Whether a site's job becomes a partner-visible card
   without an explicit choice.
5. **Validation.** Whether Arm Decision Proof stays a separate frozen program
   or becomes the paired-trial check for the one simulation lane.
6. **First outside sites.** Whether to put 3–5 real sites through the simple
   answer before more pipeline work. Their reaction is the largest unknown,
   and no engineering answers it.

## Method and limits

- One lead planned and judged. Nine read-only fact-finding passes covered the
  live job path, the candidates, the evaluation method, the record of jobs,
  Pipeline surplus, WebApp surplus, the result copy, a local walk of the
  website with screenshots, and outside research.
- The lead re-read the claims the verdict rests on: tracker lines 446-448 and
  471-476, the selection rule, the 12 enabled catalog rows and their missing
  deployment records, the outcome sentences, the required task field, the
  default sharing flag, the absent known-good control, the fit check and its
  22-model catalog, and the line counts.
- Not read: Firestore, the host, Render settings, provider bills or the live
  site. Registry figures are from the 2026-10-18 audit.
- Both clones are shallow (50 commits), so history-based claims are limited.
- Surplus buckets are classified by module name and import reachability and
  are approximate. "Mounted" means reachable in code, not observed traffic.
- Outside sources were read by the research pass; figures quoted from papers
  dated after June 2026 were not independently re-read.
- The website was rendered with test fixtures, not production data; the
  estimate screenshot uses fixture values.
