# Vision: Working Up the Stack

> Repo role in this vision: **`BlueprintCapturePipeline`** turns capture bundles into the
> maintained Site-Task Testbeds and claim-level Task Evaluation Runs that are
> the engine of every rung below. Policy improvement and post-training remain
> optional internal experiments or permitted evidence uses inside a run, not products.

This document is **shared, byte-identical, across `BlueprintCapture`, `BlueprintCapturePipeline`,
and `Blueprint-WebApp`** (inside the shared block). It is the long-horizon strategy. It sits on
top of — and never overrides — [`PLATFORM_CONTEXT.md`](PLATFORM_CONTEXT.md) and
[`WORLD_MODEL_STRATEGY_CONTEXT.md`](WORLD_MODEL_STRATEGY_CONTEXT.md), which describe what is true
and sellable **today**. Where this document describes rungs 3–5, treat them as **direction and
bets**, not current capability. Every forward claim carries an explicit proof boundary.

<!-- SHARED_VISION_START -->
## The one-sentence version

Blueprint starts as the neutral way to know **which robot policy will actually work at a specific
real site**, becomes the **default measurement the whole market routes its deployment decisions
through**, and — because that position produces proprietary, ground-truth data that compounds with
every site we capture — earns the option to move up the stack into **prediction, data generation,
per-site policies, and eventually owning deployment where we can prove we are the best operator.**

The foundation never changes: **capture-first, provenance-true, model-backend-swappable.** Each
rung is built on the same real-site capture moat, and each rung's data makes the next one cheaper
to win.

## Why now (the market we are climbing into)

Three curves make this the right decade to build a measurement-and-data layer under robotics.

1. **Deployment is going from demos to volume, across many vendors.** The industrial-robot base is
   already large — IFR counted **4.66M operational industrial robots in 2024**, with **542k installed
   that year** (World Robotics 2025). Humanoids are the fast-growing sliver on top: Goldman Sachs
   **raised its 2035 humanoid TAM ~6× (from ~$6B to $38B)** and quadrupled its unit estimate to
   ~1.4M; Morgan Stanley models **>1 billion humanoids and a ~$5T market by 2050** (~90% industrial);
   Bank of America projects **3 billion units by 2060** and shipments rising from ~90k (2026) to
   1.2M (2030). The forecasts disagree by an order of magnitude — which is itself the point: the
   *timing* is uncertain, but the *direction* (many robots, many sites) is not. Real production
   proof already exists — Figure ran 11 months at BMW Spartanburg (**~1,250 hours, 90k+ parts, 99%
   accuracy**); Agility's Digit passed **100k+ totes** live at GXO; Amazon crossed **1M robots**.

2. **The "brain" is fragmenting away from the body.** There is no single winner in robot policy.
   Physical Intelligence (π0 → π0.5 → π0.7), Google DeepMind (Gemini Robotics), NVIDIA (open GR00T
   N1/N1.5/N1.7), Skild AI (cross-embodiment "Skild Brain," valued **over $14B**), Figure (in-house
   Helix since the Feb 2025 OpenAI split), Tesla, Unitree, and **330+ Chinese humanoid models
   unveiled in 2025** now form a genuine **many-bodies × many-brains matrix**. When there are many
   interchangeable brains and bodies, **the scarce, valuable thing becomes a trustworthy way to
   compare them on a specific real site.**

3. **Evaluation is the acknowledged bottleneck — and nobody neutral owns it.** A single rigorous
   real-world evaluation of one model (OpenVLA) took **>2,500 rollouts across 4 setups and 3
   institutions and 100+ hours of human labor** (AutoEval, 2025). NVIDIA's Jim Fan called robotics
   benchmarking an **"epic disaster."** Typical papers report only 10–30 trials — statistically
   underpowered — while industrial buyers expect up to **99.99% reliability** and line downtime
   costs tens of thousands of dollars per minute (Bain, 2025). Academic efforts (RoboArena, SIMPLER,
   RoboCasa) exist *because* cross-lab comparison is broken today — but **none is a neutral,
   buyer-facing, site-specific evaluation service.** That whitespace is rung 1.

## The ladder

Each rung is a capability and moat built through the Task Evaluation Run product,
plus a launchpad for the next. We do not create a new SKU for each rung, and we
earn each claim with evidence from the previous rung.

---

### Rung 1 — The wedge: claim-bounded Task Evaluation Runs

**What it is (shipping today).** Blueprint's one product is the **Task Evaluation
Run**: bind a decision to an exact maintained Site-Task Testbed, decompose it into
claims, route each claim to qualified evidence, and return a decision, a partial
decision, or an explicit abstention with the next cheapest experiment. It returns
a bounded positive or negative decision, candidate elimination, partial decision,
explicit abstention, or the next evidence needed. Comparative policy ranking is
one possible claim, not the product definition, and candidate ordering appears
only when the evidence supports it. This is the current PMF wedge (see the
[Commercial Wedge Overlay](PLATFORM_CONTEXT.md)).

**Research direction, not Blueprint proof.** Published generated-world results
motivate a learned-evaluator evidence method, but do not qualify it for a Blueprint
claim or domain. As of June 2026 two results make that research direction concrete:

- **SC3-Eval** (NVIDIA · Physical Intelligence · Toronto/Vector · Stanford · UC Berkeley,
  arXiv:2606.18610) adapts a pre-trained video foundation model into a *closed-loop* policy evaluator
  by enforcing **forward-inverse dynamics, cross-view, and test-time consistency**. It reports
  **Pearson r = 0.984 (MMRV 0.022) in-distribution** agreement with real policy performance across
  **seven VLA policies** (381 hours of real table-bussing data), a **0.929 headline correlation**,
  and — honestly — a drop to **r = 0.870 out-of-distribution.** This is the same self-consistency
  family Blueprint's own WAM evaluator prepares (forward/inverse episode-consistency scoring, behind
  a replaceable external-scorer boundary — see [`AGENTS.md`](AGENTS.md)). The `0.929` value is
  SC3-Eval's published overall Pearson result across those seven policies, not a Blueprint
  measurement; Blueprint has not measured an equivalent correlation.
- **OSCAR** (Peking University · NVIDIA/Michigan, arXiv:2606.04463) — an **omni-embodiment,
  action-conditioned world model** (a fine-tune of Cosmos-Predict2.5-2B on a single GH200, using 2D
  kinematic-skeleton conditioning for cross-embodiment) — reports **Pearson r = 0.852 /
  Spearman ρ = 0.750** against the real **RoboArena** ranking across 7 generalist policies (65
  sessions, 1,365 pairwise comparisons), and argues explicitly for "a future where robot policies can
  be **purely evaluated in virtual generated worlds**." Two honesty caveats travel with that quote:
  OSCAR's published validation is **open-loop only** — no chained closed-loop durability result was
  published — and exposure-bias collapse in long chained generation is a named open problem in the
  2026 literature, so the "purely virtual evaluation" future is the paper's argued direction, not its
  demonstrated result. OSCAR sits behind Blueprint's swappable
  world-model adapter (see [`WORLD_MODEL_STRATEGY_CONTEXT.md`](WORLD_MODEL_STRATEGY_CONTEXT.md)).

Earlier work corroborates the direction (SIMPLER r ≈ 0.924; AutoEval r ≈ 0.942 while cutting human
supervision >99%). **SC3-Eval reports an overall closed-loop Pearson correlation of `0.929` across
seven policies under its published protocol. This is not a Blueprint measurement; Blueprint has
not measured equivalent rank fidelity.** Ranking is the honest, defensible unit; the honest caveat
is that SC3-Eval's in-distribution 0.98 becomes ~0.85–0.87 cross-embodiment / OOD — and on the OOD
online split its Pearson edge over its own Cosmos-Predict2.5 baseline is a statistical wash
(0.870 vs 0.871; it keeps only an MMRV edge, 0.171 vs 0.195). The consistency recipe's advantage is
largely an in-distribution result today — precisely the gap rung 3b has to close, and precisely why
Pipeline must qualify each method for the requested claim and may abstain.

**Proof boundary (non-negotiable).** Blueprint's current preregistered
policy-ranking verdict is `thesis_not_supported`. A Task Evaluation Run may still
resolve geometry, perception, collision, or other qualified claims and abstain on
ranking. We sell an inspectable decision or abstention with per-claim outcomes,
validation envelope, uncertainty, disagreement, claim ceiling, next experiment,
and exact provenance. We do **not** sell a guaranteed ranking or field outcome, an
off-scope validation, deployment readiness, safety certification, or a claim that
we ran the buyer's robot without separately accepted physical proof. Generated
frames are review support, never real-world proof.

---

### Rung 2 — The standard: become the default the market routes deployment decisions through

**What it is.** Blueprint becomes the **neutral referee both sides trust**: robot teams use our runs
to prove readiness and win pilots; site operators require our runs before they let a robot on the
floor. The goal the founder stated plainly — **a large portion of *all* deployments and pilots pass
through our evaluation** — because the site operators *want* it, and the robot teams need it to sell.

**This is a decision layer, not a generic marketplace.** Per platform doctrine we do **not** become
a generic asset bazaar. We become the **measurement standard** — the thing the industry transacts
*against*. History shows how durable that position is when it becomes a required gate:

- **Credit ratings** (S&P/Moody's/Fitch): ~90% share, embedded in **bank capital rules**; Moody's
  runs a **40.6% operating margin**, S&P's ratings segment **~63%.** A bond effectively can't be
  sold at scale un-rated.
- **UL / OSHA NRTL**: **38 product categories legally require** third-party safety certification —
  *and* retailers refuse to stock uncertified gear even where no law compels it. Regulatory floor
  **plus** private-market norm.
- **MLPerf/MLCommons**: in ~a decade it became the neutral scoreboard the entire AI-hardware
  industry submits to (**20 organizations** in Training v5.1); vendors now market chips by it.
- **Nielsen**: was the currency of the **~$70B US TV-ad market** — and its 2024 erosion is the
  **cautionary tale**: a measurement monopoly cracks when the substrate shifts and clients fund
  challengers. A trust layer must **continuously re-validate against the frontier or die.**

The strategic logic is Aggregation Theory: **own the "which one is best" decision and the suppliers
being rated become interchangeable beneath you.** That is the toll booth. It is worth more than any
single model we could own.

**Proof boundary.** Neutrality is the asset. The moment our ranking is perceived as bought, the
standard is worthless. Rung 2 requires a visible **methodology, re-validation cadence, and
conflict-of-interest firewall** — the same discipline that keeps a ratings agency credible.

---

### Rung 3 — Prediction + data engine (whichever matures first)

Two capabilities grow out of the data rungs 1–2 produce. They are **partly shipping, partly bets.**

**3a — Evidence reuse and optional policy experiments (inside Task Evaluation
Runs).** Post-training is not a separate current product. Rights-cleared run
evidence may become eligible for evaluation or
post-training use after provenance, robot-action alignment, quality, and leakage
gates; eligibility does not prove that training occurred or that a policy
improved. Policy improvement may be an internal candidate-generation experiment. Robotics is data-starved in a way language never was: usable
open-source real-world interaction data is **<5,000 hours** vs. trillions of text tokens; Bessemer
calls robot data **"~a billion times smaller than internet text"** and projects **>$3B** of industry
data spend in two years. High-quality teleoperation still costs **~$118–340/hour**. Two things follow:
(i) synthetic generation is real and additive — NVIDIA built GR00T N1.5 in **36 hours vs. ~3 months**
and generated **780k trajectories (~6,500 human-hours-equivalent) in 11 hours** — but it *complements*
physics sim, it doesn't yet replace real data; and (ii) — the load-bearing fact for us — **policy
generalization scales with the *diversity of real environments*, not raw demo count** ("Data Scaling
Laws in Imitation Learning," 2024). **That is exactly what proprietary multi-site capture is.** Every
site we capture makes qualifying evidence more valuable; Scale AI's own framing is that strict data
lineage is **"a moat that grows with every deployment."**

**3b — Calibrated real-world prediction ("95% on our eval ≈ 95% in real life").** This is the
founder's north star and it is **explicitly a multi-year bet gated on world-model + calibration
progress.** OSCAR and SC3-Eval (rung 1) are the concrete frontier and the honest map of the gap: rank
*correlation* is already strong **in-distribution** (SC3-Eval r = 0.984) but degrades **out-of-
distribution and cross-embodiment** (SC3-Eval 0.870 OOD; OSCAR 0.852 on RoboArena). A *calibrated
probability* that transfers to a specific new site at 95%↔95% requires closing exactly that OOD gap —
which depends on advances that are **not solved yet**: precise action-conditioning, long-horizon
consistency, and physical accuracy remain the weakest axes of even the best world models (Genie 3
holds consistency only ~minutes; Cosmos and OSCAR augment and rank, they do not yet *certify*). Every
site Blueprint captures pushes more of the target distribution in-distribution — that is why the
capture moat and this prediction bet are the same flywheel. We state the ambition **with its
dependency**, we measure ourselves against the published correlation bar (OSCAR/SC3-Eval), and we
**do not** convert a correlation into a guarantee.

**Proof boundary (heaviest here).** Rung 3b is where over-claiming does the most damage. We publish
calibration curves and out-of-sample validation, we label the world-model dependency, and we keep the
model backend swappable so "better world model later" is a drop-in, not a rebuild.

---

### Rung 4 — Our own policies, per captured site

**What it is.** Because we hold proprietary, provenance-clean, multi-site capture data — the scarcest
input in the stack — we can **fine-tune policies specialized to each site we've captured**, then use
our own neutral eval (rung 1–2) to test the honest question: **do our per-site policies beat the
robot teams' general policies on *their* site?** If yes on our own scoreboard, that is a real,
measured edge, not a marketing claim.

**Who is the customer here? (Open decision — flagged deliberately.)** The founder is right to be
unsure. The credible candidates:
- **Site operators** who don't want to shop for a brain — they want a turnkey policy that *works on
  their floor*, sold as an outcome.
- **Robot hardware makers with weak brains** — in a many-bodies-few-good-brains world, a great
  per-site policy is the missing half of their product.
- **Integrators** de-risking pilots.

This is a **strategic fork to decide with data**, not to pre-commit now. What makes it *possible* is
rungs 1–3; what makes it *safe* is that we only claim "beats" when our neutral eval says so.

**Proof boundary + the neutrality tension begins here.** The moment we ship our own policies we are a
participant in the market we grade. Rung 4 is only defensible with a **structural firewall** between
the neutral-eval business and the policy business (see rung 5).

---

### Rung 5 — Own the deployment: cheap hardware + our best per-site policy

**What it is (the most aggressive rung).** Hardware is deflating toward commodity — Unitree ships
the **G1 at ~$16k and the R1 at ~$5,900**; humanoid bill-of-materials fell **~40% in a single year**
and is projected **from ~$35k (2025) to <$17k (2030)**; Tesla targets a **$20–30k Optimus.** As the
chassis commoditizes, **durable margin migrates to the intelligence and the service.** RaaS already
clears the labor-arbitrage bar (Formic bundles robots **~$8/hour** vs. **$30–45/hour** loaded human
labor). So the end-state option is: **buy cheap commodity robots, run our specialized per-site
policies on them, and capture the deployment value ourselves** — not by manufacturing hardware
(we stay un-vertically-integrated on the body), but by owning the brain + the operating relationship.

**The honest counterargument (must be read before anyone acts on rung 5).** A neutral evaluator that
*also deploys its own robots* creates textbook **vertical channel conflict** — competing with the
customers and partners it grades. Channel-conflict literature calls this the **most damaging kind**
because it "strikes at the foundation of vendor-partner trust." **Neutrality is the asset that made
rungs 1–2 worth anything; deploying our own fleet spends it.** Three ways to hold the tension:
1. **Structural separation** — an independent, firewalled eval arm (the ratings-agency model), so the
   standard stays credibly neutral even as a separate arm operates robots.
2. **Only self-deploy where no partner will or can**, and only where our *own neutral eval* proves we
   are the best available operator for that site/task.
3. **Stay asset-light** — license the per-site policy to whoever owns the fleet, rather than becoming
   a capital-heavy, thin-margin fleet operator ourselves.

Rung 5 is an **option we earn, not a destination we assume.** Hardware commoditization also
commoditizes the deployment layer; the "best policy" edge can be transient as foundation policies
converge. We decide rung 5 with rung-1 evidence in hand — and we do not let the *possibility* of
rung 5 contaminate the neutrality that rungs 1–2 depend on.

---

## What must stay true across every rung (the invariants)

These are inherited from platform doctrine and do not bend as we climb:

1. **Capture-first.** Every rung is built on proprietary, rights-clean, provenance-true real-site
   capture. That is the moat that *grows* when models commoditize — not shrinks.
2. **Model backends stay swappable.** No rung couples the company to one checkpoint, provider, or
   world model. "Better model later" must be a drop-in behind the adapter boundary.
3. **Estimates, never guarantees.** Rank fidelity and predicted success — with proof boundaries and
   missing-proof labels — all the way up. We never launder a correlation into a promise.
4. **Neutrality is a balance-sheet asset.** From rung 2 on, we protect it structurally. Rungs 4–5
   are gated on a credible firewall.
5. **Raw capture truth is authoritative.** No downstream artifact — generated media, world-model
   output, per-site policy, deployment dashboard — is allowed to outrank raw capture and provenance.

## The flywheel (why the order compounds)

> more sites captured → better, more diverse eval runs → more deployment decisions routed through us
> → more proprietary real-world outcome data → better prediction *and* better data generation → better
> per-site policies → more deployments we can credibly serve → funds and justifies more capture.

Rungs 1–2 are a **data-acquisition strategy disguised as a product**, and the buyer-facing evidence
foundation everything above rests on. Every deployment decision we support can produce a labeled
outcome that strengthens later evaluation and policy-improvement work — an outcome no competitor
without our capture footprint can buy — without turning an unverified prediction into ground truth.

## The bets we are explicitly making (and what would have to become true)

| Bet | Current evidence | What must become true |
|-----|------------------|------------------------|
| World models get good enough to predict physical outcomes | **SC3-Eval r=0.984 in-dist (OOD ≈ its Predict2.5 baseline) / OSCAR r=0.852 RoboArena, open-loop-validated only**; Cosmos/Genie/Marble usable for augmentation *today* | Close the OOD/cross-embodiment gap (0.85–0.87 → ~0.95); action-conditioned, long-horizon, physically-accurate prediction with published chained closed-loop durability — years out |
| Synthetic + site data gets good enough for post-training | GR00T-Dreams, 780k-traj/11h, DreamGen ~10× | Sim-to-real transfer strong enough to sell improvement, not just data |
| Real-site diversity is the durable data moat | Data-scaling-law: generalization ∝ environment diversity | We out-capture competitors on breadth *and* provenance quality |
| A neutral eval standard can become a required gate | Ratings/UL/MLPerf precedents | We get embedded in procurement/insurance/pilot decisions before a rival |
| Per-site policies can beat generalists | Plausible from data-scaling-law; unproven for us | Our own neutral eval measures it — and we keep the firewall |

If a bet fails, the rung above it pauses; **rungs 1–2 stand on their own** regardless, because eval
ranking is valuable even if world models plateau.

## Open decisions (do not pre-commit)

- **Rung 4 customer:** site operators vs. brain-less hardware makers vs. integrators. Decide with
  demand data, not now.
- **Rung 5 structure:** structural-firewall operator vs. asset-light policy licensor vs. no rung 5.
  Decide only with rung-1 evidence and a neutrality plan.
- **Neutrality governance:** when do we formalize the eval firewall? (Answer: before rung 4, not
  after.)
<!-- SHARED_VISION_END -->

## Evidence base (selected, verified 2026-07-03; refreshed 2026-07-28)

Figures below were gathered by first-party web research and passed an adversarial fact-check;
corrections from that check are already reflected above. Confidence and known caveats noted.
The 2026-07-28 refresh folds in the in-repo research pass
[`docs/policy_and_wam_benchmark_research_2026-07-26.md`](docs/policy_and_wam_benchmark_research_2026-07-26.md)
(OSCAR open-loop-only validation, exposure-bias collapse in chained generation, and
baseline-checkpoint provenance findings).

**Generated-world policy evaluation — the scientific core of the wedge (rungs 1 & 3b)**
- **SC3-Eval** — *Evaluating Robot Foundation Models via Self-Consistent Video Generation* (NVIDIA · Physical Intelligence · U Toronto/Vector · Stanford · UC Berkeley; Tseng et al., Jun 2026). Video world model → closed-loop evaluator via forward-inverse dynamics + cross-view + test-time consistency. **Pearson r = 0.984 (MMRV 0.022) in-distribution, 0.929 overall headline, 0.870 OOD**; 7 VLA policies; 381h real table-bussing; 2.3s/24-frame chunk on GB200. Beats Ctrl-World / IRASim overall; against its own Cosmos-Predict2.5 baseline it wins in-distribution (0.984/0.022 vs 0.897/0.090) but the **OOD online split is a Pearson wash (0.870 vs 0.871), keeping only an MMRV edge (0.171 vs 0.195)** — the consistency recipe's advantage is largely in-distribution today, and OOD is the axis rung 3b's bet lives on. Published scope is one table-bussing scene, 12 object categories, three camera views, seven policy checkpoints, ≤20-second rollouts. *These are SC3-Eval paper results, not Blueprint measurements; Blueprint has not measured equivalent rank fidelity. The consistency family is an evaluator recipe Blueprint prepares behind an external-scorer boundary.* https://arxiv.org/html/2606.18610v3 · project page https://weichengtseng.github.io/sc3-eval/
- **OSCAR** — *Omni-Embodiment Action-Conditioned World Model for Robotics* (Peking University · NVIDIA/Michigan; Wu & Gao, Jun 2026). Cosmos-Predict2.5-2B fine-tuned on a single GH200 with 2D kinematic-skeleton cross-embodiment conditioning. **RoboArena virtual-vs-real: Pearson r = 0.852 / Spearman ρ = 0.750 / MMRV 0.571**, 7 generalist DROID policies, 65 sessions, 1,365 pairwise comparisons; 180,657 curated episodes across 4 robot + 2 human embodiments. Goal: "robot policies … purely evaluated in virtual generated worlds." **Caveat (in-repo research, 2026-07-26): the paper's validation is open-loop only — no chained closed-loop durability was published — and the 2026 literature names exposure-bias collapse in long chained generation as the open failure mode; the goal quote is argued direction, not demonstrated result.** Sits behind Blueprint's swappable world-model adapter. https://arxiv.org/html/2606.04463v2

**Market & deployment**
- Goldman Sachs: 2035 humanoid TAM raised ~6× to **$38B**, ~1.4M units; BoM fell **~40%** 2023→2024. https://www.goldmansachs.com/insights/articles/the-global-market-for-robots-could-reach-38-billion-by-2035
- Morgan Stanley: **$5T / >1B units by 2050** (~90% industrial). https://www.morganstanley.com/insights/articles/humanoid-robot-market-5-trillion-by-2050
- Bank of America: **3B units by 2060**; shipments 90k (2026)→1.2M (2030); BoM $35k→<$17k. https://fortune.com/2026/03/13/bank-of-america-humanoid-robot-forecast-3-billion-2060/
- Macquarie: **6.3M units / $139B by 2035; $3T by 2050** (the "$1.7T" figure floating in secondary coverage is not in Macquarie's sources — do not use).
- IFR World Robotics 2025: **4.66M** operational, **542k** installed in 2024. https://www.therobotreport.com/ifr-industrial-robot-deployments-have-doubled-in-10-years/
- Figure @ BMW (1,250h / 90k parts / 99%). https://www.figure.ai/news/production-at-bmw · Agility @ GXO (100k+ totes). https://roboticsandautomationnews.com/2025/11/24/agility-robotics-digit-humanoid-passes-100000-tote-milestone-in-live-gxo-implementation/96877/

**Fragmentation & eval bottleneck**
- Physical Intelligence π0→π0.7. https://www.pi.website/blog/pi0 · NVIDIA GR00T N1.5 **38.3% vs 13.1%**. https://research.nvidia.com/labs/gear/gr00t-n1_5/ · Skild "over $14B." https://www.businesswire.com/news/home/20260114335623/en/Skild-AI-Raises-$1.4B-Now-Valued-Over-$14B
- AutoEval (corroborating, 2025): OpenVLA eval = **2,500+ rollouts / 100+ hours**; matches human eval **r=0.942, MMRV=0.015**. https://arxiv.org/html/2503.24278v1
- SIMPLER (corroborating, 2024): sim-to-real ranking **Pearson r≈0.924**. https://arxiv.org/html/2405.05941v1
- RoboArena: **612 pairwise / 7 policies / 7 institutions** — but academic, not buyer-facing. https://arxiv.org/abs/2506.18123
- Bain: industrial buyers expect **up to 99.99% reliability**. https://www.bain.com/insights/humanoid-robots-from-demos-to-deployment-technology-report-2025/

**World models**
- NVIDIA Cosmos (Predict/Transfer/Reason; adopters 1X, Agility, Figure, Skild); complements Isaac Sim. https://nvidianews.nvidia.com/news/nvidia-announces-major-release-of-cosmos-world-foundation-models-and-physical-ai-data-tools
- DeepMind Genie 3: 720p/24fps, consistency **~minutes**. https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/
- World Labs "Marble," $1.23B raised (Feb 2026 round). https://www.worldlabs.ai/blog/funding-2026 · Open problems: action-conditioning, long-horizon, physical accuracy. https://arxiv.org/html/2605.00080v1

**Data moat**
- Bessemer: robot data **~1B× smaller than internet text**, **>$3B** spend in 2 years. https://www.bvp.com/atlas/can-world-models-unlock-general-purpose-robotics
- Scale AI/Forbes: **<5,000 hours** open real-world data; lineage = "moat that grows with every deployment." https://www.forbes.com/sites/josipamajic/2026/06/29/physical-ai-hits-a-data-labeling-wall-that-only-cash-can-fix/
- **Data Scaling Laws in Imitation Learning: generalization ∝ environment diversity** (the per-site-capture moat). https://arxiv.org/abs/2410.18647
- Open X-Embodiment (1M+ eps, 22 robots, 527 skills) / DROID (76k traj, 350h, 564 scenes, 52 buildings, 12 months). https://arxiv.org/abs/2310.08864 · https://arxiv.org/html/2403.12945v2

**Trust-layer economics & RaaS**
- Credit ratings: ~90% share; Moody's **40.6%** / S&P ratings **~63%** operating margin; embedded in **bank capital rules** (note: SEC Rule 2a-7's NRSRO mandate was repealed in 2016 — cite Basel, not 2a-7). https://www.sec.gov/files/jan-2025-ocr-staff-report.pdf
- UL/OSHA NRTL: **38 mandated categories** + market norm. https://www.osha.gov/nationally-recognized-testing-laboratory-program/products-requiring-approval
- MLPerf Training v5.1: **20 submitting orgs**. https://mlcommons.org/2025/11/training-v5-1-results/ · Aggregation Theory. https://stratechery.com/2015/aggregation-theory/
- Unitree G1 ~$16k / R1 ~$5,900; Formic RaaS **~$8/hr** (vs $30–45/hr labor). https://www.forbes.com/sites/jonmarkman/2026/04/27/unitree-g1-humanoid-robots-are-reshaping-the-robotics-investment-stack/ · https://formic.co/resources/articles/robots-as-a-service-raas
- Channel-conflict risk of a neutral evaluator that self-deploys (the rung-5 caution). https://www.channeltivity.com/blog/channel-conflict/

*Caveats carried from the fact-check: humanoid TAM forecasts diverge by ~10× (use ranges, not points);
some RaaS market sizes and Tesla payback figures are analyst estimates, not primary; forward BoM
figures beyond the ~40% one-year drop are lower-confidence.*

---

*Shared cross-repo doctrine, generated from a single source. Edit
`BlueprintCapturePipeline doctrine/`, then run
`python3 scripts/sync_shared_doctrine.py --write` to splice it into every repo and
re-lock. Do not hand-edit the block above: CI verifies it against
`contracts/shared-doctrine.lock.json` and will reject a drifted copy. Last synced
2026-07-31 (union merge of the previously diverged Pipeline and WebApp copies;
`BlueprintCapture` was not checked out and remains unsynced).*
