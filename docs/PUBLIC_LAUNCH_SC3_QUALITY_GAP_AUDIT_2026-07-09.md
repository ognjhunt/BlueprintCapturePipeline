# Public Launch and SC3-Quality Gap Audit

- Date: 2026-07-09 (America/Chicago)
- Repository: `BlueprintCapturePipeline`
- Audited commit: `7a462e070cc1a55b2bb829dd85620e92836eb20f` (`main`, equal to `origin/main`)
- Change policy: audit/specification only; no implementation changes were made

## Executive verdict

**Do not launch the current commit to the public. Do not describe Blueprint as achieving “93%,” `0.929`, or equivalent SC3 accuracy/rank fidelity.**

The repository has many strong contract and claim-boundary improvements, but four different facts are currently being collapsed into one launch story:

1. Local unit/contract tests can pass.
2. A generated video or package can arrive and be reviewable.
3. An evaluator can be scientifically valid and rank policies like the real world.
4. A live, secure, reliable, privacy-safe production system can serve buyers.

Those are separate proof layers. The current checkout has confirmed defects in all four layers, including a red clean-checkout CI commit, silent work loss, unauthenticated/path-traversable runtime surfaces, false geometry and training-row readiness, incorrect SC3 metrics, and claim gates that can unlock external/deployment accuracy from four anchors across two policies.

The realistic launch paths are:

- **Evaluator-bounded sim-only beta:** achievable after the base, data-truth, security, and false-SC3-proof blockers below are fixed. It may truthfully return `correlation_not_measured`; physical-robot deployment or safety proof is not a prerequisite.
- **Paid marketplace / buyer-package launch:** additionally requires consent/rights, payment/delivery, package-integrity, live operations, backup/restore, and entitlement evidence.
- **SC3-like external-fidelity claim:** requires a new, frozen scientific evaluation program. It is not unlocked by a green sim gate or by implementing the code changes alone.
- **Physical deployment/safety claim:** remains a separate claim-upgrade lane and is not a blocker for a deliberately scoped sim-only beta.

## What the research actually says

The target needs to be stated correctly before it can be engineered correctly.

- [SC3-Eval](https://weichengtseng.github.io/sc3-eval/) reports an overall closed-loop **Pearson correlation of `0.929`** and MMRV `0.119` between generated-world and real-world aggregate policy performance. That is not 92.9% task success, output accuracy, or a Blueprint measurement.
- SC3's paper is [arXiv:2606.18610v3](https://arxiv.org/html/2606.18610v3), not the second link in the request. Its main evaluation uses matched initial conditions, seven checkpoints/policies, three explicit success criteria, three synchronized views, closed-loop re-query, forward/inverse and cross-view consistency, calibrated termination, and blinded human main labels.
- The requested [arXiv:2606.04463v2](https://arxiv.org/html/2606.04463v2) is **OSCAR**, a separate skeleton-conditioned world model. OSCAR reports Pearson `0.852` on its RoboArena policy-evaluation study, not `0.929`; its automated judge agreed with humans on 78 of 100 calibration clips.
- [SIMPLER](https://simpler-env.github.io/) defines MMRV using pairwise **real-success-rate margins** for misranked pairs, not ordinal rank-position distance.

Therefore the proper north-star is: **maximize independently measured policy-ranking fidelity under a frozen, matched, task-specific protocol, while refusing to manufacture a number when the required evidence is absent.** Matching `0.929` is an empirical outcome, not an engineering acceptance threshold that can be guaranteed.

## Scope and severity model

| Label | Meaning |
| --- | --- |
| `BASE` | Blocks any public Pipeline service or public artifact claim. |
| `SIM` | Blocks an evaluator-bounded sim-only beta. |
| `SC3` | Blocks an SC3-like external-fidelity/correlation claim. |
| `PTDP` | Blocks a buyer-facing Post-Training Data Package or native training export. |
| `PAID` | Blocks paid marketplace, payment, payout, or buyer delivery. |
| `LIVE` | Blocks a live cloud/provider/runtime production launch. |
| `PHYSICAL` | Blocks only physical robot, safety, or deployment claim upgrades. |

- **P0:** must be closed before launching the named scope.
- **P1:** high-priority correctness/reliability work; must close before broadening beyond a tightly controlled beta or before making the named quality claim.
- **P2:** scale, maintainability, governance, or defense-in-depth gap; schedule before public scale and promote if threat modeling or launch scope makes it directly reachable.
- **External proof:** not necessarily a source-code bug, but a launch claim is blocked until fresh live evidence exists.

## Evidence snapshot at the audited commit

### Confirmed positive evidence

- `main` is locally clean and exactly matches `origin/main` at the audited SHA.
- Local fast lane: **2,598 passed, 4 skipped, 1,431 deselected**.
- Local full lane: **4,029 passed, 4 skipped** in 8m56s.
- Ruff, `compileall`, `pip check`, interpreter-matrix validation, `git diff --check`, Terraform formatting, and `terraform validate` passed locally.
- All 145 declared Python console entry points imported and resolved to callables.
- Wheel construction completed successfully, with only setuptools license-metadata deprecation warnings.
- GitHub secret scanning and push protection are enabled, and no open secret alert was found.
- The current GitHub **Sim-Only Local Gate** is green.
- The repository often states the right claim boundary: raw capture is authoritative; generated media is support; missing anchors should mean `correlation_not_measured`; physical success is not inferred from sim.

### Confirmed release contradictions

- GitHub [CI run 29040598378](https://github.com/ognjhunt/BlueprintCapturePipeline/actions/runs/29040598378) is red at the audited SHA: 1 failed, 2,596 passed, 5 skipped, 1,431 deselected.
- GitHub [Full Test Lane run 29040598434](https://github.com/ognjhunt/BlueprintCapturePipeline/actions/runs/29040598434) is red at the same SHA: 1 failed, 4,029 passed, 3 skipped.
- Both fail the same supposedly committed warehouse fixture test. Local green is caused by files under `tests/fixtures/warehouse_task_min/pipeline/` that exist locally but are hidden by the broad `pipeline/` ignore rule.
- GitHub [Sim-Only Local Gate run 29040598655](https://github.com/ognjhunt/BlueprintCapturePipeline/actions/runs/29040598655) is green, while its generated SC3 protocol remains blocked with zero accepted anchors, no measured correlation, no action normalization, and zero synchronized cameras. The gate proves plumbing, not SC3 fidelity.
- `main` has no GitHub branch protection and no ruleset. The audited commit was pushed directly with no associated pull request.
- A fresh paid-marketplace gate returned `automated_contracts_passed_manual_ops_required` and **zero blockers**, despite the red clean-checkout CI, the findings in this audit, and absent live/manual evidence.
- A fresh external-alpha gate failed because the required capture-truth restore-drill artifact is missing.
- A fresh no-spend WAM real-provider probe was blocked before a real provider ran: SAM3, depth, and pose runtime/model/command evidence was missing.
- A fresh live-pipeline setup audit returned `local_ready_live_external_blocked` with missing simulator, vision-label, delivery, live-agent, capture-root, and inbox configuration.
- Dependency audit found **27 known advisories across 10 installed packages**, including runtime paths that process untrusted images and HTTP requests.

### Proof boundary for this audit

Passing local tests are local contract evidence only. They do not prove live provider execution, GPU image correctness, semantic task success, buyer delivery, real payments/payouts, privacy false-negative performance, capacity, recovery, or real-world rank correlation.

## P0 launch blockers

### REL-01 — Clean-checkout CI is red and local green depends on ignored fixture state

**Scopes:** `BASE`, `SIM`, `PTDP`, `LIVE`

**Evidence:** `.gitignore:81-83`; `tests/test_eval_ready_task_grounding.py:338-362`; the two failing GitHub runs above. The test reads `pipeline/geometry/camera/intrinsics.json`, but `pipeline/` is ignored and only a different sim-only fixture is unignored.

**Impact:** release evidence is not reproducible. A checkout that looks clean can contain ignored state that changes behavior.

**How to tackle:** move source fixtures out of generated-path namespaces or explicitly unignore every committed fixture; add a clean `git archive`/fresh-clone test job; make release provenance report ignored state relevant to test inputs.

**Exit criteria:** fresh clone, `git archive`, local canonical environment, CI fast lane, and CI full lane all pass the exact same test collection with no required ignored files.

### REL-02 — `main` is unprotected and required-check policy is documentation only

**Scopes:** `BASE`, `LIVE`

**Evidence:** `docs/CI_REQUIRED_CHECKS.md:3-22` says both checks are required; GitHub API reports `main` unprotected and no rulesets.

**Impact:** red or unreviewed commits can be pushed directly to the release branch, as happened at the audited SHA.

**How to tackle:** require pull requests, `CI / test`, `Full Test Lane / Full pytest lane on CPU runner`, signed/linear history as appropriate, resolved reviews, and restricted bypass. Add CODEOWNERS for security, data-contract, evaluator, and deploy surfaces.

**Exit criteria:** a test branch cannot merge with either check red; direct push and force push are denied; an audited break-glass path is independently approved and time bounded.

### REL-03 — Launch gates can report success without consuming release-critical evidence

**Scopes:** `BASE`, `SIM`, `PTDP`, `PAID`, `LIVE`

**Evidence:** fresh paid gate returned zero blockers while current CI is red; `scripts/run_paid_marketplace_launch_gate.py` runs narrow contract slices, not the full suite, SC3 fidelity, dependency audit, restore drill, or live runtime; `scripts/build_launch_readiness_packet.py:407-465,516-553` displays failure statuses but does not require accepted statuses.

**Impact:** an operator can receive a green-sounding artifact while the actual release is broken or unproven.

**How to tackle:** define a machine-readable evidence graph with required status allowlists per scope; bind every artifact to repository SHA, image digest, schema, generation time, and expiry; give `automation_failed`, stale, malformed, wrong-SHA, or missing evidence precedence over manual-closeout language.

**Exit criteria:** red CI, blocked SC3 inputs, missing restore drill, stale artifact, wrong commit, dependency-policy failure, or failed provider canary each forces a nonzero gate and an explicit blocker.

### REL-04 — Public/source copy attributes the paper's `0.929` to Blueprint

**Scopes:** `BASE`, `SC3`

**Evidence:** `VISION.md:95-99,282-289` calls it “the 0.929 rank fidelity Blueprint reports” and “Blueprint's 0.929 rank-fidelity figure,” contradicting `docs/SC3_EVAL_PROTOCOL.md:1-27`. No nonzero accepted-anchor or Blueprint correlation result was found.

**Impact:** materially misleading scientific and buyer claim.

**How to tackle:** state: “SC3-Eval reports Pearson r=0.929 under its published protocol; Blueprint has not measured an equivalent result.” Add a claims linter for `93% accuracy`, `Blueprint 0.929`, and unqualified rank-fidelity wording.

**Exit criteria:** every public/buyer surface includes source, metric, sample/unit, scope/split, and “not a Blueprint measurement” until a locked independent Blueprint study exists.

### REL-05 — Runtime dependencies contain known vulnerabilities relevant to untrusted media and HTTP intake

**Scopes:** `BASE`, `PTDP`, `LIVE`

**Evidence:** `pip-audit` found 27 advisories in 10 packages. High-relevance examples include Pillow 12.1.1 image-parser flaws, Starlette 1.2.1 request/form flaws, `cryptography` 46.0.5 issues, and resource-exhaustion issues in `idna`, `pyasn1`, and `urllib3`.

**Impact:** malicious capture/media/request input can reach vulnerable parsers or exhaust worker resources.

**How to tackle:** update to fixed releases, regenerate and commit a lock, add a policy-based dependency scan to CI and image build, then regression-test media and API paths.

**Exit criteria:** zero known critical/high runtime vulnerabilities or documented, time-bounded exceptions with compensating controls; image and Python SBOMs match the deployed digest.

### DATA-01 — Canonical intake bypasses raw-bundle integrity verification

**Scopes:** `BASE`, `SIM`, `PTDP`

**Evidence:** v3 verification exists at `ios_manifest.py:302-400`; materialization performs weaker readability checks at `materialization.py:1065-1119`, swallows JSON errors at `:28-34,1472-1521`, and critical fields can default at `ios_manifest.py:48-71`. Hash-manifest paths are not containment-checked at `ios_manifest.py:327-347`.

**Impact:** incomplete, corrupt, tampered, schema-invalid, or path-escaping bundles can produce derived artifacts.

**How to tackle:** one mandatory pre-derivation verifier: upload complete, IDs consistent, current schema valid, size/hash coverage complete, all paths contained and nonsymlinked, immutable intake digest persisted. Legacy input must be explicitly degraded/quarantined.

**Exit criteria:** tamper, missing hash, malformed sidecar, incomplete upload, absolute path, `..`, and symlink fixtures produce typed quarantine and zero derived writes.

### DATA-02 — Object indexing mutates immutable `raw/` truth

**Scopes:** `BASE`, `SIM`, `PTDP`

**Evidence:** `object_index_stage.py:1522-1575,1755-1834` writes crops, masks, keyframes, and manifests beneath `raw/`; `:1418-1429` rewrites `raw/manifest.json`, contradicting v3 hash coverage in `ios_manifest.py:291-299,344-347`.

**Impact:** first processing run changes capture truth and invalidates later verification; reruns are not trustworthy.

**How to tackle:** make `raw/` write-once; store derived object-index runs under content-addressed `pipeline/derived/...`; reference raw artifacts without changing them; recheck raw digest at stage boundaries.

**Exit criteria:** raw tree and digest are byte-identical before/after every stage and rerun.

### DATA-03 — Geometry readiness counts empty rows as valid tensors/poses

**Scopes:** `SIM`, `PTDP`, `SC3`

**Evidence:** `geometry_stage.py:1389-1469` emits pose/depth/confidence rows even with empty matrices/paths; `:1611-1648` derives coverage from row counts and defaults pose match; metric-depth truth is inferred from provider mode.

**Impact:** missing geometry can report 100% coverage and make world/eval artifacts look ready.

**How to tackle:** validate finite SE(3), contained existing decodable tensor paths, RGB/intrinsics shape match, ranges/units/timestamps, and one-to-one frame IDs; count only verified records.

**Exit criteria:** empty, corrupt, NaN, wrong-shaped, unitless, missing, or misaligned geometry yields zero verified coverage and blocks readiness; calibrated fixtures meet explicit error bounds.

### DATA-04 — 2D detections are fabricated into metric 3D “canonical truth”

**Scopes:** `SIM`, `PTDP`, `SC3`

**Evidence:** arbitrary metric boxes are synthesized at `object_index_stage.py:1071-1132` then marked observed/canonical at `:1178-1223`; proxy geometry is created and promoted at `object_geometry_stage.py:153-186,895-923,965-1048`; downstream accepts it as grounding/physics coverage at `evaluation_prep_stage.py:2179-2197` and `robot_eval_dataset.py:1187-1264`.

**Impact:** invented placement/collision/support geometry can determine task targets and success.

**How to tackle:** distinguish 2D observation, metric estimate, provider inference, and proxy; require calibrated depth/ray, multiview triangulation, or separately labeled estimate with uncertainty for metric placement. Proxies cannot satisfy canonical/physics gates.

**Exit criteria:** a 2D-only detection cannot produce metric placement or physics-ready status; known-depth/multiview fixtures meet translation, extent, and reprojection thresholds.

### DATA-05 — PTDP LeRobot/GR00T rows fabricate state, action, and time alignment

**Scopes:** `PTDP`, `SC3`

**Evidence:** one state is chosen at `post_training_data_package.py:2308-2324` and repeated for every action at `:2507-2657`; missing actions become a 1-D success bit at `:2537-2549`; timestamps are invented at 5 FPS at `:2564-2569`; arbitrary widths are padded at `:2635-2643`; output is named SC3 7D action at `:2974-2978`; the measured floor is only 0.5.

**Impact:** a native-looking training package can be mostly synthetic while counted as measured and SC3-compatible.

**How to tackle:** construct per-step rows only from real state/action/video timestamp joins; enforce robot-profile order/units/bounds and exact 7D delta-EE semantics for the SC3 lane; keep placeholders out of training-eligible rows.

**Exit criteria:** distinct state/action/time fixture round-trips exactly; missing state/action/time or wrong width blocks native status; repeated state is never counted as multiple measurements.

### DATA-06 — Raw-video URI still propagates into derived/buyer-facing artifacts

**Scopes:** `BASE`, `PTDP`, `PAID`

**Evidence:** `qualification.py:1172-1219,4973-4992` retains the original descriptor's `raw_video_uri`; `evaluation_prep_stage.py:1612-1624,2315-2325` falls back to and emits it.

**Impact:** unredacted media location may be disclosed or consumed despite privacy launch blockers.

**How to tackle:** pass only the privacy descriptor into scene memory; raw URI remains restricted internal evidence; buyer/runtime artifacts require privacy-processed media lineage.

**Exit criteria:** recursive artifact scan finds no raw path/URI in scene memory, site world, launch bundle, PTDP, delivery, or buyer output.

### DATA-07 — Missing/unproven consent and rights can still look buyer-ready

**Scopes:** `PTDP`, `PAID`

**Evidence:** PTDP detects missing consent at `post_training_data_package.py:645-711` but blocks quality only for revocation at `:4229-4258`; buyer readout treats rights-packet presence as passed at `buyer_package_readout.py:326-382`; tests at `tests/test_buyer_package_readout.py:9-110,390-450` expect readiness with critical rights/privacy booleans false.

**Impact:** review/delivery readiness can be shown without consent scope, privacy clearance, or provenance closure.

**How to tackle:** require valid scoped consent, rights gate, privacy clearance, provenance, expiration, and paid-use/DPA/subprocessor/access decisions as applicable; parse booleans strictly.

**Exit criteria:** missing, unknown, string-false, expired, scope-mismatched, and revoked consent each block with a distinct typed reason.

### DATA-08 — PTDP final archive is incomplete and only partially covered by checksums

**Scopes:** `PTDP`, `PAID`

**Evidence:** inventory/checksums finalize at `post_training_data_package.py:3600-3715`; handoff/delivery/access files are written later at `:4318-4337`; buyer readout and final manifest are written after archive creation at `:4728-4760`.

**Impact:** the shipped package omits its final truth/readout and cannot be verified as a complete deliverable.

**How to tackle:** stage all files, write final manifest/readout, inventory every member, sign a root digest, archive, then independently extract and verify.

**Exit criteria:** archive contains final readout/manifest/delivery/access files; every member except a defined self-referential root signature is indexed; there are no extras/missing hashes.

### DATA-09 — PTDP and geometry reruns can preserve stale files/status

**Scopes:** `SIM`, `PTDP`, `PAID`

**Evidence:** PTDP reuses an existing output tree and prior handoff/access status at `post_training_data_package.py:1461-1472,1623-1770,3674-3818`; blocked geometry does not quarantine prior tensors at `geometry_stage.py:1063-1170`.

**Impact:** ready-to-blocked reruns can keep old signed-access status, archive members, or usable tensors.

**How to tackle:** immutable run directories, content fingerprint, atomic current-run pointer, no status reuse without exact fingerprint/schema match, explicit zero-usable-artifact blocked manifest.

**Exit criteria:** changed rights/input rerun contains no prior member/status; synthetic-to-blocked geometry exposes no current tensor and preserves lineage that synthetic data existed.

### DATA-10 — Curation, semantic dedup, and action normalization are not load-bearing production stages

**Scopes:** `SIM`, `PTDP`, `SC3`

**Evidence:** production does not call `run_clip_curation_stage`, `run_semantic_dedup_stage`, or `build_action_normalization_manifest`; PTDP substitutes weaker metadata checks at `post_training_data_package.py:1016-1253` and shape/all-zero checks at `:285-339`.

**Impact:** packages can claim curation/dedup/normalization from self-attested metadata without intended pixel, trajectory, or statistical checks.

**How to tackle:** enforce one signed pipeline: materialize → construct clips → pixel/pose/action curation → production embedding/trajectory dedup → grounded captions → action alignment/normalization → PTDP; PTDP consumes only accepted IDs/manifests.

**Exit criteria:** missing canonical stage or self-attested substitute blocks high-quality/native status; rejected clips never enter exports.

### SC3-01 — The default “forward/inverse consistency” proof is only visual-motion smoke

**Scopes:** `SIM`, `SC3`

**Evidence:** `wam_episode_consistency_label_local.py:1-7,69-146,239-241` samples at most five frames and maps decodability/edges/pixel delta to both forward and inverse consistency without reading action values; `tests/test_wam_episode_consistency_label_local.py:96-121` enshrines a moving shape as proof; `groot_oscar_closed_loop_image.py:50-52,215-218,307-382` makes it the sealed-lane default.

**Impact:** unrelated/reordered actions or moving noise can pass a strict SC3-sounding gate and corrupt rank confidence.

**How to tackle:** rename this to `visual_motion_smoke`; prohibit it from emitting forward/inverse proof. Implement a shared-model inverse pass that recovers normalized action chunks and computes a heldout-calibrated numeric uncertainty/error signal.

**Exit criteria:** unrelated, reordered, sign-flipped, and scaled actions fail at the correct chunk; generic moving rectangles/noise cannot prove consistency; golden vectors reproduce the published formula.

### SC3-02 — External consistency scorer omits commanded actions and accepts boolean-only attestations

**Scopes:** `SIM`, `SC3`

**Evidence:** `oscar_isaac_closed_loop_eval.py:2495-2515` sends paths/count/type but not action vector/chunk/history and does not enforce declared keys; `oscar_cosmos_wam_evaluator.py:3165-3227` defaults evidence-used flags true and omits recovered actions/error/threshold requirements; `oscar_isaac_closed_loop_eval.py:2604-2607` turns termination into missing booleans.

**Impact:** a result containing only two `true` values can pass; infrastructure failure and model uncertainty are conflated.

**How to tackle:** version a strict scorer schema containing immutable action bytes/checksum, recovered chunks, per-dimension error, numeric uncertainty, calibration-set ID, threshold, evidence refs, and termination chunk.

**Exit criteria:** boolean-only, missing checksum/action, missing threshold, and forged-evidence responses fail; infra error and model abstention have separate states.

### SC3-03 — MMRV is mathematically wrong

**Scopes:** `SC3`

**Evidence:** `robot_eval_execution.py:4280-4294,4315-4328` averages normalized ordinal rank-position differences; SIMPLER/SC3 use the real-success-rate margin of inverted pairs and each policy's maximum violation.

**Impact:** emitted MMRV is incomparable to SC3, OSCAR, or SIMPLER.

**How to tackle:** port the reference pairwise-margin definition exactly and specify ties; rename the current metric `mean_normalized_rank_position_error` if it remains useful.

**Exit criteria:** hand-calculated and reference vectors match for ties, near-ties, one large inversion, and multiple inversions.

### SC3-04 — Four anchors across two policies unlock external/deployment accuracy

**Scopes:** `SC3`

**Evidence:** minima are 4 anchors/2 policy groups at `robot_eval_execution.py:44-45,4557-4589`; `:5271-5279` allows external/deployment claims whenever a score exists; `tests/test_robot_eval_job_orchestrator.py:5959-6039` proves two policies × two trials can yield Pearson/Spearman 1.0 and unlock.

**Impact:** two nonconstant aggregate points necessarily produce Pearson ±1, creating a perfect-looking but meaningless claim.

**How to tackle:** separate diagnostic calibration from publishable fidelity; require a preregistered estimand, at least seven independent checkpoints/policies, multiple criteria/splits, substantial matched trials per cell, locked test data, and lower-confidence-bound thresholds. Remove “deployment accuracy” wording.

**Exit criteria:** two-policy/four-anchor data remains diagnostic and can never unlock a public claim even at `r=1.0`.

### SC3-05 — Calibration pools the wrong unit of analysis

**Scopes:** `SC3`

**Evidence:** `robot_eval_execution.py:4255-4260` groups only by `policy_id`, discarding checkpoint, task, criterion, scene, split, and family; SC3 evaluates checkpoint × criterion points and distinguishes InD/OOD.

**Impact:** Simpson's paradox and task-specific failures can be hidden; result cannot be compared to 0.929.

**How to tackle:** key rows by policy/checkpoint × criterion × registered split/task family; report predeclared macro and micro estimands separately; never pool InD/OOD silently.

**Exit criteria:** a constructed Simpson-reversal fixture is detected and yields distinct declared macro/micro values.

### SC3-06 — Bootstrap confidence intervals are biased and at the wrong sampling level

**Scopes:** `SC3`

**Evidence:** `robot_eval_execution.py:4349-4400` lexicographically enumerates Cartesian samples and truncates the first 512; results depend on row order and resample already aggregated policy rows.

**Impact:** confidence intervals are not a valid uniform bootstrap and omit matched-trial/criterion uncertainty.

**How to tackle:** seeded hierarchical/cluster bootstrap over matched initial conditions, tasks/criteria, and policies with a documented interval method and enough replicates.

**Exit criteria:** permutation invariance, simulated coverage validation, preserved cluster IDs, and stable intervals at a preregistered replicate count (for example ≥10,000).

### SC3-07 — Claim permission is based on any calibration score, not rank-fidelity thresholds

**Scopes:** `SC3`, `PAID`

**Evidence:** `robot_eval_execution.py:5269-5279` enables claims when `sim_vs_real_calibration_score` exists; the score is `1-MAE`, not Pearson, and no Pearson/MMRV/CI threshold is enforced. `buyer_package_readout.py:579-620` similarly treats completed calibration, any anchors, and any score as claim-ready.

**Impact:** low/negative/undefined correlation can coexist with an “external accuracy allowed” flag.

**How to tackle:** remove generic permission booleans; emit metric-specific claim eligibility based on frozen N/design, finite/range checks, CI lower bounds, MMRV upper bounds, and scope. Buyer readout must project that decision, not recompute it loosely.

**Exit criteria:** missing/low/negative Pearson, excessive MMRV, wide CI, wrong split, or insufficient cells cannot enable a public claim.

### SC3-08 — Action normalization is dead production code and missing actions can be fabricated

**Scopes:** `SIM`, `PTDP`, `SC3`

**Evidence:** `action_normalization.py:198-270` is called only by tests; orchestrator only reads an optional file at `robot_eval_job_orchestrator.py:10451-10465`; dimension/timestamps are caller-optional at `action_normalization.py:218-237`; protocol trusts status/path at `sc3_eval_protocol.py:263-291`; `oscar_cosmos_wam_command_adapter.py:227-233` synthesizes zero/open-gripper actions when absent.

**Impact:** fabricated, untimed, wrong-dimensional, or unit-incompatible actions can look validated.

**How to tackle:** generate normalization from the exact consumed trace; require 7D delta-EE order/units/bounds, timestamps/control rate, corpus provenance, and stats hash; missing actions block rather than synthesize.

**Exit criteria:** no-action, 6D, wrong-unit, missing-time, NaN, out-of-range, or invalid zero-variance streams fail closed.

### SC3-09 — API-first DeepInfra lane is unconditioned text-to-video

**Scopes:** `SIM`, `SC3`

**Evidence:** `wam_compute_providers.py:255-273,567-625` sends prompt, output type, resolution, aspect, duration, and seed, but no observation, action, skeleton, robot state, camera, or context video.

**Impact:** output cannot causally differ by policy action and cannot rank policies.

**How to tackle:** classify it as `text_to_video_preview` and exclude it from Task Evaluation Runs; use only an endpoint/runtime that consumes actual observation and action conditioning.

**Exit criteria:** evaluator request snapshot includes observation/action identities and content; action-only perturbation changes conditioning; unconditioned endpoint cannot populate policy-eval artifacts.

### SC3-10 — Sealed OSCAR lane is not driven by each fresh learned-policy action

**Scopes:** `SIM`, `SC3`

**Evidence:** action conditioning is optional at `oscar_isaac_closed_loop_eval.py:1306-1353`; CLI omits the callback at `:3770-3776`; fallback skeleton is explicitly not the step action at `:3737-3741,1803-1810`; `groot_sonic_policy_endpoint.py:9-15,52-116,150-173` uses nonsemantic action projection and surrogate state.

**Impact:** learned policy may be queried again, but different policies can receive the same seed/target-driven transition.

**How to tackle:** convert every actual policy chunk through the real controller/FK to per-frame skeleton conditioning and carry generated proprio/state forward; reject proxies in evaluation mode.

**Exit criteria:** zero/positive/negative/reordered chunks yield corresponding distinct skeletons; `not_a_learned_robot_policy_action=true` blocks evaluator completion.

### SC3-11 — Manipulation episodes terminate on robot-root proximity

**Scopes:** `SIM`, `SC3`

**Evidence:** `oscar_isaac_closed_loop_eval.py:3012-3037,3076-3080` calls base distance <0.25 m task completion, while `:1959-2033` admits root reach is not manipulation success.

**Impact:** open/grasp/lift/place/button/faucet tasks can terminate before contact or state change.

**How to tackle:** use registered task-specific observable transitions: articulation angle, object pose, contact/grasp, lift height, containment, placement tolerance, etc.

**Exit criteria:** approaching a target never completes a manipulation task; registered state transition and tolerance are required.

### SC3-12 — Success labeling sees only the last step clip and accepts low-confidence booleans

**Scopes:** `SIM`, `SC3`

**Evidence:** `oscar_isaac_closed_loop_eval.py:2053-2104` selects only the final per-step video; OpenAI/Gemini sample 5/6 frames at `wam_generated_video_success_label_openai.py:33-36` and `...gemini.py:25-30`; `oscar_cosmos_wam_evaluator.py:2703-2798` treats boolean presence as review-grade without calibrated confidence or criterion subresults.

**Impact:** earlier failures disappear; `success=true, confidence=0.01` can become simulated success.

**How to tackle:** stitch or submit the full ordered episode; score explicit task criteria; require calibrated confidence/evidence and abstention; calibrate against blinded human labels with adjudication/inter-rater reporting.

**Exit criteria:** low confidence abstains; missing subcriteria cannot pass; earlier transient failure remains visible; full episode ordering is verified.

### RUN-01 — Native runtime API lacks authentication, tenant authorization, and path containment

**Scopes:** `BASE`, `SIM`, `LIVE`

**Evidence:** no auth middleware at `runtime_service_app.py:136-137`; unauthenticated mutation/control/WebSocket routes at `:202-232,277-398,448-476,605-655`; caller can set unsafe/debug flags at `:82-97,280-297`; caller IDs form paths at `native_runtime_backend.py:576-613,973-988,2033-2049`, and unsafe flag bypasses launchability at `:2036-2039`.

**Impact:** if exposed, anonymous users can register/control worlds/sessions, bypass gates, read media/state, and traverse tenant storage.

**How to tackle:** authenticate every non-health HTTP/WebSocket route; authorize tenant/site/session ownership; use canonical server IDs and `is_relative_to` containment; remove production API control of unsafe/debug flags; add quotas/origin/expiry checks.

**Exit criteria:** anonymous=401, cross-tenant=403; traversal/absolute/symlink/encoded/confusable IDs cannot escape; unsafe flags cannot be set by public payload; WebSocket replay/cross-tenant tests fail.

### RUN-02 — Pub/Sub staging permits filesystem escape

**Scopes:** `BASE`, `LIVE`

**Evidence:** handoff fields are only nonempty strings at `pubsub_handoff_listener.py:42-95`; bucket/prefix/blob name become paths at `:98-117`; arbitrary absolute staged paths are accepted at `:925-952`.

**Impact:** compromised publisher/object writer can write/read outside staging or redirect orchestration to local files.

**How to tackle:** strict grammar for bucket/prefix/capture/blob; per-message root; resolve/contain every path; reject absolute/symlink paths; authenticate exact object prefix; enforce object count/size/nesting limits.

**Exit criteria:** traversal, prefix confusion, symlink, absolute, oversized, and malicious object-name fixtures fail before external write/read.

### RUN-03 — Pub/Sub delivery can duplicate billing and permanently lose blocked work

**Scopes:** `BASE`, `SIM`, `PAID`, `LIVE`

**Evidence:** read-then-write claim has no transaction/lease at `pubsub_handoff_listener.py:621-660`; fixed 600s ack deadline in `deploy/terraform/main.tf:716-731` is not extended; any returned result is recorded completed and acked at `pubsub_handoff_listener.py:719-725,755-807,877-880`.

**Impact:** redelivery can run/bill twice; failed/blocked work is acknowledged and lost.

**How to tackle:** atomic durable idempotency claim with owner/attempt/lease/heartbeat; extend ack deadline; ack only terminal success or permanent-invalid input; nack retryables and DLQ exhausted work; idempotent output commit.

**Exit criteria:** duplicate delivery executes once; active/expired lease behavior is correct; blocked/transient results are not acked success; crash/restart yields one artifact/charge.

### RUN-04 — Robot-eval CLI/inbox converts blocked work into exit-0 processed success

**Scopes:** `BASE`, `SIM`, `PAID`, `LIVE`

**Evidence:** direct and inbox modes return 0 regardless result at `robot_eval_job_orchestrator.py:12053-12112`; processed marker is written for blocked results at `:11666-11765` and later suppresses retry at `:11601-11615`.

**Impact:** WebApp/control plane can record queued/processed while Pipeline rejected a paid job.

**How to tackle:** explicit terminal-success, permanent-invalid, retryable-blocked, and fatal-infrastructure states mapped to exit/queue semantics; processed marker only for terminal states; persist attempts/retry blockers.

**Exit criteria:** blocked request is nonzero/retryable/unprocessed and succeeds after dependency restoration.

### RUN-05 — Remote capture archives allow SSRF, unbounded download, link escape, and bombs

**Scopes:** `BASE`, `SIM`, `PTDP`, `LIVE`

**Evidence:** arbitrary local/file/HTTP(S) sources at `robot_eval_worker.py:522-555`; whole responses read without limits at `:526-549`; archive check covers member names only and uses `extractall` at `:558-595`, not link targets/types/count/expanded size.

**Impact:** metadata/internal-service access, memory/disk exhaustion, or extraction outside sandbox.

**How to tackle:** approved origins; disable `file://`; block loopback/private/link-local/metadata, redirects and rebinding; stream with byte/member/ratio limits; reject links/devices/FIFOs/special entries; nonroot read-only sandbox.

**Exit criteria:** metadata, redirect/private, rebinding, tar symlink/hardlink, bomb, huge member count, and interrupted download tests fail safely.

### RUN-06 — Buyer policy endpoint is an SSRF and unbounded-response surface

**Scopes:** `SIM`, `PAID`, `LIVE`

**Evidence:** endpoint validation is only HTTP prefix at `robot_eval_job_orchestrator.py:1068-1078`; unrestricted `urlopen` at `robot_eval_execution.py:2220-2238,2605-2612`; response has no byte/content cap.

**Impact:** entitled submitter can reach metadata/internal control planes or exhaust worker memory.

**How to tackle:** HTTPS plus endpoint ownership; DNS/IP validation before/after redirects; private/link-local/metadata denial and egress firewall; bound bytes, redirects, time, type, and JSON depth.

**Exit criteria:** metadata/localhost/RFC1918/redirect/rebinding/oversized/wrong-type/slow-stream fixtures fail closed.

### RUN-07 — GPU spend guard fails open and cannot reap DigitalOcean

**Scopes:** `LIVE`

**Evidence:** missing credentials and inventory errors become green empty inventory at `scripts/gpu_spend_guard.py:86-98,493-521,993-1030`; termination supports RunPod/Vast, not DO at `:707-728`; failed reap does not set failure at `:1084-1095`; stale warm marker has no freshness at `:570-590`.

**Impact:** spend can continue during credential/API outage or orphaned resource while guard reports success.

**How to tackle:** explicit success/failure/unknown provider states; fail closed for configured inventory; implement/verify DO deletion; nonzero/page on reap failure; expiring ownership leases; billing-export reconciliation.

**Exit criteria:** 401/timeout is red; owned DO orphan is deleted and absence verified; stale marker expires; failed delete remains red.

### RUN-08 — Provider race treats first exit-0 submission as execution proof

**Scopes:** `SIM`, `LIVE`

**Evidence:** candidate supplies arbitrary command/name at `robot_eval_provider_race_launcher.py:218-305`; sequential runner stops at first exit 0 at `:266-405`; `provider_race_execution_proven` derives from subprocess completion at `:507-542`; no terminal provider/artifact/teardown is required and timeout leaves resource lifecycle unresolved.

**Impact:** accepted submission can “win” despite later startup failure, no result, or ongoing spend; failover never runs.

**How to tackle:** fixed adapter registry/canonical IDs; state machine launch→resource ID→startup→execution→artifact validation→teardown; win only on valid fresh job-bound terminal artifacts; guaranteed cancel/teardown before failover.

**Exit criteria:** first provider submits then stalls, is verified terminated, and second runs; no artifact=no winner; crafted names cannot escape log/result directories.

### RUN-09 — Default “full deploy” omits required services but prints completion

**Scopes:** `LIVE`

**Evidence:** `deploy/scripts/deploy.sh:959-979` leaves `apply_terraform` commented; it builds/pushes privacy/video images at `:401-425` but does not deploy those GPU services, then passes blank-default URLs into the CPU job at `:36-71,558-608` and prints complete.

**Impact:** production fail-closed dependencies can be absent behind a successful deployment message.

**How to tackle:** one authoritative IaC path; deploy all required services/IAM first; read back real URIs/digests; authenticated health/canary; no manifest or success output on missing dependency.

**Exit criteria:** clean-project deploy creates all required services and references readback URIs; removal of one dependency hard-fails with no complete manifest.

### RUN-10 — Deployment provenance is operator-asserted and permits dirty or divergent source

**Scopes:** `LIVE`

**Evidence:** `deploy.sh:218-249` compares operator-supplied commit and nonempty URL without verifying workflow/repo/SHA/conclusion/collection; build at `:381-425` lacks clean-tree/remote-parity check; bypass needs only arbitrary text at `:227-233`.

**Impact:** fake evidence or uncommitted files can deploy under a trusted SHA label.

**How to tackle:** query/verify canonical GitHub workflow evidence, exact collection and SHA; require clean tree and remote parity; build once/deploy digest; independently approved expiring emergency override.

**Exit criteria:** fake URL, wrong workflow/SHA, reduced suite, red conclusion, dirty tree, remote divergence, and tag/digest mismatch all block.

### RUN-11 — Terraform/secrets path is incomplete and exposes plaintext secret material

**Scopes:** `LIVE`

**Evidence:** remote locked backend is commented at `deploy/terraform/main.tf:12-31`; tokens enter state/env at `:314-340,884-911`; World Labs key is not wired at `:844-852`; deploy writes secret-bearing `terraform.tfvars` at `deploy.sh:472-496`, does not ignore it, and places secrets in `gcloud` args at `:577-604`.

**Impact:** broken provider authentication and exposure in state, process listing, history, logs, or accidental commit.

**How to tackle:** remote encrypted locked state; Secret Manager/workload identity references; required-secret plan checks; no persistent secret tfvars or command-line values; rotate exposed-path secrets.

**Exit criteria:** secret scanner finds none in repo/state/log/process args; blank secret blocks plan; deployed canary authenticates; concurrent state is locked.

## P1 high-priority gaps

### Release engineering, dependency, and test-system gaps

| ID / scope | Confirmed gap and evidence | How to tackle and acceptance |
| --- | --- | --- |
| **REL-06** `BASE` `LIVE` | `uv.lock` exists locally but is ignored at `.gitignore:89`; CI uses unfrozen `uv sync`. Broad constraints can resolve differently across runs. | Track the lock; use `uv sync --frozen`; verify lock/pyproject consistency. A fresh build on two clean machines must resolve identical artifacts/hashes. |
| **REL-07** `BASE` | CI runs tests but not Ruff, type checking, dependency audit, Bandit/SAST, coverage thresholds, package build, or Docker build. Local Bandit reported 5 high and 94 medium candidates, several confirmed elsewhere in this audit. | Add separate mandatory lint/type/SAST/dependency/package/container jobs with tuned baselines. New high findings and coverage regressions block; suppressions require owner, reason, and expiry. |
| **REL-08** `BASE` | `pyproject.toml` advertises Python 3.10, 3.11, and 3.12, while all workflows test only 3.12. | Either narrow supported Python to 3.12 or add a compatibility matrix for 3.10–3.12 while retaining 3.12 as canonical release evidence. All advertised interpreters must install and run an agreed compatibility suite. |
| **REL-09** `BASE` `LIVE` | `Dockerfile` uses Python 3.11 while canonical evidence is 3.12; it copies `pyproject.toml`/`src` but not the declared README before install; mutable model revision is downloaded; production runs as root. `docker-compose.yml:56-105` references a nonexistent `development` target, while the CPU `base` target does not install the package. No CI image/Compose smoke exists. | Align runtime interpreter, copy required metadata, pin model revision/digest, run nonroot/read-only, define valid Compose targets, and build/smoke every production/dev image in CI. `docker compose config/build` and health/canary must pass in a clean checkout. |
| **REL-10** `BASE` `LIVE` | Actions use moving major tags; base/GPU images and model/source downloads are not comprehensively digest/revision pinned; no SBOM, signature, provenance, or license gate. Dependabot security updates and repository vulnerability alerts are disabled; no CodeQL analysis exists. | Pin actions to commit SHAs, images to digests, source/model revisions to immutable hashes; generate CycloneDX/SPDX SBOM, sign images/provenance, enable dependency/security tooling and CodeQL or equivalent. Deployed digest must verify signature and SBOM. |
| **REL-11** `BASE` | No `LICENSE` file exists despite MIT package metadata; no `SECURITY.md` or CODEOWNERS. Setuptools reports deprecated license metadata. | Add the actual approved license, vulnerability-reporting policy, supported-version policy, and ownership; move to SPDX license expression. Package wheel must contain correct license metadata/files. |
| **REL-12** `BASE` | 364 tracked `ops/city-launch-runs` files contain stale status snapshots and absolute local paths/usernames; examples contain unsupported Python 3.13 test output. | Move operational evidence to an access-controlled artifact store or a clearly versioned/redacted fixture subset; add retention, schema, SHA/freshness, and disclosure checks. Source tree must contain no personal absolute paths or stale artifacts usable as current proof. |
| **REL-13** `BASE` `SIM` | Full local lane skips torch-dependent OSCAR/provider tests, native LeRobot export, and locally Pub/Sub integration; GitHub lanes do not exercise real GPU/provider images or deployed services. | Define mandatory CPU, container, GPU-canary, Pub/Sub emulator/integration, and native LeRobot lanes. A skipped critical capability must appear as a blocker for the scope that depends on it, not silent green. |
| **REL-14** `BASE` | Source is ~352k Python LOC with 145 CLIs and several 10k–13k-line orchestration modules (`robot_eval_job_orchestrator.py`, `robot_eval_execution.py`, kitchen/MuJoCo runners). Defects span duplicated status/claim logic. | Establish module budgets and extract shared state machines, metric implementations, validators, and claim decisions behind typed schemas. Require characterization tests before splits and prohibit new duplicated claim logic. |
| **REL-15** `BASE` | Free-form `workflow_dispatch.pytest_args` is interpolated/unquoted at `.github/workflows/full-test-lane.yml:7-13,100-107`; it can execute shell syntax or reduce a run later cited as “full.” | Remove free-form shell input or validate constrained choices and pass safe argv; canonical release evidence must bind the exact collected test IDs/count. Shell metacharacters and `-k`/path reduction cannot qualify. |

### Capture, package, and output-quality gaps

| ID / scope | Confirmed gap and evidence | How to tackle and acceptance |
| --- | --- | --- |
| **DATA-11** `PTDP` `SC3` | Semantic dedup defaults to a downsampled-pixel fixture encoder, embeds one middle frame, compares absolute world trajectories, keeps unembeddable clips, and still completes (`semantic_dedup_stage.py:93-129,205-368,487-536`). | Require a production SigLIP/DINO-class encoder, multiple keyframes, SE(3)-relative/aligned trajectories, and exclude/review unverifiable clips. Translated/rotated duplicates must cluster; missing evidence cannot pass production dedup. |
| **DATA-12** `SIM` `PTDP` `SC3` | Temporal alignment lacks a canonical timebase; raw IDs are returned verbatim, heterogeneous timestamp keys/units are mixed, ID matches can count without valid deltas, and only iPhone uses the explicit gate (`materialization.py:143-266`). | Define stream timebase ID/unit/origin, monotonic sequence, canonical IDs, one-to-one joins, duplicate/drop ledger, and p50/p95/max per modality. Mixed ms/s, epochs, duplicates, and nonmonotonic fixtures must block. |
| **DATA-13** `SIM` `SC3` | Camera validation checks positive dimensions/focal lengths and 4×4 finite shape; native runtime invents intrinsics; missing extrinsics/reprojection error can remain warning or add confidence (`cosmos_training_export.py:49-97`; `native_runtime_backend.py:290-305`; `eval_ready_task_grounding.py:606-690`). | One validator for FOV/principal point/resolution, SE(3) determinant/orthonormality/last row, units/frames, and reprojection error. NaN, absurd FOV, reflection/shear, missing extrinsics, or frame mismatch blocks projection-ready status. |
| **DATA-14** `SIM` `SC3` | The “FK” stage copies supplied Cartesian landmarks, assumes identity extrinsics, emits only frame 0, records `urdf_or_mjcf_fk_solver_executed=False`, yet may complete (`eval_ready_task_grounding.py:720-907,1396-1468`). | Parse URDF/MJCF; validate joint names/order/limits/units; solve every aligned step; transform through calibrated frames; gate visibility/continuity. A reference sequence must reproduce known landmarks; wrong order/transform/time blocks. |
| **DATA-15** `PTDP` `SC3` | There is no production clip-caption stage. LLM enrichment accepts broad schemas and unvalidated mappings (`capture_enrichment_llm.py:69-145,243-328`). | Add post-curation grounded captioning over hashed sampled frames with model/prompt/version; strict JSON Schema, vocabulary, object/task ID checks, retry/exclusion. Unsupported geometry/task claims and missing captions block premium/high-quality package status. |
| **DATA-16** `SIM` `SC3` | Missing task becomes “inspect the first object”; missing mask/keypoint/crops and proxy-state failure can remain warnings (`eval_ready_task_grounding.py:265-291,1299-1465`). | Buyer-grade runs require a versioned task contract: target, source/destination or transition, evidence, metric, tolerance, evaluator mapping. Auto-generated tasks stay support-only; ambiguous/missing grounding blocks. |
| **DATA-17** `PTDP` `PAID` | Buyer readout counts failure-label artifact presence, while PTDP writes raw/unreviewed hypotheses into training formats without consuming the review audit (`buyer_package_readout.py:513-532`; `post_training_data_package.py:1865-1964,3098-3155,3400-3424`). | Separate raw hypothesis and accepted training-eligible ledgers; require review status, evidence, failed-attempt coverage, and provenance. Pending/nonreviewable labels never enter native training export; a reviewed zero attestation is required for zero. |
| **DATA-18** `PTDP` `PAID` | Any absolute existing clip path can be accepted and copied (`post_training_data_package.py:2016-2169`). | Permit canonical contained roots only; reject symlinks/escape; probe actual media, extension, size/duration, digest lineage. `/etc/passwd`, `..`, symlink, renamed text, corrupt, and oversized inputs must block before copy. |
| **DATA-19** `SIM` `PTDP` | Shared `write_json` and many JSONL/archive writes truncate targets directly (`common.py:144-154`); output trees lack a common lock/commit protocol. | Same-filesystem temp, flush/fsync, atomic replace, directory fsync where needed, run lease/CAS, final commit marker. Kill-at-write leaves old complete or no new committed state; concurrent runs never mix. |
| **DATA-20** `SIM` `PTDP` | Cross-session RANSAC uses unseeded randomness and weak pose validation (`frame_alignment_stage.py:436-520,824-842`). | Derive seed from input digest, validate full SE(3), deterministic tie-breaks, output transform fingerprint. Repeated runs are byte-stable and malformed transforms block. |
| **DATA-21** `BASE` `SIM` | Materialization computes candidacy with different argument sets for descriptor vs readiness (`materialization.py:1418-1468`). | Compute one canonical typed decision and project all surfaces. Property-based fixtures across source types must agree exactly. |
| **DATA-22** `PTDP` | Dedup is pairwise/in-memory; PTDP expands rows in memory, duplicates videos, rescans exports, and synchronously tars (`semantic_dedup_stage.py:267-349`; `post_training_data_package.py:2507-2657,3674-3775`). | Streaming/bounded processing, ANN index, content dedup/object references, disk quota/preflight, resumable manifests, cancellation/backpressure. Maximum-size workload must meet explicit memory/disk/time SLOs without duplicate retry artifacts. |
| **DATA-23** `SIM` `LIVE` | OSCAR smoke uses `pickle.load` on checkpoint asset `caption.pickle` (`oscar_wam_provider_bundle.py:4677-4726`). | Replace with JSON/text, pin checkpoint/asset digests, and validate before inference. No pickle deserialization; changed digest blocks. |
| **DATA-24** `SIM` `LIVE` | OpenVLA loads operator/remote checkpoints with `trust_remote_code=True` and no revision/digest allowlist (`openvla_policy_command_adapter.py:442-451`); DINO/model downloads are also unpinned in multiple paths. | Only approved immutable revisions/digests; isolate remote-code models in nonroot/no-network sandbox or remove trust; signature/license scan. Unapproved revision cannot execute or access worker secrets/network. |

### SC3/evaluator-fidelity gaps

| ID / scope | Confirmed gap and evidence | How to tackle and acceptance |
| --- | --- | --- |
| **SC3-13** `SC3` | Multiview readiness is camera-count/readable-file based, not synchronized geometric consistency (`sc3_eval_protocol.py:167-179,344-350`; `wam_derived_observation_harness.py:339-408`). OSCAR closed loop is egocentric-only. | Schema for camera IDs, shared timestamps/skew, intrinsics/extrinsics, simultaneous indices; joint generation plus correspondence/occlusion/re-entry checks. Duplicate/unrelated/unsynchronized/swapped cameras must fail. |
| **SC3-14** `SC3` | SC3's 25/24/16 horizon decoupling exists as metadata, not enforced trace behavior (`sc3_eval_protocol.py:37-45`; `cosmos3_wam_command_adapter.py:82-103`). | Executable action/prediction/retention/requery contract with control rate and trace. Assert 25 proposed, 24 supplied/predicted, 16 retained/executed, remainder discarded, correct requery timestamp. |
| **SC3-15** `SC3` | Cosmos3 adapter proves model identity/MP4 existence, not an SC3-trained forward/inverse/cross-view model (`cosmos3_wam_command_adapter.py:1-15,289-506`); Cosmos3 is not wired into closed loop. | Require checkpoint-attested training dataset/split/objective hashes and golden functional probes for each mode. Base checkpoint without SC3 fine-tuning remains aspirational. |
| **SC3-16** `SIM` `SC3` | Fixture ranking accepts one attempt per scenario, point rates and a fixed tie band, and can claim a single best policy without decision-grade uncertainty (`wam_fixture_evaluator.py:1633-1848`). | Explicit replicate/seed dimension; ≥20 trials per policy×condition for decisions and preferably ~36–37 matched for SC3-comparison claims; interval/posterior winner logic and multiplicity control. One-trial 100% vs 0% is inconclusive. |
| **SC3-17** `SC3` | Autoresearch can reuse a single run for train/heldout and repeatedly optimize on heldout (`policy_autoresearch.py:378-405,1495-1554,1731-1754`). | Grouped train/dev/one-time locked test by site/scene/task/object/source trajectory/policy lineage; hash/freeze splits and dedup quarantine. Single run cannot promote; search cannot access locked-test outcomes. |
| **SC3-18** `SIM` `SC3` | Known-order ladder reads `score` while scorecards emit `predicted_success_rate`; ties count as recovered; noise actions are unbounded (`policy_ranking_ladder.py:242-343`; `wam_fixture_evaluator.py:1718-1749`; `noise_degraded_policy_command_adapter.py:109-131`). | Schema-bind actual field, make ties inconclusive, clip/validate actions, use multiple seeds and empirical accepted ground truth. All-equal scores fail discrimination. |
| **SC3-19** `SIM` `SC3` | `sc3_eval_protocol.json` trusts object/file presence and supplied metric numbers; camera count and positive anchor count can make readiness/measured status, and `or` can drop valid zero MAE (`sc3_eval_protocol.py:167-220,326-452`). | Recompute from validated joined rows and actual hashed files; finite/range/N/provenance checks; split `protocol_defined`, `runtime_ready`, and `claim_ready`. NaN/forged numbers/empty objects/duplicate cameras fail; zero remains zero. |
| **SC3-20** `SC3` | OOD is keyword matching and has no registered axes across site/task/policy family/embodiment/camera/visual/dynamics/contact (`wam_fixture_evaluator.py:652-730`). | Frozen OOD axes with leave-one-group tests and per-axis correlation/MMRV/error/abstention. No pooled OOD headline may hide a failed axis. |
| **SC3-21** `BASE` `SC3` | OSCAR and SC3 benchmark cards/metrics are blended; README uses `SISR delta` for what OSCAR reports as success-rate difference/MAE. | Separate benchmark cards, model/protocol/label/sample units and names; rename to `success_rate_difference_pp`. Claims linter prevents metric transfer. |
| **SC3-22** `SC3` | No current frozen Blueprint study contains nonzero accepted anchors, numeric Pearson/MMRV, seven-policy matched benchmark, calibrated inverse threshold, or a real three-view SC3 checkpoint/run. | Treat as external proof, not code-complete status. Execute the study in the final wave below and publish raw per-cell results, N, CIs, failures, abstention and exact digests. |

### Runtime, queue, and deployment hardening gaps

| ID / scope | Confirmed gap and evidence | How to tackle and acceptance |
| --- | --- | --- |
| **RUN-12** `SIM` `LIVE` | Caller `job_id` forms output paths without strict grammar/fingerprint/lock; CLI argument/request mismatch is not rejected and prior job dirs can be reused (`robot_eval_job_orchestrator.py:9895-9980,11672-11690,11865-11868`). | Server-generated canonical IDs, containment, immutable request fingerprint, atomic claim, mismatch rejection, per-attempt dirs. Traversal/collision/concurrent/mismatch fixtures cannot overwrite or mix. |
| **RUN-13** `LIVE` | Intake HMAC nonce cache is process-local (`live_pipeline_intake_service.py:53-54,218-281`). | Shared atomic nonce/idempotency store with TTL, scoped client identity, bounded skew. Replay is rejected across two processes and restart. |
| **RUN-14** `LIVE` | Intake accepts caller `capture_root`/absolute paths and shared HMAC has no tenant/site scope (`live_pipeline_intake_service.py:370-390,1186-1242`; `robot_eval_job_orchestrator.py:11373-11392`). | Map authenticated tenant/site IDs to server roots; never accept raw roots; scoped service identity. Tenant A cannot name tenant B root with valid tenant-A signature. |
| **RUN-15** `LIVE` | Intake has no body/rate/queue/storage limits; health exposes topology/paths; trigger uses `shell=True` (`live_pipeline_intake_service.py:996-1030,1083-1161`). | Body/depth, rate, concurrency and storage quotas; sanitized health; fixed argv/systemd activation. Test 413/429/503; health has no paths; payload cannot alter command parsing. |
| **RUN-16** `LIVE` | Provider launcher timeout records failure but does not prove local process-tree/provider teardown (`robot_eval_provider_launcher.py:901-958`). | Isolated process group, provider ID persisted immediately, cleanup in `finally`, provider absence verified. Timeout leaves no child/resource/spend. |
| **RUN-17** `LIVE` | Warm worker uses a single overwriteable object and client-local sequence; signed/local `request_id` forms result filename (`warm_render_server.py:163-334`). | Object-per-job durable queue/broker, server idempotency keys, canonical IDs/containment. Concurrent producers/restart lose no jobs; duplicate is idempotent; crafted ID cannot escape. |
| **RUN-18** `BASE` `LIVE` | Privacy/video runner authorizes when token is missing and binds all interfaces (`privacy_runner_service.py:23-43,144-159`; `video_to_world_runner_service.py:19-39,135-150`). | Production startup requires private IAM identity or nonempty token, constant-time compare, bounded request. Anonymous denied and blank auth prevents startup. |
| **RUN-19** `LIVE` | Systemd units default to root and lack strong filesystem/capability/resource isolation; staging is mode 0755. | Dedicated users, least privilege, `ProtectSystem/Home`, `ReadWritePaths`, capability/address-family/resource limits, private modes. Meet a documented `systemd-analyze security` target. |
| **RUN-20** `LIVE` | Shared JSON/ledger writes are direct/non-atomic (`common.py:144-154`; `pubsub_handoff_listener.py:362`). | Atomic temp/fsync/replace plus CAS/version for mutable ledgers. Fault-injection leaves prior or complete new JSON only. |
| **RUN-21** `LIVE` | Failure alert uses `ALIGN_RATE > 5` while prose says >5 failures/5m (`deploy/terraform/main.tf:1537-1568`), likely a units error. | Use sum/delta over 5m and live-test notification. Six failures page; four do not. |
| **RUN-22** `LIVE` | US-only beta policy conflicts with default `europe-west1` deployment (`docs/BETA_DATA_RESIDENCY_TRANSFER_POLICY_2026-07-09.md:1-10`; `deploy.sh:37-40`; Terraform regional resources). | Remove EU for beta or complete transfer terms/controls and enforce residency as policy-as-code. A US-only test must fail if non-US resources/routes exist. |
| **RUN-23** `LIVE` | Documented `$5,000 hard stop` is only an alert (`docs/BETA_CAPACITY_STORAGE_AND_RATE_LIMIT_ASSUMPTIONS_2026-07-08.md:71-87`; Terraform budget alerts at `main.tf:679-707`). | Admission lock, controlled drain/teardown, billing reconciliation, audited override. Threshold crossing must stop new paid work and page. |
| **RUN-24** `LIVE` | Two deployment paths (manual gcloud and Terraform) diverge, so no authoritative topology exists. | Choose one source of truth; disable the other; generate topology/evidence from cloud readback. Drift detection must fail launch when declared and live resources differ. |

## P2 defense-in-depth and maintainability backlog

These do not override the scoped P0/P1 work. They should be completed before broad public scale, and promoted if the final threat model exposes them directly.

| ID / scope | Gap | How to tackle and acceptance |
| --- | --- | --- |
| **P2-01** `BASE` | Install surfaces drift: `requirements.txt` installs `.`, `requirements-geometry.txt` carries a separate unpinned set, and the ignored lock is described as canonical in docs. | Declare one authoritative dependency graph/lock and generate compatibility exports from it. CI verifies no drift. |
| **P2-02** `BASE` | Sim-only workflow watches `uv.lock`, but it cannot be changed in Git while ignored; documentation claims lock alignment that repository history cannot enforce. | Track the lock and test a deliberate lock-only change triggers all required lanes. |
| **P2-03** `BASE` | Package metadata has no project/home/source/support URLs and uses deprecated license-table/classifier forms; no repository license file is included. | Complete PEP 621 metadata with SPDX and license files; wheel/sdist metadata audit passes without deprecation warnings. |
| **P2-04** `BASE` | July 2 goal/spec documents mark slices “planned,” “proposed,” or “implemented” inconsistently with partial current code. Old audit documents are easy to mistake for current launch state. | Maintain one current gap ledger with `open/partial/closed/reopened`, commit/evidence, scope, and supersession links. Stale audits get a prominent superseded banner. |
| **P2-05** `BASE` `LIVE` | GitHub workflow artifacts retain full-lane evidence only 14 days, too short for a durable release/benchmark audit trail. | Copy release evidence to immutable retention keyed by SHA/image digest; define retention/legal policy and checksum/signature. A past release remains independently reproducible after Actions expiry. |
| **P2-06** `BASE` | Bandit produced 1,254 findings (5 high, 94 medium, 1,155 low) with no repository baseline/triage policy. This audit confirmed several high/medium paths, but not every scanner finding was manually validated. | Triage by reachability; fix confirmed; baseline false positives with owner/reason/expiry. New high/medium findings block until reviewed. |
| **P2-07** `SIM` | Local/dev Compose exposes an allow-root Jupyter service on all interfaces and references inconsistent `output`/`outputs` paths; it has no documented auth/network boundary. | Bind loopback by default, require token, remove root, align storage paths/quotas, and mark dev-only. Automated Compose security/smoke test passes. |
| **P2-08** `BASE` | The full suite emits four `runpy` warnings because CLI modules are already present in `sys.modules` before `python -m` execution (Lightwheel, both OSCAR command adapters, and the robot-eval orchestrator); Python warns behavior may be unpredictable. | Remove eager package-import side effects and isolate module entry points. Canonical full-suite and `python -m` smoke runs must complete without these warnings or double initialization. |

## External, live, and manual evidence blockers

These are not all source-code defects. They are facts the repository cannot prove locally and therefore claims that must remain blocked until current evidence is attached to the exact release digest.

| ID / scope | Missing proof | Required evidence to close |
| --- | --- | --- |
| **EVID-01** `SC3` | No Blueprint-measured rank-fidelity result. | Frozen study with ≥7 independent checkpoints/policies, explicit criteria, matched conditions/replicates, locked InD/OOD test, raw per-cell outputs, human-label protocol, correct Pearson/Spearman/MMRV, hierarchical CIs, abstention/coverage and exact model/code/data digests. |
| **EVID-02** `BASE` `PTDP` | No current production mobile v3 capture has proven upload-complete → hash verification → raw immutability → privacy → real geometry → curation/dedup/caption/action join → package extraction → buyer load. | At least one clean, current, audited end-to-end run for each public capture source, with every intermediate digest and no manual file injection. |
| **EVID-03** `SIM` `LIVE` | Fresh real-provider WAM probe is blocked before provider execution. | Authenticated canary on the exact image/digest with SAM3, depth, pose, action-conditioned WAM, policy endpoint, validated terminal artifacts, semantic label, and teardown/spend proof. |
| **EVID-04** `BASE` `PTDP` | Privacy redaction is contract-tested but no current false-negative/false-positive performance study was found. | Frozen representative privacy corpus, face/person/screen/plate classes as applicable, recall/precision/error review, human escalation threshold, and raw-to-derived leakage scan. |
| **EVID-05** `LIVE` | Capture-truth restore drill is missing; fresh external-alpha gate fails. | Live lifecycle/backup/PITR readback plus restore drill measuring RPO/RTO, checksum equality, permissions, audit trail, deletion/legal-hold propagation, and named owner. |
| **EVID-06** `LIVE` | No current 25-concurrent-user/15-minute soak or 100-user capacity evidence. Dry-run validates assumptions only. | Representative payload mix, queue/Firestore/storage/provider latency, error/retry/duplicate rate, memory/disk, cold starts, package duration, spend, teardown, and SLO pass/fail under sustained load. |
| **EVID-07** `LIVE` | No current incident-response drill. | Paging-channel test, stalled-job alert, DLQ replay, worker crash/recovery, credential revocation, provider outage/failover, orphan cleanup, rollback, and named on-call/communications record. |
| **EVID-08** `LIVE` | Retention/data-residency documents are declarations, not applied topology proof. | Cloud readback showing exact regions, lifecycle, access logging, backup, encryption, service accounts, egress, and no resource outside allowed policy; drift check attached to release. |
| **EVID-09** `PAID` | No live buyer payment, Stripe Connect payout, exception reconciliation, KYC decision, or finance ownership proof. | Live-mode payment and payout settlement, webhook/ledger reconciliation, connected-account readiness, provider decision, failed-payment/payout drill, and named finance review owner. |
| **EVID-10** `PAID` | No live post-purchase entitlement/signed-URL fetch of a real current package. | Authenticated buyer session with purchase/job/package ID continuity, least-privilege short-lived URL, checksum verification, expiry/revocation/cross-tenant negative tests, and access audit log. |
| **EVID-11** `PAID` | Legal/consent/DPA/subprocessor/commercial-use signoff is open. | Signed current consent/rights posture, use-class mapping, DPA/subprocessors, US-only or transfer terms, retention/deletion/DSAR/takedown procedure, and counsel/privacy owner signoff. |
| **EVID-12** `PAID` | Real-device discovery/reservation/upload/job-ID continuity is not proven for every marketed source. | Screen recording and backend trace for each public device/source. Keep unproven glasses/Android paths internal rather than blocking an iPhone-only scope. |
| **EVID-13** `LIVE` | No authoritative deployment manifest/cloud topology at current SHA. | One IaC apply, immutable image digests, actual service URI/IAM readback, authenticated health/canary, monitoring, rollback, and clean teardown evidence. |
| **EVID-14** `PHYSICAL` | No physical-robot task success, safety, or deployment approval. | Required only if making that claim: controlled real-robot trials, task/state criteria, safety/risk review, operator approvals, incident/stop behavior, and owner-system evidence. It is not a sim-only launch blocker. |

## Recommended implementation sequence

### Wave 0 — Stop false release and benchmark claims

1. Correct `VISION.md` and all public/buyer copy; add the claims linter.
2. Fix the ignored-fixture clean-checkout failure and make both current CI lanes green.
3. Protect `main` and bind launch/deploy gates to canonical current evidence.
4. Upgrade vulnerable dependencies; track/freeze the lock; introduce dependency/SAST/image gates.
5. Disable the local-CV forward/inverse proof, four-anchor public-claim unlock, and incorrect MMRV output.

**Wave exit:** no public claim or deployment can use red/stale/wrong-SHA evidence; no false SC3 proof is emitted.

### Wave 1 — Preserve capture truth and stop silent work loss/security exposure

1. Mandatory immutable raw verifier and write-once raw tree.
2. Fix geometry false coverage and fabricated canonical 3D truth.
3. Transactional Pub/Sub claim/lease/ack/DLQ and robot-eval exit/status taxonomy.
4. Authenticate/authorize runtime and intake; close path traversal, SSRF, unsafe archive, and endpoint-response surfaces.
5. Fix provider lifecycle/teardown and spend guard fail-open behavior.
6. Atomic state writes and immutable run directories.

**Wave exit:** corrupt/unauthorized input produces no derived artifact; retryable work is never acknowledged as success; duplicate delivery is idempotent; every external request is scoped and bounded.

### Wave 2 — Build truthful training and buyer outputs

1. Wire canonical curation, production semantic/trajectory dedup, captioning, and action normalization.
2. Implement real per-step state/action/video joins and exact robot-profile semantics.
3. Close raw-URI leakage and make rights/consent/privacy provenance load-bearing.
4. Reorder PTDP finalization so the complete final package is signed and independently verified.
5. Split raw vs reviewed failure labels and close arbitrary-file copy.

**Wave exit:** no synthetic placeholder is counted as measured training data; archive is complete, signed, current, privacy-safe, and rights-valid.

### Wave 3 — Correct evaluator statistics and decision logic

1. Correct unit of analysis and MMRV.
2. Replace biased bootstrap with hierarchical/cluster uncertainty.
3. Add replicate/seed model and interval-based ranking; fix known-order ladder.
4. Freeze train/dev/locked-test splits and OOD axes.
5. Make protocol metrics recomputed evidence, not caller declarations.

**Wave exit:** all metric implementations match reference/golden vectors; small-N results remain inconclusive; no heldout leakage; claim eligibility is metric/scope specific.

### Wave 4 — Implement the actual SC3-like runtime recipe

1. Exact normalized 7D action chunks and executable 25/24/16 horizon.
2. Per-action FK/skeleton and generated state carried through closed-loop requery.
3. A genuinely trained/attested shared forward/inverse/cross-view model.
4. Synchronized calibrated multiview generation and consistency.
5. Numeric inverse recovery/uncertainty with a heldout threshold.
6. Task-state termination, full-episode criterion review, confidence/abstention and human calibration.

**Wave exit:** action perturbations cause the expected conditioned changes; every consistency/termination value is traceable and calibrated; no proxy/text-to-video lane can impersonate evaluation.

### Wave 5 — Make deployment and operations real

1. One authoritative IaC path, remote locked state, Secret Manager/workload identity, nonroot signed images.
2. Live provider canaries, topology readback, health, rollback and cleanup.
3. Capacity/soak, backup/restore, retention/deletion, incident/DLQ/provider-failover drills.
4. Complete paid/legal/device evidence only for the public scope being sold.

**Wave exit:** the exact release digest is observable, recoverable, capacity-tested and fail-closed; manual/live claims have current artifacts and named owners.

### Wave 6 — Run the frozen external-fidelity study

Execute `EVID-01` only after Waves 0–5. Publish the measured result with N, exact scope, CIs, MMRV, failure cases, abstention/coverage and locked artifacts. Do not tune the evaluator to “hit 0.929” on the final test; treat the final number as a measured outcome.

## Definitions of done by launch scope

### Evaluator-bounded sim-only beta

- All open `BASE` and `SIM` P0 items closed.
- No local-CV/VLM/media-arrival signal is labeled forward/inverse consistency or semantic success without its proper evidence.
- Every run distinguishes: request accepted, provider launch, artifact arrival, artifact validity, review-media validity, evaluator result, buyer delivery, and semantic task success.
- `correlation_not_measured` is allowed and expected without accepted real anchors.
- Current clean-checkout CI and sim gate are green and branch-protected.
- Security negative tests, quotas, idempotency, retry and teardown pass.
- Public copy is evaluator-bounded and does not imply physical deployment or SC3-equivalent correlation.

### Buyer PTDP / paid marketplace

- Sim-only definition plus all `PTDP` and in-scope `PAID` P0 items.
- Complete signed archive and buyer readout; no raw URI/path; no synthetic rows counted measured.
- Valid current consent/rights/privacy/commercial-use scope and takedown/revocation behavior.
- Live purchase→entitlement→fetch and payout/exception evidence for whatever financial flows are enabled.

### SC3-like correlation claim

- Sim-only definition plus all `SC3` P0/P1 items and `EVID-01`.
- Claim text names Pearson/Spearman/MMRV separately, reports sample unit/N and CIs, and describes InD/OOD/task/policy scope.
- Independent reproduction from frozen artifacts succeeds.
- A result below 0.929 is reported honestly; a result near/above 0.929 is not generalized beyond its registered scope.

### Physical deployment/safety claim

- Separate `PHYSICAL` evidence and approvals. Nothing in the sim-only/SC3 lanes automatically closes it.

## Audit ledger totals

This pass records **107 launch-relevant items**:

- **38 P0 scoped launch blockers**
- **47 P1 high-priority gaps**
- **8 P2 defense-in-depth/maintainability gaps**
- **14 external/live/manual evidence blockers**

The count is intentionally scope-aware: a `SC3` P0 blocks the numerical fidelity claim but is not automatically a physical-robot requirement; a `PAID` P0 does not block a free/internal sim-only lane if that paid feature is disabled and unmarketed.

## Verification commands and artifacts used

Representative commands run during this audit:

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider -m "not slow"
.venv/bin/python -m pytest -q -p no:cacheprovider -m "" -rs
.venv/bin/python -m ruff check src/blueprint_pipeline scripts tests functions
.venv/bin/python -m compileall -q src scripts tests functions
.venv/bin/python -m pip check
.venv/bin/python scripts/validate_python_interpreter_matrix.py --assert-current
uvx pip-audit --path .venv/lib/python3.12/site-packages --format json
uvx bandit -r src/blueprint_pipeline functions scripts -q -f json
uv build --wheel --out-dir /tmp/blueprint-dist-audit
terraform -chdir=deploy/terraform fmt -check -recursive
terraform -chdir=deploy/terraform validate
.venv/bin/python scripts/run_paid_marketplace_launch_gate.py \
  --json-out /tmp/blueprint-paid-gate-audit.json \
  --markdown-out /tmp/blueprint-paid-gate-audit.md
gh run list --commit 7a462e070cc1a55b2bb829dd85620e92836eb20f
gh api repos/ognjhunt/BlueprintCapturePipeline/branches/main/protection
gh api repos/ognjhunt/BlueprintCapturePipeline/rulesets
```

Additional focused suites covered runtime/intake/PubSub/provider/deploy paths (**186 passed, 1 skipped, 16 deselected**), capture/output/PTDP paths (**101 passed, 11 deselected**, plus **11 slow tests passed**), and SC3/evaluator paths. Their green results do not close the negative-path defects documented above; several current tests explicitly encode the faulty behavior.

## Important strengths to preserve during remediation

- Stable provider-agnostic and robot/policy adapter contracts.
- Raw capture as the highest-authority evidence and generated artifacts as support layers.
- Explicit fixture/local/provider/live claim boundaries in many modules.
- `correlation_not_measured` and no-winner states as valid outcomes.
- Separate sim-only and physical-deployment scope.
- Signed/job-bound artifact direction, exact join keys, and provider reliability manifests.
- Strong existing test volume and many focused fail-closed contracts.

The remediation should strengthen these seams, not replace them with a provider-specific monolith or turn optional readiness/support artifacts into raw truth.
