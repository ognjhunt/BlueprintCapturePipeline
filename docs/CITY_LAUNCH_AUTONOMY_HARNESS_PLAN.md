# City Launch Autonomy Harness Plan

Cross-repo paths in this plan use the conventional `$HOME/workspace/<repo>`
layout; resolve them against wherever the sibling repos are checked out in your
environment (see the sibling-checkout convention in `AGENTS.md`).

## Goal

Build a launch harness that can take a city and budget, delegate all repo-safe and service-safe work to agents, and continue without founder intervention until it reaches a real external blocker. The harness must never convert plans, mocks, contract tests, or generated artifacts into launch proof.

The target launch claim is:

- iPhone: externally marketable city beta only after real-device capture, upload, privacy processing, pipeline handoff, hosted output, buyer access, and payout checks are proven for the city.
- Meta glasses: limited pilot or internal video-first path until physical glasses capture, video-to-world, privacy processing, hosted output, and no-native-geometry marketing guardrails are proven.
- Android or other video-only paths: internal-only unless the same proof standard is met.

## Current Hard Stops

1. Fix the iOS compile failure before any city launch run can be considered shippable.
   - Source: `$HOME/workspace/BlueprintCapture/BlueprintCapture/ViewModels/NearbyTargetsViewModel.swift`
   - Current issue: `Task { [weak self] @MainActor in` is invalid Swift syntax.
   - Acceptance: the targeted launch test and a broader app test lane pass on a simulator, then the real-device smoke lane runs.

2. Replace contract-only launch proof with a real city proof artifact.
   - Source gate: `$HOME/workspace/BlueprintCapture/scripts/validate_launch_readiness.py`
   - Current limitation: `contract_only` proof cannot be used for a real launch.
   - Acceptance: `ops/launch-readiness/<city-slug>.launch-proof.json` is generated from live checks and passes the validator without `--contract-only`.

3. Align readiness contracts with truthful launch claims.
   - Source: `src/blueprint_pipeline/alpha_readiness.py`
   - Current issue: Meta glasses and Android can be marked `ready_for_external_alpha` even though launch docs say they remain internal-only for site-faithful claims.
   - Acceptance: tests encode iPhone external beta as the first launch path; glasses and Android remain internal-only unless a separate evidence field proves external site-faithful readiness.

## Harness Architecture

Create a city-launch harness with four durable layers:

1. Planner
   - Input: `city_slug`, budget, target launch window, enabled capture paths.
   - Output: a launch plan with capture targets, capturer onboarding tasks, buyer/demand tasks, proof requirements, and known blockers.
   - Rule: planning can use research and internal heuristics, but activation must use explicit route, artifact, and ledger evidence.

2. Delegator
   - Creates bounded agent work packets with file ownership, expected artifacts, commands, and proof schema.
   - Delegates independent lanes in parallel: iOS, city/backend, pipeline, privacy runners, runtime/WebApp sync, payments/payouts, Meta glasses, ops monitors, and marketing claims.
   - Keeps going while lanes are independent; does not wait on a blocked lane if another repo-safe lane can advance.

3. Proof Collector
   - Normalizes evidence into a single launch proof JSON.
   - Rejects screenshots or narrative-only claims unless the proof schema explicitly allows them.
   - Requires artifact paths, route responses, IDs, timestamps, and source system names.

4. Gatekeeper
   - Runs local tests, live route checks, and proof validation.
   - Produces one of four statuses:
     - `ready_to_market_iphone_city_beta`
     - `ready_for_internal_glasses_pilot`
     - `blocked_external_dependency`
     - `blocked_repo_or_contract_failure`
   - Emits the earliest hard stop first, with exact evidence paths.

## Workstream 1: iOS Compile And Real-Device Launch Lane

Owner scope: `$HOME/workspace/BlueprintCapture`

Tasks:

- Fix `NearbyTargetsViewModel.swift` syntax.
- Add a focused regression test or compile gate for the deinit cleanup path.
- Run the targeted launch tests that previously failed.
- Add a scriptable real-device smoke checklist that records:
  - app build/archive configuration
  - authenticated user
  - city slug
  - selected capture target/job ID
  - upload completion ID
  - capture submission document path
  - raw upload completion marker
  - pipeline handoff marker

Acceptance:

- Simulator tests pass.
- Real-device iPhone capture produces a bundle that the cloud bridge marks as a valid iPhone site-world candidate when ARKit, depth, rights, and site identity are present.
- The launch proof file contains real iPhone evidence, not example data.

## Workstream 2: Site Identity, Topology, Revisit Anchors, And Dense Export

Owner scope:

- `$HOME/workspace/BlueprintCapture`
- `$HOME/workspace/BlueprintCapturePipeline`

Tasks:

- Make site identity mandatory for any marketed city capture path.
- Add app-side fields for:
  - `site_id`
  - `site_slug`
  - `site_name`
  - `city_slug`
  - `capture_job_id`
  - `capture_topology`
  - `capture_mode`
  - `revisit_group_id`
  - `requested_outputs`
  - rights/provenance metadata
- Ensure open capture never falls back to unstable job-derived site IDs for marketable site-world output.
- Ensure dense frame export is generated before `retrieval_index_stage.py` attempts site memory insertion.
- Add negative tests proving captures without `world_model_candidate` or `site_id` are skipped and cannot be marketed as site memory.

Acceptance:

- `retrieval_index_stage.py` appends to site reference memory for qualified iPhone captures.
- The same site can receive multiple revisits without fragmenting memory.
- Glasses captures either carry a complete deferred-geometry site identity or are explicitly marked internal/video-first.

## Workstream 3: Readiness Contract Correction

Owner scope: `$HOME/workspace/BlueprintCapturePipeline`

Tasks:

- Change `alpha_readiness.py` so `external_alpha` is not granted to Meta glasses or Android from contract artifacts alone.
- Introduce explicit status fields:
  - `contract_ready`
  - `internal_pilot_ready`
  - `external_market_ready`
  - `site_faithful_market_ready`
- Require physical-device and live downstream proof before any non-iPhone path can be external-market ready.
- Update tests that currently expect glasses/Android external readiness.
- Update `READINESS_MATRIX.md` and `PAID_MARKETPLACE_BETA_LAUNCH_GATE.md` so docs and code match.

Acceptance:

- Contract tests still pass for orchestration.
- iPhone can pass external beta at contract level only with proper proof wording.
- Meta glasses and Android remain internal-only unless live proof fields are present.

## Workstream 4: Privacy-Safe Provider Proof

Owner scope: `$HOME/workspace/BlueprintCapturePipeline`

Tasks:

- Add a launch proof section for privacy runner inputs and outputs:
  - SAM3 detect result
  - VIP inpaint/depth result
  - DeepPrivacy2 result
  - final privacy-safe walkthrough URI
  - World Labs input URI
  - raw bypass flag
- Fail launch validation if `BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS=true` for production.
- Add a provider preview audit artifact proving World Labs consumed `privacy/final_walkthrough.mov` or a privacy-safe derivative.
- Add a canary test fixture that proves raw video is rejected for production launch.

Acceptance:

- The production launch proof shows only privacy-safe video was submitted to the provider.
- Raw fallback remains available only for internal demos and is visible as a blocker in launch status.

## Workstream 5: World-Model Runtime, WebApp Sync, Buyer Access

Owner scope:

- `$HOME/workspace/BlueprintCapturePipeline`
- `$HOME/workspace/Blueprint-WebApp`

Tasks:

- Verify live environment variables and service URLs for:
  - privacy runners
  - video-to-world
  - World Labs
  - runtime service
  - WebApp sync
  - buyer artifact access
- Add a launch proof section for hosted-session artifacts:
  - package ID
  - runtime URL
  - WebApp listing or catalog ID
  - authenticated buyer access check
  - export package checksum
- Make WebApp sync fail closed if the upstream request/job/bootstrap record is missing.
- Add a delegated WebApp agent lane that proves buyer access after sync.

Acceptance:

- A real processed capture appears in the WebApp/catalog surface for the city.
- A buyer-authenticated route can access the expected artifact.
- Missing upstream records produce a hard blocker, not placeholder launch proof.

## Workstream 6: Payments, Payouts, And Capturer Marketing

Owner scope:

- `$HOME/workspace/Blueprint-WebApp`
- `$HOME/workspace/BlueprintCapture`

Tasks:

- Add live Stripe buyer payment proof fields.
- Add capturer payout or payout-ready ledger proof fields.
- Keep open capture review-gated.
- Keep paid-anywhere claims disabled until city-specific jobs, payout policy, and Stripe state are proven.
- Add marketing-claim guardrails that prevent "paid to capture anywhere" language unless the proof file explicitly allows it.

Acceptance:

- Launch proof contains either live payout proof or a market-safe claim boundary.
- Capturer-facing copy matches the actual city/job/payout state.
- No generated marketing copy can overstate payments, locations, or glasses readiness.

## Workstream 7: Agent Delegation Harness

Owner scope:

- Pipeline scripts/docs in this repo.
- Paperclip/WebApp orchestration surfaces if the launch system runs through Blueprint agents.

Tasks:

- Add a machine-readable work packet schema:
  - `lane_id`
  - `repo`
  - `owner_agent`
  - `inputs`
  - `allowed_paths`
  - `commands`
  - `expected_artifacts`
  - `blocking_conditions`
  - `proof_fields`
- Add a launcher command, for example:
  - `python scripts/run_autonomous_city_launch_harness.py --city-slug <city> --budget-cents <budget> --capture-path iphone --capture-path meta_glasses`
- Add a resume command:
  - `python scripts/resume_autonomous_city_launch_harness.py --run-id <run-id>`
- Add append-only run state under the private external root selected by
  `BLUEPRINT_CITY_LAUNCH_OUTPUT_ROOT` (never under the source checkout):
  - `<evidence-root>/<city-slug>/<run-id>/manifest.json`
  - `<evidence-root>/<city-slug>/<run-id>/work-packets/*.json`
  - `<evidence-root>/<city-slug>/<run-id>/proof.launch-proof.json`
  - `<evidence-root>/<city-slug>/<run-id>/blockers.jsonl`
- Require the `city-launch-harness-run.v2` manifest, exact SHA-256/size inventory,
  private-root access mode, seven-day freshness check, 365-day retention date, and an
  explicit external-disclosure approval check.
- Each delegated lane must write either proof or a blocker. Silent completion is failure.
- Add a synthesizer that merges lane results into the launch proof and calls the validator.

Acceptance:

- A run can proceed across independent lanes without founder input.
- The harness stops only for explicit blockers such as missing credentials, unavailable live service, real-device capture not performed, or failing tests.
- The final output names the first hard blocker, stage reached, and evidence path.

## Workstream 8: Ops Monitors And Recovery

Owner scope:

- `$HOME/workspace/BlueprintCapture`
- `$HOME/workspace/BlueprintCapturePipeline`
- `$HOME/workspace/Blueprint-WebApp`

Tasks:

- Implement or verify monitors for:
  - failed uploads
  - capture submission registration
  - push/device sync
  - bridge pipeline handoff
  - payout exceptions
  - session events queryability
  - cloud logging handoff alerts
- Add recovery actions:
  - retry upload handoff
  - requeue pipeline processing
  - notify agent lane owner
  - produce founder blocker only if an external secret, account, hardware action, or legal approval is required.

Acceptance:

- Every monitor required by `validate_launch_readiness.py` has a real query or alert proof.
- Harness can distinguish transient failure, retryable failure, and human-required blocker.

## Implementation Sequence

Phase 0: Repo safety and first unblock

- Fix the Swift compile error.
- Run targeted iOS tests.
- Freeze current readiness truth in tests so future agents cannot reintroduce glasses overclaiming.

Phase 1: Harness skeleton

- Add run manifest, work packet schema, proof schema, and append-only blocker log.
- Add dry-run mode that creates lane packets and validates that required proof fields are known.

Phase 2: Contract correction

- Update Pipeline readiness logic and tests.
- Update docs to match the corrected statuses.
- Add launch messaging guardrails.

Phase 3: Real iPhone city proof

- Run one real-device iPhone capture in the target city.
- Generate a real launch proof artifact.
- Process through privacy, provider, runtime, and WebApp sync.

Phase 4: Meta glasses pilot proof

- Run physical Meta glasses smoke.
- Prove connection, reservation, upload, and video-first processing.
- Keep market status internal-only unless video-to-world and hosted output proof reaches the same threshold as iPhone.

Phase 5: Payments and buyer proof

- Prove Stripe buyer payment or explicitly block marketing claims.
- Prove capturer payout or payout-ready ledger state.
- Prove buyer artifact access.

Phase 6: Autonomous resume and delegation

- Make the harness resumable.
- Allow agents to pick up blocked work packets after credentials or external proof appears.
- Add a final synthesis command that prints launch status and first blocker.

## Required Commands

Minimum local verification:

```bash
pytest tests/test_alpha_readiness.py tests/test_run_e2e.py tests/test_webapp_sync.py tests/test_world_model_candidate_parity.py tests/test_site_world_packaging.py
```

Capture bridge verification:

```bash
cd $HOME/workspace/BlueprintCapture/cloud/extract-frames
npm test
```

iOS targeted verification:

```bash
cd $HOME/workspace/BlueprintCapture
xcodebuild test \
  -project BlueprintCapture.xcodeproj \
  -scheme BlueprintCapture \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro' \
  -parallel-testing-enabled NO \
  -only-testing:BlueprintCaptureTests/CaptureBundleAndInferenceTests/finalizerAndExportProducePipelineReadyBundle \
  -only-testing:BlueprintCaptureTests/LaunchCityGateTests
```

Real launch proof validation:

```bash
cd $HOME/workspace/BlueprintCapture
python3 scripts/validate_launch_readiness.py \
  --proof ops/launch-readiness/<city-slug>.launch-proof.json \
  --city-slug <city-slug> \
  --lat <latitude> \
  --lng <longitude>
```

## Definition Of Done

The launch harness is done when a single city run can:

1. Create a plan and lane packets from city and budget.
2. Delegate independent work without founder intervention.
3. Produce real evidence or explicit blockers for every required lane.
4. Generate a non-contract-only launch proof artifact.
5. Pass local tests and live route checks.
6. State exactly which capture paths are marketable.
7. Prevent iPhone, Meta glasses, Android, payment, payout, provider, or buyer-access claims that are not backed by proof.

The first marketable state should be `ready_to_market_iphone_city_beta`. Meta glasses should remain `ready_for_internal_glasses_pilot` until physical-device and downstream site-world proof are complete.
