# Pre-Beta Launch Gap Audit — 100-User Beta, Any-Location (Industrial-First)

**Date:** 2026-07-08
**Scope:** `BlueprintCapture` (capture client + cloud), `BlueprintCapturePipeline` (packaging/eval/runtime), `Blueprint-WebApp` (buyer/ops/payments)
**Goal framing:** Launch a beta to ~100 external users where the platform can package **any captured location type**, starting with **industrial sites — warehouses and factories** (where humanoids deploy first).
**Method:** Multi-agent audit across 18 dimensions, each finding adversarially re-verified against the source, then completeness-critiqued. 119 verified gaps. Grounded in current code/docs on branch `claude/pre-beta-launch-audit-1pwvpr`; cross-referenced against the prior `100_BETA_TESTER_LAUNCH_BLOCKER_AUDIT_2026-07-06.md`.

---

## Verdict

**Do not launch to 100 external users yet — but the remaining work is well-scoped, not open-ended.** The platform has a genuinely strong, honest core: capture-truth/provenance discipline is real, proof-boundary labeling is conservative and consistent, and — importantly for the stated goal — the **industrial domain model already exists** (`scene_semantics` supports warehouse/manufacturing/fulfillment/brownfield; `industrial_ontology.py` models forklift lanes, traffic zones, human-interaction zones; the `warehouse_site_knowledge` skill is genuinely detailed).

What is missing is not vision — it is **the last mile that turns that model into a shippable industrial product path, plus the operational, security, payments, and legal spine that a real 100-user external beta requires.** Four hard blockers, a kitchen-fixtured product path, and a cluster of ops/security/legal gaps stand between here and an honest launch.

### Headline numbers

| | Count |
|---|---|
| Total verified gaps | **119** |
| P0 (hard blockers) | **4** |
| P1 (high) | **55** |
| P2 (medium) | **48** |
| P3 (low) | **12** |
| 🌐 Location-type blockers (block the industrial-first goal) | **34** |
| New (missed by the 2026-07-06 audit) | **80** |
| Still-open from 2026-07-06 audit | **37** |
| Repos touched | capture 24 · pipeline 43 · webapp 22 · cross-repo 30 |

---

## The 4 P0 hard blockers

These make a truthful, paid, 100-user beta impossible until fixed.

1. **Buyers cannot receive what they buy.** The pipeline never uploads finished packages to cloud storage, so the WebApp's signed-URL delivery has no `gs://` source. The entire "buy → receive artifacts" loop is broken end-to-end. *(cross-repo — §Post-Training Data Packages)*
2. **Storage security rules are disjoint across repos** and both deploy to the **same Firebase project** (last-writer-wins), with **no parity guard**. Whichever repo deploys last silently defines who can read/write capture data. *(cross-repo — §Security)*
3. **The legal foundation is unsigned.** Operator DPA, subprocessor list, access-audit terms, and legal/EHS consent sign-off exist only as **blank templates**. You cannot lawfully process 100 external users' site data on templates. *(cross-repo — §Legal)*
4. **The consent/authorization model is retail/public-space framed with no industrial legal layer** 🌐 — no plant-manager/EHS authorization, PPE/escort conditions, restricted-zone (LOTO, forklift lane) provenance. This is a **direct blocker for the warehouse/factory-first goal**. *(cross-repo — §Legal)*

---

## The location-generalization story (the founder's emphasis)

The single most important nuance in this audit: **the industrial gap is not "the ontology doesn't exist" — it's "the ontology exists but the product path is kitchen-fixtured."** 34 findings are tagged 🌐. They cluster into a clear pattern:

**What already generalizes (keep and build on):**
- `scene_semantics._SUPPORTED_ENVIRONMENTS` distinguishes `warehouse`, `manufacturing`, `fulfillment`, `industrial_unknown`, `brownfield_site` from `kitchen`/`bedroom`.
- `industrial_ontology.py` structures forklift lanes, racks, totes, pallet zones, dock doors, traffic/hazard/human-interaction zones.
- The `warehouse_site_knowledge` skill encodes real domain truth (aisle-width classes, VNA exclusion zones, dock-plate step hazards, pick-zone humanoid fit).
- Robot/task extensibility exists as data (task-spec + robot-profile JSON), and `Large warehouse` is already a first-class pricing SKU.

**Where it breaks (the industrial last mile):**
- **Capture can't even declare the site type.** The app hardcodes `intended_space_type='industrial_unknown'` with no picker (`VideoCaptureManager.swift:2051`), so the pipeline's finer ontology is only ever reached by downstream inference, and true site type is never recorded as capture truth.
- **Success grounding is kitchen-only.** The sole automated success proxy is a faucet/handle revolute proxy; industrial tasks (tote transfer, palletizing, bin placement, line-side delivery) exist as task families but fall through to manual review — so industrial eval outputs are **structurally lower fidelity** than kitchen ones.
- **The proven path is kitchen.** The only committed truth fixtures are `kitchen_task_min` + a generic sim fixture; every merge gate is anchored on kitchen/indoor fixtures. No warehouse/factory fixture, scene asset, or gate exists — industrial regressions ship uncaught.
- **The domain knowledge is quarantined.** `industrial_ontology.py` is imported by exactly one module (the *optional* qualification trust layer). Capture-guidance, scenario grounding, and eval do not consume the structured hazard model.
- **Capture UX assumes small rooms.** Coverage % hardcodes a 100 m² target (meaningless for a 20,000 m² warehouse); onboarding is consumer/nearby-space framed; the launch-city gate hard-blocks the off-core industrial zones where plants actually are; no thermal/memory/disk safeguards for long industrial walks; no multi-visit stitching for sites too big for one charge.
- **Privacy/consent is retail.** Person-only redaction (no badge/screen/whiteboard/signage), retail-only venue-permission demo, and no worker-consent concept for sites full of identifiable staff.

**Implication:** shipping "industrial-first" honestly means investing the **industrial last mile** (site-type as capture truth → industrial success proxies → a committed warehouse fixture+gate → wiring the ontology into capture/eval → industrial-scale capture UX → industrial legal/consent layer), not building the domain model from scratch. That is a focused program, and it is called out explicitly below.

---

## Top themes beyond location

- **Delivery & buyer value loop is broken** (P0 #1 above; buyer can't download purchased artifacts; entitlements have no artifact URI, no expiration/revocation; hosted-session isolation is coarse).
- **Money path is contract-only, not live** (no live buyer-payment/payout settlement proof; no KYC/background-check decision; no dispute/chargeback handler; no named finance owner; no tax/1099 path).
- **Ops is dark** (the `/ops/*` console is mock data, publicly routed, no backend; no alerting for any core failure class; no incident runbook; no rollback; no beta cohort controls or kill switch).
- **Security has real holes** (P0 #2; client-writable `capture_submissions.status` lets a capturer self-approve; any authenticated user can read/update/delete any `scenes` doc; static shared bearer on intake; no upload size bounds).
- **Cost/scale is unmodeled** (no aggregate spend ceiling; no automated GPU-spend watchdog or pod-side TTL; unconfirmed Lambda teardown; no storage retention/lifecycle; no load/soak/capacity/cost model; concurrency capped at ~10).
- **Release/CI won't catch a bad ship** (slow/GPU lane never gates merge/deploy; Render auto-deploys decoupled from CI so a red build still ships; no versioned release/rollback target).
- **Reliability of ingest** (single-shot uploads with no resume; extractFrames OOMs on large videos; batch runner has no per-request isolation — one poison capture aborts all 100).

---

## What is healthy (do not regress it)

- Capture-truth, provenance, and rights metadata are treated as authoritative and generally flow with fail-closed intent.
- Proof-boundary and degraded-state labeling is conservative and consistent — buyer surfaces (`RequestConsole`, `RunDetail` ProofBoundary) are commendably honest; **no world-model-quality overclaiming was found**.
- The world-model/synthesis lane is correctly scoped as swappable *support*, off the beta critical path — the beta can honestly ship a **sim/review-grade** eval without live world-model claims.
- The industrial domain model (ontology, scene semantics, warehouse knowledge skill) is real and reusable.
- The prior audit's discipline ("do not treat old docs as current proof") is sound and was applied here too.

---

## Recommended launch posture & fix sequencing

**Posture:** Run a **scoped, honest industrial beta** — sim/review-grade Task Evaluation Runs + Post-Training Data Packages on warehouse/factory captures — with claims bounded to what the artifacts support (no live-robot, no physical-readiness, no world-model-fidelity claims). Do **not** market a full any-location marketplace on day one.

The 119 gaps sequence into five gates. Each gate is a hard precondition for the next.

### Gate 0 — Unblock the core loop (P0s) — *~1–2 weeks*
Buyers must be able to receive what they buy, on a secure, lawful footing.
- Ship the **package delivery producer** (pipeline → cloud upload → signed-URL source). *(P0 #1)*
- Consolidate **storage rules** into one source of truth + a **parity guard in CI**. *(P0 #2)*
- Get **DPA / subprocessor / access-audit / EHS** documents actually signed. *(P0 #3)*
- Stand up the **industrial consent/authorization layer** (authorizer, PPE/escort, restricted zones) as enforced capture truth. *(P0 #4 — also the first industrial-last-mile item)*

### Gate 1 — Industrial last mile (the goal) — *~2–4 weeks*
Make "industrial-first" true, not aspirational.
- **Site-type as capture truth** (site picker → `site_type` in raw manifest). *(§Location #1)*
- **Industrial success proxies** (containment/placement/transfer) parallel to the handle proxy. *(§Location #2)*
- **Committed warehouse/factory fixture + merge gate**; wire `industrial_ontology` into capture-guidance/scenario/eval. *(§Location #3, #10, #12)*
- **Industrial-scale capture UX**: real coverage/route progress, thermal/disk safeguards, multi-visit stitching, industrial onboarding + venue-permission flow, launch-city gate recovery. *(§Capture, §Onboarding)*
- **Industrial privacy redaction** (badge/screen/signage) + worker-consent concept. *(§Rights)*

### Gate 2 — Operate a real beta — *~2–3 weeks*
- Real ops backend behind `/ops/*` (auth'd) + support/admin queues.
- Alerting for every core failure class; incident runbook; deploy rollback.
- Beta cohort controls + a single kill switch.
- Ingest resilience: resumable uploads, extractFrames large-video path, per-request isolation/quarantine.

### Gate 3 — Money & scale — *~2–3 weeks*
- Live payment/payout settlement proof; dispute/chargeback handler; KYC/background decision; named finance owner; tax/1099 path.
- Aggregate spend ceiling + automated spend watchdog + pod-side TTL + confirmed teardown.
- Storage retention/lifecycle; load/soak + capacity + cost-per-capture model.

### Gate 4 — Release integrity — *~1 week (parallelizable)*
- Gate merges/deploys on the full lane (incl. the new industrial gate); couple Render deploy to CI; versioned release + rollback target; wire ruff.

> **Dynamic evidence still required (not provable from this clone):** this clone has no `output/` gate artifacts, so re-run and attach: `run_paid_marketplace_launch_gate.py`, the sim-only local/release/deployment gates, WebApp→Pipeline forwarding probe, a real takedown drill, and full WebApp/Capture CI. Several "still-open" items below need a live re-run to confirm current status (e.g. the Stripe CSRF 403 now has a WEB-06 native-client fix in `csrf.ts` but the paid gate must be re-run to confirm green).

---

## Findings (119)

Each finding is tagged `[severity 🌐?]` where 🌐 = location-type blocker, with `repo · effort · status-vs-prior-audit`. Findings marked _(critic)_ came from the completeness-critic pass. Ordered within each section by severity, location-blockers first.

### Location-type generalization (industrial / warehouse / factory readiness)

**1. [P1 🌐] Capture app hardcodes intended_space_type='industrial_unknown' with no site-type picker — the pipeline's site ontology is unreachable from capture, and true site type is never recorded as capture truth**  
`capture` · effort M · new  
- **Evidence:** Confirmed: VideoCaptureManager.swift:2051 and GlassesCaptureManager.swift:1768 are the ONLY two occurrences of intended_space_type in all *.swift, both literal 'industrial_unknown'; grep for spaceType/siteType/SiteType/SpaceType returns zero picker. Pipeline reads descriptor/raw_manifest/metadata site_type (episode_spec.py:203-215) and scene_semantics._SUPPORTED_ENVIRONMENTS distinguishes warehouse/manufacturing/fulfillment/brownfield/kitchen/bedroom (scene_semantics.py:20-29).
- **Impact:** Every capture lands with the same hardcoded space type; capturers cannot declare warehouse vs manufacturing vs fulfillment vs kitchen, so the pipeline's finer site ontology and site-specific task hints (episode_spec._scene_class_task_hints, verified at 217+) are only reachable via downstream inference, and a genuine kitchen capture is mislabeled 'industrial_unknown' in raw provenance. Capture truth cannot express the one dimension the founder emphasizes.
- **Fix:** Add a required site-type field in the capture flow (warehouse/manufacturing/fulfillment/kitchen/other) written into raw manifest as site_type; keep 'unknown' as an explicit fallback, not the only value.

**2. [P1 🌐] Industrial task-success grounding does not exist — eval_ready_task_grounding.py only ships a kitchen faucet/handle success proxy, so industrial tasks (tote transfer, palletizing, bin placement, line-side delivery) cannot get exact/proxy success verification**  
`pipeline` · effort L · new  
- **Evidence:** Confirmed: the ONLY success-state proxy in eval_ready_task_grounding.py is proxy_type 'revolute_sink_handle' (_build_articulated_handle_proxy at 983-1020, gated by _requires_articulated_handle_target on _SINK_TOKENS/_HANDLE_TOKENS at 210-216). grep of proxy_type/containment/placement finds no containment/placement/transfer proxy. Industrial task families move_tote/cart_to_conveyor_transfer/line_side_delivery/place_object_into_bin are first-class in robot_eval_dataset.py:188-235 but their success_criteria are prose only, with no automated proxy.
- **Impact:** Task Evaluation Runs are the primary sellable output. A kitchen 'turn on faucet' gets a concrete revolute proxy; an industrial 'move tote into bin' falls through to state_success_proven=false / manual review, making industrial eval outputs structurally lower fidelity than kitchen ones.
- **Fix:** Add industrial success proxies parallel to the handle proxy: containment (centroid inside bin/tote AABB), placement-at-target, and transfer/line-side zone arrival, keyed off material_handling/pick_place/transfer/delivery families.

**3. [P1 🌐] No industrial simulator scene/scenario catalog or committed truth fixture — the only proven end-to-end sim eval instance is kitchen**  
`pipeline` · effort L · new  
- **Evidence:** Confirmed: tests/fixtures/ contains only kitchen_task_min and sim_only_beta_local_capture — no warehouse/factory fixture. Only kitchen sim catalog exists: lightwheel_kitchen_isaac_scenarios.py, kitchen_task_scaling_preflight.py, isaac_g1_kitchen_parity_job.py; isaac_g1_site_3dgs_realistic_eval.py is generic but ships no bundled industrial asset. CLAUDE.md states the success-claim contract truth tests run against tests/fixtures/kitchen_task_min/.
- **Impact:** Although the autogen path is generic in code, the only sim eval path validated by committed fixtures/tests is kitchen; the industrial (warehouse/factory) primary lane ships unproven by any test fixture or bundled scene, so industrial regressions go uncaught in a 100-user industrial-first beta.
- **Fix:** Add a committed warehouse/factory truth fixture (e.g. warehouse_task_min) and an industrial scene asset for the site-3dgs eval so the industrial path is exercised by the same contract tests as kitchen.

**4. [P1 🌐] Scenario-variation family taxonomy is a single fixed global list, warehouse-flavored and factory-incomplete, applied to every task regardless of location type**  
`pipeline` · effort L · new  
- **Evidence:** robot_eval_dataset.py:277-333 defines exactly 11 SCENARIO_VARIATION_DEFINITIONS (lighting/object_rotation/cart_shifted/blocked_path/human_crossing/forklift_nearby/occlusion/glare/missing_label/wrong_object_nearby/narrow_approach_angle) — logistics/warehouse-leaning, zero factory/manufacturing axes. live_robot_eval_closure.py:3340-3344 defaults required_variation_names to full SCENARIO_VARIATION_NAMES; _scenario_family_task_coverage (904-943) blocks any family missing any required name. No location-type/ontology-scoped profile exists.
- **Impact:** Kitchen/home tasks are forced to satisfy forklift/cart coverage, and factory tasks cannot express conveyor motion, machine-guarding/LOTO, AGV cross-traffic, thermal surfaces, or moving-part-on-line. Scenario coverage is neither faithful to factory sites nor selectable per location type.
- **Fix:** Add location-type/site-ontology-scoped variation profiles (warehouse/factory/home) selecting required_variation_names per task's site category, and add factory hazard variations (conveyor_motion, machine_guarding_state, agv_cross_traffic, thermal_surface, moving_part_on_line).

**5. [P1 🌐] No committed industrial task-eval fixture/truth test; industrial hazard variations are template mocks (agent-inferred-needs-review), not capture-grounded**  
`pipeline` · effort L · new  
- **Evidence:** tests/fixtures/ contains only kitchen_task_min and sim_only_beta_local_capture (a neutral 'sim-only-beta-fixture-site', not industrial). CLAUDE.md: success-claim truth tests run against tests/fixtures/kitchen_task_min. robot_eval_dataset.py:294-331 sets default_status='agent-inferred-needs-review' for blocked_path/human_crossing/forklift_nearby/occlusion/glare/missing_label/wrong_object_nearby/narrow_approach_angle; robot_eval_dataset.py:1604-1618 sets sim_or_cosmos_proof_claim_allowed=False with claim_boundary 'variation_is_mock_or_review_input_until_owner_system_proof_exists'. No warehouse/factory/forklift/industrial string in any fixture.
- **Impact:** The 'supports any location incl. industrial' goal is unproven end-to-end: every committed fixture and truth test is kitchen/neutral. An industrial buyer's forklift-proximity/missing-label/narrow-approach scenarios are generator template mocks needing human review, not derived from their actual warehouse capture, so industrial coverage is synthetic until an owner-system proof exists.
- **Fix:** Add committed warehouse and factory task-eval fixtures (site+task+scenario+scorecard) wired into the success-claim truth-test sweep, and provide a path to ground industrial variations in captured site geometry rather than templates.

**6. [P1 🌐] Pipeline->WebApp ops dashboard summary contract is hardcoded to a home/residential task ontology**  
`cross-repo` · effort L · new  
- **Evidence:** server/utils/pipeline-dashboard.ts sceneDashboardSchema hardcodes a top-level whole_home object (line 6) and categories keyed pick/open_close/navigate (lines 14,37,60) with next_action enum recapture/redesign/defer (lines 27-29). No industrial categories. Schema is a live contract: consumed and validated at server/routes/admin-leads.ts:1396 (sceneDashboardSchema.safeParse). Mirrored in server/types/inbound-request.ts:1007 (whole_home). No warehouse/factory task categories (palletize/wrap, machine tending, dock/conveyor, sortation).
- **Impact:** The real admin/operator dashboard contract cannot represent a warehouse or factory site — a fulfillment or machine-tending capture has no home for its status because the schema assumes whole_home + pick/open_close/navigate. Ops visibility silently degrades for exactly the industrial sites the beta must lead with.
- **Fix:** Generalize the dashboard summary to a site-type-agnostic structure (site_type + arbitrary task/category taxonomy) so warehouse/factory task groups surface in operator dashboards without a home-specific schema.

**7. [P1 🌐] Every merge gate is anchored on kitchen/indoor fixtures — no industrial/warehouse/factory gate exists**  
`pipeline` · effort L · new  
- **Evidence:** CLAUDE.md pins success-claim truth tests to tests/fixtures/kitchen_task_min/. `ls tests/fixtures/` shows only kitchen_task_min and sim_only_beta_local_capture; the latter's site_world_spec.json:6 has site_type "fixture indoor navigation route"; grep for warehouse/factory/industrial in that fixture returns nothing.
- **Impact:** Quality gates only prove the pipeline on kitchen/indoor captures. No CI evidence the platform handles the industrial/warehouse/factory sites the founder prioritizes, so a beta can ship green while the first industrial capture is untested end-to-end.
- **Fix:** Add committed warehouse/factory capture fixtures and run the success-claim truth tests + sim-only gate against at least one industrial site type as a required check.

**8. [P1 🌐] No site scale / dimensional metadata as capture truth or Site-card field**  
`cross-repo` _(critic)_ · effort M · new  
- **Evidence:** The canonical capture manifest dataclass IOSManifest (BlueprintCapturePipeline/src/blueprint_pipeline/ios_manifest.py:13-79) records scene_id, fps, width/height, has_lidar, scale_hint_m_per_unit, intended_space_type — but no site footprint area, ceiling height, aisle width, or floor count. Grep for floor|ceiling|height|area_sq|square_met|dimensions|extent across canonical_site_package.py (724 lines) returns zero matches. Site extent exists nowhere as structured capture truth.
- **Impact:** Every downstream consumer that must scale to industrial size — pricing, capacity/cost modeling, recapture planning, eval scenario density, coverage targets — has no numeric site-extent to key on. This is the data-model root cause behind the hardcoded 100 sq m coverage target: even if the UI were fixed, the pipeline could not represent that a warehouse is 50,000 sq m vs a 15 sq m kitchen.
- **Fix:** Add first-class site-extent fields (approx_floor_area_m2, ceiling_height_m, floor_count, dominant_aisle_width_m) to the capture manifest schema and propagate them into the Site card in canonical_site_package.py; treat them as capture-recorded truth, not inferred.

**9. [P1 🌐] Site-type recognition is brittle substring keyword matching over a tiny closed vocabulary with silent 'unknown' degradation — no canonical enumerated site-type set**  
`pipeline` _(critic)_ · effort M · new  
- **Evidence:** Site type is resolved by ad-hoc substring matching against ~7 hardcoded tokens: episode_spec.py:218-239 (_scene_class_task_hints maps only stockroom/warehouse/grocery/kitchen/factory/lab/hospital) and robot_eval_dataset.py:1074-1126 (_infer_site_type keyword lists). Unmatched text falls to site_type_unrecognized -> 'generic/object-grounded task proposals remain review-only' (episode_spec.py:242-252) or 'unknown_site_type' (robot_eval_dataset.py:1126). No repo defines a canonical enumerated set of supported site types; intended_space_type is a free string defaulting to 'unknown' (ios_manifest.py:49).
- **Impact:** Common industrial descriptors the founder targets — 'distribution center', 'fulfillment center', 'cold storage', 'manufacturing plant', '3PL', 'cross-dock', 'assembly plant' — match none of the token lists, so real industrial captures silently degrade to review-only generic proposals with no operator signal that the site type was unrecognized. 'Support ANY captured location' is contradicted by a hard-coded 7-word vocabulary.
- **Fix:** Define one shared, versioned site-type enumeration (with synonym maps) consumed by capture, episode_spec, and robot_eval_dataset; make unrecognized site types an explicit, surfaced state rather than a silent review-only fallback.

**10. [P2 🌐] Industrial hazard ontology (forklift lanes, shared traffic, barriers, human-interaction zones) lives ONLY in the optional qualification trust layer, not in the eval/capture-guidance product core**  
`pipeline` · effort L · new  
- **Evidence:** Confirmed narrowly: industrial_ontology.py defines forklift_lane/traffic_zone/barrier/human_interaction_zone/floor_hazard with route/task/hazard relevance (9-52) and is imported by exactly one module (qualification.py) plus one test. HOWEVER the impact is overstated: eval lanes DO carry forklift/human hazard scenarios independently — mujoco_scene_scenario_packet.py:696 'forklift_nearby', robot_eval_dataset.py:304 forklift_nearby variation and :2979 human_paths/carts/forklifts/doors/blocked_pathways scenario families, policy_autoresearch.py:297 human/forklift/crossing markers, live_robot_eval_closure.py:230.
- **Impact:** The richest structured hazard model (hazard_relevant entity graph) is confined to an optional trust artifact, but eval scenario grounding is not hazard-blind — it has forklift/human/cart scenario families. Real but partial gap: capture-guidance and scenario grounding do not consume the ontology's structured hazard classification, so hazard coverage is coarse rather than absent.
- **Fix:** Promote structured hazard classification (hazard_relevant entities, human-interaction/traffic zones) into capture-guidance and scenario grounding so hazards inform eval scenarios/recapture regardless of whether qualification runs.

**11. [P2 🌐] scene_placement/target_resolver openable + synonym affordance tables are kitchen/home-biased and are consumed by the generic scene_eval_autogen lane**  
`pipeline` · effort M · new  
- **Evidence:** Confirmed (path is scene_placement/target_resolver.py, not root): _SYNONYM_GROUPS (56-77) dominated by faucet/sink/stove/oven/fridge/microwave/dishwasher/kettle/toilet/shower; _OPENABLE_TARGET_GROUPS (85-95) = {fridge,oven,microwave,dishwasher,cabinet,drawer,door}. scene_eval_autogen.py:72-73 imports _OPENABLE_TARGET_GROUPS/_canonical_group_for_token; used at :325,:339 to classify openable/pickable. Industrial openables (dock/rolling doors beyond bare 'door', cage gates, tote lids, lockers, containers) absent.
- **Impact:** For a warehouse/factory scene, articulated open/close task synthesis and target resolution degrade to generic 'door' or no-openable, so industrial articulation tasks are weaker than kitchen equivalents.
- **Fix:** Extend synonym/openable groups (dock_door, rolling_door, gate, cage, tote_lid, locker, container) or make the affordance table environment-scoped rather than a single kitchen-heavy frozenset.

**12. [P2 🌐] Only kitchen has a built, runnable scene family in the Isaac realistic/parity render+eval lane; no warehouse/factory equivalent**  
`pipeline` · effort L · new  
- **Evidence:** lightwheel_kitchen_isaac_scenarios.py (100 kitchen refs), isaac_g1_kitchen_parity_job.py (122), kitchen_task_scaling_preflight.py (85) are dedicated kitchen modules; `ls src/blueprint_pipeline | grep -iE warehouse|factory|industrial` returns only industrial_ontology.py (no scene/parity module). The generic isaac_g1_site_3dgs_realistic_eval.py IS capture-splat-driven (SPLAT_VISUAL_RENDER_SCHEMA_VERSION, l.45) and generalizes, but the concrete benchmark/parity scene family is kitchen-only.
- **Impact:** The render/eval support lane can only demonstrate a purpose-built scenario family for kitchen; an industrial/warehouse render or parity demo has no built scene set, so any 'render lane works for warehouses' claim is unsupported by code today. Scoped correctly at P2 because this lane is off the beta critical path (world models are swappable support, and the capture-driven splat path does generalize).
- **Fix:** Either (a) keep this benchmark lane explicitly off the beta narrative (honest and cheap, it is not the sellable product) or (b) build one warehouse/factory Isaac scenario module mirroring lightwheel_kitchen. Do not market industrial render/eval until a non-kitchen built scene family exists.

**13. [P2 🌐] Industrial 'support' in the classical-sim lane is an un-runnable asset research catalog, not a runnable scene**  
`pipeline` · effort L · new  
- **Evidence:** mujoco_scene_scenario_packet.py lists warehouse_logistics (l.146), industrial_nuclear (l.179), warehouse_factory_industrial (l.323), hospital (l.248), supermarket_retail (l.305) as scene_type catalog entries, but CLAIM_BOUNDARY sets conversion_performed=False, simulators_run=False, remote_asset_downloads_performed_by_packet_builder=False, external_scene_asset_not_raw_blueprint_capture=True (lines 37-47). DEFAULT_SCENE_ID='aws_robomaker_small_warehouse_world' (l.31) is an external GitHub catalog entry, not a converted asset.
- **Impact:** The apparent industrial breadth in the sim lane is a licensing/planning research document about external third-party assets, not Blueprint captures and not executable scenes. Combined with the kitchen-only Isaac family, there is no runnable non-kitchen scene anywhere in the render/sim lane for the beta.
- **Fix:** Keep this packet labeled as external-asset research (its claim_boundary already does) and do not let it imply runnable industrial coverage in launch messaging. Convert at least one warehouse asset end-to-end before promising any industrial sim demo.

**14. [P2 🌐] No multi-floor / vertical-structure (mezzanine, multi-level racking) representation in the site model**  
`pipeline` _(critic)_ · effort L · new  
- **Evidence:** canonical_site_package.py and industrial_ontology.py contain no concept of floor level, mezzanine, or vertical tiering — industrial_ontology.py:9-24 models only planar entities (aisle, rack, forklift_lane, threshold). Grep for floor|mezzanine|multi.?level|ceiling across the site package returns zero matches. The site geometry model is implicitly single-level 2D.
- **Impact:** Warehouses and factories are routinely multi-level (mezzanines, multi-tier racking, elevated catwalks, dock levels). Without a floor/level dimension, captures of vertically stacked space collapse into one plane, corrupting route/task grounding and making tall-racking tasks unrepresentable — a core industrial deployment case.
- **Fix:** Introduce an optional level/elevation attribute on captured entities and site cards, and a per-level coverage concept, so multi-floor industrial sites can be represented without forcing a single-plane assumption.

**15. [P2 🌐] Buyer-facing marketplace location taxonomy omits factory / manufacturing despite it being the founder's first-target site class**  
`webapp` _(critic)_ · effort S · new  
- **Evidence:** detectLocationType in Blueprint-WebApp/server/utils/marketplaceQueryParser.ts:88-149 enumerates exactly six buyer facets: Kitchens, Grocery / Retail, Warehouses, Labs, Utility Rooms, Home / Assistive. There is no Factory / Manufacturing / Assembly-plant facet; 'assembly' is only a weak alias under Labs (line 127). The marketplace item taxonomy (client/src/types/marketplace-search.ts:3) is scenes|training only.
- **Impact:** A buyer searching 'factory', 'manufacturing', 'assembly line', or 'production plant' matches no location facet, so factory capture data — the founder's stated first deployment target — is effectively undiscoverable and unfilterable on the buyer surface even once such captures exist.
- **Fix:** Add a Factory / Manufacturing location facet (with manufacturing/assembly/production/line-side synonyms) to detectLocationType and the marketplace catalog location taxonomy, aligned with the pipeline site-type enumeration.

**16. [P2 🌐] No structured site environmental / operating-condition metadata (cold storage, floor surface, lighting class, noise, hazards) captured or modeled**  
`cross-repo` _(critic)_ · effort M · new  
- **Evidence:** The capture manifest (ios_manifest.py:13-79) and site package carry no environmental-condition fields; the only environment discrimination is fixed profile keys like 'industrial_unknown' in swap_candidates.py:124-147 keyed on object-name keywords, plus a stray 'freezer' object token (swap_candidates.py:31). Grep for cold.?storage|freezer|temperature|floor_material|lighting_class|noise|humidity across src returns no structured metadata field, only render lighting constants and object tokens.
- **Impact:** Industrial operating conditions (cold-storage/freezer temps, wet or high-friction floors, dust, low light, high noise) materially affect both capture quality guidance and the deployment relevance of the sold eval/data package, yet none are recorded as structured site attributes. Buyers cannot filter for, and the pipeline cannot reason about, the operating envelope of an industrial site.
- **Fix:** Add an optional structured operating-conditions block (lighting_class, floor_surface, thermal_zone, ambient_noise, wet/dry) to the capture manifest and Site card, surfaced to capture guidance and buyer filtering.

**17. [P2] Task-aware detection-prompt augmentation hard-codes kitchen/home affordance expansions with no industrial equivalents**  
`pipeline` · effort S · new  
- **Evidence:** Confirmed: derive_task_aware_detection_prompts adds sink/faucet/tap -> 'faucet handle'/'water stream' (296-298), button/switch/panel spatial expansions (299-302), door/drawer/cabinet -> 'handle' (303-304); no pallet/tote/rack/conveyor expansion. task_targets._SEMANTIC_LABEL_BUCKETS box bucket already includes tote/bin/crate/carton/container (task_targets.py ~59), so detection is not blind.
- **Impact:** Industrial captures get thinner detector-prompt enrichment than kitchen captures; because the base is task-text driven it degrades rather than fails, but object-index recall for industrial affordances is systematically weaker.
- **Fix:** Add industrial prompt expansions (pallet -> 'pallet label'/'stringer', tote -> 'tote handle'/'label', rack -> 'upright'/'beam', conveyor -> 'roller'/'belt edge') mirroring the kitchen ones.

**18. [P2] Clip curation's default static-camera constraint rejects mobile-base capture needed for large industrial sites**  
`pipeline` · effort S · new  
- **Evidence:** clip_curation_stage.py:126-127 sets max_static_camera_travel_m=0.05 with enforce_static_camera_for_robot_pov=True by default, and _evaluate_camera_stability (lines 493-507) FAILS any robot_pov clip whose pose travel exceeds 5 cm; novelty is skipped for robot_pov (lines 528-533). No industrial/mobile preset ships. However the constraint is scoped ONLY to clip_kind=='robot_pov' (explicitly OSCAR world-model conditioning data); walkthrough clips instead bound jitter (max_pose_jitter_m, lines 508-522) and permit meters of motion, so moving capture is not wholesale rejected — only egocentric world-model-conditioning clips are, and the bound is tunable via config.
- **Impact:** Roaming robot-POV conditioning clips from a mobile humanoid are rejected by the default 5 cm bound, so that specific capture class fails closed with no documented industrial preset. Mitigated by the walkthrough path (which accepts motion) and by config tunability, so it degrades a capture class rather than blocking industrial support outright.
- **Fix:** Add a mobile/industrial curation profile (or make the static-camera constraint contingent on a declared clip mobility mode) so roaming robot-POV capture is not silently rejected, and document the industrial preset.

**19. [P3 🌐] Industrial entity ontology is isolated to the qualification trust layer, not wired into scenario/WAM-eval/render lanes**  
`pipeline` · effort L · new  
- **Evidence:** industrial_ontology.py defines rack/tote/pallet_zone/forklift_lane/traffic_zone/workcell (lines 11-37); `grep -rl industrial_ontology --include=*.py` returns only qualification.py (+ a test). HOWEVER scenario_variation_instantiator.py DOES carry industrial scenario semantics independently: forklift_nearby / forklift_actor (lines 259-262) and human_crossing (l.320), sourced from robot_eval_dataset.SCENARIO_VARIATION_DEFINITIONS.
- **Impact:** The specific ontology MODULE is only consumed by the optional qualification layer, but the finding's stronger claim — that industrial location-awareness 'never' reaches scenario generation — is contradicted: forklift/human-crossing variations already exist in the primary scenario-variation lane. Residual gap: static scene-structure entities (rack/tote/pallet_zone/aisle) are not flowed into scene/task target resolution. Downgraded to P3 given the partial mitigation.
- **Fix:** Flow the remaining industrial scene-structure entities (rack/tote/pallet_zone/traffic_zone) into scene/task target resolution so industrial scene semantics are first-class, not only forklift/human-crossing actors plus qualification reports.

**20. [P3] Per-class candidate cap tuning is defined only for residential environments; industrial captures get no tuned caps**  
`pipeline` · effort S · new  
- **Evidence:** Confirmed: task_targets.py:40 _RESIDENTIAL_ENVIRONMENTS={bedroom,kitchen}; :41-47 _RESIDENTIAL_DEFAULT_CLASS_CAPS (door/drawer/cabinet/box); :417-424 _resolve_default_class_caps_for_descriptor returns residential caps only for those, else returns _DEFAULT_CLASS_CAPS which is intentionally EMPTY (35-39, relies on spatial dedup) labeled 'industrial_unknown'.
- **Impact:** Minor and env-overridable (SWAP_PER_CLASS_MAX_COUNTS_JSON). Industrial captures rely solely on spatial dedup with no per-class caps, so a warehouse with hundreds of identical totes could produce a large candidate set. Low severity; the empty default is a deliberate design choice.
- **Fix:** Add sensible industrial default caps (tote/carton/pallet/rack) or document that high-repetition industrial sites require an explicit cap override.


### Capture client — industrial-scale capture quality

**21. [P1 🌐] No thermal / memory / disk monitoring during iPhone ARKit recording; storage only checked at upload**  
`capture` · effort M · new  
- **Evidence:** VideoCaptureManager.swift contains ProcessInfo only at line 404 (isiOSAppOnMac) — grep for `thermalState` returns 0 matches in that file. No didReceiveMemoryWarning / thermalStateDidChangeNotification observers exist anywhere in BlueprintCapture/. Disk space is only validated post-recording at CaptureUploadService.swift:327/1101 hasUsableDiskSpace(). The glasses path DOES handle thermal (GlassesCaptureManager.swift ~865-895) and battery (:898-968). No max cap on the continuous ARSession+LiDAR recording.
- **Impact:** A 20-40 min warehouse/factory walk with ARKit+LiDAR+high-res video is the classic thermal-throttle / memory-pressure / storage-exhaustion scenario. iOS can drop frames or terminate the app, and because free space is only validated at upload, a long recording can silently fill the device and lose the entire capture with no warning — breaking the large-area industrial capture the beta targets.
- **Fix:** Observe ProcessInfo.thermalStateDidChangeNotification and UIApplication.didReceiveMemoryWarning during recording; poll volumeAvailableCapacityForImportantUsage periodically mid-recording; surface a 'device hot / low storage' banner and gracefully finalize the current segment before exhaustion.

**22. [P2 🌐] Live 'coverage %' hardcodes a 100 sq m target — false/meaningless for warehouse-scale sites**  
`capture` · effort S · new  
- **Evidence:** CaptureQualityMonitor.swift:243-248 estimatedCoveragePercent: `let estimatedArea = Double(meshAnchorCount) * 1.5`, `let targetArea = 100.0`, comment 'A typical space is 50-200 sq meters.' Rendered live as `"\(Int(monitor.estimatedCoveragePercent))% coverage"` at CaptureQualityOverlayView.swift:148, shown whenever meshAnchorCount>0 (line 144).
- **Impact:** A warehouse/factory floor is 5,000-50,000+ sq m. The overlay saturates at 100% after a trivial fraction of the site is scanned, giving capturers a false 'done' signal and undermining the multi-pass site-world workflow. A hardcoded small-room assumption bleeding into the primary industrial UX.
- **Fix:** Drive coverage/progress from the SiteWorldSiteScale route plan / critical-zone checkpoints (already modeled — CaptureSessionView.swift:97-100,605-683) rather than a fixed 100 sq m mesh heuristic, or hide the percentage for medium/multi-zone scales and show checkpoint progress instead.

**23. [P2 🌐] Open-capture site identity is ephemeral per app launch — no multi-visit stitching for sites too big for one walk**  
`capture` · effort M · new  
- **Evidence:** CaptureFlowViewModel.swift:77-83 initializes openCaptureSiteId, siteVisitId, captureRouteId to fresh UUID().uuidString at VM init. Site id derivation at :513-523 uses targetId (buyer_request) or reservationId (site_submission) for a stable id, else falls back to the session-random openCaptureSiteId (open_capture). No persistence keyed to placeId/address across relaunches; AbandonedCaptureRecoveryStore is injected (:92) for recovery, not cross-visit linking.
- **Impact:** A warehouse/factory often cannot be covered in one charge or visit. After a force-quit, battery drain, or next-day return, an open capture gets a brand-new random site_id/route_id, so the pipeline cannot associate visit 2 with visit 1. Cross-visit stitching of large non-buyer-directed sites is effectively single-shot.
- **Fix:** Persist site/route identity (UserDefaults or AbandonedCaptureRecoveryStore) keyed to placeId/address so a capturer can resume or link additional visits to the same open-capture site_id.

**24. [P2 🌐] No maximum-duration safeguard or mid-recording checkpointing for long single captures**  
`capture` · effort M · new  
- **Evidence:** grep for maxDuration/maximumRecording/segmentDuration/checkpoint in VideoCaptureManager.swift returns none (elapsed time is only a displayed timer via CaptureQualityMonitor). The ARSession recording path writes one continuous file with no segmentation or periodic flush.
- **Impact:** A single multi-GB continuous file for a long industrial walk is fragile: one interruption near the end (thermal kill, storage full, phone call) risks the whole take with no incremental flush. Combined with the no-thermal/no-disk gap, long warehouse captures have no failure containment.
- **Fix:** Segment long recordings into chunks tied to the pass/checkpoint workflow, flush periodically, and warn when a single pass runs unusually long.

**25. [P3 🌐] No LiDAR depth-range guidance for high ceilings, long aisles, and tall racking**  
`capture` · effort M · new  
- **Evidence:** DeviceCapabilityService.swift exposes hasLiDAR as bool + captureMultiplier=4.0 (line 15, a weighting not a range). VideoCaptureManager.swift:2258-2299 clamps depth to UInt16 mm (values <=0 or non-finite become 0/missing) with no far-plane semantics or ~5 m messaging. No code references iPhone LiDAR's ~5 m range, ceiling height, or aisle length; no capturer-facing prompt about far geometry lacking depth.
- **Impact:** Industrial spaces have 8-12 m ceilings and 30-100 m aisles; depth beyond ~5 m is absent and the capturer is never told to do slower/closer passes on tall racking, so bundles under-represent the vertical/long-range structure a warehouse world-model needs.
- **Fix:** Add explicit capturer guidance that LiDAR depth is valid only to ~5 m and prompt slower/closer passes on high racking; optionally flag captures where a large fraction of frames are depth-sparse using the already-computed missing-depth signal.

**26. [P3 🌐] Default capture tips and onboarding tutorial are framed around small home interiors**  
`capture` · effort S · new  
- **Evidence:** CaptureSessionView.swift:531-536 baseline rotating tips: 'Move slowly and steadily', 'Scan corners and edges', 'Ensure good lighting', 'Keep phone upright', 'Overlap scanned areas'. CaptureTutorialView.swift:166-178 uses tutorial_bg_interior.mp4 documented as a landscape 'house pan'. The industrial site-world guidance (SiteWorldSiteScale, critical zones, route plan) lives in a separate in-session overlay (CaptureSessionView.swift:605-683), not in first-run onboarding.
- **Impact:** First-time industrial capturers are onboarded with room/house framing rather than aisle/dock/multi-zone framing, weakening capture quality and expectation-setting for the target use case.
- **Fix:** Add industrial-scale onboarding variants (aisle walking, dock/handoff checkpoints, keeping tall racking upright in frame) and align baseline tips with the site-world route plan.

**27. [P3 🌐] No proactive low-light detection/warning for dim industrial lighting**  
`capture` · effort S · new  
- **Evidence:** Exposure is sampled/collected (VideoCaptureManager.swift:330-332 exposure state/samples/timer, reset :547, populated :555-559) but never surfaced as a live capture-quality gate. The only low-light signal is indirect via ARKit 'insufficientFeatures' tracking in CaptureQualityMonitor.swift (:263-269). JobsRepository.swift:306 lists warehouse jobs with '21:00-23:00 dim aisle lighting' and '05:00-06:00 mixed stockroom lighting', so dim capture is an expected condition.
- **Impact:** Dim aisles/stockrooms and night shifts (known industrial capture windows) can yield under-exposed footage with no user-facing warning until tracking degrades, reducing usable quality for a documented industrial scenario.
- **Fix:** Use the already-collected exposure samples to raise a live 'too dark — add light or slow down' prompt, mirroring existing steadiness/weak-signal warnings.


### Upload resilience & canonical bundle integrity

**28. [P1] Uploads are single-shot PUTs with no intra-file resume — large captures restart from byte 0 on every network interruption**  
`capture` · effort L · new  
- **Evidence:** CaptureUploadService.swift:1256-1280 sends one PUT with X-Goog-Upload-Command:"upload, finalize" and X-Goog-Upload-Offset:"0" via session.uploadTask(with:fromFile:). On error the path re-calls startResumableUpload (:1316-1339) minting a NEW resumable session URL; there is no X-Goog-Upload-Command:"query" to fetch the committed offset. remoteObjectMatchesLocalTruth (:1077-1098) is a whole-object dedup on already-finalized errors, not partial-transfer resume. validate_upload_resilience.py:67 'resume checksum verification' asserts the metadata sha256/size dedup, NOT intra-file resume.
- **Impact:** A 100-tester beta over flaky cellular/warehouse Wi-Fi cannot reliably deliver multi-GB videos; a drop at 90% re-sends the whole file. waitsForConnectivity (:1225) only pauses/retries the whole task, not a partial body. Testers see perpetual re-uploading and burn mobile data.
- **Fix:** Persist the resumable session URL per file; on resume issue X-Goog-Upload-Command:"query" to get the committed offset, then continue with "upload" (and "upload, finalize" only on the final chunk) in bounded chunks so an interruption costs one chunk.

**29. [P2] Declared bundle hashes are never recomputed/compared server-side (bridge or pipeline) — canonical integrity relies solely on the client**  
`cross-repo` · effort M · new  
- **Evidence:** Contract lists mismatched hash as a hard-fail (CAPTURE_RAW_CONTRACT_V3.md:1122,1211 hash_mismatch). Server-side raw-contract-v3.ts:525-539 only checks hash MANIFEST COVERAGE/PRESENCE (missing_hash_manifest, hash_target_missing, hash_coverage_missing) — it never recomputes or compares any sha256 value against file bytes. capture_bridge.py contains zero occurrences of hash/sha256/verify/checksum. iOS writes per-file sha256 into GCS metadata + hashes.json but no downstream actor re-derives them.
- **Impact:** No defense-in-depth against a buggy/old client, tampering, or storage-side corruption: a bundle whose contents no longer match declared hashes ingests and forwards silently, weakening the 'capture truth + provenance authoritative' doctrine for buyer-facing packages.
- **Fix:** At the bridge and/or pipeline ingest, verify hashes.json artifacts and bundle_sha256 against downloaded bytes (or GCS-stored md5/crc32c) and hard-fail with hash_mismatch, matching the contract.

**30. [P2] Background-upload completion state is in-memory only; app termination mid-upload strands captures in 'uploading' until the next manual app launch**  
`capture` · effort M · unknown  
- **Evidence:** pendingUploads (CaptureUploadService.swift:1216) is an in-memory [Int:PendingUpload] with no persistent task->capture map. AppDelegate.handleEventsForBackgroundURLSession (:53-55) only calls setBackgroundCompletionHandler; it does not rehydrate pending uploads. After termination, didCompleteWithError (:1410-1417) hits `guard let pending else { return }` and drops, so finalizeSuccessfulUpload->ensureSubmissionRecordWritten (:782) never runs and the lifecycle record stays upload_state:"uploading" (:742). Recovery is only via UploadQueueViewModel.restorePending (:212) on the next foreground launch.
- **Impact:** If a background transfer finishes (or the app is killed) while suspended, the object can land in Storage but the submission is never finalized until the tester manually reopens the app; ops dashboards show phantom 'uploading' records. No silent data loss, but degraded reliability and support burden at 100 users.
- **Fix:** Persist a taskIdentifier->captureId map across relaunch, recreate the background URLSession with the fixed identifier in handleEventsForBackgroundURLSession, and rejoin in-flight tasks so completion finalizes without a manual foreground launch.

**31. [P2] Bridge is permissive: manifest-validation failure and a missing manifest are recorded but do not stop frame extraction or downstream forwarding**  
`capture` · effort M · new  
- **Evidence:** index.ts:1328 leaves rawManifest null if manifest.json absent after a 45s wait; validateManifest(null) returns valid:false (:1165-1167). manifestValidation.valid is only written into output metadata (:1625, :1889-1891) — it is never used to abort. The `return` statements at :1265/1271/1280/1324 gate on other conditions and all precede validation (:1347); execution proceeds to file.download (:1387), ffmpeg, and the pipeline handoff regardless of valid:false.
- **Impact:** A malformed or manifest-less bundle (schema drift, old client, corrupted write) is still processed and forwarded with manifestValid:false buried in metadata rather than quarantined. Completion-marker gating covers the happy path, but there is no hard stop for schema-invalid v3 bundles, so bad ingest can pollute Site/Task card generation.
- **Fix:** Treat validateManifest().valid===false (at least for v3 bundles) as a terminal quarantine — skip extraction/forwarding and write a rejection record — rather than a metadata annotation.


### Rights, privacy, consent & provenance

**32. [P1 🌐] Privacy redaction is person-only — no badge/ID, screen, whiteboard, signage, or license-plate redaction for industrial sites**  
`pipeline` · effort L · new  
- **Evidence:** privacy_processing.py:313 and :332 hardcode the SAM3 detection prompt to "person" (grep confirms these are the only two prompt sites); proof_contracts.py privacy_state block (~lines 152-164) sets privacy_state="cleared" only when privacy_status in {no_people_detected, person_removed}. No code path detects/redacts non-face PII. Task-aware non-person prompts (eval_ready_task_grounding.py:256 derive_task_aware_detection_prompts) are consumed only by object_index_stage.py:501 for world-model grounding, never for privacy. CaptureSessionView.swift ~line 284 tells capturers to keep 'screens, paperwork' out of frame (advisory only). CAP-10 doc claims 'faces/PII removed'.
- **Impact:** Warehouse/factory footage routinely contains worker badge/ID numbers and name patches, HMI/control-panel screens with proprietary process data, whiteboards, shipping/manifest labels, and vehicle plates. All pass the person-only privacy gate into buyer-facing Task Evaluation Runs and Post-Training Data Packages unredacted, exposing worker PII and operator trade secrets.
- **Fix:** Make the redaction class-set site-type-aware (add text/screen, badge/ID, vehicle-plate, signage for industrial sites); require those classes to be handled for 'cleared'; align the CAP-10 claim with enforced classes.

**33. [P1 🌐] 'policy_only' consent self-clears the consent-evidence gate with no operator permission document, and the capture app auto-asserts it**  
`cross-repo` · effort M · new  
- **Evidence:** proof_contracts.py ~line 115: consent_evidence_complete = (consent_status == 'policy_only') OR ('documented' AND permission_document_uri) — policy_only clears with zero permission doc. No site_type/private-property gate in proof_contracts (grep for site_type/private returns nothing). CaptureFlowViewModel.swift:504 sets consentStatus = isSpaceReviewMode ? .policyOnly : .unknown, so space-review captures auto-assert policyOnly.
- **Impact:** A tester capturing private warehouse/factory property can satisfy consent-evidence with a self-asserted 'policy_only' and no signed operator permission. Defensible only for genuinely public spaces; on private industrial property it is a document-free legal-exposure bypass.
- **Fix:** Gate 'policy_only' by site classification (public/publicly-accessible only); require permission_document_uri or lawful-basis attestation for industrial/private sites; do not let the client self-assert policy_only for private-property site kinds.

**34. [P1 🌐] Site-operator authorization (VenuePermission) is demo-only UI: never persisted, uploaded, or enforced; area restrictions are advisory text**  
`capture` · effort L · still-open  
- **Evidence:** VenuePermissionView.swift:6 declares `struct VenuePermission: Identifiable` (no Codable); :26-35 hardcodes `static let demo` with grocery-style restrictions ['No employee areas','No cash registers','No restrooms']. CaptureSessionView.swift:21 initializes `venuePermission: VenuePermission? = .demo`. Grep finds no serialization into rights_consent.json / capture bundle; restrictions never reach the pipeline.
- **Impact:** The one UI element modeling signed operator authorization and restricted areas — exactly what industrial sites require — is a mock. Authorizing person/title, validity window, and no-go zones are neither captured into provenance nor enforced before buyer delivery, so provenance cannot prove an industrial capture was authorized.
- **Fix:** Make VenuePermission a real Codable record captured at intake, persist into rights_consent.json/capture bundle, enforce presence/validity in the rights gate for industrial/private sites, and carry declared restricted zones through provenance.

**35. [P1 🌐] Venue-permission provenance is a read-only retail demo with no creation flow — industrial capturers cannot record who authorized capture or site restrictions**  
`capture` · effort L · new  
- **Evidence:** VenuePermissionView.swift:26-37 (only constructor anywhere is `.demo` = 'Fresh Market Grocery'/'Store Manager'/areas 'Sales floor, All aisles'/restrictions 'No cash registers'), :273-298 (noPermissionView has no add/sign action). Grep confirms `VenuePermission(` constructed exactly once (the demo).
- **Impact:** Rights/provenance doctrine requires authorization truth, but a first-time industrial capturer cannot record plant-manager/EHS authorization, PPE/escort conditions, or restricted zones (LOTO, forklift lanes). Only retail vocabulary scaffolding exists.
- **Fix:** Ship a real permission-capture flow (authorizer name/title, signed date/expiry, allowed areas, restrictions, optional PDF/photo upload) with an industrial vocabulary preset; wire the badge and upload bundle to the created record.

**36. [P1] Consent-revocation/takedown is pushed by the pipeline but not consumed by the webapp buyer-delivery surface (open loop)**  
`cross-repo` · effort M · new  
- **Evidence:** consent_takedown.py:600 sync_webapp_consent_revocation() fail-closes and POSTs the revoked verdict to PIPELINE_SYNC_WEBAPP_URL; blocker 'webapp_revocation_sync_not_configured' when unset. But internal-pipeline.ts:1204 /buyer-artifact-access-check mints a signed URL whenever entitlement.access_state === 'provisioned' (:1269) with NO consent/revocation check. Grep across server/ finds no route consuming a webapp consent-revocation signal; the 'revoked' access_state enum exists (accounting.ts:135, robot-agent-contract.ts:1136) but nothing wires the pipeline signal to set it.
- **Impact:** After a worker or operator revokes consent, an already-provisioned buyer can keep minting new signed artifact URLs — the revocation never reaches delivery. For industrial sites (trade-secret + worker-PII), an unhonored revocation is a direct legal/trust failure.
- **Fix:** Add an HMAC-verified webapp inbound route consuming the revocation signal that sets access_state to 'revoked'; have /buyer-artifact-access-check fail closed on it (it already blocks non-'provisioned'); add a test asserting a revoked capture returns buyer_accessible=false.

**37. [P1] Consent revocation is not self-enforcing across the delivery chain: revoked capture != revoked entitlement**  
`cross-repo` · effort L · still-open  
- **Evidence:** Pipeline on revocation only writes instruction artifacts and always reports webapp_takedown_executed=False / hosted_session_takedown_executed=False / webapp_or_hosted_takedown_execution_proven=False (post_training_data_package.py:762-803,819-868). WebApp signed-URL minting gates solely on access_state==='provisioned' (marketplace-entitlements.ts:322-334). The 'revoked' enum exists (accounting.ts:135; robot-agent-contract.ts:1136) but grep across server/ found NO runtime writer flipping access_state to 'revoked' from a takedown, and NO consumer of the pipeline's webapp_rights_privacy_takedown_notice / hosted_session_takedown_request; the only 'takedown' hits in server/+client/src are marketing copy in client/src/pages/Governance.tsx.
- **Impact:** Rights are supposed to be authoritative and continuous, but after a capturer revokes consent the buyer's entitlement stays 'provisioned' and keeps minting fresh signed download URLs until a human intervenes. The 15-min TTL bounds already-minted links, not new ones. Live privacy/legal exposure at 100 users with money involved.
- **Fix:** Build a WebApp ingestion path that consumes the pipeline takedown notice and flips affected entitlements to access_state='revoked' (blocking the provisioned check), run a takedown drill proving revoked capture blocks new signed-URL minting, and surface webapp/hosted takedown-executed as real evidence.

**38. [P2 🌐] No worker/employee consent concept or jurisdiction-specific (two-party/biometric) handling for sites full of identifiable staff**  
`cross-repo` · effort M · still-open  
- **Evidence:** grep across pipeline src/ and webapp server/+client/src for worker/employee/two-party/biometric/GDPR/CCPA/works-council/union consent returns no matches (only unrelated 'union' math). The only consent is site/venue-level (proof_contracts.py rights). CAP-10 signoff (docs/beta-launch-audit-2026-07-03/operator-actions/CAP-10-consent-posture-signoff.md) is unsigned (blank Owner/Date/Decision) and its posture predates industrial-worker analysis.
- **Impact:** Industrial sites are workplaces densely populated with identifiable employees; some jurisdictions require two-party/biometric consent and worker notice. No worker-consent basis is captured and the posture doc is unsigned, leaving the beta legally under-supported for factory/warehouse cohorts.
- **Fix:** Re-scope and obtain CAP-10 legal/EHS sign-off for industrial worker environments before onboarding industrial sites; document lawful basis for worker imagery in provenance; correct the CAP-10 redaction-scope claim.

**39. [P3] Open non-review captures default derived-generation and data-licensing to 'allowed' with consentStatus .unknown**  
`capture` · effort S · new  
- **Evidence:** CaptureFlowViewModel.swift:500-504: defaultCaptureRights sets derivedSceneGenerationAllowed=!isSpaceReviewMode and dataLicensingAllowed=!isSpaceReviewMode (true for non-space-review) while consentStatus = .unknown. Pipeline fails closed on unknown consent (proof_contracts consent_evidence_complete requires documented+doc or policy_only, so unknown -> not cleared). captureRights = reviewSeed?.captureRights ?? defaultCaptureRights (:509), so this default only applies when no seeded rights.
- **Impact:** The capture record asserts data-licensing and derived-generation rights on captures with unknown consent and no operator authorization. Caught downstream today (fails closed), but the mislabeled provenance is a truthfulness smell that would silently grant if a gate regressed.
- **Fix:** Default derivedSceneGenerationAllowed/dataLicensingAllowed to false until consent status is established; only set them from a real operator authorization record.


### Pipeline core robustness & orchestration

**40. [P1] Batch inbox runner has no per-request exception isolation, quarantine, or dead-letter — one poison request aborts all 100 captures**  
`pipeline` · effort ? · new  
- **Evidence:** robot_eval_job_orchestrator.py:11128-11145 loads every inbox request via _read_job_request (which calls read_json_any and raises ValueError on non-object payload, orchestrator.py:833-834) inside a bare for-loop, no try/except. The processing loop 11178-11258 calls resolve_local_capture_context (11184; raises PipelineError on non-scenes layout, local_capture.py:40-53) and build_robot_eval_job (11195; raises ValueError on bad provisioner/simulator 9642-9646 and calls sub-builders). inbox_run_manifest.json is written only after the loop (11298). No quarantine/dead-letter path exists (grep found none).
- **Impact:** One malformed request JSON or one request whose capture_root does not resolve raises an uncaught exception that aborts the entire inbox run; requests after the throw never execute, no processed marker is written for survivors, and no inbox_run_manifest.json is produced. A single bad capture silently blocks a whole 100-capture beta batch.
- **Fix:** Wrap each request's read + resolve + build in try/except; on failure write a quarantine/dead-letter marker under .processed (status=failed, error_type, error) and continue. Persist the run manifest incrementally or in a finally block so partial progress and failed items are always recorded.

**41. [P2] capture_batch_registry aborts the whole registry build if any one capture is malformed**  
`pipeline` · effort ? · new  
- **Evidence:** update_capture_batch_registry loop capture_batch_registry.py:255-284 calls resolve_local_capture_context (256; raises PipelineError on bad layout, local_capture.py:40-53) and _stage_statuses (260) with no per-capture try/except; write_json of the registry runs only after the loop (286).
- **Impact:** The operator's batch status/resume dashboard for 100 captures crashes entirely if a single capture root is not in the scenes/captures layout or has an unreadable artifact — the tool meant to show which captures are blocked becomes unusable when one is bad, and no partial registry is written.
- **Fix:** Wrap each capture in try/except inside the loop and record a per-capture {status:'error', error} entry instead of aborting; write the registry incrementally or in a finally block.

**42. [P2] run_e2e --resume-completed-stages replays cached stage snapshots without validating upstream inputs are unchanged (stale/non-reproducible resume)**  
`pipeline` · effort ? · new  
- **Evidence:** _resume_compatible_run_e2e_stage_ledger (run_e2e.py:132-162) gates resume only on schema_version (142), capture_root (144), provider (146), pipeline_lane (148), and the requested feature-flag set (150) — no hash of capture descriptor/raw media. _run_stage returns the stored snapshot verbatim via _completed_stage_resume_snapshot for any completed stage (508-524) skipping re-execution.
- **Impact:** If a capture's raw media/descriptor changed after a partial run (re-upload, corrected descriptor), a resumed run reuses the old preflight/materialization/capture_pipeline snapshot and produces a result that no longer reflects current capture truth, undermining artifact reproducibility and possibly reporting success for inputs that would now fail.
- **Fix:** Include a content hash of the capture inputs (descriptor + raw manifest) in the resume-compatibility check and invalidate cached snapshots when it differs; or restrict resume to stages provably pure in unchanged inputs.

**43. [P3] Four separate, divergent industrial taxonomies exist across the pipeline with no shared source of truth — drift risk in how the same site is handled by capture-guidance vs eval-autogen vs qualification**  
`pipeline` · effort M · new  
- **Evidence:** Confirmed as distinct non-shared sets: scene_semantics.py _SUPPORTED_ENVIRONMENTS/_PROMPTS_BY_ENV (20-29+, includes 'fulfillment'), industrial_ontology.py _ENTITY_RULES (9-24), scene_eval_autogen.py _ENVIRONMENT_KEYWORDS/_PICKABLE_LABELS/_FIXTURE_LABELS (116-169), robot_eval_dataset.py TASK_ONTOLOGY_DEFINITIONS. Each defines its own environment/entity notions; e.g. scene_semantics has 'fulfillment' which the autogen keyword list and task ontology do not.
- **Impact:** A change to industrial handling must be replicated in ~4 places and can diverge, producing inconsistent site handling across capture guidance, eval autogen, and qualification. Real maintainability/consistency risk but not a launch blocker for a 100-user beta.
- **Fix:** Consolidate to one shared site/entity taxonomy module (extend industrial_ontology) that scene_semantics, scene_eval_autogen, and qualification import.


### Task Evaluation Runs (primary sellable output #1)

**44. [P1] Headline task success_rate for WAM runs is a VLM judgment over GENERATED video, not physics or captured truth**  
`pipeline` · effort M · still-open  
- **Evidence:** wam_generated_video_success_label_gemini.py:190-207 (_task_success_criteria) builds LLM prompt criteria ('visible robot end effector must reach the target', failure modes end_effector_does_not_reach_target etc.) to label sampled frames of generated rollout video; openai variant mirrors it. These labels feed _task_success_summary_from_attempts in robot_eval_execution.py:2905, used at 3113 for task_success/success_rate.
- **Impact:** For precision industrial tasks (pallet placement within tolerance, part insertion) an LLM eyeballing generated frames is weak evidence and can be systematically wrong. Claim boundary refuses public-claim-upgrade, but the beta's headline success_rate provenance (generated media + VLM judge) must be disclosed to buyers or it reads as measured physical success.
- **Fix:** Require the buyer report to surface, per scorecard row, success-label provenance (generated-video VLM vs simulator physics vs recorded trace) and gate any success_rate presentation on disclosing that provenance in customer-facing copy.

**45. [P1] Live simulator execution and live policy execution are unproven by default; honest beta deliverable is a sim/review-grade eval, a P0 only if messaging implies executed policy**  
`pipeline` · effort M · still-open  
- **Evidence:** live_robot_eval_closure.py:347-364 CLAIM_BOUNDARY sets simulator_execution_proven=False, robot_policy_execution_proven=False, public_claim_upgrade_allowed=False; _policy_execution_result_audit (1120-1184) emits policy_execution_proof_flag_without_evidence_refs (1166) and policy_execution_missing_proven_executed_modality (1172); LIVE_EXTERNAL_GATES (185-197) list live_simulator_execution and live_policy_execution.
- **Impact:** Platform is correctly fail-closed, so a scoped sim-only/review-grade Task Evaluation Run is honestly deliverable. It becomes a P0 truthfulness blocker only if buyer-facing copy implies the buyer's policy was executed in a live simulator or on a real robot. The residual missing control is a launch-gate check on marketing/report copy.
- **Fix:** Pin the beta's buyer-facing claim ceiling to the closure's highest_truthful_claim and add a launch-gate check that fails if marketing/report copy asserts live simulator or live policy execution while those gates are blocked.

**46. [P2 🌐] Scorecard metric set has no industrial-assembly success semantics (dimensional/insertion tolerance, force/torque, placement accuracy)**  
`pipeline` · effort M · new  
- **Evidence:** live_robot_eval_closure.py:49-61 SCORECARD_REQUIRED_FIELDS and robot_eval_dataset.py:335-394 SCORING_METRIC_DEFINITIONS enumerate success_rate/cycle_time/intervention_rate/unsafe_proximity/collision_risk/object_drop/wrong_object/timeout/recovery_success/world_model_uncertainty/sim_vs_real_calibration_score — all manipulation/mobile-generic. live_robot_eval_closure.py:63 TASK_CARD_STANDARD_REQUIRED_METRICS = SCORECARD_REQUIRED_FIELDS. No insertion, placement tolerance, torque/force, or dimensional-accuracy metric.
- **Impact:** Factory assembly/insertion/kitting tasks cannot express their actual success condition as a first-class scored metric; success collapses to binary success_rate with no tolerance semantics, weakening an industrial Task Evaluation Run's core claim.
- **Fix:** Extend the metric registry with optional task-category-specific metrics (placement_tolerance, insertion_success, peak_contact_force, dimensional_error) a task card can declare in required_metrics, kept optional so home/kitchen tasks are unaffected.

**47. [P2] No real-world calibration anchors exist for any site, so sim-vs-real / digital-twin fidelity claims are undeliverable (boundary correctly enforced)**  
`pipeline` · effort S · still-open  
- **Evidence:** robot_eval_job_orchestrator.py:9310-9321 marks sim_vs_real_calibration_path sim_only_beta_required=False (deployment-only); live_robot_eval_closure.py:3165/3177 block with sim_vs_real_calibration_report_missing / sim_vs_real_calibration_score_invalid, and 1473-1474 require schema sim_vs_real_calibration_report.v1. Calibration is scoped out of the sim-only beta requirement set (line 9326 filters sim_only_required).
- **Impact:** Correctly scoped out for sim-only, but the eval cannot support any 'how this policy performs on the real robot at your site' claim for industrial or any location. Buyers must not read sim success_rate as predictive of real-world outcome; zero accepted real-world anchors exist.
- **Fix:** Keep calibration deployment-gated; ensure the buyer report explicitly states no sim-vs-real calibration exists and results are not real-world performance predictions, per site, until calibration anchors are captured.


### Post-Training Data Packages & delivery (primary sellable output #2)

**48. [P0] Delivery producer is missing: pipeline never uploads packages to cloud, so the WebApp signed-URL handoff has no gs:// source**  
`cross-repo` · effort L · still-open  
- **Evidence:** arena_package_delivery_local.py docstring (lines 1-10) and manifest (lines 118-123) confirm it is local-filesystem-only: storage_upload_performed=False, signed_urls=[], entitlement_verified=False, gated behind BLUEPRINT_ALLOW_PACKAGE_DELIVERY_UPLOAD. The pipeline's signed_url handling in arena_result_ingest.py:934-949 and live_robot_eval_closure.py:1983-1993 only READS signed_urls back out of a delivery-command manifest it does not produce. post_training_data_package.py writes declarative revocation/delivery/signed-access manifests only. WebApp marketplace-entitlements.ts:70-77 (parseGsUri) and 322-337 require a gs:// artifact_uri/post_training_data_package_uri on the entitlement or published marketplace item, returning artifact_access_not_configured otherwise. No pipeline code populates that URI via a real upload.
- **Impact:** Primary sellable output #2 cannot reach a buyer end-to-end: no proven producer pushes a finished package to GCS and populates entitlement.post_training_data_package_uri. A paying beta buyer hits artifact_access_not_configured / entitlement_not_provisioned.
- **Fix:** Implement and prove a real package upload (pipeline -> GCS gs:// object) and wire the URI into the marketplaceEntitlements/publishedMarketplaceInventory record the WebApp reads; run one clean capture->package->entitlement->signed-URL download.

**49. [P2] LeRobot export action contract is hardcoded to a 7D single-end-effector delta pose — no bimanual/whole-body/mobile-base support**  
`pipeline` · effort L · new  
- **Evidence:** lerobot_episode_export.py:52-57 fixes SC3_ACTION_LAYOUT to delta_position_m[0:3]+delta_rotation_axis_angle[3:6]+gripper[6:7], SC3_ACTION_DIM=7 (line 58), and the module docstring (lines 21-24) confirms any action not parsing as a valid 7D vector -> episode EXCLUDED. DEFAULT_STATE_LAYOUTS (lines 63-68) defines only a 'humanoid' base-pose state. Meanwhile scene_placement/robot_profile.py:135 declares the G1 action_space as 'whole_body_or_arm_hand_chunks' with left+right arm manipulators — a mismatch with the 7D single-EE export contract.
- **Impact:** Bimanual manipulation, whole-body control, and mobile-base locomotion cannot be encoded into the LeRobot/GR00T export — control rows fail the 7D check and episodes are excluded, silently dropping exactly the behaviors industrial humanoid tasks require. Object-agnostic so not a location_type_blocker, but embodiment/task-narrow.
- **Fix:** Drive action dim/layout from the RobotProfile action_space (support >7D bimanual/whole-body plus a base-velocity/locomotion block) instead of a fixed 7D EE constant, and version the export schema when the layout changes.


### World-model / synthesis / render support lane

**50. [P3] cosmos3_wam substrate registry entry hardcodes a specific hosted provider (DeepInfra), leaking provider choice into a provider-neutral contract**  
`pipeline` · effort S · new  
- **Evidence:** wam_eval_substrate.py cosmos3_wam entry hardcodes adapter_id='deepinfra_cosmos3_nano_api' (l.184), api_gate_env='BLUEPRINT_ALLOW_DEEPINFRA_API_CALLS' (l.186), model_id='nvidia/Cosmos3-Nano' (l.183), and command_surface '--provider deepinfra --allow-paid-provider-launch' (l.187-190). The registry advertises substrates_are_replaceable=True (l.263) and provider_kind='replaceable_live_or_owner_adapter' (l.175).
- **Impact:** A specific commercial inference provider (DeepInfra) is baked into the substrate contract rather than staying purely in the swappable adapter/command layer. Minor doctrine drift: swapping the cosmos3 host requires editing the substrate registry, not just supplying a different provider command. Low impact because execution still routes through the env-injected provider_command_env (BLUEPRINT_COSMOS3_WAM_PROVIDER_COMMAND, l.185).
- **Fix:** Move DeepInfra-specific adapter_id/api_gate/command_surface out of the substrate registry into provider-adapter config; keep the registry entry provider-agnostic (backbone + model family only).

**51. [P3] Real WAM/SAM3/depth/pose provider validation remains unproven (prior-audit items still open) but is correctly OFF the beta critical path**  
`pipeline` · effort M · still-open  
- **Evidence:** wam_provider_runtime.py defines LIVE_WAM_PROVIDER_ENV_VAR='BLUEPRINT_ALLOW_LIVE_WAM_PROVIDER' (l.59) and live_provider_gate_blockers(allow_live_provider) (l.144); run_e2e.py defaults run_cosmos_validation=False (l.445) and skips when not set (l.717). cosmos3_wam carries proof_ceiling='model_derived_support_artifact_until_real_validation' (l.193). No output/ artifacts in this clone prove a real provider ran.
- **Impact:** Only a launch blocker IF the beta markets learned-WAM rollouts, generated-world rank fidelity, or provider-backed success labels. For a sim/geometry-grounded beta it is not blocking — the claim boundaries already fail closed. Confirms the platform doctrine that world models are swappable support, not the sellable product.
- **Fix:** Keep learned-WAM out of the beta claim set (code already fails closed). Do not surface any generated-world 'rank fidelity' or 'task success' claim to the 100 users; honor the existing claim_boundary flags in webapp copy.


### Providers, spend guards, GPU orchestration & secrets

**52. [P1] GPU spend guard is a manual, dry-run-by-default tool that is never scheduled or enforced — no automated runaway-cost watchdog at 100-user scale**  
`pipeline` · effort M · still-open  
- **Evidence:** scripts/gpu_spend_guard.py:13 (default dry-run), :687 (--reap opt-in); deploy/systemd only ships control-plane + pubsub-handoff timers — grep for spend_guard/reap in deploy/ returns nothing; run_warm_render_worker.py references gpu_spend_guard only in docstrings; run_external_alpha_launch_gate.py has no spend_guard/teardown gating.
- **Impact:** Live GPU burn is watched only if an operator manually runs the script; no cron/systemd/gate ever reaps orphans or snapshots spend. A forgotten pod bills unchecked until a human notices.
- **Fix:** Add a systemd timer that runs gpu_spend_guard.py --reap --json-report on a short interval, persist snapshots as durable teardown evidence, and make the launch gate require a fresh pre/post-canary spend-guard snapshot.

**53. [P1] Booted orphan pods are never auto-reaped and render pods have no pod-side self-terminating watchdog — unbounded billing if the launching process dies**  
`pipeline` · effort M · new  
- **Evidence:** gpu_spend_guard.py:516 (is_reapable returns False when inst.booted), :659-663 ('booted-but-stalled pods... never auto-reaped'); render teardown is host-process-only (isaac_particlefield_render_job watch_and_collect :420-456,:550-579) and bootstrap docker_start_cmd :159-169 installs NO idle/TTL self-kill. Contrast eval worker: BLUEPRINT_GPU_PROVIDER_EXTERNAL_WATCHDOG_TTL_SECONDS (lambda_provider_adapter.py:461-466).
- **Impact:** If the control-plane/render process crashes/OOMs/restarts after a pod boots, the booted pod keeps rendering+billing forever: in-process watchdog dead, standalone guard refuses to reap booted pods, no pod-side backstop.
- **Fix:** Bake a self-terminating idle/hard-TTL watchdog into the render bootstrap (as the eval worker has), and extend gpu_spend_guard to reap booted pods orphaned past a hard age.

**54. [P1] No platform-wide cumulative spend / GPU concurrency ceiling — spend gate is a per-run manual boolean, so 100 users' runs have no aggregate cap**  
`pipeline` · effort L · still-open  
- **Evidence:** provider_reliability_manifest.build_pre_spend_preflight gates on a single per-run spend_gate_open bool (:120-156); budgets are per-job only (lambda requested_budget_usd :802-807; DO/vast per-launch hourly caps :660-703). Grep for cumulative/aggregate/daily/monthly spend ceiling finds no cross-run ledger.
- **Impact:** At 100 concurrent users each triggering runs, there is no global accounting to halt new paid launches once a total dollar or concurrency budget is crossed — cost-runaway and provider-quota-exhaustion exposure.
- **Fix:** Add a persistent cumulative-spend/active-pod ledger consulted by the pre-spend preflight that fail-closes new launches when a configured daily/total budget or max-concurrent-GPU ceiling is exceeded.

**55. [P1] Customer-eval cross-provider failover runtime is not implemented — eval GPU launches are single-provider (or hard-blocked)**  
`pipeline` · effort L · new  
- **Evidence:** robot_eval_provider_launcher blocks when a request declares provider-race failover unless an explicit serial single-provider override is set (:554-563; constant provider_race_runtime_launcher_not_implemented :29-31); robot_eval_provider_race_launcher is explicitly no-spend (:198-203); orchestrator records boundary 'provider_race_runtime_launcher_not_implemented' (robot_eval_job_orchestrator.py:3789-3792). Live racer race_launch is wired only into RENDER (isaac_g1_kitchen_parity_job.py:2451), not customer eval.
- **Impact:** If the single configured GPU provider is down or capacity-starved mid-beta, customer eval jobs stall; recovery requires a manual serial override rather than automatic failover.
- **Fix:** Wire the existing race_launch/ProviderCircuitBreaker orchestration (already used for render) into the robot-eval launch path so eval jobs race/failover across runpod/vast/lambda.

**56. [P1] Lambda single-adapter path never confirms teardown — termination is fire-and-forget, leaving open billing risk for paid eval canaries**  
`pipeline` · effort M · still-open  
- **Evidence:** lambda_provider_adapter TERMINATE_MODE writes a teardown manifest with status='termination_requested' / continuing_spend_requires_followup_list_instances=True (:1377-1408); never performs the follow-up list-instances to reach provider-API-confirmed terminal status that build_teardown_proof requires (provider_reliability_manifest.py:293-350). Render path DOES close this loop via post-terminate inspect() (isaac_particlefield_render_job.py:557-562).
- **Impact:** Paid Lambda eval/canary runs can leave allocations whose teardown is never verified, so open_billing_risk stays true and billing may continue silently — the 'no teardown proof' gap the prior audit flagged.
- **Fix:** Have the lambda adapter auto-poll list-instances after terminate until a billing-terminal status is confirmed, and emit a provider_api-sourced teardown proof.

**57. [P2] No rotation mechanism for GPU provider API keys (RunPod/Vast/Lambda/DigitalOcean); only the forwarding token can rotate**  
`pipeline` · effort M · still-open  
- **Evidence:** Provider keys read verbatim from ~/.blueprint-secrets with no expiry/rotation/ownership (gpu_spend_guard.py:44,:80-92; gpu_render_providers.py:36-42; lambda_provider_adapter._read_lambda_api_key:249-282; DO token :835-845). Only live_pipeline_forwarding_secret_setup supports rotation, and only for the forwarding token (:117-122).
- **Impact:** A leaked/compromised GPU provider key cannot be rotated with evidence or an owner; no rotation cadence — launch-hygiene gap for a paid multi-provider beta.
- **Fix:** Add a documented rotation runbook + helper (as exists for the forwarding token) covering runpod/vast/lambda/digitalocean keys, with a manifest recording last-rotated timestamps and owner.

**58. [P2] Reap exemption relies on a hard-coded allowlist of 8 warm pod IDs duplicated from the render module — brittle cost-exemption that drifts**  
`pipeline` · effort S · new  
- **Evidence:** gpu_spend_guard.DEFAULT_WARM_CANDIDATE_IDS is a frozenset of 8 literal pod ids (:58-69); is_reapable returns False for any runpod pod in that set regardless of state (:510-511); comment admits it is a hand-copied duplicate of isaac_particlefield_render_job.DEFAULT_WARM_CANDIDATES with no shared source of truth.
- **Impact:** A stuck/dud warm-candidate pod is never reaped (protected billing); if the two lists drift or provider ids change, the guard silently protects the wrong pods.
- **Fix:** Derive warm-worker protection from the live warm_serve_pod.json markers already scanned (find_expected_serve_pod_ids) instead of a static id allowlist, or import a single shared constant.


### Cross-repo integration: WebApp -> Pipeline forwarding & intake

**59. [P1] Runtime forwarding defaults to required=false: WebApp returns 202 "queued_for_pipeline" even when nothing reaches the Pipeline**  
`webapp` · effort ? · still-open  
- **Evidence:** robotEvalJobRequests.ts:1251-1259 `required = params.required ?? truthy(process.env.ROBOT_EVAL_JOB_REQUEST_FORWARD_REQUIRED)` -> false by default, and returns status:"not_configured",performed:false when FORWARD_URL empty. Route robot-eval-job-requests.ts:193-212 only 502s when `pipelineForward.required && !pipelineForward.performed`, else 202 with status "queued_for_pipeline". Preflight defaults require-forwarding=true (audit-robot-eval-forwarding-readiness.ts:541-545), so preflight and runtime disagree. Confirmed the only cross-repo channel is HTTP forward; inbox dir (routes:19-22) is local-only.
- **Impact:** If prod deploys without ROBOT_EVAL_JOB_REQUEST_FORWARD_REQUIRED=true (or URL/token missing), a buyer gets a truthful-looking 202 + durableStore.pipeline_inbox="stored" while the request never enters the Pipeline control plane — a fabricated "queued" operational state for 100 users.
- **Fix:** Fail (5xx) whenever forwarding is not_configured/blocked/failed regardless of the FORWARD_REQUIRED flag in production, or default required=true and gate launch on the forwarding preflight status=ready_for_required_forwarding_with_probe against the real prod endpoint.

**60. [P1] Contract parity gate cannot run — shared BlueprintContracts module is absent; both repos run independent hardcoded fallback copies**  
`cross-repo` · effort ? · still-open  
- **Evidence:** verify-robot-eval-job-request-contract.ts:70-90 loads @blueprint/contracts then sibling ../BlueprintContracts/js/robot-eval-job-request.mjs; confirmed neither node_modules/@blueprint nor sibling BlueprintContracts exists here, and main().catch emits blockers:["shared_contract_load_failed"] exitCode 1 (lines 260-269). Pipeline robot_eval_job_request_contract.py:28-48 imports blueprint_contracts (not installed here — confirmed) and falls back to hardcoded "robot_eval_job_request.v1" unless BLUEPRINT_REQUIRE_SHARED_ROBOT_EVAL_CONTRACT=true (lines 22-45 then raises).
- **Impact:** The single-source-of-truth parity check between WebApp request shape and Pipeline intake shape is non-functional as shipped; schema/version drift between the two hardcoded copies is not caught by CI unless the shared module is vendored and the strict env flag is on.
- **Fix:** Vendor or pin BlueprintContracts into both repos (or provide the sibling in the launch env) and make the parity gate + BLUEPRINT_REQUIRE_SHARED_ROBOT_EVAL_CONTRACT=true a required pre-launch check that fails closed when the shared module is missing.

**61. [P2] Lineage-ID enforcement asymmetry: WebApp validator omits request_id/owner_system that Pipeline intake requires**  
`cross-repo` · effort ? · still-open  
- **Evidence:** Pipeline WEBAPP_UPSTREAM_REQUIRED_FIELDS=(site_submission_id,request_id,buyer_request_id,capture_job_id) (live_pipeline_control_plane.py:48-53); _field_value resolves them across request/source/selection_state/owner_system/site_package (live_pipeline_input_intake.py:166-175, blocker missing_required_webapp_ids at 254-255). request_id is emitted only in owner_system.request_id (robotEvalJobRequests.ts:602-609). validateRobotEvalJobRequest checks site_package fields only (645-668) and never validates owner_system or source.
- **Impact:** A hand-built request lacking owner_system passes WebApp validation but is blocked at Pipeline intake with missing_required_webapp_ids; combined with required=false (finding 1) the buyer still sees a 202. End-to-end lineage is not enforced symmetrically.
- **Fix:** Have validateRobotEvalJobRequest assert owner_system.{request_id,buyer_request_id,site_submission_id,capture_job_id} (and source.selection_state) present and consistent, mirroring the Pipeline's required upstream fields.


### Buyer surfaces & artifact access

**62. [P1] No entitlement/authz enforcement on eval-job submission; entitlement.approved is client-supplied and unverified**  
`webapp` · effort ? · new  
- **Evidence:** Route robot-eval-job-requests.ts:105 POST "/" uses only verifyFirebaseToken and takes req.body verbatim (line 106). validateRobotEvalJobRequest (robotEvalJobRequests.ts:629-807) never inspects any entitlement/rights block — only schema/site_package/policy/proof fields. buildRobotEvalJobRequest:597-601 sets rights_privacy_scope.status="cleared_for_robot_eval" and external_use_allowed purely from client-supplied input.entitlement.approved. Pipeline intake _audit_webapp_request (live_pipeline_input_intake.py:194-287) checks only IDs + capture_root, not entitlement.
- **Impact:** Any authenticated Firebase user can queue and forward a robot-eval job for any site by self-asserting entitlement.approved / rights_privacy_scope=cleared_for_robot_eval, spending downstream pipeline/GPU orchestration budget and asserting rights clearance without a verified purchase. Trust + cost exposure across a 100-user beta.
- **Fix:** Server-side verify a marketplace entitlement for (buyer uid, site_submission_id/site_slug) before writing/forwarding; ignore client entitlement.approved and derive rights_privacy_scope from the verified entitlement.

**63. [P1] Buyer cannot download purchased Task Eval Run / Post-Training Data Package artifacts from the app: entitlement carries no artifact URI and the signed-URL endpoint is never called by the client**  
`webapp` · effort M · still-open  
- **Evidence:** Verified: markBuyerOrderPaidFromCheckout writes a marketplaceEntitlements doc with only sku/title/item_type/license_tier/access_state/granted_at and NO artifact URI (server/utils/accounting.ts:719-735). Client renders only entitlement.access?.url (EntitlementAccessTable.tsx:78-88), sourced from resolveAccessUrl which matches ONLY static content.ts arrays scenes/marketplaceScenes/trainingDatasets/syntheticDatasets (marketplace-entitlements.ts:176-231) and never consults Firestore/publishedMarketplaceInventory. The GET /:entitlementId/artifact-access endpoint that DOES sign gs:// URIs (marketplace-entitlements.ts:284-357, loadPublishedMarketplaceItem:160-176) has ZERO references in client/src (grep confirmed). Pipeline delivery stores raw result_artifacts on robotEvalJobRequests and GET /:jobId/status returns them unsigned (robot-eval-job-requests.ts:84-98,217-252) — boundary-validated but not signed.
- **Impact:** The primary sellable outputs cannot be handed over in-product for anything outside the hardcoded static catalog. A beta buyer who commissions a Task Eval Run or Post-Training Data Package sees only 'Access review' with no working download, or a raw gs:// URI they cannot open. Core-sale fulfillment depends on an out-of-band ops path.
- **Fix:** Persist the signed-able gs:// package URI onto the buyer's marketplaceEntitlements doc (a DIRECT_ARTIFACT_FIELDS key) at pipeline delivery, and have the buyer UI (DataPackages/RunDetail) call GET /api/marketplace/entitlements/:id/artifact-access to mint and present the short-lived signed URL instead of relying on resolveAccessUrl's static-catalog match.

**64. [P2] No expiration / license-term enforcement on entitlements — access is durable-forever until manual revocation, including for hosted-session RENTALS**  
`webapp` · effort M · new  
- **Evidence:** Verified: entitlement provisioning writes granted_at but no expires_at/valid_until (accounting.ts:719-735); determineEntitlementAccessState only returns provisioned|manual_review_required (accounting.ts:410-426). All gates check only access_state === 'provisioned' with no time bound: findProvisionedHostedSessionEntitlement (robot-agent-commerce.ts:392,406) and artifact-access (marketplace-entitlements.ts:311). Notably a 'hosted_session_rental' product type exists (robot-agent-commerce.ts:6,140,185,205) yet the rental entitlement never expires. The 'expired' access_state in the union is only set by markBuyerOrderPaymentFailure for checkout sessions that expired UNPAID (accounting.ts:781-790; stripe-webhooks.ts:133), not for time-boxed licenses. initRenewalTracking only schedules email outreach windows (growth-ops.ts:491-523) and does not gate access.
- **Impact:** Hosted-session rentals and any term-limited/trial/beta grants cannot expire automatically. A 'rental' effectively grants perpetual artifact and hosted-session access until someone manually flips access_state to revoked — a rights/licensing-integrity gap the platform treats as first-class.
- **Fix:** Add expires_at (and license term) at provisioning, especially for hosted_session_rental, and enforce it in findProvisionedHostedSessionEntitlement and artifact-access (treat expired as non-provisioned), surfacing an 'Expired' access_state in buyerAppData.ts.

**65. [P3] Marketplace browse/search item types do not include the primary sellable outputs (Task Eval Runs / Post-Training Data Packages)**  
`webapp` · effort M · new  
- **Evidence:** Verified: marketplace search itemType enum is restricted to ['all','scenes','training'] (marketplace.ts:28) and the retrieval corpus is marketplaceScenes + trainingDatasets only (marketplace.ts:6-7,100-110). Task Evaluation Runs are commissioned via the robot-eval-job-requests flow, not browsable. However 'trainingDatasets' plausibly already serves as the Post-Training Data Package catalog surface, so the real uncovered gap is narrower than stated (primarily eval runs, which are inherently per-buyer/per-robot commissioned work that does not require a browsable SKU catalog).
- **Impact:** Task Eval Runs have no first-class marketplace discovery surface; buyers reach them only via the separate request console. A positioning/discovery nitpick, acceptable for a commissioned model in beta.
- **Fix:** Prominently link the request-console path from marketplace results (and/or add an eval-run item type) so the flagship commissioned output is discoverable alongside catalog scenes/datasets.


### Payments, payouts, KYC & finance ops

**66. [P1] Buyer disputes/chargebacks have no local webhook handler — linked payout is not frozen and order status goes stale**  
`webapp` · effort M · still-open  
- **Evidence:** server/routes/stripe-webhooks.ts:205 lists charge.dispute.created in OPS_RELEVANT_EVENTS, but the event switch (~:263-306) has cases only for checkout.*, charge.refunded, and payout.paid/failed/canceled; a dispute falls through to default: break (~:305). Grep confirms 'dispute' appears exactly once in the file (the OPS set), with no handler. relayToPaperclipOps no-ops unless PAPERCLIP_OPS_STRIPE_WEBHOOK_URL is set (:198-200,209-211).
- **Impact:** On a buyer dispute/chargeback the buyer order is not marked disputed and any capturer payout tied to that capture is not held/clawed back. Since real buyer charges can be live, this is a genuine reconciliation/financial-loss gap; the developers clearly intended to react (event is in the OPS set) but never wired the handler.
- **Fix:** Add a charge.dispute.created/.closed case that marks the buyer order disputed and holds any not-yet-in_transit creator payout linked to that order's capture; do not depend on the optional external relay for money-affecting state.

**67. [P1] No identity/KYC or background-check provider decision — payout-fraud and physical site-access screening unaddressed**  
`cross-repo` · effort L · still-open  
- **Evidence:** Grep of server/, client/src/, scripts/ (excluding tests) found zero identity/KYC/background-check provider (no Persona/Onfido/Stripe Identity/Checkr/Jumio); only Stripe Express onboarding creates accounts (stripeConnectAccounts.ts ~:60-84, type express, country US). Pipeline gate defines identity_kyc_provider_decision and background_check_provider_decision requiring decision_record_uri/document_uri (alpha_readiness.py ~:377-379); no gate evidence artifact exists in this clone.
- **Impact:** No independent platform identity verification beyond bank-level Stripe KYC and no background check at all. Weakens payout-fraud defense (synthetic/duplicate capturers) and leaves a safety/liability exposure when sending external capturers into physical sites.
- **Fix:** blueprint-cto to select and integrate an identity/KYC provider (Stripe Identity fits the stack) and a background-check provider for physical site access, and record the decision artifacts the gate expects before onboarding paid capturers.

**68. [P1] Live buyer-payment and capturer-payout settlement are unproven — only mock/contract readiness exists**  
`cross-repo` · effort M · still-open  
- **Evidence:** Live payout execution defaults OFF via isStripeLivePayoutExecutionEnabled()/BLUEPRINT_LIVE_PAYOUT_EXECUTION_ENABLED (constants/stripe.ts:44-51). Pipeline gate manual checks buyer_payment_settlement, capturer_payout_settlement, stripe_connected_account_live_readiness require live-mode payment/payout ids + webhook/ledger reconciliation and provider_mode=live with empty blocking_requirements (alpha_readiness.py ~:349-373); no gate evidence artifact present in this clone.
- **Impact:** A paid beta would charge real buyers and (once the env flag is flipped) pay real capturers on a path never validated end-to-end in live mode, risking failed settlements/reconciliation errors with real money and untruthful readiness claims.
- **Fix:** Execute a real live-mode buyer purchase and a real capturer payout in staging, capture payment_intent/payout/transfer ids + webhook reconciliation refs, and attach them as gate evidence; keep BLUEPRINT_LIVE_PAYOUT_EXECUTION_ENABLED off until then.

**69. [P1] No named human finance-review owner for payout exceptions**  
`cross-repo` · effort S · still-open  
- **Evidence:** Payout-exception triage always sets requires_human_review=true (payout-exception-triage.ts prompt rules) and workflows.ts writes queue defaulting to 'payout_exception_queue' (~:1520-1530), but no code assigns a human owner/on-call to that queue. Pipeline gate human_finance_review_owner requires finance_owner + review_queue_uri/ref (alpha_readiness.py ~:380-384); no evidence artifact present.
- **Impact:** Failed/canceled payouts, treasury-funding failures, and disputes generate review items with no accountable human to action them, so exceptions can sit unresolved, eroding capturer trust and stranding money.
- **Fix:** Name a finance owner and a monitored review queue with a URI, document the escalation SLA, and record it as gate evidence.

**70. [P2] Payout-exception monitor is env-gated AI triage, not a proven live alerting system**  
`webapp` · effort M · still-open  
- **Evidence:** The payout_exception loop runs only when BLUEPRINT_PAYOUT_TRIAGE_ENABLED is set (opsAutomationScheduler.ts ~:124-131). runPayoutExceptionTriageLoop scans creatorPayouts in {review_required, disbursement_failed} (workflows.ts ~:1460-1464) and writes recommendations but emits no alert/dashboard. Gate payout_exception_monitor_live requires monitor_uri/query_uri/alert_policy_uri/dashboard_uri (alpha_readiness.py ~:374-376); no evidence present.
- **Impact:** There is no live alerting monitor; if the env flag is off or the scheduler unstarted in prod, payout failures accumulate silently.
- **Fix:** Confirm BLUEPRINT_PAYOUT_TRIAGE_ENABLED is on in prod, wire a real alert/dashboard for payout.failed/canceled and treasury-funding failures, and attach the monitor URI to the gate.

**71. [P2] Capturer payouts are approved independently of buyer revenue — treasury-drain / negative-margin risk**  
`webapp` · effort L · new  
- **Evidence:** upsertCreatorPayoutFromPipeline (accounting.ts ~:947-956) sets status 'approved' purely from qualification state + recommended_payout_cents, with no reference to whether the capture was purchased. Buyer orders and creator payouts are separate ledgers (accounting-ledgers.test.ts exists) with no revenue-to-payout linkage in this function.
- **Impact:** The platform can owe/pay capturers for captures that generate little or no buyer revenue, draining treasury with negative unit margins and no automated reconciliation guardrail.
- **Fix:** Introduce an explicit payout-funding policy (bounty budget cap or revenue-linked payout) plus a reconciliation report tying disbursed_amount_cents to realized buyer revenue before enabling live payout at scale.

**72. [P2] Capturer payout path has no US tax-reporting compliance (1099-NEC / W-9 collection / backup withholding)**  
`webapp` _(critic)_ · effort M · new  
- **Evidence:** server/routes/stripe.ts creates Connect onboarding links (accountLinks account_onboarding ~L374) and moves money via transfers.create/payouts.create (~L507/L534/L591) but never configures or tracks tax reporting — grep for 1099/w-9/w9/tax/backup withholding/tos_acceptance/business_type in stripe.ts returns nothing; the only 1099 references in the repo are city-launch market-research playbooks.
- **Impact:** Paying 100 US capturers real money without W-9 capture and 1099-NEC issuance is a finance/legal-compliance exposure the confirmed findings (KYC, disputes, treasury-drain, payout owner) do not cover. Stripe Express can file some 1099s only if explicitly enabled/configured, which is neither done nor verified in-repo.
- **Fix:** Decide the tax-reporting owner (Stripe 1099 product vs. in-house), enforce W-9/tax-info collection during capturer onboarding, gate payouts on completed tax info, and confirm 1099-NEC issuance for the beta cohort.

**73. [P3] iOS Stripe client omits Bearer when Firebase token is nil, yielding a confusing 403 CSRF error on state-changing calls**  
`capture` · effort S · new  
- **Evidence:** StripeConnectService.swift makeRequest sets X-Blueprint-Native-Client:ios unconditionally (:207) but perform() attaches Authorization only if currentFirebaseIdToken() is non-nil (:217-219); on failure it throws generic invalidResponse (:233-240). server/.../csrf.ts requires BOTH a Bearer token AND the native-client header to bypass CSRF (~:49-56); a native request with the header but no Bearer falls through to cookie CSRF and returns 403.
- **Impact:** An expired/unrefreshed session surfaces as a misleading 'Invalid CSRF token' 403 on bank disconnect / payout-schedule changes rather than a re-auth prompt. Not a security hole (fail-closed is correct) but degrades capturer troubleshooting.
- **Fix:** Fail fast client-side when currentFirebaseIdToken() is nil for state-changing requests and surface a re-login prompt instead of sending an unauthenticated request.


### Ops, observability, incident response & cohort controls

**74. [P1] Operator console (/ops/*) is entirely mock data with no backend and is publicly routed without auth**  
`webapp` · effort L · still-open  
- **Evidence:** client/src/app/routes.tsx:332-337 registers /ops, /ops/supply, /ops/evidence, /ops/handoff, /ops/spend as layout:"public", shell:"bare" (admin pages use layout:"protected", routes.tsx:299-307). Page files at client/src/pages/ops/{Queue,EvidenceReview,BuyerHandoff,CaptureSupply,SpendControls}.tsx import from @/components/blueprint/ops/mockData (Queue.tsx:11-15) with no fetch/useQuery/firestore call. mockData.ts:1-8 banner: 'there is NO backend behind these values.'
- **Impact:** The exact triage surfaces a 100-user beta needs — stuck-capture queue, blocked/rejected package triage, evidence-review acceptance, buyer release, spend controls — exist only as fake UI. Operators cannot triage support_triage/payout_exception outputs or SLA breaches, and public/bare routing serves a convincing fake ops console to any anonymous visitor.
- **Fix:** Wire /ops screens to real Firestore/pipeline data (worker_status, capture/package status, sla_tracking, payout exceptions), move behind layout:"protected" + admin/ops role gate, and never render fabricated operational state as live.

**75. [P1] No observability alerting for core beta failure classes (uploads, intake, provider, package, buyer-access, payout, spend)**  
`webapp` · effort M · still-open  
- **Evidence:** server/utils/ops-alerts.ts (102 lines) exports only maybeAlertOnWorkerStatusTransition (line 29) and maybeAlertOnLaunchReadinessTransition (line 72). server/utils/launch-readiness.ts checks only credential/env presence — firebaseAdminReady, stripeReady, redisReady, emailReady, agentRuntimeReady (lines 114-123) and automation-lane flags (77-89) — not operational health. No alert path for capture upload, intake/forwarding, provider run, package-generation, or buyer-access failures.
- **Impact:** During a 100-user beta, stuck uploads, failed package generation, or buyers unable to access purchased artifacts go undetected until a user complains; no alert covers the money/data-loss failure surfaces.
- **Fix:** Add metric emission + threshold alerts for upload success rate, intake/forwarding errors, provider run failures, package failure rate, buyer-access 4xx/5xx, payout exceptions, and spend-vs-budget, fed by live counters rather than the env-config health snapshot.

**76. [P1] No beta-ops incident-response runbook (owner, escalation, rollback, takedown, customer-comms) and deploy has no rollback**  
`cross-repo` · effort M · still-open  
- **Evidence:** BlueprintCapturePipeline/docs/runbooks/ contains only groot-oscar-closed-loop-sealed-image.md. Blueprint-WebApp/docs/*runbook* are all Paperclip agent-automation (paperclip-agent-run-failure/connector-recovery/runtime-session). grep for incident/escalation/on-call/takedown returns only design specs, code-of-conduct, research docs, and the prior audit — no operational runbook. deploy/scripts/deploy.sh contains no rollback/revert/health/smoke step.
- **Impact:** For a paid beta touching real captures and money there is no documented incident owner, escalation ladder, rollback target, data-takedown procedure, or customer-comms template, and deploy.sh cannot roll back a bad deploy.
- **Fix:** Author a beta incident runbook (named owner + escalation, rollback SHA/target, takedown/data-deletion drill, buyer/capturer comms templates, degraded-state copy) and add a health-checked rollback path to deploy.sh.

**77. [P1] No mobile crash/error telemetry on the capture clients (the primary data-collection tool is observability-dark)**  
`capture` _(critic)_ · effort M · new  
- **Evidence:** iOS uses FirebaseAnalytics/FirebaseMessaging/FirebaseStorage etc. (grep of FirebaseCrashlytics/Crashlytics/NSSetUncaughtExceptionHandler/recordError across BlueprintCapture/*.swift returns nothing); Android has no crashlytics/sentry in any build.gradle. SessionEventManager.swift only emits app-local 'analytics' events, not crash reports.
- **Impact:** The whole platform depends on the capture app producing intact bundles under stress (the confirmed thermal/memory/OOM/long-capture findings). With zero crash/ANR/error telemetry on iOS and Android, a build that crashes mid-walk at an industrial site produces no signal to ops; the confirmed observability finding only covers the WebApp server, leaving the highest-risk client blind for 100 field users.
- **Fix:** Add Crashlytics (or Sentry) to iOS and Android with symbolication, wire an uncaught-exception handler and capture-session breadcrumb logging (recording start/stop, thermal state, upload failures), and route crash-rate into the beta alerting plane.

**78. [P2] No beta cohort controls: no invite cap, per-cohort throttle, geo/site scope, or single beta kill switch**  
`webapp` · effort M · still-open  
- **Evidence:** Kill-switch flags are per-automation-lane only (server/utils/launch-readiness.ts:77-89 BLUEPRINT_*_AUTOMATION_ENABLED / SUPPORT_TRIAGE / PAYOUT_TRIAGE). grep across server/config for BETA_/cohort/max-users/invite-cap/geo returns nothing. The only 'beta' construct is the capturer_beta_review waitlist queue (opsAutomationScheduler.ts:96), not a 100-user cohort limiter.
- **Impact:** No way to cap the beta at 100 users, throttle onboarding rate, scope by geo/site type, or instantly disable the beta — the only levers turn off automation lanes, not user access or capture intake.
- **Fix:** Add explicit beta cohort control: allowlist/invite cap, onboarding throttle, geo/site scope, and a single beta kill switch that halts new capture intake and buyer access independent of automation flags.

**79. [P2] SLA watchdog exists but has no upload->package stage, no operator-facing surface, and no customer-facing status semantics**  
`webapp` · effort M · still-open  
- **Evidence:** server/utils/sla-enforcement.ts defines stages scoping(24h)/packaging(48h)/delivery(72h)/review_setup(24h) (lines 9,40-43), created from server/routes/inbound-request.ts:1543. No upload->package stage; createSlaTracker is only imported/used in inbound-request.ts (15,1543) with no GET route surfacing sla_tracking to operators; no customer-facing at_risk/breached mapping.
- **Impact:** Partially addresses prior-audit SLA item but measures request scoping/delivery, not capture-upload->package turnaround; breaches only push to Slack/email, no operator queue shows them, and beta users get no status semantics when a package is delayed, blocked, or review-required.
- **Fix:** Add an upload->package SLA stage, surface sla_tracking in the (to-be-real) /ops queue with at-risk/breach filters, and define customer-facing status copy for each degraded state.

**80. [P2] No transactional lifecycle notifications to buyers/capturers on the money- and data-critical events**  
`cross-repo` _(critic)_ · effort M · new  
- **Evidence:** server/utils/email.ts (SendGrid/SMTP) is wired only to growth/outbound and post-signup flows (callers: gtmSendExecutor, cityLaunchSendExecutor, outbound-reply-durability, launch-readiness, post-signup-actions). Grep for order-receipt/delivery-ready/payout-sent/takedown notification across server/routes/*.ts returns nothing; capture app has push infra but no server-driven upload-rejected/processing-failed notification.
- **Impact:** For a real-money beta, buyers get no receipt or 'your Task Eval Run / Post-Training Data Package is ready' notice, capturers get no payout-sent or capture-rejected notice, and the confirmed consent-revocation/takedown loop has no user-facing notification. Users must poll the app; combined with the confirmed missing download path this erodes trust and generates support load.
- **Fix:** Add a transactional notification lane (reuse email.ts + push) for order confirmation, delivery-ready, payout sent/failed, capture accepted/rejected, and consent-revocation notices, with per-event audit logging.


### Security, authorization & abuse/fraud

**81. [P0] Storage rules are disjoint across repos, both deploy to the same project (last-writer-wins), and there is NO storage-rules parity guard — a WebApp deploy wipes the iOS raw-capture upload grant**  
`cross-repo` · effort M · still-open  
- **Evidence:** Verified: Blueprint-WebApp/firebase.json and BlueprintCapture/firebase.json BOTH set storage.rules="storage.rules" and both .firebaserc default to project blueprint-8c1ca. The two storage.rules are disjoint: WebApp/storage.rules:62-119 defines blueprints/, users/, accounts/, captures/, capture-artifacts/, marketplace-artifacts/, menus/ then a catch-all deny at 123-125 and has NO scenes/.../raw path; BlueprintCapture/storage.rules:23-28 defines ONLY scenes/{sceneId}/captures/{captureId}/raw/{rawPath=**} then catch-all deny at 30-32. Parity guard scripts/check-firestore-rules-parity.sh (WebApp) header (lines 3-13) explicitly scopes itself to firestore.rules only; no storage equivalent exists anywhere (grep found none). validate_storage_rules.py only validates the single Capture file in isolation (lines 41-73), never compares against the WebApp file.
- **Impact:** firebase deploy --only storage from either repo is last-writer-wins for the whole project. WebApp deploying last drops the scenes/.../raw grant, so the canonical iOS raw-capture upload falls through to WebApp's catch-all deny and every capturer upload is rejected — core capture silently breaks for 100 users. Capture deploying last denies all WebApp storage paths. Same blast radius as the guarded firestore XR-01 blocker, but unguarded.
- **Fix:** Make storage.rules a single canonical byte-identical superset across both repos covering BOTH the raw-capture path and the buyer/blueprint/user paths, and add a storage-rules parity check mirroring check-firestore-rules-parity.sh to the launch gate.

**82. [P1] capture_submissions.status is client-writable despite rules comment claiming backend-only — a capturer can self-approve captures and poison referral payouts without operator review**  
`cross-repo` · effort S · new  
- **Evidence:** Verified: firestore.rules capture_submissions update rule (lines 217-221) allows any change whose affectedKeys are within captureSubmissionClientKeys(), and that allowlist INCLUDES 'status' (line 71), 'operational_state' (88), 'lifecycle' (89). Comment at 210-211 asserts status transitions are Admin-SDK-only — the rule contradicts it. onCaptureApproved (cloud/referral-earnings/src/index.ts:323-341) fires on any write transitioning status into approved/paid.
- **Impact:** A capturer can flip status to 'approved'/'paid' on their own submission via the client SDK, bypassing human review and corrupting the approval/operational/lifecycle state downstream trusts. Because a premature client flip fires onCaptureApproved (which stamps referralBonusProcessedAt with skip reason at index.ts:374-378/397-400 when payout_cents is absent), the idempotency guard at line 344 then SKIPS the later legitimate approval — poisoning/denying the referrer's real commission. Direct payout-for-gain is NOT reachable: payout at index.ts:352 requires payout_cents, which is not in the client allowlist, and hasOnlyCaptureSubmissionClientKeys(request.resource.data) (line 219) rejects any client update once a backend field like payout_cents exists on the doc.
- **Fix:** Split create-allowed vs update-allowed key sets; on update forbid client changes to status/operational_state/lifecycle (e.g. require request.resource.data.status == resource.data.status). Keep those transitions Admin-SDK-only, matching the stated invariant.

**83. [P1] Firestore scenes collection lets any authenticated user read, update, or delete ANY scene (broken object-level authorization + supply enumeration)**  
`webapp` · effort M · new  
- **Evidence:** Verified: firestore.rules scenes rule (lines 202-206) is `allow create: if isAuthenticated() && hasValidSceneCreatePayload(); allow read: if isAuthenticated(); allow update, delete: if isAuthenticated();` with no owner/creator predicate. hasValidSceneCreatePayload (lines 54-62) requires id/name/title/status/source/timestamps but no owner field, so no ownership binding exists to enforce. Contrast capture_submissions (213-222), reservations (226-229), sessions (238-240) which all gate on creator_id/userId. target_state (232-235) is likewise read-if-signed-in.
- **Impact:** Any of the 100 signed-in beta users can update or delete another user's scene records and enumerate every scene in the project — a capture-truth/provenance integrity break and supply-scraping vector.
- **Fix:** Add an owner field to scene docs at create and gate update/delete on isOwner(resource.data.ownerId) || isAdmin(); scope read to owner/admin or an explicit shared subset rather than all authenticated users.

**84. [P2] Intake auth is a single static shared bearer with non-constant-time compare and no request signing/nonce**  
`pipeline` · effort ? · new  
- **Evidence:** live_pipeline_intake_service.py:941-960 _require_token compares provided!=expected with plain != (timing side-channel) against one global BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN; no per-caller identity, HMAC body signature, timestamp, or nonce. Contrast the WebApp callback direction which verifies timestamp replay window + timingSafeEqual HMAC signature (pipelineSyncSecurity.ts:53-86).
- **Impact:** A leaked bearer token allows request injection into the control-plane inbox; captured (token+body) can be replayed for any new job_id. Weaker than the callback direction's HMAC guard, though the intake fails closed (503) when the token is unset.
- **Fix:** Use hmac.compare_digest for the token check and add HMAC-signed bodies with a timestamp/nonce replay window (mirror the WebApp buildPipelineSyncSignature scheme) on the intake endpoints.

**85. [P2] Hosted-session isolation collapses to site-world entitlement granularity: a co-entitled buyer can read/step/export another team's session**  
`webapp` · effort S · unknown  
- **Evidence:** Verified: ensureLaunchAccess returns entitled for a NON-owner as long as findProvisionedHostedSessionEntitlement matches the session's site-world ids (hosted-session-access.ts:130-157), and those ids are site/scene/capture/submission level, NOT session level (hosted-session-route-helpers.ts:25-40). skuCandidates are built from those site-world ids only — not buyer-scoped. Protected read/mutate routes call ensureLaunchAccess with the loaded session: GET/reset/step/run-batch/stop/media/render/export (site-world-sessions.ts:2218,2243,2257,2435,2558,2609,2656,2670,2753,2763). The denial message itself states access requires 'the creating robot-team account, admin access, or a matching provisioned entitlement.'
- **Impact:** Two robot teams that both license the same shared marketplace site world are not isolated at session level. Given a session UUID, a co-entitled buyer can read run config/media/exports and even drive/reset/stop another team's session — cross-tenant read AND mutation of proprietary eval data.
- **Fix:** For read/mutate on an existing session require ownership (createdBy.uid === caller) or admin; reserve entitlement-based access for session CREATION. If shared sessions are intended, gate non-owner access behind an explicit per-session share grant.

**86. [P2] Canonical raw-capture storage path has no upload size bound, enabling oversized/abusive uploads that auto-trigger the extract-frames pipeline**  
`capture` · effort S · new  
- **Evidence:** Verified: BlueprintCapture/storage.rules:23-28 grants create on scenes/{sceneId}/captures/{captureId}/raw/ with only isSignedIn() + rawCaptureMetadataMatches and NO boundedUpload() cap, unlike WebApp captures/ which caps 500MB (WebApp/storage.rules:81-83) and defines boundedUpload at 56-58. Uploads to this prefix auto-trigger extractFrames onObjectFinalized (cloud/extract-frames/src/index.ts:1246-1268, memory 2GiB/cpu 2/timeout 540s). validate_storage_rules.py (lines 49-71) has no size-limit requirement in its required_snippets.
- **Impact:** A signed-in capturer can upload arbitrarily large raw objects (bounded only by client resumable-upload limits), each spinning up 2GiB/2CPU frame-extraction compute — unbounded storage/compute cost and a DoS vector with no server-side ceiling.
- **Fix:** Add a boundedUpload() cap to the raw-capture create rule and assert it in validate_storage_rules.py; consider a per-capturer daily upload quota given the auto-processing trigger.

**87. [P2] GPU privacy runners exposed with allUsers run.invoker — cost/DoS amplification surface**  
`pipeline` · effort S · new  
- **Evidence:** deploy/terraform/main.tf:1036-1049 grants roles/run.invoker to member 'allUsers' via for_each over sam3, vip, deepprivacy2 and video_to_world services, each provisioned with nvidia.com/gpu=1 (738,822,911,995) and max_instance_count=max_concurrent_jobs. Access is guarded only by PRIVACY_RUNNER_TOKEN env checked in-app (643-644,753-754,837-838,926).
- **Impact:** Publicly invokable GPU endpoints mean a leaked/guessed token or any app-layer auth gap lets an attacker spin up billed GPU instances up to the max-instance cap — cost amplification and GPU DoS, compounded by the absent aggregate spend cap.
- **Fix:** Remove allUsers invoker; require IAM-authenticated (SA-to-SA) invocation or front with an authenticated gateway; keep the token as defense-in-depth. Add per-service max-instance and spend alerts.


### Scale, capacity, storage & cost

**88. [P1 🌐] extractFrames Cloud Function downloads the entire walkthrough video into a 2GiB memory-backed tmpfs — large industrial videos OOM or time out**  
`capture` · effort L · new  
- **Evidence:** index.ts:1246-1252 configures memory:"2GiB", timeoutSeconds:540, cpu:2. Line 1294 localVideo=join(tmpdir(),...); line 1387 `await file.download({ destination: localVideo })` pulls the full walkthrough.mov with no size guard. The only getMetadata size check (:219-220) is inside fileHasContent (content>0 probe), not a pre-download guard on the video. On gen2 /tmp is in-memory tmpfs counted against the 2GiB allocation; ffmpeg then decodes within the 540s budget.
- **Impact:** The core ingest step fails precisely for the largest, most valuable captures. A multi-GB warehouse/factory walkthrough exceeds 2GiB (video bytes in tmpfs + ffmpeg working set) → OOM, or exceeds 9-min timeout → crash. This breaks Site/Task/Scenario/Eval card generation for the prioritized industrial sites.
- **Fix:** Add a getMetadata size guard before download; route large videos to a higher-memory/longer-timeout Cloud Run job; stream to a real disk-backed volume rather than tmpfs; downscale/segment before full decode; surface an actionable failure with a documented max size.

**89. [P1] No capacity/cost/storage-volume model or bucket retention for large industrial captures at 100 users**  
`cross-repo` · effort L · still-open  
- **Evidence:** grep across BlueprintCapture/*.swift for maxDuration/maxRecording/maxFileSize/durationLimit returns nothing, including RecordingPolicyService/RecordingPolicyAIService — no client size/duration cap and no upstream oversize rejection. No GCS lifecycle/retention JSON found in the capture repo. Prior audit already flagged this: 100_BETA_TESTER_LAUNCH_BLOCKER_AUDIT_2026-07-06.md #66 'No capacity/cost model for 100 testers', #106 (~21GB output/), #107-108 (retention/cost plan, no verified retention policy).
- **Impact:** 100 testers producing large, depth-dense warehouse/factory walkthroughs create unbounded storage and egress cost, and each finalized object fans out to a 2GiB function invocation. Without size caps, retention, or a cohort budget, the beta has open-ended cost/quota exposure.
- **Fix:** Define per-capture size/duration caps and a cohort-level budget; add GCS lifecycle/retention rules for raw and derived artifacts; document capture count x average size x provider fan-out before launch.

**90. [P1] No aggregate/fleet spend budget ceiling — GPU cost guardrails are strictly per-job**  
`pipeline` · effort L · still-open  
- **Evidence:** robot_eval_job_orchestrator.py:2264 _provider_prelaunch_spend_guard validates only a single job (requested_budget_usd at 2284-2288, blocker at 2323; max_active_workers==1 at 2328-2329); _gpu_cost_control_ledger at 3977 is also per-job. Repo-wide grep for daily_spend/rolling_spend/cohort_budget/fleet_budget/kill_switch returns nothing. deploy/terraform has NO google_billing/budget resource; GPU services (main.tf:728,812,901,985) each scale to max_concurrent_jobs with nvidia.com/gpu=1 and no budget binding.
- **Impact:** 100 testers each triggering privacy + robot-eval GPU jobs produce unbounded aggregate GPU/provider spend with no platform kill-switch. The founder cannot bound beta burn; large industrial captures amplify per-run cost.
- **Fix:** Add a platform-level rolling spend ledger (daily + cohort cap) checked before any external-provider launch, with a hard stop and remaining-budget surfacing; bind a max concurrent GPU count and per-cohort budget to terraform (and a GCP billing budget resource).

**91. [P1] No storage lifecycle/retention on the primary capture bucket — unbounded storage cost**  
`pipeline` · effort M · still-open  
- **Evidence:** deploy/terraform/main.tf: the ONLY google_storage_bucket resource is function_source (1056), and the ONLY lifecycle_rule (Delete age 30) is on it (1062-1069). The primary var.storage_bucket (line 55) is referenced by every service (589,697,790,879,963,1100) but is unmanaged/pre-existing with no lifecycle/retention. arena_result_ingest.py:1425-1430 retention is a status='draft_review_required', requires_contract_confirmation=True metadata stub, not enforced deletion. No pruning script found.
- **Impact:** Raw captures, processed outputs, robot_eval_jobs and hosted artifacts on the primary bucket grow without bound or deletion plan; 100 users capturing large industrial sites (warehouses/factories >> kitchens) drives unbounded storage cost and a right-to-deletion gap since raw media is never aged out.
- **Fix:** Add GCS lifecycle rules on the primary bucket (raw -> nearline/coldline then delete; eval/hosted artifacts age-out tied to retention contract) and implement the enforced deletion the arena stub only describes. Document a per-data-class retention policy.

**92. [P1] No load/soak test, capacity model, or cost-per-capture model in any repo**  
`cross-repo` · effort L · still-open  
- **Evidence:** grep -li across BlueprintCapturePipeline (scripts/tests), BlueprintCapture/scripts and Blueprint-WebApp/scripts for k6/artillery/locust/'soak test'/'load test'/'capacity model'/'cost model'/'cost-per-capture' returned zero matching harness/doc files (node_modules excluded). No aggregate spend or capacity artifact found to substantiate throughput at target concurrency.
- **Impact:** No evidence the system sustains 100 concurrent uploaders/pipeline triggers, no captures/user, media-GB, provider-runs-per-capture, GPU-second or storage/egress projection. Launch capacity and burn are unmodeled guesses.
- **Fix:** Produce a capacity+cost model (captures/user, avg media GB, privacy+eval runs/capture, GPU-seconds, storage/egress GB, $ per capture and per 100-user month) and run a scoped load/soak test against the intake path at target concurrency.

**93. [P1] No backup / disaster-recovery / durability strategy for authoritative capture truth (Firestore + storage buckets)**  
`cross-repo` _(critic)_ · effort M · new  
- **Evidence:** Blueprint-WebApp/firebase.json declares only firestore rules/indexes, storage rules, and functions — no scheduled Firestore backup/PITR. No object-versioning/soft-delete/retentionPolicy config in storage.rules, cors.json, scripts/ or config/; pipeline docs/scripts have no backup/restore/versioning references. No restore drill exists in any repo.
- **Impact:** Capture truth is the platform's authoritative, non-regenerable asset (doctrine). An accidental deletion, ransomware, bad migration, or the confirmed last-writer-wins storage-rules clobber has no recovery path — data loss is permanent. This is distinct from the confirmed retention/lifecycle findings, which are about capping cost, not durability/restore.
- **Fix:** Enable Firestore scheduled backups/PITR, turn on bucket object versioning + soft-delete + a delete-protection/lifecycle policy on the raw-capture and delivered-package buckets, document RPO/RTO, and run one restore drill before beta.

**94. [P2] Single control-plane capture_root will block a multi-site beta unless per-site override JSON is configured on both sides**  
`cross-repo` · effort ? · new  
- **Evidence:** Pipeline intake requires request capture_root == control-plane capture_root unless allow_request_capture_root (INTAKE_ALLOW_PER_REQUEST_CAPTURE_ROOT env, default off — live_pipeline_input_intake.py:252-260; service reads env at 1018-1020), else blocker request_capture_root_does_not_match_control_plane. WebApp rewrites per-site via FORWARD_CAPTURE_ROOT_BY_SITE_JSON (robotEvalJobRequests.ts:1102-1157). The committed preflight artifact output/pipeline/robot_eval_job_requests/forwarding_preflight.json:27-35,56-57 shows capture_root_by_site site_count:0 and warning capture_root_override_not_configured.
- **Impact:** With one control-plane capture_root and no per-site map, only one site's requests pass; other sites are rejected — silently masked by the required=false 202. For 100 users spread across many industrial/warehouse sites this blocks most traffic.
- **Fix:** Before launch, configure the by-site capture_root map on both WebApp (FORWARD_CAPTURE_ROOT_BY_SITE_JSON) and Pipeline (BLUEPRINT_LIVE_PIPELINE_CAPTURE_ROOT_BY_SITE_JSON) or enable per-request capture root with existence checks, and add a preflight assertion that every beta site slug is covered.

**95. [P2] Pipeline concurrency hard-capped at ~10 with 4h job timeout; 100-user tail latency unvalidated**  
`pipeline` · effort M · new  
- **Evidence:** deploy/terraform/main.tf: var.max_concurrent_jobs default 10 (91-95); Cloud Tasks max_concurrent_dispatches=var.max_concurrent_jobs (511), max_dispatches_per_second=10 (510); Cloud Run job parallelism=1, task_count=1 (555-556), timeout=${pipeline_job_timeout_seconds}s with default 14400 i.e. 4h (97-100,561); privacy GPU max_instance_count=var.max_concurrent_jobs (728,812,901,985); queue-depth alert only fires when depth>100 for 600s (1294-1324).
- **Impact:** 100 testers funnel through ~10 concurrent slots; each job can run up to 4h and large industrial captures push toward that ceiling, so backlog/per-user latency can grow before the >100-depth alert even trips. Backpressure exists but cap-vs-cohort sizing and worst-case wait are unvalidated.
- **Fix:** Size max_concurrent_jobs and privacy max_instance_count to the modeled 100-user arrival rate, lower/segment the queue-depth alert threshold, add a per-user in-flight limit, and validate with the load test.

**96. [P3] captures Firestore index orders on monotonic createdAt — sequential-key hotspot at scale**  
`pipeline` · effort S · new  
- **Evidence:** deploy/terraform/main.tf:1233 and 1248 both define index fields with field_path='createdAt'. Monotonic timestamp keys concentrate writes on a single index range (documented Firestore hotspot).
- **Impact:** Negligible at 100-user beta scale (the finding itself concedes 'minor'); could matter only well beyond beta as capture/status-transition volume grows.
- **Fix:** If write hotspotting appears at higher scale, shard the hot index with a random/hashed prefix or distribute keys; monitor Firestore write latency during the load test.


### Quality gates, CI & release/deploy

**97. [P1] Slow/integration/GPU lane never gates a merge or deploy**  
`pipeline` · effort M · still-open  
- **Evidence:** pyproject.toml:306-314 addopts pins `-m "not slow and not gpu"`; .github/workflows/ci.yml final step is `uv run pytest -q` (fast lane) on push/PR. full-test-lane.yml triggers only on `schedule: cron 17 8 * * 1` and workflow_dispatch — never on pull_request/push.
- **Impact:** The slow/gpu/integration tests (subprocess/Isaac/render/module-entrypoint/provider paths) run at most weekly and never block a PR or deploy, so a merge can green the fast lane while breaking heavy runtime paths beta users exercise.
- **Fix:** Run scripts/pytest_full.sh (or a representative slow subset) as a required PR check, or gate deploy on a recent successful full-lane run; treat weekly cron as supplementary.

**98. [P1] Render autoDeploy is decoupled from CI — a red build still deploys to production**  
`webapp` · effort M · new  
- **Evidence:** render.yaml sets `autoDeploy: true` with buildCommand `npm ci && npm run build` and no CI-wait/buildFilter. WebApp .github/workflows/ci.yml runs check/test/e2e/build on Actions independently; nothing blocks the Render deploy on Actions outcome.
- **Impact:** A push that fails typecheck/tests/e2e/build on Actions is still auto-deployed by Render to the live buyer/ops surface (auth, entitlements, Stripe, forwarding), so CI failures do not protect production during the beta.
- **Fix:** Disable autoDeploy and deploy from a CI job gated on check/test/e2e/build passing (deploy-on-green), or use a Render deploy-hook gated on required Actions checks.

**99. [P1] No versioned release artifact, deploy SHA/tag, or rollback target**  
`cross-repo` · effort M · still-open  
- **Evidence:** deploy/scripts/deploy.sh:33 `IMAGE_TAG="${IMAGE_TAG:-latest}"`; lines 179-183/226-230/266-270 tag every image (blueprint-pipeline, sam3/vip/deepprivacy2, video-to-world) `:${IMAGE_TAG}` with no git-SHA and no rollback function (no `rollback`/`git rev-parse`/`digest` matches). pyproject.toml:3 version static `2.0.0`. render.yaml has no pinned rollback.
- **Impact:** Production runs a mutable `:latest` tag: cannot prove which build is live, cannot pin a known-good release, and has no immutable rollback target if a beta deploy regresses.
- **Fix:** Tag images/releases with git SHA (+semver), pin Cloud Run to the immutable digest, record the deployed SHA as an artifact, and document a one-command rollback to the prior digest.

**100. [P1] No client version enforcement / force-update / remote kill-switch / maintenance mode for the capture app or its bundle contract**  
`cross-repo` _(critic)_ · effort M · new  
- **Evidence:** Capture bundle is versioned (BlueprintCapture/Services/CaptureRawContractV3Validator.swift) but grep for minVersion/force_update/remoteConfig/killSwitch/maintenance/serviceStatus across BlueprintCapture/Services and across Blueprint-WebApp server/ + client/src returns nothing; intake auth is a static bearer with no client-version field. Server never rejects or upgrades a stale client.
- **Impact:** For a 100-user native-app beta with an evolving raw-capture contract, there is no way to force outdated builds to update, block a build shipping a data-corrupting bug, or put the platform into maintenance. A single bad TestFlight/Play build silently poisons canonical bundles with no remote off-switch — the confirmed 'contract parity' finding covers repo-to-repo drift, not client-version rejection.
- **Fix:** Serve a min-supported-app-version + kill-switch/maintenance flag (Firebase Remote Config or a /config endpoint), have the capture app check it at launch and refuse capture/upload when below minimum, and have intake reject bundles from unsupported client versions.

**101. [P2] Ruff is not wired into CI in any repo**  
`cross-repo` · effort S · still-open  
- **Evidence:** grep -rin 'ruff' across BlueprintCapturePipeline/.github, BlueprintCapture/.github, Blueprint-WebApp/.github returns nothing; ruff config lives only in pyproject.toml (per-file E402 ignores at lines 289-291). No workflow invokes `ruff check`.
- **Impact:** Lint (and the prior audit's known E402 red state) can never fail CI; import-order/style regressions ship silently. Quality-only concern, not a runtime blocker — corrected P1->P2.
- **Fix:** Add `ruff check` (and `ruff format --check`) as a required Pipeline CI step; add equivalent lint gates (eslint) to WebApp/Capture.

**102. [P2] Documented release gate (alpha:check/preflight, smoke:launch, paid marketplace gate) is not enforced by CI**  
`webapp` · effort M · new  
- **Evidence:** DEPLOYMENT.md:14-29,442-443 prescribes `npm run alpha:check` as the release gate plus alpha:preflight/smoke:launch; grep of Blueprint-WebApp/.github for those + marketplace returns nothing. grep of BlueprintCapturePipeline/.github for 'marketplace' also returns nothing (paid_marketplace_launch_gate not invoked).
- **Impact:** Launch/release gates are doc conventions dependent on a human running them; nothing in automation blocks a deploy when the release gate or the paid marketplace gate is red.
- **Fix:** Wire alpha:check/alpha:preflight and the paid marketplace launch gate into a required pre-deploy CI job so a deploy cannot proceed while they are red.

**103. [P2] WebApp coverage thresholds are trivially low for a money/entitlements surface**  
`webapp` · effort S · new  
- **Evidence:** vitest.config.ts:56-60 thresholds lines 25.5, functions 35, statements 25.5, branches 50. CI runs `npm run test:coverage` enforcing only these floors.
- **Impact:** ~74% of lines can be uncovered while CI stays green on the buyer/ops/licensing/Stripe/forwarding surface, so the gate gives little regression protection.
- **Fix:** Ratchet thresholds up (especially server/ payment/entitlement/forwarding) and set per-directory floors on critical paths rather than one low global floor.

**104. [P2] Cross-repo sim-only gate is path-filtered and validated against a moving WebApp main**  
`pipeline` · effort M · new  
- **Evidence:** sim-only-local-gate.yml pull_request/push `paths:` allow-lists only the workflow, pyproject, uv.lock, run_sim_only_beta_local_gate.py, src/blueprint_pipeline/**, the fixture, and its test; WEBAPP_REF defaults to `main` (moving HEAD); upload uses if-no-files-found: error.
- **Impact:** Changes outside the allow-listed paths skip the only per-PR cross-repo forwarding gate, and validating against WebApp `main` makes the pinned proof non-reproducible over time.
- **Fix:** Broaden the path filter (or run on all PRs), pin WebApp ref to a known SHA for reproducible proof, and re-run on WebApp changes via cross-repo trigger.

**105. [P2] Capture has no automated release/deploy gate; Android lint is non-blocking**  
`capture` · effort M · new  
- **Evidence:** .github/workflows/ci.yml runs cloud build+test, Swift build/test-without-building, Android `./gradlew test`, and `./gradlew lint --no-daemon --parallel || true` with `continue-on-error: true`. No TestFlight/archive/export step and no gated firebase deploy for cloud/extract-frames; archive_external_alpha.sh / android_alpha_readiness.sh are manual, absent from CI.
- **Impact:** Android lint failures never block, and there is no automated reproducible release/deploy artifact for the iOS/Android/cloud capture apps that beta capturers depend on — release readiness rests on manual script runs.
- **Fix:** Make Android lint blocking (or baseline-triage), and add a gated release job (TestFlight/Play internal track, gated firebase deploy for extract-frames) running the alpha-readiness validators as required checks.


### Legal, compliance & data lifecycle

**106. [P0 🌐] Consent/authorization model is retail/public-space framed with no industrial (warehouse/factory) legal layer**  
`cross-repo` · effort ? · new  
- **Evidence:** CAP-10 (CAP-10-consent-posture-signoff.md:8) enforces 'capture only common areas you can visibly access; avoid faces, screens, paperwork...'; open captures default to review-required + downstream redaction. Terms.tsx operator schedule requires operator authority and 'explicit approval' for employee-only spaces but there is no EHS/safety authorization step, worker-PII/works-council path, or NDA/proprietary-data attestation anywhere in code. For a working factory/warehouse, workers are continuously present (cannot be 'avoided') and processes are trade secrets.
- **Impact:** Capturing a working factory/warehouse under a consent model built for public common areas — without site EHS/safety sign-off, worker-PII consent, or an NDA on proprietary operations — is legal and safety exposure at exactly the site type the founder wants to launch first.
- **Fix:** Add an industrial site-authorization + EHS/safety sign-off + NDA/proprietary-data attestation to the operator/reserved-job path before any factory/warehouse capture; extend redaction/consent guidance to continuous worker presence.

**107. [P0] Operator DPA / subprocessor list / access-audit terms and legal-EHS consent sign-off are unsigned (only blank templates exist)**  
`cross-repo` · effort ? · still-open  
- **Evidence:** PAID_MARKETPLACE_BETA_LAUNCH_GATE.md:66-69 require `legal_consent_posture_signoff` and `operator_dpa_data_processing_terms` (retention policy + subprocessor list + access-audit terms for delivered packages/hosted review); lines 42-43,54-55 list these under 'gate does not prove'. docs/operator_launch_evidence.template.json has every check `manual_live_evidence_required` with blank signed_record_uri/verified_by (lines 5-17); a search for any filled operator_launch_evidence.json returns nothing. CAP-10 (docs/beta-launch-audit-2026-07-03/operator-actions/CAP-10-consent-posture-signoff.md:16) has blank Owner/Date/Decision.
- **Impact:** Launching a paid/external beta with no executed DPA, no disclosed subprocessor list, and no signed consent/redaction posture leaves data processing legally unpapered; the gate itself will not certify launch-ready without these ids.
- **Fix:** Execute an operator DPA (retention, named subprocessors, access-audit for delivered packages and hosted sessions), obtain the legal/EHS signature on the CAP-10 record, and commit a filled operator_launch_evidence.json before flipping launch state.

**108. [P1] Buyers and site operators accept no Terms/Privacy at webapp signup (only the capturer application does)**  
`webapp` · effort ? · new  
- **Evidence:** CapturerSignUpFlow.tsx:918-941 gates submission on an `agreedToTerms` checkbox linking /terms and /privacy. BusinessSignUpFlow.tsx (buyer + operator signup) has no agree-to-terms control — grep for agree/terms/privacy/consent returns only a `privacySecurityConstraints` free-text field (lines 281,1255-1263) and no acceptance checkbox. Operators grant site authorization yet affirmatively accept nothing.
- **Impact:** No recorded, versioned acceptance of Terms/Privacy for buyers or operators; the operator-authority representation in Terms.tsx is never affirmatively accepted, weakening enforceability of the exact site-authority and rights terms the platform depends on.
- **Fix:** Add a required, timestamped, terms-version-stamped acceptance to BusinessSignUpFlow for buyers and operators (persisted), including an explicit operator authority/permission attestation.

**109. [P1] Data-retention policy is agent-scoped to WebApp Firestore only, unenforced, and does not reach pipeline artifacts, hosted world models, or delivered packages**  
`cross-repo` · effort ? · still-open  
- **Evidence:** ops/paperclip/DATA_RETENTION_POLICY.md:3 scopes itself to 'autonomous-org agents operating against the current Blueprint-WebApp Firestore model'; deletion protocol (lines 19-23) is a manual monthly human-approved review — 'Do not hard-delete automatically in Phase 1'; SARs (lines 27-32) are escalated, never fulfilled autonomously. No retention/TTL exists for pipeline raw/derived artifacts and grep for deleteAccount/data-deletion/eraseUser/deleteUser in server/ returns nothing.
- **Impact:** A retention or deletion obligation cannot be met operationally: nothing enforces the stated windows, and the policy never covers pipeline outputs, hosted world-model artifacts, or delivered Post-Training Data Packages where most captured data lives.
- **Fix:** Extend retention to pipeline output/ and delivered/hosted artifacts with enforced TTLs, and implement a real subject/account deletion path that fans out to Firestore, storage, and pipeline lineage.

**110. [P1] Takedown propagation enumerates but never executes recall, and no takedown drill has been run**  
`pipeline` · effort ? · still-open  
- **Evidence:** consent_takedown.py hard-codes webapp_revocation_sync.executed=false (line 526) and claim_boundary.manifest_is_local_enumeration_not_downstream_execution_proof=true (line 534, with webapp_revocation_sync_executed / hosted_session_takedown_executed false at 537-538); required_actions (_REQUIRED_TAKEDOWN_ACTIONS, lines 53-61: disable_signed_delivery_access, remove_hosted_review_assets, stop_downstream_training_or_finetuning_use, notify_buyer_and_owner) are listed not performed; sync_webapp_consent_revocation returns queued_unexecuted when PIPELINE_SYNC_* env unset (lines 619-624). No executed drill artifact exists in this clone.
- **Impact:** A consent revocation produces a to-do list, not a completed recall; nothing proves hosted sessions expired, signed delivery disabled, buyers notified, or delivered/training use stopped. No individual-person (face/PII) deletion path exists, only capture-level revocation.
- **Fix:** Wire and run an end-to-end takedown drill with evidence (revoke → propagate → executed webapp revocation sync + hosted-session expiry + signed-delivery disable + buyer notification), and document the irreducible limit for already-downloaded data.

**111. [P1] No cross-border / data-residency or international-transfer handling for non-US testers**  
`cross-repo` · effort ? · new  
- **Evidence:** Terms.tsx sets governing law to North Carolina/US only ('governed by the laws of the State of North Carolina'). DATA_RETENTION_POLICY.md:62 acknowledges 'GDPR exposure where EU resident data is present' but offers no DPA/SCCs/residency controls. Privacy.tsx says rights depend on 'your location' with no transfer mechanism. Grep for SCC/data residency/adequacy/cross-border across client/server/ops returns only a marketing skill checklist file, no product code/config.
- **Impact:** If any of the 100 beta testers are non-US, or they capture non-US (e.g. EU) industrial sites, personal/worker data is transferred and processed with no legal transfer basis and no residency guarantee.
- **Fix:** Either contractually scope the beta to US testers/US sites, or add a DPA with SCCs, a subprocessor list, and a documented transfer/residency posture before admitting non-US participants.

**112. [P2] Public privacy policy is vague on retention/DSR and a dead stub PrivacyPolicy.tsx remains in the tree**  
`webapp` · effort ? · new  
- **Evidence:** Privacy.tsx states retention as 'retain information for as long as needed' (Rights/privacy/retention card) with rights 'depending on your location' and no timeline or named subprocessors. An unrouted PrivacyPolicy.tsx is a literal stub with comments '{/* Continue with other sections... */}' and '{/* Add all other sections from your privacy policy document */}' (lines ~141-144), effective July 1 2025 (line 82); grep found no route reference in App.tsx.
- **Impact:** The live policy under-discloses retention and DSR handling relative to the CCPA/GDPR exposure the internal retention doc itself acknowledges; the stub is a latent wrong-content hazard if ever linked.
- **Fix:** Flesh out retention windows and a concrete DSR process/timeline in the routed Privacy.tsx, add a subprocessor list, and delete the dead PrivacyPolicy.tsx.

**113. [P2] iOS capture consent is browsewrap with optional (nil-default) legal URLs**  
`capture` · effort ? · new  
- **Evidence:** AuthView.swift:140-147 shows passive 'By continuing, you agree to Blueprint's [Terms of Service] and [Privacy Policy]' — implied consent, no checkbox. RuntimeConfig.swift:64-65 types termsOfServiceURL/privacyPolicyURL as `URL?` defaulting to nil (RuntimeConfigTests.swift:27 asserts nil in default config; only populated when provisioned, lines 73-74), so links render inert if config is missing.
- **Impact:** Capturers (primary beta actors) give only implied consent, and the linked Terms/Privacy may be non-functional in an unprovisioned build, weakening enforceability and disclosure of the capture consent posture.
- **Fix:** Make capture-app terms/privacy acceptance explicit (clickwrap) at first sign-in, and fall back to canonical https URLs when RuntimeConfig omits them.


### Onboarding & honest degraded-state UX

**114. [P1 🌐] Launch-city gate hard-blocks capture at any off-launch-city site, and its only recovery button silently disappears when runtime URLs are unset**  
`capture` · effort M · new  
- **Evidence:** LaunchCityGateView.swift:241-249 (unsupported/failed state's only primaryAction is 'Request launch access', wrapped in `if let launchRequestURL`), :28-44 (launchRequestURL = mainWebsiteURL ?? helpCenterURL ?? supportEmailURL — all nil-able), :279/:296; RuntimeConfig.swift:61-63,201-210 (nil if unset). A tester at a warehouse/factory outside the city whitelist is fully gated; misconfigured build leaves only 'Check again'.
- **Impact:** Humanoid-first industrial sites cluster in industrial zones outside launch-city cores; those testers cannot capture at all, and in a misconfigured build have zero recovery path.
- **Fix:** Always render a recovery affordance (fallback launch-access/support URL or in-app waitlist sheet independent of RuntimeConfig); add a build/config assertion that URLs are populated for beta; consider metro/region resolution rather than strict city displayName match.

**115. [P1] No tester-facing beta cohort onboarding / what-to-expect / support-escalation doc exists**  
`cross-repo` · effort M · new  
- **Evidence:** Repo-wide search finds only internal artifacts (100_BETA_TESTER_LAUNCH_BLOCKER_AUDIT, PAID_MARKETPLACE_BETA_LAUNCH_GATE, engineering specs, beta-launch-commander agent config). No capturer- or buyer-facing 'welcome / what to expect / how to get help' document.
- **Impact:** 100 external testers have no canonical guide covering supported location types, capture expectations, review timelines, degraded-state meanings, or escalation — driving confusion and unbounded support load.
- **Fix:** Author two short cohort docs (capturer + buyer) covering scope, first-run walkthrough, blocked/review/degraded meanings, payout expectations, and a single support channel; link from onboarding completion and in-app help.

**116. [P2 🌐] First-capture onboarding is consumer/nearby-space oriented with no industrial or assigned-site path**  
`capture` · effort M · new  
- **Evidence:** OnboardingFlowView.swift:117 ('Pick one nearby or current-place capture'), :696-733 (FirstCaptureGoal offers only .currentPlace and .nearbyOpportunity), :719. No assigned/scoped industrial site option, no safety/escort/PPE step. Contrast SkuPricing.swift:34-38 where 'Large warehouse' (SKU B) is a first-class pricing tier the onboarding never guides toward.
- **Impact:** A humanoid-first industrial tester is funneled toward casual consumer venues; no onboarding affordance for the site class the platform most wants, and no surfacing of industrial safety/access realities.
- **Fix:** Add an 'assigned/industrial site' first-capture goal routing to a scoped-job + permission + safety-acknowledgement path; align welcome copy with capture-first any-location positioning.

**117. [P2] Capturer support recovery is thin: no in-app help/support view, and all recovery links resolve from nil-defaulting runtime config**  
`capture` · effort M · new  
- **Evidence:** No SupportView/HelpView exists (grep empty). Recovery only external links: SettingsView.swift:328 helpCenterURL(), AuthView.swift:177 supportEmailURL, both nil-able (RuntimeConfig.swift:61-63). On a blocker there is no in-app self-serve help.
- **Impact:** If config lacks help/support values, 'Help'/'Contact support' does nothing; even configured, it bounces to email with no in-app FAQ/status — won't scale to 100 testers.
- **Fix:** Add a lightweight in-app Support/Help screen (FAQ + blocker explanations + guaranteed mailto fallback with hard-coded address); assert support URLs non-nil in beta builds.

**118. [P2] Payout onboarding silently swallows account-state load failures, leaving a misleading default state with no error or retry**  
`capture` · effort S · new  
- **Evidence:** StripeOnboardingView.swift:384-397 (loadAccountState catches all errors with only a print — no user-facing error, no retry). On failure the view keeps nil state, so schedule card shows 'Manual' (:265) and cashout shows locked copy (:344-347) as if verified truth.
- **Impact:** A capturer whose Stripe fetch fails sees a plausible-but-wrong 'locked/manual' payout state with no indication it failed and no retry — undermines trust, generates tickets.
- **Fix:** Surface a distinct load-error banner with Retry (mirror existing errorMessage alert path); distinguish 'could not load payout status' from 'payouts locked/ineligible'.

**119. [P2] Buyer onboarding is intake-only: every action funnels to /contact forms with no run/receive path and no timeline/SLA expectation**  
`webapp` · effort M · new  
- **Evidence:** OnboardingChecklist.tsx:328-401 (all actions href to /contact?persona=..., /world-models, /settings — none initiates a run/purchase), :543-545 with no timeframe. RequestConsole.tsx:638-642 next-step copy has no SLA. (Honesty note: RequestConsole/RunDetail ProofBoundary are commendably honest about blocked states — no world-model overclaiming found.)
- **Impact:** For 100 beta buyers the discover->buy->run->receive loop reduces to 'submit a contact form and wait', with no timeline/SLA — ambiguous waiting states feel like dead-ends and require human ops per buyer.
- **Fix:** Set explicit next-step timing/SLA copy after routing; give beta buyers at least one self-serve path (sample package access or a scoped request opening a RequestConsole) rather than only contact forms.
---

## Cross-reference vs the 2026-07-06 audit

- **37 findings are still-open** from the prior audit — most concentrated in payments/live-ops (live settlement, KYC, finance owner, payout monitoring), providers/spend (spend guard not enforced, no teardown proof, no rotation, no aggregate ceiling), delivery/access, and quality gates (slow lane not gated, ruff not wired). Treat these as durable, not resolved-and-regressed.
- **80 findings are new** — the prior audit did not center the any-location/industrial lens (all 34 🌐 items are new here) and did not surface several sharp code-level defects: the missing delivery producer (P0), disjoint storage rules (P0), client-writable submission status, world-readable `scenes` collection, the extractFrames OOM, single-shot uploads with no resume, batch-runner poison-capture blast radius, the mock/public `/ops/*` console, and Render auto-deploy decoupled from CI.
- **Partially improved since 2026-07-06:** the Stripe-native-parity CSRF path now has a WEB-06 native-client + Bearer exemption in `server/middleware/csrf.ts` (the specific `DELETE /v1/stripe/accounts/:id` 403 the prior gate hit). This needs a live paid-gate re-run to confirm the gate is now green; the code-level fix is present.

The prior audit's core verdict ("do not launch; six hard clusters") still holds. This audit refines it: the clusters are real, most are tractable, and the *industrial-first* framing adds a distinct, focused last-mile program that the prior audit did not scope.

---

## Appendix — methodology & caveats

**How this was produced.** An 18-dimension fan-out audit read the three repos directly (capture client, cloud functions, pipeline stages/contracts, webapp client/server). Each dimension's findings were passed to an independent adversarial verifier that re-opened the cited files and marked each finding CONFIRMED / PLAUSIBLE / REFUTED — refuted and ungrounded findings were dropped, and severities/scope were corrected (e.g. the industrial-hazard finding was downgraded P1→P2 once the verifier found forklift/human scenario families already exist in `robot_eval_dataset.py`). Two completeness critics then surfaced missing gap classes (site-scale metadata, multi-floor structure, mobile crash telemetry, client force-update/kill-switch, backup/DR, tax compliance, transactional notifications). 88 of 95 primary findings were CONFIRMED; 7 are PLAUSIBLE (flagged where the verifier could not fully confirm).

**Caveats / what this audit is not.**
- **Static, code+doc-grounded.** It did not execute the launch gates, tests, builds, or provider runs — this clone has no `output/` artifacts and running paid/GPU/CI lanes was out of scope. Every "still-open" item that depends on runtime state (payments settlement, teardown proof, gate green/red) needs a live re-run to move from "code path present/absent" to "proven in production."
- Line numbers are current as of this branch and will drift.
- Effort tags (S/M/L/XL) are rough engineering estimates, not commitments; a few critic-sourced gaps have no effort estimate.
- Dimension coverage was complete: all 18 dimensions returned (providers/spend/GPU and onboarding/UX were re-run after an initial structured-output failure and are fully included).

**Suggested next actions in-repo.**
1. Convert Gate 0–1 items into tracked issues with owners.
2. Re-run the dynamic gates listed above and attach artifacts so the "still-open" runtime items get a current verdict.
3. Add the committed warehouse/factory truth fixture first — it makes every subsequent industrial fix testable.
