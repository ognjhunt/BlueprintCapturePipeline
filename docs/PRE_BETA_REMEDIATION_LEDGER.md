# Pre-Beta Remediation Ledger

Tracks remediation of the 119 findings in `PRE_BETA_LAUNCH_GAP_AUDIT_2026-07-08.md`. Statuses: **fixed** (code merged/pushed + verified where possible), **scaffolded** (engineering done; blocked on a human/legal/infra decision, clearly noted), **todo**.

**Progress: 8 fixed · 2 scaffolded · 109 todo · 119 total.**

| # | Sev | 🌐 | Repo | Finding | Status | Evidence / note |
|---|-----|----|------|---------|--------|-----------------|
| R001 | P0 | 🌐 | cross-repo | Consent/authorization model is retail/public-space framed with no industrial (warehouse/factory) leg | ⬜ todo |  |
| R002 | P0 |  | cross-repo | Delivery producer is missing: pipeline never uploads packages to cloud, so the WebApp signed-URL han | 🟡 scaffolded | GCS delivery producer implemented+tested (uploads bundle to marketplace-artifacts/{ent}/, records gs:// URIs + webapp_ingestion contract, fail-closed). PENDING: webapp entitlement  |
| R003 | P0 |  | cross-repo | Storage rules are disjoint across repos, both deploy to the same project (last-writer-wins), and the | ✅ fixed | Canonical superset storage.rules byte-identical across repos + check-storage-rules-parity.sh + both parity guards wired into webapp CI (RUNNER_TEMP isolation, sibling matching-bran |
| R004 | P0 |  | cross-repo | Operator DPA / subprocessor list / access-audit terms and legal-EHS consent sign-off are unsigned (o | ⬜ todo |  |
| R005 | P1 | 🌐 | capture | Capture app hardcodes intended_space_type='industrial_unknown' with no site-type picker — the pipeli | ⬜ todo |  |
| R006 | P1 | 🌐 | pipeline | Industrial task-success grounding does not exist — eval_ready_task_grounding.py only ships a kitchen | ✅ fixed | Added containment_in_receptacle / placement_at_target_pose / transfer_zone_arrival success proxies parallel to the kitchen handle proxy, routed by material-handling/pick-place/tran |
| R007 | P1 | 🌐 | pipeline | No industrial simulator scene/scenario catalog or committed truth fixture — the only proven end-to-e | ⬜ todo |  |
| R008 | P1 | 🌐 | capture | No thermal / memory / disk monitoring during iPhone ARKit recording; storage only checked at upload | ⬜ todo |  |
| R009 | P1 | 🌐 | capture | extractFrames Cloud Function downloads the entire walkthrough video into a 2GiB memory-backed tmpfs  | ⬜ todo |  |
| R010 | P1 | 🌐 | pipeline | Privacy redaction is person-only — no badge/ID, screen, whiteboard, signage, or license-plate redact | ⬜ todo |  |
| R011 | P1 | 🌐 | cross-repo | 'policy_only' consent self-clears the consent-evidence gate with no operator permission document, an | ⬜ todo |  |
| R012 | P1 | 🌐 | capture | Site-operator authorization (VenuePermission) is demo-only UI: never persisted, uploaded, or enforce | ⬜ todo |  |
| R013 | P1 | 🌐 | pipeline | Scenario-variation family taxonomy is a single fixed global list, warehouse-flavored and factory-inc | 🟡 scaffolded | Factory hazard axes + per-site-category variation profiles + helpers added and exposed in scenario_family_library; 47 tests pass. PENDING: closure auto-selecting the profile per ca |
| R014 | P1 | 🌐 | pipeline | No committed industrial task-eval fixture/truth test; industrial hazard variations are template mock | ⬜ todo |  |
| R015 | P1 | 🌐 | cross-repo | Pipeline->WebApp ops dashboard summary contract is hardcoded to a home/residential task ontology | ⬜ todo |  |
| R016 | P1 | 🌐 | pipeline | Every merge gate is anchored on kitchen/indoor fixtures — no industrial/warehouse/factory gate exist | ⬜ todo |  |
| R017 | P1 | 🌐 | cross-repo | No site scale / dimensional metadata as capture truth or Site-card field | ⬜ todo |  |
| R018 | P1 | 🌐 | pipeline | Site-type recognition is brittle substring keyword matching over a tiny closed vocabulary with silen | ✅ fixed | New shared versioned site_taxonomy.py (canonical categories + synonyms + industrial flag + resolver); episode_spec task hints now recognize expanded industrial synonyms (distributi |
| R019 | P1 | 🌐 | capture | Launch-city gate hard-blocks capture at any off-launch-city site, and its only recovery button silen | ⬜ todo |  |
| R020 | P1 | 🌐 | capture | Venue-permission provenance is a read-only retail demo with no creation flow — industrial capturers  | ⬜ todo |  |
| R021 | P1 |  | capture | Uploads are single-shot PUTs with no intra-file resume — large captures restart from byte 0 on every | ⬜ todo |  |
| R022 | P1 |  | cross-repo | No capacity/cost/storage-volume model or bucket retention for large industrial captures at 100 users | ⬜ todo |  |
| R023 | P1 |  | cross-repo | Consent-revocation/takedown is pushed by the pipeline but not consumed by the webapp buyer-delivery  | ⬜ todo |  |
| R024 | P1 |  | pipeline | Batch inbox runner has no per-request exception isolation, quarantine, or dead-letter — one poison r | ⬜ todo |  |
| R025 | P1 |  | pipeline | Headline task success_rate for WAM runs is a VLM judgment over GENERATED video, not physics or captu | ⬜ todo |  |
| R026 | P1 |  | pipeline | Live simulator execution and live policy execution are unproven by default; honest beta deliverable  | ⬜ todo |  |
| R027 | P1 |  | cross-repo | Consent revocation is not self-enforcing across the delivery chain: revoked capture != revoked entit | ⬜ todo |  |
| R028 | P1 |  | webapp | Runtime forwarding defaults to required=false: WebApp returns 202 "queued_for_pipeline" even when no | ✅ fixed | Production now returns 5xx when pipeline forwarding not performed (not_configured->503, blocked/failed->502) regardless of FORWARD_REQUIRED; non-prod unchanged. tsc clean, 25 tests |
| R029 | P1 |  | cross-repo | Contract parity gate cannot run — shared BlueprintContracts module is absent; both repos run indepen | ⬜ todo |  |
| R030 | P1 |  | webapp | No entitlement/authz enforcement on eval-job submission; entitlement.approved is client-supplied and | ⬜ todo |  |
| R031 | P1 |  | webapp | Buyer cannot download purchased Task Eval Run / Post-Training Data Package artifacts from the app: e | ⬜ todo |  |
| R032 | P1 |  | webapp | Buyer disputes/chargebacks have no local webhook handler — linked payout is not frozen and order sta | ⬜ todo |  |
| R033 | P1 |  | cross-repo | No identity/KYC or background-check provider decision — payout-fraud and physical site-access screen | ⬜ todo |  |
| R034 | P1 |  | cross-repo | Live buyer-payment and capturer-payout settlement are unproven — only mock/contract readiness exists | ⬜ todo |  |
| R035 | P1 |  | cross-repo | No named human finance-review owner for payout exceptions | ⬜ todo |  |
| R036 | P1 |  | webapp | Operator console (/ops/*) is entirely mock data with no backend and is publicly routed without auth | ⬜ todo |  |
| R037 | P1 |  | webapp | No observability alerting for core beta failure classes (uploads, intake, provider, package, buyer-a | ⬜ todo |  |
| R038 | P1 |  | cross-repo | No beta-ops incident-response runbook (owner, escalation, rollback, takedown, customer-comms) and de | ✅ fixed | Beta incident-response runbook (ownership/escalation/detection, takedown drill, 4 degraded-state playbooks + comms) + health-checked Render rollback script (shellcheck-clean) + DEP |
| R039 | P1 |  | cross-repo | capture_submissions.status is client-writable despite rules comment claiming backend-only — a captur | ✅ fixed | capture_submissions.status confined to client lifecycle states; approved/paid remain Admin-SDK-only. [webapp 8e8313d, capture 411ca0f] |
| R040 | P1 |  | webapp | Firestore scenes collection lets any authenticated user read, update, or delete ANY scene (broken ob | ✅ fixed | scenes locked to admin read + backend-only writes (was world read/update/delete over contact PII). [webapp 8e8313d, capture 411ca0f] |
| R041 | P1 |  | pipeline | No aggregate/fleet spend budget ceiling — GPU cost guardrails are strictly per-job | ⬜ todo |  |
| R042 | P1 |  | pipeline | No storage lifecycle/retention on the primary capture bucket — unbounded storage cost | ⬜ todo |  |
| R043 | P1 |  | cross-repo | No load/soak test, capacity model, or cost-per-capture model in any repo | ⬜ todo |  |
| R044 | P1 |  | pipeline | Slow/integration/GPU lane never gates a merge or deploy | ⬜ todo |  |
| R045 | P1 |  | webapp | Render autoDeploy is decoupled from CI — a red build still deploys to production | ⬜ todo |  |
| R046 | P1 |  | cross-repo | No versioned release artifact, deploy SHA/tag, or rollback target | ⬜ todo |  |
| R047 | P1 |  | webapp | Buyers and site operators accept no Terms/Privacy at webapp signup (only the capturer application do | ⬜ todo |  |
| R048 | P1 |  | cross-repo | Data-retention policy is agent-scoped to WebApp Firestore only, unenforced, and does not reach pipel | ⬜ todo |  |
| R049 | P1 |  | pipeline | Takedown propagation enumerates but never executes recall, and no takedown drill has been run | ⬜ todo |  |
| R050 | P1 |  | cross-repo | No cross-border / data-residency or international-transfer handling for non-US testers | ⬜ todo |  |
| R051 | P1 |  | capture | No mobile crash/error telemetry on the capture clients (the primary data-collection tool is observab | ⬜ todo |  |
| R052 | P1 |  | cross-repo | No client version enforcement / force-update / remote kill-switch / maintenance mode for the capture | ⬜ todo |  |
| R053 | P1 |  | cross-repo | No backup / disaster-recovery / durability strategy for authoritative capture truth (Firestore + sto | ⬜ todo |  |
| R054 | P1 |  | cross-repo | No tester-facing beta cohort onboarding / what-to-expect / support-escalation doc exists | ⬜ todo |  |
| R055 | P1 |  | pipeline | GPU spend guard is a manual, dry-run-by-default tool that is never scheduled or enforced — no automa | ⬜ todo |  |
| R056 | P1 |  | pipeline | Booted orphan pods are never auto-reaped and render pods have no pod-side self-terminating watchdog  | ⬜ todo |  |
| R057 | P1 |  | pipeline | No platform-wide cumulative spend / GPU concurrency ceiling — spend gate is a per-run manual boolean | ⬜ todo |  |
| R058 | P1 |  | pipeline | Customer-eval cross-provider failover runtime is not implemented — eval GPU launches are single-prov | ⬜ todo |  |
| R059 | P1 |  | pipeline | Lambda single-adapter path never confirms teardown — termination is fire-and-forget, leaving open bi | ⬜ todo |  |
| R060 | P2 | 🌐 | pipeline | Industrial hazard ontology (forklift lanes, shared traffic, barriers, human-interaction zones) lives | ⬜ todo |  |
| R061 | P2 | 🌐 | pipeline | scene_placement/target_resolver openable + synonym affordance tables are kitchen/home-biased and are | ⬜ todo |  |
| R062 | P2 | 🌐 | capture | Live 'coverage %' hardcodes a 100 sq m target — false/meaningless for warehouse-scale sites | ⬜ todo |  |
| R063 | P2 | 🌐 | capture | Open-capture site identity is ephemeral per app launch — no multi-visit stitching for sites too big  | ⬜ todo |  |
| R064 | P2 | 🌐 | capture | No maximum-duration safeguard or mid-recording checkpointing for long single captures | ⬜ todo |  |
| R065 | P2 | 🌐 | cross-repo | No worker/employee consent concept or jurisdiction-specific (two-party/biometric) handling for sites | ⬜ todo |  |
| R066 | P2 | 🌐 | pipeline | Scorecard metric set has no industrial-assembly success semantics (dimensional/insertion tolerance,  | ⬜ todo |  |
| R067 | P2 | 🌐 | pipeline | Only kitchen has a built, runnable scene family in the Isaac realistic/parity render+eval lane; no w | ⬜ todo |  |
| R068 | P2 | 🌐 | pipeline | Industrial 'support' in the classical-sim lane is an un-runnable asset research catalog, not a runna | ⬜ todo |  |
| R069 | P2 | 🌐 | pipeline | No multi-floor / vertical-structure (mezzanine, multi-level racking) representation in the site mode | ⬜ todo |  |
| R070 | P2 | 🌐 | webapp | Buyer-facing marketplace location taxonomy omits factory / manufacturing despite it being the founde | ⬜ todo |  |
| R071 | P2 | 🌐 | cross-repo | No structured site environmental / operating-condition metadata (cold storage, floor surface, lighti | ⬜ todo |  |
| R072 | P2 | 🌐 | capture | First-capture onboarding is consumer/nearby-space oriented with no industrial or assigned-site path | ⬜ todo |  |
| R073 | P2 |  | pipeline | Task-aware detection-prompt augmentation hard-codes kitchen/home affordance expansions with no indus | ⬜ todo |  |
| R074 | P2 |  | cross-repo | Declared bundle hashes are never recomputed/compared server-side (bridge or pipeline) — canonical in | ⬜ todo |  |
| R075 | P2 |  | capture | Background-upload completion state is in-memory only; app termination mid-upload strands captures in | ⬜ todo |  |
| R076 | P2 |  | capture | Bridge is permissive: manifest-validation failure and a missing manifest are recorded but do not sto | ⬜ todo |  |
| R077 | P2 |  | pipeline | capture_batch_registry aborts the whole registry build if any one capture is malformed | ⬜ todo |  |
| R078 | P2 |  | pipeline | run_e2e --resume-completed-stages replays cached stage snapshots without validating upstream inputs  | ⬜ todo |  |
| R079 | P2 |  | pipeline | No real-world calibration anchors exist for any site, so sim-vs-real / digital-twin fidelity claims  | ⬜ todo |  |
| R080 | P2 |  | pipeline | LeRobot export action contract is hardcoded to a 7D single-end-effector delta pose — no bimanual/who | ⬜ todo |  |
| R081 | P2 |  | pipeline | Clip curation's default static-camera constraint rejects mobile-base capture needed for large indust | ⬜ todo |  |
| R082 | P2 |  | cross-repo | Lineage-ID enforcement asymmetry: WebApp validator omits request_id/owner_system that Pipeline intak | ⬜ todo |  |
| R083 | P2 |  | pipeline | Intake auth is a single static shared bearer with non-constant-time compare and no request signing/n | ⬜ todo |  |
| R084 | P2 |  | cross-repo | Single control-plane capture_root will block a multi-site beta unless per-site override JSON is conf | ⬜ todo |  |
| R085 | P2 |  | webapp | No expiration / license-term enforcement on entitlements — access is durable-forever until manual re | ⬜ todo |  |
| R086 | P2 |  | webapp | Hosted-session isolation collapses to site-world entitlement granularity: a co-entitled buyer can re | ⬜ todo |  |
| R087 | P2 |  | webapp | Payout-exception monitor is env-gated AI triage, not a proven live alerting system | ⬜ todo |  |
| R088 | P2 |  | webapp | Capturer payouts are approved independently of buyer revenue — treasury-drain / negative-margin risk | ⬜ todo |  |
| R089 | P2 |  | webapp | No beta cohort controls: no invite cap, per-cohort throttle, geo/site scope, or single beta kill swi | ⬜ todo |  |
| R090 | P2 |  | webapp | SLA watchdog exists but has no upload->package stage, no operator-facing surface, and no customer-fa | ⬜ todo |  |
| R091 | P2 |  | capture | Canonical raw-capture storage path has no upload size bound, enabling oversized/abusive uploads that | ✅ fixed | raw-capture path gains 12 GiB per-object bound; also capped /users,/accounts (500MB) and /menus updates (10MB) per Codex review. [capture 411ca0f, ec5ed86] |
| R092 | P2 |  | pipeline | Pipeline concurrency hard-capped at ~10 with 4h job timeout; 100-user tail latency unvalidated | ⬜ todo |  |
| R093 | P2 |  | pipeline | GPU privacy runners exposed with allUsers run.invoker — cost/DoS amplification surface | ⬜ todo |  |
| R094 | P2 |  | cross-repo | Ruff is not wired into CI in any repo | ⬜ todo |  |
| R095 | P2 |  | webapp | Documented release gate (alpha:check/preflight, smoke:launch, paid marketplace gate) is not enforced | ⬜ todo |  |
| R096 | P2 |  | webapp | WebApp coverage thresholds are trivially low for a money/entitlements surface | ⬜ todo |  |
| R097 | P2 |  | pipeline | Cross-repo sim-only gate is path-filtered and validated against a moving WebApp main | ⬜ todo |  |
| R098 | P2 |  | capture | Capture has no automated release/deploy gate; Android lint is non-blocking | ⬜ todo |  |
| R099 | P2 |  | webapp | Public privacy policy is vague on retention/DSR and a dead stub PrivacyPolicy.tsx remains in the tre | ⬜ todo |  |
| R100 | P2 |  | capture | iOS capture consent is browsewrap with optional (nil-default) legal URLs | ⬜ todo |  |
| R101 | P2 |  | webapp | Capturer payout path has no US tax-reporting compliance (1099-NEC / W-9 collection / backup withhold | ⬜ todo |  |
| R102 | P2 |  | cross-repo | No transactional lifecycle notifications to buyers/capturers on the money- and data-critical events | ⬜ todo |  |
| R103 | P2 |  | capture | Capturer support recovery is thin: no in-app help/support view, and all recovery links resolve from  | ⬜ todo |  |
| R104 | P2 |  | capture | Payout onboarding silently swallows account-state load failures, leaving a misleading default state  | ⬜ todo |  |
| R105 | P2 |  | webapp | Buyer onboarding is intake-only: every action funnels to /contact forms with no run/receive path and | ⬜ todo |  |
| R106 | P2 |  | pipeline | No rotation mechanism for GPU provider API keys (RunPod/Vast/Lambda/DigitalOcean); only the forwardi | ⬜ todo |  |
| R107 | P2 |  | pipeline | Reap exemption relies on a hard-coded allowlist of 8 warm pod IDs duplicated from the render module  | ⬜ todo |  |
| R108 | P3 | 🌐 | capture | No LiDAR depth-range guidance for high ceilings, long aisles, and tall racking | ⬜ todo |  |
| R109 | P3 | 🌐 | capture | Default capture tips and onboarding tutorial are framed around small home interiors | ⬜ todo |  |
| R110 | P3 | 🌐 | capture | No proactive low-light detection/warning for dim industrial lighting | ⬜ todo |  |
| R111 | P3 | 🌐 | pipeline | Industrial entity ontology is isolated to the qualification trust layer, not wired into scenario/WAM | ⬜ todo |  |
| R112 | P3 |  | pipeline | Four separate, divergent industrial taxonomies exist across the pipeline with no shared source of tr | ⬜ todo |  |
| R113 | P3 |  | pipeline | Per-class candidate cap tuning is defined only for residential environments; industrial captures get | ⬜ todo |  |
| R114 | P3 |  | capture | Open non-review captures default derived-generation and data-licensing to 'allowed' with consentStat | ⬜ todo |  |
| R115 | P3 |  | pipeline | cosmos3_wam substrate registry entry hardcodes a specific hosted provider (DeepInfra), leaking provi | ⬜ todo |  |
| R116 | P3 |  | pipeline | Real WAM/SAM3/depth/pose provider validation remains unproven (prior-audit items still open) but is  | ⬜ todo |  |
| R117 | P3 |  | webapp | Marketplace browse/search item types do not include the primary sellable outputs (Task Eval Runs / P | ⬜ todo |  |
| R118 | P3 |  | capture | iOS Stripe client omits Bearer when Firebase token is nil, yielding a confusing 403 CSRF error on st | ⬜ todo |  |
| R119 | P3 |  | pipeline | captures Firestore index orders on monotonic createdAt — sequential-key hotspot at scale | ⬜ todo |  |
