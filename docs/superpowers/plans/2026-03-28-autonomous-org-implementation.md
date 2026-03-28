# Autonomous Organization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Blueprint's 9 new autonomous agents into Paperclip with skill files, steering files, updated plugin webhook routes, and Notion integration — so the full tiered org chart is operational.

**Architecture:** Extend the existing Paperclip `.paperclip.yaml` with 9 new agent definitions and their routines. Create skill files that define each agent's behavior, inputs, outputs, and human gates. Extend the Blueprint Automation Plugin with new webhook endpoints for Ops triggers (Firestore events, Stripe, support inbox). Wire Notion API reads/writes through the plugin's tool system so agents can update Work Queue and Knowledge databases.

**Tech Stack:** Paperclip (Node.js), TypeScript, `@paperclipai/plugin-sdk`, Notion API (`@notionhq/client`), Slack webhooks, existing Vitest test framework.

**Spec:** `docs/superpowers/specs/2026-03-28-autonomous-org-design.md`
**Guide:** `AUTONOMOUS_ORG.md`

---

## File Map

### New Files
| File | Responsibility |
|------|---------------|
| `ops/paperclip/skills/ops-lead.md` | Ops Lead agent behavior spec |
| `ops/paperclip/skills/intake-agent.md` | Waitlist & Intake agent behavior spec |
| `ops/paperclip/skills/capture-qa-agent.md` | Capture QA agent behavior spec |
| `ops/paperclip/skills/field-ops-agent.md` | Scheduling & Field Ops agent behavior spec |
| `ops/paperclip/skills/finance-support-agent.md` | Finance & Support agent behavior spec |
| `ops/paperclip/skills/growth-lead.md` | Growth Lead agent behavior spec |
| `ops/paperclip/skills/conversion-agent.md` | Conversion Optimizer agent behavior spec |
| `ops/paperclip/skills/analytics-agent.md` | Analytics Agent behavior spec |
| `ops/paperclip/skills/market-intel-agent.md` | Market Intelligence agent behavior spec |
| `ops/paperclip/programs/conversion-agent-program.md` | Conversion Optimizer steering file |
| `ops/paperclip/programs/market-intel-program.md` | Market Intelligence steering file |
| `ops/paperclip/plugins/blueprint-automation/src/notion.ts` | Notion API client + tool handlers |
| `ops/paperclip/plugins/blueprint-automation/src/ops-webhooks.ts` | Ops webhook handlers (Firestore, Stripe, support) |
| `ops/paperclip/plugins/blueprint-automation/src/slack-notify.ts` | Slack notification helpers for agent reports |

### Modified Files
| File | Changes |
|------|---------|
| `ops/paperclip/blueprint-company/.paperclip.yaml` | Add 9 agents, 9 routines, new tasks |
| `ops/paperclip/blueprint-automation.config.json` | Add Ops and Growth department config |
| `ops/paperclip/plugins/blueprint-automation/src/worker.ts` | Register new webhook routes, tools, and jobs |
| `ops/paperclip/plugins/blueprint-automation/src/manifest.ts` | Add new webhook endpoints and tool declarations |
| `ops/paperclip/plugins/blueprint-automation/src/constants.ts` | Add new webhook keys, tool names, job keys |
| `ops/paperclip/plugins/blueprint-automation/package.json` | Add `@notionhq/client` dependency |

---

## Task 1: Create Ops Department Skill Files

**Files:**
- Create: `ops/paperclip/skills/ops-lead.md`
- Create: `ops/paperclip/skills/intake-agent.md`
- Create: `ops/paperclip/skills/capture-qa-agent.md`
- Create: `ops/paperclip/skills/field-ops-agent.md`
- Create: `ops/paperclip/skills/finance-support-agent.md`

- [ ] **Step 1: Create skills directory**

```bash
mkdir -p ops/paperclip/skills
```

- [ ] **Step 2: Create ops-lead.md**

Write to `ops/paperclip/skills/ops-lead.md`:

```markdown
# Ops Lead (`ops-lead`)

## Identity
- **Department:** Ops
- **Reports to:** CEO
- **Model:** Claude (claude-sonnet-4-6)
- **Phase:** 1 (Supervised)

## Purpose
You coordinate all Blueprint product operations. You route work between the intake, QA, scheduling, and finance agents. You produce a daily ops summary and escalate blockers to the CEO.

## Schedule
- Morning review: 8:30am ET weekdays
- Afternoon review: 2:30pm ET weekdays
- On-demand: when any ops agent escalates

## What You Do Each Cycle

### Morning Review (8:30am ET)
1. Pull queue depths from Firestore collections: `waitlist`, `inbound_requests`, `capture_submissions`, `support_tickets`
2. Check Stripe event log for overnight failures or disputes
3. Review any escalations from ops agents since last review
4. Produce a priority-ranked summary of open work
5. Assign work items to specialist agents via Paperclip issues
6. Post daily ops digest to Notion Work Queue and Slack

### Afternoon Review (2:30pm ET)
1. Check progress on morning assignments
2. Review any new items that arrived since morning
3. Escalate any blockers to CEO
4. Update Notion Work Queue with status changes

## Inputs
- Firestore collections (read-only): waitlist, inbound_requests, capture_submissions, support_tickets, stripe_events
- Notion Work Queue: items tagged System=Cross-System or any Ops-related
- Reports from: intake-agent, capture-qa-agent, field-ops-agent, finance-support-agent

## Outputs
- Daily ops digest → Notion Work Queue (new page) + Slack #ops channel
- Priority assignments → Paperclip issues assigned to specialist agents
- Escalations → Paperclip issue assigned to CEO
- Weekly ops trend summary (Friday) → CEO + Growth Lead

## Human Gates (Phase 1)
- All outputs are drafts for human review
- Do not send Slack messages directly; draft them for approval
- Do not create Paperclip issues directly; propose assignments

## Graduation Criteria
- Phase 1 → 2: 2 weeks at <10% human override rate on routing decisions
- Phase 2 → 3: 1 month with no mis-routes; founder sign-off

## Tools Available
- `blueprint-scan-work` — scan repos for drift
- `notion-read-work-queue` — read Notion Work Queue items
- `notion-write-work-queue` — create/update Notion Work Queue items
- `slack-post-digest` — post formatted digest to Slack

## Do Not
- Make payout decisions (route to finance-support-agent)
- Make QA pass/fail decisions (route to capture-qa-agent)
- Send external communications (route to intake-agent or finance-support-agent)
- Change agent priorities without CEO approval
```

- [ ] **Step 3: Create intake-agent.md**

Write to `ops/paperclip/skills/intake-agent.md`:

```markdown
# Waitlist & Intake Agent (`intake-agent`)

## Identity
- **Department:** Ops
- **Reports to:** Ops Lead
- **Model:** Claude (claude-sonnet-4-6)
- **Phase:** 1 (Supervised)

## Purpose
You process capturer applications (waitlist) and buyer inbound requests. You classify each by intent, score readiness, detect missing information, and draft responses.

## Schedule
- On-demand: triggered by new signup/request webhook
- Hourly: scan for stuck items (items with no status update for >24hrs)
- On-demand: Ops Lead assignment

## What You Do

### On New Capturer Application (waitlist webhook)
1. Read the application from Firestore `waitlist` collection
2. Classify by: market region, device type, experience level, referral source
3. Score invite readiness (0-100) based on:
   - Device compatibility with capture requirements (ARKit support, camera quality)
   - Market demand in their region (do we need captures there?)
   - Completeness of application
4. Flag any missing required information
5. Draft one of: invite email, rejection email, follow-up questions email
6. Write classification + score back to Firestore record
7. Create Notion Work Queue item with classification

### On New Buyer Request (inbound_requests webhook)
1. Read the request from Firestore `inbound_requests` collection
2. Classify by: use case (navigation, simulation, inspection, other), site type, urgency
3. Score priority (P0-P3) based on:
   - Commercial readiness (budget confirmed, timeline defined)
   - Site accessibility (do we have or can we get captures?)
   - Strategic fit (target market, use case alignment)
4. Detect missing information and draft follow-up questions
5. Write classification + priority back to Firestore record
6. If capture needed: create assignment for field-ops-agent
7. Create Notion Work Queue item

## Inputs
- Firestore `waitlist` collection: capturer applications
- Firestore `inbound_requests` collection: buyer requests
- Market-device fit matrix (from Knowledge DB)
- Capturer roster (for market coverage gaps)

## Outputs
- Classification label + priority score on each Firestore record
- Draft emails (invite, reject, follow-up) → human approval queue
- Missing-info flags with specific questions
- Notion Work Queue items for tracking
- Field ops assignments when capture is needed

## Human Gates (Phase 1)
- All draft emails require human approval before sending
- All invite/reject decisions require human approval
- Classification and scoring are advisory only

## Graduation Criteria
- Phase 1 → 2: 2 weeks, classification accuracy >90% (measured by human override rate)
- Phase 2 → 3: 1 month, follow-up email quality validated; founder sign-off

## Do Not
- Send any email or message without human approval
- Make payout or financial decisions
- Access or modify capture data directly
- Override Ops Lead priority assignments
```

- [ ] **Step 4: Create capture-qa-agent.md**

Write to `ops/paperclip/skills/capture-qa-agent.md`:

```markdown
# Capture QA Agent (`capture-qa-agent`)

## Identity
- **Department:** Ops
- **Reports to:** Ops Lead
- **Model:** Claude (claude-sonnet-4-6)
- **Phase:** 1 (Supervised)

## Purpose
You review pipeline outputs for quality, completeness, and privacy compliance. You flag recapture needs and draft payout recommendations.

## Schedule
- On-demand: triggered by pipeline completion webhook
- Daily 9am ET: scan for stalled captures (no status update >48hrs)
- On-demand: Ops Lead assignment

## What You Do

### On Pipeline Completion
1. Read pipeline output artifacts from GCS:
   - `qualification_summary.json` — overall qualification decision
   - `capture_quality_summary.json` — frame quality, coverage, pose accuracy
   - `rights_and_compliance_summary.json` — privacy, consent, rights status
   - `gemini_capture_fidelity_review.json` — multimodal fidelity assessment
2. Evaluate against QA thresholds:
   - Frame quality score >= 0.7
   - Coverage completeness >= 0.8
   - Privacy compliance: all required fields present, no unresolved flags
   - Pose accuracy within acceptable drift bounds
3. Produce QA verdict: PASS, FAIL, or BORDERLINE with evidence citations
4. If FAIL or BORDERLINE:
   - Identify specific issues (e.g., "kitchen area has <30% coverage")
   - Draft recapture request with specific instructions
   - Route to field-ops-agent via Paperclip issue
5. Draft payout recommendation based on quality and completeness
6. Update Firestore capture record with QA status
7. Create Notion Work Queue item

### Weekly Quality Trends (Friday)
1. Aggregate QA results from the week
2. Identify patterns (recurring issues by capturer, device, site type)
3. Produce trends report → Growth Lead + Ops Lead

## Inputs
- GCS pipeline artifacts (read-only)
- Firestore capture records
- QA threshold configuration (Knowledge DB)

## Outputs
- QA pass/fail verdict with evidence citations
- Recapture requests → field-ops-agent
- Payout recommendation drafts → human approval
- Weekly quality trends → Growth Lead + Ops Lead
- Notion Work Queue updates

## Human Gates (Phase 1 — some permanent)
- PERMANENT: All payout approvals require human sign-off
- Phase 1: All QA pass/fail decisions require human confirmation
- Phase 1: All recapture decisions require human confirmation

## Graduation Criteria
- Phase 1 → 2: 2 weeks, QA assessment matches human >90%
- Phase 2 → 3: 1 month, no false passes; founder sign-off
- Payout approval NEVER graduates — always human

## Do Not
- Approve payouts (always draft for human)
- Modify pipeline artifacts
- Override rights/privacy/consent flags
- Send communications to capturers directly
```

- [ ] **Step 5: Create field-ops-agent.md**

Write to `ops/paperclip/skills/field-ops-agent.md`:

```markdown
# Scheduling & Field Ops Agent (`field-ops-agent`)

## Identity
- **Department:** Ops
- **Reports to:** Ops Lead
- **Model:** Claude (claude-sonnet-4-6)
- **Phase:** 1 (Supervised)

## Purpose
You coordinate capture scheduling — calendar management, timezone normalization, travel estimation, capturer assignment, and reminders.

## Schedule
- On-demand: intake-agent qualifies a request needing capture
- On-demand: capture-qa-agent requests recapture
- Daily 7am ET: calendar review for upcoming captures

## What You Do

### On New Capture Assignment
1. Read the qualified request from Firestore
2. Identify candidate capturers based on:
   - Geographic proximity to site
   - Device compatibility
   - Availability (check Google Calendar)
   - Past quality scores
3. Estimate travel time via Google Maps API
4. Propose a schedule window (considering capturer timezone)
5. Draft calendar invite with:
   - Site address and access instructions
   - Capture requirements (areas to cover, special instructions)
   - Equipment checklist
   - Contact information
6. Draft capturer notification message

### Daily Calendar Review (7am ET)
1. Check today's and tomorrow's scheduled captures
2. Send reminder for captures happening today
3. Flag any unconfirmed captures (no capturer response >24hrs)
4. Report upcoming schedule to Ops Lead

### On Recapture Request
1. Read the QA agent's recapture instructions
2. Prioritize the original capturer (they know the site)
3. If unavailable, find alternative capturer
4. Follow the same scheduling flow as new capture

## Inputs
- Qualified requests (Firestore)
- Capturer roster + availability (Firestore + Google Calendar)
- Site metadata (location, access requirements)
- Google Calendar API
- Google Maps API (travel time estimation)

## Outputs
- Calendar invite proposals → human approval (Phase 1)
- Capturer assignment recommendations
- Reminder sequences (pre-capture, day-of, post-capture)
- Travel/logistics notes
- Notion Work Queue updates

## Human Gates (Phase 1 — some permanent)
- Phase 1: All calendar sends require human approval
- Phase 1-2: Conflict resolution requires human approval
- PERMANENT: Site access/permission issues require human handling

## Graduation Criteria
- Phase 1 → 2: 2 weeks, proposals accepted >85%
- Phase 2 → 3: 1 month, no scheduling errors; founder sign-off

## Do Not
- Send calendar invites without approval (Phase 1)
- Make payout or financial decisions
- Override QA decisions
- Grant site access permissions
```

- [ ] **Step 6: Create finance-support-agent.md**

Write to `ops/paperclip/skills/finance-support-agent.md`:

```markdown
# Finance & Support Agent (`finance-support-agent`)

## Identity
- **Department:** Ops
- **Reports to:** Ops Lead
- **Model:** Claude (claude-sonnet-4-6)
- **Phase:** 1 (Supervised)

## Purpose
You monitor Stripe health, triage payout issues, handle the support inbox, and draft responses.

## Schedule
- On-demand: Stripe webhook events (payout failures, disputes, account updates)
- On-demand: support inbox (email forward or form submission)
- Daily 10am ET: ledger reconciliation check

## What You Do

### On Stripe Event
1. Classify the event: payout_failure, dispute, account_update, charge_refunded, other
2. For payout failures:
   - Identify the affected capturer/account
   - Check for common causes (bank details, requirements, limits)
   - Draft a retry recommendation or manual intervention note
3. For disputes:
   - Flag immediately to Ops Lead as P0
   - Draft dispute response with transaction evidence
4. For account updates:
   - Check if requirements are due/past due
   - Draft follow-up communication if needed
5. Update Firestore payout records
6. Create Notion Work Queue item

### On Support Ticket
1. Classify: billing, technical, account, capture, general
2. Check for existing related tickets (dedup)
3. Draft response using templates from Knowledge DB
4. If technical: check error logs for related incidents
5. Route to appropriate specialist if needed
6. Create Notion Work Queue item

### Daily Ledger Reconciliation (10am ET)
1. Pull Stripe payouts settled in last 24hrs
2. Compare against Firestore payout records
3. Flag any discrepancies
4. Produce Stripe health summary → Ops Lead

## Inputs
- Stripe events (webhooks)
- Support tickets (webhook/email forward)
- Firestore payout records
- Buyer/capturer account data (Firestore)
- Support response templates (Knowledge DB)

## Outputs
- Payout issue triage + recommended action → human approval
- Support response drafts → human approval (Phase 1)
- Ledger discrepancy reports → Ops Lead
- Stripe health summary → Ops Lead
- Notion Work Queue updates

## Human Gates (Phase 1 — some permanent)
- PERMANENT: Payout approvals above configured threshold
- PERMANENT: Dispute responses
- PERMANENT: Refund approvals
- Phase 1: All support responses require human approval

## Graduation Criteria
- Phase 1 → 2: 2 weeks, draft quality validated
- Phase 2 → 3: 1 month, support response quality >95%; founder sign-off
- Payout/dispute/refund actions NEVER graduate

## Do Not
- Execute payouts or refunds without human approval
- Respond to disputes without human approval
- Access raw capture data
- Make compliance or legal decisions
```

- [ ] **Step 7: Commit**

```bash
git add ops/paperclip/skills/
git commit -m "feat: add Ops department agent skill files

Create behavior specs for 5 Ops agents:
- ops-lead: coordination and routing
- intake-agent: waitlist and buyer request processing
- capture-qa-agent: pipeline output quality review
- field-ops-agent: scheduling and logistics
- finance-support-agent: Stripe and support triage

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Create Growth Department Skill Files

**Files:**
- Create: `ops/paperclip/skills/growth-lead.md`
- Create: `ops/paperclip/skills/conversion-agent.md`
- Create: `ops/paperclip/skills/analytics-agent.md`
- Create: `ops/paperclip/skills/market-intel-agent.md`

- [ ] **Step 1: Create growth-lead.md**

Write to `ops/paperclip/skills/growth-lead.md`:

```markdown
# Growth Lead (`growth-lead`)

## Identity
- **Department:** Growth
- **Reports to:** CEO
- **Model:** Claude (claude-sonnet-4-6)
- **Phase:** 1 (Supervised)

## Purpose
You coordinate acquisition, conversion, and retention efforts. You set experiment priorities using ICE scoring. You synthesize analytics and market intelligence into actionable growth strategy.

## Schedule
- Daily 9am ET: review overnight analytics + agent reports
- Weekly Monday 10am ET: full growth review + experiment planning
- On-demand: analytics anomaly alerts

## What You Do

### Daily Review (9am ET)
1. Read Analytics Agent daily snapshot
2. Check for anomalies or significant metric changes
3. Review any completed experiment results from Conversion Optimizer
4. Review Market Intel daily digest
5. Update Notion Work Queue with any new growth items
6. Post brief daily growth status to Slack #growth

### Weekly Growth Review (Monday 10am ET)
1. Read Analytics Agent weekly report
2. Review all experiment results from past week
3. Read Market Intel weekly synthesis
4. Score and prioritize next week's experiments using ICE:
   - Impact (1-10): how much will this move the target metric?
   - Confidence (1-10): how sure are we it will work?
   - Ease (1-10): how easy is it to implement and measure?
5. Update Conversion Optimizer's `program.md` with new priorities
6. Update Market Intel's `program.md` if research focus should shift
7. Produce weekly growth summary → CEO + Notion

## Inputs
- Analytics Agent reports (daily + weekly)
- Conversion Optimizer experiment results
- Market Intel research digests
- Notion Work Queue (Growth-tagged items)

## Outputs
- Weekly growth summary → CEO + Notion Knowledge DB
- Experiment priority queue → Conversion Optimizer program.md
- Research briefs → Market Intel program.md
- Funnel health updates → Notion
- Daily growth status → Slack #growth

## Human Gates (Phase 1)
- All experiment priorities are recommendations; human sets final queue
- Strategy documents are drafts for human review

## Graduation Criteria
- Phase 1 → 2: 2 weeks, recommendations align with founder intent
- Phase 2 → 3: 1 month, no mis-prioritizations; founder sign-off

## Do Not
- Deploy code changes (that's Conversion Optimizer's job)
- Make budget decisions without CEO approval
- Make brand or positioning changes
- Override Ops department decisions
```

- [ ] **Step 2: Create conversion-agent.md**

Write to `ops/paperclip/skills/conversion-agent.md`:

```markdown
# Conversion Optimizer (`conversion-agent`)

## Identity
- **Department:** Growth
- **Reports to:** Growth Lead
- **Model:** Claude (claude-sonnet-4-6)
- **Phase:** 1 (Supervised)

## Purpose
You run an autoresearch-style experiment loop on the Blueprint webapp and capture app marketing surfaces. You test CTAs, onboarding flows, signup copy, pricing page layout, and any measurable UI surface.

## Schedule
- Weekly Monday 11am ET: start new experiment cycle (after Growth Lead review)
- On-demand: Growth Lead assigns new experiment focus
- On-demand: measurement period complete → evaluate results

## Autoresearch Experiment Loop

### 1. Read Steering File
Read `ops/paperclip/programs/conversion-agent-program.md` for:
- Current experiment focus area
- Constraints (what not to touch)
- Success metrics
- Learnings from last cycle

### 2. Analyze Current State
- Pull funnel metrics from Analytics Agent for the target area
- Read current source code for the pages/components under test
- Review experiment history from Notion Knowledge DB
- Identify the highest-leverage change opportunity

### 3. Propose Experiment
Write a proposal including:
- **Hypothesis:** "Changing X will improve Y by Z%"
- **Change description:** Exactly what code/copy/layout changes
- **Target metric:** Which metric to watch
- **Measurement period:** 24-72hrs depending on traffic
- **Rollback plan:** How to revert if metric degrades
- **Risk assessment:** Low/Medium/High

### 4. Implement Change
- Create a branch from main
- Make the code changes (copy, CTA text, layout, form fields)
- Run `npm run check` to verify no type errors
- Create a PR with the experiment proposal as description
- Request human approval (Phase 1)

### 5. Monitor Metrics
- After deployment, wait for measurement period
- Pull daily metric snapshots from Analytics Agent
- Compare against baseline (7-day pre-experiment average)

### 6. Evaluate Results
- If metric improved >= significance threshold: KEEP. Log win.
- If metric degraded: REVERT immediately. Log lesson.
- If inconclusive: extend measurement period by 50% or revert.
- Write experiment result report → Growth Lead + Notion Knowledge DB

### 7. Loop
- Update experiment history
- Read steering file for next focus
- Return to step 2

## Inputs
- Analytics data (via Analytics Agent queries)
- Source code: Blueprint-WebApp (`client/src/pages/`, `client/src/components/`)
- Experiment history (Notion Knowledge DB)
- Steering file: `ops/paperclip/programs/conversion-agent-program.md`

## Outputs
- Experiment proposals → human approval (Phase 1)
- Code change PRs → Blueprint-WebApp repo
- Experiment result reports → Growth Lead + Notion Knowledge DB
- Running experiment history log

## Human Gates (Phase 1)
- All code changes require human review and merge
- All deploys require human approval
- Structural changes (flow reordering, new pages) always require approval

## Graduation Criteria
- Phase 1 → 2: 1 month, experiment win rate >40%
- Phase 2 → 3: 2 months, no regressions from auto-deploys; founder sign-off

## Do Not
- Deploy without human approval (Phase 1)
- Touch backend/API code (frontend surfaces only)
- Modify rights, privacy, or compliance-related UI
- Run experiments on checkout or payment flows without explicit approval
- Change brand voice or positioning without Growth Lead approval
```

- [ ] **Step 3: Create analytics-agent.md**

Write to `ops/paperclip/skills/analytics-agent.md`:

```markdown
# Analytics Agent (`analytics-agent`)

## Identity
- **Department:** Growth
- **Reports to:** Growth Lead
- **Model:** Claude (claude-sonnet-4-6)
- **Phase:** 1 (Supervised)

## Purpose
You pull, aggregate, and interpret all measurable signals across the Blueprint platform. You detect anomalies, produce daily/weekly reports, and answer ad-hoc metric queries from other agents.

## Schedule
- Daily 6am ET: metrics pull + anomaly detection
- Weekly Sunday 11pm ET: full weekly report compilation
- On-demand: metric queries from other agents
- On-demand: anomaly alert (immediate)

## What You Do

### Daily Metrics Pull (6am ET)
1. Pull from analytics platform (GA4):
   - Page views by page
   - Session count and duration
   - Bounce rate by page
   - Conversion events (signup_started, signup_completed, request_submitted, checkout_initiated, checkout_completed)
2. Pull from Stripe API (read-only):
   - Transactions settled in last 24hrs
   - Revenue (gross, net)
   - Active subscriptions count
   - Payout volume
3. Pull from Firestore:
   - New user signups (buyer + capturer)
   - New inbound requests
   - New capture submissions
   - Queue depths (waitlist, pending QA, pending payout)
   - Support ticket count
4. Calculate derived metrics:
   - Visitor → signup conversion rate
   - Signup → first action rate
   - Request → qualified → purchased funnel rates
   - Capture → QA pass → listed → sold funnel rates
5. Run anomaly detection:
   - Compare each metric against 7-day rolling average
   - Flag if >2 standard deviations from mean
   - If anomaly detected: immediate alert to Growth Lead + CEO
6. Write daily snapshot to Notion Work Queue + Slack #analytics

### Weekly Report (Sunday 11pm ET)
1. Aggregate daily snapshots into weekly summary
2. Calculate week-over-week trends
3. Highlight top 3 wins and top 3 concerns
4. Include funnel visualization data
5. Write to Notion Knowledge DB as weekly report page
6. Post summary to Slack #growth

### Ad-Hoc Metric Queries
Other agents can request specific metrics. Respond with:
- Current value
- 7-day trend
- Comparison to previous period
- Any anomalies

## Inputs
- GA4 via Measurement Protocol or Data API
- Stripe API (read-only)
- Firestore (read-only)
- GitHub traffic API (optional)

## Outputs
- Daily metrics snapshot → Notion + Slack #analytics
- Weekly growth report → Notion Knowledge DB + Slack #growth
- Anomaly alerts → Growth Lead + CEO (immediate)
- Metric query responses → requesting agent

## Key Metrics

| Funnel | Metric | Source |
|--------|--------|--------|
| Buyer | Visitors | GA4 |
| Buyer | Signups | Firestore |
| Buyer | Requests submitted | Firestore |
| Buyer | Requests qualified | Firestore |
| Buyer | Purchases | Stripe |
| Buyer | Active sessions | Firestore |
| Capturer | Visitors | GA4 |
| Capturer | Signups | Firestore |
| Capturer | Waitlist → approved | Firestore |
| Capturer | First capture | Firestore |
| Capturer | QA pass rate | Firestore |
| Revenue | MRR | Stripe |
| Revenue | Transaction volume | Stripe |
| Revenue | Avg deal size | Stripe |
| Ops | Queue depths | Firestore |
| Ops | Resolution time | Firestore |
| Engagement | Bounce rate | GA4 |
| Engagement | Time on page | GA4 |

## Human Gates (Phase 1)
- None — reporting role only
- Human validates accuracy for first 2 weeks

## Graduation Criteria
- Phase 1 → 2: Accuracy >95% vs manual spot-check for 2 weeks
- Phase 2 → 3: 1 month, no errors; founder sign-off

## Do Not
- Write to any data source (read-only)
- Make decisions based on metrics (report them; let leads decide)
- Access personally identifiable information
- Share metrics externally
```

- [ ] **Step 4: Create market-intel-agent.md**

Write to `ops/paperclip/skills/market-intel-agent.md`:

```markdown
# Market Intelligence Agent (`market-intel-agent`)

## Identity
- **Department:** Growth
- **Reports to:** Growth Lead
- **Model:** Claude (claude-sonnet-4-6)
- **Phase:** 1 (Supervised)

## Purpose
You are an autoresearch-pattern agent for business intelligence. You continuously research competitors, market trends, new papers and techniques, pricing movements, partnership opportunities, and regulatory changes.

## Schedule
- Daily 7am ET: morning research scan
- Weekly Friday 3pm ET: deep weekly synthesis
- On-demand: CEO or Growth Lead ad-hoc research question

## Autoresearch Loop

### 1. Read Steering File
Read `ops/paperclip/programs/market-intel-program.md` for:
- Current research focus areas
- Constraints and off-limits topics
- Success metrics for relevance
- Recent context from last cycle

### 2. Scan Sources
For each research domain, scan designated sources:

**Competitors:**
- Company websites and blogs of: [list maintained in steering file]
- Crunchbase/PitchBook for funding rounds
- Product Hunt / HN for launches
- LinkedIn for hiring signals

**Technology:**
- ArXiv: world models, 3D reconstruction, NeRF/3DGS, robotics sim
- Conference proceedings: CVPR, ICRA, RSS, CoRL
- GitHub trending: robotics, 3D, simulation repos
- Key researcher blogs and Twitter/X

**Market:**
- Robotics industry reports
- Enterprise adoption case studies
- Deployment trend analyses
- Adjacent market movements (digital twins, autonomous vehicles)

**Regulatory:**
- Data privacy law changes (GDPR, CCPA, new regulations)
- Robotics safety standards updates
- Commercial drone/robot regulations

### 3. Extract and Score Signals
For each finding:
- **Relevance** (1-10): How directly does this affect Blueprint?
- **Urgency** (1-10): How soon should Blueprint act?
- **Actionability** (1-10): Can Blueprint do something concrete?
- Combined score = (Relevance * 0.4) + (Urgency * 0.3) + (Actionability * 0.3)

Only include findings with combined score >= 5.0 in the digest.

### 4. Produce Digest
- Daily: 3-5 top signals with one-line summaries and relevance scores
- Weekly: Deep synthesis with themes, recommended actions, competitor movement summary

### 5. Update Context
- Add key findings to running competitor tracker (Notion)
- Update source quality ratings (drop low-signal sources, add new ones)
- Note what worked and what didn't for next cycle

## Inputs
- Web search API (SerpAPI/Brave/Tavily)
- ArXiv API
- Steering file: `ops/paperclip/programs/market-intel-program.md`
- Previous digests (Notion Knowledge DB)

## Outputs
- Daily signal digest → Growth Lead + CEO (Notion page + Slack #research)
- Weekly deep synthesis → Notion Knowledge DB + Slack #research
- Ad-hoc research answers → requesting agent
- Competitor tracker updates → Notion

## Human Gates (Phase 1)
- None — research/reporting role only
- Human evaluates relevance and accuracy for first month

## Graduation Criteria
- Phase 1 → 2: 1 month, relevance score >80% (human-judged)
- Phase 2 → 3: 2 months, consistently actionable; founder sign-off

## Do Not
- Make strategic decisions (report findings; let leads decide)
- Contact competitors or external parties
- Share Blueprint internal information externally
- Publish research publicly
```

- [ ] **Step 5: Commit**

```bash
git add ops/paperclip/skills/growth-lead.md ops/paperclip/skills/conversion-agent.md ops/paperclip/skills/analytics-agent.md ops/paperclip/skills/market-intel-agent.md
git commit -m "feat: add Growth department agent skill files

Create behavior specs for 4 Growth agents:
- growth-lead: experiment prioritization and strategy
- conversion-agent: autoresearch CTA/onboarding optimization
- analytics-agent: metrics aggregation and anomaly detection
- market-intel-agent: autoresearch competitive intelligence

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Create Autoresearch Steering Files

**Files:**
- Create: `ops/paperclip/programs/conversion-agent-program.md`
- Create: `ops/paperclip/programs/market-intel-program.md`

- [ ] **Step 1: Create programs directory**

```bash
mkdir -p ops/paperclip/programs
```

- [ ] **Step 2: Create conversion-agent-program.md**

Write to `ops/paperclip/programs/conversion-agent-program.md`:

```markdown
# Conversion Optimizer — Current Focus

## Priority
Improve capturer signup completion rate. Current baseline: measure in first Analytics Agent report.

Focus area: `/client/src/pages/CapturerSignUpFlow.tsx` and related components.

Hypothesis to test first: simplify the signup form by reducing required fields to email + device type only (defer other fields to post-signup onboarding).

## Constraints
- Do NOT touch checkout or payment flows
- Do NOT modify rights/privacy/consent UI
- Do NOT change brand voice or core messaging (see WORLD_MODEL_STRATEGY_CONTEXT.md)
- Keep changes small — one variable per experiment
- Measurement period: minimum 48hrs per experiment

## Success Metrics
- Primary: capturer signup completion rate (started → completed)
- Secondary: time-to-first-capture after signup
- Guard rail: do not degrade buyer signup or inbound request rates

## Recent Context
- First cycle — no prior experiments
- Analytics Agent will provide baseline metrics after first daily run
- Current signup flow is multi-step; potential friction points unknown until baseline data
```

- [ ] **Step 3: Create market-intel-program.md**

Write to `ops/paperclip/programs/market-intel-program.md`:

```markdown
# Market Intelligence — Current Focus

## Priority
1. **Competitor landscape:** Identify all companies offering real-world capture → world model/digital twin products. Map their pricing, go-to-market, and funding status.
2. **World model papers:** Track latest world model papers (especially video world models, scene reconstruction, and 3D generation) that could become new backend options for Blueprint.
3. **Robotics deployment market:** Size the addressable market for site-specific world models for robot teams. Who is deploying humanoids/mobile robots indoors?

## Constraints
- Focus on publicly available information only
- Do not contact competitors or external parties
- Limit daily scan to 30 minutes of research time
- Prioritize actionable intelligence over comprehensive coverage

## Success Metrics
- Relevance: >80% of reported signals rated relevant by Growth Lead
- Actionability: at least 1 actionable recommendation per weekly synthesis
- Coverage: no major competitor move goes unreported for >1 week

## Recent Context
- First cycle — building baseline competitor map from scratch
- Key known competitors to track: [to be filled after first scan]
- Blueprint's positioning: capture-first, world-model-product-first (see WORLD_MODEL_STRATEGY_CONTEXT.md)
```

- [ ] **Step 4: Commit**

```bash
git add ops/paperclip/programs/
git commit -m "feat: add autoresearch steering files for conversion and market intel agents

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Add New Agents to Paperclip Configuration

**Files:**
- Modify: `ops/paperclip/blueprint-company/.paperclip.yaml`

- [ ] **Step 1: Read current .paperclip.yaml**

```bash
cat ops/paperclip/blueprint-company/.paperclip.yaml
```

Verify the current agent list matches what we documented (9 agents: ceo, cto, 6 engineering).

- [ ] **Step 2: Add 9 new agents to .paperclip.yaml**

Append to the `agents:` section (after the existing `capture-claude` agent):

```yaml
  # ── Ops Department ──────────────────────────────────────
  ops-lead:
    role: cto
    icon: bullseye
    capabilities: >-
      Coordinates all Blueprint product operations. Routes work between
      intake, QA, scheduling, and finance agents. Produces daily ops
      summary and escalates blockers to CEO. Read skill file at
      ops/paperclip/skills/ops-lead.md before every routine.
    adapter:
      type: claude_local
      config:
        cwd: /Users/nijelhunt_1/workspace/Blueprint-WebApp
        model: claude-sonnet-4-6
        timeoutSec: 1800
        dangerouslySkipPermissions: true
    budgetMonthlyCents: 3000

  intake-agent:
    role: engineer
    icon: inbox
    capabilities: >-
      Processes capturer waitlist applications and buyer inbound requests.
      Classifies, scores readiness, detects missing info, drafts responses.
      Read skill file at ops/paperclip/skills/intake-agent.md before every routine.
    adapter:
      type: claude_local
      config:
        cwd: /Users/nijelhunt_1/workspace/Blueprint-WebApp
        model: claude-sonnet-4-6
        timeoutSec: 1200
        dangerouslySkipPermissions: true
    budgetMonthlyCents: 2000

  capture-qa-agent:
    role: engineer
    icon: search
    capabilities: >-
      Reviews pipeline outputs for quality, completeness, and privacy
      compliance. Flags recapture needs and drafts payout recommendations.
      Read skill file at ops/paperclip/skills/capture-qa-agent.md before every routine.
    adapter:
      type: claude_local
      config:
        cwd: /Users/nijelhunt_1/workspace/BlueprintCapturePipeline
        model: claude-sonnet-4-6
        timeoutSec: 1200
        dangerouslySkipPermissions: true
    budgetMonthlyCents: 2000

  field-ops-agent:
    role: engineer
    icon: calendar
    capabilities: >-
      Coordinates capture scheduling, capturer assignment, calendar
      management, timezone normalization, and reminders.
      Read skill file at ops/paperclip/skills/field-ops-agent.md before every routine.
    adapter:
      type: claude_local
      config:
        cwd: /Users/nijelhunt_1/workspace/BlueprintCapture
        model: claude-sonnet-4-6
        timeoutSec: 1200
        dangerouslySkipPermissions: true
    budgetMonthlyCents: 2000

  finance-support-agent:
    role: engineer
    icon: dollar-sign
    capabilities: >-
      Monitors Stripe health, triages payout issues, handles support
      inbox, and drafts responses. Read skill file at
      ops/paperclip/skills/finance-support-agent.md before every routine.
    adapter:
      type: claude_local
      config:
        cwd: /Users/nijelhunt_1/workspace/Blueprint-WebApp
        model: claude-sonnet-4-6
        timeoutSec: 1200
        dangerouslySkipPermissions: true
    budgetMonthlyCents: 2000

  # ── Growth Department ───────────────────────────────────
  growth-lead:
    role: cto
    icon: trending-up
    capabilities: >-
      Coordinates acquisition, conversion, and retention. Sets experiment
      priorities using ICE scoring. Synthesizes analytics and market
      intelligence into growth strategy. Read skill file at
      ops/paperclip/skills/growth-lead.md before every routine.
    adapter:
      type: claude_local
      config:
        cwd: /Users/nijelhunt_1/workspace/Blueprint-WebApp
        model: claude-sonnet-4-6
        timeoutSec: 1800
        dangerouslySkipPermissions: true
    budgetMonthlyCents: 3000

  conversion-agent:
    role: engineer
    icon: flask-conical
    capabilities: >-
      Runs autoresearch-style experiment loop on webapp and capture app.
      Tests CTAs, onboarding flows, signup copy, and pricing layout.
      Read skill file at ops/paperclip/skills/conversion-agent.md and
      steering file at ops/paperclip/programs/conversion-agent-program.md
      before every routine.
    adapter:
      type: claude_local
      config:
        cwd: /Users/nijelhunt_1/workspace/Blueprint-WebApp
        model: claude-sonnet-4-6
        timeoutSec: 1800
        dangerouslySkipPermissions: true
    budgetMonthlyCents: 3000

  analytics-agent:
    role: engineer
    icon: bar-chart
    capabilities: >-
      Pulls, aggregates, and interprets all measurable signals. Detects
      anomalies, produces daily/weekly reports, answers ad-hoc metric
      queries. Read skill file at ops/paperclip/skills/analytics-agent.md
      before every routine.
    adapter:
      type: claude_local
      config:
        cwd: /Users/nijelhunt_1/workspace/Blueprint-WebApp
        model: claude-sonnet-4-6
        timeoutSec: 1200
        dangerouslySkipPermissions: true
    budgetMonthlyCents: 2000

  market-intel-agent:
    role: engineer
    icon: globe
    capabilities: >-
      Autoresearch-pattern agent for business intelligence. Researches
      competitors, market trends, papers, pricing, and regulations.
      Read skill file at ops/paperclip/skills/market-intel-agent.md and
      steering file at ops/paperclip/programs/market-intel-program.md
      before every routine.
    adapter:
      type: claude_local
      config:
        cwd: /Users/nijelhunt_1/workspace/Blueprint-WebApp
        model: claude-sonnet-4-6
        timeoutSec: 1800
        dangerouslySkipPermissions: true
    budgetMonthlyCents: 2000
```

- [ ] **Step 3: Add 9 new routines to .paperclip.yaml**

Append to the `routines:` section (after existing `capture-claude-review-loop`):

```yaml
  # ── Ops Department Routines ─────────────────────────────
  ops-lead-morning:
    agent: ops-lead
    priority: high
    concurrencyPolicy: coalesce_if_active
    catchUpPolicy: skip_missed
    triggers:
      - kind: schedule
        cronExpression: "30 8 * * 1-5"
        timezone: America/New_York

  ops-lead-afternoon:
    agent: ops-lead
    priority: medium
    concurrencyPolicy: coalesce_if_active
    catchUpPolicy: skip_missed
    triggers:
      - kind: schedule
        cronExpression: "30 14 * * 1-5"
        timezone: America/New_York

  intake-agent-hourly:
    agent: intake-agent
    priority: medium
    concurrencyPolicy: coalesce_if_active
    catchUpPolicy: skip_missed
    triggers:
      - kind: schedule
        cronExpression: "0 * * * 1-5"
        timezone: America/New_York

  capture-qa-daily:
    agent: capture-qa-agent
    priority: medium
    concurrencyPolicy: coalesce_if_active
    catchUpPolicy: skip_missed
    triggers:
      - kind: schedule
        cronExpression: "0 9 * * 1-5"
        timezone: America/New_York

  field-ops-daily:
    agent: field-ops-agent
    priority: medium
    concurrencyPolicy: coalesce_if_active
    catchUpPolicy: skip_missed
    triggers:
      - kind: schedule
        cronExpression: "0 7 * * 1-5"
        timezone: America/New_York

  finance-support-daily:
    agent: finance-support-agent
    priority: medium
    concurrencyPolicy: coalesce_if_active
    catchUpPolicy: skip_missed
    triggers:
      - kind: schedule
        cronExpression: "0 10 * * 1-5"
        timezone: America/New_York

  # ── Growth Department Routines ──────────────────────────
  growth-lead-daily:
    agent: growth-lead
    priority: high
    concurrencyPolicy: coalesce_if_active
    catchUpPolicy: skip_missed
    triggers:
      - kind: schedule
        cronExpression: "0 9 * * 1-5"
        timezone: America/New_York

  growth-lead-weekly:
    agent: growth-lead
    priority: high
    concurrencyPolicy: coalesce_if_active
    catchUpPolicy: skip_missed
    triggers:
      - kind: schedule
        cronExpression: "0 10 * * 1"
        timezone: America/New_York

  analytics-daily:
    agent: analytics-agent
    priority: high
    concurrencyPolicy: coalesce_if_active
    catchUpPolicy: skip_missed
    triggers:
      - kind: schedule
        cronExpression: "0 6 * * *"
        timezone: America/New_York

  analytics-weekly:
    agent: analytics-agent
    priority: medium
    concurrencyPolicy: coalesce_if_active
    catchUpPolicy: skip_missed
    triggers:
      - kind: schedule
        cronExpression: "0 23 * * 0"
        timezone: America/New_York

  conversion-weekly:
    agent: conversion-agent
    priority: medium
    concurrencyPolicy: coalesce_if_active
    catchUpPolicy: skip_missed
    triggers:
      - kind: schedule
        cronExpression: "0 11 * * 1"
        timezone: America/New_York

  market-intel-daily:
    agent: market-intel-agent
    priority: medium
    concurrencyPolicy: coalesce_if_active
    catchUpPolicy: skip_missed
    triggers:
      - kind: schedule
        cronExpression: "0 7 * * 1-5"
        timezone: America/New_York

  market-intel-weekly:
    agent: market-intel-agent
    priority: medium
    concurrencyPolicy: coalesce_if_active
    catchUpPolicy: skip_missed
    triggers:
      - kind: schedule
        cronExpression: "0 15 * * 5"
        timezone: America/New_York
```

- [ ] **Step 4: Add bootstrap tasks for new agents**

Append to the `tasks:` section:

```yaml
  ops-lead-bootstrap:
    status: todo
    priority: high
    projectWorkspaceKey: blueprint-executive-ops
  intake-agent-bootstrap:
    status: todo
    priority: high
    projectWorkspaceKey: blueprint-webapp
  capture-qa-agent-bootstrap:
    status: todo
    priority: high
    projectWorkspaceKey: blueprint-capture-pipeline
  field-ops-agent-bootstrap:
    status: todo
    priority: high
    projectWorkspaceKey: blueprint-capture
  finance-support-agent-bootstrap:
    status: todo
    priority: high
    projectWorkspaceKey: blueprint-webapp
  growth-lead-bootstrap:
    status: todo
    priority: high
    projectWorkspaceKey: blueprint-executive-ops
  conversion-agent-bootstrap:
    status: todo
    priority: high
    projectWorkspaceKey: blueprint-webapp
  analytics-agent-bootstrap:
    status: todo
    priority: high
    projectWorkspaceKey: blueprint-executive-ops
  market-intel-agent-bootstrap:
    status: todo
    priority: high
    projectWorkspaceKey: blueprint-executive-ops
```

- [ ] **Step 5: Verify YAML is valid**

```bash
python3 -c "import yaml; yaml.safe_load(open('ops/paperclip/blueprint-company/.paperclip.yaml'))" && echo "YAML valid"
```

Expected: `YAML valid`

- [ ] **Step 6: Commit**

```bash
git add ops/paperclip/blueprint-company/.paperclip.yaml
git commit -m "feat: add 9 new agents, 13 routines, and bootstrap tasks to Paperclip config

Adds Ops department (ops-lead, intake-agent, capture-qa-agent,
field-ops-agent, finance-support-agent) and Growth department
(growth-lead, conversion-agent, analytics-agent, market-intel-agent)
to the Paperclip company definition.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Add Notion API Integration to Plugin

**Files:**
- Create: `ops/paperclip/plugins/blueprint-automation/src/notion.ts`
- Modify: `ops/paperclip/plugins/blueprint-automation/package.json`

- [ ] **Step 1: Add @notionhq/client dependency**

```bash
cd ops/paperclip/plugins/blueprint-automation && npm install @notionhq/client && cd ../../../..
```

- [ ] **Step 2: Create notion.ts**

Write to `ops/paperclip/plugins/blueprint-automation/src/notion.ts`:

```typescript
import { Client } from "@notionhq/client";

// Database IDs from Blueprint Hub
const WORK_QUEUE_DB = "f83b6c53-a33a-4790-9ca4-786dddadad46";
const SKILLS_DB = "4e37bd7a-e448-4f81-aa3e-b8860826e98c";
const KNOWLEDGE_DB = "7c729783-c377-4342-bf00-5555b88a2ec6";

export interface NotionConfig {
  token: string;
}

export function createNotionClient(config: NotionConfig): Client {
  return new Client({ auth: config.token });
}

// ── Work Queue Operations ────────────────────────────────

export interface WorkQueueItem {
  title: string;
  priority: "P0" | "P1" | "P2" | "P3";
  system: "Cross-System" | "WebApp" | "Capture" | "Pipeline" | "Validation";
  lifecycleStage: string;
  workType: "Task" | "Research" | "Refresh" | "SOP" | "Improvement";
  substage?: string;
}

export async function createWorkQueueItem(
  client: Client,
  item: WorkQueueItem
): Promise<string> {
  const response = await client.pages.create({
    parent: { database_id: WORK_QUEUE_DB },
    properties: {
      Title: { title: [{ text: { content: item.title } }] },
      Priority: { select: { name: item.priority } },
      System: { select: { name: item.system } },
      "Lifecycle Stage": { select: { name: item.lifecycleStage } },
      "Work Type": { select: { name: item.workType } },
      ...(item.substage
        ? { Substage: { rich_text: [{ text: { content: item.substage } }] } }
        : {}),
    },
  });
  return response.id;
}

export async function queryWorkQueue(
  client: Client,
  filters: { system?: string; priority?: string; lifecycleStage?: string }
): Promise<Array<{ id: string; title: string; priority: string; system: string }>> {
  const filterConditions: Array<Record<string, unknown>> = [];

  if (filters.system) {
    filterConditions.push({
      property: "System",
      select: { equals: filters.system },
    });
  }
  if (filters.priority) {
    filterConditions.push({
      property: "Priority",
      select: { equals: filters.priority },
    });
  }
  if (filters.lifecycleStage) {
    filterConditions.push({
      property: "Lifecycle Stage",
      select: { equals: filters.lifecycleStage },
    });
  }

  const response = await client.databases.query({
    database_id: WORK_QUEUE_DB,
    filter:
      filterConditions.length > 1
        ? { and: filterConditions as any }
        : filterConditions.length === 1
          ? (filterConditions[0] as any)
          : undefined,
    sorts: [{ property: "Priority", direction: "ascending" }],
    page_size: 50,
  });

  return response.results.map((page: any) => ({
    id: page.id,
    title: page.properties.Title?.title?.[0]?.text?.content ?? "",
    priority: page.properties.Priority?.select?.name ?? "",
    system: page.properties.System?.select?.name ?? "",
  }));
}

// ── Knowledge DB Operations ──────────────────────────────

export interface KnowledgeEntry {
  title: string;
  type: "Concept" | "Reference" | "How-To" | "Decision" | "Architecture" | "Contract";
  system: "Cross-System" | "WebApp" | "Capture" | "Pipeline" | "Validation";
  content: string;
}

export async function createKnowledgeEntry(
  client: Client,
  entry: KnowledgeEntry
): Promise<string> {
  const response = await client.pages.create({
    parent: { database_id: KNOWLEDGE_DB },
    properties: {
      Title: { title: [{ text: { content: entry.title } }] },
      Type: { select: { name: entry.type } },
      System: { select: { name: entry.system } },
      "Agent Surface": {
        multi_select: [{ name: "Shared" }, { name: "Claude" }],
      },
      "Source of Truth": { select: { name: "Notion" } },
    },
    children: [
      {
        object: "block",
        type: "paragraph",
        paragraph: {
          rich_text: [{ type: "text", text: { content: entry.content } }],
        },
      },
    ],
  });
  return response.id;
}

// ── Tool Handler Factories ───────────────────────────────

export function buildNotionToolHandlers(client: Client) {
  return {
    "notion-read-work-queue": async (params: {
      system?: string;
      priority?: string;
    }) => {
      const items = await queryWorkQueue(client, params);
      return {
        success: true,
        items,
        count: items.length,
      };
    },

    "notion-write-work-queue": async (params: WorkQueueItem) => {
      const id = await createWorkQueueItem(client, params);
      return { success: true, pageId: id };
    },

    "notion-write-knowledge": async (params: KnowledgeEntry) => {
      const id = await createKnowledgeEntry(client, params);
      return { success: true, pageId: id };
    },
  };
}
```

- [ ] **Step 3: Commit**

```bash
git add ops/paperclip/plugins/blueprint-automation/src/notion.ts ops/paperclip/plugins/blueprint-automation/package.json ops/paperclip/plugins/blueprint-automation/package-lock.json
git commit -m "feat: add Notion API client and tool handlers for agent Work Queue and Knowledge DB access

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Add Ops Webhook Handlers to Plugin

**Files:**
- Create: `ops/paperclip/plugins/blueprint-automation/src/ops-webhooks.ts`
- Create: `ops/paperclip/plugins/blueprint-automation/src/slack-notify.ts`
- Modify: `ops/paperclip/plugins/blueprint-automation/src/constants.ts`

- [ ] **Step 1: Update constants.ts with new webhook keys and tool names**

Add to `ops/paperclip/plugins/blueprint-automation/src/constants.ts`:

```typescript
// Add to WEBHOOK_KEYS
export const WEBHOOK_KEYS = {
  github: "github",
  ci: "ci",
  intake: "intake",
  // New Ops webhook endpoints
  opsFirestore: "ops-firestore",
  opsStripe: "ops-stripe",
  opsSupport: "ops-support",
} as const;

// Add to TOOL_NAMES
export const TOOL_NAMES = {
  scanWork: "blueprint-scan-work",
  upsertWorkItem: "blueprint-upsert-work-item",
  reportBlocker: "blueprint-report-blocker",
  resolveWorkItem: "blueprint-resolve-work-item",
  // New Notion tools
  notionReadWorkQueue: "notion-read-work-queue",
  notionWriteWorkQueue: "notion-write-work-queue",
  notionWriteKnowledge: "notion-write-knowledge",
  // New Slack tools
  slackPostDigest: "slack-post-digest",
} as const;

// Add to JOB_KEYS
export const JOB_KEYS = {
  repoScan: "repo-scan",
  // New ops jobs
  opsQueueScan: "ops-queue-scan",
} as const;
```

- [ ] **Step 2: Create ops-webhooks.ts**

Write to `ops/paperclip/plugins/blueprint-automation/src/ops-webhooks.ts`:

```typescript
import type { PluginSetupContext, PluginWebhookInput } from "@paperclipai/plugin-sdk";

export interface OpsWebhookResult {
  handled: boolean;
  agentAssignment?: string;
  issueTitle?: string;
}

/**
 * Handle Firestore-triggered events for new signups, requests, and captures.
 * Expected payload:
 * {
 *   event: "waitlist.created" | "request.created" | "capture.completed",
 *   documentId: string,
 *   collection: string,
 *   data: Record<string, unknown>
 * }
 */
export async function handleFirestoreWebhook(
  input: PluginWebhookInput,
  ctx: PluginSetupContext
): Promise<OpsWebhookResult> {
  const body = input.body as {
    event: string;
    documentId: string;
    collection: string;
    data: Record<string, unknown>;
  };

  if (!body.event || !body.documentId) {
    return { handled: false };
  }

  const eventHandlers: Record<string, { agent: string; prefix: string }> = {
    "waitlist.created": { agent: "intake-agent", prefix: "Waitlist" },
    "request.created": { agent: "intake-agent", prefix: "Inbound Request" },
    "capture.completed": { agent: "capture-qa-agent", prefix: "Capture QA" },
  };

  const handler = eventHandlers[body.event];
  if (!handler) {
    return { handled: false };
  }

  const title = `${handler.prefix}: ${body.documentId}`;
  const fingerprint = `firestore:${body.collection}:${body.documentId}`;

  // Check for existing issue with this fingerprint
  const existingMapping = await ctx.pluginEntities.find(
    "source-mapping",
    fingerprint
  );

  if (existingMapping) {
    return { handled: true, issueTitle: title };
  }

  // Create new issue assigned to the appropriate agent
  const issue = await ctx.issues.create({
    title,
    description: `Firestore event: ${body.event}\nDocument: ${body.collection}/${body.documentId}\n\nData: ${JSON.stringify(body.data, null, 2)}`,
    priority: body.event === "request.created" ? "high" : "medium",
    assignee: handler.agent,
  });

  // Store mapping for dedup
  await ctx.pluginEntities.upsert("source-mapping", fingerprint, {
    issueId: issue.id,
    sourceType: "firestore",
    sourceId: `${body.collection}:${body.documentId}`,
    event: body.event,
    createdAt: new Date().toISOString(),
  });

  return { handled: true, agentAssignment: handler.agent, issueTitle: title };
}

/**
 * Handle Stripe webhook events forwarded to Paperclip.
 * Expected payload: standard Stripe event object.
 */
export async function handleStripeWebhook(
  input: PluginWebhookInput,
  ctx: PluginSetupContext
): Promise<OpsWebhookResult> {
  const body = input.body as {
    type: string;
    id: string;
    data?: { object?: Record<string, unknown> };
  };

  if (!body.type || !body.id) {
    return { handled: false };
  }

  const relevantEvents = [
    "payout.failed",
    "charge.dispute.created",
    "account.updated",
    "charge.refunded",
  ];

  if (!relevantEvents.includes(body.type)) {
    return { handled: true }; // Acknowledged but not actionable
  }

  const priority = body.type === "charge.dispute.created" ? "high" : "medium";
  const title = `Stripe: ${body.type} (${body.id})`;
  const fingerprint = `stripe:${body.type}:${body.id}`;

  const existingMapping = await ctx.pluginEntities.find(
    "source-mapping",
    fingerprint
  );

  if (existingMapping) {
    return { handled: true, issueTitle: title };
  }

  const issue = await ctx.issues.create({
    title,
    description: `Stripe event: ${body.type}\nEvent ID: ${body.id}\n\nData: ${JSON.stringify(body.data?.object ?? {}, null, 2)}`,
    priority,
    assignee: "finance-support-agent",
  });

  await ctx.pluginEntities.upsert("source-mapping", fingerprint, {
    issueId: issue.id,
    sourceType: "stripe",
    sourceId: body.id,
    event: body.type,
    createdAt: new Date().toISOString(),
  });

  return {
    handled: true,
    agentAssignment: "finance-support-agent",
    issueTitle: title,
  };
}

/**
 * Handle support inbox webhook (email forward or form submission).
 * Expected payload:
 * {
 *   subject: string,
 *   from: string,
 *   body: string,
 *   source: "email" | "form",
 *   receivedAt: string
 * }
 */
export async function handleSupportWebhook(
  input: PluginWebhookInput,
  ctx: PluginSetupContext
): Promise<OpsWebhookResult> {
  const body = input.body as {
    subject: string;
    from: string;
    body: string;
    source: string;
    receivedAt: string;
  };

  if (!body.subject || !body.from) {
    return { handled: false };
  }

  const title = `Support: ${body.subject}`;
  const fingerprint = `support:${body.from}:${body.receivedAt ?? Date.now()}`;

  const issue = await ctx.issues.create({
    title,
    description: `From: ${body.from}\nSource: ${body.source}\nReceived: ${body.receivedAt}\n\n${body.body}`,
    priority: "medium",
    assignee: "finance-support-agent",
  });

  await ctx.pluginEntities.upsert("source-mapping", fingerprint, {
    issueId: issue.id,
    sourceType: "support",
    sourceId: body.from,
    event: "ticket.created",
    createdAt: new Date().toISOString(),
  });

  return {
    handled: true,
    agentAssignment: "finance-support-agent",
    issueTitle: title,
  };
}
```

- [ ] **Step 3: Create slack-notify.ts**

Write to `ops/paperclip/plugins/blueprint-automation/src/slack-notify.ts`:

```typescript
export interface SlackDigest {
  channel: string;
  title: string;
  sections: Array<{
    heading: string;
    items: string[];
  }>;
}

export async function postSlackDigest(
  webhookUrl: string,
  digest: SlackDigest
): Promise<boolean> {
  const blocks: Array<Record<string, unknown>> = [
    {
      type: "header",
      text: { type: "plain_text", text: digest.title },
    },
  ];

  for (const section of digest.sections) {
    blocks.push({
      type: "section",
      text: {
        type: "mrkdwn",
        text: `*${section.heading}*\n${section.items.map((i) => `• ${i}`).join("\n")}`,
      },
    });
  }

  const response = await fetch(webhookUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ blocks }),
  });

  return response.ok;
}

export function buildSlackToolHandler(webhookUrl: string) {
  return {
    "slack-post-digest": async (params: SlackDigest) => {
      const ok = await postSlackDigest(webhookUrl, params);
      return { success: ok };
    },
  };
}
```

- [ ] **Step 4: Commit**

```bash
git add ops/paperclip/plugins/blueprint-automation/src/ops-webhooks.ts ops/paperclip/plugins/blueprint-automation/src/slack-notify.ts ops/paperclip/plugins/blueprint-automation/src/constants.ts
git commit -m "feat: add Ops webhook handlers (Firestore, Stripe, support) and Slack notification helpers

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Wire New Handlers into Plugin Worker

**Files:**
- Modify: `ops/paperclip/plugins/blueprint-automation/src/worker.ts`
- Modify: `ops/paperclip/plugins/blueprint-automation/src/manifest.ts`

- [ ] **Step 1: Add imports to worker.ts**

Add at the top of `ops/paperclip/plugins/blueprint-automation/src/worker.ts`, after existing imports:

```typescript
import {
  handleFirestoreWebhook,
  handleStripeWebhook,
  handleSupportWebhook,
} from "./ops-webhooks.js";
import { createNotionClient, buildNotionToolHandlers } from "./notion.js";
import { buildSlackToolHandler } from "./slack-notify.js";
```

- [ ] **Step 2: Register new tool handlers in setup()**

In the `setup()` function of the plugin definition, after existing tool registrations, add:

```typescript
    // ── Notion Tools ──────────────────────────────────────
    const notionToken = await ctx.secrets.resolve("NOTION_API_TOKEN");
    if (notionToken) {
      const notionClient = createNotionClient({ token: notionToken });
      const notionTools = buildNotionToolHandlers(notionClient);
      for (const [name, handler] of Object.entries(notionTools)) {
        ctx.tools.register(name, {
          description: `Notion tool: ${name}`,
          handler: async (params: any) => handler(params),
        });
      }
    }

    // ── Slack Tools ───────────────────────────────────────
    const slackWebhookUrl = await ctx.secrets.resolve("SLACK_OPS_WEBHOOK_URL");
    if (slackWebhookUrl) {
      const slackTools = buildSlackToolHandler(slackWebhookUrl);
      for (const [name, handler] of Object.entries(slackTools)) {
        ctx.tools.register(name, {
          description: `Slack tool: ${name}`,
          handler: async (params: any) => handler(params),
        });
      }
    }
```

- [ ] **Step 3: Add new webhook routes to the onWebhook dispatcher**

In the `onWebhook()` function, add new cases to the endpoint key switch:

```typescript
    case WEBHOOK_KEYS.opsFirestore:
      assertSharedSecret(input, ctx);
      return handleFirestoreWebhook(input, ctx);

    case WEBHOOK_KEYS.opsStripe:
      assertSharedSecret(input, ctx);
      return handleStripeWebhook(input, ctx);

    case WEBHOOK_KEYS.opsSupport:
      assertSharedSecret(input, ctx);
      return handleSupportWebhook(input, ctx);
```

- [ ] **Step 4: Update manifest.ts with new webhook endpoints**

Add new webhook endpoint declarations to the manifest's `webhooks` array:

```typescript
    {
      key: "ops-firestore",
      displayName: "Firestore Ops Events",
      description: "Receives Firestore triggers for new signups, requests, and capture completions",
    },
    {
      key: "ops-stripe",
      displayName: "Stripe Ops Events",
      description: "Receives forwarded Stripe webhook events for payout and dispute triage",
    },
    {
      key: "ops-support",
      displayName: "Support Inbox",
      description: "Receives support tickets from email forward or contact form",
    },
```

And add new tool declarations to the manifest's `tools` array:

```typescript
    {
      name: "notion-read-work-queue",
      displayName: "Read Notion Work Queue",
      description: "Query Blueprint Work Queue items by system, priority, or lifecycle stage",
    },
    {
      name: "notion-write-work-queue",
      displayName: "Write Notion Work Queue",
      description: "Create or update items in Blueprint Work Queue",
    },
    {
      name: "notion-write-knowledge",
      displayName: "Write Notion Knowledge",
      description: "Create entries in Blueprint Knowledge database",
    },
    {
      name: "slack-post-digest",
      displayName: "Post Slack Digest",
      description: "Post formatted digest message to a Slack channel",
    },
```

- [ ] **Step 5: Build the plugin**

```bash
cd ops/paperclip/plugins/blueprint-automation && npm run build && cd ../../../..
```

Expected: Build succeeds with no errors.

- [ ] **Step 6: Commit**

```bash
git add ops/paperclip/plugins/blueprint-automation/src/worker.ts ops/paperclip/plugins/blueprint-automation/src/manifest.ts ops/paperclip/plugins/blueprint-automation/dist/
git commit -m "feat: wire Notion tools, Slack tools, and Ops webhook handlers into plugin worker

Extends the Blueprint Automation Plugin with:
- Notion read/write tools for Work Queue and Knowledge DB
- Slack digest posting tool
- Firestore, Stripe, and Support webhook endpoints
- Routes events to intake-agent, capture-qa-agent, finance-support-agent

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 8: Update Plugin Config and Add Secret References

**Files:**
- Modify: `ops/paperclip/blueprint-automation.config.json`

- [ ] **Step 1: Update config to include Ops and Growth department settings**

Add to `ops/paperclip/blueprint-automation.config.json`:

```json
{
  "companyName": "Blueprint Autonomous Operations",
  "githubOwner": "ognjhunt",
  "enableGitRepoScanning": true,
  "enableGithubPolling": true,
  "enableOutboundNotifications": true,
  "repoCatalog": [
    {
      "key": "webapp",
      "projectName": "blueprint-webapp",
      "githubRepo": "Blueprint-WebApp",
      "defaultBranch": "main",
      "implementationAgent": "webapp-codex",
      "reviewAgent": "webapp-claude"
    },
    {
      "key": "pipeline",
      "projectName": "blueprint-capture-pipeline",
      "githubRepo": "BlueprintCapturePipeline",
      "defaultBranch": "main",
      "implementationAgent": "pipeline-codex",
      "reviewAgent": "pipeline-claude"
    },
    {
      "key": "capture",
      "projectName": "blueprint-capture",
      "githubRepo": "BlueprintCapture",
      "defaultBranch": "main",
      "implementationAgent": "capture-codex",
      "reviewAgent": "capture-claude"
    }
  ],
  "opsDepartment": {
    "enabled": true,
    "agents": {
      "opsLead": "ops-lead",
      "intake": "intake-agent",
      "captureQa": "capture-qa-agent",
      "fieldOps": "field-ops-agent",
      "financeSupport": "finance-support-agent"
    }
  },
  "growthDepartment": {
    "enabled": true,
    "agents": {
      "growthLead": "growth-lead",
      "conversionOptimizer": "conversion-agent",
      "analytics": "analytics-agent",
      "marketIntel": "market-intel-agent"
    }
  },
  "secrets": {
    "notionApiToken": "NOTION_API_TOKEN",
    "slackOpsWebhookUrl": "SLACK_OPS_WEBHOOK_URL",
    "slackGrowthWebhookUrl": "SLACK_GROWTH_WEBHOOK_URL"
  }
}
```

- [ ] **Step 2: Update env example**

Add to `ops/paperclip/blueprint-paperclip.env.example`:

```bash
# Notion API (for agent Work Queue and Knowledge DB access)
NOTION_API_TOKEN=secret_...

# Slack Webhooks (for agent digests and alerts)
SLACK_OPS_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_GROWTH_WEBHOOK_URL=https://hooks.slack.com/services/...

# Web Search API (for Market Intel agent)
SEARCH_API_KEY=...
SEARCH_API_PROVIDER=tavily  # or serpapi, brave
```

- [ ] **Step 3: Commit**

```bash
git add ops/paperclip/blueprint-automation.config.json ops/paperclip/blueprint-paperclip.env.example
git commit -m "feat: update plugin config with Ops and Growth departments, add secret references

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 9: Run Verification and Re-import Company

- [ ] **Step 1: Verify Paperclip is running**

```bash
curl -s http://localhost:3100/health | head -20
```

Expected: Health check response (or instructions to start Paperclip if not running).

- [ ] **Step 2: Validate YAML one more time**

```bash
python3 -c "import yaml; d=yaml.safe_load(open('ops/paperclip/blueprint-company/.paperclip.yaml')); print(f'Agents: {len(d.get(\"agents\",{}))}'); print(f'Routines: {len(d.get(\"routines\",{}))}'); print(f'Tasks: {len(d.get(\"tasks\",{}))}')"
```

Expected:
```
Agents: 17
Routines: 21
Tasks: ~23
```

- [ ] **Step 3: Run the reconcile script to import updated config**

```bash
bash scripts/paperclip/reconcile-blueprint-paperclip-company.sh
```

Expected: Script completes, reports agent count matches expected.

- [ ] **Step 4: Verify new agents exist in Paperclip**

```bash
curl -s http://localhost:3100/api/companies | python3 -c "import sys,json; companies=json.load(sys.stdin); [print(f'  {a[\"name\"]}') for c in companies for a in c.get('agents',[])]"
```

Expected: All 17 agents listed.

- [ ] **Step 5: Rebuild and restart the plugin**

```bash
cd ops/paperclip/plugins/blueprint-automation && npm run build && cd ../../../..
```

Then restart the Paperclip service to pick up the new plugin build.

- [ ] **Step 6: Run the verification script**

```bash
bash scripts/paperclip/verify-blueprint-paperclip.sh
```

Expected: All checks pass.

- [ ] **Step 7: Run the smoke test**

```bash
bash scripts/paperclip/smoke-blueprint-paperclip-automation.sh
```

Expected: Smoke test passes, including webhook delivery tests.

- [ ] **Step 8: Commit any fixes**

If any verification or smoke test failures required fixes:

```bash
git add -A
git commit -m "fix: address verification and smoke test issues for autonomous org agents

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 10: Copy Plan to Other Repos and Final Verification

- [ ] **Step 1: Copy plan to BlueprintCapture and BlueprintCapturePipeline**

```bash
mkdir -p /Users/nijelhunt_1/workspace/BlueprintCapture/docs/superpowers/plans
cp docs/superpowers/plans/2026-03-28-autonomous-org-implementation.md /Users/nijelhunt_1/workspace/BlueprintCapture/docs/superpowers/plans/

mkdir -p /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/superpowers/plans
cp docs/superpowers/plans/2026-03-28-autonomous-org-implementation.md /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/docs/superpowers/plans/
```

- [ ] **Step 2: Verify all 3 repos have the full autonomous org file set**

```bash
for repo in Blueprint-WebApp BlueprintCapture BlueprintCapturePipeline; do
  echo "=== $repo ==="
  ls -la /Users/nijelhunt_1/workspace/$repo/AUTONOMOUS_ORG.md
  ls -la /Users/nijelhunt_1/workspace/$repo/docs/superpowers/specs/2026-03-28-autonomous-org-design.md
  ls -la /Users/nijelhunt_1/workspace/$repo/docs/superpowers/plans/2026-03-28-autonomous-org-implementation.md
done
```

Expected: All 9 files exist (3 files x 3 repos).

- [ ] **Step 3: Final commit in Blueprint-WebApp**

```bash
git add docs/superpowers/plans/2026-03-28-autonomous-org-implementation.md
git commit -m "docs: add autonomous org implementation plan

10-task plan covering skill files, Paperclip config, plugin extensions,
Notion integration, and verification for the 9 new autonomous agents.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
