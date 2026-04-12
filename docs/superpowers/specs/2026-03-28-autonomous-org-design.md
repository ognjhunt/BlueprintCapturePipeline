# Blueprint Autonomous Organization Design Spec

**Date:** 2026-03-28
**Status:** Approved
**Scope:** Full autonomous company — ops, growth, research, engineering

## Problem

Blueprint is in active alpha with 7 repos, 10+ operational queues, and significant human-in-the-loop overhead. The product lifecycle (capture -> pipeline -> webapp -> buyer) requires coordination across repos, external services, and manual triage at every stage. Scaling to beta and beyond requires reducing this human overhead without sacrificing quality, rights compliance, or buyer trust.

## Solution

Build a tiered autonomous organization using Paperclip as the orchestration layer, with agents modeled as employees in a formal org chart. Each operational role becomes a persistent agent with defined triggers, inputs, outputs, human gates, and a progressive autonomy graduation path.

Notion Blueprint Hub serves as the source of truth for both agents and humans. Autoresearch-pattern experiment loops (adapted from Karpathy) drive continuous optimization of conversion, growth, and market intelligence.

## Architecture: Tiered Org Chart (Approach B)

### Design Decisions

1. **Tiered hierarchy over flat pool or event mesh** — builds on existing Paperclip CEO/CTO agents, provides natural escalation paths, and maps cleanly to progressive autonomy graduation.
2. **Progressive autonomy over binary human/auto** — every agent starts with human veto gates and graduates based on demonstrated reliability.
3. **Autoresearch pattern for optimization** — the hypothesis->change->measure->iterate loop from Karpathy's autoresearch, adapted for web conversion and market intelligence (no GPU needed).
4. **Hybrid infrastructure** — local Paperclip hub for orchestration, cloud burst for heavy compute, external APIs for data.
5. **Notion as source of truth** — all agent state, knowledge, and work items flow through Blueprint Hub databases.

### Org Chart

```
                         CEO Agent
                         (Claude)
                            |
              +-------------+-------------+
              |             |             |
           CTO Agent    Ops Lead     Growth Lead
           (Claude)     (Claude)      (Claude)
              |             |             |
    +---------+----+   +----+----+   +----+----+
    |    |    |    |   |    |    |   |    |    |
  Impl Impl Impl Revw Wait QA  Fld  Conv Anly Mkt
  WA   PL   CA   x3  Intk     Ops  Opt       Intel
  Cdx  Cdx  Cdx  Cld  Cld Cld Cld  Cld  Cld  Cld
```

**Total: 17 roles** (8 existing + 9 new)

## Role Registry

### Executive Layer

#### CEO (`ceo`) — EXISTS
- **Department:** Executive
- **Model:** Claude
- **Schedule:** Daily 8am ET (company-wide priority review)
- **Purpose:** Sets company priorities, reviews cross-department status, handles escalations, interfaces with human founder.
- **Changes needed:** Expand routine to include Ops Lead and Growth Lead reports. Add growth metrics to daily review context.

#### CTO (`cto`) — EXISTS
- **Department:** Executive
- **Model:** Claude
- **Schedule:** 8:30am + 2pm ET weekdays (cross-repo triage)
- **Purpose:** Technical decisions, cross-repo coordination, architecture review.
- **Changes needed:** Route non-technical ops issues to Ops Lead instead of handling directly.

### Engineering Department — ALL EXIST

| Agent | Model | Trigger | Repo |
|-------|-------|---------|------|
| `webapp-codex` | Codex | Issue assignment | Blueprint-WebApp |
| `webapp-claude` | Claude | PR/issue events | Blueprint-WebApp |
| `pipeline-codex` | Codex | Issue assignment | BlueprintCapturePipeline |
| `pipeline-claude` | Claude | PR/issue events | BlueprintCapturePipeline |
| `capture-codex` | Codex | Issue assignment | BlueprintCapture |
| `capture-claude` | Claude | PR/issue events | BlueprintCapture |

**Changes needed:** Engineering agents should accept work items created by Ops and Growth agents, not just GitHub events.

### Ops Department — ALL NEW

#### Ops Lead (`ops-lead`)
- **Department:** Ops
- **Reports to:** CEO
- **Model:** Claude
- **Schedule:** 8:30am + 2:30pm ET weekdays
- **Purpose:** Coordinates product operations. Routes work between intake, QA, scheduling, and finance agents. Produces daily ops summary.
- **Inputs:** All Firestore ops queues, Notion Work Queue, specialist agent reports.
- **Outputs:** Daily ops digest (Notion + Slack), priority assignments (Paperclip issues), escalations to CEO.
- **Human gates:** None (coordination role).
- **External needs:** Firestore read, Notion API, Slack webhook.
- **Graduation:**
  - Phase 1: Read-only monitoring + draft summaries for human review
  - Phase 2: Auto-route P2/P3 work without approval
  - Phase 3: Auto-route all work; human reviews weekly summary only

#### Waitlist & Intake Agent (`intake-agent`)
- **Department:** Ops
- **Reports to:** Ops Lead
- **Model:** Claude
- **Triggers:** Webhook (new signup/request) + hourly queue scan
- **Purpose:** Processes capturer applications and buyer inbound requests. Classifies, scores, detects missing info, drafts responses.
- **Inputs:** Waitlist collection, inbound request collection, capturer device metadata, buyer request fields.
- **Outputs:** Classification + priority score, draft invite/reject/follow-up messages, missing-info flags, Notion Work Queue updates.
- **Human gates:** All outbound messages require approval in Phase 1.
- **External needs:** Firestore read/write, SendGrid or email API, Notion API.
- **Graduation:**
  - Phase 1: Classify + score only; human sends all messages
  - Phase 2: Auto-send follow-up questions; human approves invite/reject
  - Phase 3: Auto-approve low-risk invites (device match + market fit > threshold)

#### Capture QA Agent (`capture-qa-agent`)
- **Department:** Ops
- **Reports to:** Ops Lead
- **Model:** Claude
- **Triggers:** Pipeline completion webhook + daily stalled-capture scan
- **Purpose:** Reviews pipeline outputs for quality, completeness, privacy compliance. Flags recapture needs. Drafts payout recommendations.
- **Inputs:** Pipeline artifacts (qualification_summary, capture_quality_summary, rights_and_compliance_summary, gemini_capture_fidelity_review), raw capture metadata.
- **Outputs:** QA pass/fail verdict, recapture requests, payout recommendation drafts, quality trends (weekly).
- **Human gates:** All payout approvals. Recapture decisions in Phase 1.
- **External needs:** GCS read access, Firestore read/write, Notion API.
- **Graduation:**
  - Phase 1: Review + flag only; human makes all QA decisions
  - Phase 2: Auto-pass captures above quality threshold; human reviews borderline + fails
  - Phase 3: Auto-approve payouts under $ threshold

#### Scheduling & Field Ops Agent (`field-ops-agent`)
- **Department:** Ops
- **Reports to:** Ops Lead
- **Model:** Claude
- **Triggers:** Intake qualification event, recapture request, daily 7am calendar review
- **Purpose:** Capture scheduling, capturer assignment, calendar management, timezone normalization, reminders.
- **Inputs:** Qualified requests, capturer roster + availability, site metadata, Google Calendar.
- **Outputs:** Calendar invite proposals, capturer assignment recommendations, reminder sequences.
- **Human gates:** All calendar sends in Phase 1.
- **External needs:** Google Calendar API, Google Maps API, Firestore, Notion API.
- **Graduation:**
  - Phase 1: Proposes schedule; human confirms and sends
  - Phase 2: Auto-schedules when capturer confirms; human reviews conflicts
  - Phase 3: Fully autonomous; human handles access/permission issues only

#### Finance & Support Agent (`finance-support-agent`)
- **Department:** Ops
- **Reports to:** Ops Lead
- **Model:** Claude
- **Triggers:** Stripe webhooks, support inbox, daily ledger reconciliation
- **Purpose:** Stripe health monitoring, payout triage, support inbox handling, response drafting.
- **Inputs:** Stripe events, support tickets, Firestore payout records, buyer/capturer accounts.
- **Outputs:** Payout issue triage, support response drafts, ledger discrepancy reports, Stripe health summary.
- **Human gates:** All financial actions. All support responses in Phase 1.
- **External needs:** Stripe API, email/support platform API, Firestore, Notion API.
- **Graduation:**
  - Phase 1: Monitors + drafts only; human approves everything
  - Phase 2: Auto-sends template support responses; human approves financial actions
  - Phase 3: Auto-retries failed payouts under $ threshold; human approves disputes/refunds

### Growth Department — ALL NEW

#### Growth Lead (`growth-lead`)
- **Department:** Growth
- **Reports to:** CEO
- **Model:** Claude
- **Schedule:** Daily 9am ET + weekly Monday 10am ET
- **Purpose:** Coordinates acquisition, conversion, retention. Sets experiment priorities. Synthesizes analytics and market intel.
- **Inputs:** Analytics reports, experiment results, market intel, Notion Work Queue growth items.
- **Outputs:** Weekly growth summary, experiment priority queue, research briefs, funnel dashboard updates.
- **Human gates:** None (coordination role).
- **External needs:** Notion API, Slack webhook, analytics platform read.
- **Graduation:**
  - Phase 1: Reports + recommends; human sets final experiment queue
  - Phase 2: Auto-prioritizes experiments; human reviews weekly
  - Phase 3: Fully autonomous; human intervenes on budget/brand only

#### Conversion Optimizer (`conversion-agent`)
- **Department:** Growth
- **Reports to:** Growth Lead
- **Model:** Claude
- **Triggers:** Weekly experiment cycle (Monday) + Growth Lead assignment + measurement period completion
- **Purpose:** Autoresearch-pattern experiment loop on webapp and capture app. Tests CTAs, onboarding flows, signup copy, pricing layout.
- **Autoresearch loop:**
  1. Read `program.md` for current focus
  2. Analyze funnel data + page structure
  3. Propose change (copy, CTA, form, layout)
  4. Human approval gate -> deploy
  5. Monitor metrics (24-72hrs)
  6. Evaluate: keep improvement or revert regression
  7. Log result, loop
- **Inputs:** Analytics data, page source code, experiment history, `program.md` steering file.
- **Outputs:** Experiment proposals, code change PRs, result reports, experiment history log.
- **Human gates:** All code deploys in Phase 1.
- **External needs:** Analytics platform API, browser automation (gstack), Git access.
- **Infrastructure:** No GPU. Runs on local Paperclip hub.
- **Graduation:**
  - Phase 1: Proposes + writes PRs; human reviews and deploys everything
  - Phase 2: Auto-deploys copy/CTA tweaks behind feature flags; human approves structural changes
  - Phase 3: Auto-deploys all behind feature flags with 48hr auto-rollback on metric regression

#### Analytics Agent (`analytics-agent`)
- **Department:** Growth
- **Reports to:** Growth Lead
- **Model:** Claude
- **Schedule:** Daily 6am ET + weekly Sunday 11pm ET
- **Purpose:** Metrics aggregation, anomaly detection, reporting. Answers ad-hoc metric queries from other agents.
- **Inputs:** Analytics platform, Stripe revenue, Firestore counts, GitHub traffic, marketing channels.
- **Outputs:** Daily metrics snapshot, weekly growth report, anomaly alerts, on-demand metric answers.
- **Key metrics:** Visitor->signup, signup->first-action, request->qualified->purchased, capture->QA->listed->sold, MRR, support volume, page engagement.
- **Human gates:** None (reporting role).
- **External needs:** Analytics platform API, Stripe API (read), Firestore read, Notion API, Slack webhook.
- **Infrastructure:** No GPU. Lightweight. Local hub or scheduled cloud function.
- **Graduation:**
  - Phase 1: Reports; human validates accuracy for 2 weeks
  - Phase 2: Trusted for daily reporting; human spot-checks weekly
  - Phase 3: Fully autonomous; other agents query directly

#### Market Intelligence Agent (`market-intel-agent`)
- **Department:** Growth
- **Reports to:** Growth Lead
- **Model:** Claude
- **Schedule:** Daily 7am ET + weekly Friday 3pm ET deep synthesis
- **Purpose:** Autoresearch-pattern agent for business intelligence. Researches competitors, market trends, papers, pricing, partnerships, regulations.
- **Autoresearch loop:**
  1. Read `program.md` for research focus areas
  2. Structured web research cycle: scan -> extract signals -> synthesize
  3. Score findings by relevance + urgency
  4. Produce research digest with recommendations
  5. Update context, loop
- **Research domains:** Competitors, technology (papers/techniques), market (adoption/funding), regulatory (privacy/safety).
- **Inputs:** Web sources, ArXiv, `program.md`, previous digests.
- **Outputs:** Daily signal digest, weekly deep synthesis, ad-hoc answers, competitor tracker updates.
- **Human gates:** None (research/reporting role).
- **External needs:** Web search API, ArXiv API, Notion API, Slack webhook.
- **Infrastructure:** No GPU for daily. Optional cloud GPU burst for deep paper analysis batches.
- **Graduation:**
  - Phase 1: Digests; human evaluates relevance for first month
  - Phase 2: Trusted for daily digests; human reviews weekly synthesis
  - Phase 3: Fully autonomous; directly updates strategy docs

## Infrastructure

### Local Paperclip Hub (Your Mac)
- Paperclip server at localhost:3100
- All 17 agents run here (lightweight — mostly LLM API calls + data reads)
- Cloudflare Tunnel for public webhook URL
- LaunchAgent keeps Paperclip alive (already scripted)
- Blueprint Automation Plugin handles webhook intake

### Cloud Burst (On-Demand)
- RunPod or Lambda Labs for GPU when needed
- Used by: Market Intel (deep paper batches) — P2 priority
- Estimated cost: $0.50-2/hr, used < 10hrs/month

### External APIs to Provision

| Service | Agents | Priority | Cost |
|---------|--------|----------|------|
| Analytics (PostHog/GA4) | Analytics, Conversion, Growth Lead | P0 | Free tier |
| Web Search API (SerpAPI/Brave/Tavily) | Market Intel | P1 | ~$50/mo |
| Slack Incoming Webhook | All leads + CEO | P1 | Free |
| SendGrid / Email API | Intake, Finance/Support | P1 | Free tier |
| Notion API Token | All agents | P0 | Free (already partial) |
| Cloudflare Tunnel | Plugin webhook intake | P0 | Free (already scripted) |
| Cloud GPU (RunPod) | Market Intel | P2 | ~$5-20/mo |

### Already In Place
- Firebase/Firestore (all ops data)
- Stripe API (payments)
- Google Calendar/Maps/Sheets APIs
- GitHub API + webhooks
- Paperclip server + LaunchAgent
- Cloudflare Tunnel scripts
- Anthropic Claude API
- OpenAI/Codex API

## Progressive Autonomy Framework

### Phase Definitions

**Phase 1 — Supervised:** Agent produces outputs (classifications, drafts, recommendations) but a human must approve every action that affects external state (sending messages, approving payouts, deploying code, scheduling events). This is the default starting state for all new agents.

**Phase 2 — Semi-autonomous:** Agent can autonomously execute low-risk, reversible actions within defined boundaries. Human approves high-risk, irreversible, or above-threshold actions. Boundaries are specific to each role.

**Phase 3 — Autonomous:** Agent operates independently. Human reviews periodic summaries and intervenes only on exceptions, policy changes, or escalations from the agent itself.

### Graduation Criteria Template

An agent graduates from Phase N to Phase N+1 when:
1. **Track record:** Agent has operated at Phase N for at least 2 weeks (Phase 1->2) or 1 month (Phase 2->3)
2. **Accuracy:** Human override rate is below 10% for 2 consecutive weeks
3. **No incidents:** No escalation caused by agent error in the graduation period
4. **Human sign-off:** Founder explicitly promotes the agent

### Permanent Human Gates (Never Graduate)

These actions always require human approval regardless of agent phase:
- Payout release above configured threshold
- Rights, privacy, or consent signoff
- Legal or compliance decisions
- Final public release of sensitive previews
- Brand/positioning changes
- Budget allocation changes
- Agent promotion decisions

## Autoresearch Pattern

Adapted from Karpathy's autoresearch for non-ML experiment loops.

### Core Loop
```
while True:
    focus = read("program.md")        # human-steerable priorities
    context = gather(focus)            # data, metrics, sources
    hypothesis = analyze(context)      # what to try/investigate
    action = propose(hypothesis)       # change, research query, etc.
    if requires_approval(action):
        wait_for_human()
    result = execute_and_measure(action)
    log(result)                        # immutable experiment history
    update_context(result)             # learn for next cycle
```

### program.md Steering File Format
```markdown
# [Agent Name] Research Focus

## Current Priority
[What the agent should focus on this cycle]

## Constraints
[What to avoid, budget limits, off-limits areas]

## Success Metrics
[How to evaluate whether findings/experiments are valuable]

## Recent Context
[Key learnings from last cycle that should inform this one]
```

Updated by Growth Lead (for Conversion + Market Intel) or CEO (for ad-hoc overrides).

## Notion Sync Protocol

### Source of Truth Rules
- **Agent definitions and org chart:** Repo (`AUTONOMOUS_ORG.md`) is canonical; synced to Notion
- **Work items and task state:** Notion Work Queue is canonical; agents read/write via API
- **Knowledge and research outputs:** Notion Knowledge DB is canonical
- **Skill file content:** Repo skill files are canonical; Notion Skills DB tracks metadata + links
- **Experiment history:** Notion Knowledge DB is canonical
- **Metrics and reports:** Notion is canonical

### Sync Cadence
- Agent definitions: Updated in repo, synced to Notion on change
- Work items: Real-time via Notion API
- Reports and digests: Written to Notion on agent's schedule
- Skill metadata: Updated when skill files change

## Deliverables

1. `AUTONOMOUS_ORG.md` — guide file in all 3 main repos (WebApp, Capture, Pipeline)
2. Notion Hub updates — fix stale capture-first messaging
3. Notion Skills DB entries — one per new agent role (9 entries)
4. Notion Knowledge DB entry — this design spec as Architecture type

## Stale Content Fixes

The following Notion Hub content is out of date with the current capture-first strategy:

1. Hub callout: "qualification-first site readiness platform" -> "capture-first, world-model-product-first platform for real-world robotics data"
2. Stack callout: "Qualification first. The product center is a readiness decision" -> "Capture first. The product center is site-specific world models and hosted access from real-site captures"
3. Pipeline card: "Qualification engine / Evidence -> readiness decision -> handoff record" -> "Packaging engine / Evidence -> site-specific world-model products + hosted artifacts"
4. Product Stack ordering: Rewrite to match WORLD_MODEL_STRATEGY_CONTEXT.md build priorities
5. Add "Autonomous Org" navigation section linking to agent profiles
