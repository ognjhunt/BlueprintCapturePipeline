# Capacity & Cost Model — 100-User Industrial Beta (2026-07-08)

**Status: MODELED ESTIMATE, NOT MEASURED.** This document is the engineering
scaffolding requested by audit finding **R043 (P1): "No load/soak test, capacity
model, or cost-per-capture model exists in any repo."** Every number below is a
*calculated projection from stated assumptions*, not a measurement from a live
load test. The remaining human/infra action — running the load/soak harness at
target concurrency against staging and replacing the assumption values with
observed values — is called out in the last section. Do **not** cite these
numbers as proven throughput, proven cost, or a validated capacity ceiling.

Related repo context this model is grounded in:

- `docs/PRE_BETA_LAUNCH_GAP_AUDIT_2026-07-08.md` — R043's parent audit; frames the
  "~100 external users, industrial-first (warehouses & factories)" beta, notes
  **concurrency is capped at ~10**, and finding **#54 (P1)** — *no platform-wide
  cumulative spend / GPU-concurrency ceiling* — which this model's guardrail
  section sizes.
- `docs/PROVIDER_RELIABILITY_MANIFEST.md` — the paid GPU providers in use
  (RunPod, Vast, Lambda, DigitalOcean GPU Droplets).
- `docs/FIRST_GPU_E2E_RUNBOOK.md` — GPU classes actually targeted
  (RTX 4090 24GB, L40S 48GB, RTX 6000 Ada 48GB).
- The intake path this model sizes is exercised by the runnable harness at
  `Blueprint-WebApp/scripts/loadtest/intake-loadtest.k6.js`.

The load/soak harness (item 2 of R043) targets the WebApp intake endpoint
`POST /api/robot-eval/job-requests/` and status read
`GET /api/robot-eval/job-requests/:jobId/status`, which is where the numbers in
the "Peak concurrency" section must be validated.

---

## 1. Scope & framing

- **Cohort:** 100 external beta users (capturers), per the parent audit's goal framing.
- **Site type:** industrial — warehouses and factories. This matters because industrial
  walkthroughs are **large captures** (multi-GB), unlike a 15 sq m kitchen fixture.
  The audit explicitly notes a warehouse may be ~50,000 sq m vs a 15 sq m kitchen, so
  per-capture size and per-capture GPU-hours have a wide, site-extent-driven spread.
- **Grade:** sim/review-grade Task Evaluation Runs + Post-Training Data Packages
  (no live-robot / physical-readiness / world-model-fidelity claims), matching the
  audit's recommended honest beta posture.

All assumption cells below are **tunable** — they are the knobs the team should replace
with measured values after the first real staging load test.

---

## 2. Assumptions (baseline + range) — TUNE THESE

| # | Assumption | Symbol | Baseline | Low | High | Basis / note |
|---|-----------|--------|----------|-----|------|--------------|
| A1 | Beta users | `N` | 100 | 100 | 100 | Fixed by beta scope |
| A2 | Captures per user per week | `c_uw` | 3 | 1 | 5 | Industrial capturer doing repeat site visits |
| A3 | Avg raw media per capture (GB) | `s_cap` | 5 | 2 | 8 | Industrial multi-GB walkthrough (4K video + depth/LiDAR frames) |
| A4 | Weeks per month | `w_m` | 4.33 | 4.33 | 4.33 | Calendar constant |
| A5 | Pipeline stages per capture | — | 4 | 4 | 4 | privacy-prep, geometry, eval, packaging |
| A6 | Eval/render runs per capture | `r_eval` | 4 | 1 | 8 | tasks × scenarios per site package |
| A7 | GPU-hr per capture — geometry | `g_geo` | 1.00 | 0.50 | 3.00 | Scene reconstruction; scales with site extent (5–10× spread) |
| A8 | GPU-hr per capture — privacy prep | `g_priv` | 0.25 | 0.10 | 0.75 | Person/badge/screen/signage redaction over frames |
| A9 | GPU-hr per capture — eval + render | `g_eval` | 0.75 | 0.25 | 2.00 | `r_eval` runs; MuJoCo-first, optional RTX render (per-job hard timeout 120s) |
| A10 | **Total GPU-hr per capture** | `g_cap` | **2.00** | 0.85 | 5.75 | `g_geo + g_priv + g_eval` |
| A11 | Blended GPU $/hr | `p_gpu` | 0.90 | 0.44 | 1.50 | RunPod RTX 4090 community ≈ $0.44 → L40S / RTX 6000 Ada secure ≈ $0.79–0.89 → Lambda/DO ≈ $1.00–1.50 |
| A12 | Stored footprint multiplier (÷ raw) | `m_store` | 2.5 | 1.5 | 4.0 | raw + privacy-safe derived + geometry assets + packaged bundle |
| A13 | Storage $/GB-month | `p_store` | 0.020 | 0.010 | 0.026 | GCS/Firebase Storage standard regional ($0.020); Nearline $0.010; multi-region $0.026 |
| A14 | Packaged bundle size (GB) | `s_pkg` | 3 | 1 | 6 | Privacy-safe deliverable subset |
| A15 | Downloads per capture | `d_cap` | 1.5 | 1 | 3 | Buyer + re-downloads of signed-URL bundle |
| A16 | Egress $/GB | `p_egr` | 0.12 | 0.08 | 0.12 | GCP internet egress, first tier |
| A17 | Fixed platform $/month | `fix` | 500 | 300 | 900 | Render web/worker + Redis + Firestore baseline (excludes GPU/storage/egress) |
| A18 | Effective capturer uplink | `up` | 20 Mbps | 10 Mbps | 50 Mbps | Industrial sites often have constrained connectivity |
| A19 | Peak-hour burst factor | `burst` | 3.0 | 1.5 | 5.0 | Clustering (shift ends, launch-city cohort captures same day) |
| A20 | Pipeline wall-clock per capture (hr) | `wall` | 3.0 | 1.5 | 6.0 | Queue + CPU preflight + serial GPU stages (≥ `g_cap` GPU-hr) |

Provider $/hr (A11) and storage/egress (A13/A16) are list-style estimates as of early 2026;
they must be reconciled against the actual billing accounts before the guardrail budgets are set.

---

## 3. Formula table

| Quantity | Formula | Baseline result |
|----------|---------|-----------------|
| Captures / week | `N · c_uw` | 300 |
| Captures / month | `N · c_uw · w_m` | ≈ 1,300 |
| Ingest GB / week | `N · c_uw · s_cap` | 1,500 GB (1.5 TB) |
| Ingest GB / month | `N · c_uw · s_cap · w_m` | ≈ 6,495 GB (≈ 6.5 TB) |
| GPU-hr / month | `captures_month · g_cap` | 2,600 GPU-hr |
| **$ GPU / capture** | `g_cap · p_gpu` | **$1.80** |
| **$ egress / capture** | `s_pkg · d_cap · p_egr` | **$0.54** |
| **$ storage / capture (first month)** | `s_cap · m_store · p_store` | **$0.25** |
| **$ per capture (marginal, all-in)** | `g_cap·p_gpu + s_pkg·d_cap·p_egr + s_cap·m_store·p_store` | **≈ $2.60** |
| $ GPU / month | `captures_month · g_cap · p_gpu` | ≈ $2,340 |
| $ egress / month | `captures_month · s_pkg · d_cap · p_egr` | ≈ $702 |
| Stored GB added / month | `captures_month · s_cap · m_store` | ≈ 16,250 GB (16.25 TB) |
| $ storage / month (month 1) | `stored_GB_added · p_store` | ≈ $325 |
| **$ per 100-user-month (month 1)** | `$GPU + $egress + $storage_m1 + fix` | **≈ $3,900** |
| $ per 100-user-month (steady, ~month 3) | storage accrues ~3× if fully retained | ≈ $4,500 |
| Upload seconds / capture | `s_cap·8 / up` (GB→Gb ÷ Mbps) | ≈ 2,000 s (≈ 33 min) |
| Peak captures / hr | `(N·c_uw / (5·8)) · burst` | ≈ 22.5 /hr |
| **Peak concurrent uploads** | `peak_arrivals_hr · (upload_s / 3600)` (Little's Law) | **≈ 13** |
| Avg concurrent pipeline jobs | `(captures_month / 730) · wall` | ≈ 5.3 |
| **Peak concurrent pipeline/GPU jobs** | `avg_concurrent · burst` | **≈ 16** |

---

## 4. Headline numbers (baseline)

| Metric | Baseline | Band (low → high assumptions) |
|--------|----------|-------------------------------|
| **$ per capture (all-in marginal)** | **≈ $2.60** | ≈ $1.10 → ≈ $9.50 |
| **$ per 100-user-month** | **≈ $3,900 (mo 1)** → ~$4,500 (steady) | ≈ $2,000 → ≈ $9,000+ |
| **Peak concurrent uploads** | **≈ 13** | ≈ 6 → ≈ 30 |
| **Peak concurrent pipeline/GPU jobs** | **≈ 16** | ≈ 8 → ≈ 40 |
| Ingest volume / month (100 users) | ≈ 6.5 TB | 1.7 TB → 17 TB |
| GPU-hours / month | 2,600 | ~1,100 → ~7,500 |

> **Capacity flag:** modeled peak (~13 concurrent uploads, ~16 concurrent pipeline/GPU jobs)
> **exceeds the current ~10 concurrency ceiling** noted in the parent audit. Under baseline
> assumptions the 100-user beta does not fit the current envelope at peak. This is the single
> most important thing the real load test (below) must confirm or refute before beta.

---

## 5. Sensitivity — what dominates cost

Decomposition of the ~$2.60 marginal cost per capture:

| Component | $ / capture | Share |
|-----------|-------------|-------|
| GPU compute | $1.80 | ≈ 69% |
| Egress | $0.54 | ≈ 21% |
| Storage (first month) | $0.25 | ≈ 10% |

- **GPU compute dominates (~70%).** The two largest swing factors are (a) **geometry
  GPU-hours per capture** (A7) — driven by industrial site extent, which the audit shows
  spans ~5–10× (15 sq m kitchen vs 50,000 sq m warehouse), and (b) **blended GPU $/hr**
  (A11) — a ~3.4× spread across the provider mix. A 2× error in either roughly doubles
  program cost. **Prioritize measuring these two first.**
- **Egress (~20%)** scales with buyer download behavior and bundle size; its weight grows
  as the buyer base grows relative to the capturer base.
- **Storage (~10% monthly) is small per month but is the only line that compounds** — with
  no retention/lifecycle policy it grows ~16 TB/month unbounded (ties to the audit's
  "no storage retention/lifecycle" gap). A lifecycle rule (Standard → Nearline at 30 days,
  or delete-on-expiry) flattens the slope.

---

## 6. Guardrail budgets (sizes the missing spend ceiling — audit finding #54)

Finding #54 (P1): the spend gate is a **per-run manual boolean**; budgets are per-job only;
there is **no cross-run cumulative spend or GPU-concurrency ceiling**. This model sizes the
fail-closed ceilings that gap needs:

| Guardrail | Modeled baseline | Recommended fail-closed ceiling | Rationale |
|-----------|------------------|--------------------------------|-----------|
| Daily GPU + egress spend | ≈ $101 / day | **$150 / day** (alert at 70% = $105) | ≈ 1.5× baseline; catches runaway before it compounds |
| Monthly GPU + egress spend | ≈ $3,042 / month | **$6,000 / month** | ≈ 2× baseline headroom for burst months |
| Max concurrent GPU pods | ≈ 16 at peak | **20 pods** | Headroom over modeled peak; also caps the GPU-DoS / cost-amplification exposure the audit flags on publicly-invokable GPU endpoints |
| Per-capture cost alarm | ≈ $2.60 | **flag any capture > $8** (≈ 3×) | Catches runaway geometry on an oversized site |
| Storage growth | ≈ 16 TB / month added | **lifecycle: Standard → Nearline @ 30d or delete-on-expiry** | Prevents unbounded compounding storage |

Implementation note (not built here): these belong in a persistent cumulative-spend / active-pod
ledger consulted by `provider_reliability_manifest.build_pre_spend_preflight` before each launch,
fail-closing new launches once a ceiling is crossed. That is the fix for finding #54 and is out of
scope for R043's model+harness scaffolding.

---

## 7. What is proven vs. what remains (R043 closure boundary)

**Delivered by this task (scaffolding):**

- This quantitative capacity + cost model with explicit, tunable assumptions.
- A runnable load/soak harness for the intake path:
  `Blueprint-WebApp/scripts/loadtest/intake-loadtest.k6.js` (+ `README.md`), defaulting to a
  safe dry mode so it cannot accidentally hammer production.

**NOT proven — the remaining human/infra action to actually close R043:**

1. Stand up (or point at) a **staging** WebApp + pipeline environment with a real Firebase
   auth token, isolated from production data.
2. Run the k6 harness in `submit` mode against staging at target concurrency
   (ramp to ~16–20 concurrent, then a soak), capturing p95 latency and error rate for the
   intake endpoint and the pipeline queue depth behind it.
3. Confirm — or refute — that the intake path and pipeline sustain **≥ 16 concurrent** jobs
   without breaching the p95/error thresholds, given the current ~10 concurrency cap.
4. Replace the assumption cells in §2 (especially A7, A10, A11 — geometry GPU-hours and
   blended GPU $/hr) with **measured** values, and re-derive §4.
5. Wire the §6 ceilings into the pre-spend preflight (finding #54).

Until steps 1–4 are done, treat every figure here as a **planning estimate**, not a
capacity guarantee.
