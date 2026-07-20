# GPU Warm-Pool Economics (SCALE2-06) — 2026-07-20

Decision analysis for whether the four GPU Cloud Run services (`sam3-detect`,
`vip-inpaint`, `deepprivacy2-anonymize`, `video-to-world`) should keep
`min_instance_count = 0` (scale-to-zero, cold-start on every burst) or hold a
warm pool (`min_instance_count >= 1`, sustained idle GPU spend).

**Recommendation: keep scale-to-zero (min = 0) for all four services at
current volume.** The breakeven is roughly a sustained **20 invocations/hour
per service**; the beta capacity model runs at under **1 invocation/hour**.
The cheap win is not a warm pool — it is cutting the cold-start *tax* itself,
which this change does (in-process model caching + load-time instrumentation).

## Claim boundary (read first)

This analysis is **modeled, not measured**. It uses the repo's $2.5/GPU-hr
planning rate and an assumed ~3-minute cold start (container boot + CUDA init
+ model weight load). Two things must be sanity-checked against real metrics
before trusting the breakeven number, and only the repo owner has that
visibility:

1. **Actual invocation rate per service** (Cloud Run request count metrics).
   The model uses the beta capacity target of 300 captures/month.
2. **Actual cold-start duration per service.** This change adds a
   `gpu_model_load` structured log event (duration, cache hit/miss) to the
   in-process backends and a `gpu_backend_execution` event to the subprocess
   backends, so after one deploy the real numbers are queryable in Cloud
   Logging (`jsonPayload.blueprint_event="gpu_model_load"`).

## Inputs

| Input | Value | Source |
| --- | --- | --- |
| GPU planning rate | $2.5/GPU-hr | round-1 scaling audit planning rate; consistent with `gpu_render_providers.py` provider ceilings ($5/hr Vast cap is the hard ceiling, $2.5 is the planning midpoint) |
| Hours per month | 730 | calendar |
| Modeled cold start | ~3 min/invocation (0.05 h) | assumption pending `gpu_model_load` measurements: image pull + CUDA init (~1–2 min for multi-GiB GPU images) + weight load (SAM3 / DeepPrivacy2 / DepthAnything, ~0.5–1.5 min each) |
| Beta volume | 300 captures/month (~0.4/hr) | `docs/beta_capacity_cost_storage_model_2026-07-08.json` `beta_target.modeled_captures_per_month` |
| Invocations per capture per service | 1–3 | one privacy pass per lane; retries/multi-segment can add more |
| Spend review / hard-stop thresholds | $2,500 / $5,000 per month | `validate_beta_capacity_storage.py` `EXPECTED_COHORT_REVIEW_THRESHOLD_USD` / hard stop |

## The math

**Cost of one warm instance** (`min_instance_count = 1`, billed continuously):

```
730 h/month × $2.5/GPU-hr ≈ $1,825/month per service
```

**Cold-start tax** (what scale-to-zero costs instead):

```
tax($/month) = invocations/month × cold_start_hours × $2.5
```

At beta volume, worst case (300 captures × 3 invocations × 0.05 h × $2.5):
**≈ $112/month per service**, ≈ $150–450/month across all four services
depending on retry mix. Round-1's estimate of ~$150/month total is the p50.

**Breakeven** — a warm instance pays for itself when the tax it eliminates
exceeds its idle cost:

```
N* = 730 / cold_start_hours  invocations/month
   = 730 / 0.05 ≈ 14,600/month ≈ 20/hour sustained (per service)
```

Sensitivity: if measured cold start is 6 min, breakeven halves to ~10/hr; if
90 s, it doubles to ~40/hr. Even the most favorable plausible cold start
leaves current volume **more than an order of magnitude below breakeven**.

**Budget interaction:** one warm service ($1,825/month) alone consumes 73% of
the $2,500/month spend-review threshold; two warm services exceed it. Any
future `min_instances > 0` flip must therefore be paired with a threshold
review in `beta_capacity_cost_storage_model` — the Terraform variables'
descriptions say exactly this.

## Why "critical path" does not change the answer today

`sam3-detect` and `video-to-world` are on the critical path of every capture,
so they are the *first* candidates when volume grows — but latency, not cost,
is the only argument for warming them below breakeven, and the pipeline is an
asynchronous batch flow with hour-scale SLOs (Cloud Tasks dispatch, lane-level
resume). A 3-minute cold start is invisible against that envelope. If a
future interactive/live product path appears, revisit with a latency SLO
argument rather than a cost argument.

## What shipped instead (cold-start tax reduction)

1. **In-process model runtime cache** (`privacy_service_runtime.py`,
   `_timed_model_load` / `_MODEL_RUNTIME_CACHE`): SAM3 and DepthAnything
   runtimes were previously reloaded on *every request*, so even a warm
   instance would have paid the model load each time — a warm pool would have
   bought container boot only. Loaded runtimes now live for the instance's
   lifetime (`max_instance_request_concurrency = 1`, so no contention), which
   makes consecutive requests on a busy instance cheap **and** makes any
   future warm pool actually effective. Opt out with
   `PRIVACY_RUNNER_MODEL_CACHE=0`. The cache is keyed by (runner kind,
   weights identity) and is backend-agnostic — no model-specific behavior
   leaks out of the loader, keeping world-model backends swappable.
2. **Instrumentation**: `gpu_model_load` events (in-process backends:
   sam3, depth_anything) and `gpu_backend_execution` events (subprocess
   backends: deepprivacy2; video-to-world already logs stage durations)
   with wall-clock durations, so the assumed 3-minute cold start becomes a
   measured number.
3. **Terraform warm-pool controls, default off**: per-service
   `privacy_*_min_instances` / `video_to_world_min_instances` variables
   (default 0, capped at 2, and clamped to never exceed the per-service max)
   so the owner can flip a single service without editing resource blocks.
   Defaults are pinned by `validate_beta_capacity_storage.py` and
   `tests/test_deploy_systemd_contract.py` — changing them requires updating
   the capacity model in the same change, which is the intended review gate.

## Revisit triggers

Re-run this analysis (with measured `gpu_model_load` durations and real
request-count metrics) when any of:

- a service's sustained invocation rate crosses ~10/hour (half the modeled
  breakeven — margin for the cold-start sensitivity above);
- a latency SLO tied to a live/interactive product path appears;
- GPU pricing for Cloud Run changes materially from the $2.5/GPU-hr planning
  rate;
- the monthly cold-start tax estimated from `gpu_model_load` events exceeds
  ~$500/month (at that point the measurement, not the model, drives the call).
