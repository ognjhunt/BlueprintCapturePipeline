# Automatic beta policy screening

For the free beta, a completed customer scene can run a configurable set of
candidate policies under one persistent **$15 per site** screening allowance.
The initial catalog describes ten researched candidates. Only pi05-DROID and
GR00T N1.7-DROID have default executable bindings; the others report their
specific availability, licensing, adapter or credential blockers. Catalog
entries alone never authorize a run.

## What RoboLab does

We use NVIDIA RoboLab's unchanged `robolab.eval.run_episode` and
`InferenceClient` implementations, pinned to
`b6cfcf6ccd2113c4cd1216d35b33024a7d5691da`. Reviewed upstream sources and license
are sealed into the provider bundle, so deployment does not fetch a moving
branch. The native bundle also includes the upstream loop's small ImageIO
dependency.

RoboLab drives each policy request and simulator step. Our environment bridge
supplies the customer's reconstructed scene, robot and cameras, exact policy
input conversion, scientific reset, action checks, evidence recording and
independent task score. We retain those pieces because they describe the
customer's task; RoboLab's benchmark scenes and benchmark success conditions
do not describe it. Its rigid-object scene importer also does not replace
Blueprint's articulated scene construction. Policy servers receive the
observations they require, not the scorer's privileged state.

The same upstream loop is exercised in hermetic tests for rigid, articulated
and alternative robot episodes. These tests establish the orchestration and
evidence contract. A real GPU execution must still establish compatibility
with the installed Isaac runtime; an import or runtime failure stays visible
as a diagnostic failure and cannot become a successful evaluation.

## Automatic path

1. Existing scene preparation completes and retains its robot-free receipt.
2. The existing scene-packet worker compiles the reusable scene half.
3. `blueprint-beta-policy-screening.timer` scans completed packet requests.
   Current website task identity and recording consent must still match.
4. The planner matches catalog candidates to independently validated runtime
   bindings, robot/camera/action requirements, and bounded price quotes. Every
   selected candidate receives the same ten deterministic screening cells.
5. WebApp must first accept the queued publication against its current source
   records. Each worker pass renews that check before starting new work; failed
   delivery or a source refusal cannot reuse an old acknowledgment to spend.
   In-flight attempts continue reconciliation even when delivery is unavailable.
   The durable coordinator reserves the full candidate cost ceiling before
   launching the canonical `paid_resource_allocator gpu-canary` path. Default
   candidate limits are four hours and $4, using the registry's actual hourly
   ceiling; the provider gate also checks startup, transfer, storage and
   teardown bounds. Unknown prices are deferred.
6. The stock RoboLab loop executes the job. Independent simulator state
   determines episode success. The coordinator validates the returned scene,
   candidate, runtime, scenario and score bindings before reporting counts.
7. A durable outbox sends signed publications to WebApp's existing internal
   Pipeline API. Its consumer independently binds results to the current job,
   capture and completed scene revision. Existing site and admin pages show
   every candidate, result, blocker and budget.

The stable site key is `site-<website_request_id>` across recaptures and scene
revisions. A catalog change, restart or new campaign never resets its budget.
Reservations with uncertain outcomes remain reserved and are reconciled;
they are never blindly launched again. A proven pre-allocation refusal can
release its reservation. An allocated, closed attempt consumes its full cost
ceiling conservatively. Different sites are visited fairly even when an older
campaign needs reconciliation.

Closing a beta attempt checks its exact allocated instance and authority's
resource name against a fresh, complete account inventory, plus its own
teardown and staged-object absence receipts. Unrelated running sites remain
visible in that receipt and do not prevent closure. This proves absence for
the individual attempt; it makes no claim that the whole provider account is
empty. Missing or stale evidence keeps the reservation open. Proven teardown
with lost scientific output closes as a paid failure and publishes no score.

## Candidate configuration and intentional retries

The shipped catalog is `config/beta_policy_candidates.v1.json`; an
operator-owned absolute file can replace it through
`BLUEPRINT_BETA_POLICY_SCREENING_CATALOG_PATH`. Each entry pins policy identity,
interface, robot, cameras, control/action format, runtime, task capabilities,
licensing and credential requirements. Additional executable policies need a
sealed `screening_candidates` runtime entry in the existing operator scene
profile. Existing adapters are reused; a genuinely different interface needs
an adapter and its validation before the planner can select it.

Plans are immutable. If a quote expires or a runtime, account or credential
becomes available after planning, increment the catalog's `revision` and
re-seal the catalog to request a fresh campaign. This is an intentional retry
generation, not a fresh spending allowance. Completed and uncertain earlier
attempts remain in the site's ledger. Merely restarting or deploying code
does not rerun a site's candidates.

### Configure another compatible hosted policy

Additional candidate IDs can use `openpi_droid_https.v1` without changing code.
They use the existing DROID robot, two-camera observations, 15 Hz joint-position
actions and admitted image conversion. Another robot or wire interface still
needs its actual adapter and model bundle; changing a catalog label cannot
admit it. Local OpenPI/GR00T provisioning retains its exact pinned identities.

Add the candidate to the operator catalog with its exact model identity,
interface requirements, licensing and credential requirement. Its sealed native
execution spec must use `evaluation_backend: "robolab.v1"`, the same candidate
ID and task success contract, and `policy_interface_id: "openpi_droid_https.v1"`.
The `policy_spec` carries the policy ID, HTTPS inference and metadata URLs,
`allowed_origin`, pinned `model_identity_digest`, action chunk size and executed
horizon. `policy_endpoint` contains only the private file environment name:

```json
{"credential_env": "BLUEPRINT_BETA_POLICY_TOKEN_FILE"}
```

Seal the spec with
`canonical_digest(spec, digest_field="execution_spec_digest")`. Retain it as a
regular operator-controlled file and reference its actual bytes with an
absolute `path`, prefixed SHA-256 and `size_bytes`. Never put a token in this
spec, the catalog, a URL, command argument or shell history.

In the existing team registry, an explicit profile or
`derived_profiles.defaults.screening_candidates` can contain:

```json
{
  "hosted_policy_a": {
    "policy_execution_spec": {
      "path": "/var/lib/blueprint/operator/hosted-policy-a.json",
      "sha256": "sha256:<actual-file-sha256>",
      "size_bytes": 12345
    },
    "checkpoint_inventory": null,
    "credential_file": "/etc/blueprint/secrets/hosted-policy-a",
    "credential_origin": "https://policy.example.com",
    "provider_hard_cap_usd": "4.00",
    "hosted_inference_payment": "blueprint",
    "hosted_max_cost_per_request_usd": "0.001",
    "hosted_inference_maximum_usd": "1.00"
  }
}
```

The example digest and size are placeholders, not valid admission evidence.
The credential must be a regular, non-symlink file owned by the Pipeline
service user, readable only by that user, with mode `0600` and at most 65,536
bytes. Its declared origin must exactly match both endpoint origins. The
existing private runtime-secret channel supplies it to the provider; neither
the bundle nor publications contain its contents. Reseal the registry's
`registry_digest` after operator changes, and use the catalog module's `seal`
command to write a new sealed catalog rather than editing digest strings.

The planner derives `hosted_http_request_limit` as
`(2 * max_policy_queries + 4) * episode_count`. The existing HTTPS client makes
one metadata GET and one inference POST per query; four additional readiness
GETs are reserved per episode. It performs no HTTP retry or redirect and no
separate inference warmup. The operator's maximum price must cover **every**
request, including chargeable metadata and provider fees. The API ceiling must
cover that request limit times the per-request maximum, rounded up to cents.
Unknown or insufficient pricing makes the candidate unavailable.

Only an explicit `hosted_inference_payment` of `free` or `caller_paid` permits
both API price and API maximum to be `"0"`. For Blueprint-paid inference, both
must be positive. The candidate reservation and quote include GPU **plus** API
ceilings; the canonical GPU allocator receives only the GPU portion. The $4
GPU and $1 API example reserves $5 from the site's shared $15 allowance. Live
metadata identity is verified before every inference inside the paid session;
planning performs no potentially billable endpoint calls.

The new worker uses the existing Pipeline service account, provider secret
files, artifact transport, paid admission, watchdog and website HMAC. Canonical
deployment installs and arms its timer while preserving existing operator
holds. In particular, the unrelated paid agent dispatcher remains held.

## Observe and recover

The queue root defaults to
`/var/lib/blueprint/pipeline-control-plane/beta-policy-screening` and can be set
with `BLUEPRINT_BETA_POLICY_SCREENING_QUEUE_ROOT`. It contains immutable plans,
attempt artifacts, source statuses, the shared budget ledger and publication
outbox. Use the operator door's `journal`, `ls` and `cat` commands for inspection.
Do not delete or rewrite the ledger to retry work. Fix the reported source,
runtime, quote or delivery blocker and let the worker reconcile. Source-level
failures before planning are retried automatically.

Successful screening is **diagnostic**. It never qualifies a pilot or replaces
the separately authorized finalist protocol. Blueprint finalist scoring is
eligible under the owner decision, but implementing that final recommendation
path is separate from this ten-episode screening lane. ADP's frozen pair and
the existing team scoring protocol retain their current rules.
