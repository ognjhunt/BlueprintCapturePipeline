# Blueprint screening runtime control (AUT-10)

This contract stops additional processing within an existing admitted screening
attempt. It creates no grant, funds, provider allocation, credentials, new team
identity, publication authority or restart permission. Existing founder holds,
per-site approvals, rights and resource limits continue to apply.

## Control plane and exact identity

The existing Pipeline controller uses its existing HMAC credential to observe
`POST /api/internal/pipeline/beta-policy-screening-runs/:campaignId/execution-control`.
That credential stays on the controller. The request binds campaign/site/plan,
candidate/configuration, attempt/invocation, prepared bytes, session authority
and resource name, plus all six existing source selectors:

- `scene_intent_id`, `scene_intent_digest`
- `capture_id`, `task_id`
- `task_context_digest`, `scene_revision_digest`

The response is `blueprint.beta_policy_screening_execution_control.v1`, with
cross-runtime canonical `binding_digest` and `control_digest`. It always says
`new_execution_authorized=false` and `read_only_closeout_allowed=true`.
`current` only confirms that the already bound source remains applicable.
Missing actual scene-intake expiry is unavailable authority. Observed deadlines
only shrink. Campaign cancellation, durable owner closure, rights withdrawal
and changed source refuse continued processing; a retained stop never reopens.
Late diagnostic receipts remain evidence with their original scientific bytes,
and owner-closed/cancelled evidence is historical in actual managed readback.

## Existing provider channel

`beta_screening_control_delivery.py` registers the active allocator monitor for
the exact authority digest/resource name. The Vast adapter first joins the ID
created by that adapter to its authenticated current provider response and the
existing strict SSH host pin. It then subscribes the bounded fixed-file sink.
No service token or SSH identity is forwarded to the GPU.

Only the fixed receiver command runs over the existing SSH channel. Bounded
JSON arrives on stdin. The receiver selects exactly one matching sealed
authority under either existing bundle root:

```
/workspace/adp_arena_provider_bundle/provider_runtime
/tmp/blueprint_vast_work/adp_arena_provider_bundle/provider_runtime
```

It verifies `runtime_inputs/policy_canary_session_authority.json` and writes only
the bundle parent's `blueprint_beta_execution_control.v1.json`, under a local
lock with atomic replace/fsync. Inputs cannot supply a shell command or remote
path. A failed delivery never advances the GPU's last observation. The
subscription is removed before the adapter's existing exact-resource teardown.
When this relay is active, the legacy option to ignore local SIGTERM cannot
mask the allocator's authority stop. Unrelated/team behavior stays unchanged.

## GPU boundaries and retained stop

The GPU ships only `beta_screening_runtime_control.py` and the shared pure wire
validator, not the SSH delivery or control-plane transport modules. Initial
startup waits within the existing 600-second monitor grace and the smaller
provider deadline for the controller, without model/engine work.
Checks run before provisioning, session/runtime start, each isolated cell spawn,
policy client load, and every query through the actual loaded policy client.
The existing native/DROID and HTTPS beta transports both install the relay.

The guard joins the sealed session, runtime inputs, resource, candidate/plan and
scene revision. Its retained binding digest pins every attempt/invocation and
source field. It retains the first maximum deadline and only reduces effective
authority, including the existing provider watchdog's absolute deadline. The
existing 600-second monitor outage bound applies to the last
verified source observation, never to delivery or file-read time. Expiry,
stopped authority, stale/future observations, missing files, unsafe files or
rebound identity cause an absorbing local stop. A fresh response cannot revive
that stopped attempt.

The stop receipt preserves the exact source control when available. Completed
prefixes remain retained; a stopped mid-query child stays a hash-bound raw
partial result and cannot become a completed scored episode. Results are
historical-only. Existing finally/closeout and independent watchdog teardown
remain permitted after processing stops. Media retention/deletion policies are
separate from proof that work stopped.

## One stop signal, and retries before spend

The allocator's own monitor sends its stop. It records `stop_signalled.json`
(exclusive, atomic) before the SIGTERM. The screening timer's reconcile signals
only when no stop signal was recorded and the monitor's heartbeat
(`monitor_heartbeat.json`, one beat per poll) is older than three polls; it
records the same file first. The adapter turns the first signal into its
teardown. A repeat, or any signal once teardown has begun, is recorded in
`vast_signal_teardown_guard.json` and ignored, so the instance DELETE finishes.
Around the whole allocator run, a SIGTERM before the provider create intent
unwinds it. The intent is written only inside the adapter's pre-create hook
(native, team and G1 lanes), so a later SIGTERM finds the teardown done and is
ignored: the watchdog close and `allocator_result.json` always complete.

Once `allocator_result.json` exists, reconcile never reads the control again;
only a stop recorded while the allocator ran applies to it. While the live
monitor's last success is fresher than two polls, reconcile reuses it instead
of sending its own request.

A read that never got the WebApp's answer (408, 425, 429, 5xx, timeout,
connection failure, truncated or non-JSON body) refuses without spending.
Before spawn, the attempt claims no launch slot, keeps its reservation and reads
again. At create admission, the allocator fences its own job
(`no_create_fence.json`), so recovery retries the candidate on a fresh attempt.
A definite stopped, changed or expired answer, any other 4xx, or an invalid
response still blocks. The execution-control route answers 409 only for a
refusal that will not clear (for example an intake with no expiry), so a 409
is an absorbing stop (`beta_screening_execution_control_refused`) at once,
including for running work. The 600-second rule for running work is unchanged; a
timeout stop recorded before any create is retried like a failed read.

These retries are capped per candidate at `TRANSPORT_RETRY_CAP` (3, in
`beta_policy_screening_coordinator.py`), counting preflight reads again and
fresh attempts after the allocator fenced its own start or create together.
They wait 1, 2, then 4 minutes. The next such refusal blocks only that
candidate with `beta_screening_transport_retry_cap_exhausted` and releases its
reservation; the other candidates and the campaign continue. A fresh attempt
after a crash or a deploy before create (`allocator_lost_before_create`) is
not counted.

## Verification and remaining acceptance

Offline coverage exercises strict WebApp signatures/readback, all six source
selectors, expiry/deadline reduction, revocation and cancellation, controller
loss, absorbing restart, unrelated resource continuation, fixed SSH command and
pin checks, actual worker load/query/cell callbacks, both beta Vast branches,
partial retention and signal cleanup. External API/SSH/engine boundaries are
mocked; these tests do not establish a production stop or teardown receipt.

These worker/provider seams depend on the allocator invocation
(`allocator_invocation.py`), its active-monitor registry
(`beta_policy_screening_control.py`), the signed WebApp transport
(`beta_policy_screening_control_transport.py`) and the shared pure contracts
(`beta_policy_screening_control_contracts.py`); they ship together.

Native G1 screening runs under the same invocation, monitor and signed
create-time admission. Its provider bundle has no GPU-local relay, so a G1
stop ends the allocator, and with it the instance, rather than a single cell.

Deployments and real revoke/startup/between-cell/expiry/controller-loss tests
remain pending. This change establishes no new engine, robot/task readiness,
independent site fidelity measurements, finalist spending approval or
unattended production acceptance.
