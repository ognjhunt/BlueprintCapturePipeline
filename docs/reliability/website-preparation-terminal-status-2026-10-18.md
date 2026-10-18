# Website preparation reaches a state the site is told about

Beta journey L3/L6 (site capture → scene preparation → status and messages).
Observed blocker: a read-only audit on 2026-10-18 found no "cannot build"
signal between the Pipeline and the WebApp. A dead-lettered handoff, a spend
cap refusal, a capacity shortage and a crashed final attempt all left the site
on "preparing" or "needs more information" indefinitely. This slice extends the
existing signed status route from
[website-preparation-status-2026-10-08.md](website-preparation-status-2026-10-08.md);
it adds no queue, credential, provider call or job status.

## Receipt versions

`POST /api/live-pipeline/website-preparation-status` still answers
`website_preparation_status.v1`, byte for byte, unless the reader sends
`X-Blueprint-Preparation-Status-Version: 2`. A WebApp that predates v2 never
sends it, so it never receives a field or state it would refuse. A newer
version number is answered with the newest version this Pipeline has.

`website_preparation_status.v2` has the v1 fields plus `terminal_reason`, and
two more states:

| ledger | v1 state | v2 state |
| --- | --- | --- |
| `processing` | `preparing` | `preparing` |
| `retryable_blocked`, only capacity blockers | `awaiting_inputs` | `capacity_delayed` |
| `retryable_blocked`, other blockers | `awaiting_inputs` | `awaiting_inputs` |
| `failed_retryable`, capacity error | `failed_retryable` | `capacity_delayed` |
| `failed_retryable`, other error | `failed_retryable` | `failed_retryable` |
| any of the above with a current terminal record | unchanged | `failed_terminal` |
| `terminal_authority_ended` | `authority_ended` | `authority_ended` |
| `completed` with its handoff | `handed_off` | `handed_off` |

`terminal_reason` is `null` except for `failed_terminal`, where it is one of
`permanent_invalid`, `retry_exhausted`, `budget_exhausted`, `capacity_blocked`
or `out_of_scope`. This Pipeline emits the middle three; the WebApp accepts all
five so a later emitter needs no new version. Capacity means a control-plane
disk or staging shortage (`pubsub_handoff_staging_capacity_insufficient`,
`control_plane_disk_budget_exceeded`); a budget refusal is never capacity.

The WebApp's acceptance names the version it read
(`website_preparation_status_acceptance.v1` or `.v2`), and delivery re-reads
that version before it consumes the wake-up.

## When preparation ends for good

`website_preparation_terminal.py` adds one record,
`website_preparation_terminal.v1`, to the ledger's current attempt:

- The listener records it after the final Pub/Sub delivery attempt of the
  ledger's own source stays retryable. Pub/Sub dead-letters that delivery and
  nothing re-drives it automatically. The reason is `budget_exhausted` for the
  WebApp's typed 409 `website_scene_preparation_budget_exhausted`,
  `capacity_blocked` for a capacity-only wait, and `retry_exhausted` otherwise.
  Another source's dead letter (for example the storage trigger's duplicate
  handoff, which never claims) records nothing.
- A claim made on the final delivery attempt stores `final_delivery_attempt`.
  If that worker dies, its lease expires unrenewed; the listener tick's sweep
  records `retry_exhausted` ten minutes after the expiry. An expired lease on
  an earlier attempt waits for its redelivery as before.

The record is an annotation, not a status: claim, retry, acknowledgement and
workspace retirement are unchanged. It counts only while its observed status,
attempt, source payload and lease token still match the ledger, and the next
claim (an operator re-drive) drops it with the rest of the attempt.

## Other changes

- An attempt that failed before `pipeline/website_task_context.json` existed is
  reported, bound to the site's current task context. A retained context is
  still required to match and is never replaced, and a hand-off is never
  reported without one.
- Selectors and the delivery sweep accept the WebApp's `supplement-<40 hex>`
  captures of the same request. They stay inert until a supplement can be born:
  the owner observer on both sides is walkthrough-only today, so a supplement
  handoff is refused before any ledger exists. The WebApp's preparation watch
  alerts on those uploads instead.
- When the configured capture-root mapping does not name the selected capture,
  the route reads the listener's own workspace,
  `$BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT/<bucket>/scenes/<scene>/captures/<capture>`,
  only if that variable is set to an absolute directory and exactly one bucket
  holds the capture. Every verification above still runs. Without the
  variable nothing changes.

## Deploy order and rollback

Either order is safe: the version is negotiated per read. Deploy the WebApp
first so it starts asking for v2 (an older Pipeline ignores the header), then
the Pipeline. Rolling back either side returns both to v1.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. python -m pytest -q -p no:cacheprovider \
  tests/test_website_preparation_terminal_status.py tests/test_website_preparation_status.py \
  tests/test_pubsub_handoff_listener.py tests/test_website_assessment_resume.py
ruff check src/blueprint_pipeline/website_preparation_terminal.py src/blueprint_pipeline/website_preparation_status.py \
  src/blueprint_pipeline/website_preparation_status_http.py src/blueprint_pipeline/pubsub_handoff_scene_operations.py
```

The tests run the actual birth, lease, ledger, signed route, wake-up and Pub/Sub
pull code. The owner and context transports, the WebApp callback and the Pub/Sub
client are local fakes; no provider, network or production state is used.
