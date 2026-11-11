# Off-host GPU reaper

Everything that stops a rented Vast machine runs on the control-plane host:
each launch's watchdog (`groot_oscar_runpod_watchdog`), the spend guard
(`scripts/gpu_spend_guard.py`) and the pending-teardown reaper
(`paid_lane_guard`). If the host dies, hangs or is cut off, all of them stop
with it, and a machine keeps billing until someone notices. The off-host
reaper is the one stop that does not depend on the host.

## How it works

- **The host publishes** (`blueprint-off-host-reaper-heartbeat.timer`, every
  two minutes, `off_host_gpu_reaper_heartbeat`). It reads the command line of
  every running `groot_oscar_runpod_watchdog --provider vast` process and
  rewrites one object, `host-state/latest.json`, in the reaper bucket. The
  object holds the heartbeat time, each watchdog's label scope, the instance id
  when the launch recorded one, the watchdog's deadline, and the longest
  deadline this release's lanes can arm (`DEADLINE_CEILINGS`, today
  481,800 s from `team_candidate_quick10_admission.MAX_TTL_SECONDS`), and
  `deadlines_complete`: whether it read every watchdog's deadline. A process
  it cannot read, or a watchdog whose arguments it cannot parse, publishes
  `deadlines_complete: false`. A short list is never published as whole, and an
  object without the flag reads as not complete.
- **The reaper runs off the host** (Cloud Run job
  `blueprint-off-host-gpu-reaper`, started by Cloud Scheduler every ten minutes,
  `off_host_gpu_reaper`). Each pass reads the object and lists the Vast account.
  It destroys a machine only when its label is Blueprint's (`blueprint-...`) and:
  - it is past its published deadline plus 15 minutes, whether the host is up
    or not; or
  - the heartbeat is older than 30 minutes and the machine is past its
    published deadline; or
  - the heartbeat is older than 30 minutes, the last list was marked complete,
    the machine is absent from it, and it started more than 10 minutes
    (`ORPHAN_START_MARGIN_SECONDS`) before that heartbeat. Its watchdog died
    before the host did, and nothing else will stop it. A launch arms its
    watchdog before it creates the machine, and the next publish is at most
    about 4.25 minutes away; 10 minutes is about twice that, which leaves room
    for clock skew between Vast and the host; or
  - the heartbeat is older than 30 minutes and the machine has no published
    deadline and is older than the published longest deadline. This applies to
    a machine launched within that margin of the last heartbeat, and to every
    absent machine when the list was not marked complete.
- **What it never touches:**
  - A machine inside its published deadline, host alive or dead.
  - A machine with no usable start time from Vast: it is `unknown` and kept.
  - A label that is not Blueprint's. These are left alone and reported with
    ownership unknown; the account is shared with other operators.
  - Anything at all when no heartbeat was ever published, when the object is
    unreadable, or when the provider's list fails or is malformed. The pass is
    `unknown`. It never reads as "nothing running".
- **Read again before acting.** Before deleting, the pass reads the object and
  the list again. It destroys only a machine both readings condemn.
- **Proof.** After the deletes the list is read again, up to three times. A
  machine is `destroyed` only when it is gone from the list or shows a terminal
  status there. A delete that fails, or that the list does not confirm, is
  `failed`, never `destroyed`.
- **Report.** Every pass prints its report, which goes to the job's log. It also
  writes the report under `reports/` in the bucket, where reports are kept for
  30 days.
- **Paging.** The job exits non-zero when a pass is `unknown` or `failed`. The
  alert policy `off_host_gpu_reaper_failures` pages the existing monitoring
  channels when that happens, and also when no pass has run for 30 minutes.
- **No budget admission.** Destroying stops spend (`spend_effect="stop"`), so a
  pass needs no budget admission and reserves nothing.

## Deadline extensions and the exposure that remains

The object is rewritten on every heartbeat with each live watchdog's deadline.
When a watchdog is superseded with a later deadline
(`supersede_independent_vast_watchdog`), the successor's deadline reaches the
reaper within one heartbeat while the host is up. If the host is unreachable,
the last published deadline stands. An extension the host made after its last
publish is not seen. The machine is then reaped once it is past the old
deadline: when the heartbeat goes stale, or 15 minutes after the deadline,
whichever comes first.

After a host death, a published machine runs at most until its own deadline,
plus the 15-minute grace, plus one ten-minute tick. A machine whose watchdog
died before the last complete publish is reaped once the heartbeat is 30
minutes stale. A machine started within 10 minutes of the last heartbeat, or
any absent machine after an incomplete publish, runs until it is older than the
longest deadline, 481,800 s (about 5.6 days). That bound is weak, and it is the
code's own longest deadline. A Vast machine guarded by another kind of host
watchdog is never in the list. With a complete list and a stale heartbeat it is
treated as an orphan; with the host gone, that watchdog is gone too. Stopped Vast
instances (terminal status) are not reaped; they bill storage only.

When the heartbeat cannot publish (the bucket or key is broken), or can only
publish an incomplete list, renting goes on. The heartbeat pages once on the capacity controller's webhook
(`off_host_reaper_heartbeat_failing`) after ten minutes of failures, and again
only after a publish succeeds and then fails anew. Its status record is
`/var/lib/blueprint/pipeline-control-plane/off-host-gpu-reaper/heartbeat.json`.

## Owner steps (none is done by the code)

CI runs `terraform fmt` and `terraform validate` on this Terraform and both pass. It has never been through `terraform plan` against the real project.
No cloud credentials were available where it was written, so nothing here has been applied.

1. Create a Secret Manager secret holding a Vast API key for the reaper.
2. In Terraform, set `off_host_gpu_reaper_enabled = true` and
   `off_host_gpu_reaper_vast_api_key_secret_name`, then plan and apply. This
   creates the bucket, the job, the schedule, three service accounts and the
   alert policy.
3. Make a key for the `off-host-reaper-heartbeat` service account. Install it
   on the host as
   `/etc/blueprint/credentials/off-host-gpu-reaper-heartbeat.json`, owned
   `root:blueprint`, mode 0640.
4. Add `BLUEPRINT_OFF_HOST_GPU_REAPER_BUCKET=<bucket from the
   off_host_gpu_reaper output>` to `/etc/blueprint/pipeline-control-plane.env`.

Until step 4 the heartbeat unit runs and does nothing. Until step 2 nothing
runs off the host.

## Not built: a self-destruct on the rented machine

A second layer would be carried by the machine itself. A timer in the
instance's start script would destroy the instance when its own hard deadline
passes, with no host and no Cloud Run job involved. It is not built, because
it rests on two facts about Vast that nobody here has verified:

1. **An instance-scoped key.** Does Vast give a running instance a credential
   (for example `CONTAINER_API_KEY` with `CONTAINER_ID`) that can destroy that
   instance and only that instance? The account key must never be placed on a
   rented machine, because the machine's owner can read it.
2. **Billing after exit.** Does an exited or stopped container stop GPU
   billing, or only a destroy? If only a destroy stops it, the timer must call
   the destroy endpoint, not just exit.

Once both are confirmed with Vast, the build would be:

- A small wrapper prepended to the start script (`onstart` or `args_str`) in a
  new module next to `vast_args_payload_transport.py`. The adapter is at its
  line cap.
- The wrapper runs a background sleep until the launch's hard deadline, set at
  create time to the largest deadline the launch may be extended to. It then
  destroys its own instance with the instance-scoped key.
- The host's watchdog and this reaper stay the primary stops.
- Tests would check the generated script with fakes: the deadline is baked in,
  no account key is in the payload, and the size stays under the 16,384-byte
  limit.

## Testing at $0

`tests/test_off_host_gpu_reaper.py`, `tests/test_off_host_gpu_reaper_heartbeat.py`
and `tests/test_terraform_off_host_gpu_reaper.py` use a fake store and a fake
Vast account. They cover:

- A dead host with a machine past its deadline: destroyed, with absence proven.
- A machine inside its deadline: left, host alive or dead.
- A foreign label: left.
- A failing or partial list: `unknown`.
- No heartbeat ever published: no action.
- A failed or unconfirmed delete: `failed`.
- A re-read that shows an extended deadline: left.
- The heartbeat's single page after ten minutes of failures.
