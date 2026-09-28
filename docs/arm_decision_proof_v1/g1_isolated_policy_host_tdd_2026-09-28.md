# G1 isolated policy host — reviewed TDD plan

Owner-authorized G1 extension, ADP-050 Day 28, private evidence ADP-060.
The existing team container/archive launchers require Docker or bubblewrap;
neither is available inside the deployed Isaac container. Those modes are
still refused before paid allocation. The original completion scope remains
the shared configurator, actual scored manipulation/movement, all three
delivery modes, complete observations/media, paid closeout and owner results.

## Runtime decision and evidence

Use the existing Vast provider's GPU VM mode for an isolated policy host. Keep
the Isaac image and team policy in separate sandboxes on that VM. This permits
one provider instance, budget, launch slot and independent watchdog to own
both; no second provider or new primary service is introduced. The team image
receives stdin/stdout only, no network or scene/evidence/credential mounts. A
noncontainer archive executes through the existing regular-file/digest
verification and isolated process path on the policy host. Policy delivery
mode remains distinct from the simulator container and the cloud host type.

Primary references checked 2026-09-28:

- [Vast VM documentation](https://docs.vast.ai/guides/instances/virtual-machines):
  VM-capable offer filter, SSH-only launch, KVM image family, Linux/Docker
  support, increased disk overhead and slower boot.
- [Vast create-instance API](https://docs.vast.ai/api-reference/instances/create-instance):
  explicit `vm` flag and image/launch mode fields.

These documents do not prove current offer availability, a pinned VM image,
driver compatibility, namespace capability, valid inference, or successful
site execution. The current adapter has no VM option; a normal container
must never be relabeled as a VM. A VM launch must explicitly bind the reviewed
image digest and compatible offer, bootstrap and runtime identity. Keep the
same canonical allocator admission, credits/cap/TTL, slots, release, watchdog,
collection reservation, staging cleanup and official billing gates.

## First executable slice: authenticated private relay

Add a bounded Unix-socket relay between the trusted simulator and an already
isolated, operator-bound policy session. Keep the existing JSONL policy
protocol and semantic-v3 client; the relay never takes a shell command or
launches a provider. Bind execution packet, selected profile, source setup and
original policy delivery mode. Authenticate with a fresh challenge and an
ephemeral secret that the team policy never sees. Do not log observations,
policy replies, secrets or socket addresses in returned receipts.

Only reset and infer reach the existing policy client's exchange method.
Close is an administrative relay operation, never a model command. Require
the factory session's profile and successful synthetic conformance before
forwarding site input. A peer cannot choose another profile, send a replayed
request id, introduce arbitrary envelope fields, or invent a closed policy.
Rebind the policy reply to the caller's request id after the underlying client
has verified its own response identity. The simulator-side client still
validates actual semantic-v3 actions before marking successful inference.

Bound framing, handshake and request deadlines; reject duplicate keys,
nonfinite JSON, oversized frames and partial EOF. Timeout/disconnect never
retries a request and always closes the owned policy session. Refuse an
existing/aliased socket path and untrusted parent; unlink only the socket inode
this server created. Retained output is not a cloud teardown or paid grant.

Red first, using real local Unix sockets and an actual JSONL subprocess:

1. Reset/infer round trip preserves observations/action/profile and request id.
2. Wrong secret, packet/profile/setup/mode and changed session conformance
   refuse before any site request or unauthorized policy factory invocation.
3. Replay ids, unexpected commands, oversized/duplicate/nonfinite frames and
   unacknowledged/invalid policy actions fail closed without retry.
4. Disconnect, timeout and normal close each reap the owned policy child;
   failed policy close remains unproven. Secret/raw payloads never enter receipts.
5. Existing/symlink socket paths and foreign replacement inodes are preserved.

## Remaining integrated slices, required before admitting VM execution

Pin and review the VM/runtime images and exact bootstrap environment; add
VM-capable offer selection and SSH-only launch to the canonical adapter, with
dry and execute paths sharing full checks. Preflight guest Docker, GPU driver,
process isolation, memory and disk (including both images and evidence).
Container/archive bytes and rights must agree with the accepted packet and
current approval. Bind a host runtime/relay receipt into the worker; do not
forge an HTTPS delivery profile. Archive GPU device exposure must be an exact
operator-controlled device set, without network or extra host mounts.

Stage the relay secret privately, not into sealed/public bundles. The provider
host owns both children; all interruptions leave retained typed child/native
evidence. Extend output verification to reopen the actual host session/child
close receipts along with the existing worker, score, every lossless frame and
both videos. VM destroy plus staging absence and fresh official billing/zero
are still separate closeout requirements. Only then remove the current
missing-runtime preallocation refusal for the admitted exact modes.

Qualify authenticated HTTPS, OCI and archive policies on actual scored G1
episodes; then deploy/exercise the existing shared configurator and private
results URL. Run all four built-ins after collection admission clears. VM mode
does not remove the known control-plane collection-space gate and is unrelated
to the owner-decision gate for Plan 13b CPU workers. Public publication stays
unapproved; no storage hand deletion or cleanup-rule changes.

Design review: checked the actual paid preflight, provider create payload,
container/archive launchers, session close semantics and JSONL/G1 wire. A
separate policy host is necessary; the relay preserves the original profile
and scientific boundaries. Accepted for the first test-first implementation.
This review does not authorize a VM image, renew operator approval, qualify
runtime capabilities or complete the goal.

## Review follow-up before retaining this slice

Reopen the real runtime session's conformance and close receipt digests in the
relay; bind both into its own digest-bound, secret-free transport receipt.
Exercise the production conformance/session close implementations with a real
local JSONL child as well as the narrow transport fixture. The local process
test proves protocol and process cleanup, not container or archive isolation.

Exact CI head 7239b325146a8f38c189f98b670a3a41918e6589 exposed one owned
storage declaration mismatch: the newly classified `g1-sonic-assets` and
`g1-team-policy-approvals` are hot evidence inside the already bound inputs
tree, but the mount script does not list them in its hot-evidence declaration.
The existing mount governance test is red. Add these exact two declarations
with ownership reasons and replay that test plus the script's no-mutation
plan check. Do not relax its assertion, migrate a tree, change retention or
invoke the mount script on the live host.

## Implemented first slice and verification

`native_g1_team_policy_relay.py` now provides challenge authentication and
packet/profile/setup/mode binding, bounded strict JSONL frames and deadlines,
ordered reset/infer requests, no command or provider launch input, and owned
policy closure on timeout/disconnect. A failed exchange or invalid action
latches the client failed; no request is retried. Administrative close is not
sent to the policy. The relay verifies conformance and close digests and binds
them into its own receipt, preserving `planning_only` and provider teardown
false. Socket cleanup checks its original parent and socket inodes and never
removes a foreign replacement. Host import is standard-library-only; simulator
semantic helpers load only when its client builds/validates G1 inference.

Red-first tests failed for the missing module, then for missing failure latching
and receipt binding before those repairs. Final focused command passed **47
tests in 6.00 seconds**: 33 relay cases, 5 existing JSONL cases, 6 existing
runtime session cases and 3 mount declaration/no-mutation plan cases. These
include real Unix sockets, actual Python JSONL children, exact image/state/task
echo, invalid action, failed acknowledgment, policy stall, replay/auth failures,
production conformance and session closure, strict parser/path negatives, and
an isolated `-I -S` host import. The process fixture does not establish actual
container/archive isolation, GPU inference or scene scoring. No paid allocation,
provider mutation, mount operation or data deletion occurred.

CI 36376545785 for the preceding 7239b325 head has exact impacted/sentinels
green, shard 0 green, shards 1/2 terminal red and shard 3 still live at this
checkpoint. The owned hot-root declaration failure is repaired here. Other
logged isolation/materializer/quality/dispatcher/supervisor failures remain
tracked on #2236; relevant source/test files are unchanged versus its actual
synthetic merge base 23eb257085f1068f2da2a0e100d024538ab901b4 (merge
84f4133d03436ba1f04fe694e383ec2432c9b956). That comparison does not prove a
suite-order cause or a green full suite.

Fresh host `df -B1` remains 11,136,303,104 available work bytes, below the
28,162,041,744-byte comparable four-trial collection forecast. Paid run
admission remains closed for that reservation; current provider inventory was
not rechecked. The VM launch/bootstrap/integration, three delivery modes on
actual scored episodes, full four built-ins, working private URL and final
merge/push/deployment remain required. This transport increment does not remove
the current container/archive preallocation refusal.
