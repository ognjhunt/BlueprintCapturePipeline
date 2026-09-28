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

## Next reviewed slice: native VM offer/create contracts

Checked the existing Vast `_search_payload`, `_offer_summary`, `_select_offer`,
`_create_payload`, receipt redaction and the canonical G1 paid dispatch.
The shared search/select/create helpers currently have no VM capability field.
Add opt-in `require_virtual_machine` to search/select and `virtual_machine`
plus the selected offer to payload construction. Default container behavior
must remain unchanged. Search must filter `vms_enabled eq true`; selection
must independently require explicit provider `true` or integer `1`, while
still applying all current GPU/driver/disk/cost/geography predicates. Missing,
string or false values are not VM capability proof. Creation must require
that offer, SSH-direct mode, no hidden template defaults and an immutable
fully qualified `docker.io/vastai/kvm@sha256:` image, and set `vm: true`.
Receipts record the host type, without restoring raw bootstrap/secrets.

Red-first cases cover provider filter, capability variants, default behavior,
disk/hourly-rate predicates, immutable image/SSH/template/offer rejection and
redacted explicit-VM payload. These are the adapter's existing launch helpers,
not another provider launcher or a standalone planner. Keep VM execution
unexposed at the canonical allocator/CLI until host bootstrap, bundle and
actual worker integration pass; the G1 container/archive refusal remains.

Metadata-only registry observation: the official documented `ubuntu_terminal`
tag resolved at `https://registry-1.docker.io/v2/vastai/kvm/manifests/ubuntu_terminal`
to `docker.io/vastai/kvm@sha256:28dc36f977d4a078ee410caf08f595d91f95185a00e0d4e7970c2d11f7358738`.
The returned manifest bytes independently hash to that digest. Declared
compressed layer bytes sum to 2,981,311,243; this is not expanded disk size or
GPU/runtime qualification. No blobs were downloaded; no image authority or
runtime approval is inferred. Verify platform/guest/runtime inputs before
admitting this candidate. Design review accepts this opt-in helper slice and
preserves the full remaining runtime and user-facing objective.

### Implemented VM helper slice and next host dependency gate

The existing search/select/create functions now implement these opt-in
contracts. Selection and retained offer summaries preserve explicit true,
false and unknown VM capability. `_create_request_summary` records the
explicit VM request while keeping bootstrap bodies redacted. The canonical
allocator and public CLI still expose no VM option, and the default adapter
invocations still use container behavior. No paid dispatch predicate was
relaxed. The new tests were red before implementation and red again for
dropped VM capability in retained summaries and for converting unknown data
to false. Final **32 focused tests pass in 3.44s**: 21 VM contract cases plus
11 existing launch/driver/VRAM/disk/transfer/geography/redaction cases,
including the actual adapter's fake-provider create/poll/teardown flow.
Changed-file Ruff and diff checks pass. The real repository-wide static
`scripts/verify_paid_resource_allocator.py` reports
`paid_resource_allocator_verification=passed`; no new paid launcher exists.

Registry config metadata also independently verified
`sha256:2b2425c36395f4e9ec63926f522c01864a6ac66205d072dfab7763385459ac51`,
declaring Linux/amd64 and 13 layers. This does not establish the guest Python,
GPU driver or runtime capabilities. The existing G1 runtime review binds
CPython 3.12 wheels; they must not be installed into an assumed host Python.
For a separately verified Python 3.10 host, metadata-only candidates are:

| Host-only package | Wheel | SHA-256 | Declared bytes |
| --- | --- | --- | --- |
| NumPy 1.26.4 | `numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` | `ffa75af20b44f8dba823498024771d5ac50620e6915abac414251bd971b4529f` | 18,240,889 |
| RFC8785 0.1.4 | `rfc8785-0.1.4-py3-none-any.whl` | `520d690b448ecf0703691c76e1a34a24ddcd4fc5bc41d589cb7c58ec651bcd48` | 9,240 |

Primary metadata: `https://pypi.org/pypi/numpy/1.26.4/json` and
`https://pypi.org/pypi/rfc8785/0.1.4/json`. Exact existing owner approvals in
`docs/runtime_dependency_license_policy.json` expire 2027-08-15 and
2027-08-23 respectively. No package bytes were downloaded/installed, no
approval record changed, and the complete host import/provisioning closure
is not yet proven. These candidate host versions do not replace the
simulator's NumPy 2.3.1 or model dependencies. Next integration must bind and
verify the actual host ABI, its approved bytes, Docker/bubblewrap/GPU devices,
and simulator/relay lifetime before exposing a VM paid launch.

Current previous-head 3f96d7e35 CI 36378803566 impacted job108790383058 is
green; all four full-suite shards remain live at this checkpoint. Work disk
still has 11,136,303,104 available bytes; no fresh provider-zero assertion.
VM helper tests/pinned metadata alone do not meet any scored-episode, media,
private URL, merge or deployment completion requirement.
