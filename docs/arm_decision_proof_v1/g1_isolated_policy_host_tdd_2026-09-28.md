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

## Reviewed next integration: scored episode and remote lifecycle

Observed gap: the relay currently returns action responses and only a close
acknowledgment. The existing supervised episode must link its actual episode
digest to a native runtime session and retain the exact conformance and close
receipts. An acknowledgment alone cannot satisfy those existing boundaries.

Before building the outer VM supervisor, extend administrative relay handling
to return its verified, strictly bounded conformance receipt, link exactly one
scored episode via the host session's existing validator, and return its actual
close receipt. Only reset/infer reach the team process. The episode's profile,
setup, original delivery mode, positive query count and digest must agree;
query count must equal the relay's inference calls since its last reset.
Administrative link is never a model command and never grades the episode.
Reject unexpected fields before returning receipts to prevent payload/secret
disclosure. Bind an actual child teardown digest; never synthesize a successful
runtime close or cloud teardown.

Add a simulator-side proxy for `NativeG1TeamRuntimeSession`, using a protected
operator-created relay configuration bound to packet/profile/setup/mode. The
secret/configuration contents never enter sealed bundles, command arguments or
output receipts. Only the private file path is passed to the trusted worker.
Copy verified host conformance/close receipts byte-for-byte as JSON
values, including the actual linked episode and child teardown digest. Reuse
the existing session link validator and supervised episode/scoring flow.
Thread the optional private config through the existing worker/supervisor;
refuse relay configuration for HTTPS. Keep container/archive preallocation
refusal until host/bootstrap/bundle/output verification is complete.

Red-first tests: real local socket and JSONL child; actual production synthetic
conformance, session link/close and existing G1 scored episode fixture. Assert
original OCI/archive mode, actual queries, preserved scored task failure,
exact host/guest receipt digests, actual child exit, no secret in retained
output. Faults: foreign config/packet, wrong conformance, zero/mismatched query
link, altered close/child digest, repeated link, teardown failure and absent
config; none may become a completed episode. Existing endpoint/scoring/client
behavior must remain green. Local fixtures do not qualify VM/GPU/isolation or
real task success; the host supervisor and actual paid episodes remain required.

Design review checked the real worker/supervisor argument propagation,
`run_g1_team_supervised_episode`, `NativeG1TeamRuntimeSession` link/close, and
`verify_g1_team_paid_output`. This integration preserves their actual receipt
bindings and original profile, without endpoint substitution or paid bypass.

### Implemented lifecycle connection and verification

Red-first collection failed for the absent simulator-side session module.
The relay now returns strictly validated native conformance and close receipts,
links one actual episode through the host session validator, checks inference
queries since reset, and keeps administrative requests out of the team child.
The simulator proxy retains the host receipts unchanged. Its operator-created
configuration is private, exact-field, packet/profile/setup/mode bound, size
bounded, and rejects aliases, extra fields, unsafe permissions and missing files.
The existing supervised episode, worker and child supervisor carry this optional
private configuration; HTTPS remains on its existing direct client.

The first positive rehearsal refused an expired fixture approval, then refused
the shared telemetry-only fixture at the real deterministic scorer. The test now
uses a current fixture approval, complete frozen rigid-task predicates and the
fixture's native reset pose. No production authority or scoring check was relaxed.
Both OCI and archive rehearsals retain the actual `never_moved` failure score,
positive queries, identical host/simulator close receipts and actual child exit.

Verification commands and boundaries:

- Relay/proxy/runtime/session/episode/worker/supervisor focused suite:
  **80 passed in 27.40s**. Real local socket and JSONL process; negative packet,
  private-file, episode query/profile/setup/mode/digest, repeated link, close
  tampering and teardown checks. No actual VM, isolation or GPU qualification.
- Additional conformance fault cases, isolated sealed selected-bundle worker
  and proxy imports, provider import closure and canary lifecycle rehearsal:
  **49 passed in 117.06s** (23 proxy, 1 selected-bundle, 6 closure, 19 lifecycle).
  The sealed worker exposes `--policy-relay-config`; all package imports in its
  isolated interpreter resolve within the extracted bundle.
- Changed-file Ruff and `git diff --check`: passed.

Live host read still identifies deployed release
`4a59c6bb668ac53ca9cc3440fcc57b21242d3612`, work free bytes
11,136,303,104 and root free bytes 17,614,856,192. No new provider allocation,
inventory/provider-zero claim, mount change or deletion. The comparable
28,162,041,744-byte four-trial collection forecast still exceeds work capacity.

Preceding exact-head CI 36379973315: impacted/sentinels passed; full shards
1/2/0 have terminal failures and shard 3 remains live at inspection. Shard 1
reported three static import-isolation leaks plus the retired SAGE compatibility
test's absent verdict directory; shard 2 reported materializer, listener-budget,
SDK pricing and task-object session failures; shard 0 reported historical gap
ledger status drift. These are retained for #2236 diagnosis, not labeled green.
The actual CI merge base is `7d3b4e9179730b7f3e9ccb2010da2d52fc030eaa`.
An independent AST traversal on that base and the committed PR head finds the
same 12 hot modules, including the path `live_pipeline_intake_service` →
`control_plane_capacity_controller` → `provider_credit_admission` →
`vast_provider_adapter`. That proves this leak exists on the base, not that all
suite failures are baseline or caused by order. No SAGE production work was added.

Next required integration remains the outer VM bootstrap/supervisor, host ABI
and sandbox/GPU preflight, actual host/child receipt verification in collected
outputs, and canonical VM allocation under the existing budget/watchdog/teardown
contracts. Container/archive preallocation refusal remains active until those
pieces are qualified. Actual episodes for every required mode, all four built-ins,
private URL, merge/push and deployment remain incomplete.

## Reviewed outer-host supervision slice

The actual provider input verifier currently requires the provisioning receipt
produced inside Isaac. Factor its existing sealed-input checks from that later
provisioning check, so the host can revalidate immutable packet, scene, publisher
and SONIC bytes before launching a policy. The worker still requires the completed
provisioning receipt and original operator approval/expiry.

Use a host Python entry point with no provider-allocation API. Require Linux,
root, x86_64, matching CPython 3.12, reviewed NumPy/RFC versions, functional
NVIDIA Docker runtime, the existing Isaac driver floor and exact local simulator
image. A full VM bootstrap/dependency admission remains a later integration gate.
Keep current canonical container/archive preallocation refusal.

Open the actual approved host NativeSession synchronously for synthetic inputs,
then transfer its close ownership to one relay thread. Only the trusted simulator
mounts the private runtime relay directory. Simulator image is the existing exact
Isaac pin; bundle mount is read-only, with narrowly scoped writable provisioning
and output mounts. No Docker socket/host root/host network reaches the simulator;
the team launchers retain their existing networkless, mount-free profiles.
Simulator bridge networking permits existing approved runtime provisioning.

Use the existing supervised child runner for its real Docker CLI handle, bounded
process-group timeout, quarantined log and exit receipt. On every path explicitly
force-remove the uniquely named simulator container and inspect absence; a Docker
CLI exit is insufficient. Stop the relay's active wire and join its sole close
owner before any parent close attempt. A stopped listener must wake without waiting
for the full startup timeout. Retain actual policy/relay/simulator close receipts,
fixed error classes and a digest-bound host result; never retain private config or
secret contents in the collected output. Cloud destruction/billing stay external.

The transported provider runtime may use a bound private relay file for OCI/archive;
without it the current refusal remains. HTTPS rejects private relay configuration.
Reopen host NativeSession, original child teardown, transport receipt, simulator
absence and CLI exit, plus the existing guest score/full-media verifier, before
accepting a VM host result. Child receipt image/archive and query counts must match
the immutable packet and actual guest episode. A correct guest result with missing
or changed host cleanup evidence is blocked.

Red-first tests: missing host module; fixed mount/image/argument security; real local
JSONL policy and Unix relay with actual NativeSession closure; observed child exit,
Docker removal/inspection failure, CLI timeout/nonzero exit, wrong packet, missing
host/conformance/child receipt, changed digest, and server stop before connect or
while blocked in socket read. Fake Docker metadata/simulator seams are explicitly
local orchestration tests, not VM, device isolation or GPU qualification. Existing
HTTPS/provider/bundle/relay behavior stays green. CLI and new host imports must reopen
from the sealed bundle in an isolated interpreter before any paid integration.

Design review checked existing image pin, provider provisioning path, staged archive
binding, actual lease close methods, relay accept/request deadlines, supervised
process handles and paid score/media verification. GPU devices for a noncontainer
policy and complete host bootstrap remain required before that mode's paid admission;
this host lifecycle slice must not claim those capabilities merely from metadata.

### Implemented host lifecycle and evidence reopening

The VM host now preflights CPython 3.12, NumPy 2.3.1, RFC8785 0.1.4,
Linux/root/x86_64, local immutable policy/simulator images, NVIDIA Docker and
the existing driver floor. This supersedes the earlier speculative CPython 3.10
host candidate: that candidate never proved the package import closure and is
not admitted by this host. No host dependencies were installed or downloaded.

One actual NativeSession owns the host policy and one relay thread owns its
closure during simulator execution. The existing child supervisor runs the
fixed simulator Docker command; explicit Docker removal and absence inspection
follow CLI exit, failure and timeout. Stopping the relay wakes both a 600-second
accept wait and an active socket read. The private relay directory is removed
after its thread terminates, and its secret/config are outside collected output.
The guest provider rechecks the private file's packet/profile/setup/mode binding
before the selected worker. Missing, foreign, nonprivate and HTTPS relay files
fail before worker contact. The default paired-runtime admission refusal remains.

The output verifier reopens guest scoring, lossless frames and both review videos,
inner worker supervision/exit, actual host/guest conformance and session copies,
relay query counts, original OCI/archive child cleanup, simulator absence and
outer CLI exit. It rejects absent, changed or inconsistently linked evidence and
does not assert provider teardown, posted billing, GPU qualification or public
rights. The host CLI and verifier import from an extracted sealed bundle under
an isolated provider interpreter.

Verification: **92 passed in 81.25s** across VM host (25), Unix/JSONL relay (35),
guest provider (8), isolated sealed bundle (1), and relay runtime session (23).
The actual local JSONL processes, sockets, NativeSession/lease closure, deterministic
task scorer, lossless frames and head/overview videos are exercised. Docker,
VM platform and Isaac remain test doubles; the fixture score is `never_moved`,
not a successful physical or GPU task. **25 required pre-paid tests passed in
102.10s**: six provider-import closure cases and 19 policy-canary lifecycle
rehearsals. Changed-file Ruff and diff checks pass.

Committed predecessor `239f2c97` CI 36382156418 is terminal: impacted execution
passed, but the overall required gate is red from full-suite shards. Inspected
logs show three static import-isolation failures; materializer reachability,
historical ledger, three SDK cost assertions and malformed nested JSON failures;
listener line-budget and controls payload-cache failures. Shard 1's 6,354 tests
passed, but its expected non-root Landlock skip fails the no-skips shard gate.
The controls payload-cache test passes alone locally in 43.24s; suite/environment
causality is not established. No full-suite rerun or SAGE production work began.

Fresh read-only host observation at this checkpoint: work available bytes
11,136,524,288; root available bytes 17,583,509,504. Work capacity is below the
28,162,041,744-byte comparable four-trial collection forecast. This is not a
fresh provider-zero or billing assertion. No paid allocation, deploy, merge or
hand data deletion occurred in this slice. Still required: sealed host bootstrap,
archive staging and GPU exposure, canonical VM allocator and collected-output
integration, actual episodes for all required delivery modes/four built-ins,
private result URL, final merge/push/deploy and official teardown/billing proof.

## Reviewed offline host Python provisioning

Observed default Vast VM is Ubuntu 22.04, which does not establish the required
CPython 3.12 host. Do not install binary cp312 wheels into its default interpreter
or assume the control host's Ubuntu-built interpreter is portable to the VM.
Provide a sealed, relocatable Linux x86_64 CPython distribution and the two exact
approved host dependencies before allocation. The bootstrap must use standard
library only, so its verification/extraction can run from the VM's system Python
without importing G1 or NumPy first. No provider pip resolver or model download.

Primary metadata checked 2026-09-28:

- Vast VM docs still identify the default Ubuntu 22.04/CUDA/Docker template.
- Astral `python-build-standalone` release `20260924` publishes
  `cpython-3.12.14+20260924-x86_64-unknown-linux-gnu-install_only.tar.gz`,
  66,890,910 bytes, SHA-256
  `5eae8cf79dd47fc2496a4fccc892936be831ce7a84d984b2299dfb1cdb592682`.
  [Archive format](https://gregoryszorc.com/docs/python-build-standalone/main/distributions.html)
  describes a relocatable installation and internal Python links. Project
  licensing does not replace the archive's retained bundled license files.
- Approved NumPy 2.3.1 Linux cp312 wheel:
  `numpy-2.3.1-cp312-cp312-manylinux_2_28_x86_64.whl`, 16,632,729 bytes,
  SHA-256 `e7cbf5a5eafd8d230a3ce356d892512185230e4781a361229bd902ff403bc660`.
- Approved RFC8785 0.1.4 wheel: `rfc8785-0.1.4-py3-none-any.whl`, 9,240 bytes,
  SHA-256 `520d690b448ecf0703691c76e1a34a24ddcd4fc5bc41d589cb7c58ec651bcd48`.

The packet sealer only accepts operator-supplied matching bytes, exact catalogue
and clean implementation identity; it has no download/provider/allocation API.
It copies and verifies assets, retains their licenses, and seals immutable
metadata. Runtime extraction refuses existing destinations, aliases, traversal,
duplicates, special files, outward links and expansion bombs. Only internal
relative links in the digest-pinned trusted Python distribution are allowed;
team archives retain their existing stricter no-links rule. Extract Python first,
then the exact wheels into its private site-packages, verify wheel tags/metadata
and member collisions, then run that actual interpreter with `-I` to verify
Linux/x86_64/cp312/exact versions and actual host/output module imports from the
sealed provider source. No host global site-packages or credentials are exposed.

Red-first tests cover the missing module, exact byte/catalogue binding, unsafe
paths/members/links, missing or altered assets, consistently resealed changes,
binary wheel tag mismatch, observed interpreter mismatch and bootstrap isolation.
Design review checked the existing source/wheel extraction contracts and source
bundle dependency closure. The resulting Python receipt is a prerequisite,
not Docker/bubblewrap/GPU/image/rights/spend qualification. Canonical VM paid
admission remains closed until the full host, artifact and collection integration
is complete. No new provider or Plan13b decision is introduced.

### Offline provisioning implementation checkpoint

`native_g1_team_vm_bootstrap.py` now seals and reopens the exact catalogue,
copies supplied assets without touching their originals, and provisions Python
plus both wheels without a resolver, network or provider API. The bootstrap CLI
imports using only stdlib with `-I -S`. Small-fixture tests are red before the
module exists; a second red exposed the manifest retaining the mutable catalogue
object, repaired by taking an independent canonical copy. Boolean/number types,
duplicate JSON keys, consistently resealed catalogue changes, altered/missing
bytes, source aliases/foreign imports, outward/cyclic links and wheel ABI/path
changes are refused. Provider source manifest identity and every declared byte
are rechecked before any extraction or import; an unlisted Python/extension
source is rejected. The actual isolated interpreter must observe cp312/Linux/
x86_64/exact dependency versions and admitted source origins.

**32 focused tests passed in 31.54s**: 25 bootstrap tests, one extracted selected
bundle CLI/import case, six provider runtime import-closure cases. Changed-file
Ruff and diff checks pass. No paid execution predicate was relaxed.

All three real upstream assets were downloaded once and SHA/size verified:
83,532,879 bytes total. The Python archive has 4,534 members, 222,176,662 expanded
bytes, 1,049 internal symbolic links and no hardlinks, with bundled license files
retained. Its real NumPy/RFC wheel tags match the pinned catalogue. Mac extraction
encountered case-sensitive terminfo names on the case-insensitive filesystem;
that partial scratch is preserved, not deleted or used as qualified runtime.
Actual extraction and cold import must be rehearsed on Linux before the VM
bootstrap is integrated or admitted. Neither download hashes nor fixture probes
establish VM, Docker, GPU inference, scene scoring or public rights.

## Reviewed repair: minimal host cold-import closure

ADP-050 Day 28, owner-authorized shared G1 configuration extension. The actual
Linux rehearsal at `host-python-cold-20260928T0640Z` extracted all three exact
assets, but its fresh interpreter failed importing Torch through catalogue →
preflight → SONIC bridge. No simulator, policy query or provider allocation ran.
An import-graph inspection also found the packet verifier loading the allocator
merely for its image constant, and CAD receipt metadata loading Pillow merely
for rendering helpers. Those edges would discover the next missing dependency
on an otherwise minimal host. Do not install the simulator stack on that host.

Smallest repair: defer Torch and Pillow imports to the existing tensor/image
operations, and use the identical pinned image constant from the lightweight
launcher in the scene bundle verifier. Preserve all actual tensor validation,
image decoding, source hashes, scene verification and execution behavior.
The production cold probe must reject simulator/model/render/provider imports
even if a developer interpreter happens to have them installed.

Test first: import the actual host/output and catalogue in a fresh interpreter
with those imports explicitly denied; require pure metadata to remain usable.
Call the actual tensor conversion under the same denial and require a failure,
not a substitute tensor. Re-run existing SONIC tensor and CAD image tests to
protect unchanged numerical/image behavior, sealed provider closure and the
policy lifecycle rehearsal. Replay the complete production probe on Linux with
the retained actual Python and wheels, a new committed source snapshot, and a
new receipt. Preserve the failed receipt and all originals; no new downloads,
GPU spend, hand data deletions, or paid admission relaxation.

Design review: inspected every top-level dependency path from both host modules;
Torch, Pillow and the allocator constant are the relevant eager edges. Runtime
imports remain mandatory at the operations that need them. Accepted for a
focused red-first repair; cold-import success alone does not qualify a VM,
GPU execution, delivery mode, episode, rights or billing.

Execute-path lookahead additionally found `preflight_g1_vm_host` importing the
allocator to read its driver constant, pulling YAML even after cold import was
fixed. A fresh-process hardware-preflight test with fake Docker/driver responses
reproduced that failure before any VM spend. Move the identical `580.65.06` floor
to the lightweight launcher and re-export the existing allocator API from it.
The fake hardware test proves only that execute-path dependency boundary; actual
driver/runtime/image observations remain mandatory on an admitted VM.

Verification checkpoint: all 106 existing host/bootstrap/SONIC/image/preflight/
provider-closure/lifecycle cases pass; all four fresh-process boundary cases
pass after supplying the missing packet digest in the new preflight fixture.
The combined run's sole failure was that incomplete fixture, not another missing
runtime dependency; the corrected four-case run passes in 1.10s. The shared
launcher has 27 passing cases. A separate selected-bundle/host/closure/lifecycle
run passed all 69 cases before the additional driver-constant repair. Changed
file Ruff and diff checks pass. These proofs still use fake hardware; a retained
real Linux cold replay of the committed snapshot is required next.

### Reviewed real-replay finding: isolated Python ignores bytecode environment

The retained-runtime replay refused the changed size of
`encodings/__pycache__/__init__.cpython-312.pyc` before executing a new probe.
The first rehearsal used `-I` plus `PYTHONDONTWRITEBYTECODE=1`; isolated Python
ignores environment options and regenerated relocated archive bytecode. Preserve
that runtime and both failed scratch roots. Reuse the exact retained asset
archives, but extract into a fresh destination because that runtime is changed.
Do not waive integrity checks or treat cache identity as qualified unchanged.

Smallest encoded fix: add explicit `-B` to the actual provisioning probe, assert
`sys.dont_write_bytecode` in the fresh production probe, and reverify held source
files after it runs. Fresh-process tests and the real Linux rehearsal must use
the same explicit flag. Red-first assertion pins the production command; existing
bootstrap source/asset mutation tests protect the recheck. This repairs observed
source/runtime mutation without changing model bytes, dependencies, paid gates,
or authorizing deletion. Accepted after inspecting the retained exact archive
member and actual size, not guessing a new package gap.

Explicit no-bytecode command assertion failed before the fix; the final 32
bootstrap/fresh-process cases pass in 1.81s, including absent `-B` refusal,
source mutation and foreign bytecode created during the subprocess. Source
identity is reopened after the subprocess before returning provisioning proof.
Changed-file Ruff/diff pass. The next real rehearsal must exercise the production
sealer and materializer, not substitute an environment-variable-only probe.

## Reviewed next implementation: relocated archive and bounded GPU namespace

ADP-050 Day 28 / ADP-060 and owner-authorized shared G1 configurator extension.
Observed blocker: the archive's approval binds an operator-local staging path,
while its bubblewrap command exposes no GPU character devices. A VM policy must
use the same approved archive bytes at its VM-local path and the selected GPU.
Canonical OCI/archive allocation remains refused until complete VM integration.

Keep the approved packet/binding/digest unchanged. Add a trusted worker-side
archive-location override, accepted only for archive mode and still subject to
the approved SHA/regular-file/archive validation. VM host selects only its fixed
sealed input path; no URI fetching or arbitrary volume mount. Reject overrides
on endpoint/container modes before child creation.

Observe exactly one full GPU at index0, UUID and existing driver floor, plus
root-owned world-readable/writable character nodes: nvidia0(195:0),
nvidiactl(195:255), nvidia-uvm(the observed /proc/devices major:0). Reject aliases,
ordinary files, missing/replaced nodes, additional GPUs, mismatched UUID/profile/
packet/digest and private device permissions. Recheck observations immediately
before launch. Keep /dev otherwise private; bind ONLY those three nodes, with
all namespaces unshared, no capabilities, UID/GID65534, read-only archive and
system runtime mounts, private tmpfs and a cleared environment. No host root,
scene, credentials, Docker socket, other GPU nodes or network exposure.

Before the untrusted entrypoint, run a fixed trusted stdlib-only CUDA driver
probe IN THAT SAME bubblewrap namespace. Use ctypes/libcuda driver ABI, verify
one visible device and UUID, allocate/set/copy back16 bytes, then explicitly free
memory/destroy its context. No Torch installation or model query on the host.
The probe proves only device access/basic CUDA memory operations, not a model
inference, task result, rendering, VM/image rights or scientific qualification.
Retain the exact probe hash, binding, observed result and private stderr; require
its digest in archive child teardown and reopen it in VM output verification.
Host preflight's exposure claim stays false until this actual child probe passes.

Test first: strict device/UUID/driver/binding mutations, unchanged observation
recheck, exact narrow device mounts/environment, CUDA probe failure/timeout/extra
GPU/UUID/memory/cleanup mismatch, archive path relocation without approval edits,
mode override refusal, retained teardown/output cross-binding and altered proof
rejection. Existing artifact/session/VM/relay/closure/lifecycle tests must pass.
Run a real no-GPU Linux bubblewrap synthetic JSONL smoke to exercise namespace
setup, UID, read-only filesystem and cleanup; it cannot qualify CUDA. Actual GPU
probe and learned-policy episodes remain mandatory on the admitted VM.

Design review: checked approval's immutable binding, current runtime-session
branches, bubblewrap source --clearenv/--dev-bind semantics and NVIDIA CUDA
device/context/memory Driver API references (archive12.8.2/current13.4). Use
stable versioned v2 symbols and refuse unsupported devices/APIs, no CPU fallback
for a requested GPU. This is the bounded missing execute path, not a new
provider or Plan13b. Accepted for focused test-first implementation.

Implementation checkpoint: 86 focused archive/device/session/VM/output/cold-import
cases pass in 17.08s, including namespace command widening, fake CUDA failure
and cleanup paths, relocated wrong-SHA refusal, and tampered saved GPU proof.
Changed-file Ruff and diff checks pass. These tests simulate GPU observations;
they do not qualify actual CUDA access. The next immutable-source rehearsal runs
the production archive launcher with real Linux bubblewrap and a synthetic JSONL
policy, with no GPU, site input, task score or provider mutation.

### Reviewed real namespace finding: private ancestor traversal

Actual Linux bubblewrap0.9.0 at immutable72337b7e3 refused the archive source
with Permission denied after dropping UID. The retained child exited1 and its
teardown is terminal; source files reverified. Keep the private0700 staging and
evidence directories. Do not chmod private ancestors or expose the host root.
The installed binary supports --ro-bind-fd. Open only the extracted artifact
directory with DIRECTORY/NOFOLLOW before launch, verify its inode/device against
the path, bind that descriptor read-only to /work, and inherit only that
descriptor for the trusted CUDA probe and bubblewrap setup. Close the parent
descriptor on every error/success after Popen; bubblewrap consumes it before
the untrusted policy executes. No raw directory FD may survive into the policy.

Test first: mismatched/non-directory/closed descriptors refuse; command uses
only the approved directory descriptor; Popen and probe inherit exactly it;
client/probe failure closes it and retains existing process cleanup. The real
CPU policy also asserts no extra inherited descriptor. Repeat the same Linux
fixture against a fresh immutable successor and preserve the failed root.
Reviewed against the installed binary's help and bubblewrap's setup/descriptor
cleanup implementation. This changes archive setup, not any approval or paid
admission predicate; actual CUDA and full VM dispatch still require proof.

The449d3c4d5 real replay also refused /proc/self/fd/3: this installed
bubblewrap resolves the descriptor to its canonical path before mounting.
`namei -l` identifies the two private0750 blueprint-owned ancestors outside
our root-owned scratch. A trusted namespace observation confirms its map is
UID65534->host0/GID65534->host0, so it loses initial-namespace DAC override
across those foreign ancestors. Descriptor binding does not bypass traversal.
Do not describe449 as a successful namespace repair.

Encoded correction: before launch, check canonical ancestor execute permission
using exactly the caller UID/GID that bubblewrap maps; reject foreign private
ancestors with a typed namespace-path refusal. Check artifact read+execute too.
Keep the descriptor for source identity and closure, but require this path
predicate separately. The VM canonical staging must use private root-owned
ancestors; the control-plane replay uses its own new private root-owned staging
under /root to match that contract. Copy only this small source/fixture package;
do not move/chmod existing evidence or change a paid gate. Tests pin owner/group/
world traversal and absent execute/read permissions. Repeat against a fresh
immutable successor, retaining both prior failed roots.

Real Linux checkpoint: immutable75b463aa4be966178ddf92edd7b7aee6e40b427a
passes the production archive launcher under the mapped-root staging contract.
The fixture asserts UID/GID65534, CapEff0, read-only/work+/usr mounts, writable
private/tmp, absent host secret/evidence/Docker/GPU paths and no extra inherited
descriptor. Real JSONL reset+one synthetic observation return one valid G1
semantic-v3 action. Teardown exits-15 after the intended SIGTERM, with no close
error. All1956 held source files reverify and no bytecode is created. The final
101 focused cases pass in16.29s; changed-file Ruff/diff pass.

Retained proof:
`/root/blueprint-g1-rehearsals/archive-sandbox-mapped-root-20260928c/cpu_rehearsal.receipt.json`,
SHA25684cfb37e09c6eeca5abc4acd29207cebf1b820bd83e133ad4210daecc94a0e44;
source archive SHA25677b913395c5819e20d6df510e3c42fbd2c745d65365946cca63acf58216ab845;
synthetic archive SHA256cb10b82b59789cee04018af9ba607be016c500ca6a77b1a1e44ff0adcc1de2f9;
conformance digest05d74b2f62ed132c6a6e00ed2f6df06619dcf28a8f6d11ae7e35e1c22fa5a2ad;
teardown digestabbc58659743fb117fc462a82d7c07fe6150245c93af655e1fef032a191291a6.
Measured `du -sx --block-size=1`: the successful root is105893888 bytes;
the first and second refused roots are105889792 and105893888 bytes. Total
317677568 bytes includes source archives and extracted source. These are own G1
scratch, with no existing evidence moved or deleted. Both refused roots remain
retained. CPU only;
zero site queries, no task score, no GPU/device-memory proof, no VM qualification,
no provider mutation, no rights/spend grant. Production VM bootstrap must check
the installed bubblewrap's actual --ro-bind-fd/--clearenv support, canonical
root-owned ancestor accessibility, and required UVM nodes before untrusted
execution. The canonical bundle/VM allocator refusal stays in place until full
integration, actual CUDA, and all launch/output gates are complete.

## Reviewed selected-bundle integration: offline host assets and archive bytes

ADP-050 Day28 / ADP-060 owner extension. Observed blocker: paired-mode bundles
still contain only the Isaac guest entrypoint, no proved host CPython package
and no fixed-path approved archive. The allocator refuses these modes. Preserve
that refusal until VM allocation/bootstrap/image/runtime/output admission is
fully integrated; this transport change alone cannot authorize a run.

For container/archive bundles require the sealed host-Python package bound to
the same implementation commit, embed its exact manifest and three pinned assets
under provider_runtime/host-python-package, and verify their inventory/bytes on
every bundle reopen. Add a separate VM entrypoint that provisions offline with
system Python in isolated/no-bytecode/stdlib-only mode, then executes the actual
VM host with the proved cp312 runtime and explicit -I/-B. Keep the existing
Isaac guest shell entrypoint for the simulator container. Expose the actual
allocator entrypoint in the manifest, but no ready/spend/qualification claim.

For archives copy the trusted operator-staged file to the fixed transported
inputs/team-policy/policy.tar path only after exact approved SHA, regular-file,
safe archive and existing size/member checks. A trusted source-path override
does not edit the packet or approval. Bind transported artifact path/hash/size
to the approved profile and its actual artifact row on every reopen; reject
artifact arguments in other modes. Endpoints retain their current credential
transport and entrypoint with no host package/artifact included. Existing
endpoint retained evidence remains readable.

Lookahead found the host source verifier rejecting ALL configs/ artifact rows
despite the real bundle always including the two G1 inventories there. Admit
only the existing configs/ root alongside provider_runtime/, still verify every
listed artifact, canonical containment, and every Python source/bytecode/shared
library. Test the real manifest shape before a rented VM. No arbitrary roots.

Test first: absent/foreign-commit/changed host package, wrong staged archive SHA,
unsafe archive, forbidden endpoint/OCI artifact inputs, unchanged approval and
packet, host-vs-guest entrypoint selection, resealed foreign host-package metadata,
config artifact mutation and foreign root refusal. Execute generated shell syntax
and isolated entrypoint imports; preserve required provider import-closure and
policy lifecycle rehearsals. Package proof, cold imports, synthetic actions and
VM metadata are not a scored/GPU run. Actual system prerequisites and allocation
remain separate implementation gates.

Design review: inspected producer, retained/live readers, guest sealed-input
verifier, host materializer and canonical preallocation refusal. Reuse the proved
bootstrap and exact approved archive hash rather than a new runtime/provider.
Accepted for focused test-first implementation. Canonical paid refusal remains.

## Reviewed allocator collection gate: paired host and bootstrap closeout

ADP-050 Day28 / ADP-060 owner extension. The canonical paid lane currently
reopens only the simulator's selected-worker score/media result. That evidence
cannot prove the separately hosted policy, relay, simulator container, or outer
bootstrap completed. Preserve the current paired-mode allocation refusal while
repairing the downstream verifier before connecting VM launch.

For a paired packet require the generated bootstrap terminal receipt: its digest,
exact implementation commit, explicit integer zero exit code, vm-host stage,
host_exited status and development-only/no-provider/no-GPU-qualification flags.
Then invoke the existing full VM output verifier against the same collected root,
packet and scene bindings. It already reopens policy/relay/worker/simulator
receipts, archives' CUDA roundtrip proof, exact observation bytes and both videos.
Return its full verification so later ingest can retain the host evidence.
Endpoint output continues through the existing native score/media verifier.

Collection-to-review lookahead: the existing private projector requires the
worker verification schema and its top-level profile/score/media fields. Preserve
that interface and embed the complete reopened VM host verification under
`isolated_policy_host_verification`. Do not replace the worker schema with a VM
wrapper that the existing review URL cannot ingest. Positive tests must feed the
actual collected verification through the existing private projector for both
paired modes and preserve their selected policy delivery identity.

Test first with the existing real socket/JSONL/scoring/media rehearsal and fake
Docker/Isaac, both OCI and archive modes. Move its actual retained evidence into
the canonical attempt layout and call the real allocator output function.
Absent/resealed foreign bootstrap commit, nonzero/boolean exit, wrong stage,
invented GPU/provider qualification, absent VM host result, corrupted relay and
changed media must refuse. Fixtures prove the collection contract only; they do
not qualify a VM or learned-policy GPU execution. Preserve endpoint coverage.

Design review: this reuses the VM evidence verifier rather than another output
schema or launcher. Package/sandbox/source prerequisites and provider-zero/billing
remain separate gates. Accepted for focused test-first implementation.

## Reviewed host admission gate: installed archive sandbox capabilities

ADP-050 Day28 / ADP-060 owner extension. Current host preflight checks only
`bwrap --version`; that does not prove the flags required by the encoded archive
launcher exist. The real control-host binary's help was inspected on Sept28
08:23 UTC and declares --ro-bind-fd and --clearenv. The VM remains unobserved.
Check installed `bwrap --help` before observing GPU binding or opening the
untrusted archive policy. Require exact option tokens for all isolation/device
arguments used by isolated_artifact_command, including --ro-bind-fd, --clearenv,
--unshare-all, --uid/--gid, --cap-drop, --dev-bind and --die-with-parent.
Retain the exact observed required-feature list in the host preflight and
require it in collected archive evidence. OCI does not require bubblewrap.

Test first: a zero-exit version command with missing feature(s), a nonzero help
command, and substring-only fake support must refuse before GPU/device
observation. Complete exact help tokens pass without installing/changing the
host; the subsequent real same-namespace CUDA probe remains mandatory. Test
resealed removal of the capability receipt in retained archive evidence.

Design review: this is an early prerequisite check for the existing launcher,
not another sandbox or inferred namespace/GPU qualification. Actual VM system
bootstrap must install a pinned capable binary before this gate is admitted;
canonical paired allocation remains refused. Accepted for focused TDD.

## Sept28 implementation evidence: packaged host and collected review path

The selected bundle now embeds the exact same-commit offline CPython/NumPy/RFC
package for OCI/archive modes, the approved SHA-bound archive at its fixed VM
path, and a separate isolated host launcher. The retained reader and guest
verifier reopen the host package and policy artifact bindings. Endpoint
transport remains compatible without host assets. Missing archive entrypoints,
unsafe members and missing actual VM launcher/modules refuse before allocation.
The host bootstrap now admits the two existing configs/ inventory artifacts and
still reopens all listed source bytes. Generated shell execution tests caught and
repaired lexical ../ paths that the real materializer would refuse.

The paid output verifier now requires the bootstrap's zero integer exit and
exact commit, then reopens the full host/relay/policy/worker/simulator closure and
native score/media. It keeps the existing worker verification schema and embeds
the full VM verification under isolated_policy_host_verification, so the SAME
private review projector accepts both paired policy modes. Real socket/JSONL,
scoring and retained frame/video fixtures exercise that entire collection-to-
projection path; Docker, Isaac and archive CUDA remain explicit test doubles.

Installed sandbox preflight now checks exact help option tokens, including
--ro-bind-fd and --clearenv, before device observation. The observed feature
list is mandatory in collected archive evidence. No new binary was installed;
the previously CPU-proved control-host binary's help was inspected read-only.

Focused evidence, in execution order:

- Packaging/bootstrap/host/runtime/import/lifecycle batch: 167 passed in
  353.31s. It includes six provider import-closure and 19 lifecycle cases.
- After the capability gate: 76 passed in 111.49s across feature rejection,
  host/VM collection, cold imports, six provider closure and 19 lifecycle cases.
- Final review-interface repair: 14 passed in 14.73s, including both actual
  retained paired-mode collection/projection paths, ten tampering/refusal cases,
  existing endpoint media verification and private projection/staging.
- Changed-file Ruff and git diff --check pass. Red-first archive entrypoint,
  missing host evidence, missing sandbox features and review-schema failures
  were observed before their encoded repairs.

Read-only host observation: deployed release remains
4a59c6bb668ac53ca9cc3440fcc57b21242d3612; both inspected campaign dispatchers
are inactive/MainPID0. Work-volume free bytes 11,136,524,288; root free bytes
16,146,935,808. Neither meets the selected-policy 32,000,000,000-byte collection
reservation. Service inactivity is not current provider-zero or billing proof.
No paid allocation, deployment, model download, data deletion or runtime/image
qualification occurred in this slice. The paired dispatch admission refusal
remains until exact VM system/image/bootstrap/allocation integration is complete.
Full four built-ins, learned GPU execution for all delivery modes, actual private
URL and final merge/push/deployment/settlement remain required by the full goal.

## Reviewed VM transport slice before canonical allocation

ADP-050 Day28 / ADP-060 owner extension. The canonical Vast probe still assumes
/isaac-sim/python.sh on the host; a real VM needs system Python transport before
the sealed portable host runtime can start. Add an opt-in VM script builder to
the existing probe constructor, limited to selected G1 paired bundles and exact
bundle SHA. Its standalone stdlib runner must observe a root Linux/x86_64 VM
(systemd PID1, no detected container, KVM/QEMU), load only the provider's allowlisted
startup variables without sourcing/evaluating /etc/environment, use bounded curl
downloads, verify exact ZIP/dependency SHA and safe regular-file extraction into
new private root-owned staging, then execute the sealed VM host entrypoint.

Before pulling images or running a policy observe Docker/NVIDIA runtime, the
driver floor and installed archive sandbox features. Pull only the exact pinned
simulator/profile images from hash-bound inputs. Preserve all regular output
bytes, including files above100MB, excluding only the transported archive's
extracted execution-input directory. Retain typed transport diagnostics and a
terminal receipt even on failure; upload only through the existing signed PUT
transport. Full score, closure, output and billing verification remains external.

Official [Vast VM docs](https://github.com/vast-ai/docs/blob/main/linux-virtual-machines.mdx)
require a shebang for On-start scripts and describe environment values written
to /etc/environment. Repair _create_payload so both short and compressed VM
scripts have an outer Bash shebang and remain under the safe inline size ceiling.
Normal container creation retains its existing payload behavior.

Test first: short/compressed startup scripts, malformed VM kind/flags/SHA,
non-evaluated environment values, traversal/links/duplicates/special ZIP members,
changed downloads, private fresh staging, missing/foreign VM identity,
missing Docker/features/driver, approved mode/pinned images, exact dependency
binding, full >100MB evidence retention, transient input exclusion and terminal
failure packaging. Fixtures never assert actual VM/GPU qualification or spend.

Design review: transport stays inside the canonical adapter's probe path, no
alternative allocator, provider or public VM flag. This does not install or
qualify missing VM system components. Keep paired dispatch refused until the
pinned system bootstrap and full canonical offer/create/watchdog path are ready.
Accepted for focused test-first implementation.

Transport review follow-up: preserve the adapter's byte-count and blocked markers,
and write the ZIP at its existing `/workspace/adp_arena_provider_runtime_output.zip`
SSH-recovery location using an exclusive regular-file copy. The execution staging
root remains new/private; an existing recovery target or unsafe workspace refuses
publication. Test the same marker bytes, unchanged ZIP, existing-target refusal,
and execute the embedded runner with isolated system Python, not only `bash -n`.

### VM transport verification checkpoint

The first transport/startup batch passed 48 cases; transport plus existing SSH
recovery passed 60. The expanded batch covered 94 distinct cases, including all
six sealed provider import-closure and 19 policy lifecycle cases: 90 passed,
four newly added test cases failed because an overwrite assertion was inserted
in the wrong test function. That test-only placement was corrected; all 34
transport cases then passed in 1.33s. Runtime source was unchanged between those
two runs. The three shebang cases and exclusive recovery publication case were
observed red before their implementations. Changed-file Ruff and diff checks pass.

The embedded stdlib runner was executed with `-I -B -S --help`; this proves source
startup without Blueprint packages, not a VM, system stack, GPU or learned episode.
All external commands in episode tests remain explicit fixtures. Paired dispatch
remains refused until complete VM system/bootstrap/allocation integration.

Fresh read-only storage observation: work available 11,136,524,288B; root available
16,114,515,968B. Selected-policy collection still requires 32,000,000,000B. Release
identity remains 4a59c6bb668ac53ca9cc3440fcc57b21242d3612; both inspected campaign
services are inactive/MainPID0. This does not establish provider-zero or billing.
No GPU allocation, deployment, system-package installation or data deletion.

## Reviewed image CPU preflight before enabling canonical VM allocation

ADP-050 Day28 / ADP-060 owner extension. Fresh official registry read verifies
Vast KVM manifest28dc36f9 and config2b2425c3, Linux/amd64. Its outer history copies
`root/images/ubuntu.img`, but cannot prove guest Docker, NVIDIA or bwrap tools.
Anonymous exact Isaac manifest pull is observed authorized; the outer KVM image
login is therefore not a substitute for observing the guest. A bounded range
read identifies guest disk5196152832B in compressed layer2633977898B, SHAde25e09c;
that header observation is NOT whole-layer integrity proof.

Smallest next preflight: reuse installed local QEMU in software TCG, download
only that immutable public disk layer, verify whole compressed bytes and exact
size, then stream-extract ONLY the fixed regular ubuntu.img into a fresh stage.
Reject changed hashes, ambiguous/missing/duplicate/unsafe members and size drift;
never run a supervisor/container or an arbitrary archive command. Require local
free bytes for compressed+extracted disk+512MiB overlay+32MiB log+8GB remaining
before download and recheck at extraction/boot. Preserve retained inputs and
failed stages; no deletion or reuse of a modified base.

Boot a qcow2 overlay with <=2GiB RAM/two CPUs, no acceleration beyond TCG, no
network, no host mounts/device exposure, no captures/model data, and a NoCloud
seed containing only a fixed read-only guest diagnostic. Inspect OS/kernel,
installed driver/toolkit/Docker/bwrap packages, Docker runtime map and bwrap
help; write an explicit CPU-only terminal observation to serial then power off.
Do not install packages, upgrade drivers or simulate any score. Bound the child
to15minutes, overlay512MiB/log32MiB and host8GB floor; terminate ONLY this child
on a breached boundary. PID/resource receipt and fixed command must be retained.

Test first: exact layer/disk binding and malformed tar rejection; insufficient
capacity before fetch; exact networkless TCG command and bounded resources;
guest diagnostics cannot mutate packages, call policies or contain private data.
After focused tests and immutable commit/push, run the actual CPU boot and record
what is installed/missing. Guest package identity does not prove actual NVIDIA
hardware, driver loading, CUDA, policy inference or full VM provider admission.
Use the result to implement the complete offline system bootstrap and canonical
offer/create/watchdog path; paired dispatch stays refused until ready.

Design review: source/asset/command boundaries and current17.3GB local capacity
were inspected. Planned retained+maximum scratch fits with8GB remaining. This
uses the existing authorized public Vast image and local installed QEMU; it is
not Plan13b, a new paid CPU provider, or an alternate GPU allocator. Accepted
for a focused test-first CPU preflight, with no changes to spend/storage gates.

Implementation checkpoint:11 focused tests pass in0.41s for exact compressed
layer/guest bytes, unsafe/duplicate tar members, capacity refusal, bounded TCG
command, fixed non-policy diagnostics and absent/duplicate terminal rejection.
Missing implementation was observed red before build. Changed-file Ruff/diff
checks pass. The CPU rehearsal is still unexecuted at this checkpoint; these
fixtures do not establish guest packages, CUDA, VM-provider or episode support.
Guest boot follows only after clean immutable commit/push and a fresh local
capacity check. Primary references: official Vast VM docs, cloud-init NoCloud
CIDATA local drive, and QEMU invocation documentation.

### Reviewed observed CPU-boot transport repair

Actual immutablef9d6b7291 CPU run downloaded and verified the complete layer,
extracted the exact5,196,152,832-byte disk and booted Ubuntu22.04.5 with
5.15.0-1067-kvm/systemd. The current boot log enumerates only virtio vda; there
is no sr0/AHCI/CD-ROM or cloud-init/terminal diagnostic observation. The process
is still live under its15minute watchdog; absence of a receipt is not terminal.

Prepare the seed as a read-only virtio block drive, matching the actually
observed guest storage driver and NoCloud's labeled-local-drive contract. Test
the exact seed drive before changing it. Add a verified retained-image replay:
new stage/overlay/log only, reopen the original compressed layer SHA/size and
manifest binding, validate all tar rows, stream-hash the guest member and require
the retained base's exact same hash/size before boot. Reject changed assets or
ambiguity; never redownload or copy the unchanged5GB base. No deletion or original
base writes. Reserve only bounded overlay/log when verified inputs are reused.

Tests first: retained layer/base changes and exact unchanged stream binding,
same fixed regular-file tar validation, new virtio seed with no network/device
exposure. Preserve the live first process until its authoritative terminal;
the repaired replay must be a separate fresh stage after terminal, not a restart
on an observation timeout. This is a boot-transport finding, not yet proof of
missing packages. Accepted for focused repair and actual retained-input replay.

Repair checkpoint: exact virtio seed and all three retained-input binding cases
were red before implementation. All14 CPU-preflight cases now pass in0.41s,
with changed-file Ruff/diff clean. The source's replay path streams and reopens
both original assets; it creates only a new bounded overlay/seed/log and never
downloads or copies those assets. At09:35UTC the original f9d6b7291 controller
PID3035/QEMU10227 were confirmed live (10m28s guest elapsed), with no terminal
receipt. Keep this authoritative process under its original deadline; tests do
not convert its missing diagnostic into success or permission to replace it.

## Observed CPU result and reviewed offline system package preparation

The original f9 run became terminal under its unchanged watchdog. Fresh de4691
retained-image replay actually succeeded: guest diagnostic received, exit0,
child absent and original base unchanged. Receipt
`/private/tmp/g1-vm-system-cpu-de4691-20260928b/g1_vm_system_cpu_preflight.v1.json`,
digest `13dd4c648dd0d28f8d3d7bebb9228540728e85330d63c2b68162d337ccb5784a`.
Actual guest receipt digest `438b2d183190fcf720f84bc2ff03d36243d04ced23b8876f1a68e0b2b8d5488d`.
Ubuntu22.04.5/kernel5.15.0-1067-kvm has Docker24.0.7 with only runc runtimes,
NVIDIA535.183.01 packages, no NVIDIA container toolkit and no bubblewrap.
It has matching kernel headers, build-essential, GCC11/GCC12 and DKMS2.8.7.
This proves guest prerequisites are missing; it is not CUDA or episode proof.

ADP-050 Day28 / ADP-060 owner extension. Next complete system prerequisite
preparation binds the observed guest and exact official amd64/all Debian package
assets. Keep driver580.65.06 floor, complete graphics/compute/firmware/kernel
closure, upgrade DKMS because the official580 package requires>=3.1.8, and pin
all four NVIDIA container-toolkit packages together. Jammy security bubblewrap
is a candidate: require actual17-flag help and namespace replay later; its
version alone cannot qualify isolation. Preserve installed Docker/kernel/headers
and their observed identity. No paid launch or dispatch enablement follows a
package plan, seal, APT simulation, or CPU namespace proof.

Test first: refuse changed outer/guest observation digests, wrong base/image,
nonterminal/nonzero/changed-scope receipts, wrong kernel/build prerequisite
identity, absent/foreign/symlink/changed package assets, ambiguous inventories,
source commit drift and modified package manifests. Generate only a fixed
offline APT simulation command (no-download/no remote sources/no recommends),
with each concrete local package path passed as an argv item. No shell,
online upgrade, autoremove, kernel swap, provider call, policy or rights grant.
Reopen every package hash before preparation/replay. Keep package license
review separate from bytes and APT closure; new exact components cannot acquire
owner approval from a manifest. Collect immutable assets and embedded notices
before the concrete review packet, then replay APT against the exact guest.

Design review: the guest observation changed the next action from speculative
VM allocation to offline system closure. Official NVIDIA Jammy metadata confirms
the DKMS dependency gap; installed kernel headers/build tools avoid an unrelated
kernel migration. The existing portable Python/bundle transport does not install
system prerequisites. This preparation is accepted before implementation;
actual installation, reboot/driver loading and canonical paid routing remain
separate required proof. Current work free11.14GB/root16.10GB still do not admit
the32GB collection gate. Preserve all data, original disk and failed overlays.

Preparation checkpoint: missing module was observed red before implementation.
16 package preparation cases plus14 retained-image CPU cases pass (30 total),
with changed-file Ruff/diff clean. The pinned25-package candidate closure is
389,073,866 compressed bytes: exact580.65.06 graphics/compute/kernel/firmware,
DKMS1:3.2.1-1ubuntu2, all four toolkit1.19.0-1 packages, and Jammy security
bubblewrap0.6.1-1ubuntu0.3. The preparation reopens both observation digests,
base/kernel/build identity and every concrete package size/hash. APT simulation
disables remote sources and uses a verified empty lists directory as well as
no-download/no-recommends. Wrong source commit, modified approval flag or asset
cannot produce the retained command. The preparation explicitly retains
owner_review_required=true and runtime_installation_performed=false.
These fixtures do not establish APT dependency closure, actual package install,
sandbox features, driver loading, CUDA or any paired-policy episode.

### Reviewed exact-guest APT simulation replay

Extend the existing bounded local CPU rehearsal with an optional, fully
verified system-package preparation. Reopen every asset before making the
NoCloud seed; reserve seed directory+ISO bytes in addition to existing bounds.
Copy only the fixed preparation files and same-commit stdlib verifier, bind
source SHA and manifest digest in the CPU receipt. Inside the networkless
guest mount its read-only virtio seed as ro/nodev/nosuid/noexec, verify source
SHA, both observation receipts, package manifest and every .deb again, then
run ONLY the helper's fixed offline APT simulation. Retain exact exit/stdout/
stderr with the existing guest diagnostic. No package maintainer script,
driver configuration, reboot to a modified image, policy or GPU is run.

Tests before build: baseline diagnostic stays unchanged; package replay has
only read-only seed mount, same source/digest binding and fixed APT simulation;
unsafe commit/hash inputs refuse generation. A CPU observation containing an
APT failure is still an observed CPU result, not an installation success.
Record the actual resolver output to decide the complete install/review packet,
and keep dispatch refused. Accepted for same-guest data-only resolver replay.

Replay checkpoint: both new guest-source binding cases were observed red before
implementation. All32 package and CPU cases pass in0.73s; changed-file Ruff/diff
pass. Actual public download-only acquisition verified all25 exact files and
389,073,866 bytes in `/private/tmp/g1-vm-system-package-assets-b82d2e12e-20260928a`.
No package code or maintainer script was executed. The optional replay reserves
both seed-directory and ISO copies, verifies the same-commit verifier source
and complete package preparation on host and guest, and records exact APT
simulation output under the existing CPU-only observation. Actual APT replay
is still unexecuted at this checkpoint; execute only after clean commit/push,
fresh capacity and preparing all files against that immutable commit.

Actual immutable0d63 CPU replay is terminal: guest observed/exit0/base unchanged,
receipt digest e1c262da5f865454182969df08702bd87347c07e6be91470af3d043056082ab6.
Its package probe retained ValueError only and did not reach APT. The exact
retained ISO was then mounted read-only on Mac, its verifier SHA matched
c02651966799d000fafce85972f1a3a012565c491d6bfe5aa7861eade632416a,
and complete package verification succeeded with manifest ddeda2ea0f50f214ba751b12ddcb9e26eae0d8f5749530783a1d249f4f6c781d.
Mount detached; no data deleted. This rules out changed ISO bytes under that
reader, not a guest filesystem difference. APT closure remains unobserved.

Reviewed focused repair: retain stage and allowlisted g1_vm_system_* error code,
plus bounded seed inventory, to diagnose the actual guest refusal. Do not
substitute an exception type for its cause. Reuse read-only cached public .deb
bytes by verified same-filesystem hardlinks in package and seed preparation;
never link writable files, change originals, follow symlinks, delete inputs or
skip any hash. Retained manifests remain separate new files/source commit.
All three terminal CPU boots used<=25,034,752B overlays and small logs. For
read-only diagnostic/simulation only, reserve64MiB overlay and8MiB serial log;
keep2GiB/twoCPU/15m/no-network/8GB floor and enforce the actual child bounds.
This remains the full25-file resolver probe; it is not a shortened episode or
an installation budget. It removes duplicate seed-directory/package bytes
while retaining the entire ISO/evidence. Current local free8.487GB admits the
389MB ISO plus these measured conservative diagnostic limits, with the same
8GB floor. Reject cross-filesystem or writable asset linking before output.

Tests first: original writable fixtures default to copy; optional immutable
link rejects writable/symlink bytes and uses exact same verified inode without
changing bytes or modes. Generated guest source retains typed cause and stage;
actual replay must follow clean commit/push and fresh capacity. Accepted before
building this focused failure-diagnostic/storage repair. No package install,
rights approval, provider mutation, data deletion or dispatch enablement.

Expose the preparation through this module's local-artifact CLI (exclusive
download-only or prepare mode), rather than requiring a hand Python snippet.
Prepare mode requires observation/assets/output/implementation commit and
reopens clean exact checkout HEAD before calling the same verifier/materializer;
explicit immutable linking is the only reuse option. Test that CLI delegates to
the actual preparation with concrete paths and cannot confuse it with download.
This is not a provider launcher or approval grant; accepted with this repair.

Repair checkpoint: typed diagnostic, immutable-link and CLI cases were observed
red before implementation. All36 focused CPU/package cases pass, Ruff/diff
pass. Immutable linking leaves original bytes/modes intact, and the simulation
still reopens every25-package hash on host and guest. No install/paid launch or
original deletion. New diagnostic replay must use a fresh stage and manifest
bound to the new pushed commit; do not overwrite the terminal0d63 evidence.

## Reviewed observed ISO primary-name transport repair

Actual25bc CPU probe is terminal and names the native cause:
g1_vm_system_package_inventory_invalid at package_verification. Its bounded
inventory shows the guest ISO primary reader truncates long filenames and
removes version dots (for example dkms_321-1ubuntu2_all.deb), including the
long manifest name. Mac's extended-name reader saw the original names; that
reader was insufficient to qualify the guest transport. No APT was run.

Use v2 package preparation with short canonical transport names: packages.json,
observed.json and 16-hex SHA prefix + .deb for each exact asset. Preserve full
publisher filenames/URLs/SHA/size in the manifest and original download cache.
Reject short-name collisions before any output. Every staged name must be
ASCII, <=31chars and one suffix dot, invariant under the observed ISO primary
reader. Keep full byte/hash/identity validation and no remote sources. The
v1 prepared roots and their same-commit verifiers remain retained evidence;
do not rewrite their schemas or retroactively call them compatible.

Test first: complete fixed-name domain and collision rejection; retained
verification and APT argv use the same canonical aliases; no weakened hashes,
package list, driver floor or runtime/rights assertions. Update existing
malformed/cached asset tests to target the canonical paths. Reuse the unchanged
25 public assets, create a fresh pushed-v2 manifest/ISO and repeat the SAME
bounded CPU simulation. No download, install, GPU, original deletion or grant.
Accepted before implementation based on the actual guest inventory.

v2 repair checkpoint: the fixed ISO naming case was observed red before build.
All38 package/CPU cases pass; collision refusal is covered before output, and
all existing malformed asset/observation, readonly-link/mode and immutable CLI
checks remain covered. Publisher filenames and content pins are unchanged.
Short transport names do not establish guest resolver or installation success;
fresh same-guest APT simulation follows clean commit/push.

## Reviewed retention after an observed parent capacity refusal

The exact9594 guest emitted a digest-verified native observation with all25
package identities and offline APT exit0. The parent subsequently stopped the
child during shutdown when Mac free space fell below8GB. Its parent receipt
correctly remained blocked, but omitted the native observation because it only
parsed the serial log following a zero child exit. The retained log recovery
proved this boundary without rewriting the parent receipt. The built-in paid
campaign is now independently live; this repair affects only CPU diagnostic
retention and must not change or restart that campaign.

After stopping the owned child on a parent refusal, retain a single valid native
CPU observation if present. Verify its declared digest, CPU-only scope and,
when system packages were supplied, exact source/manifest/verifier bindings
and no-install declaration. Preserve the original blocked status and blocker;
do not promote partial CPU observation to full rehearsal, installation, GPU,
namespace or policy proof. Missing, duplicate, altered or foreign-bound native
receipts get a separate typed retention diagnostic and never mask the original
parent refusal. Do not lower any resource bound or rerun already observed APT.

Test first: simulate actual capacity refusal after a valid guest terminal log;
ensure the original refusal survives with observed APT exit0. Reject wrong
package binding, altered digest, duplicate marker, absent marker and installation
claim. Review accepted before implementation: this encodes the demonstrated
retention gap, leaves launch/rights/spend behavior unchanged and avoids another
asset-heavy replay solely to recover already finished dependency evidence.

The focused replay exposed four older fixture tests relying on real Mac free
space. Their tiny synthetic assets must use explicit hermetic capacity rather
than require8GB on the test machine. Keep the CPU insufficient-capacity case
and add the same pre-output refusal case for package preparation; production
capacity floors are unchanged. This test isolation repair is accepted after
the actual four failures and before changing their fixture environment.

Retention checkpoint: all7 new cases were observed red before implementation,
including the actual parent resource-refusal path through its owned child.
After the retention repair and explicit synthetic-fixture capacity isolation,
all46 CPU/package tests pass (1.72s); changed-file Ruff and diff checks pass.
The production8GB floor is unchanged, malformed/foreign/installation receipts
cannot be retained as accepted observations, and parent blocked status remains.
No additional VM boot, install, GPU change, deletion or qualification claim.
