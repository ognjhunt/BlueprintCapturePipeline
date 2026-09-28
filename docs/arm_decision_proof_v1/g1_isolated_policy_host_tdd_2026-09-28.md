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
This root is own G1 scratch, approximately52MiB source/archive, with no existing
evidence moved or deleted. Both refused roots remain retained. CPU only;
zero site queries, no task score, no GPU/device-memory proof, no VM qualification,
no provider mutation, no rights/spend grant. Production VM bootstrap must check
the installed bubblewrap's actual --ro-bind-fd/--clearenv support, canonical
root-owned ancestor accessibility, and required UVM nodes before untrusted
execution. The canonical bundle/VM allocator refusal stays in place until full
integration, actual CUDA, and all launch/output gates are complete.
