# Public MolmoAct2 bimanual YAM runtime

The fixed `molmoact2_bimanual_yam` registration binds
`allenai/MolmoAct2-BimanualYAM` revision
`8dcbed66f2380e4393189c303ea72488eb9e63c2` to the prepared
`i2rt_yam_bimanual` body. It is an additional Blueprint public-policy baseline
using the `customer_controller` route. The original 74 published robot–route
pairs and all existing candidate IDs remain intact. This does not imply that
Dream Labs deploys MolmoAct2 or supplies its rights.

The fixed operator installer separately verifies retained native body/import
receipts, scoped public-model rights and protected registry compare-and-swap.
Installing the registration launches no compute. The ordinary harness derives
the task scene, checks rights and budgets, seals a per-task specification, and
selects or retains an individual reason for this candidate. A valid registration
can use the canonical allocator; deployment alone does not activate or run it.

## Exact interface

`molmoact2_yam_joint14.v1` uses the six named joints of each YAM arm and two
independent `LINEAR_4310` gripper channels. Model outputs are absolute joint
radians with normalized opening; the adapter converts opening to the existing
native closure convention. The model predicts 30 rows at 30 Hz. The initial
configuration executes 25 rows before requesting again, holding each row for
two native 60 Hz ticks. Returned finite gripper openings are saturated to
`[0,1]` with both raw and applied values retained.

The camera order is overhead, left wrist, right wrist. Each input is RGB uint8,
640×360. The initial rig preserves the prepared source configuration's mount
links and poses, with horizontal FOV 69.4° overhead and 87° at each wrist.
These are explicitly recorded simulation assumptions, not publisher pose
calibration or a measured site camera rig. The overhead D405 CAD versus
simulated D435i optics discrepancy remains recorded. Different supported
camera bindings can be explicitly sealed; no hidden image resizing occurs.

## Runtime inputs and cost accounting

The exact release inventory is pinned by raw SHA256. All required checkpoint
files, including six Python modules, tokenizer, normalization, configuration,
shard index and five weight shards, are downloaded at the immutable model
revision. Each transfer streams to a partial file, enforces its byte bound and
SHA256, and atomically publishes only verified bytes. Aggregate transfer bytes
and the canonical worker's absolute deadline bound the whole preparation.
There are no download retries. Full checkpoint closure is checked before and
after hashing, before `trust_remote_code` can execute any checkpoint module.

The inference server uses its own exact CPython 3.12.12 distribution and 50
pinned wheels. Installation uses `--no-index --no-deps --require-hashes`.
The dependency inventory accounts for exactly 4,045,566,007 external bytes.
HF and Transformers run offline with local files only; dynamic module and hub
caches are outside the immutable checkpoint directory. Public-model execution
does not forward an HF token. The Isaac client uses its existing NumPy and
standard-library loopback HTTP client; it imports no inference framework.

The launch quote includes the actual sealed bundle, any external native runtime
packet, all required checkpoint files, the exact dependency bytes, a 64 GiB
cold-image allowance, and up to 8 GB of retained output. Existing fleet budget,
offer admission, zero-retry allocation, independent watchdog and server teardown
remain in force. The requested GPU floor is 48 GB including simulator headroom.
It is conservative configuration, not a measured peak. A capacity or budget
refusal remains this candidate's result.

## Evidence and limits

CPU tests cover the real registration → catalog → plan → sealed archive →
canonical transport-preflight path; exact source/dependency/input identities;
bounded download, deadline and startup teardown behavior; and the actual
native episode loop with fake physics, real adapter, named observations and
independent grippers. The native-loop test executes 52 ticks, requests twice,
and retains request/action evidence. Shared tests preserve the existing DROID,
HTTPS, Dream and PiPER paths.

The dependency audit additionally imported the actual pinned checkpoint modules
and exercised the real tokenizer/processor/normalization entirely offline with
CPU wheels. No complete weights were downloaded for these tests. GPU model
loading, native Isaac execution, measured peak VRAM and task success remain
unverified. Adapter evidence retains `native_execution_validated: false`; a
sealed execution binding identifies the caller without fabricating validation.

The [adapter source/license audit](../molmoact2-yam-public-integration.md) records
the publisher's official Apache 2.0 model statement and research/educational
intended-use note. The HF card's missing license field remains explicit.
